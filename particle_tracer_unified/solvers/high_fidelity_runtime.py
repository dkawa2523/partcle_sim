from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..core.boundary_service import (
    BoundaryService,
    build_boundary_service,
    contact_frame_on_boundary_edge_2d,
    points_inside_geometry_2d,
    runtime_bounds,
)
from ..core.catalogs import resolve_step_physics
from ..core.datamodel import PreparedRuntime, ProcessStepRow
from ..core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from ..core.geometry3d import TriangleSurface3D, build_triangle_surface
from ..core.integrator_registry import IntegratorSpec, get_integrator_spec
from .compiled_field_backend import (
    CompiledRuntimeBackendLike,
    compiled_gas_property_report,
    compile_runtime_backend as _compile_runtime_arrays,
    sample_compiled_acceleration_vectors as _sample_acceleration_vectors_at,
    sample_compiled_valid_mask_statuses as _sample_valid_mask_statuses,
)
from .charge_model import (
    ChargeModelConfig,
    apply_charge_model_update,
    charge_model_report,
    finalize_charge_model_diagnostics,
    merge_charge_model_diagnostics,
    validate_charge_model_support,
)
from .plasma_background import plasma_background_report
from .high_fidelity_collision import _advance_colliding_particle, _classify_trial_collisions, _step_segment_name
from .high_fidelity_freeflight import (
    _advance_trial_particles,
    _stage_points_from_trial,
    resolve_valid_mask_prefix,
)
from .integrator_common import (
    DRAG_MODEL_STOKES,
    effective_tau_from_slip_speed,
)
from .forces import (
    ForceRuntimeParameters,
    force_catalog_summary,
    force_runtime_parameters_summary,
)
from .diagnostics import (
    increment_named_count,
    initial_collision_diagnostics,
    invalid_stop_reason_code,
    invalid_stop_reason_name,
)
from .field_runtime import record_field_sampling_diagnostics, timed_sample_fields_for_stage
from .runtime_outputs import (
    CoatingSummaryAccumulator,
    RuntimeOutputPayload,
    build_runtime_report,
    write_runtime_outputs,
)
from .output_buffers import RuntimeBuffers
from .particle_state import (
    SolverArrays,
    activate_release_cursor_until,
    initialize_solver_arrays,
)
from .runtime_plan import ReleaseCursor, SolverPlan, build_solver_plan
from .runtime_setup import RuntimeOptions, runtime_options_from_plan
from .stochastic_motion import apply_langevin_velocity_kick, merge_stochastic_motion_diagnostics, stochastic_motion_report
from .valid_mask_retry import resolve_valid_mask_retry_then_stop

VALID_MASK_POLICY_DIAGNOSTIC = 'diagnostic'
VALID_MASK_POLICY_RETRY_THEN_STOP = 'retry_then_stop'
VALID_MASK_POLICY_STRICT_CLEAN = 'strict_clean'

_COMPILED_MEMORY_ATTRS = (
    'axes',
    'times',
    'ux',
    'uy',
    'uz',
    'electric_x',
    'electric_y',
    'electric_z',
    'gas_density',
    'gas_mu',
    'gas_temperature',
    'valid_mask',
    'core_valid_mask',
    'mesh_vertices',
    'mesh_triangles',
    'accel_origin',
    'accel_cell_size',
    'accel_cell_offsets',
    'accel_triangle_indices',
)


def _array_nbytes_once(value: object, seen: set[int]) -> int:
    if value is None:
        return 0
    if isinstance(value, tuple):
        return int(sum(_array_nbytes_once(item, seen) for item in value))
    arr = np.asarray(value)
    ident = id(arr)
    if ident in seen:
        return 0
    seen.add(ident)
    return int(arr.nbytes)


def _compiled_backend_array_bytes(compiled: CompiledRuntimeBackendLike) -> int:
    seen: set[int] = set()
    return int(sum(_array_nbytes_once(getattr(compiled, name, None), seen) for name in _COMPILED_MEMORY_ATTRS))


def _assemble_saved_positions(
    save_positions: List[np.ndarray],
    *,
    capture_snapshots: bool,
    n_particles: int,
    spatial_dim: int,
) -> tuple[np.ndarray, float]:
    assembly_t0 = time.perf_counter()
    positions = (
        np.stack(save_positions, axis=0)
        if bool(capture_snapshots) and bool(save_positions)
        else np.zeros((0, int(n_particles), int(spatial_dim)), dtype=np.float64)
    )
    return positions, float(time.perf_counter() - assembly_t0)


def _numpy_array_bytes(arrays) -> int:
    return int(sum(int(np.asarray(arr).nbytes) for arr in arrays))


def _initialize_output_buffers(plan: SolverPlan) -> RuntimeBuffers:
    return RuntimeBuffers(output=plan.output)


def _virtual_mass_factor_array(
    force_runtime: ForceRuntimeParameters | None,
    rho_g: np.ndarray,
    rho_p: np.ndarray,
) -> np.ndarray:
    gas_density = np.asarray(rho_g, dtype=np.float64)
    particle_density = np.asarray(rho_p, dtype=np.float64)
    factor = np.ones_like(particle_density, dtype=np.float64)
    if force_runtime is None or not bool(force_runtime.virtual_mass_enabled):
        return factor
    coeff = max(float(force_runtime.virtual_mass_coefficient), 0.0)
    valid = (
        np.isfinite(gas_density)
        & np.isfinite(particle_density)
        & (gas_density > 0.0)
        & (particle_density > 0.0)
    )
    factor = np.where(valid, 1.0 + coeff * gas_density / np.maximum(particle_density, 1.0e-300), 1.0)
    return np.where(np.isfinite(factor) & (factor > 0.0), factor, 1.0)


def _add_timing(timing_accumulator: Dict[str, float], key: str, elapsed_s: float) -> None:
    timing_accumulator[key] = float(timing_accumulator.get(key, 0.0)) + float(max(0.0, elapsed_s))


def _require_solver_plan(state) -> SolverPlan:
    plan = getattr(state, 'solver_plan', None)
    if plan is None:
        raise ValueError('solver_plan is required for high-fidelity runtime helpers')
    return plan


class _DiscardList(list):
    discarding = True

    def append(self, _item) -> None:
        return None


@dataclass(frozen=True)
class StepLoopContext:
    step: ProcessStepRow
    phys: Mapping[str, object]
    body_accel: np.ndarray
    integrator_spec: IntegratorSpec
    save_every: int
    step_local_counter: int
    prev_step_name: str


@dataclass(frozen=True)
class StepLoopResult:
    t: float
    step_count: int
    elapsed_s: float


@dataclass
class RuntimeState:
    """Step-loop scratch and output accumulators.

    The particle arrays below are aliases to ``SolverArrays.state``. Keep them
    as compatibility handles for the current loop, not as a second owner of
    particle state.
    """

    x: np.ndarray
    v: np.ndarray
    released: np.ndarray
    active: np.ndarray
    stuck: np.ndarray
    absorbed: np.ndarray
    contact_sliding: np.ndarray
    contact_endpoint_stopped: np.ndarray
    contact_edge_index: np.ndarray
    contact_part_id: np.ndarray
    contact_normal: np.ndarray
    escaped: np.ndarray
    invalid_mask_stopped: np.ndarray
    numerical_boundary_stopped: np.ndarray
    invalid_stop_reason_code: np.ndarray
    save_positions: List[np.ndarray]
    save_meta: List[Dict[str, object]]
    wall_rows: List[Dict[str, object]]
    coating_summary_rows: object
    max_hit_rows: List[Dict[str, object]]
    step_rows: List[Dict[str, object]]
    wall_law_counts: Dict[str, int]
    wall_summary_counts: Dict[Tuple[int, str, str], int]
    collision_diagnostics: Dict[str, object]
    rng: np.random.Generator
    prev_step_name: Optional[str]
    step_local_counter: int
    save_index: int
    x_trial: np.ndarray
    v_trial: np.ndarray
    x_mid_trial: np.ndarray
    substep_counts: np.ndarray
    valid_mask_status_flags: np.ndarray
    valid_mask_mixed_seen: np.ndarray
    valid_mask_hard_seen: np.ndarray
    charge: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    release_cursor: ReleaseCursor | None = None
    solver_plan: SolverPlan | None = None
    solver_arrays: SolverArrays | None = None
    output_buffers: RuntimeBuffers | None = None
    stochastic_pending_dt_s: float = 0.0
    charge_pending_dt_s: float = 0.0
    stochastic_rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(12345))
    timing_accumulator: Dict[str, float] = field(default_factory=dict)


def _fallback_step(runtime, t_end: float) -> ProcessStepRow:
    return ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=float(t_end), output_segment_name='run')


def _current_step(runtime, t: float, t_end: float) -> ProcessStepRow:
    pst = runtime.process_steps
    row = pst.active_at(t) if pst is not None else None
    return row if row is not None else _fallback_step(runtime, t_end)


def _particle_tau_p(p_diameter: float, p_density: float, gas_mu: float, min_tau: float) -> float:
    mu = max(float(gas_mu), 1e-30)
    diameter = max(float(p_diameter), 1e-12)
    density = max(float(p_density), 1e-9)
    return max(float(min_tau), density * diameter * diameter / (18.0 * mu))


def _append_snapshot(
    save_positions: List[np.ndarray],
    save_meta: List[Dict[str, object]],
    *,
    save_index: int,
    t: float,
    step: ProcessStepRow,
    position: np.ndarray,
) -> None:
    save_positions.append(np.asarray(position, dtype=np.float64).copy())
    save_meta.append(
        {
            'save_index': int(save_index),
            'time_s': float(t),
            'step_name': step.step_name,
            'segment_name': _step_segment_name(step),
        }
    )


def _body_acceleration_vector(phys: Mapping[str, object], spatial_dim: int) -> np.ndarray:
    body_accel = np.asarray(phys['body_acceleration'], dtype=np.float64)
    if body_accel.size < int(spatial_dim):
        body_accel = np.pad(body_accel, (0, int(spatial_dim) - body_accel.size), constant_values=0.0)
    return body_accel


def _prepare_triangle_surface(runtime, spatial_dim: int) -> Optional[TriangleSurface3D]:
    if int(spatial_dim) != 3:
        return None
    if runtime.geometry_provider is None:
        raise ValueError('3D solver requires geometry_provider')
    geom3d = runtime.geometry_provider.geometry
    if geom3d.boundary_triangles is None:
        raise ValueError('3D solver requires geometry.boundary_triangles as geometry truth source')
    return build_triangle_surface(
        np.asarray(geom3d.boundary_triangles, dtype=np.float64),
        np.asarray(
            geom3d.boundary_triangle_part_ids
            if geom3d.boundary_triangle_part_ids is not None
            else np.zeros(np.asarray(geom3d.boundary_triangles).shape[0], dtype=np.int32),
            dtype=np.int32,
        ),
        validate_closed=True,
    )


def _update_adaptive_substep_diagnostics(
    collision_diagnostics: Dict[str, object],
    *,
    adaptive_substep_enabled: int,
    active: np.ndarray,
    substep_counts: np.ndarray,
) -> None:
    if int(adaptive_substep_enabled) == 0 or not np.any(active):
        return
    active_substeps = np.asarray(substep_counts[active], dtype=np.int64)
    collision_diagnostics['adaptive_substep_segments_count'] += int(np.sum(active_substeps))
    collision_diagnostics['adaptive_substep_trigger_count'] += int(np.count_nonzero(active_substeps > 1))


def _update_valid_mask_diagnostics(
    collision_diagnostics: Dict[str, object],
    *,
    valid_mask_status_flags: np.ndarray,
    valid_mask_mixed_seen: np.ndarray,
    valid_mask_hard_seen: np.ndarray,
) -> Tuple[int, int, int]:
    statuses = np.asarray(valid_mask_status_flags, dtype=np.uint8)
    mixed_step_mask = statuses == int(VALID_MASK_STATUS_MIXED_STENCIL)
    hard_step_mask = statuses == int(VALID_MASK_STATUS_HARD_INVALID)
    mixed_count_step = int(np.count_nonzero(mixed_step_mask))
    hard_count_step = int(np.count_nonzero(hard_step_mask))
    violation_count_step = int(mixed_count_step + hard_count_step)
    valid_mask_mixed_seen |= mixed_step_mask
    valid_mask_hard_seen |= hard_step_mask
    collision_diagnostics['valid_mask_violation_count'] += int(violation_count_step)
    collision_diagnostics['valid_mask_violation_particle_count'] = int(
        np.count_nonzero(valid_mask_mixed_seen | valid_mask_hard_seen)
    )
    collision_diagnostics['valid_mask_mixed_stencil_count'] += int(mixed_count_step)
    collision_diagnostics['valid_mask_mixed_stencil_particle_count'] = int(np.count_nonzero(valid_mask_mixed_seen))
    collision_diagnostics['valid_mask_hard_invalid_count'] += int(hard_count_step)
    collision_diagnostics['valid_mask_hard_invalid_particle_count'] = int(np.count_nonzero(valid_mask_hard_seen))
    return int(violation_count_step), int(mixed_count_step), int(hard_count_step)


def _commit_particle_state(
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    escaped: np.ndarray,
    *,
    particle_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> None:
    index = int(particle_index)
    x[index] = np.asarray(position, dtype=np.float64)
    v[index] = np.asarray(velocity, dtype=np.float64)
    if active[index] and (np.any(x[index] < mins - 1e-12) or np.any(x[index] > maxs + 1e-12)):
        escaped[index] = True
        active[index] = False


def _append_runtime_step_summary(
    step_rows: List[Dict[str, object]],
    *,
    buffers: RuntimeBuffers | None = None,
    t: float,
    step: ProcessStepRow,
    released: np.ndarray,
    active: np.ndarray,
    stuck: np.ndarray,
    absorbed: np.ndarray,
    contact_sliding: np.ndarray,
    escaped: np.ndarray,
    valid_mask_violation_count_step: int,
    valid_mask_mixed_stencil_count_step: int,
    valid_mask_hard_invalid_count_step: int,
    invalid_mask_stopped_count_step: int,
    save_positions_enabled: int = 1,
    write_wall_events_enabled: int = 1,
    write_diagnostics_enabled: int = 1,
) -> None:
    if buffers is not None and buffers.step_summary is not None and bool(buffers.output.write_step_summary):
        stopped_count = int(
            np.count_nonzero(
                stuck
                | absorbed
                | contact_sliding
                | escaped
            )
        )
        buffers.step_summary.append(
            time_s=float(t),
            step_name=step.step_name,
            segment_name=_step_segment_name(step),
            released_count=int(released.sum()),
            active_count=int(active.sum()),
            stuck_count=int(stuck.sum()),
            absorbed_count=int(absorbed.sum()),
            contact_sliding_count=int(contact_sliding.sum()),
            escaped_count=int(escaped.sum()),
            stopped_count=int(stopped_count),
            save_positions_enabled=int(save_positions_enabled),
            write_wall_events_enabled=int(write_wall_events_enabled),
            write_diagnostics_enabled=int(write_diagnostics_enabled),
            valid_mask_violation_count_step=int(valid_mask_violation_count_step),
            valid_mask_mixed_stencil_count_step=int(valid_mask_mixed_stencil_count_step),
            valid_mask_hard_invalid_count_step=int(valid_mask_hard_invalid_count_step),
            invalid_mask_stopped_count_step=int(invalid_mask_stopped_count_step),
        )
        if bool(getattr(step_rows, 'discarding', False)):
            return
    step_rows.append(
        {
            'time_s': float(t),
            'step_name': step.step_name,
            'segment_name': _step_segment_name(step),
            'released_count': int(released.sum()),
            'active_count': int(active.sum()),
            'stuck_count': int(stuck.sum()),
            'absorbed_count': int(absorbed.sum()),
            'contact_sliding_count': int(contact_sliding.sum()),
            'escaped_count': int(escaped.sum()),
            'save_positions_enabled': int(save_positions_enabled),
            'write_wall_events_enabled': int(write_wall_events_enabled),
            'write_diagnostics_enabled': int(write_diagnostics_enabled),
            'valid_mask_violation_count_step': int(valid_mask_violation_count_step),
            'valid_mask_mixed_stencil_count_step': int(valid_mask_mixed_stencil_count_step),
            'valid_mask_hard_invalid_count_step': int(valid_mask_hard_invalid_count_step),
            'invalid_mask_stopped_count_step': int(invalid_mask_stopped_count_step),
        }
    )


def _activate_released_particles(
    release_cursor: ReleaseCursor,
    released: np.ndarray,
    active: np.ndarray,
    t: float,
) -> np.ndarray:
    return activate_release_cursor_until(release_cursor, released, active, float(t), tolerance_s=1.0e-15)


def _record_active_count_summary(collision_diagnostics: Dict[str, object], active_idx: np.ndarray) -> None:
    active_count = int(np.asarray(active_idx).size)
    samples = int(collision_diagnostics.get('active_count_samples', 0)) + 1
    previous_mean = float(collision_diagnostics.get('active_count_mean', 0.0))
    collision_diagnostics['active_count_samples'] = int(samples)
    collision_diagnostics['active_count_mean'] = previous_mean + (float(active_count) - previous_mean) / float(samples)
    collision_diagnostics['active_count_max'] = int(
        max(active_count, int(collision_diagnostics.get('active_count_max', 0)))
    )


def _initialize_runtime_state(
    *,
    particles,
    plan: SolverPlan,
    arrays: SolverArrays,
    buffers: RuntimeBuffers,
    options: RuntimeOptions,
    spatial_dim: int,
    capture_outputs: bool,
) -> RuntimeState:
    n_particles = int(arrays.static.count)
    output_options = options.output_options
    capture_snapshots = bool(capture_outputs) and output_options.capture_positions()
    save_positions = [] if bool(capture_snapshots) else _DiscardList()
    save_meta = [] if bool(capture_snapshots) else _DiscardList()
    wall_rows = [] if bool(capture_outputs) and int(output_options.write_wall_events) != 0 else _DiscardList()
    coating_summary_rows = (
        CoatingSummaryAccumulator()
        if bool(capture_outputs) and int(output_options.write_coating_summary) != 0
        else _DiscardList()
    )
    max_hit_rows = [] if bool(capture_outputs) and int(output_options.write_max_hit_events) != 0 else _DiscardList()
    step_rows = (
        _DiscardList()
        if buffers.step_summary is not None
        else [] if bool(capture_outputs) and int(output_options.write_runtime_step_summary) != 0 else _DiscardList()
    )
    particle_state = arrays.state
    return RuntimeState(
        x=particle_state.x,
        v=particle_state.v,
        released=particle_state.released,
        active=particle_state.active,
        stuck=particle_state.stuck,
        absorbed=particle_state.absorbed,
        contact_sliding=particle_state.contact_sliding,
        contact_endpoint_stopped=particle_state.contact_endpoint_stopped,
        contact_edge_index=particle_state.contact_edge_index,
        contact_part_id=particle_state.contact_part_id,
        contact_normal=particle_state.contact_normal,
        escaped=particle_state.escaped,
        invalid_mask_stopped=particle_state.invalid_mask_stopped,
        numerical_boundary_stopped=particle_state.numerical_boundary_stopped,
        invalid_stop_reason_code=particle_state.invalid_stop_reason_code,
        save_positions=save_positions,
        save_meta=save_meta,
        wall_rows=wall_rows,
        coating_summary_rows=coating_summary_rows,
        max_hit_rows=max_hit_rows,
        step_rows=step_rows,
        wall_law_counts={},
        wall_summary_counts={},
        collision_diagnostics=initial_collision_diagnostics(),
        timing_accumulator={},
        rng=np.random.default_rng(int(plan.rng_seed)),
        stochastic_rng=np.random.default_rng(int(options.stochastic_motion.seed)),
        prev_step_name=None,
        step_local_counter=0,
        save_index=1,
        x_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        v_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        x_mid_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        substep_counts=np.ones(n_particles, dtype=np.int32),
        valid_mask_status_flags=particle_state.valid_mask_status,
        valid_mask_mixed_seen=np.zeros(n_particles, dtype=bool),
        valid_mask_hard_seen=np.zeros(n_particles, dtype=bool),
        charge=particle_state.charge_C,
        release_cursor=arrays.release_cursor,
        solver_plan=plan,
        solver_arrays=arrays,
        output_buffers=buffers,
    )


def _resolve_step_loop_context(
    runtime,
    *,
    t: float,
    t_end: float,
    spatial_dim: int,
    base_save_every: int,
    prev_step_name: Optional[str],
    step_local_counter: int,
) -> StepLoopContext:
    step = _current_step(runtime, t, t_end)
    phys = resolve_step_physics(runtime.physics_catalog, step)
    next_prev_step_name = str(prev_step_name) if prev_step_name is not None else ''
    next_step_local_counter = int(step_local_counter)
    if next_prev_step_name != step.step_name:
        next_step_local_counter = 0
        next_prev_step_name = step.step_name
    next_step_local_counter += 1
    save_every = int(base_save_every)
    return StepLoopContext(
        step=step,
        phys=phys,
        body_accel=_body_acceleration_vector(phys, spatial_dim),
        integrator_spec=get_integrator_spec(str(phys.get('integrator', 'drag_relaxation'))),
        save_every=int(save_every),
        step_local_counter=int(next_step_local_counter),
        prev_step_name=str(next_prev_step_name),
    )


def _apply_valid_mask_retry_then_stop(
    *,
    state: RuntimeState,
    options: RuntimeOptions,
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    integrator_mode: int,
    dt_step: float,
    t_end_step: float,
    phys: Mapping[str, object],
    body_accel: np.ndarray,
    tau_p: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density: Optional[np.ndarray] = None,
    particle_mass: Optional[np.ndarray] = None,
    dep_particle_rel_permittivity: Optional[np.ndarray] = None,
    thermophoretic_coeff: Optional[np.ndarray] = None,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
    electric_q_over_m_particle: Optional[np.ndarray] = None,
    force_runtime: ForceRuntimeParameters | None = None,
    particle_indices: Optional[np.ndarray] = None,
) -> int:
    plan = _require_solver_plan(state)
    valid_mask_policy = str(plan.valid_mask_policy)
    adaptive_substep_enabled = int(plan.adaptive_substep_enabled)
    adaptive_substep_tau_ratio = float(plan.adaptive_substep_tau_ratio)
    adaptive_substep_max_splits = int(plan.adaptive_substep_max_splits)
    drag_model_mode = int(plan.drag_model_mode)
    if valid_mask_policy not in {VALID_MASK_POLICY_RETRY_THEN_STOP, VALID_MASK_POLICY_STRICT_CLEAN}:
        return 0
    if particle_indices is None:
        candidate_indices = np.flatnonzero(state.active)
    else:
        candidate_indices = np.asarray(particle_indices, dtype=np.int64)
        candidate_indices = candidate_indices[
            (candidate_indices >= 0)
            & (candidate_indices < int(state.active.size))
            & state.active[candidate_indices]
        ]
    if valid_mask_policy == VALID_MASK_POLICY_STRICT_CLEAN:
        violating = candidate_indices[
            state.valid_mask_status_flags[candidate_indices] != int(VALID_MASK_STATUS_CLEAN)
        ]
    else:
        violating = candidate_indices[
            state.valid_mask_status_flags[candidate_indices] == int(VALID_MASK_STATUS_HARD_INVALID)
        ]
    if violating.size == 0:
        return 0

    stopped_count_step = 0
    for particle_index_raw in violating:
        particle_index = int(particle_index_raw)
        x_start = np.asarray(state.x[particle_index], dtype=np.float64).copy()
        v_start = np.asarray(state.v[particle_index], dtype=np.float64).copy()
        particle_density_i = 1000.0
        if particle_density is not None:
            particle_density_i = float(np.asarray(particle_density, dtype=np.float64)[particle_index])
        particle_mass_i = 0.0
        if particle_mass is not None:
            particle_mass_i = float(np.asarray(particle_mass, dtype=np.float64)[particle_index])
        dep_particle_rel_permittivity_i = float('nan')
        if dep_particle_rel_permittivity is not None:
            dep_particle_rel_permittivity_i = float(np.asarray(dep_particle_rel_permittivity, dtype=np.float64)[particle_index])
        thermophoretic_coeff_i = float('nan')
        if thermophoretic_coeff is not None:
            thermophoretic_coeff_i = float(np.asarray(thermophoretic_coeff, dtype=np.float64)[particle_index])
        electric_q_over_m_i = None
        if electric_q_over_m_particle is not None:
            electric_q_over_m_i = float(np.asarray(electric_q_over_m_particle, dtype=np.float64)[particle_index])
        resolution = resolve_valid_mask_retry_then_stop(
            resolve_prefix=resolve_valid_mask_prefix,
            collision_diagnostics=state.collision_diagnostics,
            x0=x_start,
            v0=v_start,
            dt_segment=float(dt_step),
            t_end_segment=float(t_end_step),
            spatial_dim=int(spatial_dim),
            compiled=compiled,
            integrator_mode=int(integrator_mode),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            tau_p_i=float(tau_p[particle_index]),
            particle_diameter_i=float(particle_diameter[particle_index]),
            particle_density_i=float(particle_density_i),
            particle_mass_i=float(particle_mass_i),
            dep_particle_rel_permittivity_i=float(dep_particle_rel_permittivity_i),
            thermophoretic_coeff_i=float(thermophoretic_coeff_i),
            flow_scale_particle_i=float(flow_scale_particle[particle_index]),
            drag_scale_particle_i=float(drag_scale_particle[particle_index]),
            body_scale_particle_i=float(body_scale_particle[particle_index]),
            global_flow_scale=float(phys['flow_scale']),
            global_drag_tau_scale=float(phys['drag_tau_scale']),
            global_body_accel_scale=float(phys['body_accel_scale']),
            body_accel=body_accel,
            min_tau_p_s=float(phys['min_tau_p_s']),
            gas_density_kgm3=float(phys['gas_density_kgm3']),
            gas_mu_pas=float(phys['gas_mu_pas']),
            gas_temperature_K=float(phys.get('gas_temperature_K', 300.0)),
            gas_molecular_mass_kg=float(phys.get('gas_molecular_mass_kg', 60.0 * 1.66053906660e-27)),
            drag_model_mode=int(drag_model_mode),
            electric_q_over_m_i=electric_q_over_m_i,
            force_runtime=force_runtime,
            require_clean_prefix=valid_mask_policy == VALID_MASK_POLICY_STRICT_CLEAN,
        )

        _mark_invalid_mask_stopped(
            state=state,
            particle_index=particle_index,
            position=resolution.position,
            velocity=resolution.velocity,
            update_trial_buffers=True,
            reason=(
                'freeflight_valid_mask_hard_invalid_prefix_clipped'
                if bool(resolution.found_valid_prefix)
                else 'freeflight_valid_mask_hard_invalid_retry_exhausted'
            ),
        )
        stopped_count_step += 1
    return int(stopped_count_step)


def _mark_invalid_mask_stopped(
    *,
    state: RuntimeState,
    particle_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    update_trial_buffers: bool,
    reason: str,
) -> None:
    index = int(particle_index)
    _stop_particle_motion(
        state=state,
        particle_index=index,
        position=position,
        velocity=velocity,
        update_trial_buffers=update_trial_buffers,
    )
    state.numerical_boundary_stopped[index] = False
    if not bool(state.invalid_mask_stopped[index]):
        state.invalid_mask_stopped[index] = True
        reason_code = invalid_stop_reason_code(str(reason))
        state.invalid_stop_reason_code[index] = np.uint8(reason_code)
        reason_name = invalid_stop_reason_name(int(reason_code))
        state.collision_diagnostics['invalid_mask_stopped_count'] += 1
        increment_named_count(state.collision_diagnostics, 'invalid_mask_stop_reason_counts', reason_name)


def _stop_particle_motion(
    *,
    state: RuntimeState,
    particle_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    update_trial_buffers: bool,
) -> None:
    index = int(particle_index)
    pos = np.asarray(position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    state.x[index] = pos
    state.v[index] = vel
    if bool(update_trial_buffers):
        state.x_trial[index] = pos
        state.v_trial[index] = vel
        state.x_mid_trial[index] = pos
    state.active[index] = False
    state.stuck[index] = False
    state.absorbed[index] = False
    state.contact_sliding[index] = False
    state.contact_endpoint_stopped[index] = False
    state.contact_edge_index[index] = -1
    state.contact_part_id[index] = 0
    state.contact_normal[index] = 0.0
    state.escaped[index] = False


def _mark_numerical_boundary_stopped(
    *,
    state: RuntimeState,
    particle_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    update_trial_buffers: bool,
    reason: str,
) -> None:
    index = int(particle_index)
    _stop_particle_motion(
        state=state,
        particle_index=index,
        position=position,
        velocity=velocity,
        update_trial_buffers=update_trial_buffers,
    )
    state.invalid_mask_stopped[index] = False
    if not bool(state.numerical_boundary_stopped[index]):
        state.numerical_boundary_stopped[index] = True
        state.collision_diagnostics['numerical_boundary_stop_count'] = int(
            state.collision_diagnostics.get('numerical_boundary_stop_count', 0)
        ) + 1
        increment_named_count(state.collision_diagnostics, 'numerical_boundary_stop_reason_counts', reason)


def _boundary_edge_arrays_2d(runtime) -> Tuple[Optional[np.ndarray], np.ndarray]:
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None:
        return None, np.zeros(0, dtype=np.int32)
    geom = geometry_provider.geometry
    if int(geom.spatial_dim) != 2 or geom.boundary_edges is None:
        return None, np.zeros(0, dtype=np.int32)
    segments = np.asarray(geom.boundary_edges, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or segments.shape[0] == 0:
        return None, np.zeros(0, dtype=np.int32)
    part_ids = np.asarray(
        geom.boundary_edge_part_ids
        if geom.boundary_edge_part_ids is not None
        else np.zeros(segments.shape[0], dtype=np.int32),
        dtype=np.int32,
    )
    if part_ids.size < segments.shape[0]:
        part_ids = np.pad(part_ids, (0, int(segments.shape[0] - part_ids.size)), constant_values=0)
    return segments, part_ids[: segments.shape[0]]


def _point_triangle_barycentric_3d(point: np.ndarray, triangle: np.ndarray) -> Optional[np.ndarray]:
    p = np.asarray(point, dtype=np.float64)
    tri = np.asarray(triangle, dtype=np.float64)
    a = tri[0]
    v0 = tri[1] - a
    v1 = tri[2] - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(float(denom)) <= 1.0e-30:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.asarray([1.0 - v - w, v, w], dtype=np.float64)


def _advance_scalar_relaxation_array(
    v0: np.ndarray,
    target: np.ndarray,
    body: np.ndarray,
    tau: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dt_val = max(float(dt), 0.0)
    tau_val = np.maximum(np.asarray(tau, dtype=np.float64), 1.0e-12)
    equilibrium = np.asarray(target, dtype=np.float64) + np.asarray(body, dtype=np.float64) * tau_val
    exponent = np.clip(dt_val / tau_val, 0.0, 700.0)
    decay = np.exp(-exponent)
    velocity = equilibrium + (np.asarray(v0, dtype=np.float64) - equilibrium) * decay
    displacement = equilibrium * dt_val + (np.asarray(v0, dtype=np.float64) - equilibrium) * tau_val * (1.0 - decay)
    return displacement, velocity


def _compiled_has_transient_time(compiled: CompiledRuntimeBackendLike) -> bool:
    times_raw = compiled.get('times') if isinstance(compiled, Mapping) else getattr(compiled, 'times', None)
    if times_raw is None:
        return False
    times = np.asarray(times_raw, dtype=np.float64)
    return bool(times.size > 1)


def _electric_q_over_m_particle(
    *,
    charge_model: ChargeModelConfig,
    charge: np.ndarray,
    particle_mass: np.ndarray,
) -> Optional[np.ndarray]:
    charge_arr = np.asarray(charge, dtype=np.float64)
    mass_arr = np.asarray(particle_mass, dtype=np.float64)
    valid_charge = np.isfinite(charge_arr) & (np.abs(charge_arr) > 0.0)
    if not bool(charge_model.enabled) and not np.any(valid_charge):
        return None
    qom = np.zeros_like(charge_arr, dtype=np.float64)
    np.divide(charge_arr, mass_arr, out=qom, where=np.isfinite(mass_arr) & (mass_arr > 0.0))
    return qom


def _finite_q_over_m_summary(charge: np.ndarray, particle_mass: np.ndarray) -> dict:
    charge_arr = np.asarray(charge, dtype=np.float64)
    mass_arr = np.asarray(particle_mass, dtype=np.float64)
    qom = np.full_like(charge_arr, np.nan, dtype=np.float64)
    valid = np.isfinite(charge_arr) & np.isfinite(mass_arr) & (mass_arr > 0.0)
    np.divide(charge_arr, mass_arr, out=qom, where=valid)
    finite = qom[np.isfinite(qom)]
    charged = finite[np.abs(finite) > 0.0]
    if finite.size == 0:
        return {'count': 0, 'charged_count': 0}
    quantiles = np.quantile(finite, [0.0, 0.5, 0.9, 1.0])
    return {
        'count': int(finite.size),
        'charged_count': int(charged.size),
        'min': float(quantiles[0]),
        'median': float(quantiles[1]),
        'p90': float(quantiles[2]),
        'max': float(quantiles[3]),
    }


def _advance_contact_sliding_particles_2d(
    *,
    runtime,
    state: RuntimeState,
    options: RuntimeOptions,
    compiled: CompiledRuntimeBackendLike,
    boundary_service: BoundaryService,
    tau_p: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density: np.ndarray,
    particle_mass: np.ndarray,
    dep_particle_rel_permittivity: np.ndarray,
    thermophoretic_coeff: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
    phys: Mapping[str, object],
    body_accel: np.ndarray,
    dt_step: float,
    t_next: float,
    electric_q_over_m_particle: Optional[np.ndarray] = None,
) -> None:
    contact_mask = state.active & state.contact_sliding
    if not _compiled_has_transient_time(compiled):
        contact_mask &= ~state.contact_endpoint_stopped
    indices = np.flatnonzero(contact_mask)
    if indices.size == 0:
        return
    segments, part_ids = _boundary_edge_arrays_2d(runtime)
    if segments is None:
        diagnostics = state.collision_diagnostics
        diagnostics['contact_frame_fail_count'] = int(diagnostics.get('contact_frame_fail_count', 0)) + int(indices.size)
        return
    plan = _require_solver_plan(state)
    epsilon_offset_m = float(plan.epsilon_offset_m)
    on_boundary_tol_m = float(plan.on_boundary_tol_m)
    drag_model_mode = int(plan.drag_model_mode)
    epsilon = max(float(epsilon_offset_m), 1.0e-12)
    diagnostics = state.collision_diagnostics
    field_plan = plan.field_sample
    edge_index = np.asarray(state.contact_edge_index[indices], dtype=np.int64)
    missing_edge = (edge_index < 0) | (edge_index >= int(segments.shape[0]))
    for particle_index in indices[missing_edge]:
        normal_hint = np.asarray(state.contact_normal[int(particle_index)], dtype=np.float64)
        frame = contact_frame_on_boundary_edge_2d(
            runtime,
            state.x[int(particle_index)],
            part_id_hint=int(state.contact_part_id[int(particle_index)]),
            normal_hint=normal_hint,
        )
        if frame is None:
            diagnostics['contact_frame_fail_count'] = int(diagnostics.get('contact_frame_fail_count', 0)) + 1
            continue
        state.contact_edge_index[int(particle_index)] = int(frame.edge_index)

    indices = indices[
        (state.contact_edge_index[indices] >= 0)
        & (state.contact_edge_index[indices] < int(segments.shape[0]))
    ]
    if indices.size == 0:
        return
    edge_index = np.asarray(state.contact_edge_index[indices], dtype=np.int64)
    q0 = segments[edge_index, 0, :]
    q1 = segments[edge_index, 1, :]
    edge = q1 - q0
    length = np.linalg.norm(edge, axis=1)
    valid_edge = length > 1.0e-30
    if not np.all(valid_edge):
        failed = indices[~valid_edge]
        diagnostics['contact_frame_fail_count'] = int(diagnostics.get('contact_frame_fail_count', 0)) + int(failed.size)
        indices = indices[valid_edge]
        q0 = q0[valid_edge]
        q1 = q1[valid_edge]
        edge = edge[valid_edge]
        length = length[valid_edge]
        edge_index = edge_index[valid_edge]
        if indices.size == 0:
            return

    tangent = edge / length[:, None]
    normal = np.asarray(state.contact_normal[indices], dtype=np.float64)
    normal_mag = np.linalg.norm(normal, axis=1)
    bad_normal = normal_mag <= 1.0e-30
    if np.any(bad_normal):
        fallback_normal = np.column_stack((-tangent[bad_normal, 1], tangent[bad_normal, 0]))
        normal[bad_normal] = fallback_normal
        normal_mag[bad_normal] = 1.0
    normal = normal / normal_mag[:, None]
    edge_len2 = length * length
    alpha = np.einsum('ij,ij->i', state.x[indices] - q0, edge) / edge_len2
    alpha = np.clip(alpha, 0.0, 1.0)
    projection = q0 + alpha[:, None] * edge
    x_contact = projection - epsilon * normal

    v_old = np.asarray(state.v[indices], dtype=np.float64)
    v_normal = np.einsum('ij,ij->i', v_old[:, :2], normal)
    v_tangent = np.einsum('ij,ij->i', v_old[:, :2], tangent)
    force_runtime = options.force_runtime
    density_local = np.asarray(particle_density, dtype=np.float64)[indices]
    needs_local_gas = (
        int(drag_model_mode) != int(DRAG_MODEL_STOKES)
        or bool(force_runtime.gravity_buoyancy_enabled)
        or bool(force_runtime.virtual_mass_enabled)
    )
    sampled_contact = timed_sample_fields_for_stage(
        compiled,
        field_plan,
        x_contact,
        float(t_next),
        spatial_dim=2,
        need_flow=True,
        need_gas_properties=bool(needs_local_gas),
        need_valid_mask=False,
        fallback_density_kgm3=float(phys['gas_density_kgm3']),
        fallback_mu_pas=float(phys['gas_mu_pas']),
        fallback_temperature_K=float(phys['gas_temperature_K']),
    )
    record_field_sampling_diagnostics(diagnostics, sampled_contact.samples, sampled_contact.elapsed_s)
    flow = (
        sampled_contact.samples.flow
        if sampled_contact.samples.flow is not None
        else np.zeros((indices.size, 2), dtype=np.float64)
    )
    electric_qom = (
        None
        if electric_q_over_m_particle is None
        else np.asarray(electric_q_over_m_particle, dtype=np.float64)[indices]
    )
    accel = _sample_acceleration_vectors_at(
        compiled,
        2,
        float(t_next),
        x_contact,
        electric_q_over_m=electric_qom,
        force_runtime=options.force_runtime,
        particle_diameter=np.asarray(particle_diameter, dtype=np.float64)[indices],
        particle_density=np.asarray(particle_density, dtype=np.float64)[indices],
        particle_mass=np.asarray(particle_mass, dtype=np.float64)[indices],
        dep_particle_rel_permittivity=np.asarray(dep_particle_rel_permittivity, dtype=np.float64)[indices],
        thermophoretic_coeff=np.asarray(thermophoretic_coeff, dtype=np.float64)[indices],
        velocity=v_old[:, :2],
        gas_density_kgm3=float(phys['gas_density_kgm3']),
        gas_mu_pas=float(phys['gas_mu_pas']),
        gas_temperature_K=float(phys['gas_temperature_K']),
        gas_molecular_mass_kg=float(phys['gas_molecular_mass_kg']),
    )
    rho_g = np.full(indices.size, float(phys['gas_density_kgm3']), dtype=np.float64)
    mu_g = np.full(indices.size, float(phys['gas_mu_pas']), dtype=np.float64)
    temp_g = np.full(indices.size, float(phys['gas_temperature_K']), dtype=np.float64)
    if bool(needs_local_gas) and sampled_contact.samples.gas_density is not None:
        rho_g = sampled_contact.samples.gas_density
        mu_g = sampled_contact.samples.gas_mu if sampled_contact.samples.gas_mu is not None else mu_g
        temp_g = (
            sampled_contact.samples.gas_temperature
            if sampled_contact.samples.gas_temperature is not None
            else temp_g
        )
    body_field_scale = float(phys['body_accel_scale']) * np.asarray(body_scale_particle[indices], dtype=np.float64)
    body_base = np.asarray(body_accel, dtype=np.float64)[:2][None, :] * body_field_scale[:, None]
    gravity_factor = np.ones(indices.size, dtype=np.float64)
    if bool(force_runtime.gravity_buoyancy_enabled):
        gravity_factor = 1.0 - rho_g / np.maximum(density_local, 1.0e-300)
    mass_factor = _virtual_mass_factor_array(force_runtime, rho_g, density_local)
    body_eff = (gravity_factor[:, None] * body_base + accel[:, :2]) / mass_factor[:, None]
    target = float(phys['flow_scale']) * np.asarray(flow_scale_particle[indices], dtype=np.float64)[:, None] * flow[:, :2]

    tau_stokes = np.asarray(tau_p[indices], dtype=np.float64) * float(phys['drag_tau_scale'])
    tau_stokes *= np.maximum(np.asarray(drag_scale_particle[indices], dtype=np.float64), 1.0e-6)
    tau_stokes = np.maximum(float(phys['min_tau_p_s']), tau_stokes)
    if int(drag_model_mode) == int(DRAG_MODEL_STOKES):
        tau_eff = tau_stokes
    else:
        slip = np.linalg.norm(v_old[:, :2] - target[:, :2], axis=1)
        tau_eff = np.asarray(
            [
                effective_tau_from_slip_speed(
                    float(tau_i),
                    float(slip_i),
                    float(diameter_i),
                    float(rho_i),
                    float(mu_i),
                    int(drag_model_mode),
                    float(phys['min_tau_p_s']),
                    float(density_i),
                    float(temp_i),
                    float(phys['gas_molecular_mass_kg']),
                )
                for tau_i, slip_i, diameter_i, density_i, rho_i, mu_i, temp_i in zip(
                    tau_stokes,
                    slip,
                    particle_diameter[indices],
                    density_local,
                    rho_g,
                    mu_g,
                    temp_g,
                )
            ],
            dtype=np.float64,
        )
    tau_eff = tau_eff * mass_factor

    target_normal = np.einsum('ij,ij->i', target, normal)
    body_normal = np.einsum('ij,ij->i', body_eff, normal)
    normal_tendency = (target_normal - v_normal) / np.maximum(tau_eff, 1.0e-12) + body_normal
    release_candidate = normal_tendency < -1.0e-10
    release_mask = np.zeros(indices.size, dtype=bool)
    if np.any(release_candidate):
        candidate_rows = np.flatnonzero(release_candidate)
        probe_dt = np.minimum(float(dt_step), np.maximum(tau_eff[candidate_rows], 1.0e-12))
        probe_dt = np.maximum(probe_dt, min(float(dt_step), 1.0e-9))
        x_probe = (
            x_contact[candidate_rows]
            + v_tangent[candidate_rows, None] * probe_dt[:, None] * tangent[candidate_rows]
            + 0.5 * normal_tendency[candidate_rows, None] * (probe_dt[:, None] ** 2) * normal[candidate_rows]
        )
        probe_inside = points_inside_geometry_2d(runtime, x_probe, on_boundary_tol_m=float(on_boundary_tol_m))
        sampled_probe = timed_sample_fields_for_stage(
            compiled,
            field_plan,
            x_probe,
            float(t_next),
            spatial_dim=2,
            need_flow=False,
            need_gas_properties=False,
            need_valid_mask=True,
        )
        record_field_sampling_diagnostics(diagnostics, sampled_probe.samples, sampled_probe.elapsed_s)
        probe_status = sampled_probe.samples.valid_mask_status
        if probe_status is None:
            probe_status = _sample_valid_mask_statuses(compiled, x_probe)
        probe_clean = probe_inside & (probe_status < int(VALID_MASK_STATUS_HARD_INVALID))
        release_mask[candidate_rows[probe_clean]] = True
        rejected = int(np.count_nonzero(~probe_clean))
        if rejected:
            diagnostics['contact_release_probe_reject_count'] = int(
                diagnostics.get('contact_release_probe_reject_count', 0)
            ) + int(rejected)
    if np.any(release_mask):
        release_indices = indices[release_mask]
        state.contact_sliding[release_indices] = False
        state.contact_endpoint_stopped[release_indices] = False
        state.contact_edge_index[release_indices] = -1
        state.contact_part_id[release_indices] = 0
        state.contact_normal[release_indices] = 0.0
        state.x[release_indices] = x_contact[release_mask]
        state.v[release_indices] = v_tangent[release_mask, None] * tangent[release_mask]
        diagnostics['contact_release_count'] = int(diagnostics.get('contact_release_count', 0)) + int(release_indices.size)

    keep_mask = ~release_mask
    if not np.any(keep_mask):
        return
    endpoint_hold_mask = keep_mask & state.contact_endpoint_stopped[indices]
    if np.any(endpoint_hold_mask):
        hold_indices = indices[endpoint_hold_mask]
        state.x[hold_indices] = x_contact[endpoint_hold_mask]
        state.v[hold_indices] = 0.0
        state.x_trial[hold_indices] = state.x[hold_indices]
        state.v_trial[hold_indices] = state.v[hold_indices]
        state.x_mid_trial[hold_indices] = state.x[hold_indices]
        state.contact_edge_index[hold_indices] = edge_index[endpoint_hold_mask]
        state.contact_part_id[hold_indices] = part_ids[edge_index[endpoint_hold_mask]]
        state.contact_normal[hold_indices] = normal[endpoint_hold_mask]
        diagnostics['contact_endpoint_hold_count'] = int(
            diagnostics.get('contact_endpoint_hold_count', 0)
        ) + int(hold_indices.size)

    keep_mask = keep_mask & ~state.contact_endpoint_stopped[indices]
    if not np.any(keep_mask):
        return
    keep_indices = indices[keep_mask]
    target_tangent = np.einsum('ij,ij->i', target[keep_mask], tangent[keep_mask])
    body_tangent = np.einsum('ij,ij->i', body_eff[keep_mask], tangent[keep_mask])
    tangent_displacement, tangent_velocity = _advance_scalar_relaxation_array(
        v_tangent[keep_mask],
        target_tangent,
        body_tangent,
        tau_eff[keep_mask],
        float(dt_step),
    )
    alpha_next = alpha[keep_mask] + tangent_displacement / np.maximum(length[keep_mask], 1.0e-30)
    endpoint_hit = (alpha_next <= 0.0) | (alpha_next >= 1.0)
    alpha_clipped = np.clip(alpha_next, 0.0, 1.0)
    x_wall = q0[keep_mask] + alpha_clipped[:, None] * edge[keep_mask]
    x_next = x_wall - epsilon * normal[keep_mask]
    sampled_next = timed_sample_fields_for_stage(
        compiled,
        field_plan,
        x_next,
        float(t_next),
        spatial_dim=2,
        need_flow=False,
        need_gas_properties=False,
        need_valid_mask=True,
    )
    record_field_sampling_diagnostics(diagnostics, sampled_next.samples, sampled_next.elapsed_s)
    status = sampled_next.samples.valid_mask_status
    if status is None:
        status = _sample_valid_mask_statuses(compiled, x_next)
    inside = points_inside_geometry_2d(runtime, x_next, on_boundary_tol_m=float(on_boundary_tol_m))
    reject = (status >= int(VALID_MASK_STATUS_HARD_INVALID)) | (~inside)
    if np.any(reject):
        reject_indices = keep_indices[reject]
        diagnostics['contact_valid_mask_reject_count'] = int(
            diagnostics.get('contact_valid_mask_reject_count', 0)
        ) + int(reject_indices.size)
        state.x[reject_indices] = x_contact[keep_mask][reject]
        state.v[reject_indices] = 0.0
        state.x_trial[reject_indices] = state.x[reject_indices]
        state.v_trial[reject_indices] = state.v[reject_indices]
        state.x_mid_trial[reject_indices] = state.x[reject_indices]

    accept = ~reject
    if not np.any(accept):
        return
    accept_indices = keep_indices[accept]
    accept_endpoint = endpoint_hit[accept]
    accept_tangent_velocity = tangent_velocity[accept]
    accept_tangent = tangent[keep_mask][accept]
    state.x[accept_indices] = x_next[accept]
    state.v[accept_indices] = accept_tangent_velocity[:, None] * accept_tangent
    if np.any(accept_endpoint):
        state.v[accept_indices[accept_endpoint]] = 0.0
        state.contact_endpoint_stopped[accept_indices[accept_endpoint]] = True
        diagnostics['contact_endpoint_stop_count'] = int(diagnostics.get('contact_endpoint_stop_count', 0)) + int(
            np.count_nonzero(accept_endpoint)
        )
    state.x_trial[accept_indices] = state.x[accept_indices]
    state.v_trial[accept_indices] = state.v[accept_indices]
    state.x_mid_trial[accept_indices] = state.x[accept_indices]
    state.contact_edge_index[accept_indices] = edge_index[keep_mask][accept]
    state.contact_part_id[accept_indices] = part_ids[edge_index[keep_mask][accept]]
    state.contact_normal[accept_indices] = normal[keep_mask][accept]
    diagnostics['contact_tangent_step_count'] = int(diagnostics.get('contact_tangent_step_count', 0)) + int(
        accept_indices.size
    )
    diagnostics['contact_tangent_time_total_s'] = float(
        diagnostics.get('contact_tangent_time_total_s', 0.0)
    ) + float(dt_step) * float(accept_indices.size)


def _advance_contact_sliding_particles_3d(
    *,
    runtime,
    state: RuntimeState,
    options: RuntimeOptions,
    compiled: CompiledRuntimeBackendLike,
    boundary_service: BoundaryService,
    tau_p: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density: np.ndarray,
    particle_mass: np.ndarray,
    dep_particle_rel_permittivity: np.ndarray,
    thermophoretic_coeff: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
    phys: Mapping[str, object],
    body_accel: np.ndarray,
    dt_step: float,
    t_next: float,
    electric_q_over_m_particle: Optional[np.ndarray] = None,
) -> None:
    surface = boundary_service.triangle_surface_3d
    contact_mask = state.active & state.contact_sliding
    if not _compiled_has_transient_time(compiled):
        contact_mask &= ~state.contact_endpoint_stopped
    indices = np.flatnonzero(contact_mask)
    if indices.size == 0:
        return
    diagnostics = state.collision_diagnostics
    if surface is None:
        diagnostics['contact_frame_fail_count'] = int(diagnostics.get('contact_frame_fail_count', 0)) + int(indices.size)
        return

    plan = _require_solver_plan(state)
    epsilon_offset_m = float(plan.epsilon_offset_m)
    drag_model_mode = int(plan.drag_model_mode)
    epsilon = max(float(epsilon_offset_m), 1.0e-12)
    field_plan = plan.field_sample
    tri_index = np.asarray(state.contact_edge_index[indices], dtype=np.int64)
    missing = (tri_index < 0) | (tri_index >= int(surface.triangles.shape[0]))
    for particle_index in indices[missing]:
        hit = boundary_service.nearest_projection(state.x[int(particle_index)], state.x[int(particle_index)])
        if hit is None or int(hit.primitive_id) < 0:
            diagnostics['contact_frame_fail_count'] = int(diagnostics.get('contact_frame_fail_count', 0)) + 1
            continue
        state.contact_edge_index[int(particle_index)] = int(hit.primitive_id)
        state.contact_part_id[int(particle_index)] = int(hit.part_id)
        state.contact_normal[int(particle_index)] = np.asarray(hit.normal, dtype=np.float64)

    indices = indices[
        (state.contact_edge_index[indices] >= 0)
        & (state.contact_edge_index[indices] < int(surface.triangles.shape[0]))
    ]
    if indices.size == 0:
        return

    tri_index = np.asarray(state.contact_edge_index[indices], dtype=np.int64)
    triangles = np.asarray(surface.triangles[tri_index], dtype=np.float64)
    q0 = triangles[:, 0, :]
    normal = np.asarray(state.contact_normal[indices], dtype=np.float64)
    normal_mag = np.linalg.norm(normal, axis=1)
    bad_normal = normal_mag <= 1.0e-30
    if np.any(bad_normal):
        normal[bad_normal] = np.asarray(surface.normals[tri_index[bad_normal]], dtype=np.float64)
        normal_mag[bad_normal] = np.linalg.norm(normal[bad_normal], axis=1)
    valid_normal = normal_mag > 1.0e-30
    if not np.all(valid_normal):
        failed = indices[~valid_normal]
        diagnostics['contact_frame_fail_count'] = int(diagnostics.get('contact_frame_fail_count', 0)) + int(failed.size)
        indices = indices[valid_normal]
        tri_index = tri_index[valid_normal]
        triangles = triangles[valid_normal]
        q0 = q0[valid_normal]
        normal = normal[valid_normal]
        normal_mag = normal_mag[valid_normal]
        if indices.size == 0:
            return
    normal = normal / normal_mag[:, None]

    signed_distance = np.einsum('ij,ij->i', state.x[indices] - q0, normal)
    x_wall = state.x[indices] - signed_distance[:, None] * normal
    x_contact = x_wall - epsilon * normal
    v_old = np.asarray(state.v[indices], dtype=np.float64)
    v_normal = np.einsum('ij,ij->i', v_old[:, :3], normal)
    v_tangent = v_old[:, :3] - v_normal[:, None] * normal

    force_runtime = options.force_runtime
    density_local = np.asarray(particle_density, dtype=np.float64)[indices]
    needs_local_gas = (
        int(drag_model_mode) != int(DRAG_MODEL_STOKES)
        or bool(force_runtime.gravity_buoyancy_enabled)
        or bool(force_runtime.virtual_mass_enabled)
    )
    sampled_contact = timed_sample_fields_for_stage(
        compiled,
        field_plan,
        x_contact,
        float(t_next),
        spatial_dim=3,
        need_flow=True,
        need_gas_properties=bool(needs_local_gas),
        need_valid_mask=False,
        fallback_density_kgm3=float(phys['gas_density_kgm3']),
        fallback_mu_pas=float(phys['gas_mu_pas']),
        fallback_temperature_K=float(phys['gas_temperature_K']),
    )
    record_field_sampling_diagnostics(diagnostics, sampled_contact.samples, sampled_contact.elapsed_s)
    flow = (
        sampled_contact.samples.flow
        if sampled_contact.samples.flow is not None
        else np.zeros((indices.size, 3), dtype=np.float64)
    )
    electric_qom = (
        None
        if electric_q_over_m_particle is None
        else np.asarray(electric_q_over_m_particle, dtype=np.float64)[indices]
    )
    accel = _sample_acceleration_vectors_at(
        compiled,
        3,
        float(t_next),
        x_contact,
        electric_q_over_m=electric_qom,
        force_runtime=options.force_runtime,
        particle_diameter=np.asarray(particle_diameter, dtype=np.float64)[indices],
        particle_density=np.asarray(particle_density, dtype=np.float64)[indices],
        particle_mass=np.asarray(particle_mass, dtype=np.float64)[indices],
        dep_particle_rel_permittivity=np.asarray(dep_particle_rel_permittivity, dtype=np.float64)[indices],
        thermophoretic_coeff=np.asarray(thermophoretic_coeff, dtype=np.float64)[indices],
        velocity=v_old[:, :3],
        gas_density_kgm3=float(phys['gas_density_kgm3']),
        gas_mu_pas=float(phys['gas_mu_pas']),
        gas_temperature_K=float(phys['gas_temperature_K']),
        gas_molecular_mass_kg=float(phys['gas_molecular_mass_kg']),
    )
    rho_g = np.full(indices.size, float(phys['gas_density_kgm3']), dtype=np.float64)
    mu_g = np.full(indices.size, float(phys['gas_mu_pas']), dtype=np.float64)
    temp_g = np.full(indices.size, float(phys['gas_temperature_K']), dtype=np.float64)
    if bool(needs_local_gas) and sampled_contact.samples.gas_density is not None:
        rho_g = sampled_contact.samples.gas_density
        mu_g = sampled_contact.samples.gas_mu if sampled_contact.samples.gas_mu is not None else mu_g
        temp_g = (
            sampled_contact.samples.gas_temperature
            if sampled_contact.samples.gas_temperature is not None
            else temp_g
        )
    body_field_scale = float(phys['body_accel_scale']) * np.asarray(body_scale_particle[indices], dtype=np.float64)
    body_base = np.asarray(body_accel, dtype=np.float64)[:3][None, :] * body_field_scale[:, None]
    gravity_factor = np.ones(indices.size, dtype=np.float64)
    if bool(force_runtime.gravity_buoyancy_enabled):
        gravity_factor = 1.0 - rho_g / np.maximum(density_local, 1.0e-300)
    mass_factor = _virtual_mass_factor_array(force_runtime, rho_g, density_local)
    body_eff = (gravity_factor[:, None] * body_base + accel[:, :3]) / mass_factor[:, None]
    target = float(phys['flow_scale']) * np.asarray(flow_scale_particle[indices], dtype=np.float64)[:, None] * flow[:, :3]

    tau_stokes = np.asarray(tau_p[indices], dtype=np.float64) * float(phys['drag_tau_scale'])
    tau_stokes *= np.maximum(np.asarray(drag_scale_particle[indices], dtype=np.float64), 1.0e-6)
    tau_stokes = np.maximum(float(phys['min_tau_p_s']), tau_stokes)
    if int(drag_model_mode) == int(DRAG_MODEL_STOKES):
        tau_eff = tau_stokes
    else:
        slip = np.linalg.norm(v_old[:, :3] - target[:, :3], axis=1)
        tau_eff = np.asarray(
            [
                effective_tau_from_slip_speed(
                    float(tau_i),
                    float(slip_i),
                    float(diameter_i),
                    float(rho_i),
                    float(mu_i),
                    int(drag_model_mode),
                    float(phys['min_tau_p_s']),
                    float(density_i),
                    float(temp_i),
                    float(phys['gas_molecular_mass_kg']),
                )
                for tau_i, slip_i, diameter_i, density_i, rho_i, mu_i, temp_i in zip(
                    tau_stokes,
                    slip,
                    particle_diameter[indices],
                    density_local,
                    rho_g,
                    mu_g,
                    temp_g,
                )
            ],
            dtype=np.float64,
        )
    tau_eff = tau_eff * mass_factor

    target_normal = np.einsum('ij,ij->i', target, normal)
    body_normal = np.einsum('ij,ij->i', body_eff, normal)
    normal_tendency = (target_normal - v_normal) / np.maximum(tau_eff, 1.0e-12) + body_normal
    release_candidate = normal_tendency < -1.0e-10
    release_mask = np.zeros(indices.size, dtype=bool)
    if np.any(release_candidate):
        candidate_rows = np.flatnonzero(release_candidate)
        probe_dt = np.minimum(float(dt_step), np.maximum(tau_eff[candidate_rows], 1.0e-12))
        probe_dt = np.maximum(probe_dt, min(float(dt_step), 1.0e-9))
        x_probe = (
            x_contact[candidate_rows]
            + v_tangent[candidate_rows] * probe_dt[:, None]
            + 0.5 * normal_tendency[candidate_rows, None] * (probe_dt[:, None] ** 2) * normal[candidate_rows]
        )
        probe_inside = np.asarray([boundary_service.inside(point) for point in x_probe], dtype=bool)
        sampled_probe = timed_sample_fields_for_stage(
            compiled,
            field_plan,
            x_probe,
            float(t_next),
            spatial_dim=3,
            need_flow=False,
            need_gas_properties=False,
            need_valid_mask=True,
        )
        record_field_sampling_diagnostics(diagnostics, sampled_probe.samples, sampled_probe.elapsed_s)
        probe_status = sampled_probe.samples.valid_mask_status
        if probe_status is None:
            probe_status = _sample_valid_mask_statuses(compiled, x_probe)
        probe_clean = probe_inside & (probe_status < int(VALID_MASK_STATUS_HARD_INVALID))
        release_mask[candidate_rows[probe_clean]] = True
        rejected = int(np.count_nonzero(~probe_clean))
        if rejected:
            diagnostics['contact_release_probe_reject_count'] = int(
                diagnostics.get('contact_release_probe_reject_count', 0)
            ) + int(rejected)
    if np.any(release_mask):
        release_indices = indices[release_mask]
        state.contact_sliding[release_indices] = False
        state.contact_endpoint_stopped[release_indices] = False
        state.contact_edge_index[release_indices] = -1
        state.contact_part_id[release_indices] = 0
        state.contact_normal[release_indices] = 0.0
        state.x[release_indices] = x_contact[release_mask]
        state.v[release_indices] = v_tangent[release_mask]
        diagnostics['contact_release_count'] = int(diagnostics.get('contact_release_count', 0)) + int(release_indices.size)

    keep_mask = ~release_mask
    if not np.any(keep_mask):
        return
    endpoint_hold_mask = keep_mask & state.contact_endpoint_stopped[indices]
    if np.any(endpoint_hold_mask):
        hold_indices = indices[endpoint_hold_mask]
        state.x[hold_indices] = x_contact[endpoint_hold_mask]
        state.v[hold_indices] = 0.0
        state.x_trial[hold_indices] = state.x[hold_indices]
        state.v_trial[hold_indices] = state.v[hold_indices]
        state.x_mid_trial[hold_indices] = state.x[hold_indices]
        state.contact_edge_index[hold_indices] = tri_index[endpoint_hold_mask]
        state.contact_part_id[hold_indices] = surface.part_ids[tri_index[endpoint_hold_mask]]
        state.contact_normal[hold_indices] = normal[endpoint_hold_mask]
        diagnostics['contact_endpoint_hold_count'] = int(
            diagnostics.get('contact_endpoint_hold_count', 0)
        ) + int(hold_indices.size)

    keep_mask = keep_mask & ~state.contact_endpoint_stopped[indices]
    if not np.any(keep_mask):
        return
    keep_indices = indices[keep_mask]
    target_tangent = target[keep_mask] - np.einsum('ij,ij->i', target[keep_mask], normal[keep_mask])[:, None] * normal[keep_mask]
    body_tangent = body_eff[keep_mask] - np.einsum('ij,ij->i', body_eff[keep_mask], normal[keep_mask])[:, None] * normal[keep_mask]
    tangent_displacement, tangent_velocity = _advance_scalar_relaxation_array(
        v_tangent[keep_mask],
        target_tangent,
        body_tangent,
        tau_eff[keep_mask, None],
        float(dt_step),
    )
    x_wall_next = x_wall[keep_mask] + tangent_displacement
    plane_error = np.einsum('ij,ij->i', x_wall_next - q0[keep_mask], normal[keep_mask])
    x_wall_next = x_wall_next - plane_error[:, None] * normal[keep_mask]

    inside_triangle = np.zeros(keep_indices.size, dtype=bool)
    for row, (point, tri) in enumerate(zip(x_wall_next, triangles[keep_mask])):
        bary = _point_triangle_barycentric_3d(point, tri)
        if bary is not None and np.all(np.asarray(bary, dtype=np.float64) >= -1.0e-10):
            inside_triangle[row] = True
    x_next = x_wall_next - epsilon * normal[keep_mask]
    sampled_next = timed_sample_fields_for_stage(
        compiled,
        field_plan,
        x_next,
        float(t_next),
        spatial_dim=3,
        need_flow=False,
        need_gas_properties=False,
        need_valid_mask=True,
    )
    record_field_sampling_diagnostics(diagnostics, sampled_next.samples, sampled_next.elapsed_s)
    status = sampled_next.samples.valid_mask_status
    if status is None:
        status = _sample_valid_mask_statuses(compiled, x_next)
    inside = np.asarray([boundary_service.inside(point) for point in x_next], dtype=bool)
    reject = (status >= int(VALID_MASK_STATUS_HARD_INVALID)) | (~inside)
    endpoint_hit = ~inside_triangle
    accept = ~reject

    if np.any(reject):
        reject_indices = keep_indices[reject]
        diagnostics['contact_valid_mask_reject_count'] = int(
            diagnostics.get('contact_valid_mask_reject_count', 0)
        ) + int(reject_indices.size)
        state.x[reject_indices] = x_contact[keep_mask][reject]
        state.v[reject_indices] = 0.0
        state.x_trial[reject_indices] = state.x[reject_indices]
        state.v_trial[reject_indices] = state.v[reject_indices]
        state.x_mid_trial[reject_indices] = state.x[reject_indices]

    if not np.any(accept):
        return
    accept_indices = keep_indices[accept]
    state.x[accept_indices] = x_next[accept]
    state.v[accept_indices] = tangent_velocity[accept]
    accept_endpoint = endpoint_hit[accept]
    if np.any(accept_endpoint):
        state.x[accept_indices[accept_endpoint]] = x_contact[keep_mask][accept][accept_endpoint]
        state.v[accept_indices[accept_endpoint]] = 0.0
        state.contact_endpoint_stopped[accept_indices[accept_endpoint]] = True
        diagnostics['contact_endpoint_stop_count'] = int(diagnostics.get('contact_endpoint_stop_count', 0)) + int(
            np.count_nonzero(accept_endpoint)
        )
    state.x_trial[accept_indices] = state.x[accept_indices]
    state.v_trial[accept_indices] = state.v[accept_indices]
    state.x_mid_trial[accept_indices] = state.x[accept_indices]
    state.contact_edge_index[accept_indices] = tri_index[keep_mask][accept]
    state.contact_part_id[accept_indices] = surface.part_ids[tri_index[keep_mask][accept]]
    state.contact_normal[accept_indices] = normal[keep_mask][accept]
    diagnostics['contact_tangent_step_count'] = int(diagnostics.get('contact_tangent_step_count', 0)) + int(
        accept_indices.size
    )
    diagnostics['contact_tangent_time_total_s'] = float(
        diagnostics.get('contact_tangent_time_total_s', 0.0)
    ) + float(dt_step) * float(accept_indices.size)


def _advance_runtime_step(
    *,
    runtime,
    particles,
    state: RuntimeState,
    options: RuntimeOptions,
    compiled: CompiledRuntimeBackendLike,
    boundary_service: BoundaryService,
    spatial_dim: int,
    n_particles: int,
    mins: np.ndarray,
    maxs: np.ndarray,
    tau_p: np.ndarray,
    particle_mass: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density: np.ndarray,
    particle_id: np.ndarray,
    source_part_id: np.ndarray,
    release_time: np.ndarray,
    particle_stick_probability: np.ndarray,
    dep_particle_rel_permittivity: np.ndarray,
    thermophoretic_coeff: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
    t: float,
) -> float:
    plan = _require_solver_plan(state)
    dt = float(plan.dt)
    t_end = float(plan.t_end)
    base_save_every = int(plan.base_save_every)
    valid_mask_policy = str(plan.valid_mask_policy)
    drag_model_mode = int(plan.drag_model_mode)
    adaptive_substep_enabled = int(plan.adaptive_substep_enabled)
    adaptive_substep_tau_ratio = float(plan.adaptive_substep_tau_ratio)
    adaptive_substep_max_splits = int(plan.adaptive_substep_max_splits)
    min_remaining_dt_ratio = float(plan.min_remaining_dt_ratio)
    max_wall_hits_per_step = int(plan.max_wall_hits_per_step)
    epsilon_offset_m = float(plan.epsilon_offset_m)
    on_boundary_tol_m = float(plan.on_boundary_tol_m)
    boundary_broad_phase_enabled = bool(plan.boundary_broad_phase_enabled)

    capture_snapshots = not isinstance(state.save_positions, _DiscardList)
    capture_step_rows = (
        not isinstance(state.step_rows, _DiscardList)
        or (
            state.output_buffers is not None
            and state.output_buffers.step_summary is not None
            and bool(state.output_buffers.output.write_step_summary)
        )
    )
    dt_step = min(float(dt), float(t_end) - float(t))
    t_next = float(t) + float(dt_step)
    step_ctx = _resolve_step_loop_context(
        runtime,
        t=float(t_next),
        t_end=float(t_end),
        spatial_dim=int(spatial_dim),
        base_save_every=int(base_save_every),
        prev_step_name=state.prev_step_name,
        step_local_counter=int(state.step_local_counter),
    )
    step = step_ctx.step
    phys = dict(step_ctx.phys)
    phys['gas_density_kgm3'] = float(runtime.gas.density_kgm3)
    phys['gas_mu_pas'] = float(runtime.gas.dynamic_viscosity_Pas)
    phys['gas_temperature_K'] = float(runtime.gas.temperature)
    phys['gas_molecular_mass_kg'] = float(runtime.gas.molecular_mass_amu) * 1.66053906660e-27
    body_accel = step_ctx.body_accel
    integrator_spec = step_ctx.integrator_spec
    integrator_mode = int(integrator_spec.mode)
    save_every = int(step_ctx.save_every)
    state.step_local_counter = int(step_ctx.step_local_counter)
    state.prev_step_name = step_ctx.prev_step_name

    if state.release_cursor is None:
        raise ValueError('RuntimeState.release_cursor is required')
    _activate_released_particles(state.release_cursor, state.released, state.active, float(t_next))
    if bool(options.charge_model.enabled):
        t_section = time.perf_counter()
        state.charge_pending_dt_s += float(dt_step)
        charge_stride_dt = float(options.charge_model.update_stride) * float(dt)
        should_apply_charge = (
            state.charge_pending_dt_s >= max(charge_stride_dt, 0.0) - 1.0e-18
            or float(t_next) >= float(t_end) - 1.0e-15
        )
        if bool(should_apply_charge):
            charge_result = apply_charge_model_update(
                config=options.charge_model,
                runtime=runtime,
                spatial_dim=int(spatial_dim),
                t_eval=float(t_next),
                delta_t_s=float(state.charge_pending_dt_s),
                active_mask=state.active,
                x=state.x,
                charge=state.charge,
                particle_diameter=particle_diameter,
                plasma_background=options.plasma_background,
            )
            merge_charge_model_diagnostics(
                state.collision_diagnostics,
                options.charge_model,
                charge_result,
            )
            state.charge_pending_dt_s = 0.0
        _add_timing(state.timing_accumulator, 'charge_model_s', time.perf_counter() - t_section)
    electric_q_over_m_particle = _electric_q_over_m_particle(
        charge_model=options.charge_model,
        charge=state.charge,
        particle_mass=particle_mass,
    )
    if int(spatial_dim) == 2:
        _advance_contact_sliding_particles_2d(
            runtime=runtime,
            state=state,
            options=options,
            compiled=compiled,
            boundary_service=boundary_service,
            tau_p=tau_p,
            particle_diameter=particle_diameter,
            particle_density=particle_density,
            particle_mass=particle_mass,
            dep_particle_rel_permittivity=dep_particle_rel_permittivity,
            thermophoretic_coeff=thermophoretic_coeff,
            flow_scale_particle=flow_scale_particle,
            drag_scale_particle=drag_scale_particle,
            body_scale_particle=body_scale_particle,
            phys=phys,
            body_accel=body_accel,
            dt_step=float(dt_step),
            t_next=float(t_next),
            electric_q_over_m_particle=electric_q_over_m_particle,
        )
    elif int(spatial_dim) == 3:
        _advance_contact_sliding_particles_3d(
            runtime=runtime,
            state=state,
            options=options,
            compiled=compiled,
            boundary_service=boundary_service,
            tau_p=tau_p,
            particle_diameter=particle_diameter,
            particle_density=particle_density,
            particle_mass=particle_mass,
            dep_particle_rel_permittivity=dep_particle_rel_permittivity,
            thermophoretic_coeff=thermophoretic_coeff,
            flow_scale_particle=flow_scale_particle,
            drag_scale_particle=drag_scale_particle,
            body_scale_particle=body_scale_particle,
            phys=phys,
            body_accel=body_accel,
            dt_step=float(dt_step),
            t_next=float(t_next),
            electric_q_over_m_particle=electric_q_over_m_particle,
        )
    if state.solver_arrays is not None:
        active_idx = state.solver_arrays.active_indices()
        mobile_active_idx = state.solver_arrays.mobile_active_indices()
    else:
        active_idx = np.flatnonzero(state.active)
        mobile_active_idx = active_idx[~state.contact_sliding[active_idx]]
    mobile_active = np.zeros_like(state.active, dtype=bool)
    mobile_active[mobile_active_idx] = True
    state.valid_mask_status_flags.fill(int(VALID_MASK_STATUS_CLEAN))

    t_section = time.perf_counter()
    _advance_trial_particles(
        spatial_dim=int(spatial_dim),
        compiled=compiled,
        x=state.x,
        v=state.v,
        active=mobile_active,
        tau_p=tau_p,
        particle_diameter=particle_diameter,
        particle_mass=particle_mass,
        particle_density=particle_density,
        dep_particle_rel_permittivity=dep_particle_rel_permittivity,
        thermophoretic_coeff=thermophoretic_coeff,
        flow_scale_particle=flow_scale_particle,
        drag_scale_particle=drag_scale_particle,
        body_scale_particle=body_scale_particle,
        t=float(t_next),
        dt_step=float(dt_step),
        phys=phys,
        body_accel=body_accel,
        gas_density_kgm3=float(phys['gas_density_kgm3']),
        gas_mu_pas=float(phys['gas_mu_pas']),
        drag_model_mode=int(drag_model_mode),
        integrator_mode=int(integrator_mode),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        x_trial=state.x_trial,
        v_trial=state.v_trial,
        x_mid_trial=state.x_mid_trial,
        substep_counts=state.substep_counts,
        valid_mask_status_flags=state.valid_mask_status_flags,
        electric_q_over_m_particle=electric_q_over_m_particle,
        force_runtime=options.force_runtime,
    )
    freeflight_elapsed_s = time.perf_counter() - t_section
    _add_timing(state.timing_accumulator, 'freeflight_s', freeflight_elapsed_s)
    _add_timing(state.timing_accumulator, 'force_eval_s', freeflight_elapsed_s)

    _update_adaptive_substep_diagnostics(
        state.collision_diagnostics,
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        active=mobile_active,
        substep_counts=state.substep_counts,
    )

    t_section = time.perf_counter()
    trial_batch = _classify_trial_collisions(
        runtime,
        spatial_dim=int(spatial_dim),
        n_particles=int(n_particles),
        active=mobile_active,
        x=state.x,
        x_trial=state.x_trial,
        x_mid_trial=state.x_mid_trial,
        valid_mask_status_flags=state.valid_mask_status_flags,
        integrator_mode=int(integrator_mode),
        boundary_service=boundary_service,
        on_boundary_tol_m=float(on_boundary_tol_m),
        collision_diagnostics=state.collision_diagnostics,
        timing_accumulator=state.timing_accumulator,
        boundary_broad_phase_enabled=bool(boundary_broad_phase_enabled),
        boundary_broad_phase_debug_check=bool(plan.output.is_debug),
    )
    _add_timing(state.timing_accumulator, 'collision_classify_s', time.perf_counter() - t_section)
    colliders = trial_batch.colliders
    safe = trial_batch.safe
    prefetched_hits = trial_batch.prefetched_hits
    t_section = time.perf_counter()
    invalid_mask_stopped_count_step = _apply_valid_mask_retry_then_stop(
        state=state,
        options=options,
        compiled=compiled,
        spatial_dim=int(spatial_dim),
        integrator_mode=int(integrator_mode),
        dt_step=float(dt_step),
        t_end_step=float(t_next),
        phys=phys,
        body_accel=body_accel,
        tau_p=tau_p,
        particle_diameter=particle_diameter,
        particle_density=particle_density,
        particle_mass=particle_mass,
        dep_particle_rel_permittivity=dep_particle_rel_permittivity,
        thermophoretic_coeff=thermophoretic_coeff,
        flow_scale_particle=flow_scale_particle,
        drag_scale_particle=drag_scale_particle,
        body_scale_particle=body_scale_particle,
        electric_q_over_m_particle=electric_q_over_m_particle,
        force_runtime=options.force_runtime,
        particle_indices=safe,
    )
    _add_timing(state.timing_accumulator, 'valid_mask_retry_s', time.perf_counter() - t_section)
    safe_active = safe[mobile_active[safe]] if safe.size else safe
    if safe_active.size:
        state.x[safe_active] = state.x_trial[safe_active]
        state.v[safe_active] = state.v_trial[safe_active]

    primary_hit_fn = boundary_service.polyline_hit
    nearest_projection_fn = boundary_service.nearest_projection

    t_section = time.perf_counter()
    for particle_index_raw in colliders:
        particle_index = int(particle_index_raw)
        stage_points = _stage_points_from_trial(
            state.x_trial[particle_index],
            integrator_mode=int(integrator_mode),
            x_mid=state.x_mid_trial[particle_index] if bool(integrator_spec.uses_midpoint_stage) else None,
        )
        initial_primary_hit = prefetched_hits.get(particle_index)
        particle_result = _advance_colliding_particle(
            runtime=runtime,
            step=step,
            particles=None,
            particle_index=particle_index,
            rng=state.rng,
            t=float(t_next),
            x_start=state.x[particle_index],
            v_start=state.v[particle_index],
            dt_step=float(dt_step),
            spatial_dim=int(spatial_dim),
            compiled=compiled,
            integrator_mode=int(integrator_mode),
            base_adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            min_remaining_dt_ratio=float(min_remaining_dt_ratio),
            tau_p_i=float(tau_p[particle_index]),
            particle_diameter_i=float(particle_diameter[particle_index]),
            particle_density_i=float(particle_density[particle_index]),
            particle_mass_i=float(particle_mass[particle_index]),
            particle_id_i=int(particle_id[particle_index]),
            source_part_id_i=int(source_part_id[particle_index]),
            release_time_i=float(release_time[particle_index]),
            release_grace=plan.release_grace,
            particle_stick_probability_i=float(particle_stick_probability[particle_index]),
            dep_particle_rel_permittivity_i=float(dep_particle_rel_permittivity[particle_index]),
            thermophoretic_coeff_i=float(thermophoretic_coeff[particle_index]),
            flow_scale_particle_i=float(flow_scale_particle[particle_index]),
            drag_scale_particle_i=float(drag_scale_particle[particle_index]),
            body_scale_particle_i=float(body_scale_particle[particle_index]),
            global_flow_scale=float(phys['flow_scale']),
            global_drag_tau_scale=float(phys['drag_tau_scale']),
            global_body_accel_scale=float(phys['body_accel_scale']),
            body_accel=body_accel,
            min_tau_p_s=float(phys['min_tau_p_s']),
            gas_density_kgm3=float(phys['gas_density_kgm3']),
            gas_mu_pas=float(phys['gas_mu_pas']),
            gas_temperature_K=float(phys['gas_temperature_K']),
            gas_molecular_mass_kg=float(phys['gas_molecular_mass_kg']),
            drag_model_mode=int(drag_model_mode),
            electric_q_over_m_i=(
                None
                if electric_q_over_m_particle is None
                else float(electric_q_over_m_particle[particle_index])
            ),
            force_runtime=options.force_runtime,
            valid_mask_retry_then_stop_enabled=bool(
                str(valid_mask_policy) in {VALID_MASK_POLICY_RETRY_THEN_STOP, VALID_MASK_POLICY_STRICT_CLEAN}
            ),
            initial_x_next=state.x_trial[particle_index],
            initial_v_next=state.v_trial[particle_index],
            initial_stage_points=stage_points,
            initial_valid_mask_status=int(state.valid_mask_status_flags[particle_index]),
            initial_primary_hit=initial_primary_hit,
            initial_primary_hit_counted=False,
            inside_fn=boundary_service.inside,
            strict_inside_fn=boundary_service.inside_strict,
            primary_hit_fn=primary_hit_fn,
            nearest_projection_fn=nearest_projection_fn,
            primary_hit_counter_key=boundary_service.primary_hit_counter_key,
            collision_diagnostics=state.collision_diagnostics,
            max_hit_rows=state.max_hit_rows,
            wall_rows=state.wall_rows,
            coating_summary_rows=state.coating_summary_rows,
            wall_law_counts=state.wall_law_counts,
            wall_summary_counts=state.wall_summary_counts,
            stuck=state.stuck,
            absorbed=state.absorbed,
            escaped=state.escaped,
            active=state.active,
            max_wall_hits_per_step=int(max_wall_hits_per_step),
            epsilon_offset_m=float(epsilon_offset_m),
            on_boundary_tol_m=float(on_boundary_tol_m),
            triangle_surface_3d=boundary_service.triangle_surface_3d,
        )
        state.valid_mask_status_flags[particle_index] = np.uint8(particle_result.valid_mask_status)
        if bool(particle_result.invalid_mask_stopped):
            _mark_invalid_mask_stopped(
                state=state,
                particle_index=particle_index,
                position=particle_result.position,
                velocity=particle_result.velocity,
                update_trial_buffers=False,
                reason=str(particle_result.invalid_stop_reason),
            )
            invalid_mask_stopped_count_step += 1
        elif bool(particle_result.numerical_boundary_stopped):
            _mark_numerical_boundary_stopped(
                state=state,
                particle_index=particle_index,
                position=particle_result.position,
                velocity=particle_result.velocity,
                update_trial_buffers=False,
                reason=str(particle_result.numerical_boundary_stop_reason),
            )
        else:
            if bool(particle_result.contact_sliding):
                state.contact_sliding[particle_index] = True
                state.contact_endpoint_stopped[particle_index] = False
                if int(spatial_dim) == 2:
                    frame = contact_frame_on_boundary_edge_2d(
                        runtime,
                        particle_result.position,
                        part_id_hint=int(particle_result.contact_part_id),
                        normal_hint=particle_result.contact_normal,
                    )
                    state.contact_edge_index[particle_index] = -1 if frame is None else int(frame.edge_index)
                else:
                    state.contact_edge_index[particle_index] = int(particle_result.contact_primitive_id)
                state.contact_part_id[particle_index] = int(particle_result.contact_part_id)
                if particle_result.contact_normal is not None:
                    state.contact_normal[particle_index] = np.asarray(
                        particle_result.contact_normal,
                        dtype=np.float64,
                    )
            _commit_particle_state(
                state.x,
                state.v,
                state.active,
                state.escaped,
                particle_index=particle_index,
                position=particle_result.position,
                velocity=particle_result.velocity,
                mins=mins,
                maxs=maxs,
    )
    _add_timing(state.timing_accumulator, 'collider_resolution_s', time.perf_counter() - t_section)

    if bool(options.stochastic_motion.enabled):
        t_section = time.perf_counter()
        state.stochastic_pending_dt_s += float(dt_step)
        stochastic_stride_dt = float(options.stochastic_motion.stride) * float(dt)
        should_apply_stochastic = (
            state.stochastic_pending_dt_s >= max(stochastic_stride_dt, 0.0) - 1.0e-18
            or float(t_next) >= float(t_end) - 1.0e-15
        )
        if bool(should_apply_stochastic):
            stochastic_active = state.active & ~state.contact_sliding
            stochastic_result = apply_langevin_velocity_kick(
                config=options.stochastic_motion,
                rng=state.stochastic_rng,
                compiled=compiled,
                spatial_dim=int(spatial_dim),
                t_eval=float(t_next),
                delta_t_s=float(state.stochastic_pending_dt_s),
                active_mask=stochastic_active,
                x=state.x,
                v=state.v,
                tau_p=tau_p,
                particle_mass=particle_mass,
                particle_diameter=particle_diameter,
                particle_density=particle_density,
                flow_scale_particle=flow_scale_particle,
                drag_scale_particle=drag_scale_particle,
                global_flow_scale=float(phys['flow_scale']),
                global_drag_tau_scale=float(phys['drag_tau_scale']),
                min_tau_p_s=float(phys['min_tau_p_s']),
                gas_density_kgm3=float(phys['gas_density_kgm3']),
                gas_mu_pas=float(phys['gas_mu_pas']),
                gas_temperature_K=float(phys['gas_temperature_K']),
                gas_molecular_mass_kg=float(phys['gas_molecular_mass_kg']),
                drag_model_mode=int(drag_model_mode),
            )
            merge_stochastic_motion_diagnostics(
                state.collision_diagnostics,
                options.stochastic_motion,
                stochastic_result,
            )
            state.stochastic_pending_dt_s = 0.0
        _add_timing(state.timing_accumulator, 'stochastic_motion_s', time.perf_counter() - t_section)

    t_section = time.perf_counter()
    (
        valid_mask_violation_count_step,
        valid_mask_mixed_stencil_count_step,
        valid_mask_hard_invalid_count_step,
    ) = _update_valid_mask_diagnostics(
        state.collision_diagnostics,
        valid_mask_status_flags=state.valid_mask_status_flags,
        valid_mask_mixed_seen=state.valid_mask_mixed_seen,
        valid_mask_hard_seen=state.valid_mask_hard_seen,
    )

    if bool(capture_step_rows):
        _append_runtime_step_summary(
            state.step_rows,
            buffers=state.output_buffers,
            t=float(t_next),
            step=step,
            released=state.released,
            active=state.active,
            stuck=state.stuck,
            absorbed=state.absorbed,
            contact_sliding=state.contact_sliding,
            escaped=state.escaped,
            valid_mask_violation_count_step=int(valid_mask_violation_count_step),
            valid_mask_mixed_stencil_count_step=int(valid_mask_mixed_stencil_count_step),
            valid_mask_hard_invalid_count_step=int(valid_mask_hard_invalid_count_step),
            invalid_mask_stopped_count_step=int(invalid_mask_stopped_count_step),
            save_positions_enabled=int(capture_snapshots),
            write_wall_events_enabled=int(not isinstance(state.wall_rows, _DiscardList)),
            write_diagnostics_enabled=int(options.write_collision_diagnostics),
        )

    _record_active_count_summary(state.collision_diagnostics, np.flatnonzero(state.active))

    if bool(capture_snapshots) and state.step_local_counter % save_every == 0:
        _append_snapshot(
            state.save_positions,
            state.save_meta,
            save_index=int(state.save_index),
            t=float(t_next),
            step=step,
            position=state.x,
        )
        state.save_index += 1
    _add_timing(state.timing_accumulator, 'output_step_summary_s', time.perf_counter() - t_section)

    return float(t_next)


def _run_runtime_step_loop(
    *,
    runtime,
    particles,
    state: RuntimeState,
    options: RuntimeOptions,
    compiled: CompiledRuntimeBackendLike,
    boundary_service: BoundaryService,
    spatial_dim: int,
    n_particles: int,
    mins: np.ndarray,
    maxs: np.ndarray,
    tau_p: np.ndarray,
    particle_mass: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density: np.ndarray,
    particle_id: np.ndarray,
    source_part_id: np.ndarray,
    release_time: np.ndarray,
    particle_stick_probability: np.ndarray,
    dep_particle_rel_permittivity: np.ndarray,
    thermophoretic_coeff: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
):
    def _advance_one_step(t_current: float) -> float:
        return _advance_runtime_step(
            runtime=runtime,
            particles=particles,
            state=state,
            options=options,
            compiled=compiled,
            boundary_service=boundary_service,
            spatial_dim=int(spatial_dim),
            n_particles=int(n_particles),
            mins=mins,
            maxs=maxs,
            tau_p=tau_p,
            particle_mass=particle_mass,
            particle_diameter=particle_diameter,
            particle_density=particle_density,
            particle_id=particle_id,
            source_part_id=source_part_id,
            release_time=release_time,
            particle_stick_probability=particle_stick_probability,
            dep_particle_rel_permittivity=dep_particle_rel_permittivity,
            thermophoretic_coeff=thermophoretic_coeff,
            flow_scale_particle=flow_scale_particle,
            drag_scale_particle=drag_scale_particle,
            body_scale_particle=body_scale_particle,
            t=float(t_current),
        )

    plan = _require_solver_plan(state)
    t_end = float(plan.t_end)
    loop_t0 = time.perf_counter()
    t = 0.0
    step_count = 0
    while t < float(t_end) - 1.0e-15:
        t_previous = float(t)
        t_next = float(_advance_one_step(t_previous))
        if not np.isfinite(t_next) or t_next <= t_previous:
            raise RuntimeError(
                'solver step loop did not advance time '
                f'(t={t_previous}, next_t={t_next}, dt={float(plan.dt)})'
            )
        t = t_next
        step_count += 1
    return StepLoopResult(
        t=float(t),
        step_count=int(step_count),
        elapsed_s=float(time.perf_counter() - loop_t0),
    )


def _build_runtime_output_payload(
    prepared: PreparedRuntime,
    spatial_dim: int,
    *,
    plan: SolverPlan,
    arrays: SolverArrays,
    buffers: RuntimeBuffers,
    capture_outputs: bool,
) -> RuntimeOutputPayload:
    runtime = prepared.runtime
    particles = runtime.particles
    if particles is None:
        raise ValueError('Simulation requires particles')
    setup_t0 = time.perf_counter()
    config_payload = runtime.config_payload if isinstance(runtime.config_payload, Mapping) else {}
    options = runtime_options_from_plan(
        plan=plan,
        config_payload=config_payload,
        force_catalog=runtime.force_catalog,
    )

    n_particles = int(arrays.static.count)
    mins, maxs = runtime_bounds(runtime)
    compiled = _compile_runtime_arrays(
        runtime,
        spatial_dim,
        particles=particles,
        dynamic_electric=bool(options.charge_model.enabled),
        enable_electric=bool(options.force_catalog.enabled('electric')) if options.force_catalog is not None else True,
        force_runtime=options.force_runtime,
    )
    validate_charge_model_support(
        options.charge_model,
        runtime,
        compiled,
        int(spatial_dim),
        options.plasma_background,
    )
    gas_mu = float(runtime.gas.dynamic_viscosity_Pas)
    triangle_surface_3d = _prepare_triangle_surface(runtime, spatial_dim)
    release_time = arrays.static.release_time_s
    base_phys = resolve_step_physics(runtime.physics_catalog, None)
    min_tau_p_s = float(base_phys['min_tau_p_s'])
    base_integrator_name = get_integrator_spec(str(base_phys.get('integrator', 'drag_relaxation'))).name
    particle_mass = arrays.static.mass_kg
    particle_diameter = arrays.static.diameter_m
    particle_density = arrays.static.density_kgm3
    particle_id = arrays.static.particle_id
    source_part_id = arrays.static.source_part_id
    particle_stick_probability = arrays.static.stick_probability
    dep_particle_rel_permittivity = arrays.static.dep_particle_rel_permittivity
    thermophoretic_coeff = arrays.static.thermophoretic_coeff
    tau_p = np.asarray(
        [_particle_tau_p(particle_diameter[i], particle_density[i], gas_mu, min_tau_p_s) for i in range(n_particles)],
        dtype=np.float64,
    )
    flow_scale_particle = arrays.static.flow_scale
    drag_scale_particle = arrays.static.drag_tau_scale
    body_scale_particle = arrays.static.body_accel_scale
    output_options = options.output_options
    capture_snapshots = bool(capture_outputs) and output_options.capture_positions()
    state = _initialize_runtime_state(
        particles=particles,
        plan=plan,
        arrays=arrays,
        buffers=buffers,
        options=options,
        spatial_dim=int(spatial_dim),
        capture_outputs=bool(capture_outputs),
    )
    state.collision_diagnostics['boundary_broad_phase_enabled'] = int(bool(plan.boundary_broad_phase_enabled))
    state.collision_diagnostics['source_surface_release_grace_enabled'] = int(bool(plan.release_grace.enabled))
    state.collision_diagnostics['source_surface_release_grace_time_s'] = float(plan.release_grace.grace_time_s)
    state.collision_diagnostics['source_surface_release_grace_clearance_m'] = float(plan.release_grace.clearance_m)
    state.collision_diagnostics['source_surface_release_grace_min_outward_normal_speed_mps'] = float(
        plan.release_grace.min_outward_normal_speed_mps
    )
    state.collision_diagnostics['output_mode'] = str(plan.output.mode)
    state.collision_diagnostics['output_minimal_enabled'] = int(bool(plan.output.is_minimal))
    state.collision_diagnostics['output_debug_enabled'] = int(bool(plan.output.is_debug))
    state.collision_diagnostics['field_sampling_s'] = 0.0
    state.collision_diagnostics['field_sample_point_count'] = 0
    state.collision_diagnostics['field_sample_call_count'] = 0
    state.collision_diagnostics['force_eval_s'] = 0.0
    state.collision_diagnostics['acceleration_source'] = str(getattr(compiled, 'acceleration_source', 'none'))
    state.collision_diagnostics['acceleration_quantity_names'] = list(
        getattr(compiled, 'acceleration_quantity_names', ())
    )
    state.collision_diagnostics['electric_field_names'] = list(getattr(compiled, 'electric_field_names', ()))
    state.collision_diagnostics['electric_q_over_m_Ckg'] = float(getattr(compiled, 'electric_q_over_m_Ckg', 0.0))
    state.collision_diagnostics['drag_gas_properties'] = dict(
        compiled_gas_property_report(
            compiled,
            fallback_density_kgm3=float(base_phys.get('gas_density_kgm3', runtime.gas.density_kgm3)),
            fallback_mu_pas=float(base_phys.get('gas_mu_pas', runtime.gas.dynamic_viscosity_Pas)),
            fallback_temperature_K=float(base_phys.get('gas_temperature_K', runtime.gas.temperature)),
            drag_model_name=str(plan.drag_model_name),
        )
    )
    state.collision_diagnostics['field_backend_diagnostics'] = {
        'backend_kind': str(getattr(compiled, 'backend_kind', '')),
        'gas_density_source': str(getattr(compiled, 'gas_density_source', 'scalar_fallback')),
        'gas_mu_source': str(getattr(compiled, 'gas_mu_source', 'scalar_fallback')),
        'gas_temperature_source': str(getattr(compiled, 'gas_temperature_source', 'scalar_fallback')),
        'triangle_gradient_sources': dict(getattr(compiled, 'triangle_gradient_sources', {})),
    }
    state.collision_diagnostics['collision_boundary_geometry'] = 'linear_segment_or_triangle_boundary'
    state.collision_diagnostics['contact_tangent_model'] = 'custom_relaxation_contact_sliding'
    state.collision_diagnostics['force_catalog'] = force_catalog_summary(options.force_catalog)
    state.collision_diagnostics['force_runtime'] = force_runtime_parameters_summary(options.force_runtime)
    state.collision_diagnostics['stochastic_motion'] = stochastic_motion_report(options.stochastic_motion)
    state.collision_diagnostics['plasma_background'] = plasma_background_report(options.plasma_background)
    state.collision_diagnostics['charge_model'] = charge_model_report(
        options.charge_model,
        options.plasma_background,
    )
    init_step = _current_step(runtime, 0.0, plan.t_end)
    if bool(capture_snapshots):
        _append_snapshot(state.save_positions, state.save_meta, save_index=0, t=0.0, step=init_step, position=state.x)

    boundary_service = build_boundary_service(
        runtime,
        spatial_dim=int(spatial_dim),
        on_boundary_tol_m=float(plan.on_boundary_tol_m),
        triangle_surface_3d=triangle_surface_3d,
    )

    loop_setup_done_s = time.perf_counter()
    loop_result = _run_runtime_step_loop(
        runtime=runtime,
        particles=particles,
        state=state,
        options=options,
        compiled=compiled,
        boundary_service=boundary_service,
        spatial_dim=int(spatial_dim),
        n_particles=int(n_particles),
        mins=mins,
        maxs=maxs,
        tau_p=tau_p,
        particle_mass=particle_mass,
        particle_diameter=particle_diameter,
        particle_density=particle_density,
        particle_id=particle_id,
        source_part_id=source_part_id,
        release_time=release_time,
        particle_stick_probability=particle_stick_probability,
        dep_particle_rel_permittivity=dep_particle_rel_permittivity,
        thermophoretic_coeff=thermophoretic_coeff,
        flow_scale_particle=flow_scale_particle,
        drag_scale_particle=drag_scale_particle,
        body_scale_particle=body_scale_particle,
    )
    t = float(loop_result.t)
    loop_s = float(loop_result.elapsed_s)
    step_count = int(loop_result.step_count)
    state.collision_diagnostics['solver_step_count'] = int(step_count)
    state.collision_diagnostics['released_count_final'] = int(np.count_nonzero(state.released))
    if state.release_cursor is not None:
        state.collision_diagnostics['release_cursor_position_final'] = int(state.release_cursor.position)
        state.collision_diagnostics['release_cursor_done'] = int(bool(state.release_cursor.done))
    state.collision_diagnostics.setdefault('active_count_samples', 0)
    state.collision_diagnostics.setdefault('active_count_mean', 0.0)
    state.collision_diagnostics.setdefault('active_count_max', int(np.count_nonzero(state.active)))
    final_step = _current_step(runtime, t, plan.t_end)
    if bool(capture_snapshots) and (
        not state.save_meta or abs(float(state.save_meta[-1]['time_s']) - t) > 1e-12
    ):
        _append_snapshot(
            state.save_positions,
            state.save_meta,
            save_index=int(state.save_index),
            t=float(t),
            step=final_step,
            position=state.x,
        )

    step_rows = (
        state.output_buffers.step_summary.as_runtime_step_rows()
        if (
            state.output_buffers is not None
            and state.output_buffers.step_summary is not None
            and bool(state.output_buffers.output.write_step_summary)
        )
        else state.step_rows
    )
    positions, assembly_s = _assemble_saved_positions(
        state.save_positions,
        capture_snapshots=bool(capture_snapshots),
        n_particles=int(n_particles),
        spatial_dim=int(spatial_dim),
    )
    core_arrays = (
        state.x,
        state.v,
        state.released,
        state.active,
        state.stuck,
        state.absorbed,
        state.escaped,
        state.invalid_mask_stopped,
        state.numerical_boundary_stopped,
        state.invalid_stop_reason_code,
        state.x_trial,
        state.v_trial,
        state.x_mid_trial,
        state.substep_counts,
        state.valid_mask_status_flags,
        state.valid_mask_mixed_seen,
        state.valid_mask_hard_seen,
        state.charge,
        tau_p,
        particle_diameter,
        flow_scale_particle,
        drag_scale_particle,
        body_scale_particle,
        release_time,
    )
    core_array_bytes = _numpy_array_bytes(core_arrays)
    compiled_field_array_bytes = _compiled_backend_array_bytes(compiled)
    finalize_charge_model_diagnostics(
        state.collision_diagnostics,
        options.charge_model,
        state.charge,
    )
    state.collision_diagnostics['electric_q_over_m_particle_stats'] = _finite_q_over_m_summary(
        state.charge,
        particle_mass,
    )
    state.collision_diagnostics['force_eval_s'] = float(state.timing_accumulator.get('force_eval_s', 0.0))
    if state.output_buffers is not None:
        state.collision_diagnostics['output_buffers'] = dict(state.output_buffers.summary())
    timing_s = {
        'setup_s': float(loop_setup_done_s - setup_t0),
        'step_loop_s': float(loop_s),
        'positions_assembly_s': float(assembly_s),
        'field_sampling_s': float(state.collision_diagnostics.get('field_sampling_s', 0.0)),
        'solver_core_s': float(time.perf_counter() - setup_t0),
    }
    for key, value in sorted(state.timing_accumulator.items()):
        timing_s[str(key)] = float(value)
    memory_estimate_bytes = {
        'core_array_bytes': int(core_array_bytes),
        'compiled_field_array_bytes': int(compiled_field_array_bytes),
        'positions_array_bytes': int(positions.nbytes),
        'estimated_numpy_bytes': int(core_array_bytes + compiled_field_array_bytes + int(positions.nbytes)),
    }
    return RuntimeOutputPayload(
        prepared=prepared,
        spatial_dim=int(spatial_dim),
        particles=particles,
        release_time=release_time,
        positions=positions,
        save_meta=state.save_meta,
        final_position=state.x,
        final_velocity=state.v,
        final_charge=state.charge,
        released=state.released,
        active=state.active,
        stuck=state.stuck,
        absorbed=state.absorbed,
        contact_sliding=state.contact_sliding,
        contact_endpoint_stopped=state.contact_endpoint_stopped,
        contact_edge_index=state.contact_edge_index,
        contact_part_id=state.contact_part_id,
        contact_normal=state.contact_normal,
        escaped=state.escaped,
        invalid_mask_stopped=state.invalid_mask_stopped,
        numerical_boundary_stopped=state.numerical_boundary_stopped,
        invalid_stop_reason_code=state.invalid_stop_reason_code,
        final_step_name=final_step.step_name,
        final_segment_name=_step_segment_name(final_step),
        wall_rows=state.wall_rows,
        coating_summary_rows=(
            state.coating_summary_rows.rows()
            if isinstance(state.coating_summary_rows, CoatingSummaryAccumulator)
            else []
        ),
        wall_law_counts=state.wall_law_counts,
        wall_summary_counts=state.wall_summary_counts,
        max_hit_rows=state.max_hit_rows,
        step_rows=step_rows,
        collision_diagnostics=state.collision_diagnostics,
        base_integrator_name=str(base_integrator_name),
        write_collision_diagnostics=int(options.write_collision_diagnostics),
        max_wall_hits_per_step=int(plan.max_wall_hits_per_step),
        min_remaining_dt_ratio=float(plan.min_remaining_dt_ratio),
        on_boundary_tol_m=float(plan.on_boundary_tol_m),
        epsilon_offset_m=float(plan.epsilon_offset_m),
        adaptive_substep_enabled=int(plan.adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(plan.adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(plan.adaptive_substep_max_splits),
        plot_limit=int(plan.plot_limit),
        valid_mask_policy=str(plan.valid_mask_policy),
        output_options=output_options,
        drag_model=str(plan.drag_model_name),
        timing_s=timing_s,
        memory_estimate_bytes=memory_estimate_bytes,
    )


def run_prepared_runtime(
    prepared: PreparedRuntime,
    output_dir: Optional[Path],
    spatial_dim: int,
) -> Dict[str, object]:
    capture_outputs = output_dir is not None
    plan = build_solver_plan(prepared, spatial_dim=int(spatial_dim))
    arrays = initialize_solver_arrays(prepared, plan)
    buffers = _initialize_output_buffers(plan)
    payload = _build_runtime_output_payload(
        prepared,
        spatial_dim=int(spatial_dim),
        plan=plan,
        arrays=arrays,
        buffers=buffers,
        capture_outputs=bool(capture_outputs),
    )
    if not bool(capture_outputs):
        return build_runtime_report(payload, outputs_written=False)
    return write_runtime_outputs(payload, Path(output_dir))


__all__ = (
    'run_prepared_runtime',
)
