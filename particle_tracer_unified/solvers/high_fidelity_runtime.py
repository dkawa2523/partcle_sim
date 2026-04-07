from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..core.boundary_service import BoundaryService, build_boundary_service, runtime_bounds
from ..core.catalogs import resolve_step_physics
from ..core.datamodel import PreparedRuntime, ProcessStepRow
from ..core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from ..core.geometry3d import TriangleSurface3D, build_triangle_surface
from ..core.integrator_registry import IntegratorSpec, get_integrator_spec
from .compiled_field_backend import CompiledRuntimeBackendLike, compile_runtime_backend as _compile_runtime_arrays
from .high_fidelity_collision import _advance_colliding_particle, _classify_trial_collisions, _step_segment_name
from .high_fidelity_freeflight import (
    _advance_trial_particles,
    _stage_points_from_trial,
    resolve_valid_mask_prefix,
)
from .runtime_outputs import RuntimeOutputPayload, build_runtime_report, write_runtime_outputs
from .valid_mask_retry import resolve_valid_mask_retry_then_stop

VALID_MASK_POLICY_DIAGNOSTIC = 'diagnostic'
VALID_MASK_POLICY_RETRY_THEN_STOP = 'retry_then_stop'


class _DiscardList(list):
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
class SolverRuntimeOptions:
    dt: float
    t_end: float
    base_save_every: int
    plot_limit: int
    rng_seed: int
    max_wall_hits_per_step: int
    max_hits_retry_splits: int
    max_hits_retry_local_adaptive_enabled: int
    min_remaining_dt_ratio: float
    adaptive_substep_enabled: int
    adaptive_substep_tau_ratio: float
    adaptive_substep_max_splits: int
    epsilon_offset_m: float
    on_boundary_tol_m: float
    write_collision_diagnostics: int
    valid_mask_policy: str


@dataclass
class RuntimeState:
    x: np.ndarray
    v: np.ndarray
    released: np.ndarray
    active: np.ndarray
    stuck: np.ndarray
    absorbed: np.ndarray
    escaped: np.ndarray
    invalid_mask_stopped: np.ndarray
    save_positions: List[np.ndarray]
    save_meta: List[Dict[str, object]]
    wall_rows: List[Dict[str, object]]
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
    extension_band_sample_flags: np.ndarray
    valid_mask_mixed_seen: np.ndarray
    valid_mask_hard_seen: np.ndarray
    extension_band_seen: np.ndarray


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


def _initial_collision_diagnostics() -> Dict[str, object]:
    return {
        'primary_hit_count': 0,
        'edge_hit_count': 0,
        'triangle_hit_count': 0,
        'bisection_fallback_count': 0,
        'nearest_projection_fallback_count': 0,
        'on_boundary_promoted_inside_count': 0,
        'unresolved_crossing_count': 0,
        'multi_hit_events_count': 0,
        'max_hits_reached_count': 0,
        'collision_reintegrated_segments_count': 0,
        'adaptive_substep_segments_count': 0,
        'adaptive_substep_trigger_count': 0,
        'max_hits_retry_count': 0,
        'max_hits_retry_exhausted_count': 0,
        'dropped_remaining_dt_total_s': 0.0,
        'etd2_polyline_checks_count': 0,
        'etd2_midpoint_outside_count': 0,
        'etd2_polyline_hit_count': 0,
        'etd2_polyline_fallback_count': 0,
        'valid_mask_violation_count': 0,
        'valid_mask_violation_particle_count': 0,
        'valid_mask_mixed_stencil_count': 0,
        'valid_mask_mixed_stencil_particle_count': 0,
        'valid_mask_hard_invalid_count': 0,
        'valid_mask_hard_invalid_particle_count': 0,
        'extension_band_sample_count': 0,
        'extension_band_sample_particle_count': 0,
        'invalid_mask_retry_count': 0,
        'invalid_mask_retry_exhausted_count': 0,
        'invalid_mask_stopped_count': 0,
    }


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
    extension_band_sample_flags: np.ndarray,
    valid_mask_mixed_seen: np.ndarray,
    valid_mask_hard_seen: np.ndarray,
    extension_band_seen: np.ndarray,
) -> Tuple[int, int, int, int]:
    statuses = np.asarray(valid_mask_status_flags, dtype=np.uint8)
    extension_step_mask = np.asarray(extension_band_sample_flags, dtype=bool)
    mixed_step_mask = statuses == int(VALID_MASK_STATUS_MIXED_STENCIL)
    hard_step_mask = statuses == int(VALID_MASK_STATUS_HARD_INVALID)
    mixed_count_step = int(np.count_nonzero(mixed_step_mask))
    hard_count_step = int(np.count_nonzero(hard_step_mask))
    violation_count_step = int(mixed_count_step + hard_count_step)
    extension_band_count_step = int(np.count_nonzero(extension_step_mask))
    valid_mask_mixed_seen |= mixed_step_mask
    valid_mask_hard_seen |= hard_step_mask
    extension_band_seen |= extension_step_mask
    collision_diagnostics['valid_mask_violation_count'] += int(violation_count_step)
    collision_diagnostics['valid_mask_violation_particle_count'] = int(
        np.count_nonzero(valid_mask_mixed_seen | valid_mask_hard_seen)
    )
    collision_diagnostics['valid_mask_mixed_stencil_count'] += int(mixed_count_step)
    collision_diagnostics['valid_mask_mixed_stencil_particle_count'] = int(np.count_nonzero(valid_mask_mixed_seen))
    collision_diagnostics['valid_mask_hard_invalid_count'] += int(hard_count_step)
    collision_diagnostics['valid_mask_hard_invalid_particle_count'] = int(np.count_nonzero(valid_mask_hard_seen))
    collision_diagnostics['extension_band_sample_count'] += int(extension_band_count_step)
    collision_diagnostics['extension_band_sample_particle_count'] = int(np.count_nonzero(extension_band_seen))
    return int(violation_count_step), int(mixed_count_step), int(hard_count_step), int(extension_band_count_step)


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
    t: float,
    step: ProcessStepRow,
    released: np.ndarray,
    active: np.ndarray,
    stuck: np.ndarray,
    absorbed: np.ndarray,
    escaped: np.ndarray,
    valid_mask_violation_count_step: int,
    valid_mask_mixed_stencil_count_step: int,
    valid_mask_hard_invalid_count_step: int,
    extension_band_sample_count_step: int,
    invalid_mask_stopped_count_step: int,
) -> None:
    step_rows.append(
        {
            'time_s': float(t),
            'step_name': step.step_name,
            'segment_name': _step_segment_name(step),
            'released_count': int(released.sum()),
            'active_count': int(active.sum()),
            'stuck_count': int(stuck.sum()),
            'absorbed_count': int(absorbed.sum()),
            'escaped_count': int(escaped.sum()),
            'save_positions_enabled': int(step.output_save_positions),
            'write_wall_events_enabled': int(step.output_write_wall_events),
            'write_diagnostics_enabled': int(step.output_write_diagnostics),
            'valid_mask_violation_count_step': int(valid_mask_violation_count_step),
            'valid_mask_mixed_stencil_count_step': int(valid_mask_mixed_stencil_count_step),
            'valid_mask_hard_invalid_count_step': int(valid_mask_hard_invalid_count_step),
            'extension_band_sample_count_step': int(extension_band_sample_count_step),
            'invalid_mask_stopped_count_step': int(invalid_mask_stopped_count_step),
        }
    )


def _activate_released_particles(released: np.ndarray, active: np.ndarray, release_time: np.ndarray, t: float) -> None:
    newly = (~released) & np.isfinite(release_time) & (release_time <= float(t) + 1e-15)
    released |= newly
    active |= newly


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
    save_every = int(step.output_save_every_override) if int(step.output_save_every_override) > 0 else int(base_save_every)
    return StepLoopContext(
        step=step,
        phys=phys,
        body_accel=_body_acceleration_vector(phys, spatial_dim),
        integrator_spec=get_integrator_spec(str(phys.get('integrator', 'drag_relaxation'))),
        save_every=int(save_every),
        step_local_counter=int(next_step_local_counter),
        prev_step_name=str(next_prev_step_name),
    )


def _resolve_solver_runtime_options(config_payload: Mapping[str, object]) -> SolverRuntimeOptions:
    config = config_payload if isinstance(config_payload, Mapping) else {}
    solver_cfg = config.get('solver', {}) if isinstance(config.get('solver', {}), Mapping) else {}
    output_cfg = config.get('output', {}) if isinstance(config.get('output', {}), Mapping) else {}
    wall_cfg = config.get('wall', {}) if isinstance(config.get('wall', {}), Mapping) else {}

    dt = float(solver_cfg.get('dt', 1e-3))
    t_end = float(solver_cfg.get('t_end', 0.1))
    if dt <= 0.0:
        raise ValueError('solver.dt must be > 0')
    if t_end < 0.0:
        raise ValueError('solver.t_end must be >= 0')

    max_hits_retry_splits_raw = solver_cfg.get('max_hits_retry_splits', 0)
    try:
        max_hits_retry_splits = int(max_hits_retry_splits_raw)
    except Exception:
        max_hits_retry_splits = 0
    max_hits_retry_splits = int(max(0, max_hits_retry_splits))

    min_remaining_dt_ratio = float(solver_cfg.get('min_remaining_dt_ratio', 0.05))
    if not np.isfinite(min_remaining_dt_ratio):
        min_remaining_dt_ratio = 0.05
    min_remaining_dt_ratio = float(np.clip(min_remaining_dt_ratio, 0.0, 1.0))

    adaptive_substep_tau_ratio = float(solver_cfg.get('adaptive_substep_tau_ratio', 0.5))
    if not np.isfinite(adaptive_substep_tau_ratio):
        adaptive_substep_tau_ratio = 0.5
    adaptive_substep_tau_ratio = max(adaptive_substep_tau_ratio, 1e-8)

    epsilon_offset_m = float(wall_cfg.get('epsilon_offset_m', 1e-6))
    on_boundary_tol_raw = solver_cfg.get('on_boundary_tol_m', np.nan)
    try:
        on_boundary_tol_val = float(on_boundary_tol_raw)
    except Exception:
        on_boundary_tol_val = np.nan
    if np.isfinite(on_boundary_tol_val):
        on_boundary_tol_m = max(on_boundary_tol_val, 0.0)
    else:
        on_boundary_tol_m = max(2.0 * epsilon_offset_m, 5.0e-7)
    valid_mask_policy = str(solver_cfg.get('valid_mask_policy', VALID_MASK_POLICY_DIAGNOSTIC)).strip().lower()
    if valid_mask_policy not in {VALID_MASK_POLICY_DIAGNOSTIC, VALID_MASK_POLICY_RETRY_THEN_STOP}:
        raise ValueError(
            'solver.valid_mask_policy must be one of '
            f"'{VALID_MASK_POLICY_DIAGNOSTIC}' or '{VALID_MASK_POLICY_RETRY_THEN_STOP}'"
        )

    return SolverRuntimeOptions(
        dt=float(dt),
        t_end=float(t_end),
        base_save_every=int(max(1, solver_cfg.get('save_every', 10))),
        plot_limit=int(solver_cfg.get('plot_particle_limit', 32)),
        rng_seed=int(solver_cfg.get('seed', 12345)),
        max_wall_hits_per_step=int(max(1, solver_cfg.get('max_wall_hits_per_step', 5))),
        max_hits_retry_splits=int(max_hits_retry_splits),
        max_hits_retry_local_adaptive_enabled=int(bool(solver_cfg.get('max_hits_retry_local_adaptive_enabled', 0))),
        min_remaining_dt_ratio=float(min_remaining_dt_ratio),
        adaptive_substep_enabled=int(bool(solver_cfg.get('adaptive_substep_enabled', 0))),
        adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(max(0, solver_cfg.get('adaptive_substep_max_splits', 4))),
        epsilon_offset_m=float(epsilon_offset_m),
        on_boundary_tol_m=float(on_boundary_tol_m),
        write_collision_diagnostics=int(output_cfg.get('write_collision_diagnostics', 1)),
        valid_mask_policy=str(valid_mask_policy),
    )


def _apply_valid_mask_retry_then_stop(
    *,
    state: RuntimeState,
    options: SolverRuntimeOptions,
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    integrator_mode: int,
    dt_step: float,
    t_end_step: float,
    phys: Mapping[str, object],
    body_accel: np.ndarray,
    tau_p: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
) -> int:
    if str(options.valid_mask_policy) != VALID_MASK_POLICY_RETRY_THEN_STOP:
        return 0
    violating = np.flatnonzero(state.active & (state.valid_mask_status_flags == int(VALID_MASK_STATUS_HARD_INVALID)))
    if violating.size == 0:
        return 0

    stopped_count_step = 0
    for particle_index_raw in violating:
        particle_index = int(particle_index_raw)
        x_start = np.asarray(state.x[particle_index], dtype=np.float64).copy()
        v_start = np.asarray(state.v[particle_index], dtype=np.float64).copy()
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
            adaptive_substep_enabled=int(options.adaptive_substep_enabled),
            adaptive_substep_tau_ratio=float(options.adaptive_substep_tau_ratio),
            adaptive_substep_max_splits=int(options.adaptive_substep_max_splits),
            tau_p_i=float(tau_p[particle_index]),
            flow_scale_particle_i=float(flow_scale_particle[particle_index]),
            drag_scale_particle_i=float(drag_scale_particle[particle_index]),
            body_scale_particle_i=float(body_scale_particle[particle_index]),
            global_flow_scale=float(phys['flow_scale']),
            global_drag_tau_scale=float(phys['drag_tau_scale']),
            global_body_accel_scale=float(phys['body_accel_scale']),
            body_accel=body_accel,
            min_tau_p_s=float(phys['min_tau_p_s']),
        )

        _mark_invalid_mask_stopped(
            state=state,
            particle_index=particle_index,
            position=resolution.position,
            velocity=resolution.velocity,
            update_trial_buffers=True,
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
    state.escaped[index] = False
    if not bool(state.invalid_mask_stopped[index]):
        state.invalid_mask_stopped[index] = True
        state.collision_diagnostics['invalid_mask_stopped_count'] += 1


def _advance_runtime_step(
    *,
    runtime,
    particles,
    state: RuntimeState,
    options: SolverRuntimeOptions,
    compiled: CompiledRuntimeBackendLike,
    boundary_service: BoundaryService,
    spatial_dim: int,
    n_particles: int,
    mins: np.ndarray,
    maxs: np.ndarray,
    tau_p: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
    release_time: np.ndarray,
    t: float,
) -> float:
    capture_snapshots = not isinstance(state.save_positions, _DiscardList)
    capture_step_rows = not isinstance(state.step_rows, _DiscardList)
    dt_step = min(float(options.dt), float(options.t_end) - float(t))
    t_next = float(t) + float(dt_step)
    step_ctx = _resolve_step_loop_context(
        runtime,
        t=float(t_next),
        t_end=float(options.t_end),
        spatial_dim=int(spatial_dim),
        base_save_every=int(options.base_save_every),
        prev_step_name=state.prev_step_name,
        step_local_counter=int(state.step_local_counter),
    )
    step = step_ctx.step
    phys = step_ctx.phys
    body_accel = step_ctx.body_accel
    integrator_spec = step_ctx.integrator_spec
    integrator_mode = int(integrator_spec.mode)
    save_every = int(step_ctx.save_every)
    state.step_local_counter = int(step_ctx.step_local_counter)
    state.prev_step_name = step_ctx.prev_step_name

    def _segment_adaptive_enabled_for_retry(retry_splits_used: int) -> int:
        if int(options.adaptive_substep_enabled) != 0:
            return 1
        if int(options.max_hits_retry_local_adaptive_enabled) != 0 and int(retry_splits_used) > 0:
            return 1
        return 0

    _activate_released_particles(state.released, state.active, release_time, float(t_next))
    state.valid_mask_status_flags.fill(int(VALID_MASK_STATUS_CLEAN))
    state.extension_band_sample_flags.fill(False)

    _advance_trial_particles(
        spatial_dim=int(spatial_dim),
        compiled=compiled,
        x=state.x,
        v=state.v,
        active=state.active,
        tau_p=tau_p,
        flow_scale_particle=flow_scale_particle,
        drag_scale_particle=drag_scale_particle,
        body_scale_particle=body_scale_particle,
        t=float(t_next),
        dt_step=float(dt_step),
        phys=phys,
        body_accel=body_accel,
        integrator_mode=int(integrator_mode),
        adaptive_substep_enabled=int(options.adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(options.adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(options.adaptive_substep_max_splits),
        x_trial=state.x_trial,
        v_trial=state.v_trial,
        x_mid_trial=state.x_mid_trial,
        substep_counts=state.substep_counts,
        valid_mask_status_flags=state.valid_mask_status_flags,
        extension_band_sample_flags=state.extension_band_sample_flags,
    )

    _update_adaptive_substep_diagnostics(
        state.collision_diagnostics,
        adaptive_substep_enabled=int(options.adaptive_substep_enabled),
        active=state.active,
        substep_counts=state.substep_counts,
    )

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
        flow_scale_particle=flow_scale_particle,
        drag_scale_particle=drag_scale_particle,
        body_scale_particle=body_scale_particle,
    )

    trial_batch = _classify_trial_collisions(
        runtime,
        spatial_dim=int(spatial_dim),
        n_particles=int(n_particles),
        active=state.active,
        x=state.x,
        x_trial=state.x_trial,
        x_mid_trial=state.x_mid_trial,
        integrator_mode=int(integrator_mode),
        boundary_service=boundary_service,
        on_boundary_tol_m=float(options.on_boundary_tol_m),
        collision_diagnostics=state.collision_diagnostics,
    )
    colliders = trial_batch.colliders
    safe = trial_batch.safe
    prefetched_hits = trial_batch.prefetched_hits
    if safe.size:
        state.x[safe] = state.x_trial[safe]
        state.v[safe] = state.v_trial[safe]

    primary_hit_fn = boundary_service.polyline_hit
    nearest_projection_fn = boundary_service.nearest_projection

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
            particles=particles,
            particle_index=particle_index,
            rng=state.rng,
            t=float(t_next),
            x_start=state.x[particle_index],
            v_start=state.v[particle_index],
            dt_step=float(dt_step),
            spatial_dim=int(spatial_dim),
            compiled=compiled,
            integrator_mode=int(integrator_mode),
            base_adaptive_substep_enabled=int(options.adaptive_substep_enabled),
            adaptive_substep_tau_ratio=float(options.adaptive_substep_tau_ratio),
            adaptive_substep_max_splits=int(options.adaptive_substep_max_splits),
            min_remaining_dt_ratio=float(options.min_remaining_dt_ratio),
            segment_adaptive_enabled_for_retry=_segment_adaptive_enabled_for_retry,
            tau_p_i=float(tau_p[particle_index]),
            flow_scale_particle_i=float(flow_scale_particle[particle_index]),
            drag_scale_particle_i=float(drag_scale_particle[particle_index]),
            body_scale_particle_i=float(body_scale_particle[particle_index]),
            global_flow_scale=float(phys['flow_scale']),
            global_drag_tau_scale=float(phys['drag_tau_scale']),
            global_body_accel_scale=float(phys['body_accel_scale']),
            body_accel=body_accel,
            min_tau_p_s=float(phys['min_tau_p_s']),
            valid_mask_retry_then_stop_enabled=bool(
                str(options.valid_mask_policy) == VALID_MASK_POLICY_RETRY_THEN_STOP
            ),
            initial_x_next=state.x_trial[particle_index],
            initial_v_next=state.v_trial[particle_index],
            initial_stage_points=stage_points,
            initial_valid_mask_status=int(state.valid_mask_status_flags[particle_index]),
            initial_extension_band_sampled=bool(state.extension_band_sample_flags[particle_index]),
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
            wall_law_counts=state.wall_law_counts,
            wall_summary_counts=state.wall_summary_counts,
            stuck=state.stuck,
            absorbed=state.absorbed,
            active=state.active,
            max_wall_hits_per_step=int(options.max_wall_hits_per_step),
            max_hits_retry_splits=int(options.max_hits_retry_splits),
            epsilon_offset_m=float(options.epsilon_offset_m),
            on_boundary_tol_m=float(options.on_boundary_tol_m),
            triangle_surface_3d=boundary_service.triangle_surface_3d,
        )
        state.valid_mask_status_flags[particle_index] = np.uint8(particle_result.valid_mask_status)
        state.extension_band_sample_flags[particle_index] = bool(particle_result.extension_band_sampled)
        if bool(particle_result.invalid_mask_stopped):
            _mark_invalid_mask_stopped(
                state=state,
                particle_index=particle_index,
                position=particle_result.position,
                velocity=particle_result.velocity,
                update_trial_buffers=False,
            )
            invalid_mask_stopped_count_step += 1
        else:
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

    (
        valid_mask_violation_count_step,
        valid_mask_mixed_stencil_count_step,
        valid_mask_hard_invalid_count_step,
        extension_band_sample_count_step,
    ) = _update_valid_mask_diagnostics(
        state.collision_diagnostics,
        valid_mask_status_flags=state.valid_mask_status_flags,
        extension_band_sample_flags=state.extension_band_sample_flags,
        valid_mask_mixed_seen=state.valid_mask_mixed_seen,
        valid_mask_hard_seen=state.valid_mask_hard_seen,
        extension_band_seen=state.extension_band_seen,
    )

    if bool(capture_step_rows) and int(step.output_write_diagnostics) != 0:
        _append_runtime_step_summary(
            state.step_rows,
            t=float(t_next),
            step=step,
            released=state.released,
            active=state.active,
            stuck=state.stuck,
            absorbed=state.absorbed,
            escaped=state.escaped,
            valid_mask_violation_count_step=int(valid_mask_violation_count_step),
            valid_mask_mixed_stencil_count_step=int(valid_mask_mixed_stencil_count_step),
            valid_mask_hard_invalid_count_step=int(valid_mask_hard_invalid_count_step),
            extension_band_sample_count_step=int(extension_band_sample_count_step),
            invalid_mask_stopped_count_step=int(invalid_mask_stopped_count_step),
        )

    if bool(capture_snapshots) and int(step.output_save_positions) != 0 and state.step_local_counter % save_every == 0:
        _append_snapshot(
            state.save_positions,
            state.save_meta,
            save_index=int(state.save_index),
            t=float(t_next),
            step=step,
            position=state.x,
        )
        state.save_index += 1

    return float(t_next)


def _build_runtime_output_payload(
    prepared: PreparedRuntime,
    spatial_dim: int,
    *,
    capture_outputs: bool,
) -> RuntimeOutputPayload:
    runtime = prepared.runtime
    particles = runtime.particles
    if particles is None:
        raise ValueError('Simulation requires particles')
    resolved = prepared.source_preprocess.resolved if prepared.source_preprocess is not None else None
    config_payload = runtime.config_payload if isinstance(runtime.config_payload, Mapping) else {}
    options = _resolve_solver_runtime_options(config_payload)

    n_particles = int(particles.count)
    mins, maxs = runtime_bounds(runtime)
    compiled = _compile_runtime_arrays(runtime, spatial_dim)
    gas_mu = float(runtime.gas.dynamic_viscosity_Pas)
    triangle_surface_3d = _prepare_triangle_surface(runtime, spatial_dim)
    release_time = np.asarray(particles.release_time, dtype=np.float64)
    base_phys = resolve_step_physics(runtime.physics_catalog, None)
    min_tau_p_s = float(base_phys['min_tau_p_s'])
    base_integrator_name = get_integrator_spec(str(base_phys.get('integrator', 'drag_relaxation'))).name
    tau_p = np.asarray(
        [_particle_tau_p(particles.diameter[i], particles.density[i], gas_mu, min_tau_p_s) for i in range(n_particles)],
        dtype=np.float64,
    )
    flow_scale_particle = np.asarray(resolved.physics_flow_scale if resolved is not None else np.ones(n_particles), dtype=np.float64)
    drag_scale_particle = np.asarray(resolved.physics_drag_tau_scale if resolved is not None else np.ones(n_particles), dtype=np.float64)
    body_scale_particle = np.asarray(resolved.physics_body_accel_scale if resolved is not None else np.ones(n_particles), dtype=np.float64)
    save_positions = [] if bool(capture_outputs) else _DiscardList()
    save_meta = [] if bool(capture_outputs) else _DiscardList()
    wall_rows = [] if bool(capture_outputs) else _DiscardList()
    max_hit_rows = [] if bool(capture_outputs) else _DiscardList()
    step_rows = [] if bool(capture_outputs) else _DiscardList()
    state = RuntimeState(
        x=np.asarray(particles.position[:, :spatial_dim], dtype=np.float64).copy(),
        v=np.asarray(particles.velocity[:, :spatial_dim], dtype=np.float64).copy(),
        released=np.zeros(n_particles, dtype=bool),
        active=np.zeros(n_particles, dtype=bool),
        stuck=np.zeros(n_particles, dtype=bool),
        absorbed=np.zeros(n_particles, dtype=bool),
        escaped=np.zeros(n_particles, dtype=bool),
        invalid_mask_stopped=np.zeros(n_particles, dtype=bool),
        save_positions=save_positions,
        save_meta=save_meta,
        wall_rows=wall_rows,
        max_hit_rows=max_hit_rows,
        step_rows=step_rows,
        wall_law_counts={},
        wall_summary_counts={},
        collision_diagnostics=_initial_collision_diagnostics(),
        rng=np.random.default_rng(int(options.rng_seed)),
        prev_step_name=None,
        step_local_counter=0,
        save_index=1,
        x_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        v_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        x_mid_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        substep_counts=np.ones(n_particles, dtype=np.int32),
        valid_mask_status_flags=np.zeros(n_particles, dtype=np.uint8),
        extension_band_sample_flags=np.zeros(n_particles, dtype=bool),
        valid_mask_mixed_seen=np.zeros(n_particles, dtype=bool),
        valid_mask_hard_seen=np.zeros(n_particles, dtype=bool),
        extension_band_seen=np.zeros(n_particles, dtype=bool),
    )

    init_step = _current_step(runtime, 0.0, options.t_end)
    if bool(capture_outputs):
        _append_snapshot(state.save_positions, state.save_meta, save_index=0, t=0.0, step=init_step, position=state.x)

    boundary_service = build_boundary_service(
        runtime,
        spatial_dim=int(spatial_dim),
        on_boundary_tol_m=float(options.on_boundary_tol_m),
        triangle_surface_3d=triangle_surface_3d,
    )

    t = 0.0
    while t < options.t_end - 1e-15:
        t = _advance_runtime_step(
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
            flow_scale_particle=flow_scale_particle,
            drag_scale_particle=drag_scale_particle,
            body_scale_particle=body_scale_particle,
            release_time=release_time,
            t=float(t),
        )

    final_step = _current_step(runtime, t, options.t_end)
    if bool(capture_outputs) and int(final_step.output_save_positions) != 0 and (
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

    positions = (
        np.stack(state.save_positions, axis=0)
        if bool(capture_outputs) and state.save_positions
        else np.zeros((0, n_particles, spatial_dim), dtype=np.float64)
    )
    return RuntimeOutputPayload(
        prepared=prepared,
        spatial_dim=int(spatial_dim),
        particles=particles,
        release_time=release_time,
        positions=positions,
        save_meta=state.save_meta,
        final_position=state.x,
        final_velocity=state.v,
        released=state.released,
        active=state.active,
        stuck=state.stuck,
        absorbed=state.absorbed,
        escaped=state.escaped,
        invalid_mask_stopped=state.invalid_mask_stopped,
        final_step_name=final_step.step_name,
        final_segment_name=_step_segment_name(final_step),
        wall_rows=state.wall_rows,
        wall_law_counts=state.wall_law_counts,
        wall_summary_counts=state.wall_summary_counts,
        max_hit_rows=state.max_hit_rows,
        step_rows=state.step_rows,
        collision_diagnostics=state.collision_diagnostics,
        base_integrator_name=str(base_integrator_name),
        write_collision_diagnostics=int(options.write_collision_diagnostics),
        max_wall_hits_per_step=int(options.max_wall_hits_per_step),
        max_hits_retry_splits=int(options.max_hits_retry_splits),
        max_hits_retry_local_adaptive_enabled=int(options.max_hits_retry_local_adaptive_enabled),
        min_remaining_dt_ratio=float(options.min_remaining_dt_ratio),
        on_boundary_tol_m=float(options.on_boundary_tol_m),
        epsilon_offset_m=float(options.epsilon_offset_m),
        adaptive_substep_enabled=int(options.adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(options.adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(options.adaptive_substep_max_splits),
        plot_limit=int(options.plot_limit),
        valid_mask_policy=str(options.valid_mask_policy),
    )


def run_prepared_runtime(
    prepared: PreparedRuntime,
    output_dir: Optional[Path],
    spatial_dim: int,
) -> Dict[str, object]:
    capture_outputs = output_dir is not None
    payload = _build_runtime_output_payload(
        prepared,
        spatial_dim=int(spatial_dim),
        capture_outputs=bool(capture_outputs),
    )
    if not bool(capture_outputs):
        return build_runtime_report(payload, outputs_written=False)
    return write_runtime_outputs(payload, Path(output_dir))


__all__ = (
    'RuntimeState',
    'SolverRuntimeOptions',
    'run_prepared_runtime',
)
