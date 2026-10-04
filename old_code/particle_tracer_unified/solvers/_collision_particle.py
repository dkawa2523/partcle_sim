"""Advance one colliding particle through the remaining step segments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import numpy as np

from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryHit

from ._collision_resolution import resolve_collision_segment
from ._collision_trial import (
    advance_partial_with_inputs,
    prepare_collision_segment_trial,
)
from ._collision_types import (
    CollidingParticleAdvanceResult,
    CollisionSegmentInputs,
    CollisionSegmentResolution,
    CollisionSegmentTrial,
    WallHitStepResult,
    _CollisionAdvanceState,
    _CollisionResolutionContext,
    _CollisionSearchContext,
    _WallInteractionContext,
)
from ._collision_wall_events import (
    post_wall_acceptance_reason,
    time_tolerance,
)
from ._coupled_charge_motion import CoupledChargeMotionTrace
from ._stochastic_randomness import WallRandomContext
from .compiled_backend_types import CompiledRuntimeBackend
from .diagnostics import increment_count
from .forces import ForceRuntimeParameters
from .high_fidelity_collision import apply_wall_hit_step
from .segment_motion import trace_motion_segment
from .stochastic_motion import PiecewiseLangevinPath
from .terminal_outcome import terminal_segment_outcome


def initial_collision_advance_state(
    *,
    x_start: np.ndarray,
    v_start: np.ndarray,
    dt_step: float,
    valid_mask_status: int,
    charge_start_C: float | None = None,
) -> _CollisionAdvanceState:
    return _CollisionAdvanceState(
        position=np.asarray(x_start, dtype=np.float64).copy(),
        velocity=np.asarray(v_start, dtype=np.float64).copy(),
        remaining_dt=float(dt_step),
        valid_mask_status=int(valid_mask_status),
        hit_count=0,
        total_hit_count=0,
        hit_part_ids=[],
        hit_outcomes=[],
        use_precomputed_trial=True,
        numerical_boundary_stopped=False,
        numerical_boundary_stop_reason="",
        contact_sliding=False,
        contact_part_id=0,
        contact_normal=None,
        contact_primitive_id=-1,
        terminal_outcome=None,
        charge_C=charge_start_C,
    )


def _prepare_state_segment_trial(
    *,
    state: _CollisionAdvanceState,
    base_inputs: CollisionSegmentInputs,
    search: _CollisionSearchContext,
) -> tuple[float, CollisionSegmentInputs, CollisionSegmentTrial]:
    segment_dt = float(state.remaining_dt)
    segment_inputs = replace(
        base_inputs,
        charge_start_C=state.charge_C,
        stochastic_offset_s=float(
            np.clip(float(search.dt_step) - segment_dt, 0.0, float(search.dt_step))
        ),
    )
    trial = prepare_collision_segment_trial(
        use_precomputed_trial=bool(state.use_precomputed_trial),
        x_curr=state.position,
        v_curr=state.velocity,
        t=float(search.t),
        segment_dt=float(segment_dt),
        inputs=segment_inputs,
        base_adaptive_substep_enabled=int(search.base_adaptive_substep_enabled),
        initial_x_next=search.initial_x_next,
        initial_v_next=search.initial_v_next,
        initial_stage_points=search.initial_stage_points,
        initial_valid_mask_status=int(state.valid_mask_status),
        initial_primary_hit=search.initial_primary_hit,
        initial_primary_hit_counted=bool(search.initial_primary_hit_counted),
        inside_fn=search.inside_fn,
        primary_hit_fn=search.primary_hit_fn,
        nearest_projection_fn=search.nearest_projection_fn,
        on_boundary_tol_m=float(search.on_boundary_tol_m),
        collision_diagnostics=search.collision_diagnostics,
        initial_substep_count=int(search.initial_substep_count),
        initial_accepted_trace=search.initial_accepted_trace,
    )
    state.valid_mask_status = int(trial.particle_valid_mask_status)
    return segment_dt, segment_inputs, trial


def _terminal_trial_advance_result(
    *,
    state: _CollisionAdvanceState,
    stop: CollidingParticleAdvanceResult,
    segment_dt: float,
    dt_step: float,
) -> CollidingParticleAdvanceResult:
    relative_terminal = stop.terminal_outcome
    if relative_terminal is None:
        raise RuntimeError("terminal trial stop requires a terminal outcome")
    elapsed_before_segment = float(dt_step) - float(segment_dt)
    return CollidingParticleAdvanceResult(
        position=np.asarray(stop.position, dtype=np.float64),
        velocity=np.asarray(stop.velocity, dtype=np.float64),
        total_hits=int(state.total_hit_count) + int(stop.total_hits),
        valid_mask_status=int(stop.valid_mask_status),
        invalid_mask_stopped=bool(stop.invalid_mask_stopped),
        invalid_stop_reason=str(stop.invalid_stop_reason),
        numerical_boundary_stopped=bool(stop.numerical_boundary_stopped),
        numerical_boundary_stop_reason=str(stop.numerical_boundary_stop_reason),
        contact_sliding=bool(stop.contact_sliding),
        contact_part_id=int(stop.contact_part_id),
        contact_normal=stop.contact_normal,
        contact_primitive_id=int(stop.contact_primitive_id),
        terminal_outcome=terminal_segment_outcome(
            accepted_elapsed_s=float(elapsed_before_segment)
            + float(relative_terminal.accepted_elapsed_s),
            segment_duration_s=float(dt_step),
            position=relative_terminal.position,
            reason=relative_terminal.reason,
        ),
        charge_C=(state.charge_C if stop.charge_C is None else stop.charge_C),
    )


def _resolve_state_segment(
    *,
    state: _CollisionAdvanceState,
    trial: CollisionSegmentTrial,
    segment_inputs: CollisionSegmentInputs,
    segment_dt: float,
    search: _CollisionSearchContext,
) -> CollisionSegmentResolution:
    accepted_substeps = max(1, int(trial.accepted_substep_count))
    accepted_trace = trial.accepted_trace
    if accepted_trace is None:
        request = segment_inputs.request(
            position_m=state.position,
            velocity_mps=state.velocity,
            duration_s=float(segment_dt),
            end_time_s=float(search.t),
            adaptive_substep_enabled=0,
            minimum_substeps=accepted_substeps,
        )
        if segment_inputs.coupled_charge_tracer is None:
            accepted_trace = trace_motion_segment(request)
        else:
            accepted_trace = segment_inputs.coupled_charge_tracer.trace(
                request,
                charge_start_C=segment_inputs.coupled_charge_start(),
            )
    context = _CollisionResolutionContext(
        x_curr=state.position,
        v_curr=state.velocity,
        x_next=trial.x_next,
        v_next=trial.v_next,
        stage_points=trial.stage_points,
        inside_fn=search.inside_fn,
        strict_inside_fn=search.strict_inside_fn,
        primary_hit_fn=search.primary_hit_fn,
        nearest_projection_fn=search.nearest_projection_fn,
        primary_hit_counter_key=search.primary_hit_counter_key,
        collision_diagnostics=search.collision_diagnostics,
        t=float(search.t),
        segment_dt=float(segment_dt),
        inputs=segment_inputs,
        on_boundary_tol_m=float(search.on_boundary_tol_m),
    )
    return resolve_collision_segment(
        context=context,
        primary_hit=trial.primary_hit,
        primary_hit_counted=bool(trial.primary_hit_counted),
        advance_partial=partial(
            advance_partial_with_inputs,
            accepted_substep_count=accepted_substeps,
            accepted_trace=accepted_trace,
        ),
        time_tolerance=time_tolerance,
    )


def _particle_has_terminal_wall_state(context: _WallInteractionContext) -> bool:
    index = int(context.particle_index)
    return not bool(context.active[index]) and (
        bool(context.stuck[index])
        or bool(context.frozen[index])
        or bool(context.absorbed[index])
        or bool(context.escaped[index])
    )


def _particle_remains_active(context: _WallInteractionContext) -> bool:
    index = int(context.particle_index)
    return bool(context.active[index]) and not (
        bool(context.stuck[index])
        or bool(context.frozen[index])
        or bool(context.absorbed[index])
        or bool(context.escaped[index])
    )


def _commit_wall_result(
    *,
    state: _CollisionAdvanceState,
    wall_result: WallHitStepResult,
) -> None:
    state.position = np.asarray(wall_result.position, dtype=np.float64)
    state.velocity = np.asarray(wall_result.velocity, dtype=np.float64)
    state.remaining_dt = float(wall_result.remaining_dt)
    state.hit_count = int(wall_result.hit_count)
    state.total_hit_count = int(wall_result.total_hit_count)


def _record_wall_terminal_outcome(
    *,
    state: _CollisionAdvanceState,
    context: _WallInteractionContext,
    hit: np.ndarray,
    hit_dt: float,
    segment_dt: float,
    dt_step: float,
) -> None:
    if not _particle_has_terminal_wall_state(context):
        return
    # Terminal particles are never advanced again, so retain the physical hit
    # instead of the interior clearance used only by continuing trajectories.
    state.position = np.asarray(hit, dtype=np.float64)
    elapsed_before_hit = float(dt_step) - float(segment_dt)
    state.terminal_outcome = terminal_segment_outcome(
        accepted_elapsed_s=float(
            np.clip(
                elapsed_before_hit + np.clip(float(hit_dt), 0.0, float(segment_dt)),
                0.0,
                float(dt_step),
            )
        ),
        segment_duration_s=float(dt_step),
        position=hit,
        reason=f"wall_{state.hit_outcomes[-1]!s}",
    )


def _record_contact_state(
    *, state: _CollisionAdvanceState, wall_result: WallHitStepResult
) -> None:
    if not bool(wall_result.entered_contact):
        return
    state.contact_sliding = True
    state.contact_part_id = int(wall_result.contact_part_id)
    state.contact_normal = (
        None
        if wall_result.contact_normal is None
        else np.asarray(wall_result.contact_normal, dtype=np.float64)
    )
    state.contact_primitive_id = int(wall_result.contact_primitive_id)


def _post_wall_stop_reason(
    *,
    state: _CollisionAdvanceState,
    wall_result: WallHitStepResult,
    context: _WallInteractionContext,
    inside_fn: Callable[[np.ndarray], bool],
) -> str:
    if not _particle_remains_active(context):
        return ""
    acceptance_reason = post_wall_acceptance_reason(
        runtime=context.runtime,
        position=state.position,
        velocity=state.velocity,
        inside_fn=inside_fn,
    )
    if acceptance_reason:
        return str(acceptance_reason)
    if (
        bool(wall_result.should_break)
        and int(state.hit_count) >= int(context.max_wall_hits_per_step)
        and float(state.remaining_dt) > 0.0
    ):
        return "max_hits_reached"
    return ""


def _apply_resolved_wall_hit(
    *,
    state: _CollisionAdvanceState,
    hit_event: BoundaryHit,
    hit_velocity: np.ndarray,
    hit_dt: float,
    segment_dt: float,
    search: _CollisionSearchContext,
    wall: _WallInteractionContext,
) -> bool:
    hit = np.asarray(hit_event.position, dtype=np.float64)
    random_context = (
        None
        if wall.random_context is None
        else replace(
            wall.random_context,
            wall_event_ordinal=int(state.total_hit_count),
        )
    )
    wall_result = apply_wall_hit_step(
        runtime=wall.runtime,
        particles=wall.particles,
        particle_index=int(wall.particle_index),
        particle_id=wall.particle_id,
        particle_mass_kg=float(wall.particle_mass_kg),
        particle_diameter_m=float(wall.particle_diameter_m),
        rng=wall.rng,
        wall_random_context=random_context,
        hit=hit,
        n_out=np.asarray(hit_event.normal, dtype=np.float64),
        hit_dt=float(hit_dt),
        part_id=int(hit_event.part_id),
        primitive_id=int(hit_event.primitive_id),
        primitive_kind=str(hit_event.primitive_kind),
        is_ambiguous=bool(hit_event.is_ambiguous),
        v_hit=np.asarray(hit_velocity, dtype=np.float64),
        remaining_dt=float(state.remaining_dt),
        segment_dt=float(segment_dt),
        hit_count=int(state.hit_count),
        total_hit_count=int(state.total_hit_count),
        hit_part_ids=state.hit_part_ids,
        hit_outcomes=state.hit_outcomes,
        collision_diagnostics=wall.collision_diagnostics,
        max_hit_rows=wall.max_hit_rows,
        wall_rows=wall.wall_rows,
        wall_summary_counts=wall.wall_summary_counts,
        stuck=wall.stuck,
        frozen=wall.frozen,
        absorbed=wall.absorbed,
        escaped=wall.escaped,
        active=wall.active,
        max_wall_hits_per_step=int(wall.max_wall_hits_per_step),
        epsilon_offset_m=float(wall.epsilon_offset_m),
        on_boundary_tol_m=float(wall.on_boundary_tol_m),
        t=float(wall.t),
        triangle_surface_3d=wall.triangle_surface_3d,
        allow_contact_sliding=(
            bool(wall.contact_sliding_enabled) and state.charge_C is None
        ),
    )
    _commit_wall_result(state=state, wall_result=wall_result)
    _record_wall_terminal_outcome(
        state=state,
        context=wall,
        hit=hit,
        hit_dt=float(hit_dt),
        segment_dt=float(segment_dt),
        dt_step=float(search.dt_step),
    )
    _record_contact_state(state=state, wall_result=wall_result)
    stop_reason = _post_wall_stop_reason(
        state=state,
        wall_result=wall_result,
        context=wall,
        inside_fn=search.inside_fn,
    )
    if stop_reason:
        state.numerical_boundary_stopped = True
        state.numerical_boundary_stop_reason = str(stop_reason)
        return False
    if bool(wall_result.should_break):
        return False
    state.use_precomputed_trial = False
    return True


def _advance_resolved_segment(
    *,
    state: _CollisionAdvanceState,
    resolution: CollisionSegmentResolution,
    segment_dt: float,
    search: _CollisionSearchContext,
    wall: _WallInteractionContext,
    trial: CollisionSegmentTrial | None = None,
) -> bool:
    accepted_trace = None if trial is None else trial.accepted_trace
    if bool(resolution.advance_without_hit):
        state.position = np.asarray(resolution.x_next, dtype=np.float64)
        state.velocity = np.asarray(resolution.v_next, dtype=np.float64)
        if state.charge_C is not None and isinstance(
            accepted_trace, CoupledChargeMotionTrace
        ):
            state.charge_C = float(accepted_trace.endpoint_charge_C)
        return False
    hit_event = resolution.hit_event
    hit_velocity = resolution.v_hit
    if bool(resolution.should_break) or hit_event is None or hit_velocity is None:
        return False
    if state.charge_C is not None and isinstance(
        accepted_trace, CoupledChargeMotionTrace
    ):
        state.charge_C = float(accepted_trace.charge_at(float(resolution.hit_dt)))
    return _apply_resolved_wall_hit(
        state=state,
        hit_event=hit_event,
        hit_velocity=hit_velocity,
        hit_dt=float(resolution.hit_dt),
        segment_dt=float(segment_dt),
        search=search,
        wall=wall,
    )


def _finish_collision_advance(
    *,
    state: _CollisionAdvanceState,
    dt_step: float,
    collision_diagnostics: dict[str, object],
) -> CollidingParticleAdvanceResult:
    if bool(state.numerical_boundary_stopped):
        state.terminal_outcome = terminal_segment_outcome(
            accepted_elapsed_s=float(
                np.clip(float(dt_step) - float(state.remaining_dt), 0.0, float(dt_step))
            ),
            segment_duration_s=float(dt_step),
            position=state.position,
            reason=str(state.numerical_boundary_stop_reason),
        )
    if state.total_hit_count > 1 and bool(
        getattr(collision_diagnostics, "debug", True)
    ):
        increment_count(collision_diagnostics, "multi_hit_events_count")
    return CollidingParticleAdvanceResult(
        position=state.position,
        velocity=state.velocity,
        total_hits=int(state.total_hit_count),
        valid_mask_status=int(state.valid_mask_status),
        invalid_mask_stopped=False,
        invalid_stop_reason="",
        numerical_boundary_stopped=bool(state.numerical_boundary_stopped),
        numerical_boundary_stop_reason=str(state.numerical_boundary_stop_reason),
        contact_sliding=bool(state.contact_sliding),
        contact_part_id=int(state.contact_part_id),
        contact_normal=state.contact_normal,
        contact_primitive_id=int(state.contact_primitive_id),
        terminal_outcome=state.terminal_outcome,
        charge_C=state.charge_C,
    )


def _advance_collision_segments(
    *,
    state: _CollisionAdvanceState,
    base_inputs: CollisionSegmentInputs,
    search: _CollisionSearchContext,
    wall: _WallInteractionContext,
) -> CollidingParticleAdvanceResult:
    while bool(wall.active[wall.particle_index]) and state.remaining_dt > 0.0:
        segment_dt, segment_inputs, trial = _prepare_state_segment_trial(
            state=state,
            base_inputs=base_inputs,
            search=search,
        )
        terminal_stop = trial.terminal_stop_result
        if terminal_stop is not None:
            return _terminal_trial_advance_result(
                state=state,
                stop=terminal_stop,
                segment_dt=float(segment_dt),
                dt_step=float(search.dt_step),
            )
        resolution = _resolve_state_segment(
            state=state,
            trial=trial,
            segment_inputs=segment_inputs,
            segment_dt=float(segment_dt),
            search=search,
        )
        if not _advance_resolved_segment(
            state=state,
            resolution=resolution,
            trial=trial,
            segment_dt=float(segment_dt),
            search=search,
            wall=wall,
        ):
            break
    return _finish_collision_advance(
        state=state,
        dt_step=float(search.dt_step),
        collision_diagnostics=search.collision_diagnostics,
    )


def advance_colliding_particle(
    *,
    runtime,
    particles,
    particle_index: int,
    rng: np.random.Generator | None,
    t: float,
    x_start: np.ndarray,
    v_start: np.ndarray,
    dt_step: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackend,
    base_adaptive_substep_enabled: int,
    adaptive_substep_max_splits: int,
    tau_p_i: float,
    particle_diameter_i: float,
    body_accel: np.ndarray,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    drag_model_mode: int,
    initial_x_next: np.ndarray,
    initial_v_next: np.ndarray,
    initial_stage_points: np.ndarray,
    initial_valid_mask_status: int,
    initial_primary_hit: BoundaryHit | None,
    initial_primary_hit_counted: bool,
    inside_fn: Callable[[np.ndarray], bool],
    strict_inside_fn: Callable[[np.ndarray], bool],
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    primary_hit_counter_key: str,
    collision_diagnostics: dict[str, object],
    max_hit_rows: list[dict[str, object]] | None,
    wall_rows: list[dict[str, object]] | None,
    wall_summary_counts: dict[tuple[int, str, str], int],
    stuck: np.ndarray,
    frozen: np.ndarray | None = None,
    absorbed: np.ndarray,
    escaped: np.ndarray | None = None,
    active: np.ndarray,
    max_wall_hits_per_step: int,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
    triangle_surface_3d: TriangleSurface3D | None,
    contact_sliding_enabled: bool = True,
    electric_q_over_m_i: float | None = None,
    particle_density_i: float = 1000.0,
    particle_mass_i: float = 0.0,
    particle_id_i: int | None = None,
    dep_particle_rel_permittivity_i: float = float("nan"),
    thermophoretic_coeff_i: float = float("nan"),
    force_runtime: ForceRuntimeParameters | None = None,
    gas_temperature_K: float = float("nan"),
    gas_molecular_mass_kg: float = float("nan"),
    stochastic_path: PiecewiseLangevinPath | None = None,
    wall_random_context: WallRandomContext | None = None,
    initial_substep_count: int = 1,
    coupled_charge_tracer=None,
    charge_start_C: float | None = None,
    initial_accepted_trace=None,
) -> CollidingParticleAdvanceResult:
    if escaped is None:
        escaped = np.zeros_like(active, dtype=bool)
    if frozen is None:
        frozen = np.zeros_like(active, dtype=bool)
    base_segment_inputs = CollisionSegmentInputs(
        spatial_dim=int(spatial_dim),
        compiled=compiled,
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        tau_p_i=float(tau_p_i),
        particle_diameter_i=float(particle_diameter_i),
        particle_density_i=float(particle_density_i),
        particle_mass_i=float(particle_mass_i),
        dep_particle_rel_permittivity_i=float(dep_particle_rel_permittivity_i),
        thermophoretic_coeff_i=float(thermophoretic_coeff_i),
        body_accel=np.asarray(body_accel, dtype=np.float64),
        gas_density_kgm3=float(gas_density_kgm3),
        gas_mu_pas=float(gas_mu_pas),
        gas_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        drag_model_mode=int(drag_model_mode),
        electric_q_over_m_i=electric_q_over_m_i,
        force_runtime=force_runtime,
        stochastic_path=stochastic_path,
        coupled_charge_tracer=coupled_charge_tracer,
        charge_start_C=charge_start_C,
    )
    search = _CollisionSearchContext(
        t=float(t),
        dt_step=float(dt_step),
        base_adaptive_substep_enabled=int(base_adaptive_substep_enabled),
        initial_x_next=np.asarray(initial_x_next, dtype=np.float64),
        initial_v_next=np.asarray(initial_v_next, dtype=np.float64),
        initial_stage_points=np.asarray(initial_stage_points, dtype=np.float64),
        initial_primary_hit=initial_primary_hit,
        initial_primary_hit_counted=bool(initial_primary_hit_counted),
        inside_fn=inside_fn,
        strict_inside_fn=strict_inside_fn,
        primary_hit_fn=primary_hit_fn,
        nearest_projection_fn=nearest_projection_fn,
        primary_hit_counter_key=str(primary_hit_counter_key),
        collision_diagnostics=collision_diagnostics,
        on_boundary_tol_m=float(on_boundary_tol_m),
        initial_substep_count=max(1, int(initial_substep_count)),
        initial_accepted_trace=initial_accepted_trace,
    )
    wall = _WallInteractionContext(
        runtime=runtime,
        particles=particles,
        particle_index=int(particle_index),
        particle_id=particle_id_i,
        particle_mass_kg=float(particle_mass_i),
        particle_diameter_m=float(particle_diameter_i),
        rng=rng,
        random_context=wall_random_context,
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=max_hit_rows,
        wall_rows=wall_rows,
        wall_summary_counts=wall_summary_counts,
        stuck=stuck,
        frozen=frozen,
        absorbed=absorbed,
        escaped=escaped,
        active=active,
        max_wall_hits_per_step=int(max_wall_hits_per_step),
        epsilon_offset_m=float(epsilon_offset_m),
        on_boundary_tol_m=float(on_boundary_tol_m),
        t=float(t),
        triangle_surface_3d=triangle_surface_3d,
        contact_sliding_enabled=bool(contact_sliding_enabled),
    )
    state = initial_collision_advance_state(
        x_start=x_start,
        v_start=v_start,
        dt_step=float(dt_step),
        valid_mask_status=int(initial_valid_mask_status),
        charge_start_C=charge_start_C,
    )
    return _advance_collision_segments(
        state=state,
        base_inputs=base_segment_inputs,
        search=search,
        wall=wall,
    )
