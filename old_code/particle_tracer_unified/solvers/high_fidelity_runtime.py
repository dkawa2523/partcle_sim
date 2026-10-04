"""Orchestrate release events and high-fidelity solver step phases."""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np

from particle_tracer_unified.core.coordinate_systems import (
    canonicalize_axisymmetric_rz_state,
)
from particle_tracer_unified.core.datamodel import SolverContext
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from particle_tracer_unified.domain import BoundaryHit, StageFields

from ._collision_particle import advance_colliding_particle
from ._coupled_charge_motion import (
    CoupledChargeMotionBatch,
    trace_coupled_charge_batch,
)
from ._runtime_charge import (
    begin_charge_segment,
    finish_charge_segment,
)
from ._runtime_charge import (
    electric_q_over_m_particle as _electric_q_over_m_particle,
)
from ._runtime_collisions import classify_step_collisions, resolve_step_colliders
from ._runtime_release_schedule import (
    ParticleStepInterval,
    particle_intervals_for_macro_step,
)
from ._runtime_terminal_state import (
    mark_invalid_mask_stopped,
    mark_numerical_boundary_stopped,
)
from ._runtime_timing import add_detailed_timing_since as _add_detailed_timing_since
from ._runtime_timing import add_timing as _add_timing
from ._runtime_timing import detailed_timer_start as _detailed_timer_start
from ._runtime_trace_refinement import (
    refine_deterministic_stage_traces as _refine_deterministic_stage_traces,
)
from ._runtime_valid_mask import apply_valid_mask_retry_then_stop
from ._stochastic_composition import _compose_piecewise_langevin_paths
from ._stochastic_randomness import BrownianRandomContext
from .collision_detection import classify_trial_collisions
from .contact_sliding import advance_contact_sliding_particles
from .diagnostics import increment_count
from .field_runtime import (
    measure_sample_fields_for_stage,
    record_field_sampling_diagnostics,
    sample_fields_for_stage,
)
from .output_buffers import DebugBuffers
from .particle_state import (
    activate_release_cursor_until,
)
from .runtime_execution import (
    RunExecutionContext,
    StepLoopResult,
    append_snapshot,
    finalize_runtime_execution,
    initialize_debug_buffers,
    prepare_runtime_execution,
)
from .segment_motion import (
    SegmentMotionBatchDestination,
    SegmentMotionBatchRequest,
    SegmentMotionBatchTrace,
    trace_motion_batch,
)
from .segment_trace import (
    TraceRefinementPolicy,
    interpolation_resolution_from_backend,
    support_spacing_from_backend,
)
from .solver_outcome import SolverOutcome
from .stochastic_motion import (
    PiecewiseLangevinPath,
    merge_stochastic_motion_diagnostics,
    sample_piecewise_langevin_paths,
)
from .terminal_outcome import TerminalSegmentOutcome, terminal_segment_outcome

_RUN_NAME = "run"


def _coupled_charge_motion_enabled(
    execution: RunExecutionContext,
    *,
    electric_force_enabled: bool,
) -> bool:
    return bool(execution.options.charge_model.enabled) and bool(electric_force_enabled)


def _validate_coupled_charge_motion(execution: RunExecutionContext) -> None:
    if bool(execution.options.stochastic_motion.enabled):
        raise ValueError(
            "dynamic charge with electric force is not yet supported with "
            "stochastic motion"
        )
    if np.any(execution.state.active & execution.state.contact_sliding):
        raise ValueError(
            "dynamic charge with electric force is not yet supported during "
            "contact sliding"
        )
    if str(execution.context.coordinate_system) == "axisymmetric_rz":
        raise ValueError(
            "dynamic charge with electric force currently requires Cartesian 2D"
        )


def _sample_runtime_stage(
    diagnostics: dict[str, object],
    *args,
    **kwargs,
) -> StageFields:
    """Sample without a timer unless detailed diagnostics were requested."""

    if bool(getattr(diagnostics, "debug", True)):
        fields, metrics = measure_sample_fields_for_stage(*args, **kwargs)
        record_field_sampling_diagnostics(diagnostics, metrics)
        return fields
    return sample_fields_for_stage(*args, **kwargs)


def _update_adaptive_substep_diagnostics(
    collision_diagnostics: dict[str, object],
    *,
    adaptive_substep_enabled: int,
    adaptive_substep_max_splits: int,
    active: np.ndarray,
    substep_counts: np.ndarray,
) -> None:
    if int(adaptive_substep_enabled) == 0 or not np.any(active):
        return
    active_substeps = np.asarray(substep_counts[active], dtype=np.int64)
    maximum_substeps = 1 << max(0, int(adaptive_substep_max_splits))
    increment_count(
        collision_diagnostics,
        "adaptive_substep_segments_count",
        int(np.sum(active_substeps)),
    )
    increment_count(
        collision_diagnostics,
        "adaptive_substep_trigger_count",
        int(np.count_nonzero(active_substeps > 1)),
    )
    increment_count(
        collision_diagnostics,
        "adaptive_substep_limit_reached_count",
        int(np.count_nonzero(active_substeps == maximum_substeps)),
    )


def _update_valid_mask_diagnostics(
    collision_diagnostics: dict[str, object],
    *,
    valid_mask_status_flags: np.ndarray,
    valid_mask_mixed_seen: np.ndarray,
    valid_mask_hard_seen: np.ndarray,
) -> tuple[int, int, int]:
    statuses = np.asarray(valid_mask_status_flags, dtype=np.uint8)
    mixed_step_mask = statuses == int(VALID_MASK_STATUS_MIXED_STENCIL)
    hard_step_mask = statuses == int(VALID_MASK_STATUS_HARD_INVALID)
    mixed_count_step = int(np.count_nonzero(mixed_step_mask))
    hard_count_step = int(np.count_nonzero(hard_step_mask))
    violation_count_step = int(mixed_count_step + hard_count_step)
    increment_count(
        collision_diagnostics, "valid_mask_violation_count", violation_count_step
    )
    increment_count(
        collision_diagnostics, "valid_mask_hard_invalid_count", hard_count_step
    )
    if bool(getattr(collision_diagnostics, "debug", True)):
        valid_mask_mixed_seen |= mixed_step_mask
        valid_mask_hard_seen |= hard_step_mask
        collision_diagnostics["valid_mask_violation_particle_count"] = int(
            np.count_nonzero(valid_mask_mixed_seen | valid_mask_hard_seen)
        )
        increment_count(
            collision_diagnostics,
            "valid_mask_mixed_stencil_count",
            mixed_count_step,
        )
        collision_diagnostics["valid_mask_mixed_stencil_particle_count"] = int(
            np.count_nonzero(valid_mask_mixed_seen)
        )
        collision_diagnostics["valid_mask_hard_invalid_particle_count"] = int(
            np.count_nonzero(valid_mask_hard_seen)
        )
    return int(violation_count_step), int(mixed_count_step), int(hard_count_step)


def _append_runtime_step_summary(
    buffers: DebugBuffers,
    *,
    t: float,
    released: np.ndarray,
    active: np.ndarray,
    stuck: np.ndarray,
    frozen: np.ndarray,
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
    stopped_count = int(
        np.count_nonzero(stuck | frozen | absorbed | contact_sliding | escaped)
    )
    buffers.step_summary.append(
        time_s=float(t),
        step_name=_RUN_NAME,
        segment_name=_RUN_NAME,
        released_count=int(released.sum()),
        active_count=int(active.sum()),
        stuck_count=int(stuck.sum()),
        frozen_count=int(frozen.sum()),
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


def _record_active_count_summary(
    collision_diagnostics: dict[str, object], active_idx: np.ndarray
) -> None:
    active_count = int(np.asarray(active_idx).size)
    samples = int(collision_diagnostics.get("active_count_samples", 0)) + 1
    previous_mean = float(collision_diagnostics.get("active_count_mean", 0.0))
    collision_diagnostics["active_count_samples"] = int(samples)
    collision_diagnostics["active_count_mean"] = previous_mean + (
        float(active_count) - previous_mean
    ) / float(samples)
    collision_diagnostics["active_count_max"] = int(
        max(active_count, int(collision_diagnostics.get("active_count_max", 0)))
    )


def _stop_unresolved_traces(
    execution: RunExecutionContext,
    *,
    unresolved_traces: Mapping[int, str],
    mobile_active: np.ndarray,
    deterministic_stage_points: dict[int, np.ndarray],
    terminal_outcomes: dict[int, TerminalSegmentOutcome],
    dt_step: float,
) -> int:
    state = execution.state
    invalid_mask_stopped = 0
    for particle_index, unresolved_kind in unresolved_traces.items():
        index = int(particle_index)
        field_support_failed = str(unresolved_kind) == "field_support"
        reason = (
            "freeflight_field_support_refinement_exhausted"
            if field_support_failed
            else "trace_refinement_unresolved"
        )
        increment_count(state.collision_diagnostics, "unresolved_crossing_count")
        if field_support_failed:
            mark_invalid_mask_stopped(
                state=state,
                particle_index=index,
                position=state.x[index],
                velocity=state.v[index],
                update_trial_buffers=True,
                reason=reason,
            )
            invalid_mask_stopped += 1
        else:
            mark_numerical_boundary_stopped(
                state=state,
                particle_index=index,
                position=state.x[index],
                velocity=state.v[index],
                update_trial_buffers=True,
                reason=reason,
            )
        terminal_outcomes[index] = terminal_segment_outcome(
            accepted_elapsed_s=0.0,
            segment_duration_s=float(dt_step),
            position=state.x[index],
            reason=reason,
        )
        mobile_active[index] = False
        deterministic_stage_points.pop(index, None)
    return invalid_mask_stopped


def _sample_stochastic_step_paths(
    execution: RunExecutionContext,
    *,
    motion_batch: SegmentMotionBatchTrace | CoupledChargeMotionBatch,
    mobile_active: np.ndarray,
) -> tuple[
    dict[int, PiecewiseLangevinPath],
    dict[int, np.ndarray],
    dict[int, BoundaryHit],
]:
    options = execution.options
    if not bool(options.stochastic_motion.enabled):
        return {}, {}, {}
    if isinstance(motion_batch, CoupledChargeMotionBatch):
        raise ValueError(
            "dynamic charge with electric force is not supported with stochastic motion"
        )
    state = execution.state
    plan = execution.plan
    detailed_timing = state.timing_accumulator if bool(plan.output.is_debug) else None
    started_at = _detailed_timer_start(detailed_timing)
    particle_indices = np.flatnonzero(mobile_active)
    paths, result = sample_piecewise_langevin_paths(
        config=options.stochastic_motion,
        rng=None,
        motion_batch=motion_batch,
        particle_indices=particle_indices,
        minimum_substeps=state.substep_counts,
        particle_mass=execution.particle_mass,
        gas_temperature_K=float(execution.physics["gas_temperature_K"]),
        collect_diagnostics=bool(plan.output.is_debug),
        _random_context=BrownianRandomContext(
            particle_id=execution.particle_id,
            cohort_index=state.stochastic_cohort_index,
            macro_step_index=int(state.step_index),
        ),
    )
    sampled_indices = np.fromiter(paths, dtype=np.int64, count=len(paths))
    state.stochastic_cohort_index[sampled_indices] += 1
    composition = _compose_piecewise_langevin_paths(
        paths=paths,
        motion_batch=motion_batch,
        minimum_substeps=state.substep_counts,
        endpoint_position_m=state.x_trial,
        endpoint_velocity_mps=state.v_trial,
        midpoint_position_m=state.x_mid_trial,
        aggregate_support_status=state.valid_mask_status_flags,
        boundary_service=execution.boundary_service,
        geometry_tolerance_m=float(plan.boundary.classification_tolerance_m),
    )
    for index in composition.unresolved_indices:
        motion_batch.local_error_resolved[int(index)] = False
        paths.pop(int(index), None)
    stage_points = {
        int(index): points
        for index, points in composition.stage_points.items()
        if int(index) in paths
    }
    if bool(plan.output.is_debug):
        merge_stochastic_motion_diagnostics(
            state.collision_diagnostics,
            options.stochastic_motion,
            result,
        )
    _add_detailed_timing_since(detailed_timing, "stochastic_motion_s", started_at)
    return paths, stage_points, composition.prefetched_hits


def _compose_stochastic_interval(
    execution: RunExecutionContext,
    *,
    motion_batch: SegmentMotionBatchTrace | CoupledChargeMotionBatch,
    mobile_active: np.ndarray,
    deterministic_stage_points: dict[int, np.ndarray],
    terminal_outcomes: dict[int, TerminalSegmentOutcome],
    dt_step: float,
) -> tuple[
    dict[int, PiecewiseLangevinPath],
    dict[int, np.ndarray],
    dict[int, BoundaryHit],
    int,
]:
    paths, stage_points, prefetched_hits = _sample_stochastic_step_paths(
        execution,
        motion_batch=motion_batch,
        mobile_active=mobile_active,
    )
    unresolved = {
        int(index): "local_error"
        for index in np.flatnonzero(
            mobile_active & ~np.asarray(motion_batch.local_error_resolved, dtype=bool)
        )
    }
    stopped = _stop_unresolved_traces(
        execution,
        unresolved_traces=unresolved,
        mobile_active=mobile_active,
        deterministic_stage_points=deterministic_stage_points,
        terminal_outcomes=terminal_outcomes,
        dt_step=float(dt_step),
    )
    return paths, stage_points, prefetched_hits, int(stopped)


def _commit_safe_endpoints(
    execution: RunExecutionContext,
    *,
    safe: np.ndarray,
    mobile_active: np.ndarray,
    coupled_motion_batch: CoupledChargeMotionBatch | None,
) -> None:
    state = execution.state
    safe_active = safe[mobile_active[safe]] if safe.size else safe
    if not safe_active.size:
        return
    position = state.x_trial[safe_active]
    velocity = state.v_trial[safe_active]
    if str(execution.context.coordinate_system) == "axisymmetric_rz":
        position, velocity = canonicalize_axisymmetric_rz_state(position, velocity)
    state.x[safe_active] = position
    state.v[safe_active] = velocity
    if coupled_motion_batch is not None:
        state.charge[safe_active] = coupled_motion_batch.endpoint_charge_C[safe_active]


def _advance_active_interval(
    execution: RunExecutionContext,
    *,
    t_start: float,
    t_end: float,
) -> tuple[int, int, int, int]:
    """Advance the particles currently selected by ``state.active``."""

    runtime = execution.context
    state = execution.state
    options = execution.options
    compiled = execution.compiled
    boundary_service = execution.boundary_service
    spatial_dim = int(execution.spatial_dim)
    tau_p = execution.tau_p
    particle_mass = execution.particle_mass
    particle_diameter = execution.particle_diameter
    particle_density = execution.particle_density
    dep_particle_rel_permittivity = execution.dep_particle_rel_permittivity
    thermophoretic_coeff = execution.thermophoretic_coeff
    plan = execution.plan
    drag_model_mode = int(plan.drag_model_mode)
    adaptive_substep_enabled = int(plan.adaptive_substep_enabled)
    adaptive_substep_max_splits = int(plan.adaptive_substep_max_splits)
    on_boundary_tol_m = float(plan.boundary.classification_tolerance_m)

    detailed_timing = state.timing_accumulator if bool(plan.output.is_debug) else None
    dt_step = float(t_end) - float(t_start)
    t_next = float(t_end)
    terminal_outcomes: dict[int, TerminalSegmentOutcome] = {}
    phys = execution.physics
    body_accel = execution.body_acceleration_mps2
    electric_force_enabled = bool(
        runtime.force_catalog is not None and runtime.force_catalog.enabled("electric")
    )
    coupled_charge_motion = _coupled_charge_motion_enabled(
        execution,
        electric_force_enabled=bool(electric_force_enabled),
    )
    if coupled_charge_motion:
        _validate_coupled_charge_motion(execution)
    charge_snapshot = None
    if not coupled_charge_motion:
        charge_snapshot = begin_charge_segment(
            execution,
            t_start=t_start,
            dt_step=dt_step,
            timer_start=_detailed_timer_start,
            record_timing=_add_detailed_timing_since,
        )
    electric_q_over_m_particle = _electric_q_over_m_particle(
        electric_force_enabled=bool(
            electric_force_enabled and not coupled_charge_motion
        ),
        charge=state.charge,
        particle_mass=particle_mass,
    )
    advance_contact_sliding_particles(
        execution,
        body_acceleration=body_accel,
        duration_s=float(dt_step),
        time_s=float(t_next),
        electric_q_over_m_particle=electric_q_over_m_particle,
        sample_stage=_sample_runtime_stage,
    )
    active_idx = np.flatnonzero(state.active)
    mobile_active_idx = active_idx[~state.contact_sliding[active_idx]]
    mobile_active = np.zeros_like(state.active, dtype=bool)
    mobile_active[mobile_active_idx] = True
    state.valid_mask_status_flags.fill(int(VALID_MASK_STATUS_CLEAN))

    t_section = _detailed_timer_start(detailed_timing)
    motion_request = SegmentMotionBatchRequest(
        position_m=state.x,
        velocity_mps=state.v,
        active=mobile_active,
        tau_stokes_s=tau_p,
        particle_diameter_m=particle_diameter,
        particle_mass_kg=particle_mass,
        particle_density_kgm3=particle_density,
        dep_particle_rel_permittivity=dep_particle_rel_permittivity,
        thermophoretic_coefficient=thermophoretic_coeff,
        end_time_s=float(t_next),
        duration_s=float(dt_step),
        spatial_dim=int(spatial_dim),
        backend=compiled,
        body_acceleration_mps2=body_accel,
        gas_density_kgm3=float(phys["gas_density_kgm3"]),
        gas_dynamic_viscosity_Pas=float(phys["gas_mu_pas"]),
        gas_temperature_K=float(phys["gas_temperature_K"]),
        gas_molecular_mass_kg=float(phys["gas_molecular_mass_kg"]),
        drag_model_mode=int(drag_model_mode),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        electric_q_over_m_Ckg=electric_q_over_m_particle,
        force_runtime=options.force_runtime,
    )
    motion_destination = SegmentMotionBatchDestination(
        endpoint_position_m=state.x_trial,
        endpoint_velocity_mps=state.v_trial,
        midpoint_position_m=state.x_mid_trial,
        substep_count=state.substep_counts,
        aggregate_support_status=state.valid_mask_status_flags,
        local_error_resolved=state.local_error_resolved,
    )
    coupled_motion_batch: CoupledChargeMotionBatch | None = None
    if coupled_charge_motion:
        coupled_motion_batch = trace_coupled_charge_batch(
            motion_request,
            motion_destination,
            charge_start_C=state.charge,
            config=options.charge_model,
            runtime=runtime,
            plasma_background=options.plasma_background,
            physical_diameter_m=execution.particle_physical_diameter,
        )
        motion_batch = coupled_motion_batch
    else:
        motion_batch = trace_motion_batch(motion_request, motion_destination)
    deterministic_stage_points: dict[int, np.ndarray] = {}
    refinement_policy = TraceRefinementPolicy(
        on_boundary_tolerance_m=float(on_boundary_tol_m),
        support_spacing_m=float(support_spacing_from_backend(compiled)),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        interpolation_resolution_m=float(
            interpolation_resolution_from_backend(compiled)
        ),
    )
    unresolved_traces = _refine_deterministic_stage_traces(
        runtime=runtime,
        boundary_service=boundary_service,
        motion_batch=motion_batch,
        stage_traces=deterministic_stage_points,
        refinement_policy=refinement_policy,
    )
    trace_invalid_mask_stopped_count_step = _stop_unresolved_traces(
        execution,
        unresolved_traces=unresolved_traces,
        mobile_active=mobile_active,
        deterministic_stage_points=deterministic_stage_points,
        terminal_outcomes=terminal_outcomes,
        dt_step=dt_step,
    )
    (
        stochastic_paths,
        stochastic_stage_points,
        stochastic_prefetched_hits,
        stochastic_stopped_count,
    ) = _compose_stochastic_interval(
        execution,
        motion_batch=motion_batch,
        mobile_active=mobile_active,
        deterministic_stage_points=deterministic_stage_points,
        terminal_outcomes=terminal_outcomes,
        dt_step=dt_step,
    )
    trace_invalid_mask_stopped_count_step += int(stochastic_stopped_count)
    if detailed_timing is not None:
        freeflight_elapsed_s = time.perf_counter() - t_section
        _add_timing(detailed_timing, "freeflight_s", freeflight_elapsed_s)

    if bool(plan.output.is_debug):
        _update_adaptive_substep_diagnostics(
            state.collision_diagnostics,
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            active=mobile_active,
            substep_counts=state.substep_counts,
        )

    trial_batch = classify_step_collisions(
        execution,
        mobile_active=mobile_active,
        deterministic_stage_points=deterministic_stage_points,
        stochastic_paths=stochastic_paths,
        stochastic_prefetched_hits=stochastic_prefetched_hits,
        classify_trial=classify_trial_collisions,
        timer_start=_detailed_timer_start,
        record_timing=_add_detailed_timing_since,
    )
    safe = trial_batch.safe
    t_section = _detailed_timer_start(detailed_timing)
    invalid_mask_stopped_count_step = int(
        trace_invalid_mask_stopped_count_step
    ) + apply_valid_mask_retry_then_stop(
        execution,
        dt_step=float(dt_step),
        t_end_step=float(t_next),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        terminal_outcomes=terminal_outcomes,
        electric_q_over_m_particle=electric_q_over_m_particle,
        particle_indices=safe,
        stochastic_paths=stochastic_paths,
        coupled_motion_batch=coupled_motion_batch,
    )
    _add_detailed_timing_since(detailed_timing, "valid_mask_retry_s", t_section)
    _commit_safe_endpoints(
        execution,
        safe=safe,
        mobile_active=mobile_active,
        coupled_motion_batch=coupled_motion_batch,
    )

    invalid_mask_stopped_count_step += resolve_step_colliders(
        execution,
        trial=trial_batch,
        deterministic_stage_points=deterministic_stage_points,
        stochastic_stage_points=stochastic_stage_points,
        stochastic_paths=stochastic_paths,
        electric_q_over_m_particle=electric_q_over_m_particle,
        terminal_outcomes=terminal_outcomes,
        t_next=t_next,
        dt_step=dt_step,
        adaptive_substep_enabled=adaptive_substep_enabled,
        advance_particle=advance_colliding_particle,
        timer_start=_detailed_timer_start,
        record_timing=_add_detailed_timing_since,
        coupled_motion_batch=coupled_motion_batch,
    )

    finish_charge_segment(
        execution,
        snapshot=charge_snapshot,
        t_start=t_start,
        t_next=t_next,
        dt_step=dt_step,
        electric_force_enabled=electric_force_enabled,
        terminal_outcomes=terminal_outcomes,
        timer_start=_detailed_timer_start,
        record_timing=_add_detailed_timing_since,
    )

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

    return (
        int(valid_mask_violation_count_step),
        int(valid_mask_mixed_stencil_count_step),
        int(valid_mask_hard_invalid_count_step),
        int(invalid_mask_stopped_count_step),
    )


def _advance_particle_interval(
    execution: RunExecutionContext,
    interval: ParticleStepInterval,
) -> tuple[int, int, int, int]:
    """Advance one cohort without exposing other cohorts as active."""

    state = execution.state
    selected = np.zeros_like(state.active, dtype=bool)
    selected[np.asarray(interval.particle_indices, dtype=np.int64)] = True
    suspended = state.active & ~selected
    state.active[suspended] = False
    try:
        return _advance_active_interval(
            execution,
            t_start=float(interval.start_s),
            t_end=float(interval.end_s),
        )
    finally:
        state.active[suspended] = True


def _complete_macro_step(
    execution: RunExecutionContext,
    *,
    t_end: float,
    valid_mask_violation_count: int,
    valid_mask_mixed_stencil_count: int,
    valid_mask_hard_invalid_count: int,
    invalid_mask_stopped_count: int,
) -> None:
    """Publish one summary and apply ``save_every`` to nominal step count."""

    state = execution.state
    plan = execution.plan
    state.step_index += 1
    debug_buffers = state.debug_buffers
    if debug_buffers is not None:
        output_started = time.perf_counter()
        _append_runtime_step_summary(
            debug_buffers,
            t=float(t_end),
            released=state.released,
            active=state.active,
            stuck=state.stuck,
            frozen=state.frozen,
            absorbed=state.absorbed,
            contact_sliding=state.contact_sliding,
            escaped=state.escaped,
            valid_mask_violation_count_step=int(valid_mask_violation_count),
            valid_mask_mixed_stencil_count_step=int(valid_mask_mixed_stencil_count),
            valid_mask_hard_invalid_count_step=int(valid_mask_hard_invalid_count),
            invalid_mask_stopped_count_step=int(invalid_mask_stopped_count),
            save_positions_enabled=1,
            write_wall_events_enabled=1,
            write_diagnostics_enabled=int(plan.output.is_debug),
        )
        if state.step_index % int(plan.output.save_every) == 0:
            append_snapshot(
                debug_buffers.trajectory_positions,
                debug_buffers.save_frames,
                save_index=int(state.save_index),
                t=float(t_end),
                position=state.x,
            )
            state.save_index += 1
        _add_timing(
            state.timing_accumulator,
            "output_step_summary_s",
            time.perf_counter() - output_started,
        )

    if bool(plan.output.is_debug):
        _record_active_count_summary(
            state.collision_diagnostics, np.flatnonzero(state.active)
        )


def _advance_macro_step(
    execution: RunExecutionContext,
    *,
    t_start: float,
    t_end: float,
) -> None:
    """Advance existing, then time-ordered release cohorts to one boundary."""

    state = execution.state
    intervals = particle_intervals_for_macro_step(
        cursor=state.release_cursor,
        released=state.released,
        active=state.active,
        start_s=float(t_start),
        end_s=float(t_end),
    )
    valid_count = 0
    mixed_count = 0
    hard_count = 0
    stopped_count = 0
    for interval in intervals:
        interval_counts = _advance_particle_interval(execution, interval)
        valid_count += int(interval_counts[0])
        mixed_count += int(interval_counts[1])
        hard_count += int(interval_counts[2])
        stopped_count += int(interval_counts[3])
    _complete_macro_step(
        execution,
        t_end=float(t_end),
        valid_mask_violation_count=int(valid_count),
        valid_mask_mixed_stencil_count=int(mixed_count),
        valid_mask_hard_invalid_count=int(hard_count),
        invalid_mask_stopped_count=int(stopped_count),
    )


def _run_runtime_step_loop(
    execution: RunExecutionContext,
) -> StepLoopResult:
    state = execution.state
    plan = execution.plan
    t_end = float(plan.t_end)
    loop_t0 = time.perf_counter()
    step_count = 0
    t = 0.0
    while t < t_end:
        t_previous = float(t)
        t_next = min(float(step_count + 1) * float(plan.dt), t_end)
        if not np.isfinite(t_next) or t_next <= t_previous:
            raise RuntimeError(
                "solver step loop did not advance time "
                f"(t={t_previous}, next_t={t_next}, dt={float(plan.dt)})"
            )
        _advance_macro_step(execution, t_start=t_previous, t_end=float(t_next))
        t = float(t_next)
        step_count += 1
    # A particle released exactly at t_end exists in the final state but has
    # zero integration age, hence no motion/charge/RNG update.
    activate_release_cursor_until(
        state.release_cursor, state.released, state.active, float(t_end)
    )
    return StepLoopResult(
        t=float(t),
        step_count=int(step_count),
        elapsed_s=float(time.perf_counter() - loop_t0),
    )


def simulate_context(
    context: SolverContext,
    *,
    capture_debug: bool | None = None,
) -> SolverOutcome:
    """Run one fully resolved context without IO or configuration parsing."""

    dim = int(context.spatial_dim)
    plan = context.plan
    capture_outputs = (
        bool(plan.output.is_debug) if capture_debug is None else bool(capture_debug)
    )
    debug_buffers = initialize_debug_buffers(
        plan, capture_outputs=bool(capture_outputs)
    )
    execution = prepare_runtime_execution(
        context,
        spatial_dim=dim,
        plan=plan,
        debug_buffers=debug_buffers,
    )
    loop_result = _run_runtime_step_loop(execution)
    return finalize_runtime_execution(execution, loop_result)


__all__ = ("simulate_context",)
