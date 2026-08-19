"""Prepare one collision-search trial from deterministic and stochastic motion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    valid_mask_status_requires_stop,
)
from particle_tracer_unified.domain import BoundaryHit, BoundaryQuery

from ._collision_types import (
    AcceptedMotionTrace,
    CollidingParticleAdvanceResult,
    CollisionSegmentInputs,
    CollisionSegmentTrial,
)
from ._coupled_charge_motion import resolve_coupled_charge_valid_mask_prefix
from ._stochastic_first_passage import search_piecewise_langevin_wall_crossing
from .base_field_sampling import sample_compiled_valid_mask_statuses
from .diagnostics import increment_count
from .segment_motion import (
    SegmentMotionTrace,
    ValidMaskPrefixResolution,
    trace_motion_segment,
)
from .segment_trace import (
    TraceRefinementDecision,
    assess_trace_geometry,
)
from .stochastic_motion import (
    compose_piecewise_langevin_state,
    compose_piecewise_langevin_trace,
)
from .terminal_outcome import terminal_segment_outcome
from .valid_mask_retry import resolve_valid_mask_retry_then_stop


@dataclass(frozen=True, slots=True)
class _CollisionFirstPassageBoundary:
    inside_fn: Callable[[np.ndarray], bool]
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None]
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None]

    def inside(self, point_m: np.ndarray) -> bool:
        return bool(self.inside_fn(point_m))

    def first_hit(
        self,
        start_m: np.ndarray,
        end_m: np.ndarray,
    ) -> BoundaryHit | None:
        return self.primary_hit_fn(
            np.asarray(start_m, dtype=np.float64),
            np.asarray([end_m], dtype=np.float64),
        )

    def nearest_projection(
        self,
        point_m: np.ndarray,
        inside_reference_m: np.ndarray,
    ) -> BoundaryHit | None:
        return self.nearest_projection_fn(point_m, inside_reference_m)


def _advance_segment_with_inputs(
    *,
    inputs: CollisionSegmentInputs,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_segment: float,
    t_end_segment: float,
    adaptive_substep_enabled: int,
    minimum_substeps: int = 1,
):
    request = inputs.request(
        position_m=x0,
        velocity_mps=v0,
        duration_s=float(dt_segment),
        end_time_s=float(t_end_segment),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        minimum_substeps=int(minimum_substeps),
    )
    if inputs.coupled_charge_tracer is None:
        trace = trace_motion_segment(request)
    else:
        trace = inputs.coupled_charge_tracer.trace(
            request,
            charge_start_C=inputs.coupled_charge_start(),
        )
    path = getattr(inputs, "stochastic_path", None)
    if path is None or not bool(trace.local_error_resolved):
        return (
            trace.endpoint_position_m,
            trace.endpoint_velocity_mps,
            trace.substep_count,
            trace.positions_m,
            trace.aggregate_support_status,
            bool(trace.local_error_resolved),
            trace,
        )
    if not isinstance(trace, SegmentMotionTrace):
        raise ValueError(
            "coupled charge with electric force is not supported with stochastic motion"
        )
    composed = compose_piecewise_langevin_trace(
        path=path,
        deterministic_trace=trace,
        stochastic_offset_s=float(inputs.stochastic_offset_s),
    )
    return (
        composed.endpoint_position_m,
        composed.endpoint_velocity_mps,
        trace.substep_count,
        composed.positions_m,
        composed.aggregate_support_status,
        True,
        trace,
    )


def advance_partial_with_inputs(
    *,
    inputs: CollisionSegmentInputs,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_partial: float,
    segment_dt: float,
    t_end_segment: float,
    accepted_substep_count: int,
    accepted_trace=None,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_substeps = max(1, int(accepted_substep_count))
    if accepted_trace is None:
        request = inputs.request(
            position_m=x0,
            velocity_mps=v0,
            duration_s=float(segment_dt),
            end_time_s=float(t_end_segment),
            adaptive_substep_enabled=0,
            minimum_substeps=fixed_substeps,
        )
        if inputs.coupled_charge_tracer is None:
            accepted_trace = trace_motion_segment(request)
        else:
            accepted_trace = inputs.coupled_charge_tracer.trace(
                request,
                charge_start_C=inputs.coupled_charge_start(),
            )
    x_out, v_out = accepted_trace.state_at(float(dt_partial))
    path = inputs.stochastic_path
    if path is None:
        return x_out, v_out
    return compose_piecewise_langevin_state(
        path=path,
        deterministic_position_m=np.asarray(x_out, dtype=np.float64),
        deterministic_velocity_mps=np.asarray(v_out, dtype=np.float64),
        stochastic_offset_s=float(inputs.stochastic_offset_s),
        elapsed_s=float(dt_partial),
    )


def _resolve_valid_mask_retry_with_inputs(
    *,
    inputs: CollisionSegmentInputs,
    collision_diagnostics: dict[str, object],
    x0: np.ndarray,
    v0: np.ndarray,
    dt_segment: float,
    t_end_segment: float,
    adaptive_substep_enabled: int,
) -> ValidMaskPrefixResolution:
    request = inputs.request(
        position_m=x0,
        velocity_mps=v0,
        duration_s=float(dt_segment),
        end_time_s=float(t_end_segment),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
    )
    if inputs.coupled_charge_tracer is not None:
        resolution = resolve_coupled_charge_valid_mask_prefix(
            inputs.coupled_charge_tracer,
            request,
            charge_start_C=inputs.coupled_charge_start(),
        )
        increment_count(
            collision_diagnostics,
            "invalid_mask_retry_count",
            int(resolution.retry_count),
        )
        if not bool(resolution.found_valid_prefix):
            increment_count(collision_diagnostics, "invalid_mask_retry_exhausted_count")
        return resolution
    return resolve_valid_mask_retry_then_stop(
        request,
        collision_diagnostics=collision_diagnostics,
        stochastic_path=inputs.stochastic_path,
        stochastic_offset_s=float(inputs.stochastic_offset_s),
    )


@dataclass(frozen=True)
class _PreparedCollisionMotion:
    x_next: np.ndarray
    v_next: np.ndarray
    stage_points: np.ndarray
    valid_mask_status: int
    primary_hit: BoundaryHit | None
    primary_hit_counted: bool
    refinement_resolved: bool = True
    substep_count: int = 1
    accepted_trace: AcceptedMotionTrace | None = None


def _collision_trace_refinement_decision(
    *,
    segment_start_x: np.ndarray,
    stage_points: np.ndarray,
    inside_fn: Callable[[np.ndarray], bool],
    substep_count: int,
    max_substeps: int,
    curve_tolerance: float,
) -> TraceRefinementDecision:
    stage_inside = np.asarray(
        [bool(inside_fn(point)) for point in stage_points],
        dtype=bool,
    )
    assessment = assess_trace_geometry(
        segment_start_x,
        stage_points,
        on_boundary_tolerance_m=float(curve_tolerance),
    )
    return TraceRefinementDecision(
        geometry_risk=bool(
            np.all(stage_inside) and assessment.requires_refinement(float("nan"))
        ),
        support_substeps=int(substep_count),
        max_substeps=int(max_substeps),
    )


def _record_reintegrated_segment_diagnostics(
    *,
    collision_diagnostics: dict[str, object],
    adaptive_substep_enabled: int,
    substep_count: int,
    max_substeps: int,
) -> None:
    increment_count(collision_diagnostics, "collision_reintegrated_segments_count")
    if int(adaptive_substep_enabled) == 0:
        return
    increment_count(
        collision_diagnostics,
        "adaptive_substep_segments_count",
        int(substep_count),
    )
    if int(substep_count) > 1:
        increment_count(collision_diagnostics, "adaptive_substep_trigger_count")
    if int(substep_count) == int(max_substeps):
        increment_count(
            collision_diagnostics,
            "adaptive_substep_limit_reached_count",
        )


def _search_stochastic_remainder(
    *,
    inputs: CollisionSegmentInputs,
    accepted_trace: SegmentMotionTrace,
    inside_fn: Callable[[np.ndarray], bool],
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    on_boundary_tol_m: float,
) -> tuple[BoundaryHit | None, bool, int]:
    path = inputs.stochastic_path
    if path is None:
        return None, False, int(VALID_MASK_STATUS_CLEAN)
    boundary = _CollisionFirstPassageBoundary(
        inside_fn=inside_fn,
        primary_hit_fn=primary_hit_fn,
        nearest_projection_fn=nearest_projection_fn,
    )
    crossing = search_piecewise_langevin_wall_crossing(
        path=path,
        deterministic_trace=accepted_trace,
        boundary_service=cast(BoundaryQuery[object], boundary),
        geometry_tolerance_m=float(on_boundary_tol_m),
        stochastic_offset_s=float(inputs.stochastic_offset_s),
    )
    support_statuses = sample_compiled_valid_mask_statuses(
        accepted_trace.request.backend,
        crossing.stage_points,
    )
    support_status = (
        int(np.max(support_statuses))
        if support_statuses.size
        else int(VALID_MASK_STATUS_CLEAN)
    )
    return crossing.prefetched_hit, bool(crossing.unresolved), support_status


def _reintegrated_collision_motion(
    *,
    x_curr: np.ndarray,
    v_curr: np.ndarray,
    segment_start_x: np.ndarray,
    t: float,
    segment_dt: float,
    inputs: CollisionSegmentInputs,
    adaptive_substep_enabled: int,
    inside_fn: Callable[[np.ndarray], bool],
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None]
    | None,
    on_boundary_tol_m: float,
    collision_diagnostics: dict[str, object],
) -> _PreparedCollisionMotion:
    # Collision geometry keeps its independent two-leaf safety seed.  LTE
    # controls motion accuracy; this seed only resolves out-and-back paths.
    minimum_substeps = 2 if int(adaptive_substep_enabled) != 0 else 1
    max_substeps = 1 << int(max(0, inputs.adaptive_substep_max_splits))
    primary_hit = None
    refinement_resolved = False
    while True:
        advanced = _advance_segment_with_inputs(
            inputs=inputs,
            x0=x_curr,
            v0=v_curr,
            dt_segment=float(segment_dt),
            t_end_segment=float(t),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            minimum_substeps=int(minimum_substeps),
        )
        (
            x_next,
            v_next,
            substep_count,
            stage_points,
            valid_mask_status,
            local_error_resolved,
        ) = advanced[:6]
        accepted_trace = advanced[6] if len(advanced) > 6 else None
        stage_points_arr = np.asarray(stage_points, dtype=np.float64)
        if not bool(local_error_resolved):
            break
        if (
            getattr(inputs, "stochastic_path", None) is not None
            and nearest_projection_fn is not None
        ):
            if not isinstance(accepted_trace, SegmentMotionTrace):
                raise ValueError(
                    "coupled charge with electric force is not supported with "
                    "stochastic motion"
                )
            primary_hit, unresolved, stochastic_support_status = (
                _search_stochastic_remainder(
                    inputs=inputs,
                    accepted_trace=accepted_trace,
                    inside_fn=inside_fn,
                    primary_hit_fn=primary_hit_fn,
                    nearest_projection_fn=nearest_projection_fn,
                    on_boundary_tol_m=float(on_boundary_tol_m),
                )
            )
            valid_mask_status = max(
                int(valid_mask_status),
                int(stochastic_support_status),
            )
            refinement_resolved = not bool(unresolved)
            break
        primary_hit = primary_hit_fn(segment_start_x, stage_points_arr)
        decision = _collision_trace_refinement_decision(
            segment_start_x=segment_start_x,
            stage_points=stage_points_arr,
            inside_fn=inside_fn,
            substep_count=int(substep_count),
            max_substeps=int(max_substeps),
            curve_tolerance=float(on_boundary_tol_m),
        )
        if decision.resolved(current_substeps=int(substep_count)):
            refinement_resolved = True
            break
        if decision.limit_reached(current_substeps=int(substep_count)):
            break
        minimum_substeps = decision.minimum_substeps(
            current_substeps=int(substep_count)
        )
    _record_reintegrated_segment_diagnostics(
        collision_diagnostics=collision_diagnostics,
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        substep_count=int(substep_count),
        max_substeps=int(max_substeps),
    )
    return _PreparedCollisionMotion(
        x_next=np.asarray(x_next, dtype=np.float64),
        v_next=np.asarray(v_next, dtype=np.float64),
        stage_points=stage_points_arr,
        valid_mask_status=int(valid_mask_status),
        primary_hit=primary_hit,
        primary_hit_counted=False,
        refinement_resolved=bool(refinement_resolved),
        substep_count=int(substep_count),
        accepted_trace=accepted_trace,
    )


def _collision_trial_from_motion(
    *,
    motion: _PreparedCollisionMotion,
    particle_valid_mask_status: int,
) -> CollisionSegmentTrial:
    return CollisionSegmentTrial(
        x_next=motion.x_next,
        v_next=motion.v_next,
        stage_points=motion.stage_points,
        primary_hit=motion.primary_hit,
        primary_hit_counted=bool(motion.primary_hit_counted),
        particle_valid_mask_status=int(particle_valid_mask_status),
        accepted_substep_count=int(motion.substep_count),
        accepted_trace=motion.accepted_trace,
    )


def _unresolved_refinement_collision_trial(
    *,
    motion: _PreparedCollisionMotion,
    segment_start_x: np.ndarray,
    segment_start_v: np.ndarray,
    particle_valid_mask_status: int,
    segment_dt: float,
    collision_diagnostics: dict[str, object],
    charge_start_C: float | None,
) -> CollisionSegmentTrial:
    reason = "trace_refinement_unresolved"
    increment_count(collision_diagnostics, "unresolved_crossing_count")
    return CollisionSegmentTrial(
        x_next=motion.x_next,
        v_next=motion.v_next,
        stage_points=motion.stage_points,
        primary_hit=None,
        primary_hit_counted=False,
        particle_valid_mask_status=int(particle_valid_mask_status),
        terminal_stop_result=CollidingParticleAdvanceResult(
            position=np.asarray(segment_start_x, dtype=np.float64),
            velocity=np.asarray(segment_start_v, dtype=np.float64),
            total_hits=0,
            valid_mask_status=int(particle_valid_mask_status),
            invalid_mask_stopped=False,
            numerical_boundary_stopped=True,
            numerical_boundary_stop_reason=reason,
            terminal_outcome=terminal_segment_outcome(
                accepted_elapsed_s=0.0,
                segment_duration_s=float(segment_dt),
                position=segment_start_x,
                reason=reason,
            ),
            charge_C=charge_start_C,
        ),
    )


def _valid_mask_primary_hit(
    *,
    motion: _PreparedCollisionMotion,
    segment_start_x: np.ndarray,
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
) -> BoundaryHit | None:
    if motion.primary_hit is not None:
        return motion.primary_hit
    try:
        return primary_hit_fn(segment_start_x, motion.stage_points)
    except Exception:
        return None


def _invalid_mask_collision_trial(
    *,
    motion: _PreparedCollisionMotion,
    particle_valid_mask_status: int,
    segment_dt: float,
    resolution: ValidMaskPrefixResolution,
) -> CollisionSegmentTrial:
    reason = (
        "collision_valid_mask_hard_invalid_prefix_clipped"
        if bool(resolution.found_valid_prefix)
        else "collision_valid_mask_hard_invalid_retry_exhausted"
    )
    return CollisionSegmentTrial(
        x_next=motion.x_next,
        v_next=motion.v_next,
        stage_points=motion.stage_points,
        primary_hit=None,
        primary_hit_counted=False,
        particle_valid_mask_status=int(particle_valid_mask_status),
        terminal_stop_result=CollidingParticleAdvanceResult(
            position=resolution.position,
            velocity=resolution.velocity,
            total_hits=0,
            valid_mask_status=int(particle_valid_mask_status),
            invalid_mask_stopped=True,
            invalid_stop_reason=reason,
            terminal_outcome=terminal_segment_outcome(
                accepted_elapsed_s=float(resolution.accepted_dt),
                segment_duration_s=float(segment_dt),
                position=resolution.position,
                reason=reason,
            ),
            charge_C=getattr(resolution, "charge_C", None),
        ),
    )


def _apply_collision_valid_mask_policy(
    *,
    motion: _PreparedCollisionMotion,
    particle_valid_mask_status: int,
    segment_start_x: np.ndarray,
    segment_start_v: np.ndarray,
    t: float,
    segment_dt: float,
    inputs: CollisionSegmentInputs,
    adaptive_substep_enabled: int,
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    collision_diagnostics: dict[str, object],
) -> CollisionSegmentTrial:
    if not bool(valid_mask_status_requires_stop(int(motion.valid_mask_status))):
        return _collision_trial_from_motion(
            motion=motion,
            particle_valid_mask_status=int(particle_valid_mask_status),
        )
    primary_hit = _valid_mask_primary_hit(
        motion=motion,
        segment_start_x=segment_start_x,
        primary_hit_fn=primary_hit_fn,
    )
    if primary_hit is not None:
        return _collision_trial_from_motion(
            motion=replace(
                motion,
                primary_hit=primary_hit,
                primary_hit_counted=False,
            ),
            particle_valid_mask_status=int(particle_valid_mask_status),
        )
    resolution = _resolve_valid_mask_retry_with_inputs(
        inputs=inputs,
        collision_diagnostics=collision_diagnostics,
        x0=segment_start_x,
        v0=segment_start_v,
        dt_segment=float(segment_dt),
        t_end_segment=float(t),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
    )
    return _invalid_mask_collision_trial(
        motion=motion,
        particle_valid_mask_status=int(particle_valid_mask_status),
        segment_dt=float(segment_dt),
        resolution=resolution,
    )


def prepare_collision_segment_trial(
    *,
    use_precomputed_trial: bool,
    x_curr: np.ndarray,
    v_curr: np.ndarray,
    t: float,
    segment_dt: float,
    inputs: CollisionSegmentInputs,
    base_adaptive_substep_enabled: int,
    initial_x_next: np.ndarray,
    initial_v_next: np.ndarray,
    initial_stage_points: np.ndarray,
    initial_valid_mask_status: int,
    initial_primary_hit: BoundaryHit | None,
    initial_primary_hit_counted: bool,
    inside_fn: Callable[[np.ndarray], bool],
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    on_boundary_tol_m: float,
    collision_diagnostics: dict[str, object],
    initial_substep_count: int = 1,
    initial_accepted_trace=None,
    nearest_projection_fn: (
        Callable[[np.ndarray, np.ndarray], BoundaryHit | None] | None
    ) = None,
) -> CollisionSegmentTrial:
    segment_start_x = np.asarray(x_curr, dtype=np.float64).copy()
    segment_start_v = np.asarray(v_curr, dtype=np.float64).copy()
    if bool(use_precomputed_trial):
        motion = _PreparedCollisionMotion(
            x_next=np.asarray(initial_x_next, dtype=np.float64),
            v_next=np.asarray(initial_v_next, dtype=np.float64),
            stage_points=np.asarray(initial_stage_points, dtype=np.float64),
            valid_mask_status=int(initial_valid_mask_status),
            primary_hit=initial_primary_hit,
            primary_hit_counted=bool(initial_primary_hit_counted),
            substep_count=max(1, int(initial_substep_count)),
            accepted_trace=initial_accepted_trace,
        )
    else:
        motion = _reintegrated_collision_motion(
            x_curr=x_curr,
            v_curr=v_curr,
            segment_start_x=segment_start_x,
            t=float(t),
            segment_dt=float(segment_dt),
            inputs=inputs,
            adaptive_substep_enabled=int(base_adaptive_substep_enabled),
            inside_fn=inside_fn,
            primary_hit_fn=primary_hit_fn,
            nearest_projection_fn=nearest_projection_fn,
            on_boundary_tol_m=float(on_boundary_tol_m),
            collision_diagnostics=collision_diagnostics,
        )
    particle_valid_mask_status = max(
        int(initial_valid_mask_status), int(motion.valid_mask_status)
    )
    if not bool(motion.refinement_resolved):
        return _unresolved_refinement_collision_trial(
            motion=motion,
            segment_start_x=segment_start_x,
            segment_start_v=segment_start_v,
            particle_valid_mask_status=int(particle_valid_mask_status),
            segment_dt=float(segment_dt),
            collision_diagnostics=collision_diagnostics,
            charge_start_C=getattr(inputs, "charge_start_C", None),
        )
    return _apply_collision_valid_mask_policy(
        motion=motion,
        particle_valid_mask_status=int(particle_valid_mask_status),
        segment_start_x=segment_start_x,
        segment_start_v=segment_start_v,
        t=float(t),
        segment_dt=float(segment_dt),
        inputs=inputs,
        adaptive_substep_enabled=int(base_adaptive_substep_enabled),
        primary_hit_fn=primary_hit_fn,
        collision_diagnostics=collision_diagnostics,
    )
