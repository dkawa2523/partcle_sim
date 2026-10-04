"""Collision hit localization and conservative fallback resolution."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from particle_tracer_unified.core.coordinate_systems import (
    canonicalize_axisymmetric_rz_state,
)
from particle_tracer_unified.domain import BoundaryHit

from ._collision_types import (
    CollisionPartialMotion,
    CollisionSegmentResolution,
    _CollisionResolutionContext,
)
from .collision_hit_localization import (
    _project_primary_event_to_finite_boundary,
    locate_physical_hit_state,
)
from .diagnostics import increment_count

TimeTolerance = Callable[[float, float, float], float]


def _physical_collision_state(
    context: _CollisionResolutionContext,
    position: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    compiled = getattr(context.inputs, "compiled", None)
    if str(getattr(compiled, "coordinate_system", "cartesian_xy")) == "axisymmetric_rz":
        return canonicalize_axisymmetric_rz_state(position, velocity)
    return (
        np.asarray(position, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
    )


def _classify_primary_hit(
    *,
    context: _CollisionResolutionContext,
    primary_hit: BoundaryHit | None,
    primary_hit_counted: bool,
) -> tuple[BoundaryHit | None, bool]:
    resolved_hit = primary_hit
    hit_counted = bool(primary_hit_counted)
    use_polyline = int(context.stage_points.shape[0]) >= 2
    if resolved_hit is None:
        if use_polyline:
            increment_count(context.collision_diagnostics, "etd2_polyline_checks_count")
        resolved_hit = context.primary_hit_fn(context.x_curr, context.stage_points)
        if resolved_hit is None:
            if use_polyline:
                increment_count(
                    context.collision_diagnostics, "etd2_polyline_fallback_count"
                )
        else:
            if use_polyline:
                increment_count(
                    context.collision_diagnostics, "etd2_polyline_hit_count"
                )
            hit_counted = False
    elif use_polyline and not hit_counted:
        increment_count(context.collision_diagnostics, "etd2_polyline_checks_count")
        increment_count(context.collision_diagnostics, "etd2_polyline_hit_count")
    return resolved_hit, hit_counted


def _resolve_primary_hit(
    *,
    context: _CollisionResolutionContext,
    primary_hit: BoundaryHit | None,
    primary_hit_counted: bool,
) -> BoundaryHit | None:
    resolved_hit, hit_counted = _classify_primary_hit(
        context=context,
        primary_hit=primary_hit,
        primary_hit_counted=bool(primary_hit_counted),
    )
    if resolved_hit is not None and not hit_counted:
        increment_count(context.collision_diagnostics, context.primary_hit_counter_key)
        increment_count(context.collision_diagnostics, "primary_hit_count")
    return resolved_hit


def _stages_remain_inside(context: _CollisionResolutionContext) -> bool:
    return bool(
        np.all(
            np.asarray(
                [bool(context.inside_fn(point)) for point in context.stage_points],
                dtype=bool,
            )
        )
    )


def _localized_collision_resolution(
    *,
    context: _CollisionResolutionContext,
    primary_hit: BoundaryHit | None,
    advance_partial: CollisionPartialMotion,
    time_tolerance: TimeTolerance,
) -> CollisionSegmentResolution | None:
    # The primary-plane and containment searches can request the exact same
    # prefix more than once.  Cache only bit-identical times evaluated through
    # this callback: near-boundary containment may legitimately change one ULP
    # away, and a precomputed endpoint may use a different substep schedule.
    state_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def state_at(partial_time: float) -> tuple[np.ndarray, np.ndarray]:
        query_time = float(partial_time)
        cached = state_cache.get(query_time)
        if cached is not None:
            return cached
        position, velocity = advance_partial(
            inputs=context.inputs,
            x0=context.x_curr,
            v0=context.v_curr,
            dt_partial=query_time,
            segment_dt=float(context.segment_dt),
            t_end_segment=float(context.t),
        )
        resolved = _physical_collision_state(context, position, velocity)
        state_cache[query_time] = resolved
        return resolved

    hit_state = locate_physical_hit_state(
        x0=context.x_curr,
        v0=context.v_curr,
        segment_dt=float(context.segment_dt),
        t_end_segment=float(context.t),
        stage_times=(
            np.arange(1, int(context.stage_points.shape[0]) + 1, dtype=np.float64)
            * (float(context.segment_dt) / float(max(1, context.stage_points.shape[0])))
        ),
        primary_hit=primary_hit,
        strict_inside_fn=context.strict_inside_fn,
        nearest_projection_fn=context.nearest_projection_fn,
        state_at=state_at,
        time_tolerance=time_tolerance,
        on_boundary_tol_m=float(context.on_boundary_tol_m),
    )
    if hit_state is None:
        return None
    hit_event, v_hit, hit_dt = hit_state
    if primary_hit is None:
        increment_count(context.collision_diagnostics, "bisection_fallback_count")
    return CollisionSegmentResolution(
        advance_without_hit=False,
        should_break=False,
        x_next=context.x_next,
        v_next=context.v_next,
        hit_event=hit_event,
        v_hit=np.asarray(v_hit, dtype=np.float64),
        hit_dt=float(hit_dt),
    )


def _direct_primary_hit_resolution(
    *,
    context: _CollisionResolutionContext,
    primary_hit: BoundaryHit | None,
    advance_partial: CollisionPartialMotion,
    time_tolerance: TimeTolerance,
) -> CollisionSegmentResolution | None:
    if primary_hit is None:
        return None
    fallback_dt = float(
        np.clip(float(primary_hit.alpha_hint), 0.0, 1.0) * float(context.segment_dt)
    )
    if fallback_dt <= time_tolerance(float(context.t), context.segment_dt, 0.0):
        return None
    x_at_hit, v_at_hit = advance_partial(
        inputs=context.inputs,
        x0=context.x_curr,
        v0=context.v_curr,
        dt_partial=float(fallback_dt),
        segment_dt=float(context.segment_dt),
        t_end_segment=float(context.t),
    )
    x_at_hit, v_at_hit = _physical_collision_state(context, x_at_hit, v_at_hit)
    if not np.all(np.isfinite(x_at_hit)) or not np.all(np.isfinite(v_at_hit)):
        return None
    projected_event = _project_primary_event_to_finite_boundary(
        (
            BoundaryHit(
                position=np.asarray(x_at_hit, dtype=np.float64),
                normal=np.asarray(primary_hit.normal, dtype=np.float64),
                part_id=int(primary_hit.part_id),
                alpha_hint=float(
                    np.clip(fallback_dt / float(context.segment_dt), 0.0, 1.0)
                ),
                primitive_id=int(primary_hit.primitive_id),
                primitive_kind=str(primary_hit.primitive_kind),
                is_ambiguous=bool(primary_hit.is_ambiguous),
            ),
            np.asarray(v_at_hit, dtype=np.float64),
            float(fallback_dt),
        ),
        inside_reference=context.x_curr,
        segment_dt=float(context.segment_dt),
        t_end_segment=float(context.t),
        nearest_projection_fn=context.nearest_projection_fn,
        time_tolerance=time_tolerance,
        on_boundary_tol_m=float(context.on_boundary_tol_m),
    )
    if projected_event is None:
        return None
    hit_event, v_hit, hit_dt = projected_event
    increment_count(
        context.collision_diagnostics, "primary_hit_direct_resolution_count"
    )
    return CollisionSegmentResolution(
        advance_without_hit=False,
        should_break=False,
        x_next=context.x_next,
        v_next=context.v_next,
        hit_event=hit_event,
        v_hit=np.asarray(v_hit, dtype=np.float64),
        hit_dt=float(hit_dt),
    )


def _unresolved_collision_resolution(
    *,
    context: _CollisionResolutionContext,
    primary_hit: BoundaryHit | None,
    advance_partial: CollisionPartialMotion,
) -> CollisionSegmentResolution:
    increment_count(context.collision_diagnostics, "unresolved_crossing_count")
    fallback_dt = float(context.segment_dt)
    if primary_hit is not None:
        fallback_dt = float(
            np.clip(float(primary_hit.alpha_hint), 0.0, 1.0) * float(context.segment_dt)
        )
    x_fallback, v_fallback = advance_partial(
        inputs=context.inputs,
        x0=context.x_curr,
        v0=context.v_curr,
        dt_partial=float(fallback_dt),
        segment_dt=float(context.segment_dt),
        t_end_segment=float(context.t),
    )
    x_fallback, v_fallback = _physical_collision_state(
        context,
        x_fallback,
        v_fallback,
    )
    nearest_hit = context.nearest_projection_fn(x_fallback, context.x_curr)
    if nearest_hit is None:
        x_next, _v_next = _physical_collision_state(
            context,
            context.x_next,
            context.v_next,
        )
        nearest_hit = context.nearest_projection_fn(x_next, context.x_curr)
    if nearest_hit is None:
        return CollisionSegmentResolution(
            advance_without_hit=False,
            should_break=True,
            x_next=context.x_next,
            v_next=context.v_next,
        )
    increment_count(context.collision_diagnostics, "nearest_projection_fallback_count")
    return CollisionSegmentResolution(
        advance_without_hit=False,
        should_break=False,
        x_next=context.x_next,
        v_next=context.v_next,
        hit_event=nearest_hit,
        v_hit=np.asarray(v_fallback, dtype=np.float64),
        hit_dt=float(fallback_dt),
    )


def resolve_collision_segment(
    *,
    context: _CollisionResolutionContext,
    primary_hit: BoundaryHit | None,
    primary_hit_counted: bool,
    advance_partial: CollisionPartialMotion,
    time_tolerance: TimeTolerance,
) -> CollisionSegmentResolution:
    resolved_primary_hit = _resolve_primary_hit(
        context=context,
        primary_hit=primary_hit,
        primary_hit_counted=bool(primary_hit_counted),
    )
    if resolved_primary_hit is None and _stages_remain_inside(context):
        return CollisionSegmentResolution(
            advance_without_hit=True,
            should_break=False,
            x_next=context.x_next,
            v_next=context.v_next,
        )
    localized = _localized_collision_resolution(
        context=context,
        primary_hit=resolved_primary_hit,
        advance_partial=advance_partial,
        time_tolerance=time_tolerance,
    )
    if localized is not None:
        return localized
    direct = _direct_primary_hit_resolution(
        context=context,
        primary_hit=resolved_primary_hit,
        advance_partial=advance_partial,
        time_tolerance=time_tolerance,
    )
    if direct is not None:
        return direct
    return _unresolved_collision_resolution(
        context=context,
        primary_hit=resolved_primary_hit,
        advance_partial=advance_partial,
    )
