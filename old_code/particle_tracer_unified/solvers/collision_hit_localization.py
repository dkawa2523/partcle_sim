from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.domain import BoundaryHit

StateAtTime = Callable[[float], tuple[np.ndarray, np.ndarray]]
TimeTolerance = Callable[[float, float, float], float]


@dataclass(frozen=True)
class _PlaneState:
    time: float
    signed_distance: float
    position: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class _PlaneBracket:
    low: _PlaneState
    high: _PlaneState


@dataclass(frozen=True)
class _ContainmentBracket:
    position_low: np.ndarray
    time_low: float
    position_high: np.ndarray
    time_high: float


def _sorted_positive_search_times(
    candidates: list[float],
    segment_dt: float,
) -> np.ndarray:
    return np.asarray(
        sorted(
            {
                float(np.clip(value, 0.0, segment_dt))
                for value in candidates
                if value > 0.0
            }
        ),
        dtype=np.float64,
    )


def _physical_hit_search_times(
    segment_dt: float,
    stage_times: np.ndarray,
    primary_hit_time: float | None,
) -> np.ndarray:
    dt_segment = max(float(segment_dt), 0.0)
    candidates: list[float] = [
        float(value) for value in np.asarray(stage_times, dtype=np.float64)
    ]
    if primary_hit_time is not None and np.isfinite(primary_hit_time):
        hit_time = float(np.clip(primary_hit_time, 0.0, dt_segment))
        if hit_time > 0.0:
            candidates.extend((0.5 * hit_time, hit_time))
            remaining = max(0.0, dt_segment - hit_time)
            if remaining > 0.0:
                candidates.extend(
                    (
                        min(
                            dt_segment,
                            hit_time + max(1.0e-9 * dt_segment, 0.05 * remaining),
                        ),
                        0.5 * (hit_time + dt_segment),
                    )
                )
            candidates.append(dt_segment)
            return _sorted_positive_search_times(candidates, dt_segment)
    candidates.extend(
        float(fraction * dt_segment)
        for fraction in (
            0.125,
            0.25,
            0.375,
            0.5,
            0.625,
            0.75,
            0.875,
            1.0,
        )
    )
    return _sorted_positive_search_times(candidates, dt_segment)


def _primary_plane_event(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    segment_dt: float,
    t_end_segment: float,
    primary_hit: BoundaryHit | None,
    time_roundoff: float,
    state_at: StateAtTime,
    time_tolerance: TimeTolerance,
    on_boundary_tol_m: float,
    max_iters: int,
) -> tuple[tuple[BoundaryHit, np.ndarray, float] | None, float | None]:
    if primary_hit is None:
        return None, None
    primary_hit_time = float(
        np.clip(float(primary_hit.alpha_hint) * segment_dt, 0.0, segment_dt)
    )
    if primary_hit_time <= time_roundoff:
        return None, primary_hit_time
    event = _locate_primary_hit_by_local_plane(
        x0=x0,
        v0=v0,
        segment_dt=segment_dt,
        t_end_segment=t_end_segment,
        primary_hit=primary_hit,
        primary_hit_time=primary_hit_time,
        state_at=state_at,
        time_tolerance=time_tolerance,
        on_boundary_tol_m=on_boundary_tol_m,
        max_iters=min(int(max_iters), 18),
    )
    return event, primary_hit_time


def _find_containment_bracket(
    *,
    x0: np.ndarray,
    search_times: np.ndarray,
    time_roundoff: float,
    state_at: StateAtTime,
    strict_inside_fn: Callable[[np.ndarray], bool],
) -> _ContainmentBracket | None:
    position_low = np.asarray(x0, dtype=np.float64).copy()
    time_low = 0.0
    for candidate_time in search_times:
        if candidate_time <= time_low + time_roundoff:
            continue
        candidate_position, _candidate_velocity = state_at(float(candidate_time))
        candidate_position = np.asarray(candidate_position, dtype=np.float64)
        if strict_inside_fn(candidate_position):
            position_low = candidate_position
            time_low = float(candidate_time)
            continue
        return _ContainmentBracket(
            position_low=position_low,
            time_low=time_low,
            position_high=candidate_position,
            time_high=float(candidate_time),
        )
    return None


def _refine_containment_bracket(
    *,
    bracket: _ContainmentBracket,
    state_at: StateAtTime,
    strict_inside_fn: Callable[[np.ndarray], bool],
    stop_time_tolerance: float,
    stop_position_tolerance: float,
    max_iters: int,
) -> _ContainmentBracket:
    position_low = bracket.position_low
    time_low = bracket.time_low
    position_high = bracket.position_high
    time_high = bracket.time_high
    for _ in range(int(max(1, max_iters))):
        if float(time_high - time_low) <= stop_time_tolerance:
            break
        if (
            stop_position_tolerance > 0.0
            and float(np.linalg.norm(position_high - position_low))
            <= stop_position_tolerance
        ):
            break
        middle_time = 0.5 * (float(time_low) + float(time_high))
        middle_position, _middle_velocity = state_at(middle_time)
        middle_position = np.asarray(middle_position, dtype=np.float64)
        if strict_inside_fn(middle_position):
            position_low = middle_position
            time_low = middle_time
        else:
            position_high = middle_position
            time_high = middle_time
    return _ContainmentBracket(
        position_low=position_low,
        time_low=time_low,
        position_high=position_high,
        time_high=time_high,
    )


def _project_containment_hit(
    *,
    bracket: _ContainmentBracket,
    segment_dt: float,
    state_at: StateAtTime,
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
) -> tuple[BoundaryHit, np.ndarray, float] | None:
    hit_time = 0.5 * (float(bracket.time_low) + float(bracket.time_high))
    hit_position_state, hit_velocity = state_at(hit_time)
    nearest = nearest_projection_fn(
        np.asarray(hit_position_state, dtype=np.float64),
        bracket.position_low,
    )
    if nearest is None:
        nearest = nearest_projection_fn(
            bracket.position_high,
            bracket.position_low,
        )
    if nearest is None:
        return None
    hit_fraction = float(np.clip(hit_time / segment_dt, 0.0, 1.0))
    return (
        BoundaryHit(
            position=np.asarray(nearest.position, dtype=np.float64),
            normal=np.asarray(nearest.normal, dtype=np.float64),
            part_id=int(nearest.part_id),
            alpha_hint=hit_fraction,
            primitive_id=int(nearest.primitive_id),
            primitive_kind=str(nearest.primitive_kind),
            is_ambiguous=bool(nearest.is_ambiguous),
        ),
        np.asarray(hit_velocity, dtype=np.float64),
        hit_time,
    )


def _local_hit_plane(primary_hit: BoundaryHit) -> tuple[np.ndarray, np.ndarray] | None:
    hit_position = np.asarray(primary_hit.position, dtype=np.float64)
    normal = np.asarray(primary_hit.normal, dtype=np.float64)
    normal_magnitude = float(np.linalg.norm(normal))
    if (
        hit_position.ndim != 1
        or normal.ndim != 1
        or hit_position.size != normal.size
        or normal_magnitude <= 1.0e-30
    ):
        return None
    return hit_position, normal / normal_magnitude


def _signed_plane_state(
    state_at: StateAtTime,
    partial_time: float,
    hit_position: np.ndarray,
    unit_normal: np.ndarray,
) -> _PlaneState:
    position, velocity = state_at(float(partial_time))
    position_array = np.asarray(position, dtype=np.float64)
    signed_distance = float(np.dot(position_array - hit_position, unit_normal))
    return _PlaneState(
        time=float(partial_time),
        signed_distance=signed_distance,
        position=position_array,
        velocity=np.asarray(velocity, dtype=np.float64),
    )


def _local_plane_candidate_times(
    primary_hit_time: float,
    dt_segment: float,
) -> tuple[float, ...]:
    guessed_time = float(np.clip(primary_hit_time, 0.0, dt_segment))
    candidates = [guessed_time, dt_segment]
    if guessed_time > 0.0:
        candidates.extend(
            (
                0.5 * guessed_time,
                min(
                    dt_segment,
                    guessed_time + 0.1 * max(0.0, dt_segment - guessed_time),
                ),
            )
        )
    return tuple(
        sorted(
            {
                float(np.clip(value, 0.0, dt_segment))
                for value in candidates
                if value > 0.0
            }
        )
    )


def _find_local_plane_bracket(
    *,
    start: _PlaneState,
    candidate_times: tuple[float, ...],
    state_at: StateAtTime,
    hit_position: np.ndarray,
    unit_normal: np.ndarray,
) -> _PlaneBracket | None:
    low = start
    for candidate_time in candidate_times:
        candidate = _signed_plane_state(
            state_at,
            candidate_time,
            hit_position,
            unit_normal,
        )
        if not np.isfinite(candidate.signed_distance):
            continue
        if (
            low.signed_distance == 0.0
            or low.signed_distance * candidate.signed_distance <= 0.0
        ):
            return _PlaneBracket(low=low, high=candidate)
        low = candidate
    return None


def _plane_bracket_is_resolved(
    bracket: _PlaneBracket,
    time_tolerance: float,
    signed_tolerance: float,
) -> bool:
    return (
        bracket.high.time - bracket.low.time <= time_tolerance
        or min(
            abs(bracket.low.signed_distance),
            abs(bracket.high.signed_distance),
        )
        <= signed_tolerance
    )


def _refine_local_plane_bracket(
    *,
    bracket: _PlaneBracket,
    state_at: StateAtTime,
    hit_position: np.ndarray,
    unit_normal: np.ndarray,
    stop_time_tolerance: float,
    stop_signed_tolerance: float,
    max_iters: int,
) -> _PlaneBracket:
    current = bracket
    for _ in range(int(max(1, max_iters))):
        if _plane_bracket_is_resolved(
            current,
            stop_time_tolerance,
            stop_signed_tolerance,
        ):
            break
        middle_time = 0.5 * (current.low.time + current.high.time)
        middle = _signed_plane_state(
            state_at,
            middle_time,
            hit_position,
            unit_normal,
        )
        if not np.isfinite(middle.signed_distance):
            break
        if (
            current.low.signed_distance == 0.0
            or current.low.signed_distance * middle.signed_distance <= 0.0
        ):
            current = _PlaneBracket(low=current.low, high=middle)
        else:
            current = _PlaneBracket(low=middle, high=current.high)
    return current


def _nearest_plane_state(bracket: _PlaneBracket) -> _PlaneState:
    if abs(bracket.low.signed_distance) <= abs(bracket.high.signed_distance):
        return bracket.low
    return bracket.high


def _refined_primary_hit(
    primary_hit: BoundaryHit,
    position: np.ndarray,
    hit_fraction: float,
) -> BoundaryHit:
    return BoundaryHit(
        position=np.asarray(position, dtype=np.float64),
        normal=np.asarray(primary_hit.normal, dtype=np.float64),
        part_id=int(primary_hit.part_id),
        alpha_hint=hit_fraction,
        primitive_id=int(primary_hit.primitive_id),
        primitive_kind=str(primary_hit.primitive_kind),
        is_ambiguous=bool(primary_hit.is_ambiguous),
    )


def _locate_primary_hit_by_local_plane(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    segment_dt: float,
    t_end_segment: float,
    primary_hit: BoundaryHit,
    primary_hit_time: float,
    state_at: StateAtTime,
    time_tolerance: TimeTolerance,
    on_boundary_tol_m: float,
    max_iters: int,
) -> tuple[BoundaryHit, np.ndarray, float] | None:
    dt_segment = max(float(segment_dt), 0.0)
    if dt_segment <= 0.0:
        return None
    plane = _local_hit_plane(primary_hit)
    if plane is None:
        return None
    hit_position, unit_normal = plane
    signed_start = float(
        np.dot(np.asarray(x0, dtype=np.float64) - hit_position, unit_normal)
    )
    if not np.isfinite(signed_start) or abs(signed_start) <= float(on_boundary_tol_m):
        return None
    start = _PlaneState(
        time=0.0,
        signed_distance=signed_start,
        position=np.asarray(x0, dtype=np.float64),
        velocity=np.asarray(v0, dtype=np.float64),
    )
    bracket = _find_local_plane_bracket(
        start=start,
        candidate_times=_local_plane_candidate_times(primary_hit_time, dt_segment),
        state_at=state_at,
        hit_position=hit_position,
        unit_normal=unit_normal,
    )
    if bracket is None:
        return None
    refined = _refine_local_plane_bracket(
        bracket=bracket,
        state_at=state_at,
        hit_position=hit_position,
        unit_normal=unit_normal,
        stop_time_tolerance=time_tolerance(float(t_end_segment), dt_segment, 1.0e-7),
        stop_signed_tolerance=float(on_boundary_tol_m),
        max_iters=max_iters,
    )
    hit_state = _nearest_plane_state(refined)
    hit_fraction = float(np.clip(hit_state.time / dt_segment, 0.0, 1.0))
    return (
        _refined_primary_hit(primary_hit, hit_state.position, hit_fraction),
        np.asarray(hit_state.velocity, dtype=np.float64),
        float(hit_state.time),
    )


def _project_primary_event_to_finite_boundary(
    event: tuple[BoundaryHit, np.ndarray, float],
    *,
    inside_reference: np.ndarray,
    segment_dt: float,
    t_end_segment: float,
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    time_tolerance: TimeTolerance,
    on_boundary_tol_m: float,
) -> tuple[BoundaryHit, np.ndarray, float] | None:
    hit, velocity, hit_time = event
    physical_position = np.asarray(hit.position, dtype=np.float64)
    projected = nearest_projection_fn(
        physical_position,
        np.asarray(inside_reference, dtype=np.float64),
    )
    if projected is None:
        return None
    projected_position = np.asarray(projected.position, dtype=np.float64)
    speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))
    time_error = time_tolerance(float(t_end_segment), float(segment_dt), 1.0e-7)
    coordinate_scale = max(
        1.0,
        float(np.linalg.norm(physical_position)),
        float(np.linalg.norm(projected_position)),
    )
    spatial_tolerance = max(
        float(on_boundary_tol_m),
        speed * float(time_error),
        64.0 * float(np.spacing(np.float64(coordinate_scale))),
    )
    distance = float(np.linalg.norm(projected_position - physical_position))
    if not np.isfinite(distance) or distance > spatial_tolerance:
        return None
    return (
        BoundaryHit(
            position=projected_position,
            normal=np.asarray(projected.normal, dtype=np.float64),
            part_id=int(projected.part_id),
            alpha_hint=float(hit.alpha_hint),
            primitive_id=int(projected.primitive_id),
            primitive_kind=str(projected.primitive_kind),
            is_ambiguous=bool(projected.is_ambiguous),
        ),
        np.asarray(velocity, dtype=np.float64),
        float(hit_time),
    )


def locate_physical_hit_state(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    segment_dt: float,
    t_end_segment: float,
    stage_times: np.ndarray,
    primary_hit: BoundaryHit | None,
    strict_inside_fn: Callable[[np.ndarray], bool],
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
    state_at: StateAtTime,
    time_tolerance: TimeTolerance,
    on_boundary_tol_m: float,
    max_iters: int = 32,
) -> tuple[BoundaryHit, np.ndarray, float] | None:
    """Locate the first physical hit using only a segment-state evaluator."""

    dt_segment = max(float(segment_dt), 0.0)
    if dt_segment <= 0.0:
        return None

    time_roundoff = time_tolerance(float(t_end_segment), dt_segment, 0.0)
    primary_event, primary_hit_time = _primary_plane_event(
        x0=x0,
        v0=v0,
        segment_dt=dt_segment,
        t_end_segment=float(t_end_segment),
        primary_hit=primary_hit,
        time_roundoff=time_roundoff,
        state_at=state_at,
        time_tolerance=time_tolerance,
        on_boundary_tol_m=float(on_boundary_tol_m),
        max_iters=max_iters,
    )
    if primary_event is not None:
        projected_primary = _project_primary_event_to_finite_boundary(
            primary_event,
            inside_reference=np.asarray(x0, dtype=np.float64),
            segment_dt=dt_segment,
            t_end_segment=float(t_end_segment),
            nearest_projection_fn=nearest_projection_fn,
            time_tolerance=time_tolerance,
            on_boundary_tol_m=float(on_boundary_tol_m),
        )
        if projected_primary is not None:
            return projected_primary
    search_times = _physical_hit_search_times(
        dt_segment,
        stage_times,
        primary_hit_time,
    )
    bracket = _find_containment_bracket(
        x0=x0,
        search_times=search_times,
        time_roundoff=time_roundoff,
        state_at=state_at,
        strict_inside_fn=strict_inside_fn,
    )
    if bracket is None:
        return None
    refined = _refine_containment_bracket(
        bracket=bracket,
        state_at=state_at,
        strict_inside_fn=strict_inside_fn,
        stop_time_tolerance=time_tolerance(
            float(t_end_segment),
            dt_segment,
            1.0e-6,
        ),
        stop_position_tolerance=max(float(on_boundary_tol_m), 0.0),
        max_iters=max_iters,
    )
    return _project_containment_hit(
        bracket=refined,
        segment_dt=dt_segment,
        state_at=state_at,
        nearest_projection_fn=nearest_projection_fn,
    )


__all__ = ("locate_physical_hit_state",)
