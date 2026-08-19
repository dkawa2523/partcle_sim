from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from particle_tracer_unified.domain import BoundaryHit
from particle_tracer_unified.solvers import _collision_resolution
from particle_tracer_unified.solvers._collision_resolution import (
    _classify_primary_hit,
    _direct_primary_hit_resolution,
    resolve_collision_segment,
)
from particle_tracer_unified.solvers._collision_types import (
    CollisionSegmentInputs,
    _CollisionResolutionContext,
)


def _hit(*, alpha: float = 0.5) -> BoundaryHit:
    return BoundaryHit(
        position=np.asarray([0.5, 0.0]),
        normal=np.asarray([1.0, 0.0]),
        part_id=7,
        alpha_hint=float(alpha),
        primitive_id=3,
        primitive_kind="edge",
    )


def _context(
    *,
    inside: bool,
    primary_hit_fn=lambda _start, _points: None,
    nearest_projection_fn=lambda _point, _inside: None,
) -> _CollisionResolutionContext:
    return _CollisionResolutionContext(
        x_curr=np.asarray([0.0, 0.0]),
        v_curr=np.asarray([1.0, 0.0]),
        x_next=np.asarray([1.0, 0.0]),
        v_next=np.asarray([1.0, 0.0]),
        stage_points=np.asarray([[0.5, 0.0], [1.0, 0.0]]),
        inside_fn=lambda _point: bool(inside),
        strict_inside_fn=lambda _point: bool(inside),
        primary_hit_fn=primary_hit_fn,
        nearest_projection_fn=nearest_projection_fn,
        primary_hit_counter_key="edge_hit_count",
        collision_diagnostics={
            "etd2_polyline_checks_count": 0,
            "etd2_polyline_fallback_count": 0,
            "etd2_polyline_hit_count": 0,
            "edge_hit_count": 0,
            "primary_hit_count": 0,
            "bisection_fallback_count": 0,
            "primary_hit_direct_resolution_count": 0,
            "unresolved_crossing_count": 0,
            "nearest_projection_fallback_count": 0,
        },
        t=1.0,
        segment_dt=1.0,
        inputs=cast(CollisionSegmentInputs, SimpleNamespace()),
        on_boundary_tol_m=1.0e-10,
    )


def _state_at(
    *,
    inputs: CollisionSegmentInputs,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_partial: float,
    segment_dt: float,
    t_end_segment: float,
) -> tuple[np.ndarray, np.ndarray]:
    del inputs, segment_dt, t_end_segment
    return x0 + float(dt_partial) * v0, v0.copy()


def _nonfinite_state_at(
    *,
    inputs: CollisionSegmentInputs,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_partial: float,
    segment_dt: float,
    t_end_segment: float,
) -> tuple[np.ndarray, np.ndarray]:
    del inputs, dt_partial, segment_dt, t_end_segment
    return x0.copy(), np.full_like(v0, np.nan)


def _no_localization_tolerance(_reference: float, interval: float, _fraction: float):
    return float(interval)


def _localization_stops_but_direct_hit_remains(
    _reference: float, interval: float, fraction: float
) -> float:
    return float(interval) if fraction > 0.0 else 0.0


def test_resolution_advances_when_the_trace_remains_inside() -> None:
    context = _context(inside=True)

    result = resolve_collision_segment(
        context=context,
        primary_hit=None,
        primary_hit_counted=False,
        advance_partial=_state_at,
        time_tolerance=lambda _reference, interval, fraction: fraction * interval,
    )

    assert result.advance_without_hit is True
    assert context.collision_diagnostics["etd2_polyline_checks_count"] == 1
    assert context.collision_diagnostics["etd2_polyline_fallback_count"] == 1


def test_localization_reuses_only_bit_identical_prefix_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def counted_state_at(
        *,
        inputs: CollisionSegmentInputs,
        x0: np.ndarray,
        v0: np.ndarray,
        dt_partial: float,
        segment_dt: float,
        t_end_segment: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append(float(dt_partial))
        return _state_at(
            inputs=inputs,
            x0=x0,
            v0=v0,
            dt_partial=float(dt_partial),
            segment_dt=float(segment_dt),
            t_end_segment=float(t_end_segment),
        )

    def probe_repeated_times(**kwargs: object):
        state_at = cast(
            Callable[[float], tuple[np.ndarray, np.ndarray]],
            kwargs["state_at"],
        )
        first_position, first_velocity = state_at(0.25)
        repeated_position, repeated_velocity = state_at(0.25)
        nearby_position, _nearby_velocity = state_at(0.25 + 5.0e-13)
        endpoint_position, endpoint_velocity = state_at(1.0)
        repeated_endpoint_position, repeated_endpoint_velocity = state_at(1.0)
        np.testing.assert_array_equal(repeated_position, first_position)
        np.testing.assert_array_equal(repeated_velocity, first_velocity)
        assert not np.array_equal(nearby_position, first_position)
        np.testing.assert_array_equal(endpoint_position, [1.0, 0.0])
        np.testing.assert_array_equal(endpoint_velocity, [1.0, 0.0])
        np.testing.assert_array_equal(repeated_endpoint_position, endpoint_position)
        np.testing.assert_array_equal(repeated_endpoint_velocity, endpoint_velocity)
        return _hit(alpha=0.25), first_velocity, 0.25

    monkeypatch.setattr(
        _collision_resolution,
        "locate_physical_hit_state",
        probe_repeated_times,
    )
    result = _collision_resolution._localized_collision_resolution(
        context=_context(inside=False),
        primary_hit=_hit(alpha=0.25),
        advance_partial=counted_state_at,
        time_tolerance=lambda _reference, _interval, _fraction: 1.0e-12,
    )

    assert result is not None
    assert calls == [0.25, 0.25 + 5.0e-13, 1.0]


def test_polyline_hit_discovery_records_both_hit_counters() -> None:
    hit = _hit()
    context = _context(inside=True, primary_hit_fn=lambda _start, _points: hit)

    resolved, counted = _classify_primary_hit(
        context=context, primary_hit=None, primary_hit_counted=False
    )

    assert resolved is hit
    assert counted is False
    assert context.collision_diagnostics["etd2_polyline_checks_count"] == 1
    assert context.collision_diagnostics["etd2_polyline_hit_count"] == 1


def test_resolution_uses_direct_primary_hit_when_localization_has_no_bracket() -> None:
    primary = _hit()
    context = _context(inside=True)

    result = resolve_collision_segment(
        context=context,
        primary_hit=primary,
        primary_hit_counted=False,
        advance_partial=_nonfinite_state_at,
        time_tolerance=_localization_stops_but_direct_hit_remains,
    )

    assert result.should_break is True
    assert result.hit_event is None
    assert context.collision_diagnostics["primary_hit_direct_resolution_count"] == 0
    assert context.collision_diagnostics["edge_hit_count"] == 1


def test_unresolved_resolution_uses_nearest_projection_fallback() -> None:
    primary = _hit(alpha=0.0)
    projected = _hit(alpha=0.25)
    context = _context(
        inside=True,
        nearest_projection_fn=lambda _point, _inside: projected,
    )

    result = resolve_collision_segment(
        context=context,
        primary_hit=primary,
        primary_hit_counted=True,
        advance_partial=_state_at,
        time_tolerance=_no_localization_tolerance,
    )

    assert result.hit_event is projected
    assert result.hit_dt == 0.0
    assert context.collision_diagnostics["unresolved_crossing_count"] == 1
    assert context.collision_diagnostics["nearest_projection_fallback_count"] == 1


def test_unresolved_resolution_stops_without_a_projection() -> None:
    context = _context(inside=False)

    result = resolve_collision_segment(
        context=context,
        primary_hit=None,
        primary_hit_counted=False,
        advance_partial=_state_at,
        time_tolerance=_no_localization_tolerance,
    )

    assert result.should_break is True
    assert result.hit_event is None
    assert context.collision_diagnostics["unresolved_crossing_count"] == 1


def test_direct_primary_fallback_keeps_position_and_velocity_at_one_time() -> None:
    primary = BoundaryHit(
        position=np.asarray([0.25, 9.0]),
        normal=np.asarray([1.0, 0.0]),
        part_id=7,
        alpha_hint=0.5,
        primitive_id=3,
        primitive_kind="edge",
    )
    projected = BoundaryHit(
        position=np.asarray([0.25, 2.0]),
        normal=np.asarray([0.0, 1.0]),
        part_id=11,
        alpha_hint=0.0,
        primitive_id=5,
        primitive_kind="edge",
    )
    context = _context(
        inside=False,
        nearest_projection_fn=lambda _point, _inside: projected,
    )

    def curved_state(**kwargs) -> tuple[np.ndarray, np.ndarray]:
        partial_time = float(kwargs["dt_partial"])
        return (
            np.asarray([partial_time * partial_time, 2.0]),
            np.asarray([2.0 * partial_time, -3.0]),
        )

    result = _direct_primary_hit_resolution(
        context=context,
        primary_hit=primary,
        advance_partial=curved_state,
        time_tolerance=lambda _reference, _interval, _fraction: 0.0,
    )

    assert result is not None
    assert result.hit_event is not None
    np.testing.assert_array_equal(result.hit_event.position, [0.25, 2.0])
    assert result.hit_event.part_id == 11
    assert result.hit_event.primitive_id == 5
    np.testing.assert_array_equal(result.v_hit, [1.0, -3.0])
    assert result.hit_dt == 0.5


def test_direct_primary_fallback_rejects_a_physical_point_off_the_primitive() -> None:
    context = _context(
        inside=False,
        nearest_projection_fn=lambda _point, _inside: BoundaryHit(
            position=np.asarray([0.25, 9.0]),
            normal=np.asarray([1.0, 0.0]),
            part_id=7,
            alpha_hint=0.5,
            primitive_id=3,
            primitive_kind="edge",
        ),
    )

    result = _direct_primary_hit_resolution(
        context=context,
        primary_hit=_hit(),
        advance_partial=lambda **_kwargs: (
            np.asarray([0.25, 2.0]),
            np.asarray([1.0, -3.0]),
        ),
        time_tolerance=lambda _reference, _interval, _fraction: 0.0,
    )

    assert result is None
    assert context.collision_diagnostics["primary_hit_direct_resolution_count"] == 0
