from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from particle_tracer_unified.core._triangle_queries import _closest_point_on_triangle
from particle_tracer_unified.domain import BoundaryHit
from particle_tracer_unified.solvers.collision_hit_localization import (
    _locate_primary_hit_by_local_plane,
)

_RIGHT_TRIANGLE = np.asarray(
    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    dtype=np.float64,
)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ([-1.0, -1.0, 1.0], [0.0, 0.0, 0.0]),
        ([3.0, -1.0, 1.0], [2.0, 0.0, 0.0]),
        ([-1.0, 3.0, 1.0], [0.0, 2.0, 0.0]),
        ([1.0, -1.0, 1.0], [1.0, 0.0, 0.0]),
        ([-1.0, 1.0, 1.0], [0.0, 1.0, 0.0]),
        ([2.0, 2.0, 1.0], [1.0, 1.0, 0.0]),
        ([0.5, 0.5, 1.0], [0.5, 0.5, 0.0]),
    ],
    ids=("vertex-a", "vertex-b", "vertex-c", "edge-ab", "edge-ac", "edge-bc", "face"),
)
def test_closest_point_classifies_triangle_voronoi_regions(
    point: list[float],
    expected: list[float],
) -> None:
    closest = _closest_point_on_triangle(
        np.asarray(point, dtype=np.float64),
        _RIGHT_TRIANGLE,
    )

    assert closest.dtype == np.float64
    assert closest.shape == (3,)
    np.testing.assert_array_equal(closest, np.asarray(expected, dtype=np.float64))


@pytest.mark.parametrize(
    ("triangle", "point", "expected"),
    [
        (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
        (
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [1.5, 1.0, 0.0],
            [1.5, 0.0, 0.0],
        ),
    ],
    ids=("collapsed-to-one-point", "collinear"),
)
def test_closest_point_preserves_resolved_degenerate_boundary_results(
    triangle: list[list[float]],
    point: list[float],
    expected: list[float],
) -> None:
    closest = _closest_point_on_triangle(
        np.asarray(point, dtype=np.float64),
        np.asarray(triangle, dtype=np.float64),
    )

    np.testing.assert_array_equal(closest, np.asarray(expected, dtype=np.float64))


def test_closest_point_preserves_degenerate_edge_error() -> None:
    triangle = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError, match="closest-point edge denominator is unresolved"
    ):
        _closest_point_on_triangle(
            np.asarray([0.2, 1.0, 0.0], dtype=np.float64),
            triangle,
        )


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_closest_point_rejects_nonfinite_triangle_geometry(nonfinite: float) -> None:
    triangle = _RIGHT_TRIANGLE.copy()
    triangle[1, 0] = nonfinite

    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(
            ValueError,
            match="closest-point query requires finite resolved triangle geometry",
        ),
    ):
        _closest_point_on_triangle(
            np.asarray([0.25, 0.25, 1.0], dtype=np.float64),
            triangle,
        )


@pytest.mark.parametrize(
    ("nonfinite", "message"),
    [
        (np.nan, "closest-point face denominator is unresolved"),
        (np.inf, "closest-point query requires finite resolved triangle geometry"),
        (-np.inf, "closest-point query requires finite resolved triangle geometry"),
    ],
)
def test_closest_point_preserves_nonfinite_query_error(
    nonfinite: float,
    message: str,
) -> None:
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(
            ValueError,
            match=message,
        ),
    ):
        _closest_point_on_triangle(
            np.asarray([nonfinite, 0.25, 1.0], dtype=np.float64),
            _RIGHT_TRIANGLE,
        )


def test_closest_point_uses_scaled_terms_when_raw_dot_products_overflow() -> None:
    triangle = np.asarray(
        [
            [1.0e308, 0.0, 0.0],
            [1.0e308, 1.0e308, 0.0],
            [0.0, 1.0e308, 0.0],
        ],
        dtype=np.float64,
    )
    point = np.asarray([5.0e307, 5.0e307, 1.0], dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        closest = _closest_point_on_triangle(point, triangle)

    np.testing.assert_array_equal(
        closest,
        np.asarray([5.0e307, 5.0e307, 0.0], dtype=np.float64),
    )


def _time_tolerance(
    reference_time_s: float, interval_s: float, fraction: float
) -> float:
    magnitude = max(abs(float(reference_time_s)), abs(float(interval_s)))
    roundoff = 64.0 * abs(float(np.spacing(np.float64(magnitude))))
    return max(float(fraction) * float(interval_s), roundoff)


def _primary_hit(normal: np.ndarray) -> BoundaryHit:
    return BoundaryHit(
        position=np.asarray([0.36, 9.0], dtype=np.float64),
        normal=normal,
        part_id=7,
        alpha_hint=0.5,
        primitive_id=3,
        primitive_kind="edge",
        is_ambiguous=True,
    )


def _quadratic_state(partial_time: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([partial_time * partial_time, 3.0], dtype=np.float64),
        np.asarray([2.0 * partial_time, -4.0], dtype=np.float64),
    )


@pytest.mark.parametrize(
    "normal",
    [
        np.asarray([2.0, 0.0], dtype=np.float64),
        np.asarray([-2.0, 0.0], dtype=np.float64),
    ],
    ids=("forward-normal", "reversed-normal"),
)
def test_local_plane_refines_time_but_preserves_primary_geometry(
    normal: np.ndarray,
) -> None:
    primary = _primary_hit(normal)

    event = _locate_primary_hit_by_local_plane(
        x0=np.asarray([0.0, 3.0], dtype=np.float64),
        v0=np.asarray([0.0, -4.0], dtype=np.float64),
        segment_dt=1.0,
        t_end_segment=1.0,
        primary_hit=primary,
        primary_hit_time=0.5,
        state_at=_quadratic_state,
        time_tolerance=_time_tolerance,
        on_boundary_tol_m=1.0e-12,
        max_iters=32,
    )

    assert event is not None
    hit, velocity, hit_time = event
    assert hit_time == pytest.approx(0.6, abs=1.0e-7)
    assert hit.position.dtype == np.float64
    assert hit.normal.dtype == np.float64
    np.testing.assert_allclose(
        hit.position,
        [hit_time * hit_time, 3.0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(hit.normal, normal)
    np.testing.assert_allclose(velocity, [2.0 * hit_time, -4.0], rtol=0.0, atol=0.0)
    assert hit.alpha_hint == pytest.approx(hit_time)
    assert hit.part_id == 7
    assert hit.primitive_id == 3
    assert hit.primitive_kind == "edge"
    assert hit.is_ambiguous is True


def test_local_plane_skips_nonfinite_sample_and_keeps_refining() -> None:
    primary = _primary_hit(np.asarray([1.0, 0.0], dtype=np.float64))

    def state_with_one_nonfinite_sample(
        partial_time: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if partial_time == 0.5:
            return (
                np.asarray([np.nan, 3.0], dtype=np.float64),
                np.asarray([np.nan, -4.0], dtype=np.float64),
            )
        return _quadratic_state(partial_time)

    event = _locate_primary_hit_by_local_plane(
        x0=np.asarray([0.0, 3.0], dtype=np.float64),
        v0=np.asarray([0.0, -4.0], dtype=np.float64),
        segment_dt=1.0,
        t_end_segment=1.0,
        primary_hit=primary,
        primary_hit_time=0.5,
        state_at=state_with_one_nonfinite_sample,
        time_tolerance=_time_tolerance,
        on_boundary_tol_m=1.0e-12,
        max_iters=32,
    )

    assert event is not None
    _, velocity, hit_time = event
    assert hit_time == pytest.approx(0.6, abs=1.0e-7)
    np.testing.assert_allclose(velocity, [2.0 * hit_time, -4.0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("normal", "state_at", "on_boundary_tol_m"),
    [
        (
            np.asarray([0.0, 0.0], dtype=np.float64),
            _quadratic_state,
            1.0e-12,
        ),
        (
            np.asarray([np.nan, 0.0], dtype=np.float64),
            _quadratic_state,
            1.0e-12,
        ),
        (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            _quadratic_state,
            1.0e-12,
        ),
        (
            np.asarray([1.0, 0.0], dtype=np.float64),
            lambda partial_time: (
                np.asarray([0.1 * partial_time, 3.0], dtype=np.float64),
                np.asarray([0.1, 0.0], dtype=np.float64),
            ),
            1.0e-12,
        ),
        (
            np.asarray([1.0, 0.0], dtype=np.float64),
            _quadratic_state,
            0.37,
        ),
    ],
    ids=(
        "zero-normal",
        "nonfinite-normal",
        "shape-mismatch",
        "no-crossing",
        "start-on-boundary",
    ),
)
def test_local_plane_returns_none_when_no_resolved_bracket_exists(
    normal: np.ndarray,
    state_at: Callable[[float], tuple[np.ndarray, np.ndarray]],
    on_boundary_tol_m: float,
) -> None:
    event = _locate_primary_hit_by_local_plane(
        x0=np.asarray([0.0, 3.0], dtype=np.float64),
        v0=np.asarray([0.0, -4.0], dtype=np.float64),
        segment_dt=1.0,
        t_end_segment=1.0,
        primary_hit=_primary_hit(normal),
        primary_hit_time=0.5,
        state_at=state_at,
        time_tolerance=_time_tolerance,
        on_boundary_tol_m=on_boundary_tol_m,
        max_iters=32,
    )

    assert event is None


def test_local_plane_rejects_empty_segment_before_sampling_state() -> None:
    state_calls = 0

    def state_at(partial_time: float) -> tuple[np.ndarray, np.ndarray]:
        nonlocal state_calls
        state_calls += 1
        return _quadratic_state(partial_time)

    event = _locate_primary_hit_by_local_plane(
        x0=np.asarray([0.0, 3.0], dtype=np.float64),
        v0=np.asarray([0.0, -4.0], dtype=np.float64),
        segment_dt=0.0,
        t_end_segment=0.0,
        primary_hit=_primary_hit(np.asarray([1.0, 0.0], dtype=np.float64)),
        primary_hit_time=0.0,
        state_at=state_at,
        time_tolerance=_time_tolerance,
        on_boundary_tol_m=1.0e-12,
        max_iters=32,
    )

    assert event is None
    assert state_calls == 0
