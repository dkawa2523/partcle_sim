from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from particle_tracer_unified.core.boundary_core import points_inside_geometry_2d
from particle_tracer_unified.core.geometry2d import (
    _edge_crosses_positive_ray_2d,
    _point_on_edge_2d,
    decode_boundary_loops_2d,
    points_inside_boundary_edges_2d_with_boundary,
)


def _square_edges(scale: float = 1.0) -> np.ndarray:
    return float(scale) * np.asarray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )


def _runtime(
    *,
    scale: float = 1.0,
    boundary_edges: np.ndarray | None,
    boundary_loops_2d: tuple[np.ndarray, ...] = (),
    sdf_value: float = 1.0,
) -> SimpleNamespace:
    axis = float(scale) * np.asarray([0.0, 1.0], dtype=np.float64)
    geometry = SimpleNamespace(
        axes=(axis, axis),
        boundary_edges=boundary_edges,
        boundary_loops_2d=boundary_loops_2d,
        sdf=np.full((2, 2), sdf_value, dtype=np.float64),
    )
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=geometry),
        field_provider=None,
    )


def _python(function: Any) -> Any:
    return getattr(function, "py_func", function)


@given(
    scale=st.floats(
        min_value=1.0e-12,
        max_value=1.0e6,
        allow_nan=False,
        allow_infinity=False,
    )
)
@settings(max_examples=32, deadline=None)
def test_edge_containment_is_scale_invariant_at_corners_and_tolerance(
    scale: float,
) -> None:
    tolerance = 1.0e-7 * scale
    points = scale * np.asarray(
        [
            [-0.5e-7, 0.5],
            [-2.0e-7, 0.5],
            [0.0, 0.0],
            [0.5, 0.5],
            [1.2, 0.5],
        ],
        dtype=np.float64,
    )

    inside, on_boundary = points_inside_geometry_2d(
        _runtime(scale=scale, boundary_edges=_square_edges(scale)),
        points,
        on_boundary_tol_m=tolerance,
        return_on_boundary=True,
    )

    assert inside.dtype == on_boundary.dtype == np.dtype(bool)
    assert inside.shape == on_boundary.shape == (5,)
    assert inside.tolist() == [True, False, True, True, False]
    assert on_boundary.tolist() == [True, False, True, False, False]


def test_explicit_edges_take_precedence_over_loops_and_sdf() -> None:
    distant_loop = np.asarray(
        [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]],
        dtype=np.float64,
    )
    runtime = _runtime(
        boundary_edges=_square_edges(),
        boundary_loops_2d=(distant_loop,),
        sdf_value=1.0,
    )

    inside, on_boundary = points_inside_geometry_2d(
        runtime,
        np.asarray([[0.5, 0.5], [0.0, 0.5]], dtype=np.float64),
        on_boundary_tol_m=1.0e-9,
        return_on_boundary=True,
    )

    assert inside.tolist() == [True, True]
    assert on_boundary.tolist() == [False, True]


def test_loops_and_sdf_remain_ordered_fallbacks() -> None:
    loop = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.asarray([[0.25, 0.25], [1.1, 0.5]], dtype=np.float64)
    loop_result = points_inside_geometry_2d(
        _runtime(boundary_edges=None, boundary_loops_2d=(loop,), sdf_value=1.0),
        points,
        on_boundary_tol_m=0.0,
    )
    sdf_result, sdf_boundary = points_inside_geometry_2d(
        _runtime(boundary_edges=None, sdf_value=-1.0),
        points,
        on_boundary_tol_m=0.0,
        return_on_boundary=True,
    )

    assert isinstance(loop_result, np.ndarray)
    assert loop_result.tolist() == [True, False]
    assert sdf_result.tolist() == [True, False]
    assert not np.any(sdf_boundary)


def test_empty_and_degenerate_edge_queries_preserve_boolean_shapes() -> None:
    empty = np.empty((0, 2), dtype=np.float64)
    runtime_inside, runtime_boundary = points_inside_geometry_2d(
        _runtime(boundary_edges=_square_edges()),
        empty,
        on_boundary_tol_m=1.0e-9,
        return_on_boundary=True,
    )
    degenerate_inside, degenerate_boundary = (
        points_inside_boundary_edges_2d_with_boundary(
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            np.asarray([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float64),
            on_edge_tol=1.0,
        )
    )

    assert runtime_inside.shape == runtime_boundary.shape == (0,)
    assert runtime_inside.dtype == runtime_boundary.dtype == np.dtype(bool)
    assert degenerate_inside.tolist() == [False]
    assert degenerate_boundary.tolist() == [False]


def test_geometry_query_rejects_non_point_matrices_before_runtime_access() -> None:
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        points_inside_geometry_2d(
            SimpleNamespace(),
            np.asarray([0.5, 0.5], dtype=np.float64),
            on_boundary_tol_m=0.0,
        )


def test_edge_classification_primitives_preserve_projection_and_ray_rules() -> None:
    point_on_edge = _python(_point_on_edge_2d)
    crosses_ray = _python(_edge_crosses_positive_ray_2d)

    assert point_on_edge(-0.05, 0.05, 0.0, 0.0, 1.0, 0.0, 1.0, 0.01)
    assert point_on_edge(1.05, 0.05, 0.0, 0.0, 1.0, 0.0, 1.0, 0.01)
    assert not point_on_edge(0.5, 0.2, 0.0, 0.0, 1.0, 0.0, 1.0, 0.01)
    assert crosses_ray(0.25, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert not crosses_ray(0.75, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert not crosses_ray(0.25, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def test_boundary_loop_decoding_rejects_invalid_shapes_and_skips_empty_slices() -> None:
    assert decode_boundary_loops_2d(None, None) == ()
    assert decode_boundary_loops_2d(np.zeros(2), np.asarray([0, 2])) == ()
    decoded = decode_boundary_loops_2d(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.asarray([0, 2, 2, 3]),
    )

    assert len(decoded) == 2
    np.testing.assert_array_equal(decoded[0], [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(decoded[1], [[2.0, 0.0]])
