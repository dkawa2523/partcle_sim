from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from particle_tracer_unified.core import (
    _boundary_contact_2d,
    _boundary_hits_2d,
    _boundary_hits_3d,
    boundary_hits,
)
from particle_tracer_unified.core._boundary_hits_2d import (
    _batch_edge_intersection_2d,
    _boundary_edges_2d,
    _bounds_overlap_2d,
    _empty_edge_hit_batch,
    _first_batch_edge_intersection_2d,
    _segment_hits_from_boundary_edges_batch,
    _segment_hits_from_boundary_edges_batch_kernel,
    _segment_parameters_are_bounded,
)
from particle_tracer_unified.core.boundary_hits import (
    contact_frame_on_boundary_edge_2d,
    nearest_boundary_edge_features_2d,
    nearest_hit_on_boundary_edges,
    nearest_hit_on_boundary_triangles,
    polyline_hit_from_boundary_edges,
    polyline_hit_from_boundary_triangles,
    polyline_hits_from_boundary_edges_batch,
    segment_hit_from_boundary_edges,
    segment_hit_from_boundary_triangles,
)
from particle_tracer_unified.core.geometry3d import build_triangle_surface


def _square_runtime() -> SimpleNamespace:
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    geometry = SimpleNamespace(
        spatial_dim=2,
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([10, 20, 30, 40], dtype=np.int32),
    )
    return SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geometry))


def _python_implementation(function: Any) -> Any:
    return getattr(function, "py_func", function)


def _assert_array_tuple_equal(actual: tuple, expected: tuple) -> None:
    assert len(actual) == len(expected)
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_boundary_hit_facade_directly_reexports_owner_objects() -> None:
    owners = {
        "BoundaryEdgeFrame2D": _boundary_contact_2d,
        "contact_frame_on_boundary_edge_2d": _boundary_contact_2d,
        "nearest_boundary_edge_features_2d": _boundary_hits_2d,
        "nearest_hit_on_boundary_edges": _boundary_contact_2d,
        "nearest_hit_on_boundary_triangles": _boundary_hits_3d,
        "normalize_polyline_alpha": _boundary_hits_3d,
        "polyline_hit_from_boundary_edges": _boundary_hits_3d,
        "polyline_hit_from_boundary_triangles": _boundary_hits_3d,
        "polyline_hits_from_boundary_edges_batch": _boundary_hits_2d,
        "segment_hit_from_boundary_edges": _boundary_hits_2d,
        "segment_hit_from_boundary_triangles": _boundary_hits_3d,
    }

    for name, owner in owners.items():
        assert getattr(boundary_hits, name) is getattr(owner, name)


def test_numba_edge_helpers_match_their_python_implementations() -> None:
    runtime = _square_runtime()
    edges, part_ids = _boundary_edges_2d(runtime)
    assert edges is not None
    start = np.asarray([0.5, 0.25], dtype=np.float64)
    end = np.asarray([1.5, 0.25], dtype=np.float64)
    direction = end - start
    direction_length_squared = float(np.dot(direction, direction))
    segment_bounds = (0.5, 1.5, 0.25, 0.25)

    overlap_args = (*segment_bounds, 1.0, 1.0, 0.0, 1.0)
    assert _python_implementation(_bounds_overlap_2d)(*overlap_args) is bool(
        _bounds_overlap_2d(*overlap_args)
    )
    assert _python_implementation(_segment_parameters_are_bounded)(0.5, 0.25) is bool(
        _segment_parameters_are_bounded(0.5, 0.25)
    )

    intersection_args = (
        edges,
        1,
        float(start[0]),
        float(start[1]),
        float(direction[0]),
        float(direction[1]),
        direction_length_squared,
        segment_bounds,
        0.0,
        0.0,
    )
    python_intersection = _python_implementation(_batch_edge_intersection_2d)(
        *intersection_args
    )
    compiled_intersection = _batch_edge_intersection_2d(*intersection_args)
    np.testing.assert_allclose(
        python_intersection,
        compiled_intersection,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    first_args = (edges, start, end, 0.0, 0.0)
    python_first = _python_implementation(_first_batch_edge_intersection_2d)(
        *first_args
    )
    compiled_first = _first_batch_edge_intersection_2d(*first_args)
    np.testing.assert_allclose(
        python_first,
        compiled_first,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    python_empty = _python_implementation(_empty_edge_hit_batch)(2)
    compiled_empty = _empty_edge_hit_batch(2)
    _assert_array_tuple_equal(python_empty, compiled_empty)

    kernel_args = (
        edges,
        part_ids,
        start[None, :],
        end[None, :],
        0.0,
        0.0,
    )
    python_batch = _python_implementation(
        _segment_hits_from_boundary_edges_batch_kernel
    )(*kernel_args)
    compiled_batch = _segment_hits_from_boundary_edges_batch_kernel(*kernel_args)
    _assert_array_tuple_equal(python_batch, compiled_batch)


def test_numba_edge_rejections_match_their_python_implementations() -> None:
    edges, _part_ids = _boundary_edges_2d(_square_runtime())
    assert edges is not None
    degenerate = np.zeros((1, 2, 2), dtype=np.float64)
    rejection_cases = (
        (edges, 0, 0.0, 0.5, 1.0, 0.0, 1.0, (0.0, 1.0, 0.5, 0.5), 0.0, 0.0),
        (degenerate, 0, 0.0, 0.0, 1.0, 0.0, 1.0, (0.0, 1.0, 0.0, 0.0), 0.0, 0.0),
        (edges, 0, 0.0, 0.0, 1.0, 0.0, 1.0, (0.0, 1.0, 0.0, 0.0), 0.0, 0.0),
        (edges, 1, 0.0, 0.5, 0.25, 0.0, 0.0625, (0.0, 2.0, 0.0, 1.0), 0.0, 0.0),
    )

    for arguments in rejection_cases:
        python_result = _python_implementation(_batch_edge_intersection_2d)(*arguments)
        compiled_result = _batch_edge_intersection_2d(*arguments)
        assert python_result[0] is compiled_result[0] is False
        np.testing.assert_allclose(
            python_result[1:],
            compiled_result[1:],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    zero_length_arguments = (
        edges,
        np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.5]),
        0.0,
        0.0,
    )
    python_result = _python_implementation(_first_batch_edge_intersection_2d)(
        *zero_length_arguments
    )
    compiled_result = _first_batch_edge_intersection_2d(*zero_length_arguments)
    np.testing.assert_allclose(
        python_result,
        compiled_result,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


@settings(max_examples=24, deadline=None)
@given(
    y=st.floats(
        min_value=0.05,
        max_value=0.95,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
    end_x=st.floats(
        min_value=1.05,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
def test_scalar_and_batch_edge_hits_preserve_numeric_and_dtype_contract(
    y: float,
    end_x: float,
) -> None:
    runtime = _square_runtime()
    start = np.asarray([0.5, y], dtype=np.float64)
    end = np.asarray([end_x, y], dtype=np.float64)
    scalar = segment_hit_from_boundary_edges(runtime, start, end)
    segments, part_ids = _boundary_edges_2d(runtime)

    assert scalar is not None
    assert segments is not None
    batch = _segment_hits_from_boundary_edges_batch(
        segments,
        part_ids,
        start[None, :],
        end[None, :],
    )
    mask, positions, normals, parts, alphas, primitive_ids, ambiguous = batch

    assert mask.dtype == np.bool_
    assert positions.dtype == normals.dtype == alphas.dtype == np.float64
    assert parts.dtype == primitive_ids.dtype == np.int32
    assert ambiguous.dtype == np.bool_
    assert positions.shape == normals.shape == (1, 2)
    assert mask.tolist() == [True]
    np.testing.assert_array_max_ulp(positions[0], scalar.position, maxulp=2)
    np.testing.assert_array_max_ulp(normals[0], scalar.normal, maxulp=2)
    assert alphas[0] == pytest.approx(scalar.alpha_hint, rel=2.0e-15)
    assert parts[0] == scalar.part_id == 20
    assert primitive_ids[0] == scalar.primitive_id == 1
    assert bool(ambiguous[0]) is scalar.is_ambiguous is False

    polyline = polyline_hits_from_boundary_edges_batch(
        runtime,
        start[None, :],
        end[None, None, :],
        particle_indices=np.asarray([73], dtype=np.int64),
    )[73]
    np.testing.assert_array_max_ulp(polyline.position, scalar.position, maxulp=2)
    assert polyline.part_id == scalar.part_id
    assert polyline.primitive_id == scalar.primitive_id


def test_corner_hit_preserves_first_edge_and_ambiguity_metadata() -> None:
    runtime = _square_runtime()

    hit = segment_hit_from_boundary_edges(
        runtime,
        np.asarray([0.5, 0.5], dtype=np.float64),
        np.asarray([1.5, -0.5], dtype=np.float64),
    )

    assert hit is not None
    np.testing.assert_array_equal(hit.position, [1.0, 0.0])
    np.testing.assert_array_equal(hit.normal, [0.0, -1.0])
    assert (hit.part_id, hit.primitive_id, hit.primitive_kind) == (10, 0, "edge")
    assert hit.alpha_hint == 0.5
    assert hit.is_ambiguous is True


def test_nearest_edge_features_preserve_first_edge_ties_and_output_dtypes() -> None:
    part_ids, distances = nearest_boundary_edge_features_2d(
        _square_runtime(),
        np.asarray([[0.5, 0.5]], dtype=np.float64),
    )

    assert part_ids.dtype == np.int32
    assert distances.dtype == np.float64
    assert part_ids.tolist() == [10]
    np.testing.assert_array_equal(distances, [0.5])


def test_batch_polyline_validation_preserves_starts_before_stage_error_order() -> None:
    with pytest.raises(
        ValueError,
        match=r"2D batch boundary hit starts require shape \(n, 2\)",
    ):
        polyline_hits_from_boundary_edges_batch(
            _square_runtime(),
            np.zeros((1, 3), dtype=np.float64),
            np.zeros((2, 4), dtype=np.float64),
        )


def test_missing_geometry_returns_the_empty_public_hit_contracts() -> None:
    runtime = SimpleNamespace(geometry_provider=None)
    point = np.zeros(2, dtype=np.float64)
    stages = np.zeros((1, 1, 2), dtype=np.float64)

    assert segment_hit_from_boundary_edges(runtime, point, point) is None
    assert contact_frame_on_boundary_edge_2d(runtime, point) is None
    assert nearest_hit_on_boundary_edges(runtime, point, point) is None
    assert polyline_hit_from_boundary_edges(runtime, point, stages[0]) is None
    assert (
        polyline_hits_from_boundary_edges_batch(
            runtime,
            point[None, :],
            stages,
        )
        == {}
    )
    part_ids, distances = nearest_boundary_edge_features_2d(
        runtime,
        point[None, :],
    )
    assert part_ids.dtype == np.int32
    assert distances.dtype == np.float64
    assert part_ids.tolist() == [0]
    assert np.isnan(distances).all()


def test_public_edge_queries_validate_batch_shape_and_finite_contact_point() -> None:
    runtime = _square_runtime()
    starts = np.zeros((1, 2), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match=r"2D batch boundary hit stage_points require shape \(n, k, 2\)",
    ):
        polyline_hits_from_boundary_edges_batch(
            runtime,
            starts,
            np.zeros((1, 3), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="particle_indices length must match starts"):
        polyline_hits_from_boundary_edges_batch(
            runtime,
            starts,
            np.zeros((1, 1, 2), dtype=np.float64),
            particle_indices=np.zeros(2, dtype=np.int64),
        )
    with pytest.raises(
        ValueError,
        match=r"2D nearest-boundary diagnostics require shape \(n, 2\)",
    ):
        nearest_boundary_edge_features_2d(runtime, np.zeros((1, 3)))
    assert (
        contact_frame_on_boundary_edge_2d(
            runtime,
            np.asarray([np.nan, 0.0]),
        )
        is None
    )


def test_empty_scalar_polylines_and_missing_triangle_surface_have_no_hit() -> None:
    empty_2d = np.empty((0, 2), dtype=np.float64)
    empty_3d = np.empty((0, 3), dtype=np.float64)

    assert (
        polyline_hit_from_boundary_edges(
            _square_runtime(),
            np.zeros(2),
            empty_2d,
        )
        is None
    )
    assert segment_hit_from_boundary_triangles(None, np.zeros(3), np.ones(3)) is None
    assert nearest_hit_on_boundary_triangles(None, np.zeros(3), np.ones(3)) is None
    assert (
        polyline_hit_from_boundary_triangles(
            None,
            np.zeros(3),
            empty_3d,
        )
        is None
    )


def test_contact_frame_unknown_part_falls_back_with_orientation() -> None:
    frame = contact_frame_on_boundary_edge_2d(
        _square_runtime(),
        np.asarray([1.2, 0.25], dtype=np.float64),
        part_id_hint=999,
        normal_hint=np.asarray([1.0, 0.0], dtype=np.float64),
    )

    assert frame is not None
    assert frame.edge_index == 1
    assert frame.part_id == 20
    assert frame.projection.dtype == frame.normal.dtype == np.float64
    np.testing.assert_allclose(frame.projection, [1.0, 0.25], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(frame.normal, [1.0, 0.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(frame.tangent, [0.0, 1.0], rtol=0.0, atol=0.0)
    assert frame.alpha == 0.25
    assert frame.length == 1.0
    assert frame.distance == pytest.approx(0.2)


def test_contact_orientation_and_projection_preserve_selected_edge() -> None:
    runtime = _square_runtime()
    bottom = contact_frame_on_boundary_edge_2d(
        runtime,
        np.asarray([0.5, 0.2], dtype=np.float64),
        part_id_hint=10,
    )
    right = contact_frame_on_boundary_edge_2d(
        runtime,
        np.asarray([1.2, 0.25], dtype=np.float64),
        part_id_hint=20,
    )
    projection = nearest_hit_on_boundary_edges(
        runtime,
        np.asarray([1.2, 0.25], dtype=np.float64),
        np.asarray([0.5, 0.5], dtype=np.float64),
    )

    assert bottom is not None
    assert right is not None
    assert projection is not None
    np.testing.assert_array_equal(bottom.normal, [0.0, -1.0])
    np.testing.assert_array_equal(right.normal, [-1.0, 0.0])
    assert (projection.part_id, projection.primitive_id) == (20, 1)
    np.testing.assert_array_equal(projection.position, [1.0, 0.25])
    np.testing.assert_array_equal(projection.normal, [1.0, 0.0])


def test_contact_frame_skips_degenerate_edges_before_part_hint_fallback() -> None:
    runtime = _square_runtime()
    geometry = runtime.geometry_provider.geometry
    geometry.boundary_edges = np.concatenate(
        (np.zeros((1, 2, 2), dtype=np.float64), geometry.boundary_edges),
        axis=0,
    )
    geometry.boundary_edge_part_ids = np.asarray(
        [99, 10, 20, 30, 40],
        dtype=np.int32,
    )

    frame = contact_frame_on_boundary_edge_2d(
        runtime,
        np.asarray([1.2, 0.25], dtype=np.float64),
        part_id_hint=99,
    )

    assert frame is not None
    assert (frame.part_id, frame.edge_index) == (20, 2)


def test_nearest_edge_projection_keeps_invalid_edge_error_order() -> None:
    runtime = _square_runtime()
    geometry = runtime.geometry_provider.geometry
    geometry.boundary_edges = np.concatenate(
        (
            np.zeros((1, 2, 2), dtype=np.float64),
            geometry.boundary_edges,
        ),
        axis=0,
    )
    geometry.boundary_edge_part_ids = np.asarray(
        [99, 10, 20, 30, 40],
        dtype=np.int32,
    )

    with pytest.raises(
        ValueError,
        match="Boundary edge 0 must have finite positive length",
    ):
        nearest_hit_on_boundary_edges(
            runtime,
            np.asarray([0.5, 0.5], dtype=np.float64),
            np.asarray([0.5, 0.5], dtype=np.float64),
        )


def test_triangle_segment_polyline_and_projection_metadata_are_stable() -> None:
    triangles = np.asarray(
        [
            [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]],
            [[3.0, -1.0, -1.0], [3.0, 1.0, -1.0], [3.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    surface = build_triangle_surface(
        triangles,
        np.asarray([7, 8], dtype=np.int32),
        validate_closed=False,
    )
    start = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    end = np.asarray([2.0, 0.0, 0.0], dtype=np.float64)

    segment = segment_hit_from_boundary_triangles(surface, start, end)
    polyline = polyline_hit_from_boundary_triangles(
        surface,
        start,
        np.asarray([[0.5, 0.0, 0.0], end], dtype=np.float64),
    )
    projection = nearest_hit_on_boundary_triangles(
        surface,
        np.asarray([1.2, 0.0, 0.0], dtype=np.float64),
        start,
    )

    assert segment is not None
    assert polyline is not None
    assert projection is not None
    for hit in (segment, polyline, projection):
        assert hit.position.dtype == hit.normal.dtype == np.float64
        assert hit.part_id == 7
        assert hit.primitive_id == 0
        assert hit.is_ambiguous is False
        np.testing.assert_allclose(hit.position, [1.0, 0.0, 0.0], atol=1.0e-15)
        np.testing.assert_allclose(hit.normal, [1.0, 0.0, 0.0], atol=1.0e-15)
    assert segment.primitive_kind == polyline.primitive_kind == "triangle"
    assert segment.alpha_hint == 0.5
    assert polyline.alpha_hint == pytest.approx(2.0 / 3.0)
    assert projection.primitive_kind == "triangle_projection"
    assert projection.alpha_hint == 0.0


def test_triangle_queries_preserve_first_primitive_on_exact_ties() -> None:
    repeated = [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    triangles = np.asarray(
        [
            repeated,
            repeated,
            [[3.0, -1.0, -1.0], [3.0, 1.0, -1.0], [3.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    surface = build_triangle_surface(
        triangles,
        np.asarray([7, 8, 9], dtype=np.int32),
        validate_closed=False,
    )
    inside = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    segment = segment_hit_from_boundary_triangles(
        surface,
        inside,
        np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
    )
    projection = nearest_hit_on_boundary_triangles(
        surface,
        np.asarray([1.2, 0.0, 0.0], dtype=np.float64),
        inside,
    )

    assert segment is not None
    assert projection is not None
    assert (segment.part_id, segment.primitive_id) == (7, 0)
    assert (projection.part_id, projection.primitive_id) == (7, 0)
