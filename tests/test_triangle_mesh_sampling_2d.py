from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_HARD_INVALID,
)
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    build_triangle_candidate_grid,
    locate_triangle_containing_point,
    sample_triangle_mesh_series,
    sample_triangle_mesh_status,
    triangle_mesh_support_tolerance,
)

_VERTICES = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float64,
)
_TRIANGLES = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _mesh_field() -> SimpleNamespace:
    origin, cell_size, shape, offsets, indices = build_triangle_candidate_grid(
        _VERTICES,
        _TRIANGLES,
    )
    return SimpleNamespace(
        mesh_vertices=_VERTICES,
        mesh_triangles=_TRIANGLES,
        accel_origin=origin,
        accel_cell_size=cell_size,
        accel_shape=shape,
        accel_cell_offsets=offsets,
        accel_triangle_indices=indices,
    )


def _locate(position: np.ndarray) -> tuple[int, np.ndarray]:
    field = _mesh_field()
    return locate_triangle_containing_point(
        vertices=field.mesh_vertices,
        triangles=field.mesh_triangles,
        accel_origin=field.accel_origin,
        accel_cell_size=field.accel_cell_size,
        accel_shape=field.accel_shape,
        accel_cell_offsets=field.accel_cell_offsets,
        accel_triangle_indices=field.accel_triangle_indices,
        position=position,
        eps=triangle_mesh_support_tolerance(_VERTICES, _TRIANGLES),
    )


@pytest.mark.parametrize(
    ("vertices", "triangles", "message"),
    [
        (np.zeros(2), np.empty((0, 3)), "vertices must have shape"),
        (_VERTICES, np.zeros(3), "triangles must have shape"),
        (_VERTICES, np.empty((0, 3)), "at least one triangle"),
        (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            np.asarray([[0, 1, 2]]),
            "positive finite span",
        ),
    ],
)
def test_candidate_grid_preserves_validation_order(
    vertices: np.ndarray,
    triangles: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_triangle_candidate_grid(vertices, triangles)


def test_candidate_grid_preserves_cell_and_triangle_order() -> None:
    origin, cell_size, shape, offsets, indices = build_triangle_candidate_grid(
        _VERTICES,
        _TRIANGLES,
    )

    np.testing.assert_array_equal(origin, np.asarray([0.0, 0.0]))
    np.testing.assert_array_equal(cell_size, np.asarray([0.5, 0.5]))
    assert shape == (2, 2)
    assert origin.dtype == np.float64
    assert cell_size.dtype == np.float64
    assert offsets.dtype == np.int32
    assert indices.dtype == np.int32
    np.testing.assert_array_equal(offsets, np.asarray([0, 2, 4, 6, 8]))
    np.testing.assert_array_equal(indices, np.asarray([0, 1] * 4))


def test_support_tolerance_rejects_mesh_without_resolved_edges() -> None:
    vertices = np.zeros((3, 2), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="triangle mesh has no positive edge or altitude",
    ):
        triangle_mesh_support_tolerance(
            vertices,
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )


def test_triangle_location_uses_first_candidate_on_shared_boundary() -> None:
    common = {
        "vertices": _VERTICES,
        "triangles": np.vstack((_TRIANGLES, np.asarray([[0, 0, 1]]))),
        "accel_origin": np.asarray([0.0, 0.0]),
        "accel_cell_size": np.asarray([1.0, 1.0]),
        "accel_shape": (1, 1),
        "accel_cell_offsets": np.asarray([0, 3]),
        "position": np.asarray([0.5, 0.5]),
        "eps": 0.0,
    }

    triangle_index, barycentric = locate_triangle_containing_point(
        **common,
        accel_triangle_indices=np.asarray([1, 0, 2]),
    )

    assert triangle_index == 1
    assert barycentric.dtype == np.float64
    np.testing.assert_array_equal(barycentric, np.asarray([0.5, 0.5, 0.0]))


@given(
    u=st.floats(min_value=0.0, max_value=1.0),
    v=st.floats(min_value=0.0, max_value=1.0),
)
def test_triangle_location_reconstructs_points_from_float64_barycentrics(
    u: float,
    v: float,
) -> None:
    expected = np.asarray([u, (1.0 - u) * v, (1.0 - u) * (1.0 - v)])
    point = expected @ _VERTICES[_TRIANGLES[0]]

    triangle_index, barycentric = _locate(point)

    assert triangle_index >= 0
    assert barycentric.dtype == np.float64
    np.testing.assert_allclose(
        barycentric @ _VERTICES[_TRIANGLES[triangle_index]],
        point,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_triangle_location_returns_float64_zero_for_outside_and_empty_cell() -> None:
    field = _mesh_field()
    common = {
        "vertices": field.mesh_vertices,
        "triangles": field.mesh_triangles,
        "accel_origin": field.accel_origin,
        "accel_cell_size": field.accel_cell_size,
        "accel_shape": field.accel_shape,
        "eps": 0.0,
    }
    triangle_index, barycentric = locate_triangle_containing_point(
        **common,
        position=np.asarray([-1.0, 0.5]),
        accel_cell_offsets=field.accel_cell_offsets,
        accel_triangle_indices=field.accel_triangle_indices,
    )
    assert triangle_index == -1
    assert barycentric.dtype == np.float64
    np.testing.assert_array_equal(barycentric, np.zeros(3))

    triangle_index, barycentric = locate_triangle_containing_point(
        **common,
        position=np.asarray([0.25, 0.25]),
        accel_cell_offsets=np.zeros(5, dtype=np.int32),
        accel_triangle_indices=np.empty(0, dtype=np.int32),
    )
    assert triangle_index == -1
    np.testing.assert_array_equal(barycentric, np.zeros(3))


def test_triangle_series_preserves_spatial_and_time_interpolation_order() -> None:
    field = _mesh_field()
    series = SimpleNamespace(
        times=np.asarray([0.0, 2.0]),
        data=np.asarray([[0.0, 10.0, 20.0, 30.0], [20.0, 30.0, 40.0, 50.0]]),
    )
    position = np.asarray([0.5, 0.25])

    assert sample_triangle_mesh_series(series, field, position, 1.0) == 17.5
    assert (
        sample_triangle_mesh_series(series, field, position, 1.0, mode="nearest") == 7.5
    )
    assert (
        sample_triangle_mesh_series(series, field, position, 1.1, mode="nearest")
        == 27.5
    )
    assert sample_triangle_mesh_series(series, field, position, -1.0) == 7.5
    assert sample_triangle_mesh_series(series, field, position, 3.0) == 27.5

    stationary = SimpleNamespace(times=np.asarray([0.0]), data=series.data[0])
    assert sample_triangle_mesh_series(stationary, field, position, 100.0) == 7.5
    stationary_row = SimpleNamespace(
        times=np.asarray([0.0]),
        data=series.data[:1],
    )
    assert sample_triangle_mesh_series(stationary_row, field, position, 100.0) == 7.5
    # Support classification and value sampling answer different questions.
    # A point outside the mesh is unsupported, but a value query there clamps
    # to the nearest element so a trial step that crosses a wall stays finite
    # for the hit localization that replaces it.
    outside = np.asarray([2.0, 2.0])
    assert sample_triangle_mesh_status(field, outside) == int(
        VALID_MASK_STATUS_HARD_INVALID
    )
    clamped = sample_triangle_mesh_series(series, field, outside, 1.0)
    assert np.isfinite(clamped)
    vertex_values = series.data[:, field.mesh_triangles].reshape(-1)
    assert float(np.min(vertex_values)) <= clamped <= float(np.max(vertex_values))


def test_triangle_series_rejects_nonfinite_time_interval_after_location() -> None:
    field = _mesh_field()
    series = SimpleNamespace(
        times=np.asarray([0.0, np.nan]),
        data=np.asarray([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]),
    )

    with pytest.raises(
        ValueError,
        match="Triangle field times must be finite and strictly increasing",
    ):
        sample_triangle_mesh_series(series, field, np.asarray([0.5, 0.25]), 1.0)
