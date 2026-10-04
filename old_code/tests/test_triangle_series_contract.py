from __future__ import annotations

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import (
    QuantitySeriesND,
    TriangleMeshField2D,
)
from particle_tracer_unified.solvers.triangle_derived_fields import (
    _series_contract,
    triangle_sample_error,
    triangle_series_gradient_at_location,
    triangle_series_time_derivative_at_location,
    triangle_series_value_at_location,
    validate_triangle_gradient_geometry,
)


def _field(vertex_count: int = 3) -> TriangleMeshField2D:
    return TriangleMeshField2D(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        mesh_vertices=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64
        )[:vertex_count],
        mesh_triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        quantities={},
        accel_origin=np.zeros(2, dtype=np.float64),
        accel_cell_size=np.ones(2, dtype=np.float64),
        accel_shape=(1, 1),
        accel_cell_offsets=np.asarray([0, 1], dtype=np.int32),
        accel_triangle_indices=np.asarray([0], dtype=np.int32),
    )


def _contract(series: QuantitySeriesND) -> tuple[np.ndarray, np.ndarray]:
    return _series_contract(
        series,
        _field(),
        "temperature",
        row_index=7,
        triangle_index=0,
    )


@pytest.mark.parametrize(
    ("data", "times", "expected_shape"),
    [
        ([1, 2, 3], [0], (3,)),
        ([[1, 2, 3], [4, 5, 6]], [0, 1], (2, 3)),
    ],
)
def test_series_contract_normalizes_supported_shapes_to_float64(
    data: list[int] | list[list[int]],
    times: list[int],
    expected_shape: tuple[int, ...],
) -> None:
    values, time_axis = _contract(
        QuantitySeriesND(
            name="T",
            unit="K",
            data=np.asarray(data, dtype=np.int16),
            times=np.asarray(times, dtype=np.float32),
        )
    )

    assert values.shape == expected_shape
    assert values.dtype == np.float64
    assert time_axis.shape == (len(times),)
    assert time_axis.dtype == np.float64


@pytest.mark.parametrize(
    ("data", "times", "message"),
    [
        (
            np.zeros((2, 2)),
            np.asarray([[np.nan, np.nan]]),
            "data must have shape",
        ),
        (
            np.zeros((2, 3)),
            np.asarray([[np.nan, np.nan]]),
            "time axis does not match its 2 data rows",
        ),
        (
            np.zeros((2, 3)),
            np.asarray([0.0, np.nan]),
            "time axis contains non-finite values",
        ),
        (
            np.zeros((2, 3)),
            np.asarray([1.0, 0.0]),
            "unresolved float64 time interval",
        ),
    ],
)
def test_series_contract_validation_order_and_context_are_stable(
    data: np.ndarray,
    times: np.ndarray,
    message: str,
) -> None:
    series = QuantitySeriesND(name="T", unit="K", data=data, times=times)

    with pytest.raises(
        ValueError,
        match=rf"semantic quantity 'temperature' at row 7, triangle 0:.*{message}",
    ):
        _contract(series)


def test_series_contract_rejects_exact_roundoff_boundary_but_accepts_next_step() -> (
    None
):
    spacing = float(np.spacing(np.float64(1.0)))
    data = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="unresolved float64 time interval"):
        _contract(
            QuantitySeriesND(
                name="T",
                unit="K",
                data=data,
                times=np.asarray([1.0, 1.0 + 64.0 * spacing]),
            )
        )

    _, times = _contract(
        QuantitySeriesND(
            name="T",
            unit="K",
            data=data,
            times=np.asarray([1.0, 1.0 + 65.0 * spacing]),
        )
    )
    assert times[1] == 1.0 + 65.0 * spacing


def test_triangle_series_internal_time_tie_uses_value_and_left_derivative() -> None:
    series = QuantitySeriesND(
        name="T",
        unit="K",
        times=np.asarray([0.0, 1.0, 2.0]),
        data=np.asarray([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [30.0, 30.0, 30.0]]),
    )
    barycentric = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)

    value = triangle_series_value_at_location(
        series,
        _field(),
        0,
        barycentric,
        1.0,
        semantic_quantity="temperature",
        row_index=7,
    )
    derivative = triangle_series_time_derivative_at_location(
        series,
        _field(),
        0,
        barycentric,
        1.0,
        semantic_quantity="temperature",
        row_index=7,
    )

    assert value == pytest.approx(10.0)
    assert derivative == pytest.approx(10.0)


def test_triangle_sample_error_omits_location_when_none_is_available() -> None:
    error = triangle_sample_error(
        "temperature",
        "invalid",
        row_index=None,
        triangle_index=None,
    )

    assert str(error) == "triangle field semantic quantity 'temperature': invalid"


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        (
            np.asarray([[0.0, 0.0], [np.nan, 0.0], [0.0, 1.0]]),
            "vertices contain non-finite coordinates",
        ),
        (np.zeros((3, 2)), "no resolvable positive edge scale"),
    ],
)
def test_triangle_gradient_geometry_rejects_invalid_vertices(
    vertices: np.ndarray,
    message: str,
) -> None:
    field = _field()
    object.__setattr__(field, "mesh_vertices", vertices)

    with pytest.raises(ValueError, match=message):
        validate_triangle_gradient_geometry(field, "temperature")


def test_triangle_series_rejects_nonfinite_sample_and_interpolated_value() -> None:
    series = QuantitySeriesND(
        name="T",
        unit="K",
        times=np.asarray([0.0]),
        data=np.asarray([1.0, 2.0, 3.0]),
    )

    with pytest.raises(ValueError, match="sample time is non-finite"):
        triangle_series_value_at_location(
            series,
            _field(),
            0,
            np.asarray([1.0, 0.0, 0.0]),
            np.nan,
            semantic_quantity="temperature",
            row_index=7,
        )
    with pytest.raises(ValueError, match="non-finite interpolated value"):
        triangle_series_value_at_location(
            series,
            _field(),
            0,
            np.asarray([np.nan, 0.0, 0.0]),
            0.0,
            semantic_quantity="temperature",
            row_index=7,
        )


def test_triangle_series_steady_derivative_and_nonfinite_diagnostics() -> None:
    steady = QuantitySeriesND(
        name="T",
        unit="K",
        times=np.asarray([0.0]),
        data=np.asarray([1.0, 2.0, 3.0]),
    )
    transient = QuantitySeriesND(
        name="T",
        unit="K",
        times=np.asarray([0.0, 1.0]),
        data=np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
    )

    assert (
        triangle_series_time_derivative_at_location(
            steady,
            _field(),
            0,
            np.asarray([1.0, 0.0, 0.0]),
            0.0,
            semantic_quantity="temperature",
            row_index=7,
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="sample time is non-finite"):
        triangle_series_time_derivative_at_location(
            transient,
            _field(),
            0,
            np.asarray([1.0, 0.0, 0.0]),
            np.nan,
            semantic_quantity="temperature",
            row_index=7,
        )
    with pytest.raises(ValueError, match="non-finite time derivative"):
        triangle_series_time_derivative_at_location(
            transient,
            _field(),
            0,
            np.asarray([np.nan, 0.0, 0.0]),
            -1.0,
            semantic_quantity="temperature",
            row_index=7,
        )


def test_triangle_series_rejects_nonfinite_gradient_result() -> None:
    maximum = np.finfo(np.float64).max
    series = QuantitySeriesND(
        name="T",
        unit="K",
        times=np.asarray([0.0]),
        data=np.asarray([-maximum, maximum, -maximum]),
    )

    with pytest.raises(ValueError, match="non-finite spatial gradient"):
        triangle_series_gradient_at_location(
            series,
            _field(),
            0,
            0.0,
            semantic_quantity="temperature",
            row_index=7,
        )
