from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from particle_tracer_unified.core.grid_sampling import (
    locate_axis_interval,
    sample_grid_scalar,
    sample_grid_scalar_points_2d,
)


@given(
    x=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
    y=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
)
def test_batch_grid_sampling_matches_scalar_bilinear_contract(
    x: float,
    y: float,
) -> None:
    xs = np.asarray([0.0, 0.5, 2.0], dtype=np.float64)
    ys = np.asarray([-1.0, 0.25, 1.0], dtype=np.float64)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    data = 3.0 * x_grid - 2.0 * y_grid + 5.0
    point = np.asarray([x, y], dtype=np.float64)

    batch = sample_grid_scalar_points_2d(data, (xs, ys), point.reshape(1, 2))
    scalar = sample_grid_scalar(data, (xs, ys), point)

    assert batch.shape == (1,)
    assert batch.dtype == np.float64
    np.testing.assert_allclose(batch[0], scalar, rtol=0.0, atol=2.0e-15)


def test_batch_grid_sampling_marks_outside_and_nonfinite_points() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    data = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    points = np.asarray(
        [
            [0.5, 0.5],
            [-0.1, 0.5],
            [0.5, 1.1],
            [np.nan, 0.5],
        ],
        dtype=np.float64,
    )

    sampled = sample_grid_scalar_points_2d(data, (axis, axis), points)

    assert sampled[0] == pytest.approx(1.5)
    assert np.isnan(sampled[1:]).all()


def test_batch_grid_sampling_validation_order_is_stable() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    data = np.ones((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        sample_grid_scalar_points_2d(data, (axis,), np.zeros(2))
    with pytest.raises(ValueError, match="exactly two axes"):
        sample_grid_scalar_points_2d(data, (axis,), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="requires a 2D grid"):
        sample_grid_scalar_points_2d(np.ones(2), (axis, axis), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="at least two entries"):
        sample_grid_scalar_points_2d(
            data,
            (np.asarray([0.0]), axis),
            np.zeros((1, 2)),
        )


def test_batch_grid_sampling_reports_x_before_y_axis_failure() -> None:
    data = np.ones((3, 3), dtype=np.float64)
    bad_x = np.asarray([0.0, 1.0, 1.0], dtype=np.float64)
    bad_y = np.asarray([0.0, 2.0, 2.0], dtype=np.float64)
    points = np.asarray([[1.0, 2.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="x-axis entries"):
        sample_grid_scalar_points_2d(data, (bad_x, bad_y), points)


def test_scalar_grid_sampling_defenses_keep_nan_and_dimension_contracts() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    lo, hi, alpha = locate_axis_interval(axis, np.nan)
    assert (lo, hi) == (0, 1)
    assert np.isnan(alpha)

    with pytest.raises(ValueError, match="Axis must be 1D with at least 2 entries"):
        locate_axis_interval(np.asarray([0.0]), 0.0)
    with pytest.raises(ValueError, match="Only 2D/3D sampling is supported"):
        sample_grid_scalar(np.ones(2), (axis,), np.asarray([0.5]))
