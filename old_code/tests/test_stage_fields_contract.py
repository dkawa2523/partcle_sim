from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    QuantitySeriesND,
    RegularFieldND,
)
from particle_tracer_unified.core.field_backend import (
    VALID_MASK_QUANTITY,
    ProviderSamplingBackend,
)
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
    choose_velocity_quantity_names,
    point_within_axes,
    sample_quantity_series,
)
from particle_tracer_unified.domain import FieldRequest, sample_one


def _provider(valid_mask: np.ndarray) -> FieldProviderND:
    axes = (
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    values = np.asarray([[[2.0, 4.0], [6.0, 8.0]]], dtype=np.float64)
    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        valid_mask=np.asarray(valid_mask, dtype=bool),
        quantities={
            "ux": QuantitySeriesND(
                name="ux",
                unit="m/s",
                times=np.asarray([0.0], dtype=np.float64),
                data=values,
            )
        },
    )
    return FieldProviderND(field=field)


def _sample(provider: FieldProviderND, point: np.ndarray):
    return sample_one(
        ProviderSamplingBackend(provider),
        point,
        0.0,
        FieldRequest(("ux", VALID_MASK_QUANTITY)),
    )


def test_stage_fields_marks_clean_samples_valid() -> None:
    sample = _sample(_provider(np.ones((2, 2), dtype=bool)), np.asarray([0.5, 0.5]))

    assert int(sample.values[VALID_MASK_QUANTITY][0]) == VALID_MASK_STATUS_CLEAN
    assert sample.supported.tolist() == [True]
    assert sample.metadata["valid_mask_reason"] == ("clean",)
    assert sample.metadata["cell_id"].tolist() == [0]
    assert sample.values["ux"][0] == pytest.approx(5.0)


def test_stage_fields_reports_mixed_stencil_without_zero_filling() -> None:
    valid_mask = np.asarray([[True, False], [True, True]], dtype=bool)

    sample = _sample(_provider(valid_mask), np.asarray([0.0, 0.0]))

    assert int(sample.values[VALID_MASK_QUANTITY][0]) == VALID_MASK_STATUS_MIXED_STENCIL
    assert sample.supported.tolist() == [False]
    assert sample.metadata["valid_mask_reason"] == ("mixed_stencil",)
    assert sample.values["ux"][0] == pytest.approx(2.0)


def test_stage_fields_reports_hard_invalid_as_nan() -> None:
    sample = _sample(_provider(np.ones((2, 2), dtype=bool)), np.asarray([2.0, 2.0]))

    assert int(sample.values[VALID_MASK_QUANTITY][0]) == VALID_MASK_STATUS_HARD_INVALID
    assert sample.supported.tolist() == [False]
    assert sample.metadata["valid_mask_reason"] == ("hard_invalid",)
    assert np.isnan(sample.values["ux"][0])


def test_provider_batch_preserves_request_order_dtypes_and_missing_metadata() -> None:
    sampled = ProviderSamplingBackend(_provider(np.ones((2, 2), dtype=bool))).sample(
        np.asarray([[0.5, 0.5], [2.0, 2.0]], dtype=np.float32),
        np.float32(0.0),
        FieldRequest(("missing", VALID_MASK_QUANTITY, "ux")),
    )

    assert tuple(sampled.values) == ("missing", VALID_MASK_QUANTITY, "ux")
    assert sampled.points_m.dtype == np.float64
    assert sampled.values["missing"].dtype == np.float64
    assert sampled.values[VALID_MASK_QUANTITY].dtype == np.float64
    assert sampled.values["ux"].dtype == np.float64
    assert sampled.supported.dtype == np.bool_
    assert sampled.supported.tolist() == [True, False]
    assert sampled.metadata["valid_mask_reason"] == ("clean", "hard_invalid")
    assert sampled.metadata["cell_id"].tolist() == [0, -1]
    assert sampled.metadata["missing_quantities"] == ("missing",)
    np.testing.assert_allclose(sampled.values["ux"], [5.0, np.nan], equal_nan=True)


@pytest.mark.parametrize(
    ("points", "time_s", "message"),
    [
        (np.asarray([np.nan, 0.0]), np.inf, r"shape \(particle, 2\)"),
        (
            np.asarray([[np.nan, 0.0]], dtype=np.float64),
            np.inf,
            "finite coordinates",
        ),
        (np.asarray([[0.0, 0.0]], dtype=np.float64), np.inf, "time_s must be finite"),
    ],
)
def test_provider_batch_validation_order_is_stable(
    points: np.ndarray,
    time_s: float,
    message: str,
) -> None:
    backend = ProviderSamplingBackend(_provider(np.ones((2, 2), dtype=bool)))

    with pytest.raises(ValueError, match=message):
        backend.sample(points, time_s, FieldRequest(("ux",)))


def test_provider_sampling_rejects_non_linear_interpolation() -> None:
    with pytest.raises(ValueError, match="interpolation must be 'linear'"):
        ProviderSamplingBackend(
            _provider(np.ones((2, 2), dtype=bool)),
            interpolation="nearest",
        )


def test_field_sampling_defenses_preserve_endpoint_and_validation_order() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    axes = (axis, axis)
    assert not point_within_axes(axes, np.asarray([0.5], dtype=np.float64))
    assert not point_within_axes(axes, np.asarray([np.nan, 0.5], dtype=np.float64))
    with pytest.raises(ValueError, match="Axis must be 1D with at least 2 entries"):
        point_within_axes((np.asarray([0.0]),), np.asarray([0.0]))
    assert choose_velocity_quantity_names(None, 2) == ()

    series = SimpleNamespace(
        times=np.asarray([0.0, 1.0], dtype=np.float64),
        data=np.asarray(
            [
                [[0.0, 10.0], [20.0, 30.0]],
                [[100.0, 110.0], [120.0, 130.0]],
            ],
            dtype=np.float64,
        ),
    )
    point = np.asarray([0.5, 0.5], dtype=np.float64)
    assert sample_quantity_series(series, axes, point, 2.0) == pytest.approx(115.0)
    assert sample_quantity_series(
        series, axes, point, 0.75, mode="nearest"
    ) == pytest.approx(115.0)
