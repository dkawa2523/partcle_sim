from __future__ import annotations

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import FieldProviderND, QuantitySeriesND, RegularFieldND
from particle_tracer_unified.core.field_backend import sample_field_quantity_with_status
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)


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


def test_field_sample_marks_clean_samples_valid() -> None:
    sample = sample_field_quantity_with_status(_provider(np.ones((2, 2), dtype=bool)), "ux", np.asarray([0.5, 0.5]), 0.0)

    assert sample.status == VALID_MASK_STATUS_CLEAN
    assert sample.valid is True
    assert sample.reason == "clean"
    assert sample.cell_id == 0
    assert sample.value == pytest.approx(5.0)


def test_field_sample_reports_mixed_stencil_without_zero_filling() -> None:
    valid_mask = np.asarray([[True, False], [True, True]], dtype=bool)

    sample = sample_field_quantity_with_status(_provider(valid_mask), "ux", np.asarray([0.0, 0.0]), 0.0)

    assert sample.status == VALID_MASK_STATUS_MIXED_STENCIL
    assert sample.valid is False
    assert sample.reason == "mixed_stencil"
    assert sample.value == pytest.approx(2.0)


def test_field_sample_reports_hard_invalid_as_nan() -> None:
    sample = sample_field_quantity_with_status(_provider(np.ones((2, 2), dtype=bool)), "ux", np.asarray([2.0, 2.0]), 0.0)

    assert sample.status == VALID_MASK_STATUS_HARD_INVALID
    assert sample.valid is False
    assert sample.reason == "hard_invalid"
    assert np.isnan(sample.value)
