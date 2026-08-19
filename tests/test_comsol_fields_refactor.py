from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from particle_tracer_unified.comsol_case import _field_profile, _field_support
from particle_tracer_unified.comsol_case import fields as fields_facade
from particle_tracer_unified.comsol_case._field_normalization import (
    _normalize_bundle,
)
from particle_tracer_unified.comsol_case.fields import (
    build_profile_field_bundle,
    field_manifest,
    pack_field_bundle,
)
from particle_tracer_unified.comsol_case.profiles import BUILD_PROFILES, BuildProfile


def test_fields_facade_reexports_public_owners_directly() -> None:
    assert fields_facade.PackedField is _field_support.PackedField
    assert fields_facade.pack_field_bundle is _field_support.pack_field_bundle
    assert (
        fields_facade.build_profile_field_bundle
        is _field_profile.build_profile_field_bundle
    )
    assert fields_facade.field_manifest is _field_profile.field_manifest


def test_field_entrypoint_signatures_are_stable() -> None:
    assert tuple(inspect.signature(pack_field_bundle).parameters) == (
        "source",
        "destination",
        "axes_x",
        "axes_y",
        "geometry_inside",
        "geometry_sdf",
    )
    assert tuple(inspect.signature(build_profile_field_bundle).parameters) == (
        "samples_csv",
        "destination",
        "profile",
        "coordinate_scale_m_per_model_unit",
    )
    assert tuple(inspect.signature(field_manifest).parameters) == (
        "field_npz",
        "coordinate_system",
        "profile",
    )


def test_field_npz_rejects_object_arrays_without_writing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "object-array.npz"
    destination = tmp_path / "packed.npz"
    np.savez_compressed(
        source,
        ux=np.asarray([[object(), object()], [object(), object()]], dtype=object),
        uy=np.ones((2, 2), dtype=np.float64),
    )

    with pytest.raises(
        ValueError,
        match="Object arrays cannot be loaded when allow_pickle=False",
    ):
        pack_field_bundle(
            source,
            destination,
            axes_x=np.asarray([0.0, 1.0]),
            axes_y=np.asarray([0.0, 1.0]),
            geometry_inside=np.ones((2, 2), dtype=bool),
            geometry_sdf=-np.ones((2, 2), dtype=np.float64),
        )
    assert not destination.exists()


def test_packed_field_preserves_array_order_dtype_mask_and_nan_sentinel(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "packed.npz"
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    bundle_mask = np.asarray([[True, False], [True, True]])
    geometry_mask = np.asarray([[True, True], [False, True]])
    np.savez_compressed(
        source,
        uy=np.asarray([[5.0, np.nan], [7.0, 8.0]], dtype=np.float32),
        axis_1=axis,
        ux=np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
        extra=np.asarray([[9.0, np.nan], [11.0, 12.0]], dtype=np.float32),
        valid_mask=bundle_mask,
        axis_0=axis,
        times=np.asarray([0.0], dtype=np.float64),
        support_phi=np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
    )

    packed = pack_field_bundle(
        source,
        destination,
        axes_x=axis,
        axes_y=axis,
        geometry_inside=geometry_mask,
        geometry_sdf=-np.ones((2, 2), dtype=np.float64),
    )

    np.testing.assert_array_equal(
        packed.particle_valid_mask,
        bundle_mask & geometry_mask,
    )
    assert packed.particle_valid_mask.dtype == np.dtype(np.bool_)
    assert packed.summary["quantities"] == ["extra", "ux", "uy"]
    with np.load(destination, allow_pickle=False) as payload:
        assert payload.files == [
            "axis_0",
            "axis_1",
            "times",
            "valid_mask",
            "metadata_json",
            "support_phi",
            "uy",
            "ux",
            "extra",
        ]
        for name in ("axis_0", "axis_1", "times", "support_phi", "uy", "ux", "extra"):
            assert payload[name].dtype == np.dtype(np.float64)
        assert payload["valid_mask"].dtype == np.dtype(np.bool_)
        assert np.isnan(payload["support_phi"][0, 1])
        assert payload["uy"][0, 1] == 0.0
        assert payload["ux"][0, 1] == 0.0
        assert payload["extra"][0, 1] == 0.0


@given(
    source_x_mid=st.floats(min_value=0.05, max_value=0.95),
    source_y_mid=st.floats(min_value=0.05, max_value=0.95),
    target_x_mid=st.floats(min_value=0.05, max_value=0.95),
    target_y_mid=st.floats(min_value=0.05, max_value=0.95),
)
def test_linear_field_resampling_preserves_axis_time_quantity_order_and_dtype(
    source_x_mid: float,
    source_y_mid: float,
    target_x_mid: float,
    target_y_mid: float,
) -> None:
    source_x = np.asarray([0.0, source_x_mid, 1.0], dtype=np.float64)
    source_y = np.asarray([0.0, source_y_mid, 1.0], dtype=np.float64)
    target_x = np.asarray([0.0, target_x_mid, 1.0], dtype=np.float64)
    target_y = np.asarray([0.0, target_y_mid, 1.0], dtype=np.float64)
    steady = 2.0 * source_x[:, None] - 3.0 * source_y[None, :]
    payload = {
        "axis_0": source_x,
        "axis_1": source_y,
        "times": np.asarray([0.0, 2.0], dtype=np.float64),
        "valid_mask": np.ones((3, 3), dtype=bool),
        "uy": np.stack((steady - 4.0, steady - 2.0)),
        "ux": np.stack((steady, steady + 2.0)),
    }

    normalized = _normalize_bundle(payload, target_x, target_y)

    expected = 2.0 * target_x[:, None] - 3.0 * target_y[None, :]
    assert list(normalized) == [
        "axis_0",
        "axis_1",
        "times",
        "valid_mask",
        "uy",
        "ux",
    ]
    np.testing.assert_array_equal(normalized["axis_0"], target_x)
    np.testing.assert_array_equal(normalized["axis_1"], target_y)
    np.testing.assert_array_equal(normalized["times"], [0.0, 2.0])
    np.testing.assert_allclose(normalized["ux"][0], expected, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        normalized["ux"][1], expected + 2.0, rtol=0.0, atol=2e-15
    )
    assert normalized["ux"].dtype == np.dtype(np.float64)
    assert normalized["valid_mask"].dtype == np.dtype(np.bool_)


def test_readme_grid_can_be_resampled_from_one_mm_to_half_mm(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "packed.npz"
    source_x = np.linspace(0.0, 0.25, 251, dtype=np.float64)
    source_y = np.linspace(0.0, 0.22, 221, dtype=np.float64)
    target_x = np.linspace(0.0, 0.25, 501, dtype=np.float64)
    target_y = np.linspace(0.0, 0.22, 441, dtype=np.float64)
    field = 2.0 * source_x[:, None] - 3.0 * source_y[None, :]
    np.savez_compressed(
        source,
        axis_0=source_x,
        axis_1=source_y,
        ux=field,
        uy=-field,
        valid_mask=np.ones(field.shape, dtype=bool),
    )

    packed = pack_field_bundle(
        source,
        destination,
        axes_x=target_x,
        axes_y=target_y,
        geometry_inside=np.ones((target_x.size, target_y.size), dtype=bool),
        geometry_sdf=-np.ones((target_x.size, target_y.size), dtype=np.float64),
    )

    expected = 2.0 * target_x[:, None] - 3.0 * target_y[None, :]
    with np.load(destination, allow_pickle=False) as payload:
        np.testing.assert_allclose(payload["ux"], expected, rtol=0.0, atol=2e-15)
        np.testing.assert_allclose(payload["uy"], -expected, rtol=0.0, atol=2e-15)
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert packed.particle_valid_mask.shape == expected.shape
    assert metadata["axis_alignment"]["source_axes"]["axis_0"]["count"] == 251
    assert metadata["axis_alignment"]["geometry_axes"]["axis_0"]["count"] == 501
    assert metadata["axis_alignment"]["resampled_to_geometry_axes"] is True


def test_profile_bundle_preserves_column_order_and_nan_outside_support(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "samples.csv"
    destination = tmp_path / "profile.npz"
    pd.DataFrame(
        [
            {"x": 0.0, "y": 0.0, "valid_mask": 1, "ux": 1.0, "uy": 5.0, "rho": 9.0},
            {"x": 0.0, "y": 2.0, "valid_mask": 0, "ux": 2.0, "uy": 6.0, "rho": np.nan},
            {"x": 1.0, "y": 0.0, "valid_mask": 1, "ux": 3.0, "uy": 7.0, "rho": 11.0},
            {"x": 1.0, "y": 2.0, "valid_mask": 1, "ux": 4.0, "uy": 8.0, "rho": 12.0},
        ]
    ).to_csv(samples, index=False)

    result = build_profile_field_bundle(
        samples,
        destination,
        profile=BUILD_PROFILES["generic"],
        coordinate_scale_m_per_model_unit=0.5,
    )

    assert result == destination
    with np.load(destination, allow_pickle=False) as payload:
        assert payload.files == [
            "axis_0",
            "axis_1",
            "times",
            "valid_mask",
            "ux",
            "uy",
            "rho",
            "metadata_json",
        ]
        np.testing.assert_array_equal(payload["axis_0"], [0.0, 0.5])
        np.testing.assert_array_equal(payload["axis_1"], [0.0, 1.0])
        np.testing.assert_array_equal(
            payload["valid_mask"],
            [[True, False], [True, True]],
        )
        for name in ("axis_0", "axis_1", "times", "ux", "uy", "rho"):
            assert payload[name].dtype == np.dtype(np.float64)
        assert np.isnan(payload["ux"][0, 1])
        assert np.isnan(payload["uy"][0, 1])
        assert np.isnan(payload["rho"][0, 1])
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["skipped_columns"] == {}


def test_field_manifest_preserves_semantic_component_and_profile_order(
    tmp_path: Path,
) -> None:
    field_path = tmp_path / "field.npz"
    shape = (2, 2, 2)
    np.savez_compressed(
        field_path,
        times=np.asarray([0.25, 2.0], dtype=np.float64),
        ux=np.ones(shape),
        uy=np.ones(shape),
        E_x=np.ones(shape),
        E_y=np.ones(shape),
        mu=np.ones(shape),
        rho=np.ones(shape),
        T=np.ones(shape),
        pressure=np.ones(shape),
        ignored=np.ones(shape),
    )

    fields, time_support = field_manifest(
        field_path,
        coordinate_system="cartesian_xy",
        profile=BUILD_PROFILES["generic"],
    )

    assert list(fields) == [
        "velocity",
        "electric_field",
        "dynamic_viscosity",
        "density",
        "temperature",
        "pressure",
    ]
    assert fields["velocity"]["components"] == {"x": "ux", "y": "uy"}
    assert fields["electric_field"]["components"] == {"x": "E_x", "y": "E_y"}
    assert fields["dynamic_viscosity"] == {
        "artifact": "field",
        "components": {"value": "mu"},
        "unit": "Pa*s",
        "scale_to_si": 1.0,
    }
    assert time_support == (0.25, 2.0)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "axis_0": np.asarray([0.0, 2.0]),
                "axis_1": np.asarray([0.0, 1.0]),
                "ux": np.ones((2, 2)),
                "uy": np.ones((2, 2)),
            },
            "axis_0 must share geometry axis_0 extent",
        ),
        (
            {
                "axis_0": np.asarray([0.0, 1.0]),
                "axis_1": np.asarray([0.0, 2.0]),
                "ux": np.ones((2, 2)),
                "uy": np.ones((2, 2)),
            },
            "axis_1 must share geometry axis_1 extent",
        ),
        (
            {"ux": np.ones((3, 2)), "uy": np.ones((2, 2))},
            "quantity ux must match geometry grid shape",
        ),
        (
            {
                "times": np.asarray([0.0, 1.0]),
                "ux": np.ones((3, 2, 2)),
                "uy": np.ones((2, 2, 2)),
            },
            "quantity ux must match shape",
        ),
        (
            {"ux": np.ones(2), "uy": np.ones((2, 2))},
            "quantity ux must be 2D or 3D",
        ),
        (
            {
                "axis_0": np.asarray([0.0, 0.5, 1.0]),
                "axis_1": np.asarray([0.0, 0.5, 1.0]),
                "ux": np.ones(3),
                "uy": np.ones((3, 3)),
            },
            "quantity must be 2D or 3D",
        ),
    ],
)
def test_field_normalization_error_order_and_messages_are_stable(
    payload: dict[str, np.ndarray],
    message: str,
) -> None:
    target = np.asarray([0.0, 1.0])
    if payload.get("axis_0", target).size == 3:
        target = np.asarray([0.0, 0.25, 1.0])
    with pytest.raises(ValueError, match=message):
        _normalize_bundle(payload, target, target)


def _profile_rows(*, include_valid_mask: bool = True) -> list[dict[str, float]]:
    rows = [
        {"x": 0.0, "y": 0.0, "ux": 1.0, "uy": 5.0, "pressure": 9.0},
        {"x": 0.0, "y": 1.0, "ux": 2.0, "uy": 6.0, "pressure": 10.0},
        {"x": 1.0, "y": 0.0, "ux": 3.0, "uy": 7.0, "pressure": 11.0},
        {"x": 1.0, "y": 1.0, "ux": 4.0, "uy": 8.0, "pressure": 12.0},
    ]
    if include_valid_mask:
        for row in rows:
            row["valid_mask"] = 1.0
    return rows


def test_profile_without_explicit_mask_uses_required_fields_and_skips_bad_optional(
    tmp_path: Path,
) -> None:
    rows = _profile_rows(include_valid_mask=False)
    rows[-1]["pressure"] = np.nan
    source = tmp_path / "profile.csv"
    destination = tmp_path / "field.npz"
    pd.DataFrame(rows).to_csv(source, index=False)

    build_profile_field_bundle(
        source,
        destination,
        profile=BUILD_PROFILES["generic"],
        coordinate_scale_m_per_model_unit=1.0,
    )

    with np.load(destination, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["valid_mask"], np.ones((2, 2), bool))
        assert "pressure" not in payload.files
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["skipped_columns"] == {"pressure": "nonfinite_on_valid_support"}


@pytest.mark.parametrize(
    ("rows", "scale", "message"),
    [
        (
            [*_profile_rows()[:3], _profile_rows()[0]],
            1.0,
            "duplicate coordinate pairs",
        ),
        (_profile_rows(), 0.0, "coordinate_scale_m_per_model_unit must be positive"),
        (
            [row for row in _profile_rows() if row["x"] == 0.0],
            1.0,
            "axes must each contain at least two points",
        ),
        (_profile_rows()[:3], 1.0, "must form a complete tensor grid"),
    ],
)
def test_profile_grid_validation_order_and_messages_are_stable(
    tmp_path: Path,
    rows: list[dict[str, float]],
    scale: float,
    message: str,
) -> None:
    source = tmp_path / "profile.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    with pytest.raises(ValueError, match=message):
        build_profile_field_bundle(
            source,
            tmp_path / "field.npz",
            profile=BUILD_PROFILES["generic"],
            coordinate_scale_m_per_model_unit=scale,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("empty_support", "contain no valid support"),
        ("required_nonfinite", "required COMSOL field 'ux' is non-finite"),
    ],
)
def test_profile_support_errors_are_stable(
    tmp_path: Path,
    mutate: str,
    message: str,
) -> None:
    rows = _profile_rows()
    if mutate == "empty_support":
        for row in rows:
            row["valid_mask"] = 0.0
    else:
        rows[0]["ux"] = np.nan
    source = tmp_path / "profile.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    with pytest.raises(ValueError, match=message):
        build_profile_field_bundle(
            source,
            tmp_path / "field.npz",
            profile=BUILD_PROFILES["generic"],
            coordinate_scale_m_per_model_unit=1.0,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "times": np.asarray([0.0, np.nan]),
                "ux": np.ones((2, 2, 2)),
                "uy": np.ones((2, 2, 2)),
            },
            "times must be finite and non-decreasing",
        ),
        (
            {"times": np.asarray([0.0]), "ux": np.ones((2, 2))},
            "must contain ux and uy",
        ),
    ],
)
def test_manifest_inventory_errors_are_stable(
    tmp_path: Path,
    payload: dict[str, Any],
    message: str,
) -> None:
    source = tmp_path / "field.npz"
    np.savez_compressed(source, **payload)
    with pytest.raises(ValueError, match=message):
        field_manifest(
            source,
            coordinate_system="cartesian_xy",
            profile=BUILD_PROFILES["generic"],
        )


@pytest.mark.parametrize(
    ("source_name", "semantic", "message"),
    [
        ("velocity", "scalar", "must be a non-reserved semantic identifier"),
        ("custom", "velocity", "invalid built-in semantic 'velocity'"),
    ],
)
def test_manifest_rejects_invalid_scalar_profile_semantics(
    tmp_path: Path,
    source_name: str,
    semantic: str,
    message: str,
) -> None:
    source = tmp_path / "field.npz"
    payload: dict[str, Any] = {
        "ux": np.ones((2, 2)),
        "uy": np.ones((2, 2)),
        source_name: np.ones((2, 2)),
    }
    np.savez_compressed(source, **payload)
    profile = BuildProfile(
        name="invalid",
        coordinate_system="cartesian_xy",
        sample_axis_columns=("x", "y"),
        required_sample_columns=("ux", "uy"),
        scalar_fields={source_name: (semantic, "1")},
    )

    with pytest.raises(ValueError, match=message):
        field_manifest(
            source,
            coordinate_system="cartesian_xy",
            profile=profile,
        )


def test_packed_field_validates_summary_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "field.npz"
    axis = np.asarray([0.0, 1.0])
    np.savez_compressed(
        source,
        ux=np.ones((2, 2)),
        uy=np.ones((2, 2)),
    )

    with pytest.raises(ValueError, match="support_phi shape mismatch"):
        pack_field_bundle(
            source,
            destination,
            axes_x=axis,
            axes_y=axis,
            geometry_inside=np.ones((2, 2), dtype=bool),
            geometry_sdf=np.ones((1, 1), dtype=np.float64),
        )
    assert not destination.exists()
