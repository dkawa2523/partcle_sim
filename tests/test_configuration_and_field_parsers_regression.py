from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from particle_tracer_unified import _configuration_charge as configuration_charge
from particle_tracer_unified import _configuration_document as configuration_document
from particle_tracer_unified import _configuration_inputs as configuration_inputs
from particle_tracer_unified import _configuration_physics as configuration_physics
from particle_tracer_unified import configuration
from particle_tracer_unified.comsol_case._field_normalization import (
    _normalize_bundle,
)
from particle_tracer_unified.comsol_case.fields import pack_field_bundle
from particle_tracer_unified.configuration import (
    ChargeConfig,
    ConfigurationError,
    InputsConfig,
    ProviderConfig,
)


def test_configuration_facade_reexports_section_owners_directly() -> None:
    assert configuration.RunConfig is configuration_document.RunConfig
    assert configuration.TimeConfig is configuration_document.TimeConfig
    assert configuration.load_run_config is configuration_document.load_run_config
    assert configuration.ProviderConfig is configuration_inputs.ProviderConfig
    assert configuration.InputsConfig is configuration_inputs.InputsConfig
    assert configuration.PhysicsConfig is configuration_physics.PhysicsConfig
    assert configuration.ChargeConfig is configuration_charge.ChargeConfig
    assert (
        configuration.CHARGE_PARAMETER_KEYS
        is configuration_charge.CHARGE_PARAMETER_KEYS
    )


def test_provider_parsers_preserve_canonical_mapping() -> None:
    field_mapping = {
        "kind": "linear_shear",
        "parameters": {
            "shear_rate": -2.5,
            "dynamic_viscosity_Pas": 1.8e-5,
            "time_mode": "transient",
            "times": [0.0, 0.25, 1.0],
        },
    }
    geometry_mapping = {
        "kind": "box",
        "parameters": {
            "bounds": [-1.0, 1.0, -2.0, 2.0],
            "grid_shape": [11, 13],
            "boundary_part_ids": [1, 2, 3, 4],
        },
    }

    field = ProviderConfig.from_mapping(field_mapping, "inputs.field", role="field")
    geometry = ProviderConfig.from_mapping(
        geometry_mapping,
        "inputs.geometry",
        role="geometry",
    )

    assert field.to_mapping() == field_mapping
    assert geometry.to_mapping() == geometry_mapping


@pytest.mark.parametrize(
    ("mapping", "role", "message"),
    [
        (
            {
                "kind": "linear_shear",
                "parameters": {
                    "shear_rate": 1.0,
                    "dynamic_viscosity_Pas": 0.0,
                },
            },
            "field",
            "dynamic_viscosity_Pas: must be > 0",
        ),
        (
            {
                "kind": "linear_shear",
                "parameters": {
                    "shear_rate": 1.0,
                    "dynamic_viscosity_Pas": 1.0,
                    "times": [0.0, 0.0],
                },
            },
            "field",
            "times: must be strictly increasing",
        ),
        (
            {
                "kind": "box",
                "parameters": {
                    "bounds": [0.0, 1.0, 2.0],
                    "grid_shape": [3, 3],
                    "boundary_part_ids": [1, 2, 3, 4],
                },
            },
            "geometry",
            "bounds: must contain minimum/maximum pairs",
        ),
        (
            {
                "kind": "box",
                "parameters": {
                    "bounds": [0.0, 0.0, -1.0, 1.0],
                    "grid_shape": [3, 3],
                    "boundary_part_ids": [1, 2, 3, 4],
                },
            },
            "geometry",
            "bounds: each maximum must be greater than its minimum",
        ),
        (
            {"kind": "precomputed_npz"},
            "geometry",
            "path: is required for provider kind 'precomputed_npz'",
        ),
    ],
)
def test_provider_validation_order_is_stable(
    mapping: dict[str, object],
    role: str,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        ProviderConfig.from_mapping(mapping, "provider", role=role)


def test_inputs_parser_keeps_adapter_and_dimension_rules_separate() -> None:
    native = InputsConfig.from_mapping(
        {
            "particles": "particles.csv",
            "boundaries": "boundaries.csv",
            "geometry": {
                "kind": "box",
                "parameters": {
                    "bounds": [-1.0, 1.0, -1.0, 1.0],
                    "grid_shape": [5, 7],
                    "boundary_part_ids": [1, 2, 3, 4],
                },
            },
            "field": {
                "kind": "precomputed_npz",
                "path": "field.npz",
            },
        },
        adapter="native",
        spatial_dim=2,
    )
    assert native.to_mapping()["particles"] == "particles.csv"

    wrong_dimension = deepcopy(native.to_mapping())
    wrong_dimension["geometry"]["parameters"]["grid_shape"] = [5, 7, 9]
    with pytest.raises(
        ConfigurationError,
        match="grid_shape: must contain exactly 2 values",
    ):
        InputsConfig.from_mapping(
            wrong_dimension,
            adapter="native",
            spatial_dim=2,
        )

    with pytest.raises(ConfigurationError, match="declared only by the manifest"):
        InputsConfig.from_mapping(
            {
                "comsol_manifest": "case.yaml",
                "particles": "particles.csv",
            },
            adapter="comsol",
            spatial_dim=2,
        )


def test_charge_parser_preserves_oml_background_contract() -> None:
    mapping = {
        "enabled": True,
        "mode": "oml_linearized_relaxation",
        "parameters": {"background_source": "plasma_background"},
        "background": {
            "source": "saas_constant",
            "electron_density_m3": 1.0e15,
            "ion_density_m3": 1.0e15,
            "electron_temperature_eV": 2.0,
            "ion_temperature_eV": 0.03,
            "ion_mass_amu": 40.0,
            "ion_charge_number": 1.0,
        },
    }

    assert ChargeConfig.from_mapping(mapping).to_mapping() == mapping

    missing = deepcopy(mapping)
    del missing["background"]["ion_mass_amu"]
    with pytest.raises(
        ConfigurationError,
        match="saas_constant requires ion_mass_amu",
    ):
        ChargeConfig.from_mapping(missing)

    wrong_owner = deepcopy(mapping)
    wrong_owner["parameters"]["background_source"] = "field"
    with pytest.raises(
        ConfigurationError,
        match=r"is only valid when parameters.background_source is plasma_background",
    ):
        ChargeConfig.from_mapping(wrong_owner)


def test_field_bundle_exact_grid_preserves_float64_shape_and_mask() -> None:
    axes_x = np.array([0.0, 1.0], dtype=np.float64)
    axes_y = np.array([-1.0, 1.0], dtype=np.float64)
    payload = {
        "ux": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "uy": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        "valid_mask": np.array([[True, False], [True, True]]),
    }

    normalized = _normalize_bundle(payload, axes_x, axes_y)

    assert normalized["ux"].dtype == np.float64
    assert normalized["ux"].shape == (2, 2)
    assert normalized["times"].tolist() == [0.0]
    np.testing.assert_array_equal(normalized["valid_mask"], payload["valid_mask"])


def test_field_bundle_resampling_preserves_extent_and_transient_shape() -> None:
    source_x = np.array([0.0, 0.25, 2.0], dtype=np.float64)
    source_y = np.array([-1.0, -0.5, 1.0], dtype=np.float64)
    target_x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    target_y = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    steady = source_x[:, None] + source_y[None, :]
    payload = {
        "axis_0": source_x,
        "axis_1": source_y,
        "times": np.array([0.0, 1.0], dtype=np.float64),
        "ux": np.stack([steady, steady + 1.0]),
        "uy": np.stack([steady + 2.0, steady + 3.0]),
        "valid_mask": np.ones((3, 3), dtype=bool),
    }

    normalized = _normalize_bundle(payload, target_x, target_y)

    assert normalized["ux"].shape == (2, 3, 3)
    np.testing.assert_allclose(
        normalized["ux"][:, (0, -1), :][:, :, (0, -1)],
        payload["ux"][:, (0, -1), :][:, :, (0, -1)],
    )
    assert normalized["valid_mask"].dtype == np.bool_
    assert normalized["valid_mask"].shape == (3, 3)


def test_packed_field_bundle_has_one_mask_and_summary_owner(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "field.npz"
    axes = np.array([0.0, 1.0], dtype=np.float64)
    bundle_mask = np.array([[True, False], [True, True]])
    geometry_mask = np.array([[True, True], [False, True]])
    np.savez_compressed(
        source,
        axis_0=axes,
        axis_1=axes,
        ux=np.array([[1.0, 2.0], [3.0, 4.0]]),
        uy=np.array([[5.0, 6.0], [7.0, 8.0]]),
        support_phi=np.ones((2, 2), dtype=np.float64),
        valid_mask=bundle_mask,
    )

    packed = pack_field_bundle(
        source,
        destination,
        axes_x=axes,
        axes_y=axes,
        geometry_inside=geometry_mask,
        geometry_sdf=-np.ones((2, 2), dtype=np.float64),
    )

    np.testing.assert_array_equal(
        packed.particle_valid_mask,
        bundle_mask & geometry_mask,
    )
    assert packed.summary["field_valid_node_count"] == 3
    assert packed.summary["particle_release_valid_node_count"] == 2
    assert packed.summary["provider_support_expanded_node_count"] == 1
    with np.load(destination, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["valid_mask"], bundle_mask)
        assert payload["ux"][0, 1] == 0.0
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["field_valid_mask_source"] == (
        "bundle_valid_mask_and_finite_field_quantities"
    )
    assert metadata["provider_support_expanded_node_count"] == 1


def test_packed_field_bundle_rejects_nonfinite_values_claimed_valid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    axes = np.array([0.0, 1.0], dtype=np.float64)
    np.savez_compressed(
        source,
        ux=np.array([[np.nan, 1.0], [2.0, 3.0]]),
        uy=np.ones((2, 2), dtype=np.float64),
        valid_mask=np.ones((2, 2), dtype=bool),
    )

    with pytest.raises(
        ValueError,
        match="valid_mask marks non-finite field values as valid",
    ):
        pack_field_bundle(
            source,
            tmp_path / "field.npz",
            axes_x=axes,
            axes_y=axes,
            geometry_inside=np.ones((2, 2), dtype=bool),
            geometry_sdf=-np.ones((2, 2), dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"uy": np.ones((2, 2))}, "must include ux and uy"),
        (
            {"ux": np.ones((2, 2)), "uy": np.ones((2, 2)), "axis_0": [0.0, 2.0]},
            "axes are required when resampling",
        ),
        (
            {
                "ux": np.ones((2, 2)),
                "uy": np.ones((2, 2)),
                "valid_mask": np.ones((3, 2)),
            },
            "valid_mask must match geometry grid shape",
        ),
        (
            {
                "ux": np.ones((2, 2)),
                "uy": np.ones((2, 2)),
                "times": np.array([], dtype=np.float64),
            },
            "times must be a non-empty 1D array",
        ),
        (
            {
                "ux": np.ones((2, 2)),
                "uy": np.ones((2, 2)),
                "support_phi": np.ones((3, 2)),
            },
            "support_phi must match geometry grid shape",
        ),
    ],
)
def test_field_bundle_validation_messages_remain_specific(
    payload: dict[str, np.ndarray],
    message: str,
) -> None:
    axes = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        _normalize_bundle(payload, axes, axes)
