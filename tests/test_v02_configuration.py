from __future__ import annotations

from copy import deepcopy

import pytest

from particle_tracer_unified.configuration import ConfigurationError, RunConfig


def _native_config() -> dict:
    return {
        "schema_version": 2,
        "case": {
            "spatial_dim": 2,
            "coordinate_system": "cartesian_xy",
            "adapter": "native",
        },
        "inputs": {
            "particles": "particles.csv",
            "boundaries": "boundaries.csv",
            "geometry": {
                "kind": "box",
                "parameters": {
                    "bounds": [-1, 1, -1, 1],
                    "grid_shape": [11, 11],
                    "boundary_part_ids": [1, 1, 1, 1],
                },
            },
            "field": {
                "kind": "linear_shear",
                "parameters": {"shear_rate": 0.0, "dynamic_viscosity_Pas": 1.8e-5},
            },
        },
        "physics": {
            "drag": {"model": "stokes"},
            "gas": {"dynamic_viscosity_Pas": 1.8e-5},
            "forces": {},
            "seed": 7,
        },
        "time": {"dt": 0.01, "t_end": 1.0},
        "output": {"mode": "standard"},
    }


def test_strict_config_has_one_typed_canonical_representation() -> None:
    config = RunConfig.from_mapping(_native_config())

    assert config.physics.force_model is not None
    assert config.physics.force_model.drag.model == "stokes"
    assert config.inputs.boundaries == "boundaries.csv"
    assert config.time.dt == 0.01
    assert config.output.mode == "standard"
    assert config.to_mapping() == _native_config()


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda value: value.update({"legacy": {}}), "config: unknown key"),
        (
            lambda value: value["time"].update({"integrator": "etd"}),
            "time: unknown key",
        ),
        (
            lambda value: value["physics"].update({"min_tau_p_s": 1e-6}),
            "physics: unknown key",
        ),
        (
            lambda value: value["physics"].update({"drag": "stokes"}),
            "physics.drag: must be a mapping",
        ),
        (lambda value: value["case"].update({"adapter": "Native"}), "must be one of"),
        (
            lambda value: value["output"].pop("mode"),
            "output: missing required key 'mode'",
        ),
        (
            lambda value: value.update({"output": {"mode": "debug"}}),
            "output.trajectory_interval_steps: is required",
        ),
        (
            lambda value: value["physics"].update(
                {"forces": {"gravity": {"enabled": "true"}}}
            ),
            "must be a YAML boolean",
        ),
    ],
)
def test_unknown_and_ambiguous_legacy_values_are_rejected(
    mutate, expected: str
) -> None:
    value = _native_config()
    mutate(value)
    with pytest.raises(ConfigurationError, match=expected):
        RunConfig.from_mapping(value)


def test_schema_version_is_required() -> None:
    value = _native_config()
    value.pop("schema_version")

    with pytest.raises(
        ConfigurationError, match="missing required key 'schema_version'"
    ):
        RunConfig.from_mapping(value)


def test_explicit_none_is_ballistic_and_requires_no_gas() -> None:
    value = _native_config()
    value["physics"]["drag"] = {"model": "none"}
    value["physics"]["gas"] = {}

    config = RunConfig.from_mapping(value)

    assert config.physics.force_model is not None
    assert config.physics.force_model.drag.model == "none"


def test_native_drag_requires_only_its_physical_gas_inputs() -> None:
    value = _native_config()
    value["physics"]["drag"] = {"model": "schiller_naumann"}
    with pytest.raises(ConfigurationError, match="requires density_kgm3"):
        RunConfig.from_mapping(value)


def test_comsol_manifest_is_the_only_input_and_force_inventory() -> None:
    value = _native_config()
    value["case"]["adapter"] = "comsol"
    value["inputs"] = {"comsol_manifest": "case_manifest.yaml"}
    value["physics"].pop("drag")
    config = RunConfig.from_mapping(value)
    assert config.inputs.comsol_manifest == "case_manifest.yaml"
    assert config.physics.force_model is None

    with_provider = deepcopy(value)
    with_provider["inputs"]["field"] = {"kind": "precomputed_npz", "path": "field.npz"}
    with pytest.raises(ConfigurationError, match="declared only by the manifest"):
        RunConfig.from_mapping(with_provider)

    with_force = deepcopy(value)
    with_force["physics"]["forces"] = {"electric": {"enabled": True}}
    with pytest.raises(ConfigurationError, match="declared only by the manifest"):
        RunConfig.from_mapping(with_force)


def test_axisymmetric_rz_rejects_stochastic_motion() -> None:
    value = _native_config()
    value["case"]["coordinate_system"] = "axisymmetric_rz"
    value["physics"]["stochastic"] = {"enabled": True}
    with pytest.raises(ConfigurationError, match="not supported for axisymmetric_rz"):
        RunConfig.from_mapping(value)


def test_brownian_requires_dissipative_drag() -> None:
    value = _native_config()
    value["physics"]["drag"] = {"model": "none"}
    value["physics"]["gas"] = {}
    value["physics"]["stochastic"] = {"enabled": True}

    with pytest.raises(ConfigurationError, match="requires a dissipative drag model"):
        RunConfig.from_mapping(value)


def test_force_parameters_and_models_are_strictly_typed() -> None:
    value = _native_config()
    value["physics"]["forces"] = {
        "gravity": {
            "enabled": True,
            "model": "constant_acceleration",
            "parameters": {"acceleration_mps2": [0.0, -9.81], "buoyancy": False},
        }
    }
    parsed = RunConfig.from_mapping(value)
    assert parsed.physics.force_model is not None
    assert parsed.physics.force_model.gravity.acceleration_mps2 == (0.0, -9.81)

    unknown = deepcopy(value)
    unknown["physics"]["forces"]["gravity"]["parameters"]["body_acceleration"] = [
        0.0,
        -9.81,
    ]
    with pytest.raises(ConfigurationError, match="unknown key"):
        RunConfig.from_mapping(unknown)

    wrong_dimension = deepcopy(value)
    wrong_dimension["physics"]["forces"]["gravity"]["parameters"][
        "acceleration_mps2"
    ] = [9.81]
    with pytest.raises(ConfigurationError, match="exactly 2 components"):
        RunConfig.from_mapping(wrong_dimension)

    missing_thermal_properties = _native_config()
    missing_thermal_properties["physics"]["forces"] = {
        "thermophoresis": {"enabled": True, "model": "talbot"}
    }
    with pytest.raises(
        ConfigurationError, match="requires explicit gas_thermal_conductivity"
    ):
        RunConfig.from_mapping(missing_thermal_properties)

    missing_medium_permittivity = _native_config()
    missing_medium_permittivity["physics"]["forces"] = {
        "dielectrophoresis": {"enabled": True, "model": "dc"}
    }
    with pytest.raises(
        ConfigurationError, match="requires explicit medium_rel_permittivity"
    ):
        RunConfig.from_mapping(missing_medium_permittivity)


def test_charge_and_plasma_parameter_blocks_reject_unknown_aliases() -> None:
    missing_mode = _native_config()
    missing_mode["physics"]["charge"] = {"enabled": False}
    with pytest.raises(
        ConfigurationError, match=r"physics\.charge: missing required key 'mode'"
    ):
        RunConfig.from_mapping(missing_mode)

    value = _native_config()
    value["physics"]["charge"] = {
        "enabled": True,
        "mode": "te_relaxation",
        "parameters": {
            "relaxation_time_s": 1.0e-6,
            "te_relaxation_alpha": 2.5,
            "background_source": "plasma_background",
        },
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
    RunConfig.from_mapping(value)

    legacy = deepcopy(value)
    legacy["physics"]["charge"]["background"]["ne_m3"] = 1.0e15
    with pytest.raises(ConfigurationError, match="unknown key"):
        RunConfig.from_mapping(legacy)

    hidden_relaxation_default = _native_config()
    hidden_relaxation_default["physics"]["charge"] = {
        "enabled": True,
        "mode": "te_relaxation",
    }
    with pytest.raises(
        ConfigurationError, match="enabled te_relaxation requires explicit"
    ):
        RunConfig.from_mapping(hidden_relaxation_default)

    disabled_te = _native_config()
    disabled_te["physics"]["charge"] = {
        "enabled": False,
        "mode": "te_relaxation",
    }
    RunConfig.from_mapping(disabled_te)

    oml = deepcopy(value)
    oml["physics"]["charge"]["mode"] = "oml_linearized_relaxation"
    oml["physics"]["charge"]["parameters"].pop("te_relaxation_alpha")
    oml["physics"]["charge"]["parameters"].pop("relaxation_time_s")
    RunConfig.from_mapping(oml)

    for enabled in (False, True):
        for forbidden_name in ("te_relaxation_alpha", "relaxation_time_s"):
            invalid_oml = deepcopy(oml)
            invalid_oml["physics"]["charge"]["enabled"] = enabled
            invalid_oml["physics"]["charge"]["parameters"][forbidden_name] = 1.0
            with pytest.raises(ConfigurationError, match="OML mode does not accept"):
                RunConfig.from_mapping(invalid_oml)

    incomplete_background = deepcopy(value)
    incomplete_background["physics"]["charge"]["background"].pop("ion_mass_amu")
    with pytest.raises(ConfigurationError, match="saas_constant requires ion_mass_amu"):
        RunConfig.from_mapping(incomplete_background)
