from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from particle_tracer_unified.force_models import (
    DEP_MEDIUM_REL_PERMITTIVITY_DEFAULT,
    LIFT_COEFFICIENT_DEFAULT,
    THERMOPHORESIS_CM_DEFAULT,
    THERMOPHORESIS_CS_DEFAULT,
    THERMOPHORESIS_CT_DEFAULT,
    THERMOPHORESIS_GAS_CONDUCTIVITY_DEFAULT,
    THERMOPHORESIS_PARTICLE_CONDUCTIVITY_DEFAULT,
    VIRTUAL_MASS_COEFFICIENT_DEFAULT,
    ForceModelError,
    force_model_to_manifest_inventory,
    parse_manifest_force_model,
    parse_native_force_model,
)
from particle_tracer_unified.solvers.forces import compile_force_runtime_parameters


def _native_all_forces():
    return parse_native_force_model(
        {"model": "stokes_cunningham"},
        {
            "electric": {"enabled": True},
            "gravity": {
                "enabled": True,
                "parameters": {
                    "acceleration_mps2": [0.0, -9.81],
                    "buoyancy": True,
                },
            },
            "thermophoresis": {
                "enabled": True,
                "model": "continuum",
                "parameters": {
                    "gas_thermal_conductivity_W_mK": 0.031,
                    "particle_thermal_conductivity_W_mK": 2.4,
                    "Cs": 1.21,
                },
            },
            "dielectrophoresis": {
                "enabled": True,
                "model": "ac_clausius_mossotti",
                "parameters": {
                    "medium_rel_permittivity": 1.2,
                    "particle_rel_permittivity": 4.0,
                    "medium_conductivity_Sm": 0.1,
                    "particle_conductivity_Sm": 0.2,
                    "frequency_Hz": 13.56e6,
                },
            },
            "lift": {"enabled": True, "parameters": {"coefficient": 7.0}},
            "pressure_gradient": {"enabled": True},
            "virtual_mass": {"enabled": True, "parameters": {"coefficient": 0.73}},
        },
        spatial_dim=2,
    )


def test_native_and_manifest_resolve_to_the_same_immutable_force_model() -> None:
    native = _native_all_forces()
    manifest = parse_manifest_force_model(
        list(force_model_to_manifest_inventory(native)),
        spatial_dim=2,
    )
    assert manifest == native
    with pytest.raises(FrozenInstanceError):
        native.gravity.buoyancy = False


def _test_absent_optional_permittivity_uses_typed_none() -> None:
    native = parse_native_force_model({"model": "stokes"}, {}, spatial_dim=2)
    manifest = parse_manifest_force_model(
        list(force_model_to_manifest_inventory(native)),
        spatial_dim=2,
    )
    assert native.dielectrophoresis.particle_rel_permittivity is None
    assert manifest == native
    assert compile_force_runtime_parameters(native).dep_particle_rel_permittivity != (
        compile_force_runtime_parameters(native).dep_particle_rel_permittivity
    )


test_absent_optional_particle_permittivity_has_a_typed_none_not_a_nan_semantic = (
    _test_absent_optional_permittivity_uses_typed_none
)


def test_runtime_projection_is_a_lossless_scalar_compile_of_semantic_values() -> None:
    model = _native_all_forces()
    runtime = compile_force_runtime_parameters(model)
    assert runtime.thermophoresis_model == model.thermophoresis.model
    assert runtime.thermophoresis_Cs == model.thermophoresis.Cs
    assert runtime.dep_frequency_Hz == model.dielectrophoresis.frequency_Hz
    assert runtime.lift_coefficient == model.lift.coefficient
    assert runtime.virtual_mass_coefficient == model.virtual_mass.coefficient
    assert runtime.gravity_buoyancy_enabled == model.gravity.buoyancy


@pytest.mark.parametrize(
    ("forces", "message"),
    [
        ({"unknown": {"enabled": True}}, "unknown force"),
        ({"gravity": {"enabled": True}}, "acceleration_mps2"),
        ({"thermophoresis": {"enabled": True}}, "thermal_conductivity"),
        ({"dielectrophoresis": {"enabled": True}}, "medium_rel_permittivity"),
    ],
)
def test_invalid_force_contracts_fail_at_the_single_parser_boundary(
    forces: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ForceModelError, match=message):
        parse_native_force_model({"model": "stokes"}, forces, spatial_dim=2)


def test_optional_force_defaults_and_status_are_stable() -> None:
    model = parse_native_force_model(
        {"model": "stokes"},
        {
            "electric": {"enabled": False},
            "gravity": {"enabled": False},
            "thermophoresis": {"enabled": False},
            "dielectrophoresis": {"enabled": False},
            "lift": {"enabled": False},
            "pressure_gradient": {"enabled": False},
            "virtual_mass": {"enabled": False},
        },
        spatial_dim=3,
    )

    assert tuple(force.name for force in model.definitions()) == (
        "drag",
        "electric",
        "gravity",
        "thermophoresis",
        "dielectrophoresis",
        "lift",
        "pressure_gradient",
        "virtual_mass",
    )
    assert tuple(force.status for force in model.definitions()) == (
        "implemented",
        "implemented",
        "implemented",
        "experimental",
        "experimental",
        "experimental",
        "experimental",
        "experimental",
    )
    assert model.gravity.acceleration_mps2 == ()
    assert model.gravity.buoyancy is False
    assert model.thermophoresis.gas_thermal_conductivity_W_mK == (
        THERMOPHORESIS_GAS_CONDUCTIVITY_DEFAULT
    )
    assert model.thermophoresis.particle_thermal_conductivity_W_mK == (
        THERMOPHORESIS_PARTICLE_CONDUCTIVITY_DEFAULT
    )
    assert model.thermophoresis.Cs == THERMOPHORESIS_CS_DEFAULT
    assert model.thermophoresis.Cm == THERMOPHORESIS_CM_DEFAULT
    assert model.thermophoresis.Ct == THERMOPHORESIS_CT_DEFAULT
    assert (
        model.dielectrophoresis.medium_rel_permittivity
        == DEP_MEDIUM_REL_PERMITTIVITY_DEFAULT
    )
    assert model.dielectrophoresis.particle_rel_permittivity is None
    assert model.lift.coefficient == LIFT_COEFFICIENT_DEFAULT
    assert model.virtual_mass.coefficient == VIRTUAL_MASS_COEFFICIENT_DEFAULT


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (
            {"solver_force": "gravity", "unexpected": True},
            "forces[0].enabled: is required",
        ),
        (
            {"solver_force": "gravity", "enabled": 1, "unexpected": True},
            "forces[0].enabled: must be a YAML boolean (true or false)",
        ),
        (
            {
                "solver_force": "gravity",
                "enabled": False,
                "law": "constant_acceleration",
                "unexpected": True,
            },
            "forces[0].law: is valid only for drag",
        ),
        (
            {
                "solver_force": "drag",
                "enabled": False,
                "law": "none",
                "unexpected": True,
            },
            "forces[0]: unknown key(s): unexpected",
        ),
    ],
)
def test_manifest_entry_validation_order_is_stable(
    entry: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ForceModelError) as caught:
        parse_manifest_force_model([entry], spatial_dim=2)
    assert str(caught.value) == message


def test_manifest_duplicate_is_reported_after_entry_validation() -> None:
    entries = [
        {"solver_force": "electric", "enabled": True},
        {"solver_force": "electric", "enabled": True, "model": "particle_charge"},
    ]

    with pytest.raises(ForceModelError) as caught:
        parse_manifest_force_model(entries, spatial_dim=2)

    assert str(caught.value) == "forces[1]: unknown key(s): model"


def test_gravity_dimension_error_preserves_parameter_path() -> None:
    with pytest.raises(ForceModelError) as caught:
        parse_native_force_model(
            {"model": "stokes"},
            {
                "gravity": {
                    "enabled": True,
                    "parameters": {"acceleration_mps2": [0.0, -9.81]},
                }
            },
            spatial_dim=3,
        )

    assert str(caught.value) == (
        "physics.forces.gravity.parameters.acceleration_mps2: "
        "must contain exactly 3 components"
    )


def test_manifest_without_drag_returns_complete_but_undeclared_drag() -> None:
    model = parse_manifest_force_model(
        [{"solver_force": "electric", "enabled": True}],
        spatial_dim=2,
    )

    assert model.drag.enabled is False
    assert model.drag.model == "none"
    assert model.declared == frozenset({"electric"})
    assert force_model_to_manifest_inventory(model) == (
        {"solver_force": "electric", "enabled": True},
    )
