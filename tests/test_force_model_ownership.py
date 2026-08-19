from __future__ import annotations

from inspect import signature

import pytest

from particle_tracer_unified import (
    _force_model_parsing,
    _force_model_serialization,
    _force_model_types,
    _force_model_values,
    force_models,
)


def test_force_model_public_api_is_directly_owned() -> None:
    for name in (
        "ForceModel",
        "ForceModelError",
        "DragForce",
        "ElectricForce",
        "GravityForce",
        "ThermophoresisForce",
        "DielectrophoresisForce",
        "LiftForce",
        "PressureGradientForce",
        "VirtualMassForce",
    ):
        assert getattr(force_models, name) is getattr(_force_model_types, name)
    assert force_models.parse_drag_force is _force_model_values.parse_drag_force
    assert (
        force_models.parse_native_force_model
        is _force_model_parsing.parse_native_force_model
    )
    assert (
        force_models.parse_manifest_force_model
        is _force_model_parsing.parse_manifest_force_model
    )
    assert (
        force_models.force_model_to_native_mapping
        is _force_model_serialization.force_model_to_native_mapping
    )
    assert (
        force_models.force_model_to_manifest_inventory
        is _force_model_serialization.force_model_to_manifest_inventory
    )
    assert tuple(signature(force_models.parse_native_force_model).parameters) == (
        "drag",
        "forces",
        "spatial_dim",
        "path",
    )
    assert tuple(signature(force_models.parse_drag_force).parameters) == (
        "value",
        "path",
    )
    assert tuple(signature(force_models.parse_manifest_force_model).parameters) == (
        "entries",
        "spatial_dim",
        "path",
    )
    for serializer in (
        force_models.force_parameter_mapping,
        force_models.force_model_to_native_mapping,
        force_models.force_model_to_manifest_inventory,
    ):
        assert tuple(signature(serializer).parameters) in {("force",), ("model",)}


@pytest.mark.parametrize(
    ("drag", "message"),
    [
        ({}, "case.drag: missing required key 'model'"),
        ({"model": 1}, "case.drag.model: must be a string"),
        (
            {"model": " stokes"},
            "case.drag.model: must not contain leading or trailing whitespace",
        ),
    ],
)
def test_drag_error_type_message_and_priority_are_stable(
    drag: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(force_models.ForceModelError) as caught:
        force_models.parse_drag_force(drag, path="case.drag")
    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("coefficient", "message"),
    [
        (True, "must be a finite number"),
        (float("inf"), "must be a finite number"),
        (-1.0, "must be >= 0"),
    ],
)
def test_numeric_force_validation_preserves_exact_error(
    coefficient: object,
    message: str,
) -> None:
    with pytest.raises(force_models.ForceModelError) as caught:
        force_models.parse_native_force_model(
            {"model": "stokes"},
            {
                "lift": {
                    "enabled": False,
                    "parameters": {"coefficient": coefficient},
                }
            },
            spatial_dim=2,
        )
    assert str(caught.value) == (
        "physics.forces.lift.parameters.coefficient: " + message
    )


def test_gravity_shape_is_checked_before_component_values() -> None:
    with pytest.raises(force_models.ForceModelError) as caught:
        force_models.parse_native_force_model(
            {"model": "stokes"},
            {
                "gravity": {
                    "enabled": True,
                    "parameters": {"acceleration_mps2": "not-a-vector"},
                }
            },
            spatial_dim=2,
        )
    assert str(caught.value) == (
        "physics.forces.gravity.parameters.acceleration_mps2: "
        "must contain exactly 2 components"
    )


def test_ac_frequency_requires_a_strictly_positive_value_after_required_keys() -> None:
    with pytest.raises(force_models.ForceModelError) as caught:
        force_models.parse_native_force_model(
            {"model": "stokes"},
            {
                "dielectrophoresis": {
                    "enabled": True,
                    "model": "ac_clausius_mossotti",
                    "parameters": {
                        "medium_rel_permittivity": 1.2,
                        "medium_conductivity_Sm": 0.1,
                        "particle_conductivity_Sm": 0.2,
                        "frequency_Hz": 0.0,
                    },
                }
            },
            spatial_dim=2,
        )
    assert str(caught.value) == (
        "physics.forces.dielectrophoresis.parameters.frequency_Hz: must be > 0 for AC"
    )


def test_manifest_header_and_disabled_drag_serialization_are_stable() -> None:
    with pytest.raises(force_models.ForceModelError) as caught:
        force_models.parse_manifest_force_model(
            [{"solver_force": " drag", "enabled": False}],
            spatial_dim=2,
        )
    assert str(caught.value) == "forces[0].solver_force: must be an exact string"

    model = force_models.parse_manifest_force_model(
        [{"solver_force": "drag", "enabled": False}],
        spatial_dim=2,
    )
    assert model.definition("drag") is model.drag
    with pytest.raises(KeyError, match="unknown"):
        model.definition("unknown")
    assert force_models.force_model_to_manifest_inventory(model) == (
        {"solver_force": "drag", "enabled": False},
    )
