from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from chamber_particles.physics.catalog import (
    SUPPORTED_PHYSICS_CATEGORIES,
    ModelDescriptor,
    PhysicsConfigurationError,
    compatible_integrators,
    describe_models,
    resolve_physics_plan,
)


def _input(descriptor: ModelDescriptor) -> dict[str, object]:
    values: dict[str, object] = {
        "model": descriptor.model,
        "revision": descriptor.revision,
        "applicability": "error",
        "gravity_m_s2": [0.0, -9.80665],
        "delta": 1.2,
        "maximum_speed_ratio": 0.05,
        "diffuse_reflection_fraction": 0.5,
        "real_clausius_mossotti_factor": 0.5,
        "interval_tree_depth": 1,
        "particle_properties": {
            "mass_kg": 2.0,
            "drag_diameter_m": 1.0,
            "electrostatic_radius_m": 0.5,
            "displaced_volume_m3": 0.0,
        },
    }
    return {
        key: key.removesuffix("_field")
        if key in descriptor.field_parameters
        else values.get(key, 1.0)
        for key in descriptor.required_parameters
    }


def _models(descriptor: ModelDescriptor) -> dict[str, dict[str, object]]:
    models: dict[str, dict[str, object]] = {
        "charge": {"model": "fixed"},
        descriptor.category: _input(descriptor),
    }
    if descriptor.category == "noise":
        drag_model = (
            "epstein_linear"
            if descriptor.model == "inertial_langevin_fdt"
            else "prescribed_linear_friction"
        )
        drag = next(
            item
            for item in describe_models()
            if item.category == "drag" and item.model == drag_model
        )
        models["drag"] = _input(drag)
    return models


@pytest.mark.parametrize(
    "descriptor",
    describe_models(),
    ids=lambda item: f"{item.category}-{item.revision}",
)
def test_advertised_selector_keys_resolve_and_describe_the_selected_model(
    descriptor: ModelDescriptor,
) -> None:
    for coordinate in descriptor.coordinate_systems:
        plan = resolve_physics_plan(_models(descriptor), coordinate)
        selected = plan.resolved_models()[descriptor.category]
        assert selected["model"] == descriptor.model
        assert selected["revision"] == descriptor.revision
        assert set(compatible_integrators(plan)) <= set(descriptor.integrators)
        assert compatible_integrators(plan)
        roles = tuple(field for field in descriptor.fields if field.coordinate_system == coordinate)
        assert {field.parameter for field in roles} == set(descriptor.field_parameters)
        actual_fields = {field.name: field for field in plan.required_fields}
        selector = _input(descriptor)
        for role in roles:
            field_name = selector[role.parameter]
            assert isinstance(field_name, str)
            expected: dict[str, object] = dict(asdict(role))
            expected.pop("coordinate_system")
            expected.pop("parameter")
            expected["name"] = field_name
            assert asdict(actual_fields[field_name]) == expected
        certificate_roles = descriptor.certificates
        assert set(certificate_roles.local_applicability_fields) <= set(descriptor.field_parameters)
        assert set(certificate_roles.local_additive_fields) <= set(descriptor.field_parameters)


def test_model_descriptions_are_immutable_serializable_and_include_pair_constraints() -> None:
    descriptors = describe_models()
    json.dumps([asdict(item) for item in descriptors], allow_nan=False)
    assert {item.category for item in descriptors} == SUPPORTED_PHYSICS_CATEGORIES
    noise = next(item for item in descriptors if item.model == "prescribed_force_diffusion")
    assert noise.integrators == ("ou_langevin",)
    assert "adaptive_max_depth" in noise.optional_parameters
    assert any("prescribed_linear_friction" in note for note in noise.notes)
    assign_attribute = setattr
    with pytest.raises(FrozenInstanceError):
        assign_attribute(noise, "model", "different_model")


def test_descriptor_optional_tree_depth_retains_existing_default_and_rejects_extra_keys() -> None:
    descriptor = next(item for item in describe_models() if item.model == "inertial_langevin_fdt")
    models = _models(descriptor)
    plan = resolve_physics_plan(models, "cartesian_xy")
    assert plan.noise is not None
    assert plan.noise.adaptive_max_depth == plan.noise.interval_tree_depth
    models["noise"]["adaptive_max_depth"] = 3
    updated = resolve_physics_plan(models, "cartesian_xy")
    assert updated.noise is not None
    assert updated.noise.adaptive_max_depth == 3
    models["noise"]["unknown_parameter"] = 1
    with pytest.raises(PhysicsConfigurationError, match="keys do not match"):
        resolve_physics_plan(models, "cartesian_xy")


def test_coordinate_and_integrator_capabilities_preserve_actual_restrictions() -> None:
    fixed = resolve_physics_plan({"charge": {"model": "fixed"}}, "cartesian_xy")
    assert compatible_integrators(fixed) == ("rk4_fixed", "exponential_midpoint")
    descriptor = next(
        item for item in describe_models() if item.model == "rarefied_vorticity_sensitivity"
    )
    assert descriptor.coordinate_systems == ("axisymmetric_rz",)
    with pytest.raises(PhysicsConfigurationError, match="axisymmetric_rz motion only"):
        resolve_physics_plan(_models(descriptor), "cartesian_xy")


def test_model_field_roles_explain_si_temperature_diffusion_and_axis_conventions() -> None:
    descriptors = describe_models()
    charge = next(
        item for item in descriptors if item.revision == "oml_stationary_maxwellian_debye_huckel_v1"
    )
    electron_temperature = next(
        field for field in charge.fields if field.parameter == "electron_temperature_field"
    )
    assert electron_temperature.unit == "K"
    aggregate = next(
        item
        for item in descriptors
        if item.revision == "aggregate_relative_drift_regularized_two_current_v1"
    )
    electron_voltage = next(
        field for field in aggregate.fields if field.parameter == "electron_thermal_voltage_field"
    )
    assert electron_voltage.unit == "V"
    diffusion = next(item for item in descriptors if item.model == "prescribed_force_diffusion")
    assert all(
        field.unit == "N^2*s" and field.value_constraint == "positive" for field in diffusion.fields
    )
    lift = next(item for item in descriptors if item.model == "prescribed_rotational_lift")
    for coordinate in lift.coordinate_systems:
        rotation = next(
            field
            for field in lift.fields
            if field.parameter == "rotation_coefficient_field"
            and field.coordinate_system == coordinate
        )
        assert rotation.unit == "kg/s"
        assert rotation.components == ("value",)
        assert rotation.stored_basis == "scalar"
        assert rotation.value_constraint == "finite"
        assert rotation.zero_on_rz_axis == (coordinate == "axisymmetric_rz")


def test_model_fields_distinguish_zero_current_from_positive_density_and_signed_offset() -> None:
    descriptors = describe_models()
    three_current = next(
        model for model in descriptors if model.revision.endswith("three_current_v1")
    )
    constraints = {field.parameter: field.value_constraint for field in three_current.fields}
    assert constraints["negative_ion_number_density_field"] == "nonnegative"
    assert constraints["positive_ion_number_density_field"] == "positive"
    prescribed = next(
        model for model in descriptors if model.model == "prescribed_exponential_linear"
    )
    constraints = {field.parameter: field.value_constraint for field in prescribed.fields}
    assert constraints == {
        "electron_exponent_offset_field": "finite",
        "electron_charge_slope_field": "nonnegative",
        "ion_rate_field": "nonnegative",
        "ion_charge_slope_field": "nonnegative",
    }
