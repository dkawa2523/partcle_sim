from __future__ import annotations

import math

import numpy as np
import pytest

from chamber_particles.physics.catalog import (
    BarnesCollisionlessIonDragPlan,
    ElectricFieldDirectedImageIonDragPlan,
    PhysicsConfigurationError,
    RelativeFlowScreenedIonDragPlan,
    resolve_physics_plan,
)
from chamber_particles.physics.forces import (
    BarnesIonDragEvaluation,
    barnes_collisionless_continuous_applicability_batch,
    barnes_collisionless_global_bounds,
    barnes_collisionless_ion_drag,
    electric_field_directed_image_orbital_ion_drag,
    relative_flow_screened_collection_orbital_ion_drag,
)
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime

_ION_MASS_KG = 6.6335209e-26
_REVISION = "barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1"


def test_barnes_orbital_term_matches_independent_impact_parameter_quadrature() -> None:
    evaluation = _evaluate(
        charge_number=np.asarray([-10.0]),
        velocity_m_s=np.asarray([[0.0, 0.0]]),
        ion_velocity_m_s=np.asarray([[100.0, 20.0]]),
    )
    lower = float(evaluation.collection_impact_parameter_m[0])
    upper = float(evaluation.debye_length_m[0])
    orbital = float(evaluation.orbital_impact_parameter_m[0])
    nodes, weights = np.polynomial.legendre.leggauss(256)
    impact = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    scattering_angle = 2.0 * np.arctan2(orbital, impact)
    integrand = 2.0 * math.pi * impact * (1.0 - np.cos(scattering_angle))
    quadrature = 0.5 * (upper - lower) * float(np.dot(weights, integrand))

    assert evaluation.applicable.tolist() == [True]
    assert float(evaluation.collection_cross_section_m2[0]) == pytest.approx(
        math.pi * lower * lower,
        rel=3.0e-15,
    )
    assert float(evaluation.orbital_cross_section_m2[0]) == pytest.approx(
        quadrature,
        rel=2.0e-13,
    )
    relative = np.asarray([100.0, 20.0])
    assert float(np.dot(evaluation.acceleration_m_s2[0], relative)) > 0.0
    cross = evaluation.acceleration_m_s2[0, 0] * relative[1]
    cross -= evaluation.acceleration_m_s2[0, 1] * relative[0]
    assert cross == pytest.approx(0.0, abs=2.0e-18)


def test_barnes_zero_limits_and_positive_charge_fail_closed_without_a_floor() -> None:
    evaluation = _evaluate(
        charge_number=np.asarray([0.0, -10.0, 1.0]),
        velocity_m_s=np.asarray([[0.0, 0.0], [100.0, 20.0], [0.0, 0.0]]),
        ion_velocity_m_s=np.asarray([[100.0, 20.0], [100.0, 20.0], [100.0, 20.0]]),
    )

    assert evaluation.orbital_impact_parameter_m[0] == 0.0
    assert evaluation.orbital_cross_section_m2[0] == 0.0
    assert evaluation.collection_cross_section_m2[0] == pytest.approx(
        math.pi * 1.0e-14,
        rel=3.0e-15,
    )
    np.testing.assert_array_equal(evaluation.acceleration_m_s2[1], np.zeros(2))
    assert evaluation.applicable.tolist() == [True, True, False]
    np.testing.assert_array_equal(evaluation.acceleration_m_s2[2], np.zeros(2))


def test_barnes_global_bound_and_continuous_drift_gate_cover_sampled_domain() -> None:
    masses = np.asarray([8.0e-16, 1.2e-15])
    radii = np.asarray([6.0e-8, 1.0e-7])
    charge_lower = np.asarray([-20.0, -12.0])
    charge_upper = np.asarray([-4.0, 0.0])
    bounds = barnes_collisionless_global_bounds(
        mass_kg=masses,
        electrostatic_radius_m=radii,
        charge_number_lower=charge_lower,
        charge_number_upper=charge_upper,
        electron_number_density_lower_m3=8.0e13,
        electron_number_density_upper_m3=1.2e14,
        positive_ion_number_density_lower_m3=7.0e13,
        positive_ion_number_density_upper_m3=1.3e14,
        electron_temperature_lower_K=2.5e4,
        electron_temperature_upper_K=3.5e4,
        positive_ion_temperature_lower_K=250.0,
        positive_ion_temperature_upper_K=400.0,
        ion_neutral_mean_free_path_lower_m=8.0e-3,
        positive_ion_mass_kg=_ION_MASS_KG,
        maximum_ion_drift_ratio=1.5,
    )
    assert bounds.static_applicable.tolist() == [True, True]

    rng = np.random.default_rng(20260930)
    for _ in range(64):
        electron_density = rng.uniform(8.0e13, 1.2e14, size=2)
        ion_density = rng.uniform(7.0e13, 1.3e14, size=2)
        electron_temperature = rng.uniform(2.5e4, 3.5e4, size=2)
        ion_temperature = rng.uniform(250.0, 400.0, size=2)
        ion_thermal = np.sqrt(8.0 * 1.380649e-23 * ion_temperature / (math.pi * _ION_MASS_KG))
        relative = rng.normal(size=(2, 2))
        relative /= np.linalg.norm(relative, axis=1)[:, None]
        relative *= rng.uniform(0.0, 1.5, size=2)[:, None] * ion_thermal[:, None]
        ion_velocity = rng.uniform(-50.0, 50.0, size=(2, 2))
        evaluation = barnes_collisionless_ion_drag(
            mass_kg=masses,
            electrostatic_radius_m=radii,
            charge_number=rng.uniform(charge_lower, charge_upper),
            velocity_m_s=ion_velocity - relative,
            electron_number_density_m3=electron_density,
            positive_ion_number_density_m3=ion_density,
            electron_temperature_K=electron_temperature,
            positive_ion_temperature_K=ion_temperature,
            positive_ion_velocity_m_s=ion_velocity,
            ion_neutral_mean_free_path_m=np.full(2, 8.0e-3),
            positive_ion_mass_kg=_ION_MASS_KG,
            maximum_ion_drift_ratio=1.5,
        )
        assert bool(evaluation.applicable.all())
        assert bool(
            (np.abs(evaluation.acceleration_m_s2) <= bounds.acceleration_abs_upper_m_s2).all()
        )

    applicable, status = barnes_collisionless_continuous_applicability_batch(
        static_applicable=bounds.static_applicable,
        velocity_abs_upper_m_s=np.asarray([[20.0, 10.0], [2.0e3, 2.0e3]]),
        positive_ion_velocity_abs_upper_m_s=np.asarray([50.0, 50.0]),
        positive_ion_temperature_lower_K=250.0,
        positive_ion_mass_kg=_ION_MASS_KG,
        maximum_ion_drift_ratio=1.5,
    )
    assert applicable.tolist() == [True, False]
    assert status.tolist() == [0, 0]


@pytest.mark.parametrize(
    ("coordinate_system", "components", "basis"),
    [
        ("cartesian_xy", ("x", "y"), "cartesian_xy"),
        ("axisymmetric_rz", ("r", "z"), "axisymmetric_rz"),
    ],
)
def test_ion_drag_catalog_and_compiled_runtime_share_one_species_and_formula(
    coordinate_system: str,
    components: tuple[str, str],
    basis: str,
) -> None:
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "ion_drag": _ion_drag_model()},
        coordinate_system,  # type: ignore[arg-type]
    )
    assert isinstance(plan.ion_drag, BarnesCollisionlessIonDragPlan)
    assert plan.has_force
    requirements = {item.name: item for item in plan.required_fields}
    assert set(requirements) == {"ne", "ni", "te", "ti", "ui", "ion_mfp"}
    assert requirements["ui"].components == components
    assert requirements["ui"].stored_basis == basis
    assert plan.resolved_models()["ion_drag"] == {
        "model": "barnes_collisionless",
        "revision": _REVISION,
    }

    ranges = {
        "ne": _constant_range(1.0e14),
        "ni": _constant_range(1.0e14),
        "te": _constant_range(3.0e4),
        "ti": _constant_range(300.0),
        "ui": _constant_vector_range(100.0, 20.0),
        "ion_mfp": _constant_range(1.0e-2),
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system=coordinate_system,  # type: ignore[arg-type]
        mass_kg=np.asarray([1.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        electrostatic_radius_m=np.asarray([1.0e-7]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([-10.0]),
        primitive_ranges=ranges,
    )
    sampled = {
        "ne": np.asarray([[1.0e14]]),
        "ni": np.asarray([[1.0e14]]),
        "te": np.asarray([[3.0e4]]),
        "ti": np.asarray([[300.0]]),
        "ui": np.asarray([[100.0, 20.0]]),
        "ion_mfp": np.asarray([[1.0e-2]]),
    }
    actual = runtime.evaluate(
        np.asarray([0], dtype=np.int64),
        np.asarray([[0.0, 0.0]]),
        np.asarray([-10.0]),
        sampled,
    )
    expected = _evaluate(
        charge_number=np.asarray([-10.0]),
        velocity_m_s=np.asarray([[0.0, 0.0]]),
        ion_velocity_m_s=np.asarray([[100.0, 20.0]]),
    )
    np.testing.assert_allclose(actual.acceleration_m_s2, expected.acceleration_m_s2, rtol=2e-15)
    np.testing.assert_array_equal(actual.additive_acceleration_m_s2, actual.acceleration_m_s2)
    np.testing.assert_array_equal(actual.linear_drag_rate_s_inv, np.zeros(1))
    assert runtime.constant_acceleration_m_s2 is None
    assert runtime.continuous_applicability(
        np.asarray([0], dtype=np.int64),
        np.asarray([[0.0, 0.0]]),
    ).tolist() == [True]


def test_continuous_charge_and_ion_drag_reject_different_species_authorities() -> None:
    charge = {
        "model": "plasma_continuous",
        "revision": "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1",
        "electron_number_density_field": "ne",
        "positive_ion_number_density_field": "other_ni",
        "electron_temperature_field": "te",
        "positive_ion_temperature_field": "ti",
        "positive_ion_velocity_field": "ui",
        "positive_ion_mass_kg": _ION_MASS_KG,
        "maximum_ion_drift_ratio": 1.5,
        "applicability": "error",
    }
    with pytest.raises(PhysicsConfigurationError, match="same single-ion background"):
        resolve_physics_plan(
            {"charge": charge, "ion_drag": _ion_drag_model()},
            "cartesian_xy",
        )


def test_aggregate_ion_drag_catalog_enforces_shared_background_and_electric_field() -> None:
    charge = _aggregate_charge_model()
    theory = _relative_flow_ion_drag_model()
    image = _image_ion_drag_model()

    theory_plan = resolve_physics_plan(
        {"charge": charge, "ion_drag": theory},
        "axisymmetric_rz",
    )
    assert isinstance(theory_plan.ion_drag, RelativeFlowScreenedIonDragPlan)
    assert theory_plan.resolved_models()["ion_drag"] == {
        "model": "screened_collection_orbital",
        "revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
    }

    image_plan = resolve_physics_plan(
        {
            "charge": charge,
            "ion_drag": image,
            "electric": {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
                "electric_field": "efield",
            },
        },
        "axisymmetric_rz",
    )
    assert isinstance(image_plan.ion_drag, ElectricFieldDirectedImageIonDragPlan)
    assert image_plan.resolved_models()["ion_drag"] == {
        "model": "image_orbital_sensitivity",
        "revision": "electric_field_directed_image_orbital_sensitivity_v1",
    }

    mismatched_theory = {**theory, "maximum_relative_ion_speed_m_s": 2001.0}
    with pytest.raises(PhysicsConfigurationError, match="same aggregate-ion background"):
        resolve_physics_plan(
            {"charge": charge, "ion_drag": mismatched_theory},
            "axisymmetric_rz",
        )
    mismatched_electric = {
        "model": "coulomb",
        "revision": "electric_coulomb_v1",
        "electric_field": "other_electric_field",
    }
    with pytest.raises(PhysicsConfigurationError, match="same electric field"):
        resolve_physics_plan(
            {
                "charge": charge,
                "ion_drag": image,
                "electric": mismatched_electric,
            },
            "axisymmetric_rz",
        )


@pytest.mark.parametrize(
    "expected_kind",
    ["relative", "image"],
)
def test_aggregate_ion_drag_compiled_runtime_matches_pure_formula(
    expected_kind: str,
) -> None:
    model = (
        _relative_flow_ion_drag_model() if expected_kind == "relative" else _image_ion_drag_model()
    )
    plan = resolve_physics_plan(
        {"charge": _aggregate_charge_model(), "ion_drag": model},
        "cartesian_xy",
    )
    ranges = {
        "ne": _constant_range(8.0e13),
        "ni": _constant_range(1.0e14),
        "te_v": _constant_range(3.0),
        "ti_v": _constant_range(0.03),
        "ui": _constant_vector_range(100.0, 20.0),
        "mi": _constant_range(_ION_MASS_KG),
        "screening": _constant_range(1.0e-4),
        "ion_mfp": _constant_range(1.0e-3),
        "efield": _constant_vector_range(40.0, -10.0),
    }
    mass = np.asarray([1.0e-15])
    radius = np.asarray([1.0e-7])
    charge = np.asarray([-10.0])
    velocity = np.asarray([[4.0, -2.0]])
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=mass,
        drag_diameter_m=2.0 * radius,
        electrostatic_radius_m=radius,
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=charge,
        primitive_ranges={
            name: ranges[name] for name in {field.name for field in plan.required_fields}
        },
    )
    sampled = {
        "ne": np.asarray([[8.0e13]]),
        "ni": np.asarray([[1.0e14]]),
        "te_v": np.asarray([[3.0]]),
        "ti_v": np.asarray([[0.03]]),
        "ui": np.asarray([[100.0, 20.0]]),
        "mi": np.asarray([[_ION_MASS_KG]]),
        "screening": np.asarray([[1.0e-4]]),
        "ion_mfp": np.asarray([[1.0e-3]]),
        "efield": np.asarray([[40.0, -10.0]]),
    }
    actual = runtime.evaluate(
        np.asarray([0], dtype=np.int64),
        velocity,
        charge,
        {name: sampled[name] for name in {field.name for field in plan.required_fields}},
    )
    common = {
        "mass_kg": mass,
        "electrostatic_radius_m": radius,
        "charge_number": charge,
        "positive_ion_number_density_m3": sampled["ni"][:, 0],
        "positive_ion_thermal_voltage_V": sampled["ti_v"][:, 0],
        "positive_ion_velocity_m_s": sampled["ui"],
        "effective_positive_ion_mass_kg": sampled["mi"][:, 0],
        "screening_length_m": sampled["screening"][:, 0],
    }
    if expected_kind == "relative":
        expected = relative_flow_screened_collection_orbital_ion_drag(
            **common,
            velocity_m_s=velocity,
            ion_neutral_mean_free_path_m=sampled["ion_mfp"][:, 0],
            maximum_relative_ion_speed_m_s=2000.0,
        )
    else:
        expected = electric_field_directed_image_orbital_ion_drag(
            **common,
            electron_thermal_voltage_V=sampled["te_v"][:, 0],
            electric_field_V_m=sampled["efield"],
        )
    np.testing.assert_allclose(actual.acceleration_m_s2, expected.acceleration_m_s2, rtol=3e-15)
    assert actual.charge_rate_number_s[0] != 0.0
    assert actual.applicable.tolist() == [True]
    assert runtime.continuous_applicability(
        np.asarray([0], dtype=np.int64),
        np.asarray([[10.0, 10.0]]),
    ).tolist() == [True]


def test_ion_drag_reads_the_same_stage_charge_that_continuous_charge_evolves() -> None:
    common = {
        "electron_number_density_field": "ne",
        "positive_ion_number_density_field": "ni",
        "electron_temperature_field": "te",
        "positive_ion_temperature_field": "ti",
        "positive_ion_velocity_field": "ui",
        "positive_ion_mass_kg": _ION_MASS_KG,
        "maximum_ion_drift_ratio": 1.0,
        "applicability": "error",
    }
    plan = resolve_physics_plan(
        {
            "charge": {
                "model": "plasma_continuous",
                "revision": "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1",
                **common,
            },
            "ion_drag": {
                "model": "barnes_collisionless",
                "revision": _REVISION,
                "ion_neutral_mean_free_path_field": "ion_mfp",
                **common,
            },
        },
        "cartesian_xy",
    )
    ranges = {
        "ne": _constant_range(1.0e14),
        "ni": _constant_range(1.0e14),
        "te": _constant_range(1.0e5),
        "ti": _constant_range(1.0e4),
        "ui": _constant_vector_range(100.0, 0.0),
        "ion_mfp": _constant_range(1.0),
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([1.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        electrostatic_radius_m=np.asarray([1.0e-7]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([0.0]),
        primitive_ranges=ranges,
    )
    sampled = {
        "ne": np.asarray([[1.0e14]]),
        "ni": np.asarray([[1.0e14]]),
        "te": np.asarray([[1.0e5]]),
        "ti": np.asarray([[1.0e4]]),
        "ui": np.asarray([[100.0, 0.0]]),
        "ion_mfp": np.asarray([[1.0]]),
    }
    particle = np.asarray([0], dtype=np.int64)
    velocity = np.asarray([[0.0, 0.0]])
    first = runtime.evaluate(particle, velocity, np.asarray([-100.0]), sampled)
    second = runtime.evaluate(particle, velocity, np.asarray([-200.0]), sampled)
    expected = barnes_collisionless_ion_drag(
        mass_kg=np.asarray([1.0e-15]),
        electrostatic_radius_m=np.asarray([1.0e-7]),
        charge_number=np.asarray([-100.0]),
        velocity_m_s=velocity,
        electron_number_density_m3=sampled["ne"][:, 0],
        positive_ion_number_density_m3=sampled["ni"][:, 0],
        electron_temperature_K=sampled["te"][:, 0],
        positive_ion_temperature_K=sampled["ti"][:, 0],
        positive_ion_velocity_m_s=sampled["ui"],
        ion_neutral_mean_free_path_m=sampled["ion_mfp"][:, 0],
        positive_ion_mass_kg=_ION_MASS_KG,
        maximum_ion_drift_ratio=1.0,
    )
    np.testing.assert_allclose(first.acceleration_m_s2, expected.acceleration_m_s2, rtol=2e-15)
    assert first.charge_rate_number_s[0] != 0.0
    assert second.charge_rate_number_s[0] != first.charge_rate_number_s[0]
    assert second.acceleration_m_s2[0, 0] != first.acceleration_m_s2[0, 0]
    assert runtime.continuous_applicability(particle, velocity).tolist() == [True]


def _evaluate(
    *,
    charge_number: np.ndarray,
    velocity_m_s: np.ndarray,
    ion_velocity_m_s: np.ndarray,
) -> BarnesIonDragEvaluation:
    count = int(charge_number.size)
    return barnes_collisionless_ion_drag(
        mass_kg=np.full(count, 1.0e-15),
        electrostatic_radius_m=np.full(count, 1.0e-7),
        charge_number=charge_number,
        velocity_m_s=velocity_m_s,
        electron_number_density_m3=np.full(count, 1.0e14),
        positive_ion_number_density_m3=np.full(count, 1.0e14),
        electron_temperature_K=np.full(count, 3.0e4),
        positive_ion_temperature_K=np.full(count, 300.0),
        positive_ion_velocity_m_s=ion_velocity_m_s,
        ion_neutral_mean_free_path_m=np.full(count, 1.0e-2),
        positive_ion_mass_kg=_ION_MASS_KG,
        maximum_ion_drift_ratio=2.0,
    )


def _ion_drag_model() -> dict[str, object]:
    return {
        "model": "barnes_collisionless",
        "revision": _REVISION,
        "electron_number_density_field": "ne",
        "positive_ion_number_density_field": "ni",
        "electron_temperature_field": "te",
        "positive_ion_temperature_field": "ti",
        "positive_ion_velocity_field": "ui",
        "ion_neutral_mean_free_path_field": "ion_mfp",
        "positive_ion_mass_kg": _ION_MASS_KG,
        "maximum_ion_drift_ratio": 1.5,
        "applicability": "error",
    }


def _aggregate_charge_model() -> dict[str, object]:
    return {
        "model": "plasma_continuous",
        "revision": "aggregate_relative_drift_regularized_two_current_v1",
        "electron_number_density_field": "ne",
        "positive_ion_number_density_field": "ni",
        "electron_thermal_voltage_field": "te_v",
        "positive_ion_thermal_voltage_field": "ti_v",
        "positive_ion_velocity_field": "ui",
        "effective_positive_ion_mass_field": "mi",
        "screening_length_field": "screening",
        "maximum_relative_ion_speed_m_s": 2000.0,
        "applicability": "error",
    }


def _relative_flow_ion_drag_model() -> dict[str, object]:
    return {
        "model": "screened_collection_orbital",
        "revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        "positive_ion_number_density_field": "ni",
        "positive_ion_thermal_voltage_field": "ti_v",
        "positive_ion_velocity_field": "ui",
        "effective_positive_ion_mass_field": "mi",
        "screening_length_field": "screening",
        "ion_neutral_mean_free_path_field": "ion_mfp",
        "maximum_relative_ion_speed_m_s": 2000.0,
        "applicability": "error",
    }


def _image_ion_drag_model() -> dict[str, object]:
    return {
        "model": "image_orbital_sensitivity",
        "revision": "electric_field_directed_image_orbital_sensitivity_v1",
        "positive_ion_number_density_field": "ni",
        "electron_thermal_voltage_field": "te_v",
        "positive_ion_thermal_voltage_field": "ti_v",
        "positive_ion_velocity_field": "ui",
        "effective_positive_ion_mass_field": "mi",
        "screening_length_field": "screening",
        "electric_field": "efield",
        "applicability": "error",
    }


def _constant_range(value: float) -> PrimitiveRange:
    array = np.asarray([value])
    return PrimitiveRange(array, array, array)


def _constant_vector_range(first: float, second: float) -> PrimitiveRange:
    array = np.asarray([first, second])
    return PrimitiveRange(array, array, array)
