from __future__ import annotations

import math

import numpy as np
import pytest

from chamber_particles.physics.catalog import (
    PhysicsConfigurationError,
    RarefiedVorticityLiftPlan,
    resolve_physics_plan,
)
from chamber_particles.physics.forces import (
    rarefied_vorticity_lift,
    rarefied_vorticity_lift_acceleration_abs_upper,
    rarefied_vorticity_lift_global_bounds,
)
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime

_REVISION = "rarefied_vorticity_sensitivity_rz_v1"
_LIFT_COEFFICIENT = 0.75


def test_rarefied_lift_matches_independent_3d_cross_product_and_scaling() -> None:
    count = 7
    mass = np.full(count, 2.0e-15)
    diameter = np.full(count, 2.0e-7)
    density = np.full(count, 8.0e-2)
    mean_free_path = np.full(count, 2.0e-4)
    vorticity = np.full(count, -13.0)
    velocity = np.repeat(np.asarray([[0.3, 0.4]]), count, axis=0)
    gas_velocity = np.repeat(np.asarray([[1.2, -0.7]]), count, axis=0)

    density[1] *= 2.0
    mean_free_path[2] *= 3.0
    diameter[3] *= 2.0
    mass[4] *= 2.0
    vorticity[5] = 0.0
    gas_velocity[6] = velocity[6]
    actual = rarefied_vorticity_lift(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=velocity,
        gas_velocity_m_s=gas_velocity,
        gas_density_kg_m3=density,
        gas_mean_free_path_m=mean_free_path,
        azimuthal_gas_vorticity_s_inv=vorticity,
        lift_coefficient=_LIFT_COEFFICIENT,
    )

    angle = 0.61
    radial = np.asarray([math.cos(angle), math.sin(angle), 0.0])
    azimuthal = np.asarray([-math.sin(angle), math.cos(angle), 0.0])
    axial = np.asarray([0.0, 0.0, 1.0])
    slip_rz = gas_velocity[0] - velocity[0]
    slip_3d = slip_rz[0] * radial + slip_rz[1] * axial
    omega_3d = vorticity[0] * azimuthal
    force_scale_kg = (
        _LIFT_COEFFICIENT * math.pi * density[0] * mean_free_path[0] * (0.5 * diameter[0]) ** 2
    )
    force_3d = force_scale_kg * np.cross(omega_3d, slip_3d)
    oracle = np.asarray([np.dot(force_3d, radial), np.dot(force_3d, axial)]) / mass[0]
    np.testing.assert_allclose(actual.acceleration_m_s2[0], oracle, rtol=3.0e-15, atol=0.0)

    np.testing.assert_allclose(
        actual.acceleration_m_s2[1],
        2.0 * actual.acceleration_m_s2[0],
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.acceleration_m_s2[2],
        3.0 * actual.acceleration_m_s2[0],
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.acceleration_m_s2[3],
        4.0 * actual.acceleration_m_s2[0],
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.acceleration_m_s2[4],
        0.5 * actual.acceleration_m_s2[0],
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(actual.acceleration_m_s2[5:], np.zeros((2, 2)))
    assert bool(actual.applicable.all())

    slip = gas_velocity[0] - velocity[0]
    work_rate = float(np.dot(actual.acceleration_m_s2[0], slip))
    scale = float(np.linalg.norm(actual.acceleration_m_s2[0]) * np.linalg.norm(slip))
    assert abs(work_rate) <= 5.0e-15 * scale

    opposite = rarefied_vorticity_lift(
        mass_kg=mass[:1],
        drag_diameter_m=diameter[:1],
        velocity_m_s=velocity[:1],
        gas_velocity_m_s=gas_velocity[:1],
        gas_density_kg_m3=density[:1],
        gas_mean_free_path_m=mean_free_path[:1],
        azimuthal_gas_vorticity_s_inv=-vorticity[:1],
        lift_coefficient=_LIFT_COEFFICIENT,
    )
    np.testing.assert_allclose(
        opposite.acceleration_m_s2,
        -actual.acceleration_m_s2[:1],
        rtol=0.0,
        atol=0.0,
    )


def test_rarefied_lift_dynamic_bound_contains_declared_box_and_kn_gate() -> None:
    mass = np.asarray([1.0e-15, 2.0e-15])
    diameter = np.asarray([2.0e-7, 4.0e-4])
    density_upper = 2.0e-1
    mean_free_path_lower = 1.0e-3
    mean_free_path_upper = 2.0e-3
    vorticity_upper = 7.0
    gas_velocity_upper = np.asarray([2.0, 5.0])
    velocity_upper = np.asarray([[7.0, 11.0], [3.0, 13.0]])
    bounds = rarefied_vorticity_lift_global_bounds(
        mass_kg=mass,
        drag_diameter_m=diameter,
        gas_density_upper_kg_m3=density_upper,
        gas_mean_free_path_lower_m=mean_free_path_lower,
        gas_mean_free_path_upper_m=mean_free_path_upper,
        azimuthal_gas_vorticity_abs_upper_s_inv=vorticity_upper,
        gas_velocity_abs_upper_m_s=gas_velocity_upper,
        lift_coefficient=_LIFT_COEFFICIENT,
    )
    assert bounds.static_applicable.tolist() == [True, False]
    bound = rarefied_vorticity_lift_acceleration_abs_upper(
        coupling_rate_abs_upper_s_inv=bounds.coupling_rate_abs_upper_s_inv,
        gas_velocity_abs_upper_m_s=bounds.gas_velocity_abs_upper_m_s,
        velocity_abs_upper_m_s=velocity_upper,
    )
    expected_rate = (
        _LIFT_COEFFICIENT
        * math.pi
        * density_upper
        * mean_free_path_upper
        * (0.5 * diameter) ** 2
        * vorticity_upper
        / mass
    )
    expected = expected_rate[:, None] * np.column_stack(
        (
            gas_velocity_upper[1] + velocity_upper[:, 1],
            gas_velocity_upper[0] + velocity_upper[:, 0],
        )
    )
    assert bool((bounds.coupling_rate_abs_upper_s_inv >= expected_rate).all())
    assert bool((bound >= expected).all())
    np.testing.assert_allclose(bound, expected, rtol=3.0e-14, atol=0.0)

    rng = np.random.default_rng(20261001)
    for _ in range(64):
        density = rng.uniform(1.0e-2, density_upper, size=2)
        mean_free_path = rng.uniform(
            mean_free_path_lower,
            mean_free_path_upper,
            size=2,
        )
        vorticity = rng.uniform(-vorticity_upper, vorticity_upper, size=2)
        gas_velocity = rng.uniform(-gas_velocity_upper, gas_velocity_upper, size=(2, 2))
        velocity = rng.uniform(-velocity_upper, velocity_upper, size=(2, 2))
        evaluation = rarefied_vorticity_lift(
            mass_kg=mass,
            drag_diameter_m=diameter,
            velocity_m_s=velocity,
            gas_velocity_m_s=gas_velocity,
            gas_density_kg_m3=density,
            gas_mean_free_path_m=mean_free_path,
            azimuthal_gas_vorticity_s_inv=vorticity,
            lift_coefficient=_LIFT_COEFFICIENT,
        )
        assert bool((np.abs(evaluation.acceleration_m_s2) <= bound).all())


def test_rarefied_lift_catalog_and_compiled_runtime_share_one_rz_formula() -> None:
    with pytest.raises(PhysicsConfigurationError, match="axisymmetric_rz"):
        resolve_physics_plan(
            {"charge": {"model": "fixed"}, "lift": _model()},
            "cartesian_xy",
        )

    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "lift": _model()},
        "axisymmetric_rz",
    )
    assert isinstance(plan.lift, RarefiedVorticityLiftPlan)
    requirements = {item.name: item for item in plan.required_fields}
    assert set(requirements) == {"ug", "rho", "mfp", "omega"}
    assert requirements["ug"].components == ("r", "z")
    assert requirements["ug"].stored_basis == "axisymmetric_rz"
    assert requirements["ug"].unit == "m/s"
    assert requirements["rho"].unit == "kg/m^3"
    assert requirements["mfp"].unit == "m"
    assert requirements["omega"].unit == "1/s"
    assert plan.resolved_models()["lift"] == {
        "model": "rarefied_vorticity_sensitivity",
        "revision": _REVISION,
        "lift_coefficient": _LIFT_COEFFICIENT,
    }

    mass = np.asarray([2.0e-15, 2.0e-15])
    diameter = np.asarray([2.0e-7, 4.0e-4])
    velocity = np.asarray([[0.3, 0.4], [-0.1, 0.2]])
    gas_velocity = np.asarray([[1.2, -0.7], [1.2, -0.7]])
    density = np.asarray([8.0e-2, 8.0e-2])
    mean_free_path = np.asarray([1.0e-3, 1.0e-3])
    vorticity = np.asarray([-13.0, -13.0])
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="axisymmetric_rz",
        mass_kg=mass,
        drag_diameter_m=diameter,
        electrostatic_radius_m=np.zeros(2),
        displaced_volume_m3=np.zeros(2),
        charge_number=np.zeros(2),
        primitive_ranges={
            "ug": _constant_vector_range(1.2, -0.7),
            "rho": _constant_range(8.0e-2),
            "mfp": _constant_range(1.0e-3),
            "omega": _constant_range(-13.0),
        },
    )
    sampled = {
        "ug": gas_velocity,
        "rho": density[:, None],
        "mfp": mean_free_path[:, None],
        "omega": vorticity[:, None],
    }
    actual = runtime.evaluate(
        np.asarray([0, 1], dtype=np.int64),
        velocity,
        np.zeros(2),
        sampled,
    )
    expected = rarefied_vorticity_lift(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=velocity,
        gas_velocity_m_s=gas_velocity,
        gas_density_kg_m3=density,
        gas_mean_free_path_m=mean_free_path,
        azimuthal_gas_vorticity_s_inv=vorticity,
        lift_coefficient=_LIFT_COEFFICIENT,
    )
    np.testing.assert_allclose(
        actual.acceleration_m_s2,
        expected.acceleration_m_s2,
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(actual.applicable, expected.applicable)
    np.testing.assert_array_equal(actual.additive_acceleration_m_s2, actual.acceleration_m_s2)
    np.testing.assert_array_equal(actual.linear_drag_rate_s_inv, np.zeros(2))
    assert runtime.constant_acceleration_m_s2 is None
    assert runtime.continuous_applicability(
        np.asarray([0, 1], dtype=np.int64),
        np.abs(velocity),
    ).tolist() == [True, False]
    acceleration_bound = runtime.acceleration_abs_upper(
        np.asarray([0, 1], dtype=np.int64),
        np.abs(velocity),
    )
    assert bool((np.abs(actual.acceleration_m_s2) <= acceleration_bound).all())


def test_rarefied_lift_requires_one_compatible_rarefied_gas_background() -> None:
    epstein = {
        "model": "epstein_linear",
        "revision": "epstein_linear_v1",
        "gas_velocity_field": "other_ug",
        "gas_density_field": "rho",
        "gas_temperature_field": "tg",
        "gas_mean_free_path_field": "mfp",
        "gas_molecular_mass_kg": 6.63e-26,
        "delta": 1.0,
        "applicability": "error",
    }
    with pytest.raises(PhysicsConfigurationError, match="same neutral-gas background"):
        resolve_physics_plan(
            {"charge": {"model": "fixed"}, "drag": epstein, "lift": _model()},
            "axisymmetric_rz",
        )

    stokes = {
        "model": "stokes_cunningham",
        "revision": "stokes_cunningham_allen_raabe_air_v1",
        "gas_velocity_field": "ug",
        "gas_density_field": "rho",
        "gas_dynamic_viscosity_field": "mu",
        "gas_mean_free_path_field": "mfp",
        "applicability": "error",
    }
    with pytest.raises(PhysicsConfigurationError, match="no applicability overlap"):
        resolve_physics_plan(
            {"charge": {"model": "fixed"}, "drag": stokes, "lift": _model()},
            "axisymmetric_rz",
        )


def _model() -> dict[str, object]:
    return {
        "model": "rarefied_vorticity_sensitivity",
        "revision": _REVISION,
        "gas_velocity_field": "ug",
        "gas_density_field": "rho",
        "gas_mean_free_path_field": "mfp",
        "azimuthal_gas_vorticity_field": "omega",
        "lift_coefficient": _LIFT_COEFFICIENT,
        "applicability": "error",
    }


def _constant_range(value: float) -> PrimitiveRange:
    array = np.asarray([value])
    return PrimitiveRange(array, array, array)


def _constant_vector_range(first: float, second: float) -> PrimitiveRange:
    array = np.asarray([first, second])
    return PrimitiveRange(array, array, array)
