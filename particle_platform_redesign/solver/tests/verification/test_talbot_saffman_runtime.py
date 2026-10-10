from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from chamber_particles.physics.catalog import (
    PhysicsConfigurationError,
    SaffmanLiftPlan,
    TalbotThermophoresisPlan,
    resolve_physics_plan,
)
from chamber_particles.physics.forces import saffman_lift, talbot_thermophoresis
from chamber_particles.physics.runtime import (
    LocalPrimitiveRange,
    PrimitiveRange,
    prepare_physics_runtime,
)


@pytest.mark.parametrize("coordinate_system", ["cartesian_xy", "axisymmetric_rz"])
def test_talbot_is_an_explicit_catalog_option_and_compiled_formula(
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"],
) -> None:
    model = _talbot_model()
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "thermophoresis": model},
        coordinate_system,
    )
    assert isinstance(plan.thermophoresis, TalbotThermophoresisPlan)
    assert plan.resolved_models()["thermophoresis"] == {
        "model": "talbot",
        "revision": "talbot_cross_regime_radius_knudsen_v1",
        "particle_thermal_conductivity_W_m_K": 0.21,
        "thermal_slip_coefficient": 1.31,
        "momentum_exchange_coefficient": 0.87,
        "thermal_exchange_coefficient": 2.7,
    }
    sampled = {
        "temperature": np.asarray([[420.0]]),
        "temperature_gradient": np.asarray([[3000.0, -4000.0]]),
        "density": np.asarray([[0.42]]),
        "viscosity": np.asarray([[2.3e-5]]),
        "conductivity": np.asarray([[0.031]]),
        "mean_free_path": np.asarray([[8.0e-7]]),
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system=coordinate_system,
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([4.0e-7]),
        electrostatic_radius_m=np.zeros(1),
        displaced_volume_m3=np.zeros(1),
        charge_number=np.zeros(1),
        primitive_ranges={name: _constant_range(value[0]) for name, value in sampled.items()},
    )
    actual = runtime.evaluate(
        np.asarray([0], dtype=np.int64),
        np.zeros((1, 2)),
        np.zeros(1),
        sampled,
    )
    expected = talbot_thermophoresis(
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([4.0e-7]),
        gas_temperature_K=sampled["temperature"][:, 0],
        gas_temperature_gradient_K_m=sampled["temperature_gradient"],
        gas_dynamic_viscosity_Pa_s=sampled["viscosity"][:, 0],
        gas_density_kg_m3=sampled["density"][:, 0],
        gas_thermal_conductivity_W_m_K=sampled["conductivity"][:, 0],
        gas_mean_free_path_m=sampled["mean_free_path"][:, 0],
        particle_thermal_conductivity_W_m_K=0.21,
        thermal_slip_coefficient=1.31,
        momentum_exchange_coefficient=0.87,
        thermal_exchange_coefficient=2.7,
    )
    np.testing.assert_allclose(actual.acceleration_m_s2, expected.acceleration_m_s2, rtol=3e-15)
    assert actual.applicable.tolist() == [True]


@pytest.mark.parametrize(
    "damage",
    [
        "revision",
        "thermal_slip_coefficient",
        "momentum_exchange_coefficient",
        "thermal_exchange_coefficient",
    ],
)
def test_talbot_catalog_requires_radius_revision_and_explicit_coefficients(damage: str) -> None:
    model = _talbot_model()
    if damage == "revision":
        model[damage] = "talbot_cross_regime_diameter_knudsen_v1"
    else:
        del model[damage]
    with pytest.raises(PhysicsConfigurationError):
        resolve_physics_plan(
            {"charge": {"model": "fixed"}, "thermophoresis": model}, "cartesian_xy"
        )


def _talbot_model() -> dict[str, object]:
    return {
        "model": "talbot",
        "revision": "talbot_cross_regime_radius_knudsen_v1",
        "gas_temperature_field": "temperature",
        "gas_temperature_gradient_field": "temperature_gradient",
        "gas_density_field": "density",
        "gas_dynamic_viscosity_field": "viscosity",
        "gas_thermal_conductivity_field": "conductivity",
        "gas_mean_free_path_field": "mean_free_path",
        "particle_thermal_conductivity_W_m_K": 0.21,
        "thermal_slip_coefficient": 1.31,
        "momentum_exchange_coefficient": 0.87,
        "thermal_exchange_coefficient": 2.7,
        "applicability": "error",
    }


@pytest.mark.parametrize("coordinate_system", ["cartesian_xy", "axisymmetric_rz"])
def test_saffman_is_an_explicit_catalog_option_and_compiled_formula(
    coordinate_system: str,
) -> None:
    model = {
        "model": "saffman",
        "revision": "saffman_unbounded_creeping_shear_v1",
        "gas_velocity_field": "gas_velocity",
        "gas_density_field": "density",
        "gas_dynamic_viscosity_field": "viscosity",
        "gas_mean_free_path_field": "mean_free_path",
        "out_of_plane_gas_vorticity_field": "vorticity",
        "applicability": "error",
    }
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "lift": model},
        coordinate_system,  # type: ignore[arg-type]
    )
    assert isinstance(plan.lift, SaffmanLiftPlan)
    assert plan.resolved_models()["lift"] == {
        "model": "saffman",
        "revision": model["revision"],
    }
    sampled = {
        "gas_velocity": np.asarray([[0.004, -0.002]]),
        "density": np.asarray([[1.1]]),
        "viscosity": np.asarray([[1.8e-3]]),
        "mean_free_path": np.asarray([[5.0e-6]]),
        "vorticity": np.asarray([[40.0]]),
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system=coordinate_system,  # type: ignore[arg-type]
        mass_kg=np.asarray([3.0e-12]),
        drag_diameter_m=np.asarray([2.0e-4]),
        electrostatic_radius_m=np.zeros(1),
        displaced_volume_m3=np.zeros(1),
        charge_number=np.zeros(1),
        primitive_ranges={name: _constant_range(value[0]) for name, value in sampled.items()},
    )
    velocity = np.zeros((1, 2))
    actual = runtime.evaluate(
        np.asarray([0], dtype=np.int64),
        velocity,
        np.zeros(1),
        sampled,
    )
    expected = saffman_lift(
        mass_kg=np.asarray([3.0e-12]),
        drag_diameter_m=np.asarray([2.0e-4]),
        velocity_m_s=velocity,
        gas_velocity_m_s=sampled["gas_velocity"],
        gas_density_kg_m3=sampled["density"][:, 0],
        gas_dynamic_viscosity_Pa_s=sampled["viscosity"][:, 0],
        gas_mean_free_path_m=sampled["mean_free_path"][:, 0],
        out_of_plane_gas_vorticity_s_inv=sampled["vorticity"][:, 0],
        coordinate_system=coordinate_system,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(actual.acceleration_m_s2, expected.acceleration_m_s2, rtol=3e-15)
    np.testing.assert_array_equal(actual.applicable, expected.applicable)
    assert runtime.continuous_applicability(
        np.asarray([0], dtype=np.int64),
        np.abs(velocity),
    ).tolist() == [True]


def test_saffman_exact_zero_vorticity_certifies_nonzero_slip_globally_and_locally() -> None:
    model = {
        "model": "saffman",
        "revision": "saffman_unbounded_creeping_shear_v1",
        "gas_velocity_field": "gas_velocity",
        "gas_density_field": "density",
        "gas_dynamic_viscosity_field": "viscosity",
        "gas_mean_free_path_field": "mean_free_path",
        "out_of_plane_gas_vorticity_field": "vorticity",
        "applicability": "error",
    }
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "lift": model},
        "cartesian_xy",
    )
    sampled = {
        "gas_velocity": np.asarray([[0.004, -0.002]]),
        "density": np.asarray([[1.1]]),
        "viscosity": np.asarray([[1.8e-3]]),
        "mean_free_path": np.asarray([[5.0e-6]]),
        "vorticity": np.asarray([[0.0]]),
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([3.0e-12]),
        drag_diameter_m=np.asarray([2.0e-4]),
        electrostatic_radius_m=np.zeros(1),
        displaced_volume_m3=np.zeros(1),
        charge_number=np.zeros(1),
        primitive_ranges={name: _constant_range(value[0]) for name, value in sampled.items()},
    )
    particle = np.asarray([0], dtype=np.int64)
    velocity = np.zeros((1, 2))

    np.testing.assert_array_equal(runtime.continuous_applicability(particle, velocity), [True])
    local_ranges = {
        name: LocalPrimitiveRange(value.copy(), value.copy()) for name, value in sampled.items()
    }
    local, status = runtime.local_continuous_applicability_batch(
        particle,
        velocity,
        velocity,
        np.zeros(1),
        np.zeros(1),
        local_ranges,
    )
    np.testing.assert_array_equal(local, [True])
    np.testing.assert_array_equal(status, [0])
    evaluation = runtime.evaluate(particle, velocity, np.zeros(1), sampled)
    np.testing.assert_array_equal(evaluation.acceleration_m_s2, np.zeros((1, 2)))
    np.testing.assert_array_equal(evaluation.applicable, [True])


def _constant_range(value: np.ndarray) -> PrimitiveRange:
    array = np.asarray(value, dtype=np.float64)
    return PrimitiveRange(array, array, array)
