from __future__ import annotations

import math

import numpy as np
import pytest

from chamber_particles.physics.forces import (
    SAFFMAN_LIFT_COEFFICIENT,
    saffman_lift,
    saffman_lift_acceleration_abs_upper,
    saffman_lift_continuous_applicability_batch,
    saffman_lift_global_bounds,
    talbot_thermophoresis,
    talbot_thermophoresis_global_bounds,
)


@pytest.mark.parametrize("knudsen_radius", [1.0e-4, 0.1, 1.0, 10.0, 1.0e4])
@pytest.mark.parametrize("conductivity_ratio", [0.01, 0.5, 5.0])
@pytest.mark.parametrize("coefficients", [(1.17, 1.14, 2.18), (1.31, 0.87, 2.7)])
def test_talbot_matches_original_radius_force_and_point_bound(
    knudsen_radius: float,
    conductivity_ratio: float,
    coefficients: tuple[float, float, float],
) -> None:
    mass = np.asarray([2.0e-15, 2.0e-15])
    diameter = np.asarray([4.0e-7, 4.0e-7])
    temperature = np.asarray([420.0, 420.0])
    gradient = np.asarray([[3.0e3, -4.0e3], [0.0, 0.0]])
    viscosity = np.asarray([2.3e-5, 2.3e-5])
    density = np.asarray([0.42, 0.42])
    particle_conductivity = 0.21
    gas_conductivity = np.full(2, conductivity_ratio * particle_conductivity)
    radius = diameter[0] / 2.0
    mean_free_path = np.full(2, knudsen_radius * radius)
    cs, cm, ct = coefficients

    actual = talbot_thermophoresis(
        mass_kg=mass,
        drag_diameter_m=diameter,
        gas_temperature_K=temperature,
        gas_temperature_gradient_K_m=gradient,
        gas_dynamic_viscosity_Pa_s=viscosity,
        gas_density_kg_m3=density,
        gas_thermal_conductivity_W_m_K=gas_conductivity,
        gas_mean_free_path_m=mean_free_path,
        particle_thermal_conductivity_W_m_K=particle_conductivity,
        thermal_slip_coefficient=cs,
        momentum_exchange_coefficient=cm,
        thermal_exchange_coefficient=ct,
    )

    # Talbot et al., Eq. (15): radius R, viscosity-based lambda and 12*pi*R.
    knudsen = mean_free_path[0] / radius
    correction = cs * (conductivity_ratio + ct * knudsen)
    correction /= (1.0 + 3.0 * cm * knudsen) * (1.0 + 2.0 * conductivity_ratio + 2.0 * ct * knudsen)
    force_factor = 12.0 * math.pi * radius * viscosity[0] ** 2
    force_factor *= correction / (density[0] * temperature[0])
    oracle = -force_factor * gradient[0] / mass[0]

    np.testing.assert_allclose(actual.acceleration_m_s2[0], oracle, rtol=3.0e-15)
    np.testing.assert_array_equal(actual.acceleration_m_s2[1], np.zeros(2))
    assert float(np.dot(actual.acceleration_m_s2[0], gradient[0])) < 0.0
    assert actual.knudsen_radius[0] == pytest.approx(knudsen)
    assert actual.conductivity_ratio[0] == pytest.approx(conductivity_ratio)
    assert actual.applicable.tolist() == [True, True]

    # Diameter Kn is equivalent only after transforming both Cm and Ct.
    knudsen_diameter = mean_free_path[0] / diameter[0]
    cm_diameter, ct_diameter = 2.0 * cm, 2.0 * ct
    diameter_correction = cs * (conductivity_ratio + ct_diameter * knudsen_diameter)
    diameter_correction /= (1.0 + 3.0 * cm_diameter * knudsen_diameter) * (
        1.0 + 2.0 * conductivity_ratio + 2.0 * ct_diameter * knudsen_diameter
    )
    diameter_force = -6.0 * math.pi * diameter[0] * viscosity[0] ** 2
    diameter_force *= diameter_correction / (density[0] * temperature[0])
    np.testing.assert_allclose(
        actual.acceleration_m_s2[0], diameter_force * gradient[0] / mass[0], rtol=3.0e-15
    )

    point_bound = talbot_thermophoresis_global_bounds(
        mass_kg=mass,
        drag_diameter_m=diameter,
        gas_temperature_lower_K=float(temperature[0]),
        gas_temperature_gradient_abs_upper_K_m=np.abs(gradient[0]),
        gas_dynamic_viscosity_upper_Pa_s=float(viscosity[0]),
        gas_density_lower_kg_m3=float(density[0]),
        gas_thermal_conductivity_lower_W_m_K=float(gas_conductivity[0]),
        gas_thermal_conductivity_upper_W_m_K=float(gas_conductivity[0]),
        gas_mean_free_path_lower_m=float(mean_free_path[0]),
        gas_mean_free_path_upper_m=float(mean_free_path[0]),
        particle_thermal_conductivity_W_m_K=particle_conductivity,
        thermal_slip_coefficient=cs,
        momentum_exchange_coefficient=cm,
        thermal_exchange_coefficient=ct,
    )
    # Bounds include outward float64 roundoff; they must enclose the reference.
    assert bool((point_bound.acceleration_abs_upper_m_s2[0] >= np.abs(oracle)).all())
    np.testing.assert_allclose(
        point_bound.acceleration_abs_upper_m_s2[0], np.abs(oracle), rtol=2.0e-14
    )


def test_talbot_has_original_continuum_and_free_molecular_limits() -> None:
    mass, diameter, viscosity, density, temperature = 2.0e-15, 4.0e-7, 2.3e-5, 0.42, 420.0
    cs, cm, ct, conductivity_ratio = 1.31, 0.87, 2.7, 0.2
    knudsen_radius = np.asarray([1.0e-12, 1.0e12])
    actual = talbot_thermophoresis(
        mass_kg=np.full(2, mass),
        drag_diameter_m=np.full(2, diameter),
        gas_temperature_K=np.full(2, temperature),
        gas_temperature_gradient_K_m=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        gas_dynamic_viscosity_Pa_s=np.full(2, viscosity),
        gas_density_kg_m3=np.full(2, density),
        gas_thermal_conductivity_W_m_K=np.full(2, conductivity_ratio * 0.21),
        gas_mean_free_path_m=knudsen_radius * (diameter / 2.0),
        particle_thermal_conductivity_W_m_K=0.21,
        thermal_slip_coefficient=cs,
        momentum_exchange_coefficient=cm,
        thermal_exchange_coefficient=ct,
    )
    force_prefactor = 12.0 * math.pi * (diameter / 2.0) * viscosity**2 / (density * temperature)
    correction = -actual.acceleration_m_s2[:, 0] * mass / force_prefactor
    np.testing.assert_allclose(
        correction[0], cs * conductivity_ratio / (1.0 + 2.0 * conductivity_ratio), rtol=1.0e-10
    )
    np.testing.assert_allclose(knudsen_radius[1] * correction[1], cs / (6.0 * cm), rtol=1.0e-10)


def test_talbot_global_bound_contains_random_primitives_across_knudsen_regimes() -> None:
    mass = np.asarray([1.0e-15, 4.0e-15, 8.0e-15])
    diameter = np.asarray([2.0e-8, 2.0e-6, 2.0e-4])
    gradient_upper = np.asarray([2.0e4, 3.0e4])
    bounds = talbot_thermophoresis_global_bounds(
        mass_kg=mass,
        drag_diameter_m=diameter,
        gas_temperature_lower_K=250.0,
        gas_temperature_gradient_abs_upper_K_m=gradient_upper,
        gas_dynamic_viscosity_upper_Pa_s=4.0e-5,
        gas_density_lower_kg_m3=0.2,
        gas_thermal_conductivity_lower_W_m_K=0.01,
        gas_thermal_conductivity_upper_W_m_K=0.08,
        gas_mean_free_path_lower_m=1.0e-8,
        gas_mean_free_path_upper_m=2.0e-5,
        particle_thermal_conductivity_W_m_K=0.2,
        thermal_slip_coefficient=1.31,
        momentum_exchange_coefficient=0.87,
        thermal_exchange_coefficient=2.7,
    )
    assert bool(bounds.static_applicable.all())

    rng = np.random.default_rng(20261007)
    for _ in range(96):
        evaluation = talbot_thermophoresis(
            mass_kg=mass,
            drag_diameter_m=diameter,
            gas_temperature_K=rng.uniform(250.0, 800.0, size=3),
            gas_temperature_gradient_K_m=rng.uniform(
                -gradient_upper,
                gradient_upper,
                size=(3, 2),
            ),
            gas_dynamic_viscosity_Pa_s=rng.uniform(1.0e-5, 4.0e-5, size=3),
            gas_density_kg_m3=rng.uniform(0.2, 1.2, size=3),
            gas_thermal_conductivity_W_m_K=rng.uniform(0.01, 0.08, size=3),
            gas_mean_free_path_m=rng.uniform(1.0e-8, 2.0e-5, size=3),
            particle_thermal_conductivity_W_m_K=0.2,
            thermal_slip_coefficient=1.31,
            momentum_exchange_coefficient=0.87,
            thermal_exchange_coefficient=2.7,
        )
        assert bool(
            (np.abs(evaluation.acceleration_m_s2) <= bounds.acceleration_abs_upper_m_s2).all()
        )


@pytest.mark.parametrize(
    ("coordinate_system", "expected_sign"),
    [("cartesian_xy", 1.0), ("axisymmetric_rz", -1.0)],
)
def test_saffman_matches_cross_product_oracle_and_coordinate_orientation(
    coordinate_system: str,
    expected_sign: float,
) -> None:
    mass = np.asarray([3.0e-12, 3.0e-12, 3.0e-12])
    diameter = np.asarray([2.0e-4, 2.0e-4, 2.0e-4])
    velocity = np.asarray([[0.02, 0.01], [0.02, 0.01], [0.02, 0.01]])
    gas_velocity = np.asarray([[0.024, 0.008], [0.02, 0.01], [0.024, 0.008]])
    density = np.full(3, 1.1)
    viscosity = np.full(3, 1.8e-3)
    mean_free_path = np.full(3, 5.0e-6)
    vorticity = np.asarray([40.0, 40.0, 0.0])

    actual = saffman_lift(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=velocity,
        gas_velocity_m_s=gas_velocity,
        gas_density_kg_m3=density,
        gas_dynamic_viscosity_Pa_s=viscosity,
        gas_mean_free_path_m=mean_free_path,
        out_of_plane_gas_vorticity_s_inv=vorticity,
        coordinate_system=coordinate_system,  # type: ignore[arg-type]
    )

    slip = gas_velocity[0] - velocity[0]
    if coordinate_system == "cartesian_xy":
        slip_3d = np.asarray([slip[0], slip[1], 0.0])
        omega_3d = np.asarray([0.0, 0.0, vorticity[0]])
        projected = np.cross(slip_3d, omega_3d)[:2]
    else:
        slip_3d = np.asarray([slip[0], 0.0, slip[1]])
        omega_3d = np.asarray([0.0, vorticity[0], 0.0])
        force_3d = np.cross(slip_3d, omega_3d)
        projected = force_3d[[0, 2]]
    lift_vector_norm = float(np.linalg.norm(projected))
    force = SAFFMAN_LIFT_COEFFICIENT * (0.5 * diameter[0]) ** 2 * projected
    force *= math.sqrt(viscosity[0] * density[0] * np.linalg.norm(slip) / lift_vector_norm)
    oracle = force / mass[0]

    np.testing.assert_allclose(actual.acceleration_m_s2[0], oracle, rtol=3.0e-15)
    assert math.copysign(1.0, actual.acceleration_m_s2[0, 0]) == expected_sign * -1.0
    np.testing.assert_array_equal(actual.acceleration_m_s2[1:], np.zeros((2, 2)))
    assert bool(actual.applicable.all())


def test_saffman_scaling_and_applicability_gates_are_radius_based() -> None:
    base = saffman_lift(
        mass_kg=np.asarray([2.0e-12]),
        drag_diameter_m=np.asarray([2.0e-4]),
        velocity_m_s=np.asarray([[0.0, 0.0]]),
        gas_velocity_m_s=np.asarray([[0.01, 0.0]]),
        gas_density_kg_m3=np.asarray([1.0]),
        gas_dynamic_viscosity_Pa_s=np.asarray([1.0e-3]),
        gas_mean_free_path_m=np.asarray([1.0e-5]),
        out_of_plane_gas_vorticity_s_inv=np.asarray([100.0]),
        coordinate_system="cartesian_xy",
    )
    scaled = saffman_lift(
        mass_kg=np.asarray([2.0e-12]),
        drag_diameter_m=np.asarray([4.0e-4]),
        velocity_m_s=np.asarray([[0.0, 0.0]]),
        gas_velocity_m_s=np.asarray([[0.01, 0.0]]),
        gas_density_kg_m3=np.asarray([1.0]),
        gas_dynamic_viscosity_Pa_s=np.asarray([4.0e-3]),
        gas_mean_free_path_m=np.asarray([1.0e-5]),
        out_of_plane_gas_vorticity_s_inv=np.asarray([400.0]),
        coordinate_system="cartesian_xy",
    )
    np.testing.assert_allclose(
        scaled.acceleration_m_s2,
        16.0 * base.acceleration_m_s2,
        rtol=3.0e-15,
    )
    assert base.mean_free_path_over_radius[0] == pytest.approx(0.1)
    assert base.slip_reynolds_radius[0] == pytest.approx(0.001)
    assert base.shear_reynolds_radius[0] == pytest.approx(0.001)
    assert bool(base.applicable[0])

    rejected = saffman_lift(
        mass_kg=np.full(4, 2.0e-12),
        drag_diameter_m=np.full(4, 2.0e-4),
        velocity_m_s=np.zeros((4, 2)),
        gas_velocity_m_s=np.asarray([[0.001, 0.0], [2.0, 0.0], [0.001, 0.0], [0.1, 0.0]]),
        gas_density_kg_m3=np.ones(4),
        gas_dynamic_viscosity_Pa_s=np.full(4, 1.0e-3),
        gas_mean_free_path_m=np.asarray([2.0e-5, 1.0e-5, 1.0e-5, 1.0e-5]),
        out_of_plane_gas_vorticity_s_inv=np.asarray([100.0, 100.0, 2.0e4, 100.0]),
        coordinate_system="cartesian_xy",
    )
    assert rejected.applicable.tolist() == [False, False, False, False]


def test_saffman_global_and_continuous_bounds_contain_random_samples() -> None:
    mass = np.asarray([1.0e-12, 2.0e-12])
    diameter = np.asarray([2.0e-4, 4.0e-4])
    gas_velocity_upper = np.asarray([1.0e-4, 1.0e-4])
    velocity_upper = np.asarray([[2.0e-4, 2.0e-4], [2.0e-4, 2.0e-4]])
    bounds = saffman_lift_global_bounds(
        mass_kg=mass,
        drag_diameter_m=diameter,
        gas_density_lower_kg_m3=0.5,
        gas_density_upper_kg_m3=2.0,
        gas_dynamic_viscosity_lower_Pa_s=1.0e-3,
        gas_dynamic_viscosity_upper_Pa_s=3.0e-3,
        gas_mean_free_path_upper_m=5.0e-6,
        out_of_plane_gas_vorticity_abs_lower_s_inv=100.0,
        out_of_plane_gas_vorticity_abs_upper_s_inv=100.0,
        gas_velocity_abs_upper_m_s=gas_velocity_upper,
    )
    assert bool(bounds.static_applicable.all())
    acceleration_bound = saffman_lift_acceleration_abs_upper(
        coupling_rate_abs_upper_s_inv=bounds.coupling_rate_abs_upper_s_inv,
        gas_velocity_abs_upper_m_s=bounds.gas_velocity_abs_upper_m_s,
        velocity_abs_upper_m_s=velocity_upper,
    )
    applicable, status = saffman_lift_continuous_applicability_batch(
        static_applicable=bounds.static_applicable,
        drag_diameter_m=diameter,
        velocity_abs_upper_m_s=velocity_upper,
        gas_velocity_abs_upper_m_s=gas_velocity_upper,
        gas_density_upper_kg_m3=2.0,
        gas_dynamic_viscosity_lower_Pa_s=1.0e-3,
        shear_reynolds_lower=bounds.shear_reynolds_lower,
        shear_reynolds_upper=bounds.shear_reynolds_upper,
    )
    assert applicable.tolist() == [True, True]
    assert status.tolist() == [0, 0]

    rng = np.random.default_rng(20261008)
    for coordinate_system in ("cartesian_xy", "axisymmetric_rz"):
        for _ in range(64):
            evaluation = saffman_lift(
                mass_kg=mass,
                drag_diameter_m=diameter,
                velocity_m_s=rng.uniform(-velocity_upper, velocity_upper),
                gas_velocity_m_s=rng.uniform(
                    -gas_velocity_upper,
                    gas_velocity_upper,
                    size=(2, 2),
                ),
                gas_density_kg_m3=rng.uniform(0.5, 2.0, size=2),
                gas_dynamic_viscosity_Pa_s=rng.uniform(1.0e-3, 3.0e-3, size=2),
                gas_mean_free_path_m=rng.uniform(1.0e-7, 5.0e-6, size=2),
                out_of_plane_gas_vorticity_s_inv=rng.uniform(-100.0, 100.0, size=2),
                coordinate_system=coordinate_system,  # type: ignore[arg-type]
            )
            assert bool((np.abs(evaluation.acceleration_m_s2) <= acceleration_bound).all())

    too_fast, fast_status = saffman_lift_continuous_applicability_batch(
        static_applicable=bounds.static_applicable,
        drag_diameter_m=diameter,
        velocity_abs_upper_m_s=np.full((2, 2), 2.0),
        gas_velocity_abs_upper_m_s=gas_velocity_upper,
        gas_density_upper_kg_m3=2.0,
        gas_dynamic_viscosity_lower_Pa_s=1.0e-3,
        shear_reynolds_lower=bounds.shear_reynolds_lower,
        shear_reynolds_upper=bounds.shear_reynolds_upper,
    )
    assert too_fast.tolist() == [False, False]
    assert fast_status.tolist() == [0, 0]
