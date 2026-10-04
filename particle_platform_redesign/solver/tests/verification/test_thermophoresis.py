from __future__ import annotations

import math

import numpy as np
import pytest

from chamber_particles.physics.catalog import (
    PhysicsConfigurationError,
    WaldmannGallisThermophoresisPlan,
    resolve_physics_plan,
)
from chamber_particles.physics.forces import (
    BOLTZMANN_J_K,
    PhysicsEvaluationError,
    epstein_linear_relaxation,
    waldmann_gallis_continuous_applicability_batch,
    waldmann_gallis_global_bounds,
    waldmann_gallis_thermophoresis,
)
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime

_ARGON_MASS_KG = 6.6335209e-26
_REVISION = "waldmann_gallis_free_molecular_single_species_heat_flux_v1"


def test_waldmann_coefficient_matches_independent_chapman_enskog_moments() -> None:
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    first = nodes[:, None, None]
    second = nodes[None, :, None]
    third = nodes[None, None, :]
    weight = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    speed_square = first * first + second * second + third * third
    perturbation = first * (speed_square - 2.5)

    momentum_moment = float(np.sum(weight * np.sqrt(speed_square) * first * perturbation))
    heat_flux_moment = float(np.sum(weight * first * speed_square * perturbation))
    force_over_heat_flux_times_most_probable_speed = 2.0 * momentum_moment / heat_flux_moment
    mean_over_most_probable_speed = 2.0 / math.sqrt(math.pi)
    thermophoresis_parameter = (
        force_over_heat_flux_times_most_probable_speed * mean_over_most_probable_speed
    )

    assert thermophoresis_parameter == pytest.approx(
        32.0 / (15.0 * math.pi),
        rel=5.0e-6,
    )
    particle_mass_kg = 2.3e-15
    particle_radius_m = 1.7e-7
    gas_temperature_K = 430.0
    heat_flux_W_m2 = np.asarray([[2.5, -1.25]])
    mean_speed_m_s = math.sqrt(8.0 * BOLTZMANN_J_K * gas_temperature_K / (math.pi * _ARGON_MASS_KG))
    expected_acceleration = (
        thermophoresis_parameter
        * math.pi
        * particle_radius_m**2
        / (particle_mass_kg * mean_speed_m_s)
        * heat_flux_W_m2
    )
    production = waldmann_gallis_thermophoresis(
        mass_kg=np.asarray([particle_mass_kg]),
        drag_diameter_m=np.asarray([2.0 * particle_radius_m]),
        velocity_m_s=np.zeros((1, 2)),
        gas_velocity_m_s=np.zeros((1, 2)),
        gas_temperature_K=np.asarray([gas_temperature_K]),
        gas_translational_heat_flux_W_m2=heat_flux_W_m2,
        gas_mean_free_path_m=np.asarray([1.0e-3]),
        gas_molecular_mass_kg=_ARGON_MASS_KG,
    )
    np.testing.assert_allclose(
        production.acceleration_m_s2,
        expected_acceleration,
        rtol=5.0e-6,
    )


def test_waldmann_force_direction_zero_limit_and_particle_scaling() -> None:
    mass = np.asarray([2.0e-15, 2.0e-15, 4.0e-15, 2.0e-15])
    diameter = np.asarray([2.0e-7, 4.0e-7, 2.0e-7, 2.0e-7])
    temperature = np.asarray([300.0, 300.0, 300.0, 1200.0])
    heat_flux = np.asarray([[3.0, -4.0], [3.0, -4.0], [3.0, -4.0], [3.0, -4.0]])
    evaluation = waldmann_gallis_thermophoresis(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=np.zeros((4, 2)),
        gas_velocity_m_s=np.zeros((4, 2)),
        gas_temperature_K=temperature,
        gas_translational_heat_flux_W_m2=heat_flux,
        gas_mean_free_path_m=np.full(4, 1.0e-3),
        gas_molecular_mass_kg=_ARGON_MASS_KG,
    )

    assert bool(evaluation.applicable.all())
    assert float(np.dot(evaluation.acceleration_m_s2[0], heat_flux[0])) > 0.0
    cross = evaluation.acceleration_m_s2[0, 0] * heat_flux[0, 1]
    cross -= evaluation.acceleration_m_s2[0, 1] * heat_flux[0, 0]
    assert cross == pytest.approx(0.0, abs=1.0e-28)
    np.testing.assert_allclose(
        evaluation.acceleration_m_s2[1],
        4.0 * evaluation.acceleration_m_s2[0],
        rtol=3.0e-15,
    )
    np.testing.assert_allclose(
        evaluation.acceleration_m_s2[2],
        0.5 * evaluation.acceleration_m_s2[0],
        rtol=3.0e-15,
    )
    np.testing.assert_allclose(
        evaluation.acceleration_m_s2[3],
        0.5 * evaluation.acceleration_m_s2[0],
        rtol=3.0e-15,
    )

    zero = waldmann_gallis_thermophoresis(
        mass_kg=mass[:1],
        drag_diameter_m=diameter[:1],
        velocity_m_s=np.zeros((1, 2)),
        gas_velocity_m_s=np.zeros((1, 2)),
        gas_temperature_K=temperature[:1],
        gas_translational_heat_flux_W_m2=np.zeros((1, 2)),
        gas_mean_free_path_m=np.asarray([1.0e-3]),
        gas_molecular_mass_kg=_ARGON_MASS_KG,
    )
    np.testing.assert_array_equal(zero.acceleration_m_s2, np.zeros((1, 2)))


def test_waldmann_global_bound_and_path_gates_fail_closed() -> None:
    mass = np.asarray([1.0e-15, 2.0e-15])
    diameter = np.asarray([2.0e-7, 4.0e-4])
    bounds = waldmann_gallis_global_bounds(
        mass_kg=mass,
        drag_diameter_m=diameter,
        gas_temperature_lower_K=250.0,
        gas_translational_heat_flux_abs_upper_W_m2=np.asarray([5.0, 7.0]),
        gas_mean_free_path_lower_m=1.0e-3,
        gas_molecular_mass_kg=_ARGON_MASS_KG,
    )
    assert bounds.static_applicable.tolist() == [True, False]

    rng = np.random.default_rng(20260930)
    for _ in range(64):
        temperature = rng.uniform(250.0, 600.0, size=2)
        heat_flux = rng.uniform([-5.0, -7.0], [5.0, 7.0], size=(2, 2))
        evaluation = waldmann_gallis_thermophoresis(
            mass_kg=mass,
            drag_diameter_m=diameter,
            velocity_m_s=np.zeros((2, 2)),
            gas_velocity_m_s=np.zeros((2, 2)),
            gas_temperature_K=temperature,
            gas_translational_heat_flux_W_m2=heat_flux,
            gas_mean_free_path_m=np.full(2, 1.0e-3),
            gas_molecular_mass_kg=_ARGON_MASS_KG,
        )
        assert bool(
            (np.abs(evaluation.acceleration_m_s2) <= bounds.acceleration_abs_upper_m_s2).all()
        )

    mean_speed = math.sqrt(8.0 * BOLTZMANN_J_K * 250.0 / (math.pi * _ARGON_MASS_KG))
    applicable, status = waldmann_gallis_continuous_applicability_batch(
        static_applicable=np.asarray([True, False, True]),
        velocity_abs_upper_m_s=np.asarray(
            [[0.05 * mean_speed, 0.0], [0.0, 0.0], [0.2 * mean_speed, 0.0]]
        ),
        gas_velocity_abs_upper_m_s=np.zeros(2),
        gas_temperature_lower_K=250.0,
        gas_molecular_mass_kg=_ARGON_MASS_KG,
    )
    assert applicable.tolist() == [True, False, False]
    assert status.tolist() == [0, 0, 0]

    with pytest.raises(PhysicsEvaluationError, match="must be finite"):
        waldmann_gallis_thermophoresis(
            mass_kg=np.asarray([1.0e-15]),
            drag_diameter_m=np.asarray([2.0e-7]),
            velocity_m_s=np.zeros((1, 2)),
            gas_velocity_m_s=np.zeros((1, 2)),
            gas_temperature_K=np.asarray([300.0]),
            gas_translational_heat_flux_W_m2=np.asarray([[math.nan, 0.0]]),
            gas_mean_free_path_m=np.asarray([1.0e-3]),
            gas_molecular_mass_kg=_ARGON_MASS_KG,
        )


@pytest.mark.parametrize(
    ("coordinate_system", "components", "basis"),
    [
        ("cartesian_xy", ("x", "y"), "cartesian_xy"),
        ("axisymmetric_rz", ("r", "z"), "axisymmetric_rz"),
    ],
)
def test_waldmann_catalog_and_compiled_runtime_share_one_formula(
    coordinate_system: str,
    components: tuple[str, str],
    basis: str,
) -> None:
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "thermophoresis": _model()},
        coordinate_system,  # type: ignore[arg-type]
    )
    assert isinstance(plan.thermophoresis, WaldmannGallisThermophoresisPlan)
    requirements = {item.name: item for item in plan.required_fields}
    assert set(requirements) == {"ug", "tg", "qtr", "mfp"}
    assert requirements["ug"].components == components
    assert requirements["qtr"].stored_basis == basis
    assert plan.resolved_models()["thermophoresis"] == {
        "model": "waldmann_gallis",
        "revision": _REVISION,
    }

    ranges = {
        "ug": _constant_vector_range(0.0, 0.0),
        "tg": _constant_range(300.0),
        "qtr": _constant_vector_range(3.0, -4.0),
        "mfp": _constant_range(1.0e-3),
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system=coordinate_system,  # type: ignore[arg-type]
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        electrostatic_radius_m=np.asarray([0.0]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([0.0]),
        primitive_ranges=ranges,
    )
    sampled = {
        "ug": np.asarray([[0.0, 0.0]]),
        "tg": np.asarray([[300.0]]),
        "qtr": np.asarray([[3.0, -4.0]]),
        "mfp": np.asarray([[1.0e-3]]),
    }
    actual = runtime.evaluate(
        np.asarray([0], dtype=np.int64),
        np.asarray([[0.0, 0.0]]),
        np.asarray([0.0]),
        sampled,
    )
    expected = waldmann_gallis_thermophoresis(
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        velocity_m_s=np.asarray([[0.0, 0.0]]),
        gas_velocity_m_s=sampled["ug"],
        gas_temperature_K=sampled["tg"][:, 0],
        gas_translational_heat_flux_W_m2=sampled["qtr"],
        gas_mean_free_path_m=sampled["mfp"][:, 0],
        gas_molecular_mass_kg=_ARGON_MASS_KG,
    )
    np.testing.assert_allclose(actual.acceleration_m_s2, expected.acceleration_m_s2, rtol=2e-15)
    np.testing.assert_array_equal(actual.additive_acceleration_m_s2, actual.acceleration_m_s2)
    np.testing.assert_array_equal(actual.linear_drag_rate_s_inv, np.zeros(1))
    assert runtime.constant_acceleration_m_s2 is None
    assert runtime.continuous_applicability(
        np.asarray([0], dtype=np.int64),
        np.asarray([[0.0, 0.0]]),
    ).tolist() == [True]
    assert runtime.bound_array_nbytes == 41


def test_waldmann_and_drag_require_one_compatible_neutral_background() -> None:
    drag = {
        "model": "epstein_linear",
        "revision": "epstein_linear_v1",
        "gas_velocity_field": "ug",
        "gas_density_field": "rho",
        "gas_temperature_field": "other_tg",
        "gas_mean_free_path_field": "mfp",
        "gas_molecular_mass_kg": _ARGON_MASS_KG,
        "delta": 1.0,
        "applicability": "error",
    }
    with pytest.raises(PhysicsConfigurationError, match="same neutral-gas background"):
        resolve_physics_plan(
            {
                "charge": {"model": "fixed"},
                "drag": drag,
                "thermophoresis": _model(),
            },
            "cartesian_xy",
        )

    compatible_drag = {**drag, "gas_temperature_field": "tg"}
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": compatible_drag,
            "thermophoresis": _model(),
        },
        "cartesian_xy",
    )
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        electrostatic_radius_m=np.asarray([0.0]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([0.0]),
        primitive_ranges={
            "ug": _constant_vector_range(0.5, -0.25),
            "rho": _constant_range(1.0e-4),
            "tg": _constant_range(300.0),
            "qtr": _constant_vector_range(3.0, -4.0),
            "mfp": _constant_range(1.0e-3),
        },
    )
    sampled = {
        "ug": np.asarray([[0.5, -0.25]]),
        "rho": np.asarray([[1.0e-4]]),
        "tg": np.asarray([[300.0]]),
        "qtr": np.asarray([[3.0, -4.0]]),
        "mfp": np.asarray([[1.0e-3]]),
    }
    velocity = np.asarray([[0.1, -0.05]])
    actual = runtime.evaluate(np.asarray([0]), velocity, np.asarray([0.0]), sampled)
    drag_reference = epstein_linear_relaxation(
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        velocity_m_s=velocity,
        gas_velocity_m_s=sampled["ug"],
        gas_density_kg_m3=sampled["rho"][:, 0],
        gas_temperature_K=sampled["tg"][:, 0],
        gas_mean_free_path_m=sampled["mfp"][:, 0],
        gas_molecular_mass_kg=_ARGON_MASS_KG,
        delta=1.0,
    )
    thermo_reference = waldmann_gallis_thermophoresis(
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        velocity_m_s=velocity,
        gas_velocity_m_s=sampled["ug"],
        gas_temperature_K=sampled["tg"][:, 0],
        gas_translational_heat_flux_W_m2=sampled["qtr"],
        gas_mean_free_path_m=sampled["mfp"][:, 0],
        gas_molecular_mass_kg=_ARGON_MASS_KG,
    )
    expected = drag_reference.rate_s_inv[:, None] * (sampled["ug"] - velocity)
    expected += thermo_reference.acceleration_m_s2
    np.testing.assert_allclose(actual.acceleration_m_s2, expected, rtol=2.0e-15)

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
            {
                "charge": {"model": "fixed"},
                "drag": stokes,
                "thermophoresis": _model(),
            },
            "cartesian_xy",
        )


def _model() -> dict[str, object]:
    return {
        "model": "waldmann_gallis",
        "revision": _REVISION,
        "gas_velocity_field": "ug",
        "gas_temperature_field": "tg",
        "gas_translational_heat_flux_field": "qtr",
        "gas_mean_free_path_field": "mfp",
        "gas_molecular_mass_kg": _ARGON_MASS_KG,
        "applicability": "error",
    }


def _constant_range(value: float) -> PrimitiveRange:
    array = np.asarray([value])
    return PrimitiveRange(array, array, array)


def _constant_vector_range(first: float, second: float) -> PrimitiveRange:
    array = np.asarray([first, second])
    return PrimitiveRange(array, array, array)
