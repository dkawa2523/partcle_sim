from __future__ import annotations

import math

import numpy as np
import pytest

from chamber_particles.physics.catalog import (
    EpsteinDragPlan,
    PhysicsConfigurationError,
    PhysicsPlan,
    WaldmannGallisThermophoresisPlan,
    resolve_physics_plan,
)
from chamber_particles.physics.compiled import evaluate_physics_tile_into
from chamber_particles.physics.forces import (
    BOLTZMANN_J_K,
    epstein_finite_speed_relaxation,
    epstein_linear_relaxation,
    waldmann_gallis_thermophoresis,
)
from chamber_particles.physics.runtime import (
    PhysicsRuntime,
    PrimitiveRange,
    prepare_physics_runtime,
)

_ARGON_MASS_KG = 6.6335209e-26
_DRAG_REVISION = "epstein_linear_effective_gas_sensitivity_v1"
_THERMOPHORESIS_REVISION = "waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1"


def test_number_weighted_pseudogas_is_not_a_species_sum_even_at_equal_temperature() -> None:
    number_density = 1.0e20
    first_mass = _ARGON_MASS_KG
    second_mass = 9.0 * first_mass
    common = _formula_inputs(np.zeros((1, 2)))
    common["gas_molecular_mass_kg"] = 0.5 * (first_mass + second_mass)
    aggregate = epstein_linear_relaxation(
        **common,
        gas_density_kg_m3=np.asarray([number_density * (first_mass + second_mass)]),
        delta=1.0,
    )
    radius = 1.0e-7
    species_sum_beta = (
        (4.0 * math.pi / 3.0)
        * radius**2
        * number_density
        * sum(
            mass * math.sqrt(8.0 * BOLTZMANN_J_K * 300.0 / (math.pi * mass))
            for mass in (first_mass, second_mass)
        )
    )
    aggregate_beta = aggregate.rate_s_inv[0] * 2.0e-15
    assert aggregate_beta / species_sum_beta == pytest.approx(math.sqrt(5.0) / 2.0, rel=3.0e-15)
    common["gas_molecular_mass_kg"] = first_mass
    single = epstein_linear_relaxation(
        **common,
        gas_density_kg_m3=np.asarray([number_density * first_mass]),
        delta=1.0,
    )
    exact_single_beta = (
        (4.0 * math.pi / 3.0)
        * radius**2
        * number_density
        * first_mass
        * math.sqrt(8.0 * BOLTZMANN_J_K * 300.0 / (math.pi * first_mass))
    )
    assert single.rate_s_inv[0] * 2.0e-15 == pytest.approx(exact_single_beta, rel=3.0e-15)


def test_existing_linear_drag_and_single_species_thermophoresis_stay_at_point_one() -> None:
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": _linear_drag_model("epstein_linear_v1"),
            "thermophoresis": _thermophoresis_model(
                "waldmann_gallis_free_molecular_single_species_heat_flux_v1"
            ),
        },
        "cartesian_xy",
    )

    assert isinstance(plan.drag, EpsteinDragPlan)
    assert isinstance(plan.thermophoresis, WaldmannGallisThermophoresisPlan)
    assert plan.drag.maximum_speed_ratio == 0.1
    assert plan.thermophoresis.maximum_speed_ratio == 0.1
    assert plan.resolved_models()["drag"]["revision"] == "epstein_linear_v1"
    assert (
        plan.resolved_models()["thermophoresis"]["revision"]
        == "waldmann_gallis_free_molecular_single_species_heat_flux_v1"
    )

    mean_speed = _mean_thermal_speed(300.0)
    velocity = np.asarray([[0.11 * mean_speed, 0.0]])
    common = _formula_inputs(velocity)
    drag = epstein_linear_relaxation(
        **common,
        gas_density_kg_m3=np.asarray([1.0e-4]),
        delta=1.0,
    )
    thermophoresis = waldmann_gallis_thermophoresis(
        **common,
        gas_translational_heat_flux_W_m2=np.asarray([[3.0, -4.0]]),
    )
    assert not drag.applicable[0]
    assert not thermophoresis.applicable[0]


@pytest.mark.parametrize("category", ["drag", "thermophoresis"])
@pytest.mark.parametrize("maximum_speed_ratio", [True, 0.0, -0.1, 1.0000001, math.inf, math.nan])
def test_effective_gas_revisions_require_a_bounded_speed_ratio(
    category: str,
    maximum_speed_ratio: object,
) -> None:
    model = _effective_drag_model() if category == "drag" else _effective_thermophoresis_model()
    model["maximum_speed_ratio"] = maximum_speed_ratio

    with pytest.raises(PhysicsConfigurationError, match="maximum_speed_ratio"):
        resolve_physics_plan({"charge": {"model": "fixed"}, category: model}, "cartesian_xy")


@pytest.mark.parametrize("category", ["drag", "thermophoresis"])
def test_effective_gas_revisions_require_the_explicit_speed_ratio(
    category: str,
) -> None:
    model = _effective_drag_model() if category == "drag" else _effective_thermophoresis_model()
    del model["maximum_speed_ratio"]

    with pytest.raises(PhysicsConfigurationError):
        resolve_physics_plan({"charge": {"model": "fixed"}, category: model}, "cartesian_xy")


def test_existing_revisions_reject_a_sensitivity_limit_and_brownian_accepts_effective_drag() -> (
    None
):
    old_drag = _linear_drag_model("epstein_linear_v1")
    old_drag["maximum_speed_ratio"] = 0.5
    with pytest.raises(PhysicsConfigurationError, match="keys do not match"):
        resolve_physics_plan({"charge": {"model": "fixed"}, "drag": old_drag}, "cartesian_xy")

    old_thermophoresis = _thermophoresis_model(
        "waldmann_gallis_free_molecular_single_species_heat_flux_v1"
    )
    old_thermophoresis["maximum_speed_ratio"] = 0.5
    with pytest.raises(PhysicsConfigurationError, match="keys do not match"):
        resolve_physics_plan(
            {"charge": {"model": "fixed"}, "thermophoresis": old_thermophoresis},
            "cartesian_xy",
        )

    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": _effective_drag_model(),
            "noise": {
                "model": "inertial_langevin_fdt",
                "revision": "inertial_langevin_fdt_epstein_linear_midpoint_2d_v2",
                "interval_tree_depth": 4,
            },
        },
        "cartesian_xy",
    )
    assert plan.resolved_models()["drag"]["revision"] == (
        "epstein_linear_effective_gas_sensitivity_v1"
    )


@pytest.mark.parametrize(
    ("effective_drag", "effective_thermophoresis"),
    [(False, True), (True, False)],
)
def test_linear_drag_and_waldmann_cannot_mix_native_and_effective_backgrounds(
    effective_drag: bool,
    effective_thermophoresis: bool,
) -> None:
    drag = _effective_drag_model() if effective_drag else _linear_drag_model("epstein_linear_v1")
    thermophoresis = (
        _effective_thermophoresis_model()
        if effective_thermophoresis
        else _thermophoresis_model("waldmann_gallis_free_molecular_single_species_heat_flux_v1")
    )

    with pytest.raises(PhysicsConfigurationError, match="cannot mix single-species"):
        resolve_physics_plan(
            {
                "charge": {"model": "fixed"},
                "drag": drag,
                "thermophoresis": thermophoresis,
            },
            "cartesian_xy",
        )


@pytest.mark.parametrize("category", ["drag", "thermophoresis"])
def test_neutral_density_authority_is_shared_with_gravity_without_lift(category: str) -> None:
    models: dict[str, dict[str, object]] = {
        "charge": {"model": "fixed"},
        "gravity_buoyancy": {
            "model": "standard",
            "revision": "gravity_buoyancy_standard_v1",
            "gas_density_field": "other_density",
            "gravity_m_s2": [0.0, -9.81],
        },
    }
    models[category] = (
        _linear_drag_model("epstein_linear_v1")
        if category == "drag"
        else _talbot_thermophoresis_model()
    )

    with pytest.raises(
        PhysicsConfigurationError,
        match=r"same neutral-gas background.*gas_density_field",
    ):
        resolve_physics_plan(models, "cartesian_xy")


def test_effective_gas_limits_drive_compiled_row_and_continuous_path_gates() -> None:
    mean_speed = _mean_thermal_speed(300.0)
    velocity = np.asarray([[0.25 * mean_speed, 0.0], [0.31 * mean_speed, 0.0]])
    sampled = _sampled_values(2)
    particle_index = np.zeros(2, dtype=np.int64)
    for category, model in (
        ("drag", _effective_drag_model(maximum_speed_ratio=0.3)),
        ("thermophoresis", _effective_thermophoresis_model(maximum_speed_ratio=0.3)),
    ):
        single_plan = resolve_physics_plan(
            {"charge": {"model": "fixed"}, category: model},
            "cartesian_xy",
        )
        single_runtime = _runtime(single_plan)
        single_evaluation = single_runtime.evaluate(
            particle_index,
            velocity,
            np.zeros(2),
            sampled,
        )
        np.testing.assert_array_equal(single_evaluation.applicable, [True, False])
        np.testing.assert_array_equal(
            single_runtime.continuous_applicability(particle_index, np.abs(velocity)),
            [True, False],
        )

    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": _effective_drag_model(maximum_speed_ratio=0.3),
            "thermophoresis": _effective_thermophoresis_model(maximum_speed_ratio=0.3),
        },
        "cartesian_xy",
    )
    runtime = _runtime(plan)

    actual = runtime.evaluate(particle_index, velocity, np.zeros(2), sampled)
    drag = epstein_linear_relaxation(
        **_formula_inputs(velocity),
        gas_density_kg_m3=sampled["rho"][:, 0],
        delta=1.0,
        maximum_speed_ratio=0.3,
    )
    thermophoresis = waldmann_gallis_thermophoresis(
        **_formula_inputs(velocity),
        gas_translational_heat_flux_W_m2=sampled["qtr"],
        maximum_speed_ratio=0.3,
    )
    expected = drag.rate_s_inv[:, None] * (sampled["ug"] - velocity)
    expected += thermophoresis.acceleration_m_s2

    np.testing.assert_allclose(actual.acceleration_m_s2, expected, rtol=3.0e-15)
    np.testing.assert_array_equal(drag.applicable, [True, False])
    np.testing.assert_array_equal(thermophoresis.applicable, [True, False])
    np.testing.assert_array_equal(actual.applicable, [True, False])
    np.testing.assert_array_equal(
        runtime.continuous_applicability(particle_index, np.abs(velocity)),
        [True, False],
    )
    assert plan.resolved_models()["drag"]["revision"] == _DRAG_REVISION
    assert plan.resolved_models()["thermophoresis"]["revision"] == _THERMOPHORESIS_REVISION
    assert evaluate_physics_tile_into.nopython_signatures


def test_finite_speed_drag_composes_with_effective_gas_heat_flux() -> None:
    finite_speed_drag = {
        "model": "epstein_finite_speed",
        "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
        "gas_velocity_field": "ug",
        "gas_density_field": "rho",
        "gas_temperature_field": "tg",
        "gas_mean_free_path_field": "mfp",
        "gas_molecular_mass_kg": _ARGON_MASS_KG,
        "diffuse_reflection_fraction": 0.6,
        "maximum_speed_ratio": 1.0,
        "applicability": "error",
    }
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": finite_speed_drag,
            "thermophoresis": _effective_thermophoresis_model(maximum_speed_ratio=0.5),
        },
        "cartesian_xy",
    )
    runtime = _runtime(plan)
    velocity = np.asarray([[0.2 * _mean_thermal_speed(300.0), 0.0]])
    sampled = _sampled_values(1)

    actual = runtime.evaluate(np.asarray([0]), velocity, np.asarray([0.0]), sampled)
    drag = epstein_finite_speed_relaxation(
        **_formula_inputs(velocity),
        gas_density_kg_m3=sampled["rho"][:, 0],
        diffuse_reflection_fraction=0.6,
        maximum_speed_ratio=1.0,
    )
    thermophoresis = waldmann_gallis_thermophoresis(
        **_formula_inputs(velocity),
        gas_translational_heat_flux_W_m2=sampled["qtr"],
        maximum_speed_ratio=0.5,
    )
    expected = drag.rate_s_inv[:, None] * (sampled["ug"] - velocity)
    expected += thermophoresis.acceleration_m_s2

    np.testing.assert_allclose(actual.acceleration_m_s2, expected, rtol=3.0e-15)
    assert actual.applicable.tolist() == [True]
    assert runtime.continuous_applicability(
        np.asarray([0]),
        np.abs(velocity),
    ).tolist() == [True]


def _linear_drag_model(revision: str) -> dict[str, object]:
    return {
        "model": "epstein_linear",
        "revision": revision,
        "gas_velocity_field": "ug",
        "gas_density_field": "rho",
        "gas_temperature_field": "tg",
        "gas_mean_free_path_field": "mfp",
        "gas_molecular_mass_kg": _ARGON_MASS_KG,
        "delta": 1.0,
        "applicability": "error",
    }


def _effective_drag_model(maximum_speed_ratio: float = 0.5) -> dict[str, object]:
    return {
        **_linear_drag_model(_DRAG_REVISION),
        "maximum_speed_ratio": maximum_speed_ratio,
    }


def _thermophoresis_model(revision: str) -> dict[str, object]:
    return {
        "model": "waldmann_gallis",
        "revision": revision,
        "gas_velocity_field": "ug",
        "gas_temperature_field": "tg",
        "gas_translational_heat_flux_field": "qtr",
        "gas_mean_free_path_field": "mfp",
        "gas_molecular_mass_kg": _ARGON_MASS_KG,
        "applicability": "error",
    }


def _effective_thermophoresis_model(
    maximum_speed_ratio: float = 0.5,
) -> dict[str, object]:
    return {
        **_thermophoresis_model(_THERMOPHORESIS_REVISION),
        "maximum_speed_ratio": maximum_speed_ratio,
    }


def _talbot_thermophoresis_model() -> dict[str, object]:
    return {
        "model": "talbot",
        "revision": "talbot_cross_regime_radius_knudsen_v1",
        "gas_temperature_field": "tg",
        "gas_temperature_gradient_field": "grad_tg",
        "gas_density_field": "rho",
        "gas_dynamic_viscosity_field": "mu",
        "gas_thermal_conductivity_field": "k_g",
        "gas_mean_free_path_field": "mfp",
        "particle_thermal_conductivity_W_m_K": 0.2,
        "thermal_slip_coefficient": 1.17,
        "momentum_exchange_coefficient": 1.146,
        "thermal_exchange_coefficient": 2.2,
        "applicability": "error",
    }


def _runtime(plan: PhysicsPlan) -> PhysicsRuntime:
    return prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([2.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        electrostatic_radius_m=np.asarray([1.0e-7]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([0.0]),
        primitive_ranges={
            "ug": _constant_vector_range(0.0, 0.0),
            "rho": _constant_range(1.0e-4),
            "tg": _constant_range(300.0),
            "qtr": _constant_vector_range(3.0, -4.0),
            "mfp": _constant_range(1.0e-3),
        },
    )


def _formula_inputs(velocity: np.ndarray) -> dict[str, object]:
    count = velocity.shape[0]
    return {
        "mass_kg": np.full(count, 2.0e-15),
        "drag_diameter_m": np.full(count, 2.0e-7),
        "velocity_m_s": velocity,
        "gas_velocity_m_s": np.zeros((count, 2)),
        "gas_temperature_K": np.full(count, 300.0),
        "gas_mean_free_path_m": np.full(count, 1.0e-3),
        "gas_molecular_mass_kg": _ARGON_MASS_KG,
    }


def _sampled_values(count: int) -> dict[str, np.ndarray]:
    return {
        "ug": np.zeros((count, 2)),
        "rho": np.full((count, 1), 1.0e-4),
        "tg": np.full((count, 1), 300.0),
        "qtr": np.tile(np.asarray([[3.0, -4.0]]), (count, 1)),
        "mfp": np.full((count, 1), 1.0e-3),
    }


def _mean_thermal_speed(temperature_K: float) -> float:
    return math.sqrt(8.0 * BOLTZMANN_J_K * temperature_K / (math.pi * _ARGON_MASS_KG))


def _constant_range(value: float) -> PrimitiveRange:
    array = np.asarray([value])
    return PrimitiveRange(array, array, array)


def _constant_vector_range(first: float, second: float) -> PrimitiveRange:
    array = np.asarray([first, second])
    return PrimitiveRange(array, array, array)
