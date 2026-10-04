from __future__ import annotations

import math
from itertools import product
from typing import Literal

import numpy as np
import pytest

from chamber_particles.numerical_status import (
    FIELD_NUMERICAL_FAILURE,
    NUMERICAL_STATUS_OK,
    PHYSICS_NUMERICAL_FAILURE,
)
from chamber_particles.physics.catalog import (
    FiniteSpeedEpsteinDragPlan,
    InertialLangevinNoisePlan,
    PhysicsConfigurationError,
    PlasmaContinuousChargePlan,
    StokesCunninghamDragPlan,
    resolve_physics_plan,
)
from chamber_particles.physics.charge import (
    ELECTRON_MASS_KG,
    VACUUM_PERMITTIVITY_F_M,
    debye_huckel_capacitance_F,
    oml_debye_length_m,
    oml_local_equilibrium_bracket,
    oml_stationary_global_bounds,
    oml_stationary_maxwellian_debye_huckel_v1,
)
from chamber_particles.physics.compiled import evaluate_physics_tile_into
from chamber_particles.physics.forces import (
    BOLTZMANN_J_K,
    CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE,
    CONTINUOUS_APPLICABILITY_OK,
    ELEMENTARY_CHARGE_C,
    PhysicsEvaluationError,
    add_electric_coulomb_acceleration,
    add_gravity_buoyancy_acceleration,
    electric_acceleration_abs_upper,
    epstein_continuous_applicability,
    epstein_continuous_applicability_batch,
    epstein_finite_speed_continuous_applicability_batch,
    epstein_finite_speed_factors,
    epstein_finite_speed_relaxation,
    epstein_linear_relaxation,
    gravity_buoyancy_acceleration_abs_upper,
    linear_drag_acceleration_abs_upper,
    linear_drag_acceleration_abs_upper_batch,
    stokes_cunningham_continuous_applicability,
    stokes_cunningham_continuous_applicability_batch,
    stokes_cunningham_linear_relaxation,
)
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime


@pytest.mark.parametrize(
    ("coordinate_system", "components", "stored_basis"),
    [
        ("cartesian_xy", ("x", "y"), "cartesian_xy"),
        ("axisymmetric_rz", ("r", "z"), "axisymmetric_rz"),
    ],
)
def test_physics_catalog_requests_vectors_in_the_case_coordinate_basis(
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"],
    components: tuple[str, str],
    stored_basis: str,
) -> None:
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "electric": {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
                "electric_field": "electric_field",
            },
        },
        coordinate_system,
    )

    requirement = plan.required_fields[0]
    assert requirement.name == "electric_field"
    assert requirement.components == components
    assert requirement.stored_basis == stored_basis
    assert plan.has_force
    assert not plan.evolves_continuous_state
    assert plan.requires_stage_evaluation


@pytest.mark.parametrize(
    ("coordinate_system", "components", "stored_basis"),
    [
        ("cartesian_xy", ("x", "y"), "cartesian_xy"),
        ("axisymmetric_rz", ("r", "z"), "axisymmetric_rz"),
    ],
)
def test_stokes_cunningham_catalog_declares_exact_primitives(
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"],
    components: tuple[str, str],
    stored_basis: str,
) -> None:
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": _stokes_cunningham_model(),
        },
        coordinate_system,
    )

    assert isinstance(plan.drag, StokesCunninghamDragPlan)
    requirements = {item.name: item for item in plan.required_fields}
    assert set(requirements) == {
        "gas_velocity",
        "gas_density",
        "gas_dynamic_viscosity",
        "gas_mean_free_path",
    }
    assert requirements["gas_velocity"].components == components
    assert requirements["gas_velocity"].stored_basis == stored_basis
    assert requirements["gas_dynamic_viscosity"].unit == "Pa*s"
    assert all(item.positive for name, item in requirements.items() if name != "gas_velocity")
    assert plan.resolved_models()["drag"] == {
        "model": "stokes_cunningham",
        "revision": "stokes_cunningham_allen_raabe_air_v1",
    }


@pytest.mark.parametrize(
    ("coordinate_system", "components", "stored_basis"),
    [
        ("cartesian_xy", ("x", "y"), "cartesian_xy"),
        ("axisymmetric_rz", ("r", "z"), "axisymmetric_rz"),
    ],
)
def test_continuous_charge_catalog_declares_exact_primitives_and_capabilities(
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"],
    components: tuple[str, str],
    stored_basis: str,
) -> None:
    plan = resolve_physics_plan(
        {"charge": _plasma_continuous_charge_model()},
        coordinate_system,
    )

    assert isinstance(plan.charge, PlasmaContinuousChargePlan)
    assert plan.charge.positive_ion_mass_kg == 6.6335209e-26
    assert not plan.has_force
    assert plan.evolves_continuous_state
    assert plan.requires_stage_evaluation
    requirements = {item.name: item for item in plan.required_fields}
    assert set(requirements) == {
        "electron_number_density",
        "positive_ion_number_density",
        "electron_temperature",
        "positive_ion_temperature",
        "positive_ion_velocity",
    }
    for name in ("electron_number_density", "positive_ion_number_density"):
        assert requirements[name].unit == "1/m^3"
        assert requirements[name].components == ("value",)
        assert requirements[name].stored_basis == "scalar"
        assert requirements[name].positive
    for name in ("electron_temperature", "positive_ion_temperature"):
        assert requirements[name].unit == "K"
        assert requirements[name].components == ("value",)
        assert requirements[name].stored_basis == "scalar"
        assert requirements[name].positive
    velocity = requirements["positive_ion_velocity"]
    assert velocity.unit == "m/s"
    assert velocity.components == components
    assert velocity.stored_basis == stored_basis
    assert not velocity.positive
    assert plan.resolved_models() == {
        "charge": {
            "model": "plasma_continuous",
            "revision": "oml_stationary_maxwellian_debye_huckel_v1",
        }
    }


def test_fixed_charge_catalog_preserves_existing_capabilities() -> None:
    plan = resolve_physics_plan({"charge": {"model": "fixed"}}, "cartesian_xy")

    assert plan.charge is None
    assert not plan.has_force
    assert not plan.evolves_continuous_state
    assert not plan.requires_stage_evaluation
    assert plan.resolved_models() == {"charge": {"model": "fixed", "revision": "fixed_charge_v1"}}


def test_inertial_langevin_catalog_resolves_one_epstein_coupled_plan() -> None:
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": _epstein_linear_model(),
            "noise": _inertial_langevin_noise_model(),
        },
        "cartesian_xy",
    )

    assert isinstance(plan.noise, InertialLangevinNoisePlan)
    assert plan.noise.interval_tree_depth == 4
    assert plan.resolved_models()["noise"] == {
        "model": "inertial_langevin_fdt",
        "revision": "inertial_langevin_fdt_epstein_linear_frozen_start_v1",
    }
    assert [field.name for field in plan.required_fields].count("gas_temperature") == 1


@pytest.mark.parametrize("continuous_charge", [False, True])
@pytest.mark.parametrize(
    "drag_revision",
    ["epstein_linear_v1", "epstein_linear_effective_gas_sensitivity_v1"],
)
def test_rz_inertial_langevin_catalog_retains_revision_and_composes_supported_models(
    continuous_charge: bool,
    drag_revision: str,
) -> None:
    drag = _epstein_linear_model()
    drag["revision"] = drag_revision
    if drag_revision == "epstein_linear_effective_gas_sensitivity_v1":
        drag["maximum_speed_ratio"] = 0.6
    models: dict[str, dict[str, object]] = {
        "charge": (_plasma_continuous_charge_model() if continuous_charge else {"model": "fixed"}),
        "drag": drag,
        "noise": _rz_inertial_langevin_noise_model(),
        "thermophoresis": {
            "model": "waldmann_gallis",
            "revision": "waldmann_gallis_free_molecular_single_species_heat_flux_v1",
            "gas_velocity_field": "gas_velocity",
            "gas_temperature_field": "gas_temperature",
            "gas_translational_heat_flux_field": "gas_heat_flux",
            "gas_mean_free_path_field": "gas_mean_free_path",
            "gas_molecular_mass_kg": 4.65e-26,
            "applicability": "error",
        },
        "ion_drag": {
            "model": "barnes_collisionless",
            "revision": (
                "barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1"
            ),
            "electron_number_density_field": "electron_number_density",
            "positive_ion_number_density_field": "positive_ion_number_density",
            "electron_temperature_field": "electron_temperature",
            "positive_ion_temperature_field": "positive_ion_temperature",
            "positive_ion_velocity_field": "positive_ion_velocity",
            "ion_neutral_mean_free_path_field": "ion_mean_free_path",
            "positive_ion_mass_kg": 6.6335209e-26,
            "maximum_ion_drift_ratio": 1.5,
            "applicability": "error",
        },
        "dielectrophoresis": {
            "model": "quasistatic_spherical",
            "revision": "quasistatic_spherical_gradient_e2_v1",
            "gradient_mean_e_squared_field": "gradient_mean_e_squared",
            "medium_relative_permittivity": 1.0,
            "real_clausius_mossotti_factor": 0.5,
            "maximum_point_dipole_radius_m": 2.0e-7,
        },
        "lift": {
            "model": "rarefied_vorticity_sensitivity",
            "revision": "rarefied_vorticity_sensitivity_rz_v1",
            "gas_velocity_field": "gas_velocity",
            "gas_density_field": "gas_density",
            "gas_mean_free_path_field": "gas_mean_free_path",
            "azimuthal_gas_vorticity_field": "azimuthal_gas_vorticity",
            "lift_coefficient": 0.75,
            "applicability": "error",
        },
        "electric": {
            "model": "coulomb",
            "revision": "electric_coulomb_v1",
            "electric_field": "electric_field",
        },
        "gravity_buoyancy": {
            "model": "standard",
            "revision": "gravity_buoyancy_standard_v1",
            "gas_density_field": "gas_density",
            "gravity_m_s2": [0.0, -9.81],
        },
    }

    plan = resolve_physics_plan(models, "axisymmetric_rz")

    assert isinstance(plan.noise, InertialLangevinNoisePlan)
    assert plan.noise.revision == (
        "inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1"
    )
    assert plan.noise.interval_tree_depth == 3
    assert plan.resolved_models()["noise"]["revision"] == plan.noise.revision
    assert plan.resolved_models()["drag"]["revision"] == drag_revision
    assert set(plan.resolved_models()) == {
        "charge",
        "drag",
        "noise",
        "thermophoresis",
        "ion_drag",
        "dielectrophoresis",
        "lift",
        "electric",
        "gravity_buoyancy",
    }
    vector_fields = {
        field.name: field
        for field in plan.required_fields
        if field.name in {"gas_velocity", "electric_field"}
    }
    assert all(field.components == ("r", "z") for field in vector_fields.values())


def test_rz_inertial_langevin_catalog_rejects_wrong_coordinate_and_nonlinear_drag() -> None:
    noise = _rz_inertial_langevin_noise_model()
    with pytest.raises(PhysicsConfigurationError, match="requires axisymmetric_rz"):
        resolve_physics_plan(
            {"charge": {"model": "fixed"}, "drag": _epstein_linear_model(), "noise": noise},
            "cartesian_xy",
        )
    with pytest.raises(PhysicsConfigurationError, match="linear Epstein"):
        resolve_physics_plan(
            {
                "charge": {"model": "fixed"},
                "drag": _finite_speed_epstein_model(),
                "noise": noise,
            },
            "axisymmetric_rz",
        )


def test_rz_inertial_langevin_accepts_aggregate_charge_and_relative_ion_drag() -> None:
    drag = _epstein_linear_model()
    drag["revision"] = "epstein_linear_effective_gas_sensitivity_v1"
    drag["maximum_speed_ratio"] = 0.6
    shared = {
        "positive_ion_number_density_field": "ion_density",
        "positive_ion_thermal_voltage_field": "ion_thermal_voltage",
        "positive_ion_velocity_field": "ion_velocity",
        "effective_positive_ion_mass_field": "effective_ion_mass",
        "screening_length_field": "screening_length",
        "maximum_relative_ion_speed_m_s": 2_000.0,
    }
    plan = resolve_physics_plan(
        {
            "charge": {
                "model": "plasma_continuous",
                "revision": "aggregate_relative_drift_regularized_two_current_v1",
                "electron_number_density_field": "electron_density",
                "electron_thermal_voltage_field": "electron_thermal_voltage",
                **shared,
                "applicability": "error",
            },
            "drag": drag,
            "noise": _rz_inertial_langevin_noise_model(),
            "ion_drag": {
                "model": "screened_collection_orbital",
                "revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
                **shared,
                "ion_neutral_mean_free_path_field": "ion_mean_free_path",
                "applicability": "error",
            },
        },
        "axisymmetric_rz",
    )

    assert plan.resolved_models()["charge"]["revision"] == (
        "aggregate_relative_drift_regularized_two_current_v1"
    )
    assert plan.resolved_models()["ion_drag"]["model"] == "screened_collection_orbital"
    assert plan.noise is not None
    assert plan.noise.revision == (
        "inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1"
    )


def test_inertial_langevin_catalog_rejects_incompatible_physics() -> None:
    noise = _inertial_langevin_noise_model()
    cases: tuple[
        tuple[dict[str, dict[str, object]], Literal["cartesian_xy", "axisymmetric_rz"], str], ...
    ] = (
        (
            {"charge": {"model": "fixed"}, "drag": _epstein_linear_model(), "noise": noise},
            "axisymmetric_rz",
            "cartesian_xy",
        ),
        (
            {
                "charge": _plasma_continuous_charge_model(),
                "drag": _epstein_linear_model(),
                "noise": noise,
            },
            "cartesian_xy",
            "fixed charge",
        ),
        (
            {
                "charge": {"model": "fixed"},
                "drag": _finite_speed_epstein_model(),
                "noise": noise,
            },
            "cartesian_xy",
            "epstein_linear",
        ),
        (
            {
                "charge": {"model": "fixed"},
                "drag": _epstein_linear_model(),
                "noise": noise,
                "electric": {
                    "model": "coulomb",
                    "revision": "electric_coulomb_v1",
                    "electric_field": "electric_field",
                },
            },
            "cartesian_xy",
            "additional force",
        ),
    )
    for models, coordinate_system, message in cases:
        with pytest.raises(PhysicsConfigurationError, match=message):
            resolve_physics_plan(models, coordinate_system)


@pytest.mark.parametrize("depth", [-1, 11, True, 1.0])
def test_inertial_langevin_catalog_requires_bounded_integer_depth(depth: object) -> None:
    noise = _inertial_langevin_noise_model()
    noise["interval_tree_depth"] = depth

    with pytest.raises(PhysicsConfigurationError, match=r"integer in \[0, 10\]"):
        resolve_physics_plan(
            {
                "charge": {"model": "fixed"},
                "drag": _epstein_linear_model(),
                "noise": noise,
            },
            "cartesian_xy",
        )


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("model", "model must be inertial_langevin_fdt"),
        ("revision", "revision must be"),
        ("extra", "keys do not match"),
    ],
)
def test_inertial_langevin_catalog_requires_exact_revision_keys(
    damage: str,
    message: str,
) -> None:
    noise = _inertial_langevin_noise_model()
    if damage == "model":
        noise["model"] = "langevin"
    elif damage == "revision":
        noise["revision"] = "unversioned"
    else:
        noise["gas_temperature_field"] = "duplicated_authority"

    with pytest.raises(PhysicsConfigurationError, match=message):
        resolve_physics_plan(
            {
                "charge": {"model": "fixed"},
                "drag": _epstein_linear_model(),
                "noise": noise,
            },
            "cartesian_xy",
        )


def test_inertial_langevin_runtime_uses_fdt_temperature_and_particle_mass() -> None:
    drag = _epstein_linear_model()
    drag["gas_temperature_field"] = "neutral_temperature"
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": drag,
            "noise": _inertial_langevin_noise_model(),
        },
        "cartesian_xy",
    )
    mass = np.asarray([2.0e-15, 4.0e-15], dtype="<f8")
    diameter = np.full(2, 2.0e-6, dtype="<f8")
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=mass,
        drag_diameter_m=diameter,
        electrostatic_radius_m=0.5 * diameter,
        displaced_volume_m3=np.zeros(2, dtype="<f8"),
        charge_number=np.zeros(2, dtype="<f8"),
        primitive_ranges={
            "gas_velocity": PrimitiveRange(np.zeros(2), np.zeros(2), np.zeros(2)),
            "gas_density": _constant_range(0.01),
            "neutral_temperature": PrimitiveRange(np.asarray([300.0]), np.asarray([400.0]), None),
            "gas_mean_free_path": _constant_range(2.0e-5),
        },
    )
    particle_index = np.asarray([1, 0], dtype="<i8")
    temperature = np.asarray([[400.0], [300.0]], dtype="<f8")

    variance = runtime.brownian_thermal_velocity_variance(
        particle_index,
        {"neutral_temperature": temperature},
    )

    np.testing.assert_array_equal(
        variance,
        BOLTZMANN_J_K * temperature[:, 0] / mass[particle_index],
    )
    with pytest.raises(PhysicsEvaluationError, match="finite and positive"):
        runtime.brownian_thermal_velocity_variance(
            particle_index,
            {"neutral_temperature": np.asarray([[400.0], [0.0]], dtype="<f8")},
        )
    batch_variance, batch_status = runtime.brownian_thermal_velocity_variance_batch(
        particle_index,
        {"neutral_temperature": np.asarray([[400.0], [0.0]], dtype="<f8")},
    )
    np.testing.assert_array_equal(
        batch_variance,
        [BOLTZMANN_J_K * 400.0 / mass[1], 0.0],
    )
    np.testing.assert_array_equal(
        batch_status,
        [NUMERICAL_STATUS_OK, PHYSICS_NUMERICAL_FAILURE],
    )


def test_continuous_charge_composes_state_evolution_with_force_capability() -> None:
    plan = resolve_physics_plan(
        {
            "charge": _plasma_continuous_charge_model(),
            "electric": {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
                "electric_field": "electric_field",
            },
        },
        "cartesian_xy",
    )

    assert plan.has_force
    assert plan.evolves_continuous_state
    assert plan.requires_stage_evaluation


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("model", "oml_continuous", "model must be fixed or plasma_continuous"),
        ("revision", "oml_unversioned", "revision"),
        ("applicability", "count", "applicability=error"),
        ("electron_number_density_field", "", "nonempty string"),
        ("positive_ion_mass_kg", 0.0, "positive finite number"),
        ("positive_ion_mass_kg", True, "positive finite number"),
        ("positive_ion_mass_kg", math.inf, "positive finite number"),
        ("positive_ion_mass_kg", math.nan, "positive finite number"),
    ],
)
def test_continuous_charge_catalog_rejects_invalid_parameters(
    key: str,
    value: object,
    message: str,
) -> None:
    model = _plasma_continuous_charge_model()
    model[key] = value

    with pytest.raises(PhysicsConfigurationError, match=message):
        resolve_physics_plan({"charge": model}, "cartesian_xy")


@pytest.mark.parametrize("damage", ["missing", "extra"])
def test_continuous_charge_catalog_requires_exact_keys(damage: str) -> None:
    model = _plasma_continuous_charge_model()
    if damage == "missing":
        del model["positive_ion_temperature_field"]
    else:
        model["drift_correction"] = "hidden"

    with pytest.raises(PhysicsConfigurationError, match="keys do not match"):
        resolve_physics_plan({"charge": model}, "cartesian_xy")


def test_epstein_rate_and_declared_applicability_limits() -> None:
    molecular_mass_kg = 4.65e-26
    temperature_K = 300.0
    mean_thermal_speed_m_s = math.sqrt(
        8.0 * BOLTZMANN_J_K * temperature_K / (math.pi * molecular_mass_kg)
    )
    diameter_m = 0.02
    radius_m = 0.5 * diameter_m
    density_kg_m3 = 0.4
    mass_kg = 2.0
    delta = 1.2
    relative_speed = np.asarray([0.05, 0.05, 0.11]) * mean_thermal_speed_m_s
    gas_velocity = np.column_stack((relative_speed, np.zeros(3, dtype="<f8")))

    result = epstein_linear_relaxation(
        mass_kg=np.full(3, mass_kg, dtype="<f8"),
        drag_diameter_m=np.full(3, diameter_m, dtype="<f8"),
        velocity_m_s=np.zeros((3, 2), dtype="<f8"),
        gas_velocity_m_s=gas_velocity,
        gas_density_kg_m3=np.full(3, density_kg_m3, dtype="<f8"),
        gas_temperature_K=np.full(3, temperature_K, dtype="<f8"),
        gas_mean_free_path_m=np.asarray([0.1, 0.099, 0.1], dtype="<f8"),
        gas_molecular_mass_kg=molecular_mass_kg,
        delta=delta,
    )

    expected_rate_s_inv = (
        (4.0 * math.pi / 3.0)
        * radius_m**2
        * density_kg_m3
        * mean_thermal_speed_m_s
        * delta
        / mass_kg
    )
    np.testing.assert_allclose(result.rate_s_inv, expected_rate_s_inv, rtol=2.0e-15)
    np.testing.assert_array_equal(result.target_velocity_m_s, gas_velocity)
    np.testing.assert_array_equal(result.applicable, [True, False, False])


def test_finite_speed_epstein_catalog_declares_one_explicit_surface_model() -> None:
    model = _finite_speed_epstein_model()
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "drag": model},
        "cartesian_xy",
    )

    assert isinstance(plan.drag, FiniteSpeedEpsteinDragPlan)
    assert plan.drag.diffuse_reflection_fraction == 0.6
    assert plan.drag.maximum_speed_ratio == 3.0
    assert plan.resolved_models()["drag"] == {
        "model": "epstein_finite_speed",
        "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
    }
    assert {item.name for item in plan.required_fields} == {
        "gas_velocity",
        "gas_density",
        "gas_temperature",
        "gas_mean_free_path",
    }

    for invalid in (-0.01, 1.01, math.nan, math.inf, True):
        damaged = model.copy()
        damaged["diffuse_reflection_fraction"] = invalid
        with pytest.raises(PhysicsConfigurationError, match=r"\[0, 1\]"):
            resolve_physics_plan(
                {"charge": {"model": "fixed"}, "drag": damaged},
                "cartesian_xy",
            )


@pytest.mark.parametrize("speed_ratio", [0.2, 0.5, 1.0, 2.0])
def test_finite_speed_epstein_specular_factor_matches_velocity_integral(
    speed_ratio: float,
) -> None:
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    shifted_x = nodes[:, None, None] + speed_ratio
    transverse_y = nodes[None, :, None]
    transverse_z = nodes[None, None, :]
    molecular_speed = np.sqrt(
        shifted_x * shifted_x + transverse_y * transverse_y + transverse_z * transverse_z
    )
    expectation = (
        np.sum(
            weights[:, None, None]
            * weights[None, :, None]
            * weights[None, None, :]
            * molecular_speed
            * shifted_x
        )
        / math.pi**1.5
    )
    independent_factor = 3.0 * math.sqrt(math.pi) * expectation / (8.0 * speed_ratio)

    factor, _ = epstein_finite_speed_factors(np.asarray([speed_ratio], dtype="<f8"))

    assert factor[0] == pytest.approx(independent_factor, rel=2.0e-5)


def test_finite_speed_epstein_limits_and_radial_jacobian_factor() -> None:
    sigma = 0.7
    ratio = np.asarray([0.0, 1.0e-8, 0.1, 1.0, 1000.0], dtype="<f8")
    factor, radial = epstein_finite_speed_factors(ratio)

    assert factor[0] == 1.0
    assert radial[0] == 1.0
    assert np.all(np.diff(factor) >= 0.0)
    assert np.all(np.diff(radial) >= 0.0)
    assert factor[2] > factor[1]
    assert radial[2] > radial[1]
    drag_coefficient = (
        16.0 * (factor[-1] + sigma * math.pi / 8.0) / (3.0 * math.sqrt(math.pi) * ratio[-1])
    )
    assert drag_coefficient == pytest.approx(2.0, rel=5.0e-4)

    center = 1.2
    step = 1.0e-6
    neighboring, _ = epstein_finite_speed_factors(
        np.asarray([center - step, center + step], dtype="<f8")
    )
    finite_difference = ((center + step) * neighboring[1] - (center - step) * neighboring[0]) / (
        2.0 * step
    )
    _, center_radial = epstein_finite_speed_factors(np.asarray([center], dtype="<f8"))
    assert center_radial[0] == pytest.approx(finite_difference, rel=5.0e-10)


def test_finite_speed_epstein_low_speed_matches_linear_surface_limit() -> None:
    molecular_mass_kg = 4.65e-26
    temperature_K = 300.0
    sigma = 0.6
    most_probable_speed = math.sqrt(2.0 * BOLTZMANN_J_K * temperature_K / molecular_mass_kg)
    gas_velocity = np.asarray([[most_probable_speed * 1.0e-8, 0.0]], dtype="<f8")
    common = {
        "mass_kg": np.asarray([2.0e-15], dtype="<f8"),
        "drag_diameter_m": np.asarray([2.0e-6], dtype="<f8"),
        "velocity_m_s": np.zeros((1, 2), dtype="<f8"),
        "gas_velocity_m_s": gas_velocity,
        "gas_density_kg_m3": np.asarray([0.01], dtype="<f8"),
        "gas_temperature_K": np.asarray([temperature_K], dtype="<f8"),
        "gas_mean_free_path_m": np.asarray([2.0e-5], dtype="<f8"),
        "gas_molecular_mass_kg": molecular_mass_kg,
    }
    finite = epstein_finite_speed_relaxation(
        **common,
        diffuse_reflection_fraction=sigma,
        maximum_speed_ratio=1.0,
    )
    linear = epstein_linear_relaxation(
        **common,
        delta=1.0 + sigma * math.pi / 8.0,
    )

    np.testing.assert_allclose(finite.rate_s_inv, linear.rate_s_inv, rtol=3.0e-16)
    np.testing.assert_array_equal(finite.applicable, [True])


def test_finite_speed_epstein_continuous_gate_uses_declared_molecular_ratio() -> None:
    molecular_mass_kg = 4.65e-26
    temperature_K = 300.0
    most_probable_speed = math.sqrt(2.0 * BOLTZMANN_J_K * temperature_K / molecular_mass_kg)
    applicable, status = epstein_finite_speed_continuous_applicability_batch(
        drag_diameter_m=np.full(2, 2.0e-6, dtype="<f8"),
        velocity_abs_upper_m_s=np.asarray(
            [[0.5 * most_probable_speed, 0.0], [1.01 * most_probable_speed, 0.0]],
            dtype="<f8",
        ),
        gas_velocity_abs_upper_m_s=np.zeros(2, dtype="<f8"),
        gas_temperature_lower_K=temperature_K,
        gas_mean_free_path_lower_m=1.0e-5,
        gas_molecular_mass_kg=molecular_mass_kg,
        maximum_speed_ratio=1.0,
    )

    np.testing.assert_array_equal(applicable, [True, False])
    np.testing.assert_array_equal(status, [CONTINUOUS_APPLICABILITY_OK] * 2)


def test_finite_speed_epstein_prepares_separate_rate_and_velocity_jacobian_bounds() -> None:
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "drag": _finite_speed_epstein_model()},
        "cartesian_xy",
    )
    mass = np.asarray([2.0e-15], dtype="<f8")
    diameter = np.asarray([2.0e-6], dtype="<f8")
    density = 0.01
    temperature = 300.0
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=mass,
        drag_diameter_m=diameter,
        electrostatic_radius_m=0.5 * diameter,
        displaced_volume_m3=np.zeros(1, dtype="<f8"),
        charge_number=np.zeros(1, dtype="<f8"),
        primitive_ranges={
            "gas_velocity": PrimitiveRange(np.zeros(2), np.zeros(2), np.zeros(2)),
            "gas_density": _constant_range(density),
            "gas_temperature": _constant_range(temperature),
            "gas_mean_free_path": _constant_range(2.0e-5),
        },
    )
    specular_rate, specular_jacobian = epstein_finite_speed_factors(np.asarray([3.0], dtype="<f8"))
    mean_thermal_speed = math.sqrt(8.0 * BOLTZMANN_J_K * temperature / (math.pi * 4.65e-26))
    base_rate = (
        (4.0 * math.pi / 3.0) * (0.5 * diameter[0]) ** 2 * density * mean_thermal_speed / mass[0]
    )
    diffuse_factor = 0.6 * math.pi / 8.0
    rate_upper = runtime.linear_relaxation_abs_bounds(
        np.asarray([0], dtype="<i8")
    ).rate_upper_s_inv[0]
    expected_rate = base_rate * (specular_rate[0] + diffuse_factor)
    expected_jacobian = base_rate * (specular_jacobian[0] + diffuse_factor)

    assert rate_upper == pytest.approx(expected_rate, rel=3.0e-15)
    assert runtime.maximum_dt_over_tau(1.0) == pytest.approx(expected_jacobian, rel=3.0e-15)
    assert runtime.maximum_dt_over_tau(1.0) > rate_upper


def test_stokes_cunningham_rate_matches_independent_allen_raabe_oracle() -> None:
    diameter_m = 2.0e-6
    mean_free_path_m = 1.0e-6
    knudsen_radius = 2.0 * mean_free_path_m / diameter_m
    viscosity_Pa_s = 1.8e-5
    mass_kg = 2.0e-15
    slip_correction = 1.0 + knudsen_radius * (1.142 + 0.558 * math.exp(-0.999 / knudsen_radius))
    expected_rate = 3.0 * math.pi * viscosity_Pa_s * diameter_m / (slip_correction * mass_kg)
    gas_velocity = np.asarray([[0.05, 0.0], [0.05, 0.0], [1000.0, 0.0]], dtype="<f8")

    result = stokes_cunningham_linear_relaxation(
        mass_kg=np.full(3, mass_kg, dtype="<f8"),
        drag_diameter_m=np.full(3, diameter_m, dtype="<f8"),
        velocity_m_s=np.zeros((3, 2), dtype="<f8"),
        gas_velocity_m_s=gas_velocity,
        gas_density_kg_m3=np.full(3, 0.01, dtype="<f8"),
        gas_dynamic_viscosity_Pa_s=np.full(3, viscosity_Pa_s, dtype="<f8"),
        gas_mean_free_path_m=np.asarray(
            [mean_free_path_m, 0.01 * diameter_m, mean_free_path_m],
            dtype="<f8",
        ),
    )

    np.testing.assert_allclose(result.rate_s_inv[[0, 2]], expected_rate, rtol=2.0e-15)
    np.testing.assert_array_equal(result.target_velocity_m_s, gas_velocity)
    np.testing.assert_array_equal(result.applicable, [True, False, False])


def test_coulomb_acceleration_uses_charge_sign_and_mass_authority() -> None:
    masses_kg = np.asarray([1.602176634e-18, 3.204353268e-18], dtype="<f8")
    acceleration = np.zeros((2, 2), dtype="<f8")

    add_electric_coulomb_acceleration(
        acceleration,
        charge_number=np.asarray([-5.0, -5.0], dtype="<f8"),
        mass_kg=masses_kg,
        electric_field_V_m=np.asarray([[4.0, -2.0], [4.0, -2.0]], dtype="<f8"),
    )

    expected_force_N = -5.0 * ELEMENTARY_CHARGE_C * np.asarray([4.0, -2.0])
    np.testing.assert_allclose(
        acceleration,
        expected_force_N[None, :] / masses_kg[:, None],
        rtol=0.0,
        atol=2.0e-16,
    )


def test_gravity_buoyancy_uses_displaced_volume_independently_of_mass() -> None:
    acceleration = np.zeros((3, 2), dtype="<f8")
    masses_kg = np.asarray([4.0e-15, 4.0e-15, 8.0e-15], dtype="<f8")
    volumes_m3 = np.asarray([0.0, 1.0e-15, 1.0e-15], dtype="<f8")

    add_gravity_buoyancy_acceleration(
        acceleration,
        mass_kg=masses_kg,
        displaced_volume_m3=volumes_m3,
        gas_density_kg_m3=np.full(3, 1.2, dtype="<f8"),
        gravity_m_s2=(0.0, -10.0),
    )

    np.testing.assert_allclose(
        acceleration,
        [[0.0, -10.0], [0.0, -7.0], [0.0, -8.5]],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_cartesian_standard_gravity_accepts_a_nonzero_first_component() -> None:
    model = {
        "charge": {"model": "fixed"},
        "gravity_buoyancy": {
            "model": "standard",
            "revision": "gravity_buoyancy_standard_v1",
            "gas_density_field": "gas_density",
            "gravity_m_s2": [2.0, -10.0],
        },
    }

    plan = resolve_physics_plan(model, "cartesian_xy")
    assert plan.gravity_buoyancy is not None
    assert plan.gravity_buoyancy.gravity_m_s2 == (2.0, -10.0)
    with pytest.raises(
        PhysicsConfigurationError,
        match=r"axisymmetric_rz.*requires gravity_m_s2\[0\] = 0",
    ):
        resolve_physics_plan(model, "axisymmetric_rz")


def test_stationary_oml_branches_and_phi_zero_are_continuous() -> None:
    ion_mass_kg = 6.6335209e-26
    count = 5
    charge_number = np.asarray([-5.0, -1.0e-9, 0.0, 1.0e-9, 5.0], dtype="<f8")
    radius_m = np.full(count, 1.0e-7, dtype="<f8")
    electron_density = np.full(count, 1.0e15, dtype="<f8")
    ion_density = np.full(count, 1.0e15, dtype="<f8")
    electron_temperature = np.full(count, 2.0e4, dtype="<f8")
    ion_temperature = np.full(count, 500.0, dtype="<f8")

    result = oml_stationary_maxwellian_debye_huckel_v1(
        charge_number=charge_number,
        electrostatic_radius_m=radius_m,
        electron_number_density_m3=electron_density,
        positive_ion_number_density_m3=ion_density,
        electron_temperature_K=electron_temperature,
        positive_ion_temperature_K=ion_temperature,
        particle_velocity_m_s=np.zeros((count, 2), dtype="<f8"),
        positive_ion_velocity_m_s=np.zeros((count, 2), dtype="<f8"),
        positive_ion_mass_kg=ion_mass_kg,
    )

    expected_debye = np.sqrt(
        VACUUM_PERMITTIVITY_F_M
        * BOLTZMANN_J_K
        / (
            ELEMENTARY_CHARGE_C**2
            * (electron_density / electron_temperature + ion_density / ion_temperature)
        )
    )
    expected_capacitance = (
        4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * radius_m * (1.0 + radius_m / expected_debye)
    )
    helper_capacitance = debye_huckel_capacitance_F(
        electrostatic_radius_m=radius_m,
        debye_length_m=expected_debye,
    )
    np.testing.assert_allclose(result.debye_length_m, expected_debye, rtol=3.0e-15)
    np.testing.assert_allclose(result.capacitance_F, expected_capacitance, rtol=3.0e-15)
    np.testing.assert_allclose(helper_capacitance, expected_capacitance, rtol=3.0e-15)

    electron_voltage = BOLTZMANN_J_K * electron_temperature / ELEMENTARY_CHARGE_C
    ion_voltage = BOLTZMANN_J_K * ion_temperature / ELEMENTARY_CHARGE_C
    electron_amplitude = (
        math.pi
        * radius_m**2
        * electron_density
        * np.sqrt(8.0 * BOLTZMANN_J_K * electron_temperature / (math.pi * ELECTRON_MASS_KG))
    )
    ion_amplitude = (
        math.pi
        * radius_m**2
        * ion_density
        * np.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature / (math.pi * ion_mass_kg))
    )
    potential = charge_number * ELEMENTARY_CHARGE_C / expected_capacitance
    expected_rate = np.empty(count, dtype="<f8")
    negative = potential <= 0.0
    positive = ~negative
    expected_rate[negative] = ion_amplitude[negative] * (
        1.0 - potential[negative] / ion_voltage[negative]
    ) - electron_amplitude[negative] * np.exp(potential[negative] / electron_voltage[negative])
    expected_rate[positive] = ion_amplitude[positive] * np.exp(
        -potential[positive] / ion_voltage[positive]
    ) - electron_amplitude[positive] * (1.0 + potential[positive] / electron_voltage[positive])
    np.testing.assert_allclose(result.charge_rate_number_s, expected_rate, rtol=3.0e-15)
    assert result.charge_rate_number_s[2] == pytest.approx(
        ion_amplitude[2] - electron_amplitude[2],
        rel=2.0e-15,
    )
    np.testing.assert_allclose(
        result.charge_rate_number_s[[1, 3]],
        result.charge_rate_number_s[2],
        rtol=2.0e-9,
    )


def test_stationary_oml_rate_is_strictly_decreasing_in_charge() -> None:
    count = 81
    result = oml_stationary_maxwellian_debye_huckel_v1(
        charge_number=np.linspace(-40.0, 40.0, count, dtype=np.float64),
        electrostatic_radius_m=np.full(count, 1.0e-7, dtype="<f8"),
        electron_number_density_m3=np.full(count, 1.0e15, dtype="<f8"),
        positive_ion_number_density_m3=np.full(count, 1.0e15, dtype="<f8"),
        electron_temperature_K=np.full(count, 2.0e4, dtype="<f8"),
        positive_ion_temperature_K=np.full(count, 500.0, dtype="<f8"),
        particle_velocity_m_s=np.zeros((count, 2), dtype="<f8"),
        positive_ion_velocity_m_s=np.zeros((count, 2), dtype="<f8"),
        positive_ion_mass_kg=6.6335209e-26,
    )

    assert bool((np.diff(result.charge_rate_number_s) < 0.0).all())
    assert bool((result.charge_rate_derivative_s_inv < 0.0).all())


def test_stationary_oml_compiled_runtime_matches_pure_rate_and_derivative() -> None:
    count = 4
    ion_mass_kg = 6.6335209e-26
    radius_m = np.asarray([5.0e-8, 7.0e-8, 1.0e-7, 1.2e-7], dtype="<f8")
    charge_number = np.asarray([-40.0, -5.0, 0.0, 5.0], dtype="<f8")
    electron_density = np.asarray([0.8e15, 1.0e15, 1.2e15, 1.5e15], dtype="<f8")
    ion_density = np.asarray([1.1e15, 0.9e15, 1.3e15, 1.0e15], dtype="<f8")
    electron_temperature = np.asarray([1.8e4, 2.0e4, 2.2e4, 2.5e4], dtype="<f8")
    ion_temperature = np.asarray([400.0, 500.0, 600.0, 700.0], dtype="<f8")
    ion_velocity = np.asarray(
        [[0.0, 0.0], [20.0, -10.0], [-15.0, 5.0], [10.0, 12.0]],
        dtype="<f8",
    )
    particle_velocity = np.asarray(
        [[5.0, -2.0], [10.0, -4.0], [-8.0, 3.0], [4.0, 6.0]],
        dtype="<f8",
    )
    plan = resolve_physics_plan(
        {"charge": _plasma_continuous_charge_model()},
        "cartesian_xy",
    )
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.full(count, 2.0e-15),
        drag_diameter_m=2.0 * radius_m,
        electrostatic_radius_m=radius_m,
        displaced_volume_m3=np.zeros(count),
        charge_number=charge_number,
        primitive_ranges={
            "electron_number_density": PrimitiveRange(
                np.asarray([float(np.min(electron_density))]),
                np.asarray([float(np.max(electron_density))]),
                None,
            ),
            "positive_ion_number_density": PrimitiveRange(
                np.asarray([float(np.min(ion_density))]),
                np.asarray([float(np.max(ion_density))]),
                None,
            ),
            "electron_temperature": PrimitiveRange(
                np.asarray([float(np.min(electron_temperature))]),
                np.asarray([float(np.max(electron_temperature))]),
                None,
            ),
            "positive_ion_temperature": PrimitiveRange(
                np.asarray([float(np.min(ion_temperature))]),
                np.asarray([float(np.max(ion_temperature))]),
                None,
            ),
            "positive_ion_velocity": PrimitiveRange(
                np.min(ion_velocity, axis=0),
                np.max(ion_velocity, axis=0),
                None,
            ),
        },
    )
    sampled = {
        "electron_number_density": electron_density[:, None],
        "positive_ion_number_density": ion_density[:, None],
        "electron_temperature": electron_temperature[:, None],
        "positive_ion_temperature": ion_temperature[:, None],
        "positive_ion_velocity": ion_velocity,
    }

    actual, status = runtime.evaluate_batch(
        np.arange(count, dtype="<i8"),
        particle_velocity,
        charge_number,
        sampled,
    )
    expected = oml_stationary_maxwellian_debye_huckel_v1(
        charge_number=charge_number,
        electrostatic_radius_m=radius_m,
        electron_number_density_m3=electron_density,
        positive_ion_number_density_m3=ion_density,
        electron_temperature_K=electron_temperature,
        positive_ion_temperature_K=ion_temperature,
        particle_velocity_m_s=particle_velocity,
        positive_ion_velocity_m_s=ion_velocity,
        positive_ion_mass_kg=ion_mass_kg,
    )

    np.testing.assert_array_equal(status, NUMERICAL_STATUS_OK)
    np.testing.assert_allclose(
        actual.charge_rate_number_s,
        expected.charge_rate_number_s,
        rtol=5.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.charge_rate_derivative_s_inv,
        expected.charge_rate_derivative_s_inv,
        rtol=5.0e-15,
        atol=0.0,
    )
    assert bool((actual.charge_rate_derivative_s_inv < 0.0).all())


def test_stationary_oml_equilibrium_bracket_has_inward_endpoint_rates() -> None:
    ion_mass_kg = 6.6335209e-26
    radius = np.asarray([1.0e-7, 1.0e-7], dtype="<f8")
    electron_density = np.asarray([1.0e15, 1.0e12], dtype="<f8")
    ion_density = np.asarray([1.0e15, 1.0e18], dtype="<f8")
    electron_temperature = np.asarray([2.0e4, 2.0e4], dtype="<f8")
    ion_temperature = np.asarray([500.0, 500.0], dtype="<f8")
    bracket = oml_local_equilibrium_bracket(
        electrostatic_radius_m=radius,
        electron_number_density_m3=electron_density,
        positive_ion_number_density_m3=ion_density,
        electron_temperature_K=electron_temperature,
        positive_ion_temperature_K=ion_temperature,
        positive_ion_mass_kg=ion_mass_kg,
    )

    common = {
        "electrostatic_radius_m": radius,
        "electron_number_density_m3": electron_density,
        "positive_ion_number_density_m3": ion_density,
        "electron_temperature_K": electron_temperature,
        "positive_ion_temperature_K": ion_temperature,
        "particle_velocity_m_s": np.zeros((2, 2), dtype="<f8"),
        "positive_ion_velocity_m_s": np.zeros((2, 2), dtype="<f8"),
        "positive_ion_mass_kg": ion_mass_kg,
    }
    lower = oml_stationary_maxwellian_debye_huckel_v1(
        charge_number=bracket.lower,
        **common,
    )
    upper = oml_stationary_maxwellian_debye_huckel_v1(
        charge_number=bracket.upper,
        **common,
    )

    assert bracket.lower[0] < 0.0 and bracket.upper[0] == 0.0
    assert bracket.lower[1] == 0.0 and bracket.upper[1] > 0.0
    assert bool((lower.charge_rate_number_s >= 0.0).all())
    assert bool((upper.charge_rate_number_s <= 0.0).all())


def test_stationary_oml_global_invariant_and_bounds_dominate_field_corners() -> None:
    ion_mass_kg = 6.6335209e-26
    density_e = (8.0e14, 1.2e15)
    density_i = (7.0e14, 1.3e15)
    temperature_e = (1.5e4, 2.5e4)
    temperature_i = (400.0, 800.0)
    radii = (5.0e-8, 1.0e-7)
    bounds = oml_stationary_global_bounds(
        initial_charge_number=np.asarray([-3.0, 4.0], dtype="<f8"),
        electrostatic_radius_m=np.asarray(radii, dtype="<f8"),
        electron_number_density_lower_m3=density_e[0],
        electron_number_density_upper_m3=density_e[1],
        positive_ion_number_density_lower_m3=density_i[0],
        positive_ion_number_density_upper_m3=density_i[1],
        electron_temperature_lower_K=temperature_e[0],
        electron_temperature_upper_K=temperature_e[1],
        positive_ion_temperature_lower_K=temperature_i[0],
        positive_ion_temperature_upper_K=temperature_i[1],
        positive_ion_mass_kg=ion_mass_kg,
    )
    assert all(
        math.isfinite(value)
        for value in (
            bounds.charge_number_lower,
            bounds.charge_number_upper,
            bounds.charge_rate_abs_upper_number_s,
            bounds.charge_rate_derivative_abs_upper_s_inv,
            bounds.debye_length_lower_m,
            bounds.debye_length_upper_m,
            bounds.capacitance_lower_F,
            bounds.capacitance_upper_F,
        )
    )
    assert bounds.charge_number_lower <= -3.0
    assert bounds.charge_number_upper >= 4.0
    assert 0.0 < bounds.debye_length_lower_m <= bounds.debye_length_upper_m
    assert 0.0 < bounds.capacitance_lower_F <= bounds.capacitance_upper_F

    corners = np.asarray(
        list(product(radii, density_e, density_i, temperature_e, temperature_i)),
        dtype="<f8",
    )
    sample_count = int(corners.shape[0])
    repeated = np.vstack((corners, corners))
    charge = np.concatenate(
        (
            np.full(sample_count, bounds.charge_number_lower, dtype="<f8"),
            np.full(sample_count, bounds.charge_number_upper, dtype="<f8"),
        )
    )
    evaluation = oml_stationary_maxwellian_debye_huckel_v1(
        charge_number=charge,
        electrostatic_radius_m=repeated[:, 0],
        electron_number_density_m3=repeated[:, 1],
        positive_ion_number_density_m3=repeated[:, 2],
        electron_temperature_K=repeated[:, 3],
        positive_ion_temperature_K=repeated[:, 4],
        particle_velocity_m_s=np.zeros((2 * sample_count, 2), dtype="<f8"),
        positive_ion_velocity_m_s=np.zeros((2 * sample_count, 2), dtype="<f8"),
        positive_ion_mass_kg=ion_mass_kg,
    )

    assert bool((evaluation.charge_rate_number_s[:sample_count] >= 0.0).all())
    assert bool((evaluation.charge_rate_number_s[sample_count:] <= 0.0).all())
    assert bool(
        (np.abs(evaluation.charge_rate_number_s) <= bounds.charge_rate_abs_upper_number_s).all()
    )
    assert bool(
        (
            np.abs(evaluation.charge_rate_derivative_s_inv)
            <= bounds.charge_rate_derivative_abs_upper_s_inv
        ).all()
    )
    assert bool((evaluation.debye_length_m >= bounds.debye_length_lower_m).all())
    assert bool((evaluation.debye_length_m <= bounds.debye_length_upper_m).all())
    assert bool((evaluation.capacitance_F >= bounds.capacitance_lower_F).all())
    assert bool((evaluation.capacitance_F <= bounds.capacitance_upper_F).all())


def test_stationary_oml_applicability_exposes_both_dimensionless_gates() -> None:
    count = 3
    ion_mass_kg = 6.6335209e-26
    electron_density = np.full(count, 1.0e15, dtype="<f8")
    ion_density = np.full(count, 1.0e15, dtype="<f8")
    electron_temperature = np.full(count, 2.0e4, dtype="<f8")
    ion_temperature = np.full(count, 500.0, dtype="<f8")
    debye_length = oml_debye_length_m(
        electron_number_density_m3=electron_density,
        positive_ion_number_density_m3=ion_density,
        electron_temperature_K=electron_temperature,
        positive_ion_temperature_K=ion_temperature,
    )
    thermal_speed = math.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature[0] / (math.pi * ion_mass_kg))
    signed_drift_ratio = np.asarray([0.09, 0.09, -0.11], dtype="<f8")
    result = oml_stationary_maxwellian_debye_huckel_v1(
        charge_number=np.zeros(count, dtype="<f8"),
        electrostatic_radius_m=debye_length * np.asarray([0.09, 0.11, 0.09]),
        electron_number_density_m3=electron_density,
        positive_ion_number_density_m3=ion_density,
        electron_temperature_K=electron_temperature,
        positive_ion_temperature_K=ion_temperature,
        particle_velocity_m_s=np.zeros((count, 2), dtype="<f8"),
        positive_ion_velocity_m_s=np.column_stack(
            (signed_drift_ratio * thermal_speed, np.zeros(count, dtype="<f8"))
        ),
        positive_ion_mass_kg=ion_mass_kg,
    )

    np.testing.assert_allclose(result.radius_over_debye, [0.09, 0.11, 0.09], rtol=2.0e-15)
    np.testing.assert_allclose(result.ion_drift_ratio, np.abs(signed_drift_ratio), rtol=2.0e-15)
    np.testing.assert_array_equal(result.applicable, [True, False, False])
    assert result.charge_rate_number_s[0] == result.charge_rate_number_s[2]


def test_stationary_oml_rejects_nonpositive_or_nonfinite_inputs() -> None:
    with pytest.raises(PhysicsEvaluationError, match="electron_number_density_m3 must be positive"):
        oml_stationary_maxwellian_debye_huckel_v1(
            charge_number=np.asarray([0.0], dtype="<f8"),
            electrostatic_radius_m=np.asarray([1.0e-7], dtype="<f8"),
            electron_number_density_m3=np.asarray([0.0], dtype="<f8"),
            positive_ion_number_density_m3=np.asarray([1.0e15], dtype="<f8"),
            electron_temperature_K=np.asarray([2.0e4], dtype="<f8"),
            positive_ion_temperature_K=np.asarray([500.0], dtype="<f8"),
            particle_velocity_m_s=np.zeros((1, 2), dtype="<f8"),
            positive_ion_velocity_m_s=np.zeros((1, 2), dtype="<f8"),
            positive_ion_mass_kg=6.6335209e-26,
        )
    with pytest.raises(PhysicsEvaluationError, match="charge_number must be finite"):
        oml_stationary_maxwellian_debye_huckel_v1(
            charge_number=np.asarray([np.nan], dtype="<f8"),
            electrostatic_radius_m=np.asarray([1.0e-7], dtype="<f8"),
            electron_number_density_m3=np.asarray([1.0e15], dtype="<f8"),
            positive_ion_number_density_m3=np.asarray([1.0e15], dtype="<f8"),
            electron_temperature_K=np.asarray([2.0e4], dtype="<f8"),
            positive_ion_temperature_K=np.asarray([500.0], dtype="<f8"),
            particle_velocity_m_s=np.zeros((1, 2), dtype="<f8"),
            positive_ion_velocity_m_s=np.zeros((1, 2), dtype="<f8"),
            positive_ion_mass_kg=6.6335209e-26,
        )


def test_force_abs_upper_functions_dominate_selected_field_values() -> None:
    rate_upper = np.asarray([2.0, 5.0], dtype="<f8")
    target_abs_upper = np.asarray([0.4, 0.7], dtype="<f8")
    velocity_abs_upper = np.asarray([[0.9, 0.2], [0.3, 1.1]], dtype="<f8")
    drag_bound = linear_drag_acceleration_abs_upper(
        rate_upper_s_inv=rate_upper,
        target_velocity_abs_upper_m_s=target_abs_upper,
        velocity_abs_upper_m_s=velocity_abs_upper,
    )
    exact_drag_abs = rate_upper[:, None] * (target_abs_upper + velocity_abs_upper)
    assert bool((drag_bound > exact_drag_abs).all())
    with pytest.raises(PhysicsEvaluationError, match="uint8"):
        linear_drag_acceleration_abs_upper_batch(
            rate_upper_s_inv=rate_upper,
            target_velocity_abs_upper_m_s=target_abs_upper,
            velocity_abs_upper_m_s=velocity_abs_upper,
            numerical_status=np.asarray([256, 0], dtype="<i2"),
        )

    charge = np.asarray([-5.0, 2.0], dtype="<f8")
    mass = np.asarray([2.0e-18, 4.0e-18], dtype="<f8")
    field_abs_upper = np.asarray([4.0, 3.0], dtype="<f8")
    electric_bound = electric_acceleration_abs_upper(
        charge_number=charge,
        mass_kg=mass,
        electric_field_abs_upper_V_m=field_abs_upper,
    )
    exact_electric_abs = (
        np.abs(charge)[:, None] * ELEMENTARY_CHARGE_C * field_abs_upper / mass[:, None]
    )
    assert bool((electric_bound > exact_electric_abs).all())

    volume = np.asarray([0.0, 1.0e-15], dtype="<f8")
    gravity_bound = gravity_buoyancy_acceleration_abs_upper(
        mass_kg=mass,
        displaced_volume_m3=volume,
        gas_density_lower_kg_m3=0.8,
        gas_density_upper_kg_m3=1.2,
        gravity_m_s2=(2.0, -10.0),
    )
    for density in np.linspace(0.8, 1.2, 9):
        factor = 1.0 - density * volume / mass
        actual = np.abs(factor[:, None] * np.asarray([2.0, -10.0]))
        assert bool((actual <= gravity_bound).all())


def test_epstein_continuous_applicability_uses_path_and_field_bounds() -> None:
    molecular_mass_kg = 4.65e-26
    temperature_lower_K = 300.0
    mean_thermal_speed_m_s = math.sqrt(
        8.0 * BOLTZMANN_J_K * temperature_lower_K / (math.pi * molecular_mass_kg)
    )

    applicable = epstein_continuous_applicability(
        drag_diameter_m=np.asarray([0.02, 0.02, 0.022], dtype="<f8"),
        velocity_abs_upper_m_s=np.asarray(
            [
                [0.08 * mean_thermal_speed_m_s, 0.0],
                [0.11 * mean_thermal_speed_m_s, 0.0],
                [0.08 * mean_thermal_speed_m_s, 0.0],
            ],
            dtype="<f8",
        ),
        gas_velocity_abs_upper_m_s=np.asarray(
            [0.01 * mean_thermal_speed_m_s, 0.0],
            dtype="<f8",
        ),
        gas_temperature_lower_K=temperature_lower_K,
        gas_mean_free_path_lower_m=0.101,
        gas_molecular_mass_kg=molecular_mass_kg,
    )

    np.testing.assert_array_equal(applicable, [True, False, False])


def test_stokes_cunningham_continuous_applicability_uses_kn_and_re_bounds() -> None:
    applicable = stokes_cunningham_continuous_applicability(
        drag_diameter_m=np.asarray([2.0e-6, 100.0e-6, 0.1e-6, 2.0e-6], dtype="<f8"),
        velocity_abs_upper_m_s=np.asarray(
            [[1.0e-3, 0.0], [1.0e-3, 0.0], [1.0e-3, 0.0], [1.0, 0.0]],
            dtype="<f8",
        ),
        gas_velocity_abs_upper_m_s=np.asarray([0.0, 0.0], dtype="<f8"),
        gas_density_upper_kg_m3=1.0,
        gas_dynamic_viscosity_lower_Pa_s=1.8e-5,
        gas_mean_free_path_lower_m=0.5e-6,
        gas_mean_free_path_upper_m=1.0e-6,
    )

    np.testing.assert_array_equal(applicable, [True, False, False, False])


def test_continuous_applicability_batch_preserves_healthy_row_verdicts() -> None:
    diameter = np.asarray([2.0e-6, 100.0e-6, 0.1e-6, 2.0e-6], dtype="<f8")
    velocity = np.asarray(
        [[1.0e-3, 0.0], [1.0e-3, 0.0], [1.0e-3, 0.0], [1.0, 0.0]],
        dtype="<f8",
    )
    expected = stokes_cunningham_continuous_applicability(
        drag_diameter_m=diameter,
        velocity_abs_upper_m_s=velocity,
        gas_velocity_abs_upper_m_s=np.zeros(2, dtype="<f8"),
        gas_density_upper_kg_m3=1.0,
        gas_dynamic_viscosity_lower_Pa_s=1.8e-5,
        gas_mean_free_path_lower_m=0.5e-6,
        gas_mean_free_path_upper_m=1.0e-6,
    )

    actual, status = stokes_cunningham_continuous_applicability_batch(
        drag_diameter_m=diameter,
        velocity_abs_upper_m_s=velocity,
        gas_velocity_abs_upper_m_s=np.zeros(2, dtype="<f8"),
        gas_density_upper_kg_m3=1.0,
        gas_dynamic_viscosity_lower_Pa_s=1.8e-5,
        gas_mean_free_path_lower_m=0.5e-6,
        gas_mean_free_path_upper_m=1.0e-6,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(status, np.full(4, CONTINUOUS_APPLICABILITY_OK, dtype="u1"))


def test_continuous_applicability_batch_localizes_particle_numerical_failure() -> None:
    stokes, stokes_status = stokes_cunningham_continuous_applicability_batch(
        drag_diameter_m=np.full(2, 2.0e-6, dtype="<f8"),
        velocity_abs_upper_m_s=np.asarray([[1.0e10, 0.0], [0.0, 0.0]], dtype="<f8"),
        gas_velocity_abs_upper_m_s=np.zeros(2, dtype="<f8"),
        gas_density_upper_kg_m3=1.0e155,
        gas_dynamic_viscosity_lower_Pa_s=1.0e-155,
        gas_mean_free_path_lower_m=1.0e-6,
        gas_mean_free_path_upper_m=1.0e-6,
    )
    epstein, epstein_status = epstein_continuous_applicability_batch(
        drag_diameter_m=np.full(2, 2.0e-6, dtype="<f8"),
        velocity_abs_upper_m_s=np.asarray(
            [[np.finfo(np.float64).max, 0.0], [0.0, 0.0]],
            dtype="<f8",
        ),
        gas_velocity_abs_upper_m_s=np.zeros(2, dtype="<f8"),
        gas_temperature_lower_K=300.0,
        gas_mean_free_path_lower_m=2.0e-5,
        gas_molecular_mass_kg=4.65e-26,
    )

    np.testing.assert_array_equal(stokes, [False, True])
    np.testing.assert_array_equal(epstein, [False, True])
    expected_status = np.asarray(
        [CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE, CONTINUOUS_APPLICABILITY_OK],
        dtype="u1",
    )
    np.testing.assert_array_equal(stokes_status, expected_status)
    np.testing.assert_array_equal(epstein_status, expected_status)


def test_continuous_applicability_batch_rejects_shared_input_defects() -> None:
    with pytest.raises(PhysicsEvaluationError, match="shape"):
        epstein_continuous_applicability_batch(
            drag_diameter_m=np.full(2, 2.0e-6, dtype="<f8"),
            velocity_abs_upper_m_s=np.zeros((1, 2), dtype="<f8"),
            gas_velocity_abs_upper_m_s=np.zeros(2, dtype="<f8"),
            gas_temperature_lower_K=300.0,
            gas_mean_free_path_lower_m=2.0e-5,
            gas_molecular_mass_kg=4.65e-26,
        )
    with pytest.raises(PhysicsEvaluationError, match="nonnegative"):
        epstein_continuous_applicability_batch(
            drag_diameter_m=np.full(2, 2.0e-6, dtype="<f8"),
            velocity_abs_upper_m_s=np.asarray([[-1.0, 0.0], [0.0, 0.0]], dtype="<f8"),
            gas_velocity_abs_upper_m_s=np.zeros(2, dtype="<f8"),
            gas_temperature_lower_K=300.0,
            gas_mean_free_path_lower_m=2.0e-5,
            gas_molecular_mass_kg=4.65e-26,
        )


def test_stokes_cunningham_rejects_positive_inputs_when_rate_underflows_to_zero() -> None:
    with pytest.raises(PhysicsEvaluationError, match="rate is not finite and positive"):
        stokes_cunningham_linear_relaxation(
            mass_kg=np.asarray([1.0e308], dtype="<f8"),
            drag_diameter_m=np.asarray([1.0], dtype="<f8"),
            velocity_m_s=np.zeros((1, 2), dtype="<f8"),
            gas_velocity_m_s=np.zeros((1, 2), dtype="<f8"),
            gas_density_kg_m3=np.asarray([float(np.nextafter(0.0, np.inf))], dtype="<f8"),
            gas_dynamic_viscosity_Pa_s=np.asarray([float(np.nextafter(0.0, np.inf))], dtype="<f8"),
            gas_mean_free_path_m=np.asarray([0.5], dtype="<f8"),
        )


def test_physics_runtime_composes_stokes_and_prepares_continuous_bounds() -> None:
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "drag": _stokes_cunningham_model()},
        "cartesian_xy",
    )
    target = np.asarray([0.05, -0.02], dtype="<f8")
    ranges = {
        "gas_velocity": PrimitiveRange(target, target, target),
        "gas_density": _constant_range(0.01),
        "gas_dynamic_viscosity": _constant_range(1.8e-5),
        "gas_mean_free_path": _constant_range(1.0e-6),
    }
    mass = np.asarray([2.0e-15], dtype="<f8")
    diameter = np.asarray([2.0e-6], dtype="<f8")
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=mass,
        drag_diameter_m=diameter,
        electrostatic_radius_m=0.5 * diameter,
        displaced_volume_m3=np.asarray([0.0], dtype="<f8"),
        charge_number=np.asarray([0.0], dtype="<f8"),
        primitive_ranges=ranges,
    )
    sampled = {
        name: value.constant[None, :]
        for name, value in ranges.items()
        if value.constant is not None
    }
    velocity = np.asarray([[0.01, 0.0]], dtype="<f8")

    evaluation = runtime.evaluate(
        np.asarray([0], dtype="<i8"),
        velocity,
        np.asarray([0.0], dtype="<f8"),
        sampled,
    )
    direct = stokes_cunningham_linear_relaxation(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=velocity,
        gas_velocity_m_s=target[None, :],
        gas_density_kg_m3=np.asarray([0.01], dtype="<f8"),
        gas_dynamic_viscosity_Pa_s=np.asarray([1.8e-5], dtype="<f8"),
        gas_mean_free_path_m=np.asarray([1.0e-6], dtype="<f8"),
    )
    expected_acceleration = direct.rate_s_inv[:, None] * (target[None, :] - velocity)
    np.testing.assert_allclose(evaluation.acceleration_m_s2, expected_acceleration, rtol=1e-15)
    np.testing.assert_array_equal(evaluation.charge_rate_number_s, [0.0])
    np.testing.assert_array_equal(evaluation.charge_rate_derivative_s_inv, [0.0])
    np.testing.assert_array_equal(evaluation.applicable, [True])
    np.testing.assert_allclose(
        evaluation.linear_drag_rate_s_inv,
        direct.rate_s_inv,
        rtol=1e-15,
    )
    np.testing.assert_array_equal(evaluation.target_velocity_m_s, target[None, :])
    np.testing.assert_array_equal(evaluation.additive_acceleration_m_s2, [[0.0, 0.0]])
    assert runtime.constant_acceleration_m_s2 is None
    assert runtime.maximum_dt_over_tau(1.0e-6) > 0.0
    assert runtime.bound_array_nbytes > 0
    assert bool(
        runtime.continuous_applicability(
            np.asarray([0], dtype="<i8"),
            np.asarray([[0.06, 0.03]], dtype="<f8"),
        )[0]
    )
    batch_applicable, batch_status = runtime.continuous_applicability_batch(
        np.asarray([0], dtype="<i8"),
        np.asarray([[0.06, 0.03]], dtype="<f8"),
    )
    np.testing.assert_array_equal(batch_applicable, [True])
    np.testing.assert_array_equal(batch_status, [CONTINUOUS_APPLICABILITY_OK])
    failed_applicable, failed_status = runtime.continuous_applicability_batch(
        np.asarray([0], dtype="<i8"),
        np.asarray([[np.inf, 0.0]], dtype="<f8"),
    )
    np.testing.assert_array_equal(failed_applicable, [False])
    np.testing.assert_array_equal(
        failed_status,
        [CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE],
    )
    bound = runtime.acceleration_abs_upper(
        np.asarray([0], dtype="<i8"),
        np.asarray([[0.06, 0.03]], dtype="<f8"),
    )
    assert bool((np.abs(evaluation.acceleration_m_s2) <= bound).all())
    relaxation_bounds = runtime.linear_relaxation_abs_bounds(np.asarray([0], dtype="<i8"))
    assert bool((evaluation.linear_drag_rate_s_inv <= relaxation_bounds.rate_upper_s_inv).all())
    assert bool(
        (
            np.abs(evaluation.target_velocity_m_s)
            <= relaxation_bounds.target_velocity_abs_upper_m_s
        ).all()
    )
    np.testing.assert_array_equal(
        runtime.additive_acceleration_abs_upper(
            np.asarray([0], dtype="<i8"),
            np.asarray([[0.06, 0.03]], dtype="<f8"),
        ),
        [[0.0, 0.0]],
    )

    sampled_pair = {name: np.repeat(value, 2, axis=0) for name, value in sampled.items()}
    mixed, mixed_status = runtime.evaluate_batch(
        np.asarray([0, 0], dtype="<i8"),
        np.asarray([[0.01, 0.0], [np.inf, 0.0]], dtype="<f8"),
        np.asarray([0.0, 0.0], dtype="<f8"),
        sampled_pair,
    )
    np.testing.assert_array_equal(
        mixed_status,
        [NUMERICAL_STATUS_OK, PHYSICS_NUMERICAL_FAILURE],
    )
    np.testing.assert_array_equal(mixed.acceleration_m_s2[[0]], evaluation.acceleration_m_s2)
    np.testing.assert_array_equal(
        mixed.linear_drag_rate_s_inv[[0]], evaluation.linear_drag_rate_s_inv
    )
    np.testing.assert_array_equal(mixed.target_velocity_m_s[[0]], evaluation.target_velocity_m_s)
    assert np.isfinite(mixed.acceleration_m_s2).all()
    np.testing.assert_array_equal(mixed.acceleration_m_s2[1], [0.0, 0.0])

    preserved, preserved_status = runtime.evaluate_batch(
        np.asarray([0, 0], dtype="<i8"),
        np.repeat(velocity, 2, axis=0),
        np.asarray([0.0, 0.0], dtype="<f8"),
        sampled_pair,
        numerical_status=np.asarray(
            [NUMERICAL_STATUS_OK, FIELD_NUMERICAL_FAILURE],
            dtype="u1",
        ),
    )
    np.testing.assert_array_equal(
        preserved_status,
        [NUMERICAL_STATUS_OK, FIELD_NUMERICAL_FAILURE],
    )
    np.testing.assert_array_equal(preserved.acceleration_m_s2[[0]], evaluation.acceleration_m_s2)
    np.testing.assert_array_equal(preserved.acceleration_m_s2[1], [0.0, 0.0])
    with pytest.raises(PhysicsEvaluationError, match="uint8"):
        runtime.evaluate_batch(
            np.asarray([0], dtype="<i8"),
            velocity,
            np.asarray([0.0], dtype="<f8"),
            sampled,
            numerical_status=np.asarray([256], dtype="<i2"),
        )

    velocity_bound = np.asarray([[np.finfo(np.float64).max] * 2, [0.06, 0.03]], dtype="<f8")
    mixed_bound, bound_status = runtime.acceleration_abs_upper_batch(
        np.asarray([0, 0], dtype="<i8"),
        velocity_bound,
    )
    np.testing.assert_array_equal(
        bound_status,
        [PHYSICS_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK],
    )
    np.testing.assert_array_equal(mixed_bound[0], [0.0, 0.0])
    np.testing.assert_array_equal(
        mixed_bound[[1]],
        runtime.acceleration_abs_upper(
            np.asarray([0], dtype="<i8"),
            velocity_bound[[1]],
        ),
    )


@pytest.mark.parametrize("drag_kind", ["epstein", "finite_epstein", "stokes"])
def test_compiled_runtime_matches_numpy_force_composition(
    drag_kind: Literal["epstein", "finite_epstein", "stokes"],
) -> None:
    molecular_mass_kg = 4.65e-26
    if drag_kind == "epstein":
        drag_model: dict[str, object] = {
            "model": "epstein_linear",
            "revision": "epstein_linear_v1",
            "gas_velocity_field": "gas_velocity",
            "gas_density_field": "gas_density",
            "gas_temperature_field": "gas_temperature",
            "gas_mean_free_path_field": "gas_mean_free_path",
            "gas_molecular_mass_kg": molecular_mass_kg,
            "delta": 1.2,
            "applicability": "error",
        }
    elif drag_kind == "finite_epstein":
        drag_model = _finite_speed_epstein_model()
    else:
        drag_model = _stokes_cunningham_model()
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "drag": drag_model,
            "electric": {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
                "electric_field": "electric_field",
            },
            "gravity_buoyancy": {
                "model": "standard",
                "revision": "gravity_buoyancy_standard_v1",
                "gas_density_field": "gas_density",
                "gravity_m_s2": [0.5, -9.0],
            },
        },
        "cartesian_xy",
    )
    mass = np.asarray([2.0e-15, 3.0e-15, 5.0e-15], dtype="<f8")
    diameter = np.asarray([2.0e-6, 2.5e-6, 3.0e-6], dtype="<f8")
    displaced_volume = np.asarray([0.2e-15, 0.3e-15, 0.5e-15], dtype="<f8")
    resident_charge = np.asarray([-4.0, 0.0, 2.0], dtype="<f8")
    gas_velocity_lower = (
        np.asarray([-500.0, -500.0]) if drag_kind == "finite_epstein" else np.asarray([-0.3, -0.2])
    )
    gas_velocity_upper = (
        np.asarray([500.0, 500.0]) if drag_kind == "finite_epstein" else np.asarray([0.4, 0.5])
    )
    ranges = {
        "gas_velocity": PrimitiveRange(gas_velocity_lower, gas_velocity_upper, None),
        "gas_density": PrimitiveRange(np.asarray([0.01]), np.asarray([0.02]), None),
        "gas_temperature": PrimitiveRange(np.asarray([290.0]), np.asarray([320.0]), None),
        "gas_dynamic_viscosity": PrimitiveRange(np.asarray([1.7e-5]), np.asarray([1.9e-5]), None),
        "gas_mean_free_path": PrimitiveRange(np.asarray([2.0e-5]), np.asarray([3.0e-5]), None),
        "electric_field": PrimitiveRange(
            np.asarray([-2.0e4, -1.0e4]), np.asarray([3.0e4, 4.0e4]), None
        ),
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=mass,
        drag_diameter_m=diameter,
        electrostatic_radius_m=0.5 * diameter,
        displaced_volume_m3=displaced_volume,
        charge_number=resident_charge,
        primitive_ranges=ranges,
    )
    particle_index = np.asarray([2, 0], dtype="<i8")
    velocity = np.asarray([[0.02, -0.03], [-0.04, 0.01]], dtype="<f8")
    charge = np.asarray([2.5, -3.0], dtype="<f8")
    gas_density = np.asarray([0.012, 0.018], dtype="<f8")
    gas_temperature = np.asarray([300.0, 310.0], dtype="<f8")
    if drag_kind == "finite_epstein":
        speed_scale = np.sqrt(2.0 * BOLTZMANN_J_K * gas_temperature / molecular_mass_kg)
        gas_velocity = velocity + np.column_stack(
            (np.asarray([0.1, 1.0]) * speed_scale, np.zeros(2, dtype="<f8"))
        )
    else:
        gas_velocity = np.asarray([[0.05, -0.01], [-0.02, 0.04]], dtype="<f8")
    gas_viscosity = np.asarray([1.8e-5, 1.85e-5], dtype="<f8")
    gas_mean_free_path = np.asarray([2.5e-5, 2.8e-5], dtype="<f8")
    electric_field = np.asarray([[2.0e4, -1.0e4], [-1.5e4, 3.0e4]], dtype="<f8")
    sampled = {
        "gas_velocity": gas_velocity,
        "gas_density": gas_density[:, None],
        "gas_temperature": gas_temperature[:, None],
        "gas_dynamic_viscosity": gas_viscosity[:, None],
        "gas_mean_free_path": gas_mean_free_path[:, None],
        "electric_field": electric_field,
    }

    evaluation = runtime.evaluate(particle_index, velocity, charge, sampled)
    if drag_kind == "epstein":
        direct_drag = epstein_linear_relaxation(
            mass_kg=mass[particle_index],
            drag_diameter_m=diameter[particle_index],
            velocity_m_s=velocity,
            gas_velocity_m_s=gas_velocity,
            gas_density_kg_m3=gas_density,
            gas_temperature_K=gas_temperature,
            gas_mean_free_path_m=gas_mean_free_path,
            gas_molecular_mass_kg=molecular_mass_kg,
            delta=1.2,
        )
    elif drag_kind == "finite_epstein":
        direct_drag = epstein_finite_speed_relaxation(
            mass_kg=mass[particle_index],
            drag_diameter_m=diameter[particle_index],
            velocity_m_s=velocity,
            gas_velocity_m_s=gas_velocity,
            gas_density_kg_m3=gas_density,
            gas_temperature_K=gas_temperature,
            gas_mean_free_path_m=gas_mean_free_path,
            gas_molecular_mass_kg=molecular_mass_kg,
            diffuse_reflection_fraction=0.6,
            maximum_speed_ratio=3.0,
        )
    else:
        direct_drag = stokes_cunningham_linear_relaxation(
            mass_kg=mass[particle_index],
            drag_diameter_m=diameter[particle_index],
            velocity_m_s=velocity,
            gas_velocity_m_s=gas_velocity,
            gas_density_kg_m3=gas_density,
            gas_dynamic_viscosity_Pa_s=gas_viscosity,
            gas_mean_free_path_m=gas_mean_free_path,
        )
    expected_additive = np.zeros_like(velocity)
    add_electric_coulomb_acceleration(
        expected_additive,
        charge_number=charge,
        mass_kg=mass[particle_index],
        electric_field_V_m=electric_field,
    )
    add_gravity_buoyancy_acceleration(
        expected_additive,
        mass_kg=mass[particle_index],
        displaced_volume_m3=displaced_volume[particle_index],
        gas_density_kg_m3=gas_density,
        gravity_m_s2=(0.5, -9.0),
    )
    expected = direct_drag.rate_s_inv[:, None] * (gas_velocity - velocity)
    expected += expected_additive

    np.testing.assert_allclose(evaluation.acceleration_m_s2, expected, rtol=3.0e-15, atol=0.0)
    np.testing.assert_array_equal(evaluation.charge_rate_number_s, [0.0, 0.0])
    np.testing.assert_array_equal(evaluation.charge_rate_derivative_s_inv, [0.0, 0.0])
    np.testing.assert_array_equal(evaluation.applicable, direct_drag.applicable)
    np.testing.assert_allclose(
        evaluation.linear_drag_rate_s_inv,
        direct_drag.rate_s_inv,
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(evaluation.target_velocity_m_s, gas_velocity)
    np.testing.assert_allclose(
        evaluation.additive_acceleration_m_s2,
        expected_additive,
        rtol=3.0e-15,
        atol=0.0,
    )
    reconstructed = evaluation.linear_drag_rate_s_inv[:, None] * (
        evaluation.target_velocity_m_s - velocity
    )
    reconstructed += evaluation.additive_acceleration_m_s2
    np.testing.assert_allclose(reconstructed, evaluation.acceleration_m_s2, rtol=3.0e-15)
    relaxation_bounds = runtime.linear_relaxation_abs_bounds(particle_index)
    additive_acceleration_abs_upper = runtime.additive_acceleration_abs_upper(
        particle_index,
        np.abs(velocity),
    )
    assert bool((evaluation.linear_drag_rate_s_inv <= relaxation_bounds.rate_upper_s_inv).all())
    assert bool(
        (
            np.abs(evaluation.target_velocity_m_s)
            <= relaxation_bounds.target_velocity_abs_upper_m_s
        ).all()
    )
    assert bool(
        (np.abs(evaluation.additive_acceleration_m_s2) <= additive_acceleration_abs_upper).all()
    )
    assert evaluate_physics_tile_into.nopython_signatures


def test_physics_runtime_rejects_finite_inputs_that_overflow_combined_acceleration() -> None:
    plan = resolve_physics_plan(
        {"charge": {"model": "fixed"}, "drag": _stokes_cunningham_model()},
        "cartesian_xy",
    )
    ranges = {
        "gas_velocity": PrimitiveRange(
            np.zeros(2, dtype="<f8"),
            np.zeros(2, dtype="<f8"),
            np.zeros(2, dtype="<f8"),
        ),
        "gas_density": _constant_range(1.0e-310),
        "gas_dynamic_viscosity": _constant_range(1.0),
        "gas_mean_free_path": _constant_range(0.5),
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([2.0e-307], dtype="<f8"),
        drag_diameter_m=np.asarray([1.0], dtype="<f8"),
        electrostatic_radius_m=np.asarray([0.5], dtype="<f8"),
        displaced_volume_m3=np.asarray([0.0], dtype="<f8"),
        charge_number=np.asarray([0.0], dtype="<f8"),
        primitive_ranges=ranges,
    )
    sampled = {
        name: value.constant[None, :]
        for name, value in ranges.items()
        if value.constant is not None
    }

    with pytest.raises(PhysicsEvaluationError, match="acceleration is not finite"):
        runtime.evaluate(
            np.asarray([0], dtype="<i8"),
            np.asarray([[10.0, 0.0]], dtype="<f8"),
            np.asarray([0.0], dtype="<f8"),
            sampled,
        )


def test_force_bound_inputs_fail_instead_of_hiding_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="reversed"):
        gravity_buoyancy_acceleration_abs_upper(
            mass_kg=np.asarray([1.0], dtype="<f8"),
            displaced_volume_m3=np.asarray([0.0], dtype="<f8"),
            gas_density_lower_kg_m3=2.0,
            gas_density_upper_kg_m3=1.0,
            gravity_m_s2=(0.0, -9.8),
        )


def _stokes_cunningham_model() -> dict[str, object]:
    return {
        "model": "stokes_cunningham",
        "revision": "stokes_cunningham_allen_raabe_air_v1",
        "gas_velocity_field": "gas_velocity",
        "gas_density_field": "gas_density",
        "gas_dynamic_viscosity_field": "gas_dynamic_viscosity",
        "gas_mean_free_path_field": "gas_mean_free_path",
        "applicability": "error",
    }


def _epstein_linear_model() -> dict[str, object]:
    return {
        "model": "epstein_linear",
        "revision": "epstein_linear_v1",
        "gas_velocity_field": "gas_velocity",
        "gas_density_field": "gas_density",
        "gas_temperature_field": "gas_temperature",
        "gas_mean_free_path_field": "gas_mean_free_path",
        "gas_molecular_mass_kg": 4.65e-26,
        "delta": 1.2,
        "applicability": "error",
    }


def _inertial_langevin_noise_model() -> dict[str, object]:
    return {
        "model": "inertial_langevin_fdt",
        "revision": "inertial_langevin_fdt_epstein_linear_frozen_start_v1",
        "interval_tree_depth": 4,
    }


def _rz_inertial_langevin_noise_model() -> dict[str, object]:
    return {
        "model": "inertial_langevin_fdt",
        "revision": "inertial_langevin_fdt_epstein_linear_rz_meridional_projected_v1",
        "interval_tree_depth": 3,
    }


def _finite_speed_epstein_model() -> dict[str, object]:
    return {
        "model": "epstein_finite_speed",
        "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
        "gas_velocity_field": "gas_velocity",
        "gas_density_field": "gas_density",
        "gas_temperature_field": "gas_temperature",
        "gas_mean_free_path_field": "gas_mean_free_path",
        "gas_molecular_mass_kg": 4.65e-26,
        "diffuse_reflection_fraction": 0.6,
        "maximum_speed_ratio": 3.0,
        "applicability": "error",
    }


def _plasma_continuous_charge_model() -> dict[str, object]:
    return {
        "model": "plasma_continuous",
        "revision": "oml_stationary_maxwellian_debye_huckel_v1",
        "electron_number_density_field": "electron_number_density",
        "positive_ion_number_density_field": "positive_ion_number_density",
        "electron_temperature_field": "electron_temperature",
        "positive_ion_temperature_field": "positive_ion_temperature",
        "positive_ion_velocity_field": "positive_ion_velocity",
        "positive_ion_mass_kg": 6.6335209e-26,
        "applicability": "error",
    }


def _constant_range(value: float) -> PrimitiveRange:
    array = np.asarray([value], dtype="<f8")
    return PrimitiveRange(array, array, array)
