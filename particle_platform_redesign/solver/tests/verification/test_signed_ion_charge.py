from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray

from chamber_particles.numerical_status import NUMERICAL_STATUS_OK
from chamber_particles.physics.catalog import resolve_physics_plan
from chamber_particles.physics.charge import (
    aggregate_relative_drift_global_bounds,
    aggregate_relative_drift_regularized_three_current_v1,
    aggregate_relative_drift_regularized_two_current_v1,
    aggregate_relative_drift_three_current_global_bounds,
)
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime

_E = 1.602176634e-19
_EPSILON_0 = 8.8541878128e-12
_POSITIVE_ION_MASS_KG = 6.6335209e-26
_NEGATIVE_ION_MASS_KG = 5.3134e-26
_REVISION = "aggregate_relative_drift_regularized_three_current_v1"


class _CommonArguments(TypedDict):
    charge_number: NDArray[np.float64]
    electrostatic_radius_m: NDArray[np.float64]
    electron_number_density_m3: NDArray[np.float64]
    positive_ion_number_density_m3: NDArray[np.float64]
    electron_thermal_voltage_V: NDArray[np.float64]
    positive_ion_thermal_voltage_V: NDArray[np.float64]
    particle_velocity_m_s: NDArray[np.float64]
    positive_ion_velocity_m_s: NDArray[np.float64]
    effective_positive_ion_mass_kg: NDArray[np.float64]
    screening_length_m: NDArray[np.float64]
    maximum_relative_ion_speed_m_s: float


class _NegativeArguments(TypedDict):
    negative_ion_number_density_m3: NDArray[np.float64]
    negative_ion_thermal_voltage_V: NDArray[np.float64]
    negative_ion_velocity_m_s: NDArray[np.float64]
    effective_negative_ion_mass_kg: NDArray[np.float64]


class _BoundArguments(TypedDict):
    initial_charge_number: NDArray[np.float64]
    electrostatic_radius_m: NDArray[np.float64]
    electron_number_density_lower_m3: float
    electron_number_density_upper_m3: float
    positive_ion_number_density_lower_m3: float
    positive_ion_number_density_upper_m3: float
    electron_thermal_voltage_lower_V: float
    electron_thermal_voltage_upper_V: float
    positive_ion_thermal_voltage_lower_V: float
    positive_ion_thermal_voltage_upper_V: float
    effective_positive_ion_mass_lower_kg: float
    effective_positive_ion_mass_upper_kg: float
    screening_length_lower_m: float
    screening_length_upper_m: float
    maximum_relative_ion_speed_m_s: float


def _model(maximum_relative_speed_m_s: float = 200.0) -> dict[str, object]:
    return {
        "model": "plasma_continuous",
        "revision": _REVISION,
        "electron_number_density_field": "electron_density",
        "positive_ion_number_density_field": "positive_ion_density",
        "negative_ion_number_density_field": "negative_ion_density",
        "electron_thermal_voltage_field": "electron_voltage",
        "positive_ion_thermal_voltage_field": "positive_ion_voltage",
        "negative_ion_thermal_voltage_field": "negative_ion_voltage",
        "positive_ion_velocity_field": "positive_ion_velocity",
        "negative_ion_velocity_field": "negative_ion_velocity",
        "effective_positive_ion_mass_field": "positive_ion_mass",
        "effective_negative_ion_mass_field": "negative_ion_mass",
        "screening_length_field": "screening_length",
        "maximum_relative_ion_speed_m_s": maximum_relative_speed_m_s,
        "applicability": "error",
    }


def _common(count: int) -> _CommonArguments:
    return {
        "charge_number": np.linspace(-80.0, 20.0, count),
        "electrostatic_radius_m": np.linspace(4.0e-8, 7.0e-8, count),
        "electron_number_density_m3": np.linspace(1.0e15, 2.0e15, count),
        "positive_ion_number_density_m3": np.linspace(8.0e14, 1.5e15, count),
        "electron_thermal_voltage_V": np.linspace(2.0, 4.0, count),
        "positive_ion_thermal_voltage_V": np.linspace(0.02, 0.05, count),
        "particle_velocity_m_s": np.column_stack(
            (np.linspace(-20.0, 30.0, count), np.linspace(10.0, -15.0, count))
        ),
        "positive_ion_velocity_m_s": np.column_stack(
            (np.linspace(40.0, 80.0, count), np.linspace(-30.0, 20.0, count))
        ),
        "effective_positive_ion_mass_kg": np.linspace(
            0.8 * _POSITIVE_ION_MASS_KG,
            1.2 * _POSITIVE_ION_MASS_KG,
            count,
        ),
        "screening_length_m": np.linspace(1.0e-4, 3.0e-4, count),
        "maximum_relative_ion_speed_m_s": 300.0,
    }


def _negative_arguments(
    count: int,
    density_m3: NDArray[np.float64],
) -> _NegativeArguments:
    return {
        "negative_ion_number_density_m3": density_m3,
        "negative_ion_thermal_voltage_V": np.linspace(0.015, 0.04, count),
        "negative_ion_velocity_m_s": np.column_stack(
            (np.linspace(-50.0, 10.0, count), np.linspace(25.0, -35.0, count))
        ),
        "effective_negative_ion_mass_kg": np.linspace(
            0.7 * _NEGATIVE_ION_MASS_KG,
            1.3 * _NEGATIVE_ION_MASS_KG,
            count,
        ),
    }


def _range(lower: tuple[float, ...], upper: tuple[float, ...]) -> PrimitiveRange:
    return PrimitiveRange(np.asarray(lower), np.asarray(upper), None)


def test_three_current_zero_negative_density_is_exactly_two_current() -> None:
    common = _common(5)
    two_current = aggregate_relative_drift_regularized_two_current_v1(**common)
    three_current = aggregate_relative_drift_regularized_three_current_v1(
        **common,
        **_negative_arguments(5, np.zeros(5)),
    )

    np.testing.assert_array_equal(
        three_current.charge_rate_number_s,
        two_current.charge_rate_number_s,
    )
    np.testing.assert_array_equal(
        three_current.charge_rate_derivative_s_inv,
        two_current.charge_rate_derivative_s_inv,
    )
    np.testing.assert_array_equal(three_current.applicable, two_current.applicable)


def test_three_current_zero_negative_density_has_exactly_two_current_bounds() -> None:
    common: _BoundArguments = {
        "initial_charge_number": np.asarray([-20.0, 5.0]),
        "electrostatic_radius_m": np.asarray([4.0e-8, 8.0e-8]),
        "electron_number_density_lower_m3": 8.0e14,
        "electron_number_density_upper_m3": 2.0e15,
        "positive_ion_number_density_lower_m3": 7.0e14,
        "positive_ion_number_density_upper_m3": 1.8e15,
        "electron_thermal_voltage_lower_V": 2.0,
        "electron_thermal_voltage_upper_V": 4.0,
        "positive_ion_thermal_voltage_lower_V": 0.02,
        "positive_ion_thermal_voltage_upper_V": 0.06,
        "effective_positive_ion_mass_lower_kg": 0.8 * _POSITIVE_ION_MASS_KG,
        "effective_positive_ion_mass_upper_kg": 1.2 * _POSITIVE_ION_MASS_KG,
        "screening_length_lower_m": 1.0e-4,
        "screening_length_upper_m": 3.0e-4,
        "maximum_relative_ion_speed_m_s": 250.0,
    }
    two_current = aggregate_relative_drift_global_bounds(**common)
    three_current = aggregate_relative_drift_three_current_global_bounds(
        **common,
        negative_ion_number_density_lower_m3=0.0,
        negative_ion_number_density_upper_m3=0.0,
        negative_ion_thermal_voltage_lower_V=0.015,
        negative_ion_thermal_voltage_upper_V=0.05,
        effective_negative_ion_mass_lower_kg=0.7 * _NEGATIVE_ION_MASS_KG,
        effective_negative_ion_mass_upper_kg=1.3 * _NEGATIVE_ION_MASS_KG,
    )

    assert three_current == two_current


def test_three_current_negative_ion_rate_and_derivative_match_independent_equation() -> None:
    common = _common(3)
    common["charge_number"] = np.asarray([-100.0, 0.0, 60.0])
    negative = _negative_arguments(3, np.asarray([2.0e14, 4.0e14, 6.0e14]))
    base = aggregate_relative_drift_regularized_two_current_v1(**common)
    actual = aggregate_relative_drift_regularized_three_current_v1(**common, **negative)

    radius = common["electrostatic_radius_m"]
    charge = common["charge_number"]
    particle_velocity = common["particle_velocity_m_s"]
    assert isinstance(radius, np.ndarray)
    assert isinstance(charge, np.ndarray)
    assert isinstance(particle_velocity, np.ndarray)
    negative_velocity = negative["negative_ion_velocity_m_s"]
    negative_voltage = negative["negative_ion_thermal_voltage_V"]
    negative_mass = negative["effective_negative_ion_mass_kg"]
    relative_speed_squared = np.sum((negative_velocity - particle_velocity) ** 2, axis=1)
    effective_speed_squared = (
        relative_speed_squared + 8.0 * _E * negative_voltage / (math.pi * negative_mass) + 1.0
    )
    effective_speed = np.sqrt(effective_speed_squared)
    effective_energy = np.maximum(negative_mass * effective_speed_squared / (2.0 * _E), 0.01)
    screening = common["screening_length_m"]
    assert isinstance(screening, np.ndarray)
    capacitance = (
        4.0 * math.pi * _EPSILON_0 * radius * (1.0 + radius / np.maximum(radius, screening))
    )
    potential_per_charge = _E / capacitance
    potential = charge * potential_per_charge
    amplitude = math.pi * radius**2 * negative["negative_ion_number_density_m3"] * effective_speed
    nonpositive = potential <= 0.0
    factor = np.where(
        nonpositive,
        np.exp(np.clip(potential / effective_energy, -50.0, 50.0)),
        1.0 + potential / effective_energy,
    )
    factor_derivative = np.where(
        nonpositive,
        factor * potential_per_charge / effective_energy,
        potential_per_charge / effective_energy,
    )
    expected_rate = base.charge_rate_number_s - amplitude * factor
    expected_derivative = base.charge_rate_derivative_s_inv - amplitude * factor_derivative

    np.testing.assert_allclose(actual.charge_rate_number_s, expected_rate, rtol=5.0e-15)
    np.testing.assert_allclose(
        actual.charge_rate_derivative_s_inv,
        expected_derivative,
        rtol=5.0e-15,
    )
    assert bool((actual.charge_rate_number_s <= base.charge_rate_number_s).all())
    assert bool((actual.charge_rate_number_s < base.charge_rate_number_s).any())
    assert bool((actual.charge_rate_derivative_s_inv < 0.0).all())


def test_negative_ion_saturated_exponential_adds_no_jacobian() -> None:
    common = _common(1)
    common["charge_number"] = np.asarray([-100.0])
    negative = _negative_arguments(1, np.asarray([1.0e34]))
    base = aggregate_relative_drift_regularized_two_current_v1(**common)
    actual = aggregate_relative_drift_regularized_three_current_v1(**common, **negative)

    assert actual.surface_potential_V[0] / actual.negative_effective_ion_energy_V[0] < -50.0
    assert actual.charge_rate_number_s[0] < base.charge_rate_number_s[0]
    np.testing.assert_array_equal(
        actual.charge_rate_derivative_s_inv,
        base.charge_rate_derivative_s_inv,
    )


def test_three_current_bounds_enclose_random_primitive_samples() -> None:
    bounds = aggregate_relative_drift_three_current_global_bounds(
        initial_charge_number=np.asarray([-20.0, 5.0]),
        electrostatic_radius_m=np.asarray([4.0e-8, 8.0e-8]),
        electron_number_density_lower_m3=8.0e14,
        electron_number_density_upper_m3=2.0e15,
        positive_ion_number_density_lower_m3=7.0e14,
        positive_ion_number_density_upper_m3=1.8e15,
        negative_ion_number_density_lower_m3=0.0,
        negative_ion_number_density_upper_m3=8.0e14,
        electron_thermal_voltage_lower_V=2.0,
        electron_thermal_voltage_upper_V=4.0,
        positive_ion_thermal_voltage_lower_V=0.02,
        positive_ion_thermal_voltage_upper_V=0.06,
        negative_ion_thermal_voltage_lower_V=0.015,
        negative_ion_thermal_voltage_upper_V=0.05,
        effective_positive_ion_mass_lower_kg=0.8 * _POSITIVE_ION_MASS_KG,
        effective_positive_ion_mass_upper_kg=1.2 * _POSITIVE_ION_MASS_KG,
        effective_negative_ion_mass_lower_kg=0.7 * _NEGATIVE_ION_MASS_KG,
        effective_negative_ion_mass_upper_kg=1.3 * _NEGATIVE_ION_MASS_KG,
        screening_length_lower_m=1.0e-4,
        screening_length_upper_m=3.0e-4,
        maximum_relative_ion_speed_m_s=250.0,
    )
    rng = np.random.default_rng(90210)
    count = 512
    charge = rng.uniform(bounds.charge_number_lower, bounds.charge_number_upper, count)
    charge[:2] = (bounds.charge_number_lower, bounds.charge_number_upper)
    particle_velocity = np.zeros((count, 2))
    positive_angle = rng.uniform(0.0, 2.0 * math.pi, count)
    negative_angle = rng.uniform(0.0, 2.0 * math.pi, count)
    positive_speed = rng.uniform(0.0, 250.0, count)
    negative_speed = rng.uniform(0.0, 250.0, count)
    evaluation = aggregate_relative_drift_regularized_three_current_v1(
        charge_number=charge,
        electrostatic_radius_m=rng.uniform(4.0e-8, 8.0e-8, count),
        electron_number_density_m3=rng.uniform(8.0e14, 2.0e15, count),
        positive_ion_number_density_m3=rng.uniform(7.0e14, 1.8e15, count),
        negative_ion_number_density_m3=rng.uniform(0.0, 8.0e14, count),
        electron_thermal_voltage_V=rng.uniform(2.0, 4.0, count),
        positive_ion_thermal_voltage_V=rng.uniform(0.02, 0.06, count),
        negative_ion_thermal_voltage_V=rng.uniform(0.015, 0.05, count),
        particle_velocity_m_s=particle_velocity,
        positive_ion_velocity_m_s=np.column_stack(
            (positive_speed * np.cos(positive_angle), positive_speed * np.sin(positive_angle))
        ),
        negative_ion_velocity_m_s=np.column_stack(
            (negative_speed * np.cos(negative_angle), negative_speed * np.sin(negative_angle))
        ),
        effective_positive_ion_mass_kg=rng.uniform(
            0.8 * _POSITIVE_ION_MASS_KG,
            1.2 * _POSITIVE_ION_MASS_KG,
            count,
        ),
        effective_negative_ion_mass_kg=rng.uniform(
            0.7 * _NEGATIVE_ION_MASS_KG,
            1.3 * _NEGATIVE_ION_MASS_KG,
            count,
        ),
        screening_length_m=rng.uniform(1.0e-4, 3.0e-4, count),
        maximum_relative_ion_speed_m_s=250.0,
    )

    assert abs(evaluation.charge_rate_number_s[0]) <= bounds.charge_rate_abs_upper_number_s
    assert evaluation.charge_rate_number_s[0] >= -2.0e-13 * bounds.charge_rate_abs_upper_number_s
    assert evaluation.charge_rate_number_s[1] <= 2.0e-13 * bounds.charge_rate_abs_upper_number_s
    assert bool(
        (np.abs(evaluation.charge_rate_number_s) <= bounds.charge_rate_abs_upper_number_s).all()
    )
    assert bool(
        (
            np.abs(evaluation.charge_rate_derivative_s_inv)
            <= bounds.charge_rate_derivative_abs_upper_s_inv
        ).all()
    )


def test_three_current_catalog_and_compiled_runtime_use_both_ion_populations() -> None:
    plan = resolve_physics_plan({"charge": _model()}, "cartesian_xy")
    assert plan.charge is not None
    assert plan.charge.revision == _REVISION
    assert {field.name for field in plan.required_fields} == {
        "electron_density",
        "positive_ion_density",
        "negative_ion_density",
        "electron_voltage",
        "positive_ion_voltage",
        "negative_ion_voltage",
        "positive_ion_velocity",
        "negative_ion_velocity",
        "positive_ion_mass",
        "negative_ion_mass",
        "screening_length",
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.full(3, 2.0e-15),
        drag_diameter_m=np.full(3, 2.0e-6),
        electrostatic_radius_m=np.asarray([4.0e-8, 5.0e-8, 6.0e-8]),
        displaced_volume_m3=np.zeros(3),
        charge_number=np.asarray([-80.0, -20.0, 10.0]),
        primitive_ranges={
            "electron_density": _range((1.0e15,), (2.0e15,)),
            "positive_ion_density": _range((8.0e14,), (1.5e15,)),
            "negative_ion_density": _range((0.0,), (6.0e14,)),
            "electron_voltage": _range((2.0,), (4.0,)),
            "positive_ion_voltage": _range((0.02,), (0.05,)),
            "negative_ion_voltage": _range((0.015,), (0.04,)),
            "positive_ion_velocity": _range((-40.0, -20.0), (80.0, 40.0)),
            "negative_ion_velocity": _range((-250.0, -20.0), (20.0, 40.0)),
            "positive_ion_mass": _range(
                (0.8 * _POSITIVE_ION_MASS_KG,),
                (1.2 * _POSITIVE_ION_MASS_KG,),
            ),
            "negative_ion_mass": _range(
                (0.7 * _NEGATIVE_ION_MASS_KG,),
                (1.3 * _NEGATIVE_ION_MASS_KG,),
            ),
            "screening_length": _range((1.0e-4,), (3.0e-4,)),
        },
    )
    particle_velocity = np.asarray([[0.0, 0.0], [20.0, -10.0], [0.0, 0.0]])
    positive_velocity = np.asarray([[40.0, 0.0], [20.0, -10.0], [80.0, 40.0]])
    negative_velocity = np.asarray([[-50.0, 25.0], [0.0, 0.0], [-250.0, 0.0]])
    charge = np.asarray([-80.0, -20.0, 10.0])
    sampled = {
        "electron_density": np.asarray([[1.0e15], [1.5e15], [2.0e15]]),
        "positive_ion_density": np.asarray([[8.0e14], [1.0e15], [1.5e15]]),
        "negative_ion_density": np.asarray([[2.0e14], [0.0], [6.0e14]]),
        "electron_voltage": np.asarray([[2.0], [3.0], [4.0]]),
        "positive_ion_voltage": np.asarray([[0.02], [0.03], [0.05]]),
        "negative_ion_voltage": np.asarray([[0.015], [0.025], [0.04]]),
        "positive_ion_velocity": positive_velocity,
        "negative_ion_velocity": negative_velocity,
        "positive_ion_mass": _POSITIVE_ION_MASS_KG * np.asarray([[0.8], [1.0], [1.2]]),
        "negative_ion_mass": _NEGATIVE_ION_MASS_KG * np.asarray([[0.7], [1.0], [1.3]]),
        "screening_length": np.asarray([[1.0e-4], [2.0e-4], [3.0e-4]]),
    }
    actual, status = runtime.evaluate_batch(
        np.arange(3, dtype=np.int64),
        particle_velocity,
        charge,
        sampled,
    )
    expected = aggregate_relative_drift_regularized_three_current_v1(
        charge_number=charge,
        electrostatic_radius_m=runtime.electrostatic_radius_m,
        electron_number_density_m3=sampled["electron_density"][:, 0],
        positive_ion_number_density_m3=sampled["positive_ion_density"][:, 0],
        negative_ion_number_density_m3=sampled["negative_ion_density"][:, 0],
        electron_thermal_voltage_V=sampled["electron_voltage"][:, 0],
        positive_ion_thermal_voltage_V=sampled["positive_ion_voltage"][:, 0],
        negative_ion_thermal_voltage_V=sampled["negative_ion_voltage"][:, 0],
        particle_velocity_m_s=particle_velocity,
        positive_ion_velocity_m_s=positive_velocity,
        negative_ion_velocity_m_s=negative_velocity,
        effective_positive_ion_mass_kg=sampled["positive_ion_mass"][:, 0],
        effective_negative_ion_mass_kg=sampled["negative_ion_mass"][:, 0],
        screening_length_m=sampled["screening_length"][:, 0],
        maximum_relative_ion_speed_m_s=200.0,
    )

    np.testing.assert_array_equal(status, NUMERICAL_STATUS_OK)
    np.testing.assert_allclose(
        actual.charge_rate_number_s, expected.charge_rate_number_s, rtol=5e-15
    )
    np.testing.assert_allclose(
        actual.charge_rate_derivative_s_inv,
        expected.charge_rate_derivative_s_inv,
        rtol=5e-15,
    )
    np.testing.assert_array_equal(actual.applicable, [True, True, False])
    path_applicable, path_status = runtime.continuous_applicability_batch(
        np.asarray([0], dtype=np.int64),
        np.zeros((1, 2)),
    )
    np.testing.assert_array_equal(path_status, NUMERICAL_STATUS_OK)
    np.testing.assert_array_equal(path_applicable, [False])

    missing = _model()
    missing.pop("negative_ion_velocity_field")
    with pytest.raises(ValueError, match="keys do not match"):
        resolve_physics_plan({"charge": missing}, "cartesian_xy")
