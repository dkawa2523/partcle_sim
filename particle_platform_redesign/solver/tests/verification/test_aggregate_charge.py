from __future__ import annotations

import itertools
import math
from decimal import Decimal, getcontext

import numpy as np
import pytest

from chamber_particles.numerical_status import NUMERICAL_STATUS_OK
from chamber_particles.physics.catalog import resolve_physics_plan
from chamber_particles.physics.charge import (
    AggregateChargeBounds,
    AggregateChargeEvaluation,
    aggregate_relative_drift_global_bounds,
    aggregate_relative_drift_regularized_two_current_v1,
)
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime

_E = 1.602176634e-19
_EPSILON_0 = 8.8541878128e-12
_ELECTRON_MASS_KG = 9.1093837139e-31
_ION_MASS_KG = 6.6335209e-26
_PI_DECIMAL = Decimal("3.1415926535897932384626433832795028841971693993751")
_EXPONENT_MIN = Decimal(-50)
_SPEED_REGULARIZATION_M_S = Decimal(1)
_ION_ENERGY_FLOOR_V = Decimal("0.01")
_REVISION = "aggregate_relative_drift_regularized_two_current_v1"


def _charge_model(maximum_relative_speed_m_s: float = 200.0) -> dict[str, object]:
    return {
        "model": "plasma_continuous",
        "revision": _REVISION,
        "electron_number_density_field": "electron_density",
        "positive_ion_number_density_field": "ion_density",
        "electron_thermal_voltage_field": "electron_voltage",
        "positive_ion_thermal_voltage_field": "ion_voltage",
        "positive_ion_velocity_field": "ion_velocity",
        "effective_positive_ion_mass_field": "ion_mass",
        "screening_length_field": "screening_length",
        "maximum_relative_ion_speed_m_s": maximum_relative_speed_m_s,
        "applicability": "error",
    }


def _barnes_model() -> dict[str, object]:
    return {
        "model": "barnes_collisionless",
        "revision": (
            "barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1"
        ),
        "electron_number_density_field": "electron_density",
        "positive_ion_number_density_field": "ion_density",
        "electron_temperature_field": "electron_temperature",
        "positive_ion_temperature_field": "ion_temperature",
        "positive_ion_velocity_field": "ion_velocity",
        "ion_neutral_mean_free_path_field": "ion_mean_free_path",
        "positive_ion_mass_kg": _ION_MASS_KG,
        "maximum_ion_drift_ratio": 3.0,
        "applicability": "error",
    }


def _range(lower: tuple[float, ...], upper: tuple[float, ...]) -> PrimitiveRange:
    return PrimitiveRange(np.asarray(lower), np.asarray(upper), None)


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _oracle(
    *,
    charge_number: float,
    radius_m: float,
    electron_density_m3: float,
    ion_density_m3: float,
    electron_voltage_V: float,
    ion_voltage_V: float,
    particle_velocity_m_s: tuple[float, float],
    ion_velocity_m_s: tuple[float, float],
    ion_mass_kg: float,
    screening_length_m: float,
) -> dict[str, float]:
    """Evaluate the documented scalar equation without production helpers."""

    getcontext().prec = 60
    elementary_charge = Decimal("1.602176634e-19")
    permittivity = Decimal("8.8541878128e-12")
    electron_mass = Decimal("9.1093837139e-31")
    charge = _decimal(charge_number)
    radius = _decimal(radius_m)
    electron_density = _decimal(electron_density_m3)
    ion_density = _decimal(ion_density_m3)
    electron_voltage = _decimal(electron_voltage_V)
    ion_voltage = _decimal(ion_voltage_V)
    ion_mass = _decimal(ion_mass_kg)
    screening_length = _decimal(screening_length_m)
    effective_screening_length = max(radius, screening_length)
    relative_x = _decimal(ion_velocity_m_s[0]) - _decimal(particle_velocity_m_s[0])
    relative_y = _decimal(ion_velocity_m_s[1]) - _decimal(particle_velocity_m_s[1])

    capacitance = (
        Decimal(4)
        * _PI_DECIMAL
        * permittivity
        * radius
        * (Decimal(1) + radius / effective_screening_length)
    )
    potential_per_charge = elementary_charge / capacitance
    surface_potential = charge * potential_per_charge
    relative_speed_squared = relative_x * relative_x + relative_y * relative_y
    effective_speed_squared = (
        relative_speed_squared
        + Decimal(8) * elementary_charge * ion_voltage / (_PI_DECIMAL * ion_mass)
        + _SPEED_REGULARIZATION_M_S**2
    )
    effective_speed = effective_speed_squared.sqrt()
    effective_ion_energy = max(
        ion_mass * effective_speed_squared / (Decimal(2) * elementary_charge),
        _ION_ENERGY_FLOOR_V,
    )
    ion_amplitude = _PI_DECIMAL * radius * radius * ion_density * effective_speed
    electron_amplitude = (
        _PI_DECIMAL
        * radius
        * radius
        * electron_density
        * (Decimal(8) * elementary_charge * electron_voltage / (_PI_DECIMAL * electron_mass)).sqrt()
    )

    if surface_potential <= 0:
        raw_exponent = surface_potential / electron_voltage
        exponent = max(_EXPONENT_MIN, min(Decimal(50), raw_exponent))
        electron_factor = exponent.exp()
        ion_factor = Decimal(1) - surface_potential / effective_ion_energy
        derivative = -ion_amplitude * potential_per_charge / effective_ion_energy
        if raw_exponent >= _EXPONENT_MIN:
            derivative -= (
                electron_amplitude * electron_factor * potential_per_charge / electron_voltage
            )
    else:
        raw_exponent = -surface_potential / effective_ion_energy
        exponent = max(_EXPONENT_MIN, min(Decimal(50), raw_exponent))
        ion_factor = exponent.exp()
        electron_factor = Decimal(1) + surface_potential / electron_voltage
        derivative = -electron_amplitude * potential_per_charge / electron_voltage
        if raw_exponent >= _EXPONENT_MIN:
            derivative -= ion_amplitude * ion_factor * potential_per_charge / effective_ion_energy

    return {
        "rate": float(ion_amplitude * ion_factor - electron_amplitude * electron_factor),
        "derivative": float(derivative),
        "surface_potential": float(surface_potential),
        "capacitance": float(capacitance),
        "relative_speed": float(relative_speed_squared.sqrt()),
        "effective_speed": float(effective_speed),
        "effective_ion_energy": float(effective_ion_energy),
        "effective_screening_length": float(effective_screening_length),
    }


def _evaluate(
    *,
    charge_number: np.ndarray,
    radius_m: np.ndarray,
    electron_density_m3: np.ndarray,
    ion_density_m3: np.ndarray,
    electron_voltage_V: np.ndarray,
    ion_voltage_V: np.ndarray,
    particle_velocity_m_s: np.ndarray,
    ion_velocity_m_s: np.ndarray,
    ion_mass_kg: np.ndarray,
    screening_length_m: np.ndarray,
    maximum_relative_speed_m_s: float = 1.0e6,
) -> AggregateChargeEvaluation:
    return aggregate_relative_drift_regularized_two_current_v1(
        charge_number=charge_number,
        electrostatic_radius_m=radius_m,
        electron_number_density_m3=electron_density_m3,
        positive_ion_number_density_m3=ion_density_m3,
        electron_thermal_voltage_V=electron_voltage_V,
        positive_ion_thermal_voltage_V=ion_voltage_V,
        particle_velocity_m_s=particle_velocity_m_s,
        positive_ion_velocity_m_s=ion_velocity_m_s,
        effective_positive_ion_mass_kg=ion_mass_kg,
        screening_length_m=screening_length_m,
        maximum_relative_ion_speed_m_s=maximum_relative_speed_m_s,
    )


def _uniform_evaluation(
    charge_number: np.ndarray,
    *,
    electron_density_m3: np.ndarray | None = None,
    ion_density_m3: np.ndarray | None = None,
) -> AggregateChargeEvaluation:
    count = int(charge_number.size)
    return _evaluate(
        charge_number=charge_number,
        radius_m=np.full(count, 5.0e-8),
        electron_density_m3=(
            np.full(count, 2.0e15) if electron_density_m3 is None else electron_density_m3
        ),
        ion_density_m3=(np.full(count, 1.5e15) if ion_density_m3 is None else ion_density_m3),
        electron_voltage_V=np.full(count, 3.0),
        ion_voltage_V=np.full(count, 0.03),
        particle_velocity_m_s=np.zeros((count, 2)),
        ion_velocity_m_s=np.broadcast_to(np.asarray([100.0, -40.0]), (count, 2)).copy(),
        ion_mass_kg=np.full(count, _ION_MASS_KG),
        screening_length_m=np.full(count, 2.0e-4),
    )


def _potential_per_charge(radius_m: float, screening_length_m: float) -> float:
    effective_screening_length_m = max(radius_m, screening_length_m)
    capacitance = (
        4.0 * math.pi * _EPSILON_0 * radius_m * (1.0 + radius_m / effective_screening_length_m)
    )
    return _E / capacitance


def _effective_ion_energy_V(
    ion_voltage_V: float,
    ion_mass_kg: float,
    relative_velocity_m_s: tuple[float, float],
) -> float:
    speed_squared = (
        relative_velocity_m_s[0] ** 2
        + relative_velocity_m_s[1] ** 2
        + 8.0 * _E * ion_voltage_V / (math.pi * ion_mass_kg)
        + 1.0
    )
    return max(ion_mass_kg * speed_squared / (2.0 * _E), 0.01)


def test_aggregate_charge_matches_independent_decimal_oracle_across_all_pieces() -> None:
    radius = 5.0e-8
    screening = 2.0e-4
    electron_voltage = 3.0
    ion_voltage = 0.03
    relative_velocity = (100.0, -40.0)
    potential_per_charge = _potential_per_charge(radius, screening)
    ion_energy = _effective_ion_energy_V(
        ion_voltage,
        _ION_MASS_KG,
        relative_velocity,
    )
    charges = np.asarray(
        [
            -20.0,
            0.0,
            10.0,
            -2.0,
            -60.0 * electron_voltage / potential_per_charge,
            60.0 * ion_energy / potential_per_charge,
        ]
    )
    count = int(charges.size)
    ion_masses = _ION_MASS_KG * np.asarray([1.0, 1.25, 0.75, 1.5, 1.0, 1.0])
    ion_voltages = np.asarray(
        [ion_voltage, ion_voltage, ion_voltage, 1.0e-6, ion_voltage, ion_voltage]
    )
    screening_lengths = np.asarray([screening, screening, screening, 1.0e-9, screening, screening])
    particle_velocity = np.zeros((count, 2))
    ion_velocity = np.broadcast_to(np.asarray(relative_velocity), (count, 2)).copy()
    ion_velocity[3] = 0.0

    result = _evaluate(
        charge_number=charges,
        radius_m=np.full(count, radius),
        electron_density_m3=np.full(count, 2.0e15),
        ion_density_m3=np.full(count, 1.5e15),
        electron_voltage_V=np.full(count, electron_voltage),
        ion_voltage_V=ion_voltages,
        particle_velocity_m_s=particle_velocity,
        ion_velocity_m_s=ion_velocity,
        ion_mass_kg=ion_masses,
        screening_length_m=screening_lengths,
        maximum_relative_speed_m_s=200.0,
    )
    expected = [
        _oracle(
            charge_number=float(charges[index]),
            radius_m=radius,
            electron_density_m3=2.0e15,
            ion_density_m3=1.5e15,
            electron_voltage_V=electron_voltage,
            ion_voltage_V=float(ion_voltages[index]),
            particle_velocity_m_s=tuple(particle_velocity[index]),
            ion_velocity_m_s=tuple(ion_velocity[index]),
            ion_mass_kg=float(ion_masses[index]),
            screening_length_m=float(screening_lengths[index]),
        )
        for index in range(count)
    ]

    for attribute, key in (
        (result.charge_rate_number_s, "rate"),
        (result.charge_rate_derivative_s_inv, "derivative"),
        (result.surface_potential_V, "surface_potential"),
        (result.capacitance_F, "capacitance"),
        (result.relative_ion_speed_m_s, "relative_speed"),
        (result.effective_ion_speed_m_s, "effective_speed"),
        (result.effective_ion_energy_V, "effective_ion_energy"),
    ):
        np.testing.assert_allclose(
            attribute,
            np.asarray([row[key] for row in expected]),
            rtol=3.0e-13,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(
        result.effective_screening_length_m,
        np.asarray([row["effective_screening_length"] for row in expected]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(result.applicable, np.ones(count, dtype=np.bool_))
    assert result.effective_ion_energy_V[3] == 0.01
    assert result.charge_rate_derivative_s_inv[4] < 0.0
    assert result.charge_rate_derivative_s_inv[5] < 0.0


def test_aggregate_charge_branch_is_continuous_and_relative_velocity_is_invariant() -> None:
    potential_per_charge = _potential_per_charge(5.0e-8, 2.0e-4)
    potential_delta = 1.0e-10
    branch = _uniform_evaluation(
        np.asarray(
            [
                -potential_delta / potential_per_charge,
                0.0,
                potential_delta / potential_per_charge,
            ]
        )
    )
    np.testing.assert_allclose(
        branch.charge_rate_number_s[[0, 2]],
        branch.charge_rate_number_s[1],
        rtol=0.0,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        branch.charge_rate_derivative_s_inv[[0, 2]],
        branch.charge_rate_derivative_s_inv[1],
        rtol=2.0e-10,
        atol=0.0,
    )

    particle_velocity = np.asarray([[10.0, 20.0], [33.0, 9.0], [0.0, 0.0], [0.0, 0.0]])
    ion_velocity = np.asarray([[110.0, -20.0], [133.0, -31.0], [40.0, 100.0], [-100.0, 40.0]])
    invariant = _evaluate(
        charge_number=np.full(4, -20.0),
        radius_m=np.full(4, 5.0e-8),
        electron_density_m3=np.full(4, 2.0e15),
        ion_density_m3=np.full(4, 1.5e15),
        electron_voltage_V=np.full(4, 3.0),
        ion_voltage_V=np.full(4, 0.03),
        particle_velocity_m_s=particle_velocity,
        ion_velocity_m_s=ion_velocity,
        ion_mass_kg=np.full(4, _ION_MASS_KG),
        screening_length_m=np.full(4, 2.0e-4),
        maximum_relative_speed_m_s=200.0,
    )
    for value in (
        invariant.charge_rate_number_s,
        invariant.charge_rate_derivative_s_inv,
        invariant.relative_ion_speed_m_s,
        invariant.effective_ion_speed_m_s,
        invariant.effective_ion_energy_V,
    ):
        np.testing.assert_allclose(value, value[0], rtol=3.0e-15, atol=0.0)


def test_aggregate_charge_is_monotone_and_preserves_common_density_scaling() -> None:
    charge = np.linspace(-6_000.0, 100.0, 257)
    result = _uniform_evaluation(charge)

    assert result.charge_rate_number_s[0] > 0.0
    assert result.charge_rate_number_s[-1] < 0.0
    assert bool((np.diff(result.charge_rate_number_s) < 0.0).all())
    assert bool((result.charge_rate_derivative_s_inv < 0.0).all())

    scale = 7.0
    scaled = _uniform_evaluation(
        np.asarray([-20.0, -20.0]),
        electron_density_m3=np.asarray([2.0e15, scale * 2.0e15]),
        ion_density_m3=np.asarray([1.5e15, scale * 1.5e15]),
    )
    np.testing.assert_allclose(
        scaled.charge_rate_number_s[1],
        scale * scaled.charge_rate_number_s[0],
        rtol=3.0e-15,
    )
    np.testing.assert_allclose(
        scaled.charge_rate_derivative_s_inv[1],
        scale * scaled.charge_rate_derivative_s_inv[0],
        rtol=3.0e-15,
    )


def test_aggregate_global_bounds_enclose_piecewise_primitive_candidates() -> None:
    radii = (4.0e-8, 7.0e-8)
    electron_density = (1.0e14, 3.0e15)
    ion_density = (8.0e13, 2.5e15)
    electron_voltage = (1.5, 4.0)
    ion_voltage = (1.0e-6, 0.08)
    ion_mass = (0.5 * _ION_MASS_KG, 2.0 * _ION_MASS_KG)
    screening_length = (1.0e-9, 3.0e-4)
    maximum_relative_speed = 300.0
    bounds: AggregateChargeBounds = aggregate_relative_drift_global_bounds(
        initial_charge_number=np.asarray([-20.0, 4.0]),
        electrostatic_radius_m=np.asarray(radii),
        electron_number_density_lower_m3=electron_density[0],
        electron_number_density_upper_m3=electron_density[1],
        positive_ion_number_density_lower_m3=ion_density[0],
        positive_ion_number_density_upper_m3=ion_density[1],
        electron_thermal_voltage_lower_V=electron_voltage[0],
        electron_thermal_voltage_upper_V=electron_voltage[1],
        positive_ion_thermal_voltage_lower_V=ion_voltage[0],
        positive_ion_thermal_voltage_upper_V=ion_voltage[1],
        effective_positive_ion_mass_lower_kg=ion_mass[0],
        effective_positive_ion_mass_upper_kg=ion_mass[1],
        screening_length_lower_m=screening_length[0],
        screening_length_upper_m=screening_length[1],
        maximum_relative_ion_speed_m_s=maximum_relative_speed,
    )
    assert bounds.charge_number_lower <= -20.0
    assert bounds.charge_number_upper >= 4.0
    assert bounds.maximum_relative_ion_speed_m_s == maximum_relative_speed

    rows: list[tuple[float, ...]] = []
    for values in itertools.product(
        radii,
        electron_density,
        ion_density,
        electron_voltage,
        ion_voltage,
        ion_mass,
        screening_length,
        (0.0, maximum_relative_speed),
    ):
        radius, _ne, _ni, te, ti, mass, screening, relative_speed = values
        potential_per_charge = _potential_per_charge(radius, screening)
        effective_energy = _effective_ion_energy_V(ti, mass, (relative_speed, 0.0))
        candidates = [
            bounds.charge_number_lower,
            0.0,
            bounds.charge_number_upper,
            -50.0 * te / potential_per_charge,
            50.0 * effective_energy / potential_per_charge,
        ]
        rows.extend(
            (*values, charge)
            for charge in candidates
            if bounds.charge_number_lower <= charge <= bounds.charge_number_upper
        )

    samples = np.asarray(rows)
    evaluation = _evaluate(
        charge_number=samples[:, 8],
        radius_m=samples[:, 0],
        electron_density_m3=samples[:, 1],
        ion_density_m3=samples[:, 2],
        electron_voltage_V=samples[:, 3],
        ion_voltage_V=samples[:, 4],
        particle_velocity_m_s=np.zeros((samples.shape[0], 2)),
        ion_velocity_m_s=np.column_stack((samples[:, 7], np.zeros(samples.shape[0]))),
        ion_mass_kg=samples[:, 5],
        screening_length_m=samples[:, 6],
        maximum_relative_speed_m_s=maximum_relative_speed,
    )
    lower = samples[:, 8] == bounds.charge_number_lower
    upper = samples[:, 8] == bounds.charge_number_upper
    roundoff = 2.0e-13 * bounds.charge_rate_abs_upper_number_s
    assert bool((evaluation.charge_rate_number_s[lower] >= -roundoff).all())
    assert bool((evaluation.charge_rate_number_s[upper] <= roundoff).all())
    assert bool(
        (np.abs(evaluation.charge_rate_number_s) <= bounds.charge_rate_abs_upper_number_s).all()
    )
    assert bool(
        (
            np.abs(evaluation.charge_rate_derivative_s_inv)
            <= bounds.charge_rate_derivative_abs_upper_s_inv
        ).all()
    )
    assert bool(
        (evaluation.effective_screening_length_m >= bounds.effective_screening_length_lower_m).all()
    )
    assert bool(
        (evaluation.effective_screening_length_m <= bounds.effective_screening_length_upper_m).all()
    )
    assert bool((evaluation.capacitance_F >= bounds.capacitance_lower_F).all())
    assert bool((evaluation.capacitance_F <= bounds.capacitance_upper_F).all())
    np.testing.assert_array_equal(
        evaluation.applicable,
        np.ones(samples.shape[0], dtype=np.bool_),
    )


def test_aggregate_charge_catalog_owns_fields_and_rejects_unresolved_ion_drag() -> None:
    plan = resolve_physics_plan({"charge": _charge_model()}, "axisymmetric_rz")

    assert plan.charge is not None
    assert plan.charge.revision == _REVISION
    assert {
        (field.name, field.unit, field.components, field.stored_basis)
        for field in plan.required_fields
    } == {
        ("electron_density", "1/m^3", ("value",), "scalar"),
        ("ion_density", "1/m^3", ("value",), "scalar"),
        ("electron_voltage", "V", ("value",), "scalar"),
        ("ion_voltage", "V", ("value",), "scalar"),
        ("ion_velocity", "m/s", ("r", "z"), "axisymmetric_rz"),
        ("ion_mass", "kg", ("value",), "scalar"),
        ("screening_length", "m", ("value",), "scalar"),
    }
    assert plan.resolved_models()["charge"] == {
        "model": "plasma_continuous",
        "revision": _REVISION,
    }

    missing_field = _charge_model()
    missing_field.pop("screening_length_field")
    with pytest.raises(ValueError, match="keys do not match"):
        resolve_physics_plan({"charge": missing_field}, "cartesian_xy")
    with pytest.raises(
        ValueError,
        match="aggregate charge revisions do not compose with barnes_collisionless",
    ):
        resolve_physics_plan(
            {"charge": _charge_model(), "ion_drag": _barnes_model()},
            "cartesian_xy",
        )


def test_aggregate_charge_compiled_runtime_matches_pure_rate_and_both_speed_gates() -> None:
    plan = resolve_physics_plan({"charge": _charge_model()}, "cartesian_xy")
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.full(4, 2.0e-15),
        drag_diameter_m=np.full(4, 2.0e-6),
        electrostatic_radius_m=np.asarray([4.0e-8, 5.0e-8, 7.0e-8, 6.0e-8]),
        displaced_volume_m3=np.zeros(4),
        charge_number=np.asarray([-100.0, -20.0, 0.0, 10.0]),
        primitive_ranges={
            "electron_density": _range((1.0e15,), (3.0e15,)),
            "ion_density": _range((8.0e14,), (2.0e15,)),
            "electron_voltage": _range((2.0,), (4.0,)),
            "ion_voltage": _range((0.02,), (0.05,)),
            "ion_velocity": _range((-100.0, -50.0), (100.0, 50.0)),
            "ion_mass": _range((0.8 * _ION_MASS_KG,), (1.2 * _ION_MASS_KG,)),
            "screening_length": _range((1.0e-4,), (3.0e-4,)),
        },
    )
    particle_velocity = np.asarray([[0.0, 0.0], [100.0, -40.0], [-50.0, 30.0], [-101.0, -151.0]])
    ion_velocity = np.asarray([[0.0, 0.0], [100.0, -40.0], [40.0, 30.0], [100.0, 50.0]])
    charge = np.asarray([-100.0, -20.0, 0.0, 10.0])
    sampled = {
        "electron_density": np.asarray([[1.0e15], [1.5e15], [2.0e15], [3.0e15]]),
        "ion_density": np.asarray([[8.0e14], [1.0e15], [1.5e15], [2.0e15]]),
        "electron_voltage": np.asarray([[2.0], [2.5], [3.0], [4.0]]),
        "ion_voltage": np.asarray([[0.02], [0.03], [0.04], [0.05]]),
        "ion_velocity": ion_velocity,
        "ion_mass": _ION_MASS_KG * np.asarray([[0.8], [0.9], [1.0], [1.2]]),
        "screening_length": np.asarray([[1.0e-4], [1.5e-4], [2.0e-4], [3.0e-4]]),
    }

    actual, status = runtime.evaluate_batch(
        np.arange(4, dtype=np.int64),
        particle_velocity,
        charge,
        sampled,
    )
    expected = aggregate_relative_drift_regularized_two_current_v1(
        charge_number=charge,
        electrostatic_radius_m=runtime.electrostatic_radius_m,
        electron_number_density_m3=sampled["electron_density"][:, 0],
        positive_ion_number_density_m3=sampled["ion_density"][:, 0],
        electron_thermal_voltage_V=sampled["electron_voltage"][:, 0],
        positive_ion_thermal_voltage_V=sampled["ion_voltage"][:, 0],
        particle_velocity_m_s=particle_velocity,
        positive_ion_velocity_m_s=ion_velocity,
        effective_positive_ion_mass_kg=sampled["ion_mass"][:, 0],
        screening_length_m=sampled["screening_length"][:, 0],
        maximum_relative_ion_speed_m_s=200.0,
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
    np.testing.assert_array_equal(actual.applicable, expected.applicable)
    np.testing.assert_array_equal(actual.applicable, [True, True, True, False])

    path_applicable, path_status = runtime.continuous_applicability_batch(
        np.asarray([0, 1]),
        np.asarray([[50.0, 10.0], [101.0, 151.0]]),
    )
    np.testing.assert_array_equal(path_status, NUMERICAL_STATUS_OK)
    np.testing.assert_array_equal(path_applicable, [True, False])
