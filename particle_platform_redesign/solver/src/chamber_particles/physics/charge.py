"""Pure particle charge-rate models over already sampled plasma primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .forces import (
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
    PhysicsEvaluationError,
)

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]

ELECTRON_MASS_KG = 9.1093837139e-31
OML_MAX_RADIUS_OVER_DEBYE = 0.1
OML_MAX_ION_DRIFT_RATIO = 0.1
AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S = 1.0
AGGREGATE_MINIMUM_ION_ENERGY_V = 0.01
AGGREGATE_EXPONENT_MIN = -50.0
AGGREGATE_EXPONENT_MAX = 50.0
_SMALL_SHIFT_RATIO = math.sqrt(np.finfo(np.float64).eps)
_BOUND_ROUNDOFF_FACTOR = 1.0 + 64.0 * np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class OmlChargeEvaluation:
    """Local OML rate, derivative, and explicit revision-1 applicability."""

    charge_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray
    surface_potential_V: FloatArray
    debye_length_m: FloatArray
    capacitance_F: FloatArray
    radius_over_debye: FloatArray
    ion_drift_ratio: FloatArray
    applicable: BoolArray


@dataclass(frozen=True, slots=True)
class ChargeNumberBracket:
    """Closed charge-number bracket containing the unique local equilibrium."""

    lower: FloatArray
    upper: FloatArray


@dataclass(frozen=True, slots=True)
class OmlChargeBounds:
    """Run-wide finite invariant and conservative OML rate bounds."""

    charge_number_lower: float
    charge_number_upper: float
    charge_rate_abs_upper_number_s: float
    charge_rate_derivative_abs_upper_s_inv: float
    debye_length_lower_m: float
    debye_length_upper_m: float
    capacitance_lower_F: float
    capacitance_upper_F: float


@dataclass(frozen=True, slots=True)
class AggregateChargeEvaluation:
    """Local rate and revision-owned quantities for aggregate charging."""

    charge_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray
    surface_potential_V: FloatArray
    effective_screening_length_m: FloatArray
    capacitance_F: FloatArray
    relative_ion_speed_m_s: FloatArray
    effective_ion_speed_m_s: FloatArray
    effective_ion_energy_V: FloatArray
    applicable: BoolArray


@dataclass(frozen=True, slots=True)
class AggregateThreeCurrentChargeEvaluation:
    """Two-current result extended by one aggregate negative-ion collection term."""

    charge_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray
    surface_potential_V: FloatArray
    effective_screening_length_m: FloatArray
    capacitance_F: FloatArray
    positive_relative_ion_speed_m_s: FloatArray
    positive_effective_ion_speed_m_s: FloatArray
    positive_effective_ion_energy_V: FloatArray
    negative_relative_ion_speed_m_s: FloatArray
    negative_effective_ion_speed_m_s: FloatArray
    negative_effective_ion_energy_V: FloatArray
    applicable: BoolArray


@dataclass(frozen=True, slots=True)
class AggregateChargeBounds:
    """Finite invariant and conservative aggregate-charge rate bounds."""

    charge_number_lower: float
    charge_number_upper: float
    charge_rate_abs_upper_number_s: float
    charge_rate_derivative_abs_upper_s_inv: float
    effective_screening_length_lower_m: float
    effective_screening_length_upper_m: float
    capacitance_lower_F: float
    capacitance_upper_F: float
    maximum_relative_ion_speed_m_s: float


@dataclass(frozen=True, slots=True)
class _OmlGlobalParameters:
    debye_length_lower_m: float
    debye_length_upper_m: float
    capacitance_lower_F: float
    capacitance_upper_F: float
    electron_voltage_lower_V: float
    electron_voltage_upper_V: float
    ion_voltage_lower_V: float
    ion_voltage_upper_V: float
    electron_amplitude_lower_number_s: float
    electron_amplitude_upper_number_s: float
    ion_amplitude_lower_number_s: float
    ion_amplitude_upper_number_s: float


@dataclass(frozen=True, slots=True)
class _AggregateGlobalParameters:
    initial_charge_lower: float
    initial_charge_upper: float
    radius_upper_m: float
    effective_screening_length_lower_m: float
    effective_screening_length_upper_m: float
    capacitance_lower_F: float
    capacitance_upper_F: float
    electron_voltage_lower_V: float
    electron_voltage_upper_V: float
    positive_ion_energy_lower_V: float
    positive_ion_energy_upper_V: float
    electron_amplitude_lower_number_s: float
    electron_amplitude_upper_number_s: float
    positive_ion_amplitude_lower_number_s: float
    positive_ion_amplitude_upper_number_s: float
    potential_per_charge_lower_V: float
    potential_per_charge_upper_V: float
    maximum_relative_ion_speed_m_s: float


def aggregate_relative_drift_regularized_two_current_v1(
    *,
    charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_thermal_voltage_V: FloatArray,
    positive_ion_thermal_voltage_V: FloatArray,
    particle_velocity_m_s: FloatArray,
    positive_ion_velocity_m_s: FloatArray,
    effective_positive_ion_mass_kg: FloatArray,
    screening_length_m: FloatArray,
    maximum_relative_ion_speed_m_s: float,
) -> AggregateChargeEvaluation:
    """Evaluate the regularized aggregate two-current charge revision."""

    count = _common_count(
        charge_number,
        electrostatic_radius_m,
        electron_number_density_m3,
        positive_ion_number_density_m3,
        electron_thermal_voltage_V,
        positive_ion_thermal_voltage_V,
        effective_positive_ion_mass_kg,
        screening_length_m,
    )
    _require_finite(charge_number, "charge_number")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_positive(electron_number_density_m3, "electron_number_density_m3")
    _require_positive(positive_ion_number_density_m3, "positive_ion_number_density_m3")
    _require_positive(electron_thermal_voltage_V, "electron_thermal_voltage_V")
    _require_positive(positive_ion_thermal_voltage_V, "positive_ion_thermal_voltage_V")
    _require_positive(effective_positive_ion_mass_kg, "effective_positive_ion_mass_kg")
    _require_positive(screening_length_m, "screening_length_m")
    maximum_relative_speed = _positive_scalar(
        maximum_relative_ion_speed_m_s,
        "maximum_relative_ion_speed_m_s",
    )
    particle_velocity = _velocity(particle_velocity_m_s, count, "particle_velocity_m_s")
    ion_velocity = _velocity(
        positive_ion_velocity_m_s,
        count,
        "positive_ion_velocity_m_s",
    )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        relative_speed = np.hypot(
            ion_velocity[:, 0] - particle_velocity[:, 0],
            ion_velocity[:, 1] - particle_velocity[:, 1],
        )
        effective_speed_squared = (
            relative_speed**2
            + 8.0
            * ELEMENTARY_CHARGE_C
            * positive_ion_thermal_voltage_V
            / (math.pi * effective_positive_ion_mass_kg)
            + AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
        )
        effective_speed = np.sqrt(effective_speed_squared)
        effective_ion_energy = np.maximum(
            effective_positive_ion_mass_kg * effective_speed_squared / (2.0 * ELEMENTARY_CHARGE_C),
            AGGREGATE_MINIMUM_ION_ENERGY_V,
        )
        effective_screening = np.maximum(electrostatic_radius_m, screening_length_m)
        capacitance = (
            4.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * electrostatic_radius_m
            * (1.0 + electrostatic_radius_m / effective_screening)
        )
        potential_per_charge = ELEMENTARY_CHARGE_C / capacitance
        surface_potential = charge_number * potential_per_charge
        ion_amplitude = (
            math.pi * electrostatic_radius_m**2 * positive_ion_number_density_m3 * effective_speed
        )
        electron_amplitude = (
            math.pi
            * electrostatic_radius_m**2
            * electron_number_density_m3
            * np.sqrt(
                8.0
                * ELEMENTARY_CHARGE_C
                * electron_thermal_voltage_V
                / (math.pi * ELECTRON_MASS_KG)
            )
        )

        nonpositive = surface_potential <= 0.0
        positive = ~nonpositive
        electron_argument = surface_potential / electron_thermal_voltage_V
        ion_argument = -surface_potential / effective_ion_energy
        electron_factor = np.empty(count, dtype=np.float64)
        ion_factor = np.empty(count, dtype=np.float64)
        electron_factor[nonpositive] = np.exp(
            np.clip(
                electron_argument[nonpositive],
                AGGREGATE_EXPONENT_MIN,
                AGGREGATE_EXPONENT_MAX,
            )
        )
        electron_factor[positive] = 1.0 + electron_argument[positive]
        ion_factor[nonpositive] = 1.0 + ion_argument[nonpositive]
        ion_factor[positive] = np.exp(
            np.clip(
                ion_argument[positive],
                AGGREGATE_EXPONENT_MIN,
                AGGREGATE_EXPONENT_MAX,
            )
        )
        ion_collection = ion_amplitude * ion_factor
        electron_collection = electron_amplitude * electron_factor
        charge_rate = ion_collection - electron_collection

        electron_unclipped = (electron_argument > AGGREGATE_EXPONENT_MIN) & (
            electron_argument < AGGREGATE_EXPONENT_MAX
        )
        ion_unclipped = (ion_argument > AGGREGATE_EXPONENT_MIN) & (
            ion_argument < AGGREGATE_EXPONENT_MAX
        )
        electron_factor_derivative = np.empty(count, dtype=np.float64)
        ion_factor_derivative = np.empty(count, dtype=np.float64)
        electron_factor_derivative[nonpositive] = np.where(
            electron_unclipped[nonpositive],
            electron_factor[nonpositive]
            * potential_per_charge[nonpositive]
            / electron_thermal_voltage_V[nonpositive],
            0.0,
        )
        electron_factor_derivative[positive] = (
            potential_per_charge[positive] / electron_thermal_voltage_V[positive]
        )
        ion_factor_derivative[nonpositive] = (
            -potential_per_charge[nonpositive] / effective_ion_energy[nonpositive]
        )
        ion_factor_derivative[positive] = np.where(
            ion_unclipped[positive],
            -ion_factor[positive] * potential_per_charge[positive] / effective_ion_energy[positive],
            0.0,
        )
        rate_derivative = (
            ion_amplitude * ion_factor_derivative - electron_amplitude * electron_factor_derivative
        )

    _require_aggregate_derived_finite(
        relative_speed,
        effective_speed,
        effective_ion_energy,
        effective_screening,
        capacitance,
        surface_potential,
        ion_collection,
        electron_collection,
        charge_rate,
        rate_derivative,
    )
    if not bool((ion_collection >= 0.0).all() and (electron_collection >= 0.0).all()):
        raise PhysicsEvaluationError("aggregate collection rate is negative")
    if not bool((rate_derivative < 0.0).all()):
        raise PhysicsEvaluationError("aggregate charge-rate derivative is not strictly negative")

    return AggregateChargeEvaluation(
        charge_rate,
        rate_derivative,
        surface_potential,
        effective_screening,
        capacitance,
        relative_speed,
        effective_speed,
        effective_ion_energy,
        relative_speed <= maximum_relative_speed,
    )


def aggregate_relative_drift_regularized_three_current_v1(
    *,
    charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    negative_ion_number_density_m3: FloatArray,
    electron_thermal_voltage_V: FloatArray,
    positive_ion_thermal_voltage_V: FloatArray,
    negative_ion_thermal_voltage_V: FloatArray,
    particle_velocity_m_s: FloatArray,
    positive_ion_velocity_m_s: FloatArray,
    negative_ion_velocity_m_s: FloatArray,
    effective_positive_ion_mass_kg: FloatArray,
    effective_negative_ion_mass_kg: FloatArray,
    screening_length_m: FloatArray,
    maximum_relative_ion_speed_m_s: float,
) -> AggregateThreeCurrentChargeEvaluation:
    """Add one aggregate singly-negative-ion collection term to two-current charge.

    This is not a species-resolved closure.  ``screening_length_m`` remains
    the explicit external field authority used by the base revision; the
    negative-ion contribution does not recompute screening internally.
    """

    base = aggregate_relative_drift_regularized_two_current_v1(
        charge_number=charge_number,
        electrostatic_radius_m=electrostatic_radius_m,
        electron_number_density_m3=electron_number_density_m3,
        positive_ion_number_density_m3=positive_ion_number_density_m3,
        electron_thermal_voltage_V=electron_thermal_voltage_V,
        positive_ion_thermal_voltage_V=positive_ion_thermal_voltage_V,
        particle_velocity_m_s=particle_velocity_m_s,
        positive_ion_velocity_m_s=positive_ion_velocity_m_s,
        effective_positive_ion_mass_kg=effective_positive_ion_mass_kg,
        screening_length_m=screening_length_m,
        maximum_relative_ion_speed_m_s=maximum_relative_ion_speed_m_s,
    )
    count = _common_count(
        charge_number,
        negative_ion_number_density_m3,
        negative_ion_thermal_voltage_V,
        effective_negative_ion_mass_kg,
    )
    _require_nonnegative(negative_ion_number_density_m3, "negative_ion_number_density_m3")
    _require_positive(negative_ion_thermal_voltage_V, "negative_ion_thermal_voltage_V")
    _require_positive(effective_negative_ion_mass_kg, "effective_negative_ion_mass_kg")
    particle_velocity = _velocity(particle_velocity_m_s, count, "particle_velocity_m_s")
    negative_ion_velocity = _velocity(
        negative_ion_velocity_m_s,
        count,
        "negative_ion_velocity_m_s",
    )
    maximum_relative_speed = _positive_scalar(
        maximum_relative_ion_speed_m_s,
        "maximum_relative_ion_speed_m_s",
    )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        relative_speed = np.hypot(
            negative_ion_velocity[:, 0] - particle_velocity[:, 0],
            negative_ion_velocity[:, 1] - particle_velocity[:, 1],
        )
        effective_speed_squared = (
            relative_speed**2
            + 8.0
            * ELEMENTARY_CHARGE_C
            * negative_ion_thermal_voltage_V
            / (math.pi * effective_negative_ion_mass_kg)
            + AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
        )
        effective_speed = np.sqrt(effective_speed_squared)
        effective_energy = np.maximum(
            effective_negative_ion_mass_kg * effective_speed_squared / (2.0 * ELEMENTARY_CHARGE_C),
            AGGREGATE_MINIMUM_ION_ENERGY_V,
        )
        potential_per_charge = ELEMENTARY_CHARGE_C / base.capacitance_F
        amplitude = (
            math.pi * electrostatic_radius_m**2 * negative_ion_number_density_m3 * effective_speed
        )
        nonpositive = base.surface_potential_V <= 0.0
        argument = base.surface_potential_V / effective_energy
        factor = np.empty(count, dtype=np.float64)
        factor[nonpositive] = np.exp(
            np.clip(
                argument[nonpositive],
                AGGREGATE_EXPONENT_MIN,
                AGGREGATE_EXPONENT_MAX,
            )
        )
        factor[~nonpositive] = 1.0 + argument[~nonpositive]
        unclipped = (argument > AGGREGATE_EXPONENT_MIN) & (argument < AGGREGATE_EXPONENT_MAX)
        factor_derivative = np.empty(count, dtype=np.float64)
        factor_derivative[nonpositive] = np.where(
            unclipped[nonpositive],
            factor[nonpositive] * potential_per_charge[nonpositive] / effective_energy[nonpositive],
            0.0,
        )
        factor_derivative[~nonpositive] = (
            potential_per_charge[~nonpositive] / effective_energy[~nonpositive]
        )
        negative_collection = amplitude * factor
        negative_collection_derivative = amplitude * factor_derivative
        charge_rate = base.charge_rate_number_s - negative_collection
        rate_derivative = base.charge_rate_derivative_s_inv - negative_collection_derivative

    _require_aggregate_derived_finite(
        relative_speed,
        effective_speed,
        effective_energy,
        amplitude,
        factor,
        negative_collection,
        charge_rate,
        rate_derivative,
    )
    if not bool((negative_collection >= 0.0).all()):
        raise PhysicsEvaluationError("aggregate negative-ion collection rate is negative")
    if not bool((rate_derivative < 0.0).all()):
        raise PhysicsEvaluationError("aggregate charge-rate derivative is not strictly negative")

    return AggregateThreeCurrentChargeEvaluation(
        charge_rate,
        rate_derivative,
        base.surface_potential_V,
        base.effective_screening_length_m,
        base.capacitance_F,
        base.relative_ion_speed_m_s,
        base.effective_ion_speed_m_s,
        base.effective_ion_energy_V,
        relative_speed,
        effective_speed,
        effective_energy,
        base.applicable & (relative_speed <= maximum_relative_speed),
    )


def aggregate_relative_drift_global_bounds(
    *,
    initial_charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_lower_m3: float,
    electron_number_density_upper_m3: float,
    positive_ion_number_density_lower_m3: float,
    positive_ion_number_density_upper_m3: float,
    electron_thermal_voltage_lower_V: float,
    electron_thermal_voltage_upper_V: float,
    positive_ion_thermal_voltage_lower_V: float,
    positive_ion_thermal_voltage_upper_V: float,
    effective_positive_ion_mass_lower_kg: float,
    effective_positive_ion_mass_upper_kg: float,
    screening_length_lower_m: float,
    screening_length_upper_m: float,
    maximum_relative_ion_speed_m_s: float,
) -> AggregateChargeBounds:
    """Build a finite invariant from primitive ranges and a speed envelope."""

    parameters = _aggregate_global_parameters(
        initial_charge_number=initial_charge_number,
        electrostatic_radius_m=electrostatic_radius_m,
        electron_number_density_lower_m3=electron_number_density_lower_m3,
        electron_number_density_upper_m3=electron_number_density_upper_m3,
        positive_ion_number_density_lower_m3=positive_ion_number_density_lower_m3,
        positive_ion_number_density_upper_m3=positive_ion_number_density_upper_m3,
        electron_thermal_voltage_lower_V=electron_thermal_voltage_lower_V,
        electron_thermal_voltage_upper_V=electron_thermal_voltage_upper_V,
        positive_ion_thermal_voltage_lower_V=positive_ion_thermal_voltage_lower_V,
        positive_ion_thermal_voltage_upper_V=positive_ion_thermal_voltage_upper_V,
        effective_positive_ion_mass_lower_kg=effective_positive_ion_mass_lower_kg,
        effective_positive_ion_mass_upper_kg=effective_positive_ion_mass_upper_kg,
        screening_length_lower_m=screening_length_lower_m,
        screening_length_upper_m=screening_length_upper_m,
        maximum_relative_ion_speed_m_s=maximum_relative_ion_speed_m_s,
    )
    return _aggregate_two_current_bounds(parameters)


def aggregate_relative_drift_three_current_global_bounds(
    *,
    initial_charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_lower_m3: float,
    electron_number_density_upper_m3: float,
    positive_ion_number_density_lower_m3: float,
    positive_ion_number_density_upper_m3: float,
    negative_ion_number_density_lower_m3: float,
    negative_ion_number_density_upper_m3: float,
    electron_thermal_voltage_lower_V: float,
    electron_thermal_voltage_upper_V: float,
    positive_ion_thermal_voltage_lower_V: float,
    positive_ion_thermal_voltage_upper_V: float,
    negative_ion_thermal_voltage_lower_V: float,
    negative_ion_thermal_voltage_upper_V: float,
    effective_positive_ion_mass_lower_kg: float,
    effective_positive_ion_mass_upper_kg: float,
    effective_negative_ion_mass_lower_kg: float,
    effective_negative_ion_mass_upper_kg: float,
    screening_length_lower_m: float,
    screening_length_upper_m: float,
    maximum_relative_ion_speed_m_s: float,
) -> AggregateChargeBounds:
    """Extend aggregate bounds by one negative-ion collection term."""

    parameters = _aggregate_global_parameters(
        initial_charge_number=initial_charge_number,
        electrostatic_radius_m=electrostatic_radius_m,
        electron_number_density_lower_m3=electron_number_density_lower_m3,
        electron_number_density_upper_m3=electron_number_density_upper_m3,
        positive_ion_number_density_lower_m3=positive_ion_number_density_lower_m3,
        positive_ion_number_density_upper_m3=positive_ion_number_density_upper_m3,
        electron_thermal_voltage_lower_V=electron_thermal_voltage_lower_V,
        electron_thermal_voltage_upper_V=electron_thermal_voltage_upper_V,
        positive_ion_thermal_voltage_lower_V=positive_ion_thermal_voltage_lower_V,
        positive_ion_thermal_voltage_upper_V=positive_ion_thermal_voltage_upper_V,
        effective_positive_ion_mass_lower_kg=effective_positive_ion_mass_lower_kg,
        effective_positive_ion_mass_upper_kg=effective_positive_ion_mass_upper_kg,
        screening_length_lower_m=screening_length_lower_m,
        screening_length_upper_m=screening_length_upper_m,
        maximum_relative_ion_speed_m_s=maximum_relative_ion_speed_m_s,
    )
    _, negative_density_upper = _nonnegative_range(
        negative_ion_number_density_lower_m3,
        negative_ion_number_density_upper_m3,
        "negative_ion_number_density",
    )
    negative_voltage_lower, negative_voltage_upper = _positive_range(
        negative_ion_thermal_voltage_lower_V,
        negative_ion_thermal_voltage_upper_V,
        "negative_ion_thermal_voltage",
    )
    negative_mass_lower, _ = _positive_range(
        effective_negative_ion_mass_lower_kg,
        effective_negative_ion_mass_upper_kg,
        "effective_negative_ion_mass",
    )
    base = _aggregate_two_current_bounds(parameters)
    if negative_density_upper == 0.0:
        return base

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        negative_speed_upper = math.sqrt(
            parameters.maximum_relative_ion_speed_m_s**2
            + 8.0 * ELEMENTARY_CHARGE_C * negative_voltage_upper / (math.pi * negative_mass_lower)
            + AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
        )
        negative_energy_lower = max(
            negative_mass_lower
            * AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
            / (2.0 * ELEMENTARY_CHARGE_C)
            + 4.0 * negative_voltage_lower / math.pi,
            AGGREGATE_MINIMUM_ION_ENERGY_V,
        )
        negative_amplitude_upper = (
            math.pi * parameters.radius_upper_m**2 * negative_density_upper * negative_speed_upper
        )
    negative_amplitude_upper = _outward_nonnegative_upper(
        negative_amplitude_upper,
        "negative-ion collection-amplitude upper bound",
    )
    negative_energy_lower = _outward_positive_lower(
        negative_energy_lower,
        "negative-ion energy lower bound",
    )
    negative_ratio_upper = math.nextafter(
        (parameters.electron_amplitude_upper_number_s + negative_amplitude_upper)
        / parameters.positive_ion_amplitude_lower_number_s,
        math.inf,
    )
    negative_ratio_excess = _outward_nonnegative_upper(
        max(negative_ratio_upper - 1.0, 0.0),
        "three-current negative equilibrium ratio excess",
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        equilibrium_charge_lower = (
            -parameters.positive_ion_energy_upper_V
            * negative_ratio_excess
            / parameters.potential_per_charge_lower_V
        )
    charge_lower = _outward_nonpositive_lower(
        min(base.charge_number_lower, equilibrium_charge_lower),
        "three-current charge invariant lower bound",
    )
    charge_upper = base.charge_number_upper

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        negative_potential_abs_upper = -charge_lower * parameters.potential_per_charge_upper_V
        positive_potential_upper = charge_upper * parameters.potential_per_charge_upper_V
        positive_collection_upper = parameters.positive_ion_amplitude_upper_number_s * (
            1.0 + negative_potential_abs_upper / parameters.positive_ion_energy_lower_V
        )
        electron_collection_upper = parameters.electron_amplitude_upper_number_s * (
            1.0 + positive_potential_upper / parameters.electron_voltage_lower_V
        )
        negative_collection_upper = negative_amplitude_upper * (
            1.0 + positive_potential_upper / negative_energy_lower
        )
        rate_abs_upper = max(
            positive_collection_upper,
            electron_collection_upper + negative_collection_upper,
        )
        derivative_abs_upper = parameters.potential_per_charge_upper_V * (
            parameters.positive_ion_amplitude_upper_number_s
            / parameters.positive_ion_energy_lower_V
            + parameters.electron_amplitude_upper_number_s / parameters.electron_voltage_lower_V
            + negative_amplitude_upper / negative_energy_lower
        )
    return AggregateChargeBounds(
        charge_lower,
        charge_upper,
        _outward_nonnegative_upper(rate_abs_upper, "three-current charge-rate bound"),
        _outward_nonnegative_upper(
            derivative_abs_upper,
            "three-current charge-rate derivative bound",
        ),
        parameters.effective_screening_length_lower_m,
        parameters.effective_screening_length_upper_m,
        parameters.capacitance_lower_F,
        parameters.capacitance_upper_F,
        parameters.maximum_relative_ion_speed_m_s,
    )


def oml_stationary_maxwellian_debye_huckel_v1(
    *,
    charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    positive_ion_temperature_K: FloatArray,
    particle_velocity_m_s: FloatArray,
    positive_ion_velocity_m_s: FloatArray,
    positive_ion_mass_kg: float,
) -> OmlChargeEvaluation:
    """Evaluate the stationary-Maxwellian absorbing-sphere OML revision.

    The ion velocity is used only for the declared stationary-Maxwellian
    applicability gate; it does not alter either collection rate.
    """

    count = _common_count(
        charge_number,
        electrostatic_radius_m,
        electron_number_density_m3,
        positive_ion_number_density_m3,
        electron_temperature_K,
        positive_ion_temperature_K,
    )
    _require_finite(charge_number, "charge_number")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_positive(electron_number_density_m3, "electron_number_density_m3")
    _require_positive(positive_ion_number_density_m3, "positive_ion_number_density_m3")
    _require_positive(electron_temperature_K, "electron_temperature_K")
    _require_positive(positive_ion_temperature_K, "positive_ion_temperature_K")
    ion_mass = _positive_scalar(positive_ion_mass_kg, "positive_ion_mass_kg")
    particle_velocity = _velocity(particle_velocity_m_s, count, "particle_velocity_m_s")
    ion_velocity = _velocity(positive_ion_velocity_m_s, count, "positive_ion_velocity_m_s")

    debye_length = oml_debye_length_m(
        electron_number_density_m3=electron_number_density_m3,
        positive_ion_number_density_m3=positive_ion_number_density_m3,
        electron_temperature_K=electron_temperature_K,
        positive_ion_temperature_K=positive_ion_temperature_K,
    )
    capacitance = debye_huckel_capacitance_F(
        electrostatic_radius_m=electrostatic_radius_m,
        debye_length_m=debye_length,
    )
    electron_amplitude, ion_amplitude, electron_voltage, ion_voltage = _oml_rates(
        electrostatic_radius_m=electrostatic_radius_m,
        electron_number_density_m3=electron_number_density_m3,
        positive_ion_number_density_m3=positive_ion_number_density_m3,
        electron_temperature_K=electron_temperature_K,
        positive_ion_temperature_K=positive_ion_temperature_K,
        positive_ion_mass_kg=ion_mass,
    )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
        electron_collection = np.empty(count, dtype=np.float64)
        ion_collection = np.empty(count, dtype=np.float64)
        rate_derivative = np.empty(count, dtype=np.float64)
        nonpositive = surface_potential <= 0.0
        positive = ~nonpositive

        electron_collection[nonpositive] = electron_amplitude[nonpositive] * np.exp(
            surface_potential[nonpositive] / electron_voltage[nonpositive]
        )
        ion_collection[nonpositive] = ion_amplitude[nonpositive] * (
            1.0 - surface_potential[nonpositive] / ion_voltage[nonpositive]
        )
        electron_collection[positive] = electron_amplitude[positive] * (
            1.0 + surface_potential[positive] / electron_voltage[positive]
        )
        ion_collection[positive] = ion_amplitude[positive] * np.exp(
            -surface_potential[positive] / ion_voltage[positive]
        )
        charge_rate = ion_collection - electron_collection

        potential_per_charge = ELEMENTARY_CHARGE_C / capacitance
        rate_derivative[nonpositive] = -potential_per_charge[nonpositive] * (
            ion_amplitude[nonpositive] / ion_voltage[nonpositive]
            + electron_collection[nonpositive] / electron_voltage[nonpositive]
        )
        rate_derivative[positive] = -potential_per_charge[positive] * (
            ion_collection[positive] / ion_voltage[positive]
            + electron_amplitude[positive] / electron_voltage[positive]
        )

        radius_over_debye = electrostatic_radius_m / debye_length
        ion_mean_thermal_speed = np.sqrt(
            8.0 * BOLTZMANN_J_K * positive_ion_temperature_K / (math.pi * ion_mass)
        )
        ion_relative_speed = np.hypot(
            ion_velocity[:, 0] - particle_velocity[:, 0],
            ion_velocity[:, 1] - particle_velocity[:, 1],
        )
        ion_drift_ratio = ion_relative_speed / ion_mean_thermal_speed

    _require_derived_finite(
        surface_potential,
        electron_collection,
        ion_collection,
        charge_rate,
        rate_derivative,
        radius_over_debye,
        ion_drift_ratio,
    )
    if not bool((electron_collection >= 0.0).all() and (ion_collection >= 0.0).all()):
        raise PhysicsEvaluationError("OML collection rate is negative")
    if not bool((rate_derivative < 0.0).all()):
        raise PhysicsEvaluationError("OML charge-rate derivative is not strictly negative")

    applicable = (radius_over_debye <= OML_MAX_RADIUS_OVER_DEBYE) & (
        ion_drift_ratio <= OML_MAX_ION_DRIFT_RATIO
    )
    return OmlChargeEvaluation(
        charge_rate,
        rate_derivative,
        surface_potential,
        debye_length,
        capacitance,
        radius_over_debye,
        ion_drift_ratio,
        applicable,
    )


def oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1(
    *,
    charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    positive_ion_temperature_K: FloatArray,
    particle_velocity_m_s: FloatArray,
    positive_ion_velocity_m_s: FloatArray,
    positive_ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
) -> OmlChargeEvaluation:
    """Evaluate negative-grain OML with one shifted-Maxwellian ion species.

    Electrons are stationary Maxwellian and ions are a single, singly charged
    shifted Maxwellian.  This revision deliberately covers only nonpositive
    surface potential; emission, negative ions, and positive-grain branches
    require separate model revisions.
    """

    count = _common_count(
        charge_number,
        electrostatic_radius_m,
        electron_number_density_m3,
        positive_ion_number_density_m3,
        electron_temperature_K,
        positive_ion_temperature_K,
    )
    _require_finite(charge_number, "charge_number")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_positive(electron_number_density_m3, "electron_number_density_m3")
    _require_positive(positive_ion_number_density_m3, "positive_ion_number_density_m3")
    _require_positive(electron_temperature_K, "electron_temperature_K")
    _require_positive(positive_ion_temperature_K, "positive_ion_temperature_K")
    ion_mass = _positive_scalar(positive_ion_mass_kg, "positive_ion_mass_kg")
    maximum_drift = _positive_scalar(maximum_ion_drift_ratio, "maximum_ion_drift_ratio")
    particle_velocity = _velocity(particle_velocity_m_s, count, "particle_velocity_m_s")
    ion_velocity = _velocity(positive_ion_velocity_m_s, count, "positive_ion_velocity_m_s")

    debye_length = oml_debye_length_m(
        electron_number_density_m3=electron_number_density_m3,
        positive_ion_number_density_m3=positive_ion_number_density_m3,
        electron_temperature_K=electron_temperature_K,
        positive_ion_temperature_K=positive_ion_temperature_K,
    )
    capacitance = debye_huckel_capacitance_F(
        electrostatic_radius_m=electrostatic_radius_m,
        debye_length_m=debye_length,
    )
    electron_amplitude, ion_amplitude, electron_voltage, ion_voltage = _oml_rates(
        electrostatic_radius_m=electrostatic_radius_m,
        electron_number_density_m3=electron_number_density_m3,
        positive_ion_number_density_m3=positive_ion_number_density_m3,
        electron_temperature_K=electron_temperature_K,
        positive_ion_temperature_K=positive_ion_temperature_K,
        positive_ion_mass_kg=ion_mass,
    )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
        ion_mean_thermal_speed = np.sqrt(
            8.0 * BOLTZMANN_J_K * positive_ion_temperature_K / (math.pi * ion_mass)
        )
        ion_relative_speed = np.hypot(
            ion_velocity[:, 0] - particle_velocity[:, 0],
            ion_velocity[:, 1] - particle_velocity[:, 1],
        )
        ion_drift_ratio = ion_relative_speed / ion_mean_thermal_speed
        neutral_factor, attraction_factor = shifted_maxwellian_ion_factors(
            ion_drift_ratio * math.sqrt(8.0 / math.pi)
        )
        electron_collection = electron_amplitude * np.exp(surface_potential / electron_voltage)
        ion_collection = ion_amplitude * (
            neutral_factor - surface_potential / ion_voltage * attraction_factor
        )
        charge_rate = ion_collection - electron_collection
        potential_per_charge = ELEMENTARY_CHARGE_C / capacitance
        rate_derivative = -potential_per_charge * (
            ion_amplitude * attraction_factor / ion_voltage + electron_collection / electron_voltage
        )
        radius_over_debye = electrostatic_radius_m / debye_length

    _require_derived_finite(
        surface_potential,
        electron_collection,
        ion_collection,
        charge_rate,
        rate_derivative,
        radius_over_debye,
        ion_drift_ratio,
    )
    if not bool((surface_potential <= 0.0).all()):
        raise PhysicsEvaluationError(
            "shifted-Maxwellian OML revision requires nonpositive surface potential"
        )
    if not bool((electron_collection >= 0.0).all() and (ion_collection >= 0.0).all()):
        raise PhysicsEvaluationError("OML collection rate is negative")
    if not bool((rate_derivative < 0.0).all()):
        raise PhysicsEvaluationError("OML charge-rate derivative is not strictly negative")

    nonpositive_equilibrium = ion_amplitude * neutral_factor <= electron_amplitude
    applicable = (
        (radius_over_debye <= OML_MAX_RADIUS_OVER_DEBYE)
        & (ion_drift_ratio <= maximum_drift)
        & nonpositive_equilibrium
    )
    return OmlChargeEvaluation(
        charge_rate,
        rate_derivative,
        surface_potential,
        debye_length,
        capacitance,
        radius_over_debye,
        ion_drift_ratio,
        applicable,
    )


def shifted_maxwellian_ion_factors(
    thermal_shift_ratio: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return neutral-flux and attractive-potential SOML factors.

    The ratio is ``|u_i-v| / sqrt(k_B T_i / m_i)``.  The analytic zero-drift
    limits are evaluated with their series so the model reduces continuously
    to stationary OML without a physical speed floor.
    """

    ratio = np.asarray(thermal_shift_ratio, dtype=np.float64)
    if ratio.ndim != 1 or not bool(np.isfinite(ratio).all() and (ratio >= 0.0).all()):
        raise PhysicsEvaluationError("thermal_shift_ratio must be finite, nonnegative, and 1-D")
    neutral = np.empty_like(ratio)
    attraction = np.empty_like(ratio)
    for index, drift in enumerate(ratio):
        neutral[index], attraction[index] = _shifted_maxwellian_ion_factors_scalar(float(drift))
    _require_positive_derived(neutral, "shifted-Maxwellian neutral ion factor")
    _require_positive_derived(attraction, "shifted-Maxwellian attraction factor")
    return neutral, attraction


def _shifted_maxwellian_ion_factors_scalar(thermal_shift_ratio: float) -> tuple[float, float]:
    if thermal_shift_ratio <= _SMALL_SHIFT_RATIO:
        square = thermal_shift_ratio * thermal_shift_ratio
        fourth = square * square
        return 1.0 + square / 6.0 - fourth / 120.0, 1.0 - square / 6.0 + fourth / 40.0
    error_function = math.erf(thermal_shift_ratio / math.sqrt(2.0))
    neutral = (
        0.5 * math.exp(-0.5 * thermal_shift_ratio * thermal_shift_ratio)
        + math.sqrt(math.pi / 8.0)
        * (thermal_shift_ratio + 1.0 / thermal_shift_ratio)
        * error_function
    )
    attraction = math.sqrt(math.pi / 2.0) * error_function / thermal_shift_ratio
    return neutral, attraction


def oml_debye_length_m(
    *,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    positive_ion_temperature_K: FloatArray,
) -> FloatArray:
    """Return the two-species Debye length in metres."""

    _common_count(
        electron_number_density_m3,
        positive_ion_number_density_m3,
        electron_temperature_K,
        positive_ion_temperature_K,
    )
    _require_positive(electron_number_density_m3, "electron_number_density_m3")
    _require_positive(positive_ion_number_density_m3, "positive_ion_number_density_m3")
    _require_positive(electron_temperature_K, "electron_temperature_K")
    _require_positive(positive_ion_temperature_K, "positive_ion_temperature_K")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        inverse_square = (
            ELEMENTARY_CHARGE_C**2
            / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
            * (
                electron_number_density_m3 / electron_temperature_K
                + positive_ion_number_density_m3 / positive_ion_temperature_K
            )
        )
        result = 1.0 / np.sqrt(inverse_square)
    _require_positive_derived(result, "Debye length")
    return result


def debye_huckel_capacitance_F(
    *,
    electrostatic_radius_m: FloatArray,
    debye_length_m: FloatArray,
) -> FloatArray:
    """Return ``4*pi*eps0*a*(1+a/lambda_D)`` in farads."""

    _common_count(electrostatic_radius_m, debye_length_m)
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_positive(debye_length_m, "debye_length_m")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (
            4.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * electrostatic_radius_m
            * (1.0 + electrostatic_radius_m / debye_length_m)
        )
    _require_positive_derived(result, "Debye-Huckel capacitance")
    return result


def oml_local_equilibrium_bracket(
    *,
    electrostatic_radius_m: FloatArray,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    positive_ion_temperature_K: FloatArray,
    positive_ion_mass_kg: float,
) -> ChargeNumberBracket:
    """Return a closed bracket for each unique stationary OML equilibrium."""

    _common_count(
        electrostatic_radius_m,
        electron_number_density_m3,
        positive_ion_number_density_m3,
        electron_temperature_K,
        positive_ion_temperature_K,
    )
    debye_length = oml_debye_length_m(
        electron_number_density_m3=electron_number_density_m3,
        positive_ion_number_density_m3=positive_ion_number_density_m3,
        electron_temperature_K=electron_temperature_K,
        positive_ion_temperature_K=positive_ion_temperature_K,
    )
    capacitance = debye_huckel_capacitance_F(
        electrostatic_radius_m=electrostatic_radius_m,
        debye_length_m=debye_length,
    )
    electron_amplitude, ion_amplitude, electron_voltage, ion_voltage = _oml_rates(
        electrostatic_radius_m=electrostatic_radius_m,
        electron_number_density_m3=electron_number_density_m3,
        positive_ion_number_density_m3=positive_ion_number_density_m3,
        electron_temperature_K=electron_temperature_K,
        positive_ion_temperature_K=positive_ion_temperature_K,
        positive_ion_mass_kg=_positive_scalar(
            positive_ion_mass_kg,
            "positive_ion_mass_kg",
        ),
    )

    lower = np.zeros(electrostatic_radius_m.shape, dtype=np.float64)
    upper = np.zeros(electrostatic_radius_m.shape, dtype=np.float64)
    negative_equilibrium = ion_amplitude < electron_amplitude
    positive_equilibrium = ion_amplitude > electron_amplitude
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower[negative_equilibrium] = (
            capacitance[negative_equilibrium]
            * ion_voltage[negative_equilibrium]
            * (1.0 - electron_amplitude[negative_equilibrium] / ion_amplitude[negative_equilibrium])
            / ELEMENTARY_CHARGE_C
        )
        upper[positive_equilibrium] = (
            capacitance[positive_equilibrium]
            * electron_voltage[positive_equilibrium]
            * (ion_amplitude[positive_equilibrium] / electron_amplitude[positive_equilibrium] - 1.0)
            / ELEMENTARY_CHARGE_C
        )
    _require_derived_finite(lower, upper)
    if not bool((lower <= 0.0).all() and (upper >= 0.0).all()):
        raise PhysicsEvaluationError("OML equilibrium bracket has an invalid sign")
    return ChargeNumberBracket(lower, upper)


def oml_stationary_global_bounds(
    *,
    initial_charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_lower_m3: float,
    electron_number_density_upper_m3: float,
    positive_ion_number_density_lower_m3: float,
    positive_ion_number_density_upper_m3: float,
    electron_temperature_lower_K: float,
    electron_temperature_upper_K: float,
    positive_ion_temperature_lower_K: float,
    positive_ion_temperature_upper_K: float,
    positive_ion_mass_kg: float,
) -> OmlChargeBounds:
    """Build a finite run-wide invariant and conservative rate bounds."""

    initial_charge = np.asarray(initial_charge_number, dtype=np.float64)
    radius = np.asarray(electrostatic_radius_m, dtype=np.float64)
    if initial_charge.ndim != 1 or radius.shape != initial_charge.shape or initial_charge.size == 0:
        raise PhysicsEvaluationError(
            "initial_charge_number and electrostatic_radius_m must be nonempty matching arrays"
        )
    _require_finite(initial_charge, "initial_charge_number")
    _require_positive(radius, "electrostatic_radius_m")
    parameters = _oml_global_parameters(
        electrostatic_radius_m=radius,
        electron_number_density_lower_m3=electron_number_density_lower_m3,
        electron_number_density_upper_m3=electron_number_density_upper_m3,
        positive_ion_number_density_lower_m3=positive_ion_number_density_lower_m3,
        positive_ion_number_density_upper_m3=positive_ion_number_density_upper_m3,
        electron_temperature_lower_K=electron_temperature_lower_K,
        electron_temperature_upper_K=electron_temperature_upper_K,
        positive_ion_temperature_lower_K=positive_ion_temperature_lower_K,
        positive_ion_temperature_upper_K=positive_ion_temperature_upper_K,
        positive_ion_mass_kg=positive_ion_mass_kg,
    )
    debye_lower = parameters.debye_length_lower_m
    debye_upper = parameters.debye_length_upper_m
    capacitance_lower = parameters.capacitance_lower_F
    capacitance_upper = parameters.capacitance_upper_F
    electron_voltage_lower = parameters.electron_voltage_lower_V
    electron_voltage_upper = parameters.electron_voltage_upper_V
    ion_voltage_lower = parameters.ion_voltage_lower_V
    ion_voltage_upper = parameters.ion_voltage_upper_V
    electron_amplitude_lower = parameters.electron_amplitude_lower_number_s
    electron_amplitude_upper = parameters.electron_amplitude_upper_number_s
    ion_amplitude_lower = parameters.ion_amplitude_lower_number_s
    ion_amplitude_upper = parameters.ion_amplitude_upper_number_s

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        potential_lower = -ion_voltage_upper * max(
            electron_amplitude_upper / ion_amplitude_lower - 1.0,
            0.0,
        )
        potential_upper = electron_voltage_upper * max(
            ion_amplitude_upper / electron_amplitude_lower - 1.0,
            0.0,
        )
        equilibrium_charge_lower = capacitance_upper * potential_lower / ELEMENTARY_CHARGE_C
        equilibrium_charge_upper = capacitance_upper * potential_upper / ELEMENTARY_CHARGE_C
    charge_lower = min(float(np.min(initial_charge)), equilibrium_charge_lower)
    charge_upper = max(float(np.max(initial_charge)), equilibrium_charge_upper)
    charge_lower = _outward_nonpositive_lower(charge_lower, "charge invariant lower bound")
    charge_upper = _outward_nonnegative_upper(charge_upper, "charge invariant upper bound")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        negative_potential_abs_upper = -charge_lower * ELEMENTARY_CHARGE_C / capacitance_lower
        positive_potential_upper = charge_upper * ELEMENTARY_CHARGE_C / capacitance_lower
        ion_collection_upper = ion_amplitude_upper * (
            1.0 + negative_potential_abs_upper / ion_voltage_lower
        )
        electron_collection_upper = electron_amplitude_upper * (
            1.0 + positive_potential_upper / electron_voltage_lower
        )
        rate_abs_upper = max(ion_collection_upper, electron_collection_upper)
        derivative_abs_upper = (
            ELEMENTARY_CHARGE_C
            / capacitance_lower
            * (
                ion_amplitude_upper / ion_voltage_lower
                + electron_amplitude_upper / electron_voltage_lower
            )
        )
    rate_abs_upper = _outward_nonnegative_upper(rate_abs_upper, "charge-rate bound")
    derivative_abs_upper = _outward_nonnegative_upper(
        derivative_abs_upper,
        "charge-rate derivative bound",
    )
    return OmlChargeBounds(
        charge_lower,
        charge_upper,
        rate_abs_upper,
        derivative_abs_upper,
        debye_lower,
        debye_upper,
        capacitance_lower,
        capacitance_upper,
    )


def oml_shifted_maxwellian_global_bounds(
    *,
    initial_charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_lower_m3: float,
    electron_number_density_upper_m3: float,
    positive_ion_number_density_lower_m3: float,
    positive_ion_number_density_upper_m3: float,
    electron_temperature_lower_K: float,
    electron_temperature_upper_K: float,
    positive_ion_temperature_lower_K: float,
    positive_ion_temperature_upper_K: float,
    positive_ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
) -> OmlChargeBounds:
    """Build finite bounds for the nonpositive shifted-Maxwellian revision."""

    initial_charge = np.asarray(initial_charge_number, dtype=np.float64)
    radius = np.asarray(electrostatic_radius_m, dtype=np.float64)
    if initial_charge.ndim != 1 or radius.shape != initial_charge.shape or initial_charge.size == 0:
        raise PhysicsEvaluationError(
            "initial_charge_number and electrostatic_radius_m must be nonempty matching arrays"
        )
    _require_finite(initial_charge, "initial_charge_number")
    _require_positive(radius, "electrostatic_radius_m")
    if bool((initial_charge > 0.0).any()):
        raise PhysicsEvaluationError(
            "shifted-Maxwellian OML revision requires nonpositive initial charge_number"
        )
    maximum_drift = _positive_scalar(maximum_ion_drift_ratio, "maximum_ion_drift_ratio")
    parameters = _oml_global_parameters(
        electrostatic_radius_m=radius,
        electron_number_density_lower_m3=electron_number_density_lower_m3,
        electron_number_density_upper_m3=electron_number_density_upper_m3,
        positive_ion_number_density_lower_m3=positive_ion_number_density_lower_m3,
        positive_ion_number_density_upper_m3=positive_ion_number_density_upper_m3,
        electron_temperature_lower_K=electron_temperature_lower_K,
        electron_temperature_upper_K=electron_temperature_upper_K,
        positive_ion_temperature_lower_K=positive_ion_temperature_lower_K,
        positive_ion_temperature_upper_K=positive_ion_temperature_upper_K,
        positive_ion_mass_kg=positive_ion_mass_kg,
    )
    electron_density_lower, electron_density_upper = _positive_range(
        electron_number_density_lower_m3,
        electron_number_density_upper_m3,
        "electron_number_density",
    )
    ion_density_lower, ion_density_upper = _positive_range(
        positive_ion_number_density_lower_m3,
        positive_ion_number_density_upper_m3,
        "positive_ion_number_density",
    )
    electron_temperature_lower, electron_temperature_upper = _positive_range(
        electron_temperature_lower_K,
        electron_temperature_upper_K,
        "electron_temperature",
    )
    ion_temperature_lower, ion_temperature_upper = _positive_range(
        positive_ion_temperature_lower_K,
        positive_ion_temperature_upper_K,
        "positive_ion_temperature",
    )
    ion_mass = _positive_scalar(positive_ion_mass_kg, "positive_ion_mass_kg")
    electron_flux_lower = _outward_positive_lower(
        electron_density_lower
        * math.sqrt(
            8.0 * BOLTZMANN_J_K * electron_temperature_lower / (math.pi * ELECTRON_MASS_KG)
        ),
        "electron flux-coefficient lower bound",
    )
    electron_flux_upper = _outward_positive_upper(
        electron_density_upper
        * math.sqrt(
            8.0 * BOLTZMANN_J_K * electron_temperature_upper / (math.pi * ELECTRON_MASS_KG)
        ),
        "electron flux-coefficient upper bound",
    )
    ion_flux_lower = _outward_positive_lower(
        ion_density_lower
        * math.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_lower / (math.pi * ion_mass)),
        "ion flux-coefficient lower bound",
    )
    ion_flux_upper = _outward_positive_upper(
        ion_density_upper
        * math.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_upper / (math.pi * ion_mass)),
        "ion flux-coefficient upper bound",
    )
    maximum_shift = maximum_drift * math.sqrt(8.0 / math.pi)
    neutral_max, attraction_min = _shifted_maxwellian_ion_factors_scalar(maximum_shift)
    if not (
        math.isfinite(neutral_max)
        and neutral_max > 0.0
        and math.isfinite(attraction_min)
        and attraction_min > 0.0
    ):
        raise PhysicsEvaluationError("shifted-Maxwellian drift envelope is not finite")
    zero_ion_flux_upper = ion_flux_upper * neutral_max
    if zero_ion_flux_upper > electron_flux_lower:
        raise PhysicsEvaluationError(
            "shifted-Maxwellian OML cannot certify a nonpositive equilibrium over the "
            "declared primitive and drift ranges"
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        potential_abs_lower_bracket = (
            parameters.ion_voltage_upper_V
            / attraction_min
            * max(
                electron_flux_upper / ion_flux_lower - 1.0,
                0.0,
            )
        )
        equilibrium_charge_lower = (
            -parameters.capacitance_upper_F * potential_abs_lower_bracket / ELEMENTARY_CHARGE_C
        )
    charge_lower = _outward_nonpositive_lower(
        min(float(np.min(initial_charge)), equilibrium_charge_lower),
        "charge invariant lower bound",
    )
    charge_upper = 0.0

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        potential_abs_upper = -charge_lower * ELEMENTARY_CHARGE_C / parameters.capacitance_lower_F
        ion_collection_upper = parameters.ion_amplitude_upper_number_s * (
            neutral_max + potential_abs_upper / parameters.ion_voltage_lower_V
        )
        electron_collection_upper = parameters.electron_amplitude_upper_number_s
        rate_abs_upper = max(ion_collection_upper, electron_collection_upper)
        derivative_abs_upper = (
            ELEMENTARY_CHARGE_C
            / parameters.capacitance_lower_F
            * (
                parameters.ion_amplitude_upper_number_s / parameters.ion_voltage_lower_V
                + parameters.electron_amplitude_upper_number_s / parameters.electron_voltage_lower_V
            )
        )
    rate_abs_upper = _outward_nonnegative_upper(rate_abs_upper, "charge-rate bound")
    derivative_abs_upper = _outward_nonnegative_upper(
        derivative_abs_upper,
        "charge-rate derivative bound",
    )
    return OmlChargeBounds(
        charge_lower,
        charge_upper,
        rate_abs_upper,
        derivative_abs_upper,
        parameters.debye_length_lower_m,
        parameters.debye_length_upper_m,
        parameters.capacitance_lower_F,
        parameters.capacitance_upper_F,
    )


def _oml_global_parameters(
    *,
    electrostatic_radius_m: FloatArray,
    electron_number_density_lower_m3: float,
    electron_number_density_upper_m3: float,
    positive_ion_number_density_lower_m3: float,
    positive_ion_number_density_upper_m3: float,
    electron_temperature_lower_K: float,
    electron_temperature_upper_K: float,
    positive_ion_temperature_lower_K: float,
    positive_ion_temperature_upper_K: float,
    positive_ion_mass_kg: float,
) -> _OmlGlobalParameters:
    electron_density_lower, electron_density_upper = _positive_range(
        electron_number_density_lower_m3,
        electron_number_density_upper_m3,
        "electron_number_density",
    )
    ion_density_lower, ion_density_upper = _positive_range(
        positive_ion_number_density_lower_m3,
        positive_ion_number_density_upper_m3,
        "positive_ion_number_density",
    )
    electron_temperature_lower, electron_temperature_upper = _positive_range(
        electron_temperature_lower_K,
        electron_temperature_upper_K,
        "electron_temperature",
    )
    ion_temperature_lower, ion_temperature_upper = _positive_range(
        positive_ion_temperature_lower_K,
        positive_ion_temperature_upper_K,
        "positive_ion_temperature",
    )
    ion_mass = _positive_scalar(positive_ion_mass_kg, "positive_ion_mass_kg")
    radius_lower = float(np.min(electrostatic_radius_m))
    radius_upper = float(np.max(electrostatic_radius_m))
    debye_lower = _outward_positive_lower(
        float(
            oml_debye_length_m(
                electron_number_density_m3=np.asarray([electron_density_upper]),
                positive_ion_number_density_m3=np.asarray([ion_density_upper]),
                electron_temperature_K=np.asarray([electron_temperature_lower]),
                positive_ion_temperature_K=np.asarray([ion_temperature_lower]),
            )[0]
        ),
        "Debye-length lower bound",
    )
    debye_upper = _outward_positive_upper(
        float(
            oml_debye_length_m(
                electron_number_density_m3=np.asarray([electron_density_lower]),
                positive_ion_number_density_m3=np.asarray([ion_density_lower]),
                electron_temperature_K=np.asarray([electron_temperature_upper]),
                positive_ion_temperature_K=np.asarray([ion_temperature_upper]),
            )[0]
        ),
        "Debye-length upper bound",
    )
    capacitance_lower = _outward_positive_lower(
        float(
            debye_huckel_capacitance_F(
                electrostatic_radius_m=np.asarray([radius_lower]),
                debye_length_m=np.asarray([debye_upper]),
            )[0]
        ),
        "capacitance lower bound",
    )
    capacitance_upper = _outward_positive_upper(
        float(
            debye_huckel_capacitance_F(
                electrostatic_radius_m=np.asarray([radius_upper]),
                debye_length_m=np.asarray([debye_lower]),
            )[0]
        ),
        "capacitance upper bound",
    )
    lower_rates = _oml_rates(
        electrostatic_radius_m=np.asarray([radius_lower]),
        electron_number_density_m3=np.asarray([electron_density_lower]),
        positive_ion_number_density_m3=np.asarray([ion_density_lower]),
        electron_temperature_K=np.asarray([electron_temperature_lower]),
        positive_ion_temperature_K=np.asarray([ion_temperature_lower]),
        positive_ion_mass_kg=ion_mass,
    )
    upper_rates = _oml_rates(
        electrostatic_radius_m=np.asarray([radius_upper]),
        electron_number_density_m3=np.asarray([electron_density_upper]),
        positive_ion_number_density_m3=np.asarray([ion_density_upper]),
        electron_temperature_K=np.asarray([electron_temperature_upper]),
        positive_ion_temperature_K=np.asarray([ion_temperature_upper]),
        positive_ion_mass_kg=ion_mass,
    )
    return _OmlGlobalParameters(
        debye_lower,
        debye_upper,
        capacitance_lower,
        capacitance_upper,
        _outward_positive_lower(
            BOLTZMANN_J_K * electron_temperature_lower / ELEMENTARY_CHARGE_C,
            "electron-voltage lower bound",
        ),
        _outward_positive_upper(
            BOLTZMANN_J_K * electron_temperature_upper / ELEMENTARY_CHARGE_C,
            "electron-voltage upper bound",
        ),
        _outward_positive_lower(
            BOLTZMANN_J_K * ion_temperature_lower / ELEMENTARY_CHARGE_C,
            "ion-voltage lower bound",
        ),
        _outward_positive_upper(
            BOLTZMANN_J_K * ion_temperature_upper / ELEMENTARY_CHARGE_C,
            "ion-voltage upper bound",
        ),
        _outward_positive_lower(float(lower_rates[0][0]), "electron-amplitude lower bound"),
        _outward_positive_upper(float(upper_rates[0][0]), "electron-amplitude upper bound"),
        _outward_positive_lower(float(lower_rates[1][0]), "ion-amplitude lower bound"),
        _outward_positive_upper(float(upper_rates[1][0]), "ion-amplitude upper bound"),
    )


def _oml_rates(
    *,
    electrostatic_radius_m: FloatArray,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    positive_ion_temperature_K: FloatArray,
    positive_ion_mass_kg: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    _common_count(
        electrostatic_radius_m,
        electron_number_density_m3,
        positive_ion_number_density_m3,
        electron_temperature_K,
        positive_ion_temperature_K,
    )
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_positive(electron_number_density_m3, "electron_number_density_m3")
    _require_positive(positive_ion_number_density_m3, "positive_ion_number_density_m3")
    _require_positive(electron_temperature_K, "electron_temperature_K")
    _require_positive(positive_ion_temperature_K, "positive_ion_temperature_K")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        electron_voltage = BOLTZMANN_J_K * electron_temperature_K / ELEMENTARY_CHARGE_C
        ion_voltage = BOLTZMANN_J_K * positive_ion_temperature_K / ELEMENTARY_CHARGE_C
        electron_amplitude = (
            math.pi
            * electrostatic_radius_m**2
            * electron_number_density_m3
            * np.sqrt(8.0 * BOLTZMANN_J_K * electron_temperature_K / (math.pi * ELECTRON_MASS_KG))
        )
        ion_amplitude = (
            math.pi
            * electrostatic_radius_m**2
            * positive_ion_number_density_m3
            * np.sqrt(
                8.0 * BOLTZMANN_J_K * positive_ion_temperature_K / (math.pi * positive_ion_mass_kg)
            )
        )
    _require_positive_derived(electron_voltage, "electron thermal voltage")
    _require_positive_derived(ion_voltage, "ion thermal voltage")
    _require_positive_derived(electron_amplitude, "electron collection amplitude")
    _require_positive_derived(ion_amplitude, "ion collection amplitude")
    return electron_amplitude, ion_amplitude, electron_voltage, ion_voltage


def _aggregate_global_parameters(
    *,
    initial_charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    electron_number_density_lower_m3: float,
    electron_number_density_upper_m3: float,
    positive_ion_number_density_lower_m3: float,
    positive_ion_number_density_upper_m3: float,
    electron_thermal_voltage_lower_V: float,
    electron_thermal_voltage_upper_V: float,
    positive_ion_thermal_voltage_lower_V: float,
    positive_ion_thermal_voltage_upper_V: float,
    effective_positive_ion_mass_lower_kg: float,
    effective_positive_ion_mass_upper_kg: float,
    screening_length_lower_m: float,
    screening_length_upper_m: float,
    maximum_relative_ion_speed_m_s: float,
) -> _AggregateGlobalParameters:
    """Resolve the shared primitive envelope for aggregate charge revisions."""

    initial_charge = np.asarray(initial_charge_number, dtype=np.float64)
    radius = np.asarray(electrostatic_radius_m, dtype=np.float64)
    if initial_charge.ndim != 1 or radius.shape != initial_charge.shape or initial_charge.size == 0:
        raise PhysicsEvaluationError(
            "initial_charge_number and electrostatic_radius_m must be nonempty matching arrays"
        )
    _require_finite(initial_charge, "initial_charge_number")
    _require_positive(radius, "electrostatic_radius_m")
    electron_density_lower, electron_density_upper = _positive_range(
        electron_number_density_lower_m3,
        electron_number_density_upper_m3,
        "electron_number_density",
    )
    ion_density_lower, ion_density_upper = _positive_range(
        positive_ion_number_density_lower_m3,
        positive_ion_number_density_upper_m3,
        "positive_ion_number_density",
    )
    electron_voltage_lower, electron_voltage_upper = _positive_range(
        electron_thermal_voltage_lower_V,
        electron_thermal_voltage_upper_V,
        "electron_thermal_voltage",
    )
    ion_voltage_lower, ion_voltage_upper = _positive_range(
        positive_ion_thermal_voltage_lower_V,
        positive_ion_thermal_voltage_upper_V,
        "positive_ion_thermal_voltage",
    )
    ion_mass_lower, ion_mass_upper = _positive_range(
        effective_positive_ion_mass_lower_kg,
        effective_positive_ion_mass_upper_kg,
        "effective_positive_ion_mass",
    )
    screening_lower, screening_upper = _positive_range(
        screening_length_lower_m,
        screening_length_upper_m,
        "screening_length",
    )
    maximum_relative_speed = _positive_scalar(
        maximum_relative_ion_speed_m_s,
        "maximum_relative_ion_speed_m_s",
    )
    radius_lower = float(np.min(radius))
    radius_upper = float(np.max(radius))
    effective_screening_lower = _outward_positive_lower(
        max(radius_lower, screening_lower),
        "effective screening-length lower bound",
    )
    effective_screening_upper = _outward_positive_upper(
        max(radius_upper, screening_upper),
        "effective screening-length upper bound",
    )
    capacitance_lower = _outward_positive_lower(
        _aggregate_capacitance_scalar(radius_lower, screening_upper),
        "aggregate capacitance lower bound",
    )
    capacitance_upper = _outward_positive_upper(
        _aggregate_capacitance_scalar(radius_upper, screening_lower),
        "aggregate capacitance upper bound",
    )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        effective_speed_lower = math.sqrt(
            8.0 * ELEMENTARY_CHARGE_C * ion_voltage_lower / (math.pi * ion_mass_upper)
            + AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
        )
        effective_speed_upper = math.sqrt(
            maximum_relative_speed**2
            + 8.0 * ELEMENTARY_CHARGE_C * ion_voltage_upper / (math.pi * ion_mass_lower)
            + AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
        )
        ion_energy_lower = max(
            ion_mass_lower
            * AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
            / (2.0 * ELEMENTARY_CHARGE_C)
            + 4.0 * ion_voltage_lower / math.pi,
            AGGREGATE_MINIMUM_ION_ENERGY_V,
        )
        ion_energy_upper = max(
            ion_mass_upper
            * (maximum_relative_speed**2 + AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2)
            / (2.0 * ELEMENTARY_CHARGE_C)
            + 4.0 * ion_voltage_upper / math.pi,
            AGGREGATE_MINIMUM_ION_ENERGY_V,
        )
        electron_amplitude_lower = (
            math.pi
            * radius_lower**2
            * electron_density_lower
            * math.sqrt(
                8.0 * ELEMENTARY_CHARGE_C * electron_voltage_lower / (math.pi * ELECTRON_MASS_KG)
            )
        )
        electron_amplitude_upper = (
            math.pi
            * radius_upper**2
            * electron_density_upper
            * math.sqrt(
                8.0 * ELEMENTARY_CHARGE_C * electron_voltage_upper / (math.pi * ELECTRON_MASS_KG)
            )
        )
        ion_amplitude_lower = math.pi * radius_lower**2 * ion_density_lower * effective_speed_lower
        ion_amplitude_upper = math.pi * radius_upper**2 * ion_density_upper * effective_speed_upper
    ion_energy_lower = _outward_positive_lower(
        ion_energy_lower,
        "effective ion-energy lower bound",
    )
    ion_energy_upper = _outward_positive_upper(
        ion_energy_upper,
        "effective ion-energy upper bound",
    )
    electron_amplitude_lower = _outward_positive_lower(
        electron_amplitude_lower,
        "electron collection-amplitude lower bound",
    )
    electron_amplitude_upper = _outward_positive_upper(
        electron_amplitude_upper,
        "electron collection-amplitude upper bound",
    )
    ion_amplitude_lower = _outward_positive_lower(
        ion_amplitude_lower,
        "ion collection-amplitude lower bound",
    )
    ion_amplitude_upper = _outward_positive_upper(
        ion_amplitude_upper,
        "ion collection-amplitude upper bound",
    )
    potential_per_charge_lower = _outward_positive_lower(
        ELEMENTARY_CHARGE_C / capacitance_upper,
        "potential-per-charge lower bound",
    )
    potential_per_charge_upper = _outward_positive_upper(
        ELEMENTARY_CHARGE_C / capacitance_lower,
        "potential-per-charge upper bound",
    )
    return _AggregateGlobalParameters(
        float(np.min(initial_charge)),
        float(np.max(initial_charge)),
        radius_upper,
        effective_screening_lower,
        effective_screening_upper,
        capacitance_lower,
        capacitance_upper,
        electron_voltage_lower,
        electron_voltage_upper,
        ion_energy_lower,
        ion_energy_upper,
        electron_amplitude_lower,
        electron_amplitude_upper,
        ion_amplitude_lower,
        ion_amplitude_upper,
        potential_per_charge_lower,
        potential_per_charge_upper,
        maximum_relative_speed,
    )


def _aggregate_two_current_bounds(
    parameters: _AggregateGlobalParameters,
) -> AggregateChargeBounds:
    negative_ratio_upper = math.nextafter(
        parameters.electron_amplitude_upper_number_s
        / parameters.positive_ion_amplitude_lower_number_s,
        math.inf,
    )
    positive_ratio_upper = math.nextafter(
        parameters.positive_ion_amplitude_upper_number_s
        / parameters.electron_amplitude_lower_number_s,
        math.inf,
    )
    negative_ratio_excess = _outward_nonnegative_upper(
        max(negative_ratio_upper - 1.0, 0.0),
        "negative equilibrium ratio excess",
    )
    positive_ratio_excess = _outward_nonnegative_upper(
        max(positive_ratio_upper - 1.0, 0.0),
        "positive equilibrium ratio excess",
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        equilibrium_charge_lower = (
            -parameters.positive_ion_energy_upper_V
            * negative_ratio_excess
            / parameters.potential_per_charge_lower_V
        )
        equilibrium_charge_upper = (
            parameters.electron_voltage_upper_V
            * positive_ratio_excess
            / parameters.potential_per_charge_lower_V
        )
    charge_lower = _outward_nonpositive_lower(
        min(parameters.initial_charge_lower, equilibrium_charge_lower),
        "aggregate charge invariant lower bound",
    )
    charge_upper = _outward_nonnegative_upper(
        max(parameters.initial_charge_upper, equilibrium_charge_upper),
        "aggregate charge invariant upper bound",
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        negative_potential_abs_upper = -charge_lower * parameters.potential_per_charge_upper_V
        positive_potential_upper = charge_upper * parameters.potential_per_charge_upper_V
        ion_collection_upper = parameters.positive_ion_amplitude_upper_number_s * (
            1.0 + negative_potential_abs_upper / parameters.positive_ion_energy_lower_V
        )
        electron_collection_upper = parameters.electron_amplitude_upper_number_s * (
            1.0 + positive_potential_upper / parameters.electron_voltage_lower_V
        )
        rate_abs_upper = max(ion_collection_upper, electron_collection_upper)
        derivative_abs_upper = parameters.potential_per_charge_upper_V * (
            parameters.positive_ion_amplitude_upper_number_s
            / parameters.positive_ion_energy_lower_V
            + parameters.electron_amplitude_upper_number_s / parameters.electron_voltage_lower_V
        )
    return AggregateChargeBounds(
        charge_lower,
        charge_upper,
        _outward_nonnegative_upper(rate_abs_upper, "aggregate charge-rate bound"),
        _outward_nonnegative_upper(
            derivative_abs_upper,
            "aggregate charge-rate derivative bound",
        ),
        parameters.effective_screening_length_lower_m,
        parameters.effective_screening_length_upper_m,
        parameters.capacitance_lower_F,
        parameters.capacitance_upper_F,
        parameters.maximum_relative_ion_speed_m_s,
    )


def _aggregate_capacitance_scalar(radius_m: float, screening_length_m: float) -> float:
    effective_screening = max(radius_m, screening_length_m)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        capacitance = (
            4.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * radius_m
            * (1.0 + radius_m / effective_screening)
        )
    return _positive_scalar(capacitance, "aggregate capacitance")


def _common_count(first: FloatArray, *rest: FloatArray) -> int:
    if first.ndim != 1:
        raise PhysicsEvaluationError("particle scalar inputs must be one-dimensional")
    count = int(first.size)
    if any(array.shape != (count,) for array in rest):
        raise PhysicsEvaluationError("particle scalar inputs must have matching shape [N]")
    return count


def _velocity(value: FloatArray, count: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count, 2):
        raise PhysicsEvaluationError(f"{name} must have shape [N, 2]")
    _require_finite(result, name)
    return result


def _require_positive(value: FloatArray, name: str) -> None:
    _require_finite(value, name)
    if bool((value <= 0.0).any()):
        raise PhysicsEvaluationError(f"{name} must be positive")


def _require_nonnegative(value: FloatArray, name: str) -> None:
    _require_finite(value, name)
    if bool((value < 0.0).any()):
        raise PhysicsEvaluationError(f"{name} must be nonnegative")


def _require_finite(value: FloatArray, name: str) -> None:
    if not bool(np.isfinite(value).all()):
        raise PhysicsEvaluationError(f"{name} must be finite")


def _require_derived_finite(*values: FloatArray) -> None:
    if not all(bool(np.isfinite(value).all()) for value in values):
        raise PhysicsEvaluationError("OML evaluation produced a non-finite derived value")


def _require_aggregate_derived_finite(*values: FloatArray) -> None:
    if not all(bool(np.isfinite(value).all()) for value in values):
        raise PhysicsEvaluationError(
            "aggregate charge evaluation produced a non-finite derived value"
        )


def _require_positive_derived(value: FloatArray, name: str) -> None:
    if not bool(np.isfinite(value).all() and (value > 0.0).all()):
        raise PhysicsEvaluationError(f"{name} is not finite and positive")


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise PhysicsEvaluationError(f"{name} must be positive and finite")
    return float(value)


def _positive_range(lower: float, upper: float, name: str) -> tuple[float, float]:
    lower_value = _positive_scalar(lower, f"{name} lower bound")
    upper_value = _positive_scalar(upper, f"{name} upper bound")
    if lower_value > upper_value:
        raise PhysicsEvaluationError(f"{name} bounds are reversed")
    return lower_value, upper_value


def _nonnegative_range(lower: float, upper: float, name: str) -> tuple[float, float]:
    if any(isinstance(value, bool) or not math.isfinite(value) for value in (lower, upper)):
        raise PhysicsEvaluationError(f"{name} bounds must be finite")
    lower_value = float(lower)
    upper_value = float(upper)
    if lower_value < 0.0 or lower_value > upper_value:
        raise PhysicsEvaluationError(f"{name} bounds must be nonnegative and ordered")
    return lower_value, upper_value


def _outward_positive_lower(value: float, name: str) -> float:
    result = math.nextafter(value / _BOUND_ROUNDOFF_FACTOR, 0.0)
    return _positive_scalar(result, name)


def _outward_positive_upper(value: float, name: str) -> float:
    result = math.nextafter(value * _BOUND_ROUNDOFF_FACTOR, math.inf)
    return _positive_scalar(result, name)


def _outward_nonpositive_lower(value: float, name: str) -> float:
    if not math.isfinite(value) or value > 0.0:
        raise PhysicsEvaluationError(f"{name} must be finite and nonpositive")
    if value == 0.0:
        return 0.0
    result = math.nextafter(value * _BOUND_ROUNDOFF_FACTOR, -math.inf)
    if not math.isfinite(result):
        raise PhysicsEvaluationError(f"{name} is not finite")
    return result


def _outward_nonnegative_upper(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise PhysicsEvaluationError(f"{name} must be finite and nonnegative")
    if value == 0.0:
        return 0.0
    result = math.nextafter(value * _BOUND_ROUNDOFF_FACTOR, math.inf)
    if not math.isfinite(result):
        raise PhysicsEvaluationError(f"{name} is not finite")
    return result
