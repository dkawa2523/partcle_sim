"""Numba particle loop for the resolved deterministic physics catalog."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .charge import (
    _SMALL_SHIFT_RATIO,
    AGGREGATE_EXPONENT_MAX,
    AGGREGATE_EXPONENT_MIN,
    AGGREGATE_MINIMUM_ION_ENERGY_V,
    AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S,
    ELECTRON_MASS_KG,
    OML_MAX_ION_DRIFT_RATIO,
    OML_MAX_RADIUS_OVER_DEBYE,
)
from .forces import (
    _ALLEN_RAABE_A1,
    _ALLEN_RAABE_A2,
    _ALLEN_RAABE_A3,
    AGGREGATE_ION_DRAG_CHARGE_SQUARE_REGULARIZATION,
    AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S,
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    EPSTEIN_FINITE_SPEED_SERIES_LIMIT,
    EPSTEIN_MIN_LAMBDA_OVER_RADIUS,
    IMAGE_ION_DRAG_ELECTRIC_FIELD_SQUARE_FLOOR_V2_M2,
    IMAGE_ION_DRAG_LOG_ARGUMENT_OFFSET,
    ION_DRAG_MAX_SCALE_OVER_DEBYE,
    ION_DRAG_MIN_MEAN_FREE_PATH_OVER_DEBYE,
    RAREFIED_VORTICITY_LIFT_MIN_MEAN_FREE_PATH_OVER_RADIUS,
    SAFFMAN_LIFT_COEFFICIENT,
    SAFFMAN_MAX_MEAN_FREE_PATH_OVER_RADIUS,
    SAFFMAN_MAX_SHEAR_REYNOLDS,
    SAFFMAN_MAX_SLIP_REYNOLDS,
    SAFFMAN_MAX_SLIP_TO_SQRT_SHEAR_REYNOLDS,
    STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS,
    STOKES_CUNNINGHAM_MAX_REYNOLDS,
    STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS,
    VACUUM_PERMITTIVITY_F_M,
    WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS,
)

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type UInt8Array = NDArray[np.uint8]

DRAG_NONE = 0
DRAG_EPSTEIN = 1
DRAG_STOKES_CUNNINGHAM = 2
DRAG_EPSTEIN_FINITE_SPEED = 3

CHARGE_FIXED = 0
CHARGE_OML_STATIONARY_MAXWELLIAN_DEBYE_HUCKEL = 1
CHARGE_OML_SHIFTED_MAXWELLIAN_SINGLE_ION_NEGATIVE_DEBYE_HUCKEL = 2
CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_TWO_CURRENT = 3
CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT = 4

ION_DRAG_NONE = 0
ION_DRAG_BARNES_COLLISIONLESS_EFFECTIVE_SPEED = 1
ION_DRAG_RELATIVE_FLOW_SCREENED = 2
ION_DRAG_ELECTRIC_FIELD_DIRECTED_IMAGE = 3

THERMOPHORESIS_NONE = 0
THERMOPHORESIS_WALDMANN_GALLIS = 1
THERMOPHORESIS_TALBOT = 2

LIFT_NONE = 0
LIFT_RAREFIED_VORTICITY_RZ = 1
LIFT_SAFFMAN_XY = 2
LIFT_SAFFMAN_RZ = 3

ERROR_NONE = 0
ERROR_RATE = 1
ERROR_DERIVED_VALUE = 2
ERROR_ACCELERATION = 3
ERROR_CHARGE_INVARIANT = 4


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def evaluate_physics_tile_into(
    particle_index: Int64Array,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    electrostatic_radius_m: FloatArray,
    displaced_volume_m3: FloatArray,
    charge_code: int,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    positive_ion_temperature_K: FloatArray,
    positive_ion_velocity_m_s: FloatArray,
    positive_ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
    aggregate_electron_thermal_voltage_V: FloatArray,
    aggregate_positive_ion_thermal_voltage_V: FloatArray,
    aggregate_effective_positive_ion_mass_kg: FloatArray,
    aggregate_negative_ion_number_density_m3: FloatArray,
    aggregate_negative_ion_thermal_voltage_V: FloatArray,
    aggregate_negative_ion_velocity_m_s: FloatArray,
    aggregate_effective_negative_ion_mass_kg: FloatArray,
    aggregate_screening_length_m: FloatArray,
    aggregate_maximum_relative_ion_speed_m_s: float,
    charge_number_lower: float,
    charge_number_upper: float,
    drag_code: int,
    gas_velocity_m_s: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_temperature_K: FloatArray,
    gas_dynamic_viscosity_Pa_s: FloatArray,
    gas_mean_free_path_m: FloatArray,
    gas_molecular_mass_kg: float,
    epstein_delta: float,
    epstein_diffuse_reflection_fraction: float,
    epstein_maximum_speed_ratio: float,
    thermophoresis_code: int,
    thermophoresis_gas_velocity_m_s: FloatArray,
    thermophoresis_gas_temperature_K: FloatArray,
    thermophoresis_gas_translational_heat_flux_W_m2: FloatArray,
    thermophoresis_gas_mean_free_path_m: FloatArray,
    thermophoresis_gas_molecular_mass_kg: float,
    thermophoresis_maximum_speed_ratio: float,
    thermophoresis_gas_temperature_gradient_K_m: FloatArray,
    thermophoresis_gas_density_kg_m3: FloatArray,
    thermophoresis_gas_dynamic_viscosity_Pa_s: FloatArray,
    thermophoresis_gas_thermal_conductivity_W_m_K: FloatArray,
    thermophoresis_particle_thermal_conductivity_W_m_K: float,
    thermophoresis_thermal_slip_coefficient: float,
    thermophoresis_momentum_exchange_coefficient: float,
    thermophoresis_thermal_exchange_coefficient: float,
    ion_drag_code: int,
    ion_drag_electron_number_density_m3: FloatArray,
    ion_drag_positive_ion_number_density_m3: FloatArray,
    ion_drag_electron_temperature_K: FloatArray,
    ion_drag_positive_ion_temperature_K: FloatArray,
    ion_drag_positive_ion_velocity_m_s: FloatArray,
    ion_neutral_mean_free_path_m: FloatArray,
    ion_drag_positive_ion_mass_kg: float,
    ion_drag_maximum_ion_drift_ratio: float,
    ion_drag_electron_thermal_voltage_V: FloatArray,
    ion_drag_positive_ion_thermal_voltage_V: FloatArray,
    ion_drag_effective_positive_ion_mass_kg: FloatArray,
    ion_drag_screening_length_m: FloatArray,
    ion_drag_electric_field_V_m: FloatArray,
    ion_drag_maximum_relative_ion_speed_m_s: float,
    dielectrophoresis_enabled: bool,
    gradient_mean_e_squared_V2_m3: FloatArray,
    dep_medium_relative_permittivity: float,
    dep_real_clausius_mossotti_factor: float,
    lift_code: int,
    lift_gas_velocity_m_s: FloatArray,
    lift_gas_density_kg_m3: FloatArray,
    lift_gas_mean_free_path_m: FloatArray,
    lift_gas_dynamic_viscosity_Pa_s: FloatArray,
    lift_azimuthal_gas_vorticity_s_inv: FloatArray,
    lift_coefficient: float,
    electric_enabled: bool,
    electric_field_V_m: FloatArray,
    gravity_enabled: bool,
    gravity_density_kg_m3: FloatArray,
    gravity_x_m_s2: float,
    gravity_y_m_s2: float,
    acceleration: FloatArray,
    charge_rate: FloatArray,
    charge_rate_derivative: FloatArray,
    applicable: NDArray[np.bool_],
    linear_drag_rate: FloatArray,
    target_velocity: FloatArray,
    additive_acceleration: FloatArray,
    error_code: UInt8Array,
) -> None:
    """Evaluate physics into caller-owned, disjoint row buffers."""

    for row in range(particle_index.size):
        prior_error = error_code[row]
        resident = particle_index[row]
        mass = mass_kg[resident]
        diameter = drag_diameter_m[resident]
        velocity_x = velocity_m_s[row, 0]
        velocity_y = velocity_m_s[row, 1]
        acceleration[row, 0] = 0.0
        acceleration[row, 1] = 0.0
        charge_rate[row] = 0.0
        charge_rate_derivative[row] = 0.0
        applicable[row] = True
        linear_drag_rate[row] = 0.0
        target_velocity[row, 0] = 0.0
        target_velocity[row, 1] = 0.0
        additive_acceleration[row, 0] = 0.0
        additive_acceleration[row, 1] = 0.0
        if prior_error != ERROR_NONE:
            continue

        (
            row_charge_rate,
            row_charge_rate_derivative,
            charge_applicable,
            charge_error,
        ) = _charge_row(
            charge_code,
            charge_number[row],
            charge_number_lower,
            charge_number_upper,
            electrostatic_radius_m[resident],
            electron_number_density_m3,
            positive_ion_number_density_m3,
            electron_temperature_K,
            positive_ion_temperature_K,
            positive_ion_velocity_m_s,
            positive_ion_mass_kg,
            maximum_ion_drift_ratio,
            aggregate_electron_thermal_voltage_V,
            aggregate_positive_ion_thermal_voltage_V,
            aggregate_effective_positive_ion_mass_kg,
            aggregate_negative_ion_number_density_m3,
            aggregate_negative_ion_thermal_voltage_V,
            aggregate_negative_ion_velocity_m_s,
            aggregate_effective_negative_ion_mass_kg,
            aggregate_screening_length_m,
            aggregate_maximum_relative_ion_speed_m_s,
            velocity_x,
            velocity_y,
            row,
        )
        charge_rate[row] = row_charge_rate
        charge_rate_derivative[row] = row_charge_rate_derivative
        applicable[row] = charge_applicable
        error_code[row] = charge_error
        if charge_error != ERROR_NONE:
            continue

        drag_x, drag_y, rate, row_applicable, row_error = _drag_row(
            drag_code,
            mass,
            diameter,
            velocity_x,
            velocity_y,
            gas_velocity_m_s,
            gas_density_kg_m3,
            gas_temperature_K,
            gas_dynamic_viscosity_Pa_s,
            gas_mean_free_path_m,
            gas_molecular_mass_kg,
            epstein_delta,
            epstein_diffuse_reflection_fraction,
            epstein_maximum_speed_ratio,
            row,
        )
        acceleration[row, 0] = drag_x
        acceleration[row, 1] = drag_y
        linear_drag_rate[row] = rate
        applicable[row] = applicable[row] and row_applicable
        error_code[row] = row_error
        if row_error != ERROR_NONE:
            continue
        if drag_code != DRAG_NONE:
            target_velocity[row, 0] = gas_velocity_m_s[row, 0]
            target_velocity[row, 1] = gas_velocity_m_s[row, 1]

        thermophoresis_x, thermophoresis_y, row_applicable = _thermophoresis_row(
            thermophoresis_code,
            mass,
            diameter,
            velocity_x,
            velocity_y,
            thermophoresis_gas_velocity_m_s,
            thermophoresis_gas_temperature_K,
            thermophoresis_gas_translational_heat_flux_W_m2,
            thermophoresis_gas_mean_free_path_m,
            thermophoresis_gas_molecular_mass_kg,
            thermophoresis_maximum_speed_ratio,
            thermophoresis_gas_temperature_gradient_K_m,
            thermophoresis_gas_density_kg_m3,
            thermophoresis_gas_dynamic_viscosity_Pa_s,
            thermophoresis_gas_thermal_conductivity_W_m_K,
            thermophoresis_particle_thermal_conductivity_W_m_K,
            thermophoresis_thermal_slip_coefficient,
            thermophoresis_momentum_exchange_coefficient,
            thermophoresis_thermal_exchange_coefficient,
            row,
        )
        applicable[row] = applicable[row] and row_applicable
        additive_acceleration[row, 0] += thermophoresis_x
        additive_acceleration[row, 1] += thermophoresis_y
        acceleration[row, 0] += thermophoresis_x
        acceleration[row, 1] += thermophoresis_y

        ion_drag_x, ion_drag_y, row_applicable, row_error = _ion_drag_row(
            ion_drag_code,
            mass,
            electrostatic_radius_m[resident],
            charge_number[row],
            velocity_x,
            velocity_y,
            ion_drag_electron_number_density_m3,
            ion_drag_positive_ion_number_density_m3,
            ion_drag_electron_temperature_K,
            ion_drag_positive_ion_temperature_K,
            ion_drag_positive_ion_velocity_m_s,
            ion_neutral_mean_free_path_m,
            ion_drag_positive_ion_mass_kg,
            ion_drag_maximum_ion_drift_ratio,
            ion_drag_electron_thermal_voltage_V,
            ion_drag_positive_ion_thermal_voltage_V,
            ion_drag_effective_positive_ion_mass_kg,
            ion_drag_screening_length_m,
            ion_drag_electric_field_V_m,
            ion_drag_maximum_relative_ion_speed_m_s,
            row,
        )
        applicable[row] = applicable[row] and row_applicable
        error_code[row] = row_error
        additive_acceleration[row, 0] += ion_drag_x
        additive_acceleration[row, 1] += ion_drag_y
        acceleration[row, 0] += ion_drag_x
        acceleration[row, 1] += ion_drag_y

        dep_x, dep_y = _dielectrophoresis_row(
            dielectrophoresis_enabled,
            mass,
            electrostatic_radius_m[resident],
            gradient_mean_e_squared_V2_m3,
            dep_medium_relative_permittivity,
            dep_real_clausius_mossotti_factor,
            row,
        )
        additive_acceleration[row, 0] += dep_x
        additive_acceleration[row, 1] += dep_y
        acceleration[row, 0] += dep_x
        acceleration[row, 1] += dep_y

        lift_x, lift_y, row_applicable = _lift_row(
            lift_code,
            mass,
            diameter,
            velocity_x,
            velocity_y,
            lift_gas_velocity_m_s,
            lift_gas_density_kg_m3,
            lift_gas_mean_free_path_m,
            lift_gas_dynamic_viscosity_Pa_s,
            lift_azimuthal_gas_vorticity_s_inv,
            lift_coefficient,
            row,
        )
        applicable[row] = applicable[row] and row_applicable
        additive_acceleration[row, 0] += lift_x
        additive_acceleration[row, 1] += lift_y
        acceleration[row, 0] += lift_x
        acceleration[row, 1] += lift_y

        electric_x, electric_y = _electric_row(
            electric_enabled,
            charge_number[row],
            mass,
            electric_field_V_m,
            row,
        )
        additive_acceleration[row, 0] += electric_x
        additive_acceleration[row, 1] += electric_y
        acceleration[row, 0] += electric_x
        acceleration[row, 1] += electric_y

        gravity_x, gravity_y = _gravity_buoyancy_row(
            gravity_enabled,
            mass,
            displaced_volume_m3[resident],
            gravity_density_kg_m3,
            gravity_x_m_s2,
            gravity_y_m_s2,
            row,
        )
        additive_acceleration[row, 0] += gravity_x
        additive_acceleration[row, 1] += gravity_y
        acceleration[row, 0] += gravity_x
        acceleration[row, 1] += gravity_y

        if not (
            math.isfinite(acceleration[row, 0])
            and math.isfinite(acceleration[row, 1])
            and math.isfinite(additive_acceleration[row, 0])
            and math.isfinite(additive_acceleration[row, 1])
        ):
            error_code[row] = ERROR_ACCELERATION
            acceleration[row, 0] = 0.0
            acceleration[row, 1] = 0.0
            additive_acceleration[row, 0] = 0.0
            additive_acceleration[row, 1] = 0.0


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _charge_row(
    charge_code: int,
    charge_number: float,
    charge_number_lower: float,
    charge_number_upper: float,
    radius_m: float,
    electron_density_m3: FloatArray,
    ion_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    ion_temperature_K: FloatArray,
    ion_velocity_m_s: FloatArray,
    ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
    aggregate_electron_thermal_voltage_V: FloatArray,
    aggregate_ion_thermal_voltage_V: FloatArray,
    aggregate_effective_ion_mass_kg: FloatArray,
    aggregate_negative_ion_density_m3: FloatArray,
    aggregate_negative_ion_thermal_voltage_V: FloatArray,
    aggregate_negative_ion_velocity_m_s: FloatArray,
    aggregate_effective_negative_ion_mass_kg: FloatArray,
    aggregate_screening_length_m: FloatArray,
    aggregate_maximum_relative_ion_speed_m_s: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    row: int,
) -> tuple[float, float, bool, int]:
    """Dispatch the selected continuous-charge revision for one row."""

    if charge_code == CHARGE_FIXED:
        return 0.0, 0.0, True, ERROR_NONE
    if charge_number < charge_number_lower or charge_number > charge_number_upper:
        return 0.0, 0.0, True, ERROR_CHARGE_INVARIANT
    if charge_code == CHARGE_OML_STATIONARY_MAXWELLIAN_DEBYE_HUCKEL:
        return _oml_charge_row(
            charge_number,
            radius_m,
            electron_density_m3[row],
            ion_density_m3[row],
            electron_temperature_K[row],
            ion_temperature_K[row],
            velocity_x_m_s,
            velocity_y_m_s,
            ion_velocity_m_s[row, 0],
            ion_velocity_m_s[row, 1],
            ion_mass_kg,
        )
    if charge_code == CHARGE_OML_SHIFTED_MAXWELLIAN_SINGLE_ION_NEGATIVE_DEBYE_HUCKEL:
        return _oml_shifted_charge_row(
            charge_number,
            radius_m,
            electron_density_m3[row],
            ion_density_m3[row],
            electron_temperature_K[row],
            ion_temperature_K[row],
            velocity_x_m_s,
            velocity_y_m_s,
            ion_velocity_m_s[row, 0],
            ion_velocity_m_s[row, 1],
            ion_mass_kg,
            maximum_ion_drift_ratio,
        )
    if charge_code == CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_TWO_CURRENT:
        return _aggregate_relative_drift_charge_row(
            charge_number,
            radius_m,
            electron_density_m3[row],
            ion_density_m3[row],
            aggregate_electron_thermal_voltage_V[row],
            aggregate_ion_thermal_voltage_V[row],
            velocity_x_m_s,
            velocity_y_m_s,
            ion_velocity_m_s[row, 0],
            ion_velocity_m_s[row, 1],
            aggregate_effective_ion_mass_kg[row],
            aggregate_screening_length_m[row],
            aggregate_maximum_relative_ion_speed_m_s,
        )
    if charge_code == CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT:
        return _aggregate_relative_drift_three_current_charge_row(
            charge_number,
            radius_m,
            electron_density_m3[row],
            ion_density_m3[row],
            aggregate_negative_ion_density_m3[row],
            aggregate_electron_thermal_voltage_V[row],
            aggregate_ion_thermal_voltage_V[row],
            aggregate_negative_ion_thermal_voltage_V[row],
            velocity_x_m_s,
            velocity_y_m_s,
            ion_velocity_m_s[row, 0],
            ion_velocity_m_s[row, 1],
            aggregate_negative_ion_velocity_m_s[row, 0],
            aggregate_negative_ion_velocity_m_s[row, 1],
            aggregate_effective_ion_mass_kg[row],
            aggregate_effective_negative_ion_mass_kg[row],
            aggregate_screening_length_m[row],
            aggregate_maximum_relative_ion_speed_m_s,
        )
    return 0.0, 0.0, True, ERROR_DERIVED_VALUE


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _drag_row(
    drag_code: int,
    mass_kg: float,
    diameter_m: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    gas_velocity_m_s: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_temperature_K: FloatArray,
    gas_dynamic_viscosity_Pa_s: FloatArray,
    gas_mean_free_path_m: FloatArray,
    gas_molecular_mass_kg: float,
    epstein_delta: float,
    epstein_diffuse_reflection_fraction: float,
    epstein_maximum_speed_ratio: float,
    row: int,
) -> tuple[float, float, float, bool, int]:
    """Dispatch the selected drag law for one row."""

    if drag_code == DRAG_EPSTEIN:
        return _epstein_drag_row(
            mass_kg,
            diameter_m,
            velocity_x_m_s,
            velocity_y_m_s,
            gas_velocity_m_s[row, 0],
            gas_velocity_m_s[row, 1],
            gas_density_kg_m3[row],
            gas_temperature_K[row],
            gas_mean_free_path_m[row],
            gas_molecular_mass_kg,
            epstein_delta,
            epstein_maximum_speed_ratio,
        )
    if drag_code == DRAG_STOKES_CUNNINGHAM:
        return _stokes_cunningham_drag_row(
            mass_kg,
            diameter_m,
            velocity_x_m_s,
            velocity_y_m_s,
            gas_velocity_m_s[row, 0],
            gas_velocity_m_s[row, 1],
            gas_density_kg_m3[row],
            gas_dynamic_viscosity_Pa_s[row],
            gas_mean_free_path_m[row],
        )
    if drag_code == DRAG_EPSTEIN_FINITE_SPEED:
        return _epstein_finite_speed_drag_row(
            mass_kg,
            diameter_m,
            velocity_x_m_s,
            velocity_y_m_s,
            gas_velocity_m_s[row, 0],
            gas_velocity_m_s[row, 1],
            gas_density_kg_m3[row],
            gas_temperature_K[row],
            gas_mean_free_path_m[row],
            gas_molecular_mass_kg,
            epstein_diffuse_reflection_fraction,
            epstein_maximum_speed_ratio,
        )
    if drag_code == DRAG_NONE:
        return 0.0, 0.0, 0.0, True, ERROR_NONE
    return 0.0, 0.0, 0.0, True, ERROR_DERIVED_VALUE


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _thermophoresis_row(
    thermophoresis_code: int,
    particle_mass_kg: float,
    particle_diameter_m: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    gas_velocity_m_s: FloatArray,
    gas_temperature_K: FloatArray,
    gas_translational_heat_flux_W_m2: FloatArray,
    gas_mean_free_path_m: FloatArray,
    gas_molecular_mass_kg: float,
    maximum_speed_ratio: float,
    gas_temperature_gradient_K_m: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_dynamic_viscosity_Pa_s: FloatArray,
    gas_thermal_conductivity_W_m_K: FloatArray,
    particle_thermal_conductivity_W_m_K: float,
    thermal_slip_coefficient: float,
    momentum_exchange_coefficient: float,
    thermal_exchange_coefficient: float,
    row: int,
) -> tuple[float, float, bool]:
    """Evaluate the selected thermophoresis revision for one stage row."""

    if thermophoresis_code == THERMOPHORESIS_NONE:
        return 0.0, 0.0, True
    if thermophoresis_code == THERMOPHORESIS_TALBOT:
        return _talbot_thermophoresis_row(
            particle_mass_kg,
            particle_diameter_m,
            gas_temperature_K[row],
            gas_temperature_gradient_K_m[row, 0],
            gas_temperature_gradient_K_m[row, 1],
            gas_density_kg_m3[row],
            gas_dynamic_viscosity_Pa_s[row],
            gas_thermal_conductivity_W_m_K[row],
            gas_mean_free_path_m[row],
            particle_thermal_conductivity_W_m_K,
            thermal_slip_coefficient,
            momentum_exchange_coefficient,
            thermal_exchange_coefficient,
        )
    if thermophoresis_code != THERMOPHORESIS_WALDMANN_GALLIS:
        return math.nan, math.nan, False
    radius_m = 0.5 * particle_diameter_m
    mean_thermal_speed = math.sqrt(
        8.0 * BOLTZMANN_J_K * gas_temperature_K[row] / (math.pi * gas_molecular_mass_kg)
    )
    factor = (32.0 / 15.0) * radius_m * radius_m / (particle_mass_kg * mean_thermal_speed)
    acceleration_x = factor * gas_translational_heat_flux_W_m2[row, 0]
    acceleration_y = factor * gas_translational_heat_flux_W_m2[row, 1]
    relative_x = gas_velocity_m_s[row, 0] - velocity_x_m_s
    relative_y = gas_velocity_m_s[row, 1] - velocity_y_m_s
    relative_speed_ratio = math.hypot(relative_x, relative_y) / mean_thermal_speed
    mean_free_path_over_radius = gas_mean_free_path_m[row] / radius_m
    derived = (
        radius_m,
        mean_thermal_speed,
        factor,
        acceleration_x,
        acceleration_y,
        relative_speed_ratio,
        mean_free_path_over_radius,
    )
    for value in derived:
        if not math.isfinite(value):
            return math.nan, math.nan, False
    applicable = (
        mean_free_path_over_radius >= WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS
        and relative_speed_ratio <= maximum_speed_ratio
    )
    return acceleration_x, acceleration_y, applicable


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _talbot_thermophoresis_row(
    particle_mass_kg: float,
    particle_diameter_m: float,
    gas_temperature_K: float,
    temperature_gradient_x_K_m: float,
    temperature_gradient_y_K_m: float,
    gas_density_kg_m3: float,
    gas_dynamic_viscosity_Pa_s: float,
    gas_thermal_conductivity_W_m_K: float,
    gas_mean_free_path_m: float,
    particle_thermal_conductivity_W_m_K: float,
    thermal_slip_coefficient: float,
    momentum_exchange_coefficient: float,
    thermal_exchange_coefficient: float,
) -> tuple[float, float, bool]:
    """Evaluate the radius-Knudsen Talbot revision for one row."""

    knudsen_radius = 2.0 * (gas_mean_free_path_m / particle_diameter_m)
    conductivity_ratio = gas_thermal_conductivity_W_m_K / particle_thermal_conductivity_W_m_K
    correction = (
        thermal_slip_coefficient
        * (conductivity_ratio + thermal_exchange_coefficient * knudsen_radius)
        / (
            (1.0 + 3.0 * momentum_exchange_coefficient * knudsen_radius)
            * (1.0 + 2.0 * conductivity_ratio + 2.0 * thermal_exchange_coefficient * knudsen_radius)
        )
    )
    factor = (
        -6.0
        * math.pi
        * particle_diameter_m
        * gas_dynamic_viscosity_Pa_s**2
        * correction
        / (particle_mass_kg * gas_density_kg_m3 * gas_temperature_K)
    )
    acceleration_x = factor * temperature_gradient_x_K_m
    acceleration_y = factor * temperature_gradient_y_K_m
    derived = (
        knudsen_radius,
        conductivity_ratio,
        correction,
        factor,
        acceleration_x,
        acceleration_y,
    )
    for value in derived:
        if not math.isfinite(value):
            return math.nan, math.nan, False
    return acceleration_x, acceleration_y, True


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _dielectrophoresis_row(
    enabled: bool,
    particle_mass_kg: float,
    particle_radius_m: float,
    gradient_mean_e_squared_V2_m3: FloatArray,
    medium_relative_permittivity: float,
    real_clausius_mossotti_factor: float,
    row: int,
) -> tuple[float, float]:
    """Evaluate the selected quasistatic spherical DEP contribution."""

    if not enabled:
        return 0.0, 0.0
    factor = (
        2.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * medium_relative_permittivity
        * particle_radius_m**3
        * real_clausius_mossotti_factor
        / particle_mass_kg
    )
    return (
        factor * gradient_mean_e_squared_V2_m3[row, 0],
        factor * gradient_mean_e_squared_V2_m3[row, 1],
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _lift_row(
    lift_code: int,
    particle_mass_kg: float,
    particle_diameter_m: float,
    velocity_r_m_s: float,
    velocity_z_m_s: float,
    gas_velocity_m_s: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_mean_free_path_m: FloatArray,
    gas_dynamic_viscosity_Pa_s: FloatArray,
    azimuthal_gas_vorticity_s_inv: FloatArray,
    lift_coefficient: float,
    row: int,
) -> tuple[float, float, bool]:
    """Evaluate the selected planar lift revision for one row."""

    if lift_code == LIFT_NONE:
        return 0.0, 0.0, True
    if lift_code == LIFT_SAFFMAN_XY or lift_code == LIFT_SAFFMAN_RZ:
        return _saffman_lift_row(
            lift_code,
            particle_mass_kg,
            particle_diameter_m,
            velocity_r_m_s,
            velocity_z_m_s,
            gas_velocity_m_s[row, 0],
            gas_velocity_m_s[row, 1],
            gas_density_kg_m3[row],
            gas_dynamic_viscosity_Pa_s[row],
            gas_mean_free_path_m[row],
            azimuthal_gas_vorticity_s_inv[row],
        )
    if lift_code != LIFT_RAREFIED_VORTICITY_RZ:
        return math.nan, math.nan, False
    radius_m = 0.5 * particle_diameter_m
    coupling_rate = (
        lift_coefficient
        * math.pi
        * gas_density_kg_m3[row]
        * gas_mean_free_path_m[row]
        * radius_m**2
        * azimuthal_gas_vorticity_s_inv[row]
        / particle_mass_kg
    )
    relative_r = gas_velocity_m_s[row, 0] - velocity_r_m_s
    relative_z = gas_velocity_m_s[row, 1] - velocity_z_m_s
    acceleration_r = coupling_rate * relative_z
    acceleration_z = -coupling_rate * relative_r
    mean_free_path_over_radius = gas_mean_free_path_m[row] / radius_m
    derived = (
        radius_m,
        coupling_rate,
        acceleration_r,
        acceleration_z,
        mean_free_path_over_radius,
    )
    for value in derived:
        if not math.isfinite(value):
            return math.nan, math.nan, False
    return (
        acceleration_r,
        acceleration_z,
        mean_free_path_over_radius >= RAREFIED_VORTICITY_LIFT_MIN_MEAN_FREE_PATH_OVER_RADIUS,
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _saffman_lift_row(
    lift_code: int,
    particle_mass_kg: float,
    particle_diameter_m: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    gas_velocity_x_m_s: float,
    gas_velocity_y_m_s: float,
    gas_density_kg_m3: float,
    gas_dynamic_viscosity_Pa_s: float,
    gas_mean_free_path_m: float,
    out_of_plane_gas_vorticity_s_inv: float,
) -> tuple[float, float, bool]:
    """Evaluate Saffman's unbounded creeping-flow formula without epsilon floors."""

    radius_m = 0.5 * particle_diameter_m
    slip_x = gas_velocity_x_m_s - velocity_x_m_s
    slip_y = gas_velocity_y_m_s - velocity_y_m_s
    slip_speed = math.hypot(slip_x, slip_y)
    omega_abs = abs(out_of_plane_gas_vorticity_s_inv)
    coupling_rate = (
        SAFFMAN_LIFT_COEFFICIENT
        * radius_m**2
        * math.sqrt(gas_dynamic_viscosity_Pa_s * gas_density_kg_m3 * omega_abs)
        / particle_mass_kg
    )
    signed_coupling = math.copysign(coupling_rate, out_of_plane_gas_vorticity_s_inv)
    if lift_code == LIFT_SAFFMAN_XY:
        acceleration_x = signed_coupling * slip_y
        acceleration_y = -signed_coupling * slip_x
    else:
        acceleration_x = -signed_coupling * slip_y
        acceleration_y = signed_coupling * slip_x
    mean_free_path_over_radius = gas_mean_free_path_m / radius_m
    slip_reynolds = gas_density_kg_m3 * radius_m * slip_speed / gas_dynamic_viscosity_Pa_s
    shear_reynolds = gas_density_kg_m3 * radius_m**2 * omega_abs / gas_dynamic_viscosity_Pa_s
    derived = (
        coupling_rate,
        acceleration_x,
        acceleration_y,
        mean_free_path_over_radius,
        slip_reynolds,
        shear_reynolds,
    )
    for value in derived:
        if not math.isfinite(value):
            return math.nan, math.nan, False
    applicable = mean_free_path_over_radius <= SAFFMAN_MAX_MEAN_FREE_PATH_OVER_RADIUS
    applicable = applicable and slip_reynolds <= SAFFMAN_MAX_SLIP_REYNOLDS
    applicable = applicable and shear_reynolds <= SAFFMAN_MAX_SHEAR_REYNOLDS
    applicable = applicable and (
        omega_abs == 0.0
        or slip_reynolds <= SAFFMAN_MAX_SLIP_TO_SQRT_SHEAR_REYNOLDS * math.sqrt(shear_reynolds)
    )
    return acceleration_x, acceleration_y, applicable


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _electric_row(
    enabled: bool,
    charge_number: float,
    particle_mass_kg: float,
    electric_field_V_m: FloatArray,
    row: int,
) -> tuple[float, float]:
    """Evaluate the optional Coulomb contribution for one row."""

    if not enabled:
        return 0.0, 0.0
    charge_coulomb = charge_number * ELEMENTARY_CHARGE_C
    return (
        charge_coulomb * electric_field_V_m[row, 0] / particle_mass_kg,
        charge_coulomb * electric_field_V_m[row, 1] / particle_mass_kg,
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _gravity_buoyancy_row(
    enabled: bool,
    particle_mass_kg: float,
    displaced_volume_m3: float,
    gas_density_kg_m3: FloatArray,
    gravity_x_m_s2: float,
    gravity_y_m_s2: float,
    row: int,
) -> tuple[float, float]:
    """Evaluate the optional gravity/buoyancy contribution for one row."""

    if not enabled:
        return 0.0, 0.0
    buoyancy_factor = 1.0 - gas_density_kg_m3[row] * displaced_volume_m3 / particle_mass_kg
    return buoyancy_factor * gravity_x_m_s2, buoyancy_factor * gravity_y_m_s2


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _ion_drag_row(
    ion_drag_code: int,
    particle_mass_kg: float,
    particle_radius_m: float,
    charge_number: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    electron_density_m3: FloatArray,
    ion_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    ion_temperature_K: FloatArray,
    ion_velocity_m_s: FloatArray,
    ion_neutral_mean_free_path_m: FloatArray,
    ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
    electron_thermal_voltage_V: FloatArray,
    positive_ion_thermal_voltage_V: FloatArray,
    effective_positive_ion_mass_kg: FloatArray,
    screening_length_m: FloatArray,
    electric_field_V_m: FloatArray,
    maximum_relative_ion_speed_m_s: float,
    row: int,
) -> tuple[float, float, bool, int]:
    """Evaluate the selected ion-drag revision for one stage row."""

    if ion_drag_code == ION_DRAG_NONE:
        return 0.0, 0.0, True, ERROR_NONE
    if ion_drag_code == ION_DRAG_RELATIVE_FLOW_SCREENED:
        return _relative_flow_screened_ion_drag_row(
            particle_mass_kg,
            particle_radius_m,
            charge_number,
            velocity_x_m_s,
            velocity_y_m_s,
            ion_density_m3,
            positive_ion_thermal_voltage_V,
            ion_velocity_m_s,
            effective_positive_ion_mass_kg,
            screening_length_m,
            ion_neutral_mean_free_path_m,
            maximum_relative_ion_speed_m_s,
            row,
        )
    if ion_drag_code == ION_DRAG_ELECTRIC_FIELD_DIRECTED_IMAGE:
        return _electric_field_directed_image_ion_drag_row(
            particle_mass_kg,
            particle_radius_m,
            charge_number,
            ion_density_m3,
            electron_thermal_voltage_V,
            positive_ion_thermal_voltage_V,
            ion_velocity_m_s,
            effective_positive_ion_mass_kg,
            screening_length_m,
            electric_field_V_m,
            row,
        )
    if ion_drag_code != ION_DRAG_BARNES_COLLISIONLESS_EFFECTIVE_SPEED:
        return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    return _barnes_collisionless_ion_drag_row(
        particle_mass_kg,
        particle_radius_m,
        charge_number,
        velocity_x_m_s,
        velocity_y_m_s,
        electron_density_m3,
        ion_density_m3,
        electron_temperature_K,
        ion_temperature_K,
        ion_velocity_m_s,
        ion_neutral_mean_free_path_m,
        ion_mass_kg,
        maximum_ion_drift_ratio,
        row,
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _barnes_collisionless_ion_drag_row(
    particle_mass_kg: float,
    particle_radius_m: float,
    charge_number: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    electron_density_m3: FloatArray,
    ion_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    ion_temperature_K: FloatArray,
    ion_velocity_m_s: FloatArray,
    ion_neutral_mean_free_path_m: FloatArray,
    ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
    row: int,
) -> tuple[float, float, bool, int]:
    if charge_number > 0.0:
        return 0.0, 0.0, False, ERROR_NONE

    inverse_debye_square = (
        ELEMENTARY_CHARGE_C
        * ELEMENTARY_CHARGE_C
        / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
        * (
            electron_density_m3[row] / electron_temperature_K[row]
            + ion_density_m3[row] / ion_temperature_K[row]
        )
    )
    debye_length = 1.0 / math.sqrt(inverse_debye_square)
    relative_x = ion_velocity_m_s[row, 0] - velocity_x_m_s
    relative_y = ion_velocity_m_s[row, 1] - velocity_y_m_s
    relative_speed = math.hypot(relative_x, relative_y)
    ion_mean_thermal_speed = math.sqrt(
        8.0 * BOLTZMANN_J_K * ion_temperature_K[row] / (math.pi * ion_mass_kg)
    )
    effective_speed_square = relative_speed * relative_speed + ion_mean_thermal_speed**2
    effective_speed = math.sqrt(effective_speed_square)
    capacitance = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * particle_radius_m
        * (1.0 + particle_radius_m / debye_length)
    )
    surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    orbital_impact = (
        abs(charge_number)
        * ELEMENTARY_CHARGE_C
        * ELEMENTARY_CHARGE_C
        / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * ion_mass_kg * effective_speed_square)
    )
    collection_square = (
        particle_radius_m
        * particle_radius_m
        * (
            1.0
            - 2.0 * ELEMENTARY_CHARGE_C * surface_potential / (ion_mass_kg * effective_speed_square)
        )
    )
    collection_impact = math.sqrt(collection_square)
    coulomb_logarithm = 0.5 * math.log(
        (debye_length * debye_length + orbital_impact * orbital_impact)
        / (collection_square + orbital_impact * orbital_impact)
    )
    cross_section = (
        math.pi * collection_square
        + 4.0 * math.pi * orbital_impact * orbital_impact * coulomb_logarithm
    )
    factor = ion_density_m3[row] * ion_mass_kg * effective_speed * cross_section
    factor /= particle_mass_kg
    acceleration_x = factor * relative_x
    acceleration_y = factor * relative_y
    ion_drift_ratio = relative_speed / ion_mean_thermal_speed
    derived = (
        debye_length,
        effective_speed,
        capacitance,
        surface_potential,
        orbital_impact,
        collection_square,
        collection_impact,
        coulomb_logarithm,
        acceleration_x,
        acceleration_y,
        ion_drift_ratio,
    )
    for value in derived:
        if not math.isfinite(value):
            return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    applicable = (
        surface_potential <= 0.0
        and particle_radius_m / debye_length <= ION_DRAG_MAX_SCALE_OVER_DEBYE
        and orbital_impact / debye_length <= ION_DRAG_MAX_SCALE_OVER_DEBYE
        and collection_impact / debye_length <= ION_DRAG_MAX_SCALE_OVER_DEBYE
        and ion_neutral_mean_free_path_m[row] / debye_length
        >= ION_DRAG_MIN_MEAN_FREE_PATH_OVER_DEBYE
        and coulomb_logarithm > 0.0
        and ion_drift_ratio <= maximum_ion_drift_ratio
    )
    return acceleration_x, acceleration_y, applicable, ERROR_NONE


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _relative_flow_screened_ion_drag_row(
    particle_mass_kg: float,
    particle_radius_m: float,
    charge_number: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    ion_density_m3: FloatArray,
    ion_thermal_voltage_V: FloatArray,
    ion_velocity_m_s: FloatArray,
    ion_mass_kg: FloatArray,
    screening_length_m: FloatArray,
    ion_neutral_mean_free_path_m: FloatArray,
    maximum_relative_ion_speed_m_s: float,
    row: int,
) -> tuple[float, float, bool, int]:
    relative_x = ion_velocity_m_s[row, 0] - velocity_x_m_s
    relative_y = ion_velocity_m_s[row, 1] - velocity_y_m_s
    relative_speed = math.hypot(relative_x, relative_y)
    mass = ion_mass_kg[row]
    speed_square = (
        relative_speed * relative_speed
        + 8.0 * ELEMENTARY_CHARGE_C * ion_thermal_voltage_V[row] / (math.pi * mass)
        + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    effective_speed = math.sqrt(speed_square)
    capacitance_screening = max(particle_radius_m, screening_length_m[row])
    capacitance = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * particle_radius_m
        * (1.0 + particle_radius_m / capacitance_screening)
    )
    surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    screening_radius = max(
        particle_radius_m,
        min(screening_length_m[row], ion_neutral_mean_free_path_m[row]),
    )
    orbital_impact = (
        math.sqrt(charge_number * charge_number + AGGREGATE_ION_DRAG_CHARGE_SQUARE_REGULARIZATION)
        * ELEMENTARY_CHARGE_C**2
        / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * mass * speed_square)
    )
    collection_square = min(
        screening_radius * screening_radius,
        particle_radius_m
        * particle_radius_m
        * max(
            0.0,
            1.0 - 2.0 * ELEMENTARY_CHARGE_C * surface_potential / (mass * speed_square),
        ),
    )
    coulomb_logarithm = max(
        0.0,
        0.5
        * math.log(
            (screening_radius * screening_radius + orbital_impact * orbital_impact)
            / (collection_square + orbital_impact * orbital_impact)
        ),
    )
    cross_section = (
        math.pi * collection_square
        + 4.0 * math.pi * orbital_impact * orbital_impact * coulomb_logarithm
    )
    factor = ion_density_m3[row] * mass * effective_speed * cross_section / particle_mass_kg
    acceleration_x = factor * relative_x
    acceleration_y = factor * relative_y
    derived = (
        relative_speed,
        effective_speed,
        capacitance,
        surface_potential,
        screening_radius,
        orbital_impact,
        collection_square,
        coulomb_logarithm,
        acceleration_x,
        acceleration_y,
    )
    for value in derived:
        if not math.isfinite(value):
            return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    return (
        acceleration_x,
        acceleration_y,
        relative_speed <= maximum_relative_ion_speed_m_s,
        ERROR_NONE,
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _electric_field_directed_image_ion_drag_row(
    particle_mass_kg: float,
    particle_radius_m: float,
    charge_number: float,
    ion_density_m3: FloatArray,
    electron_thermal_voltage_V: FloatArray,
    ion_thermal_voltage_V: FloatArray,
    ion_velocity_m_s: FloatArray,
    ion_mass_kg: FloatArray,
    screening_length_m: FloatArray,
    electric_field_V_m: FloatArray,
    row: int,
) -> tuple[float, float, bool, int]:
    ion_speed = math.hypot(ion_velocity_m_s[row, 0], ion_velocity_m_s[row, 1])
    mass = ion_mass_kg[row]
    speed_square = (
        ion_speed * ion_speed
        + 8.0 * ELEMENTARY_CHARGE_C * ion_thermal_voltage_V[row] / (math.pi * mass)
        + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    effective_speed = math.sqrt(speed_square)
    capacitance_screening = max(particle_radius_m, screening_length_m[row])
    capacitance = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * particle_radius_m
        * (1.0 + particle_radius_m / capacitance_screening)
    )
    surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    collection_cross_section = (
        math.pi
        * particle_radius_m
        * particle_radius_m
        * max(0.0, 1.0 - surface_potential / ion_thermal_voltage_V[row])
    )
    image_impact = (
        ELEMENTARY_CHARGE_C**2
        * charge_number
        / (2.0 * math.pi * VACUUM_PERMITTIVITY_F_M * mass * speed_square)
    )
    image_screening_length = math.sqrt(
        VACUUM_PERMITTIVITY_F_M
        * electron_thermal_voltage_V[row]
        / (ELEMENTARY_CHARGE_C * ion_density_m3[row])
    )
    image_logarithm = math.log(
        max(
            1.0 + IMAGE_ION_DRAG_LOG_ARGUMENT_OFFSET,
            image_screening_length / particle_radius_m,
        )
    )
    orbital_cross_section = math.pi * image_impact * image_impact * image_logarithm
    force_magnitude = (
        mass
        * ion_density_m3[row]
        * effective_speed
        * ion_speed
        * (collection_cross_section + orbital_cross_section)
    )
    electric_x = electric_field_V_m[row, 0]
    electric_y = electric_field_V_m[row, 1]
    electric_norm = math.sqrt(
        electric_x * electric_x
        + electric_y * electric_y
        + IMAGE_ION_DRAG_ELECTRIC_FIELD_SQUARE_FLOOR_V2_M2
    )
    acceleration_scale = force_magnitude / (electric_norm * particle_mass_kg)
    acceleration_x = acceleration_scale * electric_x
    acceleration_y = acceleration_scale * electric_y
    derived = (
        ion_speed,
        effective_speed,
        capacitance,
        surface_potential,
        collection_cross_section,
        image_impact,
        image_screening_length,
        image_logarithm,
        orbital_cross_section,
        acceleration_x,
        acceleration_y,
    )
    for value in derived:
        if not math.isfinite(value):
            return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    return acceleration_x, acceleration_y, True, ERROR_NONE


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _oml_charge_row(
    charge_number: float,
    radius_m: float,
    electron_density_m3: float,
    ion_density_m3: float,
    electron_temperature_K: float,
    ion_temperature_K: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    ion_velocity_x_m_s: float,
    ion_velocity_y_m_s: float,
    ion_mass_kg: float,
) -> tuple[float, float, bool, int]:
    """Evaluate the revision-1 OML rate and its local applicability gates."""

    inverse_debye_square = (
        ELEMENTARY_CHARGE_C
        * ELEMENTARY_CHARGE_C
        / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
        * (electron_density_m3 / electron_temperature_K + ion_density_m3 / ion_temperature_K)
    )
    debye_length = 1.0 / math.sqrt(inverse_debye_square)
    capacitance = (
        4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * radius_m * (1.0 + radius_m / debye_length)
    )
    potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    electron_voltage = BOLTZMANN_J_K * electron_temperature_K / ELEMENTARY_CHARGE_C
    ion_voltage = BOLTZMANN_J_K * ion_temperature_K / ELEMENTARY_CHARGE_C
    electron_amplitude = (
        math.pi
        * radius_m
        * radius_m
        * electron_density_m3
        * math.sqrt(8.0 * BOLTZMANN_J_K * electron_temperature_K / (math.pi * ELECTRON_MASS_KG))
    )
    ion_amplitude = (
        math.pi
        * radius_m
        * radius_m
        * ion_density_m3
        * math.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_K / (math.pi * ion_mass_kg))
    )
    if potential <= 0.0:
        electron_collection = electron_amplitude * math.exp(potential / electron_voltage)
        ion_collection = ion_amplitude * (1.0 - potential / ion_voltage)
        rate_derivative = (
            -ELEMENTARY_CHARGE_C
            / capacitance
            * (ion_amplitude / ion_voltage + electron_collection / electron_voltage)
        )
    else:
        electron_collection = electron_amplitude * (1.0 + potential / electron_voltage)
        ion_collection = ion_amplitude * math.exp(-potential / ion_voltage)
        rate_derivative = (
            -ELEMENTARY_CHARGE_C
            / capacitance
            * (ion_collection / ion_voltage + electron_amplitude / electron_voltage)
        )
    charge_rate = ion_collection - electron_collection

    relative_x = ion_velocity_x_m_s - velocity_x_m_s
    relative_y = ion_velocity_y_m_s - velocity_y_m_s
    ion_relative_speed = math.hypot(relative_x, relative_y)
    ion_mean_thermal_speed = math.sqrt(
        8.0 * BOLTZMANN_J_K * ion_temperature_K / (math.pi * ion_mass_kg)
    )
    radius_over_debye = radius_m / debye_length
    ion_drift_ratio = ion_relative_speed / ion_mean_thermal_speed
    if not _oml_derived_values_are_valid(
        debye_length,
        capacitance,
        potential,
        electron_voltage,
        ion_voltage,
        electron_amplitude,
        ion_amplitude,
        electron_collection,
        ion_collection,
        charge_rate,
        rate_derivative,
        radius_over_debye,
        ion_drift_ratio,
    ):
        return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    applicable = (
        radius_over_debye <= OML_MAX_RADIUS_OVER_DEBYE
        and ion_drift_ratio <= OML_MAX_ION_DRIFT_RATIO
    )
    return charge_rate, rate_derivative, applicable, ERROR_NONE


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _oml_shifted_charge_row(
    charge_number: float,
    radius_m: float,
    electron_density_m3: float,
    ion_density_m3: float,
    electron_temperature_K: float,
    ion_temperature_K: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    ion_velocity_x_m_s: float,
    ion_velocity_y_m_s: float,
    ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
) -> tuple[float, float, bool, int]:
    """Evaluate negative-grain OML for one shifted-Maxwellian ion species."""

    inverse_debye_square = (
        ELEMENTARY_CHARGE_C
        * ELEMENTARY_CHARGE_C
        / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
        * (electron_density_m3 / electron_temperature_K + ion_density_m3 / ion_temperature_K)
    )
    debye_length = 1.0 / math.sqrt(inverse_debye_square)
    capacitance = (
        4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * radius_m * (1.0 + radius_m / debye_length)
    )
    potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    electron_voltage = BOLTZMANN_J_K * electron_temperature_K / ELEMENTARY_CHARGE_C
    ion_voltage = BOLTZMANN_J_K * ion_temperature_K / ELEMENTARY_CHARGE_C
    electron_amplitude = (
        math.pi
        * radius_m
        * radius_m
        * electron_density_m3
        * math.sqrt(8.0 * BOLTZMANN_J_K * electron_temperature_K / (math.pi * ELECTRON_MASS_KG))
    )
    ion_amplitude = (
        math.pi
        * radius_m
        * radius_m
        * ion_density_m3
        * math.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_K / (math.pi * ion_mass_kg))
    )
    relative_x = ion_velocity_x_m_s - velocity_x_m_s
    relative_y = ion_velocity_y_m_s - velocity_y_m_s
    ion_relative_speed = math.hypot(relative_x, relative_y)
    ion_mean_thermal_speed = math.sqrt(
        8.0 * BOLTZMANN_J_K * ion_temperature_K / (math.pi * ion_mass_kg)
    )
    ion_drift_ratio = ion_relative_speed / ion_mean_thermal_speed
    shift_ratio = ion_drift_ratio * math.sqrt(8.0 / math.pi)
    neutral_factor, attraction_factor = _shifted_ion_factors_row(shift_ratio)
    electron_collection = electron_amplitude * math.exp(potential / electron_voltage)
    ion_collection = ion_amplitude * (neutral_factor - potential / ion_voltage * attraction_factor)
    charge_rate = ion_collection - electron_collection
    rate_derivative = (
        -ELEMENTARY_CHARGE_C
        / capacitance
        * (ion_amplitude * attraction_factor / ion_voltage + electron_collection / electron_voltage)
    )
    radius_over_debye = radius_m / debye_length
    if potential > 0.0 or not _oml_derived_values_are_valid(
        debye_length,
        capacitance,
        potential,
        electron_voltage,
        ion_voltage,
        electron_amplitude,
        ion_amplitude,
        electron_collection,
        ion_collection,
        charge_rate,
        rate_derivative,
        radius_over_debye,
        ion_drift_ratio,
    ):
        return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    applicable = (
        radius_over_debye <= OML_MAX_RADIUS_OVER_DEBYE
        and ion_drift_ratio <= maximum_ion_drift_ratio
        and ion_amplitude * neutral_factor <= electron_amplitude
    )
    return charge_rate, rate_derivative, applicable, ERROR_NONE


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _aggregate_relative_drift_charge_row(
    charge_number: float,
    radius_m: float,
    electron_density_m3: float,
    ion_density_m3: float,
    electron_thermal_voltage_V: float,
    ion_thermal_voltage_V: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    ion_velocity_x_m_s: float,
    ion_velocity_y_m_s: float,
    effective_ion_mass_kg: float,
    screening_length_m: float,
    maximum_relative_ion_speed_m_s: float,
) -> tuple[float, float, bool, int]:
    """Evaluate the regularized aggregate two-current charge revision."""

    if not _aggregate_charge_inputs_are_valid(
        radius_m,
        electron_density_m3,
        ion_density_m3,
        electron_thermal_voltage_V,
        ion_thermal_voltage_V,
        effective_ion_mass_kg,
        screening_length_m,
        maximum_relative_ion_speed_m_s,
    ):
        return 0.0, 0.0, True, ERROR_DERIVED_VALUE

    relative_x = ion_velocity_x_m_s - velocity_x_m_s
    relative_y = ion_velocity_y_m_s - velocity_y_m_s
    relative_speed = math.hypot(relative_x, relative_y)
    effective_speed_squared = (
        relative_speed * relative_speed
        + 8.0 * ELEMENTARY_CHARGE_C * ion_thermal_voltage_V / (math.pi * effective_ion_mass_kg)
        + AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    effective_speed = math.sqrt(effective_speed_squared)
    effective_ion_energy = max(
        effective_ion_mass_kg * effective_speed_squared / (2.0 * ELEMENTARY_CHARGE_C),
        AGGREGATE_MINIMUM_ION_ENERGY_V,
    )
    effective_screening_length = max(radius_m, screening_length_m)
    capacitance = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * radius_m
        * (1.0 + radius_m / effective_screening_length)
    )
    surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    ion_amplitude = math.pi * radius_m * radius_m * ion_density_m3 * effective_speed
    electron_amplitude = (
        math.pi
        * radius_m
        * radius_m
        * electron_density_m3
        * math.sqrt(
            8.0 * ELEMENTARY_CHARGE_C * electron_thermal_voltage_V / (math.pi * ELECTRON_MASS_KG)
        )
    )

    if surface_potential <= 0.0:
        ion_factor = 1.0 - surface_potential / effective_ion_energy
        electron_exponent = surface_potential / electron_thermal_voltage_V
        clipped_electron_exponent = min(
            AGGREGATE_EXPONENT_MAX,
            max(AGGREGATE_EXPONENT_MIN, electron_exponent),
        )
        electron_factor = math.exp(clipped_electron_exponent)
        electron_factor_derivative = (
            electron_factor * ELEMENTARY_CHARGE_C / capacitance / electron_thermal_voltage_V
            if AGGREGATE_EXPONENT_MIN < electron_exponent < AGGREGATE_EXPONENT_MAX
            else 0.0
        )
        ion_factor_derivative = -ELEMENTARY_CHARGE_C / capacitance / effective_ion_energy
    else:
        ion_exponent = -surface_potential / effective_ion_energy
        clipped_ion_exponent = min(
            AGGREGATE_EXPONENT_MAX,
            max(AGGREGATE_EXPONENT_MIN, ion_exponent),
        )
        ion_factor = math.exp(clipped_ion_exponent)
        electron_factor = 1.0 + surface_potential / electron_thermal_voltage_V
        ion_factor_derivative = (
            -ion_factor * ELEMENTARY_CHARGE_C / capacitance / effective_ion_energy
            if AGGREGATE_EXPONENT_MIN < ion_exponent < AGGREGATE_EXPONENT_MAX
            else 0.0
        )
        electron_factor_derivative = ELEMENTARY_CHARGE_C / capacitance / electron_thermal_voltage_V
    ion_collection = ion_amplitude * ion_factor
    electron_collection = electron_amplitude * electron_factor
    charge_rate = ion_collection - electron_collection
    rate_derivative = (
        ion_amplitude * ion_factor_derivative - electron_amplitude * electron_factor_derivative
    )

    if not _aggregate_charge_derived_values_are_valid(
        relative_speed,
        effective_speed,
        effective_ion_energy,
        effective_screening_length,
        capacitance,
        surface_potential,
        ion_collection,
        electron_collection,
        charge_rate,
        rate_derivative,
    ):
        return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    return (
        charge_rate,
        rate_derivative,
        relative_speed <= maximum_relative_ion_speed_m_s,
        ERROR_NONE,
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _aggregate_relative_drift_three_current_charge_row(
    charge_number: float,
    radius_m: float,
    electron_density_m3: float,
    positive_ion_density_m3: float,
    negative_ion_density_m3: float,
    electron_thermal_voltage_V: float,
    positive_ion_thermal_voltage_V: float,
    negative_ion_thermal_voltage_V: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    positive_ion_velocity_x_m_s: float,
    positive_ion_velocity_y_m_s: float,
    negative_ion_velocity_x_m_s: float,
    negative_ion_velocity_y_m_s: float,
    effective_positive_ion_mass_kg: float,
    effective_negative_ion_mass_kg: float,
    screening_length_m: float,
    maximum_relative_ion_speed_m_s: float,
) -> tuple[float, float, bool, int]:
    """Extend two-current charge by one aggregate negative-ion collection term."""

    base_rate, base_derivative, positive_applicable, error = _aggregate_relative_drift_charge_row(
        charge_number,
        radius_m,
        electron_density_m3,
        positive_ion_density_m3,
        electron_thermal_voltage_V,
        positive_ion_thermal_voltage_V,
        velocity_x_m_s,
        velocity_y_m_s,
        positive_ion_velocity_x_m_s,
        positive_ion_velocity_y_m_s,
        effective_positive_ion_mass_kg,
        screening_length_m,
        maximum_relative_ion_speed_m_s,
    )
    if error != ERROR_NONE:
        return 0.0, 0.0, True, error
    if not _aggregate_negative_ion_inputs_are_valid(
        negative_ion_density_m3,
        negative_ion_thermal_voltage_V,
        negative_ion_velocity_x_m_s,
        negative_ion_velocity_y_m_s,
        effective_negative_ion_mass_kg,
    ):
        return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    collection, collection_derivative, relative_speed, valid = (
        _aggregate_negative_ion_collection_row(
            charge_number,
            radius_m,
            negative_ion_density_m3,
            negative_ion_thermal_voltage_V,
            velocity_x_m_s,
            velocity_y_m_s,
            negative_ion_velocity_x_m_s,
            negative_ion_velocity_y_m_s,
            effective_negative_ion_mass_kg,
            screening_length_m,
        )
    )
    if not valid:
        return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    charge_rate = base_rate - collection
    rate_derivative = base_derivative - collection_derivative
    if (
        not math.isfinite(charge_rate)
        or not math.isfinite(rate_derivative)
        or rate_derivative >= 0.0
    ):
        return 0.0, 0.0, True, ERROR_DERIVED_VALUE
    return (
        charge_rate,
        rate_derivative,
        positive_applicable and relative_speed <= maximum_relative_ion_speed_m_s,
        ERROR_NONE,
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _aggregate_negative_ion_inputs_are_valid(
    density_m3: float,
    thermal_voltage_V: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    effective_mass_kg: float,
) -> bool:
    return (
        math.isfinite(density_m3)
        and density_m3 >= 0.0
        and math.isfinite(thermal_voltage_V)
        and thermal_voltage_V > 0.0
        and math.isfinite(velocity_x_m_s)
        and math.isfinite(velocity_y_m_s)
        and math.isfinite(effective_mass_kg)
        and effective_mass_kg > 0.0
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _aggregate_negative_ion_collection_row(
    charge_number: float,
    radius_m: float,
    density_m3: float,
    thermal_voltage_V: float,
    particle_velocity_x_m_s: float,
    particle_velocity_y_m_s: float,
    ion_velocity_x_m_s: float,
    ion_velocity_y_m_s: float,
    effective_mass_kg: float,
    screening_length_m: float,
) -> tuple[float, float, float, bool]:
    relative_speed = math.hypot(
        ion_velocity_x_m_s - particle_velocity_x_m_s,
        ion_velocity_y_m_s - particle_velocity_y_m_s,
    )
    effective_speed_squared = (
        relative_speed * relative_speed
        + 8.0 * ELEMENTARY_CHARGE_C * thermal_voltage_V / (math.pi * effective_mass_kg)
        + AGGREGATE_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    effective_speed = math.sqrt(effective_speed_squared)
    effective_energy = max(
        effective_mass_kg * effective_speed_squared / (2.0 * ELEMENTARY_CHARGE_C),
        AGGREGATE_MINIMUM_ION_ENERGY_V,
    )
    effective_screening_length = max(radius_m, screening_length_m)
    capacitance = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * radius_m
        * (1.0 + radius_m / effective_screening_length)
    )
    potential_per_charge = ELEMENTARY_CHARGE_C / capacitance
    surface_potential = charge_number * potential_per_charge
    amplitude = math.pi * radius_m * radius_m * density_m3 * effective_speed
    argument = surface_potential / effective_energy
    if surface_potential <= 0.0:
        factor = math.exp(min(AGGREGATE_EXPONENT_MAX, max(AGGREGATE_EXPONENT_MIN, argument)))
        factor_derivative = (
            factor * potential_per_charge / effective_energy
            if AGGREGATE_EXPONENT_MIN < argument < AGGREGATE_EXPONENT_MAX
            else 0.0
        )
    else:
        factor = 1.0 + argument
        factor_derivative = potential_per_charge / effective_energy
    collection = amplitude * factor
    collection_derivative = amplitude * factor_derivative
    return (
        collection,
        collection_derivative,
        relative_speed,
        (
            math.isfinite(relative_speed)
            and relative_speed >= 0.0
            and math.isfinite(effective_speed)
            and effective_speed > 0.0
            and math.isfinite(effective_energy)
            and effective_energy > 0.0
            and math.isfinite(collection)
            and collection >= 0.0
            and math.isfinite(collection_derivative)
            and collection_derivative >= 0.0
        ),
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _aggregate_charge_inputs_are_valid(
    radius_m: float,
    electron_density_m3: float,
    ion_density_m3: float,
    electron_thermal_voltage_V: float,
    ion_thermal_voltage_V: float,
    effective_ion_mass_kg: float,
    screening_length_m: float,
    maximum_relative_ion_speed_m_s: float,
) -> bool:
    """Validate the positive scalar inputs owned by the aggregate revision."""

    positive_values = (
        radius_m,
        electron_density_m3,
        ion_density_m3,
        electron_thermal_voltage_V,
        ion_thermal_voltage_V,
        effective_ion_mass_kg,
        screening_length_m,
        maximum_relative_ion_speed_m_s,
    )
    for value in positive_values:
        if not math.isfinite(value) or value <= 0.0:
            return False
    return True


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _aggregate_charge_derived_values_are_valid(
    relative_speed_m_s: float,
    effective_speed_m_s: float,
    effective_ion_energy_V: float,
    effective_screening_length_m: float,
    capacitance_F: float,
    surface_potential_V: float,
    ion_collection_number_s: float,
    electron_collection_number_s: float,
    charge_rate_number_s: float,
    charge_rate_derivative_s_inv: float,
) -> bool:
    """Validate the aggregate revision's derived scalar quantities."""

    positive_values = (
        effective_speed_m_s,
        effective_ion_energy_V,
        effective_screening_length_m,
        capacitance_F,
    )
    for value in positive_values:
        if not math.isfinite(value) or value <= 0.0:
            return False
    nonnegative_values = (
        relative_speed_m_s,
        ion_collection_number_s,
        electron_collection_number_s,
    )
    for value in nonnegative_values:
        if not math.isfinite(value) or value < 0.0:
            return False
    return (
        math.isfinite(surface_potential_V)
        and math.isfinite(charge_rate_number_s)
        and math.isfinite(charge_rate_derivative_s_inv)
        and charge_rate_derivative_s_inv < 0.0
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _shifted_ion_factors_row(thermal_shift_ratio: float) -> tuple[float, float]:
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


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _oml_derived_values_are_valid(
    debye_length_m: float,
    capacitance_F: float,
    potential_V: float,
    electron_voltage_V: float,
    ion_voltage_V: float,
    electron_amplitude_number_s: float,
    ion_amplitude_number_s: float,
    electron_collection_number_s: float,
    ion_collection_number_s: float,
    charge_rate_number_s: float,
    charge_rate_derivative_s_inv: float,
    radius_over_debye: float,
    ion_drift_ratio: float,
) -> bool:
    positive_values = (
        debye_length_m,
        capacitance_F,
        electron_voltage_V,
        ion_voltage_V,
        electron_amplitude_number_s,
        ion_amplitude_number_s,
    )
    for value in positive_values:
        if not math.isfinite(value) or value <= 0.0:
            return False
    nonnegative_values = (electron_collection_number_s, ion_collection_number_s)
    for value in nonnegative_values:
        if not math.isfinite(value) or value < 0.0:
            return False
    finite_values = (
        potential_V,
        charge_rate_number_s,
        charge_rate_derivative_s_inv,
        radius_over_debye,
        ion_drift_ratio,
    )
    for value in finite_values:
        if not math.isfinite(value):
            return False
    return charge_rate_derivative_s_inv < 0.0


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _epstein_drag_row(
    mass_kg: float,
    diameter_m: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    gas_velocity_x_m_s: float,
    gas_velocity_y_m_s: float,
    gas_density_kg_m3: float,
    gas_temperature_K: float,
    gas_mean_free_path_m: float,
    gas_molecular_mass_kg: float,
    delta: float,
    maximum_speed_ratio: float,
) -> tuple[float, float, float, bool, int]:
    radius = 0.5 * diameter_m
    mean_thermal_speed = math.sqrt(
        8.0 * BOLTZMANN_J_K * gas_temperature_K / (math.pi * gas_molecular_mass_kg)
    )
    beta = (4.0 * math.pi / 3.0) * radius * radius * gas_density_kg_m3 * mean_thermal_speed * delta
    rate = beta / mass_kg
    if not math.isfinite(rate) or rate <= 0.0:
        return 0.0, 0.0, 0.0, True, ERROR_RATE
    relative_x = gas_velocity_x_m_s - velocity_x_m_s
    relative_y = gas_velocity_y_m_s - velocity_y_m_s
    relative_speed = math.sqrt(relative_x * relative_x + relative_y * relative_y)
    lambda_over_radius = gas_mean_free_path_m / radius
    speed_ratio = relative_speed / mean_thermal_speed
    if not (
        math.isfinite(mean_thermal_speed)
        and mean_thermal_speed > 0.0
        and math.isfinite(relative_speed)
        and math.isfinite(lambda_over_radius)
        and math.isfinite(speed_ratio)
    ):
        return 0.0, 0.0, 0.0, True, ERROR_DERIVED_VALUE
    applicable = (
        lambda_over_radius >= EPSTEIN_MIN_LAMBDA_OVER_RADIUS and speed_ratio <= maximum_speed_ratio
    )
    return relative_x * rate, relative_y * rate, rate, applicable, ERROR_NONE


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _epstein_finite_speed_drag_row(
    mass_kg: float,
    diameter_m: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    gas_velocity_x_m_s: float,
    gas_velocity_y_m_s: float,
    gas_density_kg_m3: float,
    gas_temperature_K: float,
    gas_mean_free_path_m: float,
    gas_molecular_mass_kg: float,
    diffuse_reflection_fraction: float,
    maximum_speed_ratio: float,
) -> tuple[float, float, float, bool, int]:
    radius = 0.5 * diameter_m
    mean_thermal_speed = math.sqrt(
        8.0 * BOLTZMANN_J_K * gas_temperature_K / (math.pi * gas_molecular_mass_kg)
    )
    base_beta = (4.0 * math.pi / 3.0) * radius * radius * gas_density_kg_m3 * mean_thermal_speed
    base_rate = base_beta / mass_kg
    relative_x = gas_velocity_x_m_s - velocity_x_m_s
    relative_y = gas_velocity_y_m_s - velocity_y_m_s
    relative_speed = math.hypot(relative_x, relative_y)
    most_probable_speed = math.sqrt(2.0 * BOLTZMANN_J_K * gas_temperature_K / gas_molecular_mass_kg)
    speed_ratio = relative_speed / most_probable_speed
    specular_factor = _epstein_finite_speed_factor_row(speed_ratio)
    rate = base_rate * (specular_factor + diffuse_reflection_fraction * math.pi / 8.0)
    lambda_over_radius = gas_mean_free_path_m / radius
    if not (
        math.isfinite(rate)
        and rate > 0.0
        and math.isfinite(relative_speed)
        and math.isfinite(most_probable_speed)
        and most_probable_speed > 0.0
        and math.isfinite(speed_ratio)
        and math.isfinite(lambda_over_radius)
    ):
        return 0.0, 0.0, 0.0, True, ERROR_DERIVED_VALUE
    applicable = (
        lambda_over_radius >= EPSTEIN_MIN_LAMBDA_OVER_RADIUS and speed_ratio <= maximum_speed_ratio
    )
    return relative_x * rate, relative_y * rate, rate, applicable, ERROR_NONE


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _epstein_finite_speed_factor_row(speed_ratio: float) -> float:
    if speed_ratio <= EPSTEIN_FINITE_SPEED_SERIES_LIMIT:
        square = speed_ratio * speed_ratio
        return 1.0 + square * (
            1.0 / 5.0 + square * (-1.0 / 70.0 + square * (1.0 / 630.0 - square / 5544.0))
        )
    inverse = 1.0 / speed_ratio
    inverse_square = inverse * inverse
    return 3.0 / 16.0 * (2.0 + inverse_square) * math.exp(
        -speed_ratio * speed_ratio
    ) + 3.0 * math.sqrt(math.pi) / 32.0 * speed_ratio * (
        4.0 + 4.0 * inverse_square - inverse_square * inverse_square
    ) * math.erf(speed_ratio)


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _stokes_cunningham_drag_row(
    mass_kg: float,
    diameter_m: float,
    velocity_x_m_s: float,
    velocity_y_m_s: float,
    gas_velocity_x_m_s: float,
    gas_velocity_y_m_s: float,
    gas_density_kg_m3: float,
    gas_dynamic_viscosity_Pa_s: float,
    gas_mean_free_path_m: float,
) -> tuple[float, float, float, bool, int]:
    knudsen_radius = 2.0 * gas_mean_free_path_m / diameter_m
    slip_correction = 1.0 + knudsen_radius * (
        _ALLEN_RAABE_A1 + _ALLEN_RAABE_A2 * math.exp(-_ALLEN_RAABE_A3 / knudsen_radius)
    )
    rate = 3.0 * math.pi * gas_dynamic_viscosity_Pa_s * diameter_m / (slip_correction * mass_kg)
    if not math.isfinite(rate) or rate <= 0.0:
        return 0.0, 0.0, 0.0, True, ERROR_RATE
    relative_x = gas_velocity_x_m_s - velocity_x_m_s
    relative_y = gas_velocity_y_m_s - velocity_y_m_s
    relative_speed = math.sqrt(relative_x * relative_x + relative_y * relative_y)
    reynolds = gas_density_kg_m3 * diameter_m * relative_speed / gas_dynamic_viscosity_Pa_s
    if not (
        math.isfinite(knudsen_radius)
        and math.isfinite(slip_correction)
        and slip_correction > 0.0
        and math.isfinite(relative_speed)
        and math.isfinite(reynolds)
    ):
        return 0.0, 0.0, 0.0, True, ERROR_DERIVED_VALUE
    applicable = (
        knudsen_radius >= STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS
        and knudsen_radius <= STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS
        and reynolds <= STOKES_CUNNINGHAM_MAX_REYNOLDS
    )
    return relative_x * rate, relative_y * rate, rate, applicable, ERROR_NONE
