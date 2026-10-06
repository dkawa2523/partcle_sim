"""Pure vectorized force formulas over already sampled primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..numerical_status import NUMERICAL_STATUS_OK, PHYSICS_NUMERICAL_FAILURE

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type UInt8Array = NDArray[np.uint8]

BOLTZMANN_J_K = 1.380649e-23
ELEMENTARY_CHARGE_C = 1.602176634e-19
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
EPSTEIN_MIN_LAMBDA_OVER_RADIUS = 10.0
EPSTEIN_MAX_SPEED_OVER_MEAN_THERMAL = 0.1
EPSTEIN_FINITE_SPEED_SERIES_LIMIT = 0.1
STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS = 0.03
STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS = 7.2
STOKES_CUNNINGHAM_MAX_REYNOLDS = 0.1
ION_DRAG_MAX_SCALE_OVER_DEBYE = 0.1
ION_DRAG_MIN_MEAN_FREE_PATH_OVER_DEBYE = 10.0
AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S = 1.0
AGGREGATE_ION_DRAG_CHARGE_SQUARE_REGULARIZATION = 1.0e-20
IMAGE_ION_DRAG_LOG_ARGUMENT_OFFSET = 1.0e-12
IMAGE_ION_DRAG_ELECTRIC_FIELD_SQUARE_FLOOR_V2_M2 = 1.0
WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS = 10.0
WALDMANN_GALLIS_MAX_SPEED_OVER_MEAN_THERMAL = 0.1
RAREFIED_VORTICITY_LIFT_MIN_MEAN_FREE_PATH_OVER_RADIUS = 10.0
_ALLEN_RAABE_A1 = 1.142
_ALLEN_RAABE_A2 = 0.558
_ALLEN_RAABE_A3 = 0.999
_BOUND_ROUNDOFF_FACTOR = 1.0 + 64.0 * np.finfo(np.float64).eps

CONTINUOUS_APPLICABILITY_OK = np.uint8(0)
CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE = np.uint8(1)


class PhysicsEvaluationError(ValueError):
    """A primitive or derived physics value cannot define a finite rate."""


@dataclass(frozen=True, slots=True)
class LinearRelaxation:
    """Instantaneous positive relaxation form and its applicability verdict."""

    rate_s_inv: FloatArray
    target_velocity_m_s: FloatArray
    applicable: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class BarnesIonDragEvaluation:
    """Local acceleration and inspectable cross sections for Barnes ion drag."""

    acceleration_m_s2: FloatArray
    collection_cross_section_m2: FloatArray
    orbital_cross_section_m2: FloatArray
    debye_length_m: FloatArray
    surface_potential_V: FloatArray
    collection_impact_parameter_m: FloatArray
    orbital_impact_parameter_m: FloatArray
    coulomb_logarithm: FloatArray
    ion_drift_ratio: FloatArray
    applicable: BoolArray


@dataclass(frozen=True, slots=True)
class BarnesIonDragGlobalBounds:
    """Prepared per-particle enclosure and velocity-independent applicability."""

    acceleration_abs_upper_m_s2: FloatArray
    static_applicable: BoolArray
    debye_length_lower_m: float
    debye_length_upper_m: float
    positive_ion_temperature_lower_K: float


@dataclass(frozen=True, slots=True)
class RelativeFlowScreenedIonDragEvaluation:
    """Inspectable result of the aggregate relative-flow sensitivity revision."""

    acceleration_m_s2: FloatArray
    collection_cross_section_m2: FloatArray
    orbital_cross_section_m2: FloatArray
    screening_radius_m: FloatArray
    surface_potential_V: FloatArray
    effective_speed_m_s: FloatArray
    applicable: BoolArray


@dataclass(frozen=True, slots=True)
class ElectricFieldDirectedImageIonDragEvaluation:
    """Inspectable result of the electric-field-directed image sensitivity revision."""

    acceleration_m_s2: FloatArray
    collection_cross_section_m2: FloatArray
    orbital_cross_section_m2: FloatArray
    surface_potential_V: FloatArray
    ion_speed_m_s: FloatArray
    effective_speed_m_s: FloatArray
    applicable: BoolArray


@dataclass(frozen=True, slots=True)
class WaldmannGallisThermophoresisEvaluation:
    """Local acceleration and applicability of the heat-flux thermophoresis model."""

    acceleration_m_s2: FloatArray
    mean_thermal_speed_m_s: FloatArray
    mean_free_path_over_radius: FloatArray
    relative_speed_ratio: FloatArray
    applicable: BoolArray


@dataclass(frozen=True, slots=True)
class WaldmannGallisThermophoresisGlobalBounds:
    """Prepared component acceleration bound and particle-size applicability."""

    acceleration_abs_upper_m_s2: FloatArray
    static_applicable: BoolArray


@dataclass(frozen=True, slots=True)
class RarefiedVorticityLiftEvaluation:
    """Local acceleration and applicability of the RZ lift sensitivity model."""

    acceleration_m_s2: FloatArray
    coupling_rate_s_inv: FloatArray
    mean_free_path_over_radius: FloatArray
    applicable: BoolArray


@dataclass(frozen=True, slots=True)
class RarefiedVorticityLiftGlobalBounds:
    """Prepared velocity-coupling and static-applicability bounds."""

    coupling_rate_abs_upper_s_inv: FloatArray
    gas_velocity_abs_upper_m_s: FloatArray
    static_applicable: BoolArray


def rarefied_vorticity_lift(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    velocity_m_s: FloatArray,
    gas_velocity_m_s: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_mean_free_path_m: FloatArray,
    azimuthal_gas_vorticity_s_inv: FloatArray,
    lift_coefficient: float,
) -> RarefiedVorticityLiftEvaluation:
    """Evaluate the documented axisymmetric rarefied-vorticity sensitivity."""

    count = _common_count(
        mass_kg,
        drag_diameter_m,
        gas_density_kg_m3,
        gas_mean_free_path_m,
        azimuthal_gas_vorticity_s_inv,
    )
    if velocity_m_s.shape != (count, 2) or gas_velocity_m_s.shape != (count, 2):
        raise PhysicsEvaluationError("rarefied-vorticity lift vectors must have shape [N, 2]")
    _require_positive(mass_kg, "mass_kg")
    _require_positive(drag_diameter_m, "drag_diameter_m")
    _require_positive(gas_density_kg_m3, "gas_density")
    _require_positive(gas_mean_free_path_m, "gas_mean_free_path")
    _require_finite(azimuthal_gas_vorticity_s_inv, "azimuthal_gas_vorticity")
    coefficient = _finite_positive_scalar(lift_coefficient, "lift_coefficient")
    if not bool(np.isfinite(velocity_m_s).all() and np.isfinite(gas_velocity_m_s).all()):
        raise PhysicsEvaluationError("rarefied-vorticity lift velocities must be finite")

    radius_m = 0.5 * drag_diameter_m
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coupling_rate = (
            coefficient
            * math.pi
            * gas_density_kg_m3
            * gas_mean_free_path_m
            * radius_m**2
            * azimuthal_gas_vorticity_s_inv
            / mass_kg
        )
        relative_velocity = gas_velocity_m_s - velocity_m_s
        acceleration = np.empty((count, 2), dtype=np.float64)
        acceleration[:, 0] = coupling_rate * relative_velocity[:, 1]
        acceleration[:, 1] = -coupling_rate * relative_velocity[:, 0]
        mean_free_path_over_radius = gas_mean_free_path_m / radius_m
    if not bool(
        np.isfinite(coupling_rate).all()
        and np.isfinite(acceleration).all()
        and np.isfinite(mean_free_path_over_radius).all()
    ):
        raise PhysicsEvaluationError("rarefied-vorticity lift produced a non-finite value")
    applicable = (
        mean_free_path_over_radius >= RAREFIED_VORTICITY_LIFT_MIN_MEAN_FREE_PATH_OVER_RADIUS
    )
    return RarefiedVorticityLiftEvaluation(
        acceleration,
        coupling_rate,
        mean_free_path_over_radius,
        applicable,
    )


def rarefied_vorticity_lift_global_bounds(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    gas_density_upper_kg_m3: float,
    gas_mean_free_path_lower_m: float,
    gas_mean_free_path_upper_m: float,
    azimuthal_gas_vorticity_abs_upper_s_inv: float,
    gas_velocity_abs_upper_m_s: FloatArray,
    lift_coefficient: float,
) -> RarefiedVorticityLiftGlobalBounds:
    """Prepare a global cross-velocity bound and the high-Kn certificate."""

    _common_count(mass_kg, drag_diameter_m)
    _require_positive(mass_kg, "mass_kg")
    _require_positive(drag_diameter_m, "drag_diameter_m")
    density_upper = _finite_positive_scalar(
        gas_density_upper_kg_m3,
        "gas_density_upper_kg_m3",
    )
    mean_free_path_lower = _finite_positive_scalar(
        gas_mean_free_path_lower_m,
        "gas_mean_free_path_lower_m",
    )
    mean_free_path_upper = _finite_positive_scalar(
        gas_mean_free_path_upper_m,
        "gas_mean_free_path_upper_m",
    )
    if mean_free_path_lower > mean_free_path_upper:
        raise PhysicsEvaluationError("gas mean-free-path bounds are reversed")
    vorticity_upper = float(azimuthal_gas_vorticity_abs_upper_s_inv)
    if not math.isfinite(vorticity_upper) or vorticity_upper < 0.0:
        raise PhysicsEvaluationError(
            "azimuthal_gas_vorticity_abs_upper_s_inv must be finite and nonnegative"
        )
    gas_velocity_upper = _component_abs_upper(
        gas_velocity_abs_upper_m_s,
        "gas_velocity_abs_upper_m_s",
    )
    coefficient = _finite_positive_scalar(lift_coefficient, "lift_coefficient")

    radius_m = 0.5 * drag_diameter_m
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coupling_upper = np.nextafter(
            coefficient
            * math.pi
            * density_upper
            * mean_free_path_upper
            * radius_m**2
            * vorticity_upper
            / mass_kg
            * _BOUND_ROUNDOFF_FACTOR,
            np.inf,
        )
        minimum_knudsen = np.nextafter(mean_free_path_lower / radius_m, -np.inf)
    if not bool(
        np.isfinite(coupling_upper).all()
        and (coupling_upper >= 0.0).all()
        and np.isfinite(minimum_knudsen).all()
    ):
        raise PhysicsEvaluationError("rarefied-vorticity lift global bound is not finite")
    return RarefiedVorticityLiftGlobalBounds(
        coupling_upper,
        gas_velocity_upper,
        minimum_knudsen >= RAREFIED_VORTICITY_LIFT_MIN_MEAN_FREE_PATH_OVER_RADIUS,
    )


def rarefied_vorticity_lift_acceleration_abs_upper(
    *,
    coupling_rate_abs_upper_s_inv: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
) -> FloatArray:
    """Bound the RZ lift acceleration over a component velocity box."""

    result, status = rarefied_vorticity_lift_acceleration_abs_upper_batch(
        coupling_rate_abs_upper_s_inv=coupling_rate_abs_upper_s_inv,
        gas_velocity_abs_upper_m_s=gas_velocity_abs_upper_m_s,
        velocity_abs_upper_m_s=velocity_abs_upper_m_s,
    )
    if bool(np.any(status != NUMERICAL_STATUS_OK)):
        raise PhysicsEvaluationError("rarefied-vorticity lift bound is not finite")
    return result


def rarefied_vorticity_lift_acceleration_abs_upper_batch(
    *,
    coupling_rate_abs_upper_s_inv: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    numerical_status: UInt8Array | None = None,
) -> tuple[FloatArray, UInt8Array]:
    """Bound the cross-coupled lift while preserving row-local failures."""

    count = _common_count(coupling_rate_abs_upper_s_inv)
    _require_nonnegative(coupling_rate_abs_upper_s_inv, "coupling_rate_abs_upper_s_inv")
    gas_velocity = _component_abs_upper(
        gas_velocity_abs_upper_m_s,
        "gas_velocity_abs_upper_m_s",
    )
    velocity, finite_velocity = _continuous_velocity_bound(velocity_abs_upper_m_s, count)
    status = _force_numerical_status(numerical_status, count)
    status[(status == NUMERICAL_STATUS_OK) & ~finite_velocity] = PHYSICS_NUMERICAL_FAILURE
    safe_velocity = velocity.copy()
    safe_velocity[status != NUMERICAL_STATUS_OK] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        relative_component = np.nextafter(
            gas_velocity[None, :] + safe_velocity,
            np.inf,
        )
        result = np.empty((count, 2), dtype=np.float64)
        result[:, 0] = coupling_rate_abs_upper_s_inv * relative_component[:, 1]
        result[:, 1] = coupling_rate_abs_upper_s_inv * relative_component[:, 0]
        result = np.nextafter(result * _BOUND_ROUNDOFF_FACTOR, np.inf)
    finite_result = np.isfinite(result).all(axis=1)
    status[(status == NUMERICAL_STATUS_OK) & ~finite_result] = PHYSICS_NUMERICAL_FAILURE
    result[status != NUMERICAL_STATUS_OK] = 0.0
    return result, status


def waldmann_gallis_thermophoresis(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    velocity_m_s: FloatArray,
    gas_velocity_m_s: FloatArray,
    gas_temperature_K: FloatArray,
    gas_translational_heat_flux_W_m2: FloatArray,
    gas_mean_free_path_m: FloatArray,
    gas_molecular_mass_kg: float,
    maximum_speed_ratio: float = WALDMANN_GALLIS_MAX_SPEED_OVER_MEAN_THERMAL,
) -> WaldmannGallisThermophoresisEvaluation:
    """Evaluate the shared free-molecular heat-flux formula and declared gate."""

    count = _common_count(
        mass_kg,
        drag_diameter_m,
        gas_temperature_K,
        gas_mean_free_path_m,
    )
    if (
        velocity_m_s.shape != (count, 2)
        or gas_velocity_m_s.shape != (count, 2)
        or gas_translational_heat_flux_W_m2.shape != (count, 2)
    ):
        raise PhysicsEvaluationError("Waldmann--Gallis vector inputs must have shape [N, 2]")
    _require_positive(mass_kg, "mass_kg")
    _require_positive(drag_diameter_m, "drag_diameter_m")
    _require_positive(gas_temperature_K, "gas_temperature")
    _require_positive(gas_mean_free_path_m, "gas_mean_free_path")
    if not math.isfinite(gas_molecular_mass_kg) or gas_molecular_mass_kg <= 0.0:
        raise PhysicsEvaluationError("gas_molecular_mass_kg must be positive and finite")
    speed_ratio_limit = _sensitivity_maximum_speed_ratio(maximum_speed_ratio)
    if not bool(
        np.isfinite(velocity_m_s).all()
        and np.isfinite(gas_velocity_m_s).all()
        and np.isfinite(gas_translational_heat_flux_W_m2).all()
    ):
        raise PhysicsEvaluationError("Waldmann--Gallis vector inputs must be finite")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        radius_m = 0.5 * drag_diameter_m
        mean_thermal_speed_m_s = np.sqrt(
            8.0 * BOLTZMANN_J_K * gas_temperature_K / (math.pi * gas_molecular_mass_kg)
        )
        acceleration_factor = (
            (32.0 / 15.0) * radius_m * radius_m / (mass_kg * mean_thermal_speed_m_s)
        )
        acceleration_m_s2 = acceleration_factor[:, None] * gas_translational_heat_flux_W_m2
        mean_free_path_over_radius = gas_mean_free_path_m / radius_m
        relative_speed = np.linalg.norm(gas_velocity_m_s - velocity_m_s, axis=1)
        relative_speed_ratio = relative_speed / mean_thermal_speed_m_s
    if not bool(
        np.isfinite(acceleration_m_s2).all()
        and np.isfinite(mean_thermal_speed_m_s).all()
        and np.isfinite(mean_free_path_over_radius).all()
        and np.isfinite(relative_speed_ratio).all()
    ):
        raise PhysicsEvaluationError(
            "Waldmann--Gallis evaluation produced a non-finite derived value"
        )
    applicable = (mean_free_path_over_radius >= WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS) & (
        relative_speed_ratio <= speed_ratio_limit
    )
    return WaldmannGallisThermophoresisEvaluation(
        acceleration_m_s2,
        mean_thermal_speed_m_s,
        mean_free_path_over_radius,
        relative_speed_ratio,
        applicable,
    )


def waldmann_gallis_global_bounds(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    gas_temperature_lower_K: float,
    gas_translational_heat_flux_abs_upper_W_m2: FloatArray,
    gas_mean_free_path_lower_m: float,
    gas_molecular_mass_kg: float,
) -> WaldmannGallisThermophoresisGlobalBounds:
    """Enclose thermophoretic acceleration over declared primitive extrema."""

    _common_count(mass_kg, drag_diameter_m)
    _require_positive(mass_kg, "mass_kg")
    _require_positive(drag_diameter_m, "drag_diameter_m")
    heat_flux = np.asarray(
        gas_translational_heat_flux_abs_upper_W_m2,
        dtype=np.float64,
    )
    scalar_values = (
        gas_temperature_lower_K,
        gas_mean_free_path_lower_m,
        gas_molecular_mass_kg,
    )
    if (
        heat_flux.shape != (2,)
        or not bool(np.isfinite(heat_flux).all())
        or bool((heat_flux < 0.0).any())
        or any(not math.isfinite(value) or value <= 0.0 for value in scalar_values)
    ):
        raise PhysicsEvaluationError("Waldmann--Gallis global bounds are invalid")

    mean_thermal_speed_lower = math.nextafter(
        math.sqrt(
            8.0 * BOLTZMANN_J_K * gas_temperature_lower_K / (math.pi * gas_molecular_mass_kg)
        ),
        0.0,
    )
    radius_m = 0.5 * drag_diameter_m
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factor = np.nextafter(
            (32.0 / 15.0) * radius_m * radius_m / (mass_kg * mean_thermal_speed_lower),
            np.inf,
        )
        acceleration_bound = np.nextafter(factor[:, None] * heat_flux[None, :], np.inf)
        minimum_knudsen = np.nextafter(
            gas_mean_free_path_lower_m / radius_m,
            -np.inf,
        )
    if not bool(
        np.isfinite(acceleration_bound).all()
        and (acceleration_bound >= 0.0).all()
        and np.isfinite(minimum_knudsen).all()
    ):
        raise PhysicsEvaluationError("Waldmann--Gallis global bound produced a non-finite value")
    return WaldmannGallisThermophoresisGlobalBounds(
        acceleration_bound,
        minimum_knudsen >= WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS,
    )


def waldmann_gallis_continuous_applicability_batch(
    *,
    static_applicable: BoolArray,
    velocity_abs_upper_m_s: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    gas_temperature_lower_K: float,
    gas_molecular_mass_kg: float,
    maximum_speed_ratio: float = WALDMANN_GALLIS_MAX_SPEED_OVER_MEAN_THERMAL,
) -> tuple[BoolArray, UInt8Array]:
    """Certify the low-relative-drift gate over one velocity/path box."""

    static = np.asarray(static_applicable, dtype=np.bool_)
    count = int(static.size)
    velocity = np.asarray(velocity_abs_upper_m_s, dtype=np.float64)
    gas_velocity = np.asarray(gas_velocity_abs_upper_m_s, dtype=np.float64)
    status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    valid_scalar = (
        math.isfinite(gas_temperature_lower_K)
        and gas_temperature_lower_K > 0.0
        and math.isfinite(gas_molecular_mass_kg)
        and gas_molecular_mass_kg > 0.0
    )
    if velocity.shape != (count, 2) or gas_velocity.shape != (2,) or not valid_scalar:
        raise PhysicsEvaluationError("Waldmann--Gallis continuous applicability bounds are invalid")
    speed_ratio_limit = _sensitivity_maximum_speed_ratio(maximum_speed_ratio)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_component = np.nextafter(velocity + gas_velocity[None, :], np.inf)
        relative_speed = np.nextafter(
            np.hypot(relative_component[:, 0], relative_component[:, 1]),
            np.inf,
        )
        mean_thermal_speed_lower = math.nextafter(
            math.sqrt(
                8.0 * BOLTZMANN_J_K * gas_temperature_lower_K / (math.pi * gas_molecular_mass_kg)
            ),
            0.0,
        )
        ratio = np.nextafter(relative_speed / mean_thermal_speed_lower, np.inf)
    numerical_ok = (
        np.isfinite(velocity).all(axis=1)
        & (velocity >= 0.0).all(axis=1)
        & np.isfinite(gas_velocity).all()
        & (gas_velocity >= 0.0).all()
        & np.isfinite(ratio)
    )
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    applicable = static & numerical_ok
    applicable &= ratio <= speed_ratio_limit
    return applicable, status


def epstein_linear_relaxation(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    velocity_m_s: FloatArray,
    gas_velocity_m_s: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_temperature_K: FloatArray,
    gas_mean_free_path_m: FloatArray,
    gas_molecular_mass_kg: float,
    delta: float,
    maximum_speed_ratio: float = EPSTEIN_MAX_SPEED_OVER_MEAN_THERMAL,
) -> LinearRelaxation:
    """Evaluate the shared linear-Epstein formula and declared applicability gate."""

    count = _common_count(
        mass_kg,
        drag_diameter_m,
        gas_density_kg_m3,
        gas_temperature_K,
        gas_mean_free_path_m,
    )
    if velocity_m_s.shape != (count, 2) or gas_velocity_m_s.shape != (count, 2):
        raise PhysicsEvaluationError("Epstein velocity inputs must have shape [N, 2]")
    rate_s_inv, mean_thermal_speed_m_s, radius_m = epstein_rate_s_inv(
        mass_kg=mass_kg,
        drag_diameter_m=drag_diameter_m,
        gas_density_kg_m3=gas_density_kg_m3,
        gas_temperature_K=gas_temperature_K,
        gas_molecular_mass_kg=gas_molecular_mass_kg,
        delta=delta,
    )
    _require_positive(gas_mean_free_path_m, "gas_mean_free_path")
    speed_ratio_limit = _sensitivity_maximum_speed_ratio(maximum_speed_ratio)
    if not bool(np.isfinite(velocity_m_s).all() and np.isfinite(gas_velocity_m_s).all()):
        raise PhysicsEvaluationError("Epstein velocity inputs must be finite")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_speed_m_s = np.linalg.norm(gas_velocity_m_s - velocity_m_s, axis=1)
        lambda_over_radius = gas_mean_free_path_m / radius_m
        speed_ratio = relative_speed_m_s / mean_thermal_speed_m_s
    if not bool(
        np.isfinite(rate_s_inv).all()
        and np.isfinite(relative_speed_m_s).all()
        and np.isfinite(lambda_over_radius).all()
        and np.isfinite(speed_ratio).all()
    ):
        raise PhysicsEvaluationError("Epstein evaluation produced a non-finite derived value")
    applicable = (lambda_over_radius >= EPSTEIN_MIN_LAMBDA_OVER_RADIUS) & (
        speed_ratio <= speed_ratio_limit
    )
    return LinearRelaxation(rate_s_inv, gas_velocity_m_s, applicable)


def epstein_rate_s_inv(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_temperature_K: FloatArray,
    gas_molecular_mass_kg: float,
    delta: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the Epstein relaxation rate, mean thermal speed, and radius."""

    _common_count(mass_kg, drag_diameter_m, gas_density_kg_m3, gas_temperature_K)
    _require_positive(mass_kg, "mass_kg")
    _require_positive(drag_diameter_m, "drag_diameter_m")
    _require_positive(gas_density_kg_m3, "gas_density")
    _require_positive(gas_temperature_K, "gas_temperature")
    if not math.isfinite(gas_molecular_mass_kg) or gas_molecular_mass_kg <= 0.0:
        raise PhysicsEvaluationError("gas_molecular_mass_kg must be positive and finite")
    if not math.isfinite(delta) or delta <= 0.0:
        raise PhysicsEvaluationError("Epstein delta must be positive and finite")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        radius_m = 0.5 * drag_diameter_m
        mean_thermal_speed_m_s = np.sqrt(
            8.0 * BOLTZMANN_J_K * gas_temperature_K / (math.pi * gas_molecular_mass_kg)
        )
        beta_kg_s = (
            (4.0 * math.pi / 3.0)
            * radius_m
            * radius_m
            * gas_density_kg_m3
            * mean_thermal_speed_m_s
            * delta
        )
        rate_s_inv = beta_kg_s / mass_kg
    if not bool(
        np.isfinite(rate_s_inv).all()
        and np.isfinite(mean_thermal_speed_m_s).all()
        and np.isfinite(radius_m).all()
        and (rate_s_inv > 0.0).all()
        and (mean_thermal_speed_m_s > 0.0).all()
        and (radius_m > 0.0).all()
    ):
        raise PhysicsEvaluationError("Epstein rate is not finite and positive")
    return rate_s_inv, mean_thermal_speed_m_s, radius_m


def epstein_finite_speed_relaxation(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    velocity_m_s: FloatArray,
    gas_velocity_m_s: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_temperature_K: FloatArray,
    gas_mean_free_path_m: FloatArray,
    gas_molecular_mass_kg: float,
    diffuse_reflection_fraction: float,
    maximum_speed_ratio: float,
) -> LinearRelaxation:
    """Evaluate the equal-temperature Maxwell mixed finite-speed sphere drag."""

    count = _common_count(
        mass_kg,
        drag_diameter_m,
        gas_density_kg_m3,
        gas_temperature_K,
        gas_mean_free_path_m,
    )
    if velocity_m_s.shape != (count, 2) or gas_velocity_m_s.shape != (count, 2):
        raise PhysicsEvaluationError("finite-speed Epstein velocity inputs must have shape [N, 2]")
    diffuse_fraction = _finite_nonnegative_scalar(
        diffuse_reflection_fraction,
        "diffuse_reflection_fraction",
    )
    if diffuse_fraction > 1.0:
        raise PhysicsEvaluationError("diffuse_reflection_fraction must be in [0, 1]")
    speed_ratio_limit = _finite_positive_scalar(maximum_speed_ratio, "maximum_speed_ratio")
    base_rate, _, radius_m = epstein_rate_s_inv(
        mass_kg=mass_kg,
        drag_diameter_m=drag_diameter_m,
        gas_density_kg_m3=gas_density_kg_m3,
        gas_temperature_K=gas_temperature_K,
        gas_molecular_mass_kg=gas_molecular_mass_kg,
        delta=1.0,
    )
    _require_positive(gas_mean_free_path_m, "gas_mean_free_path")
    if not bool(np.isfinite(velocity_m_s).all() and np.isfinite(gas_velocity_m_s).all()):
        raise PhysicsEvaluationError("finite-speed Epstein velocity inputs must be finite")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_speed_m_s = np.linalg.norm(gas_velocity_m_s - velocity_m_s, axis=1)
        most_probable_speed_m_s = np.sqrt(
            2.0 * BOLTZMANN_J_K * gas_temperature_K / gas_molecular_mass_kg
        )
        speed_ratio = relative_speed_m_s / most_probable_speed_m_s
        lambda_over_radius = gas_mean_free_path_m / radius_m
    if not bool(
        np.isfinite(most_probable_speed_m_s).all()
        and np.isfinite(speed_ratio).all()
        and np.isfinite(lambda_over_radius).all()
    ):
        raise PhysicsEvaluationError(
            "finite-speed Epstein evaluation produced a non-finite derived value"
        )
    specular_factor, _ = epstein_finite_speed_factors(speed_ratio)
    with np.errstate(over="ignore", invalid="ignore"):
        rate_s_inv = base_rate * (specular_factor + diffuse_fraction * math.pi / 8.0)
    if not bool(np.isfinite(rate_s_inv).all() and (rate_s_inv > 0.0).all()):
        raise PhysicsEvaluationError("finite-speed Epstein rate is not finite and positive")
    applicable = (lambda_over_radius >= EPSTEIN_MIN_LAMBDA_OVER_RADIUS) & (
        speed_ratio <= speed_ratio_limit
    )
    return LinearRelaxation(rate_s_inv, gas_velocity_m_s, applicable)


def epstein_finite_speed_factors(speed_ratio: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the specular rate factor and radial velocity-Jacobian factor."""

    ratio = np.asarray(speed_ratio, dtype=np.float64)
    if ratio.ndim != 1 or not bool(np.isfinite(ratio).all()) or bool((ratio < 0.0).any()):
        raise PhysicsEvaluationError(
            "finite-speed Epstein speed ratio must be finite and nonnegative"
        )
    rate_factor = np.empty_like(ratio)
    radial_factor = np.empty_like(ratio)
    small = ratio <= EPSTEIN_FINITE_SPEED_SERIES_LIMIT
    square = ratio[small] * ratio[small]
    rate_factor[small] = 1.0 + square * (
        1.0 / 5.0 + square * (-1.0 / 70.0 + square * (1.0 / 630.0 - square / 5544.0))
    )
    radial_factor[small] = 1.0 + square * (
        3.0 / 5.0 + square * (-1.0 / 14.0 + square * (1.0 / 90.0 - square / 616.0))
    )

    large = ~small
    if bool(large.any()):
        value = ratio[large]
        inverse = 1.0 / value
        inverse_square = inverse * inverse
        exponential = np.exp(-(value * value))
        error_function = np.asarray([math.erf(float(item)) for item in value], dtype=np.float64)
        rate_factor[large] = (
            3.0 / 16.0 * (2.0 + inverse_square) * exponential
            + 3.0
            * math.sqrt(math.pi)
            / 32.0
            * value
            * (4.0 + 4.0 * inverse_square - inverse_square * inverse_square)
            * error_function
        )
        radial_factor[large] = (
            3.0
            / 16.0
            * (
                (4.0 - 2.0 * inverse_square) * exponential
                + math.sqrt(math.pi)
                * value
                * (4.0 + inverse_square * inverse_square)
                * error_function
            )
        )
    if not bool(
        np.isfinite(rate_factor).all()
        and np.isfinite(radial_factor).all()
        and (rate_factor > 0.0).all()
        and (radial_factor > 0.0).all()
    ):
        raise PhysicsEvaluationError("finite-speed Epstein factor is not finite and positive")
    return rate_factor, radial_factor


def barnes_collisionless_ion_drag(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_number: FloatArray,
    velocity_m_s: FloatArray,
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    positive_ion_temperature_K: FloatArray,
    positive_ion_velocity_m_s: FloatArray,
    ion_neutral_mean_free_path_m: FloatArray,
    positive_ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
) -> BarnesIonDragEvaluation:
    """Evaluate the versioned collisionless Barnes effective-speed model."""

    count = _common_count(
        mass_kg,
        electrostatic_radius_m,
        charge_number,
        electron_number_density_m3,
        positive_ion_number_density_m3,
        electron_temperature_K,
        positive_ion_temperature_K,
        ion_neutral_mean_free_path_m,
    )
    if velocity_m_s.shape != (count, 2) or positive_ion_velocity_m_s.shape != (count, 2):
        raise PhysicsEvaluationError("Barnes ion-drag velocities must have shape [N, 2]")
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_finite(charge_number, "charge_number")
    _require_positive(electron_number_density_m3, "electron_number_density")
    _require_positive(positive_ion_number_density_m3, "positive_ion_number_density")
    _require_positive(electron_temperature_K, "electron_temperature")
    _require_positive(positive_ion_temperature_K, "positive_ion_temperature")
    _require_positive(ion_neutral_mean_free_path_m, "ion_neutral_mean_free_path")
    _require_finite(velocity_m_s, "velocity_m_s")
    _require_finite(positive_ion_velocity_m_s, "positive_ion_velocity_m_s")
    ion_mass = _finite_positive_scalar(positive_ion_mass_kg, "positive_ion_mass_kg")
    drift_limit = _finite_positive_scalar(
        maximum_ion_drift_ratio,
        "maximum_ion_drift_ratio",
    )

    debye_length = _linear_two_species_debye_length(
        electron_number_density_m3,
        positive_ion_number_density_m3,
        electron_temperature_K,
        positive_ion_temperature_K,
    )
    relative_velocity = positive_ion_velocity_m_s - velocity_m_s
    relative_speed = np.linalg.norm(relative_velocity, axis=1)
    ion_mean_thermal_speed = np.sqrt(
        8.0 * BOLTZMANN_J_K * positive_ion_temperature_K / (math.pi * ion_mass)
    )
    effective_speed_square = relative_speed * relative_speed + ion_mean_thermal_speed**2
    effective_speed = np.sqrt(effective_speed_square)
    capacitance = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * electrostatic_radius_m
        * (1.0 + electrostatic_radius_m / debye_length)
    )
    surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance

    collection_cross_section = np.zeros(count, dtype=np.float64)
    orbital_cross_section = np.zeros(count, dtype=np.float64)
    collection_impact_parameter = np.zeros(count, dtype=np.float64)
    orbital_impact_parameter = np.zeros(count, dtype=np.float64)
    coulomb_logarithm = np.zeros(count, dtype=np.float64)
    model_domain = charge_number <= 0.0
    if bool(model_domain.any()):
        selected = model_domain
        orbital_impact_parameter[selected] = (
            np.abs(charge_number[selected])
            * ELEMENTARY_CHARGE_C
            * ELEMENTARY_CHARGE_C
            / (
                4.0
                * math.pi
                * VACUUM_PERMITTIVITY_F_M
                * ion_mass
                * effective_speed_square[selected]
            )
        )
        collection_square = electrostatic_radius_m[selected] ** 2 * (
            1.0
            - 2.0
            * ELEMENTARY_CHARGE_C
            * surface_potential[selected]
            / (ion_mass * effective_speed_square[selected])
        )
        collection_impact_parameter[selected] = np.sqrt(collection_square)
        numerator = debye_length[selected] ** 2 + orbital_impact_parameter[selected] ** 2
        denominator = collection_square + orbital_impact_parameter[selected] ** 2
        coulomb_logarithm[selected] = 0.5 * np.log(numerator / denominator)
        collection_cross_section[selected] = math.pi * collection_square
        orbital_cross_section[selected] = (
            4.0 * math.pi * orbital_impact_parameter[selected] ** 2 * coulomb_logarithm[selected]
        )

    total_cross_section = collection_cross_section + orbital_cross_section
    factor = (
        positive_ion_number_density_m3 * ion_mass * effective_speed * total_cross_section / mass_kg
    )
    acceleration = factor[:, None] * relative_velocity
    ion_drift_ratio = relative_speed / ion_mean_thermal_speed
    radius_ratio = electrostatic_radius_m / debye_length
    orbital_ratio = orbital_impact_parameter / debye_length
    collection_ratio = collection_impact_parameter / debye_length
    collisionless_ratio = ion_neutral_mean_free_path_m / debye_length
    derived = (
        acceleration,
        collection_cross_section,
        orbital_cross_section,
        debye_length,
        surface_potential,
        collection_impact_parameter,
        orbital_impact_parameter,
        coulomb_logarithm,
        ion_drift_ratio,
    )
    if any(not bool(np.isfinite(value).all()) for value in derived):
        raise PhysicsEvaluationError("Barnes ion drag produced a non-finite derived value")
    applicable = model_domain & (radius_ratio <= ION_DRAG_MAX_SCALE_OVER_DEBYE)
    applicable &= orbital_ratio <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    applicable &= collection_ratio <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    applicable &= collisionless_ratio >= ION_DRAG_MIN_MEAN_FREE_PATH_OVER_DEBYE
    applicable &= coulomb_logarithm > 0.0
    applicable &= ion_drift_ratio <= drift_limit
    return BarnesIonDragEvaluation(
        acceleration,
        collection_cross_section,
        orbital_cross_section,
        debye_length,
        surface_potential,
        collection_impact_parameter,
        orbital_impact_parameter,
        coulomb_logarithm,
        ion_drift_ratio,
        applicable,
    )


def relative_flow_screened_collection_orbital_ion_drag(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_number: FloatArray,
    velocity_m_s: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    positive_ion_thermal_voltage_V: FloatArray,
    positive_ion_velocity_m_s: FloatArray,
    effective_positive_ion_mass_kg: FloatArray,
    screening_length_m: FloatArray,
    ion_neutral_mean_free_path_m: FloatArray,
    maximum_relative_ion_speed_m_s: float,
) -> RelativeFlowScreenedIonDragEvaluation:
    """Evaluate the aggregate relative-flow screened sensitivity revision."""

    count = _common_count(
        mass_kg,
        electrostatic_radius_m,
        charge_number,
        positive_ion_number_density_m3,
        positive_ion_thermal_voltage_V,
        effective_positive_ion_mass_kg,
        screening_length_m,
        ion_neutral_mean_free_path_m,
    )
    if velocity_m_s.shape != (count, 2) or positive_ion_velocity_m_s.shape != (count, 2):
        raise PhysicsEvaluationError("aggregate ion-drag velocities must have shape [N, 2]")
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_finite(charge_number, "charge_number")
    _require_finite(velocity_m_s, "velocity_m_s")
    _require_positive(positive_ion_number_density_m3, "positive_ion_number_density_m3")
    _require_positive(positive_ion_thermal_voltage_V, "positive_ion_thermal_voltage_V")
    _require_finite(positive_ion_velocity_m_s, "positive_ion_velocity_m_s")
    _require_positive(effective_positive_ion_mass_kg, "effective_positive_ion_mass_kg")
    _require_positive(screening_length_m, "screening_length_m")
    _require_positive(ion_neutral_mean_free_path_m, "ion_neutral_mean_free_path_m")
    relative_speed_limit = _finite_positive_scalar(
        maximum_relative_ion_speed_m_s,
        "maximum_relative_ion_speed_m_s",
    )

    relative_velocity = positive_ion_velocity_m_s - velocity_m_s
    relative_speed = np.linalg.norm(relative_velocity, axis=1)
    effective_speed_square = (
        relative_speed * relative_speed
        + 8.0
        * ELEMENTARY_CHARGE_C
        * positive_ion_thermal_voltage_V
        / (math.pi * effective_positive_ion_mass_kg)
        + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    effective_speed = np.sqrt(effective_speed_square)
    effective_capacitance_screening = np.maximum(
        electrostatic_radius_m,
        screening_length_m,
    )
    capacitance = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * electrostatic_radius_m
        * (1.0 + electrostatic_radius_m / effective_capacitance_screening)
    )
    surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    screening_radius = np.maximum(
        electrostatic_radius_m,
        np.minimum(screening_length_m, ion_neutral_mean_free_path_m),
    )
    orbital_impact = (
        np.sqrt(charge_number * charge_number + AGGREGATE_ION_DRAG_CHARGE_SQUARE_REGULARIZATION)
        * ELEMENTARY_CHARGE_C**2
        / (
            4.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * effective_positive_ion_mass_kg
            * effective_speed_square
        )
    )
    collection_square = np.minimum(
        screening_radius * screening_radius,
        electrostatic_radius_m**2
        * np.maximum(
            0.0,
            1.0
            - 2.0
            * ELEMENTARY_CHARGE_C
            * surface_potential
            / (effective_positive_ion_mass_kg * effective_speed_square),
        ),
    )
    coulomb_logarithm = np.maximum(
        0.0,
        0.5
        * np.log(
            (screening_radius**2 + orbital_impact**2) / (collection_square + orbital_impact**2)
        ),
    )
    collection_cross_section = math.pi * collection_square
    orbital_cross_section = 4.0 * math.pi * orbital_impact**2 * coulomb_logarithm
    force_factor = (
        positive_ion_number_density_m3
        * effective_positive_ion_mass_kg
        * effective_speed
        * (collection_cross_section + orbital_cross_section)
    )
    acceleration = force_factor[:, None] * relative_velocity / mass_kg[:, None]
    derived = (
        acceleration,
        collection_cross_section,
        orbital_cross_section,
        screening_radius,
        surface_potential,
        effective_speed,
    )
    if any(not bool(np.isfinite(value).all()) for value in derived):
        raise PhysicsEvaluationError("relative-flow screened ion drag is not finite")
    return RelativeFlowScreenedIonDragEvaluation(
        acceleration,
        collection_cross_section,
        orbital_cross_section,
        screening_radius,
        surface_potential,
        effective_speed,
        relative_speed <= relative_speed_limit,
    )


def electric_field_directed_image_orbital_ion_drag(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_number: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_thermal_voltage_V: FloatArray,
    positive_ion_thermal_voltage_V: FloatArray,
    positive_ion_velocity_m_s: FloatArray,
    effective_positive_ion_mass_kg: FloatArray,
    screening_length_m: FloatArray,
    electric_field_V_m: FloatArray,
) -> ElectricFieldDirectedImageIonDragEvaluation:
    """Evaluate the producer-independent electric-field-directed sensitivity revision."""

    count = _common_count(
        mass_kg,
        electrostatic_radius_m,
        charge_number,
        positive_ion_number_density_m3,
        electron_thermal_voltage_V,
        positive_ion_thermal_voltage_V,
        effective_positive_ion_mass_kg,
        screening_length_m,
    )
    if positive_ion_velocity_m_s.shape != (count, 2):
        raise PhysicsEvaluationError("positive_ion_velocity_m_s must have shape [N, 2]")
    if electric_field_V_m.shape != (count, 2):
        raise PhysicsEvaluationError("electric_field_V_m must have shape [N, 2]")
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_finite(charge_number, "charge_number")
    _require_positive(positive_ion_number_density_m3, "positive_ion_number_density_m3")
    _require_positive(electron_thermal_voltage_V, "electron_thermal_voltage_V")
    _require_positive(positive_ion_thermal_voltage_V, "positive_ion_thermal_voltage_V")
    _require_finite(positive_ion_velocity_m_s, "positive_ion_velocity_m_s")
    _require_positive(effective_positive_ion_mass_kg, "effective_positive_ion_mass_kg")
    _require_positive(screening_length_m, "screening_length_m")
    _require_finite(electric_field_V_m, "electric_field_V_m")

    ion_speed = np.linalg.norm(positive_ion_velocity_m_s, axis=1)
    effective_speed_square = (
        ion_speed * ion_speed
        + 8.0
        * ELEMENTARY_CHARGE_C
        * positive_ion_thermal_voltage_V
        / (math.pi * effective_positive_ion_mass_kg)
        + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    effective_speed = np.sqrt(effective_speed_square)
    effective_capacitance_screening = np.maximum(
        electrostatic_radius_m,
        screening_length_m,
    )
    capacitance = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * electrostatic_radius_m
        * (1.0 + electrostatic_radius_m / effective_capacitance_screening)
    )
    surface_potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    collection_cross_section = (
        math.pi
        * electrostatic_radius_m**2
        * np.maximum(0.0, 1.0 - surface_potential / positive_ion_thermal_voltage_V)
    )
    image_impact = (
        ELEMENTARY_CHARGE_C**2
        * charge_number
        / (
            2.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * effective_positive_ion_mass_kg
            * effective_speed_square
        )
    )
    image_screening_length = np.sqrt(
        VACUUM_PERMITTIVITY_F_M
        * electron_thermal_voltage_V
        / (ELEMENTARY_CHARGE_C * positive_ion_number_density_m3)
    )
    image_logarithm = np.log(
        np.maximum(
            1.0 + IMAGE_ION_DRAG_LOG_ARGUMENT_OFFSET,
            image_screening_length / electrostatic_radius_m,
        )
    )
    orbital_cross_section = math.pi * image_impact**2 * image_logarithm
    force_magnitude = (
        effective_positive_ion_mass_kg
        * positive_ion_number_density_m3
        * effective_speed
        * ion_speed
        * (collection_cross_section + orbital_cross_section)
    )
    electric_norm = np.sqrt(
        np.sum(electric_field_V_m * electric_field_V_m, axis=1)
        + IMAGE_ION_DRAG_ELECTRIC_FIELD_SQUARE_FLOOR_V2_M2
    )
    acceleration = (
        force_magnitude[:, None] * electric_field_V_m / electric_norm[:, None] / mass_kg[:, None]
    )
    derived = (
        acceleration,
        collection_cross_section,
        orbital_cross_section,
        surface_potential,
        ion_speed,
        effective_speed,
    )
    if any(not bool(np.isfinite(value).all()) for value in derived):
        raise PhysicsEvaluationError("electric-field-directed image ion drag is not finite")
    return ElectricFieldDirectedImageIonDragEvaluation(
        acceleration,
        collection_cross_section,
        orbital_cross_section,
        surface_potential,
        ion_speed,
        effective_speed,
        np.ones(count, dtype=np.bool_),
    )


def relative_flow_screened_ion_drag_global_bound(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_number_lower: FloatArray,
    charge_number_upper: FloatArray,
    positive_ion_number_density_upper_m3: float,
    positive_ion_thermal_voltage_lower_V: float,
    positive_ion_thermal_voltage_upper_V: float,
    effective_positive_ion_mass_lower_kg: float,
    effective_positive_ion_mass_upper_kg: float,
    screening_length_upper_m: float,
    ion_neutral_mean_free_path_upper_m: float,
    maximum_relative_ion_speed_m_s: float,
) -> FloatArray:
    """Bound relative-flow sensitivity acceleration over declared run ranges."""

    count = _common_count(
        mass_kg,
        electrostatic_radius_m,
        charge_number_lower,
        charge_number_upper,
    )
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_finite(charge_number_lower, "charge_number_lower")
    _require_finite(charge_number_upper, "charge_number_upper")
    if bool((charge_number_lower > charge_number_upper).any()):
        raise PhysicsEvaluationError("ion-drag charge bounds are reversed")
    ion_density_upper = _finite_positive_scalar(
        positive_ion_number_density_upper_m3,
        "positive_ion_number_density_upper_m3",
    )
    ion_voltage_lower = _finite_positive_scalar(
        positive_ion_thermal_voltage_lower_V,
        "positive_ion_thermal_voltage_lower_V",
    )
    ion_voltage_upper = _finite_positive_scalar(
        positive_ion_thermal_voltage_upper_V,
        "positive_ion_thermal_voltage_upper_V",
    )
    ion_mass_lower = _finite_positive_scalar(
        effective_positive_ion_mass_lower_kg,
        "effective_positive_ion_mass_lower_kg",
    )
    ion_mass_upper = _finite_positive_scalar(
        effective_positive_ion_mass_upper_kg,
        "effective_positive_ion_mass_upper_kg",
    )
    screening_upper = _finite_positive_scalar(
        screening_length_upper_m,
        "screening_length_upper_m",
    )
    mean_free_path_upper = _finite_positive_scalar(
        ion_neutral_mean_free_path_upper_m,
        "ion_neutral_mean_free_path_upper_m",
    )
    relative_speed_upper = _finite_positive_scalar(
        maximum_relative_ion_speed_m_s,
        "maximum_relative_ion_speed_m_s",
    )
    if ion_voltage_lower > ion_voltage_upper or ion_mass_lower > ion_mass_upper:
        raise PhysicsEvaluationError("aggregate ion-drag primitive bounds are reversed")

    speed_square_upper = (
        relative_speed_upper**2
        + 8.0 * ELEMENTARY_CHARGE_C * ion_voltage_upper / (math.pi * ion_mass_lower)
        + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    screening_radius_upper = np.maximum(
        electrostatic_radius_m,
        min(screening_upper, mean_free_path_upper),
    )
    total_cross_section_upper = _outward_abs_upper(
        3.0 * math.pi * screening_radius_upper**2,
        "relative-flow ion-drag cross-section bound",
    )
    scalar_upper = _outward_abs_upper(
        ion_density_upper
        * ion_mass_upper
        * math.sqrt(speed_square_upper)
        * total_cross_section_upper
        * relative_speed_upper
        / mass_kg,
        "relative-flow ion-drag acceleration bound",
    )
    return np.broadcast_to(scalar_upper[:, None], (count, 2)).copy()


def relative_flow_screened_ion_drag_local_bound(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_number_lower: FloatArray,
    charge_number_upper: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    positive_ion_number_density_upper_m3: FloatArray,
    positive_ion_thermal_voltage_lower_V: FloatArray,
    positive_ion_thermal_voltage_upper_V: FloatArray,
    positive_ion_velocity_lower_m_s: FloatArray,
    positive_ion_velocity_upper_m_s: FloatArray,
    effective_positive_ion_mass_lower_kg: FloatArray,
    effective_positive_ion_mass_upper_kg: FloatArray,
    screening_length_upper_m: FloatArray,
    ion_neutral_mean_free_path_upper_m: FloatArray,
    maximum_relative_ion_speed_m_s: float,
) -> tuple[FloatArray, BoolArray]:
    """Bound relative-flow ion drag over row-local state/primitive boxes.

    Unlike the run-global preparation bound, this certificate retains a
    positive lower bound on relative speed when the particle and ion velocity
    boxes are disjoint.  That lower bound is essential for bounding the
    charge-dependent orbital impact parameter without replacing it by the
    full screening disk.
    """

    count = _common_count(
        mass_kg,
        electrostatic_radius_m,
        charge_number_lower,
        charge_number_upper,
        positive_ion_number_density_upper_m3,
        positive_ion_thermal_voltage_lower_V,
        positive_ion_thermal_voltage_upper_V,
        effective_positive_ion_mass_lower_kg,
        effective_positive_ion_mass_upper_kg,
        screening_length_upper_m,
        ion_neutral_mean_free_path_upper_m,
    )
    vector_ranges = (
        velocity_lower_m_s,
        velocity_upper_m_s,
        positive_ion_velocity_lower_m_s,
        positive_ion_velocity_upper_m_s,
    )
    if any(value.shape != (count, 2) for value in vector_ranges):
        raise PhysicsEvaluationError("local ion-drag velocity bounds must have shape [N, 2]")
    for value in vector_ranges:
        _require_finite(value, "local ion-drag velocity bound")
    if bool(
        (charge_number_lower > charge_number_upper).any()
        or (velocity_lower_m_s > velocity_upper_m_s).any()
        or (positive_ion_velocity_lower_m_s > positive_ion_velocity_upper_m_s).any()
    ):
        raise PhysicsEvaluationError("local ion-drag bounds are reversed")
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_finite(charge_number_lower, "charge_number_lower")
    _require_finite(charge_number_upper, "charge_number_upper")
    _require_positive(
        positive_ion_number_density_upper_m3,
        "positive_ion_number_density_upper_m3",
    )
    _require_positive(
        positive_ion_thermal_voltage_lower_V,
        "positive_ion_thermal_voltage_lower_V",
    )
    _require_positive(
        positive_ion_thermal_voltage_upper_V,
        "positive_ion_thermal_voltage_upper_V",
    )
    _require_positive(
        effective_positive_ion_mass_lower_kg,
        "effective_positive_ion_mass_lower_kg",
    )
    _require_positive(
        effective_positive_ion_mass_upper_kg,
        "effective_positive_ion_mass_upper_kg",
    )
    _require_positive(screening_length_upper_m, "screening_length_upper_m")
    _require_positive(
        ion_neutral_mean_free_path_upper_m,
        "ion_neutral_mean_free_path_upper_m",
    )
    if bool(
        (positive_ion_thermal_voltage_lower_V > positive_ion_thermal_voltage_upper_V).any()
        or (effective_positive_ion_mass_lower_kg > effective_positive_ion_mass_upper_kg).any()
    ):
        raise PhysicsEvaluationError("local ion-drag primitive bounds are reversed")
    relative_speed_limit = _finite_positive_scalar(
        maximum_relative_ion_speed_m_s,
        "maximum_relative_ion_speed_m_s",
    )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_component_upper = _outward_abs_upper(
            np.maximum(
                np.abs(positive_ion_velocity_lower_m_s - velocity_upper_m_s),
                np.abs(positive_ion_velocity_upper_m_s - velocity_lower_m_s),
            ),
            "local ion-drag relative-component upper bound",
        )
        relative_component_lower = np.maximum(
            positive_ion_velocity_lower_m_s - velocity_upper_m_s,
            velocity_lower_m_s - positive_ion_velocity_upper_m_s,
        )
        relative_component_lower = _outward_nonnegative_lower(
            np.maximum(relative_component_lower, 0.0),
            "local ion-drag relative-component lower bound",
        )
        relative_speed_upper = _outward_abs_upper(
            np.hypot(relative_component_upper[:, 0], relative_component_upper[:, 1]),
            "local ion-drag relative-speed upper bound",
        )
        relative_speed_lower = _outward_nonnegative_lower(
            np.hypot(relative_component_lower[:, 0], relative_component_lower[:, 1]),
            "local ion-drag relative-speed lower bound",
        )
        speed_square_lower = _outward_nonnegative_lower(
            relative_speed_lower**2
            + 8.0
            * ELEMENTARY_CHARGE_C
            * positive_ion_thermal_voltage_lower_V
            / (math.pi * effective_positive_ion_mass_upper_kg)
            + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2,
            "local ion-drag effective-speed-square lower bound",
        )
        speed_square_upper = _outward_abs_upper(
            relative_speed_upper**2
            + 8.0
            * ELEMENTARY_CHARGE_C
            * positive_ion_thermal_voltage_upper_V
            / (math.pi * effective_positive_ion_mass_lower_kg)
            + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2,
            "local ion-drag effective-speed-square upper bound",
        )
        charge_abs_upper = np.maximum(
            np.abs(charge_number_lower),
            np.abs(charge_number_upper),
        )
        charge_abs_lower = np.minimum(
            np.abs(charge_number_lower),
            np.abs(charge_number_upper),
        )
        charge_abs_lower[(charge_number_lower <= 0.0) & (charge_number_upper >= 0.0)] = 0.0
        capacitance_lower = _outward_nonnegative_lower(
            4.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * electrostatic_radius_m
            * (
                1.0
                + electrostatic_radius_m
                / np.maximum(electrostatic_radius_m, screening_length_upper_m)
            ),
            "local ion-drag capacitance lower bound",
        )
        potential_abs_upper = _outward_abs_upper(
            charge_abs_upper * ELEMENTARY_CHARGE_C / capacitance_lower,
            "local ion-drag potential upper bound",
        )
        screening_radius_upper = np.maximum(
            electrostatic_radius_m,
            np.minimum(screening_length_upper_m, ion_neutral_mean_free_path_upper_m),
        )
        screening_square_upper = _outward_abs_upper(
            screening_radius_upper**2,
            "local ion-drag screening-square upper bound",
        )
        focused_collection_square_upper = _outward_abs_upper(
            electrostatic_radius_m**2
            * (
                1.0
                + 2.0
                * ELEMENTARY_CHARGE_C
                * potential_abs_upper
                / (effective_positive_ion_mass_lower_kg * speed_square_lower)
            ),
            "local ion-drag focused collection-square upper bound",
        )
        collection_square_upper = np.minimum(
            screening_square_upper,
            focused_collection_square_upper,
        )
        orbital_impact_lower = _outward_nonnegative_lower(
            np.sqrt(charge_abs_lower**2 + AGGREGATE_ION_DRAG_CHARGE_SQUARE_REGULARIZATION)
            * ELEMENTARY_CHARGE_C**2
            / (
                4.0
                * math.pi
                * VACUUM_PERMITTIVITY_F_M
                * effective_positive_ion_mass_upper_kg
                * speed_square_upper
            ),
            "local ion-drag orbital-impact lower bound",
        )
        orbital_impact_upper = _outward_abs_upper(
            np.sqrt(charge_abs_upper**2 + AGGREGATE_ION_DRAG_CHARGE_SQUARE_REGULARIZATION)
            * ELEMENTARY_CHARGE_C**2
            / (
                4.0
                * math.pi
                * VACUUM_PERMITTIVITY_F_M
                * effective_positive_ion_mass_lower_kg
                * speed_square_lower
            ),
            "local ion-drag orbital-impact upper bound",
        )
        logarithm_argument_upper = _outward_abs_upper(
            (
                screening_square_upper
                + _outward_abs_upper(
                    orbital_impact_upper**2,
                    "local ion-drag orbital-impact-square upper bound",
                )
            )
            / _outward_nonnegative_lower(
                orbital_impact_lower**2,
                "local ion-drag orbital-impact-square lower bound",
            ),
            "local ion-drag logarithm-argument upper bound",
        )
        logarithm_upper = _outward_abs_upper(
            0.5 * np.log(logarithm_argument_upper),
            "local ion-drag logarithm upper bound",
        )
        collection_cross_section_upper = _outward_abs_upper(
            math.pi * collection_square_upper,
            "local ion-drag collection-cross-section upper bound",
        )
        orbital_cross_section_upper = _outward_abs_upper(
            4.0 * math.pi * orbital_impact_upper**2 * np.maximum(logarithm_upper, 0.0),
            "local ion-drag orbital-cross-section upper bound",
        )
        force_factor_upper = _outward_abs_upper(
            positive_ion_number_density_upper_m3
            * effective_positive_ion_mass_upper_kg
            * np.sqrt(speed_square_upper)
            * (collection_cross_section_upper + orbital_cross_section_upper)
            / mass_kg,
            "local ion-drag force-factor upper bound",
        )
        acceleration_upper = _outward_abs_upper(
            force_factor_upper[:, None] * relative_component_upper,
            "local ion-drag acceleration upper bound",
        )

    finite = all(
        bool(np.isfinite(value).all())
        for value in (
            relative_speed_lower,
            relative_speed_upper,
            speed_square_lower,
            speed_square_upper,
            capacitance_lower,
            potential_abs_upper,
            screening_radius_upper,
            collection_square_upper,
            orbital_impact_lower,
            orbital_impact_upper,
            logarithm_upper,
            acceleration_upper,
        )
    )
    if not finite or bool((acceleration_upper < 0.0).any()):
        raise PhysicsEvaluationError("local relative-flow ion-drag bound is not finite")
    applicable = relative_speed_upper <= relative_speed_limit
    return acceleration_upper, applicable


def electric_field_directed_image_ion_drag_global_bound(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_number_lower: FloatArray,
    charge_number_upper: FloatArray,
    positive_ion_number_density_lower_m3: float,
    positive_ion_number_density_upper_m3: float,
    electron_thermal_voltage_upper_V: float,
    positive_ion_thermal_voltage_lower_V: float,
    positive_ion_thermal_voltage_upper_V: float,
    positive_ion_velocity_abs_upper_m_s: FloatArray,
    effective_positive_ion_mass_lower_kg: float,
    effective_positive_ion_mass_upper_kg: float,
) -> FloatArray:
    """Bound image-sensitivity acceleration; each direction component is at most one."""

    count = _common_count(
        mass_kg,
        electrostatic_radius_m,
        charge_number_lower,
        charge_number_upper,
    )
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_finite(charge_number_lower, "charge_number_lower")
    _require_finite(charge_number_upper, "charge_number_upper")
    if bool((charge_number_lower > charge_number_upper).any()):
        raise PhysicsEvaluationError("ion-drag charge bounds are reversed")
    ion_density_lower = _finite_positive_scalar(
        positive_ion_number_density_lower_m3,
        "positive_ion_number_density_lower_m3",
    )
    ion_density_upper = _finite_positive_scalar(
        positive_ion_number_density_upper_m3,
        "positive_ion_number_density_upper_m3",
    )
    electron_voltage_upper = _finite_positive_scalar(
        electron_thermal_voltage_upper_V,
        "electron_thermal_voltage_upper_V",
    )
    ion_voltage_lower = _finite_positive_scalar(
        positive_ion_thermal_voltage_lower_V,
        "positive_ion_thermal_voltage_lower_V",
    )
    ion_voltage_upper = _finite_positive_scalar(
        positive_ion_thermal_voltage_upper_V,
        "positive_ion_thermal_voltage_upper_V",
    )
    ion_mass_lower = _finite_positive_scalar(
        effective_positive_ion_mass_lower_kg,
        "effective_positive_ion_mass_lower_kg",
    )
    ion_mass_upper = _finite_positive_scalar(
        effective_positive_ion_mass_upper_kg,
        "effective_positive_ion_mass_upper_kg",
    )
    if (
        ion_density_lower > ion_density_upper
        or ion_voltage_lower > ion_voltage_upper
        or ion_mass_lower > ion_mass_upper
    ):
        raise PhysicsEvaluationError("image ion-drag primitive bounds are reversed")
    ion_velocity_abs = _component_abs_upper(
        positive_ion_velocity_abs_upper_m_s,
        "positive_ion_velocity_abs_upper_m_s",
    )
    ion_speed_upper = math.nextafter(math.hypot(*ion_velocity_abs), math.inf)
    speed_square_lower = (
        8.0 * ELEMENTARY_CHARGE_C * ion_voltage_lower / (math.pi * ion_mass_upper)
        + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    speed_square_upper = (
        ion_speed_upper**2
        + 8.0 * ELEMENTARY_CHARGE_C * ion_voltage_upper / (math.pi * ion_mass_lower)
        + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    charge_abs_upper = np.maximum(np.abs(charge_number_lower), np.abs(charge_number_upper))
    surface_potential_abs_upper = _outward_abs_upper(
        charge_abs_upper
        * ELEMENTARY_CHARGE_C
        / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * electrostatic_radius_m),
        "image ion-drag surface-potential bound",
    )
    collection_cross_section_upper = _outward_abs_upper(
        math.pi
        * electrostatic_radius_m**2
        * (1.0 + surface_potential_abs_upper / ion_voltage_lower),
        "image ion-drag collection-cross-section bound",
    )
    image_impact_abs_upper = _outward_abs_upper(
        ELEMENTARY_CHARGE_C**2
        * charge_abs_upper
        / (2.0 * math.pi * VACUUM_PERMITTIVITY_F_M * ion_mass_lower * speed_square_lower),
        "image ion-drag impact bound",
    )
    image_screening_upper = math.nextafter(
        math.sqrt(
            VACUUM_PERMITTIVITY_F_M
            * electron_voltage_upper
            / (ELEMENTARY_CHARGE_C * ion_density_lower)
        ),
        math.inf,
    )
    image_logarithm_upper = np.log(
        np.maximum(
            1.0 + IMAGE_ION_DRAG_LOG_ARGUMENT_OFFSET,
            image_screening_upper / electrostatic_radius_m,
        )
    )
    orbital_cross_section_upper = _outward_abs_upper(
        math.pi * image_impact_abs_upper**2 * image_logarithm_upper,
        "image ion-drag orbital-cross-section bound",
    )
    scalar_upper = _outward_abs_upper(
        ion_mass_upper
        * ion_density_upper
        * math.sqrt(speed_square_upper)
        * ion_speed_upper
        * (collection_cross_section_upper + orbital_cross_section_upper)
        / mass_kg,
        "image ion-drag acceleration bound",
    )
    return np.broadcast_to(scalar_upper[:, None], (count, 2)).copy()


def relative_flow_screened_continuous_applicability_batch(
    *,
    velocity_abs_upper_m_s: FloatArray,
    positive_ion_velocity_abs_upper_m_s: FloatArray,
    maximum_relative_ion_speed_m_s: float,
) -> tuple[BoolArray, UInt8Array]:
    """Certify the declared relative-speed envelope over one continuous path."""

    value = np.asarray(velocity_abs_upper_m_s, dtype=np.float64)
    if value.ndim != 2 or value.shape[1:] != (2,):
        raise PhysicsEvaluationError("velocity_abs_upper_m_s must have shape [N, 2]")
    count = int(value.shape[0])
    velocity, finite_velocity = _continuous_velocity_bound(value, count)
    ion_velocity = _component_abs_upper(
        positive_ion_velocity_abs_upper_m_s,
        "positive_ion_velocity_abs_upper_m_s",
    )
    speed_limit = _finite_positive_scalar(
        maximum_relative_ion_speed_m_s,
        "maximum_relative_ion_speed_m_s",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        relative_component = np.nextafter(velocity + ion_velocity[None, :], np.inf)
        relative_speed = np.nextafter(
            np.hypot(relative_component[:, 0], relative_component[:, 1]),
            np.inf,
        )
    numerical_ok = finite_velocity & np.isfinite(relative_speed)
    status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    return numerical_ok & (relative_speed <= speed_limit), status


def barnes_collisionless_global_bounds(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_number_lower: FloatArray,
    charge_number_upper: FloatArray,
    electron_number_density_lower_m3: float,
    electron_number_density_upper_m3: float,
    positive_ion_number_density_lower_m3: float,
    positive_ion_number_density_upper_m3: float,
    electron_temperature_lower_K: float,
    electron_temperature_upper_K: float,
    positive_ion_temperature_lower_K: float,
    positive_ion_temperature_upper_K: float,
    ion_neutral_mean_free_path_lower_m: float,
    positive_ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
) -> BarnesIonDragGlobalBounds:
    """Build conservative acceleration and static-applicability certificates."""

    count = _common_count(
        mass_kg,
        electrostatic_radius_m,
        charge_number_lower,
        charge_number_upper,
    )
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_finite(charge_number_lower, "charge_number_lower")
    _require_finite(charge_number_upper, "charge_number_upper")
    if bool((charge_number_lower > charge_number_upper).any()):
        raise PhysicsEvaluationError("ion-drag charge bounds are reversed")
    electron_density_lower = _finite_positive_scalar(
        electron_number_density_lower_m3,
        "electron_number_density_lower_m3",
    )
    electron_density_upper = _finite_positive_scalar(
        electron_number_density_upper_m3,
        "electron_number_density_upper_m3",
    )
    ion_density_lower = _finite_positive_scalar(
        positive_ion_number_density_lower_m3,
        "positive_ion_number_density_lower_m3",
    )
    ion_density_upper = _finite_positive_scalar(
        positive_ion_number_density_upper_m3,
        "positive_ion_number_density_upper_m3",
    )
    electron_temperature_lower = _finite_positive_scalar(
        electron_temperature_lower_K,
        "electron_temperature_lower_K",
    )
    electron_temperature_upper = _finite_positive_scalar(
        electron_temperature_upper_K,
        "electron_temperature_upper_K",
    )
    ion_temperature_lower = _finite_positive_scalar(
        positive_ion_temperature_lower_K,
        "positive_ion_temperature_lower_K",
    )
    ion_temperature_upper = _finite_positive_scalar(
        positive_ion_temperature_upper_K,
        "positive_ion_temperature_upper_K",
    )
    mean_free_path_lower = _finite_positive_scalar(
        ion_neutral_mean_free_path_lower_m,
        "ion_neutral_mean_free_path_lower_m",
    )
    ion_mass = _finite_positive_scalar(positive_ion_mass_kg, "positive_ion_mass_kg")
    drift_limit = _finite_positive_scalar(
        maximum_ion_drift_ratio,
        "maximum_ion_drift_ratio",
    )
    if (
        electron_density_lower > electron_density_upper
        or ion_density_lower > ion_density_upper
        or electron_temperature_lower > electron_temperature_upper
        or ion_temperature_lower > ion_temperature_upper
    ):
        raise PhysicsEvaluationError("ion-drag primitive bounds are reversed")

    debye_lower = math.nextafter(
        math.sqrt(
            VACUUM_PERMITTIVITY_F_M
            * BOLTZMANN_J_K
            / (
                ELEMENTARY_CHARGE_C**2
                * (
                    electron_density_upper / electron_temperature_lower
                    + ion_density_upper / ion_temperature_lower
                )
            )
        ),
        0.0,
    )
    debye_upper = math.nextafter(
        math.sqrt(
            VACUUM_PERMITTIVITY_F_M
            * BOLTZMANN_J_K
            / (
                ELEMENTARY_CHARGE_C**2
                * (
                    electron_density_lower / electron_temperature_upper
                    + ion_density_lower / ion_temperature_upper
                )
            )
        ),
        math.inf,
    )
    thermal_lower = math.nextafter(
        math.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_lower / (math.pi * ion_mass)),
        0.0,
    )
    thermal_upper = math.nextafter(
        math.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_upper / (math.pi * ion_mass)),
        math.inf,
    )
    charge_abs_upper = np.maximum(np.abs(charge_number_lower), np.abs(charge_number_upper))
    orbital_upper = _outward_abs_upper(
        charge_abs_upper
        * ELEMENTARY_CHARGE_C**2
        / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * ion_mass * thermal_lower**2),
        "ion-drag orbital impact bound",
    )
    capacitance_lower = (
        4.0
        * math.pi
        * VACUUM_PERMITTIVITY_F_M
        * electrostatic_radius_m
        * (1.0 + electrostatic_radius_m / debye_upper)
    )
    potential_abs_upper = _outward_abs_upper(
        charge_abs_upper * ELEMENTARY_CHARGE_C / capacitance_lower,
        "ion-drag surface-potential bound",
    )
    collection_square_upper = _outward_abs_upper(
        electrostatic_radius_m**2
        * (1.0 + 2.0 * ELEMENTARY_CHARGE_C * potential_abs_upper / (ion_mass * thermal_lower**2)),
        "ion-drag collection-impact bound",
    )
    collection_upper = np.sqrt(collection_square_upper)
    coulomb_log_lower = 0.5 * np.log(debye_lower**2 / (collection_square_upper + orbital_upper**2))
    coulomb_log_upper = 0.5 * np.log(
        (debye_upper**2 + orbital_upper**2) / electrostatic_radius_m**2
    )
    coulomb_log_abs_upper = np.maximum(
        np.abs(coulomb_log_lower),
        np.abs(coulomb_log_upper),
    )
    maximum_relative_speed = math.nextafter(drift_limit * thermal_upper, math.inf)
    maximum_effective_speed = math.nextafter(
        thermal_upper * math.sqrt(1.0 + drift_limit**2),
        math.inf,
    )
    cross_section_upper = _outward_abs_upper(
        math.pi * collection_square_upper
        + 4.0 * math.pi * orbital_upper**2 * coulomb_log_abs_upper,
        "ion-drag cross-section bound",
    )
    acceleration_upper = _outward_abs_upper(
        ion_density_upper
        * ion_mass
        * maximum_effective_speed
        * cross_section_upper
        * maximum_relative_speed
        / mass_kg,
        "ion-drag acceleration bound",
    )
    static_applicable = charge_number_upper <= 0.0
    static_applicable &= electrostatic_radius_m / debye_lower <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    static_applicable &= orbital_upper / debye_lower <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    static_applicable &= collection_upper / debye_lower <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    static_applicable &= (
        mean_free_path_lower / debye_upper >= ION_DRAG_MIN_MEAN_FREE_PATH_OVER_DEBYE
    )
    static_applicable &= np.isfinite(coulomb_log_lower) & (coulomb_log_lower > 0.0)
    return BarnesIonDragGlobalBounds(
        np.broadcast_to(acceleration_upper[:, None], (count, 2)).copy(),
        static_applicable,
        debye_lower,
        debye_upper,
        ion_temperature_lower,
    )


def barnes_collisionless_continuous_applicability_batch(
    *,
    static_applicable: BoolArray,
    velocity_abs_upper_m_s: FloatArray,
    positive_ion_velocity_abs_upper_m_s: FloatArray,
    positive_ion_temperature_lower_K: float,
    positive_ion_mass_kg: float,
    maximum_ion_drift_ratio: float,
) -> tuple[BoolArray, UInt8Array]:
    """Certify the remaining drift gate over one continuous path enclosure."""

    static = np.asarray(static_applicable, dtype=np.bool_)
    if static.ndim != 1:
        raise PhysicsEvaluationError("ion-drag static applicability must have shape [N]")
    count = int(static.size)
    velocity, finite_velocity = _continuous_velocity_bound(velocity_abs_upper_m_s, count)
    ion_velocity = _component_abs_upper(
        positive_ion_velocity_abs_upper_m_s,
        "positive_ion_velocity_abs_upper_m_s",
    )
    ion_temperature_lower = _finite_positive_scalar(
        positive_ion_temperature_lower_K,
        "positive_ion_temperature_lower_K",
    )
    ion_mass = _finite_positive_scalar(positive_ion_mass_kg, "positive_ion_mass_kg")
    drift_limit = _finite_positive_scalar(
        maximum_ion_drift_ratio,
        "maximum_ion_drift_ratio",
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_component = np.nextafter(velocity + ion_velocity[None, :], np.inf)
        relative_speed = np.nextafter(
            np.hypot(relative_component[:, 0], relative_component[:, 1]),
            np.inf,
        )
        thermal_speed_lower = math.nextafter(
            math.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_lower / (math.pi * ion_mass)),
            0.0,
        )
        drift_ratio = np.nextafter(relative_speed / thermal_speed_lower, np.inf)
    numerical_ok = finite_velocity & np.isfinite(drift_ratio)
    status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    return static & numerical_ok & (drift_ratio <= drift_limit), status


def _linear_two_species_debye_length(
    electron_number_density_m3: FloatArray,
    positive_ion_number_density_m3: FloatArray,
    electron_temperature_K: FloatArray,
    positive_ion_temperature_K: FloatArray,
) -> FloatArray:
    inverse_square = (
        ELEMENTARY_CHARGE_C**2
        / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
        * (
            electron_number_density_m3 / electron_temperature_K
            + positive_ion_number_density_m3 / positive_ion_temperature_K
        )
    )
    result = 1.0 / np.sqrt(inverse_square)
    if not bool(np.isfinite(result).all() and (result > 0.0).all()):
        raise PhysicsEvaluationError("two-species Debye length is not finite and positive")
    return result


def stokes_cunningham_linear_relaxation(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    velocity_m_s: FloatArray,
    gas_velocity_m_s: FloatArray,
    gas_density_kg_m3: FloatArray,
    gas_dynamic_viscosity_Pa_s: FloatArray,
    gas_mean_free_path_m: FloatArray,
) -> LinearRelaxation:
    """Evaluate the Allen--Raabe air slip-corrected Stokes model."""

    count = _common_count(
        mass_kg,
        drag_diameter_m,
        gas_density_kg_m3,
        gas_dynamic_viscosity_Pa_s,
        gas_mean_free_path_m,
    )
    if velocity_m_s.shape != (count, 2) or gas_velocity_m_s.shape != (count, 2):
        raise PhysicsEvaluationError("Stokes-Cunningham velocity inputs must have shape [N, 2]")
    _require_positive(gas_density_kg_m3, "gas_density")
    if not bool(np.isfinite(velocity_m_s).all() and np.isfinite(gas_velocity_m_s).all()):
        raise PhysicsEvaluationError("Stokes-Cunningham velocity inputs must be finite")

    rate_s_inv, knudsen_radius, _ = stokes_cunningham_rate_s_inv(
        mass_kg=mass_kg,
        drag_diameter_m=drag_diameter_m,
        gas_dynamic_viscosity_Pa_s=gas_dynamic_viscosity_Pa_s,
        gas_mean_free_path_m=gas_mean_free_path_m,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative_speed_m_s = np.linalg.norm(gas_velocity_m_s - velocity_m_s, axis=1)
        reynolds = (
            gas_density_kg_m3 * drag_diameter_m * relative_speed_m_s / gas_dynamic_viscosity_Pa_s
        )
    if not bool(np.isfinite(relative_speed_m_s).all() and np.isfinite(reynolds).all()):
        raise PhysicsEvaluationError(
            "Stokes-Cunningham evaluation produced a non-finite derived value"
        )
    applicable = (
        (knudsen_radius >= STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS)
        & (knudsen_radius <= STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS)
        & (reynolds <= STOKES_CUNNINGHAM_MAX_REYNOLDS)
    )
    return LinearRelaxation(rate_s_inv, gas_velocity_m_s, applicable)


def stokes_cunningham_rate_s_inv(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    gas_dynamic_viscosity_Pa_s: FloatArray,
    gas_mean_free_path_m: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return relaxation rate, radius-based Knudsen number, and slip factor."""

    _common_count(
        mass_kg,
        drag_diameter_m,
        gas_dynamic_viscosity_Pa_s,
        gas_mean_free_path_m,
    )
    _require_positive(mass_kg, "mass_kg")
    _require_positive(drag_diameter_m, "drag_diameter_m")
    _require_positive(gas_dynamic_viscosity_Pa_s, "gas_dynamic_viscosity")
    _require_positive(gas_mean_free_path_m, "gas_mean_free_path")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        knudsen_radius = 2.0 * gas_mean_free_path_m / drag_diameter_m
        slip_correction = 1.0 + knudsen_radius * (
            _ALLEN_RAABE_A1 + _ALLEN_RAABE_A2 * np.exp(-_ALLEN_RAABE_A3 / knudsen_radius)
        )
        rate_s_inv = (
            3.0
            * math.pi
            * gas_dynamic_viscosity_Pa_s
            * drag_diameter_m
            / (slip_correction * mass_kg)
        )
    if not bool(
        np.isfinite(rate_s_inv).all()
        and np.isfinite(knudsen_radius).all()
        and np.isfinite(slip_correction).all()
        and (rate_s_inv > 0.0).all()
        and (knudsen_radius > 0.0).all()
        and (slip_correction > 0.0).all()
    ):
        raise PhysicsEvaluationError("Stokes-Cunningham rate is not finite and positive")
    return rate_s_inv, knudsen_radius, slip_correction


def stokes_cunningham_rate_upper_s_inv(
    *,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    gas_dynamic_viscosity_upper_Pa_s: float,
    gas_mean_free_path_lower_m: float,
) -> FloatArray:
    """Bound the linear relaxation rate over independent primitive extrema."""

    count = _common_count(mass_kg, drag_diameter_m)
    viscosity_upper = _finite_positive_scalar(
        gas_dynamic_viscosity_upper_Pa_s,
        "gas_dynamic_viscosity_upper_Pa_s",
    )
    mean_free_path_lower = _finite_positive_scalar(
        gas_mean_free_path_lower_m,
        "gas_mean_free_path_lower_m",
    )
    rate, _, _ = stokes_cunningham_rate_s_inv(
        mass_kg=mass_kg,
        drag_diameter_m=drag_diameter_m,
        gas_dynamic_viscosity_Pa_s=np.full(count, viscosity_upper, dtype=np.float64),
        gas_mean_free_path_m=np.full(count, mean_free_path_lower, dtype=np.float64),
    )
    return _outward_abs_upper(rate, "Stokes-Cunningham rate bound")


def stokes_cunningham_continuous_applicability_batch(
    *,
    drag_diameter_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    gas_density_upper_kg_m3: float,
    gas_dynamic_viscosity_lower_Pa_s: float,
    gas_mean_free_path_lower_m: float,
    gas_mean_free_path_upper_m: float,
) -> tuple[BoolArray, UInt8Array]:
    """Certify Stokes applicability while localizing non-finite row arithmetic."""

    count = _common_count(drag_diameter_m)
    _require_positive(drag_diameter_m, "drag_diameter_m")
    velocity, finite_velocity = _continuous_velocity_bound(velocity_abs_upper_m_s, count)
    gas_velocity = _component_abs_upper(
        gas_velocity_abs_upper_m_s,
        "gas_velocity_abs_upper_m_s",
    )
    density_upper = _finite_positive_scalar(gas_density_upper_kg_m3, "gas_density_upper_kg_m3")
    viscosity_lower = _finite_positive_scalar(
        gas_dynamic_viscosity_lower_Pa_s,
        "gas_dynamic_viscosity_lower_Pa_s",
    )
    mean_free_path_lower = _finite_positive_scalar(
        gas_mean_free_path_lower_m,
        "gas_mean_free_path_lower_m",
    )
    mean_free_path_upper = _finite_positive_scalar(
        gas_mean_free_path_upper_m,
        "gas_mean_free_path_upper_m",
    )
    if mean_free_path_lower > mean_free_path_upper:
        raise PhysicsEvaluationError("gas mean-free-path bounds are reversed")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        knudsen_lower = np.nextafter(
            2.0 * mean_free_path_lower / drag_diameter_m,
            -np.inf,
        )
        knudsen_upper = np.nextafter(
            2.0 * mean_free_path_upper / drag_diameter_m,
            np.inf,
        )
        relative_component_upper = velocity + gas_velocity[None, :]
        relative_speed_upper = np.hypot(
            relative_component_upper[:, 0],
            relative_component_upper[:, 1],
        )
        relative_speed_upper = np.nextafter(
            relative_speed_upper * _BOUND_ROUNDOFF_FACTOR,
            np.inf,
        )
        reynolds_upper = np.nextafter(
            density_upper
            * drag_diameter_m
            * relative_speed_upper
            / viscosity_lower
            * _BOUND_ROUNDOFF_FACTOR,
            np.inf,
        )
    numerical_ok = finite_velocity
    numerical_ok &= np.isfinite(knudsen_lower)
    numerical_ok &= np.isfinite(knudsen_upper)
    numerical_ok &= np.isfinite(relative_speed_upper)
    numerical_ok &= np.isfinite(reynolds_upper)
    applicable = (
        (knudsen_lower >= STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS)
        & (knudsen_upper <= STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS)
        & (reynolds_upper <= STOKES_CUNNINGHAM_MAX_REYNOLDS)
    )
    applicable &= numerical_ok
    status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    return applicable, status


def stokes_cunningham_continuous_applicability(
    *,
    drag_diameter_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    gas_density_upper_kg_m3: float,
    gas_dynamic_viscosity_lower_Pa_s: float,
    gas_mean_free_path_lower_m: float,
    gas_mean_free_path_upper_m: float,
) -> BoolArray:
    """Certify Allen--Raabe Kn and creeping-flow Re over a path enclosure."""

    applicable, status = stokes_cunningham_continuous_applicability_batch(
        drag_diameter_m=drag_diameter_m,
        velocity_abs_upper_m_s=velocity_abs_upper_m_s,
        gas_velocity_abs_upper_m_s=gas_velocity_abs_upper_m_s,
        gas_density_upper_kg_m3=gas_density_upper_kg_m3,
        gas_dynamic_viscosity_lower_Pa_s=gas_dynamic_viscosity_lower_Pa_s,
        gas_mean_free_path_lower_m=gas_mean_free_path_lower_m,
        gas_mean_free_path_upper_m=gas_mean_free_path_upper_m,
    )
    if bool(np.any(status != CONTINUOUS_APPLICABILITY_OK)):
        raise PhysicsEvaluationError("Stokes-Cunningham applicability bound is not finite")
    return applicable


def linear_drag_acceleration_abs_upper(
    *,
    rate_upper_s_inv: FloatArray,
    target_velocity_abs_upper_m_s: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
) -> FloatArray:
    """Return a component-wise bound for ``rate * (target - velocity)``."""

    result, status = linear_drag_acceleration_abs_upper_batch(
        rate_upper_s_inv=rate_upper_s_inv,
        target_velocity_abs_upper_m_s=target_velocity_abs_upper_m_s,
        velocity_abs_upper_m_s=velocity_abs_upper_m_s,
    )
    if bool(np.any(status != NUMERICAL_STATUS_OK)):
        raise PhysicsEvaluationError("linear drag acceleration bound is not finite")
    return result


def linear_drag_acceleration_abs_upper_batch(
    *,
    rate_upper_s_inv: FloatArray,
    target_velocity_abs_upper_m_s: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    numerical_status: UInt8Array | None = None,
) -> tuple[FloatArray, UInt8Array]:
    """Bound linear drag while localizing row arithmetic overflow."""

    count = _common_count(rate_upper_s_inv)
    _require_nonnegative(rate_upper_s_inv, "rate_upper_s_inv")
    target = _component_abs_upper(
        target_velocity_abs_upper_m_s,
        "target_velocity_abs_upper_m_s",
    )
    velocity, finite_velocity = _continuous_velocity_bound(velocity_abs_upper_m_s, count)
    status = _force_numerical_status(numerical_status, count)
    status[(status == NUMERICAL_STATUS_OK) & ~finite_velocity] = PHYSICS_NUMERICAL_FAILURE
    safe_velocity = velocity.copy()
    safe_velocity[status != NUMERICAL_STATUS_OK] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        result = rate_upper_s_inv[:, None] * (target[None, :] + safe_velocity)
        result = np.nextafter(result * _BOUND_ROUNDOFF_FACTOR, np.inf)
    finite_result = np.isfinite(result).all(axis=1)
    status[(status == NUMERICAL_STATUS_OK) & ~finite_result] = PHYSICS_NUMERICAL_FAILURE
    result[status != NUMERICAL_STATUS_OK] = 0.0
    return result, status


def electric_acceleration_abs_upper(
    *,
    charge_number: FloatArray,
    mass_kg: FloatArray,
    electric_field_abs_upper_V_m: FloatArray,
) -> FloatArray:
    """Return a component-wise Coulomb-acceleration magnitude bound."""

    _common_count(charge_number, mass_kg)
    _require_finite(charge_number, "charge_number")
    _require_positive(mass_kg, "mass_kg")
    field = _component_abs_upper(
        electric_field_abs_upper_V_m,
        "electric_field_abs_upper_V_m",
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (
            np.abs(charge_number)[:, None] * ELEMENTARY_CHARGE_C * field[None, :] / mass_kg[:, None]
        )
    return _outward_abs_upper(result, "electric acceleration bound")


def quasistatic_spherical_dep_acceleration_abs_upper(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    gradient_mean_e_squared_abs_upper_V2_m3: FloatArray,
    medium_relative_permittivity: float,
    real_clausius_mossotti_factor: float,
) -> FloatArray:
    """Bound the component DEP acceleration for the versioned sphere model."""

    _common_count(mass_kg, electrostatic_radius_m)
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    gradient = _component_abs_upper(
        gradient_mean_e_squared_abs_upper_V2_m3,
        "gradient_mean_e_squared_abs_upper_V2_m3",
    )
    relative_permittivity = _finite_positive_scalar(
        medium_relative_permittivity,
        "medium_relative_permittivity",
    )
    if not math.isfinite(real_clausius_mossotti_factor) or not (
        -0.5 <= real_clausius_mossotti_factor <= 1.0
    ):
        raise PhysicsEvaluationError("real_clausius_mossotti_factor must be in [-0.5, 1]")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factor = (
            2.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * relative_permittivity
            * electrostatic_radius_m**3
            * abs(real_clausius_mossotti_factor)
            / mass_kg
        )
        result = factor[:, None] * gradient[None, :]
    return _outward_abs_upper(result, "quasistatic spherical DEP acceleration bound")


def gravity_buoyancy_acceleration_abs_upper(
    *,
    mass_kg: FloatArray,
    displaced_volume_m3: FloatArray,
    gas_density_lower_kg_m3: float,
    gas_density_upper_kg_m3: float,
    gravity_m_s2: tuple[float, float],
) -> FloatArray:
    """Bound gravity/buoyancy acceleration over a density interval."""

    _common_count(mass_kg, displaced_volume_m3)
    _require_positive(mass_kg, "mass_kg")
    _require_nonnegative(displaced_volume_m3, "displaced_volume_m3")
    density_lower = _finite_positive_scalar(
        gas_density_lower_kg_m3,
        "gas_density_lower_kg_m3",
    )
    density_upper = _finite_positive_scalar(
        gas_density_upper_kg_m3,
        "gas_density_upper_kg_m3",
    )
    if density_lower > density_upper:
        raise PhysicsEvaluationError("gas density bounds are reversed")
    gravity = np.asarray(gravity_m_s2, dtype=np.float64)
    if gravity.shape != (2,) or not bool(np.isfinite(gravity).all()):
        raise PhysicsEvaluationError("gravity_m_s2 must be a finite two-vector")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower_factor = 1.0 - density_lower * displaced_volume_m3 / mass_kg
        upper_factor = 1.0 - density_upper * displaced_volume_m3 / mass_kg
        factor_abs_upper = np.maximum(np.abs(lower_factor), np.abs(upper_factor))
        result = factor_abs_upper[:, None] * np.abs(gravity)[None, :]
    return _outward_abs_upper(result, "gravity/buoyancy acceleration bound")


def epstein_continuous_applicability_batch(
    *,
    drag_diameter_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    gas_temperature_lower_K: float,
    gas_mean_free_path_lower_m: float,
    gas_molecular_mass_kg: float,
    maximum_speed_ratio: float = EPSTEIN_MAX_SPEED_OVER_MEAN_THERMAL,
) -> tuple[BoolArray, UInt8Array]:
    """Certify Epstein applicability while localizing non-finite row arithmetic."""

    speed_ratio_limit = _sensitivity_maximum_speed_ratio(maximum_speed_ratio)
    return _epstein_continuous_applicability_batch(
        drag_diameter_m=drag_diameter_m,
        velocity_abs_upper_m_s=velocity_abs_upper_m_s,
        gas_velocity_abs_upper_m_s=gas_velocity_abs_upper_m_s,
        gas_temperature_lower_K=gas_temperature_lower_K,
        gas_mean_free_path_lower_m=gas_mean_free_path_lower_m,
        gas_molecular_mass_kg=gas_molecular_mass_kg,
        thermal_speed_squared_factor=8.0 / math.pi,
        maximum_speed_ratio=speed_ratio_limit,
    )


def epstein_finite_speed_continuous_applicability_batch(
    *,
    drag_diameter_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    gas_temperature_lower_K: float,
    gas_mean_free_path_lower_m: float,
    gas_molecular_mass_kg: float,
    maximum_speed_ratio: float,
) -> tuple[BoolArray, UInt8Array]:
    """Certify the declared finite-speed free-molecular envelope over a path."""

    return _epstein_continuous_applicability_batch(
        drag_diameter_m=drag_diameter_m,
        velocity_abs_upper_m_s=velocity_abs_upper_m_s,
        gas_velocity_abs_upper_m_s=gas_velocity_abs_upper_m_s,
        gas_temperature_lower_K=gas_temperature_lower_K,
        gas_mean_free_path_lower_m=gas_mean_free_path_lower_m,
        gas_molecular_mass_kg=gas_molecular_mass_kg,
        thermal_speed_squared_factor=2.0,
        maximum_speed_ratio=maximum_speed_ratio,
    )


def _epstein_continuous_applicability_batch(
    *,
    drag_diameter_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    gas_temperature_lower_K: float,
    gas_mean_free_path_lower_m: float,
    gas_molecular_mass_kg: float,
    thermal_speed_squared_factor: float,
    maximum_speed_ratio: float,
) -> tuple[BoolArray, UInt8Array]:
    """Shared continuous certificate for one explicit Epstein speed definition."""

    count = _common_count(drag_diameter_m)
    _require_positive(drag_diameter_m, "drag_diameter_m")
    velocity, finite_velocity = _continuous_velocity_bound(velocity_abs_upper_m_s, count)
    gas_velocity = _component_abs_upper(
        gas_velocity_abs_upper_m_s,
        "gas_velocity_abs_upper_m_s",
    )
    temperature_lower = _finite_positive_scalar(
        gas_temperature_lower_K,
        "gas_temperature_lower_K",
    )
    mean_free_path_lower = _finite_positive_scalar(
        gas_mean_free_path_lower_m,
        "gas_mean_free_path_lower_m",
    )
    molecular_mass = _finite_positive_scalar(
        gas_molecular_mass_kg,
        "gas_molecular_mass_kg",
    )
    speed_ratio_limit = _finite_positive_scalar(maximum_speed_ratio, "maximum_speed_ratio")

    radius_m = 0.5 * drag_diameter_m
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        required_mean_free_path = np.nextafter(
            EPSTEIN_MIN_LAMBDA_OVER_RADIUS * radius_m,
            np.inf,
        )
        relative_component_upper = velocity + gas_velocity[None, :]
        relative_speed_upper = np.hypot(
            relative_component_upper[:, 0],
            relative_component_upper[:, 1],
        )
        relative_speed_upper = np.nextafter(
            relative_speed_upper * _BOUND_ROUNDOFF_FACTOR,
            np.inf,
        )
        thermal_speed_lower = math.sqrt(
            thermal_speed_squared_factor * BOLTZMANN_J_K * temperature_lower / molecular_mass
        )
        speed_limit = np.nextafter(
            speed_ratio_limit * thermal_speed_lower / _BOUND_ROUNDOFF_FACTOR,
            -np.inf,
        )
    if not math.isfinite(speed_limit) or speed_limit <= 0.0:
        raise PhysicsEvaluationError("Epstein applicability bound is not finite")
    numerical_ok = finite_velocity & np.isfinite(required_mean_free_path)
    numerical_ok &= np.isfinite(relative_speed_upper)
    applicable = (mean_free_path_lower >= required_mean_free_path) & (
        relative_speed_upper <= speed_limit
    )
    applicable &= numerical_ok
    status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    return applicable, status


def epstein_continuous_applicability(
    *,
    drag_diameter_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    gas_velocity_abs_upper_m_s: FloatArray,
    gas_temperature_lower_K: float,
    gas_mean_free_path_lower_m: float,
    gas_molecular_mass_kg: float,
) -> BoolArray:
    """Certify the Epstein revision-1 envelope over a continuous path bound."""

    applicable, status = epstein_continuous_applicability_batch(
        drag_diameter_m=drag_diameter_m,
        velocity_abs_upper_m_s=velocity_abs_upper_m_s,
        gas_velocity_abs_upper_m_s=gas_velocity_abs_upper_m_s,
        gas_temperature_lower_K=gas_temperature_lower_K,
        gas_mean_free_path_lower_m=gas_mean_free_path_lower_m,
        gas_molecular_mass_kg=gas_molecular_mass_kg,
    )
    if bool(np.any(status != CONTINUOUS_APPLICABILITY_OK)):
        raise PhysicsEvaluationError("Epstein applicability bound is not finite")
    return applicable


def add_electric_coulomb_acceleration(
    acceleration_m_s2: FloatArray,
    *,
    charge_number: FloatArray,
    mass_kg: FloatArray,
    electric_field_V_m: FloatArray,
) -> None:
    """Add ``Z e E / mass`` to an existing acceleration buffer."""

    count = _common_count(charge_number, mass_kg)
    if acceleration_m_s2.shape != (count, 2) or electric_field_V_m.shape != (count, 2):
        raise PhysicsEvaluationError("electric acceleration inputs must have shape [N, 2]")
    _require_positive(mass_kg, "mass_kg")
    _require_finite(charge_number, "charge_number")
    _require_finite(electric_field_V_m, "electric_field")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        acceleration_m_s2 += (
            charge_number[:, None] * ELEMENTARY_CHARGE_C * electric_field_V_m / mass_kg[:, None]
        )
    _require_finite(acceleration_m_s2, "combined acceleration")


def quasistatic_spherical_dep_acceleration(
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    gradient_mean_e_squared_V2_m3: FloatArray,
    medium_relative_permittivity: float,
    real_clausius_mossotti_factor: float,
) -> FloatArray:
    """Evaluate ``2*pi*epsilon_m*a^3*K*grad(mean(E^2))/mass``."""

    count = _common_count(mass_kg, electrostatic_radius_m)
    if gradient_mean_e_squared_V2_m3.shape != (count, 2):
        raise PhysicsEvaluationError("DEP gradient input must have shape [N, 2]")
    _require_positive(mass_kg, "mass_kg")
    _require_positive(electrostatic_radius_m, "electrostatic_radius_m")
    _require_finite(gradient_mean_e_squared_V2_m3, "gradient_mean_e_squared")
    relative_permittivity = _finite_positive_scalar(
        medium_relative_permittivity,
        "medium_relative_permittivity",
    )
    if not math.isfinite(real_clausius_mossotti_factor) or not (
        -0.5 <= real_clausius_mossotti_factor <= 1.0
    ):
        raise PhysicsEvaluationError("real_clausius_mossotti_factor must be in [-0.5, 1]")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factor = (
            2.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * relative_permittivity
            * electrostatic_radius_m**3
            * real_clausius_mossotti_factor
            / mass_kg
        )
        result = factor[:, None] * gradient_mean_e_squared_V2_m3
    _require_finite(result, "quasistatic spherical DEP acceleration")
    return result


def add_quasistatic_spherical_dep_acceleration(
    acceleration_m_s2: FloatArray,
    *,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    gradient_mean_e_squared_V2_m3: FloatArray,
    medium_relative_permittivity: float,
    real_clausius_mossotti_factor: float,
) -> None:
    """Add the versioned quasistatic spherical DEP acceleration in place."""

    if acceleration_m_s2.shape != gradient_mean_e_squared_V2_m3.shape:
        raise PhysicsEvaluationError("DEP acceleration buffer must have shape [N, 2]")
    acceleration_m_s2 += quasistatic_spherical_dep_acceleration(
        mass_kg=mass_kg,
        electrostatic_radius_m=electrostatic_radius_m,
        gradient_mean_e_squared_V2_m3=gradient_mean_e_squared_V2_m3,
        medium_relative_permittivity=medium_relative_permittivity,
        real_clausius_mossotti_factor=real_clausius_mossotti_factor,
    )
    _require_finite(acceleration_m_s2, "combined acceleration")


def add_gravity_buoyancy_acceleration(
    acceleration_m_s2: FloatArray,
    *,
    mass_kg: FloatArray,
    displaced_volume_m3: FloatArray,
    gas_density_kg_m3: FloatArray,
    gravity_m_s2: tuple[float, float],
) -> None:
    """Add ``(1-rho*V/m) g`` using independent mass and displaced-volume authorities."""

    count = _common_count(mass_kg, displaced_volume_m3, gas_density_kg_m3)
    if acceleration_m_s2.shape != (count, 2):
        raise PhysicsEvaluationError("gravity acceleration buffer must have shape [N, 2]")
    _require_positive(mass_kg, "mass_kg")
    _require_nonnegative(displaced_volume_m3, "displaced_volume_m3")
    _require_positive(gas_density_kg_m3, "gas_density")
    gravity = np.asarray(gravity_m_s2, dtype=np.float64)
    if gravity.shape != (2,) or not bool(np.isfinite(gravity).all()):
        raise PhysicsEvaluationError("gravity_m_s2 must be a finite two-vector")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factor = 1.0 - gas_density_kg_m3 * displaced_volume_m3 / mass_kg
        acceleration_m_s2 += factor[:, None] * gravity
    _require_finite(acceleration_m_s2, "combined acceleration")


def _common_count(first: FloatArray, *rest: FloatArray) -> int:
    if first.ndim != 1:
        raise PhysicsEvaluationError("particle scalar inputs must be one-dimensional")
    count = int(first.size)
    if any(array.shape != (count,) for array in rest):
        raise PhysicsEvaluationError("particle scalar inputs must have matching shape [N]")
    return count


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


def _component_abs_upper(value: FloatArray, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,):
        raise PhysicsEvaluationError(f"{name} must have shape [2]")
    _require_nonnegative(result, name)
    return result


def _continuous_velocity_bound(
    value: FloatArray,
    count: int,
) -> tuple[FloatArray, BoolArray]:
    """Validate shared shape/sign and identify row-local non-finite bounds."""

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count, 2):
        raise PhysicsEvaluationError("velocity_abs_upper_m_s must have shape [N, 2]")
    finite = np.isfinite(result)
    if bool(((result < 0.0) & finite).any()):
        raise PhysicsEvaluationError("velocity_abs_upper_m_s must be nonnegative")
    return result, finite.all(axis=1)


def _force_numerical_status(value: UInt8Array | None, count: int) -> UInt8Array:
    if value is None:
        return np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    status = np.asarray(value)
    if status.shape != (count,) or status.dtype != np.uint8:
        raise PhysicsEvaluationError("numerical status must be a uint8 array with shape [N]")
    return status.copy()


def _finite_positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise PhysicsEvaluationError(f"{name} must be positive and finite")
    return float(value)


def _sensitivity_maximum_speed_ratio(value: float) -> float:
    result = _finite_positive_scalar(value, "maximum_speed_ratio")
    if result > 1.0:
        raise PhysicsEvaluationError("maximum_speed_ratio must be at most 1.0")
    return result


def _finite_nonnegative_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
        raise PhysicsEvaluationError(f"{name} must be nonnegative and finite")
    return float(value)


def _outward_abs_upper(value: FloatArray, name: str) -> FloatArray:
    """Expand a short nonnegative float64 expression toward positive infinity."""

    result = np.asarray(value, dtype=np.float64)
    _require_nonnegative(result, name)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.nextafter(result * _BOUND_ROUNDOFF_FACTOR, np.inf)
    if not bool(np.isfinite(result).all()):
        raise PhysicsEvaluationError(f"{name} is not finite")
    return result


def _outward_nonnegative_lower(value: FloatArray, name: str) -> FloatArray:
    """Contract a short nonnegative float64 expression toward zero."""

    result = np.asarray(value, dtype=np.float64)
    _require_nonnegative(result, name)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.nextafter(result / _BOUND_ROUNDOFF_FACTOR, 0.0)
    if not bool(np.isfinite(result).all()):
        raise PhysicsEvaluationError(f"{name} is not finite")
    return result
