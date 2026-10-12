"""Small deterministic physics runtime over sampled primitive values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..numerical_status import (
    INTEGRATOR_ACCURACY_FAILURE,
    NUMERICAL_STATUS_OK,
    PHYSICS_NUMERICAL_FAILURE,
)
from .catalog import (
    AggregateRelativeDriftChargePlan,
    BarnesCollisionlessIonDragPlan,
    ChargePlan,
    CoordinateSystem,
    DragPlan,
    ElectricFieldDirectedImageIonDragPlan,
    EpsteinDragPlan,
    FiniteSpeedEpsteinDragPlan,
    IonDragPlan,
    LiftPlan,
    PhysicsPlan,
    PlasmaContinuousChargePlan,
    QuasistaticSphericalDielectrophoresisPlan,
    RarefiedVorticityLiftPlan,
    RelativeFlowScreenedIonDragPlan,
    SaffmanLiftPlan,
    StokesCunninghamDragPlan,
    TalbotThermophoresisPlan,
    ThermophoresisPlan,
    WaldmannGallisThermophoresisPlan,
)
from .charge import (
    OML_MAX_ION_DRIFT_RATIO,
    OML_MAX_RADIUS_OVER_DEBYE,
    AggregateChargeBounds,
    OmlChargeBounds,
    aggregate_relative_drift_global_bounds,
    aggregate_relative_drift_three_current_global_bounds,
    oml_shifted_maxwellian_global_bounds,
    oml_stationary_global_bounds,
)
from .compiled import (
    CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT,
    CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_TWO_CURRENT,
    CHARGE_FIXED,
    CHARGE_OML_SHIFTED_MAXWELLIAN_SINGLE_ION_NEGATIVE_DEBYE_HUCKEL,
    CHARGE_OML_STATIONARY_MAXWELLIAN_DEBYE_HUCKEL,
    DRAG_EPSTEIN,
    DRAG_EPSTEIN_FINITE_SPEED,
    DRAG_NONE,
    DRAG_STOKES_CUNNINGHAM,
    ERROR_CHARGE_INVARIANT,
    ION_DRAG_BARNES_COLLISIONLESS_EFFECTIVE_SPEED,
    ION_DRAG_ELECTRIC_FIELD_DIRECTED_IMAGE,
    ION_DRAG_NONE,
    ION_DRAG_RELATIVE_FLOW_SCREENED,
    LIFT_NONE,
    LIFT_RAREFIED_VORTICITY_RZ,
    LIFT_SAFFMAN_RZ,
    LIFT_SAFFMAN_XY,
    THERMOPHORESIS_NONE,
    THERMOPHORESIS_TALBOT,
    THERMOPHORESIS_WALDMANN_GALLIS,
    evaluate_physics_tile_into,
)
from .forces import (
    BOLTZMANN_J_K,
    CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE,
    CONTINUOUS_APPLICABILITY_OK,
    ELEMENTARY_CHARGE_C,
    EPSTEIN_MIN_LAMBDA_OVER_RADIUS,
    ION_DRAG_MAX_SCALE_OVER_DEBYE,
    ION_DRAG_MIN_MEAN_FREE_PATH_OVER_DEBYE,
    RAREFIED_VORTICITY_LIFT_MIN_MEAN_FREE_PATH_OVER_RADIUS,
    SAFFMAN_MAX_MEAN_FREE_PATH_OVER_RADIUS,
    SAFFMAN_MAX_SHEAR_REYNOLDS,
    SAFFMAN_MAX_SLIP_REYNOLDS,
    SAFFMAN_MAX_SLIP_TO_SQRT_SHEAR_REYNOLDS,
    STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS,
    STOKES_CUNNINGHAM_MAX_REYNOLDS,
    STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS,
    VACUUM_PERMITTIVITY_F_M,
    WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS,
    PhysicsEvaluationError,
    _continuous_mean_thermal_speed_lower,
    _continuous_velocity_is_finite,
    _drift_continuous_applicability_row,
    _epstein_continuous_applicability_row,
    _epstein_continuous_speed_limit,
    _has_finite_negative_velocity,
    _outward_relative_speed,
    _relative_speed_continuous_applicability_row,
    _saffman_continuous_applicability_row,
    _stokes_continuous_applicability_row,
    add_electric_coulomb_acceleration,
    add_gravity_buoyancy_acceleration,
    add_quasistatic_spherical_dep_acceleration,
    barnes_collisionless_global_bounds,
    electric_acceleration_abs_upper,
    electric_field_directed_image_ion_drag_global_bound,
    epstein_finite_speed_factors,
    epstein_rate_s_inv,
    gravity_buoyancy_acceleration_abs_upper,
    quasistatic_spherical_dep_acceleration_abs_upper,
    rarefied_vorticity_lift_global_bounds,
    relative_flow_screened_ion_drag_global_bound,
    relative_flow_screened_ion_drag_local_bound_row,
    saffman_lift_global_bounds,
    stokes_cunningham_rate_upper_s_inv,
    talbot_thermophoresis_global_bounds,
    velocity_acceleration_abs_upper_row,
    waldmann_gallis_global_bounds,
)

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type Int64Array = NDArray[np.int64]
type UInt8Array = NDArray[np.uint8]

# Local certificates combine at most a few dozen elementary float64
# operations.  One nextafter after a compound expression is not sufficient to
# cover their accumulated rounding.  This factor matches the established
# global-bound convention and is applied before the final directed rounding.
_LOCAL_BOUND_ROUNDOFF_FACTOR = 1.0 + 64.0 * np.finfo(np.float64).eps

PHYSICS_RUNTIME_REVISION = "prepared_native_physics_runtime_v24"

_EMPTY_VECTOR = np.empty((0, 2), dtype=np.float64)
_EMPTY_BOUND_SCALAR = np.empty(0, dtype=np.float64)
_EMPTY_BOUND_APPLICABLE = np.empty(0, dtype=np.bool_)
_EMPTY_BOUND_SCALAR.setflags(write=False)
_EMPTY_BOUND_APPLICABLE.setflags(write=False)


@dataclass(frozen=True, slots=True)
class PrimitiveRange:
    """Canonical component extrema and an optional exact constant value."""

    lower: FloatArray
    upper: FloatArray
    constant: FloatArray | None


@dataclass(frozen=True, slots=True)
class LocalPrimitiveRange:
    """Per-path component extrema supplied by the field owner."""

    lower: FloatArray
    upper: FloatArray


@dataclass(frozen=True, slots=True)
class PhysicsRuntimeEvaluation:
    """Combined contribution at one integrator stage."""

    acceleration_m_s2: FloatArray
    charge_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray
    applicable: BoolArray
    linear_drag_rate_s_inv: FloatArray
    target_velocity_m_s: FloatArray
    additive_acceleration_m_s2: FloatArray


@dataclass(slots=True)
class PhysicsRuntimeWorkspace:
    """Reusable output columns for one physics stage slab."""

    acceleration_m_s2: FloatArray
    charge_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray
    applicable: BoolArray
    linear_drag_rate_s_inv: FloatArray
    target_velocity_m_s: FloatArray
    additive_acceleration_m_s2: FloatArray
    error_code: UInt8Array
    sampled_primitive_values: FloatArray
    incoming_status: UInt8Array

    @classmethod
    def allocate(cls, capacity: int) -> PhysicsRuntimeWorkspace:
        """Allocate the exact output columns for a bounded slab."""

        if capacity < 0:
            raise ValueError("physics workspace capacity must be nonnegative")
        return cls(
            np.empty((capacity, 2), dtype=np.float64),
            np.empty(capacity, dtype=np.float64),
            np.empty(capacity, dtype=np.float64),
            np.empty(capacity, dtype=np.bool_),
            np.empty(capacity, dtype=np.float64),
            np.empty((capacity, 2), dtype=np.float64),
            np.empty((capacity, 2), dtype=np.float64),
            np.empty(capacity, dtype=np.uint8),
            np.empty((capacity, 0), dtype=np.float64),
            np.empty(capacity, dtype=np.uint8),
        )

    def primitive_values(self, count: int, components: int) -> FloatArray:
        """Reuse a bounded primitive matrix for callers at the Python boundary."""

        if self.sampled_primitive_values.shape[1] != components:
            self.sampled_primitive_values = np.empty((self.capacity, components), dtype=np.float64)
        return self.sampled_primitive_values[:count]

    def numeric_outputs(self, count: int) -> PhysicsNumericOutputs:
        """Pass filled-prefix views directly to the native owner."""

        value = self.evaluation(count)
        return PhysicsNumericOutputs(
            value.acceleration_m_s2,
            value.charge_rate_number_s,
            value.charge_rate_derivative_s_inv,
            value.applicable,
            value.linear_drag_rate_s_inv,
            value.target_velocity_m_s,
            value.additive_acceleration_m_s2,
        )

    @property
    def capacity(self) -> int:
        """Return the number of particle rows owned by the workspace."""

        return int(self.charge_rate_number_s.size)

    def evaluation(self, count: int) -> PhysicsRuntimeEvaluation:
        """Expose only the filled prefix without copying numerical payload."""

        if count < 0 or count > self.capacity:
            raise ValueError("physics workspace does not cover the requested rows")
        return PhysicsRuntimeEvaluation(
            self.acceleration_m_s2[:count],
            self.charge_rate_number_s[:count],
            self.charge_rate_derivative_s_inv[:count],
            self.applicable[:count],
            self.linear_drag_rate_s_inv[:count],
            self.target_velocity_m_s[:count],
            self.additive_acceleration_m_s2[:count],
        )


@dataclass(frozen=True, slots=True)
class LinearRelaxationAbsBounds:
    """Particle-row bounds needed to enclose an exponential proposal."""

    rate_upper_s_inv: FloatArray
    target_velocity_abs_upper_m_s: FloatArray


@dataclass(frozen=True, slots=True)
class _EpsteinBounds:
    rate_upper_s_inv: FloatArray
    target_velocity_abs_upper_m_s: FloatArray
    temperature_lower_K: float
    mean_free_path_lower_m: float
    velocity_lipschitz_upper_s_inv: float


@dataclass(frozen=True, slots=True)
class _StokesCunninghamBounds:
    rate_upper_s_inv: FloatArray
    target_velocity_abs_upper_m_s: FloatArray
    density_upper_kg_m3: float
    dynamic_viscosity_lower_Pa_s: float
    mean_free_path_lower_m: float
    mean_free_path_upper_m: float


type _DragBounds = _EpsteinBounds | _StokesCunninghamBounds


@dataclass(frozen=True, slots=True)
class _OmlRuntimeBounds:
    model: OmlChargeBounds
    positive_ion_velocity_abs_upper_m_s: FloatArray
    positive_ion_temperature_lower_K: float


@dataclass(frozen=True, slots=True)
class _AggregateChargeRuntimeBounds:
    model: AggregateChargeBounds
    positive_ion_velocity_abs_upper_m_s: FloatArray
    negative_ion_velocity_abs_upper_m_s: FloatArray


type _ChargeRuntimeBounds = _OmlRuntimeBounds | _AggregateChargeRuntimeBounds


class _ChargeStageBindings(NamedTuple):
    """Resolved sampled inputs for one compiled continuous-charge pass."""

    code: int
    electron_density_m3: int
    positive_ion_density_m3: int
    electron_temperature_K: int
    positive_ion_temperature_K: int
    positive_ion_velocity_m_s: int
    positive_ion_mass_kg: float
    maximum_ion_drift_ratio: float
    aggregate_electron_thermal_voltage_V: int
    aggregate_positive_ion_thermal_voltage_V: int
    aggregate_effective_positive_ion_mass_kg: int
    aggregate_negative_ion_number_density_m3: int
    aggregate_negative_ion_thermal_voltage_V: int
    aggregate_negative_ion_velocity_m_s: int
    aggregate_effective_negative_ion_mass_kg: int
    aggregate_screening_length_m: int
    aggregate_maximum_relative_ion_speed_m_s: float
    charge_number_lower: float
    charge_number_upper: float


_NO_CHARGE_STAGE_INPUTS = _ChargeStageBindings(
    CHARGE_FIXED,
    -1,
    -1,
    -1,
    -1,
    -1,
    1.0,
    OML_MAX_ION_DRIFT_RATIO,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1.0,
    -math.inf,
    math.inf,
)


@dataclass(frozen=True, slots=True)
class _BarnesIonDragBounds:
    static_applicable: BoolArray
    positive_ion_velocity_abs_upper_m_s: FloatArray
    positive_ion_temperature_lower_K: float


@dataclass(frozen=True, slots=True)
class _RelativeFlowScreenedIonDragBounds:
    positive_ion_velocity_abs_upper_m_s: FloatArray


type _IonDragRuntimeBounds = _BarnesIonDragBounds | _RelativeFlowScreenedIonDragBounds


@dataclass(frozen=True, slots=True)
class _WaldmannGallisBounds:
    static_applicable: BoolArray
    gas_velocity_abs_upper_m_s: FloatArray
    gas_temperature_lower_K: float


@dataclass(frozen=True, slots=True)
class _TalbotBounds:
    static_applicable: BoolArray


type _ThermophoresisRuntimeBounds = _WaldmannGallisBounds | _TalbotBounds


@dataclass(frozen=True, slots=True)
class _RarefiedVorticityLiftBounds:
    """Prepared high-Kn certificate and cross-velocity acceleration bound."""

    coupling_rate_abs_upper_s_inv: FloatArray
    gas_velocity_abs_upper_m_s: FloatArray
    static_applicable: BoolArray


@dataclass(frozen=True, slots=True)
class _SaffmanLiftBounds:
    """Prepared coupling and continuum/creeping applicability bounds."""

    coupling_rate_abs_upper_s_inv: FloatArray
    gas_velocity_abs_upper_m_s: FloatArray
    shear_reynolds_lower: FloatArray
    shear_reynolds_upper: FloatArray
    static_applicable: BoolArray
    gas_density_upper_kg_m3: float
    gas_dynamic_viscosity_lower_Pa_s: float


type _LiftRuntimeBounds = _RarefiedVorticityLiftBounds | _SaffmanLiftBounds


class _ThermophoresisStageBindings(NamedTuple):
    """Sampled neutral-gas inputs for the selected thermophoresis revision."""

    code: int
    gas_velocity_m_s: int
    gas_temperature_K: int
    gas_translational_heat_flux_W_m2: int
    gas_mean_free_path_m: int
    gas_molecular_mass_kg: float
    maximum_speed_ratio: float
    gas_temperature_gradient_K_m: int
    gas_density_kg_m3: int
    gas_dynamic_viscosity_Pa_s: int
    gas_thermal_conductivity_W_m_K: int
    particle_thermal_conductivity_W_m_K: float
    thermal_slip_coefficient: float
    momentum_exchange_coefficient: float
    thermal_exchange_coefficient: float


_NO_THERMOPHORESIS_STAGE_INPUTS = _ThermophoresisStageBindings(
    THERMOPHORESIS_NONE, -1, -1, -1, -1, 1.0, 1.0, -1, -1, -1, -1, 1.0, 1.0, 1.0, 1.0
)


class _LiftStageBindings(NamedTuple):
    """Sampled neutral-gas inputs for the selected lift revision."""

    code: int
    gas_velocity_m_s: int
    gas_density_kg_m3: int
    gas_mean_free_path_m: int
    gas_dynamic_viscosity_Pa_s: int
    azimuthal_gas_vorticity_s_inv: int
    lift_coefficient: float


_NO_LIFT_STAGE_INPUTS = _LiftStageBindings(LIFT_NONE, -1, -1, -1, -1, -1, 1.0)


class _IonDragStageBindings(NamedTuple):
    """Sampled inputs for one selected ion-drag revision."""

    code: int
    electron_density_m3: int
    positive_ion_density_m3: int
    electron_temperature_K: int
    positive_ion_temperature_K: int
    positive_ion_velocity_m_s: int
    ion_neutral_mean_free_path_m: int
    positive_ion_mass_kg: float
    maximum_ion_drift_ratio: float
    electron_thermal_voltage_V: int
    positive_ion_thermal_voltage_V: int
    effective_positive_ion_mass_kg: int
    screening_length_m: int
    electric_field_V_m: int
    maximum_relative_ion_speed_m_s: float


_NO_ION_DRAG_STAGE_INPUTS = _IonDragStageBindings(
    ION_DRAG_NONE, -1, -1, -1, -1, -1, -1, 1.0, 1.0, -1, -1, -1, -1, -1, 1.0
)


class PhysicsStageBindings(NamedTuple):
    """Closed numeric projection of the resolved catalog; primitive offsets are prepare-only."""

    charge_inputs: _ChargeStageBindings
    drag_code: int
    gas_velocity: int
    gas_density: int
    gas_temperature: int
    gas_viscosity: int
    gas_mean_free_path: int
    gas_molecular_mass_kg: float
    epstein_delta: float
    epstein_diffuse_reflection_fraction: float
    epstein_maximum_speed_ratio: float
    thermophoresis_inputs: _ThermophoresisStageBindings
    electric_enabled: bool
    electric_field: int
    ion_drag_inputs: _IonDragStageBindings
    dep_enabled: bool
    dep_gradient_mean_e_squared: int
    dep_medium_relative_permittivity: float
    dep_real_clausius_mossotti_factor: float
    lift_inputs: _LiftStageBindings
    gravity_enabled: bool
    gravity_density: int
    gravity_vector: tuple[float, float]
    primitive_rules: NDArray[np.int8]


class PhysicsNumericContext(NamedTuple):
    """Read-only resident model arrays and one closed stage binding table."""

    mass_kg: FloatArray
    drag_diameter_m: FloatArray
    electrostatic_radius_m: FloatArray
    displaced_volume_m3: FloatArray
    bindings: PhysicsStageBindings


class AccelerationBoundContext(NamedTuple):
    """Borrow prepared arrays; native enclosure passes never gather resident rows."""

    external_abs_upper_m_s2: FloatArray
    local_base_abs_upper_m_s2: FloatArray
    local_available: bool
    drag_rate_upper_s_inv: FloatArray
    drag_target_abs_upper_m_s: FloatArray
    lift_rate_upper_s_inv: FloatArray
    lift_target_abs_upper_m_s: FloatArray


class _DragApplicabilityContext(NamedTuple):
    code: int
    flow_abs_upper_m_s: FloatArray
    speed_limit_m_s: float
    mean_free_path_lower_m: float
    mean_free_path_upper_m: float
    density_upper_kg_m3: float
    viscosity_lower_Pa_s: float


class _ChargeApplicabilityContext(NamedTuple):
    code: int
    positive_flow_abs_upper_m_s: FloatArray
    negative_flow_abs_upper_m_s: FloatArray
    thermal_speed_lower_m_s: float
    debye_length_lower_m: float
    maximum_drift_or_speed: float
    charge_number_lower: float
    charge_number_upper: float


class _ThermophoresisApplicabilityContext(NamedTuple):
    code: int
    static_applicable: BoolArray
    flow_abs_upper_m_s: FloatArray
    thermal_speed_lower_m_s: float
    maximum_speed_ratio: float


class _IonDragApplicabilityContext(NamedTuple):
    code: int
    static_applicable: BoolArray
    flow_abs_upper_m_s: FloatArray
    thermal_speed_lower_m_s: float
    maximum_drift_or_speed: float


class _LiftApplicabilityContext(NamedTuple):
    code: int
    static_applicable: BoolArray
    flow_abs_upper_m_s: FloatArray
    density_upper_kg_m3: float
    viscosity_lower_Pa_s: float
    shear_reynolds_lower: FloatArray
    shear_reynolds_upper: FloatArray


class ContinuousApplicabilityContext(NamedTuple):
    """Closed prepared bounds, sharing their resident arrays without copies."""

    drag_diameter_m: FloatArray
    electrostatic_radius_m: FloatArray
    drag: _DragApplicabilityContext
    charge: _ChargeApplicabilityContext
    thermophoresis: _ThermophoresisApplicabilityContext
    ion_drag: _IonDragApplicabilityContext
    lift: _LiftApplicabilityContext
    reject_negative_velocity: bool


@dataclass(frozen=True, slots=True)
class PhysicsRuntime:
    """Evaluate one resolved plan without locating fields or changing state."""

    plan: PhysicsPlan
    mass_kg: FloatArray
    drag_diameter_m: FloatArray
    electrostatic_radius_m: FloatArray
    displaced_volume_m3: FloatArray
    charge_bounds: _ChargeRuntimeBounds | None
    drag_bounds: _DragBounds | None
    thermophoresis_bounds: _ThermophoresisRuntimeBounds | None
    ion_drag_bounds: _IonDragRuntimeBounds | None
    lift_bounds: _LiftRuntimeBounds | None
    external_acceleration_abs_upper_m_s2: FloatArray
    localizable_external_base_abs_upper_m_s2: FloatArray | None
    constant_acceleration_m_s2: FloatArray | None
    native_context: PhysicsNumericContext
    native_applicability_context: ContinuousApplicabilityContext
    native_acceleration_bound_context: AccelerationBoundContext

    def brownian_thermal_velocity_variance(
        self,
        particle_index: Int64Array,
        sampled_values: Mapping[str, FloatArray],
    ) -> FloatArray:
        """Return strict FDT variance ``k_B T / m`` for verification callers."""

        result, status = self.brownian_thermal_velocity_variance_batch(
            particle_index,
            sampled_values,
        )
        if bool((status != NUMERICAL_STATUS_OK).any()):
            raise PhysicsEvaluationError(
                "Brownian thermal velocity variance must be finite and positive"
            )
        return result

    def brownian_thermal_velocity_variance_batch(
        self,
        particle_index: Int64Array,
        sampled_values: Mapping[str, FloatArray],
    ) -> tuple[FloatArray, UInt8Array]:
        """Return FDT variance and particle-local numerical status.

        Shape/schema errors remain run-fatal.  A sampled value or arithmetic
        result that is invalid for only one row is represented by the shared
        numerical-status vocabulary so a valid neighbour can continue.
        """

        if self.plan.noise is None or not isinstance(self.plan.drag, EpsteinDragPlan):
            raise PhysicsEvaluationError(
                "Brownian thermal variance requires resolved inertial Langevin noise"
            )
        indices = np.asarray(particle_index, dtype=np.int64)
        if indices.ndim != 1:
            raise PhysicsEvaluationError("physics particle indices must have shape [N]")
        if bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError("physics particle indices are outside resident state")
        count = int(indices.size)
        status = np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
        temperature = sampled_values.get(self.plan.drag.gas_temperature_field)
        if temperature is None or temperature.shape != (count, 1):
            raise PhysicsEvaluationError("sampled Brownian temperature has an invalid shape")
        result = np.empty(count, dtype=np.float64)
        brownian_thermal_variance_into(self.mass_kg, indices, temperature[:, 0], result, status)
        return result, status

    def evaluate(
        self,
        particle_index: Int64Array,
        velocity_m_s: FloatArray,
        charge_number: FloatArray,
        sampled_values: Mapping[str, FloatArray],
        workspace: PhysicsRuntimeWorkspace | None = None,
    ) -> PhysicsRuntimeEvaluation:
        """Combine charge, drag, optional lift, and the remaining additive forces."""

        evaluation, numerical_status = self.evaluate_batch(
            particle_index,
            velocity_m_s,
            charge_number,
            sampled_values,
            workspace=workspace,
        )
        if bool(np.any(numerical_status != NUMERICAL_STATUS_OK)):
            raise PhysicsEvaluationError("combined physics acceleration is not finite")
        return evaluation

    def evaluate_batch(
        self,
        particle_index: Int64Array,
        velocity_m_s: FloatArray,
        charge_number: FloatArray,
        sampled_values: Mapping[str, FloatArray],
        numerical_status: UInt8Array | None = None,
        workspace: PhysicsRuntimeWorkspace | None = None,
    ) -> tuple[PhysicsRuntimeEvaluation, UInt8Array]:
        """Pack the Python boundary once and enter the one prepared native physics pass."""

        indices = np.asarray(particle_index, dtype=np.int64)
        velocity = np.asarray(velocity_m_s, dtype=np.float64)
        charge = np.asarray(charge_number, dtype=np.float64)
        count = int(indices.size)
        if indices.ndim != 1 or velocity.shape != (count, 2) or charge.shape != (count,):
            raise PhysicsEvaluationError("physics stage state has an invalid shape")
        if bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError("physics particle indices are outside resident state")
        stage_workspace = (
            PhysicsRuntimeWorkspace.allocate(count) if workspace is None else workspace
        )
        evaluation = stage_workspace.evaluation(count)
        values = stage_workspace.primitive_values(
            count, self.native_context.bindings.primitive_rules.size
        )
        offset = 0
        for item in self.plan.required_fields:
            width = len(item.components)
            value = sampled_values.get(item.name)
            if value is None or value.shape != (count, width):
                raise PhysicsEvaluationError(f"sampled field {item.name!r} has an invalid shape")
            values[:, offset : offset + width] = value
            offset += width
        error_code = stage_workspace.error_code[:count]
        error_code[:] = _initial_numerical_status(numerical_status, count)
        evaluate_prepared_physics_into(
            self.native_context,
            indices,
            velocity,
            charge,
            values,
            stage_workspace.numeric_outputs(count),
            error_code,
            stage_workspace.incoming_status[:count],
        )
        return evaluation, error_code

    def linear_relaxation_abs_bounds(
        self,
        particle_index: Int64Array,
    ) -> LinearRelaxationAbsBounds:
        """Return prepared coefficient bounds for selected resident rows."""

        indices = np.asarray(particle_index, dtype=np.int64)
        if indices.ndim != 1 or bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError("physics particle indices are outside resident state")
        count = int(indices.size)
        rate_upper = np.empty(count, dtype=np.float64)
        target_abs_upper = np.empty((count, 2), dtype=np.float64)
        linear_relaxation_abs_bounds_into(
            self.native_acceleration_bound_context, indices, rate_upper, target_abs_upper
        )
        return LinearRelaxationAbsBounds(rate_upper, target_abs_upper)

    def additive_acceleration_abs_upper(
        self,
        particle_index: Int64Array,
        velocity_abs_upper_m_s: FloatArray,
    ) -> FloatArray:
        """Bound all non-drag acceleration over a component velocity box."""

        result, numerical_status = self.additive_acceleration_abs_upper_batch(
            particle_index,
            velocity_abs_upper_m_s,
        )
        if bool(np.any(numerical_status != NUMERICAL_STATUS_OK)):
            raise PhysicsEvaluationError("additive acceleration bound is not finite")
        return result

    def additive_acceleration_abs_upper_batch(
        self,
        particle_index: Int64Array,
        velocity_abs_upper_m_s: FloatArray,
        numerical_status: UInt8Array | None = None,
    ) -> tuple[FloatArray, UInt8Array]:
        """Bound non-drag acceleration while preserving row-local failures."""

        indices = np.asarray(particle_index, dtype=np.int64)
        velocity = np.asarray(velocity_abs_upper_m_s, dtype=np.float64)
        count = int(indices.size)
        if indices.ndim != 1 or velocity.shape != (count, 2):
            raise PhysicsEvaluationError("additive acceleration bound state has an invalid shape")
        if bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError("physics particle indices are outside resident state")
        status = _initial_numerical_status(numerical_status, count)
        _validate_bound_velocity(self.native_acceleration_bound_context, velocity, False)
        result = np.empty((count, 2), dtype=np.float64)
        acceleration_abs_upper_into(
            self.native_acceleration_bound_context, indices, velocity, result, status, False
        )
        return result, status

    def local_additive_acceleration_abs_upper_batch(
        self,
        particle_index: Int64Array,
        velocity_lower_m_s: FloatArray,
        velocity_upper_m_s: FloatArray,
        charge_lower_number: FloatArray,
        charge_upper_number: FloatArray,
        primitive_ranges: Mapping[str, LocalPrimitiveRange],
    ) -> tuple[FloatArray, BoolArray, UInt8Array]:
        """Bound midpoint additive acceleration over one local state box.

        The compact local path is available only when every charge-dependent
        force has a local interval implementation.  Other additive terms keep
        their prepared run-global bound, so this method never weakens an
        existing certificate.
        """

        indices = np.asarray(particle_index, dtype=np.int64)
        velocity_lower = np.asarray(velocity_lower_m_s, dtype=np.float64)
        velocity_upper = np.asarray(velocity_upper_m_s, dtype=np.float64)
        charge_lower = np.asarray(charge_lower_number, dtype=np.float64)
        charge_upper = np.asarray(charge_upper_number, dtype=np.float64)
        count = int(indices.size)
        if (
            indices.ndim != 1
            or velocity_lower.shape != (count, 2)
            or velocity_upper.shape != (count, 2)
            or charge_lower.shape != (count,)
            or charge_upper.shape != (count,)
        ):
            raise PhysicsEvaluationError("local additive-acceleration state has an invalid shape")
        if bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError(
                "local additive-acceleration particle indices are outside resident state"
            )
        if self.localizable_external_base_abs_upper_m_s2 is None:
            raise PhysicsEvaluationError(
                "selected charge-dependent forces have no local acceleration certificate"
            )
        lower, upper = _pack_local_ranges(self, primitive_ranges, count, True)
        result = np.empty((count, 2), dtype=np.float64)
        applicable = np.empty(count, dtype=np.bool_)
        status = np.empty(count, dtype=np.uint8)
        local_additive_acceleration_abs_upper_into(
            self.native_context,
            self.native_acceleration_bound_context,
            indices,
            velocity_lower,
            velocity_upper,
            charge_lower,
            charge_upper,
            lower,
            upper,
            np.ones(count, dtype=np.bool_),
            result,
            applicable,
            status,
        )
        return result, applicable, status

    def acceleration_abs_upper(
        self,
        particle_index: Int64Array,
        velocity_abs_upper_m_s: FloatArray,
    ) -> FloatArray:
        """Bound every configured acceleration component for an RK stage box."""

        result, numerical_status = self.acceleration_abs_upper_batch(
            particle_index,
            velocity_abs_upper_m_s,
        )
        if bool(np.any(numerical_status != NUMERICAL_STATUS_OK)):
            raise PhysicsEvaluationError("combined acceleration bound is not finite")
        return result

    def acceleration_abs_upper_batch(
        self,
        particle_index: Int64Array,
        velocity_abs_upper_m_s: FloatArray,
        numerical_status: UInt8Array | None = None,
    ) -> tuple[FloatArray, UInt8Array]:
        """Bound acceleration while preserving row-local numerical failures."""

        indices = np.asarray(particle_index, dtype=np.int64)
        velocity = np.asarray(velocity_abs_upper_m_s, dtype=np.float64)
        count = int(indices.size)
        if indices.ndim != 1 or velocity.shape != (count, 2):
            raise PhysicsEvaluationError("acceleration bound state has an invalid shape")
        if bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError("physics particle indices are outside resident state")
        _validate_bound_velocity(self.native_acceleration_bound_context, velocity, True)
        result = np.empty((count, 2), dtype=np.float64)
        status = _initial_numerical_status(numerical_status, count)
        acceleration_abs_upper_into(
            self.native_acceleration_bound_context, indices, velocity, result, status, True
        )
        return result, status

    def continuous_applicability(
        self,
        particle_index: Int64Array,
        velocity_abs_upper_m_s: FloatArray,
    ) -> BoolArray:
        """Certify every velocity-dependent model over one path enclosure."""

        applicable, status = self.continuous_applicability_batch(
            particle_index,
            velocity_abs_upper_m_s,
        )
        if bool(np.any(status != CONTINUOUS_APPLICABILITY_OK)):
            raise PhysicsEvaluationError("continuous applicability bound is not finite")
        return applicable

    def continuous_applicability_batch(
        self,
        particle_index: Int64Array,
        velocity_abs_upper_m_s: FloatArray,
    ) -> tuple[BoolArray, UInt8Array]:
        """Return applicability and row-local numerical status without retrying rows."""

        indices = np.asarray(particle_index, dtype=np.int64)
        velocity = np.asarray(velocity_abs_upper_m_s, dtype=np.float64)
        count = int(indices.size)
        if indices.ndim != 1 or velocity.shape != (count, 2):
            raise PhysicsEvaluationError("continuous applicability state has an invalid shape")
        if bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError(
                "continuous applicability particle indices are outside resident state"
            )

        applicable = np.empty(count, dtype=np.bool_)
        status = np.empty(count, dtype=np.uint8)
        evaluate_continuous_applicability_into(
            self.native_applicability_context,
            indices,
            velocity,
            applicable,
            status,
        )
        return applicable, status

    def charge_interval_inside_prepared_invariant(
        self,
        charge_lower_number: FloatArray,
        charge_upper_number: FloatArray,
    ) -> tuple[BoolArray, UInt8Array]:
        """Certify a dense charge interval against the prepared model domain."""

        lower = np.asarray(charge_lower_number, dtype=np.float64)
        upper = np.asarray(charge_upper_number, dtype=np.float64)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise PhysicsEvaluationError("charge interval must have matching shape [N]")
        status = np.full(lower.size, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
        certified = np.ones(lower.size, dtype=np.bool_)
        charge_interval_inside_prepared_invariant_into(
            self.native_applicability_context,
            lower,
            upper,
            certified,
            status,
        )
        return certified, status

    def prepared_charge_invariant_interval(
        self,
        reference_charge_number: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the conservative charge range valid for every shortened path.

        Continuous-charge preparation proves one forward-invariant interval
        over all admitted primitives.  A case without continuous charging
        retains its reference charge exactly.
        """

        reference = np.asarray(reference_charge_number, dtype=np.float64)
        if reference.ndim != 1 or not bool(np.isfinite(reference).all()):
            raise PhysicsEvaluationError("reference charge must be one finite column")
        return _ion_drag_charge_interval(reference, self.charge_bounds)

    def local_continuous_applicability_batch(
        self,
        particle_index: Int64Array,
        velocity_lower_m_s: FloatArray,
        velocity_upper_m_s: FloatArray,
        charge_lower_number: FloatArray,
        charge_upper_number: FloatArray,
        primitive_ranges: Mapping[str, LocalPrimitiveRange],
    ) -> tuple[BoolArray, UInt8Array]:
        """Certify model applicability over one signed local path box.

        A false verdict means only that the supplied interval is not proven
        safe.  The engine owns subdivision and distinguishes that outcome from
        an actual invalid stage or dense-path sample.
        """

        indices = np.asarray(particle_index, dtype=np.int64)
        velocity_lower = np.asarray(velocity_lower_m_s, dtype=np.float64)
        velocity_upper = np.asarray(velocity_upper_m_s, dtype=np.float64)
        charge_lower = np.asarray(charge_lower_number, dtype=np.float64)
        charge_upper = np.asarray(charge_upper_number, dtype=np.float64)
        count = int(indices.size)
        if (
            indices.ndim != 1
            or velocity_lower.shape != (count, 2)
            or velocity_upper.shape != (count, 2)
            or charge_lower.shape != (count,)
            or charge_upper.shape != (count,)
        ):
            raise PhysicsEvaluationError("local applicability state has an invalid shape")
        if bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError(
                "local applicability particle indices are outside resident state"
            )
        lower, upper = _pack_local_ranges(self, primitive_ranges, count, False)
        certified = np.empty(count, dtype=np.bool_)
        status = np.empty(count, dtype=np.uint8)
        evaluate_local_continuous_applicability_into(
            self.native_context,
            indices,
            velocity_lower,
            velocity_upper,
            charge_lower,
            charge_upper,
            lower,
            upper,
            np.ones(count, dtype=np.bool_),
            certified,
            status,
        )
        return certified, status

    def maximum_dt_over_tau(self, dt_s: float) -> float:
        """Return the prepared explicit-RK stiffness measure."""

        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise PhysicsEvaluationError("dt_s must be positive and finite")
        if self.drag_bounds is None:
            return 0.0
        bounds = self.drag_bounds
        maximum_rate = float(np.max(bounds.rate_upper_s_inv))
        if isinstance(bounds, _EpsteinBounds):
            maximum_rate = bounds.velocity_lipschitz_upper_s_inv
        result = dt_s * maximum_rate
        if not math.isfinite(result):
            raise PhysicsEvaluationError("linear-drag dt/tau bound is not finite")
        return result

    def maximum_dt_charge_lipschitz(self, dt_s: float) -> float:
        """Return the run-wide explicit charge stiffness certificate ``dt*L_Z``."""

        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise PhysicsEvaluationError("dt_s must be positive and finite")
        if self.charge_bounds is None:
            return 0.0
        result = dt_s * self.charge_bounds.model.charge_rate_derivative_abs_upper_s_inv
        if not math.isfinite(result):
            raise PhysicsEvaluationError("continuous-charge dt*L_Z bound is not finite")
        return result

    @property
    def maximum_charge_rate_abs_number_s(self) -> float:
        """Return the prepared run-wide ``abs(dZ/dt)`` upper bound."""

        if self.charge_bounds is None:
            return 0.0
        return self.charge_bounds.model.charge_rate_abs_upper_number_s

    @property
    def bound_array_nbytes(self) -> int:
        """Bytes owned by prepared runtime bounds for engine memory accounting."""

        total = int(self.external_acceleration_abs_upper_m_s2.nbytes)
        if self.localizable_external_base_abs_upper_m_s2 is not None:
            total += int(self.localizable_external_base_abs_upper_m_s2.nbytes)
        if self.drag_bounds is not None:
            total += int(self.drag_bounds.rate_upper_s_inv.nbytes)
            total += int(self.drag_bounds.target_velocity_abs_upper_m_s.nbytes)
        if self.charge_bounds is not None:
            total += int(self.charge_bounds.positive_ion_velocity_abs_upper_m_s.nbytes)
            if isinstance(self.charge_bounds, _AggregateChargeRuntimeBounds):
                total += int(self.charge_bounds.negative_ion_velocity_abs_upper_m_s.nbytes)
            total += 9 * np.dtype(np.float64).itemsize
        if self.thermophoresis_bounds is not None:
            total += int(self.thermophoresis_bounds.static_applicable.nbytes)
            if isinstance(self.thermophoresis_bounds, _WaldmannGallisBounds):
                total += int(self.thermophoresis_bounds.gas_velocity_abs_upper_m_s.nbytes)
                total += np.dtype(np.float64).itemsize
        if self.ion_drag_bounds is not None:
            total += int(self.ion_drag_bounds.positive_ion_velocity_abs_upper_m_s.nbytes)
            if isinstance(self.ion_drag_bounds, _BarnesIonDragBounds):
                total += int(self.ion_drag_bounds.static_applicable.nbytes)
                total += np.dtype(np.float64).itemsize
        if self.lift_bounds is not None:
            total += int(self.lift_bounds.coupling_rate_abs_upper_s_inv.nbytes)
            total += int(self.lift_bounds.gas_velocity_abs_upper_m_s.nbytes)
            total += int(self.lift_bounds.static_applicable.nbytes)
            if isinstance(self.lift_bounds, _SaffmanLiftBounds):
                total += int(self.lift_bounds.shear_reynolds_lower.nbytes)
                total += int(self.lift_bounds.shear_reynolds_upper.nbytes)
        if self.constant_acceleration_m_s2 is not None:
            total += int(self.constant_acceleration_m_s2.nbytes)
        return total + int(self.native_context.bindings.primitive_rules.nbytes)


def _validate_bound_velocity(
    context: AccelerationBoundContext, velocity: FloatArray, include_drag: bool
) -> None:
    """Reject malformed finite negative magnitude bounds at the host boundary."""

    if context.lift_rate_upper_s_inv.size or (include_drag and context.drag_rate_upper_s_inv.size):
        if _has_finite_negative_velocity(velocity):
            raise PhysicsEvaluationError("velocity_abs_upper_m_s must be nonnegative")


@njit(cache=True, fastmath=False, error_model="numpy")
def linear_relaxation_abs_bounds_into(
    context: AccelerationBoundContext,
    indices: Int64Array,
    rate_upper: FloatArray,
    target_upper: FloatArray,
) -> None:
    """Copy prepared coefficients into caller-owned filled-prefix columns."""

    for row in range(indices.size):
        rate_upper[row], target_upper[row, 0], target_upper[row, 1] = 0.0, 0.0, 0.0
        if context.drag_rate_upper_s_inv.size:
            rate_upper[row] = context.drag_rate_upper_s_inv[indices[row]]
            target_upper[row, 0] = context.drag_target_abs_upper_m_s[0]
            target_upper[row, 1] = context.drag_target_abs_upper_m_s[1]


@njit(cache=True, fastmath=False, error_model="numpy")
def acceleration_abs_upper_into(
    context: AccelerationBoundContext,
    indices: Int64Array,
    velocity: FloatArray,
    result: FloatArray,
    status: UInt8Array,
    include_drag: bool,
) -> None:
    """Compose the same additive-then-drag bounds without transient arrays."""

    for row in range(indices.size):
        particle = indices[row]
        value_x = context.external_abs_upper_m_s2[particle, 0]
        value_y = context.external_abs_upper_m_s2[particle, 1]
        if context.lift_rate_upper_s_inv.size:
            lift_x, lift_y, status[row] = velocity_acceleration_abs_upper_row(
                context.lift_rate_upper_s_inv[particle],
                context.lift_target_abs_upper_m_s[0],
                context.lift_target_abs_upper_m_s[1],
                velocity[row, 0],
                velocity[row, 1],
                True,
                status[row],
            )
            value_x = np.nextafter(value_x + lift_x, np.inf)
            value_y = np.nextafter(value_y + lift_y, np.inf)
        value_x, value_y, status[row] = _finish_acceleration_bound_row(
            value_x, value_y, status[row]
        )
        if include_drag and context.drag_rate_upper_s_inv.size:
            drag_x, drag_y, status[row] = velocity_acceleration_abs_upper_row(
                context.drag_rate_upper_s_inv[particle],
                context.drag_target_abs_upper_m_s[0],
                context.drag_target_abs_upper_m_s[1],
                velocity[row, 0],
                velocity[row, 1],
                False,
                status[row],
            )
            value_x = np.nextafter(value_x + drag_x, np.inf)
            value_y = np.nextafter(value_y + drag_y, np.inf)
        result[row, 0], result[row, 1], status[row] = _finish_acceleration_bound_row(
            value_x, value_y, status[row]
        )


@njit(cache=True, fastmath=False, error_model="numpy")
def _finish_acceleration_bound_row(
    x: float, y: float, incoming_status: int | np.uint8
) -> tuple[float, float, int]:
    status = int(incoming_status)
    if status == NUMERICAL_STATUS_OK and not (
        math.isfinite(x) and math.isfinite(y) and x >= 0.0 and y >= 0.0
    ):
        status = int(PHYSICS_NUMERICAL_FAILURE)
    if status != NUMERICAL_STATUS_OK:
        x, y = 0.0, 0.0
    return x, y, status


def _charge_stage_bindings(
    plan: ChargePlan | None, bounds: _ChargeRuntimeBounds | None, sampled_values: Mapping[str, int]
) -> _ChargeStageBindings:
    """Bind the selected charge revision to one stage's sampled primitives."""
    if plan is None:
        if bounds is not None:
            raise PhysicsEvaluationError("continuous charge bounds do not match the resolved model")
        return _NO_CHARGE_STAGE_INPUTS
    if bounds is None:
        raise PhysicsEvaluationError("continuous charge bounds were not prepared")
    electron_density = _primitive_binding(sampled_values, plan.electron_number_density_field)
    ion_density = _primitive_binding(sampled_values, plan.positive_ion_number_density_field)
    ion_velocity = _primitive_binding(sampled_values, plan.positive_ion_velocity_field)
    charge_lower = bounds.model.charge_number_lower
    charge_upper = bounds.model.charge_number_upper
    if isinstance(plan, PlasmaContinuousChargePlan):
        charge_code, maximum_drift_ratio = _charge_kernel_selection(plan)
        electron_temperature = _primitive_binding(sampled_values, plan.electron_temperature_field)
        ion_temperature = _primitive_binding(sampled_values, plan.positive_ion_temperature_field)
        return _ChargeStageBindings(
            charge_code,
            electron_density,
            ion_density,
            electron_temperature,
            ion_temperature,
            ion_velocity,
            plan.positive_ion_mass_kg,
            maximum_drift_ratio,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1.0,
            charge_lower,
            charge_upper,
        )
    charge_code, maximum_relative_speed = _charge_kernel_selection(plan)
    electron_voltage = _primitive_binding(sampled_values, plan.electron_thermal_voltage_field)
    ion_voltage = _primitive_binding(sampled_values, plan.positive_ion_thermal_voltage_field)
    ion_mass = _primitive_binding(sampled_values, plan.effective_positive_ion_mass_field)
    negative_density = -1
    negative_voltage = -1
    negative_velocity = -1
    negative_mass = -1
    if plan.has_negative_ion_current:
        negative_density = _primitive_binding(
            sampled_values, cast(str, plan.negative_ion_number_density_field)
        )
        negative_voltage = _primitive_binding(
            sampled_values, cast(str, plan.negative_ion_thermal_voltage_field)
        )
        negative_velocity = _primitive_binding(
            sampled_values, cast(str, plan.negative_ion_velocity_field)
        )
        negative_mass = _primitive_binding(
            sampled_values, cast(str, plan.effective_negative_ion_mass_field)
        )
    screening_length = _primitive_binding(sampled_values, plan.screening_length_field)
    return _ChargeStageBindings(
        charge_code,
        electron_density,
        ion_density,
        -1,
        -1,
        ion_velocity,
        1.0,
        OML_MAX_ION_DRIFT_RATIO,
        electron_voltage,
        ion_voltage,
        ion_mass,
        negative_density,
        negative_voltage,
        negative_velocity,
        negative_mass,
        screening_length,
        maximum_relative_speed,
        charge_lower,
        charge_upper,
    )


def _thermophoresis_stage_bindings(
    plan: ThermophoresisPlan | None,
    drag_plan: DragPlan | None,
    sampled_values: Mapping[str, int],
    drag_gas_velocity_m_s: int,
    drag_gas_density_kg_m3: int,
    drag_gas_temperature_K: int,
    drag_gas_dynamic_viscosity_Pa_s: int,
    drag_gas_mean_free_path_m: int,
) -> _ThermophoresisStageBindings:
    """Bind one stage's neutral background, reusing compatible drag samples."""
    if plan is None:
        return _NO_THERMOPHORESIS_STAGE_INPUTS
    if isinstance(plan, TalbotThermophoresisPlan):
        return _talbot_stage_bindings(
            plan,
            drag_plan,
            sampled_values,
            drag_gas_density_kg_m3,
            drag_gas_temperature_K,
            drag_gas_dynamic_viscosity_Pa_s,
            drag_gas_mean_free_path_m,
        )
    if isinstance(drag_plan, EpsteinDragPlan | FiniteSpeedEpsteinDragPlan):
        gas_velocity = drag_gas_velocity_m_s
        gas_temperature = drag_gas_temperature_K
        mean_free_path = drag_gas_mean_free_path_m
    elif drag_plan is None:
        gas_velocity = _primitive_binding(sampled_values, plan.gas_velocity_field)
        gas_temperature = _primitive_binding(sampled_values, plan.gas_temperature_field)
        mean_free_path = _primitive_binding(sampled_values, plan.gas_mean_free_path_field)
    else:
        raise PhysicsEvaluationError(
            "thermophoresis neutral-gas inputs do not match the resolved drag model"
        )
    heat_flux = _primitive_binding(sampled_values, plan.gas_translational_heat_flux_field)
    return _ThermophoresisStageBindings(
        THERMOPHORESIS_WALDMANN_GALLIS,
        gas_velocity,
        gas_temperature,
        heat_flux,
        mean_free_path,
        plan.gas_molecular_mass_kg,
        plan.maximum_speed_ratio,
        -1,
        -1,
        -1,
        -1,
        1.0,
        1.0,
        1.0,
        1.0,
    )


def _talbot_stage_bindings(
    plan: TalbotThermophoresisPlan,
    drag_plan: DragPlan | None,
    sampled_values: Mapping[str, int],
    drag_gas_density_kg_m3: int,
    drag_gas_temperature_K: int,
    drag_gas_dynamic_viscosity_Pa_s: int,
    drag_gas_mean_free_path_m: int,
) -> _ThermophoresisStageBindings:
    """Bind Talbot primitives and reuse matching drag samples."""
    if isinstance(drag_plan, EpsteinDragPlan | FiniteSpeedEpsteinDragPlan):
        density = drag_gas_density_kg_m3
        temperature = drag_gas_temperature_K
        viscosity = _primitive_binding(sampled_values, plan.gas_dynamic_viscosity_field)
        mean_free_path = drag_gas_mean_free_path_m
    elif isinstance(drag_plan, StokesCunninghamDragPlan):
        density = drag_gas_density_kg_m3
        temperature = _primitive_binding(sampled_values, plan.gas_temperature_field)
        viscosity = drag_gas_dynamic_viscosity_Pa_s
        mean_free_path = drag_gas_mean_free_path_m
    else:
        density = _primitive_binding(sampled_values, plan.gas_density_field)
        temperature = _primitive_binding(sampled_values, plan.gas_temperature_field)
        viscosity = _primitive_binding(sampled_values, plan.gas_dynamic_viscosity_field)
        mean_free_path = _primitive_binding(sampled_values, plan.gas_mean_free_path_field)
    gradient = _primitive_binding(sampled_values, plan.gas_temperature_gradient_field)
    conductivity = _primitive_binding(sampled_values, plan.gas_thermal_conductivity_field)
    return _ThermophoresisStageBindings(
        THERMOPHORESIS_TALBOT,
        -1,
        temperature,
        -1,
        mean_free_path,
        1.0,
        1.0,
        gradient,
        density,
        viscosity,
        conductivity,
        plan.particle_thermal_conductivity_W_m_K,
        plan.thermal_slip_coefficient,
        plan.momentum_exchange_coefficient,
        plan.thermal_exchange_coefficient,
    )


def _lift_stage_bindings(
    plan: LiftPlan | None,
    drag_plan: DragPlan | None,
    thermophoresis_plan: ThermophoresisPlan | None,
    sampled_values: Mapping[str, int],
    drag_gas_velocity_m_s: int,
    drag_gas_density_kg_m3: int,
    drag_gas_dynamic_viscosity_Pa_s: int,
    drag_gas_mean_free_path_m: int,
    thermophoresis_inputs: _ThermophoresisStageBindings,
) -> _LiftStageBindings:
    """Bind lift primitives while reusing an already sampled neutral background."""
    if plan is None:
        return _NO_LIFT_STAGE_INPUTS
    if isinstance(plan, SaffmanLiftPlan):
        return _saffman_stage_bindings(
            plan,
            drag_plan,
            thermophoresis_plan,
            sampled_values,
            drag_gas_velocity_m_s,
            drag_gas_density_kg_m3,
            drag_gas_dynamic_viscosity_Pa_s,
            drag_gas_mean_free_path_m,
            thermophoresis_inputs,
        )
    if drag_plan is not None:
        gas_velocity = drag_gas_velocity_m_s
        gas_density = drag_gas_density_kg_m3
        mean_free_path = drag_gas_mean_free_path_m
    else:
        gas_velocity = (
            thermophoresis_inputs.gas_velocity_m_s
            if isinstance(thermophoresis_plan, WaldmannGallisThermophoresisPlan)
            else _primitive_binding(sampled_values, plan.gas_velocity_field)
        )
        mean_free_path = (
            thermophoresis_inputs.gas_mean_free_path_m
            if isinstance(thermophoresis_plan, WaldmannGallisThermophoresisPlan)
            else _primitive_binding(sampled_values, plan.gas_mean_free_path_field)
        )
        gas_density = _primitive_binding(sampled_values, plan.gas_density_field)
    vorticity = _primitive_binding(sampled_values, plan.azimuthal_gas_vorticity_field)
    return _LiftStageBindings(
        LIFT_RAREFIED_VORTICITY_RZ,
        gas_velocity,
        gas_density,
        mean_free_path,
        -1,
        vorticity,
        plan.lift_coefficient,
    )


def _saffman_stage_bindings(
    plan: SaffmanLiftPlan,
    drag_plan: DragPlan | None,
    thermophoresis_plan: ThermophoresisPlan | None,
    sampled_values: Mapping[str, int],
    drag_gas_velocity_m_s: int,
    drag_gas_density_kg_m3: int,
    drag_gas_dynamic_viscosity_Pa_s: int,
    drag_gas_mean_free_path_m: int,
    thermophoresis_inputs: _ThermophoresisStageBindings,
) -> _LiftStageBindings:
    """Bind Saffman primitives and reuse a matching continuum background."""
    if isinstance(drag_plan, StokesCunninghamDragPlan):
        gas_velocity = drag_gas_velocity_m_s
        gas_density = drag_gas_density_kg_m3
        viscosity = drag_gas_dynamic_viscosity_Pa_s
        mean_free_path = drag_gas_mean_free_path_m
    else:
        gas_velocity = _primitive_binding(sampled_values, plan.gas_velocity_field)
        if isinstance(thermophoresis_plan, TalbotThermophoresisPlan):
            gas_density = thermophoresis_inputs.gas_density_kg_m3
            viscosity = thermophoresis_inputs.gas_dynamic_viscosity_Pa_s
            mean_free_path = thermophoresis_inputs.gas_mean_free_path_m
        else:
            gas_density = _primitive_binding(sampled_values, plan.gas_density_field)
            viscosity = _primitive_binding(sampled_values, plan.gas_dynamic_viscosity_field)
            mean_free_path = _primitive_binding(sampled_values, plan.gas_mean_free_path_field)
    vorticity = _primitive_binding(sampled_values, plan.out_of_plane_gas_vorticity_field)
    code = LIFT_SAFFMAN_XY if plan.coordinate_system == "cartesian_xy" else LIFT_SAFFMAN_RZ
    return _LiftStageBindings(
        code, gas_velocity, gas_density, mean_free_path, viscosity, vorticity, 1.0
    )


def _dielectrophoresis_stage_bindings(
    plan: QuasistaticSphericalDielectrophoresisPlan | None, sampled_values: Mapping[str, int]
) -> tuple[bool, int, float, float]:
    """Bind the optional DEP field and its resolved material constants."""
    if plan is None:
        return (False, -1, 1.0, 0.0)
    gradient = _primitive_binding(sampled_values, plan.gradient_mean_e_squared_field)
    return (True, gradient, plan.medium_relative_permittivity, plan.real_clausius_mossotti_factor)


def _ion_drag_stage_bindings(
    plan: IonDragPlan | None,
    charge_plan: ChargePlan | None,
    charge_inputs: _ChargeStageBindings,
    sampled_values: Mapping[str, int],
    electric_field_V_m: int,
) -> _IonDragStageBindings:
    """Bind one stage's ion background without duplicating coupled charge samples."""
    if plan is None:
        return _NO_ION_DRAG_STAGE_INPUTS
    if not isinstance(plan, BarnesCollisionlessIonDragPlan):
        return _aggregate_ion_drag_stage_bindings(
            plan, charge_plan, charge_inputs, sampled_values, electric_field_V_m
        )
    if charge_plan is None:
        electron_density = _primitive_binding(sampled_values, plan.electron_number_density_field)
        ion_density = _primitive_binding(sampled_values, plan.positive_ion_number_density_field)
        electron_temperature = _primitive_binding(sampled_values, plan.electron_temperature_field)
        ion_temperature = _primitive_binding(sampled_values, plan.positive_ion_temperature_field)
        ion_velocity = _primitive_binding(sampled_values, plan.positive_ion_velocity_field)
    elif isinstance(charge_plan, PlasmaContinuousChargePlan):
        electron_density = charge_inputs.electron_density_m3
        ion_density = charge_inputs.positive_ion_density_m3
        electron_temperature = charge_inputs.electron_temperature_K
        ion_temperature = charge_inputs.positive_ion_temperature_K
        ion_velocity = charge_inputs.positive_ion_velocity_m_s
    else:
        raise PhysicsEvaluationError(
            "aggregate continuous charge does not share the Barnes single-ion background"
        )
    mean_free_path = _primitive_binding(sampled_values, plan.ion_neutral_mean_free_path_field)
    return _IonDragStageBindings(
        ION_DRAG_BARNES_COLLISIONLESS_EFFECTIVE_SPEED,
        electron_density,
        ion_density,
        electron_temperature,
        ion_temperature,
        ion_velocity,
        mean_free_path,
        plan.positive_ion_mass_kg,
        plan.maximum_ion_drift_ratio,
        -1,
        -1,
        -1,
        -1,
        -1,
        1.0,
    )


def _aggregate_ion_drag_stage_bindings(
    plan: RelativeFlowScreenedIonDragPlan | ElectricFieldDirectedImageIonDragPlan,
    charge_plan: ChargePlan | None,
    charge_inputs: _ChargeStageBindings,
    sampled_values: Mapping[str, int],
    electric_field_V_m: int,
) -> _IonDragStageBindings:
    if isinstance(charge_plan, AggregateRelativeDriftChargePlan):
        ion_density = charge_inputs.positive_ion_density_m3
        electron_voltage = charge_inputs.aggregate_electron_thermal_voltage_V
        ion_voltage = charge_inputs.aggregate_positive_ion_thermal_voltage_V
        ion_velocity = charge_inputs.positive_ion_velocity_m_s
        ion_mass = charge_inputs.aggregate_effective_positive_ion_mass_kg
        screening_length = charge_inputs.aggregate_screening_length_m
    elif charge_plan is None:
        ion_density = _primitive_binding(sampled_values, plan.positive_ion_number_density_field)
        electron_voltage = (
            _primitive_binding(sampled_values, plan.electron_thermal_voltage_field)
            if isinstance(plan, ElectricFieldDirectedImageIonDragPlan)
            else -1
        )
        ion_voltage = _primitive_binding(sampled_values, plan.positive_ion_thermal_voltage_field)
        ion_velocity = _primitive_binding(sampled_values, plan.positive_ion_velocity_field)
        ion_mass = _primitive_binding(sampled_values, plan.effective_positive_ion_mass_field)
        screening_length = _primitive_binding(sampled_values, plan.screening_length_field)
    else:
        raise PhysicsEvaluationError(
            "aggregate-ion drag does not share the single-species charge background"
        )
    is_relative = isinstance(plan, RelativeFlowScreenedIonDragPlan)
    mean_free_path = (
        _primitive_binding(sampled_values, plan.ion_neutral_mean_free_path_field)
        if is_relative
        else -1
    )
    return _IonDragStageBindings(
        ION_DRAG_RELATIVE_FLOW_SCREENED if is_relative else ION_DRAG_ELECTRIC_FIELD_DIRECTED_IMAGE,
        -1,
        ion_density,
        -1,
        -1,
        ion_velocity,
        mean_free_path,
        1.0,
        1.0,
        electron_voltage,
        ion_voltage,
        ion_mass,
        screening_length,
        electric_field_V_m if not is_relative else -1,
        plan.maximum_relative_ion_speed_m_s if is_relative else 1.0,
    )


def _charge_kernel_selection(plan: ChargePlan) -> tuple[int, float]:
    """Resolve one catalog-validated charge revision for the compiled pass."""

    if isinstance(plan, AggregateRelativeDriftChargePlan):
        if plan.revision == "aggregate_relative_drift_regularized_two_current_v1":
            code = CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_TWO_CURRENT
        elif plan.revision == "aggregate_relative_drift_regularized_three_current_v1":
            code = CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT
        else:
            raise PhysicsEvaluationError("aggregate charge revision was not resolved")
        return code, plan.maximum_relative_ion_speed_m_s
    if plan.revision == "oml_stationary_maxwellian_debye_huckel_v1":
        return CHARGE_OML_STATIONARY_MAXWELLIAN_DEBYE_HUCKEL, OML_MAX_ION_DRIFT_RATIO
    if plan.revision == "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1":
        if plan.maximum_ion_drift_ratio is None:
            raise PhysicsEvaluationError("shifted-Maxwellian OML drift bound was not resolved")
        return (
            CHARGE_OML_SHIFTED_MAXWELLIAN_SINGLE_ION_NEGATIVE_DEBYE_HUCKEL,
            plan.maximum_ion_drift_ratio,
        )
    raise PhysicsEvaluationError("continuous charge revision was not resolved")


def _local_certificate_offsets(bindings: PhysicsStageBindings) -> set[int]:
    """Prepare-only component roster for the closed selected model families."""

    result = _local_drag_offsets(bindings)
    charge = bindings.charge_inputs
    if charge.code != CHARGE_FIXED:
        result.add(charge.positive_ion_velocity_m_s)
    if charge.code == CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT:
        result.add(charge.aggregate_negative_ion_velocity_m_s)
    if charge.code in (
        CHARGE_OML_STATIONARY_MAXWELLIAN_DEBYE_HUCKEL,
        CHARGE_OML_SHIFTED_MAXWELLIAN_SINGLE_ION_NEGATIVE_DEBYE_HUCKEL,
    ):
        result.update(
            (
                charge.electron_density_m3,
                charge.positive_ion_density_m3,
                charge.electron_temperature_K,
                charge.positive_ion_temperature_K,
            )
        )
    result.update(_local_thermophoresis_offsets(bindings.thermophoresis_inputs))
    ion = bindings.ion_drag_inputs
    if ion.code in (ION_DRAG_BARNES_COLLISIONLESS_EFFECTIVE_SPEED, ION_DRAG_RELATIVE_FLOW_SCREENED):
        result.add(ion.positive_ion_velocity_m_s)
    if ion.code == ION_DRAG_BARNES_COLLISIONLESS_EFFECTIVE_SPEED:
        result.update(
            (
                ion.electron_density_m3,
                ion.positive_ion_density_m3,
                ion.electron_temperature_K,
                ion.positive_ion_temperature_K,
                ion.ion_neutral_mean_free_path_m,
            )
        )
    lift = bindings.lift_inputs
    if lift.code != LIFT_NONE:
        result.add(lift.gas_mean_free_path_m)
    if lift.code in (LIFT_SAFFMAN_XY, LIFT_SAFFMAN_RZ):
        result.update(
            (
                lift.gas_velocity_m_s,
                lift.gas_density_kg_m3,
                lift.gas_dynamic_viscosity_Pa_s,
                lift.azimuthal_gas_vorticity_s_inv,
            )
        )
    return result


def _local_drag_offsets(bindings: PhysicsStageBindings) -> set[int]:
    if bindings.drag_code == DRAG_NONE:
        return set()
    result = {bindings.gas_velocity, bindings.gas_mean_free_path}
    if bindings.drag_code == DRAG_STOKES_CUNNINGHAM:
        result.update((bindings.gas_density, bindings.gas_viscosity))
    else:
        result.add(bindings.gas_temperature)
    return result


def _local_thermophoresis_offsets(model: _ThermophoresisStageBindings) -> set[int]:
    if model.code == THERMOPHORESIS_NONE:
        return set()
    if model.code == THERMOPHORESIS_TALBOT:
        return {
            model.gas_temperature_K,
            model.gas_density_kg_m3,
            model.gas_dynamic_viscosity_Pa_s,
            model.gas_thermal_conductivity_W_m_K,
            model.gas_mean_free_path_m,
            model.gas_temperature_gradient_K_m,
        }
    return {model.gas_velocity_m_s, model.gas_temperature_K, model.gas_mean_free_path_m}


def _local_additive_offsets(bindings: PhysicsStageBindings) -> set[int]:
    result: set[int] = set()
    if bindings.electric_enabled:
        result.add(bindings.electric_field)
    model = bindings.ion_drag_inputs
    if model.code == ION_DRAG_RELATIVE_FLOW_SCREENED:
        result.update(
            (
                model.positive_ion_density_m3,
                model.positive_ion_thermal_voltage_V,
                model.positive_ion_velocity_m_s,
                model.effective_positive_ion_mass_kg,
                model.screening_length_m,
                model.ion_neutral_mean_free_path_m,
            )
        )
    return result


def _pack_local_ranges(
    runtime: PhysicsRuntime,
    ranges: Mapping[str, LocalPrimitiveRange],
    count: int,
    additive: bool,
) -> tuple[FloatArray, FloatArray]:
    """Pack only selected certificate primitives at the validating Python boundary."""

    bindings = runtime.native_context.bindings
    required = (
        _local_additive_offsets(bindings) if additive else _local_certificate_offsets(bindings)
    )
    lower = np.zeros((count, bindings.primitive_rules.size), dtype=np.float64)
    upper = np.zeros_like(lower)
    offset = 0
    for item in runtime.plan.required_fields:
        width = len(item.components)
        if offset in required:
            value = ranges.get(item.name)
            if value is None:
                raise PhysicsEvaluationError(f"local primitive range {item.name!r} is missing")
            lo, hi = (
                np.asarray(value.lower, dtype=np.float64),
                np.asarray(value.upper, dtype=np.float64),
            )
            if lo.shape != (count, width) or hi.shape != (count, width):
                raise PhysicsEvaluationError(
                    f"local primitive range {item.name!r} has an invalid shape"
                )
            lower[:, offset : offset + width], upper[:, offset : offset + width] = lo, hi
        offset += width
    return lower, upper


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_positive_lower(value: float) -> float:
    return np.nextafter(value / _LOCAL_BOUND_ROUNDOFF_FACTOR, 0.0)


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_nonnegative_upper(value: float) -> float:
    return np.nextafter(value * _LOCAL_BOUND_ROUNDOFF_FACTOR, np.inf)


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_positive_scale_ratios_are_finite(
    scale: float, first_ratio: float, second_ratio: float
) -> bool:
    return (
        math.isfinite(scale)
        and scale > 0.0
        and math.isfinite(first_ratio)
        and math.isfinite(second_ratio)
    )


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_scalar_range_row(
    lower: FloatArray,
    upper: FloatArray,
    row: int,
    offset: int,
    positive: bool,
) -> tuple[float, float]:
    """One schema check for the local primitive endpoints actually used."""

    lo, hi = lower[row, offset], upper[row, offset]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise PhysicsEvaluationError("local primitive range is not finite")
    if lo > hi:
        raise PhysicsEvaluationError("local primitive range is reversed")
    if positive and lo <= 0.0:
        raise PhysicsEvaluationError("local primitive range must be positive")
    return lo, hi


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_relative_speed_row(
    lower: FloatArray,
    upper: FloatArray,
    row: int,
    offset: int,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
) -> tuple[float, bool]:
    target_x_lo, target_x_hi = _local_scalar_range_row(lower, upper, row, offset, False)
    target_y_lo, target_y_hi = _local_scalar_range_row(lower, upper, row, offset + 1, False)
    component_x = np.nextafter(
        max(abs(velocity_lower[row, 0] - target_x_hi), abs(velocity_upper[row, 0] - target_x_lo)),
        np.inf,
    )
    component_y = np.nextafter(
        max(abs(velocity_lower[row, 1] - target_y_hi), abs(velocity_upper[row, 1] - target_y_lo)),
        np.inf,
    )
    speed = _local_nonnegative_upper(np.hypot(component_x, component_y))
    return speed, math.isfinite(component_x) and math.isfinite(component_y) and math.isfinite(speed)


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_drag_applicability_row(
    context: PhysicsNumericContext,
    particle: int,
    row: int,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[bool, bool]:
    model = context.bindings
    if model.drag_code == DRAG_NONE:
        return True, True
    speed, numerical = _local_relative_speed_row(
        lower, upper, row, model.gas_velocity, velocity_lower, velocity_upper
    )
    diameter = context.drag_diameter_m[particle]
    mean_free_path_lo, mean_free_path_hi = _local_scalar_range_row(
        lower, upper, row, model.gas_mean_free_path, True
    )
    if model.drag_code == DRAG_STOKES_CUNNINGHAM:
        _, density_hi = _local_scalar_range_row(lower, upper, row, model.gas_density, True)
        viscosity_lo, _ = _local_scalar_range_row(lower, upper, row, model.gas_viscosity, True)
        kn_lo = _local_positive_lower(2.0 * mean_free_path_lo / diameter)
        kn_hi = _local_nonnegative_upper(2.0 * mean_free_path_hi / diameter)
        reynolds_hi = _local_nonnegative_upper(density_hi * diameter * speed / viscosity_lo)
        numerical = (
            numerical
            and math.isfinite(kn_lo)
            and math.isfinite(kn_hi)
            and math.isfinite(reynolds_hi)
        )
        certified = (
            kn_lo >= STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS
            and kn_hi <= STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS
            and reynolds_hi <= STOKES_CUNNINGHAM_MAX_REYNOLDS
        )
        return certified and numerical, numerical
    temperature_lo, _ = _local_scalar_range_row(lower, upper, row, model.gas_temperature, True)
    thermal_factor = 8.0 / math.pi if model.drag_code == DRAG_EPSTEIN else 2.0
    thermal_lo = _local_positive_lower(
        np.sqrt(thermal_factor * BOLTZMANN_J_K * temperature_lo / model.gas_molecular_mass_kg)
    )
    kn_lo = _local_positive_lower(mean_free_path_lo / (0.5 * diameter))
    speed_ratio = _local_nonnegative_upper(speed / thermal_lo)
    numerical = numerical and _local_positive_scale_ratios_are_finite(
        thermal_lo, kn_lo, speed_ratio
    )
    certified = (
        kn_lo >= EPSTEIN_MIN_LAMBDA_OVER_RADIUS and speed_ratio <= model.epstein_maximum_speed_ratio
    )
    return certified and numerical, numerical


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_charge_applicability_row(
    context: PhysicsNumericContext,
    particle: int,
    row: int,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[bool, bool]:
    model = context.bindings.charge_inputs
    if model.code == CHARGE_FIXED:
        return True, True
    speed, numerical = _local_relative_speed_row(
        lower, upper, row, model.positive_ion_velocity_m_s, velocity_lower, velocity_upper
    )
    if (
        model.code == CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_TWO_CURRENT
        or model.code == CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT
    ):
        certified = speed <= model.aggregate_maximum_relative_ion_speed_m_s
        if model.code == CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT:
            negative_speed, negative_numerical = _local_relative_speed_row(
                lower,
                upper,
                row,
                model.aggregate_negative_ion_velocity_m_s,
                velocity_lower,
                velocity_upper,
            )
            certified = (
                certified and negative_speed <= model.aggregate_maximum_relative_ion_speed_m_s
            )
            numerical = numerical and negative_numerical
        return certified and numerical, numerical
    _, electron_density_hi = _local_scalar_range_row(
        lower, upper, row, model.electron_density_m3, True
    )
    _, ion_density_hi = _local_scalar_range_row(
        lower, upper, row, model.positive_ion_density_m3, True
    )
    electron_temperature_lo, _ = _local_scalar_range_row(
        lower, upper, row, model.electron_temperature_K, True
    )
    ion_temperature_lo, _ = _local_scalar_range_row(
        lower, upper, row, model.positive_ion_temperature_K, True
    )
    inverse_debye_square_hi = _local_nonnegative_upper(
        ELEMENTARY_CHARGE_C**2
        / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
        * (electron_density_hi / electron_temperature_lo + ion_density_hi / ion_temperature_lo)
    )
    debye_lo = _local_positive_lower(1.0 / np.sqrt(inverse_debye_square_hi))
    radius_ratio = _local_nonnegative_upper(context.electrostatic_radius_m[particle] / debye_lo)
    thermal_lo = _local_positive_lower(
        np.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_lo / (math.pi * model.positive_ion_mass_kg))
    )
    drift_ratio = _local_nonnegative_upper(speed / thermal_lo)
    numerical = (
        numerical
        and _local_positive_scale_ratios_are_finite(debye_lo, radius_ratio, drift_ratio)
        and math.isfinite(thermal_lo)
        and thermal_lo > 0.0
    )
    return (
        numerical
        and radius_ratio <= OML_MAX_RADIUS_OVER_DEBYE
        and drift_ratio <= model.maximum_ion_drift_ratio,
        numerical,
    )


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_thermophoresis_applicability_row(
    context: PhysicsNumericContext,
    particle: int,
    row: int,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[bool, bool]:
    model = context.bindings.thermophoresis_inputs
    if model.code == THERMOPHORESIS_NONE:
        return True, True
    if model.code == THERMOPHORESIS_TALBOT:
        for offset in (
            model.gas_temperature_K,
            model.gas_density_kg_m3,
            model.gas_dynamic_viscosity_Pa_s,
            model.gas_thermal_conductivity_W_m_K,
            model.gas_mean_free_path_m,
        ):
            _local_scalar_range_row(lower, upper, row, offset, True)
        _local_scalar_range_row(lower, upper, row, model.gas_temperature_gradient_K_m, False)
        _local_scalar_range_row(lower, upper, row, model.gas_temperature_gradient_K_m + 1, False)
        return True, True
    speed, numerical = _local_relative_speed_row(
        lower, upper, row, model.gas_velocity_m_s, velocity_lower, velocity_upper
    )
    temperature_lo, _ = _local_scalar_range_row(lower, upper, row, model.gas_temperature_K, True)
    mean_free_path_lo, _ = _local_scalar_range_row(
        lower, upper, row, model.gas_mean_free_path_m, True
    )
    thermal_lo = _local_positive_lower(
        np.sqrt(8.0 * BOLTZMANN_J_K * temperature_lo / (math.pi * model.gas_molecular_mass_kg))
    )
    kn_lo = _local_positive_lower(mean_free_path_lo / (0.5 * context.drag_diameter_m[particle]))
    speed_ratio = _local_nonnegative_upper(speed / thermal_lo)
    numerical = numerical and _local_positive_scale_ratios_are_finite(
        thermal_lo, kn_lo, speed_ratio
    )
    return (
        numerical
        and kn_lo >= WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS
        and speed_ratio <= model.maximum_speed_ratio,
        numerical,
    )


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_barnes_static_applicability_row(
    model: _IonDragStageBindings,
    radius: float,
    charge_lo: float,
    charge_hi: float,
    temperature_lo: float,
    temperature_hi: float,
    row: int,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[bool, bool]:
    electron_density_lo, electron_density_hi = _local_scalar_range_row(
        lower, upper, row, model.electron_density_m3, True
    )
    ion_density_lo, ion_density_hi = _local_scalar_range_row(
        lower, upper, row, model.positive_ion_density_m3, True
    )
    electron_temperature_lo, electron_temperature_hi = _local_scalar_range_row(
        lower, upper, row, model.electron_temperature_K, True
    )
    mean_free_path_lo, _ = _local_scalar_range_row(
        lower, upper, row, model.ion_neutral_mean_free_path_m, True
    )
    inverse_hi = _local_nonnegative_upper(
        ELEMENTARY_CHARGE_C**2
        / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
        * (electron_density_hi / electron_temperature_lo + ion_density_hi / temperature_lo)
    )
    inverse_lo = _local_positive_lower(
        ELEMENTARY_CHARGE_C**2
        / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
        * (electron_density_lo / electron_temperature_hi + ion_density_lo / temperature_hi)
    )
    debye_lo = _local_positive_lower(1.0 / np.sqrt(inverse_hi))
    debye_hi = _local_nonnegative_upper(1.0 / np.sqrt(inverse_lo))
    thermal_lo = _local_positive_lower(
        np.sqrt(8.0 * BOLTZMANN_J_K * temperature_lo / (math.pi * model.positive_ion_mass_kg))
    )
    charge_abs_hi = max(abs(charge_lo), abs(charge_hi))
    orbital_hi = _local_nonnegative_upper(
        charge_abs_hi
        * ELEMENTARY_CHARGE_C**2
        / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * model.positive_ion_mass_kg * thermal_lo**2)
    )
    capacitance_lo = _local_positive_lower(
        4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * radius * (1.0 + radius / debye_hi)
    )
    potential_hi = _local_nonnegative_upper(charge_abs_hi * ELEMENTARY_CHARGE_C / capacitance_lo)
    collection_square_hi = _local_nonnegative_upper(
        radius**2
        * (
            1.0
            + 2.0
            * ELEMENTARY_CHARGE_C
            * potential_hi
            / (model.positive_ion_mass_kg * thermal_lo**2)
        )
    )
    collection_hi = _local_nonnegative_upper(np.sqrt(collection_square_hi))
    coulomb_ratio_lo = _local_positive_lower(
        _local_positive_lower(debye_lo**2)
        / _local_nonnegative_upper(collection_square_hi + orbital_hi**2)
    )
    coulomb_log_lo = np.nextafter(0.5 * np.log(coulomb_ratio_lo), -np.inf)
    numerical = _barnes_local_scales_are_finite(
        debye_lo, debye_hi, thermal_lo, orbital_hi, collection_hi, coulomb_log_lo
    )
    certified = charge_hi <= 0.0
    certified = (
        certified and _local_nonnegative_upper(radius / debye_lo) <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    )
    certified = (
        certified
        and _local_nonnegative_upper(orbital_hi / debye_lo) <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    )
    certified = (
        certified
        and _local_nonnegative_upper(collection_hi / debye_lo) <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    )
    certified = (
        certified
        and _local_positive_lower(mean_free_path_lo / debye_hi)
        >= ION_DRAG_MIN_MEAN_FREE_PATH_OVER_DEBYE
        and coulomb_log_lo > 0.0
    )
    return certified and numerical, numerical


@njit(cache=True, fastmath=False, error_model="numpy")
def _barnes_local_scales_are_finite(
    debye_lo: float,
    debye_hi: float,
    thermal_lo: float,
    orbital_hi: float,
    collection_hi: float,
    coulomb_log_lo: float,
) -> bool:
    return (
        math.isfinite(debye_lo)
        and debye_lo > 0.0
        and math.isfinite(debye_hi)
        and debye_hi > 0.0
        and math.isfinite(thermal_lo)
        and thermal_lo > 0.0
        and math.isfinite(orbital_hi)
        and orbital_hi >= 0.0
        and math.isfinite(collection_hi)
        and collection_hi >= 0.0
        and math.isfinite(coulomb_log_lo)
    )


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_ion_drag_applicability_row(
    context: PhysicsNumericContext,
    particle: int,
    row: int,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    charge_lower: FloatArray,
    charge_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[bool, bool]:
    model = context.bindings.ion_drag_inputs
    if model.code == ION_DRAG_NONE or model.code == ION_DRAG_ELECTRIC_FIELD_DIRECTED_IMAGE:
        return True, True
    speed, numerical = _local_relative_speed_row(
        lower, upper, row, model.positive_ion_velocity_m_s, velocity_lower, velocity_upper
    )
    if model.code == ION_DRAG_RELATIVE_FLOW_SCREENED:
        return numerical and speed <= model.maximum_relative_ion_speed_m_s, numerical
    temperature_lo, temperature_hi = _local_scalar_range_row(
        lower, upper, row, model.positive_ion_temperature_K, True
    )
    thermal_lo = _local_positive_lower(
        np.sqrt(8.0 * BOLTZMANN_J_K * temperature_lo / (math.pi * model.positive_ion_mass_kg))
    )
    drift_ratio = _local_nonnegative_upper(speed / thermal_lo)
    numerical = (
        numerical and math.isfinite(thermal_lo) and thermal_lo > 0.0 and math.isfinite(drift_ratio)
    )
    certified, static_numerical = _local_barnes_static_applicability_row(
        model,
        context.electrostatic_radius_m[particle],
        charge_lower[row],
        charge_upper[row],
        temperature_lo,
        temperature_hi,
        row,
        lower,
        upper,
    )
    return (
        certified and numerical and drift_ratio <= model.maximum_ion_drift_ratio,
        numerical and static_numerical,
    )


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_lift_applicability_row(
    context: PhysicsNumericContext,
    particle: int,
    row: int,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[bool, bool]:
    model = context.bindings.lift_inputs
    if model.code == LIFT_NONE:
        return True, True
    radius = 0.5 * context.drag_diameter_m[particle]
    mean_free_path_lo, mean_free_path_hi = _local_scalar_range_row(
        lower, upper, row, model.gas_mean_free_path_m, True
    )
    if model.code == LIFT_RAREFIED_VORTICITY_RZ:
        kn_lo = _local_positive_lower(mean_free_path_lo / radius)
        numerical = math.isfinite(kn_lo)
        return (
            numerical and kn_lo >= RAREFIED_VORTICITY_LIFT_MIN_MEAN_FREE_PATH_OVER_RADIUS,
            numerical,
        )
    speed, numerical = _local_relative_speed_row(
        lower, upper, row, model.gas_velocity_m_s, velocity_lower, velocity_upper
    )
    density_lo, density_hi = _local_scalar_range_row(
        lower, upper, row, model.gas_density_kg_m3, True
    )
    viscosity_lo, viscosity_hi = _local_scalar_range_row(
        lower, upper, row, model.gas_dynamic_viscosity_Pa_s, True
    )
    omega_lo, omega_hi = _local_scalar_range_row(
        lower, upper, row, model.azimuthal_gas_vorticity_s_inv, False
    )
    omega_abs_hi = max(abs(omega_lo), abs(omega_hi))
    omega_abs_lo = omega_lo if omega_lo > 0.0 else (-omega_hi if omega_hi < 0.0 else 0.0)
    kn_hi = _local_nonnegative_upper(mean_free_path_hi / radius)
    slip_reynolds = _local_nonnegative_upper(density_hi * radius * speed / viscosity_lo)
    shear_hi = _local_nonnegative_upper(density_hi * radius**2 * omega_abs_hi / viscosity_lo)
    shear_lo = np.nextafter(
        density_lo * radius**2 * omega_abs_lo / viscosity_hi / _LOCAL_BOUND_ROUNDOFF_FACTOR, 0.0
    )
    numerical = (
        numerical
        and math.isfinite(kn_hi)
        and math.isfinite(slip_reynolds)
        and math.isfinite(shear_lo)
        and math.isfinite(shear_hi)
    )
    certified = (
        kn_hi <= SAFFMAN_MAX_MEAN_FREE_PATH_OVER_RADIUS
        and slip_reynolds <= SAFFMAN_MAX_SLIP_REYNOLDS
        and shear_hi <= SAFFMAN_MAX_SHEAR_REYNOLDS
    )
    certified = certified and (
        omega_abs_hi == 0.0
        or slip_reynolds <= SAFFMAN_MAX_SLIP_TO_SQRT_SHEAR_REYNOLDS * np.sqrt(shear_lo)
    )
    return certified and numerical, numerical


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_state_box_is_finite(
    row: int,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    charge_lower: FloatArray,
    charge_upper: FloatArray,
) -> bool:
    return (
        _continuous_velocity_is_finite(velocity_lower[row, 0], velocity_lower[row, 1])
        and _continuous_velocity_is_finite(velocity_upper[row, 0], velocity_upper[row, 1])
        and velocity_lower[row, 0] <= velocity_upper[row, 0]
        and velocity_lower[row, 1] <= velocity_upper[row, 1]
        and math.isfinite(charge_lower[row])
        and math.isfinite(charge_upper[row])
        and charge_lower[row] <= charge_upper[row]
    )


@njit(cache=True, fastmath=False, error_model="numpy")
def evaluate_local_continuous_applicability_into(
    context: PhysicsNumericContext,
    indices: Int64Array,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    charge_lower: FloatArray,
    charge_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    available: BoolArray,
    certified: BoolArray,
    status: UInt8Array,
) -> None:
    """Certify available signed boxes; absent field ranges never read placeholders."""

    for row in range(indices.size):
        certified[row], status[row] = False, CONTINUOUS_APPLICABILITY_OK
        if not available[row]:
            continue
        particle = indices[row]
        numerical = _local_state_box_is_finite(
            row, velocity_lower, velocity_upper, charge_lower, charge_upper
        )
        model = context.bindings.charge_inputs
        verdict = numerical
        if model.code != CHARGE_FIXED:
            verdict = (
                verdict
                and charge_lower[row] >= model.charge_number_lower
                and charge_upper[row] <= model.charge_number_upper
            )
        model_results = (
            _local_drag_applicability_row(
                context, particle, row, velocity_lower, velocity_upper, lower, upper
            ),
            _local_charge_applicability_row(
                context, particle, row, velocity_lower, velocity_upper, lower, upper
            ),
            _local_thermophoresis_applicability_row(
                context, particle, row, velocity_lower, velocity_upper, lower, upper
            ),
            _local_ion_drag_applicability_row(
                context,
                particle,
                row,
                velocity_lower,
                velocity_upper,
                charge_lower,
                charge_upper,
                lower,
                upper,
            ),
            _local_lift_applicability_row(
                context, particle, row, velocity_lower, velocity_upper, lower, upper
            ),
        )
        for model_verdict, model_numerical in model_results:
            verdict = verdict and model_verdict
            numerical = numerical and model_numerical
        if not numerical:
            status[row] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        certified[row] = verdict and numerical


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_ion_acceleration_bound_row(
    context: PhysicsNumericContext,
    particle: int,
    row: int,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    charge_lower: FloatArray,
    charge_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[float, float, bool]:
    model = context.bindings.ion_drag_inputs
    _, density_hi = _local_scalar_range_row(lower, upper, row, model.positive_ion_density_m3, True)
    voltage_lo, voltage_hi = _local_scalar_range_row(
        lower, upper, row, model.positive_ion_thermal_voltage_V, True
    )
    ion_x_lo, ion_x_hi = _local_scalar_range_row(
        lower, upper, row, model.positive_ion_velocity_m_s, False
    )
    ion_y_lo, ion_y_hi = _local_scalar_range_row(
        lower, upper, row, model.positive_ion_velocity_m_s + 1, False
    )
    ion_mass_lo, ion_mass_hi = _local_scalar_range_row(
        lower, upper, row, model.effective_positive_ion_mass_kg, True
    )
    _, screening_hi = _local_scalar_range_row(lower, upper, row, model.screening_length_m, True)
    _, mean_free_path_hi = _local_scalar_range_row(
        lower, upper, row, model.ion_neutral_mean_free_path_m, True
    )
    if not _local_state_box_is_finite(
        row, velocity_lower, velocity_upper, charge_lower, charge_upper
    ):
        raise PhysicsEvaluationError("local ion-drag state bounds must be finite and ordered")
    return relative_flow_screened_ion_drag_local_bound_row(
        context.mass_kg[particle],
        context.electrostatic_radius_m[particle],
        charge_lower[row],
        charge_upper[row],
        velocity_lower[row, 0],
        velocity_lower[row, 1],
        velocity_upper[row, 0],
        velocity_upper[row, 1],
        density_hi,
        voltage_lo,
        voltage_hi,
        ion_x_lo,
        ion_y_lo,
        ion_x_hi,
        ion_y_hi,
        ion_mass_lo,
        ion_mass_hi,
        screening_hi,
        mean_free_path_hi,
        model.maximum_relative_ion_speed_m_s,
    )


@njit(cache=True, fastmath=False, error_model="numpy")
def _local_electric_acceleration_bound_row(
    context: PhysicsNumericContext,
    particle: int,
    row: int,
    charge_lower: FloatArray,
    charge_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[float, float]:
    offset = context.bindings.electric_field
    field_x_lo, field_x_hi = _local_scalar_range_row(lower, upper, row, offset, False)
    field_y_lo, field_y_hi = _local_scalar_range_row(lower, upper, row, offset + 1, False)
    field_x_hi = _local_nonnegative_upper(max(abs(field_x_lo), abs(field_x_hi)))
    field_y_hi = _local_nonnegative_upper(max(abs(field_y_lo), abs(field_y_hi)))
    charge_hi = _local_nonnegative_upper(max(abs(charge_lower[row]), abs(charge_upper[row])))
    x_hi = _local_nonnegative_upper(
        charge_hi * ELEMENTARY_CHARGE_C * field_x_hi / context.mass_kg[particle]
    )
    y_hi = _local_nonnegative_upper(
        charge_hi * ELEMENTARY_CHARGE_C * field_y_hi / context.mass_kg[particle]
    )
    return x_hi, y_hi


@njit(cache=True, fastmath=False, error_model="numpy")
def local_additive_acceleration_abs_upper_into(
    context: PhysicsNumericContext,
    bounds: AccelerationBoundContext,
    indices: Int64Array,
    velocity_lower: FloatArray,
    velocity_upper: FloatArray,
    charge_lower: FloatArray,
    charge_upper: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    available: BoolArray,
    result: FloatArray,
    applicable: BoolArray,
    status: UInt8Array,
) -> None:
    """Enclose available local boxes using the one prepared model composition."""

    if not bounds.local_available:
        raise PhysicsEvaluationError(
            "selected charge-dependent forces have no local acceleration certificate"
        )
    for row in range(indices.size):
        result[row, 0], result[row, 1] = 0.0, 0.0
        applicable[row], status[row] = False, CONTINUOUS_APPLICABILITY_OK
        if not available[row]:
            continue
        particle = indices[row]
        numerical = _local_state_box_is_finite(
            row, velocity_lower, velocity_upper, charge_lower, charge_upper
        )
        verdict = numerical
        x_hi, y_hi = (
            bounds.local_base_abs_upper_m_s2[particle, 0],
            bounds.local_base_abs_upper_m_s2[particle, 1],
        )
        if context.bindings.electric_enabled:
            electric_x, electric_y = _local_electric_acceleration_bound_row(
                context, particle, row, charge_lower, charge_upper, lower, upper
            )
            x_hi, y_hi = (
                np.nextafter(x_hi + electric_x, np.inf),
                np.nextafter(y_hi + electric_y, np.inf),
            )
        if context.bindings.ion_drag_inputs.code == ION_DRAG_RELATIVE_FLOW_SCREENED:
            ion_x, ion_y, ion_applicable = _local_ion_acceleration_bound_row(
                context,
                particle,
                row,
                velocity_lower,
                velocity_upper,
                charge_lower,
                charge_upper,
                lower,
                upper,
            )
            verdict = verdict and ion_applicable
            x_hi, y_hi = np.nextafter(x_hi + ion_x, np.inf), np.nextafter(y_hi + ion_y, np.inf)
        if bounds.lift_rate_upper_s_inv.size:
            lift_x, lift_y, lift_status = velocity_acceleration_abs_upper_row(
                bounds.lift_rate_upper_s_inv[particle],
                bounds.lift_target_abs_upper_m_s[0],
                bounds.lift_target_abs_upper_m_s[1],
                max(abs(velocity_lower[row, 0]), abs(velocity_upper[row, 0])),
                max(abs(velocity_lower[row, 1]), abs(velocity_upper[row, 1])),
                True,
                NUMERICAL_STATUS_OK,
            )
            numerical = numerical and lift_status == NUMERICAL_STATUS_OK
            x_hi, y_hi = np.nextafter(x_hi + lift_x, np.inf), np.nextafter(y_hi + lift_y, np.inf)
        numerical = (
            numerical
            and math.isfinite(x_hi)
            and math.isfinite(y_hi)
            and x_hi >= 0.0
            and y_hi >= 0.0
        )
        if not numerical:
            status[row] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
            x_hi, y_hi = 0.0, 0.0
        result[row, 0], result[row, 1], applicable[row] = x_hi, y_hi, verdict and numerical


def prepare_physics_runtime(
    *,
    plan: PhysicsPlan,
    coordinate_system: CoordinateSystem,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    electrostatic_radius_m: FloatArray,
    displaced_volume_m3: FloatArray,
    charge_number: FloatArray,
    primitive_ranges: Mapping[str, PrimitiveRange],
) -> PhysicsRuntime:
    """Prepare model bounds and exact certificates from primitive extrema."""

    count = _particle_count(
        mass_kg,
        drag_diameter_m,
        electrostatic_radius_m,
        displaced_volume_m3,
        charge_number,
    )
    if bool((mass_kg <= 0.0).any() or (drag_diameter_m <= 0.0).any()):
        raise PhysicsEvaluationError("particle mass and drag diameter must be positive")
    if bool((electrostatic_radius_m < 0.0).any() or (displaced_volume_m3 < 0.0).any()):
        raise PhysicsEvaluationError(
            "particle electrostatic radius and displaced volume must be nonnegative"
        )

    charge_bounds = _prepare_charge_bounds(
        plan,
        charge_number,
        electrostatic_radius_m,
        primitive_ranges,
    )
    drag_bounds = _prepare_drag_bounds(
        plan,
        mass_kg,
        drag_diameter_m,
        primitive_ranges,
    )
    thermophoresis_bounds, thermophoresis_acceleration_bound = _prepare_thermophoresis_bounds(
        plan,
        mass_kg,
        drag_diameter_m,
        primitive_ranges,
    )
    ion_drag_bounds, ion_drag_acceleration_bound = _prepare_ion_drag_bounds(
        plan,
        mass_kg,
        electrostatic_radius_m,
        charge_number,
        charge_bounds,
        primitive_ranges,
    )
    dielectrophoresis_acceleration_bound = _prepare_dielectrophoresis_acceleration_bound(
        plan.dielectrophoresis,
        mass_kg,
        electrostatic_radius_m,
        primitive_ranges,
    )
    lift_bounds = _prepare_lift_bounds(
        plan.lift,
        mass_kg,
        drag_diameter_m,
        primitive_ranges,
    )
    external = _prepare_external_acceleration_bound(
        plan,
        mass_kg,
        displaced_volume_m3,
        charge_number,
        charge_bounds,
        ion_drag_acceleration_bound,
        thermophoresis_acceleration_bound,
        dielectrophoresis_acceleration_bound,
        primitive_ranges,
    )
    localizable_external_base = _prepare_localizable_external_base(
        plan,
        mass_kg,
        displaced_volume_m3,
        thermophoresis_acceleration_bound,
        dielectrophoresis_acceleration_bound,
        primitive_ranges,
    )
    constant = _constant_acceleration(
        plan,
        coordinate_system,
        count,
        mass_kg,
        electrostatic_radius_m,
        displaced_volume_m3,
        charge_number,
        primitive_ranges,
    )
    external.setflags(write=False)
    if localizable_external_base is not None:
        localizable_external_base.setflags(write=False)
    if constant is not None:
        constant.setflags(write=False)
    native_context = PhysicsNumericContext(
        mass_kg,
        drag_diameter_m,
        electrostatic_radius_m,
        displaced_volume_m3,
        _prepare_physics_stage_bindings(plan, charge_bounds),
    )
    native_applicability_context = _prepare_continuous_applicability_context(
        native_context,
        plan,
        drag_bounds,
        charge_bounds,
        thermophoresis_bounds,
        ion_drag_bounds,
        lift_bounds,
    )
    native_acceleration_bound_context = AccelerationBoundContext(
        external,
        _EMPTY_VECTOR if localizable_external_base is None else localizable_external_base,
        localizable_external_base is not None,
        _EMPTY_BOUND_SCALAR if drag_bounds is None else drag_bounds.rate_upper_s_inv,
        _EMPTY_BOUND_SCALAR if drag_bounds is None else drag_bounds.target_velocity_abs_upper_m_s,
        _EMPTY_BOUND_SCALAR if lift_bounds is None else lift_bounds.coupling_rate_abs_upper_s_inv,
        _EMPTY_BOUND_SCALAR if lift_bounds is None else lift_bounds.gas_velocity_abs_upper_m_s,
    )
    return PhysicsRuntime(
        plan,
        mass_kg,
        drag_diameter_m,
        electrostatic_radius_m,
        displaced_volume_m3,
        charge_bounds,
        drag_bounds,
        thermophoresis_bounds,
        ion_drag_bounds,
        lift_bounds,
        external,
        localizable_external_base,
        constant,
        native_context,
        native_applicability_context,
        native_acceleration_bound_context,
    )


def _prepare_continuous_applicability_context(
    native: PhysicsNumericContext,
    plan: PhysicsPlan,
    drag: _DragBounds | None,
    charge: _ChargeRuntimeBounds | None,
    thermophoresis: _ThermophoresisRuntimeBounds | None,
    ion_drag: _IonDragRuntimeBounds | None,
    lift: _LiftRuntimeBounds | None,
) -> ContinuousApplicabilityContext:
    """Resolve all family gates once, retaining each prepared bound's owner."""

    bindings = native.bindings
    return ContinuousApplicabilityContext(
        native.drag_diameter_m,
        native.electrostatic_radius_m,
        _prepare_drag_applicability_context(plan.drag, drag, bindings.drag_code),
        _prepare_charge_applicability_context(plan.charge, charge, bindings.charge_inputs.code),
        _prepare_thermophoresis_applicability_context(
            plan.thermophoresis,
            thermophoresis,
            bindings.thermophoresis_inputs.code,
        ),
        _prepare_ion_drag_applicability_context(
            plan.ion_drag,
            ion_drag,
            bindings.ion_drag_inputs.code,
        ),
        _prepare_lift_applicability_context(plan.lift, lift, bindings.lift_inputs.code),
        bindings.drag_code != DRAG_NONE
        or bindings.ion_drag_inputs.code
        in (
            ION_DRAG_BARNES_COLLISIONLESS_EFFECTIVE_SPEED,
            ION_DRAG_RELATIVE_FLOW_SCREENED,
        )
        or bindings.lift_inputs.code in (LIFT_SAFFMAN_XY, LIFT_SAFFMAN_RZ),
    )


def _prepare_drag_applicability_context(
    plan: DragPlan | None,
    bounds: _DragBounds | None,
    code: int,
) -> _DragApplicabilityContext:
    if plan is None and bounds is None and code == DRAG_NONE:
        return _DragApplicabilityContext(code, _EMPTY_BOUND_SCALAR, 0.0, 0.0, 0.0, 0.0, 0.0)
    if isinstance(plan, EpsteinDragPlan | FiniteSpeedEpsteinDragPlan) and isinstance(
        bounds, _EpsteinBounds
    ):
        factor = 8.0 / math.pi if isinstance(plan, EpsteinDragPlan) else 2.0
        limit = _epstein_continuous_speed_limit(
            factor,
            bounds.temperature_lower_K,
            plan.gas_molecular_mass_kg,
            plan.maximum_speed_ratio,
        )
        return _DragApplicabilityContext(
            code,
            bounds.target_velocity_abs_upper_m_s,
            limit,
            bounds.mean_free_path_lower_m,
            bounds.mean_free_path_lower_m,
            0.0,
            0.0,
        )
    if isinstance(plan, StokesCunninghamDragPlan) and isinstance(bounds, _StokesCunninghamBounds):
        return _DragApplicabilityContext(
            code,
            bounds.target_velocity_abs_upper_m_s,
            0.0,
            bounds.mean_free_path_lower_m,
            bounds.mean_free_path_upper_m,
            bounds.density_upper_kg_m3,
            bounds.dynamic_viscosity_lower_Pa_s,
        )
    raise PhysicsEvaluationError("drag applicability bounds do not match the resolved model")


def _prepare_charge_applicability_context(
    plan: ChargePlan | None,
    bounds: _ChargeRuntimeBounds | None,
    code: int,
) -> _ChargeApplicabilityContext:
    if plan is None and bounds is None and code == CHARGE_FIXED:
        return _ChargeApplicabilityContext(
            code,
            _EMPTY_BOUND_SCALAR,
            _EMPTY_BOUND_SCALAR,
            0.0,
            0.0,
            0.0,
            -math.inf,
            math.inf,
        )
    if isinstance(plan, AggregateRelativeDriftChargePlan) and isinstance(
        bounds, _AggregateChargeRuntimeBounds
    ):
        negative_flow = (
            bounds.negative_ion_velocity_abs_upper_m_s
            if plan.has_negative_ion_current
            else _EMPTY_BOUND_SCALAR
        )
        return _ChargeApplicabilityContext(
            code,
            bounds.positive_ion_velocity_abs_upper_m_s,
            negative_flow,
            0.0,
            0.0,
            plan.maximum_relative_ion_speed_m_s,
            bounds.model.charge_number_lower,
            bounds.model.charge_number_upper,
        )
    if isinstance(plan, PlasmaContinuousChargePlan) and isinstance(bounds, _OmlRuntimeBounds):
        _, drift_limit = _charge_kernel_selection(plan)
        return _ChargeApplicabilityContext(
            code,
            bounds.positive_ion_velocity_abs_upper_m_s,
            _EMPTY_BOUND_SCALAR,
            _continuous_mean_thermal_speed_lower(
                bounds.positive_ion_temperature_lower_K,
                plan.positive_ion_mass_kg,
            ),
            bounds.model.debye_length_lower_m,
            drift_limit,
            bounds.model.charge_number_lower,
            bounds.model.charge_number_upper,
        )
    raise PhysicsEvaluationError("continuous charge bounds do not match the resolved revision")


def _prepare_thermophoresis_applicability_context(
    plan: ThermophoresisPlan | None,
    bounds: _ThermophoresisRuntimeBounds | None,
    code: int,
) -> _ThermophoresisApplicabilityContext:
    if plan is None and bounds is None and code == THERMOPHORESIS_NONE:
        return _ThermophoresisApplicabilityContext(
            code,
            _EMPTY_BOUND_APPLICABLE,
            _EMPTY_BOUND_SCALAR,
            0.0,
            0.0,
        )
    if isinstance(plan, TalbotThermophoresisPlan) and isinstance(bounds, _TalbotBounds):
        return _ThermophoresisApplicabilityContext(
            code,
            bounds.static_applicable,
            _EMPTY_BOUND_SCALAR,
            0.0,
            0.0,
        )
    if isinstance(plan, WaldmannGallisThermophoresisPlan) and isinstance(
        bounds, _WaldmannGallisBounds
    ):
        return _ThermophoresisApplicabilityContext(
            code,
            bounds.static_applicable,
            bounds.gas_velocity_abs_upper_m_s,
            _continuous_mean_thermal_speed_lower(
                bounds.gas_temperature_lower_K, plan.gas_molecular_mass_kg
            ),
            plan.maximum_speed_ratio,
        )
    raise PhysicsEvaluationError(
        "thermophoresis applicability bounds do not match the resolved revision"
    )


def _prepare_ion_drag_applicability_context(
    plan: IonDragPlan | None,
    bounds: _IonDragRuntimeBounds | None,
    code: int,
) -> _IonDragApplicabilityContext:
    if (plan is None or isinstance(plan, ElectricFieldDirectedImageIonDragPlan)) and bounds is None:
        return _IonDragApplicabilityContext(
            code,
            _EMPTY_BOUND_APPLICABLE,
            _EMPTY_BOUND_SCALAR,
            0.0,
            0.0,
        )
    if isinstance(plan, BarnesCollisionlessIonDragPlan) and isinstance(
        bounds, _BarnesIonDragBounds
    ):
        return _IonDragApplicabilityContext(
            code,
            bounds.static_applicable,
            bounds.positive_ion_velocity_abs_upper_m_s,
            _continuous_mean_thermal_speed_lower(
                bounds.positive_ion_temperature_lower_K,
                plan.positive_ion_mass_kg,
            ),
            plan.maximum_ion_drift_ratio,
        )
    if isinstance(plan, RelativeFlowScreenedIonDragPlan) and isinstance(
        bounds, _RelativeFlowScreenedIonDragBounds
    ):
        return _IonDragApplicabilityContext(
            code,
            _EMPTY_BOUND_APPLICABLE,
            bounds.positive_ion_velocity_abs_upper_m_s,
            0.0,
            plan.maximum_relative_ion_speed_m_s,
        )
    raise PhysicsEvaluationError("ion-drag applicability bounds do not match the revision")


def _prepare_lift_applicability_context(
    plan: LiftPlan | None,
    bounds: _LiftRuntimeBounds | None,
    code: int,
) -> _LiftApplicabilityContext:
    if plan is None and bounds is None and code == LIFT_NONE:
        return _LiftApplicabilityContext(
            code,
            _EMPTY_BOUND_APPLICABLE,
            _EMPTY_BOUND_SCALAR,
            0.0,
            0.0,
            _EMPTY_BOUND_SCALAR,
            _EMPTY_BOUND_SCALAR,
        )
    if isinstance(plan, SaffmanLiftPlan) and isinstance(bounds, _SaffmanLiftBounds):
        return _LiftApplicabilityContext(
            code,
            bounds.static_applicable,
            bounds.gas_velocity_abs_upper_m_s,
            bounds.gas_density_upper_kg_m3,
            bounds.gas_dynamic_viscosity_lower_Pa_s,
            bounds.shear_reynolds_lower,
            bounds.shear_reynolds_upper,
        )
    if isinstance(plan, RarefiedVorticityLiftPlan) and isinstance(
        bounds, _RarefiedVorticityLiftBounds
    ):
        return _LiftApplicabilityContext(
            code,
            bounds.static_applicable,
            bounds.gas_velocity_abs_upper_m_s,
            0.0,
            0.0,
            _EMPTY_BOUND_SCALAR,
            _EMPTY_BOUND_SCALAR,
        )
    raise PhysicsEvaluationError("lift applicability bounds do not match the revision")


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _drag_continuous_applicability_row(
    bounds: _DragApplicabilityContext,
    diameter: float,
    vx: float,
    vy: float,
) -> tuple[bool, np.uint8]:
    if bounds.code == DRAG_NONE:
        return True, CONTINUOUS_APPLICABILITY_OK
    ux, uy = bounds.flow_abs_upper_m_s[0], bounds.flow_abs_upper_m_s[1]
    if bounds.code == DRAG_STOKES_CUNNINGHAM:
        return _stokes_continuous_applicability_row(
            diameter,
            vx,
            vy,
            ux,
            uy,
            bounds.density_upper_kg_m3,
            bounds.viscosity_lower_Pa_s,
            bounds.mean_free_path_lower_m,
            bounds.mean_free_path_upper_m,
        )
    return _epstein_continuous_applicability_row(
        diameter,
        vx,
        vy,
        ux,
        uy,
        bounds.mean_free_path_lower_m,
        bounds.speed_limit_m_s,
    )


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _aggregate_charge_continuous_applicability_row(
    vx: float,
    vy: float,
    ux: float,
    uy: float,
    speed_limit: float,
) -> tuple[bool, np.uint8]:
    applicable, status = _relative_speed_continuous_applicability_row(vx, vy, ux, uy, speed_limit)
    valid = (
        vx >= 0.0
        and vy >= 0.0
        and math.isfinite(ux)
        and math.isfinite(uy)
        and ux >= 0.0
        and uy >= 0.0
    )
    if not valid:
        return False, CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    return applicable, status


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _charge_continuous_applicability_row(
    bounds: _ChargeApplicabilityContext,
    radius: float,
    vx: float,
    vy: float,
) -> tuple[bool, np.uint8]:
    if bounds.code == CHARGE_FIXED:
        return True, CONTINUOUS_APPLICABILITY_OK
    ux, uy = bounds.positive_flow_abs_upper_m_s[0], bounds.positive_flow_abs_upper_m_s[1]
    if bounds.code in (
        CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_TWO_CURRENT,
        CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT,
    ):
        applicable, status = _aggregate_charge_continuous_applicability_row(
            vx,
            vy,
            ux,
            uy,
            bounds.maximum_drift_or_speed,
        )
        if bounds.code == CHARGE_AGGREGATE_RELATIVE_DRIFT_REGULARIZED_THREE_CURRENT:
            flow = bounds.negative_flow_abs_upper_m_s
            negative_applicable, negative_status = _aggregate_charge_continuous_applicability_row(
                vx,
                vy,
                flow[0],
                flow[1],
                bounds.maximum_drift_or_speed,
            )
            applicable = applicable and negative_applicable
            if negative_status != CONTINUOUS_APPLICABILITY_OK:
                status = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        return applicable, status
    radius_ratio = np.nextafter(radius / bounds.debye_length_lower_m, np.inf)
    speed = _outward_relative_speed(vx, vy, ux, uy)
    drift_ratio = np.nextafter(speed / bounds.thermal_speed_lower_m_s, np.inf)
    numerical_ok = (
        math.isfinite(radius)
        and radius > 0.0
        and _continuous_velocity_is_finite(vx, vy)
        and vx >= 0.0
        and vy >= 0.0
        and math.isfinite(radius_ratio)
        and math.isfinite(drift_ratio)
    )
    status = (
        CONTINUOUS_APPLICABILITY_OK if numerical_ok else CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    )
    return (
        numerical_ok
        and radius_ratio <= OML_MAX_RADIUS_OVER_DEBYE
        and drift_ratio <= bounds.maximum_drift_or_speed
    ), status


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _thermophoresis_continuous_applicability_row(
    bounds: _ThermophoresisApplicabilityContext,
    particle: int,
    vx: float,
    vy: float,
) -> tuple[bool, np.uint8]:
    if bounds.code == THERMOPHORESIS_NONE:
        return True, CONTINUOUS_APPLICABILITY_OK
    if bounds.code == THERMOPHORESIS_TALBOT:
        return bounds.static_applicable[particle], CONTINUOUS_APPLICABILITY_OK
    flow = bounds.flow_abs_upper_m_s
    return _drift_continuous_applicability_row(
        bounds.static_applicable[particle],
        vx,
        vy,
        flow[0],
        flow[1],
        bounds.thermal_speed_lower_m_s,
        bounds.maximum_speed_ratio,
    )


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _ion_drag_continuous_applicability_row(
    bounds: _IonDragApplicabilityContext,
    particle: int,
    vx: float,
    vy: float,
) -> tuple[bool, np.uint8]:
    if bounds.code in (ION_DRAG_NONE, ION_DRAG_ELECTRIC_FIELD_DIRECTED_IMAGE):
        return True, CONTINUOUS_APPLICABILITY_OK
    flow = bounds.flow_abs_upper_m_s
    if bounds.code == ION_DRAG_RELATIVE_FLOW_SCREENED:
        return _relative_speed_continuous_applicability_row(
            vx,
            vy,
            flow[0],
            flow[1],
            bounds.maximum_drift_or_speed,
        )
    return _drift_continuous_applicability_row(
        bounds.static_applicable[particle],
        vx,
        vy,
        flow[0],
        flow[1],
        bounds.thermal_speed_lower_m_s,
        bounds.maximum_drift_or_speed,
    )


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _lift_continuous_applicability_row(
    bounds: _LiftApplicabilityContext,
    particle: int,
    diameter: float,
    vx: float,
    vy: float,
) -> tuple[bool, np.uint8]:
    if bounds.code == LIFT_NONE:
        return True, CONTINUOUS_APPLICABILITY_OK
    if bounds.code == LIFT_RAREFIED_VORTICITY_RZ:
        valid = _continuous_velocity_is_finite(vx, vy) and vx >= 0.0 and vy >= 0.0
        status = (
            CONTINUOUS_APPLICABILITY_OK if valid else CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        )
        return bounds.static_applicable[particle] and valid, status
    flow = bounds.flow_abs_upper_m_s
    return _saffman_continuous_applicability_row(
        bounds.static_applicable[particle],
        diameter,
        vx,
        vy,
        flow[0],
        flow[1],
        bounds.density_upper_kg_m3,
        bounds.viscosity_lower_Pa_s,
        bounds.shear_reynolds_lower[particle],
        bounds.shear_reynolds_upper[particle],
    )


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def evaluate_continuous_applicability_into(
    context: ContinuousApplicabilityContext,
    particle_index: Int64Array,
    velocity_abs_upper_m_s: FloatArray,
    applicable: BoolArray,
    status: UInt8Array,
) -> None:
    """Write one merged family verdict directly, without gathers or allocations."""

    count = particle_index.size
    if context.reject_negative_velocity and _has_finite_negative_velocity(velocity_abs_upper_m_s):
        raise PhysicsEvaluationError("velocity_abs_upper_m_s must be nonnegative")
    for row in range(count):
        particle = particle_index[row]
        vx, vy = velocity_abs_upper_m_s[row, 0], velocity_abs_upper_m_s[row, 1]
        diameter = context.drag_diameter_m[particle]
        drag_app, row_status = _drag_continuous_applicability_row(context.drag, diameter, vx, vy)
        charge_app, charge_status = _charge_continuous_applicability_row(
            context.charge,
            context.electrostatic_radius_m[particle],
            vx,
            vy,
        )
        thermal_app, thermal_status = _thermophoresis_continuous_applicability_row(
            context.thermophoresis,
            particle,
            vx,
            vy,
        )
        ion_app, ion_status = _ion_drag_continuous_applicability_row(
            context.ion_drag, particle, vx, vy
        )
        lift_app, lift_status = _lift_continuous_applicability_row(
            context.lift, particle, diameter, vx, vy
        )
        applicable[row] = drag_app and charge_app and thermal_app and ion_app and lift_app
        if row_status == CONTINUOUS_APPLICABILITY_OK:
            row_status = charge_status
        if row_status == CONTINUOUS_APPLICABILITY_OK:
            row_status = thermal_status
        if row_status == CONTINUOUS_APPLICABILITY_OK:
            row_status = ion_status
        if row_status == CONTINUOUS_APPLICABILITY_OK:
            row_status = lift_status
        status[row] = row_status


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def charge_interval_inside_prepared_invariant_into(
    context: ContinuousApplicabilityContext,
    charge_lower_number: FloatArray,
    charge_upper_number: FloatArray,
    applicable: BoolArray,
    status: UInt8Array,
) -> None:
    """Merge ordered finite charge intervals and retain any preceding failure."""

    bounds = context.charge
    for row in range(charge_lower_number.size):
        lower, upper = charge_lower_number[row], charge_upper_number[row]
        finite_ordered = math.isfinite(lower) and math.isfinite(upper) and lower <= upper
        if not finite_ordered and status[row] == CONTINUOUS_APPLICABILITY_OK:
            status[row] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        inside = finite_ordered
        if bounds.code != CHARGE_FIXED:
            inside = (
                inside
                and lower >= bounds.charge_number_lower
                and upper <= bounds.charge_number_upper
            )
        applicable[row] = applicable[row] and inside and status[row] == CONTINUOUS_APPLICABILITY_OK


def _prepare_charge_bounds(
    plan: PhysicsPlan,
    charge_number: FloatArray,
    electrostatic_radius_m: FloatArray,
    ranges: Mapping[str, PrimitiveRange],
) -> _ChargeRuntimeBounds | None:
    charge = plan.charge
    if charge is None:
        return None
    electron_density = _positive_scalar_range(ranges, charge.electron_number_density_field)
    ion_density = _positive_scalar_range(ranges, charge.positive_ion_number_density_field)
    ion_velocity = _primitive_range(ranges, charge.positive_ion_velocity_field, 2)
    ion_velocity_abs_upper = np.nextafter(
        np.maximum(np.abs(ion_velocity.lower), np.abs(ion_velocity.upper)),
        np.inf,
    )
    _freeze(ion_velocity_abs_upper)
    common_arguments = {
        "initial_charge_number": charge_number,
        "electrostatic_radius_m": electrostatic_radius_m,
        "electron_number_density_lower_m3": float(electron_density.lower[0]),
        "electron_number_density_upper_m3": float(electron_density.upper[0]),
        "positive_ion_number_density_lower_m3": float(ion_density.lower[0]),
        "positive_ion_number_density_upper_m3": float(ion_density.upper[0]),
    }
    if isinstance(charge, AggregateRelativeDriftChargePlan):
        electron_voltage = _positive_scalar_range(ranges, charge.electron_thermal_voltage_field)
        ion_voltage = _positive_scalar_range(ranges, charge.positive_ion_thermal_voltage_field)
        ion_mass = _positive_scalar_range(ranges, charge.effective_positive_ion_mass_field)
        screening_length = _positive_scalar_range(ranges, charge.screening_length_field)
        aggregate_arguments = {
            **common_arguments,
            "electron_thermal_voltage_lower_V": float(electron_voltage.lower[0]),
            "electron_thermal_voltage_upper_V": float(electron_voltage.upper[0]),
            "positive_ion_thermal_voltage_lower_V": float(ion_voltage.lower[0]),
            "positive_ion_thermal_voltage_upper_V": float(ion_voltage.upper[0]),
            "effective_positive_ion_mass_lower_kg": float(ion_mass.lower[0]),
            "effective_positive_ion_mass_upper_kg": float(ion_mass.upper[0]),
            "screening_length_lower_m": float(screening_length.lower[0]),
            "screening_length_upper_m": float(screening_length.upper[0]),
            "maximum_relative_ion_speed_m_s": charge.maximum_relative_ion_speed_m_s,
        }
        if not charge.has_negative_ion_current:
            bounds = aggregate_relative_drift_global_bounds(**aggregate_arguments)
            return _AggregateChargeRuntimeBounds(
                bounds,
                ion_velocity_abs_upper,
                _EMPTY_VECTOR,
            )
        negative_density = _nonnegative_scalar_range(
            ranges, cast(str, charge.negative_ion_number_density_field)
        )
        negative_voltage = _positive_scalar_range(
            ranges, cast(str, charge.negative_ion_thermal_voltage_field)
        )
        negative_mass = _positive_scalar_range(
            ranges, cast(str, charge.effective_negative_ion_mass_field)
        )
        negative_velocity = _primitive_range(
            ranges, cast(str, charge.negative_ion_velocity_field), 2
        )
        negative_velocity_abs_upper = np.nextafter(
            np.maximum(np.abs(negative_velocity.lower), np.abs(negative_velocity.upper)),
            np.inf,
        )
        _freeze(negative_velocity_abs_upper)
        bounds = aggregate_relative_drift_three_current_global_bounds(
            **aggregate_arguments,
            negative_ion_number_density_lower_m3=float(negative_density.lower[0]),
            negative_ion_number_density_upper_m3=float(negative_density.upper[0]),
            negative_ion_thermal_voltage_lower_V=float(negative_voltage.lower[0]),
            negative_ion_thermal_voltage_upper_V=float(negative_voltage.upper[0]),
            effective_negative_ion_mass_lower_kg=float(negative_mass.lower[0]),
            effective_negative_ion_mass_upper_kg=float(negative_mass.upper[0]),
        )
        return _AggregateChargeRuntimeBounds(
            bounds,
            ion_velocity_abs_upper,
            negative_velocity_abs_upper,
        )

    electron_temperature = _positive_scalar_range(ranges, charge.electron_temperature_field)
    ion_temperature = _positive_scalar_range(ranges, charge.positive_ion_temperature_field)
    bound_arguments = {
        **common_arguments,
        "electron_temperature_lower_K": float(electron_temperature.lower[0]),
        "electron_temperature_upper_K": float(electron_temperature.upper[0]),
        "positive_ion_temperature_lower_K": float(ion_temperature.lower[0]),
        "positive_ion_temperature_upper_K": float(ion_temperature.upper[0]),
        "positive_ion_mass_kg": charge.positive_ion_mass_kg,
    }
    if charge.revision == "oml_stationary_maxwellian_debye_huckel_v1":
        bounds = oml_stationary_global_bounds(**bound_arguments)
    elif charge.revision == "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1":
        if charge.maximum_ion_drift_ratio is None:
            raise PhysicsEvaluationError("shifted-Maxwellian OML drift bound was not resolved")
        bounds = oml_shifted_maxwellian_global_bounds(
            **bound_arguments,
            maximum_ion_drift_ratio=charge.maximum_ion_drift_ratio,
        )
    else:
        raise PhysicsEvaluationError("continuous charge revision was not resolved")
    return _OmlRuntimeBounds(
        bounds,
        ion_velocity_abs_upper,
        float(ion_temperature.lower[0]),
    )


def _prepare_drag_bounds(
    plan: PhysicsPlan,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    ranges: Mapping[str, PrimitiveRange],
) -> _DragBounds | None:
    drag = plan.drag
    if drag is None:
        return None
    velocity = _primitive_range(ranges, drag.gas_velocity_field, 2)
    target_abs = np.nextafter(np.maximum(np.abs(velocity.lower), np.abs(velocity.upper)), np.inf)

    if isinstance(drag, EpsteinDragPlan):
        density = _positive_scalar_range(ranges, drag.gas_density_field)
        temperature = _positive_scalar_range(ranges, drag.gas_temperature_field)
        mean_free_path = _positive_scalar_range(ranges, drag.gas_mean_free_path_field)
        count = int(mass_kg.size)
        rate, _, _ = epstein_rate_s_inv(
            mass_kg=mass_kg,
            drag_diameter_m=drag_diameter_m,
            gas_density_kg_m3=np.full(count, float(density.upper[0]), dtype=np.float64),
            gas_temperature_K=np.full(count, float(temperature.upper[0]), dtype=np.float64),
            gas_molecular_mass_kg=drag.gas_molecular_mass_kg,
            delta=drag.delta,
        )
        rate = np.nextafter(rate, np.inf)
        _freeze(rate, target_abs)
        return _EpsteinBounds(
            rate,
            target_abs,
            float(temperature.lower[0]),
            float(mean_free_path.lower[0]),
            float(np.max(rate)),
        )

    if isinstance(drag, FiniteSpeedEpsteinDragPlan):
        density = _positive_scalar_range(ranges, drag.gas_density_field)
        temperature = _positive_scalar_range(ranges, drag.gas_temperature_field)
        mean_free_path = _positive_scalar_range(ranges, drag.gas_mean_free_path_field)
        count = int(mass_kg.size)
        base_rate, _, _ = epstein_rate_s_inv(
            mass_kg=mass_kg,
            drag_diameter_m=drag_diameter_m,
            gas_density_kg_m3=np.full(count, float(density.upper[0]), dtype=np.float64),
            gas_temperature_K=np.full(count, float(temperature.upper[0]), dtype=np.float64),
            gas_molecular_mass_kg=drag.gas_molecular_mass_kg,
            delta=1.0,
        )
        specular_rate, specular_lipschitz = epstein_finite_speed_factors(
            np.asarray([drag.maximum_speed_ratio], dtype=np.float64)
        )
        diffuse_factor = drag.diffuse_reflection_fraction * math.pi / 8.0
        with np.errstate(over="ignore", invalid="ignore"):
            rate = np.nextafter(
                base_rate * np.nextafter(specular_rate[0] + diffuse_factor, np.inf),
                np.inf,
            )
            lipschitz = np.nextafter(
                float(np.max(base_rate))
                * np.nextafter(specular_lipschitz[0] + diffuse_factor, np.inf),
                np.inf,
            )
        if not bool(np.isfinite(rate).all()) or not math.isfinite(lipschitz):
            raise PhysicsEvaluationError("finite-speed Epstein global bound is not finite")
        _freeze(rate, target_abs)
        return _EpsteinBounds(
            rate,
            target_abs,
            float(temperature.lower[0]),
            float(mean_free_path.lower[0]),
            lipschitz,
        )

    viscosity = _positive_scalar_range(ranges, drag.gas_dynamic_viscosity_field)
    mean_free_path = _positive_scalar_range(ranges, drag.gas_mean_free_path_field)
    density = _positive_scalar_range(ranges, drag.gas_density_field)
    rate = stokes_cunningham_rate_upper_s_inv(
        mass_kg=mass_kg,
        drag_diameter_m=drag_diameter_m,
        gas_dynamic_viscosity_upper_Pa_s=float(viscosity.upper[0]),
        gas_mean_free_path_lower_m=float(mean_free_path.lower[0]),
    )
    _freeze(rate, target_abs)
    return _StokesCunninghamBounds(
        rate,
        target_abs,
        float(density.upper[0]),
        float(viscosity.lower[0]),
        float(mean_free_path.lower[0]),
        float(mean_free_path.upper[0]),
    )


def _prepare_thermophoresis_bounds(
    plan: PhysicsPlan,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    ranges: Mapping[str, PrimitiveRange],
) -> tuple[_ThermophoresisRuntimeBounds | None, FloatArray | None]:
    thermophoresis = plan.thermophoresis
    if thermophoresis is None:
        return None, None
    if isinstance(thermophoresis, TalbotThermophoresisPlan):
        temperature = _positive_scalar_range(ranges, thermophoresis.gas_temperature_field)
        gradient = _primitive_range(ranges, thermophoresis.gas_temperature_gradient_field, 2)
        density = _positive_scalar_range(ranges, thermophoresis.gas_density_field)
        viscosity = _positive_scalar_range(ranges, thermophoresis.gas_dynamic_viscosity_field)
        conductivity = _positive_scalar_range(ranges, thermophoresis.gas_thermal_conductivity_field)
        mean_free_path = _positive_scalar_range(ranges, thermophoresis.gas_mean_free_path_field)
        gradient_abs_upper = np.nextafter(
            np.maximum(np.abs(gradient.lower), np.abs(gradient.upper)), np.inf
        )
        prepared = talbot_thermophoresis_global_bounds(
            mass_kg=mass_kg,
            drag_diameter_m=drag_diameter_m,
            gas_temperature_lower_K=float(temperature.lower[0]),
            gas_temperature_gradient_abs_upper_K_m=gradient_abs_upper,
            gas_dynamic_viscosity_upper_Pa_s=float(viscosity.upper[0]),
            gas_density_lower_kg_m3=float(density.lower[0]),
            gas_thermal_conductivity_lower_W_m_K=float(conductivity.lower[0]),
            gas_thermal_conductivity_upper_W_m_K=float(conductivity.upper[0]),
            gas_mean_free_path_lower_m=float(mean_free_path.lower[0]),
            gas_mean_free_path_upper_m=float(mean_free_path.upper[0]),
            particle_thermal_conductivity_W_m_K=(
                thermophoresis.particle_thermal_conductivity_W_m_K
            ),
            thermal_slip_coefficient=thermophoresis.thermal_slip_coefficient,
            momentum_exchange_coefficient=thermophoresis.momentum_exchange_coefficient,
            thermal_exchange_coefficient=thermophoresis.thermal_exchange_coefficient,
        )
        prepared.static_applicable.setflags(write=False)
        return _TalbotBounds(prepared.static_applicable), prepared.acceleration_abs_upper_m_s2
    gas_velocity = _primitive_range(ranges, thermophoresis.gas_velocity_field, 2)
    temperature = _positive_scalar_range(ranges, thermophoresis.gas_temperature_field)
    heat_flux = _primitive_range(
        ranges,
        thermophoresis.gas_translational_heat_flux_field,
        2,
    )
    mean_free_path = _positive_scalar_range(
        ranges,
        thermophoresis.gas_mean_free_path_field,
    )
    heat_flux_abs_upper = np.nextafter(
        np.maximum(np.abs(heat_flux.lower), np.abs(heat_flux.upper)),
        np.inf,
    )
    prepared = waldmann_gallis_global_bounds(
        mass_kg=mass_kg,
        drag_diameter_m=drag_diameter_m,
        gas_temperature_lower_K=float(temperature.lower[0]),
        gas_translational_heat_flux_abs_upper_W_m2=heat_flux_abs_upper,
        gas_mean_free_path_lower_m=float(mean_free_path.lower[0]),
        gas_molecular_mass_kg=thermophoresis.gas_molecular_mass_kg,
    )
    gas_velocity_abs_upper = np.nextafter(
        np.maximum(np.abs(gas_velocity.lower), np.abs(gas_velocity.upper)),
        np.inf,
    )
    prepared.static_applicable.setflags(write=False)
    _freeze(gas_velocity_abs_upper)
    return (
        _WaldmannGallisBounds(
            prepared.static_applicable,
            gas_velocity_abs_upper,
            float(temperature.lower[0]),
        ),
        prepared.acceleration_abs_upper_m_s2,
    )


def _prepare_ion_drag_bounds(
    plan: PhysicsPlan,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_number: FloatArray,
    charge_bounds: _ChargeRuntimeBounds | None,
    ranges: Mapping[str, PrimitiveRange],
) -> tuple[_IonDragRuntimeBounds | None, FloatArray | None]:
    ion_drag = plan.ion_drag
    if ion_drag is None:
        return None, None
    charge_lower, charge_upper = _ion_drag_charge_interval(charge_number, charge_bounds)
    if isinstance(ion_drag, BarnesCollisionlessIonDragPlan):
        return _prepare_barnes_ion_drag_bounds(
            ion_drag,
            mass_kg,
            electrostatic_radius_m,
            charge_lower,
            charge_upper,
            ranges,
        )
    ion_density = _positive_scalar_range(
        ranges,
        ion_drag.positive_ion_number_density_field,
    )
    ion_voltage = _positive_scalar_range(
        ranges,
        ion_drag.positive_ion_thermal_voltage_field,
    )
    ion_velocity = _primitive_range(ranges, ion_drag.positive_ion_velocity_field, 2)
    ion_velocity_abs_upper = np.nextafter(
        np.maximum(np.abs(ion_velocity.lower), np.abs(ion_velocity.upper)),
        np.inf,
    )
    ion_mass = _positive_scalar_range(ranges, ion_drag.effective_positive_ion_mass_field)
    if isinstance(ion_drag, RelativeFlowScreenedIonDragPlan):
        screening = _positive_scalar_range(ranges, ion_drag.screening_length_field)
        mean_free_path = _positive_scalar_range(
            ranges,
            ion_drag.ion_neutral_mean_free_path_field,
        )
        acceleration = relative_flow_screened_ion_drag_global_bound(
            mass_kg=mass_kg,
            electrostatic_radius_m=electrostatic_radius_m,
            charge_number_lower=charge_lower,
            charge_number_upper=charge_upper,
            positive_ion_number_density_upper_m3=float(ion_density.upper[0]),
            positive_ion_thermal_voltage_lower_V=float(ion_voltage.lower[0]),
            positive_ion_thermal_voltage_upper_V=float(ion_voltage.upper[0]),
            effective_positive_ion_mass_lower_kg=float(ion_mass.lower[0]),
            effective_positive_ion_mass_upper_kg=float(ion_mass.upper[0]),
            screening_length_upper_m=float(screening.upper[0]),
            ion_neutral_mean_free_path_upper_m=float(mean_free_path.upper[0]),
            maximum_relative_ion_speed_m_s=ion_drag.maximum_relative_ion_speed_m_s,
        )
        _freeze(ion_velocity_abs_upper)
        return _RelativeFlowScreenedIonDragBounds(ion_velocity_abs_upper), acceleration

    electron_voltage = _positive_scalar_range(
        ranges,
        ion_drag.electron_thermal_voltage_field,
    )
    acceleration = electric_field_directed_image_ion_drag_global_bound(
        mass_kg=mass_kg,
        electrostatic_radius_m=electrostatic_radius_m,
        charge_number_lower=charge_lower,
        charge_number_upper=charge_upper,
        positive_ion_number_density_lower_m3=float(ion_density.lower[0]),
        positive_ion_number_density_upper_m3=float(ion_density.upper[0]),
        electron_thermal_voltage_upper_V=float(electron_voltage.upper[0]),
        positive_ion_thermal_voltage_lower_V=float(ion_voltage.lower[0]),
        positive_ion_thermal_voltage_upper_V=float(ion_voltage.upper[0]),
        positive_ion_velocity_abs_upper_m_s=ion_velocity_abs_upper,
        effective_positive_ion_mass_lower_kg=float(ion_mass.lower[0]),
        effective_positive_ion_mass_upper_kg=float(ion_mass.upper[0]),
    )
    return None, acceleration


def _ion_drag_charge_interval(
    charge_number: FloatArray,
    charge_bounds: _ChargeRuntimeBounds | None,
) -> tuple[FloatArray, FloatArray]:
    if charge_bounds is None:
        return charge_number, charge_number
    count = int(charge_number.size)
    return (
        np.full(count, charge_bounds.model.charge_number_lower, dtype=np.float64),
        np.full(count, charge_bounds.model.charge_number_upper, dtype=np.float64),
    )


def _prepare_barnes_ion_drag_bounds(
    ion_drag: BarnesCollisionlessIonDragPlan,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    charge_lower: FloatArray,
    charge_upper: FloatArray,
    ranges: Mapping[str, PrimitiveRange],
) -> tuple[_BarnesIonDragBounds, FloatArray]:
    electron_density = _positive_scalar_range(
        ranges,
        ion_drag.electron_number_density_field,
    )
    ion_density = _positive_scalar_range(
        ranges,
        ion_drag.positive_ion_number_density_field,
    )
    electron_temperature = _positive_scalar_range(
        ranges,
        ion_drag.electron_temperature_field,
    )
    ion_temperature = _positive_scalar_range(
        ranges,
        ion_drag.positive_ion_temperature_field,
    )
    ion_velocity = _primitive_range(ranges, ion_drag.positive_ion_velocity_field, 2)
    mean_free_path = _positive_scalar_range(
        ranges,
        ion_drag.ion_neutral_mean_free_path_field,
    )
    prepared = barnes_collisionless_global_bounds(
        mass_kg=mass_kg,
        electrostatic_radius_m=electrostatic_radius_m,
        charge_number_lower=charge_lower,
        charge_number_upper=charge_upper,
        electron_number_density_lower_m3=float(electron_density.lower[0]),
        electron_number_density_upper_m3=float(electron_density.upper[0]),
        positive_ion_number_density_lower_m3=float(ion_density.lower[0]),
        positive_ion_number_density_upper_m3=float(ion_density.upper[0]),
        electron_temperature_lower_K=float(electron_temperature.lower[0]),
        electron_temperature_upper_K=float(electron_temperature.upper[0]),
        positive_ion_temperature_lower_K=float(ion_temperature.lower[0]),
        positive_ion_temperature_upper_K=float(ion_temperature.upper[0]),
        ion_neutral_mean_free_path_lower_m=float(mean_free_path.lower[0]),
        positive_ion_mass_kg=ion_drag.positive_ion_mass_kg,
        maximum_ion_drift_ratio=ion_drag.maximum_ion_drift_ratio,
    )
    ion_velocity_abs_upper = np.nextafter(
        np.maximum(np.abs(ion_velocity.lower), np.abs(ion_velocity.upper)),
        np.inf,
    )
    prepared.static_applicable.setflags(write=False)
    _freeze(ion_velocity_abs_upper)
    return (
        _BarnesIonDragBounds(
            prepared.static_applicable,
            ion_velocity_abs_upper,
            prepared.positive_ion_temperature_lower_K,
        ),
        prepared.acceleration_abs_upper_m_s2,
    )


def _prepare_dielectrophoresis_acceleration_bound(
    dielectrophoresis: QuasistaticSphericalDielectrophoresisPlan | None,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    ranges: Mapping[str, PrimitiveRange],
) -> FloatArray | None:
    if dielectrophoresis is None:
        return None
    # A derived radius can round to the immediate float64 successor of the
    # independently serialized certified cap.  Treat that single successor as
    # the cap's outward-rounded representation; any larger value remains an
    # applicability error.
    certified_radius_upper_m = math.nextafter(
        dielectrophoresis.maximum_point_dipole_radius_m,
        math.inf,
    )
    if bool((electrostatic_radius_m > certified_radius_upper_m).any()):
        raise PhysicsEvaluationError(
            "DEP electrostatic radius exceeds maximum_point_dipole_radius_m"
        )
    gradient = _primitive_range(
        ranges,
        dielectrophoresis.gradient_mean_e_squared_field,
        2,
    )
    gradient_abs_upper = np.nextafter(
        np.maximum(np.abs(gradient.lower), np.abs(gradient.upper)),
        np.inf,
    )
    return quasistatic_spherical_dep_acceleration_abs_upper(
        mass_kg=mass_kg,
        electrostatic_radius_m=electrostatic_radius_m,
        gradient_mean_e_squared_abs_upper_V2_m3=gradient_abs_upper,
        medium_relative_permittivity=dielectrophoresis.medium_relative_permittivity,
        real_clausius_mossotti_factor=(dielectrophoresis.real_clausius_mossotti_factor),
    )


def _prepare_lift_bounds(
    lift: LiftPlan | None,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    ranges: Mapping[str, PrimitiveRange],
) -> _LiftRuntimeBounds | None:
    if lift is None:
        return None
    if isinstance(lift, SaffmanLiftPlan):
        gas_velocity = _primitive_range(ranges, lift.gas_velocity_field, 2)
        gas_density = _positive_scalar_range(ranges, lift.gas_density_field)
        viscosity = _positive_scalar_range(ranges, lift.gas_dynamic_viscosity_field)
        mean_free_path = _positive_scalar_range(ranges, lift.gas_mean_free_path_field)
        vorticity = _primitive_range(ranges, lift.out_of_plane_gas_vorticity_field, 1)
        gas_velocity_abs_upper = np.nextafter(
            np.maximum(np.abs(gas_velocity.lower), np.abs(gas_velocity.upper)), np.inf
        )
        vorticity_abs_upper_raw = float(max(abs(vorticity.lower[0]), abs(vorticity.upper[0])))
        vorticity_abs_upper = (
            0.0
            if vorticity_abs_upper_raw == 0.0
            else math.nextafter(vorticity_abs_upper_raw, math.inf)
        )
        if vorticity.lower[0] > 0.0:
            vorticity_abs_lower = math.nextafter(float(vorticity.lower[0]), 0.0)
        elif vorticity.upper[0] < 0.0:
            vorticity_abs_lower = math.nextafter(float(abs(vorticity.upper[0])), 0.0)
        else:
            vorticity_abs_lower = 0.0
        prepared = saffman_lift_global_bounds(
            mass_kg=mass_kg,
            drag_diameter_m=drag_diameter_m,
            gas_density_lower_kg_m3=float(gas_density.lower[0]),
            gas_density_upper_kg_m3=float(gas_density.upper[0]),
            gas_dynamic_viscosity_lower_Pa_s=float(viscosity.lower[0]),
            gas_dynamic_viscosity_upper_Pa_s=float(viscosity.upper[0]),
            gas_mean_free_path_upper_m=float(mean_free_path.upper[0]),
            out_of_plane_gas_vorticity_abs_lower_s_inv=vorticity_abs_lower,
            out_of_plane_gas_vorticity_abs_upper_s_inv=vorticity_abs_upper,
            gas_velocity_abs_upper_m_s=gas_velocity_abs_upper,
        )
        _freeze(
            prepared.coupling_rate_abs_upper_s_inv,
            prepared.gas_velocity_abs_upper_m_s,
            prepared.shear_reynolds_lower,
            prepared.shear_reynolds_upper,
        )
        prepared.static_applicable.setflags(write=False)
        return _SaffmanLiftBounds(
            prepared.coupling_rate_abs_upper_s_inv,
            prepared.gas_velocity_abs_upper_m_s,
            prepared.shear_reynolds_lower,
            prepared.shear_reynolds_upper,
            prepared.static_applicable,
            float(gas_density.upper[0]),
            float(viscosity.lower[0]),
        )
    gas_velocity = _primitive_range(ranges, lift.gas_velocity_field, 2)
    gas_density = _positive_scalar_range(ranges, lift.gas_density_field)
    mean_free_path = _positive_scalar_range(ranges, lift.gas_mean_free_path_field)
    vorticity = _primitive_range(ranges, lift.azimuthal_gas_vorticity_field, 1)
    gas_velocity_abs_upper = np.nextafter(
        np.maximum(np.abs(gas_velocity.lower), np.abs(gas_velocity.upper)),
        np.inf,
    )
    vorticity_abs_upper = math.nextafter(
        float(max(abs(vorticity.lower[0]), abs(vorticity.upper[0]))),
        math.inf,
    )
    prepared = rarefied_vorticity_lift_global_bounds(
        mass_kg=mass_kg,
        drag_diameter_m=drag_diameter_m,
        gas_density_upper_kg_m3=float(gas_density.upper[0]),
        gas_mean_free_path_lower_m=float(mean_free_path.lower[0]),
        gas_mean_free_path_upper_m=float(mean_free_path.upper[0]),
        azimuthal_gas_vorticity_abs_upper_s_inv=vorticity_abs_upper,
        gas_velocity_abs_upper_m_s=gas_velocity_abs_upper,
        lift_coefficient=lift.lift_coefficient,
    )
    _freeze(
        prepared.coupling_rate_abs_upper_s_inv,
        prepared.gas_velocity_abs_upper_m_s,
    )
    prepared.static_applicable.setflags(write=False)
    return _RarefiedVorticityLiftBounds(
        prepared.coupling_rate_abs_upper_s_inv,
        prepared.gas_velocity_abs_upper_m_s,
        prepared.static_applicable,
    )


def _prepare_external_acceleration_bound(
    plan: PhysicsPlan,
    mass_kg: FloatArray,
    displaced_volume_m3: FloatArray,
    charge_number: FloatArray,
    charge_bounds: _ChargeRuntimeBounds | None,
    ion_drag_acceleration_abs_upper_m_s2: FloatArray | None,
    thermophoresis_acceleration_abs_upper_m_s2: FloatArray | None,
    dielectrophoresis_acceleration_abs_upper_m_s2: FloatArray | None,
    ranges: Mapping[str, PrimitiveRange],
) -> FloatArray:
    external = np.zeros((mass_kg.size, 2), dtype=np.float64)
    electric = plan.electric
    if electric is not None:
        field = _primitive_range(ranges, electric.electric_field, 2)
        field_abs = np.nextafter(np.maximum(np.abs(field.lower), np.abs(field.upper)), np.inf)
        electric_charge = charge_number
        if charge_bounds is not None:
            maximum_abs_charge = max(
                abs(charge_bounds.model.charge_number_lower),
                abs(charge_bounds.model.charge_number_upper),
            )
            electric_charge = np.full(charge_number.size, maximum_abs_charge, dtype=np.float64)
        contribution = electric_acceleration_abs_upper(
            charge_number=electric_charge,
            mass_kg=mass_kg,
            electric_field_abs_upper_V_m=field_abs,
        )
        external = np.nextafter(external + contribution, np.inf)
    gravity = plan.gravity_buoyancy
    if gravity is not None:
        density = _positive_scalar_range(ranges, gravity.gas_density_field)
        contribution = gravity_buoyancy_acceleration_abs_upper(
            mass_kg=mass_kg,
            displaced_volume_m3=displaced_volume_m3,
            gas_density_lower_kg_m3=float(density.lower[0]),
            gas_density_upper_kg_m3=float(density.upper[0]),
            gravity_m_s2=gravity.gravity_m_s2,
        )
        external = np.nextafter(external + contribution, np.inf)
    if ion_drag_acceleration_abs_upper_m_s2 is not None:
        external = np.nextafter(external + ion_drag_acceleration_abs_upper_m_s2, np.inf)
    if thermophoresis_acceleration_abs_upper_m_s2 is not None:
        external = np.nextafter(
            external + thermophoresis_acceleration_abs_upper_m_s2,
            np.inf,
        )
    if dielectrophoresis_acceleration_abs_upper_m_s2 is not None:
        external = np.nextafter(
            external + dielectrophoresis_acceleration_abs_upper_m_s2,
            np.inf,
        )
    if not bool(np.isfinite(external).all()):
        raise PhysicsEvaluationError("force enclosure contains a non-finite bound")
    return external


def _prepare_localizable_external_base(
    plan: PhysicsPlan,
    mass_kg: FloatArray,
    displaced_volume_m3: FloatArray,
    thermophoresis_acceleration_abs_upper_m_s2: FloatArray | None,
    dielectrophoresis_acceleration_abs_upper_m_s2: FloatArray | None,
    ranges: Mapping[str, PrimitiveRange],
) -> FloatArray | None:
    """Prepare force terms unchanged by local charge/ion-flow tightening."""

    ion_drag = plan.ion_drag
    localizable = ion_drag is None or isinstance(ion_drag, RelativeFlowScreenedIonDragPlan)
    if not localizable or (plan.electric is None and ion_drag is None):
        return None
    result = np.zeros((mass_kg.size, 2), dtype=np.float64)
    gravity = plan.gravity_buoyancy
    if gravity is not None:
        density = _positive_scalar_range(ranges, gravity.gas_density_field)
        contribution = gravity_buoyancy_acceleration_abs_upper(
            mass_kg=mass_kg,
            displaced_volume_m3=displaced_volume_m3,
            gas_density_lower_kg_m3=float(density.lower[0]),
            gas_density_upper_kg_m3=float(density.upper[0]),
            gravity_m_s2=gravity.gravity_m_s2,
        )
        result = np.nextafter(result + contribution, np.inf)
    if thermophoresis_acceleration_abs_upper_m_s2 is not None:
        result = np.nextafter(
            result + thermophoresis_acceleration_abs_upper_m_s2,
            np.inf,
        )
    if dielectrophoresis_acceleration_abs_upper_m_s2 is not None:
        result = np.nextafter(
            result + dielectrophoresis_acceleration_abs_upper_m_s2,
            np.inf,
        )
    if not bool(np.isfinite(result).all()):
        raise PhysicsEvaluationError("localizable force base contains a non-finite bound")
    return result


def _constant_acceleration(
    plan: PhysicsPlan,
    coordinate_system: CoordinateSystem,
    count: int,
    mass_kg: FloatArray,
    electrostatic_radius_m: FloatArray,
    displaced_volume_m3: FloatArray,
    charge_number: FloatArray,
    ranges: Mapping[str, PrimitiveRange],
) -> FloatArray | None:
    if (
        plan.evolves_continuous_state
        or not plan.has_force
        or coordinate_system != "cartesian_xy"
        or plan.drag is not None
        or plan.thermophoresis is not None
        or plan.ion_drag is not None
        or plan.lift is not None
    ):
        return None
    acceleration = np.zeros((count, 2), dtype=np.float64)
    dielectrophoresis = plan.dielectrophoresis
    if dielectrophoresis is not None:
        gradient = _primitive_range(
            ranges,
            dielectrophoresis.gradient_mean_e_squared_field,
            2,
        ).constant
        if gradient is None:
            return None
        add_quasistatic_spherical_dep_acceleration(
            acceleration,
            mass_kg=mass_kg,
            electrostatic_radius_m=electrostatic_radius_m,
            gradient_mean_e_squared_V2_m3=np.broadcast_to(gradient, (count, 2)),
            medium_relative_permittivity=dielectrophoresis.medium_relative_permittivity,
            real_clausius_mossotti_factor=(dielectrophoresis.real_clausius_mossotti_factor),
        )
    electric = plan.electric
    if electric is not None:
        field = _primitive_range(ranges, electric.electric_field, 2).constant
        if field is None:
            return None
        add_electric_coulomb_acceleration(
            acceleration,
            charge_number=charge_number,
            mass_kg=mass_kg,
            electric_field_V_m=np.broadcast_to(field, (count, 2)),
        )
    gravity = plan.gravity_buoyancy
    if gravity is not None:
        density = _positive_scalar_range(ranges, gravity.gas_density_field).constant
        if density is None:
            return None
        add_gravity_buoyancy_acceleration(
            acceleration,
            mass_kg=mass_kg,
            displaced_volume_m3=displaced_volume_m3,
            gas_density_kg_m3=np.full(count, float(density[0]), dtype=np.float64),
            gravity_m_s2=gravity.gravity_m_s2,
        )
    return acceleration


def _initial_numerical_status(value: UInt8Array | None, count: int) -> UInt8Array:
    if value is None:
        return np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8)
    status = np.asarray(value)
    if status.shape != (count,) or status.dtype != np.uint8:
        raise PhysicsEvaluationError(
            "physics numerical status must be a uint8 array with shape [N]"
        )
    return status.copy()


def _mark_physics_failure(status: UInt8Array, failed: BoolArray) -> None:
    status[(status == NUMERICAL_STATUS_OK) & failed] = PHYSICS_NUMERICAL_FAILURE


def _primitive_range(
    ranges: Mapping[str, PrimitiveRange],
    name: str,
    component_count: int,
) -> PrimitiveRange:
    value = ranges.get(name)
    if value is None:
        raise PhysicsEvaluationError(f"primitive extrema for {name!r} were not prepared")
    if value.lower.shape != (component_count,) or value.upper.shape != (component_count,):
        raise PhysicsEvaluationError(f"primitive extrema for {name!r} have an invalid shape")
    if not bool(np.isfinite(value.lower).all() and np.isfinite(value.upper).all()):
        raise PhysicsEvaluationError(f"primitive extrema for {name!r} must be finite")
    if bool((value.lower > value.upper).any()):
        raise PhysicsEvaluationError(f"primitive extrema for {name!r} are reversed")
    if value.constant is not None:
        if value.constant.shape != (component_count,) or not bool(
            np.isfinite(value.constant).all()
        ):
            raise PhysicsEvaluationError(f"constant primitive {name!r} has an invalid value")
    return value


def _positive_scalar_range(
    ranges: Mapping[str, PrimitiveRange],
    name: str,
) -> PrimitiveRange:
    value = _primitive_range(ranges, name, 1)
    if float(value.lower[0]) <= 0.0:
        raise PhysicsEvaluationError(
            f"primitive {name!r} cannot be certified positive after interpolation roundoff"
        )
    return value


def _nonnegative_scalar_range(
    ranges: Mapping[str, PrimitiveRange],
    name: str,
) -> PrimitiveRange:
    value = _primitive_range(ranges, name, 1)
    if value.constant is not None:
        if float(value.constant[0]) < 0.0:
            raise PhysicsEvaluationError(f"constant primitive {name!r} must be nonnegative")
        return PrimitiveRange(value.constant, value.constant, value.constant)
    if float(value.lower[0]) < 0.0:
        raise PhysicsEvaluationError(
            f"primitive {name!r} cannot be certified nonnegative after interpolation roundoff"
        )
    return value


def _particle_count(first: FloatArray, *rest: FloatArray) -> int:
    if first.ndim != 1 or not bool(np.isfinite(first).all()):
        raise PhysicsEvaluationError("particle scalar inputs must be finite one-dimensional arrays")
    count = int(first.size)
    if any(value.shape != (count,) or not bool(np.isfinite(value).all()) for value in rest):
        raise PhysicsEvaluationError("particle scalar inputs must have matching finite shape [N]")
    return count


def _freeze(*arrays: FloatArray) -> None:
    for value in arrays:
        value.setflags(write=False)


def _prepare_physics_stage_bindings(
    plan: PhysicsPlan,
    charge_bounds: _ChargeRuntimeBounds | None,
) -> PhysicsStageBindings:
    """Resolve catalog branches once; no sampled-stage arithmetic is performed here."""

    sampled_values: dict[str, int] = {}
    rules: list[int] = []
    for item in plan.required_fields:
        sampled_values[item.name] = len(rules)
        rules.extend([1 if item.positive else 0] * len(item.components))
    charge_plan = plan.charge
    charge_inputs = _charge_stage_bindings(charge_plan, charge_bounds, sampled_values)
    drag = plan.drag
    drag_code = DRAG_NONE
    gas_velocity = -1
    gas_density = -1
    gas_temperature = -1
    gas_viscosity = -1
    gas_mean_free_path = -1
    gas_molecular_mass_kg = 1.0
    epstein_delta = 1.0
    epstein_diffuse_reflection_fraction = 0.0
    epstein_maximum_speed_ratio = 1.0
    if isinstance(drag, EpsteinDragPlan):
        drag_code = DRAG_EPSTEIN
        gas_velocity = _primitive_binding(sampled_values, drag.gas_velocity_field)
        gas_density = _primitive_binding(sampled_values, drag.gas_density_field)
        gas_temperature = _primitive_binding(sampled_values, drag.gas_temperature_field)
        gas_mean_free_path = _primitive_binding(sampled_values, drag.gas_mean_free_path_field)
        gas_molecular_mass_kg = drag.gas_molecular_mass_kg
        epstein_delta = drag.delta
        epstein_maximum_speed_ratio = drag.maximum_speed_ratio
    elif isinstance(drag, FiniteSpeedEpsteinDragPlan):
        drag_code = DRAG_EPSTEIN_FINITE_SPEED
        gas_velocity = _primitive_binding(sampled_values, drag.gas_velocity_field)
        gas_density = _primitive_binding(sampled_values, drag.gas_density_field)
        gas_temperature = _primitive_binding(sampled_values, drag.gas_temperature_field)
        gas_mean_free_path = _primitive_binding(sampled_values, drag.gas_mean_free_path_field)
        gas_molecular_mass_kg = drag.gas_molecular_mass_kg
        epstein_diffuse_reflection_fraction = drag.diffuse_reflection_fraction
        epstein_maximum_speed_ratio = drag.maximum_speed_ratio
    elif isinstance(drag, StokesCunninghamDragPlan):
        drag_code = DRAG_STOKES_CUNNINGHAM
        gas_velocity = _primitive_binding(sampled_values, drag.gas_velocity_field)
        gas_density = _primitive_binding(sampled_values, drag.gas_density_field)
        gas_viscosity = _primitive_binding(sampled_values, drag.gas_dynamic_viscosity_field)
        gas_mean_free_path = _primitive_binding(sampled_values, drag.gas_mean_free_path_field)
    thermophoresis_inputs = _thermophoresis_stage_bindings(
        plan.thermophoresis,
        drag,
        sampled_values,
        gas_velocity,
        gas_density,
        gas_temperature,
        gas_viscosity,
        gas_mean_free_path,
    )
    electric = plan.electric
    image_ion_drag = plan.ion_drag
    image_electric_field = (
        image_ion_drag.electric_field
        if isinstance(image_ion_drag, ElectricFieldDirectedImageIonDragPlan)
        else None
    )
    electric_field_name = electric.electric_field if electric is not None else image_electric_field
    electric_field = (
        -1
        if electric_field_name is None
        else _primitive_binding(sampled_values, electric_field_name)
    )
    ion_drag_inputs = _ion_drag_stage_bindings(
        plan.ion_drag, charge_plan, charge_inputs, sampled_values, electric_field
    )
    (
        dep_enabled,
        dep_gradient_mean_e_squared,
        dep_medium_relative_permittivity,
        dep_real_clausius_mossotti_factor,
    ) = _dielectrophoresis_stage_bindings(plan.dielectrophoresis, sampled_values)
    lift_inputs = _lift_stage_bindings(
        plan.lift,
        drag,
        plan.thermophoresis,
        sampled_values,
        gas_velocity,
        gas_density,
        gas_viscosity,
        gas_mean_free_path,
        thermophoresis_inputs,
    )
    gravity = plan.gravity_buoyancy
    gravity_density = (
        -1 if gravity is None else _primitive_binding(sampled_values, gravity.gas_density_field)
    )
    gravity_vector = (0.0, 0.0) if gravity is None else gravity.gravity_m_s2
    negative = plan.charge
    if isinstance(negative, AggregateRelativeDriftChargePlan) and negative.has_negative_ion_current:
        rules[sampled_values[cast(str, negative.negative_ion_number_density_field)]] = 2
    primitive_rules = np.asarray(rules, dtype=np.int8)
    primitive_rules.setflags(write=False)
    return PhysicsStageBindings(
        charge_inputs,
        drag_code,
        gas_velocity,
        gas_density,
        gas_temperature,
        gas_viscosity,
        gas_mean_free_path,
        gas_molecular_mass_kg,
        epstein_delta,
        epstein_diffuse_reflection_fraction,
        epstein_maximum_speed_ratio,
        thermophoresis_inputs,
        electric is not None,
        electric_field,
        ion_drag_inputs,
        dep_enabled,
        dep_gradient_mean_e_squared,
        dep_medium_relative_permittivity,
        dep_real_clausius_mossotti_factor,
        lift_inputs,
        gravity is not None,
        gravity_density,
        gravity_vector,
        primitive_rules,
    )


def _primitive_binding(bindings: Mapping[str, int], name: str) -> int:
    """Look up a validated required primitive while preparing the model table."""

    return bindings[name]


class PhysicsNumericOutputs(NamedTuple):
    """Slot-owned outputs passed across the native stage boundary."""

    acceleration_m_s2: FloatArray
    charge_rate_number_s: FloatArray
    charge_rate_derivative_s_inv: FloatArray
    applicable: BoolArray
    linear_drag_rate_s_inv: FloatArray
    target_velocity_m_s: FloatArray
    additive_acceleration_m_s2: FloatArray


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _primitive_scalar(values: FloatArray, offset: int) -> FloatArray:
    return values.reshape(values.size)[:0] if offset < 0 else values[:, offset]


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _primitive_vector(values: FloatArray, offset: int) -> FloatArray:
    return values[:0, :2] if offset < 0 else values[:, offset : offset + 2]


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def evaluate_prepared_physics_into(
    context: PhysicsNumericContext,
    particle_index: Int64Array,
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    sampled_values: FloatArray,
    outputs: PhysicsNumericOutputs,
    error_code: UInt8Array,
    incoming_status: UInt8Array,
) -> None:
    """Bind sampled columns and evaluate the existing force kernel in one native call."""

    bindings = context.bindings
    _validate_stage_rows_into(
        velocity_m_s, charge_number, sampled_values, bindings.primitive_rules, error_code
    )
    for row in range(particle_index.size):
        incoming_status[row] = error_code[row]
    evaluate_physics_tile_into(
        particle_index,
        velocity_m_s,
        charge_number,
        context.mass_kg,
        context.drag_diameter_m,
        context.electrostatic_radius_m,
        context.displaced_volume_m3,
        bindings.charge_inputs.code,
        _primitive_scalar(sampled_values, bindings.charge_inputs.electron_density_m3),
        _primitive_scalar(sampled_values, bindings.charge_inputs.positive_ion_density_m3),
        _primitive_scalar(sampled_values, bindings.charge_inputs.electron_temperature_K),
        _primitive_scalar(sampled_values, bindings.charge_inputs.positive_ion_temperature_K),
        _primitive_vector(sampled_values, bindings.charge_inputs.positive_ion_velocity_m_s),
        bindings.charge_inputs.positive_ion_mass_kg,
        bindings.charge_inputs.maximum_ion_drift_ratio,
        _primitive_scalar(
            sampled_values, bindings.charge_inputs.aggregate_electron_thermal_voltage_V
        ),
        _primitive_scalar(
            sampled_values, bindings.charge_inputs.aggregate_positive_ion_thermal_voltage_V
        ),
        _primitive_scalar(
            sampled_values, bindings.charge_inputs.aggregate_effective_positive_ion_mass_kg
        ),
        _primitive_scalar(
            sampled_values, bindings.charge_inputs.aggregate_negative_ion_number_density_m3
        ),
        _primitive_scalar(
            sampled_values, bindings.charge_inputs.aggregate_negative_ion_thermal_voltage_V
        ),
        _primitive_vector(
            sampled_values, bindings.charge_inputs.aggregate_negative_ion_velocity_m_s
        ),
        _primitive_scalar(
            sampled_values, bindings.charge_inputs.aggregate_effective_negative_ion_mass_kg
        ),
        _primitive_scalar(sampled_values, bindings.charge_inputs.aggregate_screening_length_m),
        bindings.charge_inputs.aggregate_maximum_relative_ion_speed_m_s,
        bindings.charge_inputs.charge_number_lower,
        bindings.charge_inputs.charge_number_upper,
        bindings.drag_code,
        _primitive_vector(sampled_values, bindings.gas_velocity),
        _primitive_scalar(sampled_values, bindings.gas_density),
        _primitive_scalar(sampled_values, bindings.gas_temperature),
        _primitive_scalar(sampled_values, bindings.gas_viscosity),
        _primitive_scalar(sampled_values, bindings.gas_mean_free_path),
        bindings.gas_molecular_mass_kg,
        bindings.epstein_delta,
        bindings.epstein_diffuse_reflection_fraction,
        bindings.epstein_maximum_speed_ratio,
        bindings.thermophoresis_inputs.code,
        _primitive_vector(sampled_values, bindings.thermophoresis_inputs.gas_velocity_m_s),
        _primitive_scalar(sampled_values, bindings.thermophoresis_inputs.gas_temperature_K),
        _primitive_vector(
            sampled_values, bindings.thermophoresis_inputs.gas_translational_heat_flux_W_m2
        ),
        _primitive_scalar(sampled_values, bindings.thermophoresis_inputs.gas_mean_free_path_m),
        bindings.thermophoresis_inputs.gas_molecular_mass_kg,
        bindings.thermophoresis_inputs.maximum_speed_ratio,
        _primitive_vector(
            sampled_values, bindings.thermophoresis_inputs.gas_temperature_gradient_K_m
        ),
        _primitive_scalar(sampled_values, bindings.thermophoresis_inputs.gas_density_kg_m3),
        _primitive_scalar(
            sampled_values, bindings.thermophoresis_inputs.gas_dynamic_viscosity_Pa_s
        ),
        _primitive_scalar(
            sampled_values, bindings.thermophoresis_inputs.gas_thermal_conductivity_W_m_K
        ),
        bindings.thermophoresis_inputs.particle_thermal_conductivity_W_m_K,
        bindings.thermophoresis_inputs.thermal_slip_coefficient,
        bindings.thermophoresis_inputs.momentum_exchange_coefficient,
        bindings.thermophoresis_inputs.thermal_exchange_coefficient,
        bindings.ion_drag_inputs.code,
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.electron_density_m3),
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.positive_ion_density_m3),
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.electron_temperature_K),
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.positive_ion_temperature_K),
        _primitive_vector(sampled_values, bindings.ion_drag_inputs.positive_ion_velocity_m_s),
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.ion_neutral_mean_free_path_m),
        bindings.ion_drag_inputs.positive_ion_mass_kg,
        bindings.ion_drag_inputs.maximum_ion_drift_ratio,
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.electron_thermal_voltage_V),
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.positive_ion_thermal_voltage_V),
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.effective_positive_ion_mass_kg),
        _primitive_scalar(sampled_values, bindings.ion_drag_inputs.screening_length_m),
        _primitive_vector(sampled_values, bindings.ion_drag_inputs.electric_field_V_m),
        bindings.ion_drag_inputs.maximum_relative_ion_speed_m_s,
        bindings.dep_enabled,
        _primitive_vector(sampled_values, bindings.dep_gradient_mean_e_squared),
        bindings.dep_medium_relative_permittivity,
        bindings.dep_real_clausius_mossotti_factor,
        bindings.lift_inputs.code,
        _primitive_vector(sampled_values, bindings.lift_inputs.gas_velocity_m_s),
        _primitive_scalar(sampled_values, bindings.lift_inputs.gas_density_kg_m3),
        _primitive_scalar(sampled_values, bindings.lift_inputs.gas_mean_free_path_m),
        _primitive_scalar(sampled_values, bindings.lift_inputs.gas_dynamic_viscosity_Pa_s),
        _primitive_scalar(sampled_values, bindings.lift_inputs.azimuthal_gas_vorticity_s_inv),
        bindings.lift_inputs.lift_coefficient,
        bindings.electric_enabled,
        _primitive_vector(sampled_values, bindings.electric_field),
        bindings.gravity_enabled,
        _primitive_scalar(sampled_values, bindings.gravity_density),
        bindings.gravity_vector[0],
        bindings.gravity_vector[1],
        outputs.acceleration_m_s2,
        outputs.charge_rate_number_s,
        outputs.charge_rate_derivative_s_inv,
        outputs.applicable,
        outputs.linear_drag_rate_s_inv,
        outputs.target_velocity_m_s,
        outputs.additive_acceleration_m_s2,
        error_code,
    )
    # Raw physics codes overlap the shared status vocabulary; normalize only
    # rows that entered the force kernel without a field/state failure.
    for row in range(particle_index.size):
        if incoming_status[row] != NUMERICAL_STATUS_OK:
            continue
        code = error_code[row]
        if code == ERROR_CHARGE_INVARIANT:
            error_code[row] = INTEGRATOR_ACCURACY_FAILURE
        elif code != NUMERICAL_STATUS_OK:
            error_code[row] = PHYSICS_NUMERICAL_FAILURE


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _validate_stage_rows_into(
    velocity_m_s: FloatArray,
    charge_number: FloatArray,
    sampled_values: FloatArray,
    primitive_rules: NDArray[np.int8],
    status: UInt8Array,
) -> None:
    """Row-local arithmetic checks on already validated primitive/state shapes."""

    for row in range(charge_number.size):
        if status[row] != NUMERICAL_STATUS_OK:
            continue
        valid = (
            np.isfinite(charge_number[row])
            and np.isfinite(velocity_m_s[row, 0])
            and np.isfinite(velocity_m_s[row, 1])
        )
        for column in range(primitive_rules.size):
            value = sampled_values[row, column]
            rule = primitive_rules[column]
            valid = (
                valid
                and np.isfinite(value)
                and (rule != 1 or value > 0.0)
                and (rule != 2 or value >= 0.0)
            )
        if not valid:
            status[row] = PHYSICS_NUMERICAL_FAILURE


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def brownian_thermal_variance_into(
    mass_kg: FloatArray,
    particle_index: Int64Array,
    temperature_K: FloatArray,
    variance_m2_s2: FloatArray,
    numerical_status: UInt8Array,
) -> None:
    """Evaluate FDT variance with one first-failure-preserving native row pass."""

    for row in range(particle_index.size):
        temperature = temperature_K[row]
        value = BOLTZMANN_J_K * temperature / mass_kg[particle_index[row]]
        valid = np.isfinite(temperature) and temperature > 0.0
        valid = valid and np.isfinite(value) and value > 0.0
        if not valid and numerical_status[row] == NUMERICAL_STATUS_OK:
            numerical_status[row] = PHYSICS_NUMERICAL_FAILURE
        variance_m2_s2[row] = value if numerical_status[row] == NUMERICAL_STATUS_OK else 0.0
