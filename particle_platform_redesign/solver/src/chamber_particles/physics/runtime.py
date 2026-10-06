"""Small deterministic physics runtime over sampled primitive values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
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
    PhysicsPlan,
    PlasmaContinuousChargePlan,
    QuasistaticSphericalDielectrophoresisPlan,
    RarefiedVorticityLiftPlan,
    RelativeFlowScreenedIonDragPlan,
    StokesCunninghamDragPlan,
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
    THERMOPHORESIS_NONE,
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
    STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS,
    STOKES_CUNNINGHAM_MAX_REYNOLDS,
    STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS,
    VACUUM_PERMITTIVITY_F_M,
    WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS,
    PhysicsEvaluationError,
    add_electric_coulomb_acceleration,
    add_gravity_buoyancy_acceleration,
    add_quasistatic_spherical_dep_acceleration,
    barnes_collisionless_continuous_applicability_batch,
    barnes_collisionless_global_bounds,
    electric_acceleration_abs_upper,
    electric_field_directed_image_ion_drag_global_bound,
    epstein_continuous_applicability_batch,
    epstein_finite_speed_continuous_applicability_batch,
    epstein_finite_speed_factors,
    epstein_rate_s_inv,
    gravity_buoyancy_acceleration_abs_upper,
    linear_drag_acceleration_abs_upper_batch,
    quasistatic_spherical_dep_acceleration_abs_upper,
    rarefied_vorticity_lift_acceleration_abs_upper_batch,
    rarefied_vorticity_lift_global_bounds,
    relative_flow_screened_continuous_applicability_batch,
    relative_flow_screened_ion_drag_global_bound,
    relative_flow_screened_ion_drag_local_bound,
    stokes_cunningham_continuous_applicability_batch,
    stokes_cunningham_rate_upper_s_inv,
    waldmann_gallis_continuous_applicability_batch,
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

PHYSICS_RUNTIME_REVISION = "signed_ion_compiled_physics_runtime_v20"

_EMPTY_SCALAR = np.empty(0, dtype=np.float64)
_EMPTY_VECTOR = np.empty((0, 2), dtype=np.float64)


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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class _ChargeStageInputs:
    """Resolved sampled inputs for one compiled continuous-charge pass."""

    code: int
    electron_density_m3: FloatArray
    positive_ion_density_m3: FloatArray
    electron_temperature_K: FloatArray
    positive_ion_temperature_K: FloatArray
    positive_ion_velocity_m_s: FloatArray
    positive_ion_mass_kg: float
    maximum_ion_drift_ratio: float
    aggregate_electron_thermal_voltage_V: FloatArray
    aggregate_positive_ion_thermal_voltage_V: FloatArray
    aggregate_effective_positive_ion_mass_kg: FloatArray
    aggregate_negative_ion_number_density_m3: FloatArray
    aggregate_negative_ion_thermal_voltage_V: FloatArray
    aggregate_negative_ion_velocity_m_s: FloatArray
    aggregate_effective_negative_ion_mass_kg: FloatArray
    aggregate_screening_length_m: FloatArray
    aggregate_maximum_relative_ion_speed_m_s: float
    charge_number_lower: float
    charge_number_upper: float


_NO_CHARGE_STAGE_INPUTS = _ChargeStageInputs(
    CHARGE_FIXED,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_VECTOR,
    1.0,
    OML_MAX_ION_DRIFT_RATIO,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_VECTOR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
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
class _RarefiedVorticityLiftBounds:
    """Prepared high-Kn certificate and cross-velocity acceleration bound."""

    coupling_rate_abs_upper_s_inv: FloatArray
    gas_velocity_abs_upper_m_s: FloatArray
    static_applicable: BoolArray


@dataclass(frozen=True, slots=True)
class _ThermophoresisStageInputs:
    """Sampled neutral-gas inputs for the selected thermophoresis revision."""

    code: int
    gas_velocity_m_s: FloatArray
    gas_temperature_K: FloatArray
    gas_translational_heat_flux_W_m2: FloatArray
    gas_mean_free_path_m: FloatArray
    gas_molecular_mass_kg: float
    maximum_speed_ratio: float


_NO_THERMOPHORESIS_STAGE_INPUTS = _ThermophoresisStageInputs(
    THERMOPHORESIS_NONE,
    _EMPTY_VECTOR,
    _EMPTY_SCALAR,
    _EMPTY_VECTOR,
    _EMPTY_SCALAR,
    1.0,
    1.0,
)


@dataclass(frozen=True, slots=True)
class _LiftStageInputs:
    """Sampled neutral-gas inputs for the optional RZ lift sensitivity."""

    code: int
    gas_velocity_m_s: FloatArray
    gas_density_kg_m3: FloatArray
    gas_mean_free_path_m: FloatArray
    azimuthal_gas_vorticity_s_inv: FloatArray
    lift_coefficient: float


_NO_LIFT_STAGE_INPUTS = _LiftStageInputs(
    LIFT_NONE,
    _EMPTY_VECTOR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    1.0,
)


@dataclass(frozen=True, slots=True)
class _IonDragStageInputs:
    """Sampled inputs for one selected ion-drag revision."""

    code: int
    electron_density_m3: FloatArray
    positive_ion_density_m3: FloatArray
    electron_temperature_K: FloatArray
    positive_ion_temperature_K: FloatArray
    positive_ion_velocity_m_s: FloatArray
    ion_neutral_mean_free_path_m: FloatArray
    positive_ion_mass_kg: float
    maximum_ion_drift_ratio: float
    electron_thermal_voltage_V: FloatArray
    positive_ion_thermal_voltage_V: FloatArray
    effective_positive_ion_mass_kg: FloatArray
    screening_length_m: FloatArray
    electric_field_V_m: FloatArray
    maximum_relative_ion_speed_m_s: float


_NO_ION_DRAG_STAGE_INPUTS = _IonDragStageInputs(
    ION_DRAG_NONE,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_VECTOR,
    _EMPTY_SCALAR,
    1.0,
    1.0,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_SCALAR,
    _EMPTY_VECTOR,
    1.0,
)


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
    thermophoresis_bounds: _WaldmannGallisBounds | None
    ion_drag_bounds: _IonDragRuntimeBounds | None
    lift_bounds: _RarefiedVorticityLiftBounds | None
    external_acceleration_abs_upper_m_s2: FloatArray
    localizable_external_base_abs_upper_m_s2: FloatArray | None
    constant_acceleration_m_s2: FloatArray | None

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
        temperature_K = _sampled_positive_scalar_batch(
            sampled_values,
            self.plan.drag.gas_temperature_field,
            count,
            status,
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            result = BOLTZMANN_J_K * temperature_K / self.mass_kg[indices]
        _mark_physics_failure(status, ~np.isfinite(result) | (result <= 0.0))
        result[status != NUMERICAL_STATUS_OK] = 0.0
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
        """Evaluate valid rows and preserve the first row-local numerical failure."""

        indices = np.asarray(particle_index, dtype=np.int64)
        velocity = np.asarray(velocity_m_s, dtype=np.float64)
        charge = np.asarray(charge_number, dtype=np.float64)
        count = int(indices.size)
        if indices.ndim != 1 or velocity.shape != (count, 2) or charge.shape != (count,):
            raise PhysicsEvaluationError("physics stage state has an invalid shape")
        if bool((indices < 0).any() or (indices >= self.mass_kg.size).any()):
            raise PhysicsEvaluationError("physics particle indices are outside resident state")
        incoming_status = _initial_numerical_status(numerical_status, count)
        invalid_state = ~np.isfinite(velocity).all(axis=1) | ~np.isfinite(charge)
        _mark_physics_failure(incoming_status, invalid_state)
        safe_velocity = velocity.copy()
        safe_charge = charge.copy()
        failed_input = incoming_status != NUMERICAL_STATUS_OK
        safe_velocity[failed_input] = 0.0
        safe_charge[failed_input] = 0.0

        charge_plan = self.plan.charge
        charge_inputs = _charge_stage_inputs(
            charge_plan,
            self.charge_bounds,
            sampled_values,
            count,
            incoming_status,
        )

        drag = self.plan.drag
        drag_code = DRAG_NONE
        gas_velocity = _EMPTY_VECTOR
        gas_density = _EMPTY_SCALAR
        gas_temperature = _EMPTY_SCALAR
        gas_viscosity = _EMPTY_SCALAR
        gas_mean_free_path = _EMPTY_SCALAR
        gas_molecular_mass_kg = 1.0
        epstein_delta = 1.0
        epstein_diffuse_reflection_fraction = 0.0
        epstein_maximum_speed_ratio = 1.0
        if isinstance(drag, EpsteinDragPlan):
            drag_code = DRAG_EPSTEIN
            gas_velocity = _sampled_vector_batch(
                sampled_values, drag.gas_velocity_field, count, incoming_status
            )
            gas_density = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_density_field, count, incoming_status
            )
            gas_temperature = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_temperature_field, count, incoming_status
            )
            gas_mean_free_path = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_mean_free_path_field, count, incoming_status
            )
            gas_molecular_mass_kg = drag.gas_molecular_mass_kg
            epstein_delta = drag.delta
            epstein_maximum_speed_ratio = drag.maximum_speed_ratio
        elif isinstance(drag, FiniteSpeedEpsteinDragPlan):
            drag_code = DRAG_EPSTEIN_FINITE_SPEED
            gas_velocity = _sampled_vector_batch(
                sampled_values, drag.gas_velocity_field, count, incoming_status
            )
            gas_density = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_density_field, count, incoming_status
            )
            gas_temperature = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_temperature_field, count, incoming_status
            )
            gas_mean_free_path = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_mean_free_path_field, count, incoming_status
            )
            gas_molecular_mass_kg = drag.gas_molecular_mass_kg
            epstein_diffuse_reflection_fraction = drag.diffuse_reflection_fraction
            epstein_maximum_speed_ratio = drag.maximum_speed_ratio
        elif isinstance(drag, StokesCunninghamDragPlan):
            drag_code = DRAG_STOKES_CUNNINGHAM
            gas_velocity = _sampled_vector_batch(
                sampled_values, drag.gas_velocity_field, count, incoming_status
            )
            gas_density = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_density_field, count, incoming_status
            )
            gas_viscosity = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_dynamic_viscosity_field, count, incoming_status
            )
            gas_mean_free_path = _sampled_positive_scalar_batch(
                sampled_values, drag.gas_mean_free_path_field, count, incoming_status
            )

        thermophoresis_inputs = _thermophoresis_stage_inputs(
            self.plan.thermophoresis,
            drag,
            sampled_values,
            count,
            incoming_status,
            gas_velocity,
            gas_temperature,
            gas_mean_free_path,
        )

        electric = self.plan.electric
        image_ion_drag = self.plan.ion_drag
        image_electric_field = (
            image_ion_drag.electric_field
            if isinstance(image_ion_drag, ElectricFieldDirectedImageIonDragPlan)
            else None
        )
        electric_field_name = (
            electric.electric_field if electric is not None else image_electric_field
        )
        electric_field = (
            _EMPTY_VECTOR
            if electric_field_name is None
            else _sampled_vector_batch(sampled_values, electric_field_name, count, incoming_status)
        )

        ion_drag_inputs = _ion_drag_stage_inputs(
            self.plan.ion_drag,
            charge_plan,
            charge_inputs,
            sampled_values,
            count,
            incoming_status,
            electric_field,
        )

        (
            dep_enabled,
            dep_gradient_mean_e_squared,
            dep_medium_relative_permittivity,
            dep_real_clausius_mossotti_factor,
        ) = _dielectrophoresis_stage_inputs(
            self.plan.dielectrophoresis,
            sampled_values,
            count,
            incoming_status,
        )

        lift_inputs = _lift_stage_inputs(
            self.plan.lift,
            drag,
            self.plan.thermophoresis,
            sampled_values,
            count,
            incoming_status,
            gas_velocity,
            gas_density,
            gas_mean_free_path,
            thermophoresis_inputs,
        )

        gravity = self.plan.gravity_buoyancy
        gravity_density = (
            _EMPTY_SCALAR
            if gravity is None
            else _sampled_positive_scalar_batch(
                sampled_values, gravity.gas_density_field, count, incoming_status
            )
        )
        gravity_vector = (0.0, 0.0) if gravity is None else gravity.gravity_m_s2

        stage_workspace = (
            PhysicsRuntimeWorkspace.allocate(count) if workspace is None else workspace
        )
        evaluation = stage_workspace.evaluation(count)
        error_code = stage_workspace.error_code[:count]
        error_code[:] = incoming_status
        evaluate_physics_tile_into(
            indices,
            safe_velocity,
            safe_charge,
            self.mass_kg,
            self.drag_diameter_m,
            self.electrostatic_radius_m,
            self.displaced_volume_m3,
            charge_inputs.code,
            charge_inputs.electron_density_m3,
            charge_inputs.positive_ion_density_m3,
            charge_inputs.electron_temperature_K,
            charge_inputs.positive_ion_temperature_K,
            charge_inputs.positive_ion_velocity_m_s,
            charge_inputs.positive_ion_mass_kg,
            charge_inputs.maximum_ion_drift_ratio,
            charge_inputs.aggregate_electron_thermal_voltage_V,
            charge_inputs.aggregate_positive_ion_thermal_voltage_V,
            charge_inputs.aggregate_effective_positive_ion_mass_kg,
            charge_inputs.aggregate_negative_ion_number_density_m3,
            charge_inputs.aggregate_negative_ion_thermal_voltage_V,
            charge_inputs.aggregate_negative_ion_velocity_m_s,
            charge_inputs.aggregate_effective_negative_ion_mass_kg,
            charge_inputs.aggregate_screening_length_m,
            charge_inputs.aggregate_maximum_relative_ion_speed_m_s,
            charge_inputs.charge_number_lower,
            charge_inputs.charge_number_upper,
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
            thermophoresis_inputs.code,
            thermophoresis_inputs.gas_velocity_m_s,
            thermophoresis_inputs.gas_temperature_K,
            thermophoresis_inputs.gas_translational_heat_flux_W_m2,
            thermophoresis_inputs.gas_mean_free_path_m,
            thermophoresis_inputs.gas_molecular_mass_kg,
            thermophoresis_inputs.maximum_speed_ratio,
            ion_drag_inputs.code,
            ion_drag_inputs.electron_density_m3,
            ion_drag_inputs.positive_ion_density_m3,
            ion_drag_inputs.electron_temperature_K,
            ion_drag_inputs.positive_ion_temperature_K,
            ion_drag_inputs.positive_ion_velocity_m_s,
            ion_drag_inputs.ion_neutral_mean_free_path_m,
            ion_drag_inputs.positive_ion_mass_kg,
            ion_drag_inputs.maximum_ion_drift_ratio,
            ion_drag_inputs.electron_thermal_voltage_V,
            ion_drag_inputs.positive_ion_thermal_voltage_V,
            ion_drag_inputs.effective_positive_ion_mass_kg,
            ion_drag_inputs.screening_length_m,
            ion_drag_inputs.electric_field_V_m,
            ion_drag_inputs.maximum_relative_ion_speed_m_s,
            dep_enabled,
            dep_gradient_mean_e_squared,
            dep_medium_relative_permittivity,
            dep_real_clausius_mossotti_factor,
            lift_inputs.code,
            lift_inputs.gas_velocity_m_s,
            lift_inputs.gas_density_kg_m3,
            lift_inputs.gas_mean_free_path_m,
            lift_inputs.azimuthal_gas_vorticity_s_inv,
            lift_inputs.lift_coefficient,
            electric is not None,
            electric_field,
            gravity is not None,
            gravity_density,
            gravity_vector[0],
            gravity_vector[1],
            evaluation.acceleration_m_s2,
            evaluation.charge_rate_number_s,
            evaluation.charge_rate_derivative_s_inv,
            evaluation.applicable,
            evaluation.linear_drag_rate_s_inv,
            evaluation.target_velocity_m_s,
            evaluation.additive_acceleration_m_s2,
            error_code,
        )
        invariant_failure = (incoming_status == NUMERICAL_STATUS_OK) & (
            error_code == ERROR_CHARGE_INVARIANT
        )
        error_code[invariant_failure] = INTEGRATOR_ACCURACY_FAILURE
        physics_failure = (
            (incoming_status == NUMERICAL_STATUS_OK)
            & (error_code != NUMERICAL_STATUS_OK)
            & ~invariant_failure
        )
        error_code[physics_failure] = PHYSICS_NUMERICAL_FAILURE
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
        rate_upper = np.zeros(count, dtype=np.float64)
        target_abs_upper = np.zeros((count, 2), dtype=np.float64)
        bounds = self.drag_bounds
        if bounds is not None:
            rate_upper = bounds.rate_upper_s_inv[indices].copy()
            target_abs_upper[:] = bounds.target_velocity_abs_upper_m_s
        return LinearRelaxationAbsBounds(
            rate_upper,
            target_abs_upper,
        )

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
        result = self.external_acceleration_abs_upper_m_s2[indices].copy()
        lift_bounds = self.lift_bounds
        if lift_bounds is not None:
            lift, status = rarefied_vorticity_lift_acceleration_abs_upper_batch(
                coupling_rate_abs_upper_s_inv=(lift_bounds.coupling_rate_abs_upper_s_inv[indices]),
                gas_velocity_abs_upper_m_s=lift_bounds.gas_velocity_abs_upper_m_s,
                velocity_abs_upper_m_s=velocity,
                numerical_status=status,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.nextafter(result + lift, np.inf)
        invalid = ~np.isfinite(result).all(axis=1) | (result < 0.0).any(axis=1)
        _mark_physics_failure(status, invalid)
        result[status != NUMERICAL_STATUS_OK] = 0.0
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
        finite = np.isfinite(velocity_lower).all(axis=1)
        finite &= np.isfinite(velocity_upper).all(axis=1)
        finite &= np.isfinite(charge_lower) & np.isfinite(charge_upper)
        ordered = (velocity_lower <= velocity_upper).all(axis=1)
        ordered &= charge_lower <= charge_upper
        status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
        status[~(finite & ordered)] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        applicable = finite & ordered
        result = self.localizable_external_base_abs_upper_m_s2[indices].copy()

        electric = self.plan.electric
        if electric is not None:
            field_lower, field_upper = _local_range(
                primitive_ranges,
                electric.electric_field,
                count,
                2,
            )
            field_abs_upper = _local_nonnegative_upper(
                np.maximum(np.abs(field_lower), np.abs(field_upper))
            )
            charge_abs_upper = _local_nonnegative_upper(
                np.maximum(np.abs(charge_lower), np.abs(charge_upper))
            )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                electric_bound = _local_nonnegative_upper(
                    charge_abs_upper[:, None]
                    * ELEMENTARY_CHARGE_C
                    * field_abs_upper
                    / self.mass_kg[indices, None]
                )
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.nextafter(result + electric_bound, np.inf)

        ion_drag = self.plan.ion_drag
        if ion_drag is not None:
            if not isinstance(ion_drag, RelativeFlowScreenedIonDragPlan):
                raise PhysicsEvaluationError(
                    "selected ion-drag revision has no local acceleration certificate"
                )
            ion_density_lower, ion_density_upper = _local_positive_scalar_range(
                primitive_ranges,
                ion_drag.positive_ion_number_density_field,
                count,
            )
            del ion_density_lower
            ion_voltage_lower, ion_voltage_upper = _local_positive_scalar_range(
                primitive_ranges,
                ion_drag.positive_ion_thermal_voltage_field,
                count,
            )
            ion_velocity_lower, ion_velocity_upper = _local_range(
                primitive_ranges,
                ion_drag.positive_ion_velocity_field,
                count,
                2,
            )
            ion_mass_lower, ion_mass_upper = _local_positive_scalar_range(
                primitive_ranges,
                ion_drag.effective_positive_ion_mass_field,
                count,
            )
            _, screening_upper = _local_positive_scalar_range(
                primitive_ranges,
                ion_drag.screening_length_field,
                count,
            )
            _, mean_free_path_upper = _local_positive_scalar_range(
                primitive_ranges,
                ion_drag.ion_neutral_mean_free_path_field,
                count,
            )
            ion_bound, ion_applicable = relative_flow_screened_ion_drag_local_bound(
                mass_kg=self.mass_kg[indices],
                electrostatic_radius_m=self.electrostatic_radius_m[indices],
                charge_number_lower=charge_lower,
                charge_number_upper=charge_upper,
                velocity_lower_m_s=velocity_lower,
                velocity_upper_m_s=velocity_upper,
                positive_ion_number_density_upper_m3=ion_density_upper,
                positive_ion_thermal_voltage_lower_V=ion_voltage_lower,
                positive_ion_thermal_voltage_upper_V=ion_voltage_upper,
                positive_ion_velocity_lower_m_s=ion_velocity_lower,
                positive_ion_velocity_upper_m_s=ion_velocity_upper,
                effective_positive_ion_mass_lower_kg=ion_mass_lower,
                effective_positive_ion_mass_upper_kg=ion_mass_upper,
                screening_length_upper_m=screening_upper,
                ion_neutral_mean_free_path_upper_m=mean_free_path_upper,
                maximum_relative_ion_speed_m_s=ion_drag.maximum_relative_ion_speed_m_s,
            )
            applicable &= ion_applicable
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.nextafter(result + ion_bound, np.inf)

        lift_bounds = self.lift_bounds
        velocity_abs_upper = np.maximum(np.abs(velocity_lower), np.abs(velocity_upper))
        if lift_bounds is not None:
            lift, lift_status = rarefied_vorticity_lift_acceleration_abs_upper_batch(
                coupling_rate_abs_upper_s_inv=(lift_bounds.coupling_rate_abs_upper_s_inv[indices]),
                gas_velocity_abs_upper_m_s=lift_bounds.gas_velocity_abs_upper_m_s,
                velocity_abs_upper_m_s=velocity_abs_upper,
                numerical_status=np.full(count, NUMERICAL_STATUS_OK, dtype=np.uint8),
            )
            status[lift_status != NUMERICAL_STATUS_OK] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.nextafter(result + lift, np.inf)
        invalid = ~np.isfinite(result).all(axis=1) | (result < 0.0).any(axis=1)
        status[invalid] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        applicable[status != CONTINUOUS_APPLICABILITY_OK] = False
        result[status != CONTINUOUS_APPLICABILITY_OK] = 0.0
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
        result, status = self.additive_acceleration_abs_upper_batch(
            indices,
            velocity,
            numerical_status,
        )
        bounds = self.drag_bounds
        if bounds is not None:
            drag, status = linear_drag_acceleration_abs_upper_batch(
                rate_upper_s_inv=bounds.rate_upper_s_inv[indices],
                target_velocity_abs_upper_m_s=bounds.target_velocity_abs_upper_m_s,
                velocity_abs_upper_m_s=velocity,
                numerical_status=status,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                result = np.nextafter(result + drag, np.inf)
        invalid = ~np.isfinite(result).all(axis=1) | (result < 0.0).any(axis=1)
        _mark_physics_failure(status, invalid)
        result[status != NUMERICAL_STATUS_OK] = 0.0
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

        applicable = np.ones(count, dtype=np.bool_)
        status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
        drag = self.plan.drag
        bounds = self.drag_bounds
        drag_result = _epstein_continuous_applicability(
            drag,
            bounds,
            self.drag_diameter_m[indices],
            velocity,
        )
        if drag_result is not None:
            drag_applicable, drag_status = drag_result
        elif isinstance(drag, StokesCunninghamDragPlan) and isinstance(
            bounds,
            _StokesCunninghamBounds,
        ):
            drag_applicable, drag_status = stokes_cunningham_continuous_applicability_batch(
                drag_diameter_m=self.drag_diameter_m[indices],
                velocity_abs_upper_m_s=velocity,
                gas_velocity_abs_upper_m_s=bounds.target_velocity_abs_upper_m_s,
                gas_density_upper_kg_m3=bounds.density_upper_kg_m3,
                gas_dynamic_viscosity_lower_Pa_s=bounds.dynamic_viscosity_lower_Pa_s,
                gas_mean_free_path_lower_m=bounds.mean_free_path_lower_m,
                gas_mean_free_path_upper_m=bounds.mean_free_path_upper_m,
            )
        elif drag is not None or bounds is not None:
            raise PhysicsEvaluationError(
                "drag applicability bounds do not match the resolved model"
            )
        else:
            drag_applicable = None
            drag_status = None
        if drag_applicable is not None:
            applicable &= drag_applicable
            status[drag_status != CONTINUOUS_APPLICABILITY_OK] = (
                CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
            )

        charge_result = _charge_continuous_applicability(
            self.plan.charge,
            self.charge_bounds,
            self.electrostatic_radius_m[indices],
            velocity,
        )
        _merge_continuous_applicability(applicable, status, charge_result)

        thermophoresis_result = _thermophoresis_continuous_applicability(
            self.plan.thermophoresis,
            self.thermophoresis_bounds,
            indices,
            velocity,
        )
        _merge_continuous_applicability(applicable, status, thermophoresis_result)

        ion_result = _ion_drag_continuous_applicability(
            self.plan.ion_drag,
            self.ion_drag_bounds,
            indices,
            velocity,
        )
        _merge_continuous_applicability(applicable, status, ion_result)

        lift_result = _lift_continuous_applicability(
            self.plan.lift,
            self.lift_bounds,
            indices,
            velocity,
        )
        _merge_continuous_applicability(applicable, status, lift_result)
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
        ordered_finite = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
        status[~ordered_finite] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        certified = ordered_finite.copy()
        if self.charge_bounds is not None:
            certified &= lower >= self.charge_bounds.model.charge_number_lower
            certified &= upper <= self.charge_bounds.model.charge_number_upper
        certified[status != CONTINUOUS_APPLICABILITY_OK] = False
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
        finite_velocity = np.isfinite(velocity_lower).all(axis=1)
        finite_velocity &= np.isfinite(velocity_upper).all(axis=1)
        finite_velocity &= (velocity_lower <= velocity_upper).all(axis=1)
        certified, status = self.charge_interval_inside_prepared_invariant(
            charge_lower,
            charge_upper,
        )
        status[~finite_velocity] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        certified &= finite_velocity

        _merge_local_certificate(
            certified,
            status,
            _local_drag_applicability(
                self.plan.drag,
                self.drag_diameter_m[indices],
                velocity_lower,
                velocity_upper,
                primitive_ranges,
            ),
        )
        _merge_local_certificate(
            certified,
            status,
            _local_charge_applicability(
                self.plan.charge,
                self.charge_bounds,
                self.electrostatic_radius_m[indices],
                velocity_lower,
                velocity_upper,
                primitive_ranges,
            ),
        )
        _merge_local_certificate(
            certified,
            status,
            _local_thermophoresis_applicability(
                self.plan.thermophoresis,
                self.drag_diameter_m[indices],
                velocity_lower,
                velocity_upper,
                primitive_ranges,
            ),
        )
        _merge_local_certificate(
            certified,
            status,
            _local_ion_drag_applicability(
                self.plan.ion_drag,
                self.ion_drag_bounds,
                indices,
                self.electrostatic_radius_m[indices],
                charge_lower,
                charge_upper,
                velocity_lower,
                velocity_upper,
                primitive_ranges,
            ),
        )
        _merge_local_certificate(
            certified,
            status,
            _local_lift_applicability(
                self.plan.lift,
                self.drag_diameter_m[indices],
                primitive_ranges,
            ),
        )
        certified[status != CONTINUOUS_APPLICABILITY_OK] = False
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
        if self.constant_acceleration_m_s2 is not None:
            total += int(self.constant_acceleration_m_s2.nbytes)
        return total


def _merge_continuous_applicability(
    applicable: BoolArray,
    status: UInt8Array,
    result: tuple[BoolArray, UInt8Array] | None,
) -> None:
    """Merge one optional model verdict while preserving its first failure."""

    if result is None:
        return
    model_applicable, model_status = result
    applicable &= model_applicable
    first_failure = (status == CONTINUOUS_APPLICABILITY_OK) & (
        model_status != CONTINUOUS_APPLICABILITY_OK
    )
    status[first_failure] = model_status[first_failure]


def _lift_continuous_applicability(
    plan: RarefiedVorticityLiftPlan | None,
    bounds: _RarefiedVorticityLiftBounds | None,
    particle_index: Int64Array,
    velocity_abs_upper_m_s: FloatArray,
) -> tuple[BoolArray, UInt8Array] | None:
    """Certify the fixed high-Kn gate and a finite velocity enclosure."""

    if plan is None and bounds is None:
        return None
    if plan is None or bounds is None:
        raise PhysicsEvaluationError("lift applicability bounds do not match the revision")
    numerical_ok = np.isfinite(velocity_abs_upper_m_s).all(axis=1)
    numerical_ok &= (velocity_abs_upper_m_s >= 0.0).all(axis=1)
    status = np.full(particle_index.size, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    return bounds.static_applicable[particle_index] & numerical_ok, status


def _charge_stage_inputs(
    plan: ChargePlan | None,
    bounds: _ChargeRuntimeBounds | None,
    sampled_values: Mapping[str, FloatArray],
    count: int,
    numerical_status: UInt8Array,
) -> _ChargeStageInputs:
    """Bind the selected charge revision to one stage's sampled primitives."""

    if plan is None:
        if bounds is not None:
            raise PhysicsEvaluationError("continuous charge bounds do not match the resolved model")
        return _NO_CHARGE_STAGE_INPUTS
    if bounds is None:
        raise PhysicsEvaluationError("continuous charge bounds were not prepared")

    electron_density = _sampled_positive_scalar_batch(
        sampled_values,
        plan.electron_number_density_field,
        count,
        numerical_status,
    )
    ion_density = _sampled_positive_scalar_batch(
        sampled_values,
        plan.positive_ion_number_density_field,
        count,
        numerical_status,
    )
    ion_velocity = _sampled_vector_batch(
        sampled_values,
        plan.positive_ion_velocity_field,
        count,
        numerical_status,
    )
    charge_lower = bounds.model.charge_number_lower
    charge_upper = bounds.model.charge_number_upper

    if isinstance(plan, PlasmaContinuousChargePlan):
        charge_code, maximum_drift_ratio = _charge_kernel_selection(plan)
        electron_temperature = _sampled_positive_scalar_batch(
            sampled_values,
            plan.electron_temperature_field,
            count,
            numerical_status,
        )
        ion_temperature = _sampled_positive_scalar_batch(
            sampled_values,
            plan.positive_ion_temperature_field,
            count,
            numerical_status,
        )
        return _ChargeStageInputs(
            charge_code,
            electron_density,
            ion_density,
            electron_temperature,
            ion_temperature,
            ion_velocity,
            plan.positive_ion_mass_kg,
            maximum_drift_ratio,
            _EMPTY_SCALAR,
            _EMPTY_SCALAR,
            _EMPTY_SCALAR,
            _EMPTY_SCALAR,
            _EMPTY_SCALAR,
            _EMPTY_VECTOR,
            _EMPTY_SCALAR,
            _EMPTY_SCALAR,
            1.0,
            charge_lower,
            charge_upper,
        )

    charge_code, maximum_relative_speed = _charge_kernel_selection(plan)
    electron_voltage = _sampled_positive_scalar_batch(
        sampled_values,
        plan.electron_thermal_voltage_field,
        count,
        numerical_status,
    )
    ion_voltage = _sampled_positive_scalar_batch(
        sampled_values,
        plan.positive_ion_thermal_voltage_field,
        count,
        numerical_status,
    )
    ion_mass = _sampled_positive_scalar_batch(
        sampled_values,
        plan.effective_positive_ion_mass_field,
        count,
        numerical_status,
    )
    negative_density = _EMPTY_SCALAR
    negative_voltage = _EMPTY_SCALAR
    negative_velocity = _EMPTY_VECTOR
    negative_mass = _EMPTY_SCALAR
    if plan.has_negative_ion_current:
        negative_density = _sampled_nonnegative_scalar_batch(
            sampled_values,
            cast(str, plan.negative_ion_number_density_field),
            count,
            numerical_status,
        )
        negative_voltage = _sampled_positive_scalar_batch(
            sampled_values,
            cast(str, plan.negative_ion_thermal_voltage_field),
            count,
            numerical_status,
        )
        negative_velocity = _sampled_vector_batch(
            sampled_values,
            cast(str, plan.negative_ion_velocity_field),
            count,
            numerical_status,
        )
        negative_mass = _sampled_positive_scalar_batch(
            sampled_values,
            cast(str, plan.effective_negative_ion_mass_field),
            count,
            numerical_status,
        )
    screening_length = _sampled_positive_scalar_batch(
        sampled_values,
        plan.screening_length_field,
        count,
        numerical_status,
    )
    return _ChargeStageInputs(
        charge_code,
        electron_density,
        ion_density,
        _EMPTY_SCALAR,
        _EMPTY_SCALAR,
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


def _thermophoresis_stage_inputs(
    plan: WaldmannGallisThermophoresisPlan | None,
    drag_plan: DragPlan | None,
    sampled_values: Mapping[str, FloatArray],
    count: int,
    numerical_status: UInt8Array,
    drag_gas_velocity_m_s: FloatArray,
    drag_gas_temperature_K: FloatArray,
    drag_gas_mean_free_path_m: FloatArray,
) -> _ThermophoresisStageInputs:
    """Bind one stage's neutral background, reusing compatible drag samples."""

    if plan is None:
        return _NO_THERMOPHORESIS_STAGE_INPUTS
    if isinstance(drag_plan, EpsteinDragPlan | FiniteSpeedEpsteinDragPlan):
        gas_velocity = drag_gas_velocity_m_s
        gas_temperature = drag_gas_temperature_K
        mean_free_path = drag_gas_mean_free_path_m
    elif drag_plan is None:
        gas_velocity = _sampled_vector_batch(
            sampled_values,
            plan.gas_velocity_field,
            count,
            numerical_status,
        )
        gas_temperature = _sampled_positive_scalar_batch(
            sampled_values,
            plan.gas_temperature_field,
            count,
            numerical_status,
        )
        mean_free_path = _sampled_positive_scalar_batch(
            sampled_values,
            plan.gas_mean_free_path_field,
            count,
            numerical_status,
        )
    else:
        raise PhysicsEvaluationError(
            "thermophoresis neutral-gas inputs do not match the resolved drag model"
        )
    heat_flux = _sampled_vector_batch(
        sampled_values,
        plan.gas_translational_heat_flux_field,
        count,
        numerical_status,
    )
    return _ThermophoresisStageInputs(
        THERMOPHORESIS_WALDMANN_GALLIS,
        gas_velocity,
        gas_temperature,
        heat_flux,
        mean_free_path,
        plan.gas_molecular_mass_kg,
        plan.maximum_speed_ratio,
    )


def _lift_stage_inputs(
    plan: RarefiedVorticityLiftPlan | None,
    drag_plan: DragPlan | None,
    thermophoresis_plan: WaldmannGallisThermophoresisPlan | None,
    sampled_values: Mapping[str, FloatArray],
    count: int,
    numerical_status: UInt8Array,
    drag_gas_velocity_m_s: FloatArray,
    drag_gas_density_kg_m3: FloatArray,
    drag_gas_mean_free_path_m: FloatArray,
    thermophoresis_inputs: _ThermophoresisStageInputs,
) -> _LiftStageInputs:
    """Bind lift primitives while reusing an already sampled neutral background."""

    if plan is None:
        return _NO_LIFT_STAGE_INPUTS
    if drag_plan is not None:
        gas_velocity = drag_gas_velocity_m_s
        gas_density = drag_gas_density_kg_m3
        mean_free_path = drag_gas_mean_free_path_m
    else:
        gas_velocity = (
            thermophoresis_inputs.gas_velocity_m_s
            if thermophoresis_plan is not None
            else _sampled_vector_batch(
                sampled_values,
                plan.gas_velocity_field,
                count,
                numerical_status,
            )
        )
        mean_free_path = (
            thermophoresis_inputs.gas_mean_free_path_m
            if thermophoresis_plan is not None
            else _sampled_positive_scalar_batch(
                sampled_values,
                plan.gas_mean_free_path_field,
                count,
                numerical_status,
            )
        )
        gas_density = _sampled_positive_scalar_batch(
            sampled_values,
            plan.gas_density_field,
            count,
            numerical_status,
        )
    vorticity = _sampled_scalar_batch(
        sampled_values,
        plan.azimuthal_gas_vorticity_field,
        count,
        numerical_status,
    )
    return _LiftStageInputs(
        LIFT_RAREFIED_VORTICITY_RZ,
        gas_velocity,
        gas_density,
        mean_free_path,
        vorticity,
        plan.lift_coefficient,
    )


def _dielectrophoresis_stage_inputs(
    plan: QuasistaticSphericalDielectrophoresisPlan | None,
    sampled_values: Mapping[str, FloatArray],
    count: int,
    numerical_status: UInt8Array,
) -> tuple[bool, FloatArray, float, float]:
    """Bind the optional DEP field and its resolved material constants."""

    if plan is None:
        return False, _EMPTY_VECTOR, 1.0, 0.0
    gradient = _sampled_vector_batch(
        sampled_values,
        plan.gradient_mean_e_squared_field,
        count,
        numerical_status,
    )
    return (
        True,
        gradient,
        plan.medium_relative_permittivity,
        plan.real_clausius_mossotti_factor,
    )


def _ion_drag_stage_inputs(
    plan: IonDragPlan | None,
    charge_plan: ChargePlan | None,
    charge_inputs: _ChargeStageInputs,
    sampled_values: Mapping[str, FloatArray],
    count: int,
    numerical_status: UInt8Array,
    electric_field_V_m: FloatArray,
) -> _IonDragStageInputs:
    """Bind one stage's ion background without duplicating coupled charge samples."""

    if plan is None:
        return _NO_ION_DRAG_STAGE_INPUTS
    if not isinstance(plan, BarnesCollisionlessIonDragPlan):
        return _aggregate_ion_drag_stage_inputs(
            plan,
            charge_plan,
            charge_inputs,
            sampled_values,
            count,
            numerical_status,
            electric_field_V_m,
        )
    if charge_plan is None:
        electron_density = _sampled_positive_scalar_batch(
            sampled_values,
            plan.electron_number_density_field,
            count,
            numerical_status,
        )
        ion_density = _sampled_positive_scalar_batch(
            sampled_values,
            plan.positive_ion_number_density_field,
            count,
            numerical_status,
        )
        electron_temperature = _sampled_positive_scalar_batch(
            sampled_values,
            plan.electron_temperature_field,
            count,
            numerical_status,
        )
        ion_temperature = _sampled_positive_scalar_batch(
            sampled_values,
            plan.positive_ion_temperature_field,
            count,
            numerical_status,
        )
        ion_velocity = _sampled_vector_batch(
            sampled_values,
            plan.positive_ion_velocity_field,
            count,
            numerical_status,
        )
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
    mean_free_path = _sampled_positive_scalar_batch(
        sampled_values,
        plan.ion_neutral_mean_free_path_field,
        count,
        numerical_status,
    )
    return _IonDragStageInputs(
        ION_DRAG_BARNES_COLLISIONLESS_EFFECTIVE_SPEED,
        electron_density,
        ion_density,
        electron_temperature,
        ion_temperature,
        ion_velocity,
        mean_free_path,
        plan.positive_ion_mass_kg,
        plan.maximum_ion_drift_ratio,
        _EMPTY_SCALAR,
        _EMPTY_SCALAR,
        _EMPTY_SCALAR,
        _EMPTY_SCALAR,
        _EMPTY_VECTOR,
        1.0,
    )


def _aggregate_ion_drag_stage_inputs(
    plan: RelativeFlowScreenedIonDragPlan | ElectricFieldDirectedImageIonDragPlan,
    charge_plan: ChargePlan | None,
    charge_inputs: _ChargeStageInputs,
    sampled_values: Mapping[str, FloatArray],
    count: int,
    numerical_status: UInt8Array,
    electric_field_V_m: FloatArray,
) -> _IonDragStageInputs:
    if isinstance(charge_plan, AggregateRelativeDriftChargePlan):
        ion_density = charge_inputs.positive_ion_density_m3
        electron_voltage = charge_inputs.aggregate_electron_thermal_voltage_V
        ion_voltage = charge_inputs.aggregate_positive_ion_thermal_voltage_V
        ion_velocity = charge_inputs.positive_ion_velocity_m_s
        ion_mass = charge_inputs.aggregate_effective_positive_ion_mass_kg
        screening_length = charge_inputs.aggregate_screening_length_m
    elif charge_plan is None:
        ion_density = _sampled_positive_scalar_batch(
            sampled_values,
            plan.positive_ion_number_density_field,
            count,
            numerical_status,
        )
        electron_voltage = (
            _sampled_positive_scalar_batch(
                sampled_values,
                plan.electron_thermal_voltage_field,
                count,
                numerical_status,
            )
            if isinstance(plan, ElectricFieldDirectedImageIonDragPlan)
            else _EMPTY_SCALAR
        )
        ion_voltage = _sampled_positive_scalar_batch(
            sampled_values,
            plan.positive_ion_thermal_voltage_field,
            count,
            numerical_status,
        )
        ion_velocity = _sampled_vector_batch(
            sampled_values,
            plan.positive_ion_velocity_field,
            count,
            numerical_status,
        )
        ion_mass = _sampled_positive_scalar_batch(
            sampled_values,
            plan.effective_positive_ion_mass_field,
            count,
            numerical_status,
        )
        screening_length = _sampled_positive_scalar_batch(
            sampled_values,
            plan.screening_length_field,
            count,
            numerical_status,
        )
    else:
        raise PhysicsEvaluationError(
            "aggregate-ion drag does not share the single-species charge background"
        )
    is_relative = isinstance(plan, RelativeFlowScreenedIonDragPlan)
    mean_free_path = (
        _sampled_positive_scalar_batch(
            sampled_values,
            plan.ion_neutral_mean_free_path_field,
            count,
            numerical_status,
        )
        if is_relative
        else _EMPTY_SCALAR
    )
    return _IonDragStageInputs(
        ION_DRAG_RELATIVE_FLOW_SCREENED if is_relative else ION_DRAG_ELECTRIC_FIELD_DIRECTED_IMAGE,
        _EMPTY_SCALAR,
        ion_density,
        _EMPTY_SCALAR,
        _EMPTY_SCALAR,
        ion_velocity,
        mean_free_path,
        1.0,
        1.0,
        electron_voltage,
        ion_voltage,
        ion_mass,
        screening_length,
        electric_field_V_m if not is_relative else _EMPTY_VECTOR,
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


def _merge_local_certificate(
    certified: BoolArray,
    status: UInt8Array,
    result: tuple[BoolArray, UInt8Array] | None,
) -> None:
    """Merge one model's conservative local certificate in place."""

    if result is None:
        return
    model_certified, model_status = result
    if model_certified.shape != certified.shape or model_status.shape != status.shape:
        raise PhysicsEvaluationError("local applicability result has an invalid shape")
    certified &= model_certified
    failed = model_status != CONTINUOUS_APPLICABILITY_OK
    status[(status == CONTINUOUS_APPLICABILITY_OK) & failed] = (
        CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    )


def _local_range(
    ranges: Mapping[str, LocalPrimitiveRange],
    name: str,
    count: int,
    components: int,
) -> tuple[FloatArray, FloatArray]:
    """Return one field-owned local range after structural validation."""

    value = ranges.get(name)
    if value is None:
        raise PhysicsEvaluationError(f"local primitive range {name!r} is missing")
    lower = np.asarray(value.lower, dtype=np.float64)
    upper = np.asarray(value.upper, dtype=np.float64)
    if lower.shape != (count, components) or upper.shape != (count, components):
        raise PhysicsEvaluationError(f"local primitive range {name!r} has an invalid shape")
    if not bool(np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise PhysicsEvaluationError(f"local primitive range {name!r} is not finite")
    if bool((lower > upper).any()):
        raise PhysicsEvaluationError(f"local primitive range {name!r} is reversed")
    return lower, upper


def _local_positive_scalar_range(
    ranges: Mapping[str, LocalPrimitiveRange],
    name: str,
    count: int,
) -> tuple[FloatArray, FloatArray]:
    lower, upper = _local_range(ranges, name, count, 1)
    if bool((lower[:, 0] <= 0.0).any()):
        raise PhysicsEvaluationError(f"local primitive range {name!r} must be positive")
    return lower[:, 0], upper[:, 0]


def _relative_speed_upper(
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    target_lower_m_s: FloatArray,
    target_upper_m_s: FloatArray,
) -> tuple[FloatArray, BoolArray]:
    """Bound ``|target - velocity|`` without losing co-flow cancellation."""

    count = int(velocity_lower_m_s.shape[0])
    if target_lower_m_s.shape != (count, 2) or target_upper_m_s.shape != (count, 2):
        raise PhysicsEvaluationError("local relative-velocity range has an invalid shape")
    with np.errstate(over="ignore", invalid="ignore"):
        component = np.nextafter(
            np.maximum(
                np.abs(velocity_lower_m_s - target_upper_m_s),
                np.abs(velocity_upper_m_s - target_lower_m_s),
            ),
            np.inf,
        )
        speed = _local_nonnegative_upper(np.hypot(component[:, 0], component[:, 1]))
    numerical_ok = np.isfinite(component).all(axis=1) & np.isfinite(speed)
    return speed, numerical_ok


def _local_positive_lower(value: FloatArray) -> FloatArray:
    """Round a positive compound expression toward a conservative lower bound."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.nextafter(value / _LOCAL_BOUND_ROUNDOFF_FACTOR, 0.0)


def _local_nonnegative_upper(value: FloatArray) -> FloatArray:
    """Round a nonnegative compound expression toward a conservative upper bound."""

    with np.errstate(over="ignore", invalid="ignore"):
        return np.nextafter(value * _LOCAL_BOUND_ROUNDOFF_FACTOR, np.inf)


def _certificate_result(
    certified: BoolArray, numerical_ok: BoolArray
) -> tuple[BoolArray, UInt8Array]:
    status = np.full(certified.size, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    return certified & numerical_ok, status


def _local_drag_applicability(
    plan: DragPlan | None,
    drag_diameter_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    ranges: Mapping[str, LocalPrimitiveRange],
) -> tuple[BoolArray, UInt8Array] | None:
    """Certify the selected drag gate over signed local primitive ranges."""

    if plan is None:
        return None
    count = int(drag_diameter_m.size)
    gas_lower, gas_upper = _local_range(ranges, plan.gas_velocity_field, count, 2)
    relative_speed, numerical_ok = _relative_speed_upper(
        velocity_lower_m_s,
        velocity_upper_m_s,
        gas_lower,
        gas_upper,
    )
    radius = 0.5 * drag_diameter_m
    if isinstance(plan, EpsteinDragPlan | FiniteSpeedEpsteinDragPlan):
        temperature_lower, _ = _local_positive_scalar_range(
            ranges,
            plan.gas_temperature_field,
            count,
        )
        mean_free_path_lower, _ = _local_positive_scalar_range(
            ranges,
            plan.gas_mean_free_path_field,
            count,
        )
        thermal_factor = 8.0 / math.pi if isinstance(plan, EpsteinDragPlan) else 2.0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            thermal_speed_lower = _local_positive_lower(
                np.sqrt(
                    thermal_factor * BOLTZMANN_J_K * temperature_lower / plan.gas_molecular_mass_kg
                )
            )
            knudsen_lower = _local_positive_lower(mean_free_path_lower / radius)
            speed_ratio = _local_nonnegative_upper(relative_speed / thermal_speed_lower)
        numerical_ok &= np.isfinite(thermal_speed_lower) & (thermal_speed_lower > 0.0)
        numerical_ok &= np.isfinite(knudsen_lower) & np.isfinite(speed_ratio)
        certified = knudsen_lower >= EPSTEIN_MIN_LAMBDA_OVER_RADIUS
        certified &= speed_ratio <= plan.maximum_speed_ratio
        return _certificate_result(certified, numerical_ok)

    if not isinstance(plan, StokesCunninghamDragPlan):
        raise PhysicsEvaluationError("local drag applicability model is unresolved")
    density_lower, density_upper = _local_positive_scalar_range(
        ranges,
        plan.gas_density_field,
        count,
    )
    del density_lower
    viscosity_lower, _ = _local_positive_scalar_range(
        ranges,
        plan.gas_dynamic_viscosity_field,
        count,
    )
    mean_free_path_lower, mean_free_path_upper = _local_positive_scalar_range(
        ranges,
        plan.gas_mean_free_path_field,
        count,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        knudsen_lower = _local_positive_lower(2.0 * mean_free_path_lower / drag_diameter_m)
        knudsen_upper = _local_nonnegative_upper(2.0 * mean_free_path_upper / drag_diameter_m)
        reynolds_upper = _local_nonnegative_upper(
            density_upper * drag_diameter_m * relative_speed / viscosity_lower
        )
    numerical_ok &= np.isfinite(knudsen_lower) & np.isfinite(knudsen_upper)
    numerical_ok &= np.isfinite(reynolds_upper)
    certified = knudsen_lower >= STOKES_CUNNINGHAM_MIN_KNUDSEN_RADIUS
    certified &= knudsen_upper <= STOKES_CUNNINGHAM_MAX_KNUDSEN_RADIUS
    certified &= reynolds_upper <= STOKES_CUNNINGHAM_MAX_REYNOLDS
    return _certificate_result(certified, numerical_ok)


def _local_charge_applicability(
    plan: ChargePlan | None,
    bounds: _ChargeRuntimeBounds | None,
    electrostatic_radius_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    ranges: Mapping[str, LocalPrimitiveRange],
) -> tuple[BoolArray, UInt8Array] | None:
    """Certify continuous-charge path gates from local field ranges."""

    if plan is None:
        if bounds is not None:
            raise PhysicsEvaluationError("local charge bounds do not match the resolved model")
        return None
    count = int(electrostatic_radius_m.size)
    ion_lower, ion_upper = _local_range(
        ranges,
        plan.positive_ion_velocity_field,
        count,
        2,
    )
    relative_speed, numerical_ok = _relative_speed_upper(
        velocity_lower_m_s,
        velocity_upper_m_s,
        ion_lower,
        ion_upper,
    )
    if isinstance(plan, AggregateRelativeDriftChargePlan):
        if not isinstance(bounds, _AggregateChargeRuntimeBounds):
            raise PhysicsEvaluationError("local aggregate-charge bounds are unresolved")
        certified = relative_speed <= plan.maximum_relative_ion_speed_m_s
        if plan.has_negative_ion_current:
            negative_lower, negative_upper = _local_range(
                ranges,
                cast(str, plan.negative_ion_velocity_field),
                count,
                2,
            )
            negative_relative_speed, negative_numerical_ok = _relative_speed_upper(
                velocity_lower_m_s,
                velocity_upper_m_s,
                negative_lower,
                negative_upper,
            )
            certified &= negative_relative_speed <= plan.maximum_relative_ion_speed_m_s
            numerical_ok &= negative_numerical_ok
        return _certificate_result(certified, numerical_ok)
    if not isinstance(plan, PlasmaContinuousChargePlan) or not isinstance(
        bounds,
        _OmlRuntimeBounds,
    ):
        raise PhysicsEvaluationError("local OML charge bounds are unresolved")
    electron_density_lower, electron_density_upper = _local_positive_scalar_range(
        ranges,
        plan.electron_number_density_field,
        count,
    )
    ion_density_lower, ion_density_upper = _local_positive_scalar_range(
        ranges,
        plan.positive_ion_number_density_field,
        count,
    )
    electron_temperature_lower, _ = _local_positive_scalar_range(
        ranges,
        plan.electron_temperature_field,
        count,
    )
    ion_temperature_lower, _ = _local_positive_scalar_range(
        ranges,
        plan.positive_ion_temperature_field,
        count,
    )
    del electron_density_lower, ion_density_lower
    _, maximum_ion_drift_ratio = _charge_kernel_selection(plan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inverse_debye_square_upper = _local_nonnegative_upper(
            ELEMENTARY_CHARGE_C**2
            / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
            * (
                electron_density_upper / electron_temperature_lower
                + ion_density_upper / ion_temperature_lower
            )
        )
        debye_lower = _local_positive_lower(1.0 / np.sqrt(inverse_debye_square_upper))
        radius_ratio = _local_nonnegative_upper(electrostatic_radius_m / debye_lower)
        ion_thermal_speed_lower = _local_positive_lower(
            np.sqrt(
                8.0 * BOLTZMANN_J_K * ion_temperature_lower / (math.pi * plan.positive_ion_mass_kg)
            )
        )
        drift_ratio = _local_nonnegative_upper(relative_speed / ion_thermal_speed_lower)
    numerical_ok &= np.isfinite(debye_lower) & (debye_lower > 0.0)
    numerical_ok &= np.isfinite(radius_ratio)
    numerical_ok &= np.isfinite(ion_thermal_speed_lower) & (ion_thermal_speed_lower > 0.0)
    numerical_ok &= np.isfinite(drift_ratio)
    certified = radius_ratio <= OML_MAX_RADIUS_OVER_DEBYE
    certified &= drift_ratio <= maximum_ion_drift_ratio
    return _certificate_result(certified, numerical_ok)


def _local_thermophoresis_applicability(
    plan: WaldmannGallisThermophoresisPlan | None,
    drag_diameter_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    ranges: Mapping[str, LocalPrimitiveRange],
) -> tuple[BoolArray, UInt8Array] | None:
    if plan is None:
        return None
    count = int(drag_diameter_m.size)
    gas_lower, gas_upper = _local_range(ranges, plan.gas_velocity_field, count, 2)
    temperature_lower, _ = _local_positive_scalar_range(
        ranges,
        plan.gas_temperature_field,
        count,
    )
    mean_free_path_lower, _ = _local_positive_scalar_range(
        ranges,
        plan.gas_mean_free_path_field,
        count,
    )
    relative_speed, numerical_ok = _relative_speed_upper(
        velocity_lower_m_s,
        velocity_upper_m_s,
        gas_lower,
        gas_upper,
    )
    radius = 0.5 * drag_diameter_m
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        thermal_speed_lower = _local_positive_lower(
            np.sqrt(
                8.0 * BOLTZMANN_J_K * temperature_lower / (math.pi * plan.gas_molecular_mass_kg)
            )
        )
        knudsen_lower = _local_positive_lower(mean_free_path_lower / radius)
        speed_ratio = _local_nonnegative_upper(relative_speed / thermal_speed_lower)
    numerical_ok &= np.isfinite(thermal_speed_lower) & (thermal_speed_lower > 0.0)
    numerical_ok &= np.isfinite(knudsen_lower) & np.isfinite(speed_ratio)
    certified = knudsen_lower >= WALDMANN_GALLIS_MIN_MEAN_FREE_PATH_OVER_RADIUS
    certified &= speed_ratio <= plan.maximum_speed_ratio
    return _certificate_result(certified, numerical_ok)


def _local_ion_drag_applicability(
    plan: IonDragPlan | None,
    bounds: _IonDragRuntimeBounds | None,
    particle_index: Int64Array,
    electrostatic_radius_m: FloatArray,
    charge_number_lower: FloatArray,
    charge_number_upper: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    ranges: Mapping[str, LocalPrimitiveRange],
) -> tuple[BoolArray, UInt8Array] | None:
    """Certify the selected ion-drag model from local primitive ranges."""

    if plan is None:
        if bounds is not None:
            raise PhysicsEvaluationError("local ion-drag bounds do not match the model")
        return None
    if isinstance(plan, ElectricFieldDirectedImageIonDragPlan):
        if bounds is not None:
            raise PhysicsEvaluationError("local image ion-drag bounds are inconsistent")
        return None
    count = int(particle_index.size)
    ion_lower, ion_upper = _local_range(
        ranges,
        plan.positive_ion_velocity_field,
        count,
        2,
    )
    relative_speed, numerical_ok = _relative_speed_upper(
        velocity_lower_m_s,
        velocity_upper_m_s,
        ion_lower,
        ion_upper,
    )
    if isinstance(plan, RelativeFlowScreenedIonDragPlan):
        if not isinstance(bounds, _RelativeFlowScreenedIonDragBounds):
            raise PhysicsEvaluationError("local relative-flow ion-drag bounds are unresolved")
        certified = relative_speed <= plan.maximum_relative_ion_speed_m_s
        return _certificate_result(certified, numerical_ok)
    if not isinstance(plan, BarnesCollisionlessIonDragPlan) or not isinstance(
        bounds,
        _BarnesIonDragBounds,
    ):
        raise PhysicsEvaluationError("local Barnes ion-drag bounds are unresolved")
    ion_temperature_lower, ion_temperature_upper = _local_positive_scalar_range(
        ranges,
        plan.positive_ion_temperature_field,
        count,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        thermal_speed_lower = _local_positive_lower(
            np.sqrt(
                8.0 * BOLTZMANN_J_K * ion_temperature_lower / (math.pi * plan.positive_ion_mass_kg)
            )
        )
        drift_ratio = _local_nonnegative_upper(relative_speed / thermal_speed_lower)
    numerical_ok &= np.isfinite(thermal_speed_lower) & (thermal_speed_lower > 0.0)
    numerical_ok &= np.isfinite(drift_ratio)
    certified, static_numerical = _local_barnes_static_applicability(
        plan,
        electrostatic_radius_m,
        charge_number_lower,
        charge_number_upper,
        ion_temperature_lower,
        ion_temperature_upper,
        ranges,
    )
    numerical_ok &= static_numerical
    certified &= drift_ratio <= plan.maximum_ion_drift_ratio
    return _certificate_result(certified, numerical_ok)


def _local_barnes_static_applicability(
    plan: BarnesCollisionlessIonDragPlan,
    electrostatic_radius_m: FloatArray,
    charge_number_lower: FloatArray,
    charge_number_upper: FloatArray,
    ion_temperature_lower_K: FloatArray,
    ion_temperature_upper_K: FloatArray,
    ranges: Mapping[str, LocalPrimitiveRange],
) -> tuple[BoolArray, BoolArray]:
    """Reproduce the Barnes static gates with row-local field extrema."""

    count = int(electrostatic_radius_m.size)
    electron_density_lower, electron_density_upper = _local_positive_scalar_range(
        ranges,
        plan.electron_number_density_field,
        count,
    )
    ion_density_lower, ion_density_upper = _local_positive_scalar_range(
        ranges,
        plan.positive_ion_number_density_field,
        count,
    )
    electron_temperature_lower, electron_temperature_upper = _local_positive_scalar_range(
        ranges,
        plan.electron_temperature_field,
        count,
    )
    mean_free_path_lower, _ = _local_positive_scalar_range(
        ranges,
        plan.ion_neutral_mean_free_path_field,
        count,
    )
    ion_mass = plan.positive_ion_mass_kg
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inverse_debye_square_upper = _local_nonnegative_upper(
            ELEMENTARY_CHARGE_C**2
            / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
            * (
                electron_density_upper / electron_temperature_lower
                + ion_density_upper / ion_temperature_lower_K
            )
        )
        inverse_debye_square_lower = _local_positive_lower(
            ELEMENTARY_CHARGE_C**2
            / (VACUUM_PERMITTIVITY_F_M * BOLTZMANN_J_K)
            * (
                electron_density_lower / electron_temperature_upper
                + ion_density_lower / ion_temperature_upper_K
            )
        )
        debye_lower = _local_positive_lower(1.0 / np.sqrt(inverse_debye_square_upper))
        debye_upper = _local_nonnegative_upper(1.0 / np.sqrt(inverse_debye_square_lower))
        thermal_speed_lower = _local_positive_lower(
            np.sqrt(8.0 * BOLTZMANN_J_K * ion_temperature_lower_K / (math.pi * ion_mass)),
        )
        charge_abs_upper = np.maximum(np.abs(charge_number_lower), np.abs(charge_number_upper))
        orbital_upper = _local_nonnegative_upper(
            charge_abs_upper
            * ELEMENTARY_CHARGE_C**2
            / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * ion_mass * thermal_speed_lower**2)
        )
        capacitance_lower = _local_positive_lower(
            4.0
            * math.pi
            * VACUUM_PERMITTIVITY_F_M
            * electrostatic_radius_m
            * (1.0 + electrostatic_radius_m / debye_upper)
        )
        potential_abs_upper = _local_nonnegative_upper(
            charge_abs_upper * ELEMENTARY_CHARGE_C / capacitance_lower
        )
        collection_square_upper = _local_nonnegative_upper(
            electrostatic_radius_m**2
            * (
                1.0
                + 2.0
                * ELEMENTARY_CHARGE_C
                * potential_abs_upper
                / (ion_mass * thermal_speed_lower**2)
            )
        )
        collection_upper = _local_nonnegative_upper(np.sqrt(collection_square_upper))
        coulomb_ratio_lower = _local_positive_lower(
            _local_positive_lower(debye_lower**2)
            / _local_nonnegative_upper(collection_square_upper + orbital_upper**2)
        )
        coulomb_log_lower = np.nextafter(0.5 * np.log(coulomb_ratio_lower), -np.inf)
    numerical_ok = np.isfinite(debye_lower) & (debye_lower > 0.0)
    numerical_ok &= np.isfinite(debye_upper) & (debye_upper > 0.0)
    numerical_ok &= np.isfinite(thermal_speed_lower) & (thermal_speed_lower > 0.0)
    numerical_ok &= np.isfinite(orbital_upper) & (orbital_upper >= 0.0)
    numerical_ok &= np.isfinite(collection_upper) & (collection_upper >= 0.0)
    numerical_ok &= np.isfinite(coulomb_log_lower)
    certified = charge_number_upper <= 0.0
    certified &= (
        _local_nonnegative_upper(electrostatic_radius_m / debye_lower)
        <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    )
    certified &= (
        _local_nonnegative_upper(orbital_upper / debye_lower) <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    )
    certified &= (
        _local_nonnegative_upper(collection_upper / debye_lower) <= ION_DRAG_MAX_SCALE_OVER_DEBYE
    )
    certified &= (
        _local_positive_lower(mean_free_path_lower / debye_upper)
        >= ION_DRAG_MIN_MEAN_FREE_PATH_OVER_DEBYE
    )
    certified &= coulomb_log_lower > 0.0
    return certified & numerical_ok, numerical_ok


def _local_lift_applicability(
    plan: RarefiedVorticityLiftPlan | None,
    drag_diameter_m: FloatArray,
    ranges: Mapping[str, LocalPrimitiveRange],
) -> tuple[BoolArray, UInt8Array] | None:
    if plan is None:
        return None
    count = int(drag_diameter_m.size)
    mean_free_path_lower, _ = _local_positive_scalar_range(
        ranges,
        plan.gas_mean_free_path_field,
        count,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        knudsen_lower = _local_positive_lower(mean_free_path_lower / (0.5 * drag_diameter_m))
    numerical_ok = np.isfinite(knudsen_lower)
    certified = knudsen_lower >= RAREFIED_VORTICITY_LIFT_MIN_MEAN_FREE_PATH_OVER_RADIUS
    return _certificate_result(certified, numerical_ok)


def _epstein_continuous_applicability(
    plan: DragPlan | None,
    bounds: _DragBounds | None,
    drag_diameter_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
) -> tuple[BoolArray, UInt8Array] | None:
    """Return the selected Epstein revision's common free-molecular gate."""

    if not isinstance(bounds, _EpsteinBounds):
        return None
    if isinstance(plan, EpsteinDragPlan):
        return epstein_continuous_applicability_batch(
            drag_diameter_m=drag_diameter_m,
            velocity_abs_upper_m_s=velocity_abs_upper_m_s,
            gas_velocity_abs_upper_m_s=bounds.target_velocity_abs_upper_m_s,
            gas_temperature_lower_K=bounds.temperature_lower_K,
            gas_mean_free_path_lower_m=bounds.mean_free_path_lower_m,
            gas_molecular_mass_kg=plan.gas_molecular_mass_kg,
            maximum_speed_ratio=plan.maximum_speed_ratio,
        )
    if isinstance(plan, FiniteSpeedEpsteinDragPlan):
        return epstein_finite_speed_continuous_applicability_batch(
            drag_diameter_m=drag_diameter_m,
            velocity_abs_upper_m_s=velocity_abs_upper_m_s,
            gas_velocity_abs_upper_m_s=bounds.target_velocity_abs_upper_m_s,
            gas_temperature_lower_K=bounds.temperature_lower_K,
            gas_mean_free_path_lower_m=bounds.mean_free_path_lower_m,
            gas_molecular_mass_kg=plan.gas_molecular_mass_kg,
            maximum_speed_ratio=plan.maximum_speed_ratio,
        )
    return None


def _charge_continuous_applicability(
    plan: ChargePlan | None,
    bounds: _ChargeRuntimeBounds | None,
    electrostatic_radius_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
) -> tuple[BoolArray, UInt8Array] | None:
    """Return the selected charge model's path gate, if charge evolves."""

    if plan is None:
        if bounds is not None:
            raise PhysicsEvaluationError("continuous charge bounds do not match the resolved model")
        return None
    if bounds is None:
        raise PhysicsEvaluationError("continuous charge bounds were not prepared")
    if isinstance(plan, AggregateRelativeDriftChargePlan) and isinstance(
        bounds, _AggregateChargeRuntimeBounds
    ):
        positive_result = _aggregate_charge_continuous_applicability_batch(
            velocity_abs_upper_m_s=velocity_abs_upper_m_s,
            ion_velocity_abs_upper_m_s=bounds.positive_ion_velocity_abs_upper_m_s,
            maximum_relative_ion_speed_m_s=plan.maximum_relative_ion_speed_m_s,
        )
        if not plan.has_negative_ion_current:
            return positive_result
        negative_result = _aggregate_charge_continuous_applicability_batch(
            velocity_abs_upper_m_s=velocity_abs_upper_m_s,
            ion_velocity_abs_upper_m_s=bounds.negative_ion_velocity_abs_upper_m_s,
            maximum_relative_ion_speed_m_s=plan.maximum_relative_ion_speed_m_s,
        )
        applicable, status = positive_result
        negative_applicable, negative_status = negative_result
        applicable &= negative_applicable
        status[negative_status != CONTINUOUS_APPLICABILITY_OK] = (
            CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        )
        return applicable, status
    if not isinstance(plan, PlasmaContinuousChargePlan) or not isinstance(
        bounds, _OmlRuntimeBounds
    ):
        raise PhysicsEvaluationError("continuous charge bounds do not match the resolved revision")
    _, maximum_ion_drift_ratio = _charge_kernel_selection(plan)
    return _oml_continuous_applicability_batch(
        electrostatic_radius_m=electrostatic_radius_m,
        velocity_abs_upper_m_s=velocity_abs_upper_m_s,
        positive_ion_velocity_abs_upper_m_s=bounds.positive_ion_velocity_abs_upper_m_s,
        positive_ion_temperature_lower_K=bounds.positive_ion_temperature_lower_K,
        positive_ion_mass_kg=plan.positive_ion_mass_kg,
        debye_length_lower_m=bounds.model.debye_length_lower_m,
        maximum_ion_drift_ratio=maximum_ion_drift_ratio,
    )


def _thermophoresis_continuous_applicability(
    plan: WaldmannGallisThermophoresisPlan | None,
    bounds: _WaldmannGallisBounds | None,
    particle_index: Int64Array,
    velocity_abs_upper_m_s: FloatArray,
) -> tuple[BoolArray, UInt8Array] | None:
    """Return the free-molecular size and low-drift path gate."""

    if plan is None:
        if bounds is not None:
            raise PhysicsEvaluationError(
                "thermophoresis applicability bounds do not match the resolved model"
            )
        return None
    if bounds is None:
        raise PhysicsEvaluationError("thermophoresis applicability bounds were not prepared")
    return waldmann_gallis_continuous_applicability_batch(
        static_applicable=bounds.static_applicable[particle_index],
        velocity_abs_upper_m_s=velocity_abs_upper_m_s,
        gas_velocity_abs_upper_m_s=bounds.gas_velocity_abs_upper_m_s,
        gas_temperature_lower_K=bounds.gas_temperature_lower_K,
        gas_molecular_mass_kg=plan.gas_molecular_mass_kg,
        maximum_speed_ratio=plan.maximum_speed_ratio,
    )


def _ion_drag_continuous_applicability(
    plan: IonDragPlan | None,
    bounds: _IonDragRuntimeBounds | None,
    particle_index: Int64Array,
    velocity_abs_upper_m_s: FloatArray,
) -> tuple[BoolArray, UInt8Array] | None:
    """Return the Barnes path gate while enforcing plan/bound ownership."""

    if plan is None:
        if bounds is not None:
            raise PhysicsEvaluationError(
                "ion-drag applicability bounds do not match the resolved model"
            )
        return None
    if isinstance(plan, ElectricFieldDirectedImageIonDragPlan):
        if bounds is not None:
            raise PhysicsEvaluationError(
                "image ion-drag applicability bounds do not match the resolved model"
            )
        return None
    if bounds is None:
        raise PhysicsEvaluationError("ion-drag applicability bounds were not prepared")
    if isinstance(plan, BarnesCollisionlessIonDragPlan) and isinstance(
        bounds, _BarnesIonDragBounds
    ):
        return barnes_collisionless_continuous_applicability_batch(
            static_applicable=bounds.static_applicable[particle_index],
            velocity_abs_upper_m_s=velocity_abs_upper_m_s,
            positive_ion_velocity_abs_upper_m_s=bounds.positive_ion_velocity_abs_upper_m_s,
            positive_ion_temperature_lower_K=bounds.positive_ion_temperature_lower_K,
            positive_ion_mass_kg=plan.positive_ion_mass_kg,
            maximum_ion_drift_ratio=plan.maximum_ion_drift_ratio,
        )
    if isinstance(plan, RelativeFlowScreenedIonDragPlan) and isinstance(
        bounds, _RelativeFlowScreenedIonDragBounds
    ):
        return relative_flow_screened_continuous_applicability_batch(
            velocity_abs_upper_m_s=velocity_abs_upper_m_s,
            positive_ion_velocity_abs_upper_m_s=bounds.positive_ion_velocity_abs_upper_m_s,
            maximum_relative_ion_speed_m_s=plan.maximum_relative_ion_speed_m_s,
        )
    raise PhysicsEvaluationError("ion-drag applicability bounds do not match the revision")


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
    )


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
) -> tuple[_WaldmannGallisBounds | None, FloatArray | None]:
    thermophoresis = plan.thermophoresis
    if thermophoresis is None:
        return None, None
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
    lift: RarefiedVorticityLiftPlan | None,
    mass_kg: FloatArray,
    drag_diameter_m: FloatArray,
    ranges: Mapping[str, PrimitiveRange],
) -> _RarefiedVorticityLiftBounds | None:
    if lift is None:
        return None
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


def _aggregate_charge_continuous_applicability_batch(
    *,
    velocity_abs_upper_m_s: FloatArray,
    ion_velocity_abs_upper_m_s: FloatArray,
    maximum_relative_ion_speed_m_s: float,
) -> tuple[BoolArray, UInt8Array]:
    """Certify the aggregate model's declared relative-speed envelope."""

    velocity = np.asarray(velocity_abs_upper_m_s, dtype=np.float64)
    ion_velocity = np.asarray(ion_velocity_abs_upper_m_s, dtype=np.float64)
    if (
        velocity.ndim != 2
        or velocity.shape[1] != 2
        or ion_velocity.shape != (2,)
        or not math.isfinite(maximum_relative_ion_speed_m_s)
        or maximum_relative_ion_speed_m_s <= 0.0
    ):
        raise PhysicsEvaluationError("aggregate charge applicability bounds are invalid")
    count = int(velocity.shape[0])
    status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    with np.errstate(over="ignore", invalid="ignore"):
        relative_component = np.nextafter(velocity + ion_velocity[None, :], np.inf)
        relative_speed = np.nextafter(
            np.hypot(relative_component[:, 0], relative_component[:, 1]),
            np.inf,
        )
    numerical_ok = (
        np.isfinite(velocity).all(axis=1)
        & (velocity >= 0.0).all(axis=1)
        & np.isfinite(ion_velocity).all()
        & (ion_velocity >= 0.0).all()
        & np.isfinite(relative_speed)
    )
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    return numerical_ok & (relative_speed <= maximum_relative_ion_speed_m_s), status


def _oml_continuous_applicability_batch(
    *,
    electrostatic_radius_m: FloatArray,
    velocity_abs_upper_m_s: FloatArray,
    positive_ion_velocity_abs_upper_m_s: FloatArray,
    positive_ion_temperature_lower_K: float,
    positive_ion_mass_kg: float,
    debye_length_lower_m: float,
    maximum_ion_drift_ratio: float,
) -> tuple[BoolArray, UInt8Array]:
    """Certify the OML revision gates over one velocity/path box."""

    count = int(electrostatic_radius_m.size)
    velocity = np.asarray(velocity_abs_upper_m_s, dtype=np.float64)
    ion_velocity = np.asarray(positive_ion_velocity_abs_upper_m_s, dtype=np.float64)
    status = np.full(count, CONTINUOUS_APPLICABILITY_OK, dtype=np.uint8)
    valid_input = (
        velocity.shape == (count, 2)
        and ion_velocity.shape == (2,)
        and math.isfinite(positive_ion_temperature_lower_K)
        and positive_ion_temperature_lower_K > 0.0
        and math.isfinite(positive_ion_mass_kg)
        and positive_ion_mass_kg > 0.0
        and math.isfinite(debye_length_lower_m)
        and debye_length_lower_m > 0.0
        and math.isfinite(maximum_ion_drift_ratio)
        and maximum_ion_drift_ratio > 0.0
    )
    if not valid_input:
        raise PhysicsEvaluationError("continuous OML applicability bounds are invalid")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        radius_over_debye = np.nextafter(
            electrostatic_radius_m / debye_length_lower_m,
            np.inf,
        )
        relative_component = np.nextafter(velocity + ion_velocity[None, :], np.inf)
        relative_speed = np.nextafter(
            np.hypot(relative_component[:, 0], relative_component[:, 1]),
            np.inf,
        )
        ion_mean_thermal_speed_lower = math.nextafter(
            math.sqrt(
                8.0
                * BOLTZMANN_J_K
                * positive_ion_temperature_lower_K
                / (math.pi * positive_ion_mass_kg)
            ),
            0.0,
        )
        drift_ratio = np.nextafter(relative_speed / ion_mean_thermal_speed_lower, np.inf)
    numerical_ok = (
        np.isfinite(electrostatic_radius_m)
        & (electrostatic_radius_m > 0.0)
        & np.isfinite(velocity).all(axis=1)
        & (velocity >= 0.0).all(axis=1)
        & np.isfinite(radius_over_debye)
        & np.isfinite(drift_ratio)
    )
    status[~numerical_ok] = CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    applicable = numerical_ok & (radius_over_debye <= OML_MAX_RADIUS_OVER_DEBYE)
    applicable &= drift_ratio <= maximum_ion_drift_ratio
    return applicable, status


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


def _sampled_positive_scalar_batch(
    values: Mapping[str, FloatArray],
    name: str,
    count: int,
    status: UInt8Array,
) -> FloatArray:
    value = values.get(name)
    if value is None or value.shape != (count, 1):
        raise PhysicsEvaluationError(f"sampled scalar field {name!r} has an invalid shape")
    result = np.asarray(value[:, 0], dtype=np.float64).copy()
    _mark_physics_failure(status, ~np.isfinite(result) | (result <= 0.0))
    result[status != NUMERICAL_STATUS_OK] = 1.0
    return result


def _sampled_nonnegative_scalar_batch(
    values: Mapping[str, FloatArray],
    name: str,
    count: int,
    status: UInt8Array,
) -> FloatArray:
    value = values.get(name)
    if value is None or value.shape != (count, 1):
        raise PhysicsEvaluationError(f"sampled scalar field {name!r} has an invalid shape")
    result = np.asarray(value[:, 0], dtype=np.float64).copy()
    _mark_physics_failure(status, ~np.isfinite(result) | (result < 0.0))
    result[status != NUMERICAL_STATUS_OK] = 0.0
    return result


def _sampled_scalar_batch(
    values: Mapping[str, FloatArray],
    name: str,
    count: int,
    status: UInt8Array,
) -> FloatArray:
    value = values.get(name)
    if value is None or value.shape != (count, 1):
        raise PhysicsEvaluationError(f"sampled scalar field {name!r} has an invalid shape")
    result = np.asarray(value[:, 0], dtype=np.float64).copy()
    _mark_physics_failure(status, ~np.isfinite(result))
    result[status != NUMERICAL_STATUS_OK] = 0.0
    return result


def _sampled_vector_batch(
    values: Mapping[str, FloatArray],
    name: str,
    count: int,
    status: UInt8Array,
) -> FloatArray:
    value = values.get(name)
    if value is None or value.shape != (count, 2):
        raise PhysicsEvaluationError(f"sampled vector field {name!r} has an invalid shape")
    result = np.asarray(value, dtype=np.float64).copy()
    _mark_physics_failure(status, ~np.isfinite(result).all(axis=1))
    result[status != NUMERICAL_STATUS_OK] = 0.0
    return result


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
