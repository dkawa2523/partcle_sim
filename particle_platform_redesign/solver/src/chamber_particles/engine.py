"""Preparation and the single production macro-step loop."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from os import PathLike

import numpy as np

from .boundaries import (
    BOUNDARY_ALGORITHM_REVISION,
    BOUNDARY_OUTCOME_ESCAPED,
    BOUNDARY_OUTCOME_HELD,
    BOUNDARY_OUTCOME_STUCK,
    BOUNDARY_STATUS_OK,
    BoundaryLawError,
    BoundaryResponseBatch,
    BoundaryRule,
    PreparedBoundaryRules,
    prepare_boundary_rule,
    prepare_boundary_rules,
    resolve_boundary_responses_batch,
)
from .case import CASE_FORMAT_VERSION, SimulationCase, TimeSpec
from .case_format import resident_array_bytes
from .coordinates import (
    canonicalize_rz_enclosure,
    fold_rz_position_vector,
    rz_canonical_vector_to_signed,
    rz_signed_stage_to_canonical,
)
from .cpu import (
    CPU_RUNTIME_LAYOUT_REVISION,
    MEMORY_PLAN_REVISION,
    ActiveIndex,
    CpuMemoryPlan,
    early_memory_requirement_bytes,
    plan_cpu_memory,
)
from .events import (
    CURVED_STATUS_AXIS,
    CURVED_STATUS_CLEAR,
    CURVED_STATUS_FAILURE,
    CURVED_STATUS_SPLIT,
    CURVED_STATUS_WALL,
    EVENT_ALGORITHM_REVISION,
    EXACT_PATH_LINEAR,
    EXACT_PATH_QUADRATIC,
    EXACT_STATUS_AXIS,
    EXACT_STATUS_CLEAR,
    EXACT_STATUS_FAILURE,
    EXACT_STATUS_WALL,
    SURFACE_ACTION_CURVED_DEPARTURE,
    SURFACE_ACTION_EXACT_DEPARTURE,
    SURFACE_ACTION_RESOLVED,
    SURFACE_ACTION_RESPONSE_ACCELERATION,
    SURFACE_ACTION_RESPONSE_VELOCITY,
    SURFACE_STATE_DEPARTURE,
    SURFACE_STATE_PENDING,
    SURFACE_STATE_RESOLVED,
    SURFACE_STATUS_OK,
    BoundaryHit,
    CurvedEventBatch,
    EventLocationError,
    ExactEventBatch,
    SurfaceReleaseBatch,
    classify_event_point,
    classify_surface_release_batch,
    count_curved_event_candidates,
    count_exact_event_candidates,
    locate_curved_first_event_batch,
    locate_exact_first_event_batch,
)
from .fields import (
    FIELD_LOCATION_REVISION,
    LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW,
    REQUIRED_FIELD_REVISION,
    FieldBatch,
    FieldLocationError,
    FieldWorkspace,
    PreparedFieldSet,
    RequiredFieldMetadata,
    prepare_required_fields,
)
from .geometry import (
    GEOMETRY_ALGORITHM_REVISION,
    GeometryPreparationError,
    PreparedGeometry,
    points_inside_volume,
    prepare_geometry,
)
from .integrators import (
    EXPONENTIAL_MIDPOINT_ENCLOSURE_REVISION,
    EXPONENTIAL_MIDPOINT_REVISION,
    RK4_DENSE_PATH_REVISION,
    RK4_ENCLOSURE_REVISION,
    STEP_PROPOSAL_REVISION,
    CurvedPathEnclosure,
    DynamicsEvaluation,
    ProposalSample,
    RelaxationEvaluation,
    Rk4DenseEnclosure,
    StepProposal,
    affine_exponential_charge,
    cubic_hermite_step,
    curved_chord_deviation_bounds,
    enclose_exponential_midpoint_path,
    enclose_exponential_midpoint_path_from_stage_bounds,
    enclose_rk4_path,
    exponential_frozen_start_predictor,
    exponential_frozen_start_predictor_enclosure,
    exponential_midpoint_step,
    restrict_cubic_hermite_proposal,
    rk4_step,
)
from .numerical_status import (
    FIELD_NUMERICAL_FAILURE,
    INTEGRATOR_ACCURACY_FAILURE,
    INTEGRATOR_NUMERICAL_FAILURE,
    NUMERICAL_STATUS_OK,
    PHYSICS_NUMERICAL_FAILURE,
)
from .output import (
    CHECKPOINT_SCHEMA_VERSION,
    RESULT_ALGORITHM_REVISION,
    RESULT_SCHEMA_VERSION,
    RESULT_WRITER_RESERVE_BYTES,
    BoundaryEvents,
    CheckpointState,
    FailureEvents,
    FinalParticles,
    LifecycleSeries,
    ProbeFrame,
    ReleaseEvents,
    ResultWriter,
    RunSummary,
    TrajectoryFrame,
)
from .physics.catalog import (
    BROWNIAN_FROZEN_START_REVISION,
    BROWNIAN_RZ_MIDPOINT_REVISION,
    PHYSICS_CATALOG_REVISION,
    PhysicsConfigurationError,
    PhysicsPlan,
    resolve_physics_plan,
)
from .physics.forces import (
    CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE,
    CONTINUOUS_APPLICABILITY_OK,
    PhysicsEvaluationError,
)
from .physics.runtime import (
    PHYSICS_RUNTIME_REVISION,
    LocalPrimitiveRange,
    PhysicsRuntime,
    PhysicsRuntimeEvaluation,
    PhysicsRuntimeWorkspace,
    PrimitiveRange,
    prepare_physics_runtime,
)
from .rng import (
    BROWNIAN_RNG_REVISION,
    BROWNIAN_ROOT_NORMAL_STREAM,
    BROWNIAN_SPLIT_NORMAL_STREAM,
    RNG_ALGORITHM_REVISION,
    SOURCE_FACET_DRAW,
    SOURCE_POSITION_DRAW,
    WALL_PROBABILISTIC_STICK_STREAM,
    brownian_normal_pair_batch,
    wall_uniform_batch,
)
from .sources import (
    SOURCE_ALGORITHM_REVISION,
    ParticleSchedule,
    realize_sources,
    source_particle_count,
)
from .stochastic import (
    JOINT_OU_REVISION,
    JOINT_OU_SPLIT_REVISION,
    MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT,
    JointOuIncrement,
    advance_joint_ou_with_increment,
    joint_ou_increment,
    split_joint_ou_increment_half,
)

type _PathInput = str | PathLike[str]

ENGINE_ALGORITHM_REVISION = "particle_engine_v37"
COMPILED_CPU_TILE_REVISION = "compiled_cpu_tile_v18"
DURABLE_COMMIT_CADENCE_REVISION = "cumulative_solver_work_v1"

_PREPARE_SCAN_BATCH_SIZE = 65_536
_TIME_GRID_ROUNDOFF_ULPS = 8.0
_DURABLE_COMMIT_MINIMUM_WORK = 1 << 20
_DURABLE_COMMIT_WORK_PER_PARTICLE = 128
# Numeric event staging plus the canonical payload simultaneously handed to
# the synchronous writer: raw row 134 B + stable order 8 B + payload 348 B.
# The pack-only int64 row gather is intentionally owned by the memory plan's
# 12.5% safety margin instead of being retained as another named component.
_EVENT_STAGING_BYTES_PER_ROW = 490
# Raw and canonical candidate columns coexist while the writer consumes one
# wave; each has int64 entries and each ragged offset table has one sentinel.
_EVENT_STAGING_BYTES_PER_CANDIDATE = 16
_EVENT_STAGING_FIXED_BYTES = 16
# A committed failure deactivates its particle and removes its residual stack,
# so one slab retains at most one failure record per resident particle.  The
# raw columns (22 B), stable order (8 B), and writer payload (22 B) total 52 B.
# Its pack-only int64 particle-ID gather is likewise safety-margin-owned.
_FAILURE_STAGING_BYTES_PER_PARTICLE = 52

_LIFECYCLE_PENDING = np.uint8(0)
_LIFECYCLE_ACTIVE = np.uint8(1)
_LIFECYCLE_STUCK = np.uint8(2)
_LIFECYCLE_ESCAPED = np.uint8(3)
_LIFECYCLE_FAILED = np.uint8(4)
_LIFECYCLE_HELD = np.uint8(5)

_FAILURE_NUMERICAL_EVENT_BUDGET = np.uint16(1)
_FAILURE_INDETERMINATE_EVENT = np.uint16(2)
_FAILURE_INDETERMINATE_BOUNDARY_POLICY = np.uint16(3)
_FAILURE_INDETERMINATE_SURFACE_DEPARTURE = np.uint16(4)
_FAILURE_FIELD_SUPPORT = np.uint16(5)
_FAILURE_MODEL_APPLICABILITY = np.uint16(6)
_FAILURE_NONFINITE_PHYSICS = np.uint16(7)
_FAILURE_INTEGRATOR_ACCURACY = np.uint16(8)
_FAILURE_INDETERMINATE_APPLICABILITY_CERTIFICATE = np.uint16(9)
_FAILURE_REASON_NAMES = {
    int(_FAILURE_NUMERICAL_EVENT_BUDGET): "numerical_event_budget",
    int(_FAILURE_INDETERMINATE_EVENT): "indeterminate_event",
    int(_FAILURE_INDETERMINATE_BOUNDARY_POLICY): "indeterminate_boundary_policy",
    int(_FAILURE_INDETERMINATE_SURFACE_DEPARTURE): "indeterminate_surface_departure",
    int(_FAILURE_FIELD_SUPPORT): "field_support",
    int(_FAILURE_MODEL_APPLICABILITY): "model_applicability",
    int(_FAILURE_NONFINITE_PHYSICS): "nonfinite_physics",
    int(_FAILURE_INTEGRATOR_ACCURACY): "integrator_accuracy",
    int(_FAILURE_INDETERMINATE_APPLICABILITY_CERTIFICATE): (
        "indeterminate_applicability_certificate"
    ),
}

_BOUNDARY_LAW_OUTPUT = np.asarray(
    ("", "stick", "escape", "specular", "restitution", "probabilistic_stick", "hold"),
    dtype="<U32",
)
_BOUNDARY_OUTCOME_OUTPUT = np.asarray(
    ("", "stuck", "escaped", "reflected", "held"),
    dtype="<U16",
)


class EngineError(RuntimeError):
    """The requested case is outside the implemented production capability."""


@dataclass(slots=True)
class _ApplicabilityCertificateWorkspace:
    """Reusable depth-first interval stack for one immutable dense proposal."""

    start_time_s: np.ndarray
    target_time_s: np.ndarray
    stack_top: np.ndarray
    split_count: np.ndarray
    failure_code: np.ndarray

    @classmethod
    def allocate(
        cls,
        capacity: int,
        split_budget: int,
    ) -> _ApplicabilityCertificateWorkspace:
        stack_capacity = split_budget + 1
        return cls(
            np.empty((capacity, stack_capacity), dtype="<f8"),
            np.empty((capacity, stack_capacity), dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<u2"),
        )


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    case: SimulationCase
    schedule: ParticleSchedule
    frame_times_s: tuple[float, ...]
    probe_times_s: tuple[float, ...]
    probe_particle_index: np.ndarray
    geometry: PreparedGeometry
    boundary_rules: tuple[BoundaryRule, ...]
    compiled_boundary_rules: PreparedBoundaryRules
    physics: PhysicsPlan
    fields: PreparedFieldSet
    dynamics: _StageDynamics
    constant_acceleration_m_s2: np.ndarray | None
    maximum_dt_over_tau: float
    maximum_dt_charge_lipschitz: float
    memory_plan: CpuMemoryPlan
    event_candidate_capacity: int
    certificate_workspace: _ApplicabilityCertificateWorkspace


@dataclass(slots=True)
class _BoundaryEventBuffer:
    """Capacity-bounded numeric columns for one synchronous event wave."""

    particle_index: np.ndarray
    time_s: np.ndarray
    event_ordinal: np.ndarray
    primary_facet_id: np.ndarray
    position_m: np.ndarray
    normal: np.ndarray
    velocity_pre_m_s: np.ndarray
    velocity_post_m_s: np.ndarray
    charge_pre_number: np.ndarray
    law_code: np.ndarray
    outcome_code: np.ndarray
    localization_residual_m: np.ndarray
    position_budget_m: np.ndarray
    time_budget_s: np.ndarray
    candidate_offset: np.ndarray
    candidate_facet_id: np.ndarray
    arena_capacity: int
    row_count: int = 0
    candidate_count: int = 0


@dataclass(slots=True)
class _FailureEventBuffer:
    """At-most-one-per-particle failure columns retained for one slab."""

    particle_index: np.ndarray
    time_s: np.ndarray
    event_ordinal: np.ndarray
    reason_code: np.ndarray
    count: int = 0


@dataclass(slots=True)
class _ReplayBuffer:
    """Requested macro-time states held in numeric columnar storage."""

    time_s: np.ndarray
    particle_index: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    charge_number: np.ndarray
    presence: np.ndarray
    lifecycle: np.ndarray


@dataclass(frozen=True, slots=True)
class _ExactEndpointBatch:
    """Accepted exact-prefix endpoints indexed by slab-local row."""

    valid: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    charge_number: np.ndarray


@dataclass(frozen=True, slots=True)
class _LangevinCoefficients:
    """Frozen macro-root coefficients for the supported inertial OU model."""

    drag_rate_s_inv: np.ndarray
    equilibrium_velocity_m_s: np.ndarray
    additive_acceleration_m_s2: np.ndarray
    charge_affine_rate_number_s: np.ndarray
    charge_rate_derivative_s_inv: np.ndarray
    thermal_velocity_variance_m2_s2: np.ndarray
    support_inside: np.ndarray
    applicability_inside: np.ndarray
    numerical_status: np.ndarray
    field_cell_id: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _LangevinRoot:
    """One pre-RNG coefficient state and its certified root duration."""

    coefficients: _LangevinCoefficients
    duration_s: np.ndarray
    failure_reason_code: np.ndarray


@dataclass(frozen=True, slots=True)
class _LangevinRootBatch:
    """Validated row-aligned state consumed by one conditional OU tree."""

    particles: np.ndarray
    starts_s: np.ndarray
    duration_s: np.ndarray
    requested_duration_s: np.ndarray
    start_position_m: np.ndarray
    start_velocity_m_s: np.ndarray
    start_charge_number: np.ndarray
    coefficients: _LangevinCoefficients
    numerical_status: np.ndarray
    drag_rate_s_inv: np.ndarray
    thermal_velocity_variance_m2_s2: np.ndarray
    equilibrium_velocity_m_s: np.ndarray


@dataclass(slots=True)
class _CurvedWavefront:
    """One slab's curved residual state held only in numeric SoA columns."""

    particle_index: np.ndarray
    current_time_s: np.ndarray
    current_position_m: np.ndarray
    current_velocity_m_s: np.ndarray
    current_charge_number: np.ndarray
    root_interval_s: np.ndarray
    interaction_count: np.ndarray
    certify_departure: np.ndarray
    stack_target_s: np.ndarray
    stack_depth: np.ndarray
    stack_interaction_reset: np.ndarray
    stack_top: np.ndarray
    stochastic_restart_time_s: np.ndarray | None


class _ParticleFailure(RuntimeError):
    """A numerical ambiguity localized to one particle and accepted time."""

    def __init__(
        self,
        reason_code: np.uint16,
        time_s: float,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: float,
    ) -> None:
        super().__init__(_FAILURE_REASON_NAMES[int(reason_code)])
        self.reason_code = int(reason_code)
        self.time_s = float(time_s)
        self.position_m = np.asarray(position_m, dtype="<f8").copy()
        self.velocity_m_s = np.asarray(velocity_m_s, dtype="<f8").copy()
        self.charge_number = float(charge_number)


@dataclass(slots=True)
class _EventStatistics:
    """Small aggregate for the curved material-wall and RZ-axis path."""

    accepted_particle_pieces: int = 0
    candidate_queries: int = 0
    refinements: int = 0
    maximum_refinement_depth: int = 0
    wall_interactions: int = 0
    residual_splits: int = 0
    axis_crossings: int = 0

    def merge(self, other: _EventStatistics) -> None:
        """Merge one completed tile in deterministic tile order."""

        self.accepted_particle_pieces += other.accepted_particle_pieces
        self.candidate_queries += other.candidate_queries
        self.refinements += other.refinements
        self.maximum_refinement_depth = max(
            self.maximum_refinement_depth,
            other.maximum_refinement_depth,
        )
        self.wall_interactions += other.wall_interactions
        self.residual_splits += other.residual_splits
        self.axis_crossings += other.axis_crossings


@dataclass(slots=True)
class _RunState:
    """Mutable resident state at one accepted macro-step boundary."""

    position_m: np.ndarray
    velocity_m_s: np.ndarray
    charge_number: np.ndarray
    active: np.ndarray
    active_index: ActiveIndex
    lifecycle: np.ndarray
    failure_reason_code: np.ndarray
    terminal_time_s: np.ndarray
    event_ordinal: np.ndarray
    physical_boundary_event_ordinal: np.ndarray
    exact_origin_time_s: np.ndarray
    exact_origin_position_m: np.ndarray
    exact_origin_velocity_m_s: np.ndarray
    start_contact_state: np.ndarray
    release_cursor: int
    frame_cursor: int
    probe_cursor: int
    macro_step_count: int
    macro_start_s: float
    event_statistics: _EventStatistics


@dataclass(frozen=True, slots=True)
class _SlabResult:
    """Aggregate work returned after one bounded particle slab."""

    statistics: _EventStatistics


@dataclass(slots=True)
class _StageDynamics:
    """Compose one resolved physics plan over shared stage field samples."""

    runtime: PhysicsRuntime
    fields: PreparedFieldSet
    coordinate_system: str
    last_field_cell: np.ndarray | None
    field_workspaces: tuple[FieldWorkspace, ...]
    physics_workspaces: tuple[PhysicsRuntimeWorkspace, ...]
    physics_workspace_cursor: int = 0

    def __call__(
        self,
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        evaluation, sampled, radial_sign, numerical_status = self._evaluate_stage(
            particle_index,
            time_s,
            position_m,
            velocity_m_s,
            charge_number,
        )
        acceleration = (
            evaluation.acceleration_m_s2
            if radial_sign is None
            else rz_canonical_vector_to_signed(evaluation.acceleration_m_s2, radial_sign)
        )
        return DynamicsEvaluation(
            acceleration_m_s2=acceleration,
            charge_rate_number_s=evaluation.charge_rate_number_s,
            support_inside=sampled.support_inside,
            applicability_inside=evaluation.applicable,
            numerical_status=numerical_status,
            field_cell_id=sampled.cell_id,
        )

    def evaluate_relaxation(
        self,
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        """Return the owned linear-drag decomposition for exponential midpoint."""

        evaluation, sampled, radial_sign, numerical_status = self._evaluate_stage(
            particle_index,
            time_s,
            position_m,
            velocity_m_s,
            charge_number,
        )
        target_velocity = evaluation.target_velocity_m_s
        additive_acceleration = evaluation.additive_acceleration_m_s2
        if radial_sign is not None:
            target_velocity = rz_canonical_vector_to_signed(target_velocity, radial_sign)
            additive_acceleration = rz_canonical_vector_to_signed(
                additive_acceleration,
                radial_sign,
            )
        return RelaxationEvaluation(
            linear_drag_rate_s_inv=evaluation.linear_drag_rate_s_inv,
            target_velocity_m_s=target_velocity,
            additive_acceleration_m_s2=additive_acceleration,
            charge_rate_number_s=evaluation.charge_rate_number_s,
            charge_rate_derivative_s_inv=evaluation.charge_rate_derivative_s_inv,
            support_inside=sampled.support_inside,
            applicability_inside=evaluation.applicable,
            numerical_status=numerical_status,
            field_cell_id=sampled.cell_id,
        )

    def evaluate_langevin_coefficients(
        self,
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> _LangevinCoefficients:
        """Sample and freeze the one supported Brownian model at a macro root."""

        evaluation, sampled, radial_sign, numerical_status = self._evaluate_stage(
            particle_index,
            time_s,
            position_m,
            velocity_m_s,
            charge_number,
        )
        target_velocity = evaluation.target_velocity_m_s
        additive_acceleration = evaluation.additive_acceleration_m_s2
        if radial_sign is not None:
            target_velocity = rz_canonical_vector_to_signed(target_velocity, radial_sign)
            additive_acceleration = rz_canonical_vector_to_signed(
                additive_acceleration,
                radial_sign,
            )
        thermal, thermal_status = self.runtime.brownian_thermal_velocity_variance_batch(
            particle_index,
            sampled.values,
        )
        status = numerical_status.copy()
        thermal_failed = thermal_status != NUMERICAL_STATUS_OK
        status[(status == NUMERICAL_STATUS_OK) & thermal_failed] = PHYSICS_NUMERICAL_FAILURE
        invalid = ~np.isfinite(thermal) | (thermal <= 0.0)
        invalid |= ~np.isfinite(evaluation.linear_drag_rate_s_inv)
        invalid |= evaluation.linear_drag_rate_s_inv <= 0.0
        invalid |= ~np.isfinite(target_velocity).all(axis=1)
        invalid |= ~np.isfinite(additive_acceleration).all(axis=1)
        invalid |= ~np.isfinite(evaluation.charge_rate_number_s)
        invalid |= ~np.isfinite(evaluation.charge_rate_derivative_s_inv)
        invalid |= evaluation.charge_rate_derivative_s_inv > 0.0
        status[(status == NUMERICAL_STATUS_OK) & invalid] = PHYSICS_NUMERICAL_FAILURE
        return _LangevinCoefficients(
            drag_rate_s_inv=evaluation.linear_drag_rate_s_inv.copy(),
            equilibrium_velocity_m_s=target_velocity.copy(),
            additive_acceleration_m_s2=additive_acceleration.copy(),
            charge_affine_rate_number_s=evaluation.charge_rate_number_s.copy(),
            charge_rate_derivative_s_inv=evaluation.charge_rate_derivative_s_inv.copy(),
            thermal_velocity_variance_m2_s2=thermal.copy(),
            support_inside=sampled.support_inside.copy(),
            applicability_inside=evaluation.applicable.copy(),
            numerical_status=status,
            field_cell_id=sampled.cell_id.copy(),
        )

    def _evaluate_stage(
        self,
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> tuple[PhysicsRuntimeEvaluation, FieldBatch, np.ndarray | None, np.ndarray]:
        """Sample one stage once and retain the signed-chart transform."""

        count = int(particle_index.size)
        if time_s.shape != (count,):
            raise PhysicsEvaluationError("stage times must have shape [N]")
        radial_sign: np.ndarray | None = None
        stage_position_m = position_m
        stage_velocity_m_s = velocity_m_s
        if self.coordinate_system == "axisymmetric_rz":
            stage_position_m, stage_velocity_m_s, radial_sign = rz_signed_stage_to_canonical(
                position_m,
                velocity_m_s,
            )
        cell_hint = None if self.last_field_cell is None else self.last_field_cell[particle_index]
        slot = self.physics_workspace_cursor
        sampled, numerical_status = self.fields.sample_batch(
            stage_position_m,
            cell_hint=cell_hint,
            workspace=self.field_workspaces[slot],
        )
        workspace = self.physics_workspaces[slot]
        self.physics_workspace_cursor = (self.physics_workspace_cursor + 1) % len(
            self.physics_workspaces
        )
        evaluation, numerical_status = self.runtime.evaluate_batch(
            particle_index,
            stage_velocity_m_s,
            charge_number,
            sampled.values,
            numerical_status=numerical_status,
            workspace=workspace,
        )
        return evaluation, sampled, radial_sign, numerical_status

    def commit_field_cell(
        self,
        particle_index: np.ndarray,
        field_cell_id: np.ndarray | None,
    ) -> None:
        """Commit the search hint only after the corresponding state is accepted."""

        if self.last_field_cell is None or field_cell_id is None:
            return
        self.last_field_cell[particle_index] = field_cell_id

    def acceleration_abs_upper(
        self,
        particle_index: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound every configured acceleration component for an RK stage box."""

        return self.runtime.acceleration_abs_upper_batch(
            particle_index,
            velocity_abs_upper_m_s,
        )

    def continuous_applicability_batch(
        self,
        particle_index: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return path applicability plus one row-local numerical status column."""

        return self.runtime.continuous_applicability_batch(
            particle_index,
            velocity_abs_upper_m_s,
        )

    def global_continuous_applicability_batch(
        self,
        particle_index: np.ndarray,
        velocity_lower_m_s: np.ndarray,
        velocity_upper_m_s: np.ndarray,
        charge_lower_number: np.ndarray,
        charge_upper_number: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Try the prepared run-global applicability certificate cheaply."""

        velocity_abs_upper = np.maximum(
            np.abs(velocity_lower_m_s),
            np.abs(velocity_upper_m_s),
        )
        certified, status = self.runtime.continuous_applicability_batch(
            particle_index,
            velocity_abs_upper,
        )
        charge_certified, charge_status = self.runtime.charge_interval_inside_prepared_invariant(
            charge_lower_number,
            charge_upper_number,
        )
        certified &= charge_certified
        status[charge_status != CONTINUOUS_APPLICABILITY_OK] = (
            CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
        )
        certified[status != CONTINUOUS_APPLICABILITY_OK] = False
        return certified, status

    def prepared_charge_invariant_interval(
        self,
        reference_charge_number: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expose the runtime-owned invariant used by continuous path proofs."""

        return self.runtime.prepared_charge_invariant_interval(reference_charge_number)

    def local_continuous_applicability_batch(
        self,
        particle_index: np.ndarray,
        position_lower_m: np.ndarray,
        position_upper_m: np.ndarray,
        velocity_lower_m_s: np.ndarray,
        velocity_upper_m_s: np.ndarray,
        charge_lower_number: np.ndarray,
        charge_upper_number: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Certify one signed local path box using field-cell primitive ranges."""

        position_lower = np.asarray(position_lower_m, dtype="<f8")
        position_upper = np.asarray(position_upper_m, dtype="<f8")
        velocity_lower = np.asarray(velocity_lower_m_s, dtype="<f8")
        velocity_upper = np.asarray(velocity_upper_m_s, dtype="<f8")
        if self.coordinate_system == "axisymmetric_rz":
            signed_position_lower = position_lower
            signed_position_upper = position_upper
            position_lower, position_upper = canonicalize_rz_enclosure(
                signed_position_lower,
                signed_position_upper,
            )
            velocity_lower, velocity_upper = _canonicalize_rz_velocity_enclosure(
                signed_position_lower,
                signed_position_upper,
                velocity_lower,
                velocity_upper,
            )
        local_fields = self.fields.local_component_bounds(position_lower, position_upper)
        range_available = local_fields.range_available
        if not self.fields.fields:
            range_available = np.ones(particle_index.size, dtype=np.bool_)
        certified = np.zeros(particle_index.size, dtype=np.bool_)
        status = np.full(
            particle_index.size,
            CONTINUOUS_APPLICABILITY_OK,
            dtype=np.uint8,
        )
        available_rows = np.flatnonzero(range_available).astype("<i8", copy=False)
        if available_rows.size:
            primitive_ranges = {
                name: LocalPrimitiveRange(
                    local_fields.lower[name][available_rows],
                    local_fields.upper[name][available_rows],
                )
                for name in local_fields.lower
            }
            available_certified, available_status = (
                self.runtime.local_continuous_applicability_batch(
                    particle_index[available_rows],
                    velocity_lower[available_rows],
                    velocity_upper[available_rows],
                    charge_lower_number[available_rows],
                    charge_upper_number[available_rows],
                    primitive_ranges,
                )
            )
            certified[available_rows] = available_certified
            status[available_rows] = available_status
        return certified, status, range_available

    def local_additive_acceleration_abs_upper_batch(
        self,
        particle_index: np.ndarray,
        position_lower_m: np.ndarray,
        position_upper_m: np.ndarray,
        velocity_lower_m_s: np.ndarray,
        velocity_upper_m_s: np.ndarray,
        charge_lower_number: np.ndarray,
        charge_upper_number: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bound midpoint force over one local predictor enclosure."""

        position_lower = np.asarray(position_lower_m, dtype="<f8")
        position_upper = np.asarray(position_upper_m, dtype="<f8")
        velocity_lower = np.asarray(velocity_lower_m_s, dtype="<f8")
        velocity_upper = np.asarray(velocity_upper_m_s, dtype="<f8")
        if self.coordinate_system == "axisymmetric_rz":
            signed_position_lower = position_lower
            signed_position_upper = position_upper
            position_lower, position_upper = canonicalize_rz_enclosure(
                signed_position_lower,
                signed_position_upper,
            )
            velocity_lower, velocity_upper = _canonicalize_rz_velocity_enclosure(
                signed_position_lower,
                signed_position_upper,
                velocity_lower,
                velocity_upper,
            )
        local_fields = self.fields.local_component_bounds(position_lower, position_upper)
        range_available = local_fields.range_available
        count = int(particle_index.size)
        bound = np.zeros((count, 2), dtype="<f8")
        applicable = np.zeros(count, dtype=np.bool_)
        status = np.full(
            count,
            CONTINUOUS_APPLICABILITY_OK,
            dtype=np.uint8,
        )
        available_rows = np.flatnonzero(range_available).astype("<i8", copy=False)
        if available_rows.size:
            primitive_ranges = {
                name: LocalPrimitiveRange(
                    local_fields.lower[name][available_rows],
                    local_fields.upper[name][available_rows],
                )
                for name in local_fields.lower
            }
            local_bound, local_applicable, local_status = (
                self.runtime.local_additive_acceleration_abs_upper_batch(
                    particle_index[available_rows],
                    velocity_lower[available_rows],
                    velocity_upper[available_rows],
                    charge_lower_number[available_rows],
                    charge_upper_number[available_rows],
                    primitive_ranges,
                )
            )
            bound[available_rows] = local_bound
            applicable[available_rows] = local_applicable
            status[available_rows] = local_status
        return bound, applicable, status, range_available

    def linear_relaxation_abs_bounds(
        self,
        particle_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the prepared bounds consumed by the exponential enclosure."""

        bounds = self.runtime.linear_relaxation_abs_bounds(particle_index)
        return (
            bounds.rate_upper_s_inv,
            bounds.target_velocity_abs_upper_m_s,
        )

    def additive_acceleration_abs_upper(
        self,
        particle_index: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bound non-drag acceleration for an exponential stage velocity box."""

        return self.runtime.additive_acceleration_abs_upper_batch(
            particle_index,
            velocity_abs_upper_m_s,
        )


def _canonicalize_rz_velocity_enclosure(
    position_lower_m: np.ndarray,
    position_upper_m: np.ndarray,
    velocity_lower_m_s: np.ndarray,
    velocity_upper_m_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map signed-chart velocity boxes into the canonical radial basis."""

    lower = np.asarray(velocity_lower_m_s, dtype="<f8").copy()
    upper = np.asarray(velocity_upper_m_s, dtype="<f8").copy()
    radial_position_lower = np.asarray(position_lower_m, dtype="<f8")[:, 0]
    radial_position_upper = np.asarray(position_upper_m, dtype="<f8")[:, 0]
    negative = radial_position_upper < 0.0
    crossing = (radial_position_lower < 0.0) & ~negative
    negative_lower = -upper[negative, 0]
    negative_upper = -lower[negative, 0]
    lower[negative, 0] = negative_lower
    upper[negative, 0] = negative_upper
    radial_abs = np.maximum(np.abs(lower[crossing, 0]), np.abs(upper[crossing, 0]))
    lower[crossing, 0] = -radial_abs
    upper[crossing, 0] = radial_abs
    return lower, upper


def run_simulation(case: SimulationCase, output: _PathInput) -> RunSummary:
    """Run or resume the selected production profile and publish one complete result."""

    prepared = _prepare(case)
    schedule = prepared.schedule
    count = schedule.particle_count
    commit_work_threshold = _durable_commit_work_threshold(count)
    with ResultWriter(output, count, _resume_identity(prepared)) as writer:
        if writer.completed_summary is not None:
            return writer.completed_summary
        state = _restore_or_initialize_run_state(prepared, writer.resume_state)
        epoch_start_work = _durable_commit_work(state)
        epoch_open = False
        try:
            while state.macro_start_s < case.spec.time.end_s:
                if not epoch_open:
                    writer.begin_epoch()
                    epoch_open = True
                macro_start = state.macro_start_s
                macro_end = _macro_step_end(
                    case.spec.time,
                    state.macro_step_count,
                )
                if macro_end <= macro_start:
                    raise EngineError("macro-step time no longer advances at float64 precision")

                released, state.release_cursor = _activate_releases(
                    writer,
                    schedule,
                    state.release_cursor,
                    macro_end,
                    prepared.memory_plan.slab_particles,
                    state.active,
                    state.lifecycle,
                    state.event_ordinal,
                )
                state.active_index.add(released)

                frame_stop = _frame_stop(
                    prepared.frame_times_s,
                    state.frame_cursor,
                    macro_end,
                )
                probe_stop = _frame_stop(
                    prepared.probe_times_s,
                    state.probe_cursor,
                    macro_end,
                )
                frame_batch_times_s = prepared.frame_times_s[state.frame_cursor : frame_stop]
                probe_batch_times_s = prepared.probe_times_s[state.probe_cursor : probe_stop]
                replay_times_s = tuple(sorted(set(frame_batch_times_s) | set(probe_batch_times_s)))
                replay = _allocate_replay_buffer(
                    replay_times_s,
                    schedule,
                    prepared.probe_particle_index,
                    include_all_particles=bool(frame_batch_times_s),
                )
                _advance_active_slabs(
                    prepared,
                    state.active_index.rows,
                    state.macro_step_count,
                    macro_start,
                    macro_end,
                    state.position_m,
                    state.velocity_m_s,
                    state.charge_number,
                    state.active,
                    state.lifecycle,
                    state.terminal_time_s,
                    state.event_ordinal,
                    state.physical_boundary_event_ordinal,
                    state.exact_origin_time_s,
                    state.exact_origin_position_m,
                    state.exact_origin_velocity_m_s,
                    state.start_contact_state,
                    replay,
                    state.event_statistics,
                    state.failure_reason_code,
                    writer,
                )
                state.active_index.compact(
                    state.active,
                    prepared.memory_plan.slab_particles,
                )
                _finalize_replay_buffer(
                    replay,
                    schedule,
                    state.position_m,
                    state.velocity_m_s,
                    state.charge_number,
                    state.lifecycle,
                    state.terminal_time_s,
                )

                while state.frame_cursor < frame_stop:
                    writer.write_frame(
                        _frame_from_replay(
                            prepared.frame_times_s[state.frame_cursor],
                            schedule,
                            replay,
                        )
                    )
                    state.frame_cursor += 1

                while state.probe_cursor < probe_stop:
                    writer.write_probe(
                        _probe_from_replay(
                            prepared.probe_times_s[state.probe_cursor],
                            prepared.probe_particle_index,
                            schedule,
                            replay,
                        )
                    )
                    state.probe_cursor += 1

                writer.write_lifecycle_series(_lifecycle_series_at(macro_end, state.lifecycle))
                state.macro_start_s = macro_end
                state.macro_step_count += 1
                completed_work = _durable_commit_work(state)
                if (
                    completed_work - epoch_start_work >= commit_work_threshold
                    or macro_end >= case.spec.time.end_s
                ):
                    writer.commit_epoch(_checkpoint_state(prepared, state))
                    epoch_open = False
                    epoch_start_work = completed_work

            _validate_completed_run_state(prepared, state)
            final = _final_particles(
                schedule,
                state.position_m,
                state.velocity_m_s,
                state.charge_number,
                state.lifecycle,
                state.failure_reason_code,
                case.spec.time.end_s,
            )
            return writer.finalize(
                final,
                _manifest(
                    prepared,
                    state.lifecycle,
                    state.failure_reason_code,
                    state.event_statistics,
                ),
                macro_step_count=state.macro_step_count,
            )
        except ValueError as error:
            raise EngineError(str(error)) from error


def _durable_commit_work_threshold(particle_count: int) -> int:
    """Resolve the deterministic amount of solver work retained per epoch."""

    return max(
        _DURABLE_COMMIT_MINIMUM_WORK,
        _DURABLE_COMMIT_WORK_PER_PARTICLE * particle_count,
    )


def _durable_commit_work(state: _RunState) -> int:
    """Return cumulative solver work from counters already owned by checkpoints."""

    statistics = state.event_statistics
    return (
        state.macro_step_count
        + statistics.accepted_particle_pieces
        + statistics.candidate_queries
        + statistics.refinements
    )


def _durable_commit_cadence(particle_count: int) -> dict[str, object]:
    """Describe the resolved cadence shared by the manifest and resume identity."""

    return {
        "revision": DURABLE_COMMIT_CADENCE_REVISION,
        "work_threshold": _durable_commit_work_threshold(particle_count),
        "work_components": [
            "macro_step_count",
            "accepted_particle_pieces",
            "candidate_queries",
            "refinements",
        ],
        "barrier": "accepted_macro_step",
    }


def _macro_step_end(time: TimeSpec, completed_steps: int) -> float:
    """Return the indexed macro-grid boundary, snapping only roundoff to the end.

    Repeatedly adding ``dt_s`` can leave a one-ULP tail at an intended decimal
    endpoint (for example 0.02 s repeated ten times toward 0.2 s).  Such a tail
    is not a physical integration interval and may be too short to represent a
    midpoint.  The step index avoids accumulated drift; the final comparison
    covers only the float64 construction error of ``start + n * dt``.
    """

    step_number = completed_steps + 1
    nominal_offset_s, offset_residual_s = _time_grid_product(
        float(step_number),
        time.dt_s,
    )
    nominal_end_s = math.fsum((time.start_s, nominal_offset_s, offset_residual_s))
    if not math.isfinite(nominal_end_s) or nominal_end_s >= time.end_s:
        return time.end_s
    construction_roundoff_s = _TIME_GRID_ROUNDOFF_ULPS * max(
        math.ulp(time.start_s),
        math.ulp(time.end_s),
        math.ulp(time.dt_s) * step_number,
        math.ulp(nominal_offset_s),
        math.ulp(nominal_end_s),
    )
    if time.end_s - nominal_end_s <= construction_roundoff_s:
        return time.end_s
    return nominal_end_s


def _time_grid_product(left: float, right: float) -> tuple[float, float]:
    """Return a binary64 product and residual without splitter overflow."""

    product = left * right
    if not math.isfinite(product) or product == 0.0:
        return product, 0.0
    left_fraction, left_exponent = math.frexp(left)
    right_fraction, right_exponent = math.frexp(right)
    scaled_product = left_fraction * right_fraction
    splitter = 134_217_729.0
    split_left = splitter * left_fraction
    left_high = split_left - (split_left - left_fraction)
    left_low = left_fraction - left_high
    split_right = splitter * right_fraction
    right_high = split_right - (split_right - right_fraction)
    right_low = right_fraction - right_high
    scaled_residual = (
        ((left_high * right_high - scaled_product) + left_high * right_low) + left_low * right_high
    ) + left_low * right_low
    exponent = left_exponent + right_exponent
    return (
        math.ldexp(scaled_product, exponent),
        math.ldexp(scaled_residual, exponent),
    )


def _brownian_provenance(physics: PhysicsPlan) -> dict[str, object]:
    """Return the single manifest/checkpoint description of stochastic motion."""

    if physics.noise is None:
        return {
            "rng_revision": None,
            "ou_revision": None,
            "split_revision": None,
            "coefficient_policy": None,
            "composition_revision": None,
            "charge_dense_revision": None,
            "interval_tree_depth": None,
            "root_normal_stream": None,
            "split_normal_stream": None,
        }
    midpoint_revision = physics.noise.revision == BROWNIAN_RZ_MIDPOINT_REVISION
    return {
        "rng_revision": BROWNIAN_RNG_REVISION,
        "ou_revision": JOINT_OU_REVISION,
        "split_revision": JOINT_OU_SPLIT_REVISION,
        "coefficient_policy": (
            "macro_root_frozen_midpoint_v1" if midpoint_revision else "macro_root_frozen_start_v1"
        ),
        "composition_revision": (
            "stochastic_exponential_midpoint_v1" if midpoint_revision else "joint_ou_only_v1"
        ),
        "charge_dense_revision": (
            "macro_root_affine_exponential_v2" if midpoint_revision else "fixed_charge_v1"
        ),
        "interval_tree_depth": physics.noise.interval_tree_depth,
        "root_normal_stream": BROWNIAN_ROOT_NORMAL_STREAM,
        "split_normal_stream": BROWNIAN_SPLIT_NORMAL_STREAM,
    }


def _resume_identity(prepared: _PreparedRun) -> dict[str, object]:
    """Return the exact immutable identity accepted by checkpoint resume."""

    case = prepared.case
    integrator = case.spec.solver.integrator
    brownian = _brownian_provenance(prepared.physics)
    uses_rk4_dense = _uses_rk4_dense_certificate(
        case,
        prepared.physics,
        prepared.constant_acceleration_m_s2,
    )
    particle_bytes = memoryview(prepared.schedule.particle_id).cast("B")
    return {
        "case_file_hash": case.case_file_hash,
        "data_content_hash": case.content_hash,
        "case_schema_version": CASE_FORMAT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "result_algorithm_revision": RESULT_ALGORITHM_REVISION,
        "engine_algorithm_revision": ENGINE_ALGORITHM_REVISION,
        "durable_commit_cadence": _durable_commit_cadence(prepared.schedule.particle_count),
        "compiled_cpu_tile_revision": COMPILED_CPU_TILE_REVISION,
        "cpu_runtime_layout_revision": CPU_RUNTIME_LAYOUT_REVISION,
        "memory_plan_revision": MEMORY_PLAN_REVISION,
        "step_proposal_revision": STEP_PROPOSAL_REVISION,
        "rk4_enclosure_revision": RK4_ENCLOSURE_REVISION if uses_rk4_dense else None,
        "rk4_dense_path_revision": RK4_DENSE_PATH_REVISION if uses_rk4_dense else None,
        "exponential_midpoint_revision": (
            EXPONENTIAL_MIDPOINT_REVISION if integrator == "exponential_midpoint" else None
        ),
        "exponential_midpoint_enclosure_revision": (
            EXPONENTIAL_MIDPOINT_ENCLOSURE_REVISION
            if integrator == "exponential_midpoint"
            else None
        ),
        "physics_catalog_revision": PHYSICS_CATALOG_REVISION,
        "physics_runtime_revision": PHYSICS_RUNTIME_REVISION,
        "field_location_revision": FIELD_LOCATION_REVISION,
        "required_field_revision": REQUIRED_FIELD_REVISION,
        "geometry_algorithm_revision": GEOMETRY_ALGORITHM_REVISION,
        "event_algorithm_revision": EVENT_ALGORITHM_REVISION,
        "boundary_algorithm_revision": (
            BOUNDARY_ALGORITHM_REVISION if prepared.boundary_rules else None
        ),
        "source_algorithm_revision": SOURCE_ALGORITHM_REVISION,
        "rng_algorithm_revision": RNG_ALGORITHM_REVISION,
        "brownian_rng_revision": brownian["rng_revision"],
        "joint_ou_revision": brownian["ou_revision"],
        "joint_ou_split_revision": brownian["split_revision"],
        "brownian_coefficient_policy": brownian["coefficient_policy"],
        "brownian_composition_revision": brownian["composition_revision"],
        "brownian_charge_dense_revision": brownian["charge_dense_revision"],
        "brownian_interval_tree_depth": brownian["interval_tree_depth"],
        "coordinate_system": case.data.coordinate_system,
        "motion_mode": case.spec.motion.mode,
        "resolved_integrator": integrator,
        "resolved_backend": "cpu",
        "resolved_physics_models": prepared.physics.resolved_models(),
        "particle_count": prepared.schedule.particle_count,
        "particle_id_sha256": hashlib.sha256(particle_bytes).hexdigest(),
    }


def _restore_or_initialize_run_state(
    prepared: _PreparedRun,
    checkpoint: CheckpointState | None,
) -> _RunState:
    if checkpoint is None:
        return _initial_run_state(prepared)
    count = prepared.schedule.particle_count
    _validate_checkpoint_layout(checkpoint, count)
    active, active_index = _restore_active_index(checkpoint, count)
    _restore_field_cell_hint(prepared, checkpoint, count)
    _validate_checkpoint_cursors(prepared, checkpoint)
    return _RunState(
        checkpoint.position_m,
        checkpoint.velocity_m_s,
        checkpoint.charge_number,
        active,
        active_index,
        checkpoint.lifecycle,
        checkpoint.failure_reason_code,
        checkpoint.terminal_time_s,
        checkpoint.event_ordinal,
        checkpoint.physical_boundary_event_ordinal,
        checkpoint.exact_origin_time_s,
        checkpoint.exact_origin_position_m,
        checkpoint.exact_origin_velocity_m_s,
        checkpoint.start_contact_state,
        checkpoint.release_cursor,
        checkpoint.frame_cursor,
        checkpoint.probe_cursor,
        checkpoint.macro_step_count,
        checkpoint.macro_time_s,
        _EventStatistics(
            accepted_particle_pieces=checkpoint.accepted_particle_pieces,
            candidate_queries=checkpoint.candidate_queries,
            refinements=checkpoint.refinements,
            maximum_refinement_depth=checkpoint.maximum_refinement_depth,
            wall_interactions=checkpoint.wall_interactions,
            residual_splits=checkpoint.residual_splits,
            axis_crossings=checkpoint.axis_crossings,
        ),
    )


def _initial_run_state(prepared: _PreparedRun) -> _RunState:
    schedule = prepared.schedule
    count = schedule.particle_count
    return _RunState(
        schedule.position_m.copy(),
        schedule.velocity_m_s.copy(),
        schedule.charge_number.copy(),
        np.zeros(count, dtype=np.bool_),
        ActiveIndex.allocate(count),
        np.full(count, _LIFECYCLE_PENDING, dtype="<u1"),
        np.zeros(count, dtype="<u2"),
        np.full(count, np.inf, dtype="<f8"),
        np.zeros(count, dtype="<u4"),
        np.zeros(count, dtype="<u4"),
        schedule.release_time_s.copy(),
        schedule.position_m.copy(),
        schedule.velocity_m_s.copy(),
        np.where(
            schedule.source_facet_id < 0,
            SURFACE_STATE_RESOLVED,
            SURFACE_STATE_PENDING,
        ).astype("<u1"),
        0,
        0,
        0,
        0,
        prepared.case.spec.time.start_s,
        _EventStatistics(),
    )


def _validate_checkpoint_layout(checkpoint: CheckpointState, count: int) -> None:
    vectors = (
        checkpoint.position_m,
        checkpoint.velocity_m_s,
        checkpoint.exact_origin_position_m,
        checkpoint.exact_origin_velocity_m_s,
    )
    scalars = (
        checkpoint.charge_number,
        checkpoint.lifecycle,
        checkpoint.failure_reason_code,
        checkpoint.terminal_time_s,
        checkpoint.event_ordinal,
        checkpoint.physical_boundary_event_ordinal,
        checkpoint.exact_origin_time_s,
        checkpoint.start_contact_state,
    )
    if any(array.shape != (count, 2) for array in vectors):
        raise EngineError("checkpoint vector state has an incompatible shape")
    if any(array.shape != (count,) for array in scalars):
        raise EngineError("checkpoint scalar state has an incompatible shape")
    if not math.isfinite(checkpoint.macro_time_s) or checkpoint.macro_step_count < 0:
        raise EngineError("checkpoint macro boundary is invalid")


def _restore_active_index(
    checkpoint: CheckpointState,
    count: int,
) -> tuple[np.ndarray, ActiveIndex]:
    rows = checkpoint.active_particle_index
    if rows.ndim != 1 or rows.dtype != np.dtype("<i8"):
        raise EngineError("checkpoint active particle index has an invalid layout")
    if rows.size and (
        int(rows[0]) < 0 or int(rows[-1]) >= count or bool((np.diff(rows) <= 0).any())
    ):
        raise EngineError("checkpoint active particle index is not sorted and unique")
    active = np.zeros(count, dtype=np.bool_)
    active[rows] = True
    if not np.array_equal(active, checkpoint.lifecycle == _LIFECYCLE_ACTIVE):
        raise EngineError("checkpoint active index does not match particle lifecycle")
    active_index = ActiveIndex.allocate(count)
    active_index.add(rows)
    return active, active_index


def _restore_field_cell_hint(
    prepared: _PreparedRun,
    checkpoint: CheckpointState,
    count: int,
) -> None:
    resident = prepared.dynamics.last_field_cell
    saved = checkpoint.last_field_cell
    if resident is None:
        if saved is not None:
            raise EngineError("checkpoint unexpectedly contains field-cell hints")
        return
    if saved is None or saved.shape != (count,) or saved.dtype != np.dtype("<i8"):
        raise EngineError("checkpoint field-cell hints have an invalid layout")
    resident[:] = saved


def _validate_checkpoint_cursors(
    prepared: _PreparedRun,
    checkpoint: CheckpointState,
) -> None:
    time = prepared.case.spec.time
    if not time.start_s <= checkpoint.macro_time_s <= time.end_s:
        raise EngineError("checkpoint macro time lies outside the run interval")
    expected_release = _release_stop(
        prepared.schedule,
        0,
        checkpoint.macro_time_s,
        _PREPARE_SCAN_BATCH_SIZE,
    )
    expected_frame = _frame_stop(prepared.frame_times_s, 0, checkpoint.macro_time_s)
    expected_probe = _frame_stop(prepared.probe_times_s, 0, checkpoint.macro_time_s)
    if checkpoint.release_cursor != expected_release:
        raise EngineError("checkpoint release cursor does not match its macro time")
    if checkpoint.frame_cursor != expected_frame or checkpoint.probe_cursor != expected_probe:
        raise EngineError("checkpoint output cursor does not match its macro time")


def _checkpoint_state(prepared: _PreparedRun, state: _RunState) -> CheckpointState:
    statistics = state.event_statistics
    return CheckpointState(
        macro_time_s=state.macro_start_s,
        macro_step_count=state.macro_step_count,
        release_cursor=state.release_cursor,
        frame_cursor=state.frame_cursor,
        probe_cursor=state.probe_cursor,
        position_m=state.position_m,
        velocity_m_s=state.velocity_m_s,
        charge_number=state.charge_number,
        lifecycle=state.lifecycle,
        failure_reason_code=state.failure_reason_code,
        terminal_time_s=state.terminal_time_s,
        event_ordinal=state.event_ordinal,
        physical_boundary_event_ordinal=state.physical_boundary_event_ordinal,
        exact_origin_time_s=state.exact_origin_time_s,
        exact_origin_position_m=state.exact_origin_position_m,
        exact_origin_velocity_m_s=state.exact_origin_velocity_m_s,
        start_contact_state=state.start_contact_state,
        active_particle_index=state.active_index.rows,
        last_field_cell=prepared.dynamics.last_field_cell,
        accepted_particle_pieces=statistics.accepted_particle_pieces,
        candidate_queries=statistics.candidate_queries,
        refinements=statistics.refinements,
        maximum_refinement_depth=statistics.maximum_refinement_depth,
        wall_interactions=statistics.wall_interactions,
        residual_splits=statistics.residual_splits,
        axis_crossings=statistics.axis_crossings,
    )


def _validate_completed_run_state(prepared: _PreparedRun, state: _RunState) -> None:
    if state.frame_cursor != len(prepared.frame_times_s):
        raise EngineError("not every requested trajectory frame was emitted")
    if state.probe_cursor != len(prepared.probe_times_s):
        raise EngineError("not every requested particle probe was emitted")
    if state.release_cursor != prepared.schedule.particle_count or bool(
        (state.lifecycle == _LIFECYCLE_PENDING).any()
    ):
        raise EngineError("not every validated particle was released by the run end")


def _advance_active_slabs(
    prepared: _PreparedRun,
    active_particle_index: np.ndarray,
    macro_interval: int,
    macro_start_s: float,
    macro_end_s: float,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    event_statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
    writer: ResultWriter,
) -> None:
    """Advance bounded slabs and flush each stable payload."""

    slab_size = prepared.memory_plan.slab_particles
    if active_particle_index.size and slab_size < 1:
        raise EngineError("memory plan cannot hold one proposal slab")

    def run_slab(particle_index: np.ndarray) -> _SlabResult:
        return _advance_particle_slab(
            prepared,
            particle_index,
            macro_interval,
            macro_start_s,
            macro_end_s,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            exact_origin_time_s,
            exact_origin_position_m,
            exact_origin_velocity_m_s,
            start_contact_state,
            replay,
            failure_reason_code,
            writer,
        )

    for slab_begin in range(0, active_particle_index.size, max(1, slab_size)):
        slab = active_particle_index[slab_begin : slab_begin + slab_size]
        result = run_slab(slab)
        event_statistics.merge(result.statistics)
        del result


def _advance_particle_slab(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    macro_interval: int,
    macro_start_s: float,
    macro_end_s: float,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    failure_reason_code: np.ndarray,
    writer: ResultWriter,
) -> _SlabResult:
    """Own one disjoint slab's proposal, residual work, and failures."""

    statistics = _EventStatistics()
    event_buffer = _allocate_boundary_event_buffer(
        min(particle_index.size, prepared.memory_plan.event_staging_capacity),
        prepared.event_candidate_capacity,
    )
    failure_buffer = _allocate_failure_event_buffer(particle_index.size)
    start_time_s = _proposal_start_times(
        prepared,
        particle_index,
        macro_start_s,
        exact_origin_time_s,
    )
    start_position, start_velocity = _proposal_start_state(
        prepared,
        particle_index,
        start_time_s,
        position_m,
        velocity_m_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
    )
    if prepared.physics.noise is not None:
        _advance_brownian_slab(
            prepared,
            particle_index,
            macro_interval,
            macro_end_s,
            start_time_s,
            start_position,
            start_velocity,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            exact_origin_time_s,
            exact_origin_position_m,
            exact_origin_velocity_m_s,
            start_contact_state,
            replay,
            statistics,
            failure_reason_code,
            writer,
            event_buffer,
            failure_buffer,
        )
        if event_buffer.row_count:
            raise EngineError("boundary-event wave was not synchronously flushed")
        _write_failure_events(writer, prepared, failure_buffer)
        return _SlabResult(statistics)
    proposal = _propose_root_batch(
        prepared,
        particle_index,
        start_time_s,
        macro_end_s,
        start_position,
        start_velocity,
        charge_number[particle_index],
    )
    _advance_macro_proposal(
        prepared,
        proposal,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
        start_contact_state,
        replay,
        statistics,
        failure_reason_code,
        writer,
        event_buffer,
        failure_buffer,
    )
    if event_buffer.row_count:
        raise EngineError("boundary-event wave was not synchronously flushed")
    _write_failure_events(writer, prepared, failure_buffer)
    return _SlabResult(statistics)


def _langevin_root_coefficients(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    duration_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
) -> _LangevinRoot:
    """Evaluate the selected revision's one frozen coefficient state."""

    noise = prepared.physics.noise
    if noise is None:
        raise EngineError("Langevin coefficients require a resolved noise model")
    if noise.revision == BROWNIAN_FROZEN_START_REVISION:
        return _LangevinRoot(
            prepared.dynamics.evaluate_langevin_coefficients(
                particle_index,
                start_time_s,
                start_position_m,
                start_velocity_m_s,
                start_charge_number,
            ),
            duration_s.copy(),
            np.zeros(particle_index.size, dtype="<u2"),
        )
    guarded_duration_s, predictor, guard_valid, guard_failure = _guard_langevin_root_duration(
        prepared,
        particle_index,
        start_time_s,
        duration_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
    )
    midpoint_time_s = start_time_s + 0.5 * guarded_duration_s
    midpoint = _evaluate_admitted_langevin_midpoints(
        prepared,
        particle_index,
        midpoint_time_s,
        start_charge_number,
        predictor,
        guard_valid,
    )
    return _LangevinRoot(
        midpoint,
        guarded_duration_s,
        guard_failure,
    )


def _evaluate_admitted_langevin_midpoints(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    midpoint_time_s: np.ndarray,
    root_charge_number: np.ndarray,
    predictor: ProposalSample,
    guard_valid: np.ndarray,
) -> _LangevinCoefficients:
    """Evaluate stochastic coefficients only at admitted deterministic midpoints."""

    count = int(particle_index.size)
    drag_rate = np.ones(count, dtype="<f8")
    equilibrium = np.zeros((count, 2), dtype="<f8")
    additive = np.zeros((count, 2), dtype="<f8")
    charge_affine_rate = np.zeros(count, dtype="<f8")
    charge_derivative = np.zeros(count, dtype="<f8")
    thermal = np.ones(count, dtype="<f8")
    support = predictor.support_inside.copy()
    applicability = predictor.applicability_inside.copy()
    numerical_status = predictor.numerical_status.copy()
    unresolved = (numerical_status == NUMERICAL_STATUS_OK) & ~guard_valid
    numerical_status[unresolved] = INTEGRATOR_ACCURACY_FAILURE
    rows = np.flatnonzero(guard_valid).astype("<i8", copy=False)
    if not rows.size:
        return _LangevinCoefficients(
            drag_rate_s_inv=drag_rate,
            equilibrium_velocity_m_s=equilibrium,
            additive_acceleration_m_s2=additive,
            charge_affine_rate_number_s=charge_affine_rate,
            charge_rate_derivative_s_inv=charge_derivative,
            thermal_velocity_variance_m2_s2=thermal,
            support_inside=support,
            applicability_inside=applicability,
            numerical_status=numerical_status,
            field_cell_id=None,
        )
    midpoint = prepared.dynamics.evaluate_langevin_coefficients(
        particle_index[rows],
        midpoint_time_s[rows],
        predictor.position_m[rows],
        predictor.velocity_m_s[rows],
        predictor.charge_number[rows],
    )
    drag_rate[rows] = midpoint.drag_rate_s_inv
    equilibrium[rows] = midpoint.equilibrium_velocity_m_s
    additive[rows] = midpoint.additive_acceleration_m_s2
    charge_derivative[rows] = midpoint.charge_rate_derivative_s_inv
    charge_affine_rate[rows] = midpoint.charge_affine_rate_number_s + charge_derivative[rows] * (
        root_charge_number[rows] - predictor.charge_number[rows]
    )
    thermal[rows] = midpoint.thermal_velocity_variance_m2_s2
    support[rows] &= midpoint.support_inside
    applicability[rows] &= midpoint.applicability_inside
    first_failure = (numerical_status[rows] == NUMERICAL_STATUS_OK) & (
        midpoint.numerical_status != NUMERICAL_STATUS_OK
    )
    numerical_status[rows[first_failure]] = midpoint.numerical_status[first_failure]
    field_cell_id: np.ndarray | None = None
    if midpoint.field_cell_id is not None:
        field_cell_id = np.full(count, -1, dtype="<i8")
        field_cell_id[rows] = midpoint.field_cell_id
    return _LangevinCoefficients(
        drag_rate_s_inv=drag_rate,
        equilibrium_velocity_m_s=equilibrium,
        additive_acceleration_m_s2=additive,
        charge_affine_rate_number_s=charge_affine_rate,
        charge_rate_derivative_s_inv=charge_derivative,
        thermal_velocity_variance_m2_s2=thermal,
        support_inside=support,
        applicability_inside=applicability,
        numerical_status=numerical_status,
        field_cell_id=field_cell_id,
    )


def _guard_langevin_root_duration(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    requested_duration_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
) -> tuple[np.ndarray, ProposalSample, np.ndarray, np.ndarray]:
    """Shorten rows until the deterministic coefficient midpoint is admissible.

    The first valid dyadic half-root ends no earlier than the preceding invalid
    midpoint.  An inward deterministic path therefore still spans its first
    axis/material crossing instead of approaching that crossing by a Zeno
    sequence.  No Brownian draw is consumed while this deterministic guard is
    being resolved.
    """

    duration_s = requested_duration_s.copy()
    predictor: ProposalSample | None = None
    admitted = np.zeros(particle_index.size, dtype=np.bool_)
    failure_reason = np.zeros(particle_index.size, dtype="<u2")
    maximum_refinements = prepared.case.spec.solver.event.max_refinements
    for refinement in range(maximum_refinements + 1):
        midpoint_time_s = start_time_s + 0.5 * duration_s
        predictor = exponential_frozen_start_predictor(
            particle_index,
            start_time_s,
            0.5 * duration_s,
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
            evaluator=prepared.dynamics.evaluate_relaxation,
        )
        admitted, retry, current_failure = _classify_langevin_midpoint(
            prepared,
            start_time_s,
            duration_s,
            midpoint_time_s,
            predictor,
        )
        newly_failed = (failure_reason == 0) & (current_failure != 0)
        failure_reason[newly_failed] = current_failure[newly_failed]
        admitted &= failure_reason == 0
        pending = retry & (failure_reason == 0)
        unresolved = ~admitted & (failure_reason == 0) & ~pending
        failure_reason[unresolved] = _FAILURE_INTEGRATOR_ACCURACY
        if not bool(pending.any()):
            break
        if refinement == maximum_refinements:
            failure_reason[pending] = _FAILURE_INTEGRATOR_ACCURACY
            admitted[pending] = False
            break
        duration_s[pending] *= 0.5
    if predictor is None:
        raise EngineError("Langevin midpoint guard did not evaluate a predictor")
    return duration_s, predictor, admitted, failure_reason


def _classify_langevin_midpoint(
    prepared: _PreparedRun,
    start_time_s: np.ndarray,
    duration_s: np.ndarray,
    midpoint_time_s: np.ndarray,
    predictor: ProposalSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify deterministic midpoints as admitted, retryable, or failed."""

    representable = midpoint_time_s > start_time_s
    representable &= start_time_s + duration_s > start_time_s
    midpoint_position_m = predictor.position_m
    inside_volume = points_inside_volume(prepared.geometry, midpoint_position_m)
    across_axis = np.zeros(midpoint_time_s.size, dtype=np.bool_)
    if prepared.case.data.coordinate_system == "axisymmetric_rz":
        across_axis = midpoint_position_m[:, 0] < 0.0
    failure_reason = np.zeros(midpoint_time_s.size, dtype="<u2")
    failure_reason[~representable] = _FAILURE_INTEGRATOR_ACCURACY
    numerical_failure = predictor.numerical_status != NUMERICAL_STATUS_OK
    numerical_codes = _proposal_numerical_failure_codes(predictor.numerical_status)
    failure_reason[numerical_failure] = numerical_codes[numerical_failure]
    missing_support = ~predictor.support_inside & (failure_reason == 0)
    failure_reason[missing_support] = _FAILURE_FIELD_SUPPORT
    inapplicable = ~predictor.applicability_inside & (failure_reason == 0)
    failure_reason[inapplicable] = _FAILURE_MODEL_APPLICABILITY
    retry = representable & (failure_reason == 0) & (across_axis | ~inside_volume)
    admitted = representable & (failure_reason == 0) & ~retry
    retry &= failure_reason == 0
    return admitted, retry, failure_reason


def _subset_langevin_coefficients(
    coefficients: _LangevinCoefficients,
    rows: np.ndarray,
) -> _LangevinCoefficients:
    """Retain one aligned coefficient table after pre-RNG root rejection."""

    return _LangevinCoefficients(
        drag_rate_s_inv=coefficients.drag_rate_s_inv[rows],
        equilibrium_velocity_m_s=coefficients.equilibrium_velocity_m_s[rows],
        additive_acceleration_m_s2=coefficients.additive_acceleration_m_s2[rows],
        charge_affine_rate_number_s=coefficients.charge_affine_rate_number_s[rows],
        charge_rate_derivative_s_inv=coefficients.charge_rate_derivative_s_inv[rows],
        thermal_velocity_variance_m2_s2=(coefficients.thermal_velocity_variance_m2_s2[rows]),
        support_inside=coefficients.support_inside[rows],
        applicability_inside=coefficients.applicability_inside[rows],
        numerical_status=coefficients.numerical_status[rows],
        field_cell_id=(
            None if coefficients.field_cell_id is None else coefficients.field_cell_id[rows]
        ),
    )


def _prepare_langevin_root_batch(
    prepared: _PreparedRun,
    particles: np.ndarray,
    starts_s: np.ndarray,
    requested_duration_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failure_buffer: _FailureEventBuffer,
) -> _LangevinRootBatch | None:
    """Resolve and validate all coefficients before addressing Brownian RNG."""

    try:
        root = _langevin_root_coefficients(
            prepared,
            particles,
            starts_s,
            requested_duration_s,
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
        )
    except (
        FieldLocationError,
        GeometryPreparationError,
        PhysicsEvaluationError,
        ValueError,
    ) as error:
        raise EngineError("Brownian macro-root coefficients could not be evaluated") from error
    noise = prepared.physics.noise
    if noise is None:
        raise EngineError("Brownian root preparation lost its noise model")
    coefficients = root.coefficients
    duration_s = root.duration_s
    if noise.revision == BROWNIAN_FROZEN_START_REVISION and coefficients.field_cell_id is not None:
        prepared.dynamics.commit_field_cell(particles, coefficients.field_cell_id)

    numerical_status = coefficients.numerical_status.copy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        relaxation_argument = coefficients.drag_rate_s_inv * duration_s
    representable_argument = np.isfinite(relaxation_argument)
    representable_argument &= relaxation_argument > 0.0
    representable_argument &= relaxation_argument <= MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT
    unresolved_argument = (numerical_status == NUMERICAL_STATUS_OK) & ~representable_argument
    numerical_status[unresolved_argument] = PHYSICS_NUMERICAL_FAILURE
    valid = numerical_status == NUMERICAL_STATUS_OK
    valid &= coefficients.support_inside
    valid &= coefficients.applicability_inside
    valid &= root.failure_reason_code == 0
    if noise.revision == BROWNIAN_FROZEN_START_REVISION:
        effective_equilibrium = coefficients.equilibrium_velocity_m_s
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            effective_equilibrium = (
                coefficients.equilibrium_velocity_m_s
                + coefficients.additive_acceleration_m_s2 / coefficients.drag_rate_s_inv[:, None]
            )
    finite_equilibrium = np.isfinite(effective_equilibrium).all(axis=1)
    numerical_status[(numerical_status == NUMERICAL_STATUS_OK) & ~finite_equilibrium] = (
        PHYSICS_NUMERICAL_FAILURE
    )
    valid &= finite_equilibrium

    safe_rate = np.where(valid, coefficients.drag_rate_s_inv, 1.0)
    safe_thermal = np.where(valid, coefficients.thermal_velocity_variance_m2_s2, 0.0)
    safe_equilibrium = np.where(valid[:, None], effective_equilibrium, 0.0)
    if noise.revision != BROWNIAN_RZ_MIDPOINT_REVISION:
        return _LangevinRootBatch(
            particles,
            starts_s,
            duration_s,
            requested_duration_s,
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
            coefficients,
            numerical_status,
            safe_rate,
            safe_thermal,
            safe_equilibrium,
        )

    _mark_langevin_root_failures(
        valid,
        particles,
        starts_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        numerical_status,
        coefficients,
        root.failure_reason_code,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failure_buffer,
    )
    rows = np.flatnonzero(valid).astype("<i8", copy=False)
    if not rows.size:
        return None
    return _LangevinRootBatch(
        particles[rows],
        starts_s[rows],
        duration_s[rows],
        requested_duration_s[rows],
        start_position_m[rows],
        start_velocity_m_s[rows],
        start_charge_number[rows],
        _subset_langevin_coefficients(coefficients, rows),
        numerical_status[rows],
        safe_rate[rows],
        safe_thermal[rows],
        safe_equilibrium[rows],
    )


def _fail_langevin_root_budget(
    particles: np.ndarray,
    starts_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failure_buffer: _FailureEventBuffer,
    reason_code: np.uint16,
) -> None:
    """Fail a root cohort that exhausted its bounded stochastic restarts."""

    for local_row, particle_value in enumerate(particles):
        particle = int(particle_value)
        _mark_particle_failed(
            particle,
            _ParticleFailure(
                reason_code,
                float(starts_s[local_row]),
                start_position_m[local_row],
                start_velocity_m_s[local_row],
                float(start_charge_number[local_row]),
            ),
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failure_buffer,
        )


def _initialize_zero_duration_brownian_rows(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    zero_rows: np.ndarray,
    root_start_time_s: np.ndarray,
    root_start_position_m: np.ndarray,
    root_start_velocity_m_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
    writer: ResultWriter,
    event_buffer: _BoundaryEventBuffer,
    failure_buffer: _FailureEventBuffer,
) -> None:
    """Resolve release contact when a stochastic root has no positive duration."""

    if not zero_rows.size:
        return
    particles = particle_index[zero_rows]
    _initialize_surface_releases(
        prepared,
        particles,
        root_start_time_s[zero_rows],
        root_start_position_m[zero_rows],
        root_start_velocity_m_s[zero_rows],
        charge_number[particles].copy(),
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        start_contact_state,
        event_buffer,
        failure_buffer,
        replay,
        statistics,
        failure_reason_code,
    )
    _flush_boundary_event_wave(writer, prepared, event_buffer)


def _langevin_restart_failure_reason(
    prepared: _PreparedRun,
    axis_restart_count: int,
    guard_restart_count: int,
) -> np.uint16:
    """Map independent bounded-restart exhaustion to its owned failure class."""

    if axis_restart_count > prepared.case.spec.solver.event.max_interactions_per_step:
        return _FAILURE_NUMERICAL_EVENT_BUDGET
    if guard_restart_count > prepared.case.spec.solver.event.max_refinements:
        return _FAILURE_INTEGRATOR_ACCURACY
    return np.uint16(0)


def _advance_brownian_slab(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    macro_interval: int,
    macro_end_s: float,
    root_start_time_s: np.ndarray,
    root_start_position_m: np.ndarray,
    root_start_velocity_m_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
    writer: ResultWriter,
    event_buffer: _BoundaryEventBuffer,
    failure_buffer: _FailureEventBuffer,
    *,
    root_interval: int = 0,
    axis_restart_count: int = 0,
    guard_restart_count: int = 0,
) -> None:
    """Advance one slab over a fixed-depth, conditionally split OU tree."""

    noise = prepared.physics.noise
    if noise is None:
        raise EngineError("Brownian slab requires a resolved noise model")
    root_duration_s = macro_end_s - root_start_time_s
    zero_rows = np.flatnonzero(root_duration_s == 0.0).astype("<i8", copy=False)
    _initialize_zero_duration_brownian_rows(
        prepared,
        particle_index,
        zero_rows,
        root_start_time_s,
        root_start_position_m,
        root_start_velocity_m_s,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        start_contact_state,
        replay,
        statistics,
        failure_reason_code,
        writer,
        event_buffer,
        failure_buffer,
    )
    moving_rows = np.flatnonzero(root_duration_s > 0.0).astype("<i8", copy=False)
    if not moving_rows.size:
        return
    particles = particle_index[moving_rows]
    starts = root_start_time_s[moving_rows]
    requested_durations = root_duration_s[moving_rows]
    start_positions = root_start_position_m[moving_rows]
    start_velocities = root_start_velocity_m_s[moving_rows]
    start_charges = charge_number[particles].copy()
    restart_failure = _langevin_restart_failure_reason(
        prepared,
        axis_restart_count,
        guard_restart_count,
    )
    if restart_failure != 0:
        _fail_langevin_root_budget(
            particles,
            starts,
            start_positions,
            start_velocities,
            start_charges,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failure_buffer,
            restart_failure,
        )
        return
    batch = _prepare_langevin_root_batch(
        prepared,
        particles,
        starts,
        requested_durations,
        start_positions,
        start_velocities,
        start_charges,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failure_buffer,
    )
    if batch is None:
        return
    particles = batch.particles
    starts = batch.starts_s
    durations = batch.duration_s
    requested_durations = batch.requested_duration_s
    coefficients = batch.coefficients
    numerical_status = batch.numerical_status
    safe_rate = batch.drag_rate_s_inv
    safe_thermal = batch.thermal_velocity_variance_m2_s2
    safe_equilibrium = batch.equilibrium_velocity_m_s
    leaf_count = 1 << noise.interval_tree_depth
    particle_ids = prepared.schedule.particle_id[particles].astype(np.uint64, copy=False)
    increments = _brownian_leaf_increments(
        seed=prepared.case.spec.solver.seed,
        particle_id=particle_ids,
        macro_interval=macro_interval,
        root_interval=root_interval,
        drag_rate_s_inv=safe_rate,
        thermal_velocity_variance_m2_s2=safe_thermal,
        root_duration_s=durations,
        tree_depth=noise.interval_tree_depth,
        numerically_valid=numerical_status == NUMERICAL_STATUS_OK,
    )
    root_continuing = np.ones(particles.size, dtype=np.bool_)
    for leaf_index in range(leaf_count):
        increment, increment_valid = _next_brownian_increment(increments)
        local_active = active[particles] & root_continuing
        selected = np.flatnonzero(local_active).astype("<i8", copy=False)
        if not selected.size:
            break
        leaf_start_s = starts[selected] + durations[selected] * (leaf_index / leaf_count)
        leaf_target_s = starts[selected] + durations[selected] * ((leaf_index + 1) / leaf_count)
        selected_particles = particles[selected]
        current_position = position_m[selected_particles].copy()
        current_velocity = velocity_m_s[selected_particles].copy()
        current_charge = charge_number[selected_particles].copy()
        resolved_time = leaf_target_s > leaf_start_s
        for local_row_value in np.flatnonzero(~resolved_time):
            local_row = int(local_row_value)
            particle = int(selected_particles[local_row])
            _mark_particle_failed(
                particle,
                _ParticleFailure(
                    _FAILURE_INTEGRATOR_ACCURACY,
                    float(leaf_start_s[local_row]),
                    current_position[local_row],
                    current_velocity[local_row],
                    float(current_charge[local_row]),
                ),
                position_m,
                velocity_m_s,
                charge_number,
                active,
                lifecycle,
                terminal_time_s,
                event_ordinal,
                failure_reason_code,
                failure_buffer,
            )
        selected = selected[resolved_time]
        if not selected.size:
            continue
        leaf_start_s = leaf_start_s[resolved_time]
        leaf_target_s = leaf_target_s[resolved_time]
        selected_particles = selected_particles[resolved_time]
        current_position = current_position[resolved_time]
        current_velocity = current_velocity[resolved_time]
        current_charge = current_charge[resolved_time]
        leaf_valid = increment_valid[selected]
        _mark_brownian_numerical_failures(
            leaf_valid,
            selected_particles,
            leaf_start_s,
            current_position,
            current_velocity,
            current_charge,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failure_buffer,
        )
        selected = selected[leaf_valid]
        if not selected.size:
            continue
        leaf_start_s = leaf_start_s[leaf_valid]
        leaf_target_s = leaf_target_s[leaf_valid]
        selected_particles = selected_particles[leaf_valid]
        current_position = current_position[leaf_valid]
        current_velocity = current_velocity[leaf_valid]
        current_charge = current_charge[leaf_valid]
        end_position, end_velocity, advance_valid = _advance_langevin_leaf_endpoint(
            current_position,
            current_velocity,
            safe_equilibrium[selected],
            safe_rate[selected],
            durations[selected] / leaf_count,
            increment,
            selected,
        )
        _mark_brownian_numerical_failures(
            advance_valid,
            selected_particles,
            leaf_start_s,
            current_position,
            current_velocity,
            current_charge,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failure_buffer,
        )
        if not bool(advance_valid.any()):
            continue
        selected = selected[advance_valid]
        leaf_start_s = leaf_start_s[advance_valid]
        leaf_target_s = leaf_target_s[advance_valid]
        selected_particles = selected_particles[advance_valid]
        current_position = current_position[advance_valid]
        current_velocity = current_velocity[advance_valid]
        current_charge = current_charge[advance_valid]
        end_position = end_position[advance_valid]
        end_velocity = end_velocity[advance_valid]
        end_charge = _advance_langevin_leaf_charge(
            noise.revision,
            current_charge,
            batch.start_charge_number[selected],
            leaf_target_s - starts[selected],
            coefficients.charge_affine_rate_number_s[selected],
            coefficients.charge_rate_derivative_s_inv[selected],
        )
        charge_valid = np.isfinite(end_charge)
        _mark_brownian_numerical_failures(
            charge_valid,
            selected_particles,
            leaf_start_s,
            current_position,
            current_velocity,
            current_charge,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failure_buffer,
        )
        if not bool(charge_valid.any()):
            continue
        selected = selected[charge_valid]
        leaf_start_s = leaf_start_s[charge_valid]
        leaf_target_s = leaf_target_s[charge_valid]
        selected_particles = selected_particles[charge_valid]
        current_position = current_position[charge_valid]
        current_velocity = current_velocity[charge_valid]
        current_charge = current_charge[charge_valid]
        end_position = end_position[charge_valid]
        end_velocity = end_velocity[charge_valid]
        end_charge = end_charge[charge_valid]
        proposal = _build_langevin_leaf_proposal(
            selected_particles,
            leaf_start_s,
            leaf_target_s,
            current_position,
            current_velocity,
            current_charge,
            end_position,
            end_velocity,
            end_charge,
            starts[selected],
            batch.start_charge_number[selected],
            coefficients.charge_affine_rate_number_s[selected],
            coefficients.charge_rate_derivative_s_inv[selected],
            coefficients.support_inside[selected],
            coefficients.applicability_inside[selected],
            numerical_status[selected],
        )
        restart_time_s = _advance_macro_proposal(
            prepared,
            proposal,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            exact_origin_time_s,
            exact_origin_position_m,
            exact_origin_velocity_m_s,
            start_contact_state,
            replay,
            statistics,
            failure_reason_code,
            writer,
            event_buffer,
            failure_buffer,
        )
        restart_selected = _restart_langevin_axis_rows(
            restart_time_s,
            selected,
            selected_particles,
            prepared,
            macro_interval,
            macro_end_s,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            exact_origin_time_s,
            exact_origin_position_m,
            exact_origin_velocity_m_s,
            start_contact_state,
            replay,
            statistics,
            failure_reason_code,
            writer,
            event_buffer,
            failure_buffer,
            root_interval,
            axis_restart_count,
            guard_restart_count,
        )
        root_continuing[restart_selected] = False

    if noise.revision != BROWNIAN_RZ_MIDPOINT_REVISION:
        return
    root_target_s = starts + durations
    shortened = durations < requested_durations
    residual_rows = np.flatnonzero(
        shortened & root_continuing & active[particles] & (root_target_s < macro_end_s)
    ).astype("<i8", copy=False)
    if not residual_rows.size:
        return
    residual_particles = particles[residual_rows]
    _advance_brownian_slab(
        prepared,
        residual_particles,
        macro_interval,
        macro_end_s,
        root_target_s[residual_rows],
        position_m[residual_particles].copy(),
        velocity_m_s[residual_particles].copy(),
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
        start_contact_state,
        replay,
        statistics,
        failure_reason_code,
        writer,
        event_buffer,
        failure_buffer,
        root_interval=root_interval + 1,
        axis_restart_count=axis_restart_count,
        guard_restart_count=guard_restart_count + 1,
    )


def _next_brownian_increment(
    increments: Iterator[tuple[JointOuIncrement, np.ndarray]],
) -> tuple[JointOuIncrement, np.ndarray]:
    """Read one declared leaf or fail the corrupted conditional tree."""

    try:
        return next(increments)
    except StopIteration as error:
        raise EngineError("Brownian interval tree ended before its declared depth") from error


def _advance_langevin_leaf_endpoint(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    equilibrium_velocity_m_s: np.ndarray,
    drag_rate_s_inv: np.ndarray,
    duration_s: np.ndarray,
    increment: JointOuIncrement,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one selected conditional OU leaf with localized arithmetic failure."""

    leaf_increment = JointOuIncrement(
        increment.position_m[selected],
        increment.velocity_m_s[selected],
    )
    try:
        return _advance_joint_ou_particle_local(
            position_m,
            velocity_m_s,
            equilibrium_velocity_m_s,
            drag_rate_s_inv,
            duration_s,
            leaf_increment,
        )
    except ValueError as error:
        raise EngineError("Brownian leaf proposal could not be constructed") from error


def _advance_langevin_leaf_charge(
    noise_revision: str,
    current_charge_number: np.ndarray,
    root_charge_number: np.ndarray,
    elapsed_from_root_s: np.ndarray,
    charge_affine_rate_number_s: np.ndarray,
    charge_rate_derivative_s_inv: np.ndarray,
) -> np.ndarray:
    """Advance the selected revision's root-owned charge representation."""

    if noise_revision == BROWNIAN_FROZEN_START_REVISION:
        return current_charge_number.copy()
    return affine_exponential_charge(
        root_charge_number,
        elapsed_from_root_s,
        charge_affine_rate_number_s,
        charge_rate_derivative_s_inv,
    )


def _build_langevin_leaf_proposal(
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
    end_position_m: np.ndarray,
    end_velocity_m_s: np.ndarray,
    end_charge_number: np.ndarray,
    charge_root_time_s: np.ndarray,
    charge_root_number: np.ndarray,
    charge_affine_rate_number_s: np.ndarray,
    charge_rate_derivative_s_inv: np.ndarray,
    support_inside: np.ndarray,
    applicability_inside: np.ndarray,
    numerical_status: np.ndarray,
) -> StepProposal:
    """Construct one finite-depth stochastic path with root-owned dense charge."""

    try:
        return cubic_hermite_step(
            particle_index,
            start_time_s,
            target_time_s,
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
            end_position_m,
            end_velocity_m_s,
            end_charge_number=end_charge_number,
            charge_root_time_s=charge_root_time_s,
            charge_root_number=charge_root_number,
            charge_affine_rate_number_s=charge_affine_rate_number_s,
            charge_rate_derivative_s_inv=charge_rate_derivative_s_inv,
            support_inside=support_inside,
            applicability_inside=applicability_inside,
            numerical_status=numerical_status,
        )
    except (FloatingPointError, ValueError) as error:
        raise EngineError("Brownian leaf proposal could not be constructed") from error


def _restart_langevin_axis_rows(
    restart_time_s: np.ndarray | None,
    selected: np.ndarray,
    selected_particles: np.ndarray,
    prepared: _PreparedRun,
    macro_interval: int,
    macro_end_s: float,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
    writer: ResultWriter,
    event_buffer: _BoundaryEventBuffer,
    failure_buffer: _FailureEventBuffer,
    root_interval: int,
    axis_restart_count: int,
    guard_restart_count: int,
) -> np.ndarray:
    """Restart only rows whose accepted cubic prefix reached the RZ axis."""

    if restart_time_s is None:
        return np.empty(0, dtype="<i8")
    restart_rows = np.flatnonzero(np.isfinite(restart_time_s)).astype("<i8", copy=False)
    if not restart_rows.size:
        return np.empty(0, dtype="<i8")
    restart_particles = selected_particles[restart_rows]
    _advance_brownian_slab(
        prepared,
        restart_particles,
        macro_interval,
        macro_end_s,
        restart_time_s[restart_rows],
        position_m[restart_particles].copy(),
        velocity_m_s[restart_particles].copy(),
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
        start_contact_state,
        replay,
        statistics,
        failure_reason_code,
        writer,
        event_buffer,
        failure_buffer,
        root_interval=root_interval + 1,
        axis_restart_count=axis_restart_count + 1,
        guard_restart_count=guard_restart_count,
    )
    return selected[restart_rows]


def _mark_langevin_root_failures(
    valid: np.ndarray,
    particle_index: np.ndarray,
    time_s: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    particle_charge: np.ndarray,
    numerical_status: np.ndarray,
    coefficients: _LangevinCoefficients,
    root_failure_reason_code: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failure_buffer: _FailureEventBuffer,
) -> None:
    """Fail B03 roots before their counter-based Brownian draw is addressed."""

    numerical_codes = _proposal_numerical_failure_codes(numerical_status)
    for local_row_value in np.flatnonzero(~valid):
        local_row = int(local_row_value)
        if root_failure_reason_code[local_row] != 0:
            reason = root_failure_reason_code[local_row]
        elif not coefficients.support_inside[local_row]:
            reason = _FAILURE_FIELD_SUPPORT
        elif not coefficients.applicability_inside[local_row]:
            reason = _FAILURE_MODEL_APPLICABILITY
        else:
            reason = numerical_codes[local_row]
            if reason == 0:
                raise EngineError("invalid Brownian root has no failure reason")
        particle = int(particle_index[local_row])
        _mark_particle_failed(
            particle,
            _ParticleFailure(
                reason,
                float(time_s[local_row]),
                position[local_row],
                velocity[local_row],
                float(particle_charge[local_row]),
            ),
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failure_buffer,
        )


def _mark_brownian_numerical_failures(
    numerically_valid: np.ndarray,
    particle_index: np.ndarray,
    time_s: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    particle_charge: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failure_buffer: _FailureEventBuffer,
) -> None:
    """Commit only the Brownian rows whose arithmetic was not representable."""

    for local_row_value in np.flatnonzero(~numerically_valid):
        local_row = int(local_row_value)
        particle = int(particle_index[local_row])
        _mark_particle_failed(
            particle,
            _ParticleFailure(
                _FAILURE_NONFINITE_PHYSICS,
                float(time_s[local_row]),
                position[local_row],
                velocity[local_row],
                float(particle_charge[local_row]),
            ),
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failure_buffer,
        )


def _brownian_leaf_increments(
    *,
    seed: int,
    particle_id: np.ndarray,
    macro_interval: int,
    root_interval: int,
    drag_rate_s_inv: np.ndarray,
    thermal_velocity_variance_m2_s2: np.ndarray,
    root_duration_s: np.ndarray,
    tree_depth: int,
    numerically_valid: np.ndarray,
) -> Iterator[tuple[JointOuIncrement, np.ndarray]]:
    """Yield depth-first leaves while retaining only one right child per level."""

    root, root_valid = _joint_ou_increment_particle_local(
        drag_rate_s_inv,
        thermal_velocity_variance_m2_s2,
        root_duration_s,
        _brownian_normal_tensor(
            seed,
            particle_id,
            macro_interval,
            root_interval=root_interval,
            tree_level=0,
            tree_index=0,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        ),
        numerically_valid,
    )
    if tree_depth == 0:
        yield root, root_valid
        return
    count = particle_id.size
    right_position = np.empty((tree_depth, count, 2), dtype="<f8")
    right_velocity = np.empty((tree_depth, count, 2), dtype="<f8")
    right_valid = np.empty((tree_depth, count), dtype=np.bool_)
    current = root
    current_valid = root_valid
    for level in range(tree_depth):
        left, right, child_valid = _split_joint_ou_increment_particle_local(
            current,
            drag_rate_s_inv,
            thermal_velocity_variance_m2_s2,
            root_duration_s / (1 << level),
            _brownian_normal_tensor(
                seed,
                particle_id,
                macro_interval,
                root_interval=root_interval,
                tree_level=level,
                tree_index=0,
                draw_kind=BROWNIAN_SPLIT_NORMAL_STREAM,
            ),
            current_valid,
        )
        right_position[level] = right.position_m
        right_velocity[level] = right.velocity_m_s
        right_valid[level] = child_valid
        current = left
        current_valid = child_valid
    yield current, current_valid

    for leaf_index in range(1, 1 << tree_depth):
        trailing_zeros = (leaf_index & -leaf_index).bit_length() - 1
        ancestor_level = tree_depth - 1 - trailing_zeros
        current = JointOuIncrement(
            right_position[ancestor_level],
            right_velocity[ancestor_level],
        )
        current_valid = right_valid[ancestor_level].copy()
        for level in range(ancestor_level + 1, tree_depth):
            node_index = leaf_index >> (tree_depth - level)
            left, right, child_valid = _split_joint_ou_increment_particle_local(
                current,
                drag_rate_s_inv,
                thermal_velocity_variance_m2_s2,
                root_duration_s / (1 << level),
                _brownian_normal_tensor(
                    seed,
                    particle_id,
                    macro_interval,
                    root_interval=root_interval,
                    tree_level=level,
                    tree_index=node_index,
                    draw_kind=BROWNIAN_SPLIT_NORMAL_STREAM,
                ),
                current_valid,
            )
            right_position[level] = right.position_m
            right_velocity[level] = right.velocity_m_s
            right_valid[level] = child_valid
            current = left
            current_valid = child_valid
        yield current, current_valid


def _joint_ou_increment_particle_local(
    drag_rate_s_inv: np.ndarray,
    thermal_velocity_variance_m2_s2: np.ndarray,
    duration_s: np.ndarray,
    standard_normal: np.ndarray,
    numerically_valid: np.ndarray,
) -> tuple[JointOuIncrement, np.ndarray]:
    """Draw a vectorized OU increment and localize representability failures."""

    valid = np.asarray(numerically_valid, dtype=np.bool_).copy()
    count = int(valid.size)
    rows = np.flatnonzero(valid)
    position = np.zeros((count, 2), dtype="<f8")
    velocity = np.zeros((count, 2), dtype="<f8")
    if not rows.size:
        return JointOuIncrement(position, velocity), valid
    all_valid = rows.size == count
    selected_rate = drag_rate_s_inv if all_valid else drag_rate_s_inv[rows]
    selected_thermal = (
        thermal_velocity_variance_m2_s2 if all_valid else thermal_velocity_variance_m2_s2[rows]
    )
    selected_duration = duration_s if all_valid else duration_s[rows]
    selected_normal = standard_normal if all_valid else standard_normal[rows]
    try:
        increment = joint_ou_increment(
            selected_rate,
            selected_thermal,
            selected_duration,
            selected_normal,
        )
    except FloatingPointError:
        for row_value in rows:
            row = int(row_value)
            try:
                increment = joint_ou_increment(
                    drag_rate_s_inv[row : row + 1],
                    thermal_velocity_variance_m2_s2[row : row + 1],
                    duration_s[row : row + 1],
                    standard_normal[row : row + 1],
                )
            except FloatingPointError:
                valid[row] = False
            else:
                position[row] = increment.position_m[0]
                velocity[row] = increment.velocity_m_s[0]
    else:
        if all_valid:
            return increment, valid
        position[rows] = increment.position_m
        velocity[rows] = increment.velocity_m_s
    return JointOuIncrement(position, velocity), valid


def _split_joint_ou_increment_particle_local(
    parent: JointOuIncrement,
    drag_rate_s_inv: np.ndarray,
    thermal_velocity_variance_m2_s2: np.ndarray,
    duration_s: np.ndarray,
    standard_normal: np.ndarray,
    numerically_valid: np.ndarray,
) -> tuple[JointOuIncrement, JointOuIncrement, np.ndarray]:
    """Split valid OU rows while retaining invalid rows as inert placeholders."""

    valid = np.asarray(numerically_valid, dtype=np.bool_).copy()
    count = int(valid.size)
    rows = np.flatnonzero(valid)
    left_position = np.zeros((count, 2), dtype="<f8")
    left_velocity = np.zeros((count, 2), dtype="<f8")
    right_position = np.zeros((count, 2), dtype="<f8")
    right_velocity = np.zeros((count, 2), dtype="<f8")
    if not rows.size:
        return (
            JointOuIncrement(left_position, left_velocity),
            JointOuIncrement(right_position, right_velocity),
            valid,
        )
    all_valid = rows.size == count
    selected_parent = (
        parent
        if all_valid
        else JointOuIncrement(parent.position_m[rows], parent.velocity_m_s[rows])
    )
    selected_rate = drag_rate_s_inv if all_valid else drag_rate_s_inv[rows]
    selected_thermal = (
        thermal_velocity_variance_m2_s2 if all_valid else thermal_velocity_variance_m2_s2[rows]
    )
    selected_duration = duration_s if all_valid else duration_s[rows]
    selected_normal = standard_normal if all_valid else standard_normal[rows]
    try:
        left, right = split_joint_ou_increment_half(
            selected_parent,
            selected_rate,
            selected_thermal,
            selected_duration,
            selected_normal,
        )
    except FloatingPointError:
        for row_value in rows:
            row = int(row_value)
            try:
                left, right = split_joint_ou_increment_half(
                    JointOuIncrement(
                        parent.position_m[row : row + 1],
                        parent.velocity_m_s[row : row + 1],
                    ),
                    drag_rate_s_inv[row : row + 1],
                    thermal_velocity_variance_m2_s2[row : row + 1],
                    duration_s[row : row + 1],
                    standard_normal[row : row + 1],
                )
            except FloatingPointError:
                valid[row] = False
            else:
                left_position[row] = left.position_m[0]
                left_velocity[row] = left.velocity_m_s[0]
                right_position[row] = right.position_m[0]
                right_velocity[row] = right.velocity_m_s[0]
    else:
        if all_valid:
            return left, right, valid
        left_position[rows] = left.position_m
        left_velocity[rows] = left.velocity_m_s
        right_position[rows] = right.position_m
        right_velocity[rows] = right.velocity_m_s
    return (
        JointOuIncrement(left_position, left_velocity),
        JointOuIncrement(right_position, right_velocity),
        valid,
    )


def _advance_joint_ou_particle_local(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    equilibrium_velocity_m_s: np.ndarray,
    drag_rate_s_inv: np.ndarray,
    duration_s: np.ndarray,
    increment: JointOuIncrement,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply OU rows together, falling back only after a numerical exception."""

    count = int(drag_rate_s_inv.size)
    valid = np.ones(count, dtype=np.bool_)
    result_position = np.asarray(position_m, dtype="<f8").copy()
    result_velocity = np.asarray(velocity_m_s, dtype="<f8").copy()
    try:
        return (
            *advance_joint_ou_with_increment(
                position_m,
                velocity_m_s,
                equilibrium_velocity_m_s,
                drag_rate_s_inv,
                duration_s,
                increment,
            ),
            valid,
        )
    except FloatingPointError:
        for row in range(count):
            try:
                advanced_position, advanced_velocity = advance_joint_ou_with_increment(
                    position_m[row : row + 1],
                    velocity_m_s[row : row + 1],
                    equilibrium_velocity_m_s[row : row + 1],
                    drag_rate_s_inv[row : row + 1],
                    duration_s[row : row + 1],
                    JointOuIncrement(
                        increment.position_m[row : row + 1],
                        increment.velocity_m_s[row : row + 1],
                    ),
                )
            except FloatingPointError:
                valid[row] = False
            else:
                result_position[row] = advanced_position[0]
                result_velocity[row] = advanced_velocity[0]
    return result_position, result_velocity, valid


def _brownian_normal_tensor(
    seed: int,
    particle_id: np.ndarray,
    macro_interval: int,
    *,
    root_interval: int,
    tree_level: int,
    tree_index: int,
    draw_kind: int,
) -> np.ndarray:
    """Return the two independent normal pairs needed by the 2-D OU update."""

    result = np.empty((particle_id.size, 2, 2), dtype="<f8")
    for component in range(2):
        result[:, component] = brownian_normal_pair_batch(
            seed,
            particle_id,
            macro_interval,
            root_interval,
            tree_level=tree_level,
            tree_index=tree_index,
            component=component,
            draw_kind=draw_kind,
        )
    return result


def _advance_macro_proposal(
    prepared: _PreparedRun,
    proposal: StepProposal | None,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    event_statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
    writer: ResultWriter,
    event_buffer: _BoundaryEventBuffer,
    failure_buffer: _FailureEventBuffer,
) -> np.ndarray | None:
    """Advance one prepared macro proposal through its single production path."""

    if proposal is None:
        return None
    if _uses_curved_event_path(prepared):
        return _advance_curved_events(
            prepared,
            proposal,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            start_contact_state,
            replay,
            event_statistics,
            failure_reason_code,
            writer,
            event_buffer,
            failure_buffer,
        )
    if prepared.geometry.facet_count or prepared.case.data.coordinate_system == "axisymmetric_rz":
        _advance_exact_paths(
            prepared,
            proposal,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            exact_origin_time_s,
            exact_origin_position_m,
            exact_origin_velocity_m_s,
            start_contact_state,
            replay,
            event_statistics,
            failure_reason_code,
            writer,
            event_buffer,
            failure_buffer,
        )
        return None

    valid_rows = _commit_boundaryless_proposal(
        prepared,
        proposal,
        active,
        position_m,
        velocity_m_s,
        charge_number,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failure_buffer,
    )
    # A boundaryless Cartesian proposal is still one accepted particle piece
    # per committed row.  Keep the cumulative-work cadence proportional to
    # cohort size instead of counting only the outer macro-step.
    event_statistics.accepted_particle_pieces += int(valid_rows.size)
    _record_replay_rows(replay, proposal, valid_rows, prepared.case.data.coordinate_system)
    return None


def _prepare(case: SimulationCase) -> _PreparedRun:
    spec = case.spec
    _validate_configuration(case)
    memory_limit_bytes = spec.resources.memory_limit_mb * 1024 * 1024
    try:
        physics = resolve_physics_plan(spec.physics.models, case.data.coordinate_system)
        _validate_langevin_configuration(case, physics)
        expected_particle_count = source_particle_count(case)
        early_required_bytes = early_memory_requirement_bytes(
            canonical_data_bytes=case.data_footprint.numeric_array_bytes,
            particle_count=expected_particle_count,
            requires_stage_evaluation=physics.requires_stage_evaluation,
            writer_reserve_bytes=RESULT_WRITER_RESERVE_BYTES,
        )
        if early_required_bytes > memory_limit_bytes:
            raise EngineError(
                "predicted minimum solver footprint exceeds resources.memory_limit_mb"
            )
        geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
        schedule = realize_sources(case, geometry)
        if schedule.particle_count != expected_particle_count:
            raise EngineError("source realization changed the preflight particle count")
        _validate_source_facets(schedule, geometry)
        requirements = {
            item.name: RequiredFieldMetadata(
                item.unit,
                item.components,
                item.stored_basis,
                item.positive,
            )
            for item in physics.required_fields
        }
        fields = prepare_required_fields(case.data, requirements)
        primitive_ranges = {
            item.name: PrimitiveRange(
                *fields.component_bounds(item.name),
                fields.constant_value(item.name),
            )
            for item in physics.required_fields
        }
        runtime = prepare_physics_runtime(
            plan=physics,
            coordinate_system=case.data.coordinate_system,
            mass_kg=schedule.mass_kg,
            drag_diameter_m=schedule.drag_diameter_m,
            electrostatic_radius_m=schedule.electrostatic_radius_m,
            displaced_volume_m3=schedule.displaced_volume_m3,
            charge_number=schedule.charge_number,
            primitive_ranges=primitive_ranges,
        )
        constant_acceleration = runtime.constant_acceleration_m_s2
        maximum_dt_over_tau = runtime.maximum_dt_over_tau(spec.time.dt_s)
        maximum_dt_charge_lipschitz = runtime.maximum_dt_charge_lipschitz(spec.time.dt_s)
        _validate_integrator_stability(
            spec.solver.integrator,
            maximum_dt_over_tau,
            maximum_dt_charge_lipschitz,
        )
        _validate_curved_path_capability(
            spec.solver.integrator,
            physics,
            constant_acceleration,
            geometry,
            fields,
        )
        if geometry.facet_count:
            rules = _prepare_boundary_rules(case)
            _validate_table_starts(case, schedule, geometry)
        else:
            rules = ()
    except (
        BoundaryLawError,
        FieldLocationError,
        GeometryPreparationError,
        EventLocationError,
        PhysicsConfigurationError,
        PhysicsEvaluationError,
        ValueError,
    ) as error:
        raise EngineError(str(error)) from error
    frame_times = (
        () if spec.output.trajectories is None else spec.output.trajectories.explicit_times_s
    )
    probe_times = () if spec.output.probes is None else spec.output.probes.explicit_times_s
    probe_particle_index = _resolve_probe_particle_index(case, schedule)
    memory_plan, field_cell_hint_bytes, event_candidate_capacity = _prepare_memory_plan(
        case,
        schedule,
        geometry,
        fields,
        runtime,
        physics,
        frame_times,
        probe_times,
        probe_particle_index,
        memory_limit_bytes,
    )
    dynamics = _StageDynamics(
        runtime,
        fields,
        case.data.coordinate_system,
        (np.full(schedule.particle_count, -1, dtype="<i8") if field_cell_hint_bytes else None),
        tuple(fields.allocate_workspace(memory_plan.slab_particles) for _ in range(4)),
        tuple(PhysicsRuntimeWorkspace.allocate(memory_plan.slab_particles) for _ in range(4)),
    )
    dense_certificate = _uses_rk4_dense_certificate(case, physics, constant_acceleration)
    certificate_workspace = _ApplicabilityCertificateWorkspace.allocate(
        memory_plan.slab_particles if dense_certificate else 0,
        spec.solver.event.max_refinements,
    )
    return _PreparedRun(
        case,
        schedule,
        frame_times,
        probe_times,
        probe_particle_index,
        geometry,
        rules,
        prepare_boundary_rules(rules),
        physics,
        fields,
        dynamics,
        constant_acceleration,
        maximum_dt_over_tau,
        maximum_dt_charge_lipschitz,
        memory_plan,
        event_candidate_capacity,
        certificate_workspace,
    )


def _prepare_memory_plan(
    case: SimulationCase,
    schedule: ParticleSchedule,
    geometry: PreparedGeometry,
    fields: PreparedFieldSet,
    runtime: PhysicsRuntime,
    physics: PhysicsPlan,
    frame_times_s: tuple[float, ...],
    probe_times_s: tuple[float, ...],
    probe_particle_index: np.ndarray,
    memory_limit_bytes: int,
) -> tuple[CpuMemoryPlan, int, int]:
    """Resolve the single solver-owned slab plan from prepared array owners."""

    field_cell_hint_bytes = (
        schedule.particle_count * np.dtype("<i8").itemsize if fields.uses_cell_hint else 0
    )
    canonical_data_bytes = max(
        case.data_footprint.numeric_array_bytes,
        resident_array_bytes(case.data),
    )
    event_paths = geometry.facet_count > 0 or case.data.coordinate_system == "axisymmetric_rz"
    event_candidate_capacity = _event_candidate_capacity(
        geometry,
        memory_limit_bytes,
        enabled=event_paths,
    )
    event_staging_capacity = event_candidate_capacity // 2
    dense_certificate = _uses_rk4_dense_certificate(
        case,
        physics,
        runtime.constant_acceleration_m_s2,
    )
    local_range_certificate = dense_certificate or (
        case.spec.solver.integrator == "exponential_midpoint"
        and physics.requires_stage_evaluation
        and runtime.constant_acceleration_m_s2 is None
    )
    memory_plan = plan_cpu_memory(
        limit_bytes=memory_limit_bytes,
        particle_count=schedule.particle_count,
        canonical_data_bytes=canonical_data_bytes,
        prepared_geometry_bytes=_geometry_memory_bytes(geometry),
        particle_schedule_bytes=_schedule_memory_bytes(schedule),
        physics_runtime_bytes=runtime.bound_array_nbytes,
        field_runtime_bytes=fields.prepared_nbytes + field_cell_hint_bytes,
        geometry_preparation_transient_bytes=(
            geometry.volume_bvh_build_transient_nbytes
            + (
                min(schedule.particle_count, _PREPARE_SCAN_BATCH_SIZE) * np.dtype(np.bool_).itemsize
                if geometry.facet_count
                else 0
            )
        ),
        field_preparation_transient_bytes=fields.preparation_transient_nbytes,
        probe_index_bytes=probe_particle_index.nbytes,
        output_buffer_bytes=_output_buffer_bytes(
            schedule,
            frame_times_s,
            probe_times_s,
            probe_particle_index,
        ),
        replay_work_bytes=_replay_work_bytes(
            case,
            schedule,
            frame_times_s,
            probe_times_s,
            probe_particle_index,
        ),
        writer_reserve_bytes=RESULT_WRITER_RESERVE_BYTES,
        requires_stage_evaluation=physics.requires_stage_evaluation,
        # A yielded locator result remains live while the next broad-phase
        # columns and the synchronously packed output candidate column coexist.
        geometry_query_scratch_bytes=(3 * event_candidate_capacity * np.dtype("<i8").itemsize),
        dense_path_bytes_per_particle=(176 if dense_certificate else 0),
        stochastic_tree_work_bytes_per_particle=(_stochastic_tree_work_bytes_per_particle(physics)),
        event_work_bytes_per_particle=_event_work_bytes_per_particle(
            case,
            event_paths=event_paths,
        ),
        certificate_work_bytes_per_particle=_certificate_work_bytes_per_particle(
            case,
            dense_enabled=dense_certificate,
            local_range_enabled=local_range_certificate,
        ),
        release_work_bytes_per_particle=_surface_release_work_bytes_per_particle(
            schedule,
            event_paths=event_paths,
        ),
        event_candidate_capacity=event_candidate_capacity,
        event_staging_capacity=event_staging_capacity,
        event_staging_bytes_per_row=(_EVENT_STAGING_BYTES_PER_ROW if event_staging_capacity else 0),
        event_staging_fixed_bytes=(
            event_candidate_capacity * _EVENT_STAGING_BYTES_PER_CANDIDATE
            + _EVENT_STAGING_FIXED_BYTES
            if event_staging_capacity
            else 0
        ),
        failure_staging_bytes_per_particle=_FAILURE_STAGING_BYTES_PER_PARTICLE,
    )
    if memory_plan.planned_bytes > memory_limit_bytes or (
        schedule.particle_count and memory_plan.slab_particles < 1
    ):
        raise EngineError("predicted solver memory exceeds resources.memory_limit_mb")
    return memory_plan, field_cell_hint_bytes, event_candidate_capacity


def _frame_stop(frame_times_s: tuple[float, ...], cursor: int, macro_end_s: float) -> int:
    """Return the first requested frame strictly after this macro step."""

    stop = cursor
    while stop < len(frame_times_s) and frame_times_s[stop] <= macro_end_s:
        stop += 1
    return stop


def _resolve_probe_particle_index(
    case: SimulationCase,
    schedule: ParticleSchedule,
) -> np.ndarray:
    probes = case.spec.output.probes
    if probes is None:
        return np.empty(0, dtype="<i8")
    requested = np.asarray(probes.particle_ids, dtype="<i8")
    indices = np.searchsorted(schedule.particle_id, requested)
    present = indices < schedule.particle_count
    if bool(present.any()):
        valid_rows = np.flatnonzero(present)
        present[valid_rows] = schedule.particle_id[indices[valid_rows]] == requested[valid_rows]
    missing = requested[~present].tolist()
    if missing:
        raise EngineError(f"output.probes references unknown particle IDs: {missing}")
    return np.sort(np.asarray(indices, dtype="<i8"))


def _validate_configuration(case: SimulationCase) -> None:
    spec = case.spec
    if spec.solver.backend != "cpu" or spec.solver.integrator not in {
        "rk4_fixed",
        "exponential_midpoint",
        "ou_langevin",
    }:
        raise EngineError(
            "the engine supports integrator=rk4_fixed, exponential_midpoint, or "
            "ou_langevin with backend=cpu only"
        )
    if spec.solver.event.corner_policy != "priority_then_combined_normal_v1":
        raise EngineError(
            "the Stage 1A engine supports corner_policy=priority_then_combined_normal_v1 only"
        )
    coordinate_pair = (case.data.coordinate_system, spec.motion.mode)
    valid_pairs = {
        ("cartesian_xy", "cartesian_xy"),
        ("axisymmetric_rz", "axisymmetric_rz_meridional"),
    }
    if coordinate_pair not in valid_pairs:
        raise EngineError(
            "data coordinate_system and particle motion.mode are not a supported pair"
        )


def _validate_langevin_configuration(case: SimulationCase, physics: PhysicsPlan) -> None:
    """Keep the first Brownian slice narrow enough to have one clear meaning."""

    selected = physics.noise is not None
    if (case.spec.solver.integrator == "ou_langevin") != selected:
        raise EngineError(
            "integrator=ou_langevin and physics.noise=inertial_langevin_fdt "
            "must be selected together"
        )
    if not selected:
        return
    unsupported = sorted(
        boundary.boundary_group
        for boundary in case.spec.boundaries
        if boundary.law not in {"stick", "escape", "hold"}
    )
    if unsupported:
        raise EngineError(
            f"inertial Langevin supports terminal stick/escape/hold boundaries only: {unsupported}"
        )


def _validate_integrator_stability(
    integrator: str,
    maximum_dt_over_tau: float,
    maximum_dt_charge_lipschitz: float,
) -> None:
    """Apply only the stability restriction owned by the selected method."""

    if integrator == "rk4_fixed" and maximum_dt_over_tau >= 2.5:
        raise EngineError("rk4_fixed requires dt * drag_velocity_lipschitz < 2.5")
    if integrator == "ou_langevin" and maximum_dt_over_tau > MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT:
        raise EngineError(
            f"ou_langevin requires gamma * dt <= {MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT:.0e}"
        )
    if integrator == "rk4_fixed" and maximum_dt_charge_lipschitz > 0.5:
        raise EngineError(
            "continuous charge requires dt * charge_lipschitz <= 0.5 for explicit integration"
        )


def _validate_curved_path_capability(
    integrator: str,
    physics: PhysicsPlan,
    constant_acceleration_m_s2: np.ndarray | None,
    geometry: PreparedGeometry,
    fields: PreparedFieldSet,
) -> None:
    """Require a continuous support certificate for a reintegrated curved path."""

    if not physics.requires_stage_evaluation or constant_acceleration_m_s2 is not None:
        return
    if geometry.facet_count:
        return
    if fields.regular_support_box() is None:
        method = "general RK4" if integrator == "rk4_fixed" else "exponential midpoint"
        raise EngineError(f"boundaryless {method} requires a fully supported regular field box")


def _certify_curved_path(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
) -> CurvedPathEnclosure | None:
    """Build the selected integrator's safety enclosure before arbitration."""

    if (
        not prepared.physics.requires_stage_evaluation
        or prepared.constant_acceleration_m_s2 is not None
    ):
        return None
    if prepared.case.spec.solver.integrator == "rk4_fixed":
        return enclose_rk4_path(
            particle_index,
            start_time_s,
            target_time_s,
            start_position_m,
            start_velocity_m_s,
            acceleration_abs_bounder=prepared.dynamics.acceleration_abs_upper,
        )
    rate_upper, target_upper = prepared.dynamics.linear_relaxation_abs_bounds(particle_index)
    global_enclosure = enclose_exponential_midpoint_path(
        particle_index,
        start_time_s,
        target_time_s,
        start_position_m,
        start_velocity_m_s,
        linear_drag_rate_upper_s_inv=rate_upper,
        target_velocity_abs_upper_m_s=target_upper,
        additive_acceleration_abs_bounder=(prepared.dynamics.additive_acceleration_abs_upper),
    )
    if prepared.dynamics.runtime.localizable_external_base_abs_upper_m_s2 is None:
        return global_enclosure
    return _tighten_exponential_midpoint_enclosure(
        prepared,
        particle_index,
        start_time_s,
        target_time_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        rate_upper,
        target_upper,
        global_enclosure,
    )


def _tighten_exponential_midpoint_enclosure(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
    rate_upper_s_inv: np.ndarray,
    target_velocity_abs_upper_m_s: np.ndarray,
    fallback: CurvedPathEnclosure,
) -> CurvedPathEnclosure:
    """Tighten only rows whose global enclosure cannot finish certification."""

    duration_s = target_time_s - start_time_s
    charge_lower, charge_upper = prepared.dynamics.prepared_charge_invariant_interval(
        start_charge_number
    )
    global_applicable, global_status = prepared.dynamics.global_continuous_applicability_batch(
        particle_index,
        fallback.velocity_lower_m_s,
        fallback.velocity_upper_m_s,
        charge_lower,
        charge_upper,
    )
    support_inside = _curved_position_bounds_inside_support(
        prepared,
        fallback.position_lower_m,
        fallback.position_upper_m,
        np.zeros(particle_index.size, dtype=np.bool_),
        np.zeros(particle_index.size, dtype="<f8"),
    )
    globally_proven = global_applicable & (global_status == CONTINUOUS_APPLICABILITY_OK)
    selected = np.flatnonzero((duration_s > 0.0) & (~globally_proven | ~support_inside)).astype(
        "<i8", copy=False
    )
    if not selected.size:
        return fallback
    predictor_box = exponential_frozen_start_predictor_enclosure(
        particle_index[selected],
        start_time_s[selected],
        0.5 * duration_s[selected],
        start_position_m[selected],
        start_velocity_m_s[selected],
        start_charge_number[selected],
        evaluator=prepared.dynamics.evaluate_relaxation,
    )
    predictor = predictor_box.sample
    valid = predictor.numerical_status == NUMERICAL_STATUS_OK
    valid &= predictor.support_inside & predictor.applicability_inside
    if not bool(valid.any()):
        return fallback

    predictor_position_lower = predictor_box.path.position_lower_m
    predictor_position_upper = predictor_box.path.position_upper_m
    predictor_velocity_lower = predictor_box.path.velocity_lower_m_s
    predictor_velocity_upper = predictor_box.path.velocity_upper_m_s
    predictor_charge_lower = predictor_box.charge_lower_number
    predictor_charge_upper = predictor_box.charge_upper_number
    valid &= predictor_box.path.numerical_status == NUMERICAL_STATUS_OK
    finite = np.isfinite(predictor_position_lower).all(axis=1)
    finite &= np.isfinite(predictor_position_upper).all(axis=1)
    finite &= np.isfinite(predictor_velocity_lower).all(axis=1)
    finite &= np.isfinite(predictor_velocity_upper).all(axis=1)
    finite &= np.isfinite(predictor_charge_lower) & np.isfinite(predictor_charge_upper)
    valid &= finite
    if not bool(valid.any()):
        return fallback

    try:
        local_bound, local_applicable, local_status, range_available = (
            prepared.dynamics.local_additive_acceleration_abs_upper_batch(
                particle_index[selected],
                predictor_position_lower,
                predictor_position_upper,
                predictor_velocity_lower,
                predictor_velocity_upper,
                predictor_charge_lower,
                predictor_charge_upper,
            )
        )
    except (FieldLocationError, PhysicsEvaluationError, ValueError):
        return fallback
    valid &= range_available & local_applicable
    valid &= local_status == CONTINUOUS_APPLICABILITY_OK
    local_rows = np.flatnonzero(valid).astype("<i8", copy=False)
    if not local_rows.size:
        return fallback
    local = enclose_exponential_midpoint_path_from_stage_bounds(
        particle_index[selected[local_rows]],
        start_time_s[selected[local_rows]],
        target_time_s[selected[local_rows]],
        start_position_m[selected[local_rows]],
        start_velocity_m_s[selected[local_rows]],
        linear_drag_rate_upper_s_inv=rate_upper_s_inv[selected[local_rows]],
        target_velocity_abs_upper_m_s=target_velocity_abs_upper_m_s[selected[local_rows]],
        additive_start_abs_upper_m_s2=local_bound[local_rows],
        additive_midpoint_abs_upper_m_s2=local_bound[local_rows],
        numerical_status=np.full(local_rows.size, NUMERICAL_STATUS_OK, dtype=np.uint8),
    )
    result = CurvedPathEnclosure(
        fallback.position_lower_m.copy(),
        fallback.position_upper_m.copy(),
        fallback.velocity_lower_m_s.copy(),
        fallback.velocity_upper_m_s.copy(),
        fallback.numerical_status.copy(),
    )
    local_ok = local.numerical_status == NUMERICAL_STATUS_OK
    accepted_rows = selected[local_rows[local_ok]]
    if accepted_rows.size:
        local_position_lower = np.maximum(
            fallback.position_lower_m[accepted_rows],
            local.position_lower_m[local_ok],
        )
        local_position_upper = np.minimum(
            fallback.position_upper_m[accepted_rows],
            local.position_upper_m[local_ok],
        )
        local_velocity_lower = np.maximum(
            fallback.velocity_lower_m_s[accepted_rows],
            local.velocity_lower_m_s[local_ok],
        )
        local_velocity_upper = np.minimum(
            fallback.velocity_upper_m_s[accepted_rows],
            local.velocity_upper_m_s[local_ok],
        )
        ordered = (local_position_lower <= local_position_upper).all(axis=1)
        ordered &= (local_velocity_lower <= local_velocity_upper).all(axis=1)
        accepted = accepted_rows[ordered]
        result.position_lower_m[accepted] = local_position_lower[ordered]
        result.position_upper_m[accepted] = local_position_upper[ordered]
        result.velocity_lower_m_s[accepted] = local_velocity_lower[ordered]
        result.velocity_upper_m_s[accepted] = local_velocity_upper[ordered]
    return result


def _build_step_proposal(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
    path_enclosure: CurvedPathEnclosure | None,
) -> StepProposal:
    """Dispatch one proposal without creating a second production engine."""

    constant_acceleration = prepared.constant_acceleration_m_s2
    use_exact_specialization = (
        not prepared.physics.requires_stage_evaluation or constant_acceleration is not None
    )
    if prepared.case.spec.solver.integrator == "rk4_fixed" or use_exact_specialization:
        return rk4_step(
            particle_index,
            start_time_s,
            target_time_s,
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
            requires_stage_evaluation=prepared.physics.requires_stage_evaluation,
            evaluator=(prepared.dynamics if prepared.physics.requires_stage_evaluation else None),
            constant_acceleration_m_s2=(
                None if constant_acceleration is None else constant_acceleration[particle_index]
            ),
            path_enclosure=path_enclosure,
        )
    if prepared.case.spec.solver.integrator == "ou_langevin":
        raise EngineError("OU Langevin proposals are built from conditioned increments")
    return exponential_midpoint_step(
        particle_index,
        start_time_s,
        target_time_s,
        start_position_m,
        start_velocity_m_s,
        start_charge_number,
        evaluator=prepared.dynamics.evaluate_relaxation,
        path_enclosure=path_enclosure,
    )


def _propose_root_batch(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    end_time_s: float,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
) -> StepProposal | None:
    """Build one macro proposal without a particle-wise retry path."""

    if not particle_index.size:
        return None
    target_time_s = np.full(particle_index.size, end_time_s, dtype="<f8")
    try:
        path_enclosure = _certify_curved_path(
            prepared,
            particle_index,
            start_time_s,
            target_time_s,
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
        )
        proposal = _build_step_proposal(
            prepared,
            particle_index,
            start_time_s,
            target_time_s,
            start_position_m,
            start_velocity_m_s,
            start_charge_number,
            path_enclosure,
        )
    except (FieldLocationError, PhysicsEvaluationError, ValueError) as error:
        raise EngineError("root proposal batch could not be evaluated") from error
    return proposal


def _quadratic_path_inside_regular_support(
    fields: PreparedFieldSet,
    proposal: StepProposal,
    local_rows: np.ndarray,
) -> np.ndarray:
    """Return the per-row continuous-support verdict for exact parabolas."""

    support_box = fields.regular_support_box()
    if support_box is None or proposal.constant_acceleration_m_s2 is None:
        raise EngineError("quadratic boundaryless motion lost its regular support certificate")

    rows = np.asarray(local_rows, dtype="<i8")
    if rows.ndim != 1:
        raise EngineError("quadratic support rows must be one-dimensional")
    selected = replace(
        proposal,
        particle_index=proposal.particle_index[rows],
        start_time_s=proposal.start_time_s[rows],
        target_time_s=proposal.target_time_s[rows],
        start_position_m=proposal.start_position_m[rows],
        start_velocity_m_s=proposal.start_velocity_m_s[rows],
        start_charge_number=proposal.start_charge_number[rows],
        end_position_m=proposal.end_position_m[rows],
        end_velocity_m_s=proposal.end_velocity_m_s[rows],
        end_charge_number=proposal.end_charge_number[rows],
        end_field_cell_id=(
            None if proposal.end_field_cell_id is None else proposal.end_field_cell_id[rows]
        ),
        support_inside=proposal.support_inside[rows],
        applicability_inside=proposal.applicability_inside[rows],
        numerical_status=proposal.numerical_status[rows],
        constant_acceleration_m_s2=proposal.constant_acceleration_m_s2[rows],
    )
    lower_bound, upper_bound = support_box
    interval = selected.quadratic_position_interval()
    inside = (interval.lower_m >= lower_bound).all(axis=1)
    inside &= (interval.upper_m <= upper_bound).all(axis=1)
    return inside


def _commit_boundaryless_proposal(
    prepared: _PreparedRun,
    proposal: StepProposal,
    active: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failures: _FailureEventBuffer,
) -> np.ndarray:
    """Commit valid rows and localize row-specific field/model failures."""

    codes = _proposal_numerical_failure_codes(proposal.numerical_status)
    if proposal.path_kind == "quadratic_exact":
        eligible = np.flatnonzero(codes == 0).astype("<i8", copy=False)
        if eligible.size:
            support = _quadratic_path_inside_regular_support(
                prepared.fields,
                proposal,
                eligible,
            )
            codes[eligible[~support]] = _FAILURE_FIELD_SUPPORT
    elif _is_reintegrated_path(proposal):
        rows = np.arange(proposal.particle_index.size, dtype="<i8")
        codes = _curved_row_failure_codes(
            prepared,
            proposal,
            rows,
            start_contact_certified=np.zeros(rows.size, dtype=np.bool_),
        )

    if not _is_reintegrated_path(proposal):
        stage_support_failure = (codes == 0) & ~proposal.support_inside
        codes[stage_support_failure] = _FAILURE_FIELD_SUPPORT
        stage_model_failure = (codes == 0) & ~proposal.applicability_inside
        codes[stage_model_failure] = _FAILURE_MODEL_APPLICABILITY
    inactive = ~active[proposal.particle_index]
    codes[inactive] = 0

    for local_row in np.flatnonzero(codes):
        particle_index = int(proposal.particle_index[local_row])
        _mark_particle_failed(
            particle_index,
            _proposal_row_failure(proposal, int(local_row), codes[local_row]),
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
        )
    valid = np.flatnonzero((codes == 0) & active[proposal.particle_index]).astype("<i8", copy=False)
    _commit_proposal_rows(
        prepared,
        proposal,
        valid,
        position_m,
        velocity_m_s,
        charge_number,
    )
    return valid


def _proposal_numerical_failure_codes(numerical_status: np.ndarray) -> np.ndarray:
    """Map the shared hot-path status vocabulary to public failure reasons."""

    status = np.asarray(numerical_status)
    if status.ndim != 1 or status.dtype != np.uint8:
        raise EngineError("proposal numerical status has an invalid representation")
    known = (
        (status == NUMERICAL_STATUS_OK)
        | (status == FIELD_NUMERICAL_FAILURE)
        | (status == PHYSICS_NUMERICAL_FAILURE)
        | (status == INTEGRATOR_NUMERICAL_FAILURE)
        | (status == INTEGRATOR_ACCURACY_FAILURE)
    )
    if not bool(known.all()):
        raise EngineError("proposal numerical status contains an unknown code")
    result = np.zeros(status.size, dtype="<u2")
    result[status == FIELD_NUMERICAL_FAILURE] = _FAILURE_FIELD_SUPPORT
    nonfinite = (status == PHYSICS_NUMERICAL_FAILURE) | (status == INTEGRATOR_NUMERICAL_FAILURE)
    result[nonfinite] = _FAILURE_NONFINITE_PHYSICS
    result[status == INTEGRATOR_ACCURACY_FAILURE] = _FAILURE_INTEGRATOR_ACCURACY
    return result


def _proposal_row_failure(
    proposal: StepProposal,
    local_row: int,
    reason_code: np.uint16,
) -> _ParticleFailure:
    """Create one failure at the last accepted state preceding a trial row."""

    return _ParticleFailure(
        reason_code,
        float(proposal.start_time_s[local_row]),
        proposal.start_position_m[local_row],
        proposal.start_velocity_m_s[local_row],
        float(proposal.start_charge_number[local_row]),
    )


def _uses_curved_event_path(prepared: _PreparedRun) -> bool:
    return (
        prepared.physics.requires_stage_evaluation
        and prepared.constant_acceleration_m_s2 is None
        and (
            prepared.geometry.facet_count > 0
            or prepared.case.data.coordinate_system == "axisymmetric_rz"
        )
    )


def _is_reintegrated_path(proposal: StepProposal) -> bool:
    return proposal.path_kind in {
        "cubic_hermite",
        "rk4_dense",
        "exponential_midpoint_reintegrated",
    }


def _curved_row_failure_codes(
    prepared: _PreparedRun,
    proposal: StepProposal,
    local_rows: np.ndarray,
    *,
    event_slack_m: float | np.ndarray = 0.0,
    start_contact_certified: np.ndarray | None = None,
) -> np.ndarray:
    """Classify accepted curved rows without turning local invalidity into run failure."""

    enclosure = proposal.path_enclosure
    rows = np.asarray(local_rows, dtype=np.int64)
    if not _is_reintegrated_path(proposal) or enclosure is None:
        raise EngineError("reintegrated curved piece lost its continuous safety enclosure")
    slack = np.asarray(event_slack_m, dtype="<f8")
    if slack.ndim == 0:
        slack = np.full(rows.size, float(slack), dtype="<f8")
    if slack.shape != (rows.size,) or not bool(np.isfinite(slack).all()):
        raise EngineError("curved event slack is invalid")
    if bool((slack < 0.0).any()):
        raise EngineError("curved event slack is invalid")
    contact_certified = (
        np.zeros(rows.size, dtype=np.bool_)
        if start_contact_certified is None
        else np.asarray(start_contact_certified, dtype=np.bool_)
    )
    if contact_certified.shape != (rows.size,):
        raise EngineError("curved start-contact certificate shape is invalid")
    codes = _proposal_numerical_failure_codes(proposal.numerical_status[rows])
    eligible = np.flatnonzero(codes == 0).astype("<i8", copy=False)
    if proposal.path_kind == "rk4_dense":
        # P19 localizes model applicability over the immutable dense state,
        # but it does not replace revision-3b's path-support certificate.  The
        # latter encloses every shortened same-integrator state and therefore
        # remains the authority for field support and event-prefix slack.
        if eligible.size:
            support_inside = _curved_enclosure_inside_support(
                prepared,
                proposal,
                rows[eligible],
                contact_certified[eligible],
                slack[eligible],
            )
            zero_slack = slack[eligible] == 0.0
            support_inside[zero_slack] &= proposal.support_inside[rows[eligible[zero_slack]]]
            failed = eligible[~support_inside]
            codes[failed] = _FAILURE_FIELD_SUPPORT
            eligible = eligible[support_inside]
        stage_model_failure = slack[eligible] == 0.0
        stage_model_failure &= ~proposal.applicability_inside[rows[eligible]]
        codes[eligible[stage_model_failure]] = _FAILURE_MODEL_APPLICABILITY
        eligible = eligible[~stage_model_failure]
        if eligible.size:
            codes[eligible] = _rk4_dense_certificate_failure_codes(
                prepared,
                proposal,
                rows[eligible],
                event_slack_m=slack[eligible],
                start_contact_certified=contact_certified[eligible],
            )
        return codes

    support_inside = _curved_enclosure_inside_support(
        prepared,
        proposal,
        rows,
        contact_certified,
        slack,
    )

    zero_slack = slack == 0.0
    support_inside[zero_slack] &= proposal.support_inside[rows[zero_slack]]
    codes[(codes == 0) & ~support_inside] = _FAILURE_FIELD_SUPPORT
    eligible = np.flatnonzero(codes == 0).astype("<i8", copy=False)
    stage_model_failure = slack[eligible] == 0.0
    stage_model_failure &= ~proposal.applicability_inside[rows[eligible]]
    codes[eligible[stage_model_failure]] = _FAILURE_MODEL_APPLICABILITY
    eligible = eligible[~stage_model_failure]
    if eligible.size:
        applicable, numerical_failure = _continuous_applicability_verdicts(
            prepared,
            proposal,
            rows[eligible],
        )
        eligible_codes = codes[eligible]
        eligible_codes[numerical_failure] = _FAILURE_NONFINITE_PHYSICS
        model_failure = (~numerical_failure) & ~applicable
        eligible_codes[model_failure] = _FAILURE_MODEL_APPLICABILITY
        codes[eligible] = eligible_codes
    return codes


def _rk4_dense_certificate_failure_codes(
    prepared: _PreparedRun,
    proposal: StepProposal,
    local_rows: np.ndarray,
    *,
    event_slack_m: np.ndarray,
    start_contact_certified: np.ndarray,
) -> np.ndarray:
    """Prove local applicability with bounded dyadic interval work.

    Subdivision restricts one immutable dense polynomial.  It never rebuilds
    an RK proposal and never commits a child endpoint.  A conservative range
    miss is distinguished from an actual invalid midpoint or endpoint.
    """

    rows = np.asarray(local_rows, dtype="<i8")
    count = int(rows.size)
    workspace = prepared.certificate_workspace
    if count > workspace.stack_top.size:
        raise EngineError("applicability certificate exceeds its planned slab capacity")
    budget = prepared.case.spec.solver.event.max_refinements
    stack_start = workspace.start_time_s[:count]
    stack_target = workspace.target_time_s[:count]
    stack_top = workspace.stack_top[:count]
    split_count = workspace.split_count[:count]
    codes = workspace.failure_code[:count]
    stack_start[:, 0] = proposal.start_time_s[rows]
    stack_target[:, 0] = proposal.target_time_s[rows]
    stack_top[:] = 0
    split_count[:] = 0
    codes[:] = 0

    while True:
        certificate_rows = np.flatnonzero((stack_top >= 0) & (codes == 0)).astype(
            "<i8",
            copy=False,
        )
        if not certificate_rows.size:
            break
        top = stack_top[certificate_rows]
        interval_start = stack_start[certificate_rows, top]
        interval_target = stack_target[certificate_rows, top]
        proposal_rows = rows[certificate_rows]
        dense = proposal.rk4_dense_subinterval(
            proposal_rows,
            interval_start,
            interval_target,
        )
        interval_codes = _proposal_numerical_failure_codes(dense.numerical_status)
        numerical = interval_codes != 0
        if bool(numerical.any()):
            failed_rows = certificate_rows[numerical]
            codes[failed_rows] = interval_codes[numerical]
            stack_top[failed_rows] = -1

        eligible_offsets = np.flatnonzero(~numerical).astype("<i8", copy=False)
        if not eligible_offsets.size:
            continue
        eligible_certificate_rows = certificate_rows[eligible_offsets]
        eligible_proposal_rows = proposal_rows[eligible_offsets]
        eligible_start = interval_start[eligible_offsets]
        eligible_target = interval_target[eligible_offsets]
        rightmost = eligible_target == proposal.target_time_s[eligible_proposal_rows]
        leftmost = eligible_start == proposal.start_time_s[eligible_proposal_rows]
        interval_slack = np.where(
            rightmost,
            event_slack_m[eligible_certificate_rows],
            0.0,
        )
        interval_contact = leftmost & start_contact_certified[eligible_certificate_rows]
        continuous_support = _curved_position_bounds_inside_support(
            prepared,
            dense.position_lower_m[eligible_offsets],
            dense.position_upper_m[eligible_offsets],
            interval_contact,
            interval_slack,
        )
        range_certified = _rk4_dense_interval_range_certificate(
            prepared,
            proposal,
            eligible_proposal_rows,
            dense,
            eligible_offsets,
            continuous_support,
        )
        proven_rows = eligible_certificate_rows[range_certified]
        stack_top[proven_rows] -= 1

        # A local interval calculation can fail to certify without proving an
        # actual physics failure.  Only real dense-state samples below may
        # emit field/model/nonfinite reasons; otherwise subdivision exhausts
        # as the distinct indeterminate-certificate outcome.
        unresolved = ~range_certified
        _resolve_rk4_dense_unproven_intervals(
            prepared,
            proposal,
            eligible_certificate_rows[unresolved],
            eligible_proposal_rows[unresolved],
            eligible_start[unresolved],
            eligible_target[unresolved],
            budget,
            stack_start,
            stack_target,
            stack_top,
            split_count,
            codes,
        )

    return codes.copy()


def _rk4_dense_interval_range_certificate(
    prepared: _PreparedRun,
    proposal: StepProposal,
    proposal_rows: np.ndarray,
    dense: Rk4DenseEnclosure,
    dense_rows: np.ndarray,
    continuous_support: np.ndarray,
) -> np.ndarray:
    """Use the cheap global proof before constructing local field ranges."""

    range_certified = np.zeros(dense_rows.size, dtype=np.bool_)
    support_rows = np.flatnonzero(continuous_support).astype("<i8", copy=False)
    if not support_rows.size:
        return range_certified
    try:
        global_certified, global_status = prepared.dynamics.global_continuous_applicability_batch(
            proposal.particle_index[proposal_rows[support_rows]],
            dense.velocity_lower_m_s[dense_rows[support_rows]],
            dense.velocity_upper_m_s[dense_rows[support_rows]],
            dense.charge_lower_number[dense_rows[support_rows]],
            dense.charge_upper_number[dense_rows[support_rows]],
        )
    except (PhysicsEvaluationError, ValueError) as error:
        raise EngineError("global applicability certificate could not be evaluated") from error
    _require_applicability_certificate_result(
        global_certified,
        global_status,
        support_rows.size,
        "global",
    )
    global_proven = global_certified & (global_status == CONTINUOUS_APPLICABILITY_OK)
    range_certified[support_rows[global_proven]] = True
    local_rows = support_rows[~global_proven]
    if not local_rows.size:
        return range_certified
    try:
        certified, local_status, range_available = (
            prepared.dynamics.local_continuous_applicability_batch(
                proposal.particle_index[proposal_rows[local_rows]],
                dense.position_lower_m[dense_rows[local_rows]],
                dense.position_upper_m[dense_rows[local_rows]],
                dense.velocity_lower_m_s[dense_rows[local_rows]],
                dense.velocity_upper_m_s[dense_rows[local_rows]],
                dense.charge_lower_number[dense_rows[local_rows]],
                dense.charge_upper_number[dense_rows[local_rows]],
            )
        )
    except (FieldLocationError, PhysicsEvaluationError, ValueError) as error:
        raise EngineError("local applicability certificate could not be evaluated") from error
    _require_applicability_certificate_result(
        certified,
        local_status,
        local_rows.size,
        "local",
        range_available,
    )
    local_proven = range_available & certified
    local_proven &= local_status == CONTINUOUS_APPLICABILITY_OK
    range_certified[local_rows[local_proven]] = True
    return range_certified


def _require_applicability_certificate_result(
    certified: np.ndarray,
    status: np.ndarray,
    expected_size: int,
    label: str,
    range_available: np.ndarray | None = None,
) -> None:
    """Validate the one shared row-status vocabulary at a certificate boundary."""

    if certified.shape != (expected_size,) or status.shape != (expected_size,):
        raise EngineError(f"{label} applicability certificate has an invalid shape")
    if range_available is not None and range_available.shape != (expected_size,):
        raise EngineError(f"{label} applicability certificate has an invalid shape")
    known = (status == CONTINUOUS_APPLICABILITY_OK) | (
        status == CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    )
    if not bool(known.all()):
        raise EngineError(f"{label} applicability certificate returned an unknown status")


def _resolve_rk4_dense_unproven_intervals(
    prepared: _PreparedRun,
    proposal: StepProposal,
    certificate_rows: np.ndarray,
    proposal_rows: np.ndarray,
    interval_start: np.ndarray,
    interval_target: np.ndarray,
    budget: int,
    stack_start: np.ndarray,
    stack_target: np.ndarray,
    stack_top: np.ndarray,
    split_count: np.ndarray,
    codes: np.ndarray,
) -> None:
    """Sample an unproved interval, then fail, exhaust, or push two children."""

    if not certificate_rows.size:
        return
    midpoint = interval_start + 0.5 * (interval_target - interval_start)
    actual_codes = _rk4_dense_actual_sample_failure_codes(
        prepared,
        proposal,
        proposal_rows,
        midpoint,
    )
    endpoint_codes = _rk4_dense_actual_sample_failure_codes(
        prepared,
        proposal,
        proposal_rows,
        interval_target,
    )
    endpoint_failure = (actual_codes == 0) & (endpoint_codes != 0)
    actual_codes[endpoint_failure] = endpoint_codes[endpoint_failure]
    actual_failure = actual_codes != 0
    failed_rows = certificate_rows[actual_failure]
    codes[failed_rows] = actual_codes[actual_failure]
    stack_top[failed_rows] = -1

    candidate = ~actual_failure
    candidate_rows = certificate_rows[candidate]
    candidate_start = interval_start[candidate]
    candidate_target = interval_target[candidate]
    candidate_midpoint = midpoint[candidate]
    exhausted = split_count[candidate_rows] >= budget
    exhausted |= candidate_start >= candidate_midpoint
    exhausted |= candidate_midpoint >= candidate_target
    failed_rows = candidate_rows[exhausted]
    codes[failed_rows] = _FAILURE_INDETERMINATE_APPLICABILITY_CERTIFICATE
    stack_top[failed_rows] = -1

    split_rows = candidate_rows[~exhausted]
    old_top = stack_top[split_rows]
    next_top = old_top + 1
    if bool((next_top >= stack_start.shape[1]).any()):
        raise EngineError("applicability certificate exceeded its bounded stack")
    # Keep the right child in the current slot and visit the left child first.
    stack_start[split_rows, old_top] = candidate_midpoint[~exhausted]
    stack_target[split_rows, old_top] = candidate_target[~exhausted]
    stack_start[split_rows, next_top] = candidate_start[~exhausted]
    stack_target[split_rows, next_top] = candidate_midpoint[~exhausted]
    stack_top[split_rows] = next_top
    split_count[split_rows] += 1


def _rk4_dense_actual_sample_failure_codes(
    prepared: _PreparedRun,
    proposal: StepProposal,
    local_rows: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Classify actual dense states without treating interval bounds as states."""

    sample = proposal.rk4_dense_state_at_rows(local_rows, time_s)
    evaluation = prepared.dynamics(
        sample.particle_index,
        time_s,
        sample.position_m,
        sample.velocity_m_s,
        sample.charge_number,
    )
    codes = _proposal_numerical_failure_codes(evaluation.numerical_status)
    support_failure = (codes == 0) & ~evaluation.support_inside
    codes[support_failure] = _FAILURE_FIELD_SUPPORT
    model_failure = (codes == 0) & ~evaluation.applicability_inside
    codes[model_failure] = _FAILURE_MODEL_APPLICABILITY
    return codes


def _continuous_applicability_verdicts(
    prepared: _PreparedRun,
    proposal: StepProposal,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a non-RK4 path certificate and isolate numerical row failures."""

    selected = np.asarray(rows, dtype="<i8")
    enclosure = proposal.path_enclosure
    if enclosure is None:
        raise EngineError("reintegrated curved piece lost its continuous safety enclosure")
    indices = proposal.particle_index[selected]
    velocity_lower = enclosure.velocity_lower_m_s[selected]
    velocity_upper = enclosure.velocity_upper_m_s[selected]

    try:
        if proposal.path_kind == "cubic_hermite":
            charge_lower, charge_upper = proposal.cubic_charge_interval(selected)
            verdict, status = prepared.dynamics.global_continuous_applicability_batch(
                indices,
                velocity_lower,
                velocity_upper,
                charge_lower,
                charge_upper,
            )
        else:
            charge_lower, charge_upper = prepared.dynamics.prepared_charge_invariant_interval(
                proposal.start_charge_number[selected]
            )
            verdict, status = prepared.dynamics.global_continuous_applicability_batch(
                indices,
                velocity_lower,
                velocity_upper,
                charge_lower,
                charge_upper,
            )
            _require_applicability_certificate_result(
                verdict,
                status,
                indices.size,
                "global exponential",
            )
            global_proven = verdict & (status == CONTINUOUS_APPLICABILITY_OK)
            local_rows = np.flatnonzero(~global_proven).astype("<i8", copy=False)
            if local_rows.size:
                local, local_status, range_available = (
                    prepared.dynamics.local_continuous_applicability_batch(
                        indices[local_rows],
                        enclosure.position_lower_m[selected[local_rows]],
                        enclosure.position_upper_m[selected[local_rows]],
                        velocity_lower[local_rows],
                        velocity_upper[local_rows],
                        charge_lower[local_rows],
                        charge_upper[local_rows],
                    )
                )
                _require_applicability_certificate_result(
                    local,
                    local_status,
                    local_rows.size,
                    "local exponential",
                    range_available,
                )
                resolved = np.flatnonzero(range_available).astype("<i8", copy=False)
                target_rows = local_rows[resolved]
                verdict[target_rows] = local[resolved]
                status[target_rows] = local_status[resolved]
    except (PhysicsEvaluationError, ValueError) as error:
        raise EngineError("curved-path applicability batch could not be evaluated") from error
    if verdict.shape != (indices.size,) or status.shape != (indices.size,):
        raise EngineError("curved-path applicability verdict has an invalid shape")
    known = (status == CONTINUOUS_APPLICABILITY_OK) | (
        status == CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE
    )
    if not bool(known.all()):
        raise EngineError("curved-path applicability returned an unknown row status")
    return verdict, status == CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE


def _curved_enclosure_inside_support(
    prepared: _PreparedRun,
    proposal: StepProposal,
    rows: np.ndarray,
    contact_certified: np.ndarray,
    event_slack_m: np.ndarray,
) -> np.ndarray:
    """Return a per-row continuous-support verdict."""

    enclosure = proposal.path_enclosure
    if enclosure is None:
        raise EngineError("reintegrated curved piece lost its continuous enclosure")
    return _curved_position_bounds_inside_support(
        prepared,
        enclosure.position_lower_m[rows],
        enclosure.position_upper_m[rows],
        contact_certified,
        event_slack_m,
    )


def _curved_position_bounds_inside_support(
    prepared: _PreparedRun,
    position_lower_m: np.ndarray,
    position_upper_m: np.ndarray,
    contact_certified: np.ndarray,
    event_slack_m: np.ndarray,
) -> np.ndarray:
    """Return the continuous-support verdict for explicit position bounds."""

    support_box = prepared.fields.regular_support_box()
    if support_box is None:
        if not prepared.geometry.facet_count:
            raise EngineError("curved continuous support certificate is unavailable")
        return np.ones(position_lower_m.shape[0], dtype=np.bool_)
    lower_bound, upper_bound = support_box
    path_lower = np.asarray(position_lower_m, dtype="<f8")
    path_upper = np.asarray(position_upper_m, dtype="<f8")
    if prepared.case.data.coordinate_system == "axisymmetric_rz":
        path_lower, path_upper = canonicalize_rz_enclosure(path_lower, path_upper)
    position_inside = (path_lower >= lower_bound).all(axis=1)
    position_inside &= (path_upper <= upper_bound).all(axis=1)
    padded = event_slack_m != 0.0
    if bool(padded.any()):
        padded_lower = np.nextafter(
            lower_bound[np.newaxis, :] - event_slack_m[padded, np.newaxis],
            -np.inf,
        )
        padded_upper = np.nextafter(
            upper_bound[np.newaxis, :] + event_slack_m[padded, np.newaxis],
            np.inf,
        )
        position_inside[padded] = (path_lower[padded] >= padded_lower).all(axis=1)
        position_inside[padded] &= (path_upper[padded] <= padded_upper).all(axis=1)
    geometry_lower = np.min(prepared.geometry.nodes_m, axis=0)
    geometry_upper = np.max(prepared.geometry.nodes_m, axis=0)
    support_covers_geometry = bool(
        (geometry_lower >= lower_bound).all() and (geometry_upper <= upper_bound).all()
    )
    return position_inside | (contact_certified & support_covers_geometry)


def _commit_proposal_rows(
    prepared: _PreparedRun,
    proposal: StepProposal,
    local_rows: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
) -> None:
    rows = np.asarray(local_rows, dtype=np.int64)
    particle_index = proposal.particle_index[rows]
    position_m[particle_index] = proposal.end_position_m[rows]
    velocity_m_s[particle_index] = proposal.end_velocity_m_s[rows]
    charge_number[particle_index] = proposal.end_charge_number[rows]
    _commit_proposal_field_cell(prepared, proposal, rows)


def _commit_proposal_field_cell(
    prepared: _PreparedRun,
    proposal: StepProposal,
    local_rows: np.ndarray,
) -> None:
    """Commit only the nonphysical search hint for an accepted proposal prefix."""

    field_cell_id = proposal.end_field_cell_id
    if field_cell_id is None:
        return
    rows = np.asarray(local_rows, dtype=np.int64)
    prepared.dynamics.commit_field_cell(
        proposal.particle_index[rows],
        field_cell_id[rows],
    )


def _build_curved_rows(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
) -> StepProposal:
    """Build a reintegrated curved proposal without changing failure ownership."""

    indices = np.asarray(particle_index, dtype="<i8")
    times = np.asarray(start_time_s, dtype="<f8")
    targets = np.asarray(target_time_s, dtype="<f8")
    positions = np.asarray(start_position_m, dtype="<f8")
    velocities = np.asarray(start_velocity_m_s, dtype="<f8")
    charges = np.asarray(start_charge_number, dtype="<f8")
    enclosure = _certify_curved_path(
        prepared,
        indices,
        times,
        targets,
        positions,
        velocities,
        charges,
    )
    proposal = _build_step_proposal(
        prepared,
        indices,
        times,
        targets,
        positions,
        velocities,
        charges,
        enclosure,
    )
    if _is_reintegrated_path(proposal) and proposal.path_enclosure is None:
        raise EngineError("reintegrated curved proposal lost its path enclosure")
    return proposal


def _advance_curved_events(
    prepared: _PreparedRun,
    root_proposal: StepProposal,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
    writer: ResultWriter,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
) -> np.ndarray | None:
    """Advance one slab of reintegrated paths as a flat SoA wavefront."""

    if not _is_reintegrated_path(root_proposal) or root_proposal.path_enclosure is None:
        raise EngineError("curved wavefront requires a reintegrated proposal enclosure")
    state = _allocate_curved_wavefront(prepared, root_proposal)
    root_rows, zero_rows = _initialize_curved_wavefront(
        prepared,
        root_proposal,
        state,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        start_contact_state,
        pending,
        failures,
        replay,
        statistics,
        failure_reason_code,
    )
    _flush_boundary_event_wave(writer, prepared, pending)
    if zero_rows.size:
        _commit_curved_clear_wave(
            prepared,
            root_proposal,
            zero_rows,
            zero_rows,
            root_proposal.target_time_s[zero_rows],
            np.zeros(zero_rows.size, dtype=np.bool_),
            state,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            replay,
            failures,
            statistics,
            failure_reason_code,
        )
    if root_rows.size:
        root_accuracy = root_proposal.numerical_status[root_rows] == INTEGRATOR_ACCURACY_FAILURE
        _fail_event_rows(
            root_proposal.particle_index,
            root_rows[root_accuracy],
            state.current_time_s,
            state.current_position_m,
            state.current_velocity_m_s,
            state.current_charge_number,
            state.stack_top,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
            _FAILURE_INTEGRATOR_ACCURACY,
        )
        root_rows = root_rows[~root_accuracy]
        root_codes = _proposal_numerical_failure_codes(root_proposal.numerical_status[root_rows])
        root_invalid = root_codes != 0
        _split_curved_wave_rows(
            prepared,
            root_proposal.particle_index,
            root_rows[root_invalid],
            state,
            reset_interactions=False,
            exhaustion_reason_codes=root_codes[root_invalid],
            position_m=position_m,
            velocity_m_s=velocity_m_s,
            charge_number=charge_number,
            active=active,
            lifecycle=lifecycle,
            terminal_time_s=terminal_time_s,
            event_ordinal=event_ordinal,
            failure_reason_code=failure_reason_code,
            failures=failures,
            statistics=statistics,
        )
        root_rows = root_rows[~root_invalid]
        statistics.candidate_queries += int(root_rows.size)
        for wave_rows, proposal_rows, targets, event in _locate_curved_wave_batches(
            prepared,
            root_proposal,
            root_rows,
            root_rows,
            root_proposal.target_time_s[root_rows],
            state,
        ):
            _consume_curved_wave(
                prepared,
                root_proposal,
                proposal_rows,
                wave_rows,
                targets,
                event,
                state,
                position_m,
                velocity_m_s,
                charge_number,
                active,
                lifecycle,
                terminal_time_s,
                event_ordinal,
                physical_boundary_event_ordinal,
                start_contact_state,
                replay,
                pending,
                failures,
                statistics,
                failure_reason_code,
            )
            _flush_boundary_event_wave(writer, prepared, pending)
    while True:
        wave_rows, targets = _next_curved_wave_rows(
            root_proposal.particle_index,
            state,
            active,
        )
        if not wave_rows.size:
            break
        try:
            if root_proposal.path_kind == "cubic_hermite":
                proposal = restrict_cubic_hermite_proposal(
                    root_proposal,
                    wave_rows,
                    state.current_time_s[wave_rows],
                    targets,
                )
            else:
                proposal = _build_curved_rows(
                    prepared,
                    root_proposal.particle_index[wave_rows],
                    state.current_time_s[wave_rows],
                    targets,
                    state.current_position_m[wave_rows],
                    state.current_velocity_m_s[wave_rows],
                    state.current_charge_number[wave_rows],
                )
        except (FieldLocationError, PhysicsEvaluationError, ValueError) as error:
            first_particle = int(root_proposal.particle_index[wave_rows[0]])
            particle_id = int(prepared.schedule.particle_id[first_particle])
            raise EngineError(
                f"curved proposal batch failed at or after particle {particle_id}"
            ) from error
        proposal_rows = np.arange(wave_rows.size, dtype="<i8")
        proposal_accuracy = proposal.numerical_status == INTEGRATOR_ACCURACY_FAILURE
        _fail_event_rows(
            root_proposal.particle_index,
            wave_rows[proposal_accuracy],
            state.current_time_s,
            state.current_position_m,
            state.current_velocity_m_s,
            state.current_charge_number,
            state.stack_top,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
            _FAILURE_INTEGRATOR_ACCURACY,
        )
        proposal_codes = _proposal_numerical_failure_codes(proposal.numerical_status)
        proposal_invalid = proposal_codes != 0
        proposal_invalid &= ~proposal_accuracy
        _split_curved_wave_rows(
            prepared,
            root_proposal.particle_index,
            wave_rows[proposal_invalid],
            state,
            reset_interactions=False,
            exhaustion_reason_codes=proposal_codes[proposal_invalid],
            position_m=position_m,
            velocity_m_s=velocity_m_s,
            charge_number=charge_number,
            active=active,
            lifecycle=lifecycle,
            terminal_time_s=terminal_time_s,
            event_ordinal=event_ordinal,
            failure_reason_code=failure_reason_code,
            failures=failures,
            statistics=statistics,
        )
        valid = ~proposal_invalid & ~proposal_accuracy
        wave_rows = wave_rows[valid]
        targets = targets[valid]
        proposal_rows = proposal_rows[valid]
        statistics.candidate_queries += int(wave_rows.size)
        for (
            bounded_rows,
            bounded_proposal_rows,
            bounded_targets,
            event,
        ) in _locate_curved_wave_batches(
            prepared,
            proposal,
            proposal_rows,
            wave_rows,
            targets,
            state,
        ):
            _consume_curved_wave(
                prepared,
                proposal,
                bounded_proposal_rows,
                bounded_rows,
                bounded_targets,
                event,
                state,
                position_m,
                velocity_m_s,
                charge_number,
                active,
                lifecycle,
                terminal_time_s,
                event_ordinal,
                physical_boundary_event_ordinal,
                start_contact_state,
                replay,
                pending,
                failures,
                statistics,
                failure_reason_code,
            )
            _flush_boundary_event_wave(writer, prepared, pending)
    return state.stochastic_restart_time_s


def _allocate_curved_wavefront(
    prepared: _PreparedRun,
    proposal: StepProposal,
) -> _CurvedWavefront:
    row_count = proposal.particle_index.size
    stack_capacity = prepared.case.spec.solver.event.max_refinements + 1
    restart_time_s = (
        np.full(row_count, np.nan, dtype="<f8")
        if (
            proposal.path_kind == "cubic_hermite"
            and prepared.physics.noise is not None
            and prepared.physics.noise.revision == BROWNIAN_RZ_MIDPOINT_REVISION
            and prepared.case.data.coordinate_system == "axisymmetric_rz"
        )
        else None
    )
    return _CurvedWavefront(
        proposal.particle_index.copy(),
        proposal.start_time_s.copy(),
        proposal.start_position_m.copy(),
        proposal.start_velocity_m_s.copy(),
        proposal.start_charge_number.copy(),
        proposal.target_time_s - proposal.start_time_s,
        np.zeros(row_count, dtype="<i8"),
        np.zeros(row_count, dtype=np.bool_),
        np.empty((row_count, stack_capacity), dtype="<f8"),
        np.empty((row_count, stack_capacity), dtype="<i8"),
        np.empty((row_count, stack_capacity), dtype="<i8"),
        np.full(row_count, -1, dtype="<i8"),
        restart_time_s,
    )


def _initialize_curved_wavefront(
    prepared: _PreparedRun,
    proposal: StepProposal,
    state: _CurvedWavefront,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    start_contact_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve surface release once and seed the numeric depth-first stacks."""

    interactions, _ = _initialize_surface_releases(
        prepared,
        proposal.particle_index,
        proposal.start_time_s,
        state.current_position_m,
        state.current_velocity_m_s,
        state.current_charge_number,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        start_contact_state,
        pending,
        failures,
        replay,
        statistics,
        failure_reason_code,
    )
    local_active = active[proposal.particle_index]
    response_rows = np.flatnonzero(local_active & (interactions != 0)).astype("<i8", copy=False)
    particles = proposal.particle_index[response_rows]
    state.current_position_m[response_rows] = position_m[particles]
    state.current_velocity_m_s[response_rows] = velocity_m_s[particles]
    state.current_charge_number[response_rows] = charge_number[particles]
    state.interaction_count[:] = interactions
    state.certify_departure[:] = (
        start_contact_state[proposal.particle_index] == SURFACE_STATE_DEPARTURE
    )
    eligible = np.flatnonzero(local_active).astype("<i8", copy=False)
    state.stack_top[eligible] = 0
    state.stack_target_s[eligible, 0] = proposal.target_time_s[eligible]
    state.stack_depth[eligible, 0] = 0
    state.stack_interaction_reset[eligible, 0] = -1
    reuse_root = local_active & (interactions == 0)
    positive = state.current_time_s < proposal.target_time_s
    root_rows = np.flatnonzero(reuse_root & positive).astype("<i8", copy=False)
    zero_rows = np.flatnonzero(reuse_root & ~positive).astype("<i8", copy=False)
    inactive_zero = np.flatnonzero((~reuse_root) & ~positive).astype("<i8", copy=False)
    state.stack_top[inactive_zero] = -1
    return root_rows, zero_rows


def _next_curved_wave_rows(
    particle_index: np.ndarray,
    state: _CurvedWavefront,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pop exhausted work and return one stable row per active particle."""

    while True:
        rows = np.flatnonzero((state.stack_top >= 0) & active[particle_index]).astype(
            "<i8", copy=False
        )
        if not rows.size:
            return rows, np.empty(0, dtype="<f8")
        top = state.stack_top[rows]
        reset = state.stack_interaction_reset[rows, top]
        reset_rows = reset >= 0
        if bool(reset_rows.any()):
            selected = rows[reset_rows]
            state.interaction_count[selected] = reset[reset_rows]
            state.stack_interaction_reset[selected, state.stack_top[selected]] = -1
        targets = state.stack_target_s[rows, state.stack_top[rows]]
        exhausted = state.current_time_s[rows] >= targets
        if not bool(exhausted.any()):
            return rows, targets
        state.stack_top[rows[exhausted]] -= 1


def _locate_curved_wave_batches(
    prepared: _PreparedRun,
    proposal: StepProposal,
    proposal_rows: np.ndarray,
    wave_rows: np.ndarray,
    target_time_s: np.ndarray,
    state: _CurvedWavefront,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, CurvedEventBatch]]:
    """Locate one logical curved query in stable capacity-bounded row prefixes."""

    if not wave_rows.size:
        return
    enclosure = proposal.path_enclosure
    if enclosure is None:
        raise EngineError("curved event path lost its continuous enclosure")
    event = prepared.case.spec.solver.event
    chord = _proposal_chord_deviation_bounds(
        proposal,
        proposal_rows,
        state.current_time_s[wave_rows],
        target_time_s,
    )
    (
        event_position_lower_m,
        event_position_upper_m,
        event_velocity_lower_m_s,
        event_velocity_upper_m_s,
        force_split,
        use_position_controls,
        position_control_origin_m,
        relative_position_control_lower_m,
        relative_position_control_upper_m,
    ) = _proposal_event_bounds(
        prepared,
        proposal,
        proposal_rows,
        state.current_time_s[wave_rows],
        target_time_s,
    )
    try:
        candidate_count = count_curved_event_candidates(
            prepared.geometry,
            proposal.start_position_m[proposal_rows],
            proposal.start_velocity_m_s[proposal_rows],
            proposal.end_position_m[proposal_rows],
            proposal.end_velocity_m_s[proposal_rows],
            event_position_lower_m,
            event_position_upper_m,
            event_velocity_lower_m_s,
            event_velocity_upper_m_s,
            start_time_s=proposal.start_time_s[proposal_rows],
            target_time_s=proposal.target_time_s[proposal_rows],
            root_interval_s=state.root_interval_s[wave_rows],
            geometry_rtol=event.geometry_rtol,
            roundoff_ulps=event.roundoff_ulps,
            certify_start_contact_departure=state.certify_departure[wave_rows],
            chord_deviation_bound_m=chord,
        )
    except (EventLocationError, GeometryPreparationError, ValueError) as error:
        first_particle = int(proposal.particle_index[proposal_rows[0]])
        particle_id = int(prepared.schedule.particle_id[first_particle])
        raise EngineError(
            f"curved event batch violated a shared invariant at particle {particle_id}"
        ) from error
    capacity = prepared.event_candidate_capacity
    begin = 0
    while begin < wave_rows.size:
        end = _bounded_event_row_stop(candidate_count, begin, capacity)
        bounded_rows = wave_rows[begin:end]
        bounded_proposal_rows = proposal_rows[begin:end]
        bounded_targets = target_time_s[begin:end]
        bounded_chord = None if chord is None else chord[begin:end]
        try:
            result = locate_curved_first_event_batch(
                prepared.geometry,
                proposal.start_position_m[bounded_proposal_rows],
                proposal.start_velocity_m_s[bounded_proposal_rows],
                proposal.end_position_m[bounded_proposal_rows],
                proposal.end_velocity_m_s[bounded_proposal_rows],
                event_position_lower_m[begin:end],
                event_position_upper_m[begin:end],
                event_velocity_lower_m_s[begin:end],
                event_velocity_upper_m_s[begin:end],
                start_time_s=proposal.start_time_s[bounded_proposal_rows],
                target_time_s=proposal.target_time_s[bounded_proposal_rows],
                root_interval_s=state.root_interval_s[bounded_rows],
                geometry_rtol=event.geometry_rtol,
                roundoff_ulps=event.roundoff_ulps,
                candidate_capacity=capacity,
                certify_start_contact_departure=state.certify_departure[bounded_rows],
                chord_deviation_bound_m=bounded_chord,
                certify_monotone_approach=proposal.path_kind == "cubic_hermite",
                use_position_controls=(
                    None if use_position_controls is None else use_position_controls[begin:end]
                ),
                position_control_origin_m=(
                    None
                    if position_control_origin_m is None
                    else position_control_origin_m[begin:end]
                ),
                relative_position_control_lower_m=(
                    None
                    if relative_position_control_lower_m is None
                    else relative_position_control_lower_m[begin:end]
                ),
                relative_position_control_upper_m=(
                    None
                    if relative_position_control_upper_m is None
                    else relative_position_control_upper_m[begin:end]
                ),
            )
        except (EventLocationError, GeometryPreparationError, ValueError) as error:
            raise EngineError("bounded curved event batch violated its capacity") from error
        result.status[force_split[begin:end]] = CURVED_STATUS_SPLIT
        yield bounded_rows, bounded_proposal_rows, bounded_targets, result
        begin = end


def _proposal_event_bounds(
    prepared: _PreparedRun,
    proposal: StepProposal,
    proposal_rows: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Return the continuous bounds owned by the proposal's event path.

    The global reintegration enclosure remains the authority for shortened RK4
    stages, field support, and model applicability.  Only after that enclosure
    is independently inside field support may the RK4 first-hit path use its
    tighter outward Bernstein enclosure.  If restriction cannot produce a
    valid row, retain the global bounds for bounded locator storage and force
    that row to split.
    """

    enclosure = proposal.path_enclosure
    if enclosure is None:
        raise EngineError("curved event path lost its continuous enclosure")
    rows = np.asarray(proposal_rows, dtype="<i8")
    position_lower_m = enclosure.position_lower_m[rows]
    position_upper_m = enclosure.position_upper_m[rows]
    velocity_lower_m_s = enclosure.velocity_lower_m_s[rows]
    velocity_upper_m_s = enclosure.velocity_upper_m_s[rows]
    force_split = np.zeros(rows.size, dtype=np.bool_)
    if proposal.path_kind != "rk4_dense":
        return (
            position_lower_m,
            position_upper_m,
            velocity_lower_m_s,
            velocity_upper_m_s,
            force_split,
            None,
            None,
            None,
            None,
        )

    dense = proposal.rk4_dense_subinterval(rows, start_time_s, target_time_s)
    valid = dense.numerical_status == NUMERICAL_STATUS_OK
    support_inside = _curved_position_bounds_inside_support(
        prepared,
        position_lower_m,
        position_upper_m,
        np.zeros(rows.size, dtype=np.bool_),
        np.zeros(rows.size, dtype="<f8"),
    )
    support_inside &= proposal.support_inside[rows]
    use_dense = valid & support_inside
    position_lower_m[use_dense] = dense.position_lower_m[use_dense]
    position_upper_m[use_dense] = dense.position_upper_m[use_dense]
    velocity_lower_m_s[use_dense] = dense.velocity_lower_m_s[use_dense]
    velocity_upper_m_s[use_dense] = dense.velocity_upper_m_s[use_dense]
    force_split[~valid] = True
    return (
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        force_split,
        use_dense,
        dense.position_control_origin_m,
        dense.relative_position_control_lower_m,
        dense.relative_position_control_upper_m,
    )


def _proposal_chord_deviation_bounds(
    proposal: StepProposal,
    rows: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray | None:
    """Preserve the selected integrator's certified chord arithmetic."""

    enclosure = proposal.path_enclosure
    if (
        proposal.path_kind
        not in {
            "cubic_hermite",
            "rk4_dense",
            "exponential_midpoint_reintegrated",
        }
        or enclosure is None
    ):
        raise EngineError("curved event path has no method-specific enclosure")
    if proposal.path_kind == "rk4_dense":
        return proposal.rk4_dense_chord_deviation(
            rows,
            start_time_s,
            target_time_s,
        )
    return curved_chord_deviation_bounds(
        proposal.start_position_m[rows],
        proposal.end_position_m[rows],
        enclosure.velocity_lower_m_s[rows],
        enclosure.velocity_upper_m_s[rows],
        proposal.target_time_s[rows] - proposal.start_time_s[rows],
    )


def _consume_curved_wave(
    prepared: _PreparedRun,
    proposal: StepProposal,
    proposal_rows: np.ndarray,
    wave_rows: np.ndarray,
    target_time_s: np.ndarray,
    event: CurvedEventBatch,
    state: _CurvedWavefront,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    """Consume one bounded locator result without row-local scalar retries."""

    particle_index = state.particle_index
    failed = np.flatnonzero(event.status == CURVED_STATUS_FAILURE).astype("<i8", copy=False)
    _fail_event_rows(
        particle_index,
        wave_rows[failed],
        state.current_time_s,
        state.current_position_m,
        state.current_velocity_m_s,
        state.current_charge_number,
        state.stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failures,
        _FAILURE_INDETERMINATE_EVENT,
    )
    split = np.flatnonzero(event.status == CURVED_STATUS_SPLIT).astype("<i8", copy=False)
    _split_curved_wave_rows(
        prepared,
        particle_index,
        wave_rows[split],
        state,
        reset_interactions=False,
        position_m=position_m,
        velocity_m_s=velocity_m_s,
        charge_number=charge_number,
        active=active,
        lifecycle=lifecycle,
        terminal_time_s=terminal_time_s,
        event_ordinal=event_ordinal,
        failure_reason_code=failure_reason_code,
        failures=failures,
        statistics=statistics,
    )
    wall = event.status == CURVED_STATUS_WALL
    wall &= state.interaction_count[wave_rows] >= (
        prepared.case.spec.solver.event.max_interactions_per_step
    )
    capped = np.flatnonzero(wall).astype("<i8", copy=False)
    _split_curved_wave_rows(
        prepared,
        particle_index,
        wave_rows[capped],
        state,
        reset_interactions=True,
        position_m=position_m,
        velocity_m_s=velocity_m_s,
        charge_number=charge_number,
        active=active,
        lifecycle=lifecycle,
        terminal_time_s=terminal_time_s,
        event_ordinal=event_ordinal,
        failure_reason_code=failure_reason_code,
        failures=failures,
        statistics=statistics,
    )
    statistics.residual_splits += int(capped.size)
    clear = np.flatnonzero(event.status == CURVED_STATUS_CLEAR).astype("<i8", copy=False)
    _commit_curved_clear_wave(
        prepared,
        proposal,
        proposal_rows[clear],
        wave_rows[clear],
        target_time_s[clear],
        event.start_contact_departure_certified[clear],
        state,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        replay,
        failures,
        statistics,
        failure_reason_code,
    )
    localized = (event.status == CURVED_STATUS_AXIS) | (
        (event.status == CURVED_STATUS_WALL) & ~wall
    )
    event_rows = np.flatnonzero(localized).astype("<i8", copy=False)
    _commit_curved_event_prefixes(
        prepared,
        proposal,
        proposal_rows,
        particle_index,
        wave_rows,
        event_rows,
        event,
        state,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        start_contact_state,
        replay,
        pending,
        failures,
        statistics,
        failure_reason_code,
    )


def _commit_curved_clear_wave(
    prepared: _PreparedRun,
    proposal: StepProposal,
    proposal_rows: np.ndarray,
    wave_rows: np.ndarray,
    target_time_s: np.ndarray,
    start_contact_certified: np.ndarray,
    state: _CurvedWavefront,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    replay: _ReplayBuffer,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    """Commit certified clear intervals in stable proposal-row order."""

    if not wave_rows.size:
        return
    codes = _curved_row_failure_codes(
        prepared,
        proposal,
        proposal_rows,
        start_contact_certified=start_contact_certified,
    )
    invalid = np.flatnonzero(codes).astype("<i8", copy=False)
    _commit_event_proposal_failures(
        state.particle_index,
        [
            (
                int(wave_rows[offset]),
                _proposal_row_failure(
                    proposal,
                    int(proposal_rows[offset]),
                    codes[offset],
                ),
            )
            for offset in invalid
        ],
        state.stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failures,
    )
    valid_offsets = np.flatnonzero(codes == 0).astype("<i8", copy=False)
    if not valid_offsets.size:
        return
    valid_proposal_rows = proposal_rows[valid_offsets]
    valid_wave_rows = wave_rows[valid_offsets]
    _commit_proposal_rows(
        prepared,
        proposal,
        valid_proposal_rows,
        position_m,
        velocity_m_s,
        charge_number,
    )
    state.current_time_s[valid_wave_rows] = target_time_s[valid_offsets]
    state.current_position_m[valid_wave_rows] = proposal.end_position_m[valid_proposal_rows]
    state.current_velocity_m_s[valid_wave_rows] = proposal.end_velocity_m_s[valid_proposal_rows]
    state.current_charge_number[valid_wave_rows] = proposal.end_charge_number[valid_proposal_rows]
    state.stack_top[valid_wave_rows] -= 1
    _record_replay_rows(
        replay,
        proposal,
        valid_proposal_rows,
        prepared.case.data.coordinate_system,
    )
    statistics.accepted_particle_pieces += int(valid_offsets.size)


def _commit_curved_event_prefixes(
    prepared: _PreparedRun,
    proposal: StepProposal,
    proposal_rows: np.ndarray,
    particle_index: np.ndarray,
    wave_rows: np.ndarray,
    event_rows: np.ndarray,
    event: CurvedEventBatch,
    state: _CurvedWavefront,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    start_contact_state: np.ndarray,
    replay: _ReplayBuffer,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    """Reintegrate localized prefixes once, then commit axis and wall rows."""

    if not event_rows.size:
        return
    selected_wave_rows = wave_rows[event_rows]
    needs_proposal = event.status[event_rows] == CURVED_STATUS_WALL
    needs_proposal |= event.time_s[event_rows] > state.current_time_s[selected_wave_rows]
    prefix_event_rows = event_rows[needs_proposal]
    prefix_wave_rows = wave_rows[prefix_event_rows]
    prefix_by_event = np.full(event.status.size, -1, dtype="<i8")
    valid_event = np.zeros(event.status.size, dtype=np.bool_)
    valid_event[event_rows[~needs_proposal]] = True
    prefix: StepProposal | None = None
    if prefix_event_rows.size:
        prefix = _build_curved_event_prefix_batch(
            prepared,
            proposal,
            proposal_rows[prefix_event_rows],
            particle_index,
            prefix_wave_rows,
            prefix_event_rows,
            event,
            state,
        )
        prefix_rows = np.arange(prefix_event_rows.size, dtype="<i8")
        codes = _curved_row_failure_codes(
            prepared,
            prefix,
            prefix_rows,
            event_slack_m=event.position_budget_m[prefix_event_rows],
            start_contact_certified=event.start_contact_departure_certified[prefix_event_rows],
        )
        retryable = codes == _FAILURE_FIELD_SUPPORT
        retryable |= codes == _FAILURE_MODEL_APPLICABILITY
        retryable |= codes == _FAILURE_INDETERMINATE_APPLICABILITY_CERTIFICATE
        retry = np.flatnonzero(retryable).astype("<i8", copy=False)
        _split_curved_wave_rows(
            prepared,
            particle_index,
            prefix_wave_rows[retry],
            state,
            reset_interactions=False,
            exhaustion_reason_codes=codes[retry],
            position_m=position_m,
            velocity_m_s=velocity_m_s,
            charge_number=charge_number,
            active=active,
            lifecycle=lifecycle,
            terminal_time_s=terminal_time_s,
            event_ordinal=event_ordinal,
            failure_reason_code=failure_reason_code,
            failures=failures,
            statistics=statistics,
        )
        invalid = np.flatnonzero((codes != 0) & ~retryable).astype("<i8", copy=False)
        _commit_event_proposal_failures(
            particle_index,
            [
                (
                    int(prefix_wave_rows[offset]),
                    _proposal_row_failure(prefix, int(offset), codes[offset]),
                )
                for offset in invalid
            ],
            state.stack_top,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
        )
        valid_prefix = np.flatnonzero(codes == 0).astype("<i8", copy=False)
        valid_prefix_events = prefix_event_rows[valid_prefix]
        _require_curved_event_residuals(
            prepared,
            prefix,
            valid_prefix,
            valid_prefix_events,
            event,
        )
        if valid_prefix.size:
            _commit_proposal_field_cell(prepared, prefix, valid_prefix)
            _record_replay_rows(
                replay,
                prefix,
                valid_prefix,
                prepared.case.data.coordinate_system,
                exclude_target=True,
            )
            statistics.accepted_particle_pieces += int(valid_prefix.size)
            valid_event[valid_prefix_events] = True
        prefix_by_event[prefix_event_rows] = prefix_rows
    _commit_curved_axis_wave(
        particle_index,
        wave_rows,
        event_rows[(event.status[event_rows] == CURVED_STATUS_AXIS) & valid_event[event_rows]],
        event,
        prefix,
        prefix_by_event,
        state,
        position_m,
        velocity_m_s,
        charge_number,
        replay,
        statistics,
    )
    _commit_curved_wall_wave(
        prepared,
        particle_index,
        wave_rows,
        event_rows[(event.status[event_rows] == CURVED_STATUS_WALL) & valid_event[event_rows]],
        event,
        prefix,
        prefix_by_event,
        state,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        start_contact_state,
        pending,
        failures,
        replay,
        statistics,
        failure_reason_code,
    )


def _build_curved_event_prefix_batch(
    prepared: _PreparedRun,
    proposal: StepProposal,
    proposal_rows: np.ndarray,
    particle_index: np.ndarray,
    wave_rows: np.ndarray,
    event_rows: np.ndarray,
    event: CurvedEventBatch,
    state: _CurvedWavefront,
) -> StepProposal:
    """Build one mixed-target prefix batch without scalar error retries."""

    try:
        if proposal.path_kind == "cubic_hermite":
            return restrict_cubic_hermite_proposal(
                proposal,
                proposal_rows,
                state.current_time_s[wave_rows],
                event.time_s[event_rows],
            )
        return _build_curved_rows(
            prepared,
            particle_index[wave_rows],
            state.current_time_s[wave_rows],
            event.time_s[event_rows],
            state.current_position_m[wave_rows],
            state.current_velocity_m_s[wave_rows],
            state.current_charge_number[wave_rows],
        )
    except (FieldLocationError, PhysicsEvaluationError, ValueError) as error:
        first_particle = int(particle_index[wave_rows[0]])
        particle_id = int(prepared.schedule.particle_id[first_particle])
        raise EngineError(
            f"curved event prefix batch failed at or after particle {particle_id}"
        ) from error


def _require_curved_event_residuals(
    prepared: _PreparedRun,
    prefix: StepProposal,
    prefix_rows: np.ndarray,
    event_rows: np.ndarray,
    event: CurvedEventBatch,
) -> None:
    """Verify reintegrated endpoints against the locator's certified budget."""

    for prefix_value, event_value in zip(prefix_rows, event_rows, strict=True):
        prefix_row = int(prefix_value)
        event_row = int(event_value)
        if event.status[event_row] == CURVED_STATUS_AXIS:
            residual = abs(float(prefix.end_position_m[prefix_row, 0]))
            label = "axis"
        else:
            separation = prefix.end_position_m[prefix_row] - event.position_m[event_row]
            residual = math.hypot(float(separation[0]), float(separation[1]))
            label = "hit"
        if math.isfinite(residual) and residual <= event.position_budget_m[event_row]:
            continue
        particle = int(prefix.particle_index[prefix_row])
        particle_id = int(prepared.schedule.particle_id[particle])
        raise EngineError(
            f"curved {label} state exceeds its position budget for particle {particle_id}"
        )


def _commit_curved_axis_wave(
    particle_index: np.ndarray,
    wave_rows: np.ndarray,
    event_rows: np.ndarray,
    event: CurvedEventBatch,
    prefix: StepProposal | None,
    prefix_by_event: np.ndarray,
    state: _CurvedWavefront,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
) -> None:
    """Commit RZ basis folds without changing physical-boundary identity."""

    for event_value in event_rows:
        event_row = int(event_value)
        row = int(wave_rows[event_row])
        particle = int(particle_index[row])
        prefix_row = int(prefix_by_event[event_row])
        if prefix_row >= 0:
            if prefix is None:
                raise EngineError("curved axis prefix mapping lost its proposal")
            axis_position = prefix.end_position_m[prefix_row]
            axis_velocity = prefix.end_velocity_m_s[prefix_row]
            axis_charge = float(prefix.end_charge_number[prefix_row])
        else:
            axis_position = state.current_position_m[row]
            axis_velocity = state.current_velocity_m_s[row]
            axis_charge = float(state.current_charge_number[row])
        folded_position, folded_velocity = fold_rz_position_vector(
            np.asarray([0.0, axis_position[1]], dtype="<f8"),
            axis_velocity,
        )
        time_s = float(event.time_s[event_row])
        state.current_time_s[row] = time_s
        state.current_position_m[row] = folded_position
        state.current_velocity_m_s[row] = folded_velocity
        state.current_charge_number[row] = axis_charge
        state.certify_departure[row] = False
        position_m[particle] = folded_position
        velocity_m_s[particle] = folded_velocity
        charge_number[particle] = axis_charge
        _record_replay_jump(
            replay,
            particle,
            time_s,
            folded_position,
            folded_velocity,
            axis_charge,
        )
        statistics.axis_crossings += 1
        if state.stochastic_restart_time_s is not None:
            state.stochastic_restart_time_s[row] = time_s
            state.stack_top[row] = -1


def _commit_curved_wall_wave(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    wave_rows: np.ndarray,
    event_rows: np.ndarray,
    event: CurvedEventBatch,
    prefix: StepProposal | None,
    prefix_by_event: np.ndarray,
    state: _CurvedWavefront,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    start_contact_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    """Resolve one stable wall wave with compiled rules and counter RNG."""

    if not event_rows.size:
        return
    if prefix is None:
        raise EngineError("curved wall wave lost its reintegrated prefixes")
    rows = wave_rows[event_rows]
    prefix_rows = prefix_by_event[event_rows]
    particles = particle_index[rows]
    offsets, candidates = _select_event_candidate_rows(event, event_rows)
    draws = wall_uniform_batch(
        prepared.case.spec.solver.seed,
        prepared.schedule.particle_id[particles],
        physical_boundary_event_ordinal[particles],
    )
    responses = resolve_boundary_responses_batch(
        prepared.compiled_boundary_rules,
        offsets,
        candidates,
        prepared.geometry.boundary_id,
        prepared.geometry.group_id,
        prepared.geometry.facet_normal,
        prefix.end_velocity_m_s[prefix_rows],
        draws,
        roundoff_ulps=prepared.case.spec.solver.event.roundoff_ulps,
    )
    for response_row, event_value in enumerate(event_rows):
        event_row = int(event_value)
        row = int(rows[response_row])
        prefix_row = int(prefix_rows[response_row])
        particle = int(particles[response_row])
        if responses.status[response_row] != BOUNDARY_STATUS_OK:
            failure = _ParticleFailure(
                _FAILURE_INDETERMINATE_BOUNDARY_POLICY,
                float(event.time_s[event_row]),
                event.position_m[event_row],
                prefix.end_velocity_m_s[prefix_row],
                float(prefix.end_charge_number[prefix_row]),
            )
            _commit_event_proposal_failures(
                particle_index,
                [(row, failure)],
                state.stack_top,
                position_m,
                velocity_m_s,
                charge_number,
                active,
                lifecycle,
                terminal_time_s,
                event_ordinal,
                failure_reason_code,
                failures,
            )
            continue
        begin = int(offsets[response_row])
        end = int(offsets[response_row + 1])
        _commit_curved_wall_row(
            prepared,
            row,
            particle,
            event_row,
            event,
            tuple(int(value) for value in candidates[begin:end]),
            int(responses.law[response_row]),
            int(responses.outcome[response_row]),
            responses.velocity_post_m_s[response_row],
            int(responses.primary_facet_id[response_row]),
            responses.effective_normal[response_row],
            bool(responses.remains_active[response_row]),
            prefix.end_position_m[prefix_row],
            prefix.end_velocity_m_s[prefix_row],
            float(prefix.end_charge_number[prefix_row]),
            state,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            start_contact_state,
            pending,
            replay,
            statistics,
        )


def _commit_curved_wall_row(
    prepared: _PreparedRun,
    row: int,
    particle: int,
    event_row: int,
    event: CurvedEventBatch,
    candidate_facet_ids: tuple[int, ...],
    law_code: int,
    outcome_code: int,
    velocity_post_m_s: np.ndarray,
    primary_facet_id: int,
    effective_normal: np.ndarray,
    remains_active: bool,
    endpoint_position_m: np.ndarray,
    velocity_pre_m_s: np.ndarray,
    charge_pre_number: float,
    state: _CurvedWavefront,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    start_contact_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
) -> None:
    """Commit one compiled curved response after its prefix was accepted."""

    if int(event_ordinal[particle]) >= np.iinfo(np.uint32).max:
        raise EngineError("logical event ordinal exhausted uint32")
    if int(physical_boundary_event_ordinal[particle]) >= np.iinfo(np.uint32).max:
        raise EngineError("physical boundary event ordinal exhausted uint32")
    separation = endpoint_position_m - event.position_m[event_row]
    state_residual = math.hypot(float(separation[0]), float(separation[1]))
    raw_hit = BoundaryHit(
        float(event.time_s[event_row]),
        event.position_m[event_row].copy(),
        int(event.primary_facet_id[event_row]),
        candidate_facet_ids,
        event.normal[event_row].copy(),
        float(event.position_budget_m[event_row]),
        float(event.time_budget_s[event_row]),
        max(float(event.localization_residual_m[event_row]), state_residual),
    )
    resolved_position, projection_residual = _canonical_hit_position(
        prepared.geometry,
        raw_hit,
        primary_facet_id,
    )
    resolved_hit = BoundaryHit(
        raw_hit.time_s,
        resolved_position,
        primary_facet_id,
        candidate_facet_ids,
        effective_normal.copy(),
        raw_hit.position_budget_m,
        raw_hit.time_budget_s,
        max(raw_hit.localization_residual_m, projection_residual),
    )
    position_m[particle] = resolved_position
    velocity_m_s[particle] = velocity_post_m_s
    charge_number[particle] = charge_pre_number
    event_ordinal[particle] += np.uint32(1)
    physical_boundary_event_ordinal[particle] += np.uint32(1)
    _set_boundary_lifecycle(
        particle,
        resolved_hit.time_s,
        outcome_code,
        remains_active,
        active,
        lifecycle,
        terminal_time_s,
    )
    _append_boundary_event(
        pending,
        particle,
        resolved_hit,
        law_code,
        outcome_code,
        velocity_pre_m_s,
        velocity_post_m_s,
        charge_pre_number,
        int(event_ordinal[particle]),
    )
    start_contact_state[particle] = (
        SURFACE_STATE_DEPARTURE if remains_active else SURFACE_STATE_RESOLVED
    )
    state.current_time_s[row] = resolved_hit.time_s
    state.current_position_m[row] = resolved_position
    state.current_velocity_m_s[row] = velocity_post_m_s
    state.current_charge_number[row] = charge_pre_number
    state.interaction_count[row] += 1
    state.certify_departure[row] = remains_active
    statistics.wall_interactions += 1
    if not remains_active:
        state.stack_top[row] = -1
        return
    _record_replay_jump(
        replay,
        particle,
        resolved_hit.time_s,
        resolved_position,
        velocity_post_m_s,
        charge_pre_number,
    )


def _split_curved_wave_rows(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    rows: np.ndarray,
    state: _CurvedWavefront,
    *,
    reset_interactions: bool,
    exhaustion_reason_codes: np.ndarray | None = None,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
) -> None:
    """Bisect indeterminate rows while preserving depth-first left-first order."""

    if not rows.size:
        return
    maximum = prepared.case.spec.solver.event.max_refinements
    reason = _FAILURE_NUMERICAL_EVENT_BUDGET if reset_interactions else _FAILURE_INDETERMINATE_EVENT
    row_reasons = None
    if exhaustion_reason_codes is not None:
        row_reasons = np.asarray(exhaustion_reason_codes, dtype="<u2")
        if row_reasons.shape != (rows.size,) or bool((row_reasons == 0).any()):
            raise EngineError("curved split exhaustion reasons are invalid")
    top = state.stack_top[rows]
    depth = state.stack_depth[rows, top]
    target = state.stack_target_s[rows, top]
    current = state.current_time_s[rows]
    midpoint = current + 0.5 * (target - current)
    invalid = (depth >= maximum) | (current >= midpoint) | (midpoint >= target)
    exhausted = rows[invalid]
    if row_reasons is None:
        _fail_event_rows(
            particle_index,
            exhausted,
            state.current_time_s,
            state.current_position_m,
            state.current_velocity_m_s,
            state.current_charge_number,
            state.stack_top,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
            reason,
        )
    else:
        exhausted_reasons = row_reasons[invalid]
        for row_reason in np.unique(exhausted_reasons):
            selected = exhausted[exhausted_reasons == row_reason]
            _fail_event_rows(
                particle_index,
                selected,
                state.current_time_s,
                state.current_position_m,
                state.current_velocity_m_s,
                state.current_charge_number,
                state.stack_top,
                position_m,
                velocity_m_s,
                charge_number,
                active,
                lifecycle,
                terminal_time_s,
                event_ordinal,
                failure_reason_code,
                failures,
                np.uint16(row_reason),
            )
    valid = ~invalid
    selected = rows[valid]
    if not selected.size:
        return
    selected_top = top[valid]
    child_depth = depth[valid] + 1
    reset_value = 0 if reset_interactions else -1
    state.stack_depth[selected, selected_top] = child_depth
    state.stack_interaction_reset[selected, selected_top] = reset_value
    state.stack_top[selected] = selected_top + 1
    state.stack_target_s[selected, selected_top + 1] = midpoint[valid]
    state.stack_depth[selected, selected_top + 1] = child_depth
    state.stack_interaction_reset[selected, selected_top + 1] = reset_value
    statistics.refinements += int(selected.size)
    statistics.maximum_refinement_depth = max(
        statistics.maximum_refinement_depth,
        int(child_depth.max()),
    )


def _initialize_surface_releases(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    time_s: np.ndarray,
    local_position_m: np.ndarray,
    local_velocity_m_s: np.ndarray,
    local_charge_number: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    surface_release_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify and commit one slab's initial contacts in stable row order."""

    facets = prepared.schedule.source_facet_id[particle_index]
    acceleration = prepared.constant_acceleration_m_s2
    release = classify_surface_release_batch(
        prepared.geometry,
        surface_release_state[particle_index],
        facets,
        local_position_m,
        local_velocity_m_s,
        None if acceleration is None else acceleration[particle_index],
        time_s,
        interval_s=prepared.case.spec.time.dt_s,
        curved_event_path=_uses_curved_event_path(prepared),
        geometry_rtol=prepared.case.spec.solver.event.geometry_rtol,
        roundoff_ulps=prepared.case.spec.solver.event.roundoff_ulps,
    )
    interactions = np.zeros(particle_index.size, dtype="<i8")
    invalid = np.flatnonzero(release.status != SURFACE_STATUS_OK).astype("<i8", copy=False)
    for row_value in invalid:
        row = int(row_value)
        particle = int(particle_index[row])
        failure = _ParticleFailure(
            _FAILURE_INDETERMINATE_SURFACE_DEPARTURE,
            float(time_s[row]),
            local_position_m[row],
            local_velocity_m_s[row],
            float(local_charge_number[row]),
        )
        _mark_particle_failed(
            particle,
            failure,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
        )

    resolved = (release.status == SURFACE_STATUS_OK) & (release.action == SURFACE_ACTION_RESOLVED)
    departure = (release.status == SURFACE_STATUS_OK) & (
        (release.action == SURFACE_ACTION_CURVED_DEPARTURE)
        | (release.action == SURFACE_ACTION_EXACT_DEPARTURE)
    )
    surface_release_state[particle_index[resolved]] = SURFACE_STATE_RESOLVED
    surface_release_state[particle_index[departure]] = SURFACE_STATE_DEPARTURE
    response = (release.status == SURFACE_STATUS_OK) & (
        (release.action == SURFACE_ACTION_RESPONSE_VELOCITY)
        | (release.action == SURFACE_ACTION_RESPONSE_ACCELERATION)
    )
    response_rows = np.flatnonzero(response).astype("<i8", copy=False)
    _commit_surface_release_responses(
        prepared,
        particle_index,
        response_rows,
        facets,
        release,
        time_s,
        local_position_m,
        local_velocity_m_s,
        local_charge_number,
        interactions,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        surface_release_state,
        pending,
        failures,
        replay,
        statistics,
        failure_reason_code,
    )
    return interactions, release.departure_facet_id


def _commit_surface_release_responses(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    rows: np.ndarray,
    source_facet_id: np.ndarray,
    release: SurfaceReleaseBatch,
    time_s: np.ndarray,
    local_position_m: np.ndarray,
    local_velocity_m_s: np.ndarray,
    local_charge_number: np.ndarray,
    interactions: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    surface_release_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    if not rows.size:
        return
    particles = particle_index[rows]
    ordinal_limit = np.iinfo(np.uint32).max
    if bool((event_ordinal[particles] >= ordinal_limit).any()):
        raise EngineError("logical event ordinal exhausted uint32")
    if bool((physical_boundary_event_ordinal[particles] >= ordinal_limit).any()):
        raise EngineError("physical boundary event ordinal exhausted uint32")
    offsets = np.arange(rows.size + 1, dtype="<i8")
    candidates = source_facet_id[rows].astype("<i8", copy=False)
    particle_ids = prepared.schedule.particle_id[particles].astype("<u8", copy=False)
    physical_ordinals = physical_boundary_event_ordinal[particles].astype("<u8", copy=False)
    draws = wall_uniform_batch(
        prepared.case.spec.solver.seed,
        particle_ids,
        physical_ordinals,
    )
    velocity_pre = local_velocity_m_s[rows]
    responses = resolve_boundary_responses_batch(
        prepared.compiled_boundary_rules,
        offsets,
        candidates,
        prepared.geometry.boundary_id,
        prepared.geometry.group_id,
        prepared.geometry.facet_normal,
        velocity_pre,
        draws,
        roundoff_ulps=prepared.case.spec.solver.event.roundoff_ulps,
    )
    post_release: SurfaceReleaseBatch | None = None
    post_by_response = np.full(rows.size, -1, dtype="<i8")
    if _uses_curved_event_path(prepared):
        post_response_rows = np.flatnonzero(
            (responses.status == BOUNDARY_STATUS_OK) & responses.remains_active
        ).astype("<i8", copy=False)
        if post_response_rows.size:
            post_rows = rows[post_response_rows]
            post_release = classify_surface_release_batch(
                prepared.geometry,
                np.full(post_response_rows.size, SURFACE_STATE_PENDING, dtype="<u1"),
                source_facet_id[post_rows],
                local_position_m[post_rows],
                responses.velocity_post_m_s[post_response_rows],
                None,
                time_s[post_rows],
                interval_s=prepared.case.spec.time.dt_s,
                curved_event_path=True,
                geometry_rtol=prepared.case.spec.solver.event.geometry_rtol,
                roundoff_ulps=prepared.case.spec.solver.event.roundoff_ulps,
            )
            post_by_response[post_response_rows] = np.arange(
                post_response_rows.size,
                dtype="<i8",
            )
    for response_row, row_value in enumerate(rows):
        _commit_surface_release_response_row(
            prepared,
            int(row_value),
            response_row,
            particle_index,
            source_facet_id,
            release,
            responses,
            post_release,
            post_by_response,
            time_s,
            local_position_m,
            local_velocity_m_s,
            local_charge_number,
            interactions,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            surface_release_state,
            pending,
            failures,
            replay,
            statistics,
            failure_reason_code,
        )


def _commit_surface_release_response_row(
    prepared: _PreparedRun,
    row: int,
    response_row: int,
    particle_index: np.ndarray,
    source_facet_id: np.ndarray,
    release: SurfaceReleaseBatch,
    responses: BoundaryResponseBatch,
    post_release: SurfaceReleaseBatch | None,
    post_by_response: np.ndarray,
    time_s: np.ndarray,
    local_position_m: np.ndarray,
    local_velocity_m_s: np.ndarray,
    local_charge_number: np.ndarray,
    interactions: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    surface_release_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    particle = int(particle_index[row])
    if responses.status[response_row] != BOUNDARY_STATUS_OK:
        failure = _ParticleFailure(
            _FAILURE_INDETERMINATE_BOUNDARY_POLICY,
            float(time_s[row]),
            local_position_m[row],
            local_velocity_m_s[row],
            float(local_charge_number[row]),
        )
        _mark_particle_failed(
            particle,
            failure,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
        )
        return
    law_code = int(responses.law[response_row])
    outcome_code = int(responses.outcome[response_row])
    facet_id = int(source_facet_id[row])
    hit_position = local_position_m[row].copy()
    hit_normal = prepared.geometry.facet_normal[facet_id].copy()
    hit_position.flags.writeable = False
    hit_normal.flags.writeable = False
    raw_hit = BoundaryHit(
        float(time_s[row]),
        hit_position,
        facet_id,
        (facet_id,),
        hit_normal,
        float(release.position_budget_m[row]),
        float(release.time_budget_s[row]),
        0.0,
    )
    primary_facet_id = int(responses.primary_facet_id[response_row])
    resolved_position, projection_residual = _canonical_hit_position(
        prepared.geometry,
        raw_hit,
        primary_facet_id,
    )
    effective_normal = responses.effective_normal[response_row].copy()
    resolved_hit = BoundaryHit(
        raw_hit.time_s,
        resolved_position,
        primary_facet_id,
        raw_hit.candidate_facet_ids,
        effective_normal,
        raw_hit.position_budget_m,
        raw_hit.time_budget_s,
        projection_residual,
    )
    velocity_post = responses.velocity_post_m_s[response_row].copy()
    remains_active = bool(responses.remains_active[response_row])
    charge_pre = float(local_charge_number[row])
    position_m[particle] = resolved_position
    velocity_m_s[particle] = velocity_post
    charge_number[particle] = charge_pre
    event_ordinal[particle] += np.uint32(1)
    physical_boundary_event_ordinal[particle] += np.uint32(1)
    _set_boundary_lifecycle(
        particle,
        resolved_hit.time_s,
        outcome_code,
        remains_active,
        active,
        lifecycle,
        terminal_time_s,
    )
    _append_boundary_event(
        pending,
        particle,
        resolved_hit,
        law_code,
        outcome_code,
        local_velocity_m_s[row],
        velocity_post,
        charge_pre,
        int(event_ordinal[particle]),
    )
    interactions[row] = 1
    statistics.wall_interactions += 1
    post_failure = release.action[row] == SURFACE_ACTION_RESPONSE_ACCELERATION
    post_row = int(post_by_response[response_row])
    if remains_active and _uses_curved_event_path(prepared):
        post_failure |= (
            post_release is None
            or post_row < 0
            or post_release.status[post_row] != SURFACE_STATUS_OK
            or post_release.action[post_row] != SURFACE_ACTION_CURVED_DEPARTURE
        )
    if remains_active and post_failure:
        failure = _ParticleFailure(
            _FAILURE_INDETERMINATE_SURFACE_DEPARTURE,
            resolved_hit.time_s,
            resolved_position,
            velocity_post,
            charge_pre,
        )
        _mark_particle_failed(
            particle,
            failure,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
        )
        return
    surface_release_state[particle] = (
        SURFACE_STATE_DEPARTURE
        if remains_active and _uses_curved_event_path(prepared)
        else SURFACE_STATE_RESOLVED
    )
    if not remains_active:
        return
    _record_replay_jump(
        replay,
        particle,
        resolved_hit.time_s,
        resolved_position,
        velocity_post,
        charge_pre,
    )


def _canonical_hit_position(
    geometry: PreparedGeometry,
    hit: BoundaryHit,
    primary_facet_id: int,
) -> tuple[np.ndarray, float]:
    """Place accepted wall state on its canonical facet or shared corner."""

    common_nodes: set[int] = set()
    if len(hit.candidate_facet_ids) > 1:
        common_nodes.update(
            int(value) for value in geometry.facet_node_ids[hit.candidate_facet_ids[0]]
        )
        for facet_id in hit.candidate_facet_ids[1:]:
            common_nodes.intersection_update(
                int(value) for value in geometry.facet_node_ids[facet_id]
            )
        if not common_nodes:
            raise EngineError("simultaneous boundary candidates have no shared canonical node")
    if common_nodes:
        position = geometry.nodes_m[min(common_nodes)].copy()
    else:
        start = geometry.facet_start_m[primary_facet_id]
        edge = geometry.facet_end_m[primary_facet_id] - start
        edge_squared = math.fsum((float(edge[0]) ** 2, float(edge[1]) ** 2))
        offset = hit.position_m - start
        parameter = (
            math.fsum((float(offset[0]) * float(edge[0]), float(offset[1]) * float(edge[1])))
            / edge_squared
        )
        position = start + min(max(parameter, 0.0), 1.0) * edge
    separation = position - hit.position_m
    residual = math.hypot(float(separation[0]), float(separation[1]))
    if not math.isfinite(residual) or residual > hit.position_budget_m:
        raise EngineError("canonical boundary state exceeds its localization budget")
    position.flags.writeable = False
    return position, residual


def _prepare_boundary_rules(case: SimulationCase) -> tuple[BoundaryRule, ...]:
    configured = {rule.boundary_group: rule for rule in case.spec.boundaries}
    result = []
    for group_id, name in enumerate(case.data.geometry.group_names):
        rule = configured[name]
        result.append(prepare_boundary_rule(group_id, rule.priority, rule.law, rule.parameters))
    return tuple(result)


def _validate_table_starts(
    case: SimulationCase,
    schedule: ParticleSchedule,
    geometry: PreparedGeometry,
) -> None:
    event = case.spec.solver.event
    for begin in range(0, schedule.particle_count, _PREPARE_SCAN_BATCH_SIZE):
        end = min(begin + _PREPARE_SCAN_BATCH_SIZE, schedule.particle_count)
        volume_containment = points_inside_volume(geometry, schedule.position_m[begin:end])
        for index in range(begin, end):
            if int(schedule.source_facet_id[index]) >= 0:
                continue
            position = schedule.position_m[index]
            speed = math.hypot(
                float(schedule.velocity_m_s[index, 0]),
                float(schedule.velocity_m_s[index, 1]),
            )
            classification = classify_event_point(
                geometry,
                position_m=position,
                speed_m_s=speed,
                interval_s=case.spec.time.dt_s,
                time_s=float(schedule.release_time_s[index]),
                geometry_rtol=event.geometry_rtol,
                roundoff_ulps=event.roundoff_ulps,
                volume_containment=bool(volume_containment[index - begin]),
            )
            if classification != "inside":
                particle_id = int(schedule.particle_id[index])
                raise EngineError(
                    f"P05 table particle {particle_id} must start strictly inside the particle domain"
                )


def _validate_source_facets(
    schedule: ParticleSchedule,
    geometry: PreparedGeometry,
) -> None:
    invalid = (schedule.source_facet_id < -1) | (schedule.source_facet_id >= geometry.facet_count)
    if bool(invalid.any()):
        particle = int(np.flatnonzero(invalid)[0])
        particle_id = int(schedule.particle_id[particle])
        raise ValueError(f"particle {particle_id} has an invalid source facet ID")


def _release_stop(
    schedule: ParticleSchedule,
    cursor: int,
    macro_end: float,
    batch_size: int,
) -> int:
    """Find the release boundary with only slab-sized indexed temporaries."""

    stop = cursor
    if batch_size < 1:
        raise EngineError("release scan requires a positive slab size")
    while stop < schedule.particle_count:
        block_stop = min(stop + batch_size, schedule.particle_count)
        ordered_time_s = schedule.release_time_s[schedule.release_order[stop:block_stop]]
        released_count = int(np.searchsorted(ordered_time_s, macro_end, side="right"))
        stop += released_count
        if released_count < ordered_time_s.size:
            break
    return stop


def _activate_releases(
    writer: ResultWriter,
    schedule: ParticleSchedule,
    cursor: int,
    macro_end: float,
    batch_size: int,
    active: np.ndarray,
    lifecycle: np.ndarray,
    event_ordinal: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Publish and activate the release cohort belonging to one macro step."""

    stop = _release_stop(schedule, cursor, macro_end, batch_size)
    released = schedule.release_order[cursor:stop]
    if released.size:
        for begin in range(0, released.size, batch_size):
            cohort = released[begin : begin + batch_size]
            writer.write_release_events(
                ReleaseEvents(
                    time_s=schedule.release_time_s[cohort],
                    particle_id=schedule.particle_id[cohort],
                    event_ordinal=event_ordinal[cohort],
                    source_id=schedule.source_id[cohort],
                )
            )
        active[released] = True
        lifecycle[released] = _LIFECYCLE_ACTIVE
    return released, stop


def _proposal_start_state(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current accepted state, with exact-origin reuse for zero-force motion."""

    if prepared.physics.requires_stage_evaluation:
        return position_m[particle_index].copy(), velocity_m_s[particle_index].copy()
    if not bool(
        np.isfinite(exact_origin_position_m[particle_index]).all()
        and np.isfinite(exact_origin_velocity_m_s[particle_index]).all()
    ):
        raise EngineError("linear exact proposal must retain its analytic release origin")
    return (
        exact_origin_position_m[particle_index].copy(),
        exact_origin_velocity_m_s[particle_index].copy(),
    )


def _proposal_start_times(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    macro_start: float,
    exact_origin_time_s: np.ndarray,
) -> np.ndarray:
    """Resolve per-particle integration origins without splitting the macro step."""

    schedule = prepared.schedule
    if not prepared.physics.requires_stage_evaluation:
        return exact_origin_time_s[particle_index].copy()
    return np.maximum(macro_start, schedule.release_time_s[particle_index])


def _advance_exact_paths(
    prepared: _PreparedRun,
    root_proposal: StepProposal,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    surface_release_state: np.ndarray,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
    writer: ResultWriter,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
) -> None:
    """Advance one slab of exact paths as a flat residual/event wavefront."""

    if root_proposal.path_kind not in {"linear_exact", "quadratic_exact"}:
        raise EngineError("exact residual work requires a linear or quadratic proposal")
    row_count = root_proposal.particle_index.size
    current_time = root_proposal.start_time_s.copy()
    current_position = root_proposal.start_position_m.copy()
    current_velocity = root_proposal.start_velocity_m_s.copy()
    current_charge = root_proposal.start_charge_number.copy()
    maximum_depth = prepared.case.spec.solver.event.max_refinements
    stack_target = np.empty((row_count, maximum_depth + 1), dtype="<f8")
    stack_depth = np.empty((row_count, maximum_depth + 1), dtype="<i8")
    stack_interactions = np.empty((row_count, maximum_depth + 1), dtype="<i8")
    stack_top = np.full(row_count, -1, dtype="<i8")
    departing_facet = np.full(row_count, -1, dtype="<i8")
    _initialize_exact_wavefront(
        prepared,
        root_proposal,
        current_position,
        current_velocity,
        current_charge,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
        surface_release_state,
        stack_target,
        stack_depth,
        stack_interactions,
        stack_top,
        departing_facet,
        pending,
        failures,
        replay,
        statistics,
        failure_reason_code,
    )
    _flush_boundary_event_wave(writer, prepared, pending)
    path_code = (
        EXACT_PATH_LINEAR if root_proposal.path_kind == "linear_exact" else EXACT_PATH_QUADRATIC
    )
    acceleration = _exact_wave_acceleration(prepared, root_proposal.particle_index)
    while True:
        wave_rows = np.flatnonzero((stack_top >= 0) & active[root_proposal.particle_index]).astype(
            "<i8", copy=False
        )
        if not wave_rows.size:
            break
        targets = stack_target[wave_rows, stack_top[wave_rows]]
        exhausted = current_time[wave_rows] >= targets
        if bool(exhausted.any()):
            stack_top[wave_rows[exhausted]] -= 1
            wave_rows = wave_rows[~exhausted]
            targets = targets[~exhausted]
            if not wave_rows.size:
                continue
        statistics.candidate_queries += int(wave_rows.size)
        for bounded_rows, bounded_targets, event in _locate_exact_wave_batches(
            prepared,
            path_code,
            wave_rows,
            root_proposal.particle_index,
            current_time,
            targets,
            current_position,
            current_velocity,
            acceleration,
            departing_facet,
        ):
            departing_facet[bounded_rows] = -1
            _consume_exact_wave(
                prepared,
                root_proposal.particle_index,
                bounded_rows,
                bounded_targets,
                event,
                current_time,
                current_position,
                current_velocity,
                current_charge,
                stack_target,
                stack_depth,
                stack_interactions,
                stack_top,
                position_m,
                velocity_m_s,
                charge_number,
                active,
                lifecycle,
                terminal_time_s,
                event_ordinal,
                physical_boundary_event_ordinal,
                exact_origin_time_s,
                exact_origin_position_m,
                exact_origin_velocity_m_s,
                surface_release_state,
                replay,
                pending,
                failures,
                statistics,
                failure_reason_code,
            )
            _flush_boundary_event_wave(writer, prepared, pending)
    return


def _initialize_exact_wavefront(
    prepared: _PreparedRun,
    proposal: StepProposal,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    surface_release_state: np.ndarray,
    stack_target_s: np.ndarray,
    stack_depth: np.ndarray,
    stack_interactions: np.ndarray,
    stack_top: np.ndarray,
    departing_facet_id: np.ndarray,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    """Resolve release contacts once and seed each row's depth-first work stack."""

    interactions, departing = _initialize_surface_releases(
        prepared,
        proposal.particle_index,
        proposal.start_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        surface_release_state,
        pending,
        failures,
        replay,
        statistics,
        failure_reason_code,
    )
    local_active = active[proposal.particle_index]
    response_rows = np.flatnonzero(local_active & (interactions != 0)).astype("<i8", copy=False)
    particles = proposal.particle_index[response_rows]
    current_position_m[response_rows] = position_m[particles]
    current_velocity_m_s[response_rows] = velocity_m_s[particles]
    current_charge_number[response_rows] = charge_number[particles]
    exact_origin_time_s[particles] = proposal.start_time_s[response_rows]
    exact_origin_position_m[particles] = current_position_m[response_rows]
    exact_origin_velocity_m_s[particles] = current_velocity_m_s[response_rows]

    positive = local_active & (proposal.start_time_s < proposal.target_time_s)
    rows = np.flatnonzero(positive).astype("<i8", copy=False)
    stack_top[rows] = 0
    stack_target_s[rows, 0] = proposal.target_time_s[rows]
    stack_depth[rows, 0] = 0
    stack_interactions[rows, 0] = interactions[rows]
    departing_facet_id[rows] = departing[rows]
    zero_rows = np.flatnonzero(local_active & ~positive).astype("<i8", copy=False)
    for row_value in zero_rows:
        row = int(row_value)
        _record_replay_jump(
            replay,
            int(proposal.particle_index[row]),
            float(proposal.start_time_s[row]),
            current_position_m[row],
            current_velocity_m_s[row],
            float(current_charge_number[row]),
        )


def _exact_wave_acceleration(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
) -> np.ndarray:
    acceleration = prepared.constant_acceleration_m_s2
    if acceleration is None:
        return np.zeros((particle_index.size, 2), dtype="<f8")
    return acceleration[particle_index].copy()


def _locate_exact_wave_batches(
    prepared: _PreparedRun,
    path_code: np.uint8,
    wave_rows: np.ndarray,
    particle_index: np.ndarray,
    current_time_s: np.ndarray,
    target_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    acceleration_m_s2: np.ndarray,
    departing_facet_id: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, ExactEventBatch]]:
    event = prepared.case.spec.solver.event
    try:
        path_kind = np.full(wave_rows.size, path_code, dtype="<u1")
        candidate_count = count_exact_event_candidates(
            prepared.geometry,
            path_kind,
            current_position_m[wave_rows],
            current_velocity_m_s[wave_rows],
            acceleration_m_s2[wave_rows],
            start_time_s=current_time_s[wave_rows],
            target_time_s=target_time_s,
            geometry_rtol=event.geometry_rtol,
            roundoff_ulps=event.roundoff_ulps,
            certified_departing_facet_id=departing_facet_id[wave_rows],
        )
    except (GeometryPreparationError, ValueError) as error:
        first_particle = int(particle_index[wave_rows[0]])
        particle_id = int(prepared.schedule.particle_id[first_particle])
        raise EngineError(
            f"exact event batch violated a shared invariant at particle {particle_id}"
        ) from error
    capacity = prepared.event_candidate_capacity
    begin = 0
    while begin < wave_rows.size:
        end = _bounded_event_row_stop(candidate_count, begin, capacity)
        bounded_rows = wave_rows[begin:end]
        bounded_targets = target_time_s[begin:end]
        try:
            result = locate_exact_first_event_batch(
                prepared.geometry,
                path_kind[begin:end],
                current_position_m[bounded_rows],
                current_velocity_m_s[bounded_rows],
                acceleration_m_s2[bounded_rows],
                start_time_s=current_time_s[bounded_rows],
                target_time_s=bounded_targets,
                geometry_rtol=event.geometry_rtol,
                roundoff_ulps=event.roundoff_ulps,
                candidate_capacity=capacity,
                certified_departing_facet_id=departing_facet_id[bounded_rows],
            )
        except (EventLocationError, GeometryPreparationError, ValueError) as error:
            raise EngineError("bounded exact event batch violated its capacity") from error
        yield bounded_rows, bounded_targets, result
        begin = end


def _bounded_event_row_stop(
    candidate_count: np.ndarray,
    begin: int,
    capacity: int,
) -> int:
    if capacity == 0:
        if bool((candidate_count[begin:] != 0).any()):
            raise EngineError("event candidates exist without prepared capacity")
        return candidate_count.size
    total = 0
    end = begin
    while end < candidate_count.size:
        row_count = int(candidate_count[end])
        if row_count + 1 > capacity:
            raise EngineError("one event row exceeds its prepared candidate capacity")
        event_rows = end - begin + 1
        if end > begin and total + row_count + event_rows > capacity:
            break
        total += row_count
        end += 1
    return end


def _consume_exact_wave(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    wave_rows: np.ndarray,
    target_time_s: np.ndarray,
    event: ExactEventBatch,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    stack_target_s: np.ndarray,
    stack_depth: np.ndarray,
    stack_interactions: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    surface_release_state: np.ndarray,
    replay: _ReplayBuffer,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    failed_rows = wave_rows[event.status == EXACT_STATUS_FAILURE]
    _fail_event_rows(
        particle_index,
        failed_rows,
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failures,
        _FAILURE_INDETERMINATE_EVENT,
    )
    split_mask = event.status == EXACT_STATUS_WALL
    wave_top = stack_top[wave_rows]
    interaction_count = stack_interactions[wave_rows, np.maximum(wave_top, 0)]
    split_mask &= interaction_count >= prepared.case.spec.solver.event.max_interactions_per_step
    split_rows = wave_rows[split_mask]
    _split_exact_wave_rows(
        prepared,
        particle_index,
        split_rows,
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        stack_target_s,
        stack_depth,
        stack_interactions,
        stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failures,
        statistics,
    )
    clear_rows = wave_rows[event.status == EXACT_STATUS_CLEAR]
    _commit_exact_clear_rows(
        prepared,
        particle_index,
        clear_rows,
        target_time_s[event.status == EXACT_STATUS_CLEAR],
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        replay,
        failures,
        statistics,
        failure_reason_code,
    )
    axis_wave_rows = np.flatnonzero(event.status == EXACT_STATUS_AXIS).astype("<i8", copy=False)
    _commit_exact_axis_rows(
        prepared,
        particle_index,
        wave_rows,
        axis_wave_rows,
        event,
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
        replay,
        failures,
        statistics,
        failure_reason_code,
    )
    wall_wave_rows = np.flatnonzero((event.status == EXACT_STATUS_WALL) & ~split_mask).astype(
        "<i8", copy=False
    )
    _commit_exact_wall_rows(
        prepared,
        particle_index,
        wave_rows,
        wall_wave_rows,
        event,
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        stack_interactions,
        stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        physical_boundary_event_ordinal,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
        surface_release_state,
        replay,
        pending,
        failures,
        statistics,
        failure_reason_code,
    )


def _evaluate_exact_targets(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    selected_rows: np.ndarray,
    target_time_s: np.ndarray,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
) -> tuple[_ExactEndpointBatch, list[tuple[int, _ParticleFailure]]]:
    """Reintegrate variable exact prefixes in one stable row-target batch."""

    count = selected_rows.size
    endpoint = _ExactEndpointBatch(
        np.zeros(count, dtype=np.bool_),
        np.empty((count, 2), dtype="<f8"),
        np.empty((count, 2), dtype="<f8"),
        np.empty(count, dtype="<f8"),
    )
    failures: list[tuple[int, _ParticleFailure]] = []
    if not count:
        return endpoint, failures
    try:
        proposal = _build_step_proposal(
            prepared,
            particle_index[selected_rows],
            current_time_s[selected_rows],
            target_time_s,
            current_position_m[selected_rows],
            current_velocity_m_s[selected_rows],
            current_charge_number[selected_rows],
            None,
        )
    except (FieldLocationError, PhysicsEvaluationError, ValueError) as error:
        raise EngineError("exact target batch could not be evaluated") from error
    else:
        _record_exact_target_proposal(
            prepared,
            proposal,
            np.arange(count, dtype="<i8"),
            replay,
            endpoint,
            failures,
            statistics,
            source_rows=selected_rows,
        )
    return endpoint, failures


def _record_exact_target_proposal(
    prepared: _PreparedRun,
    proposal: StepProposal,
    endpoint_rows: np.ndarray,
    replay: _ReplayBuffer,
    endpoint: _ExactEndpointBatch,
    failures: list[tuple[int, _ParticleFailure]],
    statistics: _EventStatistics,
    *,
    source_rows: np.ndarray | None = None,
) -> None:
    source = endpoint_rows if source_rows is None else source_rows
    codes = _proposal_numerical_failure_codes(proposal.numerical_status)
    codes[(codes == 0) & ~proposal.support_inside] = _FAILURE_FIELD_SUPPORT
    codes[(codes == 0) & ~proposal.applicability_inside] = _FAILURE_MODEL_APPLICABILITY
    failures.extend(
        (
            int(source[local_row]),
            _proposal_row_failure(proposal, int(local_row), codes[local_row]),
        )
        for local_row in np.flatnonzero(codes)
    )
    valid_local = np.flatnonzero(codes == 0).astype("<i8", copy=False)
    if not valid_local.size:
        return
    destination = endpoint_rows[valid_local]
    endpoint.valid[destination] = True
    endpoint.position_m[destination] = proposal.end_position_m[valid_local]
    endpoint.velocity_m_s[destination] = proposal.end_velocity_m_s[valid_local]
    endpoint.charge_number[destination] = proposal.end_charge_number[valid_local]
    _commit_proposal_field_cell(prepared, proposal, valid_local)
    _record_replay_rows(replay, proposal, valid_local, prepared.case.data.coordinate_system)
    statistics.accepted_particle_pieces += int(valid_local.size)


def _commit_exact_clear_rows(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    rows: np.ndarray,
    target_time_s: np.ndarray,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    replay: _ReplayBuffer,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    endpoint, proposal_failures = _evaluate_exact_targets(
        prepared,
        particle_index,
        rows,
        target_time_s,
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        replay,
        statistics,
    )
    _commit_event_proposal_failures(
        particle_index,
        proposal_failures,
        stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failures,
    )
    for selected_row in np.flatnonzero(endpoint.valid):
        row = int(rows[selected_row])
        particle = int(particle_index[row])
        current_time_s[row] = target_time_s[selected_row]
        current_position_m[row] = endpoint.position_m[selected_row]
        current_velocity_m_s[row] = endpoint.velocity_m_s[selected_row]
        current_charge_number[row] = endpoint.charge_number[selected_row]
        position_m[particle] = endpoint.position_m[selected_row]
        velocity_m_s[particle] = endpoint.velocity_m_s[selected_row]
        charge_number[particle] = endpoint.charge_number[selected_row]
        stack_top[row] -= 1


def _commit_exact_axis_rows(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    wave_rows: np.ndarray,
    axis_wave_rows: np.ndarray,
    event: ExactEventBatch,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    replay: _ReplayBuffer,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    if not axis_wave_rows.size:
        return
    rows = wave_rows[axis_wave_rows]
    targets = event.time_s[axis_wave_rows]
    positive = targets > current_time_s[rows]
    endpoint = _ExactEndpointBatch(
        np.ones(rows.size, dtype=np.bool_),
        current_position_m[rows].copy(),
        current_velocity_m_s[rows].copy(),
        current_charge_number[rows].copy(),
    )
    if bool(positive.any()):
        evaluated, proposal_failures = _evaluate_exact_targets(
            prepared,
            particle_index,
            rows[positive],
            targets[positive],
            current_time_s,
            current_position_m,
            current_velocity_m_s,
            current_charge_number,
            replay,
            statistics,
        )
        endpoint.valid[positive] = evaluated.valid
        endpoint.position_m[positive] = evaluated.position_m
        endpoint.velocity_m_s[positive] = evaluated.velocity_m_s
        endpoint.charge_number[positive] = evaluated.charge_number
        _commit_event_proposal_failures(
            particle_index,
            proposal_failures,
            stack_top,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
        )
    for selected_row in np.flatnonzero(endpoint.valid):
        row = int(rows[selected_row])
        particle = int(particle_index[row])
        folded_position, folded_velocity = fold_rz_position_vector(
            np.asarray([0.0, endpoint.position_m[selected_row, 1]], dtype="<f8"),
            endpoint.velocity_m_s[selected_row],
        )
        current_time_s[row] = targets[selected_row]
        current_position_m[row] = folded_position
        current_velocity_m_s[row] = folded_velocity
        current_charge_number[row] = endpoint.charge_number[selected_row]
        position_m[particle] = folded_position
        velocity_m_s[particle] = folded_velocity
        charge_number[particle] = endpoint.charge_number[selected_row]
        _set_exact_origin(
            particle,
            targets[selected_row],
            folded_position,
            folded_velocity,
            exact_origin_time_s,
            exact_origin_position_m,
            exact_origin_velocity_m_s,
        )
        _record_replay_jump(
            replay,
            particle,
            float(targets[selected_row]),
            folded_position,
            folded_velocity,
            float(endpoint.charge_number[selected_row]),
        )
        statistics.axis_crossings += 1


def _split_exact_wave_rows(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    rows: np.ndarray,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    stack_target_s: np.ndarray,
    stack_depth: np.ndarray,
    stack_interactions: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
) -> None:
    maximum = prepared.case.spec.solver.event.max_refinements
    for row_value in rows:
        row = int(row_value)
        top = int(stack_top[row])
        depth = int(stack_depth[row, top])
        target = float(stack_target_s[row, top])
        midpoint = current_time_s[row] + 0.5 * (target - current_time_s[row])
        if depth >= maximum or not current_time_s[row] < midpoint < target:
            _fail_event_rows(
                particle_index,
                np.asarray([row], dtype="<i8"),
                current_time_s,
                current_position_m,
                current_velocity_m_s,
                current_charge_number,
                stack_top,
                position_m,
                velocity_m_s,
                charge_number,
                active,
                lifecycle,
                terminal_time_s,
                event_ordinal,
                failure_reason_code,
                failures,
                _FAILURE_NUMERICAL_EVENT_BUDGET,
            )
            continue
        child_depth = depth + 1
        stack_depth[row, top] = child_depth
        stack_interactions[row, top] = 0
        stack_top[row] = top + 1
        stack_target_s[row, top + 1] = midpoint
        stack_depth[row, top + 1] = child_depth
        stack_interactions[row, top + 1] = 0
        statistics.residual_splits += 1


def _fail_event_rows(
    particle_index: np.ndarray,
    rows: np.ndarray,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failures: _FailureEventBuffer,
    reason_code: np.uint16,
) -> None:
    for row_value in rows:
        row = int(row_value)
        particle = int(particle_index[row])
        _mark_particle_failed(
            particle,
            _ParticleFailure(
                reason_code,
                float(current_time_s[row]),
                current_position_m[row],
                current_velocity_m_s[row],
                float(current_charge_number[row]),
            ),
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
        )
        stack_top[row] = -1


def _commit_event_proposal_failures(
    particle_index: np.ndarray,
    proposal_failures: list[tuple[int, _ParticleFailure]],
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failures: _FailureEventBuffer,
) -> None:
    for row, failure in proposal_failures:
        particle = int(particle_index[row])
        _mark_particle_failed(
            particle,
            failure,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failures,
        )
        stack_top[row] = -1


def _commit_exact_wall_rows(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    wave_rows: np.ndarray,
    wall_wave_rows: np.ndarray,
    event: ExactEventBatch,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    stack_interactions: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    surface_release_state: np.ndarray,
    replay: _ReplayBuffer,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
    statistics: _EventStatistics,
    failure_reason_code: np.ndarray,
) -> None:
    if not wall_wave_rows.size:
        return
    rows = wave_rows[wall_wave_rows]
    endpoint, proposal_failures = _evaluate_exact_targets(
        prepared,
        particle_index,
        rows,
        event.time_s[wall_wave_rows],
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        replay,
        statistics,
    )
    _commit_event_proposal_failures(
        particle_index,
        proposal_failures,
        stack_top,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failures,
    )
    valid = np.flatnonzero(endpoint.valid).astype("<i8", copy=False)
    if not valid.size:
        return
    selected_wave_rows = wall_wave_rows[valid]
    offsets, candidates = _select_event_candidate_rows(event, selected_wave_rows)
    selected_rows = rows[valid]
    particles = particle_index[selected_rows]
    draws = wall_uniform_batch(
        prepared.case.spec.solver.seed,
        prepared.schedule.particle_id[particles],
        physical_boundary_event_ordinal[particles],
    )
    responses = resolve_boundary_responses_batch(
        prepared.compiled_boundary_rules,
        offsets,
        candidates,
        prepared.geometry.boundary_id,
        prepared.geometry.group_id,
        prepared.geometry.facet_normal,
        endpoint.velocity_m_s[valid],
        draws,
        roundoff_ulps=prepared.case.spec.solver.event.roundoff_ulps,
    )
    for response_row, selected_value in enumerate(valid):
        endpoint_row = int(selected_value)
        row = int(rows[endpoint_row])
        event_row = int(wall_wave_rows[endpoint_row])
        particle = int(particle_index[row])
        if responses.status[response_row] != BOUNDARY_STATUS_OK:
            failure = _ParticleFailure(
                _FAILURE_INDETERMINATE_BOUNDARY_POLICY,
                float(event.time_s[event_row]),
                event.position_m[event_row],
                endpoint.velocity_m_s[endpoint_row],
                float(endpoint.charge_number[endpoint_row]),
            )
            _commit_event_proposal_failures(
                particle_index,
                [(row, failure)],
                stack_top,
                position_m,
                velocity_m_s,
                charge_number,
                active,
                lifecycle,
                terminal_time_s,
                event_ordinal,
                failure_reason_code,
                failures,
            )
            continue
        candidate_begin = int(offsets[response_row])
        candidate_end = int(offsets[response_row + 1])
        _commit_exact_wall_response(
            prepared,
            row,
            particle,
            event_row,
            event,
            tuple(int(value) for value in candidates[candidate_begin:candidate_end]),
            int(responses.law[response_row]),
            int(responses.outcome[response_row]),
            responses.velocity_post_m_s[response_row],
            int(responses.primary_facet_id[response_row]),
            responses.effective_normal[response_row],
            bool(responses.remains_active[response_row]),
            endpoint.velocity_m_s[endpoint_row],
            float(endpoint.charge_number[endpoint_row]),
            current_time_s,
            current_position_m,
            current_velocity_m_s,
            current_charge_number,
            stack_interactions,
            stack_top,
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            physical_boundary_event_ordinal,
            exact_origin_time_s,
            exact_origin_position_m,
            exact_origin_velocity_m_s,
            surface_release_state,
            pending,
            replay,
            statistics,
        )


def _select_event_candidate_rows(
    event: ExactEventBatch | CurvedEventBatch,
    event_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty(event_rows.size + 1, dtype="<i8")
    offsets[0] = 0
    for output_row, event_row_value in enumerate(event_rows):
        event_row = int(event_row_value)
        offsets[output_row + 1] = (
            offsets[output_row]
            + event.candidate_offsets[event_row + 1]
            - event.candidate_offsets[event_row]
        )
    candidates = np.empty(int(offsets[-1]), dtype="<i8")
    for output_row, event_row_value in enumerate(event_rows):
        event_row = int(event_row_value)
        source_begin = int(event.candidate_offsets[event_row])
        source_end = int(event.candidate_offsets[event_row + 1])
        candidates[offsets[output_row] : offsets[output_row + 1]] = event.candidate_facet_ids[
            source_begin:source_end
        ]
    return offsets, candidates


def _commit_exact_wall_response(
    prepared: _PreparedRun,
    row: int,
    particle_index: int,
    event_row: int,
    event: ExactEventBatch,
    candidate_facet_ids: tuple[int, ...],
    law_code: int,
    outcome_code: int,
    velocity_post_m_s: np.ndarray,
    primary_facet_id: int,
    effective_normal: np.ndarray,
    remains_active: bool,
    velocity_pre_m_s: np.ndarray,
    charge_pre_number: float,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    stack_interactions: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    surface_release_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    replay: _ReplayBuffer,
    statistics: _EventStatistics,
) -> None:
    if int(event_ordinal[particle_index]) >= np.iinfo(np.uint32).max:
        raise EngineError("logical event ordinal exhausted uint32")
    if int(physical_boundary_event_ordinal[particle_index]) >= np.iinfo(np.uint32).max:
        raise EngineError("physical boundary event ordinal exhausted uint32")
    raw_hit = BoundaryHit(
        float(event.time_s[event_row]),
        event.position_m[event_row].copy(),
        int(event.primary_facet_id[event_row]),
        candidate_facet_ids,
        event.normal[event_row].copy(),
        float(event.position_budget_m[event_row]),
        float(event.time_budget_s[event_row]),
        float(event.localization_residual_m[event_row]),
    )
    resolved_position, projection_residual = _canonical_hit_position(
        prepared.geometry,
        raw_hit,
        primary_facet_id,
    )
    resolved_hit = BoundaryHit(
        raw_hit.time_s,
        resolved_position,
        primary_facet_id,
        candidate_facet_ids,
        effective_normal.copy(),
        raw_hit.position_budget_m,
        raw_hit.time_budget_s,
        max(raw_hit.localization_residual_m, projection_residual),
    )
    position_m[particle_index] = resolved_position
    velocity_m_s[particle_index] = velocity_post_m_s
    charge_number[particle_index] = charge_pre_number
    event_ordinal[particle_index] += np.uint32(1)
    physical_boundary_event_ordinal[particle_index] += np.uint32(1)
    _set_boundary_lifecycle(
        particle_index,
        resolved_hit.time_s,
        outcome_code,
        remains_active,
        active,
        lifecycle,
        terminal_time_s,
    )
    logical_ordinal = int(event_ordinal[particle_index])
    _append_boundary_event(
        pending,
        particle_index,
        resolved_hit,
        law_code,
        outcome_code,
        velocity_pre_m_s,
        velocity_post_m_s,
        charge_pre_number,
        logical_ordinal,
    )
    surface_release_state[particle_index] = SURFACE_STATE_RESOLVED
    statistics.wall_interactions += 1
    current_time_s[row] = resolved_hit.time_s
    if not remains_active:
        stack_top[row] = -1
        return
    current_position_m[row] = resolved_position
    current_velocity_m_s[row] = velocity_post_m_s
    current_charge_number[row] = charge_pre_number
    top = int(stack_top[row])
    stack_interactions[row, top] += 1
    _set_exact_origin(
        particle_index,
        resolved_hit.time_s,
        resolved_position,
        velocity_post_m_s,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
    )
    _record_replay_jump(
        replay,
        particle_index,
        resolved_hit.time_s,
        resolved_position,
        velocity_post_m_s,
        charge_pre_number,
    )


def _set_boundary_lifecycle(
    particle_index: int,
    time_s: float,
    outcome_code: int,
    remains_active: bool,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
) -> None:
    if remains_active:
        active[particle_index] = True
        lifecycle[particle_index] = _LIFECYCLE_ACTIVE
        terminal_time_s[particle_index] = np.inf
    elif outcome_code == BOUNDARY_OUTCOME_STUCK:
        active[particle_index] = False
        lifecycle[particle_index] = _LIFECYCLE_STUCK
        terminal_time_s[particle_index] = time_s
    elif outcome_code == BOUNDARY_OUTCOME_ESCAPED:
        active[particle_index] = False
        lifecycle[particle_index] = _LIFECYCLE_ESCAPED
        terminal_time_s[particle_index] = time_s
    elif outcome_code == BOUNDARY_OUTCOME_HELD:
        active[particle_index] = False
        lifecycle[particle_index] = _LIFECYCLE_HELD
        terminal_time_s[particle_index] = time_s
    else:
        raise EngineError(f"unsupported boundary outcome code: {outcome_code}")


def _set_exact_origin(
    particle_index: int,
    time_s: float,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
) -> None:
    exact_origin_time_s[particle_index] = time_s
    exact_origin_position_m[particle_index] = position_m
    exact_origin_velocity_m_s[particle_index] = velocity_m_s


def _allocate_boundary_event_buffer(
    row_capacity: int,
    arena_capacity: int,
) -> _BoundaryEventBuffer:
    """Allocate the exact numeric columns retained across one event wave."""

    if row_capacity == 0:
        return _BoundaryEventBuffer(
            np.empty(0, dtype="<i8"),
            np.empty(0, dtype="<f8"),
            np.empty(0, dtype="<u4"),
            np.empty(0, dtype="<i8"),
            np.empty((0, 2), dtype="<f8"),
            np.empty((0, 2), dtype="<f8"),
            np.empty((0, 2), dtype="<f8"),
            np.empty((0, 2), dtype="<f8"),
            np.empty(0, dtype="<f8"),
            np.empty(0, dtype="<u1"),
            np.empty(0, dtype="<u1"),
            np.empty(0, dtype="<f8"),
            np.empty(0, dtype="<f8"),
            np.empty(0, dtype="<f8"),
            np.empty(0, dtype="<i8"),
            np.empty(0, dtype="<i8"),
            arena_capacity,
        )
    return _BoundaryEventBuffer(
        np.empty(row_capacity, dtype="<i8"),
        np.empty(row_capacity, dtype="<f8"),
        np.empty(row_capacity, dtype="<u4"),
        np.empty(row_capacity, dtype="<i8"),
        np.empty((row_capacity, 2), dtype="<f8"),
        np.empty((row_capacity, 2), dtype="<f8"),
        np.empty((row_capacity, 2), dtype="<f8"),
        np.empty((row_capacity, 2), dtype="<f8"),
        np.empty(row_capacity, dtype="<f8"),
        np.empty(row_capacity, dtype="<u1"),
        np.empty(row_capacity, dtype="<u1"),
        np.empty(row_capacity, dtype="<f8"),
        np.empty(row_capacity, dtype="<f8"),
        np.empty(row_capacity, dtype="<f8"),
        np.zeros(row_capacity + 1, dtype="<i8"),
        np.empty(arena_capacity, dtype="<i8"),
        arena_capacity,
    )


def _allocate_failure_event_buffer(capacity: int) -> _FailureEventBuffer:
    """Allocate one failure row per slab particle, the physical maximum."""

    return _FailureEventBuffer(
        np.empty(capacity, dtype="<i8"),
        np.empty(capacity, dtype="<f8"),
        np.empty(capacity, dtype="<u4"),
        np.empty(capacity, dtype="<u2"),
    )


def _append_boundary_event(
    pending: _BoundaryEventBuffer,
    particle_index: int,
    hit: BoundaryHit,
    law_code: int,
    outcome_code: int,
    velocity_pre_m_s: np.ndarray,
    velocity_post_m_s: np.ndarray,
    charge_pre_number: float,
    event_ordinal: int,
) -> None:
    """Append one event to the shared row/candidate capacity arena."""

    row = pending.row_count
    candidates = hit.candidate_facet_ids
    candidate_end = pending.candidate_count + len(candidates)
    if row >= pending.particle_index.size:
        raise EngineError("boundary-event row buffer exhausted")
    if row + 1 + candidate_end > pending.arena_capacity:
        raise EngineError("boundary-event candidate arena exhausted")
    if not (0 < law_code < _BOUNDARY_LAW_OUTPUT.size):
        raise EngineError("compiled boundary response returned an unknown law code")
    if not (0 < outcome_code < _BOUNDARY_OUTCOME_OUTPUT.size):
        raise EngineError("compiled boundary response returned an unknown outcome code")
    pending.particle_index[row] = particle_index
    pending.time_s[row] = hit.time_s
    pending.event_ordinal[row] = event_ordinal
    pending.primary_facet_id[row] = hit.facet_id
    pending.position_m[row] = hit.position_m
    pending.normal[row] = hit.normal
    pending.velocity_pre_m_s[row] = velocity_pre_m_s
    pending.velocity_post_m_s[row] = velocity_post_m_s
    pending.charge_pre_number[row] = charge_pre_number
    pending.law_code[row] = law_code
    pending.outcome_code[row] = outcome_code
    pending.localization_residual_m[row] = hit.localization_residual_m
    pending.position_budget_m[row] = hit.position_budget_m
    pending.time_budget_s[row] = hit.time_budget_s
    pending.candidate_offset[row] = pending.candidate_count
    pending.candidate_facet_id[pending.candidate_count : candidate_end] = candidates
    pending.candidate_offset[row + 1] = candidate_end
    pending.row_count = row + 1
    pending.candidate_count = candidate_end


def _append_failure_event(
    pending: _FailureEventBuffer,
    particle_index: int,
    time_s: float,
    event_ordinal: int,
    reason_code: int,
) -> None:
    row = pending.count
    if row >= pending.particle_index.size:
        raise EngineError("failure-event buffer exceeded one row per slab particle")
    pending.particle_index[row] = particle_index
    pending.time_s[row] = time_s
    pending.event_ordinal[row] = event_ordinal
    pending.reason_code[row] = reason_code
    pending.count = row + 1


def _flush_boundary_event_wave(
    writer: ResultWriter,
    prepared: _PreparedRun,
    pending: _BoundaryEventBuffer,
) -> None:
    """Synchronously flush one capacity-bounded canonical event wave."""

    count = pending.row_count
    if not count:
        return
    if count > prepared.memory_plan.slab_particles:
        raise EngineError("boundary-event wave exceeds its prepared row capacity")
    if count + pending.candidate_count > prepared.event_candidate_capacity:
        raise EngineError("boundary-event wave exceeds its prepared candidate capacity")
    particle_ids = prepared.schedule.particle_id[pending.particle_index[:count]]
    order = np.lexsort((pending.event_ordinal[:count], particle_ids, pending.time_s[:count]))
    del particle_ids
    writer.write_boundary_events(_pack_boundary_events(prepared, pending, order))
    pending.row_count = 0
    pending.candidate_count = 0
    pending.candidate_offset[0] = 0


def _write_failure_events(
    writer: ResultWriter,
    prepared: _PreparedRun,
    pending: _FailureEventBuffer,
) -> None:
    count = pending.count
    if not count:
        return
    rows = pending.particle_index[:count]
    particle_ids = prepared.schedule.particle_id[rows]
    order = np.lexsort((pending.event_ordinal[:count], particle_ids, pending.time_s[:count]))
    payload = FailureEvents(
        time_s=pending.time_s[:count][order],
        particle_id=particle_ids[order],
        event_ordinal=pending.event_ordinal[:count][order],
        reason_code=pending.reason_code[:count][order],
    )
    del particle_ids
    writer.write_failure_events(payload)


def _mark_particle_failed(
    particle_index: int,
    failure: _ParticleFailure,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
    event_ordinal: np.ndarray,
    failure_reason_code: np.ndarray,
    failures: _FailureEventBuffer,
) -> None:
    """Commit one particle-local failure without changing wall RNG identity."""

    if int(event_ordinal[particle_index]) >= np.iinfo(np.uint32).max:
        raise EngineError("logical event ordinal exhausted uint32")
    if not bool(
        math.isfinite(failure.time_s)
        and np.isfinite(failure.position_m).all()
        and np.isfinite(failure.velocity_m_s).all()
        and math.isfinite(failure.charge_number)
    ):
        raise EngineError("localized particle failure contains non-finite state")
    position_m[particle_index] = failure.position_m
    velocity_m_s[particle_index] = failure.velocity_m_s
    charge_number[particle_index] = failure.charge_number
    active[particle_index] = False
    lifecycle[particle_index] = _LIFECYCLE_FAILED
    terminal_time_s[particle_index] = failure.time_s
    failure_reason_code[particle_index] = np.uint16(failure.reason_code)
    event_ordinal[particle_index] += np.uint32(1)
    _append_failure_event(
        failures,
        particle_index,
        failure.time_s,
        int(event_ordinal[particle_index]),
        failure.reason_code,
    )


def _pack_boundary_events(
    prepared: _PreparedRun,
    pending: _BoundaryEventBuffer,
    order: np.ndarray,
) -> BoundaryEvents:
    """Build one canonical writer payload from numeric staging columns."""

    geometry = prepared.geometry
    schedule = prepared.schedule
    count = pending.row_count
    rows = pending.particle_index[:count][order]
    primary = pending.primary_facet_id[:count][order]
    candidate_offset, candidate_facet = _ordered_candidate_table(pending, order)
    return BoundaryEvents(
        time_s=pending.time_s[:count][order],
        particle_id=schedule.particle_id[rows],
        event_ordinal=pending.event_ordinal[:count][order],
        primary_facet_id=primary,
        boundary_id=geometry.boundary_id[primary],
        material_id=geometry.material_id[primary],
        position_m=pending.position_m[:count][order],
        normal=pending.normal[:count][order],
        velocity_pre_m_s=pending.velocity_pre_m_s[:count][order],
        velocity_post_m_s=pending.velocity_post_m_s[:count][order],
        charge_number_pre=pending.charge_pre_number[:count][order],
        charge_number_post=pending.charge_pre_number[:count][order],
        model_weight=schedule.model_weight[rows],
        law_id=_BOUNDARY_LAW_OUTPUT[pending.law_code[:count][order]],
        outcome=_BOUNDARY_OUTCOME_OUTPUT[pending.outcome_code[:count][order]],
        localization_residual_m=pending.localization_residual_m[:count][order],
        position_budget_m=pending.position_budget_m[:count][order],
        time_budget_s=pending.time_budget_s[:count][order],
        candidate_offset=candidate_offset,
        candidate_facet_id=candidate_facet,
    )


def _ordered_candidate_table(
    pending: _BoundaryEventBuffer,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder one numeric ragged table into canonical event-row order."""

    offsets = np.zeros(order.size + 1, dtype="<i8")
    facets = np.empty(pending.candidate_count, dtype="<i8")
    for output_row, source_row_value in enumerate(order):
        source_row = int(source_row_value)
        begin = int(pending.candidate_offset[source_row])
        end = int(pending.candidate_offset[source_row + 1])
        width = end - begin
        output_begin = int(offsets[output_row])
        facets[output_begin : output_begin + width] = pending.candidate_facet_id[begin:end]
        offsets[output_row + 1] = output_begin + width
    return offsets, facets


def _allocate_replay_buffer(
    time_s: tuple[float, ...],
    schedule: ParticleSchedule,
    probe_particle_index: np.ndarray,
    *,
    include_all_particles: bool,
) -> _ReplayBuffer:
    """Allocate the exact numeric columns needed by one macro's outputs."""

    times = np.asarray(time_s, dtype="<f8")
    if not times.size:
        rows = np.empty(0, dtype="<i8")
    elif include_all_particles:
        rows = np.arange(schedule.particle_count, dtype="<i8")
    else:
        rows = probe_particle_index.copy()
    shape = (times.size, rows.size)
    replay = _ReplayBuffer(
        times,
        rows,
        np.full((*shape, 2), np.nan, dtype="<f8"),
        np.full((*shape, 2), np.nan, dtype="<f8"),
        np.full(shape, np.nan, dtype="<f8"),
        np.zeros(shape, dtype=np.bool_),
        np.full(shape, _LIFECYCLE_PENDING, dtype="<u1"),
    )
    for time_row, requested_time_s in enumerate(times):
        released = schedule.release_time_s[rows] == requested_time_s
        selected = np.flatnonzero(released).astype("<i8", copy=False)
        particles = rows[selected]
        replay.position_m[time_row, selected] = schedule.position_m[particles]
        replay.velocity_m_s[time_row, selected] = schedule.velocity_m_s[particles]
        replay.charge_number[time_row, selected] = schedule.charge_number[particles]
        replay.presence[time_row, selected] = True
        replay.lifecycle[time_row, selected] = _LIFECYCLE_ACTIVE
    return replay


def _record_replay_rows(
    replay: _ReplayBuffer,
    proposal: StepProposal,
    local_rows: np.ndarray,
    coordinate_system: str,
    *,
    exclude_target: bool = False,
) -> None:
    """Scatter accepted proposal states directly into requested numeric slots."""

    if not replay.time_s.size or not local_rows.size or not replay.particle_index.size:
        return
    for time_row, requested_time_s in enumerate(replay.time_s):
        available = proposal.start_time_s[local_rows] <= requested_time_s
        available &= requested_time_s <= proposal.target_time_s[local_rows]
        if exclude_target:
            available &= requested_time_s < proposal.target_time_s[local_rows]
        proposal_rows = local_rows[available]
        if not proposal_rows.size:
            continue
        particles = proposal.particle_index[proposal_rows]
        slots = np.searchsorted(replay.particle_index, particles)
        selected = slots < replay.particle_index.size
        selected_rows = np.flatnonzero(selected).astype("<i8", copy=False)
        if selected_rows.size:
            selected[selected_rows] = (
                replay.particle_index[slots[selected_rows]] == particles[selected_rows]
            )
        proposal_rows = proposal_rows[selected]
        slots = slots[selected]
        if not proposal_rows.size:
            continue
        sample = proposal.state_at_rows(float(requested_time_s), proposal_rows)
        if bool((sample.numerical_status != NUMERICAL_STATUS_OK).any()):
            raise EngineError("accepted trajectory replay produced a numerical failure")
        if not bool(sample.support_inside.all() and sample.applicability_inside.all()):
            raise EngineError("trajectory frame is outside field or model support")
        sample_position = sample.position_m
        sample_velocity = sample.velocity_m_s
        if coordinate_system == "axisymmetric_rz":
            sample_position, sample_velocity, _ = rz_signed_stage_to_canonical(
                sample_position,
                sample_velocity,
            )
        replay.position_m[time_row, slots] = sample_position
        replay.velocity_m_s[time_row, slots] = sample_velocity
        replay.charge_number[time_row, slots] = sample.charge_number
        replay.presence[time_row, slots] = True
        replay.lifecycle[time_row, slots] = _LIFECYCLE_ACTIVE


def _record_replay_jump(
    replay: _ReplayBuffer,
    particle_index: int,
    time_s: float,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: float,
) -> None:
    """Overlay one right-continuous zero-time state in canonical event order."""

    time_row = int(np.searchsorted(replay.time_s, time_s))
    particle_row = int(np.searchsorted(replay.particle_index, particle_index))
    if time_row >= replay.time_s.size or replay.time_s[time_row] != time_s:
        return
    if (
        particle_row >= replay.particle_index.size
        or int(replay.particle_index[particle_row]) != particle_index
    ):
        return
    replay.position_m[time_row, particle_row] = position_m
    replay.velocity_m_s[time_row, particle_row] = velocity_m_s
    replay.charge_number[time_row, particle_row] = charge_number
    replay.presence[time_row, particle_row] = True
    replay.lifecycle[time_row, particle_row] = _LIFECYCLE_ACTIVE


def _finalize_replay_buffer(
    replay: _ReplayBuffer,
    schedule: ParticleSchedule,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    lifecycle: np.ndarray,
    terminal_time_s: np.ndarray,
) -> None:
    """Apply terminal states and verify every eligible requested slot."""

    rows = replay.particle_index
    for time_row, requested_time_s in enumerate(replay.time_s):
        unavailable = (
            (lifecycle[rows] == _LIFECYCLE_ESCAPED) | (lifecycle[rows] == _LIFECYCLE_FAILED)
        ) & (terminal_time_s[rows] <= requested_time_s)
        eligible = (schedule.release_time_s[rows] <= requested_time_s) & ~unavailable
        retained = (
            (lifecycle[rows] == _LIFECYCLE_STUCK) | (lifecycle[rows] == _LIFECYCLE_HELD)
        ) & (terminal_time_s[rows] <= requested_time_s)
        retained_rows = np.flatnonzero(retained).astype("<i8", copy=False)
        particles = rows[retained_rows]
        replay.position_m[time_row, retained_rows] = position_m[particles]
        replay.velocity_m_s[time_row, retained_rows] = velocity_m_s[particles]
        replay.charge_number[time_row, retained_rows] = charge_number[particles]
        replay.presence[time_row, retained_rows] = True
        replay.lifecycle[time_row, retained_rows] = lifecycle[particles]
        replay.presence[time_row, unavailable] = False
        if not np.array_equal(replay.presence[time_row], eligible):
            raise EngineError("trajectory replay did not fill every eligible particle state")
        if not bool(
            np.isfinite(replay.position_m[time_row, eligible]).all()
            and np.isfinite(replay.velocity_m_s[time_row, eligible]).all()
            and np.isfinite(replay.charge_number[time_row, eligible]).all()
        ):
            raise EngineError("trajectory frame contains non-finite particle state")


def _frame_from_replay(
    time_s: float,
    schedule: ParticleSchedule,
    replay: _ReplayBuffer,
    selected_particle_index: np.ndarray | None = None,
) -> TrajectoryFrame:
    """Materialize one requested result row from the completed macro buffer."""

    time_row = int(np.searchsorted(replay.time_s, time_s))
    if time_row >= replay.time_s.size or replay.time_s[time_row] != time_s:
        raise EngineError("requested output time is absent from the replay buffer")
    selected = replay.particle_index if selected_particle_index is None else selected_particle_index
    slots = np.searchsorted(replay.particle_index, selected)
    if bool((slots >= replay.particle_index.size).any()) or not np.array_equal(
        replay.particle_index[slots], selected
    ):
        raise EngineError("requested output particles are absent from the replay buffer")
    present = replay.presence[time_row, slots]
    rows = selected[present]
    slots = slots[present]
    return TrajectoryFrame(
        time_s=time_s,
        particle_id=schedule.particle_id[rows],
        position_m=replay.position_m[time_row, slots].copy(),
        velocity_m_s=replay.velocity_m_s[time_row, slots].copy(),
        charge_number=replay.charge_number[time_row, slots].copy(),
        lifecycle=replay.lifecycle[time_row, slots].copy(),
    )


def _probe_from_replay(
    time_s: float,
    particle_index: np.ndarray,
    schedule: ParticleSchedule,
    replay: _ReplayBuffer,
) -> ProbeFrame:
    frame = _frame_from_replay(time_s, schedule, replay, particle_index)
    return ProbeFrame(
        time_s=frame.time_s,
        particle_id=frame.particle_id,
        position_m=frame.position_m,
        velocity_m_s=frame.velocity_m_s,
        charge_number=frame.charge_number,
        lifecycle=frame.lifecycle,
    )


def _lifecycle_series_at(time_s: float, lifecycle: np.ndarray) -> LifecycleSeries:
    return LifecycleSeries(
        time_s=np.asarray([time_s], dtype="<f8"),
        pending=np.asarray([np.count_nonzero(lifecycle == _LIFECYCLE_PENDING)], dtype="<u8"),
        active=np.asarray([np.count_nonzero(lifecycle == _LIFECYCLE_ACTIVE)], dtype="<u8"),
        stuck=np.asarray([np.count_nonzero(lifecycle == _LIFECYCLE_STUCK)], dtype="<u8"),
        held=np.asarray([np.count_nonzero(lifecycle == _LIFECYCLE_HELD)], dtype="<u8"),
        escaped=np.asarray([np.count_nonzero(lifecycle == _LIFECYCLE_ESCAPED)], dtype="<u8"),
        failed=np.asarray([np.count_nonzero(lifecycle == _LIFECYCLE_FAILED)], dtype="<u8"),
    )


def _schedule_memory_bytes(schedule: ParticleSchedule) -> int:
    """Count arrays owned by the realized, resident particle schedule."""

    arrays = (
        schedule.particle_id,
        schedule.source_id,
        schedule.release_time_s,
        schedule.position_m,
        schedule.velocity_m_s,
        schedule.charge_number,
        schedule.mass_kg,
        schedule.drag_diameter_m,
        schedule.electrostatic_radius_m,
        schedule.displaced_volume_m3,
        schedule.model_weight,
        schedule.material_id,
        schedule.source_facet_id,
        schedule.release_order,
    )
    return sum(array.nbytes for array in arrays)


def _output_buffer_bytes(
    schedule: ParticleSchedule,
    frame_times_s: tuple[float, ...],
    probe_times_s: tuple[float, ...],
    probe_particle_index: np.ndarray,
) -> int:
    """Bound the largest materialized result frame and its row selection.

    Event/failure payloads have a separate per-slab hard bound in the CPU plan;
    the engine flushes them synchronously before reusing the slab.
    """

    frame_rows = 0
    if frame_times_s:
        frame_rows = _release_stop(
            schedule,
            0,
            frame_times_s[-1],
            _PREPARE_SCAN_BATCH_SIZE,
        )
    probe_rows = 0
    if probe_times_s:
        probe_rows = int(
            np.count_nonzero(schedule.release_time_s[probe_particle_index] <= probe_times_s[-1])
        )
    output_rows = max(frame_rows, probe_rows)
    largest_frame_bytes = 0
    if frame_times_s or probe_times_s:
        largest_frame_bytes = output_rows * 64
    return largest_frame_bytes


def _replay_work_bytes(
    case: SimulationCase,
    schedule: ParticleSchedule,
    frame_times_s: tuple[float, ...],
    probe_times_s: tuple[float, ...],
    probe_particle_index: np.ndarray,
) -> int:
    """Count the maximum concrete columnar replay arrays in one macro."""

    time_count = _maximum_replay_times_per_macro(
        frame_times_s,
        probe_times_s,
        start_s=case.spec.time.start_s,
        end_s=case.spec.time.end_s,
        dt_s=case.spec.time.dt_s,
    )
    if time_count == 0:
        return 0
    row_count = schedule.particle_count if frame_times_s else probe_particle_index.size
    state_bytes = (
        time_count
        * row_count
        * (5 * np.dtype("<f8").itemsize + np.dtype(np.bool_).itemsize + np.dtype("<u1").itemsize)
    )
    return (
        time_count * np.dtype("<f8").itemsize + row_count * np.dtype("<i8").itemsize + state_bytes
    )


def _event_work_bytes_per_particle(
    case: SimulationCase,
    *,
    event_paths: bool,
) -> int:
    """Account target, depth, and interaction columns for every event stack."""

    if not event_paths:
        return 0
    stack_capacity = case.spec.solver.event.max_refinements + 1
    return 3 * np.dtype("<i8").itemsize * stack_capacity


def _uses_rk4_dense_certificate(
    case: SimulationCase,
    physics: PhysicsPlan,
    constant_acceleration_m_s2: np.ndarray | None,
) -> bool:
    """Return whether a general stage-evaluated RK4 path is selected."""

    return (
        case.spec.solver.integrator == "rk4_fixed"
        and physics.requires_stage_evaluation
        and constant_acceleration_m_s2 is None
    )


def _certificate_work_bytes_per_particle(
    case: SimulationCase,
    *,
    dense_enabled: bool,
    local_range_enabled: bool,
) -> int:
    """Account dense DFS state and the bounded local-field candidate arena."""

    if not dense_enabled and not local_range_enabled:
        return 0
    stack_capacity = case.spec.solver.event.max_refinements + 1
    # Two float64 interval endpoints per live stack slot; int64 stack top and
    # split count; uint16 failure code.  Round the scalar tail to 24 bytes.
    # The local-field owner additionally retains at most 64 int64 candidate
    # cell IDs plus count, bounded-count, and CSR-offset columns per row.  The
    # extra eight-byte rounding covers the terminal CSR offset.  Primitive
    # lower/upper and predictor transients stay inside the general stage scratch
    # allowance. Exponential local-stage certification uses the candidate arena
    # without the RK4 dense interval stack.
    interval_stack_bytes = 16 * stack_capacity + 24 if dense_enabled else 0
    candidate_arena_bytes = (
        (LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW + 4) * np.dtype("<i8").itemsize
        if local_range_enabled
        else 0
    )
    return interval_stack_bytes + candidate_arena_bytes


def _stochastic_tree_work_bytes_per_particle(physics: PhysicsPlan) -> int:
    """Bound the depth-first OU tree without materializing all leaves."""

    if physics.noise is None:
        return 0
    # One right-child (position and velocity) per depth plus the live root,
    # split children, and normal-pair workspace.  All arrays are float64 2-D.
    return 32 * (physics.noise.interval_tree_depth + 4)


def _surface_release_work_bytes_per_particle(
    schedule: ParticleSchedule,
    *,
    event_paths: bool,
) -> int:
    """Bound compiled initial-contact classification and response subsets."""

    if not event_paths:
        return 0
    if not bool((schedule.source_facet_id >= 0).any()):
        # Event paths still materialize aligned facet/state input, classifier
        # output, interaction and departure columns for table releases.  No
        # boundary-response subset exists when every facet ID is -1.
        return 64
    # 43 B: aligned state/facet columns, action/status/departure/budgets,
    #        interaction result.
    # 108 B: stable response indices/CSR/RNG keys/draw/pre-velocity and the
    #         compiled boundary response columns.
    # 91 B: active curved-response remap and post-response classifier.
    # 8 B: the CSR terminal offset.
    # 16 B: response-row selection and post-response remap coexist in the
    #       active-response branch.  Round 266 B up to an 8-byte row stride.
    return 272


def _event_candidate_capacity(
    geometry: PreparedGeometry,
    memory_limit_bytes: int,
    *,
    enabled: bool,
) -> int:
    """Reserve a bounded CSR facet budget that can always hold one query row."""

    if not enabled or geometry.facet_count == 0:
        return 0
    minimum_bytes = 64 * 1024
    maximum_bytes = 8 * 1024 * 1024
    target_bytes = min(maximum_bytes, max(minimum_bytes, memory_limit_bytes // 64))
    target_capacity = target_bytes // np.dtype("<i8").itemsize
    return max(geometry.facet_count + 1, target_capacity)


def _maximum_replay_times_per_macro(
    frame_times_s: tuple[float, ...],
    probe_times_s: tuple[float, ...],
    *,
    start_s: float,
    end_s: float,
    dt_s: float,
) -> int:
    """Conservatively bound distinct replay times in one macro interval."""

    times = sorted(set(frame_times_s) | set(probe_times_s))
    if not times:
        return 0
    roundoff_s = 4.0 * max(math.ulp(start_s), math.ulp(end_s), math.ulp(dt_s))
    window_s = dt_s + roundoff_s
    begin = 0
    maximum = 0
    for end, time_s in enumerate(times):
        while time_s - times[begin] > window_s:
            begin += 1
        maximum = max(maximum, end - begin + 1)
    return maximum


def _geometry_memory_bytes(geometry: PreparedGeometry) -> int:
    arrays = (
        geometry.facet_node_ids,
        geometry.boundary_id,
        geometry.group_id,
        geometry.material_id,
        geometry.facet_start_m,
        geometry.facet_end_m,
        geometry.facet_normal,
        geometry.facet_length_m,
        geometry.bvh_facet_id,
        geometry.bvh_lower_m,
        geometry.bvh_upper_m,
        geometry.bvh_left,
        geometry.bvh_right,
        geometry.bvh_begin,
        geometry.bvh_end,
        geometry.bvh_skip,
        geometry.volume_bvh_cell_id,
        geometry.volume_bvh_lower_m,
        geometry.volume_bvh_upper_m,
        geometry.volume_bvh_begin,
        geometry.volume_bvh_end,
        geometry.volume_bvh_skip,
        geometry.tri3_edge_length_m,
        geometry.quad4_edge_length_m,
    )
    return sum(array.nbytes for array in arrays)


def _final_particles(
    schedule: ParticleSchedule,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    lifecycle: np.ndarray,
    failure_reason_code: np.ndarray,
    end_s: float,
) -> FinalParticles:
    count = schedule.particle_count
    valid = np.asarray(
        (lifecycle != _LIFECYCLE_ESCAPED) & (lifecycle != _LIFECYCLE_FAILED),
        dtype="<u1",
    )
    return FinalParticles(
        particle_id=schedule.particle_id,
        source_id=schedule.source_id,
        time_s=np.full(count, end_s, dtype="<f8"),
        position_m=position_m,
        velocity_m_s=velocity_m_s,
        charge_number=charge_number,
        lifecycle=lifecycle,
        kinematics_valid=valid,
        failure_reason_code=failure_reason_code,
        mass_kg=schedule.mass_kg,
        drag_diameter_m=schedule.drag_diameter_m,
        electrostatic_radius_m=schedule.electrostatic_radius_m,
        displaced_volume_m3=schedule.displaced_volume_m3,
        model_weight=schedule.model_weight,
        material_id=schedule.material_id,
    )


def _resolved_path_kind(prepared: _PreparedRun) -> str:
    """Describe the path actually selected after exact specializations."""

    if prepared.physics.noise is not None:
        return "cubic_hermite"
    if not prepared.physics.requires_stage_evaluation:
        return "linear_exact"
    if prepared.constant_acceleration_m_s2 is not None:
        return "quadratic_exact"
    if prepared.case.spec.solver.integrator == "exponential_midpoint":
        return "exponential_midpoint_reintegrated"
    return "rk4_dense"


def _manifest(
    prepared: _PreparedRun,
    lifecycle: np.ndarray,
    failure_reason_code: np.ndarray,
    event_statistics: _EventStatistics,
) -> dict[str, object]:
    case = prepared.case
    spec = case.spec
    brownian = _brownian_provenance(prepared.physics)
    uses_rk4_dense = _uses_rk4_dense_certificate(
        case,
        prepared.physics,
        prepared.constant_acceleration_m_s2,
    )
    lifecycle_counts = {
        "pending": int(np.count_nonzero(lifecycle == _LIFECYCLE_PENDING)),
        "active": int(np.count_nonzero(lifecycle == _LIFECYCLE_ACTIVE)),
        "stuck": int(np.count_nonzero(lifecycle == _LIFECYCLE_STUCK)),
        "held": int(np.count_nonzero(lifecycle == _LIFECYCLE_HELD)),
        "escaped": int(np.count_nonzero(lifecycle == _LIFECYCLE_ESCAPED)),
        "failed": int(np.count_nonzero(lifecycle == _LIFECYCLE_FAILED)),
    }
    failure_counts = {
        name: int(np.count_nonzero(failure_reason_code == code))
        for code, name in _FAILURE_REASON_NAMES.items()
    }
    return {
        "case_name": spec.name,
        "case_file_hash": case.case_file_hash,
        "data_content_hash": case.content_hash,
        "case_schema_version": CASE_FORMAT_VERSION,
        "data_coordinate_system": case.data.coordinate_system,
        "motion_mode": spec.motion.mode,
        "engine_algorithm_revision": ENGINE_ALGORITHM_REVISION,
        "durable_commit_cadence": _durable_commit_cadence(prepared.schedule.particle_count),
        "compiled_cpu_tile_revision": COMPILED_CPU_TILE_REVISION,
        "step_proposal_revision": STEP_PROPOSAL_REVISION,
        "rk4_enclosure_revision": RK4_ENCLOSURE_REVISION if uses_rk4_dense else None,
        "rk4_dense_path_revision": RK4_DENSE_PATH_REVISION if uses_rk4_dense else None,
        "exponential_midpoint_revision": (
            EXPONENTIAL_MIDPOINT_REVISION
            if spec.solver.integrator == "exponential_midpoint"
            else None
        ),
        "exponential_midpoint_enclosure_revision": (
            EXPONENTIAL_MIDPOINT_ENCLOSURE_REVISION
            if spec.solver.integrator == "exponential_midpoint"
            else None
        ),
        "physics_catalog_revision": PHYSICS_CATALOG_REVISION,
        "physics_runtime_revision": PHYSICS_RUNTIME_REVISION,
        "field_location_revision": FIELD_LOCATION_REVISION,
        "required_field_revision": REQUIRED_FIELD_REVISION,
        "geometry_algorithm_revision": (GEOMETRY_ALGORITHM_REVISION),
        "event_algorithm_revision": EVENT_ALGORITHM_REVISION,
        "boundary_algorithm_revision": (
            BOUNDARY_ALGORITHM_REVISION if prepared.boundary_rules else None
        ),
        "source_algorithm_revision": SOURCE_ALGORITHM_REVISION,
        "rng_algorithm_revision": RNG_ALGORITHM_REVISION,
        "brownian_rng_revision": brownian["rng_revision"],
        "joint_ou_revision": brownian["ou_revision"],
        "joint_ou_split_revision": brownian["split_revision"],
        "brownian_composition_revision": brownian["composition_revision"],
        "brownian_charge_dense_revision": brownian["charge_dense_revision"],
        "brownian_interval_tree_depth": brownian["interval_tree_depth"],
        "random_draw_kinds": {
            "source_facet": SOURCE_FACET_DRAW,
            "source_position": SOURCE_POSITION_DRAW,
            "wall_probabilistic_stick": WALL_PROBABILISTIC_STICK_STREAM,
            "brownian_root_normal": brownian["root_normal_stream"],
            "brownian_split_normal": brownian["split_normal_stream"],
        },
        "requested": {
            "integrator": spec.solver.integrator,
            "backend": spec.solver.backend,
            "seed": spec.solver.seed,
        },
        "resolved": {
            "integrator": spec.solver.integrator,
            "backend": "cpu",
            "path_kind": _resolved_path_kind(prepared),
            "physics_models": prepared.physics.resolved_models(),
            "brownian_coefficient_policy": brownian["coefficient_policy"],
            "sources": [
                {"source_id": index, "name": source.name, "type": source.kind}
                for index, source in enumerate(spec.sources)
            ],
            "boundary_laws": [
                {
                    "group": case.data.geometry.group_names[rule.group_id],
                    "priority": rule.priority,
                    "law": rule.law_id,
                    "normal_restitution": rule.normal_restitution,
                    "tangential_restitution": rule.tangential_restitution,
                    "stick_probability": rule.stick_probability,
                    "otherwise_law": rule.otherwise_law_id,
                }
                for rule in prepared.boundary_rules
            ],
            "required_fields": [
                {
                    "name": name,
                    "layout": field.layout,
                    "unit": field.unit,
                    "components": list(field.components),
                    "stored_basis": field.stored_basis,
                }
                for name, field in prepared.fields.fields.items()
            ],
        },
        "time": {
            "start_s": spec.time.start_s,
            "end_s": spec.time.end_s,
            "dt_s": spec.time.dt_s,
        },
        "event": {
            "geometry_rtol": spec.solver.event.geometry_rtol,
            "roundoff_ulps": spec.solver.event.roundoff_ulps,
            "max_refinements": spec.solver.event.max_refinements,
            "max_interactions_per_step": spec.solver.event.max_interactions_per_step,
            "corner_policy": spec.solver.event.corner_policy,
        },
        "event_refinement": (
            {
                "accepted_particle_pieces": event_statistics.accepted_particle_pieces,
                "candidate_queries": event_statistics.candidate_queries,
                "refinements": event_statistics.refinements,
                "maximum_refinement_depth": event_statistics.maximum_refinement_depth,
            }
            if _uses_curved_event_path(prepared)
            else None
        ),
        "boundary_interactions": {
            "wall_events": event_statistics.wall_interactions,
            "residual_splits": event_statistics.residual_splits,
            "axis_crossings": event_statistics.axis_crossings,
        },
        "source_id_to_name": list(prepared.schedule.source_names),
        "lifecycle_counts": lifecycle_counts,
        "failure_reason_codes": {name: code for code, name in _FAILURE_REASON_NAMES.items()},
        "failure_reason_counts": failure_counts,
        "maximum_dt_over_tau": prepared.maximum_dt_over_tau,
        "maximum_dt_charge_lipschitz": prepared.maximum_dt_charge_lipschitz,
        "memory_plan": prepared.memory_plan.as_manifest(),
    }
