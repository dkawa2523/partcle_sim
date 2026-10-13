"""Preparation and the single production macro-step loop."""

from __future__ import annotations

import hashlib
import math
import mmap
import multiprocessing
import os
import pickle
import traceback
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from os import PathLike
from typing import NamedTuple, Protocol, cast

import numpy as np
from numba import njit

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
    allocate_boundary_response_buffer,
    prepare_boundary_rule,
    prepare_boundary_rules,
    resolve_boundary_responses_batch,
    validate_boundary_rule_frames,
)
from .case import CASE_FORMAT_VERSION, SimulationCase, TimeSpec
from .case_format import resident_array_bytes
from .coordinates import (
    canonicalize_rz_enclosure,
    canonicalize_rz_enclosure_into,
    canonicalize_rz_velocity_enclosure_into,
    fold_rz_position_vector,
    fold_rz_position_vector_into,
    rz_canonical_vector_to_signed,
    rz_radial_interval,
    rz_signed_stage_into,
    rz_vector_to_signed_into,
)
from .cpu import (
    CPU_RUNTIME_LAYOUT_REVISION,
    MEMORY_PLAN_REVISION,
    ActiveIndex,
    CpuMemoryPlan,
    ProcessArrayReader,
    ProcessArrayStore,
    early_memory_requirement_bytes,
    plan_cpu_memory,
    process_array_views,
)
from .events import (
    CURVED_STATUS_AXIS,
    CURVED_STATUS_CLEAR,
    CURVED_STATUS_FAILURE,
    CURVED_STATUS_SPLIT,
    CURVED_STATUS_WALL,
    EVENT_ALGORITHM_REVISION,
    EVENT_QUERY_OK,
    EXACT_DEPARTURE_FINITE_CONTACT_SET,
    EXACT_DEPARTURE_NONE,
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
    EventArena,
    EventGeometry,
    EventLocationError,
    ExactEventBatch,
    SurfaceReleaseBatch,
    allocate_event_arena,
    certify_event_point_interiors,
    classify_event_point,
    classify_surface_release_batch,
    curved_event_result_view,
    exact_event_result_view,
    locate_curved_first_event_into,
    locate_exact_first_event_into,
    prepare_curved_event_batch,
    prepare_event_geometry,
    prepare_exact_event_batch,
    resolve_event_budget,
)
from .fields import (
    FIELD_LOCATION_REVISION,
    FIELD_TIME_REVISION,
    LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW,
    REQUIRED_FIELD_REVISION,
    FieldBatch,
    FieldLocationError,
    FieldWorkspace,
    NumericFieldContext,
    NumericFieldRangeWorkspace,
    NumericFieldWorkspace,
    PreparedFieldSet,
    RequiredFieldMetadata,
    local_prepared_field_ranges_into,
    prepare_required_fields,
    sample_prepared_fields_into,
    validate_periodic_field_seams,
)
from .geometry import (
    GEOMETRY_ALGORITHM_REVISION,
    GeometryPreparationError,
    PreparedGeometry,
    centers_respect_contact_radius,
    contact_normals_for_candidates,
    points_inside_volume,
    prepare_geometry,
)
from .integrators import (
    EXPONENTIAL_MIDPOINT_ENCLOSURE_REVISION,
    EXPONENTIAL_MIDPOINT_REVISION,
    RELAXATION_ADVANCE_WORK_BYTES_PER_PARTICLE,
    RK4_ADVANCE_WORK_BYTES_PER_PARTICLE,
    RK4_DENSE_FULL_ENCLOSURE_BYTES_PER_PARTICLE,
    RK4_DENSE_PATH_REVISION,
    RK4_ENCLOSURE_REVISION,
    STAGE_ADVANCE_WORK_BYTES_PER_PARTICLE,
    STEP_PROPOSAL_REVISION,
    CurvedPathEnclosure,
    DynamicsEvaluation,
    ProposalPrefixWorkspace,
    ProposalSample,
    ProposalSampleBuffer,
    RelaxationAdvanceWorkspace,
    RelaxationEvaluation,
    Rk4AdvanceWorkspace,
    Rk4DenseEnclosure,
    StageAdvanceWorkspace,
    StepProposal,
    affine_exponential_charge,
    allocate_proposal_prefix_workspace,
    allocate_proposal_sample_buffer,
    allocate_relaxation_advance_workspace,
    allocate_rk4_advance_workspace,
    allocate_stage_advance_workspace,
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
    BROWNIAN_MIDPOINT_2D_REVISION,
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
    AccelerationBoundContext,
    ContinuousApplicabilityContext,
    PhysicsNumericContext,
    PhysicsNumericOutputs,
    PhysicsRuntime,
    PhysicsRuntimeEvaluation,
    PhysicsRuntimeWorkspace,
    PrimitiveRange,
    acceleration_abs_upper_into,
    brownian_thermal_variance_into,
    charge_interval_inside_prepared_invariant_into,
    evaluate_continuous_applicability_into,
    evaluate_local_continuous_applicability_into,
    evaluate_prepared_physics_into,
    local_additive_acceleration_abs_upper_into,
    prepare_physics_runtime,
)
from .rng import (
    BROWNIAN_RNG_REVISION,
    BROWNIAN_ROOT_NORMAL_STREAM,
    BROWNIAN_SPLIT_NORMAL_STREAM,
    RNG_ALGORITHM_REVISION,
    WALL_MAXWELL_DIFFUSE_STREAM,
    WALL_MAXWELL_NORMAL_STREAM,
    WALL_MAXWELL_TANGENTIAL_STREAM,
    WALL_PROBABILISTIC_STICK_STREAM,
    _brownian_node_key,
    _brownian_pair_from_key,
    wall_standard_normal_batch,
    wall_uniform_batch,
    wall_uniform_open_batch,
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
    OU_STATUS_OK,
    JointOuIncrement,
    _finite_children,
    _half_split_coefficients_scalar,
    _joint_ou_increment_component,
    _joint_ou_increment_scales,
    _split_joint_ou_component,
    advance_joint_ou_with_increment_into,
)
from .topology import (
    TOPOLOGY_ALGORITHM_REVISION,
    TOPOLOGY_CANDIDATE_INVALID,
    TOPOLOGY_CANDIDATE_MATERIAL,
    TOPOLOGY_CANDIDATE_PERIODIC,
    PreparedPeriodicTopology,
    TopologyPreparationError,
    TranslationPairRequest,
    classify_periodic_candidate_rows,
    prepare_periodic_topology,
)

type _PathInput = str | PathLike[str]

ENGINE_ALGORITHM_REVISION = "particle_engine_v64"
COMPILED_CPU_TILE_REVISION = "compiled_cpu_tile_v31"
DURABLE_COMMIT_CADENCE_REVISION = "cumulative_solver_work_v1"
BROWNIAN_TREE_POLICY_REVISION = "conditional_boundary_refinement_absolute_target_v2"

_PREPARE_SCAN_BATCH_SIZE = 65_536
_MAX_EVENT_ORDINAL = int(np.iinfo(np.uint32).max)
_TIME_GRID_ROUNDOFF_ULPS = 8.0
_DURABLE_COMMIT_MINIMUM_WORK = 1 << 20
_DURABLE_COMMIT_WORK_PER_PARTICLE = 128
# Numeric event staging plus the canonical payload simultaneously handed to
# the synchronous writer, including topology kind/destination/post-position.
# The pack-only int64 row gather is intentionally owned by the memory plan's
# 12.5% safety margin instead of being retained as another named component.
_EVENT_STAGING_BYTES_PER_ROW = 624
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
    (
        "",
        "stick",
        "escape",
        "specular",
        "restitution",
        "probabilistic_stick",
        "hold",
        "maxwell_thermal",
    ),
    dtype="<U32",
)
_BOUNDARY_OUTCOME_OUTPUT = np.asarray(
    ("", "stuck", "escaped", "reflected", "held", "transferred"),
    dtype="<U16",
)
_BOUNDARY_INTERACTION_OUTPUT = np.asarray(
    ("", "wall", "periodic_translation"),
    dtype="<U24",
)
_INTERACTION_WALL = np.uint8(1)
_INTERACTION_PERIODIC = np.uint8(2)
_OUTCOME_TRANSFERRED = np.uint8(5)


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


class _NumericSupportContext(NamedTuple):
    """Prepared field/domain relation; contact remains an engine verdict."""

    regular_box: bool
    lower_m: tuple[float, float]
    upper_m: tuple[float, float]
    covers_geometry: bool
    axisymmetric_rz: bool


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    case: SimulationCase
    schedule: ParticleSchedule
    frame_times_s: tuple[float, ...]
    probe_times_s: tuple[float, ...]
    probe_particle_index: np.ndarray
    geometry: PreparedGeometry
    event_geometry: PreparedGeometry
    topology: PreparedPeriodicTopology | None
    boundary_rules: tuple[BoundaryRule, ...]
    compiled_boundary_rules: PreparedBoundaryRules
    physics: PhysicsPlan
    fields: PreparedFieldSet
    field_time_split_s: tuple[float, ...]
    dynamics: _StageDynamics
    constant_acceleration_m_s2: np.ndarray | None
    maximum_dt_over_tau: float
    maximum_dt_charge_lipschitz: float
    memory_plan: CpuMemoryPlan
    event_candidate_capacity: int
    certificate_workspace: _ApplicabilityCertificateWorkspace
    curved_workspace: _CurvedWavefront
    exact_workspace: _ExactWavefront
    boundary_response_buffer: BoundaryResponseBatch
    event_query_geometry: EventGeometry
    event_query_arena: EventArena
    replay_workspace: _ProposalReplayWorkspace
    support_context: _NumericSupportContext = dataclass_field(init=False)

    def __post_init__(self) -> None:
        support_box = self.fields.regular_support_box()
        lower, upper = (np.zeros(2), np.zeros(2)) if support_box is None else support_box
        covers_geometry = False
        if support_box is not None:
            geometry_lower = np.min(self.geometry.nodes_m, axis=0)
            geometry_upper = np.max(self.geometry.nodes_m, axis=0)
            covers_geometry = bool(
                (geometry_lower >= lower).all() and (geometry_upper <= upper).all()
            )
        object.__setattr__(
            self,
            "support_context",
            _NumericSupportContext(
                support_box is not None,
                (float(lower[0]), float(lower[1])),
                (float(upper[0]), float(upper[1])),
                covers_geometry,
                self.case.data.coordinate_system == "axisymmetric_rz",
            ),
        )


@dataclass(slots=True)
class _BoundaryEventBuffer:
    """Capacity-bounded numeric columns for one synchronous event wave."""

    particle_index: np.ndarray
    time_s: np.ndarray
    event_ordinal: np.ndarray
    primary_facet_id: np.ndarray
    interaction_code: np.ndarray
    destination_facet_id: np.ndarray
    position_m: np.ndarray
    position_post_m: np.ndarray
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


class _FailureCommitContext(NamedTuple):
    """Numeric resident and bounded failure columns, owned by the engine."""

    position_m: np.ndarray
    velocity_m_s: np.ndarray
    charge_number: np.ndarray
    active: np.ndarray
    lifecycle: np.ndarray
    terminal_time_s: np.ndarray
    event_ordinal: np.ndarray
    failure_reason_code: np.ndarray
    failure_particle_index: np.ndarray
    failure_time_s: np.ndarray
    failure_event_ordinal: np.ndarray
    failure_reason: np.ndarray


class _ReplayBuffer(NamedTuple):
    """Requested macro-time states held in numeric columnar storage."""

    time_s: np.ndarray
    particle_index: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    charge_number: np.ndarray
    presence: np.ndarray
    lifecycle: np.ndarray


class _ProposalReplayWorkspace(NamedTuple):
    """Exclusive bounded sampling and scatter scratch, reused across outputs."""

    sample: ProposalSampleBuffer
    prefix: ProposalPrefixWorkspace
    time_s: np.ndarray
    proposal_rows: np.ndarray
    replay_slots: np.ndarray
    radial_sign: np.ndarray

    @classmethod
    def allocate(
        cls, capacity: int, frame_times_s: tuple[float, ...], probe_times_s: tuple[float, ...]
    ) -> _ProposalReplayWorkspace:
        if not frame_times_s and not probe_times_s:
            capacity = 0
        return cls(
            allocate_proposal_sample_buffer(capacity),
            allocate_proposal_prefix_workspace(capacity),
            np.empty(capacity, dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<f8"),
        )


class _ExactEndpointBatch(NamedTuple):
    """Accepted exact-prefix endpoints indexed by slab-local row."""

    valid: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    charge_number: np.ndarray


class _LangevinCoefficients(NamedTuple):
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
    target_time_s: np.ndarray
    start_position_m: np.ndarray
    start_velocity_m_s: np.ndarray
    start_charge_number: np.ndarray
    coefficients: _LangevinCoefficients
    numerical_status: np.ndarray
    drag_rate_s_inv: np.ndarray
    thermal_velocity_variance_m2_s2: np.ndarray
    equilibrium_velocity_m_s: np.ndarray


@dataclass(frozen=True, slots=True)
class _BrownianSlabRuntime:
    """One conditional OU root and the mutable solver state it may commit."""

    prepared: _PreparedRun
    batch: _LangevinRootBatch
    particle_ids: np.ndarray
    macro_interval: int
    root_interval: np.ndarray
    stochastic_event_restart_count: np.ndarray
    guard_restart_count: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    charge_number: np.ndarray
    active: np.ndarray
    lifecycle: np.ndarray
    terminal_time_s: np.ndarray
    event_ordinal: np.ndarray
    physical_boundary_event_ordinal: np.ndarray
    exact_origin_time_s: np.ndarray
    exact_origin_position_m: np.ndarray
    exact_origin_velocity_m_s: np.ndarray
    start_contact_state: np.ndarray
    replay: _ReplayBuffer
    statistics: _EventStatistics
    failure_reason_code: np.ndarray
    writer: _EventSink
    event_buffer: _BoundaryEventBuffer
    failure_buffer: _FailureEventBuffer
    root_continuing: np.ndarray
    pending_particle_index: np.ndarray
    pending_start_time_s: np.ndarray
    pending_start_position_m: np.ndarray
    pending_start_velocity_m_s: np.ndarray
    pending_root_interval: np.ndarray
    pending_stochastic_event_restart_count: np.ndarray
    pending_guard_restart_count: np.ndarray
    pending_count: np.ndarray
    adaptive_workspace: _BrownianAdaptiveWorkspace
    leaf_cursor: _BrownianLeafCursor


class _CurvedWavefront(NamedTuple):
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
    stochastic_restart_time_s: np.ndarray
    root_rows: np.ndarray
    zero_rows: np.ndarray
    wave_rows: np.ndarray
    wave_targets_s: np.ndarray
    exhausted_rows: np.ndarray

    exhausted_reason_code: np.ndarray

    @classmethod
    def allocate(
        cls,
        capacity: int,
        stack_capacity: int,
        stochastic: bool,
        *,
        exact_enabled: bool,
        event_paths: bool,
    ) -> _CurvedWavefront:
        if exact_enabled or not event_paths:
            capacity = 0
        return cls(
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<f8"),
            np.empty((capacity, 2), dtype="<f8"),
            np.empty((capacity, 2), dtype="<f8"),
            np.empty(capacity, dtype="<f8"),
            np.empty(capacity, dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype=np.bool_),
            np.empty((capacity, stack_capacity), dtype="<f8"),
            np.empty((capacity, stack_capacity), dtype="<i8"),
            np.empty((capacity, stack_capacity), dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity if stochastic else 0, dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<u2"),
        )

    def view(self, count: int) -> _CurvedWavefront:
        """Borrow one exclusive slab; no view survives the consuming call."""

        return _CurvedWavefront(
            self.particle_index[:count],
            self.current_time_s[:count],
            self.current_position_m[:count],
            self.current_velocity_m_s[:count],
            self.current_charge_number[:count],
            self.root_interval_s[:count],
            self.interaction_count[:count],
            self.certify_departure[:count],
            self.stack_target_s[:count],
            self.stack_depth[:count],
            self.stack_interaction_reset[:count],
            self.stack_top[:count],
            self.stochastic_restart_time_s[:count],
            self.root_rows[:count],
            self.zero_rows[:count],
            self.wave_rows[:count],
            self.wave_targets_s[:count],
            self.exhausted_rows[:count],
            self.exhausted_reason_code[:count],
        )


class _ExactWavefront(NamedTuple):
    """Exclusive exact residual columns, reused across bounded slabs."""

    current_time_s: np.ndarray
    current_position_m: np.ndarray
    current_velocity_m_s: np.ndarray
    current_charge_number: np.ndarray
    acceleration_m_s2: np.ndarray
    departing_facet_id: np.ndarray
    stack_target_s: np.ndarray
    stack_depth: np.ndarray
    stack_interactions: np.ndarray
    stack_top: np.ndarray
    wave_rows: np.ndarray
    wave_targets_s: np.ndarray
    zero_rows: np.ndarray
    exhausted_rows: np.ndarray
    failed_rows: np.ndarray
    split_rows: np.ndarray
    clear_rows: np.ndarray
    clear_targets_s: np.ndarray
    axis_event_rows: np.ndarray
    wall_event_rows: np.ndarray

    @classmethod
    def allocate(
        cls, capacity: int, stack_capacity: int, *, exact_enabled: bool, event_paths: bool
    ) -> _ExactWavefront:
        if not exact_enabled or not event_paths:
            capacity = 0
        return cls(
            np.empty(capacity, dtype="<f8"),
            np.empty((capacity, 2), dtype="<f8"),
            np.empty((capacity, 2), dtype="<f8"),
            np.empty(capacity, dtype="<f8"),
            np.empty((capacity, 2), dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
            np.empty((capacity, stack_capacity), dtype="<f8"),
            np.empty((capacity, stack_capacity), dtype="<i8"),
            np.empty((capacity, stack_capacity), dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
            np.empty(capacity, dtype="<i8"),
        )

    def view(self, count: int) -> _ExactWavefront:
        return _ExactWavefront(*(column[:count] for column in self))


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


class _EventSink(Protocol):
    def write_boundary_events(self, events: BoundaryEvents) -> None: ...

    def write_failure_events(self, events: FailureEvents) -> None: ...


class _ProcessEventSink:
    """Synchronous bounded handoff; workers never open the result store."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def write_boundary_events(self, events: BoundaryEvents) -> None:
        self._handoff("boundary", events)

    def write_failure_events(self, events: FailureEvents) -> None:
        self._handoff("failure", events)

    def _handoff(self, kind: str, events: BoundaryEvents | FailureEvents) -> None:
        self.connection.send((kind, events))
        if self.connection.recv() != ("accepted",):
            raise EngineError("particle event handoff was not acknowledged")


class _ProcessState(NamedTuple):
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    charge_number: np.ndarray
    active: np.ndarray
    lifecycle: np.ndarray
    terminal_time_s: np.ndarray
    event_ordinal: np.ndarray
    physical_boundary_event_ordinal: np.ndarray
    exact_origin_time_s: np.ndarray
    exact_origin_position_m: np.ndarray
    exact_origin_velocity_m_s: np.ndarray
    start_contact_state: np.ndarray
    failure_reason_code: np.ndarray


@dataclass(slots=True)
class _ProcessWorker:
    process: BaseProcess
    connection: Connection


def _process_private_arrays(prepared: _PreparedRun) -> Iterator[np.ndarray]:
    dynamics = prepared.dynamics
    owners = (
        dynamics.field_workspace,
        dynamics.physics_workspace,
        dynamics.native_workspace,
        dynamics.native_rk4_workspace,
        dynamics.native_stage_advance_workspace,
        dynamics.native_relaxation_workspace,
        dynamics.field_range_workspace,
        dynamics.global_certificate_workspace,
        dynamics.local_certificate_workspace,
        dynamics.langevin_coefficient_workspace,
        prepared.certificate_workspace,
        prepared.curved_workspace,
        prepared.exact_workspace,
        prepared.boundary_response_buffer,
        prepared.event_query_arena,
        prepared.replay_workspace,
    )
    for owner in owners:
        yield from process_array_views(owner)


def _share_process_state(state: _RunState, store: ProcessArrayStore) -> None:
    """Move the explicit resident columns into parent-owned storage."""
    for name in _ProcessState._fields:
        setattr(state, name, store.share(getattr(state, name)))


def _process_state(state: _RunState) -> _ProcessState:
    return _ProcessState(*(getattr(state, name) for name in _ProcessState._fields))


def _process_particle_worker(connection: Connection, payload: bytes) -> None:
    """Use the existing slab engine with exclusive scratch in one interpreter."""
    reader = ProcessArrayReader()
    replay_reader = ProcessArrayReader()
    try:
        prepared, state = cast(tuple[_PreparedRun, _ProcessState], reader.loads(payload))
        sink = _ProcessEventSink(connection)
        connection.send(("ready", None))
        while True:
            command = connection.recv()
            if command[0] == "stop":
                return
            _, rows, interval, start, end, replay_payload = command
            result = _process_worker_slab(
                prepared, state, rows, interval, start, end, replay_payload, replay_reader, sink
            )
            connection.send(("done", result.statistics))
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        replay_reader.close()
        reader.close()
        connection.close()


def _process_worker_slab(
    prepared: _PreparedRun,
    state: _ProcessState,
    rows: np.ndarray,
    interval: int,
    start: float,
    end: float,
    replay_payload: bytes,
    reader: ProcessArrayReader,
    sink: _EventSink,
) -> _SlabResult:
    replay = cast(_ReplayBuffer, reader.loads(replay_payload))
    try:
        return _advance_particle_slab(
            prepared,
            rows,
            interval,
            start,
            end,
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
            state.failure_reason_code,
            sink,
        )
    finally:
        reader.release_views()


def _receive_process_message(worker: _ProcessWorker) -> tuple[str, object]:
    ready = wait([worker.connection, worker.process.sentinel])
    if worker.connection not in ready:
        raise EngineError(f"particle worker exited with code {worker.process.exitcode}")
    try:
        kind, payload = worker.connection.recv()
    except (EOFError, OSError) as error:
        raise EngineError("particle worker connection closed before completion") from error
    if kind == "error":
        raise EngineError(f"particle worker failed: {payload}")
    return kind, payload


class _ProcessTeam:
    """One run owns fixed processes, shared state, and ordered macro barriers."""

    def __init__(self, prepared: _PreparedRun, state: _RunState) -> None:
        self.prepared = prepared
        self.state = state
        self.workers: list[_ProcessWorker] = []
        self.store: ProcessArrayStore | None = None
        self.replay_store: ProcessArrayStore | None = None
        self.replay_storage: np.ndarray | None = None
        self.finished = False

    def __enter__(self) -> _ProcessTeam:
        try:
            if self.prepared.memory_plan.worker_count > 1:
                self._start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _start(self) -> None:
        prepared = self.prepared
        self.store = ProcessArrayStore()
        _share_process_state(self.state, self.store)
        hint = prepared.dynamics.last_field_cell
        if hint is not None:
            hint = self.store.share(hint)
            prepared.dynamics.last_field_cell = hint
            prepared.dynamics.native_context = prepared.dynamics.native_context._replace(
                accepted_field_cell=hint,
            )
        payload = self.store.dumps(
            (prepared, _process_state(self.state)),
            private_arrays=_process_private_arrays(prepared),
        )
        plan = prepared.memory_plan.reserve_process_padding(
            self.store.page_padding_bytes + 7 * mmap.PAGESIZE
        )
        prepared = replace(prepared, memory_plan=plan)
        self.prepared = prepared
        payload = self.store.dumps((prepared, _process_state(self.state)))
        components = prepared.memory_plan.components
        reserve = sum(value for name, value in components.items() if name.startswith("process_"))
        required = self.store.shared_bytes + (
            prepared.memory_plan.worker_count * self.store.private_bytes
        )
        if required > reserve:
            raise EngineError("process storage exceeds its predicted array reserve")
        context = multiprocessing.get_context("spawn")
        for _ in range(prepared.memory_plan.worker_count):
            parent, child = context.Pipe()
            process = context.Process(target=_process_particle_worker, args=(child, payload))
            # Windows PipeConnection implements the same documented Connection interface.
            worker = _ProcessWorker(process, cast(Connection, parent))
            self.workers.append(worker)
            try:
                process.start()
            except RuntimeError as error:
                raise EngineError(
                    "process workers require an importable main module and a guarded script entry"
                ) from error
            finally:
                child.close()
        pending = {worker.connection: worker for worker in self.workers}
        while pending:
            ready = self._ready_connections(pending)
            for connection in tuple(pending):
                if connection in ready:
                    kind, _ = _receive_process_message(pending.pop(connection))
                    if kind != "ready":
                        raise EngineError("particle worker did not complete preparation")
        self.store.release_private_templates()

    def _ready_connections(self, pending: dict[Connection, _ProcessWorker]) -> Sequence[object]:
        ready = wait([*pending, *(worker.process.sentinel for worker in self.workers)])
        for worker in self.workers:
            if worker.process.sentinel in ready and worker.connection not in ready:
                raise EngineError(f"particle worker exited with code {worker.process.exitcode}")
        return ready

    def allocate_replay(
        self, time_s: tuple[float, ...], *, include_all_particles: bool
    ) -> _ReplayBuffer:
        """Initialize macro views in one run-owned shared replay allocation."""
        if self.workers and time_s and self.replay_storage is None:
            self.replay_store = ProcessArrayStore()
            self.replay_storage = self.replay_store.share(
                np.empty(self.prepared.memory_plan.replay_work_bytes, dtype=np.uint8)
            )
        return _allocate_replay_buffer(
            time_s,
            self.prepared.schedule,
            self.prepared.probe_particle_index,
            include_all_particles=include_all_particles,
            storage=self.replay_storage,
        )

    def advance(
        self,
        interval: int,
        start: float,
        end: float,
        replay: _ReplayBuffer,
        writer: ResultWriter,
    ) -> None:
        prepared, state = self.prepared, self.state
        if not self.workers:
            _advance_active_slabs(
                prepared,
                state.active_index.rows,
                interval,
                start,
                end,
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
            return
        if not replay.time_s.size:
            # The replay owner allocates no rows or payload without requested times.
            payload = pickle.dumps(replay, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            if self.replay_store is None:
                raise EngineError("process replay storage was not prepared")
            payload = self.replay_store.describe(replay)
        self._advance_processes(interval, start, end, payload, writer)

    def release_replay(self) -> None:
        if self.replay_store is not None:
            self.replay_store.release_views()

    def _advance_processes(
        self,
        interval: int,
        start: float,
        end: float,
        replay_payload: bytes,
        writer: ResultWriter,
    ) -> None:
        rows = self.state.active_index.rows
        width = self.prepared.memory_plan.active_slab_particles(rows.size)
        pending: dict[Connection, _ProcessWorker] = {}
        next_row = 0
        for worker in self.workers:
            if next_row < rows.size:
                self._send_slab(worker, rows, next_row, width, interval, start, end, replay_payload)
                pending[worker.connection] = worker
                next_row += width
        while pending:
            ready = self._ready_connections(pending)
            for worker in tuple(pending.values()):
                if worker.connection not in ready:
                    continue
                if not self._receive_handoff(worker, writer):
                    continue
                if next_row < rows.size:
                    self._send_slab(
                        worker, rows, next_row, width, interval, start, end, replay_payload
                    )
                    next_row += width
                else:
                    del pending[worker.connection]

    @staticmethod
    def _send_slab(
        worker: _ProcessWorker,
        rows: np.ndarray,
        offset: int,
        width: int,
        interval: int,
        start: float,
        end: float,
        replay_payload: bytes,
    ) -> None:
        worker.connection.send(
            ("slab", rows[offset : offset + width], interval, start, end, replay_payload)
        )

    def _receive_handoff(self, worker: _ProcessWorker, writer: ResultWriter) -> bool:
        kind, payload = _receive_process_message(worker)
        if kind == "done":
            self.state.event_statistics.merge(cast(_EventStatistics, payload))
            return True
        if kind == "boundary":
            writer.write_boundary_events(cast(BoundaryEvents, payload))
        elif kind == "failure":
            writer.write_failure_events(cast(FailureEvents, payload))
        else:
            raise EngineError("unexpected particle worker handoff")
        worker.connection.send(("accepted",))
        return False

    def finish(self) -> None:
        if self.finished:
            return
        for worker in self.workers:
            worker.connection.send(("stop",))
        for worker in self.workers:
            worker.process.join(timeout=30.0)
            if worker.process.is_alive() or worker.process.exitcode != 0:
                raise EngineError("particle worker did not shut down successfully")
        self.finished = True

    def __exit__(self, *_: object) -> None:
        for worker in self.workers:
            if worker.process.is_alive():
                worker.process.terminate()
            if worker.process.pid is not None:
                worker.process.join(timeout=5.0)
            worker.connection.close()
            worker.process.close()
        self.replay_storage = None
        if self.replay_store is not None:
            self.replay_store.close()
            self.replay_store = None
        if self.store is not None:
            self.store.close()


class _NumericStageContext(NamedTuple):
    """Prepared stage facts and accepted resident hints, sampled without writes."""

    fields: NumericFieldContext
    physics: PhysicsNumericContext
    axisymmetric_rz: bool
    accepted_field_cell: np.ndarray


class _NumericStageWorkspace(NamedTuple):
    """One exclusive stage slot; no array is selected by operating-system thread ID."""

    fields: NumericFieldWorkspace
    physics: PhysicsNumericOutputs
    physics_incoming_status: np.ndarray
    canonical_position_m: np.ndarray
    canonical_velocity_m_s: np.ndarray
    radial_sign: np.ndarray
    cell_hint: np.ndarray


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _evaluate_prepared_stage_into(
    context: _NumericStageContext,
    workspace: _NumericStageWorkspace,
    particle_index: np.ndarray,
    time_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
) -> None:
    """Compose coordinate, field and physics owners without a Python stage return."""

    count = particle_index.size
    stage_position = position_m
    stage_velocity = velocity_m_s
    if context.axisymmetric_rz:
        stage_position = workspace.canonical_position_m[:count]
        stage_velocity = workspace.canonical_velocity_m_s[:count]
        rz_signed_stage_into(
            position_m, velocity_m_s, stage_position, stage_velocity, workspace.radial_sign[:count]
        )
    for row in range(count):
        workspace.cell_hint[row] = (
            context.accepted_field_cell[particle_index[row]]
            if context.accepted_field_cell.size
            else -1
        )
    sample_prepared_fields_into(
        context.fields, stage_position, time_s, workspace.cell_hint[:count], workspace.fields
    )
    evaluate_prepared_physics_into(
        context.physics,
        particle_index,
        stage_velocity,
        charge_number,
        workspace.fields.packed_values[:count],
        workspace.physics,
        workspace.fields.numerical_status[:count],
        workspace.physics_incoming_status[:count],
    )


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _native_stage_evaluator(
    context: _NumericStageContext,
    workspace: _NumericStageWorkspace,
    particle_index: np.ndarray,
    time_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    incoming_status: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrator-facing fixed stage tuple; accepted hints remain read-only."""

    _evaluate_prepared_stage_into(
        context, workspace, particle_index, time_s, position_m, velocity_m_s, charge_number
    )
    count = particle_index.size
    if context.axisymmetric_rz:
        rz_vector_to_signed_into(
            workspace.physics.acceleration_m_s2[:count],
            workspace.radial_sign[:count],
            workspace.physics.acceleration_m_s2[:count],
        )
    status = workspace.fields.numerical_status[:count]
    for row in range(count):
        if incoming_status[row] != NUMERICAL_STATUS_OK:
            status[row] = incoming_status[row]
    return (
        workspace.physics.acceleration_m_s2[:count],
        workspace.physics.charge_rate_number_s[:count],
        workspace.fields.support_inside[:count],
        workspace.physics.applicable[:count],
        status,
        workspace.fields.cell_id[:count],
    )


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _native_relaxation_stage_evaluator(
    context: _NumericStageContext,
    workspace: _NumericStageWorkspace,
    particle_index: np.ndarray,
    time_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    incoming_status: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Expose the physics owner's affine relaxation in the signed stage basis."""

    _evaluate_prepared_stage_into(
        context, workspace, particle_index, time_s, position_m, velocity_m_s, charge_number
    )
    count = particle_index.size
    target = workspace.physics.target_velocity_m_s[:count]
    additive = workspace.physics.additive_acceleration_m_s2[:count]
    if context.axisymmetric_rz:
        rz_vector_to_signed_into(target, workspace.radial_sign[:count], target)
        rz_vector_to_signed_into(additive, workspace.radial_sign[:count], additive)
    status = workspace.fields.numerical_status[:count]
    for row in range(count):
        if incoming_status[row] != NUMERICAL_STATUS_OK:
            status[row] = incoming_status[row]
    return (
        workspace.physics.linear_drag_rate_s_inv[:count],
        target,
        additive,
        workspace.physics.charge_rate_number_s[:count],
        workspace.physics.charge_rate_derivative_s_inv[:count],
        workspace.fields.support_inside[:count],
        workspace.physics.applicable[:count],
        status,
        workspace.fields.cell_id[:count],
    )


class _GlobalCertificateWorkspace(NamedTuple):
    velocity_abs_upper_m_s: np.ndarray
    acceleration_abs_upper_m_s2: np.ndarray
    applicable: np.ndarray
    status: np.ndarray


class _LangevinCoefficientWorkspace(NamedTuple):
    thermal_variance_m2_s2: np.ndarray
    numerical_status: np.ndarray


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _evaluate_langevin_coefficients_into(
    context: _NumericStageContext,
    stage_workspace: _NumericStageWorkspace,
    particle_index: np.ndarray,
    time_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    workspace: _LangevinCoefficientWorkspace,
) -> None:
    """Compose sampled relaxation and the FDT owner before any Brownian draw."""

    count = particle_index.size
    thermal_status = workspace.numerical_status[:count]
    thermal_status.fill(NUMERICAL_STATUS_OK)
    rate, target, additive, charge_rate, derivative, _, _, status, _ = (
        _native_relaxation_stage_evaluator(
            context,
            stage_workspace,
            particle_index,
            time_s,
            position_m,
            velocity_m_s,
            charge_number,
            thermal_status,
        )
    )
    brownian_thermal_variance_into(
        context.physics.mass_kg,
        particle_index,
        stage_workspace.fields.packed_values[:count, context.physics.bindings.gas_temperature],
        workspace.thermal_variance_m2_s2[:count],
        thermal_status,
    )
    for row in range(count):
        invalid = not (
            np.isfinite(rate[row])
            and rate[row] > 0.0
            and np.isfinite(charge_rate[row])
            and np.isfinite(derivative[row])
            and derivative[row] <= 0.0
        )
        for component in range(2):
            invalid = invalid or not (
                np.isfinite(target[row, component]) and np.isfinite(additive[row, component])
            )
        if status[row] == NUMERICAL_STATUS_OK and (
            thermal_status[row] != NUMERICAL_STATUS_OK or invalid
        ):
            status[row] = PHYSICS_NUMERICAL_FAILURE


class _LocalCertificateWorkspace(NamedTuple):
    position_lower_m: np.ndarray
    position_upper_m: np.ndarray
    velocity_lower_m_s: np.ndarray
    velocity_upper_m_s: np.ndarray
    acceleration_abs_upper_m_s2: np.ndarray
    applicable: np.ndarray
    status: np.ndarray


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _local_primitive_ranges_into(
    context: _NumericStageContext,
    time_lower_s: np.ndarray,
    time_upper_s: np.ndarray,
    position_lower_m: np.ndarray,
    position_upper_m: np.ndarray,
    velocity_lower_m_s: np.ndarray,
    velocity_upper_m_s: np.ndarray,
    field_range: NumericFieldRangeWorkspace,
    workspace: _LocalCertificateWorkspace,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose signed coordinate boxes and packed field ranges in reusable slots."""

    count = position_lower_m.shape[0]
    position_lower, position_upper = position_lower_m, position_upper_m
    velocity_lower, velocity_upper = velocity_lower_m_s, velocity_upper_m_s
    if context.axisymmetric_rz:
        position_lower, position_upper = (
            workspace.position_lower_m[:count],
            workspace.position_upper_m[:count],
        )
        velocity_lower, velocity_upper = (
            workspace.velocity_lower_m_s[:count],
            workspace.velocity_upper_m_s[:count],
        )
        canonicalize_rz_velocity_enclosure_into(
            position_lower_m,
            position_upper_m,
            velocity_lower_m_s,
            velocity_upper_m_s,
            velocity_lower,
            velocity_upper,
        )
        canonicalize_rz_enclosure_into(
            position_lower_m, position_upper_m, position_lower, position_upper
        )
    local_prepared_field_ranges_into(
        context.fields, position_lower, position_upper, time_lower_s, time_upper_s, field_range
    )
    return velocity_lower, velocity_upper


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _local_certificate_into(
    context: _NumericStageContext,
    acceleration_context: AccelerationBoundContext,
    particle_index: np.ndarray,
    time_lower_s: np.ndarray,
    time_upper_s: np.ndarray,
    position_lower_m: np.ndarray,
    position_upper_m: np.ndarray,
    velocity_lower_m_s: np.ndarray,
    velocity_upper_m_s: np.ndarray,
    charge_lower_number: np.ndarray,
    charge_upper_number: np.ndarray,
    field_range: NumericFieldRangeWorkspace,
    workspace: _LocalCertificateWorkspace,
    bound_additive: bool,
) -> None:
    """Evaluate the sole field/physics owners without dictionaries or row gathers."""

    velocity_lower, velocity_upper = _local_primitive_ranges_into(
        context,
        time_lower_s,
        time_upper_s,
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        field_range,
        workspace,
    )
    count = particle_index.size
    available = field_range.range_available[:count]
    applicable, status = workspace.applicable[:count], workspace.status[:count]
    if bound_additive:
        local_additive_acceleration_abs_upper_into(
            context.physics,
            acceleration_context,
            particle_index,
            velocity_lower,
            velocity_upper,
            charge_lower_number,
            charge_upper_number,
            field_range.packed_lower[:count],
            field_range.packed_upper[:count],
            available,
            workspace.acceleration_abs_upper_m_s2[:count],
            applicable,
            status,
        )
    else:
        if context.fields.field_count == 0:
            available.fill(True)
        evaluate_local_continuous_applicability_into(
            context.physics,
            particle_index,
            velocity_lower,
            velocity_upper,
            charge_lower_number,
            charge_upper_number,
            field_range.packed_lower[:count],
            field_range.packed_upper[:count],
            available,
            applicable,
            status,
        )


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _global_applicability_into(
    context: ContinuousApplicabilityContext,
    particle_index: np.ndarray,
    velocity_lower_m_s: np.ndarray,
    velocity_upper_m_s: np.ndarray,
    charge_lower_number: np.ndarray,
    charge_upper_number: np.ndarray,
    workspace: _GlobalCertificateWorkspace,
) -> None:
    """Compose all global model gates and the charge invariant in one call."""

    count = particle_index.size
    velocity_abs = workspace.velocity_abs_upper_m_s[:count]
    for row in range(count):
        for component in range(2):
            velocity_abs[row, component] = np.maximum(
                abs(velocity_lower_m_s[row, component]), abs(velocity_upper_m_s[row, component])
            )
    applicable, status = workspace.applicable[:count], workspace.status[:count]
    evaluate_continuous_applicability_into(
        context, particle_index, velocity_abs, applicable, status
    )
    charge_interval_inside_prepared_invariant_into(
        context, charge_lower_number, charge_upper_number, applicable, status
    )


@dataclass(slots=True)
class _StageDynamics:
    """Compose one resolved physics plan over shared stage field samples."""

    runtime: PhysicsRuntime
    fields: PreparedFieldSet
    coordinate_system: str
    last_field_cell: np.ndarray | None
    field_workspace: FieldWorkspace
    physics_workspace: PhysicsRuntimeWorkspace
    native_context: _NumericStageContext = dataclass_field(init=False)
    native_workspace: _NumericStageWorkspace = dataclass_field(init=False)
    native_rk4_workspace: Rk4AdvanceWorkspace | None = None
    native_stage_advance_workspace: StageAdvanceWorkspace | None = None
    native_relaxation_workspace: RelaxationAdvanceWorkspace | None = None
    field_range_workspace: NumericFieldRangeWorkspace | None = None
    global_certificate_workspace: _GlobalCertificateWorkspace = dataclass_field(init=False)
    local_certificate_workspace: _LocalCertificateWorkspace = dataclass_field(init=False)
    langevin_coefficient_workspace: _LangevinCoefficientWorkspace = dataclass_field(init=False)

    def __post_init__(self) -> None:
        capacity = self.field_workspace.capacity
        self.global_certificate_workspace = _GlobalCertificateWorkspace(
            np.empty((capacity, 2), dtype="<f8"),
            np.empty((capacity, 2), dtype="<f8"),
            np.empty(capacity, dtype=np.bool_),
            np.empty(capacity, dtype=np.uint8),
        )
        local_capacity = (
            self.field_range_workspace.range_available.size
            if self.field_range_workspace is not None
            else 0
        )
        box_capacity = local_capacity if self.coordinate_system == "axisymmetric_rz" else 0
        self.local_certificate_workspace = _LocalCertificateWorkspace(
            np.empty((box_capacity, 2), dtype="<f8"),
            np.empty((box_capacity, 2), dtype="<f8"),
            np.empty((box_capacity, 2), dtype="<f8"),
            np.empty((box_capacity, 2), dtype="<f8"),
            np.empty((local_capacity, 2), dtype="<f8"),
            np.empty(local_capacity, dtype=np.bool_),
            np.empty(local_capacity, dtype=np.uint8),
        )
        langevin_capacity = capacity if self.runtime.plan.noise is not None else 0
        self.langevin_coefficient_workspace = _LangevinCoefficientWorkspace(
            np.empty(langevin_capacity, dtype="<f8"),
            np.empty(langevin_capacity, dtype=np.uint8),
        )
        self.native_context = _NumericStageContext(
            self.fields.numeric_context,
            self.runtime.native_context,
            self.coordinate_system == "axisymmetric_rz",
            (np.empty(0, dtype="<i8") if self.last_field_cell is None else self.last_field_cell),
        )
        rz_capacity = capacity if self.native_context.axisymmetric_rz else 0
        self.native_workspace = _NumericStageWorkspace(
            self.field_workspace.numeric,
            self.physics_workspace.numeric_outputs(capacity),
            self.physics_workspace.incoming_status,
            np.empty((rz_capacity, 2), dtype="<f8"),
            np.empty((rz_capacity, 2), dtype="<f8"),
            np.empty(rz_capacity, dtype="<f8"),
            np.empty(capacity, dtype="<i8"),
        )

    @property
    def native_stage_evaluator(self) -> Callable:
        return _native_stage_evaluator

    @property
    def native_relaxation_stage_evaluator(self) -> Callable:
        return _native_relaxation_stage_evaluator

    def prepare_advance_workspaces(self, *, dense_enabled: bool, exact_enabled: bool) -> None:
        """Allocate the numerical advance scratch from the already planned stage capacity."""

        capacity = self.field_workspace.capacity
        if dense_enabled:
            self.native_rk4_workspace = allocate_rk4_advance_workspace(capacity)
        elif exact_enabled:
            self.native_stage_advance_workspace = allocate_stage_advance_workspace(capacity)
        else:
            self.native_relaxation_workspace = allocate_relaxation_advance_workspace(capacity)

    def invalidate_field_cell(self, particle_index: np.ndarray) -> None:
        """Discard location hints after a discontinuous topology transfer."""

        if self.last_field_cell is not None:
            self.last_field_cell[particle_index] = -1

    def __call__(
        self,
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        """Return stage views borrowed until the next dynamics evaluation."""

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
        """Return stage views borrowed until the next dynamics evaluation."""

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

        count = particle_index.size
        workspace = self.langevin_coefficient_workspace
        if count > workspace.numerical_status.size:
            raise EngineError("Langevin coefficients exceed their planned slab capacity")
        _evaluate_langevin_coefficients_into(
            self.native_context,
            self.native_workspace,
            particle_index,
            time_s,
            position_m,
            velocity_m_s,
            charge_number,
            workspace,
        )
        stage = self.native_workspace
        return _LangevinCoefficients(
            drag_rate_s_inv=stage.physics.linear_drag_rate_s_inv[:count].copy(),
            equilibrium_velocity_m_s=stage.physics.target_velocity_m_s[:count].copy(),
            additive_acceleration_m_s2=stage.physics.additive_acceleration_m_s2[:count].copy(),
            charge_affine_rate_number_s=stage.physics.charge_rate_number_s[:count].copy(),
            charge_rate_derivative_s_inv=stage.physics.charge_rate_derivative_s_inv[:count].copy(),
            thermal_velocity_variance_m2_s2=workspace.thermal_variance_m2_s2[:count].copy(),
            support_inside=stage.fields.support_inside[:count].copy(),
            applicability_inside=stage.physics.applicable[:count].copy(),
            numerical_status=stage.fields.numerical_status[:count].copy(),
            field_cell_id=stage.fields.cell_id[:count].copy(),
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
        _evaluate_prepared_stage_into(
            self.native_context,
            self.native_workspace,
            particle_index,
            time_s,
            position_m,
            velocity_m_s,
            charge_number,
        )
        sampled = self.field_workspace.batch(count)
        return (
            self.physics_workspace.evaluation(count),
            sampled,
            (
                self.native_workspace.radial_sign[:count]
                if self.native_context.axisymmetric_rz
                else None
            ),
            self.field_workspace.numerical_status[:count],
        )

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

        count = particle_index.size
        workspace = self.global_certificate_workspace
        if count > workspace.status.size:
            raise EngineError("acceleration certificate exceeds its planned slab capacity")
        workspace.status[:count].fill(NUMERICAL_STATUS_OK)
        acceleration_abs_upper_into(
            self.runtime.native_acceleration_bound_context,
            particle_index,
            velocity_abs_upper_m_s,
            workspace.acceleration_abs_upper_m_s2[:count],
            workspace.status[:count],
            True,
        )
        return (
            workspace.acceleration_abs_upper_m_s2[:count].copy(),
            workspace.status[:count].copy(),
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

        count = particle_index.size
        workspace = self.global_certificate_workspace
        if count > workspace.applicable.size:
            raise EngineError("global applicability certificate exceeds its planned slab capacity")
        _global_applicability_into(
            self.runtime.native_applicability_context,
            particle_index,
            velocity_lower_m_s,
            velocity_upper_m_s,
            charge_lower_number,
            charge_upper_number,
            workspace,
        )
        # The host controller may retain a verdict while a later certificate
        # reuses its slot. Native callers use the workspace directly.
        return workspace.applicable[:count].copy(), workspace.status[:count].copy()

    def prepared_charge_invariant_interval(
        self,
        reference_charge_number: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expose the runtime-owned invariant used by continuous path proofs."""

        return self.runtime.prepared_charge_invariant_interval(reference_charge_number)

    def local_continuous_applicability_batch(
        self,
        particle_index: np.ndarray,
        time_lower_s: np.ndarray,
        time_upper_s: np.ndarray,
        position_lower_m: np.ndarray,
        position_upper_m: np.ndarray,
        velocity_lower_m_s: np.ndarray,
        velocity_upper_m_s: np.ndarray,
        charge_lower_number: np.ndarray,
        charge_upper_number: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Certify one signed local path box using field-cell primitive ranges."""

        count = particle_index.size
        field_range = self.field_range_workspace
        workspace = self.local_certificate_workspace
        if field_range is None or count > workspace.applicable.size:
            raise EngineError("local applicability exceeds its planned slab capacity")
        _local_certificate_into(
            self.native_context,
            self.runtime.native_acceleration_bound_context,
            particle_index,
            time_lower_s,
            time_upper_s,
            position_lower_m,
            position_upper_m,
            velocity_lower_m_s,
            velocity_upper_m_s,
            charge_lower_number,
            charge_upper_number,
            field_range,
            workspace,
            False,
        )
        return (
            workspace.applicable[:count].copy(),
            workspace.status[:count].copy(),
            field_range.range_available[:count].copy(),
        )

    def local_additive_acceleration_abs_upper_batch(
        self,
        particle_index: np.ndarray,
        time_lower_s: np.ndarray,
        time_upper_s: np.ndarray,
        position_lower_m: np.ndarray,
        position_upper_m: np.ndarray,
        velocity_lower_m_s: np.ndarray,
        velocity_upper_m_s: np.ndarray,
        charge_lower_number: np.ndarray,
        charge_upper_number: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bound midpoint force over one local predictor enclosure."""

        count = particle_index.size
        field_range = self.field_range_workspace
        workspace = self.local_certificate_workspace
        if field_range is None or count > workspace.applicable.size:
            raise EngineError("local acceleration certificate exceeds its planned slab capacity")
        _local_certificate_into(
            self.native_context,
            self.runtime.native_acceleration_bound_context,
            particle_index,
            time_lower_s,
            time_upper_s,
            position_lower_m,
            position_upper_m,
            velocity_lower_m_s,
            velocity_upper_m_s,
            charge_lower_number,
            charge_upper_number,
            field_range,
            workspace,
            True,
        )
        return (
            workspace.acceleration_abs_upper_m_s2[:count].copy(),
            workspace.applicable[:count].copy(),
            workspace.status[:count].copy(),
            field_range.range_available[:count].copy(),
        )

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


def run_simulation(case: SimulationCase, output: _PathInput, *, workers: int = 1) -> RunSummary:
    """Run or resume the selected production profile and publish one complete result."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise EngineError("workers must be a positive integer")
    if workers > (os.cpu_count() or 1):
        raise EngineError("workers exceeds the available logical CPU count")
    prepared = _prepare(case, workers=workers)
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
            with _ProcessTeam(prepared, state) as team:
                prepared = team.prepared
                while state.macro_start_s < case.spec.time.end_s:
                    if not epoch_open:
                        writer.begin_epoch()
                        epoch_open = True
                    macro_start = state.macro_start_s
                    macro_end = _macro_step_end(
                        case.spec.time,
                        state.macro_step_count,
                        macro_start_s=macro_start,
                        split_times_s=prepared.field_time_split_s,
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
                    replay_times_s = tuple(
                        sorted(set(frame_batch_times_s) | set(probe_batch_times_s))
                    )
                    replay = team.allocate_replay(
                        replay_times_s,
                        include_all_particles=bool(frame_batch_times_s),
                    )
                    team.advance(state.macro_step_count, macro_start, macro_end, replay, writer)
                    state.active_index.compact(state.active)
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

                    del replay
                    team.release_replay()
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

                team.finish()
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


def _macro_step_end(
    time: TimeSpec,
    completed_steps: int,
    *,
    macro_start_s: float | None = None,
    split_times_s: tuple[float, ...] = (),
) -> float:
    """Return the indexed macro-grid boundary, snapping only roundoff to the end.

    Repeatedly adding ``dt_s`` can leave a one-ULP tail at an intended decimal
    endpoint (for example 0.02 s repeated ten times toward 0.2 s).  Such a tail
    is not a physical integration interval and may be too short to represent a
    midpoint.  Time-dependent field knots are merged into the same ordered
    interval sequence without changing the fixed grid on either side.
    """

    crossed_splits = 0
    if split_times_s:
        if macro_start_s is None:
            raise EngineError("field-time split scheduling requires the current macro time")
        crossed_splits = bisect_right(split_times_s, macro_start_s)
    regular_completed_steps = completed_steps - crossed_splits
    if regular_completed_steps < 0:
        raise EngineError("field-time split schedule disagrees with the macro-step count")
    step_number = regular_completed_steps + 1
    nominal_end_s, construction_roundoff_s = _time_grid_boundary(time, step_number)
    if not math.isfinite(nominal_end_s) or nominal_end_s >= time.end_s:
        nominal_end_s = time.end_s
    elif time.end_s - nominal_end_s <= construction_roundoff_s:
        nominal_end_s = time.end_s
    if crossed_splits < len(split_times_s):
        field_knot_s = split_times_s[crossed_splits]
        if field_knot_s < nominal_end_s:
            return field_knot_s
    return nominal_end_s


def _time_grid_boundary(time: TimeSpec, step_number: int) -> tuple[float, float]:
    """Construct one fixed-grid boundary and its binary64 roundoff budget."""

    nominal_offset_s, offset_residual_s = _time_grid_product(
        float(step_number),
        time.dt_s,
    )
    nominal_end_s = math.fsum((time.start_s, nominal_offset_s, offset_residual_s))
    if not math.isfinite(nominal_end_s):
        return nominal_end_s, 0.0
    construction_roundoff_s = _TIME_GRID_ROUNDOFF_ULPS * max(
        math.ulp(time.start_s),
        math.ulp(time.end_s),
        math.ulp(time.dt_s) * step_number,
        math.ulp(nominal_offset_s),
        math.ulp(nominal_end_s),
    )
    return nominal_end_s, construction_roundoff_s


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


def _field_time_split_times(
    time: TimeSpec,
    fields: PreparedFieldSet,
) -> tuple[float, ...]:
    """Return non-grid snapshot knots that must split production intervals."""

    result: list[float] = []
    for knot_s in fields.time_knots_s():
        if knot_s <= time.start_s or knot_s >= time.end_s:
            continue
        relative_step = (knot_s - time.start_s) / time.dt_s
        nearest_step = round(relative_step)
        aligned = False
        if nearest_step >= 1:
            grid_s, roundoff_s = _time_grid_boundary(time, nearest_step)
            aligned = abs(knot_s - grid_s) <= roundoff_s
        if not aligned:
            result.append(knot_s)
    return tuple(result)


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
            "tree_policy_revision": None,
            "interval_tree_depth": None,
            "adaptive_max_depth": None,
            "root_normal_stream": None,
            "split_normal_stream": None,
        }
    return {
        "rng_revision": BROWNIAN_RNG_REVISION,
        "ou_revision": JOINT_OU_REVISION,
        "split_revision": JOINT_OU_SPLIT_REVISION,
        "coefficient_policy": "macro_root_frozen_midpoint_v1",
        "composition_revision": "stochastic_exponential_midpoint_v1",
        "charge_dense_revision": "macro_root_affine_exponential_v2",
        "tree_policy_revision": BROWNIAN_TREE_POLICY_REVISION,
        "interval_tree_depth": physics.noise.interval_tree_depth,
        "adaptive_max_depth": physics.noise.adaptive_max_depth,
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
        "field_time_revision": FIELD_TIME_REVISION,
        "field_time_split_s": list(prepared.field_time_split_s),
        "required_field_revision": REQUIRED_FIELD_REVISION,
        "geometry_algorithm_revision": GEOMETRY_ALGORITHM_REVISION,
        "event_algorithm_revision": EVENT_ALGORITHM_REVISION,
        "topology_algorithm_revision": _topology_algorithm_revision(prepared),
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
        "brownian_tree_policy_revision": brownian["tree_policy_revision"],
        "brownian_interval_tree_depth": brownian["interval_tree_depth"],
        "brownian_adaptive_max_depth": brownian["adaptive_max_depth"],
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
    writer: _EventSink,
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
    writer: _EventSink,
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
    if noise.revision != BROWNIAN_MIDPOINT_2D_REVISION:
        raise EngineError("Langevin coefficients require the resolved 2-D midpoint revision")
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
    requested_target_time_s: float,
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
) -> tuple[_LangevinRootBatch | None, np.ndarray]:
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
        return None, rows
    return (
        _LangevinRootBatch(
            particles[rows],
            starts_s[rows],
            duration_s[rows],
            np.where(
                duration_s[rows] < requested_duration_s[rows],
                starts_s[rows] + duration_s[rows],
                requested_target_time_s,
            ),
            start_position_m[rows],
            start_velocity_m_s[rows],
            start_charge_number[rows],
            _subset_langevin_coefficients(coefficients, rows),
            numerical_status[rows],
            safe_rate[rows],
            safe_thermal[rows],
            safe_equilibrium[rows],
        ),
        rows,
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
    reason_code: np.ndarray,
) -> None:
    """Fail a root cohort that exhausted its bounded stochastic restarts."""

    for local_row, particle_value in enumerate(particles):
        particle = int(particle_value)
        _mark_particle_failed(
            particle,
            _ParticleFailure(
                reason_code[local_row],
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


def _langevin_restart_failure_reasons(
    prepared: _PreparedRun,
    stochastic_event_restart_count: np.ndarray,
    guard_restart_count: np.ndarray,
) -> np.ndarray:
    """Map independent bounded-restart exhaustion to its owned failure class."""

    reasons = np.zeros(stochastic_event_restart_count.size, dtype="<u2")
    event_exhausted = (
        stochastic_event_restart_count > prepared.case.spec.solver.event.max_interactions_per_step
    )
    reasons[event_exhausted] = _FAILURE_NUMERICAL_EVENT_BUDGET
    guard_exhausted = guard_restart_count > prepared.case.spec.solver.event.max_refinements
    reasons[(reasons == 0) & guard_exhausted] = _FAILURE_INTEGRATOR_ACCURACY
    return reasons


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
    writer: _EventSink,
    event_buffer: _BoundaryEventBuffer,
    failure_buffer: _FailureEventBuffer,
) -> None:
    """Advance fresh OU roots iteratively with one bounded pending row wave."""

    noise = prepared.physics.noise
    if noise is None:
        raise EngineError("Brownian slab requires a resolved noise model")
    capacity = int(particle_index.size)
    pending_particle_index = particle_index
    pending_start_time_s = root_start_time_s
    pending_start_position_m = root_start_position_m
    pending_start_velocity_m_s = root_start_velocity_m_s
    pending_root_interval = np.zeros(capacity, dtype="<i8")
    pending_stochastic_event_restart_count = np.zeros(capacity, dtype="<i8")
    pending_guard_restart_count = np.zeros(capacity, dtype="<i8")
    pending_count = capacity
    initial_interactions = np.zeros(capacity, dtype="<i8")
    release_rows = np.flatnonzero(
        start_contact_state[pending_particle_index] == SURFACE_STATE_PENDING
    ).astype("<i8", copy=False)
    if release_rows.size:
        release_particles = pending_particle_index[release_rows]
        release_interactions, _ = _initialize_surface_releases(
            prepared,
            release_particles,
            pending_start_time_s[release_rows],
            pending_start_position_m[release_rows],
            pending_start_velocity_m_s[release_rows],
            charge_number[release_particles].copy(),
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
        initial_interactions[release_rows] = release_interactions
        _flush_boundary_event_wave(writer, prepared, event_buffer)
    responded = (initial_interactions != 0) & active[pending_particle_index]
    if bool(responded.any()):
        pending_start_position_m = pending_start_position_m.copy()
        pending_start_velocity_m_s = pending_start_velocity_m_s.copy()
        pending_start_position_m[responded] = position_m[pending_particle_index[responded]]
        pending_start_velocity_m_s[responded] = velocity_m_s[pending_particle_index[responded]]
        pending_root_interval[responded] += initial_interactions[responded]
        pending_stochastic_event_restart_count[responded] += initial_interactions[responded]
    del particle_index
    del root_start_time_s
    del root_start_position_m
    del root_start_velocity_m_s

    adaptive_workspace = _BrownianAdaptiveWorkspace.allocate(
        capacity, noise.adaptive_max_depth - noise.interval_tree_depth
    )
    (
        next_particle_index,
        next_start_time_s,
        next_start_position_m,
        next_start_velocity_m_s,
        next_root_interval,
        next_stochastic_event_restart_count,
        next_guard_restart_count,
    ) = _allocate_brownian_pending_columns(capacity)
    next_count = np.zeros(1, dtype="<i8")
    current_bank_owned = False
    while pending_count:
        next_count[0] = 0
        runtime = _prepare_brownian_root_wave(
            prepared,
            pending_particle_index[:pending_count],
            macro_interval,
            macro_end_s,
            pending_start_time_s[:pending_count],
            pending_start_position_m[:pending_count],
            pending_start_velocity_m_s[:pending_count],
            pending_root_interval[:pending_count],
            pending_stochastic_event_restart_count[:pending_count],
            pending_guard_restart_count[:pending_count],
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
            next_particle_index,
            next_start_time_s,
            next_start_position_m,
            next_start_velocity_m_s,
            next_root_interval,
            next_stochastic_event_restart_count,
            next_guard_restart_count,
            next_count,
            adaptive_workspace,
        )
        if runtime is not None:
            while _advance_brownian_base_leaf(runtime):
                pass
            _finish_brownian_root_wave(runtime, macro_end_s)
        # Every base/adaptive consumer has returned before either pending bank
        # can be recycled for the next root wave.
        del runtime
        pending_count = int(next_count[0])
        if not pending_count:
            break
        previous_columns = None
        if current_bank_owned:
            previous_columns = (
                pending_particle_index,
                pending_start_time_s,
                pending_start_position_m,
                pending_start_velocity_m_s,
                pending_root_interval,
                pending_stochastic_event_restart_count,
                pending_guard_restart_count,
            )
        (
            pending_particle_index,
            pending_start_time_s,
            pending_start_position_m,
            pending_start_velocity_m_s,
            pending_root_interval,
            pending_stochastic_event_restart_count,
            pending_guard_restart_count,
        ) = (
            next_particle_index,
            next_start_time_s,
            next_start_position_m,
            next_start_velocity_m_s,
            next_root_interval,
            next_stochastic_event_restart_count,
            next_guard_restart_count,
        )
        # The initial source bank is borrowed.  Drop those references before
        # allocating its writable replacement; never recycle caller inputs.
        if previous_columns is None:
            previous_columns = _allocate_brownian_pending_columns(capacity)
        (
            next_particle_index,
            next_start_time_s,
            next_start_position_m,
            next_start_velocity_m_s,
            next_root_interval,
            next_stochastic_event_restart_count,
            next_guard_restart_count,
        ) = previous_columns
        del previous_columns
        current_bank_owned = True


def _allocate_brownian_pending_columns(
    capacity: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Allocate one writable 72 B/row fresh-root bank, reused after full unwind."""

    return (
        np.empty(capacity, dtype="<i8"),
        np.empty(capacity, dtype="<f8"),
        np.empty((capacity, 2), dtype="<f8"),
        np.empty((capacity, 2), dtype="<f8"),
        np.empty(capacity, dtype="<i8"),
        np.empty(capacity, dtype="<i8"),
        np.empty(capacity, dtype="<i8"),
    )


def _prepare_brownian_root_wave(
    prepared: _PreparedRun,
    particle_index: np.ndarray,
    macro_interval: int,
    macro_end_s: float,
    root_start_time_s: np.ndarray,
    root_start_position_m: np.ndarray,
    root_start_velocity_m_s: np.ndarray,
    root_interval: np.ndarray,
    stochastic_event_restart_count: np.ndarray,
    guard_restart_count: np.ndarray,
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
    writer: _EventSink,
    event_buffer: _BoundaryEventBuffer,
    failure_buffer: _FailureEventBuffer,
    pending_particle_index: np.ndarray,
    pending_start_time_s: np.ndarray,
    pending_start_position_m: np.ndarray,
    pending_start_velocity_m_s: np.ndarray,
    pending_root_interval: np.ndarray,
    pending_stochastic_event_restart_count: np.ndarray,
    pending_guard_restart_count: np.ndarray,
    pending_count: np.ndarray,
    adaptive_workspace: _BrownianAdaptiveWorkspace,
) -> _BrownianSlabRuntime | None:
    """Admit one fresh-root wave and retain its sole cursor until full unwind."""

    noise = prepared.physics.noise
    if noise is None:
        raise EngineError("Brownian root wave requires a resolved noise model")
    root_duration_s = macro_end_s - root_start_time_s
    moving_rows = np.flatnonzero((root_duration_s > 0.0) & active[particle_index]).astype(
        "<i8", copy=False
    )
    if not moving_rows.size:
        return
    particles = particle_index[moving_rows]
    starts = root_start_time_s[moving_rows]
    requested_durations = root_duration_s[moving_rows]
    start_positions = root_start_position_m[moving_rows]
    start_velocities = root_start_velocity_m_s[moving_rows]
    root_intervals = root_interval[moving_rows]
    event_restart_counts = stochastic_event_restart_count[moving_rows]
    guard_restart_counts = guard_restart_count[moving_rows]
    start_charges = charge_number[particles].copy()
    restart_failure = _langevin_restart_failure_reasons(
        prepared,
        event_restart_counts,
        guard_restart_counts,
    )
    failed_budget = restart_failure != 0
    if bool(failed_budget.any()):
        _fail_langevin_root_budget(
            particles[failed_budget],
            starts[failed_budget],
            start_positions[failed_budget],
            start_velocities[failed_budget],
            start_charges[failed_budget],
            position_m,
            velocity_m_s,
            charge_number,
            active,
            lifecycle,
            terminal_time_s,
            event_ordinal,
            failure_reason_code,
            failure_buffer,
            restart_failure[failed_budget],
        )
    within_budget = ~failed_budget
    if not bool(within_budget.any()):
        return
    particles = particles[within_budget]
    starts = starts[within_budget]
    requested_durations = requested_durations[within_budget]
    start_positions = start_positions[within_budget]
    start_velocities = start_velocities[within_budget]
    start_charges = start_charges[within_budget]
    root_intervals = root_intervals[within_budget]
    event_restart_counts = event_restart_counts[within_budget]
    guard_restart_counts = guard_restart_counts[within_budget]
    batch, source_rows = _prepare_langevin_root_batch(
        prepared,
        particles,
        starts,
        requested_durations,
        macro_end_s,
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
    numerical_status = batch.numerical_status
    safe_rate = batch.drag_rate_s_inv
    safe_thermal = batch.thermal_velocity_variance_m2_s2
    root_intervals = root_intervals[source_rows]
    event_restart_counts = event_restart_counts[source_rows]
    guard_restart_counts = guard_restart_counts[source_rows]
    del source_rows
    particle_ids = prepared.schedule.particle_id[particles].astype(np.uint64, copy=False)
    increments = _brownian_leaf_increments(
        seed=prepared.case.spec.solver.seed,
        particle_id=particle_ids,
        macro_interval=macro_interval,
        root_interval=root_intervals,
        drag_rate_s_inv=safe_rate,
        thermal_velocity_variance_m2_s2=safe_thermal,
        root_duration_s=durations,
        tree_depth=noise.interval_tree_depth,
        numerically_valid=numerical_status == NUMERICAL_STATUS_OK,
    )
    root_continuing = np.ones(particles.size, dtype=np.bool_)
    return _BrownianSlabRuntime(
        prepared=prepared,
        batch=batch,
        particle_ids=particle_ids,
        macro_interval=macro_interval,
        root_interval=root_intervals,
        stochastic_event_restart_count=event_restart_counts,
        guard_restart_count=guard_restart_counts,
        position_m=position_m,
        velocity_m_s=velocity_m_s,
        charge_number=charge_number,
        active=active,
        lifecycle=lifecycle,
        terminal_time_s=terminal_time_s,
        event_ordinal=event_ordinal,
        physical_boundary_event_ordinal=physical_boundary_event_ordinal,
        exact_origin_time_s=exact_origin_time_s,
        exact_origin_position_m=exact_origin_position_m,
        exact_origin_velocity_m_s=exact_origin_velocity_m_s,
        start_contact_state=start_contact_state,
        replay=replay,
        statistics=statistics,
        failure_reason_code=failure_reason_code,
        writer=writer,
        event_buffer=event_buffer,
        failure_buffer=failure_buffer,
        root_continuing=root_continuing,
        pending_particle_index=pending_particle_index,
        pending_start_time_s=pending_start_time_s,
        pending_start_position_m=pending_start_position_m,
        pending_start_velocity_m_s=pending_start_velocity_m_s,
        pending_root_interval=pending_root_interval,
        pending_stochastic_event_restart_count=pending_stochastic_event_restart_count,
        pending_guard_restart_count=pending_guard_restart_count,
        pending_count=pending_count,
        adaptive_workspace=adaptive_workspace,
        leaf_cursor=increments,
    )


def _advance_brownian_base_leaf(runtime: _BrownianSlabRuntime) -> bool:
    """Consume one complete base leaf; retain this same root for the next call.

    The native cursor owns both the next index and the materialized leaf index.
    Its current buffers remain borrowed only while the synchronous adaptive
    consumer runs; a return is therefore a complete base-leaf resume boundary.
    """

    increments = runtime.leaf_cursor
    leaf_index = _next_brownian_leaf_into(
        increments.seed,
        increments.macro_interval,
        increments.particle_id,
        increments.root_interval,
        increments.drag_rate_s_inv,
        increments.thermal_velocity_variance_m2_s2,
        increments.root_duration_s,
        increments.tree_depth,
        increments.next_leaf_index,
        increments.position_m,
        increments.velocity_m_s,
        increments.valid,
        increments.right_position_m,
        increments.right_velocity_m_s,
        increments.right_valid,
    )
    if leaf_index < 0:
        return False
    particles = runtime.batch.particles
    selected = np.flatnonzero(runtime.active[particles] & runtime.root_continuing).astype(
        "<i8", copy=False
    )
    if not selected.size:
        return False
    _advance_brownian_tree_node(
        runtime,
        selected,
        JointOuIncrement(increments.position_m, increments.velocity_m_s),
        selected,
        increments.valid,
        increments.tree_depth,
        leaf_index,
    )
    return True


def _finish_brownian_root_wave(runtime: _BrownianSlabRuntime, macro_end_s: float) -> None:
    """Queue shortened-root residuals once, after every leaf consumer returned."""

    particles = runtime.batch.particles
    root_target_s = runtime.batch.target_time_s
    residual_rows = np.flatnonzero(
        runtime.root_continuing & runtime.active[particles] & (root_target_s < macro_end_s)
    ).astype("<i8", copy=False)
    if not residual_rows.size:
        return
    residual_particles = particles[residual_rows]
    _queue_langevin_roots(
        runtime,
        residual_rows,
        residual_particles,
        root_target_s[residual_rows],
        event_restart_increment=0,
        guard_restart_increment=1,
    )


def _advance_brownian_tree_node(
    runtime: _BrownianSlabRuntime,
    root_rows: np.ndarray,
    increment: JointOuIncrement,
    increment_rows: np.ndarray,
    increment_valid: np.ndarray,
    tree_level: int,
    tree_index: int,
) -> None:
    """Consume a complete base leaf through one bounded numeric adaptive DFS.

    A clear subset commits before the refined left subtree and then the right
    subtree, exactly as in the recursive traversal.  Root-aligned increment
    columns and pending right frames remain owned until this leaf unwinds.
    """

    workspace = runtime.adaptive_workspace
    if workspace.right_count.size == 0:
        _consume_brownian_tree_node(
            runtime, root_rows, increment, increment_rows, increment_valid, tree_level, tree_index
        )
        return
    _begin_brownian_adaptive_into(
        root_rows,
        increment_rows,
        increment.position_m,
        increment.velocity_m_s,
        increment_valid,
        tree_level,
        tree_index,
        workspace.root_rows,
        workspace.position_m,
        workspace.velocity_m_s,
        workspace.valid,
        workspace.control,
    )
    current_increment = JointOuIncrement(workspace.position_m, workspace.velocity_m_s)
    while workspace.control[0]:
        count, level, index = (int(value) for value in workspace.control[:3])
        current_rows = workspace.root_rows[:count]
        refined = _consume_brownian_tree_node(
            runtime, current_rows, current_increment, current_rows, workspace.valid, level, index
        )
        if refined is None:
            _pop_brownian_adaptive_into(
                workspace.root_rows,
                workspace.position_m,
                workspace.velocity_m_s,
                workspace.valid,
                workspace.right_root_rows,
                workspace.right_position_m,
                workspace.right_velocity_m_s,
                workspace.right_valid,
                workspace.right_count,
                workspace.right_level,
                workspace.right_index,
                workspace.control,
            )
            continue
        refined_roots, refined_increments = refined
        _push_brownian_adaptive_into(
            np.uint64(runtime.prepared.case.spec.solver.seed),
            np.uint64(runtime.macro_interval),
            runtime.particle_ids,
            runtime.root_interval,
            runtime.batch.drag_rate_s_inv,
            runtime.batch.thermal_velocity_variance_m2_s2,
            runtime.batch.duration_s,
            refined_roots,
            refined_increments,
            workspace.root_rows,
            workspace.position_m,
            workspace.velocity_m_s,
            workspace.valid,
            workspace.right_root_rows,
            workspace.right_position_m,
            workspace.right_velocity_m_s,
            workspace.right_valid,
            workspace.right_count,
            workspace.right_level,
            workspace.right_index,
            workspace.control,
        )
        runtime.statistics.refinements += int(refined_roots.size)
        runtime.statistics.maximum_refinement_depth = max(
            runtime.statistics.maximum_refinement_depth, level + 1
        )


def _consume_brownian_tree_node(
    runtime: _BrownianSlabRuntime,
    root_rows: np.ndarray,
    increment: JointOuIncrement,
    increment_rows: np.ndarray,
    increment_valid: np.ndarray,
    tree_level: int,
    tree_index: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Commit one node's clear rows and return only its refinement row maps."""

    batch = runtime.batch
    particles = batch.particles[root_rows]
    continuing = runtime.active[particles] & runtime.root_continuing[root_rows]
    root_rows = root_rows[continuing]
    increment_rows = increment_rows[continuing]
    if not root_rows.size:
        return

    node_count = 1 << tree_level
    start_time_s = batch.starts_s[root_rows] + batch.duration_s[root_rows] * (
        tree_index / node_count
    )
    target_time_s = batch.starts_s[root_rows] + batch.duration_s[root_rows] * (
        (tree_index + 1) / node_count
    )
    if tree_index + 1 == node_count:
        # Duration is the OU transition/split authority; the absolute requested
        # endpoint is the clock authority. Re-adding a rounded remainder can
        # land one ULP before (or after) that endpoint after a wall restart.
        target_time_s = batch.target_time_s[root_rows]
    particles = batch.particles[root_rows]
    current_position = runtime.position_m[particles].copy()
    current_velocity = runtime.velocity_m_s[particles].copy()
    current_charge = runtime.charge_number[particles].copy()
    resolved_time = target_time_s > start_time_s
    for local_row_value in np.flatnonzero(~resolved_time):
        local_row = int(local_row_value)
        particle = int(particles[local_row])
        _mark_particle_failed(
            particle,
            _ParticleFailure(
                _FAILURE_INTEGRATOR_ACCURACY,
                float(start_time_s[local_row]),
                current_position[local_row],
                current_velocity[local_row],
                float(current_charge[local_row]),
            ),
            runtime.position_m,
            runtime.velocity_m_s,
            runtime.charge_number,
            runtime.active,
            runtime.lifecycle,
            runtime.terminal_time_s,
            runtime.event_ordinal,
            runtime.failure_reason_code,
            runtime.failure_buffer,
        )
    root_rows = root_rows[resolved_time]
    increment_rows = increment_rows[resolved_time]
    if not root_rows.size:
        return
    start_time_s = start_time_s[resolved_time]
    target_time_s = target_time_s[resolved_time]
    particles = particles[resolved_time]
    current_position = current_position[resolved_time]
    current_velocity = current_velocity[resolved_time]
    current_charge = current_charge[resolved_time]

    node_valid = increment_valid[increment_rows]
    _mark_brownian_numerical_failures(
        node_valid,
        particles,
        start_time_s,
        current_position,
        current_velocity,
        current_charge,
        runtime.position_m,
        runtime.velocity_m_s,
        runtime.charge_number,
        runtime.active,
        runtime.lifecycle,
        runtime.terminal_time_s,
        runtime.event_ordinal,
        runtime.failure_reason_code,
        runtime.failure_buffer,
    )
    root_rows = root_rows[node_valid]
    increment_rows = increment_rows[node_valid]
    if not root_rows.size:
        return
    start_time_s = start_time_s[node_valid]
    target_time_s = target_time_s[node_valid]
    particles = particles[node_valid]
    current_position = current_position[node_valid]
    current_velocity = current_velocity[node_valid]
    current_charge = current_charge[node_valid]

    duration_s = batch.duration_s[root_rows] / node_count
    end_position, end_velocity, advance_valid = _advance_langevin_leaf_endpoint(
        current_position,
        current_velocity,
        batch.equilibrium_velocity_m_s[root_rows],
        batch.drag_rate_s_inv[root_rows],
        duration_s,
        increment,
        increment_rows,
    )
    _mark_brownian_numerical_failures(
        advance_valid,
        particles,
        start_time_s,
        current_position,
        current_velocity,
        current_charge,
        runtime.position_m,
        runtime.velocity_m_s,
        runtime.charge_number,
        runtime.active,
        runtime.lifecycle,
        runtime.terminal_time_s,
        runtime.event_ordinal,
        runtime.failure_reason_code,
        runtime.failure_buffer,
    )
    root_rows = root_rows[advance_valid]
    increment_rows = increment_rows[advance_valid]
    if not root_rows.size:
        return
    start_time_s = start_time_s[advance_valid]
    target_time_s = target_time_s[advance_valid]
    particles = particles[advance_valid]
    current_position = current_position[advance_valid]
    current_velocity = current_velocity[advance_valid]
    current_charge = current_charge[advance_valid]
    end_position = end_position[advance_valid]
    end_velocity = end_velocity[advance_valid]

    coefficients = batch.coefficients
    noise = runtime.prepared.physics.noise
    if noise is None:
        raise EngineError("Brownian tree lost its resolved noise model")
    end_charge = _advance_langevin_leaf_charge(
        batch.start_charge_number[root_rows],
        target_time_s - batch.starts_s[root_rows],
        coefficients.charge_affine_rate_number_s[root_rows],
        coefficients.charge_rate_derivative_s_inv[root_rows],
    )
    charge_valid = np.isfinite(end_charge)
    _mark_brownian_numerical_failures(
        charge_valid,
        particles,
        start_time_s,
        current_position,
        current_velocity,
        current_charge,
        runtime.position_m,
        runtime.velocity_m_s,
        runtime.charge_number,
        runtime.active,
        runtime.lifecycle,
        runtime.terminal_time_s,
        runtime.event_ordinal,
        runtime.failure_reason_code,
        runtime.failure_buffer,
    )
    root_rows = root_rows[charge_valid]
    increment_rows = increment_rows[charge_valid]
    if not root_rows.size:
        return
    start_time_s = start_time_s[charge_valid]
    target_time_s = target_time_s[charge_valid]
    particles = particles[charge_valid]
    current_position = current_position[charge_valid]
    current_velocity = current_velocity[charge_valid]
    current_charge = current_charge[charge_valid]
    end_position = end_position[charge_valid]
    end_velocity = end_velocity[charge_valid]
    end_charge = end_charge[charge_valid]

    proposal = _build_langevin_leaf_proposal(
        particles,
        start_time_s,
        target_time_s,
        current_position,
        current_velocity,
        current_charge,
        end_position,
        end_velocity,
        end_charge,
        batch.starts_s[root_rows],
        batch.start_charge_number[root_rows],
        coefficients.charge_affine_rate_number_s[root_rows],
        coefficients.charge_rate_derivative_s_inv[root_rows],
        coefficients.support_inside[root_rows],
        coefficients.applicability_inside[root_rows],
        batch.numerical_status[root_rows],
    )
    if tree_level >= noise.adaptive_max_depth:
        _commit_brownian_tree_proposal(runtime, proposal, root_rows)
        return

    clear = _brownian_certified_clear_rows(runtime, proposal)
    clear_rows = np.flatnonzero(clear).astype("<i8", copy=False)
    if clear_rows.size:
        clear_proposal = _subset_brownian_tree_proposal(
            runtime,
            proposal,
            root_rows,
            clear_rows,
        )
        _commit_brownian_tree_proposal(runtime, clear_proposal, root_rows[clear_rows])
        del clear_proposal

    refine_rows = np.flatnonzero(~clear).astype("<i8", copy=False)
    if not refine_rows.size:
        return
    refine_root_rows = root_rows[refine_rows]
    refine_increment_rows = increment_rows[refine_rows]
    # Returning releases the full proposal and endpoint temporaries before the
    # native controller conditions children.  Only these compact maps survive.
    return refine_root_rows, refine_increment_rows


def _brownian_certified_clear_rows(
    runtime: _BrownianSlabRuntime,
    proposal: StepProposal,
) -> np.ndarray:
    """Return rows whose represented cubic interval is continuously clear."""

    eligible = proposal.numerical_status == NUMERICAL_STATUS_OK
    eligible &= proposal.support_inside
    eligible &= proposal.applicability_inside
    particle_contact = runtime.start_contact_state[proposal.particle_index]
    eligible &= particle_contact != SURFACE_STATE_PENDING
    if not _uses_curved_event_path(runtime.prepared):
        return eligible

    clear = np.zeros(proposal.particle_index.size, dtype=np.bool_)
    query_rows = np.flatnonzero(eligible).astype("<i8", copy=False)
    if not query_rows.size:
        return clear
    runtime.statistics.candidate_queries += int(query_rows.size)
    duration_s = proposal.target_time_s[query_rows] - proposal.start_time_s[query_rows]
    certify_departure = particle_contact[query_rows] == SURFACE_STATE_DEPARTURE
    for begin, end, event in _locate_curved_proposal_batches(
        runtime.prepared,
        proposal,
        query_rows,
        proposal.start_time_s[query_rows],
        proposal.target_time_s[query_rows],
        duration_s,
        certify_departure,
    ):
        bounded_rows = query_rows[begin:end]
        failure_codes = _curved_row_failure_codes(
            runtime.prepared,
            proposal,
            bounded_rows,
            start_contact_certified=event.start_contact_departure_certified,
        )
        clear[bounded_rows] = (event.status == CURVED_STATUS_CLEAR) & (failure_codes == 0)
    return clear


def _subset_brownian_tree_proposal(
    runtime: _BrownianSlabRuntime,
    proposal: StepProposal,
    root_rows: np.ndarray,
    local_rows: np.ndarray,
) -> StepProposal:
    """Select adaptive clear rows without exposing integrator-private path state."""

    if local_rows.size == proposal.particle_index.size:
        return proposal
    selected = root_rows[local_rows]
    coefficients = runtime.batch.coefficients
    return _build_langevin_leaf_proposal(
        proposal.particle_index[local_rows],
        proposal.start_time_s[local_rows],
        proposal.target_time_s[local_rows],
        proposal.start_position_m[local_rows],
        proposal.start_velocity_m_s[local_rows],
        proposal.start_charge_number[local_rows],
        proposal.end_position_m[local_rows],
        proposal.end_velocity_m_s[local_rows],
        proposal.end_charge_number[local_rows],
        runtime.batch.starts_s[selected],
        runtime.batch.start_charge_number[selected],
        coefficients.charge_affine_rate_number_s[selected],
        coefficients.charge_rate_derivative_s_inv[selected],
        proposal.support_inside[local_rows],
        proposal.applicability_inside[local_rows],
        proposal.numerical_status[local_rows],
    )


def _commit_brownian_tree_proposal(
    runtime: _BrownianSlabRuntime,
    proposal: StepProposal,
    root_rows: np.ndarray,
) -> None:
    """Commit one accepted node through the existing event/replay owner."""

    restart_time_s = _advance_macro_proposal(
        runtime.prepared,
        proposal,
        runtime.position_m,
        runtime.velocity_m_s,
        runtime.charge_number,
        runtime.active,
        runtime.lifecycle,
        runtime.terminal_time_s,
        runtime.event_ordinal,
        runtime.physical_boundary_event_ordinal,
        runtime.exact_origin_time_s,
        runtime.exact_origin_position_m,
        runtime.exact_origin_velocity_m_s,
        runtime.start_contact_state,
        runtime.replay,
        runtime.statistics,
        runtime.failure_reason_code,
        runtime.writer,
        runtime.event_buffer,
        runtime.failure_buffer,
    )
    restart_rows = _queue_langevin_event_rows(
        runtime,
        restart_time_s,
        root_rows,
        proposal.particle_index,
    )
    runtime.root_continuing[restart_rows] = False


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
    root_charge_number: np.ndarray,
    elapsed_from_root_s: np.ndarray,
    charge_affine_rate_number_s: np.ndarray,
    charge_rate_derivative_s_inv: np.ndarray,
) -> np.ndarray:
    """Advance the unified midpoint revision's root-owned charge representation."""

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


def _queue_langevin_roots(
    runtime: _BrownianSlabRuntime,
    root_rows: np.ndarray,
    particles: np.ndarray,
    start_time_s: np.ndarray,
    *,
    event_restart_increment: int,
    guard_restart_increment: int,
) -> None:
    """Append one disjoint fresh-root subset to the slab's bounded SoA wave."""

    count = int(root_rows.size)
    if not count:
        return
    begin = int(runtime.pending_count[0])
    end = begin + count
    if end > runtime.pending_particle_index.size:
        raise EngineError("Brownian pending fresh-root wave exceeded its slab capacity")
    _append_brownian_pending_into(
        root_rows,
        particles,
        start_time_s,
        runtime.position_m,
        runtime.velocity_m_s,
        runtime.root_interval,
        runtime.stochastic_event_restart_count,
        runtime.guard_restart_count,
        event_restart_increment,
        guard_restart_increment,
        runtime.pending_particle_index,
        runtime.pending_start_time_s,
        runtime.pending_start_position_m,
        runtime.pending_start_velocity_m_s,
        runtime.pending_root_interval,
        runtime.pending_stochastic_event_restart_count,
        runtime.pending_guard_restart_count,
        runtime.pending_count,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _append_brownian_pending_into(
    root_rows: np.ndarray,
    particles: np.ndarray,
    start_time_s: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    root_interval: np.ndarray,
    event_restart_count: np.ndarray,
    guard_restart_count: np.ndarray,
    event_restart_increment: int,
    guard_restart_increment: int,
    pending_particles: np.ndarray,
    pending_start_time: np.ndarray,
    pending_position: np.ndarray,
    pending_velocity: np.ndarray,
    pending_root_interval: np.ndarray,
    pending_event_restart_count: np.ndarray,
    pending_guard_restart_count: np.ndarray,
    pending_count: np.ndarray,
) -> None:
    """Append checked disjoint rows in stable commit order without gathers.

    Capacity is checked by the engine wrapper before mutation.  Pending columns
    are disjoint from both immutable current-root metadata and resident state.
    """

    begin = pending_count[0]
    for local_row in range(root_rows.size):
        target = begin + local_row
        root_row = root_rows[local_row]
        particle = particles[local_row]
        pending_particles[target] = particle
        pending_start_time[target] = start_time_s[local_row]
        pending_position[target, 0] = position[particle, 0]
        pending_position[target, 1] = position[particle, 1]
        pending_velocity[target, 0] = velocity[particle, 0]
        pending_velocity[target, 1] = velocity[particle, 1]
        pending_root_interval[target] = root_interval[root_row] + 1
        pending_event_restart_count[target] = (
            event_restart_count[root_row] + event_restart_increment
        )
        pending_guard_restart_count[target] = (
            guard_restart_count[root_row] + guard_restart_increment
        )
    pending_count[0] = begin + root_rows.size


def _queue_langevin_event_rows(
    runtime: _BrownianSlabRuntime,
    restart_time_s: np.ndarray | None,
    selected: np.ndarray,
    selected_particles: np.ndarray,
) -> np.ndarray:
    """Queue rows whose accepted cubic prefix reached an axis or active wall."""

    if restart_time_s is None:
        return np.empty(0, dtype="<i8")
    restart_rows = np.flatnonzero(np.isfinite(restart_time_s)).astype("<i8", copy=False)
    if not restart_rows.size:
        return np.empty(0, dtype="<i8")
    restart_particles = selected_particles[restart_rows]
    selected_root_rows = selected[restart_rows]
    _queue_langevin_roots(
        runtime,
        selected_root_rows,
        restart_particles,
        restart_time_s[restart_rows],
        event_restart_increment=1,
        guard_restart_increment=0,
    )
    return selected_root_rows


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


@dataclass(frozen=True, slots=True)
class _BrownianAdaptiveWorkspace:
    """Reusable numeric DFS columns, bounded by slab rows and extra depth.

    Increment columns use immutable root-row indices; compact row maps select
    the current node.  A saved right frame is copied back before its slot can
    be reused, so no live child borrows a subsequently overwritten bank.
    """

    root_rows: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    valid: np.ndarray
    right_root_rows: np.ndarray
    right_position_m: np.ndarray
    right_velocity_m_s: np.ndarray
    right_valid: np.ndarray
    right_count: np.ndarray
    right_level: np.ndarray
    right_index: np.ndarray
    control: np.ndarray

    @classmethod
    def allocate(cls, capacity: int, extra_depth: int) -> _BrownianAdaptiveWorkspace:
        rows = capacity if extra_depth else 0
        return cls(
            np.empty(rows, dtype="<i8"),
            np.empty((rows, 2), dtype="<f8"),
            np.empty((rows, 2), dtype="<f8"),
            np.empty(rows, dtype=np.bool_),
            np.empty((extra_depth, rows), dtype="<i8"),
            np.empty((extra_depth, rows, 2), dtype="<f8"),
            np.empty((extra_depth, rows, 2), dtype="<f8"),
            np.empty((extra_depth, rows), dtype=np.bool_),
            np.empty(extra_depth, dtype="<i8"),
            np.empty(extra_depth, dtype="<i8"),
            np.empty(extra_depth, dtype="<i8"),
            np.zeros(4, dtype="<i8"),
        )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _begin_brownian_adaptive_into(
    root_rows: np.ndarray,
    increment_rows: np.ndarray,
    parent_position: np.ndarray,
    parent_velocity: np.ndarray,
    parent_valid: np.ndarray,
    tree_level: int,
    tree_index: int,
    current_roots: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    valid: np.ndarray,
    control: np.ndarray,
) -> None:
    for local_row in range(root_rows.size):
        root_row = root_rows[local_row]
        increment_row = increment_rows[local_row]
        current_roots[local_row] = root_row
        position[root_row, 0] = parent_position[increment_row, 0]
        position[root_row, 1] = parent_position[increment_row, 1]
        velocity[root_row, 0] = parent_velocity[increment_row, 0]
        velocity[root_row, 1] = parent_velocity[increment_row, 1]
        valid[root_row] = parent_valid[increment_row]
    control[0] = root_rows.size
    control[1] = tree_level
    control[2] = tree_index
    control[3] = 0


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _push_brownian_adaptive_into(
    seed: np.uint64,
    macro_interval: np.uint64,
    particle_id: np.ndarray,
    root_interval: np.ndarray,
    rate: np.ndarray,
    thermal: np.ndarray,
    duration: np.ndarray,
    refined_roots: np.ndarray,
    refined_increments: np.ndarray,
    current_roots: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    valid: np.ndarray,
    right_roots: np.ndarray,
    right_position: np.ndarray,
    right_velocity: np.ndarray,
    right_valid: np.ndarray,
    right_count: np.ndarray,
    right_level: np.ndarray,
    right_index: np.ndarray,
    control: np.ndarray,
) -> None:
    top = control[3]
    level = control[1]
    index = control[2]
    _split_brownian_node_rows_into(
        seed,
        macro_interval,
        particle_id,
        root_interval,
        rate,
        thermal,
        duration,
        level,
        index,
        refined_roots,
        refined_increments,
        refined_roots.size,
        True,
        position,
        velocity,
        valid,
        right_position[top],
        right_velocity[top],
        right_valid[top],
    )
    for local_row in range(refined_roots.size):
        root_row = refined_roots[local_row]
        current_roots[local_row] = root_row
        right_roots[top, local_row] = root_row
    right_count[top] = refined_roots.size
    right_level[top] = level + 1
    right_index[top] = 2 * index + 1
    control[0] = refined_roots.size
    control[1] = level + 1
    control[2] = 2 * index
    control[3] = top + 1


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _pop_brownian_adaptive_into(
    current_roots: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    valid: np.ndarray,
    right_roots: np.ndarray,
    right_position: np.ndarray,
    right_velocity: np.ndarray,
    right_valid: np.ndarray,
    right_count: np.ndarray,
    right_level: np.ndarray,
    right_index: np.ndarray,
    control: np.ndarray,
) -> None:
    if control[3] == 0:
        control[0] = 0
        return
    top = control[3] - 1
    count = right_count[top]
    for local_row in range(count):
        root_row = right_roots[top, local_row]
        current_roots[local_row] = root_row
        position[root_row, 0] = right_position[top, root_row, 0]
        position[root_row, 1] = right_position[top, root_row, 1]
        velocity[root_row, 0] = right_velocity[top, root_row, 0]
        velocity[root_row, 1] = right_velocity[top, root_row, 1]
        valid[root_row] = right_valid[top, root_row]
    control[0] = count
    control[1] = right_level[top]
    control[2] = right_index[top]
    control[3] = top


@dataclass(frozen=True, slots=True)
class _BrownianLeafCursor:
    """One root's numeric base-tree cursor and reusable right-child banks.

    All metadata vectors remain aligned with immutable root particle IDs.  The
    current leaf buffers can be overwritten only after their synchronous tree
    consumer has returned.  The banks and index persist for the whole root.
    """

    seed: np.uint64
    macro_interval: np.uint64
    particle_id: np.ndarray
    root_interval: np.ndarray
    drag_rate_s_inv: np.ndarray
    thermal_velocity_variance_m2_s2: np.ndarray
    root_duration_s: np.ndarray
    tree_depth: int
    next_leaf_index: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    valid: np.ndarray
    right_position_m: np.ndarray
    right_velocity_m_s: np.ndarray
    right_valid: np.ndarray


def _brownian_leaf_increments(
    *,
    seed: int,
    particle_id: np.ndarray,
    macro_interval: int,
    root_interval: np.ndarray,
    drag_rate_s_inv: np.ndarray,
    thermal_velocity_variance_m2_s2: np.ndarray,
    root_duration_s: np.ndarray,
    tree_depth: int,
    numerically_valid: np.ndarray,
) -> _BrownianLeafCursor:
    """Allocate one root cursor; all root and split arithmetic is native."""

    count = particle_id.size
    return _BrownianLeafCursor(
        np.uint64(seed),
        np.uint64(macro_interval),
        particle_id,
        root_interval,
        drag_rate_s_inv,
        thermal_velocity_variance_m2_s2,
        root_duration_s,
        tree_depth,
        np.zeros(1, dtype="<i8"),
        np.empty((count, 2), dtype="<f8"),
        np.empty((count, 2), dtype="<f8"),
        np.asarray(numerically_valid, dtype=np.bool_).copy(),
        np.empty((tree_depth, count, 2), dtype="<f8"),
        np.empty((tree_depth, count, 2), dtype="<f8"),
        np.empty((tree_depth, count), dtype=np.bool_),
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _next_brownian_leaf_into(
    seed: np.uint64,
    macro_interval: np.uint64,
    particle_id: np.ndarray,
    root_interval: np.ndarray,
    rate: np.ndarray,
    thermal: np.ndarray,
    duration: np.ndarray,
    tree_depth: int,
    next_leaf_index: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    valid: np.ndarray,
    right_position: np.ndarray,
    right_velocity: np.ndarray,
    right_valid: np.ndarray,
) -> int:
    """Return the materialized base leaf index, or -1 when the cursor is exhausted.

    The caller retains every bank and the scalar index across yield/migration;
    no random key or physical interval depends on when this function is called.
    """

    leaf_index = next_leaf_index[0]
    if leaf_index >= (1 << tree_depth):
        return -1
    first_level = 0
    if leaf_index == 0:
        _draw_brownian_base_root_into(
            seed,
            macro_interval,
            particle_id,
            root_interval,
            rate,
            thermal,
            duration,
            position,
            velocity,
            valid,
        )
    else:
        remainder = leaf_index
        trailing_zeros = 0
        while (remainder & 1) == 0:
            remainder >>= 1
            trailing_zeros += 1
        ancestor_level = tree_depth - 1 - trailing_zeros
        for row in range(particle_id.size):
            position[row, 0] = right_position[ancestor_level, row, 0]
            position[row, 1] = right_position[ancestor_level, row, 1]
            velocity[row, 0] = right_velocity[ancestor_level, row, 0]
            velocity[row, 1] = right_velocity[ancestor_level, row, 1]
            valid[row] = right_valid[ancestor_level, row]
        first_level = ancestor_level + 1
    for level in range(first_level, tree_depth):
        node_index = leaf_index >> (tree_depth - level)
        _split_brownian_base_node_into(
            seed,
            macro_interval,
            particle_id,
            root_interval,
            rate,
            thermal,
            duration,
            level,
            node_index,
            position,
            velocity,
            valid,
            right_position[level],
            right_velocity[level],
            right_valid[level],
        )
    next_leaf_index[0] = leaf_index + 1
    return leaf_index


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _brownian_base_node_keys(
    seed: np.uint64,
    root_interval: np.uint64,
    level: int,
    node_index: int,
    draw_kind: int,
) -> tuple[np.uint64, np.uint64, np.uint64, np.uint64]:
    """Ask the RNG owner for both component keys, reused for adjacent root rows."""

    key00, key01 = _brownian_node_key(
        seed, root_interval, level, np.uint64(node_index), 0, draw_kind
    )
    key10, key11 = _brownian_node_key(
        seed, root_interval, level, np.uint64(node_index), 1, draw_kind
    )
    return key00, key01, key10, key11


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _draw_brownian_base_root_into(
    seed: np.uint64,
    macro_interval: np.uint64,
    particle_id: np.ndarray,
    root_interval: np.ndarray,
    rate: np.ndarray,
    thermal: np.ndarray,
    duration: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    valid: np.ndarray,
) -> None:
    macro_low = macro_interval & np.uint64(0xFFFFFFFF)
    macro_high = macro_interval >> np.uint64(32)
    previous_root = np.uint64(0)
    keys = (np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0))
    for row in range(particle_id.size):
        root = np.uint64(root_interval[row])
        if row == 0 or root != previous_root:
            keys = _brownian_base_node_keys(seed, root, 0, 0, BROWNIAN_ROOT_NORMAL_STREAM)
            previous_root = root
        scales = (0.0, 0.0, 0.0, OU_STATUS_OK)
        if valid[row]:
            scales = _joint_ou_increment_scales(rate[row], thermal[row], duration[row])
        error = scales[3]
        for component in range(2):
            first, second = _brownian_pair_from_key(
                particle_id[row],
                macro_low,
                macro_high,
                keys[2 * component],
                keys[2 * component + 1],
            )
            x, v = 0.0, 0.0
            if valid[row] and error == OU_STATUS_OK and duration[row] > 0.0 and thermal[row] > 0.0:
                x, v = _joint_ou_increment_component(scales[0], scales[1], scales[2], first, second)
                if not (np.isfinite(x) and np.isfinite(v)):
                    error = PHYSICS_NUMERICAL_FAILURE
            position[row, component] = x
            velocity[row, component] = v
        if error != OU_STATUS_OK:
            valid[row] = False
            position[row, 0], position[row, 1] = 0.0, 0.0
            velocity[row, 0], velocity[row, 1] = 0.0, 0.0


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _split_brownian_base_node_into(
    seed: np.uint64,
    macro_interval: np.uint64,
    particle_id: np.ndarray,
    root_interval: np.ndarray,
    rate: np.ndarray,
    thermal: np.ndarray,
    duration: np.ndarray,
    level: int,
    node_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    valid: np.ndarray,
    right_position: np.ndarray,
    right_velocity: np.ndarray,
    right_valid: np.ndarray,
) -> None:
    _split_brownian_node_rows_into(
        seed,
        macro_interval,
        particle_id,
        root_interval,
        rate,
        thermal,
        duration,
        level,
        node_index,
        root_interval,
        root_interval,
        particle_id.size,
        False,
        position,
        velocity,
        valid,
        right_position,
        right_velocity,
        right_valid,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def _split_brownian_node_rows_into(
    seed: np.uint64,
    macro_interval: np.uint64,
    particle_id: np.ndarray,
    root_interval: np.ndarray,
    rate: np.ndarray,
    thermal: np.ndarray,
    duration: np.ndarray,
    level: int,
    node_index: int,
    root_rows: np.ndarray,
    increment_rows: np.ndarray,
    row_count: int,
    mapped_rows: bool,
    position: np.ndarray,
    velocity: np.ndarray,
    valid: np.ndarray,
    right_position: np.ndarray,
    right_velocity: np.ndarray,
    right_valid: np.ndarray,
) -> None:
    """Condition base or adaptive rows using one numerical/RNG body.

    The left child may overwrite its corresponding parent component.  Right
    outputs are disjoint; all row maps are immutable throughout this call.
    """

    macro_low = macro_interval & np.uint64(0xFFFFFFFF)
    macro_high = macro_interval >> np.uint64(32)
    previous_root = np.uint64(0)
    keys = (np.uint64(0), np.uint64(0), np.uint64(0), np.uint64(0))
    for local_row in range(row_count):
        row = root_rows[local_row] if mapped_rows else local_row
        increment_row = increment_rows[local_row] if mapped_rows else local_row
        root = np.uint64(root_interval[row])
        if local_row == 0 or root != previous_root:
            keys = _brownian_base_node_keys(
                seed, root, level, node_index, BROWNIAN_SPLIT_NORMAL_STREAM
            )
            previous_root = root
        _condition_brownian_row_into(
            row,
            increment_row,
            particle_id[row],
            macro_low,
            macro_high,
            keys,
            rate[row],
            thermal[row],
            duration[row] / (1 << level),
            position,
            velocity,
            valid,
            right_position,
            right_velocity,
            right_valid,
        )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _condition_brownian_row_into(
    row: int,
    increment_row: int,
    particle_id: np.uint64,
    macro_low: np.uint64,
    macro_high: np.uint64,
    keys: tuple[np.uint64, np.uint64, np.uint64, np.uint64],
    rate: float,
    thermal: float,
    duration: float,
    position: np.ndarray,
    velocity: np.ndarray,
    valid: np.ndarray,
    right_position: np.ndarray,
    right_velocity: np.ndarray,
    right_valid: np.ndarray,
) -> None:
    """Condition both components and propagate one row's numerical status."""

    coefficients = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, OU_STATUS_OK)
    active = thermal > 0.0 and duration > 0.0
    parent_valid = valid[increment_row]
    if parent_valid and active:
        coefficients = _half_split_coefficients_scalar(rate, duration)
    error = coefficients[9]
    for component in range(2):
        first, second = _brownian_pair_from_key(
            particle_id, macro_low, macro_high, keys[2 * component], keys[2 * component + 1]
        )
        x_left, v_left, x_right, v_right = 0.0, 0.0, 0.0, 0.0
        if parent_valid and error == OU_STATUS_OK and active:
            x_left, v_left, x_right, v_right = _split_joint_ou_component(
                position[increment_row, component],
                velocity[increment_row, component],
                thermal,
                duration,
                first,
                second,
                coefficients,
            )
            if not _finite_children(x_left, v_left, x_right, v_right):
                error = PHYSICS_NUMERICAL_FAILURE
        position[row, component] = x_left
        velocity[row, component] = v_left
        right_position[row, component] = x_right
        right_velocity[row, component] = v_right
    valid[row] = parent_valid and error == OU_STATUS_OK
    right_valid[row] = valid[row]
    if not valid[row]:
        position[row, 0], position[row, 1] = 0.0, 0.0
        velocity[row, 0], velocity[row, 1] = 0.0, 0.0
        right_position[row, 0], right_position[row, 1] = 0.0, 0.0
        right_velocity[row, 0], right_velocity[row, 1] = 0.0, 0.0


def _advance_joint_ou_particle_local(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    equilibrium_velocity_m_s: np.ndarray,
    drag_rate_s_inv: np.ndarray,
    duration_s: np.ndarray,
    increment: JointOuIncrement,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the stochastic owner's transition once, retaining row-local status."""

    count = int(drag_rate_s_inv.size)
    status = np.empty(count, dtype=np.uint8)
    result_position = np.empty((count, 2), dtype="<f8")
    result_velocity = np.empty((count, 2), dtype="<f8")
    advance_joint_ou_with_increment_into(
        position_m,
        velocity_m_s,
        equilibrium_velocity_m_s,
        drag_rate_s_inv,
        duration_s,
        increment.position_m,
        increment.velocity_m_s,
        result_position,
        result_velocity,
        status,
    )
    valid = status == OU_STATUS_OK
    result_position[~valid] = position_m[~valid]
    result_velocity[~valid] = velocity_m_s[~valid]
    return result_position, result_velocity, valid


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
    writer: _EventSink,
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
    _record_replay_rows(replay, proposal, valid_rows, prepared)
    return None


def _prepare(case: SimulationCase, *, workers: int = 1) -> _PreparedRun:
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
        base_geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
        topology = _prepare_periodic_topology(case, base_geometry)
        geometry, event_geometry = _prepare_contact_views(case, base_geometry, topology)
        schedule = realize_sources(case, geometry)
        if schedule.particle_count != expected_particle_count:
            raise EngineError("source realization changed the preflight particle count")
        _validate_source_facets(schedule, geometry)
        _validate_brownian_surface_departures(schedule, geometry, physics)
        requirements = {
            item.name: RequiredFieldMetadata(
                item.unit,
                item.components,
                item.stored_basis,
                item.positive,
                item.zero_on_rz_axis,
            )
            for item in physics.required_fields
        }
        fields = prepare_required_fields(
            case.data,
            requirements,
            time_interval_s=(spec.time.start_s, spec.time.end_s),
        )
        validate_periodic_field_seams(fields, topology, base_geometry)
        field_time_split_s = _field_time_split_times(spec.time, fields)
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
        TopologyPreparationError,
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
        event_geometry,
        topology,
        fields,
        runtime,
        physics,
        frame_times,
        probe_times,
        probe_particle_index,
        memory_limit_bytes,
        workers,
    )
    stage_capacity = memory_plan.slab_particles * int(physics.requires_stage_evaluation)
    dynamics = _StageDynamics(
        runtime,
        fields,
        case.data.coordinate_system,
        (np.full(schedule.particle_count, -1, dtype="<i8") if field_cell_hint_bytes else None),
        fields.allocate_workspace(stage_capacity),
        PhysicsRuntimeWorkspace.allocate(stage_capacity),
        field_range_workspace=fields.allocate_range_workspace(
            memory_plan.slab_particles * int(memory_plan.certificate_work_bytes_per_particle > 0)
        ),
    )
    dense_certificate = _uses_rk4_dense_certificate(case, physics, constant_acceleration)
    exact_enabled = _uses_exact_specialization(physics, constant_acceleration)
    event_paths = geometry.facet_count > 0 or case.data.coordinate_system == "axisymmetric_rz"
    curved_workspace = _CurvedWavefront.allocate(
        memory_plan.slab_particles,
        spec.solver.event.max_refinements + 1,
        physics.noise is not None,
        exact_enabled=exact_enabled,
        event_paths=event_paths,
    )
    dynamics.prepare_advance_workspaces(
        dense_enabled=dense_certificate,
        exact_enabled=exact_enabled,
    )
    certificate_workspace = _ApplicabilityCertificateWorkspace.allocate(
        memory_plan.slab_particles if dense_certificate else 0,
        spec.solver.event.max_refinements,
    )
    compiled_rules = prepare_boundary_rules(
        rules,
        group_count=(len(case.data.geometry.group_names) if topology is not None else None),
    )
    validate_boundary_rule_frames(
        compiled_rules,
        geometry.group_id,
        geometry.facet_normal,
        roundoff_ulps=spec.solver.event.roundoff_ulps,
    )
    event_row_capacity = memory_plan.slab_particles * int(event_paths)
    boundary_row_capacity = memory_plan.slab_particles * int(bool(geometry.facet_count))
    return _PreparedRun(
        case,
        schedule,
        frame_times,
        probe_times,
        probe_particle_index,
        geometry,
        event_geometry,
        topology,
        rules,
        compiled_rules,
        physics,
        fields,
        field_time_split_s,
        dynamics,
        constant_acceleration,
        maximum_dt_over_tau,
        maximum_dt_charge_lipschitz,
        memory_plan,
        event_candidate_capacity,
        certificate_workspace,
        curved_workspace,
        _ExactWavefront.allocate(
            memory_plan.slab_particles,
            spec.solver.event.max_refinements + 1,
            exact_enabled=exact_enabled,
            event_paths=event_paths,
        ),
        allocate_boundary_response_buffer(boundary_row_capacity),
        prepare_event_geometry(event_geometry),
        allocate_event_arena(
            event_row_capacity,
            event_candidate_capacity * int(event_paths),
            exact_enabled=exact_enabled,
            separate_finite_mask=event_geometry is not geometry,
        ),
        _ProposalReplayWorkspace.allocate(memory_plan.slab_particles, frame_times, probe_times),
    )


def _prepare_memory_plan(
    case: SimulationCase,
    schedule: ParticleSchedule,
    geometry: PreparedGeometry,
    event_geometry: PreparedGeometry,
    topology: PreparedPeriodicTopology | None,
    fields: PreparedFieldSet,
    runtime: PhysicsRuntime,
    physics: PhysicsPlan,
    frame_times_s: tuple[float, ...],
    probe_times_s: tuple[float, ...],
    probe_particle_index: np.ndarray,
    memory_limit_bytes: int,
    workers: int = 1,
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
    dense_certificate = _uses_rk4_dense_certificate(
        case,
        physics,
        runtime.constant_acceleration_m_s2,
    )
    exact_specialization = _uses_exact_specialization(physics, runtime.constant_acceleration_m_s2)
    advance_work_bytes = (
        RK4_ADVANCE_WORK_BYTES_PER_PARTICLE
        if dense_certificate
        else STAGE_ADVANCE_WORK_BYTES_PER_PARTICLE
        if exact_specialization
        else RELAXATION_ADVANCE_WORK_BYTES_PER_PARTICLE
    )
    local_range_certificate = dense_certificate or (
        case.spec.solver.integrator == "exponential_midpoint"
        and physics.requires_stage_evaluation
        and runtime.constant_acceleration_m_s2 is None
    )
    prepared_geometry_bytes = _geometry_memory_bytes(geometry) + _topology_memory_bytes(topology)
    if topology is None and event_geometry is not geometry:
        prepared_geometry_bytes += event_geometry.facet_contact_enabled.nbytes
    particle_schedule_bytes = _schedule_memory_bytes(schedule)
    geometry_preparation_transient_bytes = geometry.volume_bvh_build_transient_nbytes + (
        min(schedule.particle_count, _PREPARE_SCAN_BATCH_SIZE) * np.dtype(np.bool_).itemsize
        if geometry.facet_count
        else 0
    )
    output_buffer_bytes = _output_buffer_bytes(
        schedule,
        frame_times_s,
        probe_times_s,
        probe_particle_index,
    )
    replay_work_bytes = _replay_work_bytes(
        case,
        schedule,
        frame_times_s,
        probe_times_s,
        probe_particle_index,
    )
    stochastic_tree_work_bytes = _stochastic_tree_work_bytes_per_particle(physics)
    event_work_bytes = _event_work_bytes_per_particle(
        case, event_paths=event_paths, curved_enabled=not exact_specialization
    ) + 44 * int(geometry.facet_count > 0)
    certificate_work_bytes = _certificate_work_bytes_per_particle(
        case,
        dense_enabled=dense_certificate,
        local_range_enabled=local_range_certificate,
    )
    release_work_bytes = _surface_release_work_bytes_per_particle(
        schedule,
        event_paths=event_paths,
    )
    candidate_columns = (
        3
        + int(exact_specialization)
        + int(not exact_specialization) * int(event_geometry is not geometry)
    )
    arena_offset_bytes = 16 + 8 * int(exact_specialization) * int(event_paths) * int(
        schedule.particle_count > 0
    )

    def plan_for_capacity(event_candidate_capacity: int) -> CpuMemoryPlan:
        event_staging_capacity = event_candidate_capacity // 2
        return plan_cpu_memory(
            limit_bytes=memory_limit_bytes,
            particle_count=schedule.particle_count,
            canonical_data_bytes=canonical_data_bytes,
            prepared_geometry_bytes=prepared_geometry_bytes,
            particle_schedule_bytes=particle_schedule_bytes,
            physics_runtime_bytes=runtime.bound_array_nbytes,
            field_runtime_bytes=fields.prepared_nbytes + field_cell_hint_bytes,
            geometry_preparation_transient_bytes=geometry_preparation_transient_bytes,
            field_preparation_transient_bytes=fields.preparation_transient_nbytes,
            probe_index_bytes=probe_particle_index.nbytes,
            output_buffer_bytes=output_buffer_bytes,
            replay_work_bytes=replay_work_bytes,
            writer_reserve_bytes=RESULT_WRITER_RESERVE_BYTES,
            requires_stage_evaluation=physics.requires_stage_evaluation,
            # One exclusive fixed arena serves exact or curved queries. Exact
            # reuses its point CSR bank for mixed contact, with four C columns.
            geometry_query_scratch_bytes=(
                candidate_columns * event_candidate_capacity * np.dtype("<i8").itemsize
                + arena_offset_bytes
            ),
            # One reusable native stage bank: int64 accepted hints and
            # uint8 pre-physics status; RZ additionally owns x/v and basis sign.
            stage_composition_work_bytes_per_particle=(
                (
                    9
                    + (40 if case.data.coordinate_system == "axisymmetric_rz" else 0)
                    + 34  # Global velocity/acceleration boxes and certificate verdicts.
                    + advance_work_bytes
                    + (9 if physics.noise is not None else 0)
                    + (
                        16 * int(fields.numeric_context.component_offsets[-1])
                        + 18
                        + (64 if case.data.coordinate_system == "axisymmetric_rz" else 0)
                        if local_range_certificate
                        else 0
                    )
                )
                if physics.requires_stage_evaluation
                else 0
            ),
            proposal_replay_work_bytes_per_particle=(171 if frame_times_s or probe_times_s else 0),
            dense_path_bytes_per_particle=(
                176 + 3 * RK4_DENSE_FULL_ENCLOSURE_BYTES_PER_PARTICLE if dense_certificate else 0
            ),
            stochastic_tree_work_bytes_per_particle=stochastic_tree_work_bytes,
            event_work_bytes_per_particle=event_work_bytes,
            certificate_work_bytes_per_particle=certificate_work_bytes,
            release_work_bytes_per_particle=release_work_bytes,
            event_candidate_capacity=event_candidate_capacity,
            event_staging_capacity=event_staging_capacity,
            event_staging_bytes_per_row=(
                _EVENT_STAGING_BYTES_PER_ROW if event_staging_capacity else 0
            ),
            event_staging_fixed_bytes=(
                event_candidate_capacity * _EVENT_STAGING_BYTES_PER_CANDIDATE
                + _EVENT_STAGING_FIXED_BYTES
                if event_staging_capacity
                else 0
            ),
            failure_staging_bytes_per_particle=_FAILURE_STAGING_BYTES_PER_PARTICLE,
            worker_count=workers,
        )

    event_candidate_capacity, memory_plan = _event_candidate_capacity(
        geometry,
        memory_limit_bytes,
        enabled=event_paths,
        plan_for_capacity=plan_for_capacity,
    )
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
    """Validate the coupled Langevin integrator and noise-model selection."""

    selected = physics.noise is not None
    if (case.spec.solver.integrator == "ou_langevin") != selected:
        raise EngineError(
            "integrator=ou_langevin and physics.noise=inertial_langevin_fdt "
            "must be selected together"
        )
    if not selected:
        return


def _validate_brownian_surface_departures(
    schedule: ParticleSchedule,
    geometry: PreparedGeometry,
    physics: PhysicsPlan,
) -> None:
    """Reject Brownian wall starts whose incoming half-space is undefined."""

    if physics.noise is None:
        return
    rows = np.flatnonzero(schedule.source_facet_id >= 0)
    if not rows.size:
        return
    normals = geometry.facet_normal[schedule.source_facet_id[rows]]
    normal_velocity = np.sum(schedule.velocity_m_s[rows] * normals, axis=1)
    speed = np.linalg.norm(schedule.velocity_m_s[rows], axis=1)
    margin = (
        128.0
        * np.finfo(np.float64).eps
        * np.maximum(
            speed,
            np.finfo(np.float64).tiny,
        )
    )
    if bool((np.abs(normal_velocity) <= margin).any()):
        raise EngineError(
            "Brownian surface release cannot start with zero or tangential normal speed; "
            "provide a realized velocity with an unambiguous finite normal component"
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

    # Predictor support flags certify the frozen accepted start, not its
    # advanced position. Local field ranges cover supported cells only;
    # provisional nearest-cell projection outside support can use another
    # cell. Prove support for the whole box of shortened half predictors
    # before using those ranges as a numerical-path coefficient bound.
    valid &= _exponential_predictor_box_inside_support(
        prepared, predictor_position_lower, predictor_position_upper
    )
    if not bool(valid.any()):
        return fallback

    try:
        local_bound, local_applicable, local_status, range_available = (
            prepared.dynamics.local_additive_acceleration_abs_upper_batch(
                particle_index[selected],
                start_time_s[selected],
                target_time_s[selected],
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


def _exponential_predictor_box_inside_support(
    prepared: _PreparedRun,
    position_lower_m: np.ndarray,
    position_upper_m: np.ndarray,
) -> np.ndarray:
    """Certify every shortened midpoint's field support without event slack.

    The caller already requires a supported start contained in each box.
    Required unstructured fields exactly match the fully supported volume
    mesh, and geometry preparation audits every exterior edge when facets
    exist. A connected box disjoint from all exterior boundaries therefore
    cannot leave that support. The canonical RZ image stays in r>=0, where
    an omitted axis seam is not an exit. An enclosing disk's strict segment
    clearance, strengthened by the existing event budget, proves that box
    disjointness without rejecting a sloping segment's AABB false positive.
    """
    lower, upper = position_lower_m, position_upper_m
    if prepared.support_context.axisymmetric_rz:
        lower, upper = canonicalize_rz_enclosure(lower, upper)
    support_box = prepared.fields.regular_support_box()
    if support_box is not None:
        support_lower, support_upper = support_box
        return (lower >= support_lower).all(axis=1) & (upper <= support_upper).all(axis=1)
    geometry = prepared.event_geometry
    if not geometry.facet_count or not bool(geometry.facet_contact_enabled.all()):
        return np.zeros(lower.shape[0], dtype=np.bool_)
    # The event view retains the complete exterior inventory, including
    # particle-centre and periodic faces disabled in the capsule view.
    center = 0.5 * lower + 0.5 * upper
    component_reach = np.nextafter(
        np.maximum(np.abs(center - lower), np.abs(upper - center)), np.inf
    )
    radius = np.nextafter(np.hypot(component_reach[:, 0], component_reach[:, 1]), np.inf)
    event = prepared.case.spec.solver.event
    budget = resolve_event_budget(
        facet_length_m=float(geometry.facet_length_m.max()),
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=np.max(np.abs(center), axis=0),
        speed_m_s=0.0,
        interval_s=1.0,
        time_s=0.0,
        geometry_rtol=event.geometry_rtol,
        roundoff_ulps=event.roundoff_ulps,
    )
    return centers_respect_contact_radius(
        geometry, center, radius, tolerance_m=budget.position_m, allow_contact=False
    )


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
    use_exact_specialization = _uses_exact_specialization(prepared.physics, constant_acceleration)
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
            eligible_start,
            eligible_target,
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
    interval_start_s: np.ndarray,
    interval_target_s: np.ndarray,
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
                interval_start_s[local_rows],
                interval_target_s[local_rows],
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
                        proposal.start_time_s[selected[local_rows]],
                        proposal.target_time_s[selected[local_rows]],
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

    context = prepared.support_context
    if not context.regular_box:
        if not prepared.geometry.facet_count:
            raise EngineError("curved continuous support certificate is unavailable")
        return np.ones(position_lower_m.shape[0], dtype=np.bool_)
    path_lower = np.asarray(position_lower_m, dtype="<f8")
    path_upper = np.asarray(position_upper_m, dtype="<f8")
    if context.axisymmetric_rz:
        if not bool(np.isfinite(path_lower).all() and np.isfinite(path_upper).all()):
            raise ValueError("lower and upper must contain only finite values")
        if bool((path_lower > path_upper).any()):
            raise ValueError("lower bounds must not exceed upper bounds")
    position_inside = np.empty(path_lower.shape[0], dtype=np.bool_)
    _curved_support_into(
        context, path_lower, path_upper, contact_certified, event_slack_m, position_inside
    )
    return position_inside


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def _curved_support_into(
    context: _NumericSupportContext,
    path_lower: np.ndarray,
    path_upper: np.ndarray,
    contact_certified: np.ndarray,
    event_slack_m: np.ndarray,
    position_inside: np.ndarray,
) -> None:
    """Evaluate the prepared convex support relation without query temporaries."""

    for row in range(path_lower.shape[0]):
        radial_lower, radial_upper = path_lower[row, 0], path_upper[row, 0]
        if context.axisymmetric_rz:
            radial_lower, radial_upper = rz_radial_interval(radial_lower, radial_upper)
        inside = True
        for component in range(2):
            lower = radial_lower if component == 0 else path_lower[row, 1]
            upper = radial_upper if component == 0 else path_upper[row, 1]
            support_lower = context.lower_m[component]
            support_upper = context.upper_m[component]
            if event_slack_m[row] != 0.0:
                support_lower = np.nextafter(support_lower - event_slack_m[row], -np.inf)
                support_upper = np.nextafter(support_upper + event_slack_m[row], np.inf)
            inside = inside and lower >= support_lower and upper <= support_upper
        position_inside[row] = inside or (contact_certified[row] and context.covers_geometry)


def _commit_proposal_rows(
    prepared: _PreparedRun,
    proposal: StepProposal,
    local_rows: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
) -> None:
    rows = np.asarray(local_rows, dtype=np.int64)
    _commit_proposal_rows_into(
        proposal.particle_index,
        rows,
        proposal.end_position_m,
        proposal.end_velocity_m_s,
        proposal.end_charge_number,
        position_m,
        velocity_m_s,
        charge_number,
    )
    _commit_proposal_field_cell(prepared, proposal, rows)


@njit(cache=True, fastmath=False, nogil=True)
def _commit_proposal_rows_into(
    particle_index: np.ndarray,
    rows: np.ndarray,
    end_position_m: np.ndarray,
    end_velocity_m_s: np.ndarray,
    end_charge_number: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
) -> None:
    """Commit accepted state using stable proposal rows without indexed copies."""

    for offset in range(rows.size):
        row = rows[offset]
        particle = particle_index[row]
        for component in range(2):
            position_m[particle, component] = end_position_m[row, component]
            velocity_m_s[particle, component] = end_velocity_m_s[row, component]
        charge_number[particle] = end_charge_number[row]


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
    if prepared.dynamics.last_field_cell is not None:
        _commit_proposal_field_cell_into(
            proposal.particle_index, rows, field_cell_id, prepared.dynamics.last_field_cell
        )


@njit(cache=True, fastmath=False, nogil=True)
def _commit_proposal_field_cell_into(
    particle_index: np.ndarray,
    rows: np.ndarray,
    field_cell_id: np.ndarray,
    accepted_field_cell: np.ndarray,
) -> None:
    for offset in range(rows.size):
        row = rows[offset]
        accepted_field_cell[particle_index[row]] = field_cell_id[row]


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
    writer: _EventSink,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
) -> np.ndarray | None:
    """Advance one slab of reintegrated paths as a flat SoA wavefront."""

    if not _is_reintegrated_path(root_proposal) or root_proposal.path_enclosure is None:
        raise EngineError("curved wavefront requires a reintegrated proposal enclosure")
    state = prepared.curved_workspace.view(root_proposal.particle_index.size)
    _reset_curved_wavefront_into(
        state,
        root_proposal.particle_index,
        root_proposal.start_time_s,
        root_proposal.start_position_m,
        root_proposal.start_velocity_m_s,
        root_proposal.start_charge_number,
        root_proposal.target_time_s,
    )
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
    return state.stochastic_restart_time_s if state.stochastic_restart_time_s.size else None


@njit(cache=True, fastmath=False, nogil=True)
def _reset_curved_wavefront_into(
    state: _CurvedWavefront,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
    target_time_s: np.ndarray,
) -> None:
    """Reset a bounded continuation without retaining another proposal copy."""

    for row in range(particle_index.size):
        state.particle_index[row] = particle_index[row]
        state.current_time_s[row] = start_time_s[row]
        for component in range(2):
            state.current_position_m[row, component] = start_position_m[row, component]
            state.current_velocity_m_s[row, component] = start_velocity_m_s[row, component]
        state.current_charge_number[row] = start_charge_number[row]
        state.root_interval_s[row] = target_time_s[row] - start_time_s[row]
        state.interaction_count[row] = 0
        state.certify_departure[row] = False
        state.stack_top[row] = -1
        if state.stochastic_restart_time_s.size:
            state.stochastic_restart_time_s[row] = np.nan


def _schedule_stochastic_restarts(
    state: _CurvedWavefront,
    rows: np.ndarray,
    time_s: float | np.ndarray,
) -> np.ndarray:
    """End accepted OU intervals so active rows restart from committed state."""

    if not state.stochastic_restart_time_s.size or not rows.size:
        return np.empty(0, dtype="<i8")
    state.stochastic_restart_time_s[rows] = time_s
    state.stack_top[rows] = -1
    return rows


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
    root_count, zero_count = _seed_curved_wavefront_into(
        state,
        proposal.target_time_s,
        interactions,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        start_contact_state,
    )
    return state.root_rows[:root_count], state.zero_rows[:zero_count]


@njit(cache=True, fastmath=False, nogil=True)
def _seed_curved_wavefront_into(
    state: _CurvedWavefront,
    target_time_s: np.ndarray,
    interactions: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    start_contact_state: np.ndarray,
) -> tuple[int, int]:
    """Seed stable rows after surface response, including OU root termination."""

    root_count = 0
    zero_count = 0
    for row in range(state.particle_index.size):
        particle = state.particle_index[row]
        is_active = active[particle]
        response = interactions[row] != 0
        positive = state.current_time_s[row] < target_time_s[row]
        if is_active and response:
            for component in range(2):
                state.current_position_m[row, component] = position_m[particle, component]
                state.current_velocity_m_s[row, component] = velocity_m_s[particle, component]
            state.current_charge_number[row] = charge_number[particle]
        state.interaction_count[row] = interactions[row]
        state.certify_departure[row] = start_contact_state[particle] == SURFACE_STATE_DEPARTURE
        if is_active and response and positive and state.stochastic_restart_time_s.size:
            state.stochastic_restart_time_s[row] = state.current_time_s[row]
        elif is_active and (positive or not response):
            state.stack_top[row] = 0
            state.stack_target_s[row, 0] = target_time_s[row]
            state.stack_depth[row, 0] = 0
            state.stack_interaction_reset[row, 0] = -1
        if is_active and not response:
            if positive:
                state.root_rows[root_count] = row
                root_count += 1
            else:
                state.zero_rows[zero_count] = row
                zero_count += 1
    return root_count, zero_count


def _next_curved_wave_rows(
    particle_index: np.ndarray,
    state: _CurvedWavefront,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pop exhausted work and return one stable row per active particle."""

    count = _next_curved_wave_rows_into(particle_index, state, active)
    return state.wave_rows[:count], state.wave_targets_s[:count]


@njit(cache=True, fastmath=False, nogil=True)
def _next_curved_wave_rows_into(
    particle_index: np.ndarray,
    state: _CurvedWavefront,
    active: np.ndarray,
) -> int:
    """Pop completed left-first continuations without masks or gather arrays."""

    count = 0
    for row in range(particle_index.size):
        if not active[particle_index[row]]:
            continue
        top = state.stack_top[row]
        while top >= 0:
            reset = state.stack_interaction_reset[row, top]
            if reset >= 0:
                state.interaction_count[row] = reset
                state.stack_interaction_reset[row, top] = -1
            target = state.stack_target_s[row, top]
            if state.current_time_s[row] < target:
                state.wave_rows[count] = row
                state.wave_targets_s[count] = target
                count += 1
                break
            top -= 1
        state.stack_top[row] = top
    return count


def _locate_curved_proposal_batches(
    prepared: _PreparedRun,
    proposal: StepProposal,
    proposal_rows: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
    root_interval_s: np.ndarray,
    certify_departure: np.ndarray,
) -> Iterator[tuple[int, int, CurvedEventBatch]]:
    """Locate one curved proposal in stable capacity-bounded row prefixes."""

    if not proposal_rows.size:
        return
    enclosure = proposal.path_enclosure
    if enclosure is None:
        raise EngineError("curved event path lost its continuous enclosure")
    event = prepared.case.spec.solver.event
    chord = _proposal_chord_deviation_bounds(
        proposal,
        proposal_rows,
        start_time_s,
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
        start_time_s,
        target_time_s,
    )
    try:
        prepared_query = prepare_curved_event_batch(
            prepared.event_geometry,
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
            root_interval_s=root_interval_s,
            contact_radius_m=prepared.schedule.contact_radius_m[
                proposal.particle_index[proposal_rows]
            ],
            geometry_rtol=event.geometry_rtol,
            roundoff_ulps=event.roundoff_ulps,
            certify_start_contact_departure=certify_departure,
            chord_deviation_bound_m=chord,
            certify_monotone_approach=proposal.path_kind == "cubic_hermite",
            use_position_controls=use_position_controls,
            position_control_origin_m=position_control_origin_m,
            relative_position_control_lower_m=relative_position_control_lower_m,
            relative_position_control_upper_m=relative_position_control_upper_m,
            finite_contact_enabled=(
                prepared.geometry.facet_contact_enabled
                if prepared.geometry is not prepared.event_geometry
                else None
            ),
        )
    except (EventLocationError, GeometryPreparationError, ValueError) as error:
        first_particle = int(proposal.particle_index[proposal_rows[0]])
        particle_id = int(prepared.schedule.particle_id[first_particle])
        raise EngineError(
            f"curved event batch violated a shared invariant at particle {particle_id}"
        ) from error
    capacity = prepared.event_candidate_capacity
    arena = prepared.event_query_arena
    begin = 0
    while begin < proposal_rows.size:
        end = _bounded_event_row_stop(prepared_query.broad_counts, begin, capacity)
        try:
            query_status, row_count, candidate_count, _ = locate_curved_first_event_into(
                prepared.event_query_geometry,
                prepared_query,
                arena,
                begin,
                end,
            )
            if query_status != EVENT_QUERY_OK:
                raise EventLocationError(
                    "prepared curved candidate arena could not localize prefix"
                )
            result = curved_event_result_view(arena, row_count, candidate_count)
        except (EventLocationError, GeometryPreparationError, ValueError) as error:
            raise EngineError("bounded curved event batch violated its capacity") from error
        result.status[force_split[begin:end]] = CURVED_STATUS_SPLIT
        yield begin, end, result
        begin = end


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
    for begin, end, result in _locate_curved_proposal_batches(
        prepared,
        proposal,
        proposal_rows,
        state.current_time_s[wave_rows],
        target_time_s,
        state.root_interval_s[wave_rows],
        state.certify_departure[wave_rows],
    ):
        yield (
            wave_rows[begin:end],
            proposal_rows[begin:end],
            target_time_s[begin:end],
            result,
        )


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
    _commit_failure_rows(
        state.particle_index,
        wave_rows[invalid],
        proposal_rows[invalid],
        proposal.start_time_s,
        proposal.start_position_m,
        proposal.start_velocity_m_s,
        proposal.start_charge_number,
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
        np.uint16(0),
        codes[invalid],
    )
    valid_offsets = np.flatnonzero(codes == 0).astype("<i8", copy=False)
    if not valid_offsets.size:
        return
    valid_proposal_rows = proposal_rows[valid_offsets]
    _commit_proposal_rows(
        prepared,
        proposal,
        valid_proposal_rows,
        position_m,
        velocity_m_s,
        charge_number,
    )
    _accept_curved_rows_into(
        state,
        valid_offsets,
        proposal_rows,
        wave_rows,
        target_time_s,
        proposal.end_position_m,
        proposal.end_velocity_m_s,
        proposal.end_charge_number,
    )
    _record_replay_rows(
        replay,
        proposal,
        valid_proposal_rows,
        prepared,
    )
    statistics.accepted_particle_pieces += int(valid_offsets.size)


@njit(cache=True, fastmath=False, nogil=True)
def _accept_curved_rows_into(
    state: _CurvedWavefront,
    offsets: np.ndarray,
    proposal_rows: np.ndarray,
    wave_rows: np.ndarray,
    target_time_s: np.ndarray,
    end_position_m: np.ndarray,
    end_velocity_m_s: np.ndarray,
    end_charge_number: np.ndarray,
) -> None:
    """Accept the certified interval and pop its continuation in one native pass."""

    for offset_index in range(offsets.size):
        offset = offsets[offset_index]
        row = wave_rows[offset]
        proposal_row = proposal_rows[offset]
        state.current_time_s[row] = target_time_s[offset]
        for component in range(2):
            state.current_position_m[row, component] = end_position_m[proposal_row, component]
            state.current_velocity_m_s[row, component] = end_velocity_m_s[proposal_row, component]
        state.current_charge_number[row] = end_charge_number[proposal_row]
        state.stack_top[row] -= 1


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
        _commit_failure_rows(
            particle_index,
            prefix_wave_rows[invalid],
            invalid,
            prefix.start_time_s,
            prefix.start_position_m,
            prefix.start_velocity_m_s,
            prefix.start_charge_number,
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
            np.uint16(0),
            codes[invalid],
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
                prepared,
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
        _schedule_stochastic_restarts(
            state,
            np.asarray([row], dtype="<i8"),
            time_s,
        )


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
    material_rows = np.arange(event_rows.size, dtype="<i8")
    if prepared.topology is not None:
        classification = classify_periodic_candidate_rows(
            prepared.topology,
            offsets,
            candidates,
        )
        invalid = np.flatnonzero(classification.kind == TOPOLOGY_CANDIDATE_INVALID).astype(
            "<i8", copy=False
        )
        _commit_curved_topology_failures(
            particle_index,
            rows,
            prefix_rows,
            event_rows,
            invalid,
            event,
            prefix,
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
        periodic_rows = np.flatnonzero(classification.kind == TOPOLOGY_CANDIDATE_PERIODIC).astype(
            "<i8", copy=False
        )
        for classified_value in periodic_rows:
            classified_row = int(classified_value)
            row = int(rows[classified_row])
            event_row = int(event_rows[classified_row])
            prefix_row = int(prefix_rows[classified_row])
            particle = int(particles[classified_row])
            begin = int(offsets[classified_row])
            end = int(offsets[classified_row + 1])
            committed = _commit_curved_periodic_row(
                prepared,
                row,
                particle,
                event_row,
                event,
                tuple(int(value) for value in candidates[begin:end]),
                int(classification.primary_periodic_facet_id[classified_row]),
                prefix.end_velocity_m_s[prefix_row],
                float(prefix.end_charge_number[prefix_row]),
                state,
                position_m,
                velocity_m_s,
                charge_number,
                event_ordinal,
                start_contact_state,
                pending,
                replay,
            )
            if not committed:
                _commit_curved_topology_failures(
                    particle_index,
                    rows,
                    prefix_rows,
                    event_rows,
                    np.asarray([classified_row], dtype="<i8"),
                    event,
                    prefix,
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
        material_rows = np.flatnonzero(classification.kind == TOPOLOGY_CANDIDATE_MATERIAL).astype(
            "<i8", copy=False
        )
    if not material_rows.size:
        return
    event_rows = event_rows[material_rows]
    rows = rows[material_rows]
    prefix_rows = prefix_rows[material_rows]
    particles = particles[material_rows]
    offsets, candidates = _select_event_candidate_rows(event, event_rows)
    contact_radius = prepared.schedule.contact_radius_m[particles]
    contact_normal = contact_normals_for_candidates(
        prepared.geometry,
        prefix.end_position_m[prefix_rows],
        contact_radius,
        offsets,
        candidates,
    )
    law_draws, diffuse_draws, thermal_normal_draws, thermal_tangent_draws = _wall_response_draws(
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
        contact_normal,
        prefix.end_velocity_m_s[prefix_rows],
        prepared.schedule.mass_kg[particles],
        law_draws,
        diffuse_draws,
        thermal_normal_draws,
        thermal_tangent_draws,
        roundoff_ulps=prepared.case.spec.solver.event.roundoff_ulps,
        result_buffer=prepared.boundary_response_buffer,
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


def _commit_curved_topology_failures(
    particle_index: np.ndarray,
    rows: np.ndarray,
    prefix_rows: np.ndarray,
    event_rows: np.ndarray,
    classified_rows: np.ndarray,
    event: CurvedEventBatch,
    prefix: StepProposal,
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
    """Fail only ambiguous or invalid topology rows in one curved wave."""

    proposal_failures: list[tuple[int, _ParticleFailure]] = []
    for classified_value in classified_rows:
        classified_row = int(classified_value)
        row = int(rows[classified_row])
        prefix_row = int(prefix_rows[classified_row])
        event_row = int(event_rows[classified_row])
        proposal_failures.append(
            (
                row,
                _ParticleFailure(
                    _FAILURE_INDETERMINATE_BOUNDARY_POLICY,
                    float(event.time_s[event_row]),
                    event.position_m[event_row],
                    prefix.end_velocity_m_s[prefix_row],
                    float(prefix.end_charge_number[prefix_row]),
                ),
            )
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


def _commit_curved_periodic_row(
    prepared: _PreparedRun,
    row: int,
    particle: int,
    event_row: int,
    event: CurvedEventBatch,
    candidate_facet_ids: tuple[int, ...],
    primary_facet_id: int,
    velocity_m_s_value: np.ndarray,
    charge_number_value: float,
    state: _CurvedWavefront,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    event_ordinal: np.ndarray,
    start_contact_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    replay: _ReplayBuffer,
) -> bool:
    """Commit one curved pure-translation transfer and restart its OU root."""

    topology = prepared.topology
    if topology is None or int(event_ordinal[particle]) >= _MAX_EVENT_ORDINAL:
        raise EngineError("periodic transfer lost topology or exhausted its logical ordinal")
    source_hit = BoundaryHit(
        float(event.time_s[event_row]),
        event.position_m[event_row].copy(),
        primary_facet_id,
        candidate_facet_ids,
        prepared.event_geometry.facet_normal[primary_facet_id].copy(),
        float(event.position_budget_m[event_row]),
        float(event.time_budget_s[event_row]),
        float(event.localization_residual_m[event_row]),
    )
    source_position, source_residual = _canonical_hit_position(
        prepared.event_geometry,
        source_hit,
        primary_facet_id,
        0.0,
    )
    destination_facet = int(topology.peer_facet_id[primary_facet_id])
    destination_hit = BoundaryHit(
        source_hit.time_s,
        source_position + topology.translation_m[primary_facet_id],
        destination_facet,
        (destination_facet,),
        prepared.event_geometry.facet_normal[destination_facet].copy(),
        source_hit.position_budget_m,
        source_hit.time_budget_s,
        source_hit.localization_residual_m,
    )
    destination_position, destination_residual = _canonical_hit_position(
        prepared.event_geometry,
        destination_hit,
        destination_facet,
        0.0,
    )
    radius = float(prepared.schedule.contact_radius_m[particle])
    valid_clearance = centers_respect_contact_radius(
        prepared.geometry,
        destination_position[None, :],
        np.asarray([radius], dtype="<f8"),
        tolerance_m=source_hit.position_budget_m,
        allow_contact=False,
    )
    if not bool(valid_clearance[0]):
        return False
    resolved_hit = BoundaryHit(
        source_hit.time_s,
        source_position,
        primary_facet_id,
        candidate_facet_ids,
        source_hit.normal,
        source_hit.position_budget_m,
        source_hit.time_budget_s,
        max(source_hit.localization_residual_m, source_residual, destination_residual),
    )
    position_m[particle] = destination_position
    velocity_m_s[particle] = velocity_m_s_value
    charge_number[particle] = charge_number_value
    event_ordinal[particle] += np.uint32(1)
    _append_boundary_event(
        pending,
        particle,
        resolved_hit,
        0,
        int(_OUTCOME_TRANSFERRED),
        velocity_m_s_value,
        velocity_m_s_value,
        charge_number_value,
        int(event_ordinal[particle]),
        interaction_code=int(_INTERACTION_PERIODIC),
        destination_facet_id=destination_facet,
        position_post_m=destination_position,
    )
    state.current_time_s[row] = resolved_hit.time_s
    state.current_position_m[row] = destination_position
    state.current_velocity_m_s[row] = velocity_m_s_value
    state.current_charge_number[row] = charge_number_value
    state.interaction_count[row] += 1
    state.certify_departure[row] = True
    start_contact_state[particle] = SURFACE_STATE_DEPARTURE
    prepared.dynamics.invalidate_field_cell(np.asarray([particle], dtype="<i8"))
    _record_replay_jump(
        replay,
        particle,
        resolved_hit.time_s,
        destination_position,
        velocity_m_s_value,
        charge_number_value,
    )
    _schedule_stochastic_restarts(
        state,
        np.asarray([row], dtype="<i8"),
        resolved_hit.time_s,
    )
    return True


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

    if int(event_ordinal[particle]) >= _MAX_EVENT_ORDINAL:
        raise EngineError("logical event ordinal exhausted uint32")
    if int(physical_boundary_event_ordinal[particle]) >= _MAX_EVENT_ORDINAL:
        raise EngineError("physical boundary event ordinal exhausted uint32")
    separation = endpoint_position_m - event.position_m[event_row]
    state_residual = math.hypot(float(separation[0]), float(separation[1]))
    contact_radius = float(prepared.schedule.contact_radius_m[particle])
    event_position = (
        endpoint_position_m.copy() if contact_radius > 0.0 else event.position_m[event_row].copy()
    )
    raw_hit = BoundaryHit(
        float(event.time_s[event_row]),
        event_position,
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
        contact_radius,
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
    _schedule_stochastic_restarts(
        state,
        np.asarray([row], dtype="<i8"),
        resolved_hit.time_s,
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
    exhausted_count, split_count, maximum_depth = _split_curved_wave_rows_into(
        state, rows, maximum, reset_interactions, row_reasons, reason
    )
    exhausted = state.exhausted_rows[:exhausted_count]
    _commit_failure_rows(
        particle_index,
        exhausted,
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
        state.exhausted_reason_code[:exhausted_count],
    )
    statistics.refinements += split_count
    statistics.maximum_refinement_depth = max(
        statistics.maximum_refinement_depth,
        maximum_depth,
    )


@njit(cache=True, fastmath=False, nogil=True)
def _split_curved_wave_rows_into(
    state: _CurvedWavefront,
    rows: np.ndarray,
    maximum: int,
    reset_interactions: bool,
    row_reasons: np.ndarray | None,
    reason: np.uint16,
) -> tuple[int, int, int]:
    """Retain right child and expose left child with the original clock arithmetic."""

    exhausted_count = 0
    split_count = 0
    maximum_depth = 0
    reset_value = 0 if reset_interactions else -1
    for offset in range(rows.size):
        row = rows[offset]
        top = state.stack_top[row]
        depth = state.stack_depth[row, top]
        target = state.stack_target_s[row, top]
        current = state.current_time_s[row]
        midpoint = current + 0.5 * (target - current)
        if depth >= maximum or current >= midpoint or midpoint >= target:
            state.exhausted_rows[exhausted_count] = row
            state.exhausted_reason_code[exhausted_count] = (
                reason if row_reasons is None else row_reasons[offset]
            )
            exhausted_count += 1
            continue
        child_depth = depth + 1
        state.stack_depth[row, top] = child_depth
        state.stack_interaction_reset[row, top] = reset_value
        state.stack_top[row] = top + 1
        state.stack_target_s[row, top + 1] = midpoint
        state.stack_depth[row, top + 1] = child_depth
        state.stack_interaction_reset[row, top + 1] = reset_value
        split_count += 1
        maximum_depth = max(maximum_depth, child_depth)
    return exhausted_count, split_count, maximum_depth


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
        prepared.schedule.contact_radius_m[particle_index],
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
    ordinal_limit = _MAX_EVENT_ORDINAL
    if bool((event_ordinal[particles] >= ordinal_limit).any()):
        raise EngineError("logical event ordinal exhausted uint32")
    if bool((physical_boundary_event_ordinal[particles] >= ordinal_limit).any()):
        raise EngineError("physical boundary event ordinal exhausted uint32")
    offsets = np.arange(rows.size + 1, dtype="<i8")
    candidates = source_facet_id[rows].astype("<i8", copy=False)
    particle_ids = prepared.schedule.particle_id[particles].astype("<u8", copy=False)
    physical_ordinals = physical_boundary_event_ordinal[particles].astype("<u8", copy=False)
    law_draws, diffuse_draws, thermal_normal_draws, thermal_tangent_draws = _wall_response_draws(
        prepared.case.spec.solver.seed,
        particle_ids,
        physical_ordinals,
    )
    velocity_pre = local_velocity_m_s[rows]
    contact_normal = contact_normals_for_candidates(
        prepared.geometry,
        local_position_m[rows],
        prepared.schedule.contact_radius_m[particles],
        offsets,
        candidates,
    )
    responses = resolve_boundary_responses_batch(
        prepared.compiled_boundary_rules,
        offsets,
        candidates,
        prepared.geometry.boundary_id,
        prepared.geometry.group_id,
        contact_normal,
        velocity_pre,
        prepared.schedule.mass_kg[particles],
        law_draws,
        diffuse_draws,
        thermal_normal_draws,
        thermal_tangent_draws,
        roundoff_ulps=prepared.case.spec.solver.event.roundoff_ulps,
        result_buffer=prepared.boundary_response_buffer,
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
                prepared.schedule.contact_radius_m[particles[post_response_rows]],
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


def _post_response_surface_state(
    prepared: _PreparedRun, particle: int, facet_id: int, remains_active: bool
) -> np.uint8:
    """Retain departure proof for a curved path or a finite material origin."""

    if remains_active and (
        _uses_curved_event_path(prepared)
        or (
            prepared.schedule.contact_radius_m[particle] > 0.0
            and prepared.geometry.facet_contact_enabled[facet_id]
        )
    ):
        return SURFACE_STATE_DEPARTURE
    return SURFACE_STATE_RESOLVED


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
        float(prepared.schedule.contact_radius_m[particle]),
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
    surface_release_state[particle] = _post_response_surface_state(
        prepared, particle, facet_id, remains_active
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


def _finite_contact_hit_position(
    geometry: PreparedGeometry,
    hit: BoundaryHit,
    primary_facet_id: int,
    contact_radius_m: float,
) -> tuple[np.ndarray, float]:
    if primary_facet_id not in hit.candidate_facet_ids:
        raise EngineError("primary finite-radius contact is absent from its candidate set")
    residual = 0.0
    for facet_id in hit.candidate_facet_ids:
        start = geometry.facet_start_m[facet_id]
        edge = geometry.facet_end_m[facet_id] - start
        edge_squared = math.fsum((float(edge[0]) ** 2, float(edge[1]) ** 2))
        offset = hit.position_m - start
        parameter = min(
            max(
                math.fsum((float(offset[0]) * float(edge[0]), float(offset[1]) * float(edge[1])))
                / edge_squared,
                0.0,
            ),
            1.0,
        )
        closest = start + parameter * edge
        separation = closest - hit.position_m
        contact_residual = abs(
            math.hypot(float(separation[0]), float(separation[1]))
            - (contact_radius_m if geometry.facet_contact_enabled[facet_id] else 0.0)
        )
        residual = max(residual, contact_residual)
    if not math.isfinite(residual) or residual > hit.position_budget_m:
        raise EngineError("finite-radius boundary state exceeds its localization budget")
    position = hit.position_m.copy()
    position.flags.writeable = False
    return position, residual


def _point_contact_hit_position(
    geometry: PreparedGeometry,
    hit: BoundaryHit,
    primary_facet_id: int,
) -> tuple[np.ndarray, float]:

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


def _canonical_hit_position(
    geometry: PreparedGeometry,
    hit: BoundaryHit,
    primary_facet_id: int,
    contact_radius_m: float,
) -> tuple[np.ndarray, float]:
    """Return a point hit on its wall or a finite-radius centre at first contact."""

    if not math.isfinite(contact_radius_m) or contact_radius_m < 0.0:
        raise EngineError("contact radius must be finite and nonnegative")
    if contact_radius_m > 0.0 and any(
        geometry.facet_contact_enabled[facet] for facet in hit.candidate_facet_ids
    ):
        return _finite_contact_hit_position(
            geometry,
            hit,
            primary_facet_id,
            contact_radius_m,
        )
    return _point_contact_hit_position(geometry, hit, primary_facet_id)


def _prepare_periodic_topology(
    case: SimulationCase,
    geometry: PreparedGeometry,
) -> PreparedPeriodicTopology | None:
    """Resolve YAML group names into one geometry-owned topology map."""

    spec = case.spec.topology
    if spec is None:
        return None
    group_id = {name: index for index, name in enumerate(case.data.geometry.group_names)}
    requests = tuple(
        TranslationPairRequest(
            group_id[pair.first_boundary_group],
            group_id[pair.second_boundary_group],
            pair.first_to_second_m,
        )
        for pair in spec.pairs
    )
    event = case.spec.solver.event
    return prepare_periodic_topology(
        geometry,
        requests,
        field_match_rtol=spec.field_match_rtol,
        geometry_rtol=event.geometry_rtol,
        roundoff_ulps=event.roundoff_ulps,
    )


def _prepare_contact_views(
    case: SimulationCase,
    geometry: PreparedGeometry,
    topology: PreparedPeriodicTopology | None,
) -> tuple[PreparedGeometry, PreparedGeometry]:
    """Resolve group surface modes once and reuse the same mesh/BVH for centres."""

    group_surface = np.ones(len(case.data.geometry.group_names), dtype=np.bool_)
    group_id = {name: index for index, name in enumerate(case.data.geometry.group_names)}
    for rule in case.spec.boundaries:
        group_surface[group_id[rule.boundary_group]] = rule.contact_geometry == "particle_surface"
    surface_contact = group_surface[geometry.group_id]
    if topology is not None:
        surface_contact &= ~topology.facet_is_periodic
    if bool(surface_contact.all()):
        return geometry, geometry
    surface_contact.setflags(write=False)
    return replace(geometry, facet_contact_enabled=surface_contact), geometry


def _prepare_boundary_rules(case: SimulationCase) -> tuple[BoundaryRule, ...]:
    configured = {rule.boundary_group: rule for rule in case.spec.boundaries}
    result = []
    for group_id, name in enumerate(case.data.geometry.group_names):
        rule = configured.get(name)
        if rule is None:
            continue
        result.append(prepare_boundary_rule(group_id, rule.priority, rule.law, rule.parameters))
    return tuple(result)


def _wall_response_draws(
    seed: int,
    particle_id: np.ndarray,
    physical_boundary_event_ordinal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve independent wall-policy streams from physical event identity."""

    return (
        wall_uniform_batch(
            seed,
            particle_id,
            physical_boundary_event_ordinal,
            WALL_PROBABILISTIC_STICK_STREAM,
        ),
        wall_uniform_batch(
            seed,
            particle_id,
            physical_boundary_event_ordinal,
            WALL_MAXWELL_DIFFUSE_STREAM,
        ),
        wall_uniform_open_batch(
            seed,
            particle_id,
            physical_boundary_event_ordinal,
            WALL_MAXWELL_NORMAL_STREAM,
        ),
        wall_standard_normal_batch(
            seed,
            particle_id,
            physical_boundary_event_ordinal,
            WALL_MAXWELL_TANGENTIAL_STREAM,
        ),
    )


def _validate_table_starts(
    case: SimulationCase,
    schedule: ParticleSchedule,
    geometry: PreparedGeometry,
) -> None:
    event = case.spec.solver.event
    geometry_scale = geometry.bbox_diagonal_m
    clearance_tolerance = math.fsum(
        (
            event.geometry_rtol * geometry_scale,
            float(event.roundoff_ulps) * np.finfo(np.float64).eps * geometry_scale,
        )
    )
    for begin in range(0, schedule.particle_count, _PREPARE_SCAN_BATCH_SIZE):
        end = min(begin + _PREPARE_SCAN_BATCH_SIZE, schedule.particle_count)
        positions = schedule.position_m[begin:end]
        radii = schedule.contact_radius_m[begin:end]
        source_facets = schedule.source_facet_id[begin:end]
        finite_rows = np.flatnonzero(radii > 0.0)
        if finite_rows.size:
            nonpenetrating = centers_respect_contact_radius(
                geometry,
                positions[finite_rows],
                radii[finite_rows],
                tolerance_m=clearance_tolerance,
            )
            if not bool(nonpenetrating.all()):
                local = int(finite_rows[int(np.flatnonzero(~nonpenetrating)[0])])
                particle_id = int(schedule.particle_id[begin + local])
                raise EngineError(
                    f"particle {particle_id} contact radius overlaps a material boundary at release"
                )
        if case.data.coordinate_system == "axisymmetric_rz":
            invalid_radial = positions[:, 0] < -clearance_tolerance
            if bool(invalid_radial.any()):
                local = int(np.flatnonzero(invalid_radial)[0])
                particle_id = int(schedule.particle_id[begin + local])
                raise EngineError(
                    f"particle {particle_id} centre has negative R in axisymmetric_rz"
                )
        table_rows = np.flatnonzero((source_facets < 0) & (radii > 0.0))
        if table_rows.size:
            strict = centers_respect_contact_radius(
                geometry,
                positions[table_rows],
                radii[table_rows],
                tolerance_m=clearance_tolerance,
                allow_contact=False,
            )
            if not bool(strict.all()):
                local = int(table_rows[int(np.flatnonzero(~strict)[0])])
                particle_id = int(schedule.particle_id[begin + local])
                raise EngineError(
                    f"table particle {particle_id} contact body must start strictly inside the domain"
                )
        volume_containment = points_inside_volume(geometry, positions)
        table_mask = source_facets < 0
        certified_inside = certify_event_point_interiors(
            geometry,
            positions,
            schedule.velocity_m_s[begin:end],
            schedule.release_time_s[begin:end],
            table_mask,
            volume_containment,
            interval_s=case.spec.time.dt_s,
            geometry_rtol=event.geometry_rtol,
            roundoff_ulps=event.roundoff_ulps,
        )
        for local in np.flatnonzero(table_mask & ~certified_inside):
            index = begin + int(local)
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
    writer: _EventSink,
    pending: _BoundaryEventBuffer,
    failures: _FailureEventBuffer,
) -> None:
    """Advance one slab of exact paths as a flat residual/event wavefront."""

    if root_proposal.path_kind not in {"linear_exact", "quadratic_exact"}:
        raise EngineError("exact residual work requires a linear or quadratic proposal")
    state = prepared.exact_workspace.view(root_proposal.particle_index.size)
    _reset_exact_wavefront_into(
        state,
        root_proposal.particle_index,
        root_proposal.start_time_s,
        root_proposal.start_position_m,
        root_proposal.start_velocity_m_s,
        root_proposal.start_charge_number,
        prepared.constant_acceleration_m_s2,
    )
    current_time = state.current_time_s
    current_position = state.current_position_m
    current_velocity = state.current_velocity_m_s
    current_charge = state.current_charge_number
    stack_target = state.stack_target_s
    stack_depth = state.stack_depth
    stack_interactions = state.stack_interactions
    stack_top = state.stack_top
    departing_facet = state.departing_facet_id
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
        state.zero_rows,
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
    while True:
        wave_count, pending_count = _next_exact_wave_rows_into(
            state, root_proposal.particle_index, active
        )
        if not pending_count:
            break
        if not wave_count:
            continue
        wave_rows = state.wave_rows[:wave_count]
        targets = state.wave_targets_s[:wave_count]
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
            state.acceleration_m_s2,
            departing_facet,
        ):
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
                departing_facet,
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
                state,
            )
            _mark_exact_finite_departures_into(
                root_proposal.particle_index,
                bounded_rows,
                event.status,
                event.time_s,
                event.primary_facet_id,
                active,
                prepared.schedule.contact_radius_m,
                current_time,
                departing_facet,
                prepared.geometry is not prepared.event_geometry,
                prepared.geometry.facet_contact_enabled,
            )
            _flush_boundary_event_wave(writer, prepared, pending)
    return


@njit(cache=True, fastmath=False, nogil=True)
def _reset_exact_wavefront_into(
    state: _ExactWavefront,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    start_charge_number: np.ndarray,
    acceleration_m_s2: np.ndarray | None,
) -> None:
    for row in range(particle_index.size):
        state.current_time_s[row] = start_time_s[row]
        state.current_charge_number[row] = start_charge_number[row]
        state.stack_top[row] = -1
        state.departing_facet_id[row] = EXACT_DEPARTURE_NONE
        for component in range(2):
            state.current_position_m[row, component] = start_position_m[row, component]
            state.current_velocity_m_s[row, component] = start_velocity_m_s[row, component]
            state.acceleration_m_s2[row, component] = (
                0.0
                if acceleration_m_s2 is None
                else acceleration_m_s2[particle_index[row], component]
            )


@njit(cache=True, fastmath=False, nogil=True)
def _next_exact_wave_rows_into(
    state: _ExactWavefront, particle_index: np.ndarray, active: np.ndarray
) -> tuple[int, int]:
    """Preserve each round's single pop and stable row order without gathers."""

    count = 0
    pending_count = 0
    for row in range(particle_index.size):
        top = state.stack_top[row]
        if top < 0 or not active[particle_index[row]]:
            continue
        pending_count += 1
        target = state.stack_target_s[row, top]
        if state.current_time_s[row] >= target:
            state.stack_top[row] = top - 1
        else:
            state.wave_rows[count] = row
            state.wave_targets_s[count] = target
            count += 1
    return count, pending_count


@njit(cache=True, fastmath=False, nogil=True)
def _mark_exact_finite_departures_into(
    particle_index: np.ndarray,
    rows: np.ndarray,
    status: np.ndarray,
    event_time_s: np.ndarray,
    primary_facet_id: np.ndarray,
    active: np.ndarray,
    contact_radius_m: np.ndarray,
    current_time_s: np.ndarray,
    departing_facet_id: np.ndarray,
    separate_finite_mask: bool,
    finite_contact_enabled: np.ndarray,
) -> None:
    for local in range(rows.size):
        row = rows[local]
        particle = particle_index[row]
        if (
            status[local] != EXACT_STATUS_WALL
            or not active[particle]
            or contact_radius_m[particle] <= 0.0
            or current_time_s[row] != event_time_s[local]
        ):
            continue
        primary = primary_facet_id[local]
        if separate_finite_mask and (primary < 0 or not finite_contact_enabled[primary]):
            continue
        departing_facet_id[row] = EXACT_DEPARTURE_FINITE_CONTACT_SET


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
    zero_rows: np.ndarray,
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
    zero_count = _seed_exact_wavefront_into(
        proposal.particle_index,
        proposal.start_time_s,
        proposal.target_time_s,
        interactions,
        departing,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        position_m,
        velocity_m_s,
        charge_number,
        active,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
        stack_target_s,
        stack_depth,
        stack_interactions,
        stack_top,
        departing_facet_id,
        zero_rows,
    )
    for row_value in zero_rows[:zero_count]:
        row = int(row_value)
        _record_replay_jump(
            replay,
            int(proposal.particle_index[row]),
            float(proposal.start_time_s[row]),
            current_position_m[row],
            current_velocity_m_s[row],
            float(current_charge_number[row]),
        )


@njit(cache=True, fastmath=False, nogil=True)
def _seed_exact_wavefront_into(
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
    interactions: np.ndarray,
    departing: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    active: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    stack_target_s: np.ndarray,
    stack_depth: np.ndarray,
    stack_interactions: np.ndarray,
    stack_top: np.ndarray,
    departing_facet_id: np.ndarray,
    zero_rows: np.ndarray,
) -> int:
    zero_count = 0
    for row in range(particle_index.size):
        particle = particle_index[row]
        if not active[particle]:
            continue
        if interactions[row] != 0:
            current_charge_number[row] = charge_number[particle]
            exact_origin_time_s[particle] = start_time_s[row]
            for component in range(2):
                current_position_m[row, component] = position_m[particle, component]
                current_velocity_m_s[row, component] = velocity_m_s[particle, component]
                exact_origin_position_m[particle, component] = current_position_m[row, component]
                exact_origin_velocity_m_s[particle, component] = current_velocity_m_s[
                    row, component
                ]
        if start_time_s[row] < target_time_s[row]:
            stack_top[row] = 0
            stack_target_s[row, 0] = target_time_s[row]
            stack_depth[row, 0] = 0
            stack_interactions[row, 0] = interactions[row]
            departing_facet_id[row] = departing[row]
        else:
            zero_rows[zero_count] = row
            zero_count += 1
    return zero_count


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
        prepared_query = prepare_exact_event_batch(
            prepared.event_geometry,
            path_kind,
            current_position_m[wave_rows],
            current_velocity_m_s[wave_rows],
            acceleration_m_s2[wave_rows],
            start_time_s=current_time_s[wave_rows],
            target_time_s=target_time_s,
            contact_radius_m=prepared.schedule.contact_radius_m[particle_index[wave_rows]],
            geometry_rtol=event.geometry_rtol,
            roundoff_ulps=event.roundoff_ulps,
            certified_departing_facet_id=departing_facet_id[wave_rows],
            finite_contact_enabled=(
                prepared.geometry.facet_contact_enabled
                if prepared.geometry is not prepared.event_geometry
                else None
            ),
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
        end = _bounded_event_row_stop(prepared_query.broad_counts, begin, capacity)
        bounded_rows = wave_rows[begin:end]
        bounded_targets = target_time_s[begin:end]
        try:
            status, row_count, candidate_count, _ = locate_exact_first_event_into(
                prepared.event_query_geometry,
                prepared_query,
                prepared.event_query_arena,
                begin,
                end,
            )
            if status != EVENT_QUERY_OK:
                raise EventLocationError("bounded exact arena could not retain the full query")
            result = exact_event_result_view(prepared.event_query_arena, row_count, candidate_count)
        except (EventLocationError, GeometryPreparationError, ValueError) as error:
            raise EngineError("bounded exact event batch violated its capacity") from error
        yield bounded_rows, bounded_targets, result
        begin = end


@njit(cache=True, fastmath=False, nogil=True)
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
    departing_facet_id: np.ndarray,
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
    state: _ExactWavefront,
) -> None:
    failed_count, split_count, clear_count, axis_count, wall_count = _select_exact_event_rows_into(
        state,
        wave_rows,
        target_time_s,
        event.status,
        prepared.case.spec.solver.event.max_interactions_per_step,
    )
    failed_rows = state.failed_rows[:failed_count]
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
    split_rows = state.split_rows[:split_count]
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
        state.exhausted_rows,
    )
    clear_rows = state.clear_rows[:clear_count]
    _commit_exact_clear_rows(
        prepared,
        particle_index,
        clear_rows,
        state.clear_targets_s[:clear_count],
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
    axis_wave_rows = state.axis_event_rows[:axis_count]
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
    wall_wave_rows = state.wall_event_rows[:wall_count]
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
        departing_facet_id,
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


@njit(cache=True, fastmath=False, nogil=True)
def _select_exact_event_rows_into(
    state: _ExactWavefront,
    wave_rows: np.ndarray,
    target_time_s: np.ndarray,
    status: np.ndarray,
    maximum_interactions: int,
) -> tuple[int, int, int, int, int]:
    """Stable exclusive work maps; borrowed query/wave columns stay untouched."""

    failed = split = clear = axis = wall = 0
    for local in range(wave_rows.size):
        row = wave_rows[local]
        code = status[local]
        needs_split = (
            code == EXACT_STATUS_WALL
            and state.stack_interactions[row, max(state.stack_top[row], 0)] >= maximum_interactions
        )
        if not needs_split:
            state.departing_facet_id[row] = EXACT_DEPARTURE_NONE
        if code == EXACT_STATUS_FAILURE:
            state.failed_rows[failed] = row
            failed += 1
        elif needs_split:
            state.split_rows[split] = row
            split += 1
        elif code == EXACT_STATUS_CLEAR:
            state.clear_rows[clear] = row
            state.clear_targets_s[clear] = target_time_s[local]
            clear += 1
        elif code == EXACT_STATUS_AXIS:
            state.axis_event_rows[axis] = local
            axis += 1
        elif code == EXACT_STATUS_WALL:
            state.wall_event_rows[wall] = local
            wall += 1
    return failed, split, clear, axis, wall


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
    """Evaluate exact targets and borrow their independently owned end states."""

    count = selected_rows.size
    failures: list[tuple[int, _ParticleFailure]] = []
    if not count:
        return (
            _ExactEndpointBatch(
                np.zeros(0, dtype=np.bool_),
                np.empty((0, 2), dtype="<f8"),
                np.empty((0, 2), dtype="<f8"),
                np.empty(0, dtype="<f8"),
            ),
            failures,
        )
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
    codes = _proposal_numerical_failure_codes(proposal.numerical_status)
    codes[(codes == 0) & ~proposal.support_inside] = _FAILURE_FIELD_SUPPORT
    codes[(codes == 0) & ~proposal.applicability_inside] = _FAILURE_MODEL_APPLICABILITY
    failures.extend(
        (
            int(selected_rows[local_row]),
            _proposal_row_failure(proposal, int(local_row), codes[local_row]),
        )
        for local_row in np.flatnonzero(codes)
    )
    valid = codes == 0
    valid_local = np.flatnonzero(valid).astype("<i8", copy=False)
    if valid_local.size:
        _commit_proposal_field_cell(prepared, proposal, valid_local)
        _record_replay_rows(replay, proposal, valid_local, prepared)
        statistics.accepted_particle_pieces += int(valid_local.size)
    return (
        _ExactEndpointBatch(
            valid,
            proposal.end_position_m,
            proposal.end_velocity_m_s,
            proposal.end_charge_number,
        ),
        failures,
    )


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
    _commit_exact_clear_into(
        particle_index,
        rows,
        target_time_s,
        endpoint,
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        stack_top,
        position_m,
        velocity_m_s,
        charge_number,
    )


@njit(cache=True, fastmath=False, nogil=True)
def _commit_exact_clear_into(
    particle_index: np.ndarray,
    rows: np.ndarray,
    target_time_s: np.ndarray,
    endpoint: _ExactEndpointBatch,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
) -> None:
    for selected in range(rows.size):
        if not endpoint.valid[selected]:
            continue
        row = rows[selected]
        particle = particle_index[row]
        current_time_s[row] = target_time_s[selected]
        current_charge_number[row] = endpoint.charge_number[selected]
        charge_number[particle] = endpoint.charge_number[selected]
        for component in range(2):
            current_position_m[row, component] = endpoint.position_m[selected, component]
            current_velocity_m_s[row, component] = endpoint.velocity_m_s[selected, component]
            position_m[particle, component] = endpoint.position_m[selected, component]
            velocity_m_s[particle, component] = endpoint.velocity_m_s[selected, component]
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
        current_position_m[rows],
        current_velocity_m_s[rows],
        current_charge_number[rows],
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
    statistics.axis_crossings += _commit_exact_axis_into(
        particle_index,
        rows,
        targets,
        endpoint,
        current_time_s,
        current_position_m,
        current_velocity_m_s,
        current_charge_number,
        position_m,
        velocity_m_s,
        charge_number,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
        replay,
    )


@njit(cache=True, fastmath=False, nogil=True)
def _commit_exact_axis_into(
    particle_index: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    endpoint: _ExactEndpointBatch,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    replay: _ReplayBuffer,
) -> int:
    committed = 0
    for selected in range(rows.size):
        if not endpoint.valid[selected]:
            continue
        row = rows[selected]
        particle = particle_index[row]
        endpoint.position_m[selected, 0] = 0.0
        fold_rz_position_vector_into(
            endpoint.position_m[selected : selected + 1],
            endpoint.velocity_m_s[selected : selected + 1],
        )
        current_time_s[row] = targets[selected]
        current_charge_number[row] = endpoint.charge_number[selected]
        charge_number[particle] = endpoint.charge_number[selected]
        for component in range(2):
            current_position_m[row, component] = endpoint.position_m[selected, component]
            current_velocity_m_s[row, component] = endpoint.velocity_m_s[selected, component]
            position_m[particle, component] = endpoint.position_m[selected, component]
            velocity_m_s[particle, component] = endpoint.velocity_m_s[selected, component]
        _set_exact_origin(
            particle,
            targets[selected],
            endpoint.position_m[selected],
            endpoint.velocity_m_s[selected],
            exact_origin_time_s,
            exact_origin_position_m,
            exact_origin_velocity_m_s,
        )
        _record_replay_jump(
            replay,
            particle,
            targets[selected],
            endpoint.position_m[selected],
            endpoint.velocity_m_s[selected],
            endpoint.charge_number[selected],
        )
        committed += 1
    return committed


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
    exhausted_rows: np.ndarray,
) -> None:
    exhausted_count, split_count = _split_exact_wave_rows_into(
        rows,
        current_time_s,
        stack_target_s,
        stack_depth,
        stack_interactions,
        stack_top,
        prepared.case.spec.solver.event.max_refinements,
        exhausted_rows,
    )
    _fail_event_rows(
        particle_index,
        exhausted_rows[:exhausted_count],
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
    statistics.residual_splits += split_count


@njit(cache=True, fastmath=False, nogil=True)
def _split_exact_wave_rows_into(
    rows: np.ndarray,
    current_time_s: np.ndarray,
    stack_target_s: np.ndarray,
    stack_depth: np.ndarray,
    stack_interactions: np.ndarray,
    stack_top: np.ndarray,
    maximum_depth: int,
    exhausted_rows: np.ndarray,
) -> tuple[int, int]:
    exhausted_count = 0
    split_count = 0
    for local in range(rows.size):
        row = rows[local]
        top = stack_top[row]
        depth = stack_depth[row, top]
        target = stack_target_s[row, top]
        midpoint = current_time_s[row] + 0.5 * (target - current_time_s[row])
        if depth >= maximum_depth or not current_time_s[row] < midpoint < target:
            exhausted_rows[exhausted_count] = row
            exhausted_count += 1
            continue
        child_depth = depth + 1
        stack_depth[row, top] = child_depth
        stack_interactions[row, top] = 0
        stack_top[row] = top + 1
        stack_target_s[row, top + 1] = midpoint
        stack_depth[row, top + 1] = child_depth
        stack_interactions[row, top + 1] = 0
        split_count += 1
    return exhausted_count, split_count


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
    _commit_failure_rows(
        particle_index,
        rows,
        rows,
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
        reason_code,
    )


def _commit_failure_rows(
    particle_index: np.ndarray,
    rows: np.ndarray,
    source_rows: np.ndarray,
    time_s: np.ndarray,
    source_position_m: np.ndarray,
    source_velocity_m_s: np.ndarray,
    source_charge_number: np.ndarray,
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
    row_reasons: np.ndarray | None = None,
) -> None:
    """Commit aligned wave/proposal failures without Python per-row objects."""

    if not rows.size:
        return
    context = _FailureCommitContext(
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failures.particle_index,
        failures.time_s,
        failures.event_ordinal,
        failures.reason_code,
    )
    count, status = _commit_failure_rows_into(
        context,
        failures.count,
        particle_index,
        rows,
        source_rows,
        time_s,
        source_position_m,
        source_velocity_m_s,
        source_charge_number,
        stack_top,
        reason_code,
        row_reasons,
    )
    if status == 1:
        raise EngineError("logical event ordinal exhausted uint32")
    if status == 2:
        raise EngineError("localized particle failure contains non-finite state")
    if status == 3:
        raise EngineError("failure staging capacity exhausted")
    failures.count = count


@njit(cache=True, fastmath=False, nogil=True)
def _commit_failure_rows_into(
    context: _FailureCommitContext,
    count: int,
    particle_index: np.ndarray,
    rows: np.ndarray,
    source_rows: np.ndarray,
    time_s: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    stack_top: np.ndarray,
    reason_code: np.uint16,
    row_reasons: np.ndarray | None,
) -> tuple[int, int]:
    """Check run integrity before committing the stable row-local failures."""

    if count + rows.size > context.failure_particle_index.size:
        return count, 3
    for offset in range(rows.size):
        particle = particle_index[rows[offset]]
        source = source_rows[offset]
        if context.event_ordinal[particle] >= _MAX_EVENT_ORDINAL:
            return count, 1
        if not _finite_failure_state(
            time_s[source], position_m[source], velocity_m_s[source], charge_number[source]
        ):
            return count, 2
    for offset in range(rows.size):
        row = rows[offset]
        source = source_rows[offset]
        reason = reason_code if row_reasons is None else row_reasons[offset]
        _write_particle_failure_into(
            context,
            count + offset,
            particle_index[row],
            time_s[source],
            position_m[source],
            velocity_m_s[source],
            charge_number[source],
            reason,
        )
        stack_top[row] = -1
    return count + rows.size, 0


@njit(cache=True, fastmath=False, nogil=True)
def _finite_failure_state(
    time_s: float, position_m: np.ndarray, velocity_m_s: np.ndarray, charge_number: float
) -> bool:
    return (
        math.isfinite(time_s)
        and math.isfinite(position_m[0])
        and math.isfinite(position_m[1])
        and math.isfinite(velocity_m_s[0])
        and math.isfinite(velocity_m_s[1])
        and math.isfinite(charge_number)
    )


@njit(cache=True, fastmath=False, nogil=True)
def _write_particle_failure_into(
    context: _FailureCommitContext,
    offset: int,
    particle: int,
    time_s: float,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: float,
    reason_code: np.uint16,
) -> None:
    """One accepted failure commit, shared by scalar and wave consumers."""

    for component in range(2):
        context.position_m[particle, component] = position_m[component]
        context.velocity_m_s[particle, component] = velocity_m_s[component]
    context.charge_number[particle] = charge_number
    context.active[particle] = False
    context.lifecycle[particle] = _LIFECYCLE_FAILED
    context.terminal_time_s[particle] = time_s
    context.failure_reason_code[particle] = reason_code
    context.event_ordinal[particle] += np.uint32(1)
    context.failure_particle_index[offset] = particle
    context.failure_time_s[offset] = time_s
    context.failure_event_ordinal[offset] = context.event_ordinal[particle]
    context.failure_reason[offset] = reason_code


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
    departing_facet_id: np.ndarray,
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
    material_rows = np.arange(valid.size, dtype="<i8")
    if prepared.topology is not None:
        classification = classify_periodic_candidate_rows(
            prepared.topology,
            offsets,
            candidates,
        )
        invalid = np.flatnonzero(classification.kind == TOPOLOGY_CANDIDATE_INVALID).astype(
            "<i8", copy=False
        )
        _commit_exact_topology_failures(
            particle_index,
            rows,
            wall_wave_rows,
            valid,
            invalid,
            event,
            endpoint,
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
        periodic_rows = np.flatnonzero(classification.kind == TOPOLOGY_CANDIDATE_PERIODIC).astype(
            "<i8", copy=False
        )
        for classified_value in periodic_rows:
            classified_row = int(classified_value)
            endpoint_row = int(valid[classified_row])
            row = int(rows[endpoint_row])
            event_row = int(wall_wave_rows[endpoint_row])
            begin = int(offsets[classified_row])
            end = int(offsets[classified_row + 1])
            committed = _commit_exact_periodic_response(
                prepared,
                row,
                int(particle_index[row]),
                event_row,
                event,
                tuple(int(value) for value in candidates[begin:end]),
                int(classification.primary_periodic_facet_id[classified_row]),
                endpoint.velocity_m_s[endpoint_row],
                float(endpoint.charge_number[endpoint_row]),
                current_time_s,
                current_position_m,
                current_velocity_m_s,
                current_charge_number,
                departing_facet_id,
                stack_interactions,
                stack_top,
                position_m,
                velocity_m_s,
                charge_number,
                event_ordinal,
                exact_origin_time_s,
                exact_origin_position_m,
                exact_origin_velocity_m_s,
                surface_release_state,
                pending,
                replay,
            )
            if not committed:
                _commit_exact_topology_failures(
                    particle_index,
                    rows,
                    wall_wave_rows,
                    valid,
                    np.asarray([classified_row], dtype="<i8"),
                    event,
                    endpoint,
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
        material_rows = np.flatnonzero(classification.kind == TOPOLOGY_CANDIDATE_MATERIAL).astype(
            "<i8", copy=False
        )
    if not material_rows.size:
        return
    material_valid = valid[material_rows]
    selected_wave_rows = wall_wave_rows[material_valid]
    offsets, candidates = _select_event_candidate_rows(event, selected_wave_rows)
    selected_rows = rows[material_valid]
    particles = particle_index[selected_rows]
    contact_radius = prepared.schedule.contact_radius_m[particles]
    contact_normal = contact_normals_for_candidates(
        prepared.geometry,
        event.position_m[selected_wave_rows],
        contact_radius,
        offsets,
        candidates,
    )
    law_draws, diffuse_draws, thermal_normal_draws, thermal_tangent_draws = _wall_response_draws(
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
        contact_normal,
        endpoint.velocity_m_s[material_valid],
        prepared.schedule.mass_kg[particles],
        law_draws,
        diffuse_draws,
        thermal_normal_draws,
        thermal_tangent_draws,
        roundoff_ulps=prepared.case.spec.solver.event.roundoff_ulps,
        result_buffer=prepared.boundary_response_buffer,
    )
    for response_row, selected_value in enumerate(material_valid):
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


def _commit_exact_topology_failures(
    particle_index: np.ndarray,
    rows: np.ndarray,
    wall_wave_rows: np.ndarray,
    valid: np.ndarray,
    classified_rows: np.ndarray,
    event: ExactEventBatch,
    endpoint: _ExactEndpointBatch,
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
    """Fail only ambiguous or invalid topology rows in one exact wave."""

    proposal_failures: list[tuple[int, _ParticleFailure]] = []
    for classified_value in classified_rows:
        endpoint_row = int(valid[int(classified_value)])
        row = int(rows[endpoint_row])
        event_row = int(wall_wave_rows[endpoint_row])
        proposal_failures.append(
            (
                row,
                _ParticleFailure(
                    _FAILURE_INDETERMINATE_BOUNDARY_POLICY,
                    float(event.time_s[event_row]),
                    event.position_m[event_row],
                    endpoint.velocity_m_s[endpoint_row],
                    float(endpoint.charge_number[endpoint_row]),
                ),
            )
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


def _commit_exact_periodic_response(
    prepared: _PreparedRun,
    row: int,
    particle: int,
    event_row: int,
    event: ExactEventBatch,
    candidate_facet_ids: tuple[int, ...],
    primary_facet_id: int,
    velocity_m_s_value: np.ndarray,
    charge_number_value: float,
    current_time_s: np.ndarray,
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    current_charge_number: np.ndarray,
    departing_facet_id: np.ndarray,
    stack_interactions: np.ndarray,
    stack_top: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    event_ordinal: np.ndarray,
    exact_origin_time_s: np.ndarray,
    exact_origin_position_m: np.ndarray,
    exact_origin_velocity_m_s: np.ndarray,
    surface_release_state: np.ndarray,
    pending: _BoundaryEventBuffer,
    replay: _ReplayBuffer,
) -> bool:
    """Commit one exact pure-translation transfer without wall RNG identity."""

    topology = prepared.topology
    if topology is None or int(event_ordinal[particle]) >= _MAX_EVENT_ORDINAL:
        raise EngineError("periodic transfer lost topology or exhausted its logical ordinal")
    source_hit = BoundaryHit(
        float(event.time_s[event_row]),
        event.position_m[event_row].copy(),
        primary_facet_id,
        candidate_facet_ids,
        prepared.event_geometry.facet_normal[primary_facet_id].copy(),
        float(event.position_budget_m[event_row]),
        float(event.time_budget_s[event_row]),
        float(event.localization_residual_m[event_row]),
    )
    source_position, source_residual = _canonical_hit_position(
        prepared.event_geometry,
        source_hit,
        primary_facet_id,
        0.0,
    )
    destination_facet = int(topology.peer_facet_id[primary_facet_id])
    translated = source_position + topology.translation_m[primary_facet_id]
    destination_hit = BoundaryHit(
        source_hit.time_s,
        translated,
        destination_facet,
        (destination_facet,),
        prepared.event_geometry.facet_normal[destination_facet].copy(),
        source_hit.position_budget_m,
        source_hit.time_budget_s,
        source_hit.localization_residual_m,
    )
    destination_position, destination_residual = _canonical_hit_position(
        prepared.event_geometry,
        destination_hit,
        destination_facet,
        0.0,
    )
    radius = float(prepared.schedule.contact_radius_m[particle])
    valid_clearance = centers_respect_contact_radius(
        prepared.geometry,
        destination_position[None, :],
        np.asarray([radius], dtype="<f8"),
        tolerance_m=source_hit.position_budget_m,
        allow_contact=False,
    )
    if not bool(valid_clearance[0]):
        return False
    resolved_hit = BoundaryHit(
        source_hit.time_s,
        source_position,
        primary_facet_id,
        candidate_facet_ids,
        source_hit.normal,
        source_hit.position_budget_m,
        source_hit.time_budget_s,
        max(source_hit.localization_residual_m, source_residual, destination_residual),
    )
    position_m[particle] = destination_position
    velocity_m_s[particle] = velocity_m_s_value
    charge_number[particle] = charge_number_value
    event_ordinal[particle] += np.uint32(1)
    _append_boundary_event(
        pending,
        particle,
        resolved_hit,
        0,
        int(_OUTCOME_TRANSFERRED),
        velocity_m_s_value,
        velocity_m_s_value,
        charge_number_value,
        int(event_ordinal[particle]),
        interaction_code=int(_INTERACTION_PERIODIC),
        destination_facet_id=destination_facet,
        position_post_m=destination_position,
    )
    current_time_s[row] = resolved_hit.time_s
    current_position_m[row] = destination_position
    current_velocity_m_s[row] = velocity_m_s_value
    current_charge_number[row] = charge_number_value
    top = int(stack_top[row])
    stack_interactions[row, top] += 1
    departing_facet_id[row] = destination_facet
    surface_release_state[particle] = SURFACE_STATE_RESOLVED
    prepared.dynamics.invalidate_field_cell(np.asarray([particle], dtype="<i8"))
    _set_exact_origin(
        particle,
        resolved_hit.time_s,
        destination_position,
        velocity_m_s_value,
        exact_origin_time_s,
        exact_origin_position_m,
        exact_origin_velocity_m_s,
    )
    _record_replay_jump(
        replay,
        particle,
        resolved_hit.time_s,
        destination_position,
        velocity_m_s_value,
        charge_number_value,
    )
    return True


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
    if int(event_ordinal[particle_index]) >= _MAX_EVENT_ORDINAL:
        raise EngineError("logical event ordinal exhausted uint32")
    if int(physical_boundary_event_ordinal[particle_index]) >= _MAX_EVENT_ORDINAL:
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
        float(prepared.schedule.contact_radius_m[particle_index]),
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
    surface_release_state[particle_index] = _post_response_surface_state(
        prepared, particle_index, primary_facet_id, remains_active
    )
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


@njit(cache=True, fastmath=False, nogil=True)
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
            np.empty(0, dtype="<u1"),
            np.empty(0, dtype="<i8"),
            np.empty((0, 2), dtype="<f8"),
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
        np.empty(row_capacity, dtype="<u1"),
        np.empty(row_capacity, dtype="<i8"),
        np.empty((row_capacity, 2), dtype="<f8"),
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
    *,
    interaction_code: int = int(_INTERACTION_WALL),
    destination_facet_id: int = -1,
    position_post_m: np.ndarray | None = None,
) -> None:
    """Append one event to the shared row/candidate capacity arena."""

    row = pending.row_count
    candidates = hit.candidate_facet_ids
    candidate_end = pending.candidate_count + len(candidates)
    if row >= pending.particle_index.size:
        raise EngineError("boundary-event row buffer exhausted")
    if row + 1 + candidate_end > pending.arena_capacity:
        raise EngineError("boundary-event candidate arena exhausted")
    post = _canonical_boundary_event_post(
        hit,
        interaction_code,
        law_code,
        outcome_code,
        destination_facet_id,
        position_post_m,
    )
    pending.particle_index[row] = particle_index
    pending.time_s[row] = hit.time_s
    pending.event_ordinal[row] = event_ordinal
    pending.primary_facet_id[row] = hit.facet_id
    pending.interaction_code[row] = interaction_code
    pending.destination_facet_id[row] = destination_facet_id
    pending.position_m[row] = hit.position_m
    pending.position_post_m[row] = post
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


def _canonical_boundary_event_post(
    hit: BoundaryHit,
    interaction_code: int,
    law_code: int,
    outcome_code: int,
    destination_facet_id: int,
    position_post_m: np.ndarray | None,
) -> np.ndarray:
    """Validate the two canonical event encodings and return post-position."""

    post = hit.position_m if position_post_m is None else position_post_m
    if interaction_code == int(_INTERACTION_WALL):
        if (
            not (0 < law_code < _BOUNDARY_LAW_OUTPUT.size)
            or not (0 < outcome_code < int(_OUTCOME_TRANSFERRED))
            or destination_facet_id != -1
            or not bool(np.array_equal(post, hit.position_m))
        ):
            raise EngineError("wall event has an invalid canonical encoding")
        return post
    if interaction_code == int(_INTERACTION_PERIODIC):
        if (
            law_code != 0
            or outcome_code != int(_OUTCOME_TRANSFERRED)
            or destination_facet_id < 0
            or not bool(np.isfinite(post).all())
        ):
            raise EngineError("periodic event has an invalid canonical encoding")
        return post
    raise EngineError("boundary-event interaction code is unknown")


def _flush_boundary_event_wave(
    writer: _EventSink,
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
    writer: _EventSink,
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

    if int(event_ordinal[particle_index]) >= _MAX_EVENT_ORDINAL:
        raise EngineError("logical event ordinal exhausted uint32")
    if not _finite_failure_state(
        failure.time_s, failure.position_m, failure.velocity_m_s, failure.charge_number
    ):
        raise EngineError("localized particle failure contains non-finite state")
    if failures.count >= failures.particle_index.size:
        raise EngineError("failure-event buffer exceeded one row per slab particle")
    context = _FailureCommitContext(
        position_m,
        velocity_m_s,
        charge_number,
        active,
        lifecycle,
        terminal_time_s,
        event_ordinal,
        failure_reason_code,
        failures.particle_index,
        failures.time_s,
        failures.event_ordinal,
        failures.reason_code,
    )
    _write_particle_failure_into(
        context,
        failures.count,
        particle_index,
        failure.time_s,
        failure.position_m,
        failure.velocity_m_s,
        failure.charge_number,
        np.uint16(failure.reason_code),
    )
    failures.count += 1


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
        interaction_kind=_BOUNDARY_INTERACTION_OUTPUT[pending.interaction_code[:count][order]],
        primary_facet_id=primary,
        destination_facet_id=pending.destination_facet_id[:count][order],
        boundary_id=geometry.boundary_id[primary],
        material_id=geometry.material_id[primary],
        position_m=pending.position_m[:count][order],
        position_post_m=pending.position_post_m[:count][order],
        normal=pending.normal[:count][order],
        velocity_pre_m_s=pending.velocity_pre_m_s[:count][order],
        velocity_post_m_s=pending.velocity_post_m_s[:count][order],
        charge_number_pre=pending.charge_pre_number[:count][order],
        charge_number_post=pending.charge_pre_number[:count][order],
        contact_radius_m=schedule.contact_radius_m[rows],
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
    storage: np.ndarray | None = None,
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
    replay = _replay_columns(times, rows, shape, storage)
    replay.position_m.fill(np.nan)
    replay.velocity_m_s.fill(np.nan)
    replay.charge_number.fill(np.nan)
    replay.presence.fill(False)
    replay.lifecycle.fill(_LIFECYCLE_PENDING)
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


def _replay_columns(
    times: np.ndarray,
    rows: np.ndarray,
    shape: tuple[int, int],
    storage: np.ndarray | None,
) -> _ReplayBuffer:
    if storage is None or not times.size:
        return _ReplayBuffer(
            times,
            rows,
            np.empty((*shape, 2), dtype="<f8"),
            np.empty((*shape, 2), dtype="<f8"),
            np.empty(shape, dtype="<f8"),
            np.empty(shape, dtype=np.bool_),
            np.empty(shape, dtype="<u1"),
        )
    layouts = (
        (times.shape, "<f8"),
        (rows.shape, "<i8"),
        ((*shape, 2), "<f8"),
        ((*shape, 2), "<f8"),
        (shape, "<f8"),
        (shape, "|b1"),
        (shape, "|u1"),
    )
    columns = []
    offset = 0
    for dimensions, dtype in layouts:
        size = math.prod(dimensions) * np.dtype(dtype).itemsize
        if offset + size > storage.nbytes:
            raise EngineError("macro replay exceeds its planned shared capacity")
        columns.append(np.ndarray(dimensions, dtype=dtype, buffer=storage, offset=offset))
        offset += size
    replay = _ReplayBuffer(*columns)
    replay.time_s[:] = times
    replay.particle_index[:] = rows
    return replay


def _record_replay_rows(
    replay: _ReplayBuffer,
    proposal: StepProposal,
    local_rows: np.ndarray,
    prepared: _PreparedRun,
    *,
    exclude_target: bool = False,
) -> None:
    """Sample accepted paths into bounded scratch and scatter requested outputs."""

    if not replay.time_s.size or not local_rows.size or not replay.particle_index.size:
        return
    workspace = prepared.replay_workspace
    for time_row, requested_time_s in enumerate(replay.time_s):
        count = _select_proposal_replay_rows_into(
            local_rows,
            proposal.particle_index,
            proposal.start_time_s,
            proposal.target_time_s,
            replay.particle_index,
            float(requested_time_s),
            exclude_target,
            workspace,
        )
        if not count:
            continue
        sample = proposal.state_at_times_into(
            workspace.proposal_rows[:count],
            workspace.time_s[:count],
            workspace.sample,
            workspace.prefix,
        )
        if bool((sample.numerical_status != NUMERICAL_STATUS_OK).any()):
            raise EngineError("accepted trajectory replay produced a numerical failure")
        if not bool(sample.support_inside.all() and sample.applicability_inside.all()):
            raise EngineError("trajectory frame is outside field or model support")
        if prepared.case.data.coordinate_system == "axisymmetric_rz":
            rz_signed_stage_into(
                sample.position_m,
                sample.velocity_m_s,
                sample.position_m,
                sample.velocity_m_s,
                workspace.radial_sign[:count],
            )
        _scatter_proposal_replay_into(
            time_row,
            workspace.replay_slots[:count],
            sample.position_m,
            sample.velocity_m_s,
            sample.charge_number,
            replay.position_m,
            replay.velocity_m_s,
            replay.charge_number,
            replay.presence,
            replay.lifecycle,
        )


@njit(cache=True, fastmath=False, nogil=True)
def _select_proposal_replay_rows_into(
    rows: np.ndarray,
    particle_index: np.ndarray,
    start_time_s: np.ndarray,
    target_time_s: np.ndarray,
    replay_particle_index: np.ndarray,
    time_s: float,
    exclude_target: bool,
    workspace: _ProposalReplayWorkspace,
) -> int:
    """Preserve output eligibility and stable particle matching without gathers."""

    count = 0
    for offset in range(rows.size):
        row = rows[offset]
        if start_time_s[row] > time_s or time_s > target_time_s[row]:
            continue
        if exclude_target and time_s >= target_time_s[row]:
            continue
        particle = particle_index[row]
        slot = np.searchsorted(replay_particle_index, particle)
        if slot >= replay_particle_index.size or replay_particle_index[slot] != particle:
            continue
        workspace.proposal_rows[count] = row
        workspace.replay_slots[count] = slot
        workspace.time_s[count] = time_s
        count += 1
    return count


@njit(cache=True, fastmath=False, nogil=True)
def _scatter_proposal_replay_into(
    time_row: int,
    slots: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: np.ndarray,
    replay_position_m: np.ndarray,
    replay_velocity_m_s: np.ndarray,
    replay_charge_number: np.ndarray,
    presence: np.ndarray,
    lifecycle: np.ndarray,
) -> None:
    for row in range(slots.size):
        slot = slots[row]
        for component in range(2):
            replay_position_m[time_row, slot, component] = position_m[row, component]
            replay_velocity_m_s[time_row, slot, component] = velocity_m_s[row, component]
        replay_charge_number[time_row, slot] = charge_number[row]
        presence[time_row, slot] = True
        lifecycle[time_row, slot] = _LIFECYCLE_ACTIVE


@njit(cache=True, fastmath=False, nogil=True)
def _record_replay_jump(
    replay: _ReplayBuffer,
    particle_index: int,
    time_s: float,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    charge_number: float,
) -> None:
    """Overlay one right-continuous zero-time state in canonical event order."""

    if not replay.time_s.size or not replay.particle_index.size:
        return
    time_row = int(np.searchsorted(replay.time_s, time_s))
    if time_row >= replay.time_s.size or replay.time_s[time_row] != time_s:
        return
    particle_row = int(np.searchsorted(replay.particle_index, particle_index))
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
        position_m=replay.position_m[time_row, slots],
        velocity_m_s=replay.velocity_m_s[time_row, slots],
        charge_number=replay.charge_number[time_row, slots],
        lifecycle=replay.lifecycle[time_row, slots],
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
        schedule.contact_radius_m,
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
    curved_enabled: bool,
) -> int:
    """Account target, depth, and interaction columns for every event stack."""

    if not event_paths:
        return 0
    stack_capacity = case.spec.solver.event.max_refinements + 1
    stack_bytes = 3 * np.dtype("<i8").itemsize * stack_capacity
    has_point_view = case.spec.topology is not None or any(
        rule.contact_geometry == "particle_center" for rule in case.spec.boundaries
    )
    # The point/surface arbitration holds a second row-aligned event result;
    # its candidate arena is accounted separately in the geometry plan.
    point_arbitration_bytes = 96 if has_point_view else 0
    # Curved query/selector/gather/radius and result: 232 + 100 B/row.
    # Exact continuation/selection 160, prepared query 106, gather 73,
    # radius transient 32, persistent exact result/control 124 B/row.
    # Optional point shadow owns 84 B within the conservative 96 B reserve.
    # Boundary response bank (44 B/row) is added separately only with facets.
    control_bytes = 42 + 70 + 88 + 32 + 100 if curved_enabled else 160 + 106 + 73 + 32 + 124
    return stack_bytes + point_arbitration_bytes + control_bytes


def _uses_exact_specialization(
    physics: PhysicsPlan,
    constant_acceleration_m_s2: np.ndarray | None,
) -> bool:
    """Return whether prepared facts select an exact motion path."""

    return not physics.requires_stage_evaluation or constant_acceleration_m_s2 is not None


def _uses_rk4_dense_certificate(
    case: SimulationCase,
    physics: PhysicsPlan,
    constant_acceleration_m_s2: np.ndarray | None,
) -> bool:
    """Return whether a general stage-evaluated RK4 path is selected."""

    return case.spec.solver.integrator == "rk4_fixed" and not _uses_exact_specialization(
        physics, constant_acceleration_m_s2
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
    # The local-field owner retains 64 int64 candidate IDs, an int64 total
    # count, availability, and row status: 522 B/row, rounded to 544 here.
    # All candidates are counted; excess never becomes a truncated range.
    # Packed primitive lower/upper buffers are separately charged to stage
    # composition work. Exponential uses the arena without the RK4 DFS stack.
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
    # Concurrent owned arrays for N slab rows, base depth D and extra adaptive
    # depth H=A-D: base cursor 33*N*(D+1)+8; adaptive DFS (H>0)
    # 41*N*(H+1)+24*H+32 (H=0:32); two pending-root banks 144*N+8.
    # Conditional children use scalar temporaries, with one current increment
    # and H saved right frames rather than two extra children at every level.
    # The 128*(A+4) per-row allowance covers these banks and fixed controls even
    # for N=1. Root metadata/compact maps fit its rounding; live proposal and
    # endpoint transients remain in the general stage allowance. A consumer
    # returns before splitting, and pending banks swap only after the complete
    # base/adaptive tree has unwound; caller input columns are never recycled.
    return 128 * (physics.noise.adaptive_max_depth + 4)


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
    plan_for_capacity: Callable[[int], CpuMemoryPlan],
) -> tuple[int, CpuMemoryPlan]:
    """Choose a bounded CSR facet budget that leaves one slab row runnable."""

    minimum_capacity = geometry.facet_count + 1 if enabled and geometry.facet_count else 0
    minimum_bytes = 64 * 1024
    maximum_bytes = 8 * 1024 * 1024
    target_bytes = min(maximum_bytes, max(minimum_bytes, memory_limit_bytes // 64))
    target_capacity = target_bytes // np.dtype("<i8").itemsize
    capacity = max(minimum_capacity, target_capacity) if minimum_capacity else 0
    while True:
        memory_plan = plan_for_capacity(capacity)
        fits = memory_plan.planned_bytes <= memory_limit_bytes and (
            memory_plan.particle_count == 0 or memory_plan.slab_particles >= 1
        )
        if fits:
            return capacity, memory_plan
        if capacity == minimum_capacity:
            raise EngineError("predicted solver memory exceeds resources.memory_limit_mb")
        capacity = max(minimum_capacity, capacity // 2)


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


def _topology_memory_bytes(topology: PreparedPeriodicTopology | None) -> int:
    """Account the topology map plus its material/event contact masks."""

    if topology is None:
        return 0
    arrays = (
        topology.peer_facet_id,
        topology.peer_node_ids,
        topology.translation_m,
        topology.pair_id,
        topology.facet_is_periodic,
        topology.periodic_facet_id,
    )
    mask_bytes = 2 * topology.facet_is_periodic.nbytes
    return sum(array.nbytes for array in arrays) + mask_bytes


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
        contact_radius_m=schedule.contact_radius_m,
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


def _resolved_boundary_laws(prepared: _PreparedRun) -> list[dict[str, object]]:
    """Describe the resolved response and geometric meaning of each boundary."""

    case = prepared.case
    contact_by_group = {rule.boundary_group: rule.contact_geometry for rule in case.spec.boundaries}
    return [
        {
            "group": case.data.geometry.group_names[rule.group_id],
            "contact_geometry": contact_by_group[case.data.geometry.group_names[rule.group_id]],
            "priority": rule.priority,
            "law": rule.law_id,
            "normal_restitution": rule.normal_restitution,
            "tangential_restitution": rule.tangential_restitution,
            "stick_probability": rule.stick_probability,
            "otherwise_law": rule.otherwise_law_id,
            "wall_temperature_K": rule.wall_temperature_K,
            "diffuse_reflection_fraction": rule.diffuse_reflection_fraction,
            "wall_velocity_m_s": rule.wall_velocity_m_s,
        }
        for rule in prepared.boundary_rules
    ]


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
        "field_time_revision": FIELD_TIME_REVISION,
        "required_field_revision": REQUIRED_FIELD_REVISION,
        "geometry_algorithm_revision": (GEOMETRY_ALGORITHM_REVISION),
        "event_algorithm_revision": EVENT_ALGORITHM_REVISION,
        "topology_algorithm_revision": _topology_algorithm_revision(prepared),
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
        "brownian_tree_policy_revision": brownian["tree_policy_revision"],
        "brownian_interval_tree_depth": brownian["interval_tree_depth"],
        "brownian_adaptive_max_depth": brownian["adaptive_max_depth"],
        "random_draw_kinds": {
            "wall_probabilistic_stick": WALL_PROBABILISTIC_STICK_STREAM,
            "wall_maxwell_diffuse": WALL_MAXWELL_DIFFUSE_STREAM,
            "wall_maxwell_normal": WALL_MAXWELL_NORMAL_STREAM,
            "wall_maxwell_tangential": WALL_MAXWELL_TANGENTIAL_STREAM,
            "brownian_root_normal": brownian["root_normal_stream"],
            "brownian_split_normal": brownian["split_normal_stream"],
        },
        "requested": {
            "integrator": spec.solver.integrator,
            "backend": spec.solver.backend,
            "seed": spec.solver.seed,
            "workers": prepared.memory_plan.worker_count,
        },
        "execution": {
            "worker_count": prepared.memory_plan.worker_count,
            "policy": "spawn_particle_slabs" if prepared.memory_plan.worker_count > 1 else "serial",
            "barrier": "accepted_macro_step",
            "writer_owner": "parent",
            "callback_cache_policy": "process_local_overloads",
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
            "boundary_laws": _resolved_boundary_laws(prepared),
            "required_fields": [
                {
                    "name": name,
                    "layout": field.layout,
                    "unit": field.unit,
                    "components": list(field.components),
                    "stored_basis": field.stored_basis,
                    "time_interpolation": "static" if field.time_s is None else "linear",
                    "snapshot_count": 1 if field.time_s is None else int(field.time_s.size),
                    "snapshot_range_s": (
                        None
                        if field.time_s is None
                        else [float(field.time_s[0]), float(field.time_s[-1])]
                    ),
                }
                for name, field in prepared.fields.fields.items()
            ],
        },
        "time": {
            "start_s": spec.time.start_s,
            "end_s": spec.time.end_s,
            "dt_s": spec.time.dt_s,
            "field_snapshot_splits_s": list(prepared.field_time_split_s),
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


def _topology_algorithm_revision(prepared: _PreparedRun) -> str | None:
    """Return the active topology revision without adding manifest branching."""

    if prepared.topology is None:
        return None
    return TOPOLOGY_ALGORITHM_REVISION
