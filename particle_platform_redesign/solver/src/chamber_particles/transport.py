"""One shared serial/process particle-slab update and its bounded numerical work."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from typing import NamedTuple, Protocol

import numpy as np
from numba import njit

from .boundaries import (
    BOUNDARY_OUTCOME_ESCAPED,
    BOUNDARY_OUTCOME_HELD,
    BOUNDARY_OUTCOME_STUCK,
    BOUNDARY_STATUS_OK,
    BoundaryResponseBatch,
    BoundaryRule,
    PreparedBoundaryRules,
    resolve_boundary_responses_batch,
)
from .case import SimulationCase
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
    CpuMemoryPlan,
)
from .events import (
    CURVED_STATUS_AXIS,
    CURVED_STATUS_CLEAR,
    CURVED_STATUS_FAILURE,
    CURVED_STATUS_SPLIT,
    CURVED_STATUS_WALL,
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
    classify_surface_release_batch,
    curved_event_result_view,
    exact_event_result_view,
    locate_curved_first_event_into,
    locate_exact_first_event_into,
    prepare_curved_event_batch,
    prepare_exact_event_batch,
    resolve_event_budget,
)
from .fields import (
    LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW,
    FieldBatch,
    FieldLocationError,
    FieldWorkspace,
    NumericFieldContext,
    NumericFieldRangeWorkspace,
    NumericFieldWorkspace,
    PreparedFieldSet,
    local_prepared_field_ranges_into,
    sample_prepared_fields_into,
)
from .geometry import (
    GeometryPreparationError,
    PreparedGeometry,
    centers_respect_contact_radius,
    contact_normals_for_candidates,
    points_inside_volume,
)
from .integrators import (
    PROPOSAL_PREFIX_WORK_BYTES_PER_PARTICLE,
    PROPOSAL_SAMPLE_WORK_BYTES_PER_PARTICLE,
    RELAXATION_ADVANCE_WORK_BYTES_PER_PARTICLE,
    RK4_ADVANCE_WORK_BYTES_PER_PARTICLE,
    RK4_DENSE_FULL_ENCLOSURE_BYTES_PER_PARTICLE,
    STAGE_ADVANCE_WORK_BYTES_PER_PARTICLE,
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
    BoundaryEvents,
    FailureEvents,
)
from .physics.catalog import (
    BROWNIAN_MIDPOINT_2D_REVISION,
    PhysicsPlan,
)
from .physics.forces import (
    CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE,
    CONTINUOUS_APPLICABILITY_OK,
    PhysicsEvaluationError,
)
from .physics.runtime import (
    AccelerationBoundContext,
    ContinuousApplicabilityContext,
    PhysicsNumericContext,
    PhysicsNumericOutputs,
    PhysicsRuntime,
    PhysicsRuntimeEvaluation,
    PhysicsRuntimeWorkspace,
    acceleration_abs_upper_into,
    brownian_thermal_variance_into,
    charge_interval_inside_prepared_invariant_into,
    evaluate_continuous_applicability_into,
    evaluate_local_continuous_applicability_into,
    evaluate_prepared_physics_into,
    local_additive_acceleration_abs_upper_into,
    prescribed_force_variance_into,
)
from .rng import (
    BROWNIAN_ROOT_NORMAL_STREAM,
    BROWNIAN_SPLIT_NORMAL_STREAM,
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
    ParticleSchedule,
)
from .stochastic import (
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
    TOPOLOGY_CANDIDATE_INVALID,
    TOPOLOGY_CANDIDATE_MATERIAL,
    TOPOLOGY_CANDIDATE_PERIODIC,
    PreparedPeriodicTopology,
    classify_periodic_candidate_rows,
)

_MAX_EVENT_ORDINAL = int(np.iinfo(np.uint32).max)


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
        "isotropic",
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
            *(
                np.empty(shape, dtype=dtype)
                for shape, dtype in cls.allocation_layout(capacity, stack_capacity)
            )
        )

    @staticmethod
    def allocation_layout(
        capacity: int, stack_capacity: int
    ) -> tuple[tuple[tuple[int, ...], str], ...]:
        """Declare this finite workspace's concrete shapes and storage types."""
        return (
            ((capacity, stack_capacity), "<f8"),
            ((capacity, stack_capacity), "<f8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<u2"),
        )

    @classmethod
    def bytes_for(cls, capacity: int, stack_capacity: int) -> int:
        """Count the same columns used by allocation, without allocating rows."""
        return sum(
            math.prod(shape) * np.dtype(dtype).itemsize
            for shape, dtype in cls.allocation_layout(capacity, stack_capacity)
        )


class _NumericSupportContext(NamedTuple):
    """Prepared field/domain relation; contact remains an engine verdict."""

    regular_box: bool
    lower_m: tuple[float, float]
    upper_m: tuple[float, float]
    covers_geometry: bool
    axisymmetric_rz: bool


@dataclass(frozen=True, slots=True)
class TransportCapabilities:
    """One resolved path/workspace decision shared by allocation and execution."""

    stage_evaluation: bool
    exact_motion: bool
    dense_rk4: bool
    local_ranges: bool
    event_paths: bool
    material_walls: bool
    curved_events: bool
    stochastic: bool
    path_kind: str


def resolve_transport_capabilities(
    physics: PhysicsPlan,
    integrator: str,
    constant_acceleration_m_s2: np.ndarray | None,
    coordinate_system: str,
    facet_count: int,
) -> TransportCapabilities:
    exact = not physics.requires_stage_evaluation or constant_acceleration_m_s2 is not None
    dense = integrator == "rk4_fixed" and not exact
    event_paths = facet_count > 0 or coordinate_system == "axisymmetric_rz"
    stochastic = physics.noise is not None
    if stochastic:
        path_kind = "cubic_hermite"
    elif not physics.requires_stage_evaluation:
        path_kind = "linear_exact"
    elif constant_acceleration_m_s2 is not None:
        path_kind = "quadratic_exact"
    elif integrator == "exponential_midpoint":
        path_kind = "exponential_midpoint_reintegrated"
    else:
        path_kind = "rk4_dense"
    return TransportCapabilities(
        stage_evaluation=physics.requires_stage_evaluation,
        exact_motion=exact,
        dense_rk4=dense,
        local_ranges=dense or (integrator == "exponential_midpoint" and not exact),
        event_paths=event_paths,
        material_walls=facet_count > 0,
        curved_events=not exact and event_paths,
        stochastic=stochastic,
        path_kind=path_kind,
    )


@dataclass(frozen=True, slots=True)
class PreparedTransport:
    case: SimulationCase
    capabilities: TransportCapabilities
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
    certificate_workspace: _ApplicabilityCertificateWorkspace = dataclass_field(
        metadata={"process_private": True}
    )
    curved_workspace: _CurvedWavefront = dataclass_field(metadata={"process_private": True})
    exact_workspace: _ExactWavefront = dataclass_field(metadata={"process_private": True})
    boundary_response_buffer: BoundaryResponseBatch = dataclass_field(
        metadata={"process_private": True}
    )
    event_query_geometry: EventGeometry
    event_query_arena: EventArena = dataclass_field(metadata={"process_private": True})
    replay_workspace: _ProposalReplayWorkspace = dataclass_field(metadata={"process_private": True})
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

    @property
    def private_workspaces(self) -> tuple[object, ...]:
        """Return the finite scratch owners declared with their fields."""
        return self.dynamics.private_workspaces + tuple(
            getattr(self, item.name)
            for item in dataclass_fields(self)
            if item.metadata.get("process_private")
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

    prepared: PreparedTransport
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
            *(
                np.empty(shape, dtype=dtype)
                for shape, dtype in cls.allocation_layout(capacity, stack_capacity, stochastic)
            )
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

    @staticmethod
    def allocation_layout(
        capacity: int, stack_capacity: int, stochastic: bool
    ) -> tuple[tuple[tuple[int, ...], str], ...]:
        """Declare this finite workspace's concrete shapes and storage types."""
        return (
            ((capacity,), "<i8"),
            ((capacity,), "<f8"),
            ((capacity, 2), "<f8"),
            ((capacity, 2), "<f8"),
            ((capacity,), "<f8"),
            ((capacity,), "<f8"),
            ((capacity,), "<i8"),
            ((capacity,), "?"),
            ((capacity, stack_capacity), "<f8"),
            ((capacity, stack_capacity), "<i8"),
            ((capacity, stack_capacity), "<i8"),
            ((capacity,), "<i8"),
            ((capacity if stochastic else 0,), "<f8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<f8"),
            ((capacity,), "<i8"),
            ((capacity,), "<u2"),
        )

    @classmethod
    def bytes_for(cls, capacity: int, stack_capacity: int, stochastic: bool) -> int:
        """Count the same columns used by allocation, without allocating rows."""
        return sum(
            math.prod(shape) * np.dtype(dtype).itemsize
            for shape, dtype in cls.allocation_layout(capacity, stack_capacity, stochastic)
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
            *(
                np.empty(shape, dtype=dtype)
                for shape, dtype in cls.allocation_layout(capacity, stack_capacity)
            )
        )

    def view(self, count: int) -> _ExactWavefront:
        return _ExactWavefront(*(column[:count] for column in self))

    @staticmethod
    def allocation_layout(
        capacity: int, stack_capacity: int
    ) -> tuple[tuple[tuple[int, ...], str], ...]:
        """Declare this finite workspace's concrete shapes and storage types."""
        return (
            ((capacity,), "<f8"),
            ((capacity, 2), "<f8"),
            ((capacity, 2), "<f8"),
            ((capacity,), "<f8"),
            ((capacity, 2), "<f8"),
            ((capacity,), "<i8"),
            ((capacity, stack_capacity), "<f8"),
            ((capacity, stack_capacity), "<i8"),
            ((capacity, stack_capacity), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<f8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
            ((capacity,), "<f8"),
            ((capacity,), "<i8"),
            ((capacity,), "<i8"),
        )

    @classmethod
    def bytes_for(cls, capacity: int, stack_capacity: int) -> int:
        """Count the same columns used by allocation, without allocating rows."""
        return sum(
            math.prod(shape) * np.dtype(dtype).itemsize
            for shape, dtype in cls.allocation_layout(capacity, stack_capacity)
        )


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


@dataclass(frozen=True, slots=True)
class _SlabResult:
    """Aggregate work returned after one bounded particle slab."""

    statistics: _EventStatistics


class _EventSink(Protocol):
    def write_boundary_events(self, events: BoundaryEvents) -> None: ...

    def write_failure_events(self, events: FailureEvents) -> None: ...


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

    @staticmethod
    def allocation_layout(
        capacity: int, axisymmetric_rz: bool
    ) -> tuple[tuple[tuple[int, ...], str], ...]:
        """Declare the owned numeric columns, excluding borrowed stage arrays."""
        rz_capacity = capacity if axisymmetric_rz else 0
        return (
            ((rz_capacity, 2), "<f8"),
            ((rz_capacity, 2), "<f8"),
            ((rz_capacity,), "<f8"),
            ((capacity,), "<i8"),
        )

    @classmethod
    def bytes_for(cls, capacity: int, axisymmetric_rz: bool) -> int:
        return sum(
            math.prod(shape) * np.dtype(dtype).itemsize
            for shape, dtype in cls.allocation_layout(capacity, axisymmetric_rz)
        )


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

    @staticmethod
    def allocation_layout(capacity: int) -> tuple[tuple[tuple[int, ...], str], ...]:
        """Declare the owned numeric columns, excluding borrowed stage arrays."""
        return (
            ((capacity, 2), "<f8"),
            ((capacity, 2), "<f8"),
            ((capacity,), "?"),
            ((capacity,), "<u1"),
        )

    @classmethod
    def bytes_for(cls, capacity: int) -> int:
        return sum(
            math.prod(shape) * np.dtype(dtype).itemsize
            for shape, dtype in cls.allocation_layout(capacity)
        )


class _LangevinCoefficientWorkspace(NamedTuple):
    thermal_variance_m2_s2: np.ndarray
    numerical_status: np.ndarray

    @staticmethod
    def allocation_layout(capacity: int) -> tuple[tuple[tuple[int, ...], str], ...]:
        """Declare the owned numeric columns, excluding borrowed stage arrays."""
        return (
            ((capacity,), "<f8"),
            ((capacity,), "<u1"),
        )

    @classmethod
    def bytes_for(cls, capacity: int) -> int:
        return sum(
            math.prod(shape) * np.dtype(dtype).itemsize
            for shape, dtype in cls.allocation_layout(capacity)
        )


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
    """Compose sampled relaxation and the selected noise before any draw."""

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
    prescribed = context.physics.bindings.prescribed
    if prescribed.force_diffusion >= 0:
        prescribed_force_variance_into(
            context.physics.mass_kg,
            particle_index,
            stage_workspace.fields.packed_values[:count, prescribed.friction],
            stage_workspace.fields.packed_values[:count, prescribed.force_diffusion],
            workspace.thermal_variance_m2_s2[:count],
            thermal_status,
        )
    else:
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

    @staticmethod
    def allocation_layout(
        capacity: int, axisymmetric_rz: bool
    ) -> tuple[tuple[tuple[int, ...], str], ...]:
        """Declare the owned numeric columns, excluding borrowed stage arrays."""
        box_capacity = capacity if axisymmetric_rz else 0
        return (
            ((box_capacity, 2), "<f8"),
            ((box_capacity, 2), "<f8"),
            ((box_capacity, 2), "<f8"),
            ((box_capacity, 2), "<f8"),
            ((capacity, 2), "<f8"),
            ((capacity,), "?"),
            ((capacity,), "<u1"),
        )

    @classmethod
    def bytes_for(cls, capacity: int, axisymmetric_rz: bool) -> int:
        return sum(
            math.prod(shape) * np.dtype(dtype).itemsize
            for shape, dtype in cls.allocation_layout(capacity, axisymmetric_rz)
        )


@dataclass(frozen=True, slots=True)
class TransportWorkspaceBudget:
    """Finite numerical workspace budgets beside their allocating owners.

    Concrete retained columns derive from their allocation declarations. Named
    allowances also cover coexisting proposal/query/selection temporaries; they
    intentionally remain conservative rather than charging only retained arrays.
    """

    stage: int
    replay: int
    dense_path: int
    stochastic_tree: int
    events: int
    certificates: int
    release: int

    @classmethod
    def resolve(
        cls,
        case: SimulationCase,
        schedule: ParticleSchedule,
        fields: PreparedFieldSet,
        physics: PhysicsPlan,
        capabilities: TransportCapabilities,
        *,
        observations: bool,
    ) -> TransportWorkspaceBudget:
        stack = case.spec.solver.event.max_refinements + 1
        # The stage bank's fields/physics output arrays are charged separately
        # by the CPU scratch plan; this bank owns only composed coordinate and
        # certificate columns plus the chosen advance's concrete scratch.
        stage = _stage_workspace_bytes(fields, capabilities, case.data.coordinate_system)
        replay = (
            PROPOSAL_SAMPLE_WORK_BYTES_PER_PARTICLE
            + PROPOSAL_PREFIX_WORK_BYTES_PER_PARTICLE
            + 4 * np.dtype("<i8").itemsize
            if observations
            else 0
        )
        dense = (
            176 + 3 * RK4_DENSE_FULL_ENCLOSURE_BYTES_PER_PARTICLE if capabilities.dense_rk4 else 0
        )
        # The conditional DFS/pending-root allowance bounds N=1 as well as a
        # full slab. It includes fixed controls and saved right children.
        stochastic = (
            128 * (physics.noise.adaptive_max_depth + 4) if physics.noise is not None else 0
        )
        events = _event_workspace_bytes(case, capabilities, stack)
        certificates = _certificate_workspace_bytes(capabilities, stack)
        # Includes classifier output, response subsets, remap and CSR tail.
        release = (
            (272 if bool((schedule.source_facet_id >= 0).any()) else 64)
            if capabilities.event_paths
            else 0
        )
        return cls(stage, replay, dense, stochastic, events, certificates, release)


def _stage_workspace_bytes(
    fields: PreparedFieldSet, capabilities: TransportCapabilities, coordinate_system: str
) -> int:
    # A no-force exact path has no stage evaluation; a constant-force exact
    # specialization still owns its compact stage bank.
    if not capabilities.stage_evaluation:
        return 0
    rz = coordinate_system == "axisymmetric_rz"
    advance = (
        RK4_ADVANCE_WORK_BYTES_PER_PARTICLE
        if capabilities.dense_rk4
        else STAGE_ADVANCE_WORK_BYTES_PER_PARTICLE
        if capabilities.exact_motion
        else RELAXATION_ADVANCE_WORK_BYTES_PER_PARTICLE
    )
    stage = (
        _NumericStageWorkspace.bytes_for(1, rz)
        + np.dtype("<u1").itemsize  # Borrowed incoming status belongs to physics scratch.
        + _GlobalCertificateWorkspace.bytes_for(1)
        + advance
        + _LangevinCoefficientWorkspace.bytes_for(int(capabilities.stochastic))
    )
    if capabilities.local_ranges:
        components = int(fields.numeric_context.component_offsets[-1])
        packed = NumericFieldRangeWorkspace.bytes_for(
            1, components, 0
        ) - NumericFieldRangeWorkspace.bytes_for(1, 0, 0)
        stage += packed + _LocalCertificateWorkspace.bytes_for(1, rz)
    return int(stage)


def _event_workspace_bytes(
    case: SimulationCase, capabilities: TransportCapabilities, stack: int
) -> int:
    if not capabilities.event_paths:
        return 0
    has_point_view = case.spec.topology is not None or any(
        rule.contact_geometry == "particle_center" for rule in case.spec.boundaries
    )
    if capabilities.curved_events:
        # 209 B beyond the retained wavefront covers concurrent query/select/
        # gather/radius/result columns, including its optional 8 B OU restart.
        # The allowance stays the previous 332+24*stack B/row for both paths.
        control = _CurvedWavefront.bytes_for(1, stack, False) + 209
    else:
        # Exact query/gather/radius/result control coexists with its wavefront.
        control = _ExactWavefront.bytes_for(1, stack) + 106 + 73 + 32 + 124
    return control + 96 * int(has_point_view) + 44 * int(capabilities.material_walls)


def _certificate_workspace_bytes(capabilities: TransportCapabilities, stack: int) -> int:
    interval = (
        ((_ApplicabilityCertificateWorkspace.bytes_for(1, stack) + 7) // 8) * 8
        if capabilities.dense_rk4
        else 0
    )
    arena = (
        (
            (NumericFieldRangeWorkspace.bytes_for(1, 0, LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW) + 31)
            // 32
        )
        * 32
        if capabilities.local_ranges
        else 0
    )
    return interval + arena


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
    field_workspace: FieldWorkspace = dataclass_field(metadata={"process_private": True})
    physics_workspace: PhysicsRuntimeWorkspace = dataclass_field(metadata={"process_private": True})
    native_context: _NumericStageContext = dataclass_field(init=False)
    native_workspace: _NumericStageWorkspace = dataclass_field(
        init=False, metadata={"process_private": True}
    )
    native_rk4_workspace: Rk4AdvanceWorkspace | None = dataclass_field(
        default=None, metadata={"process_private": True}
    )
    native_stage_advance_workspace: StageAdvanceWorkspace | None = dataclass_field(
        default=None, metadata={"process_private": True}
    )
    native_relaxation_workspace: RelaxationAdvanceWorkspace | None = dataclass_field(
        default=None, metadata={"process_private": True}
    )
    field_range_workspace: NumericFieldRangeWorkspace | None = dataclass_field(
        default=None, metadata={"process_private": True}
    )
    global_certificate_workspace: _GlobalCertificateWorkspace = dataclass_field(
        init=False, metadata={"process_private": True}
    )
    local_certificate_workspace: _LocalCertificateWorkspace = dataclass_field(
        init=False, metadata={"process_private": True}
    )
    langevin_coefficient_workspace: _LangevinCoefficientWorkspace = dataclass_field(
        init=False, metadata={"process_private": True}
    )

    @property
    def private_workspaces(self) -> tuple[object, ...]:
        return tuple(
            getattr(self, item.name)
            for item in dataclass_fields(self)
            if item.metadata.get("process_private")
        )

    def __post_init__(self) -> None:
        capacity = self.field_workspace.capacity
        self.global_certificate_workspace = _GlobalCertificateWorkspace(
            *(
                np.empty(shape, dtype=dtype)
                for shape, dtype in _GlobalCertificateWorkspace.allocation_layout(capacity)
            )
        )
        local_capacity = (
            self.field_range_workspace.range_available.size
            if self.field_range_workspace is not None
            else 0
        )
        axisymmetric_rz = self.coordinate_system == "axisymmetric_rz"
        self.local_certificate_workspace = _LocalCertificateWorkspace(
            *(
                np.empty(shape, dtype=dtype)
                for shape, dtype in _LocalCertificateWorkspace.allocation_layout(
                    local_capacity, axisymmetric_rz
                )
            )
        )
        langevin_capacity = capacity if self.runtime.plan.noise is not None else 0
        self.langevin_coefficient_workspace = _LangevinCoefficientWorkspace(
            *(
                np.empty(shape, dtype=dtype)
                for shape, dtype in _LangevinCoefficientWorkspace.allocation_layout(
                    langevin_capacity
                )
            )
        )
        self.native_context = _NumericStageContext(
            self.fields.numeric_context,
            self.runtime.native_context,
            axisymmetric_rz,
            (np.empty(0, dtype="<i8") if self.last_field_cell is None else self.last_field_cell),
        )
        self.native_workspace = _NumericStageWorkspace(
            self.field_workspace.numeric,
            self.physics_workspace.numeric_outputs(capacity),
            self.physics_workspace.incoming_status,
            *(
                np.empty(shape, dtype=dtype)
                for shape, dtype in _NumericStageWorkspace.allocation_layout(
                    capacity, axisymmetric_rz
                )
            ),
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


def _advance_active_slabs(
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    if noise.revision not in (
        BROWNIAN_MIDPOINT_2D_REVISION,
        "prescribed_force_diffusion_midpoint_2d_v1",
    ):
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    if not runtime.prepared.capabilities.curved_events:
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
    prepared: PreparedTransport,
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
    if prepared.capabilities.curved_events:
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


def _certify_curved_path(
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    use_exact_specialization = prepared.capabilities.exact_motion
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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


def _is_reintegrated_path(proposal: StepProposal) -> bool:
    return proposal.path_kind in {
        "cubic_hermite",
        "rk4_dense",
        "exponential_midpoint_reintegrated",
    }


def _curved_row_failure_codes(
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
        curved_event_path=prepared.capabilities.curved_events,
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
    prepared: PreparedTransport,
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
    if prepared.capabilities.curved_events:
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
    prepared: PreparedTransport, particle: int, facet_id: int, remains_active: bool
) -> np.uint8:
    """Retain departure proof for a curved path or a finite material origin."""

    if remains_active and (
        prepared.capabilities.curved_events
        or (
            prepared.schedule.contact_radius_m[particle] > 0.0
            and prepared.geometry.facet_contact_enabled[facet_id]
        )
    ):
        return SURFACE_STATE_DEPARTURE
    return SURFACE_STATE_RESOLVED


def _commit_surface_release_response_row(
    prepared: PreparedTransport,
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
    if remains_active and prepared.capabilities.curved_events:
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


def _proposal_start_state(
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
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
