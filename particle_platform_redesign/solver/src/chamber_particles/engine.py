"""Prepare, parent clocks, releases, checkpoints and the single result writer."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from os import PathLike
from types import MappingProxyType

import numpy as np

from .boundaries import (
    BOUNDARY_ALGORITHM_REVISION,
    BoundaryLawError,
    BoundaryRule,
    allocate_boundary_response_buffer,
    prepare_boundary_rule,
    prepare_boundary_rules,
    validate_boundary_rule_frames,
)
from .case import CASE_FORMAT_VERSION, SimulationCase
from .case_format import resident_array_bytes
from .cpu import (
    CPU_RUNTIME_LAYOUT_REVISION,
    MEMORY_PLAN_REVISION,
    ActiveIndex,
    CpuMemoryPlan,
    ParticleState,
    early_memory_requirement_bytes,
    plan_cpu_memory,
)
from .events import (
    EVENT_ALGORITHM_REVISION,
    SURFACE_STATE_PENDING,
    SURFACE_STATE_RESOLVED,
    EventLocationError,
    allocate_event_arena,
    certify_event_point_interiors,
    classify_event_point,
    prepare_event_geometry,
)
from .execution import (
    ProcessTeam,
    validate_worker_count,
)
from .fields import (
    FIELD_LOCATION_REVISION,
    FIELD_TIME_REVISION,
    REQUIRED_FIELD_REVISION,
    FieldLocationError,
    PreparedFieldSet,
    RequiredFieldMetadata,
    prepare_required_fields,
    validate_periodic_field_seams,
)
from .geometry import (
    GEOMETRY_ALGORITHM_REVISION,
    GeometryPreparationError,
    PreparedGeometry,
    centers_respect_contact_radius,
    points_inside_volume,
    prepare_geometry,
)
from .integrators import (
    EXPONENTIAL_MIDPOINT_ENCLOSURE_REVISION,
    EXPONENTIAL_MIDPOINT_REVISION,
    RK4_DENSE_PATH_REVISION,
    RK4_ENCLOSURE_REVISION,
    STEP_PROPOSAL_REVISION,
)
from .lifecycle import ACTIVE, ESCAPED, FAILED, HELD, PENDING, STUCK
from .output import (
    CHECKPOINT_SCHEMA_VERSION,
    RESULT_ALGORITHM_REVISION,
    RESULT_SCHEMA_VERSION,
    RESULT_WRITER_RESERVE_BYTES,
    CheckpointState,
    FinalParticles,
    LifecycleSeries,
    ProbeFrame,
    ReleaseEvents,
    ResultBoundaryCondition,
    ResultContext,
    ResultPeriodicPair,
    ResultSource,
    ResultTopology,
    ResultWriter,
    RunSummary,
    TrajectoryFrame,
    make_result_context,
    result_context_array_bytes,
)
from .physics.catalog import (
    PHYSICS_CATALOG_REVISION,
    SUPPORTED_INTEGRATORS,
    PhysicsConfigurationError,
    PhysicsPlan,
    compatible_integrators,
    resolve_physics_plan,
)
from .physics.forces import (
    PhysicsEvaluationError,
)
from .physics.runtime import (
    PHYSICS_RUNTIME_REVISION,
    PhysicsRuntime,
    PhysicsRuntimeWorkspace,
    PrimitiveRange,
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
)
from .time_grid import MACRO_GRID_REVISION, MacroGrid
from .topology import (
    TOPOLOGY_ALGORITHM_REVISION,
    PreparedPeriodicTopology,
    TopologyPreparationError,
    TranslationPairRequest,
    prepare_periodic_topology,
)
from .transport import (
    _FAILURE_REASON_NAMES,
    EngineError,
    NumericalWorkPolicy,
    PreparedTransport,
    TransportCapabilities,
    TransportWorkspaceBudget,
    _ApplicabilityCertificateWorkspace,
    _CurvedWavefront,
    _EventStatistics,
    _ExactWavefront,
    _ProposalReplayWorkspace,
    _ReplayBuffer,
    _StageDynamics,
    resolve_transport_capabilities,
)

type _PathInput = str | PathLike[str]


ENGINE_ALGORITHM_REVISION = "particle_engine_v64"


COMPILED_CPU_TILE_REVISION = "compiled_cpu_tile_v31"


DURABLE_COMMIT_CADENCE_REVISION = "cumulative_solver_work_v1"


BROWNIAN_TREE_POLICY_REVISION = "conditional_boundary_refinement_absolute_target_v2"


_PREPARE_SCAN_BATCH_SIZE = 65_536


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


@dataclass(slots=True)
class _RunState(ParticleState):
    """Accepted physical state with parent-owned clocks and cursors."""

    active_index: ActiveIndex
    release_cursor: int
    frame_cursor: int
    probe_cursor: int
    macro_step_count: int
    macro_start_s: float
    event_statistics: _EventStatistics


@dataclass(frozen=True, slots=True)
class _RunFacts:
    """Prepared scientific/numerical identity, separate from resources and statistics."""

    algorithm_revisions: tuple[tuple[str, object], ...]
    coordinate_system: str
    motion_mode: str
    integrator: str
    backend: str
    path_kind: str
    models: tuple[tuple[str, tuple[tuple[str, str | float], ...]], ...]
    brownian: Mapping[str, object]

    def revision_document(self) -> dict[str, object]:
        return dict(self.algorithm_revisions)

    def model_document(self) -> dict[str, dict[str, str | float]]:
        return {category: dict(parameters) for category, parameters in self.models}


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    """Parent-owned input provenance and facts beside the worker transport."""

    case: SimulationCase
    transport: PreparedTransport
    facts: _RunFacts


def run_simulation(case: SimulationCase, output: _PathInput, *, workers: int = 1) -> RunSummary:
    """Run or resume the selected production profile and publish one complete result."""

    run = _prepare(case, workers=workers)
    prepared = run.transport
    schedule = prepared.schedule
    count = schedule.particle_count
    commit_work_threshold = _durable_commit_work_threshold(count)
    with ResultWriter(output, count, _resume_identity(run)) as writer:
        if writer.completed_summary is not None:
            return writer.completed_summary
        if not writer.context_initialized:
            writer.initialize_context(_result_context(case))
        state = _restore_or_initialize_run_state(prepared, writer.resume_state)
        epoch_start_work = _durable_commit_work(state)
        epoch_open = False
        try:
            with ProcessTeam(prepared, state, state.active_index, state.event_statistics) as team:
                prepared = team.prepared
                run = replace(run, transport=prepared)
                macro_grid = MacroGrid(
                    case.spec.time.start_s,
                    case.spec.time.end_s,
                    case.spec.time.dt_s,
                    prepared.field_time_split_s,
                )
                while state.macro_start_s < case.spec.time.end_s:
                    if not epoch_open:
                        writer.begin_epoch()
                        epoch_open = True
                    macro_start = state.macro_start_s
                    macro_end = macro_grid.next_end(state.macro_step_count, macro_start)
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
                        run,
                        state.lifecycle,
                        state.failure_reason_code,
                        state.event_statistics,
                    ),
                    macro_step_count=state.macro_step_count,
                )
        except ValueError as error:
            raise EngineError(str(error)) from error


def _result_context(case: SimulationCase) -> ResultContext:
    """Bind immutable result interpretation without copying fields or sources."""
    specification = case.spec.topology
    topology = (
        None
        if specification is None
        else ResultTopology(
            specification.model,
            specification.field_match_rtol,
            tuple(
                ResultPeriodicPair(
                    pair.first_boundary_group,
                    pair.second_boundary_group,
                    pair.first_to_second_m,
                )
                for pair in specification.pairs
            ),
        )
    )
    return make_result_context(
        case.data,
        case_file_hash=case.case_file_hash,
        data_content_hash=case.content_hash,
        motion_mode=case.spec.motion.mode,
        sources=tuple(
            ResultSource(index, source.name, str(source.parameters["table"]), source.kind)
            for index, source in enumerate(case.spec.sources)
        ),
        boundaries=tuple(
            ResultBoundaryCondition(
                rule.boundary_group,
                rule.priority,
                rule.law,
                rule.contact_geometry,
                json.dumps(rule.parameters, default=dict, sort_keys=True, allow_nan=False),
            )
            for rule in case.spec.boundaries
        ),
        topology=topology,
    )


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


def _resolve_run_facts(case: SimulationCase, prepared: PreparedTransport) -> _RunFacts:
    """Resolve active algorithm/model facts once, at the end of preparation."""

    integrator = prepared.solver.integrator
    brownian = _brownian_provenance(prepared.physics)
    uses_rk4_dense = prepared.capabilities.dense_rk4
    revisions = {
        "engine_algorithm_revision": ENGINE_ALGORITHM_REVISION,
        "compiled_cpu_tile_revision": COMPILED_CPU_TILE_REVISION,
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
        "macro_grid_revision": MACRO_GRID_REVISION,
        "required_field_revision": REQUIRED_FIELD_REVISION,
        "geometry_algorithm_revision": GEOMETRY_ALGORITHM_REVISION,
        "event_algorithm_revision": EVENT_ALGORITHM_REVISION,
        "topology_algorithm_revision": (
            TOPOLOGY_ALGORITHM_REVISION if prepared.topology is not None else None
        ),
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
    }
    return _RunFacts(
        tuple(revisions.items()),
        case.data.coordinate_system,
        case.spec.motion.mode,
        integrator,
        prepared.solver.backend,
        prepared.capabilities.path_kind,
        tuple(
            (category, tuple(model.items()))
            for category, model in prepared.physics.resolved_models().items()
        ),
        MappingProxyType(brownian),
    )


def _resume_identity(run: _PreparedRun) -> dict[str, object]:
    """Project prepared facts with source identity and persistence/layout owners."""

    case, prepared, facts = run.case, run.transport, run.facts
    particle_bytes = memoryview(prepared.schedule.particle_id).cast("B")
    return {
        **facts.revision_document(),
        "case_file_hash": case.case_file_hash,
        "data_content_hash": case.content_hash,
        "case_schema_version": CASE_FORMAT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "result_algorithm_revision": RESULT_ALGORITHM_REVISION,
        "durable_commit_cadence": _durable_commit_cadence(prepared.schedule.particle_count),
        "cpu_runtime_layout_revision": CPU_RUNTIME_LAYOUT_REVISION,
        "memory_plan_revision": MEMORY_PLAN_REVISION,
        "field_time_split_s": list(prepared.field_time_split_s),
        "brownian_coefficient_policy": facts.brownian["coefficient_policy"],
        "brownian_interval_tree_depth": facts.brownian["interval_tree_depth"],
        "brownian_adaptive_max_depth": facts.brownian["adaptive_max_depth"],
        "coordinate_system": facts.coordinate_system,
        "motion_mode": facts.motion_mode,
        "resolved_integrator": facts.integrator,
        "resolved_backend": facts.backend,
        "resolved_physics_models": facts.model_document(),
        "particle_count": prepared.schedule.particle_count,
        "particle_id_sha256": hashlib.sha256(particle_bytes).hexdigest(),
    }


def _restore_or_initialize_run_state(
    prepared: PreparedTransport,
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
        **checkpoint.particles.columns(),
        active=active,
        active_index=active_index,
        release_cursor=checkpoint.release_cursor,
        frame_cursor=checkpoint.frame_cursor,
        probe_cursor=checkpoint.probe_cursor,
        macro_step_count=checkpoint.macro_step_count,
        macro_start_s=checkpoint.macro_time_s,
        event_statistics=_EventStatistics(
            accepted_particle_pieces=checkpoint.accepted_particle_pieces,
            candidate_queries=checkpoint.candidate_queries,
            refinements=checkpoint.refinements,
            maximum_refinement_depth=checkpoint.maximum_refinement_depth,
            wall_interactions=checkpoint.wall_interactions,
            residual_splits=checkpoint.residual_splits,
            axis_crossings=checkpoint.axis_crossings,
        ),
    )


def _initial_run_state(prepared: PreparedTransport) -> _RunState:
    schedule = prepared.schedule
    count = schedule.particle_count
    particles = ParticleState.initialize(
        schedule.position_m,
        schedule.velocity_m_s,
        schedule.charge_number,
        schedule.release_time_s,
        schedule.source_facet_id,
        pending_lifecycle=PENDING,
        pending_surface=int(SURFACE_STATE_PENDING),
        resolved_surface=int(SURFACE_STATE_RESOLVED),
    )
    return _RunState(
        **particles.columns(),
        active_index=ActiveIndex.allocate(count),
        release_cursor=0,
        frame_cursor=0,
        probe_cursor=0,
        macro_step_count=0,
        macro_start_s=prepared.time.start_s,
        event_statistics=_EventStatistics(),
    )


def _validate_checkpoint_layout(checkpoint: CheckpointState, count: int) -> None:
    try:
        checkpoint.particles.validate(count)
    except ValueError as error:
        raise EngineError(str(error)) from error
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
    if not np.array_equal(active, checkpoint.particles.lifecycle == ACTIVE):
        raise EngineError("checkpoint active index does not match particle lifecycle")
    active_index = ActiveIndex.allocate(count)
    active_index.add(rows)
    return active, active_index


def _restore_field_cell_hint(
    prepared: PreparedTransport,
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
    prepared: PreparedTransport,
    checkpoint: CheckpointState,
) -> None:
    time = prepared.time
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


def _checkpoint_state(prepared: PreparedTransport, state: _RunState) -> CheckpointState:
    statistics = state.event_statistics
    return CheckpointState(
        macro_time_s=state.macro_start_s,
        macro_step_count=state.macro_step_count,
        release_cursor=state.release_cursor,
        frame_cursor=state.frame_cursor,
        probe_cursor=state.probe_cursor,
        particles=state.persistent(),
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


def _validate_completed_run_state(prepared: PreparedTransport, state: _RunState) -> None:
    if state.frame_cursor != len(prepared.frame_times_s):
        raise EngineError("not every requested trajectory frame was emitted")
    if state.probe_cursor != len(prepared.probe_times_s):
        raise EngineError("not every requested particle probe was emitted")
    if state.release_cursor != prepared.schedule.particle_count or bool(
        (state.lifecycle == PENDING).any()
    ):
        raise EngineError("not every validated particle was released by the run end")


def _prepare(case: SimulationCase, *, workers: int = 1) -> _PreparedRun:
    validate_worker_count(workers)
    spec = case.spec
    work_policy = NumericalWorkPolicy.resolve(spec.solver.event)
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
            result_context_bytes=result_context_array_bytes(case.data),
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
                item.value_constraint,
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
        field_time_split_s = MacroGrid.resolve(
            spec.time.start_s, spec.time.end_s, spec.time.dt_s, fields.time_knots_s()
        ).snapshot_splits_s
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
        capabilities = resolve_transport_capabilities(
            physics,
            spec.solver.integrator,
            constant_acceleration,
            case.data.coordinate_system,
            geometry.facet_count,
        )
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
        capabilities,
        work_policy,
        frame_times,
        probe_times,
        probe_particle_index,
        memory_limit_bytes,
        workers,
    )
    stage_capacity = memory_plan.slab_particles * int(capabilities.stage_evaluation)
    dynamics = _StageDynamics(
        runtime,
        fields,
        case.data.coordinate_system,
        (np.full(schedule.particle_count, -1, dtype="<i8") if field_cell_hint_bytes else None),
        fields.allocate_workspace(stage_capacity),
        PhysicsRuntimeWorkspace.allocate(stage_capacity),
        field_range_workspace=fields.allocate_range_workspace(
            memory_plan.slab_particles * int(capabilities.local_ranges)
        ),
    )
    dense_certificate = capabilities.dense_rk4
    exact_enabled = capabilities.exact_motion
    event_paths = capabilities.event_paths
    curved_workspace = _CurvedWavefront.allocate(
        memory_plan.slab_particles,
        work_policy.residual_stack_capacity,
        capabilities.stochastic,
        exact_enabled=exact_enabled,
        event_paths=event_paths,
    )
    dynamics.prepare_advance_workspaces(
        dense_enabled=dense_certificate,
        exact_enabled=exact_enabled,
    )
    certificate_workspace = _ApplicabilityCertificateWorkspace.allocate(
        memory_plan.slab_particles if dense_certificate else 0,
        work_policy.certificate_max_subdivisions,
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
    boundary_row_capacity = memory_plan.slab_particles * int(capabilities.material_walls)
    prepared = PreparedTransport(
        spec.time,
        spec.solver,
        case.data.coordinate_system,
        work_policy,
        capabilities,
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
            work_policy.residual_stack_capacity,
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
    return _PreparedRun(case, prepared, _resolve_run_facts(case, prepared))


def _prepare_memory_plan(
    case: SimulationCase,
    schedule: ParticleSchedule,
    geometry: PreparedGeometry,
    event_geometry: PreparedGeometry,
    topology: PreparedPeriodicTopology | None,
    fields: PreparedFieldSet,
    runtime: PhysicsRuntime,
    physics: PhysicsPlan,
    capabilities: TransportCapabilities,
    work_policy: NumericalWorkPolicy,
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
    event_paths = capabilities.event_paths
    exact_specialization = capabilities.exact_motion
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
    workspace_budget = TransportWorkspaceBudget.resolve(
        case.data.coordinate_system,
        work_policy,
        schedule,
        fields,
        physics,
        capabilities,
        observations=bool(frame_times_s or probe_times_s),
        has_point_event_view=case.spec.topology is not None
        or any(rule.contact_geometry == "particle_center" for rule in case.spec.boundaries),
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
            result_context_bytes=result_context_array_bytes(case.data),
            requires_stage_evaluation=capabilities.stage_evaluation,
            # One exclusive fixed arena serves exact or curved queries. Exact
            # reuses its point CSR bank for mixed contact, with four C columns.
            geometry_query_scratch_bytes=(
                candidate_columns * event_candidate_capacity * np.dtype("<i8").itemsize
                + arena_offset_bytes
            ),
            # One reusable native stage bank: int64 accepted hints and
            # uint8 pre-physics status; RZ additionally owns x/v and basis sign.
            stage_composition_work_bytes_per_particle=workspace_budget.stage,
            proposal_replay_work_bytes_per_particle=workspace_budget.replay,
            dense_path_bytes_per_particle=workspace_budget.dense_path,
            stochastic_tree_work_bytes_per_particle=workspace_budget.stochastic_tree,
            event_work_bytes_per_particle=workspace_budget.events,
            certificate_work_bytes_per_particle=workspace_budget.certificates,
            release_work_bytes_per_particle=workspace_budget.release,
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
    if spec.solver.backend != "cpu" or spec.solver.integrator not in SUPPORTED_INTEGRATORS:
        raise EngineError(
            "the engine supports integrator=rk4_fixed, exponential_midpoint, or "
            "ou_langevin with backend=cpu only"
        )
    if spec.solver.event.corner_policy != "priority_then_combined_normal_v1":
        raise EngineError("corner_policy must be priority_then_combined_normal_v1")
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

    if case.spec.solver.integrator not in compatible_integrators(physics):
        raise EngineError(
            "integrator=ou_langevin and a supported physics.noise model must be selected together"
        )


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
                    f"table particle {particle_id} must start strictly inside the particle domain"
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
        lifecycle[released] = ACTIVE
    return released, stop


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
        unavailable = ((lifecycle[rows] == ESCAPED) | (lifecycle[rows] == FAILED)) & (
            terminal_time_s[rows] <= requested_time_s
        )
        eligible = (schedule.release_time_s[rows] <= requested_time_s) & ~unavailable
        retained = ((lifecycle[rows] == STUCK) | (lifecycle[rows] == HELD)) & (
            terminal_time_s[rows] <= requested_time_s
        )
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
        pending=np.asarray([np.count_nonzero(lifecycle == PENDING)], dtype="<u8"),
        active=np.asarray([np.count_nonzero(lifecycle == ACTIVE)], dtype="<u8"),
        stuck=np.asarray([np.count_nonzero(lifecycle == STUCK)], dtype="<u8"),
        held=np.asarray([np.count_nonzero(lifecycle == HELD)], dtype="<u8"),
        escaped=np.asarray([np.count_nonzero(lifecycle == ESCAPED)], dtype="<u8"),
        failed=np.asarray([np.count_nonzero(lifecycle == FAILED)], dtype="<u8"),
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
        (lifecycle != ESCAPED) & (lifecycle != FAILED),
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


def _resolved_boundary_laws(
    case: SimulationCase, prepared: PreparedTransport
) -> list[dict[str, object]]:
    """Describe the resolved response and geometric meaning of each boundary."""

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
    run: _PreparedRun,
    lifecycle: np.ndarray,
    failure_reason_code: np.ndarray,
    event_statistics: _EventStatistics,
) -> dict[str, object]:
    case, prepared, facts = run.case, run.transport, run.facts
    spec = case.spec
    brownian = facts.brownian
    lifecycle_counts = {
        "pending": int(np.count_nonzero(lifecycle == PENDING)),
        "active": int(np.count_nonzero(lifecycle == ACTIVE)),
        "stuck": int(np.count_nonzero(lifecycle == STUCK)),
        "held": int(np.count_nonzero(lifecycle == HELD)),
        "escaped": int(np.count_nonzero(lifecycle == ESCAPED)),
        "failed": int(np.count_nonzero(lifecycle == FAILED)),
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
        "data_coordinate_system": facts.coordinate_system,
        "motion_mode": facts.motion_mode,
        **facts.revision_document(),
        "durable_commit_cadence": _durable_commit_cadence(prepared.schedule.particle_count),
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
            "integrator": facts.integrator,
            "backend": facts.backend,
            "path_kind": facts.path_kind,
            "physics_models": facts.model_document(),
            "brownian_coefficient_policy": brownian["coefficient_policy"],
            "sources": [
                {"source_id": index, "name": source.name, "type": source.kind}
                for index, source in enumerate(spec.sources)
            ],
            "boundary_laws": _resolved_boundary_laws(case, prepared),
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
        "numerical_work_policy": prepared.work_policy.as_manifest(),
        "event_refinement": (
            {
                "accepted_particle_pieces": event_statistics.accepted_particle_pieces,
                "candidate_queries": event_statistics.candidate_queries,
                "refinements": event_statistics.refinements,
                "maximum_refinement_depth": event_statistics.maximum_refinement_depth,
            }
            if prepared.capabilities.curved_events
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
        "prepared_stability_upper_bounds": {
            "requested_base_dt_s": prepared.time.dt_s,
            "dt_times_drag_velocity_lipschitz": prepared.maximum_dt_over_tau,
            "dt_times_charge_lipschitz": prepared.maximum_dt_charge_lipschitz,
            "meaning": "Requested base step times prepared full-domain bounds; not observed accepted-step maxima or accuracy estimates.",
        },
        "memory_plan": prepared.memory_plan.as_manifest(),
    }
