"""Resident particle arrays, bounded slabs and solver-owned CPU memory planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from types import MappingProxyType

import numpy as np
from numba import njit

CPU_RUNTIME_LAYOUT_REVISION = "resident_soa_process_slab_v14"
MEMORY_PLAN_REVISION = "solver_owned_memory_plan_v26"

_MAX_SLAB_PARTICLES = 65_536
_MINIMUM_SAFETY_MARGIN_BYTES = 64 * 1024


@njit(cache=True, fastmath=False, nogil=True)
def next_float64(value: float, toward_positive: bool) -> float:
    """Match nextafter(value, +/-inf), including signed zero and overflow.

    Scalar bitcasts are documented Numba operations.  NaNs retain the
    platform's original nextafter payload semantics.  The solver consumes
    values and explicit statuses, never C floating-point exception flags.
    """

    bits = np.float64(value).view(np.uint64)
    magnitude = bits & np.uint64(0x7FFFFFFFFFFFFFFF)
    if magnitude >= np.uint64(0x7FF0000000000000):
        if magnitude > np.uint64(0x7FF0000000000000):
            return np.nextafter(value, np.inf if toward_positive else -np.inf)
        if bool(bits >> np.uint64(63)) != toward_positive:
            return value
    if magnitude == 0:
        result = np.uint64(1) if toward_positive else np.uint64(0x8000000000000001)
        return np.uint64(result).view(np.float64)
    if bool(bits >> np.uint64(63)) == toward_positive:
        bits -= np.uint64(1)
    else:
        bits += np.uint64(1)
    return np.uint64(bits).view(np.float64)


@dataclass(slots=True)
class ActiveIndex:
    """Sorted resident-row indices for particles that still require work."""

    particle_index: np.ndarray
    count: int = 0

    @classmethod
    def allocate(cls, particle_count: int) -> ActiveIndex:
        if particle_count < 0:
            raise ValueError("particle_count must be nonnegative")
        return cls(np.empty(particle_count, dtype="<i8"))

    @property
    def rows(self) -> np.ndarray:
        """Return the sorted active resident rows without allocating."""

        return self.particle_index[: self.count]

    def add(self, released: np.ndarray) -> None:
        """Add one release cohort while preserving resident-ID order."""

        values = np.asarray(released, dtype=np.int64)
        if values.ndim != 1 or self.count + values.size > self.particle_index.size:
            raise ValueError("released particle rows exceed the active-index capacity")
        if not values.size:
            return
        stop = self.count + int(values.size)
        self.particle_index[self.count : stop] = values
        # Resident rows are unique, so an in-place quicksort gives the same
        # canonical order without the linear workspace of a stable integer sort.
        self.particle_index[:stop].sort(kind="quicksort")
        self.count = stop

    def compact(self, active: np.ndarray) -> None:
        """Remove terminal rows stably in place without a resident mask or gather."""

        rows = self.rows
        if active.shape != self.particle_index.shape:
            raise ValueError("active-state shape does not match the active index")
        if not rows.size:
            return
        self.count = _compact_active_rows_into(self.particle_index, self.count, active)


@njit(cache=True, fastmath=False, nogil=True)
def _compact_active_rows_into(indices: np.ndarray, count: int, active: np.ndarray) -> int:
    """Stable forward compaction; writes never overwrite an unread resident index."""

    write = 0
    for read in range(count):
        resident = indices[read]
        if active[resident]:
            indices[write] = resident
            write += 1
    return write


@dataclass(frozen=True, slots=True)
class CpuMemoryPlan:
    """Predicted peak of solver-owned arrays, not a process-RSS hard limit."""

    limit_bytes: int
    particle_count: int
    slab_particles: int
    scratch_bytes_per_particle: int
    stage_composition_work_bytes_per_particle: int
    proposal_replay_work_bytes_per_particle: int
    dense_path_bytes_per_particle: int
    stochastic_tree_work_bytes_per_particle: int
    event_work_bytes_per_particle: int
    certificate_work_bytes_per_particle: int
    release_work_bytes_per_particle: int
    replay_work_bytes: int
    event_candidate_capacity: int
    event_staging_capacity: int
    event_staging_bytes_per_row: int
    event_staging_fixed_bytes: int
    failure_staging_bytes_per_particle: int
    load_peak_bytes: int
    prepare_peak_bytes: int
    run_peak_bytes: int
    planned_bytes: int
    geometry_preparation_transient_bytes: int
    field_preparation_transient_bytes: int
    components: Mapping[str, int]
    worker_count: int = 1

    def active_slab_particles(self, active_count: int) -> int:
        """Retain bounded scratch while giving the current workers enough tasks."""

        _require_nonnegative(active_count)
        return _process_slab_particles(active_count, self.slab_particles, self.worker_count)

    def reserve_process_padding(self, padding_bytes: int) -> CpuMemoryPlan:
        """Include actual mapping granularity without removing dynamic work reserves."""
        _require_nonnegative(padding_bytes)
        components = dict(self.components)
        components["process_transport_page_padding"] = padding_bytes
        owned = sum(value for name, value in components.items() if name != "safety_margin")
        components["safety_margin"] = _safety_margin(owned)
        run_peak = sum(components.values())
        planned = max(self.load_peak_bytes, self.prepare_peak_bytes, run_peak)
        if planned > self.limit_bytes:
            raise ValueError("process mapping storage exceeds resources.memory_limit_mb")
        return replace(
            self,
            components=MappingProxyType(components),
            run_peak_bytes=run_peak,
            planned_bytes=planned,
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "revision": MEMORY_PLAN_REVISION,
            "runtime_layout_revision": CPU_RUNTIME_LAYOUT_REVISION,
            "semantics": "predicted peak of solver-owned arrays; not process RSS",
            "limit_bytes": self.limit_bytes,
            "planned_bytes": self.planned_bytes,
            "slab_particles": self.slab_particles,
            "worker_count": self.worker_count,
            "scratch_bytes_per_particle": self.scratch_bytes_per_particle,
            "stage_composition_work_bytes_per_particle": self.stage_composition_work_bytes_per_particle,
            "proposal_replay_work_bytes_per_particle": self.proposal_replay_work_bytes_per_particle,
            "dense_path_bytes_per_particle": self.dense_path_bytes_per_particle,
            "stochastic_tree_work_bytes_per_particle": (
                self.stochastic_tree_work_bytes_per_particle
            ),
            "event_work_bytes_per_particle": self.event_work_bytes_per_particle,
            "certificate_work_bytes_per_particle": (self.certificate_work_bytes_per_particle),
            "release_work_bytes_per_particle": self.release_work_bytes_per_particle,
            "replay_work_bytes": self.replay_work_bytes,
            "event_candidate_capacity": self.event_candidate_capacity,
            "event_staging_capacity": self.event_staging_capacity,
            "event_staging_bytes_per_row": self.event_staging_bytes_per_row,
            "event_staging_fixed_bytes": self.event_staging_fixed_bytes,
            "failure_staging_bytes_per_particle": (self.failure_staging_bytes_per_particle),
            "geometry_preparation_transient_bytes": (self.geometry_preparation_transient_bytes),
            "field_preparation_transient_bytes": self.field_preparation_transient_bytes,
            "phase_peaks": {
                "load_case": self.load_peak_bytes,
                "prepare": self.prepare_peak_bytes,
                "run": self.run_peak_bytes,
            },
            "components": dict(self.components),
        }


def early_memory_requirement_bytes(
    *,
    canonical_data_bytes: int,
    particle_count: int,
    requires_stage_evaluation: bool,
    writer_reserve_bytes: int,
    result_context_bytes: int = 0,
) -> int:
    """Return a phase-separated lower bound before source realization."""

    _require_nonnegative(
        canonical_data_bytes, particle_count, writer_reserve_bytes, result_context_bytes
    )
    schedule_bytes = particle_count * 128
    resident_state_bytes = ParticleState.bytes_for(particle_count)
    active_index_bytes = particle_count * 8
    # The early gate runs before model-specific runtime arrays exist, so it
    # must remain a guaranteed lower bound.  The final plan below uses the
    # realized 24/32/48-byte model footprint and makes the authoritative fit
    # decision.
    physics_runtime_bytes = particle_count * 16
    scratch_per_particle = _scratch_bytes_per_particle(requires_stage_evaluation)
    load_owned = canonical_data_bytes
    prepare_owned = canonical_data_bytes + 2 * schedule_bytes + physics_runtime_bytes
    run_owned = (
        canonical_data_bytes
        + schedule_bytes
        + resident_state_bytes
        + active_index_bytes
        + physics_runtime_bytes
        + writer_reserve_bytes
        + result_context_bytes
        + (scratch_per_particle if particle_count else 0)
    )
    return max(
        load_owned + _safety_margin(load_owned),
        prepare_owned + _safety_margin(prepare_owned),
        run_owned + _safety_margin(run_owned),
    )


def plan_cpu_memory(
    *,
    limit_bytes: int,
    particle_count: int,
    canonical_data_bytes: int,
    prepared_geometry_bytes: int,
    particle_schedule_bytes: int,
    physics_runtime_bytes: int,
    probe_index_bytes: int,
    output_buffer_bytes: int,
    replay_work_bytes: int = 0,
    writer_reserve_bytes: int,
    requires_stage_evaluation: bool,
    geometry_query_scratch_bytes: int = 0,
    stage_composition_work_bytes_per_particle: int = 0,
    proposal_replay_work_bytes_per_particle: int = 0,
    dense_path_bytes_per_particle: int = 0,
    stochastic_tree_work_bytes_per_particle: int = 0,
    event_work_bytes_per_particle: int = 0,
    certificate_work_bytes_per_particle: int = 0,
    release_work_bytes_per_particle: int = 0,
    event_candidate_capacity: int = 0,
    event_staging_capacity: int = 0,
    event_staging_bytes_per_row: int = 0,
    event_staging_fixed_bytes: int = 0,
    failure_staging_bytes_per_particle: int = 0,
    field_runtime_bytes: int = 0,
    geometry_preparation_transient_bytes: int = 0,
    field_preparation_transient_bytes: int = 0,
    worker_count: int = 1,
    result_context_bytes: int = 0,
) -> CpuMemoryPlan:
    """Choose one bounded slab under the configured memory ceiling."""

    _require_nonnegative(
        limit_bytes,
        particle_count,
        canonical_data_bytes,
        prepared_geometry_bytes,
        particle_schedule_bytes,
        field_runtime_bytes,
        geometry_preparation_transient_bytes,
        field_preparation_transient_bytes,
        physics_runtime_bytes,
        probe_index_bytes,
        output_buffer_bytes,
        replay_work_bytes,
        writer_reserve_bytes,
        geometry_query_scratch_bytes,
        stage_composition_work_bytes_per_particle,
        proposal_replay_work_bytes_per_particle,
        dense_path_bytes_per_particle,
        stochastic_tree_work_bytes_per_particle,
        event_work_bytes_per_particle,
        certificate_work_bytes_per_particle,
        release_work_bytes_per_particle,
        event_candidate_capacity,
        event_staging_capacity,
        event_staging_bytes_per_row,
        event_staging_fixed_bytes,
        failure_staging_bytes_per_particle,
        result_context_bytes,
    )
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise ValueError("worker_count must be a positive integer")
    resident_state_bytes = ParticleState.bytes_for(particle_count)
    active_index_bytes = particle_count * 8
    final_state_buffer_bytes = particle_count * 9
    fixed_components = {
        "canonical_data": canonical_data_bytes,
        "prepared_geometry": prepared_geometry_bytes,
        "particle_schedule": particle_schedule_bytes,
        "field_runtime": field_runtime_bytes,
        "resident_state": resident_state_bytes,
        "active_index": active_index_bytes,
        "physics_runtime": physics_runtime_bytes,
        "probe_index": probe_index_bytes,
        "output_buffer": output_buffer_bytes + final_state_buffer_bytes,
        "replay_work": replay_work_bytes,
        "geometry_query_scratch": geometry_query_scratch_bytes,
        "writer_reserve": writer_reserve_bytes,
        "result_context_validation": result_context_bytes,
    }
    # Shared transport coexists with the original prepared owners at startup.
    # Worker scratch is copied from one shared template; parent, template and
    # all workers remain in the conservative simultaneous-live accounting.
    copies = worker_count + 2 if worker_count > 1 else 1
    if worker_count > 1:
        fixed_components["process_shared_transport"] = sum(
            value
            for name, value in fixed_components.items()
            if name not in {"writer_reserve", "result_context_validation"}
        )
        fixed_components["process_private_query"] = (copies - 1) * geometry_query_scratch_bytes
        # Each worker may own one serialized handoff while its numeric staging
        # and canonical payload remain live. ACK credit bounds this to one
        # boundary/failure batch plus the particle-index command per worker.
        fixed_components["process_handoff_serialization"] = worker_count * event_staging_fixed_bytes
    fixed_component_bytes = sum(fixed_components.values())
    fixed_run_bytes = fixed_component_bytes + copies * event_staging_fixed_bytes
    scratch_per_particle = _scratch_bytes_per_particle(requires_stage_evaluation)
    slab_bytes_per_particle = (
        scratch_per_particle
        + stage_composition_work_bytes_per_particle
        + proposal_replay_work_bytes_per_particle
        + dense_path_bytes_per_particle
        + stochastic_tree_work_bytes_per_particle
        + event_work_bytes_per_particle
        + certificate_work_bytes_per_particle
        + release_work_bytes_per_particle
        + event_staging_bytes_per_row
        + failure_staging_bytes_per_particle
    )
    handoff_bytes_per_particle = (
        worker_count * (event_staging_bytes_per_row + failure_staging_bytes_per_particle + 8)
        if worker_count > 1
        else 0
    )
    maximum_slab = min(particle_count, _MAX_SLAB_PARTICLES)
    if worker_count > 1:
        maximum_slab = _process_slab_particles(
            particle_count, min(maximum_slab, 4096), worker_count
        )
    if event_staging_capacity:
        maximum_slab = min(maximum_slab, event_staging_capacity)
    slab_particles = _largest_fitting_slab(
        limit_bytes=limit_bytes,
        fixed_run_bytes=fixed_run_bytes,
        scratch_bytes_per_particle=copies * slab_bytes_per_particle + handoff_bytes_per_particle,
        maximum_slab=maximum_slab,
    )
    scratch_bytes = slab_particles * scratch_per_particle
    stage_composition_work_bytes = slab_particles * stage_composition_work_bytes_per_particle
    proposal_replay_work_bytes = slab_particles * proposal_replay_work_bytes_per_particle
    dense_path_bytes = slab_particles * dense_path_bytes_per_particle
    stochastic_tree_work_bytes = slab_particles * stochastic_tree_work_bytes_per_particle
    event_work_bytes = slab_particles * event_work_bytes_per_particle
    certificate_work_bytes = slab_particles * certificate_work_bytes_per_particle
    release_work_bytes = slab_particles * release_work_bytes_per_particle
    event_staging_bytes = event_staging_fixed_bytes + slab_particles * event_staging_bytes_per_row
    failure_staging_bytes = slab_particles * failure_staging_bytes_per_particle
    safety_margin_bytes = _safety_margin(
        fixed_component_bytes
        + scratch_bytes
        + stage_composition_work_bytes
        + proposal_replay_work_bytes
        + dense_path_bytes
        + stochastic_tree_work_bytes
        + event_work_bytes
        + certificate_work_bytes
        + release_work_bytes
        + event_staging_bytes
        + failure_staging_bytes
    )
    components = {
        **fixed_components,
        "slab_proposal_scratch": scratch_bytes,
        "slab_stage_composition_work": stage_composition_work_bytes,
        "slab_proposal_replay_work": proposal_replay_work_bytes,
        "slab_dense_path": dense_path_bytes,
        "slab_stochastic_tree_work": stochastic_tree_work_bytes,
        "slab_event_work": event_work_bytes,
        "slab_certificate_work": certificate_work_bytes,
        "slab_release_work": release_work_bytes,
        "slab_event_staging": event_staging_bytes,
        "slab_failure_staging": failure_staging_bytes,
        "safety_margin": safety_margin_bytes,
    }
    if worker_count > 1:
        worker_scratch = (copies - 1) * (
            scratch_bytes
            + stage_composition_work_bytes
            + proposal_replay_work_bytes
            + dense_path_bytes
            + stochastic_tree_work_bytes
            + event_work_bytes
            + certificate_work_bytes
            + release_work_bytes
            + event_staging_bytes
            + failure_staging_bytes
        )
        components["process_private_scratch_and_handoff"] = worker_scratch
        components["process_handoff_serialization"] += slab_particles * handoff_bytes_per_particle
        components["safety_margin"] = _safety_margin(
            sum(value for name, value in components.items() if name != "safety_margin")
        )
    run_peak_bytes = sum(components.values())
    load_owned = canonical_data_bytes
    load_peak_bytes = load_owned + _safety_margin(load_owned)
    prepare_owned = (
        canonical_data_bytes
        + prepared_geometry_bytes
        + 2 * particle_schedule_bytes
        + field_runtime_bytes
        + geometry_preparation_transient_bytes
        + field_preparation_transient_bytes
        + physics_runtime_bytes
    )
    prepare_peak_bytes = prepare_owned + _safety_margin(prepare_owned)
    planned_bytes = max(load_peak_bytes, prepare_peak_bytes, run_peak_bytes)
    return CpuMemoryPlan(
        limit_bytes=limit_bytes,
        particle_count=particle_count,
        slab_particles=slab_particles,
        scratch_bytes_per_particle=scratch_per_particle,
        stage_composition_work_bytes_per_particle=stage_composition_work_bytes_per_particle,
        proposal_replay_work_bytes_per_particle=proposal_replay_work_bytes_per_particle,
        dense_path_bytes_per_particle=dense_path_bytes_per_particle,
        stochastic_tree_work_bytes_per_particle=(stochastic_tree_work_bytes_per_particle),
        event_work_bytes_per_particle=event_work_bytes_per_particle,
        certificate_work_bytes_per_particle=certificate_work_bytes_per_particle,
        release_work_bytes_per_particle=release_work_bytes_per_particle,
        replay_work_bytes=replay_work_bytes,
        event_candidate_capacity=event_candidate_capacity,
        event_staging_capacity=event_staging_capacity,
        event_staging_bytes_per_row=event_staging_bytes_per_row,
        event_staging_fixed_bytes=event_staging_fixed_bytes,
        failure_staging_bytes_per_particle=failure_staging_bytes_per_particle,
        load_peak_bytes=load_peak_bytes,
        prepare_peak_bytes=prepare_peak_bytes,
        run_peak_bytes=run_peak_bytes,
        planned_bytes=planned_bytes,
        geometry_preparation_transient_bytes=geometry_preparation_transient_bytes,
        field_preparation_transient_bytes=field_preparation_transient_bytes,
        components=MappingProxyType(components),
        worker_count=worker_count,
    )


def _largest_fitting_slab(
    *,
    limit_bytes: int,
    fixed_run_bytes: int,
    scratch_bytes_per_particle: int,
    maximum_slab: int,
) -> int:
    """Solve the monotone slab-scratch memory inequality exactly."""

    lower = 0
    upper = maximum_slab
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        scratch_bytes = candidate * scratch_bytes_per_particle
        owned_bytes = fixed_run_bytes + scratch_bytes
        if owned_bytes + _safety_margin(owned_bytes) <= limit_bytes:
            lower = candidate
        else:
            upper = candidate - 1
    return lower


def _process_slab_particles(particle_count: int, capacity: int, worker_count: int) -> int:
    """Four dispatch rounds per worker, bounded by the one prepared scratch slab."""

    if worker_count == 1 or particle_count == 0:
        return min(particle_count, capacity)
    tasks = 4 * worker_count
    return min(capacity, max(1, (particle_count + tasks - 1) // tasks))


def _scratch_bytes_per_particle(requires_stage_evaluation: bool) -> int:
    # Includes proposal columns, RK stages/enclosures, row verdicts,
    # compiled-field temporaries, and the residual event wavefront's current
    # state, stack-top, departure, locator-result, and response columns.  The
    # depth-dependent stack and initial surface-release batch are separate
    # named components so configuration cannot escape the memory ceiling.
    return 2048 if requires_stage_evaluation else 384


def _safety_margin(owned_bytes: int) -> int:
    return max(_MINIMUM_SAFETY_MARGIN_BYTES, (owned_bytes + 7) // 8)


def _require_nonnegative(*values: int) -> None:
    if any(value < 0 for value in values):
        raise ValueError("memory-plan inputs must be nonnegative")


@dataclass(slots=True)
class PersistentParticleState:
    """Particle columns required to resume the same accepted numerical path.

    This declaration owns their storage layout as well as allocation metadata.
    The active mask is reconstructed from the checkpoint's sorted active index.
    """

    position_m: np.ndarray = dataclass_field(metadata={"dtype": "<f8", "width": 2, "finite": True})
    velocity_m_s: np.ndarray = dataclass_field(
        metadata={"dtype": "<f8", "width": 2, "finite": True}
    )
    charge_number: np.ndarray = dataclass_field(
        metadata={"dtype": "<f8", "width": 1, "finite": True}
    )
    lifecycle: np.ndarray = dataclass_field(metadata={"dtype": "<u1", "width": 1})
    terminal_time_s: np.ndarray = dataclass_field(metadata={"dtype": "<f8", "width": 1})
    event_ordinal: np.ndarray = dataclass_field(metadata={"dtype": "<u4", "width": 1})
    physical_boundary_event_ordinal: np.ndarray = dataclass_field(
        metadata={"dtype": "<u4", "width": 1}
    )
    exact_origin_time_s: np.ndarray = dataclass_field(
        metadata={"dtype": "<f8", "width": 1, "finite": True}
    )
    exact_origin_position_m: np.ndarray = dataclass_field(
        metadata={"dtype": "<f8", "width": 2, "finite": True}
    )
    exact_origin_velocity_m_s: np.ndarray = dataclass_field(
        metadata={"dtype": "<f8", "width": 2, "finite": True}
    )
    start_contact_state: np.ndarray = dataclass_field(metadata={"dtype": "<u1", "width": 1})
    failure_reason_code: np.ndarray = dataclass_field(metadata={"dtype": "<u2", "width": 1})

    def columns(self) -> dict[str, np.ndarray]:
        """Borrow exactly the persistent columns, without copying numeric arrays."""
        return {
            item.name: getattr(self, item.name)
            for item in dataclass_fields(PersistentParticleState)
        }

    def validate(self, count: int) -> None:
        """Require the declared checkpoint layouts and finite numerical state."""
        for item in dataclass_fields(PersistentParticleState):
            array = getattr(self, item.name)
            width = item.metadata["width"]
            shape = (count,) if width == 1 else (count, width)
            if array.shape != shape or array.dtype != np.dtype(item.metadata["dtype"]):
                raise ValueError(f"checkpoint particle column {item.name} has an invalid layout")
            if item.metadata.get("finite") and not bool(np.isfinite(array).all()):
                raise ValueError(f"checkpoint particle column {item.name} must be finite")


@dataclass(slots=True)
class ParticleState(PersistentParticleState):
    """The sole resident particle columns shared by serial execution and workers."""

    active: np.ndarray = dataclass_field(metadata={"dtype": "?", "width": 1})

    def persistent(self) -> PersistentParticleState:
        """Borrow an accepted-barrier snapshot for the synchronous checkpoint writer."""
        return PersistentParticleState(**PersistentParticleState.columns(self))

    @staticmethod
    def allocate(count: int) -> ParticleState:
        """Allocate exactly the finite resident columns declared above."""
        _require_nonnegative(count)
        columns = {}
        for item in dataclass_fields(ParticleState):
            width = item.metadata["width"]
            shape = (count,) if width == 1 else (count, width)
            columns[item.name] = np.empty(shape, dtype=item.metadata["dtype"])
        return ParticleState(**columns)

    @staticmethod
    def bytes_for(count: int) -> int:
        """Use the allocation declaration without allocating any particle rows."""
        _require_nonnegative(count)
        return count * sum(
            np.dtype(item.metadata["dtype"]).itemsize * item.metadata["width"]
            for item in dataclass_fields(ParticleState)
        )

    def columns(self) -> dict[str, np.ndarray]:
        """Borrow the same resident columns for the parent and each worker."""
        return {item.name: getattr(self, item.name) for item in dataclass_fields(ParticleState)}

    @staticmethod
    def initialize(
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
        release_time_s: np.ndarray,
        source_facet_id: np.ndarray,
        *,
        pending_lifecycle: int,
        pending_surface: int,
        resolved_surface: int,
    ) -> ParticleState:
        """Copy realized inputs once; every later update uses these columns."""
        state = ParticleState.allocate(charge_number.size)
        state.position_m[:] = position_m
        state.velocity_m_s[:] = velocity_m_s
        state.charge_number[:] = charge_number
        state.active.fill(False)
        state.lifecycle.fill(pending_lifecycle)
        state.terminal_time_s.fill(np.inf)
        state.event_ordinal.fill(0)
        state.physical_boundary_event_ordinal.fill(0)
        state.exact_origin_time_s[:] = release_time_s
        state.exact_origin_position_m[:] = position_m
        state.exact_origin_velocity_m_s[:] = velocity_m_s
        state.start_contact_state[:] = np.where(
            source_facet_id < 0, resolved_surface, pending_surface
        )
        state.failure_reason_code.fill(0)
        return state
