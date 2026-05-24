from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.datamodel import ParticleTable, PreparedRuntime
from .runtime_plan import ReleaseCursor, ReleaseSchedule, SolverPlan, build_release_schedule


@dataclass(frozen=True)
class ParticleStaticArrays:
    """Per-particle values that do not change during solver execution."""

    particle_id: np.ndarray
    release_time_s: np.ndarray
    mass_kg: np.ndarray
    diameter_m: np.ndarray
    density_kgm3: np.ndarray
    charge_initial_C: np.ndarray
    material_id: np.ndarray
    source_part_id: np.ndarray
    stick_probability: np.ndarray
    dep_particle_rel_permittivity: np.ndarray
    thermophoretic_coeff: np.ndarray
    flow_scale: np.ndarray
    drag_tau_scale: np.ndarray
    body_accel_scale: np.ndarray

    @property
    def count(self) -> int:
        return int(self.particle_id.size)


@dataclass
class ParticleState:
    """Mutable solver state arrays.

    Keep this class simple. It should not perform physics decisions; it only
    owns arrays that the step loop updates.
    """

    x: np.ndarray
    v: np.ndarray
    released: np.ndarray
    active: np.ndarray
    stuck: np.ndarray
    absorbed: np.ndarray
    contact_sliding: np.ndarray
    contact_endpoint_stopped: np.ndarray
    escaped: np.ndarray
    invalid_mask_stopped: np.ndarray
    numerical_boundary_stopped: np.ndarray
    invalid_stop_reason_code: np.ndarray
    charge_C: np.ndarray
    contact_part_id: np.ndarray
    contact_normal: np.ndarray
    contact_edge_index: np.ndarray
    valid_mask_status: np.ndarray

    @property
    def count(self) -> int:
        return int(self.x.shape[0])


@dataclass
class SolverArrays:
    static: ParticleStaticArrays
    state: ParticleState
    release_schedule: ReleaseSchedule
    release_cursor: ReleaseCursor = field(init=False)

    def __post_init__(self) -> None:
        self.release_cursor = ReleaseCursor(self.release_schedule)

    def active_indices(self) -> np.ndarray:
        return active_indices(self.state)

    def mobile_active_indices(self) -> np.ndarray:
        indices = self.active_indices()
        if indices.size == 0:
            return indices
        return indices[~self.state.contact_sliding[indices]]


def _ones(count: int) -> np.ndarray:
    return np.ones(int(count), dtype=np.float64)


def _array_or_default(value: Optional[np.ndarray], count: int, default: float = 1.0) -> np.ndarray:
    if value is None:
        return np.full(int(count), float(default), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == int(count):
        return arr.copy()
    out = np.full(int(count), float(default), dtype=np.float64)
    out[: min(arr.size, int(count))] = arr[: min(arr.size, int(count))]
    return out


def static_arrays_from_particles(
    particles: ParticleTable,
    plan: SolverPlan,
    resolved: object | None = None,
) -> ParticleStaticArrays:
    count = int(particles.count)
    metadata = getattr(particles, 'metadata', {}) or {}
    source_resolution = resolved
    if source_resolution is None:
        source_resolution = metadata.get('source_resolution') if isinstance(metadata, dict) else None

    flow_scale = _ones(count)
    drag_tau_scale = _ones(count)
    body_accel_scale = _ones(count)
    if source_resolution is not None:
        flow_scale = _array_or_default(getattr(source_resolution, 'physics_flow_scale', None), count, 1.0)
        drag_tau_scale = _array_or_default(getattr(source_resolution, 'physics_drag_tau_scale', None), count, 1.0)
        body_accel_scale = _array_or_default(getattr(source_resolution, 'physics_body_accel_scale', None), count, 1.0)

    return ParticleStaticArrays(
        particle_id=np.asarray(particles.particle_id, dtype=np.int64).copy(),
        release_time_s=np.asarray(particles.release_time, dtype=np.float64).copy(),
        mass_kg=np.asarray(particles.mass, dtype=np.float64).copy(),
        diameter_m=np.asarray(particles.diameter, dtype=np.float64).copy(),
        density_kgm3=np.asarray(particles.density, dtype=np.float64).copy(),
        charge_initial_C=np.asarray(particles.charge, dtype=np.float64).copy(),
        material_id=np.asarray(particles.material_id, dtype=np.int32).copy(),
        source_part_id=np.asarray(particles.source_part_id, dtype=np.int32).copy(),
        stick_probability=np.asarray(particles.stick_probability, dtype=np.float64).copy(),
        dep_particle_rel_permittivity=np.asarray(particles.dep_particle_rel_permittivity, dtype=np.float64).copy(),
        thermophoretic_coeff=np.asarray(particles.thermophoretic_coeff, dtype=np.float64).copy(),
        flow_scale=flow_scale,
        drag_tau_scale=drag_tau_scale,
        body_accel_scale=body_accel_scale,
    )


def state_from_particles(
    particles: ParticleTable,
    plan: SolverPlan,
    static: ParticleStaticArrays | None = None,
) -> ParticleState:
    count = int(particles.count)
    dim = int(plan.spatial_dim)
    x_raw = np.asarray(particles.position, dtype=np.float64)
    v_raw = np.asarray(particles.velocity, dtype=np.float64)
    if x_raw.ndim != 2 or x_raw.shape[0] != count or x_raw.shape[1] < dim:
        raise ValueError(f'particles.position must have shape (N, >= {dim})')
    if v_raw.ndim != 2 or v_raw.shape[0] != count or v_raw.shape[1] < dim:
        raise ValueError(f'particles.velocity must have shape (N, >= {dim})')
    x = x_raw[:, :dim].copy()
    v = v_raw[:, :dim].copy()
    return ParticleState(
        x=x,
        v=v,
        released=np.zeros(count, dtype=bool),
        active=np.zeros(count, dtype=bool),
        stuck=np.zeros(count, dtype=bool),
        absorbed=np.zeros(count, dtype=bool),
        contact_sliding=np.zeros(count, dtype=bool),
        contact_endpoint_stopped=np.zeros(count, dtype=bool),
        escaped=np.zeros(count, dtype=bool),
        invalid_mask_stopped=np.zeros(count, dtype=bool),
        numerical_boundary_stopped=np.zeros(count, dtype=bool),
        invalid_stop_reason_code=np.zeros(count, dtype=np.uint8),
        charge_C=(
            np.asarray(static.charge_initial_C, dtype=np.float64).copy()
            if static is not None
            else np.asarray(particles.charge, dtype=np.float64).copy()
        ),
        contact_part_id=np.zeros(count, dtype=np.int32),
        contact_normal=np.zeros((count, dim), dtype=np.float64),
        contact_edge_index=np.full(count, -1, dtype=np.int32),
        valid_mask_status=np.zeros(count, dtype=np.uint8),
    )


def initialize_solver_arrays(prepared: PreparedRuntime, plan: SolverPlan) -> SolverArrays:
    particles = prepared.runtime.particles
    if particles is None:
        raise ValueError('PreparedRuntime.runtime.particles is required')
    resolved = prepared.source_preprocess.resolved if prepared.source_preprocess is not None else None
    static = static_arrays_from_particles(particles, plan, resolved=resolved)
    state = state_from_particles(particles, plan, static=static)
    schedule = build_release_schedule(static.release_time_s)
    return SolverArrays(static=static, state=state, release_schedule=schedule)


def active_indices(state: ParticleState) -> np.ndarray:
    return np.flatnonzero(np.asarray(state.active, dtype=bool))


def activate_release_cursor_until(
    cursor: ReleaseCursor,
    released: np.ndarray,
    active: np.ndarray,
    t: float,
    *,
    tolerance_s: float = 1.0e-15,
) -> np.ndarray:
    """Activate finite-time releases due by ``t`` using an existing state mask.

    This mirrors the old vector expression:
    ``(~released) & isfinite(release_time) & (release_time <= t + tolerance)``.
    The schedule already excludes NaN release times and keeps stable particle
    order for equal release times.
    """

    schedule = cursor.schedule
    activated: list[int] = []
    threshold = float(t) + float(tolerance_s)
    while not cursor.done:
        particle_index = int(schedule.order[int(cursor.position)])
        release_time = float(schedule.release_time_s[particle_index])
        if release_time > threshold:
            break
        if not bool(released[particle_index]):
            released[particle_index] = True
            active[particle_index] = True
            activated.append(particle_index)
        cursor.position += 1
    if not activated:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(activated, dtype=np.int64)

