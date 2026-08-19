from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.datamodel import ParticleTable
from particle_tracer_unified.solvers.runtime_plan import ReleaseCursor, SolverPlan


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
    dep_particle_rel_permittivity: np.ndarray
    thermophoretic_coeff: np.ndarray

    @property
    def count(self) -> int:
        return int(self.particle_id.size)


def static_arrays_from_particles(
    particles: ParticleTable,
) -> ParticleStaticArrays:
    count = int(particles.count)

    mass_kg = np.asarray(particles.mass, dtype=np.float64)
    diameter_m = np.asarray(particles.diameter, dtype=np.float64)
    if mass_kg.shape != (count,) or np.any(~np.isfinite(mass_kg) | (mass_kg <= 0.0)):
        raise ValueError(
            "particles.mass_kg must contain one finite positive value per particle"
        )
    if diameter_m.shape != (count,) or np.any(
        ~np.isfinite(diameter_m) | (diameter_m <= 0.0)
    ):
        raise ValueError(
            "particles.drag_diameter_m must contain one finite positive value per "
            "particle"
        )

    return ParticleStaticArrays(
        particle_id=np.asarray(particles.particle_id, dtype=np.int64),
        release_time_s=np.asarray(particles.release_time, dtype=np.float64),
        mass_kg=mass_kg,
        diameter_m=diameter_m,
        density_kgm3=np.asarray(particles.density, dtype=np.float64),
        charge_initial_C=np.asarray(particles.charge, dtype=np.float64),
        material_id=np.asarray(particles.material_id, dtype=np.int32),
        source_part_id=np.asarray(particles.source_part_id, dtype=np.int32),
        dep_particle_rel_permittivity=np.asarray(
            particles.dep_particle_rel_permittivity, dtype=np.float64
        ),
        thermophoretic_coeff=np.asarray(
            particles.thermophoretic_coeff, dtype=np.float64
        ),
    )


def initial_position_velocity(
    particles: ParticleTable,
    plan: SolverPlan,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(particles.count)
    dim = int(plan.spatial_dim)
    x_raw = np.asarray(particles.position, dtype=np.float64)
    v_raw = np.asarray(particles.velocity, dtype=np.float64)
    if x_raw.ndim != 2 or x_raw.shape[0] != count or x_raw.shape[1] < dim:
        raise ValueError(f"particles.position must have shape (N, >= {dim})")
    if v_raw.ndim != 2 or v_raw.shape[0] != count or v_raw.shape[1] < dim:
        raise ValueError(f"particles.velocity must have shape (N, >= {dim})")
    x = x_raw[:, :dim].copy()
    v = v_raw[:, :dim].copy()
    return x, v


def activate_release_cursor_until(
    cursor: ReleaseCursor,
    released: np.ndarray,
    active: np.ndarray,
    t: float,
) -> np.ndarray:
    """Activate scheduled particles whose physical release time is due."""

    schedule = cursor.schedule
    activated: list[int] = []
    threshold = float(t)
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
