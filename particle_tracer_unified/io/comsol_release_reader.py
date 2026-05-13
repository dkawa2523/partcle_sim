from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.datamodel import ParticleTable


REQUIRED_COLUMNS = (
    "particle_id",
    "release_time",
    "x",
    "y",
    "vx",
    "vy",
    "mass",
    "diameter",
    "density",
    "charge",
)


_OPTIONAL_3D_COLUMNS = ("z", "vz")


@dataclass(frozen=True)
class ComsolReleaseParticle:
    particle_id: int
    release_time: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float
    diameter: float
    density: float
    charge: float
    weight: float
    source_entity_dim: int | None
    source_entity_id: int | None
    source_selection: str | None
    material_id: str | None
    species_id: str | None


def _float(row: dict[str, str], key: str, default: float | None = None) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError(f"Missing required float column {key!r}")
        return float(default)
    return float(value)


def _int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def read_comsol_release_particles(
    path: str | Path,
    *,
    coordinate_scale_m_per_model_unit: float,
    release_velocity_scale_mps_per_input_unit: float = 1.0,
    spatial_dim: int | None = None,
    strict: bool = True,
) -> list[ComsolReleaseParticle]:
    release_path = Path(path)
    scale = float(coordinate_scale_m_per_model_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale_m_per_model_unit must be a positive finite value")
    velocity_scale = float(release_velocity_scale_mps_per_input_unit)
    if not np.isfinite(velocity_scale) or velocity_scale <= 0.0:
        raise ValueError("release_velocity_scale_mps_per_input_unit must be a positive finite value")
    with release_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if strict and spatial_dim is not None and int(spatial_dim) == 3:
            missing.extend(name for name in _OPTIONAL_3D_COLUMNS if name not in columns)
        if strict and missing:
            raise ValueError(f"{release_path} is missing required columns: {missing}")

        particles: list[ComsolReleaseParticle] = []
        seen: set[int] = set()
        for line_no, row in enumerate(reader, start=2):
            pid = int(row["particle_id"])
            if pid in seen:
                raise ValueError(f"Duplicate particle_id={pid} in {release_path}:{line_no}")
            seen.add(pid)
            particle = ComsolReleaseParticle(
                particle_id=pid,
                release_time=_float(row, "release_time"),
                x=_float(row, "x") * scale,
                y=_float(row, "y") * scale,
                z=_float(row, "z", 0.0) * scale,
                vx=_float(row, "vx") * velocity_scale,
                vy=_float(row, "vy") * velocity_scale,
                vz=_float(row, "vz", 0.0) * velocity_scale,
                mass=_float(row, "mass"),
                diameter=_float(row, "diameter"),
                density=_float(row, "density"),
                charge=_float(row, "charge"),
                weight=_float(row, "weight", 1.0),
                source_entity_dim=_int(row, "source_entity_dim"),
                source_entity_id=_int(row, "source_entity_id"),
                source_selection=row.get("source_selection") or None,
                material_id=row.get("material_id") or None,
                species_id=row.get("species_id") or None,
            )
            numeric_values = (
                particle.release_time,
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
                particle.mass,
                particle.diameter,
                particle.density,
                particle.charge,
                particle.weight,
            )
            if not all(np.isfinite(value) for value in numeric_values):
                raise ValueError(f"release table contains non-finite numeric values at {release_path}:{line_no}")
            if particle.release_time < 0.0:
                raise ValueError(f"release_time must be non-negative at {release_path}:{line_no}")
            if particle.mass <= 0.0 or particle.diameter <= 0.0 or particle.density <= 0.0:
                raise ValueError(f"mass, diameter and density must be positive at {release_path}:{line_no}")
            particles.append(particle)
    return particles


def _material_ids(particles: Iterable[ComsolReleaseParticle]) -> np.ndarray:
    values = []
    for particle in particles:
        raw = particle.material_id
        try:
            values.append(int(raw) if raw is not None and str(raw).strip() else 0)
        except ValueError:
            values.append(0)
    return np.asarray(values, dtype=np.int64)


def comsol_release_particles_to_particle_table(
    particles: list[ComsolReleaseParticle],
    *,
    spatial_dim: int,
    metadata: dict[str, object] | None = None,
) -> ParticleTable:
    dim = int(spatial_dim)
    if dim not in {2, 3}:
        raise ValueError("spatial_dim must be 2 or 3")
    count = len(particles)
    position3 = np.asarray([[p.x, p.y, p.z] for p in particles], dtype=np.float64)
    velocity3 = np.asarray([[p.vx, p.vy, p.vz] for p in particles], dtype=np.float64)
    source_part_id = np.asarray(
        [0 if p.source_entity_id is None else int(p.source_entity_id) for p in particles],
        dtype=np.int64,
    )
    event_tag = np.asarray([p.source_selection or "comsol_release" for p in particles], dtype=object)
    return ParticleTable(
        spatial_dim=dim,
        particle_id=np.asarray([p.particle_id for p in particles], dtype=np.int64),
        position=position3[:, :dim],
        velocity=velocity3[:, :dim],
        release_time=np.asarray([p.release_time for p in particles], dtype=np.float64),
        mass=np.asarray([p.mass for p in particles], dtype=np.float64),
        diameter=np.asarray([p.diameter for p in particles], dtype=np.float64),
        density=np.asarray([p.density for p in particles], dtype=np.float64),
        charge=np.asarray([p.charge for p in particles], dtype=np.float64),
        source_part_id=source_part_id,
        material_id=_material_ids(particles),
        source_event_tag=event_tag,
        source_law_override=np.full(count, "", dtype=object),
        source_speed_scale_override=np.full(count, np.nan, dtype=np.float64),
        stick_probability=np.zeros(count, dtype=np.float64),
        dep_particle_rel_permittivity=np.full(count, np.nan, dtype=np.float64),
        thermophoretic_coeff=np.full(count, np.nan, dtype=np.float64),
        metadata=dict(metadata or {}),
    )


__all__ = (
    "ComsolReleaseParticle",
    "REQUIRED_COLUMNS",
    "comsol_release_particles_to_particle_table",
    "read_comsol_release_particles",
)
