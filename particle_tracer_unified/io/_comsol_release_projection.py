"""Apply the explicit COMSOL 2D release-boundary repair."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from particle_tracer_unified.core.boundary_numerics import (
    scaled_classification_tolerance,
)
from particle_tracer_unified.core.datamodel import ParticleTable
from particle_tracer_unified.core.geometry2d import build_boundary_loops_2d


@dataclass(frozen=True)
class _ProjectionGeometry2D:
    part_ids: np.ndarray
    loops: tuple[np.ndarray, ...]
    starts: np.ndarray
    vectors: np.ndarray
    length_sq: np.ndarray
    exact_tolerance: float


def _load_projection_geometry(path: Path) -> _ProjectionGeometry2D:
    with np.load(path) as payload:
        required = {"boundary_edges", "boundary_edge_part_ids"}
        if not required.issubset(payload.files):
            raise ValueError(
                "2D COMSOL release projection requires boundary_edges and "
                "boundary_edge_part_ids"
            )
        edges = np.asarray(payload["boundary_edges"], dtype=np.float64)
        part_ids = np.asarray(payload["boundary_edge_part_ids"], dtype=np.int64)
    if (
        edges.ndim != 3
        or edges.shape[1:] != (2, 2)
        or part_ids.shape != (edges.shape[0],)
    ):
        raise ValueError("invalid 2D boundary arrays for COMSOL release projection")

    loops = build_boundary_loops_2d(edges)
    if not loops:
        raise ValueError("COMSOL release projection requires closed 2D boundary loops")
    starts = edges[:, 0, :]
    vectors = edges[:, 1, :] - starts
    length_sq = np.einsum("ij,ij->i", vectors, vectors)
    edge_lengths = np.linalg.norm(vectors, axis=1)
    positive = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
    if positive.size != edge_lengths.size:
        raise ValueError(
            "COMSOL geometry contains a non-finite or degenerate boundary segment"
        )
    _, exact_tolerance = scaled_classification_tolerance(
        edges,
        float(np.min(positive)),
    )
    return _ProjectionGeometry2D(
        part_ids=part_ids,
        loops=loops,
        starts=starts,
        vectors=vectors,
        length_sq=length_sq,
        exact_tolerance=exact_tolerance,
    )


def _nearest_segments(
    point: np.ndarray,
    geometry: _ProjectionGeometry2D,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.clip(
        np.einsum(
            "ij,ij->i",
            point[None, :] - geometry.starts,
            geometry.vectors,
        )
        / geometry.length_sq,
        0.0,
        1.0,
    )
    projections = geometry.starts + alpha[:, None] * geometry.vectors
    return np.linalg.norm(projections - point[None, :], axis=1), projections


def _boundary_surface_target(
    *,
    particle_id: int,
    source_part_id: int,
    point: np.ndarray,
    geometry: _ProjectionGeometry2D,
    tolerance: float | None,
) -> np.ndarray | None:
    """Snap a boundary release onto the entity its release table declares.

    COMSOL releases inlet particles on the boundary itself, where the inlet
    feature overrides the wall condition.  Exported coordinates land within
    roundoff of that surface, so the only repair needed is to put the point
    exactly on its declared entity; the solver then treats a segment departing
    from its own boundary as a departure rather than a hit.
    """

    distances, projections = _nearest_segments(point, geometry)
    detect_tolerance = geometry.exact_tolerance if tolerance is None else tolerance
    if float(np.min(distances)) > detect_tolerance:
        return None
    if tolerance is None:
        raise ValueError(
            f"particle_id={particle_id} is on a geometry boundary; "
            "declare metadata.release_boundary_projection in the COMSOL manifest"
        )

    source_candidates = np.flatnonzero(geometry.part_ids == source_part_id)
    if source_candidates.size == 0:
        raise ValueError(
            f"particle_id={particle_id} source_part_id={source_part_id} "
            "is absent from geometry boundary parts"
        )
    source_index = int(source_candidates[np.argmin(distances[source_candidates])])
    if float(distances[source_index]) > tolerance:
        nearest_part = int(geometry.part_ids[int(np.argmin(distances))])
        raise ValueError(
            f"particle_id={particle_id} boundary provenance mismatch: "
            f"source_part_id={source_part_id}, nearest_part_id={nearest_part}"
        )
    return np.asarray(projections[source_index], dtype=np.float64)


def _detection_tolerance(config: Mapping[str, Any] | None) -> float | None:
    """Return how far from a boundary a release still counts as being on it."""

    return None if config is None else float(config["tolerance_m"])


def _project_release_particles_2d(
    particles: ParticleTable,
    *,
    geometry_path: Path,
    projection_config: Mapping[str, Any] | None,
) -> tuple[ParticleTable, dict[str, Any]]:
    geometry = _load_projection_geometry(geometry_path)
    tolerance = _detection_tolerance(projection_config)
    positions = np.array(particles.position, dtype=np.float64, copy=True)
    changed_ids: list[int] = []
    for index in range(particles.count):
        particle_id = int(particles.particle_id[index])
        target = _boundary_surface_target(
            particle_id=particle_id,
            source_part_id=int(particles.source_part_id[index]),
            point=positions[index],
            geometry=geometry,
            tolerance=tolerance,
        )
        if target is not None:
            positions[index] = target
            changed_ids.append(particle_id)

    return replace(particles, position=positions), {
        "enabled": projection_config is not None,
        "mode": "on_boundary_surface",
        "projected_count": len(changed_ids),
        "projected_particle_ids": changed_ids,
        "tolerance_m": tolerance,
    }


def apply_release_projection(
    particles: ParticleTable,
    *,
    spatial_dim: int,
    geometry_path: Path,
    projection_config: Mapping[str, Any] | None,
) -> tuple[ParticleTable, dict[str, Any]]:
    if int(spatial_dim) == 2:
        return _project_release_particles_2d(
            particles,
            geometry_path=geometry_path,
            projection_config=projection_config,
        )
    if projection_config is not None:
        raise ValueError(
            "release boundary projection is currently supported only for 2D "
            "COMSOL geometry"
        )
    return particles, {
        "enabled": False,
        "mode": "on_boundary_surface",
        "projected_count": 0,
        "projected_particle_ids": [],
    }


__all__ = ("apply_release_projection",)
