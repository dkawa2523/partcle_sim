from __future__ import annotations

from typing import Any

import numpy as np

from .core._triangle_surface import build_geometry_surfaces_3d
from .core.boundary_core import (
    inside_geometry_with_boundary,
    points_inside_geometry_2d,
)
from .core.boundary_hits import nearest_boundary_edge_features_2d
from .core.field_backend import (
    VALID_MASK_QUANTITY,
    ProviderSamplingBackend,
    field_backend_kind,
)
from .core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from .domain import FieldRequest

SUPPORT_STATUS_NAMES = {
    int(VALID_MASK_STATUS_CLEAN): "clean",
    int(VALID_MASK_STATUS_MIXED_STENCIL): "mixed_stencil",
    int(VALID_MASK_STATUS_HARD_INVALID): "hard_invalid",
}


def support_counts(statuses: np.ndarray) -> dict[str, int]:
    values = np.asarray(statuses, dtype=np.uint8)
    clean = int(np.count_nonzero(values == int(VALID_MASK_STATUS_CLEAN)))
    mixed = int(np.count_nonzero(values == int(VALID_MASK_STATUS_MIXED_STENCIL)))
    hard = int(np.count_nonzero(values == int(VALID_MASK_STATUS_HARD_INVALID)))
    return {
        "clean": clean,
        "mixed_stencil": mixed,
        "hard_invalid": hard,
        "non_clean": mixed + hard,
    }


def sample_support_statuses(
    field_provider: Any,
    points: np.ndarray,
    time_s: float,
) -> np.ndarray:
    sampled = ProviderSamplingBackend(field_provider).sample(
        np.asarray(points, dtype=np.float64),
        float(time_s),
        FieldRequest((VALID_MASK_QUANTITY,)),
    )
    return np.asarray(sampled.values[VALID_MASK_QUANTITY], dtype=np.uint8)


def _release_support_statuses(
    field_provider: Any,
    positions: np.ndarray,
    release_times: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    statuses = np.full(
        release_times.shape,
        int(VALID_MASK_STATUS_HARD_INVALID),
        dtype=np.uint8,
    )
    finite = np.isfinite(release_times)
    checked_times = np.unique(release_times[finite])
    for time_s in checked_times:
        indices = np.flatnonzero(finite & (release_times == time_s))
        statuses[indices] = sample_support_statuses(
            field_provider,
            positions[indices],
            float(time_s),
        )
    return statuses, [float(time_s) for time_s in checked_times]


GEOMETRY_STATUS_NAMES = (
    "strict_inside",
    "on_release_boundary",
    "on_boundary",
    "outside",
)
ACCEPTED_GEOMETRY_STATUSES = frozenset({"strict_inside", "on_release_boundary"})


def _on_release_boundary_mask_2d(
    runtime: Any,
    positions: np.ndarray,
    on_boundary: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return which on-boundary points lie on their own release entity.

    COMSOL releases inlet particles on the boundary itself, where the inlet
    feature overrides the wall condition, so that position is a valid initial
    state rather than a geometry violation.  The solver reproduces it without
    displacing the particle: a segment departing from its own boundary is not
    a hit.  Preflight therefore accepts a release that sits on the entity the
    release table declares, and only that entity.
    """

    accepted = np.zeros(positions.shape[0], dtype=bool)
    candidates = np.flatnonzero(on_boundary)
    if candidates.size == 0:
        return accepted
    source_part_id = getattr(
        getattr(runtime, "particles", None), "source_part_id", None
    )
    if source_part_id is None:
        return accepted
    nearest_part_ids, distances = nearest_boundary_edge_features_2d(
        runtime,
        positions[candidates],
    )
    declared = np.asarray(source_part_id, dtype=np.int64)[candidates]
    resolved = np.asarray(nearest_part_ids, dtype=np.int64)
    on_declared_entity = (
        np.isfinite(distances)
        & (np.asarray(distances, dtype=np.float64) <= float(tolerance))
        & (resolved == declared)
    )
    accepted[candidates[on_declared_entity]] = True
    return accepted


def _geometry_statuses(
    runtime: Any,
    positions: np.ndarray,
) -> tuple[np.ndarray, float]:
    boundary = getattr(getattr(runtime, "plan", None), "boundary", None)
    tolerance = float(getattr(boundary, "classification_tolerance_m", np.nan))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "initial-particle geometry validation requires a positive boundary "
            "classification tolerance"
        )

    finite = np.all(np.isfinite(positions), axis=1)
    inside = np.zeros(positions.shape[0], dtype=bool)
    on_boundary = np.zeros(positions.shape[0], dtype=bool)
    if int(runtime.spatial_dim) == 2:
        inside[finite], on_boundary[finite] = points_inside_geometry_2d(
            runtime,
            positions[finite],
            on_boundary_tol_m=tolerance,
            return_on_boundary=True,
        )
        if str(getattr(runtime, "coordinate_system", "")) == "axisymmetric_rz":
            axis_interior = (
                inside & on_boundary & (np.abs(positions[:, 0]) <= tolerance)
            )
            provider = getattr(runtime, "geometry_provider", None)
            edges_raw = getattr(
                getattr(provider, "geometry", None), "boundary_edges", None
            )
            if edges_raw is not None:
                edges = np.asarray(edges_raw, dtype=np.float64)
                physical = edges[~np.all(np.abs(edges[..., 0]) <= tolerance, axis=1)]
                starts = physical[:, 0]
                delta = physical[:, 1] - starts
                length_squared = np.einsum("ij,ij->i", delta, delta)
                valid = length_squared > 0.0
                if np.any(valid):
                    starts, delta = starts[valid], delta[valid]
                    indices = np.flatnonzero(axis_interior)
                    offset = positions[indices, None, :] - starts
                    alpha = (
                        np.einsum("nmi,mi->nm", offset, delta) / length_squared[valid]
                    )
                    projection = starts + np.clip(alpha, 0.0, 1.0)[..., None] * delta
                    distance = np.linalg.norm(
                        projection - positions[indices, None, :], axis=2
                    )
                    axis_interior[indices] &= np.all(distance > tolerance, axis=1)
            on_boundary[axis_interior] = False
    elif int(runtime.spatial_dim) == 3:
        geometry = runtime.geometry_provider.geometry
        if geometry.boundary_triangles is None:
            raise ValueError(
                "3D initial-particle validation requires geometry.boundary_triangles"
            )
        surface = build_geometry_surfaces_3d(geometry).containment
        for index in np.flatnonzero(finite):
            inside[index], on_boundary[index] = inside_geometry_with_boundary(
                runtime,
                positions[index],
                on_boundary_tol_m=tolerance,
                triangle_surface_3d=surface,
            )
    else:
        raise ValueError("initial-particle geometry validation supports only 2D or 3D")

    statuses = np.full(positions.shape[0], "outside", dtype="<U19")
    statuses[inside & ~on_boundary] = "strict_inside"
    statuses[on_boundary] = "on_boundary"
    if int(runtime.spatial_dim) == 2 and np.any(on_boundary):
        statuses[
            _on_release_boundary_mask_2d(runtime, positions, on_boundary, tolerance)
        ] = "on_release_boundary"
    return statuses, tolerance


def _coordinate_row(
    runtime: Any,
    index: int,
    positions: np.ndarray,
    release_times: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "particle_id": int(runtime.particles.particle_id[index]),
        "release_time_s": float(release_times[index]),
    }
    for axis_index, axis in enumerate(("x", "y", "z")[: int(runtime.spatial_dim)]):
        row[axis] = float(positions[index, axis_index])
    return row


def initial_particle_support_report(
    runtime: Any,
    *,
    include_violations: bool,
) -> dict[str, Any]:
    particles = runtime.particles
    positions = np.asarray(
        particles.position[:, : int(runtime.spatial_dim)], dtype=np.float64
    )
    release_times = np.asarray(particles.release_time, dtype=np.float64)
    support_statuses, checked_times = _release_support_statuses(
        runtime.field_provider,
        positions,
        release_times,
    )
    geometry_statuses, tolerance = _geometry_statuses(runtime, positions)

    counts = support_counts(support_statuses)
    geometry_counts = {
        name: int(np.count_nonzero(geometry_statuses == name))
        for name in GEOMETRY_STATUS_NAMES
    }
    field_invalid = np.flatnonzero(support_statuses != int(VALID_MASK_STATUS_CLEAN))
    geometry_invalid = np.flatnonzero(
        ~np.isin(geometry_statuses, sorted(ACCEPTED_GEOMETRY_STATUSES))
    )

    field_violations: list[dict[str, Any]] = []
    geometry_violations: list[dict[str, Any]] = []
    if include_violations:
        for index in field_invalid:
            row = _coordinate_row(runtime, int(index), positions, release_times)
            row["status"] = SUPPORT_STATUS_NAMES.get(
                int(support_statuses[index]), "unknown"
            )
            field_violations.append(row)
        for index in geometry_invalid:
            row = _coordinate_row(runtime, int(index), positions, release_times)
            row["status"] = str(geometry_statuses[index])
            geometry_violations.append(row)

    field_passed = counts["non_clean"] == 0
    geometry_passed = geometry_invalid.size == 0
    return {
        "mode": "strict",
        "support_scope": "spatial_only",
        "sample_time_scope": "particle_release_time",
        "passed": field_passed and geometry_passed,
        "field_support_passed": field_passed,
        "geometry_passed": geometry_passed,
        "particle_count": int(particles.count),
        "field_backend_kind": str(field_backend_kind(runtime.field_provider)),
        "checked_release_times_s": checked_times,
        "status_counts": counts,
        "geometry_status_counts": geometry_counts,
        "geometry_classification_tolerance_m": tolerance,
        "violation_count": int(field_invalid.size),
        "violations": field_violations,
        "violations_truncated": bool(field_invalid.size and not include_violations),
        "geometry_violation_count": int(geometry_invalid.size),
        "geometry_violations": geometry_violations,
        "geometry_violations_truncated": bool(
            geometry_invalid.size and not include_violations
        ),
    }


__all__ = (
    "SUPPORT_STATUS_NAMES",
    "initial_particle_support_report",
    "sample_support_statuses",
    "support_counts",
)
