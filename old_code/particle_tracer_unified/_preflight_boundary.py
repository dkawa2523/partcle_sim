from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._preflight_initial_state import (
    SUPPORT_STATUS_NAMES,
    sample_support_statuses,
    support_counts,
)
from .core.boundary_service import inside_geometry
from .core.field_backend import field_backend_kind
from .core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
)


def _positive_part_ids(values: Any) -> set[int]:
    return {int(value) for value in np.unique(np.asarray(values)) if int(value) > 0}


def runtime_boundary_coverage(runtime: Any) -> dict[str, Any]:
    geometry = getattr(getattr(runtime, "geometry_provider", None), "geometry", None)
    geometry_parts: set[int] = set()
    source = ""
    if geometry is not None:
        for attr in ("boundary_edge_part_ids", "boundary_triangle_part_ids"):
            values = getattr(geometry, attr, None)
            if values is not None:
                geometry_parts.update(_positive_part_ids(values))
                source = "explicit_boundary_elements"
        if not geometry_parts:
            nearest = getattr(geometry, "nearest_boundary_part_id_map", None)
            if nearest is not None:
                geometry_parts.update(_positive_part_ids(nearest))
                source = "nearest_boundary_part_id_map"
    catalog = getattr(runtime, "wall_catalog", None)
    wall_parts = {
        int(model.part_id)
        for model in getattr(catalog, "part_models", ())
        if int(getattr(model, "part_id", 0)) > 0
    }
    return {
        "passed": bool(geometry_parts and geometry_parts == wall_parts),
        "geometry_part_source": source,
        "geometry_part_ids": sorted(geometry_parts),
        "wall_catalog_part_ids": sorted(wall_parts),
        "missing_wall_models": sorted(geometry_parts - wall_parts),
        "stale_wall_models": sorted(wall_parts - geometry_parts),
    }


def _empty_boundary_samples(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, dim), dtype=np.float64),
        np.zeros((0, dim), dtype=np.float64),
        np.zeros(0, dtype=np.int64),
    )


def _edge_samples(geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if geometry.boundary_edges is None or geometry.boundary_edge_part_ids is None:
        return _empty_boundary_samples(2)
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    part_ids: list[int] = []
    edges = np.asarray(geometry.boundary_edges, dtype=np.float64)
    parts = np.asarray(geometry.boundary_edge_part_ids, dtype=np.int64)
    for edge, part_id in zip(edges, parts, strict=False):
        delta = edge[1] - edge[0]
        length = float(np.linalg.norm(delta))
        if np.isfinite(length) and length > 0.0:
            points.append(np.mean(edge, axis=0))
            normals.append(np.asarray([-delta[1], delta[0]], dtype=np.float64) / length)
            part_ids.append(int(part_id))
    return (
        np.asarray(points, dtype=np.float64).reshape(-1, 2),
        np.asarray(normals, dtype=np.float64).reshape(-1, 2),
        np.asarray(part_ids, dtype=np.int64),
    )


def _triangle_samples(geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = geometry.boundary_triangles
    triangle_parts = geometry.boundary_triangle_part_ids
    if triangles is None or triangle_parts is None:
        return _empty_boundary_samples(3)
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    part_ids: list[int] = []
    for triangle, part_id in zip(
        np.asarray(triangles, dtype=np.float64),
        np.asarray(triangle_parts, dtype=np.int64),
        strict=False,
    ):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = float(np.linalg.norm(normal))
        if np.isfinite(length) and length > 0.0:
            points.append(np.mean(triangle, axis=0))
            normals.append(normal / length)
            part_ids.append(int(part_id))
    return (
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        np.asarray(normals, dtype=np.float64).reshape(-1, 3),
        np.asarray(part_ids, dtype=np.int64),
    )


def boundary_samples(runtime: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = runtime.geometry_provider.geometry
    return (
        _edge_samples(geometry)
        if int(runtime.spatial_dim) == 2
        else _triangle_samples(geometry)
    )


def _minimum_cell_diagonal(axes: tuple[np.ndarray, ...]) -> float:
    spacings: list[float] = []
    for axis in axes:
        diffs = np.diff(np.asarray(axis, dtype=np.float64))
        positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if positive.size:
            spacings.append(float(np.min(positive)))
    return (
        float(math.sqrt(sum(value * value for value in spacings))) if spacings else 0.0
    )


def _interior_boundary_points(
    runtime: Any,
    boundary_points: np.ndarray,
    normals: np.ndarray,
    offset_m: float,
) -> np.ndarray:
    inside_points = np.full_like(boundary_points, np.nan)
    for index, (point, normal) in enumerate(
        zip(boundary_points, normals, strict=False)
    ):
        for sign in (1.0, -1.0):
            candidate = point + sign * normal * offset_m
            if inside_geometry(runtime, candidate, on_boundary_tol_m=0.0):
                inside_points[index] = candidate
                break
    return inside_points


def _boundary_support_statuses(
    runtime: Any,
    inside_points: np.ndarray,
) -> np.ndarray:
    statuses = np.full(
        inside_points.shape[0],
        int(VALID_MASK_STATUS_HARD_INVALID),
        dtype=np.uint8,
    )
    finite_points = np.all(np.isfinite(inside_points), axis=1)
    if np.any(finite_points):
        statuses[finite_points] = sample_support_statuses(
            runtime.field_provider,
            inside_points[finite_points],
            0.0,
        )
    return statuses


def boundary_field_support_report(
    runtime: Any,
    *,
    include_violations: bool,
) -> dict[str, Any]:
    geometry = runtime.geometry_provider.geometry
    boundary_points, normals, part_ids = boundary_samples(runtime)
    if boundary_points.shape[0] == 0:
        return {
            "mode": "strict",
            "passed": True,
            "applicable": False,
            "reason": "no explicit boundary",
        }
    offset_m = _minimum_cell_diagonal(tuple(np.asarray(axis) for axis in geometry.axes))
    if not np.isfinite(offset_m) or offset_m <= 0.0:
        raise ValueError("Could not derive a positive boundary support offset")
    inside_points = _interior_boundary_points(
        runtime,
        boundary_points,
        normals,
        offset_m,
    )
    statuses = _boundary_support_statuses(runtime, inside_points)
    counts = support_counts(statuses)
    invalid = np.flatnonzero(statuses != int(VALID_MASK_STATUS_CLEAN))
    violations = []
    if include_violations:
        violations = [
            {
                "part_id": int(part_ids[index]),
                "status": SUPPORT_STATUS_NAMES.get(int(statuses[index]), "unknown"),
                "boundary_point_m": boundary_points[index].tolist(),
                "interior_point_m": inside_points[index].tolist(),
            }
            for index in invalid
        ]
    return {
        "mode": "strict",
        "support_scope": "spatial_only",
        "passed": counts["non_clean"] == 0,
        "applicable": True,
        "field_backend_kind": str(field_backend_kind(runtime.field_provider)),
        "boundary_offset_m": float(offset_m),
        "sample_count": int(statuses.size),
        "status_counts": counts,
        "violation_count": int(invalid.size),
        "violations": violations,
        "violations_truncated": bool(invalid.size and not include_violations),
    }
