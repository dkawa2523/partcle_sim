"""Triangle boundary adapters and scalar polyline traversal."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from particle_tracer_unified.domain import BoundaryHit

from ._boundary_hits_2d import segment_hit_from_boundary_edges
from ._triangle_queries import (
    nearest_surface_point_ignoring_parts,
    segment_hit_from_surface_ignoring_parts,
)
from .geometry3d import (
    TriangleSurface3D,
    point_triangle_barycentric,
)

_TRIANGLE_EDGE_TOL = 1.0e-8


def _copy_hit_with_alpha(hit: BoundaryHit, alpha_hint: float) -> BoundaryHit:
    return BoundaryHit(
        position=np.asarray(hit.position, dtype=np.float64),
        normal=np.asarray(hit.normal, dtype=np.float64),
        part_id=int(hit.part_id),
        alpha_hint=float(alpha_hint),
        primitive_id=int(hit.primitive_id),
        primitive_kind=str(hit.primitive_kind),
        is_ambiguous=bool(hit.is_ambiguous),
    )


def segment_hit_from_boundary_triangles(
    triangle_surface_3d: TriangleSurface3D | None,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    coordinate_tolerance_m: float = 0.0,
    ignored_part_ids: frozenset[int] = frozenset(),
) -> BoundaryHit | None:
    if triangle_surface_3d is None:
        return None
    hit = segment_hit_from_surface_ignoring_parts(
        triangle_surface_3d,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
        coordinate_tolerance_m=float(coordinate_tolerance_m),
        ignored_part_ids=ignored_part_ids,
    )
    if hit is None:
        return None
    point, normal, alpha, part_id, triangle_index = hit
    bary = point_triangle_barycentric(
        point, triangle_surface_3d.triangles[int(triangle_index)]
    )
    ambiguous = False if bary is None else bool(np.min(bary) <= _TRIANGLE_EDGE_TOL)
    return BoundaryHit(
        position=np.asarray(point, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        part_id=int(part_id),
        alpha_hint=float(alpha),
        primitive_id=int(triangle_index),
        primitive_kind="triangle",
        is_ambiguous=bool(ambiguous),
    )


def _nearest_hit_on_boundary_triangles(
    triangle_surface_3d: TriangleSurface3D | None,
    point: np.ndarray,
    inside_reference: np.ndarray,
    ignored_part_ids: frozenset[int],
) -> BoundaryHit | None:
    if triangle_surface_3d is None:
        return None
    try:
        hit, normal, part_id, triangle_index = nearest_surface_point_ignoring_parts(
            triangle_surface_3d,
            np.asarray(point, dtype=np.float64),
            inside_reference=np.asarray(inside_reference, dtype=np.float64),
            ignored_part_ids=ignored_part_ids,
        )
    except ValueError:
        return None
    bary = point_triangle_barycentric(
        hit, triangle_surface_3d.triangles[int(triangle_index)]
    )
    ambiguous = bool(
        bary is not None
        and np.any(np.asarray(bary, dtype=np.float64) <= _TRIANGLE_EDGE_TOL)
    )
    return BoundaryHit(
        position=np.asarray(hit, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        part_id=int(part_id),
        alpha_hint=0.0,
        primitive_id=int(triangle_index),
        primitive_kind="triangle_projection",
        is_ambiguous=ambiguous,
    )


def nearest_hit_on_boundary_triangles(
    triangle_surface_3d: TriangleSurface3D | None,
    point: np.ndarray,
    inside_reference: np.ndarray,
) -> BoundaryHit | None:
    return _nearest_hit_on_boundary_triangles(
        triangle_surface_3d,
        point,
        inside_reference,
        frozenset(),
    )


def nearest_hit_on_boundary_triangles_ignoring_parts(
    triangle_surface_3d: TriangleSurface3D | None,
    point: np.ndarray,
    inside_reference: np.ndarray,
    *,
    ignored_part_ids: frozenset[int],
) -> BoundaryHit | None:
    return _nearest_hit_on_boundary_triangles(
        triangle_surface_3d,
        point,
        inside_reference,
        frozenset(int(value) for value in ignored_part_ids),
    )


def normalize_polyline_alpha(
    segment_index: int, local_alpha: float, segment_count: int
) -> float:
    segment_count_safe = max(1, int(segment_count))
    alpha_local = float(np.clip(local_alpha, 0.0, 1.0))
    alpha = (float(segment_index) + alpha_local) / float(segment_count_safe)
    return float(np.clip(alpha, 0.0, 1.0))


def _polyline_hit(
    p0: np.ndarray,
    stage_points: np.ndarray,
    *,
    segment_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None],
) -> BoundaryHit | None:
    points = np.asarray(stage_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return None
    segment_count = int(points.shape[0])
    start = np.asarray(p0, dtype=np.float64)
    for segment_index in range(segment_count):
        end = points[segment_index]
        hit = segment_hit_fn(start, end)
        if hit is not None:
            return _copy_hit_with_alpha(
                hit,
                normalize_polyline_alpha(segment_index, hit.alpha_hint, segment_count),
            )
        start = end
    return None


def polyline_hit_from_boundary_edges(
    runtime,
    p0: np.ndarray,
    stage_points: np.ndarray,
    *,
    coordinate_tolerance_m: float = 0.0,
    departure_tolerance_m: float = 0.0,
) -> BoundaryHit | None:
    return _polyline_hit(
        p0,
        stage_points,
        segment_hit_fn=lambda a, b: segment_hit_from_boundary_edges(
            runtime,
            a,
            b,
            coordinate_tolerance_m=float(coordinate_tolerance_m),
            departure_tolerance_m=float(departure_tolerance_m),
        ),
    )


def polyline_hit_from_boundary_triangles(
    triangle_surface_3d: TriangleSurface3D | None,
    p0: np.ndarray,
    stage_points: np.ndarray,
    *,
    coordinate_tolerance_m: float = 0.0,
    ignored_part_ids: frozenset[int] = frozenset(),
) -> BoundaryHit | None:
    return _polyline_hit(
        p0,
        stage_points,
        segment_hit_fn=lambda a, b: segment_hit_from_boundary_triangles(
            triangle_surface_3d,
            a,
            b,
            coordinate_tolerance_m=float(coordinate_tolerance_m),
            ignored_part_ids=ignored_part_ids,
        ),
    )


__all__ = (
    "nearest_hit_on_boundary_triangles",
    "normalize_polyline_alpha",
    "polyline_hit_from_boundary_edges",
    "polyline_hit_from_boundary_triangles",
    "segment_hit_from_boundary_triangles",
)
