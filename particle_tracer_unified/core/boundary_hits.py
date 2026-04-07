from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .boundary_core import (
    BoundaryHit,
    inside_geometry,
    inside_geometry_with_boundary,
    sample_geometry_normal,
    sample_geometry_part_id,
)
from .geometry3d import TriangleSurface3D, nearest_surface_point, segment_hit_from_surface


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_hit_from_boundary_edges(runtime, p0: np.ndarray, p1: np.ndarray) -> Optional[BoundaryHit]:
    geometry_provider = runtime.geometry_provider
    if geometry_provider is None:
        return None
    geom = geometry_provider.geometry
    if int(geom.spatial_dim) != 2 or geom.boundary_edges is None:
        return None
    segments = np.asarray(geom.boundary_edges, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or segments.shape[0] == 0:
        return None
    part_ids = np.asarray(
        geom.boundary_edge_part_ids if geom.boundary_edge_part_ids is not None else np.zeros(segments.shape[0], dtype=np.int32),
        dtype=np.int32,
    )
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    r = b - a
    if float(np.linalg.norm(r)) <= 1e-30:
        return None
    eps = 1.0e-12
    best_alpha = 2.0
    best_hit = None
    best_normal = None
    best_part_id = 0
    for idx in range(segments.shape[0]):
        q0 = segments[idx, 0]
        q1 = segments[idx, 1]
        s = q1 - q0
        denominator = _cross2d(r, s)
        if abs(denominator) <= 1e-30:
            continue
        q_minus_p = q0 - a
        t = _cross2d(q_minus_p, s) / denominator
        u = _cross2d(q_minus_p, r) / denominator
        if t < -eps or t > 1.0 + eps or u < -eps or u > 1.0 + eps:
            continue
        alpha = float(np.clip(t, 0.0, 1.0))
        if alpha >= best_alpha:
            continue
        hit = a + alpha * r
        normal = np.array([-s[1], s[0]], dtype=np.float64)
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1e-30:
            continue
        normal /= magnitude
        if float(np.dot(a - hit, normal)) > 0.0:
            normal = -normal
        best_alpha = alpha
        best_hit = hit
        best_normal = normal
        best_part_id = int(part_ids[idx]) if idx < part_ids.size else 0
    if best_hit is None or best_normal is None:
        return None
    return BoundaryHit(
        position=np.asarray(best_hit, dtype=np.float64),
        normal=np.asarray(best_normal, dtype=np.float64),
        part_id=max(0, int(best_part_id)),
        alpha_hint=float(np.clip(best_alpha, 0.0, 1.0)),
    )


def nearest_hit_on_boundary_edges(runtime, point: np.ndarray, inside_reference: np.ndarray) -> Optional[BoundaryHit]:
    geometry_provider = runtime.geometry_provider
    if geometry_provider is None:
        return None
    geom = geometry_provider.geometry
    if int(geom.spatial_dim) != 2 or geom.boundary_edges is None:
        return None
    segments = np.asarray(geom.boundary_edges, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or segments.shape[0] == 0:
        return None
    part_ids = np.asarray(
        geom.boundary_edge_part_ids if geom.boundary_edge_part_ids is not None else np.zeros(segments.shape[0], dtype=np.int32),
        dtype=np.int32,
    )
    point_arr = np.asarray(point, dtype=np.float64)
    ref = np.asarray(inside_reference, dtype=np.float64)
    best_dist = np.inf
    best_hit = None
    best_normal = None
    best_part_id = 0
    for idx, segment in enumerate(segments):
        q0 = segment[0]
        q1 = segment[1]
        edge = q1 - q0
        denominator = float(np.dot(edge, edge))
        if denominator <= 1e-30:
            hit = q0.copy()
        else:
            alpha = float(np.clip(np.dot(point_arr - q0, edge) / denominator, 0.0, 1.0))
            hit = q0 + alpha * edge
        dist = float(np.linalg.norm(point_arr - hit))
        if dist >= best_dist:
            continue
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1e-30:
            continue
        normal /= magnitude
        if float(np.dot(ref - hit, normal)) > 0.0:
            normal = -normal
        best_dist = dist
        best_hit = hit
        best_normal = normal
        best_part_id = int(part_ids[idx]) if idx < part_ids.size else 0
    if best_hit is None or best_normal is None:
        return None
    return BoundaryHit(
        position=np.asarray(best_hit, dtype=np.float64),
        normal=np.asarray(best_normal, dtype=np.float64),
        part_id=max(0, int(best_part_id)),
        alpha_hint=0.0,
    )


def segment_hit_from_loop_bisection(
    runtime,
    p0: np.ndarray,
    p1: np.ndarray,
    on_boundary_tol_m: float,
    iters: int = 40,
) -> Optional[BoundaryHit]:
    geometry_provider = runtime.geometry_provider
    if geometry_provider is None:
        return None
    geom = geometry_provider.geometry
    if int(geom.spatial_dim) != 2 or not geom.boundary_loops_2d:
        return None
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    segment = b - a
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= 1e-30:
        return None
    if not inside_geometry(runtime, a, on_boundary_tol_m=on_boundary_tol_m):
        return None
    if inside_geometry(runtime, b, on_boundary_tol_m=on_boundary_tol_m):
        return None
    lo = a.copy()
    hi = b.copy()
    for _ in range(int(max(1, iters))):
        mid = 0.5 * (lo + hi)
        if inside_geometry(runtime, mid, on_boundary_tol_m=on_boundary_tol_m):
            lo = mid
        else:
            hi = mid
    hit = 0.5 * (lo + hi)
    alpha = float(np.clip(np.linalg.norm(hit - a) / max(segment_length, 1.0e-30), 0.0, 1.0))
    nearest = nearest_hit_on_boundary_edges(runtime, hit, a)
    if nearest is not None:
        alpha = float(np.clip(np.linalg.norm(nearest.position - a) / max(segment_length, 1.0e-30), 0.0, 1.0))
        return BoundaryHit(
            position=np.asarray(nearest.position, dtype=np.float64),
            normal=np.asarray(nearest.normal, dtype=np.float64),
            part_id=int(nearest.part_id),
            alpha_hint=alpha,
        )
    return BoundaryHit(
        position=np.asarray(hit, dtype=np.float64),
        normal=np.asarray(sample_geometry_normal(runtime, hit), dtype=np.float64),
        part_id=int(sample_geometry_part_id(runtime, hit)),
        alpha_hint=alpha,
    )


def segment_hit_from_boundary_triangles(
    triangle_surface_3d: Optional[TriangleSurface3D],
    p0: np.ndarray,
    p1: np.ndarray,
) -> Optional[BoundaryHit]:
    if triangle_surface_3d is None:
        return None
    hit = segment_hit_from_surface(triangle_surface_3d, np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
    if hit is None:
        return None
    point, normal, alpha, part_id, _triangle_index = hit
    return BoundaryHit(
        position=np.asarray(point, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        part_id=int(part_id),
        alpha_hint=float(alpha),
    )


def nearest_hit_on_boundary_triangles(
    triangle_surface_3d: Optional[TriangleSurface3D],
    point: np.ndarray,
    inside_reference: np.ndarray,
) -> Optional[BoundaryHit]:
    if triangle_surface_3d is None:
        return None
    try:
        hit, normal, part_id = nearest_surface_point(
            triangle_surface_3d,
            np.asarray(point, dtype=np.float64),
            inside_reference=np.asarray(inside_reference, dtype=np.float64),
        )
    except Exception:
        return None
    return BoundaryHit(
        position=np.asarray(hit, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        part_id=int(part_id),
        alpha_hint=0.0,
    )


def segment_hit_from_solid_bisection_3d(
    runtime,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    triangle_surface_3d: Optional[TriangleSurface3D],
    on_boundary_tol_m: float,
    iters: int = 40,
) -> Optional[BoundaryHit]:
    if triangle_surface_3d is None:
        return None
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    segment = b - a
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= 1e-30:
        return None
    inside_a, _ = inside_geometry_with_boundary(
        runtime,
        a,
        on_boundary_tol_m=on_boundary_tol_m,
        triangle_surface_3d=triangle_surface_3d,
    )
    inside_b, _ = inside_geometry_with_boundary(
        runtime,
        b,
        on_boundary_tol_m=on_boundary_tol_m,
        triangle_surface_3d=triangle_surface_3d,
    )
    if not inside_a or inside_b:
        return None
    lo = a.copy()
    hi = b.copy()
    for _ in range(int(max(1, iters))):
        mid = 0.5 * (lo + hi)
        inside_mid, _ = inside_geometry_with_boundary(
            runtime,
            mid,
            on_boundary_tol_m=on_boundary_tol_m,
            triangle_surface_3d=triangle_surface_3d,
        )
        if inside_mid:
            lo = mid
        else:
            hi = mid
    hit_guess = 0.5 * (lo + hi)
    nearest = nearest_hit_on_boundary_triangles(triangle_surface_3d, hit_guess, lo)
    if nearest is None:
        return None
    alpha = float(np.clip(np.linalg.norm(nearest.position - a) / max(segment_length, 1.0e-30), 0.0, 1.0))
    return BoundaryHit(
        position=np.asarray(nearest.position, dtype=np.float64),
        normal=np.asarray(nearest.normal, dtype=np.float64),
        part_id=int(nearest.part_id),
        alpha_hint=alpha,
    )


def normalize_polyline_alpha(segment_index: int, local_alpha: float, segment_count: int) -> float:
    segment_count_safe = max(1, int(segment_count))
    alpha_local = float(np.clip(local_alpha, 0.0, 1.0))
    alpha = (float(segment_index) + alpha_local) / float(segment_count_safe)
    return float(np.clip(alpha, 0.0, 1.0))


def _polyline_hit(
    p0: np.ndarray,
    stage_points: np.ndarray,
    *,
    segment_hit_fn: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]],
) -> Optional[BoundaryHit]:
    points = np.asarray(stage_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return None
    segment_count = int(points.shape[0])
    start = np.asarray(p0, dtype=np.float64)
    for segment_index in range(segment_count):
        end = points[segment_index]
        hit = segment_hit_fn(start, end)
        if hit is not None:
            return BoundaryHit(
                position=np.asarray(hit.position, dtype=np.float64),
                normal=np.asarray(hit.normal, dtype=np.float64),
                part_id=int(hit.part_id),
                alpha_hint=normalize_polyline_alpha(segment_index, hit.alpha_hint, segment_count),
            )
        start = end
    return None


def polyline_hit_from_boundary_edges(runtime, p0: np.ndarray, stage_points: np.ndarray) -> Optional[BoundaryHit]:
    return _polyline_hit(
        p0,
        stage_points,
        segment_hit_fn=lambda a, b: segment_hit_from_boundary_edges(runtime, a, b),
    )


def polyline_hit_from_loop_bisection(
    runtime,
    p0: np.ndarray,
    stage_points: np.ndarray,
    *,
    on_boundary_tol_m: float,
) -> Optional[BoundaryHit]:
    return _polyline_hit(
        p0,
        stage_points,
        segment_hit_fn=lambda a, b: segment_hit_from_loop_bisection(
            runtime,
            a,
            b,
            on_boundary_tol_m=on_boundary_tol_m,
        ),
    )


def polyline_hit_from_boundary_triangles(
    triangle_surface_3d: Optional[TriangleSurface3D],
    p0: np.ndarray,
    stage_points: np.ndarray,
) -> Optional[BoundaryHit]:
    return _polyline_hit(
        p0,
        stage_points,
        segment_hit_fn=lambda a, b: segment_hit_from_boundary_triangles(triangle_surface_3d, a, b),
    )


def polyline_hit_from_solid_bisection_3d(
    runtime,
    p0: np.ndarray,
    stage_points: np.ndarray,
    *,
    triangle_surface_3d: Optional[TriangleSurface3D],
    on_boundary_tol_m: float,
) -> Optional[BoundaryHit]:
    return _polyline_hit(
        p0,
        stage_points,
        segment_hit_fn=lambda a, b: segment_hit_from_solid_bisection_3d(
            runtime,
            a,
            b,
            triangle_surface_3d=triangle_surface_3d,
            on_boundary_tol_m=on_boundary_tol_m,
        ),
    )


__all__ = (
    'nearest_hit_on_boundary_edges',
    'nearest_hit_on_boundary_triangles',
    'normalize_polyline_alpha',
    'polyline_hit_from_boundary_edges',
    'polyline_hit_from_boundary_triangles',
    'polyline_hit_from_loop_bisection',
    'polyline_hit_from_solid_bisection_3d',
    'segment_hit_from_boundary_edges',
    'segment_hit_from_boundary_triangles',
    'segment_hit_from_loop_bisection',
    'segment_hit_from_solid_bisection_3d',
)
