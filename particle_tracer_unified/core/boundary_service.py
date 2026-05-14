from __future__ import annotations

from typing import Optional

import numpy as np

from .boundary_core import (
    BoundaryHit,
    BoundaryService,
    ReleasePointClassification,
    inside_geometry,
    inside_geometry_with_boundary,
    points_inside_geometry_2d,
    runtime_bounds,
    sample_geometry_normal,
    sample_geometry_part_id,
    sample_geometry_sdf,
)
from .boundary_hits import (
    BoundaryEdgeFrame2D,
    contact_frame_on_boundary_edge_2d,
    nearest_boundary_edge_features_2d,
    nearest_hit_on_boundary_edges,
    nearest_hit_on_boundary_triangles,
    normalize_polyline_alpha,
    polyline_hit_from_boundary_edges,
    polyline_hits_from_boundary_edges_batch,
    polyline_hit_from_boundary_triangles,
    polyline_hit_from_loop_bisection,
    polyline_hit_from_solid_bisection_3d,
    segment_hit_from_boundary_edges,
    segment_hit_from_boundary_triangles,
    segment_hit_from_loop_bisection,
    segment_hit_from_solid_bisection_3d,
)
from .geometry3d import TriangleSurface3D, _closest_point_on_triangle


def _empty_release_point(position: np.ndarray, spatial_dim: int) -> ReleasePointClassification:
    pos = np.asarray(position, dtype=np.float64)
    dim = int(max(1, spatial_dim))
    normal = np.zeros(dim, dtype=np.float64)
    if normal.size:
        normal[-1] = 1.0
    return ReleasePointClassification(
        is_on_boundary=False,
        inside_after_offset=False,
        nearest_part_id=0,
        normal=normal,
        distance_m=float('inf'),
        primitive_id=-1,
        ambiguous=False,
        boundary_position=None,
        offset_position=pos[:dim].copy() if pos.size >= dim else None,
    )


def _normalize_or_fallback(vector: np.ndarray, spatial_dim: int) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)[: int(spatial_dim)].copy()
    mag = float(np.linalg.norm(vec))
    if mag > 1.0e-30:
        return vec / mag
    out = np.zeros(int(spatial_dim), dtype=np.float64)
    out[-1 if int(spatial_dim) > 1 else 0] = 1.0
    return out


def _triangle_barycentric_coordinates(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangle, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    a = tri[0]
    b = tri[1]
    c = tri[2]
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1.0e-30:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.asarray([u, v, w], dtype=np.float64)


def _orient_normal_to_inside(
    *,
    inside,
    inside_strict,
    boundary_position: np.ndarray,
    normal: np.ndarray,
    probe_m: float,
) -> np.ndarray:
    n = _normalize_or_fallback(normal, np.asarray(boundary_position).size)
    probe = max(float(probe_m), 1.0e-12)
    point = np.asarray(boundary_position, dtype=np.float64)
    plus = point + probe * n
    minus = point - probe * n
    try:
        if bool(inside_strict(plus)):
            return n
        if bool(inside_strict(minus)):
            return -n
        if bool(inside(plus)):
            return n
        if bool(inside(minus)):
            return -n
    except Exception:
        pass
    return n


def _boundary_edges_for_release(runtime) -> tuple[Optional[np.ndarray], np.ndarray]:
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None:
        return None, np.zeros(0, dtype=np.int32)
    geom = geometry_provider.geometry
    if int(getattr(geom, 'spatial_dim', 0)) != 2 or getattr(geom, 'boundary_edges', None) is None:
        return None, np.zeros(0, dtype=np.int32)
    edges = np.asarray(geom.boundary_edges, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2) or edges.shape[0] == 0:
        return None, np.zeros(0, dtype=np.int32)
    part_ids = np.asarray(
        geom.boundary_edge_part_ids if geom.boundary_edge_part_ids is not None else np.zeros(edges.shape[0], dtype=np.int32),
        dtype=np.int32,
    )
    if part_ids.size < edges.shape[0]:
        part_ids = np.pad(part_ids, (0, int(edges.shape[0] - part_ids.size)), constant_values=0)
    elif part_ids.size > edges.shape[0]:
        part_ids = part_ids[: edges.shape[0]]
    return edges, part_ids


def _classify_release_point_2d(
    runtime,
    *,
    inside,
    inside_strict,
    position: np.ndarray,
    source_part_id: int,
    offset_m: float,
    tolerance_m: float,
) -> ReleasePointClassification:
    edges, part_ids = _boundary_edges_for_release(runtime)
    if edges is None:
        return _empty_release_point(position, 2)
    pos = np.asarray(position, dtype=np.float64)[:2]
    candidates = np.arange(edges.shape[0], dtype=np.int64)
    if int(source_part_id) > 0 and np.any(part_ids == int(source_part_id)):
        candidates = np.flatnonzero(part_ids == int(source_part_id)).astype(np.int64)
    best_idx = -1
    best_dist = float('inf')
    best_alpha = 0.0
    best_position = None
    best_normal = None
    for idx_raw in candidates:
        idx = int(idx_raw)
        q0 = np.asarray(edges[idx, 0], dtype=np.float64)
        q1 = np.asarray(edges[idx, 1], dtype=np.float64)
        edge = q1 - q0
        denom = float(np.dot(edge, edge))
        alpha = 0.0 if denom <= 1.0e-30 else float(np.clip(np.dot(pos - q0, edge) / denom, 0.0, 1.0))
        hit = q0 + alpha * edge
        dist = float(np.linalg.norm(pos - hit))
        if dist >= best_dist:
            continue
        raw_normal = np.asarray([-edge[1], edge[0]], dtype=np.float64)
        normal = _orient_normal_to_inside(
            inside=inside,
            inside_strict=inside_strict,
            boundary_position=hit,
            normal=raw_normal,
            probe_m=max(float(offset_m), float(tolerance_m)),
        )
        best_idx = idx
        best_dist = dist
        best_alpha = alpha
        best_position = hit
        best_normal = normal
    if best_position is None or best_normal is None:
        return _empty_release_point(position, 2)
    offset = max(float(offset_m), 0.0)
    offset_position = np.asarray(best_position, dtype=np.float64) + offset * np.asarray(best_normal, dtype=np.float64)
    return ReleasePointClassification(
        is_on_boundary=bool(best_dist <= max(float(tolerance_m), 0.0)),
        inside_after_offset=bool(inside(offset_position)),
        nearest_part_id=int(part_ids[best_idx]) if 0 <= best_idx < part_ids.size else 0,
        normal=np.asarray(best_normal, dtype=np.float64),
        distance_m=float(best_dist),
        primitive_id=int(best_idx),
        ambiguous=bool(best_alpha <= 1.0e-9 or best_alpha >= 1.0 - 1.0e-9),
        boundary_position=np.asarray(best_position, dtype=np.float64),
        offset_position=np.asarray(offset_position, dtype=np.float64),
    )


def _nearest_triangle_for_release(
    surface: TriangleSurface3D,
    position: np.ndarray,
    source_part_id: int,
    tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, int, int, float, bool]:
    point = np.asarray(position, dtype=np.float64)[:3]
    part_ids = np.asarray(surface.part_ids, dtype=np.int32)
    candidates = np.arange(surface.triangles.shape[0], dtype=np.int64)
    if int(source_part_id) > 0 and np.any(part_ids == int(source_part_id)):
        candidates = np.flatnonzero(part_ids == int(source_part_id)).astype(np.int64)
    best_idx = -1
    best_dist = float('inf')
    best_point = None
    for idx_raw in candidates:
        idx = int(idx_raw)
        nearest = _closest_point_on_triangle(point, np.asarray(surface.triangles[idx], dtype=np.float64))
        dist = float(np.linalg.norm(point - nearest))
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
            best_point = nearest
    if best_point is None or best_idx < 0:
        raise ValueError('No boundary triangle available for release point classification')
    tie_tol = max(float(tolerance_m), 1.0e-12)
    tie_count = 0
    for idx_raw in candidates:
        idx = int(idx_raw)
        nearest = _closest_point_on_triangle(point, np.asarray(surface.triangles[idx], dtype=np.float64))
        dist = float(np.linalg.norm(point - nearest))
        if abs(dist - best_dist) <= tie_tol:
            tie_count += 1
    bary = _triangle_barycentric_coordinates(best_point, np.asarray(surface.triangles[best_idx], dtype=np.float64))
    edge_or_vertex = bool(np.any(np.isfinite(bary)) and np.nanmin(bary) <= 1.0e-9)
    normal = np.asarray(surface.normals[best_idx], dtype=np.float64)
    part_id = int(part_ids[best_idx]) if best_idx < part_ids.size else 0
    ambiguous = bool(edge_or_vertex or tie_count > 1)
    return np.asarray(best_point, dtype=np.float64), normal, max(0, part_id), int(best_idx), float(best_dist), ambiguous


def _classify_release_point_3d(
    *,
    surface: Optional[TriangleSurface3D],
    inside,
    inside_strict,
    position: np.ndarray,
    source_part_id: int,
    offset_m: float,
    tolerance_m: float,
) -> ReleasePointClassification:
    if surface is None:
        return _empty_release_point(position, 3)
    try:
        boundary_position, raw_normal, part_id, primitive_id, distance, ambiguous = _nearest_triangle_for_release(
            surface,
            position,
            source_part_id,
            tolerance_m,
        )
    except Exception:
        return _empty_release_point(position, 3)
    normal = _orient_normal_to_inside(
        inside=inside,
        inside_strict=inside_strict,
        boundary_position=boundary_position,
        normal=raw_normal,
        probe_m=max(float(offset_m), float(tolerance_m)),
    )
    offset_position = np.asarray(boundary_position, dtype=np.float64) + max(float(offset_m), 0.0) * normal
    return ReleasePointClassification(
        is_on_boundary=bool(float(distance) <= max(float(tolerance_m), 0.0)),
        inside_after_offset=bool(inside(offset_position)),
        nearest_part_id=int(part_id),
        normal=np.asarray(normal, dtype=np.float64),
        distance_m=float(distance),
        primitive_id=int(primitive_id),
        ambiguous=bool(ambiguous),
        boundary_position=np.asarray(boundary_position, dtype=np.float64),
        offset_position=np.asarray(offset_position, dtype=np.float64),
    )


def _build_boundary_service_2d(runtime, *, on_boundary_tol_m: float) -> BoundaryService:
    inside = lambda pos: inside_geometry(runtime, pos, on_boundary_tol_m=on_boundary_tol_m)
    inside_strict = lambda pos: inside_geometry(runtime, pos, on_boundary_tol_m=0.0)
    return BoundaryService(
        inside=inside,
        inside_strict=inside_strict,
        segment_hit=lambda p0, p1: segment_hit_from_boundary_edges(runtime, p0, p1),
        polyline_hit=lambda p0, stage_pts: polyline_hit_from_boundary_edges(runtime, p0, stage_pts),
        nearest_projection=lambda point, inside_ref: nearest_hit_on_boundary_edges(runtime, point, inside_ref),
        release_point=lambda position, source_part_id, offset_m, tolerance_m: _classify_release_point_2d(
            runtime,
            inside=inside,
            inside_strict=inside_strict,
            position=position,
            source_part_id=int(source_part_id),
            offset_m=float(offset_m),
            tolerance_m=float(tolerance_m),
        ),
        primary_hit_counter_key='edge_hit_count',
        triangle_surface_3d=None,
    )


def _build_boundary_service_3d(
    runtime,
    *,
    on_boundary_tol_m: float,
    triangle_surface_3d: Optional[TriangleSurface3D],
) -> BoundaryService:
    inside = lambda pos: inside_geometry(
        runtime,
        pos,
        on_boundary_tol_m=on_boundary_tol_m,
        triangle_surface_3d=triangle_surface_3d,
    )
    inside_strict = lambda pos: inside_geometry(
        runtime,
        pos,
        on_boundary_tol_m=0.0,
        triangle_surface_3d=triangle_surface_3d,
    )
    return BoundaryService(
        inside=inside,
        inside_strict=inside_strict,
        segment_hit=lambda p0, p1: segment_hit_from_boundary_triangles(triangle_surface_3d, p0, p1),
        polyline_hit=lambda p0, stage_pts: polyline_hit_from_boundary_triangles(triangle_surface_3d, p0, stage_pts),
        nearest_projection=lambda point, inside_ref: nearest_hit_on_boundary_triangles(triangle_surface_3d, point, inside_ref),
        release_point=lambda position, source_part_id, offset_m, tolerance_m: _classify_release_point_3d(
            surface=triangle_surface_3d,
            inside=inside,
            inside_strict=inside_strict,
            position=position,
            source_part_id=int(source_part_id),
            offset_m=float(offset_m),
            tolerance_m=float(tolerance_m),
        ),
        primary_hit_counter_key='triangle_hit_count',
        triangle_surface_3d=triangle_surface_3d,
    )


def build_boundary_service(
    runtime,
    *,
    spatial_dim: int,
    on_boundary_tol_m: float,
    triangle_surface_3d: Optional[TriangleSurface3D],
) -> BoundaryService:
    if int(spatial_dim) == 2:
        return _build_boundary_service_2d(runtime, on_boundary_tol_m=on_boundary_tol_m)
    return _build_boundary_service_3d(
        runtime,
        on_boundary_tol_m=on_boundary_tol_m,
        triangle_surface_3d=triangle_surface_3d,
    )


__all__ = (
    'BoundaryHit',
    'BoundaryEdgeFrame2D',
    'ReleasePointClassification',
    'BoundaryService',
    'build_boundary_service',
    'contact_frame_on_boundary_edge_2d',
    'inside_geometry',
    'inside_geometry_with_boundary',
    'nearest_boundary_edge_features_2d',
    'nearest_hit_on_boundary_edges',
    'nearest_hit_on_boundary_triangles',
    'normalize_polyline_alpha',
    'points_inside_geometry_2d',
    'polyline_hit_from_boundary_edges',
    'polyline_hits_from_boundary_edges_batch',
    'polyline_hit_from_boundary_triangles',
    'polyline_hit_from_loop_bisection',
    'polyline_hit_from_solid_bisection_3d',
    'runtime_bounds',
    'sample_geometry_normal',
    'sample_geometry_part_id',
    'sample_geometry_sdf',
    'segment_hit_from_boundary_edges',
    'segment_hit_from_boundary_triangles',
    'segment_hit_from_loop_bisection',
    'segment_hit_from_solid_bisection_3d',
)
