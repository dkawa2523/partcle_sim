from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Optional, Tuple

import numpy as np
from numba import njit

from .boundary_core import (
    BoundaryHit,
    inside_geometry,
    inside_geometry_with_boundary,
    sample_geometry_normal,
    sample_geometry_part_id,
)
from .geometry3d import TriangleSurface3D, nearest_surface_point, segment_hit_from_surface

_EDGE_ENDPOINT_TOL = 1.0e-9
_TRIANGLE_EDGE_TOL = 1.0e-8


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


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


def _point_triangle_barycentric(point: np.ndarray, tri: np.ndarray) -> Optional[np.ndarray]:
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(tri[0], dtype=np.float64)
    b = np.asarray(tri[1], dtype=np.float64)
    c = np.asarray(tri[2], dtype=np.float64)
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
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.asarray([u, v, w], dtype=np.float64)


class BoundaryEdgeFrame2D(NamedTuple):
    edge_index: int
    start: np.ndarray
    end: np.ndarray
    projection: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    part_id: int
    alpha: float
    length: float
    distance: float


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
    best_idx = -1
    best_u = np.nan
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
        best_idx = int(idx)
        best_u = float(np.clip(u, 0.0, 1.0))
    if best_hit is None or best_normal is None:
        return None
    return BoundaryHit(
        position=np.asarray(best_hit, dtype=np.float64),
        normal=np.asarray(best_normal, dtype=np.float64),
        part_id=max(0, int(best_part_id)),
        alpha_hint=float(np.clip(best_alpha, 0.0, 1.0)),
        primitive_id=int(best_idx),
        primitive_kind='edge',
        is_ambiguous=bool(np.isfinite(best_u) and (best_u <= _EDGE_ENDPOINT_TOL or best_u >= 1.0 - _EDGE_ENDPOINT_TOL)),
    )


def _boundary_edges_2d(runtime) -> Tuple[Optional[np.ndarray], np.ndarray]:
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None:
        return None, np.zeros(0, dtype=np.int32)
    geom = geometry_provider.geometry
    if int(geom.spatial_dim) != 2 or geom.boundary_edges is None:
        return None, np.zeros(0, dtype=np.int32)
    segments = np.asarray(geom.boundary_edges, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or segments.shape[0] == 0:
        return None, np.zeros(0, dtype=np.int32)
    part_ids = np.asarray(
        geom.boundary_edge_part_ids if geom.boundary_edge_part_ids is not None else np.zeros(segments.shape[0], dtype=np.int32),
        dtype=np.int32,
    )
    if part_ids.size < segments.shape[0]:
        part_ids = np.pad(part_ids, (0, int(segments.shape[0] - part_ids.size)), constant_values=0)
    elif part_ids.size > segments.shape[0]:
        part_ids = part_ids[: segments.shape[0]]
    return segments, part_ids


@njit(cache=True)
def _segment_hits_from_boundary_edges_batch_kernel(
    edge_arr: np.ndarray,
    edge_part_ids: np.ndarray,
    starts_arr: np.ndarray,
    ends_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_points = starts_arr.shape[0]
    n_edges = edge_arr.shape[0]
    hit_mask = np.zeros(n_points, dtype=np.bool_)
    hit_positions = np.empty((n_points, 2), dtype=np.float64)
    hit_normals = np.empty((n_points, 2), dtype=np.float64)
    hit_part_ids = np.zeros(n_points, dtype=np.int32)
    hit_alphas = np.empty(n_points, dtype=np.float64)
    hit_primitive_ids = np.empty(n_points, dtype=np.int32)
    hit_ambiguous = np.zeros(n_points, dtype=np.bool_)
    for i in range(n_points):
        hit_positions[i, 0] = np.nan
        hit_positions[i, 1] = np.nan
        hit_normals[i, 0] = np.nan
        hit_normals[i, 1] = np.nan
        hit_alphas[i] = np.inf
        hit_primitive_ids[i] = -1

    eps = 1.0e-12
    for i in range(n_points):
        ax = starts_arr[i, 0]
        ay = starts_arr[i, 1]
        bx = ends_arr[i, 0]
        by = ends_arr[i, 1]
        rx = bx - ax
        ry = by - ay
        if rx * rx + ry * ry <= 1.0e-30:
            continue
        seg_min_x = min(ax, bx) - eps
        seg_max_x = max(ax, bx) + eps
        seg_min_y = min(ay, by) - eps
        seg_max_y = max(ay, by) + eps
        best_alpha = 2.0
        best_edge = -1
        best_u = np.nan
        best_hit_x = np.nan
        best_hit_y = np.nan
        best_nx = np.nan
        best_ny = np.nan

        for edge_idx in range(n_edges):
            q0x = edge_arr[edge_idx, 0, 0]
            q0y = edge_arr[edge_idx, 0, 1]
            q1x = edge_arr[edge_idx, 1, 0]
            q1y = edge_arr[edge_idx, 1, 1]
            edge_min_x = min(q0x, q1x)
            edge_max_x = max(q0x, q1x)
            edge_min_y = min(q0y, q1y)
            edge_max_y = max(q0y, q1y)
            if seg_min_x > edge_max_x or seg_max_x < edge_min_x or seg_min_y > edge_max_y or seg_max_y < edge_min_y:
                continue
            sx = q1x - q0x
            sy = q1y - q0y
            denominator = rx * sy - ry * sx
            if abs(denominator) <= 1.0e-30:
                continue
            qmp_x = q0x - ax
            qmp_y = q0y - ay
            t = (qmp_x * sy - qmp_y * sx) / denominator
            u = (qmp_x * ry - qmp_y * rx) / denominator
            if t < -eps or t > 1.0 + eps or u < -eps or u > 1.0 + eps:
                continue
            alpha = min(1.0, max(0.0, t))
            if alpha >= best_alpha:
                continue
            hit_x = ax + alpha * rx
            hit_y = ay + alpha * ry
            nx = -sy
            ny = sx
            mag = (nx * nx + ny * ny) ** 0.5
            if mag <= 1.0e-30:
                continue
            nx /= mag
            ny /= mag
            if (ax - hit_x) * nx + (ay - hit_y) * ny > 0.0:
                nx = -nx
                ny = -ny
            best_alpha = alpha
            best_edge = edge_idx
            best_u = min(1.0, max(0.0, u))
            best_hit_x = hit_x
            best_hit_y = hit_y
            best_nx = nx
            best_ny = ny

        if best_edge >= 0:
            hit_mask[i] = True
            hit_positions[i, 0] = best_hit_x
            hit_positions[i, 1] = best_hit_y
            hit_normals[i, 0] = best_nx
            hit_normals[i, 1] = best_ny
            hit_part_ids[i] = edge_part_ids[best_edge]
            hit_alphas[i] = best_alpha
            hit_primitive_ids[i] = best_edge
            hit_ambiguous[i] = best_u <= _EDGE_ENDPOINT_TOL or best_u >= 1.0 - _EDGE_ENDPOINT_TOL
    return hit_mask, hit_positions, hit_normals, hit_part_ids, hit_alphas, hit_primitive_ids, hit_ambiguous


def _segment_hits_from_boundary_edges_batch(
    segments: np.ndarray,
    part_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts_arr = np.asarray(starts, dtype=np.float64)
    ends_arr = np.asarray(ends, dtype=np.float64)
    edge_arr = np.asarray(segments, dtype=np.float64)
    edge_part_ids = np.asarray(part_ids, dtype=np.int32)
    n_points = int(starts_arr.shape[0])
    if n_points == 0 or edge_arr.shape[0] == 0:
        hit_mask = np.zeros(n_points, dtype=bool)
        hit_positions = np.full((n_points, 2), np.nan, dtype=np.float64)
        hit_normals = np.full((n_points, 2), np.nan, dtype=np.float64)
        hit_part_ids = np.zeros(n_points, dtype=np.int32)
        hit_alphas = np.full(n_points, np.inf, dtype=np.float64)
        hit_primitive_ids = np.full(n_points, -1, dtype=np.int32)
        hit_ambiguous = np.zeros(n_points, dtype=bool)
        return hit_mask, hit_positions, hit_normals, hit_part_ids, hit_alphas, hit_primitive_ids, hit_ambiguous
    return _segment_hits_from_boundary_edges_batch_kernel(edge_arr, edge_part_ids, starts_arr, ends_arr)


def polyline_hits_from_boundary_edges_batch(
    runtime,
    starts: np.ndarray,
    stage_points: np.ndarray,
    *,
    particle_indices: Optional[np.ndarray] = None,
    chunk_size: int = 2048,
) -> Dict[int, BoundaryHit]:
    segments, part_ids = _boundary_edges_2d(runtime)
    if segments is None:
        return {}
    starts_arr = np.asarray(starts, dtype=np.float64)
    stages = np.asarray(stage_points, dtype=np.float64)
    if starts_arr.ndim != 2 or starts_arr.shape[1] != 2:
        raise ValueError('2D batch boundary hit starts require shape (n, 2)')
    if stages.ndim != 3 or stages.shape[0] != starts_arr.shape[0] or stages.shape[2] != 2:
        raise ValueError('2D batch boundary hit stage_points require shape (n, k, 2)')
    n_points = int(starts_arr.shape[0])
    if particle_indices is None:
        particle_ids = np.arange(n_points, dtype=np.int64)
    else:
        particle_ids = np.asarray(particle_indices, dtype=np.int64)
        if particle_ids.shape[0] != n_points:
            raise ValueError('particle_indices length must match starts')

    hit_results: Dict[int, BoundaryHit] = {}
    active_rows = np.ones(n_points, dtype=bool)
    segment_start = starts_arr
    segment_count = max(1, int(stages.shape[1]))
    for segment_index in range(segment_count):
        rows = np.flatnonzero(active_rows)
        if rows.size == 0:
            break
        segment_end = stages[rows, segment_index, :]
        hit_mask, hit_pos, hit_normal, hit_part, hit_alpha, hit_primitive, hit_ambiguous = _segment_hits_from_boundary_edges_batch(
            segments,
            part_ids,
            segment_start[rows],
            segment_end,
            chunk_size=int(chunk_size),
        )
        hit_rows_local = np.flatnonzero(hit_mask)
        if hit_rows_local.size:
            original_rows = rows[hit_rows_local]
            normalized_alpha = (
                float(segment_index) + np.clip(hit_alpha[hit_rows_local], 0.0, 1.0)
            ) / float(segment_count)
            for output_row, local_idx, alpha_value in zip(original_rows, hit_rows_local, normalized_alpha):
                hit_results[int(particle_ids[int(output_row)])] = BoundaryHit(
                    position=np.asarray(hit_pos[int(local_idx)], dtype=np.float64),
                    normal=np.asarray(hit_normal[int(local_idx)], dtype=np.float64),
                    part_id=max(0, int(hit_part[int(local_idx)])),
                    alpha_hint=float(np.clip(alpha_value, 0.0, 1.0)),
                    primitive_id=int(hit_primitive[int(local_idx)]),
                    primitive_kind='edge',
                    is_ambiguous=bool(hit_ambiguous[int(local_idx)]),
                )
            active_rows[original_rows] = False
        segment_start = stages[:, segment_index, :]
    return hit_results


def nearest_boundary_edge_features_2d(
    runtime,
    points: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    segments, part_ids = _boundary_edges_2d(runtime)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('2D nearest-boundary diagnostics require shape (n, 2)')
    n_points = int(pts.shape[0])
    nearest_part_ids = np.zeros(n_points, dtype=np.int32)
    nearest_distances = np.full(n_points, np.nan, dtype=np.float64)
    if segments is None or n_points == 0:
        return nearest_part_ids, nearest_distances
    q0 = segments[:, 0, :]
    q1 = segments[:, 1, :]
    edge = q1 - q0
    edge_len2 = np.einsum('ij,ij->i', edge, edge)
    valid_edge = edge_len2 > 1.0e-30
    chunk = max(1, int(chunk_size))
    for start_idx in range(0, n_points, chunk):
        stop_idx = min(n_points, start_idx + chunk)
        p = pts[start_idx:stop_idx]
        alpha = np.zeros((stop_idx - start_idx, segments.shape[0]), dtype=np.float64)
        if np.any(valid_edge):
            alpha[:, valid_edge] = (
                (p[:, None, 0] - q0[None, valid_edge, 0]) * edge[None, valid_edge, 0]
                + (p[:, None, 1] - q0[None, valid_edge, 1]) * edge[None, valid_edge, 1]
            ) / edge_len2[None, valid_edge]
        alpha = np.clip(alpha, 0.0, 1.0)
        proj_x = q0[None, :, 0] + alpha * edge[None, :, 0]
        proj_y = q0[None, :, 1] + alpha * edge[None, :, 1]
        dist2 = (p[:, None, 0] - proj_x) ** 2 + (p[:, None, 1] - proj_y) ** 2
        dist2[:, ~valid_edge] = np.inf
        best_edge = np.argmin(dist2, axis=1)
        best_dist2 = dist2[np.arange(stop_idx - start_idx), best_edge]
        finite = np.isfinite(best_dist2)
        rows = np.arange(start_idx, stop_idx)
        nearest_part_ids[rows[finite]] = np.asarray(part_ids[best_edge[finite]], dtype=np.int32)
        nearest_distances[rows[finite]] = np.sqrt(best_dist2[finite])
    return nearest_part_ids, nearest_distances


def contact_frame_on_boundary_edge_2d(
    runtime,
    point: np.ndarray,
    *,
    part_id_hint: int = 0,
    normal_hint: Optional[np.ndarray] = None,
) -> Optional[BoundaryEdgeFrame2D]:
    segments, part_ids = _boundary_edges_2d(runtime)
    if segments is None:
        return None
    point_arr = np.asarray(point, dtype=np.float64)
    if point_arr.shape[0] != 2 or not np.all(np.isfinite(point_arr)):
        return None
    hint = int(part_id_hint)
    normal_ref = None if normal_hint is None else np.asarray(normal_hint, dtype=np.float64)
    if normal_ref is not None and (normal_ref.shape[0] != 2 or not np.all(np.isfinite(normal_ref))):
        normal_ref = None

    best: Optional[BoundaryEdgeFrame2D] = None
    best_dist = np.inf
    for idx, segment in enumerate(np.asarray(segments, dtype=np.float64)):
        part_id = int(part_ids[idx]) if idx < part_ids.size else 0
        if hint > 0 and part_id != hint:
            continue
        q0 = np.asarray(segment[0], dtype=np.float64)
        q1 = np.asarray(segment[1], dtype=np.float64)
        edge = q1 - q0
        length = float(np.linalg.norm(edge))
        if length <= 1.0e-30:
            continue
        tangent = edge / length
        alpha = float(np.clip(np.dot(point_arr - q0, edge) / (length * length), 0.0, 1.0))
        projection = q0 + alpha * edge
        dist = float(np.linalg.norm(point_arr - projection))
        if dist >= best_dist:
            continue
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        if normal_ref is not None and float(np.dot(normal, normal_ref)) < 0.0:
            normal = -normal
        elif normal_ref is None and float(np.dot(point_arr - projection, normal)) > 0.0:
            normal = -normal
        best_dist = dist
        best = BoundaryEdgeFrame2D(
            edge_index=int(idx),
            start=q0,
            end=q1,
            projection=np.asarray(projection, dtype=np.float64),
            normal=np.asarray(normal, dtype=np.float64),
            tangent=np.asarray(tangent, dtype=np.float64),
            part_id=max(0, int(part_id)),
            alpha=float(alpha),
            length=float(length),
            distance=float(dist),
        )
    if best is not None or hint <= 0:
        return best
    return contact_frame_on_boundary_edge_2d(
        runtime,
        point_arr,
        part_id_hint=0,
        normal_hint=normal_ref,
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
    best_idx = -1
    best_alpha = np.nan
    for idx, segment in enumerate(segments):
        q0 = segment[0]
        q1 = segment[1]
        edge = q1 - q0
        denominator = float(np.dot(edge, edge))
        if denominator <= 1e-30:
            hit = q0.copy()
            alpha = 0.0
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
        best_idx = int(idx)
        best_alpha = float(alpha)
    if best_hit is None or best_normal is None:
        return None
    return BoundaryHit(
        position=np.asarray(best_hit, dtype=np.float64),
        normal=np.asarray(best_normal, dtype=np.float64),
        part_id=max(0, int(best_part_id)),
        alpha_hint=0.0,
        primitive_id=int(best_idx),
        primitive_kind='edge_projection',
        is_ambiguous=bool(np.isfinite(best_alpha) and (best_alpha <= _EDGE_ENDPOINT_TOL or best_alpha >= 1.0 - _EDGE_ENDPOINT_TOL)),
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
        return _copy_hit_with_alpha(nearest, alpha)
    return BoundaryHit(
        position=np.asarray(hit, dtype=np.float64),
        normal=np.asarray(sample_geometry_normal(runtime, hit), dtype=np.float64),
        part_id=int(sample_geometry_part_id(runtime, hit)),
        alpha_hint=alpha,
        primitive_kind='loop_bisection',
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
    point, normal, alpha, part_id, triangle_index = hit
    bary = _point_triangle_barycentric(point, triangle_surface_3d.triangles[int(triangle_index)])
    ambiguous = False if bary is None else bool(np.min(bary) <= _TRIANGLE_EDGE_TOL)
    return BoundaryHit(
        position=np.asarray(point, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        part_id=int(part_id),
        alpha_hint=float(alpha),
        primitive_id=int(triangle_index),
        primitive_kind='triangle',
        is_ambiguous=bool(ambiguous),
    )


def nearest_hit_on_boundary_triangles(
    triangle_surface_3d: Optional[TriangleSurface3D],
    point: np.ndarray,
    inside_reference: np.ndarray,
) -> Optional[BoundaryHit]:
    if triangle_surface_3d is None:
        return None
    try:
        hit, normal, part_id, triangle_index = nearest_surface_point(
            triangle_surface_3d,
            np.asarray(point, dtype=np.float64),
            inside_reference=np.asarray(inside_reference, dtype=np.float64),
        )
    except ValueError:
        return None
    bary = _point_triangle_barycentric(hit, triangle_surface_3d.triangles[int(triangle_index)])
    ambiguous = False
    if bary is not None:
        ambiguous = bool(np.any(np.asarray(bary, dtype=np.float64) <= _TRIANGLE_EDGE_TOL))
    return BoundaryHit(
        position=np.asarray(hit, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        part_id=int(part_id),
        alpha_hint=0.0,
        primitive_id=int(triangle_index),
        primitive_kind='triangle_projection',
        is_ambiguous=bool(ambiguous),
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
    return _copy_hit_with_alpha(nearest, alpha)


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
            return _copy_hit_with_alpha(
                hit,
                normalize_polyline_alpha(segment_index, hit.alpha_hint, segment_count),
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
    'nearest_boundary_edge_features_2d',
    'nearest_hit_on_boundary_triangles',
    'normalize_polyline_alpha',
    'polyline_hit_from_boundary_edges',
    'polyline_hits_from_boundary_edges_batch',
    'polyline_hit_from_boundary_triangles',
    'polyline_hit_from_loop_bisection',
    'polyline_hit_from_solid_bisection_3d',
    'segment_hit_from_boundary_edges',
    'segment_hit_from_boundary_triangles',
    'segment_hit_from_loop_bisection',
    'segment_hit_from_solid_bisection_3d',
)
