from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TriangleUniformGrid:
    origin: np.ndarray
    cell_size: np.ndarray
    dims: Tuple[int, int, int]
    cell_to_triangles: Dict[Tuple[int, int, int], np.ndarray]
    triangle_mins: np.ndarray
    triangle_maxs: np.ndarray
    triangle_count: int


@dataclass(frozen=True)
class TriangleSurface3D:
    triangles: np.ndarray
    part_ids: np.ndarray
    normals: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    grid: TriangleUniformGrid


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n <= 1.0e-30:
        return np.zeros_like(arr)
    return arr / n


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangles, dtype=np.float64)
    n = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    mag = np.linalg.norm(n, axis=1)
    mag[mag <= 1.0e-30] = 1.0
    return n / mag[:, None]


def validate_closed_surface_triangles(
    triangles: np.ndarray,
    *,
    round_decimals: int = 12,
    area_tol: float = 1.0e-18,
) -> Dict[str, int]:
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f'boundary_triangles must be shaped as (n, 3, 3), got {tri.shape}')
    if tri.shape[0] == 0:
        raise ValueError('boundary_triangles must be non-empty')

    area_vec = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
    area = 0.5 * np.linalg.norm(area_vec, axis=1)
    degenerate = int(np.count_nonzero(area <= float(max(area_tol, 0.0))))
    if degenerate:
        raise ValueError(f'boundary_triangles contains {degenerate} degenerate triangle(s)')

    verts = tri.reshape(-1, 3)
    rounded = np.round(verts, int(round_decimals))
    lut: Dict[Tuple[float, float, float], int] = {}
    ids = np.zeros(rounded.shape[0], dtype=np.int64)
    for i, row in enumerate(rounded):
        key = (float(row[0]), float(row[1]), float(row[2]))
        idx = lut.get(key)
        if idx is None:
            idx = len(lut)
            lut[key] = idx
        ids[i] = idx
    tri_ids = ids.reshape(tri.shape[0], 3)

    undirected_counts: Dict[Tuple[int, int], int] = {}
    oriented_counts: Dict[Tuple[int, int], int] = {}
    for a, b, c in tri_ids:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            undirected_counts[key] = undirected_counts.get(key, 0) + 1
            okey = (u, v)
            oriented_counts[okey] = oriented_counts.get(okey, 0) + 1

    bad_cardinality = [k for k, cnt in undirected_counts.items() if cnt != 2]
    if bad_cardinality:
        raise ValueError(
            'boundary_triangles must form a closed 2-manifold: '
            f'{len(bad_cardinality)} edge(s) do not have exactly two adjacent triangles'
        )

    bad_orientation = []
    for u, v in undirected_counts:
        fw = int(oriented_counts.get((u, v), 0))
        bw = int(oriented_counts.get((v, u), 0))
        if fw != 1 or bw != 1:
            bad_orientation.append((u, v, fw, bw))
    if bad_orientation:
        raise ValueError(
            'boundary_triangles orientation mismatch detected: '
            f'{len(bad_orientation)} edge(s) are not oppositely oriented across adjacent triangles'
        )

    return {
        'triangle_count': int(tri.shape[0]),
        'unique_vertex_count': int(len(lut)),
        'edge_count': int(len(undirected_counts)),
    }


def build_triangle_uniform_grid(
    triangles: np.ndarray,
    *,
    target_triangles_per_cell: int = 24,
    min_cells_per_axis: int = 4,
    max_cells_per_axis: int = 64,
) -> TriangleUniformGrid:
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f'triangles must be shaped as (n, 3, 3), got {tri.shape}')
    n = int(tri.shape[0])
    if n <= 0:
        raise ValueError('triangles must be non-empty')
    tri_min = np.min(tri, axis=1)
    tri_max = np.max(tri, axis=1)
    bbox_min = np.min(tri_min, axis=0)
    bbox_max = np.max(tri_max, axis=0)
    span = np.maximum(bbox_max - bbox_min, 1.0e-12)
    geom = float(np.exp(np.mean(np.log(span))))
    base = max(1.0, (float(n) / max(float(target_triangles_per_cell), 1.0)) ** (1.0 / 3.0))
    raw_dims = np.round(base * (span / max(geom, 1.0e-12))).astype(np.int32)
    raw_dims = np.clip(raw_dims, int(max(1, min_cells_per_axis)), int(max(max_cells_per_axis, min_cells_per_axis)))
    dims = tuple(int(v) for v in raw_dims.tolist())
    cell = span / np.maximum(np.asarray(dims, dtype=np.float64), 1.0)

    cell_to_triangles: Dict[Tuple[int, int, int], list[int]] = {}
    for idx in range(n):
        lo = np.floor((tri_min[idx] - bbox_min) / cell).astype(np.int64)
        hi = np.floor((tri_max[idx] - bbox_min) / cell).astype(np.int64)
        lo = np.clip(lo, 0, np.asarray(dims, dtype=np.int64) - 1)
        hi = np.clip(hi, 0, np.asarray(dims, dtype=np.int64) - 1)
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    key = (ix, iy, iz)
                    bucket = cell_to_triangles.get(key)
                    if bucket is None:
                        cell_to_triangles[key] = [idx]
                    else:
                        bucket.append(idx)

    packed = {k: np.asarray(sorted(set(v)), dtype=np.int32) for k, v in cell_to_triangles.items()}
    return TriangleUniformGrid(
        origin=np.asarray(bbox_min, dtype=np.float64),
        cell_size=np.asarray(cell, dtype=np.float64),
        dims=dims,
        cell_to_triangles=packed,
        triangle_mins=np.asarray(tri_min, dtype=np.float64),
        triangle_maxs=np.asarray(tri_max, dtype=np.float64),
        triangle_count=n,
    )


def build_triangle_surface(
    triangles: np.ndarray,
    part_ids: Optional[np.ndarray] = None,
    *,
    validate_closed: bool = True,
) -> TriangleSurface3D:
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f'boundary_triangles must be shaped as (n, 3, 3), got {tri.shape}')
    if validate_closed:
        validate_closed_surface_triangles(tri)
    if part_ids is None:
        pid = np.zeros(tri.shape[0], dtype=np.int32)
    else:
        pid = np.asarray(part_ids, dtype=np.int32).reshape(-1)
        if pid.shape[0] != tri.shape[0]:
            raise ValueError(
                f'boundary_triangle_part_ids length mismatch: expected {tri.shape[0]}, got {pid.shape[0]}'
            )
    tri_normals = _triangle_normals(tri)
    bbox_min = np.min(tri.reshape(-1, 3), axis=0)
    bbox_max = np.max(tri.reshape(-1, 3), axis=0)
    grid = build_triangle_uniform_grid(tri)
    return TriangleSurface3D(
        triangles=tri,
        part_ids=pid,
        normals=tri_normals,
        bbox_min=np.asarray(bbox_min, dtype=np.float64),
        bbox_max=np.asarray(bbox_max, dtype=np.float64),
        grid=grid,
    )


def _segment_aabb_overlaps(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> bool:
    return bool(np.all(a_max >= b_min) and np.all(b_max >= a_min))


def query_triangle_candidates(grid: TriangleUniformGrid, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    seg_min = np.minimum(a, b)
    seg_max = np.maximum(a, b)
    dims = np.asarray(grid.dims, dtype=np.int64)
    idx_min = np.floor((seg_min - grid.origin) / grid.cell_size).astype(np.int64)
    idx_max = np.floor((seg_max - grid.origin) / grid.cell_size).astype(np.int64)
    idx_min = np.clip(idx_min, 0, dims - 1)
    idx_max = np.clip(idx_max, 0, dims - 1)
    span_cells = (idx_max - idx_min + 1)
    if int(span_cells[0] * span_cells[1] * span_cells[2]) > 4096:
        return np.arange(grid.triangle_count, dtype=np.int32)
    ids: set[int] = set()
    for ix in range(int(idx_min[0]), int(idx_max[0]) + 1):
        for iy in range(int(idx_min[1]), int(idx_max[1]) + 1):
            for iz in range(int(idx_min[2]), int(idx_max[2]) + 1):
                arr = grid.cell_to_triangles.get((ix, iy, iz))
                if arr is None:
                    continue
                ids.update(int(v) for v in arr.tolist())
    if not ids:
        return np.arange(grid.triangle_count, dtype=np.int32)
    return np.asarray(sorted(ids), dtype=np.int32)


def _segment_triangle_intersection_alpha(
    p0: np.ndarray,
    p1: np.ndarray,
    tri: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> Optional[float]:
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    v0 = np.asarray(tri[0], dtype=np.float64)
    v1 = np.asarray(tri[1], dtype=np.float64)
    v2 = np.asarray(tri[2], dtype=np.float64)
    d = b - a
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(d, e2)
    det = float(np.dot(e1, pvec))
    if abs(det) <= eps:
        return None
    inv_det = 1.0 / det
    tvec = a - v0
    u = float(np.dot(tvec, pvec) * inv_det)
    if u < -eps or u > 1.0 + eps:
        return None
    qvec = np.cross(tvec, e1)
    v = float(np.dot(d, qvec) * inv_det)
    if v < -eps or (u + v) > 1.0 + eps:
        return None
    alpha = float(np.dot(e2, qvec) * inv_det)
    if alpha < -eps or alpha > 1.0 + eps:
        return None
    return float(np.clip(alpha, 0.0, 1.0))


def segment_hit_from_surface(
    surface: TriangleSurface3D,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    alpha_min: float = 1.0e-8,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, int, int]]:
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    if seg_len <= 1.0e-30:
        return None

    seg_min = np.minimum(a, b)
    seg_max = np.maximum(a, b)
    candidate_ids = query_triangle_candidates(surface.grid, a, b)
    best_alpha = 2.0
    best_idx = -1
    for idx in candidate_ids:
        j = int(idx)
        if not _segment_aabb_overlaps(seg_min, seg_max, surface.grid.triangle_mins[j], surface.grid.triangle_maxs[j]):
            continue
        alpha = _segment_triangle_intersection_alpha(a, b, surface.triangles[j])
        if alpha is None or alpha < float(alpha_min):
            continue
        if alpha < best_alpha:
            best_alpha = float(alpha)
            best_idx = j
    if best_idx < 0:
        return None

    hit = a + best_alpha * seg
    normal = np.asarray(surface.normals[best_idx], dtype=np.float64)
    if float(np.dot(a - hit, normal)) > 0.0:
        normal = -normal
    normal = _normalize(normal)
    part_id = int(surface.part_ids[best_idx]) if best_idx < surface.part_ids.size else 0
    return hit, normal, float(np.clip(best_alpha, 0.0, 1.0)), max(0, part_id), int(best_idx)


def _closest_point_on_triangle(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    # Real-Time Collision Detection (Christer Ericson), closest point on triangle.
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(tri[0], dtype=np.float64)
    b = np.asarray(tri[1], dtype=np.float64)
    c = np.asarray(tri[2], dtype=np.float64)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1.0e-30)
        return a + v * ab
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1.0e-30)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-30)
        return b + w * (c - b)
    denom = max(va + vb + vc, 1.0e-30)
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


def nearest_surface_point(
    surface: TriangleSurface3D,
    point: np.ndarray,
    *,
    inside_reference: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    p = np.asarray(point, dtype=np.float64)
    r = max(float(np.max(surface.grid.cell_size)), 1.0e-12)
    seg0 = p - np.array([r, r, r], dtype=np.float64)
    seg1 = p + np.array([r, r, r], dtype=np.float64)
    candidate_ids = query_triangle_candidates(surface.grid, seg0, seg1)
    if candidate_ids.size == 0:
        candidate_ids = np.arange(surface.triangles.shape[0], dtype=np.int32)
    best_dist = np.inf
    best_point = None
    best_idx = -1
    for idx in candidate_ids:
        j = int(idx)
        cp = _closest_point_on_triangle(p, surface.triangles[j])
        dist = float(np.linalg.norm(cp - p))
        if dist < best_dist:
            best_dist = dist
            best_point = cp
            best_idx = j
    if best_point is None or best_idx < 0:
        raise ValueError('No surface triangle available for nearest projection')
    normal = np.asarray(surface.normals[best_idx], dtype=np.float64)
    if inside_reference is not None:
        ref = np.asarray(inside_reference, dtype=np.float64)
        if float(np.dot(ref - best_point, normal)) > 0.0:
            normal = -normal
    normal = _normalize(normal)
    part_id = int(surface.part_ids[best_idx]) if best_idx < surface.part_ids.size else 0
    return np.asarray(best_point, dtype=np.float64), normal, max(0, part_id), int(best_idx)


def point_inside_surface(
    surface: TriangleSurface3D,
    point: np.ndarray,
    *,
    on_boundary_tol: float = 1.0e-7,
) -> Tuple[bool, bool]:
    p = np.asarray(point, dtype=np.float64)
    if np.any(p < surface.bbox_min - 1.0e-12) or np.any(p > surface.bbox_max + 1.0e-12):
        return False, False

    nearest_point, _, _, _ = nearest_surface_point(surface, p)
    if float(np.linalg.norm(nearest_point - p)) <= float(max(on_boundary_tol, 0.0)):
        return True, True

    span_x = max(float(surface.bbox_max[0] - surface.bbox_min[0]), 1.0)
    ray_end = np.asarray(
        [
            float(surface.bbox_max[0] + 2.5 * span_x + 1.0),
            float(p[1] + 1.0e-9),
            float(p[2] + 2.0e-9),
        ],
        dtype=np.float64,
    )
    candidate_ids = query_triangle_candidates(surface.grid, p, ray_end)
    alphas: list[float] = []
    for idx in candidate_ids:
        j = int(idx)
        alpha = _segment_triangle_intersection_alpha(p, ray_end, surface.triangles[j], eps=1.0e-12)
        if alpha is None:
            continue
        if alpha <= 1.0e-8:
            continue
        alphas.append(float(alpha))
    if not alphas:
        return False, False
    alphas = sorted(alphas)
    unique = []
    last = -1.0
    for a in alphas:
        if not unique or abs(a - last) > 1.0e-7:
            unique.append(a)
            last = a
    inside = (len(unique) % 2) == 1
    return bool(inside), False
