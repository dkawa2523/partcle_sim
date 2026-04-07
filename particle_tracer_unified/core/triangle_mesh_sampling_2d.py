from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .field_sampling import VALID_MASK_STATUS_CLEAN, VALID_MASK_STATUS_HARD_INVALID


def build_triangle_candidate_grid(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError('Triangle mesh vertices must have shape (n, 2)')
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError('Triangle mesh triangles must have shape (m, 3)')
    if tris.size == 0:
        raise ValueError('Triangle mesh must contain at least one triangle')

    bbox_min = np.min(verts, axis=0)
    bbox_max = np.max(verts, axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1.0e-12)
    tri_count = int(tris.shape[0])
    aspect = float(extent[0] / max(extent[1], 1.0e-12))
    base = max(1.0, np.sqrt(float(tri_count)))
    nx = max(1, int(np.ceil(base * np.sqrt(max(aspect, 1.0e-12)))))
    ny = max(1, int(np.ceil(base / np.sqrt(max(aspect, 1.0e-12)))))
    cell_size = np.asarray([extent[0] / float(nx), extent[1] / float(ny)], dtype=np.float64)
    cell_size = np.maximum(cell_size, 1.0e-12)

    cell_lists = [[] for _ in range(nx * ny)]
    tri_pts = verts[tris]
    tri_min = np.min(tri_pts, axis=1)
    tri_max = np.max(tri_pts, axis=1)
    for tri_idx in range(tri_count):
        ix0 = int(np.clip(np.floor((tri_min[tri_idx, 0] - bbox_min[0]) / cell_size[0]), 0, nx - 1))
        ix1 = int(np.clip(np.floor((tri_max[tri_idx, 0] - bbox_min[0]) / cell_size[0]), 0, nx - 1))
        iy0 = int(np.clip(np.floor((tri_min[tri_idx, 1] - bbox_min[1]) / cell_size[1]), 0, ny - 1))
        iy1 = int(np.clip(np.floor((tri_max[tri_idx, 1] - bbox_min[1]) / cell_size[1]), 0, ny - 1))
        for ix in range(ix0, ix1 + 1):
            row_offset = ix * ny
            for iy in range(iy0, iy1 + 1):
                cell_lists[row_offset + iy].append(tri_idx)

    offsets = np.zeros(nx * ny + 1, dtype=np.int32)
    flat = []
    cursor = 0
    for idx, entries in enumerate(cell_lists):
        offsets[idx] = cursor
        flat.extend(entries)
        cursor += len(entries)
    offsets[-1] = cursor
    return (
        np.asarray(bbox_min, dtype=np.float64),
        np.asarray(cell_size, dtype=np.float64),
        (int(nx), int(ny)),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(flat, dtype=np.int32),
    )


def locate_triangle_containing_point(
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    accel_origin: np.ndarray,
    accel_cell_size: np.ndarray,
    accel_shape: Sequence[int],
    accel_cell_offsets: np.ndarray,
    accel_triangle_indices: np.ndarray,
    position: np.ndarray,
    eps: float = 2.0e-6,
) -> Tuple[int, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int32)
    point = np.asarray(position, dtype=np.float64)
    origin = np.asarray(accel_origin, dtype=np.float64)
    cell = np.asarray(accel_cell_size, dtype=np.float64)
    nx = int(accel_shape[0])
    ny = int(accel_shape[1])
    if point[0] < float(origin[0]) - eps or point[1] < float(origin[1]) - eps:
        return -1, np.zeros(3, dtype=np.float64)
    extent_max = origin + cell * np.asarray([nx, ny], dtype=np.float64)
    if point[0] > float(extent_max[0]) + eps or point[1] > float(extent_max[1]) + eps:
        return -1, np.zeros(3, dtype=np.float64)

    ix = int(np.clip(np.floor((point[0] - origin[0]) / cell[0]), 0, nx - 1))
    iy = int(np.clip(np.floor((point[1] - origin[1]) / cell[1]), 0, ny - 1))
    cell_id = ix * ny + iy
    start = int(np.asarray(accel_cell_offsets, dtype=np.int32)[cell_id])
    stop = int(np.asarray(accel_cell_offsets, dtype=np.int32)[cell_id + 1])
    if stop <= start:
        return -1, np.zeros(3, dtype=np.float64)

    tri_indices = np.asarray(accel_triangle_indices, dtype=np.int32)
    best_idx = -1
    best_bary = np.zeros(3, dtype=np.float64)
    best_margin = -np.inf
    for flat_idx in range(start, stop):
        tri_idx = int(tri_indices[flat_idx])
        tri = verts[tris[tri_idx]]
        a = tri[0]
        b = tri[1]
        c = tri[2]
        v0 = b - a
        v1 = c - a
        v2 = point - a
        den = float(v0[0] * v1[1] - v0[1] * v1[0])
        if abs(den) <= 1.0e-30:
            continue
        beta = (v2[0] * v1[1] - v2[1] * v1[0]) / den
        gamma = (v0[0] * v2[1] - v0[1] * v2[0]) / den
        alpha = 1.0 - beta - gamma
        margin = min(alpha, beta, gamma)
        if margin < -float(eps):
            continue
        if margin > best_margin:
            best_margin = margin
            best_idx = tri_idx
            best_bary[0] = alpha
            best_bary[1] = beta
            best_bary[2] = gamma
    return int(best_idx), np.asarray(best_bary, dtype=np.float64)


def sample_triangle_mesh_status(field, position: np.ndarray) -> int:
    tri_idx, _bary = locate_triangle_containing_point(
        vertices=field.mesh_vertices,
        triangles=field.mesh_triangles,
        accel_origin=field.accel_origin,
        accel_cell_size=field.accel_cell_size,
        accel_shape=field.accel_shape,
        accel_cell_offsets=field.accel_cell_offsets,
        accel_triangle_indices=field.accel_triangle_indices,
        position=np.asarray(position, dtype=np.float64),
        eps=float(getattr(field, 'metadata', {}).get('support_tolerance_m', 2.0e-6)),
    )
    return int(VALID_MASK_STATUS_CLEAN if tri_idx >= 0 else VALID_MASK_STATUS_HARD_INVALID)


def sample_triangle_mesh_series(series, field, position: np.ndarray, t_eval: float, *, mode: str = 'linear') -> float:
    tri_idx, bary = locate_triangle_containing_point(
        vertices=field.mesh_vertices,
        triangles=field.mesh_triangles,
        accel_origin=field.accel_origin,
        accel_cell_size=field.accel_cell_size,
        accel_shape=field.accel_shape,
        accel_cell_offsets=field.accel_cell_offsets,
        accel_triangle_indices=field.accel_triangle_indices,
        position=np.asarray(position, dtype=np.float64),
        eps=float(getattr(field, 'metadata', {}).get('support_tolerance_m', 2.0e-6)),
    )
    if tri_idx < 0:
        return float('nan')
    tri = np.asarray(field.mesh_triangles, dtype=np.int32)[tri_idx]
    data = np.asarray(series.data, dtype=np.float64)
    times = np.asarray(series.times, dtype=np.float64)
    if data.ndim == 1:
        return float(np.dot(np.asarray(bary, dtype=np.float64), data[tri]))
    if data.shape[0] <= 1 or times.size <= 1:
        return float(np.dot(np.asarray(bary, dtype=np.float64), data[0, tri]))
    if float(t_eval) <= float(times[0]):
        return float(np.dot(np.asarray(bary, dtype=np.float64), data[0, tri]))
    if float(t_eval) >= float(times[-1]):
        return float(np.dot(np.asarray(bary, dtype=np.float64), data[-1, tri]))
    hi = int(np.searchsorted(times, float(t_eval)))
    lo = hi - 1
    if str(mode).strip().lower() == 'nearest':
        idx = hi if abs(float(times[hi]) - float(t_eval)) < abs(float(t_eval) - float(times[lo])) else lo
        return float(np.dot(np.asarray(bary, dtype=np.float64), data[idx, tri]))
    t_lo = float(times[lo])
    t_hi = float(times[hi])
    alpha_t = 0.0 if abs(t_hi - t_lo) <= 1.0e-30 else (float(t_eval) - t_lo) / (t_hi - t_lo)
    v_lo = float(np.dot(np.asarray(bary, dtype=np.float64), data[lo, tri]))
    v_hi = float(np.dot(np.asarray(bary, dtype=np.float64), data[hi, tri]))
    return float(v_lo * (1.0 - alpha_t) + v_hi * alpha_t)
