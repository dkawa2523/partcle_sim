"""Locate and interpolate values on two-dimensional triangle meshes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from .boundary_numerics import scaled_classification_tolerance
from .field_sampling import VALID_MASK_STATUS_CLEAN, VALID_MASK_STATUS_HARD_INVALID


def triangle_mesh_support_tolerance(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> float:
    """Resolve a length tolerance from triangle altitude/edge resolution."""

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int32)
    tri_pts = verts[tris]
    edge_lengths = tuple(
        np.linalg.norm(tri_pts[:, j, :] - tri_pts[:, i, :], axis=1)
        for i, j in ((0, 1), (1, 2), (2, 0))
    )
    area2 = np.abs(
        (tri_pts[:, 1, 0] - tri_pts[:, 0, 0]) * (tri_pts[:, 2, 1] - tri_pts[:, 0, 1])
        - (tri_pts[:, 1, 1] - tri_pts[:, 0, 1]) * (tri_pts[:, 2, 0] - tri_pts[:, 0, 0])
    )
    resolved = [values[np.isfinite(values) & (values > 0.0)] for values in edge_lengths]
    for values in edge_lengths:
        valid = values > 0.0
        resolved.append(area2[valid] / values[valid])
    nonempty = [values for values in resolved if values.size]
    if not nonempty:
        raise ValueError("triangle mesh has no positive edge or altitude")
    resolution = min(float(np.min(values)) for values in nonempty)
    _roundoff, tolerance = scaled_classification_tolerance(verts, resolution)
    return float(tolerance)


def field_triangle_support_tolerance(field: object) -> float:
    """Return the mesh-derived support tolerance; metadata cannot tune it."""

    mesh_field = cast(Any, field)
    return triangle_mesh_support_tolerance(
        mesh_field.mesh_vertices,
        mesh_field.mesh_triangles,
    )


def _validated_mesh_arrays(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError("Triangle mesh vertices must have shape (n, 2)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("Triangle mesh triangles must have shape (m, 3)")
    if tris.size == 0:
        raise ValueError("Triangle mesh must contain at least one triangle")
    return verts, tris


def _candidate_grid_geometry(
    vertices: np.ndarray,
    triangle_count: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    extent = bbox_max - bbox_min
    if np.any(~np.isfinite(extent)) or np.any(extent <= 0.0):
        raise ValueError("Triangle mesh must have positive finite span on both axes")
    aspect = float(extent[0] / extent[1])
    base = max(1.0, np.sqrt(float(triangle_count)))
    nx = max(1, int(np.ceil(base * np.sqrt(aspect))))
    ny = max(1, int(np.ceil(base / np.sqrt(aspect))))
    cell_size = np.asarray(
        [extent[0] / float(nx), extent[1] / float(ny)],
        dtype=np.float64,
    )
    return bbox_min, cell_size, nx, ny


def _populate_candidate_cells(
    vertices: np.ndarray,
    triangles: np.ndarray,
    origin: np.ndarray,
    cell_size: np.ndarray,
    nx: int,
    ny: int,
) -> list[list[int]]:
    cells: list[list[int]] = [[] for _ in range(nx * ny)]
    triangle_points = vertices[triangles]
    triangle_min = np.min(triangle_points, axis=1)
    triangle_max = np.max(triangle_points, axis=1)
    for triangle_index in range(int(triangles.shape[0])):
        ix0 = int(
            np.clip(
                np.floor((triangle_min[triangle_index, 0] - origin[0]) / cell_size[0]),
                0,
                nx - 1,
            )
        )
        ix1 = int(
            np.clip(
                np.floor((triangle_max[triangle_index, 0] - origin[0]) / cell_size[0]),
                0,
                nx - 1,
            )
        )
        iy0 = int(
            np.clip(
                np.floor((triangle_min[triangle_index, 1] - origin[1]) / cell_size[1]),
                0,
                ny - 1,
            )
        )
        iy1 = int(
            np.clip(
                np.floor((triangle_max[triangle_index, 1] - origin[1]) / cell_size[1]),
                0,
                ny - 1,
            )
        )
        for ix in range(ix0, ix1 + 1):
            row_offset = ix * ny
            for iy in range(iy0, iy1 + 1):
                cells[row_offset + iy].append(triangle_index)
    return cells


def _flatten_candidate_cells(
    cells: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(cells) + 1, dtype=np.int32)
    triangle_indices: list[int] = []
    cursor = 0
    for cell_index, entries in enumerate(cells):
        offsets[cell_index] = cursor
        triangle_indices.extend(entries)
        cursor += len(entries)
    offsets[-1] = cursor
    return offsets, np.asarray(triangle_indices, dtype=np.int32)


def build_triangle_candidate_grid(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray, np.ndarray]:
    """Index each triangle under every acceleration-grid cell it overlaps."""

    verts, tris = _validated_mesh_arrays(vertices, triangles)
    origin, cell_size, nx, ny = _candidate_grid_geometry(verts, int(tris.shape[0]))
    cells = _populate_candidate_cells(
        verts,
        tris,
        origin,
        cell_size,
        nx,
        ny,
    )
    offsets, triangle_indices = _flatten_candidate_cells(cells)
    return (
        np.asarray(origin, dtype=np.float64),
        np.asarray(cell_size, dtype=np.float64),
        (int(nx), int(ny)),
        offsets,
        triangle_indices,
    )


# A trial step that crosses a wall lands just outside the mesh.  The closest
# element is then within a couple of acceleration cells, so bounding the
# fallback search keeps it local: a point genuinely far from the mesh still
# resolves to nothing and stays a hard support failure.
_OUTSIDE_MESH_RING_LIMIT = 2


def _ring_triangle_indices(
    accel_cell_offsets: np.ndarray,
    accel_triangle_indices: np.ndarray,
    nx: int,
    ny: int,
    ix: int,
    iy: int,
    ring: int,
) -> np.ndarray:
    """Return every candidate triangle in one square ring of cells."""

    chunks: list[np.ndarray] = []
    for cell_x in range(max(0, ix - ring), min(nx, ix + ring + 1)):
        for cell_y in range(max(0, iy - ring), min(ny, iy + ring + 1)):
            if ring > 0 and abs(cell_x - ix) != ring and abs(cell_y - iy) != ring:
                continue
            cell_id = cell_x * ny + cell_y
            lo = int(accel_cell_offsets[cell_id])
            hi = int(accel_cell_offsets[cell_id + 1])
            if hi > lo:
                chunks.append(accel_triangle_indices[lo:hi])
    if not chunks:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(chunks).astype(np.int32, copy=False)


def _nearest_triangle_candidate(
    vertices: np.ndarray,
    triangles: np.ndarray,
    accel_cell_offsets: np.ndarray,
    accel_triangle_indices: np.ndarray,
    nx: int,
    ny: int,
    ix: int,
    iy: int,
    point: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return the closest triangle and a clamped convex weighting.

    Used only when strict containment fails.  A regular grid keeps an
    out-of-domain trial finite by clamping its axis interpolation at the edge;
    this is the mesh equivalent, so a segment that crosses a wall still yields
    a finite trajectory for the hit localization that replaces it.  Weights are
    clamped to the simplex and renormalized, so the value stays inside the
    element's own range and never extrapolates.

    Candidates are ranked by the same distance-weighted barycentric margin the
    containment search uses, so the compiled and scalar paths agree on which
    element answers an outside point.
    """

    for ring in range(_OUTSIDE_MESH_RING_LIMIT + 1):
        candidates = _ring_triangle_indices(
            accel_cell_offsets,
            accel_triangle_indices,
            nx,
            ny,
            ix,
            iy,
            ring,
        )
        if candidates.size == 0:
            continue
        index, barycentric = _best_triangle_candidate(
            vertices,
            triangles,
            candidates,
            0,
            int(candidates.size),
            point,
            np.inf,
        )
        if index < 0:
            continue
        clamped = np.maximum(np.asarray(barycentric, dtype=np.float64), 0.0)
        total = float(np.sum(clamped))
        if total > 0.0:
            return int(index), clamped / total
    return -1, np.zeros(3, dtype=np.float64)


def _best_triangle_candidate(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_indices: np.ndarray,
    start: int,
    stop: int,
    point: np.ndarray,
    eps: float,
) -> tuple[int, np.ndarray]:
    best_index = -1
    best_barycentric = np.zeros(3, dtype=np.float64)
    best_margin = -np.inf
    for flat_index in range(start, stop):
        triangle_index = int(triangle_indices[flat_index])
        triangle = vertices[triangles[triangle_index]]
        a = triangle[0]
        b = triangle[1]
        c = triangle[2]
        v0 = b - a
        v1 = c - a
        v2 = point - a
        determinant = float(v0[0] * v1[1] - v0[1] * v1[0])
        determinant_scale = float(np.linalg.norm(v0) * np.linalg.norm(v1))
        if (
            determinant_scale <= 0.0
            or abs(determinant) <= 64.0 * np.finfo(np.float64).eps * determinant_scale
        ):
            continue
        beta = (v2[0] * v1[1] - v2[1] * v1[0]) / determinant
        gamma = (v0[0] * v2[1] - v0[1] * v2[0]) / determinant
        alpha = 1.0 - beta - gamma
        area2 = abs(determinant)
        h_alpha = area2 / float(np.linalg.norm(c - b))
        h_beta = area2 / float(np.linalg.norm(a - c))
        h_gamma = area2 / float(np.linalg.norm(b - a))
        if (
            alpha < -float(eps) / h_alpha
            or beta < -float(eps) / h_beta
            or gamma < -float(eps) / h_gamma
        ):
            continue
        margin = min(alpha * h_alpha, beta * h_beta, gamma * h_gamma)
        if margin > best_margin:
            best_margin = margin
            best_index = triangle_index
            best_barycentric[0] = alpha
            best_barycentric[1] = beta
            best_barycentric[2] = gamma
    return best_index, best_barycentric


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
    eps: float,
    nearest_fallback: bool = False,
) -> tuple[int, np.ndarray]:
    """Return the strongest containing triangle and its barycentric coordinates.

    ``nearest_fallback`` answers a different question from containment: when
    the point lies outside the mesh it returns the closest element with clamped
    weights, so a value query stays finite.  Support classification must leave
    it off, because an outside point is still outside.
    """

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.int32)
    point = np.asarray(position, dtype=np.float64)
    origin = np.asarray(accel_origin, dtype=np.float64)
    cell_size = np.asarray(accel_cell_size, dtype=np.float64)
    nx = int(accel_shape[0])
    ny = int(accel_shape[1])
    extent_max = origin + cell_size * np.asarray([nx, ny], dtype=np.float64)
    outside_bbox = (
        point[0] < float(origin[0]) - eps
        or point[1] < float(origin[1]) - eps
        or point[0] > float(extent_max[0]) + eps
        or point[1] > float(extent_max[1]) + eps
    )
    if outside_bbox and not nearest_fallback:
        return -1, np.zeros(3, dtype=np.float64)
    ix = int(
        np.clip(
            np.floor((point[0] - origin[0]) / cell_size[0]),
            0,
            nx - 1,
        )
    )
    iy = int(
        np.clip(
            np.floor((point[1] - origin[1]) / cell_size[1]),
            0,
            ny - 1,
        )
    )
    cell_id = ix * ny + iy
    cell_offsets = np.asarray(accel_cell_offsets, dtype=np.int32)
    triangle_indices = np.asarray(accel_triangle_indices, dtype=np.int32)
    start = int(cell_offsets[cell_id])
    stop = int(cell_offsets[cell_id + 1])
    if stop <= start:
        if nearest_fallback:
            return _nearest_triangle_candidate(
                verts,
                tris,
                cell_offsets,
                triangle_indices,
                nx,
                ny,
                ix,
                iy,
                point,
            )
        return -1, np.zeros(3, dtype=np.float64)
    best_index, barycentric = _best_triangle_candidate(
        verts,
        tris,
        triangle_indices,
        start,
        stop,
        point,
        eps,
    )
    if best_index < 0 and nearest_fallback:
        return _nearest_triangle_candidate(
            verts,
            tris,
            cell_offsets,
            triangle_indices,
            nx,
            ny,
            ix,
            iy,
            point,
        )
    return int(best_index), np.asarray(barycentric, dtype=np.float64)


def _locate_field_triangle(
    field,
    position: np.ndarray,
    *,
    nearest_fallback: bool = False,
) -> tuple[int, np.ndarray]:
    return locate_triangle_containing_point(
        vertices=field.mesh_vertices,
        triangles=field.mesh_triangles,
        accel_origin=field.accel_origin,
        accel_cell_size=field.accel_cell_size,
        accel_shape=field.accel_shape,
        accel_cell_offsets=field.accel_cell_offsets,
        accel_triangle_indices=field.accel_triangle_indices,
        position=np.asarray(position, dtype=np.float64),
        eps=field_triangle_support_tolerance(field),
        nearest_fallback=bool(nearest_fallback),
    )


def sample_triangle_mesh_status(field, position: np.ndarray) -> int:
    """Classify a point as supported or outside a triangle mesh."""

    triangle_index, _barycentric = _locate_field_triangle(field, position)
    if triangle_index >= 0:
        return int(VALID_MASK_STATUS_CLEAN)
    return int(VALID_MASK_STATUS_HARD_INVALID)


def _sample_vertices(
    barycentric: np.ndarray,
    vertex_values: np.ndarray,
) -> float:
    return float(
        np.dot(
            np.asarray(barycentric, dtype=np.float64),
            vertex_values,
        )
    )


def _sample_transient_series(
    data: np.ndarray,
    times: np.ndarray,
    triangle: np.ndarray,
    barycentric: np.ndarray,
    t_eval: float,
    mode: str,
) -> float:
    if float(t_eval) <= float(times[0]):
        return _sample_vertices(barycentric, data[0, triangle])
    if float(t_eval) >= float(times[-1]):
        return _sample_vertices(barycentric, data[-1, triangle])
    hi = int(np.searchsorted(times, float(t_eval)))
    lo = hi - 1
    if str(mode).strip().lower() == "nearest":
        if abs(float(times[hi]) - float(t_eval)) < abs(
            float(t_eval) - float(times[lo])
        ):
            return _sample_vertices(barycentric, data[hi, triangle])
        return _sample_vertices(barycentric, data[lo, triangle])
    t_lo = float(times[lo])
    t_hi = float(times[hi])
    interval = t_hi - t_lo
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("Triangle field times must be finite and strictly increasing")
    alpha_t = (float(t_eval) - t_lo) / interval
    value_lo = _sample_vertices(barycentric, data[lo, triangle])
    value_hi = _sample_vertices(barycentric, data[hi, triangle])
    return float(value_lo * (1.0 - alpha_t) + value_hi * alpha_t)


def sample_triangle_mesh_series(
    series,
    field,
    position: np.ndarray,
    t_eval: float,
    *,
    mode: str = "linear",
) -> float:
    """Sample a scalar vertex series at one mesh position and time."""

    triangle_index, barycentric = _locate_field_triangle(
        field,
        position,
        nearest_fallback=True,
    )
    if triangle_index < 0:
        return float("nan")
    triangle = np.asarray(field.mesh_triangles, dtype=np.int32)[triangle_index]
    data = np.asarray(series.data, dtype=np.float64)
    times = np.asarray(series.times, dtype=np.float64)
    if data.ndim == 1:
        return _sample_vertices(barycentric, data[triangle])
    if data.shape[0] <= 1 or times.size <= 1:
        return _sample_vertices(barycentric, data[0, triangle])
    return _sample_transient_series(
        data,
        times,
        triangle,
        barycentric,
        t_eval,
        mode,
    )
