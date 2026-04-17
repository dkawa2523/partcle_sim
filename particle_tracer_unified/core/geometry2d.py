from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numba import njit


def _boundary_edge_graph_2d(
    boundary_edges: np.ndarray,
    round_decimals: int = 12,
) -> Tuple[np.ndarray, np.ndarray, list[np.ndarray], Dict[int, list[int]]]:
    segs = np.asarray(boundary_edges, dtype=np.float64)
    if segs.ndim != 3 or segs.shape[1:] != (2, 2) or segs.shape[0] == 0:
        return segs, np.zeros((0, 2), dtype=np.int64), [], {}
    key_to_vertex: Dict[Tuple[float, float], int] = {}
    vertex_coords: list[np.ndarray] = []
    edge_vertices = np.zeros((segs.shape[0], 2), dtype=np.int64)
    for i, seg in enumerate(segs):
        for j in range(2):
            point = seg[j]
            key = tuple(np.round(point, round_decimals))
            vid = key_to_vertex.get(key)
            if vid is None:
                vid = len(vertex_coords)
                key_to_vertex[key] = vid
                vertex_coords.append(np.asarray(point, dtype=np.float64))
            edge_vertices[i, j] = int(vid)
    adjacency: Dict[int, list[int]] = {}
    for edge_idx, (v0, v1) in enumerate(edge_vertices):
        adjacency.setdefault(int(v0), []).append(int(edge_idx))
        adjacency.setdefault(int(v1), []).append(int(edge_idx))
    return segs, edge_vertices, vertex_coords, adjacency


def validate_boundary_edges_2d(boundary_edges: np.ndarray, round_decimals: int = 12) -> Dict[str, int]:
    segs, _edge_vertices, _vertex_coords, adjacency = _boundary_edge_graph_2d(boundary_edges, round_decimals=round_decimals)
    if segs.ndim != 3 or segs.shape[1:] != (2, 2) or segs.shape[0] == 0:
        return {
            'edge_count': int(0),
            'vertex_count': int(0),
            'branch_vertex_count': int(0),
            'dangling_vertex_count': int(0),
        }
    degree_counts = {int(vertex_id): int(len(edges)) for vertex_id, edges in adjacency.items()}
    branch_vertex_count = int(sum(1 for degree in degree_counts.values() if degree > 2))
    dangling_vertex_count = int(sum(1 for degree in degree_counts.values() if degree < 2))
    if branch_vertex_count > 0 or dangling_vertex_count > 0:
        bad_preview = []
        for vertex_id, degree in sorted(degree_counts.items()):
            if degree != 2:
                bad_preview.append(f'v{vertex_id}:degree={degree}')
            if len(bad_preview) >= 4:
                break
        preview = ', '.join(bad_preview)
        raise ValueError(
            'boundary_edges must form disjoint degree-2 loops in 2D; '
            f'found branch/dangling vertices ({preview})'
        )
    return {
        'edge_count': int(segs.shape[0]),
        'vertex_count': int(len(adjacency)),
        'branch_vertex_count': int(branch_vertex_count),
        'dangling_vertex_count': int(dangling_vertex_count),
    }


def build_boundary_loops_2d(boundary_edges: np.ndarray, round_decimals: int = 12) -> Tuple[np.ndarray, ...]:
    segs, edge_vertices, vertex_coords, adjacency = _boundary_edge_graph_2d(boundary_edges, round_decimals=round_decimals)
    if segs.ndim != 3 or segs.shape[1:] != (2, 2) or segs.shape[0] == 0:
        return tuple()
    validate_boundary_edges_2d(segs, round_decimals=round_decimals)
    unused = np.ones(segs.shape[0], dtype=bool)
    loops: list[np.ndarray] = []
    for start_edge in range(segs.shape[0]):
        if not unused[start_edge]:
            continue
        v_start, v_next = (int(edge_vertices[start_edge, 0]), int(edge_vertices[start_edge, 1]))
        current_edge = int(start_edge)
        current_vertex = int(v_next)
        loop_vertices = [v_start, v_next]
        unused[current_edge] = False
        guard = 0
        while current_vertex != v_start:
            candidates = adjacency.get(current_vertex, [])
            next_edge = None
            for cand in candidates:
                if unused[cand]:
                    next_edge = int(cand)
                    break
            if next_edge is None:
                break
            unused[next_edge] = False
            a, b = (int(edge_vertices[next_edge, 0]), int(edge_vertices[next_edge, 1]))
            current_vertex = b if a == current_vertex else a
            loop_vertices.append(current_vertex)
            current_edge = next_edge
            guard += 1
            if guard > segs.shape[0] + 1:
                break
        if loop_vertices[-1] == v_start and len(loop_vertices) >= 4:
            coords = np.asarray([vertex_coords[idx] for idx in loop_vertices[:-1]], dtype=np.float64)
            if _polygon_signed_area(coords) < 0.0:
                coords = coords[::-1].copy()
            loops.append(coords)
    return tuple(loops)


def points_inside_boundary_loops_2d_with_boundary(
    points: np.ndarray,
    loops: Tuple[np.ndarray, ...],
    on_edge_tol: float = 1.0e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(loops) == 0:
        empty = np.zeros(pts.shape[0], dtype=bool)
        return empty, empty.copy()
    tol = float(on_edge_tol) if np.isfinite(on_edge_tol) else 0.0
    tol = max(0.0, tol)
    tol2 = tol * tol

    interior = np.zeros(pts.shape[0], dtype=bool)
    on_boundary = np.zeros(pts.shape[0], dtype=bool)
    x = pts[:, 0]
    y = pts[:, 1]
    for loop in loops:
        poly = np.asarray(loop, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue
        x0 = poly[:, 0]
        y0 = poly[:, 1]
        x1 = np.roll(x0, -1)
        y1 = np.roll(y0, -1)
        dx = x1 - x0
        dy = y1 - y0
        edge_len2 = dx * dx + dy * dy
        valid_edge = edge_len2 > 1e-30
        if tol > 0.0 and np.any(valid_edge):
            px = x[:, None] - x0[None, :]
            py = y[:, None] - y0[None, :]
            t = np.zeros((pts.shape[0], poly.shape[0]), dtype=np.float64)
            t[:, valid_edge] = (
                px[:, valid_edge] * dx[None, valid_edge] + py[:, valid_edge] * dy[None, valid_edge]
            ) / edge_len2[None, valid_edge]
            t = np.clip(t, 0.0, 1.0)
            proj_x = x0[None, :] + t * dx[None, :]
            proj_y = y0[None, :] + t * dy[None, :]
            dist2 = (x[:, None] - proj_x) ** 2 + (y[:, None] - proj_y) ** 2
            on_seg = valid_edge[None, :] & (dist2 <= tol2)
            on_boundary |= np.any(on_seg, axis=1)
        denom = y1 - y0
        denom_safe = np.where(np.abs(denom) <= 1e-30, 1.0, denom)
        cond = ((y0[None, :] > y[:, None]) != (y1[None, :] > y[:, None]))
        x_cross = x0[None, :] + (y[:, None] - y0[None, :]) * (x1[None, :] - x0[None, :]) / denom_safe[None, :]
        crossings = cond & (x_cross > x[:, None])
        interior ^= (np.sum(crossings, axis=1) % 2 == 1)
    inside = interior | on_boundary
    return inside, on_boundary


@njit(cache=True)
def _points_inside_boundary_edges_2d_with_boundary_kernel(
    points: np.ndarray,
    edges: np.ndarray,
    on_edge_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_points = points.shape[0]
    n_edges = edges.shape[0]
    inside = np.zeros(n_points, dtype=np.bool_)
    on_boundary = np.zeros(n_points, dtype=np.bool_)
    tol = on_edge_tol if np.isfinite(on_edge_tol) and on_edge_tol > 0.0 else 0.0
    tol2 = tol * tol
    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        interior = False
        boundary = False
        for edge_idx in range(n_edges):
            x0 = edges[edge_idx, 0, 0]
            y0 = edges[edge_idx, 0, 1]
            x1 = edges[edge_idx, 1, 0]
            y1 = edges[edge_idx, 1, 1]
            dx = x1 - x0
            dy = y1 - y0
            edge_len2 = dx * dx + dy * dy
            if edge_len2 <= 1.0e-30:
                continue
            if tol > 0.0:
                t = ((x - x0) * dx + (y - y0) * dy) / edge_len2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                proj_x = x0 + t * dx
                proj_y = y0 + t * dy
                dist2 = (x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y)
                if dist2 <= tol2:
                    boundary = True
            crosses = (y0 > y) != (y1 > y)
            if crosses:
                denom = y1 - y0
                if abs(denom) > 1.0e-30:
                    x_cross = x0 + (y - y0) * dx / denom
                    if x_cross > x:
                        interior = not interior
        inside[i] = interior or boundary
        on_boundary[i] = boundary
    return inside, on_boundary


def points_inside_boundary_edges_2d_with_boundary(
    points: np.ndarray,
    boundary_edges: np.ndarray,
    on_edge_tol: float = 1.0e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    edges = np.asarray(boundary_edges, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or edges.ndim != 3 or edges.shape[1:] != (2, 2) or edges.shape[0] == 0:
        empty = np.zeros(pts.shape[0], dtype=bool)
        return empty, empty.copy()
    return _points_inside_boundary_edges_2d_with_boundary_kernel(pts, edges, float(on_edge_tol))


def point_inside_boundary_edges_2d_with_boundary(
    point: np.ndarray,
    boundary_edges: np.ndarray,
    on_edge_tol: float = 1.0e-9,
) -> Tuple[bool, bool]:
    p = np.asarray(point, dtype=np.float64)
    edges = np.asarray(boundary_edges, dtype=np.float64)
    if p.ndim != 1 or p.size != 2 or edges.ndim != 3 or edges.shape[1:] != (2, 2) or edges.shape[0] == 0:
        return False, False
    tol = float(on_edge_tol) if np.isfinite(on_edge_tol) else 0.0
    tol = max(0.0, tol)
    x = float(p[0])
    y = float(p[1])
    x0 = edges[:, 0, 0]
    y0 = edges[:, 0, 1]
    x1 = edges[:, 1, 0]
    y1 = edges[:, 1, 1]
    dx = x1 - x0
    dy = y1 - y0
    edge_len2 = dx * dx + dy * dy
    valid_edge = edge_len2 > 1.0e-30

    on_boundary = False
    if tol > 0.0 and np.any(valid_edge):
        t = np.zeros(edges.shape[0], dtype=np.float64)
        t[valid_edge] = ((x - x0[valid_edge]) * dx[valid_edge] + (y - y0[valid_edge]) * dy[valid_edge]) / edge_len2[valid_edge]
        t = np.clip(t, 0.0, 1.0)
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        dist2 = (x - proj_x) ** 2 + (y - proj_y) ** 2
        on_boundary = bool(np.any(valid_edge & (dist2 <= tol * tol)))

    denom = y1 - y0
    denom_safe = np.where(np.abs(denom) <= 1.0e-30, 1.0, denom)
    cond = (y0 > y) != (y1 > y)
    x_cross = x0 + (y - y0) * dx / denom_safe
    interior = bool((int(np.count_nonzero(cond & (x_cross > x))) % 2) == 1)
    return bool(interior or on_boundary), bool(on_boundary)


def points_inside_boundary_loops_2d(
    points: np.ndarray,
    loops: Tuple[np.ndarray, ...],
    on_edge_tol: float = 1.0e-9,
) -> np.ndarray:
    inside, _ = points_inside_boundary_loops_2d_with_boundary(points, loops, on_edge_tol=on_edge_tol)
    return inside


def point_inside_boundary_loops_2d(
    point: np.ndarray,
    loops: Tuple[np.ndarray, ...],
    on_edge_tol: float = 1.0e-9,
) -> bool:
    pts = np.asarray(point, dtype=np.float64).reshape(1, 2)
    return bool(points_inside_boundary_loops_2d(pts, loops, on_edge_tol=on_edge_tol)[0])


def encode_boundary_loops_2d(loops: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:
    if len(loops) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.asarray([0], dtype=np.int32)
    flat: list[np.ndarray] = []
    offsets = [0]
    for loop in loops:
        arr = np.asarray(loop, dtype=np.float64)
        flat.append(arr)
        offsets.append(offsets[-1] + int(arr.shape[0]))
    return np.vstack(flat).astype(np.float64), np.asarray(offsets, dtype=np.int32)


def decode_boundary_loops_2d(flat: np.ndarray | None, offsets: np.ndarray | None) -> Tuple[np.ndarray, ...]:
    if flat is None or offsets is None:
        return tuple()
    pts = np.asarray(flat, dtype=np.float64)
    idx = np.asarray(offsets, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2 or idx.ndim != 1 or idx.size < 2:
        return tuple()
    loops = []
    for start, end in zip(idx[:-1], idx[1:]):
        if int(end) > int(start):
            loops.append(pts[int(start):int(end)].copy())
    return tuple(loops)


def _polygon_signed_area(poly: np.ndarray) -> float:
    pts = np.asarray(poly, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 0.0
    x0 = pts[:, 0]
    y0 = pts[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    return 0.5 * float(np.sum(x0 * y1 - x1 * y0))
