from __future__ import annotations

from itertools import pairwise

import numpy as np
from numba import njit

from .boundary_numerics import (
    BOUNDARY_NUMERICS_POLICY_VERSION,
    scaled_classification_tolerance,
)


def _coordinate_identity_policy_2d(
    boundary_edges: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Resolve a scale-aware key for coordinate-only edge topology.

    COMSOL conversion itself retains integer mesh node IDs.  Runtime geometry
    artifacts contain coordinates only, so their loop-validation key is
    quantized using the smallest positive edge length and float64 coordinate
    ULPs instead of a fixed number of decimal places.
    """

    segs = np.asarray(boundary_edges, dtype=np.float64)
    if np.any(~np.isfinite(segs)):
        raise ValueError("boundary_edges must contain only finite coordinates")
    lengths = np.linalg.norm(segs[:, 1, :] - segs[:, 0, :], axis=1)
    positive = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if positive.size == 0:
        raise ValueError(
            "boundary_edges must contain at least one positive-length edge"
        )
    resolution = float(np.min(positive))
    coordinate_roundoff, tolerance = scaled_classification_tolerance(segs, resolution)
    origin = np.min(segs.reshape((-1, 2)), axis=0)
    return origin.astype(np.float64), resolution, coordinate_roundoff, tolerance


def _boundary_edge_graph_2d(
    boundary_edges: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    dict[int, list[int]],
    tuple[float, float, float] | None,
]:
    segs = np.asarray(boundary_edges, dtype=np.float64)
    if segs.ndim != 3 or segs.shape[1:] != (2, 2) or segs.shape[0] == 0:
        return segs, np.zeros((0, 2), dtype=np.int64), [], {}, None
    origin, resolution, coordinate_roundoff, tolerance = _coordinate_identity_policy_2d(
        segs
    )
    key_to_vertex: dict[tuple[int, int], int] = {}
    vertex_coords: list[np.ndarray] = []
    edge_vertices = np.zeros((segs.shape[0], 2), dtype=np.int64)
    for i, seg in enumerate(segs):
        for j in range(2):
            point = seg[j]
            scaled = (point - origin) / tolerance
            key = (int(np.rint(scaled[0])), int(np.rint(scaled[1])))
            vid = key_to_vertex.get(key)
            if vid is None:
                vid = len(vertex_coords)
                key_to_vertex[key] = vid
                vertex_coords.append(np.asarray(point, dtype=np.float64))
            edge_vertices[i, j] = int(vid)
    adjacency: dict[int, list[int]] = {}
    for edge_idx, (v0, v1) in enumerate(edge_vertices):
        adjacency.setdefault(int(v0), []).append(int(edge_idx))
        adjacency.setdefault(int(v1), []).append(int(edge_idx))
    return (
        segs,
        edge_vertices,
        vertex_coords,
        adjacency,
        (resolution, coordinate_roundoff, tolerance),
    )


def _validate_boundary_adjacency_2d(
    segs: np.ndarray,
    adjacency: dict[int, list[int]],
    identity: tuple[float, float, float] | None,
) -> dict[str, object]:
    degree_counts = {
        int(vertex_id): len(edges) for vertex_id, edges in adjacency.items()
    }
    branch_vertex_count = int(sum(1 for degree in degree_counts.values() if degree > 2))
    dangling_vertex_count = int(
        sum(1 for degree in degree_counts.values() if degree < 2)
    )
    if branch_vertex_count > 0 or dangling_vertex_count > 0:
        raise ValueError(
            "boundary_edges must form disjoint degree-2 loops in 2D; "
            f"found branch/dangling vertices ({_invalid_degree_preview(degree_counts)})"
        )
    report: dict[str, object] = {
        "edge_count": int(segs.shape[0]),
        "vertex_count": len(adjacency),
        "branch_vertex_count": int(branch_vertex_count),
        "dangling_vertex_count": int(dangling_vertex_count),
    }
    if identity is not None:
        report.update(_boundary_identity_report(identity))
    return report


def _invalid_degree_preview(degree_counts: dict[int, int]) -> str:
    invalid = (
        f"v{vertex_id}:degree={degree}"
        for vertex_id, degree in sorted(degree_counts.items())
        if degree != 2
    )
    return ", ".join(list(invalid)[:4])


def _boundary_identity_report(
    identity: tuple[float, float, float],
) -> dict[str, object]:
    resolution, coordinate_roundoff, tolerance = identity
    return {
        "identity_policy": BOUNDARY_NUMERICS_POLICY_VERSION,
        "identity_resolution_m": float(resolution),
        "identity_coordinate_roundoff_m": float(coordinate_roundoff),
        "identity_tolerance_m": float(tolerance),
    }


def validate_boundary_edges_2d(boundary_edges: np.ndarray) -> dict[str, object]:
    segs, _edge_vertices, _vertex_coords, adjacency, identity = _boundary_edge_graph_2d(
        boundary_edges
    )
    if segs.ndim != 3 or segs.shape[1:] != (2, 2) or segs.shape[0] == 0:
        return {
            "edge_count": 0,
            "vertex_count": 0,
            "branch_vertex_count": 0,
            "dangling_vertex_count": 0,
        }
    return _validate_boundary_adjacency_2d(segs, adjacency, identity)


def _next_unused_edge_2d(
    candidates: list[int],
    unused: np.ndarray,
) -> int | None:
    for candidate in candidates:
        if bool(unused[int(candidate)]):
            return int(candidate)
    return None


def _trace_boundary_loop_vertices_2d(
    start_edge: int,
    edge_vertices: np.ndarray,
    adjacency: dict[int, list[int]],
    unused: np.ndarray,
) -> list[int]:
    start_vertex = int(edge_vertices[int(start_edge), 0])
    current_vertex = int(edge_vertices[int(start_edge), 1])
    loop_vertices = [start_vertex, current_vertex]
    unused[int(start_edge)] = False
    traversed_edge_count = 0
    while current_vertex != start_vertex:
        next_edge = _next_unused_edge_2d(
            adjacency.get(current_vertex, []),
            unused,
        )
        if next_edge is None:
            break
        unused[next_edge] = False
        vertex_a = int(edge_vertices[next_edge, 0])
        vertex_b = int(edge_vertices[next_edge, 1])
        current_vertex = vertex_b if vertex_a == current_vertex else vertex_a
        loop_vertices.append(current_vertex)
        traversed_edge_count += 1
        if traversed_edge_count > edge_vertices.shape[0] + 1:
            break
    return loop_vertices


def _normalized_loop_coordinates_2d(
    loop_vertices: list[int],
    vertex_coords: list[np.ndarray],
) -> np.ndarray | None:
    if loop_vertices[-1] != loop_vertices[0] or len(loop_vertices) < 4:
        return None
    coordinates = np.asarray(
        [vertex_coords[index] for index in loop_vertices[:-1]],
        dtype=np.float64,
    )
    if _polygon_signed_area(coordinates) < 0.0:
        return coordinates[::-1].copy()
    return coordinates


def build_boundary_loops_2d(boundary_edges: np.ndarray) -> tuple[np.ndarray, ...]:
    segs, edge_vertices, vertex_coords, adjacency, identity = _boundary_edge_graph_2d(
        boundary_edges
    )
    if segs.ndim != 3 or segs.shape[1:] != (2, 2) or segs.shape[0] == 0:
        return ()
    _validate_boundary_adjacency_2d(segs, adjacency, identity)
    unused = np.ones(segs.shape[0], dtype=bool)
    loops: list[np.ndarray] = []
    for start_edge in range(segs.shape[0]):
        if not bool(unused[start_edge]):
            continue
        loop_vertices = _trace_boundary_loop_vertices_2d(
            start_edge,
            edge_vertices,
            adjacency,
            unused,
        )
        coordinates = _normalized_loop_coordinates_2d(
            loop_vertices,
            vertex_coords,
        )
        if coordinates is not None:
            loops.append(coordinates)
    return tuple(loops)


def points_inside_boundary_loops_2d_with_boundary(
    points: np.ndarray,
    loops: tuple[np.ndarray, ...],
    on_edge_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
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
        valid_edge = edge_len2 > 0.0
        if tol > 0.0 and np.any(valid_edge):
            px = x[:, None] - x0[None, :]
            py = y[:, None] - y0[None, :]
            t = np.zeros((pts.shape[0], poly.shape[0]), dtype=np.float64)
            t[:, valid_edge] = (
                px[:, valid_edge] * dx[None, valid_edge]
                + py[:, valid_edge] * dy[None, valid_edge]
            ) / edge_len2[None, valid_edge]
            t = np.clip(t, 0.0, 1.0)
            proj_x = x0[None, :] + t * dx[None, :]
            proj_y = y0[None, :] + t * dy[None, :]
            dist2 = (x[:, None] - proj_x) ** 2 + (y[:, None] - proj_y) ** 2
            on_seg = valid_edge[None, :] & (dist2 <= tol2)
            on_boundary |= np.any(on_seg, axis=1)
        denom = y1 - y0
        denom_safe = np.where(denom == 0.0, 1.0, denom)
        cond = (y0[None, :] > y[:, None]) != (y1[None, :] > y[:, None])
        x_cross = (
            x0[None, :]
            + (y[:, None] - y0[None, :])
            * (x1[None, :] - x0[None, :])
            / denom_safe[None, :]
        )
        crossings = cond & (x_cross > x[:, None])
        interior ^= np.sum(crossings, axis=1) % 2 == 1
    inside = interior | on_boundary
    return inside, on_boundary


@njit(cache=True)
def _point_on_edge_2d(
    x: float,
    y: float,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    edge_len2: float,
    tolerance2: float,
) -> bool:
    t = ((x - x0) * dx + (y - y0) * dy) / edge_len2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    dist2 = (x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y)
    return bool(dist2 <= tolerance2)


@njit(cache=True)
def _edge_crosses_positive_ray_2d(
    x: float,
    y: float,
    x0: float,
    y0: float,
    y1: float,
    dx: float,
    dy: float,
) -> bool:
    if (y0 > y) == (y1 > y):
        return False
    x_cross = x0 + (y - y0) * dx / dy
    return bool(x_cross > x)


@njit(cache=True)
def _points_inside_boundary_edges_2d_with_boundary_kernel(
    points: np.ndarray,
    edges: np.ndarray,
    on_edge_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
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
            if edge_len2 <= 0.0:
                continue
            if tol > 0.0 and _point_on_edge_2d(x, y, x0, y0, dx, dy, edge_len2, tol2):
                boundary = True
            if _edge_crosses_positive_ray_2d(x, y, x0, y0, y1, dx, dy):
                interior = not interior
        inside[i] = interior or boundary
        on_boundary[i] = boundary
    return inside, on_boundary


def points_inside_boundary_edges_2d_with_boundary(
    points: np.ndarray,
    boundary_edges: np.ndarray,
    on_edge_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    edges = np.asarray(boundary_edges, dtype=np.float64)
    if (
        pts.ndim != 2
        or pts.shape[1] != 2
        or edges.ndim != 3
        or edges.shape[1:] != (2, 2)
        or edges.shape[0] == 0
    ):
        empty = np.zeros(pts.shape[0], dtype=bool)
        return empty, empty.copy()
    return _points_inside_boundary_edges_2d_with_boundary_kernel(
        pts, edges, float(on_edge_tol)
    )


def point_inside_boundary_edges_2d_with_boundary(
    point: np.ndarray,
    boundary_edges: np.ndarray,
    on_edge_tol: float,
) -> tuple[bool, bool]:
    p = np.asarray(point, dtype=np.float64)
    edges = np.asarray(boundary_edges, dtype=np.float64)
    if (
        p.ndim != 1
        or p.size != 2
        or edges.ndim != 3
        or edges.shape[1:] != (2, 2)
        or edges.shape[0] == 0
    ):
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
    valid_edge = edge_len2 > 0.0

    on_boundary = False
    if tol > 0.0 and np.any(valid_edge):
        t = np.zeros(edges.shape[0], dtype=np.float64)
        t[valid_edge] = (
            (x - x0[valid_edge]) * dx[valid_edge]
            + (y - y0[valid_edge]) * dy[valid_edge]
        ) / edge_len2[valid_edge]
        t = np.clip(t, 0.0, 1.0)
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        dist2 = (x - proj_x) ** 2 + (y - proj_y) ** 2
        on_boundary = bool(np.any(valid_edge & (dist2 <= tol * tol)))

    denom = y1 - y0
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    cond = (y0 > y) != (y1 > y)
    x_cross = x0 + (y - y0) * dx / denom_safe
    interior = bool((int(np.count_nonzero(cond & (x_cross > x))) % 2) == 1)
    return bool(interior or on_boundary), bool(on_boundary)


def points_inside_boundary_loops_2d(
    points: np.ndarray,
    loops: tuple[np.ndarray, ...],
    on_edge_tol: float,
) -> np.ndarray:
    inside, _ = points_inside_boundary_loops_2d_with_boundary(
        points, loops, on_edge_tol=on_edge_tol
    )
    return inside


def encode_boundary_loops_2d(
    loops: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if len(loops) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.asarray([0], dtype=np.int32)
    flat: list[np.ndarray] = []
    offsets = [0]
    for loop in loops:
        arr = np.asarray(loop, dtype=np.float64)
        flat.append(arr)
        offsets.append(offsets[-1] + int(arr.shape[0]))
    return np.vstack(flat).astype(np.float64), np.asarray(offsets, dtype=np.int32)


def decode_boundary_loops_2d(
    flat: np.ndarray | None, offsets: np.ndarray | None
) -> tuple[np.ndarray, ...]:
    if flat is None or offsets is None:
        return ()
    pts = np.asarray(flat, dtype=np.float64)
    idx = np.asarray(offsets, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2 or idx.ndim != 1 or idx.size < 2:
        return ()
    loops = []
    for start, end in pairwise(idx):
        if int(end) > int(start):
            loops.append(pts[int(start) : int(end)].copy())
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
