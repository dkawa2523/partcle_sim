"""Float64 triangle resolution, barycentric, and closed-surface topology."""

from __future__ import annotations

import math

import numpy as np

from .boundary_numerics import (
    BOUNDARY_NUMERICS_POLICY_VERSION,
    scaled_classification_tolerance,
)

_TRIANGLE_RESOLUTION_ULPS = 64.0


def _stable_norm(vector: np.ndarray) -> float:
    """Return a Euclidean norm without a physical-unit underflow floor."""

    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("nan")
    return float(math.hypot(*(float(value) for value in values)))


def unresolved_triangle_indices(triangles: np.ndarray) -> np.ndarray:
    """Return rows whose edge Gram determinant is unresolved in float64.

    Coordinates are normalized by each triangle's own edge scale before the
    Gram determinant is evaluated. The predicate is therefore unchanged by a
    similarity scaling and applies to indexed 2D and 3D boundary triangles.
    """

    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1] != 3 or tri.shape[2] not in (2, 3):
        raise ValueError(f"triangles must have shape (n, 3, 2|3), got {tri.shape}")
    if tri.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    edge_1 = tri[:, 1, :] - tri[:, 0, :]
    edge_2 = tri[:, 2, :] - tri[:, 0, :]
    edge_scale = np.maximum(
        np.max(np.abs(edge_1), axis=1),
        np.max(np.abs(edge_2), axis=1),
    )
    finite = (
        np.all(np.isfinite(tri), axis=(1, 2))
        & np.isfinite(edge_scale)
        & (edge_scale > 0.0)
    )
    safe_scale = np.where(finite, edge_scale, 1.0)
    normalized_1 = edge_1 / safe_scale[:, None]
    normalized_2 = edge_2 / safe_scale[:, None]
    gram_scale = np.einsum("ij,ij->i", normalized_1, normalized_1) * np.einsum(
        "ij,ij->i", normalized_2, normalized_2
    )
    if tri.shape[2] == 2:
        cross_measure = (
            normalized_1[:, 0] * normalized_2[:, 1]
            - normalized_1[:, 1] * normalized_2[:, 0]
        )
        gram_determinant = cross_measure * cross_measure
    else:
        cross_measure = np.cross(normalized_1, normalized_2)
        gram_determinant = np.einsum("ij,ij->i", cross_measure, cross_measure)
    relative_limit = (
        _TRIANGLE_RESOLUTION_ULPS * np.finfo(np.float64).eps
    ) ** 2 * gram_scale
    resolved = (
        finite
        & np.isfinite(gram_scale)
        & np.isfinite(gram_determinant)
        & (gram_scale > 0.0)
        & (gram_determinant > relative_limit)
    )
    return np.flatnonzero(~resolved).astype(np.int64)


def _coordinate_topology_inputs(
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    edges = np.concatenate(
        (
            triangles[:, 1, :] - triangles[:, 0, :],
            triangles[:, 2, :] - triangles[:, 1, :],
            triangles[:, 0, :] - triangles[:, 2, :],
        ),
        axis=0,
    )
    edge_lengths = np.asarray(
        [_stable_norm(edge) for edge in edges],
        dtype=np.float64,
    )
    positive = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
    if positive.size == 0:
        raise ValueError("boundary_triangles must contain positive-length edges")
    resolution = float(np.min(positive))
    coordinate_roundoff, tolerance = scaled_classification_tolerance(
        triangles,
        resolution,
    )
    vertices = triangles.reshape((-1, 3))
    origin = np.min(vertices, axis=0)
    return vertices, origin, resolution, coordinate_roundoff, tolerance


def _topology_bin_candidates(
    key: tuple[int, int, int],
    bins: dict[tuple[int, int, int], list[int]],
    neighbor_offsets: tuple[tuple[int, int, int], ...],
) -> list[int]:
    candidates: list[int] = []
    for offset in neighbor_offsets:
        candidates.extend(
            bins.get(
                (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2]),
                (),
            )
        )
    return candidates


def _matching_topology_vertex_id(
    point: np.ndarray,
    candidates: list[int],
    representatives: list[np.ndarray],
    tolerance: float,
) -> int:
    vertex_id = -1
    best_distance = float("inf")
    for candidate in candidates:
        distance = float(np.max(np.abs(point - representatives[candidate])))
        if distance <= tolerance and distance < best_distance:
            vertex_id = int(candidate)
            best_distance = float(distance)
    return vertex_id


def _coordinate_topology_ids(
    triangles: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Recover coordinate-only surface topology with a scale/ULP policy."""

    tri = np.asarray(triangles, dtype=np.float64)
    vertices, origin, resolution, coordinate_roundoff, tolerance = (
        _coordinate_topology_inputs(tri)
    )

    bins: dict[tuple[int, int, int], list[int]] = {}
    representatives: list[np.ndarray] = []
    ids = np.empty(vertices.shape[0], dtype=np.int64)
    neighbor_offsets = tuple(
        (dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    )
    for row_index, point in enumerate(vertices):
        scaled = (point - origin) / tolerance
        if np.any(~np.isfinite(scaled)):
            raise ValueError(
                "boundary triangle coordinates cannot be quantized reliably"
            )
        key = (
            int(np.floor(scaled[0])),
            int(np.floor(scaled[1])),
            int(np.floor(scaled[2])),
        )
        candidates = _topology_bin_candidates(key, bins, neighbor_offsets)
        vertex_id = _matching_topology_vertex_id(
            point,
            candidates,
            representatives,
            tolerance,
        )
        if vertex_id < 0:
            vertex_id = len(representatives)
            representatives.append(np.asarray(point, dtype=np.float64))
            bins.setdefault(key, []).append(vertex_id)
        ids[row_index] = int(vertex_id)
    return ids.reshape(tri.shape[0], 3), resolution, coordinate_roundoff, tolerance


def point_triangle_barycentric(
    point: np.ndarray,
    triangle: np.ndarray,
) -> np.ndarray | None:
    """Return barycentric coordinates, or ``None`` for a degenerate triangle."""

    p = np.asarray(point, dtype=np.float64)
    tri = np.asarray(triangle, dtype=np.float64)
    if tri.shape != (3, 3) or p.shape != (3,) or np.any(~np.isfinite(p)):
        return None
    if unresolved_triangle_indices(tri.reshape((1, 3, 3))).size:
        return None
    a = tri[0]
    v0_raw = tri[1] - a
    v1_raw = tri[2] - a
    edge_scale = float(max(np.max(np.abs(v0_raw)), np.max(np.abs(v1_raw))))
    v0 = v0_raw / edge_scale
    v1 = v1_raw / edge_scale
    v2 = (p - a) / edge_scale
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    normal = np.cross(v0, v1)
    denominator = float(np.dot(normal, normal))
    determinant_scale = d00 * d11
    if (
        determinant_scale <= 0.0
        or denominator
        <= (_TRIANGLE_RESOLUTION_ULPS * np.finfo(np.float64).eps) ** 2
        * determinant_scale
    ):
        return None
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.asarray([1.0 - v - w, v, w], dtype=np.float64)


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = _stable_norm(arr)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("surface normal must be finite and non-zero")
    return arr / norm


def _triangle_normals(triangles: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangles, dtype=np.float64)
    unresolved = unresolved_triangle_indices(tri)
    if unresolved.size:
        raise ValueError(
            "boundary_triangles contains float64-unresolved triangle rows "
            f"{unresolved[:12].tolist()}"
        )
    edge_1 = tri[:, 1, :] - tri[:, 0, :]
    edge_2 = tri[:, 2, :] - tri[:, 0, :]
    edge_scale = np.maximum(
        np.max(np.abs(edge_1), axis=1), np.max(np.abs(edge_2), axis=1)
    )
    normals = np.cross(edge_1 / edge_scale[:, None], edge_2 / edge_scale[:, None])
    magnitude = np.linalg.norm(normals, axis=1)
    return normals / magnitude[:, None]


def _validated_closed_surface_triangles(triangles: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(
            f"boundary_triangles must be shaped as (n, 3, 3), got {tri.shape}"
        )
    if tri.shape[0] == 0:
        raise ValueError("boundary_triangles must be non-empty")
    unresolved = unresolved_triangle_indices(tri)
    if unresolved.size:
        raise ValueError(
            "boundary_triangles contains float64-unresolved triangle rows "
            f"{unresolved[:12].tolist()}"
        )
    return tri


def _surface_edge_counts(
    triangle_vertex_ids: np.ndarray,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    undirected_counts: dict[tuple[int, int], int] = {}
    oriented_counts: dict[tuple[int, int], int] = {}
    for a, b, c in triangle_vertex_ids:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            undirected_counts[key] = undirected_counts.get(key, 0) + 1
            oriented_counts[(u, v)] = oriented_counts.get((u, v), 0) + 1
    return undirected_counts, oriented_counts


def _surface_orientation_errors(
    undirected_counts: dict[tuple[int, int], int],
    oriented_counts: dict[tuple[int, int], int],
) -> list[tuple[int, int, int, int]]:
    errors: list[tuple[int, int, int, int]] = []
    for u, v in undirected_counts:
        forward = int(oriented_counts.get((u, v), 0))
        backward = int(oriented_counts.get((v, u), 0))
        if forward != 1 or backward != 1:
            errors.append((u, v, forward, backward))
    return errors


def validate_closed_surface_triangles(triangles: np.ndarray) -> dict[str, object]:
    tri = _validated_closed_surface_triangles(triangles)
    tri_ids, resolution, coordinate_roundoff, identity_tolerance = (
        _coordinate_topology_ids(tri)
    )
    undirected_counts, oriented_counts = _surface_edge_counts(tri_ids)

    bad_cardinality = [key for key, count in undirected_counts.items() if count != 2]
    if bad_cardinality:
        raise ValueError(
            "boundary_triangles must form a closed 2-manifold: "
            f"{len(bad_cardinality)} edge(s) do not have exactly two adjacent triangles"
        )

    bad_orientation = _surface_orientation_errors(
        undirected_counts,
        oriented_counts,
    )
    if bad_orientation:
        raise ValueError(
            "boundary_triangles orientation mismatch detected: "
            f"{len(bad_orientation)} edge(s) are not oppositely oriented "
            "across adjacent triangles"
        )

    return {
        "triangle_count": int(tri.shape[0]),
        "unique_vertex_count": int(np.unique(tri_ids).size),
        "edge_count": len(undirected_counts),
        "identity_policy": BOUNDARY_NUMERICS_POLICY_VERSION,
        "identity_resolution_m": float(resolution),
        "identity_coordinate_roundoff_m": float(coordinate_roundoff),
        "identity_tolerance_m": float(identity_tolerance),
    }
