"""Intersection, projection, and inside queries for triangle surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._triangle_surface import (
    TriangleSurface3D,
    query_triangle_candidates,
)
from ._triangle_topology import _normalize, _stable_norm


def _segment_aabb_overlaps(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> bool:
    return bool(np.all(a_max >= b_min) and np.all(b_max >= a_min))


def _segment_triangle_intersection_alpha(
    p0: np.ndarray,
    p1: np.ndarray,
    tri: np.ndarray,
    *,
    eps: float = 128.0 * np.finfo(np.float64).eps,
) -> float | None:
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    v0 = np.asarray(tri[0], dtype=np.float64)
    v1 = np.asarray(tri[1], dtype=np.float64)
    v2 = np.asarray(tri[2], dtype=np.float64)
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    dz = float(b[2] - a[2])
    e1x = float(v1[0] - v0[0])
    e1y = float(v1[1] - v0[1])
    e1z = float(v1[2] - v0[2])
    e2x = float(v2[0] - v0[0])
    e2y = float(v2[1] - v0[1])
    e2z = float(v2[2] - v0[2])
    motion_scale = max(abs(dx), abs(dy), abs(dz))
    edge_1_scale = max(abs(e1x), abs(e1y), abs(e1z))
    edge_2_scale = max(abs(e2x), abs(e2y), abs(e2z))
    determinant_scale = float(motion_scale * edge_1_scale * edge_2_scale)
    if not np.isfinite(determinant_scale) or determinant_scale <= 0.0:
        return None
    pvec_x = dy * e2z - dz * e2y
    pvec_y = dz * e2x - dx * e2z
    pvec_z = dx * e2y - dy * e2x
    determinant = e1x * pvec_x + e1y * pvec_y + e1z * pvec_z
    if abs(determinant) <= float(eps) * determinant_scale:
        return None
    inverse_determinant = 1.0 / determinant
    tx = float(a[0] - v0[0])
    ty = float(a[1] - v0[1])
    tz = float(a[2] - v0[2])
    u = float((tx * pvec_x + ty * pvec_y + tz * pvec_z) * inverse_determinant)
    if u < -eps or u > 1.0 + eps:
        return None
    qvec_x = ty * e1z - tz * e1y
    qvec_y = tz * e1x - tx * e1z
    qvec_z = tx * e1y - ty * e1x
    v = float((dx * qvec_x + dy * qvec_y + dz * qvec_z) * inverse_determinant)
    if v < -eps or (u + v) > 1.0 + eps:
        return None
    alpha = float((e2x * qvec_x + e2y * qvec_y + e2z * qvec_z) * inverse_determinant)
    if alpha < -eps or alpha > 1.0 + eps:
        return None
    return float(min(max(alpha, 0.0), 1.0))


def _nearest_segment_triangle_hit(
    surface: TriangleSurface3D,
    start: np.ndarray,
    end: np.ndarray,
    segment_min: np.ndarray,
    segment_max: np.ndarray,
    alpha_min: float,
    ignored_part_ids: frozenset[int],
) -> tuple[float, int] | None:
    candidate_ids = query_triangle_candidates(surface.grid, start, end)
    best_alpha = 2.0
    best_index = -1
    for candidate_id in candidate_ids:
        index = int(candidate_id)
        part_id = int(surface.part_ids[index]) if index < surface.part_ids.size else 0
        if part_id in ignored_part_ids:
            continue
        if not _segment_aabb_overlaps(
            segment_min,
            segment_max,
            surface.grid.triangle_mins[index],
            surface.grid.triangle_maxs[index],
        ):
            continue
        alpha = _segment_triangle_intersection_alpha(
            start,
            end,
            surface.triangles[index],
        )
        if alpha is None or alpha < alpha_min:
            continue
        if alpha < best_alpha:
            best_alpha = float(alpha)
            best_index = index
    if best_index < 0:
        return None
    return best_alpha, best_index


def segment_hit_from_surface_ignoring_parts(
    surface: TriangleSurface3D,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    alpha_min: float = 1.0e-8,
    coordinate_tolerance_m: float = 0.0,
    ignored_part_ids: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray, float, int, int] | None:
    start = np.asarray(p0, dtype=np.float64)
    end = np.asarray(p1, dtype=np.float64)
    segment = end - start
    segment_length = _stable_norm(segment)
    if not np.isfinite(segment_length) or segment_length <= 0.0:
        return None

    padding = max(float(coordinate_tolerance_m), 0.0)
    segment_min = np.minimum(start, end) - padding
    segment_max = np.maximum(start, end) + padding
    nearest = _nearest_segment_triangle_hit(
        surface,
        start,
        end,
        segment_min,
        segment_max,
        float(alpha_min),
        frozenset(int(value) for value in ignored_part_ids),
    )
    if nearest is None:
        return None
    best_alpha, best_index = nearest

    hit = start + best_alpha * segment
    normal = np.asarray(surface.normals[best_index], dtype=np.float64)
    if float(np.dot(start - hit, normal)) > 0.0:
        normal = -normal
    normal = _normalize(normal)
    part_id = (
        int(surface.part_ids[best_index]) if best_index < surface.part_ids.size else 0
    )
    return (
        hit,
        normal,
        float(np.clip(best_alpha, 0.0, 1.0)),
        max(0, part_id),
        int(best_index),
    )


def segment_hit_from_surface(
    surface: TriangleSurface3D,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    alpha_min: float = 1.0e-8,
    coordinate_tolerance_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, int, int] | None:
    return segment_hit_from_surface_ignoring_parts(
        surface,
        p0,
        p1,
        alpha_min=float(alpha_min),
        coordinate_tolerance_m=float(coordinate_tolerance_m),
    )


@dataclass(frozen=True)
class _TriangleProjectionTerms:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    ab: np.ndarray
    ac: np.ndarray
    ab2: float
    ac2: float
    ab_ac: float
    d1: float
    d2: float


@dataclass(frozen=True)
class _TriangleRegionTerms:
    d3: float
    d4: float
    d5: float
    d6: float
    va: float
    vb: float
    vc: float


def _triangle_dot_products(
    ab: np.ndarray,
    ac: np.ndarray,
    ap: np.ndarray,
) -> tuple[float, float, float, float, float]:
    return (
        float(np.dot(ab, ab)),
        float(np.dot(ac, ac)),
        float(np.dot(ab, ac)),
        float(np.dot(ab, ap)),
        float(np.dot(ac, ap)),
    )


def _resolved_triangle_dot_products(
    ab: np.ndarray,
    ac: np.ndarray,
    ap: np.ndarray,
) -> tuple[float, float, float, float, float]:
    raw = _triangle_dot_products(ab, ac, ap)
    raw_array = np.asarray(raw, dtype=np.float64)
    if np.all(np.isfinite(raw_array)) and raw[0] > 0.0 and raw[1] > 0.0:
        return raw
    scale = float(max(np.max(np.abs(ab)), np.max(np.abs(ac)), np.max(np.abs(ap))))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "closest-point query requires finite resolved triangle geometry"
        )
    return _triangle_dot_products(ab / scale, ac / scale, ap / scale)


def _triangle_projection_terms(
    point: np.ndarray,
    triangle: np.ndarray,
) -> _TriangleProjectionTerms:
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(triangle[0], dtype=np.float64)
    b = np.asarray(triangle[1], dtype=np.float64)
    c = np.asarray(triangle[2], dtype=np.float64)
    ab = b - a
    ac = c - a
    ab2, ac2, ab_ac, d1, d2 = _resolved_triangle_dot_products(ab, ac, p - a)
    return _TriangleProjectionTerms(a, b, c, ab, ac, ab2, ac2, ab_ac, d1, d2)


def _triangle_region_terms(terms: _TriangleProjectionTerms) -> _TriangleRegionTerms:
    d3 = terms.d1 - terms.ab2
    d4 = terms.d2 - terms.ab_ac
    vc = terms.d1 * d4 - d3 * terms.d2
    d5 = terms.d1 - terms.ab_ac
    d6 = terms.d2 - terms.ac2
    vb = d5 * terms.d2 - terms.d1 * d6
    va = d3 * d6 - d5 * d4
    return _TriangleRegionTerms(d3, d4, d5, d6, va, vb, vc)


def _closest_point_in_a_b_regions(
    terms: _TriangleProjectionTerms,
    regions: _TriangleRegionTerms,
) -> np.ndarray | None:
    if terms.d1 <= 0.0 and terms.d2 <= 0.0:
        return terms.a
    if regions.d3 >= 0.0 and regions.d4 <= regions.d3:
        return terms.b
    if regions.vc <= 0.0 and terms.d1 >= 0.0 and regions.d3 <= 0.0:
        denominator = terms.d1 - regions.d3
        if denominator <= 0.0:
            raise ValueError("closest-point edge denominator is unresolved")
        return terms.a + (terms.d1 / denominator) * terms.ab
    return None


def _closest_point_in_c_ac_regions(
    terms: _TriangleProjectionTerms,
    regions: _TriangleRegionTerms,
) -> np.ndarray | None:
    if regions.d6 >= 0.0 and regions.d5 <= regions.d6:
        return terms.c
    if regions.vb <= 0.0 and terms.d2 >= 0.0 and regions.d6 <= 0.0:
        denominator = terms.d2 - regions.d6
        if denominator <= 0.0:
            raise ValueError("closest-point edge denominator is unresolved")
        return terms.a + (terms.d2 / denominator) * terms.ac
    return None


def _closest_point_in_bc_or_face(
    terms: _TriangleProjectionTerms,
    regions: _TriangleRegionTerms,
) -> np.ndarray:
    if (
        regions.va <= 0.0
        and (regions.d4 - regions.d3) >= 0.0
        and (regions.d5 - regions.d6) >= 0.0
    ):
        denominator = (regions.d4 - regions.d3) + (regions.d5 - regions.d6)
        if denominator <= 0.0:
            raise ValueError("closest-point edge denominator is unresolved")
        weight = (regions.d4 - regions.d3) / denominator
        return terms.b + weight * (terms.c - terms.b)
    denominator = regions.va + regions.vb + regions.vc
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("closest-point face denominator is unresolved")
    return (
        terms.a
        + terms.ab * (regions.vb / denominator)
        + terms.ac * (regions.vc / denominator)
    )


def _closest_point_on_triangle(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Return the Ericson closest point while retaining float64 scale recovery."""

    terms = _triangle_projection_terms(point, tri)
    regions = _triangle_region_terms(terms)
    closest = _closest_point_in_a_b_regions(terms, regions)
    if closest is not None:
        return closest
    closest = _closest_point_in_c_ac_regions(terms, regions)
    if closest is not None:
        return closest
    return _closest_point_in_bc_or_face(terms, regions)


def _part_filtered_triangle_ids(
    surface: TriangleSurface3D,
    candidate_ids: np.ndarray,
    ignored_part_ids: frozenset[int],
) -> np.ndarray:
    return np.asarray(
        [
            int(index)
            for index in candidate_ids
            if int(surface.part_ids[int(index)]) not in ignored_part_ids
        ],
        dtype=np.int32,
    )


def _nearest_projection_candidate_ids(
    surface: TriangleSurface3D,
    query_point: np.ndarray,
    ignored_part_ids: frozenset[int],
) -> np.ndarray:
    radius = float(np.max(surface.grid.cell_size))
    offset = np.full(3, radius, dtype=np.float64)
    local_ids = query_triangle_candidates(
        surface.grid,
        query_point - offset,
        query_point + offset,
    )
    eligible_ids = _part_filtered_triangle_ids(
        surface,
        local_ids,
        ignored_part_ids,
    )
    if eligible_ids.size or not ignored_part_ids:
        return eligible_ids
    return _part_filtered_triangle_ids(
        surface,
        np.arange(surface.part_ids.size, dtype=np.int32),
        ignored_part_ids,
    )


def _nearest_triangle_projection(
    surface: TriangleSurface3D,
    query_point: np.ndarray,
    candidate_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    best_distance = np.inf
    best_point = None
    best_index = -1
    for candidate_id in candidate_ids:
        index = int(candidate_id)
        closest = _closest_point_on_triangle(query_point, surface.triangles[index])
        distance = _stable_norm(closest - query_point)
        if distance < best_distance:
            best_distance = distance
            best_point = closest
            best_index = index
    if best_point is None or best_index < 0:
        raise ValueError("No surface triangle available for nearest projection")
    return np.asarray(best_point, dtype=np.float64), int(best_index)


def _oriented_projection_normal(
    surface: TriangleSurface3D,
    triangle_index: int,
    projected_point: np.ndarray,
    inside_reference: np.ndarray | None,
) -> np.ndarray:
    normal = np.asarray(surface.normals[triangle_index], dtype=np.float64)
    if inside_reference is not None:
        reference = np.asarray(inside_reference, dtype=np.float64)
        if float(np.dot(reference - projected_point, normal)) > 0.0:
            normal = -normal
    return _normalize(normal)


def nearest_surface_point_ignoring_parts(
    surface: TriangleSurface3D,
    point: np.ndarray,
    *,
    inside_reference: np.ndarray | None = None,
    ignored_part_ids: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray, int, int]:
    query_point = np.asarray(point, dtype=np.float64)
    ignored = frozenset(int(value) for value in ignored_part_ids)
    candidate_ids = _nearest_projection_candidate_ids(
        surface,
        query_point,
        ignored,
    )
    best_point, best_index = _nearest_triangle_projection(
        surface,
        query_point,
        candidate_ids,
    )
    normal = _oriented_projection_normal(
        surface,
        best_index,
        best_point,
        inside_reference,
    )
    part_id = (
        int(surface.part_ids[best_index]) if best_index < surface.part_ids.size else 0
    )
    return (
        np.asarray(best_point, dtype=np.float64),
        normal,
        max(0, part_id),
        int(best_index),
    )


def nearest_surface_point(
    surface: TriangleSurface3D,
    point: np.ndarray,
    *,
    inside_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    return nearest_surface_point_ignoring_parts(
        surface,
        point,
        inside_reference=inside_reference,
    )


def _surface_ray_crossing_alphas(
    surface: TriangleSurface3D,
    start: np.ndarray,
    end: np.ndarray,
) -> list[float]:
    candidate_ids = query_triangle_candidates(surface.grid, start, end)
    alphas: list[float] = []
    for candidate_id in candidate_ids:
        index = int(candidate_id)
        alpha = _segment_triangle_intersection_alpha(
            start,
            end,
            surface.triangles[index],
        )
        if alpha is not None and alpha > 1.0e-8:
            alphas.append(float(alpha))
    return sorted(alphas)


def _unique_surface_crossing_count(alphas: list[float]) -> int:
    unique_count = 0
    last = -1.0
    for alpha in alphas:
        if unique_count == 0 or abs(alpha - last) > 1.0e-7:
            unique_count += 1
            last = alpha
    return unique_count


def point_inside_surface(
    surface: TriangleSurface3D,
    point: np.ndarray,
    *,
    on_boundary_tol: float,
) -> tuple[bool, bool]:
    query_point = np.asarray(point, dtype=np.float64)
    boundary_tolerance = max(float(on_boundary_tol), 0.0)
    if np.any(query_point < surface.bbox_min - boundary_tolerance) or np.any(
        query_point > surface.bbox_max + boundary_tolerance
    ):
        return False, False

    nearest_point, _, _, _ = nearest_surface_point(surface, query_point)
    if _stable_norm(nearest_point - query_point) <= boundary_tolerance:
        return True, True

    span = np.asarray(surface.bbox_max - surface.bbox_min, dtype=np.float64)
    domain_scale = float(np.linalg.norm(span))
    ray_end = np.asarray(
        [
            float(surface.bbox_max[0] + 3.5 * domain_scale),
            float(query_point[1] + 1.0e-9 * domain_scale),
            float(query_point[2] + 2.0e-9 * domain_scale),
        ],
        dtype=np.float64,
    )
    alphas = _surface_ray_crossing_alphas(surface, query_point, ray_end)
    if not alphas:
        return False, False
    inside = (_unique_surface_crossing_count(alphas) % 2) == 1
    return bool(inside), False
