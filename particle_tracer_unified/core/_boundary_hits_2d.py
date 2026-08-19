"""Two-dimensional boundary edge intersection and batch queries."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np
from numba import njit

from particle_tracer_unified.core.catalogs import is_internal_pass_through
from particle_tracer_unified.core.coordinate_systems import (
    canonicalize_axisymmetric_rz_positions,
)
from particle_tracer_unified.domain import BoundaryHit

_EDGE_ENDPOINT_TOL = 1.0e-9
_FLOAT64_GEOMETRY_EPS = 64.0 * np.finfo(np.float64).eps


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


class _EdgeIntersection2D(NamedTuple):
    position: np.ndarray
    normal: np.ndarray
    part_id: int
    edge_index: int
    segment_alpha: float
    edge_alpha: float


def _edge_part_id(part_ids: np.ndarray, edge_index: int) -> int:
    if edge_index >= part_ids.size:
        return 0
    return int(part_ids[edge_index])


def _edge_endpoint_is_ambiguous(edge_alpha: float) -> bool:
    return bool(
        np.isfinite(edge_alpha)
        and (edge_alpha <= _EDGE_ENDPOINT_TOL or edge_alpha >= 1.0 - _EDGE_ENDPOINT_TOL)
    )


def _is_transparent_interface(runtime, part_id: int) -> bool:
    wall_catalog = getattr(runtime, "wall_catalog", None)
    if wall_catalog is None:
        return False
    model = wall_catalog.model_for_part(int(part_id))
    return is_internal_pass_through(model)


def _radial_axis_tolerance(runtime) -> float:
    if str(getattr(runtime, "coordinate_system", "")) != "axisymmetric_rz":
        return -1.0
    boundary = getattr(getattr(runtime, "plan", None), "boundary", None)
    tolerance = float(getattr(boundary, "radial_axis_tolerance_m", np.nan))
    return tolerance if np.isfinite(tolerance) and tolerance >= 0.0 else 0.0


def _active_collision_edge_mask(
    runtime,
    segments: np.ndarray,
    part_ids: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(part_ids, dtype=np.int32)
    if ids.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(ids.shape, dtype=bool)
    for i, part_id in enumerate(ids):
        if _is_transparent_interface(runtime, int(part_id)):
            keep[i] = False
    axis_tolerance = _radial_axis_tolerance(runtime)
    if axis_tolerance >= 0.0:
        axis_edges = np.all(
            np.abs(np.asarray(segments, dtype=np.float64)[..., 0]) <= axis_tolerance,
            axis=1,
        )
        keep &= ~axis_edges
    return keep


def _segment_departs_edge_2d(
    start: np.ndarray,
    end: np.ndarray,
    edge_start: np.ndarray,
    edge: np.ndarray,
    edge_magnitude: float,
    tolerance: float,
) -> bool:
    """Return whether a segment starts on an edge line without crossing it.

    A particle released on a boundary, restarted after a reflection, or held
    in persistent contact begins its segment on the wall.  The line
    intersection is then found at ``segment_alpha = 0`` even though the motion
    never passes from one side of the wall to the other.  Treating that as a
    hit is what forces callers to displace the release point artificially.

    The predicate is local and scale-free: the start must lie on the edge line
    within the resolved geometry tolerance, and the end must stay on that same
    side.  A particle arriving from the interior starts further than the
    tolerance from the line, so a real crossing is never rejected here.
    """

    scale = 1.0 / edge_magnitude
    start_distance = _cross2d(edge, start - edge_start) * scale
    if not np.isfinite(start_distance) or abs(start_distance) > tolerance:
        return False
    end_distance = _cross2d(edge, end - edge_start) * scale
    if not np.isfinite(end_distance):
        return False
    if start_distance > 0.0:
        return bool(end_distance > -tolerance)
    if start_distance < 0.0:
        return bool(end_distance < tolerance)
    return True


@njit(cache=True)
def _segment_departs_edge_scalar_2d(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    q0x: float,
    q0y: float,
    sx: float,
    sy: float,
    edge_magnitude: float,
    tolerance: float,
) -> bool:
    """Numba form of :func:`_segment_departs_edge_2d` for the batch kernel."""

    scale = 1.0 / edge_magnitude
    start_distance = (sx * (ay - q0y) - sy * (ax - q0x)) * scale
    if not np.isfinite(start_distance) or abs(start_distance) > tolerance:
        return False
    end_distance = (sx * (by - q0y) - sy * (bx - q0x)) * scale
    if not np.isfinite(end_distance):
        return False
    if start_distance > 0.0:
        return end_distance > -tolerance
    if start_distance < 0.0:
        return end_distance < tolerance
    return True


def _segment_edge_intersection_2d(
    start: np.ndarray,
    direction: np.ndarray,
    direction_magnitude: float,
    segment: np.ndarray,
    *,
    part_id: int,
    edge_index: int,
    coordinate_tolerance: float,
    departure_tolerance: float = 0.0,
) -> _EdgeIntersection2D | None:
    edge_start = segment[0]
    edge = segment[1] - edge_start
    edge_magnitude = float(np.linalg.norm(edge))
    if edge_magnitude <= coordinate_tolerance:
        return None
    denominator = _cross2d(direction, edge)
    if abs(denominator) <= (
        _FLOAT64_GEOMETRY_EPS * direction_magnitude * edge_magnitude
    ):
        return None
    offset = edge_start - start
    segment_alpha = _cross2d(offset, edge) / denominator
    edge_alpha = _cross2d(offset, direction) / denominator
    endpoint_tolerance = 1.0e-12
    if (
        segment_alpha < -endpoint_tolerance
        or segment_alpha > 1.0 + endpoint_tolerance
        or edge_alpha < -endpoint_tolerance
        or edge_alpha > 1.0 + endpoint_tolerance
    ):
        return None
    if departure_tolerance > 0.0 and _segment_departs_edge_2d(
        start,
        start + direction,
        edge_start,
        edge,
        edge_magnitude,
        float(departure_tolerance),
    ):
        return None
    segment_alpha = float(np.clip(segment_alpha, 0.0, 1.0))
    position = start + segment_alpha * direction
    normal = np.array([-edge[1], edge[0]], dtype=np.float64)
    normal /= edge_magnitude
    if float(np.dot(start - position, normal)) > 0.0:
        normal = -normal
    return _EdgeIntersection2D(
        position=position,
        normal=normal,
        part_id=part_id,
        edge_index=edge_index,
        segment_alpha=segment_alpha,
        edge_alpha=float(np.clip(edge_alpha, 0.0, 1.0)),
    )


def _raw_segment_hit_from_boundary_edges(
    runtime,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    coordinate_tolerance_m: float = 0.0,
    departure_tolerance_m: float = 0.0,
) -> BoundaryHit | None:
    geometry_provider = runtime.geometry_provider
    segments, part_ids = _boundary_edges_2d(runtime)
    if geometry_provider is None or segments is None:
        return None
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    r = b - a
    coordinate_tolerance = max(float(coordinate_tolerance_m), 0.0)
    r_magnitude = float(np.linalg.norm(r))
    if r_magnitude <= coordinate_tolerance:
        return None
    best: _EdgeIntersection2D | None = None
    for edge_index, segment in enumerate(segments):
        part_id = _edge_part_id(part_ids, edge_index)
        candidate = _segment_edge_intersection_2d(
            a,
            r,
            r_magnitude,
            segment,
            part_id=part_id,
            edge_index=int(edge_index),
            coordinate_tolerance=coordinate_tolerance,
            departure_tolerance=max(float(departure_tolerance_m), 0.0),
        )
        if candidate is None:
            continue
        if best is None or candidate.segment_alpha < best.segment_alpha:
            best = candidate
    if best is None:
        return None
    return BoundaryHit(
        position=np.asarray(best.position, dtype=np.float64),
        normal=np.asarray(best.normal, dtype=np.float64),
        part_id=max(0, int(best.part_id)),
        alpha_hint=float(np.clip(best.segment_alpha, 0.0, 1.0)),
        primitive_id=int(best.edge_index),
        primitive_kind="edge",
        is_ambiguous=_edge_endpoint_is_ambiguous(best.edge_alpha),
    )


def _hit_with_segment_alpha(hit: BoundaryHit, alpha: float) -> BoundaryHit:
    return replace(hit, alpha_hint=float(np.clip(alpha, 0.0, 1.0)))


def segment_hit_from_boundary_edges(
    runtime,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    coordinate_tolerance_m: float = 0.0,
    departure_tolerance_m: float = 0.0,
) -> BoundaryHit | None:
    """Return the first physical 2D hit, folding signed RZ chart crossings."""

    start = np.asarray(p0, dtype=np.float64)
    end = np.asarray(p1, dtype=np.float64)
    tolerance = float(coordinate_tolerance_m)
    if _radial_axis_tolerance(runtime) < 0.0:
        return _raw_segment_hit_from_boundary_edges(
            runtime,
            start,
            end,
            coordinate_tolerance_m=tolerance,
            departure_tolerance_m=float(departure_tolerance_m),
        )

    radial_start = float(start[0])
    radial_end = float(end[0])
    canonical_start, canonical_end = canonicalize_axisymmetric_rz_positions(
        np.stack((start, end))
    )
    if radial_start * radial_end >= 0.0:
        return _raw_segment_hit_from_boundary_edges(
            runtime,
            canonical_start,
            canonical_end,
            coordinate_tolerance_m=tolerance,
            departure_tolerance_m=float(departure_tolerance_m),
        )

    crossing_fraction = radial_start / (radial_start - radial_end)
    axis_point = start + crossing_fraction * (end - start)
    axis_point[0] = 0.0
    for leg_start, leg_end, offset, fraction in (
        (canonical_start, axis_point, 0.0, crossing_fraction),
        (axis_point, canonical_end, crossing_fraction, 1.0 - crossing_fraction),
    ):
        hit = _raw_segment_hit_from_boundary_edges(
            runtime,
            leg_start,
            leg_end,
            coordinate_tolerance_m=tolerance,
            departure_tolerance_m=float(departure_tolerance_m),
        )
        if hit is not None:
            return _hit_with_segment_alpha(
                hit, offset + float(hit.alpha_hint) * fraction
            )
    return None


def _boundary_segments_2d(geometry) -> np.ndarray | None:
    if int(geometry.spatial_dim) != 2 or geometry.boundary_edges is None:
        return None
    segments = np.asarray(geometry.boundary_edges, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or segments.shape[0] == 0:
        return None
    return segments


def _boundary_edge_part_ids_2d(geometry, edge_count: int) -> np.ndarray:
    raw_part_ids = geometry.boundary_edge_part_ids
    if raw_part_ids is None:
        raw_part_ids = np.zeros(edge_count, dtype=np.int32)
    part_ids = np.asarray(raw_part_ids, dtype=np.int32)
    if part_ids.size < edge_count:
        return np.pad(
            part_ids,
            (0, edge_count - int(part_ids.size)),
            constant_values=0,
        )
    return part_ids[:edge_count]


def _boundary_edges_2d(runtime) -> tuple[np.ndarray | None, np.ndarray]:
    geometry_provider = getattr(runtime, "geometry_provider", None)
    if geometry_provider is None:
        return None, np.zeros(0, dtype=np.int32)
    geometry = geometry_provider.geometry
    segments = _boundary_segments_2d(geometry)
    if segments is None:
        return None, np.zeros(0, dtype=np.int32)
    part_ids = _boundary_edge_part_ids_2d(geometry, int(segments.shape[0]))
    keep = _active_collision_edge_mask(runtime, segments, part_ids)
    if not np.all(keep):
        segments = segments.copy()
        segments[~keep, 1, :] = segments[~keep, 0, :]
    if not np.any(keep):
        return None, np.zeros(0, dtype=np.int32)
    return segments, part_ids


@njit(cache=True)
def _empty_edge_hit_batch(n_points: int):
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
    return (
        hit_mask,
        hit_positions,
        hit_normals,
        hit_part_ids,
        hit_alphas,
        hit_primitive_ids,
        hit_ambiguous,
    )


@njit(cache=True)
def _bounds_overlap_2d(
    first_min_x: float,
    first_max_x: float,
    first_min_y: float,
    first_max_y: float,
    second_min_x: float,
    second_max_x: float,
    second_min_y: float,
    second_max_y: float,
) -> bool:
    return not (
        first_min_x > second_max_x
        or first_max_x < second_min_x
        or first_min_y > second_max_y
        or first_max_y < second_min_y
    )


@njit(cache=True)
def _segment_parameters_are_bounded(segment_alpha: float, edge_alpha: float) -> bool:
    endpoint_tolerance = 1.0e-12
    return not (
        segment_alpha < -endpoint_tolerance
        or segment_alpha > 1.0 + endpoint_tolerance
        or edge_alpha < -endpoint_tolerance
        or edge_alpha > 1.0 + endpoint_tolerance
    )


@njit(cache=True)
def _batch_edge_intersection_2d(
    edge_arr: np.ndarray,
    edge_index: int,
    ax: float,
    ay: float,
    rx: float,
    ry: float,
    r_length2: float,
    segment_bounds: tuple[float, float, float, float],
    coordinate_padding: float,
    departure_tolerance: float,
):
    q0x = edge_arr[edge_index, 0, 0]
    q0y = edge_arr[edge_index, 0, 1]
    q1x = edge_arr[edge_index, 1, 0]
    q1y = edge_arr[edge_index, 1, 1]
    if not _bounds_overlap_2d(
        *segment_bounds,
        min(q0x, q1x),
        max(q0x, q1x),
        min(q0y, q1y),
        max(q0y, q1y),
    ):
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    sx = q1x - q0x
    sy = q1y - q0y
    s_length2 = sx * sx + sy * sy
    if s_length2 <= coordinate_padding * coordinate_padding:
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    denominator = rx * sy - ry * sx
    if abs(denominator) <= (_FLOAT64_GEOMETRY_EPS * (r_length2 * s_length2) ** 0.5):
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    qmp_x = q0x - ax
    qmp_y = q0y - ay
    segment_alpha = (qmp_x * sy - qmp_y * sx) / denominator
    edge_alpha = (qmp_x * ry - qmp_y * rx) / denominator
    if not _segment_parameters_are_bounded(segment_alpha, edge_alpha):
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if departure_tolerance > 0.0 and _segment_departs_edge_scalar_2d(
        ax,
        ay,
        ax + rx,
        ay + ry,
        q0x,
        q0y,
        sx,
        sy,
        s_length2**0.5,
        departure_tolerance,
    ):
        return False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    segment_alpha = min(1.0, max(0.0, segment_alpha))
    hit_x = ax + segment_alpha * rx
    hit_y = ay + segment_alpha * ry
    normal_x = -sy
    normal_y = sx
    magnitude = s_length2**0.5
    normal_x /= magnitude
    normal_y /= magnitude
    if (ax - hit_x) * normal_x + (ay - hit_y) * normal_y > 0.0:
        normal_x = -normal_x
        normal_y = -normal_y
    return (
        True,
        segment_alpha,
        min(1.0, max(0.0, edge_alpha)),
        hit_x,
        hit_y,
        normal_x,
        normal_y,
    )


@njit(cache=True)
def _first_batch_edge_intersection_2d(
    edge_arr: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    coordinate_padding: float,
    departure_tolerance: float,
):
    ax = start[0]
    ay = start[1]
    bx = end[0]
    by = end[1]
    rx = bx - ax
    ry = by - ay
    r_length2 = rx * rx + ry * ry
    if r_length2 <= coordinate_padding * coordinate_padding:
        return -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    segment_bounds = (
        min(ax, bx) - coordinate_padding,
        max(ax, bx) + coordinate_padding,
        min(ay, by) - coordinate_padding,
        max(ay, by) + coordinate_padding,
    )
    best_edge = -1
    best_alpha = 2.0
    best_edge_alpha = np.nan
    best_hit_x = np.nan
    best_hit_y = np.nan
    best_normal_x = np.nan
    best_normal_y = np.nan
    for edge_index in range(edge_arr.shape[0]):
        candidate = _batch_edge_intersection_2d(
            edge_arr,
            edge_index,
            ax,
            ay,
            rx,
            ry,
            r_length2,
            segment_bounds,
            coordinate_padding,
            departure_tolerance,
        )
        if not candidate[0] or candidate[1] >= best_alpha:
            continue
        best_edge = edge_index
        best_alpha = candidate[1]
        best_edge_alpha = candidate[2]
        best_hit_x = candidate[3]
        best_hit_y = candidate[4]
        best_normal_x = candidate[5]
        best_normal_y = candidate[6]
    return (
        best_edge,
        best_alpha,
        best_edge_alpha,
        best_hit_x,
        best_hit_y,
        best_normal_x,
        best_normal_y,
    )


@njit(cache=True)
def _segment_hits_from_boundary_edges_batch_kernel(
    edge_arr: np.ndarray,
    edge_part_ids: np.ndarray,
    starts_arr: np.ndarray,
    ends_arr: np.ndarray,
    coordinate_tolerance_m: float,
    departure_tolerance_m: float,
):
    n_points = starts_arr.shape[0]
    outputs = _empty_edge_hit_batch(n_points)
    hit_mask = outputs[0]
    hit_positions = outputs[1]
    hit_normals = outputs[2]
    hit_part_ids = outputs[3]
    hit_alphas = outputs[4]
    hit_primitive_ids = outputs[5]
    hit_ambiguous = outputs[6]

    coordinate_padding = max(float(coordinate_tolerance_m), 0.0)
    departure_tolerance = max(float(departure_tolerance_m), 0.0)
    for i in range(n_points):
        candidate = _first_batch_edge_intersection_2d(
            edge_arr,
            starts_arr[i],
            ends_arr[i],
            coordinate_padding,
            departure_tolerance,
        )
        best_edge = candidate[0]
        if best_edge >= 0:
            hit_mask[i] = True
            hit_positions[i, 0] = candidate[3]
            hit_positions[i, 1] = candidate[4]
            hit_normals[i, 0] = candidate[5]
            hit_normals[i, 1] = candidate[6]
            hit_part_ids[i] = edge_part_ids[best_edge]
            hit_alphas[i] = candidate[1]
            hit_primitive_ids[i] = best_edge
            hit_ambiguous[i] = (
                candidate[2] <= _EDGE_ENDPOINT_TOL
                or candidate[2] >= 1.0 - _EDGE_ENDPOINT_TOL
            )
    return outputs


def _segment_hits_from_boundary_edges_batch(
    segments: np.ndarray,
    part_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    coordinate_tolerance_m: float = 0.0,
    departure_tolerance_m: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    starts_arr = np.asarray(starts, dtype=np.float64)
    ends_arr = np.asarray(ends, dtype=np.float64)
    edge_arr = np.asarray(segments, dtype=np.float64)
    edge_part_ids = np.asarray(part_ids, dtype=np.int32)
    n_points = int(starts_arr.shape[0])
    if n_points == 0 or edge_arr.shape[0] == 0:
        return _empty_edge_hit_batch(n_points)
    return _segment_hits_from_boundary_edges_batch_kernel(
        edge_arr,
        edge_part_ids,
        starts_arr,
        ends_arr,
        float(coordinate_tolerance_m),
        float(departure_tolerance_m),
    )


def _batch_polyline_inputs_2d(
    starts: np.ndarray,
    stage_points: np.ndarray,
    particle_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts_arr = np.asarray(starts, dtype=np.float64)
    stages = np.asarray(stage_points, dtype=np.float64)
    if starts_arr.ndim != 2 or starts_arr.shape[1] != 2:
        raise ValueError("2D batch boundary hit starts require shape (n, 2)")
    if (
        stages.ndim != 3
        or stages.shape[0] != starts_arr.shape[0]
        or stages.shape[2] != 2
    ):
        raise ValueError("2D batch boundary hit stage_points require shape (n, k, 2)")
    n_points = int(starts_arr.shape[0])
    if particle_indices is None:
        particle_ids = np.arange(n_points, dtype=np.int64)
    else:
        particle_ids = np.asarray(particle_indices, dtype=np.int64)
        if particle_ids.shape[0] != n_points:
            raise ValueError("particle_indices length must match starts")
    return starts_arr, stages, particle_ids


def polyline_hits_from_boundary_edges_batch(
    runtime,
    starts: np.ndarray,
    stage_points: np.ndarray,
    *,
    particle_indices: np.ndarray | None = None,
    chunk_size: int = 2048,
    coordinate_tolerance_m: float = 0.0,
    departure_tolerance_m: float = 0.0,
) -> dict[int, BoundaryHit]:
    segments, part_ids = _boundary_edges_2d(runtime)
    if segments is None:
        return {}
    starts_arr, stages, particle_ids = _batch_polyline_inputs_2d(
        starts,
        stage_points,
        particle_indices,
    )
    if _radial_axis_tolerance(runtime) >= 0.0:
        results: dict[int, BoundaryHit] = {}
        segment_count = max(1, int(stages.shape[1]))
        for row, start in enumerate(starts_arr):
            segment_start = start
            for segment_index in range(segment_count):
                hit = segment_hit_from_boundary_edges(
                    runtime,
                    segment_start,
                    stages[row, segment_index],
                    coordinate_tolerance_m=float(coordinate_tolerance_m),
                    departure_tolerance_m=float(departure_tolerance_m),
                )
                if hit is not None:
                    results[int(particle_ids[row])] = _hit_with_segment_alpha(
                        hit,
                        (float(segment_index) + float(hit.alpha_hint))
                        / float(segment_count),
                    )
                    break
                segment_start = stages[row, segment_index]
        return results
    n_points = int(starts_arr.shape[0])
    hit_results: dict[int, BoundaryHit] = {}
    active_rows = np.ones(n_points, dtype=bool)
    segment_start = starts_arr
    segment_count = max(1, int(stages.shape[1]))
    for segment_index in range(segment_count):
        rows = np.flatnonzero(active_rows)
        if rows.size == 0:
            break
        segment_end = stages[rows, segment_index, :]
        batch = _segment_hits_from_boundary_edges_batch(
            segments,
            part_ids,
            segment_start[rows],
            segment_end,
            coordinate_tolerance_m=float(coordinate_tolerance_m),
            departure_tolerance_m=float(departure_tolerance_m),
        )
        (
            hit_mask,
            hit_pos,
            hit_normal,
            hit_part,
            hit_alpha,
            hit_primitive,
            hit_ambiguous,
        ) = batch
        hit_rows_local = np.flatnonzero(hit_mask)
        if hit_rows_local.size:
            original_rows = rows[hit_rows_local]
            normalized_alpha = (
                float(segment_index) + np.clip(hit_alpha[hit_rows_local], 0.0, 1.0)
            ) / float(segment_count)
            for result_index in range(int(original_rows.size)):
                output_row = int(original_rows[result_index])
                local_idx = int(hit_rows_local[result_index])
                alpha_value = normalized_alpha[result_index]
                hit_results[int(particle_ids[int(output_row)])] = BoundaryHit(
                    position=np.asarray(hit_pos[local_idx], dtype=np.float64),
                    normal=np.asarray(hit_normal[local_idx], dtype=np.float64),
                    part_id=max(0, int(hit_part[local_idx])),
                    alpha_hint=float(np.clip(alpha_value, 0.0, 1.0)),
                    primitive_id=int(hit_primitive[local_idx]),
                    primitive_kind="edge",
                    is_ambiguous=bool(hit_ambiguous[local_idx]),
                )
            active_rows[original_rows] = False
        segment_start = stages[:, segment_index, :]
    return hit_results


def nearest_boundary_edge_features_2d(
    runtime,
    points: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    segments, part_ids = _boundary_edges_2d(runtime)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("2D nearest-boundary diagnostics require shape (n, 2)")
    n_points = int(pts.shape[0])
    nearest_part_ids = np.zeros(n_points, dtype=np.int32)
    nearest_distances = np.full(n_points, np.nan, dtype=np.float64)
    if segments is None or n_points == 0:
        return nearest_part_ids, nearest_distances
    q0 = segments[:, 0, :]
    q1 = segments[:, 1, :]
    edge = q1 - q0
    edge_len2 = np.einsum("ij,ij->i", edge, edge)
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
        nearest_part_ids[rows[finite]] = np.asarray(
            part_ids[best_edge[finite]], dtype=np.int32
        )
        nearest_distances[rows[finite]] = np.sqrt(best_dist2[finite])
    return nearest_part_ids, nearest_distances


__all__ = (
    "nearest_boundary_edge_features_2d",
    "polyline_hits_from_boundary_edges_batch",
    "segment_hit_from_boundary_edges",
)
