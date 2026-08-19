"""Conservative 2D and 3D collision candidate selection."""

from __future__ import annotations

import numpy as np

from particle_tracer_unified.core.boundary_core import sample_geometry_sdf_points_2d
from particle_tracer_unified.core.geometry3d import (
    TriangleSurface3D,
    query_triangle_candidates,
)
from particle_tracer_unified.domain import BoundaryQuery


def geometry_grid_spacing_2d(
    runtime,
    boundary_service: BoundaryQuery[TriangleSurface3D],
) -> float:
    broad_phase = getattr(boundary_service, "broad_phase_2d", None)
    if broad_phase is not None:
        return float(getattr(broad_phase, "grid_spacing_m", 0.0))
    geometry_provider = getattr(runtime, "geometry_provider", None)
    if geometry_provider is None:
        return 0.0
    geometry = geometry_provider.geometry
    if (
        int(getattr(geometry, "spatial_dim", 0)) != 2
        or len(getattr(geometry, "axes", ())) != 2
    ):
        return 0.0
    spacings = []
    for axis in geometry.axes:
        values = np.asarray(axis, dtype=np.float64)
        differences = np.diff(values)
        differences = differences[np.isfinite(differences) & (differences > 0.0)]
        if differences.size:
            spacings.append(float(np.min(differences)))
    return float(min(spacings)) if spacings else 0.0


def far_from_wall_mask_2d(
    runtime,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    indices: np.ndarray,
    x_start: np.ndarray,
    x_mid: np.ndarray,
    x_end: np.ndarray,
    *,
    on_boundary_tol_m: float,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.size == 0:
        return np.zeros(0, dtype=bool)
    geometry_provider = getattr(runtime, "geometry_provider", None)
    if geometry_provider is None or int(geometry_provider.geometry.spatial_dim) != 2:
        return np.zeros(selected.size, dtype=bool)
    grid_spacing = geometry_grid_spacing_2d(runtime, boundary_service)
    if not np.isfinite(grid_spacing) or grid_spacing <= 0.0:
        return np.zeros(selected.size, dtype=bool)
    count = int(selected.size)
    points = np.empty((3 * count, 2), dtype=np.float64)
    np.take(np.asarray(x_start, dtype=np.float64), selected, axis=0, out=points[:count])
    np.take(
        np.asarray(x_mid, dtype=np.float64),
        selected,
        axis=0,
        out=points[count : 2 * count],
    )
    np.take(
        np.asarray(x_end, dtype=np.float64), selected, axis=0, out=points[2 * count :]
    )
    sdf = sample_geometry_sdf_points_2d(runtime, points)
    start = points[:count]
    midpoint = points[count : 2 * count]
    end = points[2 * count :]
    sdf_start = sdf[:count]
    sdf_midpoint = sdf[count : 2 * count]
    sdf_end = sdf[2 * count :]
    sweep_radius = np.maximum(
        np.linalg.norm(midpoint - start, axis=1),
        np.linalg.norm(end - start, axis=1),
    )
    margin = max(float(on_boundary_tol_m), 2.0 * grid_spacing) + (0.25 * sweep_radius)
    finite = (
        np.isfinite(sdf_start)
        & np.isfinite(sdf_midpoint)
        & np.isfinite(sdf_end)
        & np.isfinite(sweep_radius)
    )
    return (
        finite
        & (sdf_start < -(sweep_radius + margin))
        & (sdf_midpoint < -margin)
        & (sdf_end < -margin)
    )


def sdf_strict_inside_mask_2d(
    runtime,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    positions: np.ndarray,
    *,
    on_boundary_tol_m: float,
) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=bool)
    grid_spacing = geometry_grid_spacing_2d(runtime, boundary_service)
    if not np.isfinite(grid_spacing) or grid_spacing <= 0.0:
        return np.zeros(points.shape[0], dtype=bool)
    sdf = sample_geometry_sdf_points_2d(runtime, points)
    margin = float(max(float(on_boundary_tol_m), 0.5 * grid_spacing))
    return np.isfinite(sdf) & (sdf < -margin)


def boundary_edge_aabb_arrays_2d(
    runtime,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    *,
    on_boundary_tol_m: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    broad_phase = getattr(boundary_service, "broad_phase_2d", None)
    if broad_phase is not None:
        return _cached_edge_bounds(
            broad_phase, requested_padding=float(on_boundary_tol_m)
        )
    geometry_provider = getattr(runtime, "geometry_provider", None)
    if geometry_provider is None:
        return None, None
    geometry = getattr(geometry_provider, "geometry", None)
    if (
        geometry is None
        or int(getattr(geometry, "spatial_dim", 0)) != 2
        or getattr(geometry, "boundary_edges", None) is None
    ):
        return None, None
    edges = np.asarray(geometry.boundary_edges, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2) or edges.shape[0] == 0:
        return None, None
    padding = float(on_boundary_tol_m)
    return np.min(edges, axis=1) - padding, np.max(edges, axis=1) + padding


def _cached_edge_bounds(
    broad_phase,
    *,
    requested_padding: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    edge_min = getattr(broad_phase, "edge_aabb_min_padded", None)
    edge_max = getattr(broad_phase, "edge_aabb_max_padded", None)
    if edge_min is None or edge_max is None:
        return None, None
    padding_delta = requested_padding - float(
        getattr(broad_phase, "edge_aabb_padding_m", 0.0)
    )
    if padding_delta == 0.0:
        return edge_min, edge_max
    return (
        np.asarray(edge_min, dtype=np.float64) - padding_delta,
        np.asarray(edge_max, dtype=np.float64) + padding_delta,
    )


def polyline_stage_points(
    index: int,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
) -> np.ndarray:
    return np.asarray((x_mid_trial[int(index)], x_trial[int(index)]), dtype=np.float64)


def edge_aabb_candidate_mask_2d(
    runtime,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    indices: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    *,
    on_boundary_tol_m: float,
) -> tuple[np.ndarray, int]:
    selected = np.asarray(indices, dtype=np.int64)
    candidate = np.ones(selected.size, dtype=bool)
    if selected.size == 0:
        return candidate, 0
    edge_min, edge_max = boundary_edge_aabb_arrays_2d(
        runtime,
        boundary_service,
        on_boundary_tol_m=float(on_boundary_tol_m),
    )
    if edge_min is None or edge_max is None:
        return candidate, int(selected.size)
    padding = float(on_boundary_tol_m)
    unknown = 0
    for row, raw_index in enumerate(selected):
        index = int(raw_index)
        points = np.vstack(
            (
                np.asarray(x[index], dtype=np.float64)[None, :],
                polyline_stage_points(index, x_trial, x_mid_trial),
            )
        )
        if points.shape[1] != 2 or not np.all(np.isfinite(points)):
            unknown += 1
            continue
        candidate[row] = any(
            _segment_overlaps_bounds(
                points[segment_index],
                points[segment_index + 1],
                edge_min,
                edge_max,
                padding,
            )
            for segment_index in range(points.shape[0] - 1)
        )
    return candidate, int(unknown)


def _segment_overlaps_bounds(
    start: np.ndarray,
    end: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    padding: float,
) -> bool:
    segment_min = np.minimum(start, end) - padding
    segment_max = np.maximum(start, end) + padding
    overlap = np.all(
        (segment_max[None, :] >= bounds_min) & (bounds_max >= segment_min[None, :]),
        axis=1,
    )
    return bool(np.any(overlap))


def triangle_aabb_candidate_mask_3d(
    surface: TriangleSurface3D | None,
    indices: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    *,
    on_boundary_tol_m: float,
) -> tuple[np.ndarray, int]:
    selected = np.asarray(indices, dtype=np.int64)
    candidate = np.ones(selected.size, dtype=bool)
    if selected.size == 0:
        return candidate, 0
    if surface is None:
        return candidate, int(selected.size)
    triangle_min = np.asarray(surface.grid.triangle_mins, dtype=np.float64)
    triangle_max = np.asarray(surface.grid.triangle_maxs, dtype=np.float64)
    padding = float(on_boundary_tol_m)
    unknown = 0
    for row, raw_index in enumerate(selected):
        index = int(raw_index)
        points = np.vstack(
            (
                np.asarray(x[index], dtype=np.float64)[None, :],
                polyline_stage_points(index, x_trial, x_mid_trial),
            )
        )
        if points.shape[1] != 3 or not np.all(np.isfinite(points)):
            unknown += 1
            continue
        candidate[row], unknown_segment = _triangle_polyline_candidate(
            surface,
            points,
            triangle_min,
            triangle_max,
            padding,
        )
        unknown += int(unknown_segment)
    return candidate, int(unknown)


def _triangle_polyline_candidate(
    surface: TriangleSurface3D,
    points: np.ndarray,
    triangle_min: np.ndarray,
    triangle_max: np.ndarray,
    padding: float,
) -> tuple[bool, bool]:
    for segment_index in range(points.shape[0] - 1):
        start = points[segment_index]
        end = points[segment_index + 1]
        ids = np.asarray(
            query_triangle_candidates(surface.grid, start, end), dtype=np.int64
        )
        if ids.size == 0:
            return True, True
        if _segment_overlaps_bounds(
            start,
            end,
            triangle_min[ids],
            triangle_max[ids],
            padding,
        ):
            return True, False
    return False, False
