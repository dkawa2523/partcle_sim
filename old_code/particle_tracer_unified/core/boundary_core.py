from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from particle_tracer_unified.domain import BoundaryHit as DomainBoundaryHit

from .coordinate_systems import canonicalize_axisymmetric_rz_positions
from .geometry2d import (
    points_inside_boundary_edges_2d_with_boundary,
    points_inside_boundary_loops_2d_with_boundary,
)
from .geometry3d import TriangleSurface3D, point_inside_surface
from .grid_sampling import sample_grid_scalar as _sample_grid_scalar
from .grid_sampling import sample_grid_scalar_points_2d as _sample_grid_scalar_points_2d


def _geometry_query_points(runtime, positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float64)
    if str(getattr(runtime, "coordinate_system", "")) != "axisymmetric_rz":
        return points
    return canonicalize_axisymmetric_rz_positions(points)


@dataclass(frozen=True)
class BoundaryBroadPhase2D:
    """Immutable geometry-derived data reused by every 2D solver step."""

    grid_spacing_m: float
    edge_aabb_min_padded: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    edge_aabb_max_padded: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    edge_aabb_padding_m: float = 0.0


@dataclass(frozen=True)
class BoundaryService:
    inside: Callable[[np.ndarray], bool]
    inside_strict: Callable[[np.ndarray], bool]
    segment_hit: Callable[[np.ndarray, np.ndarray], DomainBoundaryHit | None]
    polyline_hit: Callable[[np.ndarray, np.ndarray], DomainBoundaryHit | None]
    nearest_projection: Callable[[np.ndarray, np.ndarray], DomainBoundaryHit | None]
    primary_hit_counter_key: str
    triangle_surface_3d: TriangleSurface3D | None = None
    broad_phase_2d: BoundaryBroadPhase2D | None = None

    def contains(self, points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("BoundaryQuery.contains expects shape (point, 2|3)")
        return np.fromiter(
            (bool(self.inside(point)) for point in points),
            dtype=bool,
            count=int(points.shape[0]),
        )

    def first_hit(
        self,
        start_m: np.ndarray,
        end_m: np.ndarray,
    ) -> DomainBoundaryHit | None:
        return self.segment_hit(
            np.asarray(start_m, dtype=np.float64),
            np.asarray(end_m, dtype=np.float64),
        )


def runtime_bounds(runtime) -> tuple[np.ndarray, np.ndarray]:
    if runtime.geometry_provider is not None:
        axes = runtime.geometry_provider.geometry.axes
    elif runtime.field_provider is not None:
        field = runtime.field_provider.field
        if hasattr(field, "axes"):
            axes = field.axes
        elif hasattr(field, "mesh_vertices"):
            vertices = np.asarray(field.mesh_vertices, dtype=np.float64)
            return np.min(vertices, axis=0), np.max(vertices, axis=0)
        else:
            raise ValueError(
                "Field provider does not expose axes or mesh_vertices for "
                "runtime bounds"
            )
    else:
        raise ValueError(
            "High-fidelity solver requires geometry_provider or field_provider"
        )
    mins = np.array([float(axis[0]) for axis in axes], dtype=np.float64)
    maxs = np.array([float(axis[-1]) for axis in axes], dtype=np.float64)
    return mins, maxs


def sample_geometry_sdf(runtime, position: np.ndarray) -> float:
    query_position = _geometry_query_points(runtime, position)
    if runtime.geometry_provider is None:
        mins, maxs = runtime_bounds(runtime)
        outside = np.maximum(mins - query_position, 0.0) + np.maximum(
            query_position - maxs, 0.0
        )
        return float(np.linalg.norm(outside))
    geom = runtime.geometry_provider.geometry
    return float(
        _sample_grid_scalar(
            np.asarray(geom.sdf, dtype=np.float64),
            geom.axes,
            query_position,
        )
    )


def sample_geometry_sdf_points_2d(runtime, positions: np.ndarray) -> np.ndarray:
    pts = _geometry_query_points(runtime, positions)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("2D SDF sampling requires shape (n, 2)")
    if runtime.geometry_provider is None:
        return np.full(pts.shape[0], np.nan, dtype=np.float64)
    geom = runtime.geometry_provider.geometry
    if int(geom.spatial_dim) != 2:
        return np.full(pts.shape[0], np.nan, dtype=np.float64)
    try:
        return _sample_grid_scalar_points_2d(
            np.asarray(geom.sdf, dtype=np.float64), geom.axes, pts
        )
    except Exception:
        return np.full(pts.shape[0], np.nan, dtype=np.float64)


def sample_geometry_part_id(runtime, position: np.ndarray) -> int:
    if runtime.geometry_provider is None:
        return 0
    geom = runtime.geometry_provider.geometry
    value = _sample_grid_scalar(
        np.asarray(geom.nearest_boundary_part_id_map, dtype=np.float64),
        geom.axes,
        _geometry_query_points(runtime, position),
    )
    if not np.isfinite(value):
        return 0
    return int(max(0, round(value)))


def sample_geometry_normal(runtime, position: np.ndarray) -> np.ndarray:
    query_position = _geometry_query_points(runtime, position)
    if runtime.geometry_provider is None:
        mins, maxs = runtime_bounds(runtime)
        pos = query_position
        dim = pos.size
        distance = np.minimum(pos - mins, maxs - pos)
        axis_index = int(np.argmin(distance))
        normal = np.zeros(dim, dtype=np.float64)
        normal[axis_index] = (
            -1.0
            if abs(pos[axis_index] - mins[axis_index])
            < abs(maxs[axis_index] - pos[axis_index])
            else 1.0
        )
        return normal
    geom = runtime.geometry_provider.geometry
    values = [
        _sample_grid_scalar(
            np.asarray(component, dtype=np.float64),
            geom.axes,
            query_position,
        )
        for component in geom.normal_components
    ]
    normal = np.asarray(values, dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    if magnitude <= 1.0e-30:
        normal = np.zeros(geom.spatial_dim, dtype=np.float64)
        normal[-1] = 1.0
        return normal
    return normal / magnitude


def _geometry_bbox_candidates_2d(
    runtime,
    points: np.ndarray,
    on_boundary_tol_m: float,
) -> np.ndarray:
    mins, maxs = runtime_bounds(runtime)
    padding = max(float(on_boundary_tol_m), 0.0)
    return (
        (points[:, 0] >= mins[0] - padding)
        & (points[:, 0] <= maxs[0] + padding)
        & (points[:, 1] >= mins[1] - padding)
        & (points[:, 1] <= maxs[1] + padding)
    )


def _classify_geometry_points_2d(
    runtime,
    points: np.ndarray,
    on_boundary_tol_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    geometry_provider = runtime.geometry_provider
    if geometry_provider is not None:
        geometry = geometry_provider.geometry
        edges = geometry.boundary_edges
        loops = geometry.boundary_loops_2d
        metadata = getattr(geometry, "metadata", {})
        has_internal_collision_edges = bool(
            isinstance(metadata, Mapping)
            and int(metadata.get("internal_interface_edge_count", 0)) > 0
        )
        if edges is not None and not has_internal_collision_edges:
            return points_inside_boundary_edges_2d_with_boundary(
                points,
                edges,
                on_edge_tol=float(on_boundary_tol_m),
            )
        if loops:
            return points_inside_boundary_loops_2d_with_boundary(
                points,
                loops,
                on_edge_tol=float(on_boundary_tol_m),
            )
        if edges is not None:
            return points_inside_boundary_edges_2d_with_boundary(
                points,
                edges,
                on_edge_tol=float(on_boundary_tol_m),
            )
    inside = np.fromiter(
        (sample_geometry_sdf(runtime, point) <= 0.0 for point in points),
        dtype=bool,
        count=int(points.shape[0]),
    )
    return inside, np.zeros(points.shape[0], dtype=bool)


def points_inside_geometry_2d(
    runtime,
    positions: np.ndarray,
    on_boundary_tol_m: float,
    return_on_boundary: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    pts = _geometry_query_points(runtime, positions)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("2D geometry queries require shape (n, 2)")
    bbox = _geometry_bbox_candidates_2d(runtime, pts, on_boundary_tol_m)
    inside = np.zeros(pts.shape[0], dtype=bool)
    on_boundary = np.zeros(pts.shape[0], dtype=bool)
    if np.any(bbox):
        inside_bbox, on_boundary_bbox = _classify_geometry_points_2d(
            runtime,
            pts[bbox],
            on_boundary_tol_m,
        )
        inside[bbox] = inside_bbox
        on_boundary[bbox] = on_boundary_bbox
    return (inside, on_boundary) if return_on_boundary else inside


def inside_geometry_with_boundary(
    runtime,
    position: np.ndarray,
    *,
    on_boundary_tol_m: float,
    triangle_surface_3d: TriangleSurface3D | None = None,
) -> tuple[bool, bool]:
    mins, maxs = runtime_bounds(runtime)
    pos = _geometry_query_points(runtime, position)
    bbox_padding = max(float(on_boundary_tol_m), 0.0)
    if np.any(pos < mins - bbox_padding) or np.any(pos > maxs + bbox_padding):
        return False, False
    geometry_provider = runtime.geometry_provider
    if (
        geometry_provider is not None
        and int(geometry_provider.geometry.spatial_dim) == 2
    ):
        inside, on_boundary = points_inside_geometry_2d(
            runtime,
            pos[None, :],
            on_boundary_tol_m=float(on_boundary_tol_m),
            return_on_boundary=True,
        )
        return bool(inside[0]), bool(on_boundary[0])
    if (
        geometry_provider is not None
        and int(geometry_provider.geometry.spatial_dim) == 3
        and triangle_surface_3d is not None
    ):
        inside, on_boundary = point_inside_surface(
            triangle_surface_3d,
            pos,
            on_boundary_tol=float(on_boundary_tol_m),
        )
        return bool(inside), bool(on_boundary)
    return bool(sample_geometry_sdf(runtime, pos) <= 0.0), False


def inside_geometry(
    runtime,
    position: np.ndarray,
    *,
    on_boundary_tol_m: float,
    triangle_surface_3d: TriangleSurface3D | None = None,
) -> bool:
    inside, _ = inside_geometry_with_boundary(
        runtime,
        position,
        on_boundary_tol_m=on_boundary_tol_m,
        triangle_surface_3d=triangle_surface_3d,
    )
    return bool(inside)


__all__ = (
    "BoundaryBroadPhase2D",
    "BoundaryService",
    "inside_geometry",
    "inside_geometry_with_boundary",
    "points_inside_geometry_2d",
    "runtime_bounds",
    "sample_geometry_normal",
    "sample_geometry_part_id",
    "sample_geometry_sdf",
    "sample_geometry_sdf_points_2d",
)
