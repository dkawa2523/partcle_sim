from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .geometry2d import points_inside_boundary_loops_2d_with_boundary
from .geometry3d import TriangleSurface3D, point_inside_surface
from .grid_sampling import sample_grid_scalar as _sample_grid_scalar


@dataclass(frozen=True)
class BoundaryHit:
    position: np.ndarray
    normal: np.ndarray
    part_id: int
    alpha_hint: float = 0.0


@dataclass(frozen=True)
class BoundaryService:
    inside: Callable[[np.ndarray], bool]
    inside_strict: Callable[[np.ndarray], bool]
    segment_hit: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]]
    polyline_hit: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]]
    nearest_projection: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]]
    primary_hit_counter_key: str
    triangle_surface_3d: Optional[TriangleSurface3D] = None


def runtime_bounds(runtime) -> Tuple[np.ndarray, np.ndarray]:
    if runtime.geometry_provider is not None:
        axes = runtime.geometry_provider.geometry.axes
    elif runtime.field_provider is not None:
        field = runtime.field_provider.field
        if hasattr(field, 'axes'):
            axes = field.axes
        elif hasattr(field, 'mesh_vertices'):
            vertices = np.asarray(field.mesh_vertices, dtype=np.float64)
            return np.min(vertices, axis=0), np.max(vertices, axis=0)
        else:
            raise ValueError('Field provider does not expose axes or mesh_vertices for runtime bounds')
    else:
        raise ValueError('High-fidelity solver requires geometry_provider or field_provider')
    mins = np.array([float(axis[0]) for axis in axes], dtype=np.float64)
    maxs = np.array([float(axis[-1]) for axis in axes], dtype=np.float64)
    return mins, maxs


def sample_geometry_sdf(runtime, position: np.ndarray) -> float:
    if runtime.geometry_provider is None:
        mins, maxs = runtime_bounds(runtime)
        outside = np.maximum(mins - position, 0.0) + np.maximum(position - maxs, 0.0)
        return float(np.linalg.norm(outside))
    geom = runtime.geometry_provider.geometry
    return float(
        _sample_grid_scalar(
            np.asarray(geom.sdf, dtype=np.float64),
            geom.axes,
            np.asarray(position, dtype=np.float64),
        )
    )


def sample_geometry_part_id(runtime, position: np.ndarray) -> int:
    if runtime.geometry_provider is None:
        return 0
    geom = runtime.geometry_provider.geometry
    value = _sample_grid_scalar(
        np.asarray(geom.nearest_boundary_part_id_map, dtype=np.float64),
        geom.axes,
        np.asarray(position, dtype=np.float64),
    )
    return int(max(0, round(value)))


def sample_geometry_normal(runtime, position: np.ndarray) -> np.ndarray:
    if runtime.geometry_provider is None:
        mins, maxs = runtime_bounds(runtime)
        pos = np.asarray(position, dtype=np.float64)
        dim = pos.size
        distance = np.minimum(pos - mins, maxs - pos)
        axis_index = int(np.argmin(distance))
        normal = np.zeros(dim, dtype=np.float64)
        normal[axis_index] = -1.0 if abs(pos[axis_index] - mins[axis_index]) < abs(maxs[axis_index] - pos[axis_index]) else 1.0
        return normal
    geom = runtime.geometry_provider.geometry
    values = [
        _sample_grid_scalar(
            np.asarray(component, dtype=np.float64),
            geom.axes,
            np.asarray(position, dtype=np.float64),
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


def points_inside_geometry_2d(
    runtime,
    positions: np.ndarray,
    on_boundary_tol_m: float = 1.0e-9,
    return_on_boundary: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('2D geometry queries require shape (n, 2)')
    mins, maxs = runtime_bounds(runtime)
    bbox = (
        (pts[:, 0] >= mins[0] - 1e-12)
        & (pts[:, 0] <= maxs[0] + 1e-12)
        & (pts[:, 1] >= mins[1] - 1e-12)
        & (pts[:, 1] <= maxs[1] + 1e-12)
    )
    inside = np.zeros(pts.shape[0], dtype=bool)
    on_boundary = np.zeros(pts.shape[0], dtype=bool)
    geometry_provider = runtime.geometry_provider
    if geometry_provider is not None and geometry_provider.geometry.boundary_loops_2d:
        if np.any(bbox):
            inside_bbox, on_boundary_bbox = points_inside_boundary_loops_2d_with_boundary(
                pts[bbox],
                geometry_provider.geometry.boundary_loops_2d,
                on_edge_tol=float(on_boundary_tol_m),
            )
            inside[bbox] = inside_bbox
            on_boundary[bbox] = on_boundary_bbox
        return (inside, on_boundary) if return_on_boundary else inside
    if np.any(bbox):
        inside[bbox] = np.asarray([sample_geometry_sdf(runtime, point) <= 0.0 for point in pts[bbox]], dtype=bool)
    return (inside, on_boundary) if return_on_boundary else inside


def inside_geometry_with_boundary(
    runtime,
    position: np.ndarray,
    *,
    on_boundary_tol_m: float = 1.0e-9,
    triangle_surface_3d: Optional[TriangleSurface3D] = None,
) -> Tuple[bool, bool]:
    mins, maxs = runtime_bounds(runtime)
    pos = np.asarray(position, dtype=np.float64)
    if np.any(pos < mins - 1e-12) or np.any(pos > maxs + 1e-12):
        return False, False
    geometry_provider = runtime.geometry_provider
    if geometry_provider is not None and int(geometry_provider.geometry.spatial_dim) == 2 and geometry_provider.geometry.boundary_loops_2d:
        inside, on_boundary = points_inside_geometry_2d(
            runtime,
            pos[None, :],
            on_boundary_tol_m=on_boundary_tol_m,
            return_on_boundary=True,
        )
        return bool(inside[0]), bool(on_boundary[0])
    if geometry_provider is not None and int(geometry_provider.geometry.spatial_dim) == 3 and triangle_surface_3d is not None:
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
    on_boundary_tol_m: float = 1.0e-9,
    triangle_surface_3d: Optional[TriangleSurface3D] = None,
) -> bool:
    inside, _ = inside_geometry_with_boundary(
        runtime,
        position,
        on_boundary_tol_m=on_boundary_tol_m,
        triangle_surface_3d=triangle_surface_3d,
    )
    return bool(inside)


__all__ = (
    'BoundaryHit',
    'BoundaryService',
    'inside_geometry',
    'inside_geometry_with_boundary',
    'points_inside_geometry_2d',
    'runtime_bounds',
    'sample_geometry_normal',
    'sample_geometry_part_id',
    'sample_geometry_sdf',
)
