from __future__ import annotations

from typing import Optional

from .boundary_core import (
    BoundaryHit,
    BoundaryService,
    inside_geometry,
    inside_geometry_with_boundary,
    points_inside_geometry_2d,
    runtime_bounds,
    sample_geometry_normal,
    sample_geometry_part_id,
    sample_geometry_sdf,
)
from .boundary_hits import (
    nearest_hit_on_boundary_edges,
    nearest_hit_on_boundary_triangles,
    normalize_polyline_alpha,
    polyline_hit_from_boundary_edges,
    polyline_hit_from_boundary_triangles,
    polyline_hit_from_loop_bisection,
    polyline_hit_from_solid_bisection_3d,
    segment_hit_from_boundary_edges,
    segment_hit_from_boundary_triangles,
    segment_hit_from_loop_bisection,
    segment_hit_from_solid_bisection_3d,
)
from .geometry3d import TriangleSurface3D


def _build_boundary_service_2d(runtime, *, on_boundary_tol_m: float) -> BoundaryService:
    return BoundaryService(
        inside=lambda pos: inside_geometry(runtime, pos, on_boundary_tol_m=on_boundary_tol_m),
        inside_strict=lambda pos: inside_geometry(runtime, pos, on_boundary_tol_m=0.0),
        segment_hit=lambda p0, p1: segment_hit_from_boundary_edges(runtime, p0, p1),
        polyline_hit=lambda p0, stage_pts: polyline_hit_from_boundary_edges(runtime, p0, stage_pts),
        nearest_projection=lambda point, inside_ref: nearest_hit_on_boundary_edges(runtime, point, inside_ref),
        primary_hit_counter_key='edge_hit_count',
        triangle_surface_3d=None,
    )


def _build_boundary_service_3d(
    runtime,
    *,
    on_boundary_tol_m: float,
    triangle_surface_3d: Optional[TriangleSurface3D],
) -> BoundaryService:
    return BoundaryService(
        inside=lambda pos: inside_geometry(
            runtime,
            pos,
            on_boundary_tol_m=on_boundary_tol_m,
            triangle_surface_3d=triangle_surface_3d,
        ),
        inside_strict=lambda pos: inside_geometry(
            runtime,
            pos,
            on_boundary_tol_m=0.0,
            triangle_surface_3d=triangle_surface_3d,
        ),
        segment_hit=lambda p0, p1: segment_hit_from_boundary_triangles(triangle_surface_3d, p0, p1),
        polyline_hit=lambda p0, stage_pts: polyline_hit_from_boundary_triangles(triangle_surface_3d, p0, stage_pts),
        nearest_projection=lambda point, inside_ref: nearest_hit_on_boundary_triangles(triangle_surface_3d, point, inside_ref),
        primary_hit_counter_key='triangle_hit_count',
        triangle_surface_3d=triangle_surface_3d,
    )


def build_boundary_service(
    runtime,
    *,
    spatial_dim: int,
    on_boundary_tol_m: float,
    triangle_surface_3d: Optional[TriangleSurface3D],
) -> BoundaryService:
    if int(spatial_dim) == 2:
        return _build_boundary_service_2d(runtime, on_boundary_tol_m=on_boundary_tol_m)
    return _build_boundary_service_3d(
        runtime,
        on_boundary_tol_m=on_boundary_tol_m,
        triangle_surface_3d=triangle_surface_3d,
    )


__all__ = (
    'BoundaryHit',
    'BoundaryService',
    'build_boundary_service',
    'inside_geometry',
    'inside_geometry_with_boundary',
    'nearest_hit_on_boundary_edges',
    'nearest_hit_on_boundary_triangles',
    'normalize_polyline_alpha',
    'points_inside_geometry_2d',
    'polyline_hit_from_boundary_edges',
    'polyline_hit_from_boundary_triangles',
    'polyline_hit_from_loop_bisection',
    'polyline_hit_from_solid_bisection_3d',
    'runtime_bounds',
    'sample_geometry_normal',
    'sample_geometry_part_id',
    'sample_geometry_sdf',
    'segment_hit_from_boundary_edges',
    'segment_hit_from_boundary_triangles',
    'segment_hit_from_loop_bisection',
    'segment_hit_from_solid_bisection_3d',
)
