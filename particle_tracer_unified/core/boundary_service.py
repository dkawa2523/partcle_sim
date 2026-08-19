from __future__ import annotations

import numpy as np

from particle_tracer_unified.domain import BoundaryHit

from ._boundary_hits_3d import nearest_hit_on_boundary_triangles_ignoring_parts
from .boundary_core import (
    BoundaryBroadPhase2D,
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
    contact_frame_on_boundary_edge_2d,
    nearest_boundary_edge_features_2d,
    nearest_hit_on_boundary_edges,
    polyline_hit_from_boundary_edges,
    polyline_hit_from_boundary_triangles,
    polyline_hits_from_boundary_edges_batch,
    segment_hit_from_boundary_edges,
    segment_hit_from_boundary_triangles,
)
from .catalogs import is_internal_pass_through
from .geometry3d import TriangleSurface3D


def _readonly_float64_array(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _geometry_grid_spacing_2d(geometry) -> float:
    if geometry is None or int(getattr(geometry, "spatial_dim", 0)) != 2:
        return 0.0
    spacings: list[float] = []
    for axis in getattr(geometry, "axes", ()):
        differences = np.diff(np.asarray(axis, dtype=np.float64))
        finite = differences[np.isfinite(differences) & (differences > 0.0)]
        if finite.size:
            spacings.append(float(np.min(finite)))
    return float(min(spacings)) if spacings else 0.0


def _padded_boundary_edge_bounds_2d(
    geometry,
    padding_m: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    edges_raw = None if geometry is None else getattr(geometry, "boundary_edges", None)
    if edges_raw is None:
        return None, None
    edges = np.asarray(edges_raw, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2) or edges.shape[0] == 0:
        return None, None
    return (
        _readonly_float64_array(np.min(edges, axis=1) - padding_m),
        _readonly_float64_array(np.max(edges, axis=1) + padding_m),
    )


def _build_broad_phase_2d(
    runtime,
    *,
    on_boundary_tol_m: float,
) -> BoundaryBroadPhase2D:
    geometry_provider = getattr(runtime, "geometry_provider", None)
    geometry = None if geometry_provider is None else geometry_provider.geometry
    grid_spacing_m = _geometry_grid_spacing_2d(geometry)
    padding_m = float(on_boundary_tol_m)
    edge_min, edge_max = _padded_boundary_edge_bounds_2d(geometry, padding_m)
    return BoundaryBroadPhase2D(
        grid_spacing_m=float(grid_spacing_m),
        edge_aabb_min_padded=edge_min,
        edge_aabb_max_padded=edge_max,
        edge_aabb_padding_m=float(padding_m),
    )


def _build_boundary_service_2d(runtime, *, on_boundary_tol_m: float) -> BoundaryService:
    def inside(position: np.ndarray) -> bool:
        return inside_geometry(
            runtime,
            position,
            on_boundary_tol_m=on_boundary_tol_m,
        )

    def inside_strict(position: np.ndarray) -> bool:
        return inside_geometry(runtime, position, on_boundary_tol_m=0.0)

    return BoundaryService(
        inside=inside,
        inside_strict=inside_strict,
        # A segment that starts on a wall within the resolved geometry
        # tolerance and never crosses to the other side is a departure, not a
        # hit.  This is what lets a release sit on its own boundary the way a
        # COMSOL inlet does, and it applies equally to a restart after a
        # reflection.  A particle arriving from the interior starts further
        # than the tolerance from the wall, so a real crossing still registers.
        segment_hit=lambda p0, p1: segment_hit_from_boundary_edges(
            runtime,
            p0,
            p1,
            coordinate_tolerance_m=float(on_boundary_tol_m),
            departure_tolerance_m=float(on_boundary_tol_m),
        ),
        polyline_hit=lambda p0, stage_pts: polyline_hit_from_boundary_edges(
            runtime,
            p0,
            stage_pts,
            coordinate_tolerance_m=float(on_boundary_tol_m),
            departure_tolerance_m=float(on_boundary_tol_m),
        ),
        nearest_projection=lambda point, inside_ref: nearest_hit_on_boundary_edges(
            runtime, point, inside_ref
        ),
        primary_hit_counter_key="edge_hit_count",
        triangle_surface_3d=None,
        broad_phase_2d=_build_broad_phase_2d(
            runtime,
            on_boundary_tol_m=float(on_boundary_tol_m),
        ),
    )


def _build_boundary_service_3d(
    runtime,
    *,
    on_boundary_tol_m: float,
    triangle_surface_3d: TriangleSurface3D | None,
    containment_triangle_surface_3d: TriangleSurface3D | None,
) -> BoundaryService:
    wall_catalog = getattr(runtime, "wall_catalog", None)
    transparent_parts = frozenset(
        int(model.part_id)
        for model in getattr(wall_catalog, "part_models", ())
        if is_internal_pass_through(model)
    )
    containment_surface = (
        triangle_surface_3d
        if containment_triangle_surface_3d is None
        else containment_triangle_surface_3d
    )

    def inside(position: np.ndarray) -> bool:
        return inside_geometry(
            runtime,
            position,
            on_boundary_tol_m=on_boundary_tol_m,
            triangle_surface_3d=containment_surface,
        )

    def inside_strict(position: np.ndarray) -> bool:
        return inside_geometry(
            runtime,
            position,
            on_boundary_tol_m=0.0,
            triangle_surface_3d=containment_surface,
        )

    return BoundaryService(
        inside=inside,
        inside_strict=inside_strict,
        segment_hit=lambda p0, p1: segment_hit_from_boundary_triangles(
            triangle_surface_3d,
            p0,
            p1,
            coordinate_tolerance_m=float(on_boundary_tol_m),
            ignored_part_ids=transparent_parts,
        ),
        polyline_hit=lambda p0, stage_pts: polyline_hit_from_boundary_triangles(
            triangle_surface_3d,
            p0,
            stage_pts,
            coordinate_tolerance_m=float(on_boundary_tol_m),
            ignored_part_ids=transparent_parts,
        ),
        nearest_projection=lambda point, inside_ref: (
            nearest_hit_on_boundary_triangles_ignoring_parts(
                triangle_surface_3d,
                point,
                inside_ref,
                ignored_part_ids=transparent_parts,
            )
        ),
        primary_hit_counter_key="triangle_hit_count",
        triangle_surface_3d=triangle_surface_3d,
        broad_phase_2d=None,
    )


def build_boundary_service(
    runtime,
    *,
    spatial_dim: int,
    on_boundary_tol_m: float,
    triangle_surface_3d: TriangleSurface3D | None,
    containment_triangle_surface_3d: TriangleSurface3D | None = None,
) -> BoundaryService:
    if int(spatial_dim) == 2:
        return _build_boundary_service_2d(runtime, on_boundary_tol_m=on_boundary_tol_m)
    return _build_boundary_service_3d(
        runtime,
        on_boundary_tol_m=on_boundary_tol_m,
        triangle_surface_3d=triangle_surface_3d,
        containment_triangle_surface_3d=containment_triangle_surface_3d,
    )


__all__ = (
    "BoundaryHit",
    "build_boundary_service",
    "contact_frame_on_boundary_edge_2d",
    "inside_geometry",
    "inside_geometry_with_boundary",
    "nearest_boundary_edge_features_2d",
    "points_inside_geometry_2d",
    "polyline_hits_from_boundary_edges_batch",
    "runtime_bounds",
    "sample_geometry_normal",
    "sample_geometry_part_id",
    "sample_geometry_sdf",
)
