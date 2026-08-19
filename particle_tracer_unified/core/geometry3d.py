"""Stable entry point for 3D triangle-surface geometry."""

from ._triangle_queries import (
    nearest_surface_point,
    point_inside_surface,
    segment_hit_from_surface,
)
from ._triangle_surface import (
    TriangleSurface3D,
    TriangleUniformGrid,
    build_triangle_surface,
    build_triangle_uniform_grid,
    query_triangle_candidates,
)
from ._triangle_topology import (
    point_triangle_barycentric,
    unresolved_triangle_indices,
    validate_closed_surface_triangles,
)

__all__ = (
    "TriangleSurface3D",
    "TriangleUniformGrid",
    "build_triangle_surface",
    "build_triangle_uniform_grid",
    "nearest_surface_point",
    "point_inside_surface",
    "point_triangle_barycentric",
    "query_triangle_candidates",
    "segment_hit_from_surface",
    "unresolved_triangle_indices",
    "validate_closed_surface_triangles",
)
