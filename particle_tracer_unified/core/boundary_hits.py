"""Stable boundary-hit API backed by dimension-specific query owners."""

from __future__ import annotations

from ._boundary_contact_2d import BoundaryEdgeFrame2D as BoundaryEdgeFrame2D
from ._boundary_contact_2d import (
    contact_frame_on_boundary_edge_2d as contact_frame_on_boundary_edge_2d,
)
from ._boundary_contact_2d import nearest_hit_on_boundary_edges
from ._boundary_hits_2d import (
    nearest_boundary_edge_features_2d,
    polyline_hits_from_boundary_edges_batch,
    segment_hit_from_boundary_edges,
)
from ._boundary_hits_3d import (
    nearest_hit_on_boundary_triangles,
    normalize_polyline_alpha,
    polyline_hit_from_boundary_edges,
    polyline_hit_from_boundary_triangles,
    segment_hit_from_boundary_triangles,
)

__all__ = (
    "nearest_boundary_edge_features_2d",
    "nearest_hit_on_boundary_edges",
    "nearest_hit_on_boundary_triangles",
    "normalize_polyline_alpha",
    "polyline_hit_from_boundary_edges",
    "polyline_hit_from_boundary_triangles",
    "polyline_hits_from_boundary_edges_batch",
    "segment_hit_from_boundary_edges",
    "segment_hit_from_boundary_triangles",
)
