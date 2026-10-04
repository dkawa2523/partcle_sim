from __future__ import annotations

import inspect

from particle_tracer_unified.core import (
    _triangle_queries,
    _triangle_surface,
    _triangle_topology,
    geometry3d,
)


def test_geometry3d_facade_directly_exports_owner_objects() -> None:
    owners = {
        "TriangleSurface3D": _triangle_surface,
        "TriangleUniformGrid": _triangle_surface,
        "build_triangle_surface": _triangle_surface,
        "build_triangle_uniform_grid": _triangle_surface,
        "nearest_surface_point": _triangle_queries,
        "point_inside_surface": _triangle_queries,
        "point_triangle_barycentric": _triangle_topology,
        "query_triangle_candidates": _triangle_surface,
        "segment_hit_from_surface": _triangle_queries,
        "unresolved_triangle_indices": _triangle_topology,
        "validate_closed_surface_triangles": _triangle_topology,
    }

    assert geometry3d.__all__ == tuple(owners)
    for name, owner in owners.items():
        assert getattr(geometry3d, name) is getattr(owner, name)


def test_geometry3d_public_signatures_remain_stable() -> None:
    signatures = {
        "TriangleUniformGrid": (
            "(origin: 'np.ndarray', cell_size: 'np.ndarray', "
            "dims: 'tuple[int, int, int]', "
            "cell_to_triangles: 'dict[tuple[int, int, int], np.ndarray]', "
            "triangle_mins: 'np.ndarray', triangle_maxs: 'np.ndarray', "
            "triangle_count: 'int') -> None"
        ),
        "TriangleSurface3D": (
            "(triangles: 'np.ndarray', part_ids: 'np.ndarray', "
            "normals: 'np.ndarray', bbox_min: 'np.ndarray', "
            "bbox_max: 'np.ndarray', grid: 'TriangleUniformGrid') -> None"
        ),
        "unresolved_triangle_indices": "(triangles: 'np.ndarray') -> 'np.ndarray'",
        "point_triangle_barycentric": (
            "(point: 'np.ndarray', triangle: 'np.ndarray') -> 'np.ndarray | None'"
        ),
        "validate_closed_surface_triangles": (
            "(triangles: 'np.ndarray') -> 'dict[str, object]'"
        ),
        "build_triangle_uniform_grid": (
            "(triangles: 'np.ndarray', *, target_triangles_per_cell: 'int' = 24, "
            "min_cells_per_axis: 'int' = 4, max_cells_per_axis: 'int' = 64) "
            "-> 'TriangleUniformGrid'"
        ),
        "build_triangle_surface": (
            "(triangles: 'np.ndarray', part_ids: 'np.ndarray | None' = None, *, "
            "validate_closed: 'bool' = True) -> 'TriangleSurface3D'"
        ),
        "query_triangle_candidates": (
            "(grid: 'TriangleUniformGrid', p0: 'np.ndarray', p1: 'np.ndarray') "
            "-> 'np.ndarray'"
        ),
        "segment_hit_from_surface": (
            "(surface: 'TriangleSurface3D', p0: 'np.ndarray', p1: 'np.ndarray', "
            "*, alpha_min: 'float' = 1e-08, coordinate_tolerance_m: 'float' = "
            "0.0) -> 'tuple[np.ndarray, np.ndarray, float, int, int] | None'"
        ),
        "nearest_surface_point": (
            "(surface: 'TriangleSurface3D', point: 'np.ndarray', *, "
            "inside_reference: 'np.ndarray | None' = None) -> "
            "'tuple[np.ndarray, np.ndarray, int, int]'"
        ),
        "point_inside_surface": (
            "(surface: 'TriangleSurface3D', point: 'np.ndarray', *, "
            "on_boundary_tol: 'float') -> 'tuple[bool, bool]'"
        ),
    }

    assert {
        name: str(inspect.signature(getattr(geometry3d, name))) for name in signatures
    } == signatures
