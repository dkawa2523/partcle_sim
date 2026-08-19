"""Triangle surface data and uniform-grid candidate indexing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._triangle_topology import (
    _triangle_normals,
    validate_closed_surface_triangles,
)


@dataclass(frozen=True)
class TriangleUniformGrid:
    origin: np.ndarray
    cell_size: np.ndarray
    dims: tuple[int, int, int]
    cell_to_triangles: dict[tuple[int, int, int], np.ndarray]
    triangle_mins: np.ndarray
    triangle_maxs: np.ndarray
    triangle_count: int


@dataclass(frozen=True)
class TriangleSurface3D:
    triangles: np.ndarray
    part_ids: np.ndarray
    normals: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    grid: TriangleUniformGrid


@dataclass(frozen=True)
class GeometrySurfaces3D:
    """Collision primitives paired with the closed containment shell."""

    collision: TriangleSurface3D
    containment: TriangleSurface3D


def _validated_triangle_grid_bounds(
    triangles: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"triangles must be shaped as (n, 3, 3), got {tri.shape}")
    triangle_count = int(tri.shape[0])
    if triangle_count <= 0:
        raise ValueError("triangles must be non-empty")
    triangle_mins = np.min(tri, axis=1)
    triangle_maxs = np.max(tri, axis=1)
    bbox_min = np.min(triangle_mins, axis=0)
    bbox_max = np.max(triangle_maxs, axis=0)
    span = bbox_max - bbox_min
    if np.any(~np.isfinite(span)) or np.any(span <= 0.0):
        raise ValueError(
            "boundary triangle surface must have positive finite span on every axis"
        )
    return triangle_count, triangle_mins, triangle_maxs, bbox_min, span


def _triangle_grid_dimensions(
    triangle_count: int,
    span: np.ndarray,
    *,
    target_triangles_per_cell: int,
    min_cells_per_axis: int,
    max_cells_per_axis: int,
) -> tuple[tuple[int, int, int], np.ndarray]:
    geometric_mean = float(np.exp(np.mean(np.log(span))))
    base = max(
        1.0,
        (float(triangle_count) / max(float(target_triangles_per_cell), 1.0))
        ** (1.0 / 3.0),
    )
    raw_dims = np.round(base * (span / geometric_mean)).astype(np.int32)
    raw_dims = np.clip(
        raw_dims,
        int(max(1, min_cells_per_axis)),
        int(max(max_cells_per_axis, min_cells_per_axis)),
    )
    dims = (int(raw_dims[0]), int(raw_dims[1]), int(raw_dims[2]))
    cell_size = span / np.maximum(np.asarray(dims, dtype=np.float64), 1.0)
    return dims, cell_size


def _triangle_grid_cell_map(
    triangle_mins: np.ndarray,
    triangle_maxs: np.ndarray,
    bbox_min: np.ndarray,
    cell_size: np.ndarray,
    dims: tuple[int, int, int],
) -> dict[tuple[int, int, int], np.ndarray]:
    triangle_count = int(triangle_mins.shape[0])
    upper_cell = np.asarray(dims, dtype=np.int64) - 1
    cell_to_triangles: dict[tuple[int, int, int], list[int]] = {}
    for index in range(triangle_count):
        low = np.floor((triangle_mins[index] - bbox_min) / cell_size).astype(np.int64)
        high = np.floor((triangle_maxs[index] - bbox_min) / cell_size).astype(np.int64)
        low = np.clip(low, 0, upper_cell)
        high = np.clip(high, 0, upper_cell)
        for ix in range(int(low[0]), int(high[0]) + 1):
            for iy in range(int(low[1]), int(high[1]) + 1):
                for iz in range(int(low[2]), int(high[2]) + 1):
                    key = (ix, iy, iz)
                    bucket = cell_to_triangles.get(key)
                    if bucket is None:
                        cell_to_triangles[key] = [index]
                    else:
                        bucket.append(index)
    return {
        key: np.asarray(sorted(set(indices)), dtype=np.int32)
        for key, indices in cell_to_triangles.items()
    }


def build_triangle_uniform_grid(
    triangles: np.ndarray,
    *,
    target_triangles_per_cell: int = 24,
    min_cells_per_axis: int = 4,
    max_cells_per_axis: int = 64,
) -> TriangleUniformGrid:
    (
        triangle_count,
        triangle_mins,
        triangle_maxs,
        bbox_min,
        span,
    ) = _validated_triangle_grid_bounds(triangles)
    dims, cell_size = _triangle_grid_dimensions(
        triangle_count,
        span,
        target_triangles_per_cell=target_triangles_per_cell,
        min_cells_per_axis=min_cells_per_axis,
        max_cells_per_axis=max_cells_per_axis,
    )
    cell_to_triangles = _triangle_grid_cell_map(
        triangle_mins,
        triangle_maxs,
        bbox_min,
        cell_size,
        dims,
    )

    return TriangleUniformGrid(
        origin=np.asarray(bbox_min, dtype=np.float64),
        cell_size=np.asarray(cell_size, dtype=np.float64),
        dims=dims,
        cell_to_triangles=cell_to_triangles,
        triangle_mins=np.asarray(triangle_mins, dtype=np.float64),
        triangle_maxs=np.asarray(triangle_maxs, dtype=np.float64),
        triangle_count=triangle_count,
    )


def build_triangle_surface(
    triangles: np.ndarray,
    part_ids: np.ndarray | None = None,
    *,
    validate_closed: bool = True,
) -> TriangleSurface3D:
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(
            f"boundary_triangles must be shaped as (n, 3, 3), got {tri.shape}"
        )
    if validate_closed:
        validate_closed_surface_triangles(tri)
    if part_ids is None:
        pid = np.zeros(tri.shape[0], dtype=np.int32)
    else:
        pid = np.asarray(part_ids, dtype=np.int32).reshape(-1)
        if pid.shape[0] != tri.shape[0]:
            raise ValueError(
                "boundary_triangle_part_ids length mismatch: "
                f"expected {tri.shape[0]}, got {pid.shape[0]}"
            )
    normals = _triangle_normals(tri)
    bbox_min = np.min(tri.reshape(-1, 3), axis=0)
    bbox_max = np.max(tri.reshape(-1, 3), axis=0)
    return TriangleSurface3D(
        triangles=tri,
        part_ids=pid,
        normals=normals,
        bbox_min=np.asarray(bbox_min, dtype=np.float64),
        bbox_max=np.asarray(bbox_max, dtype=np.float64),
        grid=build_triangle_uniform_grid(tri),
    )


def build_geometry_surfaces_3d(geometry: object) -> GeometrySurfaces3D:
    """Build 3D solver surfaces from one geometry artifact contract.

    Legacy artifacts use their closed collision surface for both roles.  A
    separate containment array permits open internal collision interfaces
    without changing point-in-domain classification.
    """

    collision_values = getattr(geometry, "boundary_triangles", None)
    if collision_values is None:
        raise ValueError(
            "3D geometry requires boundary_triangles as collision geometry"
        )
    collision_triangles = np.asarray(collision_values, dtype=np.float64)
    part_values = getattr(geometry, "boundary_triangle_part_ids", None)
    part_ids = (
        np.zeros(collision_triangles.shape[0], dtype=np.int32)
        if part_values is None
        else np.asarray(part_values, dtype=np.int32)
    )
    containment_values = getattr(geometry, "containment_boundary_triangles", None)
    if containment_values is None:
        surface = build_triangle_surface(
            collision_triangles,
            part_ids,
            validate_closed=True,
        )
        return GeometrySurfaces3D(collision=surface, containment=surface)

    containment = build_triangle_surface(
        np.asarray(containment_values, dtype=np.float64),
        validate_closed=True,
    )
    collision = build_triangle_surface(
        collision_triangles,
        part_ids,
        validate_closed=False,
    )
    return GeometrySurfaces3D(collision=collision, containment=containment)


def query_triangle_candidates(
    grid: TriangleUniformGrid,
    p0: np.ndarray,
    p1: np.ndarray,
) -> np.ndarray:
    start = np.asarray(p0, dtype=np.float64)
    end = np.asarray(p1, dtype=np.float64)
    segment_min = np.minimum(start, end)
    segment_max = np.maximum(start, end)
    dims = np.asarray(grid.dims, dtype=np.int64)
    index_min = np.floor((segment_min - grid.origin) / grid.cell_size).astype(np.int64)
    index_max = np.floor((segment_max - grid.origin) / grid.cell_size).astype(np.int64)
    index_min = np.clip(index_min, 0, dims - 1)
    index_max = np.clip(index_max, 0, dims - 1)
    span_cells = index_max - index_min + 1
    if int(span_cells[0] * span_cells[1] * span_cells[2]) > 4096:
        return np.arange(grid.triangle_count, dtype=np.int32)
    ids: set[int] = set()
    for ix in range(int(index_min[0]), int(index_max[0]) + 1):
        for iy in range(int(index_min[1]), int(index_max[1]) + 1):
            for iz in range(int(index_min[2]), int(index_max[2]) + 1):
                candidates = grid.cell_to_triangles.get((ix, iy, iz))
                if candidates is not None:
                    ids.update(int(value) for value in candidates.tolist())
    if not ids:
        return np.arange(grid.triangle_count, dtype=np.int32)
    return np.asarray(sorted(ids), dtype=np.int32)
