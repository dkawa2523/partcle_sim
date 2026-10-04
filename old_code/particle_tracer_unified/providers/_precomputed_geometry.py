"""Build geometry providers from validated precomputed arrays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from particle_tracer_unified.core.coordinate_systems import (
    axisymmetric_rz_geometry_report,
)
from particle_tracer_unified.core.datamodel import GeometryND, GeometryProviderND
from particle_tracer_unified.core.geometry2d import (
    build_boundary_loops_2d,
    decode_boundary_loops_2d,
    validate_boundary_edges_2d,
)
from particle_tracer_unified.core.geometry3d import (
    unresolved_triangle_indices,
    validate_closed_surface_triangles,
)

from ._precomputed_common import (
    coordinate_scale,
    read_axes,
    read_metadata,
    resolve_path,
)


@dataclass(frozen=True)
class _GeometryGridData:
    axes: tuple[np.ndarray, ...]
    sdf: np.ndarray
    valid_mask: np.ndarray
    normal_components: tuple[np.ndarray, ...]
    part_id_map: np.ndarray


@dataclass(frozen=True)
class _GeometryBoundaryData:
    edges: np.ndarray | None
    edge_part_ids: np.ndarray | None
    loops_flat: np.ndarray | None
    loops_offsets: np.ndarray | None
    triangles: np.ndarray | None
    triangle_part_ids: np.ndarray | None
    containment_triangles: np.ndarray | None


@dataclass(frozen=True)
class _PrecomputedGeometryData:
    grid: _GeometryGridData
    boundary: _GeometryBoundaryData
    metadata: dict[str, Any]


def _read_geometry_normals(
    payload: Mapping[str, np.ndarray],
    spatial_dim: int,
    sdf: np.ndarray,
    axes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    normals = [
        np.asarray(payload[f"normal_{index}"], dtype=np.float64)
        for index in range(spatial_dim)
        if f"normal_{index}" in payload
    ]
    if len(normals) != spatial_dim:
        normals = [
            np.asarray(component, dtype=np.float64)
            for component in np.gradient(sdf, *axes, edge_order=1)
        ]
    return tuple(normals)


def _read_geometry_part_ids(
    payload: Mapping[str, np.ndarray], expected_shape: tuple[int, ...]
) -> np.ndarray:
    if "nearest_boundary_part_id_map" in payload:
        return np.asarray(payload["nearest_boundary_part_id_map"], dtype=np.int32)
    if "part_id_map" in payload:
        return np.asarray(payload["part_id_map"], dtype=np.int32)
    return np.ones(expected_shape, dtype=np.int32)


def _read_geometry_grid(
    payload: Mapping[str, np.ndarray],
    spatial_dim: int,
    coordinate_scale: float,
) -> _GeometryGridData:
    axes = read_axes(payload, spatial_dim, scale_to_si=coordinate_scale)
    sdf = np.asarray(payload["sdf"], dtype=np.float64) * coordinate_scale
    expected_shape = tuple(len(axis) for axis in axes)
    if sdf.shape != expected_shape:
        raise ValueError(
            f"Geometry sdf shape mismatch: expected {expected_shape}, got {sdf.shape}"
        )
    if not np.all(np.isfinite(sdf)):
        raise ValueError("Geometry sdf must contain only finite values")
    valid_mask = (
        np.asarray(payload["valid_mask"], dtype=bool)
        if "valid_mask" in payload
        else np.ones(expected_shape, dtype=bool)
    )
    if valid_mask.shape != expected_shape:
        raise ValueError(
            "Geometry valid_mask shape mismatch: "
            f"expected {expected_shape}, got {valid_mask.shape}"
        )
    return _GeometryGridData(
        axes,
        sdf,
        valid_mask,
        _read_geometry_normals(payload, spatial_dim, sdf, axes),
        _read_geometry_part_ids(payload, expected_shape),
    )


def _read_geometry_boundaries(
    payload: Mapping[str, np.ndarray], coordinate_scale: float
) -> _GeometryBoundaryData:
    edges = (
        np.asarray(payload["boundary_edges"], dtype=np.float64) * coordinate_scale
        if "boundary_edges" in payload
        else None
    )
    edge_part_ids = (
        np.asarray(payload["boundary_edge_part_ids"], dtype=np.int32)
        if "boundary_edge_part_ids" in payload
        else None
    )
    loops_flat = (
        np.asarray(payload["boundary_loops_2d_flat"], dtype=np.float64)
        * coordinate_scale
        if "boundary_loops_2d_flat" in payload
        else None
    )
    loops_offsets = (
        np.asarray(payload["boundary_loops_2d_offsets"], dtype=np.int32)
        if "boundary_loops_2d_offsets" in payload
        else None
    )
    triangles = (
        np.asarray(payload["boundary_triangles"], dtype=np.float64) * coordinate_scale
        if "boundary_triangles" in payload
        else None
    )
    triangle_part_ids = (
        np.asarray(payload["boundary_triangle_part_ids"], dtype=np.int32)
        if "boundary_triangle_part_ids" in payload
        else None
    )
    containment_triangles = (
        np.asarray(payload["containment_boundary_triangles"], dtype=np.float64)
        * coordinate_scale
        if "containment_boundary_triangles" in payload
        else None
    )
    return _GeometryBoundaryData(
        edges,
        edge_part_ids,
        loops_flat,
        loops_offsets,
        triangles,
        triangle_part_ids,
        containment_triangles,
    )


def _load_geometry(
    npz_path: Path, spatial_dim: int, scale: float
) -> _PrecomputedGeometryData:
    with np.load(npz_path, allow_pickle=False) as payload:
        grid = _read_geometry_grid(payload, spatial_dim, scale)
        boundary = _read_geometry_boundaries(payload, scale)
        metadata = read_metadata(payload)
    return _PrecomputedGeometryData(grid, boundary, metadata)


def _validate_collision_edges_2d(boundary: _GeometryBoundaryData) -> None:
    if boundary.edges is None:
        return
    edges = np.asarray(boundary.edges, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2) or edges.shape[0] == 0:
        raise ValueError("boundary_edges must be shaped as a non-empty (n, 2, 2) array")
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    if np.any(~np.isfinite(edges)) or np.any(~np.isfinite(lengths)):
        raise ValueError("boundary_edges must contain only finite coordinates")
    if np.any(lengths <= 0.0):
        raise ValueError("boundary_edges must contain only positive-length edges")
    if boundary.edge_part_ids is not None and boundary.edge_part_ids.shape != (
        edges.shape[0],
    ):
        raise ValueError(
            "boundary_edge_part_ids length mismatch: "
            f"expected {edges.shape[0]}, got {boundary.edge_part_ids.size}"
        )


def _containment_edges_from_loops(loops: tuple[np.ndarray, ...]) -> np.ndarray:
    edges: list[np.ndarray] = []
    for loop in loops:
        vertices = np.asarray(loop, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
            raise ValueError(
                "boundary_loops_2d must contain loops with at least three 2D vertices"
            )
        edges.append(np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1))
    return np.concatenate(edges, axis=0)


def _validated_collision_triangles_3d(
    boundary: _GeometryBoundaryData,
) -> np.ndarray:
    if boundary.triangles is None:
        raise ValueError("3D collision triangles are missing")
    triangles = np.asarray(boundary.triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(
            f"boundary_triangles must be shaped as (n, 3, 3), got {triangles.shape}"
        )
    if triangles.shape[0] == 0:
        raise ValueError("boundary_triangles must be non-empty")
    if boundary.triangle_part_ids is not None and (
        boundary.triangle_part_ids.shape != (triangles.shape[0],)
    ):
        raise ValueError(
            "boundary_triangle_part_ids length mismatch: "
            f"expected {triangles.shape[0]}, "
            f"got {boundary.triangle_part_ids.size}"
        )
    unresolved = unresolved_triangle_indices(triangles)
    if unresolved.size:
        raise ValueError(
            "boundary_triangles contains float64-unresolved triangle rows "
            f"{unresolved[:12].tolist()}"
        )
    return triangles


def _containment_surface_validation(
    boundary: _GeometryBoundaryData,
    collision_triangles: np.ndarray,
) -> dict[str, object]:
    containment = (
        collision_triangles
        if boundary.containment_triangles is None
        else boundary.containment_triangles
    )
    if containment.ndim != 3 or containment.shape[1:] != (3, 3):
        name = (
            "boundary_triangles"
            if boundary.containment_triangles is None
            else "containment_boundary_triangles"
        )
        raise ValueError(f"{name} must be shaped as (n, 3, 3), got {containment.shape}")
    try:
        validation = validate_closed_surface_triangles(containment)
    except ValueError as exc:
        if boundary.containment_triangles is None:
            raise
        raise ValueError(
            str(exc).replace("boundary_triangles", "containment_boundary_triangles")
        ) from exc
    return validation


def _add_surface_validation(
    metadata: dict[str, Any],
    boundary: _GeometryBoundaryData,
    spatial_dim: int,
) -> None:
    if spatial_dim != 3:
        return
    if boundary.containment_triangles is not None and boundary.triangles is None:
        raise ValueError("containment_boundary_triangles requires boundary_triangles")
    if boundary.triangles is None:
        return
    collision_triangles = _validated_collision_triangles_3d(boundary)
    metadata["boundary_surface_validation"] = _containment_surface_validation(
        boundary,
        collision_triangles,
    )


def _geometry_metadata_and_loops(
    data: _PrecomputedGeometryData,
    spatial_dim: int,
    coordinate_system: str,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
    boundary = data.boundary
    if spatial_dim == 2:
        _validate_collision_edges_2d(boundary)
    loops = decode_boundary_loops_2d(boundary.loops_flat, boundary.loops_offsets)
    if spatial_dim == 2 and not loops and boundary.edges is not None:
        loops = build_boundary_loops_2d(boundary.edges)
    metadata = dict(data.metadata)
    if spatial_dim == 2 and loops:
        containment_edges = _containment_edges_from_loops(loops)
        metadata.update(
            {
                "boundary_edge_topology": validate_boundary_edges_2d(containment_edges),
                "boundary_loop_count_2d": len(loops),
            }
        )
    axisymmetric_report = axisymmetric_rz_geometry_report(
        coordinate_system=coordinate_system,
        spatial_dim=spatial_dim,
        axes=cast(Sequence[Sequence[float]], data.grid.axes),
        boundary_edges=boundary.edges,
        boundary_edge_part_ids=boundary.edge_part_ids,
    )
    if axisymmetric_report:
        metadata["axisymmetric_rz"] = axisymmetric_report
    _add_surface_validation(metadata, boundary, spatial_dim)
    return metadata, loops


def _build_geometry(
    data: _PrecomputedGeometryData,
    metadata: Mapping[str, Any],
    loops: tuple[np.ndarray, ...],
    npz_path: Path,
    spatial_dim: int,
    coordinate_system: str,
    scale: float,
) -> GeometryND:
    grid = data.grid
    boundary = data.boundary
    return GeometryND(
        spatial_dim=int(spatial_dim),
        coordinate_system=str(coordinate_system),
        axes=grid.axes,
        valid_mask=grid.valid_mask,
        sdf=grid.sdf,
        normal_components=grid.normal_components,
        nearest_boundary_part_id_map=grid.part_id_map,
        source_kind=str(metadata.get("source_kind", "precomputed_npz")),
        metadata={
            "npz_path": str(npz_path),
            "provider_kind": "precomputed_npz",
            "coordinate_scale_to_si": float(scale),
            **metadata,
        },
        boundary_edges=boundary.edges,
        boundary_edge_part_ids=boundary.edge_part_ids,
        boundary_loops_2d=loops,
        boundary_triangles=boundary.triangles,
        boundary_triangle_part_ids=boundary.triangle_part_ids,
        containment_boundary_triangles=boundary.containment_triangles,
    )


def build_precomputed_geometry(
    cfg: Mapping[str, Any], spatial_dim: int, coordinate_system: str
) -> GeometryProviderND:
    npz_path = resolve_path(cfg)
    scale = coordinate_scale(cfg)
    data = _load_geometry(npz_path, spatial_dim, scale)
    metadata, loops = _geometry_metadata_and_loops(data, spatial_dim, coordinate_system)
    geometry = _build_geometry(
        data,
        metadata,
        loops,
        npz_path,
        spatial_dim,
        coordinate_system,
        scale,
    )
    return GeometryProviderND(
        geometry=geometry,
        kind=str(metadata.get("provider_kind", "precomputed_npz")),
    )
