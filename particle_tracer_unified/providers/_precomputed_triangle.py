"""Build triangle-mesh field providers from precomputed arrays."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    QuantitySeriesND,
    TriangleMeshField2D,
)
from particle_tracer_unified.core.geometry3d import unresolved_triangle_indices
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    build_triangle_candidate_grid,
    triangle_mesh_support_tolerance,
)

from ._precomputed_common import (
    coordinate_scale,
    infer_unit,
    quantity_mapping,
    quantity_metadata,
    quantity_sources,
    read_metadata,
    read_times,
    real_quantity_values,
    resolve_path,
)

_TRIANGLE_RESERVED_ARRAYS = {
    "mesh_vertices",
    "mesh_triangles",
    "times",
    "support_phi",
    "metadata_json",
}


@dataclass(frozen=True)
class _PrecomputedTriangleFieldData:
    vertices: np.ndarray
    triangles: np.ndarray
    times: np.ndarray
    quantities: dict[str, QuantitySeriesND]
    metadata: dict[str, Any]


def _validate_mesh_shapes(
    mesh_vertices: np.ndarray, mesh_triangles: np.ndarray
) -> None:
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 2:
        raise ValueError("mesh_vertices must have shape (n, 2)")
    if mesh_triangles.ndim != 2 or mesh_triangles.shape[1] != 3:
        raise ValueError("mesh_triangles must have shape (m, 3)")
    if mesh_vertices.shape[0] < 3:
        raise ValueError("mesh_vertices must contain at least three vertices")
    if mesh_triangles.shape[0] == 0:
        raise ValueError("mesh_triangles must contain at least one triangle")


def _validate_mesh_indices(
    mesh_vertices: np.ndarray, mesh_triangles: np.ndarray
) -> None:
    if int(np.min(mesh_triangles)) < 0 or int(np.max(mesh_triangles)) >= int(
        mesh_vertices.shape[0]
    ):
        raise ValueError("mesh_triangles contains vertex indices outside mesh_vertices")


def _validate_mesh(mesh_vertices: np.ndarray, mesh_triangles: np.ndarray) -> None:
    _validate_mesh_shapes(mesh_vertices, mesh_triangles)
    if not np.all(np.isfinite(mesh_vertices)):
        raise ValueError("mesh_vertices must contain only finite values")
    _validate_mesh_indices(mesh_vertices, mesh_triangles)
    unresolved = unresolved_triangle_indices(mesh_vertices[mesh_triangles])
    if unresolved.size:
        raise ValueError(
            "mesh_triangles contains float64-unresolved triangle rows "
            f"{unresolved[:12].tolist()}"
        )


def _quantity_data(
    payload: Mapping[str, np.ndarray],
    source: str,
    item: Mapping[str, Any],
    times: np.ndarray,
    expected_vertex_count: int,
) -> np.ndarray | None:
    data = real_quantity_values(payload, source) * float(item.get("scale_to_si", 1.0))
    if data.ndim == 1:
        if data.shape[0] != expected_vertex_count:
            raise ValueError(
                f"Mesh quantity {source} vertex axis mismatch: "
                f"expected {expected_vertex_count}, got {data.shape[0]}"
            )
        return data
    if data.ndim != 2:
        return None
    expected_shape = (times.size, expected_vertex_count)
    if data.shape != expected_shape:
        raise ValueError(
            f"Mesh quantity {source} shape mismatch: "
            f"expected {expected_shape}, got {data.shape}"
        )
    return data


def _read_quantities(
    payload: Mapping[str, np.ndarray],
    mapping: Mapping[str, Mapping[str, Any]],
    times: np.ndarray,
    expected_vertex_count: int,
    npz_path: Path,
) -> dict[str, QuantitySeriesND]:
    quantities: dict[str, QuantitySeriesND] = {}
    for target, source, item in quantity_sources(
        payload, mapping, _TRIANGLE_RESERVED_ARRAYS
    ):
        if source not in payload:
            raise ValueError(
                f"Manifest field component {source!r} is missing from {npz_path}"
            )
        data = _quantity_data(payload, source, item, times, expected_vertex_count)
        if data is None:
            continue
        if data.size and not np.all(np.isfinite(data)):
            raise ValueError(f"Mesh quantity {target} contains non-finite values")
        quantities[target] = QuantitySeriesND(
            name=target,
            unit=str(item.get("unit", infer_unit(target))),
            times=times,
            data=data,
            metadata=quantity_metadata(source, item),
        )
    return quantities


def _load_triangle_field(
    npz_path: Path,
    scale: float,
    mapping: Mapping[str, Mapping[str, Any]],
) -> _PrecomputedTriangleFieldData:
    with np.load(npz_path, allow_pickle=False) as payload:
        if "mesh_vertices" not in payload or "mesh_triangles" not in payload:
            raise ValueError(
                "Mesh field npz must include mesh_vertices and mesh_triangles: "
                f"{npz_path}"
            )
        vertices = np.asarray(payload["mesh_vertices"], dtype=np.float64) * scale
        triangles_raw = np.asarray(payload["mesh_triangles"])
        if not np.issubdtype(triangles_raw.dtype, np.integer):
            raise ValueError("mesh_triangles must use integer vertex indices")
        triangles = triangles_raw.astype(np.int32, copy=False)
        _validate_mesh(vertices, triangles)
        times = read_times(payload)
        metadata = read_metadata(payload)
        if "support_tolerance_m" in metadata:
            raise ValueError(
                "triangle field metadata.support_tolerance_m is obsolete; "
                "support tolerance is derived from mesh resolution and float64 roundoff"
            )
        quantities = _read_quantities(
            payload,
            mapping,
            times,
            int(vertices.shape[0]),
            npz_path,
        )
    return _PrecomputedTriangleFieldData(
        vertices, triangles, times, quantities, metadata
    )


def _build_triangle_mesh_field(
    data: _PrecomputedTriangleFieldData,
    npz_path: Path,
    coordinate_system: str,
    scale: float,
    mapping: Mapping[str, Mapping[str, Any]],
) -> TriangleMeshField2D:
    origin, cell_size, shape, offsets, triangle_indices = build_triangle_candidate_grid(
        data.vertices, data.triangles
    )
    time_mode = (
        "transient"
        if any(
            np.asarray(quantity.data).ndim == 2 and data.times.size > 1
            for quantity in data.quantities.values()
        )
        else "steady"
    )
    support_tolerance_m = triangle_mesh_support_tolerance(data.vertices, data.triangles)
    return TriangleMeshField2D(
        spatial_dim=2,
        coordinate_system=str(coordinate_system),
        mesh_vertices=data.vertices,
        mesh_triangles=data.triangles,
        quantities=data.quantities,
        accel_origin=np.asarray(origin, dtype=np.float64),
        accel_cell_size=np.asarray(cell_size, dtype=np.float64),
        accel_shape=(int(shape[0]), int(shape[1])),
        accel_cell_offsets=np.asarray(offsets, dtype=np.int32),
        accel_triangle_indices=np.asarray(triangle_indices, dtype=np.int32),
        time_mode=time_mode,
        metadata={
            "npz_path": str(npz_path),
            "provider_kind": "precomputed_triangle_mesh_npz",
            "field_backend_kind": "triangle_mesh_2d",
            "coordinate_scale_to_si": float(scale),
            "manifest_quantity_mapping": mapping,
            **data.metadata,
            "support_tolerance_m": float(support_tolerance_m),
        },
    )


def build_precomputed_triangle_mesh_field(
    cfg: Mapping[str, Any],
    spatial_dim: int,
    coordinate_system: str,
) -> FieldProviderND:
    if int(spatial_dim) != 2:
        raise ValueError(
            "precomputed_triangle_mesh_npz currently supports only spatial_dim=2"
        )
    npz_path = resolve_path(cfg)
    scale = coordinate_scale(cfg)
    mapping = quantity_mapping(cfg)
    data = _load_triangle_field(npz_path, scale, mapping)
    if not data.quantities:
        raise ValueError(f"No mesh field quantities found in {npz_path}")
    field = _build_triangle_mesh_field(
        data,
        npz_path,
        coordinate_system,
        scale,
        mapping,
    )
    return FieldProviderND(field=field, kind="precomputed_triangle_mesh_npz")
