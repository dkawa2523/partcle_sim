"""Build a triangle-mesh field artifact from COMSOL mesh-node samples.

Keeping the solution on the mesh COMSOL solved it on preserves boundary-layer
refinement and ends support where the mesh ends, which a resampled lattice
cannot do.  This module owns only that conversion; the mesh topology,
manifest, and sampling live in their own modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._mesh_parsing import ParsedMesh
from ._mesh_topology import surface_triangles_from_mesh

NODE_INDEX_COLUMN = "node_index"
_RESERVED_SAMPLE_COLUMNS = frozenset({NODE_INDEX_COLUMN, "r", "z", "x", "y"})


@dataclass(frozen=True)
class PackedMeshField:
    """One written triangle-mesh field artifact and its build summary."""

    path: Path
    summary: dict[str, Any]


def _node_sample_frame(samples_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(samples_csv)
    if NODE_INDEX_COLUMN not in frame.columns:
        raise ValueError(
            f"COMSOL node samples must declare a {NODE_INDEX_COLUMN!r} column; "
            "a mesh-native field cannot be built from a coordinate grid table"
        )
    node_index = pd.to_numeric(frame[NODE_INDEX_COLUMN], errors="raise")
    if not np.all(np.equal(np.mod(node_index, 1.0), 0.0)):
        raise ValueError("COMSOL node samples node_index must contain integers")
    if frame[NODE_INDEX_COLUMN].duplicated().any():
        raise ValueError("COMSOL node samples contain duplicate node_index values")
    return frame.assign(**{NODE_INDEX_COLUMN: node_index.astype(np.int64)})


def _quantity_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = tuple(
        str(name) for name in frame.columns if str(name) not in _RESERVED_SAMPLE_COLUMNS
    )
    if not columns:
        raise ValueError("COMSOL node samples declare no field quantity columns")
    return columns


def _compact_mesh(
    mesh: ParsedMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return SI vertices, triangles, and the global node ids they came from.

    Only vertices referenced by the selected domain elements survive.  The
    returned global ids are the identity used to join the exported node
    samples, so the join never depends on coordinate rounding.
    """

    triangles_global = np.asarray(
        surface_triangles_from_mesh(mesh.vertices, mesh.type_blocks),
        dtype=np.int64,
    )
    if triangles_global.size == 0:
        raise ValueError(
            "selected COMSOL vacuum domains contain no surface elements; "
            "a mesh-native field needs at least one triangle"
        )
    used_global_ids = np.unique(triangles_global)
    local_index = np.full(int(mesh.vertices.shape[0]), -1, dtype=np.int64)
    local_index[used_global_ids] = np.arange(used_global_ids.size, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[used_global_ids]
    triangles = local_index[triangles_global].astype(np.int32, copy=False)
    if np.any(triangles < 0):
        raise RuntimeError("COMSOL mesh compaction lost a referenced vertex")
    return vertices, triangles, used_global_ids


def _node_values(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    used_global_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    exported_ids = np.asarray(frame[NODE_INDEX_COLUMN], dtype=np.int64)
    position_by_id = np.full(int(exported_ids.max()) + 1, -1, dtype=np.int64)
    position_by_id[exported_ids] = np.arange(exported_ids.size, dtype=np.int64)
    out_of_range = used_global_ids[used_global_ids >= position_by_id.size]
    rows = np.full(used_global_ids.size, -1, dtype=np.int64)
    in_range = used_global_ids < position_by_id.size
    rows[in_range] = position_by_id[used_global_ids[in_range]]
    missing = used_global_ids[rows < 0]
    if missing.size or out_of_range.size:
        unresolved = np.unique(np.concatenate((missing, out_of_range)))
        raise ValueError(
            "COMSOL node samples are missing mesh vertices used by the selected "
            f"vacuum domains: {unresolved[:12].tolist()} "
            f"({unresolved.size} total)"
        )

    values: dict[str, np.ndarray] = {}
    for column in columns:
        series = np.asarray(
            pd.to_numeric(frame[column], errors="coerce"),
            dtype=np.float64,
        )[rows]
        if not np.all(np.isfinite(series)):
            bad = np.flatnonzero(~np.isfinite(series))
            raise ValueError(
                f"COMSOL node samples for {column!r} are non-finite at mesh "
                f"vertices {used_global_ids[bad][:12].tolist()} "
                f"({bad.size} of {series.size}).  A mesh-native field has no "
                "valid-mask fallback: every vertex of the selected vacuum "
                "domain must carry a finite value"
            )
        values[column] = series
    return values


def _mesh_quality(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, float]:
    points = vertices[triangles]
    edges = np.stack(
        [
            np.linalg.norm(points[:, j, :] - points[:, i, :], axis=1)
            for i, j in ((0, 1), (1, 2), (2, 0))
        ],
        axis=1,
    )
    twice_area = np.abs(
        (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    )
    longest = np.max(edges, axis=1)
    positive = longest > 0.0
    if not np.any(positive):
        raise ValueError("COMSOL surface mesh has no triangle with a positive edge")
    altitudes = twice_area[positive] / longest[positive]
    return {
        "min_edge_length_m": float(np.min(edges[edges > 0.0])),
        "max_edge_length_m": float(np.max(edges)),
        "min_altitude_m": float(np.min(altitudes)),
        "total_area_m2": float(0.5 * np.sum(twice_area)),
    }


def pack_mesh_field_bundle(
    node_samples_csv: Path,
    destination: Path,
    *,
    mesh: ParsedMesh,
) -> PackedMeshField:
    """Write one triangle-mesh field NPZ from SI mesh nodes and node samples.

    ``mesh`` must already be restricted to the selected vacuum domains and
    scaled to metres, exactly as the geometry artifact sees it.
    """

    frame = _node_sample_frame(Path(node_samples_csv))
    columns = _quantity_columns(frame)
    vertices, triangles, used_global_ids = _compact_mesh(mesh)
    values = _node_values(frame, columns, used_global_ids)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "mesh_vertices": vertices,
        "mesh_triangles": triangles,
        "times": np.asarray([0.0], dtype=np.float64),
        **values,
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "source_kind": "comsol_mesh_node_samples",
                    "source_samples": str(Path(node_samples_csv).resolve()),
                    "artifact_coordinate_unit": "m",
                    "node_identity": "comsol_mphtxt_global_vertex_index",
                },
                sort_keys=True,
            )
        ),
    }
    np.savez_compressed(destination, **payload)

    summary = {
        "mode": "mesh_native",
        "mesh_vertex_count": int(vertices.shape[0]),
        "mesh_triangle_count": int(triangles.shape[0]),
        "exported_node_count": int(frame.shape[0]),
        "quantities": list(columns),
        **_mesh_quality(vertices, triangles),
    }
    return PackedMeshField(path=destination, summary=summary)


__all__ = ("PackedMeshField", "pack_mesh_field_bundle")
