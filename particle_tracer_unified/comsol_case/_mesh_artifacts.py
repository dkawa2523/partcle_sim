"""Build COMSOL entity rows and serialize canonical geometry artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._mesh_parsing import ParsedMesh
from ._mesh_topology import _edge_key, _order_quad_vertices


def _surface_edge_domain_map(mesh: ParsedMesh) -> dict[tuple[int, int], set[int]]:
    edge_domains: dict[tuple[int, int], set[int]] = {}

    def add_edge(a: int, b: int, domain_id: int) -> None:
        key = _edge_key(a, b)
        edge_domains.setdefault(key, set()).add(domain_id)

    triangle_block = mesh.type_blocks.get("tri")
    if triangle_block is not None and triangle_block.elements.size:
        domain_ids = (
            np.asarray(triangle_block.geometric_entity_indices, dtype=np.int32) + 1
        ).astype(np.int32)
        for element, domain_id in zip(
            np.asarray(triangle_block.elements, dtype=np.int64),
            domain_ids,
            strict=True,
        ):
            add_edge(int(element[0]), int(element[1]), int(domain_id))
            add_edge(int(element[1]), int(element[2]), int(domain_id))
            add_edge(int(element[2]), int(element[0]), int(domain_id))

    quad_block = mesh.type_blocks.get("quad")
    if quad_block is not None and quad_block.elements.size:
        domain_ids = (
            np.asarray(quad_block.geometric_entity_indices, dtype=np.int32) + 1
        ).astype(np.int32)
        elements = _order_quad_vertices(mesh.vertices, quad_block.elements)
        for element, domain_id in zip(elements, domain_ids, strict=True):
            add_edge(int(element[0]), int(element[1]), int(domain_id))
            add_edge(int(element[1]), int(element[2]), int(domain_id))
            add_edge(int(element[2]), int(element[3]), int(domain_id))
            add_edge(int(element[3]), int(element[0]), int(domain_id))
    return edge_domains


def _part_label(part_id: int) -> str:
    return f"comsol_boundary_{int(part_id)}"


@dataclass
class _BoundaryEntitySummary:
    part_id: int
    segment_count: int = 0
    x_min: float = float("inf")
    x_max: float = float("-inf")
    y_min: float = float("inf")
    y_max: float = float("-inf")
    adjacent_domain_ids: set[int] = field(default_factory=set)

    def add_segment(
        self,
        coordinates: np.ndarray,
        adjacent_domain_ids: set[int],
    ) -> None:
        self.segment_count += 1
        self.x_min = min(self.x_min, float(np.min(coordinates[:, 0])))
        self.x_max = max(self.x_max, float(np.max(coordinates[:, 0])))
        self.y_min = min(self.y_min, float(np.min(coordinates[:, 1])))
        self.y_max = max(self.y_max, float(np.max(coordinates[:, 1])))
        self.adjacent_domain_ids.update(adjacent_domain_ids)

    def as_row(
        self,
        active_part_ids: set[int],
        solver_part_id_by_entity_id: Mapping[int, int] | None = None,
    ) -> dict[str, Any]:
        entity_id = self.part_id
        solver_part_id = (
            entity_id
            if solver_part_id_by_entity_id is None
            else solver_part_id_by_entity_id.get(entity_id)
        )
        adjacent = ";".join(str(value) for value in sorted(self.adjacent_domain_ids))
        return {
            "solver_part_id": solver_part_id,
            "comsol_edge_entity_id": entity_id,
            "raw_comsol_edge_entity_index": entity_id - 1,
            "comsol_api_selection_entity_id": entity_id - 1,
            "active_in_solver_boundary": bool(
                solver_part_id is not None
                and (not active_part_ids or solver_part_id in active_part_ids)
            ),
            "segment_count": self.segment_count,
            "x_min_m": self.x_min,
            "x_max_m": self.x_max,
            "y_min_m": self.y_min,
            "y_max_m": self.y_max,
            "adjacent_domain_ids": adjacent,
            "solver_part_name": (
                "" if solver_part_id is None else _part_label(solver_part_id)
            ),
            "comsol_material_name": "not_exported_from_mphtxt",
        }


def _comsol_boundary_entity_rows(
    mesh: ParsedMesh,
    active_part_ids: list[int] | None = None,
    solver_part_id_by_entity_id: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    edge_block = mesh.type_blocks.get("edg")
    if edge_block is None or edge_block.elements.size == 0:
        return []
    active = {int(part_id) for part_id in active_part_ids or []}
    edge_domains = _surface_edge_domain_map(mesh)
    grouped: dict[int, _BoundaryEntitySummary] = {}
    edge_ids = (
        np.asarray(edge_block.geometric_entity_indices, dtype=np.int32) + 1
    ).astype(np.int32)
    edge_elements = np.asarray(edge_block.elements, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    for element, edge_id in zip(edge_elements, edge_ids, strict=True):
        part_id = int(edge_id)
        a, b = int(element[0]), int(element[1])
        summary = grouped.get(part_id)
        if summary is None:
            summary = grouped[part_id] = _BoundaryEntitySummary(part_id)
        summary.add_segment(
            vertices[element],
            edge_domains.get(_edge_key(a, b), set()),
        )
    return [
        grouped[part_id].as_row(active, solver_part_id_by_entity_id)
        for part_id in sorted(grouped)
    ]


def _comsol_domain_entity_rows(
    mesh: ParsedMesh,
    vacuum_domain_ids: Sequence[int],
) -> list[dict[str, Any]]:
    selected = {int(domain_id) for domain_id in vacuum_domain_ids}
    grouped: dict[int, dict[str, Any]] = {}
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    for type_name in ("tri", "quad"):
        block = mesh.type_blocks.get(type_name)
        if block is None or block.elements.size == 0:
            continue
        domain_ids = (
            np.asarray(block.geometric_entity_indices, dtype=np.int32) + 1
        ).astype(np.int32)
        elements = np.asarray(block.elements, dtype=np.int64)
        if type_name == "quad":
            elements = _order_quad_vertices(vertices, elements)
        for element, domain_id in zip(elements, domain_ids, strict=True):
            domain_id = int(domain_id)
            coordinates = vertices[element]
            row = grouped.setdefault(
                domain_id,
                {
                    "raw_comsol_domain_entity_index": domain_id - 1,
                    "comsol_api_selection_entity_id": domain_id - 1,
                    "element_count": 0,
                    "mesh_element_types": set(),
                    "x_min": float("inf"),
                    "x_max": float("-inf"),
                    "y_min": float("inf"),
                    "y_max": float("-inf"),
                },
            )
            row["element_count"] += 1
            row["mesh_element_types"].add(type_name)
            row["x_min"] = min(row["x_min"], float(np.min(coordinates[:, 0])))
            row["x_max"] = max(row["x_max"], float(np.max(coordinates[:, 0])))
            row["y_min"] = min(row["y_min"], float(np.min(coordinates[:, 1])))
            row["y_max"] = max(row["y_max"], float(np.max(coordinates[:, 1])))

    return [
        {
            "comsol_domain_entity_id": domain_id,
            "raw_comsol_domain_entity_index": row["raw_comsol_domain_entity_index"],
            "comsol_api_selection_entity_id": row["comsol_api_selection_entity_id"],
            "selected_as_vacuum_domain": domain_id in selected,
            "element_count": row["element_count"],
            "mesh_element_types": ";".join(sorted(row["mesh_element_types"])),
            "x_min_m": row["x_min"],
            "x_max_m": row["x_max"],
            "y_min_m": row["y_min"],
            "y_max_m": row["y_max"],
            "comsol_material_name": "not_exported_from_mphtxt",
        }
        for domain_id, row in sorted(grouped.items())
    ]


def write_comsol_entity_maps(
    generated_dir: Path,
    mesh: ParsedMesh,
    active_part_ids: list[int],
    vacuum_domain_ids: Sequence[int],
    *,
    solver_part_id_by_entity_id: Mapping[int, int] | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    boundary_rows = _comsol_boundary_entity_rows(
        mesh,
        active_part_ids,
        solver_part_id_by_entity_id,
    )
    if boundary_rows:
        path = generated_dir / "comsol_boundary_entity_mapping.csv"
        pd.DataFrame(boundary_rows).to_csv(path, index=False)
        outputs["comsol_boundary_entity_mapping"] = path.name
    domain_rows = _comsol_domain_entity_rows(mesh, vacuum_domain_ids)
    if domain_rows:
        path = generated_dir / "comsol_domain_entity_mapping.csv"
        pd.DataFrame(domain_rows).to_csv(path, index=False)
        outputs["comsol_domain_entity_mapping"] = path.name
    return outputs


def write_geometry_npz(
    path: Path,
    *,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    arrays: Mapping[str, Any],
    mesh: ParsedMesh,
    metadata: Mapping[str, Any],
) -> None:
    np.savez_compressed(
        path,
        axis_0=np.asarray(axes_x, dtype=np.float64),
        axis_1=np.asarray(axes_y, dtype=np.float64),
        sdf=np.asarray(arrays["sdf"], dtype=np.float64),
        normal_0=np.asarray(arrays["normal_x"], dtype=np.float64),
        normal_1=np.asarray(arrays["normal_y"], dtype=np.float64),
        valid_mask=np.asarray(arrays["inside"], dtype=bool),
        nearest_boundary_part_id_map=np.asarray(
            arrays["nearest_boundary_part_id_map"],
            dtype=np.int32,
        ),
        boundary_edges=np.asarray(arrays["boundary_edges"], dtype=np.float64),
        boundary_edge_part_ids=np.asarray(arrays["boundary_part_ids"], dtype=np.int32),
        boundary_loops_2d_flat=np.asarray(
            arrays["boundary_loops_2d_flat"],
            dtype=np.float64,
        ),
        boundary_loops_2d_offsets=np.asarray(
            arrays["boundary_loops_2d_offsets"],
            dtype=np.int32,
        ),
        mesh_vertices=mesh.vertices.astype(np.float64),
        mesh_triangles=np.asarray(arrays["triangles"], dtype=np.int32),
        mesh_triangle_part_ids=np.asarray(arrays["triangle_part_ids"], dtype=np.int32),
        mesh_quads=np.asarray(arrays["quads"], dtype=np.int32),
        mesh_quad_part_ids=np.asarray(arrays["quad_part_ids"], dtype=np.int32),
        metadata_json=np.asarray(json.dumps(dict(metadata))),
    )
