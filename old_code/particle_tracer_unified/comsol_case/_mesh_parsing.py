"""Parse MPHTXT mesh blocks and select explicit surface domains."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshTypeBlock:
    type_name: str
    vertices_per_element: int
    elements: np.ndarray
    geometric_entity_indices: np.ndarray


@dataclass(frozen=True)
class ParsedMesh:
    sdim: int
    vertices: np.ndarray
    type_blocks: dict[str, MeshTypeBlock]


def _find_line(lines: list[str], start: int, marker: str) -> int:
    for index in range(start, len(lines)):
        if marker in lines[index]:
            return index
    raise ValueError(f"Could not find marker: {marker}")


def _consume_numbers(
    lines: list[str],
    start: int,
    count: int,
    cast: Any,
) -> tuple[list[Any], int]:
    values: list[Any] = []
    cursor = start
    while len(values) < count and cursor < len(lines):
        line = lines[cursor].strip()
        cursor += 1
        if not line or line.startswith("#"):
            continue
        values.extend(cast(token) for token in line.split())
    if len(values) < count:
        raise ValueError(f"Expected {count} numeric values, got {len(values)}")
    return values[:count], cursor


def parse_comsol_mphtxt(path: Path) -> ParsedMesh:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    sdim_index = _find_line(lines, 0, "# sdim")
    sdim = int(lines[sdim_index].split("#")[0].strip())
    vertex_count_index = _find_line(lines, sdim_index, "# number of mesh vertices")
    vertex_count = int(lines[vertex_count_index].split("#")[0].strip())
    coordinates_header = _find_line(
        lines,
        vertex_count_index,
        "# Mesh vertex coordinates",
    )
    coordinates, cursor = _consume_numbers(
        lines,
        coordinates_header + 1,
        vertex_count * sdim,
        float,
    )
    vertices = np.asarray(coordinates, dtype=np.float64).reshape((vertex_count, sdim))

    type_count_index = _find_line(lines, cursor, "# number of element types")
    type_count = int(lines[type_count_index].split("#")[0].strip())
    cursor = type_count_index + 1
    blocks: dict[str, MeshTypeBlock] = {}
    for _ in range(type_count):
        type_name_index = _find_line(lines, cursor, "# type name")
        tokens = lines[type_name_index].split("#")[0].split()
        if len(tokens) < 2:
            raise ValueError(f"Invalid type-name line: {lines[type_name_index]}")
        type_name = tokens[1].strip()
        vertices_index = _find_line(
            lines,
            type_name_index,
            "# number of vertices per element",
        )
        vertices_per_element = int(lines[vertices_index].split("#")[0].strip())
        count_index = _find_line(lines, vertices_index, "# number of elements")
        element_count = int(lines[count_index].split("#")[0].strip())
        elements_header = _find_line(lines, count_index, "# Elements")
        element_values, cursor = _consume_numbers(
            lines,
            elements_header + 1,
            element_count * vertices_per_element,
            int,
        )
        elements = np.asarray(element_values, dtype=np.int64).reshape(
            (element_count, vertices_per_element)
        )
        entity_count_index = _find_line(
            lines,
            cursor,
            "# number of geometric entity indices",
        )
        entity_count = int(lines[entity_count_index].split("#")[0].strip())
        entity_header = _find_line(
            lines,
            entity_count_index,
            "# Geometric entity indices",
        )
        entity_values, cursor = _consume_numbers(
            lines,
            entity_header + 1,
            entity_count,
            int,
        )
        entity_indices = np.asarray(entity_values, dtype=np.int64)
        if entity_indices.size != element_count:
            raise ValueError(
                f"Geometric entity size mismatch for {type_name}: "
                f"{entity_indices.size} vs {element_count}"
            )
        blocks[type_name] = MeshTypeBlock(
            type_name=type_name,
            vertices_per_element=vertices_per_element,
            elements=elements,
            geometric_entity_indices=entity_indices,
        )
    return ParsedMesh(sdim=sdim, vertices=vertices, type_blocks=blocks)


def scale_mesh_coordinates(
    mesh: ParsedMesh,
    scale_m_per_model_unit: float,
) -> ParsedMesh:
    scale = float(scale_m_per_model_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate scale must be a positive finite value")
    if scale == 1.0:
        return mesh
    return ParsedMesh(
        sdim=mesh.sdim,
        vertices=np.asarray(mesh.vertices, dtype=np.float64) * scale,
        type_blocks=mesh.type_blocks,
    )


def _validated_domain_ids(vacuum_domain_ids: Sequence[int]) -> tuple[int, ...]:
    selected: list[int] = []
    for raw_domain_id in vacuum_domain_ids:
        if isinstance(raw_domain_id, bool) or not isinstance(
            raw_domain_id,
            (int, np.integer),
        ):
            raise ValueError("vacuum_domain_ids must contain integers")
        domain_id = int(raw_domain_id)
        if domain_id <= 0:
            raise ValueError("vacuum_domain_ids must contain positive integers")
        selected.append(domain_id)
    if not selected:
        raise ValueError("at least one explicit vacuum_domain_id is required")
    if len(set(selected)) != len(selected):
        raise ValueError("vacuum_domain_ids must not contain duplicates")
    return tuple(selected)


# COMSOL writes a second-order mesh as tri2/quad2 with mid-side nodes.  The
# field path interpolates P1 on corner nodes only, so accepting those blocks
# would quietly discard half the mesh resolution.  Reject them by name instead.
_SECOND_ORDER_SURFACE_TYPES = ("tri2", "quad2")


def _reject_second_order_surface_elements(mesh: ParsedMesh) -> None:
    present = [name for name in _SECOND_ORDER_SURFACE_TYPES if name in mesh.type_blocks]
    if present:
        raise ValueError(
            "COMSOL mesh uses second-order surface elements "
            f"{present}; the field path interpolates P1 on corner nodes and "
            "would silently drop the mid-side nodes.  Export a first-order "
            "mesh (tri/quad) until P2 sampling exists"
        )


def _available_surface_domain_ids(mesh: ParsedMesh) -> set[int]:
    return {
        int(domain_id) + 1
        for type_name in ("tri", "quad")
        if (block := mesh.type_blocks.get(type_name)) is not None
        for domain_id in np.asarray(block.geometric_entity_indices, dtype=np.int64)
    }


def select_vacuum_domains(
    mesh: ParsedMesh,
    vacuum_domain_ids: Sequence[int],
) -> tuple[ParsedMesh, tuple[int, ...]]:
    """Retain only explicitly selected 2D particle-domain elements."""

    _reject_second_order_surface_elements(mesh)
    selected = _validated_domain_ids(vacuum_domain_ids)
    selected_set = set(selected)
    available = _available_surface_domain_ids(mesh)
    missing = sorted(selected_set - available)
    if missing:
        raise ValueError(
            "vacuum_domain_ids are not present in the COMSOL surface mesh: "
            f"missing={missing}, available={sorted(available)}"
        )

    blocks: dict[str, MeshTypeBlock] = {}
    for type_name, block in mesh.type_blocks.items():
        if type_name not in {"tri", "quad"}:
            blocks[type_name] = block
            continue
        domain_ids = np.asarray(block.geometric_entity_indices, dtype=np.int64) + 1
        keep = np.isin(domain_ids, selected)
        if np.any(keep):
            blocks[type_name] = replace(
                block,
                elements=np.asarray(block.elements)[keep].copy(),
                geometric_entity_indices=np.asarray(block.geometric_entity_indices)[
                    keep
                ].copy(),
            )
    return (
        ParsedMesh(sdim=mesh.sdim, vertices=mesh.vertices, type_blocks=blocks),
        tuple(sorted(selected_set)),
    )
