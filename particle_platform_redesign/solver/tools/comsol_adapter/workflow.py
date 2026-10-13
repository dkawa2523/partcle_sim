"""One-way adapter from supported COMSOL CSV exports to canonical data.

The adapter owns provider syntax and deterministic normalization only.  It does
not solve a field equation, select particle physics, or compare against COMSOL.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    write,
)
from chamber_particles.yaml_input import parse_document

type _BoundaryOwner = tuple[int, tuple[int, int]]
type _BoundaryRow = tuple[int, int, int, int, int, int]

ADAPTER_FORMAT_VERSION: Final = 3
ADAPTER_REVISION: Final = "comsol_selected_nodal_csv_adapter_v3"
FIELD_SEMANTICS_REVISION: Final = "explicit_selected_field_lineage_v1"
PRODUCER_VERSION: Final = version("chamber-particles")
_FINITE_NUMBER_TOKEN: Final = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?")
_EXPECTED_SOURCE_KEYS: Final = {
    "vertices_path",
    "triangles_path",
    "quadrilaterals_path",
    "boundary_edges_path",
    "field_values_path",
    "field_columns_path",
}


@dataclass(frozen=True, slots=True)
class SourceFiles:
    vertices: Path
    triangles: Path
    quadrilaterals: Path
    boundary_edges: Path
    field_values: Path
    field_columns: Path


@dataclass(frozen=True, slots=True)
class _SourceContents:
    vertices: bytes
    triangles: bytes
    quadrilaterals: bytes
    boundary_edges: bytes
    field_values: bytes
    field_columns: bytes


@dataclass(frozen=True, slots=True)
class BoundaryGroup:
    name: str
    external_ids: tuple[int, ...]
    material_id: int


@dataclass(frozen=True, slots=True)
class DomainSpecification:
    coordinate_system: str
    external_id: int
    layout_name: str
    quadrilateral_order: str
    excluded_boundary_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FieldMapping:
    name: str
    columns: tuple[str, ...]
    components: tuple[str, ...]
    stored_basis: str
    unit: str
    input_unit: str
    scale: float
    offset: float
    producer_record: dict[str, object]
    axis_zero_component: str | None
    axis_correction_limit: float
    axis_reason: str | None


@dataclass(frozen=True, slots=True)
class FieldSpecification:
    coordinate_columns: tuple[str, str]
    inside_column: str
    node_id_column: str | None
    coordinate_match_tolerance_m: float
    mappings: tuple[FieldMapping, ...]
    scope_column: str | None
    scope_mode: str | None
    scope_evidence: str | None


@dataclass(frozen=True, slots=True)
class AdapterSpecification:
    source: SourceFiles
    domain: DomainSpecification
    boundary_groups: tuple[BoundaryGroup, ...]
    fields: FieldSpecification


@dataclass(frozen=True, slots=True)
class _BoundaryRecord:
    external_id: int
    first_node: int
    second_node: int
    boundary_id: int


@dataclass(frozen=True, slots=True)
class _MeshResult:
    nodes_m: np.ndarray
    node_external_id: np.ndarray
    triangles: np.ndarray
    triangle_domain_id: np.ndarray
    boundary: BoundaryData
    group_names: tuple[str, ...]
    source_triangle_count: int
    source_quadrilateral_count: int
    split_diagonal_03_count: int
    split_diagonal_12_count: int
    minimum_triangle_quality: float
    excluded_boundary_count: int
    group_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _FieldResult:
    fields: tuple[FieldData, ...]
    selected_row_count: int
    maximum_match_distance_m: float
    join_method: str
    axis_transformations: list[dict[str, object]]
    selected_field_lineage: list[dict[str, object]]
    provider_row_count: int


def adapt_from_configuration(
    configuration_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Create one checked selected-field bundle without replacing files."""

    config_path = Path(configuration_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    report = Path(report_path).expanduser().resolve() if report_path is not None else None
    if report == output:
        raise ValueError("report_path and output_path must be different")
    if output.exists():
        raise FileExistsError(output)
    if report is not None and report.exists():
        raise FileExistsError(report)

    raw = config_path.read_bytes()
    specification = parse_configuration(raw, base_directory=config_path.parent)
    source_contents, source_hashes, aggregate_source_hash = _snapshot_sources(specification.source)
    mesh = _build_mesh(specification, source_contents)
    field_result = _read_fields(specification, mesh, source_contents)
    provenance = _provenance(
        config_sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        aggregate_source_hash=aggregate_source_hash,
        source_hashes=source_hashes,
        specification=specification,
        mesh=mesh,
        fields=field_result,
    )
    bundle = DataBundle(
        coordinate_system=cast(
            Literal["cartesian_xy", "axisymmetric_rz"], specification.domain.coordinate_system
        ),
        provenance_json=json.dumps(
            provenance,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        geometry=GeometryData(
            nodes_m=cast(np.ndarray, mesh.nodes_m),
            boundary=mesh.boundary,
            group_names=mesh.group_names,
            node_external_id=cast(np.ndarray, mesh.node_external_id),
            tri3=cast(np.ndarray, mesh.triangles),
            tri3_domain_id=cast(np.ndarray, mesh.triangle_domain_id),
        ),
        layouts=(
            P1TriLayout(
                specification.domain.layout_name,
                cast(np.ndarray, mesh.nodes_m),
                cast(np.ndarray, mesh.triangles),
                np.ones(mesh.triangles.shape[0], dtype="<u1"),
            ),
        ),
        fields=field_result.fields,
        sources=(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_info = write(output, bundle)
    result = _report(
        specification,
        mesh,
        field_result,
        output,
        output_info.content_hash,
        aggregate_source_hash,
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8", errors="strict") as stream:
            stream.write(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return result


def parse_configuration(raw: bytes, *, base_directory: Path) -> AdapterSpecification:
    """Parse the strict provider-specific adapter configuration."""

    try:
        document = parse_document(raw)
    except ValueError as error:
        raise ValueError(f"invalid adapter configuration: {error}") from error
    root = _mapping(document, "configuration")
    _exact_keys(
        root,
        {"format_version", "source", "domain", "boundary_groups", "fields"},
        "configuration",
    )
    if _integer(root["format_version"], "format_version") != ADAPTER_FORMAT_VERSION:
        raise ValueError(f"format_version must be {ADAPTER_FORMAT_VERSION}")
    source = _parse_source(_mapping(root["source"], "source"), base_directory)
    domain = _parse_domain(_mapping(root["domain"], "domain"))
    groups = _parse_boundary_groups(root["boundary_groups"], domain.excluded_boundary_ids)
    fields = _parse_fields(_mapping(root["fields"], "fields"))
    _validate_selected_meanings(domain, fields)
    return AdapterSpecification(source, domain, groups, fields)


def _validate_selected_meanings(domain: DomainSpecification, fields: FieldSpecification) -> None:
    for item in fields.mappings:
        if item.producer_record["domain_ids"] != [domain.external_id]:
            raise ValueError(
                f"field {item.name} producer domain_ids must match the selected domain"
            )
        if item.stored_basis not in ("scalar", domain.coordinate_system):
            raise ValueError(f"field {item.name} stored_basis does not match the coordinates")
        if item.axis_zero_component is not None and domain.coordinate_system != "axisymmetric_rz":
            raise ValueError("axis transforms require axisymmetric_rz")


def _parse_source(value: dict[str, object], base_directory: Path) -> SourceFiles:
    _exact_keys(value, _EXPECTED_SOURCE_KEYS, "source")

    def resolve(key: str) -> Path:
        path = Path(_text(value[key], f"source.{key}"))
        return (path if path.is_absolute() else base_directory / path).resolve()

    return SourceFiles(
        vertices=resolve("vertices_path"),
        triangles=resolve("triangles_path"),
        quadrilaterals=resolve("quadrilaterals_path"),
        boundary_edges=resolve("boundary_edges_path"),
        field_values=resolve("field_values_path"),
        field_columns=resolve("field_columns_path"),
    )


def _parse_domain(value: dict[str, object]) -> DomainSpecification:
    _exact_keys(
        value,
        {
            "coordinate_system",
            "external_id",
            "layout_name",
            "quadrilateral_order",
            "excluded_boundary_ids",
        },
        "domain",
    )
    order = _text(value["quadrilateral_order"], "domain.quadrilateral_order")
    if order != "q1_reference":
        raise ValueError("domain.quadrilateral_order must be q1_reference")
    coordinates = _text(value["coordinate_system"], "domain.coordinate_system")
    if coordinates not in ("cartesian_xy", "axisymmetric_rz"):
        raise ValueError("domain.coordinate_system must be cartesian_xy or axisymmetric_rz")
    excluded = _integer_tuple(value["excluded_boundary_ids"], "domain.excluded_boundary_ids")
    if coordinates == "cartesian_xy" and excluded:
        raise ValueError("Cartesian XY cannot exclude axis boundaries")
    return DomainSpecification(
        coordinate_system=coordinates,
        external_id=_integer(value["external_id"], "domain.external_id"),
        layout_name=_name(value["layout_name"], "domain.layout_name"),
        quadrilateral_order=order,
        excluded_boundary_ids=excluded,
    )


def _parse_boundary_groups(
    value: object, excluded_boundary_ids: tuple[int, ...]
) -> tuple[BoundaryGroup, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("boundary_groups must be a nonempty list")
    groups: list[BoundaryGroup] = []
    names: set[str] = set()
    assigned: set[int] = set(excluded_boundary_ids)
    for index, raw in enumerate(value):
        label = f"boundary_groups[{index}]"
        item = _mapping(raw, label)
        _exact_keys(item, {"name", "external_ids", "material_id"}, label)
        name = _name(item["name"], f"{label}.name")
        ids = _integer_tuple(item["external_ids"], f"{label}.external_ids")
        if not ids:
            raise ValueError(f"{label}.external_ids must not be empty")
        overlap = assigned.intersection(ids)
        if name in names or overlap:
            raise ValueError(
                f"duplicate boundary group name or external ID: {name}, {sorted(overlap)}"
            )
        names.add(name)
        assigned.update(ids)
        groups.append(
            BoundaryGroup(name, ids, _integer(item["material_id"], f"{label}.material_id"))
        )
    return tuple(groups)


def _parse_fields(value: dict[str, object]) -> FieldSpecification:
    _exact_keys(
        value,
        {
            "coordinate_columns",
            "inside_column",
            "node_id_column",
            "domain_scope",
            "coordinate_match_tolerance_m",
            "mappings",
        },
        "fields",
    )
    raw_mappings = value["mappings"]
    if not isinstance(raw_mappings, list) or not raw_mappings:
        raise ValueError("fields.mappings must be a nonempty list")
    mappings = tuple(
        _parse_field_mapping(_mapping(raw, f"fields.mappings[{index}]"), index)
        for index, raw in enumerate(raw_mappings)
    )
    if len({item.name for item in mappings}) != len(mappings):
        raise ValueError("fields.mappings must have unique canonical names")
    node_column = value["node_id_column"]
    scope_column, scope_mode, scope_evidence = _parse_domain_scope(value["domain_scope"])
    return FieldSpecification(
        _name_pair(value["coordinate_columns"], "fields.coordinate_columns"),
        _name(value["inside_column"], "fields.inside_column"),
        None if node_column is None else _name(node_column, "fields.node_id_column"),
        _positive_number(
            value["coordinate_match_tolerance_m"], "fields.coordinate_match_tolerance_m"
        ),
        mappings,
        scope_column,
        scope_mode,
        scope_evidence,
    )


def _parse_domain_scope(value: object) -> tuple[str | None, str | None, str | None]:
    if value is None:
        return None, None, None
    item = _mapping(value, "fields.domain_scope")
    _exact_keys(item, {"column", "mode", "evidence"}, "fields.domain_scope")
    mode = _text(item["mode"], "fields.domain_scope.mode")
    if mode not in ("finite", "equals_domain_id"):
        raise ValueError("domain_scope.mode must be finite or equals_domain_id")
    return (
        _name(item["column"], "fields.domain_scope.column"),
        mode,
        _text(item["evidence"], "fields.domain_scope.evidence"),
    )


def _parse_field_mapping(value: dict[str, object], index: int) -> FieldMapping:
    label = f"fields.mappings[{index}]"
    _exact_keys(
        value,
        {
            "name",
            "columns",
            "components",
            "stored_basis",
            "association",
            "unit",
            "input_unit",
            "scale",
            "offset",
            "producer_record",
            "axis_transform",
        },
        label,
    )
    if value["association"] != "node":
        raise ValueError(f"{label}: only node association is supported")
    columns = _names(value["columns"], f"{label}.columns")
    components = _names(value["components"], f"{label}.components")
    if len(columns) != len(components):
        raise ValueError(f"{label}: columns/components lengths differ")
    producer = _parse_producer_record(_mapping(value["producer_record"], label), label)
    component, limit, reason = _parse_axis_transform(value["axis_transform"], components, label)
    return FieldMapping(
        _name(value["name"], f"{label}.name"),
        columns,
        components,
        _text(value["stored_basis"], f"{label}.stored_basis"),
        _text(value["unit"], f"{label}.unit"),
        _text(value["input_unit"], f"{label}.input_unit"),
        _positive_number(value["scale"], f"{label}.scale"),
        _number(value["offset"], f"{label}.offset"),
        producer,
        component,
        limit,
        reason,
    )


def _parse_producer_record(value: dict[str, object], label: str) -> dict[str, object]:
    required = {
        "producer",
        "dataset",
        "solution",
        "member",
        "time",
        "domain_ids",
        "side",
        "recovery",
        "continuity",
        "authority",
        "topology",
        "support",
        "evidence",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{label}.producer_record missing keys: {sorted(missing)}")
    if value["topology"] != "fixed" or value["support"] != "full_selected_domain":
        raise ValueError(
            f"{label}: only fixed topology and full selected-domain support are supported"
        )
    time = _mapping(value["time"], f"{label}.producer_record.time")
    if time.get("representation") != "static":
        raise ValueError(f"{label}: time snapshots must use a separate qualified producer")
    _text(time.get("selection"), f"{label}.producer_record.time.selection")
    member = _mapping(value["member"], f"{label}.producer_record.member")
    if not member:
        raise ValueError(f"{label}.producer_record.member must state a selection or its absence")
    if value["continuity"] != "continuous" or value["side"] != "domain":
        raise ValueError(f"{label}: one-sided/discontinuous fields are not supported")
    _integer_tuple(value["domain_ids"], f"{label}.producer_record.domain_ids")
    for key in ("producer", "dataset", "solution", "recovery", "authority", "evidence"):
        _text(value[key], f"{label}.producer_record.{key}")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}.producer_record must be finite JSON data") from error
    return value


def _parse_axis_transform(
    value: object, components: tuple[str, ...], label: str
) -> tuple[str | None, float, str | None]:
    if value is None:
        return None, 0.0, None
    item = _mapping(value, f"{label}.axis_transform")
    _exact_keys(item, {"component", "max_abs_correction", "reason"}, f"{label}.axis_transform")
    component = _name(item["component"], f"{label}.axis_transform.component")
    if component not in components:
        raise ValueError(f"{label}: axis transform component is absent")
    return (
        component,
        _nonnegative_number(item["max_abs_correction"], f"{label}.max_abs_correction"),
        _text(item["reason"], f"{label}.axis_transform.reason"),
    )


def _build_mesh(specification: AdapterSpecification, contents: _SourceContents) -> _MeshResult:
    vertices = _read_vertices(
        specification.source.vertices, contents.vertices, specification.domain.coordinate_system
    )
    source_triangles = _read_cells(
        specification.source.triangles,
        contents.triangles,
        id_column="triangle_id",
        node_columns=("node1_id", "node2_id", "node3_id"),
        domain_id=specification.domain.external_id,
    )
    source_quads = _read_cells(
        specification.source.quadrilaterals,
        contents.quadrilaterals,
        id_column="quadrilateral_id",
        node_columns=("node1_id", "node2_id", "node3_id", "node4_id"),
        domain_id=specification.domain.external_id,
    )
    if not source_triangles and not source_quads:
        raise ValueError(f"domain {specification.domain.external_id} has no volume cells")
    used_external_ids = sorted(
        {node for _cell_id, nodes in (*source_triangles, *source_quads) for node in nodes}
    )
    missing = set(used_external_ids).difference(vertices)
    if missing:
        raise ValueError(f"volume cells reference unknown vertices: {sorted(missing)[:8]}")
    external_to_local = {external: local for local, external in enumerate(used_external_ids)}
    nodes = np.asarray([vertices[node] for node in used_external_ids], dtype="<f8")
    triangles: list[tuple[int, int, int]] = []
    for _cell_id, external_nodes in source_triangles:
        local = (
            external_to_local[external_nodes[0]],
            external_to_local[external_nodes[1]],
            external_to_local[external_nodes[2]],
        )
        triangles.append(_counter_clockwise_triangle(local, nodes))
    split_03 = 0
    split_12 = 0
    for _cell_id, external_nodes in source_quads:
        local = tuple(external_to_local[node] for node in external_nodes)
        first, second, choice = _split_q1_quad(local, external_nodes, nodes)
        triangles.extend((first, second))
        if choice == "03":
            split_03 += 1
        else:
            split_12 += 1
    connectivity = np.ascontiguousarray(np.asarray(triangles, dtype="<i8"))
    minimum_quality = min(
        _triangle_quality((int(cell[0]), int(cell[1]), int(cell[2])), nodes)
        for cell in connectivity
    )
    boundary, names, excluded_count, group_counts = _build_boundary(
        specification,
        connectivity,
        nodes,
        used_external_ids,
        contents.boundary_edges,
    )
    return _MeshResult(
        nodes_m=np.ascontiguousarray(nodes),
        node_external_id=np.asarray(used_external_ids, dtype="<i8"),
        triangles=connectivity,
        triangle_domain_id=np.full(
            connectivity.shape[0], specification.domain.external_id, dtype="<i4"
        ),
        boundary=boundary,
        group_names=names,
        source_triangle_count=len(source_triangles),
        source_quadrilateral_count=len(source_quads),
        split_diagonal_03_count=split_03,
        split_diagonal_12_count=split_12,
        minimum_triangle_quality=minimum_quality,
        excluded_boundary_count=excluded_count,
        group_counts=group_counts,
    )


def _read_vertices(
    path: Path, content: bytes, coordinate_system: str
) -> dict[int, tuple[float, float]]:
    coordinates = ("r_m", "z_m") if coordinate_system == "axisymmetric_rz" else ("x_m", "y_m")
    rows = _csv_rows(path, content, ("vertex_id", *coordinates))
    result: dict[int, tuple[float, float]] = {}
    for row_number, row in rows:
        identifier = _csv_int(row["vertex_id"], path, row_number, "vertex_id")
        coordinate = (
            _csv_float(row[coordinates[0]], path, row_number, coordinates[0]),
            _csv_float(row[coordinates[1]], path, row_number, coordinates[1]),
        )
        if identifier in result:
            raise ValueError(f"{path}: duplicate vertex_id {identifier}")
        if coordinate_system == "axisymmetric_rz" and coordinate[0] < 0.0:
            raise ValueError(f"{path}: radial coordinate must be nonnegative")
        result[identifier] = coordinate
    if not result:
        raise ValueError(f"{path}: no vertices")
    return result


def _read_cells(
    path: Path,
    content: bytes,
    *,
    id_column: str,
    node_columns: tuple[str, ...],
    domain_id: int,
) -> list[tuple[int, tuple[int, ...]]]:
    required = (id_column, *node_columns, "domain_id")
    selected: list[tuple[int, tuple[int, ...]]] = []
    seen: set[int] = set()
    for row_number, row in _csv_rows(path, content, required):
        cell_id = _csv_int(row[id_column], path, row_number, id_column)
        if cell_id in seen:
            raise ValueError(f"{path}: duplicate {id_column} {cell_id}")
        seen.add(cell_id)
        if _csv_int(row["domain_id"], path, row_number, "domain_id") == domain_id:
            nodes = tuple(_csv_int(row[name], path, row_number, name) for name in node_columns)
            if len(set(nodes)) != len(nodes):
                raise ValueError(f"{path}: cell {cell_id} repeats a node")
            selected.append((cell_id, nodes))
    selected.sort(key=lambda item: item[0])
    return selected


def _counter_clockwise_triangle(
    nodes: tuple[int, int, int], coordinates: np.ndarray
) -> tuple[int, int, int]:
    first, second, third = coordinates[np.asarray(nodes, dtype=np.int64)]
    area2 = _cross(second - first, third - first)
    if not math.isfinite(area2) or area2 == 0.0:
        raise ValueError("volume mesh contains a degenerate triangle")
    return nodes if area2 > 0.0 else (nodes[0], nodes[2], nodes[1])


def _split_q1_quad(
    local: tuple[int, ...], external: tuple[int, ...], coordinates: np.ndarray
) -> tuple[tuple[int, int, int], tuple[int, int, int], str]:
    if len(local) != 4:
        raise ValueError("quadrilateral must have four nodes")
    polygon = (local[0], local[1], local[3], local[2])
    points = coordinates[np.asarray(polygon, dtype=np.int64)]
    signs = [
        _cross(
            points[(index + 1) % 4] - points[index],
            points[(index + 2) % 4] - points[(index + 1) % 4],
        )
        for index in range(4)
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in signs):
        raise ValueError("quadrilateral is not a nondegenerate convex Q1 cell")
    candidates = {
        "03": ((local[0], local[1], local[3]), (local[0], local[3], local[2])),
        "12": ((local[0], local[1], local[2]), (local[1], local[3], local[2])),
    }
    scores = {
        name: min(_triangle_quality(item, coordinates) for item in pair)
        for name, pair in candidates.items()
    }
    if scores["03"] == scores["12"]:
        diagonal_03 = tuple(sorted((external[0], external[3])))
        diagonal_12 = tuple(sorted((external[1], external[2])))
        choice = "03" if diagonal_03 < diagonal_12 else "12"
    else:
        choice = "03" if scores["03"] > scores["12"] else "12"
    first, second = candidates[choice]
    return (
        _counter_clockwise_triangle(first, coordinates),
        _counter_clockwise_triangle(second, coordinates),
        choice,
    )


def _triangle_quality(nodes: tuple[int, int, int], coordinates: np.ndarray) -> float:
    points = coordinates[np.asarray(nodes, dtype=np.int64)]
    area2 = abs(_cross(points[1] - points[0], points[2] - points[0]))
    denominator = sum(
        float(
            np.dot(points[(index + 1) % 3] - points[index], points[(index + 1) % 3] - points[index])
        )
        for index in range(3)
    )
    if (
        not math.isfinite(area2)
        or not math.isfinite(denominator)
        or area2 <= 0.0
        or denominator <= 0.0
    ):
        raise ValueError("quadrilateral split produced a degenerate triangle")
    return 2.0 * math.sqrt(3.0) * area2 / denominator


def _build_boundary(
    specification: AdapterSpecification,
    triangles: np.ndarray,
    coordinates: np.ndarray,
    external_nodes: list[int],
    boundary_content: bytes,
) -> tuple[BoundaryData, tuple[str, ...], int, dict[str, int]]:
    perimeter = _perimeter_edges(triangles)
    exported_by_edge = _exported_boundary_map(specification.source.boundary_edges, boundary_content)
    group_by_external = _boundary_group_map(specification.boundary_groups)
    rows, excluded_count, group_counts = _collect_boundary_rows(
        perimeter,
        exported_by_edge,
        group_by_external,
        set(specification.domain.excluded_boundary_ids),
        specification.boundary_groups,
        external_nodes,
        coordinates,
        specification.domain.coordinate_system,
    )
    boundary = _make_boundary_data(rows, specification.boundary_groups)
    names = tuple(group.name for group in specification.boundary_groups)
    return boundary, names, excluded_count, group_counts


def _perimeter_edges(triangles: np.ndarray) -> dict[tuple[int, int], _BoundaryOwner]:
    incidence: dict[tuple[int, int], list[_BoundaryOwner]] = {}
    for cell_index, triangle in enumerate(triangles):
        for local_edge in range(3):
            edge = (int(triangle[local_edge]), int(triangle[(local_edge + 1) % 3]))
            key = (min(edge), max(edge))
            incidence.setdefault(key, []).append((cell_index, edge))
    if any(len(owners) not in (1, 2) for owners in incidence.values()):
        raise ValueError("volume mesh is non-manifold")
    return {edge: owners[0] for edge, owners in incidence.items() if len(owners) == 1}


def _exported_boundary_map(path: Path, content: bytes) -> dict[tuple[int, int], _BoundaryRecord]:
    exported_by_edge: dict[tuple[int, int], _BoundaryRecord] = {}
    for record in _read_boundaries(path, content):
        key = (
            min(record.first_node, record.second_node),
            max(record.first_node, record.second_node),
        )
        if key in exported_by_edge:
            raise ValueError(f"duplicate exported boundary edge {key}")
        exported_by_edge[key] = record
    return exported_by_edge


def _boundary_group_map(
    groups: tuple[BoundaryGroup, ...],
) -> dict[int, tuple[int, BoundaryGroup]]:
    group_by_external: dict[int, tuple[int, BoundaryGroup]] = {}
    for group_id, group in enumerate(groups):
        for external_id in group.external_ids:
            group_by_external[external_id] = (group_id, group)
    return group_by_external


def _collect_boundary_rows(
    perimeter: dict[tuple[int, int], _BoundaryOwner],
    exported_by_edge: dict[tuple[int, int], _BoundaryRecord],
    group_by_external: dict[int, tuple[int, BoundaryGroup]],
    excluded: set[int],
    groups: tuple[BoundaryGroup, ...],
    external_nodes: list[int],
    coordinates: np.ndarray,
    coordinate_system: str,
) -> tuple[list[_BoundaryRow], int, dict[str, int]]:
    rows: list[_BoundaryRow] = []
    group_counts = {group.name: 0 for group in groups}
    excluded_count = 0
    encountered_ids: set[int] = set()
    for local_edge, (owner_index, oriented_edge) in perimeter.items():
        external_edge = tuple(
            sorted((external_nodes[local_edge[0]], external_nodes[local_edge[1]]))
        )
        if external_edge not in exported_by_edge:
            raise ValueError(
                f"domain perimeter edge is absent from boundary export: {external_edge}"
            )
        record = exported_by_edge[external_edge]
        encountered_ids.add(record.boundary_id)
        is_axis = bool(
            coordinate_system == "axisymmetric_rz"
            and coordinates[local_edge[0], 0] == 0.0
            and coordinates[local_edge[1], 0] == 0.0
        )
        if record.boundary_id in excluded:
            if not is_axis:
                raise ValueError(
                    f"excluded boundary ID {record.boundary_id} contains a non-axis facet"
                )
            excluded_count += 1
            continue
        if is_axis:
            raise ValueError(
                f"axis facet {external_edge} must belong to an explicitly excluded boundary ID"
            )
        try:
            group_id, group = group_by_external[record.boundary_id]
        except KeyError as error:
            raise ValueError(
                f"boundary ID {record.boundary_id} is neither grouped nor explicitly excluded"
            ) from error
        rows.append(
            (
                owner_index,
                oriented_edge[0],
                oriented_edge[1],
                record.external_id,
                record.boundary_id,
                group_id,
            )
        )
        group_counts[group.name] += 1
    configured_ids = set(group_by_external).union(excluded)
    if encountered_ids != configured_ids:
        raise ValueError(
            "configured boundary IDs do not exactly match the selected domain perimeter: "
            f"unused={sorted(configured_ids - encountered_ids)}, "
            f"unconfigured={sorted(encountered_ids - configured_ids)}"
        )
    if not rows:
        raise ValueError("no physical boundary facets remain after exclusions")
    empty = [name for name, count in group_counts.items() if count == 0]
    if empty:
        raise ValueError(f"configured boundary groups have no domain perimeter facets: {empty}")
    rows.sort(key=lambda item: (item[3], item[0], item[1], item[2]))
    return rows, excluded_count, group_counts


def _make_boundary_data(
    rows: list[_BoundaryRow], groups: tuple[BoundaryGroup, ...]
) -> BoundaryData:
    owner = np.asarray([item[0] for item in rows], dtype="<i8")
    line = np.asarray([[item[1], item[2]] for item in rows], dtype="<i8")
    external_id = np.asarray([item[3] for item in rows], dtype="<i8")
    boundary_id = np.asarray([item[4] for item in rows], dtype="<i4")
    group_id = np.asarray([item[5] for item in rows], dtype="<i4")
    material_by_group = np.asarray([group.material_id for group in groups], dtype="<i4")
    return BoundaryData(
        line2=np.ascontiguousarray(line),
        boundary_id=np.ascontiguousarray(boundary_id),
        group_id=np.ascontiguousarray(group_id),
        material_id=np.ascontiguousarray(material_by_group[group_id], dtype="<i4"),
        owner_cell_type=np.ones(len(rows), dtype="<u1"),
        owner_cell_local_index=np.ascontiguousarray(owner),
        orientation=np.ones(len(rows), dtype="<i1"),
        external_id=np.ascontiguousarray(external_id),
    )


def _read_boundaries(path: Path, content: bytes) -> list[_BoundaryRecord]:
    required = (
        "boundary_edge_id",
        "node1_id",
        "node2_id",
        "mphtxt_boundary_index",
        "comsol_boundary_id",
    )
    records: list[_BoundaryRecord] = []
    seen_ids: set[int] = set()
    for row_number, row in _csv_rows(path, content, required):
        external_id = _csv_int(row["boundary_edge_id"], path, row_number, "boundary_edge_id")
        if external_id in seen_ids:
            raise ValueError(f"{path}: duplicate boundary_edge_id {external_id}")
        seen_ids.add(external_id)
        records.append(
            _BoundaryRecord(
                external_id,
                _csv_int(row["node1_id"], path, row_number, "node1_id"),
                _csv_int(row["node2_id"], path, row_number, "node2_id"),
                _csv_int(row["comsol_boundary_id"], path, row_number, "comsol_boundary_id"),
            )
        )
    return records


def _read_fields(
    specification: AdapterSpecification,
    mesh: _MeshResult,
    contents: _SourceContents,
) -> _FieldResult:
    names, units, expressions = _read_column_dictionary(
        specification.source.field_columns, contents.field_columns
    )
    fields = specification.fields
    index = _field_column_indices(names, units, fields)
    try:
        field_text = contents.field_values.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("field table is not valid UTF-8") from error
    values = _numeric_csv_matrix(
        field_text, expected_columns=len(names), label=str(specification.source.field_values)
    )
    selected = _select_field_rows(values, fields, index, specification.domain.external_id)
    coordinates = selected[:, [index[name] for name in fields.coordinate_columns]]
    matched, distance = _match_field_rows(mesh, selected, coordinates, fields, index)
    ordered = selected[matched]
    _validate_selected_rows(ordered, fields, index)
    converted: list[FieldData] = []
    transforms: list[dict[str, object]] = []
    for item in fields.mappings:
        raw = ordered[:, [index[name] for name in item.columns]]
        with np.errstate(over="ignore", invalid="ignore"):
            mapped = np.ascontiguousarray(raw * item.scale + item.offset, dtype="<f8")
        if not bool(np.isfinite(mapped).all()):
            raise ValueError(f"field {item.name} conversion produced nonfinite values")
        receipt = _apply_axis_transform(item, mapped, mesh, specification.domain.coordinate_system)
        if receipt is not None:
            transforms.append(receipt)
        converted.append(
            FieldData(
                item.name,
                specification.domain.layout_name,
                "node",
                item.components,
                item.stored_basis,
                mapped,
                item.unit,
            )
        )
    return _FieldResult(
        tuple(converted),
        ordered.shape[0],
        distance,
        "scoped_node_id" if fields.node_id_column is not None else "coordinate_bijection",
        transforms,
        [_field_lineage(item, expressions) for item in fields.mappings],
        values.shape[0],
    )


def _validate_selected_rows(
    ordered: np.ndarray, fields: FieldSpecification, index: dict[str, int]
) -> None:
    selected_columns = [index[name] for item in fields.mappings for name in item.columns]
    if not bool(np.isfinite(ordered[:, selected_columns]).all()):
        raise ValueError("selected domain nodes have missing or partially missing primitives")
    if bool((ordered[:, index[fields.inside_column]] != 1.0).any()):
        raise ValueError("selected domain nodes must be inside the exported model domain")


def _field_lineage(item: FieldMapping, expressions: dict[str, str]) -> dict[str, object]:
    return {
        "name": item.name,
        "columns": list(item.columns),
        "components": list(item.components),
        "stored_basis": item.stored_basis,
        "association": "node",
        "unit": item.unit,
        "input_unit": item.input_unit,
        "expressions": [expressions[column] for column in item.columns],
        "conversion": {
            "formula": "canonical = scale * input + offset",
            "scale": item.scale,
            "offset": item.offset,
        },
        "producer_record": item.producer_record,
    }


def _match_field_rows(
    mesh: _MeshResult,
    selected: np.ndarray,
    coordinates: np.ndarray,
    fields: FieldSpecification,
    index: dict[str, int],
) -> tuple[np.ndarray, float]:
    if fields.node_id_column is None:
        return _match_coordinates(mesh.nodes_m, coordinates, fields.coordinate_match_tolerance_m)
    identifiers = selected[:, index[fields.node_id_column]]
    if not bool(np.isfinite(identifiers).all()) or bool(
        (identifiers != np.floor(identifiers)).any()
    ):
        raise ValueError("field node IDs must be finite integers")
    if bool((np.abs(identifiers) > 2**53).any()):
        raise ValueError("numeric field node IDs must be exactly representable integers")
    provider = identifiers.astype(np.int64)
    order = np.argsort(provider)
    sorted_ids = provider[order]
    if np.unique(sorted_ids).size != sorted_ids.size:
        raise ValueError("field node IDs must be unique for a bijection")
    locations = np.searchsorted(sorted_ids, mesh.node_external_id)
    if bool((locations >= sorted_ids.size).any()):
        raise ValueError("field node IDs cannot form a bijection to selected mesh vertex IDs")
    matched = order[locations]
    if not np.array_equal(provider[matched], mesh.node_external_id):
        raise ValueError("field node IDs cannot form a bijection to selected mesh vertex IDs")
    distance = np.linalg.norm(coordinates[matched] - mesh.nodes_m, axis=1)
    maximum = float(np.max(distance, initial=0.0))
    if maximum > fields.coordinate_match_tolerance_m:
        raise ValueError("field node ID coordinates disagree with the mesh")
    return matched, maximum


def _apply_axis_transform(
    item: FieldMapping,
    mapped: np.ndarray,
    mesh: _MeshResult,
    coordinate_system: str,
) -> dict[str, object] | None:
    if item.axis_zero_component is None:
        return None
    if coordinate_system != "axisymmetric_rz":
        raise ValueError("axis transformation requires RZ geometry")
    component = item.components.index(item.axis_zero_component)
    axis = mesh.nodes_m[:, 0] == 0.0
    original = mapped[axis, component].copy()
    maximum = float(np.max(np.abs(original), initial=0.0))
    if maximum > item.axis_correction_limit:
        raise ValueError(f"field {item.name} axis correction exceeds the explicit limit")
    changed_nodes = np.flatnonzero(axis)[original != 0.0]
    affected_cells = np.flatnonzero(np.isin(mesh.triangles, changed_nodes).any(axis=1))
    mapped[axis, component] = 0.0
    return {
        "field": item.name,
        "component": item.axis_zero_component,
        "unit": item.unit,
        "reason": item.axis_reason,
        "limit": item.axis_correction_limit,
        "maximum_abs_correction": maximum,
        "axis_node_external_ids": mesh.node_external_id[axis].tolist(),
        "raw_axis_values": original.tolist(),
        "changed_node_count": int(changed_nodes.size),
        "affected_p1_cell_indices": affected_cells.tolist(),
        "squared_nodal_correction_sum": float(np.dot(original, original)),
        "off_axis_effect": "P1 interpolation changes inside every listed incident cell",
    }


def _numeric_csv_matrix(text: str, *, expected_columns: int, label: str) -> np.ndarray:
    rows: list[list[float]] = []
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        numbered_rows = [(reader.line_num, row) for row in reader]
    except csv.Error as error:
        raise ValueError(f"{label}: malformed CSV near line {reader.line_num}") from error
    for line_number, row in numbered_rows:
        if not row or all(not cell.strip() for cell in row):
            continue
        if row[0].lstrip().startswith("%"):
            continue
        if len(row) != expected_columns:
            raise ValueError(
                f"{label}: row {line_number} has {len(row)} columns; expected {expected_columns}"
            )
        parsed: list[float] = []
        for column, token in enumerate(row, start=1):
            value = token.strip()
            if not value:
                raise ValueError(f"{label}: row {line_number} column {column} is empty")
            if value.casefold() in {"nan", "+nan", "-nan"}:
                parsed.append(math.nan)
                continue
            if _FINITE_NUMBER_TOKEN.fullmatch(value) is None:
                raise ValueError(f"{label}: row {line_number} column {column} is not numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{label}: row {line_number} column {column} is not finite")
            parsed.append(number)
        rows.append(parsed)
    if not rows:
        raise ValueError(f"{label}: numeric table has no data rows")
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float64))


def _field_column_indices(
    names: list[str], units: dict[str, str], fields: FieldSpecification
) -> dict[str, int]:
    required_units = list(zip(fields.coordinate_columns, ("m", "m"), strict=True))
    required_units.append((fields.inside_column, "1"))
    if fields.node_id_column is not None:
        required_units.append((fields.node_id_column, "1"))
    if fields.scope_column is not None:
        scope_unit = (
            "1" if fields.scope_mode == "equals_domain_id" else units.get(fields.scope_column, "")
        )
        required_units.append((fields.scope_column, scope_unit))
    required_units.extend(
        (column, item.input_unit) for item in fields.mappings for column in item.columns
    )
    index = dict(zip(names, range(len(names)), strict=True))
    missing = set(dict(required_units)).difference(index)
    if missing:
        raise ValueError(f"field column dictionary is missing columns: {sorted(missing)}")
    wrong_units = {
        name: (units[name], expected)
        for name, expected in required_units
        if units[name] != expected
    }
    if wrong_units:
        raise ValueError(f"field column units disagree with selected input units: {wrong_units}")
    return index


def _select_field_rows(
    values: np.ndarray, fields: FieldSpecification, index: dict[str, int], domain_id: int
) -> np.ndarray:
    coordinates = values[:, [index[name] for name in fields.coordinate_columns]]
    selected = np.isfinite(coordinates).all(axis=1)
    if fields.scope_column is not None:
        scope_values = values[:, index[fields.scope_column]]
        scope_mask = (
            np.isfinite(scope_values)
            if fields.scope_mode == "finite"
            else scope_values == domain_id
        )
        selected &= scope_mask
    return values[selected]


def _read_column_dictionary(
    path: Path, content: bytes
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    names: list[str] = []
    units: dict[str, str] = {}
    expressions: dict[str, str] = {}
    for row_number, row in _csv_rows(path, content, ("column", "COMSOL_expression", "unit")):
        name = row["column"]
        if not name or name in units:
            raise ValueError(f"{path}:{row_number}: duplicate or empty column name")
        names.append(name)
        units[name] = row["unit"]
        expressions[name] = row["COMSOL_expression"]
    if not names:
        raise ValueError(f"{path}: empty field column dictionary")
    return names, units, expressions


def _match_coordinates(
    canonical: np.ndarray, provider: np.ndarray, tolerance_m: float
) -> tuple[np.ndarray, float]:
    if np.unique(provider, axis=0).shape[0] != provider.shape[0]:
        raise ValueError("selected provider coordinates are not unique")
    match = np.empty(canonical.shape[0], dtype=np.int64)
    distance2 = np.empty(canonical.shape[0], dtype=np.float64)
    tolerance2 = tolerance_m * tolerance_m
    for start in range(0, canonical.shape[0], 256):
        end = min(start + 256, canonical.shape[0])
        delta = canonical[start:end, None, :] - provider[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        within = np.count_nonzero(squared <= tolerance2, axis=1)
        if bool((within != 1).any()):
            raise ValueError(
                "each canonical node must have exactly one provider coordinate within tolerance"
            )
        local_match = np.argmin(squared, axis=1)
        match[start:end] = local_match
        distance2[start:end] = squared[np.arange(end - start), local_match]
    maximum = float(np.sqrt(np.max(distance2, initial=0.0)))
    if maximum > tolerance_m:
        raise ValueError(f"provider coordinates exceed match tolerance: {maximum} > {tolerance_m}")
    if np.unique(match).size != match.size:
        raise ValueError("provider-to-canonical coordinate match is not bijective")
    return match, maximum


def _snapshot_sources(
    source: SourceFiles,
) -> tuple[_SourceContents, dict[str, str], str]:
    paths = {
        "vertices": source.vertices,
        "triangles": source.triangles,
        "quadrilaterals": source.quadrilaterals,
        "boundary_edges": source.boundary_edges,
        "field_values": source.field_values,
        "field_columns": source.field_columns,
    }
    aggregate = hashlib.sha256()
    result: dict[str, str] = {}
    snapshots: dict[str, bytes] = {}
    for name, path in sorted(paths.items()):
        content = path.read_bytes()
        snapshots[name] = content
        digest = hashlib.sha256(content).hexdigest()
        result[name] = f"sha256:{digest}"
        encoded = name.encode("utf-8")
        aggregate.update(len(encoded).to_bytes(4, "little"))
        aggregate.update(encoded)
        aggregate.update(bytes.fromhex(digest))
    return (
        _SourceContents(
            vertices=snapshots["vertices"],
            triangles=snapshots["triangles"],
            quadrilaterals=snapshots["quadrilaterals"],
            boundary_edges=snapshots["boundary_edges"],
            field_values=snapshots["field_values"],
            field_columns=snapshots["field_columns"],
        ),
        result,
        f"sha256:{aggregate.hexdigest()}",
    )


def _provenance(
    *,
    config_sha256: str,
    aggregate_source_hash: str,
    source_hashes: dict[str, str],
    specification: AdapterSpecification,
    mesh: _MeshResult,
    fields: _FieldResult,
) -> dict[str, object]:
    return {
        "producer": "chamber_particles.tools.comsol_adapter",
        "producer_version": PRODUCER_VERSION,
        "source_sha256": aggregate_source_hash,
        "field_semantics_revision": FIELD_SEMANTICS_REVISION,
        "producer_metadata": {
            "adapter_revision": ADAPTER_REVISION,
            "configuration_sha256": config_sha256,
            "source_file_hashes": source_hashes,
            "domain_external_id": specification.domain.external_id,
            "layout": specification.domain.layout_name,
            "quadrilateral_order": specification.domain.quadrilateral_order,
            "excluded_boundary_ids": list(specification.domain.excluded_boundary_ids),
            "boundary_groups": [
                {
                    "name": group.name,
                    "external_ids": list(group.external_ids),
                    "material_id": group.material_id,
                }
                for group in specification.boundary_groups
            ],
            "source_triangle_count": mesh.source_triangle_count,
            "source_quadrilateral_count": mesh.source_quadrilateral_count,
            "p1_triangle_count": int(mesh.triangles.shape[0]),
            "split_diagonal_03_count": mesh.split_diagonal_03_count,
            "split_diagonal_12_count": mesh.split_diagonal_12_count,
            "minimum_triangle_quality": mesh.minimum_triangle_quality,
            "excluded_boundary_count": mesh.excluded_boundary_count,
            "group_facet_counts": mesh.group_counts,
            "coordinate_match_tolerance_m": specification.fields.coordinate_match_tolerance_m,
            "maximum_coordinate_match_distance_m": fields.maximum_match_distance_m,
            "coordinate_system": specification.domain.coordinate_system,
            "field_join_method": fields.join_method,
            "selected_field_lineage": fields.selected_field_lineage,
            "provider_field_row_count": fields.provider_row_count,
            "selected_domain_field_row_count": fields.selected_row_count,
            "provider_domain_scope": {
                "column": specification.fields.scope_column,
                "mode": specification.fields.scope_mode,
                "evidence": specification.fields.scope_evidence,
            },
            "axis_transformations": fields.axis_transformations,
        },
    }


def _report(
    specification: AdapterSpecification,
    mesh: _MeshResult,
    fields: _FieldResult,
    output: Path,
    output_hash: str,
    source_hash: str,
) -> dict[str, object]:
    return {
        "status": "complete",
        "scope": "selected-field conversion and canonical validation only",
        "adapter_revision": ADAPTER_REVISION,
        "source_sha256": source_hash,
        "output_path": str(output),
        "output_content_hash": output_hash,
        "domain_external_id": specification.domain.external_id,
        "layout": specification.domain.layout_name,
        "node_count": int(mesh.nodes_m.shape[0]),
        "source_triangle_count": mesh.source_triangle_count,
        "source_quadrilateral_count": mesh.source_quadrilateral_count,
        "p1_triangle_count": int(mesh.triangles.shape[0]),
        "split_diagonal_03_count": mesh.split_diagonal_03_count,
        "split_diagonal_12_count": mesh.split_diagonal_12_count,
        "minimum_triangle_quality": mesh.minimum_triangle_quality,
        "physical_boundary_facet_count": int(mesh.boundary.line2.shape[0]),
        "excluded_boundary_facet_count": mesh.excluded_boundary_count,
        "boundary_group_facet_counts": mesh.group_counts,
        "selected_field_row_count": fields.selected_row_count,
        "provider_field_row_count": fields.provider_row_count,
        "unselected_provider_row_count": fields.provider_row_count - fields.selected_row_count,
        "maximum_coordinate_match_distance_m": fields.maximum_match_distance_m,
        "field_join_method": fields.join_method,
        "axis_transformations": fields.axis_transformations,
        "selected_field_lineage": fields.selected_field_lineage,
        "fields": [field.name for field in fields.fields],
    }


def _csv_rows(
    path: Path, content: bytes, required: tuple[str, ...]
) -> list[tuple[int, dict[str, str]]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: CSV is not valid UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if reader.fieldnames is None or tuple(reader.fieldnames) != required:
        raise ValueError(f"{path}: columns must be exactly {list(required)}")
    result: list[tuple[int, dict[str, str]]] = []
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"{path}: malformed CSV near line {reader.line_num}") from error
    for index, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}:{index}: CSV row width does not match the header")
        result.append((index, cast(dict[str, str], dict(row))))
    return result


def _csv_int(value: str, path: Path, row: int, column: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{path}:{row}: {column} must be an integer") from error


def _csv_float(value: str, path: Path, row: int, column: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{row}: {column} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{path}:{row}: {column} must be finite")
    return result


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    return cast(dict[str, object], value)


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty string without NUL")
    return value


def _name(value: object, label: str) -> str:
    text = _text(value, label)
    if not text[0].isalpha() or not all(
        character.isalnum() or character == "_" for character in text
    ):
        raise ValueError(f"{label} must be an identifier-like name")
    return text


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result = tuple(_integer(item, f"{label}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _name_pair(value: object, label: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two names")
    return _name(value[0], f"{label}[0]"), _name(value[1], f"{label}[1]")


def _names(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    result = tuple(_name(item, f"{label}[]") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must contain unique names")
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_number(value: object, label: str) -> float:
    result = _number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_number(value: object, label: str) -> float:
    result = _number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result
