"""Canonical, producer-neutral ``case.h5`` schema v1.

This module owns the durable representation of reusable geometry, fields, and
realized particle sources.  It deliberately does not know about run settings,
COMSOL, or trajectory integration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import h5py
import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION: Final = 1
CONTENT_HASH_PREFIX: Final = b"chamber-particles-case\0v1\0"

_F8: Final = np.dtype("<f8")
_I8: Final = np.dtype("<i8")
_I4: Final = np.dtype("<i4")
_U1: Final = np.dtype("<u1")
_I1: Final = np.dtype("<i1")
_UTF8: Final = h5py.string_dtype(encoding="utf-8")
_NAME_RE: Final = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_SHA256_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type Int32Array = NDArray[np.int32]
type UInt8Array = NDArray[np.uint8]
type Int8Array = NDArray[np.int8]
type CoordinateSystem = Literal["cartesian_xy", "axisymmetric_rz"]


@dataclass(frozen=True, slots=True)
class BoundaryData:
    """Oriented 2-D boundary facets and their owning volume cells."""

    line2: Int64Array
    boundary_id: Int32Array
    group_id: Int32Array
    material_id: Int32Array
    owner_cell_type: UInt8Array
    owner_cell_local_index: Int64Array
    orientation: Int8Array
    external_id: Int64Array | None = None


@dataclass(frozen=True, slots=True)
class GeometryData:
    """Particle-domain topology in SI coordinates."""

    nodes_m: FloatArray
    boundary: BoundaryData
    group_names: tuple[str, ...]
    node_external_id: Int64Array | None = None
    tri3: Int64Array | None = None
    tri3_domain_id: Int32Array | None = None
    quad4: Int64Array | None = None
    quad4_domain_id: Int32Array | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_names", tuple(self.group_names))


@dataclass(frozen=True, slots=True)
class RegularLayout:
    """Rectilinear two-dimensional field layout."""

    name: str
    axis0_m: FloatArray
    axis1_m: FloatArray
    cell_support: UInt8Array


@dataclass(frozen=True, slots=True)
class P1TriLayout:
    """Linear triangular field layout."""

    name: str
    nodes_m: FloatArray
    connectivity: Int64Array
    cell_support: UInt8Array


@dataclass(frozen=True, slots=True)
class Q1QuadLayout:
    """Bilinear quadrilateral field layout in reference-node order."""

    name: str
    nodes_m: FloatArray
    connectivity: Int64Array
    cell_support: UInt8Array


type Layout = RegularLayout | P1TriLayout | Q1QuadLayout


@dataclass(frozen=True, slots=True)
class FieldData:
    """One static primitive field attached to a named layout."""

    name: str
    layout: str
    association: Literal["node", "cell"]
    components: tuple[str, ...]
    stored_basis: str
    values: FloatArray
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))


@dataclass(frozen=True, slots=True)
class RealizedTableSource:
    """A fully realized particle release table with no hidden distributions."""

    name: str
    particle_id: Int64Array
    release_time_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    charge_number: FloatArray
    mass_kg: FloatArray
    drag_diameter_m: FloatArray
    electrostatic_radius_m: FloatArray
    displaced_volume_m3: FloatArray
    model_weight: FloatArray
    material_id: Int32Array


@dataclass(frozen=True, slots=True)
class DataBundle:
    """Canonical reusable facts stored in one ``case.h5`` file."""

    coordinate_system: CoordinateSystem
    provenance_json: str
    geometry: GeometryData
    layouts: tuple[Layout, ...] = ()
    fields: tuple[FieldData, ...] = ()
    sources: tuple[RealizedTableSource, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance_json", _canonical_provenance(self.provenance_json))
        object.__setattr__(self, "layouts", tuple(self.layouts))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True, slots=True)
class CanonicalDataFootprint:
    """Cheap logical size of the numeric arrays stored in one canonical file."""

    numeric_array_bytes: int


@dataclass(frozen=True, slots=True)
class CaseFileInfo:
    """Identity returned after a canonical file is successfully published."""

    schema_version: int
    content_hash: str
    footprint: CanonicalDataFootprint


@dataclass(frozen=True, slots=True)
class _DatasetRecord:
    path: str
    dtype_tag: str
    shape: tuple[int, ...]
    numeric: NDArray[Any] | None = None
    strings: tuple[str, ...] = ()


def content_hash(data: DataBundle) -> str:
    """Return the schema-v1 logical SHA-256 for *data*.

    The digest is independent of HDF5 allocation, chunking, and creation order.
    Floating-point bytes are preserved exactly, including negative zero.
    """

    _validate_bundle(data)
    return _content_hash_validated(data)


def resident_array_bytes(data: DataBundle) -> int:
    """Count unique NumPy allocations retained by one canonical data bundle."""

    geometry = data.geometry
    boundary = geometry.boundary
    arrays: list[np.ndarray] = [
        geometry.nodes_m,
        boundary.line2,
        boundary.boundary_id,
        boundary.group_id,
        boundary.material_id,
        boundary.owner_cell_type,
        boundary.owner_cell_local_index,
        boundary.orientation,
    ]
    arrays.extend(
        array
        for array in (
            geometry.node_external_id,
            geometry.tri3,
            geometry.tri3_domain_id,
            geometry.quad4,
            geometry.quad4_domain_id,
            boundary.external_id,
        )
        if array is not None
    )
    for layout in data.layouts:
        if isinstance(layout, RegularLayout):
            arrays.extend((layout.axis0_m, layout.axis1_m, layout.cell_support))
        else:
            arrays.extend((layout.nodes_m, layout.connectivity, layout.cell_support))
    arrays.extend(field.values for field in data.fields)
    for source in data.sources:
        arrays.extend(
            (
                source.particle_id,
                source.release_time_s,
                source.position_m,
                source.velocity_m_s,
                source.charge_number,
                source.mass_kg,
                source.drag_diameter_m,
                source.electrostatic_radius_m,
                source.displaced_volume_m3,
                source.model_weight,
                source.material_id,
            )
        )
    return _unique_allocation_bytes(arrays)


def _unique_allocation_bytes(arrays: list[np.ndarray]) -> int:
    total = 0
    seen: set[int] = set()
    for array in arrays:
        allocation = array
        while isinstance(allocation.base, np.ndarray):
            allocation = allocation.base
        identity = id(allocation)
        if identity not in seen:
            seen.add(identity)
            total += int(allocation.nbytes)
    return total


def _content_hash_validated(data: DataBundle) -> str:
    digest = hashlib.sha256()
    digest.update(CONTENT_HASH_PREFIX)
    for record in sorted(_logical_records(data), key=lambda item: item.path):
        _hash_record(digest, record)
    return f"sha256:{digest.hexdigest()}"


def _numeric_array_bytes(data: DataBundle) -> int:
    return sum(
        int(record.numeric.size) * int(record.numeric.dtype.itemsize)
        for record in _logical_records(data)
        if record.numeric is not None
    )


def write(path: str | os.PathLike[str], data: DataBundle) -> CaseFileInfo:
    """Atomically publish *data* without replacing an existing destination.

    A complete sibling temporary file is linked to *path* in one filesystem
    operation.  Existing destinations and filesystems without hard-link support
    raise :class:`OSError`; there is intentionally no non-atomic fallback.
    """

    destination = Path(path)
    logical_hash = content_hash(data)
    file_info = CaseFileInfo(
        SCHEMA_VERSION,
        logical_hash,
        CanonicalDataFootprint(_numeric_array_bytes(data)),
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            _write_bundle(handle, data)
            handle.flush()
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return file_info


def read(path: str | os.PathLike[str]) -> DataBundle:
    """Read and strictly validate one canonical schema-v1 file."""

    data, _footprint = _read_validated(path)
    return data


def read_with_info(
    path: str | os.PathLike[str],
    *,
    numeric_array_limit_bytes: int | None = None,
) -> tuple[DataBundle, CaseFileInfo]:
    """Read once and return both validated data and its logical identity."""

    data, footprint = _read_validated(path, numeric_array_limit_bytes=numeric_array_limit_bytes)
    return data, CaseFileInfo(SCHEMA_VERSION, _content_hash_validated(data), footprint)


def _read_validated(
    path: str | os.PathLike[str],
    *,
    numeric_array_limit_bytes: int | None = None,
) -> tuple[DataBundle, CanonicalDataFootprint]:
    """Own the single HDF5 read and validation path used by public readers."""

    if numeric_array_limit_bytes is not None and numeric_array_limit_bytes < 0:
        raise ValueError("numeric_array_limit_bytes must be nonnegative")
    with h5py.File(path, "r") as handle:
        footprint = _validate_hdf5_objects(handle)
        if (
            numeric_array_limit_bytes is not None
            and footprint.numeric_array_bytes > numeric_array_limit_bytes
        ):
            raise ValueError(
                "canonical numeric arrays require "
                f"{footprint.numeric_array_bytes} bytes, exceeding "
                f"resources.memory_limit_mb ({numeric_array_limit_bytes} bytes)"
            )
        data = _read_bundle(handle)
    _validate_bundle(data)
    _mark_arrays_read_only(data)
    return data, footprint


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate provenance key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not canonical: {value}")


def _canonical_provenance(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("provenance_json must be a JSON string")
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("provenance_json is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("provenance_json must contain a JSON object")
    _validate_provenance(parsed)
    return json.dumps(
        parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _validate_provenance(provenance: Mapping[str, object]) -> None:
    required = {
        "producer",
        "producer_version",
        "source_sha256",
        "field_semantics_revision",
        "producer_metadata",
    }
    missing = required.difference(provenance)
    if missing:
        raise ValueError(f"provenance_json is missing keys: {sorted(missing)}")
    for key in ("producer", "producer_version", "field_semantics_revision"):
        _require_text(provenance[key], f"provenance.{key}")
    source_hash = provenance["source_sha256"]
    if not isinstance(source_hash, str) or _SHA256_RE.fullmatch(source_hash) is None:
        raise ValueError("provenance.source_sha256 must be lowercase sha256:<64 hex>")
    if not isinstance(provenance["producer_metadata"], dict):
        raise ValueError("provenance.producer_metadata must be an object")


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a nonempty UTF-8 string without NUL")
    return value


def _require_name(value: object, label: str) -> str:
    text = _require_text(value, label)
    if _NAME_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must match {_NAME_RE.pattern}")
    return text


def _require_array(
    value: object,
    dtype: np.dtype[Any],
    ndim: int,
    label: str,
) -> NDArray[Any]:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{label} must be a numpy array")
    if value.dtype.str != dtype.str:
        raise ValueError(f"{label} must have dtype {dtype.str}, got {value.dtype.str}")
    if value.ndim != ndim:
        raise ValueError(f"{label} must have rank {ndim}, got {value.ndim}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{label} must be C-contiguous")
    return value


def _require_finite(value: NDArray[Any], label: str) -> None:
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{label} contains a non-finite value")


def _require_binary_mask(value: NDArray[Any], label: str) -> None:
    if not bool(np.logical_or(value == 0, value == 1).all()):
        raise ValueError(f"{label} must contain only 0 or 1")


def _validate_bundle(data: DataBundle) -> None:
    if data.coordinate_system not in ("cartesian_xy", "axisymmetric_rz"):
        raise ValueError(f"unsupported coordinate_system: {data.coordinate_system}")
    if _canonical_provenance(data.provenance_json) != data.provenance_json:
        raise ValueError("provenance_json is not canonical")
    _validate_geometry(data.geometry, data.coordinate_system)
    layouts = _validate_layouts(data.layouts, data.coordinate_system)
    _validate_fields(data.fields, layouts)
    _validate_sources(data.sources, data.coordinate_system)


def _validate_geometry(geometry: GeometryData, coordinate_system: CoordinateSystem) -> None:
    nodes = _require_array(geometry.nodes_m, _F8, 2, "geometry.nodes_m")
    if nodes.shape[1:] != (2,) or nodes.shape[0] == 0:
        raise ValueError("geometry.nodes_m must have shape [N, 2] with N > 0")
    _require_finite(nodes, "geometry.nodes_m")
    if coordinate_system == "axisymmetric_rz" and bool((nodes[:, 0] < 0).any()):
        raise ValueError("axisymmetric geometry requires r >= 0")
    _validate_optional_ids(geometry.node_external_id, nodes.shape[0], "geometry.node_external_id")
    tri3 = _validate_cell_block(nodes, geometry.tri3, geometry.tri3_domain_id, 3, "tri3")
    quad4 = _validate_cell_block(nodes, geometry.quad4, geometry.quad4_domain_id, 4, "quad4")
    if tri3 is None and quad4 is None:
        raise ValueError("geometry must contain at least one tri3 or quad4 cell")
    if tri3 is not None:
        _validate_ccw_triangles(nodes, tri3, "geometry.tri3")
    if quad4 is not None:
        _validate_q1_quads(nodes, quad4, "geometry.quad4")
    _validate_boundary(geometry.boundary, geometry.group_names, nodes.shape[0], tri3, quad4)


def _validate_optional_ids(value: Int64Array | None, length: int, label: str) -> None:
    if value is None:
        return
    array = _require_array(value, _I8, 1, label)
    if array.shape != (length,):
        raise ValueError(f"{label} must have shape [{length}]")
    if np.unique(array).size != array.size:
        raise ValueError(f"{label} values must be unique")


def _validate_cell_block(
    nodes: FloatArray,
    connectivity: Int64Array | None,
    domain_id: Int32Array | None,
    arity: int,
    name: str,
) -> Int64Array | None:
    if (connectivity is None) != (domain_id is None):
        raise ValueError(f"geometry.{name} and geometry.{name}_domain_id must appear together")
    if connectivity is None or domain_id is None:
        return None
    cells = _require_array(connectivity, _I8, 2, f"geometry.{name}")
    domains = _require_array(domain_id, _I4, 1, f"geometry.{name}_domain_id")
    if cells.shape[1:] != (arity,) or cells.shape[0] == 0:
        raise ValueError(f"geometry.{name} must have shape [N, {arity}] with N > 0")
    if domains.shape != (cells.shape[0],):
        raise ValueError(f"geometry.{name}_domain_id length does not match cells")
    _validate_connectivity(cells, nodes.shape[0], f"geometry.{name}")
    return cells


def _validate_connectivity(connectivity: Int64Array, node_count: int, label: str) -> None:
    if bool((connectivity < 0).any()) or bool((connectivity >= node_count).any()):
        raise ValueError(f"{label} references a node outside geometry")
    sorted_nodes = np.sort(connectivity, axis=1)
    if bool((np.diff(sorted_nodes, axis=1) == 0).any()):
        raise ValueError(f"{label} contains a cell with repeated nodes")


def _cross_2d(first: FloatArray, second: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def _validate_ccw_triangles(nodes: FloatArray, connectivity: Int64Array, label: str) -> None:
    points = nodes[connectivity]
    with np.errstate(over="ignore", invalid="ignore"):
        signed_double_area = _cross_2d(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    if bool((~np.isfinite(signed_double_area) | (signed_double_area <= 0)).any()):
        raise ValueError(f"{label} must use nondegenerate counter-clockwise node order")


def _validate_q1_quads(nodes: FloatArray, connectivity: Int64Array, label: str) -> None:
    points = nodes[connectivity]
    with np.errstate(over="ignore", invalid="ignore"):
        determinants = np.column_stack(
            (
                _cross_2d(points[:, 1] - points[:, 0], points[:, 3] - points[:, 0]),
                _cross_2d(points[:, 1] - points[:, 0], points[:, 2] - points[:, 1]),
                _cross_2d(points[:, 2] - points[:, 3], points[:, 2] - points[:, 1]),
                _cross_2d(points[:, 2] - points[:, 3], points[:, 3] - points[:, 0]),
            )
        )
    if bool((~np.isfinite(determinants) | (determinants <= 0)).any()):
        raise ValueError(f"{label} must use a non-inverted Q1 reference-node order")


def _validate_boundary(
    boundary: BoundaryData,
    group_names: tuple[str, ...],
    node_count: int,
    tri3: Int64Array | None,
    quad4: Int64Array | None,
) -> None:
    line2 = _require_array(boundary.line2, _I8, 2, "geometry.boundary.line2")
    if line2.shape[1:] != (2,):
        raise ValueError("geometry.boundary.line2 must have shape [N, 2]")
    _validate_connectivity(line2, node_count, "geometry.boundary.line2")
    count = line2.shape[0]
    arrays = _boundary_arrays(boundary)
    for label, array, dtype in arrays:
        checked = _require_array(array, dtype, 1, label)
        if checked.shape != (count,):
            raise ValueError(f"{label} length does not match line2")
    _validate_optional_ids(boundary.external_id, count, "geometry.boundary.external_id")
    _validate_group_names(boundary.group_id, group_names, count)
    if not bool(np.logical_or(boundary.orientation == 1, boundary.orientation == -1).all()):
        raise ValueError("geometry.boundary.orientation must contain only +1 or -1")
    _validate_boundary_owners(boundary, tri3, quad4)


def _boundary_arrays(
    boundary: BoundaryData,
) -> tuple[tuple[str, NDArray[Any], np.dtype[Any]], ...]:
    return (
        ("geometry.boundary.boundary_id", boundary.boundary_id, _I4),
        ("geometry.boundary.group_id", boundary.group_id, _I4),
        ("geometry.boundary.material_id", boundary.material_id, _I4),
        ("geometry.boundary.owner_cell_type", boundary.owner_cell_type, _U1),
        ("geometry.boundary.owner_cell_local_index", boundary.owner_cell_local_index, _I8),
        ("geometry.boundary.orientation", boundary.orientation, _I1),
    )


def _validate_group_names(group_id: Int32Array, names: tuple[str, ...], count: int) -> None:
    for index, name in enumerate(names):
        _require_text(name, f"geometry.group_names[{index}]")
    if len(set(names)) != len(names):
        raise ValueError("geometry.group_names must be unique")
    expected = np.arange(len(names), dtype=_I4)
    actual = np.unique(group_id)
    if count == 0 and names:
        raise ValueError("boundary group names must be empty when there are no facets")
    if count > 0 and not np.array_equal(actual, expected):
        raise ValueError("boundary group IDs must densely cover 0..len(group_names)-1")


def _owner_nodes(
    cell_type: int,
    local_index: int,
    tri3: Int64Array | None,
    quad4: Int64Array | None,
) -> Int64Array:
    if cell_type == 1 and tri3 is not None and 0 <= local_index < tri3.shape[0]:
        return tri3[local_index]
    if cell_type == 2 and quad4 is not None and 0 <= local_index < quad4.shape[0]:
        return quad4[local_index]
    raise ValueError("boundary owner cell type/index does not reference an existing cell")


def _edge_orientation(line: Int64Array, owner: Int64Array) -> int:
    for index, first in enumerate(owner):
        second = owner[(index + 1) % owner.size]
        if line[0] == first and line[1] == second:
            return 1
        if line[0] == second and line[1] == first:
            return -1
    raise ValueError("boundary line2 is not an edge of its owner cell")


def _validate_boundary_owners(
    boundary: BoundaryData,
    tri3: Int64Array | None,
    quad4: Int64Array | None,
) -> None:
    for index, line in enumerate(boundary.line2):
        owner = _owner_nodes(
            int(boundary.owner_cell_type[index]),
            int(boundary.owner_cell_local_index[index]),
            tri3,
            quad4,
        )
        expected = _edge_orientation(line, owner)
        if int(boundary.orientation[index]) != expected:
            raise ValueError("boundary orientation disagrees with the owner cell edge order")


def _validate_layouts(
    layouts: tuple[Layout, ...], coordinate_system: CoordinateSystem
) -> dict[str, Layout]:
    result: dict[str, Layout] = {}
    for layout in layouts:
        _require_name(layout.name, "layout name")
        if layout.name in result:
            raise ValueError(f"duplicate layout name: {layout.name}")
        _validate_layout(layout, coordinate_system)
        result[layout.name] = layout
    return result


def _validate_layout(layout: Layout, coordinate_system: CoordinateSystem) -> None:
    if isinstance(layout, RegularLayout):
        _validate_regular_layout(layout, coordinate_system)
    elif isinstance(layout, P1TriLayout):
        _validate_unstructured_layout(layout, 3, coordinate_system)
    elif isinstance(layout, Q1QuadLayout):
        _validate_unstructured_layout(layout, 4, coordinate_system)
    else:
        raise ValueError(f"unsupported layout type: {type(layout).__name__}")


def _validate_regular_layout(layout: RegularLayout, coordinate_system: CoordinateSystem) -> None:
    axis0 = _require_array(layout.axis0_m, _F8, 1, f"layouts.{layout.name}.axis0_m")
    axis1 = _require_array(layout.axis1_m, _F8, 1, f"layouts.{layout.name}.axis1_m")
    support = _require_array(layout.cell_support, _U1, 2, f"layouts.{layout.name}.cell_support")
    if axis0.size < 2 or axis1.size < 2:
        raise ValueError("regular layout axes must each contain at least two values")
    _require_finite(axis0, f"layouts.{layout.name}.axis0_m")
    _require_finite(axis1, f"layouts.{layout.name}.axis1_m")
    if bool((np.diff(axis0) <= 0).any()) or bool((np.diff(axis1) <= 0).any()):
        raise ValueError("regular layout axes must be strictly increasing")
    if coordinate_system == "axisymmetric_rz" and bool((axis0 < 0).any()):
        raise ValueError("axisymmetric regular layout requires radial axis0_m >= 0")
    if support.shape != (axis0.size - 1, axis1.size - 1):
        raise ValueError("regular cell_support shape must match the intervals of both axes")
    _require_binary_mask(support, f"layouts.{layout.name}.cell_support")


def _validate_unstructured_layout(
    layout: P1TriLayout | Q1QuadLayout,
    arity: int,
    coordinate_system: CoordinateSystem,
) -> None:
    nodes = _require_array(layout.nodes_m, _F8, 2, f"layouts.{layout.name}.nodes_m")
    cells = _require_array(layout.connectivity, _I8, 2, f"layouts.{layout.name}.connectivity")
    support = _require_array(layout.cell_support, _U1, 1, f"layouts.{layout.name}.cell_support")
    if nodes.shape[1:] != (2,) or nodes.shape[0] == 0:
        raise ValueError("unstructured layout nodes_m must have shape [N, 2] with N > 0")
    if cells.shape[1:] != (arity,) or cells.shape[0] == 0:
        raise ValueError(f"unstructured connectivity must have shape [N, {arity}] with N > 0")
    if support.shape != (cells.shape[0],):
        raise ValueError("unstructured cell_support length must match connectivity")
    _require_finite(nodes, f"layouts.{layout.name}.nodes_m")
    _validate_connectivity(cells, nodes.shape[0], f"layouts.{layout.name}.connectivity")
    _require_binary_mask(support, f"layouts.{layout.name}.cell_support")
    if coordinate_system == "axisymmetric_rz" and bool((nodes[:, 0] < 0).any()):
        raise ValueError("axisymmetric unstructured layout requires r >= 0")
    if arity == 3:
        _validate_ccw_triangles(nodes, cells, f"layouts.{layout.name}.connectivity")
    else:
        _validate_q1_quads(nodes, cells, f"layouts.{layout.name}.connectivity")


def _validate_fields(fields: tuple[FieldData, ...], layouts: Mapping[str, Layout]) -> None:
    names: set[str] = set()
    for field in fields:
        _require_name(field.name, "field name")
        if field.name in names:
            raise ValueError(f"duplicate field name: {field.name}")
        names.add(field.name)
        _validate_field(field, layouts)


def _validate_field(field: FieldData, layouts: Mapping[str, Layout]) -> None:
    if field.layout not in layouts:
        raise ValueError(f"field {field.name} references unknown layout {field.layout}")
    if field.association not in ("node", "cell"):
        raise ValueError(f"field {field.name} has invalid association")
    if not field.components or len(set(field.components)) != len(field.components):
        raise ValueError(f"field {field.name} components must be nonempty and unique")
    for index, component in enumerate(field.components):
        _require_name(component, f"field {field.name} component {index}")
    _require_text(field.stored_basis, f"field {field.name} stored_basis")
    _require_text(field.unit, f"field {field.name} unit")
    values = _require_array(field.values, _F8, 2, f"fields.{field.name}.values")
    expected_rows = _layout_value_count(layouts[field.layout], field.association)
    if values.shape != (expected_rows, len(field.components)):
        raise ValueError(f"field {field.name} values shape does not match its layout/components")
    _require_finite(values, f"fields.{field.name}.values")


def _layout_value_count(layout: Layout, association: str) -> int:
    if isinstance(layout, RegularLayout):
        if association == "node":
            return layout.axis0_m.size * layout.axis1_m.size
        return (layout.axis0_m.size - 1) * (layout.axis1_m.size - 1)
    if association == "node":
        return int(layout.nodes_m.shape[0])
    return int(layout.connectivity.shape[0])


def _validate_sources(
    sources: tuple[RealizedTableSource, ...], coordinate_system: CoordinateSystem
) -> None:
    names: set[str] = set()
    for source in sources:
        _require_name(source.name, "source name")
        if source.name in names:
            raise ValueError(f"duplicate source name: {source.name}")
        names.add(source.name)
        _validate_source(source, coordinate_system)
    _validate_global_particle_ids(sources)


def _validate_global_particle_ids(sources: tuple[RealizedTableSource, ...]) -> None:
    if not sources:
        return
    particle_ids = (
        sources[0].particle_id
        if len(sources) == 1
        else np.concatenate(tuple(source.particle_id for source in sources))
    )
    if np.unique(particle_ids).size != particle_ids.size:
        raise ValueError("particle_id must be unique across all realized table sources")


def _validate_source(source: RealizedTableSource, coordinate_system: CoordinateSystem) -> None:
    particle_id = _require_array(source.particle_id, _I8, 1, f"sources.{source.name}.particle_id")
    count = particle_id.size
    if count == 0 or bool((particle_id < 0).any()):
        raise ValueError("source particle_id must be nonnegative and nonempty")
    material_id = _require_array(source.material_id, _I4, 1, f"sources.{source.name}.material_id")
    if material_id.shape != (count,):
        raise ValueError("source material_id length does not match particle_id")
    if bool((material_id < 0).any()):
        raise ValueError("source material_id must be nonnegative")
    position = _validate_source_vectors(source, count)
    scalars = _source_scalar_arrays(source)
    for label, array in scalars:
        checked = _require_array(array, _F8, 1, f"sources.{source.name}.{label}")
        if checked.shape != (count,):
            raise ValueError(f"source {label} length does not match particle_id")
        _require_finite(checked, f"sources.{source.name}.{label}")
    _validate_source_ranges(source)
    if coordinate_system == "axisymmetric_rz" and bool((position[:, 0] < 0).any()):
        raise ValueError("axisymmetric source positions require r >= 0")


def _validate_source_vectors(source: RealizedTableSource, count: int) -> FloatArray:
    position = _require_array(source.position_m, _F8, 2, f"sources.{source.name}.position_m")
    velocity = _require_array(source.velocity_m_s, _F8, 2, f"sources.{source.name}.velocity_m_s")
    if position.shape != (count, 2) or velocity.shape != (count, 2):
        raise ValueError("source position_m and velocity_m_s must have shape [N, 2]")
    _require_finite(position, f"sources.{source.name}.position_m")
    _require_finite(velocity, f"sources.{source.name}.velocity_m_s")
    return position


def _source_scalar_arrays(source: RealizedTableSource) -> tuple[tuple[str, FloatArray], ...]:
    return (
        ("release_time_s", source.release_time_s),
        ("charge_number", source.charge_number),
        ("mass_kg", source.mass_kg),
        ("drag_diameter_m", source.drag_diameter_m),
        ("electrostatic_radius_m", source.electrostatic_radius_m),
        ("displaced_volume_m3", source.displaced_volume_m3),
        ("model_weight", source.model_weight),
    )


def _validate_source_ranges(source: RealizedTableSource) -> None:
    if bool((source.mass_kg <= 0).any()) or bool((source.drag_diameter_m <= 0).any()):
        raise ValueError("source mass_kg and drag_diameter_m must be positive")
    if bool((source.electrostatic_radius_m < 0).any()):
        raise ValueError("source electrostatic_radius_m must be nonnegative")
    if bool((source.displaced_volume_m3 < 0).any()) or bool((source.model_weight <= 0).any()):
        raise ValueError("source displaced_volume_m3 must be nonnegative and model_weight positive")


def _logical_records(data: DataBundle) -> list[_DatasetRecord]:
    records = [
        _numeric_record("/meta/schema_version", np.asarray(SCHEMA_VERSION, dtype=_I4)),
        _string_record("/meta/coordinate_system", data.coordinate_system),
        _string_record("/meta/coordinate_units", "m"),
        _string_record("/meta/provenance_json", data.provenance_json),
        _numeric_record("/geometry/nodes_m", data.geometry.nodes_m),
        _string_record("/geometry/groups/names", data.geometry.group_names),
    ]
    records.extend(_geometry_records(data.geometry))
    for layout in data.layouts:
        records.extend(_layout_records(layout))
    for field in data.fields:
        records.extend(_field_records(field))
    for source in data.sources:
        records.extend(_source_records(source))
    return records


def _geometry_records(geometry: GeometryData) -> list[_DatasetRecord]:
    records: list[_DatasetRecord] = []
    optional = (
        ("/geometry/node_external_id", geometry.node_external_id),
        ("/geometry/cells/tri3", geometry.tri3),
        ("/geometry/cells/tri3_domain_id", geometry.tri3_domain_id),
        ("/geometry/cells/quad4", geometry.quad4),
        ("/geometry/cells/quad4_domain_id", geometry.quad4_domain_id),
        ("/geometry/boundary/external_id", geometry.boundary.external_id),
    )
    records.extend(_numeric_record(path, value) for path, value in optional if value is not None)
    boundary = geometry.boundary
    records.append(_numeric_record("/geometry/boundary/line2", boundary.line2))
    for label, array, _dtype in _boundary_arrays(boundary):
        records.append(_numeric_record(f"/geometry/boundary/{label.rsplit('.', 1)[-1]}", array))
    return records


def _layout_records(layout: Layout) -> list[_DatasetRecord]:
    root = f"/layouts/{layout.name}"
    if isinstance(layout, RegularLayout):
        return [
            _string_record(f"{root}/kind", "regular"),
            _numeric_record(f"{root}/regular/axes/axis0_m", layout.axis0_m),
            _numeric_record(f"{root}/regular/axes/axis1_m", layout.axis1_m),
            _numeric_record(f"{root}/regular/cell_support", layout.cell_support),
        ]
    kind = "p1_tri" if isinstance(layout, P1TriLayout) else "q1_quad"
    return [
        _string_record(f"{root}/kind", kind),
        _numeric_record(f"{root}/unstructured/nodes_m", layout.nodes_m),
        _numeric_record(f"{root}/unstructured/connectivity", layout.connectivity),
        _numeric_record(f"{root}/unstructured/cell_support", layout.cell_support),
    ]


def _field_records(field: FieldData) -> list[_DatasetRecord]:
    root = f"/fields/{field.name}"
    return [
        _string_record(f"{root}/layout", field.layout),
        _string_record(f"{root}/association", field.association),
        _string_record(f"{root}/components", field.components),
        _string_record(f"{root}/stored_basis", field.stored_basis),
        _numeric_record(f"{root}/values", field.values),
        _string_record(f"{root}/unit", field.unit),
    ]


def _source_records(source: RealizedTableSource) -> list[_DatasetRecord]:
    root = f"/sources/{source.name}"
    records = [
        _numeric_record(f"{root}/particle_id", source.particle_id),
        _numeric_record(f"{root}/position_m", source.position_m),
        _numeric_record(f"{root}/velocity_m_s", source.velocity_m_s),
        _numeric_record(f"{root}/material_id", source.material_id),
    ]
    records.extend(
        _numeric_record(f"{root}/{label}", array) for label, array in _source_scalar_arrays(source)
    )
    return records


def _numeric_record(path: str, value: NDArray[Any]) -> _DatasetRecord:
    return _DatasetRecord(
        path, value.dtype.str.removeprefix("<").removeprefix("|"), value.shape, value
    )


def _string_record(path: str, value: str | tuple[str, ...]) -> _DatasetRecord:
    if isinstance(value, str):
        return _DatasetRecord(path, "utf8", (), strings=(value,))
    return _DatasetRecord(path, "utf8", (len(value),), strings=value)


def _hash_record(digest: Any, record: _DatasetRecord) -> None:
    _hash_bytes(digest, record.path.encode("utf-8"))
    _hash_bytes(digest, record.dtype_tag.encode("utf-8"))
    digest.update(struct.pack("<Q", len(record.shape)))
    for dimension in record.shape:
        digest.update(struct.pack("<Q", dimension))
    if record.numeric is not None:
        if record.numeric.size:
            digest.update(memoryview(record.numeric).cast("B"))
    else:
        for value in record.strings:
            _hash_bytes(digest, value.encode("utf-8"))


def _hash_bytes(digest: Any, value: bytes) -> None:
    digest.update(struct.pack("<Q", len(value)))
    digest.update(value)


def _write_bundle(handle: h5py.File, data: DataBundle) -> None:
    meta = handle.create_group("meta")
    _write_numeric(meta, "schema_version", np.asarray(SCHEMA_VERSION, dtype=_I4))
    _write_string(meta, "coordinate_system", data.coordinate_system)
    _write_string(meta, "coordinate_units", "m")
    _write_string(meta, "provenance_json", data.provenance_json)
    _write_geometry(handle.create_group("geometry"), data.geometry)
    layouts = handle.create_group("layouts")
    for layout in data.layouts:
        _write_layout(layouts.create_group(layout.name), layout)
    fields = handle.create_group("fields")
    for field in data.fields:
        _write_field(fields.create_group(field.name), field)
    sources = handle.create_group("sources")
    for source in data.sources:
        _write_source(sources.create_group(source.name), source)


def _write_numeric(group: h5py.Group, name: str, value: NDArray[Any]) -> None:
    group.create_dataset(name, data=value, dtype=value.dtype)


def _write_string(group: h5py.Group, name: str, value: str | tuple[str, ...]) -> None:
    if isinstance(value, str):
        dataset = group.create_dataset(name, shape=(), dtype=_UTF8)
        dataset[()] = value
    else:
        dataset = group.create_dataset(name, shape=(len(value),), dtype=_UTF8)
        if value:
            dataset[:] = value


def _write_geometry(group: h5py.Group, geometry: GeometryData) -> None:
    _write_numeric(group, "nodes_m", geometry.nodes_m)
    if geometry.node_external_id is not None:
        _write_numeric(group, "node_external_id", geometry.node_external_id)
    cells = group.create_group("cells")
    for name, values in (
        ("tri3", geometry.tri3),
        ("tri3_domain_id", geometry.tri3_domain_id),
        ("quad4", geometry.quad4),
        ("quad4_domain_id", geometry.quad4_domain_id),
    ):
        if values is not None:
            _write_numeric(cells, name, values)
    boundary_group = group.create_group("boundary")
    _write_boundary(boundary_group, geometry.boundary)
    names = group.create_group("groups")
    _write_string(names, "names", geometry.group_names)


def _write_boundary(group: h5py.Group, boundary: BoundaryData) -> None:
    _write_numeric(group, "line2", boundary.line2)
    if boundary.external_id is not None:
        _write_numeric(group, "external_id", boundary.external_id)
    for label, array, _dtype in _boundary_arrays(boundary):
        _write_numeric(group, label.rsplit(".", 1)[-1], array)


def _write_layout(group: h5py.Group, layout: Layout) -> None:
    if isinstance(layout, RegularLayout):
        _write_string(group, "kind", "regular")
        regular = group.create_group("regular")
        axes = regular.create_group("axes")
        _write_numeric(axes, "axis0_m", layout.axis0_m)
        _write_numeric(axes, "axis1_m", layout.axis1_m)
        _write_numeric(regular, "cell_support", layout.cell_support)
        return
    _write_string(group, "kind", "p1_tri" if isinstance(layout, P1TriLayout) else "q1_quad")
    unstructured = group.create_group("unstructured")
    _write_numeric(unstructured, "nodes_m", layout.nodes_m)
    _write_numeric(unstructured, "connectivity", layout.connectivity)
    _write_numeric(unstructured, "cell_support", layout.cell_support)


def _write_field(group: h5py.Group, field: FieldData) -> None:
    _write_string(group, "layout", field.layout)
    _write_string(group, "association", field.association)
    _write_string(group, "components", field.components)
    _write_string(group, "stored_basis", field.stored_basis)
    _write_numeric(group, "values", field.values)
    _write_string(group, "unit", field.unit)


def _write_source(group: h5py.Group, source: RealizedTableSource) -> None:
    _write_numeric(group, "particle_id", source.particle_id)
    _write_numeric(group, "position_m", source.position_m)
    _write_numeric(group, "velocity_m_s", source.velocity_m_s)
    _write_numeric(group, "material_id", source.material_id)
    for label, array in _source_scalar_arrays(source):
        _write_numeric(group, label, array)


def _validate_hdf5_objects(handle: h5py.File) -> CanonicalDataFootprint:
    seen = {hash(handle.id)}
    return CanonicalDataFootprint(_walk_hdf5_group(handle, seen))


def _walk_hdf5_group(group: h5py.Group, seen: set[int]) -> int:
    numeric_array_bytes = 0
    if group.attrs:
        raise ValueError(f"attributes are not allowed on {group.name}")
    for name in group:
        link = group.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ValueError(f"non-hard HDF5 link is not allowed: {group.name}/{name}")
        obj = group[name]
        address = hash(obj.id)
        if address in seen:
            raise ValueError(f"HDF5 hard-link aliases are not allowed: {obj.name}")
        seen.add(address)
        if obj.attrs:
            raise ValueError(f"attributes are not allowed on {obj.name}")
        if isinstance(obj, h5py.Group):
            numeric_array_bytes += _walk_hdf5_group(obj, seen)
        elif isinstance(obj, h5py.Dataset):
            if obj.is_virtual or obj.external is not None:
                raise ValueError(f"external or virtual dataset is not allowed: {obj.name}")
            if obj.dtype.kind in "biufc":
                numeric_array_bytes += int(obj.size) * int(obj.dtype.itemsize)
        else:
            raise ValueError(f"unsupported HDF5 object: {obj.name}")
    return numeric_array_bytes


def _read_bundle(handle: h5py.File) -> DataBundle:
    _expect_children(handle, {"meta", "geometry", "layouts", "fields", "sources"})
    meta = _expect_group(handle, "meta")
    _expect_children(
        meta, {"schema_version", "coordinate_system", "coordinate_units", "provenance_json"}
    )
    schema = _read_numeric(meta, "schema_version", _I4, 0)
    if int(schema) != SCHEMA_VERSION:
        raise ValueError(f"unsupported case schema version: {int(schema)}")
    coordinate_system = _read_string_scalar(meta, "coordinate_system")
    if _read_string_scalar(meta, "coordinate_units") != "m":
        raise ValueError("schema v1 coordinate_units must be 'm'")
    provenance = _read_string_scalar(meta, "provenance_json")
    if _canonical_provenance(provenance) != provenance:
        raise ValueError("stored provenance_json is not canonical")
    geometry = _read_geometry(_expect_group(handle, "geometry"))
    layouts = _read_layouts(_expect_group(handle, "layouts"))
    fields = _read_fields(_expect_group(handle, "fields"))
    sources = _read_sources(_expect_group(handle, "sources"))
    return DataBundle(
        cast(CoordinateSystem, coordinate_system), provenance, geometry, layouts, fields, sources
    )


def _expect_children(
    group: h5py.Group,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    actual = set(group.keys())
    missing = required - actual
    unknown = actual - allowed
    if missing or unknown:
        raise ValueError(
            f"invalid objects under {group.name}: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _expect_group(parent: h5py.Group, name: str) -> h5py.Group:
    obj = parent[name]
    if not isinstance(obj, h5py.Group):
        raise ValueError(f"{obj.name} must be an HDF5 group")
    return obj


def _expect_dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
    obj = parent[name]
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"{obj.name} must be an HDF5 dataset")
    return obj


def _read_numeric(
    parent: h5py.Group,
    name: str,
    dtype: np.dtype[Any],
    ndim: int,
) -> NDArray[Any]:
    dataset = _expect_dataset(parent, name)
    if dataset.dtype.str != dtype.str or dataset.ndim != ndim:
        raise ValueError(f"{dataset.name} must have dtype {dtype.str} and rank {ndim}")
    return np.asarray(dataset[()])


def _validate_string_dataset(dataset: h5py.Dataset, ndim: int) -> None:
    info = h5py.check_string_dtype(dataset.dtype)
    if info is None or info.encoding != "utf-8" or info.length is not None or dataset.ndim != ndim:
        raise ValueError(f"{dataset.name} must be a variable-length UTF-8 dataset of rank {ndim}")


def _read_string_scalar(parent: h5py.Group, name: str) -> str:
    dataset = _expect_dataset(parent, name)
    _validate_string_dataset(dataset, 0)
    try:
        return cast(str, dataset.asstr()[()])
    except UnicodeDecodeError as error:
        raise ValueError(f"{dataset.name} contains invalid UTF-8") from error


def _read_string_vector(parent: h5py.Group, name: str) -> tuple[str, ...]:
    dataset = _expect_dataset(parent, name)
    _validate_string_dataset(dataset, 1)
    try:
        return tuple(cast(list[str], dataset.asstr()[()].tolist()))
    except UnicodeDecodeError as error:
        raise ValueError(f"{dataset.name} contains invalid UTF-8") from error


def _read_geometry(group: h5py.Group) -> GeometryData:
    _expect_children(group, {"nodes_m", "cells", "boundary", "groups"}, {"node_external_id"})
    cells = _expect_group(group, "cells")
    _expect_children(cells, set(), {"tri3", "tri3_domain_id", "quad4", "quad4_domain_id"})
    boundary = _read_boundary(_expect_group(group, "boundary"))
    names = _expect_group(group, "groups")
    _expect_children(names, {"names"})
    return GeometryData(
        nodes_m=cast(FloatArray, _read_numeric(group, "nodes_m", _F8, 2)),
        node_external_id=cast(
            Int64Array, _read_optional_numeric(group, "node_external_id", _I8, 1)
        ),
        tri3=cast(Int64Array, _read_optional_numeric(cells, "tri3", _I8, 2)),
        tri3_domain_id=cast(Int32Array, _read_optional_numeric(cells, "tri3_domain_id", _I4, 1)),
        quad4=cast(Int64Array, _read_optional_numeric(cells, "quad4", _I8, 2)),
        quad4_domain_id=cast(Int32Array, _read_optional_numeric(cells, "quad4_domain_id", _I4, 1)),
        boundary=boundary,
        group_names=_read_string_vector(names, "names"),
    )


def _read_optional_numeric(
    parent: h5py.Group,
    name: str,
    dtype: np.dtype[Any],
    ndim: int,
) -> NDArray[Any] | None:
    if name not in parent:
        return None
    return _read_numeric(parent, name, dtype, ndim)


def _read_boundary(group: h5py.Group) -> BoundaryData:
    required = {
        "line2",
        "boundary_id",
        "group_id",
        "material_id",
        "owner_cell_type",
        "owner_cell_local_index",
        "orientation",
    }
    _expect_children(group, required, {"external_id"})
    return BoundaryData(
        line2=cast(Int64Array, _read_numeric(group, "line2", _I8, 2)),
        boundary_id=cast(Int32Array, _read_numeric(group, "boundary_id", _I4, 1)),
        group_id=cast(Int32Array, _read_numeric(group, "group_id", _I4, 1)),
        material_id=cast(Int32Array, _read_numeric(group, "material_id", _I4, 1)),
        owner_cell_type=cast(UInt8Array, _read_numeric(group, "owner_cell_type", _U1, 1)),
        owner_cell_local_index=cast(
            Int64Array, _read_numeric(group, "owner_cell_local_index", _I8, 1)
        ),
        orientation=cast(Int8Array, _read_numeric(group, "orientation", _I1, 1)),
        external_id=cast(Int64Array, _read_optional_numeric(group, "external_id", _I8, 1)),
    )


def _read_layouts(group: h5py.Group) -> tuple[Layout, ...]:
    layouts: list[Layout] = []
    for name in sorted(group.keys()):
        _require_name(name, "layout name")
        layout_group = _expect_group(group, name)
        kind = _read_layout_kind(layout_group)
        if kind == "regular":
            layouts.append(_read_regular_layout(name, layout_group))
        else:
            layouts.append(_read_unstructured_layout(name, kind, layout_group))
    return tuple(layouts)


def _read_layout_kind(group: h5py.Group) -> str:
    if "kind" not in group:
        raise ValueError(f"{group.name} is missing kind")
    kind = _read_string_scalar(group, "kind")
    if kind not in ("regular", "p1_tri", "q1_quad"):
        raise ValueError(f"unsupported layout kind: {kind}")
    return kind


def _read_regular_layout(name: str, group: h5py.Group) -> RegularLayout:
    _expect_children(group, {"kind", "regular"})
    regular = _expect_group(group, "regular")
    _expect_children(regular, {"axes", "cell_support"})
    axes = _expect_group(regular, "axes")
    _expect_children(axes, {"axis0_m", "axis1_m"})
    return RegularLayout(
        name,
        cast(FloatArray, _read_numeric(axes, "axis0_m", _F8, 1)),
        cast(FloatArray, _read_numeric(axes, "axis1_m", _F8, 1)),
        cast(UInt8Array, _read_numeric(regular, "cell_support", _U1, 2)),
    )


def _read_unstructured_layout(name: str, kind: str, group: h5py.Group) -> Layout:
    _expect_children(group, {"kind", "unstructured"})
    unstructured = _expect_group(group, "unstructured")
    _expect_children(unstructured, {"nodes_m", "connectivity", "cell_support"})
    nodes = cast(FloatArray, _read_numeric(unstructured, "nodes_m", _F8, 2))
    connectivity = cast(Int64Array, _read_numeric(unstructured, "connectivity", _I8, 2))
    support = cast(UInt8Array, _read_numeric(unstructured, "cell_support", _U1, 1))
    if kind == "p1_tri":
        return P1TriLayout(name, nodes, connectivity, support)
    return Q1QuadLayout(name, nodes, connectivity, support)


def _read_fields(group: h5py.Group) -> tuple[FieldData, ...]:
    fields: list[FieldData] = []
    required = {"layout", "association", "components", "stored_basis", "values", "unit"}
    for name in sorted(group.keys()):
        _require_name(name, "field name")
        field_group = _expect_group(group, name)
        _expect_children(field_group, required)
        fields.append(
            FieldData(
                name=name,
                layout=_read_string_scalar(field_group, "layout"),
                association=cast(
                    Literal["node", "cell"], _read_string_scalar(field_group, "association")
                ),
                components=_read_string_vector(field_group, "components"),
                stored_basis=_read_string_scalar(field_group, "stored_basis"),
                values=cast(FloatArray, _read_numeric(field_group, "values", _F8, 2)),
                unit=_read_string_scalar(field_group, "unit"),
            )
        )
    return tuple(fields)


def _read_sources(group: h5py.Group) -> tuple[RealizedTableSource, ...]:
    sources: list[RealizedTableSource] = []
    required = {
        "particle_id",
        "release_time_s",
        "position_m",
        "velocity_m_s",
        "charge_number",
        "mass_kg",
        "drag_diameter_m",
        "electrostatic_radius_m",
        "displaced_volume_m3",
        "model_weight",
        "material_id",
    }
    for name in sorted(group.keys()):
        _require_name(name, "source name")
        source_group = _expect_group(group, name)
        _expect_children(source_group, required)
        sources.append(_read_source(name, source_group))
    return tuple(sources)


def _read_source(name: str, group: h5py.Group) -> RealizedTableSource:
    return RealizedTableSource(
        name=name,
        particle_id=cast(Int64Array, _read_numeric(group, "particle_id", _I8, 1)),
        release_time_s=cast(FloatArray, _read_numeric(group, "release_time_s", _F8, 1)),
        position_m=cast(FloatArray, _read_numeric(group, "position_m", _F8, 2)),
        velocity_m_s=cast(FloatArray, _read_numeric(group, "velocity_m_s", _F8, 2)),
        charge_number=cast(FloatArray, _read_numeric(group, "charge_number", _F8, 1)),
        mass_kg=cast(FloatArray, _read_numeric(group, "mass_kg", _F8, 1)),
        drag_diameter_m=cast(FloatArray, _read_numeric(group, "drag_diameter_m", _F8, 1)),
        electrostatic_radius_m=cast(
            FloatArray, _read_numeric(group, "electrostatic_radius_m", _F8, 1)
        ),
        displaced_volume_m3=cast(FloatArray, _read_numeric(group, "displaced_volume_m3", _F8, 1)),
        model_weight=cast(FloatArray, _read_numeric(group, "model_weight", _F8, 1)),
        material_id=cast(Int32Array, _read_numeric(group, "material_id", _I4, 1)),
    )


def _mark_arrays_read_only(data: DataBundle) -> None:
    for record in _logical_records(data):
        if record.numeric is not None:
            record.numeric.setflags(write=False)


__all__ = [
    "SCHEMA_VERSION",
    "BoundaryData",
    "CanonicalDataFootprint",
    "CaseFileInfo",
    "DataBundle",
    "FieldData",
    "GeometryData",
    "P1TriLayout",
    "Q1QuadLayout",
    "RealizedTableSource",
    "RegularLayout",
    "content_hash",
    "read",
    "read_with_info",
    "write",
]
