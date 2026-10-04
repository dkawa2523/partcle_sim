"""Prepared two-dimensional particle-domain geometry and boundary queries."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .case_format import CoordinateSystem, GeometryData

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type Int32Array = NDArray[np.int32]
type PointClassification = Literal["inside", "boundary", "outside"]

GEOMETRY_ALGORITHM_REVISION = "line_boundary_stackless_volume_cell_bvh_v5"
BVH_LEAF_SIZE = 8
_VOLUME_INDEX_ULPS = 64.0
_VOLUME_INDEX_BUILD_WORK_BYTES_PER_CELL = 1024
_FLOAT_EPS = np.finfo(np.float64).eps

_EMPTY_TRI3 = np.empty((0, 3), dtype="<i8")
_EMPTY_QUAD4 = np.empty((0, 4), dtype="<i8")
_EMPTY_TRI3.setflags(write=False)
_EMPTY_QUAD4.setflags(write=False)


class GeometryPreparationError(RuntimeError):
    """The canonical mesh cannot define an unambiguous particle boundary."""


@dataclass(frozen=True, slots=True)
class PreparedGeometry:
    """Read-only geometry facts derived once before particle integration."""

    coordinate_system: CoordinateSystem
    nodes_m: FloatArray
    tri3: Int64Array | None
    quad4: Int64Array | None
    facet_node_ids: Int64Array
    boundary_id: Int32Array
    group_id: Int32Array
    material_id: Int32Array
    facet_start_m: FloatArray
    facet_end_m: FloatArray
    facet_normal: FloatArray
    facet_length_m: FloatArray
    bbox_diagonal_m: float
    bvh_facet_id: Int64Array
    bvh_lower_m: FloatArray
    bvh_upper_m: FloatArray
    bvh_left: Int64Array
    bvh_right: Int64Array
    bvh_begin: Int64Array
    bvh_end: Int64Array
    bvh_skip: Int64Array
    volume_bvh_cell_id: Int64Array
    volume_bvh_lower_m: FloatArray
    volume_bvh_upper_m: FloatArray
    volume_bvh_begin: Int64Array
    volume_bvh_end: Int64Array
    volume_bvh_skip: Int64Array
    tri3_edge_length_m: FloatArray
    quad4_edge_length_m: FloatArray
    volume_bvh_build_transient_nbytes: int

    @property
    def facet_count(self) -> int:
        """Return the number of physical material-boundary facets."""

        return int(self.facet_node_ids.shape[0])


@dataclass(slots=True)
class _BvhArrays:
    facet_id: list[int]
    lower_m: list[tuple[float, float]]
    upper_m: list[tuple[float, float]]
    left: list[int]
    right: list[int]
    begin: list[int]
    end: list[int]
    skip: list[int]


@dataclass(frozen=True, slots=True)
class _VolumeBvhArrays:
    """Exact-size stackless volume-cell index built during geometry preparation."""

    cell_id: Int64Array
    lower_m: FloatArray
    upper_m: FloatArray
    begin: Int64Array
    end: Int64Array
    skip: Int64Array
    tri3_edge_length_m: FloatArray
    quad4_edge_length_m: FloatArray
    build_transient_nbytes: int


def prepare_geometry(
    geometry: GeometryData, coordinate_system: CoordinateSystem
) -> PreparedGeometry:
    """Audit global topology and build a deterministic boundary AABB tree.

    A geometry with no boundary rows is an explicit collision-free geometry and
    remains preparable. Once one physical boundary is supplied, every exterior
    volume edge must be represented exactly once, apart from an omitted RZ axis
    seam.
    """

    edge_incidence = _volume_edge_incidence(geometry)
    _audit_volume_edges(edge_incidence)
    _audit_boundary_inventory(geometry, coordinate_system, edge_incidence)
    del edge_incidence

    nodes = geometry.nodes_m
    facet_node_ids = geometry.boundary.line2
    start = nodes[facet_node_ids[:, 0]].copy()
    end = nodes[facet_node_ids[:, 1]].copy()
    edge = _finite_difference(end, start, "boundary facet")
    length = np.hypot(edge[:, 0], edge[:, 1])
    if not bool(np.isfinite(length).all()) or bool((length <= 0.0).any()):
        raise GeometryPreparationError("boundary contains an unresolved facet length")
    orientation = geometry.boundary.orientation.astype(np.float64, copy=False)
    normal = np.column_stack((edge[:, 1], -edge[:, 0]))
    normal *= orientation[:, None] / length[:, None]
    if not bool(np.isfinite(normal).all()):
        raise GeometryPreparationError("boundary normal is not finite")

    bbox_diagonal = _geometry_bbox_diagonal(nodes)
    bvh = _build_bvh(start, end)
    volume_bvh = _build_volume_bvh(geometry)
    _audit_boundary_geometry(geometry, coordinate_system, start, end, bvh)
    return PreparedGeometry(
        coordinate_system=coordinate_system,
        nodes_m=nodes,
        tri3=geometry.tri3,
        quad4=geometry.quad4,
        facet_node_ids=_read_only_copy(facet_node_ids),
        boundary_id=_read_only_copy(geometry.boundary.boundary_id),
        group_id=_read_only_copy(geometry.boundary.group_id),
        material_id=_read_only_copy(geometry.boundary.material_id),
        facet_start_m=_read_only(start),
        facet_end_m=_read_only(end),
        facet_normal=_read_only(normal),
        facet_length_m=_read_only(length.astype(np.float64, copy=False)),
        bbox_diagonal_m=bbox_diagonal,
        bvh_facet_id=_read_only(np.asarray(bvh.facet_id, dtype=np.int64)),
        bvh_lower_m=_read_only(np.asarray(bvh.lower_m, dtype=np.float64).reshape(-1, 2)),
        bvh_upper_m=_read_only(np.asarray(bvh.upper_m, dtype=np.float64).reshape(-1, 2)),
        bvh_left=_read_only(np.asarray(bvh.left, dtype=np.int64)),
        bvh_right=_read_only(np.asarray(bvh.right, dtype=np.int64)),
        bvh_begin=_read_only(np.asarray(bvh.begin, dtype=np.int64)),
        bvh_end=_read_only(np.asarray(bvh.end, dtype=np.int64)),
        bvh_skip=_read_only(np.asarray(bvh.skip, dtype=np.int64)),
        volume_bvh_cell_id=volume_bvh.cell_id,
        volume_bvh_lower_m=volume_bvh.lower_m,
        volume_bvh_upper_m=volume_bvh.upper_m,
        volume_bvh_begin=volume_bvh.begin,
        volume_bvh_end=volume_bvh.end,
        volume_bvh_skip=volume_bvh.skip,
        tri3_edge_length_m=volume_bvh.tri3_edge_length_m,
        quad4_edge_length_m=volume_bvh.quad4_edge_length_m,
        volume_bvh_build_transient_nbytes=volume_bvh.build_transient_nbytes,
    )


def query_segment_candidates(
    geometry: PreparedGeometry,
    start_m: FloatArray,
    end_m: FloatArray,
    *,
    padding_m: float,
) -> Int64Array:
    """Return canonical facet IDs whose AABBs overlap a padded segment AABB."""

    start = _finite_point(start_m, "start_m")
    end = _finite_point(end_m, "end_m")
    if not math.isfinite(padding_m) or padding_m < 0.0:
        raise ValueError("padding_m must be finite and nonnegative")
    lower = np.minimum(start, end) - padding_m
    upper = np.maximum(start, end) + padding_m
    if not bool(np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise GeometryPreparationError("padded path AABB exceeds the finite float64 range")
    return query_aabb_candidates(geometry, lower, upper)


def query_aabb_candidates(
    geometry: PreparedGeometry,
    lower_m: FloatArray,
    upper_m: FloatArray,
) -> Int64Array:
    """Return canonical facet IDs whose AABBs overlap a finite query box."""

    lower = _finite_point(lower_m, "lower_m")
    upper = _finite_point(upper_m, "upper_m")
    if bool((lower > upper).any()):
        raise ValueError("query AABB lower bounds exceed upper bounds")
    return _query_aabb_candidates_kernel(
        lower,
        upper,
        geometry.bvh_facet_id,
        geometry.bvh_lower_m,
        geometry.bvh_upper_m,
        geometry.bvh_left,
        geometry.bvh_right,
        geometry.bvh_begin,
        geometry.bvh_end,
        geometry.bvh_skip,
        geometry.facet_start_m,
        geometry.facet_end_m,
    )


def count_aabb_candidates(
    geometry: PreparedGeometry,
    lower_m: FloatArray,
    upper_m: FloatArray,
) -> Int64Array:
    """Count AABB candidates per row without allocating candidate storage.

    The caller can prefix these counts against its memory-plan capacity before
    allocating or filling a CSR candidate buffer.
    """

    lower, upper = _validated_aabb_batch(lower_m, upper_m)
    return _aabb_candidate_counts_batch_kernel(
        lower,
        upper,
        geometry.bvh_facet_id,
        geometry.bvh_lower_m,
        geometry.bvh_upper_m,
        geometry.bvh_begin,
        geometry.bvh_end,
        geometry.bvh_skip,
        geometry.facet_start_m,
        geometry.facet_end_m,
    )


def fill_aabb_candidates_csr(
    geometry: PreparedGeometry,
    lower_m: FloatArray,
    upper_m: FloatArray,
    offsets: Int64Array,
    candidates: Int64Array,
) -> None:
    """Fill caller-owned CSR storage using the same predicate as count-only.

    ``offsets`` must describe exact row counts previously returned by
    :func:`count_aabb_candidates`.  The compiled fill is bounded by each row's
    segment, so stale counts fail closed instead of writing into another row.
    Every completed segment is sorted by canonical facet ID.
    """

    lower, upper = _validated_aabb_batch(lower_m, upper_m)
    csr_offsets = _validated_candidate_offsets(offsets, lower.shape[0], candidates)
    candidate_buffer = np.asarray(candidates)
    fill_valid = np.empty(lower.shape[0], dtype=np.bool_)
    _aabb_candidate_fill_batch_kernel(
        lower,
        upper,
        csr_offsets,
        geometry.bvh_facet_id,
        geometry.bvh_lower_m,
        geometry.bvh_upper_m,
        geometry.bvh_begin,
        geometry.bvh_end,
        geometry.bvh_skip,
        geometry.facet_start_m,
        geometry.facet_end_m,
        candidate_buffer,
        fill_valid,
    )
    if not bool(fill_valid.all()):
        raise GeometryPreparationError(
            "candidate counts changed between AABB count and bounded CSR fill"
        )


def _validated_aabb_batch(
    lower_m: FloatArray,
    upper_m: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    lower = np.asarray(lower_m, dtype=np.float64)
    upper = np.asarray(upper_m, dtype=np.float64)
    if lower.ndim != 2 or lower.shape[1] != 2 or upper.shape != lower.shape:
        raise ValueError("lower_m and upper_m must have matching shape (N, 2)")
    if not bool(np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("query AABBs must contain only finite values")
    if bool((lower > upper).any()):
        raise ValueError("query AABB lower bounds exceed upper bounds")
    return lower, upper


def _validated_candidate_offsets(
    offsets: Int64Array,
    row_count: int,
    candidates: Int64Array,
) -> Int64Array:
    csr_offsets = np.asarray(offsets)
    candidate_buffer = np.asarray(candidates)
    valid_arrays = (
        csr_offsets.dtype == np.dtype(np.int64)
        and candidate_buffer.dtype == np.dtype(np.int64)
        and csr_offsets.ndim == 1
        and candidate_buffer.ndim == 1
        and candidate_buffer.flags.c_contiguous
        and candidate_buffer.flags.writeable
    )
    if not valid_arrays or csr_offsets.shape != (row_count + 1,):
        raise ValueError("candidate offsets and writable int64 buffer must form row CSR")
    if csr_offsets[0] != 0 or bool((csr_offsets[1:] < csr_offsets[:-1]).any()):
        raise ValueError("candidate offsets must be nonnegative and nondecreasing from zero")
    if int(csr_offsets[-1]) > candidate_buffer.size:
        raise ValueError("candidate offsets exceed the caller-owned buffer capacity")
    return csr_offsets


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _query_aabb_candidates_kernel(
    lower_m: FloatArray,
    upper_m: FloatArray,
    bvh_facet_id: Int64Array,
    bvh_lower_m: FloatArray,
    bvh_upper_m: FloatArray,
    bvh_left: Int64Array,
    bvh_right: Int64Array,
    bvh_begin: Int64Array,
    bvh_end: Int64Array,
    bvh_skip: Int64Array,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
) -> Int64Array:
    """Traverse one read-only BVH without a per-query stack or full-size buffer."""

    if bvh_left.size == 0:
        return np.empty(0, dtype=np.int64)
    candidate_count = _aabb_candidate_count_kernel(
        lower_m,
        upper_m,
        bvh_facet_id,
        bvh_lower_m,
        bvh_upper_m,
        bvh_begin,
        bvh_end,
        bvh_skip,
        facet_start_m,
        facet_end_m,
    )
    candidates = np.empty(candidate_count, dtype=np.int64)
    _aabb_candidate_fill_kernel(
        lower_m,
        upper_m,
        bvh_facet_id,
        bvh_lower_m,
        bvh_upper_m,
        bvh_begin,
        bvh_end,
        bvh_skip,
        facet_start_m,
        facet_end_m,
        candidates,
        0,
        candidate_count,
    )
    candidates.sort()
    return candidates


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _aabb_candidate_count_kernel(
    lower_m: FloatArray,
    upper_m: FloatArray,
    bvh_facet_id: Int64Array,
    bvh_lower_m: FloatArray,
    bvh_upper_m: FloatArray,
    bvh_begin: Int64Array,
    bvh_end: Int64Array,
    bvh_skip: Int64Array,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
) -> int:
    count = 0
    node = 0
    while node < bvh_skip.size:
        if (
            upper_m[0] < bvh_lower_m[node, 0]
            or upper_m[1] < bvh_lower_m[node, 1]
            or bvh_upper_m[node, 0] < lower_m[0]
            or bvh_upper_m[node, 1] < lower_m[1]
        ):
            node = bvh_skip[node]
            continue
        begin = bvh_begin[node]
        if begin >= 0:
            for offset in range(bvh_begin[node], bvh_end[node]):
                facet_id = bvh_facet_id[offset]
                start_x = facet_start_m[facet_id, 0]
                start_y = facet_start_m[facet_id, 1]
                end_x = facet_end_m[facet_id, 0]
                end_y = facet_end_m[facet_id, 1]
                facet_lower_x = min(start_x, end_x)
                facet_lower_y = min(start_y, end_y)
                facet_upper_x = max(start_x, end_x)
                facet_upper_y = max(start_y, end_y)
                if (
                    upper_m[0] >= facet_lower_x
                    and upper_m[1] >= facet_lower_y
                    and facet_upper_x >= lower_m[0]
                    and facet_upper_y >= lower_m[1]
                ):
                    count += 1
            node = bvh_skip[node]
            continue
        node += 1
    return count


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _aabb_candidate_fill_kernel(
    lower_m: FloatArray,
    upper_m: FloatArray,
    bvh_facet_id: Int64Array,
    bvh_lower_m: FloatArray,
    bvh_upper_m: FloatArray,
    bvh_begin: Int64Array,
    bvh_end: Int64Array,
    bvh_skip: Int64Array,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    candidates: Int64Array,
    write_begin: int,
    write_end: int,
) -> int:
    write = write_begin
    node = 0
    while node < bvh_skip.size:
        if (
            upper_m[0] < bvh_lower_m[node, 0]
            or upper_m[1] < bvh_lower_m[node, 1]
            or bvh_upper_m[node, 0] < lower_m[0]
            or bvh_upper_m[node, 1] < lower_m[1]
        ):
            node = bvh_skip[node]
            continue
        begin = bvh_begin[node]
        if begin >= 0:
            for offset in range(begin, bvh_end[node]):
                facet_id = bvh_facet_id[offset]
                start_x = facet_start_m[facet_id, 0]
                start_y = facet_start_m[facet_id, 1]
                end_x = facet_end_m[facet_id, 0]
                end_y = facet_end_m[facet_id, 1]
                if (
                    upper_m[0] >= min(start_x, end_x)
                    and upper_m[1] >= min(start_y, end_y)
                    and max(start_x, end_x) >= lower_m[0]
                    and max(start_y, end_y) >= lower_m[1]
                ):
                    if write >= write_end:
                        return -1
                    candidates[write] = facet_id
                    write += 1
            node = bvh_skip[node]
            continue
        node += 1
    return write


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _aabb_candidate_counts_batch_kernel(
    lower_m: FloatArray,
    upper_m: FloatArray,
    bvh_facet_id: Int64Array,
    bvh_lower_m: FloatArray,
    bvh_upper_m: FloatArray,
    bvh_begin: Int64Array,
    bvh_end: Int64Array,
    bvh_skip: Int64Array,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
) -> Int64Array:
    counts = np.empty(lower_m.shape[0], dtype=np.int64)
    for row in range(lower_m.shape[0]):
        counts[row] = _aabb_candidate_count_kernel(
            lower_m[row],
            upper_m[row],
            bvh_facet_id,
            bvh_lower_m,
            bvh_upper_m,
            bvh_begin,
            bvh_end,
            bvh_skip,
            facet_start_m,
            facet_end_m,
        )
    return counts


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _aabb_candidate_fill_batch_kernel(
    lower_m: FloatArray,
    upper_m: FloatArray,
    offsets: Int64Array,
    bvh_facet_id: Int64Array,
    bvh_lower_m: FloatArray,
    bvh_upper_m: FloatArray,
    bvh_begin: Int64Array,
    bvh_end: Int64Array,
    bvh_skip: Int64Array,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    candidates: Int64Array,
    fill_valid: NDArray[np.bool_],
) -> None:
    for row in range(lower_m.shape[0]):
        begin = offsets[row]
        end = _aabb_candidate_fill_kernel(
            lower_m[row],
            upper_m[row],
            bvh_facet_id,
            bvh_lower_m,
            bvh_upper_m,
            bvh_begin,
            bvh_end,
            bvh_skip,
            facet_start_m,
            facet_end_m,
            candidates,
            begin,
            offsets[row + 1],
        )
        fill_valid[row] = end == offsets[row + 1]
        if fill_valid[row]:
            candidates[begin:end].sort()


def classify_point(
    geometry: PreparedGeometry,
    position_m: FloatArray,
    *,
    candidate_padding_m: float,
    facet_position_budget_m: Callable[[int], float],
    volume_containment: bool | None = None,
) -> PointClassification:
    """Classify a point using a separate physical budget for each nearby facet."""

    point = _finite_point(position_m, "position_m")
    if not math.isfinite(candidate_padding_m) or candidate_padding_m < 0.0:
        raise ValueError("candidate_padding_m must be finite and nonnegative")
    candidates = query_segment_candidates(geometry, point, point, padding_m=candidate_padding_m)
    for facet_id in candidates:
        facet_id_value = int(facet_id)
        position_budget_m = facet_position_budget_m(facet_id_value)
        if not math.isfinite(position_budget_m) or position_budget_m < 0.0:
            raise ValueError("facet position budget must be finite and nonnegative")
        distance = _point_segment_distance(
            point,
            geometry.facet_start_m[facet_id_value],
            geometry.facet_end_m[facet_id_value],
        )
        if distance <= position_budget_m:
            return "boundary"
    if volume_containment is None:
        volume_containment = _inside_any_cell(geometry, point)
    if volume_containment:
        return "inside"
    return "outside"


def _volume_edge_incidence(
    geometry: GeometryData,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cells in (geometry.tri3, geometry.quad4):
        if cells is None:
            continue
        for cell in cells:
            for index, first_value in enumerate(cell):
                first = int(first_value)
                second = int(cell[(index + 1) % cell.size])
                key = (min(first, second), max(first, second))
                incidence.setdefault(key, []).append((first, second))
    return incidence


def _audit_volume_edges(incidence: dict[tuple[int, int], list[tuple[int, int]]]) -> None:
    for key, directed_edges in incidence.items():
        if len(directed_edges) > 2:
            raise GeometryPreparationError(f"non-manifold volume edge {key}")
        if len(directed_edges) == 2 and directed_edges[0] == directed_edges[1]:
            raise GeometryPreparationError(f"same-direction incidence on volume edge {key}")


def _audit_boundary_inventory(
    geometry: GeometryData,
    coordinate_system: CoordinateSystem,
    incidence: dict[tuple[int, int], list[tuple[int, int]]],
) -> None:
    physical_edges: set[tuple[int, int]] = set()
    for line in geometry.boundary.line2:
        first = int(line[0])
        second = int(line[1])
        key = (min(first, second), max(first, second))
        if key in physical_edges:
            raise GeometryPreparationError(f"duplicate boundary edge {key}")
        physical_edges.add(key)
        if _is_rz_axis_seam(geometry.nodes_m, key, coordinate_system):
            raise GeometryPreparationError(f"RZ axis seam cannot be a material boundary: {key}")
        edge_cells = incidence.get(key)
        if edge_cells is None:
            raise GeometryPreparationError(f"boundary edge {key} is not a volume-cell edge")
        if len(edge_cells) != 1:
            raise GeometryPreparationError(f"internal volume edge cannot be a boundary: {key}")

    if not physical_edges:
        return
    for key, edge_cells in incidence.items():
        if len(edge_cells) != 1:
            continue
        if _is_rz_axis_seam(geometry.nodes_m, key, coordinate_system):
            continue
        if key not in physical_edges:
            raise GeometryPreparationError(f"exterior volume edge is missing a boundary: {key}")


def _is_rz_axis_seam(
    nodes_m: FloatArray,
    edge: tuple[int, int],
    coordinate_system: CoordinateSystem,
) -> bool:
    return coordinate_system == "axisymmetric_rz" and (
        float(nodes_m[edge[0], 0]) == 0.0 and float(nodes_m[edge[1], 0]) == 0.0
    )


def _audit_boundary_geometry(
    geometry: GeometryData,
    coordinate_system: CoordinateSystem,
    start_m: FloatArray,
    end_m: FloatArray,
    bvh: _BvhArrays,
) -> None:
    if geometry.boundary.line2.shape[0] == 0:
        return
    degree = np.bincount(
        geometry.boundary.line2.reshape(-1),
        minlength=geometry.nodes_m.shape[0],
    )
    for node_id_value in np.flatnonzero(degree):
        node_id = int(node_id_value)
        node_degree = int(degree[node_id])
        on_rz_axis = (
            coordinate_system == "axisymmetric_rz" and float(geometry.nodes_m[node_id, 0]) == 0.0
        )
        if node_degree == 2 or (on_rz_axis and node_degree == 1):
            continue
        raise GeometryPreparationError(
            f"non-manifold boundary vertex {node_id} has degree {node_degree}"
        )

    lines = geometry.boundary.line2
    for first, second in _candidate_facet_pairs(start_m, end_m, bvh):
        shared = {int(value) for value in lines[first]}.intersection(
            int(value) for value in lines[second]
        )
        if shared:
            shared_node = next(iter(shared))
            if _adjacent_facets_overlap(lines[first], lines[second], shared_node, geometry.nodes_m):
                raise GeometryPreparationError(
                    f"boundary self-intersection between facets {first} and {second}"
                )
            continue
        if _segments_intersect(start_m[first], end_m[first], start_m[second], end_m[second]):
            raise GeometryPreparationError(
                f"boundary self-intersection between facets {first} and {second}"
            )


def _candidate_facet_pairs(
    start_m: FloatArray,
    end_m: FloatArray,
    bvh: _BvhArrays,
) -> Iterator[tuple[int, int]]:
    facet_lower = np.minimum(start_m, end_m)
    facet_upper = np.maximum(start_m, end_m)
    for first in range(start_m.shape[0]):
        stack = [0]
        while stack:
            node = stack.pop()
            if not _aabb_overlaps(
                facet_lower[first],
                facet_upper[first],
                np.asarray(bvh.lower_m[node]),
                np.asarray(bvh.upper_m[node]),
            ):
                continue
            left = bvh.left[node]
            if left < 0:
                for second in bvh.facet_id[bvh.begin[node] : bvh.end[node]]:
                    if second > first and _aabb_overlaps(
                        facet_lower[first],
                        facet_upper[first],
                        facet_lower[second],
                        facet_upper[second],
                    ):
                        yield first, second
                continue
            stack.append(bvh.right[node])
            stack.append(left)


def _adjacent_facets_overlap(
    first_line: Int64Array,
    second_line: Int64Array,
    shared_node: int,
    nodes_m: FloatArray,
) -> bool:
    first_other = next(int(value) for value in first_line if int(value) != shared_node)
    second_other = next(int(value) for value in second_line if int(value) != shared_node)
    shared = nodes_m[shared_node]
    first = nodes_m[first_other]
    second = nodes_m[second_other]
    if _orientation_sign(shared, first, second) != 0:
        return False
    return _dot_sign(shared, first, second) >= 0


def _segments_intersect(
    first_start: FloatArray,
    first_end: FloatArray,
    second_start: FloatArray,
    second_end: FloatArray,
) -> bool:
    orientations = (
        _orientation_sign(first_start, first_end, second_start),
        _orientation_sign(first_start, first_end, second_end),
        _orientation_sign(second_start, second_end, first_start),
        _orientation_sign(second_start, second_end, first_end),
    )
    if orientations[0] == 0 and _point_in_segment_box(second_start, first_start, first_end):
        return True
    if orientations[1] == 0 and _point_in_segment_box(second_end, first_start, first_end):
        return True
    if orientations[2] == 0 and _point_in_segment_box(first_start, second_start, second_end):
        return True
    if orientations[3] == 0 and _point_in_segment_box(first_end, second_start, second_end):
        return True
    return orientations[0] != orientations[1] and orientations[2] != orientations[3]


def _point_in_segment_box(point: FloatArray, start: FloatArray, end: FloatArray) -> bool:
    return bool(np.all(point >= np.minimum(start, end)) and np.all(point <= np.maximum(start, end)))


def _orientation_sign(first: FloatArray, second: FloatArray, third: FloatArray) -> int:
    first_x = Fraction.from_float(float(first[0]))
    first_y = Fraction.from_float(float(first[1]))
    second_x = Fraction.from_float(float(second[0]))
    second_y = Fraction.from_float(float(second[1]))
    third_x = Fraction.from_float(float(third[0]))
    third_y = Fraction.from_float(float(third[1]))
    determinant = (second_x - first_x) * (third_y - first_y) - (second_y - first_y) * (
        third_x - first_x
    )
    return int(determinant > 0) - int(determinant < 0)


def _dot_sign(origin: FloatArray, first: FloatArray, second: FloatArray) -> int:
    origin_x = Fraction.from_float(float(origin[0]))
    origin_y = Fraction.from_float(float(origin[1]))
    first_x = Fraction.from_float(float(first[0]))
    first_y = Fraction.from_float(float(first[1]))
    second_x = Fraction.from_float(float(second[0]))
    second_y = Fraction.from_float(float(second[1]))
    product = (first_x - origin_x) * (second_x - origin_x) + (first_y - origin_y) * (
        second_y - origin_y
    )
    return int(product > 0) - int(product < 0)


def _geometry_bbox_diagonal(nodes_m: FloatArray) -> float:
    lower = np.min(nodes_m, axis=0)
    upper = np.max(nodes_m, axis=0)
    extent = _finite_difference(upper, lower, "geometry bounding box")
    diagonal = math.hypot(float(extent[0]), float(extent[1]))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise GeometryPreparationError("geometry has no finite positive bounding-box scale")
    return diagonal


def _build_bvh(start_m: FloatArray, end_m: FloatArray) -> _BvhArrays:
    result = _BvhArrays([], [], [], [], [], [], [], [])
    if start_m.shape[0] == 0:
        return result
    facet_lower = np.minimum(start_m, end_m)
    facet_upper = np.maximum(start_m, end_m)
    centroid = 0.5 * facet_lower + 0.5 * facet_upper
    if not bool(np.isfinite(centroid).all()):
        raise GeometryPreparationError("boundary centroid is not finite")

    def append_node(facet_ids: Int64Array) -> int:
        node_id = len(result.left)
        lower = np.min(facet_lower[facet_ids], axis=0)
        upper = np.max(facet_upper[facet_ids], axis=0)
        result.lower_m.append((float(lower[0]), float(lower[1])))
        result.upper_m.append((float(upper[0]), float(upper[1])))
        result.left.append(-1)
        result.right.append(-1)
        result.begin.append(-1)
        result.end.append(-1)
        result.skip.append(-1)
        if facet_ids.size <= BVH_LEAF_SIZE:
            begin = len(result.facet_id)
            result.facet_id.extend(int(item) for item in np.sort(facet_ids))
            result.begin[node_id] = begin
            result.end[node_id] = len(result.facet_id)
            result.skip[node_id] = len(result.left)
            return node_id

        centroid_span = np.ptp(centroid[facet_ids], axis=0)
        axis = 0 if float(centroid_span[0]) >= float(centroid_span[1]) else 1
        order = np.lexsort((facet_ids, centroid[facet_ids, axis]))
        ordered = facet_ids[order]
        middle = int(ordered.size) // 2
        result.left[node_id] = append_node(ordered[:middle])
        result.right[node_id] = append_node(ordered[middle:])
        result.skip[node_id] = len(result.left)
        return node_id

    append_node(np.arange(start_m.shape[0], dtype=np.int64))
    return result


def _build_volume_bvh(geometry: GeometryData) -> _VolumeBvhArrays:
    """Build a deterministic stackless AABB tree over all volume cells."""

    tri_count = 0 if geometry.tri3 is None else int(geometry.tri3.shape[0])
    quad_count = 0 if geometry.quad4 is None else int(geometry.quad4.shape[0])
    cell_count = tri_count + quad_count
    tri_edge_length = np.empty((tri_count, 3), dtype="<f8")
    quad_edge_length = np.empty((quad_count, 4), dtype="<f8")
    if cell_count == 0:
        return _VolumeBvhArrays(
            _read_only(np.empty(0, dtype="<i8")),
            _read_only(np.empty((0, 2), dtype="<f8")),
            _read_only(np.empty((0, 2), dtype="<f8")),
            _read_only(np.empty(0, dtype="<i8")),
            _read_only(np.empty(0, dtype="<i8")),
            _read_only(np.empty(0, dtype="<i8")),
            _read_only(tri_edge_length),
            _read_only(quad_edge_length),
            0,
        )
    raw_lower = np.empty((cell_count, 2), dtype="<f8")
    raw_upper = np.empty((cell_count, 2), dtype="<f8")
    node_spacing = np.empty(cell_count, dtype="<f8")
    cursor = 0
    for connectivity in (geometry.tri3, geometry.quad4):
        if connectivity is None:
            continue
        stop = cursor + int(connectivity.shape[0])
        cell_nodes = geometry.nodes_m[connectivity]
        edge = np.roll(cell_nodes, -1, axis=1) - cell_nodes
        length = _canonical_volume_edge_lengths(edge)
        if connectivity.shape[1] == 3:
            tri_edge_length[:] = length
        else:
            quad_edge_length[:] = length
        if not bool(np.isfinite(length).all()) or bool((length <= 0.0).any()):
            raise GeometryPreparationError("volume cell contains an unresolved edge")
        cell_lower = np.min(cell_nodes, axis=1)
        cell_upper = np.max(cell_nodes, axis=1)
        cell_spacing = np.max(np.abs(np.spacing(cell_nodes)), axis=(1, 2))
        cell_extent = cell_upper - cell_lower
        cell_diameter = np.hypot(cell_extent[:, 0], cell_extent[:, 1])
        _certify_volume_cell_resolution(cell_nodes, edge, length, cell_diameter)
        raw_lower[cursor:stop] = cell_lower
        raw_upper[cursor:stop] = cell_upper
        node_spacing[cursor:stop] = cell_spacing
        cursor = stop
    extent = raw_upper - raw_lower
    diameter_bound = np.hypot(extent[:, 0], extent[:, 1])
    base_padding = np.nextafter(
        _VOLUME_INDEX_ULPS * np.maximum(node_spacing, _FLOAT_EPS * diameter_bound),
        np.inf,
    )
    float_limit = np.finfo(np.float64).max
    with np.errstate(over="ignore", invalid="ignore"):
        lower = np.maximum(
            np.nextafter(raw_lower - base_padding[:, None], -np.inf),
            -float_limit,
        )
        upper = np.minimum(
            np.nextafter(raw_upper + base_padding[:, None], np.inf),
            float_limit,
        )
    if not bool(np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise GeometryPreparationError("volume-cell index bounds are not finite")
    centroid = 0.5 * raw_lower + 0.5 * raw_upper
    node_count = _volume_bvh_node_count(cell_count)
    ordered_cell_id = np.empty(cell_count, dtype="<i8")
    node_lower = np.empty((node_count, 2), dtype="<f8")
    node_upper = np.empty((node_count, 2), dtype="<f8")
    node_begin = np.full(node_count, -1, dtype="<i8")
    node_end = np.full(node_count, -1, dtype="<i8")
    node_skip = np.empty(node_count, dtype="<i8")
    next_node = 0
    next_cell = 0
    del raw_lower, raw_upper, node_spacing, extent, diameter_bound, base_padding

    def append_node(cell_ids: Int64Array) -> None:
        nonlocal next_cell, next_node
        node_id = next_node
        next_node += 1
        node_lower[node_id] = np.min(lower[cell_ids], axis=0)
        node_upper[node_id] = np.max(upper[cell_ids], axis=0)
        if cell_ids.size <= BVH_LEAF_SIZE:
            ordered = np.sort(cell_ids)
            node_begin[node_id] = next_cell
            ordered_cell_id[next_cell : next_cell + ordered.size] = ordered
            next_cell += int(ordered.size)
            node_end[node_id] = next_cell
        else:
            span = np.ptp(centroid[cell_ids], axis=0)
            axis = 0 if float(span[0]) >= float(span[1]) else 1
            order = np.lexsort((cell_ids, centroid[cell_ids, axis]))
            sorted_ids = cell_ids[order]
            middle = int(sorted_ids.size) // 2
            append_node(sorted_ids[:middle])
            append_node(sorted_ids[middle:])
        node_skip[node_id] = next_node

    append_node(np.arange(cell_count, dtype=np.int64))
    if next_node != node_count or next_cell != cell_count:
        raise GeometryPreparationError("volume-cell index construction is incomplete")
    arrays = (
        ordered_cell_id,
        node_lower,
        node_upper,
        node_begin,
        node_end,
        node_skip,
        tri_edge_length,
        quad_edge_length,
    )
    for array in arrays:
        array.setflags(write=False)
    return _VolumeBvhArrays(
        *arrays,
        _VOLUME_INDEX_BUILD_WORK_BYTES_PER_CELL * cell_count,
    )


def _canonical_volume_edge_lengths(edge_m: FloatArray) -> FloatArray:
    """Evaluate edge lengths with the scalar predicate's CPython arithmetic."""

    result = np.empty(edge_m.shape[:2], dtype="<f8")
    for cell_index in range(edge_m.shape[0]):
        for edge_index in range(edge_m.shape[1]):
            result[cell_index, edge_index] = math.hypot(
                float(edge_m[cell_index, edge_index, 0]),
                float(edge_m[cell_index, edge_index, 1]),
            )
    return result


def _volume_bvh_node_count(cell_count: int) -> int:
    """Return the exact flat-tree size for deterministic half splits."""

    pending = [cell_count]
    result = 0
    while pending:
        size = pending.pop()
        result += 1
        if size > BVH_LEAF_SIZE:
            lower = size // 2
            pending.extend((lower, size - lower))
    return result


def _certify_volume_cell_resolution(
    cell_nodes_m: FloatArray,
    edge_m: FloatArray,
    edge_length_m: FloatArray,
    diameter_m: FloatArray,
) -> None:
    """Reject cells whose half-space signs are below float64 spatial resolution."""

    resolution_m = np.nextafter(
        _VOLUME_INDEX_ULPS
        * np.maximum(
            _FLOAT_EPS * diameter_m,
            np.abs(np.spacing(diameter_m)),
        ),
        np.inf,
    )
    minimum_edge_clearance_m = np.full(cell_nodes_m.shape[0], np.inf, dtype="<f8")
    node_count = int(cell_nodes_m.shape[1])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for edge_index in range(node_count):
            start = cell_nodes_m[:, edge_index]
            normal0 = -edge_m[:, edge_index, 1] / edge_length_m[:, edge_index]
            normal1 = edge_m[:, edge_index, 0] / edge_length_m[:, edge_index]
            edge_clearance_m = np.full(cell_nodes_m.shape[0], np.inf, dtype="<f8")
            for node_index in range(node_count):
                if node_index in (edge_index, (edge_index + 1) % node_count):
                    continue
                offset = cell_nodes_m[:, node_index] - start
                first = normal0 * offset[:, 0]
                second = normal1 * offset[:, 1]
                total = first + second
                second_virtual = total - first
                error = (first - (total - second_virtual)) + (second - second_virtual)
                clearance = total + error
                edge_clearance_m = np.minimum(edge_clearance_m, clearance)
            minimum_edge_clearance_m = np.minimum(
                minimum_edge_clearance_m,
                edge_clearance_m,
            )
    if bool(
        (
            ~np.isfinite(resolution_m)
            | ~np.isfinite(minimum_edge_clearance_m)
            | (np.min(edge_length_m, axis=1) <= resolution_m)
            | (minimum_edge_clearance_m <= resolution_m)
        ).any()
    ):
        raise GeometryPreparationError("volume cell is unresolved at float64 precision")


def _inside_any_cell(geometry: PreparedGeometry, point_m: FloatArray) -> bool:
    status = _inside_volume_point_kernel(
        geometry.nodes_m,
        _EMPTY_TRI3 if geometry.tri3 is None else geometry.tri3,
        _EMPTY_QUAD4 if geometry.quad4 is None else geometry.quad4,
        geometry.tri3_edge_length_m,
        geometry.quad4_edge_length_m,
        geometry.volume_bvh_cell_id,
        geometry.volume_bvh_lower_m,
        geometry.volume_bvh_upper_m,
        geometry.volume_bvh_begin,
        geometry.volume_bvh_end,
        geometry.volume_bvh_skip,
        float(point_m[0]),
        float(point_m[1]),
    )
    if status < 0:
        raise GeometryPreparationError("volume cell contains an unresolved edge")
    return status == 1


def points_inside_volume(geometry: PreparedGeometry, positions_m: FloatArray) -> NDArray[np.bool_]:
    """Classify finite points against the prepared volume-cell union."""

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_m must have shape (N, 2)")
    if not bool(np.isfinite(positions).all()):
        raise ValueError("positions_m must contain only finite values")
    contained, invalid_geometry = _inside_volume_batch_kernel(
        geometry.nodes_m,
        _EMPTY_TRI3 if geometry.tri3 is None else geometry.tri3,
        _EMPTY_QUAD4 if geometry.quad4 is None else geometry.quad4,
        geometry.tri3_edge_length_m,
        geometry.quad4_edge_length_m,
        geometry.volume_bvh_cell_id,
        geometry.volume_bvh_lower_m,
        geometry.volume_bvh_upper_m,
        geometry.volume_bvh_begin,
        geometry.volume_bvh_end,
        geometry.volume_bvh_skip,
        positions,
    )
    if invalid_geometry:
        raise GeometryPreparationError("volume cell contains an unresolved edge")
    return contained


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _volume_two_term_sum(first: float, second: float) -> float:
    """Match the scalar predicate's accurate sum of two rounded products."""

    total = first + second
    second_virtual = total - first
    error = (first - (total - second_virtual)) + (second - second_virtual)
    return total + error


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _volume_index_excludes_point(
    lower0: float,
    lower1: float,
    upper0: float,
    upper1: float,
    point0: float,
    point1: float,
    point_padding: float,
) -> bool:
    """Exclude only boxes separated beyond the query-coordinate roundoff budget."""

    return (
        (point0 < lower0 and np.nextafter(lower0 - point0, -np.inf) > point_padding)
        or (point0 > upper0 and np.nextafter(point0 - upper0, -np.inf) > point_padding)
        or (point1 < lower1 and np.nextafter(lower1 - point1, -np.inf) > point_padding)
        or (point1 > upper1 and np.nextafter(point1 - upper1, -np.inf) > point_padding)
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _inside_volume_cell_kernel(
    nodes_m: FloatArray,
    connectivity: Int64Array,
    edge_length_m: FloatArray,
    cell_id: int,
    node_count: int,
    point0: float,
    point1: float,
) -> int:
    """Return -1 for invalid geometry, 0 outside, or 1 inside one convex cell."""

    for index in range(node_count):
        start_id = connectivity[cell_id, index]
        end_id = connectivity[cell_id, (index + 1) % node_count]
        edge0 = nodes_m[end_id, 0] - nodes_m[start_id, 0]
        edge1 = nodes_m[end_id, 1] - nodes_m[start_id, 1]
        length = edge_length_m[cell_id, index]
        offset0 = point0 - nodes_m[start_id, 0]
        offset1 = point1 - nodes_m[start_id, 1]
        if (
            not np.isfinite(edge0)
            or not np.isfinite(edge1)
            or not np.isfinite(length)
            or length <= 0.0
            or not np.isfinite(offset0)
            or not np.isfinite(offset1)
        ):
            return -1
        signed_inward_distance = _volume_two_term_sum(
            (-edge1 / length) * offset0,
            (edge0 / length) * offset1,
        )
        if not np.isfinite(signed_inward_distance):
            return -1
        if signed_inward_distance < 0.0:
            return 0
    return 1


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _inside_volume_point_kernel(
    nodes_m: FloatArray,
    tri3: Int64Array,
    quad4: Int64Array,
    tri3_edge_length_m: FloatArray,
    quad4_edge_length_m: FloatArray,
    index_cell_id: Int64Array,
    index_lower_m: FloatArray,
    index_upper_m: FloatArray,
    index_begin: Int64Array,
    index_end: Int64Array,
    index_skip: Int64Array,
    point0: float,
    point1: float,
) -> int:
    """Return the containment status from one stackless volume-cell traversal."""

    point_padding = np.nextafter(
        _VOLUME_INDEX_ULPS
        * max(
            abs(float(np.spacing(np.float64(point0)))),
            abs(float(np.spacing(np.float64(point1)))),
        ),
        np.inf,
    )
    tri_count = tri3.shape[0]
    node = 0
    while node < index_skip.size:
        if _volume_index_excludes_point(
            index_lower_m[node, 0],
            index_lower_m[node, 1],
            index_upper_m[node, 0],
            index_upper_m[node, 1],
            point0,
            point1,
            point_padding,
        ):
            node = index_skip[node]
            continue
        begin = index_begin[node]
        if begin < 0:
            node += 1
            continue
        for offset in range(begin, index_end[node]):
            cell_id = index_cell_id[offset]
            if cell_id < tri_count:
                status = _inside_volume_cell_kernel(
                    nodes_m,
                    tri3,
                    tri3_edge_length_m,
                    cell_id,
                    3,
                    point0,
                    point1,
                )
            else:
                status = _inside_volume_cell_kernel(
                    nodes_m,
                    quad4,
                    quad4_edge_length_m,
                    cell_id - tri_count,
                    4,
                    point0,
                    point1,
                )
            if status != 0:
                return status
        node = index_skip[node]
    return 0


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _inside_volume_batch_kernel(
    nodes_m: FloatArray,
    tri3: Int64Array,
    quad4: Int64Array,
    tri3_edge_length_m: FloatArray,
    quad4_edge_length_m: FloatArray,
    index_cell_id: Int64Array,
    index_lower_m: FloatArray,
    index_upper_m: FloatArray,
    index_begin: Int64Array,
    index_end: Int64Array,
    index_skip: Int64Array,
    positions_m: FloatArray,
) -> tuple[NDArray[np.bool_], bool]:
    """Apply the point predicate to one contiguous table-source batch."""

    result = np.empty(positions_m.shape[0], dtype=np.bool_)
    invalid_geometry = False
    for index in range(positions_m.shape[0]):
        status = _inside_volume_point_kernel(
            nodes_m,
            tri3,
            quad4,
            tri3_edge_length_m,
            quad4_edge_length_m,
            index_cell_id,
            index_lower_m,
            index_upper_m,
            index_begin,
            index_end,
            index_skip,
            positions_m[index, 0],
            positions_m[index, 1],
        )
        if status < 0:
            invalid_geometry = True
        result[index] = status == 1
    return result, invalid_geometry


def _point_segment_distance(point_m: FloatArray, start_m: FloatArray, end_m: FloatArray) -> float:
    edge = _finite_difference(end_m, start_m, "boundary distance")
    length = math.hypot(float(edge[0]), float(edge[1]))
    if not math.isfinite(length) or length <= 0.0:
        raise GeometryPreparationError("boundary contains an unresolved facet")
    offset = _finite_difference(point_m, start_m, "boundary distance")
    along = math.fsum(
        (
            float(offset[0]) * (float(edge[0]) / length),
            float(offset[1]) * (float(edge[1]) / length),
        )
    )
    parameter = min(max(along / length, 0.0), 1.0)
    projected = start_m + parameter * edge
    residual = _finite_difference(point_m, projected, "boundary distance")
    distance = math.hypot(float(residual[0]), float(residual[1]))
    if not math.isfinite(distance):
        raise GeometryPreparationError("boundary distance is not finite")
    return distance


def _aabb_overlaps(
    first_lower: FloatArray,
    first_upper: FloatArray,
    second_lower: FloatArray,
    second_upper: FloatArray,
) -> bool:
    return bool(np.all(first_upper >= second_lower) and np.all(second_upper >= first_lower))


def _finite_point(value: FloatArray, label: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"{label} must have shape (2,)")
    if not bool(np.isfinite(point).all()):
        raise ValueError(f"{label} must contain only finite values")
    return point


def _finite_difference(first: FloatArray, second: FloatArray, label: str) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = first - second
    if not bool(np.isfinite(difference).all()):
        raise GeometryPreparationError(f"{label} exceeds the finite float64 range")
    return difference


def _read_only[DType: np.generic](array: NDArray[DType]) -> NDArray[DType]:
    array.setflags(write=False)
    return array


def _read_only_copy[DType: np.generic](array: NDArray[DType]) -> NDArray[DType]:
    return _read_only(array.copy())
