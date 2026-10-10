"""Prepared static two-dimensional topology transfers independent of wall laws."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .geometry import PreparedGeometry

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type Int32Array = NDArray[np.int32]
type UInt8Array = NDArray[np.uint8]

TOPOLOGY_ALGORITHM_REVISION = "translation_periodic_xy_v1"
_FLOAT_EPS = np.finfo(np.float64).eps
TOPOLOGY_CANDIDATE_INVALID = np.uint8(0)
TOPOLOGY_CANDIDATE_MATERIAL = np.uint8(1)
TOPOLOGY_CANDIDATE_PERIODIC = np.uint8(2)


class TopologyPreparationError(RuntimeError):
    """A requested topology cannot be mapped unambiguously onto the geometry."""


@dataclass(frozen=True, slots=True)
class TranslationPairRequest:
    """One resolved pair of boundary group IDs and its directed translation."""

    first_group_id: int
    second_group_id: int
    first_to_second_m: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PreparedPeriodicTopology:
    """Reciprocal per-facet maps for static Cartesian translational seams."""

    peer_facet_id: Int64Array
    peer_node_ids: Int64Array
    translation_m: FloatArray
    pair_id: Int32Array
    facet_is_periodic: NDArray[np.bool_]
    periodic_facet_id: Int64Array
    field_match_rtol: float
    position_tolerance_m: float

    @property
    def pair_count(self) -> int:
        """Return the number of configured reciprocal seam pairs."""

        if self.periodic_facet_id.size == 0:
            return 0
        return int(np.max(self.pair_id[self.periodic_facet_id])) + 1


@dataclass(frozen=True, slots=True)
class PeriodicCandidateClassification:
    """Per-event topology kind and canonical periodic source facet."""

    kind: UInt8Array
    primary_periodic_facet_id: Int64Array


def classify_periodic_candidate_rows(
    topology: PreparedPeriodicTopology,
    candidate_offsets: Int64Array,
    candidate_facet_ids: Int64Array,
) -> PeriodicCandidateClassification:
    """Classify simultaneous facets without invoking a material wall law."""

    offsets = np.asarray(candidate_offsets)
    candidates = np.asarray(candidate_facet_ids)
    if offsets.dtype != np.dtype(np.int64) or offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("periodic candidate offsets must be a nonempty int64 vector")
    if candidates.dtype != np.dtype(np.int64) or candidates.ndim != 1:
        raise ValueError("periodic candidate facets must be an int64 vector")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != candidates.size
        or bool((offsets[1:] < offsets[:-1]).any())
    ):
        raise ValueError("periodic candidate CSR offsets are inconsistent")
    if bool(((candidates < 0) | (candidates >= topology.facet_is_periodic.size)).any()):
        raise ValueError("periodic candidate facet is outside the prepared topology")
    kind = np.empty(offsets.size - 1, dtype="<u1")
    primary = np.empty(offsets.size - 1, dtype="<i8")
    _classify_periodic_candidate_rows_kernel(
        offsets,
        candidates,
        topology.facet_is_periodic,
        topology.translation_m,
        kind,
        primary,
    )
    return PeriodicCandidateClassification(kind, primary)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _classify_periodic_candidate_rows_kernel(
    offsets: Int64Array,
    candidates: Int64Array,
    facet_is_periodic: NDArray[np.bool_],
    translation_m: FloatArray,
    kind: UInt8Array,
    primary: Int64Array,
) -> None:
    for row in range(kind.size):
        begin = offsets[row]
        end = offsets[row + 1]
        if begin == end:
            kind[row] = TOPOLOGY_CANDIDATE_INVALID
            primary[row] = -1
            continue
        first = candidates[begin]
        periodic = facet_is_periodic[first]
        selected_primary = first
        valid = True
        translation_x = translation_m[first, 0]
        translation_y = translation_m[first, 1]
        for offset in range(begin + 1, end):
            facet = candidates[offset]
            if facet_is_periodic[facet] != periodic:
                valid = False
                break
            if periodic and (
                translation_m[facet, 0] != translation_x or translation_m[facet, 1] != translation_y
            ):
                valid = False
                break
            selected_primary = min(selected_primary, facet)
        if not valid:
            kind[row] = TOPOLOGY_CANDIDATE_INVALID
            primary[row] = -1
        elif periodic:
            kind[row] = TOPOLOGY_CANDIDATE_PERIODIC
            primary[row] = selected_primary
        else:
            kind[row] = TOPOLOGY_CANDIDATE_MATERIAL
            primary[row] = -1


def prepare_periodic_topology(
    geometry: PreparedGeometry,
    requests: Sequence[TranslationPairRequest],
    *,
    field_match_rtol: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> PreparedPeriodicTopology | None:
    """Pair conforming translated facets without assigning a material wall law."""

    if not requests:
        return None
    _validate_preparation_controls(geometry, field_match_rtol, geometry_rtol, roundoff_ulps)

    tolerance_m = _position_tolerance_m(geometry, geometry_rtol, roundoff_ulps)
    normal_tolerance = max(geometry_rtol, float(roundoff_ulps) * _FLOAT_EPS)
    facet_count = geometry.facet_count
    peer_facet_id = np.full(facet_count, -1, dtype="<i8")
    peer_node_ids = np.full((facet_count, 2), -1, dtype="<i8")
    translation_m = np.zeros((facet_count, 2), dtype="<f8")
    pair_id = np.full(facet_count, -1, dtype="<i4")
    facet_is_periodic = np.zeros(facet_count, dtype=np.bool_)
    used_groups: set[int] = set()

    for topology_pair_id, request in enumerate(requests):
        _bind_group_pair(
            geometry,
            request,
            topology_pair_id,
            tolerance_m,
            normal_tolerance,
            used_groups,
            peer_facet_id,
            peer_node_ids,
            translation_m,
            pair_id,
            facet_is_periodic,
        )

    periodic_facet_id = np.flatnonzero(facet_is_periodic).astype("<i8", copy=False)
    for array in (
        peer_facet_id,
        peer_node_ids,
        translation_m,
        pair_id,
        facet_is_periodic,
        periodic_facet_id,
    ):
        array.setflags(write=False)
    return PreparedPeriodicTopology(
        peer_facet_id,
        peer_node_ids,
        translation_m,
        pair_id,
        facet_is_periodic,
        periodic_facet_id,
        field_match_rtol,
        tolerance_m,
    )


def _validate_preparation_controls(
    geometry: PreparedGeometry,
    field_match_rtol: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> None:
    if geometry.coordinate_system != "cartesian_xy":
        raise TopologyPreparationError(
            "translation_periodic_xy_v1 supports only Cartesian XY geometry"
        )
    if not math.isfinite(field_match_rtol) or not 0.0 < field_match_rtol < 1.0:
        raise ValueError("periodic field_match_rtol must be finite and between zero and one")
    if not math.isfinite(geometry_rtol) or not 0.0 < geometry_rtol < 1.0:
        raise ValueError("periodic geometry_rtol must be finite and between zero and one")
    if isinstance(roundoff_ulps, bool) or not isinstance(roundoff_ulps, int):
        raise ValueError("periodic roundoff_ulps must be an integer")
    if roundoff_ulps <= 0:
        raise ValueError("periodic roundoff_ulps must be positive")


def _bind_group_pair(
    geometry: PreparedGeometry,
    request: TranslationPairRequest,
    topology_pair_id: int,
    tolerance_m: float,
    normal_tolerance: float,
    used_groups: set[int],
    peer_facet_id: Int64Array,
    peer_node_ids: Int64Array,
    translation_m: FloatArray,
    pair_id: Int32Array,
    facet_is_periodic: NDArray[np.bool_],
) -> None:
    first_group = _group_id(request.first_group_id, "first_group_id")
    second_group = _group_id(request.second_group_id, "second_group_id")
    if first_group == second_group:
        raise TopologyPreparationError("a periodic pair requires two different groups")
    if first_group in used_groups or second_group in used_groups:
        raise TopologyPreparationError("a periodic boundary group is used more than once")
    used_groups.update((first_group, second_group))
    translation = np.asarray(request.first_to_second_m, dtype=np.float64)
    if translation.shape != (2,) or not bool(np.isfinite(translation).all()):
        raise ValueError("periodic translation must contain two finite components")
    if bool((translation == 0.0).all()):
        raise ValueError("periodic translation must be nonzero")
    first_facets = np.flatnonzero(geometry.group_id == first_group).astype(np.int64, copy=False)
    second_facets = np.flatnonzero(geometry.group_id == second_group).astype(np.int64, copy=False)
    if first_facets.size == 0 or second_facets.size == 0:
        raise TopologyPreparationError("a periodic group has no boundary facets")
    if first_facets.size != second_facets.size:
        raise TopologyPreparationError("periodic groups have different facet counts")

    with np.errstate(over="ignore", invalid="ignore"):
        translated_start = geometry.facet_start_m[first_facets] + translation
        translated_end = geometry.facet_end_m[first_facets] + translation
    if not bool(np.isfinite(translated_start).all() and np.isfinite(translated_end).all()):
        raise TopologyPreparationError("periodic translated facet coordinates are not finite")
    first_order = _canonical_segment_order(translated_start, translated_end, first_facets)
    second_order = _canonical_segment_order(
        geometry.facet_start_m[second_facets],
        geometry.facet_end_m[second_facets],
        second_facets,
    )
    for first_facet, second_facet in zip(
        first_facets[first_order], second_facets[second_order], strict=True
    ):
        _bind_facet_pair(
            geometry,
            int(first_facet),
            int(second_facet),
            translation,
            topology_pair_id,
            tolerance_m,
            normal_tolerance,
            peer_facet_id,
            peer_node_ids,
            translation_m,
            pair_id,
            facet_is_periodic,
        )


def _position_tolerance_m(
    geometry: PreparedGeometry, geometry_rtol: float, roundoff_ulps: int
) -> float:
    spacing = float(np.max(np.abs(np.spacing(geometry.nodes_m))))
    scale_spacing = abs(math.ulp(float(geometry.bbox_diagonal_m)))
    return max(
        geometry_rtol * float(geometry.bbox_diagonal_m),
        float(roundoff_ulps) * max(spacing, scale_spacing),
    )


def _group_id(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"periodic {name} must be a nonnegative integer")
    return value


def _canonical_segment_order(
    start_m: FloatArray, end_m: FloatArray, facet_id: Int64Array
) -> Int64Array:
    start_precedes = (start_m[:, 0] < end_m[:, 0]) | (
        (start_m[:, 0] == end_m[:, 0]) & (start_m[:, 1] <= end_m[:, 1])
    )
    lower = np.where(start_precedes[:, None], start_m, end_m)
    upper = np.where(start_precedes[:, None], end_m, start_m)
    return np.lexsort(
        (
            facet_id,
            upper[:, 1],
            upper[:, 0],
            lower[:, 1],
            lower[:, 0],
        )
    ).astype(np.int64, copy=False)


def _bind_facet_pair(
    geometry: PreparedGeometry,
    first_facet: int,
    second_facet: int,
    translation: FloatArray,
    topology_pair_id: int,
    tolerance_m: float,
    normal_tolerance: float,
    peer_facet_id: Int64Array,
    peer_node_ids: Int64Array,
    translation_m: FloatArray,
    pair_id: Int32Array,
    facet_is_periodic: NDArray[np.bool_],
) -> None:
    first_start = geometry.facet_start_m[first_facet] + translation
    first_end = geometry.facet_end_m[first_facet] + translation
    second_start = geometry.facet_start_m[second_facet]
    second_end = geometry.facet_end_m[second_facet]
    direct_error = max(
        float(np.max(np.abs(first_start - second_start))),
        float(np.max(np.abs(first_end - second_end))),
    )
    reverse_error = max(
        float(np.max(np.abs(first_start - second_end))),
        float(np.max(np.abs(first_end - second_start))),
    )
    reverse = reverse_error < direct_error
    endpoint_error = reverse_error if reverse else direct_error
    if endpoint_error > tolerance_m:
        raise TopologyPreparationError(
            "periodic facets do not match after the configured translation"
        )
    if (
        abs(
            float(geometry.facet_length_m[first_facet])
            - float(geometry.facet_length_m[second_facet])
        )
        > tolerance_m
    ):
        raise TopologyPreparationError("periodic facet lengths do not match")
    normal_sum = geometry.facet_normal[first_facet] + geometry.facet_normal[second_facet]
    if math.hypot(float(normal_sum[0]), float(normal_sum[1])) > normal_tolerance:
        raise TopologyPreparationError("periodic facets must have opposite outward normals")
    if peer_facet_id[first_facet] >= 0 or peer_facet_id[second_facet] >= 0:
        raise TopologyPreparationError("a periodic facet was paired more than once")

    first_nodes = geometry.facet_node_ids[first_facet]
    second_nodes = geometry.facet_node_ids[second_facet]
    mapped_second = second_nodes[::-1] if reverse else second_nodes
    mapped_first = first_nodes[::-1] if reverse else first_nodes
    peer_facet_id[first_facet] = second_facet
    peer_facet_id[second_facet] = first_facet
    peer_node_ids[first_facet] = mapped_second
    peer_node_ids[second_facet] = mapped_first
    translation_m[first_facet] = translation
    translation_m[second_facet] = -translation
    pair_id[first_facet] = topology_pair_id
    pair_id[second_facet] = topology_pair_id
    facet_is_periodic[first_facet] = True
    facet_is_periodic[second_facet] = True


__all__ = [
    "TOPOLOGY_ALGORITHM_REVISION",
    "TOPOLOGY_CANDIDATE_INVALID",
    "TOPOLOGY_CANDIDATE_MATERIAL",
    "TOPOLOGY_CANDIDATE_PERIODIC",
    "PeriodicCandidateClassification",
    "PreparedPeriodicTopology",
    "TopologyPreparationError",
    "TranslationPairRequest",
    "classify_periodic_candidate_rows",
    "prepare_periodic_topology",
]
