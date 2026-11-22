"""Independent scalar reference for field verification, outside production.

The explicit algorithms below preserve the pre-refactor scalar oracle; normal
clients use PreparedFieldSet and the sole prepared numerical sampler instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from chamber_particles.case_format import (
    FieldData,
    Layout,
    P1TriLayout,
    Q1QuadLayout,
    RegularLayout,
)
from chamber_particles.field_numerics import FieldLocationError

type FloatArray = NDArray[np.float64]
type OutsideReason = Literal["outside_layout", "masked_cell"]
type Weights = tuple[float, ...]

FIELD_LOCATION_ULPS = 64.0
_FLOAT_EPS = np.finfo(np.float64).eps
_MAX_JACOBIAN_CONDITION = 1.0 / math.sqrt(_FLOAT_EPS)
_MAX_REFERENCE_UNCERTAINTY = 8.0 * math.sqrt(_FLOAT_EPS)
_Q1_MAX_ITERATIONS = 20


@dataclass(frozen=True, slots=True)
class FieldLocation:
    """One deterministic cell location and interpolation basis."""

    layout_name: str
    cell_id: int
    node_ids: tuple[int, ...]
    weights: tuple[float, ...]
    support_inside: bool
    outside_reason: OutsideReason | None


@dataclass(frozen=True, slots=True)
class SampleResult:
    """A sampled value kept separate from its spatial-support classification."""

    value: FloatArray
    support_inside: bool
    cell_id: int
    outside_reason: OutsideReason | None


@dataclass(frozen=True, slots=True)
class _Candidate:
    cell_id: int
    node_ids: tuple[int, ...]
    weights: Weights
    supported: bool


def locate_field_cell(
    layout: Layout, position_m: FloatArray, *, cell_hint: int = -1
) -> FieldLocation:
    """Locate one point, retaining a finite provisional basis outside support.

    ``cell_hint`` is a search hint only. The caller owns when a trial hint becomes
    committed state; this function never mutates either the layout or the hint.
    """

    position = _finite_position(position_m)
    _validate_cell_hint(layout, cell_hint)
    if isinstance(layout, RegularLayout):
        return _locate_regular(layout, position)
    if isinstance(layout, P1TriLayout):
        return _locate_p1(layout, position, cell_hint)
    return _locate_q1(layout, position, cell_hint)


def sample_field(
    field: FieldData,
    location: FieldLocation,
    *,
    time_s: float | None = None,
) -> SampleResult:
    """Evaluate one canonical field at a location produced for its layout."""

    if field.layout != location.layout_name:
        raise ValueError(
            f"field {field.name} uses layout {field.layout}, not {location.layout_name}"
        )
    values = field.values
    if field.time_s is not None:
        if time_s is None:
            raise ValueError("time_s is required for a time-dependent field")
        values = _interpolate_snapshot_values(field, time_s)
    with np.errstate(over="ignore", invalid="ignore"):
        if field.association == "node":
            node_ids = np.asarray(location.node_ids, dtype=np.int64)
            weights = np.asarray(location.weights, dtype=np.float64)
            value = weights @ values[node_ids]
        else:
            value = values[location.cell_id].copy()
    sampled = np.asarray(value, dtype=np.float64)
    if not bool(np.isfinite(sampled).all()):
        raise FieldLocationError(f"field {field.name!r} produced a non-finite sampled value")
    return SampleResult(
        value=sampled,
        support_inside=location.support_inside,
        cell_id=location.cell_id,
        outside_reason=location.outside_reason,
    )


def _interpolate_snapshot_values(field: FieldData, time_s: float) -> FloatArray:
    snapshot_time_s = field.time_s
    if snapshot_time_s is None:
        return field.values
    query = float(time_s)
    if (
        not math.isfinite(query)
        or query < float(snapshot_time_s[0])
        or query > float(snapshot_time_s[-1])
    ):
        raise FieldLocationError(
            f"field {field.name!r} cannot sample time outside its snapshot range"
        )
    lower = int(np.searchsorted(snapshot_time_s, query, side="right")) - 1
    lower = min(max(lower, 0), int(snapshot_time_s.size) - 2)
    weight = (query - float(snapshot_time_s[lower])) / float(
        snapshot_time_s[lower + 1] - snapshot_time_s[lower]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        result = (1.0 - weight) * field.values[lower] + weight * field.values[lower + 1]
    if not bool(np.isfinite(result).all()):
        raise FieldLocationError(f"field {field.name!r} produced a non-finite sampled value")
    return np.asarray(result, dtype=np.float64)


def _finite_position(position_m: FloatArray) -> FloatArray:
    position = np.asarray(position_m, dtype=np.float64)
    if position.shape != (2,):
        raise ValueError("position_m must have shape (2,)")
    if not bool(np.isfinite(position).all()):
        raise ValueError("position_m must contain only finite values")
    return position


def _validate_cell_hint(layout: Layout, cell_hint: int) -> None:
    if isinstance(cell_hint, bool) or not isinstance(cell_hint, int):
        raise ValueError("cell_hint must be an integer")
    cell_count = _cell_count(layout)
    if cell_hint < -1 or cell_hint >= cell_count:
        raise ValueError("cell_hint is outside the layout cell range")


def _cell_count(layout: Layout) -> int:
    if isinstance(layout, RegularLayout):
        return int((layout.axis0_m.size - 1) * (layout.axis1_m.size - 1))
    return int(layout.connectivity.shape[0])


def _locate_regular(layout: RegularLayout, point: FloatArray) -> FieldLocation:
    axis0 = layout.axis0_m
    axis1 = layout.axis1_m
    index0 = _interval_index(axis0, float(point[0]))
    index1 = _interval_index(axis1, float(point[1]))
    tolerance0 = _axis_tolerance(axis0, float(point[0]))
    tolerance1 = _axis_tolerance(axis1, float(point[1]))
    geometrically_inside = (
        float(axis0[0]) - tolerance0 <= float(point[0]) <= float(axis0[-1]) + tolerance0
        and float(axis1[0]) - tolerance1 <= float(point[1]) <= float(axis1[-1]) + tolerance1
    )
    if geometrically_inside:
        candidates = [
            _regular_candidate(layout, point, candidate0, candidate1)
            for candidate0 in _regular_axis_candidates(axis0, float(point[0]), index0)
            for candidate1 in _regular_axis_candidates(axis1, float(point[1]), index1)
        ]
        selected = _select_supported_containing(layout.name, candidates)
        if selected is not None:
            return selected
    reason: OutsideReason = "masked_cell" if geometrically_inside else "outside_layout"
    candidate = _nearest_regular_supported(layout, point)
    return _outside_location(layout.name, candidate, reason)


def _regular_candidate(
    layout: RegularLayout, point: FloatArray, index0: int, index1: int
) -> _Candidate:
    axis0 = layout.axis0_m
    axis1 = layout.axis1_m
    width0 = float(axis0[index0 + 1] - axis0[index0])
    width1 = float(axis1[index1 + 1] - axis1[index1])
    coordinate0 = (float(point[0]) - float(axis0[index0])) / width0
    coordinate1 = (float(point[1]) - float(axis1[index1])) / width1
    uncertainty = max(
        _axis_tolerance(axis0, float(point[0])) / width0,
        _axis_tolerance(axis1, float(point[1])) / width1,
    )
    if not math.isfinite(uncertainty) or uncertainty > _MAX_REFERENCE_UNCERTAINTY:
        raise FieldLocationError(
            f"regular cell ({index0}, {index1}) is unresolved at float64 coordinate precision"
        )
    bounded0 = min(max(coordinate0, 0.0), 1.0)
    bounded1 = min(max(coordinate1, 0.0), 1.0)
    axis1_cell_count = int(axis1.size) - 1
    axis1_size = int(axis1.size)
    return _Candidate(
        index0 * axis1_cell_count + index1,
        (
            index0 * axis1_size + index1,
            (index0 + 1) * axis1_size + index1,
            (index0 + 1) * axis1_size + index1 + 1,
            index0 * axis1_size + index1 + 1,
        ),
        _q1_weights(bounded0, bounded1, unit_interval=True),
        bool(layout.cell_support[index0, index1]),
    )


def _nearest_regular_supported(layout: RegularLayout, point: FloatArray) -> _Candidate:
    best: tuple[float, _Candidate] | None = None
    for index0, index1 in np.argwhere(layout.cell_support != 0):
        lower0 = float(layout.axis0_m[index0])
        upper0 = float(layout.axis0_m[index0 + 1])
        lower1 = float(layout.axis1_m[index1])
        upper1 = float(layout.axis1_m[index1 + 1])
        projected = np.asarray(
            [
                min(max(float(point[0]), lower0), upper0),
                min(max(float(point[1]), lower1), upper1),
            ],
            dtype=np.float64,
        )
        candidate = _regular_candidate(layout, projected, int(index0), int(index1))
        distance = _distance(point, projected)
        if _is_better_projection(distance, candidate.cell_id, best):
            best = (distance, candidate)
    if best is None:
        raise FieldLocationError(f"layout {layout.name!r} has no supported field cells")
    return best[1]


def _interval_index(axis: FloatArray, coordinate: float) -> int:
    index = int(np.searchsorted(axis, coordinate, side="right")) - 1
    return min(max(index, 0), int(axis.size) - 2)


def _regular_axis_candidates(
    axis: FloatArray, coordinate: float, interval_index: int
) -> tuple[int, ...]:
    candidates = {interval_index}
    tolerance = _axis_tolerance(axis, coordinate)
    for node_index in (interval_index, interval_index + 1):
        if 0 < node_index < int(axis.size) - 1:
            if abs(coordinate - float(axis[node_index])) <= tolerance:
                candidates.update((node_index - 1, node_index))
    return tuple(sorted(candidates))


def _axis_tolerance(axis: FloatArray, coordinate: float) -> float:
    spacing = max(
        float(np.max(np.abs(np.spacing(axis)))),
        abs(float(np.spacing(np.float64(coordinate)))),
        _FLOAT_EPS * float(axis[-1] - axis[0]),
    )
    return FIELD_LOCATION_ULPS * spacing


def _locate_p1(layout: P1TriLayout, point: FloatArray, cell_hint: int) -> FieldLocation:
    del cell_hint
    containing: list[_Candidate] = []
    nearest: tuple[float, _Candidate] | None = None
    geometrically_inside = False
    for cell_id in range(int(layout.connectivity.shape[0])):
        node_ids = layout.connectivity[cell_id]
        nodes = layout.nodes_m[node_ids]
        jacobian = np.column_stack((nodes[1] - nodes[0], nodes[2] - nodes[0]))
        _validate_cell_jacobian(nodes, jacobian, f"P1 cell {cell_id}")
        contained = _physical_polygon_contains(nodes, point)
        geometrically_inside = geometrically_inside or contained
        if contained:
            weights = _certified_p1_weights(nodes, point, jacobian, cell_id)
            candidate = _Candidate(
                cell_id,
                tuple(int(item) for item in node_ids),
                weights,
                bool(layout.cell_support[cell_id]),
            )
            if candidate.supported:
                containing.append(candidate)
        if bool(layout.cell_support[cell_id]):
            distance, weights = _closest_polygon_weights(nodes, point)
            candidate = _Candidate(cell_id, tuple(int(item) for item in node_ids), weights, True)
            if _is_better_projection(distance, cell_id, nearest):
                nearest = (distance, candidate)
    selected = _select_supported_containing(layout.name, containing)
    if selected is not None:
        return selected
    if nearest is None:
        raise FieldLocationError(f"layout {layout.name!r} has no supported field cells")
    reason: OutsideReason = "masked_cell" if geometrically_inside else "outside_layout"
    return _outside_location(layout.name, nearest[1], reason)


def _certified_p1_weights(
    nodes: FloatArray, point: FloatArray, jacobian: FloatArray, cell_id: int
) -> tuple[float, float, float]:
    local_point = _finite_difference(point, nodes[0], f"P1 cell {cell_id} query")
    try:
        local = np.linalg.solve(jacobian, local_point)
    except np.linalg.LinAlgError as error:
        raise FieldLocationError(f"P1 inverse mapping failed for cell {cell_id}") from error
    raw = np.asarray([1.0 - local[0] - local[1], local[0], local[1]], dtype=np.float64)
    residual = jacobian @ local - local_point
    tolerance = _certify_reference(jacobian, nodes, point, local, residual, f"P1 cell {cell_id}")
    if float(np.min(raw)) < -2.0 * tolerance or float(np.max(raw)) > 1.0 + 2.0 * tolerance:
        raise FieldLocationError(
            f"P1 physical membership and inverse mapping disagree for cell {cell_id}"
        )
    bounded = _bounded_partition(raw, f"P1 cell {cell_id}")
    return float(bounded[0]), float(bounded[1]), float(bounded[2])


def _locate_q1(layout: Q1QuadLayout, point: FloatArray, cell_hint: int) -> FieldLocation:
    del cell_hint
    containing: list[_Candidate] = []
    nearest: tuple[float, _Candidate] | None = None
    geometrically_inside = False
    for cell_id in range(int(layout.connectivity.shape[0])):
        node_ids = layout.connectivity[cell_id]
        nodes = layout.nodes_m[node_ids]
        _validate_q1_cell(nodes, cell_id)
        contained = _physical_polygon_contains(nodes, point)
        geometrically_inside = geometrically_inside or contained
        if contained:
            weights = _certified_q1_weights(nodes, point, cell_id)
            candidate = _Candidate(
                cell_id,
                tuple(int(item) for item in node_ids),
                weights,
                bool(layout.cell_support[cell_id]),
            )
            if candidate.supported:
                containing.append(candidate)
        if bool(layout.cell_support[cell_id]):
            distance, weights = _closest_polygon_weights(nodes, point)
            candidate = _Candidate(cell_id, tuple(int(item) for item in node_ids), weights, True)
            if _is_better_projection(distance, cell_id, nearest):
                nearest = (distance, candidate)
    selected = _select_supported_containing(layout.name, containing)
    if selected is not None:
        return selected
    if nearest is None:
        raise FieldLocationError(f"layout {layout.name!r} has no supported field cells")
    reason: OutsideReason = "masked_cell" if geometrically_inside else "outside_layout"
    return _outside_location(layout.name, nearest[1], reason)


def _certified_q1_weights(
    nodes: FloatArray, point: FloatArray, cell_id: int
) -> tuple[float, float, float, float]:
    reference = _invert_q1(nodes, point, cell_id)
    xi = float(reference[0])
    eta = float(reference[1])
    raw = np.asarray(_q1_weights(xi, eta, unit_interval=False), dtype=np.float64)
    local_nodes = nodes - nodes[0]
    local_point = _finite_difference(point, nodes[0], f"Q1 cell {cell_id} query")
    residual = raw @ local_nodes - local_point
    jacobian = _q1_jacobian(nodes, xi, eta)
    tolerance = _certify_reference(
        jacobian, nodes, point, reference, residual, f"Q1 cell {cell_id}"
    )
    if abs(xi) > 1.0 + tolerance or abs(eta) > 1.0 + tolerance:
        raise FieldLocationError(
            f"Q1 physical membership and inverse mapping disagree for cell {cell_id}"
        )
    bounded_xi = min(max(xi, -1.0), 1.0)
    bounded_eta = min(max(eta, -1.0), 1.0)
    return _q1_weights(bounded_xi, bounded_eta, unit_interval=False)


def _validate_q1_cell(nodes: FloatArray, cell_id: int) -> None:
    references = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, 0.0))
    for xi, eta in references:
        jacobian = _q1_jacobian(nodes, xi, eta)
        _validate_cell_jacobian(nodes, jacobian, f"Q1 inverse mapping failed for cell {cell_id}")


def _invert_q1(nodes: FloatArray, point: FloatArray, cell_id: int) -> FloatArray:
    reference = np.zeros(2, dtype=np.float64)
    residual_tolerance = _physical_tolerance(nodes, point)
    local_nodes = nodes - nodes[0]
    local_point = _finite_difference(point, nodes[0], f"Q1 cell {cell_id} query")
    for _ in range(_Q1_MAX_ITERATIONS):
        weights, derivative_xi, derivative_eta = _q1_basis(float(reference[0]), float(reference[1]))
        residual = weights @ local_nodes - local_point
        if not bool(np.isfinite(residual).all()):
            break
        if float(np.max(np.abs(residual))) <= residual_tolerance:
            return reference
        jacobian = np.column_stack((derivative_xi @ local_nodes, derivative_eta @ local_nodes))
        _jacobian_metrics(jacobian, f"Q1 inverse mapping failed for cell {cell_id}")
        reference -= np.linalg.solve(jacobian, residual)
        if not bool(np.isfinite(reference).all()):
            break
    raise FieldLocationError(f"Q1 inverse mapping failed to converge for cell {cell_id}")


def _q1_jacobian(nodes: FloatArray, xi: float, eta: float) -> FloatArray:
    _, derivative_xi, derivative_eta = _q1_basis(xi, eta)
    local_nodes = nodes - nodes[0]
    return np.column_stack((derivative_xi @ local_nodes, derivative_eta @ local_nodes))


def _q1_weights(
    first: float, second: float, *, unit_interval: bool
) -> tuple[float, float, float, float]:
    if unit_interval:
        return (
            (1.0 - first) * (1.0 - second),
            first * (1.0 - second),
            first * second,
            (1.0 - first) * second,
        )
    weights, _, _ = _q1_basis(first, second)
    return float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3])


def _q1_basis(xi: float, eta: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    weights = 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )
    derivative_xi = 0.25 * np.asarray(
        [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)], dtype=np.float64
    )
    derivative_eta = 0.25 * np.asarray(
        [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi], dtype=np.float64
    )
    return weights, derivative_xi, derivative_eta


def _physical_polygon_contains(nodes: FloatArray, point: FloatArray) -> bool:
    tolerance = _physical_tolerance(nodes, point)
    for start in range(int(nodes.shape[0])):
        end = (start + 1) % int(nodes.shape[0])
        edge = nodes[end] - nodes[start]
        length = math.hypot(float(edge[0]), float(edge[1]))
        if not math.isfinite(length) or length == 0.0:
            raise FieldLocationError("field cell has a non-finite or zero-length edge")
        offset = _finite_difference(point, nodes[start], "field membership")
        scale = max(abs(float(offset[0])), abs(float(offset[1])))
        if scale == 0.0:
            continue
        normal0 = -float(edge[1]) / length
        normal1 = float(edge[0]) / length
        signed_factor = math.fsum(
            (float(offset[0]) / scale * normal0, float(offset[1]) / scale * normal1)
        )
        if signed_factor < 0.0 and -signed_factor > tolerance / scale:
            return False
    return True


def _physical_tolerance(nodes: FloatArray, point: FloatArray) -> float:
    coordinate_spacing = max(
        float(np.max(np.abs(np.spacing(nodes)))),
        float(np.max(np.abs(np.spacing(point)))),
    )
    diameter = 0.0
    for first in range(int(nodes.shape[0])):
        for second in range(first + 1, int(nodes.shape[0])):
            delta = _finite_difference(nodes[first], nodes[second], "field cell diameter")
            diameter = max(diameter, math.hypot(float(delta[0]), float(delta[1])))
    tolerance = FIELD_LOCATION_ULPS * max(coordinate_spacing, _FLOAT_EPS * diameter)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise FieldLocationError("field cell has no resolvable float64 length scale")
    return tolerance


def _certify_reference(
    jacobian: FloatArray,
    nodes: FloatArray,
    point: FloatArray,
    reference: FloatArray,
    residual: FloatArray,
    label: str,
) -> float:
    inverse_norm, condition = _jacobian_metrics(jacobian, label)
    physical_tolerance = _physical_tolerance(nodes, point)
    reference_scale = max(1.0, float(np.max(np.abs(reference))))
    tolerance = physical_tolerance * inverse_norm + (
        FIELD_LOCATION_ULPS * _FLOAT_EPS * condition * reference_scale
    )
    if not math.isfinite(tolerance) or tolerance > _MAX_REFERENCE_UNCERTAINTY:
        raise FieldLocationError(f"{label} is unresolved at float64 coordinate precision")
    if not bool(np.isfinite(residual).all()):
        raise FieldLocationError(f"{label} reconstruction produced a non-finite residual")
    if float(np.max(np.abs(residual))) > physical_tolerance:
        raise FieldLocationError(f"{label} inverse mapping exceeds its backward-error budget")
    return tolerance


def _jacobian_metrics(jacobian: FloatArray, label: str) -> tuple[float, float]:
    if not bool(np.isfinite(jacobian).all()):
        raise FieldLocationError(f"{label} has a non-finite Jacobian")
    determinant = float(np.linalg.det(jacobian))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise FieldLocationError(f"{label} has a non-positive Jacobian")
    try:
        inverse = np.linalg.solve(jacobian, np.eye(2, dtype=np.float64))
    except np.linalg.LinAlgError as error:
        raise FieldLocationError(f"{label} has a singular Jacobian") from error
    inverse_norm = float(np.linalg.norm(inverse, ord=np.inf))
    condition = float(np.linalg.norm(jacobian, ord=np.inf)) * inverse_norm
    if not math.isfinite(condition) or condition > _MAX_JACOBIAN_CONDITION:
        raise FieldLocationError(f"{label} exceeds the field-location Jacobian conditioning limit")
    return inverse_norm, condition


def _validate_cell_jacobian(nodes: FloatArray, jacobian: FloatArray, label: str) -> None:
    inverse_norm, condition = _jacobian_metrics(jacobian, label)
    node_spacing = float(np.max(np.abs(np.spacing(nodes))))
    diameter = 0.0
    for first in range(int(nodes.shape[0])):
        for second in range(first + 1, int(nodes.shape[0])):
            delta = _finite_difference(nodes[first], nodes[second], f"{label} diameter")
            diameter = max(diameter, math.hypot(float(delta[0]), float(delta[1])))
    physical_uncertainty = FIELD_LOCATION_ULPS * max(node_spacing, _FLOAT_EPS * diameter)
    reference_uncertainty = physical_uncertainty * inverse_norm + (
        FIELD_LOCATION_ULPS * _FLOAT_EPS * condition
    )
    if not math.isfinite(reference_uncertainty) or (
        reference_uncertainty > _MAX_REFERENCE_UNCERTAINTY
    ):
        raise FieldLocationError(f"{label} is unresolved at float64 coordinate precision")


def _bounded_partition(weights: FloatArray, label: str) -> Weights:
    bounded = np.clip(weights, 0.0, 1.0)
    total = float(np.sum(bounded))
    if not math.isfinite(total) or total <= 0.0:
        raise FieldLocationError(f"{label} produced invalid interpolation weights")
    bounded /= total
    return tuple(float(item) for item in bounded)


def _closest_polygon_weights(nodes: FloatArray, point: FloatArray) -> tuple[float, Weights]:
    best_distance = math.inf
    best_weights: Weights | None = None
    node_count = int(nodes.shape[0])
    for start in range(node_count):
        end = (start + 1) % node_count
        parameter, projected = _closest_segment_parameter(nodes[start], nodes[end], point)
        distance = _distance(point, projected)
        if distance < best_distance:
            weights = [0.0] * node_count
            weights[start] = 1.0 - parameter
            weights[end] = parameter
            best_distance = distance
            best_weights = tuple(weights)
    if best_weights is None or not math.isfinite(best_distance):
        raise FieldLocationError("could not project onto a finite field-cell closure")
    return best_distance, best_weights


def _closest_segment_parameter(
    start: FloatArray, end: FloatArray, point: FloatArray
) -> tuple[float, FloatArray]:
    edge = _finite_difference(end, start, "field edge")
    length = math.hypot(float(edge[0]), float(edge[1]))
    if not math.isfinite(length) or length == 0.0:
        raise FieldLocationError("field cell has a non-finite or zero-length edge")
    offset = _finite_difference(point, start, "field projection")
    scale = max(abs(float(offset[0])), abs(float(offset[1])))
    if scale == 0.0:
        parameter = 0.0
    else:
        direction0 = float(edge[0]) / length
        direction1 = float(edge[1]) / length
        along_factor = math.fsum(
            (float(offset[0]) / scale * direction0, float(offset[1]) / scale * direction1)
        )
        if along_factor <= 0.0:
            parameter = 0.0
        elif scale > length / along_factor:
            parameter = 1.0
        else:
            parameter = scale * along_factor / length
    projected = start + parameter * edge
    if not bool(np.isfinite(projected).all()):
        raise FieldLocationError("field-cell projection produced non-finite coordinates")
    return parameter, projected


def _finite_difference(first: FloatArray, second: FloatArray, label: str) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = first - second
    if not bool(np.isfinite(difference).all()):
        raise FieldLocationError(f"{label} exceeds the finite float64 range")
    return difference


def _distance(first: FloatArray, second: FloatArray) -> float:
    difference = _finite_difference(first, second, "field projection distance")
    distance = math.hypot(float(difference[0]), float(difference[1]))
    if not math.isfinite(distance):
        raise FieldLocationError("field projection distance is non-finite")
    return distance


def _is_better_projection(
    distance: float, cell_id: int, best: tuple[float, _Candidate] | None
) -> bool:
    return best is None or distance < best[0] or (distance == best[0] and cell_id < best[1].cell_id)


def _select_supported_containing(
    layout_name: str, candidates: list[_Candidate]
) -> FieldLocation | None:
    supported = [candidate for candidate in candidates if candidate.supported]
    if not supported:
        return None
    owner = min(supported, key=lambda candidate: candidate.cell_id)
    return FieldLocation(layout_name, owner.cell_id, owner.node_ids, owner.weights, True, None)


def _outside_location(
    layout_name: str, candidate: _Candidate, reason: OutsideReason
) -> FieldLocation:
    return FieldLocation(
        layout_name,
        candidate.cell_id,
        candidate.node_ids,
        candidate.weights,
        False,
        reason,
    )
