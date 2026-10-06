"""Compiled field passes, resident particle indexing, and CPU memory planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numba import njit

CPU_RUNTIME_LAYOUT_REVISION = "resident_soa_serial_slab_v6"
MEMORY_PLAN_REVISION = "solver_owned_memory_plan_v14"

_MAX_SLAB_PARTICLES = 65_536
_MINIMUM_SAFETY_MARGIN_BYTES = 64 * 1024

FIELD_KERNEL_OK = 0
FIELD_KERNEL_NONFINITE_GEOMETRY = 1
FIELD_KERNEL_UNRESOLVED_CELL = 2
FIELD_KERNEL_NO_SUPPORTED_CELL = 3
FIELD_KERNEL_INVERSE_MAPPING = 4
FIELD_KERNEL_NONFINITE_SAMPLE = 5

_FIELD_LOCATION_ULPS = 64.0
_FIELD_FLOAT_EPS = np.finfo(np.float64).eps
_FIELD_MAX_JACOBIAN_CONDITION = 1.0 / np.sqrt(_FIELD_FLOAT_EPS)
_FIELD_MAX_REFERENCE_UNCERTAINTY = 8.0 * np.sqrt(_FIELD_FLOAT_EPS)
_FIELD_Q1_MAX_ITERATIONS = 20


@njit(cache=True, fastmath=False, parallel=False)
def _field_two_term_sum(first: float, second: float) -> float:
    """Return a compensated two-term sum without enabling reassociation."""

    total = first + second
    second_virtual = total - first
    error = (first - (total - second_virtual)) + (second - second_virtual)
    return total + error


@njit(cache=True, fastmath=False, parallel=False)
def _field_axis_spacing(axis: np.ndarray) -> float:
    spacing = 0.0
    for index in range(axis.size):
        spacing = max(spacing, abs(float(np.spacing(axis[index]))))
    return spacing


@njit(cache=True, fastmath=False, parallel=False)
def _field_axis_tolerance(axis_spacing: float, span: float, coordinate: float) -> float:
    spacing = max(
        axis_spacing,
        abs(float(np.spacing(np.float64(coordinate)))),
        _FIELD_FLOAT_EPS * span,
    )
    return _FIELD_LOCATION_ULPS * spacing


@njit(cache=True, fastmath=False, parallel=False)
def _field_interval_index(axis: np.ndarray, coordinate: float) -> int:
    lower = 0
    upper = axis.size
    while lower < upper:
        middle = (lower + upper) // 2
        if coordinate < axis[middle]:
            upper = middle
        else:
            lower = middle + 1
    return min(max(lower - 1, 0), axis.size - 2)


@njit(cache=True, fastmath=False, parallel=False)
def _field_regular_axis_candidate_range(
    axis: np.ndarray,
    coordinate: float,
    interval: int,
    axis_spacing: float,
    span: float,
) -> tuple[int, int]:
    first = interval
    last = interval
    tolerance = _field_axis_tolerance(axis_spacing, span, coordinate)
    for node_index in (interval, interval + 1):
        if 0 < node_index < axis.size - 1:
            if abs(coordinate - axis[node_index]) <= tolerance:
                first = min(first, node_index - 1)
                last = max(last, node_index)
    return first, last


@njit(cache=True, fastmath=False, parallel=False)
def _field_regular_candidate(
    axis0: np.ndarray,
    axis1: np.ndarray,
    point0: float,
    point1: float,
    index0: int,
    index1: int,
    spacing0: float,
    spacing1: float,
    span0: float,
    span1: float,
) -> tuple[int, int, float, float, float, float]:
    width0 = axis0[index0 + 1] - axis0[index0]
    width1 = axis1[index1 + 1] - axis1[index1]
    coordinate0 = (point0 - axis0[index0]) / width0
    coordinate1 = (point1 - axis1[index1]) / width1
    uncertainty = max(
        _field_axis_tolerance(spacing0, span0, point0) / width0,
        _field_axis_tolerance(spacing1, span1, point1) / width1,
    )
    if not np.isfinite(uncertainty) or uncertainty > _FIELD_MAX_REFERENCE_UNCERTAINTY:
        return FIELD_KERNEL_UNRESOLVED_CELL, -1, 0.0, 0.0, 0.0, 0.0
    bounded0 = min(max(coordinate0, 0.0), 1.0)
    bounded1 = min(max(coordinate1, 0.0), 1.0)
    cell_id = index0 * (axis1.size - 1) + index1
    return (
        FIELD_KERNEL_OK,
        cell_id,
        (1.0 - bounded0) * (1.0 - bounded1),
        bounded0 * (1.0 - bounded1),
        bounded0 * bounded1,
        (1.0 - bounded0) * bounded1,
    )


@njit(cache=True, fastmath=False, parallel=False)
def _field_regular_containing(
    axis0: np.ndarray,
    axis1: np.ndarray,
    cell_support: np.ndarray,
    point0: float,
    point1: float,
    spacing0: float,
    spacing1: float,
    span0: float,
    span1: float,
) -> tuple[int, int, float, float, float, float]:
    interval0 = _field_interval_index(axis0, point0)
    interval1 = _field_interval_index(axis1, point1)
    first0, last0 = _field_regular_axis_candidate_range(axis0, point0, interval0, spacing0, span0)
    first1, last1 = _field_regular_axis_candidate_range(axis1, point1, interval1, spacing1, span1)
    best_cell = -1
    best0 = 0.0
    best1 = 0.0
    best2 = 0.0
    best3 = 0.0
    for index0 in range(first0, last0 + 1):
        for index1 in range(first1, last1 + 1):
            status, cell_id, weight0, weight1, weight2, weight3 = _field_regular_candidate(
                axis0,
                axis1,
                point0,
                point1,
                index0,
                index1,
                spacing0,
                spacing1,
                span0,
                span1,
            )
            if status != FIELD_KERNEL_OK:
                return status, -1, 0.0, 0.0, 0.0, 0.0
            if cell_support[index0, index1] != 0 and (best_cell < 0 or cell_id < best_cell):
                best_cell = cell_id
                best0 = weight0
                best1 = weight1
                best2 = weight2
                best3 = weight3
    return FIELD_KERNEL_OK, best_cell, best0, best1, best2, best3


@njit(cache=True, fastmath=False, parallel=False)
def _field_regular_nearest(
    axis0: np.ndarray,
    axis1: np.ndarray,
    cell_support: np.ndarray,
    point0: float,
    point1: float,
    spacing0: float,
    spacing1: float,
    span0: float,
    span1: float,
) -> tuple[int, int, float, float, float, float]:
    best_distance = np.inf
    best_cell = -1
    best0 = 0.0
    best1 = 0.0
    best2 = 0.0
    best3 = 0.0
    for index0 in range(axis0.size - 1):
        for index1 in range(axis1.size - 1):
            if cell_support[index0, index1] == 0:
                continue
            projected0 = min(max(point0, axis0[index0]), axis0[index0 + 1])
            projected1 = min(max(point1, axis1[index1]), axis1[index1 + 1])
            difference0 = point0 - projected0
            difference1 = point1 - projected1
            if not np.isfinite(difference0) or not np.isfinite(difference1):
                return FIELD_KERNEL_NONFINITE_GEOMETRY, -1, 0.0, 0.0, 0.0, 0.0
            distance = np.hypot(difference0, difference1)
            if not np.isfinite(distance):
                return FIELD_KERNEL_NONFINITE_GEOMETRY, -1, 0.0, 0.0, 0.0, 0.0
            status, cell_id, weight0, weight1, weight2, weight3 = _field_regular_candidate(
                axis0,
                axis1,
                projected0,
                projected1,
                index0,
                index1,
                spacing0,
                spacing1,
                span0,
                span1,
            )
            if status != FIELD_KERNEL_OK:
                return status, -1, 0.0, 0.0, 0.0, 0.0
            if distance < best_distance or (distance == best_distance and cell_id < best_cell):
                best_distance = distance
                best_cell = cell_id
                best0 = weight0
                best1 = weight1
                best2 = weight2
                best3 = weight3
    if best_cell < 0:
        return FIELD_KERNEL_NO_SUPPORTED_CELL, -1, 0.0, 0.0, 0.0, 0.0
    return FIELD_KERNEL_OK, best_cell, best0, best1, best2, best3


@njit(cache=True, fastmath=False, parallel=False)
def _field_locate_regular_point(
    axis0: np.ndarray,
    axis1: np.ndarray,
    cell_support: np.ndarray,
    point0: float,
    point1: float,
    spacing0: float,
    spacing1: float,
    span0: float,
    span1: float,
) -> tuple[int, bool, int, float, float, float, float]:
    tolerance0 = _field_axis_tolerance(spacing0, span0, point0)
    tolerance1 = _field_axis_tolerance(spacing1, span1, point1)
    inside = (
        axis0[0] - tolerance0 <= point0 <= axis0[-1] + tolerance0
        and axis1[0] - tolerance1 <= point1 <= axis1[-1] + tolerance1
    )
    if inside:
        status, cell_id, weight0, weight1, weight2, weight3 = _field_regular_containing(
            axis0,
            axis1,
            cell_support,
            point0,
            point1,
            spacing0,
            spacing1,
            span0,
            span1,
        )
        if status != FIELD_KERNEL_OK or cell_id >= 0:
            return status, cell_id >= 0, cell_id, weight0, weight1, weight2, weight3
    status, cell_id, weight0, weight1, weight2, weight3 = _field_regular_nearest(
        axis0,
        axis1,
        cell_support,
        point0,
        point1,
        spacing0,
        spacing1,
        span0,
        span1,
    )
    return status, False, cell_id, weight0, weight1, weight2, weight3


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def locate_regular_field_batch(
    axis0: np.ndarray,
    axis1: np.ndarray,
    cell_support: np.ndarray,
    position_m: np.ndarray,
    support_inside: np.ndarray,
    cell_id: np.ndarray,
    weights: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Locate one regular-layout stage into disjoint row buffers."""

    spacing0 = _field_axis_spacing(axis0)
    spacing1 = _field_axis_spacing(axis1)
    span0 = axis0[-1] - axis0[0]
    span1 = axis1[-1] - axis1[0]
    for row in range(position_m.shape[0]):
        status, inside, owner, weight0, weight1, weight2, weight3 = _field_locate_regular_point(
            axis0,
            axis1,
            cell_support,
            position_m[row, 0],
            position_m[row, 1],
            spacing0,
            spacing1,
            span0,
            span1,
        )
        row_status[row] = status
        support_inside[row] = inside
        cell_id[row] = owner
        weights[row, 0] = weight0
        weights[row, 1] = weight1
        weights[row, 2] = weight2
        weights[row, 3] = weight3


@njit(cache=True, fastmath=False, parallel=False)
def _field_cell_resolution_scale(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    node_count: int,
) -> tuple[int, float, float]:
    node_spacing = 0.0
    diameter = 0.0
    for first in range(node_count):
        first_id = connectivity[cell_id, first]
        node_spacing = max(
            node_spacing,
            abs(float(np.spacing(nodes_m[first_id, 0]))),
            abs(float(np.spacing(nodes_m[first_id, 1]))),
        )
        for second in range(first + 1, node_count):
            second_id = connectivity[cell_id, second]
            difference0 = nodes_m[first_id, 0] - nodes_m[second_id, 0]
            difference1 = nodes_m[first_id, 1] - nodes_m[second_id, 1]
            if not np.isfinite(difference0) or not np.isfinite(difference1):
                return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0
            distance = np.hypot(difference0, difference1)
            if not np.isfinite(distance):
                return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0
            diameter = max(diameter, distance)
    return FIELD_KERNEL_OK, node_spacing, diameter


@njit(cache=True, fastmath=False, parallel=False)
def _field_physical_tolerance(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    node_count: int,
    point0: float,
    point1: float,
) -> tuple[int, float]:
    status, node_spacing, diameter = _field_cell_resolution_scale(
        nodes_m, connectivity, cell_id, node_count
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0
    coordinate_spacing = max(
        node_spacing,
        abs(float(np.spacing(np.float64(point0)))),
        abs(float(np.spacing(np.float64(point1)))),
    )
    tolerance = _FIELD_LOCATION_ULPS * max(coordinate_spacing, _FIELD_FLOAT_EPS * diameter)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        return FIELD_KERNEL_UNRESOLVED_CELL, 0.0
    return FIELD_KERNEL_OK, tolerance


@njit(cache=True, fastmath=False, parallel=False)
def _field_polygon_contains(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    node_count: int,
    point0: float,
    point1: float,
) -> tuple[int, bool, bool]:
    status, tolerance = _field_physical_tolerance(
        nodes_m, connectivity, cell_id, node_count, point0, point1
    )
    if status != FIELD_KERNEL_OK:
        return status, False, False
    strict_interior = True
    for start in range(node_count):
        end = (start + 1) % node_count
        start_id = connectivity[cell_id, start]
        end_id = connectivity[cell_id, end]
        edge0 = nodes_m[end_id, 0] - nodes_m[start_id, 0]
        edge1 = nodes_m[end_id, 1] - nodes_m[start_id, 1]
        length = np.hypot(edge0, edge1)
        if not np.isfinite(length) or length == 0.0:
            return FIELD_KERNEL_NONFINITE_GEOMETRY, False, False
        offset0 = point0 - nodes_m[start_id, 0]
        offset1 = point1 - nodes_m[start_id, 1]
        if not np.isfinite(offset0) or not np.isfinite(offset1):
            return FIELD_KERNEL_NONFINITE_GEOMETRY, False, False
        scale = max(abs(offset0), abs(offset1))
        if scale == 0.0:
            strict_interior = False
            continue
        signed_factor = _field_two_term_sum(
            offset0 / scale * (-edge1 / length),
            offset1 / scale * (edge0 / length),
        )
        if signed_factor < 0.0 and -signed_factor > tolerance / scale:
            return FIELD_KERNEL_OK, False, False
        if signed_factor <= tolerance / scale:
            strict_interior = False
    return FIELD_KERNEL_OK, True, strict_interior


@njit(cache=True, fastmath=False, parallel=False)
def _field_jacobian_metrics(
    jacobian00: float,
    jacobian01: float,
    jacobian10: float,
    jacobian11: float,
) -> tuple[int, float, float]:
    if not (
        np.isfinite(jacobian00)
        and np.isfinite(jacobian01)
        and np.isfinite(jacobian10)
        and np.isfinite(jacobian11)
    ):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0
    determinant = jacobian00 * jacobian11 - jacobian01 * jacobian10
    if not np.isfinite(determinant) or determinant <= 0.0:
        return FIELD_KERNEL_UNRESOLVED_CELL, 0.0, 0.0
    inverse_norm = max(
        (abs(jacobian11) + abs(jacobian01)) / determinant,
        (abs(jacobian10) + abs(jacobian00)) / determinant,
    )
    jacobian_norm = max(
        abs(jacobian00) + abs(jacobian01),
        abs(jacobian10) + abs(jacobian11),
    )
    condition = jacobian_norm * inverse_norm
    if not np.isfinite(condition) or condition > _FIELD_MAX_JACOBIAN_CONDITION:
        return FIELD_KERNEL_UNRESOLVED_CELL, 0.0, 0.0
    return FIELD_KERNEL_OK, inverse_norm, condition


@njit(cache=True, fastmath=False, parallel=False)
def _field_validate_jacobian(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    node_count: int,
    jacobian00: float,
    jacobian01: float,
    jacobian10: float,
    jacobian11: float,
) -> int:
    status, inverse_norm, condition = _field_jacobian_metrics(
        jacobian00, jacobian01, jacobian10, jacobian11
    )
    if status != FIELD_KERNEL_OK:
        return status
    status, node_spacing, diameter = _field_cell_resolution_scale(
        nodes_m, connectivity, cell_id, node_count
    )
    if status != FIELD_KERNEL_OK:
        return status
    physical_uncertainty = _FIELD_LOCATION_ULPS * max(node_spacing, _FIELD_FLOAT_EPS * diameter)
    reference_uncertainty = physical_uncertainty * inverse_norm + (
        _FIELD_LOCATION_ULPS * _FIELD_FLOAT_EPS * condition
    )
    if (
        not np.isfinite(reference_uncertainty)
        or reference_uncertainty > _FIELD_MAX_REFERENCE_UNCERTAINTY
    ):
        return FIELD_KERNEL_UNRESOLVED_CELL
    return FIELD_KERNEL_OK


@njit(cache=True, fastmath=False, parallel=False)
def _field_certify_reference(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    node_count: int,
    point0: float,
    point1: float,
    reference0: float,
    reference1: float,
    residual0: float,
    residual1: float,
    jacobian00: float,
    jacobian01: float,
    jacobian10: float,
    jacobian11: float,
) -> tuple[int, float]:
    status, inverse_norm, condition = _field_jacobian_metrics(
        jacobian00, jacobian01, jacobian10, jacobian11
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0
    status, physical_tolerance = _field_physical_tolerance(
        nodes_m, connectivity, cell_id, node_count, point0, point1
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0
    reference_scale = max(1.0, abs(reference0), abs(reference1))
    tolerance = physical_tolerance * inverse_norm + (
        _FIELD_LOCATION_ULPS * _FIELD_FLOAT_EPS * condition * reference_scale
    )
    if not np.isfinite(tolerance) or tolerance > _FIELD_MAX_REFERENCE_UNCERTAINTY:
        return FIELD_KERNEL_UNRESOLVED_CELL, 0.0
    if not np.isfinite(residual0) or not np.isfinite(residual1):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0
    if max(abs(residual0), abs(residual1)) > physical_tolerance:
        return FIELD_KERNEL_UNRESOLVED_CELL, 0.0
    return FIELD_KERNEL_OK, tolerance


@njit(cache=True, fastmath=False, parallel=False)
def _field_p1_weights(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, float, float, float, float]:
    node0 = connectivity[cell_id, 0]
    node1 = connectivity[cell_id, 1]
    node2 = connectivity[cell_id, 2]
    jacobian00 = nodes_m[node1, 0] - nodes_m[node0, 0]
    jacobian01 = nodes_m[node2, 0] - nodes_m[node0, 0]
    jacobian10 = nodes_m[node1, 1] - nodes_m[node0, 1]
    jacobian11 = nodes_m[node2, 1] - nodes_m[node0, 1]
    status = _field_validate_jacobian(
        nodes_m,
        connectivity,
        cell_id,
        3,
        jacobian00,
        jacobian01,
        jacobian10,
        jacobian11,
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0, 0.0, 0.0, 0.0
    local0 = point0 - nodes_m[node0, 0]
    local1 = point1 - nodes_m[node0, 1]
    if not np.isfinite(local0) or not np.isfinite(local1):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0, 0.0, 0.0
    determinant = jacobian00 * jacobian11 - jacobian01 * jacobian10
    reference0 = (jacobian11 * local0 - jacobian01 * local1) / determinant
    reference1 = (-jacobian10 * local0 + jacobian00 * local1) / determinant
    weight0 = 1.0 - reference0 - reference1
    residual0 = jacobian00 * reference0 + jacobian01 * reference1 - local0
    residual1 = jacobian10 * reference0 + jacobian11 * reference1 - local1
    status, tolerance = _field_certify_reference(
        nodes_m,
        connectivity,
        cell_id,
        3,
        point0,
        point1,
        reference0,
        reference1,
        residual0,
        residual1,
        jacobian00,
        jacobian01,
        jacobian10,
        jacobian11,
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0, 0.0, 0.0, 0.0
    if (
        min(weight0, reference0, reference1) < -2.0 * tolerance
        or max(weight0, reference0, reference1) > 1.0 + 2.0 * tolerance
    ):
        return FIELD_KERNEL_INVERSE_MAPPING, 0.0, 0.0, 0.0, 0.0
    bounded0 = min(max(weight0, 0.0), 1.0)
    bounded1 = min(max(reference0, 0.0), 1.0)
    bounded2 = min(max(reference1, 0.0), 1.0)
    total = bounded0 + bounded1 + bounded2
    if not np.isfinite(total) or total <= 0.0:
        return FIELD_KERNEL_INVERSE_MAPPING, 0.0, 0.0, 0.0, 0.0
    return FIELD_KERNEL_OK, bounded0 / total, bounded1 / total, bounded2 / total, 0.0


@njit(cache=True, fastmath=False, parallel=False)
def _field_q1_jacobian(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    xi: float,
    eta: float,
) -> tuple[int, float, float, float, float]:
    derivative_xi1 = 0.25 * (1.0 - eta)
    derivative_xi2 = 0.25 * (1.0 + eta)
    derivative_xi3 = -0.25 * (1.0 + eta)
    derivative_eta1 = -0.25 * (1.0 + xi)
    derivative_eta2 = 0.25 * (1.0 + xi)
    derivative_eta3 = 0.25 * (1.0 - xi)
    node0 = connectivity[cell_id, 0]
    local_x1 = nodes_m[connectivity[cell_id, 1], 0] - nodes_m[node0, 0]
    local_x2 = nodes_m[connectivity[cell_id, 2], 0] - nodes_m[node0, 0]
    local_x3 = nodes_m[connectivity[cell_id, 3], 0] - nodes_m[node0, 0]
    local_y1 = nodes_m[connectivity[cell_id, 1], 1] - nodes_m[node0, 1]
    local_y2 = nodes_m[connectivity[cell_id, 2], 1] - nodes_m[node0, 1]
    local_y3 = nodes_m[connectivity[cell_id, 3], 1] - nodes_m[node0, 1]
    if not (
        np.isfinite(local_x1)
        and np.isfinite(local_x2)
        and np.isfinite(local_x3)
        and np.isfinite(local_y1)
        and np.isfinite(local_y2)
        and np.isfinite(local_y3)
    ):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0, 0.0, 0.0
    jacobian00 = derivative_xi1 * local_x1 + derivative_xi2 * local_x2 + derivative_xi3 * local_x3
    jacobian01 = (
        derivative_eta1 * local_x1 + derivative_eta2 * local_x2 + derivative_eta3 * local_x3
    )
    jacobian10 = derivative_xi1 * local_y1 + derivative_xi2 * local_y2 + derivative_xi3 * local_y3
    jacobian11 = (
        derivative_eta1 * local_y1 + derivative_eta2 * local_y2 + derivative_eta3 * local_y3
    )
    return FIELD_KERNEL_OK, jacobian00, jacobian01, jacobian10, jacobian11


@njit(cache=True, fastmath=False, parallel=False)
def _field_validate_q1_cell(nodes_m: np.ndarray, connectivity: np.ndarray, cell_id: int) -> int:
    references = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, 0.0))
    for xi, eta in references:
        status, jacobian00, jacobian01, jacobian10, jacobian11 = _field_q1_jacobian(
            nodes_m, connectivity, cell_id, xi, eta
        )
        if status != FIELD_KERNEL_OK:
            return status
        status = _field_validate_jacobian(
            nodes_m,
            connectivity,
            cell_id,
            4,
            jacobian00,
            jacobian01,
            jacobian10,
            jacobian11,
        )
        if status != FIELD_KERNEL_OK:
            return status
    return FIELD_KERNEL_OK


@njit(cache=True, fastmath=False, parallel=False)
def _field_q1_map_local(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    xi: float,
    eta: float,
) -> tuple[float, float]:
    weight1 = 0.25 * (1.0 + xi) * (1.0 - eta)
    weight2 = 0.25 * (1.0 + xi) * (1.0 + eta)
    weight3 = 0.25 * (1.0 - xi) * (1.0 + eta)
    node0 = connectivity[cell_id, 0]
    local_x1 = nodes_m[connectivity[cell_id, 1], 0] - nodes_m[node0, 0]
    local_x2 = nodes_m[connectivity[cell_id, 2], 0] - nodes_m[node0, 0]
    local_x3 = nodes_m[connectivity[cell_id, 3], 0] - nodes_m[node0, 0]
    local_y1 = nodes_m[connectivity[cell_id, 1], 1] - nodes_m[node0, 1]
    local_y2 = nodes_m[connectivity[cell_id, 2], 1] - nodes_m[node0, 1]
    local_y3 = nodes_m[connectivity[cell_id, 3], 1] - nodes_m[node0, 1]
    mapped0 = weight1 * local_x1 + weight2 * local_x2 + weight3 * local_x3
    mapped1 = weight1 * local_y1 + weight2 * local_y2 + weight3 * local_y3
    return mapped0, mapped1


@njit(cache=True, fastmath=False, parallel=False)
def _field_invert_q1(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, float, float, float, float]:
    node0 = connectivity[cell_id, 0]
    local0 = point0 - nodes_m[node0, 0]
    local1 = point1 - nodes_m[node0, 1]
    if not np.isfinite(local0) or not np.isfinite(local1):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0, 0.0, 0.0
    status, residual_tolerance = _field_physical_tolerance(
        nodes_m, connectivity, cell_id, 4, point0, point1
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0, 0.0, 0.0, 0.0
    xi = 0.0
    eta = 0.0
    converged = False
    for _ in range(_FIELD_Q1_MAX_ITERATIONS):
        mapped0, mapped1 = _field_q1_map_local(nodes_m, connectivity, cell_id, xi, eta)
        residual0 = mapped0 - local0
        residual1 = mapped1 - local1
        if not np.isfinite(residual0) or not np.isfinite(residual1):
            break
        if max(abs(residual0), abs(residual1)) <= residual_tolerance:
            converged = True
            break
        status, jacobian00, jacobian01, jacobian10, jacobian11 = _field_q1_jacobian(
            nodes_m, connectivity, cell_id, xi, eta
        )
        if status != FIELD_KERNEL_OK:
            return status, 0.0, 0.0, 0.0, 0.0
        status, _, _ = _field_jacobian_metrics(jacobian00, jacobian01, jacobian10, jacobian11)
        if status != FIELD_KERNEL_OK:
            return status, 0.0, 0.0, 0.0, 0.0
        determinant = jacobian00 * jacobian11 - jacobian01 * jacobian10
        delta0 = (jacobian11 * residual0 - jacobian01 * residual1) / determinant
        delta1 = (-jacobian10 * residual0 + jacobian00 * residual1) / determinant
        xi -= delta0
        eta -= delta1
        if not np.isfinite(xi) or not np.isfinite(eta):
            break
    if not converged:
        return FIELD_KERNEL_INVERSE_MAPPING, 0.0, 0.0, 0.0, 0.0
    return FIELD_KERNEL_OK, xi, eta, local0, local1


@njit(cache=True, fastmath=False, parallel=False)
def _field_q1_weights(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, float, float, float, float]:
    status, xi, eta, local0, local1 = _field_invert_q1(
        nodes_m, connectivity, cell_id, point0, point1
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0, 0.0, 0.0, 0.0
    mapped0, mapped1 = _field_q1_map_local(nodes_m, connectivity, cell_id, xi, eta)
    residual0 = mapped0 - local0
    residual1 = mapped1 - local1
    status, jacobian00, jacobian01, jacobian10, jacobian11 = _field_q1_jacobian(
        nodes_m, connectivity, cell_id, xi, eta
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0, 0.0, 0.0, 0.0
    status, tolerance = _field_certify_reference(
        nodes_m,
        connectivity,
        cell_id,
        4,
        point0,
        point1,
        xi,
        eta,
        residual0,
        residual1,
        jacobian00,
        jacobian01,
        jacobian10,
        jacobian11,
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0, 0.0, 0.0, 0.0
    if abs(xi) > 1.0 + tolerance or abs(eta) > 1.0 + tolerance:
        return FIELD_KERNEL_INVERSE_MAPPING, 0.0, 0.0, 0.0, 0.0
    bounded_xi = min(max(xi, -1.0), 1.0)
    bounded_eta = min(max(eta, -1.0), 1.0)
    return (
        FIELD_KERNEL_OK,
        0.25 * (1.0 - bounded_xi) * (1.0 - bounded_eta),
        0.25 * (1.0 + bounded_xi) * (1.0 - bounded_eta),
        0.25 * (1.0 + bounded_xi) * (1.0 + bounded_eta),
        0.25 * (1.0 - bounded_xi) * (1.0 + bounded_eta),
    )


@njit(cache=True, fastmath=False, parallel=False)
def _field_validate_p1_cell(nodes_m: np.ndarray, connectivity: np.ndarray, cell_id: int) -> int:
    node0 = connectivity[cell_id, 0]
    node1 = connectivity[cell_id, 1]
    node2 = connectivity[cell_id, 2]
    return _field_validate_jacobian(
        nodes_m,
        connectivity,
        cell_id,
        3,
        nodes_m[node1, 0] - nodes_m[node0, 0],
        nodes_m[node2, 0] - nodes_m[node0, 0],
        nodes_m[node1, 1] - nodes_m[node0, 1],
        nodes_m[node2, 1] - nodes_m[node0, 1],
    )


@njit(cache=True, fastmath=False, parallel=False)
def _field_inside_candidate(
    element_node_count: int,
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, bool, bool, float, float, float, float]:
    if element_node_count == 3:
        status = _field_validate_p1_cell(nodes_m, connectivity, cell_id)
    else:
        status = _field_validate_q1_cell(nodes_m, connectivity, cell_id)
    if status != FIELD_KERNEL_OK:
        return status, False, False, 0.0, 0.0, 0.0, 0.0
    status, contained, strict_interior = _field_polygon_contains(
        nodes_m, connectivity, cell_id, element_node_count, point0, point1
    )
    if status != FIELD_KERNEL_OK or not contained:
        return status, False, False, 0.0, 0.0, 0.0, 0.0
    if element_node_count == 3:
        status, weight0, weight1, weight2, weight3 = _field_p1_weights(
            nodes_m, connectivity, cell_id, point0, point1
        )
    else:
        status, weight0, weight1, weight2, weight3 = _field_q1_weights(
            nodes_m, connectivity, cell_id, point0, point1
        )
    return status, True, strict_interior, weight0, weight1, weight2, weight3


@njit(cache=True, fastmath=False, parallel=False)
def _field_closest_segment(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    start: int,
    end: int,
    point0: float,
    point1: float,
) -> tuple[int, float, float]:
    start_id = connectivity[cell_id, start]
    end_id = connectivity[cell_id, end]
    edge0 = nodes_m[end_id, 0] - nodes_m[start_id, 0]
    edge1 = nodes_m[end_id, 1] - nodes_m[start_id, 1]
    length = np.hypot(edge0, edge1)
    if not np.isfinite(length) or length == 0.0:
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0
    offset0 = point0 - nodes_m[start_id, 0]
    offset1 = point1 - nodes_m[start_id, 1]
    if not np.isfinite(offset0) or not np.isfinite(offset1):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0
    scale = max(abs(offset0), abs(offset1))
    if scale == 0.0:
        parameter = 0.0
    else:
        along_factor = _field_two_term_sum(
            offset0 / scale * (edge0 / length),
            offset1 / scale * (edge1 / length),
        )
        if along_factor <= 0.0:
            parameter = 0.0
        elif scale > length / along_factor:
            parameter = 1.0
        else:
            parameter = scale * along_factor / length
    projected0 = nodes_m[start_id, 0] + parameter * edge0
    projected1 = nodes_m[start_id, 1] + parameter * edge1
    if not np.isfinite(projected0) or not np.isfinite(projected1):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0
    difference0 = point0 - projected0
    difference1 = point1 - projected1
    if not np.isfinite(difference0) or not np.isfinite(difference1):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0
    distance = np.hypot(difference0, difference1)
    if not np.isfinite(distance):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0
    return FIELD_KERNEL_OK, distance, parameter


@njit(cache=True, fastmath=False, parallel=False)
def _field_edge_weights(
    start: int, end: int, parameter: float
) -> tuple[float, float, float, float]:
    result0 = 0.0
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    if start == 0:
        result0 = 1.0 - parameter
    elif start == 1:
        result1 = 1.0 - parameter
    elif start == 2:
        result2 = 1.0 - parameter
    else:
        result3 = 1.0 - parameter
    if end == 0:
        result0 = parameter
    elif end == 1:
        result1 = parameter
    elif end == 2:
        result2 = parameter
    else:
        result3 = parameter
    return result0, result1, result2, result3


@njit(cache=True, fastmath=False, parallel=False)
def _field_closest_cell_weights(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: int,
    node_count: int,
    point0: float,
    point1: float,
) -> tuple[int, float, float, float, float, float]:
    best_distance = np.inf
    best_start = 0
    best_end = 1
    best_parameter = 0.0
    for start in range(node_count):
        end = (start + 1) % node_count
        status, distance, parameter = _field_closest_segment(
            nodes_m, connectivity, cell_id, start, end, point0, point1
        )
        if status != FIELD_KERNEL_OK:
            return status, 0.0, 0.0, 0.0, 0.0, 0.0
        if distance < best_distance:
            best_distance = distance
            best_start = start
            best_end = end
            best_parameter = parameter
    weight0, weight1, weight2, weight3 = _field_edge_weights(best_start, best_end, best_parameter)
    return FIELD_KERNEL_OK, best_distance, weight0, weight1, weight2, weight3


@njit(cache=True, fastmath=False, parallel=False)
def _field_unstructured_full_search(
    element_node_count: int,
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_support: np.ndarray,
    point0: float,
    point1: float,
) -> tuple[int, bool, int, float, float, float, float]:
    best_distance = np.inf
    best_cell = -1
    best0 = 0.0
    best1 = 0.0
    best2 = 0.0
    best3 = 0.0
    for candidate in range(connectivity.shape[0]):
        status, contained, _, weight0, weight1, weight2, weight3 = _field_inside_candidate(
            element_node_count, nodes_m, connectivity, candidate, point0, point1
        )
        if status != FIELD_KERNEL_OK:
            return status, False, -1, 0.0, 0.0, 0.0, 0.0
        if contained and cell_support[candidate] != 0:
            return FIELD_KERNEL_OK, True, candidate, weight0, weight1, weight2, weight3
        if cell_support[candidate] == 0:
            continue
        status, distance, weight0, weight1, weight2, weight3 = _field_closest_cell_weights(
            nodes_m, connectivity, candidate, element_node_count, point0, point1
        )
        if status != FIELD_KERNEL_OK:
            return status, False, -1, 0.0, 0.0, 0.0, 0.0
        if distance < best_distance or (distance == best_distance and candidate < best_cell):
            best_distance = distance
            best_cell = candidate
            best0 = weight0
            best1 = weight1
            best2 = weight2
            best3 = weight3
    if best_cell < 0:
        return FIELD_KERNEL_NO_SUPPORTED_CELL, False, -1, 0.0, 0.0, 0.0, 0.0
    return FIELD_KERNEL_OK, False, best_cell, best0, best1, best2, best3


@njit(cache=True, fastmath=False, parallel=False)
def _field_unstructured_hint_location(
    element_node_count: int,
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_support: np.ndarray,
    point0: float,
    point1: float,
    hint: int,
) -> tuple[int, bool, int, float, float, float, float]:
    (
        status,
        contained,
        strict_interior,
        weight0,
        weight1,
        weight2,
        weight3,
    ) = _field_inside_candidate(element_node_count, nodes_m, connectivity, hint, point0, point1)
    if status != FIELD_KERNEL_OK:
        return status, False, -1, 0.0, 0.0, 0.0, 0.0
    if not contained or cell_support[hint] == 0:
        return _field_unstructured_full_search(
            element_node_count, nodes_m, connectivity, cell_support, point0, point1
        )
    if strict_interior:
        return FIELD_KERNEL_OK, True, hint, weight0, weight1, weight2, weight3
    for candidate in range(hint):
        (
            status,
            lower_contained,
            _,
            lower0,
            lower1,
            lower2,
            lower3,
        ) = _field_inside_candidate(
            element_node_count, nodes_m, connectivity, candidate, point0, point1
        )
        if status != FIELD_KERNEL_OK:
            return status, False, -1, 0.0, 0.0, 0.0, 0.0
        if lower_contained and cell_support[candidate] != 0:
            return FIELD_KERNEL_OK, True, candidate, lower0, lower1, lower2, lower3
    return FIELD_KERNEL_OK, True, hint, weight0, weight1, weight2, weight3


@njit(cache=True, fastmath=False, parallel=False)
def _field_index_excludes_point(
    lower0: float,
    lower1: float,
    upper0: float,
    upper1: float,
    point0: float,
    point1: float,
    point_padding: float,
) -> bool:
    """Return true only when one tolerance-expanded node cannot contain the point."""

    return (
        (point0 < lower0 and np.nextafter(lower0 - point0, -np.inf) > point_padding)
        or (point0 > upper0 and np.nextafter(point0 - upper0, -np.inf) > point_padding)
        or (point1 < lower1 and np.nextafter(lower1 - point1, -np.inf) > point_padding)
        or (point1 > upper1 and np.nextafter(point1 - upper1, -np.inf) > point_padding)
    )


@njit(cache=True, fastmath=False, parallel=False)
def _field_index_containing_location(
    element_node_count: int,
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_support: np.ndarray,
    index_cell_id: np.ndarray,
    index_lower_m: np.ndarray,
    index_upper_m: np.ndarray,
    index_begin: np.ndarray,
    index_end: np.ndarray,
    index_skip: np.ndarray,
    point0: float,
    point1: float,
) -> tuple[int, bool, int, float, float, float, float]:
    """Find the lowest-ID supported containing cell by stackless BVH traversal."""

    point_padding = np.nextafter(
        _FIELD_LOCATION_ULPS
        * max(
            abs(float(np.spacing(np.float64(point0)))),
            abs(float(np.spacing(np.float64(point1)))),
        ),
        np.inf,
    )
    best_cell = connectivity.shape[0]
    best0 = 0.0
    best1 = 0.0
    best2 = 0.0
    best3 = 0.0
    node = 0
    while node < index_skip.size:
        if _field_index_excludes_point(
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
            candidate = index_cell_id[offset]
            if candidate >= best_cell or cell_support[candidate] == 0:
                continue
            status, contained, _, weight0, weight1, weight2, weight3 = _field_inside_candidate(
                element_node_count,
                nodes_m,
                connectivity,
                candidate,
                point0,
                point1,
            )
            if status != FIELD_KERNEL_OK:
                return status, False, -1, 0.0, 0.0, 0.0, 0.0
            if contained:
                best_cell = candidate
                best0 = weight0
                best1 = weight1
                best2 = weight2
                best3 = weight3
        node = index_skip[node]
    if best_cell == connectivity.shape[0]:
        return FIELD_KERNEL_OK, False, -1, 0.0, 0.0, 0.0, 0.0
    return FIELD_KERNEL_OK, True, best_cell, best0, best1, best2, best3


@njit(cache=True, fastmath=False, parallel=False)
def _field_unstructured_index_location(
    element_node_count: int,
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_support: np.ndarray,
    index_cell_id: np.ndarray,
    index_lower_m: np.ndarray,
    index_upper_m: np.ndarray,
    index_begin: np.ndarray,
    index_end: np.ndarray,
    index_skip: np.ndarray,
    point0: float,
    point1: float,
    hint: int,
) -> tuple[int, bool, int, float, float, float, float]:
    """Use a strict hint, indexed containment, then the exact provisional fallback."""

    if hint >= 0:
        status, contained, strict, weight0, weight1, weight2, weight3 = _field_inside_candidate(
            element_node_count,
            nodes_m,
            connectivity,
            hint,
            point0,
            point1,
        )
        if status != FIELD_KERNEL_OK:
            return status, False, -1, 0.0, 0.0, 0.0, 0.0
        if contained and strict and cell_support[hint] != 0:
            return FIELD_KERNEL_OK, True, hint, weight0, weight1, weight2, weight3
    result = _field_index_containing_location(
        element_node_count,
        nodes_m,
        connectivity,
        cell_support,
        index_cell_id,
        index_lower_m,
        index_upper_m,
        index_begin,
        index_end,
        index_skip,
        point0,
        point1,
    )
    if result[0] != FIELD_KERNEL_OK or result[1]:
        return result
    return _field_unstructured_full_search(
        element_node_count,
        nodes_m,
        connectivity,
        cell_support,
        point0,
        point1,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def locate_unstructured_field_batch(
    element_node_count: int,
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_support: np.ndarray,
    index_cell_id: np.ndarray,
    index_lower_m: np.ndarray,
    index_upper_m: np.ndarray,
    index_begin: np.ndarray,
    index_end: np.ndarray,
    index_skip: np.ndarray,
    position_m: np.ndarray,
    cell_hint: np.ndarray,
    support_inside: np.ndarray,
    cell_id: np.ndarray,
    weights: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Locate one P1/Q1 stage into disjoint row buffers."""

    for row in range(position_m.shape[0]):
        hint = cell_hint[row]
        if index_skip.size:
            status, inside, owner, weight0, weight1, weight2, weight3 = (
                _field_unstructured_index_location(
                    element_node_count,
                    nodes_m,
                    connectivity,
                    cell_support,
                    index_cell_id,
                    index_lower_m,
                    index_upper_m,
                    index_begin,
                    index_end,
                    index_skip,
                    position_m[row, 0],
                    position_m[row, 1],
                    hint,
                )
            )
        elif hint >= 0:
            status, inside, owner, weight0, weight1, weight2, weight3 = (
                _field_unstructured_hint_location(
                    element_node_count,
                    nodes_m,
                    connectivity,
                    cell_support,
                    position_m[row, 0],
                    position_m[row, 1],
                    hint,
                )
            )
        else:
            status, inside, owner, weight0, weight1, weight2, weight3 = (
                _field_unstructured_full_search(
                    element_node_count,
                    nodes_m,
                    connectivity,
                    cell_support,
                    position_m[row, 0],
                    position_m[row, 1],
                )
            )
        row_status[row] = status
        support_inside[row] = inside
        cell_id[row] = owner
        weights[row, 0] = weight0
        weights[row, 1] = weight1
        weights[row, 2] = weight2
        weights[row, 3] = weight3


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def sample_regular_nodal_field(
    axis1_size: int,
    position_m: np.ndarray,
    cell_id: np.ndarray,
    weights: np.ndarray,
    canonical_values: np.ndarray,
    zero_axis_radial: bool,
    sampled_values: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Interpolate one canonical regular nodal field into a stage buffer."""

    axis1_cell_count = axis1_size - 1
    for row in range(position_m.shape[0]):
        index0 = cell_id[row] // axis1_cell_count
        index1 = cell_id[row] - index0 * axis1_cell_count
        node0 = index0 * axis1_size + index1
        node1 = (index0 + 1) * axis1_size + index1
        node2 = node1 + 1
        node3 = node0 + 1
        status = FIELD_KERNEL_OK
        for component in range(canonical_values.shape[1]):
            value = (
                weights[row, 0] * canonical_values[node0, component]
                + weights[row, 1] * canonical_values[node1, component]
                + weights[row, 2] * canonical_values[node2, component]
                + weights[row, 3] * canonical_values[node3, component]
            )
            if not np.isfinite(value):
                status = FIELD_KERNEL_NONFINITE_SAMPLE
            sampled_values[row, component] = value
        if zero_axis_radial and position_m[row, 0] == 0.0:
            sampled_values[row, 0] = 0.0
        row_status[row] = status


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def sample_unstructured_nodal_field(
    element_node_count: int,
    connectivity: np.ndarray,
    position_m: np.ndarray,
    cell_id: np.ndarray,
    weights: np.ndarray,
    canonical_values: np.ndarray,
    zero_axis_radial: bool,
    sampled_values: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Interpolate one canonical P1/Q1 nodal field into a stage buffer."""

    for row in range(position_m.shape[0]):
        owner = cell_id[row]
        status = FIELD_KERNEL_OK
        for component in range(canonical_values.shape[1]):
            value = 0.0
            for local_node in range(element_node_count):
                node_id = connectivity[owner, local_node]
                value += weights[row, local_node] * canonical_values[node_id, component]
            if not np.isfinite(value):
                status = FIELD_KERNEL_NONFINITE_SAMPLE
            sampled_values[row, component] = value
        if zero_axis_radial and position_m[row, 0] == 0.0:
            sampled_values[row, 0] = 0.0
        row_status[row] = status


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def sample_cell_field(
    position_m: np.ndarray,
    cell_id: np.ndarray,
    canonical_values: np.ndarray,
    zero_axis_radial: bool,
    sampled_values: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Copy one canonical cell field selected by the shared stage owner."""

    for row in range(position_m.shape[0]):
        owner = cell_id[row]
        status = FIELD_KERNEL_OK
        for component in range(canonical_values.shape[1]):
            value = canonical_values[owner, component]
            if not np.isfinite(value):
                status = FIELD_KERNEL_NONFINITE_SAMPLE
            sampled_values[row, component] = value
        if zero_axis_radial and position_m[row, 0] == 0.0:
            sampled_values[row, 0] = 0.0
        row_status[row] = status


@dataclass(slots=True)
class ActiveIndex:
    """Sorted resident-row indices for particles that still require work."""

    particle_index: np.ndarray
    keep_mask: np.ndarray
    count: int = 0

    @classmethod
    def allocate(cls, particle_count: int) -> ActiveIndex:
        if particle_count < 0:
            raise ValueError("particle_count must be nonnegative")
        return cls(
            np.empty(particle_count, dtype="<i8"),
            np.empty(particle_count, dtype=np.bool_),
        )

    @property
    def rows(self) -> np.ndarray:
        """Return the sorted active resident rows without allocating."""

        return self.particle_index[: self.count]

    def add(self, released: np.ndarray) -> None:
        """Add one release cohort while preserving resident-ID order."""

        values = np.asarray(released, dtype=np.int64)
        if values.ndim != 1 or self.count + values.size > self.particle_index.size:
            raise ValueError("released particle rows exceed the active-index capacity")
        if not values.size:
            return
        stop = self.count + int(values.size)
        self.particle_index[self.count : stop] = values
        # Resident rows are unique, so an in-place quicksort gives the same
        # canonical order without the linear workspace of a stable integer sort.
        self.particle_index[:stop].sort(kind="quicksort")
        self.count = stop

    def compact(self, active: np.ndarray, batch_size: int) -> None:
        """Remove terminal rows stably with only bounded batch temporaries."""

        rows = self.rows
        if active.shape != self.particle_index.shape:
            raise ValueError("active-state shape does not match the active index")
        if batch_size < 1:
            raise ValueError("active-index compaction batch size must be positive")
        if not rows.size:
            return
        write = 0
        for begin in range(0, self.count, batch_size):
            end = min(begin + batch_size, self.count)
            block = self.particle_index[begin:end]
            mask = self.keep_mask[: end - begin]
            np.take(active, block, out=mask)
            selected = block[mask]
            kept = int(selected.size)
            self.particle_index[write : write + kept] = selected
            write += kept
        self.count = write


@dataclass(frozen=True, slots=True)
class CpuMemoryPlan:
    """Predicted peak of solver-owned arrays, not a process-RSS hard limit."""

    limit_bytes: int
    particle_count: int
    slab_particles: int
    scratch_bytes_per_particle: int
    dense_path_bytes_per_particle: int
    stochastic_tree_work_bytes_per_particle: int
    event_work_bytes_per_particle: int
    certificate_work_bytes_per_particle: int
    release_work_bytes_per_particle: int
    replay_work_bytes: int
    event_candidate_capacity: int
    event_staging_capacity: int
    event_staging_bytes_per_row: int
    event_staging_fixed_bytes: int
    failure_staging_bytes_per_particle: int
    load_peak_bytes: int
    prepare_peak_bytes: int
    run_peak_bytes: int
    planned_bytes: int
    geometry_preparation_transient_bytes: int
    field_preparation_transient_bytes: int
    components: Mapping[str, int]

    def as_manifest(self) -> dict[str, object]:
        return {
            "revision": MEMORY_PLAN_REVISION,
            "runtime_layout_revision": CPU_RUNTIME_LAYOUT_REVISION,
            "semantics": "predicted peak of solver-owned arrays; not process RSS",
            "limit_bytes": self.limit_bytes,
            "planned_bytes": self.planned_bytes,
            "slab_particles": self.slab_particles,
            "scratch_bytes_per_particle": self.scratch_bytes_per_particle,
            "dense_path_bytes_per_particle": self.dense_path_bytes_per_particle,
            "stochastic_tree_work_bytes_per_particle": (
                self.stochastic_tree_work_bytes_per_particle
            ),
            "event_work_bytes_per_particle": self.event_work_bytes_per_particle,
            "certificate_work_bytes_per_particle": (self.certificate_work_bytes_per_particle),
            "release_work_bytes_per_particle": self.release_work_bytes_per_particle,
            "replay_work_bytes": self.replay_work_bytes,
            "event_candidate_capacity": self.event_candidate_capacity,
            "event_staging_capacity": self.event_staging_capacity,
            "event_staging_bytes_per_row": self.event_staging_bytes_per_row,
            "event_staging_fixed_bytes": self.event_staging_fixed_bytes,
            "failure_staging_bytes_per_particle": (self.failure_staging_bytes_per_particle),
            "geometry_preparation_transient_bytes": (self.geometry_preparation_transient_bytes),
            "field_preparation_transient_bytes": self.field_preparation_transient_bytes,
            "phase_peaks": {
                "load_case": self.load_peak_bytes,
                "prepare": self.prepare_peak_bytes,
                "run": self.run_peak_bytes,
            },
            "components": dict(self.components),
        }


def early_memory_requirement_bytes(
    *,
    canonical_data_bytes: int,
    particle_count: int,
    requires_stage_evaluation: bool,
    writer_reserve_bytes: int,
) -> int:
    """Return a phase-separated lower bound before source realization."""

    _require_nonnegative(canonical_data_bytes, particle_count, writer_reserve_bytes)
    schedule_bytes = particle_count * 120
    resident_state_bytes = particle_count * 101
    active_index_bytes = particle_count * 9
    # The early gate runs before model-specific runtime arrays exist, so it
    # must remain a guaranteed lower bound.  The final plan below uses the
    # realized 24/32/48-byte model footprint and makes the authoritative fit
    # decision.
    physics_runtime_bytes = particle_count * 16
    scratch_per_particle = _scratch_bytes_per_particle(requires_stage_evaluation)
    load_owned = canonical_data_bytes
    prepare_owned = canonical_data_bytes + 2 * schedule_bytes + physics_runtime_bytes
    run_owned = (
        canonical_data_bytes
        + schedule_bytes
        + resident_state_bytes
        + active_index_bytes
        + physics_runtime_bytes
        + writer_reserve_bytes
        + (scratch_per_particle if particle_count else 0)
    )
    return max(
        load_owned + _safety_margin(load_owned),
        prepare_owned + _safety_margin(prepare_owned),
        run_owned + _safety_margin(run_owned),
    )


def plan_cpu_memory(
    *,
    limit_bytes: int,
    particle_count: int,
    canonical_data_bytes: int,
    prepared_geometry_bytes: int,
    particle_schedule_bytes: int,
    physics_runtime_bytes: int,
    probe_index_bytes: int,
    output_buffer_bytes: int,
    replay_work_bytes: int = 0,
    writer_reserve_bytes: int,
    requires_stage_evaluation: bool,
    geometry_query_scratch_bytes: int = 0,
    dense_path_bytes_per_particle: int = 0,
    stochastic_tree_work_bytes_per_particle: int = 0,
    event_work_bytes_per_particle: int = 0,
    certificate_work_bytes_per_particle: int = 0,
    release_work_bytes_per_particle: int = 0,
    event_candidate_capacity: int = 0,
    event_staging_capacity: int = 0,
    event_staging_bytes_per_row: int = 0,
    event_staging_fixed_bytes: int = 0,
    failure_staging_bytes_per_particle: int = 0,
    field_runtime_bytes: int = 0,
    geometry_preparation_transient_bytes: int = 0,
    field_preparation_transient_bytes: int = 0,
) -> CpuMemoryPlan:
    """Choose one bounded slab under the configured memory ceiling."""

    _require_nonnegative(
        limit_bytes,
        particle_count,
        canonical_data_bytes,
        prepared_geometry_bytes,
        particle_schedule_bytes,
        field_runtime_bytes,
        geometry_preparation_transient_bytes,
        field_preparation_transient_bytes,
        physics_runtime_bytes,
        probe_index_bytes,
        output_buffer_bytes,
        replay_work_bytes,
        writer_reserve_bytes,
        geometry_query_scratch_bytes,
        dense_path_bytes_per_particle,
        stochastic_tree_work_bytes_per_particle,
        event_work_bytes_per_particle,
        certificate_work_bytes_per_particle,
        release_work_bytes_per_particle,
        event_candidate_capacity,
        event_staging_capacity,
        event_staging_bytes_per_row,
        event_staging_fixed_bytes,
        failure_staging_bytes_per_particle,
    )
    resident_state_bytes = particle_count * 101
    active_index_bytes = particle_count * 9
    final_state_buffer_bytes = particle_count * 9
    fixed_components = {
        "canonical_data": canonical_data_bytes,
        "prepared_geometry": prepared_geometry_bytes,
        "particle_schedule": particle_schedule_bytes,
        "field_runtime": field_runtime_bytes,
        "resident_state": resident_state_bytes,
        "active_index": active_index_bytes,
        "physics_runtime": physics_runtime_bytes,
        "probe_index": probe_index_bytes,
        "output_buffer": output_buffer_bytes + final_state_buffer_bytes,
        "replay_work": replay_work_bytes,
        "geometry_query_scratch": geometry_query_scratch_bytes,
        "writer_reserve": writer_reserve_bytes,
    }
    fixed_component_bytes = sum(fixed_components.values())
    fixed_run_bytes = fixed_component_bytes + event_staging_fixed_bytes
    scratch_per_particle = _scratch_bytes_per_particle(requires_stage_evaluation)
    slab_bytes_per_particle = (
        scratch_per_particle
        + dense_path_bytes_per_particle
        + stochastic_tree_work_bytes_per_particle
        + event_work_bytes_per_particle
        + certificate_work_bytes_per_particle
        + release_work_bytes_per_particle
        + event_staging_bytes_per_row
        + failure_staging_bytes_per_particle
    )
    maximum_slab = min(particle_count, _MAX_SLAB_PARTICLES)
    if event_staging_capacity:
        maximum_slab = min(maximum_slab, event_staging_capacity)
    slab_particles = _largest_fitting_slab(
        limit_bytes=limit_bytes,
        fixed_run_bytes=fixed_run_bytes,
        scratch_bytes_per_particle=slab_bytes_per_particle,
        maximum_slab=maximum_slab,
    )
    scratch_bytes = slab_particles * scratch_per_particle
    dense_path_bytes = slab_particles * dense_path_bytes_per_particle
    stochastic_tree_work_bytes = slab_particles * stochastic_tree_work_bytes_per_particle
    event_work_bytes = slab_particles * event_work_bytes_per_particle
    certificate_work_bytes = slab_particles * certificate_work_bytes_per_particle
    release_work_bytes = slab_particles * release_work_bytes_per_particle
    event_staging_bytes = event_staging_fixed_bytes + slab_particles * event_staging_bytes_per_row
    failure_staging_bytes = slab_particles * failure_staging_bytes_per_particle
    safety_margin_bytes = _safety_margin(
        fixed_component_bytes
        + scratch_bytes
        + dense_path_bytes
        + stochastic_tree_work_bytes
        + event_work_bytes
        + certificate_work_bytes
        + release_work_bytes
        + event_staging_bytes
        + failure_staging_bytes
    )
    components = {
        **fixed_components,
        "slab_proposal_scratch": scratch_bytes,
        "slab_dense_path": dense_path_bytes,
        "slab_stochastic_tree_work": stochastic_tree_work_bytes,
        "slab_event_work": event_work_bytes,
        "slab_certificate_work": certificate_work_bytes,
        "slab_release_work": release_work_bytes,
        "slab_event_staging": event_staging_bytes,
        "slab_failure_staging": failure_staging_bytes,
        "safety_margin": safety_margin_bytes,
    }
    run_peak_bytes = sum(components.values())
    load_owned = canonical_data_bytes
    load_peak_bytes = load_owned + _safety_margin(load_owned)
    prepare_owned = (
        canonical_data_bytes
        + prepared_geometry_bytes
        + 2 * particle_schedule_bytes
        + field_runtime_bytes
        + geometry_preparation_transient_bytes
        + field_preparation_transient_bytes
        + physics_runtime_bytes
    )
    prepare_peak_bytes = prepare_owned + _safety_margin(prepare_owned)
    planned_bytes = max(load_peak_bytes, prepare_peak_bytes, run_peak_bytes)
    return CpuMemoryPlan(
        limit_bytes=limit_bytes,
        particle_count=particle_count,
        slab_particles=slab_particles,
        scratch_bytes_per_particle=scratch_per_particle,
        dense_path_bytes_per_particle=dense_path_bytes_per_particle,
        stochastic_tree_work_bytes_per_particle=(stochastic_tree_work_bytes_per_particle),
        event_work_bytes_per_particle=event_work_bytes_per_particle,
        certificate_work_bytes_per_particle=certificate_work_bytes_per_particle,
        release_work_bytes_per_particle=release_work_bytes_per_particle,
        replay_work_bytes=replay_work_bytes,
        event_candidate_capacity=event_candidate_capacity,
        event_staging_capacity=event_staging_capacity,
        event_staging_bytes_per_row=event_staging_bytes_per_row,
        event_staging_fixed_bytes=event_staging_fixed_bytes,
        failure_staging_bytes_per_particle=failure_staging_bytes_per_particle,
        load_peak_bytes=load_peak_bytes,
        prepare_peak_bytes=prepare_peak_bytes,
        run_peak_bytes=run_peak_bytes,
        planned_bytes=planned_bytes,
        geometry_preparation_transient_bytes=geometry_preparation_transient_bytes,
        field_preparation_transient_bytes=field_preparation_transient_bytes,
        components=MappingProxyType(components),
    )


def _largest_fitting_slab(
    *,
    limit_bytes: int,
    fixed_run_bytes: int,
    scratch_bytes_per_particle: int,
    maximum_slab: int,
) -> int:
    """Solve the monotone slab-scratch memory inequality exactly."""

    lower = 0
    upper = maximum_slab
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        scratch_bytes = candidate * scratch_bytes_per_particle
        owned_bytes = fixed_run_bytes + scratch_bytes
        if owned_bytes + _safety_margin(owned_bytes) <= limit_bytes:
            lower = candidate
        else:
            upper = candidate - 1
    return lower


def _scratch_bytes_per_particle(requires_stage_evaluation: bool) -> int:
    # Includes proposal columns, RK stages/enclosures, row verdicts,
    # compiled-field temporaries, and the residual event wavefront's current
    # state, stack-top, departure, locator-result, and response columns.  The
    # depth-dependent stack and initial surface-release batch are separate
    # named components so configuration cannot escape the memory ceiling.
    return 2048 if requires_stage_evaluation else 384


def _safety_margin(owned_bytes: int) -> int:
    return max(_MINIMUM_SAFETY_MARGIN_BYTES, (owned_bytes + 7) // 8)


def _require_nonnegative(*values: int) -> None:
    if any(value < 0 for value in values):
        raise ValueError("memory-plan inputs must be nonnegative")
