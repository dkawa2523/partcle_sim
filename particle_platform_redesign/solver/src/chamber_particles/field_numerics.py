"""Prepared field location, interpolation and gradient numerics.

This owner contains the existing finite float64 policies and compiled kernels.
It does not know particle state, execution resources or persistent results.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


class FieldLocationError(RuntimeError):
    """A field location could not be computed from a numerically valid basis."""


FIELD_KERNEL_OK = 0
FIELD_KERNEL_NONFINITE_GEOMETRY = 1
FIELD_KERNEL_UNRESOLVED_CELL = 2
FIELD_KERNEL_NO_SUPPORTED_CELL = 3
FIELD_KERNEL_INVERSE_MAPPING = 4
FIELD_KERNEL_NONFINITE_SAMPLE = 5
FIELD_KERNEL_TIME_OUTSIDE = 6

FIELD_LOCATION_ULPS = 64.0
FIELD_FLOAT_EPS = np.finfo(np.float64).eps
FIELD_MAX_JACOBIAN_CONDITION = 1.0 / np.sqrt(FIELD_FLOAT_EPS)
FIELD_MAX_REFERENCE_UNCERTAINTY = 8.0 * np.sqrt(FIELD_FLOAT_EPS)
FIELD_Q1_MAX_ITERATIONS = 20


@njit(cache=True, fastmath=False, parallel=False)
def _field_two_term_sum(first: float, second: float) -> float:
    """Return a compensated two-term sum without enabling reassociation."""

    total = first + second
    second_virtual = total - first
    error = (first - (total - second_virtual)) + (second - second_virtual)
    return total + error


@njit(cache=True, fastmath=False, parallel=False)
def field_axis_spacing(axis: np.ndarray) -> float:
    spacing = 0.0
    for index in range(axis.size):
        spacing = max(spacing, abs(float(np.spacing(axis[index]))))
    return spacing


@njit(cache=True, fastmath=False, parallel=False)
def _field_axis_tolerance(axis_spacing: float, span: float, coordinate: float) -> float:
    spacing = max(
        axis_spacing,
        abs(float(np.spacing(np.float64(coordinate)))),
        FIELD_FLOAT_EPS * span,
    )
    return FIELD_LOCATION_ULPS * spacing


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
    tolerance: float,
) -> tuple[int, int]:
    first = interval
    last = interval
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
    certified_uncertainty: bool,
) -> tuple[int, int, float, float, float, float]:
    width0 = axis0[index0 + 1] - axis0[index0]
    width1 = axis1[index1 + 1] - axis1[index1]
    coordinate0 = (point0 - axis0[index0]) / width0
    coordinate1 = (point1 - axis1[index1]) / width1
    if not certified_uncertainty:
        uncertainty = max(
            _field_axis_tolerance(spacing0, span0, point0) / width0,
            _field_axis_tolerance(spacing1, span1, point1) / width1,
        )
        if not np.isfinite(uncertainty) or uncertainty > FIELD_MAX_REFERENCE_UNCERTAINTY:
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
    tolerance0: float,
    tolerance1: float,
    certified_projected_uncertainty: bool,
) -> tuple[int, int, float, float, float, float]:
    interval0 = _field_interval_index(axis0, point0)
    interval1 = _field_interval_index(axis1, point1)
    first0, last0 = _field_regular_axis_candidate_range(axis0, point0, interval0, tolerance0)
    first1, last1 = _field_regular_axis_candidate_range(axis1, point1, interval1, tolerance1)
    certified_uncertainty = (
        certified_projected_uncertainty
        and axis0[0] <= point0 <= axis0[-1]
        and axis1[0] <= point1 <= axis1[-1]
    )
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
                certified_uncertainty,
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
    certified_projected_uncertainty: bool,
) -> tuple[int, int, float, float, float, float]:
    best_distance = np.inf
    best_cell = -1
    best_projected0 = 0.0
    best_projected1 = 0.0
    best_index0 = 0
    best_index1 = 0
    for index0 in range(axis0.size - 1):
        projected0 = min(max(point0, axis0[index0]), axis0[index0 + 1])
        difference0 = point0 - projected0
        for index1 in range(axis1.size - 1):
            if cell_support[index0, index1] == 0:
                continue
            projected1 = min(max(point1, axis1[index1]), axis1[index1 + 1])
            difference1 = point1 - projected1
            if not np.isfinite(difference0) or not np.isfinite(difference1):
                return FIELD_KERNEL_NONFINITE_GEOMETRY, -1, 0.0, 0.0, 0.0, 0.0
            distance = np.hypot(difference0, difference1)
            if not np.isfinite(distance):
                return FIELD_KERNEL_NONFINITE_GEOMETRY, -1, 0.0, 0.0, 0.0, 0.0
            cell_id = index0 * (axis1.size - 1) + index1
            if not certified_projected_uncertainty:
                status, cell_id, _, _, _, _ = _field_regular_candidate(
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
                    False,
                )
                if status != FIELD_KERNEL_OK:
                    return status, -1, 0.0, 0.0, 0.0, 0.0
            if distance < best_distance or (distance == best_distance and cell_id < best_cell):
                best_distance = distance
                best_cell = cell_id
                best_projected0 = projected0
                best_projected1 = projected1
                best_index0 = index0
                best_index1 = index1
    if best_cell < 0:
        return FIELD_KERNEL_NO_SUPPORTED_CELL, -1, 0.0, 0.0, 0.0, 0.0
    return _field_regular_candidate(
        axis0,
        axis1,
        best_projected0,
        best_projected1,
        best_index0,
        best_index1,
        spacing0,
        spacing1,
        span0,
        span1,
        certified_projected_uncertainty,
    )


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
    certified_projected_uncertainty: bool,
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
            tolerance0,
            tolerance1,
            certified_projected_uncertainty,
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
        certified_projected_uncertainty,
    )
    return status, False, cell_id, weight0, weight1, weight2, weight3


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
    cell_metrics: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, float]:
    coordinate_spacing = max(
        cell_metrics[cell_id, 7],
        abs(float(np.spacing(np.float64(point0)))),
        abs(float(np.spacing(np.float64(point1)))),
    )
    tolerance = FIELD_LOCATION_ULPS * max(
        coordinate_spacing, FIELD_FLOAT_EPS * cell_metrics[cell_id, 8]
    )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        return FIELD_KERNEL_UNRESOLVED_CELL, 0.0
    return FIELD_KERNEL_OK, tolerance


@njit(cache=True, fastmath=False, parallel=False)
def _field_polygon_contains(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_metrics: np.ndarray,
    cell_id: int,
    node_count: int,
    point0: float,
    point1: float,
) -> tuple[int, bool, bool]:
    status, tolerance = _field_physical_tolerance(cell_metrics, cell_id, point0, point1)
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
    if not np.isfinite(condition) or condition > FIELD_MAX_JACOBIAN_CONDITION:
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
    physical_uncertainty = FIELD_LOCATION_ULPS * max(node_spacing, FIELD_FLOAT_EPS * diameter)
    reference_uncertainty = physical_uncertainty * inverse_norm + (
        FIELD_LOCATION_ULPS * FIELD_FLOAT_EPS * condition
    )
    if (
        not np.isfinite(reference_uncertainty)
        or reference_uncertainty > FIELD_MAX_REFERENCE_UNCERTAINTY
    ):
        return FIELD_KERNEL_UNRESOLVED_CELL
    return FIELD_KERNEL_OK


@njit(cache=True, fastmath=False, parallel=False)
def _field_certify_reference(
    cell_metrics: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
    reference0: float,
    reference1: float,
    residual0: float,
    residual1: float,
    inverse_norm: float,
    condition: float,
) -> tuple[int, float]:
    status, physical_tolerance = _field_physical_tolerance(cell_metrics, cell_id, point0, point1)
    if status != FIELD_KERNEL_OK:
        return status, 0.0
    reference_scale = max(1.0, abs(reference0), abs(reference1))
    tolerance = physical_tolerance * inverse_norm + (
        FIELD_LOCATION_ULPS * FIELD_FLOAT_EPS * condition * reference_scale
    )
    if not np.isfinite(tolerance) or tolerance > FIELD_MAX_REFERENCE_UNCERTAINTY:
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
    cell_metrics: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, float, float, float, float]:
    node0 = connectivity[cell_id, 0]
    jacobian00 = cell_metrics[cell_id, 0]
    jacobian01 = cell_metrics[cell_id, 1]
    jacobian10 = cell_metrics[cell_id, 2]
    jacobian11 = cell_metrics[cell_id, 3]
    local0 = point0 - nodes_m[node0, 0]
    local1 = point1 - nodes_m[node0, 1]
    if not np.isfinite(local0) or not np.isfinite(local1):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0, 0.0, 0.0
    determinant = cell_metrics[cell_id, 4]
    reference0 = (jacobian11 * local0 - jacobian01 * local1) / determinant
    reference1 = (-jacobian10 * local0 + jacobian00 * local1) / determinant
    weight0 = 1.0 - reference0 - reference1
    residual0 = jacobian00 * reference0 + jacobian01 * reference1 - local0
    residual1 = jacobian10 * reference0 + jacobian11 * reference1 - local1
    status, tolerance = _field_certify_reference(
        cell_metrics,
        cell_id,
        point0,
        point1,
        reference0,
        reference1,
        residual0,
        residual1,
        cell_metrics[cell_id, 5],
        cell_metrics[cell_id, 6],
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
    cell_metrics: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, float, float, float, float]:
    node0 = connectivity[cell_id, 0]
    local0 = point0 - nodes_m[node0, 0]
    local1 = point1 - nodes_m[node0, 1]
    if not np.isfinite(local0) or not np.isfinite(local1):
        return FIELD_KERNEL_NONFINITE_GEOMETRY, 0.0, 0.0, 0.0, 0.0
    status, residual_tolerance = _field_physical_tolerance(cell_metrics, cell_id, point0, point1)
    if status != FIELD_KERNEL_OK:
        return status, 0.0, 0.0, 0.0, 0.0
    xi = 0.0
    eta = 0.0
    converged = False
    for _ in range(FIELD_Q1_MAX_ITERATIONS):
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
    cell_metrics: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, float, float, float, float]:
    status, xi, eta, local0, local1 = _field_invert_q1(
        nodes_m, connectivity, cell_metrics, cell_id, point0, point1
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
    status, inverse_norm, condition = _field_jacobian_metrics(
        jacobian00, jacobian01, jacobian10, jacobian11
    )
    if status != FIELD_KERNEL_OK:
        return status, 0.0, 0.0, 0.0, 0.0
    status, tolerance = _field_certify_reference(
        cell_metrics,
        cell_id,
        point0,
        point1,
        xi,
        eta,
        residual0,
        residual1,
        inverse_norm,
        condition,
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


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def prepare_field_cell_metrics(
    element_node_count: int,
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_metrics: np.ndarray,
    cell_status: np.ndarray,
) -> None:
    """Certify static cells once, retaining the locator's exact arithmetic.

    Columns contain J00/J01/J10/J11, determinant, inverse infinity norm,
    condition, coordinate spacing, and diameter.  Q1 queries retain their
    point-dependent Jacobian; only their cell certificate and scale are reused.
    """

    cell_metrics.fill(0.0)
    for cell in range(connectivity.shape[0]):
        status = (
            _field_validate_p1_cell(nodes_m, connectivity, cell)
            if element_node_count == 3
            else _field_validate_q1_cell(nodes_m, connectivity, cell)
        )
        cell_status[cell] = status
        if status != FIELD_KERNEL_OK:
            continue
        _, spacing, diameter = _field_cell_resolution_scale(
            nodes_m, connectivity, cell, element_node_count
        )
        cell_metrics[cell, 7] = spacing
        cell_metrics[cell, 8] = diameter
        if element_node_count == 3:
            node0, node1, node2 = connectivity[cell]
            j00 = nodes_m[node1, 0] - nodes_m[node0, 0]
            j01 = nodes_m[node2, 0] - nodes_m[node0, 0]
            j10 = nodes_m[node1, 1] - nodes_m[node0, 1]
            j11 = nodes_m[node2, 1] - nodes_m[node0, 1]
            _, inverse_norm, condition = _field_jacobian_metrics(j00, j01, j10, j11)
            cell_metrics[cell, 0] = j00
            cell_metrics[cell, 1] = j01
            cell_metrics[cell, 2] = j10
            cell_metrics[cell, 3] = j11
            cell_metrics[cell, 4] = j00 * j11 - j01 * j10
            cell_metrics[cell, 5] = inverse_norm
            cell_metrics[cell, 6] = condition


@njit(cache=True, fastmath=False, parallel=False)
def _field_inside_candidate(
    element_node_count: int,
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_metrics: np.ndarray,
    cell_status: np.ndarray,
    cell_id: int,
    point0: float,
    point1: float,
) -> tuple[int, bool, bool, float, float, float, float]:
    status = cell_status[cell_id]
    if status != FIELD_KERNEL_OK:
        return status, False, False, 0.0, 0.0, 0.0, 0.0
    status, contained, strict_interior = _field_polygon_contains(
        nodes_m, connectivity, cell_metrics, cell_id, element_node_count, point0, point1
    )
    if status != FIELD_KERNEL_OK or not contained:
        return status, False, False, 0.0, 0.0, 0.0, 0.0
    if element_node_count == 3:
        status, weight0, weight1, weight2, weight3 = _field_p1_weights(
            nodes_m, connectivity, cell_metrics, cell_id, point0, point1
        )
    else:
        status, weight0, weight1, weight2, weight3 = _field_q1_weights(
            nodes_m, connectivity, cell_metrics, cell_id, point0, point1
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
    cell_metrics: np.ndarray,
    cell_status: np.ndarray,
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
            element_node_count,
            nodes_m,
            connectivity,
            cell_metrics,
            cell_status,
            candidate,
            point0,
            point1,
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
    cell_metrics: np.ndarray,
    cell_status: np.ndarray,
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
    ) = _field_inside_candidate(
        element_node_count, nodes_m, connectivity, cell_metrics, cell_status, hint, point0, point1
    )
    if status != FIELD_KERNEL_OK:
        return status, False, -1, 0.0, 0.0, 0.0, 0.0
    if not contained or cell_support[hint] == 0:
        return _field_unstructured_full_search(
            element_node_count,
            nodes_m,
            connectivity,
            cell_metrics,
            cell_status,
            cell_support,
            point0,
            point1,
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
            element_node_count,
            nodes_m,
            connectivity,
            cell_metrics,
            cell_status,
            candidate,
            point0,
            point1,
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
    cell_metrics: np.ndarray,
    cell_status: np.ndarray,
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
        FIELD_LOCATION_ULPS
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
                cell_metrics,
                cell_status,
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
    cell_metrics: np.ndarray,
    cell_status: np.ndarray,
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
            cell_metrics,
            cell_status,
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
        cell_metrics,
        cell_status,
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
        cell_metrics,
        cell_status,
        cell_support,
        point0,
        point1,
    )


@njit(cache=True, fastmath=False, parallel=False)
def _field_locate_prepared_point(
    layout_kind: int,
    axis0: np.ndarray,
    axis1: np.ndarray,
    axis_metrics: tuple[float, float, float, float, bool],
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_support: np.ndarray,
    cell_metrics: np.ndarray,
    cell_status: np.ndarray,
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
    if layout_kind == 1:
        return _field_locate_regular_point(
            axis0,
            axis1,
            cell_support.reshape((axis0.size - 1, axis1.size - 1)),
            point0,
            point1,
            *axis_metrics,
        )
    if index_skip.size:
        return _field_unstructured_index_location(
            layout_kind,
            nodes_m,
            connectivity,
            cell_metrics,
            cell_status,
            cell_support,
            index_cell_id,
            index_lower_m,
            index_upper_m,
            index_begin,
            index_end,
            index_skip,
            point0,
            point1,
            hint,
        )
    if hint >= 0:
        return _field_unstructured_hint_location(
            layout_kind,
            nodes_m,
            connectivity,
            cell_metrics,
            cell_status,
            cell_support,
            point0,
            point1,
            hint,
        )
    return _field_unstructured_full_search(
        layout_kind,
        nodes_m,
        connectivity,
        cell_metrics,
        cell_status,
        cell_support,
        point0,
        point1,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def locate_prepared_field_batch(
    layout_kind: int,
    axis0: np.ndarray,
    axis1: np.ndarray,
    axis_metrics: tuple[float, float, float, float, bool],
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_support: np.ndarray,
    cell_metrics: np.ndarray,
    cell_status: np.ndarray,
    index_cell_id: np.ndarray,
    index_lower_m: np.ndarray,
    index_upper_m: np.ndarray,
    index_begin: np.ndarray,
    index_end: np.ndarray,
    index_skip: np.ndarray,
    position_m: np.ndarray,
    cell_hint: np.ndarray | None,
    support_inside: np.ndarray,
    cell_id: np.ndarray,
    weights: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Locate each valid stage row without gathering or allocating temporary rows."""

    for row in range(position_m.shape[0]):
        row_status[row] = FIELD_KERNEL_OK
        support_inside[row] = layout_kind == 0
        cell_id[row] = -1
        weights[row, :] = 0.0
        point0 = position_m[row, 0]
        point1 = position_m[row, 1]
        if not np.isfinite(point0) or not np.isfinite(point1):
            row_status[row] = FIELD_KERNEL_NONFINITE_GEOMETRY
            continue
        if layout_kind == 0:
            continue
        hint = -1 if cell_hint is None else cell_hint[row]
        status, inside, owner, w0, w1, w2, w3 = _field_locate_prepared_point(
            layout_kind,
            axis0,
            axis1,
            axis_metrics,
            nodes_m,
            connectivity,
            cell_support,
            cell_metrics,
            cell_status,
            index_cell_id,
            index_lower_m,
            index_upper_m,
            index_begin,
            index_end,
            index_skip,
            point0,
            point1,
            hint,
        )
        row_status[row] = status
        support_inside[row] = inside
        cell_id[row] = owner
        weights[row, 0] = w0
        weights[row, 1] = w1
        weights[row, 2] = w2
        weights[row, 3] = w3


@njit(cache=True, fastmath=False, parallel=False)
def _field_spatial_value(
    layout_kind: int,
    axis1_size: int,
    connectivity: np.ndarray,
    association: int,
    owner: int,
    weights: np.ndarray,
    row: int,
    canonical_values: np.ndarray,
    snapshot: int,
    component: int,
) -> float:
    if association == 1:
        return canonical_values[snapshot, owner, component]
    if layout_kind == 1:
        index0 = owner // (axis1_size - 1)
        index1 = owner - index0 * (axis1_size - 1)
        node0 = index0 * axis1_size + index1
        node1 = (index0 + 1) * axis1_size + index1
        return (
            weights[row, 0] * canonical_values[snapshot, node0, component]
            + weights[row, 1] * canonical_values[snapshot, node1, component]
            + weights[row, 2] * canonical_values[snapshot, node1 + 1, component]
            + weights[row, 3] * canonical_values[snapshot, node0 + 1, component]
        )
    value = 0.0
    for local_node in range(layout_kind):
        value += (
            weights[row, local_node]
            * canonical_values[snapshot, connectivity[owner, local_node], component]
        )
    return value


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def sample_prepared_field_into(
    layout_kind: int,
    axis1_size: int,
    connectivity: np.ndarray,
    association: int,
    position_m: np.ndarray,
    time_s: np.ndarray,
    cell_id: np.ndarray,
    weights: np.ndarray,
    snapshot_time_s: np.ndarray,
    canonical_values: np.ndarray,
    zero_axis_radial: bool,
    sampled_values: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Sample one fixed primitive slot, retaining the first failed row status."""

    for row in range(position_m.shape[0]):
        sampled_values[row, :] = 0.0
        if row_status[row] != FIELD_KERNEL_OK:
            continue
        snapshot = 0
        time_weight = 0.0
        status = FIELD_KERNEL_OK
        if snapshot_time_s.size:
            status, snapshot, time_weight = _field_time_bracket(snapshot_time_s, time_s[row])
            if status != FIELD_KERNEL_OK:
                row_status[row] = status
                continue
        for component in range(canonical_values.shape[2]):
            value = _field_spatial_value(
                layout_kind,
                axis1_size,
                connectivity,
                association,
                cell_id[row],
                weights,
                row,
                canonical_values,
                snapshot,
                component,
            )
            if snapshot_time_s.size:
                upper = _field_spatial_value(
                    layout_kind,
                    axis1_size,
                    connectivity,
                    association,
                    cell_id[row],
                    weights,
                    row,
                    canonical_values,
                    snapshot + 1,
                    component,
                )
                value = _field_linear_time_value(value, upper, time_weight)
            if not np.isfinite(value):
                status = FIELD_KERNEL_NONFINITE_SAMPLE
            sampled_values[row, component] = value
        if zero_axis_radial and position_m[row, 0] == 0.0:
            sampled_values[row, 0] = 0.0
        if status != FIELD_KERNEL_OK:
            sampled_values[row, :] = 0.0
        row_status[row] = status


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def sample_p1_nodal_gradient(
    cell_metrics: np.ndarray,
    connectivity: np.ndarray,
    cell_id: np.ndarray,
    canonical_values: np.ndarray,
    sampled_gradient: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Differentiate P1 basis using the previously located cell."""

    for row in range(cell_id.size):
        if row_status[row] != FIELD_KERNEL_OK:
            sampled_gradient[row, :, :] = 0.0
            continue
        node0, node1, node2 = connectivity[cell_id[row]]
        dx1 = cell_metrics[cell_id[row], 0]
        dx2 = cell_metrics[cell_id[row], 1]
        dy1 = cell_metrics[cell_id[row], 2]
        dy2 = cell_metrics[cell_id[row], 3]
        determinant = cell_metrics[cell_id[row], 4]
        for component in range(canonical_values.shape[1]):
            difference1 = canonical_values[node1, component] - canonical_values[node0, component]
            difference2 = canonical_values[node2, component] - canonical_values[node0, component]
            gradient0 = (dy2 * difference1 - dy1 * difference2) / determinant
            gradient1 = (dx1 * difference2 - dx2 * difference1) / determinant
            sampled_gradient[row, component, 0] = gradient0
            sampled_gradient[row, component, 1] = gradient1
            if not np.isfinite(gradient0) or not np.isfinite(gradient1):
                row_status[row] = FIELD_KERNEL_NONFINITE_SAMPLE


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def sample_q1_nodal_gradient(
    nodes_m: np.ndarray,
    connectivity: np.ndarray,
    cell_id: np.ndarray,
    weights: np.ndarray,
    canonical_values: np.ndarray,
    sampled_gradient: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Use sampled Q1 coordinates and the locator's Jacobian, without locating."""

    for row in range(cell_id.size):
        if row_status[row] != FIELD_KERNEL_OK:
            sampled_gradient[row, :, :] = 0.0
            continue
        xi = 2.0 * (weights[row, 1] + weights[row, 2]) - 1.0
        eta = 2.0 * (weights[row, 2] + weights[row, 3]) - 1.0
        status, j00, j01, j10, j11 = _field_q1_jacobian(
            nodes_m, connectivity, cell_id[row], xi, eta
        )
        if status == FIELD_KERNEL_OK:
            status, _, _ = _field_jacobian_metrics(j00, j01, j10, j11)
        if status != FIELD_KERNEL_OK:
            row_status[row] = status
            sampled_gradient[row, :, :] = 0.0
            continue
        determinant = j00 * j11 - j01 * j10
        node0, node1, node2, node3 = connectivity[cell_id[row]]
        for component in range(canonical_values.shape[1]):
            difference1 = canonical_values[node1, component] - canonical_values[node0, component]
            difference2 = canonical_values[node2, component] - canonical_values[node0, component]
            difference3 = canonical_values[node3, component] - canonical_values[node0, component]
            du_xi = 0.25 * ((1.0 - eta) * difference1 + (1.0 + eta) * (difference2 - difference3))
            du_eta = 0.25 * ((1.0 - xi) * difference3 + (1.0 + xi) * (difference2 - difference1))
            gradient0 = (j11 * du_xi - j10 * du_eta) / determinant
            gradient1 = (j00 * du_eta - j01 * du_xi) / determinant
            sampled_gradient[row, component, 0] = gradient0
            sampled_gradient[row, component, 1] = gradient1
            if not np.isfinite(gradient0) or not np.isfinite(gradient1):
                row_status[row] = FIELD_KERNEL_NONFINITE_SAMPLE


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def sample_regular_nodal_gradient(
    axis0_m: np.ndarray,
    axis1_m: np.ndarray,
    cell_id: np.ndarray,
    weights: np.ndarray,
    canonical_values: np.ndarray,
    sampled_gradient: np.ndarray,
    row_status: np.ndarray,
) -> None:
    """Differentiate the bilinear basis using the sampled stage weights."""

    axis1_size = axis1_m.size
    for row in range(cell_id.size):
        if row_status[row] != FIELD_KERNEL_OK:
            sampled_gradient[row, :, :] = 0.0
            continue
        index0 = cell_id[row] // (axis1_size - 1)
        index1 = cell_id[row] - index0 * (axis1_size - 1)
        node0 = index0 * axis1_size + index1
        node1 = node0 + axis1_size
        node2 = node1 + 1
        node3 = node0 + 1
        fraction0 = weights[row, 1] + weights[row, 2]
        fraction1 = weights[row, 2] + weights[row, 3]
        width0 = axis0_m[index0 + 1] - axis0_m[index0]
        width1 = axis1_m[index1 + 1] - axis1_m[index1]
        for component in range(canonical_values.shape[1]):
            value0 = canonical_values[node0, component]
            value1 = canonical_values[node1, component]
            value2 = canonical_values[node2, component]
            value3 = canonical_values[node3, component]
            gradient0 = (
                (1.0 - fraction1) * (value1 - value0) + fraction1 * (value2 - value3)
            ) / width0
            gradient1 = (
                (1.0 - fraction0) * (value3 - value0) + fraction0 * (value2 - value1)
            ) / width1
            sampled_gradient[row, component, 0] = gradient0
            sampled_gradient[row, component, 1] = gradient1
            if not np.isfinite(gradient0) or not np.isfinite(gradient1):
                row_status[row] = FIELD_KERNEL_NONFINITE_SAMPLE


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _field_time_bracket(snapshot_time_s: np.ndarray, query_time_s: float) -> tuple[int, int, float]:
    """Return one inclusive linear bracket or a fail-closed row status."""

    if (
        not np.isfinite(query_time_s)
        or query_time_s < snapshot_time_s[0]
        or query_time_s > snapshot_time_s[-1]
    ):
        return FIELD_KERNEL_TIME_OUTSIDE, 0, 0.0
    lower = 0
    upper = snapshot_time_s.size - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if snapshot_time_s[middle] <= query_time_s:
            lower = middle
        else:
            upper = middle
    width = snapshot_time_s[lower + 1] - snapshot_time_s[lower]
    weight = (query_time_s - snapshot_time_s[lower]) / width
    if not np.isfinite(weight) or weight < 0.0 or weight > 1.0:
        return FIELD_KERNEL_TIME_OUTSIDE, 0, 0.0
    return FIELD_KERNEL_OK, lower, weight


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _field_linear_time_value(lower: float, upper: float, upper_weight: float) -> float:
    return _field_two_term_sum((1.0 - upper_weight) * lower, upper_weight * upper)


def validate_q1_cell(nodes: FloatArray, cell_id: int) -> None:
    references = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, 0.0))
    for xi, eta in references:
        jacobian = _q1_jacobian(nodes, xi, eta)
        validate_field_cell_jacobian(
            nodes, jacobian, f"Q1 inverse mapping failed for cell {cell_id}"
        )


def _q1_jacobian(nodes: FloatArray, xi: float, eta: float) -> FloatArray:
    _, derivative_xi, derivative_eta = _q1_basis(xi, eta)
    local_nodes = nodes - nodes[0]
    return np.column_stack((derivative_xi @ local_nodes, derivative_eta @ local_nodes))


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
    if not math.isfinite(condition) or condition > FIELD_MAX_JACOBIAN_CONDITION:
        raise FieldLocationError(f"{label} exceeds the field-location Jacobian conditioning limit")
    return inverse_norm, condition


def validate_field_cell_jacobian(nodes: FloatArray, jacobian: FloatArray, label: str) -> None:
    inverse_norm, condition = _jacobian_metrics(jacobian, label)
    node_spacing = float(np.max(np.abs(np.spacing(nodes))))
    diameter = 0.0
    for first in range(int(nodes.shape[0])):
        for second in range(first + 1, int(nodes.shape[0])):
            delta = _finite_difference(nodes[first], nodes[second], f"{label} diameter")
            diameter = max(diameter, math.hypot(float(delta[0]), float(delta[1])))
    physical_uncertainty = FIELD_LOCATION_ULPS * max(node_spacing, FIELD_FLOAT_EPS * diameter)
    reference_uncertainty = physical_uncertainty * inverse_norm + (
        FIELD_LOCATION_ULPS * FIELD_FLOAT_EPS * condition
    )
    if not math.isfinite(reference_uncertainty) or (
        reference_uncertainty > FIELD_MAX_REFERENCE_UNCERTAINTY
    ):
        raise FieldLocationError(f"{label} is unresolved at float64 coordinate precision")


def _finite_difference(first: FloatArray, second: FloatArray, label: str) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = first - second
    if not bool(np.isfinite(difference).all()):
        raise FieldLocationError(f"{label} exceeds the finite float64 range")
    return difference
