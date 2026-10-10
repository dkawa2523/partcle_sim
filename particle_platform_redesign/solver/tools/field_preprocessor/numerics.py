"""Bounded common-partition checks for polynomial space and linear time caches."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from fractions import Fraction
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from chamber_particles.case_format import (
    FieldData,
    Layout,
    P1TriLayout,
    Q1QuadLayout,
    RegularLayout,
)
from chamber_particles.fields import (
    FIELD_SPATIAL_GRADIENT_REVISION,
    FieldWorkspace,
    PreparedFieldSet,
)

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type ExactPoint = tuple[Fraction, Fraction]
type ExactPolygon = list[ExactPoint]

VALIDATION_REVISION: Final = "polynomial_common_partition_time_interval_weighted_norm_v3"
SUPPORT_CHECK_REVISION: Final = "exact_binary_rational_unique_coverage_v3"
SOURCE_TIME_DIAGNOSTIC_REVISION: Final = "interior_knot_omission_weighted_norm_v1"
_FLOAT_EPS: Final = np.finfo(np.float64).eps
_GAUSS, _GAUSS_WEIGHT = np.polynomial.legendre.leggauss(4)
_GAUSS = 0.5 * (_GAUSS + 1.0)
_GAUSS_WEIGHT = 0.5 * _GAUSS_WEIGHT


@dataclass(slots=True)
class ValidationResources:
    """The external validator's explicit array and geometric-work bounds."""

    memory_limit_bytes: int
    workspace_rows: int
    max_patch_work: int
    patch_work: int = dataclass_field(default=0, init=False)

    def consume_patch_work(self, amount: int = 1) -> None:
        self.patch_work += amount
        if self.patch_work > self.max_patch_work:
            raise ValueError("unresolved_validation: max_patch_work exhausted; cache not published")


@dataclass(slots=True)
class CommonPartition:
    """A bounded stream of intersections, never an all-cell-pair array."""

    source: Layout
    target: RegularLayout
    source_ids: Int64Array
    lower_m: FloatArray
    upper_m: FloatArray
    resources: ValidationResources

    def consume_work(self) -> None:
        self.resources.consume_patch_work()

    def patches(self) -> Iterator[tuple[int, int, ExactPolygon]]:
        for target_id in range(self.target.cell_support.size):
            rectangle = _regular_polygon(self.target, target_id)
            lower = np.asarray([float(rectangle[0][0]), float(rectangle[0][1])])
            upper = np.asarray([float(rectangle[2][0]), float(rectangle[2][1])])
            for candidate in _box_candidates(self, lower, upper):
                self.consume_work()
                source_id = int(self.source_ids[candidate])
                polygon = _intersection(rectangle, _source_polygon(self.source, source_id))
                if _area(polygon) > 0:
                    yield target_id, source_id, polygon


@dataclass(slots=True)
class _ErrorAccumulator:
    count: int = 0
    measure: float = 0.0
    squared_error: float = 0.0
    squared_reference: float = 0.0
    error_roundoff_squared: float = 0.0
    reference_roundoff_squared: float = 0.0
    sample_absolute_max: float = 0.0

    def update(
        self,
        reference: FloatArray,
        candidate: FloatArray,
        weight: FloatArray,
        reference_roundoff: FloatArray,
        candidate_roundoff: FloatArray,
    ) -> None:
        difference = candidate - reference
        row_weight = weight.reshape((-1,) + (1,) * (reference.ndim - 1))
        self.count += int(difference.size)
        self.measure = math.fsum((self.measure, float(np.sum(weight))))
        self.squared_error = math.fsum(
            (self.squared_error, _weighted_square(difference, row_weight))
        )
        self.squared_reference = math.fsum(
            (self.squared_reference, _weighted_square(reference, row_weight))
        )
        self.error_roundoff_squared = math.fsum(
            (
                self.error_roundoff_squared,
                _weighted_square(reference_roundoff + candidate_roundoff, row_weight),
            )
        )
        self.reference_roundoff_squared = math.fsum(
            (self.reference_roundoff_squared, _weighted_square(reference_roundoff, row_weight))
        )
        self.sample_absolute_max = max(self.sample_absolute_max, float(np.max(np.abs(difference))))
        if not all(
            math.isfinite(value)
            for value in (
                self.measure,
                self.squared_error,
                self.squared_reference,
                self.error_roundoff_squared,
                self.reference_roundoff_squared,
            )
        ):
            raise ValueError("unresolved_validation: nonfinite weighted norm")

    def report(self) -> dict[str, float | int | None]:
        error = math.sqrt(self.squared_error)
        reference = math.sqrt(self.squared_reference)
        accumulation_margin = 32.0 * _FLOAT_EPS * max(1, self.count)
        error_upper = (error + math.sqrt(self.error_roundoff_squared)) * (1.0 + accumulation_margin)
        reference_lower = max(
            0.0,
            (reference - math.sqrt(self.reference_roundoff_squared)) * (1.0 - accumulation_margin),
        )
        relative = error / reference if reference > 0.0 else (0.0 if error == 0.0 else None)
        relative_upper = (
            error_upper / reference_lower
            if reference_lower > 0.0
            else (0.0 if error_upper == 0.0 else None)
        )
        return {
            "scalar_sample_count": self.count,
            "measure": self.measure,
            "squared_error_integral": self.squared_error,
            "squared_reference_integral": self.squared_reference,
            "absolute_l2": error,
            "reference_l2": reference,
            "relative_l2": relative,
            "relative_l2_upper": relative_upper,
            "reference_l2_lower": reference_lower,
            "sample_absolute_max": self.sample_absolute_max,
        }


def _weighted_square(value: FloatArray, weight: FloatArray) -> float:
    result = float(np.sum(weight * value**2))
    if result == 0.0 and bool(((weight > 0.0) & (value != 0.0)).any()):
        raise ValueError("unresolved_validation: weighted squared norm underflow")
    return result


def validate_pair(field: FieldData, source: Layout, target: Layout) -> None:
    """Refuse unsupported capabilities before producing a candidate cache."""
    if not isinstance(target, RegularLayout):
        raise ValueError(
            "validation_unavailable_for_layout_pair: only polynomial sources to regular are certified"
        )
    if not bool(np.all(target.cell_support == 1)):
        raise ValueError("validation_unavailable_for_partial_regular_target: cache not published")
    if isinstance(source, Q1QuadLayout):
        for cell_id in np.flatnonzero(source.cell_support == 1):
            polygon = _source_polygon(source, int(cell_id))
            if any(
                polygon[0][axis] + polygon[2][axis] != polygon[1][axis] + polygon[3][axis]
                for axis in (0, 1)
            ):
                raise ValueError("validation_unavailable_for_warped_q1: cache not published")


def prepare_partition(
    source: Layout, target: RegularLayout, resources: ValidationResources
) -> CommonPartition:
    """Certify multiplicity one and exact coverage of binary-float coordinates.

    Rational clipping cannot erase positive slivers or cancel gaps with overlaps.
    Only each intersection's small polygon is retained.
    """
    source_ids = np.flatnonzero(source.cell_support == 1)
    if not source_ids.size:
        raise ValueError(f"source layout {source.name!r} has no supported cells")
    lower = np.empty((source_ids.size, 2), dtype="<f8")
    upper = np.empty_like(lower)
    for index, source_id in enumerate(source_ids):
        resources.consume_patch_work()
        nodes = _cell_nodes(source, int(source_id))
        lower[index] = np.min(nodes, axis=0)
        upper[index] = np.max(nodes, axis=0)
    partition = CommonPartition(source, target, source_ids, lower, upper, resources)
    if not isinstance(source, RegularLayout):
        _reject_overlaps(partition)
    _check_coverage(partition)
    return partition


def _box_candidates(partition: CommonPartition, lower: FloatArray, upper: FloatArray) -> Int64Array:
    partition.resources.consume_patch_work(int(partition.source_ids.size))
    return np.flatnonzero(
        np.all(partition.upper_m > lower, axis=1) & np.all(partition.lower_m < upper, axis=1)
    )


def _reject_overlaps(partition: CommonPartition) -> None:
    for index, source_id in enumerate(partition.source_ids):
        polygon = _source_polygon(partition.source, int(source_id))
        candidates = _box_candidates(partition, partition.lower_m[index], partition.upper_m[index])
        for other in candidates[candidates > index]:
            partition.consume_work()
            intersection = _intersection(
                polygon, _source_polygon(partition.source, int(partition.source_ids[other]))
            )
            if _area(intersection) > 0:
                raise ValueError(
                    f"source support {partition.source.name!r} has positive-area overlap"
                )


def _check_coverage(partition: CommonPartition) -> None:
    target_id = 0
    covered = Fraction(0)
    for current_id, _source_id, polygon in partition.patches():
        while target_id < current_id:
            _require_coverage(partition, target_id, covered)
            target_id += 1
            covered = Fraction(0)
        covered += _area(polygon)
    while target_id < partition.target.cell_support.size:
        _require_coverage(partition, target_id, covered)
        target_id += 1
        covered = Fraction(0)


def _require_coverage(partition: CommonPartition, cell_id: int, covered: Fraction) -> None:
    if covered != _area(_regular_polygon(partition.target, cell_id)):
        raise ValueError(
            f"target layout {partition.target.name!r} cell {cell_id} extends outside source support {partition.source.name!r}"
        )


def resample_field(
    field: FieldData,
    partition: CommonPartition,
    output_name: str,
    *,
    axis_accessible: bool,
    coordinate_system: str,
    limits: tuple[float, float, float],
) -> tuple[FieldData, dict[str, object]]:
    """Keep production node sampling and replace the old fixed-point gate."""
    nodes = _layout_nodes(partition.target)
    snapshot_count = 1 if field.time_s is None else int(field.time_s.size)
    values = np.empty((snapshot_count, nodes.shape[0], len(field.components)), dtype="<f8")
    source = PreparedFieldSet(
        partition.source, MappingProxyType({field.name: field}), axis_accessible=axis_accessible
    )
    workspace = source.allocate_workspace(partition.resources.workspace_rows)
    for snapshot in range(snapshot_count):
        for start in range(0, nodes.shape[0], workspace.capacity):
            stop = min(start + workspace.capacity, nodes.shape[0])
            time_s = None if field.time_s is None else np.full(stop - start, field.time_s[snapshot])
            batch = source.sample(nodes[start:stop], workspace=workspace, time_s=time_s)
            _require_support(batch.support_inside, field.name)
            values[snapshot, start:stop] = batch.values[field.name]
    cached = FieldData(
        output_name,
        partition.target.name,
        "node",
        field.components,
        field.stored_basis,
        values[0] if field.time_s is None else values,
        field.unit,
        field.time_s,
    )
    metrics = (
        _validation_metrics(field, cached, partition, axis_accessible, coordinate_system)
        if field.time_s is None
        else _time_validation_metrics(
            field, cached, partition, axis_accessible, coordinate_system, limits
        )
    )
    return cached, {
        "source_field": field.name,
        "output_field": output_name,
        "source_layout": partition.source.name,
        "target_layout": partition.target.name,
        "source_field_sha256": field_sha256(field),
        "snapshot_count": snapshot_count,
        "inactive_target_node_placeholders": 0,
        "validation_revision": VALIDATION_REVISION,
        "support_check_revision": SUPPORT_CHECK_REVISION,
        "gradient_revision": FIELD_SPATIAL_GRADIENT_REVISION,
        "certified_pair": ("static_" if field.time_s is None else "linear_time_")
        + (
            "affine_q1"
            if isinstance(partition.source, Q1QuadLayout)
            else _layout_kind(partition.source)
        )
        + "_to_full_regular",
        "metrics": metrics,
    }


@dataclass(frozen=True, slots=True)
class _ValidationSamples:
    prepared: PreparedFieldSet
    field: FieldData
    workspace: FieldWorkspace
    gradient: FloatArray

    def evaluate(self, points: FloatArray, *, derivative: bool) -> tuple[FloatArray, FloatArray]:
        batch = self.prepared.sample(points, workspace=self.workspace)
        _require_support(batch.support_inside, self.field.name)
        gradient = self.gradient[: points.shape[0]]
        if derivative:
            self.prepared.spatial_gradient(self.field.name, self.workspace, gradient)
        return batch.values[self.field.name], gradient


def _samples(
    field: FieldData, layout: Layout, rows: int, axis_accessible: bool
) -> _ValidationSamples:
    prepared = PreparedFieldSet(
        layout, MappingProxyType({field.name: field}), axis_accessible=axis_accessible
    )
    return _ValidationSamples(
        prepared,
        field,
        prepared.allocate_workspace(rows),
        np.empty((rows, len(field.components), 2), dtype="<f8"),
    )


def _validation_metrics(
    source_field: FieldData,
    cached_field: FieldData,
    partition: CommonPartition,
    axis_accessible: bool,
    coordinate_system: str,
) -> dict[str, object]:
    rows = partition.resources.workspace_rows
    source = _samples(source_field, partition.source, rows, axis_accessible)
    cached = _samples(cached_field, partition.target, rows, axis_accessible)
    value, gradient, boundary = _ErrorAccumulator(), _ErrorAccumulator(), _ErrorAccumulator()
    patch_count = 0
    for target_id, source_id, polygon in partition.patches():
        patch_count += 1
        reference_roundoff = _roundoff_bounds(partition.source, source_field, source_id)
        candidate_roundoff = _roundoff_bounds(partition.target, cached_field, target_id)
        for triangle in _triangles(polygon):
            points, weight = _triangle_quadrature(triangle, coordinate_system)
            reference, reference_gradient = source.evaluate(points, derivative=True)
            if not bool(np.all(source.workspace.cell_id[: points.shape[0]] == source_id)):
                raise ValueError(
                    "unresolved_validation: common patch is not resolved by the production locator"
                )
            candidate, candidate_gradient = cached.evaluate(points, derivative=True)
            if not bool(np.all(cached.workspace.cell_id[: points.shape[0]] == target_id)):
                raise ValueError(
                    "unresolved_validation: target common patch is not resolved by the production locator"
                )
            value.update(reference, candidate, weight, reference_roundoff[0], candidate_roundoff[0])
            gradient.update(
                reference_gradient,
                candidate_gradient,
                weight,
                reference_roundoff[1],
                candidate_roundoff[1],
            )
        for segment in _boundary_segments(polygon, partition.target, target_id):
            points, weight = _segment_quadrature(segment, coordinate_system)
            reference, _ = source.evaluate(points, derivative=False)
            candidate, _ = cached.evaluate(points, derivative=False)
            boundary.update(
                reference, candidate, weight, reference_roundoff[0], candidate_roundoff[0]
            )
    return {
        "measure": "dx_dy" if coordinate_system == "cartesian_xy" else "2pi_r_dr_dz",
        "boundary_scope": "target_support_boundary",
        "integration_rule": "positive_duffy_gauss4x4_degree5",
        "roundoff_policy": "nodal_basis_and_coordinate_allowance_v1",
        "common_patch_count": patch_count,
        "value": value.report(),
        "gradient": gradient.report(),
        "boundary_value": boundary.report(),
    }


def source_time_diagnostic(
    field: FieldData,
    partition: CommonPartition,
    *,
    axis_accessible: bool,
    coordinate_system: str,
) -> dict[str, object]:
    """Report sensitivity to omitting saved knots, not unsaved-time accuracy."""
    result: dict[str, object] = {
        "revision": SOURCE_TIME_DIAGNOSTIC_REVISION,
        "role": "report_only_not_cache_acceptance",
        "continuous_time_fidelity": "NOT_TESTED",
        "spatial_scope": "target_support",
        "boundary_scope": "target_support_boundary",
        "measure": "dx_dy" if coordinate_system == "cartesian_xy" else "2pi_r_dr_dz",
        "omitted_knots": [],
    }
    if field.time_s is None:
        return {**result, "status": "NOT_APPLICABLE", "reason": "static_source"}
    if field.time_s.size < 3:
        return {**result, "status": "INSUFFICIENT_SNAPSHOTS", "reason": "no_interior_knot"}
    omissions: list[dict[str, object]] = []
    result["omitted_knots"] = omissions
    for index in range(1, field.time_s.size - 1):
        left, middle, right = (float(value) for value in field.time_s[index - 1 : index + 2])
        widths = (middle - left, right - middle)
        scale = max(widths)
        fraction = (widths[0] / scale) / (widths[0] / scale + widths[1] / scale)
        snapshots = tuple(
            replace(field, values=field.values[k], time_s=None)
            for k in (index - 1, index, index + 1)
        )
        try:
            metrics = _omitted_knot_norms(
                (snapshots[0], snapshots[1], snapshots[2]),
                fraction,
                partition,
                axis_accessible,
                coordinate_system,
            )
        except ValueError as error:
            status = (
                "RESOURCE_LIMITED"
                if partition.resources.patch_work > partition.resources.max_patch_work
                else "UNRESOLVED"
            )
            return {**result, "status": status, "reason": str(error), "next_knot_index": index}
        omissions.append(
            {
                "index": index,
                "left_s": left,
                "time_s": middle,
                "right_s": right,
                "right_weight": fraction,
                "metrics": metrics,
            }
        )
    return {**result, "status": "EVALUATED"}


def _omitted_knot_norms(
    snapshots: tuple[FieldData, FieldData, FieldData],
    fraction: float,
    partition: CommonPartition,
    axis_accessible: bool,
    coordinate_system: str,
) -> dict[str, object]:
    samples = tuple(
        _samples(field, partition.source, partition.resources.workspace_rows, axis_accessible)
        for field in snapshots
    )
    norms = (_ErrorAccumulator(), _ErrorAccumulator(), _ErrorAccumulator())
    patches = 0
    for target_id, source_id, polygon in partition.patches():
        patches += 1
        allowances = tuple(
            _roundoff_bounds(partition.source, field, source_id) for field in snapshots
        )
        for triangle in _triangles(polygon):
            points, weight = _triangle_quadrature(triangle, coordinate_system)
            evaluated = tuple(sample.evaluate(points, derivative=True) for sample in samples)
            _require_patch_owner(samples, source_id, points.shape[0])
            for derivative, norm in enumerate(norms[:2]):
                _update_omission_norm(
                    norm,
                    tuple(values[derivative] for values in evaluated),
                    tuple(value[derivative] for value in allowances),
                    fraction,
                    weight,
                )
        for segment in _boundary_segments(polygon, partition.target, target_id):
            points, weight = _segment_quadrature(segment, coordinate_system)
            evaluated = tuple(sample.evaluate(points, derivative=False)[0] for sample in samples)
            _require_patch_owner(samples, source_id, points.shape[0])
            _update_omission_norm(
                norms[2], evaluated, tuple(value[0] for value in allowances), fraction, weight
            )
    return {
        "common_patch_count": patches,
        "value": norms[0].report(),
        "gradient": norms[1].report(),
        "boundary_value": norms[2].report(),
    }


def _update_omission_norm(
    norm: _ErrorAccumulator,
    values: tuple[FloatArray, ...],
    allowances: tuple[FloatArray, ...],
    fraction: float,
    weight: FloatArray,
) -> None:
    left, right = (1.0 - fraction) * values[0], fraction * values[2]
    candidate = left + right
    roundoff = (
        (1.0 - fraction) * allowances[0]
        + fraction * allowances[2]
        + 8.0 * _FLOAT_EPS * (np.abs(left) + np.abs(right))
    )
    norm.update(values[1], candidate, weight, allowances[1], roundoff)


def _roundoff_bounds(
    layout: Layout, field: FieldData, cell_id: int
) -> tuple[FloatArray, FloatArray]:
    if isinstance(layout, P1TriLayout):
        nodes = layout.nodes_m[layout.connectivity[cell_id]]
        values = field.values[layout.connectivity[cell_id]]
        edge1, edge2 = nodes[1] - nodes[0], nodes[2] - nodes[0]
        determinant = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        conditioning = 1.0 + (abs(edge1[0] * edge2[1]) + abs(edge1[1] * edge2[0])) / abs(
            determinant
        )
        inverse_scale = (np.sum(np.abs(edge1)) + np.sum(np.abs(edge2))) / abs(determinant)
        mixed_derivative_upper = np.zeros(values.shape[1], dtype="<f8")
    elif isinstance(layout, Q1QuadLayout):
        nodes = layout.nodes_m[layout.connectivity[cell_id]]
        values = field.values[layout.connectivity[cell_id]]
        edge1, edge2 = nodes[1] - nodes[0], nodes[3] - nodes[0]
        determinant = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        conditioning = 1.0 + (abs(edge1[0] * edge2[1]) + abs(edge1[1] * edge2[0])) / abs(
            determinant
        )
        inverse_scale = (np.sum(np.abs(edge1)) + np.sum(np.abs(edge2))) / abs(determinant)
        lower_difference = values[1] - values[0]
        upper_difference = values[2] - values[3]
        mixed = np.abs(upper_difference - lower_difference) + 32.0 * _FLOAT_EPS * (
            np.abs(lower_difference) + np.abs(upper_difference)
        )
        mixed_derivative_upper = 2.0 * mixed * inverse_scale**2 * conditioning
    else:
        polygon = _regular_polygon(layout, cell_id)
        nodes = _float_polygon(polygon)
        values = field.values[_regular_node_ids(layout, cell_id)]
        widths = nodes[2] - nodes[0]
        inverse_scale = 2.0 * float(np.sum(1.0 / widths))
        conditioning = 1.0
        # A bilinear gradient is affine in the opposite coordinate. Its change
        # at a rounded quadrature point is bounded by the constant mixed second
        # derivative times that point's coordinate displacement. The two x-edge
        # differences avoid an allowance proportional to a constant field offset.
        lower_difference = values[1] - values[0]
        upper_difference = values[2] - values[3]
        mixed_difference = upper_difference - lower_difference
        mixed_derivative_upper = (
            np.abs(mixed_difference)
            + 32.0 * _FLOAT_EPS * (np.abs(lower_difference) + np.abs(upper_difference))
        ) / (widths[0] * widths[1])
        mixed_derivative_upper *= 1.0 + 32.0 * _FLOAT_EPS
        mixed_derivative_upper = np.where(
            (lower_difference != 0.0) | (upper_difference != 0.0),
            np.nextafter(mixed_derivative_upper, np.inf),
            0.0,
        )
    amplitude = np.max(np.abs(values), axis=0)
    differences = np.max(np.abs(values - values[0]), axis=0)
    gradient_scale = differences * inverse_scale
    coordinate_roundoff = 128.0 * _FLOAT_EPS * float(np.max(np.abs(nodes)))
    value_roundoff = (
        128.0 * _FLOAT_EPS * conditioning * amplitude
        + coordinate_roundoff * conditioning * gradient_scale
    )
    gradient_roundoff = (
        256.0 * _FLOAT_EPS * conditioning * gradient_scale
        + coordinate_roundoff * mixed_derivative_upper
    )
    if bool(((amplitude > 0.0) & (value_roundoff == 0.0)).any()):
        raise ValueError("unresolved_validation: field roundoff allowance underflow")
    return value_roundoff, np.repeat(gradient_roundoff[:, None], 2, axis=1)


def _require_support(support: NDArray[np.bool_], name: str) -> None:
    if not bool(support.all()):
        raise ValueError(f"sampling field {name!r} left layout support")


def _triangles(polygon: ExactPolygon) -> Iterator[ExactPolygon]:
    for index in range(1, len(polygon) - 1):
        triangle = [polygon[0], polygon[index], polygon[index + 1]]
        if _area(triangle) > 0:
            yield triangle


def _triangle_quadrature(
    triangle: ExactPolygon, coordinate_system: str
) -> tuple[FloatArray, FloatArray]:
    nodes = _float_polygon(triangle)
    determinant = float(2 * _area(triangle))
    if determinant <= 0.0 or not math.isfinite(determinant):
        raise ValueError("unresolved_validation: common-patch area is not representable")
    first, second = nodes[1] - nodes[0], nodes[2] - nodes[0]
    if first[0] * second[1] - first[1] * second[0] <= 0.0:
        raise ValueError("unresolved_validation: common-patch float64 geometry collapsed")
    u, v = np.meshgrid(_GAUSS, _GAUSS, indexing="ij")
    points = nodes[0] + u.ravel()[:, None] * first + ((1.0 - u) * v).ravel()[:, None] * second
    weight = (determinant * (1.0 - u) * _GAUSS_WEIGHT[:, None] * _GAUSS_WEIGHT[None, :]).ravel()
    return np.ascontiguousarray(points), _physical_weight(weight, points, coordinate_system)


def _segment_quadrature(
    segment: ExactPolygon, coordinate_system: str
) -> tuple[FloatArray, FloatArray]:
    nodes = _float_polygon(segment)
    direction = nodes[1] - nodes[0]
    length = math.hypot(float(direction[0]), float(direction[1]))
    if length == 0.0 or not math.isfinite(length):
        raise ValueError("unresolved_validation: support-boundary segment collapsed")
    points = nodes[0] + _GAUSS[:, None] * direction
    return np.ascontiguousarray(points), _physical_weight(
        length * _GAUSS_WEIGHT, points, coordinate_system
    )


def _physical_weight(weight: FloatArray, points: FloatArray, coordinate_system: str) -> FloatArray:
    if coordinate_system == "axisymmetric_rz":
        weight = weight * (2.0 * math.pi * points[:, 0])
    if not bool(np.isfinite(weight).all()) or bool((weight < 0.0).any()):
        raise ValueError("unresolved_validation: invalid physical integration measure")
    if not bool((weight > 0.0).any()) and not (
        coordinate_system == "axisymmetric_rz" and bool((points[:, 0] == 0.0).all())
    ):
        raise ValueError("unresolved_validation: physical integration measure underflow")
    return weight


def _boundary_segments(
    polygon: ExactPolygon, target: RegularLayout, cell_id: int
) -> Iterator[ExactPolygon]:
    index0, index1 = divmod(cell_id, target.axis1_m.size - 1)
    rectangle = _regular_polygon(target, cell_id)
    exterior = (
        index1 == 0,
        index0 == target.axis0_m.size - 2,
        index1 == target.axis1_m.size - 2,
        index0 == 0,
    )
    for edge, is_exterior in enumerate(exterior):
        if is_exterior:
            axis = 1 if edge % 2 == 0 else 0
            coordinate = rectangle[edge][axis]
            for index, first in enumerate(polygon):
                second = polygon[(index + 1) % len(polygon)]
                if first != second and first[axis] == coordinate and second[axis] == coordinate:
                    yield [first, second]


def _regular_polygon(layout: RegularLayout, cell_id: int) -> ExactPolygon:
    index0, index1 = divmod(cell_id, layout.axis1_m.size - 1)
    low0, high0 = map(Fraction.from_float, map(float, layout.axis0_m[index0 : index0 + 2]))
    low1, high1 = map(Fraction.from_float, map(float, layout.axis1_m[index1 : index1 + 2]))
    return [(low0, low1), (high0, low1), (high0, high1), (low0, high1)]


def _source_polygon(layout: Layout, cell_id: int) -> ExactPolygon:
    if isinstance(layout, RegularLayout):
        return _regular_polygon(layout, cell_id)
    polygon = [
        (Fraction.from_float(float(point[0])), Fraction.from_float(float(point[1])))
        for point in layout.nodes_m[layout.connectivity[cell_id]]
    ]
    return polygon if _side(polygon[0], polygon[1], polygon[2]) > 0 else polygon[::-1]


def _cell_nodes(layout: Layout, cell_id: int) -> FloatArray:
    return (
        _float_polygon(_regular_polygon(layout, cell_id))
        if isinstance(layout, RegularLayout)
        else layout.nodes_m[layout.connectivity[cell_id]]
    )


def _intersection(subject: ExactPolygon, clip: ExactPolygon) -> ExactPolygon:
    polygon = subject
    for index, first in enumerate(clip):
        polygon = _clip_edge(polygon, first, clip[(index + 1) % len(clip)])
        if not polygon:
            break
    return polygon


def _clip_edge(polygon: ExactPolygon, first: ExactPoint, second: ExactPoint) -> ExactPolygon:
    if not polygon:
        return []
    output: ExactPolygon = []
    previous = polygon[-1]
    previous_side = _side(first, second, previous)
    for current in polygon:
        current_side = _side(first, second, current)
        if (current_side >= 0) != (previous_side >= 0):
            fraction = previous_side / (previous_side - current_side)
            output.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        if current_side >= 0:
            output.append(current)
        previous, previous_side = current, current_side
    return output


def _side(first: ExactPoint, second: ExactPoint, point: ExactPoint) -> Fraction:
    return (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (
        point[0] - first[0]
    )


def _area(polygon: ExactPolygon) -> Fraction:
    if len(polygon) < 3:
        return Fraction(0)
    return (
        abs(
            sum(
                (
                    _side(polygon[0], polygon[index], polygon[index + 1])
                    for index in range(1, len(polygon) - 1)
                ),
                Fraction(0),
            )
        )
        / 2
    )


def _float_polygon(polygon: ExactPolygon) -> FloatArray:
    points = np.asarray([[float(point[0]), float(point[1])] for point in polygon], dtype="<f8")
    if not bool(np.isfinite(points).all()):
        raise ValueError("unresolved_validation: intersection coordinates are not finite")
    return points


def _regular_node_ids(layout: RegularLayout, cell_id: int) -> Int64Array:
    index0, index1 = divmod(cell_id, layout.axis1_m.size - 1)
    node0 = index0 * layout.axis1_m.size + index1
    return np.asarray(
        [node0, node0 + layout.axis1_m.size, node0 + layout.axis1_m.size + 1, node0 + 1],
        dtype="<i8",
    )


def field_sha256(field: FieldData) -> str:
    metadata = json.dumps(
        {
            "association": field.association,
            "components": field.components,
            "layout": field.layout,
            "name": field.name,
            "stored_basis": field.stored_basis,
            "unit": field.unit,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(metadata)
    digest.update(memoryview(np.ascontiguousarray(field.values)).cast("B"))
    if field.time_s is not None:
        digest.update(memoryview(np.ascontiguousarray(field.time_s)).cast("B"))
    return f"sha256:{digest.hexdigest()}"


def layout_summary(layout: Layout) -> dict[str, object]:
    nodes = _layout_nodes(layout)
    return {
        "name": layout.name,
        "kind": _layout_kind(layout),
        "layout_sha256": _layout_sha256(layout),
        "node_count": int(nodes.shape[0]),
        "cell_count": int(layout.cell_support.size),
        "supported_cell_count": int(np.count_nonzero(layout.cell_support)),
        "coordinate_min_m": np.min(nodes, axis=0).tolist(),
        "coordinate_max_m": np.max(nodes, axis=0).tolist(),
    }


def _layout_sha256(layout: Layout) -> str:
    digest = hashlib.sha256(_layout_kind(layout).encode("ascii"))
    arrays = (
        (layout.axis0_m, layout.axis1_m, layout.cell_support)
        if isinstance(layout, RegularLayout)
        else (layout.nodes_m, layout.connectivity, layout.cell_support)
    )
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        descriptor = json.dumps(
            {"dtype": contiguous.dtype.str, "shape": contiguous.shape}, separators=(",", ":")
        ).encode("ascii")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(memoryview(contiguous).cast("B"))
    return f"sha256:{digest.hexdigest()}"


def _layout_kind(layout: Layout) -> str:
    return (
        "regular"
        if isinstance(layout, RegularLayout)
        else ("p1_tri" if isinstance(layout, P1TriLayout) else "q1_quad")
    )


def _layout_nodes(layout: Layout) -> FloatArray:
    if not isinstance(layout, RegularLayout):
        return layout.nodes_m
    axis0, axis1 = np.meshgrid(layout.axis0_m, layout.axis1_m, indexing="ij")
    return np.column_stack((axis0.ravel(), axis1.ravel()))


@dataclass(slots=True)
class _TimeNormAccumulator:
    """Adjacent-snapshot Gram moments; only two sampled snapshots are resident."""

    error: FloatArray = dataclass_field(default_factory=lambda: np.zeros(3))
    reference: FloatArray = dataclass_field(default_factory=lambda: np.zeros(3))
    error_absolute: FloatArray = dataclass_field(default_factory=lambda: np.zeros(3))
    reference_absolute: FloatArray = dataclass_field(default_factory=lambda: np.zeros(3))
    error_roundoff: FloatArray = dataclass_field(default_factory=lambda: np.zeros(2))
    reference_roundoff: FloatArray = dataclass_field(default_factory=lambda: np.zeros(2))
    count: int = 0
    measure: float = 0.0
    sample_absolute_max: float = 0.0

    def update(
        self,
        references: tuple[FloatArray, FloatArray],
        candidates: tuple[FloatArray, FloatArray],
        weight: FloatArray,
        source_roundoff: tuple[FloatArray, FloatArray],
        cache_roundoff: tuple[FloatArray, FloatArray],
    ) -> None:
        row_weight = weight.reshape((-1,) + (1,) * (references[0].ndim - 1))
        differences = (candidates[0] - references[0], candidates[1] - references[1])
        _add_gram(self.error, self.error_absolute, differences, row_weight)
        _add_gram(self.reference, self.reference_absolute, references, row_weight)
        for endpoint in (0, 1):
            self.error_roundoff[endpoint] = math.fsum(
                (
                    float(self.error_roundoff[endpoint]),
                    _weighted_square(
                        source_roundoff[endpoint] + cache_roundoff[endpoint], row_weight
                    ),
                )
            )
            self.reference_roundoff[endpoint] = math.fsum(
                (
                    float(self.reference_roundoff[endpoint]),
                    _weighted_square(source_roundoff[endpoint], row_weight),
                )
            )
        self.count += int(references[0].size)
        self.measure = math.fsum((self.measure, float(np.sum(weight))))
        self.sample_absolute_max = max(
            self.sample_absolute_max, *(float(np.max(np.abs(value))) for value in differences)
        )
        if not all(
            bool(np.isfinite(array).all())
            for array in (
                self.error,
                self.reference,
                self.error_absolute,
                self.reference_absolute,
                self.error_roundoff,
                self.reference_roundoff,
            )
        ):
            raise ValueError("unresolved_validation: nonfinite temporal Gram moment")

    def allowances(self) -> tuple[float, float, float, float]:
        factor = 128.0 * _FLOAT_EPS * max(1, self.count)
        if factor >= 0.1:
            raise ValueError("unresolved_validation: temporal accumulation allowance exhausted")
        absolute = (self.error_absolute, self.reference_absolute)
        margins = tuple(math.fsum(factor * float(value) for value in array) for array in absolute)
        if any(
            bool((array > 0.0).any()) and margin == 0.0
            for array, margin in zip(absolute, margins, strict=True)
        ):
            raise ValueError("unresolved_validation: temporal allowance underflow")
        return (
            margins[0],
            margins[1],
            math.sqrt(float(np.max(self.error_roundoff))),
            math.sqrt(float(np.max(self.reference_roundoff))),
        )


def _add_gram(
    gram: FloatArray,
    absolute: FloatArray,
    endpoints: tuple[FloatArray, FloatArray],
    weight: FloatArray,
) -> None:
    for index, (first, second) in enumerate(((0, 0), (0, 1), (1, 1))):
        product = weight * endpoints[first] * endpoints[second]
        if first == second:
            _weighted_square(endpoints[first], weight)
        gram[index] = math.fsum((float(gram[index]), float(np.sum(product))))
        absolute[index] = math.fsum((float(absolute[index]), float(np.sum(np.abs(product)))))


def _gram_cross(gram: FloatArray, first: float, second: float) -> float:
    return math.fsum(
        (
            (1.0 - first) * (1.0 - second) * float(gram[0]),
            ((1.0 - first) * second + first * (1.0 - second)) * float(gram[1]),
            first * second * float(gram[2]),
        )
    )


def _gram_range(
    gram: FloatArray, left: float, right: float, allowance: float
) -> tuple[float, float]:
    # Bernstein convex-hull bounds avoid trusting a rounded extremum location.
    first, last = _gram_cross(gram, left, left), _gram_cross(gram, right, right)
    middle = _gram_cross(gram, left, right)
    return max(0.0, min(first, middle, last) - allowance), max(
        0.0, max(first, middle, last) + allowance
    )


def _ratio_bounds(
    norm: _TimeNormAccumulator, left: float, right: float
) -> tuple[float, float | None]:
    error_allowance, reference_allowance, error_roundoff, reference_roundoff = norm.allowances()
    error_lower, error_upper = _gram_range(norm.error, left, right, error_allowance)
    reference_lower, reference_upper = _gram_range(norm.reference, left, right, reference_allowance)
    lower_numerator = max(0.0, math.sqrt(error_lower) - error_roundoff)
    upper_denominator = math.sqrt(reference_upper) + reference_roundoff
    lower = (
        lower_numerator / upper_denominator
        if upper_denominator > 0.0
        else (math.inf if lower_numerator > 0.0 else 0.0)
    )
    upper_numerator = math.sqrt(error_upper) + error_roundoff
    lower_denominator = max(0.0, math.sqrt(reference_lower) - reference_roundoff)
    upper = (
        upper_numerator / lower_denominator
        if lower_denominator > 0.0
        else (0.0 if upper_numerator == 0.0 else None)
    )
    return lower, upper


def _certify_time_norm(
    norm: _TimeNormAccumulator, limit: float, resources: ValidationResources, label: str
) -> float:
    pending = [(0.0, 1.0, 0)]
    upper_max = 0.0
    while pending:
        resources.consume_patch_work()
        left, right, depth = pending.pop()
        lower, upper = _ratio_bounds(norm, left, right)
        if lower > limit:
            raise ValueError(f"{label} relative L2 {lower:.17g} exceeds limit {limit:.17g}")
        if upper is not None and upper <= limit:
            upper_max = max(upper_max, upper)
            continue
        middle = 0.5 * (left + right)
        witness, _ = _ratio_bounds(norm, middle, middle)
        if witness > limit:
            raise ValueError(f"{label} relative L2 {witness:.17g} exceeds limit {limit:.17g}")
        if depth >= 52 or middle == left or middle == right:
            raise ValueError(f"unresolved_validation: {label} temporal ratio cannot be certified")
        pending.extend(((middle, right, depth + 1), (left, middle, depth + 1)))
    return upper_max


def _estimated_time_ratio(norm: _TimeNormAccumulator) -> float | None:
    # Report only; the acceptance gate uses bounded subinterval enclosures.
    error = _scaled_power_coefficients(norm.error)
    reference = _scaled_power_coefficients(norm.reference)
    polynomial = [
        error[2] * reference[1] - error[1] * reference[2],
        2 * (error[2] * reference[0] - error[0] * reference[2]),
        error[1] * reference[0] - error[0] * reference[1],
    ]
    points = [0.0, 1.0]
    points.extend(
        float(np.real(root))
        for root in np.roots(np.trim_zeros(polynomial, "f"))
        if abs(float(np.imag(root))) < 1e-12 and 0.0 < float(np.real(root)) < 1.0
    )
    maximum = 0.0
    for point in points:
        numerator, denominator = (
            _gram_cross(norm.error, point, point),
            _gram_cross(norm.reference, point, point),
        )
        if denominator <= 0.0 and numerator > 0.0:
            return None
        if denominator > 0.0:
            maximum = max(maximum, math.sqrt(max(0.0, numerator) / denominator))
    return maximum


def _scaled_power_coefficients(gram: FloatArray) -> tuple[float, float, float]:
    amplitude = float(np.max(np.abs(gram)))
    values = gram if amplitude == 0.0 else gram / amplitude
    return (
        float(values[0]),
        float(2 * (values[1] - values[0])),
        float(values[0] - 2 * values[1] + values[2]),
    )


def _time_interval_norms(
    references: tuple[FieldData, FieldData],
    candidates: tuple[FieldData, FieldData],
    partition: CommonPartition,
    axis_accessible: bool,
    coordinate_system: str,
) -> tuple[tuple[_TimeNormAccumulator, ...], int]:
    rows = partition.resources.workspace_rows
    source = tuple(_samples(field, partition.source, rows, axis_accessible) for field in references)
    cached = tuple(_samples(field, partition.target, rows, axis_accessible) for field in candidates)
    norms = (_TimeNormAccumulator(), _TimeNormAccumulator(), _TimeNormAccumulator())
    patches = 0
    for target_id, source_id, polygon in partition.patches():
        patches += 1
        source_roundoff = tuple(
            _roundoff_bounds(partition.source, field, source_id) for field in references
        )
        cache_roundoff = tuple(
            _roundoff_bounds(partition.target, field, target_id) for field in candidates
        )
        for triangle in _triangles(polygon):
            points, weight = _triangle_quadrature(triangle, coordinate_system)
            evaluated_source = tuple(sample.evaluate(points, derivative=True) for sample in source)
            evaluated_cache = tuple(sample.evaluate(points, derivative=True) for sample in cached)
            _require_patch_owner(source, source_id, points.shape[0])
            _require_patch_owner(cached, target_id, points.shape[0])
            for derivative, norm in enumerate(norms[:2]):
                norm.update(
                    (evaluated_source[0][derivative], evaluated_source[1][derivative]),
                    (evaluated_cache[0][derivative], evaluated_cache[1][derivative]),
                    weight,
                    (source_roundoff[0][derivative], source_roundoff[1][derivative]),
                    (cache_roundoff[0][derivative], cache_roundoff[1][derivative]),
                )
        for segment in _boundary_segments(polygon, partition.target, target_id):
            points, weight = _segment_quadrature(segment, coordinate_system)
            source_values = tuple(sample.evaluate(points, derivative=False)[0] for sample in source)
            cache_values = tuple(sample.evaluate(points, derivative=False)[0] for sample in cached)
            norms[2].update(
                (source_values[0], source_values[1]),
                (cache_values[0], cache_values[1]),
                weight,
                (source_roundoff[0][0], source_roundoff[1][0]),
                (cache_roundoff[0][0], cache_roundoff[1][0]),
            )
    return norms, patches


def _require_patch_owner(samples: tuple[_ValidationSamples, ...], cell_id: int, count: int) -> None:
    if any(not bool(np.all(sample.workspace.cell_id[:count] == cell_id)) for sample in samples):
        raise ValueError(
            "unresolved_validation: common patch is not resolved by the production locator"
        )


def _time_validation_metrics(
    source_field: FieldData,
    cached_field: FieldData,
    partition: CommonPartition,
    axis_accessible: bool,
    coordinate_system: str,
    limits: tuple[float, float, float],
) -> dict[str, object]:
    if source_field.time_s is None:
        raise AssertionError("temporal validation requires source knots")
    names = ("value", "gradient", "boundary_value")
    intervals: list[dict[str, object]] = []
    norms_by_metric: dict[str, list[tuple[float, _TimeNormAccumulator, float]]] = {
        name: [] for name in names
    }
    for index in range(source_field.time_s.size - 1):
        references = tuple(
            replace(source_field, values=source_field.values[k], time_s=None)
            for k in (index, index + 1)
        )
        candidates = tuple(
            replace(cached_field, values=cached_field.values[k], time_s=None)
            for k in (index, index + 1)
        )
        norms, patches = _time_interval_norms(
            (references[0], references[1]),
            (candidates[0], candidates[1]),
            partition,
            axis_accessible,
            coordinate_system,
        )
        width = float(source_field.time_s[index + 1] - source_field.time_s[index])
        if not math.isfinite(width):
            raise ValueError("unresolved_validation: temporal interval width is not finite")
        reports: dict[str, object] = {}
        for name, norm, limit in zip(names, norms, limits, strict=True):
            upper = _certify_time_norm(
                norm, limit, partition.resources, f"field {source_field.name!r} {name}"
            )
            norms_by_metric[name].append((width, norm, upper))
            reports[name] = {
                "error_gram": norm.error.tolist(),
                "reference_gram": norm.reference.tolist(),
                "relative_l2": _estimated_time_ratio(norm),
                "relative_l2_upper": upper,
            }
        intervals.append(
            {
                "start_s": float(source_field.time_s[index]),
                "stop_s": float(source_field.time_s[index + 1]),
                "common_patch_count": patches,
                "metrics": reports,
            }
        )
    return {
        "measure": "dx_dy" if coordinate_system == "cartesian_xy" else "2pi_r_dr_dz",
        "boundary_scope": "target_support_boundary",
        "integration_rule": "positive_duffy_gauss4x4_degree5",
        "roundoff_policy": "nodal_basis_coordinate_and_gram_allowance_v2",
        "time_scope": "maximum_ratio_on_every_linear_time_interval",
        "time_bound_method": "bounded_bernstein_subinterval_enclosures",
        "time_intervals": intervals,
        **{name: _time_norm_report(values) for name, values in norms_by_metric.items()},
    }


def _time_norm_report(values: list[tuple[float, _TimeNormAccumulator, float]]) -> dict[str, object]:
    duration = math.fsum(width for width, _norm, _upper in values)
    error = math.fsum(
        (width / duration) * float(moment) / 3
        for width, norm, _upper in values
        for moment in norm.error
    )
    reference = math.fsum(
        (width / duration) * float(moment) / 3
        for width, norm, _upper in values
        for moment in norm.reference
    )
    estimated = [_estimated_time_ratio(norm) for _width, norm, _upper in values]
    return {
        "scalar_sample_count": sum(norm.count * 2 for _width, norm, _upper in values),
        "measure": values[0][1].measure,
        "squared_error_integral": max(0.0, error),
        "squared_reference_integral": max(0.0, reference),
        "integral_time_semantics": "duration_weighted_mean_of_spatial_integral",
        "relative_l2": None
        if any(value is None for value in estimated)
        else max(float(value) for value in estimated if value is not None),
        "relative_l2_upper": max(upper for _width, _norm, upper in values),
        "sample_absolute_max": max(norm.sample_absolute_max for _width, norm, _upper in values),
    }
