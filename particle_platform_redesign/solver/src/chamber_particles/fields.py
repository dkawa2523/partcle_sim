"""Spatial location and stage-time interpolation for canonical fields."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray

from . import cpu as cpu_backend
from .case_format import (
    DataBundle,
    FieldData,
    Layout,
    P1TriLayout,
    Q1QuadLayout,
    RegularLayout,
)
from .geometry import PreparedGeometry
from .numerical_status import FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK
from .topology import PreparedPeriodicTopology

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type OutsideReason = Literal["outside_layout", "masked_cell"]
type Weights = tuple[float, ...]

FIELD_LOCATION_REVISION = "field_location_v4"
FIELD_TIME_REVISION = "fixed_topology_linear_time_v2"
FIELD_SPATIAL_GRADIENT_REVISION = "static_nodal_p1_regular_q1_gradient_v2"
REQUIRED_FIELD_REVISION = "required_field_time_linear_v6"
FIELD_LOCATION_ULPS = 64.0
FIELD_CELL_BVH_LEAF_SIZE = 8
# Local path certificates deliberately keep a bounded cell arena.  A wider
# query is not treated as outside the field; its range is left unavailable so
# the engine can restrict the same immutable path and try smaller intervals.
LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW = 64
# Per cell: the vectorized Q1 coordinate, spacing, bound, and scale arrays peak
# below 224 B; preorder partition work peaks below 128 B after those arrays are
# released.  The remaining 32 B covers ufunc outputs.  Final index arrays are
# excluded because ``prepared_nbytes`` accounts for them separately.
_FIELD_CELL_INDEX_BUILD_WORK_BYTES_PER_CELL = 256

_FLOAT_EPS = np.finfo(np.float64).eps
_MAX_JACOBIAN_CONDITION = 1.0 / math.sqrt(_FLOAT_EPS)
_MAX_REFERENCE_UNCERTAINTY = 8.0 * math.sqrt(_FLOAT_EPS)
_Q1_MAX_ITERATIONS = 20
_FIELD_KERNEL_ERRORS = (
    "ok",
    "field geometry exceeds the finite float64 range",
    "field cell is unresolved at float64 coordinate precision",
    "layout has no supported field cells",
    "field inverse mapping failed",
    "field interpolation produced a non-finite sampled value",
    "stage time lies outside the field snapshot range",
)


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
class RequiredFieldMetadata:
    """Exact canonical metadata requested by a resolved physics model."""

    unit: str
    components: tuple[str, ...]
    stored_basis: str
    positive: bool
    zero_on_rz_axis: bool = False


@dataclass(frozen=True, slots=True)
class FieldBatch:
    """Values and support returned from one shared-layout stage lookup."""

    values: Mapping[str, FloatArray]
    support_inside: NDArray[np.bool_]
    cell_id: Int64Array


@dataclass(frozen=True, slots=True)
class LocalFieldRangeBatch:
    """Conservative primitive ranges over spatial cell candidates.

    ``range_available`` means that one to
    ``LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW`` supported cells overlap the row's
    query AABB.  A larger candidate set is intentionally unavailable rather
    than dynamically allocating an unbounded arena.  Availability does not
    certify that the complete AABB is inside field support; continuous support
    remains an engine/event responsibility.
    """

    lower: Mapping[str, FloatArray]
    upper: Mapping[str, FloatArray]
    candidate_count: Int64Array
    range_available: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FieldWorkspace:
    """Reusable location and interpolation columns for one stage slab."""

    field_names: tuple[str, ...]
    sampled_values: Mapping[str, FloatArray]
    support_inside: NDArray[np.bool_]
    cell_id: Int64Array
    weights: FloatArray
    row_status: NDArray[np.uint8]
    numerical_status: NDArray[np.uint8]

    @classmethod
    def allocate(cls, fields: Mapping[str, FieldData], capacity: int) -> FieldWorkspace:
        """Allocate exact columns for one prepared field catalog."""

        if capacity < 0:
            raise ValueError("field workspace capacity must be nonnegative")
        names = tuple(fields)
        sampled = {
            name: np.empty((capacity, len(fields[name].components)), dtype="<f8") for name in names
        }
        return cls(
            names,
            MappingProxyType(sampled),
            np.empty(capacity, dtype=np.bool_),
            np.empty(capacity, dtype="<i8"),
            np.empty((capacity, 4), dtype="<f8"),
            np.empty(capacity, dtype=np.uint8),
            np.empty(capacity, dtype=np.uint8),
        )

    @property
    def capacity(self) -> int:
        """Return the number of stage rows owned by the workspace."""

        return int(self.cell_id.size)

    def batch(self, count: int) -> FieldBatch:
        """Expose only the filled prefix without copying numerical payload."""

        if count < 0 or count > self.capacity:
            raise ValueError("field workspace does not cover the requested rows")
        sampled = MappingProxyType(
            {name: self.sampled_values[name][:count] for name in self.field_names}
        )
        return FieldBatch(sampled, self.support_inside[:count], self.cell_id[:count])


@dataclass(frozen=True, slots=True)
class _FieldCellIndex:
    """Stackless read-only AABB index for unstructured containing-cell queries."""

    cell_id: Int64Array
    lower_m: FloatArray
    upper_m: FloatArray
    begin: Int64Array
    end: Int64Array
    skip: Int64Array
    build_transient_nbytes: int

    @property
    def nbytes(self) -> int:
        """Return the exact resident bytes owned by this prepared index."""

        arrays = (self.cell_id, self.lower_m, self.upper_m, self.begin, self.end, self.skip)
        return sum(int(array.nbytes) for array in arrays)


@dataclass(frozen=True, slots=True)
class PreparedFieldSet:
    """Required fields certified on one layout for the complete particle domain."""

    layout: Layout | None
    fields: Mapping[str, FieldData]
    axis_accessible: bool = False
    cell_index: _FieldCellIndex | None = None
    has_time_dependent_fields: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "has_time_dependent_fields",
            any(field.time_s is not None for field in self.fields.values()),
        )

    @property
    def uses_cell_hint(self) -> bool:
        """Return whether this layout consumes a previous-cell search hint."""

        return isinstance(self.layout, P1TriLayout | Q1QuadLayout)

    @property
    def prepared_nbytes(self) -> int:
        """Return non-canonical resident bytes owned by field preparation."""

        return 0 if self.cell_index is None else self.cell_index.nbytes

    @property
    def preparation_transient_nbytes(self) -> int:
        """Return the explicit upper bound for temporary index-build arrays."""

        return 0 if self.cell_index is None else self.cell_index.build_transient_nbytes

    def allocate_workspace(self, capacity: int) -> FieldWorkspace:
        """Allocate one reusable stage workspace for this prepared catalog."""

        return FieldWorkspace.allocate(self.fields, capacity)

    def regular_support_box(self) -> tuple[FloatArray, FloatArray] | None:
        """Return the closed convex support box for a fully supported regular layout."""

        if not isinstance(self.layout, RegularLayout):
            return None
        lower = np.asarray(
            [self.layout.axis0_m[0], self.layout.axis1_m[0]],
            dtype=np.float64,
        )
        upper = np.asarray(
            [self.layout.axis0_m[-1], self.layout.axis1_m[-1]],
            dtype=np.float64,
        )
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def constant_value(self, name: str) -> FloatArray | None:
        """Return a value only when every spatial node and snapshot stores it exactly."""

        values = self.fields[name].values
        flattened = values.reshape(-1, values.shape[-1])
        value = flattened[0].copy()
        if not bool(np.array_equal(flattened, np.broadcast_to(value, flattened.shape))):
            return None
        value.setflags(write=False)
        return value

    def component_bounds(self, name: str) -> tuple[FloatArray, FloatArray]:
        """Bound every supported nodal interpolation component conservatively.

        Supported P1, Q1, and regular nodal interpolation uses nonnegative
        partition-of-unity weights, so canonical node extrema bound the exact
        interpolant.  The scale-aware expansion also covers the finite
        multiply/add roundoff of the at-most-four-node float64 interpolation.
        """

        values = self.fields[name].values
        reduction_axes = tuple(range(values.ndim - 1))
        lower = np.min(values, axis=reduction_axes)
        upper = np.max(values, axis=reduction_axes)
        scale = np.maximum(np.abs(lower), np.abs(upper))
        expansion = FIELD_LOCATION_ULPS * _FLOAT_EPS * scale
        with np.errstate(over="ignore", invalid="ignore"):
            lower = np.nextafter(lower - expansion, -np.inf)
            upper = np.nextafter(upper + expansion, np.inf)
        if not bool(np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise FieldLocationError(f"field {name!r} component bounds are not finite")
        lower.setflags(write=False)
        upper.setflags(write=False)
        return lower, upper

    def time_knots_s(self) -> tuple[float, ...]:
        """Return the sorted union of snapshot knots used by required fields."""

        return tuple(
            sorted(
                {
                    float(time_s)
                    for field in self.fields.values()
                    if field.time_s is not None
                    for time_s in field.time_s
                }
            )
        )

    def local_component_bounds(
        self,
        position_lower_m: FloatArray,
        position_upper_m: FloatArray,
        *,
        time_lower_s: FloatArray | None = None,
        time_upper_s: FloatArray | None = None,
    ) -> LocalFieldRangeBatch:
        """Bound primitives over cells and snapshots overlapping each path box.

        P1, Q1, and regular nodal bases use nonnegative
        partition-of-unity weights.  Nodal extrema over every conservative
        cell candidate and every snapshot that brackets the requested time
        interval therefore contain the interpolated value everywhere in the
        in-support part of the query box.  Omitting both time arrays retains
        the run-global all-snapshot bound for callers without a path interval.
        This method deliberately makes no claim that the full spatial query
        box is covered by those cells.
        """

        lower_m, upper_m = _validated_local_range_boxes(position_lower_m, position_upper_m)
        count = int(lower_m.shape[0])
        lower_time_s, upper_time_s = _validated_local_time_intervals(
            time_lower_s,
            time_upper_s,
            count,
        )
        if self.layout is None:
            if self.fields:
                raise FieldLocationError("constant field catalog unexpectedly owns no layout")
            empty: Mapping[str, FloatArray] = MappingProxyType({})
            candidate_count = np.zeros(count, dtype=np.int64)
            range_available = np.zeros(count, dtype=np.bool_)
            candidate_count.setflags(write=False)
            range_available.setflags(write=False)
            return LocalFieldRangeBatch(
                empty,
                empty,
                candidate_count,
                range_available,
            )

        result_lower: dict[str, FloatArray] = {}
        result_upper: dict[str, FloatArray] = {}
        cell_index = self.cell_index
        if not isinstance(self.layout, RegularLayout) and cell_index is None:
            cell_index = _build_field_cell_index(self.layout)
        if isinstance(self.layout, RegularLayout):
            candidate_count, offsets, candidates = _local_regular_candidates(
                self.layout.axis0_m,
                self.layout.axis1_m,
                self.layout.cell_support,
                lower_m,
                upper_m,
            )
        else:
            if cell_index is None:
                raise AssertionError("unstructured field index was not prepared")
            candidate_count, offsets, candidates = _local_unstructured_candidates(
                self.layout.nodes_m,
                self.layout.connectivity,
                self.layout.cell_support,
                cell_index,
                lower_m,
                upper_m,
            )
        for name, field in self.fields.items():
            if field.association != "node":
                raise FieldLocationError(f"local bounds require node-associated field {name!r}")
            field_lower = np.empty((count, field.values.shape[-1]), dtype=np.float64)
            field_upper = np.empty_like(field_lower)
            snapshot_begin, snapshot_end = _local_snapshot_ranges(
                field,
                lower_time_s,
                upper_time_s,
                count,
            )
            if isinstance(self.layout, RegularLayout):
                if field.time_s is None:
                    _local_regular_nodal_bounds(
                        int(self.layout.axis1_m.size) - 1,
                        int(self.layout.axis1_m.size),
                        offsets,
                        candidates,
                        field.values,
                        field_lower,
                        field_upper,
                    )
                else:
                    _local_regular_snapshot_bounds(
                        int(self.layout.axis1_m.size) - 1,
                        int(self.layout.axis1_m.size),
                        offsets,
                        candidates,
                        field.values,
                        snapshot_begin,
                        snapshot_end,
                        field_lower,
                        field_upper,
                    )
            else:
                if field.time_s is None:
                    _local_unstructured_nodal_bounds(
                        self.layout.connectivity,
                        offsets,
                        candidates,
                        field.values,
                        field_lower,
                        field_upper,
                    )
                else:
                    _local_unstructured_snapshot_bounds(
                        self.layout.connectivity,
                        offsets,
                        candidates,
                        field.values,
                        snapshot_begin,
                        snapshot_end,
                        field_lower,
                        field_upper,
                    )
            _expand_local_component_bounds(
                field_lower,
                field_upper,
                (candidate_count != 0) & (candidate_count <= LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW),
                name,
            )
            field_lower.setflags(write=False)
            field_upper.setflags(write=False)
            result_lower[name] = field_lower
            result_upper[name] = field_upper

        range_available = (candidate_count != 0) & (
            candidate_count <= LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW
        )
        candidate_count.setflags(write=False)
        range_available.setflags(write=False)
        return LocalFieldRangeBatch(
            MappingProxyType(result_lower),
            MappingProxyType(result_upper),
            candidate_count,
            range_available,
        )

    def sample(
        self,
        position_m: FloatArray,
        cell_hint: Int64Array | None = None,
        workspace: FieldWorkspace | None = None,
        *,
        time_s: FloatArray | None = None,
    ) -> FieldBatch:
        """Locate one stage, then sample every required field at its stage time."""

        positions = np.asarray(position_m)
        stage_workspace = workspace
        if stage_workspace is None and positions.ndim == 2:
            stage_workspace = self.allocate_workspace(int(positions.shape[0]))
        batch, numerical_status = self.sample_batch(
            position_m,
            cell_hint,
            stage_workspace,
            time_s=time_s,
        )
        failed = np.flatnonzero(numerical_status != NUMERICAL_STATUS_OK)
        if failed.size:
            row = int(failed[0])
            if stage_workspace is None:
                raise AssertionError("field workspace was not allocated")
            _raise_kernel_error(int(stage_workspace.row_status[row]), row, "field")
        return batch

    def sample_batch(
        self,
        position_m: FloatArray,
        cell_hint: Int64Array | None = None,
        workspace: FieldWorkspace | None = None,
        *,
        time_s: FloatArray | None = None,
    ) -> tuple[FieldBatch, NDArray[np.uint8]]:
        """Sample rows while returning locator/interpolation failures as row status."""

        positions = np.ascontiguousarray(np.asarray(position_m, dtype=np.float64))
        if positions.ndim != 2 or positions.shape[1:] != (2,):
            raise ValueError("stage positions must have shape [N, 2]")
        count = int(positions.shape[0])
        stage_times = _prepare_stage_times(time_s, count, self.has_time_dependent_fields)
        cell_count = 0 if self.layout is None else _cell_count(self.layout)
        hints = _prepare_cell_hints(cell_hint, count, cell_count)
        stage_workspace = self.allocate_workspace(count) if workspace is None else workspace
        if stage_workspace.field_names != tuple(self.fields):
            raise ValueError("field workspace does not match the prepared field catalog")
        batch = stage_workspace.batch(count)
        row_status = stage_workspace.row_status[:count]
        numerical_status = stage_workspace.numerical_status[:count]
        row_status.fill(cpu_backend.FIELD_KERNEL_OK)
        nonfinite_position = ~np.isfinite(positions).all(axis=1)
        row_status[nonfinite_position] = cpu_backend.FIELD_KERNEL_NONFINITE_GEOMETRY
        if self.layout is None:
            batch.support_inside.fill(True)
            batch.cell_id.fill(-1)
            _field_numerical_status(row_status, numerical_status)
            return batch, numerical_status

        support = batch.support_inside
        cell_id = batch.cell_id
        weights = stage_workspace.weights[:count]
        _locate_field_stage(
            self.layout,
            self.cell_index,
            positions,
            hints,
            support,
            cell_id,
            weights,
            row_status,
        )
        for name, field in self.fields.items():
            if field.layout != self.layout.name:
                raise ValueError(
                    f"field {field.name} uses layout {field.layout}, not {self.layout.name}"
                )
            values = batch.values[name]
            zero_axis_radial = self.axis_accessible and (
                field.components == ("r", "z") and field.stored_basis == "axisymmetric_rz"
            )
            _sample_compiled_field(
                self.layout,
                field,
                positions,
                stage_times,
                cell_id,
                weights,
                zero_axis_radial,
                values,
                row_status,
            )
        _field_numerical_status(row_status, numerical_status)
        return batch, numerical_status

    def spatial_gradient(self, name: str, workspace: FieldWorkspace, output: FloatArray) -> None:
        """Differentiate a static nodal field at an already sampled stage.

        This opt-in operation consumes the same cell IDs and basis weights as
        ``sample``. It neither locates again nor adds production-stage scratch.
        Derivatives refer to stored components in the canonical two coordinates.
        """

        field = self.fields[name]
        if field.time_s is not None or field.association != "node":
            raise ValueError("spatial gradients require a static nodal field")
        if not isinstance(self.layout, P1TriLayout | RegularLayout | Q1QuadLayout):
            raise ValueError("spatial gradients require a supported field layout")
        count = int(output.shape[0])
        if output.shape != (count, len(field.components), 2):
            raise ValueError("spatial gradient output must have shape [N, C, 2]")
        if output.dtype != np.float64 or not output.flags.c_contiguous:
            raise ValueError("spatial gradient output must be contiguous float64")
        if workspace.field_names != tuple(self.fields) or count > workspace.capacity:
            raise ValueError("spatial gradient workspace does not match the sampled stage")
        if not bool(workspace.support_inside[:count].all()):
            raise ValueError("spatial gradients require supported sampled rows")
        status = workspace.row_status[:count]
        if isinstance(self.layout, P1TriLayout):
            cpu_backend.sample_p1_nodal_gradient(
                self.layout.nodes_m,
                self.layout.connectivity,
                workspace.cell_id[:count],
                field.values,
                output,
                status,
            )
        elif isinstance(self.layout, Q1QuadLayout):
            cpu_backend.sample_q1_nodal_gradient(
                self.layout.nodes_m,
                self.layout.connectivity,
                workspace.cell_id[:count],
                workspace.weights[:count],
                field.values,
                output,
                status,
            )
        else:
            cpu_backend.sample_regular_nodal_gradient(
                self.layout.axis0_m,
                self.layout.axis1_m,
                workspace.cell_id[:count],
                workspace.weights[:count],
                field.values,
                output,
                status,
            )
        failed = np.flatnonzero(status != cpu_backend.FIELD_KERNEL_OK)
        if failed.size:
            _raise_kernel_error(int(status[int(failed[0])]), int(failed[0]), "field gradient")


class FieldLocationError(RuntimeError):
    """A field location could not be computed from a numerically valid basis."""


def _prepare_stage_times(
    time_s: FloatArray | None,
    count: int,
    required: bool,
) -> FloatArray | None:
    if not required:
        return None
    if time_s is None:
        raise ValueError("stage times are required for time-dependent fields")
    times = np.ascontiguousarray(np.asarray(time_s, dtype=np.float64))
    if times.shape != (count,):
        raise ValueError("stage times must have shape [N]")
    return times


def _prepare_cell_hints(
    cell_hint: Int64Array | None, count: int, cell_count: int
) -> Int64Array | None:
    if cell_hint is None:
        return None
    hints = np.asarray(cell_hint)
    if hints.shape != (count,) or hints.dtype.kind not in "iu":
        raise ValueError("cell_hint must be an integer array with shape [N]")
    if bool((hints < -1).any()) or bool((hints >= cell_count).any()):
        raise ValueError("cell_hint contains an entry outside the layout cell range")
    return np.ascontiguousarray(hints, dtype=np.int64)


def _validated_local_range_boxes(
    lower_m: FloatArray,
    upper_m: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    lower = np.ascontiguousarray(np.asarray(lower_m, dtype=np.float64))
    upper = np.ascontiguousarray(np.asarray(upper_m, dtype=np.float64))
    if lower.ndim != 2 or lower.shape[1:] != (2,) or upper.shape != lower.shape:
        raise ValueError("position range bounds must have matching shape [N, 2]")
    if not bool(np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("position range bounds must contain only finite values")
    if bool((lower > upper).any()):
        raise ValueError("position range lower bounds exceed upper bounds")
    return lower, upper


def _validated_local_time_intervals(
    lower_s: FloatArray | None,
    upper_s: FloatArray | None,
    count: int,
) -> tuple[FloatArray | None, FloatArray | None]:
    if lower_s is None and upper_s is None:
        return None, None
    if lower_s is None or upper_s is None:
        raise ValueError("local field bounds require both time interval arrays")
    lower = np.ascontiguousarray(np.asarray(lower_s, dtype=np.float64))
    upper = np.ascontiguousarray(np.asarray(upper_s, dtype=np.float64))
    if lower.shape != (count,) or upper.shape != (count,):
        raise ValueError("time interval bounds must have matching shape [N]")
    if not bool(np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("time interval bounds must contain only finite values")
    if bool((lower > upper).any()):
        raise ValueError("time interval lower bounds exceed upper bounds")
    return lower, upper


def _local_snapshot_ranges(
    field: FieldData,
    lower_time_s: FloatArray | None,
    upper_time_s: FloatArray | None,
    count: int,
) -> tuple[Int64Array, Int64Array]:
    """Return half-open snapshot ranges that enclose linear time interpolation."""

    snapshot_time_s = field.time_s
    if snapshot_time_s is None:
        empty = np.empty(0, dtype="<i8")
        return empty, empty
    if lower_time_s is None or upper_time_s is None:
        return (
            np.zeros(count, dtype="<i8"),
            np.full(count, snapshot_time_s.size, dtype="<i8"),
        )
    if bool(
        (lower_time_s < snapshot_time_s[0]).any() or (upper_time_s > snapshot_time_s[-1]).any()
    ):
        raise FieldLocationError(
            f"field {field.name!r} cannot bound time outside its snapshot range"
        )
    begin = np.searchsorted(snapshot_time_s, lower_time_s, side="right") - 1
    begin = np.clip(begin, 0, snapshot_time_s.size - 2).astype("<i8", copy=False)
    end = np.searchsorted(snapshot_time_s, upper_time_s, side="left") + 1
    end = np.clip(end, begin + 2, snapshot_time_s.size).astype("<i8", copy=False)
    return np.ascontiguousarray(begin), np.ascontiguousarray(end)


def _expand_local_component_bounds(
    lower: FloatArray,
    upper: FloatArray,
    available: NDArray[np.bool_],
    name: str,
) -> None:
    """Expand finite candidate extrema over interpolation roundoff."""

    lower[~available] = 0.0
    upper[~available] = 0.0
    if not bool(available.any()):
        return
    selected_lower = lower[available]
    selected_upper = upper[available]
    scale = np.maximum(np.abs(selected_lower), np.abs(selected_upper))
    expansion = FIELD_LOCATION_ULPS * _FLOAT_EPS * scale
    with np.errstate(over="ignore", invalid="ignore"):
        selected_lower = np.nextafter(selected_lower - expansion, -np.inf)
        selected_upper = np.nextafter(selected_upper + expansion, np.inf)
    if not bool(np.isfinite(selected_lower).all() and np.isfinite(selected_upper).all()):
        raise FieldLocationError(f"field {name!r} local component bounds are not finite")
    lower[available] = selected_lower
    upper[available] = selected_upper


def _local_regular_candidates(
    axis0_m: FloatArray,
    axis1_m: FloatArray,
    cell_support: NDArray[np.uint8],
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
) -> tuple[Int64Array, Int64Array, Int64Array]:
    counts = _count_local_regular_candidates(
        axis0_m,
        axis1_m,
        cell_support,
        query_lower_m,
        query_upper_m,
    )
    bounded_counts = np.where(
        counts <= LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW,
        counts,
        0,
    )
    offsets, candidates = _allocate_local_candidate_csr(bounded_counts)
    _fill_local_regular_candidates(
        axis0_m,
        axis1_m,
        cell_support,
        query_lower_m,
        query_upper_m,
        offsets,
        candidates,
    )
    return counts, offsets, candidates


def _allocate_local_candidate_csr(counts: Int64Array) -> tuple[Int64Array, Int64Array]:
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    candidates = np.empty(int(offsets[-1]), dtype=np.int64)
    return offsets, candidates


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _count_local_regular_candidates(
    axis0_m: FloatArray,
    axis1_m: FloatArray,
    cell_support: NDArray[np.uint8],
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
) -> Int64Array:
    """Count supported regular cells overlapping each query box."""

    row_count = query_lower_m.shape[0]
    counts = np.zeros(row_count, dtype=np.int64)
    for row in range(row_count):
        pad0 = _regular_query_padding(axis0_m, query_lower_m[row, 0], query_upper_m[row, 0])
        pad1 = _regular_query_padding(axis1_m, query_lower_m[row, 1], query_upper_m[row, 1])
        first0, last0 = _overlapping_regular_intervals(
            axis0_m,
            query_lower_m[row, 0] - pad0,
            query_upper_m[row, 0] + pad0,
        )
        first1, last1 = _overlapping_regular_intervals(
            axis1_m,
            query_lower_m[row, 1] - pad1,
            query_upper_m[row, 1] + pad1,
        )
        if first0 < 0 or first1 < 0:
            continue
        for index0 in range(first0, last0 + 1):
            for index1 in range(first1, last1 + 1):
                if cell_support[index0, index1] != 0:
                    counts[row] += 1
    return counts


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _fill_local_regular_candidates(
    axis0_m: FloatArray,
    axis1_m: FloatArray,
    cell_support: NDArray[np.uint8],
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
    offsets: Int64Array,
    candidates: Int64Array,
) -> None:
    axis1_cell_count = axis1_m.size - 1
    for row in range(query_lower_m.shape[0]):
        if offsets[row] == offsets[row + 1]:
            continue
        write = offsets[row]
        pad0 = _regular_query_padding(axis0_m, query_lower_m[row, 0], query_upper_m[row, 0])
        pad1 = _regular_query_padding(axis1_m, query_lower_m[row, 1], query_upper_m[row, 1])
        first0, last0 = _overlapping_regular_intervals(
            axis0_m,
            query_lower_m[row, 0] - pad0,
            query_upper_m[row, 0] + pad0,
        )
        first1, last1 = _overlapping_regular_intervals(
            axis1_m,
            query_lower_m[row, 1] - pad1,
            query_upper_m[row, 1] + pad1,
        )
        if first0 < 0 or first1 < 0:
            continue
        for index0 in range(first0, last0 + 1):
            for index1 in range(first1, last1 + 1):
                if cell_support[index0, index1] != 0:
                    candidates[write] = index0 * axis1_cell_count + index1
                    write += 1


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _local_regular_nodal_bounds(
    axis1_cell_count: int,
    axis1_size: int,
    offsets: Int64Array,
    candidates: Int64Array,
    values: FloatArray,
    result_lower: FloatArray,
    result_upper: FloatArray,
) -> None:
    """Reduce regular nodal extrema over precomputed cell candidates."""

    result_lower.fill(0.0)
    result_upper.fill(0.0)
    for row in range(offsets.size - 1):
        initialized = False
        for offset in range(offsets[row], offsets[row + 1]):
            cell_id = candidates[offset]
            index0 = cell_id // axis1_cell_count
            index1 = cell_id - index0 * axis1_cell_count
            node_ids = (
                index0 * axis1_size + index1,
                (index0 + 1) * axis1_size + index1,
                (index0 + 1) * axis1_size + index1 + 1,
                index0 * axis1_size + index1 + 1,
            )
            for node_id in node_ids:
                for component in range(values.shape[1]):
                    value = values[node_id, component]
                    if not initialized:
                        result_lower[row, component] = value
                        result_upper[row, component] = value
                    else:
                        result_lower[row, component] = min(result_lower[row, component], value)
                        result_upper[row, component] = max(result_upper[row, component], value)
                initialized = True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _local_regular_snapshot_bounds(
    axis1_cell_count: int,
    axis1_size: int,
    offsets: Int64Array,
    candidates: Int64Array,
    values: FloatArray,
    snapshot_begin: Int64Array,
    snapshot_end: Int64Array,
    result_lower: FloatArray,
    result_upper: FloatArray,
) -> None:
    """Reduce extrema over spatial candidates and row-local snapshots."""

    result_lower.fill(0.0)
    result_upper.fill(0.0)
    for row in range(offsets.size - 1):
        initialized = False
        for offset in range(offsets[row], offsets[row + 1]):
            cell_id = candidates[offset]
            index0 = cell_id // axis1_cell_count
            index1 = cell_id - index0 * axis1_cell_count
            node_ids = (
                index0 * axis1_size + index1,
                (index0 + 1) * axis1_size + index1,
                (index0 + 1) * axis1_size + index1 + 1,
                index0 * axis1_size + index1 + 1,
            )
            for snapshot in range(snapshot_begin[row], snapshot_end[row]):
                for node_id in node_ids:
                    for component in range(values.shape[2]):
                        value = values[snapshot, node_id, component]
                        if not initialized:
                            result_lower[row, component] = value
                            result_upper[row, component] = value
                        else:
                            result_lower[row, component] = min(result_lower[row, component], value)
                            result_upper[row, component] = max(result_upper[row, component], value)
                    initialized = True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _regular_query_padding(axis_m: FloatArray, lower_m: float, upper_m: float) -> float:
    spacing = max(abs(float(np.spacing(lower_m))), abs(float(np.spacing(upper_m))))
    for coordinate in axis_m:
        spacing = max(spacing, abs(float(np.spacing(coordinate))))
    spacing = max(spacing, _FLOAT_EPS * (axis_m[-1] - axis_m[0]))
    return FIELD_LOCATION_ULPS * spacing


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _overlapping_regular_intervals(
    axis_m: FloatArray,
    lower_m: float,
    upper_m: float,
) -> tuple[int, int]:
    if upper_m < axis_m[0] or axis_m[-1] < lower_m:
        return -1, -1
    first = int(np.searchsorted(axis_m, lower_m, side="left")) - 1
    last = int(np.searchsorted(axis_m, upper_m, side="right")) - 1
    return max(first, 0), min(last, axis_m.size - 2)


def _local_unstructured_candidates(
    nodes_m: FloatArray,
    connectivity: Int64Array,
    cell_support: NDArray[np.uint8],
    cell_index: _FieldCellIndex,
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
) -> tuple[Int64Array, Int64Array, Int64Array]:
    counts = _count_local_unstructured_candidates(
        nodes_m,
        connectivity,
        cell_support,
        cell_index.cell_id,
        cell_index.lower_m,
        cell_index.upper_m,
        cell_index.begin,
        cell_index.end,
        cell_index.skip,
        query_lower_m,
        query_upper_m,
    )
    bounded_counts = np.where(
        counts <= LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW,
        counts,
        0,
    )
    offsets, candidates = _allocate_local_candidate_csr(bounded_counts)
    _fill_local_unstructured_candidates(
        nodes_m,
        connectivity,
        cell_support,
        cell_index.cell_id,
        cell_index.lower_m,
        cell_index.upper_m,
        cell_index.begin,
        cell_index.end,
        cell_index.skip,
        query_lower_m,
        query_upper_m,
        offsets,
        candidates,
    )
    return counts, offsets, candidates


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _count_local_unstructured_candidates(
    nodes_m: FloatArray,
    connectivity: Int64Array,
    cell_support: NDArray[np.uint8],
    index_cell_id: Int64Array,
    index_lower_m: FloatArray,
    index_upper_m: FloatArray,
    index_begin: Int64Array,
    index_end: Int64Array,
    index_skip: Int64Array,
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
) -> Int64Array:
    """Count padded P1/Q1 cell AABBs overlapping each query box."""

    row_count = query_lower_m.shape[0]
    counts = np.zeros(row_count, dtype=np.int64)
    for row in range(row_count):
        node = 0
        while node < index_skip.size:
            if not _local_boxes_overlap(
                query_lower_m[row],
                query_upper_m[row],
                index_lower_m[node],
                index_upper_m[node],
            ):
                node = index_skip[node]
                continue
            begin = index_begin[node]
            if begin < 0:
                node += 1
                continue
            for offset in range(begin, index_end[node]):
                cell_id = index_cell_id[offset]
                if cell_support[cell_id] == 0:
                    continue
                cell_lower, cell_upper = _padded_unstructured_cell_box(
                    nodes_m,
                    connectivity[cell_id],
                )
                if not _local_boxes_overlap(
                    query_lower_m[row],
                    query_upper_m[row],
                    cell_lower,
                    cell_upper,
                ):
                    continue
                counts[row] += 1
            node = index_skip[node]
    return counts


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _fill_local_unstructured_candidates(
    nodes_m: FloatArray,
    connectivity: Int64Array,
    cell_support: NDArray[np.uint8],
    index_cell_id: Int64Array,
    index_lower_m: FloatArray,
    index_upper_m: FloatArray,
    index_begin: Int64Array,
    index_end: Int64Array,
    index_skip: Int64Array,
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
    offsets: Int64Array,
    candidates: Int64Array,
) -> None:
    for row in range(query_lower_m.shape[0]):
        if offsets[row] == offsets[row + 1]:
            continue
        write = offsets[row]
        node = 0
        while node < index_skip.size:
            if not _local_boxes_overlap(
                query_lower_m[row],
                query_upper_m[row],
                index_lower_m[node],
                index_upper_m[node],
            ):
                node = index_skip[node]
                continue
            begin = index_begin[node]
            if begin < 0:
                node += 1
                continue
            for offset in range(begin, index_end[node]):
                cell_id = index_cell_id[offset]
                if cell_support[cell_id] == 0:
                    continue
                cell_lower, cell_upper = _padded_unstructured_cell_box(
                    nodes_m,
                    connectivity[cell_id],
                )
                if _local_boxes_overlap(
                    query_lower_m[row],
                    query_upper_m[row],
                    cell_lower,
                    cell_upper,
                ):
                    candidates[write] = cell_id
                    write += 1
            node = index_skip[node]


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _local_unstructured_nodal_bounds(
    connectivity: Int64Array,
    offsets: Int64Array,
    candidates: Int64Array,
    values: FloatArray,
    result_lower: FloatArray,
    result_upper: FloatArray,
) -> None:
    """Reduce P1/Q1 nodal extrema over precomputed cell candidates."""

    result_lower.fill(0.0)
    result_upper.fill(0.0)
    for row in range(offsets.size - 1):
        initialized = False
        for offset in range(offsets[row], offsets[row + 1]):
            for node_id in connectivity[candidates[offset]]:
                for component in range(values.shape[1]):
                    value = values[node_id, component]
                    if not initialized:
                        result_lower[row, component] = value
                        result_upper[row, component] = value
                    else:
                        result_lower[row, component] = min(result_lower[row, component], value)
                        result_upper[row, component] = max(result_upper[row, component], value)
                initialized = True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _local_unstructured_snapshot_bounds(
    connectivity: Int64Array,
    offsets: Int64Array,
    candidates: Int64Array,
    values: FloatArray,
    snapshot_begin: Int64Array,
    snapshot_end: Int64Array,
    result_lower: FloatArray,
    result_upper: FloatArray,
) -> None:
    """Reduce P1/Q1 extrema over candidates and row-local snapshots."""

    result_lower.fill(0.0)
    result_upper.fill(0.0)
    for row in range(offsets.size - 1):
        initialized = False
        for offset in range(offsets[row], offsets[row + 1]):
            for snapshot in range(snapshot_begin[row], snapshot_end[row]):
                for node_id in connectivity[candidates[offset]]:
                    for component in range(values.shape[2]):
                        value = values[snapshot, node_id, component]
                        if not initialized:
                            result_lower[row, component] = value
                            result_upper[row, component] = value
                        else:
                            result_lower[row, component] = min(result_lower[row, component], value)
                            result_upper[row, component] = max(result_upper[row, component], value)
                    initialized = True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _padded_unstructured_cell_box(
    nodes_m: FloatArray,
    node_ids: Int64Array,
) -> tuple[FloatArray, FloatArray]:
    lower = np.full(2, np.inf, dtype=np.float64)
    upper = np.full(2, -np.inf, dtype=np.float64)
    node_spacing = 0.0
    for node_id in node_ids:
        for axis in range(2):
            coordinate = nodes_m[node_id, axis]
            lower[axis] = min(lower[axis], coordinate)
            upper[axis] = max(upper[axis], coordinate)
            node_spacing = max(node_spacing, abs(float(np.spacing(coordinate))))
    extent0 = upper[0] - lower[0]
    extent1 = upper[1] - lower[1]
    diameter = math.hypot(extent0, extent1)
    padding = np.nextafter(
        FIELD_LOCATION_ULPS * max(node_spacing, _FLOAT_EPS * diameter),
        np.inf,
    )
    for axis in range(2):
        lower[axis] = np.nextafter(lower[axis] - padding, -np.inf)
        upper[axis] = np.nextafter(upper[axis] + padding, np.inf)
    return lower, upper


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _local_boxes_overlap(
    first_lower: FloatArray,
    first_upper: FloatArray,
    second_lower: FloatArray,
    second_upper: FloatArray,
) -> bool:
    return not (
        first_upper[0] < second_lower[0]
        or first_upper[1] < second_lower[1]
        or second_upper[0] < first_lower[0]
        or second_upper[1] < first_lower[1]
    )


def _locate_field_stage(
    layout: Layout,
    cell_index: _FieldCellIndex | None,
    positions: FloatArray,
    hints: Int64Array | None,
    support: NDArray[np.bool_],
    cell_id: Int64Array,
    weights: FloatArray,
    row_status: NDArray[np.uint8],
) -> None:
    support.fill(False)
    cell_id.fill(-1)
    weights.fill(0.0)
    if not bool((row_status != cpu_backend.FIELD_KERNEL_OK).any()):
        _locate_field_rows(
            layout,
            cell_index,
            positions,
            hints,
            support,
            cell_id,
            weights,
            row_status,
        )
        return
    active = np.flatnonzero(row_status == cpu_backend.FIELD_KERNEL_OK)
    if not active.size:
        return
    active_hints = None if hints is None else np.ascontiguousarray(hints[active])
    active_support = np.empty(active.size, dtype=np.bool_)
    active_cell = np.empty(active.size, dtype=np.int64)
    active_weights = np.empty((active.size, 4), dtype=np.float64)
    active_status = np.empty(active.size, dtype=np.uint8)
    _locate_field_rows(
        layout,
        cell_index,
        np.ascontiguousarray(positions[active]),
        active_hints,
        active_support,
        active_cell,
        active_weights,
        active_status,
    )
    support[active] = active_support
    cell_id[active] = active_cell
    weights[active] = active_weights
    row_status[active] = active_status


def _locate_field_rows(
    layout: Layout,
    cell_index: _FieldCellIndex | None,
    positions: FloatArray,
    hints: Int64Array | None,
    support: NDArray[np.bool_],
    cell_id: Int64Array,
    weights: FloatArray,
    row_status: NDArray[np.uint8],
) -> None:
    if isinstance(layout, RegularLayout):
        cpu_backend.locate_regular_field_batch(
            layout.axis0_m,
            layout.axis1_m,
            layout.cell_support,
            positions,
            support,
            cell_id,
            weights,
            row_status,
        )
    else:
        if hints is None:
            hints = np.full(positions.shape[0], -1, dtype="<i8")
        node_count = 3 if isinstance(layout, P1TriLayout) else 4
        if cell_index is None:
            index_cell_id = np.empty(0, dtype="<i8")
            index_lower = np.empty((0, 2), dtype="<f8")
            index_upper = np.empty((0, 2), dtype="<f8")
            index_begin = np.empty(0, dtype="<i8")
            index_end = np.empty(0, dtype="<i8")
            index_skip = np.empty(0, dtype="<i8")
        else:
            index_cell_id = cell_index.cell_id
            index_lower = cell_index.lower_m
            index_upper = cell_index.upper_m
            index_begin = cell_index.begin
            index_end = cell_index.end
            index_skip = cell_index.skip
        cpu_backend.locate_unstructured_field_batch(
            node_count,
            layout.nodes_m,
            layout.connectivity,
            layout.cell_support,
            index_cell_id,
            index_lower,
            index_upper,
            index_begin,
            index_end,
            index_skip,
            positions,
            hints,
            support,
            cell_id,
            weights,
            row_status,
        )


def _sample_compiled_field(
    layout: Layout,
    field: FieldData,
    positions: FloatArray,
    time_s: FloatArray | None,
    cell_id: Int64Array,
    weights: FloatArray,
    zero_axis_radial: bool,
    sampled: FloatArray,
    row_status: NDArray[np.uint8],
) -> None:
    sampled.fill(0.0)
    if not bool((row_status != cpu_backend.FIELD_KERNEL_OK).any()):
        _sample_compiled_rows(
            layout,
            field,
            positions,
            time_s,
            cell_id,
            weights,
            zero_axis_radial,
            sampled,
            row_status,
        )
        sampled[row_status != cpu_backend.FIELD_KERNEL_OK] = 0.0
        return
    active = np.flatnonzero(row_status == cpu_backend.FIELD_KERNEL_OK)
    if not active.size:
        return
    active_sampled = np.empty((active.size, sampled.shape[1]), dtype=np.float64)
    active_status = np.empty(active.size, dtype=np.uint8)
    active_time_s = None if time_s is None else np.ascontiguousarray(time_s[active])
    _sample_compiled_rows(
        layout,
        field,
        np.ascontiguousarray(positions[active]),
        active_time_s,
        np.ascontiguousarray(cell_id[active]),
        np.ascontiguousarray(weights[active]),
        zero_axis_radial,
        active_sampled,
        active_status,
    )
    sampled[active] = active_sampled
    row_status[active] = active_status
    sampled[row_status != cpu_backend.FIELD_KERNEL_OK] = 0.0


def _sample_compiled_rows(
    layout: Layout,
    field: FieldData,
    positions: FloatArray,
    time_s: FloatArray | None,
    cell_id: Int64Array,
    weights: FloatArray,
    zero_axis_radial: bool,
    sampled: FloatArray,
    row_status: NDArray[np.uint8],
) -> None:
    if field.time_s is not None:
        if time_s is None:
            raise ValueError("stage times are required for time-dependent fields")
        if field.association == "cell":
            cpu_backend.sample_cell_time_field(
                positions,
                time_s,
                cell_id,
                field.time_s,
                field.values,
                zero_axis_radial,
                sampled,
                row_status,
            )
        elif isinstance(layout, RegularLayout):
            cpu_backend.sample_regular_nodal_time_field(
                int(layout.axis1_m.size),
                positions,
                time_s,
                cell_id,
                weights,
                field.time_s,
                field.values,
                zero_axis_radial,
                sampled,
                row_status,
            )
        else:
            node_count = 3 if isinstance(layout, P1TriLayout) else 4
            cpu_backend.sample_unstructured_nodal_time_field(
                node_count,
                layout.connectivity,
                positions,
                time_s,
                cell_id,
                weights,
                field.time_s,
                field.values,
                zero_axis_radial,
                sampled,
                row_status,
            )
        return
    if field.association == "cell":
        cpu_backend.sample_cell_field(
            positions, cell_id, field.values, zero_axis_radial, sampled, row_status
        )
    elif isinstance(layout, RegularLayout):
        cpu_backend.sample_regular_nodal_field(
            int(layout.axis1_m.size),
            positions,
            cell_id,
            weights,
            field.values,
            zero_axis_radial,
            sampled,
            row_status,
        )
    else:
        node_count = 3 if isinstance(layout, P1TriLayout) else 4
        cpu_backend.sample_unstructured_nodal_field(
            node_count,
            layout.connectivity,
            positions,
            cell_id,
            weights,
            field.values,
            zero_axis_radial,
            sampled,
            row_status,
        )


def _field_numerical_status(
    row_status: NDArray[np.uint8],
    numerical_status: NDArray[np.uint8],
) -> None:
    numerical_status.fill(NUMERICAL_STATUS_OK)
    numerical_status[row_status != cpu_backend.FIELD_KERNEL_OK] = FIELD_NUMERICAL_FAILURE


def _raise_kernel_error(status: int, row: int, owner: str) -> None:
    if status == cpu_backend.FIELD_KERNEL_OK:
        return
    if 0 <= status < len(_FIELD_KERNEL_ERRORS):
        detail = _FIELD_KERNEL_ERRORS[status]
    else:
        detail = f"unknown compiled field-kernel status {status}"
    raise FieldLocationError(f"{owner!r} failed at stage row {row}: {detail}")


@dataclass(frozen=True, slots=True)
class _Candidate:
    cell_id: int
    node_ids: tuple[int, ...]
    weights: Weights
    supported: bool


def prepare_required_fields(
    data: DataBundle,
    requirements: Mapping[str, RequiredFieldMetadata],
    *,
    time_interval_s: tuple[float, float] | None = None,
) -> PreparedFieldSet:
    """Bind exact field metadata and certify the supported particle domain."""

    if not requirements:
        return PreparedFieldSet(
            None,
            MappingProxyType({}),
            data.coordinate_system == "axisymmetric_rz"
            and bool((data.geometry.nodes_m[:, 0] == 0.0).any()),
        )
    available_fields = {field.name: field for field in data.fields}
    available_layouts = {layout.name: layout for layout in data.layouts}
    selected: dict[str, FieldData] = {}
    layout_name: str | None = None
    for name in sorted(requirements):
        requirement = requirements[name]
        field = available_fields.get(name)
        if field is None:
            raise FieldLocationError(f"required field {name!r} is missing")
        if field.association != "node":
            raise FieldLocationError(f"required field {name!r} must be node-associated")
        actual = (field.unit, field.components, field.stored_basis)
        expected = (requirement.unit, requirement.components, requirement.stored_basis)
        if actual != expected:
            raise FieldLocationError(
                f"required field {name!r} metadata does not match unit/components/basis"
            )
        if requirement.positive and bool((field.values <= 0.0).any()):
            raise FieldLocationError(f"required field {name!r} must be positive at every node")
        _certify_field_time_interval(field, time_interval_s)
        if layout_name is None:
            layout_name = field.layout
        elif field.layout != layout_name:
            raise FieldLocationError("all P06 required fields must use one common layout")
        selected[name] = field
    if layout_name is None:
        raise FieldLocationError("required field binding did not resolve a layout")
    layout = available_layouts[layout_name]
    _certify_particle_domain_coverage(data, layout)
    _certify_layout_numerics(layout)
    axis_accessible = rz_axis_accessible(data, layout)
    certify_rz_axis_field_regularity(
        data.coordinate_system,
        layout,
        selected,
        axis_accessible,
        requirements,
    )
    cell_index = _build_field_cell_index(layout)
    return PreparedFieldSet(layout, MappingProxyType(selected), axis_accessible, cell_index)


def validate_periodic_field_seams(
    prepared: PreparedFieldSet,
    topology: PreparedPeriodicTopology | None,
    geometry: PreparedGeometry,
) -> None:
    """Require every selected static primitive to be continuous across a seam.

    Pure translations preserve stored Cartesian components.  P1 and Q1 fields
    share the geometry mesh, so equality at paired edge nodes proves equality
    along their linear boundary traces.  A regular field is accepted only for
    an axis-aligned pair of opposite support-box faces.
    """

    if topology is None or not prepared.fields:
        return
    if prepared.has_time_dependent_fields:
        raise FieldLocationError(
            "translation_periodic_xy_v1 does not support time-dependent required fields"
        )
    if any(field.association != "node" for field in prepared.fields.values()):
        raise FieldLocationError(
            "translation_periodic_xy_v1 requires nodal required fields for seam matching"
        )
    layout = prepared.layout
    if layout is None:
        raise FieldLocationError("periodic required fields unexpectedly have no layout")
    if isinstance(layout, RegularLayout):
        _validate_regular_periodic_seams(prepared, topology, geometry, layout)
        return
    _validate_unstructured_periodic_seams(prepared, topology, geometry)


def _validate_unstructured_periodic_seams(
    prepared: PreparedFieldSet,
    topology: PreparedPeriodicTopology,
    geometry: PreparedGeometry,
) -> None:
    for facet_id in topology.periodic_facet_id:
        facet = int(facet_id)
        peer = int(topology.peer_facet_id[facet])
        if facet > peer:
            continue
        source_node_ids = geometry.facet_node_ids[facet]
        peer_node_ids = topology.peer_node_ids[facet]
        for name, field in prepared.fields.items():
            _require_periodic_values_match(
                name,
                field.values[source_node_ids],
                field.values[peer_node_ids],
                field.values,
                topology.field_match_rtol,
            )


def _validate_regular_periodic_seams(
    prepared: PreparedFieldSet,
    topology: PreparedPeriodicTopology,
    geometry: PreparedGeometry,
    layout: RegularLayout,
) -> None:
    checked_pairs: set[int] = set()
    for facet_id in topology.periodic_facet_id:
        facet = int(facet_id)
        seam_pair_id = int(topology.pair_id[facet])
        if seam_pair_id in checked_pairs:
            continue
        checked_pairs.add(seam_pair_id)
        translation = topology.translation_m[facet]
        axis = _regular_periodic_axis(layout, translation, topology.position_tolerance_m)
        _require_facets_on_regular_support_faces(
            geometry,
            topology,
            seam_pair_id,
            facet,
            layout,
            axis,
        )
        for name, field in prepared.fields.items():
            values = field.values.reshape(
                layout.axis0_m.size,
                layout.axis1_m.size,
                field.values.shape[-1],
            )
            first = values[0, :, :] if axis == 0 else values[:, 0, :]
            second = values[-1, :, :] if axis == 0 else values[:, -1, :]
            _require_periodic_values_match(
                name,
                first,
                second,
                field.values,
                topology.field_match_rtol,
            )


def _regular_periodic_axis(
    layout: RegularLayout, translation_m: FloatArray, tolerance_m: float
) -> int:
    spans = np.asarray(
        [layout.axis0_m[-1] - layout.axis0_m[0], layout.axis1_m[-1] - layout.axis1_m[0]],
        dtype=np.float64,
    )
    candidates = [
        axis
        for axis in (0, 1)
        if abs(float(translation_m[1 - axis])) <= tolerance_m
        and abs(abs(float(translation_m[axis])) - float(spans[axis])) <= tolerance_m
    ]
    if len(candidates) != 1:
        raise FieldLocationError(
            "regular periodic fields require one axis-aligned support-box translation"
        )
    return candidates[0]


def _require_facets_on_regular_support_faces(
    geometry: PreparedGeometry,
    topology: PreparedPeriodicTopology,
    seam_pair_id: int,
    facet: int,
    layout: RegularLayout,
    axis: int,
) -> None:
    translation = topology.translation_m[facet]
    if float(translation[axis]) > 0.0:
        source_coordinate = float((layout.axis0_m if axis == 0 else layout.axis1_m)[0])
        peer_coordinate = float((layout.axis0_m if axis == 0 else layout.axis1_m)[-1])
    else:
        source_coordinate = float((layout.axis0_m if axis == 0 else layout.axis1_m)[-1])
        peer_coordinate = float((layout.axis0_m if axis == 0 else layout.axis1_m)[0])
    pair_facets = topology.periodic_facet_id[
        topology.pair_id[topology.periodic_facet_id] == seam_pair_id
    ]
    source_facets = pair_facets[topology.translation_m[pair_facets, axis] * translation[axis] > 0.0]
    peer_facets = pair_facets[topology.translation_m[pair_facets, axis] * translation[axis] < 0.0]
    if source_facets.size == 0 or peer_facets.size == 0:
        raise FieldLocationError("regular periodic seam has an incomplete reciprocal facet map")
    source_points = np.concatenate(
        (geometry.facet_start_m[source_facets, axis], geometry.facet_end_m[source_facets, axis])
    )
    peer_points = np.concatenate(
        (geometry.facet_start_m[peer_facets, axis], geometry.facet_end_m[peer_facets, axis])
    )
    tolerance_m = topology.position_tolerance_m
    if bool((np.abs(source_points - source_coordinate) > tolerance_m).any()) or bool(
        (np.abs(peer_points - peer_coordinate) > tolerance_m).any()
    ):
        raise FieldLocationError(
            "regular periodic facets must lie on opposite field support-box faces"
        )


def _require_periodic_values_match(
    name: str,
    first: FloatArray,
    second: FloatArray,
    all_values: FloatArray,
    relative_tolerance: float,
) -> None:
    if first.shape != second.shape:
        raise FieldLocationError(f"required field {name!r} has unmatched periodic seam nodes")
    component_scale = np.max(np.abs(all_values), axis=tuple(range(all_values.ndim - 1)))
    error = np.abs(first - second)
    tolerance = relative_tolerance * component_scale
    matches = np.where(component_scale == 0.0, error == 0.0, error <= tolerance)
    if not bool(matches.all()):
        raise FieldLocationError(f"required field {name!r} is discontinuous across a periodic seam")


def _certify_field_time_interval(
    field: FieldData,
    time_interval_s: tuple[float, float] | None,
) -> None:
    if field.time_s is None or time_interval_s is None:
        return
    start_s, end_s = time_interval_s
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s < start_s:
        raise ValueError("required-field time interval must be finite and ordered")
    if start_s < float(field.time_s[0]) or end_s > float(field.time_s[-1]):
        raise FieldLocationError(
            f"required field {field.name!r} snapshot range does not cover the run interval"
        )


def _build_field_cell_index(layout: Layout) -> _FieldCellIndex | None:
    """Build one deterministic preorder BVH for P1/Q1 containment candidates."""

    if isinstance(layout, RegularLayout):
        return None
    cell_nodes = layout.nodes_m[layout.connectivity]
    raw_lower = np.min(cell_nodes, axis=1)
    raw_upper = np.max(cell_nodes, axis=1)
    node_spacing = np.max(np.abs(np.spacing(cell_nodes)), axis=(1, 2))
    extent = raw_upper - raw_lower
    diameter_bound = np.hypot(extent[:, 0], extent[:, 1])
    base_padding = np.nextafter(
        FIELD_LOCATION_ULPS * np.maximum(node_spacing, _FLOAT_EPS * diameter_bound),
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
        raise FieldLocationError("field cell index bounds are not finite")
    centroid = 0.5 * raw_lower + 0.5 * raw_upper
    cell_count = int(layout.connectivity.shape[0])
    node_count = _field_cell_index_node_count(cell_count)
    ordered_cell_id = np.empty(cell_count, dtype="<i8")
    node_lower = np.empty((node_count, 2), dtype="<f8")
    node_upper = np.empty((node_count, 2), dtype="<f8")
    node_begin = np.full(node_count, -1, dtype="<i8")
    node_end = np.full(node_count, -1, dtype="<i8")
    node_skip = np.empty(node_count, dtype="<i8")
    next_node = 0
    next_cell = 0
    del cell_nodes, raw_lower, raw_upper, node_spacing, extent, diameter_bound, base_padding

    def append_node(cell_ids: Int64Array) -> None:
        nonlocal next_cell, next_node
        node_id = next_node
        next_node += 1
        selected_lower = np.min(lower[cell_ids], axis=0)
        selected_upper = np.max(upper[cell_ids], axis=0)
        node_lower[node_id] = selected_lower
        node_upper[node_id] = selected_upper
        if cell_ids.size <= FIELD_CELL_BVH_LEAF_SIZE:
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
        raise FieldLocationError("field cell index construction is incomplete")
    arrays = (
        ordered_cell_id,
        node_lower,
        node_upper,
        node_begin,
        node_end,
        node_skip,
    )
    for array in arrays:
        array.setflags(write=False)
    build_transient_nbytes = _FIELD_CELL_INDEX_BUILD_WORK_BYTES_PER_CELL * cell_count
    return _FieldCellIndex(*arrays, build_transient_nbytes)


def _field_cell_index_node_count(cell_count: int) -> int:
    """Return the exact flat-tree size for deterministic half splits."""

    pending = [cell_count]
    result = 0
    while pending:
        size = pending.pop()
        result += 1
        if size > FIELD_CELL_BVH_LEAF_SIZE:
            lower = size // 2
            pending.extend((lower, size - lower))
    return result


def _certify_layout_numerics(layout: Layout) -> None:
    """Reject a shared interpolation basis that particle queries cannot resolve."""

    if isinstance(layout, RegularLayout):
        _certify_regular_axis_numerics(layout.axis0_m, "axis0")
        _certify_regular_axis_numerics(layout.axis1_m, "axis1")
        return
    for cell_id, node_ids in enumerate(layout.connectivity):
        nodes = layout.nodes_m[node_ids]
        if isinstance(layout, P1TriLayout):
            jacobian = np.column_stack((nodes[1] - nodes[0], nodes[2] - nodes[0]))
            _validate_cell_jacobian(
                nodes,
                jacobian,
                f"P1 inverse mapping failed for cell {cell_id}",
            )
        else:
            _validate_q1_cell(nodes, cell_id)


def _certify_regular_axis_numerics(axis: FloatArray, name: str) -> None:
    """Certify every regular cell against the locator's reference uncertainty."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        widths = np.diff(axis)
        span = float(axis[-1] - axis[0])
        coordinate_spacing = max(
            float(np.max(np.abs(np.spacing(axis)))),
            _FLOAT_EPS * span,
        )
        uncertainty = FIELD_LOCATION_ULPS * coordinate_spacing / widths
    if not bool(
        np.isfinite(widths).all()
        and (widths > 0.0).all()
        and np.isfinite(uncertainty).all()
        and (uncertainty <= _MAX_REFERENCE_UNCERTAINTY).all()
    ):
        raise FieldLocationError(
            f"regular {name} contains a cell unresolved at float64 coordinate precision"
        )


def rz_axis_accessible(data: DataBundle, layout: Layout) -> bool:
    """Return whether the certified particle domain includes the RZ axis."""

    if data.coordinate_system != "axisymmetric_rz":
        return False
    if bool((data.geometry.nodes_m[:, 0] == 0.0).any()):
        return True
    return (
        data.geometry.boundary.line2.shape[0] == 0
        and isinstance(layout, RegularLayout)
        and float(layout.axis0_m[0]) == 0.0
    )


def certify_rz_axis_field_regularity(
    coordinate_system: str,
    layout: Layout,
    selected: Mapping[str, FieldData],
    axis_accessible: bool,
    requirements: Mapping[str, RequiredFieldMetadata] | None = None,
) -> None:
    """Require axis-odd primitive components to vanish on an accessible RZ axis."""

    if coordinate_system != "axisymmetric_rz" or not axis_accessible:
        return
    if isinstance(layout, RegularLayout):
        if float(layout.axis0_m[0]) != 0.0:
            raise FieldLocationError("RZ field layout does not include the accessible axis")
        axis_node_ids = np.arange(layout.axis1_m.size, dtype=np.int64)
    else:
        axis_node_ids = np.flatnonzero(layout.nodes_m[:, 0] == 0.0)

    for name, field in selected.items():
        axis_values = (
            field.values[axis_node_ids] if field.time_s is None else field.values[:, axis_node_ids]
        )
        if (
            field.components == ("r", "z")
            and field.stored_basis == "axisymmetric_rz"
            and not bool((axis_values[..., 0] == 0.0).all())
        ):
            raise FieldLocationError(
                f"required RZ vector field {name!r} must have zero radial component on the axis"
            )
        requirement = None if requirements is None else requirements.get(name)
        if (
            requirement is not None
            and requirement.zero_on_rz_axis
            and not bool((axis_values == 0.0).all())
        ):
            raise FieldLocationError(f"required RZ scalar field {name!r} must be zero on the axis")


def _certify_particle_domain_coverage(data: DataBundle, layout: Layout) -> None:
    if not bool((layout.cell_support == 1).all()):
        raise FieldLocationError("required field layout must support every layout cell")
    geometry = data.geometry
    if isinstance(layout, RegularLayout):
        nodes = geometry.nodes_m
        covered = bool(
            (nodes[:, 0] >= layout.axis0_m[0]).all()
            and (nodes[:, 0] <= layout.axis0_m[-1]).all()
            and (nodes[:, 1] >= layout.axis1_m[0]).all()
            and (nodes[:, 1] <= layout.axis1_m[-1]).all()
        )
        if not covered:
            raise FieldLocationError(
                "required regular field layout does not cover every particle-domain node"
            )
        return
    if isinstance(layout, P1TriLayout):
        same_mesh = (
            geometry.tri3 is not None
            and geometry.quad4 is None
            and np.array_equal(layout.nodes_m, geometry.nodes_m)
            and np.array_equal(layout.connectivity, geometry.tri3)
        )
    else:
        same_mesh = (
            geometry.quad4 is not None
            and geometry.tri3 is None
            and np.array_equal(layout.nodes_m, geometry.nodes_m)
            and np.array_equal(layout.connectivity, geometry.quad4)
        )
    if not same_mesh:
        raise FieldLocationError(
            "required unstructured field layout must exactly match the particle-domain mesh"
        )


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
