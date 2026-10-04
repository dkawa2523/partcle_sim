"""Scale-aware exact first-hit localization for line and parabolic paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .geometry import (
    PointClassification,
    PreparedGeometry,
    classify_point,
    count_aabb_candidates,
    fill_aabb_candidates_csr,
    points_inside_volume,
    query_aabb_candidates,
    query_segment_candidates,
)
from .integrators import curved_chord_deviation_bound, curved_chord_deviation_component

type FloatArray = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type UInt8Array = NDArray[np.uint8]

EVENT_ALGORITHM_REVISION = "line_quadratic_rk4_axis_first_hit_v16"
_FLOAT64_EPS = np.finfo(np.float64).eps
_MAX_FLOAT64_INTEGER = int(np.finfo(np.float64).max)
_MAX_INT64 = int(np.iinfo(np.int64).max)

EXACT_PATH_LINEAR = np.uint8(1)
EXACT_PATH_QUADRATIC = np.uint8(2)

EXACT_STATUS_CLEAR = np.uint8(1)
EXACT_STATUS_WALL = np.uint8(2)
EXACT_STATUS_AXIS = np.uint8(3)
EXACT_STATUS_FAILURE = np.uint8(4)

EXACT_FAILURE_NONE = np.uint8(0)
EXACT_FAILURE_INDETERMINATE_EVENT = np.uint8(1)

CURVED_STATUS_CLEAR = np.uint8(1)
CURVED_STATUS_SPLIT = np.uint8(2)
CURVED_STATUS_WALL = np.uint8(3)
CURVED_STATUS_AXIS = np.uint8(4)
CURVED_STATUS_FAILURE = np.uint8(5)

CURVED_FAILURE_INDETERMINATE_EVENT = np.uint8(1)

SURFACE_STATE_PENDING = np.uint8(0)
SURFACE_STATE_RESOLVED = np.uint8(1)
SURFACE_STATE_DEPARTURE = np.uint8(2)

SURFACE_ACTION_RESOLVED = np.uint8(1)
SURFACE_ACTION_CURVED_DEPARTURE = np.uint8(2)
SURFACE_ACTION_EXACT_DEPARTURE = np.uint8(3)
SURFACE_ACTION_RESPONSE_VELOCITY = np.uint8(4)
SURFACE_ACTION_RESPONSE_ACCELERATION = np.uint8(5)

SURFACE_STATUS_OK = np.uint8(0)
SURFACE_STATUS_INDETERMINATE_DIRECTION = np.uint8(1)
SURFACE_STATUS_INVALID_BUDGET = np.uint8(2)

_SURFACE_DIRECTION_INDETERMINATE = np.uint8(0)
_SURFACE_DIRECTION_VELOCITY_INWARD = np.uint8(1)
_SURFACE_DIRECTION_VELOCITY_OUTWARD = np.uint8(2)
_SURFACE_DIRECTION_ACCELERATION_INWARD = np.uint8(3)
_SURFACE_DIRECTION_ACCELERATION_OUTWARD = np.uint8(4)


class EventLocationError(RuntimeError):
    """A boundary event cannot be certified at float64 precision."""


class ExactCandidateCapacityError(EventLocationError):
    """Exact-event broad candidates exceed caller-owned memory capacity."""

    def __init__(
        self,
        required_count: int,
        capacity: int,
        oversized_row: int | None,
    ) -> None:
        self.required_count = required_count
        self.capacity = capacity
        self.oversized_row = oversized_row
        row_detail = ""
        if oversized_row is not None:
            row_detail = f"; row {oversized_row} alone exceeds capacity"
        super().__init__(
            f"exact-event candidates require {required_count} facet IDs, "
            f"capacity is {capacity}{row_detail}"
        )


class CurvedCandidateCapacityError(EventLocationError):
    """Curved-event broad candidates exceed caller-owned memory capacity."""

    def __init__(
        self,
        required_count: int,
        capacity: int,
        oversized_row: int | None,
    ) -> None:
        self.required_count = required_count
        self.capacity = capacity
        self.oversized_row = oversized_row
        row_detail = ""
        if oversized_row is not None:
            row_detail = f"; row {oversized_row} alone exceeds capacity"
        super().__init__(
            f"curved-event candidates require {required_count} facet IDs, "
            f"capacity is {capacity}{row_detail}"
        )


@dataclass(frozen=True, slots=True)
class EventBudget:
    """Resolved physical position and time budgets for one facet."""

    position_m: float
    time_s: float


@dataclass(frozen=True, slots=True)
class BoundaryHit:
    """The earliest geometric hit, before any boundary law is selected."""

    time_s: float
    position_m: FloatArray
    facet_id: int
    candidate_facet_ids: tuple[int, ...]
    normal: FloatArray
    position_budget_m: float
    time_budget_s: float
    localization_residual_m: float


@dataclass(frozen=True, slots=True)
class AxisHit:
    """A certified crossing of the RZ coordinate seam at ``r = 0``."""

    time_s: float
    position_budget_m: float
    time_budget_s: float
    localization_residual_m: float


@dataclass(frozen=True, slots=True)
class ExactEventBatch:
    """Columnar first-event result for independent exact-path rows."""

    status: UInt8Array
    failure_reason: UInt8Array
    time_s: FloatArray
    position_m: FloatArray
    primary_facet_id: Int64Array
    normal: FloatArray
    position_budget_m: FloatArray
    time_budget_s: FloatArray
    localization_residual_m: FloatArray
    candidate_offsets: Int64Array
    candidate_facet_ids: Int64Array


@dataclass(frozen=True, slots=True)
class CurvedEventBatch:
    """Columnar first-event result for independent curved-path rows."""

    status: UInt8Array
    failure_reason: UInt8Array
    start_contact_departure_certified: NDArray[np.bool_]
    time_s: FloatArray
    position_m: FloatArray
    primary_facet_id: Int64Array
    normal: FloatArray
    position_budget_m: FloatArray
    time_budget_s: FloatArray
    localization_residual_m: FloatArray
    candidate_offsets: Int64Array
    candidate_facet_ids: Int64Array


@dataclass(frozen=True, slots=True)
class SurfaceReleaseBatch:
    """Compiled initial-contact decisions aligned with release rows."""

    action: UInt8Array
    status: UInt8Array
    departure_facet_id: Int64Array
    position_budget_m: FloatArray
    time_budget_s: FloatArray


@dataclass(frozen=True, slots=True)
class _PreparedExactPathBatch:
    """Numeric exact-path columns shared by sizing and localization."""

    path_kind: UInt8Array
    start_position_m: FloatArray
    velocity_m_s: FloatArray
    start_time_s: FloatArray
    target_time_s: FloatArray
    departing_facet_id: Int64Array
    query_lower_m: FloatArray
    query_upper_m: FloatArray
    linear_displacement_m: FloatArray
    quadratic_displacement_m: FloatArray
    path_position_bound_m: FloatArray
    speed_m_s: FloatArray
    parameter_speed_m: FloatArray
    ready: NDArray[np.bool_]
    failure_reason: UInt8Array


@dataclass(frozen=True, slots=True)
class _PreparedCurvedPathBatch:
    """Numeric curved-path columns shared by sizing and localization."""

    start_position_m: FloatArray
    start_velocity_m_s: FloatArray
    end_position_m: FloatArray
    end_velocity_m_s: FloatArray
    position_lower_m: FloatArray
    position_upper_m: FloatArray
    velocity_lower_m_s: FloatArray
    velocity_upper_m_s: FloatArray
    start_time_s: FloatArray
    target_time_s: FloatArray
    root_interval_s: FloatArray
    certify_start_contact_departure: NDArray[np.bool_]
    chord_deviation_m: FloatArray
    chord_deviation_valid: NDArray[np.bool_]
    query_lower_m: FloatArray
    query_upper_m: FloatArray
    speed_upper_m_s: FloatArray
    ready: NDArray[np.bool_]
    failure_reason: UInt8Array


@dataclass(frozen=True, slots=True)
class Rk4PieceDecision:
    """Conservative boundary verdict for one sequential RK4 proposal piece."""

    kind: Literal["clear", "split", "hit"]
    hit: BoundaryHit | None
    candidate_facet_count: int
    start_contact_departure_certified: bool = False


@dataclass(frozen=True, slots=True)
class Rk4AxisDecision:
    """Conservative RZ-axis verdict for one sequential RK4 proposal piece."""

    kind: Literal["clear", "split", "hit"]
    hit: AxisHit | None


@dataclass(frozen=True, slots=True)
class _Rk4Piece:
    start_m: FloatArray
    end_m: FloatArray
    lower_m: FloatArray
    upper_m: FloatArray
    velocity_lower_m_s: FloatArray
    velocity_upper_m_s: FloatArray
    start_time_s: float
    end_time_s: float
    interval_s: float
    root_interval_s: float
    speed_upper_m_s: float
    chord_deviation_m: FloatArray | None


@dataclass(frozen=True, slots=True)
class _FacetHit:
    facet_id: int
    time_s: float
    position_m: FloatArray
    budget: EventBudget
    residual_m: float


@dataclass(frozen=True, slots=True)
class _QuadraticPath:
    linear_displacement_m: FloatArray
    quadratic_displacement_m: FloatArray
    end_position_m: FloatArray
    speed_m_s: float
    parameter_speed_m: float
    position_bound_m: FloatArray
    chord_deviation_m: float


def resolve_event_budget(
    *,
    facet_length_m: float,
    geometry_bbox_diagonal_m: float,
    position_m: FloatArray,
    speed_m_s: float,
    interval_s: float,
    time_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> EventBudget:
    """Resolve the single documented scale-aware event tolerance."""

    position, roundoff_factor = _validate_event_budget_inputs(
        facet_length_m=facet_length_m,
        geometry_bbox_diagonal_m=geometry_bbox_diagonal_m,
        position_m=position_m,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )

    position_norm = math.hypot(float(position[0]), float(position[1]))
    roundoff_position = roundoff_factor * max(
        geometry_bbox_diagonal_m, position_norm, facet_length_m
    )
    position_ulp_floor = float(roundoff_ulps) * max(
        math.ulp(float(position[0])),
        math.ulp(float(position[1])),
        math.ulp(geometry_bbox_diagonal_m),
        math.ulp(facet_length_m),
    )
    position_budget = math.nextafter(
        geometry_rtol * facet_length_m + max(roundoff_position, position_ulp_floor),
        math.inf,
    )
    effective_speed = max(speed_m_s, facet_length_m / interval_s)
    time_ulp_floor = float(roundoff_ulps) * max(
        math.ulp(time_s),
        math.ulp(interval_s),
    )
    time_budget = max(
        position_budget / effective_speed,
        roundoff_factor * max(abs(time_s), interval_s),
        time_ulp_floor,
    )
    if (
        not math.isfinite(position_budget)
        or not math.isfinite(time_budget)
        or position_budget <= 0.0
        or time_budget <= 0.0
    ):
        raise EventLocationError("resolved event budget is not finite and positive")
    return EventBudget(position_budget, time_budget)


def _validate_event_budget_inputs(
    *,
    facet_length_m: float,
    geometry_bbox_diagonal_m: float,
    position_m: FloatArray,
    speed_m_s: float,
    interval_s: float,
    time_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[FloatArray, float]:
    """Validate event-budget primitives and return the float64 roundoff factor."""

    position = _finite_point(position_m, "position_m")
    scalar_values = (
        facet_length_m,
        geometry_bbox_diagonal_m,
        speed_m_s,
        interval_s,
        time_s,
        geometry_rtol,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("event-budget inputs must be finite")
    if facet_length_m <= 0.0 or geometry_bbox_diagonal_m <= 0.0 or interval_s <= 0.0:
        raise ValueError("event-budget length scales and interval_s must be positive")
    if speed_m_s < 0.0 or geometry_rtol <= 0.0 or geometry_rtol >= 1.0:
        raise ValueError("event-budget speed and relative tolerance are invalid")
    if isinstance(roundoff_ulps, bool) or not isinstance(roundoff_ulps, int):
        raise ValueError("roundoff_ulps must be an integer")
    if roundoff_ulps <= 0:
        raise ValueError("roundoff_ulps must be positive")
    if roundoff_ulps > _MAX_FLOAT64_INTEGER:
        raise ValueError("roundoff_ulps is too large for float64")
    return position, float(roundoff_ulps) * _FLOAT64_EPS


def classify_surface_release_batch(
    geometry: PreparedGeometry,
    release_state: UInt8Array,
    source_facet_id: Int64Array,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray | None,
    start_time_s: FloatArray,
    *,
    interval_s: float,
    curved_event_path: bool,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> SurfaceReleaseBatch:
    """Classify independent initial wall contacts without scalar retries."""

    states = np.asarray(release_state, dtype=np.uint8)
    facets = np.asarray(source_facet_id, dtype=np.int64)
    positions = np.asarray(position_m, dtype=np.float64)
    velocities = np.asarray(velocity_m_s, dtype=np.float64)
    times = np.asarray(start_time_s, dtype=np.float64)
    row_count = states.size
    actual_shapes = (
        states.shape,
        facets.shape,
        positions.shape,
        velocities.shape,
        times.shape,
    )
    expected_shapes = (
        (row_count,),
        (row_count,),
        (row_count, 2),
        (row_count, 2),
        (row_count,),
    )
    if actual_shapes != expected_shapes:
        raise ValueError("surface-release columns must align by row")
    if bool(
        (
            (states != SURFACE_STATE_PENDING)
            & (states != SURFACE_STATE_RESOLVED)
            & (states != SURFACE_STATE_DEPARTURE)
        ).any()
    ):
        raise ValueError("surface-release state contains an unknown code")
    if bool(((facets < -1) | (facets >= geometry.facet_count)).any()):
        raise ValueError("surface source facet ID is outside the prepared geometry")
    if not bool(np.isfinite(positions).all() and np.isfinite(velocities).all()):
        raise ValueError("surface-release position and velocity must be finite")
    if not bool(np.isfinite(times).all()) or not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("surface-release times must be finite with a positive interval")
    _validate_exact_tolerances(geometry, geometry_rtol, roundoff_ulps)
    if acceleration_m_s2 is None:
        acceleration = np.empty((0, 2), dtype=np.float64)
        has_acceleration = False
    else:
        acceleration = np.asarray(acceleration_m_s2, dtype=np.float64)
        if acceleration.shape != (row_count, 2) or not bool(np.isfinite(acceleration).all()):
            raise ValueError("surface-release acceleration must be finite and row-aligned")
        has_acceleration = True

    action = np.full(row_count, SURFACE_ACTION_RESOLVED, dtype=np.uint8)
    status = np.full(row_count, SURFACE_STATUS_OK, dtype=np.uint8)
    departure = np.full(row_count, -1, dtype=np.int64)
    position_budget = np.zeros(row_count, dtype=np.float64)
    time_budget = np.zeros(row_count, dtype=np.float64)
    _classify_surface_release_kernel(
        states,
        facets,
        positions,
        velocities,
        acceleration,
        has_acceleration,
        times,
        interval_s,
        curved_event_path,
        geometry.bbox_diagonal_m,
        geometry.facet_normal,
        geometry.facet_length_m,
        geometry_rtol,
        roundoff_ulps,
        action,
        status,
        departure,
        position_budget,
        time_budget,
    )
    return SurfaceReleaseBatch(
        action,
        status,
        departure,
        position_budget,
        time_budget,
    )


def locate_ballistic_first_hit(
    geometry: PreparedGeometry,
    start_position_m: FloatArray,
    velocity_m_s: FloatArray,
    *,
    start_time_s: float,
    end_time_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> BoundaryHit | None:
    """Return the earliest positive hit of a constant-velocity line segment."""

    start = _finite_point(start_position_m, "start_position_m")
    velocity = _finite_point(velocity_m_s, "velocity_m_s")
    if not math.isfinite(start_time_s) or not math.isfinite(end_time_s):
        raise ValueError("ballistic event times must be finite")
    interval_s = end_time_s - start_time_s
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("end_time_s must be greater than start_time_s")
    speed = math.hypot(float(velocity[0]), float(velocity[1]))
    if not math.isfinite(speed):
        raise ValueError("velocity magnitude is not finite")
    if speed == 0.0 or geometry.facet_count == 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = interval_s * velocity
        end = start + displacement
    if not bool(np.isfinite(displacement).all() and np.isfinite(end).all()):
        raise EventLocationError("ballistic path exceeds the finite float64 range")

    padding = _conservative_path_padding(
        geometry,
        start,
        end,
        speed_m_s=speed,
        interval_s=interval_s,
        time_s=start_time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    facet_ids = query_segment_candidates(geometry, start, end, padding_m=padding)
    hits: list[_FacetHit] = []
    for facet_id_value in facet_ids:
        facet_id = int(facet_id_value)
        hit = _intersect_facet(
            geometry,
            facet_id,
            start,
            displacement,
            speed,
            start_time_s,
            interval_s,
            geometry_rtol,
            roundoff_ulps,
        )
        if hit is not None:
            hits.append(hit)
    if not hits:
        return None
    return _select_earliest(geometry, hits)


def locate_constant_acceleration_first_hit(
    geometry: PreparedGeometry,
    start_position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    *,
    start_time_s: float,
    end_time_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
    certified_departing_facet_id: int | None = None,
) -> BoundaryHit | None:
    """Return the first hit of an exact Cartesian constant-acceleration path.

    ``certified_departing_facet_id`` may identify one source facet while
    velocity and constant acceleration are both tangent-or-inward.  The
    monotone one-sided certificate is checked on every interval before that
    facet is omitted; all other facets retain the ordinary first-hit search.
    """

    start = _finite_point(start_position_m, "start_position_m")
    velocity = _finite_point(velocity_m_s, "velocity_m_s")
    acceleration = _finite_point(acceleration_m_s2, "acceleration_m_s2")
    if not math.isfinite(start_time_s) or not math.isfinite(end_time_s):
        raise ValueError("constant-acceleration event times must be finite")
    interval_s = end_time_s - start_time_s
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("end_time_s must be greater than start_time_s")
    if geometry.coordinate_system != "cartesian_xy":
        raise ValueError("constant-acceleration events require cartesian_xy geometry")
    if bool(np.equal(acceleration, 0.0).all()):
        if certified_departing_facet_id is not None:
            raise ValueError("a certified quadratic departure requires nonzero acceleration")
        return locate_ballistic_first_hit(
            geometry,
            start,
            velocity,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            geometry_rtol=geometry_rtol,
            roundoff_ulps=roundoff_ulps,
        )
    if geometry.facet_count == 0:
        return None

    path = _prepare_quadratic_path(start, velocity, acceleration, interval_s)
    if path.parameter_speed_m == 0.0:
        return None
    _require_certified_quadratic_departure(
        geometry,
        certified_departing_facet_id,
        start,
        velocity,
        acceleration,
        path.speed_m_s,
        start_time_s,
        interval_s,
        geometry_rtol,
        roundoff_ulps,
    )
    budget_padding = _quadratic_path_padding(
        geometry,
        path.position_bound_m,
        path.speed_m_s,
        start_time_s,
        interval_s,
        geometry_rtol,
        roundoff_ulps,
    )
    padding = math.fsum((budget_padding, path.chord_deviation_m))
    if not math.isfinite(padding):
        raise EventLocationError("constant-acceleration broad-phase padding is not finite")

    facet_ids = query_segment_candidates(geometry, start, path.end_position_m, padding_m=padding)
    hits: list[_FacetHit] = []
    for facet_id_value in facet_ids:
        if int(facet_id_value) == certified_departing_facet_id:
            continue
        hit = _intersect_quadratic_facet(
            geometry,
            int(facet_id_value),
            start,
            path.linear_displacement_m,
            path.quadratic_displacement_m,
            path.position_bound_m,
            path.speed_m_s,
            path.parameter_speed_m,
            start_time_s,
            interval_s,
            geometry_rtol,
            roundoff_ulps,
        )
        if hit is not None:
            hits.append(hit)
    if not hits:
        return None
    return _select_earliest(geometry, hits)


def count_exact_event_candidates(
    geometry: PreparedGeometry,
    path_kind: UInt8Array,
    start_position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    *,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    certified_departing_facet_id: Int64Array | None = None,
) -> Int64Array:
    """Return broad-phase facet counts for memory-plan row partitioning.

    This sizing pass is not an additional scientific candidate query.  An
    engine may use the stable row counts to split a slab by prefix, then call
    :func:`locate_exact_first_event_batch` for each bounded row interval.
    """

    prepared = _prepare_exact_path_batch(
        geometry,
        path_kind,
        start_position_m,
        velocity_m_s,
        acceleration_m_s2,
        start_time_s,
        target_time_s,
        geometry_rtol,
        roundoff_ulps,
        certified_departing_facet_id,
    )
    return count_aabb_candidates(
        geometry,
        prepared.query_lower_m,
        prepared.query_upper_m,
    )


def locate_exact_first_event_batch(
    geometry: PreparedGeometry,
    path_kind: UInt8Array,
    start_position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    *,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    candidate_capacity: int,
    certified_departing_facet_id: Int64Array | None = None,
) -> ExactEventBatch:
    """Locate exact first events without exceeding broad-candidate capacity.

    Invalid shared inputs raise before entering compiled code.  A numerical
    ambiguity local to one row is represented by ``EXACT_STATUS_FAILURE`` so
    another row never triggers a scalar batch retry.  Candidate storage is not
    allocated unless the complete row batch fits ``candidate_capacity``.
    """

    _validate_exact_candidate_capacity(candidate_capacity)
    prepared = _prepare_exact_path_batch(
        geometry,
        path_kind,
        start_position_m,
        velocity_m_s,
        acceleration_m_s2,
        start_time_s,
        target_time_s,
        geometry_rtol,
        roundoff_ulps,
        certified_departing_facet_id,
    )
    broad_counts = count_aabb_candidates(
        geometry,
        prepared.query_lower_m,
        prepared.query_upper_m,
    )
    broad_candidate_count = _require_exact_candidate_capacity(
        broad_counts,
        candidate_capacity,
    )
    row_count = prepared.path_kind.size
    broad_offsets = np.empty(row_count + 1, dtype=np.int64)
    broad_offsets[0] = 0
    np.cumsum(broad_counts, out=broad_offsets[1:])
    broad_candidates = np.empty(broad_candidate_count, dtype=np.int64)
    fill_aabb_candidates_csr(
        geometry,
        prepared.query_lower_m,
        prepared.query_upper_m,
        broad_offsets,
        broad_candidates,
    )
    kinds = prepared.path_kind
    position = prepared.start_position_m
    velocity = prepared.velocity_m_s
    start_time = prepared.start_time_s
    target_time = prepared.target_time_s
    departing = prepared.departing_facet_id
    linear_displacement = prepared.linear_displacement_m
    quadratic_displacement = prepared.quadratic_displacement_m
    path_position_bound = prepared.path_position_bound_m
    speed = prepared.speed_m_s
    parameter_speed = prepared.parameter_speed_m
    ready = prepared.ready
    failure_reason = prepared.failure_reason
    status = np.full(row_count, EXACT_STATUS_CLEAR, dtype=np.uint8)
    event_time = np.full(row_count, np.nan, dtype=np.float64)
    hit_position = np.full((row_count, 2), np.nan, dtype=np.float64)
    primary_facet = np.full(row_count, -1, dtype=np.int64)
    normal = np.zeros((row_count, 2), dtype=np.float64)
    position_budget = np.zeros(row_count, dtype=np.float64)
    time_budget = np.zeros(row_count, dtype=np.float64)
    residual = np.zeros(row_count, dtype=np.float64)
    first_position_budget = np.zeros(row_count, dtype=np.float64)
    first_time_budget = np.zeros(row_count, dtype=np.float64)
    simultaneous_count = np.zeros(row_count, dtype=np.int64)
    _locate_exact_events_kernel(
        geometry.coordinate_system == "axisymmetric_rz",
        kinds,
        position,
        velocity,
        start_time,
        target_time,
        departing,
        linear_displacement,
        quadratic_displacement,
        path_position_bound,
        speed,
        parameter_speed,
        ready,
        broad_offsets,
        broad_candidates,
        geometry.bbox_diagonal_m,
        geometry.facet_start_m,
        geometry.facet_end_m,
        geometry.facet_normal,
        geometry.facet_length_m,
        geometry.facet_node_ids,
        geometry_rtol,
        roundoff_ulps,
        status,
        failure_reason,
        event_time,
        hit_position,
        primary_facet,
        normal,
        position_budget,
        time_budget,
        residual,
        first_position_budget,
        first_time_budget,
        simultaneous_count,
    )
    candidate_offsets = np.empty(row_count + 1, dtype=np.int64)
    candidate_offsets[0] = 0
    np.cumsum(simultaneous_count, out=candidate_offsets[1:])
    candidates = np.full(int(candidate_offsets[-1]), -1, dtype=np.int64)
    _fill_exact_event_candidates_kernel(
        kinds,
        position,
        start_time,
        target_time,
        departing,
        linear_displacement,
        quadratic_displacement,
        path_position_bound,
        speed,
        parameter_speed,
        broad_offsets,
        broad_candidates,
        geometry.bbox_diagonal_m,
        geometry.facet_start_m,
        geometry.facet_end_m,
        geometry.facet_length_m,
        geometry_rtol,
        roundoff_ulps,
        status,
        event_time,
        hit_position,
        first_position_budget,
        first_time_budget,
        candidate_offsets,
        candidates,
    )
    if bool((candidates < 0).any()):
        raise EventLocationError("compiled exact-event candidate packing is incomplete")
    return ExactEventBatch(
        status,
        failure_reason,
        event_time,
        hit_position,
        primary_facet,
        normal,
        position_budget,
        time_budget,
        residual,
        candidate_offsets,
        candidates,
    )


def count_curved_event_candidates(
    geometry: PreparedGeometry,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    *,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    certify_start_contact_departure: NDArray[np.bool_] | None = None,
    chord_deviation_bound_m: FloatArray | None = None,
) -> Int64Array:
    """Return curved broad-phase counts for stable capacity partitioning.

    The count/fill pair is one logical event query.  Callers may partition the
    returned rows by stable prefix before invoking
    :func:`locate_curved_first_event_batch` with caller-owned capacity.
    """

    prepared = _prepare_curved_path_batch(
        geometry,
        start_position_m,
        start_velocity_m_s,
        end_position_m,
        end_velocity_m_s,
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        start_time_s,
        target_time_s,
        root_interval_s,
        geometry_rtol,
        roundoff_ulps,
        certify_start_contact_departure,
        chord_deviation_bound_m,
    )
    return count_aabb_candidates(
        geometry,
        prepared.query_lower_m,
        prepared.query_upper_m,
    )


def locate_curved_first_event_batch(
    geometry: PreparedGeometry,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    *,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    candidate_capacity: int,
    certify_start_contact_departure: NDArray[np.bool_] | None = None,
    chord_deviation_bound_m: FloatArray | None = None,
    certify_monotone_approach: bool = False,
    use_position_controls: NDArray[np.bool_] | None = None,
    position_control_origin_m: FloatArray | None = None,
    relative_position_control_lower_m: FloatArray | None = None,
    relative_position_control_upper_m: FloatArray | None = None,
) -> CurvedEventBatch:
    """Classify curved first events in independent compiled rows.

    Shared shape/configuration defects raise before compiled execution.  A
    representational ambiguity local to one path is returned as a failure
    status.  Broad candidate storage is allocated only after the whole row
    batch fits ``candidate_capacity``.
    """

    _validate_exact_candidate_capacity(candidate_capacity)
    if not isinstance(certify_monotone_approach, (bool, np.bool_)):
        raise ValueError("certify_monotone_approach must be boolean")
    prepared = _prepare_curved_path_batch(
        geometry,
        start_position_m,
        start_velocity_m_s,
        end_position_m,
        end_velocity_m_s,
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        start_time_s,
        target_time_s,
        root_interval_s,
        geometry_rtol,
        roundoff_ulps,
        certify_start_contact_departure,
        chord_deviation_bound_m,
    )
    (
        position_control_rows,
        position_control_origin,
        relative_position_control_lower,
        relative_position_control_upper,
    ) = _curved_position_control_bounds(
        position_control_origin_m,
        relative_position_control_lower_m,
        relative_position_control_upper_m,
        use_position_controls,
        prepared.start_time_s.size,
    )
    broad_counts = count_aabb_candidates(
        geometry,
        prepared.query_lower_m,
        prepared.query_upper_m,
    )
    broad_candidate_count = _require_curved_candidate_capacity(
        broad_counts,
        candidate_capacity,
    )
    row_count = prepared.start_time_s.size
    broad_offsets = np.empty(row_count + 1, dtype=np.int64)
    broad_offsets[0] = 0
    np.cumsum(broad_counts, out=broad_offsets[1:])
    broad_candidates = np.empty(broad_candidate_count, dtype=np.int64)
    fill_aabb_candidates_csr(
        geometry,
        prepared.query_lower_m,
        prepared.query_upper_m,
        broad_offsets,
        broad_candidates,
    )
    volume_inside = points_inside_volume(geometry, prepared.end_position_m)
    status = np.full(row_count, CURVED_STATUS_CLEAR, dtype=np.uint8)
    failure_reason = prepared.failure_reason.copy()
    departure_certified = np.zeros(row_count, dtype=np.bool_)
    event_time = np.full(row_count, np.nan, dtype=np.float64)
    hit_position = np.full((row_count, 2), np.nan, dtype=np.float64)
    primary_facet = np.full(row_count, -1, dtype=np.int64)
    normal = np.zeros((row_count, 2), dtype=np.float64)
    position_budget = np.zeros(row_count, dtype=np.float64)
    time_budget = np.zeros(row_count, dtype=np.float64)
    residual = np.zeros(row_count, dtype=np.float64)
    event_candidate_count = np.zeros(row_count, dtype=np.int64)
    _locate_curved_events_kernel(
        geometry.coordinate_system == "axisymmetric_rz",
        prepared.start_position_m,
        prepared.start_velocity_m_s,
        prepared.end_position_m,
        prepared.end_velocity_m_s,
        prepared.position_lower_m,
        prepared.position_upper_m,
        prepared.velocity_lower_m_s,
        prepared.velocity_upper_m_s,
        prepared.start_time_s,
        prepared.target_time_s,
        prepared.root_interval_s,
        prepared.certify_start_contact_departure,
        bool(certify_monotone_approach),
        position_control_rows,
        position_control_origin,
        relative_position_control_lower,
        relative_position_control_upper,
        prepared.chord_deviation_m,
        prepared.chord_deviation_valid,
        prepared.speed_upper_m_s,
        prepared.ready,
        volume_inside,
        broad_offsets,
        broad_candidates,
        geometry.bbox_diagonal_m,
        geometry.facet_start_m,
        geometry.facet_end_m,
        geometry.facet_normal,
        geometry.facet_length_m,
        geometry_rtol,
        roundoff_ulps,
        status,
        failure_reason,
        departure_certified,
        event_time,
        hit_position,
        primary_facet,
        normal,
        position_budget,
        time_budget,
        residual,
        event_candidate_count,
    )
    candidate_offsets = np.empty(row_count + 1, dtype=np.int64)
    candidate_offsets[0] = 0
    np.cumsum(event_candidate_count, out=candidate_offsets[1:])
    event_candidates = np.empty(int(candidate_offsets[-1]), dtype=np.int64)
    _fill_curved_event_candidates_kernel(
        status,
        primary_facet,
        candidate_offsets,
        event_candidates,
    )
    return CurvedEventBatch(
        status,
        failure_reason,
        departure_certified,
        event_time,
        hit_position,
        primary_facet,
        normal,
        position_budget,
        time_budget,
        residual,
        candidate_offsets,
        event_candidates,
    )


def _prepare_curved_path_batch(
    geometry: PreparedGeometry,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    certify_start_contact_departure: NDArray[np.bool_] | None,
    chord_deviation_bound_m: FloatArray | None,
) -> _PreparedCurvedPathBatch:
    arrays = _validate_curved_batch_inputs(
        geometry,
        start_position_m,
        start_velocity_m_s,
        end_position_m,
        end_velocity_m_s,
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        start_time_s,
        target_time_s,
        root_interval_s,
        geometry_rtol,
        roundoff_ulps,
        certify_start_contact_departure,
        chord_deviation_bound_m,
    )
    (
        start_position,
        start_velocity,
        end_position,
        end_velocity,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        start_time,
        target_time,
        root_interval,
        departure,
        supplied_chord,
        use_supplied_chord,
    ) = arrays
    row_count = start_time.size
    chord_deviation = np.empty((row_count, 2), dtype=np.float64)
    chord_valid = np.empty(row_count, dtype=np.bool_)
    query_lower = np.empty((row_count, 2), dtype=np.float64)
    query_upper = np.empty((row_count, 2), dtype=np.float64)
    speed_upper = np.empty(row_count, dtype=np.float64)
    ready = np.empty(row_count, dtype=np.bool_)
    failure_reason = np.zeros(row_count, dtype=np.uint8)
    maximum_facet_length = (
        0.0 if geometry.facet_count == 0 else float(np.max(geometry.facet_length_m))
    )
    _prepare_curved_paths_kernel(
        start_position,
        end_position,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        start_time,
        target_time,
        root_interval,
        supplied_chord,
        use_supplied_chord,
        geometry.bbox_diagonal_m,
        maximum_facet_length,
        geometry_rtol,
        roundoff_ulps,
        chord_deviation,
        chord_valid,
        query_lower,
        query_upper,
        speed_upper,
        ready,
        failure_reason,
    )
    return _PreparedCurvedPathBatch(
        start_position,
        start_velocity,
        end_position,
        end_velocity,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        start_time,
        target_time,
        root_interval,
        departure,
        chord_deviation,
        chord_valid,
        query_lower,
        query_upper,
        speed_upper,
        ready,
        failure_reason,
    )


def _validate_curved_batch_inputs(
    geometry: PreparedGeometry,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    certify_start_contact_departure: NDArray[np.bool_] | None,
    chord_deviation_bound_m: FloatArray | None,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    NDArray[np.bool_],
    FloatArray,
    bool,
]:
    start_position = np.asarray(start_position_m, dtype=np.float64)
    start_velocity = np.asarray(start_velocity_m_s, dtype=np.float64)
    end_position = np.asarray(end_position_m, dtype=np.float64)
    end_velocity = np.asarray(end_velocity_m_s, dtype=np.float64)
    position_lower = np.asarray(position_lower_m, dtype=np.float64)
    position_upper = np.asarray(position_upper_m, dtype=np.float64)
    velocity_lower = np.asarray(velocity_lower_m_s, dtype=np.float64)
    velocity_upper = np.asarray(velocity_upper_m_s, dtype=np.float64)
    start_time = np.asarray(start_time_s, dtype=np.float64)
    target_time = np.asarray(target_time_s, dtype=np.float64)
    root_interval = np.asarray(root_interval_s, dtype=np.float64)
    point_arrays = (
        start_position,
        start_velocity,
        end_position,
        end_velocity,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
    )
    row_count = _validate_curved_batch_shapes(
        point_arrays,
        start_time,
        target_time,
        root_interval,
    )
    _validate_curved_batch_values(
        point_arrays,
        start_position,
        end_position,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        start_time,
        target_time,
        root_interval,
    )
    if geometry.coordinate_system == "axisymmetric_rz":
        _validate_curved_rz_values(
            start_position,
            start_velocity,
            end_velocity,
            velocity_lower,
            velocity_upper,
        )
    _validate_exact_tolerances(geometry, geometry_rtol, roundoff_ulps)
    departure = _curved_departure_flags(certify_start_contact_departure, row_count)
    supplied_chord, use_supplied_chord = _curved_chord_bounds(
        chord_deviation_bound_m,
        row_count,
    )
    return (
        start_position,
        start_velocity,
        end_position,
        end_velocity,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        start_time,
        target_time,
        root_interval,
        departure,
        supplied_chord,
        use_supplied_chord,
    )


def _validate_curved_batch_shapes(
    point_arrays: tuple[FloatArray, ...],
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
) -> int:
    row_count = int(start_time_s.size)
    if start_time_s.ndim != 1 or any(value.shape != (row_count, 2) for value in point_arrays):
        raise ValueError("curved states and enclosures must align as [N, 2]")
    if target_time_s.shape != (row_count,) or root_interval_s.shape != (row_count,):
        raise ValueError("curved event times must align with path rows")
    return row_count


def _validate_curved_batch_values(
    point_arrays: tuple[FloatArray, ...],
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
) -> None:
    finite_points = all(bool(np.isfinite(value).all()) for value in point_arrays)
    finite_times = all(
        bool(np.isfinite(value).all()) for value in (start_time_s, target_time_s, root_interval_s)
    )
    if not finite_points or not finite_times:
        raise ValueError("curved path states must be finite with positive intervals")
    if bool((target_time_s <= start_time_s).any()) or bool((root_interval_s <= 0.0).any()):
        raise ValueError("curved path states must be finite with positive intervals")
    if bool((position_lower_m > position_upper_m).any()):
        raise ValueError("curved path enclosure lower bounds exceed upper bounds")
    if bool((velocity_lower_m_s > velocity_upper_m_s).any()):
        raise ValueError("curved velocity enclosure lower bounds exceed upper bounds")
    start_outside = (start_position_m < position_lower_m).any() or (
        start_position_m > position_upper_m
    ).any()
    end_outside = (end_position_m < position_lower_m).any() or (
        end_position_m > position_upper_m
    ).any()
    if bool(start_outside or end_outside):
        raise ValueError("curved path enclosure does not contain its endpoints")


def _validate_curved_rz_values(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    end_velocity_m_s: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
) -> None:
    start_outside = (start_velocity_m_s < velocity_lower_m_s).any() or (
        start_velocity_m_s > velocity_upper_m_s
    ).any()
    end_outside = (end_velocity_m_s < velocity_lower_m_s).any() or (
        end_velocity_m_s > velocity_upper_m_s
    ).any()
    if bool(start_outside or end_outside):
        raise ValueError("curved velocity enclosure does not contain its endpoints")
    if bool((start_position_m[:, 0] < 0.0).any()):
        raise ValueError("resident RZ radius must be nonnegative")


def _curved_departure_flags(
    value: NDArray[np.bool_] | None,
    row_count: int,
) -> NDArray[np.bool_]:
    if value is None:
        return np.zeros(row_count, dtype=np.bool_)
    departure = np.asarray(value, dtype=np.bool_)
    if departure.shape != (row_count,):
        raise ValueError("curved departure flags must align with path rows")
    return departure


def _curved_chord_bounds(
    value: FloatArray | None,
    row_count: int,
) -> tuple[FloatArray, bool]:
    if value is None:
        return np.zeros((row_count, 2), dtype=np.float64), False
    supplied = np.asarray(value, dtype=np.float64)
    if supplied.shape != (row_count, 2):
        raise ValueError("chord_deviation_bound_m must have shape [N, 2]")
    if not bool(np.isfinite(supplied).all()) or bool((supplied < 0.0).any()):
        raise ValueError("chord_deviation_bound_m must be finite and nonnegative")
    return supplied, True


def _curved_position_control_bounds(
    origin_m: FloatArray | None,
    relative_lower_m: FloatArray | None,
    relative_upper_m: FloatArray | None,
    use_controls: NDArray[np.bool_] | None,
    row_count: int,
) -> tuple[NDArray[np.bool_], FloatArray, FloatArray, FloatArray]:
    values = (origin_m, relative_lower_m, relative_upper_m)
    if all(value is None for value in values):
        if use_controls is not None:
            raise ValueError("position-control rows require position-control bounds")
        return (
            np.zeros(row_count, dtype=np.bool_),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 4, 2), dtype=np.float64),
            np.empty((0, 4, 2), dtype=np.float64),
        )
    if any(value is None for value in values):
        raise ValueError("curved position-control bounds must be supplied together")
    if use_controls is None:
        controls_enabled = np.ones(row_count, dtype=np.bool_)
    else:
        controls_enabled = np.asarray(use_controls, dtype=np.bool_)
        if controls_enabled.shape != (row_count,):
            raise ValueError("use_position_controls must have shape [N]")
    origin = np.asarray(origin_m, dtype=np.float64)
    lower = np.asarray(relative_lower_m, dtype=np.float64)
    upper = np.asarray(relative_upper_m, dtype=np.float64)
    if origin.shape != (row_count, 2):
        raise ValueError("position_control_origin_m must have shape [N, 2]")
    if lower.shape != (row_count, 4, 2) or upper.shape != (row_count, 4, 2):
        raise ValueError("relative position-control bounds must have shape [N, 4, 2]")
    if not bool(
        np.isfinite(origin).all() and np.isfinite(lower).all() and np.isfinite(upper).all()
    ):
        raise ValueError("curved position-control bounds must be finite")
    if bool((lower > upper).any()):
        raise ValueError("curved position-control lower bounds exceed upper bounds")
    return controls_enabled, origin, lower, upper


def _require_curved_candidate_capacity(counts: Int64Array, candidate_capacity: int) -> int:
    if bool((counts < 0).any()):
        raise EventLocationError("curved-event candidate counts must be nonnegative")
    maximum = 0 if not counts.size else int(counts.max())
    int64_sum_is_safe = not counts.size or maximum <= _MAX_INT64 // counts.size
    required_count = (
        int(np.sum(counts, dtype=np.int64))
        if int64_sum_is_safe
        else sum(int(count) for count in counts)
    )
    if required_count <= candidate_capacity:
        return required_count
    oversized = np.flatnonzero(counts > candidate_capacity)
    oversized_row = None if not oversized.size else int(oversized[0])
    raise CurvedCandidateCapacityError(required_count, candidate_capacity, oversized_row)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_upper_product(left: float, right: float) -> float:
    return np.nextafter(left * right, np.inf)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_upper_sum(left: float, right: float) -> float:
    return np.nextafter(left + right, np.inf)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_upper_quotient(value: float, divisor: float) -> float:
    return np.nextafter(value / divisor, np.inf)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_roundoff_margin(scale: float, roundoff_ulps: int) -> float:
    return _curved_upper_product(
        float(roundoff_ulps) * _FLOAT64_EPS,
        max(scale, _FLOAT64_EPS),
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_two_diff(left: float, right: float) -> tuple[float, float]:
    difference = left - right
    virtual_right = left - difference
    virtual_left = difference + virtual_right
    residual = (left - virtual_left) + (virtual_right - right)
    return difference, residual


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_two_product(left: float, right: float) -> tuple[float, float]:
    product = left * right
    splitter = 134_217_729.0
    split_left = splitter * left
    left_high = split_left - (split_left - left)
    left_low = left - left_high
    split_right = splitter * right
    right_high = split_right - (split_right - right)
    right_low = right - right_high
    error = (
        ((left_high * right_high - product) + left_high * right_low) + left_low * right_high
    ) + left_low * right_low
    return product, error


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_hypot2(first: float, second: float) -> float:
    """Match CPython's compensated two-component ``math.hypot`` path."""

    maximum = max(abs(first), abs(second))
    if maximum == 0.0:
        return 0.0
    _, exponent = math.frexp(maximum)
    scaled_first = math.ldexp(first, -exponent)
    scaled_second = math.ldexp(second, -exponent)
    first_square, first_error = _curved_two_product(scaled_first, scaled_first)
    second_square, second_error = _curved_two_product(scaled_second, scaled_second)
    square_sum = first_square + second_square
    second_virtual = square_sum - first_square
    sum_error = (first_square - (square_sum - second_virtual)) + (second_square - second_virtual)
    low = sum_error + first_error + second_error
    root = math.sqrt(square_sum + low)
    root_square, root_error = _curved_two_product(root, root)
    difference = square_sum - root_square
    root_residual = difference + (low - root_error)
    corrected = root + root_residual / (2.0 * root)
    return math.ldexp(corrected, exponent)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_event_budget(
    facet_length_m: float,
    geometry_bbox_diagonal_m: float,
    position_x: float,
    position_y: float,
    speed_m_s: float,
    interval_s: float,
    time_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, float, float]:
    roundoff_factor = float(roundoff_ulps) * _FLOAT64_EPS
    position_norm = _curved_hypot2(position_x, position_y)
    roundoff_position = roundoff_factor * max(
        geometry_bbox_diagonal_m,
        position_norm,
        facet_length_m,
    )
    position_ulp_floor = float(roundoff_ulps) * max(
        _exact_ulp(position_x),
        _exact_ulp(position_y),
        _exact_ulp(geometry_bbox_diagonal_m),
        _exact_ulp(facet_length_m),
    )
    position_budget = _curved_upper_sum(
        geometry_rtol * facet_length_m,
        max(roundoff_position, position_ulp_floor),
    )
    effective_speed = max(speed_m_s, facet_length_m / interval_s)
    time_ulp_floor = float(roundoff_ulps) * max(
        _exact_ulp(time_s),
        _exact_ulp(interval_s),
    )
    time_budget = max(
        position_budget / effective_speed,
        roundoff_factor * max(abs(time_s), interval_s),
        time_ulp_floor,
    )
    valid = (
        math.isfinite(position_budget)
        and math.isfinite(time_budget)
        and position_budget > 0.0
        and time_budget > 0.0
    )
    return valid, position_budget, time_budget


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _prepare_curved_paths_kernel(
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    supplied_chord_deviation_m: FloatArray,
    use_supplied_chord_deviation: bool,
    geometry_bbox_diagonal_m: float,
    maximum_facet_length_m: float,
    geometry_rtol: float,
    roundoff_ulps: int,
    chord_deviation_m: FloatArray,
    chord_deviation_valid: NDArray[np.bool_],
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
    speed_upper_m_s: FloatArray,
    ready: NDArray[np.bool_],
    failure_reason: UInt8Array,
) -> None:
    for row in range(start_time_s.size):
        velocity_abs_x = max(abs(velocity_lower_m_s[row, 0]), abs(velocity_upper_m_s[row, 0]))
        velocity_abs_y = max(abs(velocity_lower_m_s[row, 1]), abs(velocity_upper_m_s[row, 1]))
        speed_upper = np.nextafter(_curved_hypot2(velocity_abs_x, velocity_abs_y), np.inf)
        speed_upper_m_s[row] = speed_upper
        interval_s = target_time_s[row] - start_time_s[row]
        chord_valid = math.isfinite(speed_upper)
        for axis in range(2):
            if use_supplied_chord_deviation:
                deviation = supplied_chord_deviation_m[row, axis]
                component_valid = True
            else:
                component_valid, deviation = curved_chord_deviation_component(
                    start_position_m[row, axis],
                    end_position_m[row, axis],
                    velocity_lower_m_s[row, axis],
                    velocity_upper_m_s[row, axis],
                    interval_s,
                )
            chord_deviation_m[row, axis] = deviation
            chord_valid = chord_valid and component_valid
        chord_deviation_valid[row] = chord_valid
        bound_x = max(abs(position_lower_m[row, 0]), abs(position_upper_m[row, 0]))
        bound_y = max(abs(position_lower_m[row, 1]), abs(position_upper_m[row, 1]))
        time_scale = (
            start_time_s[row]
            if abs(start_time_s[row]) >= abs(target_time_s[row])
            else target_time_s[row]
        )
        if maximum_facet_length_m == 0.0:
            valid_budget, padding = True, 0.0
        else:
            valid_budget, padding, _ = _curved_event_budget(
                maximum_facet_length_m,
                geometry_bbox_diagonal_m,
                bound_x,
                bound_y,
                speed_upper,
                root_interval_s[row],
                time_scale,
                geometry_rtol,
                roundoff_ulps,
            )
        lower_x = np.nextafter(position_lower_m[row, 0] - padding, -np.inf)
        lower_y = np.nextafter(position_lower_m[row, 1] - padding, -np.inf)
        upper_x = np.nextafter(position_upper_m[row, 0] + padding, np.inf)
        upper_y = np.nextafter(position_upper_m[row, 1] + padding, np.inf)
        finite_query = (
            math.isfinite(lower_x)
            and math.isfinite(lower_y)
            and math.isfinite(upper_x)
            and math.isfinite(upper_y)
        )
        ready[row] = valid_budget and finite_query and math.isfinite(speed_upper)
        if not ready[row]:
            failure_reason[row] = CURVED_FAILURE_INDETERMINATE_EVENT
            lower_x = position_lower_m[row, 0]
            lower_y = position_lower_m[row, 1]
            upper_x = position_upper_m[row, 0]
            upper_y = position_upper_m[row, 1]
        query_lower_m[row, 0] = lower_x
        query_lower_m[row, 1] = lower_y
        query_upper_m[row, 0] = upper_x
        query_upper_m[row, 1] = upper_y


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_offset_dot_bounds(
    point_x: float,
    point_y: float,
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    roundoff_ulps: int,
) -> tuple[float, float]:
    offset_x, residual_x = _curved_two_diff(point_x, origin_x)
    offset_y, residual_y = _curved_two_diff(point_y, origin_y)
    if not (
        _exact_four_finite(offset_x, offset_y, residual_x, residual_y)
        and math.isfinite(direction_x)
        and math.isfinite(direction_y)
    ):
        return -np.inf, np.inf
    first = offset_x * direction_x
    second = offset_y * direction_y
    residual_first = residual_x * direction_x
    residual_second = residual_y * direction_y
    value = _exact_sum3(
        first,
        second,
        _exact_sum2(residual_first, residual_second),
    )
    scale = _curved_upper_sum(
        _curved_upper_product(
            _curved_upper_sum(abs(offset_x), abs(residual_x)),
            abs(direction_x),
        ),
        _curved_upper_product(
            _curved_upper_sum(abs(offset_y), abs(residual_y)),
            abs(direction_y),
        ),
    )
    if not math.isfinite(value) or not math.isfinite(scale):
        return -np.inf, np.inf
    margin = _curved_roundoff_margin(scale, roundoff_ulps)
    return np.nextafter(value - margin, -np.inf), np.nextafter(value + margin, np.inf)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_distance_lower_bound(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
    roundoff_ulps: int,
) -> float:
    separation_x = first_x - second_x
    separation_y = first_y - second_y
    if not _exact_four_finite(separation_x, separation_y, 0.0, 0.0):
        return -np.inf
    distance = _curved_hypot2(separation_x, separation_y)
    scale = _curved_upper_sum(
        _curved_upper_sum(abs(first_x), abs(second_x)),
        _curved_upper_sum(abs(first_y), abs(second_y)),
    )
    if not math.isfinite(distance) or not math.isfinite(scale):
        return -np.inf
    margin = _curved_roundoff_margin(scale, roundoff_ulps)
    return np.nextafter(distance - margin, -np.inf)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_tube_may_reach_supporting_line(
    row: int,
    facet_id: int,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    speed_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    normal_x = facet_normal[facet_id, 0]
    normal_y = facet_normal[facet_id, 1]
    minimum_x = position_lower_m[row, 0] if normal_x >= 0.0 else position_upper_m[row, 0]
    minimum_y = position_lower_m[row, 1] if normal_y >= 0.0 else position_upper_m[row, 1]
    maximum_x = position_upper_m[row, 0] if normal_x >= 0.0 else position_lower_m[row, 0]
    maximum_y = position_upper_m[row, 1] if normal_y >= 0.0 else position_lower_m[row, 1]
    signed_lower, _ = _curved_offset_dot_bounds(
        minimum_x,
        minimum_y,
        facet_start_m[facet_id, 0],
        facet_start_m[facet_id, 1],
        normal_x,
        normal_y,
        roundoff_ulps,
    )
    _, signed_upper = _curved_offset_dot_bounds(
        maximum_x,
        maximum_y,
        facet_start_m[facet_id, 0],
        facet_start_m[facet_id, 1],
        normal_x,
        normal_y,
        roundoff_ulps,
    )
    bound_x = max(abs(position_lower_m[row, 0]), abs(position_upper_m[row, 0]))
    bound_y = max(abs(position_lower_m[row, 1]), abs(position_upper_m[row, 1]))
    time_scale = (
        start_time_s[row]
        if abs(start_time_s[row]) >= abs(target_time_s[row])
        else target_time_s[row]
    )
    valid, budget, _ = _curved_event_budget(
        facet_length_m[facet_id],
        geometry_bbox_diagonal_m,
        bound_x,
        bound_y,
        speed_upper_m_s[row],
        root_interval_s[row],
        time_scale,
        geometry_rtol,
        roundoff_ulps,
    )
    return valid and signed_lower <= budget and signed_upper >= -budget


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_departure_is_certified(
    row: int,
    facet_id: int,
    start_position_m: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    speed_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    normal_x = facet_normal[facet_id, 0]
    normal_y = facet_normal[facet_id, 1]
    origin_x = facet_start_m[facet_id, 0]
    origin_y = facet_start_m[facet_id, 1]
    valid, budget, _ = _curved_event_budget(
        facet_length_m[facet_id],
        geometry_bbox_diagonal_m,
        start_position_m[row, 0],
        start_position_m[row, 1],
        speed_upper_m_s[row],
        root_interval_s[row],
        start_time_s[row],
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return False
    signed_lower, signed_upper = _curved_offset_dot_bounds(
        start_position_m[row, 0],
        start_position_m[row, 1],
        origin_x,
        origin_y,
        normal_x,
        normal_y,
        roundoff_ulps,
    )
    if signed_lower < -budget or signed_upper > budget:
        return False
    length = facet_length_m[facet_id]
    tangent_x = (facet_end_m[facet_id, 0] - origin_x) / length
    tangent_y = (facet_end_m[facet_id, 1] - origin_y) / length
    start_clearance, _ = _curved_offset_dot_bounds(
        start_position_m[row, 0],
        start_position_m[row, 1],
        origin_x,
        origin_y,
        tangent_x,
        tangent_y,
        roundoff_ulps,
    )
    end_clearance, _ = _curved_offset_dot_bounds(
        start_position_m[row, 0],
        start_position_m[row, 1],
        facet_end_m[facet_id, 0],
        facet_end_m[facet_id, 1],
        -tangent_x,
        -tangent_y,
        roundoff_ulps,
    )
    if min(start_clearance, end_clearance) <= budget:
        return False
    velocity_x = velocity_upper_m_s[row, 0] if normal_x >= 0.0 else velocity_lower_m_s[row, 0]
    velocity_y = velocity_upper_m_s[row, 1] if normal_y >= 0.0 else velocity_lower_m_s[row, 1]
    normal_velocity = _exact_sum2(normal_x * velocity_x, normal_y * velocity_y)
    velocity_scale = _curved_upper_sum(
        _curved_upper_product(
            abs(normal_x),
            max(abs(velocity_lower_m_s[row, 0]), abs(velocity_upper_m_s[row, 0])),
        ),
        _curved_upper_product(
            abs(normal_y),
            max(abs(velocity_lower_m_s[row, 1]), abs(velocity_upper_m_s[row, 1])),
        ),
    )
    margin = _curved_roundoff_margin(velocity_scale, roundoff_ulps)
    normal_velocity_upper = np.nextafter(normal_velocity + margin, np.inf)
    return math.isfinite(normal_velocity_upper) and normal_velocity_upper < 0.0


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_bernstein_controls_are_inside(
    row: int,
    facet_id: int,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    speed_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    position_control_origin_m: FloatArray,
    relative_position_control_lower_m: FloatArray,
    relative_position_control_upper_m: FloatArray,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    """Prove one Bernstein path remains strictly inside a facet line band."""

    bound_x = max(abs(position_lower_m[row, 0]), abs(position_upper_m[row, 0]))
    bound_y = max(abs(position_lower_m[row, 1]), abs(position_upper_m[row, 1]))
    time_scale = (
        start_time_s[row]
        if abs(start_time_s[row]) >= abs(target_time_s[row])
        else target_time_s[row]
    )
    valid, budget, _ = _curved_event_budget(
        facet_length_m[facet_id],
        geometry_bbox_diagonal_m,
        bound_x,
        bound_y,
        speed_upper_m_s[row],
        root_interval_s[row],
        time_scale,
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return False
    normal_x = facet_normal[facet_id, 0]
    normal_y = facet_normal[facet_id, 1]
    _, origin_signed_upper = _curved_offset_dot_bounds(
        position_control_origin_m[row, 0],
        position_control_origin_m[row, 1],
        facet_start_m[facet_id, 0],
        facet_start_m[facet_id, 1],
        normal_x,
        normal_y,
        roundoff_ulps,
    )
    if not math.isfinite(origin_signed_upper):
        return False
    for control in range(4):
        relative_x = (
            relative_position_control_upper_m[row, control, 0]
            if normal_x >= 0.0
            else relative_position_control_lower_m[row, control, 0]
        )
        relative_y = (
            relative_position_control_upper_m[row, control, 1]
            if normal_y >= 0.0
            else relative_position_control_lower_m[row, control, 1]
        )
        _, relative_signed_upper = _curved_offset_dot_bounds(
            relative_x,
            relative_y,
            0.0,
            0.0,
            normal_x,
            normal_y,
            roundoff_ulps,
        )
        signed_upper = _curved_upper_sum(origin_signed_upper, relative_signed_upper)
        if not math.isfinite(signed_upper) or signed_upper >= -budget:
            return False
    return True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_monotone_approach_is_clear(
    row: int,
    facet_id: int,
    end_position_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    speed_upper_m_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    normal_x = facet_normal[facet_id, 0]
    normal_y = facet_normal[facet_id, 1]
    valid, budget, _ = _curved_event_budget(
        facet_length_m[facet_id],
        geometry_bbox_diagonal_m,
        end_position_m[row, 0],
        end_position_m[row, 1],
        speed_upper_m_s[row],
        root_interval_s[row],
        target_time_s[row],
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return False
    _, signed_upper = _curved_offset_dot_bounds(
        end_position_m[row, 0],
        end_position_m[row, 1],
        facet_start_m[facet_id, 0],
        facet_start_m[facet_id, 1],
        normal_x,
        normal_y,
        roundoff_ulps,
    )
    if not math.isfinite(signed_upper) or signed_upper >= -budget:
        return False
    velocity_x = velocity_lower_m_s[row, 0] if normal_x >= 0.0 else velocity_upper_m_s[row, 0]
    velocity_y = velocity_lower_m_s[row, 1] if normal_y >= 0.0 else velocity_upper_m_s[row, 1]
    normal_velocity = _exact_sum2(normal_x * velocity_x, normal_y * velocity_y)
    velocity_scale = _curved_upper_sum(
        _curved_upper_product(
            abs(normal_x),
            max(abs(velocity_lower_m_s[row, 0]), abs(velocity_upper_m_s[row, 0])),
        ),
        _curved_upper_product(
            abs(normal_y),
            max(abs(velocity_lower_m_s[row, 1]), abs(velocity_upper_m_s[row, 1])),
        ),
    )
    margin = _curved_roundoff_margin(velocity_scale, roundoff_ulps)
    normal_velocity_lower = np.nextafter(normal_velocity - margin, -np.inf)
    return math.isfinite(normal_velocity_lower) and normal_velocity_lower > 0.0


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_point_segment_distance(
    point_x: float,
    point_y: float,
    facet_id: int,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
) -> float:
    edge_x = facet_end_m[facet_id, 0] - facet_start_m[facet_id, 0]
    edge_y = facet_end_m[facet_id, 1] - facet_start_m[facet_id, 1]
    tangent_x = edge_x / facet_length_m[facet_id]
    tangent_y = edge_y / facet_length_m[facet_id]
    along = _exact_sum2(
        (point_x - facet_start_m[facet_id, 0]) * tangent_x,
        (point_y - facet_start_m[facet_id, 1]) * tangent_y,
    )
    parameter = min(max(along / facet_length_m[facet_id], 0.0), 1.0)
    projected_x = facet_start_m[facet_id, 0] + parameter * edge_x
    projected_y = facet_start_m[facet_id, 1] + parameter * edge_y
    return _curved_hypot2(point_x - projected_x, point_y - projected_y)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_endpoint_is_boundary(
    row: int,
    broad_begin: int,
    broad_end: int,
    broad_candidates: Int64Array,
    end_position_m: FloatArray,
    speed_upper_m_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    for offset in range(broad_begin, broad_end):
        facet_id = broad_candidates[offset]
        valid, budget, _ = _curved_event_budget(
            facet_length_m[facet_id],
            geometry_bbox_diagonal_m,
            end_position_m[row, 0],
            end_position_m[row, 1],
            speed_upper_m_s[row],
            root_interval_s[row],
            target_time_s[row],
            geometry_rtol,
            roundoff_ulps,
        )
        distance = _curved_point_segment_distance(
            end_position_m[row, 0],
            end_position_m[row, 1],
            facet_id,
            facet_start_m,
            facet_end_m,
            facet_length_m,
        )
        if valid and math.isfinite(distance) and distance <= budget:
            return True
    return False


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_piece_is_certified_split(
    row: int,
    facet_id: int,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    chord_deviation_m: FloatArray,
    speed_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_bbox_diagonal_m: float,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    bound_x = max(abs(position_lower_m[row, 0]), abs(position_upper_m[row, 0]))
    bound_y = max(abs(position_lower_m[row, 1]), abs(position_upper_m[row, 1]))
    time_scale = (
        start_time_s[row]
        if abs(start_time_s[row]) >= abs(target_time_s[row])
        else target_time_s[row]
    )
    valid, position_budget, time_budget = _curved_event_budget(
        facet_length_m[facet_id],
        geometry_bbox_diagonal_m,
        bound_x,
        bound_y,
        speed_upper_m_s[row],
        root_interval_s[row],
        time_scale,
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return False
    deviation_radius = _curved_hypot2(chord_deviation_m[row, 0], chord_deviation_m[row, 1])
    if deviation_radius <= position_budget:
        return False
    span_x = position_upper_m[row, 0] - position_lower_m[row, 0]
    span_y = position_upper_m[row, 1] - position_lower_m[row, 1]
    tube_diameter = _curved_hypot2(span_x, span_y)
    interval_s = target_time_s[row] - start_time_s[row]
    return interval_s > time_budget or tube_diameter > position_budget


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_chord(
    row: int,
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
) -> tuple[bool, float, float, float, float, float, float]:
    interval_s = target_time_s[row] - start_time_s[row]
    endpoint_x = end_position_m[row, 0] - start_position_m[row, 0]
    endpoint_y = end_position_m[row, 1] - start_position_m[row, 1]
    velocity_x = endpoint_x / interval_s
    velocity_y = endpoint_y / interval_s
    displacement_x = interval_s * velocity_x
    displacement_y = interval_s * velocity_y
    error_x = np.nextafter(abs(displacement_x - endpoint_x), np.inf)
    error_y = np.nextafter(abs(displacement_y - endpoint_y), np.inf)
    valid = (
        _exact_four_finite(velocity_x, velocity_y, displacement_x, displacement_y)
        and math.isfinite(error_x)
        and math.isfinite(error_y)
    )
    return valid, velocity_x, velocity_y, displacement_x, displacement_y, error_x, error_y


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_project_boundary_hit(
    row: int,
    facet_id: int,
    end_position_m: FloatArray,
    speed_upper_m_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, float, float, float, float, float]:
    length = facet_length_m[facet_id]
    edge_x = facet_end_m[facet_id, 0] - facet_start_m[facet_id, 0]
    edge_y = facet_end_m[facet_id, 1] - facet_start_m[facet_id, 1]
    tangent_x = edge_x / length
    tangent_y = edge_y / length
    offset_x = end_position_m[row, 0] - facet_start_m[facet_id, 0]
    offset_y = end_position_m[row, 1] - facet_start_m[facet_id, 1]
    distance_along = _exact_sum2(offset_x * tangent_x, offset_y * tangent_y)
    parameter = min(max(distance_along / length, 0.0), 1.0)
    position_x = facet_start_m[facet_id, 0] + parameter * edge_x
    position_y = facet_start_m[facet_id, 1] + parameter * edge_y
    residual = _curved_hypot2(
        end_position_m[row, 0] - position_x,
        end_position_m[row, 1] - position_y,
    )
    valid, position_budget, time_budget = _curved_event_budget(
        length,
        geometry_bbox_diagonal_m,
        position_x,
        position_y,
        speed_upper_m_s[row],
        root_interval_s[row],
        target_time_s[row],
        geometry_rtol,
        roundoff_ulps,
    )
    accepted = valid and math.isfinite(residual) and residual <= position_budget
    return accepted, position_x, position_y, position_budget, time_budget, residual


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_transverse_position_radius(
    interval_s: float,
    time_radius_s: float,
    displacement_x: float,
    displacement_y: float,
    component_deviation_x: float,
    component_deviation_y: float,
    hit_residual_m: float,
) -> float:
    time_fraction = _curved_upper_quotient(time_radius_s, interval_s)
    error_x = _curved_upper_sum(
        component_deviation_x,
        _curved_upper_product(abs(displacement_x), time_fraction),
    )
    error_y = _curved_upper_sum(
        component_deviation_y,
        _curved_upper_product(abs(displacement_y), time_fraction),
    )
    component_radius = np.nextafter(_curved_hypot2(error_x, error_y), np.inf)
    return _curved_upper_sum(component_radius, hit_residual_m)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_transverse_residual(
    row: int,
    facet_id: int,
    hit_time_s: float,
    hit_x: float,
    hit_y: float,
    hit_residual_m: float,
    position_budget_m: float,
    time_budget_s: float,
    displacement_x: float,
    displacement_y: float,
    roundtrip_error_x: float,
    roundtrip_error_y: float,
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    chord_deviation_m: FloatArray,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    roundoff_ulps: int,
) -> tuple[bool, float]:
    normal_x = facet_normal[facet_id, 0]
    normal_y = facet_normal[facet_id, 1]
    _, start_upper = _curved_offset_dot_bounds(
        start_position_m[row, 0],
        start_position_m[row, 1],
        facet_start_m[facet_id, 0],
        facet_start_m[facet_id, 1],
        normal_x,
        normal_y,
        roundoff_ulps,
    )
    end_lower, _ = _curved_offset_dot_bounds(
        end_position_m[row, 0],
        end_position_m[row, 1],
        facet_start_m[facet_id, 0],
        facet_start_m[facet_id, 1],
        normal_x,
        normal_y,
        roundoff_ulps,
    )
    if start_upper >= -position_budget_m or end_lower <= position_budget_m:
        return False, 0.0
    normal_progress = _exact_sum2(displacement_x * normal_x, displacement_y * normal_y)
    product_scale = _curved_upper_sum(
        _curved_upper_product(abs(displacement_x), abs(normal_x)),
        _curved_upper_product(abs(displacement_y), abs(normal_y)),
    )
    progress_error = _curved_roundoff_margin(product_scale, roundoff_ulps)
    progress_lower = np.nextafter(normal_progress - progress_error, -np.inf)
    if not math.isfinite(progress_lower) or progress_lower <= 0.0:
        return False, 0.0
    component_deviation_x = _curved_upper_sum(
        chord_deviation_m[row, 0],
        roundtrip_error_x,
    )
    component_deviation_y = _curved_upper_sum(
        chord_deviation_m[row, 1],
        roundtrip_error_y,
    )
    normal_deviation = _curved_upper_sum(
        _curved_upper_product(abs(normal_x), component_deviation_x),
        _curved_upper_product(abs(normal_y), component_deviation_y),
    )
    normal_uncertainty = _curved_upper_sum(normal_deviation, hit_residual_m)
    interval_s = target_time_s[row] - start_time_s[row]
    time_radius = _curved_upper_quotient(
        _curved_upper_product(interval_s, normal_uncertainty),
        progress_lower,
    )
    bracket_lower = np.nextafter(hit_time_s - time_radius, -np.inf)
    bracket_upper = np.nextafter(hit_time_s + time_radius, np.inf)
    if (
        not math.isfinite(time_radius)
        or time_radius > time_budget_s
        or bracket_lower <= start_time_s[row]
        or bracket_upper > target_time_s[row]
    ):
        return False, 0.0
    position_radius = _curved_transverse_position_radius(
        interval_s,
        time_radius,
        displacement_x,
        displacement_y,
        component_deviation_x,
        component_deviation_y,
        hit_residual_m,
    )
    if not math.isfinite(position_radius) or position_radius > position_budget_m:
        return False, 0.0
    clearance_required = _curved_upper_sum(position_radius, position_budget_m)
    start_clearance = _curved_distance_lower_bound(
        hit_x,
        hit_y,
        facet_start_m[facet_id, 0],
        facet_start_m[facet_id, 1],
        roundoff_ulps,
    )
    end_clearance = _curved_distance_lower_bound(
        hit_x,
        hit_y,
        facet_end_m[facet_id, 0],
        facet_end_m[facet_id, 1],
        roundoff_ulps,
    )
    if min(start_clearance, end_clearance) <= clearance_required:
        return False, 0.0
    return True, position_radius


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_wall_candidate(  # pyrefly: ignore [bad-return]
    row: int,
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    certify_start_contact_departure: NDArray[np.bool_],
    certify_monotone_approach: bool,
    use_position_controls: NDArray[np.bool_],
    position_control_origin_m: FloatArray,
    relative_position_control_lower_m: FloatArray,
    relative_position_control_upper_m: FloatArray,
    speed_upper_m_s: FloatArray,
    broad_offsets: Int64Array,
    broad_candidates: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[int, int, bool]:
    retained_count = 0
    retained_facet = -1
    departure_certified = False
    for offset in range(broad_offsets[row], broad_offsets[row + 1]):
        facet_id = broad_candidates[offset]
        if use_position_controls[row] and _curved_bernstein_controls_are_inside(
            row,
            facet_id,
            position_lower_m,
            position_upper_m,
            speed_upper_m_s,
            start_time_s,
            target_time_s,
            root_interval_s,
            position_control_origin_m,
            relative_position_control_lower_m,
            relative_position_control_upper_m,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_normal,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        ):
            continue
        if certify_monotone_approach and _curved_monotone_approach_is_clear(
            row,
            facet_id,
            end_position_m,
            velocity_lower_m_s,
            velocity_upper_m_s,
            speed_upper_m_s,
            target_time_s,
            root_interval_s,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_normal,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        ):
            continue
        reachable = _curved_tube_may_reach_supporting_line(
            row,
            facet_id,
            position_lower_m,
            position_upper_m,
            speed_upper_m_s,
            start_time_s,
            target_time_s,
            root_interval_s,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_normal,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        )
        if not reachable:
            continue
        departing = certify_start_contact_departure[row] and _curved_departure_is_certified(
            row,
            facet_id,
            start_position_m,
            position_lower_m,
            position_upper_m,
            velocity_lower_m_s,
            velocity_upper_m_s,
            speed_upper_m_s,
            start_time_s,
            root_interval_s,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_normal,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        )
        if departing:
            departure_certified = True
        else:
            retained_count += 1
            retained_facet = facet_id
    return retained_count, retained_facet, departure_certified


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_wall_localization(  # pyrefly: ignore [bad-return]
    row: int,
    facet_id: int,
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    speed_upper_m_s: FloatArray,
    volume_inside: NDArray[np.bool_],
    broad_offsets: Int64Array,
    broad_candidates: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[np.uint8, float, float, float, float, float, float]:
    endpoint_boundary = _curved_endpoint_is_boundary(
        row,
        broad_offsets[row],
        broad_offsets[row + 1],
        broad_candidates,
        end_position_m,
        speed_upper_m_s,
        target_time_s,
        root_interval_s,
        geometry_bbox_diagonal_m,
        facet_start_m,
        facet_end_m,
        facet_length_m,
        geometry_rtol,
        roundoff_ulps,
    )
    if not endpoint_boundary and volume_inside[row]:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    chord = _curved_chord(
        row,
        start_position_m,
        end_position_m,
        start_time_s,
        target_time_s,
    )
    if not chord[0]:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    interval_s = target_time_s[row] - start_time_s[row]
    line = _exact_line_facet_hit(
        facet_id,
        start_position_m[row, 0],
        start_position_m[row, 1],
        chord[3],
        chord[4],
        _curved_hypot2(chord[1], chord[2]),
        start_time_s[row],
        interval_s,
        geometry_bbox_diagonal_m,
        facet_start_m,
        facet_end_m,
        facet_length_m,
        geometry_rtol,
        roundoff_ulps,
    )
    if line[0]:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if line[1]:
        hit_time, hit_x, hit_y, hit_residual = line[2], line[3], line[4], line[7]
    else:
        if not endpoint_boundary:
            return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        projected = _curved_project_boundary_hit(
            row,
            facet_id,
            end_position_m,
            speed_upper_m_s,
            target_time_s,
            root_interval_s,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        )
        if not projected[0]:
            return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        hit_time = target_time_s[row]
        hit_x, hit_y, hit_residual = projected[1], projected[2], projected[5]
    valid, position_budget, time_budget = _curved_event_budget(
        facet_length_m[facet_id],
        geometry_bbox_diagonal_m,
        hit_x,
        hit_y,
        speed_upper_m_s[row],
        root_interval_s[row],
        hit_time,
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return CURVED_STATUS_WALL, hit_time, hit_x, hit_y, position_budget, time_budget, hit_residual


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_wall_decision(
    row: int,
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    certify_start_contact_departure: NDArray[np.bool_],
    certify_monotone_approach: bool,
    use_position_controls: NDArray[np.bool_],
    position_control_origin_m: FloatArray,
    relative_position_control_lower_m: FloatArray,
    relative_position_control_upper_m: FloatArray,
    chord_deviation_m: FloatArray,
    chord_deviation_valid: NDArray[np.bool_],
    speed_upper_m_s: FloatArray,
    volume_inside: NDArray[np.bool_],
    broad_offsets: Int64Array,
    broad_candidates: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[np.uint8, bool, int, float, float, float, float, float, float]:
    retained_count, retained_facet, departure_certified = _curved_wall_candidate(
        row,
        start_position_m,
        end_position_m,
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        start_time_s,
        target_time_s,
        root_interval_s,
        certify_start_contact_departure,
        certify_monotone_approach,
        use_position_controls,
        position_control_origin_m,
        relative_position_control_lower_m,
        relative_position_control_upper_m,
        speed_upper_m_s,
        broad_offsets,
        broad_candidates,
        geometry_bbox_diagonal_m,
        facet_start_m,
        facet_end_m,
        facet_normal,
        facet_length_m,
        geometry_rtol,
        roundoff_ulps,
    )
    if retained_count == 0:
        return CURVED_STATUS_CLEAR, departure_certified, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if retained_count != 1:
        return CURVED_STATUS_SPLIT, departure_certified, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if chord_deviation_valid[row] and _curved_piece_is_certified_split(
        row,
        retained_facet,
        position_lower_m,
        position_upper_m,
        chord_deviation_m,
        speed_upper_m_s,
        start_time_s,
        target_time_s,
        root_interval_s,
        geometry_bbox_diagonal_m,
        facet_length_m,
        geometry_rtol,
        roundoff_ulps,
    ):
        return CURVED_STATUS_SPLIT, departure_certified, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    localized = _curved_wall_localization(
        row,
        retained_facet,
        start_position_m,
        end_position_m,
        start_time_s,
        target_time_s,
        root_interval_s,
        speed_upper_m_s,
        volume_inside,
        broad_offsets,
        broad_candidates,
        geometry_bbox_diagonal_m,
        facet_start_m,
        facet_end_m,
        facet_length_m,
        geometry_rtol,
        roundoff_ulps,
    )
    if localized[0] != CURVED_STATUS_WALL:
        return CURVED_STATUS_SPLIT, departure_certified, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    hit_time, hit_x, hit_y = localized[1], localized[2], localized[3]
    position_budget, time_budget, hit_residual = localized[4], localized[5], localized[6]
    chord = _curved_chord(row, start_position_m, end_position_m, start_time_s, target_time_s)
    if chord_deviation_valid[row]:
        transverse, transverse_residual = _curved_transverse_residual(
            row,
            retained_facet,
            hit_time,
            hit_x,
            hit_y,
            hit_residual,
            position_budget,
            time_budget,
            chord[3],
            chord[4],
            chord[5],
            chord[6],
            start_position_m,
            end_position_m,
            start_time_s,
            target_time_s,
            chord_deviation_m,
            facet_start_m,
            facet_end_m,
            facet_normal,
            roundoff_ulps,
        )
        if transverse:
            return (
                CURVED_STATUS_WALL,
                departure_certified,
                retained_facet,
                hit_time,
                hit_x,
                hit_y,
                position_budget,
                time_budget,
                transverse_residual,
            )
    span_x = position_upper_m[row, 0] - position_lower_m[row, 0]
    span_y = position_upper_m[row, 1] - position_lower_m[row, 1]
    tube_diameter = np.nextafter(_curved_hypot2(span_x, span_y), np.inf)
    interval_s = target_time_s[row] - start_time_s[row]
    if interval_s > time_budget or tube_diameter > position_budget:
        return CURVED_STATUS_SPLIT, departure_certified, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        CURVED_STATUS_WALL,
        departure_certified,
        retained_facet,
        hit_time,
        hit_x,
        hit_y,
        position_budget,
        time_budget,
        max(hit_residual, tube_diameter),
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_radial_direction(
    lower_velocity_m_s: float,
    upper_velocity_m_s: float,
    roundoff_ulps: int,
) -> tuple[int, float]:
    margin = _curved_roundoff_margin(
        max(abs(lower_velocity_m_s), abs(upper_velocity_m_s)),
        roundoff_ulps,
    )
    lower = np.nextafter(lower_velocity_m_s - margin, -np.inf)
    upper = np.nextafter(upper_velocity_m_s + margin, np.inf)
    if lower > 0.0:
        return 1, lower
    if upper < 0.0:
        return -1, -upper
    return 0, 0.0


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_axis_start_status(
    start_radius_m: float,
    start_radial_velocity_m_s: float,
    end_radius_m: float,
    end_radial_velocity_m_s: float,
    radial_direction: int,
) -> np.uint8:
    if start_radius_m != 0.0:
        return np.uint8(0)
    if start_radial_velocity_m_s < 0.0:
        return CURVED_STATUS_AXIS
    if start_radial_velocity_m_s == 0.0 and end_radius_m == 0.0 and end_radial_velocity_m_s == 0.0:
        return CURVED_STATUS_CLEAR
    if start_radial_velocity_m_s > 0.0 and radial_direction == 1:
        return CURVED_STATUS_CLEAR
    return CURVED_STATUS_SPLIT


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_localize_axis_crossing(
    row: int,
    radial_speed_lower_m_s: float,
    position_budget_m: float,
    time_budget_s: float,
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    chord_deviation_m: FloatArray,
    chord_deviation_valid: NDArray[np.bool_],
) -> tuple[np.uint8, float, float, float]:
    if end_position_m[row, 0] == 0.0:
        return CURVED_STATUS_AXIS, target_time_s[row], end_position_m[row, 1], 0.0
    chord = _curved_chord(
        row,
        start_position_m,
        end_position_m,
        start_time_s,
        target_time_s,
    )
    if not chord[0] or not chord_deviation_valid[row] or chord[1] >= 0.0:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0
    offset_s = start_position_m[row, 0] / -chord[1]
    hit_time = start_time_s[row] + offset_s
    effective_offset = hit_time - start_time_s[row]
    radial_product = effective_offset * chord[1]
    if (
        not math.isfinite(radial_product)
        or not math.isfinite(hit_time)
        or hit_time <= start_time_s[row]
        or hit_time >= target_time_s[row]
    ):
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0
    radial_roundoff = abs(_exact_sum2(start_position_m[row, 0], radial_product))
    residual = _curved_upper_sum(
        _curved_upper_sum(chord_deviation_m[row, 0], chord[5]),
        np.nextafter(radial_roundoff, np.inf),
    )
    if not math.isfinite(residual) or residual > position_budget_m:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0
    time_radius = _curved_upper_quotient(residual, radial_speed_lower_m_s)
    if not math.isfinite(time_radius) or time_radius > time_budget_s:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0
    hit_y = start_position_m[row, 1] + effective_offset * chord[2]
    if not math.isfinite(hit_y):
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0
    return CURVED_STATUS_AXIS, hit_time, hit_y, residual


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _curved_axis_decision(
    row: int,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    chord_deviation_m: FloatArray,
    chord_deviation_valid: NDArray[np.bool_],
    speed_upper_m_s: FloatArray,
    geometry_bbox_diagonal_m: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[np.uint8, float, float, float, float, float, float]:
    bound_x = max(abs(position_lower_m[row, 0]), abs(position_upper_m[row, 0]))
    bound_y = max(abs(position_lower_m[row, 1]), abs(position_upper_m[row, 1]))
    time_scale = (
        start_time_s[row]
        if abs(start_time_s[row]) >= abs(target_time_s[row])
        else target_time_s[row]
    )
    valid_budget, position_budget, time_budget = _curved_event_budget(
        geometry_bbox_diagonal_m,
        geometry_bbox_diagonal_m,
        bound_x,
        bound_y,
        speed_upper_m_s[row],
        root_interval_s[row],
        time_scale,
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid_budget:
        return CURVED_STATUS_FAILURE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    radial_direction, radial_speed_lower = _curved_radial_direction(
        velocity_lower_m_s[row, 0],
        velocity_upper_m_s[row, 0],
        roundoff_ulps,
    )
    start_radius = start_position_m[row, 0]
    end_radius = end_position_m[row, 0]
    start_status = _curved_axis_start_status(
        start_radius,
        start_velocity_m_s[row, 0],
        end_radius,
        end_velocity_m_s[row, 0],
        radial_direction,
    )
    if start_status != 0:
        if start_status == CURVED_STATUS_AXIS:
            return (
                CURVED_STATUS_AXIS,
                start_time_s[row],
                0.0,
                start_position_m[row, 1],
                position_budget,
                time_budget,
                0.0,
            )
        return start_status, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if position_lower_m[row, 0] > position_budget or radial_direction == 1:
        return CURVED_STATUS_CLEAR, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if radial_direction == -1 and end_radius > 0.0:
        return CURVED_STATUS_CLEAR, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if end_radius > 0.0 or radial_direction != -1:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    localized = _curved_localize_axis_crossing(
        row,
        radial_speed_lower,
        position_budget,
        time_budget,
        start_position_m,
        end_position_m,
        start_time_s,
        target_time_s,
        chord_deviation_m,
        chord_deviation_valid,
    )
    if localized[0] != CURVED_STATUS_AXIS:
        return CURVED_STATUS_SPLIT, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        CURVED_STATUS_AXIS,
        localized[1],
        0.0,
        localized[2],
        position_budget,
        time_budget,
        localized[3],
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _locate_curved_events_kernel(
    axisymmetric_rz: bool,
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    root_interval_s: FloatArray,
    certify_start_contact_departure: NDArray[np.bool_],
    certify_monotone_approach: bool,
    use_position_controls: NDArray[np.bool_],
    position_control_origin_m: FloatArray,
    relative_position_control_lower_m: FloatArray,
    relative_position_control_upper_m: FloatArray,
    chord_deviation_m: FloatArray,
    chord_deviation_valid: NDArray[np.bool_],
    speed_upper_m_s: FloatArray,
    ready: NDArray[np.bool_],
    volume_inside: NDArray[np.bool_],
    broad_offsets: Int64Array,
    broad_candidates: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    status: UInt8Array,
    failure_reason: UInt8Array,
    departure_certified: NDArray[np.bool_],
    event_time_s: FloatArray,
    hit_position_m: FloatArray,
    primary_facet_id: Int64Array,
    normal: FloatArray,
    position_budget_m: FloatArray,
    time_budget_s: FloatArray,
    localization_residual_m: FloatArray,
    event_candidate_count: Int64Array,
) -> None:
    for row in range(start_time_s.size):
        if not ready[row]:
            status[row] = CURVED_STATUS_FAILURE
            failure_reason[row] = CURVED_FAILURE_INDETERMINATE_EVENT
            continue
        wall = _curved_wall_decision(
            row,
            start_position_m,
            end_position_m,
            position_lower_m,
            position_upper_m,
            velocity_lower_m_s,
            velocity_upper_m_s,
            start_time_s,
            target_time_s,
            root_interval_s,
            certify_start_contact_departure,
            certify_monotone_approach,
            use_position_controls,
            position_control_origin_m,
            relative_position_control_lower_m,
            relative_position_control_upper_m,
            chord_deviation_m,
            chord_deviation_valid,
            speed_upper_m_s,
            volume_inside,
            broad_offsets,
            broad_candidates,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_normal,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        )
        departure_certified[row] = wall[1]
        axis = (CURVED_STATUS_CLEAR, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if axisymmetric_rz:
            axis = _curved_axis_decision(
                row,
                start_position_m,
                start_velocity_m_s,
                end_position_m,
                end_velocity_m_s,
                position_lower_m,
                position_upper_m,
                velocity_lower_m_s,
                velocity_upper_m_s,
                start_time_s,
                target_time_s,
                root_interval_s,
                chord_deviation_m,
                chord_deviation_valid,
                speed_upper_m_s,
                geometry_bbox_diagonal_m,
                geometry_rtol,
                roundoff_ulps,
            )
        if wall[0] == CURVED_STATUS_FAILURE or axis[0] == CURVED_STATUS_FAILURE:
            status[row] = CURVED_STATUS_FAILURE
            failure_reason[row] = CURVED_FAILURE_INDETERMINATE_EVENT
            continue
        if wall[0] == CURVED_STATUS_SPLIT or axis[0] == CURVED_STATUS_SPLIT:
            status[row] = CURVED_STATUS_SPLIT
            continue
        wall_hit = wall[0] == CURVED_STATUS_WALL
        axis_hit = axis[0] == CURVED_STATUS_AXIS
        choose_wall = wall_hit and (not axis_hit or wall[3] - wall[7] <= axis[1] + axis[5])
        if choose_wall:
            status[row] = CURVED_STATUS_WALL
            primary_facet_id[row] = wall[2]
            event_time_s[row] = wall[3]
            hit_position_m[row, 0] = wall[4]
            hit_position_m[row, 1] = wall[5]
            position_budget_m[row] = wall[6]
            time_budget_s[row] = wall[7]
            localization_residual_m[row] = wall[8]
            normal[row, 0] = facet_normal[wall[2], 0]
            normal[row, 1] = facet_normal[wall[2], 1]
            event_candidate_count[row] = 1
        elif axis_hit:
            status[row] = CURVED_STATUS_AXIS
            event_time_s[row] = axis[1]
            hit_position_m[row, 0] = axis[2]
            hit_position_m[row, 1] = axis[3]
            position_budget_m[row] = axis[4]
            time_budget_s[row] = axis[5]
            localization_residual_m[row] = axis[6]
        else:
            status[row] = CURVED_STATUS_CLEAR


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _fill_curved_event_candidates_kernel(
    status: UInt8Array,
    primary_facet_id: Int64Array,
    candidate_offsets: Int64Array,
    candidate_facet_ids: Int64Array,
) -> None:
    for row in range(status.size):
        if status[row] == CURVED_STATUS_WALL:
            candidate_facet_ids[candidate_offsets[row]] = primary_facet_id[row]


def _prepare_exact_path_batch(
    geometry: PreparedGeometry,
    path_kind: UInt8Array,
    start_position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    certified_departing_facet_id: Int64Array | None,
) -> _PreparedExactPathBatch:
    kinds = np.asarray(path_kind, dtype=np.uint8)
    position = np.asarray(start_position_m, dtype=np.float64)
    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    acceleration = np.asarray(acceleration_m_s2, dtype=np.float64)
    start_time = np.asarray(start_time_s, dtype=np.float64)
    target_time = np.asarray(target_time_s, dtype=np.float64)
    row_count = _validate_exact_batch_inputs(
        geometry,
        kinds,
        position,
        velocity,
        acceleration,
        start_time,
        target_time,
        geometry_rtol,
        roundoff_ulps,
    )
    departing = _exact_departing_facets(
        geometry,
        kinds,
        acceleration,
        certified_departing_facet_id,
        row_count,
    )
    query_lower = np.empty((row_count, 2), dtype=np.float64)
    query_upper = np.empty((row_count, 2), dtype=np.float64)
    linear_displacement = np.empty((row_count, 2), dtype=np.float64)
    quadratic_displacement = np.empty((row_count, 2), dtype=np.float64)
    path_position_bound = np.empty((row_count, 2), dtype=np.float64)
    speed = np.empty(row_count, dtype=np.float64)
    parameter_speed = np.empty(row_count, dtype=np.float64)
    ready = np.empty(row_count, dtype=np.bool_)
    failure_reason = np.zeros(row_count, dtype=np.uint8)
    _prepare_exact_paths_kernel(
        kinds,
        position,
        velocity,
        acceleration,
        start_time,
        target_time,
        departing,
        geometry.bbox_diagonal_m,
        geometry.facet_start_m,
        geometry.facet_end_m,
        geometry.facet_normal,
        geometry.facet_length_m,
        geometry_rtol,
        roundoff_ulps,
        query_lower,
        query_upper,
        linear_displacement,
        quadratic_displacement,
        path_position_bound,
        speed,
        parameter_speed,
        ready,
        failure_reason,
    )
    return _PreparedExactPathBatch(
        kinds,
        position,
        velocity,
        start_time,
        target_time,
        departing,
        query_lower,
        query_upper,
        linear_displacement,
        quadratic_displacement,
        path_position_bound,
        speed,
        parameter_speed,
        ready,
        failure_reason,
    )


def _validate_exact_candidate_capacity(candidate_capacity: int) -> None:
    if isinstance(candidate_capacity, bool) or not isinstance(candidate_capacity, int):
        raise ValueError("candidate_capacity must be an integer")
    if candidate_capacity < 0 or candidate_capacity > _MAX_INT64:
        raise ValueError("candidate_capacity must be nonnegative and fit int64")


def _require_exact_candidate_capacity(
    counts: Int64Array,
    candidate_capacity: int,
) -> int:
    if bool((counts < 0).any()):
        raise EventLocationError("exact-event candidate counts must be nonnegative")
    maximum = 0 if not counts.size else int(counts.max())
    int64_sum_is_safe = not counts.size or maximum <= _MAX_INT64 // counts.size
    if int64_sum_is_safe:
        required_count = int(np.sum(counts, dtype=np.int64))
    else:
        required_count = sum(int(count) for count in counts)
    if required_count <= candidate_capacity:
        return required_count
    oversized = np.flatnonzero(counts > candidate_capacity)
    oversized_row = None if not oversized.size else int(oversized[0])
    raise ExactCandidateCapacityError(required_count, candidate_capacity, oversized_row)


def _validate_exact_batch_inputs(
    geometry: PreparedGeometry,
    path_kind: UInt8Array,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> int:
    row_count = _validate_exact_batch_shapes(
        path_kind,
        position_m,
        velocity_m_s,
        acceleration_m_s2,
        start_time_s,
        target_time_s,
    )
    _validate_exact_batch_values(
        position_m,
        velocity_m_s,
        acceleration_m_s2,
        start_time_s,
        target_time_s,
    )
    if bool(((path_kind != EXACT_PATH_LINEAR) & (path_kind != EXACT_PATH_QUADRATIC)).any()):
        raise ValueError("exact path kind is unsupported")
    if geometry.coordinate_system == "axisymmetric_rz" and bool(
        (path_kind == EXACT_PATH_QUADRATIC).any()
    ):
        raise ValueError("quadratic exact paths require cartesian_xy geometry")
    _validate_exact_tolerances(geometry, geometry_rtol, roundoff_ulps)
    return row_count


def _validate_exact_batch_shapes(
    path_kind: UInt8Array,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
) -> int:
    row_count = int(path_kind.size)
    if path_kind.ndim != 1 or position_m.shape != (row_count, 2):
        raise ValueError("exact path kinds and positions must align")
    if velocity_m_s.shape != position_m.shape or acceleration_m_s2.shape != position_m.shape:
        raise ValueError("exact velocity and acceleration rows must align with positions")
    if start_time_s.shape != (row_count,) or target_time_s.shape != (row_count,):
        raise ValueError("exact start and target times must align with path rows")
    return row_count


def _validate_exact_batch_values(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
) -> None:
    finite = (
        np.isfinite(position_m).all()
        and np.isfinite(velocity_m_s).all()
        and np.isfinite(acceleration_m_s2).all()
        and np.isfinite(start_time_s).all()
        and np.isfinite(target_time_s).all()
    )
    if not bool(finite) or bool((target_time_s <= start_time_s).any()):
        raise ValueError("exact path states must be finite with positive intervals")


def _validate_exact_tolerances(
    geometry: PreparedGeometry,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> None:
    if not math.isfinite(geometry.bbox_diagonal_m) or geometry.bbox_diagonal_m <= 0.0:
        raise ValueError("geometry scale must be finite and positive")
    if not math.isfinite(geometry_rtol) or not 0.0 < geometry_rtol < 1.0:
        raise ValueError("geometry_rtol must be finite and in (0, 1)")
    if isinstance(roundoff_ulps, bool) or not isinstance(roundoff_ulps, int):
        raise ValueError("roundoff_ulps must be an integer")
    if roundoff_ulps <= 0 or roundoff_ulps > _MAX_FLOAT64_INTEGER:
        raise ValueError("roundoff_ulps must be positive and representable in float64")


def _exact_departing_facets(
    geometry: PreparedGeometry,
    path_kind: UInt8Array,
    acceleration_m_s2: FloatArray,
    value: Int64Array | None,
    row_count: int,
) -> Int64Array:
    if value is None:
        return np.full(row_count, -1, dtype=np.int64)
    departing = np.asarray(value, dtype=np.int64)
    if departing.shape != (row_count,):
        raise ValueError("certified departing facet IDs must align with exact path rows")
    if bool(((departing < -1) | (departing >= geometry.facet_count)).any()):
        raise ValueError("certified departing facet ID is outside the geometry")
    if bool(((departing >= 0) & (path_kind != EXACT_PATH_QUADRATIC)).any()):
        raise ValueError("a certified acceleration departure requires a quadratic path")
    zero_acceleration = np.equal(acceleration_m_s2, 0.0).all(axis=1)
    if bool(((departing >= 0) & zero_acceleration).any()):
        raise ValueError("a certified quadratic departure requires nonzero acceleration")
    return departing


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_sum2(first: float, second: float) -> float:
    total = first + second
    second_virtual = total - first
    error = (first - (total - second_virtual)) + (second - second_virtual)
    return total + error


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_sum3(first: float, second: float, third: float) -> float:
    partial = first + second
    second_virtual = partial - first
    first_error = (first - (partial - second_virtual)) + (second - second_virtual)
    total = partial + third
    third_virtual = total - partial
    second_error = (partial - (total - third_virtual)) + (third - third_virtual)
    return _exact_sum2(total, first_error + second_error)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _surface_departure_direction(
    normal_x: float,
    normal_y: float,
    velocity_x: float,
    velocity_y: float,
    acceleration_x: float,
    acceleration_y: float,
    has_acceleration: bool,
    roundoff_ulps: int,
) -> np.uint8:
    normal_speed = _exact_sum2(velocity_x * normal_x, velocity_y * normal_y)
    speed = math.hypot(velocity_x, velocity_y)
    margin = float(roundoff_ulps) * _FLOAT64_EPS * speed
    if normal_speed < -margin:
        return _SURFACE_DIRECTION_VELOCITY_INWARD
    if normal_speed > margin:
        return _SURFACE_DIRECTION_VELOCITY_OUTWARD
    if normal_speed != 0.0 or not has_acceleration:
        return _SURFACE_DIRECTION_INDETERMINATE

    normal_acceleration = _exact_sum2(
        acceleration_x * normal_x,
        acceleration_y * normal_y,
    )
    acceleration_scale = math.hypot(acceleration_x, acceleration_y)
    acceleration_margin = float(roundoff_ulps) * _FLOAT64_EPS * acceleration_scale
    if normal_acceleration < -acceleration_margin:
        return _SURFACE_DIRECTION_ACCELERATION_INWARD
    if normal_acceleration > acceleration_margin:
        return _SURFACE_DIRECTION_ACCELERATION_OUTWARD
    return _SURFACE_DIRECTION_INDETERMINATE


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _classify_surface_release_kernel(
    release_state: UInt8Array,
    source_facet_id: Int64Array,
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    has_acceleration: bool,
    start_time_s: FloatArray,
    interval_s: float,
    curved_event_path: bool,
    geometry_bbox_diagonal_m: float,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    action: UInt8Array,
    status: UInt8Array,
    departure_facet_id: Int64Array,
    position_budget_m: FloatArray,
    time_budget_s: FloatArray,
) -> None:
    for row in range(release_state.size):
        state = release_state[row]
        facet_id = source_facet_id[row]
        if state == SURFACE_STATE_RESOLVED or facet_id < 0:
            action[row] = SURFACE_ACTION_RESOLVED
            continue
        if state == SURFACE_STATE_DEPARTURE:
            if curved_event_path:
                action[row] = SURFACE_ACTION_CURVED_DEPARTURE
            else:
                action[row] = SURFACE_ACTION_EXACT_DEPARTURE
                departure_facet_id[row] = facet_id
            continue

        acceleration_x = 0.0
        acceleration_y = 0.0
        if has_acceleration:
            acceleration_x = acceleration_m_s2[row, 0]
            acceleration_y = acceleration_m_s2[row, 1]
        direction = _surface_departure_direction(
            facet_normal[facet_id, 0],
            facet_normal[facet_id, 1],
            velocity_m_s[row, 0],
            velocity_m_s[row, 1],
            acceleration_x,
            acceleration_y,
            has_acceleration,
            roundoff_ulps,
        )
        if direction == _SURFACE_DIRECTION_VELOCITY_INWARD:
            action[row] = (
                SURFACE_ACTION_CURVED_DEPARTURE if curved_event_path else SURFACE_ACTION_RESOLVED
            )
            continue
        if direction == _SURFACE_DIRECTION_ACCELERATION_INWARD:
            action[row] = SURFACE_ACTION_EXACT_DEPARTURE
            departure_facet_id[row] = facet_id
            continue
        if direction == _SURFACE_DIRECTION_INDETERMINATE:
            status[row] = SURFACE_STATUS_INDETERMINATE_DIRECTION
            continue

        action[row] = (
            SURFACE_ACTION_RESPONSE_VELOCITY
            if direction == _SURFACE_DIRECTION_VELOCITY_OUTWARD
            else SURFACE_ACTION_RESPONSE_ACCELERATION
        )
        speed = math.hypot(velocity_m_s[row, 0], velocity_m_s[row, 1])
        valid_budget, position_budget, time_budget = _exact_event_budget(
            facet_length_m[facet_id],
            geometry_bbox_diagonal_m,
            position_m[row, 0],
            position_m[row, 1],
            speed,
            interval_s,
            start_time_s[row],
            geometry_rtol,
            roundoff_ulps,
        )
        if not valid_budget:
            status[row] = SURFACE_STATUS_INVALID_BUDGET
            continue
        position_budget_m[row] = position_budget
        time_budget_s[row] = time_budget


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_ulp(value: float) -> float:
    return abs(float(np.spacing(np.float64(value))))


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_event_budget(
    facet_length_m: float,
    geometry_bbox_diagonal_m: float,
    position_x: float,
    position_y: float,
    speed_m_s: float,
    interval_s: float,
    time_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, float, float]:
    roundoff_factor = float(roundoff_ulps) * _FLOAT64_EPS
    position_norm = math.hypot(position_x, position_y)
    roundoff_position = roundoff_factor * max(
        geometry_bbox_diagonal_m,
        position_norm,
        facet_length_m,
    )
    position_ulp_floor = float(roundoff_ulps) * max(
        _exact_ulp(position_x),
        _exact_ulp(position_y),
        _exact_ulp(geometry_bbox_diagonal_m),
        _exact_ulp(facet_length_m),
    )
    position_budget = float(
        np.nextafter(
            geometry_rtol * facet_length_m + max(roundoff_position, position_ulp_floor),
            np.inf,
        )
    )
    effective_speed = max(speed_m_s, facet_length_m / interval_s)
    time_ulp_floor = float(roundoff_ulps) * max(
        _exact_ulp(time_s),
        _exact_ulp(interval_s),
    )
    time_budget = max(
        position_budget / effective_speed,
        roundoff_factor * max(abs(time_s), interval_s),
        time_ulp_floor,
    )
    valid = (
        math.isfinite(position_budget)
        and math.isfinite(time_budget)
        and position_budget > 0.0
        and time_budget > 0.0
    )
    return valid, position_budget, time_budget


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_quadratic_component_bound(
    start: float,
    linear: float,
    quadratic: float,
) -> tuple[bool, float]:
    endpoint = _exact_sum3(start, linear, quadratic)
    bound = max(abs(start), abs(endpoint))
    if quadratic != 0.0:
        turning = -linear / (2.0 * quadratic)
        if math.isfinite(turning) and 0.0 < turning < 1.0:
            turning_value = _exact_sum3(
                start,
                turning * linear,
                turning * turning * quadratic,
            )
            bound = max(bound, abs(turning_value))
    return math.isfinite(bound), bound


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_departure_is_certified(
    facet_id: int,
    start_x: float,
    start_y: float,
    velocity_x: float,
    velocity_y: float,
    acceleration_x: float,
    acceleration_y: float,
    speed_m_s: float,
    start_time_s: float,
    interval_s: float,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    if facet_id < 0:
        return True
    length = facet_length_m[facet_id]
    edge_x = facet_end_m[facet_id, 0] - facet_start_m[facet_id, 0]
    edge_y = facet_end_m[facet_id, 1] - facet_start_m[facet_id, 1]
    tangent_x = edge_x / length
    tangent_y = edge_y / length
    offset_x = start_x - facet_start_m[facet_id, 0]
    offset_y = start_y - facet_start_m[facet_id, 1]
    distance_along = _exact_sum2(offset_x * tangent_x, offset_y * tangent_y)
    parameter = distance_along / length
    projection_x = facet_start_m[facet_id, 0] + parameter * edge_x
    projection_y = facet_start_m[facet_id, 1] + parameter * edge_y
    normal_x = facet_normal[facet_id, 0]
    normal_y = facet_normal[facet_id, 1]
    signed_distance = _exact_sum2(offset_x * normal_x, offset_y * normal_y)
    line_residual = math.hypot(start_x - projection_x, start_y - projection_y)
    valid, position_budget, _ = _exact_event_budget(
        length,
        geometry_bbox_diagonal_m,
        start_x,
        start_y,
        speed_m_s,
        interval_s,
        start_time_s,
        geometry_rtol,
        roundoff_ulps,
    )
    endpoint_margin = position_budget / length
    if not valid or not math.isfinite(signed_distance) or signed_distance > position_budget:
        return False
    if abs(signed_distance) <= position_budget and (
        not math.isfinite(parameter)
        or line_residual > position_budget
        or parameter <= endpoint_margin
        or parameter >= 1.0 - endpoint_margin
    ):
        return False
    normal_speed = _exact_sum2(velocity_x * normal_x, velocity_y * normal_y)
    velocity_margin = float(roundoff_ulps) * _FLOAT64_EPS * math.hypot(velocity_x, velocity_y)
    if normal_speed != 0.0 and normal_speed >= -velocity_margin:
        return False
    normal_acceleration = _exact_sum2(
        acceleration_x * normal_x,
        acceleration_y * normal_y,
    )
    acceleration_margin = (
        float(roundoff_ulps) * _FLOAT64_EPS * math.hypot(acceleration_x, acceleration_y)
    )
    return normal_acceleration < -acceleration_margin


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_query_padding(
    path_kind: np.uint8,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    bound_x: float,
    bound_y: float,
    speed_m_s: float,
    chord_deviation_m: float,
    start_time_s: float,
    interval_s: float,
    maximum_facet_length_m: float,
    geometry_bbox_diagonal_m: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, float]:
    if maximum_facet_length_m == 0.0:
        return True, 0.0
    if path_kind == EXACT_PATH_LINEAR:
        if math.hypot(start_x, start_y) >= math.hypot(end_x, end_y):
            representative_x, representative_y = start_x, start_y
        else:
            representative_x, representative_y = end_x, end_y
        representative_time = start_time_s
    else:
        representative_x, representative_y = bound_x, bound_y
        end_time_s = start_time_s + interval_s
        representative_time = start_time_s if abs(start_time_s) >= abs(end_time_s) else end_time_s
    valid, padding, _ = _exact_event_budget(
        maximum_facet_length_m,
        geometry_bbox_diagonal_m,
        representative_x,
        representative_y,
        speed_m_s,
        interval_s,
        representative_time,
        geometry_rtol,
        roundoff_ulps,
    )
    if path_kind == EXACT_PATH_QUADRATIC:
        padding = _exact_sum2(padding, chord_deviation_m)
    return valid and math.isfinite(padding), padding


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_path_metrics(
    path_kind: np.uint8,
    start_x: float,
    start_y: float,
    velocity_x: float,
    velocity_y: float,
    acceleration_x: float,
    acceleration_y: float,
    interval_s: float,
) -> tuple[
    bool,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    linear_x = interval_s * velocity_x
    linear_y = interval_s * velocity_y
    quadratic_x = 0.0
    quadratic_y = 0.0
    if path_kind == EXACT_PATH_QUADRATIC:
        quadratic_x = 0.5 * interval_s * (interval_s * acceleration_x)
        quadratic_y = 0.5 * interval_s * (interval_s * acceleration_y)
    end_x = start_x + linear_x + quadratic_x
    end_y = start_y + linear_y + quadratic_y
    end_velocity_x = velocity_x + interval_s * acceleration_x
    end_velocity_y = velocity_y + interval_s * acceleration_y
    end_parameter_velocity_x = linear_x + 2.0 * quadratic_x
    end_parameter_velocity_y = linear_y + 2.0 * quadratic_y
    speed = math.hypot(velocity_x, velocity_y)
    parameter_speed = math.hypot(linear_x, linear_y)
    chord_deviation = 0.0
    bound_x = max(abs(start_x), abs(end_x))
    bound_y = max(abs(start_y), abs(end_y))
    bound_valid = True
    if path_kind == EXACT_PATH_QUADRATIC:
        speed = max(speed, math.hypot(end_velocity_x, end_velocity_y))
        parameter_speed = max(
            parameter_speed,
            math.hypot(end_parameter_velocity_x, end_parameter_velocity_y),
        )
        chord_deviation = 0.25 * math.hypot(quadratic_x, quadratic_y)
        valid_x, bound_x = _exact_quadratic_component_bound(start_x, linear_x, quadratic_x)
        valid_y, bound_y = _exact_quadratic_component_bound(start_y, linear_y, quadratic_y)
        bound_valid = valid_x and valid_y
    values = (
        linear_x,
        linear_y,
        quadratic_x,
        quadratic_y,
        end_x,
        end_y,
        end_velocity_x,
        end_velocity_y,
        end_parameter_velocity_x,
        end_parameter_velocity_y,
        speed,
        parameter_speed,
        chord_deviation,
    )
    valid = bound_valid
    for value in values:
        if not math.isfinite(value):
            valid = False
    return (
        valid,
        end_x,
        end_y,
        linear_x,
        linear_y,
        quadratic_x,
        quadratic_y,
        bound_x,
        bound_y,
        speed,
        parameter_speed,
        chord_deviation,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _prepare_exact_path_row(
    row: int,
    path_kind: UInt8Array,
    start_position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    departing_facet_id: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    maximum_facet_length_m: float,
    geometry_rtol: float,
    roundoff_ulps: int,
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
    path_position_bound_m: FloatArray,
    speed_m_s: FloatArray,
    parameter_speed_m: FloatArray,
) -> tuple[bool, bool]:
    start_x = start_position_m[row, 0]
    start_y = start_position_m[row, 1]
    velocity_x = velocity_m_s[row, 0]
    velocity_y = velocity_m_s[row, 1]
    acceleration_x = acceleration_m_s2[row, 0]
    acceleration_y = acceleration_m_s2[row, 1]
    interval_s = target_time_s[row] - start_time_s[row]
    metrics = _exact_path_metrics(
        path_kind[row],
        start_x,
        start_y,
        velocity_x,
        velocity_y,
        acceleration_x,
        acceleration_y,
        interval_s,
    )
    (
        valid,
        end_x,
        end_y,
        linear_x,
        linear_y,
        quadratic_x,
        quadratic_y,
        bound_x,
        bound_y,
        speed,
        parameter_speed,
        chord_deviation,
    ) = metrics
    linear_displacement_m[row, 0] = linear_x
    linear_displacement_m[row, 1] = linear_y
    quadratic_displacement_m[row, 0] = quadratic_x
    quadratic_displacement_m[row, 1] = quadratic_y
    path_position_bound_m[row, 0] = bound_x
    path_position_bound_m[row, 1] = bound_y
    speed_m_s[row] = speed
    parameter_speed_m[row] = parameter_speed
    if not valid:
        return False, False
    departure_valid = _exact_departure_is_certified(
        departing_facet_id[row],
        start_x,
        start_y,
        velocity_x,
        velocity_y,
        acceleration_x,
        acceleration_y,
        speed,
        start_time_s[row],
        interval_s,
        geometry_bbox_diagonal_m,
        facet_start_m,
        facet_end_m,
        facet_normal,
        facet_length_m,
        geometry_rtol,
        roundoff_ulps,
    )
    if path_kind[row] == EXACT_PATH_QUADRATIC and not departure_valid:
        return False, False
    if parameter_speed == 0.0:
        return True, False
    valid_padding, padding = _exact_query_padding(
        path_kind[row],
        start_x,
        start_y,
        end_x,
        end_y,
        bound_x,
        bound_y,
        speed,
        chord_deviation,
        start_time_s[row],
        interval_s,
        maximum_facet_length_m,
        geometry_bbox_diagonal_m,
        geometry_rtol,
        roundoff_ulps,
    )
    lower_x = min(start_x, end_x) - padding
    lower_y = min(start_y, end_y) - padding
    upper_x = max(start_x, end_x) + padding
    upper_y = max(start_y, end_y) + padding
    query_values = (lower_x, lower_y, upper_x, upper_y)
    for value in query_values:
        if not math.isfinite(value):
            valid_padding = False
    if not valid_padding:
        return False, False
    query_lower_m[row, 0] = lower_x
    query_lower_m[row, 1] = lower_y
    query_upper_m[row, 0] = upper_x
    query_upper_m[row, 1] = upper_y
    return True, True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _prepare_exact_paths_kernel(
    path_kind: UInt8Array,
    start_position_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    departing_facet_id: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    query_lower_m: FloatArray,
    query_upper_m: FloatArray,
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
    path_position_bound_m: FloatArray,
    speed_m_s: FloatArray,
    parameter_speed_m: FloatArray,
    ready: NDArray[np.bool_],
    failure_reason: UInt8Array,
) -> None:
    maximum_facet_length_m = 0.0
    for facet_id in range(facet_length_m.size):
        maximum_facet_length_m = max(maximum_facet_length_m, facet_length_m[facet_id])
    for row in range(path_kind.size):
        ready[row] = False
        failure_reason[row] = EXACT_FAILURE_NONE
        query_lower_m[row, 0] = 0.0
        query_lower_m[row, 1] = 0.0
        query_upper_m[row, 0] = 0.0
        query_upper_m[row, 1] = 0.0
        valid, row_ready = _prepare_exact_path_row(
            row,
            path_kind,
            start_position_m,
            velocity_m_s,
            acceleration_m_s2,
            start_time_s,
            target_time_s,
            departing_facet_id,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_normal,
            facet_length_m,
            maximum_facet_length_m,
            geometry_rtol,
            roundoff_ulps,
            query_lower_m,
            query_upper_m,
            linear_displacement_m,
            quadratic_displacement_m,
            path_position_bound_m,
            speed_m_s,
            parameter_speed_m,
        )
        if not valid:
            failure_reason[row] = EXACT_FAILURE_INDETERMINATE_EVENT
            continue
        ready[row] = row_ready


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_line_parameters(
    displacement_x: float,
    displacement_y: float,
    facet_edge_x: float,
    facet_edge_y: float,
    offset_x: float,
    offset_y: float,
    path_length_m: float,
    facet_length_m: float,
    roundoff_ulps: int,
    path_padding_m: float,
) -> tuple[int, float, float]:
    local_scale = max(
        path_length_m,
        facet_length_m,
        abs(offset_x),
        abs(offset_y),
    )
    if not math.isfinite(local_scale) or local_scale <= 0.0:
        return 2, 0.0, 0.0
    path_x = displacement_x / local_scale
    path_y = displacement_y / local_scale
    facet_x = facet_edge_x / local_scale
    facet_y = facet_edge_y / local_scale
    local_offset_x = offset_x / local_scale
    local_offset_y = offset_y / local_scale
    denominator = _exact_sum2(path_x * facet_y, -path_y * facet_x)
    product_scale = abs(path_x * facet_y) + abs(path_y * facet_x)
    denominator_error = float(roundoff_ulps) * _FLOAT64_EPS * max(product_scale, _FLOAT64_EPS)
    if abs(denominator) <= denominator_error:
        local_length = math.hypot(facet_x, facet_y)
        if not math.isfinite(local_length) or local_length <= 0.0:
            return 2, 0.0, 0.0
        separation = (
            abs(_exact_sum2(local_offset_x * facet_y, -local_offset_y * facet_x))
            / local_length
            * local_scale
        )
        if not math.isfinite(separation) or separation <= path_padding_m:
            return 2, 0.0, 0.0
        return 0, 0.0, 0.0
    path_parameter = (
        _exact_sum2(
            local_offset_x * facet_y,
            -local_offset_y * facet_x,
        )
        / denominator
    )
    facet_parameter = (
        _exact_sum2(
            local_offset_x * path_y,
            -local_offset_y * path_x,
        )
        / denominator
    )
    if not math.isfinite(path_parameter) or not math.isfinite(facet_parameter):
        return 2, 0.0, 0.0
    return 1, path_parameter, facet_parameter


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_four_finite(first: float, second: float, third: float, fourth: float) -> bool:
    values = (first, second, third, fourth)
    for value in values:
        if not math.isfinite(value):
            return False
    return True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_line_path_padding(
    start_x: float,
    start_y: float,
    displacement_x: float,
    displacement_y: float,
    speed_m_s: float,
    start_time_s: float,
    interval_s: float,
    facet_length_m: float,
    geometry_bbox_diagonal_m: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, float]:
    end_x = start_x + displacement_x
    end_y = start_y + displacement_y
    if math.hypot(start_x, start_y) >= math.hypot(end_x, end_y):
        representative_x, representative_y = start_x, start_y
    else:
        representative_x, representative_y = end_x, end_y
    end_time_s = start_time_s + interval_s
    representative_time = start_time_s if abs(start_time_s) >= abs(end_time_s) else end_time_s
    valid, path_padding, _ = _exact_event_budget(
        facet_length_m,
        geometry_bbox_diagonal_m,
        representative_x,
        representative_y,
        speed_m_s,
        interval_s,
        representative_time,
        geometry_rtol,
        roundoff_ulps,
    )
    return valid, path_padding


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_line_parameters_are_outside(
    path_parameter: float,
    facet_parameter: float,
    bounded_path: float,
    position_budget_m: float,
    time_budget_s: float,
    path_length_m: float,
    facet_length_m: float,
    interval_s: float,
) -> bool:
    path_slack = min(position_budget_m / path_length_m, time_budget_s / interval_s)
    facet_slack = position_budget_m / facet_length_m
    return (
        path_parameter < -path_slack
        or path_parameter > 1.0 + path_slack
        or facet_parameter < -facet_slack
        or facet_parameter > 1.0 + facet_slack
        or bounded_path <= 0.0
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_line_facet_hit(
    facet_id: int,
    start_x: float,
    start_y: float,
    displacement_x: float,
    displacement_y: float,
    speed_m_s: float,
    start_time_s: float,
    interval_s: float,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, bool, float, float, float, float, float, float]:
    facet_start_x = facet_start_m[facet_id, 0]
    facet_start_y = facet_start_m[facet_id, 1]
    facet_edge_x = facet_end_m[facet_id, 0] - facet_start_x
    facet_edge_y = facet_end_m[facet_id, 1] - facet_start_y
    offset_x = facet_start_x - start_x
    offset_y = facet_start_y - start_y
    if not _exact_four_finite(facet_edge_x, facet_edge_y, offset_x, offset_y):
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    path_length = math.hypot(displacement_x, displacement_y)
    facet_length = facet_length_m[facet_id]
    valid, path_padding = _exact_line_path_padding(
        start_x,
        start_y,
        displacement_x,
        displacement_y,
        speed_m_s,
        start_time_s,
        interval_s,
        facet_length,
        geometry_bbox_diagonal_m,
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    parameter_status, path_parameter, facet_parameter = _exact_line_parameters(
        displacement_x,
        displacement_y,
        facet_edge_x,
        facet_edge_y,
        offset_x,
        offset_y,
        path_length,
        facet_length,
        roundoff_ulps,
        path_padding,
    )
    if parameter_status == 2:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if parameter_status == 0:
        return False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    bounded_path = min(max(path_parameter, 0.0), 1.0)
    approximate_x = start_x + bounded_path * displacement_x
    approximate_y = start_y + bounded_path * displacement_y
    approximate_time = start_time_s + bounded_path * interval_s
    valid, position_budget, time_budget = _exact_event_budget(
        facet_length,
        geometry_bbox_diagonal_m,
        approximate_x,
        approximate_y,
        speed_m_s,
        interval_s,
        approximate_time,
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if _exact_line_parameters_are_outside(
        path_parameter,
        facet_parameter,
        bounded_path,
        position_budget,
        time_budget,
        path_length,
        facet_length,
        interval_s,
    ):
        return False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    bounded_facet = min(max(facet_parameter, 0.0), 1.0)
    path_x = start_x + bounded_path * displacement_x
    path_y = start_y + bounded_path * displacement_y
    facet_x = facet_start_x + bounded_facet * facet_edge_x
    facet_y = facet_start_y + bounded_facet * facet_edge_y
    residual = math.hypot(path_x - facet_x, path_y - facet_y)
    hit_time = start_time_s + bounded_path * interval_s
    if not math.isfinite(residual) or residual > position_budget:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if not math.isfinite(hit_time) or hit_time <= start_time_s:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    valid, position_budget, time_budget = _exact_event_budget(
        facet_length,
        geometry_bbox_diagonal_m,
        path_x,
        path_y,
        speed_m_s,
        interval_s,
        hit_time,
        geometry_rtol,
        roundoff_ulps,
    )
    return (
        not valid,
        valid,
        hit_time,
        path_x,
        path_y,
        position_budget,
        time_budget,
        residual,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_degenerate_quadratic_roots(
    constant: float,
    linear: float,
    facet_local_x: float,
    facet_local_y: float,
    local_scale: float,
    path_padding_m: float,
) -> tuple[bool, int, float, float]:
    if linear != 0.0:
        root = -constant / linear
        return not math.isfinite(root), 1, root, 0.0
    local_length = math.hypot(facet_local_x, facet_local_y)
    if not math.isfinite(local_length) or local_length <= 0.0:
        return True, 0, 0.0, 0.0
    separation = abs(constant) / local_length * local_scale
    if not math.isfinite(separation) or separation <= path_padding_m:
        return True, 0, 0.0, 0.0
    return False, 0, 0.0, 0.0


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_nondegenerate_quadratic_roots(
    constant: float,
    linear: float,
    quadratic: float,
    roundoff_ulps: int,
) -> tuple[bool, int, float, float]:
    squared_linear = linear * linear
    four_quadratic_constant = 4.0 * quadratic * constant
    discriminant = _exact_sum2(squared_linear, -four_quadratic_constant)
    discriminant_error = (
        float(roundoff_ulps)
        * _FLOAT64_EPS
        * max(squared_linear + abs(four_quadratic_constant), _FLOAT64_EPS)
    )
    if not math.isfinite(discriminant) or not math.isfinite(discriminant_error):
        return True, 0, 0.0, 0.0
    if discriminant < -discriminant_error:
        return False, 0, 0.0, 0.0
    if abs(discriminant) <= discriminant_error:
        return True, 0, 0.0, 0.0
    square_root = math.sqrt(discriminant)
    stable_term = -0.5 * (linear + math.copysign(square_root, linear))
    if stable_term == 0.0 or not math.isfinite(stable_term):
        return True, 0, 0.0, 0.0
    first = stable_term / quadratic
    second = constant / stable_term
    if not math.isfinite(first) or not math.isfinite(second):
        return True, 0, 0.0, 0.0
    return False, 2, min(first, second), max(first, second)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_quadratic_roots(
    linear_x: float,
    linear_y: float,
    quadratic_x: float,
    quadratic_y: float,
    facet_edge_x: float,
    facet_edge_y: float,
    offset_x: float,
    offset_y: float,
    facet_length_m: float,
    roundoff_ulps: int,
    path_padding_m: float,
) -> tuple[bool, int, float, float]:
    local_scale = max(
        math.hypot(linear_x, linear_y),
        math.hypot(quadratic_x, quadratic_y),
        facet_length_m,
        abs(offset_x),
        abs(offset_y),
    )
    if not math.isfinite(local_scale) or local_scale <= 0.0:
        return True, 0, 0.0, 0.0
    linear_local_x = linear_x / local_scale
    linear_local_y = linear_y / local_scale
    quadratic_local_x = quadratic_x / local_scale
    quadratic_local_y = quadratic_y / local_scale
    facet_local_x = facet_edge_x / local_scale
    facet_local_y = facet_edge_y / local_scale
    offset_local_x = offset_x / local_scale
    offset_local_y = offset_y / local_scale
    constant = _exact_sum2(
        offset_local_x * facet_local_y,
        -offset_local_y * facet_local_x,
    )
    linear = _exact_sum2(
        linear_local_x * facet_local_y,
        -linear_local_y * facet_local_x,
    )
    quadratic = _exact_sum2(
        quadratic_local_x * facet_local_y,
        -quadratic_local_y * facet_local_x,
    )
    if not _exact_four_finite(constant, linear, quadratic, local_scale):
        return True, 0, 0.0, 0.0
    if quadratic == 0.0:
        return _exact_degenerate_quadratic_roots(
            constant,
            linear,
            facet_local_x,
            facet_local_y,
            local_scale,
            path_padding_m,
        )
    return _exact_nondegenerate_quadratic_roots(
        constant,
        linear,
        quadratic,
        roundoff_ulps,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_quadratic_root_is_outside_interval(
    root: float,
    root_slack: float,
) -> bool:
    return root < -root_slack or root > 1.0 + root_slack


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_quadratic_facet_is_outside(
    facet_parameter: float,
    facet_slack: float,
    bounded_root: float,
) -> bool:
    return (
        facet_parameter < -facet_slack or facet_parameter > 1.0 + facet_slack or bounded_root <= 0.0
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_quadratic_root_hit(
    root: float,
    start_x: float,
    start_y: float,
    linear_x: float,
    linear_y: float,
    quadratic_x: float,
    quadratic_y: float,
    facet_start_x: float,
    facet_start_y: float,
    facet_edge_x: float,
    facet_edge_y: float,
    facet_length_m: float,
    speed_m_s: float,
    parameter_speed_m: float,
    start_time_s: float,
    interval_s: float,
    geometry_bbox_diagonal_m: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, bool, float, float, float, float, float, float]:
    bounded_root = min(max(root, 0.0), 1.0)
    squared_root = bounded_root * bounded_root
    position_x = _exact_sum3(
        start_x,
        bounded_root * linear_x,
        squared_root * quadratic_x,
    )
    position_y = _exact_sum3(
        start_y,
        bounded_root * linear_y,
        squared_root * quadratic_y,
    )
    approximate_time = start_time_s + bounded_root * interval_s
    if not _exact_four_finite(position_x, position_y, 0.0, 0.0):
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    valid, position_budget, time_budget = _exact_event_budget(
        facet_length_m,
        geometry_bbox_diagonal_m,
        position_x,
        position_y,
        speed_m_s,
        interval_s,
        approximate_time,
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    root_slack = min(position_budget / parameter_speed_m, time_budget / interval_s)
    if _exact_quadratic_root_is_outside_interval(root, root_slack):
        return False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    offset_x = position_x - facet_start_x
    offset_y = position_y - facet_start_y
    if not _exact_four_finite(offset_x, offset_y, 0.0, 0.0):
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    tangent_x = facet_edge_x / facet_length_m
    tangent_y = facet_edge_y / facet_length_m
    facet_distance = _exact_sum2(offset_x * tangent_x, offset_y * tangent_y)
    facet_parameter = facet_distance / facet_length_m
    if not math.isfinite(facet_parameter):
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    facet_slack = position_budget / facet_length_m
    if _exact_quadratic_facet_is_outside(
        facet_parameter,
        facet_slack,
        bounded_root,
    ):
        return False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    bounded_facet = min(max(facet_parameter, 0.0), 1.0)
    facet_x = facet_start_x + bounded_facet * facet_edge_x
    facet_y = facet_start_y + bounded_facet * facet_edge_y
    residual = math.hypot(position_x - facet_x, position_y - facet_y)
    hit_time = start_time_s + bounded_root * interval_s
    if not math.isfinite(residual) or residual > position_budget:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if not math.isfinite(hit_time) or hit_time <= start_time_s:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    valid, position_budget, time_budget = _exact_event_budget(
        facet_length_m,
        geometry_bbox_diagonal_m,
        position_x,
        position_y,
        speed_m_s,
        interval_s,
        hit_time,
        geometry_rtol,
        roundoff_ulps,
    )
    return (
        not valid,
        valid,
        hit_time,
        position_x,
        position_y,
        position_budget,
        time_budget,
        residual,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_quadratic_facet_hit(
    facet_id: int,
    start_x: float,
    start_y: float,
    linear_x: float,
    linear_y: float,
    quadratic_x: float,
    quadratic_y: float,
    path_bound_x: float,
    path_bound_y: float,
    speed_m_s: float,
    parameter_speed_m: float,
    start_time_s: float,
    interval_s: float,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, bool, float, float, float, float, float, float]:
    facet_start_x = facet_start_m[facet_id, 0]
    facet_start_y = facet_start_m[facet_id, 1]
    facet_edge_x = facet_end_m[facet_id, 0] - facet_start_x
    facet_edge_y = facet_end_m[facet_id, 1] - facet_start_y
    offset_x = start_x - facet_start_x
    offset_y = start_y - facet_start_y
    if not (
        math.isfinite(facet_edge_x)
        and math.isfinite(facet_edge_y)
        and math.isfinite(offset_x)
        and math.isfinite(offset_y)
    ):
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    facet_length = facet_length_m[facet_id]
    end_time_s = start_time_s + interval_s
    representative_time = start_time_s if abs(start_time_s) >= abs(end_time_s) else end_time_s
    valid, path_padding, _ = _exact_event_budget(
        facet_length,
        geometry_bbox_diagonal_m,
        path_bound_x,
        path_bound_y,
        speed_m_s,
        interval_s,
        representative_time,
        geometry_rtol,
        roundoff_ulps,
    )
    if not valid:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    failure, root_count, first_root, second_root = _exact_quadratic_roots(
        linear_x,
        linear_y,
        quadratic_x,
        quadratic_y,
        facet_edge_x,
        facet_edge_y,
        offset_x,
        offset_y,
        facet_length,
        roundoff_ulps,
        path_padding,
    )
    if failure:
        return True, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    found = False
    selected = (False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for root_index in range(root_count):
        root = first_root if root_index == 0 else second_root
        candidate = _exact_quadratic_root_hit(
            root,
            start_x,
            start_y,
            linear_x,
            linear_y,
            quadratic_x,
            quadratic_y,
            facet_start_x,
            facet_start_y,
            facet_edge_x,
            facet_edge_y,
            facet_length,
            speed_m_s,
            parameter_speed_m,
            start_time_s,
            interval_s,
            geometry_bbox_diagonal_m,
            geometry_rtol,
            roundoff_ulps,
        )
        if candidate[0]:
            return candidate
        if candidate[1] and (not found or candidate[2] < selected[2]):
            selected = candidate
            found = True
    return selected


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_facet_hit(
    path_kind: np.uint8,
    facet_id: int,
    start_x: float,
    start_y: float,
    linear_x: float,
    linear_y: float,
    quadratic_x: float,
    quadratic_y: float,
    path_bound_x: float,
    path_bound_y: float,
    speed_m_s: float,
    parameter_speed_m: float,
    start_time_s: float,
    interval_s: float,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, bool, float, float, float, float, float, float]:
    if path_kind == EXACT_PATH_LINEAR:
        return _exact_line_facet_hit(
            facet_id,
            start_x,
            start_y,
            linear_x,
            linear_y,
            speed_m_s,
            start_time_s,
            interval_s,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        )
    return _exact_quadratic_facet_hit(
        facet_id,
        start_x,
        start_y,
        linear_x,
        linear_y,
        quadratic_x,
        quadratic_y,
        path_bound_x,
        path_bound_y,
        speed_m_s,
        parameter_speed_m,
        start_time_s,
        interval_s,
        geometry_bbox_diagonal_m,
        facet_start_m,
        facet_end_m,
        facet_length_m,
        geometry_rtol,
        roundoff_ulps,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _exact_axis_hit(
    axisymmetric_rz: bool,
    start_x: float,
    start_y: float,
    velocity_x: float,
    velocity_y: float,
    start_time_s: float,
    target_time_s: float,
) -> tuple[bool, bool, float, float, float]:
    if not axisymmetric_rz:
        return False, False, 0.0, 0.0, 0.0
    if start_x < 0.0:
        return True, False, 0.0, 0.0, 0.0
    if velocity_x >= 0.0:
        return False, False, 0.0, 0.0, 0.0
    interval_s = target_time_s - start_time_s
    radial_end = start_x + interval_s * velocity_x
    if radial_end > 0.0:
        return False, False, 0.0, 0.0, 0.0
    crossing_time = start_time_s + start_x / -velocity_x
    if start_x > 0.0 and not start_time_s < crossing_time <= target_time_s:
        return True, False, 0.0, 0.0, 0.0
    axial_position = start_y + (crossing_time - start_time_s) * velocity_y
    if not math.isfinite(crossing_time) or not math.isfinite(axial_position):
        return True, False, 0.0, 0.0, 0.0
    return False, True, crossing_time, 0.0, axial_position


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _find_first_exact_wall(
    path_kind: np.uint8,
    start_x: float,
    start_y: float,
    linear_x: float,
    linear_y: float,
    quadratic_x: float,
    quadratic_y: float,
    path_bound_x: float,
    path_bound_y: float,
    speed_m_s: float,
    parameter_speed_m: float,
    start_time_s: float,
    interval_s: float,
    departing_facet_id: int,
    broad_begin: int,
    broad_end: int,
    broad_candidates: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, bool, int, float, float, float, float, float, float]:
    found = False
    primary = -1
    selected_time = 0.0
    selected_x = 0.0
    selected_y = 0.0
    selected_position_budget = 0.0
    selected_time_budget = 0.0
    selected_residual = 0.0
    for offset in range(broad_begin, broad_end):
        facet_id = broad_candidates[offset]
        if facet_id == departing_facet_id:
            continue
        candidate = _exact_facet_hit(
            path_kind,
            facet_id,
            start_x,
            start_y,
            linear_x,
            linear_y,
            quadratic_x,
            quadratic_y,
            path_bound_x,
            path_bound_y,
            speed_m_s,
            parameter_speed_m,
            start_time_s,
            interval_s,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        )
        if candidate[0]:
            return True, False, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if candidate[1] and (
            not found
            or candidate[2] < selected_time
            or (candidate[2] == selected_time and facet_id < primary)
        ):
            found = True
            primary = facet_id
            selected_time = candidate[2]
            selected_x = candidate[3]
            selected_y = candidate[4]
            selected_position_budget = candidate[5]
            selected_time_budget = candidate[6]
            selected_residual = candidate[7]
    return (
        False,
        found,
        primary,
        selected_time,
        selected_x,
        selected_y,
        selected_position_budget,
        selected_time_budget,
        selected_residual,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _is_simultaneous_exact_hit(
    candidate_time: float,
    candidate_x: float,
    candidate_y: float,
    candidate_position_budget: float,
    candidate_time_budget: float,
    first_time: float,
    first_x: float,
    first_y: float,
    first_position_budget: float,
    first_time_budget: float,
) -> bool:
    position_budget = max(first_position_budget, candidate_position_budget)
    time_budget = max(first_time_budget, candidate_time_budget)
    separation = math.hypot(candidate_x - first_x, candidate_y - first_y)
    return abs(candidate_time - first_time) <= time_budget and separation <= position_budget


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _reduce_simultaneous_exact_hits(
    path_kind: np.uint8,
    start_x: float,
    start_y: float,
    linear_x: float,
    linear_y: float,
    quadratic_x: float,
    quadratic_y: float,
    path_bound_x: float,
    path_bound_y: float,
    speed_m_s: float,
    parameter_speed_m: float,
    start_time_s: float,
    interval_s: float,
    departing_facet_id: int,
    broad_begin: int,
    broad_end: int,
    broad_candidates: Int64Array,
    first_facet_id: int,
    first_time: float,
    first_x: float,
    first_y: float,
    first_position_budget: float,
    first_time_budget: float,
    first_residual: float,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    facet_node_ids: Int64Array,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[bool, int, float, float, float]:
    count = 0
    position_budget = first_position_budget
    time_budget = first_time_budget
    residual = first_residual
    common_node0 = facet_node_ids[first_facet_id, 0]
    common_node1 = facet_node_ids[first_facet_id, 1]
    for offset in range(broad_begin, broad_end):
        facet_id = broad_candidates[offset]
        if facet_id == departing_facet_id:
            continue
        candidate = _exact_facet_hit(
            path_kind,
            facet_id,
            start_x,
            start_y,
            linear_x,
            linear_y,
            quadratic_x,
            quadratic_y,
            path_bound_x,
            path_bound_y,
            speed_m_s,
            parameter_speed_m,
            start_time_s,
            interval_s,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_length_m,
            geometry_rtol,
            roundoff_ulps,
        )
        if candidate[0]:
            return True, 0, 0.0, 0.0, 0.0
        if candidate[1] and _is_simultaneous_exact_hit(
            candidate[2],
            candidate[3],
            candidate[4],
            candidate[5],
            candidate[6],
            first_time,
            first_x,
            first_y,
            first_position_budget,
            first_time_budget,
        ):
            candidate_node0 = facet_node_ids[facet_id, 0]
            candidate_node1 = facet_node_ids[facet_id, 1]
            if common_node0 != candidate_node0 and common_node0 != candidate_node1:
                common_node0 = -1
            if common_node1 != candidate_node0 and common_node1 != candidate_node1:
                common_node1 = -1
            if common_node0 < 0 and common_node1 < 0:
                return True, 0, 0.0, 0.0, 0.0
            count += 1
            position_budget = max(position_budget, candidate[5])
            time_budget = max(time_budget, candidate[6])
            residual = max(residual, candidate[7])
    return False, count, position_budget, time_budget, residual


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _locate_exact_event_row(
    axisymmetric_rz: bool,
    path_kind: np.uint8,
    start_x: float,
    start_y: float,
    velocity_x: float,
    velocity_y: float,
    start_time_s: float,
    target_time_s: float,
    departing_facet_id: int,
    linear_x: float,
    linear_y: float,
    quadratic_x: float,
    quadratic_y: float,
    path_bound_x: float,
    path_bound_y: float,
    speed_m_s: float,
    parameter_speed_m: float,
    broad_begin: int,
    broad_end: int,
    broad_candidates: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    facet_node_ids: Int64Array,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> tuple[np.uint8, int, float, float, float, float, float, float, float, float, int]:
    interval_s = target_time_s - start_time_s
    wall = _find_first_exact_wall(
        path_kind,
        start_x,
        start_y,
        linear_x,
        linear_y,
        quadratic_x,
        quadratic_y,
        path_bound_x,
        path_bound_y,
        speed_m_s,
        parameter_speed_m,
        start_time_s,
        interval_s,
        departing_facet_id,
        broad_begin,
        broad_end,
        broad_candidates,
        geometry_bbox_diagonal_m,
        facet_start_m,
        facet_end_m,
        facet_length_m,
        geometry_rtol,
        roundoff_ulps,
    )
    if wall[0]:
        return EXACT_STATUS_FAILURE, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    wall_found = wall[1]
    count = 0
    position_budget = wall[6]
    time_budget = wall[7]
    residual = wall[8]
    if wall_found:
        reduced = _reduce_simultaneous_exact_hits(
            path_kind,
            start_x,
            start_y,
            linear_x,
            linear_y,
            quadratic_x,
            quadratic_y,
            path_bound_x,
            path_bound_y,
            speed_m_s,
            parameter_speed_m,
            start_time_s,
            interval_s,
            departing_facet_id,
            broad_begin,
            broad_end,
            broad_candidates,
            wall[2],
            wall[3],
            wall[4],
            wall[5],
            wall[6],
            wall[7],
            wall[8],
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_length_m,
            facet_node_ids,
            geometry_rtol,
            roundoff_ulps,
        )
        if reduced[0]:
            return EXACT_STATUS_FAILURE, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        count, position_budget, time_budget, residual = (
            reduced[1],
            reduced[2],
            reduced[3],
            reduced[4],
        )
    axis = _exact_axis_hit(
        axisymmetric_rz,
        start_x,
        start_y,
        velocity_x,
        velocity_y,
        start_time_s,
        target_time_s,
    )
    if axis[0]:
        return EXACT_STATUS_FAILURE, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    if axis[1] and (not wall_found or wall[3] > axis[2] + time_budget):
        return (
            EXACT_STATUS_AXIS,
            -1,
            axis[2],
            axis[3],
            axis[4],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
        )
    if wall_found:
        return (
            EXACT_STATUS_WALL,
            wall[2],
            wall[3],
            wall[4],
            wall[5],
            position_budget,
            time_budget,
            residual,
            wall[6],
            wall[7],
            count,
        )
    return EXACT_STATUS_CLEAR, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _locate_exact_events_kernel(
    axisymmetric_rz: bool,
    path_kind: UInt8Array,
    start_position_m: FloatArray,
    velocity_m_s: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    departing_facet_id: Int64Array,
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
    path_position_bound_m: FloatArray,
    speed_m_s: FloatArray,
    parameter_speed_m: FloatArray,
    ready: NDArray[np.bool_],
    broad_offsets: Int64Array,
    broad_candidates: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    facet_node_ids: Int64Array,
    geometry_rtol: float,
    roundoff_ulps: int,
    status: UInt8Array,
    failure_reason: UInt8Array,
    event_time_s: FloatArray,
    hit_position_m: FloatArray,
    primary_facet_id: Int64Array,
    normal: FloatArray,
    position_budget_m: FloatArray,
    time_budget_s: FloatArray,
    localization_residual_m: FloatArray,
    first_position_budget_m: FloatArray,
    first_time_budget_s: FloatArray,
    simultaneous_count: Int64Array,
) -> None:
    for row in range(path_kind.size):
        if failure_reason[row] != EXACT_FAILURE_NONE:
            status[row] = EXACT_STATUS_FAILURE
            continue
        if not ready[row]:
            axis = _exact_axis_hit(
                axisymmetric_rz,
                start_position_m[row, 0],
                start_position_m[row, 1],
                velocity_m_s[row, 0],
                velocity_m_s[row, 1],
                start_time_s[row],
                target_time_s[row],
            )
            if axis[0]:
                status[row] = EXACT_STATUS_FAILURE
                failure_reason[row] = EXACT_FAILURE_INDETERMINATE_EVENT
            elif axis[1]:
                status[row] = EXACT_STATUS_AXIS
                event_time_s[row] = axis[2]
                hit_position_m[row, 0] = axis[3]
                hit_position_m[row, 1] = axis[4]
            continue
        event = _locate_exact_event_row(
            axisymmetric_rz,
            path_kind[row],
            start_position_m[row, 0],
            start_position_m[row, 1],
            velocity_m_s[row, 0],
            velocity_m_s[row, 1],
            start_time_s[row],
            target_time_s[row],
            departing_facet_id[row],
            linear_displacement_m[row, 0],
            linear_displacement_m[row, 1],
            quadratic_displacement_m[row, 0],
            quadratic_displacement_m[row, 1],
            path_position_bound_m[row, 0],
            path_position_bound_m[row, 1],
            speed_m_s[row],
            parameter_speed_m[row],
            broad_offsets[row],
            broad_offsets[row + 1],
            broad_candidates,
            geometry_bbox_diagonal_m,
            facet_start_m,
            facet_end_m,
            facet_length_m,
            facet_node_ids,
            geometry_rtol,
            roundoff_ulps,
        )
        status[row] = event[0]
        primary_facet_id[row] = event[1]
        event_time_s[row] = event[2]
        hit_position_m[row, 0] = event[3]
        hit_position_m[row, 1] = event[4]
        position_budget_m[row] = event[5]
        time_budget_s[row] = event[6]
        localization_residual_m[row] = event[7]
        first_position_budget_m[row] = event[8]
        first_time_budget_s[row] = event[9]
        simultaneous_count[row] = event[10]
        if event[0] == EXACT_STATUS_WALL:
            normal[row, 0] = facet_normal[event[1], 0]
            normal[row, 1] = facet_normal[event[1], 1]
        elif event[0] == EXACT_STATUS_FAILURE:
            failure_reason[row] = EXACT_FAILURE_INDETERMINATE_EVENT


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _fill_exact_event_candidates_kernel(
    path_kind: UInt8Array,
    start_position_m: FloatArray,
    start_time_s: FloatArray,
    target_time_s: FloatArray,
    departing_facet_id: Int64Array,
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
    path_position_bound_m: FloatArray,
    speed_m_s: FloatArray,
    parameter_speed_m: FloatArray,
    broad_offsets: Int64Array,
    broad_candidates: Int64Array,
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_length_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
    status: UInt8Array,
    first_time_s: FloatArray,
    first_position_m: FloatArray,
    first_position_budget_m: FloatArray,
    first_time_budget_s: FloatArray,
    candidate_offsets: Int64Array,
    candidate_facet_ids: Int64Array,
) -> None:
    for row in range(path_kind.size):
        if status[row] != EXACT_STATUS_WALL:
            continue
        write = candidate_offsets[row]
        interval_s = target_time_s[row] - start_time_s[row]
        for offset in range(broad_offsets[row], broad_offsets[row + 1]):
            facet_id = broad_candidates[offset]
            if facet_id == departing_facet_id[row]:
                continue
            hit = _exact_facet_hit(
                path_kind[row],
                facet_id,
                start_position_m[row, 0],
                start_position_m[row, 1],
                linear_displacement_m[row, 0],
                linear_displacement_m[row, 1],
                quadratic_displacement_m[row, 0],
                quadratic_displacement_m[row, 1],
                path_position_bound_m[row, 0],
                path_position_bound_m[row, 1],
                speed_m_s[row],
                parameter_speed_m[row],
                start_time_s[row],
                interval_s,
                geometry_bbox_diagonal_m,
                facet_start_m,
                facet_end_m,
                facet_length_m,
                geometry_rtol,
                roundoff_ulps,
            )
            if hit[1] and _is_simultaneous_exact_hit(
                hit[2],
                hit[3],
                hit[4],
                hit[5],
                hit[6],
                first_time_s[row],
                first_position_m[row, 0],
                first_position_m[row, 1],
                first_position_budget_m[row],
                first_time_budget_s[row],
            ):
                candidate_facet_ids[write] = facet_id
                write += 1


def _require_certified_quadratic_departure(
    geometry: PreparedGeometry,
    facet_id: int | None,
    start_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    speed_m_s: float,
    start_time_s: float,
    interval_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> None:
    """Verify that one constant-acceleration path moves monotonically inward."""

    if facet_id is None:
        return
    if not 0 <= facet_id < geometry.facet_count:
        raise ValueError("certified departing facet ID is outside the geometry")
    facet_start = geometry.facet_start_m[facet_id]
    facet_end = geometry.facet_end_m[facet_id]
    edge = facet_end - facet_start
    length = float(geometry.facet_length_m[facet_id])
    tangent = edge / length
    offset = start_m - facet_start
    distance_along = math.fsum(
        (float(offset[0]) * float(tangent[0]), float(offset[1]) * float(tangent[1]))
    )
    parameter = distance_along / length
    projection = facet_start + parameter * edge
    normal = geometry.facet_normal[facet_id]
    signed_distance = math.fsum(
        (
            float(offset[0]) * float(normal[0]),
            float(offset[1]) * float(normal[1]),
        )
    )
    line_residual = math.hypot(
        float(start_m[0] - projection[0]),
        float(start_m[1] - projection[1]),
    )
    budget = resolve_event_budget(
        facet_length_m=length,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=start_m,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=start_time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    endpoint_margin = budget.position_m / length
    if not math.isfinite(signed_distance) or signed_distance > budget.position_m:
        raise EventLocationError("quadratic departure is outside its source supporting line")
    if abs(signed_distance) <= budget.position_m and (
        not math.isfinite(parameter)
        or line_residual > budget.position_m
        or parameter <= endpoint_margin
        or parameter >= 1.0 - endpoint_margin
    ):
        raise EventLocationError("quadratic departure does not start inside its source facet")

    normal_speed = math.fsum(
        (
            float(velocity_m_s[0]) * float(normal[0]),
            float(velocity_m_s[1]) * float(normal[1]),
        )
    )
    velocity_scale = math.hypot(float(velocity_m_s[0]), float(velocity_m_s[1]))
    velocity_margin = float(roundoff_ulps) * _FLOAT64_EPS * velocity_scale
    if normal_speed != 0.0 and normal_speed >= -velocity_margin:
        raise EventLocationError("quadratic departure velocity is not provably inward")
    normal_acceleration = math.fsum(
        (
            float(acceleration_m_s2[0]) * float(normal[0]),
            float(acceleration_m_s2[1]) * float(normal[1]),
        )
    )
    acceleration_scale = math.hypot(
        float(acceleration_m_s2[0]),
        float(acceleration_m_s2[1]),
    )
    acceleration_margin = float(roundoff_ulps) * _FLOAT64_EPS * acceleration_scale
    if normal_acceleration >= -acceleration_margin:
        raise EventLocationError("quadratic departure acceleration is not provably inward")


def classify_event_point(
    geometry: PreparedGeometry,
    position_m: FloatArray,
    *,
    speed_m_s: float,
    interval_s: float,
    time_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
    volume_containment: bool | None = None,
) -> PointClassification:
    """Classify a point with broad-phase padding and facet-specific budgets."""

    position = _finite_point(position_m, "position_m")
    if geometry.facet_count == 0:
        raise ValueError("event point classification requires at least one boundary facet")
    padding = _conservative_path_padding(
        geometry,
        position,
        position,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )

    def facet_budget(facet_id: int) -> float:
        return resolve_event_budget(
            facet_length_m=float(geometry.facet_length_m[facet_id]),
            geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
            position_m=position,
            speed_m_s=speed_m_s,
            interval_s=interval_s,
            time_s=time_s,
            geometry_rtol=geometry_rtol,
            roundoff_ulps=roundoff_ulps,
        ).position_m

    return classify_point(
        geometry,
        position,
        candidate_padding_m=padding,
        facet_position_budget_m=facet_budget,
        volume_containment=volume_containment,
    )


def inspect_rk4_piece(
    geometry: PreparedGeometry,
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    *,
    start_time_s: float,
    end_time_s: float,
    root_interval_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
    certify_start_contact_departure: bool = False,
    chord_deviation_bound_m: FloatArray | None = None,
) -> Rk4PieceDecision:
    """Certify a clear RK4 tube or request refinement/localize its first exit.

    A tube disjoint from every padded facet AABB, or proved separated from a
    candidate's supporting line, is a no-hit piece.  A tube that may still
    touch the boundary is never accepted as clear.  It is bisected until an
    inside-to-outside chord is both spatially and temporally bounded by a
    certified transverse bracket or by the full tube.  Grazing and
    cross-return ambiguity therefore fail closed when the caller exhausts its
    refinement budget.  Another curved method may supply its own certified
    componentwise chord bound; omitting it preserves the RK4 calculation.
    """

    piece = _prepare_rk4_piece(
        start_position_m,
        end_position_m,
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        start_time_s,
        end_time_s,
        root_interval_s,
        chord_deviation_bound_m,
    )
    if geometry.facet_count == 0:
        return Rk4PieceDecision("clear", None, 0)
    candidates = _query_rk4_piece_candidates(
        geometry,
        piece,
        geometry_rtol,
        roundoff_ulps,
    )
    departure_certified = False
    if certify_start_contact_departure and candidates.size:
        departing = np.asarray(
            [
                _certify_rk4_departure(
                    geometry,
                    piece,
                    int(facet_id),
                    geometry_rtol,
                    roundoff_ulps,
                )
                for facet_id in candidates
            ],
            dtype=np.bool_,
        )
        departure_certified = bool(departing.any())
        candidates = candidates[~departing]
    if not candidates.size:
        return Rk4PieceDecision("clear", None, 0, departure_certified)
    if candidates.size != 1 or _rk4_piece_is_certified_split(
        geometry,
        piece,
        candidates,
        geometry_rtol,
        roundoff_ulps,
    ):
        return Rk4PieceDecision(
            "split",
            None,
            int(candidates.size),
            departure_certified,
        )
    return _localize_rk4_piece_exit(
        geometry,
        piece,
        candidates,
        geometry_rtol,
        roundoff_ulps,
        departure_certified,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _preclassification_query_bounds(
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    row: int,
    geometry_bbox_diagonal_m: float,
    maximum_facet_length_m: float,
    geometry_rtol: float,
    roundoff_factor: float,
) -> tuple[bool, float, float, float, float, float]:
    position_abs_x = max(
        abs(position_lower_m[row, 0]),
        abs(position_upper_m[row, 0]),
    )
    position_abs_y = max(
        abs(position_lower_m[row, 1]),
        abs(position_upper_m[row, 1]),
    )
    position_scale_m = max(
        1.0,
        geometry_bbox_diagonal_m,
        maximum_facet_length_m,
        position_abs_x + position_abs_y,
    )
    padding_m = np.nextafter(
        max(
            geometry_rtol * maximum_facet_length_m,
            roundoff_factor * position_scale_m,
        ),
        np.inf,
    )
    query_lower_x = np.nextafter(position_lower_m[row, 0] - padding_m, -np.inf)
    query_lower_y = np.nextafter(position_lower_m[row, 1] - padding_m, -np.inf)
    query_upper_x = np.nextafter(position_upper_m[row, 0] + padding_m, np.inf)
    query_upper_y = np.nextafter(position_upper_m[row, 1] + padding_m, np.inf)
    finite = (
        np.isfinite(query_lower_x)
        and np.isfinite(query_lower_y)
        and np.isfinite(query_upper_x)
        and np.isfinite(query_upper_y)
    )
    return finite, padding_m, query_lower_x, query_lower_y, query_upper_x, query_upper_y


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _preclassification_aabb_disjoint(
    query_lower_x: float,
    query_lower_y: float,
    query_upper_x: float,
    query_upper_y: float,
    candidate_lower_x: float,
    candidate_lower_y: float,
    candidate_upper_x: float,
    candidate_upper_y: float,
) -> bool:
    return (
        query_upper_x < candidate_lower_x
        or query_upper_y < candidate_lower_y
        or candidate_upper_x < query_lower_x
        or candidate_upper_y < query_lower_y
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _preclassification_straddles_supporting_line(
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    row: int,
    padding_m: float,
    facet_start_m: FloatArray,
    facet_normal: FloatArray,
    facet_id: int,
) -> bool:
    normal_x = facet_normal[facet_id, 0]
    normal_y = facet_normal[facet_id, 1]
    minimum_x = position_lower_m[row, 0] if normal_x >= 0.0 else position_upper_m[row, 0]
    minimum_y = position_lower_m[row, 1] if normal_y >= 0.0 else position_upper_m[row, 1]
    maximum_x = position_upper_m[row, 0] if normal_x >= 0.0 else position_lower_m[row, 0]
    maximum_y = position_upper_m[row, 1] if normal_y >= 0.0 else position_lower_m[row, 1]
    signed_minimum_m = (minimum_x - facet_start_m[facet_id, 0]) * normal_x + (
        minimum_y - facet_start_m[facet_id, 1]
    ) * normal_y
    signed_maximum_m = (maximum_x - facet_start_m[facet_id, 0]) * normal_x + (
        maximum_y - facet_start_m[facet_id, 1]
    ) * normal_y
    if not np.isfinite(signed_minimum_m) or not np.isfinite(signed_maximum_m):
        return False
    straddle_margin_m = 4.0 * padding_m
    return signed_minimum_m < -straddle_margin_m and signed_maximum_m > straddle_margin_m


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _preclassification_candidate_flags(
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    row: int,
    padding_m: float,
    query_lower_x: float,
    query_lower_y: float,
    query_upper_x: float,
    query_upper_y: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    bvh_facet_id: Int64Array,
    bvh_lower_m: FloatArray,
    bvh_upper_m: FloatArray,
    bvh_left: Int64Array,
    bvh_right: Int64Array,
    bvh_begin: Int64Array,
    bvh_end: Int64Array,
    stack: Int64Array,
) -> tuple[bool, bool]:
    stack_size = 1
    stack[0] = 0
    broad_candidate = False
    certified_straddling_candidate = False
    while stack_size:
        stack_size -= 1
        node = stack[stack_size]
        disjoint = _preclassification_aabb_disjoint(
            query_lower_x,
            query_lower_y,
            query_upper_x,
            query_upper_y,
            bvh_lower_m[node, 0],
            bvh_lower_m[node, 1],
            bvh_upper_m[node, 0],
            bvh_upper_m[node, 1],
        )
        if disjoint:
            continue
        left = bvh_left[node]
        if left >= 0:
            stack[stack_size] = bvh_right[node]
            stack_size += 1
            stack[stack_size] = left
            stack_size += 1
            continue
        for offset in range(bvh_begin[node], bvh_end[node]):
            facet_id = bvh_facet_id[offset]
            facet_lower_x = min(facet_start_m[facet_id, 0], facet_end_m[facet_id, 0])
            facet_lower_y = min(facet_start_m[facet_id, 1], facet_end_m[facet_id, 1])
            facet_upper_x = max(facet_start_m[facet_id, 0], facet_end_m[facet_id, 0])
            facet_upper_y = max(facet_start_m[facet_id, 1], facet_end_m[facet_id, 1])
            facet_disjoint = _preclassification_aabb_disjoint(
                query_lower_x,
                query_lower_y,
                query_upper_x,
                query_upper_y,
                facet_lower_x,
                facet_lower_y,
                facet_upper_x,
                facet_upper_y,
            )
            if facet_disjoint:
                continue
            broad_candidate = True
            if _preclassification_straddles_supporting_line(
                position_lower_m,
                position_upper_m,
                row,
                padding_m,
                facet_start_m,
                facet_normal,
                facet_id,
            ):
                certified_straddling_candidate = True
    return broad_candidate, certified_straddling_candidate


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _preclassification_split_proved(
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    row: int,
    interval_s: float,
    padding_m: float,
) -> bool:
    velocity_span_x = velocity_upper_m_s[row, 0] - velocity_lower_m_s[row, 0]
    velocity_span_y = velocity_upper_m_s[row, 1] - velocity_lower_m_s[row, 1]
    deviation_radius_m = math.hypot(
        interval_s * velocity_span_x,
        interval_s * velocity_span_y,
    )
    tube_diameter_m = math.hypot(
        position_upper_m[row, 0] - position_lower_m[row, 0],
        position_upper_m[row, 1] - position_lower_m[row, 1],
    )
    if not math.isfinite(deviation_radius_m) or not math.isfinite(tube_diameter_m):
        return False
    return deviation_radius_m > padding_m and tube_diameter_m > padding_m


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _preclassification_speed_is_finite(
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    row: int,
) -> bool:
    velocity_abs_x = max(
        abs(velocity_lower_m_s[row, 0]),
        abs(velocity_upper_m_s[row, 0]),
    )
    velocity_abs_y = max(
        abs(velocity_lower_m_s[row, 1]),
        abs(velocity_upper_m_s[row, 1]),
    )
    speed_upper_m_s = np.nextafter(math.hypot(velocity_abs_x, velocity_abs_y), np.inf)
    return math.isfinite(speed_upper_m_s)


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def preclassify_rk4_piece_batch(
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: FloatArray,
    end_time_s: float,
    root_interval_s: FloatArray,
    certify_start_contact_departure: NDArray[np.bool_],
    geometry_bbox_diagonal_m: float,
    facet_start_m: FloatArray,
    facet_end_m: FloatArray,
    facet_normal: FloatArray,
    facet_length_m: FloatArray,
    bvh_facet_id: Int64Array,
    bvh_lower_m: FloatArray,
    bvh_upper_m: FloatArray,
    bvh_left: Int64Array,
    bvh_right: Int64Array,
    bvh_begin: Int64Array,
    bvh_end: Int64Array,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> NDArray[np.uint8]:
    """Certify common clear/split rows; zero means use the full scalar locator."""

    count = start_time_s.size
    result = np.zeros(count, dtype=np.uint8)
    if bvh_left.size == 0:
        result[:] = 1
        return result
    maximum_facet_length_m = 0.0
    for facet_id in range(facet_length_m.size):
        maximum_facet_length_m = max(maximum_facet_length_m, facet_length_m[facet_id])
    stack = np.empty(bvh_left.size, dtype=np.int64)
    roundoff_factor = float(roundoff_ulps) * _FLOAT64_EPS
    for row in range(count):
        interval_s = end_time_s - start_time_s[row]
        invalid_for_certificate = (
            interval_s <= 0.0
            or root_interval_s[row] <= 0.0
            or not _preclassification_speed_is_finite(
                velocity_lower_m_s,
                velocity_upper_m_s,
                row,
            )
        )
        if certify_start_contact_departure[row] or invalid_for_certificate:
            continue
        finite, padding_m, lower_x, lower_y, upper_x, upper_y = _preclassification_query_bounds(
            position_lower_m,
            position_upper_m,
            row,
            geometry_bbox_diagonal_m,
            maximum_facet_length_m,
            geometry_rtol,
            roundoff_factor,
        )
        if not finite:
            continue
        broad, straddling = _preclassification_candidate_flags(
            position_lower_m,
            position_upper_m,
            row,
            padding_m,
            lower_x,
            lower_y,
            upper_x,
            upper_y,
            facet_start_m,
            facet_end_m,
            facet_normal,
            bvh_facet_id,
            bvh_lower_m,
            bvh_upper_m,
            bvh_left,
            bvh_right,
            bvh_begin,
            bvh_end,
            stack,
        )
        if not broad:
            result[row] = 1
        elif straddling and _preclassification_split_proved(
            position_lower_m,
            position_upper_m,
            velocity_lower_m_s,
            velocity_upper_m_s,
            row,
            interval_s,
            padding_m,
        ):
            result[row] = 2
    return result


def _rk4_piece_is_certified_split(
    geometry: PreparedGeometry,
    piece: _Rk4Piece,
    candidates: Int64Array,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    """Prove that neither transverse nor full-tube localization can yet pass."""

    deviation = piece.chord_deviation_m
    if deviation is None:
        return False
    position_bound = np.maximum(np.abs(piece.lower_m), np.abs(piece.upper_m))
    time_scale = max((piece.start_time_s, piece.end_time_s), key=abs)
    maximum_position_budget_m = 0.0
    maximum_time_budget_s = 0.0
    for facet_id_value in candidates:
        facet_id = int(facet_id_value)
        budget = resolve_event_budget(
            facet_length_m=float(geometry.facet_length_m[facet_id]),
            geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
            position_m=position_bound,
            speed_m_s=piece.speed_upper_m_s,
            interval_s=piece.root_interval_s,
            time_s=time_scale,
            geometry_rtol=geometry_rtol,
            roundoff_ulps=roundoff_ulps,
        )
        maximum_position_budget_m = max(maximum_position_budget_m, budget.position_m)
        maximum_time_budget_s = max(maximum_time_budget_s, budget.time_s)
    deviation_radius_m = math.hypot(float(deviation[0]), float(deviation[1]))
    if deviation_radius_m <= maximum_position_budget_m:
        return False
    span = piece.upper_m - piece.lower_m
    tube_diameter_m = math.hypot(float(span[0]), float(span[1]))
    return piece.interval_s > maximum_time_budget_s or tube_diameter_m > maximum_position_budget_m


def inspect_rk4_axis_piece(
    start_position_m: FloatArray,
    start_velocity_m_s: FloatArray,
    end_position_m: FloatArray,
    end_velocity_m_s: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    *,
    start_time_s: float,
    end_time_s: float,
    root_interval_s: float,
    geometry_bbox_diagonal_m: float,
    geometry_rtol: float,
    roundoff_ulps: int,
    chord_deviation_bound_m: FloatArray | None = None,
) -> Rk4AxisDecision:
    """Certify an RZ RK4 piece as clear, split it, or localize ``r = 0``.

    RK4 is integrated in a signed radial chart.  Resident states enter this
    function with nonnegative radius; a hit is the first transition into the
    negative half of that chart.  The coordinate seam is not a material
    boundary and therefore has its own decision and hit types.  Another curved
    method may supply its own certified componentwise chord bound.
    """

    piece = _prepare_rk4_piece(
        start_position_m,
        end_position_m,
        position_lower_m,
        position_upper_m,
        velocity_lower_m_s,
        velocity_upper_m_s,
        start_time_s,
        end_time_s,
        root_interval_s,
        chord_deviation_bound_m,
    )
    start_velocity = _finite_point(start_velocity_m_s, "start_velocity_m_s")
    end_velocity = _finite_point(end_velocity_m_s, "end_velocity_m_s")
    if bool(
        (start_velocity < piece.velocity_lower_m_s).any()
        or (start_velocity > piece.velocity_upper_m_s).any()
        or (end_velocity < piece.velocity_lower_m_s).any()
        or (end_velocity > piece.velocity_upper_m_s).any()
    ):
        raise ValueError("RK4 velocity enclosure does not contain its endpoints")
    if float(piece.start_m[0]) < 0.0:
        raise ValueError("resident RZ radius must be nonnegative")

    budget = _axis_event_budget(
        piece,
        geometry_bbox_diagonal_m,
        geometry_rtol,
        roundoff_ulps,
    )
    radial_direction, radial_speed_lower_m_s = _certified_radial_direction(
        piece,
        roundoff_ulps,
    )
    end_radius_m = float(piece.end_m[0])
    start_radial_velocity_m_s = float(start_velocity[0])
    end_radial_velocity_m_s = float(end_velocity[0])

    start_decision = _inspect_rk4_axis_start(
        piece,
        budget,
        start_radial_velocity_m_s,
        end_radius_m,
        end_radial_velocity_m_s,
        radial_direction,
    )
    if start_decision is not None:
        return start_decision
    if _rk4_axis_piece_is_clear(piece, end_radius_m, radial_direction, budget):
        return Rk4AxisDecision("clear", None)
    if end_radius_m > 0.0 or radial_direction != "inward":
        return Rk4AxisDecision("split", None)
    hit = _localize_rk4_axis_crossing(piece, budget, radial_speed_lower_m_s)
    if hit is None:
        return Rk4AxisDecision("split", None)
    return Rk4AxisDecision("hit", hit)


def _inspect_rk4_axis_start(
    piece: _Rk4Piece,
    budget: EventBudget,
    start_radial_velocity_m_s: float,
    end_radius_m: float,
    end_radial_velocity_m_s: float,
    radial_direction: Literal["inward", "outward", "indeterminate"],
) -> Rk4AxisDecision | None:
    """Resolve the right-continuous chart convention for an axis resident."""

    if float(piece.start_m[0]) != 0.0:
        return None
    if start_radial_velocity_m_s < 0.0:
        hit = AxisHit(piece.start_time_s, budget.position_m, budget.time_s, 0.0)
        return Rk4AxisDecision("hit", hit)
    if start_radial_velocity_m_s == 0.0 and end_radius_m == 0.0 and end_radial_velocity_m_s == 0.0:
        return Rk4AxisDecision("clear", None)
    if start_radial_velocity_m_s > 0.0 and radial_direction == "outward":
        return Rk4AxisDecision("clear", None)
    return Rk4AxisDecision("split", None)


def _rk4_axis_piece_is_clear(
    piece: _Rk4Piece,
    end_radius_m: float,
    radial_direction: Literal["inward", "outward", "indeterminate"],
    budget: EventBudget,
) -> bool:
    if float(piece.lower_m[0]) > budget.position_m:
        return True
    if radial_direction == "outward":
        return True
    return radial_direction == "inward" and end_radius_m > 0.0


def _axis_event_budget(
    piece: _Rk4Piece,
    geometry_bbox_diagonal_m: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> EventBudget:
    position_bound = np.maximum(np.abs(piece.lower_m), np.abs(piece.upper_m))
    return resolve_event_budget(
        facet_length_m=geometry_bbox_diagonal_m,
        geometry_bbox_diagonal_m=geometry_bbox_diagonal_m,
        position_m=position_bound,
        speed_m_s=piece.speed_upper_m_s,
        interval_s=piece.root_interval_s,
        time_s=max((piece.start_time_s, piece.end_time_s), key=abs),
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )


def _certified_radial_direction(
    piece: _Rk4Piece,
    roundoff_ulps: int,
) -> tuple[Literal["inward", "outward", "indeterminate"], float]:
    lower = float(piece.velocity_lower_m_s[0])
    upper = float(piece.velocity_upper_m_s[0])
    margin = _roundoff_margin(max(abs(lower), abs(upper)), roundoff_ulps)
    lower_with_roundoff = math.nextafter(lower - margin, -math.inf)
    upper_with_roundoff = math.nextafter(upper + margin, math.inf)
    if lower_with_roundoff > 0.0:
        return "outward", lower_with_roundoff
    if upper_with_roundoff < 0.0:
        return "inward", -upper_with_roundoff
    return "indeterminate", 0.0


def _localize_rk4_axis_crossing(
    piece: _Rk4Piece,
    budget: EventBudget,
    radial_speed_lower_m_s: float,
) -> AxisHit | None:
    if float(piece.end_m[0]) == 0.0:
        return AxisHit(piece.end_time_s, budget.position_m, budget.time_s, 0.0)
    chord = _prepare_rk4_locator_chord(piece)
    if chord is None or piece.chord_deviation_m is None:
        return None
    chord_velocity, _, chord_roundtrip_error = chord
    radial_velocity_m_s = float(chord_velocity[0])
    if radial_velocity_m_s >= 0.0:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        offset_s = float(piece.start_m[0]) / -radial_velocity_m_s
        hit_time_s = piece.start_time_s + offset_s
        effective_offset_s = hit_time_s - piece.start_time_s
        radial_product_m = effective_offset_s * radial_velocity_m_s
    if not math.isfinite(radial_product_m):
        return None
    try:
        radial_roundoff_m = abs(math.fsum((float(piece.start_m[0]), radial_product_m)))
    except (OverflowError, ValueError):
        return None
    if (
        not math.isfinite(hit_time_s)
        or hit_time_s <= piece.start_time_s
        or hit_time_s >= piece.end_time_s
    ):
        return None
    residual_m = _upper_nonnegative_sum(
        _upper_nonnegative_sum(
            float(piece.chord_deviation_m[0]),
            float(chord_roundtrip_error[0]),
        ),
        math.nextafter(radial_roundoff_m, math.inf),
    )
    if not math.isfinite(residual_m) or residual_m > budget.position_m:
        return None

    if radial_speed_lower_m_s <= 0.0:
        return None
    time_radius_s = _upper_nonnegative_quotient(residual_m, radial_speed_lower_m_s)
    if not math.isfinite(time_radius_s) or time_radius_s > budget.time_s:
        return None
    return AxisHit(hit_time_s, budget.position_m, budget.time_s, residual_m)


def _certify_rk4_departure(
    geometry: PreparedGeometry,
    piece: _Rk4Piece,
    facet_id: int,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    """Prove strict one-sided departure from one start-contact facet."""

    normal = geometry.facet_normal[facet_id]
    facet_start = geometry.facet_start_m[facet_id]
    facet_end = geometry.facet_end_m[facet_id]
    facet_length = float(geometry.facet_length_m[facet_id])
    budget = resolve_event_budget(
        facet_length_m=facet_length,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=piece.start_m,
        speed_m_s=piece.speed_upper_m_s,
        interval_s=piece.root_interval_s,
        time_s=piece.start_time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    signed_distance_lower_m, signed_distance_upper_m = _offset_dot_bounds(
        piece.start_m,
        facet_start,
        normal,
        roundoff_ulps,
    )
    if signed_distance_lower_m < -budget.position_m or signed_distance_upper_m > budget.position_m:
        return False

    edge = facet_end - facet_start
    tangent = edge / facet_length
    start_clearance_lower_m, _ = _offset_dot_bounds(
        piece.start_m,
        facet_start,
        tangent,
        roundoff_ulps,
    )
    end_clearance_lower_m, _ = _offset_dot_bounds(
        piece.start_m,
        facet_end,
        -tangent,
        roundoff_ulps,
    )
    if min(start_clearance_lower_m, end_clearance_lower_m) <= budget.position_m:
        return False

    velocity_extreme = np.where(normal >= 0.0, piece.velocity_upper_m_s, piece.velocity_lower_m_s)
    normal_velocity = math.fsum(
        (
            float(normal[0]) * float(velocity_extreme[0]),
            float(normal[1]) * float(velocity_extreme[1]),
        )
    )
    velocity_scale = _upper_nonnegative_sum(
        _upper_nonnegative_product(
            abs(float(normal[0])),
            max(
                abs(float(piece.velocity_lower_m_s[0])),
                abs(float(piece.velocity_upper_m_s[0])),
            ),
        ),
        _upper_nonnegative_product(
            abs(float(normal[1])),
            max(
                abs(float(piece.velocity_lower_m_s[1])),
                abs(float(piece.velocity_upper_m_s[1])),
            ),
        ),
    )
    margin = _roundoff_margin(velocity_scale, roundoff_ulps)
    normal_velocity_upper = math.nextafter(normal_velocity + margin, math.inf)
    return math.isfinite(normal_velocity_upper) and normal_velocity_upper < 0.0


def _prepare_rk4_piece(
    start_position_m: FloatArray,
    end_position_m: FloatArray,
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    start_time_s: float,
    end_time_s: float,
    root_interval_s: float,
    chord_deviation_bound_m: FloatArray | None,
) -> _Rk4Piece:
    start = _finite_point(start_position_m, "start_position_m")
    end = _finite_point(end_position_m, "end_position_m")
    lower = _finite_point(position_lower_m, "position_lower_m")
    upper = _finite_point(position_upper_m, "position_upper_m")
    velocity_lower = _finite_point(velocity_lower_m_s, "velocity_lower_m_s")
    velocity_upper = _finite_point(velocity_upper_m_s, "velocity_upper_m_s")
    if not all(math.isfinite(value) for value in (start_time_s, end_time_s, root_interval_s)):
        raise ValueError("RK4 event times must be finite")
    interval_s = end_time_s - start_time_s
    if interval_s <= 0.0 or root_interval_s <= 0.0:
        raise ValueError("RK4 event intervals must be positive")
    if bool((lower > upper).any()):
        raise ValueError("RK4 path enclosure lower bounds exceed upper bounds")
    if bool((velocity_lower > velocity_upper).any()):
        raise ValueError("RK4 velocity enclosure lower bounds exceed upper bounds")
    if bool((start < lower).any() or (start > upper).any()):
        raise ValueError("RK4 path enclosure does not contain its start")
    if bool((end < lower).any() or (end > upper).any()):
        raise ValueError("RK4 path enclosure does not contain its endpoint")
    velocity_abs = np.maximum(np.abs(velocity_lower), np.abs(velocity_upper))
    speed_upper = np.nextafter(
        math.hypot(float(velocity_abs[0]), float(velocity_abs[1])),
        math.inf,
    )
    if chord_deviation_bound_m is None:
        try:
            chord_deviation = curved_chord_deviation_bound(
                start,
                end,
                velocity_lower,
                velocity_upper,
                interval_s,
            )
        except ValueError:
            # The transverse certificate is optional.  If its outward arithmetic
            # cannot be represented, retain the established full-tube/split path.
            chord_deviation = None
    else:
        chord_deviation = _finite_point(
            chord_deviation_bound_m,
            "chord_deviation_bound_m",
        )
        if bool((chord_deviation < 0.0).any()):
            raise ValueError("chord_deviation_bound_m must be nonnegative")
    return _Rk4Piece(
        start,
        end,
        lower,
        upper,
        velocity_lower,
        velocity_upper,
        start_time_s,
        end_time_s,
        interval_s,
        root_interval_s,
        speed_upper,
        chord_deviation,
    )


def _query_rk4_piece_candidates(
    geometry: PreparedGeometry,
    piece: _Rk4Piece,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> Int64Array:
    position_bound = np.maximum(np.abs(piece.lower_m), np.abs(piece.upper_m))
    padding = resolve_event_budget(
        facet_length_m=float(np.max(geometry.facet_length_m)),
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=position_bound,
        speed_m_s=piece.speed_upper_m_s,
        interval_s=piece.root_interval_s,
        time_s=max((piece.start_time_s, piece.end_time_s), key=abs),
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    ).position_m
    with np.errstate(over="ignore", invalid="ignore"):
        query_lower = np.nextafter(piece.lower_m - padding, -np.inf)
        query_upper = np.nextafter(piece.upper_m + padding, np.inf)
    if not bool(np.isfinite(query_lower).all() and np.isfinite(query_upper).all()):
        raise EventLocationError("padded RK4 path enclosure exceeds float64 range")
    candidates = query_aabb_candidates(geometry, query_lower, query_upper)
    if not candidates.size:
        return candidates
    retained = np.asarray(
        [
            _rk4_tube_may_reach_supporting_line(
                geometry,
                piece,
                int(facet_id),
                position_bound,
                geometry_rtol,
                roundoff_ulps,
            )
            for facet_id in candidates
        ],
        dtype=np.bool_,
    )
    return candidates[retained]


def _rk4_tube_may_reach_supporting_line(
    geometry: PreparedGeometry,
    piece: _Rk4Piece,
    facet_id: int,
    position_bound_m: FloatArray,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> bool:
    """Retain a facet unless its supporting line is provably outside the tube."""

    normal = geometry.facet_normal[facet_id]
    origin = geometry.facet_start_m[facet_id]
    minimum_corner = np.where(normal >= 0.0, piece.lower_m, piece.upper_m)
    maximum_corner = np.where(normal >= 0.0, piece.upper_m, piece.lower_m)
    signed_lower_m, _ = _offset_dot_bounds(
        minimum_corner,
        origin,
        normal,
        roundoff_ulps,
    )
    _, signed_upper_m = _offset_dot_bounds(
        maximum_corner,
        origin,
        normal,
        roundoff_ulps,
    )
    budget_m = resolve_event_budget(
        facet_length_m=float(geometry.facet_length_m[facet_id]),
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=position_bound_m,
        speed_m_s=piece.speed_upper_m_s,
        interval_s=piece.root_interval_s,
        time_s=max((piece.start_time_s, piece.end_time_s), key=abs),
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    ).position_m
    return signed_lower_m <= budget_m and signed_upper_m >= -budget_m


def _localize_rk4_piece_exit(
    geometry: PreparedGeometry,
    piece: _Rk4Piece,
    candidates: Int64Array,
    geometry_rtol: float,
    roundoff_ulps: int,
    start_contact_departure_certified: bool,
) -> Rk4PieceDecision:
    candidate_count = int(candidates.size)
    endpoint_classification = classify_event_point(
        geometry,
        piece.end_m,
        speed_m_s=piece.speed_upper_m_s,
        interval_s=piece.root_interval_s,
        time_s=piece.end_time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    if endpoint_classification == "inside":
        return Rk4PieceDecision("split", None, candidate_count, start_contact_departure_certified)

    chord = _prepare_rk4_locator_chord(piece)
    if chord is None:
        return Rk4PieceDecision("split", None, candidate_count, start_contact_departure_certified)
    chord_velocity, chord_displacement, chord_roundtrip_error = chord
    try:
        hit = locate_ballistic_first_hit(
            geometry,
            piece.start_m,
            chord_velocity,
            start_time_s=piece.start_time_s,
            end_time_s=piece.end_time_s,
            geometry_rtol=geometry_rtol,
            roundoff_ulps=roundoff_ulps,
        )
    except EventLocationError:
        return Rk4PieceDecision("split", None, candidate_count, start_contact_departure_certified)
    candidate_ids = tuple(int(value) for value in candidates)
    if candidate_count != 1:
        return Rk4PieceDecision("split", None, candidate_count, start_contact_departure_certified)
    if hit is None:
        if endpoint_classification != "boundary":
            return Rk4PieceDecision(
                "split", None, candidate_count, start_contact_departure_certified
            )
        hit = _project_boundary_band_hit(
            geometry,
            candidate_ids[0],
            piece.end_m,
            piece.end_time_s,
            piece.speed_upper_m_s,
            piece.root_interval_s,
            geometry_rtol,
            roundoff_ulps,
        )
        if hit is None:
            return Rk4PieceDecision(
                "split", None, candidate_count, start_contact_departure_certified
            )
    elif len(hit.candidate_facet_ids) != 1 or candidate_ids != hit.candidate_facet_ids:
        return Rk4PieceDecision("split", None, candidate_count, start_contact_departure_certified)

    budget = resolve_event_budget(
        facet_length_m=float(geometry.facet_length_m[hit.facet_id]),
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=hit.position_m,
        speed_m_s=piece.speed_upper_m_s,
        interval_s=piece.root_interval_s,
        time_s=hit.time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    transverse_hit = _certify_transverse_rk4_hit(
        geometry,
        piece,
        hit,
        budget,
        roundoff_ulps,
        chord_displacement,
        chord_roundtrip_error,
    )
    if transverse_hit is not None:
        return Rk4PieceDecision(
            "hit", transverse_hit, candidate_count, start_contact_departure_certified
        )

    span = piece.upper_m - piece.lower_m
    tube_diameter = np.nextafter(
        math.hypot(float(span[0]), float(span[1])),
        math.inf,
    )
    if piece.interval_s > budget.time_s or tube_diameter > budget.position_m:
        return Rk4PieceDecision("split", None, candidate_count, start_contact_departure_certified)

    localized = BoundaryHit(
        time_s=hit.time_s,
        position_m=hit.position_m,
        facet_id=hit.facet_id,
        candidate_facet_ids=hit.candidate_facet_ids,
        normal=hit.normal,
        position_budget_m=budget.position_m,
        time_budget_s=budget.time_s,
        localization_residual_m=max(hit.localization_residual_m, tube_diameter),
    )
    return Rk4PieceDecision("hit", localized, candidate_count, start_contact_departure_certified)


def _prepare_rk4_locator_chord(
    piece: _Rk4Piece,
) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """Reproduce the ballistic locator chord and bound its endpoint bridge."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        endpoint_displacement = piece.end_m - piece.start_m
        chord_velocity = endpoint_displacement / piece.interval_s
        chord_displacement = piece.interval_s * chord_velocity
        chord_roundtrip_error = np.nextafter(
            np.abs(chord_displacement - endpoint_displacement),
            np.inf,
        )
    arrays = (chord_velocity, chord_displacement, chord_roundtrip_error)
    if not all(bool(np.isfinite(value).all()) for value in arrays):
        return None
    return chord_velocity, chord_displacement, chord_roundtrip_error


def _certify_transverse_rk4_hit(
    geometry: PreparedGeometry,
    piece: _Rk4Piece,
    hit: BoundaryHit,
    budget: EventBudget,
    roundoff_ulps: int,
    chord_displacement_m: FloatArray,
    chord_roundtrip_error_m: FloatArray,
) -> BoundaryHit | None:
    """Certify the time and position bracket of one outward chord crossing."""

    facet_start = geometry.facet_start_m[hit.facet_id]
    facet_end = geometry.facet_end_m[hit.facet_id]
    time_certificate = _transverse_time_certificate(
        piece,
        hit,
        budget,
        roundoff_ulps,
        facet_start,
        chord_displacement_m,
        chord_roundtrip_error_m,
    )
    if time_certificate is None:
        return None
    time_radius_s, component_deviation_m = time_certificate
    position_radius_m = _transverse_position_radius(
        piece,
        hit,
        time_radius_s,
        component_deviation_m,
        chord_displacement_m,
    )
    if not math.isfinite(position_radius_m) or position_radius_m > budget.position_m:
        return None

    clearance_required_m = _upper_nonnegative_sum(position_radius_m, budget.position_m)
    start_clearance_m = _distance_lower_bound(
        hit.position_m,
        facet_start,
        roundoff_ulps,
    )
    end_clearance_m = _distance_lower_bound(
        hit.position_m,
        facet_end,
        roundoff_ulps,
    )
    if min(start_clearance_m, end_clearance_m) <= clearance_required_m:
        return None

    return BoundaryHit(
        time_s=hit.time_s,
        position_m=hit.position_m,
        facet_id=hit.facet_id,
        candidate_facet_ids=hit.candidate_facet_ids,
        normal=hit.normal,
        position_budget_m=budget.position_m,
        time_budget_s=budget.time_s,
        localization_residual_m=position_radius_m,
    )


def _transverse_time_certificate(
    piece: _Rk4Piece,
    hit: BoundaryHit,
    budget: EventBudget,
    roundoff_ulps: int,
    facet_start_m: FloatArray,
    chord_displacement_m: FloatArray,
    chord_roundtrip_error_m: FloatArray,
) -> tuple[float, FloatArray] | None:
    deviation_m = piece.chord_deviation_m
    if deviation_m is None:
        return None

    normal = hit.normal
    _, start_signed_upper_m = _offset_dot_bounds(
        piece.start_m,
        facet_start_m,
        normal,
        roundoff_ulps,
    )
    end_signed_lower_m, _ = _offset_dot_bounds(
        piece.end_m,
        facet_start_m,
        normal,
        roundoff_ulps,
    )
    if start_signed_upper_m >= -budget.position_m or end_signed_lower_m <= budget.position_m:
        return None

    progress_terms = (
        float(chord_displacement_m[0]) * float(normal[0]),
        float(chord_displacement_m[1]) * float(normal[1]),
    )
    if not all(math.isfinite(value) for value in progress_terms):
        return None
    try:
        normal_progress_m = math.fsum(progress_terms)
    except OverflowError:
        return None
    product_scale = _upper_nonnegative_sum(
        _upper_nonnegative_product(
            abs(float(chord_displacement_m[0])),
            abs(float(normal[0])),
        ),
        _upper_nonnegative_product(
            abs(float(chord_displacement_m[1])),
            abs(float(normal[1])),
        ),
    )
    progress_error_m = _roundoff_margin(product_scale, roundoff_ulps)
    progress_lower_m = math.nextafter(normal_progress_m - progress_error_m, -math.inf)
    if not math.isfinite(progress_lower_m) or progress_lower_m <= 0.0:
        return None

    component_deviation_m = np.asarray(
        [
            _upper_nonnegative_sum(
                float(deviation_m[axis]),
                float(chord_roundtrip_error_m[axis]),
            )
            for axis in range(2)
        ],
        dtype=np.float64,
    )
    normal_deviation_m = _upper_nonnegative_sum(
        _upper_nonnegative_product(
            abs(float(normal[0])),
            float(component_deviation_m[0]),
        ),
        _upper_nonnegative_product(
            abs(float(normal[1])),
            float(component_deviation_m[1]),
        ),
    )
    normal_uncertainty_m = _upper_nonnegative_sum(
        normal_deviation_m,
        hit.localization_residual_m,
    )
    time_radius_s = _upper_nonnegative_quotient(
        _upper_nonnegative_product(piece.interval_s, normal_uncertainty_m),
        progress_lower_m,
    )
    bracket_lower_s = math.nextafter(hit.time_s - time_radius_s, -math.inf)
    bracket_upper_s = math.nextafter(hit.time_s + time_radius_s, math.inf)
    if (
        not math.isfinite(time_radius_s)
        or time_radius_s > budget.time_s
        or bracket_lower_s <= piece.start_time_s
        or bracket_upper_s > piece.end_time_s
    ):
        return None
    return time_radius_s, component_deviation_m


def _transverse_position_radius(
    piece: _Rk4Piece,
    hit: BoundaryHit,
    time_radius_s: float,
    component_deviation_m: FloatArray,
    chord_displacement_m: FloatArray,
) -> float:
    time_fraction = _upper_nonnegative_quotient(time_radius_s, piece.interval_s)
    chord_shift_m = np.asarray(
        [
            _upper_nonnegative_product(
                abs(float(chord_displacement_m[axis])),
                time_fraction,
            )
            for axis in range(2)
        ],
        dtype=np.float64,
    )
    component_error_m = np.asarray(
        [
            _upper_nonnegative_sum(
                float(component_deviation_m[axis]),
                float(chord_shift_m[axis]),
            )
            for axis in range(2)
        ],
        dtype=np.float64,
    )
    component_radius_m = math.nextafter(
        math.hypot(float(component_error_m[0]), float(component_error_m[1])),
        math.inf,
    )
    position_radius_m = _upper_nonnegative_sum(
        float(component_radius_m),
        hit.localization_residual_m,
    )
    return position_radius_m


def _upper_nonnegative_product(left: float, right: float) -> float:
    return math.nextafter(left * right, math.inf)


def _upper_nonnegative_sum(left: float, right: float) -> float:
    return math.nextafter(left + right, math.inf)


def _upper_nonnegative_quotient(value: float, divisor: float) -> float:
    return math.nextafter(value / divisor, math.inf)


def _roundoff_margin(scale: float, roundoff_ulps: int) -> float:
    return _upper_nonnegative_product(
        float(roundoff_ulps) * _FLOAT64_EPS,
        max(scale, _FLOAT64_EPS),
    )


def _offset_dot_bounds(
    point_m: FloatArray,
    origin_m: FloatArray,
    direction: FloatArray,
    roundoff_ulps: int,
) -> tuple[float, float]:
    with np.errstate(over="ignore", invalid="ignore"):
        offset = point_m - origin_m
        virtual_origin = point_m - offset
        virtual_point = offset + virtual_origin
        residual = (point_m - virtual_point) + (virtual_origin - origin_m)
    if not bool(np.isfinite(offset).all()) or not bool(np.isfinite(residual).all()):
        return -math.inf, math.inf
    terms = (
        float(offset[0]) * float(direction[0]),
        float(offset[1]) * float(direction[1]),
        float(residual[0]) * float(direction[0]),
        float(residual[1]) * float(direction[1]),
    )
    if not all(math.isfinite(term) for term in terms):
        return -math.inf, math.inf
    try:
        value = math.fsum(terms)
    except OverflowError:
        return -math.inf, math.inf
    scale = _upper_nonnegative_sum(
        _upper_nonnegative_product(
            _upper_nonnegative_sum(
                abs(float(offset[0])),
                abs(float(residual[0])),
            ),
            abs(float(direction[0])),
        ),
        _upper_nonnegative_product(
            _upper_nonnegative_sum(
                abs(float(offset[1])),
                abs(float(residual[1])),
            ),
            abs(float(direction[1])),
        ),
    )
    if not math.isfinite(value) or not math.isfinite(scale):
        return -math.inf, math.inf
    margin = _roundoff_margin(scale, roundoff_ulps)
    return (
        math.nextafter(value - margin, -math.inf),
        math.nextafter(value + margin, math.inf),
    )


def _distance_lower_bound(
    first_m: FloatArray,
    second_m: FloatArray,
    roundoff_ulps: int,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        separation = first_m - second_m
    if not bool(np.isfinite(separation).all()):
        return -math.inf
    distance = math.hypot(float(separation[0]), float(separation[1]))
    scale = _upper_nonnegative_sum(
        _upper_nonnegative_sum(
            abs(float(first_m[0])),
            abs(float(second_m[0])),
        ),
        _upper_nonnegative_sum(
            abs(float(first_m[1])),
            abs(float(second_m[1])),
        ),
    )
    if not math.isfinite(distance) or not math.isfinite(scale):
        return -math.inf
    margin = _roundoff_margin(scale, roundoff_ulps)
    return math.nextafter(distance - margin, -math.inf)


def _project_boundary_band_hit(
    geometry: PreparedGeometry,
    facet_id: int,
    endpoint_m: FloatArray,
    end_time_s: float,
    speed_upper_m_s: float,
    root_interval_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> BoundaryHit | None:
    """Represent entry into a facet's tolerance band by its closest point."""

    facet_start = geometry.facet_start_m[facet_id]
    facet_end = geometry.facet_end_m[facet_id]
    edge = facet_end - facet_start
    length = float(geometry.facet_length_m[facet_id])
    tangent = edge / length
    offset = endpoint_m - facet_start
    distance_along = math.fsum(
        (float(offset[0]) * float(tangent[0]), float(offset[1]) * float(tangent[1]))
    )
    parameter = min(max(distance_along / length, 0.0), 1.0)
    position = facet_start + parameter * edge
    separation = endpoint_m - position
    residual = math.hypot(float(separation[0]), float(separation[1]))
    budget = resolve_event_budget(
        facet_length_m=length,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=position,
        speed_m_s=speed_upper_m_s,
        interval_s=root_interval_s,
        time_s=end_time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    if not math.isfinite(residual) or residual > budget.position_m:
        return None
    normal = geometry.facet_normal[facet_id].copy()
    position.setflags(write=False)
    normal.setflags(write=False)
    return BoundaryHit(
        time_s=end_time_s,
        position_m=position,
        facet_id=facet_id,
        candidate_facet_ids=(facet_id,),
        normal=normal,
        position_budget_m=budget.position_m,
        time_budget_s=budget.time_s,
        localization_residual_m=residual,
    )


def _conservative_path_padding(
    geometry: PreparedGeometry,
    start_m: FloatArray,
    end_m: FloatArray,
    *,
    speed_m_s: float,
    interval_s: float,
    time_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> float:
    maximum_length = float(np.max(geometry.facet_length_m))
    representative_position = max(
        (start_m, end_m),
        key=lambda point: math.hypot(float(point[0]), float(point[1])),
    )
    padding = resolve_event_budget(
        facet_length_m=maximum_length,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=representative_position,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=time_s,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    return padding.position_m


def _prepare_quadratic_path(
    start_m: FloatArray,
    velocity_m_s: FloatArray,
    acceleration_m_s2: FloatArray,
    interval_s: float,
) -> _QuadraticPath:
    with np.errstate(over="ignore", invalid="ignore"):
        linear_displacement = interval_s * velocity_m_s
        quadratic_displacement = 0.5 * interval_s * (interval_s * acceleration_m_s2)
        end_position = start_m + linear_displacement + quadratic_displacement
        end_velocity = velocity_m_s + interval_s * acceleration_m_s2
        end_parameter_velocity = linear_displacement + 2.0 * quadratic_displacement
    finite_arrays = (
        linear_displacement,
        quadratic_displacement,
        end_position,
        end_velocity,
        end_parameter_velocity,
    )
    if not all(bool(np.isfinite(value).all()) for value in finite_arrays):
        raise EventLocationError("constant-acceleration path exceeds the finite float64 range")
    speed_m_s = max(
        math.hypot(float(velocity_m_s[0]), float(velocity_m_s[1])),
        math.hypot(float(end_velocity[0]), float(end_velocity[1])),
    )
    parameter_speed_m = max(
        math.hypot(float(linear_displacement[0]), float(linear_displacement[1])),
        math.hypot(float(end_parameter_velocity[0]), float(end_parameter_velocity[1])),
    )
    chord_deviation_m = 0.25 * math.hypot(
        float(quadratic_displacement[0]),
        float(quadratic_displacement[1]),
    )
    if not all(math.isfinite(value) for value in (speed_m_s, parameter_speed_m)):
        raise EventLocationError("constant-acceleration path speed is not finite")
    if not math.isfinite(chord_deviation_m):
        raise EventLocationError("constant-acceleration chord deviation is not finite")
    return _QuadraticPath(
        linear_displacement,
        quadratic_displacement,
        end_position,
        speed_m_s,
        parameter_speed_m,
        _quadratic_position_bound(start_m, linear_displacement, quadratic_displacement),
        chord_deviation_m,
    )


def _quadratic_position_bound(
    start_m: FloatArray,
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
) -> FloatArray:
    """Return a componentwise bound on every point of a quadratic path."""

    bound = np.empty(2, dtype=np.float64)
    for axis in range(2):
        start_value = float(start_m[axis])
        linear_value = float(linear_displacement_m[axis])
        quadratic_value = float(quadratic_displacement_m[axis])
        values = [start_value, _finite_sum(start_value, linear_value, quadratic_value)]
        if quadratic_value != 0.0:
            turning_parameter = -linear_value / (2.0 * quadratic_value)
            if math.isfinite(turning_parameter) and 0.0 < turning_parameter < 1.0:
                values.append(
                    _finite_sum(
                        start_value,
                        turning_parameter * linear_value,
                        turning_parameter * turning_parameter * quadratic_value,
                    )
                )
        bound[axis] = max(abs(value) for value in values)
    if not bool(np.isfinite(bound).all()):
        raise EventLocationError("constant-acceleration path bound is not finite")
    return bound


def _quadratic_path_padding(
    geometry: PreparedGeometry,
    path_position_bound_m: FloatArray,
    speed_m_s: float,
    start_time_s: float,
    interval_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> float:
    representative_time = max(
        (start_time_s, start_time_s + interval_s),
        key=abs,
    )
    return resolve_event_budget(
        facet_length_m=float(np.max(geometry.facet_length_m)),
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=path_position_bound_m,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=representative_time,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    ).position_m


def _intersect_quadratic_facet(
    geometry: PreparedGeometry,
    facet_id: int,
    start_m: FloatArray,
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
    path_position_bound_m: FloatArray,
    speed_m_s: float,
    parameter_speed_m: float,
    start_time_s: float,
    interval_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> _FacetHit | None:
    facet_start = geometry.facet_start_m[facet_id]
    facet_edge = geometry.facet_end_m[facet_id] - facet_start
    with np.errstate(over="ignore", invalid="ignore"):
        offset = start_m - facet_start
    if not bool(np.isfinite(facet_edge).all() and np.isfinite(offset).all()):
        raise EventLocationError("quadratic intersection coordinates exceed float64 range")
    facet_length = float(geometry.facet_length_m[facet_id])
    representative_time = max(
        (start_time_s, start_time_s + interval_s),
        key=abs,
    )
    path_padding_m = resolve_event_budget(
        facet_length_m=facet_length,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=path_position_bound_m,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=representative_time,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    ).position_m
    roots = _normalized_quadratic_roots(
        linear_displacement_m,
        quadratic_displacement_m,
        facet_edge,
        offset,
        facet_length,
        roundoff_ulps,
        path_padding_m,
    )
    hits: list[_FacetHit] = []
    for root in roots:
        hit = _quadratic_root_hit(
            geometry,
            facet_id,
            root,
            start_m,
            linear_displacement_m,
            quadratic_displacement_m,
            facet_start,
            facet_edge,
            facet_length,
            speed_m_s,
            parameter_speed_m,
            start_time_s,
            interval_s,
            geometry_rtol,
            roundoff_ulps,
        )
        if hit is not None:
            hits.append(hit)
    if not hits:
        return None
    return min(hits, key=lambda hit: hit.time_s)


def _normalized_quadratic_roots(
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
    facet_edge_m: FloatArray,
    offset_m: FloatArray,
    facet_length_m: float,
    roundoff_ulps: int,
    path_padding_m: float,
) -> tuple[float, ...]:
    linear_length = math.hypot(
        float(linear_displacement_m[0]),
        float(linear_displacement_m[1]),
    )
    quadratic_length = math.hypot(
        float(quadratic_displacement_m[0]),
        float(quadratic_displacement_m[1]),
    )
    local_scale = max(
        linear_length,
        quadratic_length,
        facet_length_m,
        abs(float(offset_m[0])),
        abs(float(offset_m[1])),
    )
    if not math.isfinite(local_scale) or local_scale <= 0.0:
        raise EventLocationError("quadratic intersection has no finite local scale")
    linear_local = linear_displacement_m / local_scale
    quadratic_local = quadratic_displacement_m / local_scale
    facet_local = facet_edge_m / local_scale
    offset_local = offset_m / local_scale
    constant = _cross(offset_local, facet_local)
    linear = _cross(linear_local, facet_local)
    quadratic = _cross(quadratic_local, facet_local)
    if not all(math.isfinite(value) for value in (constant, linear, quadratic)):
        raise EventLocationError("quadratic intersection coefficients are not finite")
    facet_local_length = math.hypot(float(facet_local[0]), float(facet_local[1]))
    return _solve_quadratic_coefficients(
        constant,
        linear,
        quadratic,
        facet_local_length,
        local_scale,
        roundoff_ulps,
        path_padding_m,
    )


def _solve_quadratic_coefficients(
    constant: float,
    linear: float,
    quadratic: float,
    facet_local_length: float,
    local_scale: float,
    roundoff_ulps: int,
    path_padding_m: float,
) -> tuple[float, ...]:
    if quadratic == 0.0:
        return _solve_degenerate_quadratic(
            constant,
            linear,
            facet_local_length,
            local_scale,
            path_padding_m,
        )

    squared_linear = linear * linear
    four_quadratic_constant = 4.0 * quadratic * constant
    discriminant = math.fsum((squared_linear, -four_quadratic_constant))
    discriminant_error = (
        float(roundoff_ulps)
        * _FLOAT64_EPS
        * max(squared_linear + abs(four_quadratic_constant), _FLOAT64_EPS)
    )
    if not math.isfinite(discriminant) or not math.isfinite(discriminant_error):
        raise EventLocationError("quadratic discriminant is not finite")
    if discriminant < -discriminant_error:
        return ()
    if abs(discriminant) <= discriminant_error:
        raise EventLocationError("quadratic boundary discriminant is indeterminate")

    square_root = math.sqrt(discriminant)
    stable_term = -0.5 * (linear + math.copysign(square_root, linear))
    if stable_term == 0.0 or not math.isfinite(stable_term):
        raise EventLocationError("quadratic roots are indeterminate")
    roots = (stable_term / quadratic, constant / stable_term)
    if not all(math.isfinite(root) for root in roots):
        raise EventLocationError("quadratic root is not finite")
    return tuple(sorted(roots))


def _solve_degenerate_quadratic(
    constant: float,
    linear: float,
    facet_local_length: float,
    local_scale: float,
    path_padding_m: float,
) -> tuple[float, ...]:
    if linear != 0.0:
        root = -constant / linear
        if not math.isfinite(root):
            raise EventLocationError("linearized quadratic root is not finite")
        return (root,)
    if not math.isfinite(facet_local_length) or facet_local_length <= 0.0:
        raise EventLocationError("normalized facet has no finite positive length")
    separation_m = abs(constant) / facet_local_length * local_scale
    if not math.isfinite(separation_m):
        raise EventLocationError("quadratic-to-facet separation is not finite")
    if separation_m <= path_padding_m:
        raise EventLocationError("collinear quadratic boundary intersection is indeterminate")
    return ()


def _quadratic_root_hit(
    geometry: PreparedGeometry,
    facet_id: int,
    root: float,
    start_m: FloatArray,
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
    facet_start_m: FloatArray,
    facet_edge_m: FloatArray,
    facet_length_m: float,
    speed_m_s: float,
    parameter_speed_m: float,
    start_time_s: float,
    interval_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> _FacetHit | None:
    bounded_root = min(max(root, 0.0), 1.0)
    path_position = _quadratic_position(
        start_m,
        linear_displacement_m,
        quadratic_displacement_m,
        bounded_root,
    )
    approximate_time = start_time_s + bounded_root * interval_s
    budget = resolve_event_budget(
        facet_length_m=facet_length_m,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=path_position,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=approximate_time,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    root_slack = min(
        budget.position_m / parameter_speed_m,
        budget.time_s / interval_s,
    )
    if root < -root_slack or root > 1.0 + root_slack:
        return None

    with np.errstate(over="ignore", invalid="ignore"):
        facet_offset = path_position - facet_start_m
    if not bool(np.isfinite(facet_offset).all()):
        raise EventLocationError("quadratic facet parameter exceeds float64 range")
    facet_tangent = facet_edge_m / facet_length_m
    facet_distance = math.fsum(
        (
            float(facet_offset[0]) * float(facet_tangent[0]),
            float(facet_offset[1]) * float(facet_tangent[1]),
        )
    )
    facet_parameter = facet_distance / facet_length_m
    if not math.isfinite(facet_parameter):
        raise EventLocationError("quadratic facet parameter is not finite")
    facet_slack = budget.position_m / facet_length_m
    if facet_parameter < -facet_slack or facet_parameter > 1.0 + facet_slack:
        return None
    if bounded_root <= 0.0:
        return None

    bounded_facet_parameter = min(max(facet_parameter, 0.0), 1.0)
    facet_position = facet_start_m + bounded_facet_parameter * facet_edge_m
    residual_vector = path_position - facet_position
    residual = math.hypot(float(residual_vector[0]), float(residual_vector[1]))
    if not math.isfinite(residual):
        raise EventLocationError("quadratic intersection residual is not finite")
    if residual > budget.position_m:
        raise EventLocationError("quadratic intersection residual exceeds its position budget")
    hit_time = start_time_s + bounded_root * interval_s
    if not math.isfinite(hit_time) or hit_time <= start_time_s:
        raise EventLocationError("positive boundary-hit time is unresolved at float64 precision")
    budget = resolve_event_budget(
        facet_length_m=facet_length_m,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=path_position,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=hit_time,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    return _FacetHit(
        facet_id,
        hit_time,
        path_position,
        budget,
        residual,
    )


def _quadratic_position(
    start_m: FloatArray,
    linear_displacement_m: FloatArray,
    quadratic_displacement_m: FloatArray,
    path_parameter: float,
) -> FloatArray:
    squared_parameter = path_parameter * path_parameter
    return np.asarray(
        [
            _finite_sum(
                float(start_m[axis]),
                path_parameter * float(linear_displacement_m[axis]),
                squared_parameter * float(quadratic_displacement_m[axis]),
            )
            for axis in range(2)
        ],
        dtype=np.float64,
    )


def _finite_sum(*values: float) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise EventLocationError("quadratic path sum exceeds float64 range") from error
    if not math.isfinite(result):
        raise EventLocationError("quadratic path sum is not finite")
    return result


def _intersect_facet(
    geometry: PreparedGeometry,
    facet_id: int,
    start_m: FloatArray,
    displacement_m: FloatArray,
    speed_m_s: float,
    start_time_s: float,
    interval_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> _FacetHit | None:
    facet_start = geometry.facet_start_m[facet_id]
    facet_edge = geometry.facet_end_m[facet_id] - facet_start
    offset = facet_start - start_m
    path_length = math.hypot(float(displacement_m[0]), float(displacement_m[1]))
    facet_length = float(geometry.facet_length_m[facet_id])
    path_padding_m = _facet_path_padding(
        geometry,
        facet_length,
        start_m,
        displacement_m,
        speed_m_s,
        start_time_s,
        interval_s,
        geometry_rtol,
        roundoff_ulps,
    )
    parameters = _normalized_line_parameters(
        displacement_m,
        facet_edge,
        offset,
        path_length,
        facet_length,
        roundoff_ulps,
        path_padding_m,
    )
    if parameters is None:
        return None
    path_parameter, facet_parameter = parameters

    bounded_path_parameter = min(max(path_parameter, 0.0), 1.0)
    approximate_position = start_m + bounded_path_parameter * displacement_m
    approximate_time = start_time_s + bounded_path_parameter * interval_s
    budget = resolve_event_budget(
        facet_length_m=facet_length,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=approximate_position,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=approximate_time,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    path_parameter_slack = min(
        budget.position_m / path_length,
        budget.time_s / interval_s,
    )
    if path_parameter < -path_parameter_slack or path_parameter > (1.0 + path_parameter_slack):
        return None
    if facet_parameter < -budget.position_m / facet_length or facet_parameter > (
        1.0 + budget.position_m / facet_length
    ):
        return None

    bounded_facet_parameter = min(max(facet_parameter, 0.0), 1.0)
    if bounded_path_parameter <= 0.0:
        return None
    path_position = start_m + bounded_path_parameter * displacement_m
    facet_position = facet_start + bounded_facet_parameter * facet_edge
    residual_vector = path_position - facet_position
    residual = math.hypot(float(residual_vector[0]), float(residual_vector[1]))
    if not math.isfinite(residual):
        raise EventLocationError("line intersection residual is not finite")
    if residual > budget.position_m:
        raise EventLocationError("line intersection residual exceeds its position budget")
    hit_time = start_time_s + bounded_path_parameter * interval_s
    if not math.isfinite(hit_time) or hit_time <= start_time_s:
        raise EventLocationError("positive boundary-hit time is unresolved at float64 precision")
    budget = resolve_event_budget(
        facet_length_m=facet_length,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=path_position,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=hit_time,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    )
    return _FacetHit(
        facet_id,
        hit_time,
        path_position,
        budget,
        residual,
    )


def _facet_path_padding(
    geometry: PreparedGeometry,
    facet_length_m: float,
    start_m: FloatArray,
    displacement_m: FloatArray,
    speed_m_s: float,
    start_time_s: float,
    interval_s: float,
    geometry_rtol: float,
    roundoff_ulps: int,
) -> float:
    end_m = start_m + displacement_m
    representative_position = max(
        (start_m, end_m),
        key=lambda point: math.hypot(float(point[0]), float(point[1])),
    )
    end_time_s = start_time_s + interval_s
    representative_time = max((start_time_s, end_time_s), key=abs)
    return resolve_event_budget(
        facet_length_m=facet_length_m,
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=representative_position,
        speed_m_s=speed_m_s,
        interval_s=interval_s,
        time_s=representative_time,
        geometry_rtol=geometry_rtol,
        roundoff_ulps=roundoff_ulps,
    ).position_m


def _normalized_line_parameters(
    displacement_m: FloatArray,
    facet_edge_m: FloatArray,
    offset_m: FloatArray,
    path_length_m: float,
    facet_length_m: float,
    roundoff_ulps: int,
    path_padding_m: float,
) -> tuple[float, float] | None:
    if not bool(np.isfinite(facet_edge_m).all() and np.isfinite(offset_m).all()):
        raise EventLocationError("local facet coordinates exceed the finite float64 range")
    local_scale = max(
        path_length_m,
        facet_length_m,
        abs(float(offset_m[0])),
        abs(float(offset_m[1])),
    )
    if not math.isfinite(local_scale) or local_scale <= 0.0:
        raise EventLocationError("line intersection has no finite local scale")
    path_local = displacement_m / local_scale
    facet_local = facet_edge_m / local_scale
    offset_local = offset_m / local_scale
    denominator = _cross(path_local, facet_local)
    product_scale = abs(float(path_local[0]) * float(facet_local[1])) + abs(
        float(path_local[1]) * float(facet_local[0])
    )
    denominator_error = float(roundoff_ulps) * _FLOAT64_EPS * max(product_scale, _FLOAT64_EPS)
    if abs(denominator) <= denominator_error:
        facet_local_length = math.hypot(float(facet_local[0]), float(facet_local[1]))
        if not math.isfinite(facet_local_length) or facet_local_length <= 0.0:
            raise EventLocationError("normalized facet has no finite positive length")
        line_separation = abs(_cross(offset_local, facet_local)) / facet_local_length * local_scale
        if not math.isfinite(line_separation):
            raise EventLocationError("near-parallel line separation is not finite")
        if line_separation <= path_padding_m:
            raise EventLocationError("near-parallel boundary intersection is indeterminate")
        return None
    path_parameter = _cross(offset_local, facet_local) / denominator
    facet_parameter = _cross(offset_local, path_local) / denominator
    if not math.isfinite(path_parameter) or not math.isfinite(facet_parameter):
        raise EventLocationError("line intersection parameters are not finite")
    return path_parameter, facet_parameter


def _select_earliest(geometry: PreparedGeometry, hits: list[_FacetHit]) -> BoundaryHit:
    first = min(hits, key=lambda hit: (hit.time_s, hit.facet_id))
    simultaneous = []
    for hit in hits:
        position_budget = max(first.budget.position_m, hit.budget.position_m)
        time_budget = max(first.budget.time_s, hit.budget.time_s)
        separation = hit.position_m - first.position_m
        position_difference = math.hypot(float(separation[0]), float(separation[1]))
        if abs(hit.time_s - first.time_s) <= time_budget and position_difference <= position_budget:
            simultaneous.append(hit)
    simultaneous.sort(key=lambda hit: hit.facet_id)
    candidate_ids = tuple(hit.facet_id for hit in simultaneous)
    position_budget = max(hit.budget.position_m for hit in simultaneous)
    time_budget = max(hit.budget.time_s for hit in simultaneous)
    residual = max(hit.residual_m for hit in simultaneous)
    position = first.position_m.copy()
    normal = geometry.facet_normal[first.facet_id].copy()
    position.setflags(write=False)
    normal.setflags(write=False)
    return BoundaryHit(
        time_s=first.time_s,
        position_m=position,
        facet_id=first.facet_id,
        candidate_facet_ids=candidate_ids,
        normal=normal,
        position_budget_m=position_budget,
        time_budget_s=time_budget,
        localization_residual_m=residual,
    )


def _cross(first: FloatArray, second: FloatArray) -> float:
    return math.fsum(
        (
            float(first[0]) * float(second[1]),
            -float(first[1]) * float(second[0]),
        )
    )


def _finite_point(value: FloatArray, label: str) -> FloatArray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"{label} must have shape (2,)")
    if not bool(np.isfinite(point).all()):
        raise ValueError(f"{label} must contain only finite values")
    return point
