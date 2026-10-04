"""Physical wall responses applied after geometric hit localization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type Int32Array = NDArray[np.int32]
type Int64Array = NDArray[np.int64]
type UInt8Array = NDArray[np.uint8]

BOUNDARY_ALGORITHM_REVISION = "point_wall_laws_v5"

_FLOAT64_EPS = float(np.finfo(np.float64).eps)

BOUNDARY_STATUS_OK = np.uint8(0)
BOUNDARY_STATUS_EMPTY_CANDIDATES = np.uint8(1)
BOUNDARY_STATUS_INVALID_REFERENCE = np.uint8(2)
BOUNDARY_STATUS_AMBIGUOUS_LAW = np.uint8(3)
BOUNDARY_STATUS_INDETERMINATE_POLICY = np.uint8(4)

BOUNDARY_LAW_STICK = np.uint8(1)
BOUNDARY_LAW_ESCAPE = np.uint8(2)
BOUNDARY_LAW_SPECULAR = np.uint8(3)
BOUNDARY_LAW_RESTITUTION = np.uint8(4)
BOUNDARY_LAW_PROBABILISTIC_STICK = np.uint8(5)
BOUNDARY_LAW_HOLD = np.uint8(6)

BOUNDARY_OUTCOME_NONE = np.uint8(0)
BOUNDARY_OUTCOME_STUCK = np.uint8(1)
BOUNDARY_OUTCOME_ESCAPED = np.uint8(2)
BOUNDARY_OUTCOME_REFLECTED = np.uint8(3)
BOUNDARY_OUTCOME_HELD = np.uint8(4)


class BoundaryLawError(ValueError):
    """A wall rule or simultaneous-candidate response is indeterminate."""


@dataclass(frozen=True, slots=True)
class BoundaryRule:
    """One prepared rule for a canonical boundary group."""

    group_id: int
    priority: int
    law_id: str
    normal_restitution: float | None = None
    tangential_restitution: float | None = None
    stick_probability: float | None = None
    otherwise_law_id: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedBoundaryRules:
    """Dense numeric wall-law table consumed by compiled event rows."""

    priority: Int64Array
    law: UInt8Array
    otherwise_law: UInt8Array
    normal_restitution: FloatArray
    tangential_restitution: FloatArray
    stick_probability: FloatArray


@dataclass(frozen=True, slots=True)
class BoundaryResponseBatch:
    """Columnar responses aligned with one CSR candidate row batch."""

    status: UInt8Array
    law: UInt8Array
    outcome: UInt8Array
    velocity_post_m_s: FloatArray
    primary_facet_id: Int64Array
    effective_normal: FloatArray
    remains_active: NDArray[np.bool_]


def prepare_boundary_rule(
    group_id: int,
    priority: int,
    law_id: str,
    parameters: Mapping[str, object],
) -> BoundaryRule:
    """Validate the deliberately small P07 wall-law catalog."""

    if group_id < 0 or priority < 0:
        raise BoundaryLawError("boundary group ID and priority must be nonnegative")
    if law_id in {"stick", "escape", "specular", "hold"}:
        if parameters:
            raise BoundaryLawError(f"{law_id} boundary law does not accept parameters")
        if law_id == "specular":
            return BoundaryRule(
                group_id,
                priority,
                law_id,
                normal_restitution=1.0,
                tangential_restitution=1.0,
            )
        return BoundaryRule(group_id, priority, law_id)
    if law_id == "restitution":
        normal, tangential = _restitution_parameters(parameters, "restitution")
        return BoundaryRule(
            group_id,
            priority,
            law_id,
            normal_restitution=normal,
            tangential_restitution=tangential,
        )
    if law_id == "probabilistic_stick":
        if set(parameters) != {"probability", "otherwise"}:
            raise BoundaryLawError("probabilistic_stick requires exactly probability and otherwise")
        probability = _probability(parameters["probability"])
        otherwise = parameters["otherwise"]
        if not isinstance(otherwise, Mapping):
            raise BoundaryLawError("probabilistic_stick.otherwise must be a law mapping")
        otherwise_law = otherwise.get("law")
        reflection_parameters = {key: value for key, value in otherwise.items() if key != "law"}
        if otherwise_law == "specular":
            if reflection_parameters:
                raise BoundaryLawError(
                    "probabilistic_stick.otherwise specular law does not accept parameters"
                )
            normal, tangential = 1.0, 1.0
        elif otherwise_law == "restitution":
            normal, tangential = _restitution_parameters(
                reflection_parameters,
                "probabilistic_stick.otherwise restitution",
            )
        else:
            raise BoundaryLawError(
                "probabilistic_stick.otherwise law must be specular or restitution"
            )
        return BoundaryRule(
            group_id,
            priority,
            law_id,
            normal_restitution=normal,
            tangential_restitution=tangential,
            stick_probability=probability,
            otherwise_law_id=otherwise_law,
        )
    raise BoundaryLawError(f"unsupported boundary law: {law_id}")


def prepare_boundary_rules(rules: Sequence[BoundaryRule]) -> PreparedBoundaryRules:
    """Pack one dense group-indexed rule table for the compiled wall phase."""

    priority = np.empty(len(rules), dtype=np.int64)
    law = np.empty(len(rules), dtype=np.uint8)
    otherwise_law = np.zeros(len(rules), dtype=np.uint8)
    normal = np.full(len(rules), np.nan, dtype=np.float64)
    tangential = np.full(len(rules), np.nan, dtype=np.float64)
    probability = np.full(len(rules), np.nan, dtype=np.float64)
    law_code = {
        "stick": BOUNDARY_LAW_STICK,
        "escape": BOUNDARY_LAW_ESCAPE,
        "specular": BOUNDARY_LAW_SPECULAR,
        "restitution": BOUNDARY_LAW_RESTITUTION,
        "probabilistic_stick": BOUNDARY_LAW_PROBABILISTIC_STICK,
        "hold": BOUNDARY_LAW_HOLD,
    }
    for group_id, rule in enumerate(rules):
        if rule.group_id != group_id or rule.priority < 0 or rule.law_id not in law_code:
            raise BoundaryLawError("boundary rules must be a dense validated group table")
        priority[group_id] = rule.priority
        law[group_id] = law_code[rule.law_id]
        if rule.law_id in {"specular", "restitution", "probabilistic_stick"}:
            if rule.normal_restitution is None or rule.tangential_restitution is None:
                raise BoundaryLawError("prepared reflection law lost its restitution coefficients")
            normal[group_id] = rule.normal_restitution
            tangential[group_id] = rule.tangential_restitution
        if rule.law_id == "probabilistic_stick":
            if rule.stick_probability is None or rule.otherwise_law_id not in {
                "specular",
                "restitution",
            }:
                raise BoundaryLawError("prepared probabilistic law lost its nested reflection law")
            probability[group_id] = rule.stick_probability
            otherwise_law[group_id] = law_code[rule.otherwise_law_id]
    arrays = (priority, law, otherwise_law, normal, tangential, probability)
    for array in arrays:
        array.setflags(write=False)
    return PreparedBoundaryRules(*arrays)


def resolve_boundary_responses_batch(
    rules: PreparedBoundaryRules,
    candidate_offsets: Int64Array,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    facet_normal: FloatArray,
    velocity_pre_m_s: FloatArray,
    uniform_draw: FloatArray,
    *,
    roundoff_ulps: int,
) -> BoundaryResponseBatch:
    """Resolve independent CSR hit rows without Python objects or callbacks."""

    offsets = np.asarray(candidate_offsets, dtype=np.int64)
    candidates = np.asarray(candidate_facet_ids, dtype=np.int64)
    velocity = np.asarray(velocity_pre_m_s, dtype=np.float64)
    draws = np.asarray(uniform_draw, dtype=np.float64)
    row_count = _validate_boundary_batch_inputs(
        offsets,
        candidates,
        boundary_id,
        group_id,
        facet_normal,
        velocity,
        draws,
        roundoff_ulps,
    )
    status = np.empty(row_count, dtype=np.uint8)
    law = np.empty(row_count, dtype=np.uint8)
    outcome = np.empty(row_count, dtype=np.uint8)
    velocity_post = np.empty((row_count, 2), dtype=np.float64)
    primary = np.empty(row_count, dtype=np.int64)
    effective_normal = np.empty((row_count, 2), dtype=np.float64)
    remains_active = np.empty(row_count, dtype=np.bool_)
    _resolve_boundary_responses_kernel(
        rules.priority,
        rules.law,
        rules.otherwise_law,
        rules.normal_restitution,
        rules.tangential_restitution,
        rules.stick_probability,
        offsets,
        candidates,
        boundary_id,
        group_id,
        facet_normal,
        velocity,
        draws,
        roundoff_ulps,
        status,
        law,
        outcome,
        velocity_post,
        primary,
        effective_normal,
        remains_active,
    )
    return BoundaryResponseBatch(
        status,
        law,
        outcome,
        velocity_post,
        primary,
        effective_normal,
        remains_active,
    )


def _validate_boundary_batch_inputs(
    offsets: Int64Array,
    candidates: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    facet_normal: FloatArray,
    velocity: FloatArray,
    draws: FloatArray,
    roundoff_ulps: int,
) -> int:
    row_count = offsets.size - 1
    if offsets.ndim != 1 or row_count < 0 or offsets[0] != 0:
        raise ValueError("candidate_offsets must be one nonempty CSR offset vector")
    if bool((np.diff(offsets) < 0).any()) or int(offsets[-1]) != candidates.size:
        raise ValueError("candidate_offsets do not delimit candidate_facet_ids")
    if velocity.shape != (row_count, 2) or draws.shape != (row_count,):
        raise ValueError("velocity and uniform draws must align with candidate rows")
    facet_count = int(boundary_id.size)
    aligned = (
        group_id.shape == (facet_count,)
        and facet_normal.shape == (facet_count, 2)
        and candidates.ndim == 1
    )
    if not aligned:
        raise ValueError("boundary arrays must have aligned facet dimensions")
    _validate_boundary_batch_values(velocity, facet_normal, draws, roundoff_ulps)
    return row_count


def _validate_boundary_batch_values(
    velocity: FloatArray,
    facet_normal: FloatArray,
    draws: FloatArray,
    roundoff_ulps: int,
) -> None:
    if not bool(np.isfinite(velocity).all() and np.isfinite(facet_normal).all()):
        raise ValueError("boundary velocities and normals must be finite")
    if not bool(np.isfinite(draws).all()) or bool(((draws < 0.0) | (draws >= 1.0)).any()):
        raise ValueError("boundary random draws must be in [0, 1)")
    if roundoff_ulps <= 0:
        raise ValueError("boundary roundoff_ulps must be positive")


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _two_term_sum(first: float, second: float) -> float:
    total = first + second
    second_virtual = total - first
    error = (first - (total - second_virtual)) + (second - second_virtual)
    return total + error


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _double_length_product(first: float, second: float) -> tuple[float, float]:
    """Return the high and error terms of one binary64 product."""

    first_split = first * 134217729.0
    first_high = first_split - (first_split - first)
    first_low = first - first_high
    second_split = second * 134217729.0
    second_high = second_split - (second_split - second)
    second_low = second - second_high
    partial = first_high * second_high
    cross = first_high * second_low + first_low * second_high
    product = partial + cross
    error = partial - product + cross + first_low * second_low
    return product, error


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _double_length_fast_sum(first: float, second: float) -> tuple[float, float]:
    """Return the high and error terms when ``abs(first) >= abs(second)``."""

    total = first + second
    return total, (first - total) + second


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _stable_hypot(first: float, second: float) -> float:
    """Match CPython's compensated binary64 two-vector norm."""

    maximum = max(abs(first), abs(second))
    if math.isinf(maximum) or maximum == 0.0:
        return maximum
    if math.isnan(maximum):
        return math.nan
    _fraction, exponent = math.frexp(maximum)
    if exponent < -1023:
        minimum_normal = np.finfo(np.float64).tiny
        return minimum_normal * _stable_hypot(
            first / minimum_normal,
            second / minimum_normal,
        )
    scale = math.ldexp(1.0, -exponent)
    compensated_sum = 1.0
    product_fraction = 0.0
    sum_fraction = 0.0
    for value in (first, second):
        scaled = value * scale
        product_high, product_low = _double_length_product(scaled, scaled)
        compensated_sum, sum_low = _double_length_fast_sum(
            compensated_sum,
            product_high,
        )
        product_fraction += product_low
        sum_fraction += sum_low
    result = math.sqrt(compensated_sum - 1.0 + (product_fraction + sum_fraction))
    product_high, product_low = _double_length_product(-result, result)
    compensated_sum, sum_low = _double_length_fast_sum(compensated_sum, product_high)
    product_fraction += product_low
    sum_fraction += sum_low
    correction = compensated_sum - 1.0 + (product_fraction + sum_fraction)
    result += correction / (2.0 * result)
    return result / scale


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _same_rule_signature(
    law: UInt8Array,
    otherwise_law: UInt8Array,
    normal_restitution: FloatArray,
    tangential_restitution: FloatArray,
    stick_probability: FloatArray,
    first_group: int,
    second_group: int,
) -> bool:
    first_law = law[first_group]
    second_law = law[second_group]
    first_reflection = first_law in (BOUNDARY_LAW_SPECULAR, BOUNDARY_LAW_RESTITUTION)
    second_reflection = second_law in (BOUNDARY_LAW_SPECULAR, BOUNDARY_LAW_RESTITUTION)
    if first_reflection and second_reflection:
        return (
            normal_restitution[first_group] == normal_restitution[second_group]
            and tangential_restitution[first_group] == tangential_restitution[second_group]
        )
    if first_law != second_law:
        return False
    if first_law == BOUNDARY_LAW_PROBABILISTIC_STICK:
        return (
            otherwise_law[first_group] == otherwise_law[second_group]
            and normal_restitution[first_group] == normal_restitution[second_group]
            and tangential_restitution[first_group] == tangential_restitution[second_group]
            and stick_probability[first_group] == stick_probability[second_group]
        )
    return True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _minimum_boundary_priority(
    priority: Int64Array,
    candidate_facet_ids: Int64Array,
    group_id: Int32Array,
    begin: int,
    end: int,
) -> tuple[np.uint8, int]:
    if begin == end:
        return BOUNDARY_STATUS_EMPTY_CANDIDATES, 0
    selected_priority = np.iinfo(np.int64).max
    for offset in range(begin, end):
        facet = candidate_facet_ids[offset]
        if facet < 0 or facet >= group_id.size:
            return BOUNDARY_STATUS_INVALID_REFERENCE, 0
        group = group_id[facet]
        if group < 0 or group >= priority.size:
            return BOUNDARY_STATUS_INVALID_REFERENCE, 0
        selected_priority = min(selected_priority, priority[group])
    return BOUNDARY_STATUS_OK, selected_priority


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _facet_precedes(
    candidate_boundary: int,
    candidate_facet: int,
    selected_boundary: int,
    selected_facet: int,
) -> bool:
    return (
        selected_facet < 0
        or candidate_boundary < selected_boundary
        or (candidate_boundary == selected_boundary and candidate_facet < selected_facet)
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _primary_boundary_rule(
    priority: Int64Array,
    rule_law: UInt8Array,
    otherwise_law: UInt8Array,
    normal_restitution: FloatArray,
    tangential_restitution: FloatArray,
    stick_probability: FloatArray,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    begin: int,
    end: int,
    selected_priority: int,
) -> tuple[np.uint8, int, int, int]:
    primary = -1
    primary_boundary = np.iinfo(np.int32).max
    for offset in range(begin, end):
        facet = candidate_facet_ids[offset]
        candidate_boundary = boundary_id[facet]
        if priority[group_id[facet]] == selected_priority and _facet_precedes(
            candidate_boundary,
            facet,
            primary_boundary,
            primary,
        ):
            primary = facet
            primary_boundary = candidate_boundary
    primary_group = group_id[primary]
    selected_count = 0
    for offset in range(begin, end):
        facet = candidate_facet_ids[offset]
        group = group_id[facet]
        if priority[group] != selected_priority:
            continue
        selected_count += 1
        if not _same_rule_signature(
            rule_law,
            otherwise_law,
            normal_restitution,
            tangential_restitution,
            stick_probability,
            primary_group,
            group,
        ):
            return BOUNDARY_STATUS_AMBIGUOUS_LAW, -1, -1, 0
    return BOUNDARY_STATUS_OK, primary, primary_group, selected_count


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _select_boundary_rule(
    priority: Int64Array,
    rule_law: UInt8Array,
    otherwise_law: UInt8Array,
    normal_restitution: FloatArray,
    tangential_restitution: FloatArray,
    stick_probability: FloatArray,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    begin: int,
    end: int,
) -> tuple[np.uint8, int, int, int, int]:
    status, selected_priority = _minimum_boundary_priority(
        priority,
        candidate_facet_ids,
        group_id,
        begin,
        end,
    )
    if status != BOUNDARY_STATUS_OK:
        return status, 0, -1, -1, 0
    status, primary, primary_group, selected_count = _primary_boundary_rule(
        priority,
        rule_law,
        otherwise_law,
        normal_restitution,
        tangential_restitution,
        stick_probability,
        candidate_facet_ids,
        boundary_id,
        group_id,
        begin,
        end,
        selected_priority,
    )
    return status, selected_priority, primary, primary_group, selected_count


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _combined_boundary_normal(
    priority: Int64Array,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    facet_normal: FloatArray,
    begin: int,
    end: int,
    selected_priority: int,
    selected_count: int,
    velocity_x: float,
    velocity_y: float,
    roundoff_ulps: int,
) -> tuple[np.uint8, float, float]:
    margin = float(roundoff_ulps) * _FLOAT64_EPS * _stable_hypot(velocity_x, velocity_y)
    total_x = 0.0
    total_y = 0.0
    previous_boundary = -1
    previous_facet = -1
    for _selection in range(selected_count):
        next_facet = -1
        next_boundary = np.iinfo(np.int32).max
        for offset in range(begin, end):
            facet = candidate_facet_ids[offset]
            if priority[group_id[facet]] != selected_priority:
                continue
            candidate_boundary = boundary_id[facet]
            after_previous = candidate_boundary > previous_boundary or (
                candidate_boundary == previous_boundary and facet > previous_facet
            )
            before_next = (
                next_facet < 0
                or candidate_boundary < next_boundary
                or (candidate_boundary == next_boundary and facet < next_facet)
            )
            if after_previous and before_next:
                next_facet = facet
                next_boundary = candidate_boundary
        if next_facet < 0:
            return BOUNDARY_STATUS_INVALID_REFERENCE, 0.0, 0.0
        normal_x = facet_normal[next_facet, 0]
        normal_y = facet_normal[next_facet, 1]
        if _two_term_sum(velocity_x * normal_x, velocity_y * normal_y) > margin:
            total_x += normal_x
            total_y += normal_y
        previous_boundary = next_boundary
        previous_facet = next_facet
    magnitude = _stable_hypot(total_x, total_y)
    normal_error = float(roundoff_ulps) * _FLOAT64_EPS * max(float(selected_count), 1.0)
    if not math.isfinite(magnitude) or magnitude <= normal_error:
        return BOUNDARY_STATUS_INDETERMINATE_POLICY, 0.0, 0.0
    return BOUNDARY_STATUS_OK, total_x / magnitude, total_y / magnitude


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _reflected_boundary_velocity(
    normal_restitution: float,
    tangential_restitution: float,
    candidate_facet_ids: Int64Array,
    facet_normal: FloatArray,
    begin: int,
    end: int,
    velocity_x: float,
    velocity_y: float,
    normal_x: float,
    normal_y: float,
    roundoff_ulps: int,
) -> tuple[np.uint8, float, float]:
    normal_speed = _two_term_sum(velocity_x * normal_x, velocity_y * normal_y)
    normal_velocity_x = normal_speed * normal_x
    normal_velocity_y = normal_speed * normal_y
    post_x = -normal_restitution * normal_velocity_x + tangential_restitution * (
        velocity_x - normal_velocity_x
    )
    post_y = -normal_restitution * normal_velocity_y + tangential_restitution * (
        velocity_y - normal_velocity_y
    )
    if not math.isfinite(post_x) or not math.isfinite(post_y):
        return BOUNDARY_STATUS_INDETERMINATE_POLICY, 0.0, 0.0
    margin = float(roundoff_ulps) * _FLOAT64_EPS * _stable_hypot(post_x, post_y)
    for offset in range(begin, end):
        facet = candidate_facet_ids[offset]
        outward_speed = _two_term_sum(
            post_x * facet_normal[facet, 0],
            post_y * facet_normal[facet, 1],
        )
        if outward_speed > margin:
            return BOUNDARY_STATUS_INDETERMINATE_POLICY, 0.0, 0.0
    return BOUNDARY_STATUS_OK, post_x, post_y


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _resolve_boundary_response_row(
    priority: Int64Array,
    rule_law: UInt8Array,
    otherwise_law: UInt8Array,
    normal_restitution: FloatArray,
    tangential_restitution: FloatArray,
    stick_probability: FloatArray,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    facet_normal: FloatArray,
    velocity_x: float,
    velocity_y: float,
    uniform_draw: float,
    roundoff_ulps: int,
    begin: int,
    end: int,
) -> tuple[np.uint8, np.uint8, np.uint8, float, float, int, float, float, bool]:
    selected = _select_boundary_rule(
        priority,
        rule_law,
        otherwise_law,
        normal_restitution,
        tangential_restitution,
        stick_probability,
        candidate_facet_ids,
        boundary_id,
        group_id,
        begin,
        end,
    )
    status, selected_priority, primary, primary_group, selected_count = selected
    if status != BOUNDARY_STATUS_OK:
        return (
            status,
            np.uint8(0),
            BOUNDARY_OUTCOME_NONE,
            velocity_x,
            velocity_y,
            -1,
            0.0,
            0.0,
            False,
        )
    selected_law = rule_law[primary_group]
    normal_x = facet_normal[primary, 0]
    normal_y = facet_normal[primary, 1]
    if selected_law == BOUNDARY_LAW_STICK:
        return (
            status,
            selected_law,
            BOUNDARY_OUTCOME_STUCK,
            0.0,
            0.0,
            primary,
            normal_x,
            normal_y,
            False,
        )
    if selected_law == BOUNDARY_LAW_ESCAPE:
        return (
            status,
            selected_law,
            BOUNDARY_OUTCOME_ESCAPED,
            velocity_x,
            velocity_y,
            primary,
            normal_x,
            normal_y,
            False,
        )
    if selected_law == BOUNDARY_LAW_HOLD:
        return (
            status,
            selected_law,
            BOUNDARY_OUTCOME_HELD,
            velocity_x,
            velocity_y,
            primary,
            normal_x,
            normal_y,
            False,
        )
    if (
        selected_law == BOUNDARY_LAW_PROBABILISTIC_STICK
        and uniform_draw < stick_probability[primary_group]
    ):
        return (
            status,
            selected_law,
            BOUNDARY_OUTCOME_STUCK,
            0.0,
            0.0,
            primary,
            normal_x,
            normal_y,
            False,
        )
    if selected_law not in (
        BOUNDARY_LAW_SPECULAR,
        BOUNDARY_LAW_RESTITUTION,
        BOUNDARY_LAW_PROBABILISTIC_STICK,
    ):
        return (
            BOUNDARY_STATUS_INVALID_REFERENCE,
            selected_law,
            BOUNDARY_OUTCOME_NONE,
            velocity_x,
            velocity_y,
            primary,
            normal_x,
            normal_y,
            False,
        )
    status, normal_x, normal_y = _combined_boundary_normal(
        priority,
        candidate_facet_ids,
        boundary_id,
        group_id,
        facet_normal,
        begin,
        end,
        selected_priority,
        selected_count,
        velocity_x,
        velocity_y,
        roundoff_ulps,
    )
    if status != BOUNDARY_STATUS_OK:
        return (
            status,
            selected_law,
            BOUNDARY_OUTCOME_NONE,
            velocity_x,
            velocity_y,
            primary,
            normal_x,
            normal_y,
            False,
        )
    status, post_x, post_y = _reflected_boundary_velocity(
        normal_restitution[primary_group],
        tangential_restitution[primary_group],
        candidate_facet_ids,
        facet_normal,
        begin,
        end,
        velocity_x,
        velocity_y,
        normal_x,
        normal_y,
        roundoff_ulps,
    )
    outcome = BOUNDARY_OUTCOME_REFLECTED if status == BOUNDARY_STATUS_OK else BOUNDARY_OUTCOME_NONE
    return (
        status,
        selected_law,
        outcome,
        post_x,
        post_y,
        primary,
        normal_x,
        normal_y,
        status == BOUNDARY_STATUS_OK,
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _resolve_boundary_responses_kernel(
    priority: Int64Array,
    rule_law: UInt8Array,
    otherwise_law: UInt8Array,
    normal_restitution: FloatArray,
    tangential_restitution: FloatArray,
    stick_probability: FloatArray,
    candidate_offsets: Int64Array,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    facet_normal: FloatArray,
    velocity_pre_m_s: FloatArray,
    uniform_draw: FloatArray,
    roundoff_ulps: int,
    status: UInt8Array,
    law: UInt8Array,
    outcome: UInt8Array,
    velocity_post_m_s: FloatArray,
    primary_facet_id: Int64Array,
    effective_normal: FloatArray,
    remains_active: NDArray[np.bool_],
) -> None:
    for row in range(status.size):
        response = _resolve_boundary_response_row(
            priority,
            rule_law,
            otherwise_law,
            normal_restitution,
            tangential_restitution,
            stick_probability,
            candidate_facet_ids,
            boundary_id,
            group_id,
            facet_normal,
            velocity_pre_m_s[row, 0],
            velocity_pre_m_s[row, 1],
            uniform_draw[row],
            roundoff_ulps,
            candidate_offsets[row],
            candidate_offsets[row + 1],
        )
        status[row], law[row], outcome[row] = response[0], response[1], response[2]
        velocity_post_m_s[row, 0], velocity_post_m_s[row, 1] = response[3], response[4]
        primary_facet_id[row] = response[5]
        effective_normal[row, 0], effective_normal[row, 1] = response[6], response[7]
        remains_active[row] = response[8]


def _restitution_parameters(
    parameters: Mapping[str, object],
    location: str,
) -> tuple[float, float]:
    if set(parameters) != {"normal_restitution", "tangential_restitution"}:
        raise BoundaryLawError(
            f"{location} requires exactly normal_restitution and tangential_restitution"
        )
    normal = _finite_number(parameters["normal_restitution"], f"{location}.normal_restitution")
    tangential = _finite_number(
        parameters["tangential_restitution"],
        f"{location}.tangential_restitution",
    )
    if not 0.0 < normal <= 1.0:
        raise BoundaryLawError(f"{location}.normal_restitution must be in (0, 1]")
    if not 0.0 <= tangential <= 1.0:
        raise BoundaryLawError(f"{location}.tangential_restitution must be in [0, 1]")
    return normal, tangential


def _probability(value: object) -> float:
    result = _finite_number(value, "probabilistic_stick.probability")
    if not 0.0 <= result <= 1.0:
        raise BoundaryLawError("probabilistic_stick.probability must be in [0, 1]")
    return result


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BoundaryLawError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise BoundaryLawError(f"{location} must be a finite number")
    return result
