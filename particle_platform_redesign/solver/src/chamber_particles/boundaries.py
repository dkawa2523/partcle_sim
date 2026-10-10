"""Physical wall responses applied after geometric hit localization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .physics.forces import BOLTZMANN_J_K

type FloatArray = NDArray[np.float64]
type Int32Array = NDArray[np.int32]
type Int64Array = NDArray[np.int64]
type UInt8Array = NDArray[np.uint8]

BOUNDARY_ALGORITHM_REVISION = "contact_wall_laws_v7"

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
BOUNDARY_LAW_MAXWELL_THERMAL = np.uint8(7)

_BOUNDARY_LAW_CODES = {
    "stick": BOUNDARY_LAW_STICK,
    "escape": BOUNDARY_LAW_ESCAPE,
    "specular": BOUNDARY_LAW_SPECULAR,
    "restitution": BOUNDARY_LAW_RESTITUTION,
    "probabilistic_stick": BOUNDARY_LAW_PROBABILISTIC_STICK,
    "hold": BOUNDARY_LAW_HOLD,
    "maxwell_thermal": BOUNDARY_LAW_MAXWELL_THERMAL,
}

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
    wall_temperature_K: float | None = None
    diffuse_reflection_fraction: float | None = None
    wall_velocity_m_s: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class PreparedBoundaryRules:
    """Dense numeric wall-law table consumed by compiled event rows."""

    priority: Int64Array
    law: UInt8Array
    otherwise_law: UInt8Array
    normal_restitution: FloatArray
    tangential_restitution: FloatArray
    stick_probability: FloatArray
    wall_temperature_K: FloatArray
    diffuse_reflection_fraction: FloatArray
    wall_velocity_m_s: FloatArray


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
        return _prepare_parameterless_rule(group_id, priority, law_id, parameters)
    if law_id == "restitution":
        return _prepare_restitution_rule(group_id, priority, parameters)
    if law_id == "maxwell_thermal":
        return _prepare_maxwell_rule(group_id, priority, parameters)
    if law_id == "probabilistic_stick":
        return _prepare_probabilistic_stick_rule(group_id, priority, parameters)
    raise BoundaryLawError(f"unsupported boundary law: {law_id}")


def _prepare_parameterless_rule(
    group_id: int,
    priority: int,
    law_id: str,
    parameters: Mapping[str, object],
) -> BoundaryRule:
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


def _prepare_restitution_rule(
    group_id: int,
    priority: int,
    parameters: Mapping[str, object],
) -> BoundaryRule:
    normal, tangential = _restitution_parameters(parameters, "restitution")
    return BoundaryRule(
        group_id,
        priority,
        "restitution",
        normal_restitution=normal,
        tangential_restitution=tangential,
    )


def _prepare_maxwell_rule(
    group_id: int,
    priority: int,
    parameters: Mapping[str, object],
) -> BoundaryRule:
    temperature, fraction, wall_velocity = _maxwell_thermal_parameters(
        parameters,
        "maxwell_thermal",
    )
    return BoundaryRule(
        group_id,
        priority,
        "maxwell_thermal",
        wall_temperature_K=temperature,
        diffuse_reflection_fraction=fraction,
        wall_velocity_m_s=wall_velocity,
    )


def _prepare_probabilistic_stick_rule(
    group_id: int,
    priority: int,
    parameters: Mapping[str, object],
) -> BoundaryRule:
    if set(parameters) != {"probability", "otherwise"}:
        raise BoundaryLawError("probabilistic_stick requires exactly probability and otherwise")
    probability = _probability(parameters["probability"])
    otherwise = parameters["otherwise"]
    if not isinstance(otherwise, Mapping):
        raise BoundaryLawError("probabilistic_stick.otherwise must be a law mapping")
    otherwise_law = otherwise.get("law")
    reflection_parameters = {key: value for key, value in otherwise.items() if key != "law"}
    reflection = _prepare_nested_reflection(otherwise_law, reflection_parameters)
    return BoundaryRule(
        group_id,
        priority,
        "probabilistic_stick",
        normal_restitution=reflection.normal_restitution,
        tangential_restitution=reflection.tangential_restitution,
        stick_probability=probability,
        otherwise_law_id=reflection.law_id,
        wall_temperature_K=reflection.wall_temperature_K,
        diffuse_reflection_fraction=reflection.diffuse_reflection_fraction,
        wall_velocity_m_s=reflection.wall_velocity_m_s,
    )


def _prepare_nested_reflection(
    law_id: object,
    parameters: Mapping[str, object],
) -> BoundaryRule:
    if law_id == "specular":
        return _prepare_parameterless_rule(0, 0, "specular", parameters)
    if law_id == "restitution":
        normal, tangential = _restitution_parameters(
            parameters,
            "probabilistic_stick.otherwise restitution",
        )
        return BoundaryRule(
            0,
            0,
            "restitution",
            normal_restitution=normal,
            tangential_restitution=tangential,
        )
    if law_id == "maxwell_thermal":
        temperature, fraction, wall_velocity = _maxwell_thermal_parameters(
            parameters,
            "probabilistic_stick.otherwise maxwell_thermal",
        )
        return BoundaryRule(
            0,
            0,
            "maxwell_thermal",
            wall_temperature_K=temperature,
            diffuse_reflection_fraction=fraction,
            wall_velocity_m_s=wall_velocity,
        )
    raise BoundaryLawError(
        "probabilistic_stick.otherwise law must be specular, restitution, or maxwell_thermal"
    )


def _pack_boundary_reflection(
    rule: BoundaryRule,
    group_id: int,
    otherwise_law: UInt8Array,
    normal: FloatArray,
    tangential: FloatArray,
    probability: FloatArray,
    wall_temperature: FloatArray,
    diffuse_fraction: FloatArray,
    wall_velocity: FloatArray,
) -> None:
    reflection_law = rule.otherwise_law_id if rule.law_id == "probabilistic_stick" else rule.law_id
    if reflection_law in {"specular", "restitution"}:
        if rule.normal_restitution is None or rule.tangential_restitution is None:
            raise BoundaryLawError("prepared reflection law lost its restitution coefficients")
        normal[group_id] = rule.normal_restitution
        tangential[group_id] = rule.tangential_restitution
    elif reflection_law == "maxwell_thermal":
        _pack_maxwell_parameters(
            rule,
            group_id,
            wall_temperature,
            diffuse_fraction,
            wall_velocity,
        )
    if rule.law_id == "probabilistic_stick":
        if rule.stick_probability is None or rule.otherwise_law_id not in {
            "specular",
            "restitution",
            "maxwell_thermal",
        }:
            raise BoundaryLawError("prepared probabilistic law lost its nested reflection law")
        probability[group_id] = rule.stick_probability
        otherwise_law[group_id] = _BOUNDARY_LAW_CODES[rule.otherwise_law_id]


def _pack_maxwell_parameters(
    rule: BoundaryRule,
    group_id: int,
    wall_temperature: FloatArray,
    diffuse_fraction: FloatArray,
    wall_velocity: FloatArray,
) -> None:
    if (
        rule.wall_temperature_K is None
        or rule.diffuse_reflection_fraction is None
        or rule.wall_velocity_m_s is None
    ):
        raise BoundaryLawError("prepared Maxwell law lost its thermal parameters")
    wall_temperature[group_id] = rule.wall_temperature_K
    diffuse_fraction[group_id] = rule.diffuse_reflection_fraction
    wall_velocity[group_id] = rule.wall_velocity_m_s


def prepare_boundary_rules(
    rules: Sequence[BoundaryRule],
    *,
    group_count: int | None = None,
) -> PreparedBoundaryRules:
    """Pack one group-indexed wall table without assigning topology a wall law.

    The ordinary no-topology path remains a dense table.  ``group_count`` is
    supplied only when some geometry groups are owned by a non-material
    topology; those rows retain law code zero and must never reach the wall
    resolver.
    """

    dense_required = group_count is None
    count = len(rules) if group_count is None else group_count
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("boundary group_count must be a nonnegative integer")
    priority = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    law = np.zeros(count, dtype=np.uint8)
    otherwise_law = np.zeros(count, dtype=np.uint8)
    normal = np.full(count, np.nan, dtype=np.float64)
    tangential = np.full(count, np.nan, dtype=np.float64)
    probability = np.full(count, np.nan, dtype=np.float64)
    wall_temperature = np.full(count, np.nan, dtype=np.float64)
    diffuse_fraction = np.full(count, np.nan, dtype=np.float64)
    wall_velocity = np.full((count, 2), np.nan, dtype=np.float64)
    seen: set[int] = set()
    for rule_index, rule in enumerate(rules):
        group_id = rule.group_id
        if (
            group_id < 0
            or group_id >= count
            or group_id in seen
            or rule.priority < 0
            or rule.law_id not in _BOUNDARY_LAW_CODES
            or (dense_required and group_id != rule_index)
        ):
            raise BoundaryLawError("boundary rules do not form a valid group-indexed wall table")
        seen.add(group_id)
        priority[group_id] = rule.priority
        law[group_id] = _BOUNDARY_LAW_CODES[rule.law_id]
        _pack_boundary_reflection(
            rule,
            group_id,
            otherwise_law,
            normal,
            tangential,
            probability,
            wall_temperature,
            diffuse_fraction,
            wall_velocity,
        )
    arrays = (
        priority,
        law,
        otherwise_law,
        normal,
        tangential,
        probability,
        wall_temperature,
        diffuse_fraction,
        wall_velocity,
    )
    for array in arrays:
        array.setflags(write=False)
    return PreparedBoundaryRules(*arrays)


def validate_boundary_rule_frames(
    rules: PreparedBoundaryRules,
    group_id: Int32Array,
    facet_normal: FloatArray,
    *,
    roundoff_ulps: int,
) -> None:
    """Reject scattering-frame normal motion against the static geometry."""

    groups = np.asarray(group_id, dtype=np.int32)
    normals = np.asarray(facet_normal, dtype=np.float64)
    if groups.ndim != 1 or normals.shape != (groups.size, 2):
        raise ValueError("boundary group IDs and facet normals must align")
    if roundoff_ulps <= 0:
        raise ValueError("boundary roundoff_ulps must be positive")
    for facet, group_value in enumerate(groups):
        group = int(group_value)
        if group < 0 or group >= rules.law.size:
            raise BoundaryLawError("boundary facet references an unknown rule group")
        reflection_law = rules.law[group]
        if reflection_law == BOUNDARY_LAW_PROBABILISTIC_STICK:
            reflection_law = rules.otherwise_law[group]
        if reflection_law != BOUNDARY_LAW_MAXWELL_THERMAL:
            continue
        wall_x, wall_y = rules.wall_velocity_m_s[group]
        wall_speed = math.hypot(float(wall_x), float(wall_y))
        margin = float(roundoff_ulps) * _FLOAT64_EPS * wall_speed
        normal_speed = _two_term_sum(
            float(wall_x) * float(normals[facet, 0]),
            float(wall_y) * float(normals[facet, 1]),
        )
        if abs(normal_speed) > margin:
            raise BoundaryLawError(
                "maxwell_thermal.wall_velocity_m_s must be tangent to every facet in its "
                "static boundary group; normal wall motion requires moving geometry"
            )


def resolve_boundary_responses_batch(
    rules: PreparedBoundaryRules,
    candidate_offsets: Int64Array,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    candidate_contact_normal: FloatArray,
    velocity_pre_m_s: FloatArray,
    particle_mass_kg: FloatArray,
    law_uniform_draw: FloatArray,
    diffuse_uniform_draw: FloatArray,
    thermal_normal_uniform_open: FloatArray,
    thermal_tangential_standard_normal: FloatArray,
    *,
    roundoff_ulps: int,
) -> BoundaryResponseBatch:
    """Resolve independent CSR hit rows without Python objects or callbacks."""

    offsets = np.asarray(candidate_offsets, dtype=np.int64)
    candidates = np.asarray(candidate_facet_ids, dtype=np.int64)
    velocity = np.asarray(velocity_pre_m_s, dtype=np.float64)
    mass = np.asarray(particle_mass_kg, dtype=np.float64)
    law_draws = np.asarray(law_uniform_draw, dtype=np.float64)
    diffuse_draws = np.asarray(diffuse_uniform_draw, dtype=np.float64)
    thermal_normal_draws = np.asarray(thermal_normal_uniform_open, dtype=np.float64)
    thermal_tangent_draws = np.asarray(thermal_tangential_standard_normal, dtype=np.float64)
    row_count = _validate_boundary_batch_inputs(
        offsets,
        candidates,
        boundary_id,
        group_id,
        candidate_contact_normal,
        velocity,
        mass,
        law_draws,
        diffuse_draws,
        thermal_normal_draws,
        thermal_tangent_draws,
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
        rules.wall_temperature_K,
        rules.diffuse_reflection_fraction,
        rules.wall_velocity_m_s,
        offsets,
        candidates,
        boundary_id,
        group_id,
        candidate_contact_normal,
        velocity,
        mass,
        law_draws,
        diffuse_draws,
        thermal_normal_draws,
        thermal_tangent_draws,
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
    candidate_contact_normal: FloatArray,
    velocity: FloatArray,
    mass: FloatArray,
    law_draws: FloatArray,
    diffuse_draws: FloatArray,
    thermal_normal_draws: FloatArray,
    thermal_tangent_draws: FloatArray,
    roundoff_ulps: int,
) -> int:
    row_count = offsets.size - 1
    if offsets.ndim != 1 or row_count < 0 or offsets[0] != 0:
        raise ValueError("candidate_offsets must be one nonempty CSR offset vector")
    if bool((np.diff(offsets) < 0).any()) or int(offsets[-1]) != candidates.size:
        raise ValueError("candidate_offsets do not delimit candidate_facet_ids")
    row_vectors = (
        mass,
        law_draws,
        diffuse_draws,
        thermal_normal_draws,
        thermal_tangent_draws,
    )
    if velocity.shape != (row_count, 2) or any(
        value.shape != (row_count,) for value in row_vectors
    ):
        raise ValueError("velocity, mass, and boundary draws must align with candidate rows")
    facet_count = int(boundary_id.size)
    aligned = (
        group_id.shape == (facet_count,)
        and candidate_contact_normal.shape == (candidates.size, 2)
        and candidates.ndim == 1
    )
    if not aligned:
        raise ValueError("boundary arrays must have aligned facet dimensions")
    _validate_boundary_batch_values(
        velocity,
        candidate_contact_normal,
        mass,
        law_draws,
        diffuse_draws,
        thermal_normal_draws,
        thermal_tangent_draws,
        roundoff_ulps,
    )
    return row_count


def _validate_boundary_batch_values(
    velocity: FloatArray,
    candidate_contact_normal: FloatArray,
    mass: FloatArray,
    law_draws: FloatArray,
    diffuse_draws: FloatArray,
    thermal_normal_draws: FloatArray,
    thermal_tangent_draws: FloatArray,
    roundoff_ulps: int,
) -> None:
    if not bool(np.isfinite(velocity).all() and np.isfinite(candidate_contact_normal).all()):
        raise ValueError("boundary velocities and normals must be finite")
    if not bool(np.isfinite(mass).all()) or bool((mass <= 0.0).any()):
        raise ValueError("particle masses must be finite and positive")
    if not bool(np.isfinite(law_draws).all()) or bool(
        ((law_draws < 0.0) | (law_draws >= 1.0)).any()
    ):
        raise ValueError("boundary random draws must be in [0, 1)")
    if not bool(np.isfinite(diffuse_draws).all()) or bool(
        ((diffuse_draws < 0.0) | (diffuse_draws >= 1.0)).any()
    ):
        raise ValueError("Maxwell mixture draws must be in [0, 1)")
    if not bool(np.isfinite(thermal_normal_draws).all()) or bool(
        ((thermal_normal_draws <= 0.0) | (thermal_normal_draws >= 1.0)).any()
    ):
        raise ValueError("thermal normal random draws must be in (0, 1)")
    if not bool(np.isfinite(thermal_tangent_draws).all()):
        raise ValueError("thermal tangential normal draws must be finite")
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
def _valid_maxwell_thermal_state(
    wall_temperature_K: float,
    particle_mass_kg: float,
    wall_velocity_x_m_s: float,
    wall_velocity_y_m_s: float,
) -> bool:
    return (
        math.isfinite(wall_temperature_K)
        and wall_temperature_K > 0.0
        and math.isfinite(particle_mass_kg)
        and particle_mass_kg > 0.0
        and math.isfinite(wall_velocity_x_m_s)
        and math.isfinite(wall_velocity_y_m_s)
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _valid_half_range_frame_draws(
    outward_normal_x: float,
    outward_normal_y: float,
    normal_uniform_open: float,
    tangential_standard_normal: float,
) -> bool:
    normal_norm = _stable_hypot(outward_normal_x, outward_normal_y)
    return (
        math.isfinite(outward_normal_x)
        and math.isfinite(outward_normal_y)
        and abs(normal_norm - 1.0) <= 64.0 * _FLOAT64_EPS
        and math.isfinite(normal_uniform_open)
        and 0.0 < normal_uniform_open < 1.0
        and math.isfinite(tangential_standard_normal)
    )


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def half_range_maxwell_flux_velocity(
    wall_temperature_K: float,
    particle_mass_kg: float,
    wall_velocity_x_m_s: float,
    wall_velocity_y_m_s: float,
    outward_normal_x: float,
    outward_normal_y: float,
    normal_uniform_open: float,
    tangential_standard_normal: float,
) -> tuple[float, float]:
    """Map independent draws to one inward half-range Maxwell flux velocity.

    ``(outward_normal_x, outward_normal_y)`` must be a finite unit normal.  The
    deterministic tangent is ``(-normal_y, normal_x)``.  The returned lab-frame
    velocity adds the supplied wall/scattering-frame velocity.  This function
    owns no RNG stream or wall/source selection policy so surface thermal-flux
    release can reuse the exact same transform.
    """

    valid_state = _valid_maxwell_thermal_state(
        wall_temperature_K,
        particle_mass_kg,
        wall_velocity_x_m_s,
        wall_velocity_y_m_s,
    )
    valid_frame_draws = _valid_half_range_frame_draws(
        outward_normal_x,
        outward_normal_y,
        normal_uniform_open,
        tangential_standard_normal,
    )
    if not valid_state or not valid_frame_draws:
        raise ValueError("half-range Maxwell flux inputs are outside their finite domains")
    thermal_variance = BOLTZMANN_J_K * wall_temperature_K / particle_mass_kg
    if not math.isfinite(thermal_variance) or thermal_variance <= 0.0:
        raise ValueError("half-range Maxwell thermal variance is not representable")
    thermal_scale = math.sqrt(thermal_variance)
    inward_normal_speed = thermal_scale * math.sqrt(-2.0 * math.log(normal_uniform_open))
    tangential_speed = thermal_scale * tangential_standard_normal
    velocity_x = (
        wall_velocity_x_m_s
        - inward_normal_speed * outward_normal_x
        - tangential_speed * outward_normal_y
    )
    velocity_y = (
        wall_velocity_y_m_s
        - inward_normal_speed * outward_normal_y
        + tangential_speed * outward_normal_x
    )
    if not math.isfinite(velocity_x) or not math.isfinite(velocity_y):
        raise ValueError("half-range Maxwell velocity is not representable")
    return velocity_x, velocity_y


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def half_range_maxwell_flux_velocity_batch(
    wall_temperature_K: float,
    particle_mass_kg: FloatArray,
    wall_velocity_m_s: FloatArray,
    outward_normal: FloatArray,
    normal_uniform_open: FloatArray,
    tangential_standard_normal: FloatArray,
) -> FloatArray:
    """Apply :func:`half_range_maxwell_flux_velocity` to one bounded batch."""

    count = particle_mass_kg.size
    if (
        particle_mass_kg.ndim != 1
        or wall_velocity_m_s.shape != (2,)
        or outward_normal.shape != (count, 2)
        or normal_uniform_open.shape != (count,)
        or tangential_standard_normal.shape != (count,)
    ):
        raise ValueError("half-range Maxwell flux batch arrays do not align")
    velocity = np.empty((count, 2), dtype=np.float64)
    for row in range(count):
        velocity[row, 0], velocity[row, 1] = half_range_maxwell_flux_velocity(
            wall_temperature_K,
            particle_mass_kg[row],
            wall_velocity_m_s[0],
            wall_velocity_m_s[1],
            outward_normal[row, 0],
            outward_normal[row, 1],
            normal_uniform_open[row],
            tangential_standard_normal[row],
        )
    return velocity


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _same_reflection_signature(
    first_reflection: np.uint8,
    second_reflection: np.uint8,
    normal_restitution: FloatArray,
    tangential_restitution: FloatArray,
    wall_temperature_K: FloatArray,
    diffuse_reflection_fraction: FloatArray,
    wall_velocity_m_s: FloatArray,
    first_group: int,
    second_group: int,
) -> bool:
    deterministic_reflection = (
        BOUNDARY_LAW_SPECULAR,
        BOUNDARY_LAW_RESTITUTION,
    )
    if (
        first_reflection in deterministic_reflection
        and second_reflection in deterministic_reflection
    ):
        return (
            normal_restitution[first_group] == normal_restitution[second_group]
            and tangential_restitution[first_group] == tangential_restitution[second_group]
        )
    if (
        first_reflection == BOUNDARY_LAW_MAXWELL_THERMAL
        and second_reflection == BOUNDARY_LAW_MAXWELL_THERMAL
    ):
        return (
            wall_temperature_K[first_group] == wall_temperature_K[second_group]
            and diffuse_reflection_fraction[first_group]
            == diffuse_reflection_fraction[second_group]
            and wall_velocity_m_s[first_group, 0] == wall_velocity_m_s[second_group, 0]
            and wall_velocity_m_s[first_group, 1] == wall_velocity_m_s[second_group, 1]
        )
    return False


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _same_rule_signature(
    law: UInt8Array,
    otherwise_law: UInt8Array,
    normal_restitution: FloatArray,
    tangential_restitution: FloatArray,
    stick_probability: FloatArray,
    wall_temperature_K: FloatArray,
    diffuse_reflection_fraction: FloatArray,
    wall_velocity_m_s: FloatArray,
    first_group: int,
    second_group: int,
) -> bool:
    first_law = law[first_group]
    second_law = law[second_group]
    first_reflection = (
        otherwise_law[first_group] if first_law == BOUNDARY_LAW_PROBABILISTIC_STICK else first_law
    )
    second_reflection = (
        otherwise_law[second_group]
        if second_law == BOUNDARY_LAW_PROBABILISTIC_STICK
        else second_law
    )
    reflection_equal = _same_reflection_signature(
        first_reflection,
        second_reflection,
        normal_restitution,
        tangential_restitution,
        wall_temperature_K,
        diffuse_reflection_fraction,
        wall_velocity_m_s,
        first_group,
        second_group,
    )
    deterministic_reflection = (
        BOUNDARY_LAW_SPECULAR,
        BOUNDARY_LAW_RESTITUTION,
        BOUNDARY_LAW_MAXWELL_THERMAL,
    )
    if first_law in deterministic_reflection:
        return second_law in deterministic_reflection and reflection_equal
    if first_law != second_law:
        return False
    if first_law == BOUNDARY_LAW_PROBABILISTIC_STICK:
        return (
            stick_probability[first_group] == stick_probability[second_group] and reflection_equal
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
    wall_temperature_K: FloatArray,
    diffuse_reflection_fraction: FloatArray,
    wall_velocity_m_s: FloatArray,
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
            wall_temperature_K,
            diffuse_reflection_fraction,
            wall_velocity_m_s,
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
    wall_temperature_K: FloatArray,
    diffuse_reflection_fraction: FloatArray,
    wall_velocity_m_s: FloatArray,
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
        wall_temperature_K,
        diffuse_reflection_fraction,
        wall_velocity_m_s,
        candidate_facet_ids,
        boundary_id,
        group_id,
        begin,
        end,
        selected_priority,
    )
    return status, selected_priority, primary, primary_group, selected_count


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _candidate_follows_and_precedes(
    candidate_boundary: int,
    facet: int,
    previous_boundary: int,
    previous_facet: int,
    next_boundary: int,
    next_facet: int,
) -> bool:
    after_previous = candidate_boundary > previous_boundary or (
        candidate_boundary == previous_boundary and facet > previous_facet
    )
    before_next = (
        next_facet < 0
        or candidate_boundary < next_boundary
        or (candidate_boundary == next_boundary and facet < next_facet)
    )
    return after_previous and before_next


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _combined_boundary_normal(
    priority: Int64Array,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    candidate_contact_normal: FloatArray,
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
        next_offset = -1
        next_boundary = np.iinfo(np.int32).max
        for offset in range(begin, end):
            facet = candidate_facet_ids[offset]
            if priority[group_id[facet]] != selected_priority:
                continue
            candidate_boundary = boundary_id[facet]
            if _candidate_follows_and_precedes(
                candidate_boundary,
                facet,
                previous_boundary,
                previous_facet,
                next_boundary,
                next_facet,
            ):
                next_facet = facet
                next_offset = offset
                next_boundary = candidate_boundary
        if next_facet < 0 or next_offset < 0:
            return BOUNDARY_STATUS_INVALID_REFERENCE, 0.0, 0.0
        normal_x = candidate_contact_normal[next_offset, 0]
        normal_y = candidate_contact_normal[next_offset, 1]
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
    candidate_contact_normal: FloatArray,
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
    if not _post_velocity_is_inward(
        candidate_facet_ids,
        candidate_contact_normal,
        begin,
        end,
        post_x,
        post_y,
        roundoff_ulps,
    ):
        return BOUNDARY_STATUS_INDETERMINATE_POLICY, 0.0, 0.0
    return BOUNDARY_STATUS_OK, post_x, post_y


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _post_velocity_is_inward(
    candidate_facet_ids: Int64Array,
    candidate_contact_normal: FloatArray,
    begin: int,
    end: int,
    post_x: float,
    post_y: float,
    roundoff_ulps: int,
) -> bool:
    margin = float(roundoff_ulps) * _FLOAT64_EPS * _stable_hypot(post_x, post_y)
    for offset in range(begin, end):
        _facet = candidate_facet_ids[offset]
        outward_speed = _two_term_sum(
            post_x * candidate_contact_normal[offset, 0],
            post_y * candidate_contact_normal[offset, 1],
        )
        if outward_speed > margin:
            return False
    return True


@njit(cache=True, fastmath=False, parallel=False, nogil=True)
def _maxwell_boundary_velocity(
    wall_temperature_K: float,
    diffuse_reflection_fraction: float,
    wall_velocity_x_m_s: float,
    wall_velocity_y_m_s: float,
    particle_mass_kg: float,
    diffuse_uniform_draw: float,
    thermal_normal_uniform_open: float,
    thermal_tangential_standard_normal: float,
    candidate_facet_ids: Int64Array,
    candidate_contact_normal: FloatArray,
    begin: int,
    end: int,
    velocity_x: float,
    velocity_y: float,
    normal_x: float,
    normal_y: float,
    roundoff_ulps: int,
) -> tuple[np.uint8, float, float]:
    if diffuse_uniform_draw < diffuse_reflection_fraction:
        post_x, post_y = half_range_maxwell_flux_velocity(
            wall_temperature_K,
            particle_mass_kg,
            wall_velocity_x_m_s,
            wall_velocity_y_m_s,
            normal_x,
            normal_y,
            thermal_normal_uniform_open,
            thermal_tangential_standard_normal,
        )
    else:
        relative_x = velocity_x - wall_velocity_x_m_s
        relative_y = velocity_y - wall_velocity_y_m_s
        normal_speed = _two_term_sum(relative_x * normal_x, relative_y * normal_y)
        post_x = relative_x - 2.0 * normal_speed * normal_x + wall_velocity_x_m_s
        post_y = relative_y - 2.0 * normal_speed * normal_y + wall_velocity_y_m_s
    if not math.isfinite(post_x) or not math.isfinite(post_y):
        return BOUNDARY_STATUS_INDETERMINATE_POLICY, 0.0, 0.0
    if not _post_velocity_is_inward(
        candidate_facet_ids,
        candidate_contact_normal,
        begin,
        end,
        post_x,
        post_y,
        roundoff_ulps,
    ):
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
    wall_temperature_K: FloatArray,
    diffuse_reflection_fraction: FloatArray,
    wall_velocity_m_s: FloatArray,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    candidate_contact_normal: FloatArray,
    velocity_x: float,
    velocity_y: float,
    particle_mass_kg: float,
    law_uniform_draw: float,
    diffuse_uniform_draw: float,
    thermal_normal_uniform_open: float,
    thermal_tangential_standard_normal: float,
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
        wall_temperature_K,
        diffuse_reflection_fraction,
        wall_velocity_m_s,
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
    primary_offset = -1
    for offset in range(begin, end):
        if candidate_facet_ids[offset] == primary:
            primary_offset = offset
            break
    if primary_offset < 0:
        return (
            BOUNDARY_STATUS_INVALID_REFERENCE,
            np.uint8(0),
            BOUNDARY_OUTCOME_NONE,
            velocity_x,
            velocity_y,
            -1,
            0.0,
            0.0,
            False,
        )
    normal_x = candidate_contact_normal[primary_offset, 0]
    normal_y = candidate_contact_normal[primary_offset, 1]
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
        and law_uniform_draw < stick_probability[primary_group]
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
    reflection_law = selected_law
    if selected_law == BOUNDARY_LAW_PROBABILISTIC_STICK:
        reflection_law = otherwise_law[primary_group]
    if reflection_law not in (
        BOUNDARY_LAW_SPECULAR,
        BOUNDARY_LAW_RESTITUTION,
        BOUNDARY_LAW_MAXWELL_THERMAL,
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
        candidate_contact_normal,
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
    if reflection_law == BOUNDARY_LAW_MAXWELL_THERMAL:
        status, post_x, post_y = _maxwell_boundary_velocity(
            wall_temperature_K[primary_group],
            diffuse_reflection_fraction[primary_group],
            wall_velocity_m_s[primary_group, 0],
            wall_velocity_m_s[primary_group, 1],
            particle_mass_kg,
            diffuse_uniform_draw,
            thermal_normal_uniform_open,
            thermal_tangential_standard_normal,
            candidate_facet_ids,
            candidate_contact_normal,
            begin,
            end,
            velocity_x,
            velocity_y,
            normal_x,
            normal_y,
            roundoff_ulps,
        )
    else:
        status, post_x, post_y = _reflected_boundary_velocity(
            normal_restitution[primary_group],
            tangential_restitution[primary_group],
            candidate_facet_ids,
            candidate_contact_normal,
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
    wall_temperature_K: FloatArray,
    diffuse_reflection_fraction: FloatArray,
    wall_velocity_m_s: FloatArray,
    candidate_offsets: Int64Array,
    candidate_facet_ids: Int64Array,
    boundary_id: Int32Array,
    group_id: Int32Array,
    candidate_contact_normal: FloatArray,
    velocity_pre_m_s: FloatArray,
    particle_mass_kg: FloatArray,
    law_uniform_draw: FloatArray,
    diffuse_uniform_draw: FloatArray,
    thermal_normal_uniform_open: FloatArray,
    thermal_tangential_standard_normal: FloatArray,
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
            wall_temperature_K,
            diffuse_reflection_fraction,
            wall_velocity_m_s,
            candidate_facet_ids,
            boundary_id,
            group_id,
            candidate_contact_normal,
            velocity_pre_m_s[row, 0],
            velocity_pre_m_s[row, 1],
            particle_mass_kg[row],
            law_uniform_draw[row],
            diffuse_uniform_draw[row],
            thermal_normal_uniform_open[row],
            thermal_tangential_standard_normal[row],
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


def _maxwell_thermal_parameters(
    parameters: Mapping[str, object],
    location: str,
) -> tuple[float, float, tuple[float, float]]:
    required = {
        "wall_temperature_K",
        "diffuse_reflection_fraction",
        "wall_velocity_m_s",
    }
    if set(parameters) != required:
        raise BoundaryLawError(
            f"{location} requires exactly wall_temperature_K, "
            "diffuse_reflection_fraction, and wall_velocity_m_s"
        )
    temperature = _finite_number(parameters["wall_temperature_K"], f"{location}.wall_temperature_K")
    fraction = _finite_number(
        parameters["diffuse_reflection_fraction"],
        f"{location}.diffuse_reflection_fraction",
    )
    velocity = _finite_vector2(parameters["wall_velocity_m_s"], f"{location}.wall_velocity_m_s")
    if temperature <= 0.0:
        raise BoundaryLawError(f"{location}.wall_temperature_K must be positive")
    if not 0.0 <= fraction <= 1.0:
        raise BoundaryLawError(f"{location}.diffuse_reflection_fraction must be in [0, 1]")
    return temperature, fraction, velocity


def _finite_vector2(value: object, location: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise BoundaryLawError(f"{location} must contain exactly two finite numbers")
    return (
        _finite_number(value[0], f"{location}[0]"),
        _finite_number(value[1], f"{location}[1]"),
    )


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
