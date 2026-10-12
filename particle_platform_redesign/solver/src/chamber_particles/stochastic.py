"""Exact inertial Ornstein--Uhlenbeck Gaussian transition arithmetic.

The serial strict kernels own the numerical body.  Validating Python wrappers
allocate results; native engine continuations supply disjoint reusable buffers.
The same two components apply to Cartesian XY and signed meridional RZ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type UInt8Array = NDArray[np.uint8]

JOINT_OU_REVISION = "inertial_joint_ou_v1"
JOINT_OU_SPLIT_REVISION = "conditional_gaussian_half_split_v1"
MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT = 1.0e6

_SMALL_RELAXATION_ARGUMENT = 5.0e-2
_PSD_ROUNDOFF_FACTOR = 256.0 * np.finfo(np.float64).eps
_FLOAT64_TINY = np.finfo(np.float64).tiny

# Local arithmetic errors, not lifecycle or integrator status.  Into kernels
# overwrite every status row; failed output rows must never be committed.
OU_STATUS_OK = 0
_ARGUMENT_NOT_FINITE = 1
_ARGUMENT_UNDERFLOW = 2
_COVARIANCE_NOT_FINITE = 3
_NEGATIVE_VARIANCE = 4
_COVARIANCE_NOT_PSD = 5
_COVARIANCE_LOST_PSD = 6
_POSITION_NOT_REPRESENTABLE = 7
_PARENT_NOT_POSITIVE_DEFINITE = 8
_CONDITIONAL_NOT_PSD = 9
_STATE_NOT_FINITE = 10
_ZERO_TEMPERATURE_PARENT = 11
_ZERO_DURATION_PARENT = 12

_OU_ERROR_MESSAGES = (
    "",
    "joint OU relaxation argument is not finite",
    "joint OU relaxation argument underflowed to zero",
    "joint OU covariance is not finite",
    "joint OU covariance has a negative variance",
    "joint OU covariance is not positive semidefinite",
    "joint OU covariance lost positive semidefiniteness",
    "joint OU position variance is not representable",
    "joint OU parent covariance is not positive definite",
    "joint OU conditional covariance is not positive semidefinite",
    "joint OU state or increment is not finite",
    "zero-temperature parent increment must be exactly zero",
    "zero-duration parent increment must be exactly zero",
)


@dataclass(frozen=True, slots=True)
class JointOuIncrement:
    """Correlated position and velocity increments in the two stored components."""

    position_m: FloatArray
    velocity_m_s: FloatArray


def joint_ou_covariance(
    drag_rate_s_inv: FloatArray,
    thermal_velocity_variance_m2_s2: FloatArray,
    duration_s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return exact frozen-coefficient ``(Q_xx, Q_xv, Q_vv)`` for one component."""

    rate, thermal, duration = _validated_coefficients(
        drag_rate_s_inv, thermal_velocity_variance_m2_s2, duration_s
    )
    variance_x = np.empty_like(rate)
    covariance_xv = np.empty_like(rate)
    variance_v = np.empty_like(rate)
    status = np.empty(rate.size, dtype=np.uint8)
    joint_ou_covariance_into(rate, thermal, duration, variance_x, covariance_xv, variance_v, status)
    _raise_ou_error(status)
    return variance_x, covariance_xv, variance_v


def joint_ou_increment(
    drag_rate_s_inv: FloatArray,
    thermal_velocity_variance_m2_s2: FloatArray,
    duration_s: FloatArray,
    standard_normal: FloatArray,
) -> JointOuIncrement:
    """Map two independent normals per component to the exact joint increment."""

    rate, thermal, duration = _validated_coefficients(
        drag_rate_s_inv, thermal_velocity_variance_m2_s2, duration_s
    )
    normals = _validated_normals(standard_normal, rate.size)
    position_noise = np.empty((rate.size, 2), dtype=np.float64)
    velocity_noise = np.empty_like(position_noise)
    status = np.empty(rate.size, dtype=np.uint8)
    joint_ou_increment_into(
        rate, thermal, duration, normals, position_noise, velocity_noise, status
    )
    _raise_ou_error(status)
    return JointOuIncrement(position_noise, velocity_noise)


def advance_joint_ou(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    equilibrium_velocity_m_s: FloatArray,
    drag_rate_s_inv: FloatArray,
    thermal_velocity_variance_m2_s2: FloatArray,
    duration_s: FloatArray,
    standard_normal: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Advance with the sole frozen-coefficient OU arithmetic body."""

    rate, thermal, duration = _validated_coefficients(
        drag_rate_s_inv, thermal_velocity_variance_m2_s2, duration_s
    )
    position, velocity, equilibrium = _validated_state(
        position_m, velocity_m_s, equilibrium_velocity_m_s, rate.size
    )
    increment = joint_ou_increment(rate, thermal, duration, standard_normal)
    return advance_joint_ou_with_increment(
        position, velocity, equilibrium, rate, duration, increment
    )


def advance_joint_ou_with_increment(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    equilibrium_velocity_m_s: FloatArray,
    drag_rate_s_inv: FloatArray,
    duration_s: FloatArray,
    increment: JointOuIncrement,
) -> tuple[FloatArray, FloatArray]:
    """Apply the already drawn root or conditional child without another draw."""

    rate, duration = _validated_apply_coefficients(drag_rate_s_inv, duration_s, "apply")
    position, velocity, equilibrium = _validated_state(
        position_m, velocity_m_s, equilibrium_velocity_m_s, rate.size
    )
    noise_position, noise_velocity = _validated_increment(increment, rate.size)
    result_position = np.empty_like(position)
    result_velocity = np.empty_like(velocity)
    status = np.empty(rate.size, dtype=np.uint8)
    advance_joint_ou_with_increment_into(
        position,
        velocity,
        equilibrium,
        rate,
        duration,
        noise_position,
        noise_velocity,
        result_position,
        result_velocity,
        status,
    )
    _raise_ou_error(status)
    return result_position, result_velocity


def split_joint_ou_increment_half(
    parent: JointOuIncrement,
    drag_rate_s_inv: FloatArray,
    thermal_velocity_variance_m2_s2: FloatArray,
    duration_s: FloatArray,
    standard_normal: FloatArray,
) -> tuple[JointOuIncrement, JointOuIncrement]:
    """Conditionally draw the left child and form the algebraic right remainder."""

    rate, thermal, duration = _validated_coefficients(
        drag_rate_s_inv, thermal_velocity_variance_m2_s2, duration_s
    )
    parent_position, parent_velocity = _validated_increment(parent, rate.size)
    normals = _validated_normals(standard_normal, rate.size)
    zero_noise = thermal == 0.0
    if bool(((parent_position[zero_noise] != 0.0) | (parent_velocity[zero_noise] != 0.0)).any()):
        raise ValueError("zero-temperature parent increment must be exactly zero")
    zero_duration = (duration == 0.0)[:, None]
    if bool((zero_duration & (parent_position != 0.0)).any()) or bool(
        (zero_duration & (parent_velocity != 0.0)).any()
    ):
        raise ValueError("zero-duration parent increment must be exactly zero")
    left_position = np.empty_like(parent_position)
    left_velocity = np.empty_like(parent_velocity)
    right_position = np.empty_like(parent_position)
    right_velocity = np.empty_like(parent_velocity)
    status = np.empty(rate.size, dtype=np.uint8)
    split_joint_ou_increment_half_into(
        parent_position,
        parent_velocity,
        rate,
        thermal,
        duration,
        normals,
        left_position,
        left_velocity,
        right_position,
        right_velocity,
        status,
    )
    _raise_ou_error(status)
    return (
        JointOuIncrement(left_position, left_velocity),
        JointOuIncrement(right_position, right_velocity),
    )


def compose_joint_ou_increments(
    left: JointOuIncrement,
    right: JointOuIncrement,
    drag_rate_s_inv: FloatArray,
    half_duration_s: FloatArray,
) -> JointOuIncrement:
    """Compose adjacent half increments with the same frozen transition body."""

    rate, half_duration = _validated_apply_coefficients(drag_rate_s_inv, half_duration_s, "compose")
    left_x, left_v = _validated_increment(left, rate.size)
    right_x, right_v = _validated_increment(right, rate.size)
    position = np.empty_like(left_x)
    velocity = np.empty_like(left_v)
    status = np.empty(rate.size, dtype=np.uint8)
    compose_joint_ou_increments_into(
        left_x, left_v, right_x, right_v, rate, half_duration, position, velocity, status
    )
    _raise_ou_error(status)
    return JointOuIncrement(position, velocity)


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def joint_ou_covariance_into(
    rate: FloatArray,
    thermal: FloatArray,
    duration: FloatArray,
    variance_x: FloatArray,
    covariance_xv: FloatArray,
    variance_v: FloatArray,
    status: UInt8Array,
) -> None:
    """Fill disjoint aligned vectors; inputs satisfy wrapper coefficient validation."""

    for row in range(rate.size):
        qxx, qxv, qvv, error = _joint_ou_covariance_scalar(rate[row], thermal[row], duration[row])
        variance_x[row] = qxx
        covariance_xv[row] = qxv
        variance_v[row] = qvv
        status[row] = error


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def joint_ou_increment_into(
    rate: FloatArray,
    thermal: FloatArray,
    duration: FloatArray,
    normals: FloatArray,
    position_noise: FloatArray,
    velocity_noise: FloatArray,
    status: UInt8Array,
) -> None:
    """Fill disjoint [N,2] buffers from normals in unchanged [row,component,pair] order."""

    for row in range(rate.size):
        scale_x, velocity_from_position, scale_v, error = _joint_ou_increment_scales(
            rate[row], thermal[row], duration[row]
        )
        for component in range(2):
            position_noise[row, component] = 0.0
            velocity_noise[row, component] = 0.0
            if error == OU_STATUS_OK and duration[row] > 0.0 and thermal[row] > 0.0:
                x, v = _joint_ou_increment_component(
                    scale_x,
                    velocity_from_position,
                    scale_v,
                    normals[row, component, 0],
                    normals[row, component, 1],
                )
                position_noise[row, component] = x
                velocity_noise[row, component] = v
                if not (np.isfinite(x) and np.isfinite(v)):
                    error = _STATE_NOT_FINITE
        status[row] = error


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def advance_joint_ou_with_increment_into(
    position: FloatArray,
    velocity: FloatArray,
    equilibrium: FloatArray,
    rate: FloatArray,
    duration: FloatArray,
    noise_position: FloatArray,
    noise_velocity: FloatArray,
    result_position: FloatArray,
    result_velocity: FloatArray,
    status: UInt8Array,
) -> None:
    """Apply validated inputs into [N,2] buffers; exact corresponding state alias is safe.

    Other overlapping views and aliasing a result with coefficients or increments
    are unsupported.  Status is separate and overwritten for every row.
    """

    for row in range(rate.size):
        decay, displacement, error = _joint_ou_transition_scalar(rate[row], duration[row])
        for component in range(2):
            x = 0.0
            v = 0.0
            if error == OU_STATUS_OK:
                x, v = _advance_joint_ou_component(
                    position[row, component],
                    velocity[row, component],
                    equilibrium[row, component],
                    duration[row],
                    decay,
                    displacement,
                    noise_position[row, component],
                    noise_velocity[row, component],
                )
                if not (np.isfinite(x) and np.isfinite(v)):
                    error = _STATE_NOT_FINITE
            result_position[row, component] = x
            result_velocity[row, component] = v
        status[row] = error


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def split_joint_ou_increment_half_into(
    parent_position: FloatArray,
    parent_velocity: FloatArray,
    rate: FloatArray,
    thermal: FloatArray,
    duration: FloatArray,
    normals: FloatArray,
    left_position: FloatArray,
    left_velocity: FloatArray,
    right_position: FloatArray,
    right_velocity: FloatArray,
    status: UInt8Array,
) -> None:
    """Fill four disjoint child buffers without allocation; parent remains owned unchanged."""

    for row in range(rate.size):
        error = _zero_parent_error(
            thermal[row],
            duration[row],
            parent_position[row, 0],
            parent_position[row, 1],
            parent_velocity[row, 0],
            parent_velocity[row, 1],
        )
        coefficients = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, OU_STATUS_OK)
        active = thermal[row] > 0.0 and duration[row] > 0.0
        if error == OU_STATUS_OK and active:
            coefficients = _half_split_coefficients_scalar(rate[row], duration[row])
            error = coefficients[9]
        for component in range(2):
            x_left, v_left, x_right, v_right = 0.0, 0.0, 0.0, 0.0
            if error == OU_STATUS_OK and active:
                x_left, v_left, x_right, v_right = _split_joint_ou_component(
                    parent_position[row, component],
                    parent_velocity[row, component],
                    thermal[row],
                    duration[row],
                    normals[row, component, 0],
                    normals[row, component, 1],
                    coefficients,
                )
                if not _finite_children(x_left, v_left, x_right, v_right):
                    error = _STATE_NOT_FINITE
            left_position[row, component] = x_left
            left_velocity[row, component] = v_left
            right_position[row, component] = x_right
            right_velocity[row, component] = v_right
        status[row] = error


@njit(cache=True, fastmath=False, parallel=False, nogil=True, error_model="numpy")
def compose_joint_ou_increments_into(
    left_x: FloatArray,
    left_v: FloatArray,
    right_x: FloatArray,
    right_v: FloatArray,
    rate: FloatArray,
    half_duration: FloatArray,
    position: FloatArray,
    velocity: FloatArray,
    status: UInt8Array,
) -> None:
    """Compose validated increments into two disjoint caller-owned buffers."""

    for row in range(rate.size):
        decay, displacement, error = _joint_ou_transition_scalar(rate[row], half_duration[row])
        for component in range(2):
            x = 0.0
            v = 0.0
            if error == OU_STATUS_OK:
                x = left_x[row, component] + displacement * left_v[row, component]
                x = x + right_x[row, component]
                v = decay * left_v[row, component] + right_v[row, component]
                if not (np.isfinite(x) and np.isfinite(v)):
                    error = _STATE_NOT_FINITE
            position[row, component] = x
            velocity[row, component] = v
        status[row] = error


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _joint_ou_covariance_scalar(
    rate: float, thermal: float, duration: float
) -> tuple[float, float, float, int]:
    argument = rate * duration
    error = _relaxation_argument_error(argument, duration)
    if error != OU_STATUS_OK:
        return 0.0, 0.0, 0.0, error
    position_factor = _position_variance_factor_scalar(argument)
    one_minus_decay = -np.expm1(-argument)
    velocity_factor = -np.expm1(-2.0 * argument)
    cross_factor = 0.0
    if duration > 0.0:
        cross_factor = one_minus_decay * one_minus_decay / argument
    variance_x = thermal * duration * duration * position_factor
    covariance_xv = thermal * duration * cross_factor
    variance_v = thermal * velocity_factor
    error = _covariance_error(variance_x, covariance_xv, variance_v)
    return variance_x, covariance_xv, variance_v, error


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _joint_ou_increment_scales(
    rate: float, thermal: float, duration: float
) -> tuple[float, float, float, int]:
    variance_x, covariance_xv, variance_v, error = _joint_ou_covariance_scalar(
        rate, thermal, duration
    )
    if error != OU_STATUS_OK or duration == 0.0 or thermal == 0.0:
        return 0.0, 0.0, 0.0, error
    position_scale = np.sqrt(variance_x)
    if position_scale == 0.0:
        return 0.0, 0.0, 0.0, _POSITION_NOT_REPRESENTABLE
    velocity_from_position = covariance_xv / position_scale
    residual = variance_v - velocity_from_position**2
    residual, error = _roundoff_clamp_nonnegative_scalar(residual, variance_v)
    return position_scale, velocity_from_position, np.sqrt(residual), error


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _joint_ou_increment_component(
    scale_x: float, velocity_from_position: float, scale_v: float, first: float, second: float
) -> tuple[float, float]:
    return scale_x * first, velocity_from_position * first + scale_v * second


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _joint_ou_transition_scalar(rate: float, duration: float) -> tuple[float, float, int]:
    argument = rate * duration
    error = _relaxation_argument_error(argument, duration)
    if error != OU_STATUS_OK:
        return 0.0, 0.0, error
    displacement = 0.0
    if duration > 0.0:
        displacement = duration * (-np.expm1(-argument) / argument)
    return np.exp(-argument), displacement, OU_STATUS_OK


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _advance_joint_ou_component(
    position: float,
    velocity: float,
    equilibrium: float,
    duration: float,
    decay: float,
    displacement: float,
    noise_position: float,
    noise_velocity: float,
) -> tuple[float, float]:
    relative = velocity - equilibrium
    result_position = position + equilibrium * duration + relative * displacement + noise_position
    result_velocity = equilibrium + decay * relative + noise_velocity
    return result_position, result_velocity


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _half_split_coefficients_scalar(
    rate: float, duration: float
) -> tuple[float, float, float, float, float, float, float, float, float, int]:
    argument = rate * duration
    error = _relaxation_argument_error(argument, duration)
    if error != OU_STATUS_OK:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, error
    half_argument = 0.5 * argument
    parent_r00 = _position_variance_factor_over_argument_scalar(argument)
    parent_r01 = _one_minus_decay_over_argument_scalar(argument) ** 2
    parent_r11 = -np.expm1(-2.0 * argument) / argument
    half_phi = _one_minus_decay_over_argument_scalar(half_argument)
    left_r00 = 0.125 * _position_variance_factor_over_argument_scalar(half_argument)
    left_r01 = 0.25 * half_phi**2
    left_r11 = 0.5 * (-np.expm1(-2.0 * half_argument) / half_argument)
    half_decay = np.exp(-half_argument)
    half_displacement_over_duration = 0.5 * half_phi
    cross00 = left_r00 + half_displacement_over_duration * left_r01
    cross01 = half_decay * left_r01
    cross10 = left_r01 + half_displacement_over_duration * left_r11
    cross11 = half_decay * left_r11
    determinant = parent_r00 * parent_r11 - parent_r01**2
    if not np.isfinite(determinant) or determinant <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _PARENT_NOT_POSITIVE_DEFINITE
    gain00 = (cross00 * parent_r11 - cross01 * parent_r01) / determinant
    gain01 = (-cross00 * parent_r01 + cross01 * parent_r00) / determinant
    gain10 = (cross10 * parent_r11 - cross11 * parent_r01) / determinant
    gain11 = (-cross10 * parent_r01 + cross11 * parent_r00) / determinant
    residual00 = left_r00 - (gain00 * cross00 + gain01 * cross01)
    # Equal-half conditioning has exactly zero residual x/v cross covariance.
    # Retain the analytic zero instead of subtracting correlated O(1) terms.
    residual01 = 0.0
    residual11 = left_r11 - (gain10 * cross10 + gain11 * cross11)
    residual00, error00 = _roundoff_clamp_nonnegative_scalar(residual00, left_r00)
    residual11, error11 = _roundoff_clamp_nonnegative_scalar(residual11, left_r11)
    error = error00 if error00 != OU_STATUS_OK else error11
    determinant_residual = residual00 * residual11 - residual01**2
    scale = max(residual00 * residual11, _FLOAT64_TINY)
    if error == OU_STATUS_OK and determinant_residual < -_PSD_ROUNDOFF_FACTOR * scale:
        error = _CONDITIONAL_NOT_PSD
    cholesky00 = np.sqrt(residual00 * argument)
    cholesky10 = residual01 * argument / cholesky00
    cholesky11_variance = residual11 * argument - cholesky10**2
    cholesky11_variance, error11 = _roundoff_clamp_nonnegative_scalar(
        cholesky11_variance, residual11 * argument
    )
    if error == OU_STATUS_OK:
        error = error11
    return (
        gain00,
        gain01,
        gain10,
        gain11,
        cholesky00,
        cholesky10,
        np.sqrt(cholesky11_variance),
        half_decay,
        half_displacement_over_duration,
        error,
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _split_joint_ou_component(
    parent_x: float,
    parent_v: float,
    thermal: float,
    duration: float,
    first: float,
    second: float,
    coefficients: tuple[float, float, float, float, float, float, float, float, float, int],
) -> tuple[float, float, float, float]:
    gain00, gain01, gain10, gain11, cholesky00, cholesky10, cholesky11, decay, displacement, _ = (
        coefficients
    )
    sqrt_thermal = np.sqrt(thermal)
    parent_x_normalized = parent_x / (sqrt_thermal * duration)
    parent_v_normalized = parent_v / sqrt_thermal
    mean_x = gain00 * parent_x_normalized + gain01 * parent_v_normalized
    mean_v = gain10 * parent_x_normalized + gain11 * parent_v_normalized
    normalized_x = mean_x + cholesky00 * first
    normalized_v = mean_v + cholesky10 * first + cholesky11 * second
    left_x = sqrt_thermal * duration * normalized_x
    left_v = sqrt_thermal * normalized_v
    right_x = parent_x - left_x - (duration * displacement) * left_v
    right_v = parent_v - decay * left_v
    return left_x, left_v, right_x, right_v


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _position_small_factor_over_argument(argument: float) -> float:
    z = argument
    return 2.0 / 3.0 + z * (
        -1.0 / 2.0
        + z
        * (
            7.0 / 30.0
            + z
            * (
                -1.0 / 12.0
                + z
                * (31.0 / 1260.0 + z * (-1.0 / 160.0 + z * (127.0 / 90720.0 - z * 17.0 / 60480.0)))
            )
        )
    )


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _position_variance_factor_scalar(argument: float) -> float:
    if argument <= 0.0:
        return 0.0
    if argument < _SMALL_RELAXATION_ARGUMENT:
        return argument * _position_small_factor_over_argument(argument)
    q = -np.expm1(-argument)
    return (2.0 * argument - 2.0 * q - q * q) / (argument * argument)


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _position_variance_factor_over_argument_scalar(argument: float) -> float:
    if argument < _SMALL_RELAXATION_ARGUMENT:
        return _position_small_factor_over_argument(argument)
    return _position_variance_factor_scalar(argument) / argument


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _one_minus_decay_over_argument_scalar(argument: float) -> float:
    z = argument
    if z < _SMALL_RELAXATION_ARGUMENT:
        return 1.0 + z * (
            -1.0 / 2.0
            + z
            * (1.0 / 6.0 + z * (-1.0 / 24.0 + z * (1.0 / 120.0 + z * (-1.0 / 720.0 + z / 5040.0))))
        )
    return -np.expm1(-z) / z


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _relaxation_argument_error(argument: float, duration: float) -> int:
    if not np.isfinite(argument):
        return _ARGUMENT_NOT_FINITE
    if duration > 0.0 and argument == 0.0:
        return _ARGUMENT_UNDERFLOW
    return OU_STATUS_OK


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _covariance_error(variance_x: float, covariance_xv: float, variance_v: float) -> int:
    if not (np.isfinite(variance_x) and np.isfinite(covariance_xv) and np.isfinite(variance_v)):
        return _COVARIANCE_NOT_FINITE
    if variance_x < 0.0 or variance_v < 0.0:
        return _NEGATIVE_VARIANCE
    if variance_x > 0.0 and variance_v > 0.0:
        correlation = covariance_xv / np.sqrt(variance_x) / np.sqrt(variance_v)
        if abs(correlation) > 1.0 + _PSD_ROUNDOFF_FACTOR:
            return _COVARIANCE_NOT_PSD
    elif covariance_xv != 0.0:
        return _COVARIANCE_NOT_PSD
    return OU_STATUS_OK


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _roundoff_clamp_nonnegative_scalar(value: float, scale: float) -> tuple[float, int]:
    tolerance = _PSD_ROUNDOFF_FACTOR * max(abs(scale), _FLOAT64_TINY)
    if value < -tolerance:
        return value, _COVARIANCE_LOST_PSD
    if value < 0.0:
        return 0.0, OU_STATUS_OK
    return value, OU_STATUS_OK


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _zero_parent_error(
    thermal: float, duration: float, x0: float, x1: float, v0: float, v1: float
) -> int:
    nonzero = x0 != 0.0 or x1 != 0.0 or v0 != 0.0 or v1 != 0.0
    if thermal == 0.0 and nonzero:
        return _ZERO_TEMPERATURE_PARENT
    if duration == 0.0 and nonzero:
        return _ZERO_DURATION_PARENT
    return OU_STATUS_OK


@njit(inline="always", fastmath=False, parallel=False, error_model="numpy")
def _finite_children(x_left: float, v_left: float, x_right: float, v_right: float) -> bool:
    return (
        np.isfinite(x_left)
        and np.isfinite(v_left)
        and np.isfinite(x_right)
        and np.isfinite(v_right)
    )


def _raise_ou_error(status: UInt8Array) -> None:
    failed = np.flatnonzero(status != OU_STATUS_OK)
    if failed.size == 0:
        return
    error = int(status[failed[0]])
    if error in (_ZERO_TEMPERATURE_PARENT, _ZERO_DURATION_PARENT):
        raise ValueError(_OU_ERROR_MESSAGES[error])
    raise FloatingPointError(_OU_ERROR_MESSAGES[error])


def _validated_apply_coefficients(
    rate_value: FloatArray, duration_value: FloatArray, operation: str
) -> tuple[FloatArray, FloatArray]:
    rate = np.asarray(rate_value, dtype=np.float64)
    duration = np.asarray(duration_value, dtype=np.float64)
    if rate.ndim != 1 or duration.shape != rate.shape:
        raise ValueError(f"joint OU {operation} coefficients must be aligned vectors")
    if not bool(np.isfinite(rate).all() and np.isfinite(duration).all()):
        raise ValueError(f"joint OU {operation} coefficients must be finite")
    if bool((rate <= 0.0).any() or (duration < 0.0).any()):
        raise ValueError(f"joint OU {operation} requires positive rate and nonnegative duration")
    return rate, duration


def _validated_coefficients(
    drag_rate_s_inv: FloatArray,
    thermal_velocity_variance_m2_s2: FloatArray,
    duration_s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    rate = np.asarray(drag_rate_s_inv, dtype=np.float64)
    thermal = np.asarray(thermal_velocity_variance_m2_s2, dtype=np.float64)
    duration = np.asarray(duration_s, dtype=np.float64)
    if rate.ndim != 1 or thermal.shape != rate.shape or duration.shape != rate.shape:
        raise ValueError("joint OU coefficients must be aligned vectors")
    if not bool(
        np.isfinite(rate).all() and np.isfinite(thermal).all() and np.isfinite(duration).all()
    ):
        raise ValueError("joint OU coefficients must be finite")
    if bool((rate <= 0.0).any() or (thermal < 0.0).any() or (duration < 0.0).any()):
        raise ValueError(
            "joint OU requires positive drag rate and nonnegative temperature and duration"
        )
    return rate, thermal, duration


def _validated_state(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    equilibrium_velocity_m_s: FloatArray,
    count: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    equilibrium = np.asarray(equilibrium_velocity_m_s, dtype=np.float64)
    if position.shape != (count, 2) or velocity.shape != (count, 2):
        raise ValueError("joint OU state must have shape [N, 2]")
    if equilibrium.shape != (count, 2):
        raise ValueError("joint OU equilibrium velocity must have shape [N, 2]")
    if not bool(
        np.isfinite(position).all()
        and np.isfinite(velocity).all()
        and np.isfinite(equilibrium).all()
    ):
        raise ValueError("joint OU state must be finite")
    return position, velocity, equilibrium


def _validated_normals(standard_normal: FloatArray, count: int) -> FloatArray:
    normals = np.asarray(standard_normal, dtype=np.float64)
    if normals.shape != (count, 2, 2) or not bool(np.isfinite(normals).all()):
        raise ValueError("joint OU normals must be one finite [N, 2, 2] array")
    return normals


def _validated_increment(
    increment: JointOuIncrement,
    count: int,
) -> tuple[FloatArray, FloatArray]:
    position = np.asarray(increment.position_m, dtype=np.float64)
    velocity = np.asarray(increment.velocity_m_s, dtype=np.float64)
    if position.shape != (count, 2) or velocity.shape != (count, 2):
        raise ValueError("joint OU increment must have shape [N, 2]")
    _require_finite_increment(position, velocity)
    return position, velocity


def _require_finite_increment(position: FloatArray, velocity: FloatArray) -> None:
    if not bool(np.isfinite(position).all() and np.isfinite(velocity).all()):
        raise FloatingPointError("joint OU state or increment is not finite")
