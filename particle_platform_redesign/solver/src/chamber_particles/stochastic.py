"""Exact inertial Ornstein--Uhlenbeck numerics for the Brownian workstream.

This module deliberately owns only Gaussian transition arithmetic.  It does
not choose physics models, locate fields, detect walls, or advance production
state.  The production engine consumes these primitives after the catalog has
selected the supported inertial-Langevin profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

JOINT_OU_REVISION = "inertial_joint_ou_v1"
JOINT_OU_SPLIT_REVISION = "conditional_gaussian_half_split_v1"
MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT = 1.0e6

_SMALL_RELAXATION_ARGUMENT = 5.0e-2
_PSD_ROUNDOFF_FACTOR = 256.0 * np.finfo(np.float64).eps


@dataclass(frozen=True, slots=True)
class JointOuIncrement:
    """Correlated stochastic position and velocity increments in Cartesian XY."""

    position_m: FloatArray
    velocity_m_s: FloatArray


def joint_ou_covariance(
    drag_rate_s_inv: FloatArray,
    thermal_velocity_variance_m2_s2: FloatArray,
    duration_s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(Q_xx, Q_xv, Q_vv)`` for one Cartesian component.

    ``thermal_velocity_variance_m2_s2`` is :math:`k_B T/m`.  The returned
    covariance is exact for coefficients frozen over each row's interval.
    """

    rate, thermal, duration = _validated_coefficients(
        drag_rate_s_inv,
        thermal_velocity_variance_m2_s2,
        duration_s,
    )
    argument = rate * duration
    _require_resolved_positive_intervals(argument, duration)
    position_factor = _position_variance_factor(argument)
    one_minus_decay = -np.expm1(-argument)
    velocity_factor = -np.expm1(-2.0 * argument)
    cross_factor = np.zeros_like(argument)
    moving = duration > 0.0
    cross_factor[moving] = one_minus_decay[moving] * one_minus_decay[moving] / argument[moving]
    with np.errstate(over="ignore", invalid="ignore"):
        variance_x = thermal * duration * duration * position_factor
        covariance_xv = thermal * duration * cross_factor
        variance_v = thermal * velocity_factor
    _require_finite_nonnegative_covariance(variance_x, covariance_xv, variance_v)
    return variance_x, covariance_xv, variance_v


def joint_ou_increment(
    drag_rate_s_inv: FloatArray,
    thermal_velocity_variance_m2_s2: FloatArray,
    duration_s: FloatArray,
    standard_normal: FloatArray,
) -> JointOuIncrement:
    """Map two independent normals per component to the exact joint increment."""

    rate, thermal, duration = _validated_coefficients(
        drag_rate_s_inv,
        thermal_velocity_variance_m2_s2,
        duration_s,
    )
    normals = _validated_normals(standard_normal, rate.size)
    variance_x, covariance_xv, variance_v = joint_ou_covariance(
        rate,
        thermal,
        duration,
    )
    position_noise = np.zeros((rate.size, 2), dtype=np.float64)
    velocity_noise = np.zeros((rate.size, 2), dtype=np.float64)
    active = (duration > 0.0) & (thermal > 0.0)
    if bool(active.any()):
        rows = np.flatnonzero(active)
        position_scale = np.sqrt(variance_x[rows])
        if bool((position_scale == 0.0).any()):
            raise FloatingPointError("joint OU position variance is not representable")
        velocity_from_position = covariance_xv[rows] / position_scale
        residual_variance = variance_v[rows] - velocity_from_position**2
        residual_variance = _roundoff_clamp_nonnegative(
            residual_variance,
            variance_v[rows],
        )
        velocity_scale = np.sqrt(residual_variance)
        for component in range(2):
            first = normals[rows, component, 0]
            second = normals[rows, component, 1]
            position_noise[rows, component] = position_scale * first
            velocity_noise[rows, component] = (
                velocity_from_position * first + velocity_scale * second
            )
    _require_finite_increment(position_noise, velocity_noise)
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
    """Advance position and velocity with an exact frozen-coefficient OU step."""

    rate, thermal, duration = _validated_coefficients(
        drag_rate_s_inv,
        thermal_velocity_variance_m2_s2,
        duration_s,
    )
    position, velocity, equilibrium = _validated_state(
        position_m,
        velocity_m_s,
        equilibrium_velocity_m_s,
        rate.size,
    )
    argument = rate * duration
    _require_resolved_positive_intervals(argument, duration)
    one_minus_decay = -np.expm1(-argument)
    displacement_factor = np.zeros_like(argument)
    moving = duration > 0.0
    displacement_factor[moving] = duration[moving] * (one_minus_decay[moving] / argument[moving])
    relative = velocity - equilibrium
    mean_position = (
        position + equilibrium * duration[:, None] + relative * displacement_factor[:, None]
    )
    mean_velocity = equilibrium + np.exp(-argument)[:, None] * relative
    increment = joint_ou_increment(rate, thermal, duration, standard_normal)
    result_position = mean_position + increment.position_m
    result_velocity = mean_velocity + increment.velocity_m_s
    _require_finite_increment(result_position, result_velocity)
    return result_position, result_velocity


def advance_joint_ou_with_increment(
    position_m: FloatArray,
    velocity_m_s: FloatArray,
    equilibrium_velocity_m_s: FloatArray,
    drag_rate_s_inv: FloatArray,
    duration_s: FloatArray,
    increment: JointOuIncrement,
) -> tuple[FloatArray, FloatArray]:
    """Apply one already-drawn frozen-coefficient joint OU increment.

    Production interval trees draw the root once and conditionally split that
    same increment.  Applying a supplied increment keeps event refinement and
    output replay from drawing a different stochastic path.
    """

    rate = np.asarray(drag_rate_s_inv, dtype=np.float64)
    duration = np.asarray(duration_s, dtype=np.float64)
    if rate.ndim != 1 or duration.shape != rate.shape:
        raise ValueError("joint OU apply coefficients must be aligned vectors")
    if not bool(np.isfinite(rate).all() and np.isfinite(duration).all()):
        raise ValueError("joint OU apply coefficients must be finite")
    if bool((rate <= 0.0).any() or (duration < 0.0).any()):
        raise ValueError("joint OU apply requires positive rate and nonnegative duration")
    position, velocity, equilibrium = _validated_state(
        position_m,
        velocity_m_s,
        equilibrium_velocity_m_s,
        rate.size,
    )
    noise_position, noise_velocity = _validated_increment(increment, rate.size)
    argument = rate * duration
    _require_resolved_positive_intervals(argument, duration)
    one_minus_decay = -np.expm1(-argument)
    displacement_factor = np.zeros_like(argument)
    moving = duration > 0.0
    displacement_factor[moving] = duration[moving] * (one_minus_decay[moving] / argument[moving])
    relative = velocity - equilibrium
    result_position = (
        position
        + equilibrium * duration[:, None]
        + relative * displacement_factor[:, None]
        + noise_position
    )
    result_velocity = equilibrium + np.exp(-argument)[:, None] * relative + noise_velocity
    _require_finite_increment(result_position, result_velocity)
    return result_position, result_velocity


def split_joint_ou_increment_half(
    parent: JointOuIncrement,
    drag_rate_s_inv: FloatArray,
    thermal_velocity_variance_m2_s2: FloatArray,
    duration_s: FloatArray,
    standard_normal: FloatArray,
) -> tuple[JointOuIncrement, JointOuIncrement]:
    """Conditionally split a parent increment into equal left and right halves.

    The left draw has its exact conditional Gaussian distribution given the
    parent increment.  The right increment is then the algebraic remainder,
    so composing both children preserves the already drawn parent endpoint.
    """

    rate, thermal, duration = _validated_coefficients(
        drag_rate_s_inv,
        thermal_velocity_variance_m2_s2,
        duration_s,
    )
    parent_position, parent_velocity = _validated_increment(parent, rate.size)
    normals = _validated_normals(standard_normal, rate.size)
    left_position = np.zeros_like(parent_position)
    left_velocity = np.zeros_like(parent_velocity)
    right_position = np.zeros_like(parent_position)
    right_velocity = np.zeros_like(parent_velocity)

    zero_noise = thermal == 0.0
    if bool(((parent_position[zero_noise] != 0.0) | (parent_velocity[zero_noise] != 0.0)).any()):
        raise ValueError("zero-temperature parent increment must be exactly zero")
    active = (duration > 0.0) & ~zero_noise
    if bool(((duration == 0.0)[:, None] & (parent_position != 0.0)).any()) or bool(
        ((duration == 0.0)[:, None] & (parent_velocity != 0.0)).any()
    ):
        raise ValueError("zero-duration parent increment must be exactly zero")
    if not bool(active.any()):
        return (
            JointOuIncrement(left_position, left_velocity),
            JointOuIncrement(right_position, right_velocity),
        )

    rows = np.flatnonzero(active)
    argument = rate[rows] * duration[rows]
    _require_resolved_positive_intervals(argument, duration[rows])
    coefficients = _half_split_coefficients(argument)
    (
        gain00,
        gain01,
        gain10,
        gain11,
        residual00,
        residual01,
        residual11,
        half_decay,
        half_displacement_over_duration,
    ) = coefficients

    sqrt_thermal = np.sqrt(thermal[rows])
    parent_x_normalized = parent_position[rows] / (sqrt_thermal[:, None] * duration[rows, None])
    parent_v_normalized = parent_velocity[rows] / sqrt_thermal[:, None]
    mean_x = gain00[:, None] * parent_x_normalized + gain01[:, None] * parent_v_normalized
    mean_v = gain10[:, None] * parent_x_normalized + gain11[:, None] * parent_v_normalized

    cholesky00 = np.sqrt(residual00 * argument)
    cholesky10 = residual01 * argument / cholesky00
    cholesky11_variance = residual11 * argument - cholesky10**2
    cholesky11_variance = _roundoff_clamp_nonnegative(
        cholesky11_variance,
        residual11 * argument,
    )
    cholesky11 = np.sqrt(cholesky11_variance)
    for component in range(2):
        first = normals[rows, component, 0]
        second = normals[rows, component, 1]
        normalized_x = mean_x[:, component] + cholesky00 * first
        normalized_v = mean_v[:, component] + cholesky10 * first + cholesky11 * second
        left_position[rows, component] = sqrt_thermal * duration[rows] * normalized_x
        left_velocity[rows, component] = sqrt_thermal * normalized_v

    right_position[rows] = (
        parent_position[rows]
        - left_position[rows]
        - (duration[rows] * half_displacement_over_duration)[:, None] * left_velocity[rows]
    )
    right_velocity[rows] = parent_velocity[rows] - half_decay[:, None] * left_velocity[rows]
    _require_finite_increment(left_position, left_velocity)
    _require_finite_increment(right_position, right_velocity)
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
    """Compose two adjacent half-interval increments for verification and reuse."""

    rate = np.asarray(drag_rate_s_inv, dtype=np.float64)
    half_duration = np.asarray(half_duration_s, dtype=np.float64)
    if rate.ndim != 1 or half_duration.shape != rate.shape:
        raise ValueError("joint OU compose coefficients must be aligned vectors")
    if not bool(np.isfinite(rate).all() and np.isfinite(half_duration).all()):
        raise ValueError("joint OU compose coefficients must be finite")
    if bool((rate <= 0.0).any() or (half_duration < 0.0).any()):
        raise ValueError("joint OU compose requires positive rate and nonnegative duration")
    left_x, left_v = _validated_increment(left, rate.size)
    right_x, right_v = _validated_increment(right, rate.size)
    argument = rate * half_duration
    _require_resolved_positive_intervals(argument, half_duration)
    decay = np.exp(-argument)
    displacement = np.zeros_like(argument)
    moving = half_duration > 0.0
    displacement[moving] = half_duration[moving] * (-np.expm1(-argument[moving]) / argument[moving])
    position = left_x + displacement[:, None] * left_v + right_x
    velocity = decay[:, None] * left_v + right_v
    _require_finite_increment(position, velocity)
    return JointOuIncrement(position, velocity)


def _half_split_coefficients(
    argument: FloatArray,
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
]:
    """Return dimensionless conditional-Gaussian coefficients for a half split."""

    half = 0.5 * argument
    parent_r00 = _position_variance_factor_over_argument(argument)
    parent_r01 = _one_minus_decay_over_argument(argument) ** 2
    parent_r11 = -np.expm1(-2.0 * argument) / argument

    half_phi = _one_minus_decay_over_argument(half)
    left_r00 = 0.125 * _position_variance_factor_over_argument(half)
    left_r01 = 0.25 * half_phi**2
    left_r11 = 0.5 * (-np.expm1(-2.0 * half) / half)
    half_decay = np.exp(-half)
    half_displacement_over_duration = 0.5 * half_phi

    cross00 = left_r00 + half_displacement_over_duration * left_r01
    cross01 = half_decay * left_r01
    cross10 = left_r01 + half_displacement_over_duration * left_r11
    cross11 = half_decay * left_r11

    determinant = parent_r00 * parent_r11 - parent_r01**2
    if bool((~np.isfinite(determinant) | (determinant <= 0.0)).any()):
        raise FloatingPointError("joint OU parent covariance is not positive definite")
    gain00 = (cross00 * parent_r11 - cross01 * parent_r01) / determinant
    gain01 = (-cross00 * parent_r01 + cross01 * parent_r00) / determinant
    gain10 = (cross10 * parent_r11 - cross11 * parent_r01) / determinant
    gain11 = (-cross10 * parent_r01 + cross11 * parent_r00) / determinant

    residual00 = left_r00 - (gain00 * cross00 + gain01 * cross01)
    # For an equal split the conditional x/v residual cross covariance
    # vanishes analytically.  Forming it as a difference of O(1) terms creates
    # a false correlation around the small-z series threshold.
    residual01 = np.zeros_like(residual00)
    residual11 = left_r11 - (gain10 * cross10 + gain11 * cross11)
    residual00 = _roundoff_clamp_nonnegative(residual00, left_r00)
    residual11 = _roundoff_clamp_nonnegative(residual11, left_r11)
    determinant_residual = residual00 * residual11 - residual01**2
    scale = np.maximum(residual00 * residual11, np.finfo(np.float64).tiny)
    bad = determinant_residual < -_PSD_ROUNDOFF_FACTOR * scale
    if bool(bad.any()):
        raise FloatingPointError("joint OU conditional covariance is not positive semidefinite")
    return (
        gain00,
        gain01,
        gain10,
        gain11,
        residual00,
        residual01,
        residual11,
        half_decay,
        half_displacement_over_duration,
    )


def _position_variance_factor(argument: FloatArray) -> FloatArray:
    """Return ``(2z - 2q - q**2) / z**2`` without small-z cancellation."""

    result = np.zeros_like(argument)
    moving = argument > 0.0
    small = moving & (argument < _SMALL_RELAXATION_ARGUMENT)
    if bool(small.any()):
        z = argument[small]
        result[small] = z * (
            2.0 / 3.0
            + z
            * (
                -1.0 / 2.0
                + z
                * (
                    7.0 / 30.0
                    + z
                    * (
                        -1.0 / 12.0
                        + z
                        * (
                            31.0 / 1260.0
                            + z * (-1.0 / 160.0 + z * (127.0 / 90720.0 - z * 17.0 / 60480.0))
                        )
                    )
                )
            )
        )
    regular = moving & ~small
    if bool(regular.any()):
        z = argument[regular]
        q = -np.expm1(-z)
        result[regular] = (2.0 * z - 2.0 * q - q * q) / (z * z)
    return result


def _position_variance_factor_over_argument(argument: FloatArray) -> FloatArray:
    """Return the position factor divided by ``z`` with its finite zero limit."""

    result = np.empty_like(argument)
    small = argument < _SMALL_RELAXATION_ARGUMENT
    if bool(small.any()):
        z = argument[small]
        result[small] = 2.0 / 3.0 + z * (
            -1.0 / 2.0
            + z
            * (
                7.0 / 30.0
                + z
                * (
                    -1.0 / 12.0
                    + z
                    * (
                        31.0 / 1260.0
                        + z * (-1.0 / 160.0 + z * (127.0 / 90720.0 - z * 17.0 / 60480.0))
                    )
                )
            )
        )
    regular = ~small
    if bool(regular.any()):
        result[regular] = _position_variance_factor(argument[regular]) / argument[regular]
    return result


def _one_minus_decay_over_argument(argument: FloatArray) -> FloatArray:
    result = np.empty_like(argument)
    small = argument < _SMALL_RELAXATION_ARGUMENT
    if bool(small.any()):
        z = argument[small]
        result[small] = 1.0 + z * (
            -1.0 / 2.0
            + z
            * (1.0 / 6.0 + z * (-1.0 / 24.0 + z * (1.0 / 120.0 + z * (-1.0 / 720.0 + z / 5040.0))))
        )
    regular = ~small
    if bool(regular.any()):
        result[regular] = -np.expm1(-argument[regular]) / argument[regular]
    return result


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


def _require_resolved_positive_intervals(argument: FloatArray, duration: FloatArray) -> None:
    if not bool(np.isfinite(argument).all()):
        raise FloatingPointError("joint OU relaxation argument is not finite")
    if bool(((duration > 0.0) & (argument == 0.0)).any()):
        raise FloatingPointError("joint OU relaxation argument underflowed to zero")


def _require_finite_nonnegative_covariance(
    variance_x: FloatArray,
    covariance_xv: FloatArray,
    variance_v: FloatArray,
) -> None:
    if not bool(
        np.isfinite(variance_x).all()
        and np.isfinite(covariance_xv).all()
        and np.isfinite(variance_v).all()
    ):
        raise FloatingPointError("joint OU covariance is not finite")
    if bool((variance_x < 0.0).any() or (variance_v < 0.0).any()):
        raise FloatingPointError("joint OU covariance has a negative variance")
    positive = (variance_x > 0.0) & (variance_v > 0.0)
    correlation = np.zeros_like(covariance_xv)
    correlation[positive] = (
        covariance_xv[positive] / np.sqrt(variance_x[positive]) / np.sqrt(variance_v[positive])
    )
    zero_variance_with_covariance = ~positive & (covariance_xv != 0.0)
    if bool(zero_variance_with_covariance.any()) or bool(
        (np.abs(correlation) > 1.0 + _PSD_ROUNDOFF_FACTOR).any()
    ):
        raise FloatingPointError("joint OU covariance is not positive semidefinite")


def _roundoff_clamp_nonnegative(value: FloatArray, scale: FloatArray) -> FloatArray:
    result = value.copy()
    tolerance = _PSD_ROUNDOFF_FACTOR * np.maximum(np.abs(scale), np.finfo(np.float64).tiny)
    if bool((result < -tolerance).any()):
        raise FloatingPointError("joint OU covariance lost positive semidefiniteness")
    result[result < 0.0] = 0.0
    return result


def _require_finite_increment(position: FloatArray, velocity: FloatArray) -> None:
    if not bool(np.isfinite(position).all() and np.isfinite(velocity).all()):
        raise FloatingPointError("joint OU state or increment is not finite")
