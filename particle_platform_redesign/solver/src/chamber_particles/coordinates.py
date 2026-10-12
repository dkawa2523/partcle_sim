"""Coordinate transforms that are independent of fields, geometry, and physics."""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def rz_signed_stage_into(
    position_m: FloatArray,
    vector: FloatArray,
    canonical_position_m: FloatArray,
    canonical_vector: FloatArray,
    radial_sign: FloatArray,
) -> None:
    """Apply the signed chart to validated rows in caller-owned stage buffers."""

    for row in range(position_m.shape[0]):
        sign = -1.0 if position_m[row, 0] < 0.0 else 1.0
        radial_sign[row] = sign
        canonical_position_m[row, 0] = abs(position_m[row, 0])
        canonical_position_m[row, 1] = position_m[row, 1]
        canonical_vector[row, 0] = vector[row, 0] * sign
        canonical_vector[row, 1] = vector[row, 1]


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def rz_vector_to_signed_into(
    vector: FloatArray, radial_sign: FloatArray, transformed: FloatArray
) -> None:
    """Transform one canonical vector, with exact in-place alias permitted."""

    for row in range(radial_sign.size):
        transformed[row, 0] = vector[row, 0] * radial_sign[row]
        transformed[row, 1] = vector[row, 1]


def rz_signed_stage_to_canonical(
    position_m: FloatArray, vector: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map batched signed-meridional trial states to the canonical RZ basis.

    The signed radial chart lets an integrator take ordinary smooth stages
    through the axis.  Fields and force models, however, are evaluated in the
    canonical ``r >= 0`` chart.  ``radial_sign`` records the basis orientation
    needed to map a canonical result back to the signed trial chart.  The axis
    itself uses the positive orientation; accepted inward states at the axis
    are handled by :func:`fold_rz_position_vector`.
    """

    position = _finite_batch(position_m, "position_m")
    transformed = _finite_batch(vector, "vector")
    radial_sign = np.empty(position.shape[0], dtype=np.float64)
    rz_signed_stage_into(position, transformed, position, transformed, radial_sign)
    return position, transformed, radial_sign


def rz_canonical_vector_to_signed(vector: FloatArray, radial_sign: FloatArray) -> FloatArray:
    """Map batched canonical RZ vectors back to a signed-meridional chart."""

    transformed = _finite_batch(vector, "vector")
    orientation = np.asarray(radial_sign, dtype=np.float64)
    if orientation.shape != (transformed.shape[0],):
        raise ValueError("radial_sign must have shape [N]")
    if not bool(np.isfinite(orientation).all()) or not bool(
        ((orientation == -1.0) | (orientation == 1.0)).all()
    ):
        raise ValueError("radial_sign must contain only -1 or +1")
    rz_vector_to_signed_into(transformed, orientation, transformed)
    return transformed


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def rz_radial_interval(lower: float, upper: float) -> tuple[float, float]:
    """Image of an ordered signed interval under the canonical radius map."""

    radius_lower = 0.0 if lower <= 0.0 <= upper else min(abs(lower), abs(upper))
    return radius_lower, max(abs(lower), abs(upper))


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def canonicalize_rz_enclosure_into(
    lower: FloatArray, upper: FloatArray, canonical_lower: FloatArray, canonical_upper: FloatArray
) -> None:
    """Map validated boxes into reusable buffers; exact aliases are allowed."""

    for row in range(lower.shape[0]):
        radial_lower, radial_upper = rz_radial_interval(lower[row, 0], upper[row, 0])
        canonical_lower[row, 0] = radial_lower
        canonical_upper[row, 0] = radial_upper
        canonical_lower[row, 1] = lower[row, 1]
        canonical_upper[row, 1] = upper[row, 1]


def canonicalize_rz_enclosure(
    lower: FloatArray, upper: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return the canonical RZ image of batched signed-coordinate boxes.

    For a radial interval that crosses zero the image under ``abs`` starts at
    zero and ends at the larger endpoint magnitude.  Axial bounds are copied
    unchanged.  This is the interval operation used for field-support checks;
    taking the absolute value of the two endpoints independently would be
    incorrect for an axis-crossing interval.
    """

    signed_lower = _finite_batch(lower, "lower")
    signed_upper = _finite_batch(upper, "upper")
    if signed_lower.shape != signed_upper.shape:
        raise ValueError("lower and upper must have the same shape")
    if bool((signed_lower > signed_upper).any()):
        raise ValueError("lower bounds must not exceed upper bounds")

    canonicalize_rz_enclosure_into(signed_lower, signed_upper, signed_lower, signed_upper)
    return signed_lower, signed_upper


@njit(cache=True, fastmath=False, nogil=True, error_model="numpy")
def canonicalize_rz_velocity_enclosure_into(
    position_lower_m: FloatArray,
    position_upper_m: FloatArray,
    velocity_lower_m_s: FloatArray,
    velocity_upper_m_s: FloatArray,
    canonical_lower_m_s: FloatArray,
    canonical_upper_m_s: FloatArray,
) -> None:
    """Image of a signed velocity box; corresponding exact aliases are safe."""

    for row in range(position_lower_m.shape[0]):
        lower = velocity_lower_m_s[row, 0]
        upper = velocity_upper_m_s[row, 0]
        if position_upper_m[row, 0] < 0.0:
            lower, upper = -upper, -lower
        elif position_lower_m[row, 0] < 0.0:
            radial_abs = np.maximum(abs(lower), abs(upper))
            lower, upper = -radial_abs, radial_abs
        canonical_lower_m_s[row, 0] = lower
        canonical_upper_m_s[row, 0] = upper
        canonical_lower_m_s[row, 1] = velocity_lower_m_s[row, 1]
        canonical_upper_m_s[row, 1] = velocity_upper_m_s[row, 1]


def fold_rz_position_vector(
    position_m: FloatArray, vector: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return the canonical meridional representation across the ``r = 0`` seam.

    A negative radial coordinate represents an axis crossing, not a wall hit.  The
    equivalent state has ``r`` and the radial vector component reflected.  At the
    axis itself an inward radial component is reflected so the following interval
    starts in the canonical ``r >= 0`` half-plane.
    """

    position = _finite_pair(position_m, "position_m")
    transformed = _finite_pair(vector, "vector")
    fold_rz_position_vector_into(position[None, :], transformed[None, :])
    return position, transformed


@njit(cache=True, fastmath=False, nogil=True)
def fold_rz_position_vector_into(position_m: FloatArray, vector: FloatArray) -> None:
    """Fold validated accepted rows in place, including inward axis departure."""

    for row in range(position_m.shape[0]):
        radius = position_m[row, 0]
        if radius < 0.0:
            position_m[row, 0] = -radius
            vector[row, 0] = -vector[row, 0]
        elif radius == 0.0:
            position_m[row, 0] = 0.0
            if vector[row, 0] < 0.0:
                vector[row, 0] = -vector[row, 0]


def _finite_pair(value: FloatArray, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _finite_batch(value: FloatArray, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise ValueError(f"{name} must have shape [N, 2]")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()
