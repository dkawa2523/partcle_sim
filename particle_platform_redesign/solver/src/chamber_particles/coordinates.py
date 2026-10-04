"""Coordinate transforms that are independent of fields, geometry, and physics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


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
    radial_sign = np.where(position[:, 0] < 0.0, -1.0, 1.0)
    position[:, 0] = np.abs(position[:, 0])
    transformed[:, 0] *= radial_sign
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
    transformed[:, 0] *= orientation
    return transformed


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

    radial_lower = signed_lower[:, 0]
    radial_upper = signed_upper[:, 0]
    crosses_axis = (radial_lower <= 0.0) & (radial_upper >= 0.0)
    endpoint_min = np.minimum(np.abs(radial_lower), np.abs(radial_upper))
    endpoint_max = np.maximum(np.abs(radial_lower), np.abs(radial_upper))
    signed_lower[:, 0] = np.where(crosses_axis, 0.0, endpoint_min)
    signed_upper[:, 0] = endpoint_max
    return signed_lower, signed_upper


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
    radius = float(position[0])
    if radius < 0.0:
        position[0] = -radius
        transformed[0] = -transformed[0]
    elif radius == 0.0:
        position[0] = 0.0
        if transformed[0] < 0.0:
            transformed[0] = -transformed[0]
    return position, transformed


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
