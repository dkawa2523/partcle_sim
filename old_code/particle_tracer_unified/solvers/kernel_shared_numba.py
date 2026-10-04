from __future__ import annotations

import numpy as np
from numba import njit

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)


@njit(cache=True)
def locate_axis(axis, value):
    if value <= axis[0]:
        return 0, 1, 0.0
    n = axis.size
    if value >= axis[n - 1]:
        return n - 2, n - 1, 1.0
    j = np.searchsorted(axis, value)
    lo = j - 1
    hi = j
    denom = axis[hi] - axis[lo]
    a = 0.0 if denom == 0.0 else (value - axis[lo]) / denom
    return lo, hi, a


@njit(cache=True)
def time_roundoff_tolerance(reference_time, interval):
    magnitude = max(abs(reference_time), abs(interval))
    return 64.0 * np.finfo(np.float64).eps * magnitude


@njit(cache=True, inline="always")
def _outside_bilinear_domain(xs, ys, x, y):
    return x < xs[0] or x > xs[xs.size - 1] or y < ys[0] or y > ys[ys.size - 1]


@njit(cache=True, inline="always")
def _mask_weight(corner_valid):
    return 1.0 if corner_valid else 0.0


@njit(cache=True, inline="always")
def _bilinear_mask_value(c00, c10, c01, c11, ax, ay):
    v00 = _mask_weight(c00)
    v10 = _mask_weight(c10)
    v01 = _mask_weight(c01)
    v11 = _mask_weight(c11)
    c0 = v00 * (1.0 - ax) + v10 * ax
    c1 = v01 * (1.0 - ax) + v11 * ax
    return c0 * (1.0 - ay) + c1 * ay


@njit(cache=True, inline="always")
def _bilinear_corner_status(c00, c10, c01, c11, point_value):
    if point_value < 0.5:
        return VALID_MASK_STATUS_HARD_INVALID
    if not (c00 and c10 and c01 and c11):
        return VALID_MASK_STATUS_MIXED_STENCIL
    return VALID_MASK_STATUS_CLEAN


@njit(cache=True)
def mask_bilinear_status(mask2d, xs, ys, x, y):
    if _outside_bilinear_domain(xs, ys, x, y):
        return VALID_MASK_STATUS_HARD_INVALID
    ix0, ix1, ax = locate_axis(xs, x)
    iy0, iy1, ay = locate_axis(ys, y)
    c00 = mask2d[ix0, iy0]
    c10 = mask2d[ix1, iy0]
    c01 = mask2d[ix0, iy1]
    c11 = mask2d[ix1, iy1]
    point_value = _bilinear_mask_value(c00, c10, c01, c11, ax, ay)
    return _bilinear_corner_status(c00, c10, c01, c11, point_value)


@njit(cache=True, inline="always")
def _outside_trilinear_domain(xs, ys, zs, x, y, z):
    return (
        x < xs[0]
        or x > xs[xs.size - 1]
        or y < ys[0]
        or y > ys[ys.size - 1]
        or z < zs[0]
        or z > zs[zs.size - 1]
    )


@njit(cache=True, inline="always")
def _trilinear_mask_value(
    c000,
    c100,
    c010,
    c110,
    c001,
    c101,
    c011,
    c111,
    ax,
    ay,
    az,
):
    v000 = _mask_weight(c000)
    v100 = _mask_weight(c100)
    v010 = _mask_weight(c010)
    v110 = _mask_weight(c110)
    v001 = _mask_weight(c001)
    v101 = _mask_weight(c101)
    v011 = _mask_weight(c011)
    v111 = _mask_weight(c111)
    c00 = v000 * (1.0 - ax) + v100 * ax
    c10 = v010 * (1.0 - ax) + v110 * ax
    c01 = v001 * (1.0 - ax) + v101 * ax
    c11 = v011 * (1.0 - ax) + v111 * ax
    c0 = c00 * (1.0 - ay) + c10 * ay
    c1 = c01 * (1.0 - ay) + c11 * ay
    return c0 * (1.0 - az) + c1 * az


@njit(cache=True, inline="always")
def _trilinear_corner_status(
    c000,
    c100,
    c010,
    c110,
    c001,
    c101,
    c011,
    c111,
    point_value,
):
    if point_value < 0.5:
        return VALID_MASK_STATUS_HARD_INVALID
    if not (c000 and c100 and c010 and c110 and c001 and c101 and c011 and c111):
        return VALID_MASK_STATUS_MIXED_STENCIL
    return VALID_MASK_STATUS_CLEAN


@njit(cache=True)
def mask_trilinear_status(mask3d, xs, ys, zs, x, y, z):
    if _outside_trilinear_domain(xs, ys, zs, x, y, z):
        return VALID_MASK_STATUS_HARD_INVALID
    ix0, ix1, ax = locate_axis(xs, x)
    iy0, iy1, ay = locate_axis(ys, y)
    iz0, iz1, az = locate_axis(zs, z)
    c000 = mask3d[ix0, iy0, iz0]
    c100 = mask3d[ix1, iy0, iz0]
    c010 = mask3d[ix0, iy1, iz0]
    c110 = mask3d[ix1, iy1, iz0]
    c001 = mask3d[ix0, iy0, iz1]
    c101 = mask3d[ix1, iy0, iz1]
    c011 = mask3d[ix0, iy1, iz1]
    c111 = mask3d[ix1, iy1, iz1]
    point_value = _trilinear_mask_value(
        c000,
        c100,
        c010,
        c110,
        c001,
        c101,
        c011,
        c111,
        ax,
        ay,
        az,
    )
    return _trilinear_corner_status(
        c000,
        c100,
        c010,
        c110,
        c001,
        c101,
        c011,
        c111,
        point_value,
    )
