from __future__ import annotations

import numpy as np
from numba import njit

from ..core.field_sampling import (
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
    a = 0.0 if abs(denom) <= 1e-30 else (value - axis[lo]) / denom
    return lo, hi, a


@njit(cache=True)
def should_capture_midpoint(has_mid, elapsed, dt_sub, target_dt):
    if has_mid:
        return False
    elapsed_next = elapsed + dt_sub
    return target_dt <= elapsed_next + 1.0e-15


@njit(cache=True)
def midpoint_local_dt(elapsed, dt_sub, target_dt):
    dt_local = target_dt - elapsed
    if dt_local < 0.0:
        return 0.0
    if dt_local > dt_sub:
        return dt_sub
    return dt_local


@njit(cache=True)
def mask_bilinear_status(mask2d, xs, ys, x, y):
    if x < xs[0] or x > xs[xs.size - 1] or y < ys[0] or y > ys[ys.size - 1]:
        return VALID_MASK_STATUS_HARD_INVALID
    ix0, ix1, ax = locate_axis(xs, x)
    iy0, iy1, ay = locate_axis(ys, y)
    c00 = mask2d[ix0, iy0]
    c10 = mask2d[ix1, iy0]
    c01 = mask2d[ix0, iy1]
    c11 = mask2d[ix1, iy1]
    stencil_invalid = (not c00) or (not c10) or (not c01) or (not c11)
    v00 = 1.0 if c00 else 0.0
    v10 = 1.0 if c10 else 0.0
    v01 = 1.0 if c01 else 0.0
    v11 = 1.0 if c11 else 0.0
    c0 = v00 * (1.0 - ax) + v10 * ax
    c1 = v01 * (1.0 - ax) + v11 * ax
    point_valid = (c0 * (1.0 - ay) + c1 * ay) >= 0.5
    if not point_valid:
        return VALID_MASK_STATUS_HARD_INVALID
    if stencil_invalid:
        return VALID_MASK_STATUS_MIXED_STENCIL
    return VALID_MASK_STATUS_CLEAN


@njit(cache=True)
def mask_bilinear_point_valid(mask2d, xs, ys, x, y):
    if x < xs[0] or x > xs[xs.size - 1] or y < ys[0] or y > ys[ys.size - 1]:
        return False
    ix0, ix1, ax = locate_axis(xs, x)
    iy0, iy1, ay = locate_axis(ys, y)
    v00 = 1.0 if mask2d[ix0, iy0] else 0.0
    v10 = 1.0 if mask2d[ix1, iy0] else 0.0
    v01 = 1.0 if mask2d[ix0, iy1] else 0.0
    v11 = 1.0 if mask2d[ix1, iy1] else 0.0
    c0 = v00 * (1.0 - ax) + v10 * ax
    c1 = v01 * (1.0 - ax) + v11 * ax
    return (c0 * (1.0 - ay) + c1 * ay) >= 0.5


@njit(cache=True)
def mask_bilinear_has_invalid(mask2d, xs, ys, x, y):
    return mask_bilinear_status(mask2d, xs, ys, x, y) != VALID_MASK_STATUS_CLEAN


@njit(cache=True)
def mask_trilinear_status(mask3d, xs, ys, zs, x, y, z):
    if (
        x < xs[0]
        or x > xs[xs.size - 1]
        or y < ys[0]
        or y > ys[ys.size - 1]
        or z < zs[0]
        or z > zs[zs.size - 1]
    ):
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
    stencil_invalid = (
        (not c000)
        or (not c100)
        or (not c010)
        or (not c110)
        or (not c001)
        or (not c101)
        or (not c011)
        or (not c111)
    )
    v000 = 1.0 if c000 else 0.0
    v100 = 1.0 if c100 else 0.0
    v010 = 1.0 if c010 else 0.0
    v110 = 1.0 if c110 else 0.0
    v001 = 1.0 if c001 else 0.0
    v101 = 1.0 if c101 else 0.0
    v011 = 1.0 if c011 else 0.0
    v111 = 1.0 if c111 else 0.0
    c00 = v000 * (1.0 - ax) + v100 * ax
    c10 = v010 * (1.0 - ax) + v110 * ax
    c01 = v001 * (1.0 - ax) + v101 * ax
    c11 = v011 * (1.0 - ax) + v111 * ax
    c0 = c00 * (1.0 - ay) + c10 * ay
    c1 = c01 * (1.0 - ay) + c11 * ay
    point_valid = (c0 * (1.0 - az) + c1 * az) >= 0.5
    if not point_valid:
        return VALID_MASK_STATUS_HARD_INVALID
    if stencil_invalid:
        return VALID_MASK_STATUS_MIXED_STENCIL
    return VALID_MASK_STATUS_CLEAN


@njit(cache=True)
def mask_trilinear_point_valid(mask3d, xs, ys, zs, x, y, z):
    if (
        x < xs[0]
        or x > xs[xs.size - 1]
        or y < ys[0]
        or y > ys[ys.size - 1]
        or z < zs[0]
        or z > zs[zs.size - 1]
    ):
        return False
    ix0, ix1, ax = locate_axis(xs, x)
    iy0, iy1, ay = locate_axis(ys, y)
    iz0, iz1, az = locate_axis(zs, z)
    v000 = 1.0 if mask3d[ix0, iy0, iz0] else 0.0
    v100 = 1.0 if mask3d[ix1, iy0, iz0] else 0.0
    v010 = 1.0 if mask3d[ix0, iy1, iz0] else 0.0
    v110 = 1.0 if mask3d[ix1, iy1, iz0] else 0.0
    v001 = 1.0 if mask3d[ix0, iy0, iz1] else 0.0
    v101 = 1.0 if mask3d[ix1, iy0, iz1] else 0.0
    v011 = 1.0 if mask3d[ix0, iy1, iz1] else 0.0
    v111 = 1.0 if mask3d[ix1, iy1, iz1] else 0.0
    c00 = v000 * (1.0 - ax) + v100 * ax
    c10 = v010 * (1.0 - ax) + v110 * ax
    c01 = v001 * (1.0 - ax) + v101 * ax
    c11 = v011 * (1.0 - ax) + v111 * ax
    c0 = c00 * (1.0 - ay) + c10 * ay
    c1 = c01 * (1.0 - ay) + c11 * ay
    return (c0 * (1.0 - az) + c1 * az) >= 0.5


@njit(cache=True)
def mask_trilinear_has_invalid(mask3d, xs, ys, zs, x, y, z):
    return mask_trilinear_status(mask3d, xs, ys, zs, x, y, z) != VALID_MASK_STATUS_CLEAN
