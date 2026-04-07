from __future__ import annotations

from typing import Tuple

import numpy as np


def locate_axis_interval(axis: np.ndarray, value: float) -> Tuple[int, int, float]:
    arr = np.asarray(axis, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError('Axis must be 1D with at least 2 entries')
    if value <= arr[0]:
        return 0, 1, 0.0
    if value >= arr[-1]:
        return arr.size - 2, arr.size - 1, 1.0
    hi = int(np.searchsorted(arr, value))
    lo = hi - 1
    denom = float(arr[hi] - arr[lo])
    alpha = 0.0 if abs(denom) <= 1e-30 else (float(value) - float(arr[lo])) / denom
    return lo, hi, alpha


def sample_grid_scalar(arr: np.ndarray, axes: Tuple[np.ndarray, ...], pos: np.ndarray) -> float:
    dim = len(axes)
    point = np.asarray(pos, dtype=np.float64)
    data = np.asarray(arr, dtype=np.float64)
    if dim == 2:
        xs, ys = axes
        x, y = float(point[0]), float(point[1])
        ix0, ix1, ax = locate_axis_interval(xs, x)
        iy0, iy1, ay = locate_axis_interval(ys, y)
        c00 = data[ix0, iy0]
        c10 = data[ix1, iy0]
        c01 = data[ix0, iy1]
        c11 = data[ix1, iy1]
        c0 = c00 * (1.0 - ax) + c10 * ax
        c1 = c01 * (1.0 - ax) + c11 * ax
        return float(c0 * (1.0 - ay) + c1 * ay)
    if dim == 3:
        xs, ys, zs = axes
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        ix0, ix1, ax = locate_axis_interval(xs, x)
        iy0, iy1, ay = locate_axis_interval(ys, y)
        iz0, iz1, az = locate_axis_interval(zs, z)
        c000 = data[ix0, iy0, iz0]
        c100 = data[ix1, iy0, iz0]
        c010 = data[ix0, iy1, iz0]
        c110 = data[ix1, iy1, iz0]
        c001 = data[ix0, iy0, iz1]
        c101 = data[ix1, iy0, iz1]
        c011 = data[ix0, iy1, iz1]
        c111 = data[ix1, iy1, iz1]
        c00 = c000 * (1.0 - ax) + c100 * ax
        c10 = c010 * (1.0 - ax) + c110 * ax
        c01 = c001 * (1.0 - ax) + c101 * ax
        c11 = c011 * (1.0 - ax) + c111 * ax
        c0 = c00 * (1.0 - ay) + c10 * ay
        c1 = c01 * (1.0 - ay) + c11 * ay
        return float(c0 * (1.0 - az) + c1 * az)
    raise ValueError('Only 2D/3D sampling is supported')
