from __future__ import annotations

import numpy as np


def locate_axis_interval(axis: np.ndarray, value: float) -> tuple[int, int, float]:
    arr = np.asarray(axis, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("Axis must be 1D with at least 2 entries")
    if value <= arr[0]:
        return 0, 1, 0.0
    if value >= arr[-1]:
        return arr.size - 2, arr.size - 1, 1.0
    if not np.isfinite(value):
        return 0, 1, float("nan")
    hi = int(np.searchsorted(arr, value))
    lo = hi - 1
    denom = float(arr[hi] - arr[lo])
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Axis entries must be finite and strictly increasing")
    alpha = (float(value) - float(arr[lo])) / denom
    return lo, hi, alpha


def sample_grid_scalar(
    arr: np.ndarray, axes: tuple[np.ndarray, ...], pos: np.ndarray
) -> float:
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
    raise ValueError("Only 2D/3D sampling is supported")


def _point_grid_inputs_2d(
    arr: np.ndarray,
    axes: tuple[np.ndarray, ...],
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("2D point sampling requires shape (n, 2)")
    if len(axes) != 2:
        raise ValueError("2D point sampling requires exactly two axes")
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("2D point sampling requires a 2D grid")
    xs, ys = (np.asarray(axis, dtype=np.float64) for axis in axes)
    if xs.size < 2 or ys.size < 2:
        raise ValueError("2D point sampling axes must have at least two entries")
    return pts, data, xs, ys


def _point_axis_intervals(
    axis: np.ndarray,
    values: np.ndarray,
    axis_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = np.searchsorted(axis, values, side="right")
    high = np.clip(high, 1, axis.size - 1).astype(np.int64)
    low = high - 1
    denominator = axis[high] - axis[low]
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError(
            f"{axis_name}-axis entries must be finite and strictly increasing"
        )
    alpha = (values - axis[low]) / denominator
    alpha = np.where(
        values <= axis[0],
        0.0,
        np.where(values >= axis[-1], 1.0, alpha),
    )
    return low, high, np.clip(alpha, 0.0, 1.0)


def sample_grid_scalar_points_2d(
    arr: np.ndarray,
    axes: tuple[np.ndarray, ...],
    positions: np.ndarray,
) -> np.ndarray:
    pts, data, xs, ys = _point_grid_inputs_2d(arr, axes, positions)
    outside = (
        (pts[:, 0] < xs[0])
        | (pts[:, 0] > xs[-1])
        | (pts[:, 1] < ys[0])
        | (pts[:, 1] > ys[-1])
    )
    lo_x, hi_x, ax = _point_axis_intervals(xs, pts[:, 0], "x")
    lo_y, hi_y, ay = _point_axis_intervals(ys, pts[:, 1], "y")
    c00 = data[lo_x, lo_y]
    c10 = data[hi_x, lo_y]
    c01 = data[lo_x, hi_y]
    c11 = data[hi_x, hi_y]
    c0 = c00 * (1.0 - ax) + c10 * ax
    c1 = c01 * (1.0 - ax) + c11 * ax
    out = c0 * (1.0 - ay) + c1 * ay
    return np.where(outside, np.nan, out)
