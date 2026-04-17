from __future__ import annotations

from typing import Tuple

import numpy as np

from .grid_sampling import locate_axis_interval, sample_grid_scalar

VALID_MASK_STATUS_CLEAN = 0
VALID_MASK_STATUS_MIXED_STENCIL = 1
VALID_MASK_STATUS_HARD_INVALID = 2


def point_within_axes(axes: Tuple[np.ndarray, ...], position: np.ndarray, *, tol: float = 0.0) -> bool:
    point = np.asarray(position, dtype=np.float64)
    if point.size < len(axes):
        return False
    for axis_index, axis in enumerate(axes):
        arr = np.asarray(axis, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError('Axis must be 1D with at least 2 entries')
        value = float(point[axis_index])
        if not np.isfinite(value):
            return False
        if value < float(arr[0]) - float(tol) or value > float(arr[-1]) + float(tol):
            return False
    return True


def choose_velocity_quantity_names(field, spatial_dim: int) -> Tuple[str, ...]:
    if field is None:
        return tuple()
    if int(spatial_dim) == 2 and getattr(field, 'coordinate_system', '') == 'axisymmetric_rz':
        candidates = (('ur', 'uz'), ('ux', 'uy'), ('vr', 'vz'))
    elif int(spatial_dim) == 2:
        candidates = (('ux', 'uy'), ('vx', 'vy'))
    else:
        candidates = (('ux', 'uy', 'uz'), ('vx', 'vy', 'vz'))
    for names in candidates:
        if all(name in field.quantities for name in names):
            return tuple(names)
    return tuple()


def choose_electric_field_quantity_names(field, spatial_dim: int) -> Tuple[str, ...]:
    if field is None:
        return tuple()
    if int(spatial_dim) == 2 and getattr(field, 'coordinate_system', '') == 'axisymmetric_rz':
        candidates = (
            ('E_r', 'E_z'),
            ('Er', 'Ez'),
            ('electric_r', 'electric_z'),
            ('electric_field_r', 'electric_field_z'),
            ('E_x', 'E_y'),
            ('Ex', 'Ey'),
        )
    elif int(spatial_dim) == 2:
        candidates = (
            ('E_x', 'E_y'),
            ('Ex', 'Ey'),
            ('electric_x', 'electric_y'),
            ('electric_field_x', 'electric_field_y'),
        )
    else:
        candidates = (
            ('E_x', 'E_y', 'E_z'),
            ('Ex', 'Ey', 'Ez'),
            ('electric_x', 'electric_y', 'electric_z'),
            ('electric_field_x', 'electric_field_y', 'electric_field_z'),
        )
    for names in candidates:
        if all(name in field.quantities for name in names):
            return tuple(names)
    return tuple()


def as_time_grid(data: np.ndarray, spatial_dim: int) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == int(spatial_dim):
        return arr[None, ...]
    return arr


def sample_time_grid_scalar(
    arr: np.ndarray,
    axes: Tuple[np.ndarray, ...],
    times: np.ndarray,
    t_eval: float,
    position: np.ndarray,
    *,
    mode: str = 'linear',
) -> float:
    data = np.asarray(arr, dtype=np.float64)
    point = np.asarray(position, dtype=np.float64)
    time_grid = np.asarray(times, dtype=np.float64)
    if data.ndim == len(axes):
        return float(sample_grid_scalar(data, axes, point))
    if data.shape[0] <= 1 or time_grid.size <= 1:
        return float(sample_grid_scalar(data[0], axes, point))
    if t_eval <= float(time_grid[0]):
        return float(sample_grid_scalar(data[0], axes, point))
    if t_eval >= float(time_grid[-1]):
        return float(sample_grid_scalar(data[-1], axes, point))
    hi = int(np.searchsorted(time_grid, float(t_eval)))
    lo = hi - 1
    if str(mode).strip().lower() == 'nearest':
        idx = hi if abs(float(time_grid[hi]) - float(t_eval)) < abs(float(t_eval) - float(time_grid[lo])) else lo
        return float(sample_grid_scalar(data[idx], axes, point))
    t_lo = float(time_grid[lo])
    t_hi = float(time_grid[hi])
    alpha = 0.0 if abs(t_hi - t_lo) <= 1e-30 else (float(t_eval) - t_lo) / (t_hi - t_lo)
    v_lo = float(sample_grid_scalar(data[lo], axes, point))
    v_hi = float(sample_grid_scalar(data[hi], axes, point))
    return float(v_lo * (1.0 - alpha) + v_hi * alpha)


def sample_quantity_series(series, axes: Tuple[np.ndarray, ...], position: np.ndarray, t_eval: float, *, mode: str = 'linear') -> float:
    return float(
        sample_time_grid_scalar(
            np.asarray(series.data, dtype=np.float64),
            axes,
            np.asarray(series.times, dtype=np.float64),
            float(t_eval),
            np.asarray(position, dtype=np.float64),
            mode=mode,
        )
    )


def sample_valid_mask(mask: np.ndarray, axes: Tuple[np.ndarray, ...], position: np.ndarray, *, threshold: float = 0.5) -> bool:
    point = np.asarray(position, dtype=np.float64)
    if not point_within_axes(axes, point):
        return False
    return bool(sample_grid_scalar(np.asarray(mask, dtype=np.float64), axes, point) >= float(threshold))


def _valid_mask_stencil_has_invalid(mask: np.ndarray, axes: Tuple[np.ndarray, ...], position: np.ndarray) -> bool:
    grid = np.asarray(mask, dtype=bool)
    point = np.asarray(position, dtype=np.float64)
    if not point_within_axes(axes, point):
        return True
    dim = len(axes)
    if dim == 2:
        ix0, ix1, _ = locate_axis_interval(axes[0], float(point[0]))
        iy0, iy1, _ = locate_axis_interval(axes[1], float(point[1]))
        return bool(
            (not grid[ix0, iy0])
            or (not grid[ix1, iy0])
            or (not grid[ix0, iy1])
            or (not grid[ix1, iy1])
        )
    if dim == 3:
        ix0, ix1, _ = locate_axis_interval(axes[0], float(point[0]))
        iy0, iy1, _ = locate_axis_interval(axes[1], float(point[1]))
        iz0, iz1, _ = locate_axis_interval(axes[2], float(point[2]))
        return bool(
            (not grid[ix0, iy0, iz0])
            or (not grid[ix1, iy0, iz0])
            or (not grid[ix0, iy1, iz0])
            or (not grid[ix1, iy1, iz0])
            or (not grid[ix0, iy0, iz1])
            or (not grid[ix1, iy0, iz1])
            or (not grid[ix0, iy1, iz1])
            or (not grid[ix1, iy1, iz1])
        )
    raise ValueError('Only 2D/3D valid-mask stencil checks are supported')


def sample_valid_mask_status(mask: np.ndarray, axes: Tuple[np.ndarray, ...], position: np.ndarray, *, threshold: float = 0.5) -> int:
    point_valid = sample_valid_mask(mask, axes, position, threshold=threshold)
    stencil_invalid = _valid_mask_stencil_has_invalid(mask, axes, position)
    if not bool(point_valid):
        return int(VALID_MASK_STATUS_HARD_INVALID)
    if bool(stencil_invalid):
        return int(VALID_MASK_STATUS_MIXED_STENCIL)
    return int(VALID_MASK_STATUS_CLEAN)


def valid_mask_stencil_has_invalid(mask: np.ndarray, axes: Tuple[np.ndarray, ...], position: np.ndarray) -> bool:
    return bool(_valid_mask_stencil_has_invalid(mask, axes, position))


def valid_mask_status_is_violation(status: int) -> bool:
    return int(status) != int(VALID_MASK_STATUS_CLEAN)


def valid_mask_status_requires_stop(status: int) -> bool:
    return int(status) == int(VALID_MASK_STATUS_HARD_INVALID)
