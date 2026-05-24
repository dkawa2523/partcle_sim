from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Mapping, Optional, Tuple

import numpy as np

from ..core.datamodel import TriangleMeshField2D
from ..core.field_backend import triangle_derived_quantity_names, triangle_mesh_gradient_source_report
from ..core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
    as_time_grid,
    choose_electric_field_quantity_names,
    choose_velocity_quantity_names,
    sample_time_grid_scalar,
    sample_valid_mask_status,
)
from ..core.triangle_mesh_sampling_2d import (
    locate_triangle_containing_point,
    sample_triangle_mesh_series,
    sample_triangle_mesh_status,
)
from .force_runtime import (
    ForceBatchSamples,
    ForceBatchState,
    ForceBatchStatic,
    build_force_pipeline,
    evaluate_force_pipeline,
)
from .forces import ForceRuntimeParameters


@dataclass(frozen=True, slots=True)
class RegularRectilinearCompiledBackend:
    axes: Tuple[np.ndarray, ...]
    times: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    gas_density: np.ndarray
    gas_mu: np.ndarray
    gas_temperature: np.ndarray
    valid_mask: np.ndarray
    core_valid_mask: np.ndarray
    uz: Optional[np.ndarray] = None
    electric_x: Optional[np.ndarray] = None
    electric_y: Optional[np.ndarray] = None
    electric_z: Optional[np.ndarray] = None
    backend_kind: str = 'regular_rectilinear'
    acceleration_source: str = 'none'
    acceleration_quantity_names: Tuple[str, ...] = ()
    electric_field_names: Tuple[str, ...] = ()
    electric_q_over_m_Ckg: float = 0.0
    gas_density_source: str = 'scalar_fallback'
    gas_mu_source: str = 'scalar_fallback'
    gas_temperature_source: str = 'scalar_fallback'
    grad_T_x: Optional[np.ndarray] = None
    grad_T_y: Optional[np.ndarray] = None
    grad_T_z: Optional[np.ndarray] = None
    grad_E2_x: Optional[np.ndarray] = None
    grad_E2_y: Optional[np.ndarray] = None
    grad_E2_z: Optional[np.ndarray] = None
    vorticity_x: Optional[np.ndarray] = None
    vorticity_y: Optional[np.ndarray] = None
    vorticity_z: Optional[np.ndarray] = None
    fluid_accel_x: Optional[np.ndarray] = None
    fluid_accel_y: Optional[np.ndarray] = None
    fluid_accel_z: Optional[np.ndarray] = None
    du_dt_x: Optional[np.ndarray] = None
    du_dt_y: Optional[np.ndarray] = None
    du_dt_z: Optional[np.ndarray] = None
    grad_ux_x: Optional[np.ndarray] = None
    grad_ux_y: Optional[np.ndarray] = None
    grad_ux_z: Optional[np.ndarray] = None
    grad_uy_x: Optional[np.ndarray] = None
    grad_uy_y: Optional[np.ndarray] = None
    grad_uy_z: Optional[np.ndarray] = None
    grad_uz_x: Optional[np.ndarray] = None
    grad_uz_y: Optional[np.ndarray] = None
    grad_uz_z: Optional[np.ndarray] = None


@dataclass(frozen=True, slots=True)
class TriangleMesh2DCompiledBackend:
    field: TriangleMeshField2D
    velocity_names: Tuple[str, ...]
    times: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    gas_density: np.ndarray
    gas_mu: np.ndarray
    gas_temperature: np.ndarray
    mesh_vertices: np.ndarray
    mesh_triangles: np.ndarray
    accel_origin: np.ndarray
    accel_cell_size: np.ndarray
    accel_shape: Tuple[int, int]
    accel_cell_offsets: np.ndarray
    accel_triangle_indices: np.ndarray
    support_tolerance_m: float
    backend_kind: str = 'triangle_mesh_2d'
    acceleration_source: str = 'none'
    acceleration_quantity_names: Tuple[str, ...] = ()
    electric_field_names: Tuple[str, ...] = ()
    electric_q_over_m_Ckg: float = 0.0
    gas_density_source: str = 'scalar_fallback'
    gas_mu_source: str = 'scalar_fallback'
    gas_temperature_source: str = 'scalar_fallback'
    triangle_gradient_sources: Mapping[str, str] = dataclass_field(default_factory=dict)


CompiledRuntimeBackend = RegularRectilinearCompiledBackend | TriangleMesh2DCompiledBackend
CompiledRuntimeBackendLike = CompiledRuntimeBackend | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FieldSample:
    """Stage-local sampled fields from the compiled backend."""

    position: np.ndarray
    time_s: float
    spatial_dim: int
    flow_velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    gas_density_kgm3: Optional[float] = None
    gas_mu_pas: Optional[float] = None
    gas_temperature_K: Optional[float] = None
    valid_mask_status: Optional[int] = None


_FIELD_BACKEND_MODE_ALIASES = {
    '': 'auto',
    'auto': 'auto',
    'default': 'auto',
    'grid': 'regular_grid',
    'regular': 'regular_grid',
    'regular_grid': 'regular_grid',
    'regular_rectilinear': 'regular_grid',
    'rectilinear': 'regular_grid',
    'triangle': 'triangle_mesh',
    'tri': 'triangle_mesh',
    'triangle_mesh': 'triangle_mesh',
    'triangle_mesh_2d': 'triangle_mesh',
}


def _runtime_field_backend_mode(runtime) -> str:
    config = getattr(runtime, 'config_payload', {})
    if not isinstance(config, Mapping):
        return 'auto'
    solver_cfg = config.get('solver', {})
    solver = solver_cfg if isinstance(solver_cfg, Mapping) else {}
    providers_cfg = config.get('providers', {})
    providers = providers_cfg if isinstance(providers_cfg, Mapping) else {}
    field_cfg = providers.get('field', {})
    field = field_cfg if isinstance(field_cfg, Mapping) else {}
    raw = (
        solver.get('field_backend_mode')
        if 'field_backend_mode' in solver
        else solver.get('field_backend', field.get('backend_mode', field.get('mode', 'auto')))
    )
    text = str(raw).strip().lower()
    if text not in _FIELD_BACKEND_MODE_ALIASES:
        allowed = ', '.join(('auto', 'regular_grid', 'triangle_mesh'))
        raise ValueError(f'solver.field_backend_mode must be one of {allowed}')
    return _FIELD_BACKEND_MODE_ALIASES[text]


def _enforce_field_backend_mode(mode: str, *, is_triangle_mesh: bool) -> None:
    requested = str(mode).strip().lower()
    if requested in {'', 'auto'}:
        return
    if requested == 'triangle_mesh' and not bool(is_triangle_mesh):
        raise ValueError('solver.field_backend_mode=triangle_mesh requires providers.field.kind=precomputed_triangle_mesh_npz')
    if requested == 'regular_grid' and bool(is_triangle_mesh):
        raise ValueError('solver.field_backend_mode=regular_grid requires a regular rectilinear field provider')


def _backend_time_grid(data: np.ndarray, spatial_dim: int, times: np.ndarray) -> np.ndarray:
    grid = as_time_grid(data, int(spatial_dim))
    time_count = int(max(1, np.asarray(times, dtype=np.float64).size))
    if grid.shape[0] == 1 and time_count > 1:
        return np.repeat(grid, time_count, axis=0)
    return grid


def _zero_like_grid(reference: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(reference, dtype=np.float64), dtype=np.float64)


def _gradient_time_grid(data: np.ndarray, axes: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, ...]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != len(axes) + 1:
        raise ValueError('gradient source must be a time grid with shape (nt, ...spatial axes)')
    spatial_axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    if any(axis.size < 2 for axis in spatial_axes):
        return tuple(_zero_like_grid(arr) for _ in spatial_axes)
    edge_order = 2 if all(axis.size >= 3 for axis in spatial_axes) else 1
    grads = np.gradient(arr, *spatial_axes, axis=tuple(range(1, arr.ndim)), edge_order=edge_order)
    return tuple(np.asarray(grad, dtype=np.float64) for grad in grads)


def _time_derivative_time_grid(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    time_grid = np.asarray(times, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[0] <= 1 or time_grid.size <= 1:
        return _zero_like_grid(arr)
    edge_order = 2 if arr.shape[0] >= 3 else 1
    return np.asarray(np.gradient(arr, time_grid, axis=0, edge_order=edge_order), dtype=np.float64)


def _vertex_time_grid(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    time_count = int(max(1, np.asarray(times, dtype=np.float64).size))
    if arr.ndim == 1:
        arr = arr.reshape(1, arr.shape[0])
    if arr.shape[0] == 1 and time_count > 1:
        return np.repeat(arr, time_count, axis=0)
    return arr


def _curl_from_velocity_grids(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: Optional[np.ndarray],
    axes: Tuple[np.ndarray, ...],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if len(axes) == 2:
        dux_dx, dux_dy = _gradient_time_grid(ux, axes)
        duy_dx, _duy_dy = _gradient_time_grid(uy, axes)
        return None, None, np.asarray(duy_dx - dux_dy, dtype=np.float64)
    if uz is None:
        return None, None, None
    dux_dx, dux_dy, dux_dz = _gradient_time_grid(ux, axes)
    duy_dx, duy_dy, duy_dz = _gradient_time_grid(uy, axes)
    duz_dx, duz_dy, duz_dz = _gradient_time_grid(uz, axes)
    _ = (dux_dx, duy_dy, duz_dz)
    return (
        np.asarray(duz_dy - duy_dz, dtype=np.float64),
        np.asarray(dux_dz - duz_dx, dtype=np.float64),
        np.asarray(duy_dx - dux_dy, dtype=np.float64),
    )


def _common_quantity_times(field, quantity_names: Tuple[str, ...]) -> np.ndarray:
    times = np.asarray([0.0], dtype=np.float64)
    first_name = ''
    for name in quantity_names:
        series = field.quantities.get(str(name))
        if series is None:
            continue
        current = np.asarray(series.times, dtype=np.float64)
        if current.size == 0:
            current = np.asarray([0.0], dtype=np.float64)
        if not first_name:
            first_name = str(name)
            times = current
            continue
        if current.shape != times.shape or not np.allclose(current, times, rtol=0.0, atol=0.0):
            raise ValueError(
                'Field quantities used by the solver must share one time axis; '
                f'{first_name} and {name} differ'
            )
    return times


def _merge_optional_quantity_times(field, base_times: np.ndarray, quantity_names: Tuple[str, ...]) -> np.ndarray:
    times = np.asarray(base_times, dtype=np.float64)
    if times.size == 0:
        times = np.asarray([0.0], dtype=np.float64)
    first_transient_name = ''
    for name in quantity_names:
        series = field.quantities.get(str(name))
        if series is None:
            continue
        current = np.asarray(series.times, dtype=np.float64)
        if current.size == 0:
            current = np.asarray([0.0], dtype=np.float64)
        if current.size <= 1:
            continue
        if times.size <= 1:
            times = current
            first_transient_name = str(name)
            continue
        if current.shape != times.shape or not np.allclose(current, times, rtol=0.0, atol=0.0):
            reference = first_transient_name or 'primary solver quantities'
            raise ValueError(
                'Field quantities used by the solver must share one transient time axis; '
                f'{reference} and {name} differ'
            )
    return times


def _gas_property_quantity_names(field) -> Mapping[str, str]:
    selected: dict[str, str] = {}
    for candidates, target in (
        (('rho_g', 'gas_density', 'density_kgm3', 'rho'), 'gas_density'),
        (('mu', 'dynamic_viscosity', 'dynamic_viscosity_Pas'), 'gas_mu'),
        (('T', 'temperature', 'temperature_K', 'gas_temperature'), 'gas_temperature'),
    ):
        for name in candidates:
            if name in field.quantities:
                selected[target] = str(name)
                break
    return selected


def _axis_intervals(axis: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(axis, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    hi = np.searchsorted(arr, vals, side='right')
    hi = np.clip(hi, 1, arr.size - 1).astype(np.int64)
    lo = hi - 1
    denom = arr[hi] - arr[lo]
    alpha = np.divide(vals - arr[lo], denom, out=np.zeros_like(vals, dtype=np.float64), where=np.abs(denom) > 1.0e-30)
    alpha = np.where(vals <= arr[0], 0.0, np.where(vals >= arr[-1], 1.0, alpha))
    return lo, hi, np.clip(alpha, 0.0, 1.0)


def _sample_regular_grid_points_2d(grid: np.ndarray, axes: Tuple[np.ndarray, ...], positions: np.ndarray) -> np.ndarray:
    pts = np.asarray(positions, dtype=np.float64)
    data = np.asarray(grid, dtype=np.float64)
    ix0, ix1, ax = _axis_intervals(axes[0], pts[:, 0])
    iy0, iy1, ay = _axis_intervals(axes[1], pts[:, 1])
    c00 = data[ix0, iy0]
    c10 = data[ix1, iy0]
    c01 = data[ix0, iy1]
    c11 = data[ix1, iy1]
    c0 = c00 * (1.0 - ax) + c10 * ax
    c1 = c01 * (1.0 - ax) + c11 * ax
    return c0 * (1.0 - ay) + c1 * ay


def _sample_regular_time_grid_points_2d(
    data: np.ndarray,
    axes: Tuple[np.ndarray, ...],
    times: np.ndarray,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    time_grid = np.asarray(times, dtype=np.float64)
    if arr.ndim == 2:
        return _sample_regular_grid_points_2d(arr, axes, positions)
    if arr.shape[0] <= 1 or time_grid.size <= 1:
        return _sample_regular_grid_points_2d(arr[0], axes, positions)
    if t_eval <= float(time_grid[0]):
        return _sample_regular_grid_points_2d(arr[0], axes, positions)
    if t_eval >= float(time_grid[-1]):
        return _sample_regular_grid_points_2d(arr[-1], axes, positions)
    hi = int(np.searchsorted(time_grid, float(t_eval)))
    lo = hi - 1
    t_lo = float(time_grid[lo])
    t_hi = float(time_grid[hi])
    alpha = 0.0 if abs(t_hi - t_lo) <= 1.0e-30 else (float(t_eval) - t_lo) / (t_hi - t_lo)
    v_lo = _sample_regular_grid_points_2d(arr[lo], axes, positions)
    v_hi = _sample_regular_grid_points_2d(arr[hi], axes, positions)
    return v_lo * (1.0 - alpha) + v_hi * alpha


def _regular_points_inside_axes_2d(axes: Tuple[np.ndarray, ...], positions: np.ndarray) -> np.ndarray:
    pts = np.asarray(positions, dtype=np.float64)
    return (
        np.all(np.isfinite(pts[:, :2]), axis=1)
        & (pts[:, 0] >= float(axes[0][0]))
        & (pts[:, 0] <= float(axes[0][-1]))
        & (pts[:, 1] >= float(axes[1][0]))
        & (pts[:, 1] <= float(axes[1][-1]))
    )


def _triangle_mesh_location(
    backend: TriangleMesh2DCompiledBackend,
    position: np.ndarray,
) -> Tuple[int, np.ndarray]:
    return locate_triangle_containing_point(
        vertices=backend.mesh_vertices,
        triangles=backend.mesh_triangles,
        accel_origin=backend.accel_origin,
        accel_cell_size=backend.accel_cell_size,
        accel_shape=backend.accel_shape,
        accel_cell_offsets=backend.accel_cell_offsets,
        accel_triangle_indices=backend.accel_triangle_indices,
        position=np.asarray(position, dtype=np.float64),
        eps=float(backend.support_tolerance_m),
    )


def _triangle_series_values_at_time(series, field: TriangleMeshField2D, tri_idx: int, t_eval: float) -> np.ndarray:
    tri = np.asarray(field.mesh_triangles, dtype=np.int32)[int(tri_idx)]
    data = np.asarray(series.data, dtype=np.float64)
    times = np.asarray(series.times, dtype=np.float64)
    if data.ndim == 1:
        return np.asarray(data[tri], dtype=np.float64)
    if data.shape[0] <= 1 or times.size <= 1:
        return np.asarray(data[0, tri], dtype=np.float64)
    if float(t_eval) <= float(times[0]):
        return np.asarray(data[0, tri], dtype=np.float64)
    if float(t_eval) >= float(times[-1]):
        return np.asarray(data[-1, tri], dtype=np.float64)
    hi = int(np.searchsorted(times, float(t_eval)))
    lo = hi - 1
    t_lo = float(times[lo])
    t_hi = float(times[hi])
    alpha = 0.0 if abs(t_hi - t_lo) <= 1.0e-30 else (float(t_eval) - t_lo) / (t_hi - t_lo)
    return np.asarray(data[lo, tri] * (1.0 - alpha) + data[hi, tri] * alpha, dtype=np.float64)


def _triangle_series_value_at_location(series, field: TriangleMeshField2D, tri_idx: int, bary: np.ndarray, t_eval: float) -> float:
    values = _triangle_series_values_at_time(series, field, int(tri_idx), float(t_eval))
    return float(np.dot(np.asarray(bary, dtype=np.float64), values))


def _triangle_series_time_derivative_at_location(
    series,
    field: TriangleMeshField2D,
    tri_idx: int,
    bary: np.ndarray,
    t_eval: float,
) -> float:
    tri = np.asarray(field.mesh_triangles, dtype=np.int32)[int(tri_idx)]
    data = np.asarray(series.data, dtype=np.float64)
    times = np.asarray(series.times, dtype=np.float64)
    if data.ndim == 1 or data.shape[0] <= 1 or times.size <= 1:
        return 0.0
    if float(t_eval) <= float(times[0]):
        lo, hi = 0, 1
    elif float(t_eval) >= float(times[-1]):
        lo, hi = int(times.size) - 2, int(times.size) - 1
    else:
        hi = int(np.searchsorted(times, float(t_eval)))
        lo = hi - 1
    dt = float(times[hi]) - float(times[lo])
    if abs(dt) <= 1.0e-30:
        return 0.0
    weights = np.asarray(bary, dtype=np.float64)
    v_lo = float(np.dot(weights, data[lo, tri]))
    v_hi = float(np.dot(weights, data[hi, tri]))
    return float((v_hi - v_lo) / dt)


def _triangle_series_gradient_at_location(series, field: TriangleMeshField2D, tri_idx: int, t_eval: float) -> np.ndarray:
    tri = np.asarray(field.mesh_triangles, dtype=np.int32)[int(tri_idx)]
    points = np.asarray(field.mesh_vertices, dtype=np.float64)[tri]
    values = _triangle_series_values_at_time(series, field, int(tri_idx), float(t_eval))
    dx1 = float(points[1, 0] - points[0, 0])
    dy1 = float(points[1, 1] - points[0, 1])
    dx2 = float(points[2, 0] - points[0, 0])
    dy2 = float(points[2, 1] - points[0, 1])
    dv1 = float(values[1] - values[0])
    dv2 = float(values[2] - values[0])
    det = dx1 * dy2 - dy1 * dx2
    if abs(det) <= 1.0e-30:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(
        [
            (dy2 * dv1 - dy1 * dv2) / det,
            (-dx2 * dv1 + dx1 * dv2) / det,
        ],
        dtype=np.float64,
    )


def _triangle_mesh_velocity_terms(
    backend: TriangleMesh2DCompiledBackend,
    t_eval: float,
    position: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = backend.field
    names = backend.velocity_names
    if len(names) < 2:
        return (
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
        )
    tri_idx, bary = _triangle_mesh_location(backend, position)
    if int(tri_idx) < 0:
        return (
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
        )
    ux_series = field.quantities[names[0]]
    uy_series = field.quantities[names[1]]
    flow = np.asarray(
        [
            _triangle_series_value_at_location(ux_series, field, int(tri_idx), bary, float(t_eval)),
            _triangle_series_value_at_location(uy_series, field, int(tri_idx), bary, float(t_eval)),
        ],
        dtype=np.float64,
    )
    du_dt = np.asarray(
        [
            _triangle_series_time_derivative_at_location(ux_series, field, int(tri_idx), bary, float(t_eval)),
            _triangle_series_time_derivative_at_location(uy_series, field, int(tri_idx), bary, float(t_eval)),
        ],
        dtype=np.float64,
    )
    grad = np.vstack(
        (
            _triangle_series_gradient_at_location(ux_series, field, int(tri_idx), float(t_eval)),
            _triangle_series_gradient_at_location(uy_series, field, int(tri_idx), float(t_eval)),
        )
    ).astype(np.float64, copy=False)
    flow = np.where(np.isfinite(flow), flow, 0.0)
    du_dt = np.where(np.isfinite(du_dt), du_dt, 0.0)
    grad = np.where(np.isfinite(grad), grad, 0.0)
    return flow, du_dt, grad


def _triangle_mesh_scalar_gradient(
    backend: TriangleMesh2DCompiledBackend,
    quantity_name: str,
    t_eval: float,
    position: np.ndarray,
) -> np.ndarray:
    field = backend.field
    series = field.quantities.get(str(quantity_name))
    if series is None:
        return np.zeros(2, dtype=np.float64)
    tri_idx, _bary = _triangle_mesh_location(backend, position)
    if int(tri_idx) < 0:
        return np.zeros(2, dtype=np.float64)
    grad = _triangle_series_gradient_at_location(series, field, int(tri_idx), float(t_eval))
    return np.where(np.isfinite(grad), grad, 0.0).astype(np.float64, copy=False)


def _triangle_mesh_scalar_value(
    backend: TriangleMesh2DCompiledBackend,
    quantity_name: str,
    t_eval: float,
    position: np.ndarray,
) -> float:
    field = backend.field
    series = field.quantities.get(str(quantity_name))
    if series is None:
        return 0.0
    value = float(sample_triangle_mesh_series(series, field, np.asarray(position, dtype=np.float64), float(t_eval)))
    return value if np.isfinite(value) else 0.0


def coerce_compiled_backend(compiled: CompiledRuntimeBackendLike) -> CompiledRuntimeBackend:
    if isinstance(compiled, (RegularRectilinearCompiledBackend, TriangleMesh2DCompiledBackend)):
        return compiled
    backend_kind = str(compiled.get('backend_kind', 'regular_rectilinear'))
    if backend_kind == 'triangle_mesh_2d':
        field = compiled.get('field')
        if not isinstance(field, TriangleMeshField2D):
            raise TypeError('triangle_mesh_2d compiled backend requires TriangleMeshField2D under key "field"')
        return TriangleMesh2DCompiledBackend(
            field=field,
            velocity_names=tuple(compiled.get('velocity_names', ())),
            times=np.asarray(compiled.get('times', np.asarray([0.0], dtype=np.float64)), dtype=np.float64),
            ux=np.asarray(compiled.get('ux', np.zeros((1, 0), dtype=np.float64)), dtype=np.float64),
            uy=np.asarray(compiled.get('uy', np.zeros((1, 0), dtype=np.float64)), dtype=np.float64),
            gas_density=np.asarray(compiled.get('gas_density', np.ones((1, 0), dtype=np.float64)), dtype=np.float64),
            gas_mu=np.asarray(compiled.get('gas_mu', np.ones((1, 0), dtype=np.float64) * 1.8e-5), dtype=np.float64),
            gas_temperature=np.asarray(compiled.get('gas_temperature', np.ones((1, 0), dtype=np.float64) * 300.0), dtype=np.float64),
            mesh_vertices=np.asarray(compiled.get('mesh_vertices', field.mesh_vertices), dtype=np.float64),
            mesh_triangles=np.asarray(compiled.get('mesh_triangles', field.mesh_triangles), dtype=np.int32),
            accel_origin=np.asarray(compiled.get('accel_origin', field.accel_origin), dtype=np.float64),
            accel_cell_size=np.asarray(compiled.get('accel_cell_size', field.accel_cell_size), dtype=np.float64),
            accel_shape=tuple(np.asarray(compiled.get('accel_shape', field.accel_shape), dtype=np.int32).tolist()),
            accel_cell_offsets=np.asarray(compiled.get('accel_cell_offsets', field.accel_cell_offsets), dtype=np.int32),
            accel_triangle_indices=np.asarray(
                compiled.get('accel_triangle_indices', field.accel_triangle_indices),
                dtype=np.int32,
            ),
            support_tolerance_m=float(compiled.get('support_tolerance_m', field.metadata.get('support_tolerance_m', 2.0e-6))),
            acceleration_source=str(compiled.get('acceleration_source', 'none')),
            acceleration_quantity_names=tuple(compiled.get('acceleration_quantity_names', ())),
            electric_field_names=tuple(compiled.get('electric_field_names', ())),
            electric_q_over_m_Ckg=float(compiled.get('electric_q_over_m_Ckg', 0.0)),
            gas_density_source=str(compiled.get('gas_density_source', 'scalar_fallback')),
            gas_mu_source=str(compiled.get('gas_mu_source', 'scalar_fallback')),
            gas_temperature_source=str(compiled.get('gas_temperature_source', 'scalar_fallback')),
        )
    axes = tuple(np.asarray(ax, dtype=np.float64) for ax in compiled.get('axes', ()))
    valid_mask = np.asarray(compiled.get('valid_mask', np.zeros((0,), dtype=bool)), dtype=bool)
    core_valid_mask_raw = compiled.get('core_valid_mask')
    core_valid_mask = valid_mask if core_valid_mask_raw is None else np.asarray(core_valid_mask_raw, dtype=bool)
    uz_raw = compiled.get('uz')
    uz = None if uz_raw is None else np.asarray(uz_raw, dtype=np.float64)
    ex_raw = compiled.get('electric_x')
    ey_raw = compiled.get('electric_y')
    ez_raw = compiled.get('electric_z')
    electric_x = None if ex_raw is None else np.asarray(ex_raw, dtype=np.float64)
    electric_y = None if ey_raw is None else np.asarray(ey_raw, dtype=np.float64)
    electric_z = None if ez_raw is None else np.asarray(ez_raw, dtype=np.float64)
    def optional_array(name: str) -> Optional[np.ndarray]:
        raw = compiled.get(name)
        return None if raw is None else np.asarray(raw, dtype=np.float64)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=np.asarray(compiled.get('times', np.asarray([0.0], dtype=np.float64)), dtype=np.float64),
        ux=np.asarray(compiled.get('ux', np.zeros((1,) + valid_mask.shape, dtype=np.float64)), dtype=np.float64),
        uy=np.asarray(compiled.get('uy', np.zeros((1,) + valid_mask.shape, dtype=np.float64)), dtype=np.float64),
        gas_density=np.asarray(compiled.get('gas_density', np.ones((1,) + valid_mask.shape, dtype=np.float64)), dtype=np.float64),
        gas_mu=np.asarray(compiled.get('gas_mu', np.ones((1,) + valid_mask.shape, dtype=np.float64) * 1.8e-5), dtype=np.float64),
        gas_temperature=np.asarray(compiled.get('gas_temperature', np.ones((1,) + valid_mask.shape, dtype=np.float64) * 300.0), dtype=np.float64),
        valid_mask=valid_mask,
        core_valid_mask=core_valid_mask,
        uz=uz,
        electric_x=electric_x,
        electric_y=electric_y,
        electric_z=electric_z,
        acceleration_source=str(compiled.get('acceleration_source', 'none')),
        acceleration_quantity_names=tuple(compiled.get('acceleration_quantity_names', ())),
        electric_field_names=tuple(compiled.get('electric_field_names', ())),
        electric_q_over_m_Ckg=float(compiled.get('electric_q_over_m_Ckg', 0.0)),
        gas_density_source=str(compiled.get('gas_density_source', 'scalar_fallback')),
        gas_mu_source=str(compiled.get('gas_mu_source', 'scalar_fallback')),
        gas_temperature_source=str(compiled.get('gas_temperature_source', 'scalar_fallback')),
        grad_T_x=optional_array('grad_T_x'),
        grad_T_y=optional_array('grad_T_y'),
        grad_T_z=optional_array('grad_T_z'),
        grad_E2_x=optional_array('grad_E2_x'),
        grad_E2_y=optional_array('grad_E2_y'),
        grad_E2_z=optional_array('grad_E2_z'),
        vorticity_x=optional_array('vorticity_x'),
        vorticity_y=optional_array('vorticity_y'),
        vorticity_z=optional_array('vorticity_z'),
        fluid_accel_x=optional_array('fluid_accel_x'),
        fluid_accel_y=optional_array('fluid_accel_y'),
        fluid_accel_z=optional_array('fluid_accel_z'),
        du_dt_x=optional_array('du_dt_x'),
        du_dt_y=optional_array('du_dt_y'),
        du_dt_z=optional_array('du_dt_z'),
        grad_ux_x=optional_array('grad_ux_x'),
        grad_ux_y=optional_array('grad_ux_y'),
        grad_ux_z=optional_array('grad_ux_z'),
        grad_uy_x=optional_array('grad_uy_x'),
        grad_uy_y=optional_array('grad_uy_y'),
        grad_uy_z=optional_array('grad_uy_z'),
        grad_uz_x=optional_array('grad_uz_x'),
        grad_uz_y=optional_array('grad_uz_y'),
        grad_uz_z=optional_array('grad_uz_z'),
    )


def compile_runtime_backend(
    runtime,
    spatial_dim: int,
    particles=None,
    *,
    dynamic_electric: bool = False,
    enable_electric: bool = True,
    force_runtime: ForceRuntimeParameters | None = None,
) -> CompiledRuntimeBackend:
    if runtime.geometry_provider is None:
        raise ValueError('High-fidelity solver requires geometry_provider')
    geom = runtime.geometry_provider.geometry
    gas = getattr(runtime, 'gas', None)
    gas_density_kgm3 = float(getattr(gas, 'density_kgm3', 1.0))
    gas_mu_pas = float(getattr(gas, 'dynamic_viscosity_Pas', 1.8e-5))
    gas_temperature_K = float(getattr(gas, 'temperature', 300.0))
    axes = tuple(np.asarray(ax, dtype=np.float64) for ax in geom.axes)
    valid_mask = np.asarray(geom.valid_mask, dtype=bool)
    core_valid_mask = valid_mask
    times = np.asarray([0.0], dtype=np.float64)
    spatial_shape = tuple(len(ax) for ax in axes)
    shape = (1,) + spatial_shape
    ux = np.zeros(shape, dtype=np.float64)
    uy = np.zeros(shape, dtype=np.float64)
    gas_density = np.full(shape, gas_density_kgm3, dtype=np.float64)
    gas_mu = np.full(shape, gas_mu_pas, dtype=np.float64)
    gas_temperature = np.full(shape, gas_temperature_K, dtype=np.float64)
    gas_density_source = 'scalar_fallback'
    gas_mu_source = 'scalar_fallback'
    gas_temperature_source = 'scalar_fallback'
    uz = np.zeros(shape, dtype=np.float64) if spatial_dim == 3 else None
    electric_x = None
    electric_y = None
    electric_z = None
    acceleration_source = 'none'
    acceleration_quantity_names: Tuple[str, ...] = ()
    electric_field_names: Tuple[str, ...] = ()
    electric_q_over_m = 0.0
    force_params = force_runtime or ForceRuntimeParameters()
    grad_T_x = grad_T_y = grad_T_z = None
    grad_E2_x = grad_E2_y = grad_E2_z = None
    vorticity_x = vorticity_y = vorticity_z = None
    fluid_accel_x = fluid_accel_y = fluid_accel_z = None
    du_dt_x = du_dt_y = du_dt_z = None
    grad_ux_x = grad_ux_y = grad_ux_z = None
    grad_uy_x = grad_uy_y = grad_uy_z = None
    grad_uz_x = grad_uz_y = grad_uz_z = None
    field_backend_mode = _runtime_field_backend_mode(runtime)
    if runtime.field_provider is None:
        _enforce_field_backend_mode(field_backend_mode, is_triangle_mesh=False)
    if runtime.field_provider is not None:
        field = runtime.field_provider.field
        need_electric_field = bool(enable_electric) or bool(force_params.dielectrophoresis_enabled)
        electric_names = choose_electric_field_quantity_names(field, spatial_dim) if bool(need_electric_field) else ()
        if isinstance(field, TriangleMeshField2D):
            _enforce_field_backend_mode(field_backend_mode, is_triangle_mesh=True)
            names = choose_velocity_quantity_names(field, spatial_dim)
            gas_quantity_names = _gas_property_quantity_names(field)
            triangle_derived_names = triangle_derived_quantity_names(field)
            triangle_gradient_sources = triangle_mesh_gradient_source_report(field)
            if bool(force_params.lift_enabled) and not names:
                raise ValueError('solver.forces.lift requires velocity field quantities')
            if bool(force_params.thermophoresis_enabled) and not gas_quantity_names.get('gas_temperature'):
                raise ValueError('solver.forces.thermophoresis requires a temperature field quantity')
            if bool(force_params.dielectrophoresis_enabled) and len(electric_names) < 2:
                raise ValueError('solver.forces.dielectrophoresis requires electric field quantities')
            if (
                bool(force_params.pressure_gradient_enabled)
                and triangle_gradient_sources.get('fluid_acceleration') == 'unavailable'
            ):
                raise ValueError(
                    'solver.forces.pressure_gradient requires velocity field quantities '
                    'or exported fluid_accel_x/fluid_accel_y on triangle mesh'
                )
            if bool(force_params.virtual_mass_enabled) and not names:
                raise ValueError('solver.forces.virtual_mass requires velocity field quantities on triangle mesh')
            time_quantity_names = tuple(names)
            if time_quantity_names:
                times = _common_quantity_times(field, time_quantity_names)
            times = _merge_optional_quantity_times(field, times, tuple(gas_quantity_names.values()))
            times = _merge_optional_quantity_times(field, times, tuple(triangle_derived_names.values()))
            vertex_shape = (1, int(field.mesh_vertices.shape[0]))
            ux_mesh = np.zeros(vertex_shape, dtype=np.float64)
            uy_mesh = np.zeros(vertex_shape, dtype=np.float64)
            if names:
                ux_mesh = _vertex_time_grid(field.quantities[names[0]].data, times)
                uy_mesh = _vertex_time_grid(field.quantities[names[1]].data, times)
            if electric_names:
                electric_field_names = tuple(electric_names)
            time_vertex_shape = (int(max(1, times.size)), int(field.mesh_vertices.shape[0]))
            if not names:
                ux_mesh = np.zeros(time_vertex_shape, dtype=np.float64)
                uy_mesh = np.zeros(time_vertex_shape, dtype=np.float64)
            gas_density_mesh = np.full(time_vertex_shape, gas_density_kgm3, dtype=np.float64)
            gas_mu_mesh = np.full(time_vertex_shape, gas_mu_pas, dtype=np.float64)
            gas_temperature_mesh = np.full(time_vertex_shape, gas_temperature_K, dtype=np.float64)
            for target, name in gas_quantity_names.items():
                values = _vertex_time_grid(field.quantities[name].data, times)
                if target == 'gas_density':
                    gas_density_mesh = values
                elif target == 'gas_mu':
                    gas_mu_mesh = values
                elif target == 'gas_temperature':
                    gas_temperature_mesh = values
            return TriangleMesh2DCompiledBackend(
                field=field,
                velocity_names=tuple(names),
                times=times,
                ux=ux_mesh,
                uy=uy_mesh,
                gas_density=gas_density_mesh,
                gas_mu=gas_mu_mesh,
                gas_temperature=gas_temperature_mesh,
                mesh_vertices=np.asarray(field.mesh_vertices, dtype=np.float64),
                mesh_triangles=np.asarray(field.mesh_triangles, dtype=np.int32),
                accel_origin=np.asarray(field.accel_origin, dtype=np.float64),
                accel_cell_size=np.asarray(field.accel_cell_size, dtype=np.float64),
                accel_shape=tuple(np.asarray(field.accel_shape, dtype=np.int32).tolist()),
                accel_cell_offsets=np.asarray(field.accel_cell_offsets, dtype=np.int32),
                accel_triangle_indices=np.asarray(field.accel_triangle_indices, dtype=np.int32),
                support_tolerance_m=float(field.metadata.get('support_tolerance_m', 2.0e-6)),
                acceleration_source=str(acceleration_source),
                acceleration_quantity_names=tuple(acceleration_quantity_names),
                electric_field_names=tuple(electric_field_names),
                electric_q_over_m_Ckg=float(electric_q_over_m),
                gas_density_source=(
                    f"field:{gas_quantity_names['gas_density']}"
                    if 'gas_density' in gas_quantity_names
                    else 'scalar_fallback'
                ),
                gas_mu_source=(
                    f"field:{gas_quantity_names['gas_mu']}"
                    if 'gas_mu' in gas_quantity_names
                    else 'scalar_fallback'
                ),
                gas_temperature_source=(
                    f"field:{gas_quantity_names['gas_temperature']}"
                    if 'gas_temperature' in gas_quantity_names
                    else 'scalar_fallback'
                ),
                triangle_gradient_sources=dict(triangle_gradient_sources),
            )
        _enforce_field_backend_mode(field_backend_mode, is_triangle_mesh=False)
        valid_mask = np.asarray(field.valid_mask, dtype=bool)
        core_valid_mask = np.asarray(
            field.core_valid_mask if field.core_valid_mask is not None else field.valid_mask,
            dtype=bool,
        )
        names = choose_velocity_quantity_names(field, spatial_dim)
        time_quantity_names = tuple(names) + tuple(electric_names)
        gas_quantity_names = _gas_property_quantity_names(field)
        if time_quantity_names:
            times = _common_quantity_times(field, time_quantity_names)
        times = _merge_optional_quantity_times(field, times, tuple(gas_quantity_names.values()))
        shape = (int(max(1, times.size)),) + spatial_shape
        ux = np.zeros(shape, dtype=np.float64)
        uy = np.zeros(shape, dtype=np.float64)
        gas_density = np.full(shape, gas_density_kgm3, dtype=np.float64)
        gas_mu = np.full(shape, gas_mu_pas, dtype=np.float64)
        gas_temperature = np.full(shape, gas_temperature_K, dtype=np.float64)
        uz = np.zeros(shape, dtype=np.float64) if spatial_dim == 3 else None
        electric_x = None
        electric_y = None
        electric_z = None
        if names:
            if spatial_dim == 2:
                ux = _backend_time_grid(field.quantities[names[0]].data, 2, times)
                uy = _backend_time_grid(field.quantities[names[1]].data, 2, times)
            else:
                ux = _backend_time_grid(field.quantities[names[0]].data, 3, times)
                uy = _backend_time_grid(field.quantities[names[1]].data, 3, times)
                uz = _backend_time_grid(field.quantities[names[2]].data, 3, times)
        if electric_names:
            electric_field_names = tuple(electric_names)
            acceleration_source = 'particle_charge_electric_field'
            if spatial_dim == 2:
                electric_x = _backend_time_grid(field.quantities[electric_names[0]].data, 2, times)
                electric_y = _backend_time_grid(field.quantities[electric_names[1]].data, 2, times)
            else:
                electric_x = _backend_time_grid(field.quantities[electric_names[0]].data, 3, times)
                electric_y = _backend_time_grid(field.quantities[electric_names[1]].data, 3, times)
                electric_z = _backend_time_grid(field.quantities[electric_names[2]].data, 3, times)
        for target, name in gas_quantity_names.items():
            values = _backend_time_grid(field.quantities[name].data, spatial_dim, times)
            if target == 'gas_density':
                gas_density = values
                gas_density_source = f'field:{name}'
            elif target == 'gas_mu':
                gas_mu = values
                gas_mu_source = f'field:{name}'
            elif target == 'gas_temperature':
                gas_temperature = values
                gas_temperature_source = f'field:{name}'
        if bool(force_params.thermophoresis_enabled):
            grad_T = _gradient_time_grid(gas_temperature, axes)
            grad_T_x = grad_T[0]
            grad_T_y = grad_T[1]
            grad_T_z = grad_T[2] if int(spatial_dim) == 3 else None
        if bool(force_params.dielectrophoresis_enabled):
            if not electric_names or electric_x is None or electric_y is None:
                raise ValueError('solver.forces.dielectrophoresis requires electric field quantities')
            e2 = electric_x * electric_x + electric_y * electric_y
            if int(spatial_dim) == 3:
                if electric_z is None:
                    raise ValueError('solver.forces.dielectrophoresis requires 3D electric field quantities')
                e2 = e2 + electric_z * electric_z
            grad_E2 = _gradient_time_grid(e2, axes)
            grad_E2_x = grad_E2[0]
            grad_E2_y = grad_E2[1]
            grad_E2_z = grad_E2[2] if int(spatial_dim) == 3 else None
        if bool(force_params.lift_enabled):
            vorticity_x, vorticity_y, vorticity_z = _curl_from_velocity_grids(ux, uy, uz, axes)
        if bool(force_params.pressure_gradient_enabled) or bool(force_params.virtual_mass_enabled):
            if not names:
                raise ValueError('solver.forces pressure_gradient/virtual_mass require velocity field quantities')
            du_dt_x = _time_derivative_time_grid(ux, times)
            du_dt_y = _time_derivative_time_grid(uy, times)
            if int(spatial_dim) == 2:
                grad_ux_x, grad_ux_y = _gradient_time_grid(ux, axes)
                grad_uy_x, grad_uy_y = _gradient_time_grid(uy, axes)
                if bool(force_params.pressure_gradient_enabled):
                    fluid_accel_x = np.asarray(du_dt_x + ux * grad_ux_x + uy * grad_ux_y, dtype=np.float64)
                    fluid_accel_y = np.asarray(du_dt_y + ux * grad_uy_x + uy * grad_uy_y, dtype=np.float64)
            elif uz is not None:
                du_dt_z = _time_derivative_time_grid(uz, times)
                grad_ux_x, grad_ux_y, grad_ux_z = _gradient_time_grid(ux, axes)
                grad_uy_x, grad_uy_y, grad_uy_z = _gradient_time_grid(uy, axes)
                grad_uz_x, grad_uz_y, grad_uz_z = _gradient_time_grid(uz, axes)
                if bool(force_params.pressure_gradient_enabled):
                    fluid_accel_x = np.asarray(du_dt_x + ux * grad_ux_x + uy * grad_ux_y + uz * grad_ux_z, dtype=np.float64)
                    fluid_accel_y = np.asarray(du_dt_y + ux * grad_uy_x + uy * grad_uy_y + uz * grad_uy_z, dtype=np.float64)
                    fluid_accel_z = np.asarray(du_dt_z + ux * grad_uz_x + uy * grad_uz_y + uz * grad_uz_z, dtype=np.float64)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=times,
        ux=ux,
        uy=uy,
        gas_density=gas_density,
        gas_mu=gas_mu,
        gas_temperature=gas_temperature,
        valid_mask=valid_mask,
        core_valid_mask=core_valid_mask,
        uz=uz if spatial_dim == 3 else None,
        electric_x=electric_x,
        electric_y=electric_y,
        electric_z=electric_z if spatial_dim == 3 else None,
        acceleration_source=str(acceleration_source),
        acceleration_quantity_names=tuple(acceleration_quantity_names),
        electric_field_names=tuple(electric_field_names),
        electric_q_over_m_Ckg=float(electric_q_over_m),
        gas_density_source=str(gas_density_source),
        gas_mu_source=str(gas_mu_source),
        gas_temperature_source=str(gas_temperature_source),
        grad_T_x=grad_T_x,
        grad_T_y=grad_T_y,
        grad_T_z=grad_T_z,
        grad_E2_x=grad_E2_x,
        grad_E2_y=grad_E2_y,
        grad_E2_z=grad_E2_z,
        vorticity_x=vorticity_x,
        vorticity_y=vorticity_y,
        vorticity_z=vorticity_z,
        fluid_accel_x=fluid_accel_x,
        fluid_accel_y=fluid_accel_y,
        fluid_accel_z=fluid_accel_z,
        du_dt_x=du_dt_x,
        du_dt_y=du_dt_y,
        du_dt_z=du_dt_z,
        grad_ux_x=grad_ux_x,
        grad_ux_y=grad_ux_y,
        grad_ux_z=grad_ux_z,
        grad_uy_x=grad_uy_x,
        grad_uy_y=grad_uy_y,
        grad_uy_z=grad_uy_z,
        grad_uz_x=grad_uz_x,
        grad_uz_y=grad_uz_y,
        grad_uz_z=grad_uz_z,
    )


def _positive_grid_stats(values: np.ndarray, valid_mask: Optional[np.ndarray]) -> Mapping[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    if valid_mask is None:
        grid = arr.reshape(-1)
        finite = np.isfinite(grid) & (grid > 0.0)
        selected = grid[finite]
        if selected.size == 0:
            return {'finite_positive_count': 0}
        return {
            'finite_positive_count': int(selected.size),
            'min': float(np.min(selected)),
            'p50': float(np.percentile(selected, 50.0)),
            'p90': float(np.percentile(selected, 90.0)),
            'max': float(np.max(selected)),
            'mean': float(np.mean(selected)),
        }
    mask = np.asarray(valid_mask, dtype=bool)
    grid = arr[0] if arr.ndim > mask.ndim else arr
    finite = np.isfinite(grid) & (grid > 0.0)
    if grid.shape == mask.shape:
        finite = mask & finite
    selected = grid[finite]
    if selected.size == 0:
        return {'finite_positive_count': 0}
    return {
        'finite_positive_count': int(selected.size),
        'min': float(np.min(selected)),
        'p50': float(np.percentile(selected, 50.0)),
        'p90': float(np.percentile(selected, 90.0)),
        'max': float(np.max(selected)),
        'mean': float(np.mean(selected)),
    }


def compiled_gas_property_report(
    compiled: CompiledRuntimeBackendLike,
    *,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
    drag_model_name: str = '',
) -> Mapping[str, object]:
    backend = coerce_compiled_backend(compiled)
    drag_model = str(drag_model_name).strip().lower()
    report = {
        'field_backend_kind': str(getattr(backend, 'backend_kind', '')),
        'drag_model': str(drag_model_name),
        'density_source': str(getattr(backend, 'gas_density_source', 'scalar_fallback')),
        'dynamic_viscosity_source': str(getattr(backend, 'gas_mu_source', 'scalar_fallback')),
        'temperature_source': str(getattr(backend, 'gas_temperature_source', 'scalar_fallback')),
        'fallback_density_kgm3': float(fallback_density_kgm3),
        'fallback_dynamic_viscosity_Pas': float(fallback_mu_pas),
        'fallback_temperature_K': float(fallback_temperature_K),
        'pressure_source': 'diagnostic_only_not_used_by_drag',
        'uses_field_density': int(str(getattr(backend, 'gas_density_source', '')).startswith('field:')),
        'uses_field_dynamic_viscosity': int(str(getattr(backend, 'gas_mu_source', '')).startswith('field:')),
        'uses_field_temperature': int(str(getattr(backend, 'gas_temperature_source', '')).startswith('field:')),
        'density_used_by_drag_model': int(drag_model in {'epstein', 'schiller_naumann', 'stokes_cunningham'}),
        'dynamic_viscosity_used_by_drag_model': int(drag_model in {'schiller_naumann', 'stokes_cunningham'}),
        'temperature_used_by_drag_model': int(drag_model in {'epstein', 'stokes_cunningham'}),
    }
    if isinstance(backend, RegularRectilinearCompiledBackend):
        mask = np.asarray(backend.core_valid_mask, dtype=bool)
        report['density_field_stats'] = dict(_positive_grid_stats(backend.gas_density, mask))
        report['dynamic_viscosity_field_stats'] = dict(_positive_grid_stats(backend.gas_mu, mask))
        report['temperature_field_stats'] = dict(_positive_grid_stats(backend.gas_temperature, mask))
    elif isinstance(backend, TriangleMesh2DCompiledBackend):
        report['density_field_stats'] = dict(_positive_grid_stats(backend.gas_density, None))
        report['dynamic_viscosity_field_stats'] = dict(_positive_grid_stats(backend.gas_mu, None))
        report['temperature_field_stats'] = dict(_positive_grid_stats(backend.gas_temperature, None))
        report['triangle_gradient_sources'] = dict(getattr(backend, 'triangle_gradient_sources', {}))
    return report


def sample_compiled_flow_vector(
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_eval: float,
    position: np.ndarray,
) -> np.ndarray:
    backend = coerce_compiled_backend(compiled)
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        field = backend.field
        names = backend.velocity_names
        if not names:
            return np.zeros(spatial_dim, dtype=np.float64)
        pos = np.asarray(position, dtype=np.float64)
        ux = float(sample_triangle_mesh_series(field.quantities[names[0]], field, pos, float(t_eval)))
        uy = float(sample_triangle_mesh_series(field.quantities[names[1]], field, pos, float(t_eval)))
        if not np.isfinite(ux):
            ux = 0.0
        if not np.isfinite(uy):
            uy = 0.0
        return np.asarray([ux, uy], dtype=np.float64)
    axes = backend.axes
    times = np.asarray(backend.times, dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)
    ux = float(sample_time_grid_scalar(backend.ux, axes, times, t_eval, pos))
    uy = float(sample_time_grid_scalar(backend.uy, axes, times, t_eval, pos))
    if not np.isfinite(ux):
        ux = 0.0
    if not np.isfinite(uy):
        uy = 0.0
    if spatial_dim == 2:
        return np.asarray([ux, uy], dtype=np.float64)
    uz_grid = backend.uz if backend.uz is not None else np.zeros((1,) + tuple(len(ax) for ax in axes), dtype=np.float64)
    uz = float(sample_time_grid_scalar(uz_grid, axes, times, t_eval, pos))
    if not np.isfinite(uz):
        uz = 0.0
    return np.asarray([ux, uy, uz], dtype=np.float64)


def sample_compiled_flow_vectors(
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray:
    backend = coerce_compiled_backend(compiled)
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError('positions must have shape (n, spatial_dim)')
    if pts.shape[0] == 0:
        return np.zeros((0, int(spatial_dim)), dtype=np.float64)
    if int(spatial_dim) == 2 and isinstance(backend, RegularRectilinearCompiledBackend):
        axes = backend.axes
        times = np.asarray(backend.times, dtype=np.float64)
        ux = _sample_regular_time_grid_points_2d(backend.ux, axes, times, float(t_eval), pts)
        uy = _sample_regular_time_grid_points_2d(backend.uy, axes, times, float(t_eval), pts)
        return np.column_stack((ux, uy)).astype(np.float64, copy=False)
    return np.asarray(
        [sample_compiled_flow_vector(backend, int(spatial_dim), float(t_eval), point) for point in pts],
        dtype=np.float64,
    )


_EPS0_F_M = 8.8541878128e-12
_K_BOLTZMANN = 1.380649e-23
_AMU_KG = 1.66053906660e-27


def _particle_mass_from_inputs(diameter: float, density: float, mass: Optional[float]) -> float:
    if mass is not None and np.isfinite(float(mass)) and float(mass) > 0.0:
        return float(mass)
    d = max(float(diameter), 0.0)
    rho = max(float(density), 0.0)
    if d <= 0.0 or rho <= 0.0:
        return 0.0
    return float(rho * np.pi * d * d * d / 6.0)


def _evaluate_2d_force_pipeline_from_samples(
    *,
    force_runtime: ForceRuntimeParameters,
    electric_q_over_m: Optional[float],
    electric_field: Optional[np.ndarray],
    diameter: float,
    density: float,
    mass: Optional[float],
    dep_particle_rel_permittivity: float,
    thermophoretic_coeff: float,
    velocity: np.ndarray,
    flow_velocity: np.ndarray,
    grad_T: np.ndarray,
    grad_E2: np.ndarray,
    vorticity_z: float,
    fluid_acceleration: np.ndarray,
    flow_time_derivative: np.ndarray,
    flow_velocity_gradient: np.ndarray,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    t_eval: float,
) -> np.ndarray:
    # 2D non-drag forces use the batch pipeline from pre-sampled fields. The
    # scalar/vector public samplers below stay as compatibility entry points
    # and as 3D fallback paths until equivalent sampled-field coverage exists.
    params = force_runtime or ForceRuntimeParameters()
    qom_arr = None
    electric = None
    if electric_q_over_m is not None and np.isfinite(float(electric_q_over_m)) and electric_field is not None:
        qom_arr = np.asarray([float(electric_q_over_m)], dtype=np.float64)
        electric = np.asarray(electric_field, dtype=np.float64).reshape(1, 2)
    mass_value = _particle_mass_from_inputs(float(diameter), float(density), mass)
    pipeline = build_force_pipeline(params, include_electric=electric is not None and qom_arr is not None)
    out = np.zeros((1, 2), dtype=np.float64)
    evaluate_force_pipeline(
        out,
        ForceBatchStatic(
            particle_diameter=np.asarray([float(diameter)], dtype=np.float64),
            particle_density=np.asarray([float(density)], dtype=np.float64),
            particle_mass=np.asarray([float(mass_value)], dtype=np.float64),
            dep_particle_rel_permittivity=np.asarray([float(dep_particle_rel_permittivity)], dtype=np.float64),
            thermophoretic_coeff=np.asarray([float(thermophoretic_coeff)], dtype=np.float64),
        ),
        ForceBatchState(velocity=np.asarray(velocity, dtype=np.float64).reshape(1, 2)),
        None,
        ForceBatchSamples(
            electric_field=electric,
            flow_velocity=np.asarray(flow_velocity, dtype=np.float64).reshape(1, 2),
            gas_density=np.asarray([float(gas_density_kgm3)], dtype=np.float64),
            gas_mu=np.asarray([float(gas_mu_pas)], dtype=np.float64),
            gas_temperature=np.asarray([float(gas_temperature_K)], dtype=np.float64),
            grad_T=np.asarray(grad_T, dtype=np.float64).reshape(1, 2),
            grad_E2=np.asarray(grad_E2, dtype=np.float64).reshape(1, 2),
            vorticity_z=np.asarray([float(vorticity_z)], dtype=np.float64),
            fluid_acceleration=np.asarray(fluid_acceleration, dtype=np.float64).reshape(1, 2),
            flow_time_derivative=np.asarray(flow_time_derivative, dtype=np.float64).reshape(1, 2),
            flow_velocity_gradient=np.asarray(flow_velocity_gradient, dtype=np.float64).reshape(1, 2, 2),
            electric_q_over_m=qom_arr,
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        ),
        pipeline,
        float(t_eval),
    )
    return np.asarray(out[0], dtype=np.float64)


def _cm_factor_real(
    particle_rel_permittivity: float,
    medium_rel_permittivity: float,
    particle_conductivity_Sm: float,
    medium_conductivity_Sm: float,
    frequency_Hz: float,
) -> float:
    eps_p = float(particle_rel_permittivity)
    eps_m = float(medium_rel_permittivity)
    if not np.isfinite(eps_p) or eps_p <= 0.0:
        eps_p = 2.0
    if not np.isfinite(eps_m) or eps_m <= 0.0:
        eps_m = 1.0006
    freq = max(float(frequency_Hz), 0.0)
    if freq <= 0.0:
        return float((eps_p - eps_m) / (eps_p + 2.0 * eps_m))
    omega = 2.0 * np.pi * freq
    rel_p = complex(eps_p, -float(particle_conductivity_Sm) / max(omega * _EPS0_F_M, 1.0e-300))
    rel_m = complex(eps_m, -float(medium_conductivity_Sm) / max(omega * _EPS0_F_M, 1.0e-300))
    value = (rel_p - rel_m) / (rel_p + 2.0 * rel_m)
    return float(value.real)


def _extra_force_acceleration_from_samples(
    *,
    force_runtime: ForceRuntimeParameters,
    diameter: float,
    density: float,
    mass: Optional[float],
    dep_particle_rel_permittivity: float,
    thermophoretic_coeff: float,
    velocity: np.ndarray,
    flow_velocity: np.ndarray,
    grad_T: np.ndarray,
    grad_E2: np.ndarray,
    vorticity: np.ndarray,
    fluid_acceleration: np.ndarray,
    flow_time_derivative: np.ndarray,
    flow_velocity_gradient: np.ndarray,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
) -> np.ndarray:
    dim = int(np.asarray(velocity, dtype=np.float64).size)
    if dim == 2:
        return _evaluate_2d_force_pipeline_from_samples(
            force_runtime=force_runtime,
            electric_q_over_m=None,
            electric_field=None,
            diameter=float(diameter),
            density=float(density),
            mass=mass,
            dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
            thermophoretic_coeff=float(thermophoretic_coeff),
            velocity=np.asarray(velocity, dtype=np.float64)[:2],
            flow_velocity=np.asarray(flow_velocity, dtype=np.float64)[:2],
            grad_T=np.asarray(grad_T, dtype=np.float64)[:2],
            grad_E2=np.asarray(grad_E2, dtype=np.float64)[:2],
            vorticity_z=float(np.asarray(vorticity, dtype=np.float64)[-1]),
            fluid_acceleration=np.asarray(fluid_acceleration, dtype=np.float64)[:2],
            flow_time_derivative=np.asarray(flow_time_derivative, dtype=np.float64)[:2],
            flow_velocity_gradient=np.asarray(flow_velocity_gradient, dtype=np.float64)[:2, :2],
            gas_density_kgm3=float(gas_density_kgm3),
            gas_mu_pas=float(gas_mu_pas),
            gas_temperature_K=float(gas_temperature_K),
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
            t_eval=0.0,
        )
    out = np.zeros(dim, dtype=np.float64)
    d = max(float(diameter), 0.0)
    radius = 0.5 * d
    m = _particle_mass_from_inputs(d, float(density), mass)
    rho_g = max(float(gas_density_kgm3), 0.0)
    rho_p = max(float(density), 0.0)
    if bool(force_runtime.pressure_gradient_enabled) and rho_p > 0.0 and rho_g > 0.0:
        fluid_accel = np.asarray(fluid_acceleration, dtype=np.float64)[:dim]
        if np.all(np.isfinite(fluid_accel)):
            out += (rho_g / rho_p) * fluid_accel
    if bool(force_runtime.virtual_mass_enabled) and rho_p > 0.0 and rho_g > 0.0:
        coeff = max(float(force_runtime.virtual_mass_coefficient), 0.0)
        dudt = np.asarray(flow_time_derivative, dtype=np.float64)[:dim]
        grad_u = np.asarray(flow_velocity_gradient, dtype=np.float64)[:dim, :dim]
        vel = np.asarray(velocity, dtype=np.float64)[:dim]
        particle_path_fluid_accel = dudt + grad_u @ vel
        if np.all(np.isfinite(particle_path_fluid_accel)):
            out += coeff * (rho_g / rho_p) * particle_path_fluid_accel
    if d <= 0.0 or radius <= 0.0 or m <= 0.0:
        return out
    mu = max(float(gas_mu_pas), 0.0)
    temp = max(float(gas_temperature_K), 1.0)
    if bool(force_runtime.thermophoresis_enabled) and rho_g > 0.0 and mu > 0.0:
        mol_mass = max(float(gas_molecular_mass_kg), 1.0e-30)
        mean_free_path = (mu / rho_g) * np.sqrt(np.pi * mol_mass / (2.0 * _K_BOLTZMANN * temp))
        kn = max(float(mean_free_path / radius), 0.0)
        if str(force_runtime.thermophoresis_model).lower() == "continuum":
            kn = 0.0
        kg = max(float(force_runtime.gas_thermal_conductivity_W_mK), 1.0e-30)
        kp = max(float(force_runtime.particle_thermal_conductivity_W_mK), 1.0e-30)
        ratio = kg / kp
        factor = (
            float(force_runtime.thermophoresis_Cs)
            * (ratio + float(force_runtime.thermophoresis_Ct) * kn)
            / max(
                (1.0 + 3.0 * float(force_runtime.thermophoresis_Cm) * kn)
                * (1.0 + 2.0 * ratio + 2.0 * float(force_runtime.thermophoresis_Ct) * kn),
                1.0e-30,
            )
        )
        multiplier = float(thermophoretic_coeff)
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            multiplier = 1.0
        drift = -multiplier * factor * mu / max(rho_g * temp, 1.0e-30) * np.asarray(grad_T, dtype=np.float64)
        tau_stokes = max(m / max(3.0 * np.pi * mu * d, 1.0e-300), 1.0e-30)
        out += drift[:dim] / tau_stokes
    if bool(force_runtime.dielectrophoresis_enabled):
        epsp = float(dep_particle_rel_permittivity)
        if not np.isfinite(epsp) or epsp <= 0.0:
            epsp = float(force_runtime.dep_particle_rel_permittivity)
        cm_real = _cm_factor_real(
            epsp,
            float(force_runtime.dep_medium_rel_permittivity),
            float(force_runtime.dep_particle_conductivity_Sm),
            float(force_runtime.dep_medium_conductivity_Sm),
            float(force_runtime.dep_frequency_Hz),
        )
        dep_coeff = 2.0 * np.pi * _EPS0_F_M * float(force_runtime.dep_medium_rel_permittivity) * radius**3 * cm_real
        out += dep_coeff * np.asarray(grad_E2, dtype=np.float64)[:dim] / m
    if bool(force_runtime.lift_enabled) and rho_g > 0.0 and mu > 0.0:
        vel = np.asarray(velocity, dtype=np.float64)
        flow = np.asarray(flow_velocity, dtype=np.float64)
        slip = vel[:dim] - flow[:dim]
        nu = mu / max(rho_g, 1.0e-30)
        if dim == 2:
            omega_z = float(np.asarray(vorticity, dtype=np.float64)[-1])
            omega_mag = abs(omega_z)
            if omega_mag > 1.0e-30:
                cross = np.asarray([slip[1] * omega_z, -slip[0] * omega_z], dtype=np.float64)
                out += float(force_runtime.lift_coefficient) * mu * radius * radius * cross / np.sqrt(nu * omega_mag) / m
        elif dim == 3:
            omega = np.asarray(vorticity, dtype=np.float64)[:3]
            omega_mag = float(np.linalg.norm(omega))
            if omega_mag > 1.0e-30:
                out += (
                    float(force_runtime.lift_coefficient)
                    * mu
                    * radius
                    * radius
                    * np.cross(slip[:3], omega)
                    / np.sqrt(nu * omega_mag)
                    / m
                )
    return out


def sample_compiled_acceleration_vector(
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_eval: float,
    position: np.ndarray,
    *,
    electric_q_over_m: Optional[float] = None,
    force_runtime: ForceRuntimeParameters | None = None,
    particle_diameter: float = 0.0,
    particle_density: float = 0.0,
    particle_mass: Optional[float] = None,
    dep_particle_rel_permittivity: float = float("nan"),
    thermophoretic_coeff: float = float("nan"),
    velocity: Optional[np.ndarray] = None,
    flow_velocity: Optional[np.ndarray] = None,
    gas_density_kgm3: float = 1.0,
    gas_mu_pas: float = 1.8e-5,
    gas_temperature_K: float = 300.0,
    gas_molecular_mass_kg: float = 60.0 * _AMU_KG,
) -> np.ndarray:
    backend = coerce_compiled_backend(compiled)
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        params = force_runtime or ForceRuntimeParameters()
        if int(spatial_dim) != 2:
            return np.zeros(spatial_dim, dtype=np.float64)
        pos = np.asarray(position, dtype=np.float64)
        electric_field = None
        if electric_q_over_m is not None and np.isfinite(float(electric_q_over_m)) and len(backend.electric_field_names) >= 2:
            ex = _triangle_mesh_scalar_value(backend, backend.electric_field_names[0], float(t_eval), pos)
            ey = _triangle_mesh_scalar_value(backend, backend.electric_field_names[1], float(t_eval), pos)
            electric_field = np.asarray([ex, ey], dtype=np.float64)
        has_extra_forces = (
            bool(params.thermophoresis_enabled)
            or bool(params.dielectrophoresis_enabled)
            or bool(params.lift_enabled)
            or bool(params.pressure_gradient_enabled)
            or bool(params.virtual_mass_enabled)
        )
        if not bool(has_extra_forces):
            out = _evaluate_2d_force_pipeline_from_samples(
                force_runtime=params,
                electric_q_over_m=electric_q_over_m,
                electric_field=electric_field,
                diameter=float(particle_diameter),
                density=float(particle_density),
                mass=particle_mass,
                dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
                thermophoretic_coeff=float(thermophoretic_coeff),
                velocity=np.zeros(2, dtype=np.float64) if velocity is None else np.asarray(velocity, dtype=np.float64)[:2],
                flow_velocity=np.zeros(2, dtype=np.float64),
                grad_T=np.zeros(2, dtype=np.float64),
                grad_E2=np.zeros(2, dtype=np.float64),
                vorticity_z=0.0,
                fluid_acceleration=np.zeros(2, dtype=np.float64),
                flow_time_derivative=np.zeros(2, dtype=np.float64),
                flow_velocity_gradient=np.zeros((2, 2), dtype=np.float64),
                gas_density_kgm3=float(gas_density_kgm3),
                gas_mu_pas=float(gas_mu_pas),
                gas_temperature_K=float(gas_temperature_K),
                gas_molecular_mass_kg=float(gas_molecular_mass_kg),
                t_eval=float(t_eval),
            )
            return np.asarray(out, dtype=np.float64)
        sampled_flow, flow_time_derivative, flow_velocity_gradient = _triangle_mesh_velocity_terms(backend, float(t_eval), pos)
        flow = (
            np.asarray(flow_velocity, dtype=np.float64)[:2]
            if flow_velocity is not None
            else sampled_flow[:2]
        )
        vel = np.zeros(2, dtype=np.float64) if velocity is None else np.asarray(velocity, dtype=np.float64)[:2]
        grad_T = np.zeros(2, dtype=np.float64)
        gas_names = _gas_property_quantity_names(backend.field)
        triangle_derived_names = triangle_derived_quantity_names(backend.field)
        temp_name = gas_names.get('gas_temperature')
        if (
            bool(params.thermophoresis_enabled)
            and 'grad_T_x' in triangle_derived_names
            and 'grad_T_y' in triangle_derived_names
        ):
            grad_T = np.asarray(
                [
                    sample_triangle_mesh_series(
                        backend.field.quantities[triangle_derived_names['grad_T_x']],
                        backend.field,
                        pos,
                        float(t_eval),
                    ),
                    sample_triangle_mesh_series(
                        backend.field.quantities[triangle_derived_names['grad_T_y']],
                        backend.field,
                        pos,
                        float(t_eval),
                    ),
                ],
                dtype=np.float64,
            )
        elif bool(params.thermophoresis_enabled) and temp_name:
            grad_T = _triangle_mesh_scalar_gradient(backend, temp_name, float(t_eval), pos)
        grad_E2 = np.zeros(2, dtype=np.float64)
        if (
            bool(params.dielectrophoresis_enabled)
            and 'grad_E2_x' in triangle_derived_names
            and 'grad_E2_y' in triangle_derived_names
        ):
            grad_E2 = np.asarray(
                [
                    sample_triangle_mesh_series(
                        backend.field.quantities[triangle_derived_names['grad_E2_x']],
                        backend.field,
                        pos,
                        float(t_eval),
                    ),
                    sample_triangle_mesh_series(
                        backend.field.quantities[triangle_derived_names['grad_E2_y']],
                        backend.field,
                        pos,
                        float(t_eval),
                    ),
                ],
                dtype=np.float64,
            )
        elif bool(params.dielectrophoresis_enabled) and len(backend.electric_field_names) >= 2:
            ex_name, ey_name = backend.electric_field_names[:2]
            ex = _triangle_mesh_scalar_value(backend, ex_name, float(t_eval), pos)
            ey = _triangle_mesh_scalar_value(backend, ey_name, float(t_eval), pos)
            grad_ex = _triangle_mesh_scalar_gradient(backend, ex_name, float(t_eval), pos)
            grad_ey = _triangle_mesh_scalar_gradient(backend, ey_name, float(t_eval), pos)
            grad_E2 = 2.0 * ex * grad_ex + 2.0 * ey * grad_ey
        vorticity = np.zeros(3, dtype=np.float64)
        if bool(params.lift_enabled) and 'vorticity_z' in triangle_derived_names:
            vorticity[2] = float(
                sample_triangle_mesh_series(
                    backend.field.quantities[triangle_derived_names['vorticity_z']],
                    backend.field,
                    pos,
                    float(t_eval),
                )
            )
        elif bool(params.lift_enabled):
            vorticity[2] = float(flow_velocity_gradient[1, 0] - flow_velocity_gradient[0, 1])
        if 'fluid_accel_x' in triangle_derived_names and 'fluid_accel_y' in triangle_derived_names:
            fluid_acceleration = np.asarray(
                [
                    sample_triangle_mesh_series(
                        backend.field.quantities[triangle_derived_names['fluid_accel_x']],
                        backend.field,
                        pos,
                        float(t_eval),
                    ),
                    sample_triangle_mesh_series(
                        backend.field.quantities[triangle_derived_names['fluid_accel_y']],
                        backend.field,
                        pos,
                        float(t_eval),
                    ),
                ],
                dtype=np.float64,
            )
        else:
            fluid_acceleration = flow_time_derivative + flow_velocity_gradient @ sampled_flow[:2]
        rho_local, mu_local, temp_local = sample_compiled_gas_properties(
            backend,
            float(t_eval),
            pos,
            fallback_density_kgm3=float(gas_density_kgm3),
            fallback_mu_pas=float(gas_mu_pas),
            fallback_temperature_K=float(gas_temperature_K),
        )
        out = _evaluate_2d_force_pipeline_from_samples(
            force_runtime=params,
            electric_q_over_m=electric_q_over_m,
            electric_field=electric_field,
            diameter=float(particle_diameter),
            density=float(particle_density),
            mass=particle_mass,
            dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
            thermophoretic_coeff=float(thermophoretic_coeff),
            velocity=vel,
            flow_velocity=flow,
            grad_T=grad_T,
            grad_E2=grad_E2,
            vorticity_z=float(vorticity[2]),
            fluid_acceleration=fluid_acceleration,
            flow_time_derivative=flow_time_derivative,
            flow_velocity_gradient=flow_velocity_gradient,
            gas_density_kgm3=float(rho_local),
            gas_mu_pas=float(mu_local),
            gas_temperature_K=float(temp_local),
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
            t_eval=float(t_eval),
        )
        return np.where(np.isfinite(out), out, 0.0).astype(np.float64, copy=False)
    axes = backend.axes
    times = np.asarray(backend.times, dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)
    ax = 0.0
    ay = 0.0
    if electric_q_over_m is not None and np.isfinite(float(electric_q_over_m)) and backend.electric_x is not None and backend.electric_y is not None:
        ax += float(electric_q_over_m) * float(sample_time_grid_scalar(backend.electric_x, axes, times, t_eval, pos))
        ay += float(electric_q_over_m) * float(sample_time_grid_scalar(backend.electric_y, axes, times, t_eval, pos))
    params = force_runtime or ForceRuntimeParameters()
    if (
        bool(params.thermophoresis_enabled)
        or bool(params.dielectrophoresis_enabled)
        or bool(params.lift_enabled)
        or bool(params.pressure_gradient_enabled)
        or bool(params.virtual_mass_enabled)
    ):
        grad_T = np.zeros(int(spatial_dim), dtype=np.float64)
        grad_E2 = np.zeros(int(spatial_dim), dtype=np.float64)
        fluid_acceleration = np.zeros(int(spatial_dim), dtype=np.float64)
        flow_time_derivative = np.zeros(int(spatial_dim), dtype=np.float64)
        flow_velocity_gradient = np.zeros((int(spatial_dim), int(spatial_dim)), dtype=np.float64)
        vorticity = np.zeros(3, dtype=np.float64)
        if backend.grad_T_x is not None and backend.grad_T_y is not None:
            grad_T[0] = float(sample_time_grid_scalar(backend.grad_T_x, axes, times, t_eval, pos))
            grad_T[1] = float(sample_time_grid_scalar(backend.grad_T_y, axes, times, t_eval, pos))
            if int(spatial_dim) == 3 and backend.grad_T_z is not None:
                grad_T[2] = float(sample_time_grid_scalar(backend.grad_T_z, axes, times, t_eval, pos))
        if backend.grad_E2_x is not None and backend.grad_E2_y is not None:
            grad_E2[0] = float(sample_time_grid_scalar(backend.grad_E2_x, axes, times, t_eval, pos))
            grad_E2[1] = float(sample_time_grid_scalar(backend.grad_E2_y, axes, times, t_eval, pos))
            if int(spatial_dim) == 3 and backend.grad_E2_z is not None:
                grad_E2[2] = float(sample_time_grid_scalar(backend.grad_E2_z, axes, times, t_eval, pos))
        if int(spatial_dim) == 2:
            if backend.vorticity_z is not None:
                vorticity[2] = float(sample_time_grid_scalar(backend.vorticity_z, axes, times, t_eval, pos))
        else:
            if backend.vorticity_x is not None:
                vorticity[0] = float(sample_time_grid_scalar(backend.vorticity_x, axes, times, t_eval, pos))
            if backend.vorticity_y is not None:
                vorticity[1] = float(sample_time_grid_scalar(backend.vorticity_y, axes, times, t_eval, pos))
            if backend.vorticity_z is not None:
                vorticity[2] = float(sample_time_grid_scalar(backend.vorticity_z, axes, times, t_eval, pos))
        if backend.fluid_accel_x is not None and backend.fluid_accel_y is not None:
            fluid_acceleration[0] = float(sample_time_grid_scalar(backend.fluid_accel_x, axes, times, t_eval, pos))
            fluid_acceleration[1] = float(sample_time_grid_scalar(backend.fluid_accel_y, axes, times, t_eval, pos))
            if int(spatial_dim) == 3 and backend.fluid_accel_z is not None:
                fluid_acceleration[2] = float(sample_time_grid_scalar(backend.fluid_accel_z, axes, times, t_eval, pos))
        if backend.du_dt_x is not None and backend.du_dt_y is not None:
            flow_time_derivative[0] = float(sample_time_grid_scalar(backend.du_dt_x, axes, times, t_eval, pos))
            flow_time_derivative[1] = float(sample_time_grid_scalar(backend.du_dt_y, axes, times, t_eval, pos))
            if int(spatial_dim) == 3 and backend.du_dt_z is not None:
                flow_time_derivative[2] = float(sample_time_grid_scalar(backend.du_dt_z, axes, times, t_eval, pos))
        if backend.grad_ux_x is not None and backend.grad_ux_y is not None:
            flow_velocity_gradient[0, 0] = float(sample_time_grid_scalar(backend.grad_ux_x, axes, times, t_eval, pos))
            flow_velocity_gradient[0, 1] = float(sample_time_grid_scalar(backend.grad_ux_y, axes, times, t_eval, pos))
            flow_velocity_gradient[1, 0] = float(sample_time_grid_scalar(backend.grad_uy_x, axes, times, t_eval, pos)) if backend.grad_uy_x is not None else 0.0
            flow_velocity_gradient[1, 1] = float(sample_time_grid_scalar(backend.grad_uy_y, axes, times, t_eval, pos)) if backend.grad_uy_y is not None else 0.0
            if int(spatial_dim) == 3:
                if backend.grad_ux_z is not None:
                    flow_velocity_gradient[0, 2] = float(sample_time_grid_scalar(backend.grad_ux_z, axes, times, t_eval, pos))
                if backend.grad_uy_z is not None:
                    flow_velocity_gradient[1, 2] = float(sample_time_grid_scalar(backend.grad_uy_z, axes, times, t_eval, pos))
                if backend.grad_uz_x is not None:
                    flow_velocity_gradient[2, 0] = float(sample_time_grid_scalar(backend.grad_uz_x, axes, times, t_eval, pos))
                if backend.grad_uz_y is not None:
                    flow_velocity_gradient[2, 1] = float(sample_time_grid_scalar(backend.grad_uz_y, axes, times, t_eval, pos))
                if backend.grad_uz_z is not None:
                    flow_velocity_gradient[2, 2] = float(sample_time_grid_scalar(backend.grad_uz_z, axes, times, t_eval, pos))
        flow = (
            np.asarray(flow_velocity, dtype=np.float64)
            if flow_velocity is not None
            else sample_compiled_flow_vector(backend, int(spatial_dim), float(t_eval), pos)
        )
        vel = np.zeros(int(spatial_dim), dtype=np.float64) if velocity is None else np.asarray(velocity, dtype=np.float64)
        rho_local, mu_local, temp_local = sample_compiled_gas_properties(
            backend,
            float(t_eval),
            pos,
            fallback_density_kgm3=float(gas_density_kgm3),
            fallback_mu_pas=float(gas_mu_pas),
            fallback_temperature_K=float(gas_temperature_K),
        )
        extra = _extra_force_acceleration_from_samples(
            force_runtime=params,
            diameter=float(particle_diameter),
            density=float(particle_density),
            mass=particle_mass,
            dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
            thermophoretic_coeff=float(thermophoretic_coeff),
            velocity=vel[: int(spatial_dim)],
            flow_velocity=flow[: int(spatial_dim)],
            grad_T=grad_T,
            grad_E2=grad_E2,
            vorticity=vorticity,
            fluid_acceleration=fluid_acceleration,
            flow_time_derivative=flow_time_derivative,
            flow_velocity_gradient=flow_velocity_gradient,
            gas_density_kgm3=float(rho_local),
            gas_mu_pas=float(mu_local),
            gas_temperature_K=float(temp_local),
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        )
        ax += float(extra[0])
        ay += float(extra[1])
    if not np.isfinite(ax):
        ax = 0.0
    if not np.isfinite(ay):
        ay = 0.0
    if spatial_dim == 2:
        return np.asarray([ax, ay], dtype=np.float64)
    az = 0.0
    if electric_q_over_m is not None and np.isfinite(float(electric_q_over_m)) and backend.electric_z is not None:
        az += float(electric_q_over_m) * float(sample_time_grid_scalar(backend.electric_z, axes, times, t_eval, pos))
    if (
        int(spatial_dim) == 3
        and (
            bool(params.thermophoresis_enabled)
            or bool(params.dielectrophoresis_enabled)
            or bool(params.lift_enabled)
            or bool(params.pressure_gradient_enabled)
            or bool(params.virtual_mass_enabled)
        )
    ):
        az += float(extra[2])
    if not np.isfinite(az):
        az = 0.0
    return np.asarray([ax, ay, az], dtype=np.float64)


def sample_compiled_gas_properties(
    compiled: CompiledRuntimeBackendLike,
    t_eval: float,
    position: np.ndarray,
    *,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
) -> Tuple[float, float, float]:
    backend = coerce_compiled_backend(compiled)
    rho = float(fallback_density_kgm3)
    mu = float(fallback_mu_pas)
    temp = float(fallback_temperature_K)
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        field = backend.field
        gas_names = _gas_property_quantity_names(field)
        meta_rho = field.metadata.get('gas_density_kgm3') if isinstance(field.metadata, Mapping) else None
        if meta_rho is not None and np.isfinite(float(meta_rho)) and float(meta_rho) > 0.0:
            rho = float(meta_rho)
        for target, name in gas_names.items():
            sample = float(sample_triangle_mesh_series(field.quantities[name], field, position, float(t_eval)))
            if not np.isfinite(sample) or sample <= 0.0:
                continue
            if target == 'gas_density':
                rho = sample
            elif target == 'gas_mu':
                mu = sample
            elif target == 'gas_temperature':
                temp = sample
        return rho, mu, temp
    axes = backend.axes
    times = np.asarray(backend.times, dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)
    rho_sample = float(sample_time_grid_scalar(backend.gas_density, axes, times, float(t_eval), pos))
    mu_sample = float(sample_time_grid_scalar(backend.gas_mu, axes, times, float(t_eval), pos))
    temp_sample = float(sample_time_grid_scalar(backend.gas_temperature, axes, times, float(t_eval), pos))
    if np.isfinite(rho_sample) and rho_sample > 0.0:
        rho = rho_sample
    if np.isfinite(mu_sample) and mu_sample > 0.0:
        mu = mu_sample
    if np.isfinite(temp_sample) and temp_sample > 0.0:
        temp = temp_sample
    return float(rho), float(mu), float(temp)


def sample_compiled_gas_properties_vectors(
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
    *,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    backend = coerce_compiled_backend(compiled)
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError('positions must have shape (n, spatial_dim)')
    n = int(pts.shape[0])
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    if int(spatial_dim) == 2 and isinstance(backend, RegularRectilinearCompiledBackend):
        axes = backend.axes
        times = np.asarray(backend.times, dtype=np.float64)
        rho = _sample_regular_time_grid_points_2d(backend.gas_density, axes, times, float(t_eval), pts)
        mu = _sample_regular_time_grid_points_2d(backend.gas_mu, axes, times, float(t_eval), pts)
        temp = _sample_regular_time_grid_points_2d(backend.gas_temperature, axes, times, float(t_eval), pts)
        rho = np.where(np.isfinite(rho) & (rho > 0.0), rho, float(fallback_density_kgm3))
        mu = np.where(np.isfinite(mu) & (mu > 0.0), mu, float(fallback_mu_pas))
        temp = np.where(np.isfinite(temp) & (temp > 0.0), temp, float(fallback_temperature_K))
        return (
            np.asarray(rho, dtype=np.float64),
            np.asarray(mu, dtype=np.float64),
            np.asarray(temp, dtype=np.float64),
        )
    values = [
        sample_compiled_gas_properties(
            backend,
            float(t_eval),
            point,
            fallback_density_kgm3=float(fallback_density_kgm3),
            fallback_mu_pas=float(fallback_mu_pas),
            fallback_temperature_K=float(fallback_temperature_K),
        )
        for point in pts
    ]
    arr = np.asarray(values, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _sample_regular_2d_acceleration_vectors_pipeline(
    backend: RegularRectilinearCompiledBackend,
    t_eval: float,
    pts: np.ndarray,
    *,
    electric_q_over_m: Optional[np.ndarray],
    force_runtime: ForceRuntimeParameters | None,
    particle_diameter: Optional[np.ndarray],
    particle_density: Optional[np.ndarray],
    particle_mass: Optional[np.ndarray],
    dep_particle_rel_permittivity: Optional[np.ndarray],
    thermophoretic_coeff: Optional[np.ndarray],
    velocity: Optional[np.ndarray],
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
) -> np.ndarray:
    axes = backend.axes
    times = np.asarray(backend.times, dtype=np.float64)
    points = np.asarray(pts, dtype=np.float64)
    n = int(points.shape[0])
    params = force_runtime or ForceRuntimeParameters()
    electric_field = None
    qom = None
    if electric_q_over_m is not None and backend.electric_x is not None and backend.electric_y is not None:
        qom = np.asarray(electric_q_over_m, dtype=np.float64).reshape(-1)
        if qom.shape[0] != n:
            raise ValueError('electric_q_over_m must match positions length')
        ex = _sample_regular_time_grid_points_2d(backend.electric_x, axes, times, float(t_eval), points)
        ey = _sample_regular_time_grid_points_2d(backend.electric_y, axes, times, float(t_eval), points)
        electric_field = np.column_stack((ex, ey)).astype(np.float64, copy=False)
    pipeline = build_force_pipeline(params, include_electric=electric_field is not None and qom is not None)

    d = (
        np.asarray(particle_diameter, dtype=np.float64).reshape(-1)
        if particle_diameter is not None
        else np.zeros(n, dtype=np.float64)
    )
    rho_p = (
        np.asarray(particle_density, dtype=np.float64).reshape(-1)
        if particle_density is not None
        else np.zeros(n, dtype=np.float64)
    )
    mass = (
        np.asarray(particle_mass, dtype=np.float64).reshape(-1)
        if particle_mass is not None
        else rho_p * np.pi * d * d * d / 6.0
    )
    epsp_arr = (
        np.asarray(dep_particle_rel_permittivity, dtype=np.float64).reshape(-1)
        if dep_particle_rel_permittivity is not None
        else np.full(n, float(params.dep_particle_rel_permittivity), dtype=np.float64)
    )
    thermo_arr = (
        np.asarray(thermophoretic_coeff, dtype=np.float64).reshape(-1)
        if thermophoretic_coeff is not None
        else np.ones(n, dtype=np.float64)
    )
    vel = (
        np.asarray(velocity, dtype=np.float64)
        if velocity is not None
        else np.zeros((n, 2), dtype=np.float64)
    )

    rho_g = mu_g = temp_g = None
    if bool(pipeline.need_gas_properties):
        rho_g, mu_g, temp_g = sample_compiled_gas_properties_vectors(
            backend,
            2,
            float(t_eval),
            points,
            fallback_density_kgm3=float(gas_density_kgm3),
            fallback_mu_pas=float(gas_mu_pas),
            fallback_temperature_K=float(gas_temperature_K),
        )

    fluid_acceleration = None
    if (
        bool(params.pressure_gradient_enabled)
        and backend.fluid_accel_x is not None
        and backend.fluid_accel_y is not None
    ):
        fluid_acceleration = np.column_stack(
            (
                _sample_regular_time_grid_points_2d(backend.fluid_accel_x, axes, times, float(t_eval), points),
                _sample_regular_time_grid_points_2d(backend.fluid_accel_y, axes, times, float(t_eval), points),
            )
        ).astype(np.float64, copy=False)

    flow_time_derivative = None
    flow_velocity_gradient = None
    if (
        bool(params.virtual_mass_enabled)
        and backend.du_dt_x is not None
        and backend.du_dt_y is not None
        and backend.grad_ux_x is not None
        and backend.grad_ux_y is not None
        and backend.grad_uy_x is not None
        and backend.grad_uy_y is not None
    ):
        flow_time_derivative = np.column_stack(
            (
                _sample_regular_time_grid_points_2d(backend.du_dt_x, axes, times, float(t_eval), points),
                _sample_regular_time_grid_points_2d(backend.du_dt_y, axes, times, float(t_eval), points),
            )
        ).astype(np.float64, copy=False)
        flow_velocity_gradient = np.zeros((n, 2, 2), dtype=np.float64)
        flow_velocity_gradient[:, 0, 0] = _sample_regular_time_grid_points_2d(
            backend.grad_ux_x,
            axes,
            times,
            float(t_eval),
            points,
        )
        flow_velocity_gradient[:, 0, 1] = _sample_regular_time_grid_points_2d(
            backend.grad_ux_y,
            axes,
            times,
            float(t_eval),
            points,
        )
        flow_velocity_gradient[:, 1, 0] = _sample_regular_time_grid_points_2d(
            backend.grad_uy_x,
            axes,
            times,
            float(t_eval),
            points,
        )
        flow_velocity_gradient[:, 1, 1] = _sample_regular_time_grid_points_2d(
            backend.grad_uy_y,
            axes,
            times,
            float(t_eval),
            points,
        )

    grad_T = None
    if bool(params.thermophoresis_enabled) and backend.grad_T_x is not None and backend.grad_T_y is not None:
        grad_T = np.column_stack(
            (
                _sample_regular_time_grid_points_2d(backend.grad_T_x, axes, times, float(t_eval), points),
                _sample_regular_time_grid_points_2d(backend.grad_T_y, axes, times, float(t_eval), points),
            )
        ).astype(np.float64, copy=False)

    grad_E2 = None
    if bool(params.dielectrophoresis_enabled) and backend.grad_E2_x is not None and backend.grad_E2_y is not None:
        grad_E2 = np.column_stack(
            (
                _sample_regular_time_grid_points_2d(backend.grad_E2_x, axes, times, float(t_eval), points),
                _sample_regular_time_grid_points_2d(backend.grad_E2_y, axes, times, float(t_eval), points),
            )
        ).astype(np.float64, copy=False)

    flow = None
    vorticity_z = None
    if bool(params.lift_enabled) and backend.vorticity_z is not None:
        flow = sample_compiled_flow_vectors(backend, 2, float(t_eval), points)
        vorticity_z = _sample_regular_time_grid_points_2d(backend.vorticity_z, axes, times, float(t_eval), points)

    out = np.zeros((n, 2), dtype=np.float64)
    evaluate_force_pipeline(
        out,
        ForceBatchStatic(
            particle_diameter=d,
            particle_density=rho_p,
            particle_mass=mass,
            dep_particle_rel_permittivity=epsp_arr,
            thermophoretic_coeff=thermo_arr,
        ),
        ForceBatchState(velocity=vel),
        None,
        ForceBatchSamples(
            electric_field=electric_field,
            flow_velocity=flow,
            gas_density=rho_g,
            gas_mu=mu_g,
            gas_temperature=temp_g,
            grad_T=grad_T,
            grad_E2=grad_E2,
            vorticity_z=vorticity_z,
            fluid_acceleration=fluid_acceleration,
            flow_time_derivative=flow_time_derivative,
            flow_velocity_gradient=flow_velocity_gradient,
            electric_q_over_m=qom,
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        ),
        pipeline,
        float(t_eval),
    )
    return out.astype(np.float64, copy=False)


def sample_compiled_acceleration_vectors(
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
    *,
    electric_q_over_m: Optional[np.ndarray] = None,
    force_runtime: ForceRuntimeParameters | None = None,
    particle_diameter: Optional[np.ndarray] = None,
    particle_density: Optional[np.ndarray] = None,
    particle_mass: Optional[np.ndarray] = None,
    dep_particle_rel_permittivity: Optional[np.ndarray] = None,
    thermophoretic_coeff: Optional[np.ndarray] = None,
    velocity: Optional[np.ndarray] = None,
    gas_density_kgm3: float = 1.0,
    gas_mu_pas: float = 1.8e-5,
    gas_temperature_K: float = 300.0,
    gas_molecular_mass_kg: float = 60.0 * _AMU_KG,
) -> np.ndarray:
    backend = coerce_compiled_backend(compiled)
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError('positions must have shape (n, spatial_dim)')
    if pts.shape[0] == 0:
        return np.zeros((0, int(spatial_dim)), dtype=np.float64)
    if int(spatial_dim) == 2 and isinstance(backend, RegularRectilinearCompiledBackend):
        return _sample_regular_2d_acceleration_vectors_pipeline(
            backend,
            float(t_eval),
            pts,
            electric_q_over_m=electric_q_over_m,
            force_runtime=force_runtime,
            particle_diameter=particle_diameter,
            particle_density=particle_density,
            particle_mass=particle_mass,
            dep_particle_rel_permittivity=dep_particle_rel_permittivity,
            thermophoretic_coeff=thermophoretic_coeff,
            velocity=velocity,
            gas_density_kgm3=float(gas_density_kgm3),
            gas_mu_pas=float(gas_mu_pas),
            gas_temperature_K=float(gas_temperature_K),
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        )
    return np.asarray(
        [
            sample_compiled_acceleration_vector(
                backend,
                int(spatial_dim),
                float(t_eval),
                point,
                electric_q_over_m=(
                    None
                    if electric_q_over_m is None
                    else float(np.asarray(electric_q_over_m, dtype=np.float64).reshape(-1)[idx])
                ),
                force_runtime=force_runtime,
                particle_diameter=(
                    0.0 if particle_diameter is None else float(np.asarray(particle_diameter, dtype=np.float64).reshape(-1)[idx])
                ),
                particle_density=(
                    0.0 if particle_density is None else float(np.asarray(particle_density, dtype=np.float64).reshape(-1)[idx])
                ),
                particle_mass=(
                    None if particle_mass is None else float(np.asarray(particle_mass, dtype=np.float64).reshape(-1)[idx])
                ),
                dep_particle_rel_permittivity=(
                    float("nan")
                    if dep_particle_rel_permittivity is None
                    else float(np.asarray(dep_particle_rel_permittivity, dtype=np.float64).reshape(-1)[idx])
                ),
                thermophoretic_coeff=(
                    float("nan")
                    if thermophoretic_coeff is None
                    else float(np.asarray(thermophoretic_coeff, dtype=np.float64).reshape(-1)[idx])
                ),
                velocity=(None if velocity is None else np.asarray(velocity, dtype=np.float64)[idx]),
                gas_density_kgm3=float(gas_density_kgm3),
                gas_mu_pas=float(gas_mu_pas),
                gas_temperature_K=float(gas_temperature_K),
                gas_molecular_mass_kg=float(gas_molecular_mass_kg),
            )
            for idx, point in enumerate(pts)
        ],
        dtype=np.float64,
    )


def sample_compiled_valid_mask_status(compiled: CompiledRuntimeBackendLike, position: np.ndarray) -> int:
    backend = coerce_compiled_backend(compiled)
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        return int(sample_triangle_mesh_status(backend.field, np.asarray(position, dtype=np.float64)))
    return int(
        sample_valid_mask_status(
            np.asarray(backend.valid_mask, dtype=bool),
            backend.axes,
            np.asarray(position, dtype=np.float64),
        )
    )


def sample_compiled_valid_mask_statuses(compiled: CompiledRuntimeBackendLike, positions: np.ndarray) -> np.ndarray:
    backend = coerce_compiled_backend(compiled)
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError('positions must have shape (n, spatial_dim)')
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=np.uint8)
    if isinstance(backend, RegularRectilinearCompiledBackend) and len(backend.axes) == 2:
        axes = backend.axes
        mask = np.asarray(backend.valid_mask, dtype=bool)
        inside_axes = _regular_points_inside_axes_2d(axes, pts)
        ix0, ix1, _ax = _axis_intervals(axes[0], pts[:, 0])
        iy0, iy1, _ay = _axis_intervals(axes[1], pts[:, 1])
        point_values = _sample_regular_grid_points_2d(mask.astype(np.float64), axes, pts)
        point_valid = inside_axes & (point_values >= 0.5)
        stencil_invalid = (
            (~inside_axes)
            | (~mask[ix0, iy0])
            | (~mask[ix1, iy0])
            | (~mask[ix0, iy1])
            | (~mask[ix1, iy1])
        )
        statuses = np.full(pts.shape[0], int(VALID_MASK_STATUS_CLEAN), dtype=np.uint8)
        statuses[stencil_invalid] = np.uint8(VALID_MASK_STATUS_MIXED_STENCIL)
        statuses[~point_valid] = np.uint8(VALID_MASK_STATUS_HARD_INVALID)
        return statuses
    return np.asarray(
        [sample_compiled_valid_mask_status(backend, point) for point in pts],
        dtype=np.uint8,
    )


def sample_compiled_field_sample(
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_eval: float,
    position: np.ndarray,
    *,
    need_flow: bool = False,
    need_acceleration: bool = False,
    need_gas_properties: bool = False,
    need_valid_mask: bool = False,
    fallback_density_kgm3: float = 1.0,
    fallback_mu_pas: float = 1.8e-5,
    fallback_temperature_K: float = 300.0,
    electric_q_over_m: Optional[float] = None,
    force_runtime: ForceRuntimeParameters | None = None,
    particle_diameter: float = 0.0,
    particle_density: float = 0.0,
    particle_mass: Optional[float] = None,
    dep_particle_rel_permittivity: float = float("nan"),
    thermophoretic_coeff: float = float("nan"),
    velocity: Optional[np.ndarray] = None,
) -> FieldSample:
    """Collect fields at one point without changing provider semantics.

    It calls the existing scalar sampling functions and therefore preserves the
    same clean/mixed/hard-invalid and scalar fallback behavior.
    """

    pos = np.asarray(position, dtype=np.float64)
    flow = (
        sample_compiled_flow_vector(compiled, int(spatial_dim), float(t_eval), pos)
        if bool(need_flow)
        else None
    )
    rho = mu = temp = None
    if bool(need_gas_properties):
        rho, mu, temp = sample_compiled_gas_properties(
            compiled,
            float(t_eval),
            pos,
            fallback_density_kgm3=float(fallback_density_kgm3),
            fallback_mu_pas=float(fallback_mu_pas),
            fallback_temperature_K=float(fallback_temperature_K),
        )
    acceleration = (
        sample_compiled_acceleration_vector(
            compiled,
            int(spatial_dim),
            float(t_eval),
            pos,
            electric_q_over_m=electric_q_over_m,
            force_runtime=force_runtime,
            particle_diameter=float(particle_diameter),
            particle_density=float(particle_density),
            particle_mass=particle_mass,
            dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
            thermophoretic_coeff=float(thermophoretic_coeff),
            velocity=velocity,
            flow_velocity=flow,
            gas_density_kgm3=float(fallback_density_kgm3 if rho is None else rho),
            gas_mu_pas=float(fallback_mu_pas if mu is None else mu),
            gas_temperature_K=float(fallback_temperature_K if temp is None else temp),
        )
        if bool(need_acceleration)
        else None
    )
    valid_mask_status = sample_compiled_valid_mask_status(compiled, pos) if bool(need_valid_mask) else None
    return FieldSample(
        position=pos.copy(),
        time_s=float(t_eval),
        spatial_dim=int(spatial_dim),
        flow_velocity=flow,
        acceleration=acceleration,
        gas_density_kgm3=None if rho is None else float(rho),
        gas_mu_pas=None if mu is None else float(mu),
        gas_temperature_K=None if temp is None else float(temp),
        valid_mask_status=None if valid_mask_status is None else int(valid_mask_status),
    )


__all__ = (
    'CompiledRuntimeBackend',
    'CompiledRuntimeBackendLike',
    'FieldSample',
    'RegularRectilinearCompiledBackend',
    'TriangleMesh2DCompiledBackend',
    'coerce_compiled_backend',
    'compiled_gas_property_report',
    'compile_runtime_backend',
    'sample_compiled_acceleration_vector',
    'sample_compiled_acceleration_vectors',
    'sample_compiled_gas_properties',
    'sample_compiled_gas_properties_vectors',
    'sample_compiled_flow_vector',
    'sample_compiled_flow_vectors',
    'sample_compiled_field_sample',
    'sample_compiled_valid_mask_status',
    'sample_compiled_valid_mask_statuses',
)
