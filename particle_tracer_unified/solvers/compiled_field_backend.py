from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from ..core.datamodel import TriangleMeshField2D
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
from ..core.triangle_mesh_sampling_2d import sample_triangle_mesh_series, sample_triangle_mesh_status
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


@dataclass(frozen=True, slots=True)
class TriangleMesh2DCompiledBackend:
    field: TriangleMeshField2D
    velocity_names: Tuple[str, ...]
    times: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
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


CompiledRuntimeBackend = RegularRectilinearCompiledBackend | TriangleMesh2DCompiledBackend
CompiledRuntimeBackendLike = CompiledRuntimeBackend | Mapping[str, object]


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
    if runtime.field_provider is not None:
        field = runtime.field_provider.field
        need_electric_field = bool(enable_electric) or bool(force_params.dielectrophoresis_enabled)
        electric_names = choose_electric_field_quantity_names(field, spatial_dim) if bool(need_electric_field) else ()
        if isinstance(field, TriangleMeshField2D):
            if (
                bool(force_params.thermophoresis_enabled)
                or bool(force_params.dielectrophoresis_enabled)
                or bool(force_params.lift_enabled)
                or bool(force_params.gravity_buoyancy_enabled)
            ):
                raise ValueError('solver.forces thermophoresis/dielectrophoresis/lift/buoyancy require the regular rectilinear field backend')
            names = choose_velocity_quantity_names(field, spatial_dim)
            time_quantity_names = tuple(names)
            if time_quantity_names:
                times = _common_quantity_times(field, time_quantity_names)
            vertex_shape = (1, int(field.mesh_vertices.shape[0]))
            ux_mesh = np.zeros(vertex_shape, dtype=np.float64)
            uy_mesh = np.zeros(vertex_shape, dtype=np.float64)
            if names:
                ux_mesh = np.asarray(field.quantities[names[0]].data, dtype=np.float64)
                uy_mesh = np.asarray(field.quantities[names[1]].data, dtype=np.float64)
            if electric_names:
                electric_field_names = tuple(electric_names)
            time_vertex_shape = (int(max(1, times.size)), int(field.mesh_vertices.shape[0]))
            if not names:
                ux_mesh = np.zeros(time_vertex_shape, dtype=np.float64)
                uy_mesh = np.zeros(time_vertex_shape, dtype=np.float64)
            return TriangleMesh2DCompiledBackend(
                field=field,
                velocity_names=tuple(names),
                times=times,
                ux=ux_mesh,
                uy=uy_mesh,
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
                gas_density_source='scalar_fallback',
                gas_mu_source='scalar_fallback',
                gas_temperature_source='scalar_fallback',
            )
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
    )


def _positive_grid_stats(values: np.ndarray, valid_mask: np.ndarray) -> Mapping[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    grid = arr[0] if arr.ndim > valid_mask.ndim else arr
    mask = np.asarray(valid_mask, dtype=bool)
    if grid.shape != mask.shape:
        finite = np.isfinite(grid) & (grid > 0.0)
    else:
        finite = mask & np.isfinite(grid) & (grid > 0.0)
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
        'density_used_by_drag_model': int(drag_model in {'epstein', 'schiller_naumann'}),
        'dynamic_viscosity_used_by_drag_model': int(drag_model == 'schiller_naumann'),
        'temperature_used_by_drag_model': int(drag_model == 'epstein'),
    }
    if isinstance(backend, RegularRectilinearCompiledBackend):
        mask = np.asarray(backend.core_valid_mask, dtype=bool)
        report['density_field_stats'] = dict(_positive_grid_stats(backend.gas_density, mask))
        report['dynamic_viscosity_field_stats'] = dict(_positive_grid_stats(backend.gas_mu, mask))
        report['temperature_field_stats'] = dict(_positive_grid_stats(backend.gas_temperature, mask))
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
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
) -> np.ndarray:
    dim = int(np.asarray(velocity, dtype=np.float64).size)
    out = np.zeros(dim, dtype=np.float64)
    d = max(float(diameter), 0.0)
    radius = 0.5 * d
    m = _particle_mass_from_inputs(d, float(density), mass)
    if d <= 0.0 or radius <= 0.0 or m <= 0.0:
        return out
    rho_g = max(float(gas_density_kgm3), 0.0)
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
        return np.zeros(spatial_dim, dtype=np.float64)
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
    ):
        grad_T = np.zeros(int(spatial_dim), dtype=np.float64)
        grad_E2 = np.zeros(int(spatial_dim), dtype=np.float64)
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
        flow = (
            np.asarray(flow_velocity, dtype=np.float64)
            if flow_velocity is not None
            else sample_compiled_flow_vector(backend, int(spatial_dim), float(t_eval), pos)
        )
        vel = np.zeros(int(spatial_dim), dtype=np.float64) if velocity is None else np.asarray(velocity, dtype=np.float64)
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
            gas_density_kgm3=float(gas_density_kgm3),
            gas_mu_pas=float(gas_mu_pas),
            gas_temperature_K=float(gas_temperature_K),
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
        axes = backend.axes
        times = np.asarray(backend.times, dtype=np.float64)
        ax = np.zeros(pts.shape[0], dtype=np.float64)
        ay = np.zeros(pts.shape[0], dtype=np.float64)
        if electric_q_over_m is not None and backend.electric_x is not None and backend.electric_y is not None:
            qom = np.asarray(electric_q_over_m, dtype=np.float64).reshape(-1)
            if qom.shape[0] != pts.shape[0]:
                raise ValueError('electric_q_over_m must match positions length')
            ex = _sample_regular_time_grid_points_2d(backend.electric_x, axes, times, float(t_eval), pts)
            ey = _sample_regular_time_grid_points_2d(backend.electric_y, axes, times, float(t_eval), pts)
            ax = ax + qom * ex
            ay = ay + qom * ey
        params = force_runtime or ForceRuntimeParameters()
        if bool(params.thermophoresis_enabled) or bool(params.dielectrophoresis_enabled) or bool(params.lift_enabled):
            n = int(pts.shape[0])
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
            rho_g, mu_g, temp_g = sample_compiled_gas_properties_vectors(
                backend,
                2,
                float(t_eval),
                pts,
                fallback_density_kgm3=float(gas_density_kgm3),
                fallback_mu_pas=float(gas_mu_pas),
                fallback_temperature_K=float(gas_temperature_K),
            )
            finite_mass = np.isfinite(mass) & (mass > 0.0)
            radius = 0.5 * np.maximum(d, 0.0)
            if bool(params.thermophoresis_enabled) and backend.grad_T_x is not None and backend.grad_T_y is not None:
                grad_tx = _sample_regular_time_grid_points_2d(backend.grad_T_x, axes, times, float(t_eval), pts)
                grad_ty = _sample_regular_time_grid_points_2d(backend.grad_T_y, axes, times, float(t_eval), pts)
                mol_mass = max(float(gas_molecular_mass_kg), 1.0e-30)
                mean_free_path = (mu_g / np.maximum(rho_g, 1.0e-30)) * np.sqrt(
                    np.pi * mol_mass / (2.0 * _K_BOLTZMANN * np.maximum(temp_g, 1.0))
                )
                kn = mean_free_path / np.maximum(radius, 1.0e-30)
                if str(params.thermophoresis_model).lower() == "continuum":
                    kn = np.zeros_like(kn)
                ratio = max(float(params.gas_thermal_conductivity_W_mK), 1.0e-30) / max(
                    float(params.particle_thermal_conductivity_W_mK),
                    1.0e-30,
                )
                factor = (
                    float(params.thermophoresis_Cs)
                    * (ratio + float(params.thermophoresis_Ct) * kn)
                    / np.maximum(
                        (1.0 + 3.0 * float(params.thermophoresis_Cm) * kn)
                        * (1.0 + 2.0 * ratio + 2.0 * float(params.thermophoresis_Ct) * kn),
                        1.0e-30,
                    )
                )
                multiplier = np.where(np.isfinite(thermo_arr) & (thermo_arr > 0.0), thermo_arr, 1.0)
                tau_stokes = mass / np.maximum(3.0 * np.pi * mu_g * np.maximum(d, 1.0e-30), 1.0e-300)
                scale = -multiplier * factor * mu_g / np.maximum(rho_g * temp_g * tau_stokes, 1.0e-300)
                valid = finite_mass & np.isfinite(scale)
                ax = ax + np.where(valid, scale * grad_tx, 0.0)
                ay = ay + np.where(valid, scale * grad_ty, 0.0)
            if bool(params.dielectrophoresis_enabled) and backend.grad_E2_x is not None and backend.grad_E2_y is not None:
                grad_e2x = _sample_regular_time_grid_points_2d(backend.grad_E2_x, axes, times, float(t_eval), pts)
                grad_e2y = _sample_regular_time_grid_points_2d(backend.grad_E2_y, axes, times, float(t_eval), pts)
                epsp = np.where(
                    np.isfinite(epsp_arr) & (epsp_arr > 0.0),
                    epsp_arr,
                    float(params.dep_particle_rel_permittivity),
                )
                epsp = np.where(np.isfinite(epsp) & (epsp > 0.0), epsp, 2.0)
                epsm = max(float(params.dep_medium_rel_permittivity), 1.0e-30)
                if float(params.dep_frequency_Hz) <= 0.0:
                    cm_real = (epsp - epsm) / (epsp + 2.0 * epsm)
                else:
                    cm_real = np.asarray(
                        [
                            _cm_factor_real(
                                float(value),
                                epsm,
                                float(params.dep_particle_conductivity_Sm),
                                float(params.dep_medium_conductivity_Sm),
                                float(params.dep_frequency_Hz),
                            )
                            for value in epsp
                        ],
                        dtype=np.float64,
                    )
                coeff = 2.0 * np.pi * _EPS0_F_M * epsm * radius**3 * cm_real / np.maximum(mass, 1.0e-300)
                valid = finite_mass & np.isfinite(coeff)
                ax = ax + np.where(valid, coeff * grad_e2x, 0.0)
                ay = ay + np.where(valid, coeff * grad_e2y, 0.0)
            if bool(params.lift_enabled) and backend.vorticity_z is not None:
                flow = sample_compiled_flow_vectors(backend, 2, float(t_eval), pts)
                omega = _sample_regular_time_grid_points_2d(backend.vorticity_z, axes, times, float(t_eval), pts)
                omega_abs = np.abs(omega)
                slip = vel[:, :2] - flow[:, :2]
                nu = mu_g / np.maximum(rho_g, 1.0e-30)
                scale = (
                    float(params.lift_coefficient)
                    * mu_g
                    * radius
                    * radius
                    / np.maximum(np.sqrt(nu * omega_abs), 1.0e-300)
                    / np.maximum(mass, 1.0e-300)
                )
                valid = finite_mass & np.isfinite(scale) & (omega_abs > 1.0e-30)
                ax = ax + np.where(valid, scale * slip[:, 1] * omega, 0.0)
                ay = ay + np.where(valid, -scale * slip[:, 0] * omega, 0.0)
        return np.column_stack((ax, ay)).astype(np.float64, copy=False)
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


__all__ = (
    'CompiledRuntimeBackend',
    'CompiledRuntimeBackendLike',
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
    'sample_compiled_valid_mask_status',
    'sample_compiled_valid_mask_statuses',
)
