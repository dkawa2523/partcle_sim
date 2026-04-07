from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from ..core.datamodel import TriangleMeshField2D
from ..core.field_sampling import (
    as_time_grid,
    choose_velocity_quantity_names,
    sample_time_grid_scalar,
    sample_valid_mask,
    sample_valid_mask_status,
)
from ..core.triangle_mesh_sampling_2d import sample_triangle_mesh_series, sample_triangle_mesh_status


@dataclass(frozen=True, slots=True)
class RegularRectilinearCompiledBackend:
    axes: Tuple[np.ndarray, ...]
    times: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    valid_mask: np.ndarray
    core_valid_mask: np.ndarray
    extension_band_mask: np.ndarray
    uz: Optional[np.ndarray] = None
    backend_kind: str = 'regular_rectilinear'


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


CompiledRuntimeBackend = RegularRectilinearCompiledBackend | TriangleMesh2DCompiledBackend
CompiledRuntimeBackendLike = CompiledRuntimeBackend | Mapping[str, object]


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
        )
    axes = tuple(np.asarray(ax, dtype=np.float64) for ax in compiled.get('axes', ()))
    valid_mask = np.asarray(compiled.get('valid_mask', np.zeros((0,), dtype=bool)), dtype=bool)
    core_valid_mask_raw = compiled.get('core_valid_mask')
    core_valid_mask = valid_mask if core_valid_mask_raw is None else np.asarray(core_valid_mask_raw, dtype=bool)
    extension_band_mask_raw = compiled.get('extension_band_mask')
    extension_band_mask = (
        np.zeros_like(valid_mask, dtype=bool)
        if extension_band_mask_raw is None
        else np.asarray(extension_band_mask_raw, dtype=bool)
    )
    uz_raw = compiled.get('uz')
    uz = None if uz_raw is None else np.asarray(uz_raw, dtype=np.float64)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=np.asarray(compiled.get('times', np.asarray([0.0], dtype=np.float64)), dtype=np.float64),
        ux=np.asarray(compiled.get('ux', np.zeros((1,) + valid_mask.shape, dtype=np.float64)), dtype=np.float64),
        uy=np.asarray(compiled.get('uy', np.zeros((1,) + valid_mask.shape, dtype=np.float64)), dtype=np.float64),
        valid_mask=valid_mask,
        core_valid_mask=core_valid_mask,
        extension_band_mask=extension_band_mask,
        uz=uz,
    )


def compile_runtime_backend(runtime, spatial_dim: int) -> CompiledRuntimeBackend:
    if runtime.geometry_provider is None:
        raise ValueError('High-fidelity solver requires geometry_provider')
    geom = runtime.geometry_provider.geometry
    axes = tuple(np.asarray(ax, dtype=np.float64) for ax in geom.axes)
    valid_mask = np.asarray(geom.valid_mask, dtype=bool)
    core_valid_mask = valid_mask
    extension_band_mask = np.zeros_like(valid_mask, dtype=bool)
    times = np.asarray([0.0], dtype=np.float64)
    shape = (1,) + tuple(len(ax) for ax in axes)
    ux = np.zeros(shape, dtype=np.float64)
    uy = np.zeros(shape, dtype=np.float64)
    uz = np.zeros(shape, dtype=np.float64) if spatial_dim == 3 else None
    if runtime.field_provider is not None:
        field = runtime.field_provider.field
        if isinstance(field, TriangleMeshField2D):
            names = choose_velocity_quantity_names(field, spatial_dim)
            if names:
                times = np.asarray(field.quantities[names[0]].times, dtype=np.float64)
                if times.size == 0:
                    times = np.asarray([0.0], dtype=np.float64)
                ux = np.asarray(field.quantities[names[0]].data, dtype=np.float64)
                uy = np.asarray(field.quantities[names[1]].data, dtype=np.float64)
            return TriangleMesh2DCompiledBackend(
                field=field,
                velocity_names=tuple(names),
                times=times,
                ux=ux,
                uy=uy,
                mesh_vertices=np.asarray(field.mesh_vertices, dtype=np.float64),
                mesh_triangles=np.asarray(field.mesh_triangles, dtype=np.int32),
                accel_origin=np.asarray(field.accel_origin, dtype=np.float64),
                accel_cell_size=np.asarray(field.accel_cell_size, dtype=np.float64),
                accel_shape=tuple(np.asarray(field.accel_shape, dtype=np.int32).tolist()),
                accel_cell_offsets=np.asarray(field.accel_cell_offsets, dtype=np.int32),
                accel_triangle_indices=np.asarray(field.accel_triangle_indices, dtype=np.int32),
                support_tolerance_m=float(field.metadata.get('support_tolerance_m', 2.0e-6)),
            )
        valid_mask = np.asarray(field.valid_mask, dtype=bool)
        core_valid_mask = np.asarray(
            field.core_valid_mask if field.core_valid_mask is not None else field.valid_mask,
            dtype=bool,
        )
        extension_band_mask = np.asarray(
            field.extension_band_mask if field.extension_band_mask is not None else np.zeros_like(valid_mask, dtype=bool),
            dtype=bool,
        )
        names = choose_velocity_quantity_names(field, spatial_dim)
        if names:
            times = np.asarray(field.quantities[names[0]].times, dtype=np.float64)
            if times.size == 0:
                times = np.asarray([0.0], dtype=np.float64)
            if spatial_dim == 2:
                ux = as_time_grid(field.quantities[names[0]].data, 2)
                uy = as_time_grid(field.quantities[names[1]].data, 2)
            else:
                ux = as_time_grid(field.quantities[names[0]].data, 3)
                uy = as_time_grid(field.quantities[names[1]].data, 3)
                uz = as_time_grid(field.quantities[names[2]].data, 3)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=times,
        ux=ux,
        uy=uy,
        valid_mask=valid_mask,
        core_valid_mask=core_valid_mask,
        extension_band_mask=extension_band_mask,
        uz=uz if spatial_dim == 3 else None,
    )


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
    if spatial_dim == 2:
        return np.asarray([ux, uy], dtype=np.float64)
    uz_grid = backend.uz if backend.uz is not None else np.zeros((1,) + tuple(len(ax) for ax in axes), dtype=np.float64)
    uz = float(sample_time_grid_scalar(uz_grid, axes, times, t_eval, pos))
    return np.asarray([ux, uy, uz], dtype=np.float64)


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


def sample_compiled_extension_band_active(compiled: CompiledRuntimeBackendLike, position: np.ndarray) -> bool:
    backend = coerce_compiled_backend(compiled)
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        return False
    pos = np.asarray(position, dtype=np.float64)
    effective_valid = bool(sample_valid_mask(np.asarray(backend.valid_mask, dtype=bool), backend.axes, pos))
    if not effective_valid:
        return False
    return not bool(sample_valid_mask(np.asarray(backend.core_valid_mask, dtype=bool), backend.axes, pos))


__all__ = (
    'CompiledRuntimeBackend',
    'CompiledRuntimeBackendLike',
    'RegularRectilinearCompiledBackend',
    'TriangleMesh2DCompiledBackend',
    'coerce_compiled_backend',
    'compile_runtime_backend',
    'sample_compiled_extension_band_active',
    'sample_compiled_flow_vector',
    'sample_compiled_valid_mask_status',
)
