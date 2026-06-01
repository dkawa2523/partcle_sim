from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import MutableMapping

import numpy as np

from ..core.field_sampling import sample_time_grid_scalar
from ..core.triangle_mesh_sampling_2d import sample_triangle_mesh_series
from .compiled_field_backend import (
    CompiledRuntimeBackendLike,
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
    coerce_compiled_backend,
    sample_compiled_field_sample,
    sample_compiled_flow_vectors,
    sample_compiled_gas_properties_vectors,
    sample_compiled_valid_mask_statuses,
)


@dataclass(frozen=True)
class FieldSamples:
    """Stage-local field samples collected through existing backend samplers."""

    flow: np.ndarray | None = None
    electric: np.ndarray | None = None
    gas_density: np.ndarray | None = None
    gas_mu: np.ndarray | None = None
    gas_temperature: np.ndarray | None = None
    valid_mask_status: np.ndarray | None = None
    call_count: int = 0
    point_count: int = 0


@dataclass(frozen=True)
class TimedFieldSamples:
    samples: FieldSamples
    elapsed_s: float


def _plan_flag(field_plan: object | None, name: str, default: bool) -> bool:
    if field_plan is None:
        return bool(default)
    return bool(getattr(field_plan, name, default))


def _resolve_flag(value: bool | None, field_plan: object | None, name: str, default: bool) -> bool:
    if value is None:
        return _plan_flag(field_plan, name, default)
    return bool(value)


def _sample_electric_vectors(
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray | None:
    backend = coerce_compiled_backend(compiled)
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError('positions must have shape (n, spatial_dim)')
    dim = int(spatial_dim)
    if pts.shape[0] == 0:
        return np.zeros((0, dim), dtype=np.float64)
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        names = tuple(getattr(backend, 'electric_field_names', ()))
        if len(names) < 2:
            return None
        field = backend.field
        values = [
            [
                float(sample_triangle_mesh_series(field.quantities[names[0]], field, point, float(t_eval))),
                float(sample_triangle_mesh_series(field.quantities[names[1]], field, point, float(t_eval))),
            ]
            for point in pts
        ]
        return np.asarray(values, dtype=np.float64)

    if not isinstance(backend, RegularRectilinearCompiledBackend):
        return None
    if backend.electric_x is None or backend.electric_y is None:
        return None
    axes = backend.axes
    times = np.asarray(backend.times, dtype=np.float64)
    values: list[list[float]] = []
    for point in pts:
        row = [
            float(sample_time_grid_scalar(backend.electric_x, axes, times, float(t_eval), point)),
            float(sample_time_grid_scalar(backend.electric_y, axes, times, float(t_eval), point)),
        ]
        if dim == 3:
            if backend.electric_z is None:
                return None
            row.append(float(sample_time_grid_scalar(backend.electric_z, axes, times, float(t_eval), point)))
        values.append(row)
    return np.asarray(values, dtype=np.float64)


def sample_fields_for_stage(
    compiled: CompiledRuntimeBackendLike,
    field_plan: object | None,
    points: np.ndarray,
    time_s: float,
    *,
    spatial_dim: int,
    need_flow: bool | None = None,
    need_electric: bool | None = None,
    need_gas_properties: bool | None = None,
    need_valid_mask: bool | None = None,
    fallback_density_kgm3: float = 1.0,
    fallback_mu_pas: float = 1.8e-5,
    fallback_temperature_K: float = 300.0,
) -> FieldSamples:
    """Sample a stage bundle without changing provider/backend semantics."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError('points must have shape (n, spatial_dim)')
    dim = int(spatial_dim)
    if pts.shape[1] < dim:
        raise ValueError('points second dimension must be at least spatial_dim')

    flow = None
    electric = None
    gas_density = None
    gas_mu = None
    gas_temperature = None
    valid_mask_status = None
    call_count = 0
    point_count = 0

    want_flow = _resolve_flag(need_flow, field_plan, 'need_flow', False)
    want_electric = _resolve_flag(need_electric, field_plan, 'need_electric', False)
    want_gas = (
        bool(need_gas_properties)
        if need_gas_properties is not None
        else bool(getattr(field_plan, 'needs_gas_properties', False)) if field_plan is not None else False
    )
    want_valid_mask = _resolve_flag(need_valid_mask, field_plan, 'need_valid_mask', False)

    if want_flow:
        flow = sample_compiled_flow_vectors(compiled, dim, float(time_s), pts)
        call_count += 1
    if want_electric:
        electric = _sample_electric_vectors(compiled, dim, float(time_s), pts)
        if electric is not None:
            call_count += 1
    if want_gas:
        gas_density, gas_mu, gas_temperature = sample_compiled_gas_properties_vectors(
            compiled,
            dim,
            float(time_s),
            pts,
            fallback_density_kgm3=float(fallback_density_kgm3),
            fallback_mu_pas=float(fallback_mu_pas),
            fallback_temperature_K=float(fallback_temperature_K),
        )
        call_count += 1
    if want_valid_mask:
        valid_mask_status = sample_compiled_valid_mask_statuses(compiled, pts)
        call_count += 1

    if call_count:
        point_count = int(pts.shape[0]) * int(call_count)
    return FieldSamples(
        flow=flow,
        electric=electric,
        gas_density=gas_density,
        gas_mu=gas_mu,
        gas_temperature=gas_temperature,
        valid_mask_status=valid_mask_status,
        call_count=int(call_count),
        point_count=int(point_count),
    )


def sample_scalar_fields_for_stage(
    compiled: CompiledRuntimeBackendLike,
    field_plan: object | None,
    position: np.ndarray,
    time_s: float,
    *,
    spatial_dim: int,
    need_flow: bool | None = None,
    need_gas_properties: bool | None = None,
    need_valid_mask: bool | None = None,
    fallback_density_kgm3: float = 1.0,
    fallback_mu_pas: float = 1.8e-5,
    fallback_temperature_K: float = 300.0,
) -> FieldSamples:
    """Scalar stage bundle that preserves existing scalar sampler behavior."""

    pos = np.asarray(position, dtype=np.float64)
    want_flow = _resolve_flag(need_flow, field_plan, 'need_flow', False)
    want_gas = (
        bool(need_gas_properties)
        if need_gas_properties is not None
        else bool(getattr(field_plan, 'needs_gas_properties', False)) if field_plan is not None else False
    )
    want_valid_mask = _resolve_flag(need_valid_mask, field_plan, 'need_valid_mask', False)

    sample = sample_compiled_field_sample(
        compiled,
        int(spatial_dim),
        float(time_s),
        pos,
        need_flow=bool(want_flow),
        need_acceleration=False,
        need_gas_properties=bool(want_gas),
        need_valid_mask=bool(want_valid_mask),
        fallback_density_kgm3=float(fallback_density_kgm3),
        fallback_mu_pas=float(fallback_mu_pas),
        fallback_temperature_K=float(fallback_temperature_K),
    )
    call_count = 1 if bool(want_flow or want_gas or want_valid_mask) else 0
    flow = None
    if sample.flow_velocity is not None:
        flow = np.asarray(sample.flow_velocity, dtype=np.float64).reshape(1, -1)
    gas_density = None if sample.gas_density_kgm3 is None else np.asarray([sample.gas_density_kgm3], dtype=np.float64)
    gas_mu = None if sample.gas_mu_pas is None else np.asarray([sample.gas_mu_pas], dtype=np.float64)
    gas_temperature = (
        None if sample.gas_temperature_K is None else np.asarray([sample.gas_temperature_K], dtype=np.float64)
    )
    valid_mask_status = (
        None if sample.valid_mask_status is None else np.asarray([sample.valid_mask_status], dtype=np.uint8)
    )
    return FieldSamples(
        flow=flow,
        electric=None,
        gas_density=gas_density,
        gas_mu=gas_mu,
        gas_temperature=gas_temperature,
        valid_mask_status=valid_mask_status,
        call_count=int(call_count),
        point_count=int(call_count),
    )


def timed_sample_fields_for_stage(*args, **kwargs) -> TimedFieldSamples:
    start = perf_counter()
    samples = sample_fields_for_stage(*args, **kwargs)
    return TimedFieldSamples(samples=samples, elapsed_s=float(perf_counter() - start))


def record_field_sampling_diagnostics(
    diagnostics: MutableMapping[str, object],
    samples: FieldSamples,
    elapsed_s: float,
) -> None:
    diagnostics['field_sampling_s'] = float(diagnostics.get('field_sampling_s', 0.0)) + float(elapsed_s)
    diagnostics['field_sample_point_count'] = int(diagnostics.get('field_sample_point_count', 0)) + int(
        samples.point_count
    )
    diagnostics['field_sample_call_count'] = int(diagnostics.get('field_sample_call_count', 0)) + int(
        samples.call_count
    )


__all__ = (
    'FieldSamples',
    'TimedFieldSamples',
    'record_field_sampling_diagnostics',
    'sample_fields_for_stage',
    'sample_scalar_fields_for_stage',
    'timed_sample_fields_for_stage',
)
