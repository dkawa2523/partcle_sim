from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .coordinate_systems import axis_names_for_coordinate_system
from .datamodel import FieldProviderND, TriangleMeshField2D
from .field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
    choose_electric_field_quantity_names,
    choose_velocity_quantity_names,
    point_within_axes,
    sample_quantity_series,
    sample_valid_mask,
    sample_valid_mask_status,
)
from .grid_sampling import locate_axis_interval
from .triangle_mesh_sampling_2d import (
    locate_triangle_containing_point,
    sample_triangle_mesh_series,
    sample_triangle_mesh_status,
)


FIELD_BACKEND_RECTILINEAR = 'regular_rectilinear'
FIELD_BACKEND_TRIANGLE_MESH_2D = 'triangle_mesh_2d'

FIELD_SAMPLE_STATUS_REASON = {
    int(VALID_MASK_STATUS_CLEAN): 'clean',
    int(VALID_MASK_STATUS_MIXED_STENCIL): 'mixed_stencil',
    int(VALID_MASK_STATUS_HARD_INVALID): 'hard_invalid',
}

TRIANGLE_DERIVED_QUANTITY_ALIASES = {
    'grad_T_x': ('grad_T_x', 'dT_dx', 'temperature_gradient_x', 'grad_temperature_x'),
    'grad_T_y': ('grad_T_y', 'dT_dy', 'temperature_gradient_y', 'grad_temperature_y'),
    'grad_E2_x': ('grad_E2_x', 'dE2_dx', 'grad_E_squared_x', 'grad_esq_x'),
    'grad_E2_y': ('grad_E2_y', 'dE2_dy', 'grad_E_squared_y', 'grad_esq_y'),
    'fluid_accel_x': ('fluid_accel_x', 'fluid_acceleration_x', 'material_accel_x', 'a_fluid_x'),
    'fluid_accel_y': ('fluid_accel_y', 'fluid_acceleration_y', 'material_accel_y', 'a_fluid_y'),
    'vorticity_z': ('vorticity_z', 'omega_z', 'curl_u_z'),
}


@dataclass(frozen=True)
class FieldSample:
    quantity_name: str
    value: Any
    valid: bool
    status: int
    reason: str
    provider_kind: str
    cell_id: int


def field_backend_kind(field_provider: FieldProviderND | None) -> str:
    if field_provider is None:
        return ''
    field = field_provider.field
    if isinstance(field, TriangleMeshField2D):
        return FIELD_BACKEND_TRIANGLE_MESH_2D
    return str(getattr(field, 'metadata', {}).get('field_backend_kind', FIELD_BACKEND_RECTILINEAR))


def _status_reason(status: int) -> str:
    return str(FIELD_SAMPLE_STATUS_REASON.get(int(status), 'unknown'))


def _regular_cell_id(field, position: np.ndarray) -> int:
    axes = tuple(getattr(field, 'axes', ()))
    if not point_within_axes(axes, np.asarray(position, dtype=np.float64)):
        return -1
    lows = []
    shape = []
    for axis_index, axis in enumerate(axes):
        lo, _hi, _alpha = locate_axis_interval(np.asarray(axis, dtype=np.float64), float(position[axis_index]))
        lows.append(int(lo))
        shape.append(max(int(np.asarray(axis).size) - 1, 1))
    cell_id = 0
    stride = 1
    for lo, count in reversed(list(zip(lows, shape))):
        cell_id += int(lo) * stride
        stride *= int(count)
    return int(cell_id)


def _triangle_cell_id(field: TriangleMeshField2D, position: np.ndarray) -> int:
    tri_idx, _bary = locate_triangle_containing_point(
        vertices=field.mesh_vertices,
        triangles=field.mesh_triangles,
        accel_origin=field.accel_origin,
        accel_cell_size=field.accel_cell_size,
        accel_shape=field.accel_shape,
        accel_cell_offsets=field.accel_cell_offsets,
        accel_triangle_indices=field.accel_triangle_indices,
        position=np.asarray(position, dtype=np.float64),
        eps=float(getattr(field, 'metadata', {}).get('support_tolerance_m', 2.0e-6)),
    )
    return int(tri_idx)


def sample_field_cell_id(field_provider: FieldProviderND, position: np.ndarray) -> int:
    field = field_provider.field
    pos = np.asarray(position, dtype=np.float64)
    if isinstance(field, TriangleMeshField2D):
        return _triangle_cell_id(field, pos)
    return _regular_cell_id(field, pos)


def _regular_field_support_report(field) -> Dict[str, Any]:
    valid_mask = np.asarray(field.valid_mask, dtype=bool)
    node_count = int(valid_mask.size)
    valid_count = int(np.count_nonzero(valid_mask))
    axes = []
    for axis in field.axes:
        arr = np.asarray(axis, dtype=np.float64)
        axes.append(
            {
                'count': int(arr.size),
                'min': float(np.nanmin(arr)) if arr.size else float('nan'),
                'max': float(np.nanmax(arr)) if arr.size else float('nan'),
            }
        )
    support_phi = getattr(field, 'support_phi', None)
    support_phi_summary: Dict[str, Any] = {'available': False}
    if support_phi is not None:
        phi = np.asarray(support_phi, dtype=np.float64)
        finite = phi[np.isfinite(phi)]
        support_phi_summary = {
            'available': True,
            'finite_count': int(finite.size),
            'min': float(np.min(finite)) if finite.size else float('nan'),
            'max': float(np.max(finite)) if finite.size else float('nan'),
        }
    return {
        'grid_shape': [int(v) for v in valid_mask.shape],
        'grid_node_count': node_count,
        'valid_node_count': valid_count,
        'invalid_node_count': int(node_count - valid_count),
        'valid_fraction': float(valid_count / node_count) if node_count else 0.0,
        'axes': axes,
        'support_phi': support_phi_summary,
    }


def triangle_derived_quantity_names(field: TriangleMeshField2D) -> Dict[str, str]:
    quantities = getattr(field, 'quantities', {})
    selected: Dict[str, str] = {}
    for target, aliases in TRIANGLE_DERIVED_QUANTITY_ALIASES.items():
        for name in aliases:
            if name in quantities:
                selected[str(target)] = str(name)
                break
    return selected


def triangle_mesh_gradient_source_report(field: TriangleMeshField2D) -> Dict[str, str]:
    quantities = getattr(field, 'quantities', {})
    quantity_names = set(quantities)
    names = triangle_derived_quantity_names(field)
    gas_names = {'T', 'temperature', 'temperature_K', 'gas_temperature'}
    electric_names = choose_electric_field_quantity_names(field, 2)
    has_velocity = len(choose_velocity_quantity_names(field, 2)) >= 2
    return {
        'grad_T': (
            'exported_quantity'
            if {'grad_T_x', 'grad_T_y'} <= set(names)
            else ('triangle_p1_fallback' if quantity_names.intersection(gas_names) else 'unavailable')
        ),
        'grad_E2': (
            'exported_quantity'
            if {'grad_E2_x', 'grad_E2_y'} <= set(names)
            else ('triangle_p1_fallback' if len(electric_names) >= 2 else 'unavailable')
        ),
        'fluid_acceleration': (
            'exported_quantity'
            if {'fluid_accel_x', 'fluid_accel_y'} <= set(names)
            else ('triangle_p1_fallback' if has_velocity else 'unavailable')
        ),
        'vorticity_z': 'exported_quantity' if 'vorticity_z' in names else ('triangle_p1_fallback' if has_velocity else 'unavailable'),
    }


def _triangle_mesh_field_support_report(field: TriangleMeshField2D) -> Dict[str, Any]:
    return {
        'mesh_vertex_count': int(field.mesh_vertices.shape[0]),
        'mesh_triangle_count': int(field.mesh_triangles.shape[0]),
        'accel_grid_shape': [int(v) for v in field.accel_shape],
        'triangle_gradient_sources': triangle_mesh_gradient_source_report(field),
    }


def _field_time_axis_report(field) -> Dict[str, Any]:
    quantities = getattr(field, 'quantities', {})
    reference_name = ''
    reference_times: np.ndarray | None = None
    mismatches = []
    for name in sorted(quantities.keys()):
        series = quantities[name]
        times = np.asarray(getattr(series, 'times', np.asarray([0.0], dtype=np.float64)), dtype=np.float64)
        if reference_times is None:
            reference_name = str(name)
            reference_times = times
            continue
        if times.shape != reference_times.shape or not np.allclose(times, reference_times, rtol=0.0, atol=0.0):
            mismatches.append(str(name))

    times = reference_times if reference_times is not None else np.asarray([0.0], dtype=np.float64)
    finite = times[np.isfinite(times)]
    return {
        'time_mode': str(getattr(field, 'time_mode', 'steady')),
        'time_count': int(times.size),
        'time_min_s': float(np.min(finite)) if finite.size else float('nan'),
        'time_max_s': float(np.max(finite)) if finite.size else float('nan'),
        'quantity_time_axis_reference': reference_name,
        'quantity_time_axis_mismatch_count': int(len(mismatches)),
        'quantity_time_axis_mismatches': mismatches[:20],
        'quantity_time_axis_mismatches_truncated': bool(len(mismatches) > 20),
    }


def field_backend_report(field_provider: FieldProviderND | None) -> Dict[str, Any]:
    if field_provider is None:
        return {
            'field_backend_kind': '',
            'field_has_support_phi': 0,
            'field_support_phi_kind': '',
        }
    field = field_provider.field
    metadata = getattr(field, 'metadata', {})
    spatial_dim = int(getattr(field, 'spatial_dim', 0))
    coordinate_system = str(getattr(field, 'coordinate_system', ''))
    raw_axis_names = getattr(field, 'axis_names', None)
    axis_names = (
        tuple(str(v) for v in raw_axis_names)
        if raw_axis_names is not None
        else axis_names_for_coordinate_system(coordinate_system, spatial_dim)
    )
    report: Dict[str, Any] = {
        'field_backend_kind': str(field_backend_kind(field_provider)),
        'spatial_dim': spatial_dim,
        'coordinate_system': coordinate_system,
        'axis_names': list(axis_names),
        'field_has_support_phi': int(getattr(field, 'support_phi', None) is not None),
        'field_support_phi_kind': str(metadata.get('field_support_phi_kind', '')),
        'quantity_count': int(len(getattr(field, 'quantities', {}))),
        'time_axis': _field_time_axis_report(field),
    }
    if isinstance(field, TriangleMeshField2D):
        report.update(_triangle_mesh_field_support_report(field))
    else:
        report.update(_regular_field_support_report(field))
    return report


def sample_field_valid_status(field_provider: FieldProviderND, position: np.ndarray, t_eval: float | None = None) -> int:
    del t_eval
    field = field_provider.field
    pos = np.asarray(position, dtype=np.float64)
    if isinstance(field, TriangleMeshField2D):
        return int(sample_triangle_mesh_status(field, pos))
    return int(sample_valid_mask_status(np.asarray(field.valid_mask, dtype=bool), field.axes, pos))


def sample_field_valid(field_provider: FieldProviderND, position: np.ndarray) -> bool:
    field = field_provider.field
    pos = np.asarray(position, dtype=np.float64)
    if isinstance(field, TriangleMeshField2D):
        return int(sample_triangle_mesh_status(field, pos)) == int(VALID_MASK_STATUS_CLEAN)
    return bool(sample_valid_mask(np.asarray(field.valid_mask, dtype=bool), field.axes, pos))


def sample_field_quantity(
    field_provider: FieldProviderND,
    quantity_name: str,
    position: np.ndarray,
    t_eval: float,
    *,
    mode: str = 'linear',
    default: float = np.nan,
) -> float:
    field = field_provider.field
    series = field.quantities.get(str(quantity_name))
    if series is None:
        return float(default)
    pos = np.asarray(position, dtype=np.float64)
    if isinstance(field, TriangleMeshField2D):
        value = sample_triangle_mesh_series(series, field, pos, float(t_eval), mode=mode)
        return float(default) if not np.isfinite(value) else float(value)
    if not bool(sample_valid_mask(np.asarray(field.valid_mask, dtype=bool), field.axes, pos)):
        return float(default)
    return float(sample_quantity_series(series, field.axes, pos, float(t_eval), mode=mode))


def sample_field_quantity_with_status(
    field_provider: FieldProviderND,
    quantity_name: str,
    position: np.ndarray,
    t_eval: float,
    *,
    mode: str = 'linear',
    default: float = np.nan,
) -> FieldSample:
    field = field_provider.field
    name = str(quantity_name)
    pos = np.asarray(position, dtype=np.float64)
    status = int(sample_field_valid_status(field_provider, pos, float(t_eval)))
    cell_id = int(sample_field_cell_id(field_provider, pos))
    provider_kind = str(field_backend_kind(field_provider))
    if name not in getattr(field, 'quantities', {}):
        return FieldSample(
            quantity_name=name,
            value=float(default),
            valid=False,
            status=int(status),
            reason='missing_quantity',
            provider_kind=provider_kind,
            cell_id=cell_id,
        )
    value = float(default)
    if status != int(VALID_MASK_STATUS_HARD_INVALID):
        value = float(
            sample_field_quantity(
                field_provider,
                name,
                pos,
                float(t_eval),
                mode=mode,
                default=default,
            )
        )
    return FieldSample(
        quantity_name=name,
        value=value,
        valid=bool(status == int(VALID_MASK_STATUS_CLEAN) and np.isfinite(value)),
        status=int(status),
        reason=_status_reason(status),
        provider_kind=provider_kind,
        cell_id=cell_id,
    )


def sample_field_vector(
    field_provider: FieldProviderND,
    component_names: Sequence[str],
    position: np.ndarray,
    t_eval: float,
    *,
    mode: str = 'linear',
    default: float = np.nan,
) -> np.ndarray:
    return np.asarray(
        [
            sample_field_quantity(
                field_provider,
                name,
                np.asarray(position, dtype=np.float64),
                float(t_eval),
                mode=mode,
                default=default,
            )
            for name in component_names
        ],
        dtype=np.float64,
    )


def sample_field_vector_with_status(
    field_provider: FieldProviderND,
    component_names: Sequence[str],
    position: np.ndarray,
    t_eval: float,
    *,
    mode: str = 'linear',
    default: float = np.nan,
) -> FieldSample:
    field = field_provider.field
    names = tuple(str(name) for name in component_names)
    pos = np.asarray(position, dtype=np.float64)
    status = int(sample_field_valid_status(field_provider, pos, float(t_eval)))
    cell_id = int(sample_field_cell_id(field_provider, pos))
    provider_kind = str(field_backend_kind(field_provider))
    missing = [name for name in names if name not in getattr(field, 'quantities', {})]
    if missing:
        return FieldSample(
            quantity_name=','.join(names),
            value=np.full(len(names), float(default), dtype=np.float64),
            valid=False,
            status=int(status),
            reason='missing_quantity:' + ','.join(missing),
            provider_kind=provider_kind,
            cell_id=cell_id,
        )
    values = np.full(len(names), float(default), dtype=np.float64)
    if status != int(VALID_MASK_STATUS_HARD_INVALID):
        values = sample_field_vector(field_provider, names, pos, float(t_eval), mode=mode, default=default)
    return FieldSample(
        quantity_name=','.join(names),
        value=np.asarray(values, dtype=np.float64),
        valid=bool(status == int(VALID_MASK_STATUS_CLEAN) and np.all(np.isfinite(values))),
        status=int(status),
        reason=_status_reason(status),
        provider_kind=provider_kind,
        cell_id=cell_id,
    )


__all__ = (
    'FIELD_BACKEND_RECTILINEAR',
    'FIELD_BACKEND_TRIANGLE_MESH_2D',
    'FIELD_SAMPLE_STATUS_REASON',
    'FieldSample',
    'field_backend_kind',
    'field_backend_report',
    'sample_field_cell_id',
    'sample_field_quantity',
    'sample_field_quantity_with_status',
    'sample_field_valid',
    'sample_field_valid_status',
    'sample_field_vector',
    'sample_field_vector_with_status',
    'triangle_derived_quantity_names',
    'triangle_mesh_gradient_source_report',
)
