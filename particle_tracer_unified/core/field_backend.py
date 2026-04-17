from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from .datamodel import FieldProviderND, TriangleMeshField2D
from .field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    sample_quantity_series,
    sample_valid_mask,
    sample_valid_mask_status,
)
from .triangle_mesh_sampling_2d import sample_triangle_mesh_series, sample_triangle_mesh_status


FIELD_BACKEND_RECTILINEAR = 'regular_rectilinear'
FIELD_BACKEND_TRIANGLE_MESH_2D = 'triangle_mesh_2d'


def field_backend_kind(field_provider: FieldProviderND | None) -> str:
    if field_provider is None:
        return ''
    field = field_provider.field
    if isinstance(field, TriangleMeshField2D):
        return FIELD_BACKEND_TRIANGLE_MESH_2D
    return str(getattr(field, 'metadata', {}).get('field_backend_kind', FIELD_BACKEND_RECTILINEAR))


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


def _triangle_mesh_field_support_report(field: TriangleMeshField2D) -> Dict[str, Any]:
    return {
        'mesh_vertex_count': int(field.mesh_vertices.shape[0]),
        'mesh_triangle_count': int(field.mesh_triangles.shape[0]),
        'accel_grid_shape': [int(v) for v in field.accel_shape],
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
    report: Dict[str, Any] = {
        'field_backend_kind': str(field_backend_kind(field_provider)),
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
