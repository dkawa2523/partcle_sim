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


def field_backend_report(field_provider: FieldProviderND | None) -> Dict[str, Any]:
    if field_provider is None:
        return {
            'field_backend_kind': '',
            'field_regularization_mode': '',
            'field_regularization_band_distance_m': 0.0,
            'field_regularization_added_node_count': 0,
            'field_regularization_probe_success_count': 0,
            'field_regularization_probe_fallback_count': 0,
        }
    field = field_provider.field
    metadata = getattr(field, 'metadata', {})
    return {
        'field_backend_kind': str(field_backend_kind(field_provider)),
        'field_regularization_mode': str(metadata.get('field_regularization_mode', '')),
        'field_regularization_band_distance_m': float(metadata.get('field_regularization_band_distance_m', 0.0)),
        'field_regularization_added_node_count': int(metadata.get('field_regularization_added_node_count', 0)),
        'field_regularization_probe_success_count': int(metadata.get('field_regularization_probe_success_count', 0)),
        'field_regularization_probe_fallback_count': int(metadata.get('field_regularization_probe_fallback_count', 0)),
    }


def sample_field_valid_status(field_provider: FieldProviderND, position: np.ndarray) -> int:
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
