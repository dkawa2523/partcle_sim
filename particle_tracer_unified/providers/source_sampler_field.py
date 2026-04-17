from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..core.datamodel import FieldProviderND
from ..core.field_backend import sample_field_quantity, sample_field_valid, sample_field_vector
from ..core.field_sampling import choose_velocity_quantity_names
from .source_sampler_geometry import _SurfaceCache, _nearest_surface_record
from .source_sampler_types import SourceFlowSampler, SourceNormalSampler, SourceScalarSampler


def _is_comsol_export_bundle(field_provider: Optional[FieldProviderND]) -> bool:
    if field_provider is None:
        return False
    source_kind = str(field_provider.field.metadata.get('source_kind', '')).strip().lower()
    return source_kind == 'comsol_export_bundle_field'


class FieldScalarSamplerND(SourceScalarSampler):
    def __init__(
        self,
        field_provider: FieldProviderND,
        quantity_names: Sequence[str],
        time_interpolation: str = 'linear',
        default: float = np.nan,
    ):
        self.field_provider = field_provider
        self.quantity_names = tuple(quantity_names)
        self.time_interpolation = str(time_interpolation)
        self.default = float(default)

    def __call__(self, position: np.ndarray, release_time: float, source_part_id: int = 0) -> float:
        pos = np.asarray(position, dtype=np.float64)
        if not bool(sample_field_valid(self.field_provider, pos)):
            return self.default
        for name in self.quantity_names:
            value = sample_field_quantity(
                self.field_provider,
                name,
                pos,
                float(release_time),
                mode=self.time_interpolation,
                default=np.nan,
            )
            if np.isfinite(value):
                return float(value)
        return self.default


class FieldVectorSamplerND(SourceFlowSampler):
    def __init__(self, field_provider: FieldProviderND, component_names: Sequence[str], time_interpolation: str = 'linear'):
        self.field_provider = field_provider
        self.component_names = tuple(component_names)
        self.time_interpolation = str(time_interpolation)

    def __call__(self, position: np.ndarray, release_time: float) -> np.ndarray:
        pos = np.asarray(position, dtype=np.float64)
        if not bool(sample_field_valid(self.field_provider, pos)):
            return np.full(len(self.component_names), np.nan, dtype=np.float64)
        return sample_field_vector(
            self.field_provider,
            self.component_names,
            pos,
            float(release_time),
            mode=self.time_interpolation,
            default=np.nan,
        )


class FieldFlowSamplerND(SourceFlowSampler):
    def __init__(
        self,
        field_provider: FieldProviderND,
        velocity_names: Optional[Sequence[str]] = None,
        time_interpolation: str = 'linear',
    ):
        self.field_provider = field_provider
        self.time_interpolation = str(time_interpolation)
        field = field_provider.field
        if velocity_names is None:
            self.velocity_names = choose_velocity_quantity_names(field, field.spatial_dim)
        else:
            self.velocity_names = tuple(velocity_names)

    def __call__(self, position: np.ndarray, release_time: float) -> np.ndarray:
        field = self.field_provider.field
        if not self.velocity_names:
            return np.zeros(field.spatial_dim, dtype=np.float64)
        pos = np.asarray(position, dtype=np.float64)
        if not bool(sample_field_valid(self.field_provider, pos)):
            return np.zeros(field.spatial_dim, dtype=np.float64)
        return sample_field_vector(
            self.field_provider,
            self.velocity_names,
            pos,
            float(release_time),
            mode=self.time_interpolation,
            default=0.0,
        )


class VectorTangentialMagnitudeSampler(SourceScalarSampler):
    def __init__(self, vector_sampler: SourceFlowSampler, normal_sampler: SourceNormalSampler):
        self.vector_sampler = vector_sampler
        self.normal_sampler = normal_sampler

    def __call__(self, position: np.ndarray, release_time: float, source_part_id: int = 0) -> float:
        pos = np.asarray(position, dtype=np.float64)
        vector = np.asarray(self.vector_sampler(pos, release_time), dtype=np.float64)
        normal = np.asarray(self.normal_sampler(pos, source_part_id), dtype=np.float64)
        magnitude = np.linalg.norm(normal)
        if magnitude <= 1e-30:
            return float(np.linalg.norm(vector))
        normal = normal / magnitude
        tangential = vector - np.dot(vector, normal) * normal
        return float(np.linalg.norm(tangential))


class SurfaceMeshScalarFieldSampler(SourceScalarSampler):
    def __init__(
        self,
        cache: _SurfaceCache,
        field_provider: FieldProviderND,
        quantity_names: Sequence[str],
        time_interpolation: str = 'linear',
        default: float = np.nan,
    ):
        self.cache = cache
        self.field_provider = field_provider
        self.quantity_names = tuple(quantity_names)
        self.time_interpolation = str(time_interpolation)
        self.default = float(default)

    def __call__(self, position: np.ndarray, release_time: float, source_part_id: int = 0) -> float:
        centroid, _normal, _part_id = _nearest_surface_record(self.cache, position, source_part_id)
        if not bool(sample_field_valid(self.field_provider, centroid)):
            return self.default
        for name in self.quantity_names:
            value = sample_field_quantity(
                self.field_provider,
                name,
                centroid,
                float(release_time),
                mode=self.time_interpolation,
                default=np.nan,
            )
            if np.isfinite(value):
                return float(value)
        return self.default


class SurfaceMeshVectorShearSampler(SourceScalarSampler):
    def __init__(
        self,
        cache: _SurfaceCache,
        field_provider: FieldProviderND,
        component_names: Sequence[str],
        time_interpolation: str = 'linear',
        default: float = np.nan,
    ):
        self.cache = cache
        self.field_provider = field_provider
        self.component_names = tuple(component_names)
        self.time_interpolation = str(time_interpolation)
        self.default = float(default)

    def __call__(self, position: np.ndarray, release_time: float, source_part_id: int = 0) -> float:
        centroid, normal, _part_id = _nearest_surface_record(self.cache, position, source_part_id)
        if not bool(sample_field_valid(self.field_provider, centroid)):
            return self.default
        values = sample_field_vector(
            self.field_provider,
            self.component_names,
            centroid,
            float(release_time),
            mode=self.time_interpolation,
            default=np.nan,
        )
        if np.any(~np.isfinite(values)):
            return self.default
        vector = np.asarray(values, dtype=np.float64)
        normal_arr = np.asarray(normal, dtype=np.float64)
        magnitude = np.linalg.norm(normal_arr)
        if magnitude <= 1e-30:
            return float(np.linalg.norm(vector))
        normal_arr = normal_arr / magnitude
        tangential = vector - np.dot(vector, normal_arr) * normal_arr
        return float(np.linalg.norm(tangential))


def _wall_vector_component_candidates(field) -> Sequence[Tuple[str, ...]]:
    if field.spatial_dim == 2 and field.coordinate_system == 'axisymmetric_rz':
        return [
            ('tauw_r', 'tauw_z'),
            ('tau_wall_r', 'tau_wall_z'),
            ('traction_r', 'traction_z'),
            ('surface_traction_r', 'surface_traction_z'),
            ('tauw_x', 'tauw_y'),
            ('tau_wall_x', 'tau_wall_y'),
            ('traction_x', 'traction_y'),
            ('surface_traction_x', 'surface_traction_y'),
        ]
    if field.spatial_dim == 2:
        return [
            ('tauw_x', 'tauw_y'),
            ('tau_wall_x', 'tau_wall_y'),
            ('traction_x', 'traction_y'),
            ('surface_traction_x', 'surface_traction_y'),
        ]
    return [
        ('tauw_x', 'tauw_y', 'tauw_z'),
        ('tau_wall_x', 'tau_wall_y', 'tau_wall_z'),
        ('traction_x', 'traction_y', 'traction_z'),
        ('surface_traction_x', 'surface_traction_y', 'surface_traction_z'),
    ]


__all__ = (
    'FieldFlowSamplerND',
    'FieldScalarSamplerND',
    'FieldVectorSamplerND',
    'SurfaceMeshScalarFieldSampler',
    'SurfaceMeshVectorShearSampler',
    'VectorTangentialMagnitudeSampler',
    '_is_comsol_export_bundle',
    '_wall_vector_component_candidates',
)
