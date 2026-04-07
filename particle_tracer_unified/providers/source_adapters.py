from __future__ import annotations

from typing import Optional

import numpy as np

from .source_sampler_field import (
    FieldFlowSamplerND,
    FieldScalarSamplerND,
    FieldVectorSamplerND,
    SurfaceMeshScalarFieldSampler,
    SurfaceMeshVectorShearSampler,
    VectorTangentialMagnitudeSampler,
    _is_comsol_export_bundle,
    _wall_vector_component_candidates,
)
from .source_sampler_geometry import (
    DirectFrictionVelocitySampler,
    DirectWallShearSampler,
    GeometryNormalSamplerND,
    _default_probe_distance,
    _surface_cache_from_geometry,
)
from .source_sampler_types import (
    ConstantScalarSampler,
    SourceFlowSampler,
    SourceNormalSampler,
    SourceScalarSampler,
    ZeroFlowSampler,
)


def build_normal_sampler(runtime) -> SourceNormalSampler:
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None:
        class _DefaultNormal(SourceNormalSampler):
            def __call__(self, position, source_part_id):
                dim = int(getattr(runtime, 'spatial_dim', 2))
                normal = np.zeros(dim, dtype=np.float64)
                normal[-1 if dim > 1 else 0] = 1.0
                return normal

        return _DefaultNormal()
    return GeometryNormalSamplerND(geometry_provider)


def build_flow_sampler(runtime) -> SourceFlowSampler:
    field_provider = getattr(runtime, 'field_provider', None)
    spatial_dim = int(getattr(runtime, 'spatial_dim', 2))
    if field_provider is None:
        return ZeroFlowSampler(spatial_dim)
    time_interpolation = getattr(runtime, 'time_interpolation', 'linear')
    return FieldFlowSamplerND(field_provider, time_interpolation=time_interpolation)


def build_viscosity_sampler(runtime) -> SourceScalarSampler:
    field_provider = getattr(runtime, 'field_provider', None)
    default = np.nan
    gas = getattr(runtime, 'gas', None)
    if gas is not None and hasattr(gas, 'dynamic_viscosity_Pas'):
        default = float(gas.dynamic_viscosity_Pas)
    if field_provider is None:
        return ConstantScalarSampler(default)
    time_interpolation = getattr(runtime, 'time_interpolation', 'linear')
    if _is_comsol_export_bundle(field_provider) and not any(
        name in field_provider.field.quantities for name in ('mu', 'dynamic_viscosity', 'dynamic_viscosity_Pas')
    ):
        raise ValueError(
            'COMSOL export bundle is missing viscosity quantity required for viscosity sampling (mu)'
        )
    return FieldScalarSamplerND(
        field_provider,
        quantity_names=('mu', 'dynamic_viscosity', 'dynamic_viscosity_Pas'),
        time_interpolation=time_interpolation,
        default=default,
    )


def build_wall_shear_sampler(
    runtime,
    normal_sampler: Optional[SourceNormalSampler] = None,
    flow_sampler: Optional[SourceFlowSampler] = None,
    viscosity_sampler: Optional[SourceScalarSampler] = None,
) -> SourceScalarSampler:
    field_provider = getattr(runtime, 'field_provider', None)
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    normal_sampler = normal_sampler or build_normal_sampler(runtime)
    cache = _surface_cache_from_geometry(geometry_provider) if geometry_provider is not None else None
    if field_provider is not None:
        time_interpolation = getattr(runtime, 'time_interpolation', 'linear')
        field = field_provider.field
        scalar_names = ('tauw', 'tau_wall', 'wall_shear_stress', 'tauw_mag')
        if any(name in field.quantities for name in scalar_names):
            if cache is not None:
                return SurfaceMeshScalarFieldSampler(
                    cache,
                    field_provider,
                    scalar_names,
                    time_interpolation=time_interpolation,
                    default=np.nan,
                )
            return FieldScalarSamplerND(
                field_provider,
                quantity_names=scalar_names,
                time_interpolation=time_interpolation,
                default=np.nan,
            )
        for components in _wall_vector_component_candidates(field):
            if all(name in field.quantities for name in components):
                if cache is not None:
                    return SurfaceMeshVectorShearSampler(
                        cache,
                        field_provider,
                        components,
                        time_interpolation=time_interpolation,
                        default=np.nan,
                    )
                vector_sampler = FieldVectorSamplerND(
                    field_provider,
                    components,
                    time_interpolation=time_interpolation,
                )
                return VectorTangentialMagnitudeSampler(vector_sampler, normal_sampler)
        if _is_comsol_export_bundle(field_provider):
            raise ValueError(
                'COMSOL export bundle is missing wall shear quantity required for wall_shear sampling '
                '(tauw or vector components)'
            )
    if field_provider is None or geometry_provider is None:
        return ConstantScalarSampler(np.nan)
    flow_sampler = flow_sampler or build_flow_sampler(runtime)
    viscosity_sampler = viscosity_sampler or build_viscosity_sampler(runtime)
    source_cfg = getattr(runtime, 'config_payload', {}).get('source', {}) if hasattr(runtime, 'config_payload') else {}
    direct_cfg = source_cfg.get('direct_wall_shear', {}) if isinstance(source_cfg.get('direct_wall_shear', {}), dict) else {}
    probe_distance = float(direct_cfg.get('probe_distance_m', _default_probe_distance(runtime)))
    return DirectWallShearSampler(
        geometry_provider,
        flow_sampler,
        normal_sampler,
        viscosity_sampler,
        probe_distance,
    )


def build_friction_velocity_sampler(
    runtime,
    wall_shear_sampler: Optional[SourceScalarSampler] = None,
) -> SourceScalarSampler:
    field_provider = getattr(runtime, 'field_provider', None)
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    cache = _surface_cache_from_geometry(geometry_provider) if geometry_provider is not None else None
    if field_provider is not None:
        time_interpolation = getattr(runtime, 'time_interpolation', 'linear')
        field = field_provider.field
        scalar_names = ('u_tau', 'utau', 'friction_velocity', 'friction_velocity_mps')
        if any(name in field.quantities for name in scalar_names):
            if cache is not None:
                return SurfaceMeshScalarFieldSampler(
                    cache,
                    field_provider,
                    scalar_names,
                    time_interpolation=time_interpolation,
                    default=np.nan,
                )
            return FieldScalarSamplerND(
                field_provider,
                quantity_names=scalar_names,
                time_interpolation=time_interpolation,
                default=np.nan,
            )
        if _is_comsol_export_bundle(field_provider):
            raise ValueError(
                'COMSOL export bundle is missing friction velocity quantity required for friction_velocity '
                'sampling (u_tau)'
            )
    gas = getattr(runtime, 'gas', None)
    gas_density = float(getattr(gas, 'density_kgm3', 1.0)) if gas is not None else 1.0
    wall_shear_sampler = wall_shear_sampler or build_wall_shear_sampler(runtime)
    return DirectFrictionVelocitySampler(wall_shear_sampler, gas_density)


__all__ = (
    'ConstantScalarSampler',
    'DirectFrictionVelocitySampler',
    'DirectWallShearSampler',
    'FieldFlowSamplerND',
    'FieldScalarSamplerND',
    'FieldVectorSamplerND',
    'GeometryNormalSamplerND',
    'SourceFlowSampler',
    'SourceNormalSampler',
    'SourceScalarSampler',
    'SurfaceMeshScalarFieldSampler',
    'SurfaceMeshVectorShearSampler',
    'VectorTangentialMagnitudeSampler',
    'ZeroFlowSampler',
    'build_flow_sampler',
    'build_friction_velocity_sampler',
    'build_normal_sampler',
    'build_viscosity_sampler',
    'build_wall_shear_sampler',
)
