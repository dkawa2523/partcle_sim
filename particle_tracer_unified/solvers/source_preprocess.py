from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..core.boundary_service import build_boundary_service
from ..core.datamodel import (
    ParticleTable,
    ProcessStepTable,
    RuntimeLike,
    SourceEventTable,
    SourcePreprocessResult,
    SourceResolutionParameters,
)
from ..core.geometry3d import build_triangle_surface
from ..core.source_materials import apply_source_models
from ..providers.source_adapters import ConstantScalarSampler, SourceFlowSampler, SourceNormalSampler, SourceScalarSampler, ZeroFlowSampler


def _bool_config(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'on', 'enabled'}:
        return True
    if text in {'0', 'false', 'no', 'off', 'disabled', ''}:
        return False
    raise ValueError('source.preprocess.boundary_release must be true or false')


def _nonnegative_float_config(value: Any, *, key: str, default: float) -> float:
    if value is None:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'source.preprocess.{key} must be a finite non-negative value') from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f'source.preprocess.{key} must be a finite non-negative value')
    return float(parsed)


def boundary_release_config(runtime: RuntimeLike, source_cfg: Mapping[str, Any]) -> Dict[str, float | bool]:
    preprocess_cfg = source_cfg.get('preprocess', {}) if isinstance(source_cfg.get('preprocess', {}), Mapping) else {}
    legacy_keys = [f'boundary_release_{suffix}' for suffix in ('enabled', 'tolerance_m', 'offset_m') if f'boundary_release_{suffix}' in preprocess_cfg]
    if legacy_keys:
        raise ValueError(
            'source.preprocess no longer accepts legacy boundary-release alias keys; use boundary_release=true '
            'and source.source_position_offset_m for any physical release offset'
        )
    raw = preprocess_cfg.get('boundary_release', False)
    if isinstance(raw, Mapping):
        raise ValueError('source.preprocess.boundary_release must be true or false; object form is not supported')
    enabled = _bool_config(raw, default=False)

    wall_cfg = runtime.config_payload.get('wall', {}) if isinstance(runtime.config_payload, Mapping) else {}
    solver_cfg = runtime.config_payload.get('solver', {}) if isinstance(runtime.config_payload, Mapping) else {}
    epsilon_offset_m = float(wall_cfg.get('epsilon_offset_m', 1.0e-6)) if isinstance(wall_cfg, Mapping) else 1.0e-6
    on_boundary_raw = solver_cfg.get('on_boundary_tol_m', np.nan) if isinstance(solver_cfg, Mapping) else np.nan
    try:
        on_boundary_value = float(on_boundary_raw)
    except (TypeError, ValueError):
        on_boundary_value = float('nan')
    on_boundary_tol_m = on_boundary_value if np.isfinite(on_boundary_value) else max(2.0 * epsilon_offset_m, 5.0e-7)
    capture_tolerance_m = _nonnegative_float_config(
        preprocess_cfg.get('boundary_capture_tolerance_m'),
        key='boundary_capture_tolerance_m',
        default=float(on_boundary_tol_m),
    )
    inward_offset_m = _nonnegative_float_config(
        preprocess_cfg.get('boundary_inward_offset_m'),
        key='boundary_inward_offset_m',
        default=max(float(capture_tolerance_m), float(epsilon_offset_m)),
    )
    return {
        'enabled': bool(enabled),
        'release_offset_m': float(inward_offset_m),
        'tolerance_m': float(capture_tolerance_m),
        'on_boundary_tol_m': float(capture_tolerance_m),
        'capture_tolerance_m': float(capture_tolerance_m),
        'inward_offset_m': float(inward_offset_m),
    }


def has_explicit_boundary_primitives(runtime: RuntimeLike) -> bool:
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None:
        return False
    geom = geometry_provider.geometry
    if int(runtime.spatial_dim) == 2:
        edges = getattr(geom, 'boundary_edges', None)
        return edges is not None and np.asarray(edges).size > 0
    triangles = getattr(geom, 'boundary_triangles', None)
    return triangles is not None and np.asarray(triangles).size > 0


def boundary_service_for_source_preprocess(runtime: RuntimeLike, on_boundary_tol_m: float):
    if not has_explicit_boundary_primitives(runtime):
        raise ValueError(
            'source.preprocess.boundary_release requires explicit boundary primitives '
            '(2D boundary_edges or 3D boundary_triangles)'
        )
    triangle_surface = None
    if int(runtime.spatial_dim) == 3 and runtime.geometry_provider is not None:
        geom = runtime.geometry_provider.geometry
        if geom.boundary_triangles is not None:
            triangle_surface = build_triangle_surface(
                np.asarray(geom.boundary_triangles, dtype=np.float64),
                np.asarray(
                    geom.boundary_triangle_part_ids
                    if geom.boundary_triangle_part_ids is not None
                    else np.zeros(np.asarray(geom.boundary_triangles).shape[0], dtype=np.int32),
                    dtype=np.int32,
                ),
                validate_closed=True,
            )
    return build_boundary_service(
        runtime,
        spatial_dim=int(runtime.spatial_dim),
        on_boundary_tol_m=float(on_boundary_tol_m),
        triangle_surface_3d=triangle_surface,
    )


def preprocess_particles_for_solver(
    particles: ParticleTable,
    source_events: Optional[SourceEventTable],
    source_cfg: Mapping[str, Any],
    gas_density_kgm3: float,
    resolved: SourceResolutionParameters,
    normal_sampler: SourceNormalSampler,
    flow_sampler: Optional[SourceFlowSampler] = None,
    wall_shear_sampler: Optional[SourceScalarSampler] = None,
    friction_velocity_sampler: Optional[SourceScalarSampler] = None,
    viscosity_sampler: Optional[SourceScalarSampler] = None,
    process_steps: Optional[ProcessStepTable] = None,
    seed: int = 12345,
    release_point_classifier: Optional[object] = None,
    use_boundary_release: bool = False,
    boundary_classifier_tolerance_m: float = 0.0,
    boundary_solver_offset_m: float = 0.0,
) -> SourcePreprocessResult:
    preprocess_cfg = source_cfg.get('preprocess', {}) if isinstance(source_cfg.get('preprocess', {}), Mapping) else {}
    normal_velocity_policy = str(
        preprocess_cfg.get('normal_velocity_policy', source_cfg.get('normal_velocity_policy', 'keep'))
    )
    return apply_source_models(
        particles=particles,
        resolved=resolved,
        normal_sampler=normal_sampler,
        flow_sampler=flow_sampler or ZeroFlowSampler(particles.spatial_dim),
        wall_shear_sampler=wall_shear_sampler or ConstantScalarSampler(float('nan')),
        friction_velocity_sampler=friction_velocity_sampler or ConstantScalarSampler(float('nan')),
        viscosity_sampler=viscosity_sampler or ConstantScalarSampler(float('nan')),
        events=source_events,
        process_steps=process_steps,
        gas_density_kgm3=float(gas_density_kgm3),
        seed=seed,
        normal_velocity_policy=normal_velocity_policy,
        release_point_classifier=release_point_classifier,
        use_boundary_release=bool(use_boundary_release),
        boundary_classifier_tolerance_m=float(boundary_classifier_tolerance_m),
        boundary_solver_offset_m=float(boundary_solver_offset_m),
    )
