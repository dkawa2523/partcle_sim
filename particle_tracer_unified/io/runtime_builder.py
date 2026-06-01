from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import numpy as np
import yaml

from ..core.catalogs import build_physics_catalog, build_wall_catalog, physics_catalog_summary, wall_catalog_summary
from ..core.coordinate_systems import (
    axis_names_for_coordinate_system,
    axisymmetric_rz_report_from_metadata,
    normalize_coordinate_system,
    ring_area_weight,
)
from ..core.datamodel import GasProperties, PreparedRuntime, RuntimeLike, replace_runtime_particles
from ..core.process_steps import process_step_control_summary
from ..providers.source_adapters import (
    build_flow_sampler,
    build_friction_velocity_sampler,
    build_normal_sampler,
    build_viscosity_sampler,
    build_wall_shear_sampler,
    ConstantScalarSampler,
)
from .runtime_builder_support import build_runtime_providers, load_runtime_inputs, resolve_runtime_input_paths
from .comsol import (
    enforce_comsol_faithful_config,
    finalize_comsol_runtime_config,
    is_comsol_faithful_config,
    load_comsol_manifest_for_config,
    load_comsol_runtime_inputs,
)
from ..solvers.forces import build_force_catalog, force_catalog_summary
from ..solvers.charge_model import parse_charge_model_config
from ..solvers.plasma_background import parse_plasma_background_config
from ..solvers.source_preprocess import (
    boundary_release_config,
    boundary_service_for_source_preprocess,
    preprocess_particles_for_solver,
)
from ..solvers.stochastic_motion import parse_stochastic_motion_config
from ..core.source_resolution import resolve_source_parameters


def _read_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('YAML root must be a mapping')
    return data


def _summary_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_config(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'1', 'true', 'yes', 'on'}:
            return True
        if text in {'0', 'false', 'no', 'off', ''}:
            return False
    return bool(value)


def _model_input_summary(
    runtime: RuntimeLike,
    force_summary: Mapping[str, Any],
    source_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    config = _summary_mapping(runtime.config_payload)
    solver_cfg = _summary_mapping(config.get('solver', {}))
    force_models = _summary_mapping(force_summary.get('force_models', {}))
    stochastic_cfg = parse_stochastic_motion_config(
        solver_cfg,
        default_seed=int(solver_cfg.get('seed', 12345)),
    )
    charge_cfg = parse_charge_model_config(solver_cfg)
    plasma_cfg = parse_plasma_background_config(solver_cfg)

    return {
        'drag_model': str(force_models.get('drag', solver_cfg.get('drag_model', 'stokes'))),
        'enabled_forces': list(force_summary.get('enabled_forces', [])),
        'force_enabled_reason': dict(_summary_mapping(force_summary.get('force_enabled_reason', {}))),
        'stochastic_motion_enabled': int(bool(stochastic_cfg.enabled)),
        'stochastic_motion_model': str(stochastic_cfg.model),
        'stochastic_motion_temperature_source': str(stochastic_cfg.temperature_source),
        'charge_model_enabled': int(bool(charge_cfg.enabled)),
        'charge_model_mode': str(charge_cfg.mode),
        'charge_background_source': str(charge_cfg.background_source),
        'charge_model_support_scope': '2d_regular_rectilinear_field_or_scalar_plasma_background',
        'plasma_background_enabled': int(str(plasma_cfg.source) != 'none'),
        'plasma_background_source': str(plasma_cfg.source),
        'near_wall_force_correction_applied': 0,
        'particle_coupling': 'one_way',
        'particle_wall_contact_geometry': 'center_position',
        'source_law_usage': dict(_summary_mapping(source_summary.get('law_usage', {}))),
    }


def _finite_weight_summary(values) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {'count': 0, 'min': None, 'max': None, 'sum': 0.0}
    return {
        'count': int(finite.size),
        'min': float(np.min(finite)),
        'max': float(np.max(finite)),
        'sum': float(np.sum(finite)),
    }


def _with_axisymmetric_source_ring_weight_report(
    runtime: RuntimeLike,
    result,
    preprocess_cfg: Mapping[str, Any],
):
    enabled = _bool_config(preprocess_cfg.get('ring_area_weighted_source_reporting', False), default=False)
    if str(runtime.coordinate_system) != 'axisymmetric_rz':
        if enabled:
            raise ValueError(
                'source.preprocess.ring_area_weighted_source_reporting requires coordinate_system=axisymmetric_rz'
            )
        return result

    summary = dict(result.source_model_summary)
    ring_summary: Dict[str, Any] = {
        'enabled': int(bool(enabled)),
        'applied_to_sampling': 0,
        'radius_axis': 'r',
        'weight_formula': '2*pi*r',
        'policy': 'explicit_reporting_only' if enabled else 'not_applied',
    }
    diagnostics_rows = tuple(dict(row) for row in result.diagnostics_rows)
    if enabled:
        radii = np.asarray(result.particles.position[:, 0], dtype=np.float64)
        weights = np.asarray(ring_area_weight(radii), dtype=np.float64)
        updated_rows = []
        for row, radius, weight in zip(diagnostics_rows, radii, weights):
            updated = dict(row)
            updated['axisymmetric_radius_m'] = float(radius)
            updated['axisymmetric_ring_area_weight'] = float(weight)
            updated_rows.append(updated)
        diagnostics_rows = tuple(updated_rows)
        ring_summary.update(_finite_weight_summary(weights))
    summary['axisymmetric_ring_area_weight'] = ring_summary
    return replace(result, source_model_summary=summary, diagnostics_rows=diagnostics_rows)


def build_runtime_from_config(config: Mapping[str, Any], config_dir: Path) -> RuntimeLike:
    config_payload = copy.deepcopy(dict(config))
    manifest = load_comsol_manifest_for_config(config_payload, config_dir)
    if manifest is not None:
        enforce_comsol_faithful_config(config_payload, manifest)

    run = dict(config_payload.get('run', {}))
    paths = dict(config_payload.get('paths', {}))
    providers_cfg = dict(config_payload.get('providers', {}))
    gas_cfg = dict(config_payload.get('gas', {}))

    spatial_dim = int(run.get('spatial_dim', 2))
    coordinate_source = run.get('coordinate_system')
    if manifest is not None:
        manifest_coordinate_system = normalize_coordinate_system(manifest.coordinate_system, spatial_dim)
        if coordinate_source is not None and str(coordinate_source).strip():
            run_coordinate_system = normalize_coordinate_system(coordinate_source, spatial_dim)
            if run_coordinate_system != manifest_coordinate_system:
                raise ValueError(
                    'run.coordinate_system must match COMSOL manifest coordinates.coordinate_system: '
                    f'{run_coordinate_system!r} != {manifest_coordinate_system!r}'
                )
            coordinate_source = run_coordinate_system
        else:
            coordinate_source = manifest_coordinate_system
    coordinate_system = normalize_coordinate_system(coordinate_source, spatial_dim)
    run['spatial_dim'] = spatial_dim
    run['coordinate_system'] = coordinate_system
    config_payload['run'] = run
    time_interpolation = str(run.get('time_interpolation', 'linear'))

    if manifest is None:
        resolved_paths = resolve_runtime_input_paths(config_dir, paths)
        runtime_inputs = load_runtime_inputs(
            paths=resolved_paths,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
        )
    else:
        runtime_inputs = load_comsol_runtime_inputs(
            config=config_payload,
            config_dir=config_dir,
            manifest=manifest,
            spatial_dim=spatial_dim,
        )

    gas = GasProperties(
        temperature=float(gas_cfg.get('temperature_K', gas_cfg.get('temperature', 300.0))),
        dynamic_viscosity_Pas=float(gas_cfg.get('dynamic_viscosity_Pas', 1.8e-5)),
        density_kgm3=float(gas_cfg.get('density_kgm3', gas_cfg.get('density', 1.0))),
        molecular_mass_amu=float(gas_cfg.get('molecular_mass_amu', gas_cfg.get('molecular_mass', 60.0))),
    )

    providers = build_runtime_providers(
        config_dir=config_dir,
        providers_cfg=providers_cfg,
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
        gas_density_kgm3=float(gas.density_kgm3),
    )

    wall_catalog = build_wall_catalog(runtime_inputs.walls, runtime_inputs.materials, config_payload)
    physics_catalog = build_physics_catalog(config_payload, spatial_dim)
    force_catalog = build_force_catalog(
        config_payload,
        field_provider=providers.field_provider,
        spatial_dim=spatial_dim,
    )
    if manifest is not None:
        finalize_comsol_runtime_config(config_payload, manifest, providers.field_provider)

    return RuntimeLike(
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
        particles=runtime_inputs.particles,
        walls=runtime_inputs.walls,
        materials=runtime_inputs.materials,
        source_events=runtime_inputs.source_events,
        process_steps=runtime_inputs.process_steps,
        compiled_source_events=runtime_inputs.compiled_source_events,
        geometry_provider=providers.geometry_provider,
        field_provider=providers.field_provider,
        gas=gas,
        time_interpolation=time_interpolation,
        config_payload=config_payload,
        wall_catalog=wall_catalog,
        physics_catalog=physics_catalog,
        force_catalog=force_catalog,
    )

def prepare_runtime(runtime: RuntimeLike, seed: Optional[int] = None) -> PreparedRuntime:
    source_cfg = runtime.config_payload.get('source', {}) if isinstance(runtime.config_payload, Mapping) else {}
    preprocess_cfg = source_cfg.get('preprocess', {}) if isinstance(source_cfg.get('preprocess', {}), Mapping) else {}
    if is_comsol_faithful_config(runtime.config_payload if isinstance(runtime.config_payload, Mapping) else {}):
        if _bool_config(preprocess_cfg.get('enabled', False), default=False):
            raise ValueError('source.preprocess.enabled must be false in COMSOL faithful mode')
        if _bool_config(preprocess_cfg.get('boundary_release', False), default=False):
            raise ValueError('source.preprocess.boundary_release must be false in COMSOL faithful mode')
        return PreparedRuntime(runtime=runtime, source_preprocess=None)
    if not bool(preprocess_cfg.get('enabled', True)):
        return PreparedRuntime(runtime=runtime, source_preprocess=None)
    boundary_release = boundary_release_config(runtime, source_cfg)
    boundary_service = (
        boundary_service_for_source_preprocess(runtime, float(boundary_release['on_boundary_tol_m']))
        if bool(boundary_release['enabled'])
        else None
    )
    normal_sampler = build_normal_sampler(runtime)
    flow_sampler = build_flow_sampler(runtime)
    source_resolution = resolve_source_parameters(
        particles=runtime.particles,
        walls=runtime.walls,
        materials=runtime.materials,
        source_cfg=source_cfg,
        gas_temperature=float(runtime.gas.temperature),
        gas_viscosity=float(runtime.gas.dynamic_viscosity_Pas),
    )
    needs_shear_source = any(str(name) == 'resuspension_shear_material' for name in source_resolution.resolved_law_name)
    if needs_shear_source:
        viscosity_sampler = build_viscosity_sampler(runtime)
        wall_shear_sampler = build_wall_shear_sampler(
            runtime,
            normal_sampler=normal_sampler,
            flow_sampler=flow_sampler,
            viscosity_sampler=viscosity_sampler,
        )
        friction_velocity_sampler = build_friction_velocity_sampler(runtime, wall_shear_sampler=wall_shear_sampler)
    else:
        viscosity_sampler = ConstantScalarSampler(float(runtime.gas.dynamic_viscosity_Pas))
        wall_shear_sampler = ConstantScalarSampler(float('nan'))
        friction_velocity_sampler = ConstantScalarSampler(float('nan'))
    result = preprocess_particles_for_solver(
        particles=runtime.particles,
        source_events=runtime.compiled_source_events,
        process_steps=runtime.process_steps,
        source_cfg=source_cfg,
        gas_density_kgm3=float(runtime.gas.density_kgm3),
        resolved=source_resolution,
        normal_sampler=normal_sampler,
        flow_sampler=flow_sampler,
        wall_shear_sampler=wall_shear_sampler,
        friction_velocity_sampler=friction_velocity_sampler,
        viscosity_sampler=viscosity_sampler,
        seed=int(seed if seed is not None else preprocess_cfg.get('seed', 12345)),
        release_point_classifier=(boundary_service.release_point if boundary_service is not None else None),
        use_boundary_release=bool(boundary_release['enabled']),
        boundary_classifier_tolerance_m=float(boundary_release['tolerance_m']),
        boundary_solver_offset_m=float(boundary_release['release_offset_m']),
    )
    result = _with_axisymmetric_source_ring_weight_report(runtime, result, preprocess_cfg)
    prepared_runtime = replace_runtime_particles(
        runtime,
        result.particles,
        source_preprocess=result,
        compiled_source_events=runtime.compiled_source_events,
    )
    return PreparedRuntime(runtime=prepared_runtime, source_preprocess=result)


def build_prepared_runtime_from_yaml(config_path: Path) -> PreparedRuntime:
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    runtime = build_runtime_from_config(config, config_path.parent)
    return prepare_runtime(runtime)


def prepared_runtime_summary(prepared: PreparedRuntime) -> Dict[str, Any]:
    runtime = prepared.runtime
    force_summary = force_catalog_summary(runtime.force_catalog)
    source_model_summary: Dict[str, Any] = {}
    event_summary: Dict[str, Any] = {}
    if prepared.source_preprocess is not None:
        source_model_summary = dict(prepared.source_preprocess.source_model_summary)
        event_summary = dict(prepared.source_preprocess.event_summary)
    summary = {
        'spatial_dim': int(runtime.spatial_dim),
        'coordinate_system': runtime.coordinate_system,
        'axis_names': list(axis_names_for_coordinate_system(runtime.coordinate_system, runtime.spatial_dim)),
        'particles': int(runtime.particles.count if runtime.particles is not None else 0),
        'has_geometry_provider': runtime.geometry_provider is not None,
        'has_field_provider': runtime.field_provider is not None,
        'has_materials': runtime.materials is not None,
        'has_walls': runtime.walls is not None,
        'has_source_events': runtime.source_events is not None,
        'has_compiled_source_events': runtime.compiled_source_events is not None,
        'time_interpolation': runtime.time_interpolation,
        'gas': {
            'temperature_K': float(runtime.gas.temperature),
            'dynamic_viscosity_Pas': float(runtime.gas.dynamic_viscosity_Pas),
            'density_kgm3': float(runtime.gas.density_kgm3),
            'molecular_mass_amu': float(runtime.gas.molecular_mass_amu),
        },
        'process_steps': process_step_control_summary(runtime.process_steps),
        'wall_catalog': wall_catalog_summary(runtime.wall_catalog),
        'physics_catalog': physics_catalog_summary(runtime.physics_catalog),
        'force_catalog': force_summary,
        'model_input_summary': _model_input_summary(runtime, force_summary, source_model_summary),
    }
    if prepared.source_preprocess is not None:
        summary['source_model_summary'] = source_model_summary
        summary['event_summary'] = event_summary
    if runtime.geometry_provider is not None:
        summary['geometry_provider'] = runtime.geometry_provider.summary()
        axisymmetric_report = axisymmetric_rz_report_from_metadata(runtime.geometry_provider.geometry.metadata)
        if axisymmetric_report:
            summary['axisymmetric_rz'] = axisymmetric_report
    if runtime.field_provider is not None:
        summary['field_provider'] = runtime.field_provider.summary()
    return summary
