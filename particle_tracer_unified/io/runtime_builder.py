from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping
import yaml

from ..core.catalogs import build_physics_catalog, build_wall_catalog, physics_catalog_summary, wall_catalog_summary
from ..core.coordinate_systems import normalize_coordinate_system
from ..core.datamodel import GasProperties, PreparedRuntime, RuntimeLike, replace_runtime_particles
from ..core.process_steps import process_step_control_summary
from ..providers.source_adapters import (
    build_flow_sampler,
    build_friction_velocity_sampler,
    build_normal_sampler,
    build_viscosity_sampler,
    build_wall_shear_sampler,
)
from .runtime_builder_support import build_runtime_providers, load_runtime_inputs, resolve_runtime_input_paths
from ..solvers.forces import build_force_catalog, force_catalog_summary
from ..solvers.source_preprocess import preprocess_particles_for_solver


def _read_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('YAML root must be a mapping')
    return data

def build_runtime_from_config(config: Mapping[str, Any], config_dir: Path) -> RuntimeLike:
    run = dict(config.get('run', {}))
    paths = dict(config.get('paths', {}))
    providers_cfg = dict(config.get('providers', {}))
    source_cfg = dict(config.get('source', {}))
    gas_cfg = dict(config.get('gas', {}))

    spatial_dim = int(run.get('spatial_dim', 2))
    coordinate_system = normalize_coordinate_system(run.get('coordinate_system'), spatial_dim)
    time_interpolation = str(run.get('time_interpolation', 'linear'))

    resolved_paths = resolve_runtime_input_paths(config_dir, paths)
    runtime_inputs = load_runtime_inputs(
        paths=resolved_paths,
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
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

    wall_catalog = build_wall_catalog(runtime_inputs.walls, runtime_inputs.materials, config)
    physics_catalog = build_physics_catalog(config, spatial_dim)
    force_catalog = build_force_catalog(
        config,
        field_provider=providers.field_provider,
        spatial_dim=spatial_dim,
    )

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
        config_payload=config,
        wall_catalog=wall_catalog,
        physics_catalog=physics_catalog,
        force_catalog=force_catalog,
    )



def prepare_runtime(runtime: RuntimeLike, seed: Optional[int] = None) -> PreparedRuntime:
    source_cfg = runtime.config_payload.get('source', {}) if isinstance(runtime.config_payload, Mapping) else {}
    preprocess_cfg = source_cfg.get('preprocess', {}) if isinstance(source_cfg.get('preprocess', {}), Mapping) else {}
    if not bool(preprocess_cfg.get('enabled', True)):
        return PreparedRuntime(runtime=runtime, source_preprocess=None)
    normal_sampler = build_normal_sampler(runtime)
    flow_sampler = build_flow_sampler(runtime)
    viscosity_sampler = build_viscosity_sampler(runtime)
    wall_shear_sampler = build_wall_shear_sampler(runtime, normal_sampler=normal_sampler, flow_sampler=flow_sampler, viscosity_sampler=viscosity_sampler)
    friction_velocity_sampler = build_friction_velocity_sampler(runtime, wall_shear_sampler=wall_shear_sampler)
    result = preprocess_particles_for_solver(
        particles=runtime.particles,
        walls=runtime.walls,
        materials=runtime.materials,
        source_events=runtime.compiled_source_events,
        process_steps=runtime.process_steps,
        source_cfg=source_cfg,
        gas_temperature=float(runtime.gas.temperature),
        gas_viscosity=float(runtime.gas.dynamic_viscosity_Pas),
        gas_density_kgm3=float(runtime.gas.density_kgm3),
        normal_sampler=normal_sampler,
        flow_sampler=flow_sampler,
        wall_shear_sampler=wall_shear_sampler,
        friction_velocity_sampler=friction_velocity_sampler,
        viscosity_sampler=viscosity_sampler,
        seed=int(seed if seed is not None else preprocess_cfg.get('seed', 12345)),
    )
    prepared_runtime = replace_runtime_particles(runtime, result.particles, source_preprocess=result, compiled_source_events=runtime.compiled_source_events)
    return PreparedRuntime(runtime=prepared_runtime, source_preprocess=result)


def build_prepared_runtime_from_yaml(config_path: Path) -> PreparedRuntime:
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    runtime = build_runtime_from_config(config, config_path.parent)
    return prepare_runtime(runtime)


def prepared_runtime_summary(prepared: PreparedRuntime) -> Dict[str, Any]:
    runtime = prepared.runtime
    summary = {
        'spatial_dim': int(runtime.spatial_dim),
        'coordinate_system': runtime.coordinate_system,
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
        'force_catalog': force_catalog_summary(runtime.force_catalog),
    }
    if prepared.source_preprocess is not None:
        summary['source_model_summary'] = dict(prepared.source_preprocess.source_model_summary)
        summary['event_summary'] = dict(prepared.source_preprocess.event_summary)
    if runtime.geometry_provider is not None:
        summary['geometry_provider'] = runtime.geometry_provider.summary()
    if runtime.field_provider is not None:
        summary['field_provider'] = runtime.field_provider.summary()
    return summary
