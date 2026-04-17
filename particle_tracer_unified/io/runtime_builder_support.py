from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..core.datamodel import (
    FieldProviderND,
    GeometryProviderND,
    MaterialTable,
    PartWallTable,
    ParticleTable,
    ProcessStepTable,
    SourceEventTable,
)
from ..core.source_events import compile_source_events
from ..io.tables import (
    load_materials_csv,
    load_part_walls_csv,
    load_particles_csv,
    load_process_steps_csv,
    load_source_events_csv,
)
from ..providers.precomputed import build_precomputed_field, build_precomputed_geometry, build_precomputed_triangle_mesh_field
from ..providers.synthetic import build_synthetic_field, build_synthetic_geometry


@dataclass(frozen=True)
class ResolvedRuntimePaths:
    particles_path: Path
    materials_path: Optional[Path]
    walls_path: Optional[Path]
    events_path: Optional[Path]
    process_steps_path: Optional[Path]


@dataclass(frozen=True)
class LoadedRuntimeInputs:
    particles: ParticleTable
    materials: Optional[MaterialTable]
    walls: Optional[PartWallTable]
    source_events: Optional[SourceEventTable]
    process_steps: Optional[ProcessStepTable]
    compiled_source_events: Optional[SourceEventTable]


@dataclass(frozen=True)
class RuntimeProviders:
    geometry_provider: Optional[GeometryProviderND]
    field_provider: Optional[FieldProviderND]


def _resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None or str(value).strip() == '':
        return None
    path = Path(str(value))
    return (base / path).resolve() if not path.is_absolute() else path


def _validate_process_steps(process_steps: Optional[ProcessStepTable]) -> None:
    if process_steps is None:
        return
    rows = tuple(process_steps.rows)
    if not rows:
        raise ValueError('Configured process steps are empty')
    names: set[str] = set()
    for row in rows:
        name = str(row.step_name).strip()
        if not name:
            raise ValueError('Process steps must have a non-empty step_name')
        if name in names:
            raise ValueError(f'Duplicate process step name: {name}')
        names.add(name)
        start_s = float(row.start_s)
        end_s = float(row.end_s)
        if not np.isfinite(start_s) or not np.isfinite(end_s):
            raise ValueError(f'Process step {name} must have finite start_s/end_s')
        if end_s <= start_s:
            raise ValueError(f'Process step {name} must have end_s > start_s')


def _validate_compiled_source_events(
    source_events: Optional[SourceEventTable],
    compiled_source_events: Optional[SourceEventTable],
) -> None:
    if source_events is None or compiled_source_events is None:
        return
    unresolved = []
    for raw_row, compiled_row in zip(source_events.rows, compiled_source_events.rows):
        if int(getattr(raw_row, 'enabled', 1)) == 0:
            continue
        status = str(getattr(compiled_row, 'metadata', {}).get('compile_status', '')).strip().lower()
        if status != 'unresolved_binding':
            continue
        bind_parts = []
        if getattr(raw_row, 'bind_step_name', ''):
            bind_parts.append(f"step={raw_row.bind_step_name}")
        bind_desc = ', '.join(bind_parts) if bind_parts else 'binding=absolute'
        unresolved.append(f"{raw_row.event_name} ({bind_desc})")
    if unresolved:
        preview = '; '.join(unresolved[:3])
        if len(unresolved) > 3:
            preview += f'; ... (+{len(unresolved) - 3} more)'
        raise ValueError(f'Unresolved source event bindings: {preview}')


def resolve_runtime_input_paths(config_dir: Path, paths_cfg: Mapping[str, Any]) -> ResolvedRuntimePaths:
    particles_path = _resolve_path(config_dir, paths_cfg.get('particles_csv'))
    materials_path = _resolve_path(config_dir, paths_cfg.get('materials_csv'))
    walls_path = _resolve_path(config_dir, paths_cfg.get('part_walls_csv'))
    events_path = _resolve_path(config_dir, paths_cfg.get('source_events_csv'))
    process_steps_path = _resolve_path(config_dir, paths_cfg.get('process_steps_csv'))
    if _resolve_path(config_dir, paths_cfg.get('recipe_manifest_yaml')) is not None:
        raise ValueError(
            'paths.recipe_manifest_yaml is no longer supported; '
            'use paths.process_steps_csv for optional time labels'
        )
    if particles_path is None:
        raise ValueError('paths.particles_csv is required')
    return ResolvedRuntimePaths(
        particles_path=particles_path,
        materials_path=materials_path,
        walls_path=walls_path,
        events_path=events_path,
        process_steps_path=process_steps_path,
    )


def load_runtime_inputs(
    *,
    paths: ResolvedRuntimePaths,
    spatial_dim: int,
    coordinate_system: str,
) -> LoadedRuntimeInputs:
    particles = load_particles_csv(paths.particles_path, spatial_dim=spatial_dim, coordinate_system=coordinate_system)
    materials = load_materials_csv(paths.materials_path) if paths.materials_path else None
    walls = load_part_walls_csv(paths.walls_path) if paths.walls_path else None
    source_events = load_source_events_csv(paths.events_path) if paths.events_path else None
    process_steps = None
    if paths.process_steps_path is not None:
        process_steps = load_process_steps_csv(paths.process_steps_path)
    _validate_process_steps(process_steps)
    compiled_source_events = compile_source_events(source_events, process_steps)
    _validate_compiled_source_events(source_events, compiled_source_events)
    return LoadedRuntimeInputs(
        particles=particles,
        materials=materials,
        walls=walls,
        source_events=source_events,
        process_steps=process_steps,
        compiled_source_events=compiled_source_events,
    )


def _resolved_provider_cfg(config_dir: Path, provider_cfg: Mapping[str, Any]) -> dict[str, Any]:
    resolved_cfg = dict(provider_cfg)
    resolved_npz = _resolve_path(config_dir, resolved_cfg.get('npz_path'))
    if resolved_npz is not None:
        resolved_cfg['npz_path'] = str(resolved_npz)
    return resolved_cfg


def _align_field_provider_to_geometry(
    field_provider: FieldProviderND,
    geometry_provider: GeometryProviderND,
) -> FieldProviderND:
    field = field_provider.field
    geom = geometry_provider.geometry
    if int(field.spatial_dim) != int(geom.spatial_dim):
        raise ValueError('Field spatial_dim must match geometry spatial_dim')
    field_kind = str(getattr(field_provider, 'kind', '')).strip().lower()
    if field_kind == 'precomputed_triangle_mesh_npz':
        return field_provider
    if any(a.shape != b.shape or not np.allclose(a, b, atol=1e-12, rtol=0.0) for a, b in zip(field.axes, geom.axes)):
        raise ValueError('Field axes must exactly match geometry axes')
    field_valid_mask = np.asarray(field.valid_mask, dtype=bool)
    geometry_valid_mask = np.asarray(geom.valid_mask, dtype=bool)
    core_valid_mask = field_valid_mask & geometry_valid_mask
    support_phi = getattr(field, 'support_phi', None)
    if support_phi is not None:
        support_phi_arr = np.asarray(support_phi, dtype=np.float64)
        if support_phi_arr.shape != field_valid_mask.shape:
            raise ValueError(f'Field support_phi shape mismatch: expected {field_valid_mask.shape}, got {support_phi_arr.shape}')
        support_phi = support_phi_arr
    aligned_field = replace(
        field,
        valid_mask=field_valid_mask,
        support_phi=support_phi,
        core_valid_mask=core_valid_mask,
        metadata={
            **field.metadata,
            'field_valid_mask_is_provider_native': True,
            'field_valid_node_count': int(np.count_nonzero(field_valid_mask)),
            'geometry_valid_node_count': int(np.count_nonzero(geometry_valid_mask)),
            'core_valid_node_count': int(np.count_nonzero(core_valid_mask)),
        },
    )
    return replace(field_provider, field=aligned_field)


def _build_geometry_provider(
    config_dir: Path,
    geom_cfg: Mapping[str, Any],
    *,
    spatial_dim: int,
    coordinate_system: str,
) -> GeometryProviderND:
    geom_kind = str(geom_cfg.get('kind', 'box')).strip().lower()
    resolved_cfg = _resolved_provider_cfg(config_dir, geom_cfg)
    if geom_kind in {'precomputed_npz', 'npz'}:
        return build_precomputed_geometry(resolved_cfg, spatial_dim=spatial_dim, coordinate_system=coordinate_system)
    return build_synthetic_geometry(resolved_cfg, spatial_dim=spatial_dim, coordinate_system=coordinate_system)


def _build_field_provider(
    config_dir: Path,
    field_cfg: Mapping[str, Any],
    geometry_provider: GeometryProviderND,
    *,
    spatial_dim: int,
    coordinate_system: str,
    gas_density_kgm3: float,
) -> FieldProviderND:
    field_kind = str(field_cfg.get('kind', 'linear_shear')).strip().lower()
    resolved_cfg = _resolved_provider_cfg(config_dir, field_cfg)
    if field_kind in {'precomputed_npz', 'npz'}:
        return build_precomputed_field(
            resolved_cfg,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
            axes=geometry_provider.geometry.axes,
            gas_density_kgm3=float(gas_density_kgm3),
        )
    if field_kind == 'precomputed_triangle_mesh_npz':
        return build_precomputed_triangle_mesh_field(
            resolved_cfg,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
            gas_density_kgm3=float(gas_density_kgm3),
        )
    return build_synthetic_field(
        resolved_cfg,
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
        axes=geometry_provider.geometry.axes,
        gas_density_kgm3=float(gas_density_kgm3),
    )


def build_runtime_providers(
    *,
    config_dir: Path,
    providers_cfg: Mapping[str, Any],
    spatial_dim: int,
    coordinate_system: str,
    gas_density_kgm3: float,
) -> RuntimeProviders:
    geom_cfg = providers_cfg.get('geometry', {}) if isinstance(providers_cfg.get('geometry', {}), Mapping) else {}
    field_cfg = providers_cfg.get('field', {}) if isinstance(providers_cfg.get('field', {}), Mapping) else {}

    geometry_provider = (
        _build_geometry_provider(config_dir, geom_cfg, spatial_dim=spatial_dim, coordinate_system=coordinate_system)
        if geom_cfg
        else None
    )
    field_provider = None
    if field_cfg:
        if geometry_provider is None:
            raise ValueError('providers.field requires providers.geometry so axes are available')
        field_provider = _build_field_provider(
            config_dir,
            field_cfg,
            geometry_provider,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
            gas_density_kgm3=float(gas_density_kgm3),
        )
    if geometry_provider is not None and bool(geometry_provider.geometry.metadata.get('requires_field_bundle', False)) and field_provider is None:
        raise ValueError('COMSOL geometry requires providers.field from a validated export bundle')
    if geometry_provider is not None and field_provider is not None:
        field_provider = _align_field_provider_to_geometry(field_provider, geometry_provider)
    return RuntimeProviders(geometry_provider=geometry_provider, field_provider=field_provider)


__all__ = (
    'LoadedRuntimeInputs',
    'ResolvedRuntimePaths',
    'RuntimeProviders',
    'build_runtime_providers',
    'load_runtime_inputs',
    'resolve_runtime_input_paths',
)
