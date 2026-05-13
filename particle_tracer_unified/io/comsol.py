from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .comsol_boundary_reader import (
    read_comsol_boundary_map,
    read_comsol_wall_laws,
    validate_wall_law_coverage,
    wall_laws_to_tables,
)
from .comsol_manifest import ComsolCaseManifest, ComsolFieldSpec, is_comsol_faithful_config
from .comsol_release_reader import (
    ComsolReleaseParticle,
    comsol_release_particles_to_particle_table,
    read_comsol_release_particles,
)
from .runtime_builder_support import LoadedRuntimeInputs, load_optional_source_timing
from ..solvers.forces import apply_manifest_force_inventory_to_solver_config


def _resolve_path(base: Path, value: Any) -> Path | None:
    if value is None or str(value).strip() == '':
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _nested_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _bool_config(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'1', 'true', 'yes', 'on'}:
            return True
        if text in {'0', 'false', 'no', 'off'}:
            return False
    return bool(value)


def load_comsol_manifest_for_config(
    config: Mapping[str, Any],
    config_dir: Path,
) -> ComsolCaseManifest | None:
    if not is_comsol_faithful_config(config):
        return None
    comsol_cfg = _nested_mapping(config, 'comsol')
    manifest_raw = comsol_cfg.get('manifest')
    if manifest_raw is None or str(manifest_raw).strip() == '':
        raise ValueError('comsol.manifest is required when mode is comsol_faithful')
    manifest_path = _resolve_path(config_dir, manifest_raw)
    if manifest_path is None:
        raise ValueError('comsol.manifest is required when mode is comsol_faithful')
    manifest = ComsolCaseManifest.load(manifest_path)
    manifest.validate(strict=True)
    return manifest


def enforce_comsol_faithful_config(config: dict[str, Any], manifest: ComsolCaseManifest) -> None:
    source_cfg = _nested_mapping(config, 'source')
    preprocess_cfg = source_cfg.get('preprocess', {}) if isinstance(source_cfg.get('preprocess', {}), Mapping) else {}
    if _bool_config(preprocess_cfg.get('enabled', False), default=False):
        raise ValueError('source.preprocess.enabled must be false in COMSOL faithful mode')

    field_support = config.setdefault('field_support', {})
    if not isinstance(field_support, dict):
        raise ValueError('field_support must be a mapping in COMSOL faithful mode')
    ghost_cfg = field_support.get('ghost_cells', {})
    if isinstance(ghost_cfg, Mapping) and _bool_config(ghost_cfg.get('enabled', False), default=False):
        raise ValueError('field_support.ghost_cells.enabled must be false in COMSOL faithful mode')
    mixed_policy = str(field_support.get('mixed_stencil_policy', 'error')).strip().lower()
    if mixed_policy != 'error':
        raise ValueError('field_support.mixed_stencil_policy must be error in COMSOL faithful mode')
    field_support['mixed_stencil_policy'] = 'error'

    solver = config.setdefault('solver', {})
    if not isinstance(solver, dict):
        raise ValueError('solver must be a mapping in COMSOL faithful mode')
    solver.setdefault('valid_mask_policy', 'strict_clean')
    if str(solver.get('valid_mask_policy', '')).strip().lower() != 'strict_clean':
        raise ValueError('solver.valid_mask_policy must be strict_clean in COMSOL faithful mode')
    apply_manifest_force_inventory_to_solver_config(solver, manifest.forces)


def finalize_comsol_runtime_config(
    config: dict[str, Any],
    manifest: ComsolCaseManifest,
    field_provider: Any,
) -> None:
    if field_provider is not None:
        field_metadata = getattr(field_provider.field, 'metadata', {})
        if int(field_metadata.get('field_ghost_cells', 0) or 0) != 0:
            raise ValueError('COMSOL faithful mode requires field bundles without ghost cells')
    comsol_cfg = config.setdefault('comsol', {})
    if not isinstance(comsol_cfg, dict):
        raise ValueError('comsol must be a mapping in COMSOL faithful mode')
    comsol_cfg['manifest_root_dir'] = str(manifest.root_dir)
    comsol_cfg['coordinate_scale_m_per_model_unit'] = float(manifest.coordinate_scale_m_per_model_unit)
    comsol_cfg['release_velocity_scale_mps_per_input_unit'] = float(
        manifest.release_velocity_scale_mps_per_input_unit
    )


def load_comsol_runtime_inputs(
    *,
    config: Mapping[str, Any],
    config_dir: Path,
    manifest: ComsolCaseManifest,
    spatial_dim: int,
) -> LoadedRuntimeInputs:
    release_path = manifest.resolve(manifest.particles.get('release_table'))
    boundary_map_path = manifest.resolve(manifest.boundaries.get('map_file'))
    wall_law_path = manifest.resolve(manifest.boundaries.get('wall_law_file'))
    if release_path is None or boundary_map_path is None or wall_law_path is None:
        raise ValueError('COMSOL manifest release/boundary/wall paths must be present')

    release_particles = read_comsol_release_particles(
        release_path,
        coordinate_scale_m_per_model_unit=manifest.coordinate_scale_m_per_model_unit,
        release_velocity_scale_mps_per_input_unit=manifest.release_velocity_scale_mps_per_input_unit,
        spatial_dim=int(spatial_dim),
        strict=True,
    )
    particles = comsol_release_particles_to_particle_table(
        release_particles,
        spatial_dim=int(spatial_dim),
        metadata={
            'source': 'comsol_release_table',
            'path': str(release_path),
            'coordinate_scale_m_per_model_unit': float(manifest.coordinate_scale_m_per_model_unit),
            'release_velocity_scale_mps_per_input_unit': float(
                manifest.release_velocity_scale_mps_per_input_unit
            ),
        },
    )
    boundary_rows = read_comsol_boundary_map(boundary_map_path, strict=True)
    wall_rows = read_comsol_wall_laws(wall_law_path, strict=True)
    validate_wall_law_coverage(boundary_rows, wall_rows)
    materials, walls = wall_laws_to_tables(wall_rows, boundary_rows)

    paths = _nested_mapping(config, 'paths')
    events_path = _resolve_path(config_dir, paths.get('source_events_csv'))
    process_steps_path = _resolve_path(config_dir, paths.get('process_steps_csv'))
    source_events, process_steps, compiled_source_events = load_optional_source_timing(
        events_path=events_path,
        process_steps_path=process_steps_path,
    )

    return LoadedRuntimeInputs(
        particles=particles,
        materials=materials,
        walls=walls,
        source_events=source_events,
        process_steps=process_steps,
        compiled_source_events=compiled_source_events,
    )


__all__ = [
    'ComsolCaseManifest',
    'ComsolFieldSpec',
    'ComsolReleaseParticle',
    'comsol_release_particles_to_particle_table',
    'enforce_comsol_faithful_config',
    'finalize_comsol_runtime_config',
    'is_comsol_faithful_config',
    'load_comsol_manifest_for_config',
    'load_comsol_runtime_inputs',
    'read_comsol_boundary_map',
    'read_comsol_release_particles',
    'read_comsol_wall_laws',
    'validate_wall_law_coverage',
    'wall_laws_to_tables',
]
