from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .datamodel import (
    MaterialRow,
    MaterialTable,
    PartWallRow,
    PartWallTable,
    PhysicsCatalog,
    ProcessStepRow,
    WallCatalog,
    WallPartModel,
)
from .integrator_registry import validate_integrator_name
from .source_material_common import pick_float, pick_str


def _body_acceleration_from_solver(solver_cfg: Mapping[str, Any], spatial_dim: int) -> Tuple[float, ...]:
    if 'body_acceleration' in solver_cfg and isinstance(solver_cfg['body_acceleration'], (list, tuple)):
        vals = [float(v) for v in solver_cfg['body_acceleration']]
        if len(vals) >= spatial_dim:
            return tuple(vals[:spatial_dim])
    if 'gravity_mps2' in solver_cfg:
        g = float(solver_cfg.get('gravity_mps2', 9.81))
        arr = [0.0] * spatial_dim
        arr[-1] = -g
        return tuple(arr)
    return tuple([0.0] * spatial_dim)


def _validated_integrator(solver_cfg: Mapping[str, Any]) -> str:
    return validate_integrator_name(solver_cfg.get('integrator', 'drag_relaxation'))


def build_physics_catalog(config: Mapping[str, Any], spatial_dim: int) -> PhysicsCatalog:
    solver_cfg = config.get('solver', {}) if isinstance(config.get('solver', {}), Mapping) else {}
    return PhysicsCatalog(
        base_flow_scale=1.0,
        base_drag_tau_scale=1.0,
        base_body_accel_scale=1.0,
        integrator=_validated_integrator(solver_cfg),
        min_tau_p_s=float(solver_cfg.get('min_tau_p_s', 1e-6)),
        body_acceleration=_body_acceleration_from_solver(solver_cfg, spatial_dim),
        metadata={'from_solver_cfg': True, 'spatial_dim': int(spatial_dim)},
    )


def build_wall_catalog(walls: Optional[PartWallTable], materials: Optional[MaterialTable], config: Mapping[str, Any]) -> WallCatalog:
    wall_cfg = config.get('wall', {}) if isinstance(config.get('wall', {}), Mapping) else {}
    materials_lu = materials.as_lookup() if materials is not None else {}
    walls_lu = walls.as_lookup() if walls is not None else {}

    def model_from_rows(part_id: int, part_name: str, wall_row: Optional[PartWallRow], mat_row: Optional[MaterialRow]) -> WallPartModel:
        law_name = pick_str(
            getattr(wall_row, 'wall_law', None) if wall_row else None,
            getattr(mat_row, 'wall_law', None) if mat_row else None,
            wall_cfg.get('default_mode', wall_cfg.get('mode', 'specular')),
            default='specular',
        )
        stick_probability = pick_float(
            getattr(wall_row, 'wall_stick_probability', np.nan) if wall_row else np.nan,
            getattr(mat_row, 'wall_stick_probability', np.nan) if mat_row else np.nan,
            wall_cfg.get('stick_probability', wall_cfg.get('default_stick_probability', 0.0)),
            default=0.0,
        )
        restitution = pick_float(
            getattr(wall_row, 'wall_restitution', np.nan) if wall_row else np.nan,
            getattr(mat_row, 'wall_restitution', np.nan) if mat_row else np.nan,
            wall_cfg.get('restitution', 1.0),
            default=1.0,
        )
        diffuse_fraction = pick_float(
            getattr(wall_row, 'wall_diffuse_fraction', np.nan) if wall_row else np.nan,
            getattr(mat_row, 'wall_diffuse_fraction', np.nan) if mat_row else np.nan,
            wall_cfg.get('diffuse_fraction', 0.0),
            default=0.0,
        )
        vcrit = pick_float(
            getattr(wall_row, 'wall_critical_sticking_velocity_mps', np.nan) if wall_row else np.nan,
            getattr(mat_row, 'wall_critical_sticking_velocity_mps', np.nan) if mat_row else np.nan,
            wall_cfg.get('critical_sticking_velocity_mps', 0.0),
            default=0.0,
        )
        reflectivity = pick_float(
            getattr(wall_row, 'wall_reflectivity', np.nan) if wall_row else np.nan,
            getattr(mat_row, 'wall_reflectivity', np.nan) if mat_row else np.nan,
            0.0,
            default=0.0,
        )
        roughness = pick_float(
            getattr(wall_row, 'wall_roughness_rms', np.nan) if wall_row else np.nan,
            getattr(mat_row, 'wall_roughness_rms', np.nan) if mat_row else np.nan,
            0.0,
            default=0.0,
        )
        material_id = int(getattr(wall_row, 'material_id', 0) if wall_row else (getattr(mat_row, 'material_id', 0) if mat_row else 0))
        material_name = pick_str(
            getattr(wall_row, 'material_name', None) if wall_row else None,
            getattr(mat_row, 'material_name', None) if mat_row else None,
            default='',
        )
        return WallPartModel(
            part_id=int(part_id),
            part_name=str(part_name),
            material_id=material_id,
            material_name=material_name,
            law_name=law_name,
            stick_probability=float(np.clip(stick_probability, 0.0, 1.0)),
            restitution=float(max(restitution, 0.0)),
            diffuse_fraction=float(np.clip(diffuse_fraction, 0.0, 1.0)),
            critical_sticking_velocity_mps=float(max(vcrit, 0.0)),
            reflectivity=float(np.clip(reflectivity, 0.0, 1.0)),
            roughness_rms=float(max(roughness, 0.0)),
            metadata={'resolved_from_material': mat_row.material_name if mat_row else '', 'resolved_from_part': part_name},
        )

    default_model = model_from_rows(0, 'default', None, None)
    part_models = []
    for part_id, wall_row in sorted(walls_lu.items(), key=lambda kv: kv[0]):
        mat_row = materials_lu.get(int(wall_row.material_id)) if int(getattr(wall_row, 'material_id', 0)) > 0 else None
        part_models.append(model_from_rows(int(part_id), str(wall_row.part_name), wall_row, mat_row))
    return WallCatalog(default_model=default_model, part_models=tuple(part_models), metadata={'wall_part_count': len(part_models)})


def resolve_step_physics(physics_catalog: Optional[PhysicsCatalog], step: Optional[ProcessStepRow]) -> Dict[str, Any]:
    if physics_catalog is None:
        base_flow_scale = 1.0
        base_drag_tau_scale = 1.0
        base_body_accel_scale = 1.0
        body_accel = np.zeros(3, dtype=np.float64)
        integrator = 'drag_relaxation'
        min_tau = 1e-6
    else:
        base_flow_scale = float(physics_catalog.base_flow_scale)
        base_drag_tau_scale = float(physics_catalog.base_drag_tau_scale)
        base_body_accel_scale = float(physics_catalog.base_body_accel_scale)
        body_accel = np.asarray(physics_catalog.body_acceleration, dtype=np.float64)
        integrator = physics_catalog.integrator
        min_tau = float(physics_catalog.min_tau_p_s)
    return {
        'flow_scale': base_flow_scale,
        'drag_tau_scale': base_drag_tau_scale,
        'body_accel_scale': base_body_accel_scale,
        'body_acceleration': body_accel,
        'integrator': integrator,
        'min_tau_p_s': min_tau,
    }


def resolve_step_wall_model(wall_catalog: Optional[WallCatalog], part_id: int, step: Optional[ProcessStepRow]) -> WallPartModel:
    if wall_catalog is None:
        base = WallPartModel(part_id=int(part_id), part_name=f'part_{int(part_id)}', material_id=0, material_name='', law_name='specular', stick_probability=0.0, restitution=1.0, diffuse_fraction=0.0, critical_sticking_velocity_mps=0.0, reflectivity=0.0, roughness_rms=0.0, metadata={})
    else:
        base = wall_catalog.model_for_part(int(part_id))
    return base


def wall_catalog_summary(wall_catalog: Optional[WallCatalog]) -> Dict[str, Any]:
    if wall_catalog is None:
        return {'has_wall_catalog': False, 'wall_part_count': 0}
    return {
        'has_wall_catalog': True,
        'wall_part_count': len(wall_catalog.part_models),
        'default_law': wall_catalog.default_model.law_name,
        'part_laws': {str(m.part_id): m.law_name for m in wall_catalog.part_models},
    }


def physics_catalog_summary(physics_catalog: Optional[PhysicsCatalog]) -> Dict[str, Any]:
    if physics_catalog is None:
        return {'has_physics_catalog': False}
    return {
        'has_physics_catalog': True,
        'integrator': physics_catalog.integrator,
        'base_flow_scale': float(physics_catalog.base_flow_scale),
        'base_drag_tau_scale': float(physics_catalog.base_drag_tau_scale),
        'base_body_accel_scale': float(physics_catalog.base_body_accel_scale),
        'body_acceleration': list(map(float, physics_catalog.body_acceleration)),
        'min_tau_p_s': float(physics_catalog.min_tau_p_s),
    }
