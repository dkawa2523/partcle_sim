from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class QuantitySeriesND:
    name: str
    unit: str
    times: np.ndarray
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegularFieldND:
    spatial_dim: int
    coordinate_system: str
    axis_names: Tuple[str, ...]
    axes: Tuple[np.ndarray, ...]
    quantities: Dict[str, QuantitySeriesND]
    valid_mask: np.ndarray
    support_phi: Optional[np.ndarray] = None
    core_valid_mask: Optional[np.ndarray] = None
    time_mode: str = 'steady'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TriangleMeshField2D:
    spatial_dim: int
    coordinate_system: str
    mesh_vertices: np.ndarray
    mesh_triangles: np.ndarray
    quantities: Dict[str, QuantitySeriesND]
    accel_origin: np.ndarray
    accel_cell_size: np.ndarray
    accel_shape: Tuple[int, int]
    accel_cell_offsets: np.ndarray
    accel_triangle_indices: np.ndarray
    time_mode: str = 'steady'
    metadata: Dict[str, Any] = field(default_factory=dict)


FieldDataND = RegularFieldND | TriangleMeshField2D


@dataclass(frozen=True)
class FieldProviderND:
    field: FieldDataND
    manifest_path: Optional[Path] = None
    kind: str = 'regular_rectilinear'

    def summary(self) -> Dict[str, Any]:
        field_obj = self.field
        if isinstance(field_obj, TriangleMeshField2D):
            return {
                'kind': self.kind,
                'field_backend_kind': str(field_obj.metadata.get('field_backend_kind', 'triangle_mesh_2d')),
                'spatial_dim': int(field_obj.spatial_dim),
                'coordinate_system': field_obj.coordinate_system,
                'mesh_vertex_count': int(field_obj.mesh_vertices.shape[0]),
                'mesh_triangle_count': int(field_obj.mesh_triangles.shape[0]),
                'quantities': sorted(field_obj.quantities.keys()),
                'time_mode': field_obj.time_mode,
                'manifest_path': str(self.manifest_path) if self.manifest_path else '',
            }
        return {
            'kind': self.kind,
            'field_backend_kind': str(field_obj.metadata.get('field_backend_kind', 'regular_rectilinear')),
            'spatial_dim': int(field_obj.spatial_dim),
            'coordinate_system': field_obj.coordinate_system,
            'axis_names': list(field_obj.axis_names),
            'grid_shape': list(field_obj.valid_mask.shape),
            'has_support_phi': field_obj.support_phi is not None,
            'quantities': sorted(field_obj.quantities.keys()),
            'time_mode': field_obj.time_mode,
            'manifest_path': str(self.manifest_path) if self.manifest_path else '',
        }


@dataclass(frozen=True)
class GeometryND:
    spatial_dim: int
    coordinate_system: str
    axes: Tuple[np.ndarray, ...]
    valid_mask: np.ndarray
    sdf: np.ndarray
    normal_components: Tuple[np.ndarray, ...]
    nearest_boundary_part_id_map: np.ndarray
    source_kind: str = 'synthetic'
    metadata: Dict[str, Any] = field(default_factory=dict)
    boundary_edges: Optional[np.ndarray] = None
    boundary_edge_part_ids: Optional[np.ndarray] = None
    boundary_loops_2d: Tuple[np.ndarray, ...] = ()
    boundary_triangles: Optional[np.ndarray] = None
    boundary_triangle_part_ids: Optional[np.ndarray] = None

    @property
    def part_id_map(self) -> np.ndarray:
        return self.nearest_boundary_part_id_map


@dataclass(frozen=True)
class GeometryProviderND:
    geometry: GeometryND
    mphtxt_path: Optional[Path] = None
    kind: str = 'synthetic'

    def summary(self) -> Dict[str, Any]:
        g = self.geometry
        return {
            'kind': self.kind,
            'spatial_dim': int(g.spatial_dim),
            'coordinate_system': g.coordinate_system,
            'source_kind': g.source_kind,
            'grid_shape': list(g.valid_mask.shape),
            'has_boundary_edges': g.boundary_edges is not None,
            'has_boundary_loops_2d': bool(g.boundary_loops_2d),
            'has_boundary_triangles': g.boundary_triangles is not None,
            'has_domain_region_map': bool(g.metadata.get('has_domain_region_map', False)),
            'mphtxt_path': str(self.mphtxt_path) if self.mphtxt_path else '',
        }


@dataclass(frozen=True)
class ParticleTable:
    spatial_dim: int
    particle_id: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    release_time: np.ndarray
    mass: np.ndarray
    diameter: np.ndarray
    density: np.ndarray
    charge: np.ndarray
    source_part_id: np.ndarray
    material_id: np.ndarray
    source_event_tag: np.ndarray
    source_law_override: np.ndarray
    source_speed_scale_override: np.ndarray
    stick_probability: np.ndarray
    dep_particle_rel_permittivity: np.ndarray
    thermophoretic_coeff: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.particle_id.size)


@dataclass(frozen=True)
class MaterialRow:
    """Material-scoped defaults in source -> wall -> physics order."""

    material_id: int
    material_name: str

    # Source defaults. Keep this block aligned with core.source_schema.
    source_law: str = 'explicit_csv'
    source_speed_scale: float = np.nan
    source_position_offset_m: float = np.nan
    source_temperature_K: float = np.nan
    source_normal_speed_mean_mps: float = np.nan
    source_normal_speed_std_mps: float = np.nan
    source_tangent_speed_std_mps: float = np.nan
    source_resuspension_velocity_scale: float = np.nan
    source_resuspension_normal_speed_mean_mps: float = np.nan
    source_resuspension_normal_speed_std_mps: float = np.nan
    source_resuspension_speed_threshold_mps: float = np.nan
    source_resuspension_tau_threshold_Pa: float = np.nan
    source_resuspension_utau_threshold_mps: float = np.nan
    source_resuspension_shear_length_m: float = np.nan
    source_dynamic_viscosity_Pas: float = np.nan
    source_thermal_accommodation: float = np.nan
    source_flake_weight: float = np.nan
    source_reflectivity: float = np.nan
    source_roughness_rms: float = np.nan
    source_roughness_corr_length_m: float = np.nan
    source_roughness_slope_rms: float = np.nan
    source_adhesion_energy_Jm2: float = np.nan
    source_resuspension_roughness_scale: float = np.nan
    source_resuspension_adhesion_scale: float = np.nan
    source_resuspension_tau_roughness_scale: float = np.nan
    source_resuspension_tau_adhesion_scale: float = np.nan
    source_resuspension_tau_slope_scale: float = np.nan
    source_burst_center_s: float = np.nan
    source_burst_sigma_s: float = np.nan
    source_burst_amplitude: float = np.nan
    source_burst_period_s: float = np.nan
    source_burst_phase_s: float = np.nan
    source_burst_min_factor: float = np.nan
    source_burst_max_factor: float = np.nan
    source_default_event_tag: str = ''

    # Wall defaults resolved after source selection.
    wall_law: str = ''
    wall_stick_probability: float = np.nan
    wall_restitution: float = np.nan
    wall_diffuse_fraction: float = np.nan
    wall_critical_sticking_velocity_mps: float = np.nan
    wall_reflectivity: float = np.nan
    wall_roughness_rms: float = np.nan

    # Optional physics scaling defaults.
    physics_flow_scale: float = np.nan
    physics_drag_tau_scale: float = np.nan
    physics_body_accel_scale: float = np.nan


@dataclass(frozen=True)
class MaterialTable:
    rows: Tuple[MaterialRow, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_lookup(self) -> Dict[int, MaterialRow]:
        return {int(r.material_id): r for r in self.rows}


@dataclass(frozen=True)
class PartWallRow:
    """Part-scoped overrides in the same source -> wall -> physics order."""

    part_id: int
    part_name: str
    material_id: int = 0
    material_name: str = ''

    # Source overrides. Keep this block aligned with MaterialRow and source_schema.
    source_law: str = ''
    source_speed_scale: float = np.nan
    source_position_offset_m: float = np.nan
    source_temperature_K: float = np.nan
    source_normal_speed_mean_mps: float = np.nan
    source_normal_speed_std_mps: float = np.nan
    source_tangent_speed_std_mps: float = np.nan
    source_resuspension_velocity_scale: float = np.nan
    source_resuspension_normal_speed_mean_mps: float = np.nan
    source_resuspension_normal_speed_std_mps: float = np.nan
    source_resuspension_speed_threshold_mps: float = np.nan
    source_resuspension_tau_threshold_Pa: float = np.nan
    source_resuspension_utau_threshold_mps: float = np.nan
    source_resuspension_shear_length_m: float = np.nan
    source_dynamic_viscosity_Pas: float = np.nan
    source_thermal_accommodation: float = np.nan
    source_flake_weight: float = np.nan
    source_reflectivity: float = np.nan
    source_roughness_rms: float = np.nan
    source_roughness_corr_length_m: float = np.nan
    source_roughness_slope_rms: float = np.nan
    source_adhesion_energy_Jm2: float = np.nan
    source_resuspension_roughness_scale: float = np.nan
    source_resuspension_adhesion_scale: float = np.nan
    source_resuspension_tau_roughness_scale: float = np.nan
    source_resuspension_tau_adhesion_scale: float = np.nan
    source_resuspension_tau_slope_scale: float = np.nan
    source_burst_center_s: float = np.nan
    source_burst_sigma_s: float = np.nan
    source_burst_amplitude: float = np.nan
    source_burst_period_s: float = np.nan
    source_burst_phase_s: float = np.nan
    source_burst_min_factor: float = np.nan
    source_burst_max_factor: float = np.nan
    source_default_event_tag: str = ''

    # Wall overrides resolved after source selection.
    wall_law: str = ''
    wall_stick_probability: float = np.nan
    wall_restitution: float = np.nan
    wall_diffuse_fraction: float = np.nan
    wall_critical_sticking_velocity_mps: float = np.nan
    wall_reflectivity: float = np.nan
    wall_roughness_rms: float = np.nan

    # Optional physics scaling overrides.
    physics_flow_scale: float = np.nan
    physics_drag_tau_scale: float = np.nan
    physics_body_accel_scale: float = np.nan


@dataclass(frozen=True)
class PartWallTable:
    rows: Tuple[PartWallRow, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_lookup(self) -> Dict[int, PartWallRow]:
        return {int(r.part_id): r for r in self.rows}


@dataclass(frozen=True)
class ProcessStepRow:
    step_id: int
    step_name: str
    start_s: float
    end_s: float
    output_segment_name: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s)

    def contains_time(self, t: float) -> bool:
        return float(self.start_s) <= float(t) < float(self.end_s)


@dataclass(frozen=True)
class ProcessStepTable:
    rows: Tuple[ProcessStepRow, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_name_lookup(self) -> Dict[str, ProcessStepRow]:
        return {str(r.step_name): r for r in self.rows}

    def active_at(self, t: float) -> Optional[ProcessStepRow]:
        tt = float(t)
        for row in self.rows:
            if row.contains_time(tt):
                return row
        return None


@dataclass(frozen=True)
class SourceEventRow:
    event_id: int
    event_name: str
    event_kind: str = 'gaussian_burst'
    enabled: int = 1
    applies_to_particle_id: int = 0
    applies_to_source_part_id: int = 0
    applies_to_material_id: int = 0
    applies_to_source_law: str = ''
    applies_to_event_tag: str = ''
    center_s: float = np.nan
    sigma_s: float = np.nan
    amplitude: float = np.nan
    period_s: float = np.nan
    phase_s: float = np.nan
    start_s: float = np.nan
    end_s: float = np.nan
    gain_multiplier: float = np.nan
    release_time_shift_s: float = np.nan
    min_factor: float = np.nan
    max_factor: float = np.nan
    bind_step_name: str = ''
    time_anchor: str = 'absolute'
    time_offset_s: float = np.nan
    duration_s: float = np.nan
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceEventTable:
    rows: Tuple[SourceEventRow, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def active_rows(self) -> Tuple[SourceEventRow, ...]:
        return tuple(r for r in self.rows if int(r.enabled) != 0)


@dataclass(frozen=True)
class SourceResolutionParameters:
    """Resolved per-particle source/physics arrays in schema order."""

    # Identity and labels resolved from particle -> wall -> material -> global defaults.
    resolved_material_id: np.ndarray
    source_material_id: np.ndarray
    resolved_law_name: Tuple[str, ...]
    resolved_law_code: np.ndarray
    resolved_event_tag: np.ndarray

    # Scalar arrays. Keep this block aligned with core.source_schema.
    source_speed_scale: np.ndarray
    source_position_offset_m: np.ndarray
    source_temperature_K: np.ndarray
    source_normal_speed_mean_mps: np.ndarray
    source_normal_speed_std_mps: np.ndarray
    source_tangent_speed_std_mps: np.ndarray
    source_resuspension_velocity_scale: np.ndarray
    source_resuspension_normal_speed_mean_mps: np.ndarray
    source_resuspension_normal_speed_std_mps: np.ndarray
    source_resuspension_speed_threshold_mps: np.ndarray
    source_resuspension_tau_threshold_Pa: np.ndarray
    source_resuspension_utau_threshold_mps: np.ndarray
    source_resuspension_shear_length_m: np.ndarray
    source_dynamic_viscosity_Pas: np.ndarray
    source_thermal_accommodation: np.ndarray
    source_flake_weight: np.ndarray
    source_reflectivity: np.ndarray
    source_roughness_rms: np.ndarray
    source_roughness_corr_length_m: np.ndarray
    source_roughness_slope_rms: np.ndarray
    source_adhesion_energy_Jm2: np.ndarray
    source_resuspension_roughness_scale: np.ndarray
    source_resuspension_adhesion_scale: np.ndarray
    source_resuspension_tau_roughness_scale: np.ndarray
    source_resuspension_tau_adhesion_scale: np.ndarray
    source_resuspension_tau_slope_scale: np.ndarray
    source_burst_center_s: np.ndarray
    source_burst_sigma_s: np.ndarray
    source_burst_amplitude: np.ndarray
    source_burst_period_s: np.ndarray
    source_burst_phase_s: np.ndarray
    source_burst_min_factor: np.ndarray
    source_burst_max_factor: np.ndarray
    physics_flow_scale: np.ndarray
    physics_drag_tau_scale: np.ndarray
    physics_body_accel_scale: np.ndarray

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourcePreprocessResult:
    particles: ParticleTable
    resolved: SourceResolutionParameters
    source_model_summary: Dict[str, Any]
    diagnostics_rows: Tuple[Dict[str, Any], ...]
    release_enabled: np.ndarray
    event_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GasProperties:
    temperature: float = 300.0
    dynamic_viscosity_Pas: float = 1.8e-5
    density_kgm3: float = 1.0
    molecular_mass_amu: float = 60.0


@dataclass(frozen=True)
class WallPartModel:
    part_id: int
    part_name: str
    material_id: int
    material_name: str
    law_name: str
    stick_probability: float
    restitution: float
    diffuse_fraction: float
    critical_sticking_velocity_mps: float
    reflectivity: float
    roughness_rms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WallCatalog:
    default_model: WallPartModel
    part_models: Tuple[WallPartModel, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _part_lookup: Dict[int, WallPartModel] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_part_lookup', {int(r.part_id): r for r in self.part_models})

    def as_lookup(self) -> Dict[int, WallPartModel]:
        return dict(self._part_lookup)

    def model_for_part(self, part_id: int) -> WallPartModel:
        return self._part_lookup.get(int(part_id), self.default_model)


@dataclass(frozen=True)
class PhysicsCatalog:
    base_flow_scale: float = 1.0
    base_drag_tau_scale: float = 1.0
    base_body_accel_scale: float = 1.0
    integrator: str = 'drag_relaxation'
    min_tau_p_s: float = 1e-6
    body_acceleration: Tuple[float, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeLike:
    spatial_dim: int
    coordinate_system: str
    particles: Optional[ParticleTable]
    walls: Optional[PartWallTable]
    materials: Optional[MaterialTable]
    source_events: Optional[SourceEventTable]
    process_steps: Optional[ProcessStepTable]
    compiled_source_events: Optional[SourceEventTable]
    geometry_provider: Optional[GeometryProviderND]
    field_provider: Optional[FieldProviderND]
    gas: GasProperties
    time_interpolation: str = 'linear'
    config_payload: Mapping[str, Any] = field(default_factory=dict)
    source_preprocess: Optional[SourcePreprocessResult] = None
    wall_catalog: Optional[WallCatalog] = None
    physics_catalog: Optional[PhysicsCatalog] = None
    force_catalog: Optional[Any] = None


@dataclass(frozen=True)
class PreparedRuntime:
    runtime: RuntimeLike
    source_preprocess: Optional[SourcePreprocessResult] = None


def replace_runtime_particles(
    runtime: RuntimeLike,
    particles: ParticleTable,
    source_preprocess: Optional[SourcePreprocessResult] = None,
    compiled_source_events: Optional[SourceEventTable] = None,
) -> RuntimeLike:
    kwargs: Dict[str, Any] = {'particles': particles}
    if source_preprocess is not None:
        kwargs['source_preprocess'] = source_preprocess
    if compiled_source_events is not None:
        kwargs['compiled_source_events'] = compiled_source_events
    return replace(runtime, **kwargs)
