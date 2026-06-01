from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from ..core.coordinate_systems import axis_names_for_coordinate_system, axisymmetric_rz_report_from_metadata
from ..core.datamodel import ParticleTable, PreparedRuntime, source_provenance_group
from ..core.field_backend import field_backend_report, sample_field_valid_status
from ..core.field_sampling import valid_mask_status_requires_stop
from ..core.boundary_service import (
    inside_geometry,
    nearest_boundary_edge_features_2d,
    sample_geometry_part_id,
    sample_geometry_sdf,
)
from ..core.source_materials import write_source_summary
from .wall_catalog_alignment import build_wall_catalog_alignment, write_wall_catalog_alignment_csv
from ..io.runtime_builder import prepared_runtime_summary
from .diagnostics import invalid_stop_reason_names
from .runtime_plan import (
    OUTPUT_MODE_DEBUG,
    OUTPUT_MODE_MINIMAL,
    OUTPUT_MODE_STANDARD,
    OutputPlan,
    normalize_output_mode,
)
from .runtime_reports import (
    build_scalar_summary_rows as _build_scalar_summary_rows,
    summary_float_or_nan as _summary_float_or_nan,
)

FINAL_STATE_ORDER = (
    'active_free_flight',
    'contact_sliding',
    'contact_endpoint_stopped',
    'invalid_mask_stopped',
    'numerical_boundary_stopped',
    'stuck',
    'absorbed',
    'escaped',
    'inactive',
)

GEOMETRY_STATE_ORDER = (
    'active',
    'contact_sliding',
    'invalid_mask_stopped',
    'numerical_boundary_stopped',
    'stuck',
    'absorbed',
    'escaped',
    'inactive',
)

EMPTY_FINITE_SUMMARY = {'count': 0}

WALL_LAW_SEMANTICS = {
    'pass_through': 'non_colliding_boundary',
    'passthrough': 'non_colliding_boundary',
    'transparent': 'non_colliding_boundary',
    'inactive': 'non_colliding_boundary',
    'continuity': 'non_colliding_boundary',
    'pair_continuity': 'non_colliding_boundary',
    'interior': 'non_colliding_boundary',
    'internal': 'non_colliding_boundary',
    'open': 'particle_exit',
    'outflow': 'particle_exit',
    'exhaust': 'particle_exit',
    'escape': 'particle_exit',
    'field_support_exit': 'particle_exit',
    'absorb': 'particle_absorbed',
    'disappear': 'particle_absorbed',
    'stick': 'particle_stuck',
    'sticking': 'particle_stuck',
    'critical_sticking_velocity': 'particle_stuck_below_threshold',
    'diffuse': 'diffuse_reflection',
    'mixed_specular_diffuse': 'mixed_reflection',
    'specular': 'specular_reflection',
}

FORCE_CONTRIBUTION_COLUMNS = [
    'name',
    'enabled',
    'model',
    'status',
    'physical_quantity',
    'required_fields',
    'optional_fields',
    'field_sources',
    'config',
]

ACCELERATION_CONTRIBUTION_FORCES = {
    'gravity',
    'electric',
    'thermophoresis',
    'dielectrophoresis',
    'lift',
    'pressure_gradient',
    'virtual_mass',
}

WALL_EVENT_COLUMNS = [
    'time_s',
    'hit_time_s',
    'particle_id',
    'part_id',
    'boundary_primitive_id',
    'boundary_primitive_kind',
    'boundary_hit_ambiguous',
    'step_name',
    'segment_name',
    'outcome',
    'wall_mode',
    'alpha_hit',
    'material_id',
    'material_name',
    'particle_mass_kg',
    'particle_diameter_m',
    'impact_speed_mps',
    'impact_normal_speed_mps',
    'impact_tangential_speed_mps',
    'impact_angle_deg_from_normal',
    'hit_x_m',
    'hit_y_m',
    'hit_z_m',
    'normal_x',
    'normal_y',
    'normal_z',
    'v_hit_x_mps',
    'v_hit_y_mps',
    'v_hit_z_mps',
]

WALL_SUMMARY_COLUMNS = ['part_id', 'outcome', 'wall_mode', 'count']
SEGMENT_SUMMARY_COLUMNS = ['segment_name', 'save_count', 't_start', 't_end']

MAX_HIT_EVENT_COLUMNS = [
    'time_s',
    'particle_id',
    'step_name',
    'segment_name',
    'hits_in_step',
    'remaining_dt_s',
    'last_part_id',
    'part_id_sequence',
    'outcome_sequence',
]

RUNTIME_STEP_COLUMNS = [
    'time_s',
    'step_name',
    'segment_name',
    'released_count',
    'active_count',
    'stuck_count',
    'absorbed_count',
    'contact_sliding_count',
    'escaped_count',
    'save_positions_enabled',
    'write_wall_events_enabled',
    'write_diagnostics_enabled',
    'valid_mask_violation_count_step',
    'valid_mask_mixed_stencil_count_step',
    'valid_mask_hard_invalid_count_step',
    'invalid_mask_stopped_count_step',
]

COATING_SUMMARY_COLUMNS = [
    'part_id',
    'material_id',
    'material_name',
    'impact_count',
    'stuck_count',
    'absorbed_count',
    'deposited_mass_kg',
    'mean_impact_speed_mps',
    'mean_impact_angle_deg_from_normal',
]

VALID_MASK_DIAGNOSTIC_FIELDS = (
    'valid_mask_violation_count',
    'valid_mask_violation_particle_count',
    'valid_mask_mixed_stencil_count',
    'valid_mask_mixed_stencil_particle_count',
    'valid_mask_hard_invalid_count',
    'valid_mask_hard_invalid_particle_count',
)

CONTACT_DIAGNOSTIC_FIELDS = (
    'contact_sliding_count',
    'contact_tangent_step_count',
    'contact_release_count',
    'contact_release_probe_reject_count',
    'contact_endpoint_stop_count',
    'contact_endpoint_hold_count',
    'invalid_mask_retry_count',
    'invalid_mask_retry_exhausted_count',
)

RELEASE_GRACE_DIAGNOSTIC_FIELDS = (
    'source_surface_release_grace_enabled',
    'source_surface_release_skip_count',
    'source_surface_release_skip_blocked_count',
)


def _comsol_boundary_map(payload: 'RuntimeOutputPayload') -> Dict[int, Mapping[str, Any]]:
    runtime = payload.prepared.runtime
    walls = getattr(runtime, 'walls', None)
    metadata = getattr(walls, 'metadata', {}) if walls is not None else {}
    rows = metadata.get('comsol_boundary_map', []) if isinstance(metadata, Mapping) else []
    out: Dict[int, Mapping[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            part_id = int(row.get('solver_part_id', row.get('part_id', 0)))
        except (TypeError, ValueError):
            continue
        if part_id:
            out[part_id] = row
    return out


def _attach_comsol_boundary_columns(df, payload: 'RuntimeOutputPayload'):
    boundary_map = _comsol_boundary_map(payload)
    if not boundary_map or 'part_id' not in df.columns:
        return df
    out = df.copy()
    out['comsol_geom_entity_id'] = [
        boundary_map.get(int(part_id), {}).get('comsol_geom_entity_id', '')
        for part_id in out['part_id'].fillna(0).astype(int)
    ]
    out['comsol_selection_name'] = [
        boundary_map.get(int(part_id), {}).get('selection_name', '')
        for part_id in out['part_id'].fillna(0).astype(int)
    ]
    out['comsol_boundary_type'] = [
        boundary_map.get(int(part_id), {}).get('boundary_type', '')
        for part_id in out['part_id'].fillna(0).astype(int)
    ]
    out['comsol_wall_node'] = [
        boundary_map.get(int(part_id), {}).get('wall_node', '')
        for part_id in out['part_id'].fillna(0).astype(int)
    ]
    return out


def _build_force_contribution_rows(payload: 'RuntimeOutputPayload') -> List[Dict[str, object]]:
    catalog = payload.prepared.runtime.force_catalog
    if catalog is None:
        return []
    rows: List[Dict[str, object]] = []
    for spec in catalog.specs:
        physical_quantity = str(spec.config.get('physical_quantity', '')).strip()
        if not physical_quantity:
            physical_quantity = (
                'acceleration'
                if str(spec.name) in ACCELERATION_CONTRIBUTION_FORCES
                else 'force'
            )
        rows.append(
            {
                'name': str(spec.name),
                'enabled': int(bool(spec.enabled)),
                'model': str(spec.model),
                'status': str(spec.status),
                'physical_quantity': physical_quantity,
                'required_fields': ';'.join(str(name) for name in spec.required_fields),
                'optional_fields': ';'.join(str(name) for name in spec.optional_fields),
                'field_sources': json.dumps(dict(spec.field_sources), sort_keys=True),
                'config': json.dumps(dict(spec.config), sort_keys=True, default=str),
            }
        )
    return rows


def _write_scalar_summary_csv(path: Path, values: Mapping[str, object]) -> bool:
    rows = _build_scalar_summary_rows(values)
    if not rows:
        return False
    _write_rows_csv(path, rows, ['quantity', 'value', 'unit'])
    return True


def _write_json(path: Path, values: Mapping[str, object]) -> None:
    Path(path).write_text(json.dumps(dict(values), indent=2), encoding='utf-8')


def _rows_frame(rows: List[Mapping[str, object]], columns: List[str]):
    import pandas as pd

    return pd.DataFrame(rows, columns=columns)


def _write_rows_csv(path: Path, rows: List[Mapping[str, object]], columns: List[str]) -> None:
    _rows_frame(rows, columns).to_csv(path, index=False)


def _diagnostic_int_fields(diagnostics: Mapping[str, object], names: Tuple[str, ...]) -> Dict[str, int]:
    return {name: int(diagnostics.get(name, 0)) for name in names}


def _diagnostic_float_field(diagnostics: Mapping[str, object], name: str) -> float:
    return float(diagnostics.get(name, 0.0))


def _release_grace_report_fields(diagnostics: Mapping[str, object]) -> Dict[str, object]:
    return {
        **_diagnostic_int_fields(diagnostics, RELEASE_GRACE_DIAGNOSTIC_FIELDS),
        'source_surface_release_grace_time_s': _diagnostic_float_field(
            diagnostics,
            'source_surface_release_grace_time_s',
        ),
        'source_surface_release_grace_clearance_m': _diagnostic_float_field(
            diagnostics,
            'source_surface_release_grace_clearance_m',
        ),
        'source_surface_release_grace_min_outward_normal_speed_mps': _diagnostic_float_field(
            diagnostics,
            'source_surface_release_grace_min_outward_normal_speed_mps',
        ),
        'source_surface_release_skip_blocked_reasons': dict(
            diagnostics.get('source_surface_release_skip_blocked_reasons', {})
        ),
    }


def _diagnostic_section_enabled(diagnostics: Mapping[str, object], name: str) -> bool:
    section = diagnostics.get(name, {})
    if not isinstance(section, Mapping):
        return False
    return int(section.get('enabled', 0)) != 0


def _heavy_diagnostics_enabled(payload: 'RuntimeOutputPayload') -> bool:
    return bool(
        str(payload.output_options.output_mode) == OUTPUT_MODE_DEBUG
        or int(payload.write_collision_diagnostics) != 0
    )


def _lightweight_report_context_enabled(payload: 'RuntimeOutputPayload') -> bool:
    return bool(
        str(payload.output_options.output_mode) == OUTPUT_MODE_MINIMAL
        and not _heavy_diagnostics_enabled(payload)
    )


def _trajectory_output_enabled(payload: 'RuntimeOutputPayload') -> bool:
    options = payload.output_options
    return bool(
        int(options.write_positions) != 0
        or int(options.write_segmented_positions) != 0
        or int(options.write_trajectory_plot) != 0
    )


def _artifact_filename(outputs_written: bool, enabled: bool, filename: str) -> str:
    return str(filename) if bool(outputs_written) and bool(enabled) else ''


def config_bool_flag(cfg: Mapping[str, object], name: str, default: int) -> int:
    raw = cfg.get(name, default)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {'0', 'false', 'no', 'off'}:
            return 0
        if value in {'1', 'true', 'yes', 'on'}:
            return 1
    return int(bool(raw))


@dataclass(frozen=True)
class RuntimeOutputOptions:
    output_mode: str = OUTPUT_MODE_DEBUG
    write_positions: int = 1
    write_segmented_positions: int = 1
    write_source_diagnostics: int = 1
    write_wall_events: int = 1
    write_max_hit_events: int = 1
    write_runtime_step_summary: int = 1
    write_prepared_summary: int = 1
    write_wall_summary: int = 1
    write_coating_summary: int = 1
    write_trajectory_plot: int = 1
    write_force_contributions: int = 1

    @classmethod
    def from_output_plan(
        cls,
        output_plan: OutputPlan,
        output_cfg: Mapping[str, object] | None = None,
    ) -> 'RuntimeOutputOptions':
        """Build writer options from the resolved OutputPlan.

        Legacy per-artifact flags remain explicit overrides, but mode defaults
        come from SolverPlan/OutputPlan so output policy is resolved once.
        """

        output_cfg = {} if output_cfg is None else output_cfg
        mode = normalize_output_mode(output_plan.mode, default=OUTPUT_MODE_STANDARD)
        default_source_diagnostics = 1
        default_max_hit_events = 1
        default_prepared_summary = 1
        default_trajectory_plot = 1
        if mode == OUTPUT_MODE_MINIMAL:
            default_source_diagnostics = 0
            default_max_hit_events = 0
            default_prepared_summary = 0
            default_trajectory_plot = 0
        elif mode == OUTPUT_MODE_STANDARD:
            default_source_diagnostics = 0
            default_max_hit_events = 0
            default_prepared_summary = 1
            default_trajectory_plot = 0

        write_positions = config_bool_flag(
            output_cfg,
            'write_positions',
            int(bool(output_plan.save_trajectory)),
        )
        return cls(
            output_mode=str(mode),
            write_positions=int(write_positions),
            write_segmented_positions=int(
                config_bool_flag(output_cfg, 'write_segmented_positions', write_positions)
            ),
            write_source_diagnostics=int(
                config_bool_flag(output_cfg, 'write_source_diagnostics', default_source_diagnostics)
            ),
            write_wall_events=int(
                config_bool_flag(output_cfg, 'write_wall_events', int(bool(output_plan.write_wall_events)))
            ),
            write_max_hit_events=int(config_bool_flag(output_cfg, 'write_max_hit_events', default_max_hit_events)),
            write_runtime_step_summary=int(
                config_bool_flag(
                    output_cfg,
                    'write_runtime_step_summary',
                    int(bool(output_plan.write_step_summary)),
                )
            ),
            write_prepared_summary=int(config_bool_flag(output_cfg, 'write_prepared_summary', default_prepared_summary)),
            write_wall_summary=int(config_bool_flag(output_cfg, 'write_wall_summary', 1)),
            write_coating_summary=int(config_bool_flag(output_cfg, 'write_coating_summary', 1)),
            write_trajectory_plot=int(config_bool_flag(output_cfg, 'write_trajectory_plot', default_trajectory_plot)),
            write_force_contributions=int(
                config_bool_flag(
                    output_cfg,
                    'write_force_contributions',
                    int(bool(output_plan.write_force_contributions)),
                )
            ),
        )

    def capture_positions(self) -> bool:
        return bool(
            int(self.write_positions) != 0
            or int(self.write_segmented_positions) != 0
            or int(self.write_trajectory_plot) != 0
        )


class CoatingSummaryAccumulator:
    discarding = False

    def __init__(self) -> None:
        self._buckets: Dict[int, Dict[str, object]] = {}

    def append(self, row: Mapping[str, object]) -> None:
        part_id = int(row.get('part_id', 0))
        bucket = self._buckets.setdefault(
            part_id,
            {
                'part_id': part_id,
                'material_id': int(row.get('material_id', 0)),
                'material_name': str(row.get('material_name', '')),
                'impact_count': 0,
                'stuck_count': 0,
                'absorbed_count': 0,
                'deposited_mass_kg': 0.0,
                'impact_speed_sum_mps': 0.0,
                'impact_speed_count': 0,
                'impact_angle_sum_deg': 0.0,
                'impact_angle_count': 0,
            },
        )
        bucket['impact_count'] = int(bucket['impact_count']) + 1
        speed = _summary_float_or_nan(row.get('impact_speed_mps', float('nan')))
        if np.isfinite(speed):
            bucket['impact_speed_sum_mps'] = float(bucket['impact_speed_sum_mps']) + speed
            bucket['impact_speed_count'] = int(bucket['impact_speed_count']) + 1
        angle = _summary_float_or_nan(row.get('impact_angle_deg_from_normal', float('nan')))
        if np.isfinite(angle):
            bucket['impact_angle_sum_deg'] = float(bucket['impact_angle_sum_deg']) + angle
            bucket['impact_angle_count'] = int(bucket['impact_angle_count']) + 1
        outcome = str(row.get('outcome', '')).strip().lower()
        if outcome == 'stuck':
            bucket['stuck_count'] = int(bucket['stuck_count']) + 1
            mass = _summary_float_or_nan(row.get('particle_mass_kg', float('nan')))
            if np.isfinite(mass):
                bucket['deposited_mass_kg'] = float(bucket['deposited_mass_kg']) + mass
        elif outcome == 'absorbed':
            bucket['absorbed_count'] = int(bucket['absorbed_count']) + 1

    def rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for bucket in sorted(self._buckets.values(), key=lambda item: int(item['part_id'])):
            speed_count = int(bucket['impact_speed_count'])
            angle_count = int(bucket['impact_angle_count'])
            rows.append(
                {
                    'part_id': int(bucket['part_id']),
                    'material_id': int(bucket['material_id']),
                    'material_name': str(bucket['material_name']),
                    'impact_count': int(bucket['impact_count']),
                    'stuck_count': int(bucket['stuck_count']),
                    'absorbed_count': int(bucket['absorbed_count']),
                    'deposited_mass_kg': float(bucket['deposited_mass_kg']),
                    'mean_impact_speed_mps': (
                        float(bucket['impact_speed_sum_mps']) / float(speed_count)
                        if speed_count
                        else float('nan')
                    ),
                    'mean_impact_angle_deg_from_normal': (
                        float(bucket['impact_angle_sum_deg']) / float(angle_count)
                        if angle_count
                        else float('nan')
                    ),
                }
            )
        return rows


@dataclass(frozen=True)
class RuntimeOutputPayload:
    prepared: PreparedRuntime
    spatial_dim: int
    particles: ParticleTable
    release_time: np.ndarray
    positions: np.ndarray
    save_meta: List[Dict[str, object]]
    final_position: np.ndarray
    final_velocity: np.ndarray
    final_charge: np.ndarray
    released: np.ndarray
    active: np.ndarray
    stuck: np.ndarray
    absorbed: np.ndarray
    contact_sliding: np.ndarray
    contact_endpoint_stopped: np.ndarray
    contact_edge_index: np.ndarray
    contact_part_id: np.ndarray
    contact_normal: np.ndarray
    escaped: np.ndarray
    invalid_mask_stopped: np.ndarray
    numerical_boundary_stopped: np.ndarray
    invalid_stop_reason_code: np.ndarray
    final_step_name: str
    final_segment_name: str
    wall_rows: List[Dict[str, object]]
    coating_summary_rows: List[Dict[str, object]]
    wall_law_counts: Dict[str, int]
    wall_summary_counts: Dict[Tuple[int, str, str], int]
    max_hit_rows: List[Dict[str, object]]
    step_rows: List[Dict[str, object]]
    collision_diagnostics: Dict[str, object]
    base_integrator_name: str
    write_collision_diagnostics: int
    max_wall_hits_per_step: int
    min_remaining_dt_ratio: float
    on_boundary_tol_m: float
    epsilon_offset_m: float
    adaptive_substep_enabled: int
    adaptive_substep_tau_ratio: float
    adaptive_substep_max_splits: int
    plot_limit: int
    valid_mask_policy: str
    output_options: RuntimeOutputOptions
    drag_model: str
    timing_s: Dict[str, float]
    memory_estimate_bytes: Dict[str, int]


@dataclass(frozen=True)
class RuntimeReportContext:
    event_counts: Dict[str, int]
    boundary_event_contract: Dict[str, object]
    max_hit_summary: Dict[str, object]
    max_hit_diag_summary: Dict[str, object]
    summary_bundle: Dict[str, object]


def _runtime_coordinate_system(payload: RuntimeOutputPayload) -> str:
    return str(payload.prepared.runtime.coordinate_system)


def _runtime_axis_names(payload: RuntimeOutputPayload) -> List[str]:
    runtime = payload.prepared.runtime
    return list(axis_names_for_coordinate_system(runtime.coordinate_system, runtime.spatial_dim))


def _runtime_axisymmetric_rz_report(payload: RuntimeOutputPayload) -> Dict[str, object]:
    geometry_provider = getattr(payload.prepared.runtime, 'geometry_provider', None)
    geometry = getattr(geometry_provider, 'geometry', None) if geometry_provider is not None else None
    return axisymmetric_rz_report_from_metadata(getattr(geometry, 'metadata', None))


def _final_state_labels(payload: RuntimeOutputPayload) -> np.ndarray:
    labels = np.full(int(payload.particles.count), 'inactive', dtype=object)
    labels[np.asarray(payload.active, dtype=bool)] = 'active_free_flight'
    labels[np.asarray(payload.contact_sliding, dtype=bool)] = 'contact_sliding'
    labels[np.asarray(payload.contact_endpoint_stopped, dtype=bool)] = 'contact_endpoint_stopped'
    for name, values in (
        ('invalid_mask_stopped', payload.invalid_mask_stopped),
        ('numerical_boundary_stopped', payload.numerical_boundary_stopped),
        ('stuck', payload.stuck),
        ('absorbed', payload.absorbed),
        ('escaped', payload.escaped),
    ):
        labels[np.asarray(values, dtype=bool)] = name
    return labels


def _final_state_count_dict(payload: RuntimeOutputPayload) -> Dict[str, int]:
    labels = _final_state_labels(payload)
    return {name: int(np.count_nonzero(labels == name)) for name in FINAL_STATE_ORDER}


_INVALID_SEGMENT_FILENAME_TRANSLATION = str.maketrans({ch: '_' for ch in '<>:"/\\|?*'})


def _safe_segment_filename(segment_name: object) -> str:
    raw = str(segment_name).strip() if segment_name is not None else ''
    if not raw:
        return 'run'
    safe = raw.translate(_INVALID_SEGMENT_FILENAME_TRANSLATION)
    safe = ''.join(ch if ch.isprintable() and ch not in {'\r', '\n', '\t'} else '_' for ch in safe)
    safe = safe.strip(' .')
    return safe or 'run'


def _segment_frame(save_meta: List[Dict[str, object]]):
    import pandas as pd

    df = pd.DataFrame(save_meta)
    rows = []
    for segment_name, sub in df.groupby('segment_name', dropna=False):
        safe = _safe_segment_filename(segment_name)
        rows.append(
            {
                'segment_name': safe,
                'save_count': int(len(sub)),
                't_start': float(sub['time_s'].min()),
                't_end': float(sub['time_s'].max()),
            }
        )
    return df, rows


def _save_segmented_positions(
    output_dir: Path,
    positions: np.ndarray,
    save_meta: List[Dict[str, object]],
    spatial_dim: int,
    *,
    skip_single_segment_arrays: bool,
) -> None:
    if positions.size == 0 or not save_meta:
        return
    df, rows = _segment_frame(save_meta)
    df.to_csv(output_dir / 'save_frames.csv', index=False)
    _write_rows_csv(output_dir / 'segment_summary.csv', rows, SEGMENT_SUMMARY_COLUMNS)
    if bool(skip_single_segment_arrays) and len(rows) <= 1:
        return
    segments_dir = output_dir / 'segments'
    segments_dir.mkdir(parents=True, exist_ok=True)
    for segment_name, sub in df.groupby('segment_name', dropna=False):
        idx = sub['save_index'].to_numpy(dtype=int)
        arr = positions[idx]
        safe = _safe_segment_filename(segment_name)
        np.save(segments_dir / f'positions_{safe}_{spatial_dim}d.npy', arr)


def _write_save_frame_metadata(output_dir: Path, save_meta: List[Dict[str, object]]) -> None:
    if not save_meta:
        return
    df, rows = _segment_frame(save_meta)
    df.to_csv(output_dir / 'save_frames.csv', index=False)
    _write_rows_csv(output_dir / 'segment_summary.csv', rows, SEGMENT_SUMMARY_COLUMNS)


def _build_final_particles_frame(payload: RuntimeOutputPayload) -> pd.DataFrame:
    import pandas as pd
    reason_codes = np.asarray(payload.invalid_stop_reason_code, dtype=np.uint8)
    reason_names = np.asarray(invalid_stop_reason_names(reason_codes), dtype=object)

    final_df = pd.DataFrame(
        {
            'particle_id': payload.particles.particle_id,
            'release_time': payload.release_time,
            'released': payload.released.astype(int),
            'active': payload.active.astype(int),
            'stuck': payload.stuck.astype(int),
            'absorbed': payload.absorbed.astype(int),
            'contact_sliding': payload.contact_sliding.astype(int),
            'contact_endpoint_stopped': payload.contact_endpoint_stopped.astype(int),
            'contact_edge_index': payload.contact_edge_index.astype(int),
            'contact_part_id': payload.contact_part_id.astype(int),
            'escaped': payload.escaped.astype(int),
            'invalid_mask_stopped': payload.invalid_mask_stopped.astype(int),
            'numerical_boundary_stopped': payload.numerical_boundary_stopped.astype(int),
            'invalid_stop_reason': reason_names,
            'final_step_name': payload.final_step_name,
            'final_segment_name': payload.final_segment_name,
            'source_part_id': payload.particles.source_part_id,
            'source_provenance_group': [
                source_provenance_group(int(value)) for value in np.asarray(payload.particles.source_part_id)
            ],
            'material_id': payload.particles.material_id,
            'initial_charge_C': payload.particles.charge,
            'charge_C': payload.final_charge,
            'charge_e': payload.final_charge / 1.602176634e-19,
        }
    )
    for j, name in enumerate(['x', 'y', 'z'][: payload.spatial_dim]):
        final_df[name] = payload.final_position[:, j]
        final_df[f'v_{name}'] = payload.final_velocity[:, j]
        final_df[f'contact_normal_{name}'] = payload.contact_normal[:, j]
    return final_df


def _write_resolved_particles(payload: RuntimeOutputPayload, output_dir: Path) -> None:
    import pandas as pd

    if payload.prepared.source_preprocess is None:
        return
    write_source_summary(payload.prepared.source_preprocess, output_dir)
    cols = {
        'particle_id': payload.particles.particle_id,
        'release_time': payload.release_time,
        'source_part_id': payload.particles.source_part_id,
        'material_id': payload.particles.material_id,
        'source_event_tag': payload.particles.source_event_tag,
    }
    for j, name in enumerate(['x', 'y', 'z'][: payload.spatial_dim]):
        cols[name] = payload.particles.position[:, j]
        cols[f'v{name}'] = payload.particles.velocity[:, j]
    pd.DataFrame(cols).to_csv(output_dir / 'resolved_particles.csv', index=False)


def _build_wall_summary_report(wall_summary_counts: Dict[Tuple[int, str, str], int]) -> Dict[str, object]:
    wall_summary_report: Dict[str, object] = {
        'total_wall_interactions': int(sum(wall_summary_counts.values())),
        'by_part': {},
        'by_outcome': {},
        'by_wall_mode': {},
    }
    by_part = wall_summary_report['by_part']
    by_outcome = wall_summary_report['by_outcome']
    by_wall_mode = wall_summary_report['by_wall_mode']
    for (part_id, outcome, wall_mode), count in wall_summary_counts.items():
        part_bucket = by_part.setdefault(str(int(part_id)), {})
        part_bucket[str(outcome)] = int(part_bucket.get(str(outcome), 0) + int(count))
        by_outcome[str(outcome)] = int(by_outcome.get(str(outcome), 0) + int(count))
        by_wall_mode[str(wall_mode)] = int(by_wall_mode.get(str(wall_mode), 0) + int(count))
    return wall_summary_report


def _field_support_exit_part_ids(payload: RuntimeOutputPayload) -> List[int]:
    wall_catalog = payload.prepared.runtime.wall_catalog
    ids: set[int] = set()
    if wall_catalog is not None:
        for model in wall_catalog.part_models:
            part_name = str(model.part_name).strip().lower()
            material_name = str(model.material_name).strip().lower()
            if part_name == 'field_support_exit' or material_name == 'field_support_exit':
                ids.add(int(model.part_id))
    if not ids:
        for (part_id, _outcome, wall_mode), _count in payload.wall_summary_counts.items():
            if int(part_id) >= 9000 and str(wall_mode).strip().lower() in {
                'open',
                'outflow',
                'exhaust',
                'escape',
                'field_support_exit',
                'disappear',
                'absorb',
            }:
                ids.add(int(part_id))
    return sorted(ids)


def _generated_dir_from_payload(payload: RuntimeOutputPayload) -> Path | None:
    runtime = payload.prepared.runtime
    for provider_name in ('geometry_provider', 'field_provider'):
        provider = getattr(runtime, provider_name, None)
        obj = getattr(provider, 'geometry', None) if provider_name == 'geometry_provider' else getattr(provider, 'field', None)
        metadata = getattr(obj, 'metadata', {}) if obj is not None else {}
        if isinstance(metadata, Mapping):
            raw = metadata.get('npz_path', '')
            if raw:
                path = Path(str(raw))
                parent = path.parent
                if parent.exists():
                    return parent
    return None


def _build_field_support_exit_summary(payload: RuntimeOutputPayload) -> Dict[str, object]:
    support_ids = set(_field_support_exit_part_ids(payload))
    support_exit_count = 0
    support_exit_absorbed_count = 0
    absorbed_event_count = 0
    total_interaction_count = 0
    for (part_id, outcome, _wall_mode), count_raw in payload.wall_summary_counts.items():
        count = int(count_raw)
        total_interaction_count += count
        if int(part_id) in support_ids:
            support_exit_count += count
        if str(outcome) == 'absorbed':
            absorbed_event_count += count
            if int(part_id) in support_ids:
                support_exit_absorbed_count += count
    physical_absorbed_count = int(max(0, absorbed_event_count - support_exit_absorbed_count))
    physical_interaction_count = int(max(0, total_interaction_count - support_exit_count))
    return {
        'field_support_exit_part_ids': [int(pid) for pid in sorted(support_ids)],
        'field_support_exit_count': int(support_exit_count),
        'field_support_exit_absorbed_count': int(support_exit_absorbed_count),
        'physical_absorbed_count': int(physical_absorbed_count),
        'physical_wall_interaction_count': int(physical_interaction_count),
        'absorbed_count_includes_field_support_exit': int(support_exit_absorbed_count > 0),
    }


def _build_coating_summary_rows(payload: RuntimeOutputPayload) -> List[Dict[str, object]]:
    if payload.coating_summary_rows:
        return [
            dict(row)
            for row in sorted(payload.coating_summary_rows, key=lambda item: int(item.get('part_id', 0)))
        ]
    if payload.wall_rows:
        accumulator = CoatingSummaryAccumulator()
        for row in payload.wall_rows:
            accumulator.append(row)
        return accumulator.rows()

    rows_by_part: Dict[int, Dict[str, object]] = {}
    for (part_id_raw, outcome_raw, _wall_mode), count_raw in payload.wall_summary_counts.items():
        part_id = int(part_id_raw)
        count = int(count_raw)
        row = rows_by_part.setdefault(
            part_id,
            {
                'part_id': part_id,
                'material_id': 0,
                'material_name': '',
                'impact_count': 0,
                'stuck_count': 0,
                'absorbed_count': 0,
                'deposited_mass_kg': float('nan'),
                'mean_impact_speed_mps': float('nan'),
                'mean_impact_angle_deg_from_normal': float('nan'),
            },
        )
        row['impact_count'] = int(row['impact_count']) + count
        outcome = str(outcome_raw).strip().lower()
        if outcome == 'stuck':
            row['stuck_count'] = int(row['stuck_count']) + count
        elif outcome == 'absorbed':
            row['absorbed_count'] = int(row['absorbed_count']) + count
    return [rows_by_part[key] for key in sorted(rows_by_part)]


def _build_coating_summary_report(rows: List[Dict[str, object]]) -> Dict[str, object]:
    finite_mass = [
        value
        for value in (_summary_float_or_nan(row.get('deposited_mass_kg')) for row in rows)
        if np.isfinite(value)
    ]
    return {
        'part_count': int(len(rows)),
        'impact_count': int(sum(int(row['impact_count']) for row in rows)),
        'stuck_count': int(sum(int(row['stuck_count']) for row in rows)),
        'absorbed_count': int(sum(int(row['absorbed_count']) for row in rows)),
        'deposited_mass_kg': float(sum(finite_mass)) if finite_mass else None,
        'mass_available': int(bool(finite_mass)),
    }


def _top_counts(values: List[object], *, key_name: str, limit: int = 8) -> List[Dict[str, object]]:
    counts = Counter(values)
    rows: List[Dict[str, object]] = []
    for value, count in counts.most_common(int(limit)):
        if isinstance(value, (bool, int, np.integer)):
            normalized: object = int(value)
        elif isinstance(value, (float, np.floating)):
            normalized = float(value)
        else:
            normalized = str(value)
        rows.append({key_name: normalized, 'count': int(count)})
    return rows


def _top_count_dict(counts: object, *, key_name: str, limit: int = 8) -> List[Dict[str, object]]:
    if not isinstance(counts, Mapping):
        return []
    pairs = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return [{key_name: str(key), 'count': int(value)} for key, value in pairs[: int(limit)]]


def _build_max_hit_diagnostic_summary(collision_diagnostics: Mapping[str, object]) -> Dict[str, object]:
    diag = collision_diagnostics if isinstance(collision_diagnostics, Mapping) else {}
    reached = int(diag.get('max_hits_reached_count', 0))
    return {
        'event_count': int(reached),
        'same_wall_count': int(diag.get('max_hit_same_wall_count', 0)),
        'multi_wall_count': int(diag.get('max_hit_multi_wall_count', 0)),
        'remaining_dt_total_s': float(diag.get('max_hit_remaining_dt_total_s', 0.0)),
        'remaining_dt_max_s': float(diag.get('max_hit_remaining_dt_max_s', 0.0)),
        'top_last_parts': _top_count_dict(diag.get('max_hit_last_part_counts', {}), key_name='part'),
        'top_last_outcomes': _top_count_dict(diag.get('max_hit_last_outcome_counts', {}), key_name='outcome'),
    }


def _build_boundary_event_contract_summary(
    *,
    numerical_boundary_stopped_count: int,
    unresolved_crossing_count: int,
    max_hits_reached_count: int,
    nearest_projection_fallback_count: int,
) -> Dict[str, object]:
    failure_count = (
        int(numerical_boundary_stopped_count)
        + int(unresolved_crossing_count)
        + int(max_hits_reached_count)
        + int(nearest_projection_fallback_count)
    )
    return {
        'passed': int(failure_count == 0),
        'failure_count': int(failure_count),
        'numerical_boundary_stopped_count': int(numerical_boundary_stopped_count),
        'unresolved_crossing_count': int(unresolved_crossing_count),
        'max_hits_reached_count': int(max_hits_reached_count),
        'nearest_projection_fallback_count': int(nearest_projection_fallback_count),
    }


def _boundary_event_counts(payload: RuntimeOutputPayload) -> Dict[str, int]:
    diagnostics = payload.collision_diagnostics
    return {
        'numerical_boundary_stopped_count': int(payload.numerical_boundary_stopped.sum()),
        'unresolved_crossing_count': int(diagnostics.get('unresolved_crossing_count', 0)),
        'max_hits_reached_count': int(diagnostics.get('max_hits_reached_count', 0)),
        'bisection_fallback_count': int(diagnostics.get('bisection_fallback_count', 0)),
        'nearest_projection_fallback_count': int(diagnostics.get('nearest_projection_fallback_count', 0)),
    }


def _boundary_event_contract_from_counts(counts: Mapping[str, int]) -> Dict[str, object]:
    return _build_boundary_event_contract_summary(
        numerical_boundary_stopped_count=int(counts.get('numerical_boundary_stopped_count', 0)),
        unresolved_crossing_count=int(counts.get('unresolved_crossing_count', 0)),
        max_hits_reached_count=int(counts.get('max_hits_reached_count', 0)),
        nearest_projection_fallback_count=int(counts.get('nearest_projection_fallback_count', 0)),
    )


def _boundary_diagnostics(payload: RuntimeOutputPayload) -> Dict[str, object]:
    diagnostics = payload.collision_diagnostics
    return {
        'wall_law_semantics': dict(WALL_LAW_SEMANTICS),
        'non_colliding_wall_laws': [
            law for law, meaning in WALL_LAW_SEMANTICS.items() if meaning == 'non_colliding_boundary'
        ],
        'particle_exit_wall_laws': [
            law for law, meaning in WALL_LAW_SEMANTICS.items() if meaning == 'particle_exit'
        ],
        'contact_tangent_model': str(diagnostics.get('contact_tangent_model', '')),
        'contact_tangent_model_scope': (
            'custom_non_comsol_standard_contact_sliding'
            if str(diagnostics.get('contact_tangent_model', ''))
            else ''
        ),
        'collision_boundary_geometry': str(
            diagnostics.get('collision_boundary_geometry', 'linear_segment_or_triangle_boundary')
        ),
        'ambiguous_hit_count': int(diagnostics.get('boundary_ambiguous_hit_count', 0)),
        'ambiguous_hit_part_counts': dict(diagnostics.get('boundary_ambiguous_part_counts', {})),
        'ambiguous_hit_wall_law_counts': dict(diagnostics.get('boundary_ambiguous_wall_law_counts', {})),
        'ambiguous_hit_primitive_kind_counts': dict(
            diagnostics.get('boundary_ambiguous_primitive_kind_counts', {})
        ),
    }


def _diagnostic_metadata(payload: RuntimeOutputPayload) -> Dict[str, object]:
    diagnostics = payload.collision_diagnostics
    return {
        'drag_gas_properties': dict(diagnostics.get('drag_gas_properties', {})),
        'field_backend_diagnostics': dict(diagnostics.get('field_backend_diagnostics', {})),
        'force_catalog': dict(diagnostics.get('force_catalog', {})),
        'force_runtime': dict(diagnostics.get('force_runtime', {})),
        'wall_catalog_alignment': dict(diagnostics.get('wall_catalog_alignment', {})),
        'stochastic_motion': dict(diagnostics.get('stochastic_motion', {})),
        'plasma_background': dict(diagnostics.get('plasma_background', {})),
        'charge_model': dict(diagnostics.get('charge_model', {})),
        'acceleration_source': str(diagnostics.get('acceleration_source', 'none')),
        'acceleration_quantity_names': list(diagnostics.get('acceleration_quantity_names', [])),
        'electric_field_names': list(diagnostics.get('electric_field_names', [])),
        'electric_q_over_m_Ckg': float(diagnostics.get('electric_q_over_m_Ckg', 0.0)),
        'electric_q_over_m_particle_stats': dict(diagnostics.get('electric_q_over_m_particle_stats', {})),
        'output_buffers': dict(diagnostics.get('output_buffers', {})),
        'contact_tangent_model': str(diagnostics.get('contact_tangent_model', '')),
        'boundary_diagnostics': _boundary_diagnostics(payload),
    }


def _shared_report_fields(payload: RuntimeOutputPayload) -> Dict[str, object]:
    output_mode = str(payload.output_options.output_mode)
    report = {
        'coordinate_system': _runtime_coordinate_system(payload),
        'axis_names': _runtime_axis_names(payload),
        'output_mode': output_mode,
        'output_minimal_enabled': int(output_mode == OUTPUT_MODE_MINIMAL),
        'output_debug_enabled': int(output_mode == OUTPUT_MODE_DEBUG),
        'output_force_contributions_enabled': int(payload.output_options.write_force_contributions),
        'output_collision_diagnostics_enabled': int(payload.write_collision_diagnostics),
        'final_state_counts': _final_state_count_dict(payload),
        'timing_s': {str(k): float(v) for k, v in payload.timing_s.items()},
        'memory_estimate_bytes': {str(k): int(v) for k, v in payload.memory_estimate_bytes.items()},
        'integrator': str(payload.base_integrator_name),
        'valid_mask_policy': str(payload.valid_mask_policy),
        'drag_model': str(payload.drag_model),
        **_diagnostic_metadata(payload),
        **field_backend_report(payload.prepared.runtime.field_provider),
    }
    axisymmetric_report = _runtime_axisymmetric_rz_report(payload)
    if axisymmetric_report:
        report['axisymmetric_rz'] = axisymmetric_report
    return report


def _finite_summary(values: np.ndarray) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {'count': 0}
    quantiles = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    return {
        'count': int(finite.size),
        'min': float(quantiles[0]),
        'p25': float(quantiles[1]),
        'median': float(quantiles[2]),
        'p75': float(quantiles[3]),
        'p90': float(quantiles[4]),
        'p99': float(quantiles[5]),
        'max': float(quantiles[6]),
        'mean': float(np.mean(finite)),
    }


def _top_part_reason_counts(part_ids: np.ndarray, reasons: List[str], *, limit: int = 12) -> List[Dict[str, object]]:
    counts = Counter((int(part_id), str(reason)) for part_id, reason in zip(part_ids, reasons))
    return [
        {'part_id': int(part_id), 'reason': str(reason), 'count': int(count)}
        for (part_id, reason), count in counts.most_common(int(limit))
    ]


def _particle_state_labels(payload: RuntimeOutputPayload) -> np.ndarray:
    labels = np.full(int(payload.particles.count), 'inactive', dtype=object)
    state_masks = _payload_state_masks(payload)
    for name in GEOMETRY_STATE_ORDER:
        if name == 'inactive':
            continue
        mask = state_masks[name]
        labels[np.asarray(mask, dtype=bool)] = name
    return labels


def _near_boundary_threshold_m(payload: RuntimeOutputPayload) -> float:
    geometry_provider = getattr(payload.prepared.runtime, 'geometry_provider', None)
    if geometry_provider is None:
        return 0.0
    spacings: List[float] = []
    for axis in getattr(geometry_provider.geometry, 'axes', ()):
        values = np.asarray(axis, dtype=np.float64)
        diffs = np.diff(values)
        positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if positive.size:
            spacings.append(float(np.min(positive)))
    return float(min(spacings)) if spacings else 0.0


def _sample_geometry_features(runtime, positions: np.ndarray) -> Dict[str, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64)
    return {
        'sdf_m': np.asarray([sample_geometry_sdf(runtime, p) for p in pos], dtype=np.float64),
        'inside_geometry': np.asarray([inside_geometry(runtime, p, on_boundary_tol_m=0.0) for p in pos], dtype=bool),
        'nearest_part_id': np.asarray([sample_geometry_part_id(runtime, p) for p in pos], dtype=np.int32),
    }


def _geometry_features_for_positions(
    payload: RuntimeOutputPayload,
    positions: np.ndarray,
    velocities: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    runtime = payload.prepared.runtime
    pos = np.asarray(positions, dtype=np.float64)
    features = _sample_geometry_features(runtime, pos)
    nearest_part_ids = np.asarray(features['nearest_part_id'], dtype=np.int32)
    nearest_distances = np.abs(np.asarray(features['sdf_m'], dtype=np.float64))
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if (
        int(payload.spatial_dim) == 2
        and geometry_provider is not None
        and getattr(geometry_provider.geometry, 'boundary_edges', None) is not None
    ):
        edge_part_ids, edge_distances = nearest_boundary_edge_features_2d(runtime, pos)
        finite_edge = np.isfinite(edge_distances)
        if np.any(finite_edge):
            nearest_part_ids = edge_part_ids
            nearest_distances = edge_distances
    velocity_values = (
        np.asarray(payload.final_velocity, dtype=np.float64)
        if velocities is None
        else np.asarray(velocities, dtype=np.float64)
    )
    speed = np.linalg.norm(velocity_values[:, : int(payload.spatial_dim)], axis=1)
    return {
        'sdf_m': np.asarray(features['sdf_m'], dtype=np.float64),
        'inside_geometry': np.asarray(features['inside_geometry'], dtype=bool),
        'nearest_part_id': nearest_part_ids,
        'nearest_boundary_distance_m': nearest_distances,
        'speed_mps': speed,
    }


def _empty_geometry_summary() -> Dict[str, object]:
    return {
        'count': 0,
        'sdf_m': dict(EMPTY_FINITE_SUMMARY),
        'abs_sdf_m': dict(EMPTY_FINITE_SUMMARY),
        'nearest_boundary_distance_m': dict(EMPTY_FINITE_SUMMARY),
        'speed_mps': dict(EMPTY_FINITE_SUMMARY),
        'outside_geometry_count': 0,
        'near_boundary_count': 0,
        'nearest_part_counts': [],
    }


def _summarize_geometry_subset(
    *,
    mask: np.ndarray,
    features: Mapping[str, np.ndarray],
    near_boundary_threshold_m: float,
) -> Dict[str, object]:
    selected = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(selected))
    if count == 0:
        return _empty_geometry_summary()
    sdf_values = np.asarray(features['sdf_m'], dtype=np.float64)[selected]
    nearest_distances = np.asarray(features['nearest_boundary_distance_m'], dtype=np.float64)[selected]
    speed = np.asarray(features['speed_mps'], dtype=np.float64)[selected]
    nearest_part_ids = np.asarray(features['nearest_part_id'], dtype=np.int32)[selected]
    inside_geometry_values = np.asarray(features.get('inside_geometry', np.ones_like(selected)), dtype=bool)[selected]
    threshold = float(near_boundary_threshold_m)
    if not np.isfinite(threshold) or threshold <= 0.0:
        near_boundary_count = 0
    else:
        near_boundary_count = int(np.count_nonzero(np.abs(sdf_values[np.isfinite(sdf_values)]) <= threshold))
    return {
        'count': int(count),
        'sdf_m': _finite_summary(sdf_values),
        'abs_sdf_m': _finite_summary(np.abs(sdf_values)),
        'nearest_boundary_distance_m': _finite_summary(nearest_distances),
        'speed_mps': _finite_summary(speed),
        'outside_geometry_count': int(np.count_nonzero(~inside_geometry_values)),
        'near_boundary_count': int(near_boundary_count),
        'nearest_part_counts': _top_counts([int(v) for v in nearest_part_ids], key_name='part_id', limit=8),
    }


def _build_state_geometry_summary(payload: RuntimeOutputPayload) -> Dict[str, object]:
    count = int(payload.particles.count)
    if count == 0:
        return {'particle_count': 0, 'near_boundary_threshold_m': 0.0, 'by_state': {}}
    labels = _particle_state_labels(payload)
    features = _geometry_features_for_positions(
        payload,
        np.asarray(payload.final_position, dtype=np.float64),
        np.asarray(payload.final_velocity, dtype=np.float64),
    )
    threshold = _near_boundary_threshold_m(payload)
    by_state: Dict[str, object] = {}
    for state_name in GEOMETRY_STATE_ORDER:
        by_state[state_name] = _summarize_geometry_subset(
            mask=labels == state_name,
            features=features,
            near_boundary_threshold_m=float(threshold),
        )
    return {
        'particle_count': int(count),
        'near_boundary_threshold_m': float(threshold),
        'by_state': by_state,
    }


def _build_source_initial_geometry_summary(payload: RuntimeOutputPayload) -> Dict[str, object]:
    count = int(payload.particles.count)
    if count == 0:
        return {
            'particle_count': 0,
            'near_boundary_threshold_m': 0.0,
            'all': _empty_geometry_summary(),
            'by_release_state': {},
        }
    features = _geometry_features_for_positions(
        payload,
        np.asarray(payload.particles.position, dtype=np.float64),
        np.asarray(payload.particles.velocity, dtype=np.float64),
    )
    threshold = _near_boundary_threshold_m(payload)
    released = np.asarray(payload.released, dtype=bool)
    all_mask = np.ones(count, dtype=bool)
    return {
        'particle_count': int(count),
        'near_boundary_threshold_m': float(threshold),
        'all': _summarize_geometry_subset(
            mask=all_mask,
            features=features,
            near_boundary_threshold_m=float(threshold),
        ),
        'by_release_state': {
            'released_by_end': _summarize_geometry_subset(
                mask=released,
                features=features,
                near_boundary_threshold_m=float(threshold),
            ),
            'unreleased_by_end': _summarize_geometry_subset(
                mask=~released,
                features=features,
                near_boundary_threshold_m=float(threshold),
            ),
        },
    }


def _build_invalid_stop_geometry_summary(payload: RuntimeOutputPayload) -> Dict[str, object]:
    stopped_mask = np.asarray(payload.invalid_mask_stopped, dtype=bool)
    count = int(np.count_nonzero(stopped_mask))
    if count == 0:
        return {
            'count': 0,
            'sdf_m': dict(EMPTY_FINITE_SUMMARY),
            'abs_sdf_m': dict(EMPTY_FINITE_SUMMARY),
            'nearest_boundary_distance_m': dict(EMPTY_FINITE_SUMMARY),
            'nearest_part_counts': [],
            'nearest_part_reason_counts': [],
        }

    positions = np.asarray(payload.final_position, dtype=np.float64)[stopped_mask]
    velocities = np.asarray(payload.final_velocity, dtype=np.float64)[stopped_mask]
    reasons = invalid_stop_reason_names(np.asarray(payload.invalid_stop_reason_code, dtype=np.uint8)[stopped_mask])
    features = _geometry_features_for_positions(payload, positions, velocities)
    sdf_values = np.asarray(features['sdf_m'], dtype=np.float64)
    nearest_part_ids = np.asarray(features['nearest_part_id'], dtype=np.int32)
    nearest_distances = np.asarray(features['nearest_boundary_distance_m'], dtype=np.float64)
    return {
        'count': int(count),
        'sdf_m': _finite_summary(sdf_values),
        'abs_sdf_m': _finite_summary(np.abs(sdf_values)),
        'nearest_boundary_distance_m': _finite_summary(nearest_distances),
        'nearest_part_counts': _top_counts([int(v) for v in nearest_part_ids], key_name='part_id', limit=12),
        'nearest_part_reason_counts': _top_part_reason_counts(nearest_part_ids, reasons, limit=16),
    }


def _lightweight_state_geometry_summary(payload: RuntimeOutputPayload) -> Dict[str, object]:
    labels = _particle_state_labels(payload)
    by_state: Dict[str, object] = {}
    for state_name in GEOMETRY_STATE_ORDER:
        by_state[state_name] = {
            'count': int(np.count_nonzero(labels == state_name)),
            'geometry_sampling_skipped': 1,
        }
    return {
        'particle_count': int(payload.particles.count),
        'near_boundary_threshold_m': 0.0,
        'geometry_sampling_skipped': 1,
        'by_state': by_state,
    }


def _lightweight_source_initial_geometry_summary(payload: RuntimeOutputPayload) -> Dict[str, object]:
    released = np.asarray(payload.released, dtype=bool)
    return {
        'particle_count': int(payload.particles.count),
        'near_boundary_threshold_m': 0.0,
        'geometry_sampling_skipped': 1,
        'all': {'count': int(payload.particles.count), 'geometry_sampling_skipped': 1},
        'by_release_state': {
            'released_by_end': {
                'count': int(np.count_nonzero(released)),
                'geometry_sampling_skipped': 1,
            },
            'unreleased_by_end': {
                'count': int(np.count_nonzero(~released)),
                'geometry_sampling_skipped': 1,
            },
        },
    }


def _lightweight_final_state_consistency_summary(payload: RuntimeOutputPayload) -> Dict[str, object]:
    masks = _payload_state_masks(payload)
    terminal_state_matrix = np.vstack(
        [
            masks['active'],
            masks['invalid_mask_stopped'],
            masks['numerical_boundary_stopped'],
            masks['stuck'],
            masks['absorbed'],
            masks['escaped'],
        ]
    )
    return {
        'geometry_sampling_skipped': 1,
        'multiple_terminal_state_count': int(np.count_nonzero(np.sum(terminal_state_matrix, axis=0) > 1)),
        'nonfinite_position_count': _nonfinite_row_count(payload.final_position),
        'nonfinite_velocity_count': _nonfinite_row_count(payload.final_velocity),
        'contact_sliding_particle_count': int(np.count_nonzero(masks['contact_sliding'])),
        'contact_endpoint_stopped_count': int(np.count_nonzero(masks['contact_endpoint_stopped'])),
        'numerical_boundary_stopped_count': int(np.count_nonzero(masks['numerical_boundary_stopped'])),
    }


def _payload_state_masks(payload: RuntimeOutputPayload) -> Dict[str, np.ndarray]:
    return {
        'active': np.asarray(payload.active, dtype=bool),
        'invalid_mask_stopped': np.asarray(payload.invalid_mask_stopped, dtype=bool),
        'numerical_boundary_stopped': np.asarray(payload.numerical_boundary_stopped, dtype=bool),
        'stuck': np.asarray(payload.stuck, dtype=bool),
        'absorbed': np.asarray(payload.absorbed, dtype=bool),
        'escaped': np.asarray(payload.escaped, dtype=bool),
        'contact_sliding': np.asarray(payload.contact_sliding, dtype=bool),
        'contact_endpoint_stopped': np.asarray(payload.contact_endpoint_stopped, dtype=bool),
    }


def _state_summary(state_summary: Mapping[str, object], name: str) -> Dict[str, object]:
    by_state = state_summary.get('by_state', {})
    if not isinstance(by_state, Mapping):
        return {}
    value = by_state.get(name, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _nonfinite_row_count(values: np.ndarray) -> int:
    arr = np.asarray(values)
    return int(np.count_nonzero(~np.all(np.isfinite(arr), axis=1)))


def _build_final_state_consistency_summary(
    payload: RuntimeOutputPayload,
    state_summary: Mapping[str, object],
) -> Dict[str, object]:
    masks = _payload_state_masks(payload)
    terminal_state_matrix = np.vstack(
        [
            masks['active'],
            masks['invalid_mask_stopped'],
            masks['numerical_boundary_stopped'],
            masks['stuck'],
            masks['absorbed'],
            masks['escaped'],
        ]
    )
    active_summary = _state_summary(state_summary, 'active')
    contact_summary = _state_summary(state_summary, 'contact_sliding')
    active_outside = int(active_summary.get('outside_geometry_count', 0))
    contact_outside = int(contact_summary.get('outside_geometry_count', 0))
    active_hard_invalid = _count_hard_invalid_final_positions(payload, masks['active'])
    return {
        'active_outside_geometry_count': int(active_outside),
        'contact_sliding_outside_geometry_count': int(contact_outside),
        'active_hard_invalid_count': int(active_hard_invalid),
        'multiple_terminal_state_count': int(np.count_nonzero(np.sum(terminal_state_matrix, axis=0) > 1)),
        'nonfinite_position_count': _nonfinite_row_count(payload.final_position),
        'nonfinite_velocity_count': _nonfinite_row_count(payload.final_velocity),
        'contact_sliding_particle_count': int(np.count_nonzero(masks['contact_sliding'])),
        'contact_endpoint_stopped_count': int(np.count_nonzero(masks['contact_endpoint_stopped'])),
        'numerical_boundary_stopped_count': int(np.count_nonzero(masks['numerical_boundary_stopped'])),
    }


def _count_hard_invalid_final_positions(payload: RuntimeOutputPayload, mask: np.ndarray) -> int:
    field_provider = payload.prepared.runtime.field_provider
    if field_provider is None:
        return 0
    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    if selected.size == 0:
        return 0
    positions = np.asarray(payload.final_position, dtype=np.float64)
    count = 0
    for idx in selected:
        try:
            status = int(sample_field_valid_status(field_provider, positions[int(idx)]))
        except (AttributeError, ValueError, TypeError, FloatingPointError, ArithmeticError):
            continue
        if valid_mask_status_requires_stop(status):
            count += 1
    return int(count)


def _build_max_hit_event_summary(max_hit_rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not max_hit_rows:
        return {
            'event_count': 0,
            'unique_particle_count': 0,
            'remaining_dt_total_s': 0.0,
            'remaining_dt_mean_s': 0.0,
            'remaining_dt_max_s': 0.0,
            'top_last_part_ids': [],
            'top_time_bins_s': [],
            'top_outcome_sequences': [],
        }
    remaining = np.asarray([_summary_float_or_nan(row.get('remaining_dt_s')) for row in max_hit_rows], dtype=np.float64)
    remaining = remaining[np.isfinite(remaining)]
    particle_ids = [int(row.get('particle_id', 0)) for row in max_hit_rows]
    last_part_ids = [int(row.get('last_part_id', 0)) for row in max_hit_rows]
    time_bins = []
    for row in max_hit_rows:
        t = _summary_float_or_nan(row.get('time_s'))
        if np.isfinite(t):
            time_bins.append(round(float(t), 1))
    outcome_sequences = [
        str(row.get('outcome_sequence', ''))
        for row in max_hit_rows
        if str(row.get('outcome_sequence', '')).strip()
    ]
    return {
        'event_count': int(len(max_hit_rows)),
        'unique_particle_count': int(len(set(particle_ids))),
        'remaining_dt_total_s': float(remaining.sum()) if remaining.size else 0.0,
        'remaining_dt_mean_s': float(remaining.mean()) if remaining.size else 0.0,
        'remaining_dt_max_s': float(remaining.max()) if remaining.size else 0.0,
        'top_last_part_ids': _top_counts(last_part_ids, key_name='part_id'),
        'top_time_bins_s': _top_counts(time_bins, key_name='time_s'),
        'top_outcome_sequences': _top_counts(outcome_sequences, key_name='outcome_sequence', limit=5),
    }


def _geometry_summary_bundle(
    payload: RuntimeOutputPayload,
    *,
    invalid_stop_geometry_summary: Mapping[str, object] | None = None,
    state_geometry_summary: Mapping[str, object] | None = None,
    source_initial_geometry_summary: Mapping[str, object] | None = None,
    lightweight: bool = False,
) -> Dict[str, object]:
    if bool(lightweight):
        invalid_count = int(np.count_nonzero(np.asarray(payload.invalid_mask_stopped, dtype=bool)))
        return {
            'invalid_stop_geometry_summary': {
                'count': int(invalid_count),
                'geometry_sampling_skipped': 1,
                'sdf_m': dict(EMPTY_FINITE_SUMMARY),
                'abs_sdf_m': dict(EMPTY_FINITE_SUMMARY),
                'nearest_boundary_distance_m': dict(EMPTY_FINITE_SUMMARY),
                'nearest_part_counts': [],
                'nearest_part_reason_counts': [],
            },
            'state_geometry_summary': _lightweight_state_geometry_summary(payload),
            'final_state_consistency_summary': _lightweight_final_state_consistency_summary(payload),
            'source_initial_geometry_summary': _lightweight_source_initial_geometry_summary(payload),
            'field_support_exit_summary': _build_field_support_exit_summary(payload),
        }
    invalid_stop_summary = (
        dict(invalid_stop_geometry_summary)
        if invalid_stop_geometry_summary is not None
        else _build_invalid_stop_geometry_summary(payload)
    )
    state_summary = (
        dict(state_geometry_summary)
        if state_geometry_summary is not None
        else _build_state_geometry_summary(payload)
    )
    source_summary = (
        dict(source_initial_geometry_summary)
        if source_initial_geometry_summary is not None
        else _build_source_initial_geometry_summary(payload)
    )
    return {
        'invalid_stop_geometry_summary': invalid_stop_summary,
        'state_geometry_summary': state_summary,
        'final_state_consistency_summary': _build_final_state_consistency_summary(payload, state_summary),
        'source_initial_geometry_summary': source_summary,
        'field_support_exit_summary': _build_field_support_exit_summary(payload),
    }


def _build_runtime_report_context(
    payload: RuntimeOutputPayload,
    *,
    invalid_stop_geometry_summary: Mapping[str, object] | None = None,
    state_geometry_summary: Mapping[str, object] | None = None,
    source_initial_geometry_summary: Mapping[str, object] | None = None,
    lightweight_geometry: bool = False,
) -> RuntimeReportContext:
    event_counts = _boundary_event_counts(payload)
    boundary_event_contract = _boundary_event_contract_from_counts(event_counts)
    return RuntimeReportContext(
        event_counts=event_counts,
        boundary_event_contract=boundary_event_contract,
        max_hit_summary=_build_max_hit_event_summary(payload.max_hit_rows),
        max_hit_diag_summary=_build_max_hit_diagnostic_summary(payload.collision_diagnostics),
        summary_bundle=_geometry_summary_bundle(
            payload,
            invalid_stop_geometry_summary=invalid_stop_geometry_summary,
            state_geometry_summary=state_geometry_summary,
            source_initial_geometry_summary=source_initial_geometry_summary,
            lightweight=bool(lightweight_geometry),
        ),
    )


def _build_collision_diag_report(
    payload: RuntimeOutputPayload,
    *,
    context: RuntimeReportContext | None = None,
    invalid_stop_geometry_summary: Mapping[str, object] | None = None,
    state_geometry_summary: Mapping[str, object] | None = None,
    source_initial_geometry_summary: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    comsol_boundary_map = _comsol_boundary_map(payload)
    report_context = context or _build_runtime_report_context(
        payload,
        invalid_stop_geometry_summary=invalid_stop_geometry_summary,
        state_geometry_summary=state_geometry_summary,
        source_initial_geometry_summary=source_initial_geometry_summary,
        lightweight_geometry=_lightweight_report_context_enabled(payload),
    )
    event_counts = report_context.event_counts
    boundary_event_contract = report_context.boundary_event_contract
    return {
        **payload.collision_diagnostics,
        **_shared_report_fields(payload),
        'comsol_boundary_map': {
            str(int(part_id)): dict(row) for part_id, row in sorted(comsol_boundary_map.items(), key=lambda item: int(item[0]))
        },
        'numerical_boundary_stopped_count': int(event_counts['numerical_boundary_stopped_count']),
        'boundary_event_contract': boundary_event_contract,
        'boundary_event_contract_passed': int(boundary_event_contract['passed']),
        'boundary_event_failure_count': int(boundary_event_contract['failure_count']),
        'max_hit_event_summary': report_context.max_hit_summary,
        'max_hit_diagnostic_summary': report_context.max_hit_diag_summary,
        **report_context.summary_bundle,
        'max_wall_hits_per_step': int(payload.max_wall_hits_per_step),
        'min_remaining_dt_ratio': float(payload.min_remaining_dt_ratio),
        'on_boundary_tol_m': float(payload.on_boundary_tol_m),
        'epsilon_offset_m': float(payload.epsilon_offset_m),
        'adaptive_substep_enabled': int(payload.adaptive_substep_enabled),
        'adaptive_substep_tau_ratio': float(payload.adaptive_substep_tau_ratio),
        'adaptive_substep_max_splits': int(payload.adaptive_substep_max_splits),
    }


def _runtime_output_filenames(payload: RuntimeOutputPayload, outputs_written: bool) -> Dict[str, str]:
    options = payload.output_options
    diagnostics = payload.collision_diagnostics
    return {
        'positions_file': _artifact_filename(
            outputs_written,
            int(options.write_positions) != 0,
            f'positions_{payload.spatial_dim}d.npy',
        ),
        'wall_summary_file': _artifact_filename(
            outputs_written,
            int(options.write_wall_summary) != 0,
            'wall_summary.json',
        ),
        'wall_summary_by_part_file': _artifact_filename(
            outputs_written,
            int(options.write_wall_summary) != 0,
            'wall_summary_by_part.csv',
        ),
        'max_hit_events_file': _artifact_filename(
            outputs_written,
            int(options.write_max_hit_events) != 0,
            'max_hit_events.csv',
        ),
        'coating_summary_file': _artifact_filename(
            outputs_written,
            int(options.write_coating_summary) != 0,
            'coating_summary_by_part.csv',
        ),
        'plasma_background_summary_file': _artifact_filename(
            outputs_written,
            _diagnostic_section_enabled(diagnostics, 'plasma_background'),
            'plasma_background_summary.csv',
        ),
        'charge_model_summary_file': _artifact_filename(
            outputs_written,
            _diagnostic_section_enabled(diagnostics, 'charge_model'),
            'charge_model_summary.csv',
        ),
        'collision_diagnostics_file': _artifact_filename(
            outputs_written,
            int(payload.write_collision_diagnostics) != 0,
            'collision_diagnostics.json',
        ),
        'force_contributions_file': _artifact_filename(
            outputs_written,
            int(options.write_force_contributions) != 0,
            'force_contributions.csv',
        ),
        'runtime_step_summary_file': _artifact_filename(
            outputs_written,
            int(options.write_runtime_step_summary) != 0,
            'runtime_step_summary.csv',
        ),
    }


def _runtime_output_row_counts(payload: RuntimeOutputPayload) -> Dict[str, int]:
    return {
        'wall_events': int(len(payload.wall_rows)),
        'max_hit_events': int(len(payload.max_hit_rows)),
        'runtime_steps': int(len(payload.step_rows)),
        'coating_summary': int(len(payload.coating_summary_rows)),
        'force_contributions': (
            int(len(_build_force_contribution_rows(payload)))
            if int(payload.output_options.write_force_contributions) != 0
            else 0
        ),
    }


def _boundary_report_fields(
    event_counts: Mapping[str, int],
    boundary_event_contract: Mapping[str, object],
) -> Dict[str, object]:
    return {
        'unresolved_crossing_count': int(event_counts['unresolved_crossing_count']),
        'max_hits_reached_count': int(event_counts['max_hits_reached_count']),
        'bisection_fallback_count': int(event_counts['bisection_fallback_count']),
        'nearest_projection_fallback_count': int(event_counts['nearest_projection_fallback_count']),
        'boundary_event_contract': boundary_event_contract,
        'boundary_event_contract_passed': int(boundary_event_contract['passed']),
        'boundary_event_failure_count': int(boundary_event_contract['failure_count']),
    }


def _stop_reason_report_fields(collision_diagnostics: Mapping[str, object]) -> Dict[str, object]:
    return {
        'invalid_mask_stop_reason_counts': dict(
            collision_diagnostics.get('invalid_mask_stop_reason_counts', {})
        ),
        'numerical_boundary_stop_reason_counts': dict(
            collision_diagnostics.get('numerical_boundary_stop_reason_counts', {})
        ),
    }


def build_runtime_report(
    payload: RuntimeOutputPayload,
    *,
    outputs_written: bool,
    context: RuntimeReportContext | None = None,
    invalid_stop_geometry_summary: Mapping[str, object] | None = None,
    state_geometry_summary: Mapping[str, object] | None = None,
    source_initial_geometry_summary: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    invalid_mask_stopped_count = int(payload.invalid_mask_stopped.sum())
    report_context = context or _build_runtime_report_context(
        payload,
        invalid_stop_geometry_summary=invalid_stop_geometry_summary,
        state_geometry_summary=state_geometry_summary,
        source_initial_geometry_summary=source_initial_geometry_summary,
    )
    event_counts = report_context.event_counts
    boundary_event_contract = report_context.boundary_event_contract
    diagnostic_counts = _diagnostic_int_fields(
        payload.collision_diagnostics,
        VALID_MASK_DIAGNOSTIC_FIELDS + CONTACT_DIAGNOSTIC_FIELDS,
    )
    contact_sliding_time_total_s = _diagnostic_float_field(
        payload.collision_diagnostics,
        'contact_sliding_time_total_s',
    )
    support_exit_summary = dict(report_context.summary_bundle['field_support_exit_summary'])
    return {
        'particle_count': int(payload.particles.count),
        **_shared_report_fields(payload),
        'released_count': int(payload.released.sum()),
        'stuck_count': int(payload.stuck.sum()),
        'absorbed_count': int(payload.absorbed.sum()),
        'field_support_exit_count': int(support_exit_summary['field_support_exit_count']),
        'physical_absorbed_count': int(support_exit_summary['physical_absorbed_count']),
        'contact_sliding_particle_count': int(payload.contact_sliding.sum()),
        'escaped_count': int(payload.escaped.sum()),
        'invalid_mask_stopped_count': int(invalid_mask_stopped_count),
        'numerical_boundary_stopped_count': int(event_counts['numerical_boundary_stopped_count']),
        'save_frame_count': int(len(payload.save_meta)),
        'outputs_written': int(bool(outputs_written)),
        'trajectory_written': int(bool(outputs_written) and _trajectory_output_enabled(payload) and len(payload.save_meta) > 0),
        'wall_events_written': int(
            bool(outputs_written)
            and int(payload.output_options.write_wall_events) != 0
            and len(payload.wall_rows) > 0
        ),
        'heavy_diagnostics_written': int(bool(outputs_written) and _heavy_diagnostics_enabled(payload)),
        'trajectory_suppressed': int(not _trajectory_output_enabled(payload)),
        'wall_events_suppressed': int(int(payload.output_options.write_wall_events) == 0),
        'heavy_diagnostics_suppressed': int(not _heavy_diagnostics_enabled(payload)),
        **_runtime_output_filenames(payload, outputs_written),
        'output_row_counts': _runtime_output_row_counts(payload),
        **_boundary_report_fields(event_counts, boundary_event_contract),
        'max_hit_unique_particle_count': int(report_context.max_hit_summary['unique_particle_count']),
        'max_hit_remaining_dt_total_s': float(report_context.max_hit_diag_summary['remaining_dt_total_s']),
        'max_hit_remaining_dt_max_s': float(report_context.max_hit_diag_summary['remaining_dt_max_s']),
        'max_hit_diagnostic_summary': report_context.max_hit_diag_summary,
        'contact_sliding_count': int(diagnostic_counts['contact_sliding_count']),
        'contact_sliding_time_total_s': float(contact_sliding_time_total_s),
        **{name: int(diagnostic_counts[name]) for name in CONTACT_DIAGNOSTIC_FIELDS if name != 'contact_sliding_count'},
        **_release_grace_report_fields(payload.collision_diagnostics),
        **_stop_reason_report_fields(payload.collision_diagnostics),
        **report_context.summary_bundle,
        'wall_law_counts': payload.wall_law_counts,
        'kernel_backend': f'numba_{payload.spatial_dim}d_freeflight',
        **{name: int(diagnostic_counts[name]) for name in VALID_MASK_DIAGNOSTIC_FIELDS},
    }


def _write_trajectory_plot(output_dir: Path, positions: np.ndarray, spatial_dim: int, plot_limit: int) -> None:
    import matplotlib.pyplot as plt

    particle_count = int(positions.shape[1]) if positions.ndim == 3 else 0
    if int(plot_limit) <= 0 or particle_count == 0:
        return
    if int(spatial_dim) == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        for i in range(min(particle_count, int(plot_limit))):
            arr = positions[:, i, :]
            ax.plot(arr[:, 0], arr[:, 1], alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Prepared-runtime trajectories 2D')
        fig.tight_layout()
        fig.savefig(output_dir / 'trajectories.png', dpi=150)
        plt.close(fig)
        return
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(min(particle_count, int(plot_limit))):
        arr = positions[:, i, :]
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Prepared-runtime trajectories 3D')
    fig.tight_layout()
    fig.savefig(output_dir / 'trajectories_3d.png', dpi=150)
    plt.close(fig)


def _write_force_contributions(payload: RuntimeOutputPayload, output_dir: Path) -> None:
    _write_rows_csv(
        output_dir / 'force_contributions.csv',
        _build_force_contribution_rows(payload),
        FORCE_CONTRIBUTION_COLUMNS,
    )


def _wall_summary_rows(payload: RuntimeOutputPayload) -> List[Dict[str, object]]:
    return [
        {
            'part_id': int(part_id),
            'outcome': str(outcome),
            'wall_mode': str(wall_mode),
            'count': int(count),
        }
        for (part_id, outcome, wall_mode), count in sorted(
            payload.wall_summary_counts.items(),
            key=lambda item: (-int(item[1]), int(item[0][0]), str(item[0][1]), str(item[0][2])),
        )
    ]


def _write_wall_tables(payload: RuntimeOutputPayload, output_dir: Path) -> None:
    output_options = payload.output_options
    if int(output_options.write_wall_events) != 0:
        wall_df = _attach_comsol_boundary_columns(_rows_frame(payload.wall_rows, WALL_EVENT_COLUMNS), payload)
        wall_df.to_csv(output_dir / 'wall_events.csv', index=False)
    if int(output_options.write_wall_summary) != 0:
        wall_summary_df = _attach_comsol_boundary_columns(
            _rows_frame(_wall_summary_rows(payload), WALL_SUMMARY_COLUMNS),
            payload,
        )
        wall_summary_df.to_csv(output_dir / 'wall_summary_by_part.csv', index=False)


def _write_optional_event_tables(payload: RuntimeOutputPayload, output_dir: Path) -> None:
    output_options = payload.output_options
    if int(output_options.write_max_hit_events) != 0:
        _write_rows_csv(output_dir / 'max_hit_events.csv', payload.max_hit_rows, MAX_HIT_EVENT_COLUMNS)
    if int(output_options.write_runtime_step_summary) != 0:
        _write_rows_csv(output_dir / 'runtime_step_summary.csv', payload.step_rows, RUNTIME_STEP_COLUMNS)


def _write_coating_outputs(payload: RuntimeOutputPayload, output_dir: Path) -> None:
    coating_rows = _build_coating_summary_rows(payload)
    _write_rows_csv(output_dir / 'coating_summary_by_part.csv', coating_rows, COATING_SUMMARY_COLUMNS)
    _write_json(output_dir / 'coating_summary.json', _build_coating_summary_report(coating_rows))


def write_runtime_outputs(payload: RuntimeOutputPayload, output_dir: Path) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_options = payload.output_options
    if int(output_options.write_positions) != 0:
        np.save(output_dir / f'positions_{payload.spatial_dim}d.npy', payload.positions)
    if int(output_options.write_segmented_positions) != 0:
        _save_segmented_positions(
            output_dir,
            payload.positions,
            payload.save_meta,
            payload.spatial_dim,
            skip_single_segment_arrays=int(output_options.write_positions) != 0,
        )
    elif payload.save_meta:
        _write_save_frame_metadata(output_dir, payload.save_meta)

    final_df = _build_final_particles_frame(payload)
    final_df.to_csv(output_dir / 'final_particles.csv', index=False)
    if int(output_options.write_force_contributions) != 0:
        _write_force_contributions(payload, output_dir)
    _write_wall_tables(payload, output_dir)
    _write_optional_event_tables(payload, output_dir)

    if int(output_options.write_prepared_summary) != 0:
        _write_json(output_dir / 'prepared_runtime_summary.json', prepared_runtime_summary(payload.prepared))
    if int(output_options.write_source_diagnostics) != 0:
        _write_resolved_particles(payload, output_dir)

    wall_summary_report = _build_wall_summary_report(payload.wall_summary_counts)
    wall_summary_report['field_support_exit_summary'] = _build_field_support_exit_summary(payload)
    if int(output_options.write_wall_summary) != 0:
        _write_json(output_dir / 'wall_summary.json', wall_summary_report)

    if int(output_options.write_coating_summary) != 0:
        _write_coating_outputs(payload, output_dir)

    wall_alignment_summary: Dict[str, object] = {'enabled': 0, 'skipped_by_output_mode': 1}
    if _heavy_diagnostics_enabled(payload):
        wall_alignment_summary, wall_alignment_rows = build_wall_catalog_alignment(
            generated_dir=_generated_dir_from_payload(payload),
            wall_catalog=payload.prepared.runtime.wall_catalog,
        )
        if wall_alignment_rows:
            write_wall_catalog_alignment_csv(output_dir / 'wall_catalog_alignment.csv', wall_alignment_rows)
            wall_alignment_summary['wall_catalog_alignment_file'] = 'wall_catalog_alignment.csv'
    payload.collision_diagnostics['wall_catalog_alignment'] = wall_alignment_summary

    lightweight_geometry = _lightweight_report_context_enabled(payload)
    invalid_stop_geometry_summary = None if bool(lightweight_geometry) else _build_invalid_stop_geometry_summary(payload)
    state_geometry_summary = None if bool(lightweight_geometry) else _build_state_geometry_summary(payload)
    source_initial_geometry_summary = (
        None if bool(lightweight_geometry) else _build_source_initial_geometry_summary(payload)
    )
    report_context = _build_runtime_report_context(
        payload,
        invalid_stop_geometry_summary=invalid_stop_geometry_summary,
        state_geometry_summary=state_geometry_summary,
        source_initial_geometry_summary=source_initial_geometry_summary,
        lightweight_geometry=bool(lightweight_geometry),
    )
    if int(payload.write_collision_diagnostics) != 0:
        collision_diag_report = _build_collision_diag_report(
            payload,
            context=report_context,
            invalid_stop_geometry_summary=invalid_stop_geometry_summary,
            state_geometry_summary=state_geometry_summary,
            source_initial_geometry_summary=source_initial_geometry_summary,
        )
        _write_json(output_dir / 'collision_diagnostics.json', collision_diag_report)

    report = build_runtime_report(
        payload,
        outputs_written=True,
        context=report_context,
        invalid_stop_geometry_summary=invalid_stop_geometry_summary,
        state_geometry_summary=state_geometry_summary,
        source_initial_geometry_summary=source_initial_geometry_summary,
    )
    _write_json(output_dir / 'solver_report.json', report)
    plasma_summary = dict(report.get('plasma_background', {}))
    if int(plasma_summary.get('enabled', 0)) != 0:
        _write_scalar_summary_csv(output_dir / 'plasma_background_summary.csv', plasma_summary)
    charge_summary = dict(report.get('charge_model', {}))
    if int(charge_summary.get('enabled', 0)) != 0:
        _write_scalar_summary_csv(output_dir / 'charge_model_summary.csv', charge_summary)

    if int(output_options.write_trajectory_plot) != 0:
        _write_trajectory_plot(output_dir, payload.positions, payload.spatial_dim, payload.plot_limit)
    return report
