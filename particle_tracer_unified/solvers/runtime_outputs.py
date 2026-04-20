from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

from ..core.datamodel import ParticleTable, PreparedRuntime
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

INVALID_STOP_REASON_NAMES = {
    0: '',
    1: 'freeflight_valid_mask_hard_invalid_prefix_clipped',
    2: 'freeflight_valid_mask_hard_invalid_retry_exhausted',
    3: 'collision_valid_mask_hard_invalid_prefix_clipped',
    4: 'collision_valid_mask_hard_invalid_retry_exhausted',
    255: 'unknown',
}

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


def _summary_float_or_nan(value: object) -> float:
    if value is None:
        return float('nan')
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return float('nan')
        try:
            return float(text)
        except ValueError:
            return float('nan')
    return float('nan')


def _summary_unit_for_key(key: str) -> str:
    name = str(key)
    suffix_units = (
        ('_m3', '1/m^3'),
        ('_kgm3', 'kg/m^3'),
        ('_m2Vs', 'm^2/(V s)'),
        ('_rad_s', 'rad/s'),
        ('_mps', 'm/s'),
        ('_eV', 'eV'),
        ('_Pa', 'Pa'),
        ('_K', 'K'),
        ('_amu', 'amu'),
        ('_kg', 'kg'),
        ('_Sm', 'S/m'),
        ('_s', 's'),
        ('_m2', 'm^2'),
        ('_m', 'm'),
        ('_C', 'C'),
        ('_e', 'e'),
    )
    for suffix, unit in suffix_units:
        if name.endswith(suffix):
            return unit
    return ''


def _build_scalar_summary_rows(values: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, value in values.items():
        if isinstance(value, Mapping) or isinstance(value, (list, tuple, set)):
            continue
        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                continue
            value = value.item()
        rows.append(
            {
                'quantity': str(key),
                'value': value,
                'unit': _summary_unit_for_key(str(key)),
            }
        )
    return rows


def _write_scalar_summary_csv(path: Path, values: Mapping[str, object]) -> bool:
    import pandas as pd

    rows = _build_scalar_summary_rows(values)
    if not rows:
        return False
    pd.DataFrame(rows, columns=['quantity', 'value', 'unit']).to_csv(path, index=False)
    return True


@dataclass(frozen=True)
class RuntimeOutputOptions:
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
    contact_tangent_motion_enabled: bool
    timing_s: Dict[str, float]
    memory_estimate_bytes: Dict[str, int]


def _runtime_coordinate_system(payload: RuntimeOutputPayload) -> str:
    return str(payload.prepared.runtime.coordinate_system)


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
        rows.append({'segment_name': safe, 'save_count': int(len(sub)), 't_start': float(sub['time_s'].min()), 't_end': float(sub['time_s'].max())})
    return df, rows


def _save_segmented_positions(
    output_dir: Path,
    positions: np.ndarray,
    save_meta: List[Dict[str, object]],
    spatial_dim: int,
    *,
    skip_single_segment_arrays: bool,
) -> None:
    import pandas as pd

    if positions.size == 0 or not save_meta:
        return
    df, rows = _segment_frame(save_meta)
    df.to_csv(output_dir / 'save_frames.csv', index=False)
    pd.DataFrame(rows).to_csv(output_dir / 'segment_summary.csv', index=False)
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
    import pandas as pd

    if not save_meta:
        return
    df, rows = _segment_frame(save_meta)
    df.to_csv(output_dir / 'save_frames.csv', index=False)
    pd.DataFrame(rows).to_csv(output_dir / 'segment_summary.csv', index=False)


def _build_final_particles_frame(payload: RuntimeOutputPayload) -> pd.DataFrame:
    import pandas as pd
    reason_codes = np.asarray(payload.invalid_stop_reason_code, dtype=np.uint8)
    reason_names = np.asarray(
        [INVALID_STOP_REASON_NAMES.get(int(code), 'unknown') for code in reason_codes],
        dtype=object,
    )

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


def _invalid_stop_reason_names(reason_codes: np.ndarray) -> List[str]:
    codes = np.asarray(reason_codes, dtype=np.uint8)
    return [INVALID_STOP_REASON_NAMES.get(int(code), 'unknown') or 'unknown' for code in codes]


def _top_part_reason_counts(part_ids: np.ndarray, reasons: List[str], *, limit: int = 12) -> List[Dict[str, object]]:
    counts = Counter((int(part_id), str(reason)) for part_id, reason in zip(part_ids, reasons))
    return [
        {'part_id': int(part_id), 'reason': str(reason), 'count': int(count)}
        for (part_id, reason), count in counts.most_common(int(limit))
    ]


def _particle_state_labels(payload: RuntimeOutputPayload) -> np.ndarray:
    labels = np.full(int(payload.particles.count), 'inactive', dtype=object)
    for name, mask in (
        ('active', payload.active),
        ('contact_sliding', payload.contact_sliding),
        ('invalid_mask_stopped', payload.invalid_mask_stopped),
        ('numerical_boundary_stopped', payload.numerical_boundary_stopped),
        ('stuck', payload.stuck),
        ('absorbed', payload.absorbed),
        ('escaped', payload.escaped),
    ):
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


def _geometry_features_for_positions(
    payload: RuntimeOutputPayload,
    positions: np.ndarray,
    velocities: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    runtime = payload.prepared.runtime
    pos = np.asarray(positions, dtype=np.float64)
    count = int(pos.shape[0])
    sdf_values = np.asarray([sample_geometry_sdf(runtime, p) for p in pos], dtype=np.float64)
    inside_values = np.asarray([inside_geometry(runtime, p, on_boundary_tol_m=0.0) for p in pos], dtype=bool)
    grid_part_ids = np.asarray([sample_geometry_part_id(runtime, p) for p in pos], dtype=np.int32)
    nearest_part_ids = grid_part_ids
    nearest_distances = np.abs(sdf_values)
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
        'sdf_m': sdf_values,
        'inside_geometry': inside_values,
        'nearest_part_id': nearest_part_ids,
        'nearest_boundary_distance_m': nearest_distances,
        'speed_mps': speed,
    }


def _empty_geometry_summary() -> Dict[str, object]:
    return {
        'count': 0,
        'sdf_m': {'count': 0},
        'abs_sdf_m': {'count': 0},
        'nearest_boundary_distance_m': {'count': 0},
        'speed_mps': {'count': 0},
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
    for state_name in (
        'active',
        'contact_sliding',
        'invalid_mask_stopped',
        'numerical_boundary_stopped',
        'stuck',
        'absorbed',
        'escaped',
        'inactive',
    ):
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
            'sdf_m': {'count': 0},
            'abs_sdf_m': {'count': 0},
            'nearest_boundary_distance_m': {'count': 0},
            'nearest_part_counts': [],
            'nearest_part_reason_counts': [],
        }

    positions = np.asarray(payload.final_position, dtype=np.float64)[stopped_mask]
    velocities = np.asarray(payload.final_velocity, dtype=np.float64)[stopped_mask]
    reasons = _invalid_stop_reason_names(np.asarray(payload.invalid_stop_reason_code, dtype=np.uint8)[stopped_mask])
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


def _build_final_state_consistency_summary(
    payload: RuntimeOutputPayload,
    state_summary: Mapping[str, object],
) -> Dict[str, object]:
    active = np.asarray(payload.active, dtype=bool)
    invalid = np.asarray(payload.invalid_mask_stopped, dtype=bool)
    numerical = np.asarray(payload.numerical_boundary_stopped, dtype=bool)
    stuck = np.asarray(payload.stuck, dtype=bool)
    absorbed = np.asarray(payload.absorbed, dtype=bool)
    contact_sliding = np.asarray(payload.contact_sliding, dtype=bool)
    contact_endpoint_stopped = np.asarray(payload.contact_endpoint_stopped, dtype=bool)
    escaped = np.asarray(payload.escaped, dtype=bool)
    state_matrix = np.vstack([active, invalid, numerical, stuck, absorbed, escaped])
    active_summary = (
        dict(state_summary.get('by_state', {}).get('active', {}))
        if isinstance(state_summary.get('by_state', {}), Mapping)
        else {}
    )
    contact_summary = (
        dict(state_summary.get('by_state', {}).get('contact_sliding', {}))
        if isinstance(state_summary.get('by_state', {}), Mapping)
        else {}
    )
    active_outside = int(active_summary.get('outside_geometry_count', 0))
    contact_outside = int(contact_summary.get('outside_geometry_count', 0))
    active_hard_invalid = _count_hard_invalid_final_positions(payload, active)
    nonfinite_position = int(np.count_nonzero(~np.all(np.isfinite(np.asarray(payload.final_position)), axis=1)))
    nonfinite_velocity = int(np.count_nonzero(~np.all(np.isfinite(np.asarray(payload.final_velocity)), axis=1)))
    return {
        'active_outside_geometry_count': int(active_outside),
        'contact_sliding_outside_geometry_count': int(contact_outside),
        'active_hard_invalid_count': int(active_hard_invalid),
        'multiple_terminal_state_count': int(np.count_nonzero(np.sum(state_matrix, axis=0) > 1)),
        'nonfinite_position_count': int(nonfinite_position),
        'nonfinite_velocity_count': int(nonfinite_velocity),
        'contact_sliding_particle_count': int(np.count_nonzero(contact_sliding)),
        'contact_endpoint_stopped_count': int(np.count_nonzero(contact_endpoint_stopped)),
        'numerical_boundary_stopped_count': int(np.count_nonzero(numerical)),
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
    outcome_sequences = [str(row.get('outcome_sequence', '')) for row in max_hit_rows if str(row.get('outcome_sequence', '')).strip()]
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


def _build_collision_diag_report(
    payload: RuntimeOutputPayload,
    *,
    invalid_stop_geometry_summary: Mapping[str, object] | None = None,
    state_geometry_summary: Mapping[str, object] | None = None,
    source_initial_geometry_summary: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    backend_report = field_backend_report(payload.prepared.runtime.field_provider)
    max_hit_summary = _build_max_hit_event_summary(payload.max_hit_rows)
    max_hit_diag_summary = _build_max_hit_diagnostic_summary(payload.collision_diagnostics)
    numerical_boundary_stopped_count = int(payload.numerical_boundary_stopped.sum())
    unresolved_crossing_count = int(payload.collision_diagnostics.get('unresolved_crossing_count', 0))
    max_hits_reached_count = int(payload.collision_diagnostics.get('max_hits_reached_count', 0))
    nearest_projection_fallback_count = int(payload.collision_diagnostics.get('nearest_projection_fallback_count', 0))
    boundary_event_contract = _build_boundary_event_contract_summary(
        numerical_boundary_stopped_count=numerical_boundary_stopped_count,
        unresolved_crossing_count=unresolved_crossing_count,
        max_hits_reached_count=max_hits_reached_count,
        nearest_projection_fallback_count=nearest_projection_fallback_count,
    )
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
    consistency_summary = _build_final_state_consistency_summary(payload, state_summary)
    support_exit_summary = _build_field_support_exit_summary(payload)
    return {
        **payload.collision_diagnostics,
        'coordinate_system': _runtime_coordinate_system(payload),
        'final_state_counts': _final_state_count_dict(payload),
        'numerical_boundary_stopped_count': int(numerical_boundary_stopped_count),
        'boundary_event_contract': boundary_event_contract,
        'boundary_event_contract_passed': int(boundary_event_contract['passed']),
        'boundary_event_failure_count': int(boundary_event_contract['failure_count']),
        'max_hit_event_summary': max_hit_summary,
        'max_hit_diagnostic_summary': max_hit_diag_summary,
        'invalid_stop_geometry_summary': invalid_stop_summary,
        'state_geometry_summary': state_summary,
        'final_state_consistency_summary': consistency_summary,
        'source_initial_geometry_summary': source_summary,
        'field_support_exit_summary': support_exit_summary,
        'timing_s': {str(k): float(v) for k, v in payload.timing_s.items()},
        'memory_estimate_bytes': {str(k): int(v) for k, v in payload.memory_estimate_bytes.items()},
        'integrator': str(payload.base_integrator_name),
        'valid_mask_policy': str(payload.valid_mask_policy),
        'drag_model': str(payload.drag_model),
        'drag_gas_properties': dict(payload.collision_diagnostics.get('drag_gas_properties', {})),
        'force_catalog': dict(payload.collision_diagnostics.get('force_catalog', {})),
        'force_runtime': dict(payload.collision_diagnostics.get('force_runtime', {})),
        'wall_catalog_alignment': dict(payload.collision_diagnostics.get('wall_catalog_alignment', {})),
        'stochastic_motion': dict(payload.collision_diagnostics.get('stochastic_motion', {})),
        'plasma_background': dict(payload.collision_diagnostics.get('plasma_background', {})),
        'charge_model': dict(payload.collision_diagnostics.get('charge_model', {})),
        'acceleration_source': str(payload.collision_diagnostics.get('acceleration_source', 'none')),
        'acceleration_quantity_names': list(payload.collision_diagnostics.get('acceleration_quantity_names', [])),
        'electric_field_names': list(payload.collision_diagnostics.get('electric_field_names', [])),
        'electric_q_over_m_Ckg': float(payload.collision_diagnostics.get('electric_q_over_m_Ckg', 0.0)),
        'contact_tangent_motion_enabled': int(bool(payload.contact_tangent_motion_enabled)),
        **backend_report,
        'max_wall_hits_per_step': int(payload.max_wall_hits_per_step),
        'min_remaining_dt_ratio': float(payload.min_remaining_dt_ratio),
        'on_boundary_tol_m': float(payload.on_boundary_tol_m),
        'epsilon_offset_m': float(payload.epsilon_offset_m),
        'adaptive_substep_enabled': int(payload.adaptive_substep_enabled),
        'adaptive_substep_tau_ratio': float(payload.adaptive_substep_tau_ratio),
        'adaptive_substep_max_splits': int(payload.adaptive_substep_max_splits),
    }


def build_runtime_report(
    payload: RuntimeOutputPayload,
    *,
    outputs_written: bool,
    invalid_stop_geometry_summary: Mapping[str, object] | None = None,
    state_geometry_summary: Mapping[str, object] | None = None,
    source_initial_geometry_summary: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    backend_report = field_backend_report(payload.prepared.runtime.field_provider)
    valid_mask_violation_count = int(payload.collision_diagnostics.get('valid_mask_violation_count', 0))
    valid_mask_violation_particle_count = int(payload.collision_diagnostics.get('valid_mask_violation_particle_count', 0))
    valid_mask_mixed_stencil_count = int(payload.collision_diagnostics.get('valid_mask_mixed_stencil_count', 0))
    valid_mask_mixed_stencil_particle_count = int(payload.collision_diagnostics.get('valid_mask_mixed_stencil_particle_count', 0))
    valid_mask_hard_invalid_count = int(payload.collision_diagnostics.get('valid_mask_hard_invalid_count', 0))
    valid_mask_hard_invalid_particle_count = int(payload.collision_diagnostics.get('valid_mask_hard_invalid_particle_count', 0))
    invalid_mask_stopped_count = int(payload.invalid_mask_stopped.sum())
    numerical_boundary_stopped_count = int(payload.numerical_boundary_stopped.sum())
    unresolved_crossing_count = int(payload.collision_diagnostics.get('unresolved_crossing_count', 0))
    max_hits_reached_count = int(payload.collision_diagnostics.get('max_hits_reached_count', 0))
    bisection_fallback_count = int(payload.collision_diagnostics.get('bisection_fallback_count', 0))
    nearest_projection_fallback_count = int(payload.collision_diagnostics.get('nearest_projection_fallback_count', 0))
    boundary_event_contract = _build_boundary_event_contract_summary(
        numerical_boundary_stopped_count=numerical_boundary_stopped_count,
        unresolved_crossing_count=unresolved_crossing_count,
        max_hits_reached_count=max_hits_reached_count,
        nearest_projection_fallback_count=nearest_projection_fallback_count,
    )
    contact_sliding_count = int(payload.collision_diagnostics.get('contact_sliding_count', 0))
    contact_sliding_time_total_s = float(payload.collision_diagnostics.get('contact_sliding_time_total_s', 0.0))
    contact_tangent_step_count = int(payload.collision_diagnostics.get('contact_tangent_step_count', 0))
    contact_release_count = int(payload.collision_diagnostics.get('contact_release_count', 0))
    contact_release_probe_reject_count = int(payload.collision_diagnostics.get('contact_release_probe_reject_count', 0))
    contact_endpoint_stop_count = int(payload.collision_diagnostics.get('contact_endpoint_stop_count', 0))
    contact_endpoint_hold_count = int(payload.collision_diagnostics.get('contact_endpoint_hold_count', 0))
    invalid_mask_retry_count = int(payload.collision_diagnostics.get('invalid_mask_retry_count', 0))
    invalid_mask_retry_exhausted_count = int(payload.collision_diagnostics.get('invalid_mask_retry_exhausted_count', 0))
    max_hit_summary = _build_max_hit_event_summary(payload.max_hit_rows)
    max_hit_diag_summary = _build_max_hit_diagnostic_summary(payload.collision_diagnostics)
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
    consistency_summary = _build_final_state_consistency_summary(payload, state_summary)
    output_options = payload.output_options
    support_exit_summary = _build_field_support_exit_summary(payload)
    return {
        'particle_count': int(payload.particles.count),
        'coordinate_system': _runtime_coordinate_system(payload),
        'final_state_counts': _final_state_count_dict(payload),
        'released_count': int(payload.released.sum()),
        'stuck_count': int(payload.stuck.sum()),
        'absorbed_count': int(payload.absorbed.sum()),
        'field_support_exit_count': int(support_exit_summary['field_support_exit_count']),
        'physical_absorbed_count': int(support_exit_summary['physical_absorbed_count']),
        'contact_sliding_particle_count': int(payload.contact_sliding.sum()),
        'escaped_count': int(payload.escaped.sum()),
        'invalid_mask_stopped_count': int(invalid_mask_stopped_count),
        'numerical_boundary_stopped_count': int(numerical_boundary_stopped_count),
        'save_frame_count': int(len(payload.save_meta)),
        'outputs_written': int(bool(outputs_written)),
        'positions_file': (
            f'positions_{payload.spatial_dim}d.npy'
            if bool(outputs_written) and int(output_options.write_positions) != 0
            else ''
        ),
        'timing_s': {str(k): float(v) for k, v in payload.timing_s.items()},
        'memory_estimate_bytes': {str(k): int(v) for k, v in payload.memory_estimate_bytes.items()},
        'output_row_counts': {
            'wall_events': int(len(payload.wall_rows)),
            'max_hit_events': int(len(payload.max_hit_rows)),
            'runtime_steps': int(len(payload.step_rows)),
            'coating_summary': int(len(payload.coating_summary_rows)),
        },
        'integrator': str(payload.base_integrator_name),
        'valid_mask_policy': str(payload.valid_mask_policy),
        'drag_model': str(payload.drag_model),
        'drag_gas_properties': dict(payload.collision_diagnostics.get('drag_gas_properties', {})),
        'force_catalog': dict(payload.collision_diagnostics.get('force_catalog', {})),
        'force_runtime': dict(payload.collision_diagnostics.get('force_runtime', {})),
        'wall_catalog_alignment': dict(payload.collision_diagnostics.get('wall_catalog_alignment', {})),
        'stochastic_motion': dict(payload.collision_diagnostics.get('stochastic_motion', {})),
        'plasma_background': dict(payload.collision_diagnostics.get('plasma_background', {})),
        'charge_model': dict(payload.collision_diagnostics.get('charge_model', {})),
        'acceleration_source': str(payload.collision_diagnostics.get('acceleration_source', 'none')),
        'acceleration_quantity_names': list(payload.collision_diagnostics.get('acceleration_quantity_names', [])),
        'electric_field_names': list(payload.collision_diagnostics.get('electric_field_names', [])),
        'electric_q_over_m_Ckg': float(payload.collision_diagnostics.get('electric_q_over_m_Ckg', 0.0)),
        'contact_tangent_motion_enabled': int(bool(payload.contact_tangent_motion_enabled)),
        'unresolved_crossing_count': int(unresolved_crossing_count),
        'max_hits_reached_count': int(max_hits_reached_count),
        'bisection_fallback_count': int(bisection_fallback_count),
        'nearest_projection_fallback_count': int(nearest_projection_fallback_count),
        'boundary_event_contract': boundary_event_contract,
        'boundary_event_contract_passed': int(boundary_event_contract['passed']),
        'boundary_event_failure_count': int(boundary_event_contract['failure_count']),
        'max_hit_unique_particle_count': int(max_hit_summary['unique_particle_count']),
        'max_hit_remaining_dt_total_s': float(max_hit_diag_summary['remaining_dt_total_s']),
        'max_hit_remaining_dt_max_s': float(max_hit_diag_summary['remaining_dt_max_s']),
        'max_hit_diagnostic_summary': max_hit_diag_summary,
        'contact_sliding_count': int(contact_sliding_count),
        'contact_sliding_time_total_s': float(contact_sliding_time_total_s),
        'contact_tangent_step_count': int(contact_tangent_step_count),
        'contact_release_count': int(contact_release_count),
        'contact_release_probe_reject_count': int(contact_release_probe_reject_count),
        'contact_endpoint_stop_count': int(contact_endpoint_stop_count),
        'contact_endpoint_hold_count': int(contact_endpoint_hold_count),
        'invalid_mask_retry_count': int(invalid_mask_retry_count),
        'invalid_mask_retry_exhausted_count': int(invalid_mask_retry_exhausted_count),
        'invalid_mask_stop_reason_counts': dict(
            payload.collision_diagnostics.get('invalid_mask_stop_reason_counts', {})
        ),
        'numerical_boundary_stop_reason_counts': dict(
            payload.collision_diagnostics.get('numerical_boundary_stop_reason_counts', {})
        ),
        'invalid_stop_geometry_summary': invalid_stop_summary,
        'state_geometry_summary': state_summary,
        'final_state_consistency_summary': consistency_summary,
        'source_initial_geometry_summary': source_summary,
        'field_support_exit_summary': support_exit_summary,
        'wall_law_counts': payload.wall_law_counts,
        'wall_summary_file': (
            'wall_summary.json' if bool(outputs_written) and int(output_options.write_wall_summary) != 0 else ''
        ),
        'wall_summary_by_part_file': (
            'wall_summary_by_part.csv' if bool(outputs_written) and int(output_options.write_wall_summary) != 0 else ''
        ),
        'max_hit_events_file': (
            'max_hit_events.csv' if bool(outputs_written) and int(output_options.write_max_hit_events) != 0 else ''
        ),
        'coating_summary_file': (
            'coating_summary_by_part.csv'
            if bool(outputs_written) and int(output_options.write_coating_summary) != 0
            else ''
        ),
        'plasma_background_summary_file': (
            'plasma_background_summary.csv'
            if bool(outputs_written)
            and int(dict(payload.collision_diagnostics.get('plasma_background', {})).get('enabled', 0)) != 0
            else ''
        ),
        'charge_model_summary_file': (
            'charge_model_summary.csv'
            if bool(outputs_written)
            and int(dict(payload.collision_diagnostics.get('charge_model', {})).get('enabled', 0)) != 0
            else ''
        ),
        'collision_diagnostics_file': (
            'collision_diagnostics.json'
            if bool(outputs_written) and int(payload.write_collision_diagnostics) != 0
            else ''
        ),
        'runtime_step_summary_file': (
            'runtime_step_summary.csv'
            if bool(outputs_written) and int(output_options.write_runtime_step_summary) != 0
            else ''
        ),
        'kernel_backend': f'numba_{payload.spatial_dim}d_freeflight',
        'valid_mask_violation_count': int(valid_mask_violation_count),
        'valid_mask_violation_particle_count': int(valid_mask_violation_particle_count),
        'valid_mask_mixed_stencil_count': int(valid_mask_mixed_stencil_count),
        'valid_mask_mixed_stencil_particle_count': int(valid_mask_mixed_stencil_particle_count),
        'valid_mask_hard_invalid_count': int(valid_mask_hard_invalid_count),
        'valid_mask_hard_invalid_particle_count': int(valid_mask_hard_invalid_particle_count),
        **backend_report,
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


def write_runtime_outputs(payload: RuntimeOutputPayload, output_dir: Path) -> Dict[str, object]:
    import pandas as pd

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

    wall_cols = [
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
    if int(output_options.write_wall_events) != 0:
        pd.DataFrame(payload.wall_rows, columns=wall_cols).to_csv(output_dir / 'wall_events.csv', index=False)

    wall_summary_rows = [
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
    if int(output_options.write_wall_summary) != 0:
        pd.DataFrame(wall_summary_rows, columns=['part_id', 'outcome', 'wall_mode', 'count']).to_csv(
            output_dir / 'wall_summary_by_part.csv',
            index=False,
        )

    max_hit_cols = [
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
    if int(output_options.write_max_hit_events) != 0:
        pd.DataFrame(payload.max_hit_rows, columns=max_hit_cols).to_csv(output_dir / 'max_hit_events.csv', index=False)

    step_cols = [
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
    if int(output_options.write_runtime_step_summary) != 0:
        pd.DataFrame(payload.step_rows, columns=step_cols).to_csv(output_dir / 'runtime_step_summary.csv', index=False)

    if int(output_options.write_prepared_summary) != 0:
        (output_dir / 'prepared_runtime_summary.json').write_text(
            json.dumps(prepared_runtime_summary(payload.prepared), indent=2),
            encoding='utf-8',
        )
    if int(output_options.write_source_diagnostics) != 0:
        _write_resolved_particles(payload, output_dir)

    wall_summary_report = _build_wall_summary_report(payload.wall_summary_counts)
    wall_summary_report['field_support_exit_summary'] = _build_field_support_exit_summary(payload)
    if int(output_options.write_wall_summary) != 0:
        (output_dir / 'wall_summary.json').write_text(json.dumps(wall_summary_report, indent=2), encoding='utf-8')

    if int(output_options.write_coating_summary) != 0:
        coating_rows = _build_coating_summary_rows(payload)
        coating_cols = [
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
        pd.DataFrame(coating_rows, columns=coating_cols).to_csv(output_dir / 'coating_summary_by_part.csv', index=False)
        (output_dir / 'coating_summary.json').write_text(
            json.dumps(_build_coating_summary_report(coating_rows), indent=2),
            encoding='utf-8',
        )

    wall_alignment_summary, wall_alignment_rows = build_wall_catalog_alignment(
        generated_dir=_generated_dir_from_payload(payload),
        wall_catalog=payload.prepared.runtime.wall_catalog,
    )
    if wall_alignment_rows:
        write_wall_catalog_alignment_csv(output_dir / 'wall_catalog_alignment.csv', wall_alignment_rows)
        wall_alignment_summary['wall_catalog_alignment_file'] = 'wall_catalog_alignment.csv'
    payload.collision_diagnostics['wall_catalog_alignment'] = wall_alignment_summary

    invalid_stop_geometry_summary = _build_invalid_stop_geometry_summary(payload)
    state_geometry_summary = _build_state_geometry_summary(payload)
    source_initial_geometry_summary = _build_source_initial_geometry_summary(payload)
    collision_diag_report = _build_collision_diag_report(
        payload,
        invalid_stop_geometry_summary=invalid_stop_geometry_summary,
        state_geometry_summary=state_geometry_summary,
        source_initial_geometry_summary=source_initial_geometry_summary,
    )
    if int(payload.write_collision_diagnostics) != 0:
        (output_dir / 'collision_diagnostics.json').write_text(json.dumps(collision_diag_report, indent=2), encoding='utf-8')

    report = build_runtime_report(
        payload,
        outputs_written=True,
        invalid_stop_geometry_summary=invalid_stop_geometry_summary,
        state_geometry_summary=state_geometry_summary,
        source_initial_geometry_summary=source_initial_geometry_summary,
    )
    (output_dir / 'solver_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plasma_summary = dict(report.get('plasma_background', {}))
    if int(plasma_summary.get('enabled', 0)) != 0:
        _write_scalar_summary_csv(output_dir / 'plasma_background_summary.csv', plasma_summary)
    charge_summary = dict(report.get('charge_model', {}))
    if int(charge_summary.get('enabled', 0)) != 0:
        _write_scalar_summary_csv(output_dir / 'charge_model_summary.csv', charge_summary)

    if int(output_options.write_trajectory_plot) != 0:
        _write_trajectory_plot(output_dir, payload.positions, payload.spatial_dim, payload.plot_limit)
    return report
