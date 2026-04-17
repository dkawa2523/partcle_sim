from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .boundary_service import inside_geometry, nearest_boundary_edge_features_2d, sample_geometry_part_id, sample_geometry_sdf
from .datamodel import PreparedRuntime
from .field_backend import field_backend_kind, sample_field_valid_status
from .field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)


STATUS_NAMES = {
    int(VALID_MASK_STATUS_CLEAN): 'clean',
    int(VALID_MASK_STATUS_MIXED_STENCIL): 'mixed_stencil',
    int(VALID_MASK_STATUS_HARD_INVALID): 'hard_invalid',
}


def _initial_support_mode(config_payload: Mapping[str, Any]) -> str:
    cfg = config_payload.get('input_contract', {}) if isinstance(config_payload, Mapping) else {}
    if not isinstance(cfg, Mapping):
        return 'strict'
    mode = str(cfg.get('initial_particle_field_support', 'strict')).strip().lower()
    if mode in {'', 'true', '1'}:
        return 'strict'
    if mode in {'false', '0', 'disabled'}:
        return 'off'
    if mode not in {'strict', 'warn', 'off'}:
        raise ValueError('input_contract.initial_particle_field_support must be strict, warn, or off')
    return mode


def _axis_spacing_summary(axes: Tuple[np.ndarray, ...]) -> Dict[str, float]:
    min_steps: List[float] = []
    max_steps: List[float] = []
    for axis in axes:
        values = np.asarray(axis, dtype=np.float64)
        diffs = np.diff(values)
        positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if positive.size:
            min_steps.append(float(np.min(positive)))
            max_steps.append(float(np.max(positive)))
    if not min_steps:
        return {'min_axis_step_m': 0.0, 'max_cell_diagonal_m': 0.0}
    return {
        'min_axis_step_m': float(min(min_steps)),
        'max_cell_diagonal_m': float(np.sqrt(np.sum(np.square(max_steps)))),
    }


def _geometry_features(runtime, positions: np.ndarray) -> Dict[str, np.ndarray]:
    count = int(positions.shape[0])
    sdf = np.full(count, np.nan, dtype=np.float64)
    grid_part_id = np.zeros(count, dtype=np.int32)
    inside = np.zeros(count, dtype=bool)
    inside_strict = np.zeros(count, dtype=bool)
    nearest_part_id = grid_part_id.copy()
    nearest_distance = np.full(count, np.nan, dtype=np.float64)
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None:
        return {
            'sdf_m': sdf,
            'grid_part_id': grid_part_id,
            'inside_geometry': inside,
            'inside_geometry_strict': inside_strict,
            'nearest_part_id': nearest_part_id,
            'nearest_boundary_distance_m': nearest_distance,
        }

    for i, point in enumerate(positions):
        p = np.asarray(point, dtype=np.float64)
        sdf[i] = float(sample_geometry_sdf(runtime, p))
        grid_part_id[i] = int(sample_geometry_part_id(runtime, p))
        inside[i] = bool(inside_geometry(runtime, p, on_boundary_tol_m=1.0e-12))
        inside_strict[i] = bool(inside_geometry(runtime, p, on_boundary_tol_m=0.0))
    nearest_part_id = grid_part_id.copy()
    nearest_distance = np.abs(sdf)
    if (
        int(getattr(runtime, 'spatial_dim', 0)) == 2
        and getattr(geometry_provider.geometry, 'boundary_edges', None) is not None
    ):
        edge_part_ids, edge_distances = nearest_boundary_edge_features_2d(runtime, positions)
        finite = np.isfinite(edge_distances)
        nearest_part_id[finite] = np.asarray(edge_part_ids, dtype=np.int32)[finite]
        nearest_distance[finite] = np.asarray(edge_distances, dtype=np.float64)[finite]
    return {
        'sdf_m': sdf,
        'grid_part_id': grid_part_id,
        'inside_geometry': inside,
        'inside_geometry_strict': inside_strict,
        'nearest_part_id': nearest_part_id,
        'nearest_boundary_distance_m': nearest_distance,
    }


def build_initial_particle_field_support_report(prepared: PreparedRuntime) -> Dict[str, Any]:
    runtime = prepared.runtime
    particles = runtime.particles
    field_provider = runtime.field_provider
    mode = _initial_support_mode(runtime.config_payload if isinstance(runtime.config_payload, Mapping) else {})
    if particles is None:
        raise ValueError('Simulation requires particles')
    if field_provider is None:
        return {
            'mode': mode,
            'passed': True,
            'particle_count': int(particles.count),
            'field_backend_kind': '',
            'status_counts': {'clean': int(particles.count), 'mixed_stencil': 0, 'hard_invalid': 0, 'non_clean': 0},
            'violations': [],
            'notes': ['No field provider is configured; initial field-support check is not applicable.'],
        }

    positions = np.asarray(particles.position[:, : int(runtime.spatial_dim)], dtype=np.float64)
    release_times = np.asarray(particles.release_time, dtype=np.float64)
    statuses = np.asarray(
        [
            sample_field_valid_status(field_provider, p, float(release_times[i]))
            for i, p in enumerate(positions)
        ],
        dtype=np.uint8,
    )
    clean = int(np.count_nonzero(statuses == int(VALID_MASK_STATUS_CLEAN)))
    mixed = int(np.count_nonzero(statuses == int(VALID_MASK_STATUS_MIXED_STENCIL)))
    hard = int(np.count_nonzero(statuses == int(VALID_MASK_STATUS_HARD_INVALID)))
    non_clean = int(mixed + hard)
    geometry_features = _geometry_features(runtime, positions)
    field = field_provider.field
    spacing = _axis_spacing_summary(tuple(getattr(field, 'axes', ())))
    near_threshold = float(spacing.get('max_cell_diagonal_m', 0.0))
    nearest_distance = np.asarray(geometry_features['nearest_boundary_distance_m'], dtype=np.float64)
    near_boundary = np.isfinite(nearest_distance) & (nearest_distance <= near_threshold) if near_threshold > 0.0 else np.zeros(particles.count, dtype=bool)

    violations: List[Dict[str, Any]] = []
    for idx in np.flatnonzero(statuses != int(VALID_MASK_STATUS_CLEAN)):
        row: Dict[str, Any] = {
            'particle_id': int(particles.particle_id[int(idx)]),
            'status': STATUS_NAMES.get(int(statuses[int(idx)]), 'unknown'),
            'source_part_id': int(particles.source_part_id[int(idx)]),
            'material_id': int(particles.material_id[int(idx)]),
            'release_time_s': float(particles.release_time[int(idx)]),
            'checked_time_s': float(release_times[int(idx)]),
            'geometry_inside': int(bool(geometry_features['inside_geometry'][int(idx)])),
            'geometry_inside_strict': int(bool(geometry_features['inside_geometry_strict'][int(idx)])),
            'geometry_sdf_m': float(geometry_features['sdf_m'][int(idx)]),
            'nearest_boundary_distance_m': float(nearest_distance[int(idx)]),
            'nearest_boundary_part_id': int(geometry_features['nearest_part_id'][int(idx)]),
            'near_boundary_by_cell_diagonal': int(bool(near_boundary[int(idx)])),
        }
        for dim, name in enumerate(('x', 'y', 'z')[: int(runtime.spatial_dim)]):
            row[name] = float(positions[int(idx), dim])
        violations.append(row)

    return {
        'mode': mode,
        'passed': bool(non_clean == 0 or mode in {'warn', 'off'}),
        'particle_count': int(particles.count),
        'field_backend_kind': str(field_backend_kind(field_provider)),
        'time_mode': str(getattr(field, 'time_mode', 'steady')),
        'checked_time_min_s': float(np.nanmin(release_times)) if release_times.size else 0.0,
        'checked_time_max_s': float(np.nanmax(release_times)) if release_times.size else 0.0,
        'status_counts': {
            'clean': int(clean),
            'mixed_stencil': int(mixed),
            'hard_invalid': int(hard),
            'non_clean': int(non_clean),
        },
        'near_boundary_threshold_m': float(near_threshold),
        'non_clean_near_boundary_count': int(np.count_nonzero((statuses != int(VALID_MASK_STATUS_CLEAN)) & near_boundary)),
        'non_clean_geometry_inside_count': int(
            np.count_nonzero((statuses != int(VALID_MASK_STATUS_CLEAN)) & np.asarray(geometry_features['inside_geometry'], dtype=bool))
        ),
        'violations': violations,
    }


def write_input_contract_report(prepared: PreparedRuntime, output_dir: Path) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_initial_particle_field_support_report(prepared)
    (out / 'input_contract_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    rows = list(report.get('violations', []))
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (out / 'input_particle_violations.csv').open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return report


def enforce_initial_particle_field_support(prepared: PreparedRuntime, output_dir: Path) -> Dict[str, Any]:
    mode = _initial_support_mode(prepared.runtime.config_payload if isinstance(prepared.runtime.config_payload, Mapping) else {})
    if mode == 'off':
        return build_initial_particle_field_support_report(prepared)
    report = write_input_contract_report(prepared, output_dir)
    non_clean = int(report.get('status_counts', {}).get('non_clean', 0))
    if mode == 'strict' and non_clean > 0:
        raise ValueError(
            'Initial particles must be inside the clean field sample domain; '
            f'found {non_clean} non-clean particles. '
            f'See {Path(output_dir) / "input_contract_report.json"}'
        )
    return report


__all__ = (
    'STATUS_NAMES',
    'build_initial_particle_field_support_report',
    'enforce_initial_particle_field_support',
    'write_input_contract_report',
)
