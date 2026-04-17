from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .boundary_core import inside_geometry
from .datamodel import PreparedRuntime
from .field_backend import field_backend_kind, field_backend_report, sample_field_valid_status
from .field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)


_STATUS_NAMES = {
    int(VALID_MASK_STATUS_CLEAN): 'clean',
    int(VALID_MASK_STATUS_MIXED_STENCIL): 'mixed_stencil',
    int(VALID_MASK_STATUS_HARD_INVALID): 'hard_invalid',
}


def _new_violation_part_summary(part_id: int, spatial_dim: int) -> Dict[str, Any]:
    inf = np.full(int(spatial_dim), np.inf, dtype=np.float64)
    ninf = np.full(int(spatial_dim), -np.inf, dtype=np.float64)
    return {
        'part_id': int(part_id),
        'violation_count': 0,
        'mixed_stencil': 0,
        'hard_invalid': 0,
        'no_interior_offset': 0,
        '_boundary_min': inf.copy(),
        '_boundary_max': ninf.copy(),
        '_offset_min': inf.copy(),
        '_offset_max': ninf.copy(),
    }


def _update_violation_part_summary(summary: Dict[str, Any], status_name: str, boundary_point: np.ndarray, offset_point: np.ndarray) -> None:
    summary['violation_count'] += 1
    if status_name in {'mixed_stencil', 'hard_invalid', 'no_interior_offset'}:
        summary[status_name] += 1
    boundary = np.asarray(boundary_point, dtype=np.float64)
    offset = np.asarray(offset_point, dtype=np.float64)
    summary['_boundary_min'] = np.minimum(np.asarray(summary['_boundary_min'], dtype=np.float64), boundary)
    summary['_boundary_max'] = np.maximum(np.asarray(summary['_boundary_max'], dtype=np.float64), boundary)
    finite_offset = np.isfinite(offset)
    if np.any(finite_offset):
        off_min = np.asarray(summary['_offset_min'], dtype=np.float64)
        off_max = np.asarray(summary['_offset_max'], dtype=np.float64)
        off_min[finite_offset] = np.minimum(off_min[finite_offset], offset[finite_offset])
        off_max[finite_offset] = np.maximum(off_max[finite_offset], offset[finite_offset])
        summary['_offset_min'] = off_min
        summary['_offset_max'] = off_max


def _finish_violation_part_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    def _finite_list(values: np.ndarray) -> List[float | None]:
        arr = np.asarray(values, dtype=np.float64)
        return [float(v) if np.isfinite(v) else None for v in arr.tolist()]

    return {
        'part_id': int(summary['part_id']),
        'violation_count': int(summary['violation_count']),
        'mixed_stencil': int(summary['mixed_stencil']),
        'hard_invalid': int(summary['hard_invalid']),
        'no_interior_offset': int(summary['no_interior_offset']),
        'boundary_min': _finite_list(summary['_boundary_min']),
        'boundary_max': _finite_list(summary['_boundary_max']),
        'offset_min': _finite_list(summary['_offset_min']),
        'offset_max': _finite_list(summary['_offset_max']),
    }


def _violation_csv_row(row: Dict[str, Any], spatial_dim: int) -> Dict[str, Any]:
    out = {
        'sample_index': int(row.get('sample_index', 0)),
        'part_id': int(row.get('part_id', 0)),
        'boundary_index': int(row.get('boundary_index', 0)),
        'sample_kind': str(row.get('sample_kind', '')),
        'checked_time_s': float(row.get('checked_time_s', 0.0)),
        'status': str(row.get('status', '')),
    }
    boundary = list(row.get('boundary_position', []))
    offset = list(row.get('offset_position', []))
    for dim, axis in enumerate(('x', 'y', 'z')[: int(spatial_dim)]):
        out[f'boundary_{axis}'] = boundary[dim] if dim < len(boundary) else ''
        out[f'offset_{axis}'] = offset[dim] if dim < len(offset) else ''
    return out


def _summary_csv_row(row: Dict[str, Any], spatial_dim: int) -> Dict[str, Any]:
    out = {
        'part_id': int(row.get('part_id', 0)),
        'violation_count': int(row.get('violation_count', 0)),
        'mixed_stencil': int(row.get('mixed_stencil', 0)),
        'hard_invalid': int(row.get('hard_invalid', 0)),
        'no_interior_offset': int(row.get('no_interior_offset', 0)),
    }
    for prefix, values in (
        ('boundary_min', list(row.get('boundary_min', []))),
        ('boundary_max', list(row.get('boundary_max', []))),
        ('offset_min', list(row.get('offset_min', []))),
        ('offset_max', list(row.get('offset_max', []))),
    ):
        for dim, axis in enumerate(('x', 'y', 'z')[: int(spatial_dim)]):
            value = values[dim] if dim < len(values) else None
            out[f'{prefix}_{axis}'] = '' if value is None else value
    return out


def _contract_config(prepared: PreparedRuntime) -> Tuple[str, float, int, int]:
    payload = prepared.runtime.config_payload if isinstance(prepared.runtime.config_payload, dict) else {}
    cfg = payload.get('provider_contract', {}) if isinstance(payload.get('provider_contract', {}), dict) else {}
    mode = str(cfg.get('boundary_field_support', 'strict')).strip().lower()
    if mode not in {'strict', 'warn', 'off'}:
        raise ValueError('provider_contract.boundary_field_support must be strict, warn, or off')
    offset_cells = float(cfg.get('boundary_offset_cells', 1.0))
    if not np.isfinite(offset_cells) or offset_cells <= 0.0:
        raise ValueError('provider_contract.boundary_offset_cells must be positive')
    max_samples = int(cfg.get('max_boundary_samples', 2000))
    if max_samples <= 0:
        raise ValueError('provider_contract.max_boundary_samples must be positive')
    max_time_samples = int(cfg.get('max_time_samples', 3))
    if max_time_samples <= 0:
        raise ValueError('provider_contract.max_time_samples must be positive')
    return mode, offset_cells, max_samples, max_time_samples


def _field_time_samples(field_provider, max_time_samples: int) -> List[float]:
    field = field_provider.field
    quantity_times: List[np.ndarray] = []
    for series in getattr(field, 'quantities', {}).values():
        times = np.asarray(getattr(series, 'times', np.asarray([0.0], dtype=np.float64)), dtype=np.float64)
        times = times[np.isfinite(times)]
        if times.size:
            quantity_times.append(times)
    if not quantity_times:
        return [0.0]
    base = quantity_times[0]
    if base.size <= 1:
        return [float(base[0])]
    limit = max(1, int(max_time_samples))
    if base.size <= limit:
        selected = base
    else:
        selected = base[np.linspace(0, base.size - 1, limit, dtype=np.int64)]
    return [float(v) for v in np.unique(selected)]


def _geometry_boundary_report(runtime) -> Dict[str, Any]:
    if runtime.geometry_provider is None:
        return {'available': False}
    geom = runtime.geometry_provider.geometry
    metadata = getattr(geom, 'metadata', {})
    if int(geom.spatial_dim) == 2:
        edges = geom.boundary_edges
        return {
            'available': edges is not None,
            'spatial_dim': 2,
            'boundary_edge_count': int(edges.shape[0]) if edges is not None else 0,
            'boundary_loop_count': int(metadata.get('boundary_loop_count_2d', len(getattr(geom, 'boundary_loops_2d', ())))),
            'boundary_edge_topology': metadata.get('boundary_edge_topology', {}),
        }
    triangles = geom.boundary_triangles
    validation = metadata.get('boundary_surface_validation', {})
    return {
        'available': triangles is not None,
        'spatial_dim': int(geom.spatial_dim),
        'boundary_triangle_count': int(triangles.shape[0]) if triangles is not None else 0,
        'boundary_triangle_part_id_count': int(geom.boundary_triangle_part_ids.shape[0])
        if geom.boundary_triangle_part_ids is not None
        else 0,
        'boundary_surface_validation': validation if isinstance(validation, dict) else {},
    }


def _cell_diagonal(axes: Tuple[np.ndarray, ...]) -> float:
    spacings = []
    for axis in axes:
        arr = np.asarray(axis, dtype=np.float64)
        if arr.size < 2:
            continue
        diffs = np.diff(arr)
        finite = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if finite.size:
            spacings.append(float(np.min(finite)))
    if not spacings:
        return 0.0
    return float(math.sqrt(sum(v * v for v in spacings)))


def _downsample_records(records: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if len(records) <= int(max_samples):
        return records
    indices = np.linspace(0, len(records) - 1, int(max_samples), dtype=np.int64)
    return [records[int(i)] for i in indices]


def _boundary_records_2d(runtime, offset_m: float, max_samples: int) -> List[Dict[str, Any]]:
    geom = runtime.geometry_provider.geometry
    edges = np.asarray(geom.boundary_edges, dtype=np.float64) if geom.boundary_edges is not None else np.zeros((0, 2, 2), dtype=np.float64)
    part_ids = (
        np.asarray(geom.boundary_edge_part_ids, dtype=np.int32)
        if geom.boundary_edge_part_ids is not None
        else np.ones(edges.shape[0], dtype=np.int32)
    )
    if part_ids.size != edges.shape[0]:
        part_ids = np.ones(edges.shape[0], dtype=np.int32)
    cell = max(_cell_diagonal(tuple(np.asarray(ax, dtype=np.float64) for ax in geom.axes)), float(offset_m))
    records: List[Dict[str, Any]] = []
    for edge_index, edge in enumerate(edges):
        p0 = np.asarray(edge[0], dtype=np.float64)
        p1 = np.asarray(edge[1], dtype=np.float64)
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if not np.isfinite(length) or length <= 0.0:
            continue
        steps = max(1, int(math.ceil(length / max(cell, 1.0e-30))))
        n0 = np.asarray([-delta[1], delta[0]], dtype=np.float64) / length
        for j in range(steps):
            frac = (float(j) + 0.5) / float(steps)
            point = p0 * (1.0 - frac) + p1 * frac
            records.append(
                {
                    'part_id': int(part_ids[edge_index]),
                    'boundary_index': int(edge_index),
                    'sample_kind': 'edge_midspan',
                    'boundary_point': point,
                    'normal_candidates': (n0, -n0),
                }
            )
    return _downsample_records(records, max_samples)


def _boundary_records_3d(runtime, offset_m: float, max_samples: int) -> List[Dict[str, Any]]:
    del offset_m
    geom = runtime.geometry_provider.geometry
    triangles = (
        np.asarray(geom.boundary_triangles, dtype=np.float64)
        if geom.boundary_triangles is not None
        else np.zeros((0, 3, 3), dtype=np.float64)
    )
    part_ids = (
        np.asarray(geom.boundary_triangle_part_ids, dtype=np.int32)
        if geom.boundary_triangle_part_ids is not None
        else np.ones(triangles.shape[0], dtype=np.int32)
    )
    if part_ids.size != triangles.shape[0]:
        part_ids = np.ones(triangles.shape[0], dtype=np.int32)
    records: List[Dict[str, Any]] = []
    for tri_index, tri in enumerate(triangles):
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        magnitude = float(np.linalg.norm(normal))
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            continue
        n0 = normal / magnitude
        sample_points = (
            ('face_centroid', np.mean(tri, axis=0)),
            ('edge_mid_0', 0.5 * (tri[0] + tri[1])),
            ('edge_mid_1', 0.5 * (tri[1] + tri[2])),
            ('edge_mid_2', 0.5 * (tri[2] + tri[0])),
            ('vertex_0', tri[0]),
            ('vertex_1', tri[1]),
            ('vertex_2', tri[2]),
        )
        for sample_kind, point in sample_points:
            records.append(
                {
                    'part_id': int(part_ids[tri_index]),
                    'boundary_index': int(tri_index),
                    'sample_kind': str(sample_kind),
                    'boundary_point': np.asarray(point, dtype=np.float64),
                    'normal_candidates': (n0, -n0),
                }
            )
    return _downsample_records(records, max_samples)


def _inside_offset_point(runtime, point: np.ndarray, normals: Tuple[np.ndarray, ...], offset_m: float) -> np.ndarray | None:
    for normal in normals:
        candidate = np.asarray(point, dtype=np.float64) + np.asarray(normal, dtype=np.float64) * float(offset_m)
        if inside_geometry(runtime, candidate, on_boundary_tol_m=0.0):
            return candidate
    return None


def _empty_report(mode: str, reason: str, field_kind: str = '') -> Dict[str, Any]:
    return {
        'mode': mode,
        'passed': True,
        'applicable': False,
        'reason': reason,
        'field_backend_kind': field_kind,
        'field_support': {},
        'geometry_boundary': {},
        'time_axis_mismatch_count': 0,
        'boundary_offset_cells': 0.0,
        'boundary_offset_m': 0.0,
        'checked_times_s': [],
        'checked_time_count': 0,
        'boundary_sample_kind_counts': {},
        'sample_count': 0,
        'status_counts': {'clean': 0, 'mixed_stencil': 0, 'hard_invalid': 0, 'non_clean': 0, 'no_interior_offset': 0},
        'per_part': [],
        'violation_summary_by_part': [],
        'violation_count': 0,
        'violations_truncated': False,
        'violations': [],
    }


def build_boundary_field_support_report(prepared: PreparedRuntime) -> Dict[str, Any]:
    runtime = prepared.runtime
    mode, offset_cells, max_samples, max_time_samples = _contract_config(prepared)
    if mode == 'off':
        return _empty_report(mode, 'provider contract disabled')
    if runtime.geometry_provider is None:
        return _empty_report(mode, 'no geometry provider')
    if runtime.field_provider is None:
        return _empty_report(mode, 'no field provider')
    geom = runtime.geometry_provider.geometry
    field_kind = str(field_backend_kind(runtime.field_provider))
    has_boundary = (int(geom.spatial_dim) == 2 and geom.boundary_edges is not None) or (
        int(geom.spatial_dim) == 3 and geom.boundary_triangles is not None
    )
    if not has_boundary:
        return _empty_report(mode, 'geometry provider has no explicit boundary surface', field_kind)
    offset_m = _cell_diagonal(tuple(np.asarray(ax, dtype=np.float64) for ax in geom.axes)) * float(offset_cells)
    if not np.isfinite(offset_m) or offset_m <= 0.0:
        raise ValueError('Could not compute a positive provider boundary offset from geometry axes')
    records = (
        _boundary_records_2d(runtime, offset_m, max_samples)
        if int(geom.spatial_dim) == 2
        else _boundary_records_3d(runtime, offset_m, max_samples)
    )
    checked_times = _field_time_samples(runtime.field_provider, max_time_samples)
    status_counts = {'clean': 0, 'mixed_stencil': 0, 'hard_invalid': 0, 'non_clean': 0, 'no_interior_offset': 0}
    sample_kind_counts: Dict[str, int] = {}
    for rec in records:
        kind = str(rec.get('sample_kind', ''))
        sample_kind_counts[kind] = int(sample_kind_counts.get(kind, 0)) + 1
    per_part: Dict[int, Dict[str, int]] = {}
    violation_summary_by_part: Dict[int, Dict[str, Any]] = {}
    violations: List[Dict[str, Any]] = []
    sample_index = 0
    for rec in records:
        part_id = int(rec['part_id'])
        part = per_part.setdefault(
            part_id,
            {'part_id': part_id, 'sample_count': 0, 'clean': 0, 'mixed_stencil': 0, 'hard_invalid': 0, 'non_clean': 0, 'no_interior_offset': 0},
        )
        offset_point = _inside_offset_point(runtime, rec['boundary_point'], rec['normal_candidates'], offset_m)
        for checked_time_s in checked_times:
            part['sample_count'] += 1
            if offset_point is None:
                status_name = 'no_interior_offset'
                status_counts[status_name] += 1
                status_counts['non_clean'] += 1
                part[status_name] += 1
                part['non_clean'] += 1
            else:
                status = int(sample_field_valid_status(runtime.field_provider, offset_point, float(checked_time_s)))
                status_name = _STATUS_NAMES.get(status, 'hard_invalid')
                status_counts[status_name] += 1
                if status != int(VALID_MASK_STATUS_CLEAN):
                    status_counts['non_clean'] += 1
                    part['non_clean'] += 1
                part[status_name] += 1
            if status_name != 'clean':
                point = np.asarray(rec['boundary_point'], dtype=np.float64)
                off = np.asarray(offset_point, dtype=np.float64) if offset_point is not None else np.full(point.shape, np.nan)
                part_summary = violation_summary_by_part.setdefault(
                    part_id,
                    _new_violation_part_summary(part_id, int(geom.spatial_dim)),
                )
                _update_violation_part_summary(part_summary, status_name, point, off)
                violations.append(
                    {
                        'sample_index': int(sample_index),
                        'part_id': int(part_id),
                        'boundary_index': int(rec['boundary_index']),
                        'sample_kind': str(rec.get('sample_kind', '')),
                        'checked_time_s': float(checked_time_s),
                        'status': status_name,
                        'boundary_position': [float(v) for v in point.tolist()],
                        'offset_position': [float(v) for v in off.tolist()],
                    }
                )
            sample_index += 1
    backend_report = field_backend_report(runtime.field_provider)
    time_axis = backend_report.get('time_axis', {}) if isinstance(backend_report, dict) else {}
    time_axis_mismatch_count = int(time_axis.get('quantity_time_axis_mismatch_count', 0)) if isinstance(time_axis, dict) else 0
    return {
        'mode': mode,
        'passed': int(status_counts['non_clean']) == 0 and time_axis_mismatch_count == 0,
        'applicable': True,
        'reason': '',
        'field_backend_kind': field_kind,
        'field_support': backend_report,
        'geometry_boundary': _geometry_boundary_report(runtime),
        'time_axis_mismatch_count': int(time_axis_mismatch_count),
        'spatial_dim': int(geom.spatial_dim),
        'boundary_offset_cells': float(offset_cells),
        'boundary_offset_m': float(offset_m),
        'checked_times_s': [float(v) for v in checked_times],
        'checked_time_count': int(len(checked_times)),
        'boundary_point_sample_count': int(len(records)),
        'boundary_sample_kind_counts': dict(sorted(sample_kind_counts.items())),
        'sample_count': int(len(records) * len(checked_times)),
        'status_counts': {k: int(v) for k, v in status_counts.items()},
        'per_part': sorted(({k: int(v) for k, v in part.items()} for part in per_part.values()), key=lambda item: item['part_id']),
        'violation_summary_by_part': sorted(
            (_finish_violation_part_summary(summary) for summary in violation_summary_by_part.values()),
            key=lambda item: item['part_id'],
        ),
        'violation_count': int(len(violations)),
        'violations_truncated': False,
        'violations': violations,
    }


def write_provider_contract_report(prepared: PreparedRuntime, output_dir: Path) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_boundary_field_support_report(prepared)
    (out / 'provider_contract_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    violations = report.get('violations', [])
    summaries = report.get('violation_summary_by_part', [])
    violations_path = out / 'provider_boundary_violations.csv'
    summary_path = out / 'provider_boundary_summary.csv'
    if violations:
        spatial_dim = int(report.get('spatial_dim', 0))
        axis_names = ('x', 'y', 'z')[:spatial_dim]
        fieldnames = ['sample_index', 'part_id', 'boundary_index', 'sample_kind', 'checked_time_s', 'status']
        fieldnames.extend([f'boundary_{axis}' for axis in axis_names])
        fieldnames.extend([f'offset_{axis}' for axis in axis_names])
        with violations_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in violations:
                writer.writerow(_violation_csv_row(row, spatial_dim))
    elif violations_path.exists():
        violations_path.unlink()
    if summaries:
        spatial_dim = int(report.get('spatial_dim', 0))
        axis_names = ('x', 'y', 'z')[:spatial_dim]
        fieldnames = ['part_id', 'violation_count', 'mixed_stencil', 'hard_invalid', 'no_interior_offset']
        for prefix in ('boundary_min', 'boundary_max', 'offset_min', 'offset_max'):
            fieldnames.extend([f'{prefix}_{axis}' for axis in axis_names])
        with summary_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(_summary_csv_row(row, spatial_dim))
    elif summary_path.exists():
        summary_path.unlink()
    return report


def enforce_boundary_field_support(prepared: PreparedRuntime, output_dir: Path) -> Dict[str, Any]:
    report = write_provider_contract_report(prepared, output_dir)
    if str(report.get('mode', 'strict')) == 'strict' and not bool(report.get('passed', True)):
        counts = report.get('status_counts', {})
        time_axis_mismatch_count = int(report.get('time_axis_mismatch_count', 0))
        raise ValueError(
            'Field provider does not cover the explicit boundary support domain; '
            f"non_clean={int(counts.get('non_clean', 0))}, "
            f"mixed_stencil={int(counts.get('mixed_stencil', 0))}, "
            f"hard_invalid={int(counts.get('hard_invalid', 0))}, "
            f"time_axis_mismatch={time_axis_mismatch_count}. "
            'Fix the field/boundary export bundle or use a mesh-native field provider.'
        )
    return report
