from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml

from particle_tracer_unified.core.boundary_service import (
    nearest_boundary_edge_features_2d,
    sample_geometry_part_id,
    sample_geometry_sdf,
)
from particle_tracer_unified.io.runtime_builder import build_runtime_from_config

try:
    from tools.state_contract import particle_class_frame
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from state_contract import particle_class_frame


def _resolve_path(repo_root: Path, path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else (repo_root / path_value).resolve()


def _run_case(repo_root: Path, config_path: Path, output_dir: Path, timeout_s: float = 0.0) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(repo_root / 'run_from_yaml.py'), str(config_path), '--output-dir', str(output_dir)]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, cwd=str(repo_root), check=False, timeout=float(timeout_s) if float(timeout_s) > 0.0 else None)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        raise TimeoutError(f'run_from_yaml timed out after {elapsed:.1f}s for {config_path}') from exc
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f'run_from_yaml failed for {config_path} with exit code {result.returncode}')
    return float(elapsed)


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'YAML root must be a mapping: {path}')
    return payload


def _absolutize_config_paths(config: Dict[str, Any], source_config: Path) -> None:
    base_dir = source_config.parent
    paths_cfg = config.get('paths', {})
    if isinstance(paths_cfg, dict):
        for key, value in list(paths_cfg.items()):
            if value is None or str(value).strip() == '':
                continue
            path = Path(str(value))
            paths_cfg[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
    providers_cfg = config.get('providers', {})
    if isinstance(providers_cfg, dict):
        for section in ('geometry', 'field'):
            provider = providers_cfg.get(section, {})
            if isinstance(provider, dict) and provider.get('npz_path') is not None:
                path = Path(str(provider['npz_path']))
                provider['npz_path'] = str(path if path.is_absolute() else (base_dir / path).resolve())


def _write_config_variant(
    *,
    source_config: Path,
    output_config: Path,
    override_t_end: float | None,
    artifact_mode: str | None,
) -> Path:
    config = _load_yaml_mapping(source_config)
    _absolutize_config_paths(config, source_config)
    if override_t_end is not None:
        config.setdefault('solver', {})['t_end'] = float(override_t_end)
    if artifact_mode is not None:
        config.setdefault('output', {})['artifact_mode'] = str(artifact_mode)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return output_config


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def _load_final_particles(output_dir: Path) -> pd.DataFrame:
    final_csv = output_dir / 'final_particles.csv'
    if not final_csv.exists():
        raise FileNotFoundError(final_csv)
    return pd.read_csv(final_csv)


def _particle_classes(df: pd.DataFrame) -> pd.DataFrame:
    required = {'particle_id', 'active', 'stuck', 'absorbed', 'escaped'}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f'final_particles.csv is missing required columns: {missing}')
    return particle_class_frame(df)


def class_match_ratio(candidate_final: pd.DataFrame, reference_final: pd.DataFrame) -> Tuple[float, int]:
    candidate_classes = _particle_classes(candidate_final)
    reference_classes = _particle_classes(reference_final)
    merged = candidate_classes.merge(reference_classes, on='particle_id', how='inner', suffixes=('_candidate', '_reference'))
    if merged.empty:
        return 0.0, 0
    matches = merged['particle_class_candidate'].astype(str) == merged['particle_class_reference'].astype(str)
    return float(matches.mean()), int(len(merged))


def class_transition_summary(candidate_final: pd.DataFrame, reference_final: pd.DataFrame, *, top_n: int = 12) -> Dict[str, Any]:
    candidate_classes = _particle_classes(candidate_final)
    reference_classes = _particle_classes(reference_final)
    merged = candidate_classes.merge(reference_classes, on='particle_id', how='inner', suffixes=('_candidate', '_reference'))
    if merged.empty:
        return {'compared_particles': 0, 'mismatch_count': 0, 'top_transitions': []}
    matches = merged['particle_class_candidate'].astype(str) == merged['particle_class_reference'].astype(str)
    transitions = (
        merged.groupby(['particle_class_reference', 'particle_class_candidate'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    top_rows = [
        {
            'reference_class': str(row['particle_class_reference']),
            'candidate_class': str(row['particle_class_candidate']),
            'count': int(row['count']),
        }
        for _, row in transitions.head(int(top_n)).iterrows()
    ]
    mismatch_rows = transitions[transitions['particle_class_reference'] != transitions['particle_class_candidate']]
    top_mismatches = [
        {
            'reference_class': str(row['particle_class_reference']),
            'candidate_class': str(row['particle_class_candidate']),
            'count': int(row['count']),
        }
        for _, row in mismatch_rows.head(int(top_n)).iterrows()
    ]
    return {
        'compared_particles': int(len(merged)),
        'mismatch_count': int((~matches).sum()),
        'top_transitions': top_rows,
        'top_mismatches': top_mismatches,
    }


def _finite_summary(values: np.ndarray) -> Dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {'count': 0}
    return {
        'count': int(finite.size),
        'min': float(np.min(finite)),
        'p50': float(np.percentile(finite, 50.0)),
        'p90': float(np.percentile(finite, 90.0)),
        'p99': float(np.percentile(finite, 99.0)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
    }


def _final_spatial_dim(final_df: pd.DataFrame) -> int:
    return int(sum(1 for name in ('x', 'y', 'z') if name in final_df.columns))


def _final_position_array(final_df: pd.DataFrame, spatial_dim: int) -> np.ndarray:
    names = ['x', 'y', 'z'][: int(spatial_dim)]
    missing = [name for name in names if name not in final_df.columns]
    if missing:
        raise ValueError(f'final_particles.csv is missing position columns: {missing}')
    return final_df[names].to_numpy(dtype=np.float64)


def _final_velocity_array(final_df: pd.DataFrame, spatial_dim: int) -> np.ndarray:
    values: List[np.ndarray] = []
    for name in ['x', 'y', 'z'][: int(spatial_dim)]:
        col = f'v_{name}'
        if col in final_df.columns:
            values.append(final_df[col].to_numpy(dtype=np.float64))
        else:
            values.append(np.zeros(len(final_df), dtype=np.float64))
    if not values:
        return np.zeros((len(final_df), 0), dtype=np.float64)
    return np.stack(values, axis=1)


def _feature_near_boundary_threshold_m(runtime) -> float:
    geometry_provider = getattr(runtime, 'geometry_provider', None)
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


def _geometry_feature_frame(runtime, final_df: pd.DataFrame) -> pd.DataFrame:
    spatial_dim = int(getattr(runtime, 'spatial_dim', _final_spatial_dim(final_df)))
    spatial_dim = min(spatial_dim, _final_spatial_dim(final_df))
    positions = _final_position_array(final_df, spatial_dim)
    velocities = _final_velocity_array(final_df, spatial_dim)
    sdf_values = np.asarray([sample_geometry_sdf(runtime, pos) for pos in positions], dtype=np.float64)
    nearest_part_ids = np.asarray([sample_geometry_part_id(runtime, pos) for pos in positions], dtype=np.int32)
    nearest_distances = np.abs(sdf_values)
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if (
        spatial_dim == 2
        and geometry_provider is not None
        and getattr(geometry_provider.geometry, 'boundary_edges', None) is not None
    ):
        edge_part_ids, edge_distances = nearest_boundary_edge_features_2d(runtime, positions)
        finite_edge = np.isfinite(edge_distances)
        if np.any(finite_edge):
            nearest_part_ids = np.where(finite_edge, edge_part_ids, nearest_part_ids)
            nearest_distances = np.where(finite_edge, edge_distances, nearest_distances)

    frame = pd.DataFrame(
        {
            'particle_id': final_df['particle_id'].astype(np.int64),
            'sdf_m': sdf_values,
            'abs_sdf_m': np.abs(sdf_values),
            'nearest_boundary_distance_m': nearest_distances,
            'nearest_part_id': nearest_part_ids.astype(np.int32),
            'speed_mps': np.linalg.norm(velocities, axis=1) if velocities.size else np.zeros(len(final_df), dtype=np.float64),
        }
    )
    for axis_index, axis_name in enumerate(['x', 'y', 'z'][:spatial_dim]):
        frame[axis_name] = positions[:, axis_index]
    classes = particle_class_frame(final_df)
    return frame.merge(classes, on='particle_id', how='left')


def _top_part_transitions(merged: pd.DataFrame, *, top_n: int = 12) -> List[Dict[str, Any]]:
    transitions = (
        merged.groupby(['nearest_part_id_reference', 'nearest_part_id_candidate'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )
    return [
        {
            'reference_part_id': int(row['nearest_part_id_reference']),
            'candidate_part_id': int(row['nearest_part_id_candidate']),
            'count': int(row['count']),
        }
        for _, row in transitions.head(int(top_n)).iterrows()
    ]


def geometry_feature_delta_summary(
    candidate_final: pd.DataFrame,
    reference_final: pd.DataFrame,
    runtime,
    *,
    top_n: int = 12,
) -> Dict[str, Any]:
    candidate_features = _geometry_feature_frame(runtime, candidate_final)
    reference_features = _geometry_feature_frame(runtime, reference_final)
    merged = reference_features.merge(
        candidate_features,
        on='particle_id',
        how='inner',
        suffixes=('_reference', '_candidate'),
    )
    if merged.empty:
        return {'compared_particles': 0}

    dim = int(sum(1 for name in ('x', 'y', 'z') if f'{name}_reference' in merged.columns and f'{name}_candidate' in merged.columns))
    if dim > 0:
        ref_pos = merged[[f'{name}_reference' for name in ['x', 'y', 'z'][:dim]]].to_numpy(dtype=np.float64)
        cand_pos = merged[[f'{name}_candidate' for name in ['x', 'y', 'z'][:dim]]].to_numpy(dtype=np.float64)
        position_error = np.linalg.norm(cand_pos - ref_pos, axis=1)
    else:
        position_error = np.full(len(merged), np.nan, dtype=np.float64)

    sdf_ref = merged['sdf_m_reference'].to_numpy(dtype=np.float64)
    sdf_cand = merged['sdf_m_candidate'].to_numpy(dtype=np.float64)
    dist_ref = merged['nearest_boundary_distance_m_reference'].to_numpy(dtype=np.float64)
    dist_cand = merged['nearest_boundary_distance_m_candidate'].to_numpy(dtype=np.float64)
    speed_ref = merged['speed_mps_reference'].to_numpy(dtype=np.float64)
    speed_cand = merged['speed_mps_candidate'].to_numpy(dtype=np.float64)
    threshold = _feature_near_boundary_threshold_m(runtime)
    ref_near = np.isfinite(sdf_ref) & (np.abs(sdf_ref) <= float(threshold)) if threshold > 0.0 else np.zeros(len(merged), dtype=bool)
    cand_near = np.isfinite(sdf_cand) & (np.abs(sdf_cand) <= float(threshold)) if threshold > 0.0 else np.zeros(len(merged), dtype=bool)
    class_matches = (
        merged['particle_class_reference'].astype(str).to_numpy()
        == merged['particle_class_candidate'].astype(str).to_numpy()
    )
    mismatch_mask = ~class_matches

    return {
        'compared_particles': int(len(merged)),
        'near_boundary_threshold_m': float(threshold),
        'position_error_m': _finite_summary(position_error),
        'sdf_error_m': _finite_summary(np.abs(sdf_cand - sdf_ref)),
        'abs_sdf_error_m': _finite_summary(np.abs(np.abs(sdf_cand) - np.abs(sdf_ref))),
        'nearest_boundary_distance_error_m': _finite_summary(np.abs(dist_cand - dist_ref)),
        'speed_error_mps': _finite_summary(np.abs(speed_cand - speed_ref)),
        'outside_geometry_count_reference': int(np.count_nonzero(np.isfinite(sdf_ref) & (sdf_ref > 0.0))),
        'outside_geometry_count_candidate': int(np.count_nonzero(np.isfinite(sdf_cand) & (sdf_cand > 0.0))),
        'outside_geometry_count_delta': int(
            np.count_nonzero(np.isfinite(sdf_cand) & (sdf_cand > 0.0))
            - np.count_nonzero(np.isfinite(sdf_ref) & (sdf_ref > 0.0))
        ),
        'near_boundary_count_reference': int(np.count_nonzero(ref_near)),
        'near_boundary_count_candidate': int(np.count_nonzero(cand_near)),
        'near_boundary_count_delta': int(np.count_nonzero(cand_near) - np.count_nonzero(ref_near)),
        'nearest_part_transition_summary': _top_part_transitions(merged, top_n=top_n),
        'mismatched_state_feature_summary': {
            'count': int(np.count_nonzero(mismatch_mask)),
            'position_error_m': _finite_summary(position_error[mismatch_mask]),
            'candidate_sdf_m': _finite_summary(sdf_cand[mismatch_mask]),
            'reference_sdf_m': _finite_summary(sdf_ref[mismatch_mask]),
            'candidate_nearest_boundary_distance_m': _finite_summary(dist_cand[mismatch_mask]),
            'reference_nearest_boundary_distance_m': _finite_summary(dist_ref[mismatch_mask]),
        },
    }


def _summarize_run(output_dir: Path, runtime_s: float) -> Dict[str, Any]:
    report = _load_json(output_dir / 'solver_report.json')
    diag_path = output_dir / 'collision_diagnostics.json'
    diag = _load_json(diag_path) if diag_path.exists() else {}
    final_df = _load_final_particles(output_dir)
    timing = report.get('timing_s', {})
    memory = report.get('memory_estimate_bytes', {})
    numerical_boundary_stopped_count = int(
        report.get(
            'numerical_boundary_stopped_count',
            int(final_df.get('numerical_boundary_stopped', pd.Series(dtype=int)).sum()),
        )
    )
    unresolved_crossing_count = int(diag.get('unresolved_crossing_count', report.get('unresolved_crossing_count', 0)))
    max_hits_reached_count = int(diag.get('max_hits_reached_count', report.get('max_hits_reached_count', 0)))
    nearest_projection_fallback_count = int(
        diag.get('nearest_projection_fallback_count', report.get('nearest_projection_fallback_count', 0))
    )
    bisection_fallback_count = int(diag.get('bisection_fallback_count', report.get('bisection_fallback_count', 0)))
    boundary_event_failure_count = int(
        report.get(
            'boundary_event_failure_count',
            numerical_boundary_stopped_count
            + unresolved_crossing_count
            + max_hits_reached_count
            + nearest_projection_fallback_count,
        )
    )
    return {
        'runtime_s': float(runtime_s),
        'solver_core_s': float(timing.get('solver_core_s', 0.0)) if isinstance(timing, Mapping) else 0.0,
        'solver_step_loop_s': float(timing.get('step_loop_s', 0.0)) if isinstance(timing, Mapping) else 0.0,
        'estimated_numpy_bytes': int(memory.get('estimated_numpy_bytes', 0)) if isinstance(memory, Mapping) else 0,
        'positions_array_bytes': int(memory.get('positions_array_bytes', 0)) if isinstance(memory, Mapping) else 0,
        'particle_count': int(report.get('particle_count', len(final_df))),
        'released_count': int(report.get('released_count', int(final_df.get('released', pd.Series(dtype=int)).sum()))),
        'stuck_count': int(report.get('stuck_count', int(final_df.get('stuck', pd.Series(dtype=int)).sum()))),
        'absorbed_count': int(report.get('absorbed_count', int(final_df.get('absorbed', pd.Series(dtype=int)).sum()))),
        'escaped_count': int(report.get('escaped_count', int(final_df.get('escaped', pd.Series(dtype=int)).sum()))),
        'invalid_mask_stopped_count': int(
            report.get('invalid_mask_stopped_count', int(final_df.get('invalid_mask_stopped', pd.Series(dtype=int)).sum()))
        ),
        'invalid_mask_stop_reason_counts': dict(
            diag.get(
                'invalid_mask_stop_reason_counts',
                report.get('invalid_mask_stop_reason_counts', {}),
            )
        ),
        'invalid_stop_geometry_summary': dict(
            diag.get(
                'invalid_stop_geometry_summary',
                report.get('invalid_stop_geometry_summary', {}),
            )
        ),
        'state_geometry_summary': dict(
            diag.get(
                'state_geometry_summary',
                report.get('state_geometry_summary', {}),
            )
        ),
        'source_initial_geometry_summary': dict(
            diag.get(
                'source_initial_geometry_summary',
                report.get('source_initial_geometry_summary', {}),
            )
        ),
        'valid_mask_mixed_stencil_count': int(diag.get('valid_mask_mixed_stencil_count', report.get('valid_mask_mixed_stencil_count', 0))),
        'valid_mask_mixed_stencil_particle_count': int(
            diag.get('valid_mask_mixed_stencil_particle_count', report.get('valid_mask_mixed_stencil_particle_count', 0))
        ),
        'valid_mask_hard_invalid_count': int(diag.get('valid_mask_hard_invalid_count', report.get('valid_mask_hard_invalid_count', 0))),
        'valid_mask_hard_invalid_particle_count': int(
            diag.get('valid_mask_hard_invalid_particle_count', report.get('valid_mask_hard_invalid_particle_count', 0))
        ),
        'valid_mask_policy': str(report.get('valid_mask_policy', '')),
        'diagnostic_hard_invalid_failed': bool(
            str(report.get('valid_mask_policy', '')).strip().lower() == 'diagnostic'
            and int(diag.get('valid_mask_hard_invalid_count', report.get('valid_mask_hard_invalid_count', 0))) > 0
        ),
        'field_backend_kind': str(report.get('field_backend_kind', '')),
        'field_has_support_phi': int(report.get('field_has_support_phi', 0)),
        'field_support_phi_kind': str(report.get('field_support_phi_kind', '')),
        'integrator': str(report.get('integrator', '')),
        'numerical_boundary_stopped_count': int(numerical_boundary_stopped_count),
        'unresolved_crossing_count': int(unresolved_crossing_count),
        'max_hits_reached_count': int(max_hits_reached_count),
        'bisection_fallback_count': int(bisection_fallback_count),
        'nearest_projection_fallback_count': int(nearest_projection_fallback_count),
        'boundary_event_failure_count': int(boundary_event_failure_count),
        'boundary_event_contract_passed': int(boundary_event_failure_count == 0),
        'primary_hit_count': int(diag.get('primary_hit_count', 0)),
        'output_dir': str(output_dir),
        '_final_df': final_df,
    }


def _pair_delta(base_run: Mapping[str, Any], candidate_run: Mapping[str, Any]) -> Dict[str, Any]:
    base_runtime = float(base_run.get('runtime_s', 0.0))
    candidate_runtime = float(candidate_run.get('runtime_s', 0.0))
    runtime_increase_ratio = 0.0 if base_runtime <= 0.0 else (candidate_runtime - base_runtime) / base_runtime
    return {
        'base_run': str(base_run.get('run', '')),
        'candidate_run': str(candidate_run.get('run', '')),
        'runtime_increase_ratio': float(runtime_increase_ratio),
        'class_match_ratio_delta': float(candidate_run.get('class_match_ratio_vs_reference', 0.0) - base_run.get('class_match_ratio_vs_reference', 0.0)),
        'unresolved_crossing_count_delta': int(candidate_run.get('unresolved_crossing_count', 0) - base_run.get('unresolved_crossing_count', 0)),
        'max_hits_reached_count_delta': int(candidate_run.get('max_hits_reached_count', 0) - base_run.get('max_hits_reached_count', 0)),
        'nearest_projection_fallback_count_delta': int(
            candidate_run.get('nearest_projection_fallback_count', 0) - base_run.get('nearest_projection_fallback_count', 0)
        ),
        'boundary_event_failure_count_delta': int(
            candidate_run.get('boundary_event_failure_count', 0) - base_run.get('boundary_event_failure_count', 0)
        ),
        'stuck_count_delta': int(candidate_run.get('stuck_count', 0) - base_run.get('stuck_count', 0)),
        'invalid_mask_stopped_count_delta': int(
            candidate_run.get('invalid_mask_stopped_count', 0) - base_run.get('invalid_mask_stopped_count', 0)
        ),
        'valid_mask_mixed_stencil_count_delta': int(
            candidate_run.get('valid_mask_mixed_stencil_count', 0) - base_run.get('valid_mask_mixed_stencil_count', 0)
        ),
        'valid_mask_hard_invalid_count_delta': int(
            candidate_run.get('valid_mask_hard_invalid_count', 0) - base_run.get('valid_mask_hard_invalid_count', 0)
        ),
    }


def _parse_named_run(value: str) -> Tuple[str, Path]:
    name, sep, raw_path = str(value).partition('=')
    if not sep or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f'Expected NAME=path, got: {value}')
    return str(name).strip(), Path(raw_path.strip())


def _strip_internal_fields(run_summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in run_summary.items() if not str(k).startswith('_')}


def main(argv: Iterable[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Run a reference config plus named candidate configs, then summarize runtime, unresolved crossings, and class-match ratio.'
    )
    parser.add_argument('--reference-config', type=Path, required=True, help='Reference run_config.yaml')
    parser.add_argument(
        '--run',
        action='append',
        type=_parse_named_run,
        required=True,
        metavar='NAME=PATH',
        help='Named candidate config, e.g. --run etd2_base=examples/.../run_config_prod_etd2_base.yaml',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('demo_output/reference_compare'),
        help='Root directory where a timestamped comparison folder will be created',
    )
    parser.add_argument(
        '--summary-json',
        type=Path,
        default=None,
        help='Optional explicit JSON output path. Defaults to <output-root>/<timestamp>/comparison_summary.json',
    )
    parser.add_argument(
        '--per-run-timeout-s',
        type=float,
        default=0.0,
        help='Abort each run after this many seconds. Use 0 to disable the timeout.',
    )
    parser.add_argument(
        '--override-t-end',
        type=float,
        default=None,
        help='Write generated run configs under the comparison output and override solver.t_end for lightweight gates.',
    )
    parser.add_argument(
        '--artifact-mode',
        choices=('full', 'minimal'),
        default=None,
        help='Write generated run configs under the comparison output and set output.artifact_mode.',
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reference_config = _resolve_path(repo_root, args.reference_config)
    run_specs = [(name, _resolve_path(repo_root, path)) for name, path in args.run]
    output_root = _resolve_path(repo_root, args.output_root)
    comparison_dir = output_root / f'compare_{timestamp}'
    summary_path = (
        _resolve_path(repo_root, args.summary_json)
        if args.summary_json is not None
        else (comparison_dir / 'comparison_summary.json')
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if not reference_config.exists():
        raise FileNotFoundError(f'reference config not found: {reference_config}')
    for run_name, run_config in run_specs:
        if not run_config.exists():
            raise FileNotFoundError(f'run config not found for {run_name}: {run_config}')

    if args.override_t_end is not None or args.artifact_mode is not None:
        config_dir = comparison_dir / 'configs'
        reference_config = _write_config_variant(
            source_config=reference_config,
            output_config=config_dir / f'{reference_config.stem}_reference.yaml',
            override_t_end=args.override_t_end,
            artifact_mode=args.artifact_mode,
        )
        run_specs = [
            (
                run_name,
                _write_config_variant(
                    source_config=run_config,
                    output_config=config_dir / f'{run_name}.yaml',
                    override_t_end=args.override_t_end,
                    artifact_mode=args.artifact_mode,
                ),
            )
            for run_name, run_config in run_specs
        ]

    reference_output_dir = comparison_dir / 'reference'
    reference_runtime_s = _run_case(repo_root, reference_config, reference_output_dir, timeout_s=float(args.per_run_timeout_s))
    reference_summary = _summarize_run(reference_output_dir, runtime_s=reference_runtime_s)
    reference_final = reference_summary['_final_df']
    feature_runtime = build_runtime_from_config(_load_yaml_mapping(reference_config), reference_config.parent)

    runs: List[Dict[str, Any]] = []
    for run_name, run_config in run_specs:
        run_output_dir = comparison_dir / run_name
        run_runtime_s = _run_case(repo_root, run_config, run_output_dir, timeout_s=float(args.per_run_timeout_s))
        run_summary = _summarize_run(run_output_dir, runtime_s=run_runtime_s)
        match_ratio, compared_particles = class_match_ratio(run_summary['_final_df'], reference_final)
        transition_summary = class_transition_summary(run_summary['_final_df'], reference_final)
        feature_summary = geometry_feature_delta_summary(run_summary['_final_df'], reference_final, feature_runtime)
        run_summary.update(
            {
                'run': run_name,
                'config': str(run_config),
                'class_match_ratio_vs_reference': float(match_ratio),
                'compared_particles_vs_reference': int(compared_particles),
                'class_mismatch_count_vs_reference': int(transition_summary['mismatch_count']),
                'class_transition_summary_vs_reference': transition_summary,
                'geometry_feature_delta_vs_reference': feature_summary,
            }
        )
        runs.append(_strip_internal_fields(run_summary))

    summary: Dict[str, Any] = {
        'timestamp': timestamp,
        'overrides': {
            't_end': None if args.override_t_end is None else float(args.override_t_end),
            'artifact_mode': args.artifact_mode,
            'per_run_timeout_s': float(args.per_run_timeout_s),
        },
        'reference': {
            'run': 'reference',
            'config': str(reference_config),
            **_strip_internal_fields(reference_summary),
        },
        'runs': runs,
    }
    if len(runs) == 2:
        summary['pair_delta'] = _pair_delta(runs[0], runs[1])
    diagnostic_failures = [
        item.get('run', '')
        for item in [summary['reference'], *runs]
        if bool(item.get('diagnostic_hard_invalid_failed', False))
    ]
    if diagnostic_failures:
        summary['diagnostic_hard_invalid_failures'] = diagnostic_failures
    boundary_event_failures = [
        item.get('run', '')
        for item in [summary['reference'], *runs]
        if int(item.get('boundary_event_failure_count', 0)) > 0
    ]
    if boundary_event_failures:
        summary['boundary_event_failures'] = boundary_event_failures

    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 1 if diagnostic_failures or boundary_event_failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
