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

try:
    from tools.state_contract import particle_class_frame
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from state_contract import particle_class_frame


def _resolve_path(repo_root: Path, path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else (repo_root / path_value).resolve()


def _run_case(repo_root: Path, config_path: Path, output_dir: Path) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(repo_root / 'run_from_yaml.py'), str(config_path), '--output-dir', str(output_dir)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(repo_root), check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f'run_from_yaml failed for {config_path} with exit code {result.returncode}')
    return float(elapsed)


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


def _summarize_run(output_dir: Path, runtime_s: float) -> Dict[str, Any]:
    report = _load_json(output_dir / 'solver_report.json')
    diag = _load_json(output_dir / 'collision_diagnostics.json')
    final_df = _load_final_particles(output_dir)
    return {
        'runtime_s': float(runtime_s),
        'particle_count': int(report.get('particle_count', len(final_df))),
        'released_count': int(report.get('released_count', int(final_df.get('released', pd.Series(dtype=int)).sum()))),
        'stuck_count': int(report.get('stuck_count', int(final_df.get('stuck', pd.Series(dtype=int)).sum()))),
        'absorbed_count': int(report.get('absorbed_count', int(final_df.get('absorbed', pd.Series(dtype=int)).sum()))),
        'escaped_count': int(report.get('escaped_count', int(final_df.get('escaped', pd.Series(dtype=int)).sum()))),
        'invalid_mask_stopped_count': int(
            report.get('invalid_mask_stopped_count', int(final_df.get('invalid_mask_stopped', pd.Series(dtype=int)).sum()))
        ),
        'valid_mask_mixed_stencil_count': int(diag.get('valid_mask_mixed_stencil_count', report.get('valid_mask_mixed_stencil_count', 0))),
        'valid_mask_mixed_stencil_particle_count': int(
            diag.get('valid_mask_mixed_stencil_particle_count', report.get('valid_mask_mixed_stencil_particle_count', 0))
        ),
        'valid_mask_hard_invalid_count': int(diag.get('valid_mask_hard_invalid_count', report.get('valid_mask_hard_invalid_count', 0))),
        'valid_mask_hard_invalid_particle_count': int(
            diag.get('valid_mask_hard_invalid_particle_count', report.get('valid_mask_hard_invalid_particle_count', 0))
        ),
        'extension_band_sample_count': int(diag.get('extension_band_sample_count', report.get('extension_band_sample_count', 0))),
        'extension_band_sample_particle_count': int(
            diag.get('extension_band_sample_particle_count', report.get('extension_band_sample_particle_count', 0))
        ),
        'field_backend_kind': str(report.get('field_backend_kind', '')),
        'field_regularization_band_distance_m': float(report.get('field_regularization_band_distance_m', 0.0)),
        'field_regularization_added_node_count': int(report.get('field_regularization_added_node_count', 0)),
        'field_regularization_mode': str(report.get('field_regularization_mode', '')),
        'field_regularization_probe_success_count': int(report.get('field_regularization_probe_success_count', 0)),
        'field_regularization_probe_fallback_count': int(report.get('field_regularization_probe_fallback_count', 0)),
        'integrator': str(report.get('integrator', '')),
        'unresolved_crossing_count': int(diag.get('unresolved_crossing_count', 0)),
        'max_hits_reached_count': int(diag.get('max_hits_reached_count', 0)),
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
        'extension_band_sample_count_delta': int(
            candidate_run.get('extension_band_sample_count', 0) - base_run.get('extension_band_sample_count', 0)
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

    reference_output_dir = comparison_dir / 'reference'
    reference_runtime_s = _run_case(repo_root, reference_config, reference_output_dir)
    reference_summary = _summarize_run(reference_output_dir, runtime_s=reference_runtime_s)
    reference_final = reference_summary['_final_df']

    runs: List[Dict[str, Any]] = []
    for run_name, run_config in run_specs:
        run_output_dir = comparison_dir / run_name
        run_runtime_s = _run_case(repo_root, run_config, run_output_dir)
        run_summary = _summarize_run(run_output_dir, runtime_s=run_runtime_s)
        match_ratio, compared_particles = class_match_ratio(run_summary['_final_df'], reference_final)
        run_summary.update(
            {
                'run': run_name,
                'config': str(run_config),
                'class_match_ratio_vs_reference': float(match_ratio),
                'compared_particles_vs_reference': int(compared_particles),
            }
        )
        runs.append(_strip_internal_fields(run_summary))

    summary: Dict[str, Any] = {
        'timestamp': timestamp,
        'reference': {
            'run': 'reference',
            'config': str(reference_config),
            **_strip_internal_fields(reference_summary),
        },
        'runs': runs,
    }
    if len(runs) == 2:
        summary['pair_delta'] = _pair_delta(runs[0], runs[1])

    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
