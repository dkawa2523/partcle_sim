from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple


def _run_case(repo_root: Path, config_path: Path, output_dir: Path) -> float:
    cmd = [sys.executable, str(repo_root / 'run_from_yaml.py'), str(config_path), '--output-dir', str(output_dir)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(repo_root), check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f'run_from_yaml failed for {config_path} with exit code {result.returncode}')
    return float(elapsed)


def _load_collision_diag(output_dir: Path) -> Dict[str, Any]:
    diag_path = output_dir / 'collision_diagnostics.json'
    if not diag_path.exists():
        raise FileNotFoundError(f'collision diagnostics not found: {diag_path}')
    return json.loads(diag_path.read_text(encoding='utf-8'))


def _max_hits_reduction_ratio(base_hits: int, candidate_hits: int) -> Tuple[float, bool]:
    if int(base_hits) <= 0:
        return (0.0, int(candidate_hits) == 0)
    ratio = (float(base_hits) - float(candidate_hits)) / float(base_hits)
    return (ratio, True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(
        description='Run base/candidate configs and verify max-hits reduction, unresolved crossings, and runtime increase.'
    )
    ap.add_argument('--base-config', type=Path, required=True, help='Base run_config.yaml')
    ap.add_argument('--candidate-config', type=Path, required=True, help='Candidate run_config.yaml')
    ap.add_argument(
        '--base-output-dir',
        type=Path,
        default=Path('demo_output/stability_check/base'),
        help='Output directory for base run (default: demo_output/stability_check/base)',
    )
    ap.add_argument(
        '--candidate-output-dir',
        type=Path,
        default=Path('demo_output/stability_check/candidate'),
        help='Output directory for candidate run (default: demo_output/stability_check/candidate)',
    )
    ap.add_argument(
        '--max-runtime-increase-ratio',
        type=float,
        default=0.20,
        help='Allowed candidate runtime increase ratio, e.g. 0.20 = +20%%',
    )
    ap.add_argument(
        '--min-max-hits-reduction-ratio',
        type=float,
        default=0.30,
        help='Required minimum reduction ratio for max_hits_reached_count, e.g. 0.30 = 30%%',
    )
    ap.add_argument(
        '--summary-json',
        type=Path,
        default=None,
        help='Optional path to write comparison summary as JSON',
    )
    args = ap.parse_args()

    base_config = args.base_config if args.base_config.is_absolute() else (repo_root / args.base_config)
    candidate_config = args.candidate_config if args.candidate_config.is_absolute() else (repo_root / args.candidate_config)
    base_output_dir = args.base_output_dir if args.base_output_dir.is_absolute() else (repo_root / args.base_output_dir)
    candidate_output_dir = (
        args.candidate_output_dir if args.candidate_output_dir.is_absolute() else (repo_root / args.candidate_output_dir)
    )

    if not base_config.exists():
        raise FileNotFoundError(f'base config not found: {base_config}')
    if not candidate_config.exists():
        raise FileNotFoundError(f'candidate config not found: {candidate_config}')

    base_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_output_dir.mkdir(parents=True, exist_ok=True)

    base_runtime_s = _run_case(repo_root, base_config, base_output_dir)
    candidate_runtime_s = _run_case(repo_root, candidate_config, candidate_output_dir)

    base_diag = _load_collision_diag(base_output_dir)
    candidate_diag = _load_collision_diag(candidate_output_dir)

    base_hits = int(base_diag.get('max_hits_reached_count', 0))
    candidate_hits = int(candidate_diag.get('max_hits_reached_count', 0))
    reduction_ratio, reduction_ratio_comparable = _max_hits_reduction_ratio(base_hits, candidate_hits)

    base_unresolved = int(base_diag.get('unresolved_crossing_count', 0))
    candidate_unresolved = int(candidate_diag.get('unresolved_crossing_count', 0))

    if float(base_runtime_s) <= 0.0:
        runtime_increase_ratio = 0.0 if float(candidate_runtime_s) <= 0.0 else float('inf')
    else:
        runtime_increase_ratio = (float(candidate_runtime_s) - float(base_runtime_s)) / float(base_runtime_s)

    reduction_ok = bool(reduction_ratio_comparable and reduction_ratio >= float(args.min_max_hits_reduction_ratio))
    if not reduction_ratio_comparable:
        reduction_ok = int(candidate_hits) == 0
    unresolved_ok = int(candidate_unresolved) == 0
    runtime_ok = float(runtime_increase_ratio) <= float(args.max_runtime_increase_ratio)

    summary = {
        'base_config': str(base_config),
        'candidate_config': str(candidate_config),
        'base_output_dir': str(base_output_dir),
        'candidate_output_dir': str(candidate_output_dir),
        'thresholds': {
            'max_runtime_increase_ratio': float(args.max_runtime_increase_ratio),
            'min_max_hits_reduction_ratio': float(args.min_max_hits_reduction_ratio),
        },
        'base': {
            'runtime_s': float(base_runtime_s),
            'max_hits_reached_count': int(base_hits),
            'unresolved_crossing_count': int(base_unresolved),
        },
        'candidate': {
            'runtime_s': float(candidate_runtime_s),
            'max_hits_reached_count': int(candidate_hits),
            'unresolved_crossing_count': int(candidate_unresolved),
        },
        'metrics': {
            'max_hits_reduction_ratio': float(reduction_ratio),
            'runtime_increase_ratio': float(runtime_increase_ratio),
            'max_hits_reduction_ratio_comparable': bool(reduction_ratio_comparable),
        },
        'checks': {
            'max_hits_reduction_ok': bool(reduction_ok),
            'unresolved_crossing_ok': bool(unresolved_ok),
            'runtime_increase_ok': bool(runtime_ok),
        },
    }

    if args.summary_json is not None:
        summary_path = args.summary_json if args.summary_json.is_absolute() else (repo_root / args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary, indent=2))
    return 0 if (reduction_ok and unresolved_ok and runtime_ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
