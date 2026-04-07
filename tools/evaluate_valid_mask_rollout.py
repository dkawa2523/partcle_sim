from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

try:
    from tools.compare_against_reference import main as compare_against_reference_main
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from compare_against_reference import main as compare_against_reference_main


def _resolve_path(repo_root: Path, path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else (repo_root / path_value).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'config must deserialize to a mapping: {path}')
    return payload


def _absolutize_config_paths(payload: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    out = dict(payload)
    paths_cfg = out.get('paths', {})
    if isinstance(paths_cfg, Mapping):
        paths_out: Dict[str, Any] = dict(paths_cfg)
        for key, value in list(paths_out.items()):
            if value is None or str(value).strip() == '':
                continue
            p = Path(str(value))
            paths_out[key] = str((base_dir / p).resolve() if not p.is_absolute() else p)
        out['paths'] = paths_out
    providers_cfg = out.get('providers', {})
    if isinstance(providers_cfg, Mapping):
        providers_out: Dict[str, Any] = dict(providers_cfg)
        for provider_name in ('geometry', 'field'):
            provider_cfg = providers_out.get(provider_name, {})
            if not isinstance(provider_cfg, Mapping):
                continue
            provider_out: Dict[str, Any] = dict(provider_cfg)
            npz_path = provider_out.get('npz_path')
            if npz_path is None or str(npz_path).strip() == '':
                providers_out[provider_name] = provider_out
                continue
            p = Path(str(npz_path))
            provider_out['npz_path'] = str((base_dir / p).resolve() if not p.is_absolute() else p)
            providers_out[provider_name] = provider_out
        out['providers'] = providers_out
    return out


def _write_policy_variant(base_payload: Mapping[str, Any], *, policy: str, output_path: Path) -> Path:
    payload = dict(base_payload)
    solver_cfg = payload.get('solver', {})
    solver_dict = dict(solver_cfg) if isinstance(solver_cfg, Mapping) else {}
    solver_dict['valid_mask_policy'] = str(policy)
    payload['solver'] = solver_dict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return output_path


def _build_rollout_summary(
    *,
    compare_summary: Mapping[str, Any],
    max_runtime_increase_ratio: float,
    min_class_match_ratio: float | None,
) -> Dict[str, Any]:
    runs = list(compare_summary.get('runs', []))
    diagnostic = next((run for run in runs if str(run.get('run', '')) == 'diagnostic'), None)
    retry_then_stop = next((run for run in runs if str(run.get('run', '')) == 'retry_then_stop'), None)
    if diagnostic is None or retry_then_stop is None:
        raise ValueError('compare summary must contain diagnostic and retry_then_stop runs')

    diagnostic_runtime = float(diagnostic.get('runtime_s', 0.0))
    retry_runtime = float(retry_then_stop.get('runtime_s', 0.0))
    runtime_increase_ratio = 0.0 if diagnostic_runtime <= 0.0 else (retry_runtime - diagnostic_runtime) / diagnostic_runtime
    class_match_ratio = retry_then_stop.get('class_match_ratio_vs_reference')
    class_match_ok = False if min_class_match_ratio is None else class_match_ratio is not None
    if min_class_match_ratio is not None and class_match_ratio is not None:
        class_match_ok = float(class_match_ratio) >= float(min_class_match_ratio)
    unresolved_ok = int(retry_then_stop.get('unresolved_crossing_count', 0)) <= int(diagnostic.get('unresolved_crossing_count', 0))
    runtime_ok = float(runtime_increase_ratio) <= float(max_runtime_increase_ratio)

    checks = {
        'runtime_increase_ok': bool(runtime_ok),
        'unresolved_crossing_ok': bool(unresolved_ok),
        'class_match_ratio_ok': bool(class_match_ok),
    }
    recommendation = 'candidate_ready_for_default' if all(checks.values()) else 'keep_opt_in'
    return {
        'diagnostic': diagnostic,
        'retry_then_stop': retry_then_stop,
        'metrics': {
            'runtime_increase_ratio': float(runtime_increase_ratio),
            'class_match_ratio_vs_reference': float(class_match_ratio) if class_match_ratio is not None else None,
            'class_match_ratio_delta_vs_diagnostic': float(
                float(retry_then_stop.get('class_match_ratio_vs_reference', 0.0))
                - float(diagnostic.get('class_match_ratio_vs_reference', 0.0))
            ),
            'valid_mask_mixed_stencil_count_delta': int(
                retry_then_stop.get('valid_mask_mixed_stencil_count', 0) - diagnostic.get('valid_mask_mixed_stencil_count', 0)
            ),
            'valid_mask_hard_invalid_count_delta': int(
                retry_then_stop.get('valid_mask_hard_invalid_count', 0) - diagnostic.get('valid_mask_hard_invalid_count', 0)
            ),
            'extension_band_sample_count_delta': int(
                retry_then_stop.get('extension_band_sample_count', 0) - diagnostic.get('extension_band_sample_count', 0)
            ),
            'invalid_mask_stopped_count_delta': int(
                retry_then_stop.get('invalid_mask_stopped_count', 0) - diagnostic.get('invalid_mask_stopped_count', 0)
            ),
            'unresolved_crossing_count_delta': int(
                retry_then_stop.get('unresolved_crossing_count', 0) - diagnostic.get('unresolved_crossing_count', 0)
            ),
        },
        'checks': checks,
        'rollout_recommendation': recommendation,
    }


def main(argv: Iterable[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Create diagnostic and retry_then_stop config variants from one base config, compare both against a reference, and summarize rollout readiness.'
    )
    parser.add_argument('--config', type=Path, required=True, help='Base candidate run_config.yaml')
    parser.add_argument('--reference-config', type=Path, required=True, help='Reference run_config.yaml')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('demo_output/valid_mask_rollout'),
        help='Root directory where a timestamped rollout folder will be created',
    )
    parser.add_argument(
        '--summary-json',
        type=Path,
        default=None,
        help='Optional explicit output path for the rollout summary JSON',
    )
    parser.add_argument(
        '--max-runtime-increase-ratio',
        type=float,
        default=0.10,
        help='Allowed runtime increase ratio for retry_then_stop versus diagnostic',
    )
    parser.add_argument(
        '--min-class-match-ratio',
        type=float,
        default=None,
        help='Optional minimum class_match_ratio_vs_reference required for retry_then_stop',
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_config = _resolve_path(repo_root, args.config)
    reference_config = _resolve_path(repo_root, args.reference_config)
    output_root = _resolve_path(repo_root, args.output_root)
    if not base_config.exists():
        raise FileNotFoundError(f'config not found: {base_config}')
    if not reference_config.exists():
        raise FileNotFoundError(f'reference config not found: {reference_config}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rollout_dir = output_root / f'valid_mask_rollout_{timestamp}'
    variants_dir = rollout_dir / 'configs'
    compare_dir = rollout_dir / 'compare'
    compare_summary_path = rollout_dir / 'compare_summary.json'
    summary_path = _resolve_path(repo_root, args.summary_json) if args.summary_json is not None else (rollout_dir / 'rollout_summary.json')

    base_payload = _absolutize_config_paths(_load_yaml(base_config), base_dir=base_config.parent)
    diagnostic_config = _write_policy_variant(base_payload, policy='diagnostic', output_path=variants_dir / 'run_config_diagnostic.yaml')
    retry_config = _write_policy_variant(
        base_payload,
        policy='retry_then_stop',
        output_path=variants_dir / 'run_config_retry_then_stop.yaml',
    )

    compare_rc = compare_against_reference_main(
        [
            '--reference-config',
            str(reference_config),
            '--run',
            f'diagnostic={diagnostic_config}',
            '--run',
            f'retry_then_stop={retry_config}',
            '--output-root',
            str(compare_dir),
            '--summary-json',
            str(compare_summary_path),
        ]
    )
    if int(compare_rc) != 0:
        return int(compare_rc)

    compare_summary = json.loads(compare_summary_path.read_text(encoding='utf-8'))
    rollout_summary = {
        'timestamp': timestamp,
        'base_config': str(base_config),
        'reference_config': str(reference_config),
        'generated_configs': {
            'diagnostic': str(diagnostic_config),
            'retry_then_stop': str(retry_config),
        },
        'thresholds': {
            'max_runtime_increase_ratio': float(args.max_runtime_increase_ratio),
            'min_class_match_ratio': (
                None if args.min_class_match_ratio is None else float(args.min_class_match_ratio)
            ),
        },
        'comparison_summary_path': str(compare_summary_path),
        **_build_rollout_summary(
            compare_summary=compare_summary,
            max_runtime_increase_ratio=float(args.max_runtime_increase_ratio),
            min_class_match_ratio=(None if args.min_class_match_ratio is None else float(args.min_class_match_ratio)),
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rollout_summary, indent=2), encoding='utf-8')
    print(json.dumps(rollout_summary, indent=2))
    return 0 if all(rollout_summary['checks'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
