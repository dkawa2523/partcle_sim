from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.input_contract import write_input_contract_report
from .core.provider_contract import write_provider_contract_report
from .core.source_materials import write_source_summary
from .io.runtime_builder import build_prepared_runtime_from_yaml, prepared_runtime_summary
from .solvers.solver_entrypoints import run_solver_for_dim


def _spatial_dim(prepared) -> int:
    dim = int(prepared.runtime.spatial_dim)
    if dim not in {2, 3}:
        raise ValueError('run.spatial_dim must be 2 or 3')
    return dim


def _default_output_dir(config: Path, spatial_dim: int, *, reports_only: bool) -> Path:
    if reports_only:
        return config.parent / 'prepared_runtime_output'
    return config.parent / f'run_output_{spatial_dim}d'


def _write_prepared_reports(prepared, output_dir: Path) -> tuple[dict, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'prepared_runtime_summary.json'
    summary_path.write_text(json.dumps(prepared_runtime_summary(prepared), indent=2), encoding='utf-8')
    if prepared.source_preprocess is not None:
        write_source_summary(prepared.source_preprocess, output_dir)
    provider_report = write_provider_contract_report(prepared, output_dir)
    input_report = write_input_contract_report(prepared, output_dir)
    return provider_report, input_report


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a prepared runtime and run the particle trajectory solver.')
    parser.add_argument('config', type=Path)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--check-input', action='store_true', help='Write provider/input contract reports and exit before solving.')
    args = parser.parse_args()

    prepared = build_prepared_runtime_from_yaml(args.config)
    dim = _spatial_dim(prepared)

    reports_only = bool(args.prepare_only or args.check_input)
    output_dir = args.output_dir or _default_output_dir(args.config, dim, reports_only=reports_only)
    if reports_only:
        provider_report, input_report = _write_prepared_reports(prepared, output_dir)
        if args.check_input and (
            not bool(provider_report.get('passed', True))
            or not bool(input_report.get('passed', True))
        ):
            return 1
        return 0

    run_solver_for_dim(prepared, output_dir=output_dir, spatial_dim=dim)
    return 0
