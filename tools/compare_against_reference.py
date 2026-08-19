from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tools import _reference_compare_inputs as comparison_inputs
    from tools import _reference_compare_metrics as comparison_metrics
    from tools import _reference_compare_runs as comparison_runs
    from tools._reference_compare_metrics import (
        class_match_ratio,
        class_transition_summary,
        geometry_feature_delta_summary,
    )
else:
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    comparison_inputs = importlib.import_module("tools._reference_compare_inputs")
    comparison_metrics = importlib.import_module("tools._reference_compare_metrics")
    comparison_runs = importlib.import_module("tools._reference_compare_runs")
    class_match_ratio = comparison_metrics.class_match_ratio
    class_transition_summary = comparison_metrics.class_transition_summary
    geometry_feature_delta_summary = comparison_metrics.geometry_feature_delta_summary

__all__ = [
    "class_match_ratio",
    "class_transition_summary",
    "geometry_feature_delta_summary",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="particle-tracer compare reference",
        description=(
            "Run a reference config plus named candidate configs, then summarize "
            "runtime, unresolved crossings, and class-match ratio."
        ),
    )
    parser.add_argument(
        "--reference-config", type=Path, required=True, help="Reference run_config.yaml"
    )
    parser.add_argument(
        "--run",
        action="append",
        type=comparison_inputs.parse_named_run,
        required=True,
        metavar="NAME=PATH",
        help=(
            "Named candidate config, e.g. "
            "--run etd2_base=examples/.../run_config_prod_etd2_base.yaml"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("demo_output/reference_compare"),
        help="Root directory where a timestamped comparison folder will be created",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help=(
            "Optional additional atomic copy; the canonical summary is always "
            "stored in the comparison folder."
        ),
    )
    parser.add_argument(
        "--override-t-end",
        type=float,
        default=None,
        help=(
            "Write generated run configs under the comparison output and override "
            "solver.t_end for lightweight gates."
        ),
    )
    parser.add_argument(
        "--artifact-mode",
        choices=("standard", "debug"),
        default=None,
        help=(
            "Write generated run configs under the comparison output and set "
            "output.mode."
        ),
    )
    parser.add_argument(
        "--reference-scope",
        choices=("sampled", "full", "unspecified"),
        default="unspecified",
        help="Record whether the supplied reference is sampled, full, or unspecified.",
    )
    return parser


def _execute_comparison(
    args: argparse.Namespace,
    *,
    timestamp: str,
    execution_reference_config: Path,
    reported_reference_config: Path,
    execution_runs: list[tuple[str, Path, Path]],
    staging_dir: Path,
    comparison_dir: Path,
) -> tuple[dict[str, Any], int]:
    reference, reference_final, feature_runtime = comparison_runs.run_reference(
        execution_reference_config,
        reported_config=reported_reference_config,
        staging_dir=staging_dir,
        comparison_dir=comparison_dir,
    )
    runs = comparison_runs.run_candidates(
        execution_runs,
        staging_dir=staging_dir,
        comparison_dir=comparison_dir,
        reference_final=reference_final,
        feature_runtime=feature_runtime,
    )
    return comparison_metrics.comparison_summary(
        args,
        timestamp=timestamp,
        comparison_dir=comparison_dir,
        reference=reference,
        runs=runs,
    )


def main(argv: Iterable[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    reference_config, run_specs, output_root = (
        comparison_inputs.resolve_comparison_inputs(args, repo_root)
    )
    comparison_dir = output_root / f"compare_{timestamp}"
    if comparison_dir.exists():
        raise FileExistsError(f"comparison directory already exists: {comparison_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    with comparison_runs.staging_directory(output_root, comparison_dir) as staging_dir:
        execution_reference, reported_reference, execution_runs = (
            comparison_inputs.execution_configs(
                args,
                reference_config=reference_config,
                run_specs=run_specs,
                staging_dir=staging_dir,
                comparison_dir=comparison_dir,
            )
        )
        summary, exit_code = _execute_comparison(
            args,
            timestamp=timestamp,
            execution_reference_config=execution_reference,
            reported_reference_config=reported_reference,
            execution_runs=execution_runs,
            staging_dir=staging_dir,
            comparison_dir=comparison_dir,
        )
        comparison_runs.write_json_atomic(
            staging_dir / "comparison_summary.json", summary
        )
        staging_dir.rename(comparison_dir)
    comparison_runs.write_summary_alias(
        args.summary_json,
        repo_root=repo_root,
        output_root=output_root,
        comparison_dir=comparison_dir,
        summary=summary,
    )
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
