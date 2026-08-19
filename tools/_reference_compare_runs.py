from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Any

import pandas as pd

from particle_tracer_unified import (
    SimulationCase,
    load_case,
    simulate,
    validate_case,
    write_result,
)
from tools import _reference_compare_inputs as comparison_inputs
from tools import _reference_compare_metrics as comparison_metrics


def _run_case(config_path: Path, output_dir: Path) -> tuple[float, SimulationCase]:
    case = load_case(config_path)
    report = validate_case(case, detail="summary")
    if not report.passed:
        messages = "; ".join(str(issue.message) for issue in report.errors[:8])
        raise ValueError(f"preflight failed for {config_path}: {messages}")
    started_at = time.perf_counter()
    result = simulate(case)
    write_result(result, output_dir)
    return float(time.perf_counter() - started_at), case


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_final_particles(output_dir: Path) -> pd.DataFrame:
    final_csv = output_dir / "final_particles.csv"
    if not final_csv.exists():
        raise FileNotFoundError(final_csv)
    return pd.read_csv(final_csv)


def _summarize_run(output_dir: Path, runtime_s: float) -> dict[str, Any]:
    report = _load_json(output_dir / "run_summary.json")
    diagnostics_path = output_dir / "debug_diagnostics.json"
    diagnostics = _load_json(diagnostics_path) if diagnostics_path.exists() else {}
    final_particles = _load_final_particles(output_dir)
    timing = (
        report.get("timing_s", {})
        if isinstance(report.get("timing_s"), Mapping)
        else {}
    )
    memory = (
        report.get("memory_estimate_bytes", {})
        if isinstance(report.get("memory_estimate_bytes"), Mapping)
        else {}
    )
    states = final_particles["final_state"].fillna("inactive").astype(str)
    nested_counters = report.get("safety_counters")
    counters = nested_counters if isinstance(nested_counters, Mapping) else report
    unresolved_crossings = int(counters.get("unresolved_crossing_count", 0))
    max_hits_reached = int(counters.get("max_hits_reached_count", 0))
    numerical_boundary_stopped = int((states == "numerical_boundary_stopped").sum())
    boundary_event_failures = (
        unresolved_crossings + max_hits_reached + numerical_boundary_stopped
    )
    return {
        "runtime_s": float(runtime_s),
        "solver_core_s": float(timing.get("solver_core_s", 0.0)),
        "solver_step_loop_s": float(timing.get("step_loop_s", 0.0)),
        "estimated_numpy_bytes": int(memory.get("estimated_numpy_bytes", 0)),
        "positions_array_bytes": int(memory.get("positions_array_bytes", 0)),
        "particle_count": int(report.get("particle_count", len(final_particles))),
        "coordinate_system": str(report.get("coordinate_system", "")),
        "released_count": int(
            report.get(
                "released_count",
                int(final_particles.get("released", pd.Series(dtype=int)).sum()),
            )
        ),
        "stuck_count": int((states == "stuck").sum()),
        "absorbed_count": int((states == "absorbed").sum()),
        "escaped_count": int((states == "escaped").sum()),
        "invalid_mask_stopped_count": int((states == "invalid_mask_stopped").sum()),
        "invalid_mask_stop_reason_counts": dict(
            diagnostics.get("invalid_mask_stop_reason_counts", {})
        ),
        "integrator": "etd2",
        "numerical_boundary_stopped_count": numerical_boundary_stopped,
        "unresolved_crossing_count": unresolved_crossings,
        "max_hits_reached_count": max_hits_reached,
        "boundary_event_failure_count": boundary_event_failures,
        "boundary_event_contract_passed": int(boundary_event_failures == 0),
        "output_dir": str(output_dir),
        "_final_df": final_particles,
    }


def _strip_internal_fields(run_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in run_summary.items() if not str(key).startswith("_")
    }


def _relocate_path_text(value: str, source_root: Path, destination_root: Path) -> str:
    source = str(source_root)
    if (
        value == source
        or value.startswith(source + "/")
        or value.startswith(source + "\\")
    ):
        return str(destination_root) + value[len(source) :]
    return value


def relocate_value(value: Any, source_root: Path, destination_root: Path) -> Any:
    if isinstance(value, Mapping):
        return _relocate_mapping(value, source_root, destination_root)
    if isinstance(value, (list, tuple)):
        return [relocate_value(item, source_root, destination_root) for item in value]
    if isinstance(value, str):
        return _relocate_path_text(value, source_root, destination_root)
    return value


def _relocate_mapping(
    value: Mapping[str, Any], source_root: Path, destination_root: Path
) -> dict[str, Any]:
    return {
        str(key): relocate_value(item, source_root, destination_root)
        for key, item in value.items()
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            temporary_path = Path(handle.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _relocate_run_summary(
    output_dir: Path,
    *,
    staging_root: Path,
    comparison_dir: Path,
) -> None:
    path = output_dir / "run_summary.json"
    payload = _load_json(path)
    relocated = _relocate_mapping(payload, staging_root, comparison_dir)
    write_json_atomic(path, relocated)


def run_reference(
    execution_config: Path,
    *,
    reported_config: Path,
    staging_dir: Path,
    comparison_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    output_dir = staging_dir / "reference"
    runtime_s, case = _run_case(execution_config, output_dir)
    _relocate_run_summary(
        output_dir,
        staging_root=staging_dir,
        comparison_dir=comparison_dir,
    )
    summary = _summarize_run(output_dir, runtime_s=runtime_s)
    summary["output_dir"] = str(comparison_dir / "reference")
    reported = {
        "run": "reference",
        "config": str(reported_config),
        **_strip_internal_fields(summary),
    }
    return reported, summary["_final_df"], case.solver_context


def _run_candidate(
    run_name: str,
    execution_config: Path,
    reported_config: Path,
    *,
    staging_dir: Path,
    comparison_dir: Path,
    reference_final: pd.DataFrame,
    feature_runtime: Any,
) -> dict[str, Any]:
    output_dir = staging_dir / run_name
    runtime_s, _ = _run_case(execution_config, output_dir)
    _relocate_run_summary(
        output_dir,
        staging_root=staging_dir,
        comparison_dir=comparison_dir,
    )
    summary = _summarize_run(output_dir, runtime_s=runtime_s)
    summary["output_dir"] = str(comparison_dir / run_name)
    match_ratio, compared_particles = comparison_metrics.class_match_ratio(
        summary["_final_df"], reference_final
    )
    transitions = comparison_metrics.class_transition_summary(
        summary["_final_df"], reference_final
    )
    summary.update(
        {
            "run": run_name,
            "config": str(reported_config),
            "class_match_ratio_vs_reference": float(match_ratio),
            "compared_particles_vs_reference": int(compared_particles),
            "class_mismatch_count_vs_reference": int(transitions["mismatch_count"]),
            "class_transition_summary_vs_reference": transitions,
            "geometry_feature_delta_vs_reference": (
                comparison_metrics.geometry_feature_delta_summary(
                    summary["_final_df"], reference_final, feature_runtime
                )
            ),
        }
    )
    return _strip_internal_fields(summary)


def run_candidates(
    execution_runs: Iterable[tuple[str, Path, Path]],
    *,
    staging_dir: Path,
    comparison_dir: Path,
    reference_final: pd.DataFrame,
    feature_runtime: Any,
) -> list[dict[str, Any]]:
    return [
        _run_candidate(
            run_name,
            execution_config,
            reported_config,
            staging_dir=staging_dir,
            comparison_dir=comparison_dir,
            reference_final=reference_final,
            feature_runtime=feature_runtime,
        )
        for run_name, execution_config, reported_config in execution_runs
    ]


@contextmanager
def staging_directory(output_root: Path, comparison_dir: Path) -> Iterator[Path]:
    staging_dir = Path(
        mkdtemp(prefix=f".{comparison_dir.name}.staging-", dir=output_root)
    )
    try:
        yield staging_dir
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def write_summary_alias(
    summary_json: Path | None,
    *,
    repo_root: Path,
    output_root: Path,
    comparison_dir: Path,
    summary: Mapping[str, Any],
) -> None:
    canonical_path = comparison_dir / "comparison_summary.json"
    if summary_json is None:
        write_json_atomic(output_root / "comparison_summary.json", summary)
        return
    summary_path = comparison_inputs.resolve_path(repo_root, summary_json)
    if summary_path != canonical_path:
        write_json_atomic(summary_path, summary)
