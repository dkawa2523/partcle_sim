from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


BASE_REQUIRED = (
    "solver_report.json",
    "prepared_runtime_summary.json",
    "wall_summary_by_part.csv",
)
OPTIONAL_ARTIFACTS = (
    "source_model_summary.json",
    "source_particle_diagnostics.csv",
    "first_step_compare_summary.json",
    "collision_diagnostics.json",
)
JSON_ARTIFACTS = {
    "solver_report.json",
    "prepared_runtime_summary.json",
    "source_model_summary.json",
    "comparison_summary.json",
    "first_step_compare_summary.json",
    "collision_diagnostics.json",
}
CSV_ARTIFACTS = {
    "wall_summary_by_part.csv",
    "source_particle_diagnostics.csv",
}

ACTIONABLE_MESSAGES = {
    "solver_report.json": "Run the solver first; this root is missing solver_report.json.",
    "prepared_runtime_summary.json": "Run a normal solver case or `run_from_yaml.py --prepare-only/--check-input` to write prepared_runtime_summary.json.",
    "wall_summary_by_part.csv": "Use standard/debug output with wall summaries enabled, or aggregate shards with tools/collect_run_summaries.py --root-artifacts-dir.",
    "comparison_summary.json": "Run tools/compare_against_reference.py with --output-root <root>; it writes stable root comparison_summary.json.",
    "source_model_summary.json": "Run --check-input or enable source diagnostics/debug output when source preprocessing diagnostics are needed.",
    "source_particle_diagnostics.csv": "Run --check-input, output.mode: debug, or explicit output.write_source_diagnostics when source particle rows are required.",
    "first_step_compare_summary.json": "Run py -3 -m particle_tracer_unified.compare.first_step_compare --output-dir <root> or copy/aggregate first-step artifacts into the root.",
    "collision_diagnostics.json": "Use output.mode: debug or explicit output.write_collision_diagnostics when collision diagnostics are required.",
}


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, Mapping):
        return None, "JSON root is not an object"
    return dict(payload), ""


def _csv_has_header(path: Path) -> bool:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except (IndexError, OSError, UnicodeDecodeError):
        return False
    return bool(first_line.strip())


def _embedded_source_summary(prepared_path: Path) -> bool:
    if not prepared_path.exists():
        return False
    payload, _error = _load_json(prepared_path)
    embedded = payload.get("source_model_summary", {}) if isinstance(payload, Mapping) else {}
    return isinstance(embedded, Mapping) and bool(embedded)


def _artifact_status(root: Path, filename: str, *, prepared_has_source_summary: bool) -> dict[str, Any]:
    path = root / filename
    if filename == "source_model_summary.json" and not path.exists() and prepared_has_source_summary:
        return {
            "status": "available_embedded",
            "path": str(root / "prepared_runtime_summary.json"),
            "message": "source_model_summary is embedded in prepared_runtime_summary.json",
        }
    if not path.exists():
        return {"status": "missing", "path": str(path), "message": ACTIONABLE_MESSAGES.get(filename, "")}
    if filename in JSON_ARTIFACTS:
        _payload, error = _load_json(path)
        if error:
            return {"status": "malformed", "path": str(path), "message": f"Malformed JSON: {error}"}
    if filename in CSV_ARTIFACTS and not _csv_has_header(path):
        return {"status": "malformed", "path": str(path), "message": "CSV is empty or missing a header row"}
    return {"status": "found", "path": str(path), "message": ""}


def validate_artifacts(
    root: Path,
    *,
    workflow: str,
    require_first_step: bool = False,
    require_debug: bool = False,
    require_source_diagnostics: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    prepared_has_source_summary = _embedded_source_summary(root / "prepared_runtime_summary.json")
    required = set(BASE_REQUIRED)
    if str(workflow) in {"sampled", "full"}:
        required.add("comparison_summary.json")
    if bool(require_first_step):
        required.add("first_step_compare_summary.json")
    if bool(require_debug):
        required.add("collision_diagnostics.json")
    if bool(require_source_diagnostics):
        required.add("source_particle_diagnostics.csv")

    names = sorted(required.union(OPTIONAL_ARTIFACTS).union({"comparison_summary.json"}))
    artifacts = {
        name: _artifact_status(root, name, prepared_has_source_summary=prepared_has_source_summary)
        for name in names
    }
    missing_required = [
        name
        for name in sorted(required)
        if artifacts[name]["status"] in {"missing", "malformed"}
    ]
    missing_optional = [
        name
        for name in names
        if name not in required and artifacts[name]["status"] == "missing"
    ]
    malformed = [
        name
        for name in names
        if artifacts[name]["status"] == "malformed"
    ]
    warnings = [
        f"Optional artifact missing: {name}. {ACTIONABLE_MESSAGES.get(name, '')}".strip()
        for name in missing_optional
    ]
    if str(workflow) == "sharded" and artifacts["comparison_summary.json"]["status"] == "missing":
        warnings.append(
            "Sharded aggregation does not merge per-shard comparison summaries; run compare tools at the root to create comparison_summary.json."
        )
    actionable_messages = [
        f"Required artifact problem: {name}. {artifacts[name].get('message') or ACTIONABLE_MESSAGES.get(name, '')}".strip()
        for name in missing_required
    ]
    actionable_messages.extend(
        f"Malformed artifact: {name}. {artifacts[name].get('message', '')}".strip()
        for name in malformed
        if name not in missing_required
    )
    status = "fail" if missing_required else "pass"
    return {
        "schema_version": 1,
        "workflow": str(workflow),
        "root": str(root),
        "status": status,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "warnings": warnings,
        "artifacts": artifacts,
        "actionable_messages": actionable_messages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate comparison root artifacts for sampled, full, or sharded workflows.")
    parser.add_argument("root", type=Path, help="Root output directory to validate.")
    parser.add_argument("--workflow", choices=("sampled", "full", "sharded"), required=True)
    parser.add_argument("--require-first-step", action="store_true")
    parser.add_argument("--require-debug", action="store_true")
    parser.add_argument("--require-source-diagnostics", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None, help="Defaults to <root>/artifact_validation_summary.json.")
    args = parser.parse_args(argv)

    summary = validate_artifacts(
        args.root,
        workflow=str(args.workflow),
        require_first_step=bool(args.require_first_step),
        require_debug=bool(args.require_debug),
        require_source_diagnostics=bool(args.require_source_diagnostics),
    )
    output_path = args.json_out or (Path(args.root) / "artifact_validation_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
