from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
STANDARD_ARTIFACTS = (
    "final_particles.csv",
    "run_summary.json",
    "wall_summary.csv",
)
DEBUG_ARTIFACTS = (
    "trajectory.npy",
    "trajectory_frames.csv",
    "wall_events.csv",
    "step_summary.csv",
    "force_contributions.csv",
    "debug_diagnostics.json",
)
_JSON_ARTIFACT_TYPES = {
    "run_summary.json": "particle_tracer.run_summary",
    "debug_diagnostics.json": "particle_tracer.debug_diagnostics",
}


def _validate_json(path: Path, *, artifact_type: str) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"invalid JSON: {exc}"
    if not isinstance(value, Mapping):
        return "JSON root must be an object"
    if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
        return f"schema_version must be {SCHEMA_VERSION}"
    if str(value.get("artifact_type", "")) != artifact_type:
        return f"artifact_type must be {artifact_type!r}"
    return None


def _validate_csv(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return "CSV header is missing"
            if "schema_version" not in reader.fieldnames:
                return "schema_version column is missing"
            for line_number, row in enumerate(reader, start=2):
                if str(row.get("schema_version", "")).strip() != str(SCHEMA_VERSION):
                    return (
                        f"schema_version must be {SCHEMA_VERSION} at line {line_number}"
                    )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return f"invalid CSV: {exc}"
    return None


def _artifact_error(path: Path, name: str) -> str | None:
    if not path.is_file():
        return "missing"
    if artifact_type := _JSON_ARTIFACT_TYPES.get(name):
        return _validate_json(path, artifact_type=artifact_type)
    if path.suffix == ".csv":
        return _validate_csv(path)
    if name == "trajectory.npy" and path.stat().st_size == 0:
        return "empty file"
    return None


def _artifact_status(path: Path, name: str) -> dict[str, str]:
    error = _artifact_error(path, name)
    return {
        "path": str(path),
        "status": "ok" if error is None else "error",
        "message": error or "",
    }


def _unexpected_artifacts(base: Path, expected: set[str]) -> list[str]:
    if not base.is_dir():
        return []
    return sorted(path.name for path in base.iterdir() if path.name not in expected)


def validate_artifacts(root: Path, *, require_debug: bool = False) -> dict[str, Any]:
    """Validate only the canonical v0.2 artifact set."""

    base = Path(root).resolve()
    required = STANDARD_ARTIFACTS + (DEBUG_ARTIFACTS if require_debug else ())
    status = {name: _artifact_status(base / name, name) for name in required}
    unexpected = _unexpected_artifacts(base, set(required))
    failures = [name for name, item in status.items() if item["status"] == "error"]
    failures.extend(f"unexpected:{name}" for name in unexpected)
    return {
        "artifact_type": "particle_tracer.artifact_validation",
        "schema_version": SCHEMA_VERSION,
        "root": str(base),
        "mode": "debug" if require_debug else "standard",
        "passed": not failures,
        "failures": failures,
        "unexpected_artifacts": unexpected,
        "artifacts": status,
    }


__all__ = ("DEBUG_ARTIFACTS", "STANDARD_ARTIFACTS", "validate_artifacts")
