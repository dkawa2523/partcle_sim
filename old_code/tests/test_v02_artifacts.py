from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from particle_tracer_unified import load_case, simulate, write_result
from particle_tracer_unified.artifacts import (
    DEBUG_ARTIFACTS,
    STANDARD_ARTIFACTS,
    validate_artifacts,
)

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "v02_minimal" / "run_config.yaml"
)


def _write_standard_result(output_dir: Path) -> None:
    write_result(simulate(load_case(EXAMPLE)), output_dir)


def test_standard_artifact_validation_accepts_public_writer_contract(tmp_path) -> None:
    _write_standard_result(tmp_path)
    report = validate_artifacts(tmp_path)

    assert report["passed"] is True
    assert report["failures"] == []


def test_debug_artifact_validation_reports_missing_debug_files(tmp_path) -> None:
    _write_standard_result(tmp_path)
    report = validate_artifacts(tmp_path, require_debug=True)

    assert report["passed"] is False
    assert "trajectory.npy" in report["failures"]


def test_artifact_validation_rejects_undeclared_files(tmp_path) -> None:
    _write_standard_result(tmp_path)
    (tmp_path / "legacy_solver_report.json").write_text("{}", encoding="utf-8")

    report = validate_artifacts(tmp_path)

    assert report["passed"] is False
    assert report["unexpected_artifacts"] == ["legacy_solver_report.json"]


def _write_minimal_artifacts(root: Path) -> None:
    json_artifacts = {
        "run_summary.json": "particle_tracer.run_summary",
        "debug_diagnostics.json": "particle_tracer.debug_diagnostics",
    }
    for name in (*STANDARD_ARTIFACTS, *DEBUG_ARTIFACTS):
        path = root / name
        if name in json_artifacts:
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "artifact_type": json_artifacts[name],
                    }
                ),
                encoding="utf-8",
            )
        elif name.endswith(".csv"):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["schema_version"])
                writer.writerow([2])
        else:
            path.write_bytes(b"npy")


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("run_summary.json", "{", "invalid JSON:"),
        ("run_summary.json", "[]", "JSON root must be an object"),
        (
            "debug_diagnostics.json",
            '{"schema_version": 1}',
            "schema_version must be 2",
        ),
        (
            "debug_diagnostics.json",
            '{"schema_version": 2, "artifact_type": "other"}',
            "artifact_type must be 'particle_tracer.debug_diagnostics'",
        ),
        ("wall_summary.csv", "", "CSV header is missing"),
        ("wall_summary.csv", "value\n1\n", "schema_version column is missing"),
        (
            "force_contributions.csv",
            "schema_version\n1\n",
            "schema_version must be 2 at line 2",
        ),
    ],
)
def test_artifact_validation_preserves_type_specific_errors(
    tmp_path: Path,
    name: str,
    content: str,
    message: str,
) -> None:
    _write_minimal_artifacts(tmp_path)
    (tmp_path / name).write_text(content, encoding="utf-8")

    report = validate_artifacts(tmp_path, require_debug=True)

    assert report["failures"] == [name]
    assert report["artifacts"][name]["message"].startswith(message)


def test_artifact_validation_rejects_empty_trajectory(tmp_path: Path) -> None:
    _write_minimal_artifacts(tmp_path)
    (tmp_path / "trajectory.npy").write_bytes(b"")

    report = validate_artifacts(tmp_path, require_debug=True)

    assert report["failures"] == ["trajectory.npy"]
    assert report["artifacts"]["trajectory.npy"]["message"] == "empty file"


def test_artifact_validation_handles_missing_root(tmp_path: Path) -> None:
    report = validate_artifacts(tmp_path / "missing")

    assert report["failures"] == list(STANDARD_ARTIFACTS)
    assert report["unexpected_artifacts"] == []
