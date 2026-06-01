from __future__ import annotations

import json
from pathlib import Path

from tools.validate_comparison_artifacts import main, validate_artifacts


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, text: str = "part_id,outcome,wall_mode,count\n1,stuck,stick,1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_base_root(root: Path) -> None:
    _write_json(root / "solver_report.json", {"particle_count": 1})
    _write_json(root / "prepared_runtime_summary.json", {"particles": 1, "source_model_summary": {"released_count": 1}})
    _write_csv(root / "wall_summary_by_part.csv")
    _write_json(root / "comparison_summary.json", {"summary_schema_version": 1})


def test_complete_sampled_artifact_root_passes(tmp_path: Path) -> None:
    root = tmp_path / "sampled"
    _write_base_root(root)

    summary = validate_artifacts(root, workflow="sampled")

    assert summary["status"] == "pass"
    assert summary["missing_required"] == []
    assert summary["artifacts"]["source_model_summary.json"]["status"] == "available_embedded"


def test_missing_solver_report_fails_with_actionable_message(tmp_path: Path) -> None:
    root = tmp_path / "sampled"
    _write_json(root / "prepared_runtime_summary.json", {"particles": 1})
    _write_csv(root / "wall_summary_by_part.csv")
    _write_json(root / "comparison_summary.json", {"summary_schema_version": 1})

    summary = validate_artifacts(root, workflow="sampled")

    assert summary["status"] == "fail"
    assert "solver_report.json" in summary["missing_required"]
    assert any("Run the solver first" in message for message in summary["actionable_messages"])


def test_first_step_is_optional_by_default_and_required_by_flag(tmp_path: Path) -> None:
    root = tmp_path / "sampled"
    _write_base_root(root)

    default_summary = validate_artifacts(root, workflow="sampled")
    required_summary = validate_artifacts(root, workflow="sampled", require_first_step=True)

    assert default_summary["status"] == "pass"
    assert "first_step_compare_summary.json" in default_summary["missing_optional"]
    assert required_summary["status"] == "fail"
    assert "first_step_compare_summary.json" in required_summary["missing_required"]


def test_collision_diagnostics_are_optional_unless_debug_required(tmp_path: Path) -> None:
    root = tmp_path / "full"
    _write_base_root(root)

    default_summary = validate_artifacts(root, workflow="full")
    required_summary = validate_artifacts(root, workflow="full", require_debug=True)

    assert default_summary["status"] == "pass"
    assert "collision_diagnostics.json" in default_summary["missing_optional"]
    assert required_summary["status"] == "fail"
    assert "collision_diagnostics.json" in required_summary["missing_required"]


def test_validator_cli_writes_default_summary(tmp_path: Path) -> None:
    root = tmp_path / "sampled"
    _write_base_root(root)

    assert main([str(root), "--workflow", "sampled"]) == 0

    summary = json.loads((root / "artifact_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
