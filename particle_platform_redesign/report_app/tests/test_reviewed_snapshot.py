"""Report snapshot preserves identity and distinguishes missing execution evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "report_snapshot", ROOT / "analysis/build_report_snapshot.py"
)
assert SPEC is not None and SPEC.loader is not None
snapshot_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot_module)


def _existing(tmp_path: Path) -> Path:
    data = json.loads((ROOT / "report_app/src/data.json").read_text(encoding="utf-8"))
    data["presentation"] = {
        "hiddenBlocks": ["physics-table"],
        "textEdits": {"report:introduction": "User text"},
    }
    path = tmp_path / "existing.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_snapshot_preserves_app_identity_and_presentation(tmp_path: Path) -> None:
    path = _existing(tmp_path)
    original = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = snapshot_module.build_snapshot(path)
    for key in ("id", "title", "presentation", "filters"):
        assert rebuilt[key] == original[key]
    assert tuple(rebuilt["queries"]) == snapshot_module.QUERY_IDS
    assert rebuilt["buildStatus"] == "updating"
    assert rebuilt["report"]["asOf"] is None
    assert len(rebuilt["queries"]["physics_scope"]["rows"]) > 0
    for binding in rebuilt["reproducibility"]["authorities"]:
        assert (
            binding["sha256"]
            == "sha256:" + hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
        )


def test_missing_execution_evidence_is_unavailable_not_zero(tmp_path: Path) -> None:
    rebuilt = snapshot_module.build_snapshot(_existing(tmp_path), build_status="complete")
    for query_id in ("case_audit", "trajectory_divergence"):
        row = rebuilt["queries"][query_id]["rows"][0]
        assert row["status"] == "unavailable"
        assert row["value"] is None
    assert rebuilt["queries"]["quality_findings"]["rows"][0]["evidence"] is None


def test_optional_reviewed_quality_preserves_exact_rows_and_source(tmp_path: Path) -> None:
    path = _existing(tmp_path)
    reviewed = {
        "queries": {
            "quality_findings": {
                "rows": [
                    {
                        "severity": "not_tested",
                        "finding": "No execution performed",
                        "disposition": "Design only",
                    }
                ],
                "source": {
                    "files": ["local-quality-receipt.json"],
                    "label": "Explicit local evidence",
                },
            }
        }
    }
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps(reviewed), encoding="utf-8")
    rebuilt = snapshot_module.build_snapshot(path, quality=quality)
    query = rebuilt["queries"]["quality_findings"]
    assert query["rows"] == reviewed["queries"]["quality_findings"]["rows"]
    assert query["source"]["files"] == ["local-quality-receipt.json"]
    assert (
        rebuilt["reproducibility"]["optional_inputs"][0]["sha256"]
        == "sha256:" + hashlib.sha256(quality.read_bytes()).hexdigest()
    )


def test_optional_input_cannot_replace_unrelated_queries(tmp_path: Path) -> None:
    quality = tmp_path / "quality.json"
    quality.write_text(
        json.dumps({"queries": {"design_decisions": {"rows": [], "source": {"files": ["x"]}}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unrequested query"):
        snapshot_module.build_snapshot(_existing(tmp_path), quality=quality)
