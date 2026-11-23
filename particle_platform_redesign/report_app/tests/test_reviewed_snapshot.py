"""Report snapshot preserves identity and distinguishes missing execution evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "report_snapshot", ROOT / "analysis/build_report_snapshot.py"
)
assert SPEC is not None and SPEC.loader is not None
snapshot_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = snapshot_module
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


@pytest.mark.parametrize(
    "query",
    [
        {"rows": [17], "source": {"files": ["historical.json"]}},
        {"rows": [], "source": {"files": "historical.json"}},
        {"rows": [], "source": {"files": [17]}},
        {"rows": [{"status": "unavailable", "value": 0}], "source": {"files": ["old.json"]}},
        {"rows": [{"severity": "passed"}], "source": {"files": ["old.json"]}},
    ],
)
def test_malformed_reviewed_rows_or_sources_fail_before_publication(
    tmp_path: Path, query: dict
) -> None:
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({"queries": {"quality_findings": query}}))
    with pytest.raises(ValueError):
        snapshot_module.build_snapshot(_existing(tmp_path), quality=quality)


def test_projection_rows_and_hash_use_the_same_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _existing(tmp_path)
    quality = tmp_path / "quality.json"
    document = {
        "queries": {
            "quality_findings": {
                "rows": [
                    {
                        "severity": "not_tested",
                        "finding": "Version A",
                        "disposition": "No execution",
                    }
                ],
                "source": {"files": ["historical.json"]},
            }
        }
    }
    raw = json.dumps(document).encode()
    quality.write_bytes(raw)
    read_bytes = Path.read_bytes

    def changed_after_read(path: Path) -> bytes:
        result = read_bytes(path)
        if path == quality:
            path.write_bytes(raw.replace(b"Version A", b"Version B"))
        return result

    monkeypatch.setattr(Path, "read_bytes", changed_after_read)
    snapshot = snapshot_module.build_snapshot(existing, quality=quality)
    assert snapshot["queries"]["quality_findings"]["rows"][0]["finding"] == "Version A"
    assert (
        snapshot["reproducibility"]["optional_inputs"][0]["sha256"]
        == "sha256:" + hashlib.sha256(raw).hexdigest()
    )


def test_structured_evidence_keeps_history_scope_and_supported_wire_shape(tmp_path: Path) -> None:
    quality, local = tmp_path / "quality.json", tmp_path / "execution.json"
    local.write_text('{"executed":true}')
    digest = "sha256:" + hashlib.sha256(local.read_bytes()).hexdigest()
    references = [
        {
            "location": str(local),
            "availability": "bound_local",
            "sha256": digest,
            "execution_source": "prior-source-sha",
            "scope": "One prior tiny execution",
        },
        {
            "location": "missing-old-native.csv",
            "availability": "historical_unavailable",
            "scope": "Historical citation; no current native recomputation",
        },
    ]
    quality.write_text(
        json.dumps(
            {
                "queries": {
                    "quality_findings": {
                        "rows": [
                            {
                                "severity": "not_tested",
                                "finding": "Scoped receipt",
                                "disposition": "Retain history",
                            }
                        ],
                        "source": {"files": references},
                    }
                }
            }
        )
    )
    snapshot = snapshot_module.build_snapshot(_existing(tmp_path), quality=quality)
    source = snapshot["queries"]["quality_findings"]["source"]
    assert source["files"] == [{"label": str(local)}, {"label": "missing-old-native.csv"}]
    evidence = snapshot["reproducibility"]["reviewed_evidence"]
    assert evidence[0]["execution_source"] == "prior-source-sha"
    assert evidence[0]["sha256"] == digest
    assert evidence[1]["availability"] == "historical_unavailable"
    assert digest in source["evidenceFlow"][0]["detail"]
    local.write_text("changed")
    with pytest.raises(ValueError, match="bound_local evidence missing or hash mismatch"):
        snapshot_module.build_snapshot(_existing(tmp_path), quality=quality)


@pytest.mark.parametrize("boundary", ["fsync", "replace"])
def test_snapshot_publication_preserves_previous_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    destination = tmp_path / "data.json"
    original = b'{"old":"valid"}\n'
    destination.write_bytes(original)

    def fail(*_args, **_kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(snapshot_module.os, boundary, fail)
    with pytest.raises(OSError, match="injected publication failure"):
        snapshot_module.publish_snapshot({"new": "complete"}, destination)
    assert destination.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data.json"]


def test_snapshot_publication_binds_the_completed_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "data.json"
    binding = snapshot_module.publish_snapshot({"rows": [{"value": 3}]}, destination)
    assert json.loads(destination.read_text()) == {"rows": [{"value": 3}]}
    assert binding["sha256"] == "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest()
