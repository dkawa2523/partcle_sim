"""Report snapshot preserves identity and distinguishes missing execution evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
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
    assert rebuilt["reproducibility"]["reproduction"]["scope"] == "design_only"


def test_current_authority_drift_and_roster_are_rejected_without_rebinding_history(
    tmp_path: Path,
) -> None:
    paths = (*snapshot_module.AUTHORITY_PATHS, "analysis/build_report_snapshot.py")
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"current authority")
    digest = "sha256:" + hashlib.sha256(b"current authority").hexdigest()
    snapshot = {
        "reproducibility": {
            "authorities": [{"path": name, "sha256": digest} for name in paths[:-1]],
            "producer": {"path": paths[-1], "sha256": digest},
            "reviewed_evidence": [{"execution_source": "old-commit", "sha256": "historical-hash"}],
        }
    }
    assert snapshot_module.validate_current_authorities(snapshot, tmp_path) == []
    (tmp_path / "README.md").write_bytes(b"new current prose")
    assert (
        "hash drift: README.md"
        in snapshot_module.validate_current_authorities(snapshot, tmp_path)[0]
    )
    snapshot["reproducibility"]["authorities"].append({"path": "../untrusted", "sha256": digest})
    assert "roster differs" in snapshot_module.validate_current_authorities(snapshot, tmp_path)[0]
    assert snapshot["reproducibility"]["reviewed_evidence"][0]["execution_source"] == "old-commit"


def test_reconstruction_includes_all_supplied_projections_and_preserves_external_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = {}
    for name, query in (
        ("assessment", "case_audit"),
        ("quality", "quality_findings"),
        ("performance", "implementation_work_packages"),
    ):
        path = tmp_path / f"{name} with spaces.json"
        path.write_text(
            json.dumps(
                {
                    "queries": {
                        query: {
                            "rows": [{"status": "unavailable", "value": None}],
                            "source": {
                                "files": ["historical-receipt.json"],
                                "evidenceFlow": [
                                    {
                                        "title": "Original observation",
                                        "detail": "Historical execution",
                                    }
                                ],
                            },
                            "methods": [
                                {"language": "text", "code": "Original measurement method"}
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        inputs[name] = path
    output = tmp_path / "new report.json"
    rebuilt = snapshot_module.build_snapshot(_existing(tmp_path), output=output, **inputs)
    reproduction = rebuilt["reproducibility"]["reproduction"]
    assert reproduction["scope"] == "declared_projections"
    arguments = reproduction["arguments"]
    for name, path in (*inputs.items(), ("output", output)):
        assert Path(ROOT / "solver" / arguments[arguments.index(f"--{name}") + 1]).resolve() == path
    methods = rebuilt["queries"]["quality_findings"]["methods"]
    assert methods[0]["code"] == "Original measurement method"
    assert len(methods) == 2 and "--assessment" in methods[1]["code"]
    original_methods = list(methods)
    inputs["quality"].write_text(
        json.dumps({"queries": {"quality_findings": rebuilt["queries"]["quality_findings"]}}),
        encoding="utf-8",
    )
    rebuilt_again = snapshot_module.build_snapshot(_existing(tmp_path), output=output, **inputs)
    assert rebuilt_again["queries"]["quality_findings"]["methods"] == original_methods
    flow = rebuilt_again["queries"]["quality_findings"]["source"]["evidenceFlow"]
    instructions = [step for step in flow if step["title"] == "Reconstruct this report snapshot"]
    assert len(instructions) == 1
    assert all(f"--{name}" in instructions[0]["detail"] for name in inputs)
    captured = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: captured.append((argv, kwargs)))
    exec(instructions[0]["detail"].rsplit("\n", 1)[-1], {})
    assert captured == [(arguments, {"check": True})]
    assert {"title": "Original observation", "detail": "Historical execution"} in flow


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
        {"rows": [], "source": {"files": ["old.json"], "evidenceFlow": [17]}},
        {"rows": [], "source": {"files": ["old.json"]}, "methods": "shell"},
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
