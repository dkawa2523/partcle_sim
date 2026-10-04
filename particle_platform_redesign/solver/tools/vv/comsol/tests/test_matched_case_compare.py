from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import cast

import pytest
import yaml
from tools.vv.comsol.compare_matched_case import (
    compare_matched_case,
    preflight_matched_case,
)


def _record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _records(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [_record(item) for item in value]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_table(path: Path, *, value: float = 1.0, label: str = "same") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "value", "label"])
        writer.writerow([1, value, label])


def _layer(artifact: str) -> dict[str, object]:
    return {
        "required": True,
        "artifact": artifact,
        "keys": [
            {
                "name": "id",
                "reference_column": "id",
                "candidate_column": "id",
                "kind": "integer",
            }
        ],
        "numeric": [
            {
                "name": "value",
                "reference_column": "value",
                "candidate_column": "value",
                "unit": "1",
                "absolute_tolerance": 0.1,
                "relative_tolerance": 0.0,
                "scale": 1.0,
            }
        ],
        "categorical": [
            {
                "name": "label",
                "reference_column": "label",
                "candidate_column": "label",
            }
        ],
    }


def _case(
    tmp_path: Path,
    *,
    changed_layer: str | None = None,
    categorical_change: bool = False,
) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    artifact_names = {
        "field": "field_probe",
        "force": "force_probe",
        "integrator": "integrator_probe",
        "trajectory": "trajectory",
        "boundary": "events",
    }
    artifacts: dict[str, dict[str, dict[str, object]]] = {
        "reference": {},
        "candidate": {},
    }
    for layer, artifact in artifact_names.items():
        for side in ("reference", "candidate"):
            path = artifact_root / side / f"{artifact}.csv"
            changed = layer == changed_layer and side == "candidate"
            _write_table(
                path,
                value=2.0 if changed and not categorical_change else 1.0,
                label="different" if changed and categorical_change else "same",
            )
            artifacts[side][artifact] = {
                "path": str(path.relative_to(artifact_root)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "comparison_id": "synthetic-matched-case",
        "classification": "external_comsol_matched_case_diagnostic",
        "claim_policy": "within_preregistered_tolerances_only",
        "readiness": [
            {
                "name": "deterministic_reference",
                "required": True,
                "status": "CONFIRMED",
                "evidence": "synthetic test receipt",
                "reason": "test setup",
            }
        ],
        "setting_alignment": [
            {
                "name": "integrator",
                "required_value": "rk4",
                "reference_value": "rk4",
                "candidate_value": "rk4",
                "reference_evidence": "synthetic reference settings",
                "candidate_evidence": "synthetic candidate settings",
            }
        ],
        "artifacts": artifacts,
        "layers": {name: _layer(artifact) for name, artifact in artifact_names.items()},
        "accuracy_evidence": [
            {
                "name": "time_convergence",
                "status": "NOT_TESTED",
                "evidence": "not part of this synthetic unit test",
                "reason": "comparison behavior only",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path, artifact_root


def test_preflight_blocks_unconfirmed_deterministic_reference(tmp_path: Path) -> None:
    manifest_path, artifact_root = _case(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["readiness"][0]["status"] = "BLOCKED"
    manifest["readiness"][0]["reason"] = "Brownian is still enabled"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    report = preflight_matched_case(manifest_path, artifact_root)

    assert report["status"] == "BLOCKED"
    assert report["ready_for_comparison"] is False
    assert _records(report["blockers"])[0]["category"] == "readiness"
    comparison = compare_matched_case(manifest_path, artifact_root)
    assert comparison["status"] == "BLOCKED"
    assert _record(comparison["claim"])["comsol_equal_accuracy"] == "NOT_CLAIMED"


@pytest.mark.parametrize(
    ("changed_layer", "categorical", "expected_owner"),
    [
        ("field", False, "field_or_geometry_sampling"),
        ("force", False, "force_or_charge_rhs"),
        ("integrator", False, "integrator_or_stage_coupling"),
        ("trajectory", False, "trajectory_accumulation_after_probes"),
        ("boundary", True, "boundary_event_or_wall_mapping"),
    ],
)
def test_first_discrepancy_is_attributed_to_its_layer(
    tmp_path: Path,
    changed_layer: str,
    categorical: bool,
    expected_owner: str,
) -> None:
    manifest_path, artifact_root = _case(
        tmp_path,
        changed_layer=changed_layer,
        categorical_change=categorical,
    )

    report = compare_matched_case(manifest_path, artifact_root)

    assert report["status"] == "FAIL"
    assert report["first_discrepancy_layer"] == expected_owner
    layer_report = _record(_record(report["layers"])[changed_layer])
    assert layer_report["status"] == "FAIL"
    assert _record(report["claim"])["comsol_equal_accuracy"] == "NOT_CLAIMED"


def test_pass_is_limited_to_registered_tolerances_not_equal_accuracy(tmp_path: Path) -> None:
    manifest_path, artifact_root = _case(tmp_path)

    report = compare_matched_case(manifest_path, artifact_root)

    assert report["status"] == "PASS"
    assert report["first_discrepancy_layer"] == "none_within_registered_tolerances"
    assert (
        _record(report["claim"])["supported_claim"]
        == "within_preregistered_tolerances_for_this_artifact_set"
    )
    assert _record(report["claim"])["comsol_equal_accuracy"] == "NOT_CLAIMED"
    assert _record(report["claim"])["golden_truth"] == "NOT_CLAIMED"
    assert _record(report["accuracy_evidence"])["status"] == "NOT_TESTED"


def test_preflight_rejects_changed_artifact_hash(tmp_path: Path) -> None:
    manifest_path, artifact_root = _case(tmp_path)
    changed = artifact_root / "candidate" / "field_probe.csv"
    _write_table(changed, value=1.01)

    report = preflight_matched_case(manifest_path, artifact_root)

    assert report["status"] == "BLOCKED"
    assert any(blocker["category"] == "artifact_hash" for blocker in _records(report["blockers"]))
