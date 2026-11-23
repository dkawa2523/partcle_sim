from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from tools.vv.comsol.evaluate_xy_minimal_suite import evaluate

REFERENCE = Path(__file__).resolve().parents[1] / "references" / "xy"


def test_current_xy_public_runs_match_native_and_analytic_motion(tmp_path: Path) -> None:
    report = evaluate(
        REFERENCE / "raw",
        tmp_path / "evaluation",
        run_receipt=REFERENCE / "run_receipt.json",
    )
    assert report["scientific_status"] == "PASS"
    judgments = report["judgments"]
    assert isinstance(judgments, dict)
    assert set(judgments.values()) == {"PASS"}
    assert report["scenario_count"] == 7
    metrics = report["metrics"]
    assert isinstance(metrics, list)
    assert len(metrics) == 21


def test_native_reference_tampering_is_rejected_before_execution(tmp_path: Path) -> None:
    copied = tmp_path / "reference"
    shutil.copytree(REFERENCE, copied)
    raw = next((copied / "raw").rglob("state_raw_wide.csv"))
    raw.write_bytes(raw.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="raw COMSOL artifact lineage differs"):
        evaluate(copied / "raw", tmp_path / "evaluation", run_receipt=copied / "run_receipt.json")
