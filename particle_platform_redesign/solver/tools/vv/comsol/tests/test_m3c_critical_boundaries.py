from __future__ import annotations

import hashlib
from pathlib import Path

from tools.vv.comsol.evaluate_m3c_critical_boundaries import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_current_rz_public_runs_match_native_and_piecewise_analytic_motion(tmp_path: Path) -> None:
    reference = ROOT / "references" / "rz"
    before = {
        p.relative_to(reference): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in reference.rglob("*")
        if p.is_file()
    }
    report = evaluate(
        reference,
        ROOT / "cases" / "m3c_critical_boundaries_v1.json",
        tmp_path / "evaluation",
    )
    assert report["scientific_status"] == "PASS"
    assert report["gate_counts"] == {"pass": 132, "fail": 0}
    after = {
        p.relative_to(reference): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in reference.rglob("*")
        if p.is_file()
    }
    assert after == before
