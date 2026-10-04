from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

COMSOL_ROOT = Path(__file__).resolve().parents[1]
SOLVER_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = COMSOL_ROOT / "cases" / "m3c_critical_boundaries_v1.json"
EVIDENCE_ROOT = SOLVER_ROOT / "evidence" / "m3c0" / "critical_boundaries_v1"
REGISTERED_SOURCES = {
    "config": CONFIG_PATH,
    "evaluator": COMSOL_ROOT / "evaluate_m3c_critical_boundaries.py",
    "java": COMSOL_ROOT / "comsol" / "RunM3CCriticalBoundaries.java",
    "runner": COMSOL_ROOT / "run_m3c_critical_boundaries.ps1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compact_evidence_is_bound_to_registered_sources_and_gates() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE_ROOT / "comparison_manifest.json").read_text(encoding="utf-8"))
    with (EVIDENCE_ROOT / "gates.csv").open(newline="", encoding="utf-8") as stream:
        gates = list(csv.DictReader(stream))

    source_hashes = {name: _sha256(path) for name, path in REGISTERED_SOURCES.items()}
    assert manifest["comsol_provenance"]["staged_sources"] == source_hashes
    assert manifest["configuration_sha256"] == source_hashes["config"]
    assert manifest["evaluation_id"] == config["evaluation_id"]
    assert manifest["evaluation_revision"] == config["evaluation_revision"]

    statuses = Counter(row["status"] for row in gates)
    assert manifest["gate_counts"] == {
        "pass": statuses["PASS"],
        "fail": statuses["FAIL"],
    }
    assert manifest["scientific_status"] == "PASS"
    assert statuses == Counter({"PASS": 132})

    expected_steps = {step["label"] for step in config["case"]["fixed_rk4_steps"]}
    assert {row["step_label"] for row in gates} == expected_steps
    assert {row["producer"] for row in gates} == {
        "candidate",
        "comsol",
        "cross_solver",
    }
    assert set(manifest["comsol_raw_sha256"]) == expected_steps
