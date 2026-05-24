from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VV_DIR = ROOT / "docs" / "productization" / "sim_rev3" / "vv"


def test_vv_acceptance_matrix_is_parseable_and_metric_based() -> None:
    matrix_path = VV_DIR / "acceptance_matrix.csv"
    rows = list(csv.DictReader(matrix_path.open("r", encoding="utf-8", newline="")))

    assert rows
    assert {"workflow", "case_id", "acceptance_metrics", "must_not_accept"}.issubset(rows[0].keys())
    assert {row["workflow"] for row in rows} >= {"verification", "validation"}
    assert {"V3", "V8", "C3", "C6"}.issubset({row["case_id"] for row in rows})
    for row in rows:
        metrics = [part.strip() for part in row["acceptance_metrics"].split(";") if part.strip()]
        assert len(metrics) >= 2
        assert row["must_not_accept"] != "endpoint_count_only" or "endpoint" not in row["acceptance_metrics"].lower()


def test_vv_workflow_docs_name_required_root_artifacts() -> None:
    readme = (VV_DIR / "README.md").read_text(encoding="utf-8")

    assert "import -> preprocess -> first-step -> wall events -> ensemble" in readme
    assert "--reference-scope sampled" in readme
    assert "--reference-scope full" in readme
    assert "run_summary_compare.csv" in readme
    assert "shard_artifacts_manifest.json" in readme
    assert "endpoint counts alone" in readme


def test_phase_9_cleanup_report_and_ignore_rules_cover_generated_artifacts() -> None:
    report = (ROOT / "docs" / "productization" / "sim_rev3" / "codex_notes" / "phase_9_cleanup_report.md").read_text(
        encoding="utf-8"
    )
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for name in (
        "_case_focus_ring_plasma_assumption_*/",
        "/report.md",
        "/particle_tracer_all_figures.zip",
        "/particle_tracer_decision_deck_with_figures_v4_complete.pptx",
        "/docs/assets/icp_validation/",
    ):
        assert name in ignore_text
    for name in (
        "_out_focus_ring_100_check/",
        "_out_focus_ring_100_run/",
        "_case_focus_ring_plasma_assumption_100/",
        "data/*.mph",
        "output.mode: standard",
        "output.mode: debug",
    ):
        assert name in report
