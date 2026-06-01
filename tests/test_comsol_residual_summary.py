from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.summarize_comsol_residual_gap import main


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_comsol_residual_gap_summary_reads_compact_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    preflight_dir = tmp_path / "check"
    first_step_dir = tmp_path / "first_step"
    output_dir = tmp_path / "residual_gap"
    boundary_summary = tmp_path / "boundary" / "boundary_hit_comparison.json"
    ensemble_summary = tmp_path / "ensemble" / "comparison_summary.json"
    near_wall_summary = tmp_path / "near_wall" / "near_wall_nohit_summary.json"
    full_diagnostics_summary = tmp_path / "full" / "full_comsol_diagnostics_summary.json"

    _write_json(
        run_dir / "solver_report.json",
        {
            "particle_count": 2,
            "released_count": 2,
            "stuck_count": 1,
            "absorbed_count": 0,
            "escaped_count": 0,
            "output_mode": "standard",
            "timing_s": {"solver_core_s": 0.12},
            "valid_mask_mixed_stencil_count": 1,
            "valid_mask_hard_invalid_count": 0,
            "unresolved_crossing_count": 0,
            "state_geometry_summary": {
                "near_boundary_threshold_m": 0.01,
                "by_state": {
                    "active_free_flight": {
                        "count": 1,
                        "near_boundary_count": 1,
                        "nearest_part_counts": [{"part_id": 7, "count": 1}],
                    }
                },
            },
        },
    )
    _write_json(run_dir / "wall_summary.json", {"total_wall_interactions": 1})
    _write_csv(
        run_dir / "final_particles.csv",
        [
            {"particle_id": 1, "active": 1, "stuck": 0, "absorbed": 0, "escaped": 0, "invalid_mask_stopped": 0},
            {"particle_id": 2, "active": 0, "stuck": 1, "absorbed": 0, "escaped": 0, "invalid_mask_stopped": 0},
        ],
    )
    _write_json(preflight_dir / "provider_contract_report.json", {"passed": True, "status_counts": {"clean": 4}})
    _write_json(preflight_dir / "input_contract_report.json", {"passed": True, "mode": "strict", "status_counts": {"clean": 2}})
    _write_json(
        preflight_dir / "prepared_runtime_summary.json",
        {
            "spatial_dim": 2,
            "coordinate_system": "cartesian_xy",
            "axis_names": ["x", "y"],
            "source_model_summary": {
                "boundary_release_enabled": 1,
                "boundary_release_applied_count": 2,
                "boundary_release_capture_tolerance_m": 5e-4,
                "boundary_release_inward_offset_m": 2e-6,
            },
        },
    )
    _write_csv(
        preflight_dir / "source_particle_diagnostics.csv",
        [{"particle_id": 1, "boundary_release_applied": 1}, {"particle_id": 2, "boundary_release_applied": 1}],
    )
    _write_json(
        first_step_dir / "first_step_compare_summary.json",
        {
            "particle_count": 2,
            "compared_particle_count": 2,
            "stochastic_policy": "off",
            "enabled_forces": ["drag", "electric"],
            "position_error_m": {"count": 2, "mean": 1e-6, "max": 2e-6},
            "velocity_error_mps": {"count": 2, "mean": 0.1, "max": 0.2},
        },
    )
    _write_csv(
        first_step_dir / "first_step_error.csv",
        [
            {"particle_id": 1, "position_error_m": 1e-6, "velocity_error_mps": 0.1, "speed_ratio": 1.01},
            {"particle_id": 2, "position_error_m": 2e-6, "velocity_error_mps": 0.2, "speed_ratio": 0.99},
        ],
    )
    _write_csv(
        first_step_dir / "force_contributions.csv",
        [
            {"particle_id": 1, "drag_ax": 1.0, "drag_ay": 0.0, "total_ax": 1.5, "total_ay": 0.0},
            {"particle_id": 2, "drag_ax": 2.0, "drag_ay": 0.0, "total_ax": 2.5, "total_ay": 0.0},
        ],
    )
    _write_json(boundary_summary, {"matched_first_hit_count": 2, "hit_time_error_s": {"count": 2, "max": 1e-8}})
    _write_json(
        near_wall_summary,
        {
            "near_wall_nohit_count": 1,
            "near_wall_active_count": 1,
            "classification_counts": {"unknown_source_provenance": 1},
        },
    )
    _write_json(
        ensemble_summary,
        {
            "reference_scope": "sampled",
            "comparison_dir": "compare_x",
            "runs": [
                {
                    "run": "candidate",
                    "particle_count": 2,
                    "class_match_ratio_vs_reference": 0.5,
                    "class_mismatch_count_vs_reference": 1,
                    "unresolved_crossing_count": 0,
                    "boundary_event_failure_count": 0,
                }
            ],
        },
    )
    _write_json(
        full_diagnostics_summary,
        {
            "reference_scope": "sampled",
            "comsol_reference_counts": {"full_particle_count": 2, "sampled_particle_count": 2},
            "final_state": {"matched_particle_count": 2},
            "wall_events": {"comsol_only_event_count": 0},
        },
    )

    assert main(
        [
            "--run-output-dir",
            str(run_dir),
            "--preflight-dir",
            str(preflight_dir),
            "--first-step-dir",
            str(first_step_dir),
            "--boundary-summary",
            str(boundary_summary),
            "--ensemble-summary",
            str(ensemble_summary),
            "--near-wall-nohit-summary",
            str(near_wall_summary),
            "--comsol-full-diagnostics-summary",
            str(full_diagnostics_summary),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads((output_dir / "current_residual_gap_summary.json").read_text(encoding="utf-8"))
    assert summary["reference_scope"] == "sampled"
    assert summary["import_preflight"]["input"]["passed"] is True
    assert summary["release"]["boundary_release_applied_count"] == 2
    assert summary["field_support"]["runtime_mixed_stencil_count"] == 1
    assert summary["first_step"]["speed_ratio"]["count"] == 2
    assert summary["force_contributions"]["component_acceleration_norm_mps2"]["drag"]["max"] == 2.0
    assert summary["wall_interactions"]["boundary_first_hit_compare"]["matched_first_hit_count"] == 2
    assert summary["near_wall_active_no_hit"]["summary"]["near_wall_nohit_count"] == 1
    assert summary["final_state"]["metric_scope"] == "final_snapshot"
    assert summary["full_comsol_diagnostics"]["comsol_reference_counts"]["full_particle_count"] == 2
    assert (output_dir / "current_residual_gap_report.md").exists()


def test_comsol_residual_gap_summary_reports_missing_optional_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "residual_gap"
    _write_json(run_dir / "solver_report.json", {"particle_count": 1, "released_count": 1})
    _write_csv(run_dir / "final_particles.csv", [{"particle_id": 1, "active": 1, "stuck": 0, "absorbed": 0, "escaped": 0}])

    assert main(["--run-output-dir", str(run_dir), "--output-dir", str(output_dir)]) == 0

    summary = json.loads((output_dir / "current_residual_gap_summary.json").read_text(encoding="utf-8"))
    assert summary["missing_required_artifacts"] == []
    assert "first_step_compare_summary" in summary["missing_optional_artifacts"]
    assert summary["first_step"]["available"] == 0
    assert summary["ensemble"]["available"] == 0


def test_comsol_residual_gap_summary_schema_is_stable(tmp_path: Path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "residual_gap"
    _write_json(run_dir / "solver_report.json", {"particle_count": 1, "released_count": 1})
    _write_csv(run_dir / "final_particles.csv", [{"particle_id": 1, "active": 1, "stuck": 0, "absorbed": 0, "escaped": 0}])

    assert main(["--run-output-dir", str(run_dir), "--output-dir", str(output_dir)]) == 0

    summary = json.loads((output_dir / "current_residual_gap_summary.json").read_text(encoding="utf-8"))
    expected_keys = {
        "summary_schema_version",
        "reference_scope",
        "metric_scope_notes",
        "artifact_status",
        "missing_optional_artifacts",
        "missing_required_artifacts",
        "import_preflight",
        "release",
        "field_support",
        "first_step",
        "force_contributions",
        "wall_interactions",
        "near_wall_active_no_hit",
        "final_state",
        "first_crossing_vacuum_time",
        "runtime_collision_counters",
        "ensemble",
        "full_comsol_diagnostics",
    }
    assert set(summary) == expected_keys
    assert summary["summary_schema_version"] == 1
