from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from particle_tracer_unified.compare.comsol_full_diagnostics import main


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_full_comsol_diagnostics_reads_synthetic_artifacts(tmp_path: Path) -> None:
    solver_dir = tmp_path / "solver"
    first_step_dir = tmp_path / "first_step"
    comsol_dir = tmp_path / "comsol"
    output_dir = tmp_path / "diagnostics"

    _write_json(
        solver_dir / "solver_report.json",
        {
            "particle_count": 3,
            "released_count": 3,
            "stuck_count": 1,
            "escaped_count": 1,
            "timing_s": {"solver_core_s": 0.25, "collision_resolution_s": 0.02},
            "unresolved_crossing_count": 0,
            "source_surface_release_skip_count": 1,
            "state_geometry_summary": {
                "near_boundary_threshold_m": 0.01,
                "by_state": {
                    "active_free_flight": {
                        "count": 1,
                        "near_boundary_count": 1,
                        "nearest_part_counts": [{"part_id": 10, "count": 1}],
                    }
                },
            },
        },
    )
    _write_json(solver_dir / "collision_diagnostics.json", {"primary_hit_count": 1, "valid_mask_mixed_stencil_count": 2})
    _write_csv(
        solver_dir / "final_particles.csv",
        [
            {"particle_id": 1, "x": 0.0, "y": 0.0, "active": 1, "stuck": 0, "absorbed": 0, "escaped": 0, "source_part_id": 10},
            {"particle_id": 2, "x": 1.0, "y": 0.0, "active": 0, "stuck": 1, "absorbed": 0, "escaped": 0, "source_part_id": 10},
            {"particle_id": 3, "x": 2.0, "y": 0.0, "active": 0, "stuck": 0, "absorbed": 0, "escaped": 1, "source_part_id": 20},
        ],
    )
    _write_csv(solver_dir / "wall_events.csv", [{"particle_id": 2, "hit_time_s": 0.2, "part_id": 10}])
    _write_csv(
        solver_dir / "source_particle_diagnostics.csv",
        [
            {"particle_id": 1, "boundary_release_applied": 1, "projection_distance_m": 1e-6, "source_provenance_group": "production_generated_source"},
            {"particle_id": 2, "boundary_release_applied": 1, "projection_distance_m": 2e-6, "source_provenance_group": "production_generated_source"},
            {"particle_id": 3, "boundary_release_applied": 0, "projection_distance_m": 0.0, "source_provenance_group": "known_source"},
        ],
    )
    _write_json(
        first_step_dir / "first_step_compare_summary.json",
        {
            "particle_count": 3,
            "compared_particle_count": 3,
            "stochastic_policy": "off",
            "position_error_m": {"count": 3, "mean": 1e-7, "max": 3e-7},
            "velocity_error_mps": {"count": 3, "mean": 0.01, "max": 0.03},
        },
    )
    _write_csv(
        first_step_dir / "first_step_error.csv",
        [
            {"particle_id": 1, "position_error_m": 1e-7, "velocity_error_mps": 0.01, "speed_ratio": 1.01},
            {"particle_id": 2, "position_error_m": 2e-7, "velocity_error_mps": 0.02, "speed_ratio": 0.99},
            {"particle_id": 3, "position_error_m": 3e-7, "velocity_error_mps": 0.03, "speed_ratio": 1.10},
        ],
    )
    trajectory = _write_csv(
        comsol_dir / "trajectory.csv",
        [
            {"particle_id": 1, "time_s": 0.0, "x": 0.0, "y": 0.0, "state": "active", "wall_hit": 0, "part_id": 0},
            {"particle_id": 1, "time_s": 1.0, "x": 0.2, "y": 0.0, "state": "active", "wall_hit": 1, "part_id": 11},
            {"particle_id": 2, "time_s": 0.0, "x": 1.0, "y": 0.0, "state": "active", "wall_hit": 0, "part_id": 0},
            {"particle_id": 2, "time_s": 1.0, "x": 1.0, "y": 0.0, "state": "stuck", "wall_hit": 1, "part_id": 10},
            {"particle_id": 3, "time_s": 0.0, "x": 2.0, "y": 0.0, "state": "active", "wall_hit": 0, "part_id": 0},
            {"particle_id": 3, "time_s": 1.0, "x": 2.0, "y": 0.0, "state": "active", "wall_hit": 0, "part_id": 0},
        ],
    )
    release = _write_csv(
        comsol_dir / "release.csv",
        [
            {"particle_id": 1, "source_part_id": 10},
            {"particle_id": 2, "source_part_id": 10},
            {"particle_id": 3, "source_part_id": 20},
        ],
    )

    assert main(
        [
            "--solver-output-dir",
            str(solver_dir),
            "--comsol-trajectory-csv",
            str(trajectory),
            "--comsol-release-csv",
            str(release),
            "--first-step-dir",
            str(first_step_dir),
            "--reference-scope",
            "sampled",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads((output_dir / "full_comsol_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["summary_schema_version"] == 1
    assert summary["reference_scope"] == "sampled"
    assert summary["comsol_reference_counts"]["full_particle_count"] == 3
    assert summary["comsol_reference_counts"]["sampled_particle_count"] == 3
    assert summary["final_state"]["solver"]["escaped_or_vacuum_fraction"] == 1 / 3
    assert summary["final_state"]["comsol"]["fractions"]["active"] == 2 / 3
    assert summary["preprocess"]["boundary_release_applied_ratio"] == 2 / 3
    assert summary["first_step"]["speed_ratio"]["max"] == 1.10
    assert summary["near_wall_active"]["active_near_boundary_count"] == 1
    assert summary["wall_events"]["zero_wallhit_fraction_solver"] == 2 / 3
    assert summary["wall_events"]["zero_wallhit_fraction_comsol"] == 1 / 3
    assert summary["wall_events"]["comsol_only_event_count"] == 1
    assert summary["runtime_collision_counters"]["counters"]["valid_mask_mixed_stencil_count"] == 2
    assert {item["source_part_id"] for item in summary["top_source_parts_for_residuals"]} == {10, 20}
    assert (output_dir / "suspicious_particles.csv").exists()


def test_full_comsol_diagnostics_reports_missing_optional_artifacts(tmp_path: Path) -> None:
    solver_dir = tmp_path / "solver"
    comsol_dir = tmp_path / "comsol"
    output_dir = tmp_path / "diagnostics"

    _write_json(solver_dir / "solver_report.json", {"particle_count": 1, "released_count": 1})
    _write_csv(
        solver_dir / "final_particles.csv",
        [{"particle_id": 1, "x": 0.0, "y": 0.0, "active": 1, "stuck": 0, "absorbed": 0, "escaped": 0}],
    )
    trajectory = _write_csv(
        comsol_dir / "trajectory.csv",
        [{"particle_id": 1, "time_s": 0.0, "x": 0.0, "y": 0.0, "state": "active"}],
    )
    release = _write_csv(comsol_dir / "release.csv", [{"particle_id": 1, "source_part_id": 0}])

    assert main(
        [
            "--solver-output-dir",
            str(solver_dir),
            "--comsol-trajectory-csv",
            str(trajectory),
            "--comsol-release-csv",
            str(release),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    summary = json.loads((output_dir / "full_comsol_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["missing_required_artifacts"] == []
    assert "source_particle_diagnostics" in summary["missing_optional_artifacts"]
    assert "first_step_error" in summary["missing_optional_artifacts"]
    assert summary["preprocess"]["available"] == 0
    assert summary["first_step"]["available"] == 0
    assert summary["wall_events"]["solver_events_available"] == 0
