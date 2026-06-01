from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.collect_run_summaries import collect_run_summaries, collect_run_summary, collect_shard_root_artifacts
from tools.validate_comparison_artifacts import validate_artifacts
from tools.visualization_common import write_run_summary


def _write_scalar_summary(path: Path, rows: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value", "unit"])
        for key, value in rows.items():
            writer.writerow([key, value, ""])


def test_collect_run_summary_flattens_solver_plasma_and_charge_outputs(tmp_path: Path) -> None:
    out = tmp_path / "run_a"
    out.mkdir()
    (out / "solver_report.json").write_text(
        json.dumps(
            {
                "particle_count": 4,
                "released_count": 4,
                "coordinate_system": "axisymmetric_rz",
                "integrator": "etd2",
                "valid_mask_policy": "retry_then_stop",
                "drag_model": "epstein",
                "acceleration_source": "particle_charge_electric_field",
                "final_state_counts": {
                    "active_free_flight": 3,
                    "stuck": 1,
                    "invalid_mask_stopped": 0,
                    "numerical_boundary_stopped": 0,
                },
                "timing_s": {
                    "solver_core_s": 1.25,
                    "step_loop_s": 1.2,
                    "freeflight_s": 0.8,
                    "charge_model_s": 0.05,
                },
                "memory_estimate_bytes": {"estimated_numpy_bytes": 123456},
                "drag_gas_properties": {
                    "density_source": "field:rho_g",
                    "temperature_source": "field:T",
                    "fallback_density_kgm3": 2.0e-5,
                    "fallback_temperature_K": 320.0,
                },
                "boundary_event_contract_passed": 1,
                "unresolved_crossing_count": 0,
                "max_hits_reached_count": 0,
            }
        ),
        encoding="utf-8",
    )
    _write_scalar_summary(
        out / "plasma_background_summary.csv",
        {
            "source": "saas_constant",
            "electron_density_m3": "1e16",
            "debye_length_m": "1.2e-5",
        },
    )
    _write_scalar_summary(
        out / "charge_model_summary.csv",
        {
            "enabled": "1",
            "mode": "finite_rate_flux_balance",
            "final_mean_charge_e": "-14.5",
        },
    )

    row = collect_run_summary(out)

    assert row["status"] == "pass"
    assert row["particle_count"] == 4
    assert row["stuck"] == 1
    assert row["drag_density_source"] == "field:rho_g"
    assert row["plasma_source"] == "saas_constant"
    assert row["charge_mode"] == "finite_rate_flux_balance"
    assert row["charge_final_mean_charge_e"] == "-14.5"


def test_collect_run_summaries_writes_one_row_per_output_dir(tmp_path: Path) -> None:
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    out_a.mkdir()
    out_b.mkdir()
    for out, stopped in ((out_a, 0), (out_b, 2)):
        (out / "solver_report.json").write_text(
            json.dumps(
                {
                    "particle_count": 2,
                    "final_state_counts": {"active_free_flight": 2 - stopped, "invalid_mask_stopped": stopped},
                    "boundary_event_contract_passed": 1,
                    "unresolved_crossing_count": 0,
                    "max_hits_reached_count": 0,
                }
            ),
            encoding="utf-8",
        )

    csv_path = collect_run_summaries([out_a, out_b], tmp_path / "summary.csv")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["run_name"] for row in rows] == ["run_a", "run_b"]
    assert [row["status"] for row in rows] == ["pass", "review"]


def test_run_summary_lists_optional_compact_summary_files(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    summary_path = write_run_summary(
        out,
        {
            "health_summary": {"status": "pass", "particle_count": 1, "released_count": 1},
            "summary_files": {
                "charge_model_summary.csv": str(out / "charge_model_summary.csv"),
                "plasma_background_summary.csv": str(out / "plasma_background_summary.csv"),
            },
            "modules": {},
        },
    )

    text = summary_path.read_text(encoding="utf-8")
    assert "## Compact Summary Files" in text
    assert "charge_model_summary.csv" in text
    assert "plasma_background_summary.csv" in text


def test_collect_shard_root_artifacts_writes_root_outputs(tmp_path: Path) -> None:
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    root = tmp_path / "root"
    for shard, particle_count, skip_count in ((shard_a, 2, 1), (shard_b, 3, 2)):
        shard.mkdir()
        (shard / "solver_report.json").write_text(
            json.dumps(
                {
                    "particle_count": particle_count,
                    "released_count": particle_count,
                    "coordinate_system": "cartesian_xy",
                    "axis_names": ["x", "y"],
                    "final_state_counts": {"active_free_flight": particle_count, "invalid_mask_stopped": 0},
                    "boundary_event_contract_passed": 1,
                    "unresolved_crossing_count": 0,
                    "max_hits_reached_count": 0,
                    "source_surface_release_skip_count": skip_count,
                }
            ),
            encoding="utf-8",
        )
        (shard / "prepared_runtime_summary.json").write_text(
            json.dumps(
                {
                    "coordinate_system": "cartesian_xy",
                    "axis_names": ["x", "y"],
                    "spatial_dim": 2,
                    "particles": particle_count,
                    "source_model_summary": {
                        "released_count": particle_count,
                        "boundary_release_applied_count": particle_count,
                    },
                }
            ),
            encoding="utf-8",
        )
        with (shard / "wall_summary_by_part.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["part_id", "outcome", "wall_mode", "count"])
            writer.writeheader()
            writer.writerow({"part_id": "10", "outcome": "stuck", "wall_mode": "stick", "count": particle_count})
        (shard / "source_model_summary.json").write_text(
            json.dumps(
                {
                    "released_count": particle_count,
                    "boundary_release_applied_count": particle_count,
                    "source_provenance_counts": {"known": particle_count},
                }
            ),
            encoding="utf-8",
        )
        (shard / "first_step_compare_summary.json").write_text(
            json.dumps({"particle_count": particle_count, "compared_particle_count": particle_count, "stochastic_policy": "off"}),
            encoding="utf-8",
        )
        (shard / "collision_diagnostics.json").write_text(
            json.dumps(
                {
                    "source_surface_release_skip_count": skip_count,
                    "source_surface_release_skip_blocked_count": 0,
                    "source_surface_release_skip_blocked_reasons": {},
                }
            ),
            encoding="utf-8",
        )
        with (shard / "source_particle_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["particle_id", "projection_distance_m"])
            writer.writeheader()
            writer.writerow({"particle_id": particle_count, "projection_distance_m": "0.0"})
    (shard_a / "comparison_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    summary_csv, manifest_path = collect_shard_root_artifacts([shard_a, shard_b], root)

    assert summary_csv == root.resolve() / "run_summary_compare.csv"
    assert manifest_path == root.resolve() / "shard_artifacts_manifest.json"
    assert summary_csv.exists()
    assert (root / "solver_report.json").exists()
    assert (root / "prepared_runtime_summary.json").exists()
    assert (root / "wall_summary_by_part.csv").exists()
    assert (root / "source_model_summary.json").exists()
    assert (root / "first_step_compare_summary.json").exists()
    assert (root / "collision_diagnostics.json").exists()
    assert (root / "source_particle_diagnostics.csv").exists()

    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert [row["run_name"] for row in summary_rows] == ["shard_a", "shard_b"]

    with (root / "source_particle_diagnostics.csv").open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    assert [row["shard_name"] for row in source_rows] == ["shard_a", "shard_b"]
    assert {row["particle_id"] for row in source_rows} == {"2", "3"}

    solver_report = json.loads((root / "solver_report.json").read_text(encoding="utf-8"))
    assert solver_report["source_kind"] == "sharded_root_aggregate"
    assert solver_report["shard_count"] == 2
    assert solver_report["particle_count"] == 5
    assert solver_report["released_count"] == 5
    assert solver_report["final_state_counts"]["active_free_flight"] == 5
    assert solver_report["source_surface_release_skip_count"] == 3

    wall_rows = list(csv.DictReader((root / "wall_summary_by_part.csv").open("r", encoding="utf-8", newline="")))
    assert wall_rows == [{"part_id": "10", "outcome": "stuck", "wall_mode": "stick", "count": "5"}]

    source_summary = json.loads((root / "source_model_summary.json").read_text(encoding="utf-8"))
    assert source_summary["boundary_release_applied_count"] == 5
    assert source_summary["source_provenance_counts"] == {"known": 5}

    first_step_summary = json.loads((root / "first_step_compare_summary.json").read_text(encoding="utf-8"))
    assert first_step_summary["particle_count"] == 5

    collision_summary = json.loads((root / "collision_diagnostics.json").read_text(encoding="utf-8"))
    assert collision_summary["source_surface_release_skip_count"] == 3

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["shard_count"] == 2
    assert manifest["source_particle_diagnostics_rows"] == 2
    assert manifest["generated_artifacts"]["run_summary_compare.csv"] == str(summary_csv)
    assert manifest["generated_artifacts"]["solver_report.json"] == str((root / "solver_report.json").resolve())
    assert str((shard_a / "comparison_summary.json").resolve()) in manifest["comparison_summary_paths"]
    assert manifest["shards"][0]["artifacts"]["source_particle_diagnostics.csv"]["exists"]
    assert not (root / "comparison_summary.json").exists()

    validation = validate_artifacts(
        root,
        workflow="sharded",
        require_first_step=True,
        require_debug=True,
        require_source_diagnostics=True,
    )
    assert validation["status"] == "pass"
