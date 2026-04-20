from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.collect_run_summaries import collect_run_summaries, collect_run_summary
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
