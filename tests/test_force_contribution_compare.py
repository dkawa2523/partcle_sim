from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from particle_tracer_unified.solvers.forces import ForceContribution, build_force_catalog
from particle_tracer_unified.solvers.runtime_outputs import _build_force_contribution_rows


def test_force_contribution_row_contains_component_norms() -> None:
    contribution = ForceContribution(
        name="electric",
        acceleration=np.asarray([1.0, -2.0], dtype=float),
        force=np.asarray([3.0, 4.0], dtype=float),
        physical_quantity="acceleration",
        metadata={"comsol_feature": "fpt.force1"},
    )

    row = contribution.as_row()

    assert row["name"] == "electric"
    assert row["accel_norm"] == np.sqrt(5.0)
    assert row["force_norm"] == 5.0
    assert row["metadata_comsol_feature"] == "fpt.force1"


def test_force_contribution_rows_label_added_mass_terms_as_acceleration() -> None:
    catalog = build_force_catalog(
        {
            "solver": {
                "forces": {
                    "pressure_gradient": {"enabled": False},
                    "virtual_mass": {"enabled": False},
                }
            }
        }
    )
    payload = SimpleNamespace(prepared=SimpleNamespace(runtime=SimpleNamespace(force_catalog=catalog)))

    rows = {row["name"]: row for row in _build_force_contribution_rows(payload)}

    assert rows["pressure_gradient"]["physical_quantity"] == "acceleration"
    assert rows["virtual_mass"]["physical_quantity"] == "acceleration"


def test_compare_modules_expose_help() -> None:
    modules = [
        "particle_tracer_unified.compare.field_compare",
        "particle_tracer_unified.compare.acceleration_compare",
        "particle_tracer_unified.compare.trajectory_compare",
        "particle_tracer_unified.compare.boundary_compare",
    ]
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
        assert "usage:" in completed.stdout.lower()


def test_boundary_compare_uses_first_hit_and_writes_summary(tmp_path: Path) -> None:
    python_csv = tmp_path / "python_wall_events.csv"
    comsol_csv = tmp_path / "comsol_wall_events.csv"
    out_csv = tmp_path / "boundary_hit_comparison.csv"
    summary_json = tmp_path / "boundary_hit_comparison.json"
    pd.DataFrame(
        [
            {"particle_id": 1, "hit_time_s": 0.2, "part_id": 20, "outcome": "late", "hit_x_m": 1.0},
            {"particle_id": 1, "hit_time_s": 0.1, "part_id": 10, "outcome": "bounce", "hit_x_m": 0.5},
            {"particle_id": 2, "hit_time_s": 0.3, "part_id": 30, "outcome": "stick", "hit_x_m": 2.0},
        ]
    ).to_csv(python_csv, index=False)
    pd.DataFrame(
        [
            {"particle_id": 1, "hit_time_s": 0.11, "comsol_entity_id": 10, "outcome": "bounce", "hit_x": 0.55},
            {"particle_id": 2, "hit_time_s": 0.31, "comsol_entity_id": 31, "outcome": "freeze", "hit_x": 2.0},
        ]
    ).to_csv(comsol_csv, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "particle_tracer_unified.compare.boundary_compare",
            "--python",
            str(python_csv),
            "--comsol",
            str(comsol_csv),
            "--output",
            str(out_csv),
            "--summary",
            str(summary_json),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    rows = pd.read_csv(out_csv).sort_values("particle_id")
    assert rows.loc[rows["particle_id"] == 1, "hit_time_s_python"].iloc[0] == 0.1
    assert rows.loc[rows["particle_id"] == 1, "part_id_match"].iloc[0]
    assert not rows.loc[rows["particle_id"] == 2, "part_id_match"].iloc[0]
    assert summary_json.exists()
