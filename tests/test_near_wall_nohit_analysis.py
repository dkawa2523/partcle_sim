from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from particle_tracer_unified.compare.near_wall_nohit import NEAR_WALL_NOHIT_COLUMNS, main as near_wall_nohit_main


def _write_final_particles(output_dir: Path, *, distance_m: float, particle_id: int = 1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "particle_id": int(particle_id),
                "release_time": 0.0,
                "released": 1,
                "active": 1,
                "stuck": 0,
                "absorbed": 0,
                "contact_sliding": 0,
                "contact_endpoint_stopped": 0,
                "escaped": 0,
                "invalid_mask_stopped": 0,
                "numerical_boundary_stopped": 0,
                "source_part_id": 7,
                "source_provenance_group": "known_source",
                "x": float(distance_m),
                "y": 0.5,
                "v_x": -1.0,
                "v_y": 0.0,
                "nearest_boundary_part_id": 7,
                "nearest_boundary_distance_m": float(distance_m),
                "sdf_m": -float(distance_m),
                "inside_geometry": 1,
                "normal_x": 1.0,
                "normal_y": 0.0,
            }
        ]
    ).to_csv(output_dir / "final_particles.csv", index=False)


def _write_empty_wall_events(output_dir: Path) -> None:
    pd.DataFrame(columns=["particle_id", "part_id", "hit_time_s", "outcome"]).to_csv(
        output_dir / "wall_events.csv",
        index=False,
    )


def test_near_wall_nohit_flags_tiny_synthetic_active_particle(tmp_path: Path) -> None:
    _write_final_particles(tmp_path, distance_m=1.0e-7)
    _write_empty_wall_events(tmp_path)

    rc = near_wall_nohit_main(["--output-dir", str(tmp_path), "--threshold-m", "1e-6"])

    assert rc == 0
    rows = pd.read_csv(tmp_path / "near_wall_nohit_particles.csv")
    summary = json.loads((tmp_path / "near_wall_nohit_summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert int(rows.loc[0, "particle_id"]) == 1
    assert rows.loc[0, "classification"] == "no_segment_crossing_recorded"
    assert summary["suspicious_particle_count"] == 1


def test_near_wall_nohit_ignores_far_from_wall_active_particle(tmp_path: Path) -> None:
    _write_final_particles(tmp_path, distance_m=0.1)
    _write_empty_wall_events(tmp_path)

    rc = near_wall_nohit_main(["--output-dir", str(tmp_path), "--threshold-m", "1e-6"])

    assert rc == 0
    rows = pd.read_csv(tmp_path / "near_wall_nohit_particles.csv")
    summary = json.loads((tmp_path / "near_wall_nohit_summary.json").read_text(encoding="utf-8"))
    assert rows.empty
    assert summary["near_wall_active_count"] == 0
    assert summary["suspicious_particle_count"] == 0


def test_near_wall_nohit_handles_missing_wall_events(tmp_path: Path) -> None:
    _write_final_particles(tmp_path, distance_m=1.0e-7)

    rc = near_wall_nohit_main(["--output-dir", str(tmp_path), "--threshold-m", "1e-6"])

    assert rc == 0
    rows = pd.read_csv(tmp_path / "near_wall_nohit_particles.csv")
    summary = json.loads((tmp_path / "near_wall_nohit_summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows.loc[0, "classification"] == "no_wall_events_available"
    assert int(rows.loc[0, "wall_events_available"]) == 0
    assert summary["wall_events_available"] == 0


def test_near_wall_nohit_output_schema_is_stable(tmp_path: Path) -> None:
    _write_final_particles(tmp_path, distance_m=0.1)
    _write_empty_wall_events(tmp_path)

    rc = near_wall_nohit_main(["--output-dir", str(tmp_path), "--threshold-m", "1e-6"])

    assert rc == 0
    rows = pd.read_csv(tmp_path / "near_wall_nohit_particles.csv")
    assert list(rows.columns) == NEAR_WALL_NOHIT_COLUMNS
