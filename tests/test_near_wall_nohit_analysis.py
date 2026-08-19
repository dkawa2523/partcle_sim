from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from particle_tracer_unified.compare.near_wall_nohit import NEAR_WALL_NOHIT_COLUMNS
from particle_tracer_unified.compare.near_wall_nohit import main as near_wall_nohit_main


def _write_final_particles(
    output_dir: Path, *, distance_m: float, particle_id: int = 1
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "schema_version": 2,
                "particle_id": int(particle_id),
                "release_time_s": 0.0,
                "final_state": "active_free_flight",
                "source_part_id": 7,
                "x_m": float(distance_m),
                "y_m": 0.5,
                "vx_mps": -1.0,
                "vy_mps": 0.0,
                "nearest_boundary_part_id": 7,
                "nearest_boundary_distance_m": float(distance_m),
                "sdf_m": -float(distance_m),
                "inside_geometry": 1,
                "contact_normal_x": 1.0,
                "contact_normal_y": 0.0,
            }
        ]
    ).to_csv(output_dir / "final_particles.csv", index=False)


def _write_empty_wall_events(output_dir: Path) -> None:
    pd.DataFrame(columns=["particle_id", "part_id", "hit_time_s", "outcome"]).to_csv(
        output_dir / "wall_events.csv",
        index=False,
    )


def test_near_wall_nohit_flags_tiny_synthetic_active_particle(tmp_path: Path) -> None:
    output = tmp_path / "run"
    analysis = tmp_path / "analysis"
    _write_final_particles(output, distance_m=1.0e-7)
    _write_empty_wall_events(output)

    rc = near_wall_nohit_main(
        [
            "--output-dir",
            str(output),
            "--analysis-output-dir",
            str(analysis),
            "--threshold-m",
            "1e-6",
        ]
    )

    assert rc == 0
    rows = pd.read_csv(analysis / "near_wall_nohit_particles.csv")
    summary = json.loads(
        (analysis / "near_wall_nohit_summary.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 1
    assert int(rows.loc[0, "particle_id"]) == 1
    assert rows.loc[0, "classification"] == "no_segment_crossing_recorded"
    assert summary["suspicious_particle_count"] == 1


def test_near_wall_nohit_ignores_far_from_wall_active_particle(tmp_path: Path) -> None:
    output = tmp_path / "run"
    analysis = tmp_path / "analysis"
    _write_final_particles(output, distance_m=0.1)
    _write_empty_wall_events(output)

    rc = near_wall_nohit_main(
        [
            "--output-dir",
            str(output),
            "--analysis-output-dir",
            str(analysis),
            "--threshold-m",
            "1e-6",
        ]
    )

    assert rc == 0
    rows = pd.read_csv(analysis / "near_wall_nohit_particles.csv")
    summary = json.loads(
        (analysis / "near_wall_nohit_summary.json").read_text(encoding="utf-8")
    )
    assert rows.empty
    assert summary["near_wall_active_count"] == 0
    assert summary["suspicious_particle_count"] == 0


def test_near_wall_nohit_handles_missing_wall_events(tmp_path: Path) -> None:
    output = tmp_path / "run"
    analysis = tmp_path / "analysis"
    _write_final_particles(output, distance_m=1.0e-7)

    rc = near_wall_nohit_main(
        [
            "--output-dir",
            str(output),
            "--analysis-output-dir",
            str(analysis),
            "--threshold-m",
            "1e-6",
        ]
    )

    assert rc == 0
    rows = pd.read_csv(analysis / "near_wall_nohit_particles.csv")
    summary = json.loads(
        (analysis / "near_wall_nohit_summary.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 1
    assert rows.loc[0, "classification"] == "no_wall_events_available"
    assert int(rows.loc[0, "wall_events_available"]) == 0
    assert summary["wall_events_available"] == 0


def test_near_wall_nohit_output_schema_is_stable(tmp_path: Path) -> None:
    output = tmp_path / "run"
    analysis = tmp_path / "analysis"
    _write_final_particles(output, distance_m=0.1)
    _write_empty_wall_events(output)

    rc = near_wall_nohit_main(
        [
            "--output-dir",
            str(output),
            "--analysis-output-dir",
            str(analysis),
            "--threshold-m",
            "1e-6",
        ]
    )

    assert rc == 0
    rows = pd.read_csv(analysis / "near_wall_nohit_particles.csv")
    assert list(rows.columns) == NEAR_WALL_NOHIT_COLUMNS


def test_near_wall_analysis_cannot_modify_the_immutable_run_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_final_particles(output, distance_m=0.1)

    with pytest.raises(ValueError, match="must be separate"):
        near_wall_nohit_main(
            [
                "--output-dir",
                str(output),
                "--analysis-output-dir",
                str(output / "analysis"),
                "--threshold-m",
                "1e-6",
            ]
        )


def test_near_wall_analysis_uses_resolved_boundary_policy(tmp_path: Path) -> None:
    output = tmp_path / "run"
    analysis = tmp_path / "analysis"
    _write_final_particles(output, distance_m=5.0e-8)
    _write_empty_wall_events(output)
    (output / "run_summary.json").write_text(
        json.dumps(
            {
                "execution": {
                    "numerics": {
                        "boundary": {
                            "classification_tolerance_m": 1.0e-9,
                            "contact_offset_m": 1.0e-7,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    rc = near_wall_nohit_main(
        ["--output-dir", str(output), "--analysis-output-dir", str(analysis)]
    )

    assert rc == 0
    summary = json.loads(
        (analysis / "near_wall_nohit_summary.json").read_text(encoding="utf-8")
    )
    assert summary["threshold_m"] == pytest.approx(1.0e-7)


def test_near_wall_analysis_has_no_fixed_threshold_fallback(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _write_final_particles(output, distance_m=5.0e-8)

    with pytest.raises(ValueError, match="--threshold-m is required"):
        near_wall_nohit_main(
            [
                "--output-dir",
                str(output),
                "--analysis-output-dir",
                str(tmp_path / "analysis"),
            ]
        )
