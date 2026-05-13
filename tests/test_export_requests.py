from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "comsol_particle_export"
sys.path.insert(0, str(EXTERNAL))

from comsol_particle_export.export_requests import write_reextract_request_bundle  # noqa: E402


def test_reextract_bundle_skips_existing_canonical_trajectory_and_cleans_stale_files(tmp_path: Path) -> None:
    trajectory = tmp_path / "comsol_particle_trajectory.csv"
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "time_s": 0.0,
                "x": 1.0e-3,
                "y": 2.0e-3,
                "v_x": 0.1,
                "v_y": 0.2,
            }
        ]
    ).to_csv(trajectory, index=False)

    out_dir = tmp_path / "requests"
    out_dir.mkdir()
    stale_config = out_dir / "particle_trajectory_xy_velocity_config.json"
    stale_config.write_text("{}", encoding="utf-8")
    stale_probe_dir = out_dir / "wall_event_expression_probes"
    stale_probe_dir.mkdir()
    (stale_probe_dir / "probe.json").write_text("{}", encoding="utf-8")

    summary = write_reextract_request_bundle(
        case_name="case",
        field_manifest={
            "coordinate_scale_m_per_model_unit": 0.001,
            "coordinate_model_unit": "mm",
            "dataset": "dset1",
            "mesh_tag": "mesh1",
        },
        particle_manifest={"data_export_dataset": "part1"},
        trajectory_report={"trajectory_time_count": 1, "time_min_s": 0.0, "time_max_s": 0.0},
        out_dir=out_dir,
        trajectory_csv=trajectory,
    )

    written = json.loads((out_dir / "reextract_request_summary.json").read_text(encoding="utf-8"))
    assert summary["request_count"] == 0
    assert written["runnable_config_count"] == 0
    assert not stale_config.exists()
    assert not stale_probe_dir.exists()
    assert (out_dir / "run_reextract_requests.ps1").exists()
