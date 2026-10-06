from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tools.vv.comsol import evaluate_p18h_hold_freeze as evaluator
from tools.vv.comsol import run_p18h_hold_candidate as candidate

CONFIG = Path(__file__).parents[1] / "cases" / "p18h_hold_freeze_v1.json"


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _synthetic_config(tmp_path: Path) -> tuple[Path, Path]:
    payload: dict[str, Any] = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["expected_candidate"]["engine_algorithm_revision"] = "particle_engine_v37"
    payload["expected_candidate"]["result_algorithm_revision"] = "durable_segmented_result_v5"
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    compact_manifest = reference_root / "comparison_manifest.json"
    compact_manifest.write_text(
        json.dumps(
            {
                "evaluation_status": "COMPLETE",
                "scientific_status": "PASS",
                "gate_counts": {"pass": 1, "fail": 0},
            }
        ),
        encoding="utf-8",
    )
    compact_gates = reference_root / "gates.csv"
    compact_gates.write_text("gate,status\nlocked,PASS\n", encoding="utf-8")
    compact_readme = reference_root / "README.md"
    compact_readme.write_text("# locked synthetic reference\n", encoding="utf-8")
    state = reference_root / "state.csv"
    times = np.linspace(0.0, 1.5e-4, 61)
    state_rows: list[tuple[object, ...]] = []
    for index, time_s in enumerate(times):
        active = index < 30
        state_rows.append(
            (
                1,
                time_s,
                0.23927 + 10.0 * time_s if active else 0.24,
                0.115,
                10.0,
                0.0,
                1 if active else 2,
                2,
                7.3e-5,
            )
        )
    _write_csv(
        state,
        (
            "particle_id",
            "time_s",
            "r_m",
            "z_m",
            "velocity_r_m_per_s",
            "velocity_z_m_per_s",
            "current_status_code",
            "final_status_code",
            "stop_or_event_time_s",
        ),
        state_rows,
    )
    event = reference_root / "event_observation.csv"
    _write_csv(
        event,
        ("event_time_s", "hit_r_m", "hit_z_m", "wall_condition"),
        [(7.3e-5, 0.24, 0.115, "Freeze")],
    )
    summary = reference_root / "step_summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    paths = {
        "compact_manifest": compact_manifest,
        "compact_gates": compact_gates,
        "compact_readme": compact_readme,
        "state": state,
        "event": event,
        "summary": summary,
    }
    reference = payload["reference"]
    for name, path in paths.items():
        reference[f"{name}_relative_path"] = str(path.resolve())
        reference[f"{name}_sha256"] = candidate._sha256(path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path, state


def test_locked_configuration_selects_one_force_free_rz_hold_probe() -> None:
    payload = candidate._load_config(CONFIG)

    assert payload["case"]["coordinate_system"] == "axisymmetric_rz"
    assert payload["case"]["particle_id"] == 1
    assert payload["case"]["output_frames"] == 61
    assert payload["expected_candidate"] == {
        "case_schema_version": 2,
        "result_schema_version": 2,
        "engine_algorithm_revision": "particle_engine_v32",
        "boundary_algorithm_revision": "point_wall_laws_v5",
        "result_algorithm_revision": "durable_segmented_result_v4",
        "boundary_law": "hold",
        "boundary_outcome": "held",
        "lifecycle_code": 5,
    }


def test_runs_and_evaluates_hold_against_locked_freeze_shape(tmp_path: Path) -> None:
    config, _state = _synthetic_config(tmp_path)
    candidate_root = tmp_path / "candidate"
    evidence_root = tmp_path / "evidence"

    run_report = candidate.run(config, candidate_root)
    comparison = evaluator.evaluate(config, candidate_root, evidence_root)

    assert run_report["status"] == "COMPLETE"
    assert run_report["event_rows"] == 1
    assert run_report["failure_event_count"] == 0
    assert run_report["final_lifecycle"] == 5
    assert comparison["status"] == "PASS"
    assert comparison["gate_summary"] == {"pass": 15, "fail": 0}
    metrics = comparison["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["active_frames"] == 30
    assert metrics["held_frames"] == 31

    trajectory = candidate_root / candidate.TRAJECTORY_FILENAME
    trajectory.write_text(trajectory.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate trajectory_sha256 differs"):
        evaluator.evaluate(config, candidate_root, tmp_path / "tampered-evidence")


def test_rejects_changed_locked_reference_before_running_solver(tmp_path: Path) -> None:
    config, state = _synthetic_config(tmp_path)
    state.write_text(state.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    output = tmp_path / "candidate"

    with pytest.raises(ValueError, match="locked state hash differs"):
        candidate.run(config, output)

    assert not output.exists()
