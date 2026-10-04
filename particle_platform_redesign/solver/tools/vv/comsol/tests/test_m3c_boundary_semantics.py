from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest
from tools.vv.comsol import normalize_m3c_boundary_semantics as boundary


def _config(*, mappings: bool = False) -> dict[str, Any]:
    steps: object = [
        {"label": label, "seconds": seconds}
        for label, seconds in boundary.EXPECTED_STEP_SECONDS.items()
    ]
    scenarios: object = [
        {
            "id": "freeze_inlet_37",
            "boundary_feature": "outin",
            "boundary_id": 37,
            "wall_condition": "Freeze",
            "source_position_m": [0.23927, 0.115],
            "source_velocity_m_s": [10.0, 0.0],
            "expected_status_code": 2,
            "analytic_event_time_s": 7.3e-5,
            "analytic_hit_position_m": [0.24, 0.115],
        },
        {
            "id": "disappear_pump_35",
            "boundary_feature": "outpump",
            "boundary_id": 35,
            "wall_condition": "Disappear",
            "source_position_m": [0.21, 0.00073],
            "source_velocity_m_s": [0.0, -10.0],
            "expected_status_code": 4,
            "analytic_event_time_s": 7.3e-5,
            "analytic_hit_position_m": [0.21, 0.0],
        },
    ]
    if mappings:
        steps = {
            item["label"]: item["seconds"]
            for item in steps  # type: ignore[union-attr]
        }
        scenarios = {
            item["id"]: {key: value for key, value in item.items() if key != "id"}
            for item in scenarios  # type: ignore[union-attr]
        }
    return {
        "schema_version": 1,
        "evaluation_id": "M3-C0-boundary-semantics-test",
        "evaluation_revision": 2,
        "source_model": {
            "staged_filename": "source_copy.mph",
            "saved": False,
        },
        "case": {
            "particle_count_per_scenario": 1,
            "output_interval_s": 2.5e-6,
            "output_frames": 61,
            "time_end_s": 1.5e-4,
            "fixed_rk4_steps": steps,
        },
        "numerics": {"integrator": "classical_rk4"},
        "physics": {"force_free": True, "dynamic_charge_active": False},
        "scenarios": scenarios,
        "raw_export": {
            "revision": 1,
            "table": "state_raw_wide.csv",
            "columns_per_frame": 9,
            "columns": list(boundary.RAW_COLUMNS),
        },
        "acceptance": {
            "maximum_event_time_absolute_error_s": 5e-9,
            "maximum_event_time_step_spread_s": 5e-9,
            "maximum_hit_position_absolute_error_m": 5e-8,
            "maximum_freeze_post_event_position_spread_m": 5e-12,
            "maximum_active_position_absolute_error_m": 1e-12,
            "maximum_active_velocity_absolute_error_m_per_s": 1e-12,
            "require_expected_final_status": True,
            "require_single_terminal_event": True,
        },
    }


def _scenario(config: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    configured = config["scenarios"]
    if isinstance(configured, dict):
        return {"id": scenario_id, **configured[scenario_id]}
    return next(item for item in configured if item["id"] == scenario_id)


def _write_process_log(root: Path, config: dict[str, Any]) -> None:
    lines = ["M3CB|run_start|scenario_count=2"]
    for scenario_id in ("freeze_inlet_37", "disappear_pump_35"):
        scenario = _scenario(config, scenario_id)
        for seconds in boundary.EXPECTED_STEP_SECONDS.values():
            fields = {
                "scenario": scenario_id,
                "boundary_feature": scenario["boundary_feature"],
                "boundary_id": scenario["boundary_id"],
                "wall_condition": scenario["wall_condition"],
                "expected_status": scenario["expected_status_code"],
                "step_s": f"{seconds:g}",
                "force_free": "true",
                "dynamic_charge_active": "false",
                "integrator": "classical_rk4",
                "output_times": 61,
                "particle_rows": 1,
                "source_model": "source_copy.mph",
                "model_saved": "false",
            }
            lines.append(
                boundary.RECEIPT_PREFIX
                + "|".join(f"{key}={value}" for key, value in fields.items())
            )
    lines.append("M3CB|run_pass|model_saved=false")
    (root / "comsol_process.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record(
    scenario: dict[str, Any],
    frame: int,
    event_time_s: float,
    *,
    freeze_velocity: tuple[float, float],
    freeze_drift_m: float,
    disappear_finite_after_event: bool,
    time_offset_s: float = 0.0,
    active_position_error_m: float = 0.0,
    active_velocity_error_m_per_s: float = 0.0,
) -> list[float]:
    time_s = frame * boundary.EXPECTED_OUTPUT_INTERVAL_S + time_offset_s
    terminal = time_s >= 7.5e-5
    source_r, source_z = scenario["source_position_m"]
    source_vr, source_vz = scenario["source_velocity_m_s"]
    r_m = source_r + source_vr * time_s
    z_m = source_z + source_vz * time_s
    velocity_r = source_vr
    velocity_z = source_vz
    status = boundary.ACTIVE_STATUS_CODE
    if not terminal and frame == 10:
        r_m += active_position_error_m
        velocity_r += active_velocity_error_m_per_s
    if terminal:
        status = scenario["expected_status_code"]
        if scenario["wall_condition"] == "Freeze":
            terminal_index = frame - 30
            r_m = scenario["analytic_hit_position_m"][0] + terminal_index * freeze_drift_m
            z_m = scenario["analytic_hit_position_m"][1]
            velocity_r, velocity_z = freeze_velocity
        elif not disappear_finite_after_event:
            r_m = z_m = velocity_r = velocity_z = math.nan
    return [
        1.0,
        time_s,
        r_m,
        z_m,
        velocity_r,
        velocity_z,
        float(status),
        float(scenario["expected_status_code"]),
        event_time_s,
    ]


def _write_wide(
    path: Path,
    scenario: dict[str, Any],
    event_time_s: float,
    *,
    freeze_velocity: tuple[float, float],
    freeze_drift_m: float = 0.0,
    disappear_finite_after_event: bool = False,
    bad_grid: bool = False,
    duplicate_particle: bool = False,
    active_position_error_m: float = 0.0,
    active_velocity_error_m_per_s: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True)
    records = [
        _record(
            scenario,
            frame,
            event_time_s,
            freeze_velocity=freeze_velocity,
            freeze_drift_m=freeze_drift_m,
            disappear_finite_after_event=disappear_finite_after_event,
            time_offset_s=1.0e-7 if bad_grid and frame == 10 else 0.0,
            active_position_error_m=active_position_error_m,
            active_velocity_error_m_per_s=active_velocity_error_m_per_s,
        )
        for frame in range(boundary.EXPECTED_FRAMES)
    ]
    row = [value for record in records for value in record]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["% synthetic M3-C boundary export"])
        writer.writerow(row)
        if duplicate_particle:
            writer.writerow(row)


def _fixture(
    root: Path,
    *,
    mappings: bool = False,
    event_offsets_s: dict[str, float] | None = None,
    freeze_drift_m: float = 0.0,
    disappear_finite_after_event: bool = False,
    bad_grid: bool = False,
    duplicate_particle: bool = False,
    active_position_error_m: float = 0.0,
    active_velocity_error_m_per_s: float = 0.0,
) -> tuple[Path, Path]:
    root.mkdir()
    config = _config(mappings=mappings)
    offsets = event_offsets_s or dict.fromkeys(boundary.EXPECTED_STEP_SECONDS, 0.0)
    freeze_velocities = {
        "dt_10us": (10.0, 0.0),
        "dt_5us": (0.0, 0.0),
        "dt_2p5us": (3.0, 4.0),
    }
    for scenario_id in ("freeze_inlet_37", "disappear_pump_35"):
        scenario = _scenario(config, scenario_id)
        for step_label in boundary.EXPECTED_STEP_SECONDS:
            _write_wide(
                root / scenario_id / step_label / "state_raw_wide.csv",
                scenario,
                scenario["analytic_event_time_s"] + offsets[step_label],
                freeze_velocity=freeze_velocities[step_label],
                freeze_drift_m=freeze_drift_m,
                disappear_finite_after_event=disappear_finite_after_event,
                bad_grid=bad_grid and scenario_id == "freeze_inlet_37" and step_label == "dt_10us",
                duplicate_particle=(
                    duplicate_particle
                    and scenario_id == "freeze_inlet_37"
                    and step_label == "dt_10us"
                ),
                active_position_error_m=(
                    active_position_error_m
                    if scenario_id == "freeze_inlet_37" and step_label == "dt_10us"
                    else 0.0
                ),
                active_velocity_error_m_per_s=(
                    active_velocity_error_m_per_s
                    if scenario_id == "freeze_inlet_37" and step_label == "dt_10us"
                    else 0.0
                ),
            )
    _write_process_log(root, config)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return root, config_path


def _first_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def test_normalizes_six_runs_and_characterizes_velocity(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path / "probe",
        event_offsets_s={"dt_10us": 2e-10, "dt_5us": -1e-10, "dt_2p5us": 0.0},
    )

    report = boundary.normalize(root, config)

    assert report["status"] == "PASS"
    assert report["gate_counts"]["fail"] == 0
    assert report["gate_counts"]["characterized_not_gated"] == 6
    assert report["configuration_receipts"]["validated_count"] == 6
    freeze_runs = report["scenarios"]["freeze_inlet_37"]["runs"]
    assert freeze_runs["dt_10us"]["active_force_free_path"]["active_frame_count"] == 30
    assert freeze_runs["dt_10us"]["active_force_free_path"][
        "maximum_position_absolute_error_m"
    ] == pytest.approx(0.0, abs=1e-15)
    assert freeze_runs["dt_10us"]["post_event_velocity"]["classification"] == (
        "exact_source_velocity_retained"
    )
    assert freeze_runs["dt_5us"]["post_event_velocity"]["classification"] == "exact_zero"
    assert freeze_runs["dt_2p5us"]["post_event_velocity"]["classification"] == "finite_other"
    assert freeze_runs["dt_10us"]["event"]["event_time_s"] == pytest.approx(7.30002e-5)
    assert freeze_runs["dt_10us"]["event"]["first_terminal_output_time_s"] == pytest.approx(7.5e-5)
    assert (root / "boundary_semantics_report.json").is_file()
    assert (root / "gates.csv").is_file()


def test_distinguishes_observed_from_reconstructed_hit(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path / "probe",
        event_offsets_s={"dt_10us": 2e-10, "dt_5us": -1e-10, "dt_2p5us": 0.0},
    )

    boundary.normalize(root, config)

    freeze_event = _first_row(root / "freeze_inlet_37/dt_10us/event_observation.csv")
    disappear_event = _first_row(root / "disappear_pump_35/dt_10us/event_observation.csv")
    assert freeze_event["hit_position_basis"] == "DIRECT_COMSOL_FROZEN_STATE_OBSERVATION"
    assert freeze_event["direct_comsol_r_m"] != ""
    assert disappear_event["hit_position_basis"] == (
        "ANALYTIC_RECONSTRUCTION_FROM_COMSOL_INITIAL_STATE_NOT_DIRECT_EVENT_OBSERVATION"
    )
    assert disappear_event["direct_comsol_r_m"] == ""
    assert float(disappear_event["hit_z_m"]) == pytest.approx(-2e-9, abs=1e-16)

    with (root / "disappear_pump_35/dt_2p5us/state.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        states = list(csv.DictReader(stream))
    assert len(states) == 61
    assert math.isnan(float(states[30]["r_m"]))
    assert all(
        (root / scenario / step / "step_summary.json").is_file()
        for scenario in ("freeze_inlet_37", "disappear_pump_35")
        for step in boundary.EXPECTED_STEP_SECONDS
    )


def test_accepts_mapping_forms_for_steps_and_scenarios(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path / "probe", mappings=True)

    report = boundary.normalize(root, config)

    assert report["status"] == "PASS"
    assert set(report["scenarios"]) == {"freeze_inlet_37", "disappear_pump_35"}


@pytest.mark.parametrize(
    ("duplicate_particle", "bad_grid", "message"),
    [
        (True, False, "expected exactly 1 particle row"),
        (False, True, "output time grid differs"),
    ],
)
def test_rejects_malformed_particle_or_time_grid(
    tmp_path: Path, duplicate_particle: bool, bad_grid: bool, message: str
) -> None:
    root, config = _fixture(
        tmp_path / "probe",
        duplicate_particle=duplicate_particle,
        bad_grid=bad_grid,
    )

    with pytest.raises(ValueError, match=message):
        boundary.normalize(root, config)


def test_records_semantic_failures_without_rejecting_valid_evidence(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path / "probe",
        freeze_drift_m=1e-9,
        disappear_finite_after_event=True,
    )

    report = boundary.normalize(root, config)

    assert report["status"] == "FAIL"
    assert report["scenarios"]["freeze_inlet_37"]["status"] == "FAIL"
    assert report["scenarios"]["disappear_pump_35"]["status"] == "FAIL"
    with (root / "gates.csv").open(newline="", encoding="utf-8") as stream:
        gates = list(csv.DictReader(stream))
    failed = {row["gate"] for row in gates if row["status"] == "FAIL"}
    assert "freeze_post_event_position_retention" in failed
    assert "disappear_post_event_position_is_nan" in failed


def test_event_time_step_spread_is_an_independent_gate(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path / "probe",
        event_offsets_s={"dt_10us": -4e-9, "dt_5us": 4e-9, "dt_2p5us": 0.0},
    )

    report = boundary.normalize(root, config)

    assert report["status"] == "FAIL"
    for scenario in report["scenarios"].values():
        consistency = scenario["event_time_self_consistency"]
        assert consistency["status"] == "FAIL"
        assert consistency["observed_value"] == pytest.approx(8e-9)
        for run in scenario["runs"].values():
            analytic_gate = next(
                gate for gate in run["gates"] if gate["gate"] == "analytic_event_time"
            )
            assert analytic_gate["status"] == "PASS"


@pytest.mark.parametrize("corruption", ["mismatch", "missing"])
def test_rejects_configuration_receipt_mismatch(tmp_path: Path, corruption: str) -> None:
    root, config = _fixture(tmp_path / "probe")
    log_path = root / "comsol_process.log"
    text = log_path.read_text(encoding="utf-8")
    if corruption == "mismatch":
        text = text.replace("force_free=true", "force_free=false", 1)
        message = "expected force_free=true"
    else:
        lines = text.splitlines()
        text = "\n".join(
            line
            for index, line in enumerate(lines)
            if not (index == 1 and boundary.RECEIPT_PREFIX in line)
        )
        message = "expected exactly six"
    log_path.write_text(text + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        boundary.normalize(root, config)


def test_active_path_error_is_a_scientific_failure_not_malformed_data(tmp_path: Path) -> None:
    root, config = _fixture(
        tmp_path / "probe",
        active_position_error_m=1e-6,
        active_velocity_error_m_per_s=1e-4,
    )

    report = boundary.normalize(root, config)

    assert report["status"] == "FAIL"
    run = report["scenarios"]["freeze_inlet_37"]["runs"]["dt_10us"]
    assert run["active_force_free_path"]["maximum_position_absolute_error_m"] == pytest.approx(1e-6)
    failed = {gate["gate"] for gate in run["gates"] if gate["status"] == "FAIL"}
    assert failed == {
        "force_free_active_position_path",
        "force_free_active_velocity_path",
    }
