"""Compare the P18-H ``hold`` candidate with the locked COMSOL Freeze probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Final

import numpy as np

from chamber_particles import open_result
from tools.vv.comsol.run_p18h_hold_candidate import (
    CONFIG_FILENAME,
    EVENT_FILENAME,
    PRODUCER_FILENAME,
    REPORT_FILENAME,
    RESULT_DIRECTORY,
    TRAJECTORY_FILENAME,
    _load_config,
    _locked_references,
    _mapping,
    _sha256,
)
from tools.vv.comsol.run_p18h_hold_candidate import TOOL_REVISION as PRODUCER_REVISION

TOOL_REVISION: Final = "p18h_hold_freeze_evaluator_v1"
RESULT_FILENAME: Final = "comparison_result.json"
GATES_FILENAME: Final = "gates.csv"
README_FILENAME: Final = "README.md"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _gate(name: str, passed: bool, observed: object, limit: object) -> dict[str, object]:
    return {
        "gate": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "limit": limit,
    }


def _candidate_integrity(
    config_path: Path, candidate_root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    report_path = candidate_root / REPORT_FILENAME
    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "candidate report")
    config_hash = _sha256(config_path)
    if report.get("status") != "COMPLETE" or report.get("tool_revision") != PRODUCER_REVISION:
        raise ValueError("candidate report is not a complete expected producer run")
    if any(
        report.get(key) != config_hash
        for key in ("configuration_sha256_before", "configuration_sha256_after")
    ):
        raise ValueError("candidate report configuration hash differs")
    if _sha256(candidate_root / CONFIG_FILENAME) != config_hash:
        raise ValueError("staged candidate configuration differs")
    current_producer = Path(__file__).with_name(PRODUCER_FILENAME)
    staged_producer = candidate_root / PRODUCER_FILENAME
    producer_hash = _sha256(current_producer)
    if (
        _sha256(staged_producer) != producer_hash
        or report.get("producer_source_sha256") != producer_hash
    ):
        raise ValueError("candidate producer source identity differs")
    locked = _locked_references(config)
    if report.get("locked_references") != locked:
        raise ValueError("candidate reference locks differ from current locked inputs")
    expected_hashes = {
        "result_manifest_sha256": candidate_root / RESULT_DIRECTORY / "run.json",
        "trajectory_sha256": candidate_root / TRAJECTORY_FILENAME,
        "events_sha256": candidate_root / EVENT_FILENAME,
        "case_sha256": candidate_root / "candidate_case.yaml",
        "canonical_input_sha256": candidate_root / "candidate_input.h5",
    }
    for receipt_key, path in expected_hashes.items():
        if _sha256(path) != report.get(receipt_key):
            raise ValueError(f"candidate {receipt_key} differs from its receipt")
    result = open_result(candidate_root / RESULT_DIRECTORY)
    expected = _mapping(config["expected_candidate"], "expected_candidate")
    for key in (
        "case_schema_version",
        "result_schema_version",
        "engine_algorithm_revision",
        "boundary_algorithm_revision",
        "result_algorithm_revision",
    ):
        if result.manifest.get(key) != expected[key]:
            raise ValueError(f"candidate {key} differs from the preregistered revision")
    return report, result


def _reference_observation(
    config: dict[str, Any], locked: dict[str, dict[str, str]]
) -> tuple[np.ndarray, np.ndarray]:
    case = _mapping(config["case"], "case")
    reference = _mapping(config["reference"], "reference")
    rows = _rows(Path(locked["state"]["path"]))
    event_rows = _rows(Path(locked["event"]["path"]))
    if len(rows) != case["output_frames"] or len(event_rows) != 1:
        raise ValueError("locked reference cardinality differs")
    values = np.asarray(
        [
            [
                float(row["time_s"]),
                float(row["r_m"]),
                float(row["z_m"]),
                float(row["velocity_r_m_per_s"]),
                float(row["velocity_z_m_per_s"]),
                float(row["current_status_code"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    expected_times = np.linspace(0.0, float(case["time_end_s"]), int(case["output_frames"]))
    if not np.allclose(values[:, 0], expected_times, rtol=0.0, atol=5.0e-18):
        raise ValueError("locked reference frame grid differs")
    expected_status = np.concatenate(
        (np.ones(30, dtype=np.float64), np.full(31, reference["expected_status_code"]))
    )
    if not np.array_equal(values[:, 5], expected_status):
        raise ValueError("locked reference terminal status sequence differs")
    event = event_rows[0]
    event_values = np.asarray(
        [float(event["event_time_s"]), float(event["hit_r_m"]), float(event["hit_z_m"])],
        dtype=np.float64,
    )
    if event["wall_condition"] != reference["wall_condition"]:
        raise ValueError("locked reference wall condition differs")
    return values, event_values


def _candidate_projection(
    config: dict[str, Any], candidate_root: Path
) -> tuple[list[dict[str, str]], dict[str, str]]:
    case = _mapping(config["case"], "case")
    trajectory = _rows(candidate_root / TRAJECTORY_FILENAME)
    events = _rows(candidate_root / EVENT_FILENAME)
    if len(trajectory) != case["output_frames"] or len(events) != 1:
        raise ValueError("candidate projection cardinality differs")
    expected_times = np.linspace(0.0, float(case["time_end_s"]), int(case["output_frames"]))
    times = np.asarray([float(row["time_s"]) for row in trajectory])
    if not np.array_equal(times, expected_times):
        raise ValueError("candidate projection frame grid differs")
    return trajectory, events[0]


def _terminal_observation(result: Any, expected: dict[str, Any]) -> tuple[dict[str, object], bool]:
    final = result.read_final()
    series = result.read_lifecycle_series()
    counts = result.manifest["lifecycle_counts"]
    observed: dict[str, object] = {
        "final_lifecycle": int(final.lifecycle[0]),
        "kinematics_valid": int(final.kinematics_valid[0]),
        "manifest_terminal_counts": {
            "held": int(counts["held"]),
            "stuck": int(counts["stuck"]),
            "escaped": int(counts["escaped"]),
            "failed": int(counts["failed"]),
        },
        "series_held": int(series.held[-1]),
        "series_stuck": int(series.stuck[-1]),
        "series_escaped": int(series.escaped[-1]),
        "series_failed": int(series.failed[-1]),
    }
    passed = (
        observed["final_lifecycle"] == expected["lifecycle_code"]
        and observed["kinematics_valid"] == 1
        and observed["manifest_terminal_counts"]
        == {"held": 1, "stuck": 0, "escaped": 0, "failed": 0}
        and observed["series_held"] == 1
        and observed["series_stuck"] == 0
        and observed["series_escaped"] == 0
        and observed["series_failed"] == 0
    )
    return observed, passed


def _metrics_and_gates(
    config: dict[str, Any],
    report: dict[str, Any],
    result: Any,
    reference: np.ndarray,
    reference_event: np.ndarray,
    trajectory: list[dict[str, str]],
    event: dict[str, str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    case = _mapping(config["case"], "case")
    expected = _mapping(config["expected_candidate"], "expected_candidate")
    acceptance = _mapping(config["acceptance"], "acceptance")
    times = np.asarray([float(row["time_s"]) for row in trajectory])
    position = np.asarray([[float(row["r_m"]), float(row["z_m"])] for row in trajectory])
    velocity = np.asarray(
        [[float(row["velocity_r_m_per_s"]), float(row["velocity_z_m_per_s"])] for row in trajectory]
    )
    charge = np.asarray([float(row["charge_number_e"]) for row in trajectory])
    lifecycle = tuple(row["lifecycle"] for row in trajectory)
    active = np.asarray([name == "active" for name in lifecycle])
    held = np.asarray([name == "held" for name in lifecycle])
    initial_position = np.asarray(case["source_position_m"], dtype=np.float64)
    initial_velocity = np.asarray(case["source_velocity_m_per_s"], dtype=np.float64)
    analytic_position = initial_position + times[:, None] * initial_velocity
    active_position_error = float(
        np.max(np.linalg.norm(position[active] - analytic_position[active], axis=1))
    )
    active_velocity_error = float(
        np.max(np.linalg.norm(velocity[active] - initial_velocity, axis=1))
    )
    cross_active_position = float(
        np.max(np.linalg.norm(position[active] - reference[active, 1:3], axis=1))
    )
    cross_active_velocity = float(
        np.max(np.linalg.norm(velocity[active] - reference[active, 3:5], axis=1))
    )
    event_time = float(event["event_time_s"])
    hit = np.asarray([float(event["hit_r_m"]), float(event["hit_z_m"])])
    hit_analytic = np.asarray(case["analytic_hit_position_m"], dtype=np.float64)
    velocity_pre = np.asarray(
        [float(event["pre_velocity_r_m_per_s"]), float(event["pre_velocity_z_m_per_s"])]
    )
    velocity_post = np.asarray(
        [float(event["post_velocity_r_m_per_s"]), float(event["post_velocity_z_m_per_s"])]
    )
    held_spread = float(np.max(np.linalg.norm(position[held] - position[held][0], axis=1)))
    held_cross = float(np.max(np.linalg.norm(position[held] - reference[held, 1:3], axis=1)))
    held_velocity_error = float(np.max(np.linalg.norm(velocity[held] - initial_velocity, axis=1)))
    event_velocity_error = max(
        float(np.linalg.norm(velocity_pre - initial_velocity)),
        float(np.linalg.norm(velocity_post - initial_velocity)),
    )
    event_charge_error = max(
        abs(float(event["charge_number_pre_e"]) - float(case["source_charge_number_e"])),
        abs(float(event["charge_number_post_e"]) - float(case["source_charge_number_e"])),
        float(np.max(np.abs(charge[held] - float(case["source_charge_number_e"])))),
    )
    failures = result.read_failure_events()
    boundary = result.read_boundary_events()
    terminal_observed, terminal_passed = _terminal_observation(result, expected)
    event_observed = {
        "boundary_events": int(boundary.particle_id.size),
        "failure_events": int(failures.particle_id.size),
        "law": event["law"],
        "outcome": event["outcome"],
    }
    event_passed = event_observed == {
        "boundary_events": 1,
        "failure_events": 0,
        "law": "hold",
        "outcome": "held",
    }
    gates = [
        _gate(
            "candidate_active_position_vs_analytic",
            active_position_error <= acceptance["active_position_absolute_m"],
            active_position_error,
            acceptance["active_position_absolute_m"],
        ),
        _gate(
            "candidate_active_velocity_vs_analytic",
            active_velocity_error <= acceptance["active_velocity_absolute_m_per_s"],
            active_velocity_error,
            acceptance["active_velocity_absolute_m_per_s"],
        ),
        _gate(
            "candidate_active_position_vs_locked_comsol",
            cross_active_position <= acceptance["active_position_absolute_m"],
            cross_active_position,
            acceptance["active_position_absolute_m"],
        ),
        _gate(
            "candidate_active_velocity_vs_locked_comsol",
            cross_active_velocity <= acceptance["active_velocity_absolute_m_per_s"],
            cross_active_velocity,
            acceptance["active_velocity_absolute_m_per_s"],
        ),
        _gate(
            "exactly_one_hold_event_and_zero_failures",
            event_passed,
            event_observed,
            {"boundary_events": 1, "failure_events": 0, "law": "hold", "outcome": "held"},
        ),
        _gate(
            "candidate_event_time_vs_analytic",
            abs(event_time - float(case["analytic_event_time_s"]))
            <= acceptance["event_time_absolute_s"],
            abs(event_time - float(case["analytic_event_time_s"])),
            acceptance["event_time_absolute_s"],
        ),
        _gate(
            "candidate_event_time_vs_locked_comsol",
            abs(event_time - reference_event[0]) <= acceptance["event_time_absolute_s"],
            abs(event_time - reference_event[0]),
            acceptance["event_time_absolute_s"],
        ),
        _gate(
            "candidate_hit_position_vs_analytic",
            float(np.linalg.norm(hit - hit_analytic)) <= acceptance["hit_position_norm_m"],
            float(np.linalg.norm(hit - hit_analytic)),
            acceptance["hit_position_norm_m"],
        ),
        _gate(
            "candidate_hit_position_vs_locked_comsol",
            float(np.linalg.norm(hit - reference_event[1:3])) <= acceptance["hit_position_norm_m"],
            float(np.linalg.norm(hit - reference_event[1:3])),
            acceptance["hit_position_norm_m"],
        ),
        _gate(
            "saved_lifecycle_is_30_active_then_31_held",
            lifecycle == ("active",) * 30 + ("held",) * 31,
            {
                "active_frames": int(np.count_nonzero(active)),
                "held_frames": int(np.count_nonzero(held)),
            },
            {"active_frames": 30, "held_frames": 31},
        ),
        _gate(
            "candidate_held_position_retention",
            held_spread <= acceptance["held_position_spread_m"],
            held_spread,
            acceptance["held_position_spread_m"],
        ),
        _gate(
            "candidate_held_position_vs_locked_comsol",
            held_cross <= acceptance["held_position_cross_absolute_m"],
            held_cross,
            acceptance["held_position_cross_absolute_m"],
        ),
        _gate(
            "candidate_hold_retains_incident_velocity",
            max(event_velocity_error, held_velocity_error)
            <= acceptance["retained_velocity_absolute_m_per_s"],
            {
                "event_maximum_error_m_per_s": event_velocity_error,
                "held_maximum_error_m_per_s": held_velocity_error,
            },
            acceptance["retained_velocity_absolute_m_per_s"],
        ),
        _gate(
            "candidate_hold_retains_charge",
            event_charge_error <= acceptance["retained_charge_absolute_e"],
            event_charge_error,
            acceptance["retained_charge_absolute_e"],
        ),
        _gate(
            "candidate_terminal_state_is_held_and_valid",
            terminal_passed,
            terminal_observed,
            {"final_lifecycle": 5, "kinematics_valid": 1, "held": 1, "other_terminal": 0},
        ),
    ]
    metrics: dict[str, object] = {
        "active_frames": int(np.count_nonzero(active)),
        "held_frames": int(np.count_nonzero(held)),
        "active_position_analytic_maximum_m": active_position_error,
        "active_velocity_analytic_maximum_m_per_s": active_velocity_error,
        "active_position_cross_maximum_m": cross_active_position,
        "active_velocity_cross_maximum_m_per_s": cross_active_velocity,
        "candidate_event_time_s": event_time,
        "reference_event_time_s": float(reference_event[0]),
        "candidate_hit_position_m": hit.tolist(),
        "reference_hit_position_m": reference_event[1:3].tolist(),
        "held_position_spread_m": held_spread,
        "held_position_cross_maximum_m": held_cross,
        "event_velocity_retention_error_m_per_s": event_velocity_error,
        "held_velocity_retention_error_m_per_s": held_velocity_error,
        "charge_retention_error_e": event_charge_error,
        "candidate_receipt": {
            "trajectory_rows": report["trajectory_rows"],
            "event_rows": report["event_rows"],
            "failure_event_count": report["failure_event_count"],
        },
    }
    return metrics, gates


def _write_gates(path: Path, gates: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("gate", "status", "observed", "limit"), lineterminator="\n"
        )
        writer.writeheader()
        for gate in gates:
            writer.writerow(
                {
                    **gate,
                    "observed": json.dumps(gate["observed"], sort_keys=True),
                    "limit": json.dumps(gate["limit"], sort_keys=True),
                }
            )


def _readme(result: dict[str, Any]) -> str:
    metrics = _mapping(result["metrics"], "metrics")
    summary = _mapping(result["gate_summary"], "gate_summary")
    return f"""# P18-H hold / COMSOL Freeze microcase

Status: **{result["status"]}** ({summary["pass"]} PASS, {summary["fail"]} FAIL).

This compact external V&V package compares the production `hold` / `held`
terminal-boundary semantics with the already locked COMSOL 6.4 boundary 37
Freeze probe.  COMSOL was not rerun.  The one particle starts at
`(r,z)=(0.23927,0.115) m` with velocity `(10,0) m/s` and reaches `r=0.24 m`
in a force-free R-Z domain.

The candidate event time is `{metrics["candidate_event_time_s"]:.17g} s`; the
locked COMSOL value is `{metrics["reference_event_time_s"]:.17g} s`.  The
candidate has 30 active saved frames and 31 held saved frames.  Its held
position spread is `{metrics["held_position_spread_m"]:.3e} m` and its maximum
held-position difference from the aligned COMSOL frames is
`{metrics["held_position_cross_maximum_m"]:.3e} m`.

This result is deliberately narrow.  It does not make COMSOL a golden truth,
does not certify grazing/corner impacts or full physics, and does not define a
paused particle that can resume.  `hold` is terminal: position, impact velocity,
and charge remain queryable, while no later physics or charge evolution occurs.
The locked COMSOL files contain no charge column, and this external candidate
uses zero charge; its charge gate is therefore a candidate self-consistency
check, not evidence of COMSOL or nonzero-charge retention. Nonzero retention is
covered independently by the public solver scenario.
Exact gates, hashes, algorithm revisions, and exclusions are recorded in
`comparison_result.json` and `gates.csv`.  Candidate raw output remains in the
external run directory named by the comparison manifest and is not duplicated
here.
"""


def evaluate(config_path: Path, candidate_root: Path, output: Path) -> dict[str, object]:
    """Evaluate one locked candidate/reference pair and write compact evidence."""

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config_path = config_path.resolve()
    candidate_root = candidate_root.resolve()
    config = _load_config(config_path)
    locked = _locked_references(config)
    report, result = _candidate_integrity(config_path, candidate_root, config)
    reference, reference_event = _reference_observation(config, locked)
    trajectory, event = _candidate_projection(config, candidate_root)
    metrics, gates = _metrics_and_gates(
        config, report, result, reference, reference_event, trajectory, event
    )
    status = "PASS" if all(gate["status"] == "PASS" for gate in gates) else "FAIL"
    output.mkdir(parents=True)
    evaluator_hash = _sha256(Path(__file__).resolve())
    result_payload: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "evaluation_id": config["evaluation_id"],
        "status": status,
        "scope": {
            "particles": 1,
            "frames": 61,
            "force_free": True,
            "dynamic_charge": False,
            "coordinate_system": "axisymmetric_rz",
            "comsol_rerun": False,
        },
        "metrics": metrics,
        "gates": gates,
        "gate_summary": {
            "pass": sum(gate["status"] == "PASS" for gate in gates),
            "fail": sum(gate["status"] == "FAIL" for gate in gates),
        },
        "provenance": {
            "configuration": str(config_path),
            "configuration_sha256": _sha256(config_path),
            "candidate_root": str(candidate_root),
            "candidate_report_sha256": _sha256(candidate_root / REPORT_FILENAME),
            "candidate_manifest_sha256": _sha256(candidate_root / RESULT_DIRECTORY / "run.json"),
            "candidate_trajectory_sha256": _sha256(candidate_root / TRAJECTORY_FILENAME),
            "candidate_events_sha256": _sha256(candidate_root / EVENT_FILENAME),
            "producer_source_sha256": _sha256(Path(__file__).with_name(PRODUCER_FILENAME)),
            "evaluator_source_sha256": evaluator_hash,
            "locked_reference": locked,
        },
        "revisions": report["revisions"],
        "claim_policy": config["claim_policy"],
        "interpretation": {
            "comsol_role": "external locked reference, not golden truth",
            "post_event_velocity": "candidate retention is gated; COMSOL retention is characterized",
            "charge": "candidate self-retention only; COMSOL charge was not exported",
            "held_semantics": "terminal and inactive, with retained queryable kinematics",
        },
    }
    with (output / RESULT_FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result_payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_gates(output / GATES_FILENAME, gates)
    (output / README_FILENAME).write_text(_readme(result_payload), encoding="utf-8")
    return result_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = evaluate(arguments.config, arguments.candidate_root, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
