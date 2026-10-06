"""Normalize the single deterministic M3-C3 Case-P COMSOL reference.

The normalizer owns only the external-reference representation.  It validates
the fixed 287-particle, 121-frame execution and the candidate-owned release
state; it does not select comparison tolerances or modify solver inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Final

TOOL_REVISION: Final = "m3c3_caseP_three_current_comsol_normalizer_v3"
EXPECTED_PARTICLES: Final = 287
EXPECTED_FRAMES: Final = 121
TIME_END_S: Final = 0.03
TIME_TOLERANCE_S: Final = 3.0e-13
STATE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "current_status_code",
    "final_status_code",
    "stop_or_event_time_s",
)
TRAJECTORY_HEADER: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
EVENT_HEADER: Final = (
    "particle_id",
    "event_time_s",
    "event_type",
    "outcome",
    "boundary_semantic",
)
RELEASE_HEADER: Final = (
    "particle_id",
    "release_time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
)
STATUS: Final = {1: "active", 2: "held", 3: "stuck", 4: "escaped"}
BOUNDARY: Final = {
    "held": "gas_inlet_hold",
    "stuck": "material_stick_boundary_unspecified",
    "escaped": "pump_outlet_escape",
}
CONFIGURATION_PREFIX: Final = "M3C3_CASEP|configuration|"
SOLVE_PREFIX: Final = "M3C3_CASEP|solve_pass|"
RUN_PREFIX: Final = "M3C3_CASEP|run_pass|"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_times() -> list[float]:
    return (
        [index * 1.0e-5 for index in range(51)]
        + [index * 1.0e-4 for index in range(6, 51)]
        + [index * 1.0e-3 for index in range(6, 31)]
    )


def _fields(line: str, prefix: str) -> dict[str, str]:
    payload = line[len(prefix) :].strip()
    result: dict[str, str] = {}
    for item in payload.split("|"):
        if "=" not in item:
            raise ValueError(f"malformed COMSOL receipt item: {item}")
        key, value = item.split("=", 1)
        if not key or key in result:
            raise ValueError(f"duplicate or blank COMSOL receipt key: {key}")
        result[key] = value
    return result


def _single_receipt(log: Path, prefix: str) -> dict[str, str]:
    rows = [
        _fields(line, prefix)
        for line in log.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if line.startswith(prefix)
    ]
    if len(rows) != 1:
        raise ValueError(f"{log}: expected one {prefix} receipt, found {len(rows)}")
    return rows[0]


def _validate_configuration(values: dict[str, str], step_text: str) -> None:
    expected = {
        "case": "caseP_100nm_three_current",
        "step_s": step_text,
        "time_end_s": "0.03",
        "output_times": "121",
        "particle_rows": "287",
        "physics": "fpt",
        "background_study": "std2",
        "background_step": "ftper",
        "background_solution": "sol2",
        "brownian_active": "false",
        "saffman_active": "false",
        "dynamic_charge_active": "true",
        "charge_revision": "aggregate_relative_drift_regularized_three_current_v1",
        "ion_drag_revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        "drag_revision": "epstein_linear_effective_gas_sensitivity_v1",
        "drag_implementation": "explicit_custom_force",
        "maximum_relative_ion_speed_m_s": "1e6",
        "integrator": "classical_rk4",
        "integrator_order": "4",
        "relative_tolerance": "1e-8",
        "field_source": "canonical_exact_connectivity_P1_sectionwise",
        "release_source": "shared_three_current_release_table",
        "boundary_material": "stick",
        "boundary_37": "freeze_hold",
        "boundary_35": "disappear_escape",
        "source_model": "source_copy.mph",
        "model_saved": "false",
    }
    if values != expected:
        missing = sorted(set(expected) - set(values))
        extra = sorted(set(values) - set(expected))
        mismatched = sorted(
            name for name in set(expected) & set(values) if values[name] != expected[name]
        )
        raise ValueError(
            "COMSOL configuration receipt differs; "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )


def _numerical_run(inputs: dict[str, Any]) -> tuple[float, str, str]:
    numerical = inputs.get("numerical_run")
    if not isinstance(numerical, dict):
        raise ValueError("execution_inputs.json: numerical_run is missing")
    step_s = float(numerical.get("fixed_rk4_step_s", math.nan))
    base_step_s = float(numerical.get("base_config_fixed_rk4_step_s", math.nan))
    role = str(numerical.get("role", ""))
    if base_step_s != 1.0e-5:
        raise ValueError("execution_inputs.json: base COMSOL step differs")
    if step_s == 1.0e-5 and role == "baseline":
        return step_s, "1e-5", role
    if step_s == 5.0e-6 and role == "time_step_refinement":
        return step_s, "5e-6", role
    if step_s == 2.5e-6 and role == "time_step_refinement":
        return step_s, "2.5e-6", role
    if step_s == 1.25e-6 and role == "time_step_refinement":
        return step_s, "1.25e-6", role
    raise ValueError("execution_inputs.json: unsupported numerical_run")


def _release_state(path: Path) -> dict[int, tuple[float, float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RELEASE_HEADER:
            raise ValueError(f"{path}: release header differs")
        for row in reader:
            particle_id = int(row["particle_id"])
            if float(row["release_time_s"]) != 0.0:
                raise ValueError(f"{path}: nonzero release time")
            values = tuple(
                float(row[name])
                for name in (
                    "r_m",
                    "z_m",
                    "velocity_r_m_per_s",
                    "velocity_z_m_per_s",
                    "charge_number_e",
                )
            )
            if particle_id in result or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: duplicate or nonfinite release {particle_id}")
            result[particle_id] = values  # type: ignore[assignment]
    if set(result) != set(range(1, EXPECTED_PARTICLES + 1)):
        raise ValueError(f"{path}: release IDs must be exactly 1..287")
    return result


def _raw_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.startswith("%") or not line.strip():
                continue
            rows.append([float(value) for value in next(csv.reader([line]))])
    return rows


def _integer(value: float, context: str) -> int:
    rounded = round(value)
    if not math.isfinite(value) or abs(value - rounded) > 1.0e-9:
        raise ValueError(f"{context}: expected integer-valued data, found {value}")
    return rounded


def _finite_state(record: list[float]) -> bool:
    return all(math.isfinite(record[index]) for index in range(7))


def _status(record: list[float], index: int, raw_path: Path) -> int:
    code = _integer(record[index], str(raw_path))
    if code not in STATUS:
        raise ValueError(f"{raw_path}: unknown particle status {code}")
    return code


def _terminal(records: list[list[float]], raw_path: Path) -> tuple[str, float] | None:
    terminal: tuple[str, float] | None = None
    for record in records:
        if not _finite_state(record):
            continue
        current = _status(record, 7, raw_path)
        final = _status(record, 8, raw_path)
        outcome_code = current if current != 1 else final
        if outcome_code == 1:
            continue
        event_time = record[9]
        if not math.isfinite(event_time) or not 0.0 <= event_time <= TIME_END_S + TIME_TOLERANCE_S:
            raise ValueError(f"{raw_path}: invalid terminal event time")
        candidate = (STATUS[outcome_code], event_time)
        if terminal is not None and terminal != candidate:
            raise ValueError(f"{raw_path}: inconsistent terminal state")
        terminal = candidate
    return terminal


def _initial_tolerance(observed: float, expected: float, scale: float) -> float:
    return 4096.0 * math.ulp(max(abs(observed), abs(expected), scale, 1.0e-300))


def _write_missing_escape(
    writer: Any,
    record: list[float],
    terminal: tuple[str, float] | None,
    particle_id: int,
    expected_time: float,
    raw_path: Path,
) -> None:
    partially_finite = any(math.isfinite(record[index]) for index in range(2, 7))
    valid_escape = (
        terminal is not None
        and terminal[0] == "escaped"
        and expected_time + TIME_TOLERANCE_S >= terminal[1]
    )
    if partially_finite or not valid_escape:
        raise ValueError(f"{raw_path}: invalid missing state for particle {particle_id}")
    writer.writerow(
        (
            particle_id,
            expected_time,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            "escaped",
        )
    )


def _initial_errors(
    record: list[float],
    expected: tuple[float, float, float, float, float],
    scales: tuple[float, ...],
    particle_id: int,
    raw_path: Path,
) -> list[float]:
    errors: list[float] = []
    for actual, reference, scale in zip(record[2:7], expected, scales, strict=True):
        error = abs(actual - reference)
        if error > _initial_tolerance(actual, reference, scale):
            raise ValueError(f"{raw_path}: t=0 differs from shared release for {particle_id}")
        errors.append(error)
    return errors


def _normalize_particle(
    writer: Any,
    row: list[float],
    release: dict[int, tuple[float, float, float, float, float]],
    scales: tuple[float, ...],
    expected_times: list[float],
    raw_path: Path,
) -> tuple[int, str, tuple[object, ...] | None, list[float]]:
    expected_width = len(STATE_COLUMNS) * EXPECTED_FRAMES
    if len(row) != expected_width:
        raise ValueError(f"{raw_path}: expected width {expected_width}, found {len(row)}")
    particle_id = _integer(row[0], str(raw_path))
    records = [
        row[index * len(STATE_COLUMNS) : (index + 1) * len(STATE_COLUMNS)]
        for index in range(EXPECTED_FRAMES)
    ]
    terminal = _terminal(records, raw_path)
    initial_errors = [0.0] * 5
    last_lifecycle = "active"
    for frame, (record, expected_time) in enumerate(zip(records, expected_times, strict=True)):
        if not _finite_state(record):
            _write_missing_escape(writer, record, terminal, particle_id, expected_time, raw_path)
            last_lifecycle = "escaped"
            continue
        if _integer(record[0], str(raw_path)) != particle_id:
            raise ValueError(f"{raw_path}: particle ID changed within a row")
        if not math.isclose(record[1], expected_time, rel_tol=0.0, abs_tol=TIME_TOLERANCE_S):
            raise ValueError(f"{raw_path}: unexpected output time")
        last_lifecycle = STATUS[_status(record, 7, raw_path)]
        writer.writerow((particle_id, *record[1:7], last_lifecycle))
        if frame == 0:
            initial_errors = _initial_errors(
                record, release[particle_id], scales, particle_id, raw_path
            )
    final_lifecycle = terminal[0] if terminal is not None else last_lifecycle
    event = None
    if terminal is not None:
        outcome, event_time = terminal
        event = (
            particle_id,
            event_time,
            "terminal_boundary",
            outcome,
            BOUNDARY[outcome],
        )
    return particle_id, final_lifecycle, event, initial_errors


def _normalize_trajectories(
    root: Path,
    release: dict[int, tuple[float, float, float, float, float]],
) -> dict[str, Any]:
    raw_path = root / "trajectory_raw_wide.csv"
    rows = _raw_rows(raw_path)
    if len(rows) != EXPECTED_PARTICLES:
        raise ValueError(f"{raw_path}: expected 287 rows, found {len(rows)}")
    expected_times = _output_times()
    scales = tuple(max(abs(values[index]) for values in release.values()) for index in range(5))
    trajectory_path = root / "trajectory_reference.csv"
    events_path = root / "events_reference.csv"
    ids: set[int] = set()
    lifecycle_counts: Counter[str] = Counter()
    events: list[tuple[object, ...]] = []
    initial_error_max = [0.0] * 5
    with trajectory_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(TRAJECTORY_HEADER)
        for row in rows:
            particle_id, final_lifecycle, event, initial_errors = _normalize_particle(
                writer, row, release, scales, expected_times, raw_path
            )
            if particle_id in ids:
                raise ValueError(f"{raw_path}: duplicate particle {particle_id}")
            ids.add(particle_id)
            lifecycle_counts[final_lifecycle] += 1
            initial_error_max = [
                max(current, observed)
                for current, observed in zip(initial_error_max, initial_errors, strict=True)
            ]
            if event is not None:
                events.append(event)
    if ids != set(range(1, EXPECTED_PARTICLES + 1)):
        raise ValueError(f"{raw_path}: particle IDs must be exactly 1..287")
    with events_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(EVENT_HEADER)
        writer.writerows(sorted(events))
    return {
        "trajectory_rows": EXPECTED_PARTICLES * EXPECTED_FRAMES,
        "event_count": len(events),
        "final_lifecycle_counts": dict(sorted(lifecycle_counts.items())),
        "initial_state_absolute_error_max": dict(
            zip(
                ("r_m", "z_m", "velocity_r_m_per_s", "velocity_z_m_per_s", "charge_number_e"),
                initial_error_max,
                strict=True,
            )
        ),
        "trajectory_reference_sha256": _sha256(trajectory_path),
        "events_reference_sha256": _sha256(events_path),
        "trajectory_raw_wide_sha256": _sha256(raw_path),
    }


def _verified_execution_inputs(root: Path) -> dict[str, Any]:
    path = root / "execution_inputs.json"
    inputs = json.loads(path.read_text(encoding="utf-8-sig"))
    if inputs.get("schema_version") != 1 or inputs.get("status") != "LOCKED_FOR_EXECUTION":
        raise ValueError(f"{path}: execution input record is not locked")
    artifacts = inputs.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{path}: execution artifacts are missing")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError(f"{path}: malformed execution artifact")
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        if not artifact_path.is_file() or _sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"{path}: execution artifact differs: {artifact_path}")
    before = inputs.get("source_mph_sha256_before")
    after = inputs.get("source_mph_sha256_after")
    if before != after or before != inputs.get("expected_source_mph_sha256"):
        raise ValueError(f"{path}: source MPH hash lock failed")
    return inputs


def normalize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    execution_inputs = _verified_execution_inputs(root)
    fixed_step_s, step_text, numerical_run_role = _numerical_run(execution_inputs)
    log = root / "comsol_process.log"
    configuration = _single_receipt(log, CONFIGURATION_PREFIX)
    _validate_configuration(configuration, step_text)
    solve = _single_receipt(log, SOLVE_PREFIX)
    run = _single_receipt(log, RUN_PREFIX)
    if solve.get("step_s") != step_text or float(solve.get("seconds", "nan")) <= 0.0:
        raise ValueError("COMSOL solve receipt differs")
    if run != {
        "case": "caseP_100nm_three_current",
        "step_s": step_text,
        "time_end_s": "0.03",
        "output_times": "121",
        "particles": "287",
        "model_saved": "false",
    }:
        raise ValueError("COMSOL run-pass receipt differs")
    release_path = root / "three_current_release_state.csv"
    metrics = _normalize_trajectories(root, _release_state(release_path))
    process_metrics = json.loads(
        (root / "comsol_process_metrics.json").read_text(encoding="utf-8-sig")
    )
    if float(process_metrics.get("wall_time_s", 0.0)) <= 0.0:
        raise ValueError("COMSOL process wall time was not measured")
    if (
        float(process_metrics.get("fixed_rk4_step_s", math.nan)) != fixed_step_s
        or process_metrics.get("numerical_run_role") != numerical_run_role
    ):
        raise ValueError("COMSOL process metrics do not bind the numerical run")

    summary = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "COMPLETE_NORMALIZED_NOT_EVALUATED",
        "case_id": "caseP_100nm_three_current",
        "coordinate_system": "axisymmetric_rz_no_swirl",
        "particle_count": EXPECTED_PARTICLES,
        "output_count": EXPECTED_FRAMES,
        "observation_times_s": _output_times(),
        "fixed_rk4_step_s": fixed_step_s,
        "numerical_run_role": numerical_run_role,
        "brownian_active": False,
        "charge_revision": "aggregate_relative_drift_regularized_three_current_v1",
        "ion_drag_revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        "maximum_relative_ion_speed_m_s": 1.0e6,
        "release_authority": "candidate_owned_three_current_release_state",
        "reference_run_config_sha256": execution_inputs["reference_run_config_sha256"],
        "artifacts": {
            "trajectory_reference.csv": metrics["trajectory_reference_sha256"],
            "events_reference.csv": metrics["events_reference_sha256"],
            "trajectory_raw_wide.csv": metrics["trajectory_raw_wide_sha256"],
        },
        "trajectory_rows": metrics["trajectory_rows"],
        "event_count": metrics["event_count"],
        "final_lifecycle_counts": metrics["final_lifecycle_counts"],
        "initial_state_absolute_error_max": metrics["initial_state_absolute_error_max"],
        "claim_policy": {
            "solver_accuracy": "NOT_EVALUATED",
            "comsol_golden_truth": False,
            "universal_comsol_equivalence": "NOT_CLAIMED",
        },
    }
    (root / "normalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_receipt = {
        "schema_version": 1,
        "status": "PASS",
        "case_id": summary["case_id"],
        "reference_run_config_sha256": execution_inputs["reference_run_config_sha256"],
        "configuration": configuration,
        "solve": solve,
        "run": run,
        "execution_inputs": {
            "path": "execution_inputs.json",
            "sha256": _sha256(root / "execution_inputs.json"),
            "record": execution_inputs,
        },
        "process_metrics": process_metrics,
        "normalization_summary": {
            "path": "normalization_summary.json",
            "sha256": _sha256(root / "normalization_summary.json"),
        },
    }
    (root / "run_receipt.json").write_text(
        json.dumps(run_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    normalize(arguments.output_directory)


if __name__ == "__main__":
    main()
