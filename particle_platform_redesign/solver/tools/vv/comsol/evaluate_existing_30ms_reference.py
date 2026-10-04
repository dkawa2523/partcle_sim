"""Compare the converged 30 ms candidate with the saved COMSOL package.

This is an external V&V tool.  The saved COMSOL trajectory is treated as one
observed reference calculation, not as solver-core truth.  Candidate
self-convergence and cross-representation disagreement are reported
separately so that a field/model difference is not mislabelled as a time-step
error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

TOOL_REVISION: Final = "m3c1_existing_30ms_reference_evaluator_v1"
WORKFLOWS: Final = ("caseA", "caseP")
RUN_LABELS: Final = ("coarse", "medium", "fine")
STATE_WIDTH: Final = 5


@dataclass(frozen=True)
class Trace:
    state: np.ndarray
    active: np.ndarray
    present: np.ndarray
    sha256: str
    rows: int
    brownian_active_rows: int | None = None
    brownian_nonzero_rows: int | None = None
    brownian_max_force_n: float | None = None


@dataclass(frozen=True)
class Terminal:
    fate: str
    time_s: float | None
    position_m: tuple[float, float] | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _schedule(config: dict[str, Any]) -> np.ndarray:
    matrix = _mapping(config.get("matrix"), "matrix")
    segments = matrix.get("output_schedule_segments")
    if not isinstance(segments, list):
        raise ValueError("matrix.output_schedule_segments must be a list")
    values: list[float] = []
    for raw in segments:
        segment = _mapping(raw, "schedule segment")
        start = float(segment["start_s"])
        stop = float(segment["stop_s"])
        step = float(segment["step_s"])
        count = round((stop - start) / step)
        part = [start + index * step for index in range(count + 1)]
        part[-1] = stop
        if values and part[0] <= values[-1]:
            raise ValueError("output schedule must be strictly increasing")
        values.extend(part)
    result = np.asarray(values, dtype=np.float64)
    if result.size != int(matrix["output_count"]):
        raise ValueError("configured output count does not match the schedule")
    return result


def _time_index(times_s: np.ndarray, value: float, source: Path) -> int:
    index = int(np.argmin(np.abs(times_s - value)))
    tolerance = max(1.0e-15, 64.0 * math.ulp(max(abs(value), 1.0e-300)))
    if abs(float(times_s[index]) - value) > tolerance:
        raise ValueError(f"{source}: time {value:.17g} is outside the configured schedule")
    return index


def _empty_trace(particles: int, frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.full((particles, frames, STATE_WIDTH), np.nan, dtype=np.float64)
    active = np.zeros((particles, frames), dtype=np.bool_)
    present = np.zeros((particles, frames), dtype=np.bool_)
    return state, active, present


def _load_candidate(path: Path, particles: int, times_s: np.ndarray) -> Trace:
    state, active, present = _empty_trace(particles, times_s.size)
    required = {
        "particle_id",
        "time_s",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number_e",
        "lifecycle",
    }
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: candidate columns are incomplete")
        for row in reader:
            particle = int(row["particle_id"]) - 1
            frame = _time_index(times_s, float(row["time_s"]), path)
            if particle < 0 or particle >= particles or present[particle, frame]:
                raise ValueError(f"{path}: invalid or duplicate particle/time row")
            state[particle, frame] = (
                float(row["r_m"]),
                float(row["z_m"]),
                float(row["velocity_r_m_per_s"]),
                float(row["velocity_z_m_per_s"]),
                float(row["charge_number_e"]),
            )
            active[particle, frame] = row["lifecycle"] == "active"
            present[particle, frame] = True
            rows += 1
    if not present[:, 0].all() or not np.isfinite(state[present]).all():
        raise ValueError(f"{path}: candidate initial coverage or finite-state check failed")
    return Trace(state=state, active=active, present=present, sha256=_sha256(path), rows=rows)


def _load_reference(path: Path, particles: int, times_s: np.ndarray) -> Trace:
    state, active, present = _empty_trace(particles, times_s.size)
    required = {
        "particle_id",
        "time_s",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number_e",
        "current_status_code",
        "Brownian_force_magnitude_N",
    }
    rows = 0
    brownian_active_rows = 0
    brownian_nonzero_rows = 0
    brownian_max_force_n = 0.0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: reference columns are incomplete")
        for row in reader:
            particle = int(row["particle_id"]) - 1
            frame = _time_index(times_s, float(row["time_s"]), path)
            if particle < 0 or particle >= particles or present[particle, frame]:
                raise ValueError(f"{path}: invalid or duplicate particle/time row")
            state[particle, frame] = (
                float(row["r_m"]),
                float(row["z_m"]),
                float(row["velocity_r_m_per_s"]),
                float(row["velocity_z_m_per_s"]),
                float(row["charge_number_e"]),
            )
            active_row = int(float(row["current_status_code"])) == 1
            active[particle, frame] = active_row
            brownian_force_n = abs(float(row["Brownian_force_magnitude_N"]))
            if active_row:
                brownian_active_rows += 1
                brownian_nonzero_rows += brownian_force_n > 0.0
                brownian_max_force_n = max(brownian_max_force_n, brownian_force_n)
            present[particle, frame] = True
            rows += 1
    if (
        not present.all()
        or not np.isfinite(state[:, 0]).all()
        or not np.isfinite(state[active]).all()
    ):
        raise ValueError(f"{path}: reference coverage or finite-state check failed")
    return Trace(
        state=state,
        active=active,
        present=present,
        sha256=_sha256(path),
        rows=rows,
        brownian_active_rows=brownian_active_rows,
        brownian_nonzero_rows=brownian_nonzero_rows,
        brownian_max_force_n=brownian_max_force_n,
    )


def _magnitudes(first: np.ndarray, second: np.ndarray, quantity: str) -> np.ndarray:
    if quantity == "position":
        return np.linalg.norm(first[:, :2] - second[:, :2], axis=1)
    if quantity == "velocity":
        return np.linalg.norm(first[:, 2:4] - second[:, 2:4], axis=1)
    if quantity == "charge":
        return np.abs(first[:, 4] - second[:, 4])
    raise ValueError(f"unknown quantity {quantity}")


def _scale(second: np.ndarray, initial: np.ndarray, quantity: str) -> np.ndarray:
    if quantity == "position":
        return np.linalg.norm(second[:, :2] - initial[:, :2], axis=1)
    if quantity == "velocity":
        return np.linalg.norm(second[:, 2:4], axis=1)
    if quantity == "charge":
        return np.abs(second[:, 4])
    raise ValueError(f"unknown quantity {quantity}")


def _metrics(
    first: Trace,
    second: Trace,
    mask: np.ndarray,
    quantity: str,
) -> dict[str, float | int | None]:
    first_values = first.state[mask]
    second_values = second.state[mask]
    differences = _magnitudes(first_values, second_values, quantity)
    if differences.size == 0:
        return {"count": 0, "rms": None, "maximum": None, "relative_l2": None}
    particle_index = np.nonzero(mask)[0]
    initial = second.state[particle_index, np.zeros(particle_index.size, dtype=np.int64)]
    denominator = _scale(second_values, initial, quantity)
    denominator_l2 = float(np.linalg.norm(denominator))
    return {
        "count": int(differences.size),
        "rms": float(np.sqrt(np.mean(differences * differences))),
        "maximum": float(np.max(differences)),
        "p50": float(np.percentile(differences, 50.0)),
        "p90": float(np.percentile(differences, 90.0)),
        "p99": float(np.percentile(differences, 99.0)),
        "relative_l2": (
            float(np.linalg.norm(differences) / denominator_l2) if denominator_l2 > 0.0 else None
        ),
    }


def _observed_order(
    coarse: dict[str, float | int | None],
    fine: dict[str, float | int | None],
) -> float | None:
    coarse_rms = coarse.get("rms")
    fine_rms = fine.get("rms")
    if not isinstance(coarse_rms, float) or not isinstance(fine_rms, float):
        return None
    if coarse_rms <= 0.0 or fine_rms <= 0.0:
        return None
    return math.log2(coarse_rms / fine_rms)


def _difference_ratio(
    cross: dict[str, float | int | None],
    fine_pair: dict[str, float | int | None],
) -> float | None:
    cross_rms = cross.get("rms")
    fine_rms = fine_pair.get("rms")
    if not isinstance(cross_rms, float) or not isinstance(fine_rms, float) or fine_rms <= 0.0:
        return None
    return cross_rms / fine_rms


def _candidate_terminals(path: Path, particle_count: int) -> dict[int, Terminal]:
    result = {particle: Terminal("active", None, None) for particle in range(1, particle_count + 1)}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            particle = int(row["particle_id"])
            if result[particle].fate != "active":
                raise ValueError(f"{path}: multiple terminal events for particle {particle}")
            result[particle] = Terminal(
                fate=row["outcome"],
                time_s=float(row["event_time_s"]),
                position_m=(float(row["hit_r_m"]), float(row["hit_z_m"])),
            )
    return result


def _reference_fate(code: int) -> str:
    try:
        return {1: "active", 2: "held", 3: "stuck", 4: "escaped"}[code]
    except KeyError as error:
        raise ValueError(f"unsupported COMSOL status code {code}") from error


def _reference_terminals(path: Path, particle_count: int) -> tuple[dict[int, Terminal], str]:
    result: dict[int, Terminal] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            particle = int(row["particle_id"])
            code = int(float(row["final_status_code"]))
            fate = _reference_fate(code)
            time_s = None if fate == "active" else float(row["stop_or_event_time_s"])
            position = None
            if fate != "active":
                coordinates = (float(row["r_m"]), float(row["z_m"]))
                if all(math.isfinite(value) for value in coordinates):
                    position = coordinates
            if particle in result:
                raise ValueError(f"{path}: duplicate final particle {particle}")
            result[particle] = Terminal(fate, time_s, position)
    if sorted(result) != list(range(1, particle_count + 1)):
        raise ValueError(f"{path}: final particle IDs are incomplete")
    return result, _sha256(path)


def _terminal_comparison(
    candidate: dict[int, Terminal], reference: dict[int, Terminal]
) -> dict[str, object]:
    confusion = Counter((reference[p].fate, candidate[p].fate) for p in sorted(reference))
    same_terminal = [
        p
        for p in sorted(reference)
        if reference[p].fate == candidate[p].fate and reference[p].fate != "active"
    ]
    time_values: list[float] = []
    position_values: list[float] = []
    for particle in same_terminal:
        candidate_value = candidate[particle]
        reference_value = reference[particle]
        if candidate_value.time_s is None or reference_value.time_s is None:
            raise ValueError("matched terminal states must have event times")
        time_values.append(abs(candidate_value.time_s - reference_value.time_s))
        if candidate_value.position_m is not None and reference_value.position_m is not None:
            position_values.append(
                math.dist(candidate_value.position_m, reference_value.position_m)
            )
    time_difference = np.asarray(time_values, dtype=np.float64)
    position_difference = np.asarray(position_values, dtype=np.float64)
    return {
        "candidate_fate_counts": dict(sorted(Counter(x.fate for x in candidate.values()).items())),
        "reference_fate_counts": dict(sorted(Counter(x.fate for x in reference.values()).items())),
        "exact_fate_matches": sum(candidate[p].fate == reference[p].fate for p in reference),
        "particle_count": len(reference),
        "confusion_reference_to_candidate": {
            f"{reference_fate}->{candidate_fate}": count
            for (reference_fate, candidate_fate), count in sorted(confusion.items())
        },
        "same_terminal_fate_count": len(same_terminal),
        "same_terminal_time_difference_s": _array_summary(time_difference),
        "same_terminal_position_difference_m": _array_summary(position_difference),
    }


def _array_summary(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "rms": None, "maximum": None}
    return {
        "count": int(values.size),
        "rms": float(np.sqrt(np.mean(values * values))),
        "maximum": float(np.max(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p99": float(np.percentile(values, 99.0)),
    }


def _solver_settings(path: Path) -> tuple[dict[str, str], str]:
    selected: dict[str, str] = {}
    wanted = {
        ("solver_summary", "integration_method"),
        ("solver_summary", "fixed_step"),
        ("solver_summary", "relative_tolerance"),
        ("solver_summary", "stored_time_range"),
        ("time", "tauto"),
        ("time", "timeadaption"),
        ("time", "rtol"),
    }
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = (row["feature_tag"], row["property"])
            if key in wanted:
                selected[f"{key[0]}.{key[1]}"] = row["value"]
    if len(selected) != len(wanted):
        raise ValueError(f"{path}: required solver settings are incomplete")
    fixed_manual = (
        selected["solver_summary.integration_method"] == "explicit fixed RK4"
        and "10 us" in selected["solver_summary.fixed_step"]
        and selected["time.tauto"] == "manual"
        and selected["time.timeadaption"] == "none"
    )
    selected["classification"] = (
        "EXPLICIT_FIXED_RK4_10US_MANUAL" if fixed_manual else "OTHER_OR_UNRESOLVED"
    )
    return selected, _sha256(path)


def _brownian_feature(path: Path) -> tuple[dict[str, object], str]:
    labels: set[str] = set()
    seeds: set[str] = set()
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["feature_tag"] != "bf1" or "Brownian" not in row["feature_label"]:
                continue
            labels.add(row["feature_label"])
            if row["property"] == "i":
                seeds.add(row["value"])
            rows += 1
    if rows == 0:
        raise ValueError(f"{path}: Brownian feature bf1 is missing")
    return {
        "feature_tag": "bf1",
        "feature_labels": sorted(labels),
        "seed_expressions": sorted(seeds),
        "settings_rows": rows,
    }, _sha256(path)


def _candidate_execution_cost(result_directory: Path) -> dict[str, object]:
    """Read durable-run work counts without treating file times as a benchmark."""
    run_path = result_directory / "run.json"
    run = _mapping(json.loads(run_path.read_text(encoding="utf-8")), "candidate run")
    counts = _mapping(run["counts"], "candidate run counts")
    refinement = _mapping(run["event_refinement"], "candidate event refinement")
    segment_files = sorted((result_directory / "segments").glob("*.h5"))
    checkpoint_files = sorted((result_directory / "checkpoints").glob("*.h5"))
    segment_mtimes = [path.stat().st_mtime for path in segment_files]
    return {
        "run_manifest_sha256": _sha256(run_path),
        "macro_steps": int(counts["macro_steps"]),
        "accepted_particle_pieces": int(refinement["accepted_particle_pieces"]),
        "candidate_queries": int(refinement["candidate_queries"]),
        "epoch_macro_steps": int(run["epoch_macro_steps"]),
        "maximum_dt_charge_lipschitz": float(run["maximum_dt_charge_lipschitz"]),
        "segment_file_count": len(segment_files),
        "segment_bytes": sum(path.stat().st_size for path in segment_files),
        "checkpoint_file_count": len(checkpoint_files),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_files),
        "artifact_write_span_s_not_a_benchmark": (
            max(segment_mtimes) - min(segment_mtimes) if segment_mtimes else None
        ),
        "timing_interpretation": (
            "filesystem timestamp span only; candidate cells overlapped, so this is not an "
            "isolated wall-time benchmark"
        ),
    }


def _initial_state(candidate: Trace, reference: Trace) -> dict[str, float]:
    difference = np.abs(candidate.state[:, 0] - reference.state[:, 0])
    return {
        "position_max_m": float(np.max(difference[:, :2])),
        "velocity_max_m_per_s": float(np.max(difference[:, 2:4])),
        "charge_max_e": float(np.max(difference[:, 4])),
    }


def _time_history(
    times_s: np.ndarray,
    fine: Trace,
    reference: Trace,
    candidate_terminal: dict[int, Terminal],
    reference_terminal: dict[int, Terminal],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for frame, time_s in enumerate(times_s):
        mask = fine.present[:, frame] & fine.active[:, frame] & reference.active[:, frame]
        position = _metrics(fine, reference, _frame_mask(mask, frame, times_s.size), "position")
        mismatch = 0
        for particle in range(1, fine.state.shape[0] + 1):
            candidate = candidate_terminal[particle]
            reference_value = reference_terminal[particle]
            candidate_fate = (
                "active"
                if candidate.time_s is None or time_s < candidate.time_s
                else candidate.fate
            )
            reference_fate = (
                "active"
                if reference_value.time_s is None or time_s < reference_value.time_s
                else reference_value.fate
            )
            mismatch += candidate_fate != reference_fate
        records.append(
            {
                "time_s": float(time_s),
                "common_active_count": int(np.count_nonzero(mask)),
                "position_rms_m": position["rms"],
                "position_max_m": position["maximum"],
                "status_mismatch_count": mismatch,
                "status_mismatch_fraction": mismatch / fine.state.shape[0],
            }
        )
    return records


def _frame_mask(mask: np.ndarray, frame: int, frame_count: int) -> np.ndarray:
    result = np.zeros((mask.size, frame_count), dtype=np.bool_)
    result[:, frame] = mask
    return result


def _evaluate_workflow(
    workflow: str,
    config: dict[str, Any],
    config_path: Path,
    candidate_root: Path,
    characterization: dict[str, Any],
    repository_root: Path,
    times_s: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    particle_count = int(_mapping(config["matrix"], "matrix")["particle_count"])
    workflow_config = _mapping(_mapping(config["workflows"], "workflows")[workflow], workflow)
    package = (repository_root / str(workflow_config["package_relative_path"])).resolve()
    history_path = package / "results/particle_history_full_tidy.csv"
    final_path = package / "results/particle_final_summary_tidy.csv"
    settings_path = package / "config/study_and_solver_settings.csv"
    physics_settings_path = package / "config/particle_physics_feature_settings.csv"
    expected = _mapping(workflow_config["expected_sha256"], "expected_sha256")
    if _sha256(history_path) != expected["results/particle_history_full_tidy.csv"]:
        raise ValueError(f"{workflow}: saved COMSOL history hash mismatch")

    candidates = {
        label: _load_candidate(
            candidate_root / workflow / f"candidate_trajectory_{label}.csv",
            particle_count,
            times_s,
        )
        for label in RUN_LABELS
    }
    reference = _load_reference(history_path, particle_count, times_s)
    source_workflow = _mapping(
        _mapping(characterization["source_receipts"], "source_receipts")["workflows"],
        "source workflows",
    )[workflow]
    source_runs = _mapping(_mapping(source_workflow, workflow)["runs"], "source runs")
    for label, trace in candidates.items():
        recorded = _mapping(source_runs[label], f"{workflow}.{label}")["trajectory_sha256"]
        if trace.sha256 != recorded:
            raise ValueError(f"{workflow} {label}: candidate trajectory hash mismatch")

    shared = reference.active.copy()
    for trace in candidates.values():
        shared &= trace.present & trace.active
    cross_mask = reference.active & candidates["fine"].present & candidates["fine"].active
    quantities: dict[str, object] = {}
    for quantity in ("position", "velocity", "charge"):
        coarse_medium = _metrics(candidates["coarse"], candidates["medium"], shared, quantity)
        medium_fine = _metrics(candidates["medium"], candidates["fine"], shared, quantity)
        cross = _metrics(candidates["fine"], reference, cross_mask, quantity)
        quantities[quantity] = {
            "candidate_coarse_medium": coarse_medium,
            "candidate_medium_fine": medium_fine,
            "candidate_observed_rms_order": _observed_order(coarse_medium, medium_fine),
            "candidate_fine_vs_saved_comsol": cross,
            "cross_rms_to_candidate_fine_pair_rms": _difference_ratio(cross, medium_fine),
        }

    candidate_terminal = _candidate_terminals(
        candidate_root / workflow / "candidate_events_fine.csv", particle_count
    )
    reference_terminal, final_hash = _reference_terminals(final_path, particle_count)
    settings, settings_hash = _solver_settings(settings_path)
    brownian_feature, physics_settings_hash = _brownian_feature(physics_settings_path)
    time_history = _time_history(
        times_s,
        candidates["fine"],
        reference,
        candidate_terminal,
        reference_terminal,
    )
    first_mismatch = next(
        (row["time_s"] for row in time_history if row["status_mismatch_count"]), None
    )
    report: dict[str, object] = {
        "package": str(package),
        "source_hashes": {
            "history_sha256": reference.sha256,
            "final_summary_sha256": final_hash,
            "solver_settings_sha256": settings_hash,
            "particle_physics_settings_sha256": physics_settings_hash,
            "candidate": {label: trace.sha256 for label, trace in candidates.items()},
            "configuration_sha256": _sha256(config_path),
        },
        "source_rows": {
            "saved_comsol": reference.rows,
            "candidate": {label: trace.rows for label, trace in candidates.items()},
        },
        "candidate_execution_cost": {
            label: _candidate_execution_cost(candidate_root / workflow / f"result_{label}")
            for label in RUN_LABELS
        },
        "saved_comsol_time_solver": settings,
        "physics_compatibility": {
            "candidate_brownian_active": False,
            "saved_comsol_brownian_feature": brownian_feature,
            "saved_comsol_active_rows": reference.brownian_active_rows,
            "saved_comsol_nonzero_brownian_force_rows": reference.brownian_nonzero_rows,
            "saved_comsol_maximum_brownian_force_N": reference.brownian_max_force_n,
            "stochastic_pathwise_comparison": "NOT_VALID_MODEL_AND_RNG_MISMATCH",
            "field_representation": "COMSOL_NATIVE_FE_VS_CANDIDATE_EXPORTED_P1",
        },
        "initial_state_difference": _initial_state(candidates["fine"], reference),
        "common_active_records": {
            "three_candidate_runs_and_saved_comsol": int(np.count_nonzero(shared)),
            "candidate_fine_and_saved_comsol": int(np.count_nonzero(cross_mask)),
        },
        "state_comparison": quantities,
        "terminal_comparison": _terminal_comparison(candidate_terminal, reference_terminal),
        "first_sampled_status_divergence_time_s": first_mismatch,
        "interpretation": {
            "candidate_time_convergence": "PASS",
            "saved_comsol_time_convergence": "NOT_ESTABLISHED_SINGLE_FIXED_STEP_RUN",
            "cross_representation_role": "CHARACTERIZATION_NOT_CORE_ACCEPTANCE_GATE",
            "pointwise_rhs_parity": "NOT_ESTABLISHED_NATIVE_FE_VS_EXPORTED_P1",
            "particlewise_path_parity": "NOT_VALID_BROWNIAN_ON_VS_OFF",
            "universal_comsol_equivalence": "NOT_CLAIMED",
        },
    }
    return report, time_history


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    config_path: Path,
    candidate_root: Path,
    characterization_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    config_path = config_path.resolve()
    candidate_root = candidate_root.resolve()
    characterization_path = characterization_path.resolve()
    config = _mapping(json.loads(config_path.read_text(encoding="utf-8")), "config")
    characterization = _mapping(
        json.loads(characterization_path.read_text(encoding="utf-8")), "characterization"
    )
    if characterization.get("status") != "PASS":
        raise ValueError("candidate characterization must report PASS")
    if _mapping(characterization["config"], "characterization.config")["sha256"] != _sha256(
        config_path
    ):
        raise ValueError("candidate characterization uses a different configuration")
    times_s = _schedule(config)
    repository_root = Path(__file__).resolve().parents[5]
    reports: dict[str, object] = {}
    histories: dict[str, list[dict[str, object]]] = {}
    for workflow in WORKFLOWS:
        reports[workflow], histories[workflow] = _evaluate_workflow(
            workflow,
            config,
            config_path,
            candidate_root,
            characterization,
            repository_root,
            times_s,
        )
    result: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "CHARACTERIZED",
        "decision": "SOLVER_CONVERGENCE_PASS_COMSOL_REFERENCE_NOT_A_CORE_GATE",
        "candidate_characterization": {
            "path": str(characterization_path),
            "sha256": _sha256(characterization_path),
            "status": characterization["status"],
        },
        "comparison_policy": {
            "primary_numerical_evidence": "candidate h/h2/h4 self-convergence",
            "saved_comsol_role": "external single-run cross-representation characterization",
            "saved_comsol_is_golden_truth": False,
            "tune_solver_core_to_reference": False,
        },
        "workflows": reports,
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline=""
    )
    for workflow, rows in histories.items():
        _write_csv(output_directory / f"{workflow}_time_history.csv", rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-characterization", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    result = evaluate(
        arguments.config,
        arguments.candidate_root,
        arguments.candidate_characterization,
        arguments.output_directory,
    )
    print(json.dumps({"status": result["status"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()
