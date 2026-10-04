"""Evaluate the locked M3-C1 pre-event cross-representation comparison.

COMSOL integrates its native finite-element fields.  The solver integrates a
canonical exact-connectivity P1 projection of exported nodal values.  This
tool names that representation difference, locks candidate trajectories to
their producer receipts, and makes no same-field accuracy claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

TOOL_REVISION: Final = "m3c1_cross_representation_pre_event_v4"
REPORT_SCHEMA_VERSION: Final = 3
STEP_NAMES: Final = ("0p625us", "0p3125us", "0p15625us")
RUN_KEYS: Final = tuple(f"dt_{name}" for name in STEP_NAMES)
_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
_STATE_COLUMNS: Final = _COLUMNS[2:7]
type SolverLabel = Literal["solver_canonical_p1_projection", "comsol_native_field"]


@dataclass(frozen=True)
class _Config:
    path: Path
    sha256: str
    raw: dict[str, object]
    labels: dict[str, str]
    particles: int
    frames: int
    output_interval_s: float
    time_window_s: tuple[float, float]
    steps_s: tuple[float, float, float]
    minimum_order: float
    relative_limits: dict[str, float]
    parity_factor: float
    roundoff_multiplier: float
    reference_hashes: dict[str, str]


@dataclass(frozen=True)
class _Trajectory:
    path: Path
    sha256: str
    values: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_json(path: Path, name: str) -> tuple[dict[str, object], Path, str]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{resolved}: {name} must be a JSON object")
    return value, resolved, _sha256(resolved)


def _load_config(path: Path) -> _Config:
    raw, resolved, digest = _load_json(path, "evaluation config")
    if raw.get("evaluation_id") != "M3-C1-caseA-100nm-pre-event":
        raise ValueError(f"{resolved}: unexpected evaluation_id")
    labels_raw = _mapping(raw.get("labels"), "labels")
    labels = {
        "candidate": _string(labels_raw.get("candidate"), "candidate label"),
        "reference": _string(labels_raw.get("reference"), "reference label"),
    }
    expected_labels = {
        "candidate": "solver_canonical_p1_projection",
        "reference": "comsol_native_field",
    }
    if labels != expected_labels:
        raise ValueError("config labels must preserve both field representations")
    scope = _mapping(raw.get("scope"), "scope")
    steps_raw = scope.get("internal_steps_s")
    if not isinstance(steps_raw, list) or len(steps_raw) != 3:
        raise ValueError("scope.internal_steps_s must have three entries")
    steps = (
        _number(steps_raw[0], "first internal step"),
        _number(steps_raw[1], "second internal step"),
        _number(steps_raw[2], "third internal step"),
    )
    if (
        any(step <= 0.0 for step in steps)
        or steps[0] != 2.0 * steps[1]
        or steps[1] != 2.0 * steps[2]
    ):
        raise ValueError("scope.internal_steps_s must be exact positive h, h/2, h/4")
    time_window_raw = scope.get("time_window_s")
    if not isinstance(time_window_raw, list) or len(time_window_raw) != 2:
        raise ValueError("scope.time_window_s must have start and end entries")
    time_window = (
        _number(time_window_raw[0], "time window start"),
        _number(time_window_raw[1], "time window end"),
    )
    if time_window[0] != 0.0:
        raise ValueError("scope.time_window_s must start at zero for this evaluator")
    if time_window[1] <= time_window[0]:
        raise ValueError("scope.time_window_s end must be after start")
    acceptance = _mapping(raw.get("acceptance"), "acceptance")
    limits_raw = _mapping(
        acceptance.get("maximum_fine_pair_relative_l2"),
        "maximum_fine_pair_relative_l2",
    )
    limits = {
        name: _number(limits_raw.get(name), f"{name} relative limit")
        for name in ("position", "velocity", "charge")
    }
    reference = _mapping(raw.get("reference"), "reference")
    hashes_raw = _mapping(
        reference.get("trajectory_sha256_by_step"),
        "reference trajectory hashes",
    )
    hashes = {name: _string(hashes_raw.get(name), f"reference {name} hash") for name in STEP_NAMES}
    return _Config(
        path=resolved,
        sha256=digest,
        raw=raw,
        labels=labels,
        particles=_positive_int(scope.get("particles"), "scope.particles"),
        frames=_positive_int(scope.get("frames"), "scope.frames"),
        output_interval_s=_number(scope.get("output_interval_s"), "output interval"),
        time_window_s=time_window,
        steps_s=steps,
        minimum_order=_number(acceptance.get("minimum_rms_order"), "minimum order"),
        relative_limits=limits,
        parity_factor=_number(
            acceptance.get("candidate_precision_vs_reference_factor"),
            "precision parity factor",
        ),
        roundoff_multiplier=_number(
            acceptance.get("roundoff_multiplier"),
            "roundoff multiplier",
        ),
        reference_hashes=hashes,
    )


def _read_trajectory(path: Path, config: _Config) -> _Trajectory:
    resolved = path.expanduser().resolve()
    values = np.empty((config.frames, config.particles, 5), dtype=np.float64)
    seen = np.zeros((config.frames, config.particles), dtype=np.bool_)
    with resolved.open(encoding="utf-8-sig", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != _COLUMNS:
            raise ValueError(f"{resolved}: trajectory columns must be exactly {_COLUMNS}")
        for line_number, row in enumerate(reader, start=2):
            try:
                particle_id = int(row["particle_id"])
                time_s = float(row["time_s"])
                state = np.asarray([float(row[name]) for name in _STATE_COLUMNS])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{resolved}:{line_number}: invalid numeric value") from error
            if not 1 <= particle_id <= config.particles:
                raise ValueError(f"{resolved}:{line_number}: particle ID outside expected range")
            frame = round(time_s / config.output_interval_s)
            expected_time = frame * config.output_interval_s
            tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(time_s))
            if not 0 <= frame < config.frames or abs(time_s - expected_time) > tolerance:
                raise ValueError(f"{resolved}:{line_number}: time is not on the output grid")
            particle = particle_id - 1
            if seen[frame, particle]:
                raise ValueError(f"{resolved}:{line_number}: duplicate particle/time key")
            if not np.isfinite(state).all():
                raise ValueError(f"{resolved}:{line_number}: nonfinite state")
            if row["lifecycle"] != "active":
                raise ValueError(f"{resolved}:{line_number}: pre-event state must be active")
            seen[frame, particle] = True
            values[frame, particle] = state
    if not seen.all():
        frame, particle = np.argwhere(~seen)[0]
        raise ValueError(f"{resolved}: missing frame={int(frame)}, particle={int(particle) + 1}")
    return _Trajectory(resolved, _sha256(resolved), values)


def _receipt_path(root: Path, value: object, name: str) -> Path:
    relative = Path(_string(value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be receipt-relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes its receipt directory")
    return resolved


def _verify_hash(path: Path, expected: object, name: str) -> str:
    expected_hash = _string(expected, f"{name} sha256")
    observed = _sha256(path)
    if observed != expected_hash:
        raise ValueError(f"{name} hash differs from its producer receipt")
    return observed


def _revisions(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in manifest.items()
        if key.endswith("_revision") and value is not None
    }


def _verify_result_execution(
    manifest: dict[str, object],
    run_key: str,
    case_hash: str,
    expected_dt_s: float,
    time_window_s: tuple[float, float],
) -> None:
    if manifest.get("status") != "complete":
        raise ValueError(f"result manifest {run_key} is incomplete")
    if manifest.get("case_file_hash") != f"sha256:{case_hash}":
        raise ValueError(f"result manifest {run_key} case_file_hash differs")
    time = _mapping(manifest.get("time"), f"result manifest {run_key} time")
    if _number(time.get("dt_s"), f"result manifest {run_key} time.dt_s") != expected_dt_s:
        raise ValueError(f"result manifest {run_key} time.dt_s differs")
    if _number(time.get("start_s"), f"result manifest {run_key} time.start_s") != time_window_s[0]:
        raise ValueError(f"result manifest {run_key} time.start_s differs")
    if _number(time.get("end_s"), f"result manifest {run_key} time.end_s") != time_window_s[1]:
        raise ValueError(f"result manifest {run_key} time.end_s differs")


def _verify_candidate_receipts(
    run_report_path: Path,
    prepare_report_path: Path,
    trajectories: dict[str, _Trajectory],
    config: _Config,
) -> dict[str, object]:
    """Verify one candidate producer receipt without defining a generic framework."""

    run, run_path, run_hash = _load_json(run_report_path, "candidate run report")
    prepare, prepare_path, prepare_hash = _load_json(prepare_report_path, "prepare report")
    if run_path.parent != prepare_path.parent:
        raise ValueError("candidate and prepare reports must share one directory")
    if run.get("tool_revision") != prepare.get("tool_revision"):
        raise ValueError("candidate and prepare tool revisions differ")
    if run.get("status") != "complete" or run.get("decision") != "COMPLETE":
        raise ValueError("candidate run receipt is blocked or incomplete")
    root = run_path.parent
    input_path = _receipt_path(root, prepare.get("input_data"), "candidate input")
    input_hash = _verify_hash(input_path, prepare.get("input_file_sha256"), "candidate input")
    prepared_cases = _mapping(prepare.get("cases"), "prepare cases")
    runs = _mapping(run.get("runs"), "candidate runs")
    bindings: dict[str, object] = {}
    for step, run_key, expected_dt_s in zip(
        STEP_NAMES,
        RUN_KEYS,
        config.steps_s,
        strict=True,
    ):
        record = _mapping(runs.get(run_key), f"candidate run {run_key}")
        if record.get("status") != "complete" or record.get("decision") != "COMPLETE":
            raise ValueError(f"candidate run {run_key} is incomplete")
        if record.get("trajectory_rows") != config.particles * config.frames:
            raise ValueError(f"candidate run {run_key} has no complete trajectory matrix")
        case_name = prepared_cases.get(run_key)
        if record.get("case") != case_name:
            raise ValueError(f"candidate run {run_key} does not use its prepared case")
        case_path = _receipt_path(root, case_name, f"candidate case {run_key}")
        case_hash = _verify_hash(case_path, record.get("case_sha256"), f"candidate case {run_key}")
        result_path = _receipt_path(root, record.get("result"), f"result {run_key}")
        manifest_path = result_path / "run.json"
        manifest, _, manifest_hash = _load_json(manifest_path, f"result manifest {run_key}")
        _verify_hash(
            manifest_path,
            record.get("result_manifest_sha256"),
            f"result manifest {run_key}",
        )
        _verify_result_execution(
            manifest,
            run_key,
            case_hash,
            expected_dt_s,
            config.time_window_s,
        )
        if manifest.get("data_content_hash") != prepare.get("input_content_hash"):
            raise ValueError(f"result manifest {run_key} uses another candidate input")
        if record.get("engine_algorithm_revision") != manifest.get("engine_algorithm_revision"):
            raise ValueError(f"candidate run {run_key} engine revision differs")
        resolved = _mapping(manifest.get("resolved"), "resolved manifest")
        if record.get("physics_models") != resolved.get("physics_models"):
            raise ValueError(f"candidate run {run_key} physics revisions differ")
        trajectory = trajectories[step]
        trajectory_path = _receipt_path(
            root,
            record.get("trajectory"),
            f"candidate trajectory {run_key}",
        )
        if trajectory.path != trajectory_path:
            raise ValueError(f"candidate trajectory {run_key} is not the receipt artifact")
        _verify_hash(
            trajectory_path,
            record.get("trajectory_sha256"),
            f"candidate trajectory {run_key}",
        )
        bindings[step] = {
            "case": {"path": str(case_path), "sha256": case_hash},
            "result_manifest": {"path": str(manifest_path), "sha256": manifest_hash},
            "trajectory": {"path": str(trajectory.path), "sha256": trajectory.sha256},
            "revisions": _revisions(manifest),
        }
    return {
        "candidate_run_report": {"path": str(run_path), "sha256": run_hash},
        "prepare_report": {"path": str(prepare_path), "sha256": prepare_hash},
        "input_h5": {"path": str(input_path), "sha256": input_hash},
        "runs": bindings,
    }


def _quantity_metrics(
    difference: np.ndarray,
    scale_values: np.ndarray,
    output_interval_s: float,
) -> dict[str, object]:
    magnitude = np.abs(difference) if difference.ndim == 2 else np.linalg.norm(difference, axis=2)
    scale = np.abs(scale_values) if scale_values.ndim == 2 else np.linalg.norm(scale_values, axis=2)
    square_sum = float(np.sum(magnitude * magnitude, dtype=np.float64))
    scale_square_sum = float(np.sum(scale * scale, dtype=np.float64))
    maximum_index = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    particle_maximum = np.max(magnitude, axis=0)
    particle_rms = np.sqrt(np.mean(magnitude * magnitude, axis=0))
    worst_particles = np.argsort(particle_maximum)[-10:][::-1]
    per_time = []
    for frame, row in enumerate(magnitude):
        worst = int(np.argmax(row))
        per_time.append(
            {
                "frame": frame,
                "time_s": frame * output_interval_s,
                "rms": float(np.sqrt(np.mean(row * row))),
                "p99": float(np.quantile(row, 0.99)),
                "maximum": float(row[worst]),
                "worst_particle_id": worst + 1,
            }
        )
    return {
        "count": int(magnitude.size),
        "rms": math.sqrt(square_sum / magnitude.size),
        "maximum": float(magnitude[maximum_index]),
        "relative_l2": math.sqrt(square_sum / scale_square_sum) if scale_square_sum > 0.0 else None,
        "percentiles": {
            f"p{percent}": float(np.quantile(magnitude, percent / 100.0))
            for percent in (50, 90, 95, 99)
        },
        "maximum_location": {
            "particle_id": int(maximum_index[1]) + 1,
            "frame": int(maximum_index[0]),
            "time_s": int(maximum_index[0]) * output_interval_s,
        },
        "per_time": per_time,
        "worst_particles": [
            {
                "particle_id": int(index) + 1,
                "rms": float(particle_rms[index]),
                "maximum": float(particle_maximum[index]),
            }
            for index in worst_particles
        ],
    }


def _difference_metrics(
    first: np.ndarray, second: np.ndarray, interval: float
) -> dict[str, object]:
    difference = first - second
    displacement = second[:, :, :2] - second[0:1, :, :2]
    return {
        "position": _quantity_metrics(difference[:, :, :2], displacement, interval),
        "velocity": _quantity_metrics(difference[:, :, 2:4], second[:, :, 2:4], interval),
        "charge": _quantity_metrics(difference[:, :, 4], second[:, :, 4], interval),
        "position_relative_l2_scale": "fine_displacement_from_each_particle_initial_position",
        "velocity_relative_l2_scale": "fine_velocity",
        "charge_relative_l2_scale": "fine_charge_number",
    }


def _order(coarse_fine: float, fine_finer: float) -> float | None:
    if coarse_fine <= 0.0 or fine_finer <= 0.0:
        return None
    return math.log2(coarse_fine / fine_finer)


def _assess_self_convergence(
    coarse_medium: dict[str, object],
    medium_fine: dict[str, object],
    representation_scale: dict[str, float],
    config: _Config,
) -> tuple[dict[str, float | None], dict[str, object], list[str]]:
    orders: dict[str, float | None] = {}
    assessments: dict[str, object] = {}
    blockers: list[str] = []
    for quantity in ("position", "velocity", "charge"):
        first_metrics = _mapping(coarse_medium[quantity], quantity)
        second_metrics = _mapping(medium_fine[quantity], quantity)
        first = _number(
            first_metrics.get("rms"),
            "coarse rms",
        )
        second = _number(
            second_metrics.get("rms"),
            "fine rms",
        )
        first_maximum = _number(first_metrics.get("maximum"), "coarse maximum")
        second_maximum = _number(second_metrics.get("maximum"), "fine maximum")
        observed = _order(first, second)
        orders[quantity] = observed
        scale = representation_scale[quantity]
        roundoff_floor = (
            config.roundoff_multiplier * np.finfo(np.float64).eps * max(scale, 1.0e-300)
        )
        roundoff_limited = all(
            value <= roundoff_floor for value in (first, first_maximum, second, second_maximum)
        )
        order_passed = observed is not None and observed >= config.minimum_order
        if roundoff_limited:
            reason = "both_adjacent_pair_rms_and_maximum_at_or_below_roundoff_floor"
            classification = "ROUNDOFF_LIMITED"
            order_evaluated = False
            effective_order: float | str | None = "NOT_EVALUATED_ROUNDOFF_PLATEAU"
            assessment_passed = True
        else:
            classification = "ORDER_EVALUATED"
            order_evaluated = True
            effective_order = observed
            assessment_passed = order_passed
            reason = (
                "observed_order_meets_minimum"
                if order_passed
                else "observed_order_unavailable_or_below_minimum"
            )
        assessments[quantity] = {
            "classification": classification,
            "order_evaluated": order_evaluated,
            "observed_order": observed,
            "effective_order": effective_order,
            "minimum_required_order": config.minimum_order,
            "representation_scale": scale,
            "roundoff_floor": roundoff_floor,
            "coarse_pair_rms": first,
            "coarse_pair_maximum": first_maximum,
            "fine_pair_rms": second,
            "fine_pair_maximum": second_maximum,
            "pass": assessment_passed,
            "reason": reason,
        }
        if not assessment_passed:
            blockers.append(f"{quantity}.rms observed order is below {config.minimum_order}")
        relative = second_metrics.get("relative_l2")
        if (
            not isinstance(relative, int | float)
            or float(relative) > config.relative_limits[quantity]
        ):
            blockers.append(f"{quantity}.relative_l2 exceeds its configured limit")
    return orders, assessments, blockers


def characterize(
    config_path: Path,
    label: SolverLabel,
    coarse_path: Path,
    medium_path: Path,
    fine_path: Path,
    *,
    candidate_run_report: Path | None = None,
    prepare_report: Path | None = None,
) -> dict[str, object]:
    """Characterize one representation without reading cross differences."""

    config = _load_config(config_path)
    if label not in config.labels.values():
        raise ValueError("label is not owned by the evaluation config")
    trajectories = {
        name: _read_trajectory(path, config)
        for name, path in zip(STEP_NAMES, (coarse_path, medium_path, fine_path), strict=True)
    }
    if label == config.labels["candidate"]:
        if candidate_run_report is None or prepare_report is None:
            raise ValueError("candidate characterization requires both producer receipts")
        provenance = _verify_candidate_receipts(
            candidate_run_report,
            prepare_report,
            trajectories,
            config,
        )
    else:
        for name, trajectory in trajectories.items():
            if trajectory.sha256 != config.reference_hashes[name]:
                raise ValueError(f"COMSOL reference {name} differs from its locked hash")
        provenance = {"locked_reference": _mapping(config.raw.get("reference"), "reference")}
    initial = [item.values[0] for item in trajectories.values()]
    if not all(np.array_equal(initial[0], state) for state in initial[1:]):
        raise ValueError("self-convergence trajectories do not share one exact initial state")
    coarse_medium = _difference_metrics(
        trajectories[STEP_NAMES[0]].values,
        trajectories[STEP_NAMES[1]].values,
        config.output_interval_s,
    )
    medium_fine = _difference_metrics(
        trajectories[STEP_NAMES[1]].values,
        trajectories[STEP_NAMES[2]].values,
        config.output_interval_s,
    )
    fine = trajectories[STEP_NAMES[2]].values
    representation_scale = {
        "position": float(np.max(np.abs(fine[:, :, :2]))),
        "velocity": float(np.max(np.linalg.norm(fine[:, :, 2:4], axis=2))),
        "charge": float(np.max(np.abs(fine[:, :, 4]))),
    }
    orders, order_assessment, blockers = _assess_self_convergence(
        coarse_medium,
        medium_fine,
        representation_scale,
        config,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool_revision": TOOL_REVISION,
        "report_kind": "m3c1_single_representation_self_convergence",
        "label": label,
        "status": "PASS" if not blockers else "BLOCKED",
        "evaluation_config": {"path": str(config.path), "sha256": config.sha256},
        "scope": config.raw["scope"],
        "source_scope": {
            "comparison": "COMSOL native FE versus solver canonical exported-nodal P1",
            "provenance": provenance,
        },
        "artifacts": {
            name: {"path": str(item.path), "sha256": item.sha256}
            for name, item in trajectories.items()
        },
        "initial_state_sha256": hashlib.sha256(fine[0].tobytes(order="C")).hexdigest(),
        "representation_scale": {
            "position_m": representation_scale["position"],
            "velocity_m_per_s": representation_scale["velocity"],
            "charge_number_e": representation_scale["charge"],
        },
        f"{STEP_NAMES[0]}_vs_{STEP_NAMES[1]}": coarse_medium,
        f"{STEP_NAMES[1]}_vs_{STEP_NAMES[2]}": medium_fine,
        "observed_rms_order": orders,
        "rms_order_assessment": order_assessment,
        "acceptance": {
            "minimum_rms_order": config.minimum_order,
            "maximum_fine_pair_relative_l2": config.relative_limits,
            "roundoff_multiplier": config.roundoff_multiplier,
            "roundoff_plateau_rule": (
                "both adjacent-pair RMS and maximum must not exceed the "
                "float64 representation-scale floor"
            ),
            "roundoff_plateau_interpretation": (
                "precision stability only; integrator order and error below the "
                "roundoff floor are not established"
            ),
        },
        "blockers": blockers,
        "claim_separation": {
            "frozen_field_rhs": "SEPARATE_ARTIFACT_REQUIRED",
            "cross_representation_trajectory": "SELF_CONVERGENCE_ONLY",
            "same_field_solver_agreement": "NOT_TESTED",
            "roundoff_limited_order": "NOT_ESTABLISHED_BELOW_ROUNDOFF_FLOOR",
        },
    }


def _load_report(path: Path, kind: str) -> tuple[dict[str, object], str]:
    report, resolved, digest = _load_json(path, "evaluation report")
    if report.get("report_kind") != kind or report.get("tool_revision") != TOOL_REVISION:
        raise ValueError(f"{resolved}: incompatible report")
    return report, digest


def _fine_metric(report: dict[str, object], quantity: str, metric: str) -> float:
    pair = _mapping(report.get(f"{STEP_NAMES[1]}_vs_{STEP_NAMES[2]}"), "fine pair")
    return _number(_mapping(pair.get(quantity), quantity).get(metric), f"{quantity}.{metric}")


def _fine_artifact(report: dict[str, object]) -> dict[str, str]:
    artifact = _mapping(
        _mapping(report.get("artifacts"), "artifacts").get(STEP_NAMES[2]),
        "fine artifact",
    )
    return {
        "path": _string(artifact.get("path"), "fine artifact path"),
        "sha256": _string(artifact.get("sha256"), "fine artifact hash"),
    }


def register_budget(
    config_path: Path, candidate_path: Path, reference_path: Path
) -> dict[str, object]:
    """Register a cross-representation envelope before reading differences."""

    config = _load_config(config_path)
    kind = "m3c1_single_representation_self_convergence"
    candidate, candidate_hash = _load_report(candidate_path, kind)
    reference, reference_hash = _load_report(reference_path, kind)
    blockers: list[str] = []
    for side, report in (("candidate", candidate), ("reference", reference)):
        if report.get("label") != config.labels[side]:
            blockers.append(f"{side} report label is invalid")
        if report.get("status") != "PASS":
            blockers.append(f"{side} self-convergence did not pass")
        if (
            _mapping(report.get("evaluation_config"), "evaluation config").get("sha256")
            != config.sha256
        ):
            blockers.append(f"{side} used another evaluation config")
    if candidate.get("scope") != reference.get("scope"):
        blockers.append("candidate and reference scopes differ")
    if candidate.get("initial_state_sha256") != reference.get("initial_state_sha256"):
        blockers.append("initial-state hashes differ")
    envelopes: dict[str, object] = {}
    precision: dict[str, object] = {}
    scale_names = {
        "position": "position_m",
        "velocity": "velocity_m_per_s",
        "charge": "charge_number_e",
    }
    for quantity, scale_key in scale_names.items():
        candidate_scale = _mapping(candidate.get("representation_scale"), "candidate scale")
        reference_scale = _mapping(reference.get("representation_scale"), "reference scale")
        scale = max(
            _number(candidate_scale.get(scale_key), scale_key),
            _number(reference_scale.get(scale_key), scale_key),
        )
        roundoff = config.roundoff_multiplier * np.finfo(np.float64).eps * max(scale, 1.0e-300)
        quantity_envelope: dict[str, float] = {}
        quantity_precision: dict[str, object] = {}
        for metric in ("rms", "maximum"):
            candidate_change = _fine_metric(candidate, quantity, metric)
            reference_change = _fine_metric(reference, quantity, metric)
            limit = config.parity_factor * reference_change + roundoff
            passed = bool(candidate_change <= limit)
            quantity_envelope[metric] = candidate_change + reference_change + roundoff
            quantity_precision[metric] = {
                "candidate_fine_step_change": candidate_change,
                "reference_fine_step_change": reference_change,
                "limit": limit,
                "pass": passed,
            }
            if not passed:
                blockers.append(f"candidate {quantity}.{metric} is not at reference precision")
        envelopes[quantity] = quantity_envelope
        precision[quantity] = quantity_precision
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool_revision": TOOL_REVISION,
        "report_kind": "m3c1_preregistered_cross_representation_budget",
        "status": "REGISTERED" if not blockers else "BLOCKED",
        "evaluation_config": {"path": str(config.path), "sha256": config.sha256},
        "scope": candidate.get("scope"),
        "policy": {
            "representations": "DIFFERENT_BY_DESIGN",
            "result_dependent_tolerance_tuning": "PROHIBITED",
        },
        "self_convergence_reports": {
            "candidate": {"path": str(candidate_path.resolve()), "sha256": candidate_hash},
            "reference": {"path": str(reference_path.resolve()), "sha256": reference_hash},
        },
        "fine_trajectory_artifacts": {
            "candidate": _fine_artifact(candidate),
            "reference": _fine_artifact(reference),
        },
        "envelopes": envelopes,
        "candidate_precision_vs_reference": precision,
        "blockers": blockers,
        "claim_separation": {
            "cross_representation_trajectory": "BUDGET_REGISTERED" if not blockers else "BLOCKED",
            "same_field_solver_agreement": "NOT_TESTED",
        },
    }


def _registered_trajectory(
    budget: dict[str, object], side: str, supplied: Path, config: _Config
) -> _Trajectory:
    identity = _mapping(
        _mapping(budget.get("fine_trajectory_artifacts"), "fine artifacts").get(side),
        f"{side} fine artifact",
    )
    trajectory = _read_trajectory(supplied, config)
    if trajectory.sha256 != identity.get("sha256"):
        raise ValueError(f"{side} trajectory hash differs from registered budget")
    return trajectory


def compare(
    config_path: Path,
    budget_path: Path,
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    """Compare locked fine trajectories without claiming same-field agreement."""

    config = _load_config(config_path)
    budget, budget_hash = _load_report(
        budget_path,
        "m3c1_preregistered_cross_representation_budget",
    )
    if (
        _mapping(budget.get("evaluation_config"), "evaluation config").get("sha256")
        != config.sha256
    ):
        raise ValueError("comparison budget uses another evaluation config")
    if budget.get("status") != "REGISTERED":
        raise ValueError("comparison budget is blocked")
    candidate = _registered_trajectory(budget, "candidate", candidate_path, config)
    reference = _registered_trajectory(budget, "reference", reference_path, config)
    metrics = _difference_metrics(candidate.values, reference.values, config.output_interval_s)
    envelopes = _mapping(budget.get("envelopes"), "envelopes")
    gates: dict[str, object] = {}
    blockers: list[str] = []
    for quantity in ("position", "velocity", "charge"):
        envelope = _mapping(envelopes.get(quantity), f"{quantity} envelope")
        observed_report = _mapping(metrics.get(quantity), quantity)
        quantity_gates: dict[str, object] = {}
        for metric in ("rms", "maximum"):
            observed = _number(observed_report.get(metric), f"observed {quantity}.{metric}")
            tolerance = _number(envelope.get(metric), f"tolerance {quantity}.{metric}")
            passed = observed <= tolerance
            quantity_gates[metric] = {"observed": observed, "tolerance": tolerance, "pass": passed}
            if not passed:
                blockers.append(f"cross-representation {quantity}.{metric} exceeds envelope")
        gates[quantity] = quantity_gates
    passed = not blockers
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool_revision": TOOL_REVISION,
        "report_kind": "m3c1_locked_cross_representation_trajectory_comparison",
        "status": "PASS" if passed else "FAIL",
        "evaluation_config": {"path": str(config.path), "sha256": config.sha256},
        "scope": budget.get("scope"),
        "budget": {"path": str(budget_path.resolve()), "sha256": budget_hash},
        "artifacts": {
            "candidate": {"path": str(candidate.path), "sha256": candidate.sha256},
            "reference": {"path": str(reference.path), "sha256": reference.sha256},
        },
        "trajectory_difference": metrics,
        "acceptance_gates": gates,
        "blockers": blockers,
        "claim_separation": {
            "cross_representation_trajectory": "PASS" if passed else "FAIL",
            "same_field_solver_agreement": "NOT_TESTED",
            "common_field_diagnostic": "CONDITIONAL_NOT_RUN",
        },
        "accuracy_claim": {
            "locked_cross_representation_case_and_window": (
                "SUPPORTED_WITHIN_PREREGISTERED_ENVELOPE" if passed else "NOT_SUPPORTED"
            ),
            "same_field_solver_agreement": "NOT_TESTED",
            "comsol_universal_equal_accuracy": "NOT_CLAIMED",
            "physical_model_validity": "NOT_CLAIMED",
            "boundary_accuracy": "NOT_TESTED_PRE_EVENT_WINDOW",
        },
    }


def _write(path: Path, report: dict[str, object]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", errors="strict") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    characterize_parser = commands.add_parser("characterize")
    characterize_parser.add_argument("--config", required=True, type=Path)
    characterize_parser.add_argument(
        "--label", required=True, choices=("solver_canonical_p1_projection", "comsol_native_field")
    )
    characterize_parser.add_argument("--coarse", required=True, type=Path)
    characterize_parser.add_argument("--medium", required=True, type=Path)
    characterize_parser.add_argument("--fine", required=True, type=Path)
    characterize_parser.add_argument("--candidate-run-report", type=Path)
    characterize_parser.add_argument("--prepare-report", type=Path)
    characterize_parser.add_argument("--output", required=True, type=Path)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--config", required=True, type=Path)
    register_parser.add_argument("--candidate", required=True, type=Path)
    register_parser.add_argument("--reference", required=True, type=Path)
    register_parser.add_argument("--output", required=True, type=Path)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--config", required=True, type=Path)
    compare_parser.add_argument("--budget", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--reference", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "characterize":
        report = characterize(
            args.config,
            args.label,
            args.coarse,
            args.medium,
            args.fine,
            candidate_run_report=args.candidate_run_report,
            prepare_report=args.prepare_report,
        )
    elif args.command == "register":
        report = register_budget(args.config, args.candidate, args.reference)
    else:
        report = compare(args.config, args.budget, args.candidate, args.reference)
    _write(args.output, report)
    return 0 if report["status"] in {"PASS", "REGISTERED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
