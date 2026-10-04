"""Certify and compare one deterministic pre-event trajectory match.

This external V&V tool deliberately has three phases.  Each solver is first
characterized without seeing the other solver's trajectory difference.  A
comparison envelope is then registered from those two self-convergence
reports.  Only the final phase reads both fine-step trajectories.  The split
prevents result-dependent tolerance tuning and keeps COMSOL concerns out of
the production solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

TOOL_REVISION: Final = "m3v_matched_trajectory_evaluation_v1"
EXPECTED_PARTICLES: Final = 287
EXPECTED_FRAMES: Final = 41
OUTPUT_INTERVAL_S: Final = 1.0e-5
MINIMUM_RMS_ORDER: Final = 1.0
MINIMUM_MAXIMUM_ORDER: Final = 0.0
PRECISION_PARITY_FACTOR: Final = 2.0
ROUNDOFF_MULTIPLIER: Final = 2048.0
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
_COMPONENTS: Final = ("r_m", "z_m", "velocity_r_m_per_s", "velocity_z_m_per_s")


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


def _read_trajectory(path: Path) -> _Trajectory:
    resolved = path.expanduser().resolve()
    values = np.empty((EXPECTED_FRAMES, EXPECTED_PARTICLES, 4), dtype=np.float64)
    seen = np.zeros((EXPECTED_FRAMES, EXPECTED_PARTICLES), dtype=np.bool_)
    with resolved.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _COLUMNS:
            raise ValueError(f"{resolved}: trajectory columns must be exactly {_COLUMNS}")
        for line_number, row in enumerate(reader, start=2):
            try:
                particle_id = int(row["particle_id"])
                time_s = float(row["time_s"])
                charge = float(row["charge_number_e"])
                state = np.asarray([float(row[name]) for name in _COMPONENTS])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{resolved}:{line_number}: invalid numeric value") from error
            if not 1 <= particle_id <= EXPECTED_PARTICLES:
                raise ValueError(f"{resolved}:{line_number}: particle ID outside 1..287")
            frame = round(time_s / OUTPUT_INTERVAL_S)
            expected_time = frame * OUTPUT_INTERVAL_S
            time_tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(time_s))
            if not 0 <= frame < EXPECTED_FRAMES or abs(time_s - expected_time) > time_tolerance:
                raise ValueError(f"{resolved}:{line_number}: time is not on the fixed output grid")
            particle = particle_id - 1
            if seen[frame, particle]:
                raise ValueError(f"{resolved}:{line_number}: duplicate particle/time key")
            if not np.isfinite(state).all() or not math.isfinite(charge):
                raise ValueError(f"{resolved}:{line_number}: nonfinite state")
            if charge != -1.0 or row["lifecycle"] != "active":
                raise ValueError(
                    f"{resolved}:{line_number}: pre-event match requires charge -1 and active state"
                )
            seen[frame, particle] = True
            values[frame, particle] = state
    if not seen.all():
        missing = np.argwhere(~seen)[0]
        raise ValueError(
            f"{resolved}: missing frame={int(missing[0])}, particle={int(missing[1]) + 1}"
        )
    return _Trajectory(resolved, _sha256(resolved), values)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def _norm_metrics(
    difference: np.ndarray,
    comparison: np.ndarray,
) -> dict[str, object]:
    magnitude = np.linalg.norm(difference, axis=2)
    comparison_magnitude = np.linalg.norm(comparison, axis=2)
    square_sum = float(np.sum(magnitude * magnitude, dtype=np.float64))
    scale_square_sum = float(np.sum(comparison_magnitude * comparison_magnitude, dtype=np.float64))
    maximum_index = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    per_time: list[dict[str, object]] = []
    for frame in range(magnitude.shape[0]):
        row = magnitude[frame]
        worst = int(np.argmax(row))
        per_time.append(
            {
                "frame": frame,
                "time_s": frame * OUTPUT_INTERVAL_S,
                "rms": float(np.sqrt(np.mean(row * row))),
                "p99": float(np.quantile(row, 0.99)),
                "maximum": float(row[worst]),
                "worst_particle_id": worst + 1,
            }
        )
    particle_maximum = np.max(magnitude, axis=0)
    particle_rms = np.sqrt(np.mean(magnitude * magnitude, axis=0))
    worst_particles = np.argsort(particle_maximum)[-10:][::-1]
    return {
        "count": int(magnitude.size),
        "rms": math.sqrt(square_sum / magnitude.size),
        "maximum": float(magnitude[maximum_index]),
        "relative_l2": math.sqrt(square_sum / scale_square_sum) if scale_square_sum > 0.0 else None,
        "percentiles": _percentiles(magnitude),
        "maximum_location": {
            "particle_id": int(maximum_index[1]) + 1,
            "frame": int(maximum_index[0]),
            "time_s": int(maximum_index[0]) * OUTPUT_INTERVAL_S,
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


def _difference_metrics(coarse: np.ndarray, fine: np.ndarray) -> dict[str, object]:
    difference = coarse - fine
    components: dict[str, object] = {}
    for index, name in enumerate(_COMPONENTS):
        absolute = np.abs(difference[:, :, index])
        location = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
        components[name] = {
            "rms": float(np.sqrt(np.mean(absolute * absolute))),
            "maximum": float(absolute[location]),
            "percentiles": _percentiles(absolute),
            "maximum_location": {
                "particle_id": int(location[1]) + 1,
                "frame": int(location[0]),
                "time_s": int(location[0]) * OUTPUT_INTERVAL_S,
            },
        }
    return {
        "position": _norm_metrics(difference[:, :, :2], fine[:, :, :2]),
        "velocity": _norm_metrics(difference[:, :, 2:], fine[:, :, 2:]),
        "components": components,
    }


def _order(coarse_fine: float, fine_finer: float) -> float | None:
    if coarse_fine <= 0.0 or fine_finer <= 0.0:
        return None
    return math.log2(coarse_fine / fine_finer)


def characterize(
    label: str,
    coarse_path: Path,
    medium_path: Path,
    fine_path: Path,
) -> dict[str, object]:
    """Characterize one solver without reading the other solver's results."""

    trajectories = {
        "10us": _read_trajectory(coarse_path),
        "5us": _read_trajectory(medium_path),
        "2p5us": _read_trajectory(fine_path),
    }
    coarse_medium = _difference_metrics(trajectories["10us"].values, trajectories["5us"].values)
    medium_fine = _difference_metrics(trajectories["5us"].values, trajectories["2p5us"].values)
    orders: dict[str, object] = {}
    blockers: list[str] = []
    for quantity in ("position", "velocity"):
        quantity_orders: dict[str, float | None] = {}
        for metric, minimum_order in (
            ("rms", MINIMUM_RMS_ORDER),
            ("maximum", MINIMUM_MAXIMUM_ORDER),
        ):
            first = float(coarse_medium[quantity][metric])  # type: ignore[index]
            second = float(medium_fine[quantity][metric])  # type: ignore[index]
            observed = _order(first, second)
            quantity_orders[metric] = observed
            if observed is None or observed <= minimum_order:
                blockers.append(
                    f"{quantity}.{metric} observed order {observed!r} is below "
                    f"the required threshold {minimum_order}"
                )
        orders[quantity] = quantity_orders
    return {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "report_kind": "single_solver_self_convergence",
        "label": label,
        "scope": {
            "case": "Case A 100 nm deterministic common physics",
            "time_window_s": [0.0, 4.0e-4],
            "particles": EXPECTED_PARTICLES,
            "frames": EXPECTED_FRAMES,
            "boundary": "NOT_TESTED_PRE_EVENT_WINDOW",
        },
        "artifacts": {
            name: {"path": str(item.path), "sha256": item.sha256}
            for name, item in trajectories.items()
        },
        "initial_state_sha256": hashlib.sha256(
            trajectories["2p5us"].values[0].tobytes(order="C")
        ).hexdigest(),
        "state_scale": {
            "maximum_position_norm_m": float(
                np.max(np.linalg.norm(trajectories["2p5us"].values[:, :, :2], axis=2))
            ),
            "maximum_velocity_norm_m_per_s": float(
                np.max(np.linalg.norm(trajectories["2p5us"].values[:, :, 2:], axis=2))
            ),
        },
        "10us_vs_5us": coarse_medium,
        "5us_vs_2p5us": medium_fine,
        "observed_order": orders,
        "minimum_observed_order": {
            "rms": MINIMUM_RMS_ORDER,
            "maximum": MINIMUM_MAXIMUM_ORDER,
        },
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "interpretation": (
            "This report measures fixed-step self-convergence only; it does not compare "
            "against the other solver or establish physical-model validity."
        ),
    }


def _load_json(path: Path, expected_kind: str) -> tuple[dict[str, object], str]:
    resolved = path.expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("report_kind") != expected_kind:
        raise ValueError(f"{resolved}: expected report_kind={expected_kind}")
    if value.get("tool_revision") != TOOL_REVISION:
        raise ValueError(f"{resolved}: incompatible tool revision")
    return value, _sha256(resolved)


def _quantity(report: dict[str, object], quantity: str, metric: str) -> float:
    pair = report["5us_vs_2p5us"]
    if not isinstance(pair, dict) or not isinstance(pair.get(quantity), dict):
        raise ValueError(f"invalid convergence report quantity: {quantity}")
    value = pair[quantity][metric]
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"invalid convergence report metric: {quantity}.{metric}")
    return float(value)


def _fine_artifact(report: dict[str, object]) -> dict[str, str]:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("2p5us"), dict):
        raise ValueError("self-convergence report has no fine artifact")
    artifact = artifacts["2p5us"]
    path = artifact.get("path")
    digest = artifact.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ValueError("fine artifact identity is invalid")
    return {"path": path, "sha256": digest}


def _registration_blockers(candidate: dict[str, object], reference: dict[str, object]) -> list[str]:
    blockers: list[str] = []
    if candidate.get("label") != "candidate" or reference.get("label") != "comsol_reference":
        blockers.append("self-convergence report labels are not candidate/comsol_reference")
    for name, report in (("candidate", candidate), ("reference", reference)):
        if report.get("status") != "PASS":
            blockers.append(f"{name} self-convergence did not pass")
    if candidate.get("scope") != reference.get("scope"):
        blockers.append("candidate and reference scopes differ")
    if candidate.get("initial_state_sha256") != reference.get("initial_state_sha256"):
        blockers.append("initial-state hashes differ before cross-trajectory comparison")
    return blockers


def register_budget(candidate_path: Path, reference_path: Path) -> dict[str, object]:
    """Register a cross-solver envelope before reading cross-solver differences."""

    candidate, candidate_hash = _load_json(candidate_path, "single_solver_self_convergence")
    reference, reference_hash = _load_json(reference_path, "single_solver_self_convergence")
    blockers = _registration_blockers(candidate, reference)

    scales: dict[str, float] = {}
    for quantity, scale_key in (
        ("position", "maximum_position_norm_m"),
        ("velocity", "maximum_velocity_norm_m_per_s"),
    ):
        values: list[float] = []
        for report in (candidate, reference):
            state_scale = report.get("state_scale")
            if not isinstance(state_scale, dict):
                raise ValueError("self-convergence report state_scale is invalid")
            value = state_scale.get(scale_key)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError(f"self-convergence scale {scale_key} is invalid")
            values.append(float(value))
        scales[quantity] = max(1.0, *values)

    envelopes: dict[str, object] = {}
    precision: dict[str, object] = {}
    for quantity in ("position", "velocity"):
        roundoff = ROUNDOFF_MULTIPLIER * np.finfo(np.float64).eps * scales[quantity]
        quantity_envelope: dict[str, float] = {}
        quantity_precision: dict[str, object] = {}
        for metric in ("rms", "maximum"):
            candidate_change = _quantity(candidate, quantity, metric)
            reference_change = _quantity(reference, quantity, metric)
            quantity_envelope[metric] = candidate_change + reference_change + roundoff
            parity_limit = PRECISION_PARITY_FACTOR * reference_change + roundoff
            parity = bool(candidate_change <= parity_limit)
            quantity_precision[metric] = {
                "candidate_fine_step_change": candidate_change,
                "reference_fine_step_change": reference_change,
                "candidate_to_reference_ratio": candidate_change / reference_change
                if reference_change > 0.0
                else None,
                "limit": parity_limit,
                "pass": parity,
            }
            if not parity:
                blockers.append(f"candidate {quantity}.{metric} is not at COMSOL precision")
        envelopes[quantity] = quantity_envelope
        precision[quantity] = quantity_precision
    return {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "report_kind": "preregistered_cross_solver_budget",
        "status": "REGISTERED" if not blockers else "BLOCKED",
        "scope": candidate.get("scope"),
        "policy": {
            "minimum_observed_order": {
                "rms": MINIMUM_RMS_ORDER,
                "maximum": MINIMUM_MAXIMUM_ORDER,
            },
            "precision_parity_factor": PRECISION_PARITY_FACTOR,
            "roundoff_multiplier": ROUNDOFF_MULTIPLIER,
            "cross_envelope": (
                "candidate 5us-vs-2.5us change + reference 5us-vs-2.5us change + "
                "float64 representation floor"
            ),
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
        "claim_policy": (
            "Registration does not compare fine trajectories.  A later PASS can support only "
            "the hashed deterministic pre-event case, not universal COMSOL equivalence."
        ),
    }


def _registered_artifact(budget: dict[str, object], side: str, supplied: Path) -> _Trajectory:
    identities = budget.get("fine_trajectory_artifacts")
    if not isinstance(identities, dict) or not isinstance(identities.get(side), dict):
        raise ValueError(f"budget has no registered {side} trajectory")
    identity = identities[side]
    trajectory = _read_trajectory(supplied)
    if trajectory.sha256 != identity.get("sha256"):
        raise ValueError(f"{side} trajectory hash differs from registered budget")
    return trajectory


def compare(
    budget_path: Path,
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    """Compare the locked fine trajectories against the preregistered envelope."""

    budget, budget_hash = _load_json(budget_path, "preregistered_cross_solver_budget")
    if budget.get("status") != "REGISTERED":
        raise ValueError("comparison budget is blocked")
    candidate = _registered_artifact(budget, "candidate", candidate_path)
    reference = _registered_artifact(budget, "reference", reference_path)
    metrics = _difference_metrics(candidate.values, reference.values)
    initial_metrics = _difference_metrics(candidate.values[:1], reference.values[:1])
    envelopes = budget.get("envelopes")
    if not isinstance(envelopes, dict):
        raise ValueError("comparison budget envelopes are invalid")
    gates: dict[str, object] = {}
    blockers: list[str] = []
    for quantity in ("position", "velocity"):
        if not isinstance(envelopes.get(quantity), dict):
            raise ValueError(f"comparison budget has no {quantity} envelope")
        quantity_gates: dict[str, object] = {}
        for metric in ("rms", "maximum"):
            observed = float(metrics[quantity][metric])  # type: ignore[index]
            tolerance = float(envelopes[quantity][metric])
            passed = observed <= tolerance
            quantity_gates[metric] = {
                "observed": observed,
                "tolerance": tolerance,
                "ratio": observed / tolerance if tolerance > 0.0 else None,
                "pass": passed,
            }
            if not passed:
                blockers.append(f"cross-solver {quantity}.{metric} exceeds envelope")
        gates[quantity] = quantity_gates
    passed = not blockers
    return {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "report_kind": "locked_cross_solver_trajectory_comparison",
        "status": "PASS" if passed else "FAIL",
        "scope": budget.get("scope"),
        "budget": {"path": str(budget_path.resolve()), "sha256": budget_hash},
        "artifacts": {
            "candidate": {"path": str(candidate.path), "sha256": candidate.sha256},
            "reference": {"path": str(reference.path), "sha256": reference.sha256},
        },
        "initial_state_difference": initial_metrics,
        "trajectory_difference": metrics,
        "acceptance_gates": gates,
        "blockers": blockers,
        "accuracy_claim": {
            "same_accuracy_for_hashed_case_and_time_window": (
                "SUPPORTED_WITHIN_PREREGISTERED_ENVELOPE" if passed else "NOT_SUPPORTED"
            ),
            "comsol_universal_equal_accuracy": "NOT_CLAIMED",
            "bitwise_equivalence": "NOT_CLAIMED",
            "physical_model_validity": "NOT_CLAIMED",
            "boundary_accuracy": "NOT_TESTED_PRE_EVENT_WINDOW",
            "interpretation": (
                "PASS means both solvers self-converged and their locked 2.5 us histories agree "
                "within the envelope registered before cross differences were read."
            ),
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
    characterize_parser.add_argument("--label", required=True)
    characterize_parser.add_argument("--coarse", required=True, type=Path)
    characterize_parser.add_argument("--medium", required=True, type=Path)
    characterize_parser.add_argument("--fine", required=True, type=Path)
    characterize_parser.add_argument("--output", required=True, type=Path)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--candidate", required=True, type=Path)
    register_parser.add_argument("--reference", required=True, type=Path)
    register_parser.add_argument("--output", required=True, type=Path)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--budget", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--reference", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "characterize":
        report = characterize(args.label, args.coarse, args.medium, args.fine)
    elif args.command == "register":
        report = register_budget(args.candidate, args.reference)
    else:
        report = compare(args.budget, args.candidate, args.reference)
    _write(args.output, report)
    return 0 if report["status"] in {"PASS", "REGISTERED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
