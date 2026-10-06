"""Evaluate one Case-A size/ion-drag companion cell.

The report compares the three fixed-step solver trajectories with the three
normalized COMSOL common-P1 trajectories.  The fixed acceptance limits are
inherited from the predeclared M3-C1 common-P1 policy; this tool does not derive
tolerances from the size/ion-drag results.
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

from tools.vv.comsol.prepare_m3c_casea_companion import RUN_SPEC_KEYS, STEP_ROWS

TOOL_REVISION: Final = "m3c_casea_size_iondrag_companion_evaluation_v2"
_HEADER: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
_REFERENCE_STEP = {
    "coarse": "dt_0p625us",
    "medium": "dt_0p3125us",
    "fine": "dt_0p15625us",
}
_GATE_SOURCE: Final = Path(__file__).with_name("cases") / "m3c1_caseA_100nm_common_p1_v1.json"
_OUTPUT_INTERVAL_S: Final = 1.0e-5
_STATE_NAMES: Final = ("r_m", "z_m", "velocity_r_m_s", "velocity_z_m_s", "charge_e")


@dataclass(frozen=True, slots=True)
class Trajectory:
    path: Path
    sha256: str
    values: np.ndarray
    lifecycle: np.ndarray


@dataclass(frozen=True, slots=True)
class GatePolicy:
    source_path: Path
    source_sha256: str
    minimum_order: float
    relative_limits: dict[str, float]
    absolute_limits: dict[str, dict[str, float]]
    initial_state_roundoff_multiplier: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_spec(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        if not raw.strip():
            continue
        if "=" not in raw:
            raise ValueError(f"{path}:{line_number}: expected key=value")
        key, value = (part.strip() for part in raw.split("=", 1))
        if not key or not value or key in values:
            raise ValueError(f"{path}:{line_number}: duplicate or empty run-spec entry")
        values[key] = value
    if tuple(values) != RUN_SPEC_KEYS:
        raise ValueError(f"{path}: run-spec key order or set differs")
    return values


def _read_trajectory(path: Path) -> Trajectory:
    values = np.empty((46, 287, 5), dtype=np.float64)
    lifecycle = np.empty((46, 287), dtype="U8")
    seen = np.zeros((46, 287), dtype=np.uint8)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != _HEADER:
            raise ValueError(f"{path}: trajectory columns differ")
        for line_number, row in enumerate(reader, start=2):
            particle = int(row["particle_id"]) - 1
            time_s = float(row["time_s"])
            frame = round(time_s / 1.0e-5)
            if (
                particle not in range(287)
                or frame not in range(46)
                or not math.isclose(time_s, frame * 1.0e-5, rel_tol=0.0, abs_tol=2.0e-14)
                or seen[frame, particle]
            ):
                raise ValueError(f"{path}:{line_number}: invalid or duplicate trajectory key")
            record = np.asarray(
                [
                    float(row["r_m"]),
                    float(row["z_m"]),
                    float(row["velocity_r_m_per_s"]),
                    float(row["velocity_z_m_per_s"]),
                    float(row["charge_number_e"]),
                ],
                dtype=np.float64,
            )
            if not bool(np.isfinite(record).all()):
                raise ValueError(f"{path}:{line_number}: nonfinite trajectory state")
            values[frame, particle] = record
            lifecycle[frame, particle] = row["lifecycle"]
            seen[frame, particle] = 1
    if not bool(seen.all()):
        raise ValueError(f"{path}: trajectory is not a complete 287 x 46 matrix")
    return Trajectory(path.resolve(), _sha256(path), values, lifecycle)


def _gate_policy() -> GatePolicy:
    raw = json.loads(_GATE_SOURCE.read_text(encoding="utf-8"))
    acceptance = raw["acceptance"]
    relative = acceptance["maximum_fine_pair_relative_l2"]
    absolute = acceptance["same_field_absolute_limits"]
    return GatePolicy(
        source_path=_GATE_SOURCE.resolve(),
        source_sha256=_sha256(_GATE_SOURCE),
        minimum_order=float(acceptance["minimum_rms_order"]),
        relative_limits={
            name: float(relative[name]) for name in ("position", "velocity", "charge")
        },
        absolute_limits={
            name: {metric: float(absolute[name][metric]) for metric in ("rms", "maximum")}
            for name in ("position", "velocity", "charge")
        },
        initial_state_roundoff_multiplier=float(
            raw["reference"]["validation"]["initial_state_roundoff_multiplier"]
        ),
    )


def _quantity_metrics(
    difference: np.ndarray,
    relative_scale: np.ndarray,
) -> dict[str, object]:
    magnitude = np.linalg.norm(difference, axis=2)
    denominator = float(np.linalg.norm(relative_scale.ravel()))
    maximum_index = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    return {
        "rms": float(np.sqrt(np.mean(magnitude * magnitude))),
        "maximum": float(magnitude[maximum_index]),
        "relative_l2": (
            float(np.linalg.norm(difference.ravel())) / denominator if denominator > 0.0 else None
        ),
        "p99": float(np.percentile(magnitude, 99.0)),
        "maximum_location": {
            "frame": int(maximum_index[0]),
            "particle_id": int(maximum_index[1]) + 1,
            "time_s": int(maximum_index[0]) * _OUTPUT_INTERVAL_S,
        },
    }


def _metrics(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    slices = {"position": slice(0, 2), "velocity": slice(2, 4), "charge": slice(4, 5)}
    for name, selection in slices.items():
        difference = left[..., selection] - right[..., selection]
        scale = right[..., selection]
        if name == "position":
            scale = scale - scale[0:1]
        result[name] = _quantity_metrics(difference, scale)
    result["position_relative_l2_scale"] = "right_displacement_from_each_particle_initial_position"
    result["velocity_relative_l2_scale"] = "right_velocity"
    result["charge_relative_l2_scale"] = "right_charge_number"
    return result


def _self_convergence(trajectories: dict[str, Trajectory], policy: GatePolicy) -> dict[str, object]:
    coarse_medium = _metrics(trajectories["coarse"].values, trajectories["medium"].values)
    medium_fine = _metrics(trajectories["medium"].values, trajectories["fine"].values)
    orders: dict[str, float | None] = {}
    for quantity in ("position", "velocity", "charge"):
        coarse = float(coarse_medium[quantity]["rms"])  # type: ignore[index]
        fine = float(medium_fine[quantity]["rms"])  # type: ignore[index]
        orders[quantity] = math.log(coarse / fine, 2.0) if coarse > 0.0 and fine > 0.0 else None
    gates: dict[str, object] = {}
    blockers: list[str] = []
    fine_values = trajectories["fine"].values
    for quantity, selection in {
        "position": slice(0, 2),
        "velocity": slice(2, 4),
        "charge": slice(4, 5),
    }.items():
        order = orders[quantity]
        first = float(coarse_medium[quantity]["rms"])  # type: ignore[index]
        second = float(medium_fine[quantity]["rms"])  # type: ignore[index]
        representation_scale = float(np.max(np.abs(fine_values[..., selection])))
        roundoff_floor = 2048.0 * np.finfo(np.float64).eps * max(representation_scale, 1.0e-300)
        roundoff_limited = bool(first <= roundoff_floor and second <= roundoff_floor)
        order_passed = bool(
            roundoff_limited or (order is not None and order >= policy.minimum_order)
        )
        relative = medium_fine[quantity]["relative_l2"]  # type: ignore[index]
        relative_passed = (
            isinstance(relative, int | float)
            and float(relative) <= policy.relative_limits[quantity]
        )
        gates[quantity] = {
            "observed_order": order,
            "minimum_order": policy.minimum_order,
            "roundoff_limited": roundoff_limited,
            "order_pass": order_passed,
            "fine_pair_relative_l2": relative,
            "maximum_fine_pair_relative_l2": policy.relative_limits[quantity],
            "relative_l2_pass": relative_passed,
        }
        if not order_passed:
            blockers.append(f"{quantity} observed order is below the fixed minimum")
        if not relative_passed:
            blockers.append(f"{quantity} fine-pair relative L2 exceeds the fixed limit")
    return {
        "status": "PASS" if not blockers else "FAIL",
        "coarse_vs_medium": coarse_medium,
        "medium_vs_fine": medium_fine,
        "observed_rms_order_base_2": orders,
        "acceptance_gates": gates,
        "blockers": blockers,
    }


def _initial_state_gate(
    candidate: np.ndarray, reference: np.ndarray, multiplier: float
) -> dict[str, object]:
    components: dict[str, object] = {}
    passed = True
    for index, name in enumerate(_STATE_NAMES):
        difference = np.abs(candidate[:, index] - reference[:, index])
        scale = max(
            float(np.max(np.abs(candidate[:, index]))),
            float(np.max(np.abs(reference[:, index]))),
            1.0e-300,
        )
        ulp = math.ulp(scale)
        limit = multiplier * ulp
        maximum = float(np.max(difference))
        component_passed = maximum <= limit
        passed &= component_passed
        components[name] = {
            "maximum_absolute_difference": maximum,
            "roundoff_limit": limit,
            "maximum_difference_in_scale_ulp": maximum / ulp,
            "pass": component_passed,
        }
    return {
        "status": "PASS" if passed else "FAIL",
        "roundoff_multiplier": multiplier,
        "components": components,
    }


def _comparison_gates(
    metrics: dict[str, object], policy: GatePolicy
) -> tuple[dict[str, object], list[str]]:
    gates: dict[str, object] = {}
    blockers: list[str] = []
    for quantity in ("position", "velocity", "charge"):
        observed = metrics[quantity]
        quantity_gates: dict[str, object] = {}
        for metric in ("rms", "maximum"):
            value = float(observed[metric])  # type: ignore[index]
            limit = policy.absolute_limits[quantity][metric]
            passed = value <= limit
            quantity_gates[metric] = {"observed": value, "limit": limit, "pass": passed}
            if not passed:
                blockers.append(f"{quantity}.{metric} exceeds the fixed same-field limit")
        relative = observed["relative_l2"]  # type: ignore[index]
        limit = policy.relative_limits[quantity]
        passed = isinstance(relative, int | float) and float(relative) <= limit
        quantity_gates["relative_l2"] = {"observed": relative, "limit": limit, "pass": passed}
        if not passed:
            blockers.append(f"{quantity}.relative_l2 exceeds the fixed same-field limit")
        gates[quantity] = quantity_gates
    return gates, blockers


def _lifecycle_gate(
    candidate: dict[str, Trajectory], reference: dict[str, Trajectory]
) -> tuple[dict[str, bool], list[str]]:
    equal = all(
        np.array_equal(candidate[label].lifecycle, reference[label].lifecycle)
        for label, _reference_name, _dt_s in STEP_ROWS
    )
    all_active = all(
        bool(np.all(trajectory.lifecycle == "active"))
        for trajectories in (candidate, reference)
        for trajectory in trajectories.values()
    )
    blockers: list[str] = []
    if not equal:
        blockers.append("candidate/reference lifecycle histories differ")
    if not all_active:
        blockers.append("the comparison is not an event-free pre-event history")
    return {
        "exact_candidate_reference_equality": equal,
        "all_active": all_active,
        "pass": equal and all_active,
    }, blockers


def _load_candidate(case_directory: Path, spec: dict[str, str]) -> dict[str, Trajectory]:
    trajectories: dict[str, Trajectory] = {}
    expected_revision = spec["ion_drag_revision"]
    for label, _reference_name, _dt_s in STEP_ROWS:
        report_path = case_directory / "candidate_results" / f"candidate_run_{label}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("case_id") != spec["case_id"] or report.get("step") != label:
            raise ValueError(f"{report_path}: candidate identity differs")
        physics = report.get("physics_models")
        if not isinstance(physics, dict) or physics.get("ion_drag") != {
            "model": (
                "image_orbital_sensitivity"
                if spec["case_id"] == "caseA_100nm_image"
                else "screened_collection_orbital"
            ),
            "revision": expected_revision,
        }:
            raise ValueError(f"{report_path}: candidate ion-drag model differs")
        trajectory_path = Path(str(report["trajectory"])).resolve()
        trajectory = _read_trajectory(trajectory_path)
        if trajectory.sha256 != report.get("trajectory_sha256"):
            raise ValueError(f"{report_path}: candidate trajectory hash differs")
        trajectories[label] = trajectory
    return trajectories


def _load_reference(reference_directory: Path, spec: dict[str, str]) -> dict[str, Trajectory]:
    summary_path = reference_directory / "normalization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE" or summary.get("campaign_spec") != spec:
        raise ValueError(f"{summary_path}: normalized campaign identity differs")
    trajectories: dict[str, Trajectory] = {}
    for label, step_name in _REFERENCE_STEP.items():
        trajectory = _read_trajectory(reference_directory / step_name / "trajectory_reference.csv")
        run = summary["runs"][step_name]
        if trajectory.sha256 != run.get("trajectory_sha256"):
            raise ValueError(f"{summary_path}: reference trajectory hash differs")
        trajectories[label] = trajectory
    return trajectories


def evaluate(case_directory: Path, reference_directory: Path, output: Path) -> dict[str, object]:
    """Write one fixed-gate companion comparison report."""

    case_root = case_directory.resolve()
    reference_root = reference_directory.resolve()
    policy = _gate_policy()
    spec = _run_spec(case_root / "run_spec.properties")
    candidate = _load_candidate(case_root, spec)
    reference = _load_reference(reference_root, spec)
    candidate_convergence = _self_convergence(candidate, policy)
    reference_convergence = _self_convergence(reference, policy)
    initial_gate = _initial_state_gate(
        candidate["fine"].values[0],
        reference["fine"].values[0],
        policy.initial_state_roundoff_multiplier,
    )
    fine_difference = _metrics(candidate["fine"].values, reference["fine"].values)
    comparison_gates, blockers = _comparison_gates(fine_difference, policy)
    if candidate_convergence["status"] != "PASS":
        blockers.append("candidate self-convergence did not pass")
    if reference_convergence["status"] != "PASS":
        blockers.append("reference self-convergence did not pass")
    if initial_gate["status"] != "PASS":
        blockers.append("candidate/reference initial states exceed the fixed roundoff gate")
    lifecycle_gate, lifecycle_blockers = _lifecycle_gate(candidate, reference)
    blockers.extend(lifecycle_blockers)
    passed = not blockers
    report: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "PASS" if passed else "FAIL",
        "classification": "fixed_gate_external_same_common_p1_pre_event_comparison",
        "campaign_spec": spec,
        "scope": {
            "particles": 287,
            "frames": 46,
            "time_window_s": [0.0, 4.5e-4],
            "common_field": "canonical_exact_connectivity_P1",
        },
        "acceptance_policy": {
            "source": str(policy.source_path),
            "source_sha256": policy.source_sha256,
            "result_dependent_tolerance_tuning": "PROHIBITED",
            "minimum_rms_order": policy.minimum_order,
            "maximum_fine_pair_relative_l2": policy.relative_limits,
            "same_field_absolute_limits": policy.absolute_limits,
            "initial_state_roundoff_multiplier": policy.initial_state_roundoff_multiplier,
        },
        "candidate_self_convergence": candidate_convergence,
        "reference_self_convergence": reference_convergence,
        "initial_state_gate": initial_gate,
        "fine_pair_difference": fine_difference,
        "acceptance_gates": comparison_gates,
        "lifecycle_gate": lifecycle_gate,
        "lifecycle_counts": {
            "candidate_fine": {
                name: int(np.count_nonzero(candidate["fine"].lifecycle == name))
                for name in np.unique(candidate["fine"].lifecycle)
            },
            "reference_fine": {
                name: int(np.count_nonzero(reference["fine"].lifecycle == name))
                for name in np.unique(reference["fine"].lifecycle)
            },
        },
        "artifacts": {
            side: {
                label: {"path": str(item.path), "sha256": item.sha256}
                for label, item in trajectories.items()
            }
            for side, trajectories in (("candidate", candidate), ("reference", reference))
        },
        "claim_policy": {
            "same_field_time_integration_agreement": "PASS" if passed else "NOT_SUPPORTED",
            "comsol_is_golden_truth": False,
            "native_field_agreement": "NOT_CERTIFIED",
            "general_boundary_accuracy": "NOT_TESTED_PRE_EVENT_WINDOW",
            "brownian_accuracy": "NOT_TESTED",
            "physical_model_validity": "NOT_CERTIFIED_BY_TRAJECTORY_AGREEMENT",
        },
        "blockers": blockers,
    }
    output.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("reference_directory", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    evaluate(arguments.case_directory, arguments.reference_directory, arguments.output)


if __name__ == "__main__":
    main()
