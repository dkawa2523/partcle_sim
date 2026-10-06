"""Evaluate deterministic M3-C3 self-convergence and COMSOL agreement.

All gates are declared before reading results.  COMSOL is an external observed
reference, not a definition of the solver physics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from tools.vv.comsol.run_m3c3_casep_three_current_candidate import (
    CASE_ID,
    EVENT_HEADER,
    LEVELS,
    OUTPUT_COUNT,
    PARTICLE_COUNT,
    TIME_END_S,
    TRAJECTORY_HEADER,
)
from tools.vv.comsol.run_m3c3_casep_three_current_candidate import (
    TOOL_REVISION as CANDIDATE_TOOL_REVISION,
)

TOOL_REVISION: Final = "m3c3_casep_three_current_evaluation_v5"
ROUND_OFF_MULTIPLIER: Final = 2048.0
COMSOL_REFERENCE_STEPS_S: Final = (5.0e-6, 2.5e-6, 1.25e-6)
COMSOL_REFERENCE_LEVELS: Final = (
    ("dt_5us", 5.0e-6),
    ("dt_2p5us", 2.5e-6),
    ("dt_1p25us", 1.25e-6),
)
QUANTITIES: Final = ("position", "velocity", "charge")
GATE_POLICY_RELATIVE_PATH: Final = "tools/vv/comsol/cases/m3c1_caseA_100nm_common_p1_v1.json"
COARSE_LEVEL, MEDIUM_LEVEL, FINE_LEVEL = (name for name, _step_s in LEVELS)


@dataclass(frozen=True, slots=True)
class Trajectory:
    path: Path
    sha256: str
    values: np.ndarray
    lifecycle: np.ndarray
    finite: np.ndarray


@dataclass(frozen=True, slots=True)
class BoundaryEvents:
    path: Path
    sha256: str
    particle_id: np.ndarray
    time_s: np.ndarray
    event_type: np.ndarray
    outcome: np.ndarray
    boundary_semantic: np.ndarray


@dataclass(frozen=True, slots=True)
class GatePolicy:
    source_path: Path
    source_sha256: str
    minimum_order: float
    fine_relative: dict[str, float]
    inherited_absolute: dict[str, dict[str, float]]
    direct_multiplier: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(cast(Mapping[str, Any], value))


def _output_times() -> np.ndarray:
    return np.asarray(
        [index * 1.0e-5 for index in range(51)]
        + [index * 1.0e-4 for index in range(6, 51)]
        + [index * 1.0e-3 for index in range(6, 31)],
        dtype=np.float64,
    )


def _frame_index(time_s: float, expected: np.ndarray, path: Path, line: int) -> int:
    index = int(np.searchsorted(expected, time_s))
    candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < expected.size]
    if not candidates:
        raise ValueError(f"{path}:{line}: trajectory time lies outside the schedule")
    selected = min(candidates, key=lambda candidate: abs(float(expected[candidate]) - time_s))
    if not math.isclose(time_s, float(expected[selected]), rel_tol=0.0, abs_tol=3.0e-13):
        raise ValueError(f"{path}:{line}: trajectory time differs from the schedule")
    return selected


def _read_trajectory(path: Path) -> Trajectory:
    expected = _output_times()
    values = np.empty((OUTPUT_COUNT, PARTICLE_COUNT, 5), dtype=np.float64)
    lifecycle = np.empty((OUTPUT_COUNT, PARTICLE_COUNT), dtype="U8")
    seen = np.zeros((OUTPUT_COUNT, PARTICLE_COUNT), dtype=np.bool_)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != TRAJECTORY_HEADER:
            raise ValueError(f"{path}: trajectory header differs")
        for line, row in enumerate(reader, start=2):
            particle = int(row["particle_id"]) - 1
            frame = _frame_index(float(row["time_s"]), expected, path, line)
            if particle not in range(PARTICLE_COUNT) or seen[frame, particle]:
                raise ValueError(f"{path}:{line}: duplicate or invalid particle/time key")
            state = np.asarray(
                [
                    float(row["r_m"]),
                    float(row["z_m"]),
                    float(row["velocity_r_m_per_s"]),
                    float(row["velocity_z_m_per_s"]),
                    float(row["charge_number_e"]),
                ],
                dtype=np.float64,
            )
            state_finite = np.isfinite(state)
            life = row["lifecycle"]
            if not bool(state_finite.all()) and not (
                life == "escaped" and not bool(state_finite.any())
            ):
                raise ValueError(f"{path}:{line}: state is partially nonfinite or not escaped")
            values[frame, particle] = state
            lifecycle[frame, particle] = life
            seen[frame, particle] = True
    if not bool(seen.all()):
        raise ValueError(f"{path}: trajectory is not a complete 287 x 121 matrix")
    finite = np.isfinite(values).all(axis=2)
    return Trajectory(path.resolve(), _sha256(path), values, lifecycle, finite)


def _read_events(path: Path) -> BoundaryEvents:
    rows: list[tuple[int, float, str, str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != EVENT_HEADER:
            raise ValueError(f"{path}: boundary-event header differs")
        for line, row in enumerate(reader, start=2):
            particle_id = int(row["particle_id"])
            time_s = float(row["event_time_s"])
            if particle_id not in range(1, PARTICLE_COUNT + 1):
                raise ValueError(f"{path}:{line}: boundary-event particle ID is invalid")
            if not math.isfinite(time_s) or not 0.0 <= time_s <= TIME_END_S:
                raise ValueError(f"{path}:{line}: boundary-event time is invalid")
            rows.append(
                (
                    particle_id,
                    time_s,
                    row["event_type"],
                    row["outcome"],
                    row["boundary_semantic"],
                )
            )
    rows.sort(key=lambda item: item[0])
    particle_id = np.asarray([row[0] for row in rows], dtype=np.int64)
    if particle_id.size and np.unique(particle_id).size != particle_id.size:
        raise ValueError(f"{path}: more than one terminal event exists for a particle")
    return BoundaryEvents(
        path.resolve(),
        _sha256(path),
        particle_id,
        np.asarray([row[1] for row in rows], dtype=np.float64),
        np.asarray([row[2] for row in rows], dtype="U32"),
        np.asarray([row[3] for row in rows], dtype="U16"),
        np.asarray([row[4] for row in rows], dtype="U64"),
    )


def _policy(prepared_root: Path) -> GatePolicy:
    config_path = prepared_root / "campaign_config.json"
    prepare_report = _mapping(
        json.loads((prepared_root / "prepare_report.json").read_text(encoding="utf-8")),
        "prepare report",
    )
    if _sha256(config_path) != prepare_report.get("configuration_sha256"):
        raise ValueError("prepared campaign config differs from its preparation hash binding")
    campaign = _mapping(
        json.loads(config_path.read_text(encoding="utf-8")),
        "campaign config",
    )
    acceptance = _mapping(campaign["acceptance"], "campaign acceptance")
    expected_acceptance = {
        "policy_source_solver_relative_path": GATE_POLICY_RELATIVE_PATH,
        "policy_source_sha256": (
            "db85d2187edd2b8807717a9f7272b6e198b3dc1a1a080df46e67980458fc763d"
        ),
        "minimum_rms_order": 0.75,
        "direct_uncertainty_multiplier": 4.0,
        "direct_uncertainty_components": [
            "candidate_fine_pair",
            "comsol_reference_pair",
        ],
        "lifecycle_exact": True,
    }
    if acceptance != expected_acceptance:
        raise ValueError("M3-C3 predeclared acceptance policy differs")
    source = (prepared_root / str(acceptance["policy_source_solver_relative_path"])).resolve()
    if not source.is_file():
        source = (
            Path(__file__).resolve().parents[3]
            / str(acceptance["policy_source_solver_relative_path"])
        ).resolve()
    if _sha256(source) != acceptance["policy_source_sha256"]:
        raise ValueError("inherited M3-C1 gate source differs from its hash lock")
    source_config = _mapping(json.loads(source.read_text(encoding="utf-8")), "gate source")
    inherited = _mapping(source_config["acceptance"], "inherited acceptance")
    relative = _mapping(inherited["maximum_fine_pair_relative_l2"], "fine relative limits")
    absolute = _mapping(inherited["same_field_absolute_limits"], "absolute limits")
    return GatePolicy(
        source,
        _sha256(source),
        float(acceptance["minimum_rms_order"]),
        {name: float(relative[name]) for name in QUANTITIES},
        {
            name: {
                metric: float(_mapping(absolute[name], name)[metric])
                for metric in ("rms", "maximum")
            }
            for name in QUANTITIES
        },
        float(acceptance["direct_uncertainty_multiplier"]),
    )


def _metric(
    left: Trajectory,
    right: Trajectory,
    selection: slice,
    position: bool,
    *,
    active_only: bool = False,
) -> dict[str, float]:
    common = left.finite & right.finite
    if active_only:
        common &= (left.lifecycle == "active") & (right.lifecycle == "active")
    if not bool(common.any()):
        raise ValueError("trajectory metric has no comparable states")
    difference = left.values[..., selection] - right.values[..., selection]
    difference = difference[common]
    scale = right.values[..., selection].copy()
    if position:
        scale -= right.values[0:1, :, selection]
    scale = scale[common]
    magnitude = np.linalg.norm(difference, axis=1)
    denominator = float(np.linalg.norm(scale.ravel()))
    numerator = float(np.linalg.norm(difference.ravel()))
    relative = (
        numerator / denominator if denominator > 0.0 else (0.0 if numerator == 0.0 else math.inf)
    )
    return {
        "rms": float(np.sqrt(np.mean(magnitude * magnitude))),
        "maximum": float(np.max(magnitude)),
        "relative_l2": relative,
        "p99": float(np.percentile(magnitude, 99.0)),
    }


def _metrics(left: Trajectory, right: Trajectory) -> dict[str, object]:
    return {
        "position": _metric(left, right, slice(0, 2), True),
        "velocity": _metric(left, right, slice(2, 4), False, active_only=True),
        "charge": _metric(left, right, slice(4, 5), False),
        "quantity_scope": {
            "position": "all common finite lifecycle states",
            "velocity": "common active states only; terminal Freeze velocity is not compared",
            "charge": "all common finite lifecycle states",
        },
        "kinematics_valid_exact": bool(np.array_equal(left.finite, right.finite)),
        "lifecycle_exact": bool(np.array_equal(left.lifecycle, right.lifecycle)),
    }


def _event_identity_equal(left: BoundaryEvents, right: BoundaryEvents) -> bool:
    return bool(
        np.array_equal(left.particle_id, right.particle_id)
        and np.array_equal(left.event_type, right.event_type)
        and np.array_equal(left.outcome, right.outcome)
        and np.array_equal(left.boundary_semantic, right.boundary_semantic)
    )


def _event_time_metrics(left: BoundaryEvents, right: BoundaryEvents) -> dict[str, object]:
    identity = _event_identity_equal(left, right)
    if not identity or left.time_s.size == 0:
        return {
            "identity_exact": identity,
            "event_count": int(left.time_s.size),
            "rms_s": None,
            "maximum_s": None,
        }
    difference = np.abs(left.time_s - right.time_s)
    return {
        "identity_exact": True,
        "event_count": int(left.time_s.size),
        "rms_s": float(np.sqrt(np.mean(difference * difference))),
        "maximum_s": float(np.max(difference)),
    }


def _event_roundoff_floor(*events: BoundaryEvents) -> float:
    maximum = max(
        (float(np.max(np.abs(item.time_s))) for item in events if item.time_s.size),
        default=TIME_END_S,
    )
    return float(ROUND_OFF_MULTIPLIER * np.finfo(np.float64).eps * max(maximum, TIME_END_S))


def _roundoff_floor(left: Trajectory, right: Trajectory, selection: slice) -> float:
    finite = np.concatenate(
        (
            np.abs(left.values[..., selection][left.finite]),
            np.abs(right.values[..., selection][right.finite]),
        ),
        axis=0,
    )
    scale = max(float(np.max(finite)), 1.0e-300)
    return float(ROUND_OFF_MULTIPLIER * np.finfo(np.float64).eps * scale)


def _self_convergence(
    trajectories: Mapping[str, Trajectory],
    policy: GatePolicy,
    levels: tuple[tuple[str, float], ...] = LEVELS,
) -> dict[str, object]:
    coarse_level, medium_level, fine_level = (name for name, _step_s in levels)
    coarse_medium = _metrics(trajectories[coarse_level], trajectories[medium_level])
    medium_fine = _metrics(trajectories[medium_level], trajectories[fine_level])
    gates: dict[str, object] = {}
    passed = True
    for name, selection in {
        "position": slice(0, 2),
        "velocity": slice(2, 4),
        "charge": slice(4, 5),
    }.items():
        first = float(_mapping(coarse_medium[name], name)["rms"])
        second = float(_mapping(medium_fine[name], name)["rms"])
        floor = _roundoff_floor(trajectories[medium_level], trajectories[fine_level], selection)
        roundoff_limited = bool(first <= floor and second <= floor)
        order = math.log(first / second, 2.0) if first > 0.0 and second > 0.0 else None
        order_pass = bool(roundoff_limited or (order is not None and order >= policy.minimum_order))
        relative = float(_mapping(medium_fine[name], name)["relative_l2"])
        relative_pass = bool(relative <= policy.fine_relative[name])
        gates[name] = {
            "observed_rms_order": order,
            "minimum_rms_order": policy.minimum_order,
            "roundoff_limited": roundoff_limited,
            "order_pass": order_pass,
            "fine_pair_relative_l2": relative,
            "fine_pair_relative_l2_limit": policy.fine_relative[name],
            "fine_pair_relative_l2_pass": relative_pass,
        }
        passed = passed and order_pass and relative_pass
    identity = all(
        bool(result[key])
        for result in (coarse_medium, medium_fine)
        for key in ("kinematics_valid_exact", "lifecycle_exact")
    )
    return {
        "status": "PASS" if passed and identity else "BLOCKED",
        "coarse_vs_medium": coarse_medium,
        "medium_vs_fine": medium_fine,
        "levels_s": dict(levels),
        "gates": gates,
        "lifecycle_and_validity_exact": identity,
    }


def _event_self_convergence(
    events: Mapping[str, BoundaryEvents],
    minimum_order: float,
    levels: tuple[tuple[str, float], ...] = LEVELS,
) -> dict[str, object]:
    coarse_level, medium_level, fine_level = (name for name, _step_s in levels)
    coarse_medium = _event_time_metrics(events[coarse_level], events[medium_level])
    medium_fine = _event_time_metrics(events[medium_level], events[fine_level])
    identity = bool(coarse_medium["identity_exact"] and medium_fine["identity_exact"])
    if not identity:
        return {
            "status": "BLOCKED",
            "coarse_vs_medium": coarse_medium,
            "medium_vs_fine": medium_fine,
            "identity_exact": False,
            "observed_rms_order": None,
            "minimum_rms_order": minimum_order,
        }
    if events[fine_level].time_s.size == 0:
        return {
            "status": "PASS",
            "coarse_vs_medium": coarse_medium,
            "medium_vs_fine": medium_fine,
            "identity_exact": True,
            "observed_rms_order": None,
            "minimum_rms_order": minimum_order,
            "not_applicable": "no terminal boundary events",
        }
    first = float(_mapping(coarse_medium, "coarse event metrics")["rms_s"])
    second = float(_mapping(medium_fine, "fine event metrics")["rms_s"])
    floor = _event_roundoff_floor(*events.values())
    roundoff_limited = first <= floor and second <= floor
    order = math.log(first / second, 2.0) if first > 0.0 and second > 0.0 else None
    order_pass = bool(roundoff_limited or (order is not None and order >= minimum_order))
    return {
        "status": "PASS" if order_pass else "BLOCKED",
        "coarse_vs_medium": coarse_medium,
        "medium_vs_fine": medium_fine,
        "levels_s": dict(levels),
        "identity_exact": True,
        "observed_rms_order": order,
        "minimum_rms_order": minimum_order,
        "roundoff_limited": roundoff_limited,
        "order_pass": order_pass,
    }


def _direct_gate(
    fine: Trajectory,
    reference: Trajectory,
    candidate_fine_pair: Mapping[str, object],
    reference_fine_pair: Mapping[str, object],
    policy: GatePolicy,
) -> dict[str, object]:
    direct = _metrics(fine, reference)
    gates: dict[str, object] = {}
    passed = bool(direct["lifecycle_exact"] and direct["kinematics_valid_exact"])
    for name, selection in {
        "position": slice(0, 2),
        "velocity": slice(2, 4),
        "charge": slice(4, 5),
    }.items():
        observed = _mapping(direct[name], name)
        candidate_numerical = _mapping(candidate_fine_pair[name], name)
        reference_numerical = _mapping(reference_fine_pair[name], name)
        absolute_floor = _roundoff_floor(fine, reference, selection)
        relative_floor = float(ROUND_OFF_MULTIPLIER * np.finfo(np.float64).eps)
        limits = {
            "rms": max(
                policy.inherited_absolute[name]["rms"],
                policy.direct_multiplier
                * (float(candidate_numerical["rms"]) + float(reference_numerical["rms"]))
                + absolute_floor,
            ),
            "maximum": max(
                policy.inherited_absolute[name]["maximum"],
                policy.direct_multiplier
                * (float(candidate_numerical["maximum"]) + float(reference_numerical["maximum"]))
                + absolute_floor,
            ),
            "relative_l2": max(
                policy.fine_relative[name],
                policy.direct_multiplier
                * (
                    float(candidate_numerical["relative_l2"])
                    + float(reference_numerical["relative_l2"])
                )
                + relative_floor,
            ),
        }
        metric_pass = {
            metric: bool(float(observed[metric]) <= limit) for metric, limit in limits.items()
        }
        gates[name] = {
            "observed": {metric: float(observed[metric]) for metric in limits},
            "limits": limits,
            "uncertainty_components": {
                "candidate_fine_pair": {
                    metric: float(candidate_numerical[metric]) for metric in limits
                },
                "comsol_reference_pair": {
                    metric: float(reference_numerical[metric]) for metric in limits
                },
            },
            "pass": metric_pass,
            "inherited_absolute_only_diagnostic": {
                metric: bool(float(observed[metric]) <= policy.inherited_absolute[name][metric])
                for metric in ("rms", "maximum")
            },
        }
        passed = passed and all(metric_pass.values())
    return {
        "status": "PASS" if passed else "BLOCKED",
        "metrics": direct,
        "gates": gates,
        "lifecycle_exact_required": True,
        "lifecycle_exact": direct["lifecycle_exact"],
        "kinematics_valid_exact": direct["kinematics_valid_exact"],
    }


def _event_direct_gate(
    fine: BoundaryEvents,
    reference: BoundaryEvents,
    candidate_fine_pair: Mapping[str, object],
    reference_fine_pair: Mapping[str, object],
    multiplier: float,
) -> dict[str, object]:
    observed = _event_time_metrics(fine, reference)
    identity = bool(observed["identity_exact"])
    if not identity:
        return {
            "status": "BLOCKED",
            "metrics": observed,
            "identity_exact_required": True,
        }
    if fine.time_s.size == 0:
        return {
            "status": "PASS",
            "metrics": observed,
            "identity_exact_required": True,
            "not_applicable": "no terminal boundary events",
        }
    roundoff = _event_roundoff_floor(fine, reference)
    candidate_metrics = _mapping(candidate_fine_pair, "candidate fine-pair event metrics")
    reference_metrics = _mapping(reference_fine_pair, "reference fine-pair event metrics")
    observed_metrics = _mapping(observed, "direct event metrics")
    limits = {
        name: multiplier * (float(candidate_metrics[name]) + float(reference_metrics[name]))
        + roundoff
        for name in ("rms_s", "maximum_s")
    }
    metric_pass = {
        name: bool(float(observed_metrics[name]) <= limit) for name, limit in limits.items()
    }
    return {
        "status": "PASS" if all(metric_pass.values()) else "BLOCKED",
        "metrics": observed,
        "limits": limits,
        "pass": metric_pass,
        "identity_exact_required": True,
        "uncertainty_components": {
            "candidate_fine_pair": {name: float(candidate_metrics[name]) for name in limits},
            "comsol_reference_pair": {name: float(reference_metrics[name]) for name in limits},
        },
    }


def _candidate_artifacts(
    prepared_root: Path,
) -> tuple[dict[str, Trajectory], dict[str, BoundaryEvents]]:
    prepare_report = _mapping(
        json.loads((prepared_root / "prepare_report.json").read_text(encoding="utf-8")),
        "prepare report",
    )
    if (
        prepare_report.get("status") != "PREPARED"
        or prepare_report.get("tool_revision") != CANDIDATE_TOOL_REVISION
    ):
        raise ValueError("prepared root does not belong to the M3-C3 candidate tool")
    trajectories: dict[str, Trajectory] = {}
    events: dict[str, BoundaryEvents] = {}
    for level, _dt_s in LEVELS:
        cell = prepared_root / "candidate" / level
        receipt = _mapping(
            json.loads((cell / "run_receipt.json").read_text(encoding="utf-8")),
            f"{level} run receipt",
        )
        trajectory_path = cell / "trajectory.csv"
        events_path = cell / "events.csv"
        counts = _mapping(receipt.get("result_counts"), f"{level} result counts")
        if (
            receipt.get("status") != "PASS"
            or receipt.get("trajectory_rows") != PARTICLE_COUNT * OUTPUT_COUNT
            or receipt.get("trajectory_sha256") != _sha256(trajectory_path)
            or receipt.get("derived_input_sha256") != prepare_report.get("derived_input_sha256")
            or receipt.get("release_state_sha256") != prepare_report.get("release_state_sha256")
            or counts.get("failure_events") != 0
            or receipt.get("events_sha256") != _sha256(events_path)
            or receipt.get("event_rows") != counts.get("boundary_events")
        ):
            raise ValueError(f"{level} candidate receipt or authority binding differs")
        trajectory = _read_trajectory(trajectory_path)
        receipt_speed = float(receipt.get("maximum_observed_particle_speed_m_s", math.nan))
        if not math.isclose(
            receipt_speed,
            _maximum_speed(trajectory),
            rel_tol=32.0 * np.finfo(np.float64).eps,
            abs_tol=0.0,
        ):
            raise ValueError(f"{level} maximum-speed receipt differs from trajectory")
        trajectories[level] = trajectory
        events[level] = _read_events(events_path)
    return trajectories, events


def _reference_artifacts(
    prepared_root: Path, reference_root: Path, expected_step_s: float
) -> tuple[Trajectory, BoundaryEvents]:
    report = _mapping(
        json.loads((prepared_root / "prepare_report.json").read_text(encoding="utf-8")),
        "prepare report",
    )
    release = reference_root / "three_current_release_state.csv"
    if _sha256(release) != report["release_state_sha256"]:
        raise ValueError("COMSOL reference did not consume the candidate-owned Z0 release table")
    summary = _mapping(
        json.loads((reference_root / "normalization_summary.json").read_text(encoding="utf-8")),
        "normalization summary",
    )
    receipt = _mapping(
        json.loads((reference_root / "run_receipt.json").read_text(encoding="utf-8")),
        "reference run receipt",
    )
    trajectory_path = reference_root / "trajectory_reference.csv"
    events_path = reference_root / "events_reference.csv"
    artifacts = _mapping(summary.get("artifacts"), "reference artifacts")
    if (
        summary.get("status") != "COMPLETE_NORMALIZED_NOT_EVALUATED"
        or summary.get("case_id") != CASE_ID
        or summary.get("particle_count") != PARTICLE_COUNT
        or summary.get("output_count") != OUTPUT_COUNT
        or summary.get("trajectory_rows") != PARTICLE_COUNT * OUTPUT_COUNT
        or float(summary.get("fixed_rk4_step_s", math.nan)) != expected_step_s
        or artifacts.get("trajectory_reference.csv") != _sha256(trajectory_path)
        or artifacts.get("events_reference.csv") != _sha256(events_path)
        or receipt.get("status") != "PASS"
        or receipt.get("case_id") != CASE_ID
    ):
        raise ValueError("COMSOL reference receipt or normalized trajectory differs")
    events = _read_events(events_path)
    if summary.get("event_count") != int(events.particle_id.size):
        raise ValueError("COMSOL reference event count differs from normalized events")
    return _read_trajectory(trajectory_path), events


def _fates(trajectory: Trajectory) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in trajectory.lifecycle[-1]).items()))


def _maximum_speed(trajectory: Trajectory) -> float:
    velocity = trajectory.values[..., 2:4][trajectory.finite]
    return float(np.max(np.linalg.norm(velocity, axis=1)))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate(
    prepared_root: Path,
    reference_roots: tuple[Path, ...],
    output: Path,
) -> dict[str, object]:
    """Evaluate candidate and COMSOL three-level convergence and agreement."""

    prepared_root = prepared_root.resolve()
    if len(reference_roots) != len(COMSOL_REFERENCE_LEVELS):
        raise ValueError("exactly three COMSOL reference roots are required")
    reference_roots = tuple(path.resolve() for path in reference_roots)
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    policy = _policy(prepared_root)
    candidates, candidate_events = _candidate_artifacts(prepared_root)
    references: dict[str, Trajectory] = {}
    reference_events: dict[str, BoundaryEvents] = {}
    for (level, step_s), reference_root in zip(
        COMSOL_REFERENCE_LEVELS, reference_roots, strict=True
    ):
        references[level], reference_events[level] = _reference_artifacts(
            prepared_root, reference_root, step_s
        )
    self_result = _self_convergence(candidates, policy)
    fine_pair = _mapping(self_result["medium_vs_fine"], "candidate fine pair")
    reference_self = _self_convergence(references, policy, COMSOL_REFERENCE_LEVELS)
    reference_fine_pair = _mapping(reference_self["medium_vs_fine"], "COMSOL reference fine pair")
    direct = _direct_gate(
        candidates[FINE_LEVEL],
        references[COMSOL_REFERENCE_LEVELS[-1][0]],
        fine_pair,
        reference_fine_pair,
        policy,
    )
    event_self = _event_self_convergence(candidate_events, policy.minimum_order)
    event_fine_pair = _mapping(event_self["medium_vs_fine"], "candidate event fine pair")
    reference_event_self = _event_self_convergence(
        reference_events, policy.minimum_order, COMSOL_REFERENCE_LEVELS
    )
    reference_event_fine_pair = _mapping(
        reference_event_self["medium_vs_fine"], "COMSOL reference event fine pair"
    )
    event_direct = _event_direct_gate(
        candidate_events[FINE_LEVEL],
        reference_events[COMSOL_REFERENCE_LEVELS[-1][0]],
        event_fine_pair,
        reference_event_fine_pair,
        policy.direct_multiplier,
    )
    overall = all(
        result["status"] == "PASS"
        for result in (
            self_result,
            reference_self,
            direct,
            event_self,
            reference_event_self,
            event_direct,
        )
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "PASS" if overall else "BLOCKED",
        "case_id": CASE_ID,
        "scope": {
            "brownian_active": False,
            "candidate_integrator": "exponential_midpoint",
            "candidate_steps_s": [value for _, value in LEVELS],
            "comsol_integrator": "classical_rk4",
            "comsol_reference_steps_s": list(COMSOL_REFERENCE_STEPS_S),
            "comsol_is_observed_reference_not_truth": True,
            "boundary_event_scope": (
                "terminal particle/outcome/semantic identity plus event-time convergence"
            ),
            "reference_step_convergence": "MEASURED_WITH_THREE_FIXED_RK4_LEVELS",
        },
        "gate_policy": {
            "source": GATE_POLICY_RELATIVE_PATH,
            "source_sha256": policy.source_sha256,
            "minimum_rms_order": policy.minimum_order,
            "fine_pair_relative_l2_limits": policy.fine_relative,
            "inherited_absolute_limits": policy.inherited_absolute,
            "direct_limit_rule": (
                "max(inherited_gate,4*(candidate_fine_pair+comsol_fine_pair)"
                "+float64_roundoff_floor)"
            ),
            "event_direct_limit_rule": (
                "4*(candidate_fine_pair+comsol_fine_pair)+float64_roundoff_floor"
            ),
            "result_dependent_tolerance_tuning": "PROHIBITED",
        },
        "self_convergence": self_result,
        "comsol_reference_convergence": reference_self,
        "candidate_fine_vs_comsol": direct,
        "boundary_event_self_convergence": event_self,
        "comsol_boundary_event_convergence": reference_event_self,
        "boundary_events_fine_vs_comsol": event_direct,
        "final_fates": {
            **{level: _fates(trajectory) for level, trajectory in candidates.items()},
            **{f"comsol_{level}": _fates(reference) for level, reference in references.items()},
        },
        "observed_particle_speed_m_s": {
            **{level: _maximum_speed(trajectory) for level, trajectory in candidates.items()},
            **{
                f"comsol_{level}": _maximum_speed(reference)
                for level, reference in references.items()
            },
            "preparation_envelope_m_s": 1.0e3,
            "preparation_envelope_is_not_runtime_gate": True,
            "runtime_applicability_owner": "physics relative-ion-speed gate at 1.0e6 m/s",
        },
        "artifacts": {
            "candidate": {
                level: {
                    "trajectory_sha256": candidates[level].sha256,
                    "events_sha256": candidate_events[level].sha256,
                }
                for level, _dt_s in LEVELS
            },
            "reference": {
                level: {
                    "trajectory_sha256": references[level].sha256,
                    "events_sha256": reference_events[level].sha256,
                }
                for level, _step_s in COMSOL_REFERENCE_LEVELS
            },
        },
        "claim_policy": {
            "three_current_same_field_agreement": "PASS" if overall else "NOT_ESTABLISHED",
            "physical_model_validity": "NOT_CLAIMED_BY_NUMERICAL_AGREEMENT",
            "universal_comsol_equivalence": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--reference-coarse-root", required=True, type=Path)
    parser.add_argument("--reference-medium-root", required=True, type=Path)
    parser.add_argument("--reference-fine-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    evaluate(
        arguments.prepared_root,
        (
            arguments.reference_coarse_root,
            arguments.reference_medium_root,
            arguments.reference_fine_root,
        ),
        arguments.output,
    )


if __name__ == "__main__":
    main()
