"""Characterize bound 2-DOF deterministic pre-event trajectories externally.

Integration steps and observation times have independent owners. Step-pair
differences are sensitivity measurements, never certified error bounds. A
registered physical budget can be evaluated descriptively while unknown model
correspondence keeps qualified solver parity NOT_TESTED.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from chamber_particles import load_case, open_result
from tools.vv.comsol.meaning_preflight import InventoryError, require_supported_comparison

TOOL_REVISION: Final = "deterministic_pre_event_trajectory_v3"
_MODES: Final = {"cartesian_xy": ("x", "y"), "axisymmetric_rz_meridional": ("r", "z")}
_LEVELS: Final = ("coarse", "medium", "fine")
_QUANTITIES: Final = {"position": (0, 2, "m"), "velocity": (2, 4, "m/s"), "charge": (4, 5, "e")}
type Record = dict[str, Any]
type StateArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class _Trajectory:
    ids: np.ndarray
    times: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    active: np.ndarray
    support_known: np.ndarray
    support_inside: np.ndarray
    evidence: Record
    scope_issues: tuple[str, ...]

    @property
    def usable(self) -> np.ndarray:
        return self.observed & self.active & np.isfinite(self.values).all(axis=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Record:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _path(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty string")
    return (base / value).expanduser().resolve()


def _artifact(path: Path) -> Record:
    return {"path": str(path), "sha256": _sha256(path)}


def _pointer(value: object, pointer: object) -> object:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("metadata integration_step_pointer must be an absolute JSON pointer")
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list):
            value = value[int(key)]
        else:
            raise ValueError(f"JSON pointer cannot traverse {pointer}")
    return value


def _clock(config: Record) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(config["observation_times_s"], dtype=np.float64)
    ids = np.asarray(config["particle_ids"], dtype=np.int64)
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError("observation_times_s requires at least two finite times")
    if not bool((np.diff(times) > 0).all()):
        raise ValueError("observation times must be strictly increasing; no resampling is implicit")
    if ids.ndim != 1 or not ids.size or np.unique(ids).size != ids.size:
        raise ValueError("particle_ids must be a nonempty unique stable-ID list")
    if any(type(pid) is not int for pid in config["particle_ids"]):
        raise ValueError("particle IDs must be integers")
    return np.sort(ids), times


def _empty(ids: np.ndarray, times: np.ndarray) -> StateArrays:
    shape = (times.size, ids.size)
    return (
        np.full((*shape, 5), np.nan),
        np.zeros(shape, dtype=np.bool_),
        np.zeros(shape, dtype=np.bool_),
        np.zeros(shape, dtype=np.bool_),
        np.zeros(shape, dtype=np.bool_),
    )


def _store(
    arrays: StateArrays,
    key: tuple[int, int],
    state: np.ndarray,
    active: bool,
    support: bool | None,
) -> None:
    values, observed, lifecycle, known, inside = arrays
    if observed[key]:
        raise ValueError(f"duplicate selected particle/time key {key}")
    values[key] = state
    observed[key] = True
    lifecycle[key] = active
    known[key] = support is not None
    inside[key] = bool(support)


def _csv_columns(level: Record) -> tuple[list[str], str | None]:
    mapping = level["columns"]
    state_columns = [*mapping["position"], *mapping["velocity"], mapping["charge"]]
    if len(mapping["position"]) != 2 or len(mapping["velocity"]) != 2:
        raise ValueError(
            "only two position/velocity components are supported; third DOF is excluded"
        )
    required = [mapping["particle_id"], mapping["time"], *state_columns, mapping["lifecycle"]]
    if len(set(required)) != len(required) or not all(isinstance(key, str) for key in required):
        raise ValueError("column mapping must contain distinct string column names")
    return state_columns, mapping.get("support_inside")


def _csv_evidence(level: Record, base: Path, path: Path) -> tuple[Record, list[str]]:
    evidence = {"trajectory": _artifact(path), "execution": "UNVERIFIED_EXPORT"}
    issues = ["CSV does not by itself prove a completed deterministic solver run"]
    metadata = level.get("metadata")
    if metadata is not None:
        metadata_path = _path(base, metadata["path"])
        binding = _artifact(metadata_path)
        if binding["sha256"] != metadata["sha256"]:
            raise ValueError("metadata hash differs from registered identity")
        actual_step = _pointer(_json(metadata_path), metadata["integration_step_pointer"])
        if actual_step != level["step_s"]:
            raise ValueError("declared integration step differs from bound metadata readback")
        evidence.update(metadata=binding, execution="BOUND_METADATA_NOT_EXECUTION_PROOF")
    return evidence, issues


def _read_csv(level: Record, base: Path, ids: np.ndarray, times: np.ndarray) -> _Trajectory:
    path = _path(base, level["csv_path"])
    columns, support_column = _csv_columns(level)
    mapping = level["columns"]
    arrays = _empty(ids, times)
    selected_ids = {int(pid): index for index, pid in enumerate(ids)}
    selected_times = {float(time): index for index, time in enumerate(times)}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*columns, mapping["particle_id"], mapping["time"], mapping["lifecycle"]}
        if support_column is not None:
            required.add(support_column)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing mapped CSV columns")
        for row in reader:
            pid, time = int(row[mapping["particle_id"]]), float(row[mapping["time"]])
            if pid not in selected_ids or time not in selected_times:
                continue
            support = None if support_column is None else _support_value(row[support_column])
            _store(
                arrays,
                (selected_times[time], selected_ids[pid]),
                np.asarray([float(row[name]) for name in columns]),
                row[mapping["lifecycle"]] == mapping.get("active_value", "active"),
                support,
            )
    evidence, issues = _csv_evidence(level, base, path)
    issues.append("external event-free/deterministic meaning requires bound native evidence")
    return _Trajectory(ids, times, *arrays, evidence, tuple(issues))


def _support_value(value: str) -> bool | None:
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    if value in {"unknown", ""}:
        return None
    raise ValueError("support column requires true/false/unknown (or 1/0)")


def _result_evidence(level: Record, base: Path, mode: str) -> tuple[Any, Record]:
    case = load_case(_path(base, level["case_path"]))
    result = open_result(_path(base, level["result_path"]))
    manifest: Record = dict(result.manifest)
    if manifest["case_file_hash"] != case.case_file_hash:
        raise ValueError("result manifest is not bound to supplied case file")
    if manifest["data_content_hash"] != case.content_hash:
        raise ValueError("result manifest is not bound to supplied canonical data")
    if manifest["time"]["dt_s"] != level["step_s"] or case.spec.time.dt_s != level["step_s"]:
        raise ValueError("declared integration step differs from actual case/run manifest")
    if manifest["motion_mode"] != mode or case.spec.motion.mode != mode:
        raise ValueError("result coordinate mode differs from series")
    document = yaml.safe_load(case.case_path.read_text(encoding="utf-8"))
    evidence = {
        "execution": "COMPLETED_PUBLIC_RESULT_BOUND_TO_CASE",
        "result_path": str(result.path),
        "manifest": _artifact(result.path / "run.json"),
        "case": _artifact(case.case_path),
        "data_content_hash": case.content_hash,
        "resolved": manifest["resolved"],
        "time": manifest["time"],
        "counts": manifest["counts"],
        "engine_algorithm_revision": manifest["engine_algorithm_revision"],
        "physical_problem": {
            "data_content_hash": case.content_hash,
            "configuration": {
                key: document.get(key)
                for key in ("motion", "physics", "sources", "boundaries", "topology")
            },
        },
        "numerical_method": {
            "integrator": manifest["resolved"]["integrator"],
            "event": manifest["event"],
            "algorithm_revisions": {
                key: value for key, value in manifest.items() if key.endswith("revision")
            },
        },
        "segments": [_artifact(path) for path in result.segment_paths],
    }
    return result, evidence


def _read_result(
    level: Record, base: Path, ids: np.ndarray, times: np.ndarray, mode: str
) -> _Trajectory:
    result, evidence = _result_evidence(level, base, mode)
    arrays = _empty(ids, times)
    indices = {int(pid): index for index, pid in enumerate(ids)}
    clocks = {float(time): index for index, time in enumerate(times)}
    for frame in chain(result.iter_frames(), result.iter_probes()):
        if frame.time_s not in clocks:
            continue
        for row, pid in enumerate(frame.particle_id):
            if int(pid) in indices:
                state = np.r_[
                    frame.position_m[row], frame.velocity_m_s[row], frame.charge_number[row]
                ]
                key = (clocks[frame.time_s], indices[int(pid)])
                active = frame.lifecycle[row] == 1
                if arrays[1][key]:
                    if not np.array_equal(arrays[0][key], state) or arrays[2][key] != active:
                        raise ValueError("public frame/probe states disagree at a selected key")
                    continue
                _store(arrays, key, state, active, True if active else None)
    issues = _result_scope_issues(result, ids, times)
    return _Trajectory(ids, times, *arrays, evidence, tuple(issues))


def _result_scope_issues(result: Any, ids: np.ndarray, times: np.ndarray) -> list[str]:
    issues: list[str] = []
    if result.manifest.get("brownian_composition_revision") is not None:
        issues.append("stochastic runs are outside deterministic pre-event qualification")
    releases = result.read_release_events()
    selected = np.isin(releases.particle_id, ids)
    if np.count_nonzero(selected) != ids.size or not bool(
        (releases.time_s[selected] == times[0]).all()
    ):
        issues.append(
            "first observation is not the actual common release state of every selected ID"
        )
    for batch in result.iter_boundary_event_batches():
        if bool((np.isin(batch.particle_id, ids) & (batch.time_s <= times[-1])).any()):
            issues.append("selected cohort has a boundary event before the final observation")
            break
    failure = result.read_failure_events()
    if bool((np.isin(failure.particle_id, ids) & (failure.time_s <= times[-1])).any()):
        issues.append("selected cohort has a failure before the final observation")
    return issues


def _read_level(
    level: Record, base: Path, ids: np.ndarray, times: np.ndarray, mode: str
) -> _Trajectory:
    if ("result_path" in level) == ("csv_path" in level):
        raise ValueError("each level requires exactly one result_path or csv_path")
    if "result_path" in level:
        return _read_result(level, base, ids, times, mode)
    return _read_csv(level, base, ids, times)


def _coverage(trajectory: _Trajectory) -> Record:
    return {
        "denominator": int(trajectory.observed.size),
        "observed": int(trajectory.observed.sum()),
        "missing": int((~trajectory.observed).sum()),
        "nonfinite_observed": int(
            (trajectory.observed & ~np.isfinite(trajectory.values).all(axis=2)).sum()
        ),
        "active_observed": int((trajectory.observed & trajectory.active).sum()),
        "support_unknown_observed": int((trajectory.observed & ~trajectory.support_known).sum()),
        "support_outside_observed": int(
            (trajectory.observed & trajectory.support_known & ~trajectory.support_inside).sum()
        ),
        "usable": int(trajectory.usable.sum()),
    }


def _statistics(magnitude: np.ndarray, mask: np.ndarray) -> Record:
    selected = magnitude[mask]
    if not selected.size:
        return {"count": 0, "rms": None, "maximum": None}
    return {
        "count": int(selected.size),
        "rms": float(np.sqrt(np.mean(selected * selected))),
        "maximum": float(np.max(selected)),
    }


def _magnitude_metrics(magnitude: np.ndarray, mask: np.ndarray, trajectory: _Trajectory) -> Record:
    result = _statistics(magnitude, mask)
    if result["count"]:
        worst = np.unravel_index(np.argmax(np.where(mask, magnitude, -np.inf)), mask.shape)
        result["maximum_location"] = {
            "particle_id": int(trajectory.ids[worst[1]]),
            "time_s": float(trajectory.times[worst[0]]),
        }
    result["per_time"] = [
        {"time_s": float(time), **_statistics(magnitude[row], mask[row])}
        for row, time in enumerate(trajectory.times)
    ]
    result["per_particle"] = [
        {"particle_id": int(pid), **_statistics(magnitude[:, column], mask[:, column])}
        for column, pid in enumerate(trajectory.ids)
    ]
    return result


def _difference(left: _Trajectory, right: _Trajectory) -> Record:
    if not np.array_equal(left.ids, right.ids) or not np.array_equal(left.times, right.times):
        raise ValueError("stable IDs and explicit observation clocks differ")
    difference = left.values - right.values
    mask = left.usable & right.usable
    result: Record = {
        "denominator": int(mask.size),
        "paired_usable": int(mask.sum()),
        "mask": "joint observed finite active; support/event qualification separately reported",
        "weighting": "equal retained particle-observation rows; not time weighted",
    }
    for name, (start, stop, unit) in _QUANTITIES.items():
        magnitude = np.linalg.norm(difference[:, :, start:stop], axis=2)
        result[name] = {
            "unit": unit,
            "definition": "Euclidean component norm"
            if stop - start == 2
            else "absolute scalar difference",
            **_magnitude_metrics(magnitude, mask, left),
        }
    result["components"] = [
        {
            "state_component": index,
            "unit": ("m", "m", "m/s", "m/s", "e")[index],
            "definition": "absolute component difference; indices are position0,position1,velocity0,velocity1,Z",
            **_magnitude_metrics(np.abs(difference[:, :, index]), mask, left),
        }
        for index in range(5)
    ]
    return result


def _initial(trajectories: list[_Trajectory]) -> Record:
    first = trajectories[0]
    complete = all(bool(item.usable[0].all()) for item in trajectories)
    equal = complete and all(
        np.array_equal(first.values[0], item.values[0]) for item in trajectories[1:]
    )
    digest = hashlib.sha256(
        first.ids.tobytes() + first.times[:1].tobytes() + first.values[0].tobytes()
    ).hexdigest()
    return {
        "all_initial_rows_usable": complete,
        "exact_x_v_Z_id_time_equality": equal,
        "sha256": digest if complete else None,
        "definition": "sorted stable IDs, actual first observation time, both position/velocity components and charge",
    }


def _orders(first: Record, second: Record, steps: list[float]) -> Record:
    ratios = [steps[0] / steps[1], steps[1] / steps[2]]
    result: Record = {
        "step_ratios": ratios,
        "interpretation": "observed sensitivity order only; no accuracy certification or mandatory smooth order for nonsmooth fields",
    }
    for quantity in _QUANTITIES:
        values: Record = {}
        for metric in ("rms", "maximum"):
            a, b = first[quantity][metric], second[quantity][metric]
            values[metric] = (
                math.log(a / b) / math.log(ratios[0])
                if a is not None
                and b is not None
                and a > 0
                and b > 0
                and math.isclose(*ratios, rel_tol=1e-12)
                else None
            )
        result[quantity] = values
    return result


def _read_series(path: Path) -> tuple[Record, dict[str, _Trajectory]]:
    config = _json(path)
    mode = config["coordinate_mode"]
    if mode not in _MODES:
        raise ValueError("only Cartesian XY and RZ meridional 2-DOF are supported")
    ids, times = _clock(config)
    steps = [float(config["levels"][name]["step_s"]) for name in _LEVELS]
    if (
        not all(math.isfinite(step) and step > 0 for step in steps)
        or not steps[0] > steps[1] > steps[2]
    ):
        raise ValueError("coarse > medium > fine positive integration steps are required")
    return config, {
        name: _read_level(config["levels"][name], path.parent, ids, times, mode) for name in _LEVELS
    }


def _series_qualification(trajectories: dict[str, _Trajectory]) -> tuple[Record, Record, list[str]]:
    """Qualify the sampled cohort and unchanged problem before interpreting changes."""
    initial = _initial(list(trajectories.values()))
    issues = [
        f"{name}: {issue}" for name, item in trajectories.items() for issue in item.scope_issues
    ]
    coverage = {name: _coverage(item) for name, item in trajectories.items()}
    for name, item in coverage.items():
        if item["usable"] != item["denominator"] or item["support_outside_observed"]:
            issues.append(f"{name}: incomplete finite active inside-support coverage")
    if not initial["exact_x_v_Z_id_time_equality"]:
        issues.append("initial states differ or are incomplete across integration levels")
    bindings = [item.evidence for item in trajectories.values()]
    if all("physical_problem" in item for item in bindings):
        issues.extend(
            f"{key} differs across integration levels"
            for key in ("physical_problem", "numerical_method")
            if any(item[key] != bindings[0][key] for item in bindings[1:])
        )
    return initial, coverage, issues


def characterize(series_path: Path) -> Record:
    """Read actual bound levels; summarize self-sensitivity without cross data."""
    path = series_path.expanduser().resolve()
    config, trajectories = _read_series(path)
    coarse_medium = _difference(trajectories["coarse"], trajectories["medium"])
    medium_fine = _difference(trajectories["medium"], trajectories["fine"])
    initial, coverage, issues = _series_qualification(trajectories)
    fine = trajectories["fine"]
    steps = [float(config["levels"][name]["step_s"]) for name in _LEVELS]
    return {
        "schema_version": 2,
        "tool_revision": TOOL_REVISION,
        "report_kind": "single_solver_self_sensitivity",
        "label": config["label"],
        "series": _artifact(path),
        "status": "CHARACTERIZED",
        "qualification_issues": issues,
        "scope": {
            "coordinate_mode": config["coordinate_mode"],
            "component_basis": list(_MODES[config["coordinate_mode"]]),
            "particle_ids": fine.ids.tolist(),
            "observation_times_s": fine.times.tolist(),
            "time_window_s": [float(fine.times[0]), float(fine.times[-1])],
            "particles": int(fine.ids.size),
            "observations": int(fine.times.size),
            "deterministic_pre_event": "SUPPORTED" if not issues else "NOT_TESTED",
            "excluded": [
                "third DOF",
                "statistical equivalence",
                "event/post-event accuracy",
                "physical validation",
            ],
        },
        "integration_steps_s": dict(zip(_LEVELS, steps, strict=True)),
        "artifacts": {name: item.evidence for name, item in trajectories.items()},
        "initial_state": initial,
        "coverage": coverage,
        "coarse_medium": coarse_medium,
        "medium_fine": medium_fine,
        "observed_order": _orders(coarse_medium, medium_fine, steps),
        "accuracy_claim": "Step-pair changes are sensitivity observations, not certified error bounds or COMSOL precision targets.",
    }


def _load_report(path: Path, kind: str) -> Record:
    report = _json(path)
    if report.get("tool_revision") != TOOL_REVISION or report.get("report_kind") != kind:
        raise ValueError("report kind or evaluator revision is incompatible; no historical shim")
    return report


def _validated_budgets(budgets: Record) -> Record:
    for quantity in _QUANTITIES:
        for metric in ("rms", "maximum"):
            value = budgets[quantity][metric]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "physical budgets require finite nonnegative RMS/maximum values in SI"
                )
    return budgets


def _self_sensitivity_budgets(policy: Record) -> Record | None:
    budgets = policy.get("self_sensitivity_budgets")
    if budgets is None:
        return None
    return {side: _validated_budgets(budgets[side]) for side in ("candidate", "reference")}


def register_budget(candidate_path: Path, reference_path: Path, policy_path: Path) -> Record:
    """Lock user-selected physical budgets before reading cross differences."""
    candidate = _load_report(candidate_path, "single_solver_self_sensitivity")
    reference = _load_report(reference_path, "single_solver_self_sensitivity")
    policy = _json(policy_path)
    budgets = _validated_budgets(policy["physical_budgets"])
    return {
        "schema_version": 2,
        "tool_revision": TOOL_REVISION,
        "report_kind": "registered_physical_trajectory_budget",
        "status": "REGISTERED",
        "physical_budgets": budgets,
        "self_sensitivity_budgets": _self_sensitivity_budgets(policy),
        "policy": _artifact(policy_path.resolve()),
        "self_reports": {
            "candidate": _artifact(candidate_path.resolve()),
            "reference": _artifact(reference_path.resolve()),
        },
        "series": {"candidate": candidate["series"], "reference": reference["series"]},
        "meaning_preflight_inventory": policy.get("meaning_preflight_inventory"),
        "comparison_question": policy.get(
            "comparison_question", "same_canonical_field_solver_parity"
        ),
        "interpretation": "Budgets are physical decision allowances, not sums of finest-pair changes; registration makes no run, parity or error-bound claim.",
    }


def _locked_series(budget: Record, side: str) -> tuple[Record, dict[str, _Trajectory]]:
    identity = budget["series"][side]
    path = Path(identity["path"])
    if _sha256(path) != identity["sha256"]:
        raise ValueError(f"{side} series hash differs from registered budget")
    report_identity = budget["self_reports"][side]
    report_path = Path(report_identity["path"])
    if _sha256(report_path) != report_identity["sha256"]:
        raise ValueError(f"{side} self report hash differs from registered budget")
    original = _load_report(report_path, "single_solver_self_sensitivity")
    refreshed = characterize(path)
    if json.dumps(refreshed, sort_keys=True) != json.dumps(original, sort_keys=True):
        raise ValueError(f"{side} source artifacts differ from the characterized readback")
    return _read_series(path)


def _meaning(budget: Record, base: Path, left: _Trajectory, right: _Trajectory) -> Record:
    inventory = budget.get("meaning_preflight_inventory")
    if inventory is None:
        return {
            "status": "NOT_TESTED",
            "reason": "No bound supported meaning inventory; raw differences remain available",
        }
    observed = frozenset(
        item.evidence.get("manifest", item.evidence.get("trajectory"))["sha256"]
        for item in (left, right)
    )
    try:
        binding = require_supported_comparison(
            inventory, base, budget["comparison_question"], required_observed_sha256=observed
        )
    except InventoryError as error:
        return {"status": "NOT_TESTED", "reason": str(error)}
    return {"status": "SUPPORTED", "inventory": binding, "question": budget["comparison_question"]}


def _gates(metrics: Record, budgets: Record) -> Record:
    return {
        name: {
            metric: {
                "observed": metrics[name][metric],
                "physical_budget": budgets[name][metric],
                "unit": unit,
                "pass": metrics[name][metric] is not None
                and metrics[name][metric] <= budgets[name][metric],
            }
            for metric in ("rms", "maximum")
        }
        for name, (_, _, unit) in _QUANTITIES.items()
    }


def _budget_verdict(metrics: Record, gates: Record) -> str:
    passed = all(entry["pass"] for quantities in gates.values() for entry in quantities.values())
    if not metrics["paired_usable"]:
        return "NOT_TESTED"
    if not passed:
        return "FAIL"
    return "PASS" if metrics["paired_usable"] == metrics["denominator"] else "NOT_TESTED"


def _registered_policy(budget: Record) -> Path:
    """Keep the consumed allowance and meaning request identical to its policy."""
    identity = budget["policy"]
    path = Path(identity["path"])
    if _sha256(path) != identity["sha256"]:
        raise ValueError("registered physical policy hash differs")
    policy = _json(path)
    expected = {
        "physical_budgets": _validated_budgets(policy["physical_budgets"]),
        "self_sensitivity_budgets": _self_sensitivity_budgets(policy),
        "meaning_preflight_inventory": policy.get("meaning_preflight_inventory"),
        "comparison_question": policy.get(
            "comparison_question", "same_canonical_field_solver_parity"
        ),
    }
    if any(budget.get(key) != value for key, value in expected.items()):
        raise ValueError("budget values differ from registered physical policy")
    return path


def _macro_resolution(trajectories: dict[str, _Trajectory]) -> Record:
    evidence = [trajectories[name].evidence for name in _LEVELS]
    if any(item["execution"] != "COMPLETED_PUBLIC_RESULT_BOUND_TO_CASE" for item in evidence):
        return {
            "status": "NOT_TESTED",
            "reason": "Actual accepted macro partitions are not bound by public result manifests",
        }
    clocks = [(item["time"]["start_s"], item["time"]["end_s"]) for item in evidence]
    counts = [item["counts"]["macro_steps"] for item in evidence]
    times = trajectories["fine"].times
    fine_steps_per_window = float((times[-1] - times[0]) / evidence[-1]["time"]["dt_s"])
    coarse_steps_per_window = float((times[-1] - times[0]) / evidence[0]["time"]["dt_s"])
    supported = (
        len(set(clocks)) == 1 and counts[0] < counts[1] < counts[2] and coarse_steps_per_window > 1
    )
    return {
        "status": "SUPPORTED" if supported else "NOT_TESTED",
        "accepted_macro_counts": counts,
        "run_time_spans_s": clocks,
        "fine_base_steps_per_observation_window": fine_steps_per_window,
        "coarse_base_steps_per_observation_window": coarse_steps_per_window,
        "definition": "Conservative empirical prerequisite: common run span, increasing actual accepted macro counts and more than one coarsest base step in the selected window. Distinct dt labels alone do not prove refined partitions; hidden event/safety history is not reconstructed.",
    }


def _self_resolution(budget: Record, participants: dict[str, dict[str, _Trajectory]]) -> Record:
    allowances = budget.get("self_sensitivity_budgets")
    if allowances is None:
        return {
            "status": "NOT_TESTED",
            "reason": "No independently registered candidate/reference self_sensitivity_budgets",
        }
    sides: Record = {}
    for side, trajectories in participants.items():
        changes = _difference(trajectories["medium"], trajectories["fine"])
        gates = _gates(changes, allowances[side])
        sides[side] = {
            "observed_medium_fine_changes": {
                quantity: {metric: changes[quantity][metric] for metric in ("rms", "maximum")}
                for quantity in _QUANTITIES
            },
            "gates": gates,
            "empirical_target_verdict": _budget_verdict(changes, gates),
            "macro_resolution": _macro_resolution(trajectories),
        }
    known = all(item["macro_resolution"]["status"] == "SUPPORTED" for item in sides.values())
    met = all(item["empirical_target_verdict"] == "PASS" for item in sides.values())
    return {
        "status": ("PASS" if met else "FAIL") if known else "NOT_TESTED",
        "participants": sides,
        "interpretation": "Empirical resolution prerequisite only. These independent absolute SI allowances and finest-pair observations are not rigorous error bounds or reference-precision parity targets.",
    }


def compare(budget_path: Path) -> Record:
    """Refresh locked evidence and report raw differences independently of parity."""
    budget = _load_report(budget_path, "registered_physical_trajectory_budget")
    policy_path = _registered_policy(budget)
    candidate_config, candidate_levels = _locked_series(budget, "candidate")
    reference_config, reference_levels = _locked_series(budget, "reference")
    if candidate_config["coordinate_mode"] != reference_config["coordinate_mode"]:
        raise ValueError("cross comparison coordinate modes differ")
    left, right = candidate_levels["fine"], reference_levels["fine"]
    metrics = _difference(left, right)
    initial = _initial([left, right])
    meaning = _meaning(budget, policy_path.parent, left, right)
    issues = [
        *_series_qualification(candidate_levels)[2],
        *_series_qualification(reference_levels)[2],
    ]
    if not initial["exact_x_v_Z_id_time_equality"]:
        issues.append("cross-solver initial x/v/Z/id/time equality is not established")
    issues.extend(
        "full observed finite active inside-support coverage is not established"
        for item in (left, right)
        if not bool(item.usable.all()) or bool((item.support_known & ~item.support_inside).any())
    )
    resolution = _self_resolution(
        budget, {"candidate": candidate_levels, "reference": reference_levels}
    )
    qualified = meaning["status"] == "SUPPORTED" and not issues and resolution["status"] == "PASS"
    gates = _gates(metrics, budget["physical_budgets"])
    verdict = _budget_verdict(metrics, gates)
    return {
        "schema_version": 2,
        "tool_revision": TOOL_REVISION,
        "report_kind": "locked_cross_solver_trajectory_comparison",
        "status": "COMPARED",
        "budget": _artifact(budget_path.resolve()),
        "initial_state": initial,
        "coverage": {"candidate": _coverage(left), "reference": _coverage(right)},
        "trajectory_difference": metrics,
        "acceptance_gates": gates,
        "descriptive_physical_budget_verdict": verdict,
        "qualified_solver_parity": verdict if qualified else "NOT_TESTED",
        "meaning": meaning,
        "empirical_self_resolution": resolution,
        "qualification_issues": issues,
        "accuracy_claim": {
            "same_accuracy": "NOT_CLAIMED",
            "finest_pair_is_error_bound": False,
            "physical_model_validity": "NOT_TESTED",
            "boundary_accuracy": "NOT_TESTED_PRE_EVENT_SCOPE",
            "statistical_equivalence": "NOT_TESTED",
            "third_DOF": "UNSUPPORTED",
            "interpretation": "Raw finite-active differences and physical-budget gates are separate from qualified parity. No reference-precision matching, rigorous numerical bound, universal accuracy or physical validity follows from this comparison.",
        },
    }


def _read_pair_participant(path: Path) -> tuple[Record, _Trajectory]:
    config = _json(path)
    mode = config["coordinate_mode"]
    if mode not in _MODES:
        raise ValueError("only Cartesian XY and RZ meridional 2-DOF are supported")
    ids, times = _clock(config)
    step = config["level"]["step_s"]
    if (
        isinstance(step, bool)
        or not isinstance(step, int | float)
        or not math.isfinite(step)
        or step <= 0
    ):
        raise ValueError("pair descriptor integration step must be a finite positive SI value")
    return config, _read_level(config["level"], path.parent, ids, times, mode)


def compare_pair(candidate_path: Path, reference_path: Path, policy_path: Path) -> Record:
    """Characterize two saved participants without inventing convergence levels."""
    candidate_config, left = _read_pair_participant(candidate_path.resolve())
    reference_config, right = _read_pair_participant(reference_path.resolve())
    if candidate_config["coordinate_mode"] != reference_config["coordinate_mode"]:
        raise ValueError("cross comparison coordinate modes differ")
    policy = _json(policy_path)
    question = policy.get("comparison_question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("saved-pair policy requires an explicit comparison_question")
    metrics = _difference(left, right)
    initial = _initial([left, right])
    gates = _gates(metrics, _validated_budgets(policy["physical_budgets"]))
    verdict = _budget_verdict(metrics, gates)
    issues = [
        *left.scope_issues,
        *right.scope_issues,
        "single-pair saved-data comparison has no independent resolution or statistical-equivalence qualification",
    ]
    if not initial["exact_x_v_Z_id_time_equality"]:
        issues.append("paired initial x/v/Z/id/time differ or are incomplete")
    return {
        "schema_version": 2,
        "tool_revision": TOOL_REVISION,
        "report_kind": "saved_pair_trajectory_characterization",
        "status": "COMPARED",
        "comparison_intent": policy.get("comparison_intent", "saved_data_characterization"),
        "comparison_question": question,
        "known_model_differences": policy.get("known_model_differences", []),
        "policy": _artifact(policy_path.resolve()),
        "descriptors": {
            "candidate": _artifact(candidate_path.resolve()),
            "reference": _artifact(reference_path.resolve()),
        },
        "artifacts": {"candidate": left.evidence, "reference": right.evidence},
        "scope": {
            "coordinate_mode": candidate_config["coordinate_mode"],
            "particle_ids": left.ids.tolist(),
            "observation_times_s": left.times.tolist(),
            "integration_steps_s": {
                "candidate": candidate_config["level"]["step_s"],
                "reference": reference_config["level"]["step_s"],
            },
        },
        "initial_state": initial,
        "coverage": {"candidate": _coverage(left), "reference": _coverage(right)},
        "trajectory_difference": metrics,
        "acceptance_gates": gates,
        "descriptive_physical_budget_verdict": verdict,
        "qualified_solver_parity": "NOT_TESTED",
        "qualification_issues": issues,
        "accuracy_claim": "Raw same-ID finite-active differences include declared model/field/source/noise differences. Equal IDs/seeds do not couple stochastic paths. No third level, precision matching, certified error bound, event accuracy or physical validation is inferred.",
    }


def _write(path: Path, report: Record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("characterize")
    single.add_argument("--series", required=True, type=Path)
    register = commands.add_parser("register")
    register.add_argument("--candidate", required=True, type=Path)
    register.add_argument("--reference", required=True, type=Path)
    register.add_argument("--policy", required=True, type=Path)
    cross = commands.add_parser("compare")
    cross.add_argument("--budget", required=True, type=Path)
    pair = commands.add_parser("compare-pair")
    pair.add_argument("--candidate", required=True, type=Path)
    pair.add_argument("--reference", required=True, type=Path)
    pair.add_argument("--policy", required=True, type=Path)
    for command in (single, register, cross, pair):
        command.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "characterize":
        report = characterize(args.series)
    elif args.command == "register":
        report = register_budget(args.candidate, args.reference, args.policy)
    elif args.command == "compare":
        report = compare(args.budget)
    else:
        report = compare_pair(args.candidate, args.reference, args.policy)
    _write(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
