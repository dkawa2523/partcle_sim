"""Formal phased evaluator for the M3-C1 Case-A/Case-P 100 nm matrix.

The three commands are intentionally separate:

``characterize`` compares h/h2/h4 within one solver, ``register`` derives and
freezes cross-solver envelopes without reading post-t0 cross differences, and
``compare`` evaluates only the registered fine runs.  This file is external
V&V; it is not imported by the particle solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from tools.vv.comsol.normalize_m3c1_theory_100nm_30ms import (
    EVENT_COLUMNS,
    RUN_KEYS,
    STEP_LABELS,
    TRAJECTORY_COLUMNS,
    WORKFLOWS,
)

TOOL_REVISION: Final = "m3c1_theory_100nm_30ms_evaluator_v4"
CANDIDATE_COLUMNS: Final = (*TRAJECTORY_COLUMNS[:7], "lifecycle")
CANDIDATE_EVENT_COLUMNS: Final = (
    "particle_id",
    "event_ordinal",
    "event_time_s",
    "hit_r_m",
    "hit_z_m",
    "normal_r",
    "normal_z",
    "pre_velocity_r_m_per_s",
    "pre_velocity_z_m_per_s",
    "post_velocity_r_m_per_s",
    "post_velocity_z_m_per_s",
    "charge_number_pre_e",
    "charge_number_post_e",
    "boundary_id",
    "material_id",
    "law",
    "outcome",
    "localization_residual_m",
    "position_budget_m",
    "time_budget_s",
)
type Solver = Literal["candidate", "reference"]


@dataclass(frozen=True)
class EvaluationConfig:
    path: Path
    sha256: str
    raw: dict[str, object]
    particle_count: int
    times_s: tuple[float, ...]
    steps_s_by_workflow: dict[str, tuple[float, float, float]]
    charge_lipschitz_s_inv_by_workflow: dict[str, float]
    maximum_dt_charge_lipschitz: float
    minimum_order: float
    initial_ulp_multiplier: float
    roundoff_multiplier: float
    safety_factor: float


@dataclass(frozen=True)
class Trajectory:
    path: Path
    sha256: str
    state: np.ndarray
    lifecycle: np.ndarray
    final_fate: np.ndarray
    stop_time_s: np.ndarray
    present: np.ndarray


@dataclass(frozen=True)
class Event:
    particle_id: int
    ordinal: int
    time_s: float
    fate: str
    group: str
    directly_observed: bool
    position_m: tuple[float, float] | None
    charge_e: float | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
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


def _schedule(raw_segments: object) -> tuple[float, ...]:
    if not isinstance(raw_segments, list):
        raise ValueError("matrix.output_schedule_segments must be a list")
    values: list[float] = []
    for raw in raw_segments:
        segment = _mapping(raw, "output schedule segment")
        start = _number(segment.get("start_s"), "schedule start")
        stop = _number(segment.get("stop_s"), "schedule stop")
        step = _number(segment.get("step_s"), "schedule step")
        count = round((stop - start) / step)
        part = [start + index * step for index in range(count + 1)]
        part[-1] = stop
        if step <= 0.0 or stop < start or (values and part[0] <= values[-1]):
            raise ValueError("output schedule segments are invalid")
        values.extend(part)
    return tuple(values)


def _workflow_steps(
    raw: object, workflow: str, end_s: float, maximum_product: float
) -> tuple[tuple[float, float, float], float]:
    record = _mapping(raw, f"workflows.{workflow}")
    raw_steps = record.get("fixed_rk4_steps_s")
    if not isinstance(raw_steps, list) or len(raw_steps) != len(RUN_KEYS):
        raise ValueError(f"{workflow} fixed_rk4_steps_s must contain h, h/2, h/4")
    steps = tuple(_number(value, f"{workflow} fixed RK4 step") for value in raw_steps)
    if (
        any(step <= 0.0 for step in steps)
        or steps[0] != 2.0 * steps[1]
        or steps[1] != 2.0 * steps[2]
    ):
        raise ValueError(f"{workflow} fixed RK4 steps must be positive exact h, h/2, h/4")
    lipschitz = _number(record.get("charge_lipschitz_s_inv"), f"{workflow} charge Lipschitz bound")
    if lipschitz <= 0.0:
        raise ValueError(f"{workflow} charge Lipschitz bound must be positive")
    for step in steps:
        count = round(end_s / step)
        if not math.isclose(count * step, end_s, rel_tol=0.0, abs_tol=math.ulp(end_s)):
            raise ValueError(f"{workflow} fixed step does not divide the configured time span")
        if step * lipschitz > maximum_product:
            raise ValueError(f"{workflow} fixed step violates the dt-charge Lipschitz limit")
    return (steps[0], steps[1], steps[2]), lipschitz


def _load_config(path: Path) -> EvaluationConfig:
    resolved = path.expanduser().resolve()
    raw = _mapping(json.loads(resolved.read_text(encoding="utf-8")), "configuration")
    if raw.get("evaluation_id") != "M3-C1-theory-consistent-100nm-30ms":
        raise ValueError("unexpected M3-C1 evaluation_id")
    if raw.get("evaluation_revision") != 2:
        raise ValueError("unsupported M3-C1 evaluation revision")
    matrix = _mapping(raw.get("matrix"), "matrix")
    times = _schedule(matrix.get("output_schedule_segments"))
    if len(times) != _positive_int(matrix.get("output_count"), "matrix.output_count"):
        raise ValueError("output schedule count differs")
    if matrix.get("run_keys") != list(RUN_KEYS):
        raise ValueError("matrix.run_keys must be coarse, medium, fine")
    workflows_raw = _mapping(raw.get("workflows"), "workflows")
    if set(workflows_raw) != set(WORKFLOWS):
        raise ValueError("configuration workflows must be exactly caseA and caseP")
    end_s = _number(matrix.get("time_end_s"), "matrix.time_end_s")
    steps_by_workflow: dict[str, tuple[float, float, float]] = {}
    lipschitz_by_workflow: dict[str, float] = {}
    acceptance = _mapping(raw.get("acceptance"), "acceptance")
    maximum_product = _number(
        acceptance.get("maximum_dt_charge_lipschitz"),
        "maximum dt-charge Lipschitz product",
    )
    if maximum_product <= 0.0:
        raise ValueError("maximum dt-charge Lipschitz product must be positive")
    for workflow in WORKFLOWS:
        steps, lipschitz = _workflow_steps(
            workflows_raw[workflow], workflow, end_s, maximum_product
        )
        steps_by_workflow[workflow] = steps
        lipschitz_by_workflow[workflow] = lipschitz
    values = {
        "minimum_order": _number(acceptance.get("minimum_rms_order"), "minimum RMS order"),
        "initial": _number(
            acceptance.get("initial_state_ulp_multiplier"), "initial-state ULP multiplier"
        ),
        "roundoff": _number(acceptance.get("roundoff_multiplier"), "roundoff multiplier"),
        "safety": _number(
            acceptance.get("cross_envelope_safety_factor"), "cross-envelope safety factor"
        ),
    }
    if any(value <= 0.0 for value in values.values()):
        raise ValueError("all acceptance constants must be positive")
    return EvaluationConfig(
        path=resolved,
        sha256=_sha256(resolved),
        raw=raw,
        particle_count=_positive_int(matrix.get("particle_count"), "particle count"),
        times_s=times,
        steps_s_by_workflow=steps_by_workflow,
        charge_lipschitz_s_inv_by_workflow=lipschitz_by_workflow,
        maximum_dt_charge_lipschitz=maximum_product,
        minimum_order=values["minimum_order"],
        initial_ulp_multiplier=values["initial"],
        roundoff_multiplier=values["roundoff"],
        safety_factor=values["safety"],
    )


def _trajectory_path(root: Path, workflow: str, solver: Solver, label: str) -> Path:
    prefix = "candidate" if solver == "candidate" else "reference"
    return root / workflow / f"{prefix}_trajectory_{label}.csv"


def _event_path(root: Path, workflow: str, solver: Solver, label: str) -> Path:
    prefix = "candidate" if solver == "candidate" else "reference"
    return root / workflow / f"{prefix}_events_{label}.csv"


def _step_sequence(
    config: EvaluationConfig, workflow: str, actual_steps_s: tuple[float, ...] | None = None
) -> list[dict[str, object]]:
    lipschitz = config.charge_lipschitz_s_inv_by_workflow[workflow]
    steps = actual_steps_s or config.steps_s_by_workflow[workflow]
    return [
        {
            "run_key": run_key,
            "step_s": step_s,
            "charge_lipschitz_s_inv": lipschitz,
            "dt_charge_lipschitz": step_s * lipschitz,
            "maximum_dt_charge_lipschitz": config.maximum_dt_charge_lipschitz,
        }
        for run_key, step_s in zip(RUN_KEYS, steps, strict=True)
    ]


def _time_index(time_s: float, config: EvaluationConfig, path: Path) -> int:
    index = min(range(len(config.times_s)), key=lambda item: abs(config.times_s[item] - time_s))
    tolerance = 64.0 * math.ulp(max(abs(time_s), abs(config.times_s[index]), 1.0e-300))
    if abs(time_s - config.times_s[index]) > tolerance:
        raise ValueError(f"{path}: time is not on the formal output schedule")
    return index


def _read_optional(row: dict[str, str], name: str) -> float:
    value = row[name].strip()
    return float(value) if value else math.nan


def _validate_trajectory_payload(
    status: str, values: np.ndarray, path: Path, line_number: int
) -> None:
    if status not in {"active", "held", "stuck", "escaped"}:
        raise ValueError(f"{path}:{line_number}: unsupported lifecycle {status!r}")
    if status == "active" and not np.isfinite(values).all():
        raise ValueError(f"{path}:{line_number}: active state is nonfinite")
    if status in {"held", "stuck"} and not np.isfinite(values[[0, 1, 4]]).all():
        raise ValueError(f"{path}:{line_number}: retained terminal state is incomplete")
    if status == "escaped" and np.isfinite(values).any():
        raise ValueError(f"{path}:{line_number}: escaped state must remain unobserved")


def _trajectory_row(
    row: dict[str, str], solver: Solver, config: EvaluationConfig, path: Path, line_number: int
) -> tuple[int, int, np.ndarray, str, str | None, float]:
    try:
        particle_id = int(row["particle_id"])
        time_s = float(row["time_s"])
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid particle/time") from error
    if not 1 <= particle_id <= config.particle_count:
        raise ValueError(f"{path}:{line_number}: particle ID outside range")
    values = np.asarray(
        [
            _read_optional(row, "r_m"),
            _read_optional(row, "z_m"),
            _read_optional(row, "velocity_r_m_per_s"),
            _read_optional(row, "velocity_z_m_per_s"),
            _read_optional(row, "charge_number_e"),
        ],
        dtype=np.float64,
    )
    status = row["lifecycle"] if solver == "candidate" else row["current_status"]
    _validate_trajectory_payload(status, values, path, line_number)
    final = row["final_status"] if solver == "reference" else None
    stop_time = _read_optional(row, "stop_time_s") if solver == "reference" else math.nan
    if solver == "reference":
        if final not in {"active", "held", "stuck", "escaped"}:
            raise ValueError(f"{path}:{line_number}: unsupported final status {final!r}")
        if (final == "active") != math.isnan(stop_time):
            raise ValueError(f"{path}:{line_number}: final status and stop time disagree")
    return particle_id, _time_index(time_s, config, path), values, status, final, stop_time


def _same_time(first: float, second: float) -> bool:
    tolerance = 64.0 * math.ulp(max(abs(first), abs(second), 1.0e-300))
    return abs(first - second) <= tolerance


def _terminal_events(
    events: dict[tuple[int, int], Event], config: EvaluationConfig, path: Path
) -> dict[int, Event]:
    by_particle: dict[int, Event] = {}
    for event in events.values():
        if not 1 <= event.particle_id <= config.particle_count:
            raise ValueError(f"{path}: event particle ID outside range")
        if event.particle_id in by_particle:
            raise ValueError(f"{path}: more than one terminal event for a particle")
        by_particle[event.particle_id] = event
    return by_particle


def _store_reference_metadata(
    final_fate: np.ndarray,
    stop_time_s: np.ndarray,
    metadata_seen: np.ndarray,
    particle: int,
    final: str | None,
    stop: float,
    path: Path,
    line_number: int,
) -> None:
    if not metadata_seen[particle]:
        final_fate[particle] = str(final)
        stop_time_s[particle] = stop
        metadata_seen[particle] = True
        return
    stop_matches = (math.isnan(stop_time_s[particle]) and math.isnan(stop)) or (
        math.isfinite(stop_time_s[particle])
        and math.isfinite(stop)
        and _same_time(stop_time_s[particle], stop)
    )
    if final_fate[particle] != final or not stop_matches:
        raise ValueError(f"{path}:{line_number}: final status metadata changes within history")


def _read_trajectory(
    path: Path,
    solver: Solver,
    config: EvaluationConfig,
    events: dict[tuple[int, int], Event],
) -> Trajectory:
    expected_columns = CANDIDATE_COLUMNS if solver == "candidate" else TRAJECTORY_COLUMNS
    state = np.full((len(config.times_s), config.particle_count, 5), np.nan, dtype=np.float64)
    lifecycle = np.full((len(config.times_s), config.particle_count), "", dtype="<U8")
    final_fate = np.full(config.particle_count, "", dtype="<U8")
    stop_time = np.full(config.particle_count, np.nan, dtype=np.float64)
    metadata_seen = np.zeros(config.particle_count, dtype=np.bool_)
    seen = np.zeros(lifecycle.shape, dtype=np.bool_)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise ValueError(f"{path}: trajectory columns differ")
        for line_number, row in enumerate(reader, start=2):
            particle_id, frame, values, status, final, stop = _trajectory_row(
                row, solver, config, path, line_number
            )
            particle = particle_id - 1
            if seen[frame, particle]:
                raise ValueError(f"{path}:{line_number}: duplicate particle/time key")
            state[frame, particle] = values
            lifecycle[frame, particle] = status
            if solver == "reference":
                _store_reference_metadata(
                    final_fate,
                    stop_time,
                    metadata_seen,
                    particle,
                    final,
                    stop,
                    path,
                    line_number,
                )
            seen[frame, particle] = True
    terminal_events = _terminal_events(events, config, path)
    if solver == "candidate":
        final_fate[:] = "active"
        for particle_id, event in terminal_events.items():
            final_fate[particle_id - 1] = event.fate
            stop_time[particle_id - 1] = event.time_s
    _validate_lifecycle(
        lifecycle, seen, final_fate, stop_time, config.times_s, terminal_events, path
    )
    return Trajectory(path.resolve(), _sha256(path), state, lifecycle, final_fate, stop_time, seen)


def _validate_active_history(
    column: np.ndarray,
    observed_count: int,
    frame_count: int,
    stop_time_s: float,
    event: Event | None,
    path: Path,
    particle_id: int,
) -> None:
    if event is not None or observed_count != frame_count:
        raise ValueError(
            f"{path}: non-terminal trajectory has a missing suffix for particle={particle_id}"
        )
    if np.any(column != "active") or math.isfinite(stop_time_s):
        raise ValueError(f"{path}: active final lifecycle is inconsistent")


def _validate_terminal_rows(
    statuses: np.ndarray,
    observed: np.ndarray,
    fate: str,
    stop_time_s: float,
    times_s: tuple[float, ...],
    path: Path,
) -> None:
    terminal_rows = np.flatnonzero(statuses != "active")
    if terminal_rows.size and np.any(statuses[terminal_rows[0] :] != fate):
        raise ValueError(f"{path}: lifecycle changes after terminal event")
    for frame, status in zip(observed, statuses, strict=True):
        time_s = times_s[int(frame)]
        if _same_time(time_s, stop_time_s):
            if status != fate:
                raise ValueError(f"{path}: lifecycle disagrees at terminal time")
        elif time_s < stop_time_s and status != "active":
            raise ValueError(f"{path}: terminal lifecycle precedes its event")
        elif time_s > stop_time_s and status != fate:
            raise ValueError(f"{path}: active lifecycle follows its terminal event")


def _validate_terminal_history(
    lifecycle: np.ndarray,
    observed: np.ndarray,
    fate: str,
    stop_time_s: float,
    times_s: tuple[float, ...],
    event: Event | None,
    path: Path,
    particle: int,
) -> None:
    if fate not in {"held", "stuck", "escaped"} or event is None:
        raise ValueError(f"{path}: terminal fate lacks one matching event")
    if (
        event.fate != fate
        or not math.isfinite(stop_time_s)
        or not 0.0 < stop_time_s <= times_s[-1]
        or not _same_time(stop_time_s, event.time_s)
    ):
        raise ValueError(f"{path}: terminal event and final fate metadata disagree")
    _validate_terminal_rows(
        lifecycle[observed, particle], observed, fate, stop_time_s, times_s, path
    )
    if observed.size < lifecycle.shape[0]:
        if fate != "escaped":
            raise ValueError(
                f"{path}: retained terminal trajectory has a missing suffix "
                f"for particle={particle + 1}"
            )
        if np.any(lifecycle[observed, particle] != "active"):
            raise ValueError(f"{path}: sparse escaped trajectory contains a terminal row")
        last_present_time = times_s[int(observed[-1])]
        first_missing_time = times_s[observed.size]
        if stop_time_s < last_present_time or _same_time(stop_time_s, last_present_time):
            raise ValueError(f"{path}: escaped trajectory is present at its terminal time")
        if first_missing_time < stop_time_s and not _same_time(first_missing_time, stop_time_s):
            raise ValueError(
                f"{path}: trajectory is missing before its terminal event for particle={particle + 1}"
            )


def _validate_lifecycle(
    lifecycle: np.ndarray,
    present: np.ndarray,
    final_fate: np.ndarray,
    stop_time_s: np.ndarray,
    times_s: tuple[float, ...],
    events: dict[int, Event],
    path: Path,
) -> None:
    if not present[0].all() or np.any(lifecycle[0] != "active"):
        raise ValueError(f"{path}: all particles must be present and active at t0")
    for particle in range(lifecycle.shape[1]):
        observed = np.flatnonzero(present[:, particle])
        expected_prefix = np.arange(observed[-1] + 1)
        if not np.array_equal(observed, expected_prefix):
            raise ValueError(f"{path}: interior trajectory hole for particle={particle + 1}")
        fate = str(final_fate[particle])
        event = events.get(particle + 1)
        if fate == "active":
            _validate_active_history(
                lifecycle[:, particle],
                observed.size,
                lifecycle.shape[0],
                float(stop_time_s[particle]),
                event,
                path,
                particle + 1,
            )
            continue
        _validate_terminal_history(
            lifecycle,
            observed,
            fate,
            float(stop_time_s[particle]),
            times_s,
            event,
            path,
            particle,
        )


def _event_group(law: str, outcome: str) -> tuple[str, str]:
    normalized_law = law.strip().lower()
    normalized_outcome = outcome.strip().lower()
    if normalized_law == "hold" or normalized_outcome in {"hold", "held"}:
        return "held", "gas_inlet"
    if normalized_law == "escape" or normalized_outcome in {"escape", "escaped"}:
        return "escaped", "pump_outlet"
    if normalized_law in {"stick", "probabilistic_stick"} or normalized_outcome in {
        "stick",
        "stuck",
    }:
        return "stuck", "material"
    raise ValueError(f"unsupported terminal boundary outcome: {law}/{outcome}")


def _event_identity(
    row: dict[str, str], solver: Solver, path: Path, line_number: int
) -> tuple[int, int]:
    particle_id = int(row["particle_id"])
    raw_ordinal = int(row["event_ordinal"])
    if solver == "candidate" and raw_ordinal < 1:
        raise ValueError(
            f"{path}:{line_number}: candidate boundary ordinal must follow release ordinal 0"
        )
    ordinal = raw_ordinal - 1 if solver == "candidate" else raw_ordinal
    if particle_id <= 0 or ordinal < 0:
        raise ValueError(f"{path}:{line_number}: invalid event identity")
    return particle_id, ordinal


def _event_payload(
    row: dict[str, str], solver: Solver
) -> tuple[str, str, bool, tuple[float, float] | None, float | None]:
    if solver == "candidate":
        fate, group = _event_group(row["law"], row["outcome"])
        direct = fate != "escaped"
        position = (float(row["hit_r_m"]), float(row["hit_z_m"])) if direct else None
        charge = float(row["charge_number_post_e"]) if direct else None
        return fate, group, direct, position, charge
    fate = row["fate"]
    group = row["boundary_group"]
    direct = row["observation_basis"] != "NOT_DIRECTLY_OBSERVED"
    position = (float(row["terminal_r_m"]), float(row["terminal_z_m"])) if direct else None
    charge = float(row["terminal_charge_number_e"]) if direct else None
    return fate, group, direct, position, charge


def _read_events(path: Path, solver: Solver) -> dict[tuple[int, int], Event]:
    expected = CANDIDATE_EVENT_COLUMNS if solver == "candidate" else EVENT_COLUMNS
    events: dict[tuple[int, int], Event] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path}: event columns differ")
        for line_number, row in enumerate(reader, start=2):
            particle_id, ordinal = _event_identity(row, solver, path, line_number)
            key = (particle_id, ordinal)
            if key in events:
                raise ValueError(f"{path}:{line_number}: duplicate event identity")
            fate, group, direct, position, charge = _event_payload(row, solver)
            event = Event(
                particle_id=particle_id,
                ordinal=ordinal,
                time_s=float(row["event_time_s"]),
                fate=fate,
                group=group,
                directly_observed=direct,
                position_m=position,
                charge_e=charge,
            )
            if not math.isfinite(event.time_s):
                raise ValueError(f"{path}:{line_number}: nonfinite event time")
            events[key] = event
    return events


def _quantity_metrics(
    difference: np.ndarray,
    scale: np.ndarray,
    roundoff_scale: float,
    roundoff_multiplier: float,
) -> dict[str, float | int | None]:
    if difference.size == 0:
        return {
            "count": 0,
            "rms": None,
            "maximum": None,
            "relative_l2": None,
            "scale_l2": 0.0,
            "roundoff_floor": 0.0,
            "roundoff_relative_l2": None,
        }
    square_sum = float(np.sum(difference * difference, dtype=np.float64))
    scale_square_sum = float(np.sum(scale * scale, dtype=np.float64))
    count = int(difference.size)
    floor = roundoff_multiplier * math.ulp(max(roundoff_scale, 1.0e-300))
    relative_roundoff = (
        floor * math.sqrt(count) / math.sqrt(scale_square_sum) if scale_square_sum > 0.0 else None
    )
    return {
        "count": count,
        "rms": math.sqrt(square_sum / count),
        "maximum": float(np.max(np.abs(difference))),
        "relative_l2": math.sqrt(square_sum / scale_square_sum) if scale_square_sum > 0.0 else None,
        "scale_l2": math.sqrt(scale_square_sum),
        "roundoff_floor": floor,
        "roundoff_relative_l2": relative_roundoff,
    }


def _state_difference_metrics(
    first: Trajectory,
    second: Trajectory,
    mask: np.ndarray,
    config: EvaluationConfig,
) -> dict[str, object]:
    delta = first.state - second.state
    initial = second.state[0, :, :2]
    displacement = second.state[:, :, :2] - initial[np.newaxis, :, :]
    position_difference = np.linalg.norm(delta[:, :, :2], axis=2)[mask]
    position_scale = np.linalg.norm(displacement, axis=2)[mask]
    velocity_difference = np.linalg.norm(delta[:, :, 2:4], axis=2)[mask]
    velocity_scale = np.linalg.norm(second.state[:, :, 2:4], axis=2)[mask]
    charge_difference = np.abs(delta[:, :, 4])[mask]
    charge_scale = np.abs(second.state[:, :, 4])[mask]
    active_values = second.state[mask]
    scales = {
        "position": float(np.max(np.abs(active_values[:, :2]), initial=0.0)),
        "velocity": float(np.max(np.abs(active_values[:, 2:4]), initial=0.0)),
        "charge": float(np.max(np.abs(active_values[:, 4]), initial=0.0)),
    }
    return {
        "position": _quantity_metrics(
            position_difference, position_scale, scales["position"], config.roundoff_multiplier
        ),
        "velocity": _quantity_metrics(
            velocity_difference, velocity_scale, scales["velocity"], config.roundoff_multiplier
        ),
        "charge": _quantity_metrics(
            charge_difference, charge_scale, scales["charge"], config.roundoff_multiplier
        ),
        "normalization": {
            "position": "second-trajectory displacement from each particle initial position",
            "velocity": "second-trajectory velocity L2 norm",
            "charge": "second-trajectory charge-number L2 norm",
        },
        "trajectory_rms_definition": {
            "aggregation": (
                "unweighted_over_shared_active_particle_records_at_configured_scheduled_frames"
            ),
            "configured_scheduled_frame_count": len(config.times_s),
            "time_interval_weighted": False,
            "time_integrated": False,
            "continuous_trajectory_metric": False,
        },
    }


def _rms_order(coarse: object, fine: object) -> float | None:
    if not isinstance(coarse, int | float) or not isinstance(fine, int | float):
        return None
    if coarse <= 0.0 or fine <= 0.0:
        return None
    return math.log2(float(coarse) / float(fine))


def _state_convergence(
    runs: dict[str, Trajectory], config: EvaluationConfig
) -> tuple[dict[str, object], bool]:
    active = np.ones_like(next(iter(runs.values())).lifecycle, dtype=np.bool_)
    for trajectory in runs.values():
        active &= trajectory.present & (trajectory.lifecycle == "active")
    coarse_medium = _state_difference_metrics(runs[RUN_KEYS[0]], runs[RUN_KEYS[1]], active, config)
    medium_fine = _state_difference_metrics(runs[RUN_KEYS[1]], runs[RUN_KEYS[2]], active, config)
    quantities: dict[str, object] = {}
    all_passed = True
    for quantity in ("position", "velocity", "charge"):
        coarse = _mapping(coarse_medium[quantity], f"{quantity} coarse pair")
        fine = _mapping(medium_fine[quantity], f"{quantity} fine pair")
        order = _rms_order(coarse.get("rms"), fine.get("rms"))
        floor = max(float(coarse["roundoff_floor"]), float(fine["roundoff_floor"]))
        roundoff_limited = all(
            isinstance(metrics[name], int | float) and float(metrics[name]) <= floor
            for metrics in (coarse, fine)
            for name in ("rms", "maximum")
        )
        passed = roundoff_limited or (order is not None and order >= config.minimum_order)
        all_passed &= passed
        quantities[quantity] = {
            "observed_rms_order": order,
            "minimum_rms_order": config.minimum_order,
            "classification": "ROUNDOFF_LIMITED" if roundoff_limited else "ORDER_EVALUATED",
            "status": "PASS" if passed else "FAIL",
        }
    return (
        {
            "shared_pre_terminal_active_records": int(np.count_nonzero(active)),
            "shared_pre_terminal_active_records_by_frame": [
                int(count) for count in np.count_nonzero(active, axis=1)
            ],
            "comparison_population": (
                "exact three-run intersection of observed active particle records at configured "
                "scheduled frames; RMS is unweighted by time interval"
            ),
            "trajectory_coverage": {
                run_key: {
                    "observed_records": int(np.count_nonzero(trajectory.present)),
                    "active_records": int(
                        np.count_nonzero(trajectory.present & (trajectory.lifecycle == "active"))
                    ),
                    "terminal_particles": int(np.count_nonzero(trajectory.final_fate != "active")),
                }
                for run_key, trajectory in runs.items()
            },
            "coarse_medium": coarse_medium,
            "medium_fine": medium_fine,
            "quantities": quantities,
        },
        all_passed,
    )


def _scalar_metrics(
    values: list[float], scale: float, config: EvaluationConfig
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "rms": None, "maximum": None, "roundoff_floor": 0.0}
    floor = config.roundoff_multiplier * math.ulp(max(abs(scale), 1.0e-300))
    return {
        "count": int(array.size),
        "rms": float(np.sqrt(np.mean(array * array))),
        "maximum": float(np.max(np.abs(array))),
        "roundoff_floor": floor,
    }


def _event_pair_metrics(
    first: dict[tuple[int, int], Event],
    second: dict[tuple[int, int], Event],
    config: EvaluationConfig,
) -> tuple[dict[str, object], bool]:
    identities_match = set(first) == set(second)
    semantics_match = identities_match and all(
        (first[key].fate, first[key].group, first[key].directly_observed)
        == (second[key].fate, second[key].group, second[key].directly_observed)
        for key in first
    )
    if not semantics_match:
        return {
            "identity_match": identities_match,
            "semantic_match": False,
            "time": _scalar_metrics([], 0.0, config),
            "terminal_position": _scalar_metrics([], 0.0, config),
            "terminal_charge": _scalar_metrics([], 0.0, config),
        }, False
    time_errors = [abs(first[key].time_s - second[key].time_s) for key in first]
    position_errors: list[float] = []
    charge_errors: list[float] = []
    position_scale = 0.0
    charge_scale = 0.0
    for key in first:
        left = first[key]
        right = second[key]
        if not left.directly_observed:
            continue
        if left.position_m is None or right.position_m is None:
            raise ValueError("direct terminal event lacks position")
        position_errors.append(math.dist(left.position_m, right.position_m))
        position_scale = max(
            position_scale, *(abs(value) for value in (*left.position_m, *right.position_m))
        )
        if left.charge_e is None or right.charge_e is None:
            raise ValueError("direct terminal event lacks charge")
        charge_errors.append(abs(left.charge_e - right.charge_e))
        charge_scale = max(charge_scale, abs(left.charge_e), abs(right.charge_e))
    return {
        "identity_match": True,
        "semantic_match": True,
        "time": _scalar_metrics(time_errors, config.times_s[-1], config),
        "terminal_position": _scalar_metrics(position_errors, position_scale, config),
        "terminal_charge": _scalar_metrics(charge_errors, charge_scale, config),
    }, True


def _event_convergence(
    runs: dict[str, dict[tuple[int, int], Event]], config: EvaluationConfig
) -> tuple[dict[str, object], bool]:
    coarse, coarse_ok = _event_pair_metrics(runs[RUN_KEYS[0]], runs[RUN_KEYS[1]], config)
    fine, fine_ok = _event_pair_metrics(runs[RUN_KEYS[1]], runs[RUN_KEYS[2]], config)
    orders: dict[str, float | None] = {}
    passed = coarse_ok and fine_ok
    for quantity in ("time", "terminal_position", "terminal_charge"):
        first = _mapping(coarse[quantity], f"event {quantity} coarse")
        second = _mapping(fine[quantity], f"event {quantity} fine")
        order = _rms_order(first.get("rms"), second.get("rms"))
        orders[quantity] = order
        if int(first["count"]) > 0:
            floor = max(float(first["roundoff_floor"]), float(second["roundoff_floor"]))
            roundoff = all(
                isinstance(metrics[name], int | float) and float(metrics[name]) <= floor
                for metrics in (first, second)
                for name in ("rms", "maximum")
            )
            passed &= roundoff or (order is not None and order >= config.minimum_order)
    event_counts = {run_key: len(runs[run_key]) for run_key in RUN_KEYS}
    no_events = all(count == 0 for count in event_counts.values())
    status = "NOT_APPLICABLE_NO_EVENTS" if no_events else ("PASS" if passed else "FAIL")
    return {
        "coarse_medium": coarse,
        "medium_fine": fine,
        "observed_rms_order": orders,
        "event_counts": event_counts,
        "status": status,
        "boundary_accuracy_evidence": False if no_events else passed,
    }, passed


def _reference_receipts(root: Path, config: EvaluationConfig) -> dict[str, object]:
    workflows: dict[str, object] = {}
    for workflow in WORKFLOWS:
        path = root / workflow / "reference_normalization_report.json"
        report = _mapping(json.loads(path.read_text(encoding="utf-8")), "normalization report")
        config_record = _mapping(report.get("config"), "normalization config")
        if report.get("status") != "COMPLETE" or config_record.get("sha256") != config.sha256:
            raise ValueError(f"{path}: incomplete or uses another configuration")
        if report.get("tool_revision") != "m3c1_theory_100nm_30ms_reference_normalizer_v3":
            raise ValueError(f"{path}: unsupported normalizer revision")
        if report.get("workflow") != workflow:
            raise ValueError(f"{path}: normalized workflow identity differs")
        run_records = _mapping(report.get("runs"), f"{workflow} normalized runs")
        if set(run_records) != set(RUN_KEYS):
            raise ValueError(f"{path}: normalized run keys differ")
        expected_steps = config.steps_s_by_workflow[workflow]
        actual_steps: list[float] = []
        for run_key, step_s in zip(RUN_KEYS, expected_steps, strict=True):
            run = _mapping(run_records[run_key], f"{workflow} normalized {run_key}")
            actual_step = _number(run.get("step_s"), f"{workflow} normalized {run_key}.step_s")
            if actual_step != step_s:
                raise ValueError(f"{path}: normalized {run_key} step differs")
            actual_steps.append(actual_step)
            trajectory_path = root / workflow / f"reference_trajectory_{run_key}.csv"
            event_path = root / workflow / f"reference_events_{run_key}.csv"
            if (
                run.get("trajectory") != trajectory_path.name
                or run.get("event_ledger") != event_path.name
                or _sha256(trajectory_path) != run.get("trajectory_sha256")
                or _sha256(event_path) != run.get("event_ledger_sha256")
            ):
                raise ValueError(f"{path}: normalized {run_key} artifact receipt differs")
        common = _mapping(report.get("common_p1_receipt"), "common-P1 receipt")
        if common.get("status") != "PASS":
            raise ValueError(f"{path}: common-P1 receipt did not pass")
        workflows[workflow] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "canonical_content_hash": common.get("canonical_content_hash"),
            "candidate_input_sha256": common.get("candidate_input_sha256"),
            "step_sequence": _step_sequence(config, workflow, tuple(actual_steps)),
            "runs": {
                run_key: {
                    "step_s": step_s,
                    "trajectory_sha256": _mapping(run_records[run_key], run_key).get(
                        "trajectory_sha256"
                    ),
                    "event_sha256": _mapping(run_records[run_key], run_key).get(
                        "event_ledger_sha256"
                    ),
                }
                for run_key, step_s in zip(RUN_KEYS, expected_steps, strict=True)
            },
        }
    return {"normalization_reports": workflows}


def _candidate_receipts(root: Path, config: EvaluationConfig) -> dict[str, object]:
    path = root / "candidate_run_report.json"
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "candidate run report")
    if report.get("status") != "COMPLETE" or report.get("configuration_sha256") != config.sha256:
        raise ValueError("candidate run report is incomplete or uses another configuration")
    workflows_raw = _mapping(report.get("workflows"), "candidate workflows")
    workflows: dict[str, object] = {}
    for workflow in WORKFLOWS:
        record = _mapping(workflows_raw.get(workflow), f"candidate {workflow}")
        if record.get("status") != "COMPLETE":
            raise ValueError(f"candidate {workflow} is incomplete")
        runs_raw = _mapping(record.get("runs"), f"candidate {workflow} runs")
        if set(runs_raw) != set(RUN_KEYS):
            raise ValueError(f"candidate {workflow} run keys differ")
        runs: dict[str, object] = {}
        actual_steps: list[float] = []
        for run_key, step_s in zip(RUN_KEYS, config.steps_s_by_workflow[workflow], strict=True):
            run = _mapping(runs_raw[run_key], f"candidate {workflow} {run_key}")
            actual_step = _number(run.get("dt_s"), f"candidate {workflow} {run_key}.dt_s")
            trajectory_path = root / workflow / f"candidate_trajectory_{run_key}.csv"
            event_path = root / workflow / f"candidate_events_{run_key}.csv"
            if (
                run.get("status") != "COMPLETE"
                or run.get("workflow") != workflow
                or run.get("step_label") != run_key
                or actual_step != step_s
                or run.get("trajectory") != trajectory_path.name
                or run.get("events") != event_path.name
                or _sha256(trajectory_path) != run.get("trajectory_sha256")
                or _sha256(event_path) != run.get("events_sha256")
            ):
                raise ValueError(
                    f"candidate {workflow} {run_key} receipt differs from configured run"
                )
            runs[run_key] = {
                "step_s": actual_step,
                "trajectory_sha256": run.get("trajectory_sha256"),
                "event_sha256": run.get("events_sha256"),
            }
            actual_steps.append(actual_step)
        workflows[workflow] = {
            "canonical_content_hash": record.get("input_content_hash"),
            "candidate_input_sha256": record.get("input_sha256"),
            "step_sequence": _step_sequence(config, workflow, tuple(actual_steps)),
            "runs": runs,
        }
    return {"path": str(path.resolve()), "sha256": _sha256(path), "workflows": workflows}


def _source_receipts(root: Path, solver: Solver, config: EvaluationConfig) -> dict[str, object]:
    return (
        _candidate_receipts(root, config)
        if solver == "candidate"
        else _reference_receipts(root, config)
    )


def characterize(
    config_path: Path, solver: Solver, root_path: Path, output_path: Path
) -> dict[str, Any]:
    config = _load_config(config_path)
    root = root_path.expanduser().resolve()
    source_receipts = _source_receipts(root, solver, config)
    workflow_results: dict[str, object] = {}
    overall = True
    source_kind = "workflows" if solver == "candidate" else "normalization_reports"
    source_workflows = _mapping(source_receipts.get(source_kind), "source workflow receipts")
    for workflow in WORKFLOWS:
        source_workflow = _mapping(source_workflows.get(workflow), f"{workflow} source receipt")
        events = {
            run_key: _read_events(_event_path(root, workflow, solver, label), solver)
            for run_key, label in zip(RUN_KEYS, STEP_LABELS, strict=True)
        }
        trajectories = {
            run_key: _read_trajectory(
                _trajectory_path(root, workflow, solver, label),
                solver,
                config,
                events[run_key],
            )
            for run_key, label in zip(RUN_KEYS, STEP_LABELS, strict=True)
        }
        state_convergence, state_passed = _state_convergence(trajectories, config)
        event_convergence, event_passed = _event_convergence(events, config)
        overall &= state_passed and event_passed
        workflow_results[workflow] = {
            "step_sequence": source_workflow.get("step_sequence"),
            "trajectory_artifacts": {
                run_key: {"path": str(item.path), "sha256": item.sha256}
                for run_key, item in trajectories.items()
            },
            "event_artifacts": {
                run_key: {
                    "path": str(_event_path(root, workflow, solver, label).resolve()),
                    "sha256": _sha256(_event_path(root, workflow, solver, label)),
                }
                for run_key, label in zip(RUN_KEYS, STEP_LABELS, strict=True)
            },
            "state_self_convergence": state_convergence,
            "event_self_convergence": event_convergence,
            "status": "PASS" if state_passed and event_passed else "FAIL",
        }
    report = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "phase": "characterize",
        "solver": solver,
        "status": "PASS" if overall else "FAIL",
        "config": {"path": str(config.path), "sha256": config.sha256},
        "source_receipts": source_receipts,
        "workflows": workflow_results,
    }
    _write_json_exclusive(output_path.expanduser().resolve(), report)
    return report


def _load_report(path: Path, phase: str) -> tuple[dict[str, object], str]:
    resolved = path.expanduser().resolve()
    report = _mapping(json.loads(resolved.read_text(encoding="utf-8")), f"{phase} report")
    if report.get("tool_revision") != TOOL_REVISION or report.get("phase") != phase:
        raise ValueError(f"{resolved}: wrong tool revision or phase")
    return report, _sha256(resolved)


def _t0_states(path: Path, solver: Solver, config: EvaluationConfig) -> np.ndarray:
    columns = CANDIDATE_COLUMNS if solver == "candidate" else TRAJECTORY_COLUMNS
    values = np.full((config.particle_count, 5), np.nan, dtype=np.float64)
    seen = np.zeros(config.particle_count, dtype=np.bool_)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"{path}: trajectory columns differ")
        for row in reader:
            time_s = float(row["time_s"])
            if time_s != 0.0:
                continue
            particle = int(row["particle_id"]) - 1
            if not 0 <= particle < config.particle_count or seen[particle]:
                raise ValueError(f"{path}: invalid t0 particle identity")
            values[particle] = [
                float(row["r_m"]),
                float(row["z_m"]),
                float(row["velocity_r_m_per_s"]),
                float(row["velocity_z_m_per_s"]),
                float(row["charge_number_e"]),
            ]
            seen[particle] = True
    if not seen.all() or not np.isfinite(values).all():
        raise ValueError(f"{path}: t0 state matrix is incomplete")
    return values


def _initial_state_gate(
    candidate: np.ndarray, reference: np.ndarray, config: EvaluationConfig
) -> dict[str, object]:
    names = ("r_m", "z_m", "vr_m_s", "vz_m_s", "charge_e")
    components: dict[str, object] = {}
    passed = True
    for index, name in enumerate(names):
        difference = np.abs(candidate[:, index] - reference[:, index])
        scale = float(
            max(
                np.max(np.abs(candidate[:, index]), initial=0.0),
                np.max(np.abs(reference[:, index]), initial=0.0),
                1.0e-300,
            )
        )
        limit = config.initial_ulp_multiplier * math.ulp(scale)
        maximum = float(np.max(difference, initial=0.0))
        component_passed = maximum <= limit
        passed &= component_passed
        components[name] = {
            "maximum_absolute_difference": maximum,
            "limit": limit,
            "status": "PASS" if component_passed else "FAIL",
        }
    return {"status": "PASS" if passed else "FAIL", "components": components}


def _envelope(
    candidate: dict[str, Any], reference: dict[str, Any], factor: float
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("rms", "maximum", "relative_l2"):
        left = candidate.get(metric)
        right = reference.get(metric)
        if left is None or right is None:
            result[metric] = None
            continue
        if metric == "relative_l2":
            left_floor = candidate.get("roundoff_relative_l2") or 0.0
            right_floor = reference.get("roundoff_relative_l2") or 0.0
        else:
            left_floor = candidate.get("roundoff_floor") or 0.0
            right_floor = reference.get("roundoff_floor") or 0.0
        result[metric] = factor * (
            float(left) + float(right) + float(left_floor) + float(right_floor)
        )
    return result


def _event_envelope(
    candidate: dict[str, Any], reference: dict[str, Any], factor: float
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("rms", "maximum"):
        left = candidate.get(metric)
        right = reference.get(metric)
        if left is None or right is None:
            result[metric] = None
        else:
            result[metric] = factor * (
                float(left)
                + float(right)
                + float(candidate.get("roundoff_floor") or 0.0)
                + float(reference.get("roundoff_floor") or 0.0)
            )
    return result


def _characterization_workflow(report: dict[str, object], workflow: str) -> dict[str, object]:
    return _mapping(_mapping(report.get("workflows"), "workflows").get(workflow), workflow)


def _common_p1_identity(
    candidate: dict[str, object], reference: dict[str, object], workflow: str
) -> dict[str, object]:
    candidate_sources = _mapping(candidate.get("source_receipts"), "candidate source receipts")
    candidate_workflows = _mapping(candidate_sources.get("workflows"), "candidate receipts")
    candidate_record = _mapping(candidate_workflows.get(workflow), f"candidate {workflow} receipt")
    reference_sources = _mapping(reference.get("source_receipts"), "reference source receipts")
    reference_workflows = _mapping(
        reference_sources.get("normalization_reports"), "reference receipts"
    )
    reference_record = _mapping(reference_workflows.get(workflow), f"reference {workflow} receipt")
    checks = {
        "canonical_content_hash": candidate_record.get("canonical_content_hash")
        == reference_record.get("canonical_content_hash"),
        "candidate_input_sha256": candidate_record.get("candidate_input_sha256")
        == reference_record.get("candidate_input_sha256"),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def register(
    config_path: Path,
    candidate_characterization: Path,
    reference_characterization: Path,
    output_path: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    candidate, candidate_hash = _load_report(candidate_characterization, "characterize")
    reference, reference_hash = _load_report(reference_characterization, "characterize")
    if candidate.get("solver") != "candidate" or reference.get("solver") != "reference":
        raise ValueError("characterization solver labels differ")
    for report in (candidate, reference):
        if _mapping(report.get("config"), "characterization config").get("sha256") != config.sha256:
            raise ValueError("characterization uses another configuration")
    workflows: dict[str, object] = {}
    ready = candidate.get("status") == "PASS" and reference.get("status") == "PASS"
    for workflow in WORKFLOWS:
        candidate_case = _characterization_workflow(candidate, workflow)
        reference_case = _characterization_workflow(reference, workflow)
        common_p1 = _common_p1_identity(candidate, reference, workflow)
        ready &= common_p1["status"] == "PASS"
        candidate_artifacts = _mapping(
            candidate_case.get("trajectory_artifacts"), "candidate artifacts"
        )
        reference_artifacts = _mapping(
            reference_case.get("trajectory_artifacts"), "reference artifacts"
        )
        candidate_fine = _mapping(candidate_artifacts.get(RUN_KEYS[2]), "candidate fine artifact")
        reference_fine = _mapping(reference_artifacts.get(RUN_KEYS[2]), "reference fine artifact")
        candidate_t0 = _t0_states(Path(str(candidate_fine["path"])), "candidate", config)
        reference_t0 = _t0_states(Path(str(reference_fine["path"])), "reference", config)
        initial = _initial_state_gate(candidate_t0, reference_t0, config)
        ready &= initial["status"] == "PASS"
        candidate_state = _mapping(candidate_case.get("state_self_convergence"), "candidate state")
        reference_state = _mapping(reference_case.get("state_self_convergence"), "reference state")
        candidate_pair = _mapping(candidate_state.get("medium_fine"), "candidate fine pair")
        reference_pair = _mapping(reference_state.get("medium_fine"), "reference fine pair")
        state_envelopes = {
            quantity: _envelope(
                _mapping(candidate_pair.get(quantity), f"candidate {quantity}"),
                _mapping(reference_pair.get(quantity), f"reference {quantity}"),
                config.safety_factor,
            )
            for quantity in ("position", "velocity", "charge")
        }
        candidate_event = _mapping(candidate_case.get("event_self_convergence"), "candidate event")
        reference_event = _mapping(reference_case.get("event_self_convergence"), "reference event")
        candidate_event_pair = _mapping(candidate_event.get("medium_fine"), "candidate event pair")
        reference_event_pair = _mapping(reference_event.get("medium_fine"), "reference event pair")
        event_envelopes = {
            quantity: _event_envelope(
                _mapping(candidate_event_pair.get(quantity), f"candidate event {quantity}"),
                _mapping(reference_event_pair.get(quantity), f"reference event {quantity}"),
                config.safety_factor,
            )
            for quantity in ("time", "terminal_position", "terminal_charge")
        }
        workflows[workflow] = {
            "common_p1_identity": common_p1,
            "step_sequence": candidate_case.get("step_sequence"),
            "reference_step_sequence": reference_case.get("step_sequence"),
            "initial_state": initial,
            "state_envelopes": state_envelopes,
            "event_envelopes": event_envelopes,
            "fine_artifacts": {
                "candidate": candidate_fine,
                "reference": reference_fine,
                "candidate_event": _mapping(
                    _mapping(
                        candidate_case.get("event_artifacts"), "candidate event artifacts"
                    ).get(RUN_KEYS[2]),
                    "candidate fine event",
                ),
                "reference_event": _mapping(
                    _mapping(
                        reference_case.get("event_artifacts"), "reference event artifacts"
                    ).get(RUN_KEYS[2]),
                    "reference fine event",
                ),
            },
        }
    report = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "phase": "register",
        "status": "REGISTERED" if ready else "BLOCKED",
        "config": {"path": str(config.path), "sha256": config.sha256},
        "characterizations": {
            "candidate": {
                "path": str(candidate_characterization.resolve()),
                "sha256": candidate_hash,
            },
            "reference": {
                "path": str(reference_characterization.resolve()),
                "sha256": reference_hash,
            },
        },
        "registration_policy": {
            "post_t0_cross_state_values_read": False,
            "envelope": "safety_factor * (candidate fine-pair + reference fine-pair + both roundoff floors)",
            "safety_factor": config.safety_factor,
        },
        "workflows": workflows,
    }
    _write_json_exclusive(output_path.expanduser().resolve(), report)
    return report


def _artifact_from_budget(budget_case: dict[str, object], name: str) -> tuple[Path, str]:
    artifacts = _mapping(budget_case.get("fine_artifacts"), "fine artifacts")
    record = _mapping(artifacts.get(name), f"fine artifact {name}")
    path = Path(str(record["path"])).resolve()
    expected = str(record["sha256"])
    if _sha256(path) != expected:
        raise ValueError(f"registered fine artifact changed: {path}")
    return path, expected


def _metric_gate(
    gates: list[dict[str, object]],
    workflow: str,
    category: str,
    metric: str,
    observed: object,
    limit: object,
) -> bool:
    passed = (
        isinstance(observed, int | float) and isinstance(limit, int | float) and observed <= limit
    )
    gates.append(
        {
            "workflow": workflow,
            "category": category,
            "metric": metric,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "limit": limit,
            "notes": "registered before post-t0 cross comparison",
        }
    )
    return passed


def _event_cross_metrics(
    candidate: dict[tuple[int, int], Event],
    reference: dict[tuple[int, int], Event],
    config: EvaluationConfig,
) -> tuple[dict[str, object], dict[str, bool]]:
    metrics, semantic = _event_pair_metrics(candidate, reference, config)
    identity = set(candidate) == set(reference)
    fate = identity and all(candidate[key].fate == reference[key].fate for key in candidate)
    order = identity and all(candidate[key].ordinal == reference[key].ordinal for key in candidate)
    group = identity and all(candidate[key].group == reference[key].group for key in candidate)
    escape_count = sum(event.fate == "escaped" for event in reference.values())
    metrics["escape_position"] = {
        "status": "NOT_DIRECTLY_OBSERVED",
        "count": escape_count,
        "gated": False,
    }
    metrics["terminal_velocity"] = {"status": "CHARACTERIZED_NOT_CROSS_GATED"}
    no_events = not candidate and not reference
    metrics["evidence_classification"] = {
        "status": "NOT_APPLICABLE_NO_EVENTS" if no_events else "EVENTS_PRESENT",
        "candidate_event_count": len(candidate),
        "reference_event_count": len(reference),
        "boundary_accuracy_evidence": not no_events,
    }
    return metrics, {
        "identity": identity,
        "fate": fate,
        "order": order,
        "group": group,
        "semantic": semantic,
    }


def _population_rows(
    candidate: Trajectory, reference: Trajectory, config: EvaluationConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for time_s in config.times_s:
        row: dict[str, object] = {"time_s": time_s}
        active_counts: dict[str, int] = {}
        for solver, trajectory in (("candidate", candidate), ("reference", reference)):
            population = np.full(config.particle_count, "active", dtype="<U8")
            for particle, stop_time_s in enumerate(trajectory.stop_time_s):
                if math.isfinite(stop_time_s) and (
                    time_s > stop_time_s or _same_time(time_s, stop_time_s)
                ):
                    population[particle] = trajectory.final_fate[particle]
            for fate in ("active", "held", "stuck", "escaped"):
                row[f"{solver}_{fate}"] = int(np.count_nonzero(population == fate))
            active_counts[solver] = int(np.count_nonzero(population == "active"))
        row["active_count_difference"] = active_counts["candidate"] - active_counts["reference"]
        rows.append(row)
    return rows


def _write_population(path: Path, rows: list[dict[str, object]]) -> None:
    columns = (
        "time_s",
        "candidate_active",
        "reference_active",
        "candidate_held",
        "reference_held",
        "candidate_stuck",
        "reference_stuck",
        "candidate_escaped",
        "reference_escaped",
        "active_count_difference",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fate_gate(candidate: Trajectory, reference: Trajectory) -> tuple[bool, dict[str, object]]:
    exact = bool(np.array_equal(candidate.final_fate, reference.final_fate))
    counts = {
        solver: {
            fate: int(np.count_nonzero(trajectory.final_fate == fate))
            for fate in ("active", "held", "stuck", "escaped")
        }
        for solver, trajectory in (("candidate", candidate), ("reference", reference))
    }
    return exact, {"particle_identity_exact": exact, "counts": counts}


def _write_gates(path: Path, gates: list[dict[str, object]]) -> None:
    columns = ("workflow", "category", "metric", "status", "observed", "limit", "notes")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(gates)


def _readme(status: str, case_status: dict[str, str], config: EvaluationConfig) -> str:
    lines = [
        "# M3-C1 theory-consistent 100 nm, 30 ms comparison",
        "",
        f"Formal same-field particle-solver decision: **{status}**.",
        "",
        "| workflow | decision |",
        "|---|---|",
        *(f"| {workflow} | {case_status[workflow]} |" for workflow in WORKFLOWS),
        "",
        "The comparison uses the canonical exact-connectivity P1 fields, dynamic charge,",
        "all seven deterministic force contributions, and Brownian/Saffman disabled.",
        *(
            f"{workflow} fixed RK4 coarse/medium/fine steps (s): "
            + ", ".join(f"{step:.12g}" for step in config.steps_s_by_workflow[workflow])
            for workflow in WORKFLOWS
        ),
        f"The h/h2/h4 RMS-order minimum is {config.minimum_order:g}; cross envelopes use",
        f"the preregistered safety factor {config.safety_factor:g}.",
        (
            "Trajectory RMS values are unweighted across shared-active particle samples at "
            f"the {len(config.times_s)} configured scheduled frames; they are neither "
            "time-integrated nor continuous-trajectory RMS values."
        ),
        "",
        "Only shared pre-terminal active samples enter trajectory metrics. Held/stuck",
        "terminal position and charge are compared when directly observed. COMSOL",
        "Disappear coordinates are NOT_DIRECTLY_OBSERVED and are never reconstructed or",
        "gated; no solver's post-terminal coordinates are invented. Population survival",
        "comes from the sparse terminal-event ledgers; terminal velocity is not a gate.",
        "A zero-event workflow is NOT_APPLICABLE_NO_EVENTS for event convergence and is",
        "not boundary-accuracy evidence; it does not by itself block state/fate comparison.",
        "Shared configuration/common-P1 input identity is recorded separately from",
        "pointwise force/RHS parity, which is NOT_SEPARATELY_ESTABLISHED here.",
        "",
        "Case-A field production remains CHARACTERIZED (F02), Case-P product field",
        "adapter coverage is NOT_TESTED, physical applicability is not certified",
        "(especially Case-P negative-ion aggregate assumptions), Brownian is NOT_TESTED,",
        "and universal COMSOL equivalence is NOT_CLAIMED.",
        "",
    ]
    return "\n".join(lines)


def _boolean_gate(
    gates: list[dict[str, object]],
    workflow: str,
    category: str,
    metric: str,
    passed: bool,
    *,
    notes: str = "particle identity, fate, ordinal, and semantic boundary group",
    not_applicable_status: str | None = None,
) -> None:
    gates.append(
        {
            "workflow": workflow,
            "category": category,
            "metric": metric,
            "status": not_applicable_status or ("PASS" if passed else "FAIL"),
            "observed": "NO_EVENTS" if not_applicable_status else passed,
            "limit": "NOT_APPLICABLE" if not_applicable_status else True,
            "notes": notes,
        }
    )


def _compare_workflow(
    workflow: str,
    budget_case: dict[str, object],
    config: EvaluationConfig,
    output: Path,
    gates: list[dict[str, object]],
) -> dict[str, object]:
    candidate_path, _ = _artifact_from_budget(budget_case, "candidate")
    reference_path, _ = _artifact_from_budget(budget_case, "reference")
    candidate_event_path, _ = _artifact_from_budget(budget_case, "candidate_event")
    reference_event_path, _ = _artifact_from_budget(budget_case, "reference_event")
    candidate_events = _read_events(candidate_event_path, "candidate")
    reference_events = _read_events(reference_event_path, "reference")
    candidate = _read_trajectory(candidate_path, "candidate", config, candidate_events)
    reference = _read_trajectory(reference_path, "reference", config, reference_events)
    mask = (
        candidate.present
        & reference.present
        & (candidate.lifecycle == "active")
        & (reference.lifecycle == "active")
    )
    state_metrics = _state_difference_metrics(candidate, reference, mask, config)
    state_envelopes = _mapping(budget_case.get("state_envelopes"), "state envelopes")
    passed = True
    for quantity in ("position", "velocity", "charge"):
        metrics = _mapping(state_metrics.get(quantity), f"cross {quantity}")
        limits = _mapping(state_envelopes.get(quantity), f"{quantity} envelope")
        for metric in ("rms", "maximum", "relative_l2"):
            passed &= _metric_gate(
                gates,
                workflow,
                "trajectory",
                f"{quantity}_{metric}",
                metrics.get(metric),
                limits.get(metric),
            )
    event_metrics, event_exact = _event_cross_metrics(candidate_events, reference_events, config)
    no_events = not candidate_events and not reference_events
    for name, exact in event_exact.items():
        _boolean_gate(
            gates,
            workflow,
            "event",
            name,
            exact,
            notes=(
                "vacuous zero-event identity check; not boundary-accuracy evidence"
                if no_events
                else "particle identity, fate, ordinal, and semantic boundary group"
            ),
            not_applicable_status="NOT_APPLICABLE_NO_EVENTS" if no_events else None,
        )
        if not no_events:
            passed &= exact
    event_envelopes = _mapping(budget_case.get("event_envelopes"), "event envelopes")
    for quantity in ("time", "terminal_position", "terminal_charge"):
        metrics = _mapping(event_metrics.get(quantity), f"cross event {quantity}")
        limits = _mapping(event_envelopes.get(quantity), f"event {quantity} envelope")
        if int(metrics["count"]) == 0:
            continue
        for metric in ("rms", "maximum"):
            passed &= _metric_gate(
                gates,
                workflow,
                "event",
                f"{quantity}_{metric}",
                metrics.get(metric),
                limits.get(metric),
            )
    fate_passed, fate = _fate_gate(candidate, reference)
    _boolean_gate(gates, workflow, "fate", "particle_final_fate_identity", fate_passed)
    passed &= fate_passed
    population = _population_rows(candidate, reference, config)
    population_name = f"{workflow}_population_survival.csv"
    _write_population(output / population_name, population)
    return {
        "status": "PASS" if passed else "FAIL",
        "step_sequence": budget_case.get("step_sequence"),
        "reference_step_sequence": budget_case.get("reference_step_sequence"),
        "shared_pre_terminal_active_records": int(np.count_nonzero(mask)),
        "shared_pre_terminal_active_records_by_frame": [
            int(count) for count in np.count_nonzero(mask, axis=1)
        ],
        "comparison_population": (
            "exact candidate/reference intersection of observed active particle records at configured "
            "scheduled frames; RMS is unweighted by time interval"
        ),
        "trajectory_coverage": {
            "candidate_observed_records": int(np.count_nonzero(candidate.present)),
            "reference_observed_records": int(np.count_nonzero(reference.present)),
            "scheduled_particle_records": int(candidate.present.size),
        },
        "trajectory_metrics": state_metrics,
        "event_metrics": event_metrics,
        "final_fates": fate,
        "population_survival_csv": population_name,
    }


def compare(config_path: Path, budget_path: Path, output_directory: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    budget, budget_hash = _load_report(budget_path, "register")
    if budget.get("status") != "REGISTERED":
        raise ValueError("comparison budget is not REGISTERED")
    if _mapping(budget.get("config"), "budget config").get("sha256") != config.sha256:
        raise ValueError("comparison budget uses another configuration")
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    gates: list[dict[str, object]] = []
    results: dict[str, object] = {}
    case_status: dict[str, str] = {}
    budget_workflows = _mapping(budget.get("workflows"), "budget workflows")
    for workflow in WORKFLOWS:
        budget_case = _mapping(budget_workflows.get(workflow), f"budget {workflow}")
        case_result = _compare_workflow(workflow, budget_case, config, output, gates)
        case_status[workflow] = str(case_result["status"])
        result_name = f"{workflow}_metrics.json"
        _write_json_exclusive(output / result_name, case_result)
        results[workflow] = {**case_result, "metrics_file": result_name}
    overall = "PASS" if all(status == "PASS" for status in case_status.values()) else "FAIL"
    _write_gates(output / "gates.csv", gates)
    claims = {
        "same_field_particle_solver": overall,
        "configured_formula_identity": {
            "status": "SHARED_DECLARATION_CONFIRMED",
            "basis": "shared_configuration_and_common_p1_input_identity",
            "pointwise_rhs_parity_implied": False,
        },
        "pointwise_rhs_formula_parity": "NOT_SEPARATELY_ESTABLISHED",
        "caseA_field_production": "CHARACTERIZED_F02_NOT_CERTIFIED_HERE",
        "caseP_product_field_adapter": "NOT_TESTED",
        "physical_applicability": {
            "caseA": "NOT_CERTIFIED_MODEL_FORM_SENSITIVITY",
            "caseP": "NOT_CERTIFIED_NEGATIVE_ION_AGGREGATE_ASSUMPTIONS",
        },
        "brownian": "NOT_TESTED_BROWNIAN_DISABLED",
        "universal_comsol_equivalence": "NOT_CLAIMED",
    }
    manifest = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "phase": "compare",
        "status": overall,
        "config": {"path": str(config.path), "sha256": config.sha256},
        "registered_budget": {"path": str(budget_path.resolve()), "sha256": budget_hash},
        "workflows": results,
        "gate_count": len(gates),
        "failed_gate_count": sum(row["status"] == "FAIL" for row in gates),
        "claim_status": claims,
    }
    _write_json_exclusive(output / "comparison_manifest.json", manifest)
    (output / "README.md").write_text(
        _readme(overall, case_status, config), encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    characterize_parser = subparsers.add_parser("characterize")
    characterize_parser.add_argument("--config", required=True, type=Path)
    characterize_parser.add_argument("--solver", required=True, choices=("candidate", "reference"))
    characterize_parser.add_argument("--root", required=True, type=Path)
    characterize_parser.add_argument("--output", required=True, type=Path)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--config", required=True, type=Path)
    register_parser.add_argument("--candidate-characterization", required=True, type=Path)
    register_parser.add_argument("--reference-characterization", required=True, type=Path)
    register_parser.add_argument("--output", required=True, type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--config", required=True, type=Path)
    compare_parser.add_argument("--budget", required=True, type=Path)
    compare_parser.add_argument("--output-directory", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.command == "characterize":
        characterize(arguments.config, arguments.solver, arguments.root, arguments.output)
    elif arguments.command == "register":
        register(
            arguments.config,
            arguments.candidate_characterization,
            arguments.reference_characterization,
            arguments.output,
        )
    else:
        compare(arguments.config, arguments.budget, arguments.output_directory)


if __name__ == "__main__":
    main()
