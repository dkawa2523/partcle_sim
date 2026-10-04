"""Normalize and evaluate the M3-C0 COMSOL boundary-semantics probe.

This external V&V tool reads the deliberately small force-free Freeze and
Disappear exports.  It does not import or exercise the production solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

TOOL_REVISION: Final = "m3c_boundary_semantics_v2"
RECEIPT_PREFIX: Final = "M3CB|configuration|"
RAW_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "current_status_code",
    "final_status_code",
    "stop_or_event_time_s",
)
EXPECTED_STEP_SECONDS: Final = {
    "dt_10us": 1.0e-5,
    "dt_5us": 5.0e-6,
    "dt_2p5us": 2.5e-6,
}
EXPECTED_FRAMES = 61
EXPECTED_OUTPUT_INTERVAL_S = 2.5e-6
EXPECTED_END_S = 1.5e-4
ACTIVE_STATUS_CODE = 1
TIME_GRID_ABSOLUTE_TOLERANCE_S = 2.0e-14
EXPECTED_RECEIPT_COUNT = 6
RECEIPT_FIELDS: Final = frozenset(
    {
        "scenario",
        "boundary_feature",
        "boundary_id",
        "wall_condition",
        "expected_status",
        "step_s",
        "force_free",
        "dynamic_charge_active",
        "integrator",
        "output_times",
        "particle_rows",
        "source_model",
        "model_saved",
    }
)


@dataclass(frozen=True)
class StepSpec:
    label: str
    seconds: float


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    boundary_feature: str
    boundary_id: int
    wall_condition: str
    source_position_m: tuple[float, float]
    source_velocity_m_s: tuple[float, float]
    expected_status_code: int
    analytic_event_time_s: float
    analytic_hit_position_m: tuple[float, float]


@dataclass(frozen=True)
class StateRecord:
    particle_id: int
    time_s: float
    r_m: float
    z_m: float
    velocity_r_m_per_s: float
    velocity_z_m_per_s: float
    current_status_code: int
    final_status_code: int
    stop_or_event_time_s: float

    def csv_row(self) -> tuple[int | float, ...]:
        return (
            self.particle_id,
            self.time_s,
            self.r_m,
            self.z_m,
            self.velocity_r_m_per_s,
            self.velocity_z_m_per_s,
            self.current_status_code,
            self.final_status_code,
            self.stop_or_event_time_s,
        )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str) -> int:
    number = _finite_float(value, name)
    result = round(number)
    if abs(number - result) > 1.0e-9:
        raise ValueError(f"{name} must be integer-valued")
    return result


def _pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return (_finite_float(value[0], f"{name}[0]"), _finite_float(value[1], f"{name}[1]"))


def _parse_steps(case: dict[str, Any]) -> tuple[StepSpec, ...]:
    raw = case.get("fixed_rk4_steps")
    parsed: dict[str, float] = {}
    if isinstance(raw, dict):
        items = raw.items()
        for label, value in items:
            seconds = value.get("seconds") if isinstance(value, dict) else value
            parsed[str(label)] = _finite_float(seconds, f"fixed_rk4_steps.{label}")
    elif isinstance(raw, list):
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"fixed_rk4_steps[{index}] must be an object")
            label = item.get("label")
            if not isinstance(label, str):
                raise ValueError(f"fixed_rk4_steps[{index}].label must be a string")
            if label in parsed:
                raise ValueError(f"duplicate fixed-step label {label!r}")
            parsed[label] = _finite_float(item.get("seconds"), f"fixed_rk4_steps[{index}].seconds")
    else:
        raise ValueError("case.fixed_rk4_steps must be a mapping or list")

    if set(parsed) != set(EXPECTED_STEP_SECONDS):
        raise ValueError("fixed-step labels must be dt_10us, dt_5us, and dt_2p5us")
    for label, expected in EXPECTED_STEP_SECONDS.items():
        if not math.isclose(parsed[label], expected, rel_tol=0.0, abs_tol=1.0e-18):
            raise ValueError(f"{label} must specify {expected:.17g} s")
    return tuple(StepSpec(label, EXPECTED_STEP_SECONDS[label]) for label in EXPECTED_STEP_SECONDS)


def _scenario_items(raw: object) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        if not all(isinstance(item, dict) for item in raw):
            raise ValueError("every scenarios list entry must be an object")
        return [dict(item) for item in raw]
    if isinstance(raw, dict):
        result: list[dict[str, Any]] = []
        for scenario_id, item in raw.items():
            if not isinstance(item, dict):
                raise ValueError(f"scenario {scenario_id!r} must be an object")
            entry = dict(item)
            configured_id = entry.setdefault("id", scenario_id)
            if configured_id != scenario_id:
                raise ValueError(f"scenario mapping key {scenario_id!r} differs from its id")
            result.append(entry)
        return result
    raise ValueError("scenarios must be a mapping or list")


def _parse_scenarios(raw: object) -> tuple[ScenarioSpec, ...]:
    scenarios: list[ScenarioSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(_scenario_items(raw)):
        scenario_id = item.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"scenarios[{index}].id must be a nonempty string")
        if Path(scenario_id).name != scenario_id or scenario_id in {".", ".."}:
            raise ValueError(f"unsafe scenario id {scenario_id!r}")
        if scenario_id in seen:
            raise ValueError(f"duplicate scenario id {scenario_id!r}")
        seen.add(scenario_id)
        feature = item.get("boundary_feature")
        if not isinstance(feature, str) or not feature:
            raise ValueError(f"scenario {scenario_id}: boundary_feature must be nonempty")
        wall_condition = item.get("wall_condition")
        if wall_condition not in {"Freeze", "Disappear"}:
            raise ValueError(f"scenario {scenario_id}: wall_condition must be Freeze or Disappear")
        expected_status = _integer(
            item.get("expected_status_code"), f"scenario {scenario_id}.expected_status_code"
        )
        if expected_status == ACTIVE_STATUS_CODE:
            raise ValueError(f"scenario {scenario_id}: terminal status cannot be active")
        scenarios.append(
            ScenarioSpec(
                scenario_id=scenario_id,
                boundary_feature=feature,
                boundary_id=_integer(
                    item.get("boundary_id"), f"scenario {scenario_id}.boundary_id"
                ),
                wall_condition=wall_condition,
                source_position_m=_pair(
                    item.get("source_position_m"), f"scenario {scenario_id}.source_position_m"
                ),
                source_velocity_m_s=_pair(
                    item.get("source_velocity_m_s"),
                    f"scenario {scenario_id}.source_velocity_m_s",
                ),
                expected_status_code=expected_status,
                analytic_event_time_s=_finite_float(
                    item.get("analytic_event_time_s"),
                    f"scenario {scenario_id}.analytic_event_time_s",
                ),
                analytic_hit_position_m=_pair(
                    item.get("analytic_hit_position_m"),
                    f"scenario {scenario_id}.analytic_hit_position_m",
                ),
            )
        )
    if not scenarios:
        raise ValueError("at least one boundary scenario is required")
    return tuple(scenarios)


def _acceptance(payload: dict[str, Any]) -> dict[str, float | bool]:
    raw = payload.get("acceptance")
    if not isinstance(raw, dict):
        raise ValueError("acceptance must be an object")
    result: dict[str, float | bool] = {}
    for name in (
        "maximum_event_time_absolute_error_s",
        "maximum_event_time_step_spread_s",
        "maximum_hit_position_absolute_error_m",
        "maximum_freeze_post_event_position_spread_m",
        "maximum_active_position_absolute_error_m",
        "maximum_active_velocity_absolute_error_m_per_s",
    ):
        value = _finite_float(raw.get(name), f"acceptance.{name}")
        if value < 0.0:
            raise ValueError(f"acceptance.{name} must be nonnegative")
        result[name] = value
    for name in ("require_expected_final_status", "require_single_terminal_event"):
        value = raw.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"acceptance.{name} must be boolean")
        result[name] = value
    return result


def _locked_runtime_config(
    payload: dict[str, Any], path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_model = payload.get("source_model")
    numerics = payload.get("numerics")
    physics = payload.get("physics")
    if not all(isinstance(value, dict) for value in (source_model, numerics, physics)):
        raise ValueError(f"{path}: source_model, numerics, and physics must be objects")
    assert isinstance(source_model, dict)
    assert isinstance(numerics, dict)
    assert isinstance(physics, dict)
    if (
        source_model.get("staged_filename") != "source_copy.mph"
        or source_model.get("saved") is not False
    ):
        raise ValueError(f"{path}: source_model staging and no-save settings differ")
    if numerics.get("integrator") != "classical_rk4":
        raise ValueError(f"{path}: numerics.integrator must be classical_rk4")
    if physics.get("force_free") is not True or physics.get("dynamic_charge_active") is not False:
        raise ValueError(f"{path}: physics must select force-free fixed-charge operation")
    return source_model, numerics, physics


def _locked_case(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    case = payload.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"{path}: case must be an object")
    if _integer(case.get("particle_count_per_scenario"), "case.particle_count_per_scenario") != 1:
        raise ValueError("case.particle_count_per_scenario must be 1")
    if _integer(case.get("output_frames"), "case.output_frames") != EXPECTED_FRAMES:
        raise ValueError(f"case.output_frames must be {EXPECTED_FRAMES}")
    interval = _finite_float(case.get("output_interval_s"), "case.output_interval_s")
    end = _finite_float(case.get("time_end_s"), "case.time_end_s")
    if not math.isclose(interval, EXPECTED_OUTPUT_INTERVAL_S, rel_tol=0.0, abs_tol=1.0e-18):
        raise ValueError("case.output_interval_s must be 2.5 us")
    if not math.isclose(end, EXPECTED_END_S, rel_tol=0.0, abs_tol=1.0e-18):
        raise ValueError("case.time_end_s must be 150 us")
    return case


def _validate_raw_export(payload: dict[str, Any], path: Path) -> None:
    raw_export = payload.get("raw_export")
    if not isinstance(raw_export, dict):
        raise ValueError(f"{path}: raw_export must be an object")
    if raw_export.get("table") != "state_raw_wide.csv":
        raise ValueError("raw_export.table must be state_raw_wide.csv")
    if raw_export.get("columns_per_frame") != len(RAW_COLUMNS):
        raise ValueError(f"raw_export.columns_per_frame must be {len(RAW_COLUMNS)}")
    if tuple(raw_export.get("columns", ())) != RAW_COLUMNS:
        raise ValueError("raw_export.columns do not match the boundary probe contract")


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], tuple[StepSpec, ...], tuple[ScenarioSpec, ...], dict[str, float | bool]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: configuration root must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    if payload.get("evaluation_revision") != 2:
        raise ValueError(f"{path}: unsupported evaluation_revision")
    if not isinstance(payload.get("evaluation_id"), str) or not payload["evaluation_id"]:
        raise ValueError(f"{path}: evaluation_id must be a nonempty string")
    _locked_runtime_config(payload, path)
    case = _locked_case(payload, path)
    _validate_raw_export(payload, path)
    return (
        payload,
        _parse_steps(case),
        _parse_scenarios(payload.get("scenarios")),
        _acceptance(payload),
    )


def _receipt_fields(log_path: Path, receipt_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in receipt_text.split("|")[2:]:
        if "=" not in token:
            raise ValueError(f"{log_path}: malformed configuration receipt token {token!r}")
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ValueError(f"{log_path}: duplicate or empty receipt key {key!r}")
        fields[key] = value
    missing = sorted(RECEIPT_FIELDS - fields.keys())
    unexpected = sorted(fields.keys() - RECEIPT_FIELDS)
    if missing or unexpected:
        raise ValueError(
            f"{log_path}: configuration receipt fields differ; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return fields


def _receipt_step(fields: dict[str, str], steps: tuple[StepSpec, ...]) -> StepSpec | None:
    try:
        seconds = float(fields["step_s"])
    except ValueError:
        return None
    return next(
        (
            step
            for step in steps
            if math.isclose(seconds, step.seconds, rel_tol=0.0, abs_tol=1.0e-18)
        ),
        None,
    )


def _expected_receipt_fields(
    config: dict[str, Any], scenario: ScenarioSpec, step: StepSpec
) -> dict[str, str]:
    case = config["case"]
    physics = config["physics"]
    numerics = config["numerics"]
    source_model = config["source_model"]
    return {
        "scenario": scenario.scenario_id,
        "boundary_feature": scenario.boundary_feature,
        "boundary_id": str(scenario.boundary_id),
        "wall_condition": scenario.wall_condition,
        "expected_status": str(scenario.expected_status_code),
        "step_s": f"{step.seconds:.17g}",
        "force_free": str(physics["force_free"]).lower(),
        "dynamic_charge_active": str(physics["dynamic_charge_active"]).lower(),
        "integrator": str(numerics["integrator"]),
        "output_times": str(case["output_frames"]),
        "particle_rows": str(case["particle_count_per_scenario"]),
        "source_model": str(source_model["staged_filename"]),
        "model_saved": str(source_model["saved"]).lower(),
    }


def _validate_receipt_values(
    log_path: Path,
    fields: dict[str, str],
    expected: dict[str, str],
    expected_step_s: float,
) -> None:
    try:
        step_matches = math.isclose(
            float(fields["step_s"]), expected_step_s, rel_tol=0.0, abs_tol=1.0e-18
        )
    except ValueError:
        step_matches = False
    if not step_matches:
        raise ValueError(f"{log_path}: configuration receipt has an unknown step_s")
    for key in RECEIPT_FIELDS - {"step_s"}:
        if fields[key] != expected[key]:
            raise ValueError(f"{log_path}: expected {key}={expected[key]}, found {fields[key]}")


def _parse_configuration_receipts(
    log_path: Path,
    config: dict[str, Any],
    steps: tuple[StepSpec, ...],
    scenarios: tuple[ScenarioSpec, ...],
) -> dict[tuple[str, str], dict[str, str]]:
    expected_pairs = {
        (scenario.scenario_id, step.label) for scenario in scenarios for step in steps
    }
    if len(expected_pairs) != EXPECTED_RECEIPT_COUNT:
        raise ValueError("configuration must define exactly six scenario/step runs")
    lines = [
        line[line.index(RECEIPT_PREFIX) :].strip()
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if RECEIPT_PREFIX in line
    ]
    if len(lines) != EXPECTED_RECEIPT_COUNT:
        raise ValueError(
            f"{log_path}: expected exactly six M3CB configuration receipts, found {len(lines)}"
        )

    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    receipts: dict[tuple[str, str], dict[str, str]] = {}
    for line in lines:
        fields = _receipt_fields(log_path, line)
        scenario = scenario_by_id.get(fields["scenario"])
        step = _receipt_step(fields, steps)
        if scenario is None or step is None:
            raise ValueError(f"{log_path}: receipt has an unknown scenario or step_s")
        key = (scenario.scenario_id, step.label)
        if key in receipts:
            raise ValueError(f"{log_path}: duplicate configuration receipt for {key}")
        expected = _expected_receipt_fields(config, scenario, step)
        _validate_receipt_values(log_path, fields, expected, step.seconds)
        receipts[key] = fields
    if receipts.keys() != expected_pairs:
        raise ValueError(f"{log_path}: configuration receipts do not cover all six runs")
    return receipts


def _parse_status(value: float, name: str, source: Path) -> int:
    if not math.isfinite(value):
        raise ValueError(f"{source}: {name} must be finite")
    result = round(value)
    if abs(value - result) > 1.0e-9:
        raise ValueError(f"{source}: {name} must be integer-valued")
    return result


def _read_wide_row(path: Path) -> list[float]:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("%"):
                continue
            try:
                rows.append([float(value) for value in next(csv.reader([line]))])
            except ValueError as error:
                raise ValueError(f"{path}: raw table contains a nonnumeric value") from error
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly 1 particle row, found {len(rows)}")
    expected_width = EXPECTED_FRAMES * len(RAW_COLUMNS)
    if len(rows[0]) != expected_width:
        raise ValueError(f"{path}: expected {expected_width} values, found {len(rows[0])}")
    return rows[0]


def _state_record(values: list[float], frame: int, path: Path) -> StateRecord:
    if not math.isfinite(values[1]):
        raise ValueError(f"{path}: time_s must be finite at frame {frame}")
    if any(math.isinf(value) for value in values[2:6]) or math.isinf(values[8]):
        raise ValueError(f"{path}: infinite state value at frame {frame}")
    record = StateRecord(
        particle_id=_parse_status(values[0], "particle_id", path),
        time_s=values[1],
        r_m=values[2],
        z_m=values[3],
        velocity_r_m_per_s=values[4],
        velocity_z_m_per_s=values[5],
        current_status_code=_parse_status(values[6], "current_status_code", path),
        final_status_code=_parse_status(values[7], "final_status_code", path),
        stop_or_event_time_s=values[8],
    )
    active_values = (
        record.r_m,
        record.z_m,
        record.velocity_r_m_per_s,
        record.velocity_z_m_per_s,
    )
    if record.current_status_code == ACTIVE_STATUS_CODE and not all(
        math.isfinite(value) for value in active_values
    ):
        raise ValueError(f"{path}: active state is nonfinite at frame {frame}")
    return record


def _validate_state_grid(records: list[StateRecord], path: Path) -> None:
    particle_ids = {record.particle_id for record in records}
    if len(particle_ids) != 1:
        raise ValueError(f"{path}: particle_id changes within the wide row")
    for frame, record in enumerate(records):
        expected_time = frame * EXPECTED_OUTPUT_INTERVAL_S
        if not math.isclose(
            record.time_s,
            expected_time,
            rel_tol=0.0,
            abs_tol=TIME_GRID_ABSOLUTE_TOLERANCE_S,
        ):
            raise ValueError(
                f"{path}: output time grid differs at frame {frame}: "
                f"{record.time_s:.17g} != {expected_time:.17g}"
            )


def _read_state(path: Path) -> list[StateRecord]:
    values = _read_wide_row(path)
    records = [
        _state_record(
            values[frame * len(RAW_COLUMNS) : (frame + 1) * len(RAW_COLUMNS)], frame, path
        )
        for frame in range(EXPECTED_FRAMES)
    ]
    _validate_state_grid(records, path)
    return records


def _write_state(path: Path, records: list[StateRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(RAW_COLUMNS)
        writer.writerows(record.csv_row() for record in records)


def _event_time(records: list[StateRecord]) -> tuple[float | None, float | None]:
    terminal = [record for record in records if record.current_status_code != ACTIVE_STATUS_CODE]
    candidates = [
        record.stop_or_event_time_s
        for record in terminal
        if math.isfinite(record.stop_or_event_time_s) and record.stop_or_event_time_s > 0.0
    ]
    if not candidates:
        candidates = [
            record.stop_or_event_time_s
            for record in records
            if math.isfinite(record.stop_or_event_time_s) and record.stop_or_event_time_s > 0.0
        ]
    if not candidates:
        return None, None
    return math.fsum(candidates) / len(candidates), max(candidates) - min(candidates)


def _single_terminal_transition(records: list[StateRecord], expected_status: int) -> bool:
    statuses = [record.current_status_code for record in records]
    transitions = sum(left != right for left, right in itertools.pairwise(statuses))
    try:
        first_terminal = next(index for index, status in enumerate(statuses) if status != 1)
    except StopIteration:
        return False
    return (
        first_terminal > 0
        and transitions == 1
        and all(status == ACTIVE_STATUS_CODE for status in statuses[:first_terminal])
        and all(status == expected_status for status in statuses[first_terminal:])
    )


def _active_force_free_path(
    records: list[StateRecord], scenario: ScenarioSpec
) -> dict[str, int | float | None]:
    active = [record for record in records if record.current_status_code == ACTIVE_STATUS_CODE]
    position_errors = [
        math.hypot(
            record.r_m
            - (scenario.source_position_m[0] + scenario.source_velocity_m_s[0] * record.time_s),
            record.z_m
            - (scenario.source_position_m[1] + scenario.source_velocity_m_s[1] * record.time_s),
        )
        for record in active
    ]
    velocity_errors = [
        math.hypot(
            record.velocity_r_m_per_s - scenario.source_velocity_m_s[0],
            record.velocity_z_m_per_s - scenario.source_velocity_m_s[1],
        )
        for record in active
    ]
    return {
        "active_frame_count": len(active),
        "maximum_position_absolute_error_m": max(position_errors, default=None),
        "maximum_velocity_absolute_error_m_per_s": max(velocity_errors, default=None),
    }


def _velocity_characterization(
    records: list[StateRecord], source_velocity: tuple[float, float]
) -> dict[str, Any]:
    finite: list[tuple[float, float]] = []
    nan_pairs = 0
    partial_nonfinite = 0
    for record in records:
        velocity = (record.velocity_r_m_per_s, record.velocity_z_m_per_s)
        if all(math.isfinite(value) for value in velocity):
            finite.append(velocity)
        elif all(math.isnan(value) for value in velocity):
            nan_pairs += 1
        else:
            partial_nonfinite += 1
    maximum_speed = max((math.hypot(*velocity) for velocity in finite), default=None)
    maximum_source_difference = max(
        (
            math.hypot(velocity[0] - source_velocity[0], velocity[1] - source_velocity[1])
            for velocity in finite
        ),
        default=None,
    )
    classification = _velocity_classification(finite, nan_pairs, partial_nonfinite, source_velocity)
    return {
        "gate_status": "CHARACTERIZED_NOT_GATED",
        "classification": classification,
        "finite_pair_count": len(finite),
        "nan_pair_count": nan_pairs,
        "partial_nonfinite_pair_count": partial_nonfinite,
        "maximum_speed_m_per_s": maximum_speed,
        "maximum_difference_from_source_velocity_m_per_s": maximum_source_difference,
        "all_finite_pairs_exactly_zero": bool(finite)
        and all(velocity == (0.0, 0.0) for velocity in finite),
        "all_finite_pairs_exactly_source_velocity": bool(finite)
        and all(velocity == source_velocity for velocity in finite),
    }


def _velocity_classification(
    finite: list[tuple[float, float]],
    nan_pairs: int,
    partial_nonfinite: int,
    source_velocity: tuple[float, float],
) -> str:
    if partial_nonfinite or (finite and nan_pairs):
        return "mixed_finite_and_nonfinite"
    if not finite:
        return "not_available_after_event"
    if all(velocity == (0.0, 0.0) for velocity in finite):
        return "exact_zero"
    if all(velocity == source_velocity for velocity in finite):
        return "exact_source_velocity_retained"
    return "finite_other"


def _gate(
    name: str,
    passed: bool,
    observed: object,
    limit: object,
    unit: str,
    detail: str,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "gate": name,
        "status": ("PASS" if passed else "FAIL") if enabled else "CHARACTERIZED_NOT_GATED",
        "observed_value": observed,
        "limit_value": limit,
        "unit": unit,
        "detail": detail,
    }


def _freeze_position_semantics(
    terminal: list[StateRecord], limit_m: float
) -> tuple[dict[str, Any], tuple[float, float] | None, str]:
    finite = all(math.isfinite(record.r_m) and math.isfinite(record.z_m) for record in terminal)
    hit = (terminal[0].r_m, terminal[0].z_m) if terminal and finite else None
    spread = (
        max(math.hypot(record.r_m - hit[0], record.z_m - hit[1]) for record in terminal)
        if hit is not None
        else None
    )
    passed = bool(terminal) and finite and spread is not None and spread <= limit_m
    gate = _gate(
        "freeze_post_event_position_retention",
        passed,
        spread,
        limit_m,
        "m",
        "all post-event positions are finite and fixed relative to the first held state",
    )
    return gate, hit, "DIRECT_COMSOL_FROZEN_STATE_OBSERVATION"


def _disappear_position_semantics(
    terminal: list[StateRecord],
) -> tuple[dict[str, Any], None, str]:
    all_nan = bool(terminal) and all(
        math.isnan(record.r_m) and math.isnan(record.z_m) for record in terminal
    )
    gate = _gate(
        "disappear_post_event_position_is_nan",
        all_nan,
        sum(math.isnan(record.r_m) and math.isnan(record.z_m) for record in terminal),
        len(terminal),
        "post-event frames",
        "both saved position components are NaN after the Disappear event",
    )
    return (
        gate,
        None,
        "ANALYTIC_RECONSTRUCTION_FROM_COMSOL_INITIAL_STATE_NOT_DIRECT_EVENT_OBSERVATION",
    )


def _position_semantics(
    scenario: ScenarioSpec,
    terminal: list[StateRecord],
    limit_m: float,
) -> tuple[dict[str, Any], tuple[float, float] | None, str]:
    if scenario.wall_condition == "Freeze":
        return _freeze_position_semantics(terminal, limit_m)
    return _disappear_position_semantics(terminal)


def _write_event_observation(
    path: Path,
    scenario: ScenarioSpec,
    particle_id: int,
    event_time_s: float | None,
    hit_position: tuple[float, float] | None,
    direct_position: tuple[float, float] | None,
    position_basis: str,
    observed_status_code: int | None,
) -> None:
    header = (
        "scenario_id",
        "particle_id",
        "event_ordinal",
        "boundary_feature",
        "boundary_id",
        "wall_condition",
        "status_code",
        "event_time_s",
        "event_time_basis",
        "hit_r_m",
        "hit_z_m",
        "hit_position_basis",
        "direct_comsol_r_m",
        "direct_comsol_z_m",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        if event_time_s is not None and hit_position is not None:
            writer.writerow(
                (
                    scenario.scenario_id,
                    particle_id,
                    0,
                    scenario.boundary_feature,
                    scenario.boundary_id,
                    scenario.wall_condition,
                    observed_status_code,
                    event_time_s,
                    "DIRECT_COMSOL_STOP_OR_EVENT_TIME_OBSERVATION",
                    *hit_position,
                    position_basis,
                    *(direct_position if direct_position is not None else ("", "")),
                )
            )


def _event_window(
    records: list[StateRecord], event_time_s: float | None
) -> tuple[float | None, float | None, bool]:
    first_terminal_index = next(
        (
            index
            for index, record in enumerate(records)
            if record.current_status_code != ACTIVE_STATUS_CODE
        ),
        None,
    )
    if first_terminal_index is None or first_terminal_index == 0:
        return None, None, False
    last_active_time_s = records[first_terminal_index - 1].time_s
    first_terminal_time_s = records[first_terminal_index].time_s
    bracketed = event_time_s is not None and all(
        (
            last_active_time_s < event_time_s,
            event_time_s <= first_terminal_time_s + TIME_GRID_ABSOLUTE_TOLERANCE_S,
        )
    )
    return last_active_time_s, first_terminal_time_s, bracketed


def _hit_position(
    scenario: ScenarioSpec,
    initial: StateRecord,
    event_time_s: float | None,
    direct_position: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if scenario.wall_condition != "Disappear" or event_time_s is None:
        return direct_position
    return (
        initial.r_m + initial.velocity_r_m_per_s * event_time_s,
        initial.z_m + initial.velocity_z_m_per_s * event_time_s,
    )


def _hit_error(
    hit_position: tuple[float, float] | None, analytic_position: tuple[float, float]
) -> float | None:
    if hit_position is None:
        return None
    return math.hypot(
        hit_position[0] - analytic_position[0], hit_position[1] - analytic_position[1]
    )


def _run_gates(
    records: list[StateRecord],
    scenario: ScenarioSpec,
    acceptance: dict[str, float | bool],
    semantics_gate: dict[str, Any],
    event_time_s: float | None,
    within_run_spread_s: float | None,
    event_bracketed: bool,
    event_time_error_s: float | None,
    hit_error_m: float | None,
    active_path: dict[str, int | float | None],
    velocity: dict[str, Any],
) -> list[dict[str, Any]]:
    final_status_matches = (
        all(record.final_status_code == scenario.expected_status_code for record in records)
        and records[-1].current_status_code == scenario.expected_status_code
    )
    single_transition = _single_terminal_transition(records, scenario.expected_status_code)
    single_event_pass = all(
        (
            single_transition,
            event_bracketed,
            event_time_s is not None,
            within_run_spread_s is not None,
            within_run_spread_s is not None
            and within_run_spread_s <= TIME_GRID_ABSOLUTE_TOLERANCE_S,
        )
    )
    event_limit = float(acceptance["maximum_event_time_absolute_error_s"])
    hit_limit = float(acceptance["maximum_hit_position_absolute_error_m"])
    active_position_error = active_path["maximum_position_absolute_error_m"]
    active_velocity_error = active_path["maximum_velocity_absolute_error_m_per_s"]
    active_position_limit = float(acceptance["maximum_active_position_absolute_error_m"])
    active_velocity_limit = float(acceptance["maximum_active_velocity_absolute_error_m_per_s"])
    gates = [
        _gate("exactly_one_particle", True, 1, 1, "particles", "one wide particle row"),
        _gate(
            "complete_61_frame_time_grid",
            True,
            len(records),
            EXPECTED_FRAMES,
            "frames",
            "0 through 150 us at 2.5 us intervals",
        ),
        _gate(
            "force_free_active_position_path",
            isinstance(active_position_error, float)
            and active_position_error <= active_position_limit,
            active_position_error,
            active_position_limit,
            "m maximum Euclidean error",
            "every active frame versus x(t) = x0 + v0*t in the configured force-free case",
        ),
        _gate(
            "force_free_active_velocity_path",
            isinstance(active_velocity_error, float)
            and active_velocity_error <= active_velocity_limit,
            active_velocity_error,
            active_velocity_limit,
            "m/s maximum Euclidean error",
            "every active frame velocity versus the configured constant v0",
        ),
        _gate(
            "expected_final_status",
            final_status_matches,
            sorted({record.final_status_code for record in records}),
            scenario.expected_status_code,
            "status code",
            "all final-status values and the final current status equal the configured code",
            enabled=bool(acceptance["require_expected_final_status"]),
        ),
        _gate(
            "single_terminal_event",
            single_event_pass,
            1 if single_transition else 0,
            1,
            "active-to-terminal transitions",
            "one leading active segment followed by the configured terminal status, with the "
            "event time bracketed by the adjacent saved frames",
            enabled=bool(acceptance["require_single_terminal_event"]),
        ),
        _gate(
            "analytic_event_time",
            event_time_error_s is not None and event_time_error_s <= event_limit,
            event_time_error_s,
            event_limit,
            "s absolute error",
            "COMSOL stop_or_event_time_s versus the force-free analytic event time",
        ),
        _gate(
            "analytic_hit_position",
            hit_error_m is not None and hit_error_m <= hit_limit,
            hit_error_m,
            hit_limit,
            "m Euclidean error",
            "Freeze uses the held COMSOL position; Disappear uses the labelled reconstruction",
        ),
        semantics_gate,
        _gate(
            "post_event_velocity",
            True,
            velocity["classification"],
            None,
            "classification",
            "velocity is characterized only; this configuration declares no velocity gate",
            enabled=False,
        ),
    ]
    return gates


def _normalize_run(
    directory: Path,
    scenario: ScenarioSpec,
    step: StepSpec,
    acceptance: dict[str, float | bool],
    configuration_receipt: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_path = directory / "state_raw_wide.csv"
    records = _read_state(raw_path)
    state_path = directory / "state.csv"
    _write_state(state_path, records)

    terminal = [record for record in records if record.current_status_code != ACTIVE_STATUS_CODE]
    event_time_s, within_run_event_time_spread_s = _event_time(records)
    last_active_time_s, first_terminal_time_s, event_bracketed = _event_window(
        records, event_time_s
    )
    freeze_limit = float(acceptance["maximum_freeze_post_event_position_spread_m"])
    semantics_gate, direct_position, position_basis = _position_semantics(
        scenario, terminal, freeze_limit
    )
    hit_position = _hit_position(scenario, records[0], event_time_s, direct_position)
    event_time_error_s = (
        abs(event_time_s - scenario.analytic_event_time_s) if event_time_s is not None else None
    )
    hit_error_m = _hit_error(hit_position, scenario.analytic_hit_position_m)
    active_path = _active_force_free_path(records, scenario)
    velocity = _velocity_characterization(terminal, scenario.source_velocity_m_s)
    gates = _run_gates(
        records,
        scenario,
        acceptance,
        semantics_gate,
        event_time_s,
        within_run_event_time_spread_s,
        event_bracketed,
        event_time_error_s,
        hit_error_m,
        active_path,
        velocity,
    )

    event_path = directory / "event_observation.csv"
    _write_event_observation(
        event_path,
        scenario,
        records[0].particle_id,
        event_time_s,
        hit_position,
        direct_position,
        position_basis,
        terminal[0].current_status_code if terminal else None,
    )
    summary = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "PASS" if all(gate["status"] != "FAIL" for gate in gates) else "FAIL",
        "scenario_id": scenario.scenario_id,
        "step_label": step.label,
        "fixed_step_s": step.seconds,
        "particle_id": records[0].particle_id,
        "frames": len(records),
        "time_window_s": [records[0].time_s, records[-1].time_s],
        "configuration_receipt": configuration_receipt,
        "active_force_free_path": active_path,
        "raw": {
            "path": raw_path.name,
            "sha256": _sha256(raw_path),
            "size_bytes": raw_path.stat().st_size,
        },
        "artifacts": {
            "state.csv": {"sha256": _sha256(state_path), "rows": len(records)},
            "event_observation.csv": {
                "sha256": _sha256(event_path),
                "rows": int(event_time_s is not None and hit_position is not None),
            },
        },
        "event": {
            "event_time_s": event_time_s,
            "within_run_event_time_spread_s": within_run_event_time_spread_s,
            "last_active_output_time_s": last_active_time_s,
            "first_terminal_output_time_s": first_terminal_time_s,
            "hit_position_m": list(hit_position) if hit_position is not None else None,
            "direct_comsol_hit_position_m": (
                list(direct_position) if direct_position is not None else None
            ),
            "hit_position_basis": position_basis,
            "event_time_absolute_error_s": event_time_error_s,
            "hit_position_euclidean_error_m": hit_error_m,
        },
        "post_event_velocity": velocity,
        "gates": gates,
    }
    _write_json(directory / "step_summary.json", summary)
    return summary, gates


def _step_consistency_gate(
    runs: Mapping[str, dict[str, Any]], acceptance: dict[str, float | bool]
) -> dict[str, Any]:
    event_times = {label: summary["event"]["event_time_s"] for label, summary in runs.items()}
    finite_times = [float(value) for value in event_times.values() if value is not None]
    spread = max(finite_times) - min(finite_times) if len(finite_times) == len(runs) else None
    limit = float(acceptance["maximum_event_time_step_spread_s"])
    pairwise = {
        "dt_10us_vs_dt_5us_s": (
            abs(event_times["dt_10us"] - event_times["dt_5us"])
            if event_times["dt_10us"] is not None and event_times["dt_5us"] is not None
            else None
        ),
        "dt_5us_vs_dt_2p5us_s": (
            abs(event_times["dt_5us"] - event_times["dt_2p5us"])
            if event_times["dt_5us"] is not None and event_times["dt_2p5us"] is not None
            else None
        ),
    }
    gate = _gate(
        "h_h2_h4_event_time_self_consistency",
        spread is not None and spread <= limit,
        spread,
        limit,
        "s maximum spread",
        "maximum event-time spread across dt_10us, dt_5us, and dt_2p5us",
    )
    gate["event_times_s"] = event_times
    gate["pairwise_absolute_differences"] = pairwise
    return gate


def _write_gates(path: Path, rows: list[dict[str, Any]]) -> None:
    header = (
        "scenario_id",
        "step_label",
        "gate",
        "status",
        "observed_value",
        "limit_value",
        "unit",
        "detail",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for name in ("observed_value", "limit_value"):
                if isinstance(output.get(name), (list, dict)):
                    output[name] = json.dumps(output[name], sort_keys=True, separators=(",", ":"))
            writer.writerow({name: output.get(name, "") for name in header})


def normalize(output_root: Path, config_path: Path) -> dict[str, Any]:
    """Normalize all configured scenarios and return the root evaluation report."""
    root = output_root.resolve()
    config, steps, scenarios, acceptance = _load_config(config_path.resolve())
    process_log = root / "comsol_process.log"
    receipts = _parse_configuration_receipts(process_log, config, steps, scenarios)
    scenario_reports: dict[str, Any] = {}
    flat_gates: list[dict[str, Any]] = []
    for scenario in scenarios:
        runs: dict[str, Any] = {}
        for step in steps:
            directory = root / scenario.scenario_id / step.label
            summary, gates = _normalize_run(
                directory,
                scenario,
                step,
                acceptance,
                receipts[(scenario.scenario_id, step.label)],
            )
            runs[step.label] = summary
            flat_gates.extend(
                {"scenario_id": scenario.scenario_id, "step_label": step.label, **gate}
                for gate in gates
            )
        consistency = _step_consistency_gate(runs, acceptance)
        flat_gates.append({"scenario_id": scenario.scenario_id, "step_label": "", **consistency})
        scenario_reports[scenario.scenario_id] = {
            "status": (
                "PASS"
                if consistency["status"] == "PASS"
                and all(summary["status"] == "PASS" for summary in runs.values())
                else "FAIL"
            ),
            "boundary_feature": scenario.boundary_feature,
            "boundary_id": scenario.boundary_id,
            "wall_condition": scenario.wall_condition,
            "expected_status_code": scenario.expected_status_code,
            "analytic_event_time_s": scenario.analytic_event_time_s,
            "analytic_hit_position_m": list(scenario.analytic_hit_position_m),
            "runs": runs,
            "event_time_self_consistency": consistency,
        }

    failures = [gate for gate in flat_gates if gate["status"] == "FAIL"]
    report = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "evaluation_id": config["evaluation_id"],
        "evaluation_revision": config.get("evaluation_revision"),
        "classification": "external_comsol_boundary_semantics_probe",
        "status": "PASS" if not failures else "FAIL",
        "configuration": config,
        "configuration_receipts": {
            "path": process_log.name,
            "sha256": _sha256(process_log),
            "expected_count": EXPECTED_RECEIPT_COUNT,
            "validated_count": len(receipts),
            "all_scenario_step_pairs_validated": True,
        },
        "scenarios": scenario_reports,
        "gate_counts": {
            "pass": sum(gate["status"] == "PASS" for gate in flat_gates),
            "fail": len(failures),
            "characterized_not_gated": sum(
                gate["status"] == "CHARACTERIZED_NOT_GATED" for gate in flat_gates
            ),
        },
        "claim_scope": {
            "freeze_and_disappear_semantics": "PASS" if not failures else "FAIL",
            "post_event_velocity": "CHARACTERIZED_NOT_GATED",
            "production_solver_boundary_parity": "NOT_TESTED",
        },
    }
    _write_gates(root / "gates.csv", flat_gates)
    _write_json(root / "boundary_semantics_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize and evaluate the M3-C0 COMSOL boundary-semantics probe."
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    normalize(arguments.output_root, arguments.config)


if __name__ == "__main__":
    main()
