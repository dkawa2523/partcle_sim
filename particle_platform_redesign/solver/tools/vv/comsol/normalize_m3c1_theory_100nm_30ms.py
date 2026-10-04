"""Normalize the formal M3-C1 100 nm, 30 ms COMSOL matrix.

This is an external V&V tool.  It converts COMSOL's particle-major wide state
tables into one producer-neutral long trajectory and one sparse terminal-event
ledger per fixed step.  It deliberately leaves escaped hit coordinates absent:
COMSOL Disappear does not expose those coordinates after the event.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

TOOL_REVISION: Final = "m3c1_theory_100nm_30ms_reference_normalizer_v3"
STEP_LABELS: Final = ("coarse", "medium", "fine")
RUN_KEYS: Final = STEP_LABELS
COMSOL_RUN_REVISION: Final = "m3c1_theory_100nm_30ms_comsol_exact_p1_v2"
WORKFLOWS: Final = ("caseA", "caseP")
EXPECTED_SHARED_CONFIG_SHA256: Final = (
    "9a98f4edf2a6d7048e0a6aa4e3eacdc1e4b538b8ea11a9082baa212971dbe3a1"
)
EXPECTED_COMSOL_CONFIG_SHA256: Final = (
    "10290e936b642523e72a4cd482befbdb50d7e4b4ae7aa72ee8e275eee068fa6a"
)
EXPECTED_SOURCE_MPH_SHA256: Final = (
    "3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524"
)
EXPECTED_JAVA_TEMPLATE_SHA256: Final = (
    "dffbd671f0fad6b54919ea81bcbdf20e62fb16c9619d45fa957f170944f3ec9f"
)
EXPECTED_EXECUTED_RUNNER_SHA256: Final = (
    "52b21eaca6f7620b4002ec8ddd7fda43914b24fa2558689f18fa5d6fe14adf75"
)
EXPECTED_COMSOL_VERSION: Final = "COMSOL Multiphysics 6.4.0.429"
EXPECTED_CANDIDATE_IDENTITIES: Final = {
    "caseA": (
        "9a268218da145ae0897336a40fed8f0b4b7ddedbe6be0213181b1b696c439285",
        "sha256:64ac2fd9e87bbc1fed40b3e8336c86a6f2ceddfb292433ce84c1bbff0f1a336d",
    ),
    "caseP": (
        "c54b4b658213230e82307ca89018538c0240bfcc83e793206d5093f4f08908e9",
        "sha256:8d0fb04a19283194d45ed84db73154e946cfa9c827c5048dba107287b2470c83",
    ),
}
EXPECTED_EXECUTED_ARTIFACTS: Final = {
    "run_spec.properties",
    "RunM3C1Theory100nm30ms.class",
    "RunM3C1Theory100nm30ms.class.status",
    "RunM3C1Theory100nm30ms.java",
    "run_m3c1_theory_100nm_30ms_reference.executed.ps1",
}
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
    "charge_rate_e_per_s",
    "particle_mass_kg",
    "sampled_release_velocity_r_m_per_s",
    "sampled_release_velocity_z_m_per_s",
    "sampled_release_charge_number_e",
)
FORCE_COLUMN_COUNT: Final = 20
PRIMITIVE_COLUMN_COUNT: Final = 24
PREPARED_COMPONENT_SPECS: Final = (
    (
        "gas_density",
        "value",
        "m3c1_rhog_sectionwise.txt",
        "m3c1_rhog",
        "gas_density_kg_per_m3",
        "kg/m^3",
    ),
    (
        "gas_dynamic_viscosity",
        "value",
        "m3c1_mug_sectionwise.txt",
        "m3c1_mug",
        "gas_dynamic_viscosity_Pa_s",
        "Pa*s",
    ),
    ("gas_temperature", "value", "m3c1_Tg_sectionwise.txt", "m3c1_Tg", "gas_temperature_K", "K"),
    (
        "gas_mean_free_path",
        "value",
        "m3c1_lambdag_sectionwise.txt",
        "m3c1_lambdag",
        "gas_mean_free_path_m",
        "m",
    ),
    (
        "electron_number_density",
        "value",
        "m3c1_ne_sectionwise.txt",
        "m3c1_ne",
        "electron_density_per_m3",
        "1/m^3",
    ),
    (
        "positive_ion_number_density",
        "value",
        "m3c1_ni_sectionwise.txt",
        "m3c1_ni",
        "positive_ion_density_per_m3",
        "1/m^3",
    ),
    (
        "electron_thermal_voltage",
        "value",
        "m3c1_Te_sectionwise.txt",
        "m3c1_Te",
        "electron_thermal_voltage_V",
        "V",
    ),
    (
        "positive_ion_thermal_voltage",
        "value",
        "m3c1_TiV_sectionwise.txt",
        "m3c1_TiV",
        "positive_ion_thermal_voltage_V",
        "V",
    ),
    (
        "effective_positive_ion_mass",
        "value",
        "m3c1_mi_sectionwise.txt",
        "m3c1_mi",
        "effective_positive_ion_mass_kg",
        "kg",
    ),
    (
        "screening_length",
        "value",
        "m3c1_lambdaD_sectionwise.txt",
        "m3c1_lambdaD",
        "screening_length_m",
        "m",
    ),
    (
        "ion_neutral_mean_free_path",
        "value",
        "m3c1_lambdaIn_sectionwise.txt",
        "m3c1_lambdaIn",
        "ion_neutral_mean_free_path_m",
        "m",
    ),
    (
        "azimuthal_gas_vorticity",
        "value",
        "m3c1_omegaPhi_sectionwise.txt",
        "m3c1_omegaPhi",
        "azimuthal_gas_vorticity_per_s",
        "1/s",
    ),
    ("gas_velocity", "r", "m3c1_ugr_sectionwise.txt", "m3c1_ugr", "gas_velocity_r_m_per_s", "m/s"),
    ("gas_velocity", "z", "m3c1_ugz_sectionwise.txt", "m3c1_ugz", "gas_velocity_z_m_per_s", "m/s"),
    (
        "electric_field",
        "r",
        "m3c1_Er_sectionwise.txt",
        "m3c1_Er",
        "electric_field_r_V_per_m",
        "V/m",
    ),
    (
        "electric_field",
        "z",
        "m3c1_Ez_sectionwise.txt",
        "m3c1_Ez",
        "electric_field_z_V_per_m",
        "V/m",
    ),
    (
        "positive_ion_velocity",
        "r",
        "m3c1_uir_sectionwise.txt",
        "m3c1_uir",
        "positive_ion_velocity_r_m_per_s",
        "m/s",
    ),
    (
        "positive_ion_velocity",
        "z",
        "m3c1_uiz_sectionwise.txt",
        "m3c1_uiz",
        "positive_ion_velocity_z_m_per_s",
        "m/s",
    ),
    (
        "gradient_mean_e_squared",
        "r",
        "m3c1_gradE2r_sectionwise.txt",
        "m3c1_gradE2r",
        "gradient_mean_e_squared_r_V2_per_m3",
        "V^2/m^3",
    ),
    (
        "gradient_mean_e_squared",
        "z",
        "m3c1_gradE2z_sectionwise.txt",
        "m3c1_gradE2z",
        "gradient_mean_e_squared_z_V2_per_m3",
        "V^2/m^3",
    ),
    (
        "gas_translational_heat_flux",
        "r",
        "m3c1_qr_sectionwise.txt",
        "m3c1_qr",
        "gas_translational_heat_flux_r_W_per_m2",
        "W/m^2",
    ),
    (
        "gas_translational_heat_flux",
        "z",
        "m3c1_qz_sectionwise.txt",
        "m3c1_qz",
        "gas_translational_heat_flux_z_W_per_m2",
        "W/m^2",
    ),
)
PREPARED_FIELD_SPECS: Final = (
    ("gas_density", ("value",), "scalar", "kg/m^3"),
    ("gas_dynamic_viscosity", ("value",), "scalar", "Pa*s"),
    ("gas_temperature", ("value",), "scalar", "K"),
    ("gas_mean_free_path", ("value",), "scalar", "m"),
    ("electron_number_density", ("value",), "scalar", "1/m^3"),
    ("positive_ion_number_density", ("value",), "scalar", "1/m^3"),
    ("electron_thermal_voltage", ("value",), "scalar", "V"),
    ("positive_ion_thermal_voltage", ("value",), "scalar", "V"),
    ("effective_positive_ion_mass", ("value",), "scalar", "kg"),
    ("screening_length", ("value",), "scalar", "m"),
    ("ion_neutral_mean_free_path", ("value",), "scalar", "m"),
    ("azimuthal_gas_vorticity", ("value",), "scalar", "1/s"),
    ("gas_velocity", ("r", "z"), "axisymmetric_rz", "m/s"),
    ("electric_field", ("r", "z"), "axisymmetric_rz", "V/m"),
    ("positive_ion_velocity", ("r", "z"), "axisymmetric_rz", "m/s"),
    ("gradient_mean_e_squared", ("r", "z"), "axisymmetric_rz", "V^2/m^3"),
    ("gas_translational_heat_flux", ("r", "z"), "axisymmetric_rz", "W/m^2"),
)
PREPARED_RELEASE_SPECS: Final = (
    ("m3c1_vr0.txt", "m3c1_vr0", "velocity_r_m_per_s", "m/s"),
    ("m3c1_vz0.txt", "m3c1_vz0", "velocity_z_m_per_s", "m/s"),
    ("m3c1_Z0.txt", "m3c1_Z0", "charge_number", "1"),
)
TRAJECTORY_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "current_status",
    "final_status",
    "stop_time_s",
)
EVENT_COLUMNS: Final = (
    "particle_id",
    "event_ordinal",
    "event_time_s",
    "first_observed_time_s",
    "fate",
    "boundary_group",
    "observation_basis",
    "terminal_r_m",
    "terminal_z_m",
    "terminal_charge_number_e",
)
STATUS_NAMES: Final = {1: "active", 2: "held", 3: "stuck", 4: "escaped"}
BOUNDARY_GROUPS: Final = {2: "gas_inlet", 3: "material", 4: "pump_outlet"}


@dataclass(frozen=True)
class Protocol:
    config_path: Path
    config_sha256: str
    particle_count: int
    output_times_s: tuple[float, ...]
    steps_s_by_workflow: dict[str, tuple[float, float, float]]
    charge_lipschitz_s_inv_by_workflow: dict[str, float]
    maximum_dt_charge_lipschitz: float
    initial_ulp_multiplier: float
    roundoff_multiplier: float


@dataclass(frozen=True)
class StateRecord:
    particle_id: int
    time_s: float
    state: tuple[float, float, float, float, float]
    current_status: int
    final_status: int
    stop_time_s: float | None
    mass_kg: float
    sampled_release: tuple[float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
    return value.lower()


def _require_finite_json(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a nonfinite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, f"{name}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json(child, f"{name}[{index}]")


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


def _inclusive_decimal_range(start: object, stop: object, step: object) -> list[float]:
    first = Decimal(str(start))
    last = Decimal(str(stop))
    increment = Decimal(str(step))
    if increment <= 0 or last < first:
        raise ValueError("output schedule segment is invalid")
    count, remainder = divmod(last - first, increment)
    if remainder != 0:
        raise ValueError("output schedule segment does not end on its step")
    return [float(first + index * increment) for index in range(int(count) + 1)]


def _output_schedule(matrix: dict[str, object]) -> tuple[float, ...]:
    raw_segments = matrix.get("output_schedule_segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("matrix.output_schedule_segments must be a nonempty list")
    values: list[float] = []
    for index, raw in enumerate(raw_segments):
        segment = _mapping(raw, f"output schedule segment {index}")
        part = _inclusive_decimal_range(
            segment.get("start_s"), segment.get("stop_s"), segment.get("step_s")
        )
        if values and part[0] <= values[-1]:
            raise ValueError("output schedule segments must be strictly ordered")
        values.extend(part)
    expected_count = _positive_int(matrix.get("output_count"), "matrix.output_count")
    if len(values) != expected_count or values[0] != 0.0:
        raise ValueError("output schedule count or initial time differs")
    end_s = _number(matrix.get("time_end_s"), "matrix.time_end_s")
    if values[-1] != end_s:
        raise ValueError("output schedule final time differs from matrix.time_end_s")
    return tuple(values)


def _workflow_steps(
    raw: object, workflow: str, time_end_s: float, maximum_lipschitz: float
) -> tuple[tuple[float, float, float], float]:
    record = _mapping(raw, f"workflows.{workflow}")
    raw_steps = record.get("fixed_rk4_steps_s")
    if not isinstance(raw_steps, list) or len(raw_steps) != len(RUN_KEYS):
        raise ValueError(f"workflows.{workflow}.fixed_rk4_steps_s must contain 3 steps")
    steps = tuple(_number(value, f"{workflow} fixed RK4 step") for value in raw_steps)
    if (
        any(step <= 0.0 for step in steps)
        or steps[0] != 2.0 * steps[1]
        or steps[1] != 2.0 * steps[2]
    ):
        raise ValueError(f"{workflow} fixed RK4 steps must be positive h, h/2, h/4")
    lipschitz = _number(
        record.get("charge_lipschitz_s_inv"),
        f"workflows.{workflow}.charge_lipschitz_s_inv",
    )
    if lipschitz <= 0.0:
        raise ValueError(f"{workflow} charge Lipschitz bound must be positive")
    for step in steps:
        count = round(time_end_s / step)
        if not math.isclose(count * step, time_end_s, rel_tol=0.0, abs_tol=math.ulp(time_end_s)):
            raise ValueError(f"{workflow} fixed step does not divide the configured time span")
        if step * lipschitz > maximum_lipschitz:
            raise ValueError(f"{workflow} fixed step violates the dt-charge Lipschitz limit")
    return (steps[0], steps[1], steps[2]), lipschitz


def _load_protocol(config_path: Path) -> tuple[Protocol, dict[str, Any]]:
    resolved = config_path.expanduser().resolve()
    config_sha256 = _sha256(resolved)
    if config_sha256 != EXPECTED_SHARED_CONFIG_SHA256:
        raise ValueError("shared M3-C1 configuration hash differs from the fixed formal contract")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    config = _mapping(raw, "configuration")
    _require_finite_json(config, "configuration")
    if config.get("evaluation_id") != "M3-C1-theory-consistent-100nm-30ms":
        raise ValueError("unexpected M3-C1 evaluation_id")
    if config.get("evaluation_revision") != 2:
        raise ValueError("unsupported M3-C1 evaluation revision")
    matrix = _mapping(config.get("matrix"), "matrix")
    if matrix.get("run_keys") != list(RUN_KEYS):
        raise ValueError("matrix.run_keys must be coarse, medium, fine")
    workflows = _mapping(config.get("workflows"), "workflows")
    if set(workflows) != set(WORKFLOWS):
        raise ValueError("configuration workflows must be exactly caseA and caseP")
    time_end_s = _number(matrix.get("time_end_s"), "matrix.time_end_s")
    acceptance = _mapping(config.get("acceptance"), "acceptance")
    maximum_lipschitz = _number(
        acceptance.get("maximum_dt_charge_lipschitz"),
        "acceptance.maximum_dt_charge_lipschitz",
    )
    if maximum_lipschitz <= 0.0:
        raise ValueError("maximum dt-charge Lipschitz product must be positive")
    steps_by_workflow: dict[str, tuple[float, float, float]] = {}
    charge_lipschitz_by_workflow: dict[str, float] = {}
    for workflow in WORKFLOWS:
        steps, lipschitz = _workflow_steps(
            workflows[workflow], workflow, time_end_s, maximum_lipschitz
        )
        steps_by_workflow[workflow] = steps
        charge_lipschitz_by_workflow[workflow] = lipschitz
    multiplier = _number(
        acceptance.get("initial_state_ulp_multiplier"),
        "acceptance.initial_state_ulp_multiplier",
    )
    roundoff_multiplier = _number(
        acceptance.get("roundoff_multiplier"), "acceptance.roundoff_multiplier"
    )
    if multiplier <= 0.0 or roundoff_multiplier <= 0.0:
        raise ValueError("ULP multipliers must be positive")
    protocol = Protocol(
        config_path=resolved,
        config_sha256=config_sha256,
        particle_count=_positive_int(matrix.get("particle_count"), "matrix.particle_count"),
        output_times_s=_output_schedule(matrix),
        steps_s_by_workflow=steps_by_workflow,
        charge_lipschitz_s_inv_by_workflow=charge_lipschitz_by_workflow,
        maximum_dt_charge_lipschitz=maximum_lipschitz,
        initial_ulp_multiplier=multiplier,
        roundoff_multiplier=roundoff_multiplier,
    )
    return protocol, config


def _wide_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            if line.startswith("%") or not line.strip():
                continue
            rows.append([float(value) for value in next(csv.reader([line]))])
    return rows


def _integer(value: float, name: str, path: Path) -> int:
    result = round(value)
    if not math.isfinite(value) or abs(value - result) > 1.0e-9:
        raise ValueError(f"{path}: {name} is not integer-valued")
    return result


def _matches_time(actual: float, expected: float) -> bool:
    tolerance = 64.0 * math.ulp(max(abs(actual), abs(expected), 1.0e-300))
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def _optional_finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _state_record(values: list[float], path: Path) -> StateRecord:
    particle_id = _integer(values[0], "particle_id", path)
    current = _integer(values[7], "current_status_code", path)
    final = _integer(values[8], "final_status_code", path)
    if current not in STATUS_NAMES or final not in STATUS_NAMES:
        raise ValueError(f"{path}: unknown COMSOL particle status")
    return StateRecord(
        particle_id=particle_id,
        time_s=values[1],
        state=(values[2], values[3], values[4], values[5], values[6]),
        current_status=current,
        final_status=final,
        stop_time_s=_optional_finite(values[9]),
        mass_kg=values[11],
        sampled_release=(values[12], values[13], values[14]),
    )


def _validate_record_state(record: StateRecord, path: Path) -> None:
    finite = tuple(math.isfinite(value) for value in record.state)
    if record.current_status == 1 and not all(finite):
        raise ValueError(f"{path}: active state is nonfinite for particle {record.particle_id}")
    if record.current_status in (2, 3) and not (finite[0] and finite[1] and finite[4]):
        raise ValueError(f"{path}: directly observed terminal state is incomplete")
    if record.current_status == 4 and any(finite):
        raise ValueError(f"{path}: Disappear payload must remain unobserved (NaN)")
    if record.current_status == 1 and (not math.isfinite(record.mass_kg) or record.mass_kg <= 0.0):
        raise ValueError(f"{path}: active particle mass is nonfinite or nonpositive")


def _read_state_table(path: Path, protocol: Protocol) -> list[list[StateRecord]]:
    raw_rows = _wide_rows(path)
    if len(raw_rows) != protocol.particle_count:
        raise ValueError(f"{path}: expected {protocol.particle_count} particle rows")
    width = len(STATE_COLUMNS) * len(protocol.output_times_s)
    histories: list[list[StateRecord]] = []
    seen: set[int] = set()
    for raw in raw_rows:
        if len(raw) != width:
            raise ValueError(f"{path}: expected {width} values per particle row")
        records: list[StateRecord] = []
        for frame, expected_time in enumerate(protocol.output_times_s):
            begin = frame * len(STATE_COLUMNS)
            record = _state_record(raw[begin : begin + len(STATE_COLUMNS)], path)
            if not _matches_time(record.time_s, expected_time):
                raise ValueError(f"{path}: output schedule differs at frame {frame}")
            _validate_record_state(record, path)
            records.append(record)
        particle_id = records[0].particle_id
        if particle_id in seen or any(record.particle_id != particle_id for record in records):
            raise ValueError(f"{path}: duplicate or changing particle ID {particle_id}")
        seen.add(particle_id)
        _validate_status_history(records, path, protocol.roundoff_multiplier)
        histories.append(records)
    if seen != set(range(1, protocol.particle_count + 1)):
        raise ValueError(f"{path}: particle IDs must be exactly 1..{protocol.particle_count}")
    return sorted(histories, key=lambda records: records[0].particle_id)


def _validate_status_history(
    records: list[StateRecord], path: Path, roundoff_multiplier: float
) -> None:
    terminal_status = _terminal_status(records, path)
    if terminal_status is None:
        return
    first, status = terminal_status
    stop = records[first].stop_time_s
    if stop is None or not 0.0 <= stop <= records[first].time_s:
        raise ValueError(f"{path}: terminal stop time is missing or outside its observed frame")
    if any(
        record.stop_time_s is not None and not _matches_time(record.stop_time_s, stop)
        for record in records[first:]
    ):
        raise ValueError(f"{path}: terminal stop time changes after the event")
    if status in (2, 3):
        _validate_terminal_tail(records[first:], path, roundoff_multiplier)


def _terminal_status(records: list[StateRecord], path: Path) -> tuple[int, int] | None:
    final = records[0].final_status
    if (
        any(record.final_status != final for record in records)
        or records[-1].current_status != final
    ):
        raise ValueError(f"{path}: final status is inconsistent within a particle history")
    terminal = [index for index, record in enumerate(records) if record.current_status != 1]
    if not terminal:
        if final != 1:
            raise ValueError(f"{path}: terminal final status has no terminal frame")
        return
    first = terminal[0]
    terminal_status = records[first].current_status
    if any(record.current_status != terminal_status for record in records[first:]):
        raise ValueError(f"{path}: terminal lifecycle changes after its first observation")
    return first, terminal_status


def _validate_terminal_tail(
    records: list[StateRecord], path: Path, roundoff_multiplier: float
) -> None:
    terminal_state = records[0].state
    for record in records[1:]:
        for index, label in ((0, "r"), (1, "z"), (4, "charge")):
            observed = record.state[index]
            expected = terminal_state[index]
            scale = max(abs(observed), abs(expected), 1.0e-300)
            limit = roundoff_multiplier * math.ulp(scale)
            if abs(observed - expected) > limit:
                raise ValueError(
                    f"{path}: held/stuck terminal {label} tail changes after the event"
                )


def _read_auxiliary_table(
    path: Path,
    protocol: Protocol,
    column_count: int,
    histories: list[list[StateRecord]],
) -> dict[int, list[tuple[float, ...]]]:
    rows = _wide_rows(path)
    if len(rows) != protocol.particle_count:
        raise ValueError(f"{path}: auxiliary particle-row count differs")
    width = column_count * len(protocol.output_times_s)
    state_by_particle = {records[0].particle_id: records for records in histories}
    result: dict[int, list[tuple[float, ...]]] = {}
    seen: set[int] = set()
    for row in rows:
        if len(row) != width:
            raise ValueError(f"{path}: auxiliary wide-table width differs")
        particle_id = _integer(row[0], "particle_id", path)
        if particle_id in seen or particle_id not in state_by_particle:
            raise ValueError(f"{path}: duplicate auxiliary particle ID")
        seen.add(particle_id)
        records: list[tuple[float, ...]] = []
        for frame, expected_time in enumerate(protocol.output_times_s):
            begin = frame * column_count
            record = tuple(row[begin : begin + column_count])
            if _integer(record[0], "particle_id", path) != particle_id:
                raise ValueError(f"{path}: auxiliary particle ID changes within a row")
            if not _matches_time(record[1], expected_time):
                raise ValueError(f"{path}: auxiliary output schedule differs")
            if state_by_particle[particle_id][frame].current_status == 1 and not all(
                math.isfinite(value) for value in record
            ):
                raise ValueError(f"{path}: active auxiliary record is nonfinite")
            records.append(record)
        result[particle_id] = records
    if seen != set(state_by_particle):
        raise ValueError(f"{path}: auxiliary particle IDs differ from the state table")
    return result


def _roundoff_limit(values: tuple[float, ...], multiplier: float) -> float:
    scale = max(*(abs(value) for value in values), 1.0e-300)
    return multiplier * len(values) * math.ulp(scale)


def _validate_force_consistency(
    path: Path,
    histories: list[list[StateRecord]],
    forces: dict[int, list[tuple[float, ...]]],
    roundoff_multiplier: float,
) -> dict[str, object]:
    maximum_force_residual = [0.0, 0.0]
    maximum_acceleration_residual = [0.0, 0.0]
    active_records = 0
    for state_records in histories:
        particle_id = state_records[0].particle_id
        for state, force in zip(state_records, forces[particle_id], strict=True):
            if state.current_status != 1:
                continue
            active_records += 1
            for component in range(2):
                contributions = tuple(force[index + component] for index in range(2, 16, 2))
                total = force[16 + component]
                force_residual = abs(total - sum(contributions))
                force_limit = _roundoff_limit((*contributions, total), roundoff_multiplier)
                if force_residual > force_limit:
                    raise ValueError(f"{path}: total force differs from its component sum")
                acceleration = force[18 + component]
                expected_acceleration = total / state.mass_kg
                acceleration_residual = abs(acceleration - expected_acceleration)
                acceleration_limit = _roundoff_limit(
                    (acceleration, expected_acceleration), roundoff_multiplier
                )
                if acceleration_residual > acceleration_limit:
                    raise ValueError(f"{path}: acceleration differs from total force / mass")
                maximum_force_residual[component] = max(
                    maximum_force_residual[component], force_residual
                )
                maximum_acceleration_residual[component] = max(
                    maximum_acceleration_residual[component], acceleration_residual
                )
    return {
        "status": "PASS",
        "active_records": active_records,
        "criterion": "residual <= roundoff_multiplier * term_count * ulp(max_abs_term)",
        "roundoff_multiplier": roundoff_multiplier,
        "maximum_force_sum_residual_N": maximum_force_residual,
        "maximum_acceleration_identity_residual_m_per_s2": maximum_acceleration_residual,
    }


def _read_release(path: Path, particle_count: int) -> dict[int, tuple[float, ...]]:
    required = (
        "particle_id",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number",
    )
    result: dict[int, tuple[float, ...]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"{path}: release probe columns are incomplete")
        for row in reader:
            particle_id = int(row["particle_id"])
            values = tuple(float(row[name]) for name in required[1:])
            if particle_id in result or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: duplicate or nonfinite release row")
            result[particle_id] = values
    if set(result) != set(range(1, particle_count + 1)):
        raise ValueError(f"{path}: release particle IDs differ")
    return result


def _initial_state_check(
    histories: list[list[StateRecord]],
    release: dict[int, tuple[float, ...]],
    multiplier: float,
) -> dict[str, object]:
    names = ("r_m", "z_m", "velocity_r_m_per_s", "velocity_z_m_per_s", "charge_number_e")
    differences = {name: [] for name in names}
    values = {name: [] for name in names}
    sampled_differences = {name: [] for name in names[2:]}
    for records in histories:
        first = records[0]
        expected = release[first.particle_id]
        for name, observed, target in zip(names, first.state, expected, strict=True):
            differences[name].append(abs(observed - target))
            values[name].extend((observed, target))
        for name, observed, target in zip(
            names[2:], first.sampled_release, expected[2:], strict=True
        ):
            sampled_differences[name].append(abs(observed - target))
    maxima = {name: max(component) for name, component in differences.items()}
    limits = {
        name: multiplier * math.ulp(max(max(abs(value) for value in component), 1.0e-300))
        for name, component in values.items()
    }
    sampled_maxima = {name: max(component) for name, component in sampled_differences.items()}
    if any(maxima[name] > limits[name] for name in names):
        raise ValueError("COMSOL t0 state differs from the common-P1 release probe")
    if any(sampled_maxima[name] > limits[name] for name in sampled_maxima):
        raise ValueError("COMSOL sampled release function differs from the release probe")
    return {
        "status": "PASS",
        "criterion": "component-global maximum difference <= multiplier * ulp(max_abs)",
        "ulp_multiplier": multiplier,
        "maximum_absolute_difference": maxima,
        "roundoff_limit": limits,
        "sampled_release_maximum_absolute_difference": sampled_maxima,
    }


def _write_normalized(
    root: Path,
    run_key: str,
    histories: list[list[StateRecord]],
) -> dict[str, object]:
    label = run_key.removeprefix("dt_")
    trajectory_path = root / f"reference_trajectory_{label}.csv"
    event_path = root / f"reference_events_{label}.csv"
    with trajectory_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(TRAJECTORY_COLUMNS)
        for frame in range(len(histories[0])):
            for records in histories:
                record = records[frame]
                state = record.state if record.current_status != 4 else ("",) * 5
                writer.writerow(
                    (
                        record.particle_id,
                        record.time_s,
                        *state,
                        STATUS_NAMES[record.current_status],
                        STATUS_NAMES[record.final_status],
                        "" if record.final_status == 1 else record.stop_time_s,
                    )
                )
    event_count = 0
    with event_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(EVENT_COLUMNS)
        for records in histories:
            terminal = next((record for record in records if record.current_status != 1), None)
            if terminal is None:
                continue
            event_count += 1
            observed = terminal.current_status != 4
            writer.writerow(
                (
                    terminal.particle_id,
                    0,
                    terminal.stop_time_s,
                    terminal.time_s,
                    STATUS_NAMES[terminal.current_status],
                    BOUNDARY_GROUPS[terminal.current_status],
                    ("terminal_state_directly_observed" if observed else "NOT_DIRECTLY_OBSERVED"),
                    terminal.state[0] if observed else "",
                    terminal.state[1] if observed else "",
                    terminal.state[4] if observed else "",
                )
            )
    final_counts = dict.fromkeys(STATUS_NAMES.values(), 0)
    for records in histories:
        final_counts[STATUS_NAMES[records[-1].current_status]] += 1
    return {
        "trajectory": trajectory_path.name,
        "trajectory_sha256": _sha256(trajectory_path),
        "event_ledger": event_path.name,
        "event_ledger_sha256": _sha256(event_path),
        "trajectory_rows": len(histories) * len(histories[0]),
        "event_rows": event_count,
        "final_fate_counts": final_counts,
    }


def _component_signature(raw: object) -> tuple[str, str, str, str, str, str]:
    record = _mapping(raw, "prepared component")
    keys = ("field", "component", "file", "function", "probe_column", "unit")
    values = tuple(record.get(key) for key in keys)
    if not all(isinstance(value, str) for value in values):
        raise ValueError("prepared component metadata is incomplete")
    return values  # type: ignore[return-value]


def _field_signature(raw: object) -> tuple[str, tuple[str, ...], str, str]:
    record = _mapping(raw, "prepared field")
    components = record.get("components")
    if not isinstance(components, list) or not all(
        isinstance(component, str) for component in components
    ):
        raise ValueError("prepared field components are incomplete")
    if record.get("association") != "node":
        raise ValueError("prepared fields must be node-associated")
    values = (
        record.get("field"),
        tuple(components),
        record.get("stored_basis"),
        record.get("unit"),
    )
    if not all(isinstance(value, str) for value in (values[0], values[2], values[3])):
        raise ValueError("prepared field metadata is incomplete")
    return values  # type: ignore[return-value]


def _validate_numeric_artifact(path: Path) -> int:
    values_checked = 0
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header:
                raise ValueError(f"prepared artifact has no CSV header: {path.name}")
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"prepared artifact CSV width changes: {path.name}")
                for token in row:
                    value = float(token)
                    if not math.isfinite(value):
                        raise ValueError(f"prepared artifact is nonfinite: {path.name}")
                    values_checked += 1
    else:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith("%"):
                    continue
                for token in line.split():
                    value = float(token)
                    if not math.isfinite(value):
                        raise ValueError(f"prepared artifact is nonfinite: {path.name}")
                    values_checked += 1
    if values_checked == 0:
        raise ValueError(f"prepared artifact has no numeric payload: {path.name}")
    return values_checked


def _require_exact_signatures(
    actual: list[Any], expected: tuple[Any, ...], declared_count: object, label: str
) -> None:
    if (
        declared_count != len(expected)
        or len(actual) != len(set(actual))
        or set(actual) != set(expected)
    ):
        raise ValueError(f"common-P1 {label} coverage differs from the formal contract")


def _validate_prepared_field_coverage(receipt: dict[str, Any]) -> None:
    components = receipt.get("components")
    fields = receipt.get("fields")
    if not isinstance(components, list) or not isinstance(fields, list):
        raise ValueError("common-P1 receipt field/component coverage is missing")
    component_signatures = [_component_signature(record) for record in components]
    field_signatures = [_field_signature(record) for record in fields]
    _require_exact_signatures(
        component_signatures,
        PREPARED_COMPONENT_SPECS,
        receipt.get("component_count"),
        "component",
    )
    _require_exact_signatures(
        field_signatures, PREPARED_FIELD_SPECS, receipt.get("field_count"), "field"
    )


def _validate_prepared_release(receipt: dict[str, Any]) -> None:
    release = _mapping(receipt.get("release"), "prepared release")
    functions = release.get("functions")
    if not isinstance(functions, list):
        raise ValueError("prepared release functions are missing")
    release_signatures: list[tuple[object, object, object, object]] = []
    for raw in functions:
        record = _mapping(raw, "prepared release function")
        if record.get("representation") != "rectangular_grid_point_table_r_z_value":
            raise ValueError("prepared release representation differs")
        release_signatures.append(
            (
                record.get("file"),
                record.get("function"),
                record.get("source_column"),
                record.get("unit"),
            )
        )
    _require_exact_signatures(
        release_signatures,
        PREPARED_RELEASE_SPECS,
        len(release_signatures),
        "release-function",
    )
    expected_release_metadata = {
        "grid": {"r_count": 41, "z_count": 7},
        "initial_state_columns": [
            "particle_id",
            "r_m",
            "z_m",
            "velocity_r_m_per_s",
            "velocity_z_m_per_s",
            "charge_number",
        ],
        "ordering": "particle_id_1_to_287_and_lexicographic_r_then_z",
        "probe_count": 287,
        "probe_field_sampling": "canonical_candidate_P1_sampler_at_realized_release_positions",
        "probe_file": "common_p1_release_probes.csv",
        "source_name": "particles",
    }
    if any(release.get(key) != value for key, value in expected_release_metadata.items()):
        raise ValueError("prepared release metadata differs")


def _validate_prepared_layout_and_candidate(receipt: dict[str, Any], workflow: str) -> None:
    layout = _mapping(receipt.get("layout"), "prepared P1 layout")
    expected_layout = {
        "type": "P1TriLayout",
        "name": "plasma",
        "node_count": 1987,
        "cell_count": 3779,
        "connectivity_width": 3,
        "connectivity_indexing_in_candidate": "zero_based",
        "connectivity_indexing_in_sectionwise_files": "one_based",
        "interpolation_contract": "first_order_triangular_sectionwise",
    }
    if any(layout.get(key) != value for key, value in expected_layout.items()):
        raise ValueError("prepared common-P1 layout differs")
    candidate = _mapping(receipt.get("candidate"), "prepared table candidate")
    expected_file_hash, expected_content_hash = EXPECTED_CANDIDATE_IDENTITIES[workflow]
    if (
        candidate.get("file_sha256") != expected_file_hash
        or candidate.get("content_hash") != expected_content_hash
    ):
        raise ValueError("prepared tables use the wrong formal candidate input identity")


def _validate_prepared_schema(receipt: dict[str, Any], workflow: str) -> set[str]:
    if receipt.get("schema_version") != 1 or receipt.get("tool_revision") != (
        "m3c1_full_physics_common_p1_tables_v1"
    ):
        raise ValueError("unsupported common-P1 prepared-table receipt")
    _validate_prepared_field_coverage(receipt)
    _validate_prepared_release(receipt)
    _validate_prepared_layout_and_candidate(receipt, workflow)
    return {
        *(spec[2] for spec in PREPARED_COMPONENT_SPECS),
        *(spec[0] for spec in PREPARED_RELEASE_SPECS),
        "common_p1_release_probes.csv",
    }


def _verify_prepared_artifacts(
    root: Path, artifacts: dict[str, Any]
) -> tuple[dict[str, tuple[str, int]], int]:
    verified: dict[str, tuple[str, int]] = {}
    numeric_values_checked = 0
    for relative, raw_record in artifacts.items():
        record = _mapping(raw_record, f"prepared artifact {relative}")
        path = (root / relative).resolve()
        if Path(relative).name != relative or not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"prepared artifact is missing or escapes root: {relative}")
        digest = _require_sha256(record.get("sha256"), f"prepared artifact {relative} hash")
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"prepared artifact size is invalid: {relative}")
        if _sha256(path) != digest or path.stat().st_size != size:
            raise ValueError(f"prepared artifact hash or size differs: {relative}")
        numeric_values_checked += _validate_numeric_artifact(path)
        verified[relative] = (digest, size)
    return verified, numeric_values_checked


def _prepared_validation_records(validation: dict[str, Any]) -> dict[str, tuple[str, int]]:
    validation_artifacts = validation.get("artifacts")
    if not isinstance(validation_artifacts, list):
        raise ValueError("prepared table validation artifact list is missing")
    validation_records: dict[str, tuple[str, int]] = {}
    for raw in validation_artifacts:
        record = _mapping(raw, "prepared validation artifact")
        relative = record.get("path")
        if not isinstance(relative, str) or relative in validation_records:
            raise ValueError("prepared validation artifact path is invalid")
        digest = _require_sha256(record.get("sha256"), f"validation artifact {relative} hash")
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"validation artifact size is invalid: {relative}")
        validation_records[relative] = (digest, size)
    return validation_records


def _validate_prepared_attestation(
    validation: dict[str, Any], receipt_hash: str, verified: dict[str, tuple[str, int]]
) -> None:
    if validation.get("schema_version") != 1 or validation.get("status") != "PASS":
        raise ValueError("prepared common-P1 table validation did not pass")
    if validation.get("receipt_sha256") != receipt_hash:
        raise ValueError("prepared common-P1 receipt hash differs from validation")
    validation_records = _prepared_validation_records(validation)
    total_size = sum(size for _, size in verified.values())
    if (
        validation_records != verified
        or validation.get("artifact_count") != len(verified)
        or validation.get("total_size_bytes") != total_size
        or _mapping(validation.get("pre_comsol"), "pre-COMSOL validation").get("status") != "PASS"
        or _mapping(validation.get("post_comsol"), "post-COMSOL validation").get("status") != "PASS"
    ):
        raise ValueError("prepared table validation does not attest the exact artifact set")


def _verify_prepared_tables(root: Path, workflow: str) -> dict[str, object]:
    receipt_path = root / "common_p1_table_receipt.json"
    validation_path = root / "prepared_table_validation.json"
    receipt = _mapping(json.loads(receipt_path.read_text(encoding="utf-8")), "table receipt")
    validation = _mapping(
        json.loads(validation_path.read_text(encoding="utf-8")), "table validation"
    )
    _require_finite_json(receipt, "table receipt")
    _require_finite_json(validation, "table validation")
    expected_artifacts = _validate_prepared_schema(receipt, workflow)
    artifacts = _mapping(receipt.get("artifacts"), "prepared table artifacts")
    if set(artifacts) != expected_artifacts:
        raise ValueError("prepared common-P1 artifact coverage differs")
    verified, numeric_values_checked = _verify_prepared_artifacts(root, artifacts)
    receipt_hash = _sha256(receipt_path)
    _validate_prepared_attestation(validation, receipt_hash, verified)
    candidate = _mapping(receipt.get("candidate"), "prepared table candidate")
    return {
        "status": "PASS",
        "receipt_sha256": receipt_hash,
        "validation_sha256": _sha256(validation_path),
        "artifact_count": len(artifacts),
        "finite_numeric_values_checked": numeric_values_checked,
        "candidate_input_sha256": candidate.get("file_sha256"),
        "canonical_content_hash": candidate.get("content_hash"),
    }


def _verify_artifact_ledger(root: Path) -> dict[str, object]:
    ledger_path = root / "artifact_hashes.csv"
    checked = 0
    seen: set[str] = set()
    with ledger_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != ("path", "sha256", "bytes"):
            raise ValueError("artifact_hashes.csv columns differ")
        for row in reader:
            relative = Path(row["path"])
            path = (root / relative).resolve()
            normalized = relative.as_posix()
            if (
                relative.is_absolute()
                or normalized in seen
                or not path.is_relative_to(root)
                or not path.is_file()
            ):
                raise ValueError(f"artifact ledger path is invalid: {relative}")
            digest = _require_sha256(row["sha256"], f"artifact ledger {relative} hash")
            if _sha256(path) != digest or path.stat().st_size != int(row["bytes"]):
                raise ValueError(f"artifact ledger hash or size differs: {relative}")
            seen.add(normalized)
            checked += 1
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != ledger_path
    }
    if seen != actual:
        raise ValueError("artifact ledger does not cover the exact reference artifact set")
    return {
        "status": "PASS",
        "path": ledger_path.name,
        "sha256": _sha256(ledger_path),
        "files": checked,
    }


def _load_comsol_contract() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "cases" / "m3c1_theory_100nm_30ms_comsol_v1.json"
    if _sha256(path) != EXPECTED_COMSOL_CONFIG_SHA256:
        raise ValueError("COMSOL adapter contract hash differs from the fixed formal contract")
    contract = _mapping(json.loads(path.read_text(encoding="utf-8")), "COMSOL contract")
    _require_finite_json(contract, "COMSOL contract")
    source = _mapping(contract.get("source_model"), "COMSOL source model contract")
    expected = {
        "sha256": EXPECTED_SOURCE_MPH_SHA256,
        "comsol_version": "6.4.0.429",
        "load_policy": "isolated_source_copy_loaded_with_ModelUtil.loadCopy",
        "save_policy": "never_save_model",
        "process_policy": "comsolbatch_-nosave_-np_1",
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("COMSOL source-model execution contract differs")
    return contract


def _read_run_spec(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line or "=" not in line:
            raise ValueError("executed run_spec.properties is malformed")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            raise ValueError("executed run_spec.properties has a duplicate or empty value")
        result[key] = value
    return result


def _verify_run_spec(
    values: dict[str, str],
    protocol: Protocol,
    workflow: str,
    contract: dict[str, Any],
) -> None:
    adapter = _mapping(
        _mapping(contract.get("workflows"), "COMSOL workflows").get(workflow),
        f"COMSOL workflow {workflow}",
    )
    expected_text = {
        "case_name": workflow,
        "physics_tag": adapter.get("physics_tag"),
        "background_study": adapter.get("background_study"),
        "background_solution": adapter.get("background_solution"),
        "source_dataset": adapter.get("source_dataset"),
        "particle_geometry": adapter.get("particle_geometry"),
        "position_dof_r": adapter.get("position_dofs", [None, None])[0],
        "position_dof_z": adapter.get("position_dofs", [None, None])[1],
        "charge_state": adapter.get("charge_state"),
        "run_keys": ",".join(RUN_KEYS),
        "common_config_sha256": EXPECTED_SHARED_CONFIG_SHA256,
        "comsol_config_sha256": EXPECTED_COMSOL_CONFIG_SHA256,
    }
    expected_keys = {
        *expected_text,
        "fixed_rk4_steps_s",
        "charge_lipschitz_s_inv",
        "maximum_dt_charge_lipschitz",
    }
    if set(values) != expected_keys or any(
        values.get(key) != value for key, value in expected_text.items()
    ):
        raise ValueError("executed run spec differs from the formal workflow contract")
    raw_steps = values["fixed_rk4_steps_s"].split(",")
    steps = tuple(_number(float(value), "executed fixed step") for value in raw_steps)
    if steps != protocol.steps_s_by_workflow[workflow]:
        raise ValueError("executed fixed-step sequence differs")
    if (
        float(values["charge_lipschitz_s_inv"])
        != (protocol.charge_lipschitz_s_inv_by_workflow[workflow])
        or float(values["maximum_dt_charge_lipschitz"]) != protocol.maximum_dt_charge_lipschitz
    ):
        raise ValueError("executed charge-step safety values differ")


def _expected_staged_java(run_spec: dict[str, str]) -> bytes:
    template_path = Path(__file__).resolve().parent / "comsol" / "RunM3C1Theory100nm30ms.java"
    if _sha256(template_path) != EXPECTED_JAVA_TEMPLATE_SHA256:
        raise ValueError("COMSOL Java template hash differs from the fixed formal template")
    text = template_path.read_bytes().decode("utf-8")
    steps = run_spec["fixed_rk4_steps_s"].split(",")
    replacements = {
        "__M3C1_CASE_NAME__": run_spec["case_name"],
        "__M3C1_PHYSICS_TAG__": run_spec["physics_tag"],
        "__M3C1_BACKGROUND_STUDY__": run_spec["background_study"],
        "__M3C1_BACKGROUND_SOLUTION__": run_spec["background_solution"],
        "__M3C1_SOURCE_DATASET__": run_spec["source_dataset"],
        "__M3C1_PARTICLE_GEOMETRY__": run_spec["particle_geometry"],
        "__M3C1_POSITION_DOF_R__": run_spec["position_dof_r"],
        "__M3C1_POSITION_DOF_Z__": run_spec["position_dof_z"],
        "__M3C1_CHARGE_STATE__": run_spec["charge_state"],
        "__M3C1_COMMON_CONFIG_SHA256__": run_spec["common_config_sha256"],
        "__M3C1_COMSOL_CONFIG_SHA256__": run_spec["comsol_config_sha256"],
        "__M3C1_STEP_COARSE_S__": steps[0],
        "__M3C1_STEP_MEDIUM_S__": steps[1],
        "__M3C1_STEP_FINE_S__": steps[2],
        "__M3C1_CHARGE_LIPSCHITZ_S_INV__": run_spec["charge_lipschitz_s_inv"],
        "__M3C1_MAXIMUM_DT_CHARGE_LIPSCHITZ__": run_spec["maximum_dt_charge_lipschitz"],
    }
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    if "__M3C1_" in text:
        raise ValueError("COMSOL staged Java still contains an unresolved contract token")
    return text.encode("utf-8")


def _validate_captured_process(receipt: dict[str, Any]) -> None:
    process = _mapping(receipt.get("active_process"), "captured active COMSOL process")
    executable = process.get("executable")
    if (
        not isinstance(executable, str)
        or not executable.replace("\\", "/")
        .lower()
        .endswith("/comsol64/multiphysics_copy1/bin/win64/comsolbatch.exe")
        or process.get("input_class") != "../RunM3C1Theory100nm30ms.class"
        or process.get("flags") != ["-nosave", "-np", "1"]
        or process.get("batch_log") != "../comsol_process.log"
    ):
        raise ValueError("captured active COMSOL command differs from the formal command")


def _verify_executed_artifacts(directory: Path, receipt: dict[str, Any]) -> None:
    artifacts = _mapping(receipt.get("artifacts"), "executed-code artifacts")
    if set(artifacts) != EXPECTED_EXECUTED_ARTIFACTS:
        raise ValueError("executed-code receipt artifact set differs")
    for filename, raw in artifacts.items():
        record = _mapping(raw, f"executed-code artifact {filename}")
        path = directory / filename
        digest = _require_sha256(record.get("sha256"), f"executed-code artifact {filename} hash")
        size = record.get("bytes")
        if (
            not path.is_file()
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or path.stat().st_size != size
            or _sha256(path) != digest
        ):
            raise ValueError(f"executed-code artifact differs: {filename}")
        if (
            filename == "run_m3c1_theory_100nm_30ms_reference.executed.ps1"
            and digest != EXPECTED_EXECUTED_RUNNER_SHA256
        ):
            raise ValueError("executed COMSOL runner differs from the fixed campaign runner")


def _verify_captured_execution(
    directory: Path, protocol: Protocol, workflow: str, contract: dict[str, Any]
) -> None:
    run_spec = _read_run_spec(directory / "run_spec.properties")
    _verify_run_spec(run_spec, protocol, workflow, contract)
    if (directory / "RunM3C1Theory100nm30ms.java").read_bytes() != _expected_staged_java(run_spec):
        raise ValueError("captured staged Java differs from the fixed template and run spec")
    status_lines = (
        (directory / "RunM3C1Theory100nm30ms.class.status")
        .read_text(encoding="utf-8-sig")
        .splitlines()
    )
    if not status_lines or status_lines[-1] != "Running":
        raise ValueError("compiled-class receipt was not captured while COMSOL was running")


def _verify_executed_code_receipt(
    root: Path, protocol: Protocol, workflow: str, contract: dict[str, Any]
) -> dict[str, object]:
    directory = root / "executed_code_receipt"
    receipt_path = directory / "receipt.json"
    receipt = _mapping(
        json.loads(receipt_path.read_text(encoding="utf-8")), "executed-code receipt"
    )
    _require_finite_json(receipt, "executed-code receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "CAPTURED_WHILE_PROCESS_ACTIVE"
        or receipt.get("workflow") != workflow
    ):
        raise ValueError("executed-code receipt identity differs")
    _validate_captured_process(receipt)
    _verify_executed_artifacts(directory, receipt)
    _verify_captured_execution(directory, protocol, workflow, contract)
    return {
        "status": "PASS",
        "path": receipt_path.relative_to(root).as_posix(),
        "sha256": _sha256(receipt_path),
        "staged_java_sha256": _sha256(directory / "RunM3C1Theory100nm30ms.java"),
        "compiled_class_sha256": _sha256(directory / "RunM3C1Theory100nm30ms.class"),
    }


def _verify_reference_run_records(
    root: Path, report: dict[str, Any], protocol: Protocol, workflow: str
) -> None:
    run_records = _mapping(report.get("runs"), "reference run records")
    if set(run_records) != set(RUN_KEYS):
        raise ValueError("reference run records must contain coarse, medium, and fine")
    for run_key, step_s in zip(RUN_KEYS, protocol.steps_s_by_workflow[workflow], strict=True):
        run = _mapping(run_records[run_key], f"reference run {run_key}")
        if _number(run.get("step_s"), f"reference run {run_key}.step_s") != step_s:
            raise ValueError(f"reference run {run_key} step differs from workflow configuration")
        lipschitz = protocol.charge_lipschitz_s_inv_by_workflow[workflow]
        expected_run = {
            "charge_lipschitz_s_inv": lipschitz,
            "dt_charge_lipschitz": step_s * lipschitz,
            "maximum_dt_charge_lipschitz": protocol.maximum_dt_charge_lipschitz,
            "step_count_30ms": round(protocol.output_times_s[-1] / step_s),
        }
        if any(run.get(key) != value for key, value in expected_run.items()):
            raise ValueError(f"reference run {run_key} safety receipt differs")
        tables = _mapping(run.get("raw_tables"), f"reference run {run_key} raw tables")
        if set(tables) != {"state_raw_wide.csv", "force_raw_wide.csv", "primitive_raw_wide.csv"}:
            raise ValueError(f"reference run {run_key} raw table set differs")
        for filename, raw_record in tables.items():
            table = _mapping(raw_record, f"reference run {run_key} {filename}")
            expected_path = f"{run_key}/{filename}"
            if table.get("path") != expected_path:
                raise ValueError(f"reference run {run_key} raw table path differs")
            path = root / expected_path
            digest = _require_sha256(
                table.get("sha256"), f"reference run {run_key} {filename} hash"
            )
            size = table.get("size_bytes")
            if (
                not path.is_file()
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or path.stat().st_size != size
                or _sha256(path) != digest
            ):
                raise ValueError(f"reference run {run_key} raw table identity differs")


def _verify_reference_identity(
    report: dict[str, Any], protocol: Protocol, workflow: str, prepared: dict[str, object]
) -> None:
    expected = {
        "schema_version": 1,
        "status": "COMPLETE",
        "tool_revision": COMSOL_RUN_REVISION,
        "config_sha256": EXPECTED_SHARED_CONFIG_SHA256,
        "comsol_config_sha256": EXPECTED_COMSOL_CONFIG_SHA256,
        "source_sha256_before": EXPECTED_SOURCE_MPH_SHA256,
        "source_sha256_after": EXPECTED_SOURCE_MPH_SHA256,
        "comsol_version": EXPECTED_COMSOL_VERSION,
        "java_source_sha256": EXPECTED_JAVA_TEMPLATE_SHA256,
        "field_representation": "canonical_exact_connectivity_p1",
        "primitive_source": "prepared_common_p1_tables",
        "workflow": workflow,
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        raise ValueError(f"reference run report differs at {', '.join(mismatches)}")
    if protocol.config_sha256 != EXPECTED_SHARED_CONFIG_SHA256:
        raise ValueError("normalizer protocol is not the fixed shared contract")
    process = _mapping(report.get("process"), "reference process receipt")
    if (
        process.get("load") != "ModelUtil.loadCopy"
        or process.get("flags") != ["-nosave", "-np", "1"]
        or process.get("model_saved") is not False
    ):
        raise ValueError("reference process did not use isolated loadCopy/-nosave/-np 1")
    if report.get("candidate_input_sha256") != prepared.get("candidate_input_sha256"):
        raise ValueError("reference report and common-P1 receipt use different candidate input")
    if report.get("candidate_input_content_hash") != prepared.get("canonical_content_hash"):
        raise ValueError("reference report and common-P1 receipt use different canonical content")


def _verify_reference_physics_and_scope(report: dict[str, Any], protocol: Protocol) -> None:
    physics = _mapping(report.get("physics"), "reference physics receipt")
    required_forces = {
        "electric",
        "relative_flow_ion_drag",
        "epstein_drag",
        "thermophoresis",
        "lift",
        "dielectrophoresis",
        "gravity_buoyancy",
    }
    contributions = physics.get("deterministic_contributions")
    if (
        physics.get("brownian_active") is not False
        or physics.get("saffman_active") is not False
        or physics.get("dynamic_charge_active") is not True
        or not isinstance(contributions, list)
        or set(contributions) != required_forces
    ):
        raise ValueError("reference physics receipt is not the formal deterministic composition")
    scope = _mapping(report.get("scope"), "reference scope")
    expected_scope = {
        "diameter_m": 1.0e-7,
        "particle_count": protocol.particle_count,
        "time_start_s": protocol.output_times_s[0],
        "time_end_s": protocol.output_times_s[-1],
        "output_count": len(protocol.output_times_s),
    }
    if any(scope.get(key) != value for key, value in expected_scope.items()):
        raise ValueError("reference scope differs from the formal 100 nm, 30 ms matrix")
    expected_boundaries = {
        "material": "stick",
        "gas_inlet_37": "freeze_hold",
        "pump_35": "disappear_escape",
        "axis_5": "coordinate_axis_not_material_event",
        "escape_hit_position": "NOT_DIRECTLY_OBSERVED_WHEN_NAN",
    }
    if report.get("boundary_semantics") != expected_boundaries:
        raise ValueError("reference boundary semantics differ")


def _verify_reference_receipts(
    report: dict[str, Any],
    protocol: Protocol,
    workflow: str,
    prepared: dict[str, object],
    contract: dict[str, Any],
) -> None:
    if report.get("claim_policy") != contract.get("claim_policy"):
        raise ValueError("reference claim policy differs from the COMSOL contract")
    prepared_record = _mapping(report.get("prepared_table"), "reference prepared-table receipt")
    if (
        prepared_record.get("status") != "PASS"
        or prepared_record.get("receipt_sha256") != prepared.get("receipt_sha256")
        or prepared_record.get("validation_path") != "prepared_table_validation.json"
        or prepared_record.get("artifact_count") != prepared.get("artifact_count")
    ):
        raise ValueError("reference prepared-table receipt differs")
    safety = _mapping(report.get("charge_step_safety"), "reference charge-step safety")
    if (
        safety.get("status") != "PASS"
        or safety.get("run_keys") != list(RUN_KEYS)
        or safety.get("charge_lipschitz_s_inv")
        != protocol.charge_lipschitz_s_inv_by_workflow[workflow]
        or safety.get("maximum_dt_charge_lipschitz") != protocol.maximum_dt_charge_lipschitz
        or safety.get("all_steps_admissible") is not True
    ):
        raise ValueError("reference charge-step safety receipt differs")


def _verify_reference_report(
    root: Path, protocol: Protocol, workflow: str, prepared: dict[str, object]
) -> dict[str, object]:
    path = root / "reference_run_report.json"
    report = _mapping(json.loads(path.read_text(encoding="utf-8")), "reference run report")
    _require_finite_json(report, "reference run report")
    contract = _load_comsol_contract()
    _verify_reference_identity(report, protocol, workflow, prepared)
    _verify_reference_run_records(root, report, protocol, workflow)
    _verify_reference_physics_and_scope(report, protocol)
    _verify_reference_receipts(report, protocol, workflow, prepared, contract)
    executed = _verify_executed_code_receipt(root, protocol, workflow, contract)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "source_sha256": report.get("source_sha256_before"),
        "executed_code": executed,
    }


def normalize_case(reference_root: Path, config_path: Path, workflow: str) -> dict[str, Any]:
    if workflow not in WORKFLOWS:
        raise ValueError("workflow must be caseA or caseP")
    root = reference_root.expanduser().resolve()
    protocol, _ = _load_protocol(config_path)
    output_path = root / "reference_normalization_report.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    prepared = _verify_prepared_tables(root, workflow)
    provenance = _verify_reference_report(root, protocol, workflow, prepared)
    ledger = _verify_artifact_ledger(root)
    release_path = root / "common_p1_release_probes.csv"
    release = _read_release(release_path, protocol.particle_count)
    runs: dict[str, object] = {}
    step_rows = zip(RUN_KEYS, protocol.steps_s_by_workflow[workflow], strict=True)
    lipschitz = protocol.charge_lipschitz_s_inv_by_workflow[workflow]
    for run_key, step_s in step_rows:
        directory = root / run_key
        state_path = directory / "state_raw_wide.csv"
        histories = _read_state_table(state_path, protocol)
        force_path = directory / "force_raw_wide.csv"
        forces = _read_auxiliary_table(force_path, protocol, FORCE_COLUMN_COUNT, histories)
        force_consistency = _validate_force_consistency(
            force_path, histories, forces, protocol.roundoff_multiplier
        )
        _read_auxiliary_table(
            directory / "primitive_raw_wide.csv",
            protocol,
            PRIMITIVE_COLUMN_COUNT,
            histories,
        )
        initial = _initial_state_check(histories, release, protocol.initial_ulp_multiplier)
        artifacts = _write_normalized(root, run_key, histories)
        runs[run_key] = {
            "step_s": step_s,
            "charge_lipschitz_s_inv": lipschitz,
            "dt_charge_lipschitz": step_s * lipschitz,
            "raw_state_sha256": _sha256(state_path),
            "raw_force_sha256": _sha256(directory / "force_raw_wide.csv"),
            "raw_primitive_sha256": _sha256(directory / "primitive_raw_wide.csv"),
            "force_consistency": force_consistency,
            "initial_state": initial,
            **artifacts,
        }
    report = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "COMPLETE",
        "workflow": workflow,
        "config": {"path": str(protocol.config_path), "sha256": protocol.config_sha256},
        "scope": {
            "particles": protocol.particle_count,
            "frames": len(protocol.output_times_s),
            "time_window_s": [protocol.output_times_s[0], protocol.output_times_s[-1]],
        },
        "step_sequence": [
            {
                "run_key": run_key,
                "step_s": step_s,
                "charge_lipschitz_s_inv": lipschitz,
                "dt_charge_lipschitz": step_s * lipschitz,
                "maximum_dt_charge_lipschitz": protocol.maximum_dt_charge_lipschitz,
            }
            for run_key, step_s in zip(
                RUN_KEYS, protocol.steps_s_by_workflow[workflow], strict=True
            )
        ],
        "provenance": provenance,
        "artifact_ledger": ledger,
        "common_p1_receipt": prepared,
        "release_probe_sha256": _sha256(release_path),
        "runs": runs,
        "escape_position_policy": "NOT_DIRECTLY_OBSERVED_never_reconstructed_or_gated",
        "terminal_velocity_policy": "CHARACTERIZED_NOT_CROSS_GATED",
    }
    _write_json_exclusive(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workflow", required=True, choices=WORKFLOWS)
    parser.add_argument("--reference-root", required=True, type=Path)
    arguments = parser.parse_args()
    normalize_case(arguments.reference_root, arguments.config, arguments.workflow)


if __name__ == "__main__":
    main()
