"""Normalize the M3-C0b Brownian-off deterministic COMSOL pilot.

This is an external V&V tool.  It deliberately owns neither solver physics nor
COMSOL model construction; it only validates and reshapes a completed export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Protocol, TextIO

TOOL_REVISION: Final = "m3c0b_deterministic_pilot_v6"
EXPECTED_PARTICLES = 287
EXPECTED_OUTPUT_TIMES = 46
EXPECTED_START_S = 0.0
EXPECTED_END_S = 4.5e-4
STEP_S: Final = {
    "dt_0p625us": 6.25e-7,
    "dt_0p3125us": 3.125e-7,
    "dt_0p15625us": 1.5625e-7,
}

COLUMNS: Final = (
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
    "electric_force_r_N",
    "electric_force_z_N",
    "ion_drag_force_r_N",
    "ion_drag_force_z_N",
    "epstein_force_r_N",
    "epstein_force_z_N",
    "thermophoretic_force_r_N",
    "thermophoretic_force_z_N",
    "lift_force_r_N",
    "lift_force_z_N",
    "dep_force_r_N",
    "dep_force_z_N",
    "gravity_buoyancy_force_r_N",
    "gravity_buoyancy_force_z_N",
    "total_force_r_N",
    "total_force_z_N",
    "acceleration_r_m_per_s2",
    "acceleration_z_m_per_s2",
    "gas_velocity_r_m_per_s",
    "gas_velocity_z_m_per_s",
    "gas_temperature_K",
    "gas_density_kg_per_m3",
    "gas_dynamic_viscosity_Pa_s",
    "gas_thermal_conductivity_W_per_mK",
    "gas_mean_free_path_m",
    "electric_field_r_V_per_m",
    "electric_field_z_V_per_m",
    "electric_field_squared_V2_per_m2",
    "gradient_E2_r_V2_per_m3",
    "gradient_E2_z_V2_per_m3",
    "electron_density_per_m3",
    "positive_ion_density_per_m3",
    "positive_ion_mass_kg",
    "ion_velocity_r_m_per_s",
    "ion_velocity_z_m_per_s",
    "ion_thermal_energy_eV_as_V",
    "particle_surface_potential_per_charge_V",
    "charging_current_scale_per_s",
    "screening_length_m",
    "ion_neutral_mean_free_path_m",
    "temperature_gradient_r_K_per_m",
    "temperature_gradient_z_K_per_m",
    "effective_heat_flux_r_W_per_m2",
    "effective_heat_flux_z_W_per_m2",
    "azimuthal_vorticity_per_s",
    "particle_diameter_m",
)

FORCE_COLUMNS: Final = COLUMNS[12:28]
STATE_RAW_COLUMNS: Final = COLUMNS[:12]
FORCE_RAW_COLUMNS: Final = ("particle_id", "time_s", *COLUMNS[12:26])
NEUTRAL_RAW_COLUMNS: Final = ("particle_id", "time_s", *COLUMNS[30:37], *COLUMNS[52:58])
ELECTRIC_RAW_COLUMNS: Final = ("particle_id", "time_s", *COLUMNS[37:42], COLUMNS[48])
PLASMA_RAW_COLUMNS: Final = ("particle_id", "time_s", *COLUMNS[42:48], *COLUMNS[49:52])
RAW_EXPORT_TABLES: Final = (
    "state_raw_wide.csv",
    "force_raw_wide.csv",
    "neutral_raw_wide.csv",
    "electric_raw_wide.csv",
    "plasma_raw_wide.csv",
)
RAW_COLUMN_GROUPS: Final = (
    STATE_RAW_COLUMNS,
    FORCE_RAW_COLUMNS,
    NEUTRAL_RAW_COLUMNS,
    ELECTRIC_RAW_COLUMNS,
    PLASMA_RAW_COLUMNS,
)
STATUS_NAMES: Final = {1: "active", 2: "frozen", 3: "stuck", 4: "escaped"}
RECEIPT_PREFIX: Final = "M3C0B|configuration|"
EXPECTED_DETERMINISTIC_CONTRIBUTIONS: Final = (
    "electric",
    "relative_flow_ion_drag",
    "epstein_drag",
    "waldmann_thermophoresis",
    "free_molecular_lift_sensitivity",
    "dielectrophoresis",
    "gravity_buoyancy",
)

type HistoryState = tuple[float, float, float, float, float, float, int, float]
type History = dict[tuple[int, int], HistoryState]


class CsvWriter(Protocol):
    def writerow(self, row: Any) -> object: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _valid_sha256(value: str) -> bool:
    try:
        return len(value) == 64 and len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _all_finite(values: tuple[float, ...]) -> bool:
    return all(map(math.isfinite, values))


def _validated_raw_export(payload: object, config_path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: raw_export mapping is required")
    if payload.get("revision") != 3 or payload.get("tables") != list(RAW_EXPORT_TABLES):
        raise ValueError(f"{config_path}: raw_export must select revision 3 and five tables")
    if payload.get("join_key") != ["particle_id", "time_s"]:
        raise ValueError(f"{config_path}: raw_export join_key must be particle_id/time_s")
    return {
        "revision": 3,
        "tables": list(RAW_EXPORT_TABLES),
        "join_key": ["particle_id", "time_s"],
    }


def _validated_numerics(payload: object, config_path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: revision 6 requires a numerics mapping")
    numerics = {
        "integrator": payload.get("integrator"),
        "integrator_order": payload.get("integrator_order"),
        "relative_tolerance": payload.get("relative_tolerance"),
        "wall_accuracy_order": payload.get("wall_accuracy_order"),
        "store_particle_status": payload.get("store_particle_status"),
        "store_extra": payload.get("store_extra"),
    }
    expected = {
        "integrator": "classical_rk4",
        "integrator_order": 4,
        "relative_tolerance": 1e-8,
        "wall_accuracy_order": 1,
        "store_particle_status": True,
        "store_extra": False,
    }
    if numerics != expected:
        raise ValueError(f"{config_path}: numerics do not match the M3-C0b protocol")
    return numerics


def _validated_acceptance(payload: object, config_path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: revision 6 requires an acceptance mapping")
    minimum_order = float(payload.get("minimum_observed_order", 0.0))
    position_limit = float(payload.get("maximum_fine_pair_position_displacement_relative_l2", 0.0))
    velocity_limit = float(payload.get("maximum_fine_pair_velocity_relative_l2", 0.0))
    charge_limit = float(payload.get("maximum_fine_pair_charge_relative_l2", 0.0))
    if (
        any(
            not math.isfinite(value) or value <= 0.0
            for value in (minimum_order, position_limit, velocity_limit, charge_limit)
        )
        or payload.get("require_all_records_active") is not True
    ):
        raise ValueError(f"{config_path}: acceptance thresholds must be positive and finite")
    return {
        "minimum_observed_order": minimum_order,
        "maximum_fine_pair_position_displacement_relative_l2": position_limit,
        "maximum_fine_pair_velocity_relative_l2": velocity_limit,
        "maximum_fine_pair_charge_relative_l2": charge_limit,
        "require_all_records_active": True,
    }


def _validated_physics(payload: dict[str, Any], config_path: Path) -> list[str]:
    expected_selections = {
        "brownian_active": False,
        "saffman_active": False,
        "dynamic_charge_active": True,
    }
    if any(payload.get(key) is not value for key, value in expected_selections.items()):
        raise ValueError(f"{config_path}: stochastic and charge selections do not match the pilot")
    contributions = payload.get("deterministic_contributions")
    if contributions != list(EXPECTED_DETERMINISTIC_CONTRIBUTIONS):
        raise ValueError(f"{config_path}: deterministic contributions do not match the pilot")
    return list(EXPECTED_DETERMINISTIC_CONTRIBUTIONS)


def _receipt_fields(log_path: Path, receipt_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in receipt_text.split("|")[2:]:
        if "=" not in token:
            raise ValueError(f"{log_path}: malformed receipt token {token!r}")
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ValueError(f"{log_path}: duplicate or empty receipt key {key!r}")
        fields[key] = value
    return fields


def _parse_configuration_receipts(
    log_path: Path, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    lines = [
        line.strip()
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if RECEIPT_PREFIX in line
    ]
    if len(lines) != len(STEP_S):
        raise ValueError(
            f"{log_path}: expected {len(STEP_S)} M3C0B configuration receipts, found {len(lines)}"
        )
    receipts: dict[str, dict[str, Any]] = {}
    for line in lines:
        receipt_text = line[line.index(RECEIPT_PREFIX) :]
        fields = _receipt_fields(log_path, receipt_text)
        name = next(
            (
                candidate
                for candidate, step_s in STEP_S.items()
                if math.isclose(
                    float(fields.get("step_s", "nan")), step_s, rel_tol=0.0, abs_tol=1e-18
                )
            ),
            None,
        )
        if name is None or name in receipts:
            raise ValueError(f"{log_path}: receipt has unknown or duplicate step_s")
        receipts[name] = _validate_configuration_receipt(
            log_path,
            receipt_text,
            fields,
            STEP_S[name],
            config.get("numerics"),
        )
    if receipts.keys() != STEP_S.keys():
        raise ValueError(f"{log_path}: configuration receipts do not cover all three steps")
    return receipts


def _validate_configuration_receipt(
    log_path: Path,
    receipt_text: str,
    fields: dict[str, str],
    expected_step_s: float,
    expected_numerics: dict[str, Any] | None,
) -> dict[str, Any]:

    required = {
        "step_s",
        "brownian_active",
        "saffman_active",
        "dynamic_charge_active",
        "physics",
        "study",
        "solution",
        "output_times",
        "particle_rows",
        "source_model",
        "model_saved",
        "store_particle_status",
        "store_extra",
        "wall_accuracy_order",
        "background_study",
        "background_solution",
        "deterministic_contributions",
    }
    if expected_numerics is not None:
        required.update({"integrator", "integrator_order", "relative_tolerance"})
    if missing := sorted(required - fields.keys()):
        raise ValueError(f"{log_path}: missing receipt keys: {', '.join(missing)}")
    if not math.isclose(float(fields["step_s"]), expected_step_s, rel_tol=0.0, abs_tol=1e-18):
        raise ValueError(f"{log_path}: step_s does not match its step directory")
    expected_values = {
        "brownian_active": "false",
        "saffman_active": "false",
        "dynamic_charge_active": "true",
        "physics": "fptas",
        "output_times": str(EXPECTED_OUTPUT_TIMES),
        "particle_rows": str(EXPECTED_PARTICLES),
        "source_model": "source_copy.mph",
        "model_saved": "false",
        "store_particle_status": "true",
        "store_extra": "false",
        "wall_accuracy_order": "1",
        "background_study": "stdASf",
        "background_solution": "sol26",
        "deterministic_contributions": ",".join(EXPECTED_DETERMINISTIC_CONTRIBUTIONS),
    }
    if expected_numerics is not None:
        expected_values.update(
            {
                "integrator": str(expected_numerics["integrator"]),
                "integrator_order": str(expected_numerics["integrator_order"]),
                "wall_accuracy_order": str(expected_numerics["wall_accuracy_order"]),
                "store_particle_status": str(expected_numerics["store_particle_status"]).lower(),
                "store_extra": str(expected_numerics["store_extra"]).lower(),
            }
        )
    for key, expected in expected_values.items():
        if fields[key].lower() != expected.lower():
            raise ValueError(f"{log_path}: expected {key}={expected}, found {fields[key]}")
    if expected_numerics is not None and not math.isclose(
        float(fields["relative_tolerance"]),
        float(expected_numerics["relative_tolerance"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"{log_path}: expected relative_tolerance="
            f"{expected_numerics['relative_tolerance']}, found {fields['relative_tolerance']}"
        )
    if not fields["study"] or not fields["solution"]:
        raise ValueError(f"{log_path}: study and solution must be nonempty")
    return {
        "tool_revision": TOOL_REVISION,
        "configuration": fields,
        "configuration_receipt": receipt_text,
        "process_log_sha256": _sha256(log_path),
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    source = payload.get("source_model")
    case = payload.get("case")
    physics = payload.get("physics")
    if not isinstance(source, dict) or not isinstance(case, dict) or not isinstance(physics, dict):
        raise ValueError(f"{config_path}: source_model, case, and physics mappings are required")
    source_hash = str(source.get("sha256", "")).lower()
    if not _valid_sha256(source_hash) or not source.get("relative_path"):
        raise ValueError(f"{config_path}: source model path and SHA-256 are required")
    expected_case = {
        "workflow": "caseA",
        "particle_count": EXPECTED_PARTICLES,
        "output_times": EXPECTED_OUTPUT_TIMES,
    }
    for key, expected in expected_case.items():
        if case.get(key) != expected:
            raise ValueError(f"{config_path}: expected case.{key}={expected}")
    if not math.isclose(float(case.get("diameter_m", 0.0)), 1e-7, rel_tol=0.0, abs_tol=1e-20):
        raise ValueError(f"{config_path}: pilot diameter must be 100 nm")
    if not math.isclose(float(case.get("time_end_s", 0.0)), EXPECTED_END_S, abs_tol=1e-15):
        raise ValueError(f"{config_path}: pilot end time does not match the normalizer")
    configured_steps = [float(value) for value in case.get("fixed_rk4_steps_s", [])]
    if configured_steps != list(STEP_S.values()):
        raise ValueError(f"{config_path}: fixed RK4 steps do not match the three-step pilot")
    deterministic_contributions = _validated_physics(physics, config_path)
    raw_export = _validated_raw_export(payload.get("raw_export"), config_path)
    evaluation_revision = int(payload.get("evaluation_revision", 0))
    if evaluation_revision != 6:
        raise ValueError(f"{config_path}: only M3-C0b evaluation revision 6 is active")
    numerics = _validated_numerics(payload.get("numerics"), config_path)
    acceptance = _validated_acceptance(payload.get("acceptance"), config_path)
    return {
        "sha256": _sha256(config_path),
        "evaluation_revision": evaluation_revision,
        "source_model_relative_path": source["relative_path"],
        "source_model_sha256": source_hash,
        "workflow": case["workflow"],
        "diameter_m": case["diameter_m"],
        "ion_drag_revision": case.get("ion_drag_revision"),
        "fixed_rk4_steps_s": configured_steps,
        "deterministic_contributions": deterministic_contributions,
        "numerics": numerics,
        "acceptance": acceptance,
        "raw_export": raw_export,
    }


def _lifecycle(status_code: int, source: Path) -> str:
    if status_code not in STATUS_NAMES:
        raise ValueError(f"{source}: unknown particle status code {status_code}")
    return STATUS_NAMES[status_code]


def _integer(value: float, name: str, source: Path) -> int:
    result = round(value)
    if abs(value - result) > 1e-9:
        raise ValueError(f"{source}: {name} is not integer-valued: {value}")
    return result


def _artifact_headers() -> dict[str, tuple[str, ...]]:
    return {
        "trajectory_reference.csv": (
            "particle_id",
            "time_s",
            "r_m",
            "z_m",
            "velocity_r_m_per_s",
            "velocity_z_m_per_s",
            "charge_number_e",
            "lifecycle",
        ),
        "force_reference.csv": ("particle_id", "time_s", *FORCE_COLUMNS),
        "rhs_reference.csv": (
            "particle_id",
            "time_s",
            "position_rate_r_m_per_s",
            "position_rate_z_m_per_s",
            "velocity_rate_r_m_per_s2",
            "velocity_rate_z_m_per_s2",
            "charge_rate_e_per_s",
        ),
        "event_observations.csv": (
            "particle_id",
            "event_ordinal",
            "event_time_s",
            "first_observed_time_s",
            "observed_r_m",
            "observed_z_m",
            "event_type",
            "observation_basis",
        ),
    }


def _open_artifact_writers(
    directory: Path,
) -> tuple[dict[str, TextIO], dict[str, CsvWriter]]:
    streams: dict[str, TextIO] = {}
    writers: dict[str, CsvWriter] = {}
    for name, header in _artifact_headers().items():
        stream = (directory / name).open("w", newline="", encoding="utf-8")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        streams[name] = stream
        writers[name] = writer
    return streams, writers


def _write_record(
    values: list[float],
    particle_id: int,
    frame: int,
    source: Path,
    writers: dict[str, CsvWriter],
    history: History,
) -> int:
    required_key_indices = (0, 1, 7, 8, 9)
    if not all(math.isfinite(values[index]) for index in required_key_indices):
        raise ValueError(f"{source}: nonfinite identity, time, or status value")
    record_id = _integer(values[0], "particle_id", source)
    if record_id != particle_id:
        raise ValueError(f"{source}: particle ID changes within a wide row")
    current_status = _integer(values[7], "current_status_code", source)
    _integer(values[8], "final_status_code", source)
    lifecycle = _lifecycle(current_status, source)
    if current_status != 4 and not all(math.isfinite(value) for value in values[2:30]):
        raise ValueError(
            f"{source}: nonfinite state or ODE RHS for particle {particle_id}, frame {frame}"
        )
    if current_status != 4 and (
        not math.isfinite(values[57]) or values[11] <= 0.0 or values[57] <= 0.0
    ):
        raise ValueError(f"{source}: nonpositive particle mass or diameter")

    writers["trajectory_reference.csv"].writerow((particle_id, *values[1:7], lifecycle))
    writers["force_reference.csv"].writerow((particle_id, values[1], *values[12:28]))
    writers["rhs_reference.csv"].writerow(
        (particle_id, values[1], values[4], values[5], values[28], values[29], values[10])
    )
    history[(particle_id, frame)] = (
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        current_status,
        values[9],
    )

    return current_status


def _normalize_particle_row(
    values: list[float],
    row_number: int,
    source: Path,
    writers: dict[str, CsvWriter],
    history: History,
    reference_times: list[float] | None,
) -> tuple[int, list[float], dict[str, Any]]:
    width = len(COLUMNS)
    expected_width = width * EXPECTED_OUTPUT_TIMES
    if len(values) != expected_width:
        raise ValueError(f"{source}: expected {expected_width} values, found {len(values)}")
    particle_id = _integer(values[0], "particle_id", source)
    times = [values[frame * width + 1] for frame in range(EXPECTED_OUTPUT_TIMES)]
    if reference_times is not None and any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-14)
        for actual, expected in zip(times, reference_times, strict=True)
    ):
        raise ValueError(f"{source}: time grid differs for particle {particle_id}")

    status_counts: dict[str, int] = {}
    first_event: tuple[int, list[float]] | None = None
    for frame in range(EXPECTED_OUTPUT_TIMES):
        record = values[frame * width : (frame + 1) * width]
        status = _write_record(record, particle_id, frame, source, writers, history)
        lifecycle = STATUS_NAMES[status]
        status_counts[lifecycle] = status_counts.get(lifecycle, 0) + 1
        if status != 1 and first_event is None:
            first_event = (frame, record)

    if first_event is not None:
        frame, record = first_event
        writers["event_observations.csv"].writerow(
            (
                particle_id,
                0,
                record[9],
                record[1],
                record[2],
                record[3],
                STATUS_NAMES[_integer(record[7], "current_status_code", source)],
                "first_saved_nonactive_state",
            )
        )
    return (
        particle_id,
        times,
        {
            "row_number": row_number,
            "status_counts": status_counts,
            "event_observed": first_event is not None,
        },
    )


def _validate_step_grid(
    raw: Path,
    row_count: int,
    particle_ids: set[int],
    reference_times: list[float] | None,
) -> list[float]:
    expected_ids = set(range(1, EXPECTED_PARTICLES + 1))
    if row_count != EXPECTED_PARTICLES or particle_ids != expected_ids:
        raise ValueError(
            f"{raw}: particle IDs must be exactly 1..{EXPECTED_PARTICLES}; "
            f"found {row_count} rows and {len(particle_ids)} unique IDs"
        )
    if reference_times is None:
        raise ValueError(f"{raw}: no particle rows")
    if any(later <= earlier for earlier, later in itertools.pairwise(reference_times)):
        raise ValueError(f"{raw}: output times are not strictly increasing")
    if not math.isclose(reference_times[0], EXPECTED_START_S, rel_tol=0.0, abs_tol=2e-14):
        raise ValueError(f"{raw}: unexpected initial output time")
    if not math.isclose(reference_times[-1], EXPECTED_END_S, rel_tol=0.0, abs_tol=2e-14):
        raise ValueError(f"{raw}: unexpected final output time")
    return reference_times


def _wide_rows(path: Path) -> Iterator[list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.startswith("%") or not line.strip():
                continue
            yield [float(value) for value in next(csv.reader([line]))]


def _require_wide_width(values: list[float], columns: tuple[str, ...], source: Path) -> None:
    expected = len(columns) * EXPECTED_OUTPUT_TIMES
    if len(values) != expected:
        raise ValueError(f"{source}: expected {expected} values, found {len(values)}")


def _frame_record(values: list[float], columns: tuple[str, ...], frame: int) -> list[float]:
    start = frame * len(columns)
    return values[start : start + len(columns)]


def _same_raw_key(records: tuple[list[float], ...]) -> bool:
    keys = tuple(record[:2] for record in records)
    if not all(math.isfinite(value) for key in keys for value in key):
        return False
    return all(key == keys[0] for key in keys[1:])


def _combine_raw_rows(
    rows: tuple[list[float], ...],
    sources: tuple[Path, ...],
) -> list[float]:
    for values, columns, source in zip(rows, RAW_COLUMN_GROUPS, sources, strict=True):
        _require_wide_width(values, columns, source)
    combined: list[float] = []
    for frame in range(EXPECTED_OUTPUT_TIMES):
        records = tuple(
            _frame_record(values, columns, frame)
            for values, columns in zip(rows, RAW_COLUMN_GROUPS, strict=True)
        )
        if not _same_raw_key(records):
            raise ValueError(
                f"{sources[0].parent}: raw table particle ID/time mismatch at frame {frame}"
            )
        state_record, force_record, neutral_record, electric_record, plasma_record = records
        total_r = math.fsum(force_record[index] for index in range(2, 16, 2))
        total_z = math.fsum(force_record[index] for index in range(3, 16, 2))
        mass = state_record[11]
        acceleration_r = total_r / mass if math.isfinite(mass) and mass != 0.0 else math.nan
        acceleration_z = total_z / mass if math.isfinite(mass) and mass != 0.0 else math.nan
        primitive_record = (
            *neutral_record[:9],
            *electric_record[2:7],
            *plasma_record[2:8],
            electric_record[7],
            *plasma_record[8:11],
            *neutral_record[9:15],
        )
        combined.extend(
            (
                *state_record,
                *force_record[2:],
                total_r,
                total_z,
                acceleration_r,
                acceleration_z,
                *primitive_record[2:],
            )
        )
    return combined


def iter_combined_records(directory: Path) -> Iterator[dict[str, float]]:
    """Yield validated long-form records from one M3-C0b raw step export.

    The M3-C1 frozen-state evaluator uses this narrow reader so the five-table
    join and column ordering continue to have one owner.  It does not change or
    normalize the source artifacts.
    """

    raw_paths = tuple(directory / name for name in RAW_EXPORT_TABLES)
    rows = itertools.zip_longest(*(_wide_rows(path) for path in raw_paths))
    for raw_rows in rows:
        state, force, neutral, electric, plasma = raw_rows
        if state is None or force is None or neutral is None or electric is None or plasma is None:
            raise ValueError(f"{directory}: raw tables have different particle row counts")
        combined = _combine_raw_rows(
            (state, force, neutral, electric, plasma),
            raw_paths,
        )
        for frame in range(EXPECTED_OUTPUT_TIMES):
            values = _frame_record(combined, COLUMNS, frame)
            yield dict(zip(COLUMNS, values, strict=True))


def _normalize_step(
    directory: Path,
    expected_step_s: float,
    receipt: dict[str, Any],
    process_log: Path,
) -> tuple[dict[str, Any], History]:
    raw_paths = tuple(directory / name for name in RAW_EXPORT_TABLES)
    raw = raw_paths[0]
    _write_json(directory / "run_receipt.json", receipt)
    streams, writers = _open_artifact_writers(directory)
    history: History = {}
    particle_ids: set[int] = set()
    reference_times: list[float] | None = None
    status_counts: dict[str, int] = {}
    event_count = 0
    row_count = 0
    try:
        rows = itertools.zip_longest(*(_wide_rows(path) for path in raw_paths))
        for raw_rows in rows:
            state, force, neutral, electric, plasma = raw_rows
            if (
                state is None
                or force is None
                or neutral is None
                or electric is None
                or plasma is None
            ):
                raise ValueError(f"{directory}: raw tables have different particle row counts")
            row_count += 1
            values = _combine_raw_rows((state, force, neutral, electric, plasma), raw_paths)
            particle_id, times, summary = _normalize_particle_row(
                values, row_count, raw, writers, history, reference_times
            )
            if particle_id in particle_ids:
                raise ValueError(f"{raw}: duplicate particle ID {particle_id}")
            particle_ids.add(particle_id)
            if reference_times is None:
                reference_times = times
            for name, count in summary["status_counts"].items():
                status_counts[name] = status_counts.get(name, 0) + count
            event_count += int(summary["event_observed"])
    finally:
        for stream in streams.values():
            stream.close()

    reference_times = _validate_step_grid(raw, row_count, particle_ids, reference_times)

    artifacts = {
        name: {"path": name, "sha256": _sha256(directory / name)} for name in _artifact_headers()
    }
    summary = {
        "internal_step_s": expected_step_s,
        "particles": EXPECTED_PARTICLES,
        "output_times": EXPECTED_OUTPUT_TIMES,
        "records": EXPECTED_PARTICLES * EXPECTED_OUTPUT_TIMES,
        "time_window_s": [reference_times[0], reference_times[-1]],
        "status_record_counts": status_counts,
        "particles_with_observed_nonactive_state": event_count,
        "brownian_active": False,
        "raw": {
            path.stem.removesuffix("_raw_wide"): {"path": path.name, "sha256": _sha256(path)}
            for path in raw_paths
        },
        "process_log": {"path": "../comsol_process.log", "sha256": _sha256(process_log)},
        "receipt": {"path": "run_receipt.json", "sha256": _sha256(directory / "run_receipt.json")},
        "artifacts": artifacts,
    }
    _write_json(directory / "step_summary.json", summary)
    return summary, history


def _safe_relative_l2(numerator_sq: float, denominator_sq: float) -> float | None:
    if denominator_sq == 0.0:
        return 0.0 if numerator_sq == 0.0 else None
    return math.sqrt(numerator_sq / denominator_sq)


def _rms(squared_errors: list[float]) -> float | None:
    if not squared_errors:
        return None
    return math.sqrt(math.fsum(squared_errors) / len(squared_errors))


def _maximum_error(squared_errors: list[float]) -> float | None:
    return math.sqrt(max(squared_errors)) if squared_errors else None


def _first_nonactive_events(history: History) -> dict[int, tuple[int, float]]:
    events: dict[int, tuple[int, float]] = {}
    for (particle_id, _frame), state in sorted(history.items()):
        status = state[6]
        if status != 1 and particle_id not in events:
            events[particle_id] = (status, state[7])
    return events


def _event_type_counts(events: dict[int, tuple[int, float]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for status, _event_time in events.values():
        name = STATUS_NAMES[status]
        counts[name] = counts.get(name, 0) + 1
    return counts


def _compare_first_events(coarse: History, fine: History) -> dict[str, Any]:
    coarse_events = _first_nonactive_events(coarse)
    fine_events = _first_nonactive_events(fine)
    coarse_ids = set(coarse_events)
    fine_ids = set(fine_events)
    both = sorted(coarse_ids & fine_ids)
    same_type = [
        particle_id
        for particle_id in both
        if coarse_events[particle_id][0] == fine_events[particle_id][0]
    ]
    time_error_sq = [
        (coarse_events[particle_id][1] - fine_events[particle_id][1]) ** 2
        for particle_id in same_type
        if math.isfinite(coarse_events[particle_id][1])
        and math.isfinite(fine_events[particle_id][1])
    ]
    missing_in_coarse = fine_ids - coarse_ids
    missing_in_fine = coarse_ids - fine_ids
    return {
        "coarse_event_particles": len(coarse_events),
        "fine_event_particles": len(fine_events),
        "coarse_event_type_counts": _event_type_counts(coarse_events),
        "fine_event_type_counts": _event_type_counts(fine_events),
        "both_event_count": len(both),
        "type_agreement_fraction": len(same_type) / len(both) if both else None,
        "same_type_finite_time_pair_count": len(time_error_sq),
        "event_time_rms_s": _rms(time_error_sq),
        "event_time_max_s": _maximum_error(time_error_sq),
        "missing_in_coarse_count": len(missing_in_coarse),
        "missing_in_fine_count": len(missing_in_fine),
        "missing_on_one_side_count": len(missing_in_coarse | missing_in_fine),
    }


def _compare_histories(coarse: History, fine: History) -> dict[str, Any]:
    if coarse.keys() != fine.keys():
        raise ValueError("step histories do not have identical particle/frame keys")
    position_sq: list[float] = []
    velocity_sq: list[float] = []
    charge_sq: list[float] = []
    position_displacement_ref_sq = 0.0
    velocity_ref_sq = 0.0
    charge_ref_sq = 0.0
    status_matches = 0
    common_active = 0
    common_active_position_sq: list[float] = []
    common_active_velocity_sq: list[float] = []
    fine_initial_position = {
        particle_id: (state[1], state[2])
        for (particle_id, frame), state in fine.items()
        if frame == 0
    }
    for key in sorted(coarse):
        left = coarse[key]
        right = fine[key]
        if not math.isclose(left[0], right[0], rel_tol=0.0, abs_tol=2e-14):
            raise ValueError(f"step histories have different output time at {key}")
        status_matches += int(left[6] == right[6])
        if _all_finite((*left[1:3], *right[1:3])):
            dp = math.hypot(left[1] - right[1], left[2] - right[2])
            position_sq.append(dp * dp)
            initial_r, initial_z = fine_initial_position[key[0]]
            position_displacement_ref_sq += (right[1] - initial_r) ** 2 + (
                right[2] - initial_z
            ) ** 2
        if _all_finite((*left[3:5], *right[3:5])):
            dv = math.hypot(left[3] - right[3], left[4] - right[4])
            velocity_sq.append(dv * dv)
            velocity_ref_sq += right[3] ** 2 + right[4] ** 2
        if math.isfinite(left[5]) and math.isfinite(right[5]):
            dq = abs(left[5] - right[5])
            charge_sq.append(dq * dq)
            charge_ref_sq += right[5] ** 2
        if left[6] == 1 and right[6] == 1:
            common_active += 1
            dp = math.hypot(left[1] - right[1], left[2] - right[2])
            dv = math.hypot(left[3] - right[3], left[4] - right[4])
            common_active_position_sq.append(dp * dp)
            common_active_velocity_sq.append(dv * dv)
    count = len(coarse)
    return {
        "records": count,
        "position_finite_records": len(position_sq),
        "position_rms_m": _rms(position_sq),
        "position_max_m": _maximum_error(position_sq),
        "position_displacement_relative_l2": (
            _safe_relative_l2(math.fsum(position_sq), position_displacement_ref_sq)
            if position_sq
            else None
        ),
        "velocity_finite_records": len(velocity_sq),
        "velocity_rms_m_per_s": _rms(velocity_sq),
        "velocity_max_m_per_s": _maximum_error(velocity_sq),
        "velocity_relative_l2": (
            _safe_relative_l2(math.fsum(velocity_sq), velocity_ref_sq) if velocity_sq else None
        ),
        "charge_finite_records": len(charge_sq),
        "charge_rms_e": _rms(charge_sq),
        "charge_max_e": _maximum_error(charge_sq),
        "charge_relative_l2": (
            _safe_relative_l2(math.fsum(charge_sq), charge_ref_sq) if charge_sq else None
        ),
        "status_agreement_fraction": status_matches / count,
        "common_active_records": common_active,
        "common_active_position_rms_m": _rms(common_active_position_sq),
        "common_active_velocity_rms_m_per_s": _rms(common_active_velocity_sq),
        "first_nonactive_event_comparison": _compare_first_events(coarse, fine),
    }


def _observed_order(coarse_fine: float | None, fine_finer: float | None) -> float | None:
    if coarse_fine is None or fine_finer is None or coarse_fine <= 0.0 or fine_finer <= 0.0:
        return None
    return math.log(coarse_fine / fine_finer, 2.0)


def _self_convergence(histories: dict[str, History], acceptance: dict[str, Any]) -> dict[str, Any]:
    coarse_middle = _compare_histories(histories["dt_0p625us"], histories["dt_0p3125us"])
    middle_fine = _compare_histories(histories["dt_0p3125us"], histories["dt_0p15625us"])
    observed_orders = {
        name: _observed_order(coarse_middle[key], middle_fine[key])
        for name, key in (
            ("position_all_records", "position_rms_m"),
            ("velocity_all_records", "velocity_rms_m_per_s"),
            ("charge_all_records", "charge_rms_e"),
            ("position_common_active", "common_active_position_rms_m"),
            ("velocity_common_active", "common_active_velocity_rms_m_per_s"),
        )
    }
    all_active = all(state[6] == 1 for history in histories.values() for state in history.values())
    gates = {
        "all_records_active": all_active,
        "position_order": observed_orders["position_all_records"] is not None
        and observed_orders["position_all_records"] >= acceptance["minimum_observed_order"],
        "velocity_order": observed_orders["velocity_all_records"] is not None
        and observed_orders["velocity_all_records"] >= acceptance["minimum_observed_order"],
        "charge_order": observed_orders["charge_all_records"] is not None
        and observed_orders["charge_all_records"] >= acceptance["minimum_observed_order"],
        "position_fine_pair": middle_fine["position_displacement_relative_l2"] is not None
        and middle_fine["position_displacement_relative_l2"]
        <= acceptance["maximum_fine_pair_position_displacement_relative_l2"],
        "velocity_fine_pair": middle_fine["velocity_relative_l2"] is not None
        and middle_fine["velocity_relative_l2"]
        <= acceptance["maximum_fine_pair_velocity_relative_l2"],
        "charge_fine_pair": middle_fine["charge_relative_l2"] is not None
        and middle_fine["charge_relative_l2"] <= acceptance["maximum_fine_pair_charge_relative_l2"],
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "pre_event_fixed_step_admissibility",
        "acceptance": acceptance,
        "gates": gates,
        "dt_0p625us_vs_0p3125us": coarse_middle,
        "dt_0p3125us_vs_0p15625us": middle_fine,
        "observed_order_base_2": observed_orders,
    }


def normalize(output_directory: Path, config_path: Path) -> dict[str, Any]:
    root = output_directory.resolve()
    config = _load_config(config_path.resolve())
    process_log = root / "comsol_process.log"
    receipts = _parse_configuration_receipts(process_log, config)
    summaries: dict[str, Any] = {}
    histories: dict[str, History] = {}
    for name, step_s in STEP_S.items():
        summary, history = _normalize_step(root / name, step_s, receipts[name], process_log)
        summaries[name] = summary
        histories[name] = history
    convergence = _self_convergence(histories, config["acceptance"])
    _write_json(root / "comsol_self_convergence.json", convergence)
    manifest = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "classification": "external_comsol_deterministic_reference_pilot",
        "case": "formal_iondrag_theory_consistent/caseA_100nm",
        "coordinate_system": "axisymmetric_rz_no_swirl",
        "brownian_active": False,
        "saffman_active": False,
        "dynamic_charge_active": True,
        "reference_scope": "full_deterministic_physics_with_brownian_disabled",
        "claim_policy": "pre_event_step_admissibility_only_not_solver_agreement",
        "configuration": config,
        "runs": summaries,
        "self_convergence": convergence,
    }
    _write_json(root / "reference_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize an M3-C0b three-step COMSOL deterministic pilot export."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    normalize(arguments.output_directory, arguments.config)


if __name__ == "__main__":
    main()
