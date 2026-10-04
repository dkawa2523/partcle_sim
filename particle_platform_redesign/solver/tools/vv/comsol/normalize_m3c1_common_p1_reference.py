"""Normalize the M3-C1 full-physics common-P1 COMSOL reference.

The tool is external V&V only. It validates the fixed Case-A 100 nm export,
checks the realized source state against the candidate table at t=0, and emits
deterministic long-form state, force, RHS, primitive, and event artifacts. It
does not select tolerances for solver agreement and does not modify the solver.
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

TOOL_REVISION: Final = "m3c1_full_physics_common_p1_reference_v1"
EXPECTED_PARTICLES = 287
EXPECTED_FRAMES = 46
EXPECTED_START_S = 0.0
EXPECTED_END_S = 4.5e-4
OUTPUT_INTERVAL_S = 1.0e-5
STEP_S: dict[str, float] = {
    "dt_0p625us": 6.25e-7,
    "dt_0p3125us": 3.125e-7,
    "dt_0p15625us": 1.5625e-7,
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
    "candidate_initial_velocity_r_m_per_s",
    "candidate_initial_velocity_z_m_per_s",
    "candidate_initial_charge_number_e",
)
FORCE_COLUMNS: Final = (
    "particle_id",
    "time_s",
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
)
PRIMITIVE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "gas_density_kg_per_m3",
    "gas_dynamic_viscosity_Pa_s",
    "gas_temperature_K",
    "gas_mean_free_path_m",
    "electron_density_per_m3",
    "positive_ion_density_per_m3",
    "electron_thermal_energy_eV_as_V",
    "positive_ion_thermal_energy_eV_as_V",
    "effective_positive_ion_mass_kg",
    "screening_length_m",
    "ion_neutral_mean_free_path_m",
    "azimuthal_vorticity_per_s",
    "gas_velocity_r_m_per_s",
    "gas_velocity_z_m_per_s",
    "electric_field_r_V_per_m",
    "electric_field_z_V_per_m",
    "ion_velocity_r_m_per_s",
    "ion_velocity_z_m_per_s",
    "gradient_E2_r_V2_per_m3",
    "gradient_E2_z_V2_per_m3",
    "gas_translational_heat_flux_r_W_per_m2",
    "gas_translational_heat_flux_z_W_per_m2",
)
# The left name is the COMSOL raw-export column above; the right name is the
# corresponding candidate probe column emitted by prepare_m3c1_common_p1_tables.
# Keeping the mapping explicit avoids treating column position as a scientific
# contract when the two artifacts use slightly different descriptive names.
PRIMITIVE_PROBE_COLUMNS: Final = (
    ("gas_density_kg_per_m3", "gas_density_kg_per_m3"),
    ("gas_dynamic_viscosity_Pa_s", "gas_dynamic_viscosity_Pa_s"),
    ("gas_temperature_K", "gas_temperature_K"),
    ("gas_mean_free_path_m", "gas_mean_free_path_m"),
    ("electron_density_per_m3", "electron_density_per_m3"),
    ("positive_ion_density_per_m3", "positive_ion_density_per_m3"),
    ("electron_thermal_energy_eV_as_V", "electron_thermal_voltage_V"),
    ("positive_ion_thermal_energy_eV_as_V", "positive_ion_thermal_voltage_V"),
    ("effective_positive_ion_mass_kg", "effective_positive_ion_mass_kg"),
    ("screening_length_m", "screening_length_m"),
    ("ion_neutral_mean_free_path_m", "ion_neutral_mean_free_path_m"),
    ("azimuthal_vorticity_per_s", "azimuthal_gas_vorticity_per_s"),
    ("gas_velocity_r_m_per_s", "gas_velocity_r_m_per_s"),
    ("gas_velocity_z_m_per_s", "gas_velocity_z_m_per_s"),
    ("electric_field_r_V_per_m", "electric_field_r_V_per_m"),
    ("electric_field_z_V_per_m", "electric_field_z_V_per_m"),
    ("ion_velocity_r_m_per_s", "positive_ion_velocity_r_m_per_s"),
    ("ion_velocity_z_m_per_s", "positive_ion_velocity_z_m_per_s"),
    ("gradient_E2_r_V2_per_m3", "gradient_mean_e_squared_r_V2_per_m3"),
    ("gradient_E2_z_V2_per_m3", "gradient_mean_e_squared_z_V2_per_m3"),
    ("gas_translational_heat_flux_r_W_per_m2", "gas_translational_heat_flux_r_W_per_m2"),
    ("gas_translational_heat_flux_z_W_per_m2", "gas_translational_heat_flux_z_W_per_m2"),
)
RAW_TABLES: Final = {
    "state_raw_wide.csv": STATE_COLUMNS,
    "force_raw_wide.csv": FORCE_COLUMNS,
    "primitive_raw_wide.csv": PRIMITIVE_COLUMNS,
}
STATUS_NAMES: Final = {1: "active", 2: "frozen", 3: "stuck", 4: "escaped"}
RECEIPT_PREFIX: Final = "M3C1_COMMON_P1|configuration|"
DETERMINISTIC_CONTRIBUTIONS: Final = (
    "electric,relative_flow_ion_drag,epstein_drag,waldmann_heat_flux_thermophoresis,"
    "free_molecular_lift_sensitivity,dielectrophoresis,gravity_buoyancy"
)


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


def _integer(value: float, name: str, source: Path) -> int:
    result = round(value)
    if not math.isfinite(value) or abs(value - result) > 1e-9:
        raise ValueError(f"{source}: {name} is not integer-valued: {value}")
    return result


def _wide_rows(path: Path) -> Iterator[list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.startswith("%") or not line.strip():
                continue
            yield [float(value) for value in next(csv.reader([line]))]


def _split_particle_row(
    values: list[float], path: Path, columns: tuple[str, ...]
) -> tuple[int, list[float], list[list[float]]]:
    expected_width = len(columns) * EXPECTED_FRAMES
    if len(values) != expected_width:
        raise ValueError(f"{path}: expected {expected_width} values, found {len(values)}")
    particle_id = _integer(values[0], "particle_id", path)
    times: list[float] = []
    records: list[list[float]] = []
    for frame in range(EXPECTED_FRAMES):
        start = frame * len(columns)
        record = values[start : start + len(columns)]
        if _integer(record[0], "particle_id", path) != particle_id:
            raise ValueError(f"{path}: particle ID changes within wide row {particle_id}")
        if not all(math.isfinite(value) for value in record):
            raise ValueError(f"{path}: nonfinite value for particle {particle_id}, frame {frame}")
        times.append(record[1])
        records.append(record)
    return particle_id, times, records


def _validate_wide_grid(
    path: Path,
    row_count: int,
    particle_ids: set[int],
    reference_times: list[float] | None,
) -> None:
    expected_ids = set(range(1, EXPECTED_PARTICLES + 1))
    if row_count != EXPECTED_PARTICLES or particle_ids != expected_ids:
        raise ValueError(
            f"{path}: particle IDs must be exactly 1..{EXPECTED_PARTICLES}; "
            f"found {row_count} rows and {len(particle_ids)} unique IDs"
        )
    if reference_times is None:
        raise ValueError(f"{path}: no particle records")
    if any(later <= earlier for earlier, later in itertools.pairwise(reference_times)):
        raise ValueError(f"{path}: output times are not strictly increasing")
    expected_times = [
        EXPECTED_START_S + frame * OUTPUT_INTERVAL_S for frame in range(EXPECTED_FRAMES)
    ]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-14)
        for actual, expected in zip(reference_times, expected_times, strict=True)
    ):
        raise ValueError(f"{path}: output times do not match the locked 0..450 us schedule")
    if not math.isclose(reference_times[-1], EXPECTED_END_S, rel_tol=0.0, abs_tol=2e-14):
        raise ValueError(f"{path}: unexpected final output time")


def _read_wide(path: Path, columns: tuple[str, ...]) -> dict[tuple[int, int], list[float]]:
    records: dict[tuple[int, int], list[float]] = {}
    particle_ids: set[int] = set()
    reference_times: list[float] | None = None
    row_count = 0
    for values in _wide_rows(path):
        row_count += 1
        particle_id, times, particle_records = _split_particle_row(values, path, columns)
        if particle_id in particle_ids:
            raise ValueError(f"{path}: duplicate particle row {particle_id}")
        particle_ids.add(particle_id)
        for frame, record in enumerate(particle_records):
            records[(particle_id, frame)] = record
        if reference_times is None:
            reference_times = times
        elif any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=2e-14)
            for actual, expected in zip(times, reference_times, strict=True)
        ):
            raise ValueError(f"{path}: output time grid differs for particle {particle_id}")

    _validate_wide_grid(path, row_count, particle_ids, reference_times)
    return records


def _read_release(
    path: Path,
) -> tuple[
    dict[int, tuple[float, float, float, float, float]],
    dict[int, tuple[float, ...]],
]:
    required = {
        "particle_id",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number",
        *(probe_column for _, probe_column in PRIMITIVE_PROBE_COLUMNS),
    }
    states: dict[int, tuple[float, float, float, float, float]] = {}
    primitives: dict[int, tuple[float, ...]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: release probe header is incomplete")
        for row in reader:
            particle_id = int(row["particle_id"])
            values = tuple(
                float(row[name])
                for name in (
                    "r_m",
                    "z_m",
                    "velocity_r_m_per_s",
                    "velocity_z_m_per_s",
                    "charge_number",
                )
            )
            primitive_values = tuple(
                float(row[probe_column]) for _, probe_column in PRIMITIVE_PROBE_COLUMNS
            )
            if (
                particle_id in states
                or not all(math.isfinite(value) for value in values)
                or not all(math.isfinite(value) for value in primitive_values)
            ):
                raise ValueError(f"{path}: duplicate or nonfinite release row {particle_id}")
            states[particle_id] = values  # type: ignore[assignment]
            primitives[particle_id] = primitive_values
    if set(states) != set(range(1, EXPECTED_PARTICLES + 1)):
        raise ValueError(f"{path}: release particle IDs must be exactly 1..{EXPECTED_PARTICLES}")
    return states, primitives


def _roundoff_limit(values: list[float], multiplier: float) -> float:
    scale = max((abs(value) for value in values), default=0.0)
    return multiplier * math.ulp(max(scale, 1.0e-300))


def _validate_initial_state(
    state: dict[tuple[int, int], list[float]],
    release: dict[int, tuple[float, float, float, float, float]],
    multiplier: float,
    source: Path,
) -> dict[str, Any]:
    differences = {"r_m": [], "z_m": [], "vr_m_s": [], "vz_m_s": [], "charge_e": []}
    values_by_quantity = {name: [] for name in differences}
    function_differences = {"vr_m_s": [], "vz_m_s": [], "charge_e": []}
    for particle_id, expected in release.items():
        actual = state[(particle_id, 0)]
        observed = (actual[2], actual[3], actual[4], actual[5], actual[6])
        for name, left, right in zip(differences, observed, expected, strict=True):
            differences[name].append(abs(left - right))
            values_by_quantity[name].extend((left, right))
        function_differences["vr_m_s"].append(abs(actual[12] - expected[2]))
        function_differences["vz_m_s"].append(abs(actual[13] - expected[3]))
        function_differences["charge_e"].append(abs(actual[14] - expected[4]))

    limits = {name: _roundoff_limit(values_by_quantity[name], multiplier) for name in differences}
    maxima = {name: max(values, default=0.0) for name, values in differences.items()}
    function_maxima = {
        name: max(values, default=0.0) for name, values in function_differences.items()
    }
    function_limits = {
        "vr_m_s": limits["vr_m_s"],
        "vz_m_s": limits["vz_m_s"],
        "charge_e": limits["charge_e"],
    }
    if any(maxima[name] > limits[name] for name in maxima):
        raise ValueError(f"{source}: COMSOL t=0 state differs from the realized candidate source")
    if any(function_maxima[name] > function_limits[name] for name in function_maxima):
        raise ValueError(f"{source}: release interpolation is not exact within float64 roundoff")
    return {
        "status": "PASS",
        "criterion": "absolute difference <= multiplier * ulp(max_abs_quantity)",
        "roundoff_multiplier": multiplier,
        "maximum_absolute_difference": maxima,
        "roundoff_limit": limits,
        "release_function_maximum_absolute_difference": function_maxima,
    }


def _validate_initial_primitives(
    primitive: dict[tuple[int, int], list[float]],
    release: dict[int, tuple[float, ...]],
    multiplier: float,
    source: Path,
) -> dict[str, Any]:
    component_metrics: dict[str, dict[str, float | int | str]] = {}
    overall_maximum_component_scale_ulp = 0.0
    for component_index, (raw_column, probe_column) in enumerate(PRIMITIVE_PROBE_COLUMNS):
        comparisons: list[tuple[int, float, float, float]] = []
        for particle_id, expected_values in release.items():
            observed = primitive[(particle_id, 0)][component_index + 2]
            expected = expected_values[component_index]
            comparisons.append((particle_id, abs(observed - expected), observed, expected))
        component_scale = max(
            max(abs(observed), abs(expected)) for _, _, observed, expected in comparisons
        )
        component_ulp = math.ulp(max(component_scale, 1.0e-300))
        worst_particle_id, maximum_absolute_difference, _, _ = max(
            comparisons, key=lambda record: record[1]
        )
        maximum_component_scale_ulp = maximum_absolute_difference / component_ulp
        component_metrics[raw_column] = {
            "probe_column": probe_column,
            "maximum_absolute_difference": maximum_absolute_difference,
            "component_maximum_absolute_value": component_scale,
            "maximum_difference_in_component_scale_ulp": maximum_component_scale_ulp,
            "roundoff_limit": multiplier * component_ulp,
            "worst_particle_id": worst_particle_id,
        }
        overall_maximum_component_scale_ulp = max(
            overall_maximum_component_scale_ulp, maximum_component_scale_ulp
        )
        if maximum_component_scale_ulp > multiplier:
            raise ValueError(
                f"{source}: COMSOL t=0 primitive {raw_column} differs from the "
                f"candidate probe at particle {worst_particle_id}: "
                f"{maximum_component_scale_ulp:.17g} component-scale ulp > "
                f"{multiplier:.17g}"
            )
    return {
        "status": "PASS",
        "criterion": (
            "for every component: maximum absolute difference across particles <= "
            "multiplier * ulp(maximum absolute COMSOL/candidate value for that component, "
            "floored at 1e-300); this component scale remains valid near cancellation; "
            "COMSOL full-precision decimal export and candidate probes are parsed as float64"
        ),
        "roundoff_multiplier": multiplier,
        "checked_particle_count": len(release),
        "checked_component_count": len(PRIMITIVE_PROBE_COLUMNS),
        "checked_value_count": len(release) * len(PRIMITIVE_PROBE_COLUMNS),
        "maximum_difference_in_component_scale_ulp": overall_maximum_component_scale_ulp,
        "components": component_metrics,
    }


def _receipt_fields(log_path: Path, text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in text.split("|")[2:]:
        if "=" not in token:
            raise ValueError(f"{log_path}: malformed receipt token {token!r}")
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ValueError(f"{log_path}: duplicate or empty receipt key {key!r}")
        fields[key] = value
    return fields


def _parse_receipts(log_path: Path) -> dict[str, dict[str, str]]:
    lines = [
        line[line.index(RECEIPT_PREFIX) :].strip()
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if RECEIPT_PREFIX in line
    ]
    if len(lines) != len(STEP_S):
        raise ValueError(f"{log_path}: expected three configuration receipts, found {len(lines)}")
    result: dict[str, dict[str, str]] = {}
    required = {
        "step_s",
        "brownian_active",
        "saffman_active",
        "dynamic_charge_active",
        "native_thermophoresis_active",
        "common_heat_flux_force_active",
        "field_source",
        "initial_state_source",
        "primitive_function_count",
        "deterministic_contributions",
        "integrator",
        "integrator_order",
        "relative_tolerance",
        "wall_accuracy_order",
        "store_particle_status",
        "store_extra",
        "physics",
        "study",
        "solution",
        "output_times",
        "particle_rows",
        "source_model",
        "model_saved",
    }
    expected_static = {
        "brownian_active": "false",
        "saffman_active": "false",
        "dynamic_charge_active": "true",
        "native_thermophoresis_active": "false",
        "common_heat_flux_force_active": "true",
        "field_source": "canonical_exact_connectivity_P1_sectionwise",
        "initial_state_source": "candidate_realized_source_table",
        "primitive_function_count": "22",
        "deterministic_contributions": DETERMINISTIC_CONTRIBUTIONS,
        "integrator": "classical_rk4",
        "integrator_order": "4",
        "relative_tolerance": "1e-8",
        "wall_accuracy_order": "1",
        "store_particle_status": "true",
        "store_extra": "false",
        "physics": "fptas",
        "output_times": str(EXPECTED_FRAMES),
        "particle_rows": str(EXPECTED_PARTICLES),
        "source_model": "source_copy.mph",
        "model_saved": "false",
    }
    for line in lines:
        fields = _receipt_fields(log_path, line)
        if missing := sorted(required - fields.keys()):
            raise ValueError(f"{log_path}: missing receipt keys: {', '.join(missing)}")
        name = next(
            (
                candidate
                for candidate, step in STEP_S.items()
                if math.isclose(float(fields["step_s"]), step, rel_tol=0.0, abs_tol=1e-18)
            ),
            None,
        )
        if name is None or name in result:
            raise ValueError(f"{log_path}: unknown or duplicate step receipt")
        for key, expected in expected_static.items():
            if fields[key].lower() != expected.lower():
                raise ValueError(f"{log_path}: expected {key}={expected}, found {fields[key]}")
        result[name] = fields
    return result


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("evaluation_revision") != 1:
        raise ValueError(f"{path}: unsupported reference-run config revision")
    case = payload.get("case")
    validation = payload.get("validation")
    raw = payload.get("raw_export")
    if not isinstance(case, dict) or not isinstance(validation, dict) or not isinstance(raw, dict):
        raise ValueError(f"{path}: case, validation, and raw_export mappings are required")
    expected_case = {
        "particle_count": EXPECTED_PARTICLES,
        "output_times": EXPECTED_FRAMES,
        "time_start_s": EXPECTED_START_S,
        "time_end_s": EXPECTED_END_S,
        "output_interval_s": OUTPUT_INTERVAL_S,
        "fixed_rk4_steps_s": list(STEP_S.values()),
    }
    for key, expected in expected_case.items():
        if case.get(key) != expected:
            raise ValueError(f"{path}: case.{key} does not match the normalizer")
    if raw.get("revision") != 1 or raw.get("tables") != list(RAW_TABLES):
        raise ValueError(f"{path}: raw export contract does not match the normalizer")
    if validation.get("require_all_records_active") is not True:
        raise ValueError(f"{path}: all pre-event records must be active")
    state_multiplier = float(validation.get("initial_state_roundoff_multiplier", 0.0))
    if not math.isfinite(state_multiplier) or state_multiplier <= 0.0:
        raise ValueError(f"{path}: initial-state roundoff multiplier must be positive")
    primitive_multiplier = float(validation.get("initial_primitive_roundoff_multiplier", 0.0))
    if not math.isfinite(primitive_multiplier) or primitive_multiplier <= 0.0:
        raise ValueError(f"{path}: initial-primitive roundoff multiplier must be positive")
    return {
        "sha256": _sha256(path),
        "state_roundoff_multiplier": state_multiplier,
        "primitive_roundoff_multiplier": primitive_multiplier,
        "payload": payload,
    }


def _verify_prepared_table_files(
    root: Path,
    receipt_path: Path,
    receipt_artifacts: dict[str, Any],
    validation_by_name: dict[str, dict[str, Any]],
) -> int:
    total_size = 0
    for name, expected in receipt_artifacts.items():
        if Path(name).name != name or name in {".", ".."} or not isinstance(expected, dict):
            raise ValueError(f"{receipt_path}: unsafe or malformed artifact record {name!r}")
        artifact_path = root / name
        actual_size = artifact_path.stat().st_size
        actual_hash = _sha256(artifact_path)
        attested = validation_by_name[name]
        if (
            actual_size != int(expected.get("size_bytes", -1))
            or actual_hash != expected.get("sha256")
            or actual_size != int(attested.get("size_bytes", -1))
            or actual_hash != attested.get("sha256")
        ):
            raise ValueError(f"{artifact_path}: prepared-table hash or size differs")
        total_size += actual_size
    return total_size


def _validated_table_phase(
    validation: dict[str, Any],
    phase: str,
    expected_count: int,
    expected_size: int,
    source: Path,
) -> str:
    phase_record = validation.get(phase)
    if not isinstance(phase_record, dict) or phase_record.get("status") != "PASS":
        raise ValueError(f"{source}: {phase} validation is not PASS")
    if (
        int(phase_record.get("artifact_count", -1)) != expected_count
        or int(phase_record.get("total_size_bytes", -1)) != expected_size
    ):
        raise ValueError(f"{source}: {phase} artifact summary differs")
    return "PASS"


def _load_prepared_table_validation(root: Path) -> dict[str, Any]:
    receipt_path = root / "common_p1_table_receipt.json"
    validation_path = root / "prepared_table_validation.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    if receipt.get("schema_version") != 1 or validation.get("schema_version") != 1:
        raise ValueError(f"{root}: unsupported prepared-table validation schema")
    if validation.get("status") != "PASS":
        raise ValueError(f"{validation_path}: prepared-table validation is not PASS")
    receipt_hash = _sha256(receipt_path)
    if validation.get("receipt_sha256") != receipt_hash:
        raise ValueError(f"{validation_path}: table-receipt hash differs")

    receipt_artifacts = receipt.get("artifacts")
    validation_artifacts = validation.get("artifacts")
    if not isinstance(receipt_artifacts, dict) or not isinstance(validation_artifacts, list):
        raise ValueError(f"{validation_path}: prepared-table artifact records are incomplete")
    validation_by_name = {
        str(record.get("path")): record
        for record in validation_artifacts
        if isinstance(record, dict)
    }
    if len(receipt_artifacts) != 26 or set(validation_by_name) != set(receipt_artifacts):
        raise ValueError(f"{validation_path}: expected the same 26 prepared-table artifacts")

    total_size = _verify_prepared_table_files(
        root, receipt_path, receipt_artifacts, validation_by_name
    )
    phase_status = {
        phase: _validated_table_phase(
            validation, phase, len(receipt_artifacts), total_size, validation_path
        )
        for phase in ("pre_comsol", "post_comsol")
    }

    return {
        "status": "PASS",
        "criterion": validation.get("criterion"),
        "receipt_sha256": receipt_hash,
        "validation_artifact_sha256": _sha256(validation_path),
        "artifact_count": len(receipt_artifacts),
        "total_size_bytes": total_size,
        "pre_comsol_status": phase_status["pre_comsol"],
        "post_comsol_status": phase_status["post_comsol"],
    }


def _artifact_streams(directory: Path) -> tuple[dict[str, TextIO], dict[str, CsvWriter]]:
    headers = {
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
        "force_reference.csv": FORCE_COLUMNS,
        "rhs_reference.csv": (
            "particle_id",
            "time_s",
            "position_rate_r_m_per_s",
            "position_rate_z_m_per_s",
            "velocity_rate_r_m_per_s2",
            "velocity_rate_z_m_per_s2",
            "charge_rate_e_per_s",
        ),
        "primitive_reference.csv": PRIMITIVE_COLUMNS,
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
    streams: dict[str, TextIO] = {}
    writers: dict[str, CsvWriter] = {}
    for name, header in headers.items():
        stream = (directory / name).open("x", newline="", encoding="utf-8")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        streams[name] = stream
        writers[name] = writer
    return streams, writers


def _validate_joined_record(
    directory: Path,
    key: tuple[int, int],
    state_row: list[float],
    force_row: list[float],
    primitive_row: list[float],
) -> int:
    if not math.isclose(state_row[1], force_row[1], abs_tol=2e-14) or not math.isclose(
        state_row[1], primitive_row[1], abs_tol=2e-14
    ):
        raise ValueError(f"{directory}: table time-key mismatch at {key}")
    status = _integer(state_row[7], "current_status_code", directory / "state_raw_wide.csv")
    _integer(state_row[8], "final_status_code", directory / "state_raw_wide.csv")
    if status not in STATUS_NAMES:
        raise ValueError(f"{directory}: unknown particle status {status}")
    if state_row[11] <= 0.0:
        raise ValueError(f"{directory}: nonpositive particle mass at {key}")

    sum_r = math.fsum(force_row[index] for index in (2, 4, 6, 8, 10, 12, 14))
    sum_z = math.fsum(force_row[index] for index in (3, 5, 7, 9, 11, 13, 15))
    scale = max(abs(sum_r), abs(sum_z), abs(force_row[16]), abs(force_row[17]), 1e-300)
    tolerance = 8192.0 * math.ulp(scale)
    if abs(sum_r - force_row[16]) > tolerance or abs(sum_z - force_row[17]) > tolerance:
        raise ValueError(f"{directory}: total force does not equal component sum at {key}")
    radial_ok = math.isclose(
        force_row[18], force_row[16] / state_row[11], rel_tol=2e-12, abs_tol=1e-20
    )
    axial_ok = math.isclose(
        force_row[19], force_row[17] / state_row[11], rel_tol=2e-12, abs_tol=1e-20
    )
    if not radial_ok or not axial_ok:
        raise ValueError(f"{directory}: acceleration is inconsistent with force/mass at {key}")
    return status


def _write_joined_record(
    writers: dict[str, CsvWriter],
    particle_id: int,
    lifecycle: str,
    state_row: list[float],
    force_row: list[float],
    primitive_row: list[float],
) -> None:
    writers["trajectory_reference.csv"].writerow((particle_id, *state_row[1:7], lifecycle))
    writers["force_reference.csv"].writerow(force_row)
    writers["rhs_reference.csv"].writerow(
        (
            particle_id,
            state_row[1],
            state_row[4],
            state_row[5],
            force_row[18],
            force_row[19],
            state_row[10],
        )
    )
    writers["primitive_reference.csv"].writerow(primitive_row)


def _write_event(writer: CsvWriter, particle_id: int, row: list[float], directory: Path) -> None:
    writer.writerow(
        (
            particle_id,
            0,
            row[9],
            row[1],
            row[2],
            row[3],
            STATUS_NAMES[_integer(row[7], "current_status_code", directory)],
            "first_saved_nonactive_state",
        )
    )


def _write_step_artifacts(
    directory: Path,
    state: dict[tuple[int, int], list[float]],
    force: dict[tuple[int, int], list[float]],
    primitive: dict[tuple[int, int], list[float]],
) -> tuple[dict[tuple[int, int], tuple[float, ...]], int]:
    streams, writers = _artifact_streams(directory)
    histories: dict[tuple[int, int], tuple[float, ...]] = {}
    event_count = 0
    try:
        for particle_id in range(1, EXPECTED_PARTICLES + 1):
            first_event: list[float] | None = None
            for frame in range(EXPECTED_FRAMES):
                key = (particle_id, frame)
                state_row = state[key]
                force_row = force[key]
                primitive_row = primitive[key]
                status = _validate_joined_record(
                    directory, key, state_row, force_row, primitive_row
                )
                if status != 1 and first_event is None:
                    first_event = state_row
                _write_joined_record(
                    writers, particle_id, STATUS_NAMES[status], state_row, force_row, primitive_row
                )
                histories[key] = (
                    state_row[1],
                    state_row[2],
                    state_row[3],
                    state_row[4],
                    state_row[5],
                    state_row[6],
                )
            if first_event is not None:
                _write_event(writers["event_observations.csv"], particle_id, first_event, directory)
                event_count += 1
    finally:
        for stream in streams.values():
            stream.close()
    return histories, event_count


def _normalize_step(
    directory: Path,
    release: dict[int, tuple[float, float, float, float, float]],
    release_primitives: dict[int, tuple[float, ...]],
    state_multiplier: float,
    primitive_multiplier: float,
    receipt: dict[str, str],
) -> tuple[dict[str, Any], dict[tuple[int, int], tuple[float, ...]]]:
    tables = {name: _read_wide(directory / name, columns) for name, columns in RAW_TABLES.items()}
    keys = set(tables["state_raw_wide.csv"])
    if any(set(records) != keys for records in tables.values()):
        raise ValueError(f"{directory}: raw export tables do not share particle/frame keys")
    state = tables["state_raw_wide.csv"]
    force = tables["force_raw_wide.csv"]
    primitive = tables["primitive_raw_wide.csv"]
    initial = _validate_initial_state(
        state, release, state_multiplier, directory / "state_raw_wide.csv"
    )
    initial_primitives = _validate_initial_primitives(
        primitive,
        release_primitives,
        primitive_multiplier,
        directory / "primitive_raw_wide.csv",
    )

    histories, event_count = _write_step_artifacts(directory, state, force, primitive)

    all_active = event_count == 0 and all(round(row[7]) == 1 for row in state.values())
    if not all_active:
        raise ValueError(f"{directory}: the locked pre-event window contains a nonactive state")
    trajectory = directory / "trajectory_reference.csv"
    return (
        {
            "dt_s": float(receipt["step_s"]),
            "trajectory_path": f"{directory.name}/trajectory_reference.csv",
            "trajectory_sha256": _sha256(trajectory),
            "rows": len(histories),
            "all_active": all_active,
            "event_count": event_count,
            "initial_state": initial,
            "initial_primitives": initial_primitives,
            "raw_sha256": {name: _sha256(directory / name) for name in RAW_TABLES},
        },
        histories,
    )


def _comparison(
    coarse: dict[tuple[int, int], tuple[float, ...]],
    fine: dict[tuple[int, int], tuple[float, ...]],
) -> dict[str, float]:
    if coarse.keys() != fine.keys():
        raise ValueError("self-convergence histories do not have identical keys")
    errors = {"position": [], "velocity": [], "charge": []}
    for key, left in coarse.items():
        right = fine[key]
        errors["position"].append(math.hypot(left[1] - right[1], left[2] - right[2]))
        errors["velocity"].append(math.hypot(left[3] - right[3], left[4] - right[4]))
        errors["charge"].append(abs(left[5] - right[5]))
    result: dict[str, float] = {}
    for name, values in errors.items():
        result[f"{name}_rms"] = math.sqrt(
            math.fsum(value * value for value in values) / len(values)
        )
        result[f"{name}_max"] = max(values, default=0.0)
    return result


def _self_convergence(
    histories: dict[str, dict[tuple[int, int], tuple[float, ...]]],
) -> dict[str, Any]:
    coarse_middle = _comparison(histories["dt_0p625us"], histories["dt_0p3125us"])
    middle_fine = _comparison(histories["dt_0p3125us"], histories["dt_0p15625us"])
    orders: dict[str, float | None] = {}
    for quantity in ("position", "velocity", "charge"):
        coarse_error = coarse_middle[f"{quantity}_rms"]
        fine_error = middle_fine[f"{quantity}_rms"]
        orders[quantity] = (
            math.log(coarse_error / fine_error, 2.0)
            if coarse_error > 0.0 and fine_error > 0.0
            else None
        )
    return {
        "classification": "characterization_only_no_acceptance_gate",
        "dt_0p625us_vs_0p3125us": coarse_middle,
        "dt_0p3125us_vs_0p15625us": middle_fine,
        "observed_order_base_2": orders,
    }


def normalize(output_directory: Path, config_path: Path) -> dict[str, Any]:
    root = output_directory.resolve()
    config = _load_config(config_path.resolve())
    table_validation = _load_prepared_table_validation(root)
    release_path = root / "common_p1_release_probes.csv"
    release, release_primitives = _read_release(release_path)
    receipts = _parse_receipts(root / "comsol_process.log")
    runs: dict[str, Any] = {}
    histories: dict[str, dict[tuple[int, int], tuple[float, ...]]] = {}
    for name in STEP_S:
        summary, history = _normalize_step(
            root / name,
            release,
            release_primitives,
            config["state_roundoff_multiplier"],
            config["primitive_roundoff_multiplier"],
            receipts[name],
        )
        runs[name] = summary
        histories[name] = history
    convergence = _self_convergence(histories)
    _write_json(root / "comsol_self_convergence.json", convergence)
    summary = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "COMPLETE",
        "config_sha256": config["sha256"],
        "release_table_sha256": _sha256(release_path),
        "prepared_table_artifact_validation": table_validation,
        "scope": {
            "particles": EXPECTED_PARTICLES,
            "frames": EXPECTED_FRAMES,
            "time_window_s": [EXPECTED_START_S, EXPECTED_END_S],
            "output_interval_s": OUTPUT_INTERVAL_S,
        },
        "runs": runs,
        "self_convergence": convergence,
        "claim_policy": {
            "common_field_time_integration_diagnostic_only": True,
            "native_field_agreement": "NOT_CERTIFIED",
            "physical_model_validity": "NOT_CERTIFIED",
            "boundary_accuracy": "NOT_TESTED_PRE_EVENT_WINDOW",
            "brownian_accuracy": "NOT_TESTED",
        },
    }
    _write_json(root / "normalization_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize the M3-C1 full-physics canonical-P1 COMSOL reference."
    )
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    normalize(arguments.output_directory, arguments.config)


if __name__ == "__main__":
    main()
