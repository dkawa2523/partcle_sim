"""Validate and bind the raw COMSOL common-P1 material-event run.

This is deliberately a raw-run validator, not a trajectory comparator.  It
checks that COMSOL executed the pre-registered profile and that the three wide
tables are complete enough for the separate event evaluator to consume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

TOOL_REVISION: Final = "m3c1_common_p1_material_event_raw_v1"
RECEIPT_PREFIX: Final = "M3C1_COMMON_P1|configuration|"
RUN_PASS_PREFIX: Final = "M3C1_COMMON_P1|run_pass|"
STEP_DIRECTORY: Final = "dt_0p15625us"
RAW_TABLE_EXPRESSIONS: Final = {
    "state_raw_wide.csv": 15,
    "force_raw_wide.csv": 20,
    "primitive_raw_wide.csv": 24,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _process_log_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")
        return raw.decode("utf-8-sig")
    except UnicodeError as error:
        raise ValueError(f"{path}: process log is neither UTF-8 nor BOM-marked UTF-16") from error


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    case = payload.get("case")
    revision = (payload.get("schema_version"), payload.get("evaluation_revision"))
    if revision != (1, 1):
        raise ValueError(f"{path}: unsupported material-event configuration revision")
    if not isinstance(case, dict):
        raise ValueError(f"{path}: case mapping is required")
    output_times = case.get("output_times_s")
    if not isinstance(output_times, list) or len(output_times) != 48:
        raise ValueError(f"{path}: exactly 48 output times are required")
    expected_prefix = [index * 1e-5 for index in range(46)]
    expected_times = [*expected_prefix, 4.58e-4, 4.5875e-4]
    if any(
        not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-16)
        for actual, expected in zip(output_times, expected_times, strict=True)
    ):
        raise ValueError(f"{path}: output schedule differs from the locked event window")
    expected_case = {
        "particle_count": 287,
        "pre_event_end_s": 4.5e-4,
        "event_window_end_s": 4.5875e-4,
        "fixed_rk4_step_s": 1.5625e-7,
    }
    for key, expected in expected_case.items():
        if case.get(key) != expected:
            raise ValueError(f"{path}: case.{key} differs from {expected!r}")
    return payload


def _fields(line: str, path: Path) -> dict[str, str]:
    parts = line.split("|")[2:]
    values: dict[str, str] = {}
    for part in parts:
        key, separator, value = part.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"{path}: malformed or duplicate receipt field {part!r}")
        values[key] = value
    return values


def _receipt_lines(text: str, prefix: str) -> list[str]:
    return [line[line.index(prefix) :] for line in text.splitlines() if prefix in line]


def _configuration_receipt(log_path: Path, config: dict[str, Any]) -> dict[str, str]:
    text = _process_log_text(log_path)
    lines = _receipt_lines(text, RECEIPT_PREFIX)
    if len(lines) != 1:
        raise ValueError(f"{log_path}: expected one configuration receipt, found {len(lines)}")
    fields = _fields(lines[0], log_path)
    expected = {
        "run_profile": "material_event",
        "brownian_active": "false",
        "saffman_active": "false",
        "dynamic_charge_active": "true",
        "native_thermophoresis_active": "false",
        "common_heat_flux_force_active": "true",
        "field_source": "canonical_exact_connectivity_P1_sectionwise",
        "initial_state_source": "candidate_realized_source_table",
        "primitive_function_count": "22",
        "deterministic_contributions": (
            "electric,relative_flow_ion_drag,epstein_drag,"
            "waldmann_heat_flux_thermophoresis,free_molecular_lift_sensitivity,"
            "dielectrophoresis,gravity_buoyancy"
        ),
        "integrator": "classical_rk4",
        "integrator_order": "4",
        "relative_tolerance": "1e-8",
        "wall_accuracy_order": "1",
        "store_particle_status": "true",
        "store_extra": "false",
        "physics": "fptas",
        "output_times": str(len(config["case"]["output_times_s"])),
        "particle_rows": str(config["case"]["particle_count"]),
        "source_model": "source_copy.mph",
        "model_saved": "false",
    }
    missing = sorted(
        {*expected, "step_s", "time_start_s", "time_end_s", "study", "solution"} - fields.keys()
    )
    if missing:
        raise ValueError(f"{log_path}: configuration receipt lacks {', '.join(missing)}")
    mismatches = [key for key, value in expected.items() if fields[key].lower() != value.lower()]
    if mismatches:
        raise ValueError(f"{log_path}: configuration receipt differs at {', '.join(mismatches)}")
    numeric_expected = {
        "step_s": config["case"]["fixed_rk4_step_s"],
        "time_start_s": 0.0,
        "time_end_s": config["case"]["event_window_end_s"],
    }
    if any(
        not math.isclose(float(fields[key]), float(value), rel_tol=0.0, abs_tol=1e-16)
        for key, value in numeric_expected.items()
    ):
        raise ValueError(f"{log_path}: configuration receipt has the wrong time contract")
    if not fields["study"] or not fields["solution"]:
        raise ValueError(f"{log_path}: study and solution receipts must be nonempty")
    return fields


def _validate_run_pass(log_path: Path) -> None:
    text = _process_log_text(log_path)
    lines = _receipt_lines(text, RUN_PASS_PREFIX)
    if len(lines) != 1:
        raise ValueError(f"{log_path}: expected one run-pass receipt, found {len(lines)}")
    fields = _fields(lines[0], log_path)
    expected = {
        "case": "caseA_100nm",
        "run_profile": "material_event",
        "steps": "1.5625e-7",
        "common_field": "canonical_exact_connectivity_P1_sectionwise",
        "model_saved": "false",
    }
    if fields != expected:
        raise ValueError(f"{log_path}: run-pass receipt is not the locked material-event run")


def _metadata(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    data_rows = 0
    with path.open(encoding="utf-8-sig", errors="strict") as stream:
        for line in stream:
            if line.startswith("% Nodes,"):
                values["nodes"] = int(line.rstrip().split(",", 1)[1])
            elif line.startswith("% Expressions,"):
                values["expressions"] = int(line.rstrip().split(",", 1)[1])
            elif not line.startswith("%") and line.strip():
                expressions = values.get("expressions")
                if expressions is None:
                    raise ValueError(f"{path}: data precedes the expressions header")
                row = next(csv.reader([line]))
                if len(row) != expressions:
                    raise ValueError(
                        f"{path}: data row {data_rows + 1} has {len(row)} values, "
                        f"expected {expressions}"
                    )
                try:
                    finite = all(math.isfinite(float(value)) for value in row)
                except ValueError as error:
                    raise ValueError(
                        f"{path}: data row {data_rows + 1} contains a nonnumeric value"
                    ) from error
                if not finite:
                    raise ValueError(f"{path}: data row {data_rows + 1} is not finite")
                data_rows += 1
    values["data_rows"] = data_rows
    return values


def _raw_table_records(root: Path, particle_count: int, frame_count: int) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, values_per_frame in RAW_TABLE_EXPRESSIONS.items():
        path = root / STEP_DIRECTORY / name
        metadata = _metadata(path)
        expected = {
            "nodes": particle_count,
            "expressions": frame_count * values_per_frame,
            "data_rows": particle_count,
        }
        if metadata != expected:
            raise ValueError(f"{path}: raw table shape {metadata} differs from {expected}")
        records[name] = {
            "relative_path": f"{STEP_DIRECTORY}/{name}",
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            **metadata,
        }
    return records


def validate(output_directory: Path, config_path: Path) -> dict[str, Any]:
    root = output_directory.resolve()
    locked_config = config_path.resolve()
    config = _load_config(locked_config)
    process_log = root / "comsol_process.log"
    receipt = _configuration_receipt(process_log, config)
    _validate_run_pass(process_log)
    raw_tables = _raw_table_records(
        root, int(config["case"]["particle_count"]), len(config["case"]["output_times_s"])
    )
    summary = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "PASS",
        "config_sha256": _sha256(locked_config),
        "configuration_receipt": receipt,
        "raw_tables": raw_tables,
        "claim_policy": {
            "configuration_and_raw_shape_only": True,
            "event_result": "NOT_EVALUATED",
            "candidate_agreement": "NOT_EVALUATED",
        },
    }
    output = root / "material_event_raw_validation.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    validate(arguments.output_directory, arguments.config)


if __name__ == "__main__":
    main()
