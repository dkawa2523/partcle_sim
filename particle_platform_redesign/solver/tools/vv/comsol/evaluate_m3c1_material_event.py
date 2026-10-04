"""Evaluate the locked M3-C1 common-P1 first material-stick event.

This is an external V&V utility.  It binds one completed candidate run and one
COMSOL raw export to their receipts, then compares only the pre-registered
first wafer-stick observables.  The registered historical pre-event comparison
owns the budget and reference identity; its numbers and the current candidate
prefix are independently recomputed here.  COMSOL terminal velocity is
deliberately reported rather than accepted as a cross-solver contract because
COMSOL's Freeze storage convention and the solver's physical stick state differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import h5py
import numpy as np

from chamber_particles import open_result
from tools.vv.comsol import evaluate_m3c1_common_field as common_field
from tools.vv.comsol import evaluate_m3c1_pre_event as common_shared

TOOL_REVISION: Final = "m3c1_common_p1_first_material_event_evaluation_v4"
EVALUATION_ID: Final = "M3-C1-caseA-100nm-common-P1-first-material-event"
REFERENCE_TOOL_REVISION: Final = "m3c1_common_p1_material_event_comsol_v1"
RAW_VALIDATION_REVISION: Final = "m3c1_common_p1_material_event_raw_v1"
CANDIDATE_TOOL_REVISION: Final = "m3c1_common_p1_material_event_candidate_v2"
CANDIDATE_PRODUCER_FILENAME: Final = "run_m3c1_material_event_candidate.py"
EVALUATOR_SOURCE_FILENAME: Final = "evaluate_m3c1_material_event.py"
PRE_EVENT_TRAJECTORY_FILENAME: Final = "candidate_pre_event_trajectory.csv"
LOCKED_CONFIG_SHA256: Final = "d46a9f6040125e66d62c219861ab1b037d9b655dbeed9f029df13a0139ac3fed"
LOCKED_CANDIDATE_CASE_SHA256: Final = (
    "a8876aab0ff133c4c09a257c70445b286ceabe051420711b575ac0f34962832a"
)
LOCKED_ACCEPTANCE: Final = {
    "event_time_absolute_s": 1.0e-9,
    "hit_position_norm_m": 2.0e-9,
    "terminal_position_hold_m": 2.0e-12,
    "terminal_charge_cross_absolute_e": 0.0045,
    "terminal_charge_hold_roundoff_multiplier": 4096.0,
}
LOCKED_CLAIM_POLICY: Final = {
    "pre_event_same_field_authority": "separate_locked_0_to_450_us_nine_gate_PASS",
    "material_event_scope": "first_wafer_stick_only",
    "comsol_is_golden_truth": False,
    "certifies_native_field_agreement": False,
    "certifies_terminal_velocity_parity": False,
    "certifies_physical_model_validity": False,
    "certifies_brownian_accuracy": False,
    "certifies_universal_comsol_accuracy": False,
    "result_dependent_tolerance_tuning": "PROHIBITED",
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
TRAJECTORY_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
EVENT_COLUMNS: Final = (
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
    "primary_facet_id",
    "boundary_id",
    "law",
    "outcome",
)


@dataclass(frozen=True)
class _ReferenceEvent:
    particle_id: int
    nonactive_particle_ids: tuple[int, ...]
    terminal: np.ndarray
    event_time_s: float
    hit_position_m: np.ndarray
    position_hold_m: float
    charge_hold: dict[str, float]
    terminal_charge_e: float
    current_status_codes: tuple[int, ...]
    transition_ok: bool
    event_time_bracket_ok: bool
    last_active_time_s: float
    first_terminal_time_s: float
    pre_event_final_status_codes: tuple[int, ...]


@dataclass(frozen=True)
class _CandidateEvent:
    identity: dict[str, object]
    identity_matches: bool
    terminal: tuple[dict[str, str], ...]
    event_time_s: float
    hit_position_m: np.ndarray
    position_hold_m: float
    charge_hold: dict[str, float]
    terminal_charge_e: float
    lifecycle_history: tuple[str, ...]
    transition_ok: bool
    event_time_bracket_ok: bool
    last_active_time_s: float
    first_stuck_time_s: float
    nonactive_particle_ids: tuple[int, ...]
    canonical_segment_m: np.ndarray
    canonical_surface_ok: bool
    hit_distance_to_segment_m: float
    event_post_velocity_m_s: np.ndarray
    stuck_velocity_maximum_absolute_m_s: float
    stuck_velocity_row_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _integer(value: float, name: str) -> int:
    rounded = round(value)
    if not math.isfinite(value) or abs(value - rounded) > 1e-9:
        raise ValueError(f"{name} is not integer-valued: {value}")
    return rounded


def _load_json(path: Path, name: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {path}") from error
    return _mapping(payload, name), _sha256(path)


def _solver_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stage_evaluator_source(output: Path) -> dict[str, str]:
    source = Path(__file__).resolve()
    staged = output / EVALUATOR_SOURCE_FILENAME
    with source.open("rb") as source_stream, staged.open("xb") as staged_stream:
        for block in iter(lambda: source_stream.read(1024 * 1024), b""):
            staged_stream.write(block)
    digest = _sha256(staged)
    if digest != _sha256(source):
        raise OSError("staged evaluator differs from the executing source")
    return {"file": EVALUATOR_SOURCE_FILENAME, "sha256": digest}


def _validate_case_scope(config: dict[str, Any]) -> None:
    case = _mapping(config.get("case"), "configuration.case")
    times = case.get("output_times_s")
    if (
        case.get("workflow") != "caseA"
        or float(case.get("diameter_m", math.nan)) != 1.0e-7
        or case.get("particle_count") != 287
        or float(case.get("pre_event_end_s", math.nan)) != 4.5e-4
        or float(case.get("event_window_end_s", math.nan)) != 4.5875e-4
        or float(case.get("fixed_rk4_step_s", math.nan)) != 1.5625e-7
        or not isinstance(times, list)
        or len(times) != 48
    ):
        raise ValueError("material-event scope must contain 287 particles and 48 frames")
    numeric_times = np.asarray(times, dtype=np.float64)
    locked_times = np.asarray(
        [*(index * 1.0e-5 for index in range(46)), 4.58e-4, 4.5875e-4],
        dtype=np.float64,
    )
    valid = (
        bool(np.isfinite(numeric_times).all())
        and numeric_times[0] == 0.0
        and numeric_times[-1] == float(case.get("event_window_end_s", math.nan))
        and not bool((np.diff(numeric_times) <= 0.0).any())
        and bool(np.allclose(numeric_times, locked_times, rtol=0.0, atol=1.0e-18))
    )
    if not valid:
        raise ValueError("material-event output schedule is invalid")


def _validate_acceptance(config: dict[str, Any]) -> None:
    acceptance = _mapping(config.get("acceptance"), "configuration.acceptance")
    if acceptance != LOCKED_ACCEPTANCE:
        raise ValueError("material-event acceptance limits differ from the registered values")
    values = [float(value) for value in acceptance.values()]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("material-event acceptance limits must be fixed positive numbers")


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    config, digest = _load_json(path, "material-event configuration")
    if digest != LOCKED_CONFIG_SHA256:
        raise ValueError("material-event configuration differs from its registered SHA-256")
    if (
        config.get("schema_version") != 1
        or config.get("evaluation_id") != EVALUATION_ID
        or config.get("evaluation_revision") != 1
    ):
        raise ValueError("unsupported material-event configuration identity")
    _validate_case_scope(config)
    _validate_acceptance(config)
    policy = _mapping(config.get("claim_policy"), "configuration.claim_policy")
    if policy != LOCKED_CLAIM_POLICY:
        raise ValueError("material-event claim policy differs from the registered policy")
    expected = _mapping(config.get("expected_first_event"), "expected_first_event")
    if expected != {
        "particle_id": 57,
        "candidate_boundary_id": 6,
        "candidate_external_id": 134,
        "boundary_group": "wafer",
        "comsol_status_code": 3,
        "candidate_lifecycle": "stuck",
        "candidate_law": "stick",
        "wafer_z_m": 0.022,
    }:
        raise ValueError("expected first-event identity differs from the registered identity")
    return config, digest


def _safe_relative(root: Path, relative: object, name: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError(f"{name} path must be text")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{name} path must stay inside its receipt root")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{name} path escapes its receipt root")
    return resolved


def _verify_file(path: Path, digest: object, name: str, size: object | None = None) -> None:
    if not isinstance(digest, str) or not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"{name} differs from its receipt")
    if size is not None:
        if not isinstance(size, (int, str)):
            raise ValueError(f"{name} byte size is malformed")
        if path.stat().st_size != int(size):
            raise ValueError(f"{name} byte size differs from its receipt")


def _artifact_ledger(
    path: Path,
    root: Path,
    expected_hash: object,
    receipt_name: str,
) -> dict[str, str]:
    _verify_file(path, expected_hash, "reference artifact ledger")
    records: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != ("path", "sha256", "bytes"):
            raise ValueError("reference artifact ledger columns are invalid")
        for line_number, row in enumerate(reader, start=2):
            relative = row["path"]
            if relative in records:
                raise ValueError(f"duplicate ledger path at line {line_number}: {relative}")
            artifact = _safe_relative(root, relative, "reference artifact")
            _verify_file(artifact, row["sha256"], f"reference artifact {relative}", row["bytes"])
            records[relative] = row["sha256"]
    excluded = {path.name, "run_status.json", receipt_name}
    expected = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file() and artifact.relative_to(root).as_posix() not in excluded
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise ValueError(
            f"reference artifact ledger coverage differs; missing={missing}, extra={extra}"
        )
    return records


def _verify_run_status(path: Path) -> tuple[dict[str, Any], str]:
    status, status_hash = _load_json(path, "reference run status")
    expected = {
        "status": "COMPLETE",
        "failure": "",
        "source_copy_retained": False,
        "compiled_class_retained": False,
        "class_status_retained": False,
    }
    mismatches = [key for key, value in expected.items() if status.get(key) != value]
    if mismatches:
        raise ValueError(f"reference run status differs at {', '.join(mismatches)}")
    return status, status_hash


def _verify_reference_identity(
    config: dict[str, Any], config_hash: str, receipt: dict[str, Any]
) -> None:
    source = _mapping(config.get("source_model"), "configuration.source_model")
    candidate = _mapping(config.get("candidate"), "configuration.candidate")
    case = _mapping(config.get("case"), "configuration.case")
    expected = {
        "schema_version": 1,
        "tool_revision": REFERENCE_TOOL_REVISION,
        "classification": config.get("classification"),
        "status": "COMPLETE",
        "config_sha256": config_hash,
        "candidate_input_sha256": candidate.get("input_sha256"),
        "source_sha256_before": source.get("sha256"),
        "source_sha256_after": source.get("sha256"),
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise ValueError(f"reference run receipt differs at {', '.join(mismatches)}")
    expected_version = config.get("expected_comsol_version")
    actual_version = receipt.get("comsol_version")
    if not isinstance(expected_version, str) or not isinstance(actual_version, str):
        raise ValueError("reference COMSOL version receipt is malformed")
    if expected_version not in actual_version:
        raise ValueError("reference COMSOL version differs from the configuration")
    expected_scope = {
        "particles": case["particle_count"],
        "frames": len(case["output_times_s"]),
        "time_window_s": [0.0, case["event_window_end_s"]],
        "fixed_rk4_step_s": case["fixed_rk4_step_s"],
    }
    if receipt.get("scope") != expected_scope:
        raise ValueError("reference run scope differs from the configuration")


def _validated_raw_records(
    config_hash: str,
    receipt: dict[str, Any],
    validation: dict[str, Any],
    validation_hash: str,
) -> dict[str, dict[str, Any]]:
    if receipt.get("raw_validation_sha256") != validation_hash:
        raise ValueError("reference raw validation hash differs from the run receipt")
    expected_validation = {
        "schema_version": 1,
        "tool_revision": RAW_VALIDATION_REVISION,
        "status": "PASS",
        "config_sha256": config_hash,
    }
    mismatches = [key for key, value in expected_validation.items() if validation.get(key) != value]
    if mismatches:
        raise ValueError(f"reference raw validation differs at {', '.join(mismatches)}")
    receipt_tables = _mapping(receipt.get("raw_tables"), "reference receipt raw_tables")
    validation_tables = _mapping(validation.get("raw_tables"), "raw validation raw_tables")
    expected_names = {
        "state_raw_wide.csv",
        "force_raw_wide.csv",
        "primitive_raw_wide.csv",
    }
    if set(receipt_tables) != expected_names or set(validation_tables) != expected_names:
        raise ValueError("reference raw-table receipts do not cover the three required tables")
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(expected_names):
        receipt_record = _mapping(receipt_tables.get(name), f"receipt {name}")
        validation_record = _mapping(validation_tables.get(name), f"validation {name}")
        if receipt_record != validation_record:
            raise ValueError(f"reference {name} records differ between receipts")
        expected_expressions = {
            "state_raw_wide.csv": 720,
            "force_raw_wide.csv": 960,
            "primitive_raw_wide.csv": 1152,
        }
        expected_metadata = {
            "nodes": 287,
            "data_rows": 287,
            "expressions": expected_expressions[name],
        }
        if any(receipt_record.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(f"reference {name} metadata differs from the registered scope")
        records[name] = receipt_record
    return records


def _declared_prepared_files(table_receipt: dict[str, Any]) -> set[str]:
    components = table_receipt.get("components")
    release = _mapping(table_receipt.get("release"), "prepared-table release")
    functions = release.get("functions")
    if not isinstance(components, list) or not isinstance(functions, list):
        raise ValueError("prepared-table component/release declarations are invalid")
    if table_receipt.get("component_count") != 22 or len(components) != 22:
        raise ValueError("prepared-table component scope is invalid")
    if len(functions) != 3:
        raise ValueError("prepared-table release-function scope is invalid")
    files = {
        str(_mapping(component, "prepared-table component").get("file")) for component in components
    }
    files.update(
        str(_mapping(function, "prepared-table release function").get("file"))
        for function in functions
    )
    files.add(str(release.get("probe_file")))
    return files


def _load_prepared_table_receipt(
    config: dict[str, Any], receipt: dict[str, Any], root: Path
) -> tuple[Path, dict[str, Any], str, dict[str, Any]]:
    candidate = _mapping(config.get("candidate"), "configuration.candidate")
    table_receipt_path = root / "common_p1_table_receipt.json"
    table_receipt, table_receipt_hash = _load_json(table_receipt_path, "prepared-table receipt")
    if table_receipt_hash != receipt.get("table_receipt_sha256"):
        raise ValueError("prepared-table receipt hash differs from the run receipt")
    expected_identity = {
        "schema_version": 1,
        "tool_revision": "m3c1_full_physics_common_p1_tables_v1",
        "classification": "external_vv_full_physics_exact_p1_common_field_input",
    }
    if any(table_receipt.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("prepared-table receipt identity is invalid")
    table_candidate = _mapping(table_receipt.get("candidate"), "prepared-table candidate")
    if table_candidate.get("file_sha256") != candidate.get("input_sha256") or table_candidate.get(
        "content_hash"
    ) != candidate.get("input_content_hash"):
        raise ValueError("prepared-table receipt uses another candidate input")
    artifacts = _mapping(table_receipt.get("artifacts"), "prepared-table artifacts")
    required_files = _declared_prepared_files(table_receipt)
    if set(artifacts) != required_files or len(artifacts) != 26:
        raise ValueError("prepared-table receipt does not cover its declared files exactly")
    return table_receipt_path, table_receipt, table_receipt_hash, artifacts


def _verify_prepared_artifact_records(
    artifacts: dict[str, Any],
    validation: dict[str, Any],
    root: Path,
    ledger: dict[str, str],
) -> None:
    validated_artifacts = validation.get("artifacts")
    if not isinstance(validated_artifacts, list) or len(validated_artifacts) != len(artifacts):
        raise ValueError("prepared-table validation artifact count is invalid")
    validation_by_path = {
        str(_mapping(record, "prepared-table validation record").get("path")): _mapping(
            record, "prepared-table validation record"
        )
        for record in validated_artifacts
    }
    if set(validation_by_path) != set(artifacts):
        raise ValueError("prepared-table validation paths differ from the table receipt")
    for relative, raw_record in artifacts.items():
        record = _mapping(raw_record, f"prepared-table artifact {relative}")
        validation_record = validation_by_path[relative]
        expected = (record.get("sha256"), record.get("size_bytes"))
        observed = (validation_record.get("sha256"), validation_record.get("size_bytes"))
        if observed != expected:
            raise ValueError(f"prepared-table validation differs for {relative}")
        artifact = _safe_relative(root, relative, "prepared-table artifact")
        _verify_file(
            artifact,
            expected[0],
            f"prepared-table artifact {relative}",
            expected[1],
        )
        if ledger.get(relative) != record.get("sha256"):
            raise ValueError(f"prepared-table artifact is absent from the final ledger: {relative}")


def _verify_reference_provenance_record(
    config: dict[str, Any],
    config_hash: str,
    table_receipt_hash: str,
    provenance: dict[str, Any],
    root: Path,
    ledger: dict[str, str],
) -> None:
    candidate = _mapping(config.get("candidate"), "configuration.candidate")
    source_hash = _mapping(config.get("source_model"), "configuration.source_model").get("sha256")
    expected_provenance = {
        "classification": config.get("classification"),
        "evaluation_id": EVALUATION_ID,
        "evaluation_revision": 1,
        "config_sha256": config_hash,
        "source_sha256_before": source_hash,
        "source_sha256_after": source_hash,
        "source_unchanged": True,
        "source_load_mode": "ModelUtil.loadCopy",
        "model_saved": False,
        "isolated_source_copy_sha256": source_hash,
        "isolated_source_copy_retained": False,
        "candidate_input_sha256": candidate.get("input_sha256"),
        "candidate_content_hash": candidate.get("input_content_hash"),
        "table_receipt_sha256": table_receipt_hash,
        "prepared_table_validation_status": "PASS",
        "run_profile": "material_event",
        "process_count": 1,
        "comsol_version": f"COMSOL Multiphysics {config.get('expected_comsol_version')}",
        "runner_sha256": _sha256(Path(__file__).with_name("run_m3c1_common_p1_reference.ps1")),
    }
    mismatches = [key for key, value in expected_provenance.items() if provenance.get(key) != value]
    if mismatches:
        raise ValueError(f"reference provenance differs at {', '.join(mismatches)}")
    staged_tools = {
        "staged_config": ("config_sha256", config_hash),
        "staged_java": ("java_source_sha256", provenance.get("java_source_sha256")),
        "material_event_entry_java": (
            "material_event_entry_java_sha256",
            provenance.get("material_event_entry_java_sha256"),
        ),
        "staged_preparer": ("preparer_source_sha256", provenance.get("preparer_source_sha256")),
        "staged_postprocessor": (
            "postprocessor_source_sha256",
            provenance.get("postprocessor_source_sha256"),
        ),
    }
    for path_key, (_, digest) in staged_tools.items():
        staged = _safe_relative(root, provenance.get(path_key), f"reference {path_key}")
        _verify_file(staged, digest, f"reference {path_key}")
        if ledger.get(staged.relative_to(root).as_posix()) != digest:
            raise ValueError(f"reference {path_key} is absent from the final ledger")


def _verify_prepared_tables(
    config: dict[str, Any],
    config_hash: str,
    receipt: dict[str, Any],
    root: Path,
    ledger: dict[str, str],
) -> dict[str, object]:
    table_receipt_path, _, table_receipt_hash, artifacts = _load_prepared_table_receipt(
        config, receipt, root
    )
    validation_path = root / "prepared_table_validation.json"
    validation, validation_hash = _load_json(validation_path, "prepared-table artifact validation")
    if validation != receipt.get("prepared_table_validation"):
        raise ValueError("prepared-table validation differs from the run receipt")
    if (
        validation.get("schema_version") != 1
        or validation.get("status") != "PASS"
        or validation.get("receipt_sha256") != table_receipt_hash
    ):
        raise ValueError("prepared-table validation identity is invalid")
    _verify_prepared_artifact_records(artifacts, validation, root, ledger)
    provenance_path = root / "provenance.json"
    provenance, provenance_hash = _load_json(provenance_path, "reference provenance")
    _verify_reference_provenance_record(
        config, config_hash, table_receipt_hash, provenance, root, ledger
    )
    for fixed_path, digest in (
        (table_receipt_path, table_receipt_hash),
        (validation_path, validation_hash),
        (provenance_path, provenance_hash),
    ):
        relative = fixed_path.relative_to(root).as_posix()
        if ledger.get(relative) != digest:
            raise ValueError(f"reference artifact is absent from the final ledger: {relative}")
    return {
        "table_receipt": str(table_receipt_path.resolve()),
        "table_receipt_sha256": table_receipt_hash,
        "prepared_table_validation": str(validation_path.resolve()),
        "prepared_table_validation_sha256": validation_hash,
        "prepared_table_artifacts": len(artifacts),
        "provenance": str(provenance_path.resolve()),
        "provenance_sha256": provenance_hash,
    }


def _reference_provenance(
    config: dict[str, Any],
    config_hash: str,
    receipt_path: Path,
    validation_path: Path,
    state_path: Path,
    ledger_path: Path,
    status_path: Path,
) -> dict[str, object]:
    receipt, receipt_hash = _load_json(receipt_path, "reference run receipt")
    validation, validation_hash = _load_json(validation_path, "reference raw validation")
    _, status_hash = _verify_run_status(status_path)
    _verify_reference_identity(config, config_hash, receipt)
    raw_records = _validated_raw_records(config_hash, receipt, validation, validation_hash)
    receipt_state = raw_records["state_raw_wide.csv"]
    root = receipt_path.parent.resolve()
    canonical_paths = {
        "receipt": root / "common_p1_material_event_run_receipt.json",
        "validation": root / "material_event_raw_validation.json",
        "ledger": root / "artifact_hashes.csv",
        "status": root / "run_status.json",
    }
    actual_paths = {
        "receipt": receipt_path.resolve(),
        "validation": validation_path.resolve(),
        "ledger": ledger_path.resolve(),
        "status": status_path.resolve(),
    }
    if actual_paths != canonical_paths:
        raise ValueError("reference receipt paths are not the canonical run artifacts")
    if status_path.resolve().parent != root:
        raise ValueError("reference run status must share the receipt root")
    expected_state_path = _safe_relative(root, receipt_state.get("relative_path"), "state table")
    if state_path.resolve() != expected_state_path:
        raise ValueError("explicit reference state path differs from its receipt path")
    if validation_path.resolve().parent != root or ledger_path.resolve().parent != root:
        raise ValueError("reference receipt artifacts must share one root")
    ledger = _artifact_ledger(
        ledger_path, root, receipt.get("artifact_hashes_sha256"), receipt_path.name
    )
    raw_provenance: dict[str, object] = {}
    for name, record in raw_records.items():
        raw_path = _safe_relative(root, record.get("relative_path"), f"reference {name}")
        _verify_file(raw_path, record.get("sha256"), f"reference {name}", record.get("size_bytes"))
        relative = raw_path.relative_to(root).as_posix()
        if ledger.get(relative) != record.get("sha256"):
            raise ValueError(f"reference {name} is absent from the final artifact ledger")
        raw_provenance[name] = {
            "path": str(raw_path),
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    validation_relative = validation_path.resolve().relative_to(root).as_posix()
    if ledger.get(validation_relative) != validation_hash:
        raise ValueError("reference validation is absent from the final artifact ledger")
    prepared = _verify_prepared_tables(config, config_hash, receipt, root, ledger)
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": receipt_hash,
        "raw_validation": str(validation_path.resolve()),
        "raw_validation_sha256": validation_hash,
        "state_raw_wide": str(state_path.resolve()),
        "state_raw_wide_sha256": receipt_state["sha256"],
        "raw_tables": raw_provenance,
        "artifact_ledger": str(ledger_path.resolve()),
        "artifact_ledger_sha256": receipt["artifact_hashes_sha256"],
        "verified_artifact_count": len(ledger),
        "comsol_version": receipt["comsol_version"],
        "run_status": str(status_path.resolve()),
        "run_status_sha256": status_hash,
        "prepared_common_p1": prepared,
    }


def _expected_prepare_inputs(config: dict[str, Any]) -> dict[str, tuple[Path, object]]:
    candidate_config = _mapping(config.get("candidate"), "configuration.candidate")
    reference_config = _mapping(config.get("reference"), "configuration.reference")
    parent_root = (
        _solver_root() / str(candidate_config.get("parent_root_relative_path"))
    ).resolve()
    reference_root = (
        _solver_root() / str(reference_config.get("pre_event_root_relative_path"))
    ).resolve()
    return {
        "case": (
            _safe_relative(
                parent_root, candidate_config.get("parent_case_filename"), "parent case"
            ),
            candidate_config.get("parent_case_sha256"),
        ),
        "comparison": (
            (
                _solver_root() / str(reference_config.get("pre_event_comparison_relative_path"))
            ).resolve(),
            reference_config.get("pre_event_comparison_sha256"),
        ),
        "input": (
            _safe_relative(parent_root, candidate_config.get("input_filename"), "candidate input"),
            candidate_config.get("input_sha256"),
        ),
        "manifest": (
            _safe_relative(
                parent_root,
                candidate_config.get("parent_result_manifest_relative_path"),
                "parent manifest",
            ),
            candidate_config.get("parent_result_manifest_sha256"),
        ),
        "reference_receipt": (
            reference_root / "common_p1_run_receipt.json",
            reference_config.get("pre_event_run_receipt_sha256"),
        ),
        "run_report": (
            _safe_relative(
                parent_root,
                candidate_config.get("parent_run_report_filename"),
                "parent run report",
            ),
            candidate_config.get("parent_run_report_sha256"),
        ),
        "trajectory": (
            _safe_relative(
                parent_root,
                candidate_config.get("parent_trajectory_filename"),
                "parent trajectory",
            ),
            candidate_config.get("parent_trajectory_sha256"),
        ),
    }


def _verify_candidate_prepare(
    config: dict[str, Any],
    config_hash: str,
    prepare: dict[str, Any],
    case_path: Path,
) -> None:
    expected_identity = {
        "status": "PREPARED",
        "tool_revision": CANDIDATE_TOOL_REVISION,
        "configuration_sha256": config_hash,
        "configuration_sha256_before": config_hash,
        "configuration_sha256_after": config_hash,
        "case_sha256": _sha256(case_path),
    }
    mismatches = [key for key, value in expected_identity.items() if prepare.get(key) != value]
    if mismatches:
        raise ValueError(f"candidate prepare receipt differs at {', '.join(mismatches)}")
    case_scope = _mapping(config.get("case"), "configuration.case")
    expected_scope = {
        "particles": case_scope["particle_count"],
        "frames": len(case_scope["output_times_s"]),
        "time_window_s": [0.0, case_scope["event_window_end_s"]],
        "dt_s": case_scope["fixed_rk4_step_s"],
    }
    if prepare.get("scope") != expected_scope:
        raise ValueError("candidate prepare scope differs from the configuration")
    if prepare.get("claim_policy") != config.get("claim_policy"):
        raise ValueError("candidate prepare claim policy differs from the configuration")
    expected_locked = _expected_prepare_inputs(config)
    locked_inputs = _mapping(prepare.get("locked_inputs"), "candidate prepare locked_inputs")
    if set(locked_inputs) != set(expected_locked):
        raise ValueError("candidate prepare locked-input names are invalid")
    for name, (expected_path, expected_hash) in expected_locked.items():
        record = _mapping(locked_inputs.get(name), f"candidate prepare locked input {name}")
        if (
            Path(str(record.get("path"))).resolve() != expected_path
            or record.get("sha256") != expected_hash
        ):
            raise ValueError(f"candidate prepare locked input differs: {name}")
        _verify_file(expected_path, expected_hash, f"candidate prepare locked input {name}")


def _verify_candidate_producer(
    root: Path, prepare: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    prepared = _mapping(prepare.get("producer_source"), "candidate prepare producer source")
    completed = _mapping(report.get("producer_source"), "candidate report producer source")
    if prepared != completed or prepared.get("file") != CANDIDATE_PRODUCER_FILENAME:
        raise ValueError("candidate producer source identity differs between receipts")
    source = _safe_relative(root, prepared.get("file"), "candidate producer source")
    _verify_file(source, prepared.get("sha256"), "candidate producer source")
    return {"file": source.name, "sha256": _sha256(source)}


def _verify_candidate_counts(
    config: dict[str, Any], counts: dict[str, Any], report: dict[str, Any]
) -> None:
    if counts.get("failure_events") != report.get("failure_event_count"):
        raise ValueError("candidate failure count differs between receipts")
    receipt_counts = (counts.get("frame_rows"), counts.get("boundary_events"))
    report_counts = (report.get("trajectory_rows"), report.get("event_rows"))
    if receipt_counts != report_counts:
        raise ValueError("candidate row counts differ between receipts")
    case = _mapping(config.get("case"), "configuration.case")
    expected_counts = {
        "particles": case["particle_count"],
        "frames": len(case["output_times_s"]),
        "frame_rows": case["particle_count"] * len(case["output_times_s"]),
        "release_events": case["particle_count"],
        "boundary_events": 1,
        "failure_events": 0,
    }
    mismatches = [key for key, value in expected_counts.items() if counts.get(key) != value]
    if mismatches:
        raise ValueError(f"candidate manifest counts differ at {', '.join(mismatches)}")


def _verify_candidate_execution(manifest: dict[str, Any]) -> None:
    expected_execution = {
        "engine_algorithm_revision": "particle_engine_v31",
        "boundary_algorithm_revision": "point_wall_laws_v4",
        "event_algorithm_revision": "line_quadratic_rk4_axis_first_hit_v14",
        "physics_runtime_revision": "inertial_langevin_compiled_physics_runtime_v17",
        "motion_mode": "axisymmetric_rz_meridional",
    }
    mismatches = [key for key, value in expected_execution.items() if manifest.get(key) != value]
    resolved = _mapping(manifest.get("resolved"), "candidate manifest resolved")
    if (resolved.get("backend"), resolved.get("integrator")) != ("cpu", "rk4_fixed"):
        mismatches.append("resolved backend/integrator")
    if manifest.get("brownian_rng_revision") is not None:
        mismatches.append("brownian_rng_revision")
    if mismatches:
        raise ValueError(
            f"candidate manifest execution identity differs at {', '.join(mismatches)}"
        )


def _verify_candidate_manifest(
    config: dict[str, Any],
    manifest: dict[str, Any],
    report: dict[str, Any],
    case_path: Path,
) -> None:
    counts = _mapping(manifest.get("counts"), "candidate manifest counts")
    if manifest.get("status") != "complete" or report.get("result_manifest_status") != "complete":
        raise ValueError("candidate result is incomplete")
    _verify_candidate_counts(config, counts, report)
    case = _mapping(config.get("case"), "configuration.case")
    candidate_config = _mapping(config.get("candidate"), "configuration.candidate")
    if manifest.get("case_file_hash") != f"sha256:{_sha256(case_path)}":
        raise ValueError("candidate manifest case_file_hash differs")
    if manifest.get("data_content_hash") != candidate_config.get("input_content_hash"):
        raise ValueError("candidate manifest uses another canonical input")
    time = _mapping(manifest.get("time"), "candidate manifest time")
    expected_time = {
        "start_s": 0.0,
        "end_s": case["event_window_end_s"],
        "dt_s": case["fixed_rk4_step_s"],
    }
    if any(float(time.get(key, math.nan)) != float(value) for key, value in expected_time.items()):
        raise ValueError("candidate manifest time scope differs from the configuration")
    _verify_candidate_execution(manifest)
    lifecycle = _mapping(manifest.get("lifecycle_counts"), "candidate lifecycle counts")
    expected_lifecycle = {"active": 286, "pending": 0, "stuck": 1, "escaped": 0, "failed": 0}
    if lifecycle != expected_lifecycle:
        raise ValueError("candidate lifecycle counts do not describe one stuck particle")


def _candidate_provenance(
    config: dict[str, Any], config_hash: str, root: Path
) -> tuple[dict[str, Any], Path, Path, Path, Any, dict[str, Any]]:
    report_path = root / "candidate_run_report.json"
    report, _ = _load_json(report_path, "candidate run report")
    expected_report = {
        "status": "COMPLETE",
        "tool_revision": CANDIDATE_TOOL_REVISION,
        "configuration_sha256": config_hash,
        "configuration_sha256_before": config_hash,
        "configuration_sha256_after": config_hash,
        "trajectory": "candidate_trajectory.csv",
        "pre_event_trajectory": PRE_EVENT_TRAJECTORY_FILENAME,
        "events": "candidate_events.csv",
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise ValueError("candidate run receipt identity is invalid")
    prepare_path = root / "prepare_report.json"
    case_path = root / "candidate_material_event.yaml"
    manifest_path = root / "result_material_event" / "run.json"
    _verify_file(prepare_path, report.get("prepare_report_sha256"), "candidate prepare receipt")
    _verify_file(case_path, report.get("case_sha256"), "candidate case")
    if _sha256(case_path) != LOCKED_CANDIDATE_CASE_SHA256:
        raise ValueError("candidate case differs from its registered SHA-256")
    _verify_file(manifest_path, report.get("result_manifest_sha256"), "candidate manifest")
    prepare, _ = _load_json(prepare_path, "candidate prepare receipt")
    _verify_candidate_prepare(config, config_hash, prepare, case_path)
    producer_source = _verify_candidate_producer(root, prepare, report)
    candidate_config = _mapping(config.get("candidate"), "configuration.candidate")
    lineage = _mapping(report.get("pre_event_lineage"), "candidate pre-event lineage")
    if lineage != {
        "historical_parent_trajectory_sha256": candidate_config.get("parent_trajectory_sha256"),
        "bitwise_parent_equality_required": False,
        "acceptance_owner": "evaluate_m3c1_common_field.py",
    }:
        raise ValueError("candidate pre-event lineage is invalid")
    if report.get("claim_policy") != config.get("claim_policy"):
        raise ValueError("candidate run claim policy differs from the configuration")
    durable_result = open_result(root / "result_material_event")
    manifest = _mapping(dict(durable_result.manifest), "candidate durable manifest")
    _verify_candidate_manifest(config, manifest, report, case_path)
    trajectory = _safe_relative(root, report.get("trajectory"), "candidate trajectory")
    pre_event = _safe_relative(
        root,
        report.get("pre_event_trajectory"),
        "candidate pre-event trajectory",
    )
    events = _safe_relative(root, report.get("events"), "candidate events")
    _verify_file(trajectory, report.get("trajectory_sha256"), "candidate trajectory")
    _verify_file(
        pre_event,
        report.get("pre_event_trajectory_sha256"),
        "candidate pre-event trajectory",
    )
    _verify_file(events, report.get("events_sha256"), "candidate events")
    return (
        report,
        trajectory,
        pre_event,
        events,
        durable_result,
        producer_source,
    )


def _wide_state(path: Path, particle_count: int, times: np.ndarray) -> dict[int, np.ndarray]:
    histories: dict[int, np.ndarray] = {}
    width = len(STATE_COLUMNS)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line in stream:
            if line.startswith("%") or not line.strip():
                continue
            values = np.asarray(next(csv.reader([line])), dtype=np.float64)
            if values.size != times.size * width or not bool(np.isfinite(values).all()):
                raise ValueError(f"{path}: invalid finite wide-row width")
            rows = values.reshape(times.size, width)
            particle_id = _integer(float(rows[0, 0]), "reference particle_id")
            if particle_id in histories or not bool((rows[:, 0] == particle_id).all()):
                raise ValueError(f"{path}: duplicate or changing particle ID {particle_id}")
            if not bool(np.allclose(rows[:, 1], times, rtol=0.0, atol=2e-14)):
                raise ValueError(f"{path}: output schedule differs for particle {particle_id}")
            histories[particle_id] = rows
    if set(histories) != set(range(1, particle_count + 1)):
        raise ValueError(f"{path}: particle IDs must be exactly 1..{particle_count}")
    return histories


def _read_rows(path: Path, columns: tuple[str, ...], name: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"{name} columns are invalid")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{name} contains malformed rows")
    return rows


def _candidate_trajectory(
    path: Path, particle_count: int, times: np.ndarray
) -> dict[tuple[int, float], dict[str, str]]:
    rows = _read_rows(path, TRAJECTORY_COLUMNS, "candidate trajectory")
    expected = particle_count * times.size
    if len(rows) != expected:
        raise ValueError(f"candidate trajectory has {len(rows)} rows, expected {expected}")
    result: dict[tuple[int, float], dict[str, str]] = {}
    valid_times = {float(value) for value in times}
    for row in rows:
        particle_id = int(row["particle_id"])
        time_s = float(row["time_s"])
        values = [float(row[column]) for column in TRAJECTORY_COLUMNS[2:7]]
        if (
            particle_id not in range(1, particle_count + 1)
            or time_s not in valid_times
            or not all(math.isfinite(value) for value in values)
            or row["lifecycle"] not in {"pending", "active", "stuck", "escaped", "failed"}
        ):
            raise ValueError("candidate trajectory contains an invalid state row")
        key = (particle_id, time_s)
        if key in result:
            raise ValueError(f"candidate trajectory contains duplicate key {key}")
        result[key] = row
    if len(result) != expected:
        raise ValueError("candidate trajectory key grid is incomplete")
    return result


def _candidate_event(path: Path) -> dict[str, str]:
    rows = _read_rows(path, EVENT_COLUMNS, "candidate events")
    if len(rows) != 1:
        raise ValueError(f"candidate must contain exactly one event, found {len(rows)}")
    numeric = [*EVENT_COLUMNS[:15]]
    if not all(math.isfinite(float(rows[0][name])) for name in numeric):
        raise ValueError("candidate event contains a nonfinite value")
    return rows[0]


def _same_float64(text: str, value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        return False
    observed = np.asarray(float(text), dtype="<f8")
    expected = np.asarray(float(value), dtype="<f8")
    return observed.tobytes() == expected.tobytes()


def _verify_durable_trajectory(
    durable_result: Any, candidate: dict[tuple[int, float], dict[str, str]]
) -> int:
    seen: set[tuple[int, float]] = set()
    for frame in durable_result.iter_frames():
        for row_index, particle_id_raw in enumerate(frame.particle_id):
            key = (int(particle_id_raw), float(frame.time_s))
            exported = candidate.get(key)
            if exported is None or key in seen:
                raise ValueError("candidate trajectory is not an exact durable-result projection")
            expected_values = (
                frame.position_m[row_index, 0],
                frame.position_m[row_index, 1],
                frame.velocity_m_s[row_index, 0],
                frame.velocity_m_s[row_index, 1],
                frame.charge_number[row_index],
            )
            if not all(
                _same_float64(exported[column], expected)
                for column, expected in zip(TRAJECTORY_COLUMNS[2:7], expected_values, strict=True)
            ):
                raise ValueError("candidate trajectory values differ from the durable result")
            if (
                exported["lifecycle"]
                != ("pending", "active", "stuck", "escaped", "failed")[
                    int(frame.lifecycle[row_index])
                ]
            ):
                raise ValueError("candidate trajectory lifecycle differs from the durable result")
            seen.add(key)
    if seen != set(candidate):
        raise ValueError("candidate trajectory key grid differs from the durable result")
    return len(seen)


def _verify_durable_event(durable_result: Any, event: dict[str, str]) -> None:
    events = durable_result.read_boundary_events()
    if int(events.particle_id.size) != 1:
        raise ValueError("durable result must contain exactly one boundary event")
    integer_fields = {
        "particle_id": events.particle_id[0],
        "event_ordinal": events.event_ordinal[0],
        "primary_facet_id": events.primary_facet_id[0],
        "boundary_id": events.boundary_id[0],
    }
    if any(int(event[name]) != int(value) for name, value in integer_fields.items()):
        raise ValueError("candidate event identity differs from the durable result")
    numeric_fields = {
        "event_time_s": events.time_s[0],
        "hit_r_m": events.position_m[0, 0],
        "hit_z_m": events.position_m[0, 1],
        "normal_r": events.normal[0, 0],
        "normal_z": events.normal[0, 1],
        "pre_velocity_r_m_per_s": events.velocity_pre_m_s[0, 0],
        "pre_velocity_z_m_per_s": events.velocity_pre_m_s[0, 1],
        "post_velocity_r_m_per_s": events.velocity_post_m_s[0, 0],
        "post_velocity_z_m_per_s": events.velocity_post_m_s[0, 1],
        "charge_number_pre_e": events.charge_number_pre[0],
        "charge_number_post_e": events.charge_number_post[0],
    }
    if any(not _same_float64(event[name], value) for name, value in numeric_fields.items()):
        raise ValueError("candidate event values differ from the durable result")
    if event["law"] != str(events.law_id[0]) or event["outcome"] != str(events.outcome[0]):
        raise ValueError("candidate event response differs from the durable result")


def _verify_durable_projection(
    durable_result: Any,
    candidate: dict[tuple[int, float], dict[str, str]],
    event: dict[str, str],
) -> dict[str, int]:
    trajectory_rows = _verify_durable_trajectory(durable_result, candidate)
    _verify_durable_event(durable_result, event)
    failures = durable_result.read_failure_events()
    if int(failures.particle_id.size) != 0:
        raise ValueError("durable result contains a particle failure")
    return {"trajectory_rows": trajectory_rows, "event_rows": 1, "failure_rows": 0}


def _verify_pre_event_projection(
    path: Path,
    candidate: dict[tuple[int, float], dict[str, str]],
    particle_count: int,
    times: np.ndarray,
) -> dict[str, object]:
    prefix = _candidate_trajectory(path, particle_count, times[:46])
    if any(candidate.get(key) != row for key, row in prefix.items()):
        raise ValueError("pre-event trajectory is not an exact subset of the full trajectory")
    return {"rows": len(prefix), "frames": 46, "sha256": _sha256(path)}


def _common_field_gates(
    metrics: dict[str, object], config: common_field._Config
) -> dict[str, object]:
    gates: dict[str, object] = {}
    for quantity in ("position", "velocity", "charge"):
        values = common_shared._mapping(metrics.get(quantity), f"{quantity} metrics")
        quantity_gates: dict[str, object] = {}
        for metric in ("rms", "maximum"):
            observed = common_shared._number(values.get(metric), f"{quantity}.{metric}")
            tolerance = config.absolute_limits[quantity][metric]
            quantity_gates[metric] = {
                "observed": observed,
                "tolerance": tolerance,
                "pass": observed <= tolerance,
            }
        relative = values.get("relative_l2")
        tolerance = config.metrics.relative_limits[quantity]
        passed = (
            isinstance(relative, int | float)
            and not isinstance(relative, bool)
            and math.isfinite(float(relative))
            and float(relative) <= tolerance
        )
        quantity_gates["relative_l2"] = {
            "observed": relative,
            "tolerance": tolerance,
            "pass": passed,
        }
        gates[quantity] = quantity_gates
    return gates


def _passed_common_field_gates(gates: dict[str, object]) -> int:
    return sum(
        _mapping(_mapping(gates.get(quantity), quantity).get(metric), f"{quantity}.{metric}").get(
            "pass"
        )
        is True
        for quantity in ("position", "velocity", "charge")
        for metric in ("rms", "maximum", "relative_l2")
    )


def _verified_common_artifact(
    artifacts: dict[str, Any], side: str, name: str
) -> tuple[Path, dict[str, Any]]:
    artifact = _mapping(artifacts.get(side), f"{name} {side} artifact")
    path = Path(str(artifact.get("path"))).resolve()
    _verify_file(path, artifact.get("sha256"), f"{name} {side} trajectory")
    return path, artifact


def _verify_common_budget(
    comparison: dict[str, Any], config: common_field._Config
) -> dict[str, object]:
    budget_identity = _mapping(comparison.get("budget"), "locked comparison budget")
    budget_path = Path(str(budget_identity.get("path"))).resolve()
    _verify_file(
        budget_path,
        budget_identity.get("sha256"),
        "locked pre-event comparison budget",
    )
    budget, budget_hash = _load_json(budget_path, "locked pre-event comparison budget")
    expected_identity = {
        "schema_version": 1,
        "tool_revision": common_field.TOOL_REVISION,
        "report_kind": "m3c1_preregistered_common_field_budget",
        "status": "REGISTERED",
        "blockers": [],
        "scope": comparison.get("scope"),
        "evaluation_config": comparison.get("evaluation_config"),
    }
    if any(budget.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("locked pre-event budget identity or scope is invalid")
    expected_acceptance = {
        "absolute_limits": config.absolute_limits,
        "relative_l2_limits": config.metrics.relative_limits,
    }
    if budget.get("acceptance") != expected_acceptance:
        raise ValueError("locked pre-event budget bounds differ from the evaluation config")
    if budget.get("fine_trajectory_artifacts") != comparison.get("artifacts"):
        raise ValueError("locked pre-event budget and comparison artifacts differ")
    policy = _mapping(budget.get("policy"), "locked pre-event budget policy")
    if policy.get("result_dependent_tolerance_tuning") != "PROHIBITED":
        raise ValueError("locked pre-event budget permits result-dependent tolerance tuning")
    initial_gate = _mapping(budget.get("initial_state_gate"), "locked initial-state gate")
    if initial_gate.get("status") != "PASS":
        raise ValueError("locked pre-event budget initial-state gate is not PASS")
    return {"path": str(budget_path), "sha256": budget_hash}


def _verify_locked_comparison(
    comparison: dict[str, Any], comparison_hash: str
) -> tuple[common_field._Config, Path, dict[str, Any], dict[str, object]]:
    expected_identity = {
        "schema_version": 1,
        "tool_revision": common_field.TOOL_REVISION,
        "report_kind": "m3c1_locked_common_field_trajectory_comparison",
        "status": "PASS",
        "blockers": [],
    }
    if any(comparison.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("locked pre-event comparison identity is invalid")
    claim = _mapping(comparison.get("claim_separation"), "locked comparison claim separation")
    if claim.get("same_field_solver_agreement") != "PASS":
        raise ValueError("locked pre-event comparison does not claim same-field agreement")
    evaluation = _mapping(comparison.get("evaluation_config"), "locked evaluation config")
    evaluation_path = Path(str(evaluation.get("path"))).resolve()
    _verify_file(
        evaluation_path,
        evaluation.get("sha256"),
        "locked pre-event evaluation configuration",
    )
    field_config = common_field._load_config(evaluation_path)
    if comparison.get("scope") != field_config.metrics.raw.get("scope"):
        raise ValueError("locked pre-event comparison scope differs from its evaluation config")
    budget = _verify_common_budget(comparison, field_config)
    artifacts = _mapping(comparison.get("artifacts"), "locked comparison artifacts")
    historical_path, _ = _verified_common_artifact(artifacts, "candidate", "locked comparison")
    reference_path, reference_artifact = _verified_common_artifact(
        artifacts, "reference", "locked comparison"
    )
    historical = common_shared._read_trajectory(historical_path, field_config.metrics)
    reference = common_shared._read_trajectory(reference_path, field_config.metrics)
    metrics = common_shared._difference_metrics(
        historical.values,
        reference.values,
        field_config.metrics.output_interval_s,
    )
    gates = _common_field_gates(metrics, field_config)
    if comparison.get("trajectory_difference") != metrics:
        raise ValueError("locked pre-event trajectory_difference is not reproducible")
    if comparison.get("acceptance_gates") != gates:
        raise ValueError("locked pre-event acceptance gates are not reproducible")
    if _passed_common_field_gates(gates) != 9:
        raise ValueError("locked pre-event comparison does not independently pass nine gates")
    authority: dict[str, object] = {
        "budget": budget,
        "historical_candidate": {
            "path": str(historical.path),
            "sha256": historical.sha256,
        },
        "comparison_sha256": comparison_hash,
    }
    return field_config, reference_path, reference_artifact, authority


def _verify_pre_event_comparison(
    config: dict[str, Any], comparison_path: Path, prefix_path: Path
) -> dict[str, object]:
    reference_config = _mapping(config.get("reference"), "configuration.reference")
    locked_path = (
        _solver_root() / str(reference_config.get("pre_event_comparison_relative_path"))
    ).resolve()
    supplied_path = comparison_path.resolve()
    if supplied_path != locked_path:
        raise ValueError("pre-event comparison must be the registered locked authority")
    expected_hash = reference_config.get("pre_event_comparison_sha256")
    _verify_file(locked_path, expected_hash, "locked pre-event comparison")
    comparison, comparison_hash = _load_json(locked_path, "locked pre-event comparison")
    field_config, reference_path, reference_artifact, authority = _verify_locked_comparison(
        comparison, comparison_hash
    )
    prefix = common_shared._read_trajectory(prefix_path, field_config.metrics)
    reference = common_shared._read_trajectory(reference_path, field_config.metrics)
    metrics = common_shared._difference_metrics(
        prefix.values,
        reference.values,
        field_config.metrics.output_interval_s,
    )
    gates = _common_field_gates(metrics, field_config)
    passed = _passed_common_field_gates(gates)
    return {
        "status": "PASS" if passed == 9 else "FAIL",
        "passed_gates": passed,
        "authority": {
            "comparison": {"path": str(locked_path), "sha256": comparison_hash},
            **authority,
        },
        "artifacts": {
            "candidate": {"path": str(prefix.path), "sha256": prefix.sha256},
            "reference": reference_artifact,
        },
        "trajectory_difference": metrics,
        "acceptance_gates": gates,
    }


def _candidate_boundary_identity(
    config: dict[str, Any], event: dict[str, str]
) -> dict[str, object]:
    candidate = _mapping(config.get("candidate"), "configuration.candidate")
    parent_root = (_solver_root() / str(candidate.get("parent_root_relative_path"))).resolve()
    input_path = _safe_relative(parent_root, candidate.get("input_filename"), "candidate input")
    _verify_file(input_path, candidate.get("input_sha256"), "candidate input")
    facet = int(event["primary_facet_id"])
    with h5py.File(input_path, "r") as handle:
        external = np.asarray(handle["geometry/boundary/external_id"], dtype=np.int64)
        boundary = np.asarray(handle["geometry/boundary/boundary_id"], dtype=np.int64)
        group_id = np.asarray(handle["geometry/boundary/group_id"], dtype=np.int64)
        line2 = np.asarray(handle["geometry/boundary/line2"], dtype=np.int64)
        nodes = np.asarray(handle["geometry/nodes_m"], dtype=np.float64)
        names_raw = np.asarray(handle["geometry/groups/names"])
    if facet < 0 or facet >= external.size:
        raise ValueError("candidate event primary facet is outside canonical geometry")
    group_index = int(group_id[facet])
    if group_index < 0 or group_index >= names_raw.size:
        raise ValueError("candidate event group index is outside canonical geometry")
    raw_name = names_raw[group_index]
    name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
    segment_nodes = line2[facet]
    if (
        segment_nodes.shape != (2,)
        or bool((segment_nodes < 0).any())
        or bool((segment_nodes >= nodes.shape[0]).any())
    ):
        raise ValueError("candidate event facet has invalid line connectivity")
    segment = nodes[segment_nodes]
    if segment.shape != (2, 2) or not bool(np.isfinite(segment).all()):
        raise ValueError("candidate event facet has invalid node coordinates")
    return {
        "primary_facet_id": facet,
        "canonical_boundary_id": int(boundary[facet]),
        "canonical_external_id": int(external[facet]),
        "canonical_boundary_group": name,
        "canonical_segment_m": segment,
        "canonical_input_sha256": _sha256(input_path),
    }


def _distance_to_segment(point: np.ndarray, segment: np.ndarray) -> float:
    direction = segment[1] - segment[0]
    denominator = float(np.dot(direction, direction))
    if denominator <= 0.0:
        raise ValueError("canonical event facet is degenerate")
    fraction = float(np.dot(point - segment[0], direction) / denominator)
    closest = segment[0] + min(1.0, max(0.0, fraction)) * direction
    return float(np.linalg.norm(point - closest))


def _hold(values: Iterable[float], multiplier: float) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValueError("terminal hold values must be finite and nonempty")
    span = float(np.max(array) - np.min(array))
    scale = float(np.max(np.abs(array)))
    limit = multiplier * math.ulp(scale)
    return {"span": span, "limit": limit}


def _position_hold(points: Iterable[tuple[float, float]]) -> float:
    array = np.asarray(tuple(points), dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 2:
        raise ValueError("terminal positions must be a nonempty collection of 2-vectors")
    return float(np.max(np.linalg.norm(array - array[0], axis=1), initial=0.0))


def _gate(name: str, passed: bool, observed: object, limit: object) -> dict[str, object]:
    return {
        "gate": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "limit": limit,
    }


def _reference_transition(
    history: np.ndarray, expected_status: int
) -> tuple[bool, bool, float, float]:
    status = tuple(_integer(float(value), "reference current status") for value in history[:, 7])
    first_terminal = next((index for index, value in enumerate(status) if value != 1), None)
    if first_terminal is None or first_terminal == 0:
        return False, False, math.nan, float(history[0, 1])
    transition_ok = all(value == 1 for value in status[:first_terminal]) and all(
        value == expected_status for value in status[first_terminal:]
    )
    last_active_time_s = float(history[first_terminal - 1, 1])
    first_terminal_time_s = float(history[first_terminal, 1])
    event_time_s = float(history[first_terminal, 9])
    return (
        transition_ok,
        last_active_time_s <= event_time_s <= first_terminal_time_s,
        last_active_time_s,
        first_terminal_time_s,
    )


def _reference_event(config: dict[str, Any], reference: dict[int, np.ndarray]) -> _ReferenceEvent:
    expected = _mapping(config.get("expected_first_event"), "expected_first_event")
    acceptance = _mapping(config.get("acceptance"), "acceptance")
    particle_id = int(expected["particle_id"])
    nonactive = {
        key: history[history[:, 7] != 1.0]
        for key, history in reference.items()
        if bool((history[:, 7] != 1.0).any())
    }
    terminal = nonactive.get(particle_id)
    if terminal is None:
        raise ValueError("reference has no terminal state for the expected event particle")
    transition_ok, event_time_bracket_ok, last_active_time_s, first_terminal_time_s = (
        _reference_transition(reference[particle_id], int(expected["comsol_status_code"]))
    )
    event_times = terminal[:, 9]
    event_time_hold = _hold(event_times, 4096.0)
    if event_time_hold["span"] > event_time_hold["limit"]:
        raise ValueError("reference stop/event time changes across terminal frames")
    final_codes = tuple(
        sorted(
            {
                _integer(float(value), "reference final status")
                for value in reference[particle_id][reference[particle_id][:, 7] == 1.0, 8]
            }
        )
    )
    return _ReferenceEvent(
        particle_id=particle_id,
        nonactive_particle_ids=tuple(sorted(nonactive)),
        terminal=terminal,
        event_time_s=float(event_times[0]),
        hit_position_m=terminal[0, 2:4],
        position_hold_m=_position_hold((float(row[2]), float(row[3])) for row in terminal),
        charge_hold=_hold(
            terminal[:, 6], float(acceptance["terminal_charge_hold_roundoff_multiplier"])
        ),
        terminal_charge_e=float(terminal[-1, 6]),
        current_status_codes=tuple(
            sorted({_integer(float(value), "reference current status") for value in terminal[:, 7]})
        ),
        transition_ok=transition_ok,
        event_time_bracket_ok=event_time_bracket_ok,
        last_active_time_s=last_active_time_s,
        first_terminal_time_s=first_terminal_time_s,
        pre_event_final_status_codes=final_codes,
    )


def _active_to_stuck(history: tuple[str, ...]) -> bool:
    first_stuck = next((index for index, value in enumerate(history) if value == "stuck"), None)
    return (
        first_stuck is not None
        and first_stuck > 0
        and all(value == "active" for value in history[:first_stuck])
        and all(value == "stuck" for value in history[first_stuck:])
    )


def _candidate_transition(
    particle_rows: tuple[dict[str, str], ...], event_time_s: float
) -> tuple[tuple[str, ...], bool, bool, float, float]:
    history = tuple(row["lifecycle"] for row in particle_rows)
    first_stuck = next((index for index, value in enumerate(history) if value == "stuck"), None)
    if first_stuck is None or first_stuck == 0:
        return history, False, False, math.nan, float(particle_rows[0]["time_s"])
    last_active_time_s = float(particle_rows[first_stuck - 1]["time_s"])
    first_stuck_time_s = float(particle_rows[first_stuck]["time_s"])
    return (
        history,
        _active_to_stuck(history),
        last_active_time_s <= event_time_s <= first_stuck_time_s,
        last_active_time_s,
        first_stuck_time_s,
    )


def _candidate_event_observation(
    config: dict[str, Any],
    candidate: dict[tuple[int, float], dict[str, str]],
    event: dict[str, str],
    boundary: dict[str, object],
) -> _CandidateEvent:
    expected = _mapping(config.get("expected_first_event"), "expected_first_event")
    acceptance = _mapping(config.get("acceptance"), "acceptance")
    particle_id = int(expected["particle_id"])
    particle_rows = tuple(
        row for (row_particle, _), row in sorted(candidate.items()) if row_particle == particle_id
    )
    event_time_s = float(event["event_time_s"])
    history, transition_ok, event_time_bracket_ok, last_active_time_s, first_stuck_time_s = (
        _candidate_transition(particle_rows, event_time_s)
    )
    terminal = tuple(
        row
        for (row_particle, _), row in sorted(candidate.items())
        if row_particle == particle_id and row["lifecycle"] == "stuck"
    )
    if not terminal:
        raise ValueError("candidate trajectory has no saved stuck state")
    identity = {
        "particle_id": int(event["particle_id"]),
        "event_ordinal": int(event["event_ordinal"]),
        "law": event["law"],
        "outcome": event["outcome"],
        "event_boundary_id": int(event["boundary_id"]),
        **{key: value for key, value in boundary.items() if key != "canonical_segment_m"},
    }
    expected_identity = {
        "particle_id": particle_id,
        "event_ordinal": 1,
        "law": expected["candidate_law"],
        "outcome": expected["candidate_lifecycle"],
        "event_boundary_id": expected["candidate_boundary_id"],
        "canonical_boundary_id": expected["candidate_boundary_id"],
        "canonical_external_id": expected["candidate_external_id"],
        "canonical_boundary_group": expected["boundary_group"],
    }
    hit = np.asarray([float(event["hit_r_m"]), float(event["hit_z_m"])])
    segment = np.asarray(boundary["canonical_segment_m"], dtype=np.float64)
    nonactive_particle_ids = tuple(
        sorted(
            {
                row_particle
                for (row_particle, _), row in candidate.items()
                if row["lifecycle"] != "active"
            }
        )
    )
    charge_values = [
        float(event["charge_number_pre_e"]),
        float(event["charge_number_post_e"]),
        *(float(row["charge_number_e"]) for row in terminal),
    ]
    event_post_velocity = np.asarray(
        [
            float(event["post_velocity_r_m_per_s"]),
            float(event["post_velocity_z_m_per_s"]),
        ],
        dtype=np.float64,
    )
    stuck_velocity = np.asarray(
        [(float(row["velocity_r_m_per_s"]), float(row["velocity_z_m_per_s"])) for row in terminal],
        dtype=np.float64,
    )
    return _CandidateEvent(
        identity=identity,
        identity_matches=all(
            identity.get(key) == value for key, value in expected_identity.items()
        ),
        terminal=terminal,
        event_time_s=event_time_s,
        hit_position_m=hit,
        position_hold_m=_position_hold(
            [
                (float(event["hit_r_m"]), float(event["hit_z_m"])),
                *((float(row["r_m"]), float(row["z_m"])) for row in terminal),
            ]
        ),
        charge_hold=_hold(
            charge_values, float(acceptance["terminal_charge_hold_roundoff_multiplier"])
        ),
        terminal_charge_e=float(terminal[-1]["charge_number_e"]),
        lifecycle_history=history,
        transition_ok=transition_ok,
        event_time_bracket_ok=event_time_bracket_ok,
        last_active_time_s=last_active_time_s,
        first_stuck_time_s=first_stuck_time_s,
        nonactive_particle_ids=nonactive_particle_ids,
        canonical_segment_m=segment,
        canonical_surface_ok=bool(
            np.allclose(
                segment[:, 1],
                float(expected["wafer_z_m"]),
                rtol=0.0,
                atol=float(acceptance["terminal_position_hold_m"]),
            )
        ),
        hit_distance_to_segment_m=_distance_to_segment(hit, segment),
        event_post_velocity_m_s=event_post_velocity,
        stuck_velocity_maximum_absolute_m_s=float(np.max(np.abs(stuck_velocity))),
        stuck_velocity_row_count=int(stuck_velocity.shape[0]),
    )


def _cross_differences(reference: _ReferenceEvent, candidate: _CandidateEvent) -> dict[str, float]:
    return {
        "event_time_absolute_s": abs(candidate.event_time_s - reference.event_time_s),
        "hit_position_norm_m": float(
            np.linalg.norm(candidate.hit_position_m - reference.hit_position_m)
        ),
        "terminal_charge_absolute_e": abs(
            candidate.terminal_charge_e - reference.terminal_charge_e
        ),
    }


def _event_gates(
    config: dict[str, Any],
    reference: _ReferenceEvent,
    candidate: _CandidateEvent,
    differences: dict[str, float],
) -> list[dict[str, object]]:
    expected = _mapping(config.get("expected_first_event"), "expected_first_event")
    acceptance = _mapping(config.get("acceptance"), "acceptance")
    expected_identity = {
        "particle_id": reference.particle_id,
        "event_ordinal": 1,
        "law": expected["candidate_law"],
        "outcome": expected["candidate_lifecycle"],
        "event_boundary_id": expected["candidate_boundary_id"],
        "canonical_boundary_id": expected["candidate_boundary_id"],
        "canonical_external_id": expected["candidate_external_id"],
        "canonical_boundary_group": expected["boundary_group"],
    }
    position_limit = float(acceptance["terminal_position_hold_m"])
    reference_surface_distance = _distance_to_segment(
        reference.hit_position_m, candidate.canonical_segment_m
    )
    return [
        _gate(
            "reference_exactly_one_nonactive_stuck_particle",
            reference.nonactive_particle_ids == (reference.particle_id,)
            and reference.current_status_codes == (int(expected["comsol_status_code"]),)
            and reference.transition_ok,
            {
                "particle_ids": reference.nonactive_particle_ids,
                "current_status_codes": reference.current_status_codes,
            },
            {"particle_ids": [reference.particle_id], "current_status_codes": [3]},
        ),
        _gate(
            "reference_event_time_bracketed_by_saved_status_transition",
            reference.event_time_bracket_ok,
            {
                "last_active_time_s": reference.last_active_time_s,
                "event_time_s": reference.event_time_s,
                "first_terminal_time_s": reference.first_terminal_time_s,
            },
            "last active saved time <= fptas.st <= first terminal saved time",
        ),
        _gate(
            "candidate_first_event_identity",
            candidate.identity_matches,
            candidate.identity,
            expected_identity,
        ),
        _gate(
            "candidate_exactly_one_nonactive_stuck_particle",
            candidate.nonactive_particle_ids == (reference.particle_id,),
            candidate.nonactive_particle_ids,
            [reference.particle_id],
        ),
        _gate(
            "candidate_active_to_stuck_transition",
            candidate.transition_ok,
            candidate.lifecycle_history,
            "one active-to-stuck transition with a terminal stuck tail",
        ),
        _gate(
            "candidate_event_time_bracketed_by_saved_lifecycle_transition",
            candidate.event_time_bracket_ok,
            {
                "last_active_time_s": candidate.last_active_time_s,
                "event_time_s": candidate.event_time_s,
                "first_stuck_time_s": candidate.first_stuck_time_s,
            },
            "last active saved time <= event time <= first stuck saved time",
        ),
        _gate(
            "candidate_stick_event_post_velocity_is_zero",
            bool(np.all(candidate.event_post_velocity_m_s == 0.0)),
            candidate.event_post_velocity_m_s.tolist(),
            {"maximum_absolute_component_m_per_s": 0.0},
        ),
        _gate(
            "candidate_all_saved_stuck_velocities_are_zero",
            candidate.stuck_velocity_maximum_absolute_m_s == 0.0,
            {
                "maximum_absolute_component_m_per_s": (
                    candidate.stuck_velocity_maximum_absolute_m_s
                ),
                "saved_stuck_rows": candidate.stuck_velocity_row_count,
            },
            {"maximum_absolute_component_m_per_s": 0.0},
        ),
        _gate(
            "canonical_event_facet_is_expected_wafer_surface",
            candidate.canonical_surface_ok,
            candidate.canonical_segment_m.tolist(),
            {"z_m": float(expected["wafer_z_m"]), "absolute_m": position_limit},
        ),
        _gate(
            "reference_hit_on_canonical_event_facet",
            reference_surface_distance <= position_limit,
            reference_surface_distance,
            position_limit,
        ),
        _gate(
            "candidate_hit_on_canonical_event_facet",
            candidate.hit_distance_to_segment_m <= position_limit,
            candidate.hit_distance_to_segment_m,
            position_limit,
        ),
        _gate(
            "event_time_absolute_difference",
            differences["event_time_absolute_s"] <= float(acceptance["event_time_absolute_s"]),
            differences["event_time_absolute_s"],
            float(acceptance["event_time_absolute_s"]),
        ),
        _gate(
            "hit_position_norm_difference",
            differences["hit_position_norm_m"] <= float(acceptance["hit_position_norm_m"]),
            differences["hit_position_norm_m"],
            float(acceptance["hit_position_norm_m"]),
        ),
        _gate(
            "reference_terminal_position_hold",
            reference.position_hold_m <= position_limit,
            reference.position_hold_m,
            position_limit,
        ),
        _gate(
            "candidate_terminal_position_hold",
            candidate.position_hold_m <= position_limit,
            candidate.position_hold_m,
            position_limit,
        ),
        _gate(
            "terminal_charge_absolute_difference",
            differences["terminal_charge_absolute_e"]
            <= float(acceptance["terminal_charge_cross_absolute_e"]),
            differences["terminal_charge_absolute_e"],
            float(acceptance["terminal_charge_cross_absolute_e"]),
        ),
        _gate(
            "reference_terminal_charge_hold",
            reference.charge_hold["span"] <= reference.charge_hold["limit"],
            reference.charge_hold["span"],
            reference.charge_hold["limit"],
        ),
        _gate(
            "candidate_terminal_charge_hold",
            candidate.charge_hold["span"] <= candidate.charge_hold["limit"],
            candidate.charge_hold["span"],
            candidate.charge_hold["limit"],
        ),
    ]


def _event_report(
    event: dict[str, str],
    reference: _ReferenceEvent,
    candidate: _CandidateEvent,
    differences: dict[str, float],
) -> dict[str, object]:
    return {
        "reference": {
            "particle_id": reference.particle_id,
            "event_time_s": reference.event_time_s,
            "first_saved_nonactive_time_s": float(reference.terminal[0, 1]),
            "hit_position_m": reference.hit_position_m.tolist(),
            "terminal_charge_e": reference.terminal_charge_e,
            "terminal_velocity_first_m_s": reference.terminal[0, 4:6].tolist(),
            "terminal_velocity_last_m_s": reference.terminal[-1, 4:6].tolist(),
            "terminal_saved_frames": int(reference.terminal.shape[0]),
            "last_active_saved_time_s": reference.last_active_time_s,
            "first_terminal_saved_time_s": reference.first_terminal_time_s,
            "current_status_codes": reference.current_status_codes,
            "pre_event_final_status_codes": reference.pre_event_final_status_codes,
        },
        "candidate": {
            **candidate.identity,
            "canonical_segment_m": candidate.canonical_segment_m.tolist(),
            "event_time_s": candidate.event_time_s,
            "hit_position_m": candidate.hit_position_m.tolist(),
            "terminal_charge_e": candidate.terminal_charge_e,
            "event_pre_velocity_m_s": [
                float(event["pre_velocity_r_m_per_s"]),
                float(event["pre_velocity_z_m_per_s"]),
            ],
            "event_post_velocity_m_s": [
                *candidate.event_post_velocity_m_s.tolist(),
            ],
            "terminal_velocity_last_m_s": [
                float(candidate.terminal[-1]["velocity_r_m_per_s"]),
                float(candidate.terminal[-1]["velocity_z_m_per_s"]),
            ],
            "terminal_saved_frames": len(candidate.terminal),
            "terminal_velocity_maximum_absolute_component_m_per_s": (
                candidate.stuck_velocity_maximum_absolute_m_s
            ),
            "last_active_saved_time_s": candidate.last_active_time_s,
            "first_stuck_saved_time_s": candidate.first_stuck_time_s,
        },
        "cross_differences": differences,
        "own_terminal_holds": {
            "reference_position_m": reference.position_hold_m,
            "candidate_position_m": candidate.position_hold_m,
            "reference_charge_e": reference.charge_hold,
            "candidate_charge_e": candidate.charge_hold,
        },
        "status_semantics": {
            "event_detection_authority": "current_status_code",
            "final_status_code_role": "characterization_only",
        },
        "velocity_semantics": {
            "cross_solver_gate": False,
            "reason": "COMSOL Freeze storage and solver stick-state velocity are distinct contracts",
        },
    }


def _event_metrics(
    config: dict[str, Any],
    reference_rows: dict[int, np.ndarray],
    candidate_rows: dict[tuple[int, float], dict[str, str]],
    event: dict[str, str],
    boundary: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    reference = _reference_event(config, reference_rows)
    candidate = _candidate_event_observation(config, candidate_rows, event, boundary)
    differences = _cross_differences(reference, candidate)
    return (
        _event_report(event, reference, candidate, differences),
        _event_gates(config, reference, candidate, differences),
    )


def _write_gates(path: Path, gates: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("gate", "status", "observed", "limit"), lineterminator="\n"
        )
        writer.writeheader()
        for gate in gates:
            writer.writerow(
                {
                    **gate,
                    "observed": json.dumps(gate["observed"], sort_keys=True),
                    "limit": json.dumps(gate["limit"], sort_keys=True),
                }
            )


def evaluate(
    config_path: Path,
    candidate_root: Path,
    pre_event_comparison_path: Path,
    reference_receipt_path: Path,
    reference_validation_path: Path,
    reference_state_path: Path,
    reference_ledger_path: Path,
    reference_status_path: Path,
    output: Path,
) -> dict[str, object]:
    """Evaluate one locked first-material-event pair and write compact evidence."""

    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    config, config_hash = _load_config(config_path.resolve())
    reference_provenance = _reference_provenance(
        config,
        config_hash,
        reference_receipt_path.resolve(),
        reference_validation_path.resolve(),
        reference_state_path.resolve(),
        reference_ledger_path.resolve(),
        reference_status_path.resolve(),
    )
    candidate_root = candidate_root.resolve()
    report, trajectory_path, pre_event_path, event_path, durable_result, producer_source = (
        _candidate_provenance(config, config_hash, candidate_root)
    )
    case = _mapping(config.get("case"), "configuration.case")
    times = np.asarray(case["output_times_s"], dtype=np.float64)
    reference = _wide_state(reference_state_path.resolve(), int(case["particle_count"]), times)
    candidate = _candidate_trajectory(trajectory_path, int(case["particle_count"]), times)
    event = _candidate_event(event_path)
    durable_projection = _verify_durable_projection(durable_result, candidate, event)
    prefix_projection = _verify_pre_event_projection(
        pre_event_path,
        candidate,
        int(case["particle_count"]),
        times,
    )
    same_field = _verify_pre_event_comparison(config, pre_event_comparison_path, pre_event_path)
    expected_prefix_rows = int(case["particle_count"]) * 46
    if (
        report.get("trajectory_rows") != len(candidate)
        or report.get("event_rows") != 1
        or report.get("pre_event_trajectory_rows") != expected_prefix_rows
    ):
        raise ValueError("candidate CSV row counts differ from the run receipt")
    reported_first = _mapping(report.get("first_event"), "candidate first_event receipt")
    expected_reported_first = {
        "particle_id": int(event["particle_id"]),
        "event_time_s": float(event["event_time_s"]),
        "hit_position_m": [float(event["hit_r_m"]), float(event["hit_z_m"])],
        "velocity_pre_m_s": [
            float(event["pre_velocity_r_m_per_s"]),
            float(event["pre_velocity_z_m_per_s"]),
        ],
        "velocity_post_m_s": [
            float(event["post_velocity_r_m_per_s"]),
            float(event["post_velocity_z_m_per_s"]),
        ],
        "charge_number_pre_e": float(event["charge_number_pre_e"]),
        "charge_number_post_e": float(event["charge_number_post_e"]),
        "primary_facet_id": int(event["primary_facet_id"]),
        "boundary_id": int(event["boundary_id"]),
        "law": event["law"],
        "outcome": event["outcome"],
    }
    if reported_first != expected_reported_first:
        raise ValueError("candidate first-event receipt differs from its event ledger")
    boundary = _candidate_boundary_identity(config, event)
    event_metrics, gates = _event_metrics(config, reference, candidate, event, boundary)
    manifest = _mapping(dict(durable_result.manifest), "candidate durable manifest")
    manifest_hash = _sha256(candidate_root / "result_material_event" / "run.json")
    zero_failures = (
        report.get("failure_event_count") == 0
        and _mapping(manifest.get("counts"), "candidate counts").get("failure_events") == 0
        and _mapping(manifest.get("lifecycle_counts"), "candidate lifecycle counts").get("failed")
        == 0
    )
    gates[:0] = [
        _gate("candidate_zero_failures", zero_failures, report.get("failure_event_count"), 0),
        _gate(
            "candidate_current_pre_event_same_field_agreement",
            same_field["passed_gates"] == 9,
            {"status": same_field["status"], "passed_gates": same_field["passed_gates"]},
            {"passed_gates": 9},
        ),
    ]
    status = "PASS" if all(gate["status"] == "PASS" for gate in gates) else "FAIL"
    evaluator_source = {
        "file": EVALUATOR_SOURCE_FILENAME,
        "sha256": _sha256(Path(__file__).resolve()),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "evaluation_id": EVALUATION_ID,
        "status": status,
        "configuration": str(config_path.resolve()),
        "configuration_sha256": config_hash,
        "scope": {
            "particles": int(case["particle_count"]),
            "frames": len(times),
            "time_window_s": [float(times[0]), float(times[-1])],
            "field_representation": "common_canonical_exact_connectivity_P1",
        },
        "provenance": {
            "reference": reference_provenance,
            "candidate": {
                "root": str(candidate_root),
                "run_report_sha256": _sha256(candidate_root / "candidate_run_report.json"),
                "manifest_sha256": manifest_hash,
                "trajectory_sha256": _sha256(trajectory_path),
                "pre_event_trajectory_sha256": _sha256(pre_event_path),
                "events_sha256": _sha256(event_path),
                "producer_source": producer_source,
                "evaluator_source": evaluator_source,
                "durable_projection": durable_projection,
                "comparison_authority": (
                    "validated durable public result; CSVs are exact checked projections"
                ),
            },
        },
        "pre_event_prefix": {
            **prefix_projection,
            "same_field_comparison": same_field,
            "historical_parent_bitwise_equality_required": False,
            "comparison_numbers_recomputed_by_material_evaluator": True,
        },
        "material_event": event_metrics,
        "gates": gates,
        "gate_summary": {
            "pass": sum(gate["status"] == "PASS" for gate in gates),
            "fail": sum(gate["status"] == "FAIL" for gate in gates),
        },
        "claim_policy": config["claim_policy"],
    }
    output.mkdir(parents=True)
    if _stage_evaluator_source(output) != evaluator_source:
        raise OSError("staged evaluator source identity differs")
    _write_gates(output / "gates.csv", gates)
    with (output / "comparison_result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pre-event-comparison",
        type=Path,
        required=True,
        help="registered historical comparison that owns the budget and reference identity",
    )
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--reference-ledger", type=Path, required=True)
    parser.add_argument("--reference-status", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = evaluate(
        arguments.config,
        arguments.candidate_root,
        arguments.pre_event_comparison,
        arguments.reference_receipt,
        arguments.reference_validation,
        arguments.reference_state,
        arguments.reference_ledger,
        arguments.reference_status,
        arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
