"""Evaluate the M3-C1 full-physics common-canonical-P1 diagnostic.

Both integrations consume the same canonical exact-connectivity P1 fields.
This external V&V tool binds the solver artifacts to their normal producer
receipts and the COMSOL artifacts to one no-save runner receipt.  Acceptance
tolerances are fixed in the evaluation config before cross differences are
read; no result-derived tolerance is constructed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from tools.vv.comsol import evaluate_m3c1_pre_event as shared

TOOL_REVISION: Final = "m3c1_common_canonical_p1_pre_event_v2"
REPORT_SCHEMA_VERSION: Final = 1
EVALUATION_ID: Final = "M3-C1-caseA-100nm-common-canonical-p1-pre-event"
STEP_NAMES: Final = shared.STEP_NAMES
RUN_KEYS: Final = shared.RUN_KEYS
_SELF_KIND: Final = "m3c1_common_field_single_solver_self_convergence"
_BUDGET_KIND: Final = "m3c1_preregistered_common_field_budget"
_COMPARISON_KIND: Final = "m3c1_locked_common_field_trajectory_comparison"
_REFERENCE_CLASSIFICATION: Final = "external_comsol_full_physics_common_p1_reference"
_INITIAL_STATE_QUANTITIES: Final = ("r_m", "z_m", "vr_m_s", "vz_m_s", "charge_e")
_EXPECTED_LABELS: Final = {
    "candidate": "solver_canonical_p1_projection",
    "reference": "comsol_common_canonical_p1",
}
type SolverLabel = Literal["solver_canonical_p1_projection", "comsol_common_canonical_p1"]


@dataclass(frozen=True)
class _Config:
    metrics: shared._Config
    absolute_limits: dict[str, dict[str, float]]
    candidate_input_sha256: str
    candidate_input_content_hash: str
    candidate_tool_revision: str
    candidate_configuration_sha256: str
    candidate_engine_revision: str
    candidate_physics_models: dict[str, object]
    source_mph_sha256: str
    reference_tool_revision: str
    table_tool_revision: str
    table_receipt_filename: str
    artifact_hashes_filename: str
    normalization_summary_filename: str
    prepared_table_artifact_count: int
    primitive_component_count: int
    primitive_value_count: int
    primitive_roundoff_multiplier: float
    initial_state_roundoff_multiplier: float


def _scope_values(
    scope: dict[str, object],
) -> tuple[int, int, float, tuple[float, float], tuple[float, float, float]]:
    steps_raw = scope.get("internal_steps_s")
    if not isinstance(steps_raw, list) or len(steps_raw) != 3:
        raise ValueError("scope.internal_steps_s must have three entries")
    steps = tuple(
        shared._number(value, f"internal step {index}") for index, value in enumerate(steps_raw)
    )
    if (
        any(step <= 0.0 for step in steps)
        or steps[0] != 2.0 * steps[1]
        or steps[1] != 2.0 * steps[2]
    ):
        raise ValueError("scope.internal_steps_s must be exact positive h, h/2, h/4")
    time_raw = scope.get("time_window_s")
    if not isinstance(time_raw, list) or len(time_raw) != 2:
        raise ValueError("scope.time_window_s must have start and end entries")
    time_window = (
        shared._number(time_raw[0], "time window start"),
        shared._number(time_raw[1], "time window end"),
    )
    if time_window[0] != 0.0 or time_window[1] <= time_window[0]:
        raise ValueError("scope.time_window_s must be a positive zero-origin window")
    return (
        shared._positive_int(scope.get("particles"), "scope.particles"),
        shared._positive_int(scope.get("frames"), "scope.frames"),
        shared._number(scope.get("output_interval_s"), "output interval"),
        time_window,
        (steps[0], steps[1], steps[2]),
    )


def _acceptance_values(
    acceptance: dict[str, object],
) -> tuple[dict[str, float], dict[str, dict[str, float]], float, float]:
    relative_raw = shared._mapping(
        acceptance.get("maximum_fine_pair_relative_l2"),
        "maximum_fine_pair_relative_l2",
    )
    relative_limits = {
        quantity: shared._number(relative_raw.get(quantity), f"{quantity} relative limit")
        for quantity in ("position", "velocity", "charge")
    }
    absolute_raw = shared._mapping(
        acceptance.get("same_field_absolute_limits"),
        "same_field_absolute_limits",
    )
    absolute_limits: dict[str, dict[str, float]] = {}
    for quantity in ("position", "velocity", "charge"):
        quantity_raw = shared._mapping(absolute_raw.get(quantity), f"{quantity} limits")
        absolute_limits[quantity] = {
            metric: shared._number(quantity_raw.get(metric), f"{quantity}.{metric} limit")
            for metric in ("rms", "maximum")
        }
        if any(value <= 0.0 for value in absolute_limits[quantity].values()):
            raise ValueError(f"{quantity} absolute limits must be positive")
    if acceptance.get("result_dependent_tolerance_tuning") != "PROHIBITED":
        raise ValueError("result-dependent tolerance tuning must remain prohibited")
    return (
        relative_limits,
        absolute_limits,
        shared._number(acceptance.get("minimum_rms_order"), "minimum RMS order"),
        shared._number(acceptance.get("roundoff_multiplier"), "roundoff multiplier"),
    )


def _load_config(path: Path) -> _Config:
    raw, resolved, digest = shared._load_json(path, "common-field evaluation config")
    if raw.get("evaluation_id") != EVALUATION_ID:
        raise ValueError(f"{resolved}: unexpected common-field evaluation_id")
    labels_raw = shared._mapping(raw.get("labels"), "labels")
    labels = {
        "candidate": shared._string(labels_raw.get("candidate"), "candidate label"),
        "reference": shared._string(labels_raw.get("reference"), "reference label"),
    }
    if labels != _EXPECTED_LABELS:
        raise ValueError("config labels must identify the two common-field solvers")

    scope = shared._mapping(raw.get("scope"), "scope")
    particles, frames, output_interval, time_window, steps = _scope_values(scope)
    acceptance = shared._mapping(raw.get("acceptance"), "acceptance")
    relative_limits, absolute_limits, minimum_order, roundoff_multiplier = _acceptance_values(
        acceptance
    )

    candidate = shared._mapping(raw.get("candidate"), "candidate")
    reference = shared._mapping(raw.get("reference"), "reference")
    reference_validation = shared._mapping(reference.get("validation"), "reference validation")
    primitive_roundoff_multiplier = shared._number(
        reference_validation.get("initial_primitive_roundoff_multiplier"),
        "initial primitive roundoff multiplier",
    )
    if primitive_roundoff_multiplier <= 0.0:
        raise ValueError("initial primitive roundoff multiplier must be positive")
    initial_state_roundoff_multiplier = shared._number(
        reference_validation.get("initial_state_roundoff_multiplier"),
        "initial state roundoff multiplier",
    )
    if initial_state_roundoff_multiplier <= 0.0:
        raise ValueError("initial state roundoff multiplier must be positive")
    metrics = shared._Config(
        path=resolved,
        sha256=digest,
        raw=raw,
        labels=labels,
        particles=particles,
        frames=frames,
        output_interval_s=output_interval,
        time_window_s=time_window,
        steps_s=(steps[0], steps[1], steps[2]),
        minimum_order=minimum_order,
        relative_limits=relative_limits,
        parity_factor=1.0,
        roundoff_multiplier=roundoff_multiplier,
        reference_hashes={},
    )
    return _Config(
        metrics=metrics,
        absolute_limits=absolute_limits,
        candidate_input_sha256=shared._string(
            candidate.get("input_file_sha256"), "candidate input file hash"
        ),
        candidate_input_content_hash=shared._string(
            candidate.get("input_content_hash"), "candidate input content hash"
        ),
        candidate_tool_revision=shared._string(
            candidate.get("producer_tool_revision"), "candidate producer revision"
        ),
        candidate_configuration_sha256=shared._string(
            candidate.get("configuration_sha256"), "candidate configuration hash"
        ),
        candidate_engine_revision=shared._string(
            candidate.get("engine_algorithm_revision"), "candidate engine revision"
        ),
        candidate_physics_models=shared._mapping(
            candidate.get("physics_models"), "candidate physics models"
        ),
        source_mph_sha256=shared._string(reference.get("source_mph_sha256"), "source MPH hash"),
        reference_tool_revision=shared._string(
            reference.get("producer_tool_revision"), "reference producer revision"
        ),
        table_tool_revision=shared._string(
            reference.get("table_tool_revision"), "table producer revision"
        ),
        table_receipt_filename=shared._string(
            reference.get("table_receipt_filename"), "table receipt filename"
        ),
        artifact_hashes_filename=shared._string(
            reference.get("artifact_hashes_filename"), "artifact hash filename"
        ),
        normalization_summary_filename=shared._string(
            reference.get("normalization_summary_filename"),
            "normalization summary filename",
        ),
        prepared_table_artifact_count=shared._positive_int(
            reference_validation.get("prepared_table_artifact_count"),
            "prepared table artifact count",
        ),
        primitive_component_count=shared._positive_int(
            reference_validation.get("initial_primitive_component_count"),
            "initial primitive component count",
        ),
        primitive_value_count=shared._positive_int(
            reference_validation.get("initial_primitive_value_count"),
            "initial primitive value count",
        ),
        primitive_roundoff_multiplier=primitive_roundoff_multiplier,
        initial_state_roundoff_multiplier=initial_state_roundoff_multiplier,
    )


def _verify_candidate_run_lock(
    run_key: str,
    step: str,
    runs: dict[str, object],
    bindings: dict[str, object],
    config: _Config,
) -> None:
    record = shared._mapping(runs.get(run_key), f"candidate run {run_key}")
    if record.get("engine_algorithm_revision") != config.candidate_engine_revision:
        raise ValueError(f"candidate run {run_key} engine revision differs from the lock")
    if record.get("physics_models") != config.candidate_physics_models:
        raise ValueError(f"candidate run {run_key} physics models differ from the lock")
    binding = shared._mapping(bindings.get(step), f"candidate binding {step}")
    revisions = shared._mapping(binding.get("revisions"), f"candidate revisions {step}")
    if revisions.get("engine_algorithm_revision") != config.candidate_engine_revision:
        raise ValueError(f"candidate manifest {run_key} engine revision differs from the lock")


def _verify_candidate_provenance_lock(
    run_report_path: Path,
    prepare_report_path: Path,
    provenance: dict[str, object],
    config: _Config,
) -> dict[str, object]:
    """Bind this diagnostic to its one audited candidate producer."""

    run, _, _ = shared._load_json(run_report_path, "candidate run report")
    prepare, _, _ = shared._load_json(prepare_report_path, "prepare report")
    if (
        run.get("tool_revision") != config.candidate_tool_revision
        or prepare.get("tool_revision") != config.candidate_tool_revision
    ):
        raise ValueError("candidate producer revision differs from the evaluation lock")
    input_h5 = shared._mapping(provenance.get("input_h5"), "candidate input provenance")
    if (
        input_h5.get("sha256") != config.candidate_input_sha256
        or prepare.get("input_file_sha256") != config.candidate_input_sha256
    ):
        raise ValueError("candidate input file hash differs from the evaluation lock")
    if prepare.get("input_content_hash") != config.candidate_input_content_hash:
        raise ValueError("candidate input content hash differs from the evaluation lock")
    if prepare.get("configuration_sha256") != config.candidate_configuration_sha256:
        raise ValueError("candidate configuration hash differs from the evaluation lock")
    configuration_path = (
        Path(shared._string(prepare.get("configuration"), "candidate configuration path"))
        .expanduser()
        .resolve()
    )
    if (
        not configuration_path.is_file()
        or shared._sha256(configuration_path) != config.candidate_configuration_sha256
    ):
        raise ValueError("candidate configuration artifact differs from the evaluation lock")

    runs = shared._mapping(run.get("runs"), "candidate runs")
    bindings = shared._mapping(provenance.get("runs"), "candidate run provenance")
    for step, run_key in zip(STEP_NAMES, RUN_KEYS, strict=True):
        _verify_candidate_run_lock(run_key, step, runs, bindings, config)

    return {
        **provenance,
        "evaluation_lock": {
            "producer_tool_revision": config.candidate_tool_revision,
            "input_file_sha256": config.candidate_input_sha256,
            "input_content_hash": config.candidate_input_content_hash,
            "configuration": {
                "path": str(configuration_path),
                "sha256": config.candidate_configuration_sha256,
            },
            "engine_algorithm_revision": config.candidate_engine_revision,
            "physics_models": config.candidate_physics_models,
        },
    }


def _relative_path(root: Path, value: object, name: str) -> Path:
    relative = Path(shared._string(value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be receipt-relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes its receipt directory")
    return resolved


def _artifact_ledger(path: Path) -> dict[str, tuple[str, int]]:
    records: dict[str, tuple[str, int]] = {}
    with path.open(encoding="utf-8-sig", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != ("path", "sha256", "bytes"):
            raise ValueError("artifact hash ledger columns are invalid")
        for line_number, row in enumerate(reader, start=2):
            artifact = shared._string(row.get("path"), f"artifact path at line {line_number}")
            if artifact in records:
                raise ValueError(f"duplicate artifact hash entry: {artifact}")
            size = shared._positive_int(int(row["bytes"]), f"artifact bytes at line {line_number}")
            digest = shared._string(row.get("sha256"), f"artifact hash at line {line_number}")
            records[artifact] = (digest, size)
    return records


def _verify_table_artifacts(
    root: Path,
    table: dict[str, object],
    ledger: dict[str, tuple[str, int]],
) -> None:
    artifacts = shared._mapping(table.get("artifacts"), "common-P1 table artifacts")
    if not artifacts:
        raise ValueError("common-P1 table receipt has no artifacts")
    for name, raw_record in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != name:
            raise ValueError(f"common-P1 table artifact name is invalid: {name}")
        record = shared._mapping(raw_record, f"common-P1 table artifact {name}")
        digest = shared._string(record.get("sha256"), f"common-P1 table artifact {name} hash")
        size = shared._positive_int(
            record.get("size_bytes"), f"common-P1 table artifact {name} size"
        )
        if ledger.get(name) != (digest, size):
            raise ValueError(f"common-P1 table artifact {name} differs from the final ledger")
        staged = root / name
        if (
            not staged.is_file()
            or staged.stat().st_size != size
            or shared._sha256(staged) != digest
        ):
            raise ValueError(f"staged common-P1 table artifact {name} differs from its receipt")


def _verify_prepared_table_validation(receipt: dict[str, object], config: _Config) -> None:
    validation = shared._mapping(
        receipt.get("prepared_table_validation"), "prepared table validation"
    )
    if validation.get("status") != "PASS":
        raise ValueError("prepared common-P1 table validation did not pass")
    if validation.get("artifact_count") != config.prepared_table_artifact_count:
        raise ValueError("prepared common-P1 table artifact count differs from the lock")


def _verify_reference_identity(receipt: dict[str, object], config: _Config) -> str:
    if receipt.get("tool_revision") != config.reference_tool_revision:
        raise ValueError("COMSOL reference producer revision is invalid")
    if receipt.get("classification") != _REFERENCE_CLASSIFICATION:
        raise ValueError("COMSOL reference classification is invalid")
    if receipt.get("status") != "COMPLETE":
        raise ValueError("COMSOL common-field run is incomplete")
    if receipt.get("candidate_input_sha256") != config.candidate_input_sha256:
        raise ValueError("COMSOL reference uses another canonical candidate input")
    if (
        receipt.get("source_sha256_before") != config.source_mph_sha256
        or receipt.get("source_sha256_after") != config.source_mph_sha256
    ):
        raise ValueError("COMSOL source MPH hash is invalid or changed during execution")
    return shared._string(receipt.get("comsol_version"), "COMSOL version")


def _verify_reference_inputs(
    root: Path, receipt: dict[str, object], config: _Config
) -> tuple[Path, str, Path, str, dict[str, tuple[str, int]]]:
    table_path = _relative_path(root, config.table_receipt_filename, "table receipt")
    table_hash = shared._sha256(table_path)
    if receipt.get("table_receipt_sha256") != table_hash:
        raise ValueError("common-P1 table receipt hash differs")
    table, _, _ = shared._load_json(table_path, "common-P1 table receipt")
    if table.get("tool_revision") != config.table_tool_revision:
        raise ValueError("common-P1 table producer revision is invalid")
    candidate = shared._mapping(table.get("candidate"), "table candidate")
    if candidate.get("file_sha256") != config.candidate_input_sha256:
        raise ValueError("common-P1 tables use another canonical candidate input")
    artifact_path = _relative_path(root, config.artifact_hashes_filename, "artifact ledger")
    artifact_hash = shared._sha256(artifact_path)
    if receipt.get("artifact_hashes_sha256") != artifact_hash:
        raise ValueError("COMSOL artifact ledger hash differs")
    ledger = _artifact_ledger(artifact_path)
    _verify_table_artifacts(root, table, ledger)
    _verify_prepared_table_validation(receipt, config)
    return table_path, table_hash, artifact_path, artifact_hash, ledger


def _verify_reference_scope(receipt: dict[str, object], config: _Config) -> None:
    scope = shared._mapping(receipt.get("scope"), "COMSOL run scope")
    expected = {
        "particles": config.metrics.particles,
        "frames": config.metrics.frames,
        "time_window_s": list(config.metrics.time_window_s),
        "output_interval_s": config.metrics.output_interval_s,
    }
    if scope != expected:
        raise ValueError("COMSOL run scope differs from the locked evaluation scope")


def _verified_initial_state_summary(
    run: dict[str, object], run_key: str, config: _Config
) -> dict[str, object]:
    validation = shared._mapping(
        run.get("initial_state"), f"COMSOL run {run_key} initial state validation"
    )
    if validation.get("status") != "PASS":
        raise ValueError(f"COMSOL run {run_key} initial state validation did not pass")
    if validation.get("roundoff_multiplier") != config.initial_state_roundoff_multiplier:
        raise ValueError(f"COMSOL run {run_key} initial state multiplier differs from the lock")
    maxima = shared._mapping(
        validation.get("maximum_absolute_difference"),
        f"COMSOL run {run_key} initial state maxima",
    )
    limits = shared._mapping(
        validation.get("roundoff_limit"), f"COMSOL run {run_key} initial state limits"
    )
    if set(maxima) != set(_INITIAL_STATE_QUANTITIES) or set(limits) != set(
        _INITIAL_STATE_QUANTITIES
    ):
        raise ValueError(f"COMSOL run {run_key} initial state quantities differ")
    checked: dict[str, object] = {}
    for quantity in _INITIAL_STATE_QUANTITIES:
        maximum = shared._number(maxima.get(quantity), f"{run_key} {quantity} maximum")
        limit = shared._number(limits.get(quantity), f"{run_key} {quantity} limit")
        if maximum < 0.0 or limit <= 0.0 or maximum > limit:
            raise ValueError(f"COMSOL run {run_key} initial state {quantity} exceeds its limit")
        checked[quantity] = {"maximum_absolute_difference": maximum, "roundoff_limit": limit}
    return {
        "status": "PASS",
        "checked_particle_count": config.metrics.particles,
        "checked_component_count": len(_INITIAL_STATE_QUANTITIES),
        "checked_value_count": config.metrics.particles * len(_INITIAL_STATE_QUANTITIES),
        "roundoff_multiplier": config.initial_state_roundoff_multiplier,
        "components": checked,
    }


def _verify_reference_summary(
    root: Path,
    receipt: dict[str, object],
    ledger: dict[str, tuple[str, int]],
    config: _Config,
) -> dict[str, object]:
    path = _relative_path(root, config.normalization_summary_filename, "normalization summary")
    identity = ledger.get(config.normalization_summary_filename)
    if identity is None:
        raise ValueError("normalization summary is absent from the final artifact ledger")
    if not path.is_file() or (shared._sha256(path), path.stat().st_size) != identity:
        raise ValueError("normalization summary differs from the final artifact ledger")
    summary, _, digest = shared._load_json(path, "normalization summary")
    if summary.get("tool_revision") != config.reference_tool_revision:
        raise ValueError("normalization summary producer revision is invalid")
    if summary.get("status") != "COMPLETE":
        raise ValueError("normalization summary is incomplete")
    _verify_reference_scope(summary, config)
    summary_runs = shared._mapping(summary.get("runs"), "normalization summary runs")
    receipt_runs = shared._mapping(receipt.get("runs"), "COMSOL receipt runs")
    if set(summary_runs) != set(RUN_KEYS):
        raise ValueError("normalization summary must contain exactly the three locked runs")
    validations: dict[str, object] = {}
    for step, run_key, expected_dt in zip(
        STEP_NAMES, RUN_KEYS, config.metrics.steps_s, strict=True
    ):
        run = shared._mapping(summary_runs.get(run_key), f"normalization run {run_key}")
        receipt_run = shared._mapping(receipt_runs.get(run_key), f"COMSOL receipt run {run_key}")
        if shared._number(run.get("dt_s"), f"normalization {run_key}.dt_s") != expected_dt:
            raise ValueError(f"normalization run {run_key} dt differs")
        if run.get("trajectory_path") != receipt_run.get("trajectory_path") or run.get(
            "trajectory_sha256"
        ) != receipt_run.get("trajectory_sha256"):
            raise ValueError(f"normalization run {run_key} trajectory identity differs")
        validations[step] = _verified_initial_state_summary(run, run_key, config)
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": identity[1],
        "initial_state_validation": validations,
    }


def _verify_initial_primitive_validation(
    run: dict[str, object], run_key: str, config: _Config
) -> None:
    validation = shared._mapping(
        run.get("initial_primitive_validation"),
        f"COMSOL run {run_key} initial primitive validation",
    )
    expected: dict[str, object] = {
        "status": "PASS",
        "checked_particle_count": config.metrics.particles,
        "checked_component_count": config.primitive_component_count,
        "checked_value_count": config.primitive_value_count,
        "roundoff_multiplier": config.primitive_roundoff_multiplier,
    }
    for key, value in expected.items():
        if validation.get(key) != value:
            raise ValueError(f"COMSOL run {run_key} initial primitive {key} differs from the lock")


def _verify_reference_runs(
    root: Path,
    receipt: dict[str, object],
    trajectories: dict[str, shared._Trajectory],
    config: _Config,
    ledger: dict[str, tuple[str, int]],
) -> dict[str, object]:
    runs = shared._mapping(receipt.get("runs"), "COMSOL runs")
    if set(runs) != set(RUN_KEYS):
        raise ValueError("COMSOL receipt must contain exactly the three locked runs")
    bindings: dict[str, object] = {}
    for step, run_key, expected_dt in zip(
        STEP_NAMES, RUN_KEYS, config.metrics.steps_s, strict=True
    ):
        run = shared._mapping(runs.get(run_key), f"COMSOL run {run_key}")
        if shared._number(run.get("dt_s"), f"{run_key}.dt_s") != expected_dt:
            raise ValueError(f"COMSOL run {run_key} dt differs")
        if run.get("rows") != config.metrics.particles * config.metrics.frames:
            raise ValueError(f"COMSOL run {run_key} has no complete trajectory matrix")
        if run.get("all_active") is not True or run.get("event_count") != 0:
            raise ValueError(f"COMSOL run {run_key} is not an event-free pre-event history")
        _verify_initial_primitive_validation(run, run_key, config)
        trajectory_path = _relative_path(
            root, run.get("trajectory_path"), f"COMSOL trajectory {run_key}"
        )
        trajectory = trajectories[step]
        if trajectory.path != trajectory_path:
            raise ValueError(f"COMSOL trajectory {run_key} is not the receipt artifact")
        expected_hash = shared._string(
            run.get("trajectory_sha256"), f"COMSOL trajectory {run_key} hash"
        )
        if trajectory.sha256 != expected_hash:
            raise ValueError(f"COMSOL trajectory {run_key} hash differs")
        ledger_key = trajectory_path.relative_to(root).as_posix()
        if ledger.get(ledger_key) != (expected_hash, trajectory_path.stat().st_size):
            raise ValueError(f"COMSOL trajectory {run_key} is not locked by the artifact ledger")
        bindings[step] = {
            "trajectory": {"path": str(trajectory.path), "sha256": trajectory.sha256},
            "dt_s": expected_dt,
        }
    return bindings


def _verify_reference_receipt(
    receipt_path: Path,
    trajectories: dict[str, shared._Trajectory],
    config: _Config,
) -> dict[str, object]:
    receipt, resolved, receipt_hash = shared._load_json(
        receipt_path, "common-field COMSOL run receipt"
    )
    root = resolved.parent
    comsol_version = _verify_reference_identity(receipt, config)
    table_path, table_hash, artifact_path, artifact_hash, ledger = _verify_reference_inputs(
        root, receipt, config
    )
    _verify_reference_scope(receipt, config)
    bindings = _verify_reference_runs(root, receipt, trajectories, config, ledger)
    normalization_summary = _verify_reference_summary(root, receipt, ledger, config)
    return {
        "run_receipt": {"path": str(resolved), "sha256": receipt_hash},
        "table_receipt": {"path": str(table_path), "sha256": table_hash},
        "artifact_hashes": {"path": str(artifact_path), "sha256": artifact_hash},
        "normalization_summary": normalization_summary,
        "source_mph_sha256": config.source_mph_sha256,
        "comsol_version": comsol_version,
        "runs": bindings,
    }


def characterize(
    config_path: Path,
    label: SolverLabel,
    coarse_path: Path,
    medium_path: Path,
    fine_path: Path,
    *,
    candidate_run_report: Path | None = None,
    prepare_report: Path | None = None,
    reference_run_report: Path | None = None,
) -> dict[str, object]:
    """Characterize one solver before reading any cross-solver difference."""

    config = _load_config(config_path)
    if label not in config.metrics.labels.values():
        raise ValueError("label is not owned by the common-field evaluation config")
    trajectories = {
        name: shared._read_trajectory(path, config.metrics)
        for name, path in zip(STEP_NAMES, (coarse_path, medium_path, fine_path), strict=True)
    }
    if label == config.metrics.labels["candidate"]:
        if candidate_run_report is None or prepare_report is None:
            raise ValueError("candidate characterization requires both producer receipts")
        provenance = shared._verify_candidate_receipts(
            candidate_run_report, prepare_report, trajectories, config.metrics
        )
        provenance = _verify_candidate_provenance_lock(
            candidate_run_report,
            prepare_report,
            provenance,
            config,
        )
    else:
        if reference_run_report is None:
            raise ValueError("COMSOL characterization requires its common-field run receipt")
        provenance = _verify_reference_receipt(reference_run_report, trajectories, config)

    initial = [trajectory.values[0] for trajectory in trajectories.values()]
    if not all(np.array_equal(initial[0], state) for state in initial[1:]):
        raise ValueError("self-convergence trajectories do not share one exact initial state")
    coarse_medium = shared._difference_metrics(
        trajectories[STEP_NAMES[0]].values,
        trajectories[STEP_NAMES[1]].values,
        config.metrics.output_interval_s,
    )
    medium_fine = shared._difference_metrics(
        trajectories[STEP_NAMES[1]].values,
        trajectories[STEP_NAMES[2]].values,
        config.metrics.output_interval_s,
    )
    fine = trajectories[STEP_NAMES[2]].values
    representation_scale = {
        "position": float(np.max(np.abs(fine[:, :, :2]))),
        "velocity": float(np.max(np.linalg.norm(fine[:, :, 2:4], axis=2))),
        "charge": float(np.max(np.abs(fine[:, :, 4]))),
    }
    orders, assessments, blockers = shared._assess_self_convergence(
        coarse_medium,
        medium_fine,
        representation_scale,
        config.metrics,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool_revision": TOOL_REVISION,
        "report_kind": _SELF_KIND,
        "label": label,
        "status": "PASS" if not blockers else "BLOCKED",
        "evaluation_config": {
            "path": str(config.metrics.path),
            "sha256": config.metrics.sha256,
        },
        "scope": config.metrics.raw["scope"],
        "source_scope": {
            "comparison": "two solvers consuming one canonical exact-connectivity P1 field",
            "provenance": provenance,
        },
        "artifacts": {
            name: {"path": str(item.path), "sha256": item.sha256}
            for name, item in trajectories.items()
        },
        "initial_state_sha256": hashlib.sha256(fine[0].tobytes(order="C")).hexdigest(),
        "representation_scale": {
            "position_m": representation_scale["position"],
            "velocity_m_per_s": representation_scale["velocity"],
            "charge_number_e": representation_scale["charge"],
        },
        f"{STEP_NAMES[0]}_vs_{STEP_NAMES[1]}": coarse_medium,
        f"{STEP_NAMES[1]}_vs_{STEP_NAMES[2]}": medium_fine,
        "observed_rms_order": orders,
        "rms_order_assessment": assessments,
        "acceptance": {
            "minimum_rms_order": config.metrics.minimum_order,
            "maximum_fine_pair_relative_l2": config.metrics.relative_limits,
            "roundoff_multiplier": config.metrics.roundoff_multiplier,
            "roundoff_plateau_interpretation": (
                "precision stability only; order and error below the floor are not established"
            ),
        },
        "blockers": blockers,
        "claim_separation": {
            "same_field_solver_agreement": "SELF_CONVERGENCE_ONLY",
            "physical_model_validity": "NOT_CLAIMED",
            "boundary_accuracy": "NOT_TESTED_PRE_EVENT_WINDOW",
            "roundoff_limited_order": "NOT_ESTABLISHED_BELOW_ROUNDOFF_FLOOR",
        },
    }


def _load_report(path: Path, kind: str) -> tuple[dict[str, object], str]:
    report, resolved, digest = shared._load_json(path, "common-field evaluation report")
    if report.get("report_kind") != kind or report.get("tool_revision") != TOOL_REVISION:
        raise ValueError(f"{resolved}: incompatible common-field report")
    return report, digest


def _fine_artifact(report: dict[str, object]) -> dict[str, str]:
    artifact = shared._mapping(
        shared._mapping(report.get("artifacts"), "artifacts").get(STEP_NAMES[2]),
        "fine artifact",
    )
    return {
        "path": shared._string(artifact.get("path"), "fine artifact path"),
        "sha256": shared._string(artifact.get("sha256"), "fine artifact hash"),
    }


def _t0_record(
    row: dict[str, str], line_number: int, side: str
) -> tuple[int, tuple[float, ...]] | None:
    time_s = float(shared._string(row.get("time_s"), f"{side} time at line {line_number}"))
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError(f"{side} fine trajectory has an invalid time")
    if time_s > 0.0:
        return None
    particle_id = int(
        shared._string(row.get("particle_id"), f"{side} particle at line {line_number}")
    )
    if row.get("lifecycle") != "active":
        raise ValueError(f"{side} t=0 state has an inactive particle")
    values = tuple(
        float(shared._string(row.get(column), f"{side} {column} at line {line_number}"))
        for column in shared._STATE_COLUMNS
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{side} t=0 state contains nonfinite values")
    return particle_id, values


def _read_t0_state(path: Path, side: str, particles: int) -> np.ndarray:
    rows: dict[int, tuple[float, ...]] = {}
    with path.open(encoding="utf-8-sig", errors="strict", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if tuple(reader.fieldnames or ()) != shared._COLUMNS:
            raise ValueError(f"{side} fine trajectory columns are invalid")
        for line_number, row in enumerate(reader, start=2):
            record = _t0_record(row, line_number, side)
            if record is None:
                continue
            particle_id, values = record
            if particle_id in rows:
                raise ValueError(f"{side} t=0 state has a duplicate particle")
            rows[particle_id] = values
    expected_ids = set(range(1, particles + 1))
    if set(rows) != expected_ids:
        raise ValueError(f"{side} t=0 state must contain exactly the locked particle IDs")
    return np.asarray([rows[particle_id] for particle_id in sorted(rows)], dtype=np.float64)


def _registered_initial_state(
    report: dict[str, object], config: _Config, side: str
) -> tuple[np.ndarray, dict[str, str]]:
    artifact = _fine_artifact(report)
    path = Path(artifact["path"]).expanduser().resolve()
    if not path.is_file() or shared._sha256(path) != artifact["sha256"]:
        raise ValueError(f"{side} fine trajectory differs from its self-convergence report")
    values = _read_t0_state(path, side, config.metrics.particles)
    observed_hash = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    if report.get("initial_state_sha256") != observed_hash:
        raise ValueError(f"{side} t=0 state differs from its self-convergence report")
    return values, artifact


def _initial_state_gate(
    candidate: np.ndarray,
    reference: np.ndarray,
    multiplier: float,
) -> dict[str, object]:
    components: dict[str, object] = {}
    passed = True
    for index, quantity in enumerate(_INITIAL_STATE_QUANTITIES):
        differences = np.abs(candidate[:, index] - reference[:, index])
        scale = max(
            float(np.max(np.abs(candidate[:, index]))),
            float(np.max(np.abs(reference[:, index]))),
            1.0e-300,
        )
        ulp = math.ulp(scale)
        limit = multiplier * ulp
        worst_index = int(np.argmax(differences))
        maximum = float(differences[worst_index])
        component_passed = maximum <= limit
        passed = passed and component_passed
        components[quantity] = {
            "status": "PASS" if component_passed else "FAIL",
            "component_maximum_absolute_value": scale,
            "maximum_absolute_difference": maximum,
            "component_scale_ulp": ulp,
            "maximum_difference_in_component_scale_ulp": maximum / ulp,
            "roundoff_limit": limit,
            "worst_particle_id": worst_index + 1,
        }
    return {
        "status": "PASS" if passed else "FAIL",
        "criterion": (
            "for each quantity across all locked particles: maximum absolute t=0 difference "
            "<= multiplier * ulp(max(max_abs_candidate, max_abs_reference, 1e-300))"
        ),
        "read_scope": "only t=0 state values are parsed; post-t0 rows are skipped before state parsing",
        "roundoff_multiplier": multiplier,
        "checked_particle_count": candidate.shape[0],
        "checked_component_count": candidate.shape[1],
        "checked_value_count": int(candidate.size),
        "bitwise_identical": bool(np.array_equal(candidate, reference)),
        "components": components,
    }


def register_budget(
    config_path: Path, candidate_path: Path, reference_path: Path
) -> dict[str, object]:
    """Bind both characterizations to the config's predeclared tolerances."""

    config = _load_config(config_path)
    candidate, candidate_hash = _load_report(candidate_path, _SELF_KIND)
    reference, reference_hash = _load_report(reference_path, _SELF_KIND)
    blockers: list[str] = []
    for side, report in (("candidate", candidate), ("reference", reference)):
        if report.get("label") != config.metrics.labels[side]:
            blockers.append(f"{side} report label is invalid")
        if report.get("status") != "PASS":
            blockers.append(f"{side} self-convergence did not pass")
        evaluation = shared._mapping(report.get("evaluation_config"), "evaluation config")
        if evaluation.get("sha256") != config.metrics.sha256:
            blockers.append(f"{side} used another evaluation config")
    if candidate.get("scope") != reference.get("scope"):
        blockers.append("candidate and reference scopes differ")
    candidate_initial, candidate_fine = _registered_initial_state(candidate, config, "candidate")
    reference_initial, reference_fine = _registered_initial_state(reference, config, "reference")
    initial_state_gate = _initial_state_gate(
        candidate_initial,
        reference_initial,
        config.initial_state_roundoff_multiplier,
    )
    if initial_state_gate["status"] != "PASS":
        blockers.append("candidate and reference t=0 states exceed the locked roundoff criterion")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool_revision": TOOL_REVISION,
        "report_kind": _BUDGET_KIND,
        "status": "REGISTERED" if not blockers else "BLOCKED",
        "evaluation_config": {
            "path": str(config.metrics.path),
            "sha256": config.metrics.sha256,
        },
        "scope": candidate.get("scope"),
        "policy": {
            "field_representation": "COMMON_CANONICAL_EXACT_CONNECTIVITY_P1",
            "tolerance_source": "PREDECLARED_EVALUATION_CONFIG",
            "result_dependent_tolerance_tuning": "PROHIBITED",
            "initial_state_scope": "T0_ONLY_NO_POST_T0_CROSS_DIFFERENCE_READ",
        },
        "self_convergence_reports": {
            "candidate": {"path": str(candidate_path.resolve()), "sha256": candidate_hash},
            "reference": {"path": str(reference_path.resolve()), "sha256": reference_hash},
        },
        "fine_trajectory_artifacts": {
            "candidate": candidate_fine,
            "reference": reference_fine,
        },
        "initial_state_gate": initial_state_gate,
        "acceptance": {
            "absolute_limits": config.absolute_limits,
            "relative_l2_limits": config.metrics.relative_limits,
        },
        "blockers": blockers,
        "claim_separation": {
            "same_field_solver_agreement": ("BUDGET_REGISTERED" if not blockers else "BLOCKED"),
            "physical_model_validity": "NOT_CLAIMED",
        },
    }


def _registered_trajectory(
    budget: dict[str, object], side: str, supplied: Path, config: _Config
) -> shared._Trajectory:
    identity = shared._mapping(
        shared._mapping(budget.get("fine_trajectory_artifacts"), "fine artifacts").get(side),
        f"{side} fine artifact",
    )
    trajectory = shared._read_trajectory(supplied, config.metrics)
    if trajectory.sha256 != identity.get("sha256"):
        raise ValueError(f"{side} trajectory hash differs from registered budget")
    return trajectory


def compare(
    config_path: Path,
    budget_path: Path,
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    """Compare the two hash-locked fine trajectories against fixed gates."""

    config = _load_config(config_path)
    budget, budget_hash = _load_report(budget_path, _BUDGET_KIND)
    evaluation = shared._mapping(budget.get("evaluation_config"), "evaluation config")
    if evaluation.get("sha256") != config.metrics.sha256:
        raise ValueError("comparison budget uses another evaluation config")
    if budget.get("status") != "REGISTERED":
        raise ValueError("comparison budget is blocked")
    candidate = _registered_trajectory(budget, "candidate", candidate_path, config)
    reference = _registered_trajectory(budget, "reference", reference_path, config)
    metrics = shared._difference_metrics(
        candidate.values, reference.values, config.metrics.output_interval_s
    )
    gates: dict[str, object] = {}
    blockers: list[str] = []
    for quantity in ("position", "velocity", "charge"):
        observed_report = shared._mapping(metrics.get(quantity), quantity)
        quantity_gates: dict[str, object] = {}
        for metric in ("rms", "maximum"):
            observed = shared._number(observed_report.get(metric), f"{quantity}.{metric}")
            tolerance = config.absolute_limits[quantity][metric]
            passed = observed <= tolerance
            quantity_gates[metric] = {
                "observed": observed,
                "tolerance": tolerance,
                "pass": passed,
            }
            if not passed:
                blockers.append(f"same-field {quantity}.{metric} exceeds its fixed limit")
        relative = observed_report.get("relative_l2")
        relative_limit = config.metrics.relative_limits[quantity]
        relative_passed = isinstance(relative, int | float) and float(relative) <= relative_limit
        quantity_gates["relative_l2"] = {
            "observed": relative,
            "tolerance": relative_limit,
            "pass": relative_passed,
        }
        if not relative_passed:
            blockers.append(f"same-field {quantity}.relative_l2 exceeds its fixed limit")
        gates[quantity] = quantity_gates
    passed = not blockers
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool_revision": TOOL_REVISION,
        "report_kind": _COMPARISON_KIND,
        "status": "PASS" if passed else "FAIL",
        "evaluation_config": {
            "path": str(config.metrics.path),
            "sha256": config.metrics.sha256,
        },
        "scope": budget.get("scope"),
        "budget": {"path": str(budget_path.resolve()), "sha256": budget_hash},
        "artifacts": {
            "candidate": {"path": str(candidate.path), "sha256": candidate.sha256},
            "reference": {"path": str(reference.path), "sha256": reference.sha256},
        },
        "trajectory_difference": metrics,
        "acceptance_gates": gates,
        "blockers": blockers,
        "claim_separation": {
            "same_field_solver_agreement": "PASS" if passed else "FAIL",
            "cross_representation_trajectory": "SEPARATE_ARTIFACT",
        },
        "accuracy_claim": {
            "locked_common_field_case_and_window": (
                "SUPPORTED_WITHIN_PREDECLARED_LIMITS" if passed else "NOT_SUPPORTED"
            ),
            "comsol_universal_equal_accuracy": "NOT_CLAIMED",
            "physical_model_validity": "NOT_CLAIMED",
            "boundary_accuracy": "NOT_TESTED_PRE_EVENT_WINDOW",
            "brownian_accuracy": "NOT_TESTED_DISABLED",
        },
    }


def _write(path: Path, report: dict[str, object]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", errors="strict") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    characterize_parser = commands.add_parser("characterize")
    characterize_parser.add_argument("--config", required=True, type=Path)
    characterize_parser.add_argument(
        "--label", required=True, choices=tuple(_EXPECTED_LABELS.values())
    )
    characterize_parser.add_argument("--coarse", required=True, type=Path)
    characterize_parser.add_argument("--medium", required=True, type=Path)
    characterize_parser.add_argument("--fine", required=True, type=Path)
    characterize_parser.add_argument("--candidate-run-report", type=Path)
    characterize_parser.add_argument("--prepare-report", type=Path)
    characterize_parser.add_argument("--reference-run-report", type=Path)
    characterize_parser.add_argument("--output", required=True, type=Path)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--config", required=True, type=Path)
    register_parser.add_argument("--candidate", required=True, type=Path)
    register_parser.add_argument("--reference", required=True, type=Path)
    register_parser.add_argument("--output", required=True, type=Path)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--config", required=True, type=Path)
    compare_parser.add_argument("--budget", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--reference", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "characterize":
        report = characterize(
            args.config,
            args.label,
            args.coarse,
            args.medium,
            args.fine,
            candidate_run_report=args.candidate_run_report,
            prepare_report=args.prepare_report,
            reference_run_report=args.reference_run_report,
        )
    elif args.command == "register":
        report = register_budget(args.config, args.candidate, args.reference)
    else:
        report = compare(args.config, args.budget, args.candidate, args.reference)
    _write(args.output, report)
    return 0 if report["status"] in {"PASS", "REGISTERED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
