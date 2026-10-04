from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from tools.vv.comsol.evaluate_m3c1_common_field import (
    RUN_KEYS,
    STEP_NAMES,
    characterize,
    compare,
    register_budget,
)

_STEPS = (6.25e-7, 3.125e-7, 1.5625e-7)
_SOURCE_HASH = "3" * 64
_CANDIDATE_TOOL_REVISION = "candidate_builder_v1"
_CANDIDATE_CONTENT_HASH = "sha256:synthetic-common-field-content"
_CANDIDATE_ENGINE_REVISION = "engine_v1"
_CANDIDATE_PHYSICS = {"dynamic_charge": {"model": "synthetic", "revision": "charge_v1"}}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _write_trajectory(
    path: Path,
    *,
    dt_s: float,
    coefficient: float,
    initial_shift: float = 0.0,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "time_s",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number_e",
                "lifecycle",
            )
        )
        for frame in range(4):
            time_s = frame * 1.0e-5
            error = coefficient * (dt_s / _STEPS[0]) ** 2 * frame / 3.0
            for particle_id in range(1, 4):
                writer.writerow(
                    (
                        particle_id,
                        time_s,
                        0.14 + particle_id * 1.0e-5 + initial_shift + 0.01 * time_s + error,
                        0.023 + particle_id * 2.0e-6 + initial_shift + 0.02 * time_s - error,
                        0.01 + particle_id * 1.0e-6 + initial_shift + error,
                        0.02 - particle_id * 1.0e-6 + initial_shift - error,
                        -1.0 + initial_shift + 1000.0 * time_s + error,
                        "active",
                    )
                )


def _write_candidate_receipts(root: Path, trajectories: list[Path]) -> tuple[Path, Path, str, str]:
    input_path = root / "candidate_input.h5"
    input_path.write_bytes(b"synthetic canonical common-field input")
    input_hash = _sha256(input_path)
    configuration_path = root / "candidate_source_config.json"
    _write_json(configuration_path, {"evaluation": "synthetic candidate producer"})
    configuration_hash = _sha256(configuration_path)
    cases: dict[str, str] = {}
    runs: dict[str, object] = {}
    for run_key, trajectory, dt_s in zip(RUN_KEYS, trajectories, _STEPS, strict=True):
        case_path = root / f"candidate_{run_key}.yaml"
        case_path.write_text(f"name: {run_key}\n", encoding="utf-8")
        result_path = root / f"result_{run_key}"
        result_path.mkdir()
        manifest_path = result_path / "run.json"
        manifest = {
            "status": "complete",
            "case_file_hash": f"sha256:{_sha256(case_path)}",
            "data_content_hash": _CANDIDATE_CONTENT_HASH,
            "engine_algorithm_revision": _CANDIDATE_ENGINE_REVISION,
            "result_algorithm_revision": "result_v1",
            "resolved": {"physics_models": _CANDIDATE_PHYSICS},
            "time": {"dt_s": dt_s, "start_s": 0.0, "end_s": 3.0e-5},
        }
        _write_json(manifest_path, manifest)
        cases[run_key] = case_path.name
        runs[run_key] = {
            "status": "complete",
            "decision": "COMPLETE",
            "case": case_path.name,
            "case_sha256": _sha256(case_path),
            "result": result_path.name,
            "result_manifest_sha256": _sha256(manifest_path),
            "trajectory": trajectory.name,
            "trajectory_sha256": _sha256(trajectory),
            "trajectory_rows": 12,
            "engine_algorithm_revision": _CANDIDATE_ENGINE_REVISION,
            "physics_models": _CANDIDATE_PHYSICS,
        }
    prepare_path = root / "prepare_report.json"
    _write_json(
        prepare_path,
        {
            "tool_revision": _CANDIDATE_TOOL_REVISION,
            "configuration": str(configuration_path.resolve()),
            "configuration_sha256": configuration_hash,
            "input_data": input_path.name,
            "input_file_sha256": input_hash,
            "input_content_hash": _CANDIDATE_CONTENT_HASH,
            "cases": cases,
        },
    )
    run_path = root / "candidate_run_report.json"
    _write_json(
        run_path,
        {
            "tool_revision": _CANDIDATE_TOOL_REVISION,
            "status": "complete",
            "decision": "COMPLETE",
            "runs": runs,
        },
    )
    return run_path, prepare_path, input_hash, configuration_hash


def _write_reference_receipt(
    root: Path, trajectories: list[Path], candidate_input_hash: str
) -> Path:
    table_artifact = root / "m3c1_synthetic_sectionwise.txt"
    table_artifact.write_text("synthetic P1 table\n", encoding="utf-8")
    table_path = root / "common_p1_table_receipt.json"
    _write_json(
        table_path,
        {
            "tool_revision": "m3c1_full_physics_common_p1_tables_v1",
            "candidate": {"file_sha256": candidate_input_hash},
            "artifacts": {
                table_artifact.name: {
                    "sha256": _sha256(table_artifact),
                    "size_bytes": table_artifact.stat().st_size,
                }
            },
        },
    )
    summary_path = root / "normalization_summary.json"
    summary_runs = {
        run_key: {
            "dt_s": dt_s,
            "rows": 12,
            "trajectory_path": trajectory.relative_to(root).as_posix(),
            "trajectory_sha256": _sha256(trajectory),
            "initial_state": {
                "status": "PASS",
                "roundoff_multiplier": 4096.0,
                "maximum_absolute_difference": dict.fromkeys(
                    ("r_m", "z_m", "vr_m_s", "vz_m_s", "charge_e"), 0.0
                ),
                "roundoff_limit": dict.fromkeys(
                    ("r_m", "z_m", "vr_m_s", "vz_m_s", "charge_e"), 1.0e-10
                ),
            },
        }
        for run_key, dt_s, trajectory in zip(RUN_KEYS, _STEPS, trajectories, strict=True)
    }
    _write_json(
        summary_path,
        {
            "tool_revision": "m3c1_full_physics_common_p1_reference_v1",
            "status": "COMPLETE",
            "scope": {
                "particles": 3,
                "frames": 4,
                "time_window_s": [0.0, 3.0e-5],
                "output_interval_s": 1.0e-5,
            },
            "runs": summary_runs,
        },
    )
    ledger_path = root / "artifact_hashes.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("path", "sha256", "bytes"))
        for path in (*trajectories, table_artifact, table_path, summary_path):
            writer.writerow((path.relative_to(root).as_posix(), _sha256(path), path.stat().st_size))
    receipt_path = root / "common_p1_run_receipt.json"
    runs = {
        run_key: {
            "dt_s": dt_s,
            "trajectory_path": trajectory.relative_to(root).as_posix(),
            "trajectory_sha256": _sha256(trajectory),
            "rows": 12,
            "all_active": True,
            "event_count": 0,
            "initial_primitive_validation": {
                "status": "PASS",
                "checked_particle_count": 3,
                "checked_component_count": 22,
                "checked_value_count": 66,
                "roundoff_multiplier": 4096.0,
            },
        }
        for run_key, dt_s, trajectory in zip(RUN_KEYS, _STEPS, trajectories, strict=True)
    }
    _write_json(
        receipt_path,
        {
            "tool_revision": "m3c1_full_physics_common_p1_reference_v1",
            "classification": "external_comsol_full_physics_common_p1_reference",
            "status": "COMPLETE",
            "candidate_input_sha256": candidate_input_hash,
            "table_receipt_sha256": _sha256(table_path),
            "source_sha256_before": _SOURCE_HASH,
            "source_sha256_after": _SOURCE_HASH,
            "comsol_version": "synthetic-COMSOL",
            "scope": {
                "particles": 3,
                "frames": 4,
                "time_window_s": [0.0, 3.0e-5],
                "output_interval_s": 1.0e-5,
            },
            "runs": runs,
            "artifact_hashes_sha256": _sha256(ledger_path),
            "prepared_table_validation": {"status": "PASS", "artifact_count": 1},
        },
    )
    return receipt_path


def _write_config(path: Path, candidate_input_hash: str, candidate_configuration_hash: str) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "evaluation_id": "M3-C1-caseA-100nm-common-canonical-p1-pre-event",
            "labels": {
                "candidate": "solver_canonical_p1_projection",
                "reference": "comsol_common_canonical_p1",
            },
            "scope": {
                "particles": 3,
                "frames": 4,
                "output_interval_s": 1.0e-5,
                "time_window_s": [0.0, 3.0e-5],
                "internal_steps_s": list(_STEPS),
            },
            "candidate": {
                "producer_tool_revision": _CANDIDATE_TOOL_REVISION,
                "configuration_sha256": candidate_configuration_hash,
                "input_file_sha256": candidate_input_hash,
                "input_content_hash": _CANDIDATE_CONTENT_HASH,
                "engine_algorithm_revision": _CANDIDATE_ENGINE_REVISION,
                "physics_models": _CANDIDATE_PHYSICS,
            },
            "reference": {
                "producer_tool_revision": "m3c1_full_physics_common_p1_reference_v1",
                "table_tool_revision": "m3c1_full_physics_common_p1_tables_v1",
                "source_mph_sha256": _SOURCE_HASH,
                "table_receipt_filename": "common_p1_table_receipt.json",
                "artifact_hashes_filename": "artifact_hashes.csv",
                "normalization_summary_filename": "normalization_summary.json",
                "validation": {
                    "prepared_table_artifact_count": 1,
                    "initial_state_roundoff_multiplier": 4096.0,
                    "initial_primitive_component_count": 22,
                    "initial_primitive_value_count": 66,
                    "initial_primitive_roundoff_multiplier": 4096.0,
                },
            },
            "acceptance": {
                "minimum_rms_order": 0.75,
                "maximum_fine_pair_relative_l2": {
                    "position": 100.0,
                    "velocity": 100.0,
                    "charge": 100.0,
                },
                "same_field_absolute_limits": {
                    quantity: {"rms": 1.0e-8, "maximum": 1.0e-8}
                    for quantity in ("position", "velocity", "charge")
                },
                "roundoff_multiplier": 2048.0,
                "result_dependent_tolerance_tuning": "PROHIBITED",
            },
        },
    )


def _setup(
    tmp_path: Path,
    *,
    candidate_coefficient: float = 1.0e-5,
    reference_coefficient: float = 1.0e-5,
    reference_initial_shift: float = 0.0,
) -> tuple[Path, list[Path], list[Path], Path, Path, Path]:
    candidate_root = tmp_path / "candidate"
    reference_root = tmp_path / "reference"
    candidate_root.mkdir()
    reference_root.mkdir()
    candidate: list[Path] = []
    reference: list[Path] = []
    for name, dt_s in zip(STEP_NAMES, _STEPS, strict=True):
        candidate_path = candidate_root / f"candidate_{name}.csv"
        reference_path = reference_root / f"reference_{name}.csv"
        _write_trajectory(
            candidate_path,
            dt_s=dt_s,
            coefficient=candidate_coefficient,
        )
        _write_trajectory(
            reference_path,
            dt_s=dt_s,
            coefficient=reference_coefficient,
            initial_shift=reference_initial_shift,
        )
        candidate.append(candidate_path)
        reference.append(reference_path)
    candidate_run, prepare_report, input_hash, configuration_hash = _write_candidate_receipts(
        candidate_root, candidate
    )
    reference_run = _write_reference_receipt(reference_root, reference, input_hash)
    config = tmp_path / "config.json"
    _write_config(config, input_hash, configuration_hash)
    return config, candidate, reference, candidate_run, prepare_report, reference_run


def _characterizations(
    config: Path,
    candidate: list[Path],
    reference: list[Path],
    candidate_run: Path,
    prepare_report: Path,
    reference_run: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    candidate_report = characterize(
        config,
        "solver_canonical_p1_projection",
        *candidate,
        candidate_run_report=candidate_run,
        prepare_report=prepare_report,
    )
    reference_report = characterize(
        config,
        "comsol_common_canonical_p1",
        *reference,
        reference_run_report=reference_run,
    )
    return candidate_report, reference_report


def test_same_field_three_stage_pass_uses_predeclared_limits(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    config, candidate, reference, *_ = setup
    candidate_report, reference_report = _characterizations(*setup)
    assert candidate_report["status"] == "PASS"
    assert reference_report["status"] == "PASS"
    candidate_path = tmp_path / "candidate_report.json"
    reference_path = tmp_path / "reference_report.json"
    _write_json(candidate_path, candidate_report)
    _write_json(reference_path, reference_report)

    budget = register_budget(config, candidate_path, reference_path)
    assert budget["status"] == "REGISTERED"
    assert _mapping(budget["policy"])["tolerance_source"] == "PREDECLARED_EVALUATION_CONFIG"
    budget_path = tmp_path / "budget.json"
    _write_json(budget_path, budget)

    comparison = compare(config, budget_path, candidate[-1], reference[-1])
    assert comparison["status"] == "PASS"
    claims = _mapping(comparison["accuracy_claim"])
    assert claims["comsol_universal_equal_accuracy"] == "NOT_CLAIMED"
    assert claims["physical_model_validity"] == "NOT_CLAIMED"
    assert claims["boundary_accuracy"] == "NOT_TESTED_PRE_EVENT_WINDOW"


def test_same_field_difference_fails_fixed_gates(tmp_path: Path) -> None:
    setup = _setup(tmp_path, reference_coefficient=2.0e-5)
    config, candidate, reference, *_ = setup
    candidate_report, reference_report = _characterizations(*setup)
    candidate_path = tmp_path / "candidate_report.json"
    reference_path = tmp_path / "reference_report.json"
    _write_json(candidate_path, candidate_report)
    _write_json(reference_path, reference_report)
    budget = register_budget(config, candidate_path, reference_path)
    budget_path = tmp_path / "budget.json"
    _write_json(budget_path, budget)

    comparison = compare(config, budget_path, candidate[-1], reference[-1])

    assert comparison["status"] == "FAIL"
    assert _mapping(comparison["claim_separation"])["same_field_solver_agreement"] == "FAIL"
    assert comparison["blockers"]


def test_reference_trajectory_must_match_run_receipt_and_artifact_ledger(
    tmp_path: Path,
) -> None:
    config, _, reference, _, _, reference_run = _setup(tmp_path)
    reference[-1].write_text(reference[-1].read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        characterize(
            config,
            "comsol_common_canonical_p1",
            *reference,
            reference_run_report=reference_run,
        )


def test_reference_table_receipt_must_match_final_artifact_ledger(tmp_path: Path) -> None:
    config, _, reference, _, _, reference_run = _setup(tmp_path)
    ledger_path = reference_run.parent / "artifact_hashes.csv"
    with ledger_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["path"] == "m3c1_synthetic_sectionwise.txt":
            row["sha256"] = "0" * 64
    with ledger_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"))
        writer.writeheader()
        writer.writerows(rows)
    receipt = json.loads(reference_run.read_text(encoding="utf-8"))
    receipt["artifact_hashes_sha256"] = _sha256(ledger_path)
    _write_json(reference_run, receipt)

    with pytest.raises(ValueError, match="differs from the final ledger"):
        characterize(
            config,
            "comsol_common_canonical_p1",
            *reference,
            reference_run_report=reference_run,
        )


def test_reference_initial_primitive_validation_must_pass(tmp_path: Path) -> None:
    config, _, reference, _, _, reference_run = _setup(tmp_path)
    receipt = json.loads(reference_run.read_text(encoding="utf-8"))
    runs = _mapping(receipt["runs"])
    validation = _mapping(_mapping(runs[RUN_KEYS[0]])["initial_primitive_validation"])
    validation["status"] = "FAIL"
    _write_json(reference_run, receipt)

    with pytest.raises(ValueError, match="initial primitive status differs from the lock"):
        characterize(
            config,
            "comsol_common_canonical_p1",
            *reference,
            reference_run_report=reference_run,
        )


def test_reference_initial_state_summary_must_pass(tmp_path: Path) -> None:
    config, _, reference, _, _, reference_run = _setup(tmp_path)
    summary_path = reference_run.parent / "normalization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = _mapping(summary["runs"])
    validation = _mapping(_mapping(runs[RUN_KEYS[0]])["initial_state"])
    validation["status"] = "FAIL"
    _write_json(summary_path, summary)
    ledger_path = reference_run.parent / "artifact_hashes.csv"
    with ledger_path.open(encoding="utf-8", newline="") as stream:
        ledger = list(csv.DictReader(stream))
    for row in ledger:
        if row["path"] == summary_path.name:
            row["sha256"] = _sha256(summary_path)
            row["bytes"] = str(summary_path.stat().st_size)
    with ledger_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"))
        writer.writeheader()
        writer.writerows(ledger)
    receipt = json.loads(reference_run.read_text(encoding="utf-8"))
    receipt["artifact_hashes_sha256"] = _sha256(ledger_path)
    _write_json(reference_run, receipt)

    with pytest.raises(ValueError, match="initial state validation did not pass"):
        characterize(
            config,
            "comsol_common_canonical_p1",
            *reference,
            reference_run_report=reference_run,
        )


def test_candidate_engine_revision_must_match_evaluation_lock(tmp_path: Path) -> None:
    config, candidate, _, candidate_run, prepare_report, _ = _setup(tmp_path)
    receipt = json.loads(candidate_run.read_text(encoding="utf-8"))
    runs = _mapping(receipt["runs"])
    record = _mapping(runs[RUN_KEYS[0]])
    result_path = candidate_run.parent / str(record["result"])
    manifest_path = result_path / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["engine_algorithm_revision"] = "engine_v2"
    manifest["engine_algorithm_revision"] = "engine_v2"
    _write_json(manifest_path, manifest)
    record["result_manifest_sha256"] = _sha256(manifest_path)
    _write_json(candidate_run, receipt)

    with pytest.raises(ValueError, match="engine revision differs from the lock"):
        characterize(
            config,
            "solver_canonical_p1_projection",
            *candidate,
            candidate_run_report=candidate_run,
            prepare_report=prepare_report,
        )


def test_roundoff_plateau_is_reported_without_order_claim(tmp_path: Path) -> None:
    setup = _setup(tmp_path, candidate_coefficient=0.0, reference_coefficient=0.0)
    _, reference_report = _characterizations(*setup)

    assert reference_report["status"] == "PASS"
    assessments = _mapping(reference_report["rms_order_assessment"])
    for quantity in ("position", "velocity", "charge"):
        assessment = _mapping(assessments[quantity])
        assert assessment["classification"] == "ROUNDOFF_LIMITED"
        assert assessment["order_evaluated"] is False
        assert assessment["effective_order"] == "NOT_EVALUATED_ROUNDOFF_PLATEAU"


def test_register_accepts_nonbitwise_initial_state_inside_locked_roundoff(
    tmp_path: Path,
) -> None:
    setup = _setup(tmp_path, reference_initial_shift=1.0e-15)
    config, *_ = setup
    candidate_report, reference_report = _characterizations(*setup)
    candidate_path = tmp_path / "candidate_report.json"
    reference_path = tmp_path / "reference_report.json"
    _write_json(candidate_path, candidate_report)
    _write_json(reference_path, reference_report)

    budget = register_budget(config, candidate_path, reference_path)

    assert budget["status"] == "REGISTERED"
    gate = _mapping(budget["initial_state_gate"])
    assert gate["status"] == "PASS"
    assert gate["bitwise_identical"] is False
    assert gate["roundoff_multiplier"] == 4096.0


def test_register_does_not_parse_post_t0_state_values(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    config, candidate, *_ = setup
    candidate_report, reference_report = _characterizations(*setup)
    fine = candidate[-1]
    with fine.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    post_t0 = next(row for row in rows if float(row["time_s"]) > 0.0)
    post_t0["r_m"] = "POST_T0_STATE_MUST_NOT_BE_PARSED"
    with fine.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    artifacts = _mapping(candidate_report["artifacts"])
    _mapping(artifacts[STEP_NAMES[2]])["sha256"] = _sha256(fine)
    candidate_path = tmp_path / "candidate_report.json"
    reference_path = tmp_path / "reference_report.json"
    _write_json(candidate_path, candidate_report)
    _write_json(reference_path, reference_report)

    budget = register_budget(config, candidate_path, reference_path)

    assert budget["status"] == "REGISTERED"
    assert _mapping(budget["policy"])["initial_state_scope"] == (
        "T0_ONLY_NO_POST_T0_CROSS_DIFFERENCE_READ"
    )


def test_register_blocks_initial_state_outside_locked_roundoff(tmp_path: Path) -> None:
    setup = _setup(tmp_path, reference_initial_shift=1.0e-6)
    config, _, _, *_ = setup
    candidate_report, reference_report = _characterizations(*setup)
    candidate_path = tmp_path / "candidate_report.json"
    reference_path = tmp_path / "reference_report.json"
    _write_json(candidate_path, candidate_report)
    _write_json(reference_path, reference_report)

    budget = register_budget(config, candidate_path, reference_path)

    assert budget["status"] == "BLOCKED"
    blockers = budget["blockers"]
    assert isinstance(blockers, list)
    assert "candidate and reference t=0 states exceed the locked roundoff criterion" in blockers


def test_native_field_v4_report_cannot_enter_common_field_budget(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    config, *_ = setup
    candidate_report, reference_report = _characterizations(*setup)
    candidate_path = tmp_path / "candidate_report.json"
    reference_path = tmp_path / "native_v4_report.json"
    _write_json(candidate_path, candidate_report)
    reference_report["tool_revision"] = "m3c1_cross_representation_pre_event_v4"
    _write_json(reference_path, reference_report)

    with pytest.raises(ValueError, match="incompatible common-field report"):
        register_budget(config, candidate_path, reference_path)


def test_production_config_locks_audited_source_and_pre_run_tolerance_basis() -> None:
    config_path = Path(__file__).parents[1] / "cases" / "m3c1_caseA_100nm_common_p1_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate = _mapping(config["candidate"])
    reference = _mapping(config["reference"])
    reference_validation = _mapping(reference["validation"])
    acceptance = _mapping(config["acceptance"])
    basis = _mapping(acceptance["limit_basis"])

    assert candidate["producer_tool_revision"] == (
        "m3c1_solver_canonical_p1_projection_candidate_v3"
    )
    assert candidate["configuration_sha256"] == (
        "acdee16230e8330a9012ec0c817ac4022fc8b28ec6449704a94cbfa35d642e4d"
    )
    assert candidate["input_file_sha256"] == (
        "14cede2c85e7888368c04da000c5fcaf50cf2135c85be0497633b59d83febea1"
    )
    assert candidate["input_content_hash"] == (
        "sha256:d30e9048cf8e142c3689508f0de2f20c503e7e787c80809e9fbe1806c8b45a1c"
    )
    assert candidate["engine_algorithm_revision"] == "particle_engine_v31"
    assert set(_mapping(candidate["physics_models"])) == {
        "charge",
        "dielectrophoresis",
        "drag",
        "electric",
        "gravity_buoyancy",
        "ion_drag",
        "lift",
        "thermophoresis",
    }
    assert reference["source_mph_sha256"] == (
        "3bbf08e3469758313eac5de473a7a0dd4cc9a6f72c9722393229f0b856e9b524"
    )
    assert reference_validation == {
        "prepared_table_artifact_count": 26,
        "initial_state_roundoff_multiplier": 4096.0,
        "initial_primitive_component_count": 22,
        "initial_primitive_value_count": 6314,
        "initial_primitive_roundoff_multiplier": 4096.0,
    }
    assert basis["source_sha256"] == (
        "b9c2d044c1f01ea0aa7d43885521c82ac54e0802c6737d5e8e1780ceaf5095b7"
    )
    assert basis["common_field_cross_difference_was_read"] is False
    assert acceptance["result_dependent_tolerance_tuning"] == "PROHIBITED"
