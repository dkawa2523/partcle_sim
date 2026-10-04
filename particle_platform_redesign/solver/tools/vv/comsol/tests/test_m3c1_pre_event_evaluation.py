from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from tools.vv.comsol.evaluate_m3c1_pre_event import (
    RUN_KEYS,
    STEP_NAMES,
    characterize,
    compare,
    register_budget,
)

_STEPS = (6.25e-7, 3.125e-7, 1.5625e-7)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _float(value: object) -> float:
    assert isinstance(value, int | float) and not isinstance(value, bool)
    return float(value)


def _write_trajectory(
    path: Path,
    *,
    dt_s: float,
    coefficient: float,
    error_order: float = 2.0,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "particle_id",
                "time_s",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number_e",
                "lifecycle",
            ]
        )
        for frame in range(4):
            time_s = frame * 1.0e-5
            error = coefficient * dt_s**error_order * frame / 3.0
            for particle_id in range(1, 4):
                writer.writerow(
                    [
                        particle_id,
                        time_s,
                        0.14 + particle_id * 1.0e-5 + 0.01 * time_s + error,
                        0.023 + particle_id * 2.0e-6 + 0.02 * time_s - 0.5 * error,
                        0.01 + particle_id * 1.0e-6 + error,
                        0.02 - particle_id * 1.0e-6 - 0.25 * error,
                        -1.0 + 1000.0 * time_s + 0.5 * error,
                        "active",
                    ]
                )


def _write_candidate_receipts(
    root: Path,
    trajectories: list[Path],
) -> tuple[Path, Path]:
    input_path = root / "candidate_input.h5"
    input_path.write_bytes(b"synthetic canonical input")
    content_hash = "sha256:synthetic-content"
    cases: dict[str, str] = {}
    runs: dict[str, object] = {}
    physics = {"electric": {"model": "coulomb", "revision": "electric_v1"}}
    for run_key, trajectory, dt_s in zip(RUN_KEYS, trajectories, _STEPS, strict=True):
        case_path = root / f"candidate_{run_key}.yaml"
        case_path.write_text(f"name: {run_key}\n", encoding="utf-8")
        result_path = root / f"result_{run_key}"
        result_path.mkdir()
        manifest_path = result_path / "run.json"
        manifest = {
            "status": "complete",
            "case_file_hash": f"sha256:{_sha256(case_path)}",
            "data_content_hash": content_hash,
            "engine_algorithm_revision": "engine_v1",
            "result_algorithm_revision": "result_v1",
            "resolved": {"physics_models": physics},
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
            "engine_algorithm_revision": "engine_v1",
            "physics_models": physics,
        }
    prepare_path = root / "prepare_report.json"
    _write_json(
        prepare_path,
        {
            "tool_revision": "candidate_builder_v1",
            "input_data": input_path.name,
            "input_file_sha256": _sha256(input_path),
            "input_content_hash": content_hash,
            "cases": cases,
        },
    )
    run_path = root / "candidate_run_report.json"
    _write_json(
        run_path,
        {
            "tool_revision": "candidate_builder_v1",
            "status": "complete",
            "decision": "COMPLETE",
            "runs": runs,
        },
    )
    return run_path, prepare_path


def _setup(tmp_path: Path) -> tuple[Path, list[Path], list[Path], Path, Path]:
    candidate_root = tmp_path / "candidate"
    reference_root = tmp_path / "reference"
    candidate_root.mkdir()
    reference_root.mkdir()
    candidate: list[Path] = []
    reference: list[Path] = []
    for name, dt_s in zip(STEP_NAMES, _STEPS, strict=True):
        candidate_path = candidate_root / f"candidate_{name}.csv"
        reference_path = reference_root / f"reference_{name}.csv"
        _write_trajectory(candidate_path, dt_s=dt_s, coefficient=1000.0)
        _write_trajectory(reference_path, dt_s=dt_s, coefficient=1500.0)
        candidate.append(candidate_path)
        reference.append(reference_path)
    config_path = tmp_path / "config.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "evaluation_id": "M3-C1-caseA-100nm-pre-event",
            "labels": {
                "candidate": "solver_canonical_p1_projection",
                "reference": "comsol_native_field",
            },
            "scope": {
                "particles": 3,
                "frames": 4,
                "output_interval_s": 1.0e-5,
                "time_window_s": [0.0, 3.0e-5],
                "internal_steps_s": list(_STEPS),
            },
            "reference": {
                "source": "synthetic locked COMSOL reference",
                "trajectory_sha256_by_step": {
                    name: _sha256(path) for name, path in zip(STEP_NAMES, reference, strict=True)
                },
            },
            "acceptance": {
                "minimum_rms_order": 0.75,
                "maximum_fine_pair_relative_l2": {
                    "position": 1.0,
                    "velocity": 1.0,
                    "charge": 1.0,
                },
                "candidate_precision_vs_reference_factor": 2.0,
                "roundoff_multiplier": 2048.0,
            },
        },
    )
    run_report, prepare_report = _write_candidate_receipts(candidate_root, candidate)
    return config_path, candidate, reference, run_report, prepare_report


def _lock_reference_hashes(config_path: Path, trajectories: list[Path]) -> None:
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["reference"]["trajectory_sha256_by_step"] = {
        name: _sha256(path) for name, path in zip(STEP_NAMES, trajectories, strict=True)
    }
    _write_json(config_path, document)


def _rewrite_run_manifest(
    run_path: Path,
    run_key: str,
    updates: dict[str, object],
) -> None:
    receipt = json.loads(run_path.read_text(encoding="utf-8"))
    record = _mapping(_mapping(receipt["runs"])[run_key])
    result_name = record["result"]
    assert isinstance(result_name, str)
    manifest_path = run_path.parent / result_name / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(updates)
    _write_json(manifest_path, manifest)
    record["result_manifest_sha256"] = _sha256(manifest_path)
    _write_json(run_path, receipt)


def test_candidate_requires_complete_producer_receipts(tmp_path: Path) -> None:
    config, candidate, _, _, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="requires both producer receipts"):
        characterize(
            config,
            "solver_canonical_p1_projection",
            *candidate,
        )


def test_wrong_candidate_trajectory_hash_is_rejected(tmp_path: Path) -> None:
    config, candidate, _, run_path, prepare_path = _setup(tmp_path)
    receipt = json.loads(run_path.read_text(encoding="utf-8"))
    receipt["runs"][RUN_KEYS[0]]["trajectory_sha256"] = "0" * 64
    _write_json(run_path, receipt)

    with pytest.raises(ValueError, match="hash differs"):
        characterize(
            config,
            "solver_canonical_p1_projection",
            *candidate,
            candidate_run_report=run_path,
            prepare_report=prepare_path,
        )


@pytest.mark.parametrize(
    "steps",
    (
        (_STEPS[0], _STEPS[0], _STEPS[2]),
        (_STEPS[0], _STEPS[2], _STEPS[1]),
        (0.0, _STEPS[1], _STEPS[2]),
        (float.fromhex("0x0.0000000000001p-1022"), 0.0, 0.0),
    ),
)
def test_internal_steps_must_be_distinct_exact_halvings(
    tmp_path: Path,
    steps: tuple[float, float, float],
) -> None:
    config, _, reference, _, _ = _setup(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["scope"]["internal_steps_s"] = list(steps)
    _write_json(config, document)

    with pytest.raises(ValueError, match="exact positive h, h/2, h/4"):
        characterize(config, "comsol_native_field", *reference)


def test_nonzero_scope_start_is_rejected_by_zero_origin_evaluator(tmp_path: Path) -> None:
    config, _, reference, _, _ = _setup(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["scope"]["time_window_s"] = [1.0e-5, 4.0e-5]
    _write_json(config, document)

    with pytest.raises(ValueError, match="must start at zero"):
        characterize(config, "comsol_native_field", *reference)


def test_mislabeled_candidate_step_is_rejected(tmp_path: Path) -> None:
    config, candidate, _, run_path, prepare_path = _setup(tmp_path)
    _rewrite_run_manifest(
        run_path,
        RUN_KEYS[0],
        {"time": {"dt_s": _STEPS[1], "start_s": 0.0, "end_s": 3.0e-5}},
    )

    with pytest.raises(ValueError, match=r"time\.dt_s differs"):
        characterize(
            config,
            "solver_canonical_p1_projection",
            *candidate,
            candidate_run_report=run_path,
            prepare_report=prepare_path,
        )


def test_wrong_case_hash_in_result_manifest_is_rejected(tmp_path: Path) -> None:
    config, candidate, _, run_path, prepare_path = _setup(tmp_path)
    _rewrite_run_manifest(
        run_path,
        RUN_KEYS[0],
        {"case_file_hash": f"sha256:{'0' * 64}"},
    )

    with pytest.raises(ValueError, match="case_file_hash differs"):
        characterize(
            config,
            "solver_canonical_p1_projection",
            *candidate,
            candidate_run_report=run_path,
            prepare_report=prepare_path,
        )


@pytest.mark.parametrize(
    "time_value, field",
    (
        ({"dt_s": _STEPS[0], "start_s": 1.0e-9, "end_s": 3.0e-5}, "start_s"),
        ({"dt_s": _STEPS[0], "start_s": 0.0, "end_s": 2.9e-5}, "end_s"),
    ),
)
def test_wrong_candidate_time_window_is_rejected(
    tmp_path: Path,
    time_value: dict[str, float],
    field: str,
) -> None:
    config, candidate, _, run_path, prepare_path = _setup(tmp_path)
    _rewrite_run_manifest(run_path, RUN_KEYS[0], {"time": time_value})

    with pytest.raises(ValueError, match=rf"time\.{field} differs"):
        characterize(
            config,
            "solver_canonical_p1_projection",
            *candidate,
            candidate_run_report=run_path,
            prepare_report=prepare_path,
        )


def test_locked_cross_representation_receipts_can_proceed(tmp_path: Path) -> None:
    config, candidate, reference, run_path, prepare_path = _setup(tmp_path)
    candidate_report = characterize(
        config,
        "solver_canonical_p1_projection",
        *candidate,
        candidate_run_report=run_path,
        prepare_report=prepare_path,
    )
    reference_report = characterize(config, "comsol_native_field", *reference)
    assert candidate_report["status"] == "PASS"
    assert reference_report["status"] == "PASS"
    separation = _mapping(candidate_report["claim_separation"])
    assert separation["same_field_solver_agreement"] == "NOT_TESTED"

    candidate_report_path = tmp_path / "candidate.json"
    reference_report_path = tmp_path / "reference.json"
    _write_json(candidate_report_path, candidate_report)
    _write_json(reference_report_path, reference_report)
    budget = register_budget(config, candidate_report_path, reference_report_path)
    assert budget["status"] == "REGISTERED"
    budget_path = tmp_path / "budget.json"
    _write_json(budget_path, budget)

    comparison = compare(config, budget_path, candidate[-1], reference[-1])
    assert comparison["status"] == "PASS"
    assert comparison["report_kind"] == "m3c1_locked_cross_representation_trajectory_comparison"
    accuracy_claim = _mapping(comparison["accuracy_claim"])
    assert accuracy_claim["same_field_solver_agreement"] == "NOT_TESTED"


def test_roundoff_plateau_passes_without_claiming_observed_order(tmp_path: Path) -> None:
    config, _, reference, _, _ = _setup(tmp_path)
    for path, dt_s in zip(reference, _STEPS, strict=True):
        _write_trajectory(path, dt_s=dt_s, coefficient=0.0)
    _lock_reference_hashes(config, reference)

    report = characterize(config, "comsol_native_field", *reference)

    assert report["status"] == "PASS"
    separation = _mapping(report["claim_separation"])
    assert separation["roundoff_limited_order"] == ("NOT_ESTABLISHED_BELOW_ROUNDOFF_FLOOR")
    assessments = _mapping(report["rms_order_assessment"])
    for quantity in ("position", "velocity", "charge"):
        assessment = _mapping(assessments[quantity])
        assert assessment["classification"] == "ROUNDOFF_LIMITED"
        assert assessment["order_evaluated"] is False
        assert assessment["observed_order"] is None
        assert assessment["effective_order"] == "NOT_EVALUATED_ROUNDOFF_PLATEAU"
        assert assessment["fine_pair_rms"] == 0.0
        assert assessment["fine_pair_maximum"] == 0.0
        assert assessment["pass"] is True
        assert assessment["reason"] == (
            "both_adjacent_pair_rms_and_maximum_at_or_below_roundoff_floor"
        )


def test_above_roundoff_low_order_remains_blocked(tmp_path: Path) -> None:
    config, _, reference, _, _ = _setup(tmp_path)
    for path, dt_s in zip(reference, _STEPS, strict=True):
        _write_trajectory(
            path,
            dt_s=dt_s,
            coefficient=1.0e-6,
            error_order=0.5,
        )
    _lock_reference_hashes(config, reference)

    report = characterize(config, "comsol_native_field", *reference)

    assert report["status"] == "BLOCKED"
    assessments = _mapping(report["rms_order_assessment"])
    for quantity in ("position", "velocity", "charge"):
        assessment = _mapping(assessments[quantity])
        assert assessment["classification"] == "ORDER_EVALUATED"
        assert assessment["order_evaluated"] is True
        assert 0.49 < _float(assessment["observed_order"]) < 0.51
        assert _float(assessment["fine_pair_rms"]) > _float(assessment["roundoff_floor"])
        assert _float(assessment["fine_pair_maximum"]) > _float(assessment["roundoff_floor"])
        assert assessment["pass"] is False
        assert assessment["reason"] == "observed_order_unavailable_or_below_minimum"
