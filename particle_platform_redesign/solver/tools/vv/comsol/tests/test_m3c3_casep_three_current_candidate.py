from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from tools.vv.comsol import run_m3c3_casep_three_current_candidate as candidate_module
from tools.vv.comsol.evaluate_m3c3_casep_three_current import (
    COMSOL_REFERENCE_LEVELS,
    _candidate_events_path,
    evaluate,
)
from tools.vv.comsol.run_m3c3_casep_three_current_candidate import (
    EVENT_HEADER,
    LEVELS,
    OUTPUT_COUNT,
    PARTICLE_COUNT,
    TRAJECTORY_HEADER,
    prepare,
)
from tools.vv.comsol.tests.common_p1_fixture import (
    write_current_common_p1_template,
    write_manufactured_common_p1,
    write_synthetic_meaning_inventory,
)

from chamber_particles import load_case
from chamber_particles.case_format import read_with_info

CONFIG = Path(__file__).resolve().parents[1] / "cases" / "m3c3_caseP_100nm_three_current_v1.json"
SOLVER_ROOT = Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _times() -> list[float]:
    return (
        [index * 1.0e-5 for index in range(51)]
        + [index * 1.0e-4 for index in range(6, 51)]
        + [index * 1.0e-3 for index in range(6, 31)]
    )


def _write_trajectory(
    path: Path,
    error: float,
    event_time_s: float | None = None,
    *,
    zero_stuck_velocity: bool = False,
) -> float:
    maximum_speed = 0.0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(TRAJECTORY_HEADER)
        for time_s in _times():
            for particle_id in range(1, PARTICLE_COUNT + 1):
                stuck = (
                    event_time_s is not None and particle_id in {1, 2} and time_s >= event_time_s
                )
                velocity_r = 0.1 + 1.0e-5 * error
                velocity_z = -0.05
                if stuck and zero_stuck_velocity:
                    velocity_r = 0.0
                    velocity_z = 0.0
                maximum_speed = max(maximum_speed, float(np.hypot(velocity_r, velocity_z)))
                writer.writerow(
                    (
                        particle_id,
                        time_s,
                        0.01 + particle_id * 1.0e-7 + time_s * 0.1 + error,
                        0.02 - time_s * 0.05,
                        velocity_r,
                        velocity_z,
                        -500.0 + 1.0e4 * error,
                        "stuck" if stuck else "active",
                    )
                )
    return maximum_speed


def _write_events(path: Path, event_time_s: float | None) -> int:
    rows = []
    if event_time_s is not None:
        rows = [
            (
                particle_id,
                event_time_s + particle_id * 1.0e-8,
                "terminal_boundary",
                "stuck",
                "material_stick_boundary_unspecified",
            )
            for particle_id in (1, 2)
        ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(EVENT_HEADER)
        writer.writerows(rows)
    return len(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def current_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repository = tmp_path / "repository"
    solver = repository / "particle_platform_redesign" / "solver"
    solver.mkdir(parents=True)
    input_path = solver / "input.h5"
    info = write_manufactured_common_p1(input_path, negative_ions=True)
    template = solver / "template.yaml"
    write_current_common_p1_template(
        SOLVER_ROOT / config["candidate_template"]["solver_relative_path"],
        template,
        info.content_hash,
    )
    receipt = solver / "primitive_receipt.json"
    _write_json(
        receipt, {"scope": "manufactured_fixture_not_COMSOL_export", "comsol_executed": False}
    )
    source = repository / "identity_only_source.mph"
    source.write_bytes(b"manufactured source identity fixture; never opened by COMSOL")
    config["primitive_input"] = {
        "solver_relative_path": input_path.name,
        "sha256": _sha256(input_path),
        "content_hash": info.content_hash,
    }
    config["primitive_receipt"] = {"solver_relative_path": receipt.name, "sha256": _sha256(receipt)}
    config["candidate_template"] = {
        "solver_relative_path": template.name,
        "sha256": _sha256(template),
    }
    config["source_mph"] = {"repository_relative_path": source.name, "sha256": _sha256(source)}
    path = solver / "campaign.json"
    _write_json(path, config)
    monkeypatch.setattr(candidate_module, "_solver_root", lambda: solver)
    monkeypatch.setattr(candidate_module, "_repository_root", lambda: repository)
    return path


@pytest.fixture
def prepared_campaign(tmp_path: Path, current_configuration: Path) -> Path:
    prepared = tmp_path / "prepared"
    prepare(current_configuration, prepared)
    return prepared


def test_prepare_owns_one_three_current_release_state(prepared_campaign: Path) -> None:
    prepared = prepared_campaign
    report = json.loads((prepared / "prepare_report.json").read_text())

    assert report["status"] == "PREPARED"
    derived, info = read_with_info(prepared / "candidate_input_three_current_z0.h5")
    assert info.content_hash == report["derived_input_content_hash"]
    assert derived.sources[0].particle_id.size == PARTICLE_COUNT
    assert not np.array_equal(derived.sources[0].charge_number, np.full(PARTICLE_COUNT, -1.0))

    with (prepared / "three_current_release_state.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        release = list(csv.DictReader(stream, strict=True))
    csv_charge = np.asarray([float(row["charge_number_e"]) for row in release])
    np.testing.assert_array_equal(csv_charge, derived.sources[0].charge_number)

    reference = json.loads((prepared / "reference_run_config.json").read_text())
    assert reference["release_state"]["sha256"] == report["release_state_sha256"]
    assert reference["maximum_relative_ion_speed_m_s"] == 1.0e6


def test_prepare_records_equilibrium_and_speed_proof(prepared_campaign: Path) -> None:
    receipt = json.loads((prepared_campaign / "three_current_release_receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["single_charge_authority"] is True
    assert receipt["comsol_recomputes_equilibrium"] is False
    assert receipt["root_solution"]["converged"] is True
    assert receipt["root_solution"]["maximum_abs_newton_correction_charge_number"] < 1e-10
    assert receipt["effective_gas_speed_ratio"]["configured_maximum_speed_ratio"] == 1.0
    assert receipt["effective_gas_speed_ratio"]["initial_release_maximum_speed_ratio"] < 0.25
    for ion in ("positive_ion", "negative_ion"):
        envelope = receipt["speed_envelope"][ion]
        assert envelope["relative_speed_bound_m_s"] < 1.0e6
        assert envelope["clipping"] is False


def test_prepare_emits_three_exponential_midpoint_cases(prepared_campaign: Path) -> None:
    for level, dt_s in LEVELS:
        case_path = prepared_campaign / "candidate" / level / "case.yaml"
        load_case(case_path)
        document = yaml.safe_load(case_path.read_text())
        assert document["time"]["dt_s"] == dt_s
        assert document["solver"]["integrator"] == "exponential_midpoint"
        assert document["solver"]["event"]["geometry_rtol"] == 1.0e-9
        assert "noise" not in document["physics"]
        assert document["physics"]["charge"]["revision"].endswith("three_current_v1")
        assert document["physics"]["drag"]["maximum_speed_ratio"] == 1.0
        assert document["physics"]["thermophoresis"]["maximum_speed_ratio"] == 1.0


def test_evaluate_rejects_post_prepare_gate_change(prepared_campaign: Path) -> None:
    config_path = prepared_campaign / "campaign_config.json"
    config = json.loads(config_path.read_text())
    config["acceptance"]["direct_uncertainty_multiplier"] = 5.0
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="preparation hash binding"):
        evaluate(
            prepared_campaign,
            tuple(prepared_campaign / name for name, _step_s in COMSOL_REFERENCE_LEVELS),
            prepared_campaign / "eval.json",
        )


def _graft_reference(root: Path, graft: str) -> None:
    summary_path = root / "normalization_summary.json"
    summary = json.loads(summary_path.read_text())
    if graft in {"unknown_revision", "summary_hash"}:
        summary["tool_revision"] = (
            "unrecognized_revision"
            if graft == "unknown_revision"
            else "m3c3_caseP_three_current_comsol_normalizer_v3"
        )
    if graft in {"model", "field"}:
        inventory, _ = write_synthetic_meaning_inventory(
            root,
            "2" * 64 if graft == "model" else summary["source_model_sha256"],
            "sha256:" + "c" * 64 if graft == "field" else summary["canonical_input_content_hash"],
        )
        summary["meaning_preflight_inventory"] = inventory
    if graft == "receipt":
        actual = json.loads((root / "actual_binding_receipt.json").read_text())
        actual["unrelated_run"] = True
        _write_json(root / "actual_binding_receipt.json", actual)
    if graft == "config_summary":
        summary["reference_run_config_sha256"] = "0" * 64
    _write_json(summary_path, summary)
    if graft != "summary_hash":
        receipt_path = root / "run_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if graft == "config_receipt":
            receipt["reference_run_config_sha256"] = "0" * 64
        receipt["normalization_summary"]["sha256"] = _sha256(summary_path)
        _write_json(receipt_path, receipt)


@pytest.mark.parametrize(
    "graft",
    [
        None,
        "unknown_revision",
        "summary_hash",
        "model",
        "field",
        "receipt",
        "config_summary",
        "config_receipt",
    ],
)
def test_evaluate_applies_predeclared_order_and_direct_gates(
    tmp_path: Path, current_configuration: Path, graft: str | None
) -> None:
    prepared = tmp_path / "prepared"
    prepare(current_configuration, prepared)
    prepare_report = json.loads((prepared / "prepare_report.json").read_text())
    errors = {level: 4.0e-9 / (2**index) for index, (level, _dt_s) in enumerate(LEVELS)}
    event_errors = {level: 4.0e-7 / (2**index) for index, (level, _dt_s) in enumerate(LEVELS)}
    for level, _dt_s in LEVELS:
        cell = prepared / "candidate" / level
        trajectory = cell / "trajectory.csv"
        event_time_s = 7.3e-4 + event_errors[level]
        maximum_speed = _write_trajectory(
            trajectory, errors[level], event_time_s, zero_stuck_velocity=True
        )
        events = cell / "events.csv"
        event_count = _write_events(events, event_time_s)
        _write_json(
            cell / "run_receipt.json",
            {
                "status": "PASS",
                "trajectory_rows": PARTICLE_COUNT * OUTPUT_COUNT,
                "trajectory_sha256": _sha256(trajectory),
                "events_sha256": _sha256(events),
                "event_rows": event_count,
                "derived_input_sha256": prepare_report["derived_input_sha256"],
                "release_state_sha256": prepare_report["release_state_sha256"],
                "maximum_observed_particle_speed_m_s": maximum_speed,
                "result_counts": {
                    "boundary_events": event_count,
                    "failure_events": 0,
                },
            },
        )

    references: list[Path] = []
    for index, (level, step_s) in enumerate(COMSOL_REFERENCE_LEVELS):
        reference = tmp_path / f"reference_{level}"
        reference.mkdir()
        references.append(reference)
        shutil.copyfile(
            prepared / "three_current_release_state.csv",
            reference / "three_current_release_state.csv",
        )
        trajectory = reference / "trajectory_reference.csv"
        reference_event_time_s = 7.3e-4 + 4.0e-8 / (2**index)
        _write_trajectory(trajectory, 4.0e-10 / (2**index), reference_event_time_s)
        events = reference / "events_reference.csv"
        event_count = _write_events(events, reference_event_time_s)
        _write_json(
            reference / "normalization_summary.json",
            {
                "tool_revision": "m3c3_caseP_three_current_comsol_normalizer_v4",
                "reference_run_config_sha256": prepare_report["reference_config_sha256"],
                "source_model_sha256": json.loads(
                    (prepared / "reference_run_config.json").read_text()
                )["source_mph"]["sha256"],
                "canonical_input_content_hash": prepare_report["derived_input_content_hash"],
                "meaning_preflight_inventory": write_synthetic_meaning_inventory(
                    reference,
                    json.loads((prepared / "reference_run_config.json").read_text())["source_mph"][
                        "sha256"
                    ],
                    prepare_report["derived_input_content_hash"],
                )[0],
                "actual_run_readback": {
                    "path": "actual_binding_receipt.json",
                    "sha256": _sha256(reference / "actual_binding_receipt.json"),
                    "observation": "EXPORTED",
                },
                "status": "COMPLETE_NORMALIZED_NOT_EVALUATED",
                "case_id": "caseP_100nm_three_current",
                "particle_count": PARTICLE_COUNT,
                "output_count": OUTPUT_COUNT,
                "trajectory_rows": PARTICLE_COUNT * OUTPUT_COUNT,
                "event_count": event_count,
                "fixed_rk4_step_s": step_s,
                "artifacts": {
                    "trajectory_reference.csv": _sha256(trajectory),
                    "events_reference.csv": _sha256(events),
                },
            },
        )
        _write_json(
            reference / "run_receipt.json",
            {
                "status": "PASS",
                "case_id": "caseP_100nm_three_current",
                "reference_run_config_sha256": prepare_report["reference_config_sha256"],
                "normalization_summary": {
                    "path": "normalization_summary.json",
                    "sha256": _sha256(reference / "normalization_summary.json"),
                },
            },
        )

    output = tmp_path / "evaluation.json"
    if graft is not None:
        _graft_reference(references[0], graft)
        with pytest.raises(ValueError):
            evaluate(prepared, tuple(references), output)
        assert not output.exists()
        return
    evaluate(prepared, tuple(references), output)
    report = json.loads(output.read_text())
    expected_values = (
        (report["status"], "PASS"),
        (report["self_convergence"]["status"], "PASS"),
        (report["comsol_reference_convergence"]["status"], "PASS"),
        (report["candidate_fine_vs_comsol"]["status"], "PASS"),
        (report["boundary_event_self_convergence"]["status"], "PASS"),
        (
            report["boundary_event_self_convergence"]["observed_rms_order"],
            pytest.approx(1.0),
        ),
        (report["comsol_boundary_event_convergence"]["status"], "PASS"),
        (report["boundary_events_fine_vs_comsol"]["status"], "PASS"),
        (report["gate_policy"]["minimum_rms_order"], 0.75),
    )
    for actual, expected in expected_values:
        assert actual == expected
    assert report["boundary_events_fine_vs_comsol"]["identity_exact_required"] is True
    assert report["candidate_fine_vs_comsol"]["lifecycle_exact"] is True
    assert report["observed_particle_speed_m_s"]["preparation_envelope_is_not_runtime_gate"]


def test_event_projection_keeps_observed_id_groups_and_rejects_unknown_ids(tmp_path: Path) -> None:
    meaning = tmp_path / "boundary_meaning.json"
    _write_json(meaning, {"boundary_groups": {"8": "grounded_wall", "6": "wafer"}})
    events = SimpleNamespace(
        particle_id=np.array([1, 2]),
        boundary_id=np.array([8, 6]),
        outcome=np.array(["stuck", "stuck"]),
        time_s=np.array([0.001, 0.002]),
    )
    result = SimpleNamespace(read_boundary_events=lambda: events)
    path = tmp_path / "events.csv"
    assert candidate_module._write_events(path, result, meaning) == 2
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["boundary_semantic"] for row in rows] == ["grounded_wall", "wafer"]
    events.boundary_id[1] = 999
    with pytest.raises(ValueError, match="no canonical semantic group"):
        candidate_module._write_events(tmp_path / "unknown.csv", result, meaning)


def test_reporting_reprojection_preserves_execution_and_binds_repaired_rows(
    prepared_campaign: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    level = LEVELS[0][0]
    cell = prepared_campaign / "candidate" / level
    result_path = cell / "result"
    result_path.mkdir()
    _write_json(result_path / "run.json", {"fixture": "manufactured_not_simulated"})
    data, _info = read_with_info(prepared_campaign / "candidate_input_three_current_z0.h5")
    boundary_id = int(data.geometry.boundary.boundary_id[0])
    group = data.geometry.group_names[int(data.geometry.boundary.group_id[0])]
    events = SimpleNamespace(
        particle_id=np.array([1]),
        boundary_id=np.array([boundary_id]),
        outcome=np.array(["stuck"]),
        time_s=np.array([0.001]),
    )
    monkeypatch.setattr(
        candidate_module,
        "open_result",
        lambda _path: SimpleNamespace(read_boundary_events=lambda: events),
    )
    old_events = cell / "events.csv"
    _write_events(old_events, 0.001)
    receipt: dict[str, object] = {
        "status": "PASS",
        "case_sha256": _sha256(cell / "case.yaml"),
        "events_sha256": _sha256(old_events),
        "event_rows": 1,
        "result_manifest_sha256": _sha256(result_path / "run.json"),
        "derived_input_sha256": _sha256(prepared_campaign / "candidate_input_three_current_z0.h5"),
    }
    receipt_path = cell / "run_receipt.json"
    _write_json(receipt_path, receipt)
    original_bytes = receipt_path.read_bytes(), old_events.read_bytes()
    repair = candidate_module.reproject_events(prepared_campaign, level)
    assert (receipt_path.read_bytes(), old_events.read_bytes()) == original_bytes
    assert repair["candidate_simulated"] is False
    corrected = _candidate_events_path(cell, receipt, prepared_campaign)
    with corrected.open(newline="", encoding="utf-8") as stream:
        assert next(iter(csv.DictReader(stream)))["boundary_semantic"] == group
    repair["result_manifest_sha256"] = "0" * 64
    _write_json(cell / "event_projection_receipt.json", repair)
    with pytest.raises(ValueError, match="immutable execution"):
        _candidate_events_path(cell, receipt, prepared_campaign)
