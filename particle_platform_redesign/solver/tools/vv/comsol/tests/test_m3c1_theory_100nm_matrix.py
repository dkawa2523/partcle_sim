"""Focused checks for the current M3-C1 100 nm deterministic matrix tool."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from tools.vv.comsol import prepare_m3c1_theory_100nm_matrix as matrix

CONFIG = Path(__file__).parents[1] / "cases/m3c1_theory_100nm_30ms_v2.json"


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def _output_times() -> np.ndarray:
    return np.asarray(
        [index * 1.0e-5 for index in range(51)]
        + [index * 1.0e-4 for index in range(6, 51)]
        + [index * 1.0e-3 for index in range(6, 31)],
        dtype=np.float64,
    )


def _write_projection(path: Path, header: tuple[str, ...], row_count: int) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(["0"] * len(header) for _ in range(row_count))


def test_config_preregisters_the_two_case_matrix_and_acceptance() -> None:
    config = matrix._load_config(CONFIG)

    assert set(config["workflows"]) == {"caseA", "caseP"}
    assert config["workflows"]["caseA"]["maximum_neutral_speed_ratio"] == 0.1
    assert config["workflows"]["caseP"]["maximum_neutral_speed_ratio"] == 0.25
    assert config["matrix"]["run_keys"] == ["coarse", "medium", "fine"]
    assert config["workflows"]["caseA"]["fixed_rk4_steps_s"] == [
        6.25e-7,
        3.125e-7,
        1.5625e-7,
    ]
    assert config["workflows"]["caseP"]["fixed_rk4_steps_s"] == [
        4.6875e-8,
        2.34375e-8,
        1.171875e-8,
    ]
    assert config["matrix"]["output_count"] == 121
    assert config["matrix"]["time_end_s"] == 0.03
    assert config["acceptance"] == {
        "initial_state_ulp_multiplier": 4096,
        "minimum_rms_order": 0.75,
        "roundoff_multiplier": 4096,
        "cross_envelope_safety_factor": 2.0,
        "maximum_dt_charge_lipschitz": 0.5,
    }
    assert config["physics"]["brownian_active"] is False
    assert config["physics"]["saffman_active"] is False
    matrix._validate_output_segments(_output_times())


def test_charge_steps_are_admissible_and_divide_thirty_milliseconds() -> None:
    case_a_receipt = matrix._charge_step_receipt("caseA")
    case_a_runs = cast(list[dict[str, Any]], case_a_receipt["runs"])
    assert case_a_receipt["all_steps_admissible"] is True
    assert [row["dt_charge_lipschitz"] for row in case_a_runs] == pytest.approx(
        [0.17938322753018368, 0.08969161376509184, 0.04484580688254592]
    )
    assert [row["step_count_30ms"] for row in case_a_runs] == [
        48000,
        96000,
        192000,
    ]
    case_p_receipt = matrix._charge_step_receipt("caseP")
    case_p_runs = cast(list[dict[str, Any]], case_p_receipt["runs"])
    assert case_p_receipt["all_steps_admissible"] is True
    assert [row["dt_charge_lipschitz"] for row in case_p_runs] == pytest.approx(
        [0.48766203138663356, 0.24383101569331678, 0.12191550784665839]
    )
    assert [row["step_count_30ms"] for row in case_p_runs] == [
        640000,
        1280000,
        2560000,
    ]


def test_case_document_uses_common_physics_and_workflow_specific_sensitivity() -> None:
    config = matrix._load_config(CONFIG)
    output_times = _output_times()
    documents = {
        name: matrix._case_document(
            name,
            config["workflows"][name],
            f"sha256:{name}",
            6.25e-7,
            output_times,
            config,
        )
        for name in matrix.WORKFLOWS
    }

    for name, expected_ratio in (("caseA", 0.1), ("caseP", 0.25)):
        document = documents[name]
        physics = _mapping(document["physics"])
        assert set(physics) == {
            "charge",
            "drag",
            "electric",
            "ion_drag",
            "thermophoresis",
            "dielectrophoresis",
            "lift",
            "gravity_buoyancy",
        }
        assert _mapping(physics["drag"])["maximum_speed_ratio"] == expected_ratio
        assert _mapping(physics["thermophoresis"])["maximum_speed_ratio"] == expected_ratio
        assert _mapping(physics["ion_drag"])["revision"] == (
            "relative_flow_screened_collection_orbital_aggregate_ion_v1"
        )
        boundary_rows = document["boundaries"]
        assert isinstance(boundary_rows, list)
        boundaries = {row["boundary_group"]: row["law"] for row in boundary_rows}
        assert boundaries["gas_inlet"] == "hold"
        assert boundaries["pump_outlet"] == "escape"
        assert all(
            boundaries[group] == "stick"
            for group in ("wafer", "grounded_wall", "focus_transition", "outer_dielectric")
        )
        output = _mapping(document["output"])
        trajectories = _mapping(output["trajectories"])
        schedule = _mapping(trajectories["schedule"])
        assert schedule["explicit_times_s"] == output_times.tolist()


def test_release_source_preserves_all_t0_state_columns(tmp_path: Path) -> None:
    package = tmp_path / "package"
    results = package / "results"
    results.mkdir(parents=True)
    path = results / "release_state_t0_tidy.csv"
    fieldnames = (
        "particle_id",
        "time_s",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_phi_m_per_s",
        "velocity_z_m_per_s",
        "charge_number_e",
        "particle_diameter_m",
        "particle_radius_m",
        "particle_mass_kg",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for particle_id in range(1, matrix.PARTICLE_COUNT + 1):
            writer.writerow(
                {
                    "particle_id": particle_id,
                    "time_s": 0.0,
                    "r_m": 0.1 + particle_id * 1.0e-6,
                    "z_m": 0.02 + particle_id * 2.0e-6,
                    "velocity_r_m_per_s": particle_id * 1.0e-3,
                    "velocity_phi_m_per_s": 0.0,
                    "velocity_z_m_per_s": -particle_id * 2.0e-3,
                    "charge_number_e": -1.0 - particle_id * 1.0e-4,
                    "particle_diameter_m": 1.0e-7,
                    "particle_radius_m": 5.0e-8,
                    "particle_mass_kg": 1.0e-18 + particle_id * 1.0e-22,
                }
            )

    source, receipt = matrix._release_source(package)

    assert source.particle_id.size == matrix.PARTICLE_COUNT
    assert source.position_m[-1].tolist() == pytest.approx([0.100287, 0.020574])
    assert source.velocity_m_s[-1].tolist() == pytest.approx([0.287, -0.574])
    assert source.charge_number[-1] == pytest.approx(-1.0287)
    assert source.mass_kg[-1] == pytest.approx(1.0287e-18)
    assert receipt["preserved_t0_columns"] == [
        "particle_id",
        "time_s",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number_e",
        "particle_diameter_m",
        "particle_radius_m",
        "particle_mass_kg",
    ]


def test_result_projection_includes_hold_events_and_failure_ledger(tmp_path: Path) -> None:
    frame = SimpleNamespace(
        time_s=0.03,
        particle_id=np.asarray([1], dtype=np.int64),
        position_m=np.asarray([[0.1, 0.2]]),
        velocity_m_s=np.asarray([[0.3, 0.4]]),
        charge_number=np.asarray([-2.0]),
        lifecycle=np.asarray([5], dtype=np.uint8),
    )
    events = SimpleNamespace(
        particle_id=np.asarray([1], dtype=np.int64),
        event_ordinal=np.asarray([0], dtype=np.uint32),
        time_s=np.asarray([0.01]),
        position_m=np.asarray([[0.1, 0.2]]),
        normal=np.asarray([[1.0, 0.0]]),
        velocity_pre_m_s=np.asarray([[0.3, 0.4]]),
        velocity_post_m_s=np.asarray([[0.3, 0.4]]),
        charge_number_pre=np.asarray([-2.0]),
        charge_number_post=np.asarray([-2.0]),
        boundary_id=np.asarray([37], dtype=np.int32),
        material_id=np.asarray([0], dtype=np.int32),
        law_id=np.asarray(["hold"]),
        outcome=np.asarray(["held"]),
        localization_residual_m=np.asarray([1.0e-15]),
        position_budget_m=np.asarray([1.0e-12]),
        time_budget_s=np.asarray([1.0e-12]),
    )
    failures = SimpleNamespace(
        particle_id=np.asarray([2], dtype=np.int64),
        event_ordinal=np.asarray([3], dtype=np.uint32),
        time_s=np.asarray([0.02]),
        reason_code=np.asarray([9], dtype=np.uint16),
    )
    result = SimpleNamespace(
        iter_frames=lambda: iter((frame,)),
        read_boundary_events=lambda: events,
        read_failure_events=lambda: failures,
    )

    assert matrix._write_trajectory(tmp_path / "trajectory.csv", result) == 1
    assert matrix._write_events(tmp_path / "events.csv", result) == 1
    assert matrix._write_failures(tmp_path / "failures.csv", result) == 1
    assert (
        (tmp_path / "trajectory.csv")
        .read_text(encoding="utf-8")
        .rstrip()
        .endswith(
            "1,0.029999999999999999,0.10000000000000001,0.20000000000000001,"
            "0.29999999999999999,0.40000000000000002,-2,held"
        )
    )
    assert ",37,0,hold,held," in (tmp_path / "events.csv").read_text(encoding="utf-8")
    assert (tmp_path / "failures.csv").read_text(encoding="utf-8").rstrip().endswith("2,3,0.02,9")


def test_trajectory_projection_omits_particle_after_escape(tmp_path: Path) -> None:
    frames = (
        SimpleNamespace(
            time_s=0.0,
            particle_id=np.asarray([1, 2], dtype=np.int64),
            position_m=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            velocity_m_s=np.asarray([[0.0, 0.0], [0.0, 0.0]]),
            charge_number=np.asarray([-1.0, -2.0]),
            lifecycle=np.asarray([1, 1], dtype=np.uint8),
        ),
        SimpleNamespace(
            time_s=0.01,
            particle_id=np.asarray([1], dtype=np.int64),
            position_m=np.asarray([[0.11, 0.21]]),
            velocity_m_s=np.asarray([[0.01, 0.01]]),
            charge_number=np.asarray([-1.1]),
            lifecycle=np.asarray([1], dtype=np.uint8),
        ),
    )
    result = SimpleNamespace(iter_frames=lambda: iter(frames))
    trajectory = tmp_path / "trajectory.csv"

    assert matrix._write_trajectory(trajectory, result) == 3
    with trajectory.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["particle_id"], row["time_s"]) for row in rows] == [
        ("1", "0"),
        ("2", "0"),
        ("1", "0.01"),
    ]


def _prepared_matrix(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config_path = prepared / matrix.CONFIG_COPY
    config_path.write_bytes(CONFIG.read_bytes())
    workflows: dict[str, object] = {}
    for workflow_name in matrix.WORKFLOWS:
        root = prepared / workflow_name
        root.mkdir()
        cases: dict[str, object] = {}
        for label, dt_s in matrix._step_rows(workflow_name):
            case = root / f"candidate_{label}.yaml"
            case.write_text("format_version: 3\n", encoding="utf-8")
            cases[label] = {"path": case.name, "dt_s": dt_s}
        workflows[workflow_name] = {
            "input_content_hash": f"sha256:{workflow_name}",
            "input_sha256": f"input-{workflow_name}",
            "neutral_transport": {"maximum_speed_ratio": 0.1},
            "charge_step_receipt": matrix._charge_step_receipt(workflow_name),
            "cases": cases,
        }
    report = {
        "status": "PREPARED",
        "tool_revision": matrix.TOOL_REVISION,
        "configuration_sha256": matrix._sha256(config_path),
        "workflows": workflows,
    }
    (prepared / matrix.PREPARE_REPORT).write_text(json.dumps(report), encoding="utf-8")
    return prepared


def test_run_cell_is_no_clobber_and_records_public_api_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_matrix(tmp_path)
    manifest = {
        "status": "complete",
        "engine_algorithm_revision": "particle_engine_v34",
        "failure_reason_counts": {},
        "counts": {
            "particles": matrix.PARTICLE_COUNT,
            "frames": matrix.OUTPUT_COUNT,
            "frame_rows": matrix.PARTICLE_COUNT * matrix.OUTPUT_COUNT,
            "release_events": matrix.PARTICLE_COUNT,
            "boundary_events": 12,
            "failure_events": 0,
        },
        "resolved": {"physics_models": {"charge": {"revision": "charge-v1"}}},
    }
    fake_result = SimpleNamespace(manifest=MappingProxyType(manifest))
    monkeypatch.setattr(matrix, "load_case", lambda _path: SimpleNamespace())

    def simulate(_case: object, output: Path) -> None:
        output.mkdir()
        (output / "run.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(matrix, "simulate", simulate)
    monkeypatch.setattr(matrix, "open_result", lambda _path: fake_result)

    def write(path: Path, _result: object, rows: int) -> int:
        path.write_text("header\n", encoding="utf-8")
        return rows

    monkeypatch.setattr(
        matrix,
        "_write_trajectory",
        lambda path, result: write(path, result, matrix.PARTICLE_COUNT * matrix.OUTPUT_COUNT),
    )
    monkeypatch.setattr(matrix, "_write_events", lambda path, result: write(path, result, 12))
    monkeypatch.setattr(matrix, "_write_failures", lambda path, result: write(path, result, 0))

    receipt = matrix.run_cell(prepared, "caseA", "coarse")

    assert receipt["status"] == "COMPLETE"
    assert receipt["public_api_path"] == ["load_case", "simulate", "open_result"]
    assert receipt["trajectory_rows"] == matrix.PARTICLE_COUNT * matrix.OUTPUT_COUNT
    assert receipt["event_rows"] == 12
    assert receipt["failure_rows"] == 0
    assert receipt["algorithm_revisions"] == {"engine_algorithm_revision": "particle_engine_v34"}
    with pytest.raises(FileExistsError, match="already exists"):
        matrix.run_cell(prepared, "caseA", "coarse")


def test_recover_cell_manifest_reuses_completed_artifacts_without_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared_matrix(tmp_path)
    paths = matrix._cell_paths(prepared / "caseA", "coarse")
    sparse_trajectory_rows = matrix.PARTICLE_COUNT * matrix.OUTPUT_COUNT - 1
    manifest = {
        "status": "complete",
        "engine_algorithm_revision": "particle_engine_v34",
        "failure_reason_counts": {},
        "counts": {
            "particles": matrix.PARTICLE_COUNT,
            "frames": matrix.OUTPUT_COUNT,
            "frame_rows": sparse_trajectory_rows,
            "release_events": matrix.PARTICLE_COUNT,
            "boundary_events": 2,
            "failure_events": 0,
        },
        "resolved": {"physics_models": {"charge": {"revision": "charge-v1"}}},
    }
    paths["result"].mkdir()
    (paths["result"] / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_projection(
        paths["trajectory"],
        matrix._TRAJECTORY_HEADER,
        sparse_trajectory_rows,
    )
    _write_projection(paths["events"], matrix._EVENT_HEADER, 2)
    _write_projection(paths["failures"], matrix._FAILURE_HEADER, 0)
    reused = (paths["result"] / "run.json", paths["trajectory"], paths["events"], paths["failures"])
    hashes_before = {path: matrix._sha256(path) for path in reused}
    fake_result = SimpleNamespace(manifest=MappingProxyType(manifest))
    monkeypatch.setattr(matrix, "open_result", lambda _path: fake_result)

    def reject_simulation(_case: object, _output: Path) -> None:
        raise AssertionError("manifest recovery must not run the solver")

    monkeypatch.setattr(matrix, "simulate", reject_simulation)

    receipt = matrix.recover_cell_manifest(prepared, "caseA", "coarse")

    assert receipt["status"] == "COMPLETE"
    assert receipt["trajectory_rows"] == sparse_trajectory_rows
    assert _mapping(receipt["result_counts"])["frame_rows"] == sparse_trajectory_rows
    recovery = _mapping(receipt["manifest_recovery"])
    assert recovery == {
        "revision": matrix.MANIFEST_RECOVERY_REVISION,
        "reason": "post_run_result_manifest_read_only_mapping_rejected",
        "solver_reexecuted": False,
        "projections_rewritten": False,
        "reused_artifacts": [
            "result_coarse",
            "candidate_trajectory_coarse.csv",
            "candidate_events_coarse.csv",
            "candidate_failures_coarse.csv",
        ],
        "recovery_source": "prepare_m3c1_theory_100nm_matrix.py",
        "recovery_source_sha256": matrix._sha256(Path(matrix.__file__).resolve()),
    }
    assert {path: matrix._sha256(path) for path in reused} == hashes_before
    assert paths["manifest"].is_file()
    with pytest.raises(FileExistsError, match="manifest already exists"):
        matrix.recover_cell_manifest(prepared, "caseA", "coarse")


def test_projection_row_counter_rejects_malformed_csv(tmp_path: Path) -> None:
    projection = tmp_path / "projection.csv"
    projection.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header differs"):
        matrix._count_projection_rows(projection, ("first", "second"))

    projection.write_text("first,second\nonly-one-column\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row width differs"):
        matrix._count_projection_rows(projection, ("first", "second"))


def test_finalize_collects_the_six_hashed_cell_receipts(tmp_path: Path) -> None:
    prepared = _prepared_matrix(tmp_path)
    configuration_sha256 = matrix._sha256(prepared / matrix.CONFIG_COPY)
    for workflow_name in matrix.WORKFLOWS:
        for label, _ in matrix._step_rows(workflow_name):
            receipt = {
                "status": "COMPLETE",
                "workflow": workflow_name,
                "step_label": label,
                "configuration_sha256": configuration_sha256,
                "physics_receipt": {"brownian_active": False, "saffman_active": False},
            }
            path = prepared / workflow_name / f"candidate_run_manifest_{label}.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")

    report = matrix.finalize(prepared)

    assert report["status"] == "COMPLETE"
    assert report["configuration_sha256"] == configuration_sha256
    workflows = _mapping(report["workflows"])
    for workflow_name in matrix.WORKFLOWS:
        workflow = _mapping(workflows[workflow_name])
        assert workflow["status"] == "COMPLETE"
        runs = _mapping(workflow["runs"])
        assert set(runs) == set(matrix.RUN_KEYS)
        assert all(len(_mapping(run)["manifest_sha256"]) == 64 for run in runs.values())
    with pytest.raises(FileExistsError, match="final report already exists"):
        matrix.finalize(prepared)
