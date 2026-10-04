"""Focused checks for the external M3-C1 canonical P1 candidate preparer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tools.vv.comsol import prepare_m3c1_candidate as candidate

CONFIG = Path(__file__).parents[1] / "cases" / "m3c1_caseA_100nm_exported_p1_v1.json"


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def test_config_locks_the_three_step_full_deterministic_matrix() -> None:
    config = candidate._load_config(CONFIG)

    assert config["case"] == {
        "workflow": "caseA",
        "diameter_m": 1.0e-7,
        "particle_count": 287,
        "output_times": 46,
        "time_end_s": 4.5e-4,
        "output_interval_s": 1.0e-5,
        "fixed_rk4_steps_s": [6.25e-7, 3.125e-7, 1.5625e-7],
    }
    assert config["physics"]["brownian_active"] is False
    assert config["physics"]["saffman_active"] is False
    assert config["scope"]["claims_solver_agreement"] is False
    assert config["scope"]["claims_physical_applicability"] is False
    assert (
        config["scope"]["heat_flux_authority"] == "derived_export_not_ppr_nojac_primitive_authority"
    )


def test_strict_and_diagnostic_cases_keep_authorities_separate() -> None:
    config = candidate._load_config(CONFIG)
    strict = candidate._case_document("input.h5", "sha256:test", 6.25e-7, config)
    diagnostic = candidate._case_document(
        "input.h5",
        "sha256:test",
        6.25e-7,
        config,
        dep_radius_limit_m=5.0000000000000004e-8,
        ion_speed_limit_m_s=30_000.0,
    )

    physics = _mapping(strict["physics"])
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
    charge = _mapping(physics["charge"])
    ion_drag = _mapping(physics["ion_drag"])
    dep = _mapping(physics["dielectrophoresis"])
    diagnostic_physics = _mapping(diagnostic["physics"])
    diagnostic_charge = _mapping(diagnostic_physics["charge"])
    diagnostic_ion_drag = _mapping(diagnostic_physics["ion_drag"])
    diagnostic_dep = _mapping(diagnostic_physics["dielectrophoresis"])
    assert charge["maximum_relative_ion_speed_m_s"] == 25_000.0
    assert ion_drag["maximum_relative_ion_speed_m_s"] == 25_000.0
    assert dep["maximum_point_dipole_radius_m"] == 5.0e-8
    assert diagnostic_charge["maximum_relative_ion_speed_m_s"] == 30_000.0
    assert diagnostic_ion_drag["maximum_relative_ion_speed_m_s"] == 30_000.0
    assert diagnostic_dep["maximum_point_dipole_radius_m"] == 5.0000000000000004e-8
    output = _mapping(diagnostic["output"])
    trajectories = _mapping(output["trajectories"])
    schedule = _mapping(trajectories["schedule"])
    times = schedule["explicit_times_s"]
    assert isinstance(times, list)
    assert len(times) == 46
    assert times[0] == 0.0
    assert times[-1] == 4.5e-4


def test_dep_execution_guard_accepts_only_one_float64_successor() -> None:
    limit_m = 5.0e-8
    immediate_successor_m = math.nextafter(limit_m, math.inf)
    second_successor_m = math.nextafter(immediate_successor_m, math.inf)

    rounded = candidate._dep_point_dipole_receipt(limit_m, immediate_successor_m)
    outside = candidate._dep_point_dipole_receipt(limit_m, second_successor_m)

    assert rounded["status"] == "NOT_TESTED_POINT_DIPOLE_CERTIFICATION_MISSING"
    assert rounded["benchmark_sensitivity_limit_m"] == limit_m
    assert rounded["maximum_electrostatic_radius_m"] == immediate_successor_m
    rounded_excess_m = rounded["excess_m"]
    assert isinstance(rounded_excess_m, float)
    assert rounded_excess_m > 0.0
    assert rounded["execution_guard_rounding_policy"] == (
        "accept_immediate_float64_successor_for_independent_serialization"
    )
    assert rounded["execution_guard_upper_m"] == immediate_successor_m
    assert rounded["execution_guard_excess_m"] == 0.0
    assert rounded["execution_guard_status"] == "within_limit"

    assert outside["status"] == "NOT_TESTED_POINT_DIPOLE_CERTIFICATION_MISSING"
    assert outside["execution_guard_upper_m"] == immediate_successor_m
    outside_excess_m = outside["execution_guard_excess_m"]
    assert isinstance(outside_excess_m, float)
    assert outside_excess_m > 0.0
    assert outside["execution_guard_status"] == "BLOCKED"


def test_coordinate_matching_is_unique_and_outcome_is_fail_closed() -> None:
    canonical = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    provider = np.asarray([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

    matched, maximum_distance = candidate._coordinate_match(canonical, provider, 1.0e-14)

    np.testing.assert_array_equal(matched, [1, 2, 0])
    assert maximum_distance == 0.0
    assert candidate._run_outcome(
        {
            "coarse": {"status": "complete"},
            "medium": {"status": "blocked"},
            "fine": {"status": "complete"},
        }
    ) == ("blocked", "BLOCKED")
    assert candidate._run_outcome(
        {
            "coarse": {"status": "complete"},
            "medium": {"status": "complete"},
            "fine": {"status": "complete"},
        }
    ) == ("complete", "COMPLETE")


def test_execute_case_emits_complete_evaluator_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    label = "dt_0p625us"
    case_path = tmp_path / f"candidate_{label}.yaml"
    case_path.write_text("case: synthetic\n", encoding="utf-8")
    result_path = tmp_path / f"result_{label}"
    result_path.mkdir()
    manifest = {
        "status": "complete",
        "case_file_hash": "sha256:synthetic-case",
        "engine_algorithm_revision": "engine_v31",
        "result_algorithm_revision": "result_v3",
        "rk4_enclosure_revision": "rk4_local_certificate_v1",
        "resolved": {
            "physics_models": {
                "lift": {
                    "model": "rarefied_vorticity_sensitivity",
                    "revision": "lift_v1",
                }
            }
        },
        "failure_reason_counts": {},
    }
    manifest_path = result_path / "run.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fake_case = SimpleNamespace(case_file_hash="sha256:synthetic-case")
    fake_result = SimpleNamespace(
        manifest=manifest,
        read_failure_events=lambda: SimpleNamespace(
            particle_id=np.empty(0, dtype=np.int64),
            time_s=np.empty(0, dtype=np.float64),
        ),
    )

    monkeypatch.setattr(candidate, "load_case", lambda _path: fake_case)
    monkeypatch.setattr(candidate, "open_result", lambda _path: fake_result)

    def write_trajectory(path: Path, _result: object) -> int:
        path.write_text("trajectory\n", encoding="utf-8")
        return candidate._EXPECTED_PARTICLES * candidate._EXPECTED_FRAMES

    def write_events(path: Path, _result: object) -> int:
        path.write_text("events\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(candidate, "_write_trajectory", write_trajectory)
    monkeypatch.setattr(candidate, "_write_events", write_events)

    receipt = candidate._execute_case(tmp_path, label, case_path.name)

    assert receipt["status"] == "complete"
    assert receipt["decision"] == "COMPLETE"
    assert receipt["case"] == case_path.name
    assert receipt["case_sha256"] == candidate._sha256(case_path)
    assert receipt["result"] == result_path.name
    assert receipt["result_manifest_sha256"] == candidate._sha256(manifest_path)
    assert receipt["trajectory_sha256"] == candidate._sha256(
        tmp_path / f"candidate_trajectory_{label}.csv"
    )
    assert receipt["engine_algorithm_revision"] == "engine_v31"
    assert receipt["revisions"] == {
        "engine_algorithm_revision": "engine_v31",
        "result_algorithm_revision": "result_v3",
        "rk4_enclosure_revision": "rk4_local_certificate_v1",
    }
    assert receipt["physics_models"] == manifest["resolved"]["physics_models"]


def test_run_uses_strict_cases_and_publishes_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = {label: f"candidate_{label}.yaml" for label, _step_s in candidate._STEP_ROWS}
    prepare_report = {
        "tool_revision": candidate.TOOL_REVISION,
        "cases": cases,
        "diagnostic_cases": {
            label: f"candidate_diagnostic_{label}.yaml" for label, _step_s in candidate._STEP_ROWS
        },
    }
    report_path = tmp_path / "prepare_report.json"
    report_path.write_text(json.dumps(prepare_report), encoding="utf-8")
    observed_cases: list[str] = []

    def execute_case(_root: Path, label: str, case_name: str) -> dict[str, object]:
        observed_cases.append(case_name)
        return {
            "status": "complete",
            "decision": "COMPLETE",
            "case": case_name,
            "trajectory_rows": candidate._EXPECTED_PARTICLES * candidate._EXPECTED_FRAMES,
        }

    monkeypatch.setattr(candidate, "_execute_case", execute_case)

    report = candidate.run(tmp_path)

    assert report["status"] == "complete"
    assert report["decision"] == "COMPLETE"
    assert report["prepare_report"] == report_path.name
    assert report["prepare_report_sha256"] == candidate._sha256(report_path)
    assert set(_mapping(report["runs"])) == set(cases)
    assert "diagnostic_runs" not in report
    assert observed_cases == [cases[label] for label, _step_s in candidate._STEP_ROWS]
