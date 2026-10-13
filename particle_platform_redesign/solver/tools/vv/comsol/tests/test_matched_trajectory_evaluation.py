from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from tests.verification.microcases import materialize_microcase
from tools.vv.comsol.evaluate_matched_trajectory import (
    characterize,
    compare,
    compare_pair,
    register_budget,
)

from chamber_particles import load_case, simulate
from chamber_particles.case_format import write

_IDS = [3099, 7, 101]
_TIMES = [3.0, 3.003, 3.014, 3.031]
_STEPS = [0.004, 0.002, 0.001]
_COLUMNS = ["id", "t", "p0", "p1", "v0", "v1", "Z", "state", "support"]
_MAPPING = {
    "particle_id": "id",
    "time": "t",
    "position": ["p0", "p1"],
    "velocity": ["v0", "v1"],
    "charge": "Z",
    "lifecycle": "state",
    "support_inside": "support",
}


def _dump(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _write_analytic(path: Path, step: float, coefficient: float) -> None:
    """Analytic Z'=2, v0'=Z, p'=v with a known h^2 synthetic export error."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(_COLUMNS)
        for time in reversed(_TIMES):
            elapsed = time - _TIMES[0]
            error = coefficient * step**2 * elapsed
            for pid in _IDS:
                charge = -2.0 + pid * 0.001
                v0 = 0.2 + charge * elapsed + elapsed**2
                p0 = 0.4 + 0.2 * elapsed + 0.5 * charge * elapsed**2 + elapsed**3 / 3
                writer.writerow(
                    [
                        pid,
                        time,
                        p0 + error,
                        0.3 + 0.1 * elapsed - error / 2,
                        v0 + 2 * error,
                        0.1 - error,
                        charge + 2 * elapsed + 3 * error,
                        "active",
                        "true",
                    ]
                )


def _series(tmp_path: Path, label: str, coefficient: float, mode: str = "cartesian_xy") -> Path:
    levels = {}
    for name, step in zip(("coarse", "medium", "fine"), _STEPS, strict=True):
        csv_path = tmp_path / f"{label}_{name}.csv"
        _write_analytic(csv_path, step, coefficient)
        metadata_path = _dump(
            tmp_path / f"{label}_{name}_metadata.json",
            {"time": {"dt_s": step}, "statement": "synthetic analytic export, not a solver run"},
        )
        levels[name] = {
            "step_s": step,
            "csv_path": csv_path.name,
            "columns": _MAPPING,
            "metadata": {
                "path": metadata_path.name,
                "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "integration_step_pointer": "/time/dt_s",
            },
        }
    return _dump(
        tmp_path / f"{label}_series.json",
        {
            "label": label,
            "coordinate_mode": mode,
            "particle_ids": _IDS,
            "observation_times_s": _TIMES,
            "levels": levels,
        },
    )


def _registered(
    tmp_path: Path, candidate: Path, reference: Path, self_budget: float | None = None
) -> Path:
    candidate_report = _dump(tmp_path / "candidate_report.json", characterize(candidate))
    reference_report = _dump(tmp_path / "reference_report.json", characterize(reference))
    value: dict[str, Any] = {
        "physical_budgets": {
            name: {"rms": 1e-4, "maximum": 1e-4} for name in ("position", "velocity", "charge")
        }
    }
    if self_budget is not None:
        value["self_sensitivity_budgets"] = {
            side: {
                name: {"rms": self_budget, "maximum": self_budget}
                for name in ("position", "velocity", "charge")
            }
            for side in ("candidate", "reference")
        }
    policy = _dump(
        tmp_path / "policy.json",
        value,
    )
    return _dump(
        tmp_path / "budget.json", register_budget(candidate_report, reference_report, policy)
    )


@pytest.mark.parametrize("mode", ["cartesian_xy", "axisymmetric_rz_meridional"])
def test_arbitrary_ids_nonuniform_absolute_clock_and_coupled_charge(
    tmp_path: Path, mode: str
) -> None:
    candidate = _series(tmp_path, "arbitrary candidate", 1.0, mode)
    reference = _series(tmp_path, "independent analytic export", 2.0, mode)
    report = characterize(candidate)
    assert report["scope"]["particle_ids"] == [7, 101, 3099]
    assert report["scope"]["observation_times_s"] == _TIMES
    assert report["observed_order"]["step_ratios"] == [2.0, 2.0]
    assert report["initial_state"]["exact_x_v_Z_id_time_equality"]
    delta = (_STEPS[1] ** 2 - _STEPS[2] ** 2) * (_TIMES[-1] - _TIMES[0])
    assert report["medium_fine"]["position"]["maximum"] == pytest.approx(
        math.sqrt(1.25) * delta, abs=1e-15
    )
    assert report["medium_fine"]["velocity"]["maximum"] == pytest.approx(
        math.sqrt(5) * delta, abs=1e-15
    )
    assert report["medium_fine"]["charge"]["maximum"] == pytest.approx(3 * delta, abs=1e-14)
    assert report["observed_order"]["charge"]["rms"] == pytest.approx(2.0, abs=1e-7)
    assert report["medium_fine"]["position"]["per_particle"][0]["particle_id"] == 7
    assert report["artifacts"]["fine"]["execution"] == "BOUND_METADATA_NOT_EXECUTION_PROOF"
    budget = _registered(tmp_path, candidate, reference)
    comparison = compare(budget)
    assert comparison["descriptive_physical_budget_verdict"] == "PASS"
    assert comparison["qualified_solver_parity"] == "NOT_TESTED"
    assert comparison["accuracy_claim"]["finest_pair_is_error_bound"] is False
    assert comparison["trajectory_difference"]["denominator"] == 12
    csv_path = tmp_path / "arbitrary candidate_fine.csv"
    with csv_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source artifacts differ"):
        compare(budget)


def _edit_rows(path: Path, edit: str) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if edit == "missing":
        rows.pop(0)
    elif edit == "initial_charge":
        rows[-1]["Z"] = "9"
    elif edit == "unsupported":
        rows[0]["support"] = "false"
    elif edit == "terminal":
        rows[0]["state"] = "stuck"
    elif edit == "duplicate":
        rows.append(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, _COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("edit", ["missing", "initial_charge", "unsupported", "terminal"])
def test_coverage_and_initial_mismatch_preserve_raw_measurements(tmp_path: Path, edit: str) -> None:
    series = _series(tmp_path, "candidate", 1.0)
    _edit_rows(tmp_path / "candidate_fine.csv", edit)
    report = characterize(series)
    assert report["scope"]["deterministic_pre_event"] == "NOT_TESTED"
    assert report["medium_fine"]["denominator"] == 12
    if edit in {"missing", "terminal"}:
        assert report["medium_fine"]["paired_usable"] == 11
    elif edit == "initial_charge":
        assert not report["initial_state"]["exact_x_v_Z_id_time_equality"]
    else:
        assert report["coverage"]["fine"]["support_outside_observed"] == 1


def test_duplicate_and_third_dof_are_not_silently_reduced(tmp_path: Path) -> None:
    series = _series(tmp_path, "candidate", 1.0)
    _edit_rows(tmp_path / "candidate_fine.csv", "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        characterize(series)
    config = json.loads(series.read_text(encoding="utf-8"))
    config["coordinate_mode"] = "cartesian_xyz"
    _dump(series, config)
    with pytest.raises(ValueError, match="2-DOF"):
        characterize(series)


def test_step_binding_and_unequal_ratios_are_explicit(tmp_path: Path) -> None:
    series = _series(tmp_path, "candidate", 1.0)
    config = json.loads(series.read_text(encoding="utf-8"))
    config["levels"]["coarse"]["step_s"] = 0.005
    _dump(series, config)
    with pytest.raises(ValueError, match="bound metadata"):
        characterize(series)
    config["levels"]["coarse"].pop("metadata")
    _dump(series, config)
    report = characterize(series)
    assert report["observed_order"]["step_ratios"] == [2.5, 2.0]
    assert report["observed_order"]["position"]["rms"] is None


def test_registered_physical_allowance_cannot_be_retuned_in_budget(tmp_path: Path) -> None:
    candidate = _series(tmp_path, "candidate", 1.0)
    reference = _series(tmp_path, "reference", 2.0)
    budget_path = _registered(tmp_path, candidate, reference)
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    budget["physical_budgets"]["charge"]["maximum"] = 100.0
    _dump(budget_path, budget)
    with pytest.raises(ValueError, match="registered physical policy"):
        compare(budget_path)


def test_identical_finest_paths_do_not_hide_unresolved_self_sensitivity(tmp_path: Path) -> None:
    candidate = _series(tmp_path, "candidate", 1.0)
    reference = _series(tmp_path, "reference", 1.0)
    budget = _registered(tmp_path, candidate, reference, self_budget=1e-8)
    report = compare(budget)
    assert report["trajectory_difference"]["position"]["maximum"] == 0.0
    assert report["descriptive_physical_budget_verdict"] == "PASS"
    resolution = report["empirical_self_resolution"]
    assert resolution["participants"]["candidate"]["empirical_target_verdict"] == "FAIL"
    assert not resolution["participants"]["reference"]["gates"]["charge"]["maximum"]["pass"]
    assert report["qualified_solver_parity"] == "NOT_TESTED"


def test_saved_pair_uses_two_actual_levels_and_keeps_unknown_parity(tmp_path: Path) -> None:
    descriptors = []
    for name, step, coefficient in (("candidate", 0.001, 1.0), ("original", 0.01, 2.0)):
        csv_path = tmp_path / f"{name}.csv"
        _write_analytic(csv_path, step, coefficient)
        descriptors.append(
            _dump(
                tmp_path / f"{name}.json",
                {
                    "label": name,
                    "coordinate_mode": "cartesian_xy",
                    "particle_ids": _IDS,
                    "observation_times_s": _TIMES,
                    "level": {"step_s": step, "csv_path": csv_path.name, "columns": _MAPPING},
                },
            )
        )
    policy = _dump(
        tmp_path / "policy.json",
        {
            "comparison_question": "original_saved_vs_analogous_model",
            "known_model_differences": ["native Brownian versus deterministic analogue"],
            "physical_budgets": {
                name: {"rms": 1.0, "maximum": 1.0} for name in ("position", "velocity", "charge")
            },
        },
    )
    report = compare_pair(descriptors[0], descriptors[1], policy)
    assert report["trajectory_difference"]["paired_usable"] == 12
    assert report["descriptive_physical_budget_verdict"] == "PASS"
    assert report["qualified_solver_parity"] == "NOT_TESTED"
    assert report["known_model_differences"] == ["native Brownian versus deterministic analogue"]
    assert report["scope"]["integration_steps_s"] == {"candidate": 0.001, "reference": 0.01}
    _edit_rows(tmp_path / "original.csv", "missing")
    partial = compare_pair(descriptors[0], descriptors[1], policy)
    assert partial["trajectory_difference"]["paired_usable"] == 11
    assert partial["descriptive_physical_budget_verdict"] == "NOT_TESTED"
    assert partial["coverage"]["reference"]["missing"] == 1


def test_public_completed_results_bind_configuration_and_release(tmp_path: Path) -> None:
    levels = {}
    for name, step in zip(("coarse", "medium", "fine"), [0.2, 0.1, 0.05], strict=True):
        paths = materialize_microcase("C01", tmp_path / name)
        document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
        document["time"]["dt_s"] = step
        paths.case_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        result_path = tmp_path / f"result_{name}"
        simulate(load_case(paths.case_path), result_path)
        levels[name] = {
            "step_s": step,
            "case_path": str(paths.case_path),
            "result_path": str(result_path),
        }
    series = _dump(
        tmp_path / "public_series.json",
        {
            "label": "independent ballistic",
            "coordinate_mode": "cartesian_xy",
            "particle_ids": [101],
            "observation_times_s": [0.0, 0.35, 0.8],
            "levels": levels,
        },
    )
    report = characterize(series)
    assert report["scope"]["deterministic_pre_event"] == "SUPPORTED"
    assert report["initial_state"]["exact_x_v_Z_id_time_equality"]
    assert report["artifacts"]["fine"]["execution"] == "COMPLETED_PUBLIC_RESULT_BOUND_TO_CASE"
    assert report["medium_fine"]["charge"]["maximum"] == 0.0
    budget = _registered(tmp_path, series, series, self_budget=1e-4)
    comparison = compare(budget)
    assert comparison["empirical_self_resolution"]["status"] == "PASS"
    assert comparison["empirical_self_resolution"]["participants"]["candidate"]["macro_resolution"][
        "accepted_macro_counts"
    ] == [4, 8, 16]
    config = json.loads(series.read_text(encoding="utf-8"))
    config["levels"]["fine"]["step_s"] = 0.04
    _dump(series, config)
    with pytest.raises(ValueError, match="actual case/run manifest"):
        characterize(series)


def test_distinct_step_labels_with_single_macro_are_not_resolution_evidence(tmp_path: Path) -> None:
    levels = {}
    for name, step in zip(("coarse", "medium", "fine"), [0.2, 0.1, 0.05], strict=True):
        paths = materialize_microcase("C01", tmp_path / name)
        document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
        document["time"].update(dt_s=step, end_s=step)
        document["output"]["trajectories"]["schedule"]["explicit_times_s"] = [0.0, 0.01]
        paths.case_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        result_path = tmp_path / f"result_{name}"
        simulate(load_case(paths.case_path), result_path)
        levels[name] = {
            "step_s": step,
            "case_path": str(paths.case_path),
            "result_path": str(result_path),
        }
    series = _dump(
        tmp_path / "one_macro.json",
        {
            "label": "exact ballistic prefixes",
            "coordinate_mode": "cartesian_xy",
            "particle_ids": [101],
            "observation_times_s": [0.0, 0.01],
            "levels": levels,
        },
    )
    budget = _registered(tmp_path, series, series, self_budget=1e-4)
    report = compare(budget)
    witness = report["empirical_self_resolution"]["participants"]["candidate"]["macro_resolution"]
    assert witness["accepted_macro_counts"] == [1, 1, 1]
    assert witness["status"] == "NOT_TESTED"
    assert report["empirical_self_resolution"]["status"] == "NOT_TESTED"
    assert report["qualified_solver_parity"] == "NOT_TESTED"


def test_global_refinement_does_not_certify_one_late_release_interval(tmp_path: Path) -> None:
    levels = {}
    for name, step in zip(("coarse", "medium", "fine"), [4.0, 2.0, 1.0], strict=True):
        paths = materialize_microcase("C01", tmp_path / name)
        original = load_case(paths.case_path)
        source = replace(original.data.sources[0], release_time_s=np.asarray([6.5]))
        data_path = paths.case_path.parent / "late_source.h5"
        info = write(data_path, replace(original.data, sources=(source,)))
        document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
        document["case"]["data_path"] = data_path.name
        document["case"]["expected_content_hash"] = info.content_hash
        document["time"].update(dt_s=step, end_s=8.0)
        document["output"]["trajectories"]["schedule"]["explicit_times_s"] = [6.5, 8.0]
        paths.case_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        result_path = tmp_path / f"result_{name}"
        simulate(load_case(paths.case_path), result_path)
        levels[name] = {
            "step_s": step,
            "case_path": str(paths.case_path),
            "result_path": str(result_path),
        }
    series = _dump(
        tmp_path / "late_release.json",
        {
            "label": "late release",
            "coordinate_mode": "cartesian_xy",
            "particle_ids": [101],
            "observation_times_s": [6.5, 8.0],
            "levels": levels,
        },
    )
    budget = _registered(tmp_path, series, series, self_budget=1e-4)
    report = compare(budget)
    witness = report["empirical_self_resolution"]["participants"]["candidate"]["macro_resolution"]
    assert witness["accepted_macro_counts"] == [2, 4, 8]
    assert witness["fine_base_steps_per_observation_window"] == 1.5
    assert witness["coarse_base_steps_per_observation_window"] == 0.375
    assert witness["status"] == "NOT_TESTED"
