from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from tests.verification.microcases import materialize_microcase
from tools.vv.comsol.evaluate_saved_population import evaluate

from chamber_particles import load_case, simulate
from chamber_particles.case_format import RealizedSurfaceSource, RealizedTableSource, write

_COLUMNS = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "vr_m_s",
    "vz_m_s",
    "charge_number",
    "native_status",
    "release_time_s",
    "stop_time_s",
    "native_final_status",
)


def _write(path: Path, rows: list[list[float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(_COLUMNS)
        writer.writerows(rows)


def _mixed_rows() -> list[list[float | int]]:
    rows = []
    for time in (0, 0.1, 0.4):
        for pid, stop, final in ((7, 0.05, 3), (101, 0.2, 4), (999, np.nan, 1)):
            status = final if time >= stop else 1
            state = [np.nan] * 5 if status == 4 else [0.2, 0.1, 3.0, 4.0, -2.0]
            rows.append([pid, time, *state, status, 0.0, stop, final])
    return rows[::-1]  # Identity and clock never depend on CSV row order.


def test_saved_status_and_nonuniform_clock_preserve_unknown_disappearance(tmp_path: Path) -> None:
    path = tmp_path / "native.csv"
    _write(path, _mixed_rows())
    report = evaluate(path, reference_terminal_times=True)
    native = report["reference"]
    assert native["final_counts"] == {
        "pending": 0,
        "active": 1,
        "frozen": 0,
        "stuck": 1,
        "disappeared_unknown": 1,
    }
    assert native["coverage"] == {
        "expected_rows": 9,
        "observed_rows": 9,
        "missing_rows": 0,
        "lifecycle_known_rows": 9,
        "lifecycle_unknown_rows": 0,
        "nonfinite_state_rows": 1,
        "nonfinite_observed_state_rows": 1,
        "nonfinite_active_state_rows": 0,
    }
    assert native["series"][0]["status_counts"]["active"] == 3
    assert native["series"][1]["active_speed_m_s"]["p50"] == 5
    assert native["saved_active_fraction_integral_s"] == pytest.approx(7 / 30)
    assert native["restricted_mean_active_lifetime_s"] == pytest.approx((0.05 + 0.2 + 0.4) / 3)
    assert native["capture_terminal_time_s"]["p50"] == 0.05
    assert report["qualified_solver_parity"] == "NOT_TESTED"

    assert "not equated" in report["interpretation"]["native_disappeared"]


def test_unqualified_stop_time_keeps_observation_brackets(tmp_path: Path) -> None:
    path = tmp_path / "native.csv"
    _write(path, _mixed_rows())
    native = evaluate(path)["reference"]
    assert native["capture_terminal_time_s"]["finite_count"] == 0
    assert native["capture_terminal_time_s"]["p50"] is None
    assert native["restricted_mean_active_lifetime_s"] is None
    assert native["capture_terminal_observation_brackets_s"]["lower"]["p50"] == 0
    assert native["capture_terminal_observation_brackets_s"]["upper"]["p50"] == 0.1


@pytest.mark.parametrize("defect", ["duplicate", "status", "stop", "terminal_return", "metadata"])
def test_population_rejects_ambiguous_identity_and_terminal_evidence(
    tmp_path: Path,
    defect: str,
) -> None:
    rows = _mixed_rows()
    if defect == "duplicate":
        rows.append(rows[-1])
    elif defect == "status":
        rows[0][7] = 9
    elif defect == "stop":
        for row in rows:
            if row[0] == 7:
                row[9] = 0.2
    elif defect == "terminal_return":
        rows.extend([[7, 0.15, 0.2, 0.1, 3, 4, -2, 1, 0, 0.05, 3]])
    else:
        rows[0][10] = 3
    path = tmp_path / "bad.csv"
    _write(path, rows)
    with pytest.raises(ValueError):
        evaluate(path, reference_terminal_times=True)


def _rz_result(
    tmp_path: Path, *, complete_clock: bool = True, law: str = "stick", immediate: bool = False
) -> Path:
    paths = materialize_microcase("C08" if immediate else "C07", tmp_path / "case")
    case = load_case(paths.case_path)
    boundary = case.data.geometry.boundary
    boundary = replace(
        boundary,
        line2=boundary.line2[:3],
        boundary_id=boundary.boundary_id[:3],
        group_id=np.zeros(3, dtype="<i4"),
        material_id=boundary.material_id[:3],
        owner_cell_type=boundary.owner_cell_type[:3],
        owner_cell_local_index=boundary.owner_cell_local_index[:3],
        orientation=boundary.orientation[:3],
    )
    sources = case.data.sources
    if immediate:
        source = sources[0]
        assert isinstance(source, RealizedSurfaceSource)
        sources = (
            replace(
                source, facet_id=np.asarray([1], dtype="<i8"), velocity_m_s=np.asarray([[1.0, 0.0]])
            ),
        )
    info = write(
        paths.case_path.parent / "rz.h5",
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=replace(case.data.geometry, boundary=boundary, group_names=("wall",)),
            sources=sources,
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path="rz.h5", expected_content_hash=info.content_hash)
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    document["time"] = {"start_s": 0, "end_s": 2, "dt_s": 2}
    document["boundaries"] = [{"boundary_group": "wall", "priority": 10, "law": law}]
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {
            "explicit_times_s": [0, 1, 1.5, 2] if complete_clock else [0, 1],
        },
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "result"
    simulate(load_case(paths.case_path), output)
    return output


def _analytic_stick_native(path: Path) -> None:
    rows = []
    for time in (0, 1, 1.5, 2):
        moving = time < 1.5
        rows.append(
            [
                701,
                time,
                0.25 + 0.5 * min(time, 1.5),
                0.25 + 0.25 * min(time, 1.5),
                0.5 if moving else 0,
                0.25 if moving else 0,
                3,
                1 if moving else 3,
                0,
                1.5,
                3,
            ]
        )
    _write(path, rows)


def test_completed_public_population_and_analytic_terminal_capture(tmp_path: Path) -> None:
    result = _rz_result(tmp_path)
    path = tmp_path / "native.csv"
    _analytic_stick_native(path)
    report = evaluate(path, reference_terminal_times=True, candidate=result)
    comparison = report["comparison"]
    assert comparison["common_exact_observation_times_s"] == [0, 1, 1.5, 2]
    assert comparison["population_curves"]["stuck"]["maximum_abs_fraction_difference"] == 0
    for quantity in ("position_m", "velocity_m_s", "charge_number"):
        assert (
            comparison["same_ID_finite_active_differences_descriptive_only"][quantity]["maximum"]
            == 0
        )
    assert report["candidate"]["capture_terminal_time_s"]["p50"] == 1.5
    assert report["reference"]["capture_terminal_time_s"]["p50"] == 1.5
    assert report["candidate"]["saved_active_fraction_integral_s"] == 1.25
    assert report["candidate"]["restricted_mean_active_lifetime_s"] == 1.5
    assert report["qualified_solver_parity"] == "NOT_TESTED"

    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.vv.comsol.evaluate_saved_population",
            "--reference",
            str(path),
            "--reference-terminal-times",
            "--candidate",
            str(result),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_candidate_must_cover_the_full_executed_window(tmp_path: Path) -> None:
    result = _rz_result(tmp_path, complete_clock=False)
    path = tmp_path / "native.csv"
    _analytic_stick_native(path)
    with pytest.raises(ValueError, match="complete executed time window"):
        evaluate(path, candidate=result)


@pytest.mark.parametrize(
    ("immediate", "expected_observed", "expected_integral"),
    [(False, 2, 1.25), (True, 0, 0.0)],
)
def test_escaped_suffix_lifecycle_comes_from_event_time_without_invented_states(
    tmp_path: Path, immediate: bool, expected_observed: int, expected_integral: float
) -> None:
    result = _rz_result(tmp_path, law="escape", immediate=immediate)
    stop, pid = (0.0, 801) if immediate else (1.5, 701)
    rows = []
    for time in (0, 1, 1.5, 2):
        state = (
            [0.25 + 0.5 * time, 0.25 + 0.25 * time, 0.5, 0.25, 3] if time < stop else [np.nan] * 5
        )
        rows.append([pid, time, *state, 1 if time < stop else 4, 0, stop, 4])
    path = tmp_path / "native.csv"
    _write(path, rows)
    report = evaluate(path, reference_terminal_times=True, candidate=result)
    candidate = report["candidate"]
    assert (
        candidate["final_counts"]["escaped"],
        report["reference"]["final_counts"]["disappeared_unknown"],
    ) == (1, 1)
    assert candidate["series"][0]["status_counts"]["active"] == int(not immediate)
    assert (
        candidate["coverage"]["lifecycle_known_rows"],
        candidate["coverage"]["nonfinite_state_rows"],
        candidate["coverage"]["observed_rows"],
    ) == (4, 4 - expected_observed, expected_observed)
    assert candidate["restricted_mean_active_lifetime_s"] == stop
    assert candidate["saved_active_fraction_integral_s"] == expected_integral


def test_missing_state_before_recorded_terminal_is_rejected(tmp_path: Path) -> None:
    result = _rz_result(tmp_path, law="escape")
    # A durable but inconsistent external result: move its event after an
    # already omitted frame. Final status alone must not explain that gap.
    for segment in (result / "segments").glob("*.h5"):
        with h5py.File(segment, "r+") as handle:
            dataset = handle["events/boundary/time_s"]
            dataset[...] = dataset[...] + 0.1
    path = tmp_path / "native.csv"
    _analytic_stick_native(path)
    with pytest.raises(ValueError, match=r"missing active/pre-terminal.*701.*1\.5"):
        evaluate(path, candidate=result)


def test_failed_suffix_is_known_from_failure_ledger_and_excluded_from_lifetime(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C09", tmp_path / "case")
    case = load_case(paths.case_path)
    nodes = case.data.geometry.nodes_m.copy()
    nodes[:, 0] += 1.0
    source = case.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    position = source.position_m.copy()
    position[:, 0] += 1.0
    source = replace(source, position_m=position)
    info = write(
        paths.case_path.parent / "rz.h5",
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            sources=(source,),
            geometry=replace(case.data.geometry, nodes_m=nodes),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path="rz.h5", expected_content_hash=info.content_hash)
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    document["solver"]["event"].update(max_interactions_per_step=1, max_refinements=1)
    end = document["time"]["end_s"]
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0, end]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "result"
    simulate(load_case(paths.case_path), output)
    reference = tmp_path / "native.csv"
    _write(
        reference,
        [
            [901, time, position[0, 0] + time, position[0, 1], 1, 0, 0, 1, 0, np.nan, 1]
            for time in (0, end)
        ],
    )
    candidate = evaluate(reference, candidate=output)["candidate"]
    assert candidate["final_counts"]["failed"] == 1
    assert candidate["final_counts"]["escaped"] == 0
    assert candidate["coverage"]["lifecycle_unknown_rows"] == 0
    assert candidate["restricted_mean_active_lifetime_s"] is None
