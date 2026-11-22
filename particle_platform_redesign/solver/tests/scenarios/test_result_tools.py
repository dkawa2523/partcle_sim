"""T03 result tools stay outside the solver and consume only ResultView data."""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import h5py
import numpy as np
import pytest
import yaml
from tools.analysis import summarize_result
from tools.visualization import render_result

from chamber_particles import load_case, open_result, simulate
from tests.verification.microcases import materialize_microcase


def test_summary_and_svg_tools_read_one_completed_result(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "case")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0.0, 0.5, 1.0, 1.5, 2.0, 2.25]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    result_path = tmp_path / "result"
    simulate(load_case(paths.case_path), result_path)

    summary = summarize_result(result_path, event_batch_rows=1)

    assert summary["analysis_revision"] == "result_fate_deposition_arrival_v2"
    assert summary["parameters"] == {"event_batch_rows": 1}
    assert summary["fate"] == [{"lifecycle": "stuck", "particles": 1, "model_weight": 1.0}]
    assert summary["deposition"] == [
        {"boundary_id": 30, "material_id": 0, "events": 1, "model_weight": 1.0}
    ]
    assert summary["arrival"] == [
        {
            "boundary_id": 20,
            "material_id": 0,
            "outcome": "reflected",
            "events": 1,
            "model_weight": 1.0,
            "first_time_s": 1.0,
            "last_time_s": 1.0,
            "weighted_mean_time_s": 1.0,
        },
        {
            "boundary_id": 30,
            "material_id": 0,
            "outcome": "stuck",
            "events": 1,
            "model_weight": 1.0,
            "first_time_s": 2.0,
            "last_time_s": 2.0,
            "weighted_mean_time_s": 2.0,
        },
    ]

    report = render_result(
        result_path,
        tmp_path / "visualization",
        event_batch_rows=1,
        max_event_points=1,
    )

    assert report["visualization_revision"] == "result_trajectory_boundary_svg_v3"
    assert report["parameters"] == {
        "event_batch_rows": 1,
        "max_event_points": 1,
        "requested_particle_id": None,
    }
    assert report["trajectory_source"] == "frames"
    assert report["particle_id"] == 801
    assert report["trajectory_points"] == 6
    assert report["boundary_events"] == 2
    assert report["rendered_boundary_points"] == 1
    trajectory_svg = Path(str(report["trajectory_svg"]))
    events_svg = Path(str(report["boundary_events_svg"]))
    metadata_json = Path(str(report["metadata_json"]))
    assert "Representative trajectory: particle 801" in trajectory_svg.read_text(encoding="utf-8")
    assert "Boundary events: 1 rendered of 2" in events_svg.read_text(encoding="utf-8")
    assert '"visualization_revision": "result_trajectory_boundary_svg_v3"' in (
        metadata_json.read_text(encoding="utf-8")
    )

    with pytest.raises(ValueError, match="outside the source result"):
        render_result(result_path, result_path / "derived")


def test_hold_is_a_terminal_fate_but_not_deposition(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "hold-case")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["time"] = {"start_s": 0.0, "end_s": 1.25, "dt_s": 0.25}
    for boundary in document["boundaries"]:
        if boundary["boundary_group"] == "mirror":
            boundary["law"] = "hold"
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    result_path = tmp_path / "hold-result"
    simulate(load_case(paths.case_path), result_path)

    summary = summarize_result(result_path)

    assert summary["fate"] == [{"lifecycle": "held", "particles": 1, "model_weight": 1.0}]
    assert summary["deposition"] == []
    assert summary["arrival"] == [
        {
            "boundary_id": 20,
            "material_id": 0,
            "outcome": "held",
            "events": 1,
            "model_weight": 1.0,
            "first_time_s": 1.0,
            "last_time_s": 1.0,
            "weighted_mean_time_s": 1.0,
        }
    ]


def test_bounded_event_views_are_independent_of_storage_order_and_batch_size(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C08", tmp_path / "case")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0.0, 2.25]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    source = tmp_path / "result"
    simulate(load_case(paths.case_path), source)
    reordered = tmp_path / "reordered"
    copytree(source, reordered)
    reversed_rows = 0
    for segment in (reordered / "segments").glob("*.h5"):
        with h5py.File(segment, "r+") as handle:
            group = handle["events/boundary"]
            count = group["particle_id"].shape[0]
            permutation = np.arange(count - 1, -1, -1)
            offsets = group["candidate_offset"][...]
            facets = group["candidate_facet_id"][...]
            candidates = [facets[offsets[row] : offsets[row + 1]] for row in permutation]
            for name in group:
                if name not in {"candidate_offset", "candidate_facet_id"}:
                    group[name][...] = group[name][...][permutation]
            if count:
                group["candidate_offset"][...] = np.concatenate(
                    ([0], np.cumsum([row.size for row in candidates]))
                )
                group["candidate_facet_id"][...] = np.concatenate(candidates)
            reversed_rows += count if count > 1 else 0
    assert reversed_rows == 2
    original_events = open_result(source).read_boundary_events()
    reordered_events = open_result(reordered).read_boundary_events()
    np.testing.assert_array_equal(original_events.time_s, reordered_events.time_s)
    np.testing.assert_array_equal(original_events.event_ordinal, reordered_events.event_ordinal)
    original = render_result(source, tmp_path / "first", event_batch_rows=1, max_event_points=1)
    changed = render_result(reordered, tmp_path / "second", event_batch_rows=3, max_event_points=1)
    assert original["rendered_boundary_points"] == changed["rendered_boundary_points"] == 1
    assert original["event_selection"] == changed["event_selection"]
    for figure in ("trajectory_svg", "boundary_events_svg"):
        assert Path(str(original[figure])).read_text(encoding="utf-8") == Path(
            str(changed[figure])
        ).read_text(encoding="utf-8")
    trajectory = Path(str(original["trajectory_svg"])).read_text(encoding="utf-8")
    assert "<title>reflected</title>" in trajectory
    assert "<title>stuck</title>" not in trajectory
