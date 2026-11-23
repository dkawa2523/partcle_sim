"""T03 result tools stay outside the solver and consume only ResultView data."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from shutil import copytree
from typing import Any
from xml.etree import ElementTree

import h5py
import numpy as np
import pytest
import yaml

from chamber_particles import SimulationError, load_case, open_result, simulate
from chamber_particles.case_format import write
from chamber_particles.output import ResultView, ResultWriter
from chamber_particles_tools.analysis import export_result, summarize_result
from chamber_particles_tools.selection import select_result
from chamber_particles_tools.visualization import render_result
from tests.scenarios.test_periodic_run import _materialize_periodic_case
from tests.verification.microcases import build_microcase, materialize_microcase
from tests.verification.test_output import (
    _checkpoint_state,
    _final_particles,
    _identity,
    _initialize_context,
)


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
    paths.case_path.unlink()
    (paths.case_path.parent / "case.h5").unlink()

    summary: dict[str, Any] = summarize_result(result_path, event_batch_rows=1)

    assert summary["analysis_revision"] == "result_population_interaction_terminal_v4"
    assert summary["parameters"]["event_batch_rows"] == 1
    assert summary["fate"] == [{"lifecycle": "stuck", "particles": 1, "model_weight": 1.0}]
    assert summary["deposition"] == [
        {"boundary_id": 30, "material_id": 0, "particles": 1, "model_weight": 1.0}
    ]
    assert summary["boundary_interactions"] == [
        {
            "boundary_id": 20,
            "material_id": 0,
            "interaction_kind": "wall",
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
            "interaction_kind": "wall",
            "outcome": "stuck",
            "events": 1,
            "model_weight": 1.0,
            "first_time_s": 2.0,
            "last_time_s": 2.0,
            "weighted_mean_time_s": 2.0,
        },
    ]

    assert summary["first_boundary_hit"]["time_s_quantiles_p10_p50_p90"] == [1.0] * 3
    assert summary["terminal"][0]["time_s_quantiles_p10_p50_p90"] == [2.0] * 3
    report: dict[str, Any] = render_result(
        result_path,
        tmp_path / "visualization",
        focus_particle_id=801,
        event_batch_rows=1,
        max_event_points=1,
    )

    assert report["visualization_revision"] == "result_context_population_svg_v5"
    assert report["parameters"]["event_batch_rows"] == 1
    assert report["trajectory_source"] == "frames"
    assert report["focus_particle_id"] == 801
    assert report["trajectory_points"] == 6
    assert report["boundary_events"] == 2
    assert report["rendered_boundary_points"] == 1
    trajectory_svg = Path(str(report["trajectory_svg"]))
    events_svg = Path(str(report["boundary_events_svg"]))
    metadata_json = Path(str(report["metadata_json"]))
    assert "Trajectory detail: particle 801" in trajectory_svg.read_text(encoding="utf-8")
    assert "Boundary interactions: 1 rendered of 2" in events_svg.read_text(encoding="utf-8")
    assert '"visualization_revision": "result_context_population_svg_v5"' in (
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

    summary: dict[str, Any] = summarize_result(result_path)

    assert summary["fate"] == [{"lifecycle": "held", "particles": 1, "model_weight": 1.0}]
    assert summary["deposition"] == []
    assert summary["boundary_interactions"] == [
        {
            "boundary_id": 20,
            "material_id": 0,
            "interaction_kind": "wall",
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
    original_view = open_result(source)
    # This alternate reader fixture declares its changed storage bytes. A
    # completed run with the original prefix/checkpoint must reject the edit.
    manifest = original_view.manifest_document()
    segments = tuple(reordered / path.relative_to(source) for path in original_view.segment_paths)
    for path in segments:
        manifest["content_files"][path.relative_to(reordered).as_posix()] = (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
    reordered_view = ResultView(
        reordered, manifest, segments, True, original_view.context_source_count
    )
    original_events = original_view.read_boundary_events()
    reordered_events = reordered_view.read_boundary_events()
    np.testing.assert_array_equal(original_events.time_s, reordered_events.time_s)
    np.testing.assert_array_equal(original_events.event_ordinal, reordered_events.event_ordinal)
    original = render_result(
        source, tmp_path / "first", focus_particle_id=801, event_batch_rows=1, max_event_points=1
    )
    changed = render_result(
        reordered_view,
        tmp_path / "second",
        focus_particle_id=801,
        event_batch_rows=3,
        max_event_points=1,
    )
    assert original["rendered_boundary_points"] == changed["rendered_boundary_points"] == 1
    assert original["event_selection"] == changed["event_selection"]
    for figure in ("trajectory_svg", "boundary_events_svg"):
        first_svg = Path(str(original[figure])).read_text(encoding="utf-8")
        second_svg = Path(str(changed[figure])).read_text(encoding="utf-8")
        assert first_svg.replace(
            original_view.manifest_sha256, "SOURCE_IDENTITY"
        ) == second_svg.replace(reordered_view.manifest_sha256, "SOURCE_IDENTITY")
    trajectory = Path(str(original["trajectory_svg"])).read_text(encoding="utf-8")
    assert "<title>reflected</title>" in trajectory
    assert "<title>stuck</title>" not in trajectory


def _cohort_result(
    directory: Path, second_id: int = 9001, *, shared_boundary_id: bool = False
) -> Path:
    definition = build_microcase("C08")
    first = definition.data.sources[0]
    second = replace(
        first,
        name="second_table",
        particle_id=np.asarray([second_id], dtype="<i8"),
        release_time_s=np.asarray([0.5], dtype="<f8"),
        model_weight=np.asarray([2.5], dtype="<f8"),
    )
    sources = (first, second)
    geometry = definition.data.geometry
    if shared_boundary_id:
        geometry = replace(
            geometry,
            boundary=replace(
                geometry.boundary, boundary_id=np.full_like(geometry.boundary.boundary_id, 7)
            ),
        )
    data = replace(definition.data, sources=sources, geometry=geometry)
    directory.mkdir()
    info = write(directory / "case.h5", data)
    document = definition.spec
    document["case"]["expected_content_hash"] = info.content_hash
    later = {"name": "later", "type": "surface", "table": "second_table"}
    document["sources"] = [*document["sources"], later]
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0.0, 0.5, 1.0, 1.5, 2.0, 2.25]},
    }
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    result = directory / "result"
    simulate(load_case(case_path), result)
    return result


def test_release_names_and_exact_id_file_share_one_population_across_clients(
    tmp_path: Path,
) -> None:
    identifier = 2**53 + 1
    result = open_result(_cohort_result(tmp_path / "cohort", identifier, shared_boundary_id=True))
    selection_path = tmp_path / "ids.csv"
    selection_raw = f"particle_id\n{identifier}.0\n".encode()
    selection_path.write_bytes(selection_raw)
    cohort = select_result(result, source_names=("later",), particle_ids_file=selection_path)
    np.testing.assert_array_equal(cohort.selected_ids, [identifier])
    np.testing.assert_array_equal(
        cohort.selected_ids,
        select_result(result, source_ids=(1,), particle_ids=(identifier,)).selected_ids,
    )
    selection_path.unlink()  # Bound consumers reuse exact IDs, not a mutable path.
    summary: dict[str, Any] = summarize_result(cohort)
    exported: dict[str, Any] = export_result(cohort, tmp_path / "export")
    view: dict[str, Any] = render_result(cohort, tmp_path / "view", focus_particle_id=identifier)
    receipt = cohort.selection_document()
    assert (
        summary["population"]["selected_particles"]
        == exported["selection"]["selected_particles"]
        == view["population"]["selected_particles"]
        == 1
    )
    assert receipt["source_names"] == ("later",) and receipt["source_ids"] == (1,)
    assert receipt["particle_ids"] is None
    assert receipt["particle_ids_file"] == {
        "path": str(selection_path.resolve()),
        "sha256": "sha256:" + hashlib.sha256(selection_raw).hexdigest(),
        "rows": 1,
    }
    for key, value in receipt.items():
        assert (
            summary["parameters"][key]
            == exported["selection"][key]
            == view["parameters"][key]
            == value
        )
    assert view["focus_particle_id"] == identifier
    assert summary["labels"] == exported["labels"] == view["labels"]
    assert summary["labels"]["sources"][1]["table"] == "second_table"
    assert summary["labels"]["sources"][1]["name"] == "later"
    assert {item["boundary_id"] for item in summary["labels"]["boundaries"]} == {7}
    assert {item["group_name"] for item in summary["labels"]["boundaries"]} == set(
        cohort.context.geometry.group_names
    )
    with pytest.raises(ValueError, match="unknown release names"):
        select_result(result, source_names=("second_table",))
    with pytest.raises(ValueError, match="unique nonempty"):
        select_result(result, source_names=("later", "later"))
    with pytest.raises(ValueError, match="choose source_ids"):
        select_result(result, source_names=("later",), source_ids=(1,))
    with pytest.raises(ValueError, match="choose particle_ids"):
        select_result(result, particle_ids=(identifier,), particle_ids_file=selection_path)


@pytest.mark.parametrize(
    "contents, message",
    [
        ("id\n9001\n", "single header"),
        ("particle_id\n9001,extra\n", "exactly one cell"),
        ("particle_id\n9001\n9001\n", "repeat"),
        ("particle_id\nNaN\n", "nonnegative int64"),
        ("particle_id\n9001.1\n", "nonnegative int64"),
        (f"particle_id\n{2**63}\n", "nonnegative int64"),
    ],
)
def test_id_file_rejects_ambiguous_or_invalid_population(
    tmp_path: Path, contents: str, message: str
) -> None:
    result = _cohort_result(tmp_path / "cohort")
    selection_path = tmp_path / "ids.csv"
    selection_path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        select_result(result, particle_ids_file=selection_path)


def test_exact_cohort_weight_histogram_and_streaming_export(tmp_path: Path) -> None:
    result = _cohort_result(tmp_path / "cohort")
    summary: dict[str, Any] = summarize_result(
        result,
        source_ids=(1,),
        particle_ids=(9001,),
        coordinate0_bin_edges_m=(0.0, 0.5, 1.0),
        terminal_time_bin_edges_s=(0.0, 1.0, 3.0),
        event_batch_rows=1,
    )
    assert summary["population"]["selected_particles"] == 1
    assert summary["population"]["model_weight"] == 2.5
    assert summary["sources"][1]["source_id"] == 1
    assert summary["sources"][1]["name"] == "later"
    assert summary["sources"][1]["table"] == "second_table"
    assert summary["sources"][1]["kind"] == "surface"
    assert summary["fate"] == [{"lifecycle": "active", "particles": 1, "model_weight": 2.5}]
    assert summary["deposition"] == []
    assert summary["first_boundary_hit"]["time_s_quantiles_p10_p50_p90"] == [1.5] * 3
    assert summary["active_curve"]["rows"][0]["active_particles"] == 0
    assert summary["active_curve"]["rows"][-1]["active_model_weight"] == 2.5
    assert summary["histograms"]["terminal_time_s"]["missing_values"] == 1
    with pytest.raises(ValueError, match="unknown ID"):
        summarize_result(result, particle_ids=(9000,))
    with pytest.raises(ValueError, match="repeat"):
        summarize_result(result, particle_ids=(9001, 9001))
    with pytest.raises(ValueError, match="strictly increasing"):
        summarize_result(result, coordinate0_bin_edges_m=(1.0, 0.0))
    receipt: dict[str, Any] = export_result(
        result, tmp_path / "csv", particle_ids=(9001,), event_batch_rows=1
    )
    assert receipt["rows"]["final"] == 1 and receipt["rows"]["frames"] == 5
    with (tmp_path / "csv" / "final.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["particle_id"] == "9001" and row["model_weight"] == "2.5"
    assert (tmp_path / "csv" / "context.h5").is_file()
    rendered: dict[str, Any] = render_result(
        result, tmp_path / "population", source_ids=(1,), max_particle_points=1
    )
    assert rendered["population"]["binned_positions"] == 1
    assert rendered["population"]["rendered_cells"] <= 1
    assert rendered["focus_particle_id"] is None and rendered["trajectory_source"] == "unrequested"


def test_opened_result_cohort_is_reused_by_all_consumers_without_reopening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = open_result(_cohort_result(tmp_path / "cohort"))
    expected = summarize_result(view, source_ids=(1,), particle_ids=(9001,))
    cohort = select_result(view, source_ids=(1,), particle_ids=(9001,))
    np.testing.assert_array_equal(cohort.selected_ids, [9001])
    with pytest.raises(ValueError, match="read-only"):
        cohort.selected[0] = True
    with pytest.raises(ValueError, match="read-only"):
        cohort.selected_ids[0] = 801
    with pytest.raises(ValueError, match="already defines"):
        select_result(cohort, particle_ids=(801,))
    for arguments in (
        {"source_ids": (-1,)},
        {"source_ids": (2,)},
        {"particle_ids": (-1,)},
        {"particle_ids": (2**63,)},
    ):
        with pytest.raises(ValueError, match="unknown ID"):
            select_result(view, **arguments)
    np.testing.assert_array_equal(
        select_result(view, particle_ids=(9001, 801)).selected_ids, [801, 9001]
    )
    assert not select_result(view, source_ids=(0,), particle_ids=(9001,)).selected_ids.size
    assert not select_result(view, particle_ids=()).selected_ids.size

    def unexpected_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an already bound cohort must not reopen or reread final/context")

    monkeypatch.setattr("chamber_particles_tools.selection.open_result", unexpected_read)
    monkeypatch.setattr(ResultView, "read_context", unexpected_read)
    monkeypatch.setattr(ResultView, "read_final", unexpected_read)
    summary = summarize_result(cohort)
    assert summary == expected
    exported = export_result(cohort, tmp_path / "export")
    rendered = render_result(cohort, tmp_path / "view")
    assert exported["source_result"] == summary["source_result"]
    assert rendered["source_manifest_sha256"] == view.manifest_sha256
    assert rendered["population"]["selected_particles"] == 1
    assert exported["selection"]["selected_particles"] == 1


@pytest.mark.parametrize("operation", ["export", "view"])
def test_derived_publication_failure_leaves_no_partial_package_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    cohort = select_result(_cohort_result(tmp_path / "cohort"))
    destination = tmp_path / "derived"
    write_text = Path.write_text
    receipt_name = "export.json" if operation == "export" else "visualization.json"
    publish = export_result if operation == "export" else render_result

    def fail_receipt(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name == receipt_name:
            raise OSError("receipt publication interrupted")
        return write_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_receipt)
        with pytest.raises(OSError, match="publication interrupted"):
            publish(cohort, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".result-*-*"))
    publish(cohort, destination)
    receipt = (destination / receipt_name).read_bytes()
    with pytest.raises(FileExistsError):
        publish(cohort, destination)
    assert (destination / receipt_name).read_bytes() == receipt


def test_export_rejects_a_context_replaced_after_cohort_was_opened(tmp_path: Path) -> None:
    cohort = select_result(_cohort_result(tmp_path / "cohort"))
    with (cohort.result.path / "context.h5").open("ab") as stream:
        stream.write(b"changed after opening")
    destination = tmp_path / "export"
    with pytest.raises(ValueError, match="context changed after opening"):
        export_result(cohort, destination)
    assert not destination.exists()


def test_pending_no_frames_and_no_events_render_context_without_nan(tmp_path: Path) -> None:
    result = tmp_path / "pending"
    final = replace(
        _final_particles(),
        lifecycle=np.zeros(3, dtype="<u1"),
        kinematics_valid=np.zeros(3, dtype="<u1"),
        failure_reason_code=np.zeros(3, dtype="<u2"),
    )
    checkpoint = _checkpoint_state(macro_step_count=1)
    checkpoint = replace(
        checkpoint,
        particles=replace(
            checkpoint.particles,
            lifecycle=final.lifecycle,
            failure_reason_code=final.failure_reason_code,
            terminal_time_s=np.full(3, np.inf),
        ),
        active_particle_index=np.empty(0, dtype="<i8"),
        release_cursor=0,
        probe_cursor=0,
    )
    with ResultWriter(result, 3, _identity("all-pending-display")) as writer:
        _initialize_context(writer)
        writer.begin_epoch()
        writer.commit_epoch(checkpoint)
        writer.finalize(final, {}, macro_step_count=1)
    summary: dict[str, Any] = summarize_result(result)
    assert summary["fate"] == [{"lifecycle": "pending", "particles": 3, "model_weight": 4.5}]
    assert summary["boundary_interactions"] == []
    assert summary["first_boundary_hit"]["time_s_quantiles_p10_p50_p90"] is None
    report: dict[str, Any] = render_result(result, tmp_path / "empty", focus_particle_id=10)
    assert report["boundary_events"] == 0 and report["trajectory_points"] == 0
    assert report["population"]["omitted_nonspatial_or_pending"] == 3
    for key in ("population_svg", "trajectory_svg", "boundary_events_svg"):
        svg = Path(report[key]).read_text(encoding="utf-8")
        assert 'stroke="#64748b"' in svg
        for element in ElementTree.fromstring(svg).iter():
            for name, value in element.attrib.items():
                if name in {
                    "x",
                    "y",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "cx",
                    "cy",
                    "r",
                    "width",
                    "height",
                    "opacity",
                }:
                    assert np.isfinite(float(value.removesuffix("%")))


def test_periodic_interactions_are_not_first_wall_hits_and_break_saved_chords(
    tmp_path: Path,
) -> None:
    case = _materialize_periodic_case(
        tmp_path / "periodic", position_m=(0.25, 0.5), velocity_m_s=(3.0, 0.0), end_s=1.0, dt_s=1.0
    )
    document = yaml.safe_load(case.case_path.read_text(encoding="utf-8"))
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
    }
    case.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    case = load_case(case.case_path)
    result = tmp_path / "periodic-result"
    simulate(case, result)
    context = open_result(result).read_context()
    assert context.topology is not None
    assert context.topology.pairs[0].first_to_second_m == (1.0, 0.0)
    summary: dict[str, Any] = summarize_result(result, event_batch_rows=1)
    assert summary["first_boundary_hit"]["particles"] == 0
    assert sum(row["events"] for row in summary["boundary_interactions"]) == 3
    report: dict[str, Any] = render_result(
        result,
        tmp_path / "periodic-svg",
        focus_particle_id=int(case.data.sources[0].particle_id[0]),
        event_batch_rows=1,
    )
    svg = Path(report["trajectory_svg"]).read_text(encoding="utf-8")
    assert svg.count('stroke="#111827"') == 2


def test_no_frame_cohort_curve_uses_complete_right_continuous_ledgers(tmp_path: Path) -> None:
    first = _cohort_result(tmp_path / "no-frame-case")
    case_path = first.parent / "case.yaml"
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["time"]["dt_s"] = 0.25
    document["output"]["trajectories"] = None
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    result = tmp_path / "no-frame-result"
    simulate(load_case(case_path), result)
    summary: dict[str, Any] = summarize_result(result)
    curve = summary["active_curve"]
    assert curve["available"] and "ledgers" in curve["source"]
    bytime = {row["time_s"]: row for row in curve["rows"]}
    assert bytime[0.25]["active_particles"] == 1
    assert bytime[0.5]["active_particles"] == 2
    assert bytime[0.5]["active_model_weight"] == 3.5
    assert bytime[2.0]["active_particles"] == 1
    assert bytime[2.0]["active_model_weight"] == 2.5
    selected: dict[str, Any] = summarize_result(result, source_ids=(1,))
    selected_time = {row["time_s"]: row for row in selected["active_curve"]["rows"]}
    assert selected_time[0.25]["active_particles"] == 0
    assert selected_time[0.5]["active_model_weight"] == 2.5
    assert summary["histograms"]["coordinate0_axis"] == {"name": "x", "unit": "m"}
    assert not list(open_result(result).iter_frames())


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "top_case",
        "top_data",
        "missing_identity",
        "missing_case_parent",
        "missing_data_parent",
        "changed_seed",
        "changed_seed_rehashed",
        "missing_top_case",
        "missing_top_data",
        "missing_context_identity",
        "missing_identity_hash",
    ],
)
def test_fresh_result_manifest_identity_is_required_rehashed_and_checkpoint_bound(
    tmp_path: Path, recovery: bool, damage: str
) -> None:
    paths = materialize_microcase("C08", tmp_path / "identity-case")
    case = load_case(paths.case_path)
    output = tmp_path / "identity-result"
    simulate(case, output)
    open_result(output)
    directory = output
    if recovery:
        directory = output.with_name(output.name + ".partial")
        output.rename(directory)
        (directory / "_SUCCESS").unlink()
    context_bytes = (directory / "context.h5").read_bytes()
    manifest_path = directory / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if damage in ("top_case", "top_data"):
        key = "case_file_hash" if damage == "top_case" else "data_content_hash"
        manifest[key] = "sha256:" + "f" * 64
    elif damage.startswith("missing_top_"):
        manifest.pop("case_file_hash" if damage.endswith("case") else "data_content_hash")
    elif damage == "missing_identity":
        manifest.pop("resume_identity")
    elif damage == "missing_identity_hash":
        manifest.pop("resume_identity_hash")
    else:
        identity = manifest["resume_identity"]
        if damage == "missing_case_parent":
            identity.pop("case_file_hash")
        elif damage == "missing_data_parent":
            identity.pop("data_content_hash")
        elif damage == "missing_context_identity":
            identity.pop("result_context")
        else:
            identity["seed"] = 999
        if damage != "changed_seed":
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
            manifest["resume_identity_hash"] = (
                "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
            )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SimulationError):
        open_result(output, recovery=recovery)
    if recovery:
        with pytest.raises(SimulationError):
            simulate(case, output)
    assert (directory / "context.h5").read_bytes() == context_bytes
