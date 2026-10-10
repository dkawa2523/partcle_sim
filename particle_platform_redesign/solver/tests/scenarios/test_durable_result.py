from __future__ import annotations

import copy
import json
import os
import shutil
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import yaml

import chamber_particles.engine as engine_module
import chamber_particles.output as output_module
from chamber_particles import (
    IncompleteResultError,
    SimulationError,
    load_case,
    open_result,
    simulate,
)
from chamber_particles.case_format import write
from tests.verification.microcases import build_microcase

_DT_S = 2.0e-5
_END_S = 130 * _DT_S
_SECOND_RELEASE_S = 64 * _DT_S
_THIRD_RELEASE_S = 128 * _DT_S
_OUTPUT_TIMES_S = (0.0, _SECOND_RELEASE_S, 65 * _DT_S, _THIRD_RELEASE_S, _END_S)


@pytest.fixture(autouse=True)
def _lower_durable_commit_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep failure-injection cases small while retaining the production 128*N term."""

    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 192)


def test_completed_result_combines_every_closed_epoch(tmp_path: Path) -> None:
    case = _materialize_durable_case(tmp_path / "case")
    output = tmp_path / "result"

    summary = simulate(case, output)
    result = open_result(output)

    segment_paths = sorted((output / "segments").glob("epoch-*.h5"))
    assert [path.name for path in segment_paths] == [
        "epoch-000000.h5",
        "epoch-000001.h5",
    ]
    for index, path in enumerate(segment_paths):
        with h5py.File(path, "r") as segment:
            assert segment.attrs["closed"] == np.uint8(1)
            assert segment.attrs["segment_index"] == index

    release = result.read_release_events()
    boundary = result.read_boundary_events()
    boundary_batches = list(result.iter_boundary_event_batches(batch_rows=2))
    failure = result.read_failure_events()
    series = result.read_lifecycle_series()
    frames = list(result.iter_frames())
    probes = list(result.iter_probes())

    np.testing.assert_array_equal(release.particle_id, [901, 902, 903])
    np.testing.assert_array_equal(release.time_s, [0.0, _SECOND_RELEASE_S, _THIRD_RELEASE_S])
    assert boundary.particle_id.size > 2
    assert 901 in boundary.particle_id
    assert sum(batch.particle_id.size for batch in boundary_batches) == boundary.particle_id.size
    assert all(0 < batch.particle_id.size <= 2 for batch in boundary_batches)
    assert all(not batch.position_m.flags.writeable for batch in boundary_batches)
    for batch in boundary_batches:
        assert int(batch.candidate_offset[0]) == 0
        assert int(batch.candidate_offset[-1]) == batch.candidate_facet_id.size
        for index, primary in enumerate(batch.primary_facet_id):
            begin = int(batch.candidate_offset[index])
            end = int(batch.candidate_offset[index + 1])
            assert primary in batch.candidate_facet_id[begin:end]
    with pytest.raises(ValueError, match="positive integer"):
        list(result.iter_boundary_event_batches(batch_rows=0))
    np.testing.assert_array_equal(failure.particle_id, [902])
    assert series.time_s.size == summary.macro_step_count == 130
    assert len(frames) == len(probes) == len(_OUTPUT_TIMES_S)
    np.testing.assert_array_equal([frame.time_s for frame in frames], _OUTPUT_TIMES_S)
    np.testing.assert_array_equal([probe.time_s for probe in probes], _OUTPUT_TIMES_S)

    assert int(boundary.candidate_offset[0]) == 0
    assert int(boundary.candidate_offset[-1]) == boundary.candidate_facet_id.size
    assert bool((np.diff(boundary.candidate_offset) > 0).all())
    for index, primary in enumerate(boundary.primary_facet_id):
        begin = int(boundary.candidate_offset[index])
        end = int(boundary.candidate_offset[index + 1])
        assert primary in boundary.candidate_facet_id[begin:end]

    counts = result.manifest["counts"]
    assert isinstance(counts, dict)
    assert counts == {
        "particles": 3,
        "release_events": release.particle_id.size,
        "boundary_events": boundary.particle_id.size,
        "failure_events": failure.particle_id.size,
        "series": series.time_s.size,
        "frames": len(frames),
        "frame_rows": sum(frame.particle_id.size for frame in frames),
        "probes": len(probes),
        "probe_rows": sum(probe.particle_id.size for probe in probes),
        "macro_steps": 130,
    }
    cadence = result.manifest["durable_commit_cadence"]
    assert cadence == {
        "revision": "cumulative_solver_work_v1",
        "work_threshold": 128 * 3,
        "work_components": [
            "macro_step_count",
            "accepted_particle_pieces",
            "candidate_queries",
            "refinements",
        ],
        "barrier": "accepted_macro_step",
    }
    assert result.manifest["resume_identity"]["durable_commit_cadence"] == cadence
    assert "epoch_macro_steps" not in result.manifest
    assert _segment_macro_counts(result) == [124, 6]
    memory_plan = result.manifest["memory_plan"]
    assert isinstance(memory_plan, dict)
    components = memory_plan["components"]
    assert isinstance(components, dict)
    assert components["slab_event_staging"] == (
        memory_plan["event_staging_fixed_bytes"]
        + memory_plan["slab_particles"] * memory_plan["event_staging_bytes_per_row"]
    )
    assert components["slab_failure_staging"] == (
        memory_plan["slab_particles"] * memory_plan["failure_staging_bytes_per_particle"]
    )
    np.testing.assert_array_equal(result.read_final().particle_id, [901, 902, 903])


def test_default_commit_work_floor_is_resolved_and_final_epoch_is_forced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 1 << 20)
    case = _materialize_durable_case(tmp_path / "case")
    output = tmp_path / "default-cadence"

    simulate(case, output)
    result = open_result(output)

    assert result.manifest["durable_commit_cadence"]["work_threshold"] == 1 << 20
    assert result.manifest["segment_count"] == 1
    assert _segment_macro_counts(result) == [130]


@pytest.mark.parametrize("damage", ("shortened", "negative", "duplicate", "reversed"))
def test_completed_result_rejects_final_count_and_particle_identity_corruption(
    tmp_path: Path, damage: str
) -> None:
    case = _materialize_durable_case(tmp_path / "case")
    original = tmp_path / "original"
    simulate(case, original)
    output = tmp_path / damage
    shutil.copytree(original, output)
    opened = open_result(output)
    with h5py.File(output / "final.h5", "r+") as handle:
        particles = handle["particles"]
        if damage == "shortened":
            for name in tuple(particles):
                values = particles[name][:-1]
                del particles[name]
                particles.create_dataset(name, data=values)
        elif damage == "negative":
            particles["particle_id"][0] = -1
        elif damage == "duplicate":
            particles["particle_id"][1] = particles["particle_id"][0]
        else:
            particles["particle_id"][...] = particles["particle_id"][...][::-1]
    with pytest.raises(SimulationError):
        open_result(output)
    with pytest.raises(output_module.ResultOpenError):
        opened.read_final()


def test_completed_result_rejects_duplicate_id_across_a_large_final_table(tmp_path: Path) -> None:
    case, _ = _materialize_cadence_identity_cases(tmp_path / "case")
    output = tmp_path / "large-final"
    simulate(case, output)
    with h5py.File(output / "final.h5", "r+") as handle:
        ids = handle["particles/particle_id"]
        assert ids.shape == (5_000,)
        ids[4096] = ids[4095]
    with pytest.raises(SimulationError):
        open_result(output)


@pytest.mark.parametrize(
    ("name", "value"),
    (("particles", True), ("particles", -1), ("macro_steps", 130.0), ("macro_steps", -1)),
)
def test_completed_result_requires_integer_nonnegative_manifest_counts(
    tmp_path: Path, name: str, value: bool | int | float
) -> None:
    case = _materialize_durable_case(tmp_path / "case")
    output = tmp_path / "invalid-count"
    simulate(case, output)
    manifest_path = output / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"][name] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SimulationError):
        open_result(output)


def test_commit_cadence_is_output_schedule_and_slab_independent(tmp_path: Path) -> None:
    tight_case, wide_case = _materialize_cadence_identity_cases(tmp_path / "cadence-cases")
    tight_output = tmp_path / "tight-framed"
    wide_output = tmp_path / "wide-unframed"

    simulate(tight_case, tight_output)
    simulate(wide_case, wide_output)
    tight = open_result(tight_output)
    wide = open_result(wide_output)

    tight_plan = tight.manifest["memory_plan"]
    wide_plan = wide.manifest["memory_plan"]
    assert tight_plan["slab_particles"] < 5_000
    assert wide_plan["slab_particles"] == 5_000
    assert tight_plan["planned_bytes"] <= tight_plan["limit_bytes"]
    assert wide_plan["planned_bytes"] <= wide_plan["limit_bytes"]

    cadence = tight.manifest["durable_commit_cadence"]
    assert cadence == wide.manifest["durable_commit_cadence"]
    assert cadence["work_threshold"] == 128 * 5_000
    assert tight.manifest["resume_identity"]["durable_commit_cadence"] == cadence
    assert wide.manifest["resume_identity"]["durable_commit_cadence"] == cadence
    assert _segment_macro_counts(tight) == _segment_macro_counts(wide) == [63, 3]
    assert _checkpoint_work(tight, "A.h5") == _checkpoint_work(wide, "A.h5") == 640_063

    for reader in (
        "read_final",
        "read_release_events",
        "read_boundary_events",
        "read_failure_events",
        "read_lifecycle_series",
    ):
        _assert_record_identity(getattr(tight, reader)(), getattr(wide, reader)())


@pytest.mark.parametrize(
    ("failure_boundary", "expected_committed_macros"),
    (("segment", 124), ("checkpoint", 124), ("latest", 130)),
)
def test_interrupted_run_recovers_latest_commit_and_resumes_without_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
    expected_committed_macros: int,
) -> None:
    case = _materialize_durable_case(tmp_path / "case")
    uninterrupted_path = tmp_path / "uninterrupted"
    interrupted_path = tmp_path / f"interrupted-{failure_boundary}"
    simulate(case, uninterrupted_path)

    with monkeypatch.context() as patch:
        patch.setattr(
            output_module.os,
            "replace",
            _replace_failure_after_second_commit(failure_boundary),
        )
        with pytest.raises(SimulationError):
            simulate(case, interrupted_path)

    with pytest.raises(IncompleteResultError):
        open_result(interrupted_path)
    recovered = open_result(interrupted_path, recovery=True)
    assert recovered.read_lifecycle_series().time_s.size == expected_committed_macros
    with pytest.raises(RuntimeError, match=r"final|incomplete"):
        recovered.read_final()

    if failure_boundary == "segment":
        orphan = interrupted_path.with_name(f"{interrupted_path.name}.partial") / "segments"
        orphan_segment = orphan / "epoch-000001.h5"
        assert orphan_segment.is_file()
        orphan_segment.write_bytes(b"uncommitted orphan must be ignored")
        incompatible_case = _materialize_durable_case(
            tmp_path / "incompatible-case",
            case_name="P13 incompatible resume identity",
        )
        with pytest.raises(SimulationError):
            simulate(incompatible_case, interrupted_path)
        with monkeypatch.context() as patch:
            patch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 128 * 3 + 1)
            with pytest.raises(SimulationError):
                simulate(case, interrupted_path)

    simulate(case, interrupted_path)
    _assert_public_result_identity(
        open_result(uninterrupted_path),
        open_result(interrupted_path),
    )


def test_failure_before_first_latest_restarts_from_initial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _materialize_durable_case(tmp_path / "case")
    uninterrupted_path = tmp_path / "uninterrupted"
    interrupted_path = tmp_path / "interrupted-before-first-latest"
    simulate(case, uninterrupted_path)

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", _replace_failure_after_segment(0))
        with pytest.raises(SimulationError):
            simulate(case, interrupted_path)

    partial = interrupted_path.with_name(f"{interrupted_path.name}.partial")
    assert not (partial / "LATEST").exists()
    simulate(case, interrupted_path)
    _assert_public_result_identity(
        open_result(uninterrupted_path),
        open_result(interrupted_path),
    )


@pytest.mark.parametrize("failure_boundary", ("final", "manifest", "success", "publication"))
def test_finalization_failure_is_safely_republished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
) -> None:
    case = _materialize_durable_case(tmp_path / "case")
    uninterrupted_path = tmp_path / "uninterrupted"
    interrupted_path = tmp_path / f"interrupted-{failure_boundary}"
    simulate(case, uninterrupted_path)

    with monkeypatch.context() as patch:
        if failure_boundary == "publication":
            patch.setattr(
                output_module.os,
                "rename",
                _rename_failure_before_publication(interrupted_path),
            )
        else:
            patch.setattr(
                output_module.os,
                "replace",
                _replace_failure_during_finalization(failure_boundary),
            )
        with pytest.raises(SimulationError):
            simulate(case, interrupted_path)

    with pytest.raises(IncompleteResultError):
        open_result(interrupted_path)
    simulate(case, interrupted_path)
    _assert_public_result_identity(
        open_result(uninterrupted_path),
        open_result(interrupted_path),
    )


def test_probabilistic_wall_rng_is_identical_after_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _materialize_probabilistic_resume_case(tmp_path / "case")
    uninterrupted_path = tmp_path / "uninterrupted"
    interrupted_path = tmp_path / "interrupted"
    simulate(case, uninterrupted_path)

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", _replace_failure_after_second_commit("latest"))
        with pytest.raises(SimulationError):
            simulate(case, interrupted_path)

    recovered = open_result(interrupted_path, recovery=True)
    np.testing.assert_array_equal(recovered.read_boundary_events().outcome, ["reflected"])
    simulate(case, interrupted_path)
    resumed = open_result(interrupted_path)
    np.testing.assert_array_equal(
        resumed.read_boundary_events().outcome,
        ["reflected", "stuck"],
    )
    _assert_public_result_identity(open_result(uninterrupted_path), resumed)


def test_maxwell_wall_rng_is_identical_after_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _materialize_maxwell_resume_case(tmp_path / "maxwell-case")
    uninterrupted_path = tmp_path / "maxwell-uninterrupted"
    interrupted_path = tmp_path / "maxwell-interrupted"
    simulate(case, uninterrupted_path)

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", _replace_failure_after_second_commit("latest"))
        with pytest.raises(SimulationError):
            simulate(case, interrupted_path)

    recovered = open_result(interrupted_path, recovery=True)
    np.testing.assert_array_equal(recovered.read_boundary_events().law_id, ["maxwell_thermal"])
    simulate(case, interrupted_path)
    resumed = open_result(interrupted_path)
    _assert_public_result_identity(open_result(uninterrupted_path), resumed)


def test_held_checkpoint_resumes_without_reactivating_particle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _materialize_hold_resume_case(tmp_path / "hold-case")
    uninterrupted_path = tmp_path / "hold-uninterrupted"
    interrupted_path = tmp_path / "hold-interrupted"
    simulate(case, uninterrupted_path)

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", _replace_failure_after_second_commit("latest"))
        with pytest.raises(SimulationError):
            simulate(case, interrupted_path)

    recovered = open_result(interrupted_path, recovery=True)
    recovered_series = recovered.read_lifecycle_series()
    np.testing.assert_array_equal(recovered_series.held[-1:], [1])
    np.testing.assert_array_equal(recovered_series.active[-1:], [0])

    simulate(case, interrupted_path)
    resumed = open_result(interrupted_path)
    np.testing.assert_array_equal(resumed.read_boundary_events().law_id, ["hold"])
    np.testing.assert_array_equal(resumed.read_boundary_events().outcome, ["held"])
    np.testing.assert_array_equal(resumed.read_final().lifecycle, [5])
    _assert_public_result_identity(open_result(uninterrupted_path), resumed)


@pytest.mark.parametrize("artifact", ("checkpoint", "segment"))
def test_recovery_rejects_corrupt_referenced_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    case = _materialize_durable_case(tmp_path / "case")
    output = tmp_path / f"corrupt-{artifact}"
    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", _replace_failure_after_second_commit("latest"))
        with pytest.raises(SimulationError):
            simulate(case, output)

    partial = output.with_name(f"{output.name}.partial")
    latest = json.loads((partial / "LATEST").read_text(encoding="utf-8"))
    if artifact == "checkpoint":
        referenced = partial / "checkpoints" / latest["checkpoint"]
        referenced.write_bytes(b"corrupt committed checkpoint")
    else:
        referenced = partial / "segments" / f"epoch-{latest['segment_index']:06d}.h5"
        with h5py.File(referenced, "r+") as segment:
            segment.attrs["corruption_marker"] = np.uint8(1)

    with pytest.raises(SimulationError):
        open_result(output, recovery=True)
    with pytest.raises(SimulationError):
        simulate(case, output)


def test_recovery_rejects_valid_hdf5_with_missing_rows_in_older_segment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _materialize_durable_case(tmp_path / "case")
    output = tmp_path / "missing-committed-row"
    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", _replace_failure_after_second_commit("latest"))
        with pytest.raises(SimulationError):
            simulate(case, output)

    partial = output.with_name(f"{output.name}.partial")
    older_segment = partial / "segments" / "epoch-000000.h5"
    with h5py.File(older_segment, "r+") as segment:
        series = segment["series"]
        for name in ("time_s", "pending", "active", "stuck", "held", "escaped", "failed"):
            dataset = series[name]
            dataset.resize(dataset.shape[0] - 1, axis=0)

    with pytest.raises(SimulationError, match="counts do not match LATEST"):
        open_result(output, recovery=True)
    with pytest.raises(SimulationError, match="counts do not match LATEST"):
        simulate(case, output)


def _materialize_durable_case(
    directory: Path,
    *,
    case_name: str = "P13 durable result",
) -> Any:
    definition = build_microcase("C09")
    source = definition.data.sources[0]
    source_type = type(source)
    particle_count = 3
    repeated = {
        name: np.repeat(getattr(source, name), particle_count, axis=0)
        for name in (
            "charge_number",
            "mass_kg",
            "drag_diameter_m",
            "electrostatic_radius_m",
            "displaced_volume_m3",
            "model_weight",
            "material_id",
            "contact_radius_m",
        )
    }
    particles = source_type(
        name=source.name,
        particle_id=np.asarray([901, 902, 903], dtype="<i8"),
        release_time_s=np.asarray([0.0, _SECOND_RELEASE_S, _THIRD_RELEASE_S], dtype="<f8"),
        position_m=np.repeat(source.position_m, particle_count, axis=0),
        velocity_m_s=np.asarray([[1.0, 0.0], [1000.0, 0.0], [0.0, 0.0]], dtype="<f8"),
        **repeated,
    )
    data = type(definition.data)(
        coordinate_system=definition.data.coordinate_system,
        provenance_json=definition.data.provenance_json,
        geometry=definition.data.geometry,
        layouts=definition.data.layouts,
        fields=definition.data.fields,
        sources=(particles,),
    )

    spec = copy.deepcopy(definition.spec)
    spec["case"]["name"] = case_name
    spec["time"] = {"start_s": 0.0, "end_s": _END_S, "dt_s": _DT_S}
    spec["solver"]["event"]["max_interactions_per_step"] = 1
    spec["solver"]["event"]["max_refinements"] = 1
    spec["output"] = {
        "trajectories": {
            "selection": "all",
            "schedule": {"explicit_times_s": list(_OUTPUT_TIMES_S)},
        },
        "probes": {
            "particle_ids": [901, 902, 903],
            "schedule": {"explicit_times_s": list(_OUTPUT_TIMES_S)},
        },
    }

    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "case.h5"
    info = write(data_path, data)
    spec["case"]["data_path"] = data_path.name
    spec["case"]["expected_content_hash"] = info.content_hash
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return load_case(case_path)


def _materialize_cadence_identity_cases(directory: Path) -> tuple[Any, Any]:
    definition = build_microcase("C09")
    source = definition.data.sources[0]
    particle_count = 5_000
    repeated = {
        name: np.repeat(getattr(source, name), particle_count, axis=0)
        for name in (
            "release_time_s",
            "position_m",
            "velocity_m_s",
            "charge_number",
            "mass_kg",
            "drag_diameter_m",
            "electrostatic_radius_m",
            "displaced_volume_m3",
            "model_weight",
            "material_id",
            "contact_radius_m",
        )
    }
    particles = replace(
        source,
        particle_id=np.arange(10_000, 10_000 + particle_count, dtype="<i8"),
        **repeated,
    )
    data = replace(definition.data, sources=(particles,))

    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "case.h5"
    info = write(data_path, data)
    base = copy.deepcopy(definition.spec)
    base["case"]["data_path"] = data_path.name
    base["case"]["expected_content_hash"] = info.content_hash
    base["time"] = {
        "start_s": 0.0,
        "end_s": 66 * _DT_S,
        "dt_s": _DT_S,
    }

    cases = []
    for label, memory_limit_mb, trajectories in (
        (
            "tight-framed",
            8,
            {
                "selection": "all",
                "schedule": {"explicit_times_s": [0.0, 33 * _DT_S, 66 * _DT_S]},
            },
        ),
        ("wide-unframed", 128, None),
    ):
        spec = copy.deepcopy(base)
        spec["case"]["name"] = f"durable cadence {label}"
        spec["resources"]["memory_limit_mb"] = memory_limit_mb
        spec["output"] = {"trajectories": trajectories}
        case_path = directory / f"{label}.yaml"
        case_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        cases.append(load_case(case_path))
    return cases[0], cases[1]


def _materialize_probabilistic_resume_case(directory: Path) -> Any:
    definition = build_microcase("C08")
    spec = copy.deepcopy(definition.spec)
    spec["case"]["name"] = "P13 probabilistic checkpoint resume"
    spec["time"] = {"start_s": 0.0, "end_s": 2.25, "dt_s": 0.01}
    for boundary in spec["boundaries"]:
        if boundary["boundary_group"] in {"mirror", "collector"}:
            boundary["law"] = "probabilistic_stick"
            boundary["probability"] = 0.5
            boundary["otherwise"] = {"law": "specular"}

    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "case.h5"
    info = write(data_path, definition.data)
    spec["case"]["data_path"] = data_path.name
    spec["case"]["expected_content_hash"] = info.content_hash
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return load_case(case_path)


def _materialize_maxwell_resume_case(directory: Path) -> Any:
    definition = build_microcase("C08")
    spec = copy.deepcopy(definition.spec)
    spec["case"]["name"] = "Maxwell thermal checkpoint resume"
    spec["time"] = {"start_s": 0.0, "end_s": 2.25, "dt_s": 0.01}
    for boundary in spec["boundaries"]:
        if boundary["boundary_group"] == "mirror":
            boundary.clear()
            boundary.update(
                {
                    "boundary_group": "mirror",
                    "priority": 20,
                    "law": "maxwell_thermal",
                    "wall_temperature_K": 300.0,
                    "diffuse_reflection_fraction": 1.0,
                    "wall_velocity_m_s": [0.0, 0.0],
                }
            )

    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "case.h5"
    info = write(data_path, definition.data)
    spec["case"]["data_path"] = data_path.name
    spec["case"]["expected_content_hash"] = info.content_hash
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return load_case(case_path)


def _materialize_hold_resume_case(directory: Path) -> Any:
    definition = build_microcase("C08")
    spec = copy.deepcopy(definition.spec)
    spec["case"]["name"] = "P18-H held checkpoint resume"
    spec["time"] = {"start_s": 0.0, "end_s": 2.25, "dt_s": 0.01}
    for boundary in spec["boundaries"]:
        if boundary["boundary_group"] == "mirror":
            boundary["law"] = "hold"

    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "case.h5"
    info = write(data_path, definition.data)
    spec["case"]["data_path"] = data_path.name
    spec["case"]["expected_content_hash"] = info.content_hash
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return load_case(case_path)


def _replace_failure_after_second_commit(boundary: str) -> Any:
    real_replace = os.replace
    checkpoint_count = 0
    latest_count = 0

    def replace_and_fail(source: Any, destination: Any) -> None:
        nonlocal checkpoint_count, latest_count
        destination_path = Path(destination)
        real_replace(source, destination)
        if destination_path.parent.name == "checkpoints" and destination_path.suffix == ".h5":
            checkpoint_count += 1
            if boundary == "checkpoint" and checkpoint_count == 2:
                raise OSError("injected P13 failure after checkpoint replace")
        if destination_path.name == "LATEST":
            latest_count += 1
            if boundary == "latest" and latest_count == 2:
                raise OSError("injected P13 failure after LATEST replace")
        if boundary == "segment" and destination_path.name == "epoch-000001.h5":
            raise OSError("injected P13 failure after segment replace")

    return replace_and_fail


def _replace_failure_after_segment(segment_index: int) -> Any:
    real_replace = os.replace
    expected_name = f"epoch-{segment_index:06d}.h5"

    def replace_and_fail(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        real_replace(source, destination)
        if destination_path.parent.name == "segments" and destination_path.name == expected_name:
            raise OSError("injected P13 failure after segment replace")

    return replace_and_fail


def _replace_failure_during_finalization(boundary: str) -> Any:
    real_replace = os.replace
    manifest_count = 0

    def replace_and_fail(source: Any, destination: Any) -> None:
        nonlocal manifest_count
        destination_path = Path(destination)
        real_replace(source, destination)
        if destination_path.name == "run.json":
            manifest_count += 1
        should_fail = (
            (boundary == "final" and destination_path.name == "final.h5")
            or (boundary == "manifest" and manifest_count == 2)
            or (boundary == "success" and destination_path.name == "_SUCCESS")
        )
        if should_fail:
            raise OSError(f"injected P13 failure after {boundary} replace")

    return replace_and_fail


def _rename_failure_before_publication(output: Path) -> Any:
    real_rename = os.rename
    expected_source = output.with_name(f"{output.name}.partial").resolve()
    expected_destination = output.resolve()

    def rename_and_fail(source: Any, destination: Any) -> None:
        if (
            Path(source).resolve() == expected_source
            and Path(destination).resolve() == expected_destination
        ):
            raise OSError("injected P13 failure before result publication")
        real_rename(source, destination)

    return rename_and_fail


def _segment_macro_counts(result: Any) -> list[int]:
    counts = []
    for path in result.segment_paths:
        with h5py.File(path, "r") as segment:
            counts.append(int(segment["series"]["time_s"].shape[0]))
    return counts


def _checkpoint_work(result: Any, name: str) -> int:
    with h5py.File(result.path / "checkpoints" / name, "r") as checkpoint:
        return sum(
            int(checkpoint.attrs[counter])
            for counter in (
                "macro_step_count",
                "accepted_particle_pieces",
                "candidate_queries",
                "refinements",
            )
        )


def _assert_record_identity(expected: Any, actual: Any) -> None:
    assert type(expected) is type(actual)
    for field in dataclass_fields(expected):
        expected_value = getattr(expected, field.name)
        actual_value = getattr(actual, field.name)
        if isinstance(expected_value, np.ndarray):
            assert isinstance(actual_value, np.ndarray)
            assert expected_value.dtype == actual_value.dtype
            assert expected_value.shape == actual_value.shape
            assert expected_value.tobytes(order="C") == actual_value.tobytes(order="C")
        else:
            assert expected_value == actual_value


def _assert_public_result_identity(reference: Any, resumed: Any) -> None:
    science_manifest_keys = (
        "status",
        "result_schema_version",
        "result_algorithm_revision",
        "durable_commit_cadence",
        "segment_count",
        "latest_commit_id",
        "case_name",
        "case_file_hash",
        "data_content_hash",
        "case_schema_version",
        "data_coordinate_system",
        "motion_mode",
        "time",
        "event",
        "random_draw_kinds",
        "event_refinement",
        "boundary_interactions",
        "source_id_to_name",
        "lifecycle_counts",
        "failure_reason_codes",
        "failure_reason_counts",
        "maximum_dt_over_tau",
        "maximum_dt_charge_lipschitz",
        "counts",
    )
    for name in science_manifest_keys:
        assert reference.manifest[name] == resumed.manifest[name]
    assert {
        name: value for name, value in reference.manifest.items() if name.endswith("_revision")
    } == {name: value for name, value in resumed.manifest.items() if name.endswith("_revision")}
    assert reference.manifest["requested"] == resumed.manifest["requested"]
    assert reference.manifest["resolved"] == resumed.manifest["resolved"]
    assert _segment_macro_counts(reference) == _segment_macro_counts(resumed)

    readers = (
        "read_final",
        "read_release_events",
        "read_boundary_events",
        "read_failure_events",
        "read_lifecycle_series",
    )
    reference_records = [getattr(reference, name)() for name in readers]
    resumed_records = [getattr(resumed, name)() for name in readers]
    for iterator in ("iter_frames", "iter_probes"):
        reference_records.extend(getattr(reference, iterator)())
        resumed_records.extend(getattr(resumed, iterator)())
    assert len(reference_records) == len(resumed_records)
    for expected, actual in zip(reference_records, resumed_records, strict=True):
        _assert_record_identity(expected, actual)
