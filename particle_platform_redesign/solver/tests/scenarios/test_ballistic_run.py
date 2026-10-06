from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import chamber_particles.engine as engine_module
from chamber_particles import (
    CaseError,
    IncompleteResultError,
    SimulationError,
    load_case,
    open_result,
    simulate,
)
from chamber_particles.case_format import (
    BoundaryData,
    FieldData,
    GeometryData,
    RealizedTableSource,
    RegularLayout,
    write,
)
from chamber_particles.integrators import ProposalSample, StepProposal
from chamber_particles.numerical_status import INTEGRATOR_NUMERICAL_FAILURE
from tests.verification.microcases import materialize_microcase


def test_c01_runs_through_all_three_public_operations(tmp_path: Path) -> None:
    paths = materialize_microcase("C01", tmp_path / "C01")
    expected = json.loads(paths.expected_path.read_text(encoding="utf-8"))
    case = load_case(paths.case_path)
    output = tmp_path / "result"

    summary = simulate(case, output)
    result = open_result(output)
    events = result.read_release_events()
    boundary_events = result.read_boundary_events()
    failure_events = result.read_failure_events()
    lifecycle = result.read_lifecycle_series()
    frames = list(result.iter_frames())
    final = result.read_final()

    assert summary.output_path == output.resolve()
    assert summary.particle_count == 1
    assert summary.release_event_count == 1
    assert summary.boundary_event_count == 0
    assert summary.failure_event_count == 0
    assert summary.series_count == summary.macro_step_count == 4
    assert summary.frame_count == len(expected["time_s"])
    assert summary.frame_row_count == len(expected["time_s"])
    assert summary.probe_count == 0
    assert summary.probe_row_count == 0
    np.testing.assert_array_equal(events.particle_id, [101])
    np.testing.assert_array_equal(events.time_s, [0.0])
    np.testing.assert_array_equal(events.event_ordinal, [0])
    assert [frame.time_s for frame in frames] == expected["time_s"]
    for index, frame in enumerate(frames):
        np.testing.assert_array_equal(frame.particle_id, [101])
        np.testing.assert_allclose(
            frame.position_m,
            expected["position_m"][index],
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            frame.velocity_m_s,
            expected["velocity_m_s"][index],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(frame.lifecycle, [1])
    np.testing.assert_array_equal(final.particle_id, [101])
    np.testing.assert_allclose(
        final.position_m,
        expected["position_m"][-1],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        final.velocity_m_s,
        expected["velocity_m_s"][-1],
        rtol=0.0,
        atol=0.0,
    )
    assert result.manifest["case_name"] == "C01"
    assert result.manifest["case_file_hash"] == case.case_file_hash
    assert result.manifest["data_content_hash"] == case.content_hash
    assert result.manifest["data_coordinate_system"] == "cartesian_xy"
    assert result.manifest["motion_mode"] == "cartesian_xy"
    assert boundary_events.particle_id.size == 0
    assert boundary_events.candidate_offset.tolist() == [0]
    assert failure_events.particle_id.size == 0
    np.testing.assert_allclose(
        lifecycle.time_s,
        [0.2, 0.4, 0.6, 0.8],
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_array_equal(lifecycle.pending, [0, 0, 0, 0])
    np.testing.assert_array_equal(lifecycle.active, [1, 1, 1, 1])
    np.testing.assert_array_equal(lifecycle.stuck, [0, 0, 0, 0])
    np.testing.assert_array_equal(lifecycle.held, [0, 0, 0, 0])
    np.testing.assert_array_equal(lifecycle.escaped, [0, 0, 0, 0])
    np.testing.assert_array_equal(lifecycle.failed, [0, 0, 0, 0])
    population = (
        lifecycle.pending
        + lifecycle.active
        + lifecycle.stuck
        + lifecycle.held
        + lifecycle.escaped
        + lifecycle.failed
    )
    np.testing.assert_array_equal(population, np.full(summary.macro_step_count, 1, dtype="<u8"))
    assert result.manifest["engine_algorithm_revision"] == "particle_engine_v37"
    assert result.manifest["compiled_cpu_tile_revision"] == "compiled_cpu_tile_v18"
    assert result.manifest["step_proposal_revision"] == "coupled_fixed_step_proposal_v10"
    assert result.manifest["rk4_enclosure_revision"] is None
    assert result.manifest["rk4_dense_path_revision"] is None
    assert result.manifest["field_location_revision"] == "field_location_v4"
    assert (
        result.manifest["geometry_algorithm_revision"]
        == "line_boundary_stackless_volume_cell_bvh_v5"
    )
    assert result.manifest["event_algorithm_revision"] == "line_quadratic_rk4_axis_first_hit_v16"
    assert result.manifest["boundary_algorithm_revision"] is None
    assert result.manifest["result_algorithm_revision"] == "durable_segmented_result_v5"
    assert result.manifest["counts"] == {
        "particles": summary.particle_count,
        "release_events": summary.release_event_count,
        "boundary_events": summary.boundary_event_count,
        "failure_events": summary.failure_event_count,
        "series": summary.series_count,
        "frames": summary.frame_count,
        "frame_rows": summary.frame_row_count,
        "probes": summary.probe_count,
        "probe_rows": summary.probe_row_count,
        "macro_steps": summary.macro_step_count,
    }
    np.testing.assert_array_equal(final.source_id, [0])
    np.testing.assert_array_equal(final.kinematics_valid, [1])
    np.testing.assert_array_equal(final.charge_number, case.data.sources[0].charge_number)
    np.testing.assert_array_equal(final.mass_kg, case.data.sources[0].mass_kg)
    assert (output / "_SUCCESS").is_file()


def test_boundaryless_durable_cadence_counts_direct_particle_pieces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = materialize_microcase("C01", tmp_path / "boundaryless-cadence")
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 4)
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_WORK_PER_PARTICLE", 0)
    output = tmp_path / "boundaryless-cadence-result"

    simulate(load_case(paths.case_path), output)
    result = open_result(output)

    assert result.manifest["segment_count"] == 2
    assert result.manifest["durable_commit_cadence"]["work_threshold"] == 4


def test_accepted_replay_numerical_failure_is_run_integrity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = materialize_microcase("C01", tmp_path / "replay-numerical-failure")
    original = StepProposal.state_at_rows

    def failed_replay(
        proposal: StepProposal,
        time_s: float,
        local_rows: np.ndarray,
    ) -> ProposalSample:
        sample = original(proposal, time_s, local_rows)
        return replace(
            sample,
            numerical_status=np.full(
                sample.particle_index.size,
                INTEGRATOR_NUMERICAL_FAILURE,
                dtype="<u1",
            ),
        )

    monkeypatch.setattr(StepProposal, "state_at_rows", failed_replay)
    output = tmp_path / "replay-numerical-failure-result"

    with pytest.raises(
        SimulationError,
        match="accepted trajectory replay produced a numerical failure",
    ):
        simulate(load_case(paths.case_path), output)

    assert not (output / "_SUCCESS").exists()


def test_staggered_sources_and_terminal_compaction_preserve_particle_identity(
    tmp_path: Path,
) -> None:
    base = materialize_microcase("C07", tmp_path / "base")
    base_case = load_case(base.case_path)
    first = _table(
        "first_table",
        particle_id=[20, 10],
        release_time_s=[0.4, -0.25],
        position_m=[[0.125, 0.75], [0.75, 0.5]],
        velocity_m_s=[[0.25, 0.0], [1.0, -0.0]],
    )
    second = _table(
        "second_table",
        particle_id=[30, 15, 35],
        release_time_s=[0.2, 0.4, 0.8],
        position_m=[[0.8, 0.5], [0.75, 0.25], [0.5, 0.5]],
        velocity_m_s=[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
    )
    data = replace(base_case.data, sources=(first, second), layouts=(), fields=())
    data_path = tmp_path / "staggered.h5"
    info = write(data_path, data)
    first_case_path = _write_case(
        tmp_path / "first.yaml",
        data_path.name,
        info.content_hash,
        dt_s=0.3,
        frame_times=[-0.5, -0.25, 0.0, 0.2, 0.4, 0.55, 0.8],
    )
    second_case_path = _write_case(
        tmp_path / "second.yaml",
        data_path.name,
        info.content_hash,
        dt_s=0.17,
        frame_times=[-0.25, 0.4, 0.8],
    )

    first_output = tmp_path / "first-result"
    second_output = tmp_path / "second-result"
    simulate(load_case(first_case_path), first_output)
    simulate(load_case(second_case_path), second_output)
    first_result = open_result(first_output)
    second_result = open_result(second_output)
    first_final = first_result.read_final()
    second_final = second_result.read_final()
    first_events = first_result.read_release_events()
    second_events = second_result.read_release_events()

    np.testing.assert_array_equal(first_final.particle_id, [10, 15, 20, 30, 35])
    for name in (
        "particle_id",
        "source_id",
        "time_s",
        "position_m",
        "velocity_m_s",
        "charge_number",
        "lifecycle",
        "kinematics_valid",
        "failure_reason_code",
        "mass_kg",
        "drag_diameter_m",
        "electrostatic_radius_m",
        "displaced_volume_m3",
        "model_weight",
        "material_id",
    ):
        np.testing.assert_array_equal(getattr(first_final, name), getattr(second_final, name))
    np.testing.assert_array_equal(first_events.particle_id, [10, 30, 15, 20, 35])
    np.testing.assert_array_equal(first_events.time_s, [-0.25, 0.2, 0.4, 0.4, 0.8])
    np.testing.assert_array_equal(first_events.source_id, [1, 0, 0, 1, 0])
    np.testing.assert_array_equal(first_events.particle_id, second_events.particle_id)
    np.testing.assert_array_equal(first_events.time_s, second_events.time_s)
    first_boundary = first_result.read_boundary_events()
    second_boundary = second_result.read_boundary_events()
    np.testing.assert_array_equal(first_boundary.particle_id, [10, 30, 15])
    np.testing.assert_allclose(
        first_boundary.time_s,
        [0.0, 0.4, 0.65],
        rtol=0.0,
        atol=6.0e-17,
    )
    np.testing.assert_array_equal(first_boundary.event_ordinal, [1, 1, 1])
    np.testing.assert_array_equal(first_boundary.particle_id, second_boundary.particle_id)
    np.testing.assert_array_equal(first_boundary.time_s, second_boundary.time_s)
    np.testing.assert_array_equal(first_boundary.event_ordinal, second_boundary.event_ordinal)
    first_frames = {frame.time_s: frame for frame in first_result.iter_frames()}
    second_frames = {frame.time_s: frame for frame in second_result.iter_frames()}
    assert first_frames[-0.5].particle_id.size == 0
    np.testing.assert_array_equal(first_frames[-0.25].particle_id, [10])
    np.testing.assert_array_equal(first_frames[-0.25].position_m, [[0.75, 0.5]])
    assert np.signbit(first_frames[-0.25].velocity_m_s[0, 1])
    np.testing.assert_array_equal(first_frames[0.4].particle_id, [10, 15, 20, 30])
    np.testing.assert_array_equal(first_frames[0.4].lifecycle, [2, 1, 1, 2])
    np.testing.assert_array_equal(first_frames[0.8].particle_id, [10, 15, 20, 30, 35])
    np.testing.assert_array_equal(first_frames[0.8].lifecycle, [2, 2, 1, 2, 1])
    np.testing.assert_array_equal(first_frames[0.8].position_m[-1], [0.5, 0.5])
    for time_s in second_frames:
        np.testing.assert_array_equal(
            first_frames[time_s].particle_id, second_frames[time_s].particle_id
        )
        np.testing.assert_array_equal(
            first_frames[time_s].position_m, second_frames[time_s].position_m
        )
        np.testing.assert_array_equal(
            first_frames[time_s].lifecycle, second_frames[time_s].lifecycle
        )
    for frame in first_frames.values():
        assert np.unique(frame.particle_id).size == frame.particle_id.size
        assert bool((np.diff(frame.particle_id) > 0).all())

    assert np.unique(first_events.particle_id).size == first_events.particle_id.size
    assert np.unique(first_boundary.particle_id).size == first_boundary.particle_id.size
    np.testing.assert_array_equal(
        first_final.particle_id,
        np.sort(first_events.particle_id),
    )
    np.testing.assert_array_equal(first_final.lifecycle, [2, 2, 1, 2, 1])
    assert first_result.manifest["lifecycle_counts"] == {
        "pending": 0,
        "active": 2,
        "stuck": 3,
        "held": 0,
        "escaped": 0,
        "failed": 0,
    }
    for result in (first_result, second_result):
        lifecycle = result.read_lifecycle_series()
        population = (
            lifecycle.pending
            + lifecycle.active
            + lifecycle.stuck
            + lifecycle.held
            + lifecycle.escaped
            + lifecycle.failed
        )
        np.testing.assert_array_equal(population, np.full(population.size, 5, dtype="<u8"))


def test_result_publication_is_no_clobber_and_completed_recovery_open_is_valid(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C01", tmp_path / "C01")
    case = load_case(paths.case_path)
    output = tmp_path / "result"
    simulate(case, output)
    manifest_before = (output / "run.json").read_bytes()

    with pytest.raises(SimulationError, match="already exists"):
        simulate(case, output)
    assert (output / "run.json").read_bytes() == manifest_before
    recovered = open_result(output, recovery=True)
    np.testing.assert_array_equal(recovered.read_final().particle_id, [101])

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(IncompleteResultError):
        open_result(incomplete)


def test_trajectory_output_can_be_disabled_without_disabling_mandatory_results(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C01", tmp_path / "C01-no-frames")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["output"]["trajectories"] = None
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "result-no-frames"

    summary = simulate(load_case(paths.case_path), output)
    result = open_result(output)

    assert summary.frame_count == 0
    assert summary.frame_row_count == 0
    assert list(result.iter_frames()) == []
    assert result.manifest["memory_plan"]["components"]["replay_work"] == 0
    assert result.read_release_events().particle_id.size == 1
    assert result.read_final().particle_id.size == 1


def test_memory_gate_counts_unused_canonical_field_arrays(tmp_path: Path) -> None:
    paths = materialize_microcase("C01", tmp_path / "memory-case")
    case = load_case(paths.case_path)
    axis = np.linspace(0.0, 1.0, 400, dtype="<f8")
    layout = RegularLayout(
        "unused_grid",
        axis,
        axis.copy(),
        np.ones((399, 399), dtype="<u1"),
    )
    field = FieldData(
        "unused_scalar",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.zeros((400 * 400, 1), dtype="<f8"),
        "1",
    )
    data_path = paths.case_path.parent / "memory-case.h5"
    info = write(
        data_path,
        replace(case.data, layouts=(layout,), fields=(field,)),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["resources"]["memory_limit_mb"] = 1
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(CaseError, match=r"resources\.memory_limit_mb"):
        load_case(paths.case_path)
    assert not (tmp_path / "memory-result").exists()


def test_boundaryless_case_still_runs_global_volume_topology_audit(tmp_path: Path) -> None:
    paths = materialize_microcase("C01", tmp_path / "invalid-topology")
    case = load_case(paths.case_path)
    empty_boundary = BoundaryData(
        line2=np.empty((0, 2), dtype="<i8"),
        boundary_id=np.empty(0, dtype="<i4"),
        group_id=np.empty(0, dtype="<i4"),
        material_id=np.empty(0, dtype="<i4"),
        owner_cell_type=np.empty(0, dtype="<u1"),
        owner_cell_local_index=np.empty(0, dtype="<i8"),
        orientation=np.empty(0, dtype="<i1"),
    )
    geometry = GeometryData(
        nodes_m=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, -1.0], [0.5, 2.0]],
            dtype="<f8",
        ),
        boundary=empty_boundary,
        group_names=(),
        tri3=np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype="<i8"),
        tri3_domain_id=np.zeros(3, dtype="<i4"),
    )
    data_path = paths.case_path.parent / "invalid-topology.h5"
    info = write(data_path, replace(case.data, geometry=geometry))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    malformed = load_case(paths.case_path)
    with pytest.raises(SimulationError, match="non-manifold volume edge"):
        simulate(malformed, tmp_path / "invalid-topology-result")


def test_open_result_rejects_unclosed_or_misaligned_segment(tmp_path: Path) -> None:
    paths = materialize_microcase("C01", tmp_path / "C01-corrupt")
    output = tmp_path / "result-corrupt"
    simulate(load_case(paths.case_path), output)
    segment_path = output / "segments" / "epoch-000000.h5"

    with h5py.File(segment_path, "r+") as segment:
        segment.attrs["closed"] = np.uint8(0)
    with pytest.raises(SimulationError, match="closed"):
        open_result(output)

    with h5py.File(segment_path, "r+") as segment:
        segment.attrs["closed"] = np.uint8(1)
        charge = segment["frames/charge_number"]
        assert isinstance(charge, h5py.Dataset)
        charge.resize(charge.shape[0] - 1, axis=0)
    with pytest.raises(SimulationError, match="inconsistent shapes"):
        open_result(output)

    with h5py.File(segment_path, "r+") as segment:
        charge = segment["frames/charge_number"]
        offsets = segment["frames/offset"]
        assert isinstance(charge, h5py.Dataset)
        assert isinstance(offsets, h5py.Dataset)
        charge.resize(charge.shape[0] + 1, axis=0)
        offset_values = offsets[...]
        offset_values[1:3] = [2, 1]
        offsets[...] = offset_values
    with pytest.raises(SimulationError, match="offsets"):
        open_result(output)


def test_p06_rejects_unsupported_coordinate_pair(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C01", tmp_path / "mismatch")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(SimulationError, match="supported pair"):
        simulate(load_case(paths.case_path), tmp_path / "result-mismatch")


def test_nonfinite_derived_ballistic_state_is_published_as_local_failure(tmp_path: Path) -> None:
    base = materialize_microcase("C01", tmp_path / "overflow-base")
    base_case = load_case(base.case_path)
    overflow_source = _table(
        "particles",
        particle_id=[1],
        release_time_s=[0.0],
        position_m=[[1.0e308, 0.0]],
        velocity_m_s=[[1.0e308, 0.0]],
    )
    data_path = tmp_path / "overflow.h5"
    info = write(
        data_path,
        replace(base_case.data, sources=(overflow_source,), layouts=(), fields=()),
    )
    case_path = _write_single_source_case(
        tmp_path / "overflow.yaml", data_path.name, info.content_hash, motion="cartesian_xy"
    )
    output = tmp_path / "overflow-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    failure = result.read_failure_events()
    reason = result.manifest["failure_reason_codes"]["nonfinite_physics"]
    np.testing.assert_array_equal(final.lifecycle, [4])
    np.testing.assert_array_equal(final.failure_reason_code, [reason])
    np.testing.assert_array_equal(failure.reason_code, [reason])


def test_rz_ballistic_run_splits_and_folds_at_the_axis(tmp_path: Path) -> None:
    base = materialize_microcase("C01", tmp_path / "base-rz")
    base_case = load_case(base.case_path)
    geometry = replace(
        base_case.data.geometry,
        nodes_m=np.asarray([[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]]),
    )
    noncrossing = _table(
        "particles",
        particle_id=[1],
        release_time_s=[0.0],
        position_m=[[1.0, 0.0]],
        velocity_m_s=[[-0.25, 1.0]],
    )
    data_path = tmp_path / "rz.h5"
    info = write(
        data_path,
        replace(
            base_case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            sources=(noncrossing,),
            layouts=(),
            fields=(),
        ),
    )
    case_path = _write_single_source_case(
        tmp_path / "rz.yaml", data_path.name, info.content_hash, motion="axisymmetric_rz_meridional"
    )
    result_path = tmp_path / "rz-result"
    simulate(load_case(case_path), result_path)
    np.testing.assert_allclose(
        open_result(result_path).read_final().position_m, [[0.75, 1.0]], rtol=0.0, atol=0.0
    )

    crossing = replace(
        noncrossing,
        position_m=np.asarray([[0.1, 0.0]], dtype="<f8"),
        velocity_m_s=np.asarray([[-0.2, 0.0]], dtype="<f8"),
    )
    crossing_path = tmp_path / "rz-crossing.h5"
    crossing_info = write(
        crossing_path,
        replace(
            base_case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            sources=(crossing,),
            layouts=(),
            fields=(),
        ),
    )
    crossing_case = _write_single_source_case(
        tmp_path / "rz-crossing.yaml",
        crossing_path.name,
        crossing_info.content_hash,
        motion="axisymmetric_rz_meridional",
    )
    crossing_result_path = tmp_path / "rz-crossing-result"
    simulate(load_case(crossing_case), crossing_result_path)
    crossing_result = open_result(crossing_result_path)
    np.testing.assert_allclose(
        crossing_result.read_final().position_m,
        [[0.1, 0.0]],
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_array_equal(crossing_result.read_final().velocity_m_s, [[0.2, 0.0]])
    assert crossing_result.manifest["boundary_interactions"]["axis_crossings"] == 1


def test_unrepresentable_rz_axis_time_fails_only_that_particle(tmp_path: Path) -> None:
    base = materialize_microcase("C01", tmp_path / "axis-precision-base")
    base_case = load_case(base.case_path)
    geometry = replace(
        base_case.data.geometry,
        nodes_m=np.asarray([[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]]),
    )
    start_s = float(2**52)
    source = _table(
        "particles",
        particle_id=[1, 2],
        release_time_s=[start_s, start_s],
        position_m=[[0.25, 0.0], [1.0, 0.0]],
        velocity_m_s=[[-1.0, 0.0], [0.1, 0.0]],
    )
    data_path = tmp_path / "axis-precision.h5"
    info = write(
        data_path,
        replace(
            base_case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            sources=(source,),
            layouts=(),
            fields=(),
        ),
    )
    case_path = _write_single_source_case(
        tmp_path / "axis-precision.yaml",
        data_path.name,
        info.content_hash,
        motion="axisymmetric_rz_meridional",
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["time"] = {"start_s": start_s, "end_s": start_s + 1.0, "dt_s": 1.0}
    document["output"]["trajectories"]["schedule"]["explicit_times_s"] = [
        start_s,
        start_s + 1.0,
    ]
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "axis-precision-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    failure = result.read_failure_events()
    reason = result.manifest["failure_reason_codes"]["indeterminate_event"]
    np.testing.assert_array_equal(final.lifecycle, [4, 1])
    np.testing.assert_array_equal(final.failure_reason_code, [reason, 0])
    np.testing.assert_array_equal(failure.particle_id, [1])
    np.testing.assert_array_equal(failure.reason_code, [reason])
    assert failure.time_s[0] == start_s


def _table(
    name: str,
    *,
    particle_id: list[int],
    release_time_s: list[float],
    position_m: list[list[float]],
    velocity_m_s: list[list[float]],
) -> RealizedTableSource:
    count = len(particle_id)
    return RealizedTableSource(
        name=name,
        particle_id=np.asarray(particle_id, dtype="<i8"),
        release_time_s=np.asarray(release_time_s, dtype="<f8"),
        position_m=np.asarray(position_m, dtype="<f8"),
        velocity_m_s=np.asarray(velocity_m_s, dtype="<f8"),
        charge_number=np.zeros(count, dtype="<f8"),
        mass_kg=np.full(count, 1.0e-15, dtype="<f8"),
        drag_diameter_m=np.full(count, 1.0e-6, dtype="<f8"),
        electrostatic_radius_m=np.full(count, 5.0e-7, dtype="<f8"),
        displaced_volume_m3=np.zeros(count, dtype="<f8"),
        model_weight=np.ones(count, dtype="<f8"),
        material_id=np.zeros(count, dtype="<i4"),
    )


def _write_case(
    path: Path,
    data_path: str,
    content_hash: str,
    *,
    dt_s: float,
    frame_times: list[float],
) -> Path:
    document = {
        "format_version": 2,
        "case": {
            "name": path.stem,
            "data_path": data_path,
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": "cartesian_xy"},
        "time": {"start_s": -0.5, "end_s": 0.8, "dt_s": dt_s},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 7,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": {"charge": {"model": "fixed"}},
        "sources": [
            {"name": "second", "type": "table", "table": "second_table"},
            {"name": "first", "type": "table", "table": "first_table"},
        ],
        "boundaries": [{"boundary_group": "wall", "priority": 10, "law": "stick"}],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": frame_times},
            }
        },
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _write_single_source_case(
    path: Path, data_path: str, content_hash: str, *, motion: str
) -> Path:
    document = {
        "format_version": 2,
        "case": {
            "name": path.stem,
            "data_path": data_path,
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": motion},
        "time": {"start_s": 0.0, "end_s": 1.0, "dt_s": 0.3},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 0,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": {"charge": {"model": "fixed"}},
        "sources": [{"name": "source", "type": "table", "table": "particles"}],
        "boundaries": [],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": [0.0, 1.0]},
            }
        },
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path
