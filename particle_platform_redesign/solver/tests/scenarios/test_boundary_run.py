from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from chamber_particles import SimulationError, load_case, open_result, simulate
from chamber_particles.boundaries import half_range_maxwell_flux_velocity
from chamber_particles.case_format import BoundaryData, write
from chamber_particles.rng import (
    WALL_MAXWELL_NORMAL_STREAM,
    WALL_MAXWELL_TANGENTIAL_STREAM,
    wall_standard_normal_batch,
    wall_uniform_open_batch,
)
from tests.verification.microcases import materialize_microcase


def test_c07_stick_event_and_terminal_state_are_published_by_public_api(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07")
    output = tmp_path / "result"

    summary = simulate(load_case(paths.case_path), output)
    result = open_result(output)
    release = result.read_release_events()
    boundary = result.read_boundary_events()
    final = result.read_final()

    assert summary.particle_count == 1
    assert summary.release_event_count == 1
    assert summary.boundary_event_count == 1
    np.testing.assert_array_equal(release.event_ordinal, [0])
    np.testing.assert_array_equal(boundary.particle_id, [701])
    np.testing.assert_array_equal(boundary.event_ordinal, [1])
    np.testing.assert_array_equal(boundary.time_s, [1.5])
    np.testing.assert_array_equal(boundary.primary_facet_id, [1])
    np.testing.assert_array_equal(boundary.candidate_offset, [0, 1])
    np.testing.assert_array_equal(boundary.candidate_facet_id, [1])
    np.testing.assert_array_equal(boundary.boundary_id, [10])
    np.testing.assert_array_equal(boundary.material_id, [0])
    np.testing.assert_array_equal(boundary.position_m, [[1.0, 0.625]])
    np.testing.assert_array_equal(boundary.normal, [[1.0, 0.0]])
    np.testing.assert_array_equal(boundary.velocity_pre_m_s, [[0.5, 0.25]])
    np.testing.assert_array_equal(boundary.velocity_post_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(boundary.charge_number_pre, [3.0])
    np.testing.assert_array_equal(boundary.charge_number_post, [3.0])
    np.testing.assert_array_equal(boundary.model_weight, [1.0])
    np.testing.assert_array_equal(boundary.law_id, ["stick"])
    np.testing.assert_array_equal(boundary.outcome, ["stuck"])
    assert boundary.localization_residual_m[0] <= boundary.position_budget_m[0]
    assert boundary.position_budget_m[0] > 0.0
    assert boundary.time_budget_s[0] > 0.0

    np.testing.assert_array_equal(final.particle_id, [701])
    np.testing.assert_array_equal(final.time_s, [2.0])
    np.testing.assert_array_equal(final.position_m, [[1.0, 0.625]])
    np.testing.assert_array_equal(final.velocity_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(final.charge_number, [3.0])
    np.testing.assert_array_equal(final.lifecycle, [2])
    np.testing.assert_array_equal(final.kinematics_valid, [1])
    assert result.manifest["counts"]["boundary_events"] == 1
    assert result.manifest["lifecycle_counts"]["stuck"] == 1
    assert result.manifest["geometry_algorithm_revision"] == "line_boundary_capsule_contact_bvh_v7"
    assert result.manifest["event_algorithm_revision"] == (
        "line_quadratic_curved_capsule_periodic_first_hit_v22"
    )
    assert result.manifest["boundary_algorithm_revision"] == "contact_wall_laws_v7"
    memory_plan = result.manifest["memory_plan"]
    assert isinstance(memory_plan, dict)
    assert memory_plan["geometry_preparation_transient_bytes"] > 0
    assert memory_plan["components"]["prepared_geometry"] > 0


def test_escape_is_right_continuous_and_independent_of_trajectory_output(
    tmp_path: Path,
) -> None:
    scheduled_paths = materialize_microcase("C07", tmp_path / "scheduled-case")
    disabled_paths = materialize_microcase("C07", tmp_path / "disabled-case")
    _set_terminal_case(scheduled_paths.case_path, law="escape", frame_times=[1.0, 1.5, 2.0])
    _set_terminal_case(disabled_paths.case_path, law="escape", frame_times=None)

    scheduled_output = tmp_path / "scheduled-result"
    disabled_output = tmp_path / "disabled-result"
    simulate(load_case(scheduled_paths.case_path), scheduled_output)
    simulate(load_case(disabled_paths.case_path), disabled_output)
    scheduled = open_result(scheduled_output)
    disabled = open_result(disabled_output)

    frames = list(scheduled.iter_frames())
    assert [frame.time_s for frame in frames] == [1.0, 1.5, 2.0]
    np.testing.assert_array_equal(frames[0].particle_id, [701])
    np.testing.assert_array_equal(frames[0].position_m, [[0.75, 0.5]])
    np.testing.assert_array_equal(frames[0].velocity_m_s, [[0.5, 0.25]])
    np.testing.assert_array_equal(frames[0].lifecycle, [1])
    assert frames[1].particle_id.size == 0
    assert frames[2].particle_id.size == 0
    assert list(disabled.iter_frames()) == []

    scheduled_event = scheduled.read_boundary_events()
    disabled_event = disabled.read_boundary_events()
    for name in (
        "time_s",
        "particle_id",
        "event_ordinal",
        "primary_facet_id",
        "position_m",
        "normal",
        "velocity_pre_m_s",
        "velocity_post_m_s",
        "law_id",
        "outcome",
        "candidate_offset",
        "candidate_facet_id",
    ):
        np.testing.assert_array_equal(getattr(scheduled_event, name), getattr(disabled_event, name))
    np.testing.assert_array_equal(scheduled_event.law_id, ["escape"])
    np.testing.assert_array_equal(scheduled_event.outcome, ["escaped"])
    np.testing.assert_array_equal(scheduled_event.velocity_post_m_s, [[0.5, 0.25]])

    scheduled_final = scheduled.read_final()
    disabled_final = disabled.read_final()
    for name in (
        "particle_id",
        "time_s",
        "position_m",
        "velocity_m_s",
        "charge_number",
        "lifecycle",
        "kinematics_valid",
    ):
        np.testing.assert_array_equal(getattr(scheduled_final, name), getattr(disabled_final, name))
    np.testing.assert_array_equal(scheduled_final.time_s, [2.0])
    np.testing.assert_array_equal(scheduled_final.position_m, [[1.0, 0.625]])
    np.testing.assert_array_equal(scheduled_final.velocity_m_s, [[0.5, 0.25]])
    np.testing.assert_array_equal(scheduled_final.lifecycle, [3])
    np.testing.assert_array_equal(scheduled_final.kinematics_valid, [0])
    assert np.isfinite(scheduled_final.position_m).all()
    assert scheduled.manifest["lifecycle_counts"]["escaped"] == 1


def test_maxwell_thermal_reflection_is_output_schedule_independent(
    tmp_path: Path,
) -> None:
    scheduled_paths = materialize_microcase("C07", tmp_path / "scheduled-maxwell-case")
    disabled_paths = materialize_microcase("C07", tmp_path / "disabled-maxwell-case")
    _set_maxwell_case(scheduled_paths.case_path, frame_times=[1.0, 1.5, 2.0])
    _set_maxwell_case(disabled_paths.case_path, frame_times=None)

    scheduled_output = tmp_path / "scheduled-maxwell-result"
    disabled_output = tmp_path / "disabled-maxwell-result"
    scheduled_case = load_case(scheduled_paths.case_path)
    disabled_case = load_case(disabled_paths.case_path)
    simulate(scheduled_case, scheduled_output)
    simulate(disabled_case, disabled_output)
    scheduled = open_result(scheduled_output)
    disabled = open_result(disabled_output)

    event = scheduled.read_boundary_events()
    disabled_event = disabled.read_boundary_events()
    np.testing.assert_array_equal(event.law_id, ["maxwell_thermal"])
    np.testing.assert_array_equal(event.outcome, ["reflected"])
    np.testing.assert_array_equal(event.normal, [[1.0, 0.0]])
    for name in (
        "time_s",
        "particle_id",
        "event_ordinal",
        "position_m",
        "normal",
        "velocity_pre_m_s",
        "velocity_post_m_s",
        "law_id",
        "outcome",
    ):
        np.testing.assert_array_equal(getattr(event, name), getattr(disabled_event, name))

    particle_id = np.asarray([701], dtype="<u8")
    event_ordinal = np.asarray([0], dtype="<u8")
    normal_draw = wall_uniform_open_batch(
        0,
        particle_id,
        event_ordinal,
        WALL_MAXWELL_NORMAL_STREAM,
    )[0]
    tangent_draw = wall_standard_normal_batch(
        0,
        particle_id,
        event_ordinal,
        WALL_MAXWELL_TANGENTIAL_STREAM,
    )[0]
    expected_velocity = half_range_maxwell_flux_velocity(
        300.0,
        float(scheduled_case.data.sources[0].mass_kg[0]),
        0.0,
        0.0,
        1.0,
        0.0,
        float(normal_draw),
        float(tangent_draw),
    )
    np.testing.assert_array_equal(event.velocity_post_m_s[0], expected_velocity)
    expected_final_position = event.position_m[0] + 0.5 * np.asarray(expected_velocity)
    np.testing.assert_allclose(
        scheduled.read_final().position_m[0],
        expected_final_position,
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_array_equal(
        scheduled.read_final().velocity_m_s,
        disabled.read_final().velocity_m_s,
    )
    assert scheduled.manifest["boundary_algorithm_revision"] == "contact_wall_laws_v7"
    assert scheduled.manifest["rng_algorithm_revision"] == "philox4x32_10_v2"
    assert scheduled.manifest["random_draw_kinds"]["wall_maxwell_normal"] == (
        WALL_MAXWELL_NORMAL_STREAM
    )


def test_hold_retains_impact_payload_and_is_independent_of_trajectory_output(
    tmp_path: Path,
) -> None:
    scheduled_paths = materialize_microcase("C07", tmp_path / "scheduled-hold-case")
    disabled_paths = materialize_microcase("C07", tmp_path / "disabled-hold-case")
    _set_terminal_case(scheduled_paths.case_path, law="hold", frame_times=[1.0, 1.5, 2.0])
    _set_terminal_case(disabled_paths.case_path, law="hold", frame_times=None)

    scheduled_output = tmp_path / "scheduled-hold-result"
    disabled_output = tmp_path / "disabled-hold-result"
    simulate(load_case(scheduled_paths.case_path), scheduled_output)
    simulate(load_case(disabled_paths.case_path), disabled_output)
    scheduled = open_result(scheduled_output)
    disabled = open_result(disabled_output)

    frames = list(scheduled.iter_frames())
    assert [frame.time_s for frame in frames] == [1.0, 1.5, 2.0]
    np.testing.assert_array_equal(frames[0].position_m, [[0.75, 0.5]])
    np.testing.assert_array_equal(frames[0].velocity_m_s, [[0.5, 0.25]])
    np.testing.assert_array_equal(frames[0].lifecycle, [1])
    for frame in frames[1:]:
        np.testing.assert_array_equal(frame.particle_id, [701])
        np.testing.assert_array_equal(frame.position_m, [[1.0, 0.625]])
        np.testing.assert_array_equal(frame.velocity_m_s, [[0.5, 0.25]])
        np.testing.assert_array_equal(frame.charge_number, [3.0])
        np.testing.assert_array_equal(frame.lifecycle, [5])
    assert list(disabled.iter_frames()) == []

    scheduled_event = scheduled.read_boundary_events()
    disabled_event = disabled.read_boundary_events()
    for name in (
        "time_s",
        "particle_id",
        "event_ordinal",
        "primary_facet_id",
        "position_m",
        "normal",
        "velocity_pre_m_s",
        "velocity_post_m_s",
        "charge_number_pre",
        "charge_number_post",
        "model_weight",
        "law_id",
        "outcome",
        "candidate_offset",
        "candidate_facet_id",
    ):
        np.testing.assert_array_equal(getattr(scheduled_event, name), getattr(disabled_event, name))
    np.testing.assert_array_equal(scheduled_event.time_s, [1.5])
    np.testing.assert_array_equal(scheduled_event.position_m, [[1.0, 0.625]])
    np.testing.assert_array_equal(scheduled_event.velocity_pre_m_s, [[0.5, 0.25]])
    np.testing.assert_array_equal(scheduled_event.velocity_post_m_s, [[0.5, 0.25]])
    np.testing.assert_array_equal(scheduled_event.charge_number_pre, [3.0])
    np.testing.assert_array_equal(scheduled_event.charge_number_post, [3.0])
    np.testing.assert_array_equal(scheduled_event.model_weight, [1.0])
    np.testing.assert_array_equal(scheduled_event.law_id, ["hold"])
    np.testing.assert_array_equal(scheduled_event.outcome, ["held"])

    scheduled_final = scheduled.read_final()
    disabled_final = disabled.read_final()
    for name in (
        "particle_id",
        "time_s",
        "position_m",
        "velocity_m_s",
        "charge_number",
        "lifecycle",
        "kinematics_valid",
    ):
        np.testing.assert_array_equal(getattr(scheduled_final, name), getattr(disabled_final, name))
    np.testing.assert_array_equal(scheduled_final.time_s, [2.0])
    np.testing.assert_array_equal(scheduled_final.position_m, [[1.0, 0.625]])
    np.testing.assert_array_equal(scheduled_final.velocity_m_s, [[0.5, 0.25]])
    np.testing.assert_array_equal(scheduled_final.charge_number, [3.0])
    np.testing.assert_array_equal(scheduled_final.lifecycle, [5])
    np.testing.assert_array_equal(scheduled_final.kinematics_valid, [1])

    series = scheduled.read_lifecycle_series()
    np.testing.assert_array_equal(series.held[-1:], [1])
    np.testing.assert_array_equal(series.stuck[-1:], [0])
    np.testing.assert_array_equal(series.escaped[-1:], [0])
    np.testing.assert_array_equal(series.failed[-1:], [0])
    assert scheduled.manifest["lifecycle_counts"] == {
        "pending": 0,
        "active": 0,
        "stuck": 0,
        "held": 1,
        "escaped": 0,
        "failed": 0,
    }


@pytest.mark.parametrize(
    ("positions_m", "expected_particle_id"),
    (
        ([[1.0, 0.5], [1.1, 0.5]], 901),
        ([[0.5, 0.5], [1.1, 0.5]], 902),
    ),
)
def test_table_start_validation_preserves_row_order_and_particle_id(
    tmp_path: Path,
    positions_m: list[list[float]],
    expected_particle_id: int,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "invalid-table-start")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    source = replace(
        original,
        particle_id=np.asarray([901, 902], dtype="<i8"),
        release_time_s=np.repeat(original.release_time_s, 2),
        position_m=np.asarray(positions_m, dtype="<f8"),
        velocity_m_s=np.repeat(original.velocity_m_s, 2, axis=0),
        charge_number=np.repeat(original.charge_number, 2),
        mass_kg=np.repeat(original.mass_kg, 2),
        drag_diameter_m=np.repeat(original.drag_diameter_m, 2),
        electrostatic_radius_m=np.repeat(original.electrostatic_radius_m, 2),
        contact_radius_m=np.repeat(original.contact_radius_m, 2),
        displaced_volume_m3=np.repeat(original.displaced_volume_m3, 2),
        model_weight=np.repeat(original.model_weight, 2),
        material_id=np.repeat(original.material_id, 2),
    )
    data_path = paths.case_path.parent / "invalid-table-start.h5"
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        SimulationError,
        match=rf"P05 table particle {expected_particle_id} must start strictly inside",
    ):
        simulate(load_case(paths.case_path), tmp_path / "invalid-table-start-result")


def test_rz_omitted_axis_seam_orders_wall_hit_and_axis_fold(
    tmp_path: Path,
) -> None:
    wall_paths = materialize_microcase("C07", tmp_path / "rz-wall")
    _make_rz_axis_seam_case(
        wall_paths.case_path,
        position_m=[0.25, 0.5],
        velocity_m_s=[-1.0, 4.0],
    )
    wall_output = tmp_path / "rz-wall-result"

    simulate(load_case(wall_paths.case_path), wall_output)
    wall_result = open_result(wall_output)
    wall_event = wall_result.read_boundary_events()
    np.testing.assert_array_equal(wall_event.time_s, [0.125])
    np.testing.assert_array_equal(wall_event.position_m, [[0.125, 1.0]])
    np.testing.assert_array_equal(wall_event.primary_facet_id, [2])
    np.testing.assert_array_equal(wall_result.read_final().lifecycle, [2])

    axis_paths = materialize_microcase("C07", tmp_path / "rz-axis")
    _make_rz_axis_seam_case(
        axis_paths.case_path,
        position_m=[0.25, 0.5],
        velocity_m_s=[-1.0, 0.0],
    )
    axis_output = tmp_path / "rz-axis-result"
    simulate(load_case(axis_paths.case_path), axis_output)
    axis_result = open_result(axis_output)
    assert axis_result.read_boundary_events().particle_id.size == 0
    np.testing.assert_allclose(
        axis_result.read_final().position_m,
        [[0.25, 0.5]],
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_array_equal(axis_result.read_final().velocity_m_s, [[1.0, 0.0]])
    assert axis_result.manifest["boundary_interactions"]["axis_crossings"] == 1


def test_result_validation_rejects_corrupt_boundary_offsets_and_validity(
    tmp_path: Path,
) -> None:
    offset_paths = materialize_microcase("C07", tmp_path / "offset-case")
    offset_output = tmp_path / "offset-result"
    simulate(load_case(offset_paths.case_path), offset_output)
    with h5py.File(offset_output / "segments" / "epoch-000000.h5", "r+") as segment:
        segment["events/boundary/candidate_offset"][...] = np.asarray([0, 0], dtype="<i8")
    with pytest.raises(SimulationError, match="candidate offsets"):
        open_result(offset_output)

    validity_paths = materialize_microcase("C07", tmp_path / "validity-case")
    validity_output = tmp_path / "validity-result"
    simulate(load_case(validity_paths.case_path), validity_output)
    with h5py.File(validity_output / "final.h5", "r+") as final_file:
        final_file["particles/kinematics_valid"][0] = np.uint8(2)
    with pytest.raises(SimulationError, match="kinematics_valid"):
        open_result(validity_output)


def _set_terminal_case(path: Path, *, law: str, frame_times: list[float] | None) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["boundaries"][0]["law"] = law
    if frame_times is None:
        document["output"]["trajectories"] = None
    else:
        document["output"]["trajectories"] = {
            "selection": "all",
            "schedule": {"explicit_times_s": frame_times},
        }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _set_maxwell_case(path: Path, *, frame_times: list[float] | None) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["boundaries"][0] = {
        "boundary_group": "wall",
        "priority": 10,
        "law": "maxwell_thermal",
        "wall_temperature_K": 300.0,
        "diffuse_reflection_fraction": 1.0,
        "wall_velocity_m_s": [0.0, 0.0],
    }
    if frame_times is None:
        document["output"]["trajectories"] = None
    else:
        document["output"]["trajectories"] = {
            "selection": "all",
            "schedule": {"explicit_times_s": frame_times},
        }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _make_rz_axis_seam_case(
    path: Path,
    *,
    position_m: list[float],
    velocity_m_s: list[float],
) -> None:
    case = load_case(path)
    old_boundary = case.data.geometry.boundary
    boundary = BoundaryData(
        line2=old_boundary.line2[:3].copy(),
        boundary_id=old_boundary.boundary_id[:3].copy(),
        group_id=old_boundary.group_id[:3].copy(),
        material_id=old_boundary.material_id[:3].copy(),
        owner_cell_type=old_boundary.owner_cell_type[:3].copy(),
        owner_cell_local_index=old_boundary.owner_cell_local_index[:3].copy(),
        orientation=old_boundary.orientation[:3].copy(),
    )
    geometry = replace(case.data.geometry, boundary=boundary)
    source = replace(
        case.data.sources[0],
        position_m=np.asarray([position_m], dtype="<f8"),
        velocity_m_s=np.asarray([velocity_m_s], dtype="<f8"),
    )
    data_path = path.parent / "rz-case.h5"
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    document["time"]["end_s"] = 0.5
    document["time"]["dt_s"] = 0.5
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
