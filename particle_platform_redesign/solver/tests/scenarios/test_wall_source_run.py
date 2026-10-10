from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import CaseError, load_case, open_result, simulate
from chamber_particles.case_format import RealizedSurfaceSource, write
from tests.verification.microcases import materialize_microcase


def test_c08_surface_departure_reflection_and_reimpact_use_one_event_path(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C08", tmp_path / "C08")
    output = tmp_path / "result"

    summary = simulate(load_case(paths.case_path), output)
    result = open_result(output)
    boundary = result.read_boundary_events()
    final = result.read_final()

    assert summary.boundary_event_count == 2
    np.testing.assert_array_equal(boundary.particle_id, [801, 801])
    np.testing.assert_array_equal(boundary.event_ordinal, [1, 2])
    np.testing.assert_array_equal(boundary.time_s, [1.0, 2.0])
    np.testing.assert_array_equal(boundary.primary_facet_id, [1, 3])
    np.testing.assert_array_equal(boundary.candidate_offset, [0, 1, 2])
    np.testing.assert_array_equal(boundary.candidate_facet_id, [1, 3])
    np.testing.assert_array_equal(boundary.law_id, ["specular", "stick"])
    np.testing.assert_array_equal(boundary.outcome, ["reflected", "stuck"])
    np.testing.assert_array_equal(boundary.velocity_post_m_s, [[-1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(final.position_m, [[0.0, 0.5]], rtol=0.0, atol=3.0e-16)
    np.testing.assert_array_equal(final.lifecycle, [2])
    assert result.manifest["source_algorithm_revision"] == (
        "realized_internal_surface_contact_schedule_v5"
    )
    assert result.manifest["rng_algorithm_revision"] == "philox4x32_10_v2"
    assert result.manifest["boundary_interactions"]["wall_events"] == 2
    memory_plan = result.manifest["memory_plan"]
    assert memory_plan["release_work_bytes_per_particle"] == 272
    assert memory_plan["components"]["slab_release_work"] == (
        memory_plan["slab_particles"] * memory_plan["release_work_bytes_per_particle"]
    )


def test_restitution_is_published_by_the_public_event_and_manifest(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "restitution")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    mirror = next(item for item in document["boundaries"] if item["boundary_group"] == "mirror")
    mirror.update(
        {
            "law": "restitution",
            "normal_restitution": 0.5,
            "tangential_restitution": 0.25,
        }
    )
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / "result"
    simulate(load_case(paths.case_path), output)
    result = open_result(output)
    boundary = result.read_boundary_events()

    np.testing.assert_array_equal(boundary.law_id, ["restitution"])
    np.testing.assert_array_equal(boundary.outcome, ["reflected"])
    np.testing.assert_array_equal(boundary.velocity_post_m_s, [[-0.5, 0.0]])
    resolved = next(
        item for item in result.manifest["resolved"]["boundary_laws"] if item["group"] == "mirror"
    )
    assert resolved == {
        "group": "mirror",
        "contact_geometry": "particle_surface",
        "priority": 20,
        "law": "restitution",
        "normal_restitution": 0.5,
        "tangential_restitution": 0.25,
        "stick_probability": None,
        "otherwise_law": None,
        "wall_temperature_K": None,
        "diffuse_reflection_fraction": None,
        "wall_velocity_m_s": None,
    }


def test_surface_release_at_run_end_has_a_right_continuous_frame(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "release-at-end")
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedSurfaceSource)
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    end_s = document["time"]["end_s"]
    data_path = paths.case_path.with_name("release-at-end.h5")
    info = write(
        data_path,
        replace(
            case.data,
            sources=(replace(source, release_time_s=np.asarray([end_s], dtype="<f8")),),
        ),
    )
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [end_s]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / "result"
    simulate(load_case(paths.case_path), output)
    result = open_result(output)
    frames = list(result.iter_frames())

    assert result.read_boundary_events().particle_id.size == 0
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0].particle_id, [801])
    np.testing.assert_array_equal(frames[0].position_m, [[0.0, 0.5]])
    np.testing.assert_array_equal(frames[0].velocity_m_s, [[1.0, 0.0]])
    np.testing.assert_array_equal(frames[0].lifecycle, [1])


def test_large_realized_surface_table_fails_input_memory_gate_before_output(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C08", tmp_path / "large-surface-count")
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedSurfaceSource)
    count = 20_000
    large_source = replace(
        source,
        particle_id=np.arange(801, 801 + count, dtype="<i8"),
        release_time_s=np.zeros(count, dtype="<f8"),
        facet_id=np.full(count, 3, dtype="<i8"),
        facet_parameter=np.full(count, 0.5, dtype="<f8"),
        velocity_m_s=np.broadcast_to(source.velocity_m_s, (count, 2)).copy(),
        charge_number=np.zeros(count, dtype="<f8"),
        mass_kg=np.full(count, source.mass_kg[0], dtype="<f8"),
        drag_diameter_m=np.full(count, source.drag_diameter_m[0], dtype="<f8"),
        electrostatic_radius_m=np.full(count, source.electrostatic_radius_m[0], dtype="<f8"),
        contact_radius_m=np.full(count, source.contact_radius_m[0], dtype="<f8"),
        displaced_volume_m3=np.zeros(count, dtype="<f8"),
        model_weight=np.ones(count, dtype="<f8"),
        material_id=np.zeros(count, dtype="<i4"),
    )
    data_path = paths.case_path.with_name("large-surface.h5")
    info = write(data_path, replace(case.data, sources=(large_source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["resources"]["memory_limit_mb"] = 1
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / "result"
    with pytest.raises(CaseError, match=r"canonical numeric arrays require"):
        load_case(paths.case_path)

    assert not output.exists()
    assert not output.with_name(f"{output.name}.partial").exists()


def test_reflection_restarts_from_canonical_wall_state_without_zero_time_rehit(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C08", tmp_path / "two-reflections")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    collector = next(
        item for item in document["boundaries"] if item["boundary_group"] == "collector"
    )
    collector["law"] = "specular"
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / "result"
    simulate(load_case(paths.case_path), output)
    result = open_result(output)
    boundary = result.read_boundary_events()

    np.testing.assert_array_equal(boundary.time_s, [1.0, 2.0])
    np.testing.assert_array_equal(boundary.position_m, [[1.0, 0.5], [0.0, 0.5]])
    np.testing.assert_allclose(result.read_final().position_m, [[0.25, 0.5]], atol=0.0)
    np.testing.assert_array_equal(result.read_final().velocity_m_s, [[1.0, 0.0]])


def test_c09_residual_subdivision_preserves_all_hits_and_fails_at_its_budget(
    tmp_path: Path,
) -> None:
    expected_times = np.asarray(
        [2.0**-11 + index * 2.0**-10 for index in range(5)],
        dtype=np.float64,
    )
    default_paths = materialize_microcase("C09", tmp_path / "default-case")
    split_paths = materialize_microcase("C09", tmp_path / "split-case")
    _set_event_limits(split_paths.case_path, interactions=1, refinements=2)

    default_output = tmp_path / "default-result"
    split_output = tmp_path / "split-result"
    simulate(load_case(default_paths.case_path), default_output)
    simulate(load_case(split_paths.case_path), split_output)
    default = open_result(default_output)
    split = open_result(split_output)

    np.testing.assert_array_equal(split.read_boundary_events().time_s, expected_times)
    for name in (
        "time_s",
        "particle_id",
        "event_ordinal",
        "primary_facet_id",
        "candidate_offset",
        "candidate_facet_id",
        "velocity_pre_m_s",
        "velocity_post_m_s",
    ):
        np.testing.assert_array_equal(
            getattr(default.read_boundary_events(), name),
            getattr(split.read_boundary_events(), name),
        )
    np.testing.assert_array_equal(default.read_final().position_m, split.read_final().position_m)
    np.testing.assert_array_equal(
        default.read_final().velocity_m_s, split.read_final().velocity_m_s
    )
    assert split.manifest["boundary_interactions"]["residual_splits"] == 3
    memory_plan = split.manifest["memory_plan"]
    assert memory_plan["event_work_bytes_per_particle"] == 24 * (2 + 1)
    assert memory_plan["release_work_bytes_per_particle"] == 64
    assert memory_plan["event_staging_capacity"] == (memory_plan["event_candidate_capacity"] // 2)
    assert memory_plan["slab_particles"] <= memory_plan["event_staging_capacity"]
    assert memory_plan["event_staging_bytes_per_row"] == 624
    assert memory_plan["event_staging_fixed_bytes"] == (
        2 * memory_plan["event_candidate_capacity"] * np.dtype("<i8").itemsize + 16
    )
    assert memory_plan["failure_staging_bytes_per_particle"] == 52
    assert (
        memory_plan["components"]["slab_event_work"]
        == memory_plan["slab_particles"] * memory_plan["event_work_bytes_per_particle"]
    )
    assert memory_plan["components"]["slab_release_work"] == (
        memory_plan["slab_particles"] * memory_plan["release_work_bytes_per_particle"]
    )
    assert memory_plan["components"]["slab_event_staging"] == (
        memory_plan["event_staging_fixed_bytes"]
        + memory_plan["slab_particles"] * memory_plan["event_staging_bytes_per_row"]
    )
    assert memory_plan["components"]["slab_failure_staging"] == (
        memory_plan["slab_particles"] * memory_plan["failure_staging_bytes_per_particle"]
    )
    assert memory_plan["components"]["geometry_query_scratch"] == (
        4 * memory_plan["event_candidate_capacity"] * np.dtype("<i8").itemsize
    )
    # One C09 particle produces five interactions, while only one output row
    # is solver-owned at once.  The completed payload therefore demonstrates
    # that event rows are flushed wave by wave rather than retained to slab end.
    assert split.read_boundary_events().particle_id.size == 5
    assert memory_plan["slab_particles"] == 1
    assert memory_plan["components"]["slab_event_staging"] == (
        memory_plan["event_staging_fixed_bytes"] + 624
    )

    failed_paths = materialize_microcase("C09", tmp_path / "failed-case")
    _set_event_limits(failed_paths.case_path, interactions=1, refinements=1)
    failed_output = tmp_path / "failed-result"
    failed_summary = simulate(load_case(failed_paths.case_path), failed_output)
    failed = open_result(failed_output)
    failed_final = failed.read_final()
    failed_event = failed.read_failure_events()

    assert failed.manifest["status"] == "complete"
    assert failed_summary.failure_event_count == 1
    np.testing.assert_array_equal(failed_final.lifecycle, [4])
    assert int(failed_final.failure_reason_code[0]) > 0
    reason_code = failed.manifest["failure_reason_codes"]["numerical_event_budget"]
    np.testing.assert_array_equal(failed_final.failure_reason_code, [reason_code])
    np.testing.assert_array_equal(failed_event.particle_id, [901])
    np.testing.assert_array_equal(failed_event.time_s, [expected_times[1]])
    np.testing.assert_array_equal(
        failed_event.reason_code,
        failed_final.failure_reason_code,
    )

    mixed_paths = materialize_microcase("C09", tmp_path / "mixed-case")
    _add_stationary_c09_particle(mixed_paths.case_path)
    _set_event_limits(mixed_paths.case_path, interactions=1, refinements=1)
    mixed_output = tmp_path / "mixed-result"
    mixed_case = load_case(mixed_paths.case_path)
    mixed_summary = simulate(mixed_case, mixed_output)
    mixed = open_result(mixed_output)
    mixed_final = mixed.read_final()
    mixed_failure = mixed.read_failure_events()

    assert mixed_summary.failure_event_count == 1
    np.testing.assert_array_equal(mixed.read_boundary_events().particle_id, [901, 901])
    np.testing.assert_array_equal(mixed_final.particle_id, [901, 902])
    np.testing.assert_array_equal(mixed_final.lifecycle, [4, 1])
    np.testing.assert_array_equal(mixed_failure.particle_id, [901])
    assert int(mixed_failure.reason_code[0]) > 0
    np.testing.assert_array_equal(
        mixed_final.failure_reason_code,
        [mixed_failure.reason_code[0], 0],
    )
    np.testing.assert_array_equal(mixed_failure.reason_code, failed_event.reason_code)

    lifecycle = mixed.read_lifecycle_series()
    assert lifecycle.time_s.size == mixed_summary.macro_step_count
    np.testing.assert_array_equal(lifecycle.time_s, [mixed_case.spec.time.end_s])
    np.testing.assert_array_equal(lifecycle.pending, [0])
    np.testing.assert_array_equal(lifecycle.active, [1])
    np.testing.assert_array_equal(lifecycle.stuck, [0])
    np.testing.assert_array_equal(lifecycle.held, [0])
    np.testing.assert_array_equal(lifecycle.escaped, [0])
    np.testing.assert_array_equal(lifecycle.failed, [1])
    population = (
        lifecycle.pending
        + lifecycle.active
        + lifecycle.stuck
        + lifecycle.held
        + lifecycle.escaped
        + lifecycle.failed
    )
    np.testing.assert_array_equal(population, [mixed_summary.particle_count])


def test_c10_corner_policy_and_probe_schedule_use_the_same_event_path(tmp_path: Path) -> None:
    paths = materialize_microcase("C10", tmp_path / "C10")
    output = tmp_path / "result"

    simulate(load_case(paths.case_path), output)
    result = open_result(output)
    boundary = result.read_boundary_events()
    final = result.read_final()

    np.testing.assert_array_equal(boundary.time_s, [0.75, 0.75])
    np.testing.assert_array_equal(boundary.primary_facet_id, [1, 4])
    np.testing.assert_array_equal(boundary.candidate_offset, [0, 2, 4])
    np.testing.assert_array_equal(boundary.candidate_facet_id, [1, 2, 4, 5])
    np.testing.assert_allclose(
        boundary.velocity_post_m_s[0],
        [1.0 / math.sqrt(10.0), -3.0 / math.sqrt(10.0)],
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_array_equal(boundary.law_id, ["specular", "stick"])
    np.testing.assert_array_equal(boundary.outcome, ["reflected", "stuck"])
    np.testing.assert_array_equal(final.lifecycle, [1, 2])

    probed_paths = materialize_microcase("C10", tmp_path / "C10-probed")
    document = yaml.safe_load(probed_paths.case_path.read_text(encoding="utf-8"))
    document["output"]["probes"] = {
        "particle_ids": [1001],
        "schedule": {"explicit_times_s": [0.25, 0.75, 1.0]},
    }
    probed_paths.case_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    probed_output = tmp_path / "probed-result"
    probed_summary = simulate(load_case(probed_paths.case_path), probed_output)
    probed = open_result(probed_output)
    probes = list(probed.iter_probes())

    assert probed_summary.probe_count == 3
    assert probed_summary.probe_row_count == 3
    base_plan = result.manifest["memory_plan"]
    probe_plan = probed.manifest["memory_plan"]
    assert isinstance(base_plan, dict)
    assert isinstance(probe_plan, dict)
    base_components = base_plan["components"]
    probe_components = probe_plan["components"]
    assert isinstance(base_components, dict)
    assert isinstance(probe_components, dict)
    assert probe_components["probe_index"] == np.dtype("<i8").itemsize
    assert probe_components["output_buffer"] > base_components["output_buffer"]
    assert probe_components["replay_work"] > base_components["replay_work"]
    assert probe_components["replay_work"] == 3 * 8 + 1 * 8 + 3 * 1 * 42
    assert probe_plan["planned_bytes"] <= probe_plan["limit_bytes"]
    assert [probe.time_s for probe in probes] == [0.25, 0.75, 1.0]
    for probe in probes:
        np.testing.assert_array_equal(probe.particle_id, [1001])
    np.testing.assert_array_equal(probes[0].position_m, [[1.0, 0.5]])
    np.testing.assert_array_equal(probes[0].velocity_m_s, [[0.0, 1.0]])
    np.testing.assert_array_equal(probes[1].position_m, [[1.0, 1.0]])
    np.testing.assert_allclose(
        probes[1].velocity_m_s,
        [[1.0 / math.sqrt(10.0), -3.0 / math.sqrt(10.0)]],
        rtol=0.0,
        atol=2.0e-16,
    )

    probed_boundary = probed.read_boundary_events()
    for name in (
        "time_s",
        "particle_id",
        "event_ordinal",
        "primary_facet_id",
        "position_m",
        "velocity_pre_m_s",
        "velocity_post_m_s",
        "law_id",
        "outcome",
        "candidate_offset",
        "candidate_facet_id",
    ):
        np.testing.assert_array_equal(getattr(probed_boundary, name), getattr(boundary, name))
    probed_final = probed.read_final()
    for name in (
        "particle_id",
        "position_m",
        "velocity_m_s",
        "charge_number",
        "lifecycle",
        "failure_reason_code",
    ):
        np.testing.assert_array_equal(getattr(probed_final, name), getattr(final, name))


@pytest.mark.parametrize(
    ("probability", "event_count", "outcomes", "final_position", "lifecycle"),
    [
        (0.0, 2, ["reflected", "stuck"], [0.0, 0.5], 2),
        (1.0, 1, ["stuck"], [1.0, 0.5], 2),
    ],
)
def test_probabilistic_stick_endpoints_select_exact_wall_outcome(
    tmp_path: Path,
    probability: float,
    event_count: int,
    outcomes: list[str],
    final_position: list[float],
    lifecycle: int,
) -> None:
    paths = materialize_microcase("C08", tmp_path / f"probability-{probability}")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    mirror = next(item for item in document["boundaries"] if item["boundary_group"] == "mirror")
    _set_probabilistic_boundary(mirror, probability)
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / f"probability-{probability}-result"
    simulate(load_case(paths.case_path), output)
    result = open_result(output)
    boundary = result.read_boundary_events()
    final = result.read_final()

    assert boundary.particle_id.size == event_count
    np.testing.assert_array_equal(boundary.outcome, outcomes)
    assert boundary.law_id[0] == "probabilistic_stick"
    np.testing.assert_allclose(final.position_m[0], final_position, rtol=0.0, atol=3.0e-16)
    assert int(final.lifecycle[0]) == lifecycle


def test_wall_rng_uses_the_physical_per_particle_event_ordinal(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "physical-ordinal")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    for boundary in document["boundaries"]:
        if boundary["boundary_group"] in {"mirror", "collector"}:
            _set_probabilistic_boundary(boundary, 0.5)
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    unscheduled_output = tmp_path / "unscheduled-result"
    simulate(load_case(paths.case_path), unscheduled_output)
    unscheduled = open_result(unscheduled_output)

    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [1.0, 1.5, 2.0]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    scheduled_output = tmp_path / "scheduled-result"
    simulate(load_case(paths.case_path), scheduled_output)
    scheduled = open_result(scheduled_output)
    boundary = scheduled.read_boundary_events()

    # For seed=0, particle=801, physical draws 0 and 1 are about 0.960 and 0.271.
    np.testing.assert_array_equal(boundary.outcome, ["reflected", "stuck"])
    np.testing.assert_array_equal(
        boundary.law_id,
        ["probabilistic_stick", "probabilistic_stick"],
    )
    np.testing.assert_array_equal(boundary.event_ordinal, [1, 2])
    np.testing.assert_array_equal(
        unscheduled.read_boundary_events().outcome,
        boundary.outcome,
    )
    np.testing.assert_array_equal(
        unscheduled.read_final().position_m,
        scheduled.read_final().position_m,
    )
    frames = list(scheduled.iter_frames())
    np.testing.assert_array_equal([frame.time_s for frame in frames], [1.0, 1.5, 2.0])
    np.testing.assert_array_equal(
        [frame.position_m[0] for frame in frames],
        [[1.0, 0.5], [0.5, 0.5], [0.0, 0.5]],
    )
    np.testing.assert_array_equal(
        [frame.velocity_m_s[0] for frame in frames],
        [[-1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]],
    )
    np.testing.assert_array_equal([frame.lifecycle[0] for frame in frames], [1, 1, 2])
    assert scheduled.manifest["random_draw_kinds"]["wall_probabilistic_stick"] == 0x57414C01


def _set_event_limits(path: Path, *, interactions: int, refinements: int) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["solver"]["event"]["max_interactions_per_step"] = interactions
    document["solver"]["event"]["max_refinements"] = refinements
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _add_stationary_c09_particle(path: Path) -> None:
    case = load_case(path)
    original = case.data.sources[0]
    source = replace(
        original,
        particle_id=np.concatenate((original.particle_id, np.asarray([902], dtype="<i8"))),
        release_time_s=np.concatenate((original.release_time_s, np.asarray([0.0], dtype="<f8"))),
        position_m=np.concatenate(
            (original.position_m, np.asarray([[2.0**-11, 0.0625]], dtype="<f8")),
            axis=0,
        ),
        velocity_m_s=np.concatenate(
            (original.velocity_m_s, np.zeros((1, 2), dtype="<f8")),
            axis=0,
        ),
        charge_number=np.concatenate((original.charge_number, original.charge_number[:1])),
        mass_kg=np.concatenate((original.mass_kg, original.mass_kg[:1])),
        drag_diameter_m=np.concatenate((original.drag_diameter_m, original.drag_diameter_m[:1])),
        electrostatic_radius_m=np.concatenate(
            (original.electrostatic_radius_m, original.electrostatic_radius_m[:1])
        ),
        contact_radius_m=np.concatenate((original.contact_radius_m, original.contact_radius_m[:1])),
        displaced_volume_m3=np.concatenate(
            (original.displaced_volume_m3, original.displaced_volume_m3[:1])
        ),
        model_weight=np.concatenate((original.model_weight, original.model_weight[:1])),
        material_id=np.concatenate((original.material_id, original.material_id[:1])),
    )
    data_path = path.with_name("mixed-case.h5")
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _set_probabilistic_boundary(boundary: dict[str, object], probability: float) -> None:
    boundary["law"] = "probabilistic_stick"
    boundary["probability"] = probability
    boundary["otherwise"] = {"law": "specular"}
