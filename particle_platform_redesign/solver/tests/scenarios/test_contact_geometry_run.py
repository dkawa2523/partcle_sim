from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

import chamber_particles.engine as engine_module
import chamber_particles.output as output_module
from chamber_particles import CaseError, SimulationError, load_case, open_result, simulate
from chamber_particles.case_format import (
    BoundaryData,
    RealizedSurfaceSource,
    RealizedTableSource,
    write,
)
from tests.scenarios.test_force_coupled_run import _harmonic_electric_case
from tests.scenarios.test_periodic_run import _fail_after_second_segment_replace
from tests.verification.microcases import materialize_microcase


def _mixed_case(
    directory: Path,
    *,
    mode: str = "particle_center",
    radius: tuple[float, ...] = (0.15,),
    width: float | None = None,
    velocity: tuple[float, float] = (1.0, 0.0),
    law: str = "escape",
    wall_law: str = "stick",
    frames: bool = True,
    memory_limit_mb: int = 128,
) -> Path:
    paths = materialize_microcase("C07", directory)
    case = load_case(paths.case_path)
    geometry = case.data.geometry
    nodes = geometry.nodes_m
    line2 = geometry.boundary.line2
    groups = np.asarray([0, 1, 0, 0], dtype="<i4")
    if width is not None:
        low, high = 0.5 - width / 2.0, 0.5 + width / 2.0
        nodes = np.asarray([[0, 0], [1, 0], [1, low], [1, high], [1, 1], [0, 1]], dtype="<f8")
        line2 = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], dtype="<i8")
        groups = np.asarray([0, 0, 1, 0, 0, 0], dtype="<i4")
        geometry = replace(
            geometry,
            quad4=None,
            quad4_domain_id=None,
            tri3=np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]], dtype="<i8"),
            tri3_domain_id=np.zeros(4, dtype="<i4"),
        )
    count = len(radius)
    source = case.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    source = replace(
        source,
        particle_id=np.arange(701, 701 + count, dtype="<i8"),
        position_m=np.tile(np.asarray([0.25, 0.5]), (count, 1)),
        velocity_m_s=np.tile(np.asarray(velocity), (count, 1)),
        contact_radius_m=np.asarray(radius, dtype="<f8"),
        **{
            name: np.repeat(getattr(source, name)[:1], count, axis=0)
            for name in (
                "release_time_s",
                "charge_number",
                "mass_kg",
                "drag_diameter_m",
                "electrostatic_radius_m",
                "displaced_volume_m3",
                "model_weight",
                "material_id",
            )
        },
    )
    boundary = BoundaryData(
        line2=line2,
        boundary_id=np.where(groups == 1, 20, 10).astype("<i4"),
        group_id=groups,
        material_id=np.zeros(groups.size, dtype="<i4"),
        owner_cell_type=np.full(groups.size, 2 if width is None else 1, dtype="u1"),
        owner_cell_local_index=(
            np.zeros(groups.size, dtype="<i8")
            if width is None
            else np.asarray([0, 0, 1, 2, 3, 3], dtype="<i8")
        ),
        orientation=np.ones(groups.size, dtype="i1"),
    )
    geometry = replace(geometry, nodes_m=nodes, group_names=("wall", "outlet"), boundary=boundary)
    data_path = directory / "mixed.h5"
    info = write(data_path, replace(case.data, geometry=geometry, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path=data_path.name, expected_content_hash=info.content_hash)
    document["time"].update(end_s=1.0, dt_s=1.0)
    document["resources"]["memory_limit_mb"] = memory_limit_mb
    document["boundaries"] = [
        {"boundary_group": "wall", "priority": 10, "law": wall_law},
        {"boundary_group": "outlet", "priority": 20, "law": law, "contact_geometry": mode},
    ]
    document["output"]["trajectories"] = (
        {"selection": "all", "schedule": {"explicit_times_s": [0.0, 0.8, 1.0]}} if frames else None
    )
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return paths.case_path


@pytest.mark.parametrize("mode", ["particle_surface", "particle_center"])
def test_contact_geometry_plane_has_independent_radius_and_hit_time(
    tmp_path: Path, mode: str
) -> None:
    radii = (0.05, 0.15)
    case_path = _mixed_case(tmp_path / mode, mode=mode, radius=radii)
    output = tmp_path / f"result-{mode}"
    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    order = np.argsort(event.particle_id)
    effective = np.asarray(radii) if mode == "particle_surface" else np.zeros(2)
    np.testing.assert_allclose(event.time_s[order], 0.75 - effective, rtol=0, atol=3e-12)
    np.testing.assert_allclose(event.position_m[order, 0], 1.0 - effective, rtol=0, atol=3e-12)
    np.testing.assert_array_equal(event.normal[order], [[1, 0], [1, 0]])
    np.testing.assert_array_equal(event.contact_radius_m[order], radii)
    np.testing.assert_array_equal(result.read_final().contact_radius_m, radii)
    assert result.manifest["counts"]["failure_events"] == 0
    resolved = {
        item["group"]: item["contact_geometry"]
        for item in result.manifest["resolved"]["boundary_laws"]
    }
    assert resolved == {"wall": "particle_surface", "outlet": mode}


@pytest.mark.parametrize("mode", ["particle_center", "particle_surface"])
def test_rz_contact_geometry_preserves_radial_chart_and_radius(tmp_path: Path, mode: str) -> None:
    case_path = _mixed_case(tmp_path / "rz", mode=mode)
    case = load_case(case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    shift = np.asarray([1.0, 0.0])
    data_path = case_path.with_name("rz.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=replace(case.data.geometry, nodes_m=case.data.geometry.nodes_m + shift),
            sources=(replace(source, position_m=source.position_m + shift),),
        ),
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path=data_path.name, expected_content_hash=info.content_hash)
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "result"
    simulate(load_case(case_path), output)
    result = open_result(output)
    expected_radius = 2.0 if mode == "particle_center" else 1.85
    event = result.read_boundary_events()
    np.testing.assert_allclose(event.position_m, [[expected_radius, 0.5]], rtol=0, atol=3e-12)
    np.testing.assert_allclose(event.time_s, [expected_radius - 1.25], rtol=0, atol=3e-12)
    np.testing.assert_array_equal(event.contact_radius_m, [0.15])
    assert result.manifest["counts"]["failure_events"] == 0


@pytest.mark.parametrize("width,escaped", [(0.4, True), (0.2, False)])
def test_virtual_opening_retains_material_endcap_first_contact(
    tmp_path: Path, width: float, escaped: bool
) -> None:
    radius = 0.15
    case_path = _mixed_case(tmp_path / str(width), width=width)
    output = tmp_path / f"result-{width}"
    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    assert event.particle_id.size == 1
    if escaped:
        np.testing.assert_allclose(event.time_s, [0.75], rtol=0, atol=3e-12)
        np.testing.assert_array_equal(event.outcome, ["escaped"])
        np.testing.assert_array_equal(event.candidate_facet_id, [2])
    else:
        separation_x = math.sqrt(radius**2 - (width / 2.0) ** 2)
        np.testing.assert_allclose(event.time_s, [0.75 - separation_x], rtol=0, atol=3e-12)
        np.testing.assert_array_equal(event.outcome, ["stuck"])
        np.testing.assert_array_equal(event.candidate_facet_id, [1, 3])
        np.testing.assert_allclose(event.normal[:, 0], [separation_x / radius], rtol=0, atol=3e-12)
        np.testing.assert_allclose(
            np.abs(event.normal[:, 1]), [width / (2 * radius)], rtol=0, atol=3e-12
        )
    assert result.manifest["counts"]["failure_events"] == 0


@pytest.mark.parametrize("integrator", ["rk4_fixed", "exponential_midpoint"])
def test_curved_hit_uses_center_mode_with_positive_body_radius(
    tmp_path: Path, integrator: str
) -> None:
    case_path = _harmonic_electric_case(
        tmp_path / integrator,
        step_s=0.03125,
        end_time_s=0.5,
        angular_frequency_s_inv=2.0,
        initial_position_m=np.asarray([0.5, 0.5]),
        initial_velocity_m_s=np.asarray([2.0, 0.0]),
        frame_times=None,
        material_wall=True,
        contact_radius_m=0.2,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["solver"]["integrator"] = integrator
    document["boundaries"][0]["contact_geometry"] = "particle_center"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / f"result-{integrator}"
    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    # x(t)=0.5 cos(2t)+sin(2t); the first ascending x=1 root.
    expected = (math.asin(1 / math.sqrt(1.25)) - math.atan(0.5)) / 2.0
    np.testing.assert_allclose(event.position_m, [[1.0, 0.5]], rtol=0, atol=3e-10)
    np.testing.assert_allclose(event.time_s, [expected], rtol=0, atol=4e-4)
    np.testing.assert_array_equal(event.contact_radius_m, [0.2])
    assert result.manifest["counts"]["failure_events"] == 0


def test_contact_geometry_does_not_depend_on_output_or_slab(tmp_path: Path) -> None:
    events = []
    finals = []
    plans = []
    for index, (frames, memory) in enumerate(((True, 5), (False, 64))):
        case_path = _mixed_case(
            tmp_path / f"case-{index}",
            radius=(0.05, 0.15) * 512,
            frames=frames,
            memory_limit_mb=memory,
        )
        output = tmp_path / f"result-{index}"
        simulate(load_case(case_path), output)
        result = open_result(output)
        events.append(result.read_boundary_events())
        finals.append(result.read_final())
        plans.append(result.manifest["memory_plan"])
    assert plans[0]["slab_particles"] < plans[1]["slab_particles"]
    for column in (
        "time_s",
        "particle_id",
        "event_ordinal",
        "position_m",
        "normal",
        "candidate_facet_id",
        "outcome",
    ):
        np.testing.assert_array_equal(getattr(events[0], column), getattr(events[1], column))
    for column in ("position_m", "velocity_m_s", "lifecycle", "contact_radius_m"):
        np.testing.assert_array_equal(getattr(finals[0], column), getattr(finals[1], column))


def test_center_surface_source_retains_radius_without_offset_or_nudge(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "surface")
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedSurfaceSource)
    data_path = paths.case_path.with_name("positive.h5")
    info = write(
        data_path,
        replace(case.data, sources=(replace(source, contact_radius_m=np.asarray([0.15])),)),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path=data_path.name, expected_content_hash=info.content_hash)
    for item in document["boundaries"]:
        item["contact_geometry"] = "particle_center"
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0.0]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "surface-result"
    simulate(load_case(paths.case_path), output)
    result = open_result(output)
    np.testing.assert_array_equal(next(result.iter_frames()).position_m, [[0, 0.5]])
    np.testing.assert_allclose(result.read_boundary_events().time_s, [1, 2], rtol=0, atol=3e-12)
    np.testing.assert_array_equal(result.read_final().contact_radius_m, [0.15])
    assert result.manifest["counts"]["failure_events"] == 0


@pytest.mark.parametrize("macro_duration_s", [1.4, 0.2])
def test_center_surface_source_retains_material_reflection_origin_across_macros(
    tmp_path: Path, macro_duration_s: float
) -> None:
    paths = materialize_microcase("C08", tmp_path / "center-source")
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedSurfaceSource)
    radius = 0.15
    data_path = paths.case_path.with_name("positive.h5")
    info = write(
        data_path,
        replace(case.data, sources=(replace(source, contact_radius_m=np.asarray([radius])),)),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path=data_path.name, expected_content_hash=info.content_hash)
    document["time"].update(end_s=1.4, dt_s=macro_duration_s)
    for item in document["boundaries"]:
        item["contact_geometry"] = (
            "particle_center" if item["boundary_group"] == "collector" else "particle_surface"
        )
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0.0, 1.0, 1.4]},
    }
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "center-source-result"
    simulate(load_case(paths.case_path), output)
    result = open_result(output)
    frames = list(result.iter_frames())
    np.testing.assert_array_equal(frames[0].position_m, [[0, 0.5]])
    hit_x = 1.0 - radius
    event = result.read_boundary_events()
    np.testing.assert_allclose(event.time_s, [hit_x], rtol=0, atol=3e-12)
    np.testing.assert_allclose(event.position_m, [[hit_x, 0.5]], rtol=0, atol=3e-12)
    np.testing.assert_array_equal(event.primary_facet_id, [1])
    np.testing.assert_array_equal(event.contact_radius_m, [radius])
    np.testing.assert_array_equal(event.velocity_post_m_s, [[-1.0, 0.0]])
    for frame, time in zip(frames[1:], (1.0, 1.4), strict=True):
        np.testing.assert_allclose(frame.position_m, [[2 * hit_x - time, 0.5]], rtol=0, atol=3e-12)
    final = result.read_final()
    np.testing.assert_allclose(final.position_m, [[2 * hit_x - 1.4, 0.5]], rtol=0, atol=3e-12)
    np.testing.assert_array_equal(final.velocity_m_s, [[-1.0, 0.0]])
    np.testing.assert_array_equal(final.contact_radius_m, [radius])
    np.testing.assert_array_equal(final.lifecycle, [1])
    assert result.manifest["counts"]["failure_events"] == 0


def test_contact_geometry_checkpoint_resume_preserves_full_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_path = _mixed_case(tmp_path / "case", width=0.2, wall_law="specular")
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["time"]["dt_s"] = 0.2
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    case = load_case(case_path)
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 1)
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_WORK_PER_PARTICLE", 0)
    reference_path, resumed_path = tmp_path / "reference", tmp_path / "resumed"
    simulate(case, reference_path)
    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", _fail_after_second_segment_replace())
        with pytest.raises(SimulationError):
            simulate(case, resumed_path)
    simulate(case, resumed_path)
    reference, resumed = open_result(reference_path), open_result(resumed_path)
    for reader in ("read_final", "read_boundary_events", "read_failure_events"):
        expected, actual = getattr(reference, reader)(), getattr(resumed, reader)()
        for field in dataclass_fields(expected):
            np.testing.assert_array_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )
    assert (
        resumed.manifest["resolved"]["boundary_laws"]
        == (reference.manifest["resolved"]["boundary_laws"])
    )
    assert resumed.manifest["counts"]["failure_events"] == 0


def test_unknown_contact_geometry_is_rejected_by_public_loader(tmp_path: Path) -> None:
    case_path = _mixed_case(tmp_path / "invalid", mode="automatic_from_law")
    with pytest.raises(CaseError, match="contact_geometry"):
        load_case(case_path)


def test_material_caps_combine_normal_and_recompute_reflected_residual(tmp_path: Path) -> None:
    case_path = _mixed_case(tmp_path / "reflecting-caps", width=0.2, wall_law="specular")
    output = tmp_path / "reflecting-caps-result"
    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    hit_x = 1 - math.sqrt(0.15**2 - 0.1**2)
    np.testing.assert_allclose(event.time_s, [hit_x - 0.25], rtol=0, atol=3e-12)
    np.testing.assert_allclose(event.normal, [[1, 0]], rtol=0, atol=3e-12)
    np.testing.assert_array_equal(event.candidate_facet_id, [1, 3])
    np.testing.assert_allclose(event.velocity_post_m_s, [[-1, 0]], rtol=0, atol=3e-12)
    np.testing.assert_allclose(
        result.read_final().position_m, [[2 * hit_x - 1.25, 0.5]], rtol=0, atol=3e-12
    )
    assert result.manifest["counts"]["failure_events"] == 0


def test_simultaneous_surface_and_center_contact_fails_without_guessed_response(
    tmp_path: Path,
) -> None:
    case_path = _mixed_case(tmp_path / "simultaneous", velocity=(1.0, 7.0 / 15.0))
    output = tmp_path / "simultaneous-result"
    simulate(load_case(case_path), output)
    result = open_result(output)
    assert result.read_boundary_events().particle_id.size == 0
    np.testing.assert_array_equal(result.read_final().lifecycle, [4])
    failure = result.read_failure_events()
    assert failure.particle_id.size == 1
    np.testing.assert_array_equal(
        failure.reason_code, [result.manifest["failure_reason_codes"]["indeterminate_event"]]
    )


def test_mixed_exact_periodic_and_finite_reflections_preserve_slab_and_output_identity(
    tmp_path: Path,
) -> None:
    """Reused exact work must retain each row's origin, targets, and contact mode."""

    count = 1025
    paths = materialize_microcase("C07", tmp_path / "mixed-exact")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    assert isinstance(original, RealizedTableSource)
    kind = np.arange(count) % 5
    # Stationary finite bodies must remain active alongside moving point/finite paths.
    velocity = np.asarray(
        [[0.0, 0.0], [0.5, 0.125], [0.0, 0.6], [2.0, 0.6], [64.0, 0.25]],
        dtype="<f8",
    )[kind]
    release = np.resize(np.asarray([0.0, 0.071, 0.20708774409941713]), count)
    radii = np.resize(np.asarray([0.0, 0.05, 0.1]), count)
    # The last row has a zero-length root at the final macro boundary.
    release[-1] = 1.0
    velocity[-1] = 0.0
    source = replace(
        original,
        particle_id=np.arange(7001, 7001 + count, dtype="<i8"),
        release_time_s=release,
        position_m=np.tile(np.asarray([0.25, 0.5]), (count, 1)),
        velocity_m_s=velocity,
        contact_radius_m=radii,
        **{
            name: np.repeat(getattr(original, name)[:1], count, axis=0)
            for name in (
                "charge_number",
                "mass_kg",
                "drag_diameter_m",
                "electrostatic_radius_m",
                "displaced_volume_m3",
                "model_weight",
                "material_id",
            )
        },
    )
    geometry = replace(
        case.data.geometry,
        group_names=("bottom", "right", "top", "left"),
        boundary=replace(
            case.data.geometry.boundary,
            boundary_id=np.asarray([10, 20, 30, 40], dtype="<i4"),
            group_id=np.arange(4, dtype="<i4"),
        ),
    )
    data_path = paths.case_path.with_name("mixed-exact.h5")
    info = write(data_path, replace(case.data, geometry=geometry, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path=data_path.name, expected_content_hash=info.content_hash)
    document["time"].update(end_s=1.0, dt_s=1.0)
    document["solver"]["event"].update(max_interactions_per_step=1, max_refinements=3)
    document["boundaries"] = [
        {
            "boundary_group": group,
            "priority": priority,
            "law": "specular",
            "contact_geometry": "particle_surface",
        }
        for group, priority in (("bottom", 10), ("top", 20))
    ]
    document["topology"] = {
        "model": "translation_periodic_xy_v1",
        "field_match_rtol": 1.0e-12,
        "pairs": [
            {
                "first_boundary_group": "left",
                "second_boundary_group": "right",
                "first_to_second_m": [1.0, 0.0],
            }
        ],
    }
    frame_times = [0.0, 0.20708774409941713, 0.46, 1.0]
    results = []
    for label, budget_mb, frames in (
        ("tight", 5, True),
        ("loose", 64, True),
        ("no-frames", 64, False),
    ):
        document["resources"]["memory_limit_mb"] = budget_mb
        document["output"]["trajectories"] = (
            {"selection": "all", "schedule": {"explicit_times_s": frame_times}} if frames else None
        )
        variant = paths.case_path.with_name(f"mixed-exact-{label}.yaml")
        variant.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"mixed-exact-{label}-result"
        simulate(load_case(variant), output)
        results.append(open_result(output))

    plans = [result.manifest["memory_plan"] for result in results]
    assert plans[0]["slab_particles"] < plans[1]["slab_particles"]
    assert plans[1]["slab_particles"] == count
    for actual in results[1:]:
        for reader in ("read_final", "read_boundary_events", "read_failure_events"):
            expected_payload = getattr(results[0], reader)()
            actual_payload = getattr(actual, reader)()
            for field in dataclass_fields(expected_payload):
                np.testing.assert_array_equal(
                    getattr(actual_payload, field.name), getattr(expected_payload, field.name)
                )
    tight_frames, loose_frames = list(results[0].iter_frames()), list(results[1].iter_frames())
    assert len(tight_frames) == len(loose_frames) == len(frame_times)
    for expected_frame, actual_frame in zip(tight_frames, loose_frames, strict=True):
        for field in dataclass_fields(expected_frame):
            np.testing.assert_array_equal(
                getattr(actual_frame, field.name), getattr(expected_frame, field.name)
            )

    reference = results[0]
    final = reference.read_final()
    healthy = velocity[:, 0] < 64.0
    np.testing.assert_array_equal(final.lifecycle[healthy], np.ones(healthy.sum(), dtype="u1"))
    elapsed = 1.0 - release[healthy]
    expected_x = np.remainder(0.25 + velocity[healthy, 0] * elapsed, 1.0)
    gap = 1.0 - 2.0 * radii[healthy]
    unfolded_y = 0.5 - radii[healthy] + velocity[healthy, 1] * elapsed
    remainder = np.remainder(unfolded_y, 2.0 * gap)
    expected_y = radii[healthy] + np.minimum(remainder, 2.0 * gap - remainder)
    reflections = np.floor(unfolded_y / gap).astype(np.int64)
    expected_vy = velocity[healthy, 1] * np.where(reflections % 2 == 0, 1.0, -1.0)
    np.testing.assert_allclose(
        final.position_m[healthy], np.column_stack((expected_x, expected_y)), rtol=0, atol=3e-12
    )
    np.testing.assert_allclose(
        final.velocity_m_s[healthy],
        np.column_stack((velocity[healthy, 0], expected_vy)),
        rtol=0,
        atol=3e-12,
    )
    np.testing.assert_array_equal(final.contact_radius_m, radii)
    failure = reference.read_failure_events()
    assert 0 < failure.particle_id.size < count
    failed_rows = failure.particle_id - source.particle_id[0]
    np.testing.assert_array_equal(velocity[failed_rows, 0], np.full(failed_rows.size, 64.0))
    np.testing.assert_array_equal(
        failure.reason_code,
        np.full(
            failed_rows.size,
            reference.manifest["failure_reason_codes"]["numerical_event_budget"],
        ),
    )
    assert reference.manifest["boundary_interactions"]["residual_splits"] > 0

    events = reference.read_boundary_events()
    periodic = events.interaction_kind == "periodic_translation"
    wall = events.interaction_kind == "wall"
    assert bool(periodic.any() and wall.any())
    event_radii = radii[events.particle_id - source.particle_id[0]]
    np.testing.assert_array_equal(events.contact_radius_m, event_radii)
    # Periodic seams use the centre even when the physical body radius is positive.
    np.testing.assert_allclose(events.position_m[periodic, 0], 1.0, rtol=0, atol=3e-12)
    np.testing.assert_allclose(events.position_post_m[periodic, 0], 0.0, rtol=0, atol=3e-12)
    np.testing.assert_array_equal(events.primary_facet_id[periodic], 1)
    np.testing.assert_array_equal(events.destination_facet_id[periodic], 3)
    np.testing.assert_array_equal(events.law_id[periodic], "")
    np.testing.assert_array_equal(
        events.velocity_pre_m_s[periodic], events.velocity_post_m_s[periodic]
    )
    expected_wall_y = np.where(
        events.primary_facet_id[wall] == 0, event_radii[wall], 1.0 - event_radii[wall]
    )
    np.testing.assert_allclose(events.position_m[wall, 1], expected_wall_y, rtol=0, atol=3e-12)
    assert reference.manifest["boundary_interactions"]["wall_events"] == int(wall.sum())
