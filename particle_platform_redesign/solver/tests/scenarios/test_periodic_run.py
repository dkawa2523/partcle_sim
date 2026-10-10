from __future__ import annotations

import copy
import os
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import chamber_particles.engine as engine_module
import chamber_particles.output as output_module
from chamber_particles import SimulationError, load_case, open_result, simulate
from chamber_particles.case_format import BoundaryData, GeometryData, write
from tests.verification.microcases import build_microcase


def test_translation_periodic_multiple_wraps_preserve_state_and_record_transfer(
    tmp_path: Path,
) -> None:
    case = _materialize_periodic_case(
        tmp_path / "multiple-wrap",
        position_m=(0.25, 0.5),
        velocity_m_s=(3.0, 0.0),
        end_s=1.0,
        dt_s=1.0,
    )

    output = tmp_path / "multiple-wrap-result"
    simulate(case, output)
    result = open_result(output)
    events = result.read_boundary_events()
    final = result.read_final()

    np.testing.assert_allclose(final.position_m, [[0.25, 0.5]], rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(final.velocity_m_s, [[3.0, 0.0]])
    np.testing.assert_array_equal(events.event_ordinal, [1, 2, 3])
    np.testing.assert_array_equal(
        events.interaction_kind,
        ["periodic_translation", "periodic_translation", "periodic_translation"],
    )
    np.testing.assert_array_equal(events.outcome, ["transferred"] * 3)
    np.testing.assert_array_equal(events.law_id, [""] * 3)
    np.testing.assert_allclose(events.position_m, [[1.0, 0.5]] * 3, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        events.position_post_m,
        [[0.0, 0.5]] * 3,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_array_equal(events.primary_facet_id, [1, 1, 1])
    np.testing.assert_array_equal(events.destination_facet_id, [3, 3, 3])
    assert result.manifest["topology_algorithm_revision"] == "translation_periodic_xy_v1"
    assert result.manifest["boundary_interactions"]["wall_events"] == 0


def test_finite_radius_material_contact_precedes_later_periodic_center_crossing(
    tmp_path: Path,
) -> None:
    case = _materialize_periodic_case(
        tmp_path / "finite-ordering",
        position_m=(0.8, 0.2),
        velocity_m_s=(1.0, -1.0),
        contact_radius_m=0.1,
        end_s=0.4,
        dt_s=0.4,
    )

    output = tmp_path / "finite-ordering-result"
    simulate(case, output)
    result = open_result(output)
    events = result.read_boundary_events()
    final = result.read_final()

    np.testing.assert_array_equal(events.interaction_kind, ["wall"])
    np.testing.assert_array_equal(events.law_id, ["stick"])
    np.testing.assert_array_equal(events.outcome, ["stuck"])
    np.testing.assert_array_equal(events.primary_facet_id, [0])
    np.testing.assert_array_equal(events.destination_facet_id, [-1])
    np.testing.assert_allclose(events.time_s, [0.1], rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(events.position_m, [[0.9, 0.1]], rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(final.lifecycle, [2])


def test_same_translation_split_node_is_one_periodic_transfer(tmp_path: Path) -> None:
    case = _materialize_periodic_case(
        tmp_path / "split-node",
        position_m=(0.25, 0.5),
        velocity_m_s=(1.0, 0.0),
        end_s=0.8,
        dt_s=0.8,
        split_seams=True,
    )

    output = tmp_path / "split-node-result"
    simulate(case, output)
    result = open_result(output)
    events = result.read_boundary_events()

    np.testing.assert_array_equal(events.interaction_kind, ["periodic_translation"])
    np.testing.assert_array_equal(events.event_ordinal, [1])
    np.testing.assert_array_equal(events.candidate_offset, [0, 2])
    np.testing.assert_array_equal(events.candidate_facet_id, [1, 2])
    np.testing.assert_allclose(
        result.read_final().position_m,
        [[0.05, 0.5]],
        rtol=0.0,
        atol=2.0e-14,
    )


def test_mixed_material_periodic_corner_fails_only_that_particle(tmp_path: Path) -> None:
    case = _materialize_periodic_case(
        tmp_path / "mixed-corner",
        position_m=(0.5, 0.5),
        velocity_m_s=(1.0, -1.0),
        end_s=0.75,
        dt_s=0.75,
    )

    output = tmp_path / "mixed-corner-result"
    simulate(case, output)
    result = open_result(output)
    failure = result.read_failure_events()
    final = result.read_final()

    reason = result.manifest["failure_reason_codes"]["indeterminate_boundary_policy"]
    np.testing.assert_array_equal(failure.particle_id, [701])
    np.testing.assert_array_equal(failure.reason_code, [reason])
    np.testing.assert_array_equal(final.failure_reason_code, [reason])
    np.testing.assert_array_equal(final.lifecycle, [4])
    assert result.read_boundary_events().particle_id.size == 0


def test_different_periodic_translations_at_corner_fail_closed(tmp_path: Path) -> None:
    case = _materialize_periodic_case(
        tmp_path / "periodic-corner",
        position_m=(0.5, 0.5),
        velocity_m_s=(1.0, 1.0),
        end_s=0.75,
        dt_s=0.75,
        periodic_y=True,
    )

    output = tmp_path / "periodic-corner-result"
    simulate(case, output)
    result = open_result(output)
    reason = result.manifest["failure_reason_codes"]["indeterminate_boundary_policy"]

    np.testing.assert_array_equal(result.read_failure_events().reason_code, [reason])
    np.testing.assert_array_equal(result.read_final().failure_reason_code, [reason])
    assert result.read_boundary_events().particle_id.size == 0


def test_brownian_path_restarts_across_periodic_transfers(tmp_path: Path) -> None:
    case = _materialize_brownian_periodic_case(tmp_path / "brownian")

    output = tmp_path / "brownian-result"
    simulate(case, output)
    result = open_result(output)
    events = result.read_boundary_events()

    assert events.particle_id.size >= 1
    np.testing.assert_array_equal(
        events.interaction_kind,
        np.full(events.particle_id.size, "periodic_translation"),
    )
    assert result.read_failure_events().particle_id.size == 0
    np.testing.assert_array_equal(result.read_final().lifecycle, [1])
    assert result.manifest["boundary_interactions"]["wall_events"] == 0
    assert result.manifest["brownian_rng_revision"] is not None


def test_periodic_checkpoint_resume_is_bitwise_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _materialize_periodic_case(
        tmp_path / "checkpoint",
        position_m=(0.25, 0.5),
        velocity_m_s=(3.0, 0.0),
        end_s=1.0,
        dt_s=0.2,
    )
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 1)
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_WORK_PER_PARTICLE", 0)
    reference_path = tmp_path / "checkpoint-reference"
    resumed_path = tmp_path / "checkpoint-resumed"
    simulate(case, reference_path)

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", _fail_after_second_segment_replace())
        with pytest.raises(SimulationError):
            simulate(case, resumed_path)
    simulate(case, resumed_path)

    reference = open_result(reference_path)
    resumed = open_result(resumed_path)
    for reader in ("read_final", "read_boundary_events", "read_failure_events"):
        expected = getattr(reference, reader)()
        actual = getattr(resumed, reader)()
        for field in dataclass_fields(expected):
            np.testing.assert_array_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )
    assert resumed.manifest["resume_identity"]["topology_algorithm_revision"] == (
        "translation_periodic_xy_v1"
    )


def _materialize_periodic_case(
    directory: Path,
    *,
    position_m: tuple[float, float],
    velocity_m_s: tuple[float, float],
    end_s: float,
    dt_s: float,
    contact_radius_m: float = 0.0,
    split_seams: bool = False,
    periodic_y: bool = False,
) -> Any:
    definition = build_microcase("C07")
    geometry = _split_geometry() if split_seams else _square_geometry(definition.data.geometry)
    source = replace(
        definition.data.sources[0],
        position_m=np.asarray([position_m], dtype="<f8"),
        velocity_m_s=np.asarray([velocity_m_s], dtype="<f8"),
        contact_radius_m=np.asarray([contact_radius_m], dtype="<f8"),
    )
    data = replace(definition.data, geometry=geometry, sources=(source,))

    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "case.h5"
    info = write(data_path, data)
    document = copy.deepcopy(definition.spec)
    document["case"] = {
        "name": "translation periodic XY scenario",
        "data_path": data_path.name,
        "expected_content_hash": info.content_hash,
    }
    document["time"] = {"start_s": 0.0, "end_s": end_s, "dt_s": dt_s}
    document["solver"]["event"]["max_interactions_per_step"] = 16
    document["boundaries"] = (
        []
        if periodic_y
        else [
            {"boundary_group": "bottom", "priority": 10, "law": "stick"},
            {"boundary_group": "top", "priority": 20, "law": "stick"},
        ]
    )
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
    if periodic_y:
        document["topology"]["pairs"].append(
            {
                "first_boundary_group": "bottom",
                "second_boundary_group": "top",
                "first_to_second_m": [0.0, 1.0],
            }
        )
    document["output"] = {"trajectories": None}
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_case(case_path)


def _materialize_brownian_periodic_case(directory: Path) -> Any:
    fields = build_microcase("C02")
    walls = build_microcase("C07")
    geometry = _square_geometry(walls.data.geometry)
    source = replace(
        fields.data.sources[0],
        position_m=np.asarray([[0.25, 0.5]], dtype="<f8"),
        velocity_m_s=np.asarray([[20.0, 0.0]], dtype="<f8"),
    )
    layouts = tuple(
        replace(
            layout,
            axis0_m=np.asarray([0.0, 1.0], dtype="<f8"),
            axis1_m=np.asarray([0.0, 1.0], dtype="<f8"),
        )
        for layout in fields.data.layouts
    )
    data = replace(fields.data, geometry=geometry, layouts=layouts, sources=(source,))
    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "case.h5"
    info = write(data_path, data)
    document = copy.deepcopy(fields.spec)
    document["case"] = {
        "name": "Brownian translation periodic XY scenario",
        "data_path": data_path.name,
        "expected_content_hash": info.content_hash,
    }
    document["time"] = {"start_s": 0.0, "end_s": 0.1, "dt_s": 0.1}
    document["solver"]["integrator"] = "ou_langevin"
    document["solver"]["seed"] = 1741
    document["solver"]["event"]["max_interactions_per_step"] = 16
    document["physics"]["noise"] = {
        "model": "inertial_langevin_fdt",
        "revision": "inertial_langevin_fdt_epstein_linear_midpoint_2d_v2",
        "interval_tree_depth": 8,
    }
    document["boundaries"] = [
        {"boundary_group": "bottom", "priority": 10, "law": "stick"},
        {"boundary_group": "top", "priority": 20, "law": "stick"},
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
    document["output"] = {"trajectories": None}
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return load_case(case_path)


def _fail_after_second_segment_replace() -> Any:
    real_replace = os.replace

    def replace_and_fail(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        real_replace(source, destination)
        if destination_path.name == "epoch-000001.h5":
            raise OSError("injected periodic checkpoint interruption")

    return replace_and_fail


def _square_geometry(base: GeometryData) -> GeometryData:
    return replace(
        base,
        boundary=replace(
            base.boundary,
            boundary_id=np.asarray([10, 20, 30, 40], dtype="<i4"),
            group_id=np.asarray([0, 1, 2, 3], dtype="<i4"),
        ),
        group_names=("bottom", "right", "top", "left"),
    )


def _split_geometry() -> GeometryData:
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [1.0, 1.0], [0.0, 1.0]],
        dtype="<f8",
    )
    return GeometryData(
        nodes_m=nodes,
        boundary=BoundaryData(
            line2=np.asarray([[0, 1], [1, 2], [2, 4], [4, 5], [5, 3], [3, 0]], dtype="<i8"),
            boundary_id=np.asarray([10, 20, 21, 30, 40, 41], dtype="<i4"),
            group_id=np.asarray([0, 1, 1, 2, 3, 3], dtype="<i4"),
            material_id=np.zeros(6, dtype="<i4"),
            owner_cell_type=np.full(6, 2, dtype="<u1"),
            owner_cell_local_index=np.asarray([0, 0, 1, 1, 1, 0], dtype="<i8"),
            orientation=np.ones(6, dtype="<i1"),
        ),
        group_names=("bottom", "right", "top", "left"),
        quad4=np.asarray([[0, 1, 2, 3], [3, 2, 4, 5]], dtype="<i8"),
        quad4_domain_id=np.asarray([0, 0], dtype="<i4"),
    )
