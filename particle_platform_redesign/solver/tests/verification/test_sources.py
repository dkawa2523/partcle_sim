from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import load_case
from chamber_particles.case_format import write
from chamber_particles.geometry import prepare_geometry
from chamber_particles.rng import SOURCE_POSITION_DRAW, source_uniform_open
from chamber_particles.sources import realize_sources
from tests.verification.microcases import materialize_microcase


def test_rz_revolved_area_surface_position_uses_radial_area_measure(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "surface-rz")
    case = load_case(paths.case_path)
    boundary = replace(
        case.data.geometry.boundary,
        group_id=np.asarray([0, 1, 1, 2], dtype="<i4"),
    )
    geometry_data = replace(
        case.data.geometry,
        nodes_m=case.data.geometry.nodes_m + np.asarray([1.0, 0.0]),
        boundary=boundary,
    )
    data_path = paths.case_path.parent / "surface-rz.h5"
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry_data,
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    source = document["sources"][0]
    source["boundary_group"] = "caps"
    source["position"] = {"model": "uniform", "measure": "revolved_area"}
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    loaded = load_case(paths.case_path)
    geometry = prepare_geometry(loaded.data.geometry, loaded.data.coordinate_system)
    schedule = realize_sources(loaded, geometry)
    uniform = source_uniform_open(0, 0, 0, SOURCE_POSITION_DRAW)
    radius0 = float(geometry.facet_start_m[0, 0])
    radius1 = float(geometry.facet_end_m[0, 0])
    radial = math.sqrt((1.0 - uniform) * radius0**2 + uniform * radius1**2)
    expected_parameter = uniform * (radius0 + radius1) / (radius0 + radial)
    expected_position = geometry.facet_start_m[0] + expected_parameter * (
        geometry.facet_end_m[0] - geometry.facet_start_m[0]
    )

    np.testing.assert_array_equal(schedule.source_facet_id, [0])
    np.testing.assert_allclose(schedule.position_m[0], expected_position, rtol=0.0, atol=0.0)
    assert 0.0 < expected_parameter < 1.0


def test_surface_position_that_rounds_to_a_vertex_fails_closed(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "large-offset")
    case = load_case(paths.case_path)
    nodes = case.data.geometry.nodes_m.copy()
    nodes[:, 0] += 2.0**52
    boundary = replace(
        case.data.geometry.boundary,
        group_id=np.asarray([0, 1, 1, 2], dtype="<i4"),
    )
    data_path = paths.case_path.parent / "large-offset.h5"
    info = write(
        data_path,
        replace(
            case.data,
            geometry=replace(case.data.geometry, nodes_m=nodes, boundary=boundary),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    source = document["sources"][0]
    source["boundary_group"] = "caps"
    source["position"] = {"model": "uniform", "measure": "line_length"}
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    loaded = load_case(paths.case_path)
    geometry = prepare_geometry(loaded.data.geometry, loaded.data.coordinate_system)
    with pytest.raises(ValueError, match="rounded to a facet endpoint"):
        realize_sources(loaded, geometry)
