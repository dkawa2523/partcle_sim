from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import load_case
from chamber_particles.case_format import RealizedSurfaceSource, write
from chamber_particles.geometry import prepare_geometry
from chamber_particles.sources import realize_sources
from tests.verification.microcases import materialize_microcase


def test_realized_surface_rows_preserve_each_particle_fact_and_facet_position(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C08", tmp_path / "surface-rz")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    assert isinstance(original, RealizedSurfaceSource)
    surface = replace(
        original,
        particle_id=np.asarray([802, 801], dtype="<i8"),
        release_time_s=np.asarray([0.2, 0.1], dtype="<f8"),
        facet_id=np.asarray([0, 1], dtype="<i8"),
        facet_parameter=np.asarray([0.25, 0.75], dtype="<f8"),
        velocity_m_s=np.asarray([[2.0, 3.0], [-4.0, 5.0]], dtype="<f8"),
        charge_number=np.asarray([6.0, 7.0], dtype="<f8"),
        mass_kg=np.asarray([8.0, 9.0], dtype="<f8"),
        drag_diameter_m=np.asarray([10.0, 11.0], dtype="<f8"),
        contact_radius_m=np.asarray([0.1, 0.2], dtype="<f8"),
        electrostatic_radius_m=np.asarray([12.0, 13.0], dtype="<f8"),
        displaced_volume_m3=np.asarray([14.0, 15.0], dtype="<f8"),
        model_weight=np.asarray([16.0, 17.0], dtype="<f8"),
        material_id=np.asarray([18, 19], dtype="<i4"),
    )
    data_path = paths.case_path.parent / "surface-rz.h5"
    info = write(data_path, replace(case.data, sources=(surface,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"]["end_s"] = 1.0
    document["time"]["dt_s"] = 1.0
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    loaded = load_case(paths.case_path)
    geometry = prepare_geometry(loaded.data.geometry, loaded.data.coordinate_system)
    schedule = realize_sources(loaded, geometry)

    np.testing.assert_array_equal(schedule.particle_id, [801, 802])
    np.testing.assert_array_equal(schedule.release_order, [0, 1])
    np.testing.assert_array_equal(schedule.source_facet_id, [1, 0])
    expected = np.asarray([[0.8, 0.75], [0.25, 0.1]], dtype="<f8")
    np.testing.assert_array_equal(schedule.position_m, expected)
    np.testing.assert_array_equal(schedule.contact_radius_m, [0.2, 0.1])
    np.testing.assert_array_equal(schedule.velocity_m_s, [[-4.0, 5.0], [2.0, 3.0]])
    np.testing.assert_array_equal(schedule.charge_number, [7.0, 6.0])
    np.testing.assert_array_equal(schedule.mass_kg, [9.0, 8.0])
    np.testing.assert_array_equal(schedule.material_id, [19, 18])


def test_surface_position_that_rounds_to_a_vertex_fails_closed(tmp_path: Path) -> None:
    paths = materialize_microcase("C08", tmp_path / "large-offset")
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedSurfaceSource)
    nodes = case.data.geometry.nodes_m.copy()
    nodes[:, 0] += 2.0**52
    data_path = paths.case_path.parent / "large-offset.h5"
    info = write(
        data_path,
        replace(
            case.data,
            geometry=replace(case.data.geometry, nodes_m=nodes),
            sources=(replace(source, facet_id=np.asarray([0], dtype="<i8")),),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    loaded = load_case(paths.case_path)
    geometry = prepare_geometry(loaded.data.geometry, loaded.data.coordinate_system)
    with pytest.raises(ValueError, match="rounded to a facet endpoint"):
        realize_sources(loaded, geometry)
