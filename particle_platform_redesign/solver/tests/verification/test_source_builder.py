"""Independent source conversion expectations and canonical/public round trips."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from tools.source_builder import convert_velocity, prepare_case, read_csv_source

from chamber_particles import load_case
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    GeometryData,
    RealizedTableSource,
    write,
)


def _source() -> RealizedTableSource:
    return RealizedTableSource(
        name="particles",
        particle_id=np.array([4, 9], dtype="<i8"),
        release_time_s=np.array([1.0, 1.1]),
        position_m=np.array([[0.25, 0.3], [0.6, 0.7]]),
        velocity_m_s=np.zeros((2, 2)),
        charge_number=np.array([-1.0, 2.0]),
        mass_kg=np.array([2.0, 8.0]),
        drag_diameter_m=np.array([1e-7, 2e-7]),
        contact_radius_m=np.array([0.0, 1e-8]),
        electrostatic_radius_m=np.array([4e-8, 8e-8]),
        displaced_volume_m3=np.array([1e-22, 8e-22]),
        model_weight=np.array([3.0, 7.0]),
        material_id=np.array([2, 3], dtype="<i4"),
    )


def _bundle() -> DataBundle:
    empty = BoundaryData(
        line2=np.empty((0, 2), dtype="<i8"),
        boundary_id=np.empty(0, dtype="<i4"),
        group_id=np.empty(0, dtype="<i4"),
        material_id=np.empty(0, dtype="<i4"),
        owner_cell_type=np.empty(0, dtype="u1"),
        owner_cell_local_index=np.empty(0, dtype="<i8"),
        orientation=np.empty(0, dtype="i1"),
    )
    geometry = GeometryData(
        nodes_m=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        boundary=empty,
        group_names=(),
        quad4=np.array([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.array([0], dtype="<i4"),
    )
    provenance = json.dumps(
        {
            "producer": "analytic-source-test",
            "producer_version": "1",
            "source_sha256": "sha256:" + "0" * 64,
            "field_semantics_revision": "no_fields_v1",
            "producer_metadata": {},
        }
    )
    return DataBundle("cartesian_xy", provenance, geometry, sources=(_source(),))


def _config() -> dict:
    return {
        "format_version": 1,
        "table": "particles",
        "columns": {"particle_id": "id", "release_time_s": "t", "position_m": ["x", "y"]},
        "units": {"release_time_s": "ms", "position_m": "cm"},
        "velocity_mode": "vector",
        "velocity_columns": ["vx", "vy"],
        "velocity_unit": "cm/s",
        "components": ["x", "y"],
        "basis_to_case": [[1, 0], [0, 1]],
        "frame": "inertial",
        "metadata": {"scope": "analytic realized rows", "unresolved": []},
    }


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("mode", "values", "unit", "direction", "expected"),
    [
        ("vector", [[300, -400], [-100, 200]], "cm/s", None, [[3, -4], [-1, 2]]),
        ("momentum", [[6, -8], [-8, 16]], "kg*m/s", None, [[3, -4], [-1, 2]]),
        ("speed_direction", [5, 5], "m/s", [[3, -4], [-3, 4]], [[3, -4], [-3, 4]]),
        ("energy_direction", [25, 100], "J", [[3, -4], [-3, 4]], [[3, -4], [-3, 4]]),
    ],
)
def test_independent_mass_vector_energy_expectations(
    mode, values, unit, direction, expected
) -> None:
    result = convert_velocity(
        values,
        [2, 8],
        mode=mode,
        unit=unit,
        components=("x", "y"),
        coordinate_system="cartesian_xy",
        basis_to_case=np.eye(2),
        frame="inertial",
        direction=direction,
    )
    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=0)


def test_ev_and_per_row_oriented_basis() -> None:
    # 1 eV at mass e/2 gives speed 2 m/s; fourfold mass gives half the speed.
    e = 1.602176634e-19
    result = convert_velocity(
        [1, 1],
        [e / 2, 2 * e],
        mode="energy_direction",
        unit="eV",
        components=("r", "z"),
        coordinate_system="axisymmetric_rz",
        basis_to_case=[[[0, -1], [1, 0]], [[1, 0], [0, 1]]],
        frame="inertial",
        direction=[[1, 0], [1, 0]],
    )
    np.testing.assert_array_equal(result, [[0, 2], [1, 0]])


@pytest.mark.parametrize(
    "coordinate,components,values",
    [
        ("cartesian_xy", ("x", "y", "z"), [[1, 2, 1e-300]]),
        ("axisymmetric_rz", ("r", "phi", "z"), [[1, 1e-300, 2]]),
    ],
)
def test_nonzero_third_component_is_never_projected(coordinate, components, values) -> None:
    with pytest.raises(ValueError, match="third component"):
        convert_velocity(
            values,
            [1],
            mode="vector",
            unit="m/s",
            components=components,
            coordinate_system=coordinate,
            basis_to_case=np.eye(2),
            frame="inertial",
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"frame": "rotating"}, "frames"),
        ({"frame": "unknown"}, "frames"),
        ({"unit": "keV"}, "unit"),
        ({"basis_to_case": [[1, 1], [0, 1]]}, "orthonormal"),
        ({"basis_to_case": [[-1, 0], [0, 1]]}, "orthonormal"),
    ],
)
def test_unsupported_input_requires_resolution(change, match) -> None:
    options = {
        "mode": "vector",
        "unit": "m/s",
        "components": ("x", "y"),
        "coordinate_system": "cartesian_xy",
        "basis_to_case": np.eye(2),
        "frame": "inertial",
    }
    options.update(change)
    with pytest.raises(ValueError, match=match):
        convert_velocity([[1, 2]], [1], **options)


def test_zero_direction_rejected_and_extreme_direction_normalized() -> None:
    options = {
        "mode": "speed_direction",
        "unit": "m/s",
        "components": ("x", "y"),
        "coordinate_system": "cartesian_xy",
        "basis_to_case": np.eye(2),
        "frame": "inertial",
    }
    with pytest.raises(ValueError, match="zero direction"):
        convert_velocity([1], [1], direction=[[0, 0]], **options)
    np.testing.assert_allclose(
        convert_velocity([5], [1], direction=[[3e300, -4e300]], **options), [[3, -4]]
    )


def test_csv_preserves_joint_properties_by_id_and_filters_initial_rows(tmp_path: Path) -> None:
    path = tmp_path / "initial.csv"
    rows = [
        {"id": "9", "t": 1200, "x": 60, "y": 70, "vx": -100, "vy": 200, "cohort": "saved"},
        {"id": "4", "t": 1100, "x": 25, "y": 30, "vx": 300, "vy": -400, "cohort": "saved"},
        {"id": "9", "t": 1300, "x": 60, "y": 70, "vx": 0, "vy": 0, "cohort": "later"},
    ]
    _csv(path, rows)
    config = _config() | {"filter_equals": {"cohort": "saved"}}
    source = read_csv_source(path, _bundle(), config)
    np.testing.assert_array_equal(source.particle_id, [9, 4])
    np.testing.assert_array_equal(source.mass_kg, [8, 2])
    np.testing.assert_array_equal(source.model_weight, [7, 3])
    np.testing.assert_allclose(source.release_time_s, [1.2, 1.1])
    np.testing.assert_allclose(source.velocity_m_s, [[-1, 2], [3, -4]])
    with pytest.raises(ValueError, match="duplicate particle"):
        read_csv_source(path, _bundle(), _config())


def test_unknown_id_requires_all_properties_and_can_replace_population(tmp_path: Path) -> None:
    path = tmp_path / "new.csv"
    row = {"id": 81, "t": 1100, "x": 20, "y": 40, "vx": 100, "vy": -100}
    _csv(path, [row])
    with pytest.raises(ValueError, match="new particle IDs"):
        read_csv_source(path, _bundle(), _config())
    properties = {
        "mass_kg": 3,
        "drag_diameter_m": 4e-7,
        "contact_radius_m": 3e-8,
        "electrostatic_radius_m": 2e-8,
        "displaced_volume_m3": 2e-22,
        "charge_number": 5,
        "model_weight": 11,
        "material_id": 6,
    }
    _csv(path, [row | properties])
    config = _config()
    config["columns"].update({name: name for name in properties})
    config["units"].update(
        {
            "mass_kg": "kg",
            "drag_diameter_m": "m",
            "contact_radius_m": "m",
            "electrostatic_radius_m": "m",
            "displaced_volume_m3": "m3",
        }
    )
    source = read_csv_source(path, _bundle(), config)
    assert source.particle_id.tolist() == [81]
    assert source.mass_kg.tolist() == [3]
    assert source.contact_radius_m.tolist() == [3e-8]
    write(tmp_path / "new.h5", replace(_bundle(), sources=(source,)))


def test_public_case_roundtrip_records_frame_assumption_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    info = write(tmp_path / "template.h5", _bundle())
    document = {
        "format_version": 3,
        "case": {
            "name": "source_test",
            "data_path": "template.h5",
            "expected_content_hash": info.content_hash,
        },
        "motion": {"mode": "cartesian_xy"},
        "time": {"start_s": 1, "end_s": 2, "dt_s": 0.1},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 0,
            "event": {
                "geometry_rtol": 1e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": {"charge": {"model": "fixed"}},
        "sources": [{"name": "release", "type": "table", "table": "particles"}],
        "boundaries": [],
        "output": {"trajectories": {"selection": "all", "schedule": {"explicit_times_s": [1, 2]}}},
    }
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(document), encoding="utf-8")
    csv_path, config_path = tmp_path / "rows.csv", tmp_path / "config.json"
    _csv(csv_path, [{"id": 4, "t": 1200, "x": 25, "y": 30, "vx": 300, "vy": -400}])
    config = _config()
    config["metadata"] = {"scope": "same-component realization", "unresolved": ["native frame"]}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    receipt = prepare_case(template, csv_path, config_path, tmp_path / "output")
    loaded = load_case(receipt["case"])
    np.testing.assert_array_equal(loaded.data.sources[0].particle_id, [4])
    provenance = json.loads(loaded.data.provenance_json)
    assert provenance["producer_metadata"]["source_builder"]["configuration"]["metadata"][
        "unresolved"
    ] == ["native frame"]
    with pytest.raises(FileExistsError):
        prepare_case(template, csv_path, config_path, tmp_path / "output")

    # Conflicting velocity units would otherwise silently change every speed by100.
    ambiguous = json.dumps(config).replace(
        '"velocity_unit": "cm/s"', '"velocity_unit": "m/s", "velocity_unit": "cm/s"'
    )
    assert ambiguous != json.dumps(config)
    config_path.write_text(ambiguous, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        prepare_case(template, csv_path, config_path, tmp_path / "ambiguous")
    assert not (tmp_path / "ambiguous" / "case.h5").exists()
