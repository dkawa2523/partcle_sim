from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml
from tools.comsol_adapter import ADAPTER_REVISION, adapt_from_configuration

from chamber_particles.case_format import DataBundle, read


def _write_csv(path: Path, header: Sequence[object], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture(tmp_path: Path, *, axis_velocity: float = 1.0e-5) -> Path:
    _write_csv(
        tmp_path / "vertices.csv",
        ["vertex_id", "r_m", "z_m"],
        [[40, 1.0, 1.0], [10, 0.0, 0.0], [30, 0.0, 1.0], [20, 1.0, 0.0]],
    )
    _write_csv(
        tmp_path / "triangles.csv",
        ["triangle_id", "node1_id", "node2_id", "node3_id", "domain_id"],
        [],
    )
    _write_csv(
        tmp_path / "quads.csv",
        [
            "quadrilateral_id",
            "node1_id",
            "node2_id",
            "node3_id",
            "node4_id",
            "domain_id",
        ],
        [[7, 10, 20, 30, 40, 3]],
    )
    _write_csv(
        tmp_path / "boundaries.csv",
        [
            "boundary_edge_id",
            "node1_id",
            "node2_id",
            "mphtxt_boundary_index",
            "comsol_boundary_id",
        ],
        [
            [104, 30, 10, 0, 5],
            [101, 10, 20, 1, 6],
            [103, 40, 30, 2, 8],
            [102, 20, 40, 3, 7],
        ],
    )
    columns = [
        ("r_m", "r", "m"),
        ("z_m", "z", "m"),
        ("inside_model_domain", "1", "1"),
        ("gas_velocity_r_m_per_s", "ur", "m/s"),
        ("gas_velocity_z_m_per_s", "uz", "m/s"),
        ("gas_temperature_K", "T", "K"),
        ("gas_density_kg_per_m3", "rho", "kg/m^3"),
        ("dynamic_viscosity_Pa_s", "mu", "Pa*s"),
        ("gas_mean_free_path_m", "lambda", "m"),
    ]
    _write_csv(tmp_path / "columns.csv", ["column", "COMSOL_expression", "unit"], columns)
    field_rows = [
        [1.0, 1.0, 1, 0.4, 1.4, 340.0, 4.0, 4.0e-5, 4.0e-3],
        [0.0, 1.0 + 4.0e-16, 1, -axis_velocity, 1.3, 330.0, 3.0, 3.0e-5, 3.0e-3],
        [9.0, 9.0, 1, "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"],
        [1.0, 0.0, 1, 0.2, 1.2, 320.0, 2.0, 2.0e-5, 2.0e-3],
        [0.0, 0.0, 1, axis_velocity, 1.1, 310.0, 1.0, 1.0e-5, 1.0e-3],
    ]
    with (tmp_path / "fields.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("% synthetic provider table\n")
        writer = csv.writer(stream)
        writer.writerows(field_rows)
    configuration = {
        "format_version": 1,
        "source": {
            "vertices_path": "vertices.csv",
            "triangles_path": "triangles.csv",
            "quadrilaterals_path": "quads.csv",
            "boundary_edges_path": "boundaries.csv",
            "field_values_path": "fields.csv",
            "field_columns_path": "columns.csv",
        },
        "domain": {
            "external_id": 3,
            "layout_name": "plasma",
            "quadrilateral_order": "q1_reference",
            "excluded_boundary_ids": [5],
        },
        "boundary_groups": [{"name": "wall", "external_ids": [6, 7, 8], "material_id": 0}],
        "fields": {
            "coordinate_columns": ["r_m", "z_m"],
            "inside_column": "inside_model_domain",
            "gas_velocity_columns": [
                "gas_velocity_r_m_per_s",
                "gas_velocity_z_m_per_s",
            ],
            "gas_temperature_column": "gas_temperature_K",
            "gas_density_column": "gas_density_kg_per_m3",
            "gas_dynamic_viscosity_column": "dynamic_viscosity_Pa_s",
            "gas_mean_free_path_column": "gas_mean_free_path_m",
            "coordinate_match_tolerance_m": 1.0e-14,
            "axis_radial_velocity_tolerance_m_s": 2.0e-5,
        },
    }
    path = tmp_path / "adapter.yaml"
    path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    return path


def _assert_geometry(bundle: DataBundle) -> None:
    external_ids = bundle.geometry.node_external_id
    triangles = bundle.geometry.tri3
    boundary_external_ids = bundle.geometry.boundary.external_id
    assert external_ids is not None
    assert triangles is not None
    assert boundary_external_ids is not None
    assert np.array_equal(external_ids, [10, 20, 30, 40])
    assert np.array_equal(triangles, [[0, 1, 3], [0, 3, 2]])
    assert bundle.geometry.group_names == ("wall",)
    assert np.array_equal(boundary_external_ids, [101, 102, 103])
    assert np.all(bundle.geometry.boundary.orientation == 1)


def _assert_fields(bundle: DataBundle) -> None:
    by_name = {field.name: field for field in bundle.fields}
    assert set(by_name) == {
        "gas_density",
        "gas_dynamic_viscosity",
        "gas_mean_free_path",
        "gas_temperature",
        "gas_velocity",
    }
    velocity = by_name["gas_velocity"].values
    assert np.array_equal(velocity[:, 0], [0.0, 0.2, 0.0, 0.4])
    assert np.array_equal(by_name["gas_temperature"].values[:, 0], [310.0, 320.0, 330.0, 340.0])


def test_adapter_builds_deterministic_p1_bundle_and_projects_axis(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    output = tmp_path / "case.h5"
    report_path = tmp_path / "report.json"

    report = adapt_from_configuration(configuration, output, report_path=report_path)
    bundle = read(output)

    assert report["adapter_revision"] == ADAPTER_REVISION
    assert report["node_count"] == 4
    assert report["source_quadrilateral_count"] == 1
    assert report["p1_triangle_count"] == 2
    assert report["split_diagonal_03_count"] == 1
    assert report["split_diagonal_12_count"] == 0
    assert report["minimum_triangle_quality"] == pytest.approx(3.0**0.5 / 2.0)
    assert report["physical_boundary_facet_count"] == 3
    assert report["excluded_boundary_facet_count"] == 1
    assert report["corrected_axis_node_count"] == 2
    assert report_path.is_file()
    _assert_geometry(bundle)
    _assert_fields(bundle)
    repeat = adapt_from_configuration(configuration, tmp_path / "repeat.h5")
    assert repeat["output_content_hash"] == report["output_content_hash"]
    with pytest.raises(FileExistsError):
        adapt_from_configuration(configuration, output)


def test_adapter_rejects_axis_projection_above_explicit_limit(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path, axis_velocity=3.0e-5)

    with pytest.raises(ValueError, match="axis radial gas velocity"):
        adapt_from_configuration(configuration, tmp_path / "case.h5")


def test_adapter_requires_exact_boundary_id_inventory(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    document["boundary_groups"][0]["external_ids"].append(999)
    configuration.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly match"):
        adapt_from_configuration(configuration, tmp_path / "case.h5")


def test_adapter_requires_exclusions_to_be_exact_axis_facets(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    document["domain"]["excluded_boundary_ids"] = [6]
    document["boundary_groups"][0]["external_ids"] = [5, 7, 8]
    configuration.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="non-axis facet"):
        adapt_from_configuration(configuration, tmp_path / "case.h5")


def test_adapter_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    with configuration.open("a", encoding="utf-8") as stream:
        stream.write("format_version: 1\n")

    with pytest.raises(ValueError, match="duplicate YAML mapping key"):
        adapt_from_configuration(configuration, tmp_path / "case.h5")


@pytest.mark.parametrize("token", ["not-a-number", "Inf"])
def test_adapter_rejects_invalid_field_tokens_even_in_unselected_rows(
    tmp_path: Path, token: str
) -> None:
    configuration = _fixture(tmp_path)
    field_path = tmp_path / "fields.csv"
    field_path.write_text(
        field_path.read_text(encoding="utf-8").replace("NaN", token),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"is not numeric|is not finite"):
        adapt_from_configuration(configuration, tmp_path / "case.h5")


@pytest.mark.parametrize(
    "vertices_text",
    [
        "vertex_id,r_m,z_m,z_m\n40,1.0,1.0,1.0\n",
        "vertex_id,r_m,z_m\n40,1.0,1.0,unexpected\n",
        'vertex_id,r_m,z_m\n"40,1.0,1.0\n',
    ],
)
def test_adapter_rejects_malformed_csv_rows(tmp_path: Path, vertices_text: str) -> None:
    configuration = _fixture(tmp_path)
    (tmp_path / "vertices.csv").write_text(vertices_text, encoding="utf-8")

    with pytest.raises(ValueError, match=r"columns must be exactly|row width|malformed CSV"):
        adapt_from_configuration(configuration, tmp_path / "case.h5")
