from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml
from tools.comsol_adapter import ADAPTER_REVISION, adapt_from_configuration

from chamber_particles.case_format import DataBundle, read
from chamber_particles.fields import locate_field_cell, sample_field


def _write_csv(path: Path, header: Sequence[object], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _mapping(
    name: str,
    columns: list[str],
    components: list[str],
    basis: str,
    unit: str,
    *,
    axis: bool = False,
) -> dict[str, object]:
    return {
        "name": name,
        "columns": columns,
        "components": components,
        "stored_basis": basis,
        "association": "node",
        "unit": unit,
        "input_unit": unit,
        "scale": 1.0,
        "offset": 0.0,
        "producer_record": {
            "producer": "manufactured affine reference",
            "dataset": "synthetic",
            "solution": "analytic",
            "member": {"indexing": "none", "selection": "single"},
            "time": {"representation": "static", "selection": "steady"},
            "domain_ids": [3],
            "side": "domain",
            "recovery": "analytic nodal sampling",
            "continuity": "continuous",
            "authority": "supplied analytic primitive",
            "topology": "fixed",
            "support": "full_selected_domain",
            "evidence": "test analytic coefficients",
        },
        "axis_transform": {
            "component": "r",
            "max_abs_correction": 2.0e-5,
            "reason": "explicit synthetic radial parity projection",
        }
        if axis
        else None,
    }


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
        "format_version": 3,
        "source": {
            "vertices_path": "vertices.csv",
            "triangles_path": "triangles.csv",
            "quadrilaterals_path": "quads.csv",
            "boundary_edges_path": "boundaries.csv",
            "field_values_path": "fields.csv",
            "field_columns_path": "columns.csv",
        },
        "domain": {
            "coordinate_system": "axisymmetric_rz",
            "external_id": 3,
            "layout_name": "plasma",
            "quadrilateral_order": "q1_reference",
            "excluded_boundary_ids": [5],
        },
        "boundary_groups": [{"name": "wall", "external_ids": [6, 7, 8], "material_id": 0}],
        "fields": {
            "coordinate_columns": ["r_m", "z_m"],
            "inside_column": "inside_model_domain",
            "node_id_column": None,
            "domain_scope": None,
            "coordinate_match_tolerance_m": 1.0e-14,
            "mappings": [
                _mapping(
                    "gas_velocity",
                    ["gas_velocity_r_m_per_s", "gas_velocity_z_m_per_s"],
                    ["r", "z"],
                    "axisymmetric_rz",
                    "m/s",
                    axis=True,
                ),
                _mapping("gas_temperature", ["gas_temperature_K"], ["value"], "scalar", "K"),
                _mapping("gas_density", ["gas_density_kg_per_m3"], ["value"], "scalar", "kg/m^3"),
                _mapping(
                    "gas_dynamic_viscosity", ["dynamic_viscosity_Pa_s"], ["value"], "scalar", "Pa*s"
                ),
                _mapping("gas_mean_free_path", ["gas_mean_free_path_m"], ["value"], "scalar", "m"),
            ],
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
    transformations = cast(list[dict[str, object]], report["axis_transformations"])
    assert transformations[0]["changed_node_count"] == 2
    assert report_path.is_file()
    _assert_geometry(bundle)
    _assert_fields(bundle)
    repeat = adapt_from_configuration(configuration, tmp_path / "repeat.h5")
    assert repeat["output_content_hash"] == report["output_content_hash"]
    with pytest.raises(FileExistsError):
        adapt_from_configuration(configuration, output)


def test_adapter_rejects_axis_projection_above_explicit_limit(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path, axis_velocity=3.0e-5)

    with pytest.raises(ValueError, match="axis correction"):
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
        stream.write("format_version: 3\n")

    with pytest.raises(ValueError, match="duplicate YAML key"):
        adapt_from_configuration(configuration, tmp_path / "case.h5")


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ("  external_id: 3\n  external_id: 4", "duplicate YAML key"),
        ("  <<: {external_id: 3}\n  external_id: 3", "merge keys"),
    ],
)
def test_adapter_rejects_ambiguous_nested_configuration(
    tmp_path: Path, replacement: str, error: str
) -> None:
    configuration = _fixture(tmp_path)
    configuration.write_text(
        configuration.read_text(encoding="utf-8").replace("  external_id: 3", replacement),
        encoding="utf-8",
    )
    output = tmp_path / "case.h5"
    with pytest.raises(ValueError, match=error):
        adapt_from_configuration(configuration, output)
    assert not output.exists()


def test_adapter_reports_unrepresentable_numeric_setting_as_value_error(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    document["fields"]["coordinate_match_tolerance_m"] = 10**400
    configuration.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate_match_tolerance_m must be finite"):
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


def _replace_document(path: Path, document: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_generic_xy_fields_preserve_affine_values_units_and_lineage(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    document["domain"]["coordinate_system"] = "cartesian_xy"
    document["domain"]["excluded_boundary_ids"] = []
    document["boundary_groups"][0]["external_ids"] = [5, 6, 7, 8]
    field = _mapping("electric_field", ["Er", "Ez"], ["x", "y"], "cartesian_xy", "V/m")
    field["input_unit"] = "kV/m"
    field["scale"] = 1000.0
    temperature = _mapping("gas_temperature", ["Tc"], ["value"], "scalar", "K")
    temperature["input_unit"] = "degC"
    temperature["offset"] = 273.15
    document["fields"]["coordinate_columns"] = ["x_m", "y_m"]
    document["fields"]["mappings"] = [field, temperature]
    _replace_document(configuration, document)
    _write_csv(
        tmp_path / "vertices.csv",
        ["vertex_id", "x_m", "y_m"],
        [[10, -1, 0], [20, 1, 0], [30, -1, 1], [40, 1, 1]],
    )
    _write_csv(
        tmp_path / "columns.csv",
        ["column", "COMSOL_expression", "unit"],
        [
            ["x_m", "x", "m"],
            ["y_m", "y", "m"],
            ["inside_model_domain", "1", "1"],
            ["Er", "2*x+3*y", "kV/m"],
            ["Ez", "-x+4*y", "kV/m"],
            ["Tc", "20+x+2*y", "degC"],
        ],
    )
    with (tmp_path / "fields.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(
            [
                [x, y, 1, 2 * x + 3 * y, -x + 4 * y, 20 + x + 2 * y]
                for x, y in [(1, 1), (-1, 1), (1, 0), (-1, 0)]
            ]
        )
    report = adapt_from_configuration(configuration, tmp_path / "case.h5")
    bundle = read(tmp_path / "case.h5")
    location = locate_field_cell(bundle.layouts[0], np.asarray([0.25, 0.4]))
    by_name = {item.name: item for item in bundle.fields}
    np.testing.assert_allclose(
        sample_field(by_name["electric_field"], location).value,
        [1700.0, 1350.0],
        rtol=0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        sample_field(by_name["gas_temperature"], location).value, [294.20], rtol=0, atol=1.0e-12
    )
    metadata = json.loads(bundle.provenance_json)["producer_metadata"]
    assert metadata["selected_field_lineage"][0]["expressions"] == ["2*x+3*y", "-x+4*y"]
    assert report["axis_transformations"] == []


def _add_node_ids(configuration: Path, tmp_path: Path) -> None:
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    document["fields"]["node_id_column"] = "vertex_id"
    _replace_document(configuration, document)
    columns = (tmp_path / "columns.csv").read_text(encoding="utf-8")
    (tmp_path / "columns.csv").write_text(columns + "vertex_id,mesh node ID,1\n", encoding="utf-8")
    path = tmp_path / "fields.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    identities = iter([40, 30, 999, 20, 10])
    path.write_text(
        "\n".join(row if row.startswith("%") else row + "," + str(next(identities)) for row in rows)
        + "\n",
        encoding="utf-8",
    )


def test_scoped_node_id_join_uses_identity_and_verifies_coordinates(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    _add_node_ids(configuration, tmp_path)
    report = adapt_from_configuration(configuration, tmp_path / "case.h5")
    assert report["field_join_method"] == "scoped_node_id"
    _assert_fields(read(tmp_path / "case.h5"))
    path = tmp_path / "fields.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("1.0,1.0,1,0.4", "0.5,1.0,1,0.4"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="coordinates disagree"):
        adapt_from_configuration(configuration, tmp_path / "bad.h5")


def test_node_id_join_rejects_duplicate_and_probe_identity(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    _add_node_ids(configuration, tmp_path)
    path = tmp_path / "fields.csv"
    text = path.read_text(encoding="utf-8")
    for wrong_id in (20, 1123):
        path.write_text(text.replace(",40\n", f",{wrong_id}\n"), encoding="utf-8")
        with pytest.raises(ValueError, match="bijection"):
            adapt_from_configuration(configuration, tmp_path / "bad.h5")


@pytest.mark.parametrize("setting", ["time", "association", "continuity", "side", "domain"])
def test_adapter_rejects_unimplemented_or_mismatched_semantics(
    tmp_path: Path, setting: str
) -> None:
    configuration = _fixture(tmp_path)
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    item = document["fields"]["mappings"][0]
    if setting == "association":
        item["association"] = "cell"
    elif setting == "domain":
        item["producer_record"]["domain_ids"] = [999]
    elif setting == "time":
        item["producer_record"]["time"]["representation"] = "snapshots"
    else:
        item["producer_record"][setting] = "one_sided"
    _replace_document(configuration, document)
    with pytest.raises(ValueError):
        adapt_from_configuration(configuration, tmp_path / "case.h5")
    assert not (tmp_path / "case.h5").exists()


def test_selected_field_gaps_and_unacknowledged_axis_values_are_not_filled(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    document["fields"]["mappings"][0]["axis_transform"] = None
    _replace_document(configuration, document)
    report = adapt_from_configuration(configuration, tmp_path / "raw.h5")
    by_name = {item.name: item for item in read(tmp_path / "raw.h5").fields}
    assert by_name["gas_velocity"].values[0, 0] == 1.0e-5
    assert report["axis_transformations"] == []
    field_path = tmp_path / "fields.csv"
    field_path.write_text(
        field_path.read_text(encoding="utf-8").replace("340.0", "NaN"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="partially missing"):
        adapt_from_configuration(configuration, tmp_path / "gap.h5")


def test_domain_mesh_owns_selection_not_finite_primitives_outside_it(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    path = tmp_path / "fields.csv"
    text = path.read_text(encoding="utf-8")
    text = text.replace("9.0,9.0,1,NaN,NaN,NaN,NaN,NaN,NaN", "9.0,9.0,1,NaN,NaN,NaN,2.0,NaN,NaN")
    path.write_text(text, encoding="utf-8")
    report = adapt_from_configuration(configuration, tmp_path / "outside.h5")
    assert report["provider_field_row_count"] == 5
    assert report["selected_field_row_count"] == 4
    assert report["unselected_provider_row_count"] == 1
    _assert_fields(read(tmp_path / "outside.h5"))
    metadata = json.loads(read(tmp_path / "outside.h5").provenance_json)["producer_metadata"]
    assert metadata["source_file_hashes"]["field_values"].startswith("sha256:")
    path.write_text(text.replace("340.0", "NaN"), encoding="utf-8")
    with pytest.raises(ValueError, match="selected domain nodes"):
        adapt_from_configuration(configuration, tmp_path / "inside_gap.h5")


def test_explicit_domain_scope_resolves_provider_sides_without_ambiguous_join(
    tmp_path: Path,
) -> None:
    configuration = _fixture(tmp_path)
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    document["fields"]["domain_scope"] = {
        "column": "gas_temperature_K",
        "mode": "finite",
        "evidence": "Manufactured export declares temperature present exclusively on domain 3.",
    }
    _replace_document(configuration, document)
    path = tmp_path / "fields.csv"
    text = path.read_text(encoding="utf-8").replace(
        "9.0,9.0,1,NaN,NaN,NaN,NaN,NaN,NaN", "0.0,0.0,1,NaN,NaN,NaN,2.0,NaN,NaN"
    )
    path.write_text(text, encoding="utf-8")
    adapt_from_configuration(configuration, tmp_path / "scoped.h5")
    _assert_fields(read(tmp_path / "scoped.h5"))
    path.write_text(
        text.replace("0.0,0.0,1,NaN,NaN,NaN,2.0,NaN,NaN", "0.0,0.0,1,NaN,NaN,315.0,2.0,NaN,NaN"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coordinates are not unique"):
        adapt_from_configuration(configuration, tmp_path / "ambiguous.h5")


def test_explicit_domain_id_column_selects_same_coordinate_owner(tmp_path: Path) -> None:
    configuration = _fixture(tmp_path)
    document = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    document["fields"]["domain_scope"] = {
        "column": "domain_id",
        "mode": "equals_domain_id",
        "evidence": "Manufactured row domain IDs are bound to volume domain ownership.",
    }
    _replace_document(configuration, document)
    columns = tmp_path / "columns.csv"
    columns.write_text(
        columns.read_text(encoding="utf-8") + "domain_id,domain ID,1\n", encoding="utf-8"
    )
    path = tmp_path / "fields.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    scoped = [row if row.startswith("%") else row + ",3" for row in rows]
    text = (
        "\n".join(scoped).replace(
            "9.0,9.0,1,NaN,NaN,NaN,NaN,NaN,NaN,3", "0.0,0.0,1,0.0,12.0,999.0,12.0,1.0,1.0,4"
        )
        + "\n"
    )
    path.write_text(text, encoding="utf-8")
    adapt_from_configuration(configuration, tmp_path / "scoped_ids.h5")
    _assert_fields(read(tmp_path / "scoped_ids.h5"))
