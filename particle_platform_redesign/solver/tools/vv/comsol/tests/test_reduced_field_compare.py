from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from tools.vv.comsol.compare_reduced_fields import (
    COMPARISON_REVISION,
    compare_reduced_fields,
)

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    write,
)


def _field(name: str, values: object, unit: str, *, vector: bool = False) -> FieldData:
    return FieldData(
        name,
        "plasma",
        "node",
        ("r", "z") if vector else ("value",),
        "axisymmetric_rz" if vector else "scalar",
        np.asarray(values, dtype="<f8"),
        unit,
    )


def _case(path: Path, *, electric_field_unit: str = "V/m") -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype="<f8")
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<i8")
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 3]], dtype="<i8"),
        boundary_id=np.asarray([6, 6, 6], dtype="<i4"),
        group_id=np.zeros(3, dtype="<i4"),
        material_id=np.zeros(3, dtype="<i4"),
        owner_cell_type=np.ones(3, dtype="<u1"),
        owner_cell_local_index=np.asarray([0, 0, 1], dtype="<i8"),
        orientation=np.ones(3, dtype="<i1"),
    )
    scalar = np.arange(1.0, 5.0)[:, None]
    vector = np.column_stack((np.arange(1.0, 5.0), np.arange(2.0, 6.0)))
    provenance = json.dumps(
        {
            "producer": "test-builder",
            "producer_version": "1",
            "source_sha256": f"sha256:{'0' * 64}",
            "field_semantics_revision": "test",
            "producer_metadata": {
                "convergence": {
                    "total_space_charge_C": 1.0,
                    "boundary_reaction_charge_C": 1.0,
                    "charge_balance_error_C": 0.0,
                }
            },
        }
    )
    fields = (
        _field("electric_potential", scalar, "V"),
        _field("electric_field", vector, electric_field_unit, vector=True),
        _field("electron_number_density", scalar * 1.0e15, "1/m^3"),
        _field("positive_ion_number_density", scalar * 1.1e15, "1/m^3"),
        _field("space_charge_density", scalar * 1.0e-6, "C/m^3"),
        _field("positive_ion_speed", scalar * 10.0, "m/s"),
        _field("positive_ion_velocity", vector * 10.0, "m/s", vector=True),
    )
    write(
        path,
        DataBundle(
            "axisymmetric_rz",
            provenance,
            GeometryData(
                nodes,
                boundary,
                ("wall",),
                tri3=triangles,
                tri3_domain_id=np.full(2, 3, dtype="<i4"),
            ),
            (P1TriLayout("plasma", nodes, triangles, np.ones(2, dtype="<u1")),),
            fields,
        ),
    )


def _reference(directory: Path) -> tuple[Path, Path]:
    names = [
        "r_m",
        "z_m",
        "SASS_potential_V",
        "electric_field_r_V_per_m",
        "electric_field_z_V_per_m",
        "electron_density_per_m3",
        "total_positive_ion_density_per_m3",
        "space_charge_density_C_per_m3",
        "ion_speed_m_per_s",
        "ion_velocity_r_m_per_s",
        "ion_velocity_z_m_per_s",
    ]
    units = ["m", "m", "V", "V/m", "V/m", "1/m^3", "1/m^3", "C/m^3", "m/s", "m/s", "m/s"]
    columns = directory / "columns.csv"
    with columns.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["column", "COMSOL_expression", "unit"])
        writer.writerows(zip(names, names, units, strict=True))
    values = directory / "values.csv"
    rows = []
    for index, (r_m, z_m) in enumerate(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), start=1):
        rows.append(
            [
                r_m,
                z_m,
                index + 0.5,
                index + 0.5,
                index + 1.5,
                index * 1.0e15 + 1.0,
                index * 1.1e15 + 1.0,
                index * 1.0e-6 + 1.0e-8,
                index * 10.0 + 0.5,
                index * 10.0 + 0.5,
                (index + 1) * 10.0 + 0.5,
            ]
        )
    with values.open("w", encoding="utf-8", newline="") as stream:
        stream.write("% synthetic reference\n")
        csv.writer(stream).writerows(reversed(rows))
    return values, columns


def test_reduced_field_comparison_is_descriptive_and_no_clobber(tmp_path: Path) -> None:
    case = tmp_path / "case.h5"
    _case(case)
    values, columns = _reference(tmp_path)
    output = tmp_path / "comparison.json"

    report = compare_reduced_fields(
        case,
        values,
        columns,
        output,
        coordinate_tolerance_m=1.0e-12,
    )

    assert report["comparison_revision"] == COMPARISON_REVISION
    assert report["classification"] == "descriptive_external_vv_not_solver_acceptance"
    metrics = report["weighted_field_metrics"]
    assert isinstance(metrics, dict)
    assert metrics["electric_potential"]["absolute_linf"] == 0.5
    assert metrics["electric_potential"]["difference_weighted_l2"] == pytest.approx(
        0.5 * math.sqrt(math.pi)
    )
    trace = report["boundary_potential_trace"]
    assert isinstance(trace, dict)
    assert trace["wall"]["node_count"] == 4
    assert trace["wall"]["exclusive_node_count"] == 4
    assert trace["shared_group_corner_node_count"] == 0
    coverage = report["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["independent_mesh_convergence"] == "NOT_TESTED_SINGLE_REFERENCE_MESH"
    assert output.is_file()
    with pytest.raises(FileExistsError):
        compare_reduced_fields(
            case,
            values,
            columns,
            output,
            coordinate_tolerance_m=1.0e-12,
        )


def test_reduced_field_comparison_rejects_generated_field_metadata(tmp_path: Path) -> None:
    case = tmp_path / "case.h5"
    _case(case, electric_field_unit="kV/m")
    values, columns = _reference(tmp_path)

    with pytest.raises(ValueError, match="metadata does not match"):
        compare_reduced_fields(
            case,
            values,
            columns,
            tmp_path / "comparison.json",
            coordinate_tolerance_m=1.0e-12,
        )


def test_reduced_field_comparison_rejects_reference_units(tmp_path: Path) -> None:
    case = tmp_path / "case.h5"
    _case(case)
    values, columns = _reference(tmp_path)
    text = columns.read_text(encoding="utf-8")
    columns.write_text(
        text.replace(
            "electric_field_r_V_per_m,electric_field_r_V_per_m,V/m",
            "electric_field_r_V_per_m,electric_field_r_V_per_m,kV/m",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference column units"):
        compare_reduced_fields(
            case,
            values,
            columns,
            tmp_path / "comparison.json",
            coordinate_tolerance_m=1.0e-12,
        )


def test_reduced_field_comparison_rejects_nonnumeric_reference_tokens(
    tmp_path: Path,
) -> None:
    case = tmp_path / "case.h5"
    _case(case)
    values, columns = _reference(tmp_path)
    text = values.read_text(encoding="utf-8")
    values.write_text(text.replace("0.0", "not-a-number", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="is not numeric"):
        compare_reduced_fields(
            case,
            values,
            columns,
            tmp_path / "comparison.json",
            coordinate_tolerance_m=1.0e-12,
        )


def test_reduced_field_comparison_rejects_malformed_column_csv(tmp_path: Path) -> None:
    case = tmp_path / "case.h5"
    _case(case)
    values, columns = _reference(tmp_path)
    columns.write_text(
        'column,COMSOL_expression,unit\n"r_m,r,m\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed"):
        compare_reduced_fields(
            case,
            values,
            columns,
            tmp_path / "comparison.json",
            coordinate_tolerance_m=1.0e-12,
        )
