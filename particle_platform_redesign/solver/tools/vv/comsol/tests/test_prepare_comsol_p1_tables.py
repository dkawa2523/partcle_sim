from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.verification.microcases import build_microcase
from tools.vv.comsol.prepare_comsol_p1_tables import prepare, write_sectionwise_p1

from chamber_particles.case_format import FieldData, P1TriLayout, RealizedTableSource, read, write
from chamber_particles.fields import FieldLocationError


def test_sectionwise_writer_preserves_exact_p1_connectivity(tmp_path: Path) -> None:
    nodes = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
        dtype=np.float64,
    )
    connectivity = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    values = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    path = tmp_path / "field.txt"

    write_sectionwise_p1(path, nodes, connectivity, "field", values)

    sections: dict[str, list[str]] = {}
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("%"):
            current = line
            sections[current] = []
        else:
            sections[current].append(line)
    np.testing.assert_array_equal(
        np.loadtxt(sections["%Coordinates"]),
        nodes,
    )
    np.testing.assert_array_equal(
        np.loadtxt(sections["%Elements"], dtype=np.int64),
        connectivity + 1,
    )
    np.testing.assert_array_equal(
        np.loadtxt(sections["%Data (field)"]),
        values,
    )


def _input(path: Path, count: int = 5) -> None:
    base = build_microcase("C07").data
    nodes = base.geometry.nodes_m + np.array([1.0, 0.0])
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<i8")
    boundary = replace(
        base.geometry.boundary,
        owner_cell_type=np.ones(4, dtype="<u1"),
        owner_cell_local_index=np.asarray([0, 0, 1, 1], dtype="<i8"),
    )
    geometry = replace(
        base.geometry,
        nodes_m=nodes,
        boundary=boundary,
        tri3=triangles,
        tri3_domain_id=np.zeros(2, dtype="<i4"),
        quad4=None,
        quad4_domain_id=None,
    )
    layout = P1TriLayout("field", nodes, triangles, np.ones(triangles.shape[0], dtype="<u1"))
    r, z = nodes.T
    scalar = 2 * r + 3 * z
    data_fields = tuple(
        FieldData(name, "field", "node", components, basis, np.asarray(values, dtype="<f8"), unit)
        for name, components, basis, values, unit in (
            (
                "gas_velocity",
                ("r", "z"),
                "axisymmetric_rz",
                np.column_stack((r + z, 2 * r - z)),
                "m/s",
            ),
            ("gas_temperature", ("value",), "scalar", (300 + scalar)[:, None], "K"),
            ("gas_density", ("value",), "scalar", (1e-4 * (1 + scalar))[:, None], "kg/m^3"),
            ("gas_dynamic_viscosity", ("value",), "scalar", (2e-5 * (1 + scalar))[:, None], "Pa*s"),
            (
                "electric_field",
                ("r", "z"),
                "axisymmetric_rz",
                np.column_stack((r, 2 * z + 1)),
                "V/m",
            ),
        )
    )
    source = base.sources[0]
    assert isinstance(source, RealizedTableSource)
    source = RealizedTableSource(
        source.name,
        particle_id=np.arange(1, count + 1, dtype="<i8"),
        release_time_s=np.zeros(count, dtype="<f8"),
        position_m=np.tile([1.25, 0.25], (count, 1)).astype("<f8"),
        velocity_m_s=np.tile(source.velocity_m_s[0], (count, 1)).astype("<f8"),
        charge_number=np.full(count, source.charge_number[0], dtype="<f8"),
        mass_kg=np.full(count, source.mass_kg[0], dtype="<f8"),
        drag_diameter_m=np.full(count, source.drag_diameter_m[0], dtype="<f8"),
        contact_radius_m=np.full(count, source.contact_radius_m[0], dtype="<f8"),
        electrostatic_radius_m=np.full(count, source.electrostatic_radius_m[0], dtype="<f8"),
        displaced_volume_m3=np.full(count, source.displaced_volume_m3[0], dtype="<f8"),
        model_weight=np.ones(count, dtype="<f8"),
        material_id=np.zeros(count, dtype="<i4"),
    )
    write(
        path,
        replace(
            base,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            layouts=(layout,),
            fields=data_fields,
            sources=(source,),
        ),
    )


@pytest.mark.parametrize("count", [1, 5])
def test_complete_p1_tables_and_affine_release_probes_match_independent_values(
    tmp_path: Path,
    count: int,
) -> None:
    source, output = tmp_path / "input.h5", tmp_path / "tables"
    _input(source, count)
    prepare(source, output)
    nodes = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)
    lines = (output / "m3v_Ez_sectionwise.txt").read_text(encoding="utf-8").splitlines()
    data_start = lines.index("%Data (m3v_Ez)") + 1
    np.testing.assert_array_equal(np.loadtxt(lines[data_start:]), 2 * nodes[:, 1] + 1)
    with (output / "canonical_p1_release_probes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == count + 1
    assert rows[1][0] == "p1_t0" and rows[-1][0] == f"p{count}_t0"
    expected = np.array([1.25, 0.25, 1.5, 2.25, 303.25, 4.25e-4, 8.5e-5, 1.25, 1.5])
    np.testing.assert_allclose(
        np.asarray(rows[1:], dtype=object)[:, 1:].astype(float),
        np.tile(expected, (count, 1)),
        rtol=2e-15,
        atol=0,
    )
    receipt = json.loads((output / "table_receipt.json").read_text(encoding="utf-8"))
    assert receipt["release_probe_count"] == count
    assert receipt["node_count"] == 4 and receipt["cell_count"] == 2
    assert not any(p.name.startswith(".p1-tables-") for p in tmp_path.iterdir())


def test_p1_receipt_failure_leaves_no_partial_input_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output = tmp_path / "input.h5", tmp_path / "tables"
    _input(source)
    original = source.read_bytes()
    write_text = Path.write_text

    def fail_receipt(path: Path, *args, **kwargs):
        if path.name == "table_receipt.json":
            raise OSError("injected table receipt failure")
        return write_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_receipt)
        with pytest.raises(OSError, match="injected table receipt"):
            prepare(source, output)
    assert not output.exists()
    assert source.read_bytes() == original
    assert not any(p.name.startswith(".p1-tables-") for p in tmp_path.iterdir())
    prepare(source, output)
    assert (output / "table_receipt.json").is_file()
    with pytest.raises(FileExistsError):
        prepare(source, output)


@pytest.mark.parametrize("defect", ["unit", "time", "cell"])
def test_exact_p1_export_rejects_unsupported_field_meaning_before_publication(
    tmp_path: Path, defect: str
) -> None:
    source = tmp_path / "source.h5"
    _input(source)
    data = read(source)
    fields = list(data.fields)
    index = next(i for i, field in enumerate(fields) if field.name == "gas_temperature")
    field = fields[index]
    if defect == "unit":
        fields[index] = replace(field, unit="m")
    elif defect == "time":
        fields[index] = replace(
            field, time_s=np.array([0.0, 1.0]), values=np.stack((field.values, field.values + 1.0))
        )
    else:
        fields[index] = replace(field, association="cell", values=field.values[:2])
    unsupported = tmp_path / "unsupported.h5"
    write(unsupported, replace(data, fields=tuple(fields)))
    output = tmp_path / "tables"
    with pytest.raises((ValueError, FieldLocationError)):
        prepare(unsupported, output)
    assert not output.exists()
