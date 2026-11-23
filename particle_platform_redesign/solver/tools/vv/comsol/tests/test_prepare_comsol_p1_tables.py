from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.verification.microcases import build_microcase
from tools.vv.comsol.prepare_comsol_p1_tables import prepare, write_sectionwise_p1

from chamber_particles.case_format import FieldData, P1TriLayout, RealizedTableSource, write


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


def _input(path: Path) -> None:
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
        particle_id=np.arange(1, 288, dtype="<i8"),
        release_time_s=np.zeros(287, dtype="<f8"),
        position_m=np.tile([1.25, 0.25], (287, 1)).astype("<f8"),
        velocity_m_s=np.tile(source.velocity_m_s[0], (287, 1)).astype("<f8"),
        charge_number=np.full(287, source.charge_number[0], dtype="<f8"),
        mass_kg=np.full(287, source.mass_kg[0], dtype="<f8"),
        drag_diameter_m=np.full(287, source.drag_diameter_m[0], dtype="<f8"),
        contact_radius_m=np.full(287, source.contact_radius_m[0], dtype="<f8"),
        electrostatic_radius_m=np.full(287, source.electrostatic_radius_m[0], dtype="<f8"),
        displaced_volume_m3=np.full(287, source.displaced_volume_m3[0], dtype="<f8"),
        model_weight=np.ones(287, dtype="<f8"),
        material_id=np.zeros(287, dtype="<i4"),
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


def test_complete_p1_tables_and_affine_release_probes_match_independent_values(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "input.h5", tmp_path / "tables"
    _input(source)
    prepare(source, output)
    nodes = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)
    actual = np.loadtxt(output / "m3v_Ez.txt")
    np.testing.assert_array_equal(actual, np.column_stack((nodes, 2 * nodes[:, 1] + 1)))
    with (output / "canonical_p1_release_probes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 288
    assert rows[1][0] == "p1_t0" and rows[-1][0] == "p287_t0"
    expected = np.array([1.25, 0.25, 1.5, 2.25, 303.25, 4.25e-4, 8.5e-5, 1.25, 1.5])
    np.testing.assert_allclose(
        np.asarray(rows[1:], dtype=object)[:, 1:].astype(float),
        np.tile(expected, (287, 1)),
        rtol=2e-15,
        atol=0,
    )
    receipt = json.loads((output / "table_receipt.json").read_text(encoding="utf-8"))
    assert receipt["release_probe_count"] == 287
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
