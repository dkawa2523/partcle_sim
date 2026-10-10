from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tools.vv.comsol.prepare_m3c1_common_p1_tables import (
    COMPONENT_EXPORTS,
    ComponentExport,
    prepare,
)

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    RealizedTableSource,
    content_hash,
    write,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle() -> DataBundle:
    nodes = np.asarray([[0.05, 0.0], [1.0, 0.0], [1.0, 1.0], [0.05, 1.0]], dtype="<f8")
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<i8")
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [2, 1], [2, 3], [3, 0]], dtype="<i8"),
        boundary_id=np.asarray([10, 11, 12, 13], dtype="<i4"),
        group_id=np.asarray([0, 0, 0, 0], dtype="<i4"),
        material_id=np.asarray([1, 1, 1, 1], dtype="<i4"),
        owner_cell_type=np.asarray([1, 1, 1, 1], dtype="<u1"),
        owner_cell_local_index=np.asarray([0, 0, 1, 1], dtype="<i8"),
        orientation=np.asarray([1, -1, 1, 1], dtype="<i1"),
    )
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("wall",),
        tri3=triangles,
        tri3_domain_id=np.asarray([1, 1], dtype="<i4"),
    )
    layout = P1TriLayout(
        "common_p1",
        nodes,
        triangles,
        np.asarray([1, 1], dtype="<u1"),
    )

    grouped: dict[str, list[tuple[int, ComponentExport]]] = defaultdict(list)
    for index, export in enumerate(COMPONENT_EXPORTS, start=1):
        grouped[export.field].append((index, export))
    fields = []
    for name, exports in grouped.items():
        components = tuple(item.component for _index, item in exports)
        values = np.column_stack(
            [index + nodes[:, 0] + 2.0 * nodes[:, 1] for index, _item in exports]
        ).astype("<f8")
        fields.append(
            FieldData(
                name=name,
                layout=layout.name,
                association="node",
                components=components,
                stored_basis="scalar" if components == ("value",) else "axisymmetric_rz",
                values=values,
                unit=exports[0][1].unit,
            )
        )

    positions = np.asarray(
        [(r, z) for r in np.linspace(0.1, 0.9, 41) for z in np.linspace(0.1, 0.9, 7)],
        dtype="<f8",
    )
    count = positions.shape[0]
    velocity = np.column_stack(
        (0.25 + positions[:, 0] - positions[:, 1], -0.5 + 2.0 * positions[:, 1])
    ).astype("<f8")
    source = RealizedTableSource(
        name="particles",
        particle_id=np.arange(1, count + 1, dtype="<i8"),
        release_time_s=np.zeros(count, dtype="<f8"),
        position_m=positions,
        velocity_m_s=velocity,
        charge_number=np.full(count, -1.0, dtype="<f8"),
        mass_kg=np.full(count, 1.0e-18, dtype="<f8"),
        drag_diameter_m=np.full(count, 1.0e-7, dtype="<f8"),
        contact_radius_m=np.zeros(count, dtype="<f8"),
        electrostatic_radius_m=np.full(count, 5.0e-8, dtype="<f8"),
        displaced_volume_m3=np.full(count, 5.0e-22, dtype="<f8"),
        model_weight=np.ones(count, dtype="<f8"),
        material_id=np.ones(count, dtype="<i4"),
    )
    provenance = json.dumps(
        {
            "source_sha256": f"sha256:{'0' * 64}",
            "producer_metadata": {"case": "synthetic-common-field"},
            "producer_version": "1",
            "producer": "verification",
            "field_semantics_revision": "primitive-v1",
        }
    )
    return DataBundle(
        "axisymmetric_rz",
        provenance,
        geometry,
        (layout,),
        tuple(fields),
        (source,),
    )


def _sections(path: Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("%"):
            current = line
            sections[current] = []
        else:
            sections[current].append(line)
    return sections


@pytest.fixture
def prepared_common(tmp_path: Path) -> tuple[DataBundle, Path, Path]:
    bundle = _bundle()
    candidate = tmp_path / "candidate.h5"
    write(candidate, bundle)
    output = tmp_path / "common"
    prepare(candidate, output)
    return bundle, candidate, output


def test_full_mapping_connectivity_and_receipt_are_locked(
    prepared_common: tuple[DataBundle, Path, Path],
) -> None:
    bundle, candidate, output = prepared_common

    receipt_path = output / "common_p1_table_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["candidate"] == {
        "path": str(candidate.resolve()),
        "file_sha256": _sha256(candidate),
        "content_hash": content_hash(bundle),
    }
    assert receipt["field_count"] == 17
    assert receipt["component_count"] == 22
    assert receipt["layout"] == {
        "name": "common_p1",
        "type": "P1TriLayout",
        "node_count": 4,
        "cell_count": 2,
        "connectivity_width": 3,
        "connectivity_indexing_in_candidate": "zero_based",
        "connectivity_indexing_in_sectionwise_files": "one_based",
        "interpolation_contract": "first_order_triangular_sectionwise",
    }
    expected_functions = [export.function for export in COMPONENT_EXPORTS]
    assert [item["function"] for item in receipt["components"]] == expected_functions
    assert set(receipt["artifacts"]) == {
        *(f"{name}_sectionwise.txt" for name in expected_functions),
        "m3c1_vr0.txt",
        "m3c1_vz0.txt",
        "m3c1_Z0.txt",
        "common_p1_release_probes.csv",
        "boundary_meaning.json",
    }
    for name, record in receipt["artifacts"].items():
        assert record["sha256"] == _sha256(output / name)
        assert record["size_bytes"] == (output / name).stat().st_size

    meaning = json.loads((output / "boundary_meaning.json").read_text(encoding="utf-8"))
    assert meaning == {
        "canonical_input_sha256": _sha256(candidate),
        "canonical_content_hash": content_hash(bundle),
        "boundary_groups": {"10": "wall", "11": "wall", "12": "wall", "13": "wall"},
        "authority": "canonical_geometry_boundary_ids_and_groups",
    }

    sections = _sections(output / "m3c1_Ez_sectionwise.txt")
    layout = bundle.layouts[0]
    assert isinstance(layout, P1TriLayout)
    np.testing.assert_array_equal(
        np.loadtxt(sections["%Elements"], dtype=np.int64),
        layout.connectivity + 1,
    )
    electric = next(field for field in bundle.fields if field.name == "electric_field")
    np.testing.assert_array_equal(
        np.loadtxt(sections["%Data (m3c1_Ez)"]),
        electric.values[:, 1],
    )


def test_release_probe_and_initial_state_tables_preserve_candidate_values(
    prepared_common: tuple[DataBundle, Path, Path],
) -> None:
    bundle, _candidate, output = prepared_common
    with (output / "common_p1_release_probes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][:6] == [
        "particle_id",
        "r_m",
        "z_m",
        "velocity_r_m_per_s",
        "velocity_z_m_per_s",
        "charge_number",
    ]
    assert rows[0][6:] == [export.probe_column for export in COMPONENT_EXPORTS]
    assert len(rows) == 288
    first = np.asarray(rows[1][1:], dtype=np.float64)
    np.testing.assert_allclose(first[5:], np.arange(1.0, 23.0) + 0.1 + 2.0 * 0.1)
    source = bundle.sources[0]
    assert isinstance(source, RealizedTableSource)
    np.testing.assert_array_equal(
        np.loadtxt(output / "m3c1_vr0.txt")[0],
        np.asarray([*source.position_m[0], source.velocity_m_s[0, 0]]),
    )
    np.testing.assert_array_equal(
        np.loadtxt(output / "m3c1_Z0.txt")[:, 2],
        -np.ones(287),
    )


def test_prepare_is_no_clobber_and_missing_field_fails_before_output(tmp_path: Path) -> None:
    bundle = _bundle()
    candidate = tmp_path / "candidate.h5"
    write(candidate, bundle)
    output = tmp_path / "common"
    prepare(candidate, output)
    receipt_before = (output / "common_p1_table_receipt.json").read_bytes()

    with pytest.raises(FileExistsError):
        prepare(candidate, output)
    assert (output / "common_p1_table_receipt.json").read_bytes() == receipt_before

    missing_bundle = replace(bundle, fields=bundle.fields[:-1])
    missing_candidate = tmp_path / "missing.h5"
    write(missing_candidate, missing_bundle)
    missing_output = tmp_path / "missing-output"
    with pytest.raises(ValueError, match="candidate field set mismatch"):
        prepare(missing_candidate, missing_output)
    assert not missing_output.exists()
