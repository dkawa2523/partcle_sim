from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from pytest import MonkeyPatch
from tools.vv.comsol import normalize_m3c3_negative_ion_primitives as primitives


def _provider(path: Path, rows: list[tuple[float, ...]]) -> None:
    header = (
        "% r_geom_cm,z_geom_cm,negative_ion_density_per_m3,"
        "negative_ion_number_flux_r_per_m2_s,"
        "negative_ion_number_flux_z_per_m2_s,"
        "negative_ion_mass_density_kg_per_m3\n"
    )
    body = "".join(",".join(str(value) for value in row) + "\n" for row in rows)
    path.write_text("% Model,test.mph\n" + header + body, encoding="utf-8")


def test_normalize_maps_domain_and_boundary_dofs_and_projects_rz_axis(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    canonical = tmp_path / "candidate.h5"
    with h5py.File(canonical, "w") as output:
        geometry = output.create_group("geometry")
        geometry.create_dataset("nodes_m", data=[[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])
        boundary = geometry.create_group("boundary")
        boundary.create_dataset("line2", data=[[0, 1]])
        fields = output.create_group("fields")
        gas = fields.create_group("gas_temperature")
        gas.create_dataset("values", data=[[300.0], [310.0], [320.0]])

    ion_mass = 0.018 / primitives.AVOGADRO_PER_MOL
    domain_csv = tmp_path / "domain.csv"
    boundary_csv = tmp_path / "boundary.csv"
    _provider(domain_csv, [(0.0, 1.0, 4.0, 20.0, 28.0, 4.0 * ion_mass)])
    _provider(
        boundary_csv,
        [
            (0.0, 0.0, 2.0, 10.0, 14.0, 2.0 * ion_mass),
            (1.0, 0.0, 3.0, 15.0, 21.0, 3.0 * ion_mass),
        ],
    )
    source_mph = tmp_path / "source.mph"
    java_source = tmp_path / "export.java"
    runner = tmp_path / "run.ps1"
    for path in (source_mph, java_source, runner):
        path.write_text(path.name, encoding="utf-8")
    monkeypatch.setattr(primitives, "EXPECTED_MPH_SHA256", primitives._sha256(source_mph))
    monkeypatch.setattr(primitives, "EXPECTED_INPUT_SHA256", primitives._sha256(canonical))

    result = tmp_path / "three_current.h5"
    receipt = tmp_path / "receipt.json"
    primitives.normalize(
        argparse.Namespace(
            canonical_input=canonical,
            domain_csv=domain_csv,
            boundary_csv=boundary_csv,
            source_mph=source_mph,
            java_source=java_source,
            runner_script=runner,
            output=result,
            receipt=receipt,
        )
    )

    with h5py.File(result, "r") as output:
        velocity = np.asarray(output["fields/negative_ion_velocity/values"])
        np.testing.assert_array_equal(velocity[:, 0], [0.0, 5.0, 0.0])
        np.testing.assert_array_equal(velocity[:, 1], [7.0, 7.0, 7.0])
        np.testing.assert_allclose(
            output["fields/negative_ion_number_density/values"][:, 0], [2.0, 3.0, 4.0]
        )
    evidence = json.loads(receipt.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    assert evidence["axis_regularity_projection"]["projected_node_count"] == 2
    assert evidence["axis_regularity_projection"]["source_abs_max_m_per_s"] == 5.0
    assert evidence["coordinate_provenance"]["conversion_factor_cm_to_m"] == 0.01
    assert evidence["coordinate_nudge"] is False
    assert evidence["missing_value_imputation"] is False
