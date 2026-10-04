from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from particle_tracer_unified.comsol_case.cli import main
from particle_tracer_unified.comsol_case.contracts import validate_raw_export
from particle_tracer_unified.comsol_case.fields import build_profile_field_bundle
from particle_tracer_unified.comsol_case.profiles import BUILD_PROFILES
from particle_tracer_unified.io.comsol_manifest import ComsolCaseManifest


def _samples() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for r in (0.0, 50.0, 100.0):
        for z in (0.0, 50.0, 100.0):
            rows.append(
                {
                    "r": r,
                    "z": z,
                    "valid_mask": 1,
                    "ux": r + 2.0 * z,
                    "uy": r - z,
                    "mu": 1.8e-5,
                    "E_x": 2.0 * r + z,
                    "E_y": r - 3.0 * z,
                    "T": 300.0 + z,
                    "ne": 1.0e16 + r,
                }
            )
    return pd.DataFrame(rows)


def _write_square_mesh(path: Path) -> Path:
    path.write_text(
        """2 # sdim
4 # number of mesh vertices
# Mesh vertex coordinates
0 0
100 0
100 100
0 100
2 # number of element types
3 edg # type name
2 # number of vertices per element
4 # number of elements
# Elements
0 1
1 2
2 3
3 0
4 # number of geometric entity indices
# Geometric entity indices
0
1
2
3
4 quad # type name
4 # number of vertices per element
1 # number of elements
# Elements
0 1 2 3
1 # number of geometric entity indices
# Geometric entity indices
0
""",
        encoding="utf-8",
    )
    return path


def _write_boundaries(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "part_id": part_id,
                "part_name": f"icp_wall_{part_id}",
                "comsol_entity_id": part_id,
                "role": "wall",
                "wall_law": "specular",
                "wall_stick_probability": 0.0,
                "wall_restitution": 1.0,
                "wall_diffuse_fraction": 0.0,
                "wall_critical_sticking_velocity_mps": 0.0,
                "material_id": part_id,
                "material_name": "reviewed_material",
                "metadata_json": "{}",
            }
            for part_id in (1, 2, 3, 4)
        ]
    ).to_csv(path, index=False)
    return path


def _write_release(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "release_time_s": 0.0,
                "r_m": 0.5,
                "z_m": 0.5,
                "vr_mps": 0.0,
                "vz_mps": 0.0,
                "mass_kg": 1.0e-15,
                "drag_diameter_m": 1.0e-6,
                "charge_C": -1.0e-17,
                "source_part_id": 1,
                "density_kgm3": 1200.0,
            }
        ]
    ).to_csv(path, index=False)
    return path


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_export_contract_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    mesh_path = _write_square_mesh(tmp_path / "mesh.mphtxt")
    samples_path = tmp_path / "field_samples.csv"
    _samples()[["r", "z", "valid_mask", "ux", "uy", "mu", "E_x", "E_y"]].to_csv(
        samples_path, index=False
    )
    payload: dict[str, Any] = {
        "source_kind": "comsol_java_api_external_export",
        "model_name": "model",
        "study": "std1",
        "dataset": "dset1",
        "solution": "sol1",
        "solution_number": 1,
        "comsol_version": "6.4",
        "mesh_tag": "mesh1",
        "parameter_name": "Vrf",
        "parameter_value": "20[V]",
        "geometry_model_unit": "cm",
        "geometry_scale_m_per_model_unit": 0.01,
        "solver_coordinate_unit": "m",
        "vacuum_domain_ids": [1],
        "mph_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "mesh_sha256": _file_digest(mesh_path),
        "field_samples_sha256": _file_digest(samples_path),
        "expression_mapping": {
            "ux": "u",
            "uy": "w",
            "mu": "spf.mu",
            "E_x": "-d(V,r)",
            "E_y": "-d(V,z)",
        },
        "expression_units": {
            "ux": "m/s",
            "uy": "m/s",
            "mu": "Pa*s",
            "E_x": "V/m",
            "E_y": "V/m",
        },
    }
    manifest_path = tmp_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path, manifest_path, payload


def test_icp_profile_converts_export_grid_to_si_without_reference_particle_force(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "field_samples.csv"
    _samples().to_csv(samples, index=False)

    bundle = build_profile_field_bundle(
        samples,
        tmp_path / "field.npz",
        profile=BUILD_PROFILES["icp_cf4_o2"],
        coordinate_scale_m_per_model_unit=0.01,
    )

    with np.load(bundle) as payload:
        np.testing.assert_allclose(payload["axis_0"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(payload["axis_1"], [0.0, 0.5, 1.0])
        assert {"ux", "uy", "mu", "E_x", "E_y", "T", "ne"}.issubset(payload.files)
        assert "ax" not in payload.files
        assert "ay" not in payload.files
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        assert metadata["profile"] == "icp_cf4_o2"
        assert metadata["artifact_coordinate_unit"] == "m"


def test_icp_profile_rejects_incomplete_export_columns(tmp_path: Path) -> None:
    samples = _samples().drop(columns=["E_y"])
    path = tmp_path / "field_samples.csv"
    samples.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        build_profile_field_bundle(
            path,
            tmp_path / "field.npz",
            profile=BUILD_PROFILES["icp_cf4_o2"],
            coordinate_scale_m_per_model_unit=0.01,
        )


def test_icp_profile_is_integrated_in_generic_builder_cli(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    mesh_path = _write_square_mesh(raw / "mesh.mphtxt")
    samples_path = raw / "field_samples.csv"
    _samples().to_csv(samples_path, index=False)
    (raw / "export_manifest.json").write_text(
        json.dumps(
            {
                "source_kind": "comsol_java_api_external_export",
                "model_name": "icp-cf4-o2",
                "study": "std1",
                "dataset": "dset3",
                "solution": "sol2",
                "solution_number": 17,
                "comsol_version": "6.4.0.123",
                "mesh_tag": "mesh1",
                "parameter_name": "Vrf",
                "parameter_value": "20[V]",
                "geometry_model_unit": "cm",
                "geometry_scale_m_per_model_unit": 0.01,
                "solver_coordinate_unit": "m",
                "vacuum_domain_ids": [1],
                "mph_sha256": "1" * 64,
                "config_sha256": "2" * 64,
                "mesh_sha256": _file_digest(mesh_path),
                "field_samples_sha256": _file_digest(samples_path),
                "expression_mapping": {
                    "ux": "u",
                    "uy": "w",
                    "mu": "spf.mu",
                    "E_x": "-d(ptp.Vav,r)",
                    "E_y": "-d(ptp.Vav,z)",
                    "T": "ht.T",
                    "ne": "ptp.neav",
                },
                "expression_units": {
                    "ux": "m/s",
                    "uy": "m/s",
                    "mu": "Pa*s",
                    "E_x": "V/m",
                    "E_y": "V/m",
                    "T": "K",
                    "ne": "1/m^3",
                },
            }
        ),
        encoding="utf-8",
    )
    release = _write_release(tmp_path / "release.csv")
    boundaries = _write_boundaries(tmp_path / "boundaries_input.csv")
    out = tmp_path / "case"

    assert (
        main(
            [
                "--profile",
                "icp_cf4_o2",
                "--raw-export-dir",
                str(raw),
                "--release-table",
                str(release),
                "--boundaries",
                str(boundaries),
                "--out-dir",
                str(out),
                "--diagnostic-grid-spacing-m",
                "0.5",
                "--dt-s",
                "0.01",
                "--t-end-s",
                "0.2",
                "--drag-law",
                "stokes",
                "--force",
                "electric",
                "--gas-dynamic-viscosity-Pas",
                "1.8e-5",
            ]
        )
        == 0
    )
    config = yaml.safe_load((out / "run_config.yaml").read_text(encoding="utf-8"))
    manifest_payload = yaml.safe_load(
        (out / "comsol_manifest.yaml").read_text(encoding="utf-8")
    )
    assert config["case"] == {
        "spatial_dim": 2,
        "coordinate_system": "axisymmetric_rz",
        "adapter": "comsol",
    }
    assert config["inputs"] == {"comsol_manifest": "comsol_manifest.yaml"}
    assert config["time"] == {"dt": 0.01, "t_end": 0.2}
    assert manifest_payload["model"]["dataset"] == "dset3"
    assert manifest_payload["coordinates"]["axis_order"] == ["r", "z"]
    assert manifest_payload["fields"]["velocity"]["components"] == {
        "r": "ux",
        "z": "uy",
    }
    assert manifest_payload["fields"]["dynamic_viscosity"]["components"] == {
        "value": "mu"
    }
    assert manifest_payload["fields"]["ne"] == {
        "artifact": "field",
        "components": {"value": "ne"},
        "unit": "1/m^3",
        "scale_to_si": 1.0,
    }
    assert all(
        not ({"name", "physical_quantity", "mesh", "interpolation"} & set(field_spec))
        for field_spec in manifest_payload["fields"].values()
    )
    assert manifest_payload["metadata"]["profile"] == "icp_cf4_o2"
    assert manifest_payload["metadata"]["raw_export_manifest_sha256"]
    assert manifest_payload["metadata"]["source_solution_number"] == 17
    assert manifest_payload["metadata"]["vacuum_domain_ids"] == [1]
    assert manifest_payload["metadata"]["source_comsol_version"] == "6.4.0.123"
    assert manifest_payload["metadata"]["source_parameter"] == {
        "name": "Vrf",
        "value": "20[V]",
    }
    assert manifest_payload["metadata"]["source_expression_units"]["E_x"] == "V/m"
    assert (
        ComsolCaseManifest.load(out / "comsol_manifest.yaml").validate(strict=True)
        == []
    )


def test_model_specific_java_exporter_remains_in_external_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (
        root / "external" / "comsol_icp_export" / "java" / "IcpCf4O2SiEtchExporter.java"
    ).is_file()


def test_model_specific_java_exporter_validates_saved_solution_provenance() -> None:
    java = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "comsol_icp_export"
        / "java"
        / "IcpCf4O2SiEtchExporter.java"
    ).read_text(encoding="utf-8")

    assert 'call(configuredDataset, "getString", "solution")' in java
    assert 'call(configuredSolution, "study")' in java
    assert 'call(configuredSolution, "getPNames")' in java
    assert 'call(configuredSolution, "getPVals", solutionNumber)' in java
    assert 'call(parameters, "evaluate", expectedExpression)' in java
    assert "setModelParameter" not in java
    assert 'call(call(model, "result"), "run")' not in java


def _verify_model_specific_export_has_no_implicit_fallbacks() -> None:
    root = Path(__file__).resolve().parents[1]
    java = (
        root / "external" / "comsol_icp_export" / "java" / "IcpCf4O2SiEtchExporter.java"
    ).read_text(encoding="utf-8")
    config = json.loads(
        (
            root / "external" / "comsol_icp_export" / "config" / "icp_cf4_o2_v20.json"
        ).read_text(encoding="utf-8")
    )

    assert "new int[]{17}" not in java
    assert "selectedDatasets" not in java
    assert "expression_dataset" not in java
    assert all(len(candidates) == 1 for candidates in config["expressions"].values())
    assert set(config["expressions"]) == set(config["units"])
    assert config["study"] == "std2"
    assert config["dataset"] == "dset3"
    assert config["solution"] == "sol2"
    assert config["solution_number"] == 1
    assert (
        config["vacuum_domain_ids"] == []
    )  # Must be populated after reviewing COMSOL domain IDs.


test_model_specific_export_contract_has_no_implicit_solution_or_expression_fallback = (
    _verify_model_specific_export_has_no_implicit_fallbacks
)


def test_raw_export_contract_returns_stable_normalized_schema(tmp_path: Path) -> None:
    raw_dir, manifest_path, payload = _raw_export_contract_inputs(tmp_path)

    normalized = validate_raw_export(
        raw_dir,
        manifest_path,
        payload,
        profile=BUILD_PROFILES["icp_cf4_o2"],
    )

    assert normalized == {
        "model_name": "model",
        "study": "std1",
        "dataset": "dset1",
        "solution": "sol1",
        "comsol_version": "6.4",
        "mesh_tag": "mesh1",
        "parameter_name": "Vrf",
        "parameter_value": "20[V]",
        "geometry_model_unit": "cm",
        "solution_number": 1,
        "geometry_scale_m_per_model_unit": 0.01,
        "vacuum_domain_ids": (1,),
        "expression_mapping": payload["expression_mapping"],
        "expression_units": payload["expression_units"],
        "mesh_sha256": payload["mesh_sha256"],
        "field_samples_sha256": payload["field_samples_sha256"],
        "mph_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "manifest_sha256": _file_digest(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
    }


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        (
            {"source_kind": "other", "model_name": ""},
            "raw export manifest source_kind must be comsol_java_api_external_export",
        ),
        (
            {"model_name": " model ", "solution_number": 0},
            "raw export manifest model_name must be a non-empty canonical string",
        ),
        (
            {"solution_number": True, "geometry_scale_m_per_model_unit": 0.0},
            "raw export manifest solution_number must be a positive integer",
        ),
        (
            {
                "geometry_scale_m_per_model_unit": float("nan"),
                "solver_coordinate_unit": "cm",
            },
            "raw export manifest geometry_scale_m_per_model_unit must be "
            "positive and finite",
        ),
        (
            {"solver_coordinate_unit": "cm", "vacuum_domain_ids": "1"},
            "raw export manifest solver_coordinate_unit must be exactly m",
        ),
        (
            {"vacuum_domain_ids": "1", "expression_mapping": []},
            "raw export manifest vacuum_domain_ids must be a non-empty integer list",
        ),
        (
            {"vacuum_domain_ids": [1, True], "expression_mapping": []},
            "raw export manifest vacuum_domain_ids must contain positive integers",
        ),
        (
            {"vacuum_domain_ids": [1, 1], "expression_mapping": []},
            "raw export manifest vacuum_domain_ids must be non-empty and unique",
        ),
        (
            {"expression_mapping": [], "mesh_sha256": "invalid"},
            "raw export manifest requires expression_mapping and "
            "expression_units mappings",
        ),
        (
            {
                "expression_mapping": {"unexpected": "value"},
                "expression_units": {"unexpected": "m/s"},
                "mesh_sha256": "invalid",
            },
            "raw export manifest contains quantities not declared by the selected "
            "profile: ['unexpected']",
        ),
        (
            {"mesh_sha256": "invalid", "field_samples_sha256": "invalid"},
            "raw export manifest mesh_sha256 must be a lowercase SHA-256 hex digest",
        ),
    ],
)
def test_raw_export_contract_preserves_validation_order_and_messages(
    tmp_path: Path,
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    raw_dir, manifest_path, payload = _raw_export_contract_inputs(tmp_path)

    with pytest.raises(ValueError, match=re.escape(expected_message)) as captured:
        validate_raw_export(
            raw_dir,
            manifest_path,
            {**payload, **overrides},
            profile=BUILD_PROFILES["icp_cf4_o2"],
        )

    assert str(captured.value) == expected_message


def test_raw_export_contract_checks_sample_schema_before_source_digests(
    tmp_path: Path,
) -> None:
    raw_dir, manifest_path, payload = _raw_export_contract_inputs(tmp_path)
    samples_path = raw_dir / "field_samples.csv"
    samples = pd.read_csv(samples_path)
    samples["unexpected"] = 1.0
    samples.to_csv(samples_path, index=False)
    payload["field_samples_sha256"] = _file_digest(samples_path)
    payload["mph_sha256"] = "invalid"
    expected_message = (
        "raw field sample quantities must exactly match export manifest "
        "expression_mapping: samples_only=['unexpected'], manifest_only=[]"
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)) as captured:
        validate_raw_export(
            raw_dir,
            manifest_path,
            payload,
            profile=BUILD_PROFILES["icp_cf4_o2"],
        )

    assert str(captured.value) == expected_message


def test_raw_export_contract_rejects_tampered_artifact_and_wrong_unit(
    tmp_path: Path,
) -> None:
    raw_dir, manifest, payload = _raw_export_contract_inputs(tmp_path)

    tampered = dict(payload)
    tampered["mesh_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"artifact hash mismatch for mesh\.mphtxt"):
        validate_raw_export(
            raw_dir,
            manifest,
            tampered,
            profile=BUILD_PROFILES["icp_cf4_o2"],
        )

    wrong_unit = dict(payload)
    wrong_unit["expression_units"] = {**payload["expression_units"], "E_x": "V"}
    with pytest.raises(ValueError, match="expression units do not match"):
        validate_raw_export(
            raw_dir,
            manifest,
            wrong_unit,
            profile=BUILD_PROFILES["icp_cf4_o2"],
        )
