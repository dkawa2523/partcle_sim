from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from particle_tracer_unified.io.comsol_manifest import ComsolCaseManifest


def _write_support_files(root: Path) -> None:
    (root / "release.csv").write_text(
        "particle_id,release_time,x,y,vx,vy,mass,diameter,density,charge\n"
        "1,0,0,0,0,0,1e-18,1e-6,1000,0\n",
        encoding="utf-8",
    )
    (root / "boundary.csv").write_text(
        "solver_part_id,comsol_geom_entity_id,selection_name,boundary_type,wall_node,material\n"
        "1,10,wall,wall,pt.wall1,mat\n",
        encoding="utf-8",
    )
    (root / "walls.csv").write_text(
        "solver_part_id,wall_type,stick_probability,restitution_n,restitution_t,diffuse_temperature,material_id\n"
        "1,stick,1,0,0,,mat\n",
        encoding="utf-8",
    )


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": {"study": "std1", "dataset": "dset1", "solution": "sol1"},
        "coordinates": {
            "coordinate_system": "cartesian_xy",
            "coordinate_scale_m_per_model_unit": 0.01,
        },
        "fields": [
            {
                "name": "u",
                "physical_quantity": "velocity",
                "components": {"x": "ux", "y": "uy"},
            }
        ],
        "particles": {"release_table": "release.csv"},
        "boundaries": {"map_file": "boundary.csv", "wall_law_file": "walls.csv"},
        "forces": [{"solver_force": "drag", "enabled": True, "law": "stokes_cunningham"}],
    }


def test_comsol_manifest_validates_required_faithful_metadata(tmp_path: Path) -> None:
    _write_support_files(tmp_path)
    manifest_path = tmp_path / "comsol_case_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest_payload()), encoding="utf-8")

    manifest = ComsolCaseManifest.load(manifest_path)

    assert manifest.coordinate_system == "cartesian_xy"
    assert manifest.coordinate_scale_m_per_model_unit == pytest.approx(0.01)
    assert manifest.release_velocity_scale_mps_per_input_unit == pytest.approx(1.0)
    assert manifest.validate(strict=True) == []


def test_comsol_manifest_validates_optional_release_velocity_scale(tmp_path: Path) -> None:
    _write_support_files(tmp_path)
    payload = _manifest_payload()
    payload["particles"] = {
        "release_table": "release.csv",
        "release_velocity_scale_mps_per_input_unit": 0.01,
    }
    manifest_path = tmp_path / "comsol_case_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    manifest = ComsolCaseManifest.load(manifest_path)

    assert manifest.release_velocity_scale_mps_per_input_unit == pytest.approx(0.01)
    assert manifest.validate(strict=True) == []


def test_comsol_manifest_rejects_invalid_release_velocity_scale(tmp_path: Path) -> None:
    _write_support_files(tmp_path)
    payload = _manifest_payload()
    payload["particles"] = {
        "release_table": "release.csv",
        "release_velocity_scale_mps_per_input_unit": 0.0,
    }
    manifest_path = tmp_path / "comsol_case_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    manifest = ComsolCaseManifest.load(manifest_path)

    with pytest.raises(ValueError, match="release_velocity_scale_mps_per_input_unit"):
        manifest.validate(strict=True)


def test_comsol_manifest_rejects_missing_coordinate_scale(tmp_path: Path) -> None:
    _write_support_files(tmp_path)
    payload = _manifest_payload()
    payload["coordinates"] = {"coordinate_system": "cartesian_xy"}
    manifest_path = tmp_path / "comsol_case_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    manifest = ComsolCaseManifest.load(manifest_path)

    with pytest.raises(ValueError, match="coordinate_scale_m_per_model_unit"):
        manifest.validate(strict=True)


def test_comsol_manifest_rejects_missing_force_inventory(tmp_path: Path) -> None:
    _write_support_files(tmp_path)
    payload = _manifest_payload()
    payload["forces"] = []
    manifest_path = tmp_path / "comsol_case_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    manifest = ComsolCaseManifest.load(manifest_path)

    with pytest.raises(ValueError, match="forces must list"):
        manifest.validate(strict=True)
