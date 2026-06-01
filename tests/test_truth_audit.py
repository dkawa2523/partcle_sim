from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "comsol_particle_export"
sys.path.insert(0, str(EXTERNAL))

from comsol_particle_export.truth_audit import build_truth_audit  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_release(path: Path, *, dx: float = 0.0, full: bool = True) -> None:
    rows = [
        {
            "particle_id": 1,
            "release_time": 0.0,
            "x": 1.0e-3 + dx,
            "y": 2.0e-3,
            "v_x": 0.0,
            "v_y": 0.0,
        },
        {
            "particle_id": 2,
            "release_time": 0.05,
            "x": 1.5e-3 + dx,
            "y": 2.5e-3,
            "v_x": 0.0,
            "v_y": 0.0,
        },
    ]
    if full:
        rows[0].update(
            {
                "mass": 1.0e-12,
                "diameter": 1.0e-5,
                "density": 2200.0,
                "charge": 0.0,
                "source_part_id": 1,
            }
        )
        rows[1].update(
            {
                "mass": 1.0e-12,
                "diameter": 1.0e-5,
                "density": 2200.0,
                "charge": 0.0,
                "source_part_id": 5,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_truth_audit_flags_micromixer_parity_gaps(tmp_path: Path) -> None:
    field_raw = tmp_path / "field_raw"
    particle_raw = tmp_path / "particle_raw"
    case_dir = tmp_path / "case"
    generated = case_dir / "generated"
    field_raw.mkdir()
    particle_raw.mkdir()
    generated.mkdir(parents=True)

    _write_json(
        field_raw / "export_manifest.json",
        {
            "mph_path": "data/micromixer_particle_tracing.mph",
            "mph_sha256": "abc123",
            "axis_names": ["x", "y"],
            "coordinate_model_unit": "mm",
            "coordinate_scale_m_per_model_unit": 0.001,
            "dataset": "dset1",
            "mesh_tag": "mesh1",
            "field_sample_context_count": 1,
        },
    )
    _write_json(
        particle_raw / "export_manifest.json",
        {
            "mph_path": "data/micromixer_particle_tracing.mph",
            "mph_sha256": "abc123",
            "data_export_dataset": "part1",
        },
    )
    _write_json(
        particle_raw / "comsol_particle_trajectory_report.json",
        {
            "metadata": {"Version": "COMSOL 6.4"},
            "trajectory_time_count": 3,
            "time_min_s": 0.0,
            "time_max_s": 2.0,
        },
    )
    _write_json(
        field_raw / "physics_feature_inventory.json",
        {
            "features": [
                {
                    "physics_tag": "spf",
                    "feature_tag": "rot1",
                    "type": "RotatingFrame",
                    "property_values": {"omega": "2*pi*t"},
                },
                {
                    "physics_tag": "fpt",
                    "physics_label": "Particle Tracing for Fluid Flow",
                    "feature_tag": "df1",
                    "label": "Drag Force",
                    "type": "DragForce",
                    "property_values": {
                        "DragLaw": "Stokes",
                        "Rarefaction_Effects": "CunninghamMillikanDavies",
                        "IncludeVirtualMassAndPressureGradientForces": "1",
                        "u_src": "Velocity field",
                        "mu_mat": "From material",
                        "rho_mat": "From material",
                    },
                },
                {
                    "physics_tag": "fpt",
                    "physics_label": "Particle Tracing for Fluid Flow",
                    "feature_tag": "pc1",
                    "label": "Pair Continuity",
                    "type": "PairContinuity",
                    "selection_entities": [10],
                    "property_values": {},
                },
                {
                    "physics_tag": "fpt",
                    "physics_label": "Particle Tracing for Fluid Flow",
                    "feature_tag": "out1",
                    "label": "Outlet",
                    "type": "Outlet",
                    "selection_entities": [20],
                    "property_values": {"WallCondition": "Freeze"},
                },
                {
                    "physics_tag": "fpt",
                    "physics_label": "Particle Tracing for Fluid Flow",
                    "feature_tag": "wall1",
                    "label": "Wall",
                    "type": "Wall",
                    "selection_entities": [30],
                    "property_values": {"WallCondition": "Bounce", "e": "1"},
                },
            ]
        },
    )
    _write_json(
        field_raw / "particle_release_inventory.json",
        {
            "features": [
                {
                    "physics_tag": "fpt",
                    "feature_tag": "pp1",
                    "label": "Particle Properties",
                    "type": "ParticleProperties",
                    "release_kind": "particle_properties",
                    "known_settings": {"diameter": "10[um]", "density": "2200[kg/m^3]"},
                },
                {
                    "physics_tag": "fpt",
                    "feature_tag": "rel1",
                    "label": "Inlet",
                    "type": "Inlet",
                    "release_kind": "release",
                    "selection_entities": [1, 5],
                    "known_settings": {"tlist": "range(0,0.05,1)", "v0": "0"},
                },
            ]
        },
    )

    _write_release(particle_raw / "comsol_release_particles.csv", full=False)
    _write_release(case_dir / "particles.csv")
    _write_release(case_dir / "particles_inward_clean.csv", dx=1.0e-5)
    pd.DataFrame(
        [
            {"comsol_api_selection_entity_id": 10, "solver_part_id": 100},
            {"comsol_api_selection_entity_id": 20, "solver_part_id": 200},
            {"comsol_api_selection_entity_id": 30, "solver_part_id": 300},
        ]
    ).to_csv(generated / "comsol_boundary_entity_mapping.csv", index=False)
    _write_json(
        generated / "exact_compare_input_contract_summary.json",
        {
            "passed": False,
            "status_counts": {"clean": 1, "mixed_stencil": 1, "hard_invalid": 0, "non_clean": 1},
            "non_clean_near_boundary_count": 1,
            "geometry_inside_violation_count": 1,
        },
    )
    pd.DataFrame(
        [
            {"part_id": 100, "wall_law": "pass_through"},
            {"part_id": 200, "wall_law": "stick"},
            {"part_id": 300, "wall_law": "specular"},
        ]
    ).to_csv(case_dir / "part_walls.csv", index=False)
    pd.DataFrame([{"material_id": 1, "material_name": "synthetic"}]).to_csv(case_dir / "materials.csv", index=False)
    run_config = case_dir / "run_config_inward_clean_trend_mesh.yaml"
    run_config.write_text("solver:\n  forces:\n    drag:\n      model: stokes\n", encoding="utf-8")

    summary = build_truth_audit(
        case_name="synthetic_micromixer",
        field_raw_dir=field_raw,
        particle_raw_dir=particle_raw,
        solver_case_dir=case_dir,
        out_dir=tmp_path / "audit",
        run_config=run_config,
        compare_field_replay=False,
    )

    manifest = json.loads((tmp_path / "audit" / "micromixer_truth_manifest.json").read_text(encoding="utf-8"))
    assert Path(summary["truth_manifest_yaml"]).exists()
    assert manifest["field_truth"]["preferred_backend"] == "triangle_mesh_2d"
    assert manifest["cleanup_policy"]["inward_clean_is_comsol_truth"] is False
    assert "COMSOL particle status/stop-time export is missing; boundary-event parity cannot be direct" in manifest[
        "missing_comsol_exports"
    ]
    assert (
        "COMSOL wall-hit entity/normal export is unavailable; fpt.st/fpt.fs only provide particle stop-time/status"
        in manifest["missing_comsol_exports"]
    )
    assert "time-resolved mesh field export is required before claiming time-varying field parity" in manifest["missing_comsol_exports"]
    assert "source" in summary["release_contract_missing"]
    assert "mass" in summary["release_contract_missing"]
    assert "COMSOL uses Cunningham-Millikan-Davies rarefaction but solver config is not stokes_cunningham" in manifest["forces"]["parity_gaps"]
    assert (
        "COMSOL enables virtual mass/pressure-gradient forces; missing enabled solver contribution(s): "
        "virtual_mass, pressure_gradient"
    ) in manifest["forces"]["parity_gaps"]
    assert manifest["particles"]["exact_release_alignment"]["release_position_error_m"]["max"] == pytest.approx(0.0)
    assert manifest["particles"]["inward_clean_release_alignment"]["release_position_error_m"]["max"] == pytest.approx(1.0e-5)
    assert manifest["boundaries"]["boundary_role_alignment"]["mismatch_count"] == 0
    request_summary = Path(summary["reextract_request_summary"])
    assert request_summary.exists()
    requests = json.loads(request_summary.read_text(encoding="utf-8"))
    assert requests["request_count"] == 29
    assert requests["runnable_config_count"] == 27
    assert Path(requests["run_script"]).exists()
    time_config = json.loads(
        (request_summary.parent / "mesh_field_time_resolved_config.json").read_text(encoding="utf-8")
    )
    assert time_config["export_grid_field_samples"] is False
    assert time_config["export_mesh_field_samples"] is True
    assert time_config["time_values"] == pytest.approx([0.0, 1.0, 2.0])
    assert manifest["particles"]["release_table_contract"]["comsol_release"]["exact_parity_ready"] is False
    assert manifest["parity_readiness"]["ready_for_exact_solver_comparison"] is False
    assert manifest["parity_readiness"]["blocker_count"] >= 1
    backlog_path = Path(summary["solver_improvement_backlog"])
    backlog = json.loads(backlog_path.read_text(encoding="utf-8"))
    assert [item["category"] for item in backlog["items"]] == ["force_model", "initial_support"]
    assert {item["category"] for item in backlog["excluded_from_solver_code_backlog"]} >= {
        "missing_comsol_truth",
        "force_configuration",
    }


def test_truth_audit_rejects_implicit_coordinate_scale(tmp_path: Path) -> None:
    field_raw = tmp_path / "field_raw"
    particle_raw = tmp_path / "particle_raw"
    case_dir = tmp_path / "case"
    field_raw.mkdir()
    particle_raw.mkdir()
    case_dir.mkdir()
    _write_json(
        field_raw / "export_manifest.json",
        {
            "axis_names": ["x", "y"],
            "coordinate_model_unit": "mm",
            "dataset": "dset1",
            "mesh_tag": "mesh1",
        },
    )
    _write_json(particle_raw / "export_manifest.json", {"data_export_dataset": "part1"})
    _write_json(particle_raw / "comsol_particle_trajectory_report.json", {})

    with pytest.raises(ValueError, match="coordinate_scale_m_per_model_unit"):
        build_truth_audit(
            case_name="synthetic_micromixer",
            field_raw_dir=field_raw,
            particle_raw_dir=particle_raw,
            solver_case_dir=case_dir,
            out_dir=tmp_path / "audit",
            compare_field_replay=False,
        )


def test_truth_audit_uses_schema_based_wall_event_detection(tmp_path: Path) -> None:
    field_raw = tmp_path / "field_raw"
    particle_raw = tmp_path / "particle_raw"
    case_dir = tmp_path / "case"
    generated = case_dir / "generated"
    field_raw.mkdir()
    particle_raw.mkdir()
    generated.mkdir(parents=True)
    _write_json(
        field_raw / "export_manifest.json",
        {
            "mph_path": "data/micromixer_particle_tracing.mph",
            "axis_names": ["x", "y"],
            "coordinate_model_unit": "mm",
            "coordinate_scale_m_per_model_unit": 0.001,
            "dataset": "dset1",
            "mesh_tag": "mesh1",
            "field_sample_context_count": 1,
        },
    )
    _write_json(particle_raw / "export_manifest.json", {"data_export_dataset": "part1"})
    _write_json(particle_raw / "comsol_particle_trajectory_report.json", {"trajectory_time_count": 1})
    _write_json(field_raw / "physics_feature_inventory.json", {"features": []})
    _write_json(field_raw / "particle_release_inventory.json", {"features": []})
    _write_release(particle_raw / "comsol_release_particles.csv", full=True)
    _write_release(case_dir / "particles.csv")
    pd.DataFrame([{"particle_id": 1, "hit_time_s": 0.2, "comsol_entity_id": 10, "outcome": "bounce"}]).to_csv(
        particle_raw / "renamed_export.csv",
        index=False,
    )

    summary = build_truth_audit(
        case_name="synthetic_micromixer",
        field_raw_dir=field_raw,
        particle_raw_dir=particle_raw,
        solver_case_dir=case_dir,
        out_dir=tmp_path / "audit",
        compare_field_replay=False,
    )
    manifest = json.loads((tmp_path / "audit" / "micromixer_truth_manifest.json").read_text(encoding="utf-8"))
    assert str(particle_raw / "renamed_export.csv") in manifest["boundaries"]["wall_event_export_candidates"]
    assert "COMSOL wall-hit entity/normal export is unavailable; fpt.st/fpt.fs only provide particle stop-time/status" not in summary[
        "missing_comsol_exports"
    ]
