from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "comsol_particle_export"
sys.path.insert(0, str(EXTERNAL))

from comsol_particle_export.validate_export import validate_raw_export  # noqa: E402
from comsol_particle_export.compare_particle_results import compare_particle_results  # noqa: E402


def _write_minimal_export(raw: Path) -> Path:
    raw.mkdir()
    config = raw / "config.json"
    config.write_text(
        json.dumps(
            {
                "case_name": "synthetic",
                "spatial_dim": 2,
                "axis_names": ["r", "z"],
                "required": ["ux", "uy", "mu", "E_x", "E_y"],
                "force_models": {
                    "thermophoresis": {"enabled": False},
                    "dielectrophoresis": {"enabled": False},
                    "lift": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )
    (raw / "model_inventory.json").write_text(
        json.dumps({"source_kind": "external_comsol_particle_export_inventory"}),
        encoding="utf-8",
    )
    (raw / "export_manifest.json").write_text(
        json.dumps(
            {
                "source_kind": "external_comsol_particle_export",
                "case_name": "synthetic",
                "axis_names": ["r", "z"],
            }
        ),
        encoding="utf-8",
    )
    (raw / "expression_inventory.json").write_text(
        json.dumps(
            {
                "selected": {
                    "ux": {"available": True, "expression": "u"},
                    "uy": {"available": True, "expression": "v"},
                    "mu": {"available": True, "expression": "mu"},
                    "rho_g": {"available": True, "expression": "rho"},
                    "T": {"available": True, "expression": "T"},
                    "E_x": {"available": True, "expression": "Ex"},
                    "E_y": {"available": True, "expression": "Ey"},
                }
            }
        ),
        encoding="utf-8",
    )
    (raw / "particle_release_inventory.json").write_text(
        json.dumps(
            {
                "features": [
                    {
                        "component_tag": "comp1",
                        "physics_tag": "fpt",
                        "feature_tag": "rel1",
                        "label": "Release from Grid",
                        "type": "ReleaseGrid",
                        "release_kind": "release_grid",
                        "selection_entities": [10],
                        "known_settings": {"tlist": "range(0,1e-6,1e-5)", "Nx": "2", "Ny": "3"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for r in [0.0, 1.0]:
        for z in [0.0, 1.0, 2.0]:
            rows.append(
                {
                    "r": r,
                    "z": z,
                    "valid_mask": 1,
                    "ux": r + z,
                    "uy": r - z,
                    "mu": 1.8e-5,
                    "rho_g": 1.0 + 0.1 * r,
                    "T": 300.0 + 10.0 * z,
                    "E_x": 10.0 + r,
                    "E_y": -5.0 + z,
                }
            )
    pd.DataFrame(rows).to_csv(raw / "field_samples.csv", index=False)
    return config


def test_validate_raw_export_accepts_complete_tensor_grid(tmp_path: Path) -> None:
    config = _write_minimal_export(tmp_path / "raw")

    summary = validate_raw_export(tmp_path / "raw", config)

    assert summary["case_name"] == "synthetic"
    assert summary["files"]["field_samples.csv"] is True
    assert summary["field_samples"]["row_count"] == 6
    assert summary["field_samples"]["valid_node_count"] == 6
    assert summary["field_samples"]["axes"]["r"]["count"] == 2
    assert summary["field_samples"]["axes"]["z"]["count"] == 3
    assert summary["field_samples"]["required_fields"]["E_y"]["variation"] == pytest.approx(2.0)
    assert summary["field_samples"]["selected_fields"]["T"]["variation"] == pytest.approx(20.0)
    assert summary["particle_release_inventory"]["feature_count"] == 1
    assert summary["particle_release_inventory"]["time_dependent_feature_count"] == 1


def test_validate_raw_export_rejects_missing_required_expression(tmp_path: Path) -> None:
    config = _write_minimal_export(tmp_path / "raw")
    inventory = json.loads((tmp_path / "raw" / "expression_inventory.json").read_text(encoding="utf-8"))
    inventory["selected"]["E_y"]["available"] = False
    (tmp_path / "raw" / "expression_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="required expression"):
        validate_raw_export(tmp_path / "raw", config)


def test_validate_raw_export_rejects_incomplete_grid(tmp_path: Path) -> None:
    config = _write_minimal_export(tmp_path / "raw")
    table = pd.read_csv(tmp_path / "raw" / "field_samples.csv").iloc[:-1]
    table.to_csv(tmp_path / "raw" / "field_samples.csv", index=False)

    with pytest.raises(ValueError, match="complete tensor grid"):
        validate_raw_export(tmp_path / "raw", config)


def test_validate_raw_export_rejects_enabled_force_with_missing_field(tmp_path: Path) -> None:
    config = _write_minimal_export(tmp_path / "raw")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["force_models"]["thermophoresis"]["enabled"] = True
    inventory = json.loads((tmp_path / "raw" / "expression_inventory.json").read_text(encoding="utf-8"))
    inventory["selected"]["T"]["available"] = False
    config.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "raw" / "expression_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="enabled force model"):
        validate_raw_export(tmp_path / "raw", config)


def test_compare_particle_results_writes_minimal_metrics(tmp_path: Path) -> None:
    solver = tmp_path / "solver"
    solver.mkdir()
    (solver / "collision_diagnostics.json").write_text(
        json.dumps(
            {
                "force_runtime": {
                    "thermophoresis_enabled": 1,
                    "thermophoresis_model": "talbot",
                    "dielectrophoresis_enabled": 1,
                    "dielectrophoresis_model": "dc",
                    "lift_enabled": 0,
                    "gravity_buoyancy_enabled": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    solver_particles = tmp_path / "solver_particles.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "release_time": 0.0, "x": 0.10, "y": 0.20, "v_x": 1.0, "v_y": 0.0, "source_part_id": 10},
            {"particle_id": 2, "release_time": 1.0e-6, "x": 0.30, "y": 0.40, "v_x": 0.0, "v_y": 2.0, "source_part_id": 20},
        ]
    ).to_csv(solver_particles, index=False)
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "active": 0,
                "stuck": 1,
                "absorbed": 0,
                "escaped": 0,
                "invalid_mask_stopped": 0,
                "numerical_boundary_stopped": 0,
                "x": 0.10,
                "y": 0.20,
                "v_x": 1.0,
                "v_y": 0.0,
                "charge_C": -1.0e-18,
            },
            {
                "particle_id": 2,
                "active": 0,
                "stuck": 0,
                "absorbed": 1,
                "escaped": 0,
                "invalid_mask_stopped": 0,
                "numerical_boundary_stopped": 0,
                "x": 0.30,
                "y": 0.40,
                "v_x": 0.0,
                "v_y": 2.0,
                "charge_C": -2.0e-18,
            },
        ]
    ).to_csv(solver / "final_particles.csv", index=False)
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "hit_time_s": 1.0e-6,
                "part_id": 10,
                "outcome": "stuck",
                "hit_x_m": 0.10,
                "hit_y_m": 0.20,
                "impact_speed_mps": 1.0,
            },
            {
                "particle_id": 2,
                "hit_time_s": 2.0e-6,
                "part_id": 20,
                "outcome": "absorbed",
                "hit_x_m": 0.30,
                "hit_y_m": 0.40,
                "impact_speed_mps": 2.0,
            },
        ]
    ).to_csv(solver / "wall_events.csv", index=False)

    comsol = tmp_path / "comsol_particles.csv"
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "state": "freeze",
                "r": 0.10,
                "z": 0.20,
                "vx": 1.0,
                "vy": 0.0,
                "charge_C": -1.1e-18,
                "hit_time_s": 1.1e-6,
                "hit_boundary_id": 100,
                "hit_x_m": 0.10,
                "hit_y_m": 0.20,
            },
            {
                "particle_id": 2,
                "state": "disappear",
                "r": 0.31,
                "z": 0.40,
                "vx": 0.0,
                "vy": 2.0,
                "charge_C": -2.0e-18,
                "hit_time_s": 2.0e-6,
                "hit_boundary_id": 200,
                "hit_x_m": 0.30,
                "hit_y_m": 0.40,
            },
        ]
    ).to_csv(comsol, index=False)
    boundary_map = tmp_path / "boundary_map.csv"
    pd.DataFrame(
        [
            {"comsol_boundary_id": 100, "solver_part_id": 10},
            {"comsol_boundary_id": 200, "solver_part_id": 20},
        ]
    ).to_csv(boundary_map, index=False)
    raw = tmp_path / "raw_export"
    raw.mkdir()
    (raw / "expression_inventory.json").write_text(
        json.dumps(
            {
                "selected": {
                    "T": {"available": True, "expression": "T"},
                    "rho_g": {"available": True, "expression": "rho"},
                    "mu": {"available": True, "expression": "mu"},
                    "E_x": {"available": True, "expression": "Ex"},
                    "E_y": {"available": True, "expression": "Ey"},
                }
            }
        ),
        encoding="utf-8",
    )
    (raw / "physics_feature_inventory.json").write_text(
        json.dumps(
            {
                "features": [
                    {
                        "component_tag": "comp1",
                        "physics_tag": "fpt",
                        "feature_tag": "tf1",
                        "label": "Thermophoretic Force",
                        "type": "ThermophoreticForce",
                        "force_kind": "thermophoresis",
                        "selection_entities": [1],
                    },
                    {
                        "component_tag": "comp1",
                        "physics_tag": "fpt",
                        "feature_tag": "dep1",
                        "label": "Dielectrophoretic Force",
                        "type": "DielectrophoreticForce",
                        "force_kind": "dielectrophoresis",
                        "selection_entities": [1],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (raw / "particle_release_inventory.json").write_text(
        json.dumps(
            {
                "features": [
                    {
                        "component_tag": "comp1",
                        "physics_tag": "fpt",
                        "feature_tag": "rel1",
                        "label": "Release from Grid",
                        "type": "ReleaseGrid",
                        "release_kind": "release_grid",
                        "selection_entities": [10, 20],
                        "known_settings": {"tlist": "range(0,1e-6,1e-6)", "Nx": "2"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    comsol_release = tmp_path / "comsol_release.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "release_time": 0.0, "r": 0.10, "z": 0.20, "vx": 1.0, "vy": 0.0},
            {"particle_id": 2, "release_time": 1.2e-6, "r": 0.30, "z": 0.40, "vx": 0.0, "vy": 2.0},
        ]
    ).to_csv(comsol_release, index=False)

    summary = compare_particle_results(
        solver_output_dir=solver,
        comsol_particle_csv=comsol,
        out_dir=tmp_path / "compare",
        boundary_map_csv=boundary_map,
        raw_export_dir=raw,
        solver_particles_csv=solver_particles,
        comsol_release_csv=comsol_release,
    )

    assert summary["matched_particle_count"] == 2
    assert summary["state_match_ratio"] == pytest.approx(1.0)
    assert summary["first_hit_boundary_match_ratio"] == pytest.approx(1.0)
    assert summary["hit_time_error_s"]["max"] == pytest.approx(1.0e-7)
    assert summary["final_position_error_m"]["max"] == pytest.approx(0.01)
    assert summary["force_model_alignment"]["comsol_force_kinds"] == ["dielectrophoresis", "thermophoresis"]
    assert summary["force_model_alignment"]["enabled_solver_force_missing_export_field"] == []
    assert summary["release_alignment"]["comsol_release_feature_count"] == 1
    assert summary["release_alignment"]["comsol_release_kinds"] == ["release_grid"]
    assert summary["release_alignment"]["solver_particles_available"] is True
    assert summary["release_alignment"]["comsol_release_particles_available"] is True
    assert (tmp_path / "compare" / "comparison_summary.json").exists()
    assert (tmp_path / "compare" / "comparison_by_state.csv").exists()
    assert (tmp_path / "compare" / "comparison_by_boundary.csv").exists()
    assert (tmp_path / "compare" / "matched_particle_errors.csv").exists()
    assert (tmp_path / "compare" / "force_model_alignment.json").exists()
    assert (tmp_path / "compare" / "release_alignment.json").exists()
