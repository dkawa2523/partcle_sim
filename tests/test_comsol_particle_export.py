from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "comsol_particle_export"
sys.path.insert(0, str(EXTERNAL))

from comsol_particle_export.validate_export import validate_raw_export  # noqa: E402
from comsol_particle_export.compare_particle_results import _write_field_alignment  # noqa: E402
from comsol_particle_export.compare_particle_results import compare_particle_results  # noqa: E402
from comsol_particle_export.data_export import canonicalize_particle_xy_data_export  # noqa: E402
from comsol_particle_export.data_export import derive_particle_tables_from_trajectory  # noqa: E402
from comsol_particle_export.boundary_roles import derive_boundary_roles  # noqa: E402
from comsol_particle_export.field_bundle import build_field_bundle_from_samples  # noqa: E402
from comsol_particle_export.field_bundle import build_triangle_mesh_field_bundle_from_samples  # noqa: E402
from comsol_particle_export.release_alignment import compare_release_tables  # noqa: E402


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
    pd.DataFrame(
        [
            {"vertex_id": 0, "r": 0.0, "z": 0.0, "valid_mask": 1, "ux": 0.0, "uy": 0.0, "mu": 1.8e-5, "rho_g": 1.0, "T": 300.0, "E_x": 10.0, "E_y": -5.0},
            {"vertex_id": 1, "r": 1.0, "z": 0.0, "valid_mask": 1, "ux": 1.0, "uy": 1.0, "mu": 1.8e-5, "rho_g": 1.1, "T": 300.0, "E_x": 11.0, "E_y": -5.0},
            {"vertex_id": 2, "r": 0.0, "z": 1.0, "valid_mask": 1, "ux": 1.0, "uy": -1.0, "mu": 1.8e-5, "rho_g": 1.0, "T": 310.0, "E_x": 10.0, "E_y": -4.0},
        ]
    ).to_csv(raw / "mesh_field_samples.csv", index=False)
    return config


def test_validate_raw_export_accepts_complete_tensor_grid(tmp_path: Path) -> None:
    config = _write_minimal_export(tmp_path / "raw")

    summary = validate_raw_export(tmp_path / "raw", config)

    assert summary["case_name"] == "synthetic"
    assert summary["files"]["field_samples.csv"] is True
    assert summary["files"]["mesh_field_samples.csv"] is True
    assert summary["field_samples"]["row_count"] == 6
    assert summary["field_samples"]["valid_node_count"] == 6
    assert summary["field_samples"]["axes"]["r"]["count"] == 2
    assert summary["field_samples"]["axes"]["z"]["count"] == 3
    assert summary["field_samples"]["required_fields"]["E_y"]["variation"] == pytest.approx(2.0)
    assert summary["field_samples"]["selected_fields"]["T"]["variation"] == pytest.approx(20.0)
    assert summary["mesh_field_samples"]["vertex_count"] == 3
    assert summary["mesh_field_samples"]["required_fields"]["E_y"]["variation"] == pytest.approx(1.0)
    assert summary["particle_release_inventory"]["feature_count"] == 1
    assert summary["particle_release_inventory"]["time_dependent_feature_count"] == 1


def test_validate_raw_export_accepts_inventory_only_without_required_fields(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    config = raw / "config.json"
    config.write_text(
        json.dumps({"case_name": "inventory_only", "required": [], "axis_names": ["x", "y"]}),
        encoding="utf-8",
    )
    (raw / "model_inventory.json").write_text(json.dumps({"source_kind": "inventory"}), encoding="utf-8")
    (raw / "export_manifest.json").write_text(
        json.dumps({"source_kind": "external_comsol_particle_export", "case_name": "inventory_only"}),
        encoding="utf-8",
    )
    (raw / "expression_inventory.json").write_text(json.dumps({"selected": {}}), encoding="utf-8")

    summary = validate_raw_export(raw, config)

    assert summary["case_name"] == "inventory_only"
    assert summary["required"] == []
    assert summary["files"]["field_samples.csv"] is False


def test_validate_raw_export_accepts_multi_time_tensor_grid(tmp_path: Path) -> None:
    config = _write_minimal_export(tmp_path / "raw")
    rows = []
    for t in [0.0, 1.0e-6]:
        for r in [0.0, 1.0]:
            for z in [0.0, 1.0, 2.0]:
                rows.append(
                    {
                        "time_s": t,
                        "solnum": 1 if t == 0.0 else 2,
                        "r": r,
                        "z": z,
                        "valid_mask": 1,
                        "ux": r + z + t,
                        "uy": r - z,
                        "mu": 1.8e-5,
                        "rho_g": 1.0 + 0.1 * r,
                        "T": 300.0 + 10.0 * z,
                        "E_x": 10.0 + r,
                        "E_y": -5.0 + z,
                    }
                )
    pd.DataFrame(rows).to_csv(tmp_path / "raw" / "field_samples.csv", index=False)

    summary = validate_raw_export(tmp_path / "raw", config)

    assert summary["field_samples"]["row_count"] == 12
    assert summary["field_samples"]["sample_context"]["count"] == 2
    assert summary["field_samples"]["sample_context"]["time_count"] == 2
    assert summary["field_samples"]["sample_context"]["solnums"] == [1, 2]


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


def test_validate_raw_export_rejects_required_release_table_with_missing_columns(tmp_path: Path) -> None:
    config = _write_minimal_export(tmp_path / "raw")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["require_release_table"] = True
    config.write_text(json.dumps(payload), encoding="utf-8")
    pd.DataFrame([{"particle_id": 1, "release_time": 0.0, "x": 0.1, "y": 0.2}]).to_csv(
        tmp_path / "raw" / "comsol_release_particles.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="velocity"):
        validate_raw_export(tmp_path / "raw", config)


def test_build_field_bundle_from_multi_time_samples(tmp_path: Path) -> None:
    rows = []
    for t in [0.0, 1.0]:
        for x in [0.0, 1.0]:
            for y in [0.0, 2.0]:
                rows.append({"time_s": t, "x": x, "y": y, "valid_mask": 1, "ux": x + t, "uy": y - t})
    bundle = build_field_bundle_from_samples(
        pd.DataFrame(rows),
        axis_names=["x", "y"],
        quantities=["ux", "uy"],
    )

    assert bundle["axis_0"].tolist() == [0.0, 1.0]
    assert bundle["axis_1"].tolist() == [0.0, 2.0]
    assert bundle["times"].tolist() == [0.0, 1.0]
    assert bundle["valid_mask"].shape == (2, 2)
    assert bundle["ux"].shape == (2, 2, 2)


def test_build_triangle_mesh_field_bundle_from_mesh_vertex_samples(tmp_path: Path) -> None:
    mesh_vertices = np.asarray([[0.0, 0.0], [0.001, 0.0], [0.0, 0.001]], dtype=float)
    mesh_triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    rows = []
    for t in [0.0, 1.0]:
        rows.extend(
            [
                {"time_s": t, "vertex_id": 0, "x": 0.0, "y": 0.0, "valid_mask": 1, "ux": t, "uy": 0.0, "mu": 0.001},
                {"time_s": t, "vertex_id": 1, "x": 1.0, "y": 0.0, "valid_mask": 1, "ux": 1.0 + t, "uy": 0.0, "mu": 0.001},
                {"time_s": t, "vertex_id": 2, "x": 0.0, "y": 1.0, "valid_mask": 1, "ux": t, "uy": 1.0, "mu": 0.001},
            ]
        )

    bundle = build_triangle_mesh_field_bundle_from_samples(
        pd.DataFrame(rows),
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        axis_names=["x", "y"],
        quantities=["ux", "uy", "mu"],
        coordinate_scale_m_per_model_unit=0.001,
        coordinate_model_unit="mm",
        metadata={"support_tolerance_m": 1.0e-5},
    )

    assert bundle["mesh_vertices"].shape == (3, 2)
    assert bundle["mesh_triangles"].shape == (1, 3)
    assert bundle["times"].tolist() == [0.0, 1.0]
    assert bundle["ux"].shape == (2, 3)
    metadata = json.loads(str(bundle["metadata_json"].item()))
    assert metadata["field_backend_kind"] == "triangle_mesh_2d"
    assert metadata["support_tolerance_m"] == pytest.approx(1.0e-5)


def test_build_triangle_mesh_field_bundle_rejects_missing_quantity(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {"vertex_id": 0, "x": 0.0, "y": 0.0, "ux": 0.0},
            {"vertex_id": 1, "x": 1.0, "y": 0.0, "ux": 1.0},
            {"vertex_id": 2, "x": 0.0, "y": 1.0, "ux": 0.0},
        ]
    )

    with pytest.raises(ValueError, match="uy"):
        build_triangle_mesh_field_bundle_from_samples(
            rows,
            mesh_vertices=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
            mesh_triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
            axis_names=["x", "y"],
            quantities=["ux", "uy"],
        )


def test_canonicalize_particle_xy_data_export_wide_table(tmp_path: Path) -> None:
    export = tmp_path / "particle_xy.csv"
    export.write_text(
        "\n".join(
            [
                "% Model,synthetic.mph",
                "% Version,COMSOL 6.4",
                "% Dimension,1",
                "% Nodes,2",
                "% Expressions,4",
                "% Description,\"x-coordinate, y-coordinate\"",
                "% Index,x (mm) @ t=0,y (mm) @ t=0,x (mm) @ t=0.5,y (mm) @ t=0.5",
                "1,-3.0,1.0,-2.0,1.5",
                "2,NaN,NaN,4.0,5.0",
            ]
        ),
        encoding="utf-8",
    )

    frame, report = canonicalize_particle_xy_data_export(export)

    assert list(frame.columns) == ["particle_id", "time_s", "x", "y"]
    assert len(frame) == 3
    assert frame.iloc[0].to_dict() == pytest.approx({"particle_id": 1, "time_s": 0.0, "x": -0.003, "y": 0.001})
    assert frame.iloc[-1].to_dict() == pytest.approx({"particle_id": 2, "time_s": 0.5, "x": 0.004, "y": 0.005})
    assert report["raw_particle_count"] == 2
    assert report["trajectory_row_count"] == 3
    assert report["axis_scale_to_m"] == {"x": 1.0e-3, "y": 1.0e-3}


def test_derive_particle_tables_from_trajectory(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "time_s": 0.0, "x": 0.0, "y": 0.0, "v_x": 1.0, "v_y": 2.0},
            {"particle_id": 1, "time_s": 1.0, "x": 0.1, "y": 0.2, "v_x": 3.0, "v_y": 4.0},
            {"particle_id": 2, "time_s": 0.5, "x": 0.3, "y": 0.4, "v_x": 5.0, "v_y": 6.0},
        ]
    ).to_csv(trajectory, index=False)
    release = tmp_path / "release.csv"
    final = tmp_path / "final.csv"

    report = derive_particle_tables_from_trajectory(
        trajectory,
        release_csv=release,
        final_csv=final,
        initial_velocity={"v_x": 0.0, "v_y": 0.0},
    )

    release_frame = pd.read_csv(release)
    final_frame = pd.read_csv(final)
    assert report["particle_count"] == 2
    assert release_frame.loc[0, "release_time"] == pytest.approx(0.0)
    assert release_frame.loc[0, "v_x"] == pytest.approx(0.0)
    assert final_frame.loc[0, "x"] == pytest.approx(0.1)
    assert final_frame.loc[0, "v_y"] == pytest.approx(4.0)


def test_compare_release_tables_reports_level1_errors(tmp_path: Path) -> None:
    solver_particles = tmp_path / "particles.csv"
    comsol_release = tmp_path / "comsol_release.csv"
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "release_time": 0.0,
                "x": 0.10,
                "y": 0.20,
                "vx": 0.0,
                "vy": 1.0,
                "diameter": 1.0e-6,
                "density": 2000.0,
                "charge": 0.0,
            },
            {
                "particle_id": 2,
                "release_time": 1.0,
                "x": 0.30,
                "y": 0.40,
                "vx": 2.0,
                "vy": 3.0,
                "diameter": 1.0e-6,
                "density": 2000.0,
                "charge": 0.0,
            },
        ]
    ).to_csv(solver_particles, index=False)
    pd.DataFrame(
        [
            {"particle_id": 1, "release_time": 0.0, "x": 0.10, "y": 0.20, "v_x": 0.0, "v_y": 1.0},
            {"particle_id": 2, "release_time": 1.5, "x": 0.31, "y": 0.40, "v_x": 2.0, "v_y": 4.0},
        ]
    ).to_csv(comsol_release, index=False)

    summary = compare_release_tables(solver_particles, comsol_release, out_dir=tmp_path / "release_compare")

    assert summary["matched_particle_count"] == 2
    assert summary["release_time_error_s"]["max"] == pytest.approx(0.5)
    assert summary["release_position_error_m"]["max"] == pytest.approx(0.01)
    assert summary["release_velocity_error_mps"]["max"] == pytest.approx(1.0)
    assert (tmp_path / "release_compare" / "matched_release_errors.csv").exists()
    assert (tmp_path / "release_compare" / "release_alignment_summary.json").exists()


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
                    "pressure_gradient_enabled": 0,
                    "virtual_mass_enabled": 0,
                    "gravity_buoyancy_enabled": 0,
                },
                "force_catalog": {
                    "enabled_forces": ["drag", "thermophoresis", "dielectrophoresis"],
                    "force_models": {"drag": "stokes", "pressure_gradient": "fluid_material_acceleration"},
                    "force_status": {
                        "drag": "implemented",
                        "thermophoresis": "implemented",
                        "dielectrophoresis": "implemented",
                        "pressure_gradient": "implemented",
                        "virtual_mass": "implemented",
                    },
                },
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
    pd.DataFrame(
        [
            {"save_index": 0, "time_s": 0.0, "step_name": "run", "segment_name": "run"},
            {"save_index": 1, "time_s": 1.0e-6, "step_name": "run", "segment_name": "run"},
        ]
    ).to_csv(solver / "save_frames.csv", index=False)
    import numpy as np

    np.save(
        solver / "positions_2d.npy",
        np.asarray(
            [
                [[0.10, 0.20], [0.30, 0.40]],
                [[0.11, 0.20], [0.30, 0.42]],
            ],
            dtype=float,
        ),
    )

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
                        "physics_tag": "spf",
                        "physics_label": "Laminar Flow",
                        "physics_type": "LaminarFlow",
                        "feature_tag": "grav1",
                        "label": "Gravity 1",
                        "type": "Gravity",
                        "force_kind": "gravity",
                        "selection_entities": [],
                    },
                    {
                        "component_tag": "comp1",
                        "physics_tag": "fpt",
                        "physics_label": "Particle Tracing for Fluid Flow",
                        "physics_type": "FluidParticleTracing",
                        "feature_tag": "df1",
                        "label": "Drag Force",
                        "type": "DragForce",
                        "force_kind": "drag",
                        "selection_entities": [1],
                        "property_values": {
                            "Rarefaction_Effects": "CunninghamMillikanDavies",
                            "IncludeVirtualMassAndPressureGradientForces": "1",
                        },
                    },
                    {
                        "component_tag": "comp1",
                        "physics_tag": "fpt",
                        "physics_label": "Particle Tracing for Fluid Flow",
                        "physics_type": "FluidParticleTracing",
                        "feature_tag": "tf1",
                        "label": "Thermophoretic Force",
                        "type": "ThermophoreticForce",
                        "force_kind": "thermophoresis",
                        "selection_entities": [1],
                    },
                    {
                        "component_tag": "comp1",
                        "physics_tag": "fpt",
                        "physics_label": "Particle Tracing for Fluid Flow",
                        "physics_type": "FluidParticleTracing",
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
    comsol_trajectory = tmp_path / "comsol_trajectory.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "time_s": 0.0, "x": 0.10, "y": 0.20, "v_x": 1.0, "v_y": 0.0},
            {"particle_id": 2, "time_s": 0.0, "x": 0.30, "y": 0.40, "v_x": 1.0, "v_y": 0.0},
            {"particle_id": 1, "time_s": 1.0e-6, "x": 0.12, "y": 0.20, "v_x": 1.0, "v_y": 0.0},
            {"particle_id": 2, "time_s": 1.0e-6, "x": 0.30, "y": 0.41, "v_x": 1.0, "v_y": 0.0},
        ]
    ).to_csv(comsol_trajectory, index=False)
    field_npz = tmp_path / "field.npz"
    import numpy as np

    np.savez(
        field_npz,
        axis_0=np.asarray([0.0, 0.5], dtype=float),
        axis_1=np.asarray([0.0, 0.5], dtype=float),
        times=np.asarray([0.0], dtype=float),
        valid_mask=np.ones((2, 2), dtype=bool),
        ux=np.ones((1, 2, 2), dtype=float),
        uy=np.zeros((1, 2, 2), dtype=float),
    )

    summary = compare_particle_results(
        solver_output_dir=solver,
        comsol_particle_csv=comsol,
        out_dir=tmp_path / "compare",
        boundary_map_csv=boundary_map,
        raw_export_dir=raw,
        solver_particles_csv=solver_particles,
        comsol_release_csv=comsol_release,
        comsol_trajectory_csv=comsol_trajectory,
        field_npz=field_npz,
    )

    assert summary["matched_particle_count"] == 2
    assert summary["state_match_ratio"] == pytest.approx(1.0)
    assert summary["first_hit_boundary_match_ratio"] == pytest.approx(1.0)
    assert summary["hit_time_error_s"]["max"] == pytest.approx(1.0e-7)
    assert summary["final_position_error_m"]["max"] == pytest.approx(0.01)
    assert summary["force_model_alignment"]["comsol_force_kinds"] == ["dielectrophoresis", "drag", "thermophoresis"]
    assert summary["force_model_alignment"]["comsol_non_particle_force_kinds"] == ["gravity"]
    assert [gap["category"] for gap in summary["force_model_alignment"]["force_physics_gaps"]] == [
        "drag_model",
        "force_not_enabled",
    ]
    assert {
        row["contribution"]
        for row in summary["force_model_alignment"]["force_contribution_alignment"]
        if row["solver_status"] == "missing_solver_force"
    } == {"pressure_gradient", "virtual_mass"}
    assert summary["force_model_alignment"]["enabled_solver_force_missing_export_field"] == []
    assert summary["comparison_readiness"]["ready_for_exact_solver_comparison"] is False
    assert {
        blocker["category"] for blocker in summary["comparison_readiness"]["blockers"]
    } >= {"force_physics", "release_source"}
    assert summary["release_alignment"]["comsol_release_feature_count"] == 1
    assert summary["release_alignment"]["comsol_release_kinds"] == ["release_grid"]
    assert summary["release_alignment"]["solver_particles_available"] is True
    assert summary["release_alignment"]["comsol_release_particles_available"] is True
    assert summary["release_alignment"]["matched_release_errors"]["release_position_error_m"]["count"] == 2
    assert summary["release_alignment"]["matched_release_errors"]["release_position_error_m"]["max"] == pytest.approx(0.0)
    assert summary["release_alignment"]["matched_release_errors"]["release_velocity_error_mps"]["count"] == 2
    assert summary["release_alignment"]["matched_release_errors"]["release_velocity_error_mps"]["max"] == pytest.approx(0.0)
    assert summary["trajectory_alignment"]["available"] is True
    assert summary["trajectory_alignment"]["matched_sample_count"] == 4
    assert summary["field_alignment"]["available"] is True
    assert summary["field_alignment"]["sample_count"] == 4
    assert summary["field_alignment"]["velocity_residual_mps"]["max"] == pytest.approx(0.0)
    assert summary["trend_alignment"]["available"] is True
    assert summary["trend_alignment"]["comsol_finite_at_final_count"] == 2
    assert summary["divergence_alignment"]["available"] is True
    assert summary["divergence_alignment"]["particles_with_wall_event_count"] == 2
    assert summary["divergence_alignment"]["by_threshold"]["0.0001"]["diverged_count"] == 2
    assert (tmp_path / "compare" / "comparison_summary.json").exists()
    assert (tmp_path / "compare" / "comparison_by_state.csv").exists()
    assert (tmp_path / "compare" / "comparison_by_boundary.csv").exists()
    assert (tmp_path / "compare" / "matched_particle_errors.csv").exists()
    assert (tmp_path / "compare" / "force_model_alignment.json").exists()
    assert (tmp_path / "compare" / "force_contribution_alignment.csv").exists()
    assert (tmp_path / "compare" / "release_alignment.json").exists()
    assert (tmp_path / "compare" / "trajectory_alignment.json").exists()
    assert (tmp_path / "compare" / "matched_trajectory_errors.csv").exists()
    assert (tmp_path / "compare" / "distribution_alignment.csv").exists()
    assert (tmp_path / "compare" / "field_alignment.json").exists()
    assert (tmp_path / "compare" / "trend_alignment.json").exists()
    assert (tmp_path / "compare" / "divergence_alignment.json").exists()
    assert (tmp_path / "compare" / "divergence_alignment.csv").exists()
    assert (tmp_path / "compare" / "comparison_readiness.json").exists()


def test_compare_particle_results_uses_separate_comsol_particle_status(tmp_path: Path) -> None:
    solver = tmp_path / "solver"
    solver.mkdir()
    pd.DataFrame(
        [
            {"particle_id": 1, "active": 0, "stuck": 1, "x": 0.0, "y": 0.0, "v_x": 0.0, "v_y": 0.0},
            {"particle_id": 2, "active": 0, "absorbed": 1, "x": 1.0, "y": 0.0, "v_x": 0.0, "v_y": 0.0},
        ]
    ).to_csv(solver / "final_particles.csv", index=False)
    pd.DataFrame(
        [
            {"particle_id": 1, "hit_time_s": 0.10, "part_id": 10, "outcome": "stuck"},
            {"particle_id": 2, "hit_time_s": 0.20, "part_id": 20, "outcome": "absorbed"},
        ]
    ).to_csv(solver / "wall_events.csv", index=False)
    comsol = tmp_path / "comsol_particles.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "state": "freeze", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
            {"particle_id": 2, "state": "disappear", "x": 1.0, "y": 0.0, "vx": 0.0, "vy": 0.0},
        ]
    ).to_csv(comsol, index=False)
    particle_status = tmp_path / "comsol_particle_status.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "stop_time_s": 0.11, "final_status": "frozen"},
            {"particle_id": 2, "stop_time_s": 0.25, "final_status": "disappeared"},
        ]
    ).to_csv(particle_status, index=False)

    summary = compare_particle_results(
        solver_output_dir=solver,
        comsol_particle_csv=comsol,
        comsol_particle_status_csv=particle_status,
        out_dir=tmp_path / "compare_events",
    )

    assert summary["first_hit_time_comparison_count"] == 0
    assert summary["particle_status_stop_time_comparison_count"] == 2
    assert summary["solver_first_wall_vs_comsol_stop_time_error_s"]["max"] == pytest.approx(0.05)
    assert summary["first_hit_boundary_match_ratio"] is None
    assert {
        blocker["category"] for blocker in summary["comparison_readiness"]["blockers"]
    } >= {"boundary_events"}


def test_field_alignment_replays_triangle_mesh_backend(tmp_path: Path) -> None:
    field_npz = tmp_path / "mesh_field.npz"
    np.savez(
        field_npz,
        mesh_vertices=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        mesh_triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        times=np.asarray([0.0], dtype=float),
        ux=np.asarray([[0.0, 1.0, 0.0]], dtype=float),
        uy=np.asarray([[0.0, 0.0, 1.0]], dtype=float),
    )
    trajectory = tmp_path / "comsol_trajectory.csv"
    pd.DataFrame(
        [
            {"particle_id": 1, "time_s": 0.0, "x": 0.25, "y": 0.25, "v_x": 0.25, "v_y": 0.25},
            {"particle_id": 2, "time_s": 0.0, "x": 0.50, "y": 0.25, "v_x": 0.50, "v_y": 0.25},
        ]
    ).to_csv(trajectory, index=False)

    payload = _write_field_alignment(
        field_npz=field_npz,
        comsol_trajectory_csv=trajectory,
        solver_particles_csv=None,
        out_dir=tmp_path / "field_compare",
    )

    assert payload["available"] is True
    assert payload["field_backend_kind"] == "triangle_mesh_2d"
    assert payload["clean_stencil_fraction"] == pytest.approx(1.0)
    assert payload["velocity_residual_mps"]["max"] == pytest.approx(0.0)
    assert (tmp_path / "field_compare" / "field_alignment_by_source.csv").exists()


def test_derive_boundary_roles_maps_pair_continuity_to_pass_through(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "physics_feature_inventory.json").write_text(
        json.dumps(
            {
                "features": [
                    {
                        "physics_tag": "fpt",
                        "physics_label": "Particle Tracing",
                        "physics_type": "FluidParticleTracing",
                        "feature_tag": "wall1",
                        "label": "Wall 1",
                        "type": "Wall",
                        "selection_entities": [1],
                        "property_values": {"WallCondition": "Bounce", "e": "1"},
                    },
                    {
                        "physics_tag": "fpt",
                        "physics_label": "Particle Tracing",
                        "physics_type": "FluidParticleTracing",
                        "feature_tag": "out1",
                        "label": "Outlet 1",
                        "type": "Outlet",
                        "selection_entities": [2],
                        "property_values": {"WallCondition": "Freeze"},
                    },
                    {
                        "physics_tag": "fpt",
                        "physics_label": "Particle Tracing",
                        "physics_type": "FluidParticleTracing",
                        "feature_tag": "pc1",
                        "label": "Particle Continuity 1",
                        "type": "PairContinuity",
                        "selection_entities": [3],
                        "property_values": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    boundary_map = tmp_path / "map.csv"
    pd.DataFrame(
        [
            {"comsol_api_selection_entity_id": 1, "solver_part_id": 11},
            {"comsol_api_selection_entity_id": 2, "solver_part_id": 12},
            {"comsol_api_selection_entity_id": 3, "solver_part_id": 13},
        ]
    ).to_csv(boundary_map, index=False)
    current_walls = tmp_path / "part_walls.csv"
    pd.DataFrame(
        [
            {"part_id": 11, "wall_law": "specular"},
            {"part_id": 12, "wall_law": "stick"},
            {"part_id": 13, "wall_law": "specular"},
        ]
    ).to_csv(current_walls, index=False)
    out_walls = tmp_path / "derived_part_walls.csv"

    summary = derive_boundary_roles(
        raw_export_dir=raw,
        boundary_map_csv=boundary_map,
        part_walls_csv=current_walls,
        out_dir=tmp_path / "roles",
        write_part_walls_csv=out_walls,
    )

    assert summary["mismatch_count"] == 1
    derived = pd.read_csv(out_walls)
    laws = dict(zip(derived["part_id"], derived["wall_law"]))
    assert laws[11] == "specular"
    assert laws[12] == "stick"
    assert laws[13] == "pass_through"
