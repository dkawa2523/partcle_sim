from __future__ import annotations

import json
from pathlib import Path

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

from tools._result_graph_summary import (
    export_result_graphs as result_graph_export_implementation,
)
from tools.export_mechanics_visuals import export_mechanics_visuals
from tools.export_result_graphs import export_result_graphs
from tools.export_trajectory_animation import export_trajectory_animations
from tools.export_visualizations import export_visualizations


def test_result_graph_facade_reexports_the_implementation() -> None:
    assert export_result_graphs is result_graph_export_implementation


def _writer_final_particles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": [2, 2, 2],
            "particle_id": [0, 1, 2],
            "final_state": ["active_free_flight", "contact_sliding", "stuck"],
            "x_m": [0.1, 0.5, 0.9],
            "y_m": [0.2, 0.5, 0.8],
            "vx_mps": [3.0, 0.0, 1.0],
            "vy_mps": [4.0, 2.0, 0.0],
            "contact_part_id": [-1, 7, 8],
        }
    )


def _write_full_run(output_dir: Path) -> None:
    output_dir.mkdir()
    final_particles = _writer_final_particles()
    final_particles.to_csv(output_dir / "final_particles.csv", index=False)
    positions = np.asarray(
        [
            [[0.0, 0.1], [0.4, 0.4], [0.8, 0.7]],
            final_particles[["x_m", "y_m"]].to_numpy(dtype=np.float64),
        ],
        dtype=np.float64,
    )
    np.save(output_dir / "trajectory.npy", positions)
    pd.DataFrame({"time_s": [0.0, 1.0]}).to_csv(
        output_dir / "trajectory_frames.csv", index=False
    )
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "active_count": [3, 2],
            "stuck_count": [0, 1],
            "absorbed_count": [0, 0],
            "escaped_count": [0, 0],
            "invalid_mask_stopped_count_step": [0, 0],
        }
    ).to_csv(output_dir / "step_summary.csv", index=False)
    pd.DataFrame(
        {
            "time_s": [0.5],
            "particle_id": [2],
            "part_id": [8],
            "hit_x_m": [0.85],
            "hit_y_m": [0.75],
            "outcome": ["reflected_specular"],
            "wall_mode": ["specular"],
        }
    ).to_csv(output_dir / "wall_events.csv", index=False)
    pd.DataFrame(
        {
            "schema_version": [2],
            "part_id": [8],
            "outcome": ["reflected_specular"],
            "wall_mode": ["specular"],
            "count": [1],
        }
    ).to_csv(output_dir / "wall_summary.csv", index=False)
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "particle_count": 3,
                "released_count": 3,
                "final_state_counts": {
                    "active_free_flight": 1,
                    "contact_sliding": 1,
                    "stuck": 1,
                },
                "wall_law_counts": {"stick": 1},
                "timing_s": {"solver_core_s": 0.25},
                "memory_estimate_bytes": {"estimated_numpy_bytes": 128},
            }
        ),
        encoding="utf-8",
    )


def _write_full_run_3d(output_dir: Path) -> None:
    output_dir.mkdir()
    final_particles = pd.DataFrame(
        {
            "particle_id": [0, 1],
            "final_state": ["active_free_flight", "escaped"],
            "x_m": [0.1, 0.9],
            "y_m": [0.2, 0.8],
            "z_m": [0.3, 0.7],
            "vx_mps": [1.0, 0.0],
            "vy_mps": [0.0, 2.0],
            "vz_mps": [0.0, 0.0],
        }
    )
    final_particles.to_csv(output_dir / "final_particles.csv", index=False)
    np.save(
        output_dir / "trajectory.npy",
        np.asarray(
            [
                [[0.0, 0.1, 0.2], [0.8, 0.7, 0.6]],
                final_particles[["x_m", "y_m", "z_m"]].to_numpy(dtype=np.float64),
            ],
            dtype=np.float64,
        ),
    )
    pd.DataFrame({"time_s": [0.0, 1.0]}).to_csv(
        output_dir / "trajectory_frames.csv", index=False
    )
    pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "active_count": [2, 1],
            "stuck_count": [0, 0],
            "absorbed_count": [0, 0],
            "escaped_count": [0, 1],
            "invalid_mask_stopped_count_step": [0, 0],
        }
    ).to_csv(output_dir / "step_summary.csv", index=False)
    pd.DataFrame(
        columns=["schema_version", "part_id", "outcome", "wall_mode", "count"]
    ).to_csv(output_dir / "wall_summary.csv", index=False)


def _write_mechanics_case(case_dir: Path) -> None:
    generated = case_dir / "generated"
    generated.mkdir(parents=True)
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    shape = (2, 2)
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    edges = np.asarray(
        [
            [vertices[0], vertices[1]],
            [vertices[1], vertices[2]],
            [vertices[2], vertices[3]],
            [vertices[3], vertices[0]],
        ],
        dtype=np.float64,
    )
    np.savez(
        generated / "comsol_geometry_2d.npz",
        axis_0=axis,
        axis_1=axis,
        sdf=-np.ones(shape, dtype=np.float64),
        normal_0=np.ones(shape, dtype=np.float64),
        normal_1=np.zeros(shape, dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
        nearest_boundary_part_id_map=np.full(shape, 7, dtype=np.int32),
        boundary_edges=edges,
        boundary_edge_part_ids=np.full(4, 7, dtype=np.int32),
        mesh_vertices=vertices,
        mesh_quads=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        mesh_quad_part_ids=np.asarray([7], dtype=np.int32),
    )
    np.savez(
        generated / "comsol_field_2d.npz",
        axis_0=axis,
        axis_1=axis,
        ux=np.ones(shape, dtype=np.float64),
        uy=np.zeros(shape, dtype=np.float64),
        E_x=np.full(shape, 2.0, dtype=np.float64),
        E_y=np.zeros(shape, dtype=np.float64),
        mu=np.full(shape, 1.8e-5, dtype=np.float64),
        T=np.full(shape, 300.0, dtype=np.float64),
        p=np.full(shape, 101325.0, dtype=np.float64),
        rho_g=np.full(shape, 1.2, dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
    )


def test_compact_graph_export_accepts_writer_particle_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "compact"
    output_dir.mkdir()
    _writer_final_particles().to_csv(output_dir / "final_particles.csv", index=False)

    graph_dir = export_result_graphs(output_dir)

    summary = json.loads((graph_dir / "graph_summary.json").read_text())
    assert summary["graph_mode"] == "compact_final_state"
    assert summary["spatial_dim"] == 2
    assert summary["axis_names"] == ["x", "y"]
    assert summary["missing_trajectory_artifacts"] == [
        "trajectory_frames.csv",
        "step_summary.csv",
        "trajectory.npy",
    ]
    assert (graph_dir / "03_final_state_scatter_geometry.png").is_file()


def test_standard_visualization_export_preserves_report_artifacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "standard"
    _write_full_run(output_dir)

    index_path = export_visualizations(output_dir, modules=("graphs",))

    assert index_path == (
        output_dir / "visualizations" / "reports" / "visualization_index.json"
    )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert list(payload) == [
        "output_dir",
        "visualizations_root",
        "clean",
        "health_summary",
        "modules",
        "run_summary_md",
    ]
    assert payload["modules"]["graphs"]["status"] == "pass"
    assert payload["health_summary"]["particle_count"] == 3
    summary_path = Path(payload["run_summary_md"])
    assert summary_path == output_dir / "visualizations" / "reports" / "run_summary.md"
    assert summary_path.is_file()


def test_graph_export_accepts_legacy_non_object_case_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "compact"
    output_dir.mkdir()
    _writer_final_particles().to_csv(output_dir / "final_particles.csv", index=False)
    case_dir = Path(__file__).parent / "fixtures" / "legacy_npz_2d"

    graph_dir = export_result_graphs(output_dir, case_dir=case_dir)

    summary = json.loads((graph_dir / "graph_summary.json").read_text())
    assert summary["graph_mode"] == "compact_final_state"
    assert summary["case_dir"] == str(case_dir.resolve())


def test_compact_graph_export_preserves_three_dimensional_schema(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "compact-3d"
    output_dir.mkdir()
    pd.DataFrame(
        {
            "particle_id": [0, 1],
            "final_state": ["active_free_flight", "escaped"],
            "x_m": [0.1, 0.9],
            "y_m": [0.2, 0.8],
            "z_m": [0.3, 0.7],
        }
    ).to_csv(output_dir / "final_particles.csv", index=False)

    graph_dir = export_result_graphs(output_dir)

    summary = json.loads((graph_dir / "graph_summary.json").read_text())
    assert summary["spatial_dim"] == 3
    assert summary["axis_names"] == ["x", "y", "z"]
    assert summary["recommended_for_reports"] == [
        "02_final_state_bar_and_pie.png",
        "02_final_state_counts.csv",
        "03_final_state_scatter_geometry.png",
    ]
    assert (graph_dir / "03_final_state_scatter_geometry.png").is_file()


def test_compact_graph_export_handles_empty_coordinate_free_result(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "compact-empty"
    output_dir.mkdir()
    pd.DataFrame(columns=["particle_id", "final_state"]).to_csv(
        output_dir / "final_particles.csv", index=False
    )

    graph_dir = export_result_graphs(output_dir)

    summary = json.loads((graph_dir / "graph_summary.json").read_text())
    assert summary["spatial_dim"] == 0
    assert summary["axis_names"] == []
    assert summary["recommended_for_reports"] == [
        "02_final_state_bar_and_pie.png",
        "02_final_state_counts.csv",
    ]
    assert not (graph_dir / "03_final_state_scatter_geometry.png").exists()


def test_full_graph_export_uses_writer_coordinates_and_velocities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    output_dir = tmp_path / "full"
    _write_mechanics_case(case_dir)
    _write_full_run(output_dir)
    captured_speeds: list[float] = []
    original_hist = matplotlib.axes.Axes.hist

    def capture_hist(self, values, *args, **kwargs):
        captured_speeds.extend(np.asarray(values, dtype=np.float64).ravel())
        return original_hist(self, values, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", capture_hist)

    graph_dir = export_result_graphs(
        output_dir,
        case_dir=case_dir,
        sample_trajectories=3,
    )

    assert sorted(captured_speeds) == [1.0, 2.0, 5.0]
    contact_summary = pd.read_csv(graph_dir / "07_contact_states_by_boundary_part.csv")
    assert contact_summary.to_dict("records") == [
        {"contact_part_id": 7, "contact_state": "contact_sliding", "count": 1}
    ]
    summary = json.loads((graph_dir / "graph_summary.json").read_text())
    assert summary["graph_mode"] == "trajectory_full"
    assert summary["spatial_dim"] == 2
    assert summary["contact_state_counts_by_part"] == [
        {"contact_part_id": 7, "contact_state": "contact_sliding", "count": 1}
    ]
    assert summary["extra_graph_files"] == [
        "11_device_parts_geometry.png",
        "12_device_parts_with_ids.png",
        "22_domain_part_medium_summary.csv",
        "22_domain_parts_medium_support.png",
        "13_signed_distance_field_sdf.png",
        "14_geometry_field_support_mask.png",
        "15_mechanics_field_totals.png",
        "16_flow_components_ux_uy.png",
        "18_electric_field_components_ex_ey.png",
        "19_scalar_physics_fields.png",
        "27_drag_gas_property_sources.csv",
        "27_drag_gas_properties_used_by_drag.png",
        "20_wall_event_locations_by_outcome.png",
        "21_trajectories_by_final_state.png",
        "23_comsol_style_field_and_trajectories.png",
        "24_comsol_style_particle_density_and_events.png",
    ]
    drag_sources = pd.read_csv(graph_dir / "27_drag_gas_property_sources.csv")
    assert drag_sources.columns.tolist() == [
        "role",
        "field_quantity",
        "source",
        "fallback_value",
        "used_by_drag",
        "field_min",
        "field_p50",
        "field_p90",
        "field_max",
        "field_mean",
    ]
    assert (graph_dir / "10_stuck_counts_by_boundary_part.csv").is_file()
    assert (graph_dir / "15_mechanics_field_totals.png").is_file()
    assert (graph_dir / "16_flow_components_ux_uy.png").is_file()
    assert (graph_dir / "18_electric_field_components_ex_ey.png").is_file()
    assert (graph_dir / "19_scalar_physics_fields.png").is_file()
    assert (graph_dir / "23_comsol_style_field_and_trajectories.png").is_file()


def test_full_graph_export_preserves_three_dimensional_projections(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "full-3d"
    _write_full_run_3d(output_dir)

    graph_dir = export_result_graphs(output_dir, sample_trajectories=2)

    summary = json.loads((graph_dir / "graph_summary.json").read_text())
    assert summary["graph_mode"] == "trajectory_full"
    assert summary["spatial_dim"] == 3
    assert summary["extra_graph_files"] == []
    assert summary["final_state_counts"]["active_free_flight"] == 1
    assert summary["final_state_counts"]["escaped"] == 1
    assert (graph_dir / "03_final_state_scatter_geometry.png").is_file()
    assert (graph_dir / "04_trajectory_density_heatmap.png").is_file()
    assert (graph_dir / "06_sampled_trajectories_overlay.png").is_file()


def test_full_graph_export_rejects_pickled_case_arrays(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    output_dir = tmp_path / "full"
    _write_mechanics_case(case_dir)
    _write_full_run(output_dir)
    geometry_path = case_dir / "generated" / "comsol_geometry_2d.npz"
    with np.load(geometry_path) as geometry:
        payload = {name: geometry[name] for name in geometry.files}
    payload["unsafe_object"] = np.asarray([{"part_id": 7}], dtype=object)
    np.savez(geometry_path, **payload)

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        export_result_graphs(output_dir, case_dir=case_dir)


def test_mechanics_summary_uses_writer_coordinate_columns(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    output_dir = tmp_path / "run"
    _write_mechanics_case(case_dir)
    _write_full_run(output_dir)

    mechanics_dir = export_mechanics_visuals(
        case_dir,
        output_dir,
        sample_trajectories=3,
        quiver_stride=1,
    )

    summary = pd.read_csv(mechanics_dir / "final_state_by_nearest_boundary_part.csv")
    assert summary[["nearest_boundary_part_id", "state", "count"]].to_dict(
        "records"
    ) == [
        {
            "nearest_boundary_part_id": 7,
            "state": "active_free_flight",
            "count": 1,
        },
        {
            "nearest_boundary_part_id": 7,
            "state": "contact_sliding",
            "count": 1,
        },
        {"nearest_boundary_part_id": 7, "state": "stuck", "count": 1},
    ]

    expected_files = [
        "domain_part_medium_summary.csv",
        "mechanics_distribution_on_geometry.csv",
        "final_state_by_nearest_boundary_part.csv",
        "geometry_layout_part_ids.png",
        "mechanics_maps_with_geometry.png",
        "mechanics_component_maps_with_geometry.png",
        "trajectories_geometry_flow_overlay.png",
        "final_states_over_geometry.png",
    ]
    report = json.loads((mechanics_dir / "visualization_report.json").read_text())
    assert report == {
        "case_dir": str(case_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mechanics_dir": str(mechanics_dir.resolve()),
        "n_particles": 3,
        "sample_trajectories": 3,
        "boundary_region_summary_status": (
            "computed_from_nearest_boundary_part_id_map"
        ),
        "final_state_by_nearest_boundary_part": summary.to_dict("records"),
        "files": expected_files,
    }
    assert all((mechanics_dir / name).is_file() for name in expected_files)


def test_mechanics_visuals_reject_field_without_velocity_pair(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    output_dir = tmp_path / "run"
    _write_mechanics_case(case_dir)
    _write_full_run(output_dir)
    field_path = case_dir / "generated" / "comsol_field_2d.npz"
    with np.load(field_path) as field:
        payload = {name: field[name] for name in field.files if name != "uy"}
    np.savez(field_path, **payload)

    with pytest.raises(
        ValueError,
        match="mechanics visuals require ux and uy in the field bundle",
    ):
        export_mechanics_visuals(case_dir, output_dir)


def test_trajectory_animation_preserves_two_dimensional_artifacts(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    output_dir = tmp_path / "run"
    _write_mechanics_case(case_dir)
    _write_full_run(output_dir)

    animation_dir = export_trajectory_animations(
        output_dir,
        case_dir=case_dir,
        fps=1,
        sample_count=2,
        max_frames=2,
        max_particles=2,
    )

    report = json.loads((animation_dir / "animation_report.json").read_text())
    assert list(report) == [
        "output_dir",
        "animations_dir",
        "spatial_dim",
        "overlay_wall_events",
        "interpolate_wall_event_positions",
        "interpolate_factor",
        "fps",
        "sample_count",
        "input_frame_count",
        "input_particle_count",
        "animation_frame_count",
        "animation_particle_count",
        "max_frames",
        "max_particles",
        "downsample_mode",
        "write_all_particles",
        "progress_enabled",
        "files",
    ]
    assert report["spatial_dim"] == 2
    assert report["animation_frame_count"] == 2
    assert report["animation_particle_count"] == 2
    assert report["files"] == [
        "trajectories_all_particles.gif",
        "trajectories_sampled_trails.gif",
    ]
    assert all((animation_dir / name).is_file() for name in report["files"])


def test_trajectory_animation_preserves_three_dimensional_projection_order(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    _write_full_run_3d(output_dir)

    animation_dir = export_trajectory_animations(
        output_dir,
        fps=1,
        sample_count=1,
        max_frames=2,
        max_particles=1,
        write_all_particles=False,
    )

    report = json.loads((animation_dir / "animation_report.json").read_text())
    assert report["spatial_dim"] == 3
    assert report["files"] == [
        "trajectories_sampled_trails_xy.gif",
        "trajectories_sampled_trails_xz.gif",
        "trajectories_sampled_trails_yz.gif",
    ]
    assert all((animation_dir / name).is_file() for name in report["files"])


def test_trajectory_animation_reports_missing_required_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    np.save(output_dir / "trajectory.npy", np.zeros((1, 1, 2), dtype=np.float64))

    with pytest.raises(FileNotFoundError, match="trajectory_frames file not found"):
        export_trajectory_animations(output_dir)

    pd.DataFrame({"time_s": [0.0]}).to_csv(
        output_dir / "trajectory_frames.csv", index=False
    )
    with pytest.raises(FileNotFoundError, match="final_particles file not found"):
        export_trajectory_animations(output_dir)


def test_trajectory_animation_rejects_object_trajectory(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    positions = np.empty((1, 1, 2), dtype=object)
    positions.fill(0.0)
    np.save(output_dir / "trajectory.npy", positions)

    with pytest.raises(ValueError, match="memory-mapped"):
        export_trajectory_animations(output_dir)


def test_trajectory_animation_rejects_pickled_case_arrays(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    output_dir = tmp_path / "run"
    _write_mechanics_case(case_dir)
    _write_full_run(output_dir)
    geometry_path = case_dir / "generated" / "comsol_geometry_2d.npz"
    with np.load(geometry_path) as geometry:
        payload = {name: geometry[name] for name in geometry.files}
    payload["unsafe_object"] = np.asarray([{"part_id": 7}], dtype=object)
    np.savez(geometry_path, **payload)

    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        export_trajectory_animations(
            output_dir,
            case_dir=case_dir,
            fps=1,
            sample_count=1,
            write_all_particles=False,
        )
