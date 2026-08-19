from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools import visualization_common as visual
from tools import visualization_data, visualization_reports


def _test_visualization_helpers_have_focused_owners() -> None:
    moved_names = {
        "axis_limits",
        "load_boundary_geometry",
        "state_labels",
        "ensure_visualization_dirs",
        "write_run_summary",
    }
    assert moved_names.isdisjoint(vars(visual))


test_visualization_compatibility_facade_has_single_owners = (
    _test_visualization_helpers_have_focused_owners
)


def test_visualization_io_and_interpolation_contracts(tmp_path: Path) -> None:
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "particle_count": 2,
                "released_count": 2,
                "unresolved_crossing_count": 1,
                "final_state_counts": {"escaped": 1, "absorbed": 1},
                "timing_s": {"solver_core_s": 0.25},
                "memory_estimate_bytes": {"estimated_numpy_bytes": 128},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "debug_diagnostics.json").write_text(
        json.dumps(
            {
                "collision": {
                    "unresolved_crossing_count": 99,
                    "nearest_projection_fallback_count": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "final_particles.csv").write_text("not,a,valid,state,file\n")

    health = visualization_reports.build_run_health_summary(tmp_path)

    assert health["status"] == "review"
    assert health["particle_count"] == 2
    assert health["unresolved_crossing_count"] == 1
    assert health["nearest_projection_fallback_count"] == 2
    state_counts = health["final_state_counts"]
    assert isinstance(state_counts, dict)
    assert state_counts["escaped"] == 1
    assert state_counts["absorbed"] == 1
    assert state_counts["stuck"] == 0
    assert "boundary_event_failure_count" not in health
    assert "active_outside_geometry_count" not in health
    assert health["solver_core_s"] == 0.25
    assert visualization_reports.write_visualization_index(
        tmp_path, {"health": health}
    ).is_file()
    assert visualization_reports.write_run_summary(
        tmp_path,
        {
            "health_summary": health,
            "modules": {"trajectory": {}},
            "summary_files": {"health": "run_summary.json"},
        },
    ).is_file()

    positions = np.asarray(
        [[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]],
        dtype=np.float64,
    )
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    np.save(tmp_path / "trajectory.npy", positions)
    path, spatial_dim = visualization_data.resolve_positions_path(tmp_path)
    assert path.name == "trajectory.npy"
    assert spatial_dim == 2
    dense_positions, dense_times = visualization_data.interpolate_frames(
        positions, times, 2
    )
    assert dense_positions.shape == (3, 2, 2)
    assert dense_times.tolist() == [0.0, 0.5, 1.0]
    assert visualization_data.interpolate_particle_position(
        positions, times, 0, 0.5
    ).tolist() == [1.0, 1.0]
    overlay, frame_ids = visualization_data.prepare_event_overlay(
        pd.DataFrame({"time_s": [0.5], "particle_id": [0]}),
        np.asarray([0], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        positions,
        times,
        True,
    )
    assert overlay.tolist() == [[1.0, 1.0]]
    assert frame_ids.tolist() == [1]
    np.save(tmp_path / "trajectory.npy", np.zeros((2, 2)))
    with np.testing.assert_raises(ValueError):
        visualization_data.resolve_positions_path(tmp_path)

    optional_json = tmp_path / "optional.json"
    assert visualization_reports.read_optional_json_object(optional_json) == {}
    optional_json.write_text("[]", encoding="utf-8")
    assert visualization_reports.read_optional_json_object(optional_json) == {}
    optional_json.write_text("{", encoding="utf-8")
    with np.testing.assert_raises(json.JSONDecodeError):
        visualization_reports.read_optional_json_object(optional_json)


def test_visualization_geometry_and_wall_contracts(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    generated = case_dir / "generated"
    generated.mkdir(parents=True)
    edges = np.asarray([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float64)
    np.savez(
        generated / "comsol_geometry_2d.npz",
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([7], dtype=np.int32),
    )
    loaded_edges, loaded_ids = visualization_data.load_boundary_geometry(case_dir)
    assert loaded_edges is not None
    assert np.array_equal(loaded_edges, edges)
    assert loaded_ids is not None
    assert loaded_ids.tolist() == [7]

    pd.DataFrame(
        {
            "time_s": [0.1],
            "particle_id": [3],
            "part_id": [7],
            "outcome": ["absorbed"],
            "wall_mode": ["absorb"],
        }
    ).to_csv(tmp_path / "wall_events.csv", index=False)
    assert len(visualization_data.load_wall_events(tmp_path)) == 1
    pd.DataFrame(
        {"part_id": [8], "outcome": ["stuck"], "wall_mode": ["stick"], "count": [2]}
    ).to_csv(tmp_path / "wall_summary.csv", index=False)
    assert visualization_data.load_wall_part_summary(tmp_path).iloc[0]["count"] == 2
    (tmp_path / "wall_summary.csv").unlink()
    with np.testing.assert_raises_regex(
        FileNotFoundError, "wall_summary.csv not found:"
    ):
        visualization_data.load_wall_part_summary(tmp_path)
    steps = pd.DataFrame(
        {
            "active_count": [2, 1],
            "invalid_mask_stopped_count_step": [0, 1],
        }
    )
    assert visualization_data.step_state_count_series(
        steps, "active_total"
    ).tolist() == [2.0, 1.0]
    assert visualization_data.step_state_count_series(
        steps, "invalid_mask_stopped"
    ).tolist() == [0.0, 1.0]
    assert visualization_data.axis_limits(edges.reshape(1, 2, 2), edges)[0][0] < 0.0

    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    part_ids = np.asarray([7, 8], dtype=np.int32)
    summary = visual.domain_part_medium_summary(
        vertices,
        triangles,
        part_ids,
        None,
        None,
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([[True, False], [False, True]]),
    )
    assert set(summary["part_id"]) == {7, 8}
    quad_summary = visual.domain_part_medium_summary(
        vertices,
        None,
        None,
        np.asarray([[0, 1, 3, 2]], dtype=np.int32),
        np.asarray([9], dtype=np.int32),
        None,
        None,
        None,
    )
    assert quad_summary.iloc[0]["part_id"] == 9
    assert set(visual.medium_status_by_part(summary)) == {7, 8}
    figure, axis = plt.subplots()
    visual.draw_boundary_edges(axis, edges, np.asarray([7], dtype=np.int32))
    visual.draw_domain_part_outlines(axis, vertices, triangles, part_ids)
    visual.draw_domain_parts(axis, vertices, triangles, part_ids)
    visual.draw_domain_parts_by_medium(
        axis,
        vertices,
        triangles,
        part_ids,
        medium_summary=summary,
        show_legend=True,
    )
    plt.close(figure)

    grid = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    sampled = visual.sample_grid_points(
        grid,
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([[0.5, 0.5]]),
    )
    assert sampled.tolist() == [1.5]
    with np.testing.assert_raises(ValueError):
        visual.sample_grid_points(
            grid,
            np.asarray([0.0, np.nan]),
            np.asarray([0.0, 1.0]),
            np.asarray([[0.5, 0.5]]),
        )
    assert visualization_data.as_2d(grid[np.newaxis, ...]).shape == (2, 2)
    assert "wall_events.csv" in visualization_data.list_files(tmp_path, {".csv"})
