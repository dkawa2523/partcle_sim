from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba

from tools import visualization_common as visual
from tools import visualization_reports

MEDIUM_COLUMNS = [
    "part_id",
    "element_count",
    "field_supported_element_count",
    "support_fraction",
    "medium_status",
    "x_min_m",
    "x_max_m",
    "y_min_m",
    "y_max_m",
]


def _disjoint_triangles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for x in range(5):
        start = len(vertices)
        vertices.extend(
            [
                [x - 0.1, 0.6],
                [x + 0.1, 0.6],
                [float(x), 0.9],
            ]
        )
        triangles.append([start, start + 1, start + 2])
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.int32),
        np.asarray([20, 10, 10, 10, 30], dtype=np.int32),
    )


def test_run_summary_preserves_section_order_and_mapping_fallbacks(
    tmp_path: Path,
) -> None:
    payload = {
        "health_summary": {
            "status": "review",
            "particle_count": 4,
            "released_count": 3,
            "solver_core_s": 1.25,
            "estimated_numpy_bytes": 512,
            "invalid_mask_stopped_count": 1,
            "final_state_counts": {"escaped": 2, "stuck": 1},
        },
        "summary_files": {"zeta": "z.csv", "alpha": "a.csv"},
        "modules": {"trajectory": {}, "health": {}},
    }

    path = visualization_reports.write_run_summary(tmp_path, payload)

    assert path == tmp_path / "visualizations" / "reports" / "run_summary.md"
    assert path.read_text(encoding="utf-8") == (
        "# Run Summary\n"
        "\n"
        "- status: review\n"
        f"- output_dir: {tmp_path.resolve()}\n"
        "- particles: 4\n"
        "- released: 3\n"
        "- solver_core_s: 1.250\n"
        "- estimated_numpy_bytes: 512\n"
        "\n"
        "## Solver Health\n"
        "\n"
        "- invalid_mask_stopped_count: 1\n"
        "- numerical_boundary_stopped_count: 0\n"
        "- max_hits_reached_count: 0\n"
        "- unresolved_crossing_count: 0\n"
        "- nearest_projection_fallback_count: 0\n"
        "- contact_sliding_particle_count: 0\n"
        "- contact_endpoint_stopped_count: 0\n"
        "- nonfinite_position_count: 0\n"
        "- nonfinite_velocity_count: 0\n"
        "\n"
        "## Final States\n"
        "\n"
        "- escaped: 2\n"
        "- stuck: 1\n"
        "\n"
        "## Compact Summary Files\n"
        "\n"
        "- alpha: a.csv\n"
        "- zeta: z.csv\n"
        "\n"
        "## Visualization Modules\n"
        "\n"
        "- health\n"
        "- trajectory\n"
    )

    fallback = visualization_reports.write_run_summary(
        tmp_path / "fallback",
        {"health_summary": "invalid", "summary_files": [], "modules": None},
    ).read_text(encoding="utf-8")
    assert "- status: unknown" in fallback
    assert "- solver_core_s: not_recorded" in fallback
    assert "## Final States" not in fallback
    assert fallback.endswith("## Visualization Modules\n\n- none\n")


def test_domain_medium_summary_preserves_schema_order_and_nearest_grid_policy() -> None:
    vertices, triangles, part_ids = _disjoint_triangles()
    mask = np.zeros((6, 2), dtype=bool)
    mask[0, 1] = True
    mask[1, 1] = True

    summary = visual.domain_part_medium_summary(
        vertices,
        triangles,
        part_ids,
        None,
        None,
        np.arange(6, dtype=np.float64),
        np.asarray([0.0, 1.0]),
        mask,
    )

    assert summary.columns.tolist() == MEDIUM_COLUMNS
    assert summary[
        [
            "part_id",
            "element_count",
            "field_supported_element_count",
            "support_fraction",
            "medium_status",
        ]
    ].to_dict("records") == [
        {
            "part_id": 10,
            "element_count": 3,
            "field_supported_element_count": 1,
            "support_fraction": 1.0 / 3.0,
            "medium_status": "device_part_touching_solver_field",
        },
        {
            "part_id": 20,
            "element_count": 1,
            "field_supported_element_count": 1,
            "support_fraction": 1.0,
            "medium_status": "solver_medium_region",
        },
        {
            "part_id": 30,
            "element_count": 1,
            "field_supported_element_count": 0,
            "support_fraction": 0.0,
            "medium_status": "device_part_no_solver_field",
        },
    ]
    assert summary.loc[summary["part_id"] == 10, "x_min_m"].item() == pytest.approx(0.9)
    assert summary.loc[summary["part_id"] == 10, "x_max_m"].item() == pytest.approx(3.1)


def test_domain_medium_summary_preserves_missing_and_invalid_grid_fallbacks() -> None:
    vertices, triangles, _part_ids = _disjoint_triangles()
    empty = visual.domain_part_medium_summary(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert empty.columns.tolist() == MEDIUM_COLUMNS
    assert empty.empty

    fallback = visual.domain_part_medium_summary(
        vertices,
        triangles[:1],
        None,
        None,
        None,
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.ones((3, 3), dtype=bool),
    )
    assert fallback.to_dict("records") == [
        {
            "part_id": 0,
            "element_count": 1,
            "field_supported_element_count": 0,
            "support_fraction": 0.0,
            "medium_status": "device_part_no_solver_field",
            "x_min_m": -0.1,
            "x_max_m": 0.1,
            "y_min_m": 0.6,
            "y_max_m": 0.9,
        }
    ]


def test_boundary_edge_drawing_preserves_artist_order_styles_and_part_labels() -> None:
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots()

    visual.draw_boundary_edges(
        axis,
        edges,
        np.asarray([2, 1, 2], dtype=np.int64),
        linewidth=1.25,
        alpha=0.4,
        label_part_ids=True,
        label_fontsize=9.0,
    )

    assert len(axis.lines) == 3
    assert [np.asarray(line.get_xdata()).tolist() for line in axis.lines] == [
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    assert all(line.get_color() == "k" for line in axis.lines)
    assert all(line.get_linewidth() == 1.25 for line in axis.lines)
    assert all(line.get_alpha() == 0.4 for line in axis.lines)
    assert [text.get_text() for text in axis.texts] == ["1", "2"]
    assert [text.get_position() for text in axis.texts] == [
        (0.5, 1.0),
        (0.75, 0.25),
    ]
    assert all(text.get_fontsize() == 9.0 for text in axis.texts)
    plt.close(figure)

    figure, axis = plt.subplots()
    visual.draw_boundary_edges(
        axis,
        edges,
        np.asarray([1], dtype=np.int32),
        label_part_ids=True,
    )
    assert len(axis.lines) == 3
    assert len(axis.texts) == 0
    visual.draw_boundary_edges(axis, np.zeros((0, 2, 2)), label_part_ids=True)
    assert len(axis.lines) == 3
    plt.close(figure)


def test_medium_part_drawing_preserves_fill_outline_label_and_legend_fallbacks() -> (
    None
):
    vertices, triangles, _ = _disjoint_triangles()
    summary = pd.DataFrame(
        {
            "part_id": [20, 10],
            "medium_status": ["solver_medium_region", "unexpected_status"],
        }
    )
    figure, axis = plt.subplots()

    visual.draw_domain_parts_by_medium(
        axis,
        vertices,
        triangles[:3],
        np.asarray([20, 10, 30], dtype=np.int32),
        medium_summary=summary,
        alpha=0.5,
        label_part_ids=True,
        label_fontsize=7.0,
        show_legend=True,
    )

    assert len(axis.collections) == 4
    fill = axis.collections[0]
    assert isinstance(fill, PolyCollection)
    assert all(isinstance(item, LineCollection) for item in axis.collections[1:])
    np.testing.assert_allclose(
        np.asarray(fill.get_facecolor(), dtype=np.float64),
        np.asarray(
            [
                to_rgba("#f7f7f7", 0.5),
                to_rgba("#d9d9d9", 0.5),
                to_rgba("#f7f7f7", 0.5),
            ]
        ),
    )
    outline_widths: list[list[float]] = []
    for item in axis.collections[1:]:
        assert isinstance(item, LineCollection)
        outline_widths.append(np.asarray(item.get_linewidth()).tolist())
    assert outline_widths == [
        [0.78],
        [0.55],
        [0.55],
    ]
    assert [text.get_text() for text in axis.texts] == ["10", "20", "30"]
    assert all(text.get_fontsize() == 7.0 for text in axis.texts)
    legend = axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "solver medium region",
        "device part touching solver field",
        "device part without solver field",
    ]
    plt.close(figure)

    figure, axis = plt.subplots()
    visual.draw_domain_parts_by_medium(axis, None, show_legend=True)
    assert len(axis.collections) == 0
    assert axis.get_legend() is None
    plt.close(figure)

    figure, axis = plt.subplots()
    visual.draw_domain_parts_by_medium(
        axis,
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        np.asarray([[0, 0, 0]], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
    )
    assert len(axis.collections) == 1
    assert isinstance(axis.collections[0], PolyCollection)
    assert len(axis.texts) == 0
    assert axis.get_legend() is None
    plt.close(figure)
