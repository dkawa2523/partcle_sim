"""Generate result graphs when trajectory artifacts are unavailable."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tools._result_graph_common import PYPLOT as plt
from tools._result_graph_common import (
    case_field_payload,
    case_geometry_payload,
    final_position_columns,
    limits_from_points,
    medium_status_counts,
)
from tools._result_graph_maps import domain_medium_summary, draw_device_structure
from tools.state_contract import (
    STATE_ORDER,
    classify_particle_states,
    final_state_counts,
)
from tools.visualization_common import STATE_COLORS
from tools.visualization_data import list_files, load_boundary_geometry
from tools.visualization_reports import (
    ensure_visualization_dirs,
    read_optional_json_object,
)


@dataclass(frozen=True)
class CompactGraphData:
    output_dir: Path
    case_dir: Path | None
    out_dir: Path
    final_df: pd.DataFrame
    coordinates: list[str]
    axis_names: list[str]
    spatial_dim: int
    final_labels: np.ndarray
    state_counts: dict[str, int]
    edges: np.ndarray | None
    edge_part_ids: np.ndarray | None
    geometry_payload: dict[str, np.ndarray]
    medium_summary: pd.DataFrame
    run_report: dict[str, object]


def write_final_state_count_graphs(out_dir: Path, state_counts: dict[str, int]) -> None:
    names = list(STATE_ORDER)
    values = [int(state_counts.get(name, 0)) for name in names]
    pd.DataFrame({"state": names, "count": values}).to_csv(
        out_dir / "02_final_state_counts.csv", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    _draw_final_state_count_bars(axes[0], names, values)
    _draw_final_state_share(axes[1], names, values)
    fig.tight_layout()
    fig.savefig(out_dir / "02_final_state_bar_and_pie.png", dpi=170)
    plt.close(fig)


def _draw_final_state_count_bars(
    ax: plt.Axes, names: list[str], values: list[int]
) -> None:
    colors = [STATE_COLORS[name] for name in names]
    ax.bar(names, values, color=colors)
    ax.set_title("Final State Counts")
    ax.set_ylabel("count")
    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=9)


def _draw_final_state_share(ax: plt.Axes, names: list[str], values: list[int]) -> None:
    pie_values = [value for value in values if value > 0]
    pie_labels = [name for name, value in zip(names, values, strict=True) if value > 0]
    pie_colors = [STATE_COLORS[name] for name in pie_labels]
    if pie_values:
        ax.pie(
            pie_values,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
        )
    ax.set_title("Final State Share")


def _prepare_compact_graph_data(
    output_dir: Path, case_dir: Path | None, final_df: pd.DataFrame
) -> CompactGraphData:
    coordinates, axis_names = final_position_columns(final_df)
    spatial_dim = len(coordinates) if coordinates else 0
    if spatial_dim == 2:
        edges, edge_part_ids = load_boundary_geometry(case_dir)
        geometry_payload = case_geometry_payload(case_dir)
        field_payload = case_field_payload(case_dir)
        medium_summary = domain_medium_summary(geometry_payload, field_payload)
    else:
        edges, edge_part_ids = None, None
        geometry_payload = {}
        medium_summary = pd.DataFrame()
    out_dir = ensure_visualization_dirs(output_dir)["graphs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_report = read_optional_json_object(output_dir / "run_summary.json")
    return CompactGraphData(
        output_dir=output_dir,
        case_dir=case_dir,
        out_dir=out_dir,
        final_df=final_df,
        coordinates=coordinates,
        axis_names=axis_names,
        spatial_dim=spatial_dim,
        final_labels=classify_particle_states(final_df),
        state_counts=final_state_counts(final_df),
        edges=edges,
        edge_part_ids=edge_part_ids,
        geometry_payload=geometry_payload,
        medium_summary=medium_summary,
        run_report=run_report,
    )


def _write_compact_state_scatter(data: CompactGraphData) -> None:
    if data.spatial_dim < 2:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.9))
    if data.spatial_dim == 2:
        draw_device_structure(
            ax,
            data.geometry_payload,
            data.edges,
            data.edge_part_ids,
            domain_alpha=0.28,
            boundary_linewidth=0.9,
            medium_summary=data.medium_summary,
        )
    for name in STATE_ORDER:
        mask = data.final_labels == name
        if np.any(mask):
            particles = data.final_df.loc[mask]
            ax.scatter(
                particles[data.coordinates[0]],
                particles[data.coordinates[1]],
                s=5,
                color=STATE_COLORS[name],
                alpha=0.7,
                label=f"{name} ({int(mask.sum())})",
                zorder=2,
            )
    points = data.final_df[data.coordinates[:2]].to_numpy(dtype=np.float64)
    x_lim, y_lim = limits_from_points(points, data.edges)
    ax.set_title("Final Particle States")
    ax.set_xlabel(f"{data.axis_names[0]} [m]")
    ax.set_ylabel(f"{data.axis_names[1]} [m]")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(data.out_dir / "03_final_state_scatter_geometry.png", dpi=170)
    plt.close(fig)


def _compact_recommended_files(spatial_dim: int) -> list[str]:
    filenames = [
        "02_final_state_bar_and_pie.png",
        "02_final_state_counts.csv",
    ]
    if spatial_dim >= 2:
        filenames.append("03_final_state_scatter_geometry.png")
    return filenames


def _write_compact_graph_summary(
    data: CompactGraphData, missing_trajectory_artifacts: list[str]
) -> None:
    summary = {
        "plot_dir": str(data.out_dir.resolve()),
        "output_dir": str(data.output_dir.resolve()),
        "case_dir": str(data.case_dir.resolve()) if data.case_dir is not None else "",
        "graph_mode": "compact_final_state",
        "trajectory_artifacts_available": False,
        "missing_trajectory_artifacts": list(missing_trajectory_artifacts),
        "spatial_dim": data.spatial_dim,
        "axis_names": data.axis_names,
        "files": list_files(data.out_dir, (".png", ".csv", ".json")),
        "save_frame_count": 0,
        "particle_count": len(data.final_df),
        "final_state_counts": data.state_counts,
        "contact_state_counts_by_part": [],
        "used_wall_part_summary": False,
        "extra_graph_files": [],
        "domain_medium_status_counts": medium_status_counts(data.medium_summary),
        "recommended_for_reports": _compact_recommended_files(data.spatial_dim),
        "run_summary_coordinate_system": data.run_report.get("coordinate_system", ""),
    }
    (data.out_dir / "graph_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def export_compact_result_graphs(
    *,
    output_dir: Path,
    case_dir: Path | None,
    final_df: pd.DataFrame,
    missing_trajectory_artifacts: list[str],
) -> Path:
    data = _prepare_compact_graph_data(output_dir, case_dir, final_df)
    write_final_state_count_graphs(data.out_dir, data.state_counts)
    _write_compact_state_scatter(data)
    _write_compact_graph_summary(data, missing_trajectory_artifacts)
    return data.out_dir
