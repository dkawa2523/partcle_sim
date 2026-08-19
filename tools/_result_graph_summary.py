"""Orchestrate result graphs and build their stable summary schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools._result_graph_common import PYPLOT as plt
from tools._result_graph_common import (
    TRAJECTORY_ARTIFACTS,
    FullGraphData,
    medium_status_counts,
    nearest_boundary_part_ids,
    representative_particle_sample,
)
from tools._result_graph_compact import (
    export_compact_result_graphs,
    write_final_state_count_graphs,
)
from tools._result_graph_events import (
    save_comsol_style_overlays,
    save_trajectories_by_state,
    save_wall_event_locations,
)
from tools._result_graph_fields import (
    save_drag_gas_property_maps,
    save_field_maps,
)
from tools._result_graph_maps import save_geometry_maps
from tools._result_graph_trajectories import (
    load_full_graph_data,
    write_final_state_scatter,
    write_sampled_trajectories,
    write_speed_distribution,
    write_state_timeline,
    write_trajectory_density,
)
from tools.state_contract import final_state_counts
from tools.visualization_common import STATE_COLORS
from tools.visualization_data import list_files


def missing_trajectory_artifacts(output_dir: Path) -> list[str]:
    return [name for name in TRAJECTORY_ARTIFACTS if not (output_dir / name).exists()]


def _contact_state_summary(data: FullGraphData) -> pd.DataFrame:
    if "contact_part_id" not in data.final_df.columns:
        return pd.DataFrame()
    contact_mask = np.isin(
        data.final_labels, ("contact_sliding", "contact_endpoint_stopped")
    )
    contact_rows = data.final_df.loc[contact_mask, ["contact_part_id"]].copy()
    if contact_rows.empty:
        return pd.DataFrame()
    contact_rows["contact_state"] = data.final_labels[contact_mask]
    return (
        contact_rows.groupby(["contact_part_id", "contact_state"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["contact_part_id", "contact_state"])
    )


def _write_contact_state_counts(data: FullGraphData) -> list[dict[str, object]]:
    contact_summary = _contact_state_summary(data)
    if contact_summary.empty:
        return []
    contact_summary.to_csv(
        data.out_dir / "07_contact_states_by_boundary_part.csv", index=False
    )
    pivot = contact_summary.pivot_table(
        index="contact_part_id",
        columns="contact_state",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bottom = np.zeros(pivot.shape[0], dtype=np.float64)
    x_indices = np.arange(pivot.shape[0], dtype=np.float64)
    for state_name in ("contact_sliding", "contact_endpoint_stopped"):
        if state_name in pivot.columns:
            values = pivot[state_name].to_numpy(dtype=np.float64)
            ax.bar(
                x_indices,
                values,
                bottom=bottom,
                color=STATE_COLORS[state_name],
                label=state_name,
                width=0.78,
            )
            bottom += values
    ax.set_xticks(x_indices, [str(int(value)) for value in pivot.index.to_numpy()])
    ax.set_title("Contact States by Boundary Part")
    ax.set_xlabel("boundary part_id")
    ax.set_ylabel("particle count")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(data.out_dir / "07_contact_states_by_boundary_part.png", dpi=170)
    plt.close(fig)
    return [
        {
            "contact_part_id": int(row["contact_part_id"]),
            "contact_state": str(row["contact_state"]),
            "count": int(row["count"]),
        }
        for _, row in contact_summary.iterrows()
    ]


def _write_wall_law_counts(data: FullGraphData) -> None:
    wall_law_counts = data.report.get("wall_law_counts", {})
    if not isinstance(wall_law_counts, dict) or not wall_law_counts:
        return
    names = list(wall_law_counts.keys())
    values = [int(wall_law_counts[name]) for name in names]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(names, values, color="#4c78a8")
    ax.set_title("Wall Interaction Counts by Law")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(data.out_dir / "08_wall_law_counts.png", dpi=170)
    plt.close(fig)


def _ordered_wall_outcomes(columns: pd.Index) -> list[str]:
    preferred = (
        "stuck",
        "reflected_specular",
        "reflected_diffuse",
        "absorbed",
        "escaped",
    )
    ordered = [name for name in preferred if name in columns]
    ordered.extend(name for name in columns if name not in ordered)
    return ordered


def _write_wall_part_interactions(data: FullGraphData) -> None:
    if data.wall_part_summary.empty:
        return
    pivot = data.wall_part_summary.pivot_table(
        index="part_id",
        columns="outcome",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()
    if pivot.empty:
        return
    color_map = {
        "stuck": "#d62728",
        "reflected_specular": "#4c78a8",
        "reflected_diffuse": "#72b7b2",
        "absorbed": "#2ca02c",
        "escaped": "#ff7f0e",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bottom = np.zeros(pivot.shape[0], dtype=np.float64)
    x_indices = np.arange(pivot.shape[0], dtype=np.float64)
    for outcome in _ordered_wall_outcomes(pivot.columns):
        values = pivot[outcome].to_numpy(dtype=np.float64)
        ax.bar(
            x_indices,
            values,
            bottom=bottom,
            color=color_map.get(outcome, "#999999"),
            label=outcome,
            width=0.78,
        )
        bottom += values
    ax.set_xticks(x_indices, [str(int(value)) for value in pivot.index.to_numpy()])
    ax.set_title("Wall Interactions by Boundary Part / Outcome")
    ax.set_xlabel("boundary part_id")
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(data.out_dir / "09_wall_interactions_by_part_outcome.png", dpi=170)
    plt.close(fig)


def _wall_stuck_counts(wall_part_summary: pd.DataFrame) -> pd.DataFrame:
    if wall_part_summary.empty:
        return pd.DataFrame()
    stuck_summary = wall_part_summary[wall_part_summary["outcome"] == "stuck"]
    if stuck_summary.empty:
        return pd.DataFrame()
    stuck_counts = pd.DataFrame(
        stuck_summary.groupby("part_id", as_index=False).agg(
            stuck_count=("count", "sum")
        )
    )
    return stuck_counts.sort_values(["stuck_count", "part_id"], ascending=[False, True])


def _nearest_stuck_counts(data: FullGraphData) -> pd.DataFrame:
    stuck_mask = data.final_labels == "stuck"
    if data.edges is None or data.edge_part_ids is None or not np.any(stuck_mask):
        return pd.DataFrame()
    stuck_points = data.final_df.loc[stuck_mask, data.position_columns[:2]].to_numpy(
        dtype=np.float64
    )
    stuck_part_ids, stuck_distance = nearest_boundary_part_ids(
        stuck_points, data.edges, data.edge_part_ids
    )
    return (
        pd.DataFrame(
            {
                "part_id": stuck_part_ids.astype(int),
                "distance_to_edge_m": stuck_distance.astype(float),
            }
        )
        .groupby("part_id", as_index=False)
        .agg(
            stuck_count=("part_id", "size"),
            mean_distance_to_edge_m=("distance_to_edge_m", "mean"),
        )
        .sort_values(["stuck_count", "part_id"], ascending=[False, True])
    )


def _write_stuck_counts(data: FullGraphData) -> None:
    stuck_counts = _wall_stuck_counts(data.wall_part_summary)
    used_wall_summary = not stuck_counts.empty
    if not used_wall_summary:
        stuck_counts = _nearest_stuck_counts(data)
    if stuck_counts.empty:
        return
    stuck_counts.to_csv(
        data.out_dir / "10_stuck_counts_by_boundary_part.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        stuck_counts["part_id"].astype(str),
        stuck_counts["stuck_count"],
        color="#c44e52",
    )
    title = (
        "Wall Sticking Counts by Boundary Part"
        if used_wall_summary
        else "Final Stuck Positions by Boundary Part"
    )
    ax.set_title(title)
    ax.set_xlabel("boundary part_id")
    ax.set_ylabel("stuck count")
    for index, value in enumerate(stuck_counts["stuck_count"].tolist()):
        ax.text(index, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(data.out_dir / "10_stuck_counts_by_boundary_part.png", dpi=170)
    plt.close(fig)


def _save_extra_graphs(data: FullGraphData, pick: np.ndarray) -> list[str]:
    if data.spatial_dim != 2:
        return []
    saved = save_geometry_maps(
        data.out_dir,
        data.geometry_payload,
        data.edges,
        data.edge_part_ids,
        data.medium_summary,
    )
    saved.extend(
        save_field_maps(
            data.out_dir,
            data.field_payload,
            data.geometry_payload,
            data.edges,
            data.edge_part_ids,
            data.medium_summary,
        )
    )
    saved.extend(
        save_drag_gas_property_maps(
            data.out_dir,
            data.field_payload,
            data.geometry_payload,
            data.edges,
            data.edge_part_ids,
            data.report,
            data.medium_summary,
        )
    )
    event_plot = save_wall_event_locations(
        data.out_dir,
        data.output_dir,
        data.geometry_payload,
        data.edges,
        data.edge_part_ids,
        data.medium_summary,
    )
    if event_plot is not None:
        saved.append(event_plot)
    trajectory_plot = save_trajectories_by_state(
        data.out_dir,
        data.positions,
        data.final_labels,
        pick,
        data.geometry_payload,
        data.edges,
        data.edge_part_ids,
        data.medium_summary,
    )
    if trajectory_plot is not None:
        saved.append(trajectory_plot)
    saved.extend(
        save_comsol_style_overlays(
            data.out_dir,
            data.positions,
            data.final_labels,
            pick,
            data.field_payload,
            data.geometry_payload,
            data.edges,
            data.edge_part_ids,
            data.wall_events,
            data.medium_summary,
        )
    )
    return saved


def _write_full_graph_summary(
    data: FullGraphData,
    state_counts: dict[str, int],
    contact_state_counts: list[dict[str, object]],
    extra_graph_files: list[str],
) -> None:
    summary = {
        "plot_dir": str(data.out_dir.resolve()),
        "output_dir": str(data.output_dir.resolve()),
        "case_dir": str(data.case_dir.resolve()) if data.case_dir is not None else "",
        "graph_mode": "trajectory_full",
        "trajectory_artifacts_available": True,
        "missing_trajectory_artifacts": [],
        "spatial_dim": data.spatial_dim,
        "files": list_files(data.out_dir, (".png", ".csv", ".json")),
        "save_frame_count": len(data.frame_df),
        "particle_count": len(data.final_df),
        "final_state_counts": state_counts,
        "contact_state_counts_by_part": contact_state_counts,
        "used_wall_part_summary": bool(not data.wall_part_summary.empty),
        "extra_graph_files": extra_graph_files,
        "domain_medium_status_counts": medium_status_counts(data.medium_summary),
    }
    (data.out_dir / "graph_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def export_result_graphs(
    output_dir: Path, case_dir: Path | None = None, sample_trajectories: int = 300
) -> Path:
    final_csv = output_dir / "final_particles.csv"
    if not final_csv.exists():
        raise FileNotFoundError(f"final_particles.csv not found: {final_csv}")
    final_df = pd.read_csv(final_csv)
    missing_artifacts = missing_trajectory_artifacts(output_dir)
    if missing_artifacts:
        return export_compact_result_graphs(
            output_dir=output_dir,
            case_dir=case_dir,
            final_df=final_df,
            missing_trajectory_artifacts=missing_artifacts,
        )

    data = load_full_graph_data(output_dir, case_dir, final_df)
    state_counts = final_state_counts(final_df)
    pick = representative_particle_sample(
        data.final_labels, min(sample_trajectories, data.positions.shape[1])
    )
    write_state_timeline(data)
    write_final_state_count_graphs(data.out_dir, state_counts)
    write_final_state_scatter(data)
    write_trajectory_density(data)
    write_speed_distribution(data)
    write_sampled_trajectories(data, pick)
    contact_state_counts = _write_contact_state_counts(data)
    _write_wall_law_counts(data)
    _write_wall_part_interactions(data)
    _write_stuck_counts(data)
    extra_graph_files = _save_extra_graphs(data, pick)
    _write_full_graph_summary(
        data, state_counts, contact_state_counts, extra_graph_files
    )
    return data.out_dir
