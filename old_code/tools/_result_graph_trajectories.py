"""Load and render full trajectory, timeline, density, and speed graphs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools._result_graph_common import (
    PROJECTIONS_3D,
    FullGraphData,
    case_field_payload,
    case_geometry_payload,
    cumulative_event_count,
    final_result_columns,
)
from tools._result_graph_common import PYPLOT as plt
from tools._result_graph_maps import domain_medium_summary, draw_device_structure
from tools.state_contract import STATE_ORDER, classify_particle_states
from tools.visualization_common import (
    STATE_COLORS,
    STEP_STATE_ORDER,
    draw_boundary_edges,
    draw_domain_parts_by_medium,
)
from tools.visualization_data import (
    axis_limits,
    load_boundary_geometry,
    load_wall_events,
    load_wall_part_summary,
    step_state_count_series,
)
from tools.visualization_reports import (
    ensure_visualization_dirs,
    read_optional_json_object,
)


def load_full_graph_data(
    output_dir: Path, case_dir: Path | None, final_df: pd.DataFrame
) -> FullGraphData:
    step_df = pd.read_csv(output_dir / "step_summary.csv")
    frame_df = pd.read_csv(output_dir / "trajectory_frames.csv")
    positions = np.asarray(np.load(output_dir / "trajectory.npy"), dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] not in {2, 3}:
        raise ValueError(
            "positions file must have shape (frames, particles, 2|3), "
            f"got {positions.shape}"
        )
    spatial_dim = int(positions.shape[2])
    position_columns, axis_names, velocity_columns = final_result_columns(
        final_df, spatial_dim
    )
    report = read_optional_json_object(output_dir / "run_summary.json")
    edges, edge_part_ids = (
        load_boundary_geometry(case_dir) if spatial_dim == 2 else (None, None)
    )
    geometry_payload = case_geometry_payload(case_dir) if spatial_dim == 2 else {}
    field_payload = case_field_payload(case_dir) if spatial_dim == 2 else {}
    medium_summary = (
        domain_medium_summary(geometry_payload, field_payload)
        if spatial_dim == 2
        else pd.DataFrame()
    )
    wall_events = load_wall_events(output_dir)
    wall_part_summary = load_wall_part_summary(output_dir)
    out_dir = ensure_visualization_dirs(output_dir)["graphs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return FullGraphData(
        output_dir=output_dir,
        case_dir=case_dir,
        final_df=final_df,
        frame_df=frame_df,
        step_df=step_df,
        positions=positions,
        spatial_dim=spatial_dim,
        position_columns=position_columns,
        axis_names=axis_names,
        velocity_columns=velocity_columns,
        report=report,
        edges=edges,
        edge_part_ids=edge_part_ids,
        geometry_payload=geometry_payload,
        field_payload=field_payload,
        medium_summary=medium_summary,
        wall_events=wall_events,
        wall_part_summary=wall_part_summary,
        out_dir=out_dir,
        final_labels=classify_particle_states(final_df),
    )


def write_state_timeline(data: FullGraphData) -> None:
    time_s = data.step_df["time_s"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name in STEP_STATE_ORDER:
        values = step_state_count_series(data.step_df, name)
        if np.any(values):
            ax.plot(
                time_s,
                values,
                label=name,
                color=STATE_COLORS[name],
                linewidth=2.0,
            )
    ax.set_title("Solver State Counts vs Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    reflected = cumulative_event_count(
        time_s, data.wall_events, {"reflected_specular", "reflected_diffuse"}
    )
    if np.any(reflected):
        _add_reflection_timeline(ax, time_s, reflected)
    else:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(data.out_dir / "01_state_counts_time_series.png", dpi=170)
    plt.close(fig)
    if not data.wall_events.empty:
        _write_wall_event_timeline(data.out_dir, time_s, data.wall_events, reflected)


def _add_reflection_timeline(
    ax: plt.Axes, time_s: np.ndarray, reflected: np.ndarray
) -> None:
    ax2 = ax.twinx()
    ax2.plot(
        time_s,
        reflected,
        label="cumulative_reflections",
        color="#4c78a8",
        linewidth=1.8,
        linestyle="--",
    )
    ax2.set_ylabel("cumulative reflected wall events")
    ax2.tick_params(axis="y", colors="#4c78a8")
    ax.text(
        0.02,
        0.96,
        f"reflections: {int(reflected[-1])}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#999999",
            "alpha": 0.84,
        },
    )
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)


def _write_wall_event_timeline(
    out_dir: Path,
    time_s: np.ndarray,
    wall_events: pd.DataFrame,
    reflected: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            "time_s": time_s,
            "cumulative_reflected": reflected.astype(int),
            "cumulative_stuck": cumulative_event_count(
                time_s, wall_events, {"stuck"}
            ).astype(int),
            "cumulative_absorbed": cumulative_event_count(
                time_s, wall_events, {"absorbed"}
            ).astype(int),
            "cumulative_escaped": cumulative_event_count(
                time_s, wall_events, {"escaped"}
            ).astype(int),
        }
    ).to_csv(out_dir / "01_wall_event_cumulative_counts.csv", index=False)


def write_final_state_scatter(data: FullGraphData) -> None:
    if data.spatial_dim == 2:
        _write_final_state_scatter_2d(data)
    else:
        _write_final_state_scatter_3d(data)


def _write_final_state_scatter_2d(data: FullGraphData) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.9))
    x_lim, y_lim = axis_limits(data.positions, data.edges)
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
            subset = data.final_df.loc[mask]
            ax.scatter(
                subset[data.position_columns[0]],
                subset[data.position_columns[1]],
                s=5,
                color=STATE_COLORS[name],
                alpha=0.7,
                label=f"{name} ({int(mask.sum())})",
                zorder=2,
            )
    ax.set_title("Final Particle States over Geometry")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(data.out_dir / "03_final_state_scatter_geometry.png", dpi=170)
    plt.close(fig)


def _write_final_state_scatter_3d(data: FullGraphData) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
    for ax, (tag, axis_a, axis_b) in zip(axes, PROJECTIONS_3D, strict=True):
        x_lim, y_lim = axis_limits(data.positions, None, projection=(axis_a, axis_b))
        for name in STATE_ORDER:
            mask = data.final_labels == name
            if np.any(mask):
                subset = data.final_df.loc[mask]
                ax.scatter(
                    subset[data.position_columns[axis_a]],
                    subset[data.position_columns[axis_b]],
                    s=4,
                    color=STATE_COLORS[name],
                    alpha=0.65,
                    label=name if tag == "xy" else "",
                    zorder=2,
                )
        ax.set_title(f"Final States ({tag.upper()} projection)")
        ax.set_xlabel(f"{data.axis_names[axis_a]} [m]")
        ax.set_ylabel(f"{data.axis_names[axis_b]} [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(data.out_dir / "03_final_state_scatter_geometry.png", dpi=170)
    plt.close(fig)


def write_trajectory_density(data: FullGraphData) -> None:
    if data.spatial_dim == 2:
        _write_trajectory_density_2d(data)
    else:
        _write_trajectory_density_3d(data)


def _write_trajectory_density_2d(data: FullGraphData) -> None:
    all_points = data.positions.reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(8.2, 5.9))
    x_lim, y_lim = axis_limits(data.positions, data.edges)
    draw_device_structure(
        ax,
        data.geometry_payload,
        data.edges,
        data.edge_part_ids,
        domain_alpha=0.24,
        boundary_linewidth=0.75,
        medium_summary=data.medium_summary,
    )
    hist = ax.hist2d(
        all_points[:, 0], all_points[:, 1], bins=180, cmap="magma", alpha=0.82
    )
    draw_domain_parts_by_medium(
        ax,
        data.geometry_payload.get("mesh_vertices"),
        data.geometry_payload.get("mesh_triangles"),
        data.geometry_payload.get("mesh_triangle_part_ids"),
        data.geometry_payload.get("mesh_quads"),
        data.geometry_payload.get("mesh_quad_part_ids"),
        medium_summary=data.medium_summary,
        alpha=0.12,
    )
    draw_boundary_edges(ax, data.edges, data.edge_part_ids, linewidth=0.75, alpha=0.95)
    ax.set_title("Trajectory Density Heatmap")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.02, label="samples")
    fig.tight_layout()
    fig.savefig(data.out_dir / "04_trajectory_density_heatmap.png", dpi=170)
    plt.close(fig)


def _write_trajectory_density_3d(data: FullGraphData) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
    for ax, (tag, axis_a, axis_b) in zip(axes, PROJECTIONS_3D, strict=True):
        points = data.positions[:, :, [axis_a, axis_b]].reshape(-1, 2)
        x_lim, y_lim = axis_limits(data.positions, None, projection=(axis_a, axis_b))
        hist = ax.hist2d(points[:, 0], points[:, 1], bins=150, cmap="magma")
        fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(f"Trajectory Density ({tag.upper()})")
        ax.set_xlabel(f"{'xyz'[axis_a]} [m]")
        ax.set_ylabel(f"{'xyz'[axis_b]} [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
    fig.tight_layout()
    fig.savefig(data.out_dir / "04_trajectory_density_heatmap.png", dpi=170)
    plt.close(fig)


def write_speed_distribution(data: FullGraphData) -> None:
    speed = np.linalg.norm(
        data.final_df[data.velocity_columns].to_numpy(dtype=np.float64), axis=1
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for name in STATE_ORDER:
        mask = data.final_labels == name
        if np.any(mask):
            ax.hist(
                speed[mask],
                bins=40,
                alpha=0.55,
                color=STATE_COLORS[name],
                label=name,
            )
    ax.set_title("Final Speed Distribution by State")
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(data.out_dir / "05_speed_distribution_by_state.png", dpi=170)
    plt.close(fig)


def write_sampled_trajectories(data: FullGraphData, pick: np.ndarray) -> None:
    if data.spatial_dim == 2:
        _write_sampled_trajectories_2d(data, pick)
    else:
        _write_sampled_trajectories_3d(data, pick)


def _write_sampled_trajectories_2d(data: FullGraphData, pick: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.9))
    x_lim, y_lim = axis_limits(data.positions, data.edges)
    draw_device_structure(
        ax,
        data.geometry_payload,
        data.edges,
        data.edge_part_ids,
        domain_alpha=0.26,
        boundary_linewidth=0.8,
        medium_summary=data.medium_summary,
    )
    for index in pick:
        trajectory = data.positions[:, index, :]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            linewidth=0.65,
            alpha=0.35,
            color="#1f77b4",
            zorder=2,
        )
    ax.set_title(f"Sampled Trajectories Overlay ({len(pick)} particles)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    fig.tight_layout()
    fig.savefig(data.out_dir / "06_sampled_trajectories_overlay.png", dpi=170)
    plt.close(fig)


def _write_sampled_trajectories_3d(data: FullGraphData, pick: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
    for ax, (tag, axis_a, axis_b) in zip(axes, PROJECTIONS_3D, strict=True):
        x_lim, y_lim = axis_limits(data.positions, None, projection=(axis_a, axis_b))
        for index in pick:
            trajectory = data.positions[:, index, :]
            ax.plot(
                trajectory[:, axis_a],
                trajectory[:, axis_b],
                linewidth=0.6,
                alpha=0.35,
                color="#1f77b4",
                zorder=2,
            )
        ax.set_title(f"Sampled Trajectories ({tag.upper()})")
        ax.set_xlabel(f"{'xyz'[axis_a]} [m]")
        ax.set_ylabel(f"{'xyz'[axis_b]} [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(data.out_dir / "06_sampled_trajectories_overlay.png", dpi=170)
    plt.close(fig)
