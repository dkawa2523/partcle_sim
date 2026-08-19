"""Render wall events, state trajectories, and COMSOL overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools._result_graph_common import PYPLOT as plt
from tools._result_graph_maps import (
    as_2d_mask,
    draw_device_structure,
    masked_field,
    robust_limits,
)
from tools.state_contract import STATE_ORDER
from tools.visualization_common import (
    STATE_COLORS,
    draw_boundary_edges,
    draw_domain_parts_by_medium,
)
from tools.visualization_data import axis_limits


def save_wall_event_locations(
    out_dir: Path,
    output_dir: Path,
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None = None,
) -> str | None:
    path = output_dir / "wall_events.csv"
    if not path.exists():
        return None
    wall_events = pd.read_csv(path)
    if wall_events.empty or not {"hit_x_m", "hit_y_m", "outcome"}.issubset(
        wall_events.columns
    ):
        return None
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_device_structure(
        ax,
        geometry,
        edges,
        edge_part_ids,
        label_domain_parts=True,
        label_boundary_parts=True,
        domain_alpha=0.34,
        boundary_linewidth=1.0,
        medium_summary=medium_summary,
        show_medium_legend=True,
    )
    color_map = {
        "stuck": "#d62728",
        "reflected_specular": "#4c78a8",
        "reflected_diffuse": "#72b7b2",
        "absorbed": "#2ca02c",
    }
    for outcome, group in wall_events.groupby("outcome"):
        ax.scatter(
            group["hit_x_m"].to_numpy(dtype=float),
            group["hit_y_m"].to_numpy(dtype=float),
            s=5,
            alpha=0.35,
            color=color_map.get(str(outcome), "#777777"),
            label=f"{outcome} ({len(group)})",
            linewidths=0,
        )
    ax.set_title("Wall Event Locations by Outcome")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    if edges is not None and edges.size:
        ax.set_xlim(float(np.nanmin(edges[:, :, 0])), float(np.nanmax(edges[:, :, 0])))
        ax.set_ylim(float(np.nanmin(edges[:, :, 1])), float(np.nanmax(edges[:, :, 1])))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    filename = "20_wall_event_locations_by_outcome.png"
    fig.savefig(out_dir / filename, dpi=170)
    plt.close(fig)
    return filename


def save_trajectories_by_state(
    out_dir: Path,
    positions: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None = None,
) -> str | None:
    if positions.shape[2] != 2 or sample_indices.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_device_structure(
        ax,
        geometry,
        edges,
        edge_part_ids,
        label_domain_parts=True,
        label_boundary_parts=True,
        domain_alpha=0.30,
        boundary_linewidth=0.95,
        medium_summary=medium_summary,
        show_medium_legend=True,
    )
    for name in STATE_ORDER:
        indices = sample_indices[labels[sample_indices] == name]
        if indices.size == 0:
            continue
        for index in indices:
            trajectory = positions[:, int(index), :]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=STATE_COLORS[name],
                linewidth=0.75,
                alpha=0.42,
            )
        end = positions[-1, indices, :]
        ax.scatter(
            end[:, 0],
            end[:, 1],
            s=8,
            color=STATE_COLORS[name],
            alpha=0.75,
            label=f"{name} ({indices.size})",
        )
    ax.set_title(
        "Sampled Particle Trajectories by Final State "
        f"({sample_indices.size} particles)"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    x_lim, y_lim = axis_limits(positions, edges)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    filename = "21_trajectories_by_final_state.png"
    fig.savefig(out_dir / filename, dpi=170)
    plt.close(fig)
    return filename


ComsolOverlaySpec = tuple[str, str, np.ndarray, str]


def _comsol_overlay_specs(
    field: dict[str, np.ndarray], mask: np.ndarray
) -> list[ComsolOverlaySpec]:
    specs: list[ComsolOverlaySpec] = []
    if "E_x" in field and "E_y" in field:
        electric_x = masked_field(field["E_x"], mask)
        electric_y = masked_field(field["E_y"], mask)
        specs.append(
            (
                "Electric field |E|",
                "|E| [V/m]",
                np.sqrt(electric_x * electric_x + electric_y * electric_y),
                "plasma",
            )
        )
    if "ux" in field and "uy" in field:
        ux = masked_field(field["ux"], mask)
        uy = masked_field(field["uy"], mask)
        specs.append(
            (
                "Flow speed |u|",
                "|u| [m/s]",
                np.sqrt(ux * ux + uy * uy),
                "viridis",
            )
        )
    return specs


def _draw_sampled_trajectories_by_state(
    ax: plt.Axes,
    positions: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
) -> None:
    for name in STATE_ORDER:
        indices = sample_indices[labels[sample_indices] == name]
        for index in indices:
            trajectory = positions[:, int(index), :]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=STATE_COLORS[name],
                linewidth=0.65,
                alpha=0.42,
                zorder=3,
            )


def _save_comsol_field_trajectory_overlay(
    out_dir: Path,
    positions: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    spec: ComsolOverlaySpec,
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None,
) -> str:
    title, label, array, cmap = spec
    vmin, vmax = robust_limits(array)
    x_lim, y_lim = axis_limits(positions, edges)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    draw_device_structure(
        ax,
        geometry,
        edges,
        edge_part_ids,
        label_domain_parts=True,
        label_boundary_parts=True,
        domain_alpha=0.30,
        boundary_linewidth=0.95,
        medium_summary=medium_summary,
        show_medium_legend=True,
    )
    pcm = ax.pcolormesh(
        xx,
        yy,
        np.ma.masked_invalid(array),
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.72,
    )
    _draw_sampled_trajectories_by_state(ax, positions, labels, sample_indices)
    draw_boundary_edges(
        ax, edges, edge_part_ids, linewidth=0.9, alpha=0.98, label_part_ids=False
    )
    ax.set_title(f"COMSOL-style {title} + Particle Trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label=label)
    fig.tight_layout()
    filename = "23_comsol_style_field_and_trajectories.png"
    fig.savefig(out_dir / filename, dpi=170)
    plt.close(fig)
    return filename


def _draw_wall_event_markers(ax: plt.Axes, wall_events: pd.DataFrame) -> bool:
    required = {"hit_x_m", "hit_y_m", "outcome"}
    if wall_events.empty or not required.issubset(wall_events.columns):
        return False
    for outcome, group in wall_events.groupby("outcome"):
        ax.scatter(
            group["hit_x_m"].to_numpy(dtype=float),
            group["hit_y_m"].to_numpy(dtype=float),
            s=4,
            alpha=0.32,
            label=f"{outcome} ({len(group)})",
            linewidths=0,
            zorder=4,
        )
    return True


def _save_comsol_density_event_overlay(
    out_dir: Path,
    positions: np.ndarray,
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    wall_events: pd.DataFrame,
    medium_summary: pd.DataFrame | None,
) -> str:
    x_lim, y_lim = axis_limits(positions, edges)
    all_points = positions.reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    draw_device_structure(
        ax,
        geometry,
        edges,
        edge_part_ids,
        label_domain_parts=True,
        label_boundary_parts=True,
        domain_alpha=0.34,
        boundary_linewidth=0.95,
        medium_summary=medium_summary,
        show_medium_legend=True,
    )
    hist = ax.hist2d(
        all_points[:, 0], all_points[:, 1], bins=190, cmap="magma", alpha=0.76
    )
    draw_domain_parts_by_medium(
        ax,
        geometry.get("mesh_vertices"),
        geometry.get("mesh_triangles"),
        geometry.get("mesh_triangle_part_ids"),
        geometry.get("mesh_quads"),
        geometry.get("mesh_quad_part_ids"),
        medium_summary=medium_summary,
        alpha=0.16,
        label_part_ids=False,
    )
    wall_event_legend_added = _draw_wall_event_markers(ax, wall_events)
    draw_boundary_edges(
        ax, edges, edge_part_ids, linewidth=0.9, alpha=0.98, label_part_ids=False
    )
    ax.set_title("COMSOL-style Particle Density + Wall Events")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.02, label="trajectory samples")
    if wall_event_legend_added:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    filename = "24_comsol_style_particle_density_and_events.png"
    fig.savefig(out_dir / filename, dpi=170)
    plt.close(fig)
    return filename


def save_comsol_style_overlays(
    out_dir: Path,
    positions: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    field: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    wall_events: pd.DataFrame,
    medium_summary: pd.DataFrame | None = None,
) -> list[str]:
    required = {"axis_0", "axis_1", "valid_mask"}
    if positions.shape[2] != 2 or not required.issubset(field):
        return []
    mask = as_2d_mask(field["valid_mask"])
    specs = _comsol_overlay_specs(field, mask)
    if not specs:
        return []
    field_plot = _save_comsol_field_trajectory_overlay(
        out_dir,
        positions,
        labels,
        sample_indices,
        np.asarray(field["axis_0"], dtype=np.float64),
        np.asarray(field["axis_1"], dtype=np.float64),
        specs[0],
        geometry,
        edges,
        edge_part_ids,
        medium_summary,
    )
    density_plot = _save_comsol_density_event_overlay(
        out_dir,
        positions,
        geometry,
        edges,
        edge_part_ids,
        wall_events,
        medium_summary,
    )
    return [field_plot, density_plot]
