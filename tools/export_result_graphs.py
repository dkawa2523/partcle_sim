from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.visualization_common import (
    STATE_ORDER,
    STATE_COLORS,
    axis_limits,
    draw_edges,
    ensure_visualization_dirs,
    final_state_counts,
    list_files,
    load_boundary_geometry,
    load_wall_part_summary,
    state_labels,
    step_state_count_series,
)


def _nearest_boundary_part_ids(points: np.ndarray, edges: np.ndarray, part_ids: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    segs = np.asarray(edges, dtype=np.float64)
    if pts.size == 0 or segs.size == 0:
        return np.zeros(pts.shape[0], dtype=np.int32), np.full(pts.shape[0], np.nan, dtype=np.float64)
    p0 = segs[:, 0, :]
    p1 = segs[:, 1, :]
    ab = p1 - p0
    ab2 = np.sum(ab * ab, axis=1)
    out_ids = np.zeros(pts.shape[0], dtype=np.int32)
    out_dist = np.full(pts.shape[0], np.inf, dtype=np.float64)
    edge_part_ids = np.asarray(part_ids, dtype=np.int32) if part_ids is not None else np.zeros(segs.shape[0], dtype=np.int32)
    for i, point in enumerate(pts):
        ap = point[None, :] - p0
        t = np.zeros(segs.shape[0], dtype=np.float64)
        mask = ab2 > 1e-30
        t[mask] = np.clip(np.sum(ap[mask] * ab[mask], axis=1) / ab2[mask], 0.0, 1.0)
        proj = p0 + t[:, None] * ab
        d = np.linalg.norm(proj - point[None, :], axis=1)
        j = int(np.argmin(d))
        out_ids[i] = int(edge_part_ids[j]) if j < edge_part_ids.size else 0
        out_dist[i] = float(d[j])
    return out_ids, out_dist


def export_result_graphs(output_dir: Path, case_dir: Path | None = None, sample_trajectories: int = 300) -> Path:
    final_csv = output_dir / "final_particles.csv"
    frames_csv = output_dir / "save_frames.csv"
    steps_csv = output_dir / "runtime_step_summary.csv"
    positions_2d_npy = output_dir / "positions_2d.npy"
    positions_3d_npy = output_dir / "positions_3d.npy"
    report_json = output_dir / "solver_report.json"
    if not final_csv.exists():
        raise FileNotFoundError(f"final_particles.csv not found: {final_csv}")
    if not frames_csv.exists():
        raise FileNotFoundError(f"save_frames.csv not found: {frames_csv}")
    if not steps_csv.exists():
        raise FileNotFoundError(f"runtime_step_summary.csv not found: {steps_csv}")
    if positions_2d_npy.exists():
        positions_npy = positions_2d_npy
    elif positions_3d_npy.exists():
        positions_npy = positions_3d_npy
    else:
        raise FileNotFoundError(f"positions_2d.npy or positions_3d.npy not found in: {output_dir}")

    final_df = pd.read_csv(final_csv)
    step_df = pd.read_csv(steps_csv)
    frame_df = pd.read_csv(frames_csv)
    positions = np.asarray(np.load(positions_npy), dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] not in {2, 3}:
        raise ValueError(f"positions file must have shape (frames, particles, 2|3), got {positions.shape}")
    spatial_dim = int(positions.shape[2])
    report = json.loads(report_json.read_text(encoding="utf-8")) if report_json.exists() else {}
    edges, edge_part_ids = load_boundary_geometry(case_dir) if spatial_dim == 2 else (None, None)
    wall_part_summary = load_wall_part_summary(output_dir)
    out_dir = ensure_visualization_dirs(output_dir)["graphs"]
    final_labels = state_labels(final_df)

    time_s = step_df["time_s"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name in STATE_ORDER:
        values = step_state_count_series(step_df, name)
        if not np.any(values):
            continue
        ax.plot(time_s, values, label=name, color=STATE_COLORS[name], linewidth=2.0)
    ax.set_title("Particle State Counts vs Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_state_counts_time_series.png", dpi=170)
    plt.close(fig)

    state_counts = final_state_counts(final_df)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    names = list(STATE_ORDER)
    vals = [state_counts[name] for name in names]
    colors = [STATE_COLORS[name] for name in names]
    axes[0].bar(names, vals, color=colors)
    axes[0].set_title("Final State Counts")
    axes[0].set_ylabel("count")
    for idx, value in enumerate(vals):
        axes[0].text(idx, value, str(value), ha="center", va="bottom", fontsize=9)
    pie_vals = [v for v in vals if v > 0]
    pie_labels = [n for n, v in zip(names, vals) if v > 0]
    pie_colors = [STATE_COLORS[n] for n in pie_labels]
    if pie_vals:
        axes[1].pie(pie_vals, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Final State Share")
    fig.tight_layout()
    fig.savefig(out_dir / "02_final_state_bar_and_pie.png", dpi=170)
    plt.close(fig)

    if spatial_dim == 2:
        fig, ax = plt.subplots(figsize=(8.2, 5.9))
        (x_lim, y_lim) = axis_limits(positions, edges)
        draw_edges(ax, edges, linewidth=0.9, alpha=0.95, color="#444444")
        for name in STATE_ORDER:
            mask = final_labels == name
            if not np.any(mask):
                continue
            sub = final_df.loc[mask]
            ax.scatter(sub["x"], sub["y"], s=5, color=STATE_COLORS[name], alpha=0.7, label=f"{name} ({int(mask.sum())})", zorder=2)
        ax.set_title("Final Particle States over Geometry")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "03_final_state_scatter_geometry.png", dpi=170)
        plt.close(fig)
    else:
        projections = [("xy", "x", "y", (0, 1)), ("xz", "x", "z", (0, 2)), ("yz", "y", "z", (1, 2))]
        fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
        for ax, (tag, xa, ya, proj) in zip(axes, projections):
            x_lim, y_lim = axis_limits(positions, None, projection=proj)
            for name in STATE_ORDER:
                mask = final_labels == name
                if not np.any(mask):
                    continue
                sub = final_df.loc[mask]
                ax.scatter(sub[xa], sub[ya], s=4, color=STATE_COLORS[name], alpha=0.65, label=name if tag == "xy" else "", zorder=2)
            ax.set_title(f"Final States ({tag.upper()} projection)")
            ax.set_xlabel(f"{xa} [m]")
            ax.set_ylabel(f"{ya} [m]")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            ax.grid(alpha=0.2)
        axes[0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "03_final_state_scatter_geometry.png", dpi=170)
        plt.close(fig)

    if spatial_dim == 2:
        all_points = positions.reshape(-1, 2)
        fig, ax = plt.subplots(figsize=(8.2, 5.9))
        hist = ax.hist2d(all_points[:, 0], all_points[:, 1], bins=180, cmap="magma")
        draw_edges(ax, edges, linewidth=0.75, alpha=0.7, color="#444444")
        ax.set_title("Trajectory Density Heatmap")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.02, label="samples")
        fig.tight_layout()
        fig.savefig(out_dir / "04_trajectory_density_heatmap.png", dpi=170)
        plt.close(fig)
    else:
        projections = [("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]
        fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
        for ax, (tag, a, b) in zip(axes, projections):
            pts = positions[:, :, [a, b]].reshape(-1, 2)
            x_lim_proj, y_lim_proj = axis_limits(positions, None, projection=(a, b))
            hist = ax.hist2d(pts[:, 0], pts[:, 1], bins=150, cmap="magma")
            fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(f"Trajectory Density ({tag.upper()})")
            ax.set_xlabel(f"{'xyz'[a]} [m]")
            ax.set_ylabel(f"{'xyz'[b]} [m]")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*x_lim_proj)
            ax.set_ylim(*y_lim_proj)
        fig.tight_layout()
        fig.savefig(out_dir / "04_trajectory_density_heatmap.png", dpi=170)
        plt.close(fig)

    speed = np.sqrt(
        np.square(final_df.get("v_x", 0.0))
        + np.square(final_df.get("v_y", 0.0))
        + np.square(final_df.get("v_z", 0.0))
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for name in STATE_ORDER:
        mask = final_labels == name
        if not np.any(mask):
            continue
        ax.hist(speed[mask], bins=40, alpha=0.55, color=STATE_COLORS[name], label=name)
    ax.set_title("Final Speed Distribution by State")
    ax.set_xlabel("speed [m/s]")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_speed_distribution_by_state.png", dpi=170)
    plt.close(fig)

    rng = np.random.default_rng(20260402)
    n_particles = positions.shape[1]
    pick = np.sort(rng.choice(n_particles, size=min(sample_trajectories, n_particles), replace=False))
    if spatial_dim == 2:
        fig, ax = plt.subplots(figsize=(8.2, 5.9))
        draw_edges(ax, edges, linewidth=0.8, alpha=0.9, color="#444444")
        for idx in pick:
            tr = positions[:, idx, :]
            ax.plot(tr[:, 0], tr[:, 1], linewidth=0.65, alpha=0.35, color="#1f77b4", zorder=2)
        ax.set_title(f"Sampled Trajectories Overlay ({len(pick)} particles)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        fig.tight_layout()
        fig.savefig(out_dir / "06_sampled_trajectories_overlay.png", dpi=170)
        plt.close(fig)
    else:
        projections = [("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)]
        fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
        for ax, (tag, a, b) in zip(axes, projections):
            x_lim_proj, y_lim_proj = axis_limits(positions, None, projection=(a, b))
            for idx in pick:
                tr = positions[:, idx, :]
                ax.plot(tr[:, a], tr[:, b], linewidth=0.6, alpha=0.35, color="#1f77b4", zorder=2)
            ax.set_title(f"Sampled Trajectories ({tag.upper()})")
            ax.set_xlabel(f"{'xyz'[a]} [m]")
            ax.set_ylabel(f"{'xyz'[b]} [m]")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*x_lim_proj)
            ax.set_ylim(*y_lim_proj)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "06_sampled_trajectories_overlay.png", dpi=170)
        plt.close(fig)

    wall_law_counts = report.get("wall_law_counts", {})
    if isinstance(wall_law_counts, dict) and wall_law_counts:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        names = list(wall_law_counts.keys())
        vals = [int(wall_law_counts[name]) for name in names]
        ax.bar(names, vals, color="#4c78a8")
        ax.set_title("Wall Interaction Counts by Law")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(out_dir / "07_wall_law_counts.png", dpi=170)
        plt.close(fig)

    if not wall_part_summary.empty:
        pivot = (
            wall_part_summary.pivot_table(index="part_id", columns="outcome", values="count", aggfunc="sum", fill_value=0)
            .sort_index()
        )
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(9.2, 5.2))
            outcome_order = [name for name in ("stuck", "reflected_specular", "reflected_diffuse", "absorbed") if name in pivot.columns]
            outcome_order.extend(name for name in pivot.columns if name not in outcome_order)
            bottom = np.zeros(pivot.shape[0], dtype=np.float64)
            color_map = {
                "stuck": "#d62728",
                "reflected_specular": "#4c78a8",
                "reflected_diffuse": "#72b7b2",
                "absorbed": "#2ca02c",
            }
            x_idx = np.arange(pivot.shape[0], dtype=np.float64)
            for outcome in outcome_order:
                values = pivot[outcome].to_numpy(dtype=np.float64)
                ax.bar(
                    x_idx,
                    values,
                    bottom=bottom,
                    color=color_map.get(outcome, "#999999"),
                    label=outcome,
                    width=0.78,
                )
                bottom += values
            ax.set_xticks(x_idx, [str(int(v)) for v in pivot.index.to_numpy()])
            ax.set_title("Wall Interactions by Boundary Part / Outcome")
            ax.set_xlabel("boundary part_id")
            ax.set_ylabel("count")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "08_wall_interactions_by_part_outcome.png", dpi=170)
            plt.close(fig)

    stuck_summary = wall_part_summary[wall_part_summary["outcome"] == "stuck"].copy() if not wall_part_summary.empty else pd.DataFrame()
    if not stuck_summary.empty:
        stuck_counts = (
            stuck_summary.groupby("part_id", as_index=False)
            .agg(stuck_count=("count", "sum"))
            .sort_values(["stuck_count", "part_id"], ascending=[False, True])
        )
        stuck_counts.to_csv(out_dir / "09_stuck_counts_by_boundary_part.csv", index=False)

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(stuck_counts["part_id"].astype(str), stuck_counts["stuck_count"], color="#c44e52")
        ax.set_title("Wall Sticking Counts by Boundary Part")
        ax.set_xlabel("boundary part_id")
        ax.set_ylabel("stuck count")
        for idx, value in enumerate(stuck_counts["stuck_count"].tolist()):
            ax.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "09_stuck_counts_by_boundary_part.png", dpi=170)
        plt.close(fig)
    else:
        stuck_mask = final_df.get("stuck", pd.Series(np.zeros(len(final_df), dtype=int))).to_numpy(dtype=bool)
        if edges is not None and edge_part_ids is not None and np.any(stuck_mask):
            stuck_pts = final_df.loc[stuck_mask, ["x", "y"]].to_numpy(dtype=np.float64)
            stuck_part_ids, stuck_dist = _nearest_boundary_part_ids(stuck_pts, edges, edge_part_ids)
            stuck_counts = (
                pd.DataFrame(
                    {
                        "part_id": stuck_part_ids.astype(int),
                        "distance_to_edge_m": stuck_dist.astype(float),
                    }
                )
                .groupby("part_id", as_index=False)
                .agg(stuck_count=("part_id", "size"), mean_distance_to_edge_m=("distance_to_edge_m", "mean"))
                .sort_values(["stuck_count", "part_id"], ascending=[False, True])
            )
            stuck_counts.to_csv(out_dir / "09_stuck_counts_by_boundary_part.csv", index=False)

            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            ax.bar(stuck_counts["part_id"].astype(str), stuck_counts["stuck_count"], color="#c44e52")
            ax.set_title("Final Stuck Positions by Boundary Part")
            ax.set_xlabel("boundary part_id")
            ax.set_ylabel("stuck count")
            for idx, value in enumerate(stuck_counts["stuck_count"].tolist()):
                ax.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "09_stuck_counts_by_boundary_part.png", dpi=170)
            plt.close(fig)

    summary = {
        "plot_dir": str(out_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "case_dir": str(case_dir.resolve()) if case_dir is not None else "",
        "spatial_dim": int(spatial_dim),
        "files": list_files(out_dir, (".png", ".csv", ".json")),
        "save_frame_count": int(len(frame_df)),
        "particle_count": int(len(final_df)),
        "final_state_counts": state_counts,
        "used_wall_part_summary": bool(not wall_part_summary.empty),
    }
    (out_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export summary graphs from a simulation output directory.")
    parser.add_argument("--output-dir", required=True, help="Simulation output directory")
    parser.add_argument("--case-dir", default="", help="Case directory used to overlay geometry boundaries")
    parser.add_argument("--sample-trajectories", type=int, default=300, help="Trajectory sample size for overlay plot")
    args = parser.parse_args()
    from tools.export_visualizations import export_visualizations

    index_path = export_visualizations(
        output_dir=Path(args.output_dir),
        case_dir=Path(args.case_dir) if args.case_dir else None,
        modules=("graphs",),
        sample_trajectories=int(args.sample_trajectories),
    )
    print(f"wrote graphs via unified pipeline: {index_path}")


if __name__ == "__main__":
    main()
