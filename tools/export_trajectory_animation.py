from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from tools.visualization_common import (
    STATE_COLORS,
    axis_limits,
    draw_edges,
    ensure_visualization_dirs,
    interpolate_frames,
    load_boundary_edges,
    load_wall_events,
    prepare_event_overlay,
    resolve_positions_path,
    state_labels,
)

_interpolate_frames = interpolate_frames
_prepare_event_overlay = prepare_event_overlay


def _save_points_animation(
    positions: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray | None,
    out_path: Path,
    fps: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    (x_lim, y_lim) = axis_limits(positions, edges)
    draw_edges(ax, edges, linewidth=0.8, alpha=0.9, color="#666666")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    colors = np.full(labels.shape[0], STATE_COLORS["active"], dtype=object)
    for name, color in STATE_COLORS.items():
        colors[labels == name] = color
    scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], s=3.0, c=colors, alpha=0.4, linewidths=0, zorder=2)
    title = ax.set_title("")
    ax.grid(alpha=0.25)

    def _update(i: int):
        scat.set_offsets(positions[i])
        title.set_text(f"Particle Trajectories (all particles)  t={times[i]:.4f} s  frame={i+1}/{len(times)}")
        return scat, title

    anim = FuncAnimation(fig, _update, frames=len(times), interval=1000 / max(1, fps), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def _save_trails_animation(
    positions: np.ndarray,
    times: np.ndarray,
    edges: np.ndarray | None,
    out_path: Path,
    fps: int,
    sample_count: int,
    particle_ids: np.ndarray,
    wall_events: pd.DataFrame,
    overlay_wall_events: bool,
    interpolate_wall_event_positions: bool,
) -> None:
    n_frames, n_particles, _ = positions.shape
    sample_count = max(1, min(sample_count, n_particles))
    rng = np.random.default_rng(7)
    sample_ids = np.sort(rng.choice(n_particles, size=sample_count, replace=False))
    p = positions[:, sample_ids, :]

    event_xy = np.zeros((0, 2), dtype=np.float64)
    event_frame_ids = np.zeros(0, dtype=np.int64)
    if overlay_wall_events:
        event_xy, event_frame_ids = _prepare_event_overlay(
            wall_events=wall_events,
            sample_indices=sample_ids,
            particle_ids=particle_ids,
            positions=positions,
            times=times,
            interpolate_positions=interpolate_wall_event_positions,
        )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    (x_lim, y_lim) = axis_limits(positions, edges)
    draw_edges(ax, edges, linewidth=0.8, alpha=0.9, color="#666666")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)

    lines = [ax.plot([], [], color="#1f77b4", linewidth=0.8, alpha=0.45, zorder=2)[0] for _ in range(sample_count)]
    markers = ax.scatter(p[0, :, 0], p[0, :, 1], s=8.0, c="#111111", alpha=0.85, linewidths=0, zorder=3)
    event_markers = ax.scatter([], [], s=24.0, marker="x", c="#ffd34d", alpha=0.95, linewidths=1.2, zorder=4)
    title = ax.set_title("")

    def _update(i: int):
        for j, line in enumerate(lines):
            line.set_data(p[: i + 1, j, 0], p[: i + 1, j, 1])
        markers.set_offsets(p[i])
        if overlay_wall_events and event_frame_ids.size:
            shown = event_frame_ids <= i
            event_markers.set_offsets(event_xy[shown] if np.any(shown) else np.zeros((0, 2), dtype=np.float64))
        else:
            event_markers.set_offsets(np.zeros((0, 2), dtype=np.float64))
        title.set_text(
            f"Particle Trajectories (sampled trails n={sample_count})  t={times[i]:.4f} s  frame={i+1}/{n_frames}"
        )
        return [*lines, markers, event_markers, title]

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 / max(1, fps), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)


def export_trajectory_animations(
    output_dir: Path,
    *,
    case_dir: Path | None = None,
    fps: int = 6,
    sample_count: int = 450,
    interpolate_factor: int = 1,
    overlay_wall_events: bool = False,
    interpolate_wall_event_positions: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    positions_path, spatial_dim = resolve_positions_path(output_dir)
    frames_path = output_dir / "save_frames.csv"
    final_particles_path = output_dir / "final_particles.csv"
    if not frames_path.exists():
        raise FileNotFoundError(f"save_frames file not found: {frames_path}")
    if not final_particles_path.exists():
        raise FileNotFoundError(f"final_particles file not found: {final_particles_path}")

    positions = np.asarray(np.load(positions_path), dtype=float)
    if positions.ndim != 3 or positions.shape[2] not in {2, 3}:
        raise ValueError(f"positions file must be shaped as (frames, particles, 2|3), got {positions.shape}")
    if positions.shape[2] != spatial_dim:
        raise ValueError(f"positions dimensionality mismatch: expected {spatial_dim}, got {positions.shape[2]}")

    frame_df = pd.read_csv(frames_path)
    times = frame_df["time_s"].to_numpy(dtype=float)
    if len(times) != positions.shape[0]:
        raise ValueError(f"time frame count mismatch: save_frames={len(times)} positions={positions.shape[0]}")

    final_particles = pd.read_csv(final_particles_path)
    if len(final_particles) != positions.shape[1]:
        raise ValueError(f"particle count mismatch: final_particles={len(final_particles)} positions={positions.shape[1]}")
    labels = state_labels(final_particles)
    particle_ids = (
        final_particles["particle_id"].to_numpy(dtype=np.int64)
        if "particle_id" in final_particles.columns
        else np.arange(len(final_particles), dtype=np.int64)
    )

    factor = max(1, int(interpolate_factor))
    positions_anim, times_anim = _interpolate_frames(positions, times, factor=factor)

    edges = load_boundary_edges(case_dir) if spatial_dim == 2 else None
    wall_events = load_wall_events(output_dir) if bool(overlay_wall_events) else pd.DataFrame(columns=["time_s", "particle_id"])
    anim_dir = ensure_visualization_dirs(output_dir)["animations"]

    saved_paths: list[Path] = []
    if spatial_dim == 2:
        points_path = anim_dir / "trajectories_all_particles.gif"
        trails_path = anim_dir / "trajectories_sampled_trails.gif"
        _save_points_animation(positions=positions_anim, times=times_anim, labels=labels, edges=edges, out_path=points_path, fps=fps)
        _save_trails_animation(
            positions=positions_anim,
            times=times_anim,
            edges=edges,
            out_path=trails_path,
            fps=fps,
            sample_count=sample_count,
            particle_ids=particle_ids,
            wall_events=wall_events,
            overlay_wall_events=bool(overlay_wall_events),
            interpolate_wall_event_positions=bool(interpolate_wall_event_positions),
        )
        saved_paths.extend([points_path, trails_path])
    else:
        projections = [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]
        for name, (a, b) in projections:
            pos_proj = positions_anim[:, :, [a, b]]
            points_path = anim_dir / f"trajectories_all_particles_{name}.gif"
            trails_path = anim_dir / f"trajectories_sampled_trails_{name}.gif"
            _save_points_animation(positions=pos_proj, times=times_anim, labels=labels, edges=None, out_path=points_path, fps=fps)
            _save_trails_animation(
                positions=pos_proj,
                times=times_anim,
                edges=None,
                out_path=trails_path,
                fps=fps,
                sample_count=sample_count,
                particle_ids=particle_ids,
                wall_events=wall_events,
                overlay_wall_events=bool(overlay_wall_events),
                interpolate_wall_event_positions=bool(interpolate_wall_event_positions),
            )
            saved_paths.extend([points_path, trails_path])

    report = {
        "output_dir": str(output_dir.resolve()),
        "animations_dir": str(anim_dir.resolve()),
        "spatial_dim": int(spatial_dim),
        "overlay_wall_events": bool(overlay_wall_events),
        "interpolate_wall_event_positions": bool(interpolate_wall_event_positions),
        "interpolate_factor": int(factor),
        "fps": int(fps),
        "sample_count": int(sample_count),
        "files": [p.name for p in sorted(saved_paths)],
    }
    (anim_dir / "animation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return anim_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export particle trajectory GIF animations.")
    parser.add_argument("--output-dir", required=True, help="Simulation output directory containing positions_2d.npy or positions_3d.npy and save_frames.csv")
    parser.add_argument("--case-dir", default="", help="Case directory (optional), used to overlay COMSOL boundary edges")
    parser.add_argument("--fps", type=int, default=6, help="GIF frame rate")
    parser.add_argument("--sample-count", type=int, default=450, help="Sample size for trail animation")
    parser.add_argument(
        "--interpolate-factor",
        type=int,
        default=1,
        help="Linear interpolation factor between saved frames (1 = no interpolation)",
    )
    parser.add_argument(
        "--overlay-wall-events",
        action="store_true",
        help="Overlay sampled wall-event points on sampled-trails animation",
    )
    parser.add_argument(
        "--interpolate-wall-event-positions",
        action="store_true",
        help="Linearly interpolate overlay event positions by event time",
    )
    args = parser.parse_args()

    from tools.export_visualizations import export_visualizations

    index_path = export_visualizations(
        output_dir=Path(args.output_dir),
        case_dir=Path(args.case_dir) if args.case_dir else None,
        modules=("animations",),
        animation_fps=int(args.fps),
        animation_sample_count=int(args.sample_count),
        animation_interpolate_factor=int(args.interpolate_factor),
        overlay_wall_events=bool(args.overlay_wall_events),
        interpolate_wall_event_positions=bool(args.interpolate_wall_event_positions),
    )
    print(f"wrote animations via unified pipeline: {index_path}")


if __name__ == "__main__":
    main()
