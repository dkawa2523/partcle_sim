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
    domain_part_medium_summary,
    draw_boundary_edges,
    draw_domain_part_outlines,
    draw_domain_parts_by_medium,
    ensure_visualization_dirs,
    interpolate_frames,
    load_boundary_geometry,
    load_wall_events,
    prepare_event_overlay,
    resolve_positions_path,
    state_labels,
)

_interpolate_frames = interpolate_frames
_prepare_event_overlay = prepare_event_overlay


def _select_frame_indices(n_frames: int, max_frames: int) -> np.ndarray:
    n_frames = int(n_frames)
    max_frames = int(max_frames)
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_frames <= 0 or n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    if max_frames == 1:
        return np.asarray([0], dtype=np.int64)
    raw = np.linspace(0, n_frames - 1, max_frames)
    indices = np.unique(np.concatenate([np.rint(raw).astype(np.int64), np.asarray([0, n_frames - 1], dtype=np.int64)]))
    if indices.size <= max_frames:
        return indices
    middle = indices[(indices != 0) & (indices != n_frames - 1)]
    keep_middle = max(0, max_frames - 2)
    if keep_middle:
        middle_pick = np.unique(np.rint(np.linspace(0, middle.size - 1, keep_middle)).astype(np.int64))
        middle = middle[middle_pick[:keep_middle]]
    else:
        middle = np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate([np.asarray([0], dtype=np.int64), middle, np.asarray([n_frames - 1], dtype=np.int64)]))


def _select_particle_indices(n_particles: int, max_particles: int, mode: str) -> np.ndarray:
    n_particles = int(n_particles)
    max_particles = int(max_particles)
    if n_particles <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_particles <= 0 or n_particles <= max_particles:
        return np.arange(n_particles, dtype=np.int64)
    if mode == "random":
        rng = np.random.default_rng(20260526)
        return np.sort(rng.choice(n_particles, size=max_particles, replace=False).astype(np.int64))
    if mode != "uniform":
        raise ValueError(f"unsupported animation downsample mode: {mode}")
    raw = np.linspace(0, n_particles - 1, max_particles)
    indices = np.unique(np.rint(raw).astype(np.int64))
    if indices.size < max_particles:
        remaining = np.setdiff1d(np.arange(n_particles, dtype=np.int64), indices, assume_unique=False)
        indices = np.sort(np.concatenate([indices, remaining[: max_particles - indices.size]]))
    return indices[:max_particles]


def _load_geometry_payload(case_dir: Path | None) -> dict[str, np.ndarray]:
    if case_dir is None:
        return {}
    geom_path = Path(case_dir) / "generated" / "comsol_geometry_2d.npz"
    if not geom_path.exists():
        return {}
    with np.load(geom_path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_field_payload(case_dir: Path | None) -> dict[str, np.ndarray]:
    if case_dir is None:
        return {}
    field_path = Path(case_dir) / "generated" / "comsol_field_2d.npz"
    if not field_path.exists():
        return {}
    with np.load(field_path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _as_2d_mask(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    return None


def _medium_summary(geom: dict[str, np.ndarray], field: dict[str, np.ndarray]) -> pd.DataFrame:
    if not geom:
        return pd.DataFrame()
    return domain_part_medium_summary(
        geom.get("mesh_vertices"),
        geom.get("mesh_triangles"),
        geom.get("mesh_triangle_part_ids"),
        geom.get("mesh_quads"),
        geom.get("mesh_quad_part_ids"),
        field.get("axis_0", geom.get("axis_0")),
        field.get("axis_1", geom.get("axis_1")),
        _as_2d_mask(field.get("valid_mask", geom.get("valid_mask"))),
    )


def _draw_geometry_context(
    ax: plt.Axes,
    geom: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None,
) -> None:
    if medium_summary is not None and not medium_summary.empty:
        draw_domain_parts_by_medium(
            ax,
            geom.get("mesh_vertices"),
            geom.get("mesh_triangles"),
            geom.get("mesh_triangle_part_ids"),
            geom.get("mesh_quads"),
            geom.get("mesh_quad_part_ids"),
            medium_summary=medium_summary,
            alpha=0.25,
            linewidth=0.04,
            label_part_ids=False,
        )
    else:
        draw_domain_part_outlines(
            ax,
            geom.get("mesh_vertices"),
            geom.get("mesh_triangles"),
            geom.get("mesh_triangle_part_ids"),
            geom.get("mesh_quads"),
            geom.get("mesh_quad_part_ids"),
            linewidth=0.65,
        )
    draw_boundary_edges(ax, edges, edge_part_ids, linewidth=0.8, alpha=0.95)


def _save_points_animation(
    positions: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    geom: dict[str, np.ndarray],
    medium_summary: pd.DataFrame | None,
    out_path: Path,
    fps: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    (x_lim, y_lim) = axis_limits(positions, edges)
    _draw_geometry_context(ax, geom, edges, edge_part_ids, medium_summary)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    colors = np.full(labels.shape[0], STATE_COLORS["active_free_flight"], dtype=object)
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
    edge_part_ids: np.ndarray | None,
    geom: dict[str, np.ndarray],
    medium_summary: pd.DataFrame | None,
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
    _draw_geometry_context(ax, geom, edges, edge_part_ids, medium_summary)
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
    max_frames: int = 240,
    max_particles: int = 2000,
    downsample_mode: str = "uniform",
    write_all_particles: bool = True,
    progress: bool = False,
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
    input_frame_count = int(positions.shape[0])
    input_particle_count = int(positions.shape[1])
    frame_indices = _select_frame_indices(int(positions_anim.shape[0]), int(max_frames))
    if frame_indices.size and frame_indices.size != positions_anim.shape[0]:
        positions_anim = positions_anim[frame_indices]
        times_anim = times_anim[frame_indices]
    particle_indices = _select_particle_indices(int(positions_anim.shape[1]), int(max_particles), str(downsample_mode))
    if particle_indices.size and particle_indices.size != positions_anim.shape[1]:
        positions_anim = positions_anim[:, particle_indices, :]
        labels = labels[particle_indices]
        particle_ids = particle_ids[particle_indices]

    edges, edge_part_ids = load_boundary_geometry(case_dir) if spatial_dim == 2 else (None, None)
    geom = _load_geometry_payload(case_dir) if spatial_dim == 2 else {}
    field = _load_field_payload(case_dir) if spatial_dim == 2 else {}
    medium_summary = _medium_summary(geom, field) if spatial_dim == 2 else pd.DataFrame()
    wall_events = load_wall_events(output_dir) if bool(overlay_wall_events) else pd.DataFrame(columns=["time_s", "particle_id"])
    anim_dir = ensure_visualization_dirs(output_dir)["animations"]
    anim_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    if progress:
        print(
            "[animations] exporting "
            f"{positions_anim.shape[0]} frames, {positions_anim.shape[1]} particles, "
            f"mode={downsample_mode}, fps={fps}"
        )
    if spatial_dim == 2:
        points_path = anim_dir / "trajectories_all_particles.gif"
        trails_path = anim_dir / "trajectories_sampled_trails.gif"
        if bool(write_all_particles):
            _save_points_animation(
                positions=positions_anim,
                times=times_anim,
                labels=labels,
                edges=edges,
                edge_part_ids=edge_part_ids,
                geom=geom,
                medium_summary=medium_summary,
                out_path=points_path,
                fps=fps,
            )
            saved_paths.append(points_path)
        _save_trails_animation(
            positions=positions_anim,
            times=times_anim,
            edges=edges,
            edge_part_ids=edge_part_ids,
            geom=geom,
            medium_summary=medium_summary,
            out_path=trails_path,
            fps=fps,
            sample_count=sample_count,
            particle_ids=particle_ids,
            wall_events=wall_events,
            overlay_wall_events=bool(overlay_wall_events),
            interpolate_wall_event_positions=bool(interpolate_wall_event_positions),
        )
        saved_paths.append(trails_path)
    else:
        projections = [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]
        for name, (a, b) in projections:
            pos_proj = positions_anim[:, :, [a, b]]
            points_path = anim_dir / f"trajectories_all_particles_{name}.gif"
            trails_path = anim_dir / f"trajectories_sampled_trails_{name}.gif"
            if bool(write_all_particles):
                _save_points_animation(
                    positions=pos_proj,
                    times=times_anim,
                    labels=labels,
                    edges=None,
                    edge_part_ids=None,
                    geom={},
                    medium_summary=None,
                    out_path=points_path,
                    fps=fps,
                )
                saved_paths.append(points_path)
            _save_trails_animation(
                positions=pos_proj,
                times=times_anim,
                edges=None,
                edge_part_ids=None,
                geom={},
                medium_summary=None,
                out_path=trails_path,
                fps=fps,
                sample_count=sample_count,
                particle_ids=particle_ids,
                wall_events=wall_events,
                overlay_wall_events=bool(overlay_wall_events),
                interpolate_wall_event_positions=bool(interpolate_wall_event_positions),
            )
            saved_paths.append(trails_path)

    report = {
        "output_dir": str(output_dir.resolve()),
        "animations_dir": str(anim_dir.resolve()),
        "spatial_dim": int(spatial_dim),
        "overlay_wall_events": bool(overlay_wall_events),
        "interpolate_wall_event_positions": bool(interpolate_wall_event_positions),
        "interpolate_factor": int(factor),
        "fps": int(fps),
        "sample_count": int(sample_count),
        "input_frame_count": int(input_frame_count),
        "input_particle_count": int(input_particle_count),
        "animation_frame_count": int(positions_anim.shape[0]),
        "animation_particle_count": int(positions_anim.shape[1]),
        "max_frames": int(max_frames),
        "max_particles": int(max_particles),
        "downsample_mode": str(downsample_mode),
        "write_all_particles": bool(write_all_particles),
        "progress_enabled": bool(progress),
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
    parser.add_argument("--max-frames", type=int, default=240, help="Maximum GIF frames after interpolation (0 = no limit)")
    parser.add_argument("--max-particles", type=int, default=2000, help="Maximum particles drawn in GIFs (0 = no limit)")
    parser.add_argument(
        "--downsample-mode",
        choices=("uniform", "random"),
        default="uniform",
        help="Particle downsample mode used when max-particles is exceeded",
    )
    parser.add_argument(
        "--skip-all-particles-animation",
        action="store_true",
        help="Write only sampled-trails GIFs, skipping all-particle GIFs",
    )
    parser.add_argument("--progress", action="store_true", help="Print compact animation export progress")
    args = parser.parse_args()

    from tools.export_visualizations import export_visualizations

    index_path = export_visualizations(
        output_dir=Path(args.output_dir),
        case_dir=Path(args.case_dir) if args.case_dir else None,
        modules=("animations",),
        animation_fps=int(args.fps),
        animation_sample_count=int(args.sample_count),
        animation_interpolate_factor=int(args.interpolate_factor),
        animation_max_frames=int(args.max_frames),
        animation_max_particles=int(args.max_particles),
        animation_downsample_mode=str(args.downsample_mode),
        animation_write_all_particles=not bool(args.skip_all_particles_animation),
        animation_progress=bool(args.progress),
        overlay_wall_events=bool(args.overlay_wall_events),
        interpolate_wall_event_positions=bool(args.interpolate_wall_event_positions),
    )
    print(f"wrote animations via unified pipeline: {index_path}")


if __name__ == "__main__":
    main()
