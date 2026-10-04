from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from tools._result_graph_common import case_field_payload, case_geometry_payload
from tools.state_contract import classify_particle_states
from tools.visualization_common import (
    STATE_COLORS,
    domain_part_medium_summary,
    draw_boundary_edges,
    draw_domain_part_outlines,
    draw_domain_parts_by_medium,
)
from tools.visualization_data import (
    axis_limits,
    interpolate_frames,
    load_boundary_geometry,
    load_wall_events,
    prepare_event_overlay,
    resolve_positions_path,
)
from tools.visualization_reports import ensure_visualization_dirs


class AnimationExportError(RuntimeError):
    """Recoverable failure while exporting an optional animation."""


class AnimationInputError(ValueError, AnimationExportError):
    """Animation input is present but invalid."""


class AnimationInputNotFoundError(FileNotFoundError, AnimationExportError):
    """A required animation input artifact is missing."""


class AnimationWriteError(AnimationExportError):
    """The GIF or animation report could not be written."""


def _select_frame_indices(n_frames: int, max_frames: int) -> np.ndarray:
    n_frames = int(n_frames)
    max_frames = int(max_frames)
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_frames <= 0 or n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    if max_frames == 1:
        return np.asarray([0], dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, n_frames - 1, max_frames)).astype(np.int64))


def _select_particle_indices(
    n_particles: int, max_particles: int, mode: str
) -> np.ndarray:
    n_particles = int(n_particles)
    max_particles = int(max_particles)
    if n_particles <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_particles <= 0 or n_particles <= max_particles:
        return np.arange(n_particles, dtype=np.int64)
    if mode == "random":
        rng = np.random.default_rng(20260526)
        return np.sort(
            rng.choice(n_particles, size=max_particles, replace=False).astype(np.int64)
        )
    if mode != "uniform":
        raise AnimationInputError(f"unsupported animation downsample mode: {mode}")
    return np.rint(np.linspace(0, n_particles - 1, max_particles)).astype(np.int64)


def _as_2d_mask(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    return None


def _medium_summary(
    geom: dict[str, np.ndarray], field: dict[str, np.ndarray]
) -> pd.DataFrame:
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


def _write_gif(
    fig: Figure,
    update: Callable[[int], Sequence[Artist]],
    *,
    frame_count: int,
    out_path: Path,
    fps: int,
) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        animation = FuncAnimation(
            fig,
            update,
            frames=frame_count,
            interval=1000 / max(1, fps),
            blit=False,
        )
        animation.save(out_path, writer=PillowWriter(fps=max(1, fps)))
    except (OSError, RuntimeError) as exc:
        raise AnimationWriteError(
            f"failed to write animation {out_path}: {exc}"
        ) from exc
    finally:
        plt.close(fig)


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
    scat = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        s=3.0,
        c=colors,
        alpha=0.4,
        linewidths=0,
        zorder=2,
    )
    title = ax.set_title("")
    ax.grid(alpha=0.25)

    def _update(i: int):
        scat.set_offsets(positions[i])
        title.set_text(
            "Particle Trajectories (all particles)  "
            f"t={times[i]:.4f} s  frame={i + 1}/{len(times)}"
        )
        return scat, title

    _write_gif(
        fig,
        _update,
        frame_count=len(times),
        out_path=out_path,
        fps=fps,
    )


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
        event_xy, event_frame_ids = prepare_event_overlay(
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

    lines = [
        ax.plot([], [], color="#1f77b4", linewidth=0.8, alpha=0.45, zorder=2)[0]
        for _ in range(sample_count)
    ]
    markers = ax.scatter(
        p[0, :, 0], p[0, :, 1], s=8.0, c="#111111", alpha=0.85, linewidths=0, zorder=3
    )
    event_markers = ax.scatter(
        [], [], s=24.0, marker="x", c="#ffd34d", alpha=0.95, linewidths=1.2, zorder=4
    )
    title = ax.set_title("")

    def _update(i: int):
        for j, line in enumerate(lines):
            line.set_data(p[: i + 1, j, 0], p[: i + 1, j, 1])
        markers.set_offsets(p[i])
        if overlay_wall_events and event_frame_ids.size:
            shown = event_frame_ids <= i
            event_markers.set_offsets(
                event_xy[shown] if np.any(shown) else np.zeros((0, 2), dtype=np.float64)
            )
        else:
            event_markers.set_offsets(np.zeros((0, 2), dtype=np.float64))
        title.set_text(
            f"Particle Trajectories (sampled trails n={sample_count})  "
            f"t={times[i]:.4f} s  frame={i + 1}/{n_frames}"
        )
        return [*lines, markers, event_markers, title]

    _write_gif(
        fig,
        _update,
        frame_count=n_frames,
        out_path=out_path,
        fps=fps,
    )


@dataclass(frozen=True)
class _AnimationInput:
    spatial_dim: int
    positions: np.ndarray
    times: np.ndarray
    labels: np.ndarray
    particle_ids: np.ndarray


@dataclass(frozen=True)
class _PreparedAnimation:
    positions: np.ndarray
    times: np.ndarray
    labels: np.ndarray
    particle_ids: np.ndarray
    interpolation_factor: int
    input_frame_count: int
    input_particle_count: int


@dataclass(frozen=True)
class _AnimationGeometry:
    edges: np.ndarray | None
    edge_part_ids: np.ndarray | None
    payload: dict[str, np.ndarray]
    medium_summary: pd.DataFrame | None


@dataclass(frozen=True)
class _AnimationOptions:
    fps: int
    sample_count: int
    overlay_wall_events: bool
    interpolate_wall_event_positions: bool
    write_all_particles: bool


def _validate_positions(positions: np.ndarray, spatial_dim: int) -> None:
    if positions.ndim != 3 or positions.shape[2] not in {2, 3}:
        raise AnimationInputError(
            "positions file must be shaped as (frames, particles, 2|3), "
            f"got {positions.shape}"
        )
    if positions.shape[2] != spatial_dim:
        raise AnimationInputError(
            "positions dimensionality mismatch: "
            f"expected {spatial_dim}, got {positions.shape[2]}"
        )
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise AnimationInputError(
            "positions file must contain at least one frame and one particle"
        )
    if np.isinf(positions).any() or not np.all(
        np.any(np.isfinite(positions), axis=(0, 1))
    ):
        raise AnimationInputError(
            "positions file must contain finite values for every coordinate"
        )


def _load_animation_input(output_dir: Path) -> _AnimationInput:
    try:
        positions_path, spatial_dim = resolve_positions_path(output_dir)
    except FileNotFoundError as exc:
        raise AnimationInputNotFoundError(str(exc)) from exc
    except (OSError, EOFError, ValueError) as exc:
        raise AnimationInputError(str(exc)) from exc
    frames_path = output_dir / "trajectory_frames.csv"
    final_particles_path = output_dir / "final_particles.csv"
    if not frames_path.exists():
        raise AnimationInputNotFoundError(
            f"trajectory_frames file not found: {frames_path}"
        )
    if not final_particles_path.exists():
        raise AnimationInputNotFoundError(
            f"final_particles file not found: {final_particles_path}"
        )
    try:
        positions = np.asarray(np.load(positions_path, allow_pickle=False), dtype=float)
        _validate_positions(positions, spatial_dim)
        times = pd.read_csv(frames_path)["time_s"].to_numpy(dtype=float)
        if len(times) != positions.shape[0]:
            raise AnimationInputError(
                "time frame count mismatch: "
                f"trajectory_frames={len(times)} positions={positions.shape[0]}"
            )
        final_particles = pd.read_csv(final_particles_path)
        if len(final_particles) != positions.shape[1]:
            raise AnimationInputError(
                "particle count mismatch: "
                f"final_particles={len(final_particles)} "
                f"positions={positions.shape[1]}"
            )
        particle_ids = (
            final_particles["particle_id"].to_numpy(dtype=np.int64)
            if "particle_id" in final_particles.columns
            else np.arange(len(final_particles), dtype=np.int64)
        )
        labels = classify_particle_states(final_particles)
    except AnimationInputError:
        raise
    except (OSError, EOFError, KeyError, ValueError) as exc:
        raise AnimationInputError(str(exc)) from exc
    return _AnimationInput(
        spatial_dim=spatial_dim,
        positions=positions,
        times=times,
        labels=labels,
        particle_ids=particle_ids,
    )


def _downsample_animation_frames(
    positions: np.ndarray, times: np.ndarray, max_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = _select_frame_indices(int(positions.shape[0]), int(max_frames))
    if indices.size and indices.size != positions.shape[0]:
        return positions[indices], times[indices]
    return positions, times


def _downsample_animation_particles(
    positions: np.ndarray,
    labels: np.ndarray,
    particle_ids: np.ndarray,
    max_particles: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = _select_particle_indices(
        int(positions.shape[1]), int(max_particles), str(mode)
    )
    if indices.size and indices.size != positions.shape[1]:
        return positions[:, indices, :], labels[indices], particle_ids[indices]
    return positions, labels, particle_ids


def _prepare_animation(
    source: _AnimationInput,
    *,
    interpolate_factor: int,
    max_frames: int,
    max_particles: int,
    downsample_mode: str,
) -> _PreparedAnimation:
    factor = max(1, int(interpolate_factor))
    frame_limit = int(max_frames)
    particle_limit = int(max_particles)
    positions, times = interpolate_frames(source.positions, source.times, factor=factor)
    positions, times = _downsample_animation_frames(positions, times, frame_limit)
    positions, labels, particle_ids = _downsample_animation_particles(
        positions,
        source.labels,
        source.particle_ids,
        particle_limit,
        downsample_mode,
    )
    return _PreparedAnimation(
        positions=positions,
        times=times,
        labels=labels,
        particle_ids=particle_ids,
        interpolation_factor=factor,
        input_frame_count=int(source.positions.shape[0]),
        input_particle_count=int(source.positions.shape[1]),
    )


def _load_animation_geometry(
    case_dir: Path | None, spatial_dim: int
) -> _AnimationGeometry:
    if spatial_dim != 2:
        return _AnimationGeometry(None, None, {}, None)
    try:
        edges, edge_part_ids = load_boundary_geometry(case_dir)
        geometry = case_geometry_payload(case_dir)
        field = case_field_payload(case_dir)
        medium_summary = _medium_summary(geometry, field)
    except (OSError, EOFError, KeyError, ValueError) as exc:
        raise AnimationInputError(str(exc)) from exc
    return _AnimationGeometry(
        edges=edges,
        edge_part_ids=edge_part_ids,
        payload=geometry,
        medium_summary=medium_summary,
    )


def _load_animation_wall_events(output_dir: Path, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return pd.DataFrame(columns=["time_s", "particle_id"])
    try:
        return load_wall_events(output_dir)
    except (OSError, EOFError, KeyError, ValueError) as exc:
        raise AnimationInputError(str(exc)) from exc


def _save_projection_animations(
    *,
    anim_dir: Path,
    positions: np.ndarray,
    prepared: _PreparedAnimation,
    geometry: _AnimationGeometry,
    wall_events: pd.DataFrame,
    options: _AnimationOptions,
    suffix: str,
) -> list[Path]:
    saved: list[Path] = []
    points_path = anim_dir / f"trajectories_all_particles{suffix}.gif"
    trails_path = anim_dir / f"trajectories_sampled_trails{suffix}.gif"
    if options.write_all_particles:
        _save_points_animation(
            positions=positions,
            times=prepared.times,
            labels=prepared.labels,
            edges=geometry.edges,
            edge_part_ids=geometry.edge_part_ids,
            geom=geometry.payload,
            medium_summary=geometry.medium_summary,
            out_path=points_path,
            fps=options.fps,
        )
        saved.append(points_path)
    _save_trails_animation(
        positions=positions,
        times=prepared.times,
        edges=geometry.edges,
        edge_part_ids=geometry.edge_part_ids,
        geom=geometry.payload,
        medium_summary=geometry.medium_summary,
        out_path=trails_path,
        fps=options.fps,
        sample_count=options.sample_count,
        particle_ids=prepared.particle_ids,
        wall_events=wall_events,
        overlay_wall_events=options.overlay_wall_events,
        interpolate_wall_event_positions=options.interpolate_wall_event_positions,
    )
    saved.append(trails_path)
    return saved


def _save_animation_files(
    *,
    anim_dir: Path,
    source: _AnimationInput,
    prepared: _PreparedAnimation,
    geometry: _AnimationGeometry,
    wall_events: pd.DataFrame,
    options: _AnimationOptions,
) -> list[Path]:
    if source.spatial_dim == 2:
        return _save_projection_animations(
            anim_dir=anim_dir,
            positions=prepared.positions,
            prepared=prepared,
            geometry=geometry,
            wall_events=wall_events,
            options=options,
            suffix="",
        )
    saved: list[Path] = []
    empty_geometry = _AnimationGeometry(None, None, {}, None)
    for name, (axis_a, axis_b) in (
        ("xy", (0, 1)),
        ("xz", (0, 2)),
        ("yz", (1, 2)),
    ):
        saved.extend(
            _save_projection_animations(
                anim_dir=anim_dir,
                positions=prepared.positions[:, :, [axis_a, axis_b]],
                prepared=prepared,
                geometry=empty_geometry,
                wall_events=wall_events,
                options=options,
                suffix=f"_{name}",
            )
        )
    return saved


def _write_animation_report(
    *,
    output_dir: Path,
    anim_dir: Path,
    source: _AnimationInput,
    prepared: _PreparedAnimation,
    options: _AnimationOptions,
    max_frames: int,
    max_particles: int,
    downsample_mode: str,
    progress: bool,
    saved_paths: list[Path],
) -> None:
    report = {
        "output_dir": str(output_dir.resolve()),
        "animations_dir": str(anim_dir.resolve()),
        "spatial_dim": int(source.spatial_dim),
        "overlay_wall_events": options.overlay_wall_events,
        "interpolate_wall_event_positions": (options.interpolate_wall_event_positions),
        "interpolate_factor": prepared.interpolation_factor,
        "fps": int(options.fps),
        "sample_count": int(options.sample_count),
        "input_frame_count": prepared.input_frame_count,
        "input_particle_count": prepared.input_particle_count,
        "animation_frame_count": int(prepared.positions.shape[0]),
        "animation_particle_count": int(prepared.positions.shape[1]),
        "max_frames": int(max_frames),
        "max_particles": int(max_particles),
        "downsample_mode": str(downsample_mode),
        "write_all_particles": options.write_all_particles,
        "progress_enabled": bool(progress),
        "files": [path.name for path in sorted(saved_paths)],
    }
    report_path = anim_dir / "animation_report.json"
    try:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        raise AnimationWriteError(
            f"failed to write animation report {report_path}: {exc}"
        ) from exc


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
    source = _load_animation_input(output_dir)
    prepared = _prepare_animation(
        source,
        interpolate_factor=interpolate_factor,
        max_frames=max_frames,
        max_particles=max_particles,
        downsample_mode=downsample_mode,
    )
    geometry = _load_animation_geometry(case_dir, source.spatial_dim)
    options = _AnimationOptions(
        fps=fps,
        sample_count=sample_count,
        overlay_wall_events=bool(overlay_wall_events),
        interpolate_wall_event_positions=bool(interpolate_wall_event_positions),
        write_all_particles=bool(write_all_particles),
    )
    wall_events = _load_animation_wall_events(output_dir, options.overlay_wall_events)
    try:
        anim_dir = ensure_visualization_dirs(output_dir)["animations"]
        anim_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AnimationWriteError(
            f"failed to create animation directory for {output_dir}: {exc}"
        ) from exc
    if progress:
        print(
            "[animations] exporting "
            f"{prepared.positions.shape[0]} frames, "
            f"{prepared.positions.shape[1]} particles, "
            f"mode={downsample_mode}, fps={fps}"
        )
    saved_paths = _save_animation_files(
        anim_dir=anim_dir,
        source=source,
        prepared=prepared,
        geometry=geometry,
        wall_events=wall_events,
        options=options,
    )
    _write_animation_report(
        output_dir=output_dir,
        anim_dir=anim_dir,
        source=source,
        prepared=prepared,
        options=options,
        max_frames=max_frames,
        max_particles=max_particles,
        downsample_mode=downsample_mode,
        progress=progress,
        saved_paths=saved_paths,
    )
    return anim_dir
