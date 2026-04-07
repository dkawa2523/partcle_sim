from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from tools.state_contract import STATE_ORDER, classify_particle_states, final_state_counts as _final_state_counts

STATE_COLORS = {
    "active": "#1f77b4",
    "invalid_mask_stopped": "#8c564b",
    "stuck": "#d62728",
    "escaped": "#ff7f0e",
    "absorbed": "#2ca02c",
}

_STEP_STATE_COUNT_COLUMNS = {
    "active": "active_count",
    "stuck": "stuck_count",
    "absorbed": "absorbed_count",
    "escaped": "escaped_count",
}


def ensure_visualization_dirs(output_dir: Path, clean: bool = False) -> dict[str, Path]:
    base = Path(output_dir)
    if clean:
        for legacy in ("graphs", "animations", "visuals"):
            legacy_dir = base / legacy
            if legacy_dir.exists() and legacy_dir.is_dir():
                shutil.rmtree(legacy_dir)
    root = base / "visualizations"
    dirs = {
        "root": root,
        "graphs": root / "graphs",
        "animations": root / "animations",
        "mechanics": root / "mechanics",
        "boundary_diagnostics": root / "boundary_diagnostics",
        "reports": root / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_visualization_index(output_dir: Path, payload: Mapping[str, object]) -> Path:
    dirs = ensure_visualization_dirs(output_dir, clean=False)
    index_path = dirs["reports"] / "visualization_index.json"
    index_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return index_path


def resolve_positions_path(output_dir: Path) -> tuple[Path, int]:
    base = Path(output_dir)
    p2 = base / "positions_2d.npy"
    p3 = base / "positions_3d.npy"
    if p2.exists():
        return p2, 2
    if p3.exists():
        return p3, 3
    raise FileNotFoundError(f"positions_2d.npy or positions_3d.npy not found in: {base}")


def load_boundary_geometry(case_dir: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if case_dir is None:
        return None, None
    geom_path = Path(case_dir) / "generated" / "comsol_geometry_2d.npz"
    if not geom_path.exists():
        return None, None
    with np.load(geom_path) as data:
        if "boundary_edges" not in data:
            return None, None
        edges = np.asarray(data["boundary_edges"], dtype=np.float64)
        part_ids = np.asarray(data["boundary_edge_part_ids"], dtype=np.int32) if "boundary_edge_part_ids" in data else None
    if edges.ndim != 3 or edges.shape[1:] != (2, 2):
        return None, None
    if part_ids is not None and part_ids.shape[0] != edges.shape[0]:
        part_ids = None
    return edges, part_ids


def load_boundary_edges(case_dir: Path | None) -> np.ndarray | None:
    edges, _ = load_boundary_geometry(case_dir)
    return edges


def load_wall_events(output_dir: Path) -> pd.DataFrame:
    path = Path(output_dir) / "wall_events.csv"
    if not path.exists():
        return pd.DataFrame(columns=["time_s", "particle_id"])
    df = pd.read_csv(path)
    required = {"time_s", "particle_id"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["time_s", "particle_id"])
    cols = sorted(set(df.columns) & {"time_s", "particle_id", "part_id", "outcome", "wall_mode"})
    return df.loc[:, cols].copy()


def load_wall_part_summary(output_dir: Path) -> pd.DataFrame:
    base = Path(output_dir)
    wall_events_csv = base / "wall_events.csv"
    wall_summary_csv = base / "wall_summary_by_part.csv"
    if wall_events_csv.exists():
        wall_events = pd.read_csv(wall_events_csv)
        if not wall_events.empty and {"part_id", "outcome", "wall_mode"}.issubset(wall_events.columns):
            return (
                wall_events.groupby(["part_id", "outcome", "wall_mode"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values(["part_id", "outcome", "wall_mode"])
            )
    if wall_summary_csv.exists():
        wall_summary = pd.read_csv(wall_summary_csv)
        if not wall_summary.empty and {"part_id", "outcome", "wall_mode", "count"}.issubset(wall_summary.columns):
            return wall_summary.sort_values(["part_id", "outcome", "wall_mode"]).reset_index(drop=True)
    return pd.DataFrame(columns=["part_id", "outcome", "wall_mode", "count"])


def state_labels(final_particles: pd.DataFrame) -> np.ndarray:
    return classify_particle_states(final_particles)


def final_state_counts(final_particles: pd.DataFrame) -> dict[str, int]:
    return _final_state_counts(final_particles)


def step_state_count_series(step_df: pd.DataFrame, state_name: str) -> np.ndarray:
    name = str(state_name)
    if name == "invalid_mask_stopped":
        if "invalid_mask_stopped_count_step" not in step_df:
            return np.zeros(len(step_df), dtype=np.float64)
        return np.cumsum(step_df["invalid_mask_stopped_count_step"].to_numpy(dtype=np.float64))
    col = _STEP_STATE_COUNT_COLUMNS.get(name, "")
    if col not in step_df:
        return np.zeros(len(step_df), dtype=np.float64)
    return step_df[col].to_numpy(dtype=np.float64)


def axis_limits(
    positions: np.ndarray,
    edges: np.ndarray | None = None,
    projection: tuple[int, int] = (0, 1),
) -> tuple[tuple[float, float], tuple[float, float]]:
    a, b = projection
    pts = np.asarray(positions, dtype=np.float64)[:, :, [a, b]].reshape(-1, 2)
    x_min = float(np.nanmin(pts[:, 0]))
    x_max = float(np.nanmax(pts[:, 0]))
    y_min = float(np.nanmin(pts[:, 1]))
    y_max = float(np.nanmax(pts[:, 1]))
    if edges is not None and edges.size and projection == (0, 1):
        x_min = min(x_min, float(np.nanmin(edges[:, :, 0])))
        x_max = max(x_max, float(np.nanmax(edges[:, :, 0])))
        y_min = min(y_min, float(np.nanmin(edges[:, :, 1])))
        y_max = max(y_max, float(np.nanmax(edges[:, :, 1])))
    dx = max(1e-6, x_max - x_min)
    dy = max(1e-6, y_max - y_min)
    return (x_min - 0.05 * dx, x_max + 0.05 * dx), (y_min - 0.05 * dy, y_max + 0.05 * dy)


def draw_edges(
    ax: plt.Axes,
    edges: np.ndarray | None,
    *,
    linewidth: float = 0.9,
    alpha: float = 0.9,
    color: str = "#555555",
) -> None:
    if edges is None:
        return
    segs = np.asarray(edges, dtype=np.float64)
    if segs.ndim != 3 or segs.shape[1:] != (2, 2):
        return
    for seg in segs:
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=1)


def interpolate_frames(positions: np.ndarray, times: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    if factor <= 1 or positions.shape[0] <= 1:
        return positions, times
    n_frames, n_particles, spatial_dim = positions.shape
    new_frames = (n_frames - 1) * factor + 1
    out_pos = np.zeros((new_frames, n_particles, spatial_dim), dtype=np.float64)
    out_t = np.zeros(new_frames, dtype=np.float64)
    w = np.linspace(0.0, 1.0, factor + 1, dtype=np.float64)
    idx = 0
    for i in range(n_frames - 1):
        p0 = positions[i]
        p1 = positions[i + 1]
        t0 = float(times[i])
        t1 = float(times[i + 1])
        for j in range(factor):
            a = float(w[j])
            out_pos[idx] = (1.0 - a) * p0 + a * p1
            out_t[idx] = (1.0 - a) * t0 + a * t1
            idx += 1
    out_pos[idx] = positions[-1]
    out_t[idx] = times[-1]
    return out_pos, out_t


def interpolate_particle_position(
    positions: np.ndarray,
    times: np.ndarray,
    particle_index: int,
    event_time: float,
) -> np.ndarray:
    if positions.shape[0] == 0:
        return np.zeros(positions.shape[2], dtype=np.float64)
    if event_time <= float(times[0]):
        return positions[0, particle_index].astype(np.float64, copy=True)
    if event_time >= float(times[-1]):
        return positions[-1, particle_index].astype(np.float64, copy=True)
    right = int(np.searchsorted(times, event_time, side="right"))
    left = max(0, right - 1)
    right = min(right, len(times) - 1)
    t0 = float(times[left])
    t1 = float(times[right])
    if t1 <= t0 + 1e-30:
        return positions[right, particle_index].astype(np.float64, copy=True)
    a = float(np.clip((event_time - t0) / (t1 - t0), 0.0, 1.0))
    return ((1.0 - a) * positions[left, particle_index] + a * positions[right, particle_index]).astype(np.float64, copy=True)


def prepare_event_overlay(
    wall_events: pd.DataFrame,
    sample_indices: np.ndarray,
    particle_ids: np.ndarray,
    positions: np.ndarray,
    times: np.ndarray,
    interpolate_positions: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if wall_events.empty or sample_indices.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64)
    sample_particle_ids = particle_ids[sample_indices].astype(np.int64, copy=False)
    event_rows = wall_events[wall_events["particle_id"].astype(np.int64).isin(sample_particle_ids)]
    if event_rows.empty:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64)
    idx_by_particle = {int(pid): int(i) for i, pid in enumerate(particle_ids.tolist())}
    xy: list[np.ndarray] = []
    frame_ids: list[int] = []
    for row in event_rows.itertuples(index=False):
        pid = int(getattr(row, "particle_id"))
        evt_time = float(getattr(row, "time_s"))
        p_idx = idx_by_particle.get(pid)
        if p_idx is None:
            continue
        if interpolate_positions:
            pos = interpolate_particle_position(positions, times, p_idx, evt_time)
        else:
            nearest = int(np.argmin(np.abs(times - evt_time)))
            pos = positions[nearest, p_idx].astype(np.float64, copy=True)
        frame_id = int(np.searchsorted(times, evt_time, side="left"))
        frame_id = max(0, min(frame_id, len(times) - 1))
        xy.append(pos)
        frame_ids.append(frame_id)
    if not xy:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.vstack(xy), np.asarray(frame_ids, dtype=np.int64)


def as_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[0]
    raise ValueError(f"Expected 2D or 3D array, got shape={a.shape}")


def require_2d_quantity(payload: np.lib.npyio.NpzFile, name: str, label: str) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"{label} require field quantity: {name}")
    return as_2d(payload[name])


def draw_boundary_edges(
    ax: plt.Axes,
    boundary_edges: np.ndarray | None,
    boundary_part_ids: np.ndarray | None = None,
    *,
    linewidth: float = 1.0,
    alpha: float = 0.9,
) -> None:
    if boundary_edges is None:
        return
    segs = np.asarray(boundary_edges, dtype=np.float64)
    if segs.ndim != 3 or segs.shape[1:] != (2, 2) or segs.shape[0] == 0:
        return
    pids = None
    if boundary_part_ids is not None:
        arr = np.asarray(boundary_part_ids, dtype=np.int32)
        if arr.shape[0] == segs.shape[0]:
            pids = arr
    if pids is None:
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], color="k", linewidth=linewidth, alpha=alpha)
        return
    unique = np.unique(pids)
    cmap = plt.get_cmap("tab20")
    denom = max(1, len(unique) - 1)
    color_map = {int(pid): cmap(i / denom) for i, pid in enumerate(unique)}
    for seg, pid in zip(segs, pids):
        ax.plot(seg[:, 0], seg[:, 1], color=color_map.get(int(pid), "k"), linewidth=linewidth, alpha=alpha)


def sample_grid_points(arr: np.ndarray, x: np.ndarray, y: np.ndarray, points: np.ndarray) -> np.ndarray:
    grid = np.asarray(arr, dtype=np.float64)
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    out = np.zeros(pts.shape[0], dtype=np.float64)

    def locate(axis: np.ndarray, value: float) -> tuple[int, int, float]:
        if value <= axis[0]:
            return 0, 1, 0.0
        if value >= axis[-1]:
            return axis.size - 2, axis.size - 1, 1.0
        j = int(np.searchsorted(axis, value))
        lo = j - 1
        hi = j
        denom = float(axis[hi] - axis[lo])
        a = 0.0 if abs(denom) <= 1e-30 else (value - axis[lo]) / denom
        return lo, hi, a

    for i, p in enumerate(pts):
        ix0, ix1, ax = locate(xs, float(p[0]))
        iy0, iy1, ay = locate(ys, float(p[1]))
        c00 = grid[ix0, iy0]
        c10 = grid[ix1, iy0]
        c01 = grid[ix0, iy1]
        c11 = grid[ix1, iy1]
        c0 = c00 * (1.0 - ax) + c10 * ax
        c1 = c01 * (1.0 - ax) + c11 * ax
        out[i] = c0 * (1.0 - ay) + c1 * ay
    return out


def mesh_polygons(mesh_vertices: np.ndarray, mesh_quads: np.ndarray) -> list[np.ndarray]:
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    quads = np.asarray(mesh_quads, dtype=np.int32)
    return [verts[q] for q in quads]


def add_mesh_fill(ax: plt.Axes, polygons: list[np.ndarray], facecolor: str, alpha: float = 1.0) -> None:
    coll = PolyCollection(polygons, facecolors=facecolor, edgecolors="none", alpha=alpha)
    ax.add_collection(coll)


def add_mesh_scalar(
    fig: plt.Figure,
    ax: plt.Axes,
    polygons: list[np.ndarray],
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
) -> None:
    coll = PolyCollection(polygons, array=np.asarray(values, dtype=np.float64), cmap=cmap, edgecolors="none")
    ax.add_collection(coll)
    ax.set_title(title)
    fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.02)


def list_files(path: Path, suffixes: Iterable[str]) -> list[str]:
    out: list[str] = []
    for p in sorted(Path(path).glob("*")):
        if p.is_file() and any(p.name.endswith(s) for s in suffixes):
            out.append(p.name)
    return out
