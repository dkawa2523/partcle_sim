"""Load and normalize simulation artifacts used by visualizations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_STEP_STATE_COUNT_COLUMNS = {
    "active_total": "active_count",
    "numerical_boundary_stopped": "numerical_boundary_stopped_count",
    "stuck": "stuck_count",
    "absorbed": "absorbed_count",
    "escaped": "escaped_count",
}


def list_files(path: Path, suffixes: Iterable[str]) -> list[str]:
    return [
        item.name
        for item in sorted(Path(path).glob("*"))
        if item.is_file() and any(item.name.endswith(suffix) for suffix in suffixes)
    ]


def resolve_positions_path(output_dir: Path) -> tuple[Path, int]:
    base = Path(output_dir)
    path = base / "trajectory.npy"
    if not path.exists():
        raise FileNotFoundError(f"trajectory.npy not found in debug output: {base}")
    shape = np.load(path, mmap_mode="r").shape
    if len(shape) != 3 or shape[2] not in (2, 3):
        raise ValueError(
            f"trajectory.npy must have shape (frame, particle, 2|3), got {shape}"
        )
    return path, int(shape[2])


def load_boundary_geometry(
    case_dir: Path | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if case_dir is None:
        return None, None
    geom_path = Path(case_dir) / "generated" / "comsol_geometry_2d.npz"
    if not geom_path.exists():
        return None, None
    with np.load(geom_path) as data:
        if "boundary_edges" not in data:
            return None, None
        edges = np.asarray(data["boundary_edges"], dtype=np.float64)
        part_ids = (
            np.asarray(data["boundary_edge_part_ids"], dtype=np.int32)
            if "boundary_edge_part_ids" in data
            else None
        )
    if edges.ndim != 3 or edges.shape[1:] != (2, 2):
        return None, None
    if part_ids is not None and part_ids.shape[0] != edges.shape[0]:
        part_ids = None
    return filter_display_boundary_geometry(edges, part_ids)


def filter_display_boundary_geometry(
    boundary_edges: np.ndarray | None,
    boundary_part_ids: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if boundary_edges is None:
        return None, None
    edges = np.asarray(boundary_edges, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2) or edges.shape[0] == 0:
        return None, None
    if boundary_part_ids is None:
        return edges, None
    part_ids = np.asarray(boundary_part_ids, dtype=np.int32)
    if part_ids.shape[0] != edges.shape[0]:
        return edges, None
    return edges, part_ids


def load_wall_events(output_dir: Path) -> pd.DataFrame:
    path = Path(output_dir) / "wall_events.csv"
    if not path.exists():
        return pd.DataFrame(columns=["time_s", "particle_id"])
    df = pd.read_csv(path)
    required = {"time_s", "particle_id"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["time_s", "particle_id"])
    cols = sorted(
        set(df.columns) & {"time_s", "particle_id", "part_id", "outcome", "wall_mode"}
    )
    return df.loc[:, cols].copy()


def load_wall_part_summary(output_dir: Path) -> pd.DataFrame:
    columns = ["part_id", "outcome", "wall_mode", "count"]
    path = Path(output_dir) / "wall_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"wall_summary.csv not found: {path}")
    summary = pd.read_csv(path)
    missing = sorted(set(columns).difference(summary.columns))
    if missing:
        raise ValueError(f"wall_summary.csv missing required columns: {missing}")
    return summary.sort_values(["part_id", "outcome", "wall_mode"]).reset_index(
        drop=True
    )


def step_state_count_series(step_df: pd.DataFrame, state_name: str) -> np.ndarray:
    name = str(state_name)
    if name == "invalid_mask_stopped":
        if "invalid_mask_stopped_count_step" not in step_df:
            return np.zeros(len(step_df), dtype=np.float64)
        return np.cumsum(
            step_df["invalid_mask_stopped_count_step"].to_numpy(dtype=np.float64)
        )
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
    return (x_min - 0.05 * dx, x_max + 0.05 * dx), (
        y_min - 0.05 * dy,
        y_max + 0.05 * dy,
    )


def interpolate_frames(
    positions: np.ndarray, times: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray]:
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
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return positions[right, particle_index].astype(np.float64, copy=True)
    a = float(np.clip((event_time - t0) / (t1 - t0), 0.0, 1.0))
    return (
        (1.0 - a) * positions[left, particle_index]
        + a * positions[right, particle_index]
    ).astype(np.float64, copy=True)


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
    event_rows = wall_events[
        wall_events["particle_id"].astype(np.int64).isin(sample_particle_ids)
    ]
    if event_rows.empty:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.int64)
    idx_by_particle = {int(pid): int(i) for i, pid in enumerate(particle_ids.tolist())}
    xy: list[np.ndarray] = []
    frame_ids: list[int] = []
    for row in event_rows.itertuples(index=False):
        pid = int(cast(Any, row.particle_id))
        evt_time = float(cast(Any, row.time_s))
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
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[0]
    raise ValueError(f"Expected 2D or 3D array, got shape={data.shape}")


def require_2d_quantity(
    payload: np.lib.npyio.NpzFile, name: str, label: str
) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"{label} require field quantity: {name}")
    return as_2d(payload[name])
