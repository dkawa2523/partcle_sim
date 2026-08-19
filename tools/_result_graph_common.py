"""Shared validated data structures and calculations for result graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.state_contract import STATE_ORDER

PYPLOT = plt


@dataclass(frozen=True)
class FullGraphData:
    output_dir: Path
    case_dir: Path | None
    final_df: pd.DataFrame
    frame_df: pd.DataFrame
    step_df: pd.DataFrame
    positions: np.ndarray
    spatial_dim: int
    position_columns: list[str]
    axis_names: list[str]
    velocity_columns: list[str]
    report: dict[str, object]
    edges: np.ndarray | None
    edge_part_ids: np.ndarray | None
    geometry_payload: dict[str, np.ndarray]
    field_payload: dict[str, np.ndarray]
    medium_summary: pd.DataFrame
    wall_events: pd.DataFrame
    wall_part_summary: pd.DataFrame
    out_dir: Path
    final_labels: np.ndarray


TRAJECTORY_ARTIFACTS = (
    "trajectory_frames.csv",
    "step_summary.csv",
    "trajectory.npy",
)
PROJECTIONS_3D = (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2))


def nearest_boundary_part_ids(
    points: np.ndarray, edges: np.ndarray, part_ids: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    segs = np.asarray(edges, dtype=np.float64)
    if pts.size == 0 or segs.size == 0:
        return np.zeros(pts.shape[0], dtype=np.int32), np.full(
            pts.shape[0], np.nan, dtype=np.float64
        )
    p0 = segs[:, 0, :]
    p1 = segs[:, 1, :]
    ab = p1 - p0
    ab2 = np.sum(ab * ab, axis=1)
    out_ids = np.zeros(pts.shape[0], dtype=np.int32)
    out_dist = np.full(pts.shape[0], np.inf, dtype=np.float64)
    edge_part_ids = (
        np.asarray(part_ids, dtype=np.int32)
        if part_ids is not None
        else np.zeros(segs.shape[0], dtype=np.int32)
    )
    for index, point in enumerate(pts):
        ap = point[None, :] - p0
        t = np.zeros(segs.shape[0], dtype=np.float64)
        mask = np.isfinite(ab2) & (ab2 > 0.0)
        t[mask] = np.clip(np.sum(ap[mask] * ab[mask], axis=1) / ab2[mask], 0.0, 1.0)
        projection = p0 + t[:, None] * ab
        distances = np.linalg.norm(projection - point[None, :], axis=1)
        edge_index = int(np.argmin(distances))
        out_ids[index] = (
            int(edge_part_ids[edge_index]) if edge_index < edge_part_ids.size else 0
        )
        out_dist[index] = float(distances[edge_index])
    return out_ids, out_dist


def representative_particle_sample(
    labels: np.ndarray, sample_size: int, seed: int = 20260402
) -> np.ndarray:
    n_particles = int(labels.shape[0])
    if n_particles == 0:
        return np.zeros(0, dtype=np.int64)
    sample_size = max(1, min(int(sample_size), n_particles))
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for name in STATE_ORDER:
        state_indices = np.flatnonzero(labels == name)
        if state_indices.size == 0:
            continue
        take = min(state_indices.size, max(1, sample_size // max(1, len(STATE_ORDER))))
        picks.extend(
            int(value) for value in rng.choice(state_indices, size=take, replace=False)
        )
    if len(picks) < sample_size:
        remaining = np.setdiff1d(
            np.arange(n_particles, dtype=np.int64),
            np.asarray(picks, dtype=np.int64),
            assume_unique=False,
        )
        if remaining.size:
            picks.extend(
                int(value)
                for value in rng.choice(
                    remaining,
                    size=min(sample_size - len(picks), remaining.size),
                    replace=False,
                )
            )
    return np.sort(np.asarray(picks[:sample_size], dtype=np.int64))


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def case_geometry_payload(case_dir: Path | None) -> dict[str, np.ndarray]:
    if case_dir is None:
        return {}
    return load_npz_arrays(Path(case_dir) / "generated" / "comsol_geometry_2d.npz")


def case_field_payload(case_dir: Path | None) -> dict[str, np.ndarray]:
    if case_dir is None:
        return {}
    return load_npz_arrays(Path(case_dir) / "generated" / "comsol_field_2d.npz")


def final_position_columns(final_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if {"x_m", "y_m", "z_m"}.issubset(final_df.columns):
        return ["x_m", "y_m", "z_m"], ["x", "y", "z"]
    if {"x_m", "y_m"}.issubset(final_df.columns):
        return ["x_m", "y_m"], ["x", "y"]
    if {"r_m", "z_m"}.issubset(final_df.columns):
        return ["r_m", "z_m"], ["r", "z"]
    return [], []


def final_result_columns(
    final_df: pd.DataFrame, spatial_dim: int
) -> tuple[list[str], list[str], list[str]]:
    position_columns, axis_names = final_position_columns(final_df)
    if len(position_columns) != spatial_dim:
        raise ValueError(
            "final_particles.csv position columns do not match trajectory.npy: "
            f"columns={position_columns}, spatial_dim={spatial_dim}"
        )
    velocity_columns = [f"v{axis_name}_mps" for axis_name in axis_names]
    missing_velocity_columns = [
        name for name in velocity_columns if name not in final_df.columns
    ]
    if missing_velocity_columns:
        raise ValueError(
            "final_particles.csv is missing velocity columns: "
            f"{missing_velocity_columns}"
        )
    return position_columns, axis_names, velocity_columns


def limits_from_points(
    points: np.ndarray, edges: np.ndarray | None = None
) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.asarray(points, dtype=np.float64)
    chunks = []
    if pts.ndim == 2 and pts.shape[1] >= 2 and pts.size:
        chunks.append(pts[:, :2])
    if edges is not None:
        edge_points = np.asarray(edges, dtype=np.float64).reshape(-1, 2)
        if edge_points.size:
            chunks.append(edge_points)
    if not chunks:
        return (-1.0, 1.0), (-1.0, 1.0)
    merged = np.vstack(chunks)
    finite = merged[np.all(np.isfinite(merged), axis=1)]
    if finite.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    lo = np.nanmin(finite, axis=0)
    hi = np.nanmax(finite, axis=0)
    span = np.maximum(hi - lo, 1.0e-12)
    pad = np.maximum(span * 0.04, 1.0e-12)
    return (float(lo[0] - pad[0]), float(hi[0] + pad[0])), (
        float(lo[1] - pad[1]),
        float(hi[1] + pad[1]),
    )


def cumulative_event_count(
    times: np.ndarray, wall_events: pd.DataFrame, outcomes: set[str]
) -> np.ndarray:
    if (
        wall_events.empty
        or "time_s" not in wall_events.columns
        or "outcome" not in wall_events.columns
    ):
        return np.zeros_like(times, dtype=np.float64)
    selected = (
        wall_events.loc[wall_events["outcome"].astype(str).isin(outcomes), "time_s"]
        .to_numpy(dtype=np.float64)
        .copy()
    )
    if selected.size == 0:
        return np.zeros_like(times, dtype=np.float64)
    selected.sort()
    return np.searchsorted(selected, times, side="right").astype(np.float64)


def medium_status_counts(medium_summary: pd.DataFrame) -> dict[str, int]:
    if medium_summary.empty or "medium_status" not in medium_summary.columns:
        return {}
    counts = medium_summary["medium_status"].value_counts().astype(int).to_dict()
    return {str(name): int(count) for name, count in counts.items()}
