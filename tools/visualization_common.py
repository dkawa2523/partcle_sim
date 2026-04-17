from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch

from tools.state_contract import (
    STATE_ORDER,
    classify_particle_states,
    final_state_counts as _final_state_counts,
)

STATE_COLORS = {
    "active_total": "#1f77b4",
    "active_free_flight": "#1f77b4",
    "contact_sliding": "#17becf",
    "contact_endpoint_stopped": "#bcbd22",
    "invalid_mask_stopped": "#8c564b",
    "numerical_boundary_stopped": "#9467bd",
    "stuck": "#d62728",
    "escaped": "#ff7f0e",
    "absorbed": "#2ca02c",
    "inactive": "#7f7f7f",
}

_STEP_STATE_COUNT_COLUMNS = {
    "active_total": "active_count",
    "numerical_boundary_stopped": "numerical_boundary_stopped_count",
    "stuck": "stuck_count",
    "absorbed": "absorbed_count",
    "escaped": "escaped_count",
}

STEP_STATE_ORDER = (
    "active_total",
    "invalid_mask_stopped",
    "numerical_boundary_stopped",
    "stuck",
    "absorbed",
    "escaped",
)

SYNTHETIC_BOUNDARY_PART_ID_MIN = 9000


def ensure_visualization_dirs(output_dir: Path, clean: bool = False) -> dict[str, Path]:
    base = Path(output_dir)
    if clean:
        for legacy in ("graphs", "animations", "visuals"):
            legacy_dir = base / legacy
            if legacy_dir.exists() and legacy_dir.is_dir():
                shutil.rmtree(legacy_dir)
        existing_root = base / "visualizations"
        if existing_root.exists() and existing_root.is_dir():
            shutil.rmtree(existing_root)
    root = base / "visualizations"
    dirs = {
        "root": root,
        "graphs": root / "graphs",
        "animations": root / "animations",
        "mechanics": root / "mechanics",
        "boundary_diagnostics": root / "boundary_diagnostics",
        "reports": root / "reports",
    }
    dirs["root"].mkdir(parents=True, exist_ok=True)
    dirs["reports"].mkdir(parents=True, exist_ok=True)
    return dirs


def write_visualization_index(output_dir: Path, payload: Mapping[str, object]) -> Path:
    dirs = ensure_visualization_dirs(output_dir, clean=False)
    index_path = dirs["reports"] / "visualization_index.json"
    index_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return index_path


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _int_from_reports(key: str, *reports: Mapping[str, object], default: int = 0) -> int:
    for report in reports:
        if key in report:
            try:
                return int(report[key])
            except (TypeError, ValueError):
                return int(default)
    return int(default)


def _optional_float_from_reports(key: str, *reports: Mapping[str, object]) -> float | None:
    for report in reports:
        if key in report:
            try:
                return float(report[key])
            except (TypeError, ValueError):
                return None
    return None


def _format_optional_seconds(value: object) -> str:
    try:
        if value is None:
            return "not_recorded"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "not_recorded"


def build_run_health_summary(output_dir: Path) -> dict[str, object]:
    base = Path(output_dir)
    report = _read_json_if_exists(base / "solver_report.json")
    diagnostics = _read_json_if_exists(base / "collision_diagnostics.json")
    timing_raw = report.get("timing_s", report.get("timing", {}))
    timing = timing_raw if isinstance(timing_raw, Mapping) else {}
    raw_memory = report.get("memory_estimate_bytes", report.get("memory", {}))
    memory = raw_memory if isinstance(raw_memory, Mapping) else {}
    final_csv = base / "final_particles.csv"
    state_counts: dict[str, int] = {}
    if final_csv.exists():
        try:
            state_counts = final_state_counts(pd.read_csv(final_csv))
        except (OSError, ValueError, pd.errors.ParserError):
            state_counts = {}
    health = {
        "particle_count": _int_from_reports("particle_count", report, diagnostics, default=sum(state_counts.values())),
        "released_count": _int_from_reports("released_count", report, diagnostics),
        "invalid_mask_stopped_count": _int_from_reports("invalid_mask_stopped_count", report, diagnostics),
        "numerical_boundary_stopped_count": _int_from_reports("numerical_boundary_stopped_count", report, diagnostics),
        "stuck_count": _int_from_reports("stuck_count", report, diagnostics),
        "absorbed_count": _int_from_reports("absorbed_count", report, diagnostics),
        "field_support_exit_count": _int_from_reports("field_support_exit_count", report, diagnostics),
        "physical_absorbed_count": _int_from_reports("physical_absorbed_count", report, diagnostics),
        "max_hits_reached_count": _int_from_reports("max_hits_reached_count", report, diagnostics),
        "unresolved_crossing_count": _int_from_reports("unresolved_crossing_count", report, diagnostics),
        "nearest_projection_fallback_count": _int_from_reports("nearest_projection_fallback_count", report, diagnostics),
        "boundary_event_failure_count": _int_from_reports("boundary_event_failure_count", report, diagnostics),
        "boundary_event_contract_passed": _int_from_reports("boundary_event_contract_passed", report, diagnostics),
        "contact_sliding_particle_count": int(
            state_counts.get(
                "contact_sliding",
                _int_from_reports("contact_sliding_particle_count", report, diagnostics),
            )
        ),
        "contact_endpoint_stopped_count": int(
            state_counts.get(
                "contact_endpoint_stopped",
                _int_from_reports("contact_endpoint_stopped_count", report, diagnostics),
            )
        ),
        "contact_tangent_step_count": _int_from_reports("contact_tangent_step_count", report, diagnostics),
        "active_outside_geometry_count": _int_from_reports("active_outside_geometry_count", report, diagnostics),
        "contact_sliding_outside_geometry_count": _int_from_reports("contact_sliding_outside_geometry_count", report, diagnostics),
        "nonfinite_position_count": _int_from_reports("nonfinite_position_count", report, diagnostics),
        "nonfinite_velocity_count": _int_from_reports("nonfinite_velocity_count", report, diagnostics),
        "solver_core_s": _optional_float_from_reports("solver_core_s", timing, report, diagnostics),
        "estimated_numpy_bytes": _int_from_reports("estimated_numpy_bytes", memory),
        "final_state_counts": state_counts,
    }
    failure_keys = (
        "invalid_mask_stopped_count",
        "numerical_boundary_stopped_count",
        "max_hits_reached_count",
        "unresolved_crossing_count",
        "nearest_projection_fallback_count",
        "boundary_event_failure_count",
        "active_outside_geometry_count",
        "contact_sliding_outside_geometry_count",
        "nonfinite_position_count",
        "nonfinite_velocity_count",
    )
    failed = any(int(health.get(key, 0)) > 0 for key in failure_keys)
    if int(health.get("boundary_event_contract_passed", 1)) == 0:
        failed = True
    health["status"] = "pass" if not failed else "review"
    return health


def write_run_summary(output_dir: Path, payload: Mapping[str, object]) -> Path:
    dirs = ensure_visualization_dirs(output_dir, clean=False)
    health = dict(payload.get("health_summary", {})) if isinstance(payload.get("health_summary", {}), Mapping) else {}
    modules = payload.get("modules", {})
    module_names = sorted(modules.keys()) if isinstance(modules, Mapping) else []
    lines = [
        "# Run Summary",
        "",
        f"- status: {health.get('status', 'unknown')}",
        f"- output_dir: {Path(output_dir).resolve()}",
        f"- particles: {health.get('particle_count', 0)}",
        f"- released: {health.get('released_count', 0)}",
        f"- solver_core_s: {_format_optional_seconds(health.get('solver_core_s'))}",
        f"- estimated_numpy_bytes: {health.get('estimated_numpy_bytes', 0)}",
        "",
        "## Solver Health",
        "",
    ]
    for key in (
        "invalid_mask_stopped_count",
        "numerical_boundary_stopped_count",
        "max_hits_reached_count",
        "unresolved_crossing_count",
        "nearest_projection_fallback_count",
        "boundary_event_failure_count",
        "contact_sliding_particle_count",
        "contact_endpoint_stopped_count",
        "active_outside_geometry_count",
        "contact_sliding_outside_geometry_count",
        "nonfinite_position_count",
        "nonfinite_velocity_count",
    ):
        lines.append(f"- {key}: {health.get(key, 0)}")
    state_counts = health.get("final_state_counts", {})
    if isinstance(state_counts, Mapping) and state_counts:
        lines.extend(["", "## Final States", ""])
        for name, count in state_counts.items():
            lines.append(f"- {name}: {count}")
    lines.extend(["", "## Visualization Modules", ""])
    if module_names:
        for name in module_names:
            lines.append(f"- {name}")
    else:
        lines.append("- none")
    summary_path = dirs["reports"] / "run_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


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
    return filter_display_boundary_geometry(edges, part_ids)


def filter_display_boundary_geometry(
    boundary_edges: np.ndarray | None,
    boundary_part_ids: np.ndarray | None,
    *,
    synthetic_part_id_min: int = SYNTHETIC_BOUNDARY_PART_ID_MIN,
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
    keep = part_ids < int(synthetic_part_id_min)
    return edges[keep], part_ids[keep]


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
    label_part_ids: bool = False,
    label_fontsize: float = 8.0,
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
    for seg in segs:
        ax.plot(seg[:, 0], seg[:, 1], color="k", linewidth=linewidth, alpha=alpha)
    if label_part_ids:
        for pid in unique:
            mask = pids == pid
            if not np.any(mask):
                continue
            center = segs[mask].mean(axis=(0, 1))
            ax.text(
                float(center[0]),
                float(center[1]),
                str(int(pid)),
                fontsize=label_fontsize,
                ha="center",
                va="center",
                color="black",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
                zorder=5,
            )


def _domain_part_polygons(
    mesh_vertices: np.ndarray | None,
    mesh_triangles: np.ndarray | None = None,
    mesh_triangle_part_ids: np.ndarray | None = None,
    mesh_quads: np.ndarray | None = None,
    mesh_quad_part_ids: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    if mesh_vertices is None:
        return [], np.zeros(0, dtype=np.int32)
    verts = np.asarray(mesh_vertices, dtype=np.float64)
    polygons: list[np.ndarray] = []
    part_ids: list[int] = []

    if mesh_triangles is not None:
        tri = np.asarray(mesh_triangles, dtype=np.int32)
        tri_ids = np.asarray(mesh_triangle_part_ids, dtype=np.int32) if mesh_triangle_part_ids is not None else None
        if tri.ndim == 2 and tri.shape[1] == 3 and tri.size:
            polygons.extend([verts[t] for t in tri])
            if tri_ids is not None and tri_ids.shape[0] == tri.shape[0]:
                part_ids.extend(int(pid) for pid in tri_ids)
            else:
                part_ids.extend([0] * tri.shape[0])

    if mesh_quads is not None:
        quads = np.asarray(mesh_quads, dtype=np.int32)
        quad_ids = np.asarray(mesh_quad_part_ids, dtype=np.int32) if mesh_quad_part_ids is not None else None
        if quads.ndim == 2 and quads.shape[1] == 4 and quads.size:
            polygons.extend([verts[q] for q in quads])
            if quad_ids is not None and quad_ids.shape[0] == quads.shape[0]:
                part_ids.extend(int(pid) for pid in quad_ids)
            else:
                part_ids.extend([0] * quads.shape[0])

    return polygons, np.asarray(part_ids, dtype=np.int32)


def _domain_summary_status(support_fraction: float) -> str:
    if support_fraction >= 0.50:
        return "solver_medium_region"
    if support_fraction > 0.0:
        return "device_part_touching_solver_field"
    return "device_part_no_solver_field"


def domain_part_medium_summary(
    mesh_vertices: np.ndarray | None,
    mesh_triangles: np.ndarray | None,
    mesh_triangle_part_ids: np.ndarray | None,
    mesh_quads: np.ndarray | None,
    mesh_quad_part_ids: np.ndarray | None,
    axis_0: np.ndarray | None,
    axis_1: np.ndarray | None,
    valid_mask: np.ndarray | None,
) -> pd.DataFrame:
    """Classify COMSOL domain part IDs by overlap with solver field support.

    COMSOL geometry can contain device parts that touch the solver medium.
    Those are still parts, not "partial medium"; the support fraction is only
    a diagnostic of how the field grid overlaps each COMSOL domain ID.
    """
    polygons, pids = _domain_part_polygons(
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
    )
    columns = [
        "part_id",
        "element_count",
        "field_supported_element_count",
        "support_fraction",
        "medium_status",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
    ]
    if not polygons:
        return pd.DataFrame(columns=columns)
    axes_ok = axis_0 is not None and axis_1 is not None and valid_mask is not None
    if axes_ok:
        xs = np.asarray(axis_0, dtype=np.float64)
        ys = np.asarray(axis_1, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=bool)
        axes_ok = xs.ndim == 1 and ys.ndim == 1 and mask.shape == (xs.size, ys.size) and xs.size > 1 and ys.size > 1
    supported = np.zeros(len(polygons), dtype=bool)
    if axes_ok:
        centroids = np.asarray([poly.mean(axis=0) for poly in polygons], dtype=np.float64)
        ix = np.searchsorted(xs, centroids[:, 0], side="left")
        iy = np.searchsorted(ys, centroids[:, 1], side="left")
        ix = np.clip(ix, 0, xs.size - 1)
        iy = np.clip(iy, 0, ys.size - 1)
        prev_ix = np.clip(ix - 1, 0, xs.size - 1)
        prev_iy = np.clip(iy - 1, 0, ys.size - 1)
        ix = np.where(np.abs(xs[prev_ix] - centroids[:, 0]) < np.abs(xs[ix] - centroids[:, 0]), prev_ix, ix)
        iy = np.where(np.abs(ys[prev_iy] - centroids[:, 1]) < np.abs(ys[iy] - centroids[:, 1]), prev_iy, iy)
        supported = mask[ix, iy]

    rows: list[dict[str, object]] = []
    for pid in np.unique(pids):
        elem_idx = np.flatnonzero(pids == pid)
        pts = np.vstack([polygons[int(i)] for i in elem_idx])
        count = int(elem_idx.size)
        supported_count = int(np.count_nonzero(supported[elem_idx])) if axes_ok else 0
        fraction = float(supported_count / count) if count else 0.0
        rows.append(
            {
                "part_id": int(pid),
                "element_count": count,
                "field_supported_element_count": supported_count,
                "support_fraction": fraction,
                "medium_status": _domain_summary_status(fraction),
                "x_min_m": float(np.nanmin(pts[:, 0])),
                "x_max_m": float(np.nanmax(pts[:, 0])),
                "y_min_m": float(np.nanmin(pts[:, 1])),
                "y_max_m": float(np.nanmax(pts[:, 1])),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("part_id").reset_index(drop=True)


def medium_status_by_part(summary: pd.DataFrame | None) -> dict[int, str]:
    if summary is None or summary.empty or not {"part_id", "medium_status"}.issubset(summary.columns):
        return {}
    aliases = {
        "active_medium": "solver_medium_region",
        "partial_field_support": "device_part_touching_solver_field",
        "no_medium_or_no_field": "device_part_no_solver_field",
    }
    return {int(row["part_id"]): aliases.get(str(row["medium_status"]), str(row["medium_status"])) for _, row in summary.iterrows()}


def _domain_outline_segments(polygons: list[np.ndarray], part_ids: np.ndarray) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    if not polygons:
        return out
    for pid in np.unique(part_ids):
        counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
        segments: dict[tuple[tuple[float, float], tuple[float, float]], np.ndarray] = {}
        for poly in [polygons[int(i)] for i in np.flatnonzero(part_ids == pid)]:
            n = int(poly.shape[0])
            for i in range(n):
                a = tuple(np.round(poly[i], 12).tolist())
                b = tuple(np.round(poly[(i + 1) % n], 12).tolist())
                key = (a, b) if a <= b else (b, a)
                counts[key] = counts.get(key, 0) + 1
                segments[key] = np.asarray([poly[i], poly[(i + 1) % n]], dtype=np.float64)
        out[int(pid)] = [segments[key] for key, count in counts.items() if count == 1]
    return out


def draw_domain_part_outlines(
    ax: plt.Axes,
    mesh_vertices: np.ndarray | None,
    mesh_triangles: np.ndarray | None = None,
    mesh_triangle_part_ids: np.ndarray | None = None,
    mesh_quads: np.ndarray | None = None,
    mesh_quad_part_ids: np.ndarray | None = None,
    *,
    color: str = "#222222",
    linewidth: float = 0.65,
    alpha: float = 0.95,
    label_part_ids: bool = False,
    label_fontsize: float = 8.0,
) -> None:
    polygons, pids = _domain_part_polygons(
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
    )
    if not polygons:
        return
    outlines = _domain_outline_segments(polygons, pids)
    for segments in outlines.values():
        if segments:
            ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth, alpha=alpha, zorder=4))
    if label_part_ids:
        for pid in np.unique(pids):
            mask = pids == pid
            if not np.any(mask):
                continue
            pts = np.vstack([polygons[i] for i in np.flatnonzero(mask)])
            center = pts.mean(axis=0)
            ax.text(
                float(center[0]),
                float(center[1]),
                str(int(pid)),
                fontsize=label_fontsize,
                ha="center",
                va="center",
                color="black",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.84},
                zorder=6,
            )


def draw_domain_parts(
    ax: plt.Axes,
    mesh_vertices: np.ndarray | None,
    mesh_triangles: np.ndarray | None = None,
    mesh_triangle_part_ids: np.ndarray | None = None,
    mesh_quads: np.ndarray | None = None,
    mesh_quad_part_ids: np.ndarray | None = None,
    *,
    alpha: float = 0.24,
    linewidth: float = 0.08,
    edgecolor: str = "#ffffff",
    label_part_ids: bool = False,
    label_fontsize: float = 8.0,
) -> None:
    polygons, pids = _domain_part_polygons(
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
    )
    if not polygons:
        return
    unique = np.unique(pids)
    facecolors = ["#e6e6e6" for _ in pids]
    coll = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        zorder=0,
    )
    ax.add_collection(coll)
    draw_domain_part_outlines(
        ax,
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
        color="#333333",
        linewidth=max(0.40, linewidth * 5.0),
        alpha=0.80,
        label_part_ids=False,
    )

    if label_part_ids:
        for pid in unique:
            mask = pids == pid
            if not np.any(mask):
                continue
            pts = np.vstack([polygons[i] for i in np.flatnonzero(mask)])
            center = pts.mean(axis=0)
            ax.text(
                float(center[0]),
                float(center[1]),
                str(int(pid)),
                fontsize=label_fontsize,
                ha="center",
                va="center",
                color="black",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                zorder=5,
            )


def draw_domain_parts_by_medium(
    ax: plt.Axes,
    mesh_vertices: np.ndarray | None,
    mesh_triangles: np.ndarray | None = None,
    mesh_triangle_part_ids: np.ndarray | None = None,
    mesh_quads: np.ndarray | None = None,
    mesh_quad_part_ids: np.ndarray | None = None,
    *,
    medium_summary: pd.DataFrame | None = None,
    alpha: float = 0.36,
    linewidth: float = 0.06,
    edgecolor: str = "#ffffff",
    label_part_ids: bool = False,
    label_fontsize: float = 8.0,
    show_legend: bool = False,
) -> None:
    polygons, pids = _domain_part_polygons(
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
    )
    if not polygons:
        return
    status_map = medium_status_by_part(medium_summary)
    unique = np.unique(pids)
    status_colors = {
        "solver_medium_region": "#f7f7f7",
        "device_part_touching_solver_field": "#eeeeee",
        "device_part_no_solver_field": "#d9d9d9",
    }
    facecolors = []
    for pid in pids:
        status = status_map.get(int(pid), "solver_medium_region")
        facecolors.append(status_colors.get(status, status_colors["device_part_no_solver_field"]))
    coll = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0.0,
        alpha=alpha,
        zorder=0,
    )
    ax.add_collection(coll)
    outlines = _domain_outline_segments(polygons, pids)
    outline_specs = {
        "solver_medium_region": ("#222222", "dashed", 0.55),
        "device_part_touching_solver_field": ("#222222", "solid", 0.78),
        "device_part_no_solver_field": ("#222222", "solid", 0.78),
    }
    for pid in unique:
        status = status_map.get(int(pid), "solver_medium_region")
        color, linestyle, width = outline_specs.get(status, outline_specs["device_part_no_solver_field"])
        segments = outlines.get(int(pid), [])
        if segments:
            ax.add_collection(LineCollection(segments, colors=color, linewidths=width, linestyles=linestyle, alpha=0.96, zorder=4))

    if label_part_ids:
        for pid in unique:
            mask = pids == pid
            if not np.any(mask):
                continue
            pts = np.vstack([polygons[i] for i in np.flatnonzero(mask)])
            center = pts.mean(axis=0)
            status = status_map.get(int(pid), "unknown")
            ax.text(
                float(center[0]),
                float(center[1]),
                str(int(pid)),
                fontsize=label_fontsize,
                ha="center",
                va="center",
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#777777" if status != "solver_medium_region" else "none",
                    "alpha": 0.82,
                },
                zorder=6,
            )

    if show_legend:
        handles = [
            Patch(facecolor=status_colors["solver_medium_region"], edgecolor="#222222", linestyle="--", alpha=alpha, label="solver medium region"),
            Patch(facecolor=status_colors["device_part_touching_solver_field"], edgecolor="#222222", alpha=alpha, label="device part touching solver field"),
            Patch(facecolor=status_colors["device_part_no_solver_field"], edgecolor="#222222", alpha=alpha, label="device part without solver field"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.86)


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
