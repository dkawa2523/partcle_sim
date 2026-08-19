"""Shared geometry drawing primitives."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch

from tools.visualization_data import filter_display_boundary_geometry

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

STEP_STATE_ORDER = (
    "active_total",
    "invalid_mask_stopped",
    "numerical_boundary_stopped",
    "stuck",
    "absorbed",
    "escaped",
)


def _draw_boundary_segments(
    ax: plt.Axes,
    segments: np.ndarray,
    linewidth: float = 1.0,
    alpha: float = 0.9,
) -> None:
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color="k", linewidth=linewidth, alpha=alpha)


def _label_boundary_parts(
    ax: plt.Axes,
    segments: np.ndarray,
    part_ids: np.ndarray,
    fontsize: float,
) -> None:
    for part_id in np.unique(part_ids):
        center = segments[part_ids == part_id].mean(axis=(0, 1))
        ax.text(
            float(center[0]),
            float(center[1]),
            str(int(part_id)),
            fontsize=fontsize,
            ha="center",
            va="center",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=5,
        )


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
    segments, part_ids = filter_display_boundary_geometry(
        boundary_edges, boundary_part_ids
    )
    if segments is None:
        return
    _draw_boundary_segments(ax, segments, linewidth, alpha)
    if label_part_ids and part_ids is not None:
        _label_boundary_parts(ax, segments, part_ids, label_fontsize)


def _valid_domain_elements(elements: np.ndarray, vertex_count: int) -> bool:
    return bool(
        elements.ndim == 2 and elements.shape[1] == vertex_count and elements.size
    )


def _domain_element_part_ids(
    part_ids: np.ndarray | None,
    element_count: int,
) -> list[int]:
    if part_ids is not None and part_ids.shape[0] == element_count:
        return [int(part_id) for part_id in part_ids]
    return [0] * element_count


def _domain_elements(
    vertices: np.ndarray,
    connectivity: np.ndarray | None,
    raw_part_ids: np.ndarray | None,
    vertex_count: int,
) -> tuple[list[np.ndarray], list[int]]:
    if connectivity is None:
        return [], []
    elements = np.asarray(connectivity, dtype=np.int32)
    part_ids = (
        np.asarray(raw_part_ids, dtype=np.int32) if raw_part_ids is not None else None
    )
    if not _valid_domain_elements(elements, vertex_count):
        return [], []
    polygons = [vertices[element] for element in elements]
    return polygons, _domain_element_part_ids(part_ids, elements.shape[0])


def _domain_part_polygons(
    mesh_vertices: np.ndarray | None,
    mesh_triangles: np.ndarray | None = None,
    mesh_triangle_part_ids: np.ndarray | None = None,
    mesh_quads: np.ndarray | None = None,
    mesh_quad_part_ids: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    if mesh_vertices is None:
        return [], np.zeros(0, dtype=np.int32)
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    triangle_polygons, triangle_part_ids = _domain_elements(
        vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        3,
    )
    quad_polygons, quad_part_ids = _domain_elements(
        vertices,
        mesh_quads,
        mesh_quad_part_ids,
        4,
    )
    return (
        triangle_polygons + quad_polygons,
        np.asarray(triangle_part_ids + quad_part_ids, dtype=np.int32),
    )


def _domain_summary_status(support_fraction: float) -> str:
    if support_fraction >= 0.50:
        return "solver_medium_region"
    if support_fraction > 0.0:
        return "device_part_touching_solver_field"
    return "device_part_no_solver_field"


_DOMAIN_MEDIUM_SUMMARY_COLUMNS = (
    "part_id",
    "element_count",
    "field_supported_element_count",
    "support_fraction",
    "medium_status",
    "x_min_m",
    "x_max_m",
    "y_min_m",
    "y_max_m",
)


def _field_support_grid(
    axis_0: np.ndarray | None,
    axis_1: np.ndarray | None,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if axis_0 is None or axis_1 is None or valid_mask is None:
        return None
    xs = np.asarray(axis_0, dtype=np.float64)
    ys = np.asarray(axis_1, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    valid = (
        xs.ndim == 1
        and ys.ndim == 1
        and mask.shape == (xs.size, ys.size)
        and xs.size > 1
        and ys.size > 1
    )
    return (xs, ys, mask) if valid else None


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(axis, values, side="left")
    indices = np.clip(indices, 0, axis.size - 1)
    previous = np.clip(indices - 1, 0, axis.size - 1)
    return np.where(
        np.abs(axis[previous] - values) < np.abs(axis[indices] - values),
        previous,
        indices,
    )


def _supported_domain_elements(
    polygons: list[np.ndarray],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if grid is None:
        return np.zeros(len(polygons), dtype=bool)
    xs, ys, mask = grid
    centroids = np.asarray(
        [polygon.mean(axis=0) for polygon in polygons], dtype=np.float64
    )
    x_indices = _nearest_axis_indices(xs, centroids[:, 0])
    y_indices = _nearest_axis_indices(ys, centroids[:, 1])
    return mask[x_indices, y_indices]


def _domain_medium_summary_row(
    part_id: int,
    polygons: list[np.ndarray],
    part_ids: np.ndarray,
    supported: np.ndarray,
) -> dict[str, object]:
    element_indices = np.flatnonzero(part_ids == part_id)
    points = np.vstack([polygons[int(index)] for index in element_indices])
    count = int(element_indices.size)
    supported_count = int(np.count_nonzero(supported[element_indices]))
    fraction = float(supported_count / count)
    return {
        "part_id": int(part_id),
        "element_count": count,
        "field_supported_element_count": supported_count,
        "support_fraction": fraction,
        "medium_status": _domain_summary_status(fraction),
        "x_min_m": float(np.nanmin(points[:, 0])),
        "x_max_m": float(np.nanmax(points[:, 0])),
        "y_min_m": float(np.nanmin(points[:, 1])),
        "y_max_m": float(np.nanmax(points[:, 1])),
    }


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
    if not polygons:
        return pd.DataFrame(columns=_DOMAIN_MEDIUM_SUMMARY_COLUMNS)
    supported = _supported_domain_elements(
        polygons, _field_support_grid(axis_0, axis_1, valid_mask)
    )
    rows = [
        _domain_medium_summary_row(int(part_id), polygons, pids, supported)
        for part_id in np.unique(pids)
    ]
    return (
        pd.DataFrame(rows, columns=_DOMAIN_MEDIUM_SUMMARY_COLUMNS)
        .sort_values("part_id")
        .reset_index(drop=True)
    )


def medium_status_by_part(summary: pd.DataFrame | None) -> dict[int, str]:
    if (
        summary is None
        or summary.empty
        or not {"part_id", "medium_status"}.issubset(summary.columns)
    ):
        return {}
    aliases = {
        "active_medium": "solver_medium_region",
        "partial_field_support": "device_part_touching_solver_field",
        "no_medium_or_no_field": "device_part_no_solver_field",
    }
    return {
        int(row["part_id"]): aliases.get(
            str(row["medium_status"]), str(row["medium_status"])
        )
        for _, row in summary.iterrows()
    }


def _domain_outline_segments(
    polygons: list[np.ndarray], part_ids: np.ndarray
) -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {}
    if not polygons:
        return out
    for pid in np.unique(part_ids):
        counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
        segments: dict[tuple[tuple[float, float], tuple[float, float]], np.ndarray] = {}
        for poly in [polygons[int(i)] for i in np.flatnonzero(part_ids == pid)]:
            n = int(poly.shape[0])
            for i in range(n):
                a = tuple(np.asarray(poly[i], dtype=np.float64).tolist())
                b = tuple(np.asarray(poly[(i + 1) % n], dtype=np.float64).tolist())
                key = (a, b) if a <= b else (b, a)
                counts[key] = counts.get(key, 0) + 1
                segments[key] = np.asarray(
                    [poly[i], poly[(i + 1) % n]], dtype=np.float64
                )
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
            ax.add_collection(
                LineCollection(
                    segments, colors=color, linewidths=linewidth, alpha=alpha, zorder=4
                )
            )
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
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#777777",
                    "alpha": 0.84,
                },
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
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
                zorder=5,
            )


_MEDIUM_STATUS_COLORS = {
    "solver_medium_region": "#f7f7f7",
    "device_part_touching_solver_field": "#eeeeee",
    "device_part_no_solver_field": "#d9d9d9",
}

_MEDIUM_OUTLINE_SPECS = {
    "solver_medium_region": ("#222222", "dashed", 0.55),
    "device_part_touching_solver_field": ("#222222", "solid", 0.78),
    "device_part_no_solver_field": ("#222222", "solid", 0.78),
}


def _medium_facecolors(
    part_ids: np.ndarray,
    status_by_part: Mapping[int, str],
) -> list[str]:
    return [
        _MEDIUM_STATUS_COLORS.get(
            status_by_part.get(int(part_id), "solver_medium_region"),
            _MEDIUM_STATUS_COLORS["device_part_no_solver_field"],
        )
        for part_id in part_ids
    ]


def _add_medium_fill(
    ax: plt.Axes,
    polygons: list[np.ndarray],
    part_ids: np.ndarray,
    status_by_part: Mapping[int, str],
    alpha: float,
) -> None:
    ax.add_collection(
        PolyCollection(
            polygons,
            facecolors=_medium_facecolors(part_ids, status_by_part),
            edgecolors="none",
            linewidths=0.0,
            alpha=alpha,
            zorder=0,
        )
    )


def _add_medium_outlines(
    ax: plt.Axes,
    polygons: list[np.ndarray],
    part_ids: np.ndarray,
    status_by_part: Mapping[int, str],
) -> None:
    outlines = _domain_outline_segments(polygons, part_ids)
    for part_id in np.unique(part_ids):
        status = status_by_part.get(int(part_id), "solver_medium_region")
        color, linestyle, width = _MEDIUM_OUTLINE_SPECS.get(
            status, _MEDIUM_OUTLINE_SPECS["device_part_no_solver_field"]
        )
        segments = outlines.get(int(part_id), [])
        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    linewidths=width,
                    linestyles=linestyle,
                    alpha=0.96,
                    zorder=4,
                )
            )


def _label_medium_parts(
    ax: plt.Axes,
    polygons: list[np.ndarray],
    part_ids: np.ndarray,
    status_by_part: Mapping[int, str],
    fontsize: float,
) -> None:
    for part_id in np.unique(part_ids):
        element_indices = np.flatnonzero(part_ids == part_id)
        center = np.vstack([polygons[index] for index in element_indices]).mean(axis=0)
        status = status_by_part.get(int(part_id), "unknown")
        ax.text(
            float(center[0]),
            float(center[1]),
            str(int(part_id)),
            fontsize=fontsize,
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


def _medium_legend_handles(alpha: float) -> list[Patch]:
    return [
        Patch(
            facecolor=_MEDIUM_STATUS_COLORS["solver_medium_region"],
            edgecolor="#222222",
            linestyle="--",
            alpha=alpha,
            label="solver medium region",
        ),
        Patch(
            facecolor=_MEDIUM_STATUS_COLORS["device_part_touching_solver_field"],
            edgecolor="#222222",
            alpha=alpha,
            label="device part touching solver field",
        ),
        Patch(
            facecolor=_MEDIUM_STATUS_COLORS["device_part_no_solver_field"],
            edgecolor="#222222",
            alpha=alpha,
            label="device part without solver field",
        ),
    ]


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
    _add_medium_fill(ax, polygons, pids, status_map, alpha)
    _add_medium_outlines(ax, polygons, pids, status_map)
    if label_part_ids:
        _label_medium_parts(ax, polygons, pids, status_map, label_fontsize)
    if show_legend:
        ax.legend(
            handles=_medium_legend_handles(alpha),
            loc="upper right",
            fontsize=8,
            framealpha=0.86,
        )


def sample_grid_points(
    arr: np.ndarray, x: np.ndarray, y: np.ndarray, points: np.ndarray
) -> np.ndarray:
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
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError(
                "visualization grid axes must be finite and strictly increasing"
            )
        a = (value - axis[lo]) / denom
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
