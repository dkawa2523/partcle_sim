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
    STEP_STATE_ORDER,
    axis_limits,
    domain_part_medium_summary,
    draw_boundary_edges,
    draw_domain_parts,
    draw_domain_parts_by_medium,
    ensure_visualization_dirs,
    final_state_counts,
    list_files,
    load_boundary_geometry,
    load_wall_events,
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


def _representative_particle_sample(labels: np.ndarray, sample_size: int, seed: int = 20260402) -> np.ndarray:
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
        picks.extend(int(v) for v in rng.choice(state_indices, size=take, replace=False))
    if len(picks) < sample_size:
        remaining = np.setdiff1d(np.arange(n_particles, dtype=np.int64), np.asarray(picks, dtype=np.int64), assume_unique=False)
        if remaining.size:
            picks.extend(int(v) for v in rng.choice(remaining, size=min(sample_size - len(picks), remaining.size), replace=False))
    return np.sort(np.asarray(picks[:sample_size], dtype=np.int64))


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _case_geometry_payload(case_dir: Path | None) -> dict[str, np.ndarray]:
    if case_dir is None:
        return {}
    return _load_npz_arrays(Path(case_dir) / "generated" / "comsol_geometry_2d.npz")


def _case_field_payload(case_dir: Path | None) -> dict[str, np.ndarray]:
    if case_dir is None:
        return {}
    return _load_npz_arrays(Path(case_dir) / "generated" / "comsol_field_2d.npz")


def _as_2d_mask(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    return np.zeros((0, 0), dtype=bool)


def _domain_medium_summary(geom: dict[str, np.ndarray], field: dict[str, np.ndarray]) -> pd.DataFrame:
    if not geom:
        return pd.DataFrame()
    axis_0 = field.get("axis_0", geom.get("axis_0"))
    axis_1 = field.get("axis_1", geom.get("axis_1"))
    valid_mask = field.get("valid_mask", geom.get("valid_mask"))
    if valid_mask is not None:
        valid_mask = _as_2d_mask(valid_mask)
    return domain_part_medium_summary(
        geom.get("mesh_vertices"),
        geom.get("mesh_triangles"),
        geom.get("mesh_triangle_part_ids"),
        geom.get("mesh_quads"),
        geom.get("mesh_quad_part_ids"),
        axis_0,
        axis_1,
        valid_mask,
    )


def _as_2d_quantity(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    raise ValueError(f"expected 2D or steady 3D quantity, got shape={arr.shape}")


def _masked_field(value: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    arr = _as_2d_quantity(value)
    if valid_mask is None or valid_mask.shape != arr.shape:
        return np.where(np.isfinite(arr), arr, np.nan)
    return np.where(np.asarray(valid_mask, dtype=bool) & np.isfinite(arr), arr, np.nan)


def _robust_limits(arr: np.ndarray, *, symmetric: bool = False) -> tuple[float, float] | tuple[None, None]:
    vals = np.asarray(arr, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return None, None
    if symmetric:
        lim = float(np.nanpercentile(np.abs(finite), 99.0))
        lim = max(lim, 1.0e-30)
        return -lim, lim
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        pad = max(abs(float(hi)), 1.0) * 1.0e-6
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def _plot_scalar_map(
    fig: plt.Figure,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    arr: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    symmetric: bool = False,
    edges: np.ndarray | None = None,
    edge_part_ids: np.ndarray | None = None,
    label_parts: bool = False,
    geometry_payload: dict[str, np.ndarray] | None = None,
    medium_summary: pd.DataFrame | None = None,
) -> None:
    xx, yy = np.meshgrid(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), indexing="ij")
    _draw_device_structure(
        ax,
        geometry_payload or {},
        edges,
        edge_part_ids,
        label_domain_parts=False,
        label_boundary_parts=False,
        medium_summary=medium_summary,
    )
    vmin, vmax = _robust_limits(arr, symmetric=symmetric)
    pcm = ax.pcolormesh(xx, yy, np.ma.masked_invalid(arr), shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.88)
    if medium_summary is not None and not medium_summary.empty:
        draw_domain_parts_by_medium(
            ax,
            (geometry_payload or {}).get("mesh_vertices"),
            (geometry_payload or {}).get("mesh_triangles"),
            (geometry_payload or {}).get("mesh_triangle_part_ids"),
            (geometry_payload or {}).get("mesh_quads"),
            (geometry_payload or {}).get("mesh_quad_part_ids"),
            medium_summary=medium_summary,
            alpha=0.14,
            linewidth=0.05,
            edgecolor="#222222",
            label_part_ids=False,
        )
    else:
        draw_domain_parts(
            ax,
            (geometry_payload or {}).get("mesh_vertices"),
            (geometry_payload or {}).get("mesh_triangles"),
            (geometry_payload or {}).get("mesh_triangle_part_ids"),
            (geometry_payload or {}).get("mesh_quads"),
            (geometry_payload or {}).get("mesh_quad_part_ids"),
            alpha=0.12,
            linewidth=0.05,
            edgecolor="#222222",
            label_part_ids=False,
        )
    draw_boundary_edges(ax, edges, edge_part_ids, linewidth=0.8, alpha=0.95, label_part_ids=label_parts)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)


def _draw_device_structure(
    ax: plt.Axes,
    geom: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    *,
    label_domain_parts: bool = False,
    label_boundary_parts: bool = False,
    domain_alpha: float = 0.24,
    boundary_linewidth: float = 0.9,
    medium_summary: pd.DataFrame | None = None,
    show_medium_legend: bool = False,
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
            alpha=domain_alpha,
            linewidth=0.04,
            label_part_ids=label_domain_parts,
            show_legend=show_medium_legend,
        )
    else:
        draw_domain_parts(
            ax,
            geom.get("mesh_vertices"),
            geom.get("mesh_triangles"),
            geom.get("mesh_triangle_part_ids"),
            geom.get("mesh_quads"),
            geom.get("mesh_quad_part_ids"),
            alpha=domain_alpha,
            linewidth=0.04,
            label_part_ids=label_domain_parts,
        )
    draw_boundary_edges(
        ax,
        edges,
        edge_part_ids,
        linewidth=boundary_linewidth,
        alpha=0.95,
        label_part_ids=label_boundary_parts,
    )


def _cumulative_event_count(times: np.ndarray, wall_events: pd.DataFrame, outcomes: set[str]) -> np.ndarray:
    if wall_events.empty or "time_s" not in wall_events.columns or "outcome" not in wall_events.columns:
        return np.zeros_like(times, dtype=np.float64)
    selected = wall_events.loc[wall_events["outcome"].astype(str).isin(outcomes), "time_s"].to_numpy(dtype=np.float64).copy()
    if selected.size == 0:
        return np.zeros_like(times, dtype=np.float64)
    selected.sort()
    return np.searchsorted(selected, times, side="right").astype(np.float64)


def _save_geometry_maps(
    out_dir: Path,
    geom: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None = None,
) -> list[str]:
    if not geom or "axis_0" not in geom or "axis_1" not in geom:
        return []
    saved: list[str] = []
    x = np.asarray(geom["axis_0"], dtype=np.float64)
    y = np.asarray(geom["axis_1"], dtype=np.float64)
    if edges is not None:
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        _draw_device_structure(
            ax,
            geom,
            edges,
            edge_part_ids,
            label_domain_parts=False,
            label_boundary_parts=False,
            domain_alpha=0.42,
            boundary_linewidth=1.1,
            medium_summary=medium_summary,
            show_medium_legend=True,
        )
        ax.set_title("COMSOL Device Parts and Solver Boundary")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        fig.tight_layout()
        fig.savefig(out_dir / "11_device_parts_geometry.png", dpi=170)
        plt.close(fig)
        saved.append("11_device_parts_geometry.png")

        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        _draw_device_structure(
            ax,
            geom,
            edges,
            edge_part_ids,
            label_domain_parts=True,
            label_boundary_parts=True,
            domain_alpha=0.42,
            boundary_linewidth=1.1,
            medium_summary=medium_summary,
            show_medium_legend=True,
        )
        ax.set_title("COMSOL Device Parts with Domain and Boundary IDs")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        fig.tight_layout()
        fig.savefig(out_dir / "12_device_parts_with_ids.png", dpi=170)
        plt.close(fig)
        saved.append("12_device_parts_with_ids.png")
    if medium_summary is not None and not medium_summary.empty:
        medium_summary.to_csv(out_dir / "22_domain_part_medium_summary.csv", index=False)
        saved.append("22_domain_part_medium_summary.csv")
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        _draw_device_structure(
            ax,
            geom,
            edges,
            edge_part_ids,
            label_domain_parts=True,
            label_boundary_parts=True,
            domain_alpha=0.58,
            boundary_linewidth=1.1,
            medium_summary=medium_summary,
            show_medium_legend=True,
        )
        ax.set_title("COMSOL Domain Parts Classified by Field Support")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        fig.tight_layout()
        fig.savefig(out_dir / "22_domain_parts_medium_support.png", dpi=170)
        plt.close(fig)
        saved.append("22_domain_parts_medium_support.png")
    if "sdf" in geom:
        sdf = np.asarray(geom["sdf"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        _plot_scalar_map(
            fig,
            ax,
            x,
            y,
            sdf,
            title="Signed Distance Field (SDF)",
            cbar_label="sdf [m]",
            cmap="coolwarm",
            symmetric=True,
            edges=edges,
            edge_part_ids=edge_part_ids,
            label_parts=True,
            geometry_payload=geom,
            medium_summary=medium_summary,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "13_signed_distance_field_sdf.png", dpi=170)
        plt.close(fig)
        saved.append("13_signed_distance_field_sdf.png")
    if "valid_mask" in geom:
        valid = np.asarray(geom["valid_mask"], dtype=bool).astype(float)
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        _plot_scalar_map(
            fig,
            ax,
            x,
            y,
            valid,
            title="Geometry/Field Support Mask",
            cbar_label="inside/support mask",
            cmap="Blues",
            edges=edges,
            edge_part_ids=edge_part_ids,
            label_parts=False,
            geometry_payload=geom,
            medium_summary=medium_summary,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "14_geometry_field_support_mask.png", dpi=170)
        plt.close(fig)
        saved.append("14_geometry_field_support_mask.png")
    return saved


def _save_field_maps(
    out_dir: Path,
    field: dict[str, np.ndarray],
    geom: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None = None,
) -> list[str]:
    required = {"axis_0", "axis_1", "valid_mask"}
    if not required.issubset(field):
        return []
    saved: list[str] = []
    x = np.asarray(field["axis_0"], dtype=np.float64)
    y = np.asarray(field["axis_1"], dtype=np.float64)
    mask = np.asarray(field["valid_mask"], dtype=bool)

    def maybe(name: str) -> np.ndarray | None:
        if name not in field:
            return None
        return _masked_field(field[name], mask)

    ux = maybe("ux")
    uy = maybe("uy")
    ex = maybe("E_x")
    ey = maybe("E_y")
    mu = maybe("mu")
    totals: list[tuple[str, str, np.ndarray, str, bool]] = []
    if ux is not None and uy is not None:
        totals.append(("Flow speed |u|", "|u| [m/s]", np.sqrt(ux * ux + uy * uy), "viridis", False))
    if ex is not None and ey is not None:
        totals.append(("Electric field |E|", "|E| [V/m]", np.sqrt(ex * ex + ey * ey), "plasma", False))
    if mu is not None:
        totals.append(("Dynamic viscosity mu", "mu [Pa s]", mu, "cividis", False))
    if totals:
        n = len(totals)
        cols = min(2, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.6 * cols, 5.4 * rows), squeeze=False)
        for i, (title, label, arr, cmap, symmetric) in enumerate(totals):
            _plot_scalar_map(
                fig,
                axes.ravel()[i],
                x,
                y,
                arr,
                title=title,
                cbar_label=label,
                cmap=cmap,
                symmetric=symmetric,
                edges=edges,
                edge_part_ids=edge_part_ids,
                geometry_payload=geom,
                medium_summary=medium_summary,
            )
        for ax_empty in axes.ravel()[len(totals):]:
            ax_empty.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "15_mechanics_field_totals.png", dpi=170)
        plt.close(fig)
        saved.append("15_mechanics_field_totals.png")

    component_groups = [
        (
            "16_flow_components_ux_uy.png",
            [("ux", "u_x [m/s]", ux, "coolwarm", True), ("uy", "u_y [m/s]", uy, "coolwarm", True)],
        ),
        (
            "18_electric_field_components_ex_ey.png",
            [("E_x", "E_x [V/m]", ex, "coolwarm", True), ("E_y", "E_y [V/m]", ey, "coolwarm", True)],
        ),
    ]
    for filename, specs in component_groups:
        specs = [spec for spec in specs if spec[2] is not None]
        if not specs:
            continue
        fig, axes = plt.subplots(1, len(specs), figsize=(6.7 * len(specs), 5.4), squeeze=False)
        for ax_plot, (title, label, arr, cmap, symmetric) in zip(axes.ravel(), specs):
            _plot_scalar_map(
                fig,
                ax_plot,
                x,
                y,
                arr,
                title=title,
                cbar_label=label,
                cmap=cmap,
                symmetric=symmetric,
                edges=edges,
                edge_part_ids=edge_part_ids,
                geometry_payload=geom,
                medium_summary=medium_summary,
            )
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=170)
        plt.close(fig)
        saved.append(filename)

    scalar_specs = []
    for name, label, cmap, symmetric in [
        ("T", "T [K]", "inferno", False),
        ("p", "p", "coolwarm", True),
        ("rho_g", "rho_g [kg/m^3]", "viridis", False),
        ("phi", "phi [V]", "coolwarm", True),
        ("ne", "n_e [1/m^3]", "magma", False),
        ("Te", "T_e [eV]", "plasma", False),
    ]:
        arr = maybe(name)
        if arr is not None:
            scalar_specs.append((name, label, arr, cmap, symmetric))
    if scalar_specs:
        cols = 3 if len(scalar_specs) >= 3 else len(scalar_specs)
        rows = int(np.ceil(len(scalar_specs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.1 * rows), squeeze=False)
        for ax_plot, (title, label, arr, cmap, symmetric) in zip(axes.ravel(), scalar_specs):
            _plot_scalar_map(
                fig,
                ax_plot,
                x,
                y,
                arr,
                title=title,
                cbar_label=label,
                cmap=cmap,
                symmetric=symmetric,
                edges=edges,
                edge_part_ids=edge_part_ids,
                geometry_payload=geom,
                medium_summary=medium_summary,
            )
        for ax_empty in axes.ravel()[len(scalar_specs):]:
            ax_empty.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "19_scalar_physics_fields.png", dpi=170)
        plt.close(fig)
        saved.append("19_scalar_physics_fields.png")
    return saved


def _save_drag_gas_property_maps(
    out_dir: Path,
    field: dict[str, np.ndarray],
    geom: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    report: dict,
    medium_summary: pd.DataFrame | None = None,
) -> list[str]:
    gas_report = report.get("drag_gas_properties", {}) if isinstance(report, dict) else {}
    if not isinstance(gas_report, dict):
        gas_report = {}
    rows = []
    saved: list[str] = []
    source_rows = [
        (
            "rho_g",
            "density",
            "rho_g [kg/m^3]",
            gas_report.get("density_source", "unknown"),
            gas_report.get("fallback_density_kgm3", np.nan),
            gas_report.get("density_used_by_drag_model", 0),
        ),
        (
            "T",
            "temperature",
            "T [K]",
            gas_report.get("temperature_source", "unknown"),
            gas_report.get("fallback_temperature_K", np.nan),
            gas_report.get("temperature_used_by_drag_model", 0),
        ),
        (
            "mu",
            "dynamic_viscosity",
            "mu [Pa s]",
            gas_report.get("dynamic_viscosity_source", "unknown"),
            gas_report.get("fallback_dynamic_viscosity_Pas", np.nan),
            gas_report.get("dynamic_viscosity_used_by_drag_model", 0),
        ),
        ("p", "pressure_diagnostic", "p", "diagnostic_only_not_used_by_drag", np.nan, 0),
    ]
    mask = _as_2d_mask(field["valid_mask"]) if "valid_mask" in field else None
    for field_name, role, label, source, fallback, used_by_drag_model in source_rows:
        row = {
            "role": role,
            "field_quantity": field_name,
            "source": str(source),
            "fallback_value": fallback,
            "used_by_drag": int(used_by_drag_model),
        }
        if field_name in field:
            arr = _masked_field(field[field_name], mask)
            vals = arr[np.isfinite(arr)]
            if vals.size:
                row.update(
                    {
                        "field_min": float(np.nanmin(vals)),
                        "field_p50": float(np.nanpercentile(vals, 50.0)),
                        "field_p90": float(np.nanpercentile(vals, 90.0)),
                        "field_max": float(np.nanmax(vals)),
                        "field_mean": float(np.nanmean(vals)),
                    }
                )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "27_drag_gas_property_sources.csv", index=False)
    saved.append("27_drag_gas_property_sources.csv")
    required = {"axis_0", "axis_1", "valid_mask"}
    if not required.issubset(field):
        return saved
    x = np.asarray(field["axis_0"], dtype=np.float64)
    y = np.asarray(field["axis_1"], dtype=np.float64)
    specs = []
    for field_name, role, label, source, _fallback, _used_by_drag_model in source_rows:
        if field_name not in field:
            continue
        arr = _masked_field(field[field_name], mask)
        title = f"{field_name}: {source}"
        symmetric = role == "pressure_diagnostic"
        cmap = "coolwarm" if symmetric else "viridis"
        if field_name == "T":
            cmap = "inferno"
        elif field_name == "mu":
            cmap = "cividis"
        specs.append((title, label, arr, cmap, symmetric))
    if not specs:
        return saved
    cols = 2 if len(specs) > 1 else 1
    rows_n = int(np.ceil(len(specs) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(6.6 * cols, 5.2 * rows_n), squeeze=False)
    for ax_plot, (title, label, arr, cmap, symmetric) in zip(axes.ravel(), specs):
        _plot_scalar_map(
            fig,
            ax_plot,
            x,
            y,
            arr,
            title=title,
            cbar_label=label,
            cmap=cmap,
            symmetric=symmetric,
            edges=edges,
            edge_part_ids=edge_part_ids,
            geometry_payload=geom,
            medium_summary=medium_summary,
        )
    for ax_empty in axes.ravel()[len(specs):]:
        ax_empty.axis("off")
    fig.suptitle(
        "Gas properties for drag; pressure is diagnostic only",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "27_drag_gas_properties_used_by_drag.png", dpi=170)
    plt.close(fig)
    saved.append("27_drag_gas_properties_used_by_drag.png")
    return saved


def _save_wall_event_locations(
    out_dir: Path,
    output_dir: Path,
    geom: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None = None,
) -> str | None:
    path = output_dir / "wall_events.csv"
    if not path.exists():
        return None
    wall_events = pd.read_csv(path)
    if wall_events.empty or not {"hit_x_m", "hit_y_m", "outcome"}.issubset(wall_events.columns):
        return None
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    _draw_device_structure(
        ax,
        geom,
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


def _save_trajectories_by_state(
    out_dir: Path,
    positions: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    geom: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None = None,
) -> str | None:
    if positions.shape[2] != 2 or sample_indices.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    _draw_device_structure(
        ax,
        geom,
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
        idxs = sample_indices[labels[sample_indices] == name]
        if idxs.size == 0:
            continue
        for idx in idxs:
            tr = positions[:, int(idx), :]
            ax.plot(tr[:, 0], tr[:, 1], color=STATE_COLORS[name], linewidth=0.75, alpha=0.42)
        end = positions[-1, idxs, :]
        ax.scatter(end[:, 0], end[:, 1], s=8, color=STATE_COLORS[name], alpha=0.75, label=f"{name} ({idxs.size})")
    ax.set_title(f"Sampled Particle Trajectories by Final State ({sample_indices.size} particles)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    (x_lim, y_lim) = axis_limits(positions, edges)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    filename = "21_trajectories_by_final_state.png"
    fig.savefig(out_dir / filename, dpi=170)
    plt.close(fig)
    return filename


def _save_comsol_style_overlays(
    out_dir: Path,
    positions: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    field: dict[str, np.ndarray],
    geom: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    wall_events: pd.DataFrame,
    medium_summary: pd.DataFrame | None = None,
) -> list[str]:
    if positions.shape[2] != 2 or not {"axis_0", "axis_1", "valid_mask"}.issubset(field):
        return []
    saved: list[str] = []
    x = np.asarray(field["axis_0"], dtype=np.float64)
    y = np.asarray(field["axis_1"], dtype=np.float64)
    mask = _as_2d_mask(field["valid_mask"])
    xx, yy = np.meshgrid(x, y, indexing="ij")

    field_specs: list[tuple[str, str, np.ndarray, str]] = []
    if "E_x" in field and "E_y" in field:
        ex = _masked_field(field["E_x"], mask)
        ey = _masked_field(field["E_y"], mask)
        field_specs.append(("Electric field |E|", "|E| [V/m]", np.sqrt(ex * ex + ey * ey), "plasma"))
    if "ux" in field and "uy" in field:
        ux = _masked_field(field["ux"], mask)
        uy = _masked_field(field["uy"], mask)
        field_specs.append(("Flow speed |u|", "|u| [m/s]", np.sqrt(ux * ux + uy * uy), "viridis"))
    if not field_specs:
        return []

    title, label, arr, cmap = field_specs[0]
    vmin, vmax = _robust_limits(arr)
    x_lim, y_lim = axis_limits(positions, edges)
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    _draw_device_structure(
        ax,
        geom,
        edges,
        edge_part_ids,
        label_domain_parts=True,
        label_boundary_parts=True,
        domain_alpha=0.30,
        boundary_linewidth=0.95,
        medium_summary=medium_summary,
        show_medium_legend=True,
    )
    pcm = ax.pcolormesh(xx, yy, np.ma.masked_invalid(arr), shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.72)
    for name in STATE_ORDER:
        idxs = sample_indices[labels[sample_indices] == name]
        if idxs.size == 0:
            continue
        for idx in idxs:
            tr = positions[:, int(idx), :]
            ax.plot(tr[:, 0], tr[:, 1], color=STATE_COLORS[name], linewidth=0.65, alpha=0.42, zorder=3)
    draw_boundary_edges(ax, edges, edge_part_ids, linewidth=0.9, alpha=0.98, label_part_ids=False)
    ax.set_title(f"COMSOL-style {title} + Particle Trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label=label)
    fig.tight_layout()
    fig.savefig(out_dir / "23_comsol_style_field_and_trajectories.png", dpi=170)
    plt.close(fig)
    saved.append("23_comsol_style_field_and_trajectories.png")

    all_points = positions.reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    _draw_device_structure(
        ax,
        geom,
        edges,
        edge_part_ids,
        label_domain_parts=True,
        label_boundary_parts=True,
        domain_alpha=0.34,
        boundary_linewidth=0.95,
        medium_summary=medium_summary,
        show_medium_legend=True,
    )
    hist = ax.hist2d(all_points[:, 0], all_points[:, 1], bins=190, cmap="magma", alpha=0.76)
    draw_domain_parts_by_medium(
        ax,
        geom.get("mesh_vertices"),
        geom.get("mesh_triangles"),
        geom.get("mesh_triangle_part_ids"),
        geom.get("mesh_quads"),
        geom.get("mesh_quad_part_ids"),
        medium_summary=medium_summary,
        alpha=0.16,
        linewidth=0.05,
        edgecolor="#222222",
        label_part_ids=False,
    )
    wall_event_legend_added = False
    if not wall_events.empty and {"hit_x_m", "hit_y_m", "outcome"}.issubset(wall_events.columns):
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
            wall_event_legend_added = True
    draw_boundary_edges(ax, edges, edge_part_ids, linewidth=0.9, alpha=0.98, label_part_ids=False)
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
    fig.savefig(out_dir / "24_comsol_style_particle_density_and_events.png", dpi=170)
    plt.close(fig)
    saved.append("24_comsol_style_particle_density_and_events.png")
    return saved


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
    geometry_payload = _case_geometry_payload(case_dir) if spatial_dim == 2 else {}
    field_payload = _case_field_payload(case_dir) if spatial_dim == 2 else {}
    medium_summary = _domain_medium_summary(geometry_payload, field_payload) if spatial_dim == 2 else pd.DataFrame()
    wall_events = load_wall_events(output_dir)
    wall_part_summary = load_wall_part_summary(output_dir)
    out_dir = ensure_visualization_dirs(output_dir)["graphs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    final_labels = state_labels(final_df)
    contact_state_counts_by_part: list[dict[str, object]] = []

    time_s = step_df["time_s"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name in STEP_STATE_ORDER:
        values = step_state_count_series(step_df, name)
        if not np.any(values):
            continue
        ax.plot(time_s, values, label=name, color=STATE_COLORS[name], linewidth=2.0)
    ax.set_title("Solver State Counts vs Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    reflected = _cumulative_event_count(time_s, wall_events, {"reflected_specular", "reflected_diffuse"})
    if np.any(reflected):
        ax2 = ax.twinx()
        ax2.plot(time_s, reflected, label="cumulative_reflections", color="#4c78a8", linewidth=1.8, linestyle="--")
        ax2.set_ylabel("cumulative reflected wall events")
        ax2.tick_params(axis="y", colors="#4c78a8")
        total_reflected = int(reflected[-1])
        ax.text(
            0.02,
            0.96,
            f"reflections: {total_reflected}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.84},
        )
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "01_state_counts_time_series.png", dpi=170)
    plt.close(fig)
    if not wall_events.empty:
        pd.DataFrame(
            {
                "time_s": time_s,
                "cumulative_reflected": reflected.astype(int),
                "cumulative_stuck": _cumulative_event_count(time_s, wall_events, {"stuck"}).astype(int),
                "cumulative_absorbed": _cumulative_event_count(time_s, wall_events, {"absorbed"}).astype(int),
                "cumulative_escaped": _cumulative_event_count(time_s, wall_events, {"escaped"}).astype(int),
            }
        ).to_csv(out_dir / "01_wall_event_cumulative_counts.csv", index=False)

    state_counts = final_state_counts(final_df)
    pd.DataFrame(
        [{'state': name, 'count': int(state_counts.get(name, 0))} for name in STATE_ORDER]
    ).to_csv(out_dir / "02_final_state_counts.csv", index=False)
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
        _draw_device_structure(
            ax,
            geometry_payload,
            edges,
            edge_part_ids,
            domain_alpha=0.28,
            boundary_linewidth=0.9,
            medium_summary=medium_summary,
        )
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
        _draw_device_structure(
            ax,
            geometry_payload,
            edges,
            edge_part_ids,
            domain_alpha=0.24,
            boundary_linewidth=0.75,
            medium_summary=medium_summary,
        )
        hist = ax.hist2d(all_points[:, 0], all_points[:, 1], bins=180, cmap="magma", alpha=0.82)
        draw_domain_parts_by_medium(
            ax,
            geometry_payload.get("mesh_vertices"),
            geometry_payload.get("mesh_triangles"),
            geometry_payload.get("mesh_triangle_part_ids"),
            geometry_payload.get("mesh_quads"),
            geometry_payload.get("mesh_quad_part_ids"),
            medium_summary=medium_summary,
            alpha=0.12,
            linewidth=0.04,
            edgecolor="#222222",
        )
        draw_boundary_edges(ax, edges, edge_part_ids, linewidth=0.75, alpha=0.95)
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

    n_particles = positions.shape[1]
    pick = _representative_particle_sample(final_labels, min(sample_trajectories, n_particles))
    if spatial_dim == 2:
        fig, ax = plt.subplots(figsize=(8.2, 5.9))
        _draw_device_structure(
            ax,
            geometry_payload,
            edges,
            edge_part_ids,
            domain_alpha=0.26,
            boundary_linewidth=0.8,
            medium_summary=medium_summary,
        )
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

    if {"contact_part_id", "contact_sliding", "contact_endpoint_stopped"}.issubset(final_df.columns):
        contact_rows = final_df.loc[
            final_df["contact_sliding"].astype(bool) | final_df["contact_endpoint_stopped"].astype(bool),
            ["contact_part_id", "contact_sliding", "contact_endpoint_stopped"],
        ].copy()
        if not contact_rows.empty:
            contact_rows["contact_state"] = np.where(
                contact_rows["contact_endpoint_stopped"].astype(bool),
                "contact_endpoint_stopped",
                "contact_sliding",
            )
            contact_summary = (
                contact_rows.groupby(["contact_part_id", "contact_state"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values(["contact_part_id", "contact_state"])
            )
            contact_state_counts_by_part = [
                {
                    "contact_part_id": int(row["contact_part_id"]),
                    "contact_state": str(row["contact_state"]),
                    "count": int(row["count"]),
                }
                for _, row in contact_summary.iterrows()
            ]
            contact_summary.to_csv(out_dir / "07_contact_states_by_boundary_part.csv", index=False)
            pivot = contact_summary.pivot_table(
                index="contact_part_id",
                columns="contact_state",
                values="count",
                aggfunc="sum",
                fill_value=0,
            ).sort_index()
            fig, ax = plt.subplots(figsize=(8.8, 4.8))
            bottom = np.zeros(pivot.shape[0], dtype=np.float64)
            x_idx = np.arange(pivot.shape[0], dtype=np.float64)
            for state_name in ("contact_sliding", "contact_endpoint_stopped"):
                if state_name not in pivot.columns:
                    continue
                values = pivot[state_name].to_numpy(dtype=np.float64)
                ax.bar(x_idx, values, bottom=bottom, color=STATE_COLORS[state_name], label=state_name, width=0.78)
                bottom += values
            ax.set_xticks(x_idx, [str(int(v)) for v in pivot.index.to_numpy()])
            ax.set_title("Contact States by Boundary Part")
            ax.set_xlabel("boundary part_id")
            ax.set_ylabel("particle count")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "07_contact_states_by_boundary_part.png", dpi=170)
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
        fig.savefig(out_dir / "08_wall_law_counts.png", dpi=170)
        plt.close(fig)

    if not wall_part_summary.empty:
        pivot = (
            wall_part_summary.pivot_table(index="part_id", columns="outcome", values="count", aggfunc="sum", fill_value=0)
            .sort_index()
        )
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(9.2, 5.2))
            outcome_order = [
                name
                for name in ("stuck", "reflected_specular", "reflected_diffuse", "absorbed", "escaped")
                if name in pivot.columns
            ]
            outcome_order.extend(name for name in pivot.columns if name not in outcome_order)
            bottom = np.zeros(pivot.shape[0], dtype=np.float64)
            color_map = {
                "stuck": "#d62728",
                "reflected_specular": "#4c78a8",
                "reflected_diffuse": "#72b7b2",
                "absorbed": "#2ca02c",
                "escaped": "#ff7f0e",
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
            fig.savefig(out_dir / "09_wall_interactions_by_part_outcome.png", dpi=170)
            plt.close(fig)

    stuck_summary = wall_part_summary[wall_part_summary["outcome"] == "stuck"].copy() if not wall_part_summary.empty else pd.DataFrame()
    if not stuck_summary.empty:
        stuck_counts = (
            stuck_summary.groupby("part_id", as_index=False)
            .agg(stuck_count=("count", "sum"))
            .sort_values(["stuck_count", "part_id"], ascending=[False, True])
        )
        stuck_counts.to_csv(out_dir / "10_stuck_counts_by_boundary_part.csv", index=False)

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(stuck_counts["part_id"].astype(str), stuck_counts["stuck_count"], color="#c44e52")
        ax.set_title("Wall Sticking Counts by Boundary Part")
        ax.set_xlabel("boundary part_id")
        ax.set_ylabel("stuck count")
        for idx, value in enumerate(stuck_counts["stuck_count"].tolist()):
            ax.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "10_stuck_counts_by_boundary_part.png", dpi=170)
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
            stuck_counts.to_csv(out_dir / "10_stuck_counts_by_boundary_part.csv", index=False)

            fig, ax = plt.subplots(figsize=(8.0, 4.8))
            ax.bar(stuck_counts["part_id"].astype(str), stuck_counts["stuck_count"], color="#c44e52")
            ax.set_title("Final Stuck Positions by Boundary Part")
            ax.set_xlabel("boundary part_id")
            ax.set_ylabel("stuck count")
            for idx, value in enumerate(stuck_counts["stuck_count"].tolist()):
                ax.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "10_stuck_counts_by_boundary_part.png", dpi=170)
            plt.close(fig)

    extra_graph_files: list[str] = []
    if spatial_dim == 2:
        extra_graph_files.extend(_save_geometry_maps(out_dir, geometry_payload, edges, edge_part_ids, medium_summary))
        extra_graph_files.extend(_save_field_maps(out_dir, field_payload, geometry_payload, edges, edge_part_ids, medium_summary))
        extra_graph_files.extend(
            _save_drag_gas_property_maps(
                out_dir,
                field_payload,
                geometry_payload,
                edges,
                edge_part_ids,
                report,
                medium_summary,
            )
        )
        event_plot = _save_wall_event_locations(out_dir, output_dir, geometry_payload, edges, edge_part_ids, medium_summary)
        if event_plot is not None:
            extra_graph_files.append(event_plot)
        trajectory_state_plot = _save_trajectories_by_state(out_dir, positions, final_labels, pick, geometry_payload, edges, edge_part_ids, medium_summary)
        if trajectory_state_plot is not None:
            extra_graph_files.append(trajectory_state_plot)
        extra_graph_files.extend(
            _save_comsol_style_overlays(
                out_dir,
                positions,
                final_labels,
                pick,
                field_payload,
                geometry_payload,
                edges,
                edge_part_ids,
                wall_events,
                medium_summary,
            )
        )

    summary = {
        "plot_dir": str(out_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "case_dir": str(case_dir.resolve()) if case_dir is not None else "",
        "spatial_dim": int(spatial_dim),
        "files": list_files(out_dir, (".png", ".csv", ".json")),
        "save_frame_count": int(len(frame_df)),
        "particle_count": int(len(final_df)),
        "final_state_counts": state_counts,
        "contact_state_counts_by_part": contact_state_counts_by_part,
        "used_wall_part_summary": bool(not wall_part_summary.empty),
        "extra_graph_files": extra_graph_files,
        "domain_medium_status_counts": (
            medium_summary["medium_status"].value_counts().astype(int).to_dict()
            if not medium_summary.empty and "medium_status" in medium_summary.columns
            else {}
        ),
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
