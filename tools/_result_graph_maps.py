"""Render geometry, domain, and scalar field maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tools._result_graph_common import PYPLOT as plt
from tools.visualization_common import (
    domain_part_medium_summary,
    draw_boundary_edges,
    draw_domain_parts,
    draw_domain_parts_by_medium,
)


def as_2d_mask(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    return np.zeros((0, 0), dtype=bool)


def domain_medium_summary(
    geometry: dict[str, np.ndarray], field: dict[str, np.ndarray]
) -> pd.DataFrame:
    if not geometry:
        return pd.DataFrame()
    axis_0 = field.get("axis_0", geometry.get("axis_0"))
    axis_1 = field.get("axis_1", geometry.get("axis_1"))
    valid_mask = field.get("valid_mask", geometry.get("valid_mask"))
    if valid_mask is not None:
        valid_mask = as_2d_mask(valid_mask)
    return domain_part_medium_summary(
        geometry.get("mesh_vertices"),
        geometry.get("mesh_triangles"),
        geometry.get("mesh_triangle_part_ids"),
        geometry.get("mesh_quads"),
        geometry.get("mesh_quad_part_ids"),
        axis_0,
        axis_1,
        valid_mask,
    )


def as_2d_quantity(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    raise ValueError(f"expected 2D or steady 3D quantity, got shape={arr.shape}")


def masked_field(value: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    arr = as_2d_quantity(value)
    if valid_mask is None or valid_mask.shape != arr.shape:
        return np.where(np.isfinite(arr), arr, np.nan)
    return np.where(np.asarray(valid_mask, dtype=bool) & np.isfinite(arr), arr, np.nan)


def robust_limits(
    arr: np.ndarray, *, symmetric: bool = False
) -> tuple[float, float] | tuple[None, None]:
    values = np.asarray(arr, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), 99.0))
        limit = max(limit, 1.0e-30)
        return -limit, limit
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        pad = max(abs(float(hi)), 1.0) * 1.0e-6
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def _draw_scalar_domain_overlay(
    ax: plt.Axes,
    geometry: dict[str, np.ndarray],
    medium_summary: pd.DataFrame | None,
) -> None:
    if medium_summary is not None and not medium_summary.empty:
        draw_domain_parts_by_medium(
            ax,
            geometry.get("mesh_vertices"),
            geometry.get("mesh_triangles"),
            geometry.get("mesh_triangle_part_ids"),
            geometry.get("mesh_quads"),
            geometry.get("mesh_quad_part_ids"),
            medium_summary=medium_summary,
            alpha=0.14,
            label_part_ids=False,
        )
    else:
        draw_domain_parts(
            ax,
            geometry.get("mesh_vertices"),
            geometry.get("mesh_triangles"),
            geometry.get("mesh_triangle_part_ids"),
            geometry.get("mesh_quads"),
            geometry.get("mesh_quad_part_ids"),
            alpha=0.12,
            linewidth=0.05,
            edgecolor="#222222",
            label_part_ids=False,
        )


def plot_scalar_map(
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
    geometry = {} if geometry_payload is None else geometry_payload
    xx, yy = np.meshgrid(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), indexing="ij"
    )
    draw_device_structure(
        ax,
        geometry,
        edges,
        edge_part_ids,
        label_domain_parts=False,
        label_boundary_parts=False,
        medium_summary=medium_summary,
    )
    vmin, vmax = robust_limits(arr, symmetric=symmetric)
    pcm = ax.pcolormesh(
        xx,
        yy,
        np.ma.masked_invalid(arr),
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
    )
    _draw_scalar_domain_overlay(ax, geometry, medium_summary)
    draw_boundary_edges(
        ax, edges, edge_part_ids, linewidth=0.8, alpha=0.95, label_part_ids=label_parts
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label=cbar_label)


def draw_device_structure(
    ax: plt.Axes,
    geometry: dict[str, np.ndarray],
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
            geometry.get("mesh_vertices"),
            geometry.get("mesh_triangles"),
            geometry.get("mesh_triangle_part_ids"),
            geometry.get("mesh_quads"),
            geometry.get("mesh_quad_part_ids"),
            medium_summary=medium_summary,
            alpha=domain_alpha,
            label_part_ids=label_domain_parts,
            show_legend=show_medium_legend,
        )
    else:
        draw_domain_parts(
            ax,
            geometry.get("mesh_vertices"),
            geometry.get("mesh_triangles"),
            geometry.get("mesh_triangle_part_ids"),
            geometry.get("mesh_quads"),
            geometry.get("mesh_quad_part_ids"),
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


def save_geometry_maps(
    out_dir: Path,
    geometry: dict[str, np.ndarray],
    edges: np.ndarray | None,
    edge_part_ids: np.ndarray | None,
    medium_summary: pd.DataFrame | None = None,
) -> list[str]:
    if not geometry or "axis_0" not in geometry or "axis_1" not in geometry:
        return []
    saved: list[str] = []
    x = np.asarray(geometry["axis_0"], dtype=np.float64)
    y = np.asarray(geometry["axis_1"], dtype=np.float64)
    if edges is not None:
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        draw_device_structure(
            ax,
            geometry,
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
        draw_device_structure(
            ax,
            geometry,
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
        medium_summary.to_csv(
            out_dir / "22_domain_part_medium_summary.csv", index=False
        )
        saved.append("22_domain_part_medium_summary.csv")
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        draw_device_structure(
            ax,
            geometry,
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
    if "sdf" in geometry:
        sdf = np.asarray(geometry["sdf"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        plot_scalar_map(
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
            geometry_payload=geometry,
            medium_summary=medium_summary,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "13_signed_distance_field_sdf.png", dpi=170)
        plt.close(fig)
        saved.append("13_signed_distance_field_sdf.png")
    if "valid_mask" in geometry:
        valid = np.asarray(geometry["valid_mask"], dtype=bool).astype(float)
        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        plot_scalar_map(
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
            geometry_payload=geometry,
            medium_summary=medium_summary,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "14_geometry_field_support_mask.png", dpi=170)
        plt.close(fig)
        saved.append("14_geometry_field_support_mask.png")
    return saved
