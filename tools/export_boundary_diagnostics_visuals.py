from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
    sample_valid_mask_status,
)
from tools.visualization_common import (
    domain_part_medium_summary,
    draw_boundary_edges,
    draw_domain_parts_by_medium,
    ensure_visualization_dirs,
    filter_display_boundary_geometry,
    require_2d_quantity,
)


def _valid_mask_status_grid(valid_mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    axes = (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    status = np.zeros(np.asarray(valid_mask).shape, dtype=np.uint8)
    for ix, xv in enumerate(axes[0]):
        for iy, yv in enumerate(axes[1]):
            status[ix, iy] = np.uint8(
                sample_valid_mask_status(
                    valid_mask,
                    axes,
                    np.asarray([float(xv), float(yv)], dtype=np.float64),
                )
            )
    return status


def export_boundary_diagnostics(case_dir: Path, output_dir: Path, normal_band_m: float = 2.5e-3, quiver_stride: int = 10) -> Path:
    geom_npz = case_dir / "generated" / "comsol_geometry_2d.npz"
    field_npz = case_dir / "generated" / "comsol_field_2d.npz"
    if not geom_npz.exists():
        raise FileNotFoundError(f"Geometry npz not found: {geom_npz}")
    if not field_npz.exists():
        raise FileNotFoundError(f"Field npz not found: {field_npz}")

    with np.load(geom_npz) as g:
        x = np.asarray(g["axis_0"], dtype=np.float64)
        y = np.asarray(g["axis_1"], dtype=np.float64)
        sdf = np.asarray(g["sdf"], dtype=np.float64)
        nx = np.asarray(g["normal_0"], dtype=np.float64)
        ny = np.asarray(g["normal_1"], dtype=np.float64)
        geom_valid_mask = np.asarray(g["valid_mask"], dtype=bool)
        boundary_edges = np.asarray(g["boundary_edges"], dtype=np.float64)
        boundary_part_ids = np.asarray(g["boundary_edge_part_ids"], dtype=np.int32)
        mesh_vertices = np.asarray(g["mesh_vertices"], dtype=np.float64)
        mesh_triangles = np.asarray(g["mesh_triangles"], dtype=np.int32) if "mesh_triangles" in g else None
        mesh_triangle_part_ids = np.asarray(g["mesh_triangle_part_ids"], dtype=np.int32) if "mesh_triangle_part_ids" in g else None
        mesh_quads = np.asarray(g["mesh_quads"], dtype=np.int32)
        mesh_quad_part_ids = np.asarray(g["mesh_quad_part_ids"], dtype=np.int32) if "mesh_quad_part_ids" in g else None
    boundary_edges, boundary_part_ids = filter_display_boundary_geometry(boundary_edges, boundary_part_ids)
    if boundary_edges is None or boundary_part_ids is None:
        raise ValueError("boundary diagnostics require displayable boundary edges with part IDs")

    with np.load(field_npz) as f:
        ux = require_2d_quantity(f, "ux", "boundary diagnostics")
        uy = require_2d_quantity(f, "uy", "boundary diagnostics")
        field_valid_mask = np.asarray(f["valid_mask"], dtype=bool) if "valid_mask" in f else None

    part_centers = []
    for pid in np.unique(boundary_part_ids):
        mask = boundary_part_ids == pid
        c = boundary_edges[mask].mean(axis=(0, 1))
        part_centers.append((int(pid), c))

    xx, yy = np.meshgrid(x, y, indexing="ij")
    valid_mask = geom_valid_mask if field_valid_mask is None else (geom_valid_mask & field_valid_mask)
    valid_mask_status = _valid_mask_status_grid(valid_mask, x, y)
    mixed_stencil_mask = valid_mask_status == int(VALID_MASK_STATUS_MIXED_STENCIL)
    hard_invalid_mask = valid_mask_status == int(VALID_MASK_STATUS_HARD_INVALID)
    speed = np.where(valid_mask, np.sqrt(ux * ux + uy * uy), np.nan)
    out = ensure_visualization_dirs(output_dir)["boundary_diagnostics"]
    out.mkdir(parents=True, exist_ok=True)
    medium_summary = domain_part_medium_summary(
        mesh_vertices,
        mesh_triangles,
        mesh_triangle_part_ids,
        mesh_quads,
        mesh_quad_part_ids,
        x,
        y,
        valid_mask,
    )
    if not medium_summary.empty:
        medium_summary.to_csv(out / "domain_part_medium_summary.csv", index=False)

    def draw_parts(
        ax: plt.Axes,
        *,
        alpha: float,
        linewidth: float,
        edgecolor: str = "#ffffff",
        label_part_ids: bool = False,
        show_legend: bool = False,
    ) -> None:
        draw_domain_parts_by_medium(
            ax,
            mesh_vertices,
            mesh_triangles,
            mesh_triangle_part_ids,
            mesh_quads,
            mesh_quad_part_ids,
            medium_summary=medium_summary,
            alpha=alpha,
            linewidth=linewidth,
            edgecolor=edgecolor,
            label_part_ids=label_part_ids,
            show_legend=show_legend,
        )
    final_csv = output_dir / "final_particles.csv"
    invalid_stop_points = np.zeros((0, 2), dtype=np.float64)
    if final_csv.exists():
        final_df = pd.read_csv(final_csv)
        required_cols = {"x", "y", "invalid_mask_stopped"}
        if required_cols.issubset(final_df.columns):
            invalid_stop_points = final_df.loc[
                final_df["invalid_mask_stopped"].astype(bool),
                ["x", "y"],
            ].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_parts(ax, alpha=0.50, linewidth=0.04, label_part_ids=True, show_legend=True)
    draw_boundary_edges(ax, boundary_edges, boundary_part_ids, linewidth=1.35, alpha=0.95, label_part_ids=True)
    for pid, center in part_centers:
        ax.text(float(center[0]), float(center[1]), str(pid), fontsize=8, ha="center", va="center", color="black")
    ax.set_title("Recognized Boundary Geometry (edge parts)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    fig.tight_layout()
    fig.savefig(out / "01_recognized_boundary_geometry.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_parts(ax, alpha=0.24, linewidth=0.02)
    pcm = ax.pcolormesh(xx, yy, valid_mask.astype(float), shading="nearest", cmap="Blues", vmin=0.0, vmax=1.0, alpha=0.82)
    draw_boundary_edges(ax, boundary_edges, None, linewidth=1.0, alpha=0.95)
    ax.set_title("Recognized Domain Mask (inside/outside)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("inside mask")
    fig.tight_layout()
    fig.savefig(out / "02_recognized_domain_mask.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_parts(ax, alpha=0.24, linewidth=0.02)
    lim = max(abs(float(np.nanmin(sdf))), abs(float(np.nanmax(sdf))))
    pcm = ax.pcolormesh(xx, yy, sdf, shading="nearest", cmap="coolwarm", vmin=-lim, vmax=lim, alpha=0.86)
    draw_parts(ax, alpha=0.10, linewidth=0.04, edgecolor="#222222")
    ax.contour(xx, yy, sdf, levels=[0.0], colors="black", linewidths=1.1)
    draw_boundary_edges(ax, boundary_edges, None, linewidth=0.8, alpha=0.85)
    ax.set_title("Diagnostic Signed Distance Field")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("sdf [m]")
    fig.tight_layout()
    fig.savefig(out / "03_signed_distance_field.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_parts(ax, alpha=0.30, linewidth=0.02)
    draw_boundary_edges(ax, boundary_edges, None, linewidth=1.0, alpha=0.95)
    band = valid_mask & (np.abs(sdf) <= float(normal_band_m))
    bx = xx[band]
    by = yy[band]
    bnx = nx[band]
    bny = ny[band]
    if bx.size:
        step = max(1, int(np.ceil(bx.size / 500)))
        ax.quiver(
            bx[::step],
            by[::step],
            bnx[::step],
            bny[::step],
            angles="xy",
            scale_units="xy",
            scale=250.0,
            width=0.0018,
            color="#b22222",
            alpha=0.75,
        )
    ax.set_title("Boundary Normals sampled near the Wall")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    fig.tight_layout()
    fig.savefig(out / "04_boundary_normals_near_wall.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_parts(ax, alpha=0.20, linewidth=0.02)
    pcm = ax.pcolormesh(xx, yy, np.ma.masked_invalid(speed), shading="nearest", cmap="viridis", alpha=0.86)
    draw_boundary_edges(ax, boundary_edges, None, linewidth=0.9, alpha=0.9)
    sx = slice(None, None, max(1, int(quiver_stride)))
    sy = slice(None, None, max(1, int(quiver_stride)))
    qmask = valid_mask[sx, sy]
    ax.quiver(
        xx[sx, sy][qmask],
        yy[sx, sy][qmask],
        ux[sx, sy][qmask],
        uy[sx, sy][qmask],
        angles="xy",
        scale_units="xy",
        scale=20.0,
        width=0.0018,
        color="black",
        alpha=0.35,
    )
    ax.set_title("Flow Speed / Vectors over Recognized Geometry")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label="speed [m/s]")
    fig.tight_layout()
    fig.savefig(out / "05_flow_speed_vectors_over_geometry.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_parts(ax, alpha=0.24, linewidth=0.02)
    pcm = ax.pcolormesh(xx, yy, mixed_stencil_mask.astype(float), shading="nearest", cmap="OrRd", vmin=0.0, vmax=1.0, alpha=0.82)
    draw_parts(ax, alpha=0.10, linewidth=0.04, edgecolor="#222222")
    draw_boundary_edges(ax, boundary_edges, boundary_part_ids, linewidth=1.0, alpha=0.9)
    ax.contour(xx, yy, sdf, levels=[0.0], colors="black", linewidths=0.85)
    ax.set_title("Mixed-Stencil Hotspots (point valid, stencil mixed)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("mixed stencil mask")
    fig.tight_layout()
    fig.savefig(out / "06_mixed_stencil_hotspots.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    draw_parts(ax, alpha=0.24, linewidth=0.02)
    pcm = ax.pcolormesh(xx, yy, hard_invalid_mask.astype(float), shading="nearest", cmap="Reds", vmin=0.0, vmax=1.0, alpha=0.82)
    draw_parts(ax, alpha=0.10, linewidth=0.04, edgecolor="#222222")
    draw_boundary_edges(ax, boundary_edges, boundary_part_ids, linewidth=1.0, alpha=0.9)
    if invalid_stop_points.size:
        ax.scatter(
            invalid_stop_points[:, 0],
            invalid_stop_points[:, 1],
            s=10,
            c="#4c1d95",
            alpha=0.8,
            label=f"invalid_mask_stopped ({int(invalid_stop_points.shape[0])})",
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Hard-Invalid Region and Stop Hotspots")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("hard invalid mask")
    fig.tight_layout()
    fig.savefig(out / "07_hard_invalid_stop_hotspots.png", dpi=170)
    plt.close(fig)

    report = {
        "case_dir": str(case_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "boundary_edge_count": int(boundary_edges.shape[0]),
        "boundary_part_ids": [int(v) for v in np.unique(boundary_part_ids)],
        "domain_grid_shape": [int(valid_mask.shape[0]), int(valid_mask.shape[1])],
        "mixed_stencil_grid_count": int(np.count_nonzero(mixed_stencil_mask)),
        "hard_invalid_grid_count": int(np.count_nonzero(hard_invalid_mask)),
        "invalid_mask_stopped_point_count": int(invalid_stop_points.shape[0]),
        "files": sorted(p.name for p in out.glob("*.png")) + sorted(p.name for p in out.glob("*.csv")),
    }
    (out / "boundary_diagnostics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export boundary-recognition diagnostic visuals.")
    ap.add_argument("--case-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--normal-band-m", type=float, default=2.5e-3)
    ap.add_argument("--quiver-stride", type=int, default=10)
    args = ap.parse_args()
    from tools.export_visualizations import export_visualizations

    index_path = export_visualizations(
        output_dir=args.output_dir.resolve(),
        case_dir=args.case_dir.resolve(),
        modules=("boundary",),
        boundary_normal_band_m=float(args.normal_band_m),
        boundary_quiver_stride=max(1, int(args.quiver_stride)),
    )
    print(f"wrote boundary diagnostics via unified pipeline: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
