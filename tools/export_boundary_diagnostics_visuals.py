from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
    sample_valid_mask_status,
)
from tools.visualization_common import (
    domain_part_medium_summary,
    draw_boundary_edges,
    draw_domain_parts_by_medium,
)
from tools.visualization_data import (
    filter_display_boundary_geometry,
    require_2d_quantity,
)
from tools.visualization_reports import ensure_visualization_dirs


@dataclass(frozen=True)
class _Geometry:
    x: np.ndarray
    y: np.ndarray
    sdf: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    valid_mask: np.ndarray
    boundary_edges: np.ndarray
    boundary_part_ids: np.ndarray
    mesh_vertices: np.ndarray
    mesh_triangles: np.ndarray | None
    mesh_triangle_part_ids: np.ndarray | None
    mesh_quads: np.ndarray
    mesh_quad_part_ids: np.ndarray | None


@dataclass(frozen=True)
class _BoundaryData:
    geometry: _Geometry
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    valid_mask: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    mixed_stencil_mask: np.ndarray
    hard_invalid_mask: np.ndarray
    speed: np.ndarray
    medium_summary: pd.DataFrame
    invalid_stop_points: np.ndarray


def _optional_array(
    payload: np.lib.npyio.NpzFile, name: str, dtype: npt.DTypeLike
) -> np.ndarray | None:
    return np.asarray(payload[name], dtype=dtype) if name in payload else None


def _load_geometry(case_dir: Path) -> _Geometry:
    path = case_dir / "generated" / "comsol_geometry_2d.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Geometry npz not found: {path}")
    with np.load(path) as payload:
        geometry = _Geometry(
            x=np.asarray(payload["axis_0"], dtype=np.float64),
            y=np.asarray(payload["axis_1"], dtype=np.float64),
            sdf=np.asarray(payload["sdf"], dtype=np.float64),
            normal_x=np.asarray(payload["normal_0"], dtype=np.float64),
            normal_y=np.asarray(payload["normal_1"], dtype=np.float64),
            valid_mask=np.asarray(payload["valid_mask"], dtype=bool),
            boundary_edges=np.asarray(payload["boundary_edges"], dtype=np.float64),
            boundary_part_ids=np.asarray(
                payload["boundary_edge_part_ids"], dtype=np.int32
            ),
            mesh_vertices=np.asarray(payload["mesh_vertices"], dtype=np.float64),
            mesh_triangles=_optional_array(payload, "mesh_triangles", np.int32),
            mesh_triangle_part_ids=_optional_array(
                payload, "mesh_triangle_part_ids", np.int32
            ),
            mesh_quads=np.asarray(payload["mesh_quads"], dtype=np.int32),
            mesh_quad_part_ids=_optional_array(payload, "mesh_quad_part_ids", np.int32),
        )
    edges, part_ids = filter_display_boundary_geometry(
        geometry.boundary_edges, geometry.boundary_part_ids
    )
    if edges is None or part_ids is None:
        raise ValueError(
            "boundary diagnostics require displayable boundary edges with part IDs"
        )
    return replace(geometry, boundary_edges=edges, boundary_part_ids=part_ids)


def _load_fields(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = case_dir / "generated" / "comsol_field_2d.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Field npz not found: {path}")
    with np.load(path) as payload:
        return (
            require_2d_quantity(payload, "ux", "boundary diagnostics"),
            require_2d_quantity(payload, "uy", "boundary diagnostics"),
            np.asarray(payload["valid_mask"], dtype=bool),
        )


def _invalid_stop_points(output_dir: Path) -> np.ndarray:
    path = output_dir / "final_particles.csv"
    particles = pd.read_csv(path)
    return particles.loc[
        particles["final_state"].astype(str).eq("invalid_mask_stopped"),
        ["x_m", "y_m"],
    ].to_numpy(dtype=np.float64)


def _valid_mask_status_grid(
    valid_mask: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    axes = (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    status = np.zeros(np.asarray(valid_mask).shape, dtype=np.uint8)
    for ix, xv in enumerate(axes[0]):
        for iy, yv in enumerate(axes[1]):
            point = np.asarray([float(xv), float(yv)], dtype=np.float64)
            status[ix, iy] = np.uint8(sample_valid_mask_status(valid_mask, axes, point))
    return status


def _load_boundary_data(case_dir: Path, output_dir: Path) -> _BoundaryData:
    geometry = _load_geometry(case_dir)
    velocity_x, velocity_y, field_mask = _load_fields(case_dir)
    valid_mask = geometry.valid_mask & field_mask
    grid_x, grid_y = np.meshgrid(geometry.x, geometry.y, indexing="ij")
    mask_status = _valid_mask_status_grid(valid_mask, geometry.x, geometry.y)
    medium_summary = domain_part_medium_summary(
        geometry.mesh_vertices,
        geometry.mesh_triangles,
        geometry.mesh_triangle_part_ids,
        geometry.mesh_quads,
        geometry.mesh_quad_part_ids,
        geometry.x,
        geometry.y,
        valid_mask,
    )
    return _BoundaryData(
        geometry=geometry,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        valid_mask=valid_mask,
        grid_x=grid_x,
        grid_y=grid_y,
        mixed_stencil_mask=(mask_status == int(VALID_MASK_STATUS_MIXED_STENCIL)),
        hard_invalid_mask=(mask_status == int(VALID_MASK_STATUS_HARD_INVALID)),
        speed=np.where(valid_mask, np.hypot(velocity_x, velocity_y), np.nan),
        medium_summary=medium_summary,
        invalid_stop_points=_invalid_stop_points(output_dir),
    )


def _draw_parts(
    ax: plt.Axes,
    data: _BoundaryData,
    *,
    alpha: float,
    label_part_ids: bool = False,
    show_legend: bool = False,
) -> None:
    geometry = data.geometry
    draw_domain_parts_by_medium(
        ax,
        geometry.mesh_vertices,
        geometry.mesh_triangles,
        geometry.mesh_triangle_part_ids,
        geometry.mesh_quads,
        geometry.mesh_quad_part_ids,
        medium_summary=data.medium_summary,
        alpha=alpha,
        label_part_ids=label_part_ids,
        show_legend=show_legend,
    )


def _save_spatial_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    geometry: _Geometry,
    output_dir: Path,
    filename: str,
) -> None:
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(geometry.x.min()), float(geometry.x.max()))
    ax.set_ylim(float(geometry.y.min()), float(geometry.y.max()))
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=170)
    plt.close(fig)


def _plot_geometry(data: _BoundaryData, out: Path) -> None:
    geometry = data.geometry
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    _draw_parts(ax, data, alpha=0.50, label_part_ids=True, show_legend=True)
    draw_boundary_edges(
        ax,
        geometry.boundary_edges,
        geometry.boundary_part_ids,
        linewidth=1.35,
        alpha=0.95,
        label_part_ids=True,
    )
    ax.set_title("Recognized Boundary Geometry (edge parts)")
    _save_spatial_figure(fig, ax, geometry, out, "01_recognized_boundary_geometry.png")


def _plot_scalar(
    data: _BoundaryData,
    out: Path,
    values: np.ndarray,
    *,
    filename: str,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    contour_sdf: bool = False,
    stop_points: np.ndarray | None = None,
) -> None:
    geometry = data.geometry
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    _draw_parts(ax, data, alpha=0.24)
    colors = ax.pcolormesh(
        data.grid_x,
        data.grid_y,
        values,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.82,
    )
    draw_boundary_edges(ax, geometry.boundary_edges, linewidth=1.0, alpha=0.9)
    if contour_sdf:
        ax.contour(
            data.grid_x,
            data.grid_y,
            geometry.sdf,
            levels=[0.0],
            colors="black",
            linewidths=0.9,
        )
    if stop_points is not None and stop_points.size:
        ax.scatter(
            stop_points[:, 0],
            stop_points[:, 1],
            s=10,
            c="#4c1d95",
            alpha=0.8,
            label=f"invalid_mask_stopped ({int(stop_points.shape[0])})",
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    fig.colorbar(colors, ax=ax, fraction=0.046, pad=0.02, label=colorbar_label)
    _save_spatial_figure(fig, ax, geometry, out, filename)


def _plot_normals(data: _BoundaryData, out: Path, normal_band_m: float) -> None:
    geometry = data.geometry
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    _draw_parts(ax, data, alpha=0.30)
    draw_boundary_edges(ax, geometry.boundary_edges, linewidth=1.0, alpha=0.95)
    band = data.valid_mask & (np.abs(geometry.sdf) <= normal_band_m)
    x = data.grid_x[band]
    y = data.grid_y[band]
    if x.size:
        stride = max(1, int(np.ceil(x.size / 500)))
        ax.quiver(
            x[::stride],
            y[::stride],
            geometry.normal_x[band][::stride],
            geometry.normal_y[band][::stride],
            angles="xy",
            scale_units="xy",
            scale=250.0,
            width=0.0018,
            color="#b22222",
            alpha=0.75,
        )
    ax.set_title("Boundary Normals sampled near the Wall")
    _save_spatial_figure(fig, ax, geometry, out, "04_boundary_normals_near_wall.png")


def _plot_flow(data: _BoundaryData, out: Path, quiver_stride: int) -> None:
    geometry = data.geometry
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    _draw_parts(ax, data, alpha=0.20)
    colors = ax.pcolormesh(
        data.grid_x,
        data.grid_y,
        np.ma.masked_invalid(data.speed),
        shading="nearest",
        cmap="viridis",
        alpha=0.86,
    )
    draw_boundary_edges(ax, geometry.boundary_edges, linewidth=0.9, alpha=0.9)
    stride = slice(None, None, max(1, quiver_stride))
    quiver_mask = data.valid_mask[stride, stride]
    ax.quiver(
        data.grid_x[stride, stride][quiver_mask],
        data.grid_y[stride, stride][quiver_mask],
        data.velocity_x[stride, stride][quiver_mask],
        data.velocity_y[stride, stride][quiver_mask],
        angles="xy",
        scale_units="xy",
        scale=20.0,
        width=0.0018,
        color="black",
        alpha=0.35,
    )
    ax.set_title("Flow Speed / Vectors over Recognized Geometry")
    fig.colorbar(colors, ax=ax, fraction=0.046, pad=0.02, label="speed [m/s]")
    _save_spatial_figure(
        fig, ax, geometry, out, "05_flow_speed_vectors_over_geometry.png"
    )


def _write_report(
    data: _BoundaryData, case_dir: Path, output_dir: Path, out: Path
) -> None:
    geometry = data.geometry
    report = {
        "case_dir": str(case_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "boundary_edge_count": int(geometry.boundary_edges.shape[0]),
        "boundary_part_ids": [
            int(value) for value in np.unique(geometry.boundary_part_ids)
        ],
        "domain_grid_shape": [int(size) for size in data.valid_mask.shape],
        "mixed_stencil_grid_count": int(np.count_nonzero(data.mixed_stencil_mask)),
        "hard_invalid_grid_count": int(np.count_nonzero(data.hard_invalid_mask)),
        "invalid_mask_stopped_point_count": int(data.invalid_stop_points.shape[0]),
        "files": sorted(path.name for path in out.glob("*.png"))
        + sorted(path.name for path in out.glob("*.csv")),
    }
    (out / "boundary_diagnostics_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def export_boundary_diagnostics(
    case_dir: Path,
    output_dir: Path,
    normal_band_m: float = 2.5e-3,
    quiver_stride: int = 10,
) -> Path:
    data = _load_boundary_data(case_dir, output_dir)
    out = ensure_visualization_dirs(output_dir)["boundary_diagnostics"]
    out.mkdir(parents=True, exist_ok=True)
    data.medium_summary.to_csv(out / "domain_part_medium_summary.csv", index=False)

    _plot_geometry(data, out)
    _plot_scalar(
        data,
        out,
        data.valid_mask.astype(float),
        filename="02_recognized_domain_mask.png",
        title="Recognized Domain Mask (inside/outside)",
        colorbar_label="inside mask",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    sdf_limit = max(
        abs(float(np.nanmin(data.geometry.sdf))),
        abs(float(np.nanmax(data.geometry.sdf))),
    )
    _plot_scalar(
        data,
        out,
        data.geometry.sdf,
        filename="03_signed_distance_field.png",
        title="Diagnostic Signed Distance Field",
        colorbar_label="sdf [m]",
        cmap="coolwarm",
        vmin=-sdf_limit,
        vmax=sdf_limit,
        contour_sdf=True,
    )
    _plot_normals(data, out, normal_band_m)
    _plot_flow(data, out, quiver_stride)
    _plot_scalar(
        data,
        out,
        data.mixed_stencil_mask.astype(float),
        filename="06_mixed_stencil_hotspots.png",
        title="Mixed-Stencil Hotspots (point valid, stencil mixed)",
        colorbar_label="mixed stencil mask",
        cmap="OrRd",
        vmin=0.0,
        vmax=1.0,
        contour_sdf=True,
    )
    _plot_scalar(
        data,
        out,
        data.hard_invalid_mask.astype(float),
        filename="07_hard_invalid_stop_hotspots.png",
        title="Hard-Invalid Region and Stop Hotspots",
        colorbar_label="hard invalid mask",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        stop_points=data.invalid_stop_points,
    )
    _write_report(data, case_dir, output_dir, out)
    return out
