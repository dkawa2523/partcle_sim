from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import DTypeLike

from tools.state_contract import STATE_ORDER, classify_particle_states
from tools.visualization_common import (
    STATE_COLORS,
    domain_part_medium_summary,
    draw_boundary_edges,
    draw_domain_parts_by_medium,
    sample_grid_points,
)
from tools.visualization_data import filter_display_boundary_geometry
from tools.visualization_reports import ensure_visualization_dirs

_REPORT_FILES = (
    "domain_part_medium_summary.csv",
    "mechanics_distribution_on_geometry.csv",
    "final_state_by_nearest_boundary_part.csv",
    "geometry_layout_part_ids.png",
    "mechanics_maps_with_geometry.png",
    "mechanics_component_maps_with_geometry.png",
    "trajectories_geometry_flow_overlay.png",
    "final_states_over_geometry.png",
)


@dataclass(frozen=True)
class _InputPaths:
    geometry: Path
    field: Path
    trajectory: Path
    final_particles: Path


@dataclass(frozen=True)
class _Geometry:
    x: np.ndarray
    y: np.ndarray
    sdf: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    nearest_boundary_part_id: np.ndarray
    boundary_edges: np.ndarray | None
    boundary_part_ids: np.ndarray | None
    valid_mask: np.ndarray | None
    mesh_vertices: np.ndarray | None
    mesh_triangles: np.ndarray | None
    mesh_triangle_part_ids: np.ndarray | None
    mesh_quads: np.ndarray | None
    mesh_quad_part_ids: np.ndarray | None


@dataclass(frozen=True)
class _Fields:
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    viscosity: np.ndarray | None
    electric_x: np.ndarray | None
    electric_y: np.ndarray | None
    scalars: dict[str, np.ndarray]
    valid_mask: np.ndarray | None


@dataclass(frozen=True)
class _Grid:
    x: np.ndarray
    y: np.ndarray
    inside: np.ndarray
    speed: np.ndarray
    electric_magnitude: np.ndarray
    viscosity: np.ndarray
    electric_x: np.ndarray
    electric_y: np.ndarray
    nearest_boundary_part_id: np.ndarray


@dataclass(frozen=True)
class _FinalStates:
    particles: pd.DataFrame
    labels: np.ndarray
    summary_rows: list[dict[str, object]]


def _as_2d_quantity(payload: np.lib.npyio.NpzFile, name: str) -> np.ndarray | None:
    if name not in payload:
        return None
    arr = np.asarray(payload[name], dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    return None


def _optional_array(
    payload: np.lib.npyio.NpzFile,
    name: str,
    dtype: DTypeLike,
) -> np.ndarray | None:
    if name not in payload:
        return None
    return np.asarray(payload[name], dtype=dtype)


def _masked(arr: np.ndarray | None, inside: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.full(inside.shape, np.nan, dtype=np.float64)
    return np.where(inside & np.isfinite(arr), arr, np.nan)


def _input_paths(case_dir: Path, output_dir: Path) -> _InputPaths:
    paths = _InputPaths(
        geometry=case_dir / "generated" / "comsol_geometry_2d.npz",
        field=case_dir / "generated" / "comsol_field_2d.npz",
        trajectory=output_dir / "trajectory.npy",
        final_particles=output_dir / "final_particles.csv",
    )
    required = (
        ("Geometry npz", paths.geometry),
        ("Field npz", paths.field),
        ("trajectory.npy", paths.trajectory),
        ("final_particles.csv", paths.final_particles),
    )
    for label, path in required:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    return paths


def _load_geometry(path: Path) -> _Geometry:
    with np.load(path) as payload:
        x = np.asarray(payload["axis_0"], dtype=np.float64)
        y = np.asarray(payload["axis_1"], dtype=np.float64)
        sdf = np.asarray(payload["sdf"], dtype=np.float64)
        normal_x = _optional_array(payload, "normal_0", np.float64)
        normal_y = _optional_array(payload, "normal_1", np.float64)
        nearest = _optional_array(payload, "nearest_boundary_part_id_map", np.int32)
        if nearest is None:
            nearest = _optional_array(payload, "part_id_map", np.int32)
        boundary_edges = _optional_array(payload, "boundary_edges", np.float64)
        boundary_part_ids = _optional_array(payload, "boundary_edge_part_ids", np.int32)
        valid_mask = _optional_array(payload, "valid_mask", bool)
        mesh_vertices = _optional_array(payload, "mesh_vertices", np.float64)
        mesh_triangles = _optional_array(payload, "mesh_triangles", np.int32)
        mesh_triangle_part_ids = _optional_array(
            payload, "mesh_triangle_part_ids", np.int32
        )
        mesh_quads = _optional_array(payload, "mesh_quads", np.int32)
        mesh_quad_part_ids = _optional_array(payload, "mesh_quad_part_ids", np.int32)
    boundary_edges, boundary_part_ids = filter_display_boundary_geometry(
        boundary_edges, boundary_part_ids
    )
    return _Geometry(
        x=x,
        y=y,
        sdf=sdf,
        normal_x=np.zeros_like(sdf) if normal_x is None else normal_x,
        normal_y=np.zeros_like(sdf) if normal_y is None else normal_y,
        nearest_boundary_part_id=(
            np.ones_like(sdf, dtype=np.int32) if nearest is None else nearest
        ),
        boundary_edges=boundary_edges,
        boundary_part_ids=boundary_part_ids,
        valid_mask=valid_mask,
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        mesh_triangle_part_ids=mesh_triangle_part_ids,
        mesh_quads=mesh_quads,
        mesh_quad_part_ids=mesh_quad_part_ids,
    )


def _load_fields(path: Path) -> _Fields:
    with np.load(path) as payload:
        velocity_x = _as_2d_quantity(payload, "ux")
        velocity_y = _as_2d_quantity(payload, "uy")
        viscosity = _as_2d_quantity(payload, "mu")
        electric_x = _as_2d_quantity(payload, "E_x")
        electric_y = _as_2d_quantity(payload, "E_y")
        scalars = {}
        for name in ("T", "p", "rho_g", "phi", "ne", "Te"):
            value = _as_2d_quantity(payload, name)
            if value is not None:
                scalars[name] = value
        valid_mask = _optional_array(payload, "valid_mask", bool)
    if velocity_x is None or velocity_y is None:
        raise ValueError("mechanics visuals require ux and uy in the field bundle")
    return _Fields(
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        viscosity=viscosity,
        electric_x=electric_x,
        electric_y=electric_y,
        scalars=scalars,
        valid_mask=valid_mask,
    )


def _matching_mask(
    candidate: np.ndarray | None,
    shape: tuple[int, ...],
    default: np.ndarray,
) -> np.ndarray:
    if candidate is None or candidate.shape != shape:
        return default
    return candidate


def _prepare_grid(geometry: _Geometry, fields: _Fields) -> _Grid:
    geometry_mask = _matching_mask(
        geometry.valid_mask,
        geometry.sdf.shape,
        geometry.sdf <= 0.0,
    )
    field_mask = _matching_mask(
        fields.valid_mask,
        geometry.sdf.shape,
        np.ones_like(geometry_mask, dtype=bool),
    )
    inside = geometry_mask & field_mask
    speed = np.sqrt(fields.velocity_x**2 + fields.velocity_y**2)
    electric_magnitude = None
    if fields.electric_x is not None and fields.electric_y is not None:
        electric_magnitude = np.sqrt(fields.electric_x**2 + fields.electric_y**2)
    x, y = np.meshgrid(geometry.x, geometry.y, indexing="ij")
    return _Grid(
        x=x,
        y=y,
        inside=inside,
        speed=np.where(inside, speed, np.nan),
        electric_magnitude=_masked(electric_magnitude, inside),
        viscosity=_masked(fields.viscosity, inside),
        electric_x=_masked(fields.electric_x, inside),
        electric_y=_masked(fields.electric_y, inside),
        nearest_boundary_part_id=np.where(inside, geometry.nearest_boundary_part_id, 0),
    )


def _medium_summary(geometry: _Geometry, fields: _Fields) -> pd.DataFrame:
    return domain_part_medium_summary(
        geometry.mesh_vertices,
        geometry.mesh_triangles,
        geometry.mesh_triangle_part_ids,
        geometry.mesh_quads,
        geometry.mesh_quad_part_ids,
        geometry.x,
        geometry.y,
        fields.valid_mask,
    )


def _draw_medium(
    ax: Axes,
    geometry: _Geometry,
    medium_summary: pd.DataFrame,
    *,
    alpha: float,
) -> None:
    draw_domain_parts_by_medium(
        ax,
        geometry.mesh_vertices,
        geometry.mesh_triangles,
        geometry.mesh_triangle_part_ids,
        geometry.mesh_quads,
        geometry.mesh_quad_part_ids,
        medium_summary=medium_summary,
        alpha=alpha,
    )


def _set_spatial_axes(ax: Axes, geometry: _Geometry) -> None:
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(geometry.x.min()), float(geometry.x.max()))
    ax.set_ylim(float(geometry.y.min()), float(geometry.y.max()))


def _draw_boundary_or_sdf(
    ax: Axes,
    geometry: _Geometry,
    grid: _Grid,
    *,
    linewidth: float,
    alpha: float = 0.95,
    color: str = "k",
    sdf_linewidth: float | None = None,
) -> None:
    if geometry.boundary_edges is not None:
        draw_boundary_edges(
            ax,
            geometry.boundary_edges,
            None,
            linewidth=linewidth,
            alpha=alpha,
        )
        return
    ax.contour(
        grid.x,
        grid.y,
        geometry.sdf,
        levels=[0.0],
        colors=color,
        linewidths=linewidth if sdf_linewidth is None else sdf_linewidth,
    )


def _write_grid_distribution(
    output_dir: Path,
    geometry: _Geometry,
    fields: _Fields,
    grid: _Grid,
) -> None:
    distribution = pd.DataFrame(
        {
            "x": grid.x.ravel(),
            "y": grid.y.ravel(),
            "inside": grid.inside.astype(np.int32).ravel(),
            "sdf_m": geometry.sdf.ravel(),
            "distance_to_wall_m": np.abs(geometry.sdf).ravel(),
            "normal_x": geometry.normal_x.ravel(),
            "normal_y": geometry.normal_y.ravel(),
            "nearest_boundary_part_id": grid.nearest_boundary_part_id.ravel(),
            "ux_mps": np.where(grid.inside, fields.velocity_x, np.nan).ravel(),
            "uy_mps": np.where(grid.inside, fields.velocity_y, np.nan).ravel(),
            "speed_mps": grid.speed.ravel(),
            "mu_Pas": grid.viscosity.ravel(),
            "electric_field_Vpm": grid.electric_magnitude.ravel(),
        }
    )
    distribution.loc[distribution["inside"] == 1].reset_index(drop=True).to_csv(
        output_dir / "mechanics_distribution_on_geometry.csv",
        index=False,
    )


def _plot_geometry_layout(
    output_dir: Path,
    geometry: _Geometry,
    grid: _Grid,
    medium_summary: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.8))
    draw_domain_parts_by_medium(
        ax,
        geometry.mesh_vertices,
        geometry.mesh_triangles,
        geometry.mesh_triangle_part_ids,
        geometry.mesh_quads,
        geometry.mesh_quad_part_ids,
        medium_summary=medium_summary,
        alpha=0.48,
        label_part_ids=True,
        show_legend=True,
    )
    if geometry.boundary_edges is not None and geometry.boundary_part_ids is not None:
        draw_boundary_edges(
            ax,
            geometry.boundary_edges,
            geometry.boundary_part_ids,
            linewidth=1.35,
            alpha=0.95,
            label_part_ids=True,
        )
    else:
        masked_part = np.ma.masked_where(
            ~grid.inside,
            geometry.nearest_boundary_part_id.astype(float),
        )
        color = ax.pcolormesh(
            grid.x,
            grid.y,
            masked_part,
            shading="nearest",
            cmap="Greys",
            alpha=0.75,
        )
        fig.colorbar(color, ax=ax, fraction=0.046, pad=0.02).set_label(
            "Nearest Boundary Part ID"
        )
        ax.contour(
            grid.x,
            grid.y,
            geometry.sdf,
            levels=[0.0],
            colors="k",
            linewidths=1.1,
        )
    ax.set_title("Boundary Part IDs over Geometry")
    _set_spatial_axes(ax, geometry)
    fig.tight_layout()
    fig.savefig(output_dir / "geometry_layout_part_ids.png", dpi=170)
    plt.close(fig)


def _scalar_plot_specs(fields: _Fields, grid: _Grid) -> list[tuple[str, np.ndarray]]:
    specs = [("Speed |u| [m/s]", grid.speed)]
    if fields.electric_x is not None and fields.electric_y is not None:
        specs.append(("Electric Field |E| [V/m]", grid.electric_magnitude))
    if fields.viscosity is not None:
        specs.append(("Dynamic Viscosity mu [Pa*s]", grid.viscosity))
    specs.extend(
        (name, _masked(values, grid.inside)) for name, values in fields.scalars.items()
    )
    return specs


def _subplot_grid(count: int) -> tuple[int, int]:
    columns = min(3, max(1, count))
    return int(np.ceil(count / columns)), columns


def _plot_scalar_maps(
    output_dir: Path,
    geometry: _Geometry,
    fields: _Fields,
    grid: _Grid,
    medium_summary: pd.DataFrame,
) -> None:
    specs = _scalar_plot_specs(fields, grid)
    rows, columns = _subplot_grid(len(specs))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.8 * columns, 4.9 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (title, values) in zip(axes.ravel(), specs, strict=False):
        _draw_medium(ax, geometry, medium_summary, alpha=0.22)
        color = ax.pcolormesh(
            grid.x,
            grid.y,
            np.ma.masked_where(~grid.inside, values),
            shading="nearest",
            cmap="viridis",
            alpha=0.86,
        )
        ax.set_title(title)
        fig.colorbar(color, ax=ax, fraction=0.046, pad=0.02)
        _draw_medium(
            ax,
            geometry,
            medium_summary,
            alpha=0.10,
        )
        _draw_boundary_or_sdf(
            ax,
            geometry,
            grid,
            linewidth=0.85,
            alpha=0.9,
            sdf_linewidth=0.8,
        )
        _set_spatial_axes(ax, geometry)
    for ax in axes.ravel()[len(specs) :]:
        ax.axis("off")
    fig.savefig(output_dir / "mechanics_maps_with_geometry.png", dpi=170)
    plt.close(fig)


def _component_plot_specs(
    fields: _Fields,
    grid: _Grid,
) -> list[tuple[str, np.ndarray]]:
    candidates = (
        ("ux [m/s]", np.where(grid.inside, fields.velocity_x, np.nan)),
        ("uy [m/s]", np.where(grid.inside, fields.velocity_y, np.nan)),
        ("E_x [V/m]", grid.electric_x),
        ("E_y [V/m]", grid.electric_y),
    )
    return [
        (title, values) for title, values in candidates if np.isfinite(values).any()
    ]


def _plot_component_maps(
    output_dir: Path,
    geometry: _Geometry,
    fields: _Fields,
    grid: _Grid,
    medium_summary: pd.DataFrame,
) -> None:
    specs = _component_plot_specs(fields, grid)
    if not specs:
        return
    rows, columns = _subplot_grid(len(specs))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.8 * columns, 4.9 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (title, values) in zip(axes.ravel(), specs, strict=False):
        _draw_medium(ax, geometry, medium_summary, alpha=0.22)
        color = ax.pcolormesh(
            grid.x,
            grid.y,
            np.ma.masked_invalid(values),
            shading="nearest",
            cmap="coolwarm",
            alpha=0.86,
        )
        _draw_medium(
            ax,
            geometry,
            medium_summary,
            alpha=0.10,
        )
        draw_boundary_edges(
            ax,
            geometry.boundary_edges,
            geometry.boundary_part_ids,
            linewidth=0.75,
            alpha=0.9,
        )
        ax.set_title(title)
        _set_spatial_axes(ax, geometry)
        fig.colorbar(color, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.ravel()[len(specs) :]:
        ax.axis("off")
    fig.savefig(output_dir / "mechanics_component_maps_with_geometry.png", dpi=170)
    plt.close(fig)


def _plot_trajectories(
    output_dir: Path,
    trajectory_path: Path,
    geometry: _Geometry,
    fields: _Fields,
    grid: _Grid,
    medium_summary: pd.DataFrame,
    sample_trajectories: int,
    quiver_stride: int,
) -> tuple[int, int]:
    positions = np.load(trajectory_path)
    _, particle_count, _ = positions.shape
    rng = np.random.default_rng(20260401)
    selected = rng.choice(
        particle_count,
        size=min(sample_trajectories, particle_count),
        replace=False,
    )
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    _draw_medium(ax, geometry, medium_summary, alpha=0.32)
    _draw_boundary_or_sdf(ax, geometry, grid, linewidth=1.0, color="#444")
    for particle_index in selected:
        trajectory = positions[:, particle_index, :]
        ax.plot(trajectory[:, 0], trajectory[:, 1], lw=0.7, alpha=0.7)
    stride = slice(None, None, max(1, int(quiver_stride)))
    quiver_mask = grid.inside[stride, stride]
    ax.quiver(
        grid.x[stride, stride][quiver_mask],
        grid.y[stride, stride][quiver_mask],
        fields.velocity_x[stride, stride][quiver_mask],
        fields.velocity_y[stride, stride][quiver_mask],
        angles="xy",
        scale_units="xy",
        scale=20.0,
        width=0.0018,
        color="black",
        alpha=0.35,
    )
    ax.set_title(
        "Trajectories + Geometry + Flow Vectors "
        f"(sample {len(selected)} / {particle_count})"
    )
    _set_spatial_axes(ax, geometry)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectories_geometry_flow_overlay.png", dpi=170)
    plt.close(fig)
    return particle_count, len(selected)


def _summarize_final_states(
    final_particles_path: Path,
    output_dir: Path,
    geometry: _Geometry,
) -> _FinalStates:
    particles = pd.read_csv(final_particles_path)
    labels = classify_particle_states(particles)
    rows: list[dict[str, object]] = []
    if {"x_m", "y_m"}.issubset(particles.columns):
        points = particles.loc[:, ["x_m", "y_m"]].to_numpy(dtype=np.float64)
        nearest_part_ids = np.rint(
            sample_grid_points(
                geometry.nearest_boundary_part_id,
                geometry.x,
                geometry.y,
                points,
            )
        ).astype(np.int32)
        summary = (
            pd.DataFrame(
                {
                    "nearest_boundary_part_id": nearest_part_ids,
                    "state": labels,
                }
            )
            .groupby(["nearest_boundary_part_id", "state"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["nearest_boundary_part_id", "state"])
        )
        summary.to_csv(
            output_dir / "final_state_by_nearest_boundary_part.csv",
            index=False,
        )
        part_ids = summary["nearest_boundary_part_id"].to_numpy(dtype=np.int32).tolist()
        states = summary["state"].astype(str).tolist()
        counts = summary["count"].to_numpy(dtype=np.int64).tolist()
        rows = [
            {
                "nearest_boundary_part_id": part_id,
                "state": state,
                "count": count,
            }
            for part_id, state, count in zip(part_ids, states, counts, strict=True)
        ]
    return _FinalStates(particles=particles, labels=labels, summary_rows=rows)


def _plot_final_states(
    output_dir: Path,
    geometry: _Geometry,
    grid: _Grid,
    medium_summary: pd.DataFrame,
    final_states: _FinalStates,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    _draw_medium(ax, geometry, medium_summary, alpha=0.30)
    _draw_boundary_or_sdf(ax, geometry, grid, linewidth=1.0)
    for state in STATE_ORDER:
        particles = final_states.particles.loc[final_states.labels == state]
        if particles.empty:
            continue
        ax.scatter(
            particles["x_m"],
            particles["y_m"],
            s=4,
            c=STATE_COLORS[state],
            label=f"{state} ({len(particles)})",
            alpha=0.7,
        )
    ax.set_title("Final Particle States over Geometry")
    ax.legend(loc="best", fontsize=8)
    _set_spatial_axes(ax, geometry)
    fig.tight_layout()
    fig.savefig(output_dir / "final_states_over_geometry.png", dpi=170)
    plt.close(fig)


def _write_report(
    case_dir: Path,
    output_dir: Path,
    mechanics_dir: Path,
    particle_count: int,
    sample_count: int,
    summary_rows: list[dict[str, object]],
) -> None:
    report = {
        "case_dir": str(case_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "mechanics_dir": str(mechanics_dir.resolve()),
        "n_particles": particle_count,
        "sample_trajectories": sample_count,
        "boundary_region_summary_status": (
            "computed_from_nearest_boundary_part_id_map"
        ),
        "final_state_by_nearest_boundary_part": summary_rows,
        "files": list(_REPORT_FILES),
    }
    (mechanics_dir / "visualization_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def export_mechanics_visuals(
    case_dir: Path,
    output_dir: Path,
    sample_trajectories: int = 500,
    quiver_stride: int = 12,
) -> Path:
    paths = _input_paths(case_dir, output_dir)
    geometry = _load_geometry(paths.geometry)
    fields = _load_fields(paths.field)
    grid = _prepare_grid(geometry, fields)
    mechanics_dir = ensure_visualization_dirs(output_dir)["mechanics"]
    mechanics_dir.mkdir(parents=True, exist_ok=True)

    medium_summary = _medium_summary(geometry, fields)
    if not medium_summary.empty:
        medium_summary.to_csv(
            mechanics_dir / "domain_part_medium_summary.csv",
            index=False,
        )
    _write_grid_distribution(mechanics_dir, geometry, fields, grid)
    _plot_geometry_layout(mechanics_dir, geometry, grid, medium_summary)
    _plot_scalar_maps(mechanics_dir, geometry, fields, grid, medium_summary)
    _plot_component_maps(mechanics_dir, geometry, fields, grid, medium_summary)
    particle_count, sample_count = _plot_trajectories(
        mechanics_dir,
        paths.trajectory,
        geometry,
        fields,
        grid,
        medium_summary,
        sample_trajectories,
        quiver_stride,
    )
    final_states = _summarize_final_states(
        paths.final_particles,
        mechanics_dir,
        geometry,
    )
    _plot_final_states(
        mechanics_dir,
        geometry,
        grid,
        medium_summary,
        final_states,
    )
    _write_report(
        case_dir,
        output_dir,
        mechanics_dir,
        particle_count,
        sample_count,
        final_states.summary_rows,
    )
    return mechanics_dir
