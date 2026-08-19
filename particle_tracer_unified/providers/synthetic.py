from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
    axisymmetric_rz_geometry_report,
)
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GeometryND,
    GeometryProviderND,
    QuantitySeriesND,
    RegularFieldND,
)
from particle_tracer_unified.core.geometry2d import (
    build_boundary_loops_2d,
    validate_boundary_edges_2d,
)
from particle_tracer_unified.core.geometry3d import validate_closed_surface_triangles


@dataclass(frozen=True)
class _BoxSpec:
    bounds: tuple[float, ...]
    grid_shape: tuple[int, ...]


def _required_synthetic_kind(
    cfg: Mapping[str, Any], section: str, expected: str
) -> str:
    if "kind" not in cfg:
        raise ValueError(f"providers.{section}.kind is required")
    kind = cfg["kind"]
    if not isinstance(kind, str):
        raise ValueError(f"providers.{section}.kind must be a string")
    if kind != kind.strip():
        raise ValueError(
            f"providers.{section}.kind must not contain leading or trailing whitespace"
        )
    if kind != expected:
        raise ValueError(f"Unsupported synthetic {section} kind: {kind}")
    return kind


def _box_spec(cfg: Mapping[str, Any], spatial_dim: int) -> _BoxSpec:
    default_bounds = (
        [-1.0, 1.0, -1.0, 1.0]
        if spatial_dim == 2
        else [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    )
    default_shape = [81, 81] if spatial_dim == 2 else [41, 41, 41]
    raw_bounds = cfg.get("bounds")
    bounds = default_bounds if raw_bounds is None else raw_bounds
    return _BoxSpec(
        bounds=tuple(float(value) for value in bounds),
        grid_shape=tuple(int(value) for value in cfg.get("grid_shape", default_shape)),
    )


def _box_axes(spec: _BoxSpec) -> tuple[np.ndarray, ...]:
    return tuple(
        np.linspace(spec.bounds[2 * index], spec.bounds[2 * index + 1], count)
        for index, count in enumerate(spec.grid_shape)
    )


def _box_signed_distance_and_normal(
    axes: tuple[np.ndarray, ...], bounds: Sequence[float]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    grids = np.meshgrid(*axes, indexing="ij")
    if len(axes) == 2:
        return _box_signed_distance_and_normal_2d(grids, axes, bounds)
    return _box_signed_distance_and_normal_3d(grids, axes, bounds)


def _box_signed_distance_and_normal_2d(
    grids: tuple[np.ndarray, ...],
    axes: tuple[np.ndarray, ...],
    bounds: Sequence[float],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    x, y = grids
    xmin, xmax, ymin, ymax = [float(value) for value in bounds]
    wall_distances = np.stack([x - xmin, xmax - x, y - ymin, ymax - y], axis=0)
    outside_dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
    outside_dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
    outside_distance = np.sqrt(outside_dx**2 + outside_dy**2)
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    sdf = np.where(inside, -np.min(wall_distances, axis=0), outside_distance)
    gradient = np.gradient(sdf, axes[0], axes[1], edge_order=1)
    return sdf, tuple(gradient)


def _box_signed_distance_and_normal_3d(
    grids: tuple[np.ndarray, ...],
    axes: tuple[np.ndarray, ...],
    bounds: Sequence[float],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    x, y, z = grids
    xmin, xmax, ymin, ymax, zmin, zmax = [float(value) for value in bounds]
    wall_distances = np.stack(
        [
            x - xmin,
            xmax - x,
            y - ymin,
            ymax - y,
            z - zmin,
            zmax - z,
        ],
        axis=0,
    )
    outside_dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
    outside_dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
    outside_dz = np.maximum(np.maximum(zmin - z, 0.0), z - zmax)
    outside_distance = np.sqrt(outside_dx**2 + outside_dy**2 + outside_dz**2)
    inside = (
        (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
        & (z >= zmin)
        & (z <= zmax)
    )
    sdf = np.where(inside, -np.min(wall_distances, axis=0), outside_distance)
    gradient = np.gradient(sdf, axes[0], axes[1], axes[2], edge_order=1)
    return sdf, tuple(gradient)


def _box_boundary_edges(bounds: tuple[float, ...]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    return np.asarray(
        [
            [[xmin, ymin], [xmax, ymin]],
            [[xmax, ymin], [xmax, ymax]],
            [[xmax, ymax], [xmin, ymax]],
            [[xmin, ymax], [xmin, ymin]],
        ],
        dtype=np.float64,
    )


def _box_surface_triangles(bounds: tuple[float, ...]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = np.asarray(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    triangle_vertices = (
        (0, 2, 1),
        (0, 3, 2),  # z = zmin, outward -z
        (4, 5, 6),
        (4, 6, 7),  # z = zmax, outward +z
        (0, 1, 5),
        (0, 5, 4),  # y = ymin, outward -y
        (1, 2, 6),
        (1, 6, 5),  # x = xmax, outward +x
        (3, 6, 2),
        (3, 7, 6),  # y = ymax, outward +y
        (0, 7, 3),
        (0, 4, 7),  # x = xmin, outward -x
    )
    return np.asarray(
        [[corners[a], corners[b], corners[c]] for a, b, c in triangle_vertices],
        dtype=np.float64,
    )


def _surface_part_ids(cfg: Mapping[str, Any], triangle_count: int) -> np.ndarray:
    part_ids = np.asarray(
        cfg.get("boundary_part_ids", [1] * triangle_count), dtype=np.int32
    )
    if part_ids.size == triangle_count:
        return part_ids
    return np.full(
        triangle_count,
        int(part_ids[0]) if part_ids.size else 1,
        dtype=np.int32,
    )


def _build_synthetic_geometry_2d(
    cfg: Mapping[str, Any],
    spec: _BoxSpec,
    axes: tuple[np.ndarray, ...],
    sdf: np.ndarray,
    normals: tuple[np.ndarray, ...],
    coordinate_system: str,
) -> GeometryND:
    boundary_edges = _box_boundary_edges(spec.bounds)
    boundary_part_ids = np.asarray(
        cfg.get("boundary_part_ids", [1, 1, 1, 1]), dtype=np.int32
    )
    metadata: dict[str, Any] = {
        "bounds": list(spec.bounds),
        "boundary_edge_topology": validate_boundary_edges_2d(boundary_edges),
        "boundary_loop_count_2d": 1,
    }
    axisymmetric_report = axisymmetric_rz_geometry_report(
        coordinate_system=coordinate_system,
        spatial_dim=2,
        axes=tuple(axis.tolist() for axis in axes),
        boundary_edges=boundary_edges,
        boundary_edge_part_ids=boundary_part_ids,
    )
    if axisymmetric_report:
        metadata["axisymmetric_rz"] = axisymmetric_report
    return GeometryND(
        spatial_dim=2,
        coordinate_system=coordinate_system,
        axes=axes,
        valid_mask=np.ones(spec.grid_shape, dtype=bool),
        sdf=sdf,
        normal_components=normals,
        nearest_boundary_part_id_map=np.ones(spec.grid_shape, dtype=np.int32),
        source_kind="synthetic_box",
        metadata=metadata,
        boundary_edges=boundary_edges,
        boundary_edge_part_ids=boundary_part_ids,
        boundary_loops_2d=build_boundary_loops_2d(boundary_edges),
    )


def _build_synthetic_geometry_3d(
    cfg: Mapping[str, Any],
    spec: _BoxSpec,
    axes: tuple[np.ndarray, ...],
    sdf: np.ndarray,
    normals: tuple[np.ndarray, ...],
    coordinate_system: str,
) -> GeometryND:
    boundary_triangles = _box_surface_triangles(spec.bounds)
    part_ids = _surface_part_ids(cfg, int(boundary_triangles.shape[0]))
    return GeometryND(
        spatial_dim=3,
        coordinate_system=coordinate_system,
        axes=axes,
        valid_mask=np.ones(spec.grid_shape, dtype=bool),
        sdf=sdf,
        normal_components=normals,
        nearest_boundary_part_id_map=np.ones(spec.grid_shape, dtype=np.int32),
        source_kind="synthetic_box",
        metadata={
            "bounds": list(spec.bounds),
            "boundary_surface_validation": validate_closed_surface_triangles(
                boundary_triangles
            ),
        },
        boundary_triangles=boundary_triangles,
        boundary_triangle_part_ids=part_ids,
    )


def build_synthetic_geometry(
    cfg: Mapping[str, Any], spatial_dim: int, coordinate_system: str
) -> GeometryProviderND:
    _required_synthetic_kind(cfg, "geometry", "box")
    spec = _box_spec(cfg, spatial_dim)
    axes = _box_axes(spec)
    sdf, normals = _box_signed_distance_and_normal(axes, spec.bounds)
    geometry = (
        _build_synthetic_geometry_2d(cfg, spec, axes, sdf, normals, coordinate_system)
        if spatial_dim == 2
        else _build_synthetic_geometry_3d(
            cfg, spec, axes, sdf, normals, coordinate_system
        )
    )
    return GeometryProviderND(geometry=geometry, kind="synthetic_box")


def _field_time_contract(cfg: Mapping[str, Any]) -> tuple[str, np.ndarray]:
    time_mode = cfg.get("time_mode", "steady")
    if not isinstance(time_mode, str):
        raise ValueError("providers.field.time_mode must be a string")
    if time_mode != time_mode.strip():
        raise ValueError(
            "providers.field.time_mode must not contain leading or trailing whitespace"
        )
    if time_mode not in {"steady", "transient"}:
        raise ValueError("providers.field.time_mode must be steady or transient")
    times = _field_times(cfg)
    if time_mode == "steady" and times.size != 1:
        raise ValueError(
            "providers.field.time_mode steady requires exactly one time value"
        )
    if time_mode == "transient" and times.size < 2:
        raise ValueError(
            "providers.field.time_mode transient requires at least two time values"
        )
    return time_mode, times


def _field_times(cfg: Mapping[str, Any]) -> np.ndarray:
    times = np.asarray(cfg.get("times", [0.0]), dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("providers.field.times must be a non-empty 1D array")
    if not np.all(np.isfinite(times)):
        raise ValueError("providers.field.times must contain only finite values")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("providers.field.times must be strictly increasing")
    return times


def _velocity_at_times(base: np.ndarray, times: np.ndarray) -> np.ndarray:
    if times.size == 1:
        return base
    return np.stack(
        [
            (1.0 + 0.2 * np.sin(2 * np.pi * time / max(times[-1], 1.0))) * base
            for time in times
        ],
        axis=0,
    )


def _constant_at_times(base: np.ndarray, times: np.ndarray) -> np.ndarray:
    if times.size == 1:
        return base
    return np.stack([base for _ in times], axis=0)


def _quantity(
    name: str, unit: str, times: np.ndarray, data: np.ndarray
) -> QuantitySeriesND:
    return QuantitySeriesND(
        name,
        unit,
        times=times,
        data=np.asarray(data),
        metadata={},
    )


def _synthetic_quantities(
    cfg: Mapping[str, Any],
    spatial_dim: int,
    axes: tuple[np.ndarray, ...],
    times: np.ndarray,
) -> dict[str, QuantitySeriesND]:
    grids = np.meshgrid(*axes, indexing="ij")
    ux = float(cfg.get("shear_rate", 5.0)) * grids[1]
    zero_velocity = np.zeros_like(ux)
    quantities = {
        "ux": _quantity("ux", "m/s", times, _velocity_at_times(ux, times)),
        "uy": _quantity("uy", "m/s", times, _constant_at_times(zero_velocity, times)),
    }
    if spatial_dim != 2:
        quantities["uz"] = _quantity(
            "uz", "m/s", times, _constant_at_times(zero_velocity, times)
        )
    viscosity = np.full_like(
        grids[0],
        float(cfg.get("dynamic_viscosity_Pas", 1.8e-5)),
        dtype=np.float64,
    )
    quantities["mu"] = _quantity(
        "mu", "Pa*s", times, _constant_at_times(viscosity, times)
    )
    return quantities


def build_synthetic_field(
    cfg: Mapping[str, Any],
    spatial_dim: int,
    coordinate_system: str,
    axes: tuple[np.ndarray, ...],
) -> FieldProviderND:
    kind = _required_synthetic_kind(cfg, "field", "linear_shear")
    time_mode, times = _field_time_contract(cfg)
    field = RegularFieldND(
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
        axis_names=axis_names_for_coordinate_system(coordinate_system, spatial_dim),
        axes=axes,
        quantities=_synthetic_quantities(cfg, spatial_dim, axes, times),
        valid_mask=np.ones(tuple(len(axis) for axis in axes), dtype=bool),
        time_mode=time_mode,
        metadata={"synthetic_kind": kind},
    )
    return FieldProviderND(field=field, kind="synthetic_field")
