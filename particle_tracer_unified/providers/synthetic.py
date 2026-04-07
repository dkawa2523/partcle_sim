from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ..core.datamodel import FieldProviderND, GeometryND, GeometryProviderND, QuantitySeriesND, RegularFieldND
from ..core.geometry2d import build_boundary_loops_2d, validate_boundary_edges_2d
from ..core.geometry3d import validate_closed_surface_triangles


def _box_signed_distance_and_normal(axes: Tuple[np.ndarray, ...], bounds: Sequence[float]) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    dim = len(axes)
    grids = np.meshgrid(*axes, indexing='ij')
    if dim == 2:
        x, y = grids
        xmin, xmax, ymin, ymax = [float(v) for v in bounds]
        d_left = x - xmin
        d_right = xmax - x
        d_bottom = y - ymin
        d_top = ymax - y
        stack = np.stack([d_left, d_right, d_bottom, d_top], axis=0)
        min_inside = np.min(stack, axis=0)
        outside_dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
        outside_dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
        outside_dist = np.sqrt(outside_dx ** 2 + outside_dy ** 2)
        inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        sdf = np.where(inside, -min_inside, outside_dist)
        gx, gy = np.gradient(sdf, axes[0], axes[1], edge_order=1)
        return sdf, (gx, gy)
    x, y, z = grids
    xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
    d_left = x - xmin
    d_right = xmax - x
    d_bottom = y - ymin
    d_top = ymax - y
    d_back = z - zmin
    d_front = zmax - z
    stack = np.stack([d_left, d_right, d_bottom, d_top, d_back, d_front], axis=0)
    min_inside = np.min(stack, axis=0)
    outside_dx = np.maximum(np.maximum(xmin - x, 0.0), x - xmax)
    outside_dy = np.maximum(np.maximum(ymin - y, 0.0), y - ymax)
    outside_dz = np.maximum(np.maximum(zmin - z, 0.0), z - zmax)
    outside_dist = np.sqrt(outside_dx ** 2 + outside_dy ** 2 + outside_dz ** 2)
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (z <= zmax)
    sdf = np.where(inside, -min_inside, outside_dist)
    gx, gy, gz = np.gradient(sdf, axes[0], axes[1], axes[2], edge_order=1)
    return sdf, (gx, gy, gz)


def build_synthetic_geometry(cfg: Mapping[str, Any], spatial_dim: int, coordinate_system: str) -> GeometryProviderND:
    kind = str(cfg.get('kind', 'box')).strip()
    if kind != 'box':
        raise ValueError(f'Unsupported synthetic geometry kind: {kind}')
    bounds = cfg.get('bounds')
    if bounds is None:
        bounds = [-1.0, 1.0, -1.0, 1.0] if spatial_dim == 2 else [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    grid_shape = tuple(int(v) for v in cfg.get('grid_shape', [81, 81] if spatial_dim == 2 else [41, 41, 41]))
    if spatial_dim == 2:
        xmin, xmax, ymin, ymax = [float(v) for v in bounds]
        axes = (np.linspace(xmin, xmax, grid_shape[0]), np.linspace(ymin, ymax, grid_shape[1]))
        valid_mask = np.ones(grid_shape, dtype=bool)
        sdf, normals = _box_signed_distance_and_normal(axes, bounds)
        boundary_edges = np.array([
            [[xmin, ymin], [xmax, ymin]],
            [[xmax, ymin], [xmax, ymax]],
            [[xmax, ymax], [xmin, ymax]],
            [[xmin, ymax], [xmin, ymin]],
        ], dtype=np.float64)
        boundary_edge_part_ids = np.asarray(cfg.get('boundary_part_ids', [1, 1, 1, 1]), dtype=np.int32)
        part_id_map = np.ones(grid_shape, dtype=np.int32)
        geometry = GeometryND(
            spatial_dim=2,
            coordinate_system=coordinate_system,
            axes=axes,
            valid_mask=valid_mask,
            sdf=sdf,
            normal_components=normals,
            nearest_boundary_part_id_map=part_id_map,
            source_kind='synthetic_box',
            metadata={
                'bounds': list(map(float, bounds)),
                'boundary_edge_topology': validate_boundary_edges_2d(boundary_edges),
                'boundary_loop_count_2d': 1,
            },
            boundary_edges=boundary_edges,
            boundary_edge_part_ids=boundary_edge_part_ids,
            boundary_loops_2d=build_boundary_loops_2d(boundary_edges),
        )
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(v) for v in bounds]
        axes = (
            np.linspace(xmin, xmax, grid_shape[0]),
            np.linspace(ymin, ymax, grid_shape[1]),
            np.linspace(zmin, zmax, grid_shape[2]),
        )
        valid_mask = np.ones(grid_shape, dtype=bool)
        sdf, normals = _box_signed_distance_and_normal(axes, bounds)
        corners = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        ], dtype=np.float64)
        tri_vertex_ids = [
            (0, 2, 1), (0, 3, 2),  # z = zmin, outward -z
            (4, 5, 6), (4, 6, 7),  # z = zmax, outward +z
            (0, 1, 5), (0, 5, 4),  # y = ymin, outward -y
            (1, 2, 6), (1, 6, 5),  # x = xmax, outward +x
            (3, 6, 2), (3, 7, 6),  # y = ymax, outward +y
            (0, 7, 3), (0, 4, 7),  # x = xmin, outward -x
        ]
        tris = [[corners[a], corners[b], corners[c]] for a, b, c in tri_vertex_ids]
        boundary_triangles = np.asarray(tris, dtype=np.float64)
        part_ids = np.asarray(cfg.get('boundary_part_ids', [1] * len(boundary_triangles)), dtype=np.int32)
        if part_ids.size != boundary_triangles.shape[0]:
            part_ids = np.full(boundary_triangles.shape[0], int(part_ids[0]) if part_ids.size else 1, dtype=np.int32)
        part_id_map = np.ones(grid_shape, dtype=np.int32)
        surface_validation = validate_closed_surface_triangles(boundary_triangles)
        geometry = GeometryND(
            spatial_dim=3,
            coordinate_system=coordinate_system,
            axes=axes,
            valid_mask=valid_mask,
            sdf=sdf,
            normal_components=normals,
            nearest_boundary_part_id_map=part_id_map,
            source_kind='synthetic_box',
            metadata={'bounds': list(map(float, bounds)), 'boundary_surface_validation': surface_validation},
            boundary_triangles=boundary_triangles,
            boundary_triangle_part_ids=part_ids,
        )
    return GeometryProviderND(geometry=geometry, kind='synthetic_box')


def build_synthetic_field(cfg: Mapping[str, Any], spatial_dim: int, coordinate_system: str, axes: Tuple[np.ndarray, ...], gas_density_kgm3: float = 1.0) -> FieldProviderND:
    kind = str(cfg.get('kind', 'linear_shear')).strip()
    time_mode = str(cfg.get('time_mode', 'steady')).strip()
    times = np.asarray(cfg.get('times', [0.0]), dtype=np.float64)
    if times.size == 0:
        times = np.asarray([0.0], dtype=np.float64)
    grids = np.meshgrid(*axes, indexing='ij')
    quantities = {}
    if spatial_dim == 2:
        x, y = grids
        shear_rate = float(cfg.get('shear_rate', 5.0))
        ux = shear_rate * y
        uy = np.zeros_like(ux)
        data_shape = (times.size,) + ux.shape if times.size > 1 else ux.shape
        if times.size > 1:
            arr_ux = np.stack([(1.0 + 0.2 * np.sin(2 * np.pi * t / max(times[-1], 1.0))) * ux for t in times], axis=0)
            arr_uy = np.stack([uy for _ in times], axis=0)
        else:
            arr_ux, arr_uy = ux, uy
        quantities['ux'] = QuantitySeriesND('ux', 'm/s', times=times, data=np.asarray(arr_ux), metadata={})
        quantities['uy'] = QuantitySeriesND('uy', 'm/s', times=times, data=np.asarray(arr_uy), metadata={})
    else:
        x, y, z = grids
        shear_rate = float(cfg.get('shear_rate', 5.0))
        ux = shear_rate * y
        uy = np.zeros_like(ux)
        uz = np.zeros_like(ux)
        if times.size > 1:
            arr_ux = np.stack([(1.0 + 0.2 * np.sin(2 * np.pi * t / max(times[-1], 1.0))) * ux for t in times], axis=0)
            arr_uy = np.stack([uy for _ in times], axis=0)
            arr_uz = np.stack([uz for _ in times], axis=0)
        else:
            arr_ux, arr_uy, arr_uz = ux, uy, uz
        quantities['ux'] = QuantitySeriesND('ux', 'm/s', times=times, data=np.asarray(arr_ux), metadata={})
        quantities['uy'] = QuantitySeriesND('uy', 'm/s', times=times, data=np.asarray(arr_uy), metadata={})
        quantities['uz'] = QuantitySeriesND('uz', 'm/s', times=times, data=np.asarray(arr_uz), metadata={})
    mu = np.full_like(grids[0], float(cfg.get('dynamic_viscosity_Pas', 1.8e-5)), dtype=np.float64)
    quantities['mu'] = QuantitySeriesND('mu', 'Pa*s', times=times, data=(np.stack([mu for _ in times], axis=0) if times.size > 1 else mu), metadata={})
    if bool(cfg.get('provide_tauw', True)):
        tau = np.full_like(grids[0], float(cfg.get('tauw_value_Pa', 0.8)), dtype=np.float64)
        quantities['tauw'] = QuantitySeriesND('tauw', 'Pa', times=times, data=(np.stack([tau for _ in times], axis=0) if times.size > 1 else tau), metadata={})
        if bool(cfg.get('provide_utau', True)):
            rho = max(float(gas_density_kgm3), 1e-30)
            utau = np.sqrt(np.maximum(tau, 0.0) / rho)
            quantities['u_tau'] = QuantitySeriesND('u_tau', 'm/s', times=times, data=(np.stack([utau for _ in times], axis=0) if times.size > 1 else utau), metadata={})
    valid_mask = np.ones(tuple(len(ax) for ax in axes), dtype=bool)
    field = RegularFieldND(
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
        axis_names=tuple('xyz'[:spatial_dim]),
        axes=axes,
        quantities=quantities,
        valid_mask=valid_mask,
        time_mode=time_mode,
        metadata={'synthetic_kind': kind},
    )
    return FieldProviderND(field=field, kind='synthetic_field')
