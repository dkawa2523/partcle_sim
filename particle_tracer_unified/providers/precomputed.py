from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import json
import numpy as np

from ..core.datamodel import (
    FieldProviderND,
    GeometryND,
    GeometryProviderND,
    QuantitySeriesND,
    RegularFieldND,
    TriangleMeshField2D,
)
from ..core.geometry2d import build_boundary_loops_2d, decode_boundary_loops_2d, validate_boundary_edges_2d
from ..core.geometry3d import validate_closed_surface_triangles
from ..core.triangle_mesh_sampling_2d import build_triangle_candidate_grid


def _resolve_path(cfg: Mapping[str, Any], key: str = 'npz_path') -> Path:
    value = cfg.get(key)
    if value is None or str(value).strip() == '':
        raise ValueError(f'providers.{key} is required for precomputed_npz providers')
    return Path(str(value)).resolve()


def _read_axes(payload: Mapping[str, np.ndarray], spatial_dim: int) -> Tuple[np.ndarray, ...]:
    axes = []
    for i in range(spatial_dim):
        key = f'axis_{i}'
        if key not in payload:
            raise ValueError(f'Missing axis in npz: {key}')
        ax = np.asarray(payload[key], dtype=np.float64)
        if ax.ndim != 1 or ax.size < 2:
            raise ValueError(f'Axis {key} must be 1D with at least 2 entries')
        axes.append(ax)
    return tuple(axes)


def _infer_unit(name: str) -> str:
    if name in {'ux', 'uy', 'uz', 'ur', 'vz', 'u_tau', 'utau', 'friction_velocity', 'friction_velocity_mps'}:
        return 'm/s'
    if name in {'mu', 'dynamic_viscosity', 'dynamic_viscosity_Pas'}:
        return 'Pa*s'
    if name in {'tauw', 'tau_wall', 'wall_shear_stress', 'tauw_mag'}:
        return 'Pa'
    return ''


def _read_metadata(payload: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    raw = payload['metadata_json'] if 'metadata_json' in payload else None
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            raw = raw.item()
        elif raw.size == 1:
            raw = raw.reshape(()).item()
    if raw is None:
        return {}
    try:
        text = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_precomputed_geometry(cfg: Mapping[str, Any], spatial_dim: int, coordinate_system: str) -> GeometryProviderND:
    npz_path = _resolve_path(cfg)
    with np.load(npz_path) as payload:
        axes = _read_axes(payload, spatial_dim)
        sdf = np.asarray(payload['sdf'], dtype=np.float64)
        expected_shape = tuple(len(ax) for ax in axes)
        if sdf.shape != expected_shape:
            raise ValueError(f'Geometry sdf shape mismatch: expected {expected_shape}, got {sdf.shape}')
        valid_mask = np.asarray(payload['valid_mask'], dtype=bool) if 'valid_mask' in payload else np.ones(expected_shape, dtype=bool)
        if 'nearest_boundary_part_id_map' in payload:
            part_id_map = np.asarray(payload['nearest_boundary_part_id_map'], dtype=np.int32)
        elif 'part_id_map' in payload:
            part_id_map = np.asarray(payload['part_id_map'], dtype=np.int32)
        else:
            part_id_map = np.ones(expected_shape, dtype=np.int32)
        normals = []
        for i in range(spatial_dim):
            key = f'normal_{i}'
            if key in payload:
                normals.append(np.asarray(payload[key], dtype=np.float64))
        if len(normals) != spatial_dim:
            grads = np.gradient(sdf, *axes, edge_order=1)
            normals = [np.asarray(g, dtype=np.float64) for g in grads]
        boundary_edges = np.asarray(payload['boundary_edges'], dtype=np.float64) if 'boundary_edges' in payload else None
        boundary_edge_part_ids = np.asarray(payload['boundary_edge_part_ids'], dtype=np.int32) if 'boundary_edge_part_ids' in payload else None
        loops_flat = np.asarray(payload['boundary_loops_2d_flat'], dtype=np.float64) if 'boundary_loops_2d_flat' in payload else None
        loops_offsets = np.asarray(payload['boundary_loops_2d_offsets'], dtype=np.int32) if 'boundary_loops_2d_offsets' in payload else None
        boundary_triangles = np.asarray(payload['boundary_triangles'], dtype=np.float64) if 'boundary_triangles' in payload else None
        boundary_triangle_part_ids = np.asarray(payload['boundary_triangle_part_ids'], dtype=np.int32) if 'boundary_triangle_part_ids' in payload else None
        metadata = _read_metadata(payload)
    boundary_loops_2d = decode_boundary_loops_2d(loops_flat, loops_offsets)
    if spatial_dim == 2 and not boundary_loops_2d and boundary_edges is not None:
        boundary_loops_2d = build_boundary_loops_2d(boundary_edges)
    if spatial_dim == 2 and boundary_edges is not None:
        metadata = {
            **metadata,
            'boundary_edge_topology': validate_boundary_edges_2d(boundary_edges),
            'boundary_loop_count_2d': int(len(boundary_loops_2d)),
        }
    if spatial_dim == 3 and boundary_triangles is not None:
        if boundary_triangle_part_ids is not None and boundary_triangle_part_ids.shape[0] != boundary_triangles.shape[0]:
            raise ValueError(
                'boundary_triangle_part_ids length mismatch: '
                f'expected {boundary_triangles.shape[0]}, got {boundary_triangle_part_ids.shape[0]}'
            )
        metadata = {
            **metadata,
            'boundary_surface_validation': validate_closed_surface_triangles(boundary_triangles),
        }
    geometry = GeometryND(
        spatial_dim=int(spatial_dim),
        coordinate_system=str(coordinate_system),
        axes=axes,
        valid_mask=valid_mask,
        sdf=sdf,
        normal_components=tuple(normals),
        nearest_boundary_part_id_map=part_id_map,
        source_kind=str(metadata.get('source_kind', 'precomputed_npz')),
        metadata={'npz_path': str(npz_path), 'provider_kind': 'precomputed_npz', **metadata},
        boundary_edges=boundary_edges,
        boundary_edge_part_ids=boundary_edge_part_ids,
        boundary_loops_2d=boundary_loops_2d,
        boundary_triangles=boundary_triangles,
        boundary_triangle_part_ids=boundary_triangle_part_ids,
    )
    return GeometryProviderND(geometry=geometry, kind=str(metadata.get('provider_kind', 'precomputed_npz')))


def build_precomputed_field(
    cfg: Mapping[str, Any],
    spatial_dim: int,
    coordinate_system: str,
    axes: Tuple[np.ndarray, ...],
    gas_density_kgm3: float = 1.0,
) -> FieldProviderND:
    npz_path = _resolve_path(cfg)
    with np.load(npz_path) as payload:
        field_axes = _read_axes(payload, spatial_dim) if f'axis_{spatial_dim - 1}' in payload else tuple(np.asarray(ax, dtype=np.float64) for ax in axes)
        if len(field_axes) != len(axes) or any(a.shape != b.shape or not np.allclose(a, b, atol=1e-12, rtol=0.0) for a, b in zip(field_axes, axes)):
            raise ValueError(f'Field axes must exactly match geometry axes in {npz_path}')
        expected_shape = tuple(len(ax) for ax in field_axes)
        valid_mask = np.asarray(payload['valid_mask'], dtype=bool) if 'valid_mask' in payload else np.ones(expected_shape, dtype=bool)
        times = np.asarray(payload['times'], dtype=np.float64) if 'times' in payload else np.asarray([0.0], dtype=np.float64)
        metadata = _read_metadata(payload)
        reserved = {
            'axis_0', 'axis_1', 'axis_2', 'times', 'valid_mask', 'metadata_json',
            'sdf', 'part_id_map', 'nearest_boundary_part_id_map', 'normal_0', 'normal_1', 'normal_2',
            'boundary_edges', 'boundary_edge_part_ids', 'boundary_triangles', 'boundary_triangle_part_ids',
            'boundary_loops_2d_flat', 'boundary_loops_2d_offsets',
        }
        quantities: Dict[str, QuantitySeriesND] = {}
        for key in payload.files:
            if key in reserved or key.startswith('axis_'):
                continue
            data = np.asarray(payload[key], dtype=np.float64)
            if data.ndim == spatial_dim:
                arr = data
            elif data.ndim == spatial_dim + 1:
                if data.shape[0] != times.size:
                    raise ValueError(f'Quantity {key} time axis mismatch: data has {data.shape[0]} steps, times has {times.size}')
                arr = data
            else:
                continue
            quantities[key] = QuantitySeriesND(name=key, unit=_infer_unit(key), times=times, data=arr, metadata={})
    if not quantities:
        raise ValueError(f'No field quantities found in {npz_path}')
    any_transient = any(np.asarray(q.data).ndim == spatial_dim + 1 and times.size > 1 for q in quantities.values())
    field = RegularFieldND(
        spatial_dim=int(spatial_dim),
        coordinate_system=str(coordinate_system),
        axis_names=tuple('xyz'[:spatial_dim]),
        axes=field_axes,
        quantities=quantities,
        valid_mask=valid_mask,
        time_mode='transient' if any_transient else 'steady',
        metadata={'npz_path': str(npz_path), 'provider_kind': 'precomputed_npz', 'gas_density_kgm3': float(gas_density_kgm3), **metadata},
    )
    return FieldProviderND(field=field, kind=str(metadata.get('provider_kind', 'precomputed_npz')))


def build_precomputed_triangle_mesh_field(
    cfg: Mapping[str, Any],
    spatial_dim: int,
    coordinate_system: str,
    gas_density_kgm3: float = 1.0,
) -> FieldProviderND:
    if int(spatial_dim) != 2:
        raise ValueError('precomputed_triangle_mesh_npz currently supports only spatial_dim=2')
    npz_path = _resolve_path(cfg)
    with np.load(npz_path) as payload:
        if 'mesh_vertices' not in payload or 'mesh_triangles' not in payload:
            raise ValueError(f'Mesh field npz must include mesh_vertices and mesh_triangles: {npz_path}')
        mesh_vertices = np.asarray(payload['mesh_vertices'], dtype=np.float64)
        mesh_triangles = np.asarray(payload['mesh_triangles'], dtype=np.int32)
        if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 2:
            raise ValueError('mesh_vertices must have shape (n, 2)')
        if mesh_triangles.ndim != 2 or mesh_triangles.shape[1] != 3:
            raise ValueError('mesh_triangles must have shape (m, 3)')
        times = np.asarray(payload['times'], dtype=np.float64) if 'times' in payload else np.asarray([0.0], dtype=np.float64)
        metadata = _read_metadata(payload)
        reserved = {'mesh_vertices', 'mesh_triangles', 'times', 'metadata_json'}
        quantities: Dict[str, QuantitySeriesND] = {}
        expected_vertex_count = int(mesh_vertices.shape[0])
        for key in payload.files:
            if key in reserved:
                continue
            data = np.asarray(payload[key], dtype=np.float64)
            if data.ndim == 1:
                if data.shape[0] != expected_vertex_count:
                    raise ValueError(f'Mesh quantity {key} vertex axis mismatch: expected {expected_vertex_count}, got {data.shape[0]}')
                arr = data
            elif data.ndim == 2:
                if data.shape[0] != times.size or data.shape[1] != expected_vertex_count:
                    raise ValueError(
                        f'Mesh quantity {key} shape mismatch: expected {(times.size, expected_vertex_count)}, got {data.shape}'
                    )
                arr = data
            else:
                continue
            quantities[key] = QuantitySeriesND(name=key, unit=_infer_unit(key), times=times, data=arr, metadata={})
    if not quantities:
        raise ValueError(f'No mesh field quantities found in {npz_path}')
    accel_origin, accel_cell_size, accel_shape, accel_offsets, accel_triangle_indices = build_triangle_candidate_grid(
        mesh_vertices,
        mesh_triangles,
    )
    any_transient = any(np.asarray(q.data).ndim == 2 and times.size > 1 for q in quantities.values())
    field = TriangleMeshField2D(
        spatial_dim=2,
        coordinate_system=str(coordinate_system),
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        quantities=quantities,
        accel_origin=np.asarray(accel_origin, dtype=np.float64),
        accel_cell_size=np.asarray(accel_cell_size, dtype=np.float64),
        accel_shape=(int(accel_shape[0]), int(accel_shape[1])),
        accel_cell_offsets=np.asarray(accel_offsets, dtype=np.int32),
        accel_triangle_indices=np.asarray(accel_triangle_indices, dtype=np.int32),
        time_mode='transient' if any_transient else 'steady',
        metadata={
            'npz_path': str(npz_path),
            'provider_kind': 'precomputed_triangle_mesh_npz',
            'field_backend_kind': 'triangle_mesh_2d',
            'support_tolerance_m': float(metadata.get('support_tolerance_m', 2.0e-6)),
            'gas_density_kgm3': float(gas_density_kgm3),
            **metadata,
        },
    )
    return FieldProviderND(field=field, kind='precomputed_triangle_mesh_npz')
