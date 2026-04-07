from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..core.datamodel import GeometryProviderND
from ..core.grid_sampling import sample_grid_scalar as _sample_grid_scalar
from .source_sampler_types import SourceFlowSampler, SourceNormalSampler, SourceScalarSampler


@dataclass
class _SurfaceCache:
    centroids: np.ndarray
    normals: np.ndarray
    part_ids: np.ndarray
    tree: Optional[cKDTree]
    part_to_indices: Dict[int, np.ndarray]


def _sample_sdf_scalar(geometry_provider: GeometryProviderND, position: np.ndarray) -> float:
    geom = geometry_provider.geometry
    return _sample_grid_scalar(np.asarray(geom.sdf, dtype=np.float64), geom.axes, np.asarray(position, dtype=np.float64))


def _sample_sdf_normal(geometry_provider: GeometryProviderND, position: np.ndarray) -> np.ndarray:
    geom = geometry_provider.geometry
    values = [_sample_grid_scalar(np.asarray(component, dtype=np.float64), geom.axes, position) for component in geom.normal_components]
    normal = np.asarray(values, dtype=np.float64)
    magnitude = np.linalg.norm(normal)
    if magnitude <= 1e-30:
        normal = np.zeros(geom.spatial_dim, dtype=np.float64)
        normal[-1 if geom.spatial_dim > 1 else 0] = 1.0
        return normal
    return normal / magnitude


def _surface_cache_from_geometry(geometry_provider: GeometryProviderND) -> Optional[_SurfaceCache]:
    geom = geometry_provider.geometry
    if int(geom.spatial_dim) == 2 and geom.boundary_edges is not None:
        segments = np.asarray(geom.boundary_edges, dtype=np.float64)
        centroids = 0.5 * (segments[:, 0, :] + segments[:, 1, :])
        tangents = segments[:, 1, :] - segments[:, 0, :]
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        magnitude = np.linalg.norm(normals, axis=1)
        magnitude[magnitude <= 1e-30] = 1.0
        normals = normals / magnitude[:, None]
        part_ids = np.asarray(
            geom.boundary_edge_part_ids if geom.boundary_edge_part_ids is not None else np.zeros(segments.shape[0], dtype=np.int32),
            dtype=np.int32,
        )
    elif int(geom.spatial_dim) == 3 and geom.boundary_triangles is not None:
        triangles = np.asarray(geom.boundary_triangles, dtype=np.float64)
        centroids = triangles.mean(axis=1)
        normals = np.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :])
        magnitude = np.linalg.norm(normals, axis=1)
        magnitude[magnitude <= 1e-30] = 1.0
        normals = normals / magnitude[:, None]
        part_ids = np.asarray(
            geom.boundary_triangle_part_ids if geom.boundary_triangle_part_ids is not None else np.zeros(triangles.shape[0], dtype=np.int32),
            dtype=np.int32,
        )
    else:
        return None
    tree = cKDTree(centroids) if centroids.size else None
    unique_part_ids = np.unique(part_ids.astype(np.int32))
    part_to_indices = {
        int(part_id): np.flatnonzero(part_ids == part_id)
        for part_id in unique_part_ids
        if int(part_id) > 0
    }
    return _SurfaceCache(
        centroids=centroids,
        normals=normals,
        part_ids=part_ids,
        tree=tree,
        part_to_indices=part_to_indices,
    )


def _nearest_surface_record(cache: _SurfaceCache, position: np.ndarray, source_part_id: int) -> Tuple[np.ndarray, np.ndarray, int]:
    pos = np.asarray(position, dtype=np.float64)
    if cache.tree is None or cache.centroids.size == 0:
        raise ValueError('Surface cache is empty')
    if int(source_part_id) > 0 and int(source_part_id) in cache.part_to_indices:
        candidate_indices = cache.part_to_indices[int(source_part_id)]
        if candidate_indices.size:
            d2 = np.sum((cache.centroids[candidate_indices] - pos[None, :]) ** 2, axis=1)
            idx = int(candidate_indices[int(np.argmin(d2))])
            return cache.centroids[idx], cache.normals[idx], int(cache.part_ids[idx])
    _distance, idx = cache.tree.query(pos, k=1)
    idx = int(np.atleast_1d(idx)[0])
    return cache.centroids[idx], cache.normals[idx], int(cache.part_ids[idx])


class GeometryNormalSamplerND(SourceNormalSampler):
    def __init__(self, geometry_provider: GeometryProviderND, k_nearest: int = 8):
        self.geometry_provider = geometry_provider
        self.k_nearest = int(max(1, k_nearest))
        self.cache = _surface_cache_from_geometry(geometry_provider)

    def __call__(self, position: np.ndarray, source_part_id: int) -> np.ndarray:
        pos = np.asarray(position, dtype=np.float64)
        if self.cache is not None and self.cache.tree is not None:
            indices = None
            if int(source_part_id) > 0 and int(source_part_id) in self.cache.part_to_indices:
                candidate_indices = self.cache.part_to_indices[int(source_part_id)]
                if candidate_indices.size:
                    d2 = np.sum((self.cache.centroids[candidate_indices] - pos[None, :]) ** 2, axis=1)
                    indices = np.array([candidate_indices[int(np.argmin(d2))]], dtype=np.int64)
            if indices is None:
                k = min(self.k_nearest, self.cache.centroids.shape[0])
                _, idx = self.cache.tree.query(pos, k=k)
                indices = np.atleast_1d(idx).astype(np.int64)
            normal = np.mean(self.cache.normals[indices], axis=0)
            magnitude = np.linalg.norm(normal)
            if magnitude > 1e-30:
                return normal / magnitude
        return _sample_sdf_normal(self.geometry_provider, pos)


class DirectWallShearSampler(SourceScalarSampler):
    def __init__(
        self,
        geometry_provider: GeometryProviderND,
        flow_sampler: SourceFlowSampler,
        normal_sampler: SourceNormalSampler,
        viscosity_sampler: SourceScalarSampler,
        probe_distance_m: float,
    ):
        self.geometry_provider = geometry_provider
        self.flow_sampler = flow_sampler
        self.normal_sampler = normal_sampler
        self.viscosity_sampler = viscosity_sampler
        self.probe_distance_m = float(max(probe_distance_m, 1e-9))

    def __call__(self, position: np.ndarray, release_time: float, source_part_id: int = 0) -> float:
        pos = np.asarray(position, dtype=np.float64)
        normal = np.asarray(self.normal_sampler(pos, source_part_id), dtype=np.float64)
        magnitude = np.linalg.norm(normal)
        if magnitude <= 1e-30:
            return float('nan')
        normal = normal / magnitude
        probe_distance = self.probe_distance_m
        p_minus = pos - probe_distance * normal
        p_plus = pos + probe_distance * normal
        sdf_minus = _sample_sdf_scalar(self.geometry_provider, p_minus)
        sdf_plus = _sample_sdf_scalar(self.geometry_provider, p_plus)
        inside_minus = sdf_minus <= 0.0
        inside_plus = sdf_plus <= 0.0
        u_minus = np.asarray(self.flow_sampler(p_minus, release_time), dtype=np.float64)
        u_plus = np.asarray(self.flow_sampler(p_plus, release_time), dtype=np.float64)
        ut_minus = u_minus - np.dot(u_minus, normal) * normal
        ut_plus = u_plus - np.dot(u_plus, normal) * normal
        if inside_minus and inside_plus:
            du_t_dn = np.linalg.norm(ut_plus - ut_minus) / max(2.0 * probe_distance, 1e-30)
        elif inside_minus:
            du_t_dn = np.linalg.norm(ut_minus) / max(probe_distance, 1e-30)
        elif inside_plus:
            du_t_dn = np.linalg.norm(ut_plus) / max(probe_distance, 1e-30)
        else:
            du_t_dn = max(np.linalg.norm(ut_plus), np.linalg.norm(ut_minus)) / max(probe_distance, 1e-30)
        mu_local = float(self.viscosity_sampler(pos, release_time, source_part_id))
        if not np.isfinite(mu_local):
            return float('nan')
        return float(abs(mu_local) * du_t_dn)


class DirectFrictionVelocitySampler(SourceScalarSampler):
    def __init__(self, wall_shear_sampler: SourceScalarSampler, gas_density_kgm3: float):
        self.wall_shear_sampler = wall_shear_sampler
        self.gas_density_kgm3 = float(max(gas_density_kgm3, 1e-30))

    def __call__(self, position: np.ndarray, release_time: float, source_part_id: int = 0) -> float:
        tau = float(self.wall_shear_sampler(position, release_time, source_part_id))
        if not np.isfinite(tau) or tau < 0.0:
            return float('nan')
        return float(np.sqrt(tau / self.gas_density_kgm3))


def _default_probe_distance(runtime) -> float:
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None:
        return 1e-6
    ds = []
    for axis in geometry_provider.geometry.axes:
        if len(axis) > 1:
            ds.append(float(np.min(np.diff(axis))))
    return max(1e-9, 0.5 * min(ds) if ds else 1e-6)


__all__ = (
    'DirectFrictionVelocitySampler',
    'DirectWallShearSampler',
    'GeometryNormalSamplerND',
    '_SurfaceCache',
    '_default_probe_distance',
    '_nearest_surface_record',
    '_sample_sdf_normal',
    '_sample_sdf_scalar',
    '_surface_cache_from_geometry',
)
