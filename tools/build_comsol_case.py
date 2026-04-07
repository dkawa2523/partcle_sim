from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN, sample_valid_mask_status
from particle_tracer_unified.core.geometry2d import (
    build_boundary_loops_2d,
    encode_boundary_loops_2d,
    points_inside_boundary_loops_2d,
)
from particle_tracer_unified.core.grid_sampling import sample_grid_scalar


@dataclass(frozen=True)
class MeshTypeBlock:
    type_name: str
    vertices_per_element: int
    elements: np.ndarray
    geometric_entity_indices: np.ndarray


@dataclass(frozen=True)
class ParsedMesh:
    sdim: int
    vertices: np.ndarray
    type_blocks: Dict[str, MeshTypeBlock]


def _find_line(lines: List[str], start: int, marker: str) -> int:
    for i in range(start, len(lines)):
        if marker in lines[i]:
            return i
    raise ValueError(f'Could not find marker: {marker}')


def _consume_numbers(lines: List[str], start: int, count: int, cast):
    values = []
    i = start
    while len(values) < count and i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s or s.startswith('#'):
            continue
        values.extend(cast(x) for x in s.split())
    if len(values) < count:
        raise ValueError(f'Expected {count} numeric values, got {len(values)}')
    return values[:count], i


def parse_comsol_mphtxt(path: Path) -> ParsedMesh:
    lines = path.read_text(encoding='utf-8').splitlines()
    sdim_idx = _find_line(lines, 0, '# sdim')
    sdim = int(lines[sdim_idx].split('#')[0].strip())
    nverts_idx = _find_line(lines, sdim_idx, '# number of mesh vertices')
    nverts = int(lines[nverts_idx].split('#')[0].strip())
    coords_header_idx = _find_line(lines, nverts_idx, '# Mesh vertex coordinates')
    coords, cursor = _consume_numbers(lines, coords_header_idx + 1, nverts * sdim, float)
    vertices = np.asarray(coords, dtype=np.float64).reshape((nverts, sdim))

    ntypes_idx = _find_line(lines, cursor, '# number of element types')
    ntypes = int(lines[ntypes_idx].split('#')[0].strip())
    cursor = ntypes_idx + 1
    blocks: Dict[str, MeshTypeBlock] = {}
    for _ in range(ntypes):
        tname_idx = _find_line(lines, cursor, '# type name')
        tname_tokens = lines[tname_idx].split('#')[0].split()
        if len(tname_tokens) < 2:
            raise ValueError(f'Invalid type-name line: {lines[tname_idx]}')
        type_name = tname_tokens[1].strip()

        nvp_idx = _find_line(lines, tname_idx, '# number of vertices per element')
        nvp = int(lines[nvp_idx].split('#')[0].strip())
        nelem_idx = _find_line(lines, nvp_idx, '# number of elements')
        nelem = int(lines[nelem_idx].split('#')[0].strip())
        elem_header_idx = _find_line(lines, nelem_idx, '# Elements')
        elem_values, cursor = _consume_numbers(lines, elem_header_idx + 1, nelem * nvp, int)
        elements = np.asarray(elem_values, dtype=np.int64).reshape((nelem, nvp))

        ngeom_idx = _find_line(lines, cursor, '# number of geometric entity indices')
        ngeom = int(lines[ngeom_idx].split('#')[0].strip())
        geom_header_idx = _find_line(lines, ngeom_idx, '# Geometric entity indices')
        geom_values, cursor = _consume_numbers(lines, geom_header_idx + 1, ngeom, int)
        geom = np.asarray(geom_values, dtype=np.int64)
        if geom.size != nelem:
            raise ValueError(f'Geometric entity size mismatch for {type_name}: {geom.size} vs {nelem}')
        blocks[type_name] = MeshTypeBlock(
            type_name=type_name,
            vertices_per_element=nvp,
            elements=elements,
            geometric_entity_indices=geom,
        )
    return ParsedMesh(sdim=sdim, vertices=vertices, type_blocks=blocks)


def _inside_mask_from_quads(vertices: np.ndarray, quads: np.ndarray, axes_x: np.ndarray, axes_y: np.ndarray) -> np.ndarray:
    ordered_quads = _order_quad_vertices(vertices, quads)
    tri = np.vstack([ordered_quads[:, [0, 1, 2]], ordered_quads[:, [0, 2, 3]]]).astype(np.int32)
    xx, yy = np.meshgrid(axes_x, axes_y, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside_flat = np.zeros(points.shape[0], dtype=bool)
    eps = 1e-14

    for t in tri:
        a = vertices[t[0]]
        b = vertices[t[1]]
        c = vertices[t[2]]
        xmin = min(a[0], b[0], c[0]) - eps
        xmax = max(a[0], b[0], c[0]) + eps
        ymin = min(a[1], b[1], c[1]) - eps
        ymax = max(a[1], b[1], c[1]) + eps
        cand = np.where(
            (~inside_flat)
            & (points[:, 0] >= xmin)
            & (points[:, 0] <= xmax)
            & (points[:, 1] >= ymin)
            & (points[:, 1] <= ymax)
        )[0]
        if cand.size == 0:
            continue
        p = points[cand]
        v0 = c - a
        v1 = b - a
        v2 = p - a[None, :]
        den = float(v0[0] * v1[1] - v1[0] * v0[1])
        if abs(den) <= 1e-30:
            continue
        u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
        v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
        hit = (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps)
        inside_flat[cand[hit]] = True
    return inside_flat.reshape(xx.shape)


def _distance_and_nearest_edge(points: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray):
    min_dist = np.full(points.shape[0], np.inf, dtype=np.float64)
    nearest = np.zeros(points.shape[0], dtype=np.int64)
    for k in range(edge_start.shape[0]):
        p0 = edge_start[k]
        p1 = edge_end[k]
        ab = p1 - p0
        denom = float(np.dot(ab, ab))
        if denom <= 1e-30:
            dist = np.linalg.norm(points - p0[None, :], axis=1)
        else:
            ap = points - p0[None, :]
            t = np.clip((ap @ ab) / denom, 0.0, 1.0)
            proj = p0[None, :] + t[:, None] * ab[None, :]
            dist = np.linalg.norm(points - proj, axis=1)
        better = dist < min_dist
        min_dist[better] = dist[better]
        nearest[better] = k
    return min_dist, nearest


def _points_inside_quads(vertices: np.ndarray, quads: np.ndarray, points: np.ndarray) -> np.ndarray:
    ordered_quads = _order_quad_vertices(vertices, quads)
    tri = np.vstack([ordered_quads[:, [0, 1, 2]], ordered_quads[:, [0, 2, 3]]]).astype(np.int32)
    pts = np.asarray(points, dtype=np.float64)
    inside = np.zeros(pts.shape[0], dtype=bool)
    eps = 1e-14
    for t in tri:
        a = vertices[t[0]]
        b = vertices[t[1]]
        c = vertices[t[2]]
        xmin = min(a[0], b[0], c[0]) - eps
        xmax = max(a[0], b[0], c[0]) + eps
        ymin = min(a[1], b[1], c[1]) - eps
        ymax = max(a[1], b[1], c[1]) + eps
        cand = np.where(
            (~inside)
            & (pts[:, 0] >= xmin)
            & (pts[:, 0] <= xmax)
            & (pts[:, 1] >= ymin)
            & (pts[:, 1] <= ymax)
        )[0]
        if cand.size == 0:
            continue
        p = pts[cand]
        v0 = c - a
        v1 = b - a
        v2 = p - a[None, :]
        den = float(v0[0] * v1[1] - v1[0] * v0[1])
        if abs(den) <= 1e-30:
            continue
        u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
        v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
        hit = (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps)
        inside[cand[hit]] = True
    return inside


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return 0.5 * abs(float(ab[0] * ac[1] - ab[1] * ac[0]))


def _order_quad_vertices(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    elems = np.asarray(quads, dtype=np.int64)
    ordered = np.empty_like(elems)
    for i, elem in enumerate(elems):
        pts = verts[elem]
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        ordered[i] = elem[np.argsort(angles)]
    return ordered


def _merge_near_duplicate_axis(axis: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    vals = np.sort(np.asarray(axis, dtype=np.float64))
    if vals.size <= 1:
        return vals.copy()
    merged: list[float] = []
    bucket = [float(vals[0])]
    for value in vals[1:]:
        current = float(value)
        if abs(current - bucket[-1]) <= float(atol):
            bucket.append(current)
            continue
        merged.append(float(np.mean(bucket)))
        bucket = [current]
    merged.append(float(np.mean(bucket)))
    return np.asarray(merged, dtype=np.float64)


def _make_uniform_axis(vmin: float, vmax: float, spacing: float) -> np.ndarray:
    lo = float(vmin)
    hi = float(vmax)
    dx = max(float(spacing), 1e-6)
    count = max(2, int(np.ceil((hi - lo) / dx)) + 1)
    axis = lo + dx * np.arange(count, dtype=np.float64)
    axis[-1] = hi
    return axis


def _load_npz_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _validate_field_bundle_payload(payload: Mapping[str, np.ndarray], axes_x: np.ndarray, axes_y: np.ndarray) -> Dict[str, np.ndarray]:
    if 'ux' not in payload or 'uy' not in payload:
        raise ValueError('field bundle must include ux and uy')
    bundle_axes_x = np.asarray(payload['axis_0'], dtype=np.float64) if 'axis_0' in payload else None
    bundle_axes_y = np.asarray(payload['axis_1'], dtype=np.float64) if 'axis_1' in payload else None
    if bundle_axes_x is not None and (bundle_axes_x.shape != axes_x.shape or not np.allclose(bundle_axes_x, axes_x, atol=1e-12, rtol=0.0)):
        raise ValueError('field bundle axis_0 must exactly match geometry axis_0')
    if bundle_axes_y is not None and (bundle_axes_y.shape != axes_y.shape or not np.allclose(bundle_axes_y, axes_y, atol=1e-12, rtol=0.0)):
        raise ValueError('field bundle axis_1 must exactly match geometry axis_1')
    expected_shape = (axes_x.size, axes_y.size)
    times = np.asarray(payload['times'], dtype=np.float64) if 'times' in payload else np.asarray([0.0], dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError('field bundle times must be a non-empty 1D array when provided')
    normalized: Dict[str, np.ndarray] = {
        'axis_0': axes_x.astype(np.float64),
        'axis_1': axes_y.astype(np.float64),
        'times': times.astype(np.float64),
    }
    valid_mask = np.asarray(payload['valid_mask'], dtype=bool) if 'valid_mask' in payload else np.ones(expected_shape, dtype=bool)
    if valid_mask.shape != expected_shape:
        raise ValueError(f'field bundle valid_mask must match geometry grid shape {expected_shape}')
    normalized['valid_mask'] = valid_mask
    reserved = {'axis_0', 'axis_1', 'times', 'valid_mask', 'metadata_json'}
    for key, value in payload.items():
        if key in reserved:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape != expected_shape:
                raise ValueError(f'field bundle quantity {key} must match geometry grid shape {expected_shape}')
        elif arr.ndim == 3:
            if arr.shape[0] != times.size or arr.shape[1:] != expected_shape:
                raise ValueError(f'field bundle quantity {key} must match shape {(times.size,) + expected_shape}')
        else:
            raise ValueError(f'field bundle quantity {key} must be 2D or 3D, got shape {arr.shape}')
        normalized[key] = arr
    return normalized


def _apply_field_valid_mask(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if arr.ndim == 2:
        out = arr.copy()
        out[~mask] = 0.0
        return out
    out = arr.copy()
    out[:, ~mask] = 0.0
    return out


def _split_quads_to_triangles(quads: np.ndarray) -> np.ndarray:
    ordered = np.asarray(quads, dtype=np.int32)
    return np.vstack([ordered[:, [0, 1, 2]], ordered[:, [0, 2, 3]]]).astype(np.int32)


def _max_cell_diagonal(axes_x: np.ndarray, axes_y: np.ndarray) -> float:
    dx = float(np.max(np.diff(np.asarray(axes_x, dtype=np.float64))))
    dy = float(np.max(np.diff(np.asarray(axes_y, dtype=np.float64))))
    return float(np.sqrt(dx * dx + dy * dy))


def _sample_grid_normal(normal_x: np.ndarray, normal_y: np.ndarray, axes_x: np.ndarray, axes_y: np.ndarray, position: np.ndarray) -> np.ndarray:
    px = float(sample_grid_scalar(np.asarray(normal_x, dtype=np.float64), (axes_x, axes_y), np.asarray(position, dtype=np.float64)))
    py = float(sample_grid_scalar(np.asarray(normal_y, dtype=np.float64), (axes_x, axes_y), np.asarray(position, dtype=np.float64)))
    normal = np.asarray([px, py], dtype=np.float64)
    mag = float(np.linalg.norm(normal))
    if mag <= 1.0e-30:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    return normal / mag


def _sample_bundle_quantity_at_point(bundle_payload: Mapping[str, np.ndarray], quantity_name: str, axes_x: np.ndarray, axes_y: np.ndarray, position: np.ndarray) -> np.ndarray:
    arr = np.asarray(bundle_payload[quantity_name], dtype=np.float64)
    point = np.asarray(position, dtype=np.float64)
    if arr.ndim == 2:
        return np.asarray(sample_grid_scalar(arr, (axes_x, axes_y), point), dtype=np.float64)
    out = np.zeros(arr.shape[0], dtype=np.float64)
    for time_index in range(arr.shape[0]):
        out[time_index] = float(sample_grid_scalar(arr[time_index], (axes_x, axes_y), point))
    return out


def _build_triangle_mesh_field_payload(
    *,
    mesh: ParsedMesh,
    arrays: Mapping[str, Any],
    bundle_payload: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    boundary_vertex_ids = np.unique(np.asarray(mesh.type_blocks['edg'].elements, dtype=np.int64).reshape(-1))
    boundary_vertex_mask = np.zeros(vertices.shape[0], dtype=bool)
    boundary_vertex_mask[boundary_vertex_ids] = True
    axes_x = np.asarray(arrays['axes_x'], dtype=np.float64)
    axes_y = np.asarray(arrays['axes_y'], dtype=np.float64)
    valid_mask = np.asarray(bundle_payload['valid_mask'], dtype=bool)
    normal_x = np.asarray(arrays['normal_x'], dtype=np.float64)
    normal_y = np.asarray(arrays['normal_y'], dtype=np.float64)
    cell_diagonal = _max_cell_diagonal(axes_x, axes_y)
    donor_positions = np.full((vertices.shape[0], 2), np.nan, dtype=np.float64)
    direct_valid_mask = np.zeros(vertices.shape[0], dtype=bool)
    probe_success_count = 0

    for vertex_id, position in enumerate(vertices):
        status = int(sample_valid_mask_status(valid_mask, (axes_x, axes_y), position))
        if status == int(VALID_MASK_STATUS_CLEAN):
            donor_positions[vertex_id] = position
            direct_valid_mask[vertex_id] = True
            continue
        if not bool(boundary_vertex_mask[vertex_id]):
            continue
        normal = _sample_grid_normal(normal_x, normal_y, axes_x, axes_y, position)
        inward = -normal
        for factor in (0.5, 1.0, 1.5, 2.0):
            donor = np.asarray(position + inward * (float(factor) * cell_diagonal), dtype=np.float64)
            status = int(sample_valid_mask_status(valid_mask, (axes_x, axes_y), donor))
            if status == int(VALID_MASK_STATUS_CLEAN):
                donor_positions[vertex_id] = donor
                probe_success_count += 1
                break

    interior_candidate_mask = (~boundary_vertex_mask) & direct_valid_mask
    if not np.any(interior_candidate_mask):
        interior_candidate_mask = direct_valid_mask.copy()
    if not np.any(interior_candidate_mask):
        raise ValueError('Could not find any inside mesh vertices with valid bundle support')
    interior_positions = vertices[interior_candidate_mask]
    interior_tree = cKDTree(interior_positions)
    interior_indices = np.flatnonzero(interior_candidate_mask)
    fallback_count = 0
    for vertex_id, position in enumerate(vertices):
        if np.all(np.isfinite(donor_positions[vertex_id])):
            continue
        _dist, local_idx = interior_tree.query(position, k=1)
        donor_positions[vertex_id] = vertices[int(interior_indices[int(np.atleast_1d(local_idx)[0])])]
        fallback_count += 1

    mesh_triangles = _split_quads_to_triangles(np.asarray(arrays['quads'], dtype=np.int32))
    reserved = {'axis_0', 'axis_1', 'times', 'valid_mask', 'metadata_json'}
    payload: Dict[str, Any] = {
        'mesh_vertices': vertices.astype(np.float64),
        'mesh_triangles': mesh_triangles.astype(np.int32),
        'times': np.asarray(bundle_payload['times'], dtype=np.float64),
    }
    for key, value in bundle_payload.items():
        if key in reserved:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 2:
            sampled = np.zeros(vertices.shape[0], dtype=np.float64)
        elif arr.ndim == 3:
            sampled = np.zeros((arr.shape[0], vertices.shape[0]), dtype=np.float64)
        else:
            continue
        for vertex_id in range(vertices.shape[0]):
            sampled_at_vertex = _sample_bundle_quantity_at_point(bundle_payload, key, axes_x, axes_y, donor_positions[vertex_id])
            if np.asarray(sampled_at_vertex).ndim == 0:
                sampled[vertex_id] = float(sampled_at_vertex)
            else:
                sampled[:, vertex_id] = np.asarray(sampled_at_vertex, dtype=np.float64)
        payload[key] = sampled
    metadata = {
        'provider_kind': 'precomputed_triangle_mesh_npz',
        'field_backend_kind': 'triangle_mesh_2d',
        'support_tolerance_m': 2.0e-6,
        'source_kind': 'comsol_triangle_mesh_field',
        'mesh_vertex_count': int(vertices.shape[0]),
        'mesh_triangle_count': int(mesh_triangles.shape[0]),
        'boundary_vertex_count': int(np.count_nonzero(boundary_vertex_mask)),
        'boundary_vertex_probe_success_count': int(probe_success_count),
        'boundary_vertex_fallback_count': int(fallback_count),
    }
    payload['metadata_json'] = np.asarray(json.dumps(metadata))
    return payload


def _sample_points_in_quads(vertices: np.ndarray, quads: np.ndarray, count: int, seed: int = 12345) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    q = _order_quad_vertices(v, quads)
    n = int(max(0, count))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    quad_vertices = v[q]
    tri0 = quad_vertices[:, [0, 1, 2], :]
    tri1 = quad_vertices[:, [0, 2, 3], :]
    area0 = np.asarray([_triangle_area(t[0], t[1], t[2]) for t in tri0], dtype=np.float64)
    area1 = np.asarray([_triangle_area(t[0], t[1], t[2]) for t in tri1], dtype=np.float64)
    quad_area = area0 + area1
    if not np.any(quad_area > 0.0):
        return quad_vertices.mean(axis=1)[:n].copy()
    prob = quad_area / max(float(quad_area.sum()), 1e-30)
    rng = np.random.default_rng(int(seed))
    quad_idx = rng.choice(q.shape[0], size=n, replace=True, p=prob)
    out = np.zeros((n, 2), dtype=np.float64)
    for i, idx in enumerate(quad_idx):
        verts4 = quad_vertices[idx]
        a0 = float(area0[idx])
        a1 = float(area1[idx])
        tri = verts4[[0, 1, 2]] if rng.random() < a0 / max(a0 + a1, 1e-30) else verts4[[0, 2, 3]]
        r1 = float(rng.random())
        r2 = float(rng.random())
        if r1 + r2 > 1.0:
            r1 = 1.0 - r1
            r2 = 1.0 - r2
        out[i] = tri[0] + r1 * (tri[1] - tri[0]) + r2 * (tri[2] - tri[0])
    return out


def build_precomputed_arrays(mesh: ParsedMesh, diagnostic_grid_spacing_m: float = 5e-4):
    if mesh.sdim != 2:
        raise ValueError('Current exporter supports only 2D mesh (sdim=2).')
    if 'edg' not in mesh.type_blocks or 'quad' not in mesh.type_blocks:
        raise ValueError('mphtxt must include edg and quad elements.')

    vertices = mesh.vertices
    edg_elements = mesh.type_blocks['edg'].elements
    edge_geo = mesh.type_blocks['edg'].geometric_entity_indices
    quads = _order_quad_vertices(vertices, mesh.type_blocks['quad'].elements)
    # COMSOL `edg` in this export is the boundary representation; keeping it intact
    # preserves the true geometry better than trying to infer boundaries from quad occupancy.
    boundary_edges = vertices[edg_elements].astype(np.float64)
    boundary_part_ids = (edge_geo + 1).astype(np.int32)
    boundary_loops_2d = build_boundary_loops_2d(boundary_edges)
    if not boundary_loops_2d:
        raise ValueError('Could not recover closed boundary loops from COMSOL boundary edges')

    xmin = float(np.min(boundary_edges[:, :, 0]))
    xmax = float(np.max(boundary_edges[:, :, 0]))
    ymin = float(np.min(boundary_edges[:, :, 1]))
    ymax = float(np.max(boundary_edges[:, :, 1]))
    axes_x = _make_uniform_axis(xmin, xmax, float(diagnostic_grid_spacing_m))
    axes_y = _make_uniform_axis(ymin, ymax, float(diagnostic_grid_spacing_m))

    xx, yy = np.meshgrid(axes_x, axes_y, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = points_inside_boundary_loops_2d(points, boundary_loops_2d).reshape(xx.shape)
    edge_start = boundary_edges[:, 0]
    edge_end = boundary_edges[:, 1]
    dist, nearest = _distance_and_nearest_edge(points, edge_start, edge_end)
    dist_grid = dist.reshape(xx.shape)
    nearest_boundary_part_id_map = boundary_part_ids[nearest].reshape(xx.shape).astype(np.int32)
    sdf = np.where(inside, -dist_grid, dist_grid)
    gx, gy = np.gradient(sdf, axes_x, axes_y, edge_order=1)
    loops_flat, loops_offsets = encode_boundary_loops_2d(boundary_loops_2d)

    return {
        'axes_x': axes_x,
        'axes_y': axes_y,
        'sdf': sdf.astype(np.float64),
        'normal_x': gx.astype(np.float64),
        'normal_y': gy.astype(np.float64),
        'inside': inside,
        'nearest_boundary_part_id_map': nearest_boundary_part_id_map,
        'boundary_edges': boundary_edges,
        'boundary_part_ids': boundary_part_ids,
        'boundary_loops_2d': boundary_loops_2d,
        'boundary_loops_2d_flat': loops_flat,
        'boundary_loops_2d_offsets': loops_offsets,
        'boundary_edge_count': int(boundary_edges.shape[0]),
        'vertices': vertices,
        'quads': quads,
    }


def write_case_files(
    mphtxt_path: Path,
    out_dir: Path,
    field_bundle_path: Optional[Path] = None,
    geometry_only: bool = False,
    diagnostic_grid_spacing_m: float = 5e-4,
) -> None:
    if field_bundle_path is None and not geometry_only:
        raise ValueError('COMSOL runnable case requires --field-bundle; use --geometry-only to build geometry only')
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = out_dir / 'generated'
    generated.mkdir(parents=True, exist_ok=True)

    mesh = parse_comsol_mphtxt(mphtxt_path)
    arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=float(diagnostic_grid_spacing_m))
    axes_x = arrays['axes_x']
    axes_y = arrays['axes_y']
    xmin, xmax = float(axes_x[0]), float(axes_x[-1])
    ymin, ymax = float(axes_y[0]), float(axes_y[-1])
    geom_metadata = {
        'provider_kind': 'precomputed_npz',
        'source_kind': 'comsol_mphtxt_geometry',
        'requires_field_bundle': True,
        'has_domain_region_map': False,
        'domain_region_map_status': 'not_implemented',
        'diagnostic_grid_spacing_m': float(diagnostic_grid_spacing_m),
    }

    geom_npz = generated / 'comsol_geometry_2d.npz'
    np.savez_compressed(
        geom_npz,
        axis_0=axes_x,
        axis_1=axes_y,
        sdf=arrays['sdf'],
        normal_0=arrays['normal_x'],
        normal_1=arrays['normal_y'],
        valid_mask=arrays['inside'].astype(bool),
        nearest_boundary_part_id_map=arrays['nearest_boundary_part_id_map'],
        boundary_edges=arrays['boundary_edges'],
        boundary_edge_part_ids=arrays['boundary_part_ids'],
        boundary_loops_2d_flat=arrays['boundary_loops_2d_flat'],
        boundary_loops_2d_offsets=arrays['boundary_loops_2d_offsets'],
        mesh_vertices=mesh.vertices.astype(np.float64),
        mesh_quads=arrays['quads'].astype(np.int32),
        metadata_json=np.asarray(json.dumps(geom_metadata)),
    )
    field_npz = generated / 'comsol_field_2d.npz'
    field_mesh_npz = generated / 'comsol_field_mesh_2d.npz'
    field_summary: Dict[str, Any] = {'mode': 'geometry_only'}
    if field_bundle_path is not None:
        bundle_payload = _validate_field_bundle_payload(_load_npz_payload(field_bundle_path), axes_x, axes_y)
        field_valid_mask = np.asarray(bundle_payload['valid_mask'], dtype=bool) & arrays['inside'].astype(bool)
        field_metadata = {
            'provider_kind': 'precomputed_npz',
            'source_kind': 'comsol_export_bundle_field',
            'geometry_mask_applied': True,
            'bundle_path': str(field_bundle_path.resolve()),
            'has_domain_region_map': False,
        }
        save_payload: Dict[str, np.ndarray] = {
            'axis_0': axes_x.astype(np.float64),
            'axis_1': axes_y.astype(np.float64),
            'times': np.asarray(bundle_payload['times'], dtype=np.float64),
            'valid_mask': field_valid_mask.astype(bool),
            'metadata_json': np.asarray(json.dumps(field_metadata)),
        }
        for key, value in bundle_payload.items():
            if key in {'axis_0', 'axis_1', 'times', 'valid_mask'}:
                continue
            save_payload[key] = _apply_field_valid_mask(value, field_valid_mask)
        np.savez_compressed(field_npz, **save_payload)
        mesh_save_payload = _build_triangle_mesh_field_payload(
            mesh=mesh,
            arrays=arrays,
            bundle_payload=save_payload,
        )
        np.savez_compressed(field_mesh_npz, **mesh_save_payload)
        field_summary = {
            'mode': 'validated_export_bundle',
            'bundle_path': str(field_bundle_path.resolve()),
            'quantities': sorted(k for k in save_payload.keys() if k not in {'axis_0', 'axis_1', 'times', 'valid_mask', 'metadata_json'}),
            'mesh_field_path': str(field_mesh_npz.relative_to(out_dir)),
        }
    elif field_npz.exists() or field_mesh_npz.exists():
        if field_npz.exists():
            field_npz.unlink()
        if field_mesh_npz.exists():
            field_mesh_npz.unlink()

    boundary_parts = np.unique(arrays['boundary_part_ids']).astype(int).tolist()
    mat_rows = pd.DataFrame([
        {
            'material_id': 1,
            'material_name': 'steel',
            'source_law': 'explicit_csv',
            'source_speed_scale': 1.0,
            'wall_law': 'specular',
            'wall_restitution': 0.95,
            'wall_diffuse_fraction': 0.0,
            'wall_stick_probability': 0.0,
        },
        {
            'material_id': 2,
            'material_name': 'ceramic',
            'source_law': 'explicit_csv',
            'source_speed_scale': 1.0,
            'wall_law': 'mixed_specular_diffuse',
            'wall_restitution': 0.90,
            'wall_diffuse_fraction': 0.20,
            'wall_stick_probability': 0.0,
        },
    ])
    mat_rows.to_csv(out_dir / 'materials.csv', index=False)

    wall_rows = []
    for i, pid in enumerate(boundary_parts):
        mat_id = 1 if (i % 2 == 0) else 2
        wall_rows.append(
            {
                'part_id': pid,
                'part_name': f'boundary_{pid}',
                'material_id': mat_id,
                'material_name': 'steel' if mat_id == 1 else 'ceramic',
                'wall_law': 'specular' if mat_id == 1 else 'mixed_specular_diffuse',
                'wall_restitution': 0.95 if mat_id == 1 else 0.9,
                'wall_diffuse_fraction': 0.0 if mat_id == 1 else 0.2,
                'wall_stick_probability': 0.0,
            }
        )
    pd.DataFrame(wall_rows).to_csv(out_dir / 'part_walls.csv', index=False)

    particle_points = _sample_points_in_quads(mesh.vertices, arrays['quads'], count=24, seed=24680)
    particle_rows = []
    for i, pxy in enumerate(particle_points, start=1):
        particle_rows.append(
            {
                'particle_id': i,
                'x': float(pxy[0]),
                'y': float(pxy[1]),
                'vx': 0.0,
                'vy': 0.0,
                'release_time': float((i - 1) * 0.02),
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200.0,
                'charge': 0.0,
                'source_part_id': int(boundary_parts[(i - 1) % len(boundary_parts)]),
                'material_id': 1 if (i % 2 == 1) else 2,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        )
    pd.DataFrame(particle_rows).to_csv(out_dir / 'particles.csv', index=False)

    steps = pd.DataFrame(
        [
            {
                'step_id': 1,
                'step_name': 'ignite',
                'start_s': 0.0,
                'end_s': 0.6,
                'physics_flow_scale': 1.0,
                'physics_drag_tau_scale': 1.0,
                'physics_body_accel_scale': 1.0,
                'wall_mode': 'inherit',
                'wall_restitution': 1.0,
                'wall_diffuse_fraction': 0.0,
                'wall_stick_probability_scale': 1.0,
                'source_law_override': '',
                'source_speed_scale': 1.0,
                'source_event_gain_scale': 1.2,
                'output_segment_name': 'ignite',
                'output_save_every_override': 0,
                'output_save_positions': 1,
                'output_write_wall_events': 1,
                'output_write_diagnostics': 1,
            },
            {
                'step_id': 2,
                'step_name': 'sustain',
                'start_s': 0.6,
                'end_s': 1.2,
                'physics_flow_scale': 0.8,
                'physics_drag_tau_scale': 1.1,
                'physics_body_accel_scale': 1.0,
                'wall_mode': 'mixed_specular_diffuse',
                'wall_restitution': 0.9,
                'wall_diffuse_fraction': 0.2,
                'wall_stick_probability_scale': 1.0,
                'source_law_override': '',
                'source_speed_scale': 1.0,
                'source_event_gain_scale': 0.9,
                'output_segment_name': 'sustain',
                'output_save_every_override': 1,
                'output_save_positions': 1,
                'output_write_wall_events': 1,
                'output_write_diagnostics': 1,
            },
        ]
    )
    steps.to_csv(out_dir / 'process_steps.csv', index=False)

    events = pd.DataFrame(
        [
            {
                'event_id': 1,
                'event_name': 'ignite_boost',
                'event_kind': 'gaussian_burst',
                'enabled': 1,
                'center_s': 0.05,
                'sigma_s': 0.03,
                'amplitude': 0.3,
                'bind_step_name': 'ignite',
                'time_anchor': 'step_start',
            }
        ]
    )
    events.to_csv(out_dir / 'source_events.csv', index=False)

    summary = {
        'source_mphtxt': str(mphtxt_path.resolve()),
        'sdim': int(mesh.sdim),
        'vertex_count': int(mesh.vertices.shape[0]),
        'edge_count': int(mesh.type_blocks['edg'].elements.shape[0]),
        'quad_count': int(mesh.type_blocks['quad'].elements.shape[0]),
        'derived_boundary_edge_count': int(arrays['boundary_edge_count']),
        'boundary_part_ids': boundary_parts,
        'bounds': {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
        },
        'generated_files': {
            'geometry_npz': str(geom_npz.relative_to(out_dir)),
        },
        'grid_axes': {
            'x_count': int(len(arrays['axes_x'])),
            'y_count': int(len(arrays['axes_y'])),
        },
        'geometry_mode': 'boundary_edges_plus_diagnostic_sdf',
        'field_mode': field_summary['mode'],
        'field_summary': field_summary,
        'diagnostic_grid_spacing_m': float(diagnostic_grid_spacing_m),
        'note': 'COMSOL geometry is extracted from mphtxt edges/quads. Field data must come from a validated export bundle.',
    }
    if field_bundle_path is not None:
        config = {
            'run': {
                'spatial_dim': 2,
                'coordinate_system': 'cartesian_xy',
                'time_interpolation': 'linear',
            },
            'paths': {
                'particles_csv': 'particles.csv',
                'materials_csv': 'materials.csv',
                'part_walls_csv': 'part_walls.csv',
                'source_events_csv': 'source_events.csv',
                'process_steps_csv': 'process_steps.csv',
            },
            'providers': {
                'geometry': {
                    'kind': 'precomputed_npz',
                    'npz_path': 'generated/comsol_geometry_2d.npz',
                },
                'field': {
                    'kind': 'precomputed_npz',
                    'npz_path': 'generated/comsol_field_2d.npz',
                },
            },
            'gas': {
                'temperature_K': 320.0,
                'dynamic_viscosity_Pas': 1.8e-5,
                'density_kgm3': 1.2,
            },
            'source': {
                'preprocess': {
                    'enabled': True,
                    'seed': 24680,
                },
                'default_law': 'explicit_csv',
                'source_speed_scale': 1.0,
                'source_position_offset_m': 0.0,
                'direct_wall_shear': {
                    'probe_distance_m': 0.001,
                },
            },
            'process': {
                'step_defaults': {
                    'output': {
                        'segment_name': 'run',
                        'save_every_override': 0,
                    }
                }
            },
            'solver': {
                'dt': 0.01,
                't_end': 1.2,
                'save_every': 5,
                'integrator': 'drag_relaxation',
                'min_tau_p_s': 1e-5,
                'plot_particle_limit': 24,
                'seed': 12345,
            },
        }
        (out_dir / 'run_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        summary['generated_files']['run_config'] = 'run_config.yaml'
        mesh_config = {
            **config,
            'providers': {
                **dict(config['providers']),
                'field': {
                    'kind': 'precomputed_triangle_mesh_npz',
                    'npz_path': 'generated/comsol_field_mesh_2d.npz',
                },
            },
        }
        (out_dir / 'run_config_mesh.yaml').write_text(yaml.safe_dump(mesh_config, sort_keys=False), encoding='utf-8')
        summary['generated_files']['run_config_mesh'] = 'run_config_mesh.yaml'
    if field_bundle_path is not None:
        summary['generated_files']['field_npz'] = str(field_npz.relative_to(out_dir))
        summary['generated_files']['field_mesh_npz'] = str(field_mesh_npz.relative_to(out_dir))
    (generated / 'comsol_case_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='Build a COMSOL-derived particle-tracer case from mphtxt geometry and an optional validated field bundle.')
    ap.add_argument('--mphtxt', type=Path, default=Path('data/argon_gec_ccp_base2.mphtxt'))
    ap.add_argument('--out-dir', type=Path, default=Path('examples/comsol_from_data_2d'))
    ap.add_argument('--field-bundle', type=Path, default=None)
    ap.add_argument('--geometry-only', action='store_true')
    ap.add_argument('--diagnostic-grid-spacing-m', type=float, default=5e-4)
    args = ap.parse_args()
    write_case_files(
        args.mphtxt.resolve(),
        args.out_dir.resolve(),
        field_bundle_path=args.field_bundle.resolve() if args.field_bundle is not None else None,
        geometry_only=bool(args.geometry_only),
        diagnostic_grid_spacing_m=float(args.diagnostic_grid_spacing_m),
    )
    print(f'Wrote COMSOL-derived case to: {args.out_dir.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
