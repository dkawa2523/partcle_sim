from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from particle_tracer_unified.core.coordinate_systems import normalize_coordinate_system
from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN, sample_valid_mask_status
from particle_tracer_unified.core.geometry2d import (
    build_boundary_loops_2d,
    encode_boundary_loops_2d,
    points_inside_boundary_loops_2d,
)
from particle_tracer_unified.core.grid_sampling import sample_grid_scalar

DEFAULT_FIELD_GHOST_CELLS = 8
FIELD_SUPPORT_BOUNDARY_PART_ID = 9001
PHYSICAL_WALL_STICK_PROBABILITY = 0.5

AXIS_SYMMETRY_PART_IDS = frozenset({2, 4, 6, 8, 10})
WAFER_PART_IDS = frozenset({3})
CHAMBER_WALL_PART_IDS = frozenset({12, 32, 36})
SIDEWALL_PART_IDS = frozenset({42, 43, 44, 45})
KNOWN_COMSOL_WALL_PART_IDS = (
    AXIS_SYMMETRY_PART_IDS | WAFER_PART_IDS | CHAMBER_WALL_PART_IDS | SIDEWALL_PART_IDS
)


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


def _scale_mesh_coordinates(mesh: ParsedMesh, scale_m_per_model_unit: float) -> ParsedMesh:
    scale = float(scale_m_per_model_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError('coordinate scale must be a positive finite value')
    if np.isclose(scale, 1.0, atol=0.0, rtol=0.0):
        return mesh
    return ParsedMesh(
        sdim=mesh.sdim,
        vertices=np.asarray(mesh.vertices, dtype=np.float64) * scale,
        type_blocks=mesh.type_blocks,
    )


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


def _surface_triangles_from_mesh(vertices: np.ndarray, blocks: Mapping[str, MeshTypeBlock]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if 'tri' in blocks:
        chunks.append(np.asarray(blocks['tri'].elements, dtype=np.int64))
    if 'quad' in blocks:
        quads = _order_quad_vertices(vertices, blocks['quad'].elements)
        chunks.append(quads[:, [0, 1, 2]].astype(np.int64))
        chunks.append(quads[:, [0, 2, 3]].astype(np.int64))
    if not chunks:
        raise ValueError('mphtxt must include tri or quad elements for 2D domain geometry.')
    return np.vstack(chunks).astype(np.int64)


def _surface_triangle_part_ids_from_mesh(blocks: Mapping[str, MeshTypeBlock]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if 'tri' in blocks:
        chunks.append((np.asarray(blocks['tri'].geometric_entity_indices, dtype=np.int32) + 1).astype(np.int32))
    if 'quad' in blocks:
        quad_ids = (np.asarray(blocks['quad'].geometric_entity_indices, dtype=np.int32) + 1).astype(np.int32)
        chunks.append(quad_ids)
        chunks.append(quad_ids)
    if not chunks:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(chunks).astype(np.int32)


def _domain_boundary_edges_from_surface_elements(
    vertices: np.ndarray,
    blocks: Mapping[str, MeshTypeBlock],
    round_decimals: int = 12,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    edge_counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    edge_vertices: dict[tuple[tuple[float, float], tuple[float, float]], tuple[int, int]] = {}

    def point_key(vertex_id: int) -> tuple[float, float]:
        point = verts[int(vertex_id)]
        return (float(np.round(point[0], round_decimals)), float(np.round(point[1], round_decimals)))

    def add_edge(a: int, b: int) -> None:
        if int(a) == int(b):
            return
        ka = point_key(int(a))
        kb = point_key(int(b))
        key = (ka, kb) if ka <= kb else (kb, ka)
        edge_counts[key] = edge_counts.get(key, 0) + 1
        edge_vertices.setdefault(key, (int(a), int(b)))

    if 'tri' in blocks:
        for elem in np.asarray(blocks['tri'].elements, dtype=np.int64):
            add_edge(int(elem[0]), int(elem[1]))
            add_edge(int(elem[1]), int(elem[2]))
            add_edge(int(elem[2]), int(elem[0]))
    if 'quad' in blocks:
        for elem in _order_quad_vertices(verts, blocks['quad'].elements):
            add_edge(int(elem[0]), int(elem[1]))
            add_edge(int(elem[1]), int(elem[2]))
            add_edge(int(elem[2]), int(elem[3]))
            add_edge(int(elem[3]), int(elem[0]))

    boundary = [edge_vertices[key] for key, count in edge_counts.items() if count == 1]
    if not boundary:
        raise ValueError('Could not derive exterior boundary edges from COMSOL 2D elements.')
    return verts[np.asarray(boundary, dtype=np.int64)].astype(np.float64)


def _assign_part_ids_from_edge_entities(
    vertices: np.ndarray,
    blocks: Mapping[str, MeshTypeBlock],
    boundary_edges: np.ndarray,
) -> np.ndarray:
    if 'edg' not in blocks or blocks['edg'].elements.size == 0:
        return np.ones(boundary_edges.shape[0], dtype=np.int32)
    edge_block = blocks['edg']
    entity_edges = np.asarray(vertices, dtype=np.float64)[np.asarray(edge_block.elements, dtype=np.int64)]
    if entity_edges.shape[0] == 0:
        return np.ones(boundary_edges.shape[0], dtype=np.int32)
    midpoints = 0.5 * (boundary_edges[:, 0, :] + boundary_edges[:, 1, :])
    _dist, nearest = _distance_and_nearest_edge(midpoints, entity_edges[:, 0, :], entity_edges[:, 1, :])
    edge_geo = np.asarray(edge_block.geometric_entity_indices, dtype=np.int64)
    return (edge_geo[np.asarray(nearest, dtype=np.int64)] + 1).astype(np.int32)


def _all_comsol_edge_entity_reference(mesh: ParsedMesh) -> tuple[np.ndarray, np.ndarray]:
    if 'edg' not in mesh.type_blocks or mesh.type_blocks['edg'].elements.size == 0:
        return np.zeros((0, 2, int(mesh.sdim)), dtype=np.float64), np.zeros(0, dtype=np.int32)
    edge_block = mesh.type_blocks['edg']
    edge_elements = np.asarray(edge_block.elements, dtype=np.int64)
    edge_ids = (np.asarray(edge_block.geometric_entity_indices, dtype=np.int32) + 1).astype(np.int32)
    return np.asarray(mesh.vertices, dtype=np.float64)[edge_elements].astype(np.float64), edge_ids


def _field_support_reference_edges(
    mesh: ParsedMesh,
    fallback_boundary_edges: np.ndarray,
    fallback_boundary_part_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    entity_edges, entity_part_ids = _all_comsol_edge_entity_reference(mesh)
    if entity_edges.size > 0 and entity_part_ids.size > 0:
        return entity_edges, entity_part_ids
    return (
        np.asarray(fallback_boundary_edges, dtype=np.float64),
        np.asarray(fallback_boundary_part_ids, dtype=np.int32),
    )


def _axis_spacing(axis: np.ndarray) -> float:
    values = np.asarray(axis, dtype=np.float64)
    diffs = np.diff(values)
    positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if positive.size == 0:
        return 0.0
    return float(np.median(positive))


def _support_boundary_part_ids_from_reference(
    boundary_edges: np.ndarray,
    reference_boundary_edges: Optional[np.ndarray],
    reference_boundary_part_ids: Optional[np.ndarray],
    *,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
) -> np.ndarray:
    if (
        reference_boundary_edges is None
        or reference_boundary_part_ids is None
        or np.asarray(reference_boundary_edges).size == 0
        or np.asarray(reference_boundary_part_ids).size == 0
    ):
        return np.ones(boundary_edges.shape[0], dtype=np.int32)

    ref_edges = np.asarray(reference_boundary_edges, dtype=np.float64)
    ref_part_ids = np.asarray(reference_boundary_part_ids, dtype=np.int32)
    midpoints = 0.5 * (boundary_edges[:, 0, :] + boundary_edges[:, 1, :])
    dist, nearest = _distance_and_nearest_edge(midpoints, ref_edges[:, 0, :], ref_edges[:, 1, :])
    out = ref_part_ids[np.asarray(nearest, dtype=np.int64)].astype(np.int32)

    spacing = max(_axis_spacing(axes_x), _axis_spacing(axes_y), 1.0e-12)
    artificial_exit = np.asarray(dist, dtype=np.float64) > (2.5 * spacing)
    out[artificial_exit] = FIELD_SUPPORT_BOUNDARY_PART_ID
    return out


def _wall_role_for_part_id(part_id: int) -> Dict[str, Any]:
    pid = int(part_id)
    if pid in AXIS_SYMMETRY_PART_IDS:
        return {
            'part_name': f'axis_symmetry_{pid}',
            'material_id': 10,
            'material_name': 'axis_symmetry',
            'wall_law': 'specular',
            'wall_restitution': 1.0,
            'wall_diffuse_fraction': 0.0,
            'wall_stick_probability': 0.0,
        }
    if pid == FIELD_SUPPORT_BOUNDARY_PART_ID:
        return {
            'part_name': 'field_support_boundary',
            'material_id': 90,
            'material_name': 'field_support_boundary',
            'wall_law': 'specular',
            'wall_restitution': 0.95,
            'wall_diffuse_fraction': 0.0,
            'wall_stick_probability': PHYSICAL_WALL_STICK_PROBABILITY,
        }
    if pid in WAFER_PART_IDS:
        role_name = 'wafer'
        material_id = 20
    elif pid in SIDEWALL_PART_IDS:
        role_name = 'sidewall'
        material_id = 50
    elif pid in CHAMBER_WALL_PART_IDS:
        role_name = 'chamber_wall'
        material_id = 40
    else:
        role_name = 'comsol_wall'
        material_id = 99
    return {
        'part_name': f'{role_name}_{pid}',
        'material_id': material_id,
        'material_name': role_name,
        'wall_law': 'specular',
        'wall_restitution': 0.95,
        'wall_diffuse_fraction': 0.0,
        'wall_stick_probability': PHYSICAL_WALL_STICK_PROBABILITY,
    }


def _material_and_wall_rows(boundary_parts: List[int]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, int]]:
    material_by_id: Dict[int, Dict[str, Any]] = {}
    wall_rows: List[Dict[str, Any]] = []
    fallback_materials: Dict[int, int] = {}
    all_parts = sorted({int(p) for p in boundary_parts} | {int(p) for p in KNOWN_COMSOL_WALL_PART_IDS})
    for pid in all_parts:
        role = _wall_role_for_part_id(pid)
        material_id = int(role['material_id'])
        fallback_materials[int(pid)] = material_id
        material_by_id.setdefault(
            material_id,
            {
                'material_id': material_id,
                'material_name': str(role['material_name']),
                'source_law': 'explicit_csv',
                'source_speed_scale': 1.0,
                'wall_law': str(role['wall_law']),
                'wall_restitution': float(role['wall_restitution']),
                'wall_diffuse_fraction': float(role['wall_diffuse_fraction']),
                'wall_stick_probability': float(role['wall_stick_probability']),
            },
        )
        wall_rows.append(
            {
                'part_id': int(pid),
                'part_name': str(role['part_name']),
                'material_id': material_id,
                'material_name': str(role['material_name']),
                'wall_law': str(role['wall_law']),
                'wall_restitution': float(role['wall_restitution']),
                'wall_diffuse_fraction': float(role['wall_diffuse_fraction']),
                'wall_stick_probability': float(role['wall_stick_probability']),
            }
        )
    return list(material_by_id.values()), wall_rows, fallback_materials


def _surface_edge_domain_map(mesh: ParsedMesh) -> Dict[tuple[int, int], set[int]]:
    edge_domains: Dict[tuple[int, int], set[int]] = {}

    def add_edge(a: int, b: int, domain_id: int) -> None:
        key = (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
        edge_domains.setdefault(key, set()).add(int(domain_id))

    tri_block = mesh.type_blocks.get('tri')
    if tri_block is not None and tri_block.elements.size:
        tri_entities = (np.asarray(tri_block.geometric_entity_indices, dtype=np.int32) + 1).astype(np.int32)
        for elem, domain_id in zip(np.asarray(tri_block.elements, dtype=np.int64), tri_entities):
            add_edge(int(elem[0]), int(elem[1]), int(domain_id))
            add_edge(int(elem[1]), int(elem[2]), int(domain_id))
            add_edge(int(elem[2]), int(elem[0]), int(domain_id))

    quad_block = mesh.type_blocks.get('quad')
    if quad_block is not None and quad_block.elements.size:
        quad_entities = (np.asarray(quad_block.geometric_entity_indices, dtype=np.int32) + 1).astype(np.int32)
        for elem, domain_id in zip(_order_quad_vertices(mesh.vertices, quad_block.elements), quad_entities):
            add_edge(int(elem[0]), int(elem[1]), int(domain_id))
            add_edge(int(elem[1]), int(elem[2]), int(domain_id))
            add_edge(int(elem[2]), int(elem[3]), int(domain_id))
            add_edge(int(elem[3]), int(elem[0]), int(domain_id))

    return edge_domains


def _comsol_boundary_entity_rows(mesh: ParsedMesh, active_part_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    edge_block = mesh.type_blocks.get('edg')
    if edge_block is None or edge_block.elements.size == 0:
        return []
    active = {int(pid) for pid in active_part_ids} if active_part_ids is not None else set()
    edge_domains = _surface_edge_domain_map(mesh)
    grouped: Dict[int, Dict[str, Any]] = {}
    edge_ids = (np.asarray(edge_block.geometric_entity_indices, dtype=np.int32) + 1).astype(np.int32)
    edge_elements = np.asarray(edge_block.elements, dtype=np.int64)
    for elem, edge_id in zip(edge_elements, edge_ids):
        pid = int(edge_id)
        coords = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(elem, dtype=np.int64)]
        key = (int(elem[0]), int(elem[1])) if int(elem[0]) <= int(elem[1]) else (int(elem[1]), int(elem[0]))
        row = grouped.setdefault(
            pid,
            {
                'solver_part_id': pid,
                'comsol_edge_entity_id': pid,
                'raw_comsol_edge_entity_index': pid - 1,
                'segment_count': 0,
                'x_min': float('inf'),
                'x_max': float('-inf'),
                'y_min': float('inf'),
                'y_max': float('-inf'),
                'adjacent_domain_ids': set(),
            },
        )
        row['segment_count'] = int(row['segment_count']) + 1
        row['x_min'] = min(float(row['x_min']), float(np.min(coords[:, 0])))
        row['x_max'] = max(float(row['x_max']), float(np.max(coords[:, 0])))
        row['y_min'] = min(float(row['y_min']), float(np.min(coords[:, 1])))
        row['y_max'] = max(float(row['y_max']), float(np.max(coords[:, 1])))
        row['adjacent_domain_ids'].update(edge_domains.get(key, set()))

    rows: List[Dict[str, Any]] = []
    for pid in sorted(grouped):
        row = grouped[pid]
        role = _wall_role_for_part_id(pid)
        adjacent = sorted(int(v) for v in row['adjacent_domain_ids'])
        rows.append(
            {
                'solver_part_id': int(pid),
                'comsol_edge_entity_id': int(row['comsol_edge_entity_id']),
                'raw_comsol_edge_entity_index': int(row['raw_comsol_edge_entity_index']),
                'active_in_solver_boundary': bool((not active) or (pid in active)),
                'segment_count': int(row['segment_count']),
                'x_min_m': float(row['x_min']),
                'x_max_m': float(row['x_max']),
                'y_min_m': float(row['y_min']),
                'y_max_m': float(row['y_max']),
                'adjacent_domain_ids': ';'.join(str(v) for v in adjacent),
                'solver_part_name': str(role['part_name']),
                'solver_material_name': str(role['material_name']),
                'comsol_material_name': 'not_exported_from_mphtxt',
            }
        )
    return rows


def _comsol_domain_entity_rows(mesh: ParsedMesh) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for type_name in ('tri', 'quad'):
        block = mesh.type_blocks.get(type_name)
        if block is None or block.elements.size == 0:
            continue
        domain_ids = (np.asarray(block.geometric_entity_indices, dtype=np.int32) + 1).astype(np.int32)
        elements = np.asarray(block.elements, dtype=np.int64)
        if type_name == 'quad':
            elements = _order_quad_vertices(mesh.vertices, elements)
        for elem, domain_id in zip(elements, domain_ids):
            did = int(domain_id)
            coords = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(elem, dtype=np.int64)]
            row = grouped.setdefault(
                did,
                {
                    'comsol_domain_entity_id': did,
                    'raw_comsol_domain_entity_index': did - 1,
                    'element_count': 0,
                    'mesh_element_types': set(),
                    'x_min': float('inf'),
                    'x_max': float('-inf'),
                    'y_min': float('inf'),
                    'y_max': float('-inf'),
                },
            )
            row['element_count'] = int(row['element_count']) + 1
            row['mesh_element_types'].add(type_name)
            row['x_min'] = min(float(row['x_min']), float(np.min(coords[:, 0])))
            row['x_max'] = max(float(row['x_max']), float(np.max(coords[:, 0])))
            row['y_min'] = min(float(row['y_min']), float(np.min(coords[:, 1])))
            row['y_max'] = max(float(row['y_max']), float(np.max(coords[:, 1])))

    rows: List[Dict[str, Any]] = []
    for did in sorted(grouped):
        row = grouped[did]
        rows.append(
            {
                'comsol_domain_entity_id': int(did),
                'raw_comsol_domain_entity_index': int(row['raw_comsol_domain_entity_index']),
                'element_count': int(row['element_count']),
                'mesh_element_types': ';'.join(sorted(str(v) for v in row['mesh_element_types'])),
                'x_min_m': float(row['x_min']),
                'x_max_m': float(row['x_max']),
                'y_min_m': float(row['y_min']),
                'y_max_m': float(row['y_max']),
                'comsol_material_name': 'not_exported_from_mphtxt',
            }
        )
    return rows


def _write_comsol_entity_maps(generated_dir: Path, mesh: ParsedMesh, active_part_ids: List[int]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    boundary_rows = _comsol_boundary_entity_rows(mesh, active_part_ids=active_part_ids)
    if boundary_rows:
        path = generated_dir / 'comsol_boundary_entity_mapping.csv'
        pd.DataFrame(boundary_rows).to_csv(path, index=False)
        out['comsol_boundary_entity_mapping'] = path.name
    domain_rows = _comsol_domain_entity_rows(mesh)
    if domain_rows:
        path = generated_dir / 'comsol_domain_entity_mapping.csv'
        pd.DataFrame(domain_rows).to_csv(path, index=False)
        out['comsol_domain_entity_mapping'] = path.name
    return out


def _boundary_edges_from_valid_cells(
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    x = np.asarray(axes_x, dtype=np.float64)
    y = np.asarray(axes_y, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != (x.size, y.size):
        raise ValueError('valid_mask shape must match axes when deriving field-support geometry')
    if x.size < 2 or y.size < 2:
        raise ValueError('field-support geometry requires at least two nodes per axis')
    valid_cells = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    edge_counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    edge_coords: dict[tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]] = {}

    def add_edge(a: tuple[float, float], b: tuple[float, float]) -> None:
        key = (a, b) if a <= b else (b, a)
        edge_counts[key] = edge_counts.get(key, 0) + 1
        edge_coords.setdefault(key, (a, b))

    for i, j in np.argwhere(valid_cells):
        x0 = float(x[int(i)])
        x1 = float(x[int(i) + 1])
        y0 = float(y[int(j)])
        y1 = float(y[int(j) + 1])
        add_edge((x0, y0), (x1, y0))
        add_edge((x1, y0), (x1, y1))
        add_edge((x1, y1), (x0, y1))
        add_edge((x0, y1), (x0, y0))
    boundary = [edge_coords[key] for key, count in edge_counts.items() if count == 1]
    if not boundary:
        raise ValueError('Could not derive field-support boundary from valid_mask')
    return np.asarray(boundary, dtype=np.float64)


def _geometry_arrays_from_field_support(
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    valid_mask: np.ndarray,
    *,
    reference_boundary_edges: Optional[np.ndarray] = None,
    reference_boundary_part_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    x = np.asarray(axes_x, dtype=np.float64)
    y = np.asarray(axes_y, dtype=np.float64)
    boundary_edges = _boundary_edges_from_valid_cells(x, y, valid_mask)
    boundary_part_ids = _support_boundary_part_ids_from_reference(
        boundary_edges,
        reference_boundary_edges,
        reference_boundary_part_ids,
        axes_x=x,
        axes_y=y,
    )
    boundary_loops_2d = build_boundary_loops_2d(boundary_edges)
    if not boundary_loops_2d:
        raise ValueError('Could not recover closed field-support boundary loops')
    xx, yy = np.meshgrid(x, y, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = points_inside_boundary_loops_2d(points, boundary_loops_2d).reshape(xx.shape)
    dist, nearest = _distance_and_nearest_edge(points, boundary_edges[:, 0], boundary_edges[:, 1])
    dist_grid = dist.reshape(xx.shape)
    sdf = np.where(inside, -dist_grid, dist_grid)
    gx, gy = np.gradient(sdf, x, y, edge_order=1)
    loops_flat, loops_offsets = encode_boundary_loops_2d(boundary_loops_2d)
    return {
        'axes_x': x,
        'axes_y': y,
        'sdf': sdf.astype(np.float64),
        'normal_x': gx.astype(np.float64),
        'normal_y': gy.astype(np.float64),
        'inside': inside,
        'nearest_boundary_part_id_map': boundary_part_ids[np.asarray(nearest, dtype=np.int64)].reshape(xx.shape).astype(np.int32),
        'boundary_edges': boundary_edges,
        'boundary_part_ids': boundary_part_ids,
        'boundary_loops_2d': boundary_loops_2d,
        'boundary_loops_2d_flat': loops_flat,
        'boundary_loops_2d_offsets': loops_offsets,
        'boundary_edge_count': int(boundary_edges.shape[0]),
    }


def _sample_points_in_triangles(vertices: np.ndarray, triangles: np.ndarray, count: int, seed: int = 12345) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    n = int(max(0, count))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    tri_vertices = v[tri]
    areas = np.asarray([_triangle_area(t[0], t[1], t[2]) for t in tri_vertices], dtype=np.float64)
    positive = np.flatnonzero(areas > 0.0)
    if positive.size == 0:
        return tri_vertices.mean(axis=1)[:n].copy()
    prob = areas[positive] / max(float(areas[positive].sum()), 1e-30)
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(positive, size=n, replace=True, p=prob)
    out = np.zeros((n, 2), dtype=np.float64)
    for i, idx in enumerate(picked):
        verts3 = tri_vertices[int(idx)]
        r1 = float(rng.random())
        r2 = float(rng.random())
        if r1 + r2 > 1.0:
            r1 = 1.0 - r1
            r2 = 1.0 - r2
        out[i] = verts3[0] + r1 * (verts3[1] - verts3[0]) + r2 * (verts3[2] - verts3[0])
    return out


def _sample_clean_field_points_in_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    valid_mask: np.ndarray,
    count: int,
    seed: int = 12345,
) -> np.ndarray:
    n = int(max(0, count))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    accepted: List[np.ndarray] = []
    batch = int(max(128, 4 * n))
    mask = np.asarray(valid_mask, dtype=bool)
    axes = (np.asarray(axes_x, dtype=np.float64), np.asarray(axes_y, dtype=np.float64))
    for attempt in range(50):
        candidates = _sample_points_in_triangles(vertices, triangles, batch, seed=int(seed) + attempt)
        clean = [
            p
            for p in candidates
            if int(sample_valid_mask_status(mask, axes, np.asarray(p, dtype=np.float64))) == int(VALID_MASK_STATUS_CLEAN)
        ]
        if clean:
            accepted.append(np.asarray(clean, dtype=np.float64))
        accepted_count = int(sum(arr.shape[0] for arr in accepted))
        if accepted_count >= n:
            return np.vstack(accepted)[:n].copy()
    raise ValueError(
        'Could not sample enough particles inside the clean field sample domain; '
        'check field.valid_mask coverage or reduce the requested particle count'
    )


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
    span = max(0.0, hi - lo)
    intervals = span / dx if dx > 0.0 else 1.0
    rounded_intervals = int(round(intervals))
    if abs(intervals - rounded_intervals) <= 1.0e-9 * max(1.0, abs(intervals)):
        count = rounded_intervals + 1
    else:
        count = int(np.ceil(intervals)) + 1
    count = max(2, count)
    axis = np.linspace(lo, hi, count, dtype=np.float64)
    return axis


def _pad_axis(axis: np.ndarray, ghost_cells: int) -> np.ndarray:
    arr = np.asarray(axis, dtype=np.float64)
    cells = int(max(0, ghost_cells))
    if cells == 0:
        return arr.copy()
    diffs = np.diff(arr)
    finite = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if finite.size == 0:
        raise ValueError('Cannot pad a field axis without positive spacing')
    step = float(np.median(finite))
    left = arr[0] - step * np.arange(cells, 0, -1, dtype=np.float64)
    right = arr[-1] + step * np.arange(1, cells + 1, dtype=np.float64)
    return np.concatenate([left, arr, right]).astype(np.float64)


def _pad_field_array(data: np.ndarray, ghost_cells: int) -> np.ndarray:
    cells = int(max(0, ghost_cells))
    arr = np.asarray(data)
    if cells == 0:
        return arr.copy()
    if arr.ndim == 2:
        return np.pad(arr, ((cells, cells), (cells, cells)), mode='edge')
    if arr.ndim == 3:
        return np.pad(arr, ((0, 0), (cells, cells), (cells, cells)), mode='edge')
    return arr.copy()


def _pad_field_bundle_payload(
    payload: Mapping[str, np.ndarray],
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    ghost_cells: int,
) -> Dict[str, np.ndarray]:
    cells = int(max(0, ghost_cells))
    out: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key == 'axis_0':
            out[key] = np.asarray(axes_x, dtype=np.float64)
        elif key == 'axis_1':
            out[key] = np.asarray(axes_y, dtype=np.float64)
        elif key in {'times', 'metadata_json'}:
            out[key] = np.asarray(value).copy()
        elif key == 'valid_mask' and cells > 0:
            out[key] = np.pad(np.asarray(value, dtype=bool), ((cells, cells), (cells, cells)), mode='constant', constant_values=False)
        else:
            out[key] = _pad_field_array(np.asarray(value), cells)
    return out


def _load_npz_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _axes_match(source: np.ndarray | None, target: np.ndarray) -> bool:
    if source is None:
        return True
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    return bool(src.shape == dst.shape and np.allclose(src, dst, atol=1e-12, rtol=0.0))


def _interp_field_2d_to_axes(data: np.ndarray, src_x: np.ndarray, src_y: np.ndarray, dst_x: np.ndarray, dst_y: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    tmp = np.empty((dst_x.size, src_y.size), dtype=np.float64)
    for j in range(src_y.size):
        tmp[:, j] = np.interp(dst_x, src_x, arr[:, j])
    out = np.empty((dst_x.size, dst_y.size), dtype=np.float64)
    for i in range(dst_x.size):
        out[i, :] = np.interp(dst_y, src_y, tmp[i, :])
    return out


def _resample_field_to_geometry_axes(data: np.ndarray, src_x: np.ndarray, src_y: np.ndarray, dst_x: np.ndarray, dst_y: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        return _interp_field_2d_to_axes(arr, src_x, src_y, dst_x, dst_y)
    if arr.ndim == 3:
        return np.stack(
            [_interp_field_2d_to_axes(arr[i], src_x, src_y, dst_x, dst_y) for i in range(arr.shape[0])],
            axis=0,
        )
    raise ValueError(f'field bundle quantity must be 2D or 3D, got shape {arr.shape}')


def _same_axis_extent(source: np.ndarray, target: np.ndarray) -> bool:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    return bool(
        src.ndim == 1
        and dst.ndim == 1
        and src.size == dst.size
        and src.size > 0
        and np.isclose(src[0], dst[0], atol=1e-12, rtol=0.0)
        and np.isclose(src[-1], dst[-1], atol=1e-12, rtol=0.0)
    )


def _validate_field_bundle_payload(payload: Mapping[str, np.ndarray], axes_x: np.ndarray, axes_y: np.ndarray) -> Dict[str, np.ndarray]:
    if 'ux' not in payload or 'uy' not in payload:
        raise ValueError('field bundle must include ux and uy')
    bundle_axes_x = np.asarray(payload['axis_0'], dtype=np.float64) if 'axis_0' in payload else None
    bundle_axes_y = np.asarray(payload['axis_1'], dtype=np.float64) if 'axis_1' in payload else None
    resample_to_geometry_axes = not (_axes_match(bundle_axes_x, axes_x) and _axes_match(bundle_axes_y, axes_y))
    if resample_to_geometry_axes and (bundle_axes_x is None or bundle_axes_y is None):
        raise ValueError('field bundle axes are required when resampling to geometry axes')
    if resample_to_geometry_axes and not _same_axis_extent(bundle_axes_x, axes_x):
        raise ValueError('field bundle axis_0 must share geometry axis_0 extent to be resampled')
    if resample_to_geometry_axes and not _same_axis_extent(bundle_axes_y, axes_y):
        raise ValueError('field bundle axis_1 must share geometry axis_1 extent to be resampled')
    expected_shape = (axes_x.size, axes_y.size)
    source_shape = expected_shape if bundle_axes_x is None or bundle_axes_y is None else (bundle_axes_x.size, bundle_axes_y.size)
    times = np.asarray(payload['times'], dtype=np.float64) if 'times' in payload else np.asarray([0.0], dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError('field bundle times must be a non-empty 1D array when provided')
    normalized: Dict[str, np.ndarray] = {
        'axis_0': axes_x.astype(np.float64),
        'axis_1': axes_y.astype(np.float64),
        'times': times.astype(np.float64),
    }
    valid_mask = np.asarray(payload['valid_mask'], dtype=bool) if 'valid_mask' in payload else np.ones(source_shape, dtype=bool)
    if resample_to_geometry_axes:
        valid_mask = _resample_field_to_geometry_axes(valid_mask.astype(np.float64), bundle_axes_x, bundle_axes_y, axes_x, axes_y) >= 0.5
    elif valid_mask.shape != expected_shape:
        raise ValueError(f'field bundle valid_mask must match geometry grid shape {expected_shape}')
    normalized['valid_mask'] = valid_mask
    if 'support_phi' in payload:
        support_phi = np.asarray(payload['support_phi'], dtype=np.float64)
        if resample_to_geometry_axes:
            support_phi = _resample_field_to_geometry_axes(support_phi, bundle_axes_x, bundle_axes_y, axes_x, axes_y)
        elif support_phi.shape != expected_shape:
            raise ValueError(f'field bundle support_phi must match geometry grid shape {expected_shape}')
        normalized['support_phi'] = support_phi
    reserved = {'axis_0', 'axis_1', 'times', 'valid_mask', 'support_phi', 'metadata_json'}
    for key, value in payload.items():
        if key in reserved:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if resample_to_geometry_axes:
            arr = _resample_field_to_geometry_axes(arr, bundle_axes_x, bundle_axes_y, axes_x, axes_y)
        elif arr.ndim == 2:
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


def _field_quantity_keys(payload: Mapping[str, np.ndarray]) -> List[str]:
    reserved = {'axis_0', 'axis_1', 'times', 'valid_mask', 'support_phi', 'metadata_json'}
    return [
        str(key)
        for key, value in payload.items()
        if key not in reserved and np.asarray(value).ndim in {2, 3}
    ]


def _finite_field_support_mask(payload: Mapping[str, np.ndarray], expected_shape: tuple[int, int]) -> np.ndarray:
    quantity_keys = _field_quantity_keys(payload)
    if not quantity_keys:
        raise ValueError('field bundle must include at least one finite field quantity')
    support = np.ones(expected_shape, dtype=bool)
    for key in quantity_keys:
        arr = np.asarray(payload[key], dtype=np.float64)
        finite = np.all(np.isfinite(arr), axis=0) if arr.ndim == 3 else np.isfinite(arr)
        if finite.shape != expected_shape:
            raise ValueError(f'field bundle quantity {key} must match geometry grid shape {expected_shape}')
        support &= finite
    return support


def _support_phi_quality_summary(support_phi: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
    phi = np.asarray(support_phi, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if phi.shape != mask.shape:
        raise ValueError(f'support_phi shape mismatch: expected {mask.shape}, got {phi.shape}')
    finite = np.isfinite(phi)
    finite_values = phi[finite]
    inside = mask & finite
    outside = (~mask) & finite
    summary: Dict[str, Any] = {
        'grid_node_count': int(phi.size),
        'finite_node_count': int(np.count_nonzero(finite)),
        'nonfinite_node_count': int(phi.size - np.count_nonzero(finite)),
        'positive_node_count': int(np.count_nonzero(finite_values > 0.0)),
        'zero_node_count': int(np.count_nonzero(finite_values == 0.0)),
        'negative_node_count': int(np.count_nonzero(finite_values < 0.0)),
        'valid_mask_inside_node_count': int(np.count_nonzero(mask)),
        'inside_nonpositive_count': int(np.count_nonzero(inside & (phi <= 0.0))),
        'outside_positive_count': int(np.count_nonzero(outside & (phi > 0.0))),
    }
    if finite_values.size:
        summary.update(
            {
                'min': float(np.min(finite_values)),
                'max': float(np.max(finite_values)),
                'mean': float(np.mean(finite_values)),
            }
        )
    return summary


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


def _sample_clean_field_points_in_quads(
    vertices: np.ndarray,
    quads: np.ndarray,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    valid_mask: np.ndarray,
    count: int,
    seed: int = 12345,
) -> np.ndarray:
    n = int(max(0, count))
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    accepted: List[np.ndarray] = []
    batch = int(max(128, 4 * n))
    mask = np.asarray(valid_mask, dtype=bool)
    axes = (np.asarray(axes_x, dtype=np.float64), np.asarray(axes_y, dtype=np.float64))
    for attempt in range(50):
        candidates = _sample_points_in_quads(vertices, quads, batch, seed=int(seed) + attempt)
        clean = [
            p
            for p in candidates
            if int(sample_valid_mask_status(mask, axes, np.asarray(p, dtype=np.float64))) == int(VALID_MASK_STATUS_CLEAN)
        ]
        if clean:
            accepted.append(np.asarray(clean, dtype=np.float64))
        accepted_count = int(sum(arr.shape[0] for arr in accepted))
        if accepted_count >= n:
            return np.vstack(accepted)[:n].copy()
    raise ValueError(
        'Could not sample enough particles inside the clean field sample domain; '
        'check field.valid_mask coverage or reduce the requested particle count'
    )


def _sample_clean_boundary_release_points(
    arrays: Mapping[str, Any],
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    valid_mask: np.ndarray,
    count: int,
    seed: int = 24680,
    min_release_offset_cells: float = 1.0,
    source_part_ids: Optional[List[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(max(0, count))
    if n == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty.copy(), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32)
    edges = np.asarray(arrays['boundary_edges'], dtype=np.float64)
    part_ids = np.asarray(arrays['boundary_part_ids'], dtype=np.int32)
    edge_vec = edges[:, 1, :] - edges[:, 0, :]
    lengths = np.linalg.norm(edge_vec, axis=1)
    valid_edges = np.flatnonzero(lengths > 1.0e-30)
    if source_part_ids:
        allowed = {int(pid) for pid in source_part_ids}
        valid_edges = valid_edges[np.isin(part_ids[valid_edges], np.asarray(sorted(allowed), dtype=np.int32))]
    if valid_edges.size == 0:
        raise ValueError('boundary release generation requires non-degenerate boundary edges for the selected source parts')
    weights = lengths[valid_edges] / float(np.sum(lengths[valid_edges]))
    cell_diag = _max_cell_diagonal(axes_x, axes_y)
    base_multipliers = np.asarray([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], dtype=np.float64)
    min_multiplier = float(max(0.0, min_release_offset_cells))
    multipliers = base_multipliers[base_multipliers >= min_multiplier]
    if multipliers.size == 0:
        multipliers = np.asarray([min_multiplier], dtype=np.float64)
    elif min_multiplier > 0.0 and not np.any(np.isclose(multipliers, min_multiplier, rtol=0.0, atol=1.0e-12)):
        multipliers = np.unique(np.concatenate([np.asarray([min_multiplier], dtype=np.float64), multipliers]))
    probe_distances = cell_diag * multipliers
    mask = np.asarray(valid_mask, dtype=bool)
    axes = (np.asarray(axes_x, dtype=np.float64), np.asarray(axes_y, dtype=np.float64))
    rng = np.random.default_rng(int(seed))
    release_points: List[np.ndarray] = []
    source_points: List[np.ndarray] = []
    offsets: List[float] = []
    source_parts: List[int] = []
    max_attempts = int(max(1000, 30 * n))
    for _attempt in range(max_attempts):
        edge_idx = int(rng.choice(valid_edges, p=weights))
        alpha = float(rng.random())
        source = edges[edge_idx, 0, :] + alpha * edge_vec[edge_idx]
        normal = _sample_grid_normal(arrays['normal_x'], arrays['normal_y'], axes_x, axes_y, source)
        placed = False
        for distance in probe_distances:
            for direction in (-normal, normal):
                candidate = source + direction * float(distance)
                status = int(sample_valid_mask_status(mask, axes, candidate))
                if status != int(VALID_MASK_STATUS_CLEAN):
                    continue
                release_points.append(candidate.astype(np.float64))
                source_points.append(source.astype(np.float64))
                offsets.append(float(np.linalg.norm(candidate - source)))
                source_parts.append(int(part_ids[edge_idx]))
                placed = True
                break
            if placed:
                break
        if len(release_points) >= n:
            return (
                np.vstack(release_points[:n]).astype(np.float64),
                np.vstack(source_points[:n]).astype(np.float64),
                np.asarray(offsets[:n], dtype=np.float64),
                np.asarray(source_parts[:n], dtype=np.int32),
            )
    raise ValueError(
        'Could not place enough boundary-release particles inside the clean field sample domain; '
        'the rectilinear field support may be too far from the requested source boundary'
    )


def _nearest_boundary_part_ids_for_points(arrays: Mapping[str, Any], points: np.ndarray) -> np.ndarray:
    boundary_edges = np.asarray(arrays['boundary_edges'], dtype=np.float64)
    boundary_part_ids = np.asarray(arrays['boundary_part_ids'], dtype=np.int32)
    _, nearest_edge = _distance_and_nearest_edge(
        np.asarray(points, dtype=np.float64),
        boundary_edges[:, 0, :],
        boundary_edges[:, 1, :],
    )
    return boundary_part_ids[np.asarray(nearest_edge, dtype=np.int64)]


def _material_ids_for_parts(out_dir: Path, source_part_ids: np.ndarray, fallback_material_ids: Mapping[int, int]) -> np.ndarray:
    part_to_material = {int(k): int(v) for k, v in fallback_material_ids.items()}
    walls_path = Path(out_dir) / 'part_walls.csv'
    if walls_path.exists():
        walls = pd.read_csv(walls_path)
        if {'part_id', 'material_id'}.issubset(walls.columns):
            part_to_material.update({int(row.part_id): int(row.material_id) for row in walls.itertuples(index=False)})
    return np.asarray([part_to_material.get(int(part_id), 1) for part_id in source_part_ids], dtype=np.int32)


def _infer_axis_padding(base_axis: np.ndarray, target_axis: np.ndarray, max_cells: int = 16) -> int:
    base = np.asarray(base_axis, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    if base.shape == target.shape and np.allclose(base, target, atol=1.0e-12, rtol=0.0):
        return 0
    for cells in range(1, int(max_cells) + 1):
        if target.size == base.size + 2 * cells and np.allclose(target[cells:-cells], base, atol=1.0e-12, rtol=0.0):
            return int(cells)
    raise ValueError('existing field axes do not match COMSOL geometry axes or a supported ghost-cell padding')


def _parse_part_id_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not str(value).strip():
        return None
    out: List[int] = []
    for chunk in str(value).replace(';', ',').split(','):
        text = chunk.strip()
        if not text:
            continue
        out.append(int(text))
    return out or None


def _particle_rows_from_points(
    points: np.ndarray,
    *,
    source_part_ids: np.ndarray,
    material_ids: np.ndarray,
    release_span_s: Optional[float],
    source_points: Optional[np.ndarray] = None,
    release_offsets_m: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    p = np.asarray(points, dtype=np.float64)
    count = int(p.shape[0])
    if count <= 1:
        release_times = np.zeros(count, dtype=np.float64)
    else:
        span = float(0.02 * (count - 1) if release_span_s is None else release_span_s)
        release_times = np.linspace(0.0, span, count, dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    src = None if source_points is None else np.asarray(source_points, dtype=np.float64)
    offsets = None if release_offsets_m is None else np.asarray(release_offsets_m, dtype=np.float64)
    for i, pxy in enumerate(p, start=1):
        row = {
            'particle_id': i,
            'x': float(pxy[0]),
            'y': float(pxy[1]),
            'vx': 0.0,
            'vy': 0.0,
            'release_time': float(release_times[i - 1]),
            'mass': 1e-15,
            'diameter': 1e-6,
            'density': 1200.0,
            'charge': 0.0,
            'source_part_id': int(source_part_ids[i - 1]),
            'material_id': int(material_ids[i - 1]),
            'source_event_tag': '',
            'stick_probability': 0.0,
        }
        if src is not None and offsets is not None:
            row.update(
                {
                    'source_x': float(src[i - 1, 0]),
                    'source_y': float(src[i - 1, 1]),
                    'release_offset_m': float(offsets[i - 1]),
                }
            )
        rows.append(row)
    return rows


def write_particles_for_case(
    mphtxt_path: Path,
    out_dir: Path,
    *,
    particle_count: int,
    release_span_s: Optional[float] = None,
    seed: int = 24680,
    min_release_offset_cells: float = 1.0,
    diagnostic_grid_spacing_m: float = 5e-4,
    coordinate_scale_m_per_model_unit: float = 1.0,
    source_part_ids: Optional[List[int]] = None,
) -> None:
    mesh = _scale_mesh_coordinates(parse_comsol_mphtxt(mphtxt_path), float(coordinate_scale_m_per_model_unit))
    base_arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=float(diagnostic_grid_spacing_m))
    field_npz = Path(out_dir) / 'generated' / 'comsol_field_2d.npz'
    if not field_npz.exists():
        raise FileNotFoundError(f'clean particle generation requires existing field bundle output: {field_npz}')
    with np.load(field_npz) as payload:
        axes_x = np.asarray(payload['axis_0'], dtype=np.float64)
        axes_y = np.asarray(payload['axis_1'], dtype=np.float64)
        valid_mask = np.asarray(payload['valid_mask'], dtype=bool)
    pad_x = _infer_axis_padding(np.asarray(base_arrays['axes_x'], dtype=np.float64), axes_x)
    pad_y = _infer_axis_padding(np.asarray(base_arrays['axes_y'], dtype=np.float64), axes_y)
    if int(pad_x) != int(pad_y):
        raise ValueError('existing field axes use inconsistent x/y ghost-cell padding')
    arrays = build_precomputed_arrays(
        mesh,
        diagnostic_grid_spacing_m=float(diagnostic_grid_spacing_m),
        grid_padding_cells=int(pad_x),
    )
    reference_edges, reference_part_ids = _field_support_reference_edges(
        mesh,
        np.asarray(arrays['boundary_edges'], dtype=np.float64),
        np.asarray(arrays['boundary_part_ids'], dtype=np.int32),
    )
    support_arrays = _geometry_arrays_from_field_support(
        axes_x,
        axes_y,
        valid_mask,
        reference_boundary_edges=reference_edges,
        reference_boundary_part_ids=reference_part_ids,
    )
    support_arrays['vertices'] = arrays['vertices']
    support_arrays['quads'] = arrays['quads']
    support_arrays['quad_part_ids'] = arrays['quad_part_ids']
    support_arrays['triangles'] = arrays['triangles']
    support_arrays['triangle_part_ids'] = arrays['triangle_part_ids']
    arrays = {**arrays, **support_arrays}

    points, source_points, offsets, source_part_ids = _sample_clean_boundary_release_points(
        arrays,
        axes_x,
        axes_y,
        valid_mask,
        count=int(particle_count),
        seed=int(seed),
        min_release_offset_cells=float(min_release_offset_cells),
        source_part_ids=source_part_ids,
    )
    boundary_parts = np.unique(arrays['boundary_part_ids']).astype(int).tolist()
    _material_rows, _wall_rows, fallback_materials = _material_and_wall_rows(boundary_parts)
    material_ids = _material_ids_for_parts(out_dir, source_part_ids, fallback_materials)
    rows = _particle_rows_from_points(
        points,
        source_part_ids=source_part_ids,
        material_ids=material_ids,
        release_span_s=release_span_s,
        source_points=source_points,
        release_offsets_m=offsets,
    )
    pd.DataFrame(rows).to_csv(Path(out_dir) / 'particles.csv', index=False)


def build_precomputed_arrays(mesh: ParsedMesh, diagnostic_grid_spacing_m: float = 5e-4, grid_padding_cells: int = 0):
    if mesh.sdim != 2:
        raise ValueError('Current exporter supports only 2D mesh (sdim=2).')
    if 'tri' not in mesh.type_blocks and 'quad' not in mesh.type_blocks:
        raise ValueError('mphtxt must include tri or quad elements.')

    vertices = mesh.vertices
    surface_triangles = _surface_triangles_from_mesh(vertices, mesh.type_blocks)
    surface_triangle_part_ids = _surface_triangle_part_ids_from_mesh(mesh.type_blocks)
    quads = (
        _order_quad_vertices(vertices, mesh.type_blocks['quad'].elements)
        if 'quad' in mesh.type_blocks
        else np.zeros((0, 4), dtype=np.int64)
    )
    quad_part_ids = (
        (np.asarray(mesh.type_blocks['quad'].geometric_entity_indices, dtype=np.int32) + 1).astype(np.int32)
        if 'quad' in mesh.type_blocks
        else np.zeros(0, dtype=np.int32)
    )
    boundary_edges = _domain_boundary_edges_from_surface_elements(vertices, mesh.type_blocks)
    boundary_part_ids = _assign_part_ids_from_edge_entities(vertices, mesh.type_blocks, boundary_edges)
    boundary_loops_2d = build_boundary_loops_2d(boundary_edges)
    if not boundary_loops_2d:
        raise ValueError('Could not recover closed boundary loops from COMSOL boundary edges')

    xmin = float(np.min(boundary_edges[:, :, 0]))
    xmax = float(np.max(boundary_edges[:, :, 0]))
    ymin = float(np.min(boundary_edges[:, :, 1]))
    ymax = float(np.max(boundary_edges[:, :, 1]))
    axes_x = _pad_axis(_make_uniform_axis(xmin, xmax, float(diagnostic_grid_spacing_m)), int(grid_padding_cells))
    axes_y = _pad_axis(_make_uniform_axis(ymin, ymax, float(diagnostic_grid_spacing_m)), int(grid_padding_cells))

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
        'quad_part_ids': quad_part_ids,
        'triangles': surface_triangles,
        'triangle_part_ids': surface_triangle_part_ids,
    }


def _write_geometry_npz(
    path: Path,
    *,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    arrays: Mapping[str, Any],
    mesh: ParsedMesh,
    metadata: Mapping[str, Any],
) -> None:
    np.savez_compressed(
        path,
        axis_0=np.asarray(axes_x, dtype=np.float64),
        axis_1=np.asarray(axes_y, dtype=np.float64),
        sdf=np.asarray(arrays['sdf'], dtype=np.float64),
        normal_0=np.asarray(arrays['normal_x'], dtype=np.float64),
        normal_1=np.asarray(arrays['normal_y'], dtype=np.float64),
        valid_mask=np.asarray(arrays['inside'], dtype=bool),
        nearest_boundary_part_id_map=np.asarray(arrays['nearest_boundary_part_id_map'], dtype=np.int32),
        boundary_edges=np.asarray(arrays['boundary_edges'], dtype=np.float64),
        boundary_edge_part_ids=np.asarray(arrays['boundary_part_ids'], dtype=np.int32),
        boundary_loops_2d_flat=np.asarray(arrays['boundary_loops_2d_flat'], dtype=np.float64),
        boundary_loops_2d_offsets=np.asarray(arrays['boundary_loops_2d_offsets'], dtype=np.int32),
        mesh_vertices=mesh.vertices.astype(np.float64),
        mesh_triangles=np.asarray(arrays.get('triangles', np.zeros((0, 3), dtype=np.int64)), dtype=np.int32),
        mesh_triangle_part_ids=np.asarray(arrays.get('triangle_part_ids', np.zeros(0, dtype=np.int32)), dtype=np.int32),
        mesh_quads=np.asarray(arrays.get('quads', np.zeros((0, 4), dtype=np.int64)), dtype=np.int32),
        mesh_quad_part_ids=np.asarray(arrays.get('quad_part_ids', np.zeros(0, dtype=np.int32)), dtype=np.int32),
        metadata_json=np.asarray(json.dumps(dict(metadata))),
    )


def _write_generated_provider_contract(config_path: Path, generated_dir: Path, out_dir: Path) -> Dict[str, Any]:
    from particle_tracer_unified.core.provider_contract import write_provider_contract_report
    from particle_tracer_unified.io.runtime_builder import build_prepared_runtime_from_yaml

    prepared = build_prepared_runtime_from_yaml(config_path)
    report = write_provider_contract_report(prepared, generated_dir)
    counts = report.get('status_counts', {}) if isinstance(report.get('status_counts', {}), dict) else {}
    summary = {
        'passed': bool(report.get('passed', True)),
        'applicable': bool(report.get('applicable', False)),
        'field_backend_kind': str(report.get('field_backend_kind', '')),
        'sample_count': int(report.get('sample_count', 0)),
        'non_clean': int(counts.get('non_clean', 0)),
        'mixed_stencil': int(counts.get('mixed_stencil', 0)),
        'hard_invalid': int(counts.get('hard_invalid', 0)),
        'report_path': (generated_dir / 'provider_contract_report.json').relative_to(out_dir).as_posix(),
    }
    violations_path = generated_dir / 'provider_boundary_violations.csv'
    if violations_path.exists():
        summary['violations_path'] = violations_path.relative_to(out_dir).as_posix()
    summary_path = generated_dir / 'provider_boundary_summary.csv'
    if summary_path.exists():
        summary['summary_path'] = summary_path.relative_to(out_dir).as_posix()
    return summary


def write_case_files(
    mphtxt_path: Path,
    out_dir: Path,
    field_bundle_path: Optional[Path] = None,
    geometry_only: bool = False,
    diagnostic_grid_spacing_m: float = 5e-4,
    field_ghost_cells: int = DEFAULT_FIELD_GHOST_CELLS,
    coordinate_scale_m_per_model_unit: float = 1.0,
    coordinate_system: str = 'cartesian_xy',
) -> None:
    if field_bundle_path is None and not geometry_only:
        raise ValueError('COMSOL case generation requires --field-bundle; use --geometry-only to build geometry only')
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = out_dir / 'generated'
    generated.mkdir(parents=True, exist_ok=True)

    coordinate_scale = float(coordinate_scale_m_per_model_unit)
    coordinate_system = normalize_coordinate_system(coordinate_system, 2)
    mesh = _scale_mesh_coordinates(parse_comsol_mphtxt(mphtxt_path), coordinate_scale)
    base_arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=float(diagnostic_grid_spacing_m))
    ghost_cells = int(max(0, field_ghost_cells if field_bundle_path is not None else 0))
    arrays = (
        build_precomputed_arrays(
            mesh,
            diagnostic_grid_spacing_m=float(diagnostic_grid_spacing_m),
            grid_padding_cells=ghost_cells,
        )
        if ghost_cells > 0
        else base_arrays
    )
    axes_x = arrays['axes_x']
    axes_y = arrays['axes_y']
    xmin, xmax = float(axes_x[0]), float(axes_x[-1])
    ymin, ymax = float(axes_y[0]), float(axes_y[-1])
    geom_metadata = {
        'provider_kind': 'precomputed_npz',
        'source_kind': 'comsol_mphtxt_geometry',
        'requires_field_bundle': True,
        'has_nearest_boundary_part_id_map': True,
        'boundary_region_map_status': 'nearest_boundary_part_id_map',
        'diagnostic_grid_spacing_m': float(diagnostic_grid_spacing_m),
        'field_ghost_cells': int(ghost_cells),
        'coordinate_unit': 'm',
        'coordinate_scale_m_per_model_unit': float(coordinate_scale),
    }

    geom_npz = generated / 'comsol_geometry_2d.npz'
    _write_geometry_npz(geom_npz, axes_x=axes_x, axes_y=axes_y, arrays=arrays, mesh=mesh, metadata=geom_metadata)
    field_npz = generated / 'comsol_field_2d.npz'
    field_mesh_npz = generated / 'comsol_field_mesh_2d.npz'
    field_summary: Dict[str, Any] = {'mode': 'geometry_only'}
    particle_field_valid_mask: Optional[np.ndarray] = None
    if field_bundle_path is not None:
        bundle_payload = _validate_field_bundle_payload(
            _load_npz_payload(field_bundle_path),
            np.asarray(base_arrays['axes_x'], dtype=np.float64),
            np.asarray(base_arrays['axes_y'], dtype=np.float64),
        )
        bundle_payload = _pad_field_bundle_payload(bundle_payload, axes_x, axes_y, ghost_cells)
        bundle_valid_mask = np.asarray(bundle_payload['valid_mask'], dtype=bool)
        finite_field_mask = _finite_field_support_mask(bundle_payload, bundle_valid_mask.shape)
        invalid_claimed_support = bundle_valid_mask & (~finite_field_mask)
        if np.any(invalid_claimed_support):
            raise ValueError(
                'field bundle valid_mask marks non-finite field values as valid; '
                f'invalid_claimed_node_count={int(np.count_nonzero(invalid_claimed_support))}'
            )
        field_valid_mask = bundle_valid_mask & finite_field_mask
        field_valid_mask_source = (
            'finite_field_quantities'
            if bool(np.all(bundle_valid_mask))
            else 'bundle_valid_mask_and_finite_field_quantities'
        )
        particle_field_valid_mask = field_valid_mask & arrays['inside'].astype(bool)
        has_bundle_support_phi = 'support_phi' in bundle_payload
        field_metadata = {
            'provider_kind': 'precomputed_npz',
            'source_kind': 'comsol_export_bundle_field',
            'geometry_mask_applied': False,
            'field_ghost_cells': int(ghost_cells),
            'field_valid_mask_source': field_valid_mask_source,
            'bundle_valid_node_count': int(np.count_nonzero(bundle_valid_mask)),
            'finite_field_node_count': int(np.count_nonzero(finite_field_mask)),
            'provider_support_expanded_node_count': int(np.count_nonzero(finite_field_mask & (~bundle_valid_mask))),
            'provider_support_removed_nonfinite_node_count': int(np.count_nonzero(invalid_claimed_support)),
            'provider_support_outside_geometry_node_count': int(
                np.count_nonzero(field_valid_mask & (~arrays['inside'].astype(bool)))
            ),
            'field_support_phi_kind': 'provider_support_phi' if has_bundle_support_phi else '',
            'bundle_path': str(field_bundle_path.resolve()),
            'has_domain_region_map': False,
        }
        geometry_support_phi = -np.asarray(arrays['sdf'], dtype=np.float64)
        if has_bundle_support_phi:
            field_support_phi = np.asarray(bundle_payload['support_phi'], dtype=np.float64)
        else:
            field_support_phi = None
        save_payload: Dict[str, np.ndarray] = {
            'axis_0': axes_x.astype(np.float64),
            'axis_1': axes_y.astype(np.float64),
            'times': np.asarray(bundle_payload['times'], dtype=np.float64),
            'valid_mask': field_valid_mask.astype(bool),
            'metadata_json': np.asarray(json.dumps(field_metadata)),
        }
        if field_support_phi is not None:
            save_payload['support_phi'] = np.asarray(field_support_phi, dtype=np.float64)
        for key, value in bundle_payload.items():
            if key in {'axis_0', 'axis_1', 'times', 'valid_mask', 'support_phi'}:
                continue
            save_payload[key] = _apply_field_valid_mask(value, field_valid_mask)
        np.savez_compressed(field_npz, **save_payload)
        field_summary = {
            'mode': 'validated_export_bundle',
            'bundle_path': str(field_bundle_path.resolve()),
            'geometry_mask_applied': False,
            'field_ghost_cells': int(ghost_cells),
            'field_valid_mask_source': field_valid_mask_source,
            'bundle_valid_node_count': int(np.count_nonzero(bundle_valid_mask)),
            'finite_field_node_count': int(np.count_nonzero(finite_field_mask)),
            'provider_support_expanded_node_count': int(np.count_nonzero(finite_field_mask & (~bundle_valid_mask))),
            'provider_support_removed_nonfinite_node_count': int(np.count_nonzero(invalid_claimed_support)),
            'provider_support_outside_geometry_node_count': int(
                np.count_nonzero(field_valid_mask & (~arrays['inside'].astype(bool)))
            ),
            'field_valid_node_count': int(np.count_nonzero(field_valid_mask)),
            'geometry_valid_node_count': int(np.count_nonzero(arrays['inside'])),
            'particle_release_valid_node_count': int(np.count_nonzero(particle_field_valid_mask)),
            'quantities': sorted(k for k in save_payload.keys() if k not in {'axis_0', 'axis_1', 'times', 'valid_mask', 'support_phi', 'metadata_json'}),
            'support_phi_quality': (
                _support_phi_quality_summary(field_support_phi, field_valid_mask)
                if field_support_phi is not None
                else None
            ),
            'geometry_sdf_quality_against_field_valid_mask': _support_phi_quality_summary(
                geometry_support_phi,
                field_valid_mask,
            ),
        }
        raw_boundary_edges = np.asarray(arrays['boundary_edges'], dtype=np.float64)
        raw_boundary_part_ids = np.asarray(arrays['boundary_part_ids'], dtype=np.int32)
        reference_edges, reference_part_ids = _field_support_reference_edges(mesh, raw_boundary_edges, raw_boundary_part_ids)
        support_arrays = _geometry_arrays_from_field_support(
            axes_x,
            axes_y,
            field_valid_mask,
            reference_boundary_edges=reference_edges,
            reference_boundary_part_ids=reference_part_ids,
        )
        support_arrays['vertices'] = arrays['vertices']
        support_arrays['quads'] = arrays['quads']
        support_arrays['quad_part_ids'] = arrays['quad_part_ids']
        support_arrays['triangles'] = arrays['triangles']
        support_arrays['triangle_part_ids'] = arrays['triangle_part_ids']
        raw_boundary_edge_count = int(arrays['boundary_edge_count'])
        arrays = {**arrays, **support_arrays}
        field_support_fallback_boundary_edge_count = int(
            np.count_nonzero(np.asarray(arrays['boundary_part_ids'], dtype=np.int32) == FIELD_SUPPORT_BOUNDARY_PART_ID)
        )
        particle_field_valid_mask = field_valid_mask & arrays['inside'].astype(bool)
        geom_metadata = {
            **geom_metadata,
            'source_kind': 'field_support_mask_boundary',
            'geometry_mask_applied': True,
            'raw_comsol_boundary_edge_count': raw_boundary_edge_count,
            'comsol_edge_entity_reference_count': int(reference_edges.shape[0]),
            'field_support_boundary_edge_count': int(arrays['boundary_edge_count']),
            'field_support_fallback_boundary_edge_count': field_support_fallback_boundary_edge_count,
            'field_support_wall_part_ids': sorted(np.unique(arrays['boundary_part_ids']).astype(int).tolist()),
        }
        _write_geometry_npz(geom_npz, axes_x=axes_x, axes_y=axes_y, arrays=arrays, mesh=mesh, metadata=geom_metadata)
        field_summary.update(
            {
                'raw_comsol_boundary_edge_count': raw_boundary_edge_count,
                'comsol_edge_entity_reference_count': int(reference_edges.shape[0]),
                'field_support_boundary_edge_count': int(arrays['boundary_edge_count']),
                'field_support_fallback_boundary_edge_count': field_support_fallback_boundary_edge_count,
                'field_support_wall_part_ids': sorted(np.unique(arrays['boundary_part_ids']).astype(int).tolist()),
                'geometry_valid_node_count': int(np.count_nonzero(arrays['inside'])),
                'particle_release_valid_node_count': int(np.count_nonzero(particle_field_valid_mask)),
            }
        )
        if field_mesh_npz.exists():
            field_mesh_npz.unlink()
    elif field_npz.exists() or field_mesh_npz.exists():
        if field_npz.exists():
            field_npz.unlink()
        if field_mesh_npz.exists():
            field_mesh_npz.unlink()

    boundary_parts = np.unique(arrays['boundary_part_ids']).astype(int).tolist()
    material_rows, wall_rows, fallback_materials = _material_and_wall_rows(boundary_parts)
    pd.DataFrame(material_rows).to_csv(out_dir / 'materials.csv', index=False)
    pd.DataFrame(wall_rows).to_csv(out_dir / 'part_walls.csv', index=False)
    entity_map_files = _write_comsol_entity_maps(generated, mesh, boundary_parts)
    if particle_field_valid_mask is None:
        particle_points = _sample_points_in_triangles(mesh.vertices, arrays['triangles'], count=24, seed=24680)
    else:
        particle_points = _sample_clean_field_points_in_triangles(
            mesh.vertices,
            arrays['triangles'],
            axes_x,
            axes_y,
            particle_field_valid_mask,
            count=24,
            seed=24680,
        )
    source_part_ids = _nearest_boundary_part_ids_for_points(arrays, particle_points)
    material_ids = _material_ids_for_parts(out_dir, source_part_ids, fallback_materials)
    particle_rows = _particle_rows_from_points(
        particle_points,
        source_part_ids=source_part_ids,
        material_ids=material_ids,
        release_span_s=None,
    )
    pd.DataFrame(particle_rows).to_csv(out_dir / 'particles.csv', index=False)
    stale_process_steps = out_dir / 'process_steps.csv'
    if stale_process_steps.exists():
        stale_process_steps.unlink()
    stale_source_events = out_dir / 'source_events.csv'
    if stale_source_events.exists():
        stale_source_events.unlink()

    summary = {
        'source_mphtxt': str(mphtxt_path.resolve()),
        'sdim': int(mesh.sdim),
        'vertex_count': int(mesh.vertices.shape[0]),
        'edge_count': int(mesh.type_blocks['edg'].elements.shape[0]) if 'edg' in mesh.type_blocks else 0,
        'tri_count': int(mesh.type_blocks['tri'].elements.shape[0]) if 'tri' in mesh.type_blocks else 0,
        'quad_count': int(mesh.type_blocks['quad'].elements.shape[0]) if 'quad' in mesh.type_blocks else 0,
        'surface_triangle_count': int(arrays['triangles'].shape[0]),
        'surface_triangle_part_ids': sorted(np.unique(np.asarray(arrays.get('triangle_part_ids', []), dtype=np.int32)).astype(int).tolist()),
        'quad_part_ids': sorted(np.unique(np.asarray(arrays.get('quad_part_ids', []), dtype=np.int32)).astype(int).tolist()),
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
            **{key: str((generated / value).relative_to(out_dir)) for key, value in entity_map_files.items()},
        },
        'grid_axes': {
            'x_count': int(len(arrays['axes_x'])),
            'y_count': int(len(arrays['axes_y'])),
        },
        'geometry_mode': str(geom_metadata.get('source_kind', 'surface_element_boundary_edges_plus_diagnostic_sdf')),
        'field_mode': field_summary['mode'],
        'field_summary': field_summary,
        'diagnostic_grid_spacing_m': float(diagnostic_grid_spacing_m),
        'field_ghost_cells': int(ghost_cells),
        'coordinate_unit': 'm',
        'coordinate_scale_m_per_model_unit': float(coordinate_scale),
        'coordinate_system': coordinate_system,
        'wall_policy': {
            'physical_wall_law': 'specular',
            'physical_wall_stick_probability': PHYSICAL_WALL_STICK_PROBABILITY,
            'axis_symmetry_law': 'specular',
            'field_support_boundary_law': 'specular',
        },
        'note': 'COMSOL exterior geometry is scaled to SI metres, then field-support geometry is used when field data are present.',
    }
    if field_bundle_path is not None:
        config = {
            'run': {
                'spatial_dim': 2,
                'coordinate_system': coordinate_system,
                'time_interpolation': 'linear',
            },
            'paths': {
                'particles_csv': 'particles.csv',
                'materials_csv': 'materials.csv',
                'part_walls_csv': 'part_walls.csv',
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
                    'enabled': False,
                    'seed': 24680,
                },
                'default_law': 'explicit_csv',
                'source_speed_scale': 1.0,
                'source_position_offset_m': 0.0,
            },
            'input_contract': {
                'initial_particle_field_support': 'strict',
            },
            'provider_contract': {
                'boundary_field_support': 'strict',
                'boundary_offset_cells': 1.0,
            },
            'solver': {
                'dt': 2.0e-8,
                't_end': 2.0e-6,
                'save_every': 10,
                'integrator': 'etd2',
                'min_tau_p_s': 1e-5,
                'valid_mask_policy': 'retry_then_stop',
                'plot_particle_limit': 24,
                'seed': 12345,
                'max_hits_retry_splits': 2,
            },
        }
        (out_dir / 'run_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        summary['generated_files']['run_config'] = 'run_config.yaml'
        provider_contract_summary = _write_generated_provider_contract(out_dir / 'run_config.yaml', generated, out_dir)
        summary['provider_contract'] = provider_contract_summary
        summary['generated_files']['provider_contract_report'] = provider_contract_summary['report_path']
        if 'violations_path' in provider_contract_summary:
            summary['generated_files']['provider_boundary_violations'] = provider_contract_summary['violations_path']
        if 'summary_path' in provider_contract_summary:
            summary['generated_files']['provider_boundary_summary'] = provider_contract_summary['summary_path']
        if (out_dir / 'run_config_mesh.yaml').exists():
            (out_dir / 'run_config_mesh.yaml').unlink()
    if field_bundle_path is not None:
        summary['generated_files']['field_npz'] = str(field_npz.relative_to(out_dir))
    (generated / 'comsol_case_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='Build a COMSOL-derived particle-tracer case and write provider contract diagnostics.')
    ap.add_argument('--mphtxt', type=Path, default=Path('data/argon_gec_ccp_base2.mphtxt'))
    ap.add_argument('--out-dir', type=Path, default=Path('examples/comsol_from_data_2d'))
    ap.add_argument('--field-bundle', type=Path, default=None)
    ap.add_argument('--geometry-only', action='store_true')
    ap.add_argument('--particles-only', action='store_true', help='Rewrite only particles.csv from the clean field sample domain.')
    ap.add_argument('--particle-count', type=int, default=24)
    ap.add_argument('--particle-release-span-s', type=float, default=None)
    ap.add_argument('--particle-seed', type=int, default=24680)
    ap.add_argument('--particle-min-release-offset-cells', type=float, default=1.0)
    ap.add_argument('--source-part-ids', type=str, default=None, help='Comma-separated boundary part IDs to use as particle release sources.')
    ap.add_argument('--diagnostic-grid-spacing-m', type=float, default=5e-4)
    ap.add_argument('--field-ghost-cells', type=int, default=DEFAULT_FIELD_GHOST_CELLS)
    ap.add_argument('--coordinate-scale-m-per-model-unit', type=float, default=1.0)
    ap.add_argument(
        '--coordinate-system',
        default='cartesian_xy',
        choices=['cartesian_xy', 'axisymmetric_rz'],
        help='2D coordinate interpretation for the generated run_config.',
    )
    args = ap.parse_args()
    if bool(args.particles_only):
        write_particles_for_case(
            args.mphtxt.resolve(),
            args.out_dir.resolve(),
            particle_count=int(args.particle_count),
            release_span_s=args.particle_release_span_s,
            seed=int(args.particle_seed),
            min_release_offset_cells=float(args.particle_min_release_offset_cells),
            diagnostic_grid_spacing_m=float(args.diagnostic_grid_spacing_m),
            coordinate_scale_m_per_model_unit=float(args.coordinate_scale_m_per_model_unit),
            source_part_ids=_parse_part_id_list(args.source_part_ids),
        )
        return 0
    write_case_files(
        args.mphtxt.resolve(),
        args.out_dir.resolve(),
        field_bundle_path=args.field_bundle.resolve() if args.field_bundle is not None else None,
        geometry_only=bool(args.geometry_only),
        diagnostic_grid_spacing_m=float(args.diagnostic_grid_spacing_m),
        field_ghost_cells=int(args.field_ghost_cells),
        coordinate_scale_m_per_model_unit=float(args.coordinate_scale_m_per_model_unit),
        coordinate_system=str(args.coordinate_system),
    )
    print(f'Wrote COMSOL-derived case to: {args.out_dir.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
