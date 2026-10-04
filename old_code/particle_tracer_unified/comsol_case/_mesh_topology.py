"""Derive physical 2D topology and diagnostic arrays from mesh node IDs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from particle_tracer_unified.core.boundary_numerics import (
    scaled_classification_tolerance,
)
from particle_tracer_unified.core.geometry2d import (
    build_boundary_loops_2d,
    encode_boundary_loops_2d,
    points_inside_boundary_loops_2d,
    validate_boundary_edges_2d,
)

from ._mesh_parsing import MeshTypeBlock, ParsedMesh

_MAX_DIAGNOSTIC_AXIS_POINTS = 1_000_001
_MAX_DIAGNOSTIC_GRID_POINTS = 10_000_000


def _order_quad_vertices(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    elements = np.asarray(quads, dtype=np.int64)
    ordered = np.empty_like(elements)
    for index, element in enumerate(elements):
        points = vertices[element]
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        ordered[index] = element[np.argsort(angles)]
    return ordered


def surface_triangles_from_mesh(
    vertices: np.ndarray,
    blocks: Mapping[str, MeshTypeBlock],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if "tri" in blocks:
        chunks.append(np.asarray(blocks["tri"].elements, dtype=np.int64))
    if "quad" in blocks:
        quads = _order_quad_vertices(vertices, blocks["quad"].elements)
        chunks.extend(
            (
                quads[:, [0, 1, 2]].astype(np.int64),
                quads[:, [0, 2, 3]].astype(np.int64),
            )
        )
    return np.vstack(chunks).astype(np.int64)


def _surface_triangle_part_ids_from_mesh(
    blocks: Mapping[str, MeshTypeBlock],
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if "tri" in blocks:
        chunks.append(
            (
                np.asarray(blocks["tri"].geometric_entity_indices, dtype=np.int32) + 1
            ).astype(np.int32)
        )
    if "quad" in blocks:
        quad_ids = (
            np.asarray(blocks["quad"].geometric_entity_indices, dtype=np.int32) + 1
        ).astype(np.int32)
        chunks.extend((quad_ids, quad_ids))
    return np.concatenate(chunks).astype(np.int32) if chunks else np.zeros(0, np.int32)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _surface_element_edges(
    vertices: np.ndarray,
    blocks: Mapping[str, MeshTypeBlock],
) -> Iterator[tuple[int, int]]:
    triangle_block = blocks.get("tri")
    if triangle_block is not None:
        for element in np.asarray(triangle_block.elements, dtype=np.int64):
            yield int(element[0]), int(element[1])
            yield int(element[1]), int(element[2])
            yield int(element[2]), int(element[0])

    quad_block = blocks.get("quad")
    if quad_block is not None:
        for element in _order_quad_vertices(vertices, quad_block.elements):
            yield int(element[0]), int(element[1])
            yield int(element[1]), int(element[2])
            yield int(element[2]), int(element[3])
            yield int(element[3]), int(element[0])


def _record_mesh_edge(
    edge_counts: dict[tuple[int, int], int],
    edge_vertices: dict[tuple[int, int], tuple[int, int]],
    vertex_count: int,
    a: int,
    b: int,
) -> None:
    if not (0 <= a < vertex_count and 0 <= b < vertex_count):
        raise ValueError(
            "COMSOL surface element references a mesh vertex outside the "
            f"vertex table: edge=({a}, {b}), vertex_count={vertex_count}"
        )
    if a == b:
        return
    key = _edge_key(a, b)
    edge_counts[key] = edge_counts.get(key, 0) + 1
    edge_vertices.setdefault(key, (a, b))


def _domain_edge_inventory(
    vertices: np.ndarray,
    blocks: Mapping[str, MeshTypeBlock],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or np.any(~np.isfinite(vertices)):
        raise ValueError("COMSOL 2D mesh vertices must be a finite (n, 2) array")
    vertex_count = int(vertices.shape[0])
    edge_counts: dict[tuple[int, int], int] = {}
    edge_vertices: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b in _surface_element_edges(vertices, blocks):
        _record_mesh_edge(edge_counts, edge_vertices, vertex_count, a, b)
    return edge_counts, edge_vertices


def domain_boundary_edge_vertex_ids(
    vertices: np.ndarray,
    blocks: Mapping[str, MeshTypeBlock],
) -> np.ndarray:
    """Return the selected-domain containment shell using global vertex IDs."""

    edge_counts, edge_vertices = _domain_edge_inventory(vertices, blocks)

    boundary = [edge_vertices[key] for key, count in edge_counts.items() if count == 1]
    if not boundary:
        raise ValueError(
            "Could not derive exterior boundary edges from COMSOL 2D elements."
        )
    return np.asarray(boundary, dtype=np.int64)


def _domain_collision_edge_vertex_ids(
    vertices: np.ndarray,
    blocks: Mapping[str, MeshTypeBlock],
) -> np.ndarray:
    """Return shell edges plus explicit COMSOL interfaces touching the domain."""

    edge_counts, edge_vertices = _domain_edge_inventory(vertices, blocks)
    edge_block = blocks.get("edg")
    entity_edges = set() if edge_block is None else set(_edge_part_ids(edge_block))
    collision_edges = [
        edge_vertices[key]
        for key, count in edge_counts.items()
        if count == 1 or key in entity_edges
    ]
    return np.asarray(collision_edges, dtype=np.int64)


def _edge_part_ids(edge_block: MeshTypeBlock) -> dict[tuple[int, int], int]:
    entity_edges = np.asarray(edge_block.elements, dtype=np.int64)
    if entity_edges.ndim != 2 or entity_edges.shape[1] != 2:
        raise ValueError(
            "COMSOL mphtxt edge entities must contain exactly two mesh vertex IDs"
        )
    part_ids = (
        np.asarray(edge_block.geometric_entity_indices, dtype=np.int64) + 1
    ).astype(np.int32)
    by_edge: dict[tuple[int, int], int] = {}
    for edge, part_id in zip(entity_edges, part_ids, strict=True):
        key = _edge_key(int(edge[0]), int(edge[1]))
        part_id = int(part_id)
        previous = by_edge.get(key)
        if previous is not None and previous != part_id:
            raise ValueError(
                "COMSOL mphtxt assigns conflicting entity IDs to one physical "
                f"edge: edge={key}, part_ids={[previous, part_id]}"
            )
        by_edge[key] = part_id
    return by_edge


def _part_id_for_edge(
    part_ids_by_edge: Mapping[tuple[int, int], int],
    edge: np.ndarray,
) -> int:
    key = _edge_key(int(edge[0]), int(edge[1]))
    if key not in part_ids_by_edge:
        raise ValueError(
            "selected vacuum-domain boundary is missing an explicit COMSOL "
            f"edge entity: edge={key}"
        )
    return part_ids_by_edge[key]


def assign_part_ids_from_edge_entities(
    blocks: Mapping[str, MeshTypeBlock],
    boundary_edge_vertex_ids: np.ndarray,
) -> np.ndarray:
    edge_block = blocks.get("edg")
    if edge_block is None or edge_block.elements.size == 0:
        raise ValueError(
            "COMSOL mphtxt must include edge entities so every physical "
            "boundary has an explicit part ID"
        )
    selected_edges = np.asarray(boundary_edge_vertex_ids, dtype=np.int64)
    if selected_edges.ndim != 2 or selected_edges.shape[1] != 2:
        raise ValueError(
            "selected boundary topology must be an (n, 2) mesh vertex ID array"
        )
    part_ids_by_edge = _edge_part_ids(edge_block)
    return np.fromiter(
        (_part_id_for_edge(part_ids_by_edge, edge) for edge in selected_edges),
        dtype=np.int32,
        count=selected_edges.shape[0],
    )


def _make_uniform_axis(vmin: float, vmax: float, spacing: float) -> np.ndarray:
    lo, hi, spacing = float(vmin), float(vmax), float(spacing)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("diagnostic grid bounds must be finite and strictly ordered")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("diagnostic_grid_spacing_m must be positive and finite")
    intervals = (hi - lo) / spacing
    if not np.isfinite(intervals) or intervals > _MAX_DIAGNOSTIC_AXIS_POINTS - 1:
        raise ValueError(
            "diagnostic_grid_spacing_m would create too many points on one axis; "
            "choose a coarser explicit spacing"
        )
    rounded = round(intervals)
    count = (
        rounded + 1
        if abs(intervals - rounded) <= 1.0e-9 * max(1.0, abs(intervals))
        else int(np.ceil(intervals)) + 1
    )
    return np.linspace(lo, hi, max(2, count), dtype=np.float64)


def _distance_and_nearest_edge(
    points: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    minimum_distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    nearest = np.zeros(points.shape[0], dtype=np.int64)
    for index, (start, end) in enumerate(zip(edge_start, edge_end, strict=True)):
        edge = end - start
        denominator = float(np.dot(edge, edge))
        fraction = np.clip(((points - start) @ edge) / denominator, 0.0, 1.0)
        projection = start + fraction[:, None] * edge
        distance = np.linalg.norm(points - projection, axis=1)
        better = distance < minimum_distance
        minimum_distance[better] = distance[better]
        nearest[better] = index
    return minimum_distance, nearest


def build_precomputed_arrays(
    mesh: ParsedMesh,
    *,
    diagnostic_grid_spacing_m: float,
) -> dict[str, Any]:
    """Build authoritative boundary topology plus diagnostic regular arrays."""

    if mesh.sdim != 2:
        raise ValueError("Current exporter supports only 2D mesh (sdim=2).")
    if "tri" not in mesh.type_blocks and "quad" not in mesh.type_blocks:
        raise ValueError("mphtxt must include tri or quad elements.")

    vertices = mesh.vertices
    triangles = surface_triangles_from_mesh(vertices, mesh.type_blocks)
    triangle_part_ids = _surface_triangle_part_ids_from_mesh(mesh.type_blocks)
    quads = (
        _order_quad_vertices(vertices, mesh.type_blocks["quad"].elements)
        if "quad" in mesh.type_blocks
        else np.zeros((0, 4), dtype=np.int64)
    )
    quad_part_ids = (
        (
            np.asarray(mesh.type_blocks["quad"].geometric_entity_indices, np.int32) + 1
        ).astype(np.int32)
        if "quad" in mesh.type_blocks
        else np.zeros(0, dtype=np.int32)
    )
    containment_vertex_ids = domain_boundary_edge_vertex_ids(vertices, mesh.type_blocks)
    collision_vertex_ids = _domain_collision_edge_vertex_ids(vertices, mesh.type_blocks)
    containment_edges = vertices[containment_vertex_ids].astype(np.float64)
    containment_part_ids = assign_part_ids_from_edge_entities(
        mesh.type_blocks,
        containment_vertex_ids,
    )
    boundary_edges = vertices[collision_vertex_ids].astype(np.float64)
    boundary_part_ids = assign_part_ids_from_edge_entities(
        mesh.type_blocks,
        collision_vertex_ids,
    )
    boundary_loops = build_boundary_loops_2d(containment_edges)

    axes_x = _make_uniform_axis(
        np.min(boundary_edges[:, :, 0]),
        np.max(boundary_edges[:, :, 0]),
        diagnostic_grid_spacing_m,
    )
    axes_y = _make_uniform_axis(
        np.min(boundary_edges[:, :, 1]),
        np.max(boundary_edges[:, :, 1]),
        diagnostic_grid_spacing_m,
    )
    if int(axes_x.size) * int(axes_y.size) > _MAX_DIAGNOSTIC_GRID_POINTS:
        raise ValueError(
            "diagnostic_grid_spacing_m would create too many 2D grid points; "
            "choose a coarser explicit spacing"
        )

    grid_x, grid_y = np.meshgrid(axes_x, axes_y, indexing="ij")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    edge_lengths = np.linalg.norm(
        boundary_edges[:, 1] - boundary_edges[:, 0],
        axis=1,
    )
    positive_lengths = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0.0)]
    if positive_lengths.size != edge_lengths.size:
        raise ValueError(
            "COMSOL geometry contains a non-finite or degenerate boundary edge"
        )
    _, classification_tolerance = scaled_classification_tolerance(
        boundary_edges,
        float(np.min(positive_lengths)),
    )
    inside = points_inside_boundary_loops_2d(
        points,
        boundary_loops,
        on_edge_tol=classification_tolerance,
    ).reshape(grid_x.shape)
    distance, nearest = _distance_and_nearest_edge(
        points,
        containment_edges[:, 0],
        containment_edges[:, 1],
    )
    distance = distance.reshape(grid_x.shape)
    nearest_part_ids = (
        containment_part_ids[nearest].reshape(grid_x.shape).astype(np.int32)
    )
    signed_distance = np.where(inside, -distance, distance)
    normal_x, normal_y = np.gradient(
        signed_distance,
        axes_x,
        axes_y,
        edge_order=1,
    )
    loops_flat, loop_offsets = encode_boundary_loops_2d(boundary_loops)
    return {
        "axes_x": axes_x,
        "axes_y": axes_y,
        "sdf": signed_distance.astype(np.float64),
        "normal_x": normal_x.astype(np.float64),
        "normal_y": normal_y.astype(np.float64),
        "inside": inside,
        "nearest_boundary_part_id_map": nearest_part_ids,
        "boundary_edges": boundary_edges,
        "boundary_part_ids": boundary_part_ids,
        "boundary_loops_2d": boundary_loops,
        "boundary_loops_2d_flat": loops_flat,
        "boundary_loops_2d_offsets": loop_offsets,
        "boundary_edge_count": int(boundary_edges.shape[0]),
        "containment_boundary_edge_count": int(containment_edges.shape[0]),
        "internal_interface_edge_count": int(
            boundary_edges.shape[0] - containment_edges.shape[0]
        ),
        "boundary_edge_topology": validate_boundary_edges_2d(containment_edges),
        "vertices": vertices,
        "quads": quads,
        "quad_part_ids": quad_part_ids,
        "triangles": triangles,
        "triangle_part_ids": triangle_part_ids,
    }
