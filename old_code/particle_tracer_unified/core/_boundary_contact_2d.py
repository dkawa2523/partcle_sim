"""Two-dimensional boundary contact frames and nearest projections."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from particle_tracer_unified.domain import BoundaryHit

from ._boundary_hits_2d import (
    _active_collision_edge_mask,
    _boundary_edges_2d,
    _edge_endpoint_is_ambiguous,
    _edge_part_id,
)


class BoundaryEdgeFrame2D(NamedTuple):
    edge_index: int
    start: np.ndarray
    end: np.ndarray
    projection: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    part_id: int
    alpha: float
    length: float
    distance: float


class _EdgeProjection2D(NamedTuple):
    position: np.ndarray
    normal: np.ndarray
    part_id: int
    edge_index: int
    edge_alpha: float
    distance: float


def _finite_vector_2d(value: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape[0] != 2 or not np.all(np.isfinite(vector)):
        return None
    return vector


def _oriented_edge_normal_2d(
    normal: np.ndarray,
    point_offset: np.ndarray,
    normal_reference: np.ndarray | None,
) -> np.ndarray:
    if normal_reference is not None:
        if float(np.dot(normal, normal_reference)) < 0.0:
            return -normal
        return normal
    if float(np.dot(point_offset, normal)) > 0.0:
        return -normal
    return normal


def _contact_edge_frame_2d(
    segment: np.ndarray,
    *,
    edge_index: int,
    part_id: int,
    point: np.ndarray,
    normal_reference: np.ndarray | None,
) -> BoundaryEdgeFrame2D | None:
    edge_start = np.asarray(segment[0], dtype=np.float64)
    edge_end = np.asarray(segment[1], dtype=np.float64)
    edge = edge_end - edge_start
    length = float(np.linalg.norm(edge))
    if length <= 1.0e-30:
        return None
    tangent = edge / length
    alpha = float(
        np.clip(np.dot(point - edge_start, edge) / (length * length), 0.0, 1.0)
    )
    projection = edge_start + alpha * edge
    distance = float(np.linalg.norm(point - projection))
    normal = _oriented_edge_normal_2d(
        np.asarray([-tangent[1], tangent[0]], dtype=np.float64),
        point - projection,
        normal_reference,
    )
    return BoundaryEdgeFrame2D(
        edge_index=edge_index,
        start=edge_start,
        end=edge_end,
        projection=np.asarray(projection, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        tangent=np.asarray(tangent, dtype=np.float64),
        part_id=max(0, part_id),
        alpha=alpha,
        length=length,
        distance=distance,
    )


def _nearest_contact_edge_frame_2d(
    segments: np.ndarray,
    part_ids: np.ndarray,
    point: np.ndarray,
    *,
    part_id_hint: int,
    normal_reference: np.ndarray | None,
) -> BoundaryEdgeFrame2D | None:
    best: BoundaryEdgeFrame2D | None = None
    best_distance = np.inf
    for edge_index, segment in enumerate(np.asarray(segments, dtype=np.float64)):
        part_id = _edge_part_id(part_ids, edge_index)
        if part_id_hint > 0 and part_id != part_id_hint:
            continue
        candidate = _contact_edge_frame_2d(
            segment,
            edge_index=int(edge_index),
            part_id=part_id,
            point=point,
            normal_reference=normal_reference,
        )
        if candidate is None or candidate.distance >= best_distance:
            continue
        best_distance = candidate.distance
        best = candidate
    return best


def contact_frame_on_boundary_edge_2d(
    runtime,
    point: np.ndarray,
    *,
    part_id_hint: int = 0,
    normal_hint: np.ndarray | None = None,
) -> BoundaryEdgeFrame2D | None:
    segments, part_ids = _boundary_edges_2d(runtime)
    if segments is None:
        return None
    point_arr = _finite_vector_2d(point)
    if point_arr is None:
        return None
    hint = int(part_id_hint)
    normal_ref = None if normal_hint is None else _finite_vector_2d(normal_hint)
    best = _nearest_contact_edge_frame_2d(
        segments,
        part_ids,
        point_arr,
        part_id_hint=hint,
        normal_reference=normal_ref,
    )
    if best is not None or hint <= 0:
        return best
    return _nearest_contact_edge_frame_2d(
        segments,
        part_ids,
        point_arr,
        part_id_hint=0,
        normal_reference=normal_ref,
    )


def _edge_projection_2d(
    segment: np.ndarray,
    point: np.ndarray,
    inside_reference: np.ndarray,
    *,
    edge_index: int,
    part_id: int,
) -> _EdgeProjection2D:
    edge_start = segment[0]
    edge = segment[1] - edge_start
    denominator = float(np.dot(edge, edge))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"Boundary edge {edge_index} must have finite positive length")
    edge_alpha = float(
        np.clip(np.dot(point - edge_start, edge) / denominator, 0.0, 1.0)
    )
    position = edge_start + edge_alpha * edge
    distance = float(np.linalg.norm(point - position))
    normal = np.array([-edge[1], edge[0]], dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    normal /= magnitude
    if float(np.dot(inside_reference - position, normal)) > 0.0:
        normal = -normal
    return _EdgeProjection2D(
        position=position,
        normal=normal,
        part_id=part_id,
        edge_index=edge_index,
        edge_alpha=edge_alpha,
        distance=distance,
    )


def nearest_hit_on_boundary_edges(
    runtime,
    point: np.ndarray,
    inside_reference: np.ndarray,
) -> BoundaryHit | None:
    geometry_provider = runtime.geometry_provider
    segments, part_ids = _boundary_edges_2d(runtime)
    if geometry_provider is None or segments is None:
        return None
    active = _active_collision_edge_mask(runtime, segments, part_ids)
    point_arr = np.asarray(point, dtype=np.float64)
    ref = np.asarray(inside_reference, dtype=np.float64)
    best: _EdgeProjection2D | None = None
    for idx, segment in enumerate(segments):
        if not active[idx]:
            continue
        part_id = _edge_part_id(part_ids, idx)
        candidate = _edge_projection_2d(
            segment,
            point_arr,
            ref,
            edge_index=int(idx),
            part_id=part_id,
        )
        if best is not None and candidate.distance >= best.distance:
            continue
        best = candidate
    if best is None:
        return None
    return BoundaryHit(
        position=np.asarray(best.position, dtype=np.float64),
        normal=np.asarray(best.normal, dtype=np.float64),
        part_id=max(0, int(best.part_id)),
        alpha_hint=0.0,
        primitive_id=int(best.edge_index),
        primitive_kind="edge_projection",
        is_ambiguous=_edge_endpoint_is_ambiguous(best.edge_alpha),
    )


__all__ = (
    "BoundaryEdgeFrame2D",
    "contact_frame_on_boundary_edge_2d",
    "nearest_hit_on_boundary_edges",
)
