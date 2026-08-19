"""Build and repair wall-contact frames for 2D edges and 3D triangles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.boundary_service import (
    contact_frame_on_boundary_edge_2d,
)

from .diagnostics import increment_count
from .runtime_execution import RunExecutionContext


@dataclass(frozen=True)
class _ContactFrame2D:
    indices: np.ndarray
    edge_index: np.ndarray
    q0: np.ndarray
    edge: np.ndarray
    length: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    alpha: np.ndarray
    x_contact: np.ndarray
    velocity_old: np.ndarray
    tangent_velocity_old: np.ndarray


@dataclass(frozen=True)
class _ContactFrame3D:
    indices: np.ndarray
    triangle_index: np.ndarray
    triangles: np.ndarray
    q0: np.ndarray
    normal: np.ndarray
    x_wall: np.ndarray
    x_contact: np.ndarray
    velocity_old: np.ndarray
    tangent_velocity_old: np.ndarray


def _boundary_edge_arrays_2d(runtime) -> tuple[np.ndarray | None, np.ndarray]:
    geometry = runtime.geometry_provider.geometry
    if geometry.boundary_edges is None:
        return None, np.zeros(0, dtype=np.int32)
    segments = np.asarray(geometry.boundary_edges, dtype=np.float64)
    part_ids = np.asarray(
        geometry.boundary_edge_part_ids
        if geometry.boundary_edge_part_ids is not None
        else np.zeros(segments.shape[0], dtype=np.int32),
        dtype=np.int32,
    )
    if part_ids.size < segments.shape[0]:
        part_ids = np.pad(
            part_ids,
            (0, int(segments.shape[0] - part_ids.size)),
            constant_values=0,
        )
    return segments, part_ids[: segments.shape[0]]


def _repair_contact_edges_2d(
    execution: RunExecutionContext,
    indices: np.ndarray,
    segments: np.ndarray,
    diagnostics: dict[str, object],
) -> np.ndarray:
    state = execution.state
    edge_index = np.asarray(state.contact_edge_index[indices], dtype=np.int64)
    missing_edge = (edge_index < 0) | (edge_index >= int(segments.shape[0]))
    for particle_index in indices[missing_edge]:
        index = int(particle_index)
        frame = contact_frame_on_boundary_edge_2d(
            execution.context,
            state.x[index],
            part_id_hint=int(state.contact_part_id[index]),
            normal_hint=np.asarray(state.contact_normal[index], dtype=np.float64),
        )
        if frame is None:
            increment_count(diagnostics, "contact_frame_fail_count")
            continue
        state.contact_edge_index[index] = int(frame.edge_index)
    return indices[
        (state.contact_edge_index[indices] >= 0)
        & (state.contact_edge_index[indices] < int(segments.shape[0]))
    ]


def _build_contact_frame_2d(
    execution: RunExecutionContext,
    indices: np.ndarray,
    segments: np.ndarray,
    diagnostics: dict[str, object],
) -> _ContactFrame2D | None:
    state = execution.state
    edge_index = np.asarray(state.contact_edge_index[indices], dtype=np.int64)
    q0 = segments[edge_index, 0, :]
    q1 = segments[edge_index, 1, :]
    edge = q1 - q0
    length = np.linalg.norm(edge, axis=1)
    valid_edge = length > 1.0e-30
    if not np.all(valid_edge):
        increment_count(
            diagnostics,
            "contact_frame_fail_count",
            int(np.count_nonzero(~valid_edge)),
        )
        indices = indices[valid_edge]
        q0 = q0[valid_edge]
        edge = edge[valid_edge]
        length = length[valid_edge]
        edge_index = edge_index[valid_edge]
        if indices.size == 0:
            return None
    tangent = edge / length[:, None]
    normal = np.asarray(state.contact_normal[indices], dtype=np.float64)
    normal_magnitude = np.linalg.norm(normal, axis=1)
    bad_normal = normal_magnitude <= 1.0e-30
    if np.any(bad_normal):
        normal[bad_normal] = np.column_stack(
            (-tangent[bad_normal, 1], tangent[bad_normal, 0])
        )
        normal_magnitude[bad_normal] = 1.0
    normal = normal / normal_magnitude[:, None]
    alpha = np.einsum("ij,ij->i", state.x[indices] - q0, edge) / (length * length)
    alpha = np.clip(alpha, 0.0, 1.0)
    epsilon = float(execution.plan.boundary.contact_offset_m)
    x_contact = q0 + alpha[:, None] * edge - epsilon * normal
    velocity_old = np.asarray(state.v[indices], dtype=np.float64)
    velocity_tangent = np.einsum("ij,ij->i", velocity_old[:, :2], tangent)
    return _ContactFrame2D(
        indices=indices,
        edge_index=edge_index,
        q0=q0,
        edge=edge,
        length=length,
        tangent=tangent,
        normal=normal,
        alpha=alpha,
        x_contact=x_contact,
        velocity_old=velocity_old,
        tangent_velocity_old=velocity_tangent,
    )


def _repair_contact_triangles_3d(
    execution: RunExecutionContext,
    indices: np.ndarray,
    triangle_count: int,
    diagnostics: dict[str, object],
) -> np.ndarray:
    state = execution.state
    triangle_index = np.asarray(state.contact_edge_index[indices], dtype=np.int64)
    missing = (triangle_index < 0) | (triangle_index >= int(triangle_count))
    for particle_index in indices[missing]:
        index = int(particle_index)
        hit = execution.boundary_service.nearest_projection(
            state.x[index], state.x[index]
        )
        if hit is None or int(hit.primitive_id) < 0:
            increment_count(diagnostics, "contact_frame_fail_count")
            continue
        state.contact_edge_index[index] = int(hit.primitive_id)
        state.contact_part_id[index] = int(hit.part_id)
        state.contact_normal[index] = np.asarray(hit.normal, dtype=np.float64)
    return indices[
        (state.contact_edge_index[indices] >= 0)
        & (state.contact_edge_index[indices] < int(triangle_count))
    ]


def _build_contact_frame_3d(
    execution: RunExecutionContext,
    indices: np.ndarray,
    surface,
    diagnostics: dict[str, object],
) -> _ContactFrame3D | None:
    state = execution.state
    triangle_index = np.asarray(state.contact_edge_index[indices], dtype=np.int64)
    triangles = np.asarray(surface.triangles[triangle_index], dtype=np.float64)
    q0 = triangles[:, 0, :]
    normal = np.asarray(state.contact_normal[indices], dtype=np.float64)
    magnitude = np.linalg.norm(normal, axis=1)
    bad_normal = magnitude <= 1.0e-30
    if np.any(bad_normal):
        normal[bad_normal] = np.asarray(
            surface.normals[triangle_index[bad_normal]],
            dtype=np.float64,
        )
        magnitude[bad_normal] = np.linalg.norm(normal[bad_normal], axis=1)
    valid = magnitude > 1.0e-30
    if not np.all(valid):
        increment_count(
            diagnostics,
            "contact_frame_fail_count",
            int(np.count_nonzero(~valid)),
        )
        indices = indices[valid]
        triangle_index = triangle_index[valid]
        triangles = triangles[valid]
        q0 = q0[valid]
        normal = normal[valid]
        magnitude = magnitude[valid]
    if indices.size == 0:
        return None
    normal = normal / magnitude[:, None]
    signed_distance = np.einsum("ij,ij->i", state.x[indices] - q0, normal)
    x_wall = state.x[indices] - signed_distance[:, None] * normal
    epsilon = float(execution.plan.boundary.contact_offset_m)
    x_contact = x_wall - epsilon * normal
    velocity_old = np.asarray(state.v[indices], dtype=np.float64)
    normal_velocity = np.einsum("ij,ij->i", velocity_old[:, :3], normal)
    tangent_velocity_old = velocity_old[:, :3] - normal_velocity[:, None] * normal
    return _ContactFrame3D(
        indices=indices,
        triangle_index=triangle_index,
        triangles=triangles,
        q0=q0,
        normal=normal,
        x_wall=x_wall,
        x_contact=x_contact,
        velocity_old=velocity_old,
        tangent_velocity_old=tangent_velocity_old,
    )
