"""Advance sliding contacts on three-dimensional triangle surfaces."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from particle_tracer_unified.core.geometry3d import (
    TriangleSurface3D,
    point_triangle_barycentric,
)
from particle_tracer_unified.domain import StageFields

from . import _contact_dynamics, _contact_geometry, _contact_state
from .diagnostics import increment_count
from .runtime_execution import RunExecutionContext

StageSampler = Callable[..., StageFields]


def _points_inside(execution: RunExecutionContext, points: np.ndarray) -> np.ndarray:
    return np.asarray(
        [execution.boundary_service.inside(point) for point in points],
        dtype=bool,
    )


def _release_contacts_3d(
    execution: RunExecutionContext,
    frame: _contact_geometry._ContactFrame3D,
    dynamics: _contact_dynamics.ContactDynamicsBatch,
    *,
    duration_s: float,
    time_s: float,
    sample_stage: StageSampler,
) -> np.ndarray:
    probe_duration = np.minimum(float(duration_s), dynamics.relaxation_time_s)
    probe_displacement, _ = _contact_dynamics.advance_contact_relaxation(
        frame.tangent_velocity_old,
        dynamics.target_velocity,
        dynamics.body_acceleration,
        dynamics.relaxation_time_s[:, None],
        probe_duration[:, None],
    )
    release_candidate = np.einsum("ij,ij->i", probe_displacement, frame.normal) < 0.0
    release_mask = np.zeros(frame.indices.size, dtype=bool)
    if np.any(release_candidate):
        candidate_rows = np.flatnonzero(release_candidate)
        probe_position = (
            frame.x_contact[candidate_rows] + probe_displacement[candidate_rows]
        )
        clean = _contact_state._clean_support(
            execution,
            points=probe_position,
            time_s=float(time_s),
            inside=_points_inside(execution, probe_position),
            sample_stage=sample_stage,
        )
        release_mask[candidate_rows[clean]] = True
        _contact_state._record_release_probe_rejections(
            execution.state.collision_diagnostics, clean
        )
    if np.any(release_mask):
        state = execution.state
        release_indices = frame.indices[release_mask]
        _contact_state._release_contact_rows(
            state,
            release_indices,
            frame.x_contact[release_mask],
            frame.tangent_velocity_old[release_mask],
        )
    return release_mask


def _hold_contact_endpoints_3d(
    execution: RunExecutionContext,
    frame: _contact_geometry._ContactFrame3D,
    keep_mask: np.ndarray,
    surface: TriangleSurface3D,
) -> np.ndarray:
    state = execution.state
    hold_mask = keep_mask & state.contact_endpoint_stopped[frame.indices]
    if np.any(hold_mask):
        hold_indices = frame.indices[hold_mask]
        primitive_ids = frame.triangle_index[hold_mask]
        _contact_state._hold_contact_rows(
            state,
            hold_indices,
            frame.x_contact[hold_mask],
            primitive_ids,
            surface.part_ids[primitive_ids],
            frame.normal[hold_mask],
        )
    return keep_mask & ~state.contact_endpoint_stopped[frame.indices]


def _inside_contact_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    inside = np.zeros(points.shape[0], dtype=bool)
    for row, (point, triangle) in enumerate(zip(points, triangles, strict=True)):
        barycentric = point_triangle_barycentric(point, triangle)
        if barycentric is not None and np.all(
            np.asarray(barycentric, dtype=np.float64) >= -1.0e-10
        ):
            inside[row] = True
    return inside


def _advance_contact_tangent_3d(
    execution: RunExecutionContext,
    frame: _contact_geometry._ContactFrame3D,
    dynamics: _contact_dynamics.ContactDynamicsBatch,
    mobile_mask: np.ndarray,
    surface: TriangleSurface3D,
    *,
    duration_s: float,
    time_s: float,
    sample_stage: StageSampler,
) -> None:
    state = execution.state
    diagnostics = state.collision_diagnostics
    keep_indices = frame.indices[mobile_mask]
    normal = frame.normal[mobile_mask]
    target_tangent = (
        dynamics.target_velocity[mobile_mask]
        - np.einsum(
            "ij,ij->i",
            dynamics.target_velocity[mobile_mask],
            normal,
        )[:, None]
        * normal
    )
    body_tangent = (
        dynamics.body_acceleration[mobile_mask]
        - np.einsum(
            "ij,ij->i",
            dynamics.body_acceleration[mobile_mask],
            normal,
        )[:, None]
        * normal
    )
    displacement, tangent_velocity = _contact_dynamics.advance_contact_relaxation(
        frame.tangent_velocity_old[mobile_mask],
        target_tangent,
        body_tangent,
        dynamics.relaxation_time_s[mobile_mask, None],
        float(duration_s),
    )
    x_wall_next = frame.x_wall[mobile_mask] + displacement
    plane_error = np.einsum(
        "ij,ij->i",
        x_wall_next - frame.q0[mobile_mask],
        normal,
    )
    x_wall_next -= plane_error[:, None] * normal
    inside_triangle = _inside_contact_triangles(
        x_wall_next,
        frame.triangles[mobile_mask],
    )
    epsilon = float(execution.plan.boundary.contact_offset_m)
    x_next = x_wall_next - epsilon * normal
    clean = _contact_state._clean_support(
        execution,
        points=x_next,
        time_s=float(time_s),
        inside=_points_inside(execution, x_next),
        sample_stage=sample_stage,
    )
    reject = ~clean
    if np.any(reject):
        _contact_state._reject_contact_rows(
            state,
            keep_indices[reject],
            frame.x_contact[mobile_mask][reject],
        )
    if not np.any(clean):
        return
    accept_indices = keep_indices[clean]
    state.x[accept_indices] = x_next[clean]
    state.v[accept_indices] = tangent_velocity[clean]
    accept_endpoint = (~inside_triangle)[clean]
    if np.any(accept_endpoint):
        stopped_indices = accept_indices[accept_endpoint]
        state.x[stopped_indices] = frame.x_contact[mobile_mask][clean][accept_endpoint]
        state.v[stopped_indices] = 0.0
        state.contact_endpoint_stopped[stopped_indices] = True
        increment_count(
            diagnostics,
            "contact_endpoint_stop_count",
            int(np.count_nonzero(accept_endpoint)),
        )
    state.x_trial[accept_indices] = state.x[accept_indices]
    state.v_trial[accept_indices] = state.v[accept_indices]
    state.x_mid_trial[accept_indices] = state.x[accept_indices]
    state.contact_edge_index[accept_indices] = frame.triangle_index[mobile_mask][clean]
    state.contact_part_id[accept_indices] = surface.part_ids[
        frame.triangle_index[mobile_mask][clean]
    ]
    state.contact_normal[accept_indices] = normal[clean]
    increment_count(diagnostics, "contact_tangent_step_count", int(accept_indices.size))
    tangent_time = np.asarray(
        diagnostics.get("contact_tangent_time_total_s", 0.0),
        dtype=np.float64,
    ).item()
    diagnostics["contact_tangent_time_total_s"] = float(tangent_time) + float(
        duration_s
    ) * float(accept_indices.size)


def advance_contact_sliding_3d(
    execution: RunExecutionContext,
    *,
    body_acceleration: np.ndarray,
    duration_s: float,
    time_s: float,
    electric_q_over_m_particle: np.ndarray | None,
    sample_stage: StageSampler,
) -> None:
    """Advance active contacts on the configured triangle surface."""

    state = execution.state
    surface = execution.boundary_service.triangle_surface_3d
    diagnostics = state.collision_diagnostics
    indices = _contact_state._active_contact_indices(execution)
    if indices.size == 0:
        return
    if surface is None:
        increment_count(diagnostics, "contact_frame_fail_count", int(indices.size))
        return
    indices = _contact_geometry._repair_contact_triangles_3d(
        execution,
        indices,
        int(surface.triangles.shape[0]),
        diagnostics,
    )
    if indices.size == 0:
        return
    frame = _contact_geometry._build_contact_frame_3d(
        execution, indices, surface, diagnostics
    )
    if frame is None:
        return
    dynamics = _contact_dynamics._evaluate_contact_dynamics(
        execution,
        indices=frame.indices,
        contact_position=frame.x_contact,
        velocity=frame.velocity_old,
        body_acceleration=body_acceleration,
        time_s=float(time_s),
        electric_q_over_m_particle=electric_q_over_m_particle,
        sample_stage=sample_stage,
    )
    release_mask = _release_contacts_3d(
        execution,
        frame,
        dynamics,
        duration_s=float(duration_s),
        time_s=float(time_s),
        sample_stage=sample_stage,
    )
    keep_mask = ~release_mask
    if not np.any(keep_mask):
        return
    mobile_mask = _hold_contact_endpoints_3d(
        execution,
        frame,
        keep_mask,
        surface,
    )
    if not np.any(mobile_mask):
        return
    _advance_contact_tangent_3d(
        execution,
        frame,
        dynamics,
        mobile_mask,
        surface,
        duration_s=float(duration_s),
        time_s=float(time_s),
        sample_stage=sample_stage,
    )
