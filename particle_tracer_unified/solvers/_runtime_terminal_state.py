"""Terminal particle-state transitions for the high-fidelity runtime."""

from __future__ import annotations

import numpy as np

from .diagnostics import (
    increment_count,
    increment_named_count,
    invalid_stop_reason_code,
    invalid_stop_reason_name,
)
from .runtime_state import SolverState


def commit_particle_state(
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    escaped: np.ndarray,
    *,
    particle_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    boundary_tolerance_m: float,
) -> None:
    index = int(particle_index)
    x[index] = np.asarray(position, dtype=np.float64)
    v[index] = np.asarray(velocity, dtype=np.float64)
    padding = max(float(boundary_tolerance_m), 0.0)
    if active[index] and (
        np.any(x[index] < mins - padding) or np.any(x[index] > maxs + padding)
    ):
        escaped[index] = True
        active[index] = False


def stop_particle_motion(
    *,
    state: SolverState,
    particle_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    update_trial_buffers: bool,
) -> None:
    index = int(particle_index)
    pos = np.asarray(position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    state.x[index] = pos
    state.v[index] = vel
    if bool(update_trial_buffers):
        state.x_trial[index] = pos
        state.v_trial[index] = vel
        state.x_mid_trial[index] = pos
    state.active[index] = False
    state.stuck[index] = False
    state.frozen[index] = False
    state.absorbed[index] = False
    state.contact_sliding[index] = False
    state.contact_endpoint_stopped[index] = False
    state.contact_edge_index[index] = -1
    state.contact_part_id[index] = 0
    state.contact_normal[index] = 0.0
    state.escaped[index] = False


def mark_invalid_mask_stopped(
    *,
    state: SolverState,
    particle_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    update_trial_buffers: bool,
    reason: str,
) -> None:
    index = int(particle_index)
    stop_particle_motion(
        state=state,
        particle_index=index,
        position=position,
        velocity=velocity,
        update_trial_buffers=update_trial_buffers,
    )
    state.numerical_boundary_stopped[index] = False
    if not bool(state.invalid_mask_stopped[index]):
        state.invalid_mask_stopped[index] = True
        reason_code = invalid_stop_reason_code(str(reason))
        state.invalid_stop_reason_code[index] = np.uint8(reason_code)
        reason_name = invalid_stop_reason_name(int(reason_code))
        increment_count(state.collision_diagnostics, "invalid_mask_stopped_count")
        increment_named_count(
            state.collision_diagnostics, "invalid_mask_stop_reason_counts", reason_name
        )


def mark_numerical_boundary_stopped(
    *,
    state: SolverState,
    particle_index: int,
    position: np.ndarray,
    velocity: np.ndarray,
    update_trial_buffers: bool,
    reason: str,
) -> None:
    index = int(particle_index)
    stop_particle_motion(
        state=state,
        particle_index=index,
        position=position,
        velocity=velocity,
        update_trial_buffers=update_trial_buffers,
    )
    state.invalid_mask_stopped[index] = False
    if not bool(state.numerical_boundary_stopped[index]):
        state.numerical_boundary_stopped[index] = True
        increment_count(state.collision_diagnostics, "numerical_boundary_stop_count")
        increment_named_count(
            state.collision_diagnostics, "numerical_boundary_stop_reason_counts", reason
        )
