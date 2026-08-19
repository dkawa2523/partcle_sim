"""Validation and destination-buffer ownership for motion batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN

from ._segment_motion_contracts import (
    SegmentMotionBatchDestination,
    SegmentMotionBatchRequest,
)


@dataclass(frozen=True, slots=True)
class _NormalizedMotionBatchState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    active: np.ndarray
    spatial_dim: int
    particle_count: int


@dataclass(frozen=True, slots=True)
class _MotionBatchBuffers:
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    midpoint_position_m: np.ndarray
    substep_count: np.ndarray
    aggregate_support_status: np.ndarray
    local_error_resolved: np.ndarray


def _normalize_motion_batch_state(
    request: SegmentMotionBatchRequest,
) -> _NormalizedMotionBatchState:
    position = np.asarray(request.position_m, dtype=np.float64)
    velocity = np.asarray(request.velocity_mps, dtype=np.float64)
    if position.ndim != 2 or velocity.shape != position.shape:
        raise ValueError(
            "batch position_m and velocity_mps must have the same 2D shape"
        )
    spatial_dim = int(request.spatial_dim)
    if spatial_dim not in (2, 3) or position.shape[1] != spatial_dim:
        raise ValueError("batch state dimension must match spatial_dim=2 or 3")
    particle_count = int(position.shape[0])
    active = np.asarray(request.active, dtype=bool)
    if active.shape != (particle_count,):
        raise ValueError("batch active mask must have shape (particle_count,)")
    return _NormalizedMotionBatchState(
        position_m=position,
        velocity_mps=velocity,
        active=active,
        spatial_dim=spatial_dim,
        particle_count=particle_count,
    )


def _motion_batch_buffers(
    state: _NormalizedMotionBatchState,
    destination: SegmentMotionBatchDestination | None,
) -> _MotionBatchBuffers:
    if destination is None:
        return _MotionBatchBuffers(
            endpoint_position_m=state.position_m.copy(),
            endpoint_velocity_mps=state.velocity_mps.copy(),
            midpoint_position_m=state.position_m.copy(),
            substep_count=np.ones(state.particle_count, dtype=np.int32),
            aggregate_support_status=np.full(
                state.particle_count,
                VALID_MASK_STATUS_CLEAN,
                dtype=np.uint8,
            ),
            local_error_resolved=np.ones(state.particle_count, dtype=bool),
        )
    buffers = _MotionBatchBuffers(
        endpoint_position_m=np.asarray(
            destination.endpoint_position_m,
            dtype=np.float64,
        ),
        endpoint_velocity_mps=np.asarray(
            destination.endpoint_velocity_mps,
            dtype=np.float64,
        ),
        midpoint_position_m=np.asarray(
            destination.midpoint_position_m,
            dtype=np.float64,
        ),
        substep_count=np.asarray(destination.substep_count, dtype=np.int32),
        aggregate_support_status=np.asarray(
            destination.aggregate_support_status,
            dtype=np.uint8,
        ),
        local_error_resolved=np.asarray(
            destination.local_error_resolved,
            dtype=bool,
        ),
    )
    vector_shape = state.position_m.shape
    particle_shape = (state.particle_count,)
    if (
        buffers.endpoint_position_m.shape != vector_shape
        or buffers.endpoint_velocity_mps.shape != vector_shape
        or buffers.midpoint_position_m.shape != vector_shape
        or buffers.substep_count.shape != particle_shape
        or buffers.aggregate_support_status.shape != particle_shape
        or buffers.local_error_resolved.shape != particle_shape
    ):
        raise ValueError("batch destination buffers do not match request state")
    return buffers
