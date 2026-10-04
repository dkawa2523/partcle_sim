"""Compose saved stochastic paths with deterministic segment motion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    valid_mask_status_requires_stop,
)
from particle_tracer_unified.domain import BoundaryHit, BoundaryQuery

from ._stochastic_first_passage import StochasticCompositionResult
from ._stochastic_first_passage import (
    search_piecewise_langevin_wall_crossing as search_piecewise_langevin_wall_crossing,
)
from ._stochastic_path import PiecewiseLangevinPath
from .base_field_sampling import sample_compiled_valid_mask_statuses
from .segment_motion import (
    SegmentMotionBatchTrace,
    SegmentMotionRequest,
    SegmentMotionTrace,
    ValidMaskPrefixResolution,
    trace_motion_segment,
)


def trace_particle_motion(
    motion_batch: SegmentMotionBatchTrace,
    particle_index: int,
    minimum_substeps: np.ndarray,
) -> SegmentMotionTrace:
    request = motion_batch.request.particle_request(
        int(particle_index)
    ).with_minimum_substeps(int(max(1, minimum_substeps[int(particle_index)])))
    return trace_motion_segment(request)


@dataclass(frozen=True, slots=True)
class PiecewiseLangevinSegmentTrace:
    """One deterministic segment composed with a saved Brownian path interval."""

    elapsed_times_s: np.ndarray
    positions_m: np.ndarray
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    midpoint_position_m: np.ndarray
    aggregate_support_status: int


@dataclass(frozen=True, slots=True)
class _StochasticBatchComposition:
    stage_points: dict[int, np.ndarray]
    prefetched_hits: dict[int, BoundaryHit]
    unresolved_indices: tuple[int, ...]


def compose_piecewise_langevin_state(
    *,
    path: PiecewiseLangevinPath,
    deterministic_position_m: np.ndarray,
    deterministic_velocity_mps: np.ndarray,
    stochastic_offset_s: float,
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a deterministic endpoint with one saved-path innovation."""

    noise_position, noise_velocity = path.replay(
        float(stochastic_offset_s),
        float(stochastic_offset_s) + float(elapsed_s),
    )
    dimension = int(np.asarray(deterministic_position_m).size)
    return (
        np.asarray(deterministic_position_m, dtype=np.float64)
        + noise_position[:dimension],
        np.asarray(deterministic_velocity_mps, dtype=np.float64)
        + noise_velocity[:dimension],
    )


def compose_piecewise_langevin_trace(
    *,
    path: PiecewiseLangevinPath,
    deterministic_trace: SegmentMotionTrace,
    stochastic_offset_s: float = 0.0,
    minimum_node_count: int = 8,
) -> PiecewiseLangevinSegmentTrace:
    """Compose one saved OU interval with one canonical deterministic trace.

    Retained nodes are uniformly timed because the boundary query's polyline
    fraction is also its time fraction. Accepted ETD2 nodes use their stored
    deterministic values; added Brownian nodes query the deterministic prefix.
    """

    duration = float(deterministic_trace.request.duration_s)
    if duration <= 0.0:
        raise ValueError(
            "piecewise Langevin composition requires positive segment duration"
        )
    offset = float(stochastic_offset_s)
    if offset < 0.0 or offset + duration > path.duration_s + 64.0 * np.spacing(
        path.duration_s
    ):
        raise ValueError(
            "piecewise Langevin segment lies outside the saved path interval"
        )
    accepted_count = int(np.asarray(deterministic_trace.times_s).size)
    if accepted_count <= 0:
        raise RuntimeError("deterministic motion trace has no accepted stage nodes")
    node_count = max(int(minimum_node_count), accepted_count)
    remainder = node_count % accepted_count
    if remainder:
        node_count += accepted_count - remainder
    elapsed_times = duration * (
        np.arange(1, node_count + 1, dtype=np.float64) / float(node_count)
    )
    accepted_stride = node_count // accepted_count
    accepted_by_node = {
        (row + 1) * accepted_stride - 1: row for row in range(accepted_count)
    }
    dimension = int(deterministic_trace.request.spatial_dim)
    positions = np.empty((node_count, dimension), dtype=np.float64)
    for node_row, elapsed in enumerate(elapsed_times):
        accepted_row = accepted_by_node.get(node_row)
        if accepted_row is None:
            deterministic_position, _deterministic_velocity = (
                deterministic_trace.state_at(float(elapsed))
            )
        else:
            deterministic_position = np.asarray(
                deterministic_trace.positions_m[accepted_row],
                dtype=np.float64,
            )
        noise_position, _noise_velocity = path.replay(
            offset,
            offset + float(elapsed),
        )
        positions[node_row] = (
            np.asarray(deterministic_position, dtype=np.float64)[:dimension]
            + noise_position[:dimension]
        )
    endpoint_position, endpoint_velocity = compose_piecewise_langevin_state(
        path=path,
        deterministic_position_m=np.asarray(
            deterministic_trace.endpoint_position_m,
            dtype=np.float64,
        )[:dimension],
        deterministic_velocity_mps=np.asarray(
            deterministic_trace.endpoint_velocity_mps,
            dtype=np.float64,
        )[:dimension],
        stochastic_offset_s=offset,
        elapsed_s=duration,
    )
    statuses = sample_compiled_valid_mask_statuses(
        deterministic_trace.request.backend,
        positions,
    )
    aggregate_support_status = int(deterministic_trace.aggregate_support_status)
    if statuses.size:
        aggregate_support_status = max(
            aggregate_support_status,
            int(np.max(statuses)),
        )
    return PiecewiseLangevinSegmentTrace(
        elapsed_times_s=elapsed_times,
        positions_m=positions,
        endpoint_position_m=endpoint_position,
        endpoint_velocity_mps=endpoint_velocity,
        midpoint_position_m=positions[node_count // 2 - 1].copy(),
        aggregate_support_status=int(aggregate_support_status),
    )


def resolve_piecewise_valid_mask_prefix(
    request: SegmentMotionRequest,
    path: PiecewiseLangevinPath,
    *,
    stochastic_offset_s: float = 0.0,
    max_halving_count: int,
    require_clean_prefix: bool = False,
) -> ValidMaskPrefixResolution:
    """Find a dyadic valid prefix without replacing the saved Brownian path."""

    x_start = np.asarray(request.position_m, dtype=np.float64).copy()
    v_start = np.asarray(request.velocity_mps, dtype=np.float64).copy()
    duration = max(float(request.duration_s), 0.0)
    halving_limit = int(max(0, max_halving_count))
    if duration <= 0.0 or halving_limit <= 0:
        return ValidMaskPrefixResolution(
            position=x_start,
            velocity=v_start,
            accepted_dt=0.0,
            retry_count=0,
            found_valid_prefix=False,
        )
    retry_count = 0
    for split_index in range(1, halving_limit + 1):
        retry_count += 1
        prefix_dt = duration * (0.5**split_index)
        deterministic_trace = trace_motion_segment(request.prefix(prefix_dt))
        composed = compose_piecewise_langevin_trace(
            path=path,
            deterministic_trace=deterministic_trace,
            stochastic_offset_s=float(stochastic_offset_s),
        )
        status = int(composed.aggregate_support_status)
        status_ok = (
            status == int(VALID_MASK_STATUS_CLEAN)
            if bool(require_clean_prefix)
            else not bool(valid_mask_status_requires_stop(status))
        )
        if status_ok:
            return ValidMaskPrefixResolution(
                position=composed.endpoint_position_m,
                velocity=composed.endpoint_velocity_mps,
                accepted_dt=float(prefix_dt),
                retry_count=int(retry_count),
                found_valid_prefix=True,
            )
    return ValidMaskPrefixResolution(
        position=x_start,
        velocity=v_start,
        accepted_dt=0.0,
        retry_count=int(retry_count),
        found_valid_prefix=False,
    )


def _compose_piecewise_langevin_paths(
    *,
    paths: Mapping[int, PiecewiseLangevinPath],
    motion_batch: SegmentMotionBatchTrace,
    minimum_substeps: np.ndarray,
    endpoint_position_m: np.ndarray,
    endpoint_velocity_mps: np.ndarray,
    midpoint_position_m: np.ndarray,
    aggregate_support_status: np.ndarray,
    boundary_service: BoundaryQuery[object] | None,
    geometry_tolerance_m: float,
) -> _StochasticBatchComposition:
    """Compose paths and optionally resolve adaptive dyadic wall crossings."""

    stage_points: dict[int, np.ndarray] = {}
    prefetched_hits: dict[int, BoundaryHit] = {}
    unresolved_indices: list[int] = []
    dimension = int(motion_batch.request.spatial_dim)
    counts = np.asarray(minimum_substeps, dtype=np.int32)
    for particle_index, path in paths.items():
        index = int(particle_index)
        trace = trace_particle_motion(motion_batch, index, counts)
        composed = compose_piecewise_langevin_trace(
            path=path,
            deterministic_trace=trace,
        )
        endpoint_position_m[index, :dimension] = composed.endpoint_position_m
        endpoint_velocity_mps[index, :dimension] = composed.endpoint_velocity_mps
        midpoint_position_m[index, :dimension] = composed.midpoint_position_m
        aggregate_support_status[index] = np.uint8(composed.aggregate_support_status)
        stage_points[index] = composed.positions_m
        if boundary_service is None:
            continue
        crossing: StochasticCompositionResult = search_piecewise_langevin_wall_crossing(
            path=path,
            deterministic_trace=trace,
            boundary_service=boundary_service,
            geometry_tolerance_m=float(geometry_tolerance_m),
        )
        crossing_statuses = sample_compiled_valid_mask_statuses(
            trace.request.backend,
            crossing.stage_points,
        )
        if crossing_statuses.size:
            aggregate_support_status[index] = np.uint8(
                max(
                    int(aggregate_support_status[index]),
                    int(np.max(crossing_statuses)),
                )
            )
        if crossing.unresolved:
            unresolved_indices.append(index)
        elif crossing.prefetched_hit is not None:
            prefetched_hits[index] = crossing.prefetched_hit
    return _StochasticBatchComposition(
        stage_points=stage_points,
        prefetched_hits=prefetched_hits,
        unresolved_indices=tuple(unresolved_indices),
    )


def compose_piecewise_langevin_paths(
    *,
    paths: Mapping[int, PiecewiseLangevinPath],
    motion_batch: SegmentMotionBatchTrace,
    minimum_substeps: np.ndarray,
    endpoint_position_m: np.ndarray,
    endpoint_velocity_mps: np.ndarray,
    midpoint_position_m: np.ndarray,
    aggregate_support_status: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compose saved noise with deterministic states from the same primitive."""

    return _compose_piecewise_langevin_paths(
        paths=paths,
        motion_batch=motion_batch,
        minimum_substeps=minimum_substeps,
        endpoint_position_m=endpoint_position_m,
        endpoint_velocity_mps=endpoint_velocity_mps,
        midpoint_position_m=midpoint_position_m,
        aggregate_support_status=aggregate_support_status,
        boundary_service=None,
        geometry_tolerance_m=0.0,
    ).stage_points
