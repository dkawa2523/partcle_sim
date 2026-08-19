"""Accepted scalar traces for dynamically charged electric-force motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    valid_mask_status_requires_stop,
)

from ._coupled_charge_leaf import (
    CoupledChargeLeafContext,
    advance_coupled_charge_embedded,
    advance_coupled_charge_leaf,
    coupled_charge_leaf_context,
)
from ._segment_motion_contracts import (
    SegmentMotionBatchDestination,
    SegmentMotionBatchRequest,
    SegmentMotionRequest,
    ValidMaskPrefixResolution,
)
from .base_field_sampling import sample_compiled_valid_mask_status
from .charge_model import ChargeModelConfig
from .integrator_common import doubled_substep_count, uniform_substep_schedule
from .plasma_background import PreparedPlasmaBackground


@dataclass(frozen=True, slots=True)
class CoupledChargeMotionTrace:
    tracer: CoupledChargeParticleTracer
    request: SegmentMotionRequest
    start_charge_C: float
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    charges_C: np.ndarray
    substep_count: int
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    endpoint_charge_C: float
    aggregate_support_status: int
    local_error_resolved: bool

    def state_at(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        elapsed, node_index = self._elapsed_and_node(float(elapsed_s))
        if elapsed <= 0.0:
            return self.request.position_m.copy(), self.request.velocity_mps.copy()
        if node_index is not None:
            return (
                self.positions_m[node_index].copy(),
                self.velocities_mps[node_index].copy(),
            )
        position, velocity, _charge = self._partial_state(elapsed)
        return position, velocity

    def charge_at(self, elapsed_s: float) -> float:
        elapsed, node_index = self._elapsed_and_node(float(elapsed_s))
        if elapsed <= 0.0:
            return float(self.start_charge_C)
        if node_index is not None:
            return float(self.charges_C[node_index])
        _position, _velocity, charge = self._partial_state(elapsed)
        return float(charge)

    def _elapsed_and_node(self, elapsed_s: float) -> tuple[float, int | None]:
        duration = max(float(self.request.duration_s), 0.0)
        elapsed = float(np.clip(float(elapsed_s), 0.0, duration))
        if elapsed >= duration:
            return elapsed, int(self.positions_m.shape[0] - 1)
        half_leaf = duration / float(2 * max(1, int(self.substep_count)))
        node_index = round(elapsed / half_leaf) - 1
        if (
            0 <= node_index < self.positions_m.shape[0]
            and elapsed == float(node_index + 1) * half_leaf
        ):
            return elapsed, int(node_index)
        return elapsed, None

    def _partial_state(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray, float]:
        duration = max(float(self.request.duration_s), 0.0)
        leaf_count = max(1, int(self.substep_count))
        leaf_duration = duration / float(leaf_count)
        leaf_index = min(int(elapsed_s / leaf_duration), leaf_count - 1)
        leaf_elapsed = float(leaf_index) * leaf_duration
        if leaf_index == 0:
            position = np.asarray(self.request.position_m, dtype=np.float64)
            velocity = np.asarray(self.request.velocity_mps, dtype=np.float64)
            charge = float(self.start_charge_C)
        else:
            endpoint_row = 2 * leaf_index - 1
            position = self.positions_m[endpoint_row]
            velocity = self.velocities_mps[endpoint_row]
            charge = float(self.charges_C[endpoint_row])
        leaf = advance_coupled_charge_leaf(
            self.tracer.leaf_context(self.request),
            position_m=position,
            velocity_mps=velocity,
            charge_C=charge,
            start_time_s=self.request.start_time_s + leaf_elapsed,
            duration_s=float(elapsed_s) - leaf_elapsed,
        )
        return leaf.position_m, leaf.velocity_mps, float(leaf.charge_C)


@dataclass(frozen=True, slots=True)
class CoupledChargeParticleTracer:
    config: ChargeModelConfig
    runtime: object
    plasma_background: PreparedPlasmaBackground | None
    physical_diameter_m: float

    def leaf_context(self, request: SegmentMotionRequest) -> CoupledChargeLeafContext:
        return coupled_charge_leaf_context(
            request,
            config=self.config,
            runtime=self.runtime,
            plasma_background=self.plasma_background,
            physical_diameter_m=float(self.physical_diameter_m),
        )

    def trace(
        self,
        request: SegmentMotionRequest,
        *,
        charge_start_C: float,
    ) -> CoupledChargeMotionTrace:
        return trace_coupled_charge_motion(
            self,
            request,
            charge_start_C=float(charge_start_C),
        )


@dataclass(frozen=True, slots=True)
class CoupledChargeMotionBatch:
    """Batch endpoints backed by the same scalar coupled trace used by collisions."""

    request: SegmentMotionBatchRequest
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    midpoint_position_m: np.ndarray
    substep_count: np.ndarray
    aggregate_support_status: np.ndarray
    local_error_resolved: np.ndarray
    endpoint_charge_C: np.ndarray
    start_charge_C: np.ndarray
    tracers: dict[int, CoupledChargeParticleTracer]
    traces: dict[int, CoupledChargeMotionTrace]

    def particle_trace(
        self, index: int, *, minimum_substeps: int = 1
    ) -> CoupledChargeMotionTrace:
        i = int(index)
        request = self.request.particle_request(i).with_minimum_substeps(
            int(minimum_substeps)
        )
        trace = self.tracers[i].trace(
            request,
            charge_start_C=float(self.start_charge_C[i]),
        )
        self.traces[i] = trace
        return trace


def _stationary_trace(
    tracer: CoupledChargeParticleTracer,
    request: SegmentMotionRequest,
    charge_start_C: float,
) -> CoupledChargeMotionTrace:
    position = np.asarray(request.position_m, dtype=np.float64).copy()
    velocity = np.asarray(request.velocity_mps, dtype=np.float64).copy()
    return CoupledChargeMotionTrace(
        tracer=tracer,
        request=request,
        start_charge_C=float(charge_start_C),
        positions_m=position.reshape(1, -1),
        velocities_mps=velocity.reshape(1, -1),
        charges_C=np.asarray([float(charge_start_C)], dtype=np.float64),
        substep_count=1,
        endpoint_position_m=position,
        endpoint_velocity_mps=velocity,
        endpoint_charge_C=float(charge_start_C),
        aggregate_support_status=int(VALID_MASK_STATUS_CLEAN),
        local_error_resolved=True,
    )


def _support_status(request: SegmentMotionRequest, position_m: np.ndarray) -> int:
    return int(sample_compiled_valid_mask_status(request.backend, position_m))


def _fixed_schedule_trace(
    tracer: CoupledChargeParticleTracer,
    request: SegmentMotionRequest,
    *,
    charge_start_C: float,
    substep_count: int,
    substep_duration_s: float,
    estimate_local_error: bool,
) -> tuple[CoupledChargeMotionTrace, bool]:
    context = tracer.leaf_context(request)
    position = np.asarray(request.position_m, dtype=np.float64).copy()
    velocity = np.asarray(request.velocity_mps, dtype=np.float64).copy()
    charge = float(charge_start_C)
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    charges: list[float] = []
    aggregate_status = int(VALID_MASK_STATUS_CLEAN)
    refinement_required = False
    for index in range(int(substep_count)):
        start_time = request.start_time_s + float(index) * substep_duration_s
        start_position = position.copy()
        start_velocity = velocity.copy()
        start_charge = float(charge)
        if bool(estimate_local_error) and not refinement_required:
            embedded = advance_coupled_charge_embedded(
                context,
                position_m=start_position,
                velocity_mps=start_velocity,
                charge_C=start_charge,
                start_time_s=start_time,
                duration_s=substep_duration_s,
            )
            leaf = embedded.full
            half = embedded.refined_mid
            sampled_leaves = (leaf, half, embedded.refined_end)
            refinement_required = bool(embedded.refinement_required)
        else:
            leaf = advance_coupled_charge_leaf(
                context,
                position_m=start_position,
                velocity_mps=start_velocity,
                charge_C=start_charge,
                start_time_s=start_time,
                duration_s=substep_duration_s,
            )
            half = advance_coupled_charge_leaf(
                context,
                position_m=start_position,
                velocity_mps=start_velocity,
                charge_C=start_charge,
                start_time_s=start_time,
                duration_s=0.5 * substep_duration_s,
            )
            sampled_leaves = (leaf, half)
        position = leaf.position_m
        velocity = leaf.velocity_mps
        charge = float(leaf.charge_C)
        half_position = half.position_m
        half_velocity = half.velocity_mps
        half_charge = float(half.charge_C)
        sample_status = max(
            (
                _support_status(request, sample_position)
                for sampled_leaf in sampled_leaves
                for sample_position in sampled_leaf.sample_positions_m
            ),
            default=int(VALID_MASK_STATUS_CLEAN),
        )
        half_status = max(sample_status, _support_status(request, half_position))
        endpoint_status = _support_status(request, position)
        aggregate_status = max(
            aggregate_status,
            _support_status(request, start_position),
            half_status,
            endpoint_status,
        )
        positions.extend((half_position.copy(), position.copy()))
        velocities.extend((half_velocity.copy(), velocity.copy()))
        charges.extend((half_charge, charge))
    trace = CoupledChargeMotionTrace(
        tracer=tracer,
        request=request,
        start_charge_C=float(charge_start_C),
        positions_m=np.asarray(positions, dtype=np.float64),
        velocities_mps=np.asarray(velocities, dtype=np.float64),
        charges_C=np.asarray(charges, dtype=np.float64),
        substep_count=int(substep_count),
        endpoint_position_m=np.asarray(position, dtype=np.float64),
        endpoint_velocity_mps=np.asarray(velocity, dtype=np.float64),
        endpoint_charge_C=float(charge),
        aggregate_support_status=int(aggregate_status),
        local_error_resolved=not refinement_required,
    )
    return trace, refinement_required


def trace_coupled_charge_motion(
    tracer: CoupledChargeParticleTracer,
    request: SegmentMotionRequest,
    *,
    charge_start_C: float,
) -> CoupledChargeMotionTrace:
    duration = max(float(request.duration_s), 0.0)
    if duration == 0.0:
        return _stationary_trace(tracer, request, float(charge_start_C))
    count, step, _start = uniform_substep_schedule(
        duration,
        request.end_time_s,
        int(request.adaptive_substep_max_splits),
        int(request.minimum_substeps),
    )
    while True:
        trace, refinement_required = _fixed_schedule_trace(
            tracer,
            request,
            charge_start_C=float(charge_start_C),
            substep_count=int(count),
            substep_duration_s=float(step),
            estimate_local_error=bool(int(request.adaptive_substep_enabled) != 0),
        )
        refined_count = int(
            doubled_substep_count(int(count), int(request.adaptive_substep_max_splits))
        )
        if not refinement_required or refined_count == count:
            return trace
        count = refined_count
        step = duration / float(count)


def _commit_batch_trace(
    batch: CoupledChargeMotionBatch,
    index: int,
    trace: CoupledChargeMotionTrace,
) -> None:
    i = int(index)
    dimension = int(batch.request.spatial_dim)
    midpoint_row = max(0, int(trace.substep_count) - 1)
    batch.endpoint_position_m[i, :dimension] = trace.endpoint_position_m[:dimension]
    batch.endpoint_velocity_mps[i, :dimension] = trace.endpoint_velocity_mps[:dimension]
    batch.midpoint_position_m[i, :dimension] = trace.positions_m[
        midpoint_row, :dimension
    ]
    batch.substep_count[i] = int(trace.substep_count)
    batch.aggregate_support_status[i] = np.uint8(trace.aggregate_support_status)
    batch.local_error_resolved[i] = bool(trace.local_error_resolved)
    batch.endpoint_charge_C[i] = float(trace.endpoint_charge_C)
    batch.traces[i] = trace


def trace_coupled_charge_batch(
    request: SegmentMotionBatchRequest,
    destination: SegmentMotionBatchDestination,
    *,
    charge_start_C: np.ndarray,
    config: ChargeModelConfig,
    runtime: object,
    plasma_background: PreparedPlasmaBackground | None,
    physical_diameter_m: np.ndarray,
) -> CoupledChargeMotionBatch:
    """Trace active particles without freezing charge inside a solver segment."""

    start_charge = np.asarray(charge_start_C, dtype=np.float64).copy()
    endpoint_charge = start_charge.copy()
    tracers: dict[int, CoupledChargeParticleTracer] = {}
    traces: dict[int, CoupledChargeMotionTrace] = {}
    batch = CoupledChargeMotionBatch(
        request=request,
        endpoint_position_m=destination.endpoint_position_m,
        endpoint_velocity_mps=destination.endpoint_velocity_mps,
        midpoint_position_m=destination.midpoint_position_m,
        substep_count=destination.substep_count,
        aggregate_support_status=destination.aggregate_support_status,
        local_error_resolved=destination.local_error_resolved,
        endpoint_charge_C=endpoint_charge,
        start_charge_C=start_charge,
        tracers=tracers,
        traces=traces,
    )
    active = np.asarray(request.active, dtype=bool)
    diameters = np.asarray(physical_diameter_m, dtype=np.float64)
    dimension = int(request.spatial_dim)
    for index in range(int(active.size)):
        if not bool(active[index]):
            batch.endpoint_position_m[index, :dimension] = request.position_m[
                index, :dimension
            ]
            batch.endpoint_velocity_mps[index, :dimension] = request.velocity_mps[
                index, :dimension
            ]
            batch.midpoint_position_m[index, :dimension] = request.position_m[
                index, :dimension
            ]
            batch.substep_count[index] = 1
            batch.aggregate_support_status[index] = np.uint8(VALID_MASK_STATUS_CLEAN)
            batch.local_error_resolved[index] = True
            continue
        tracer = CoupledChargeParticleTracer(
            config=config,
            runtime=runtime,
            plasma_background=plasma_background,
            physical_diameter_m=float(diameters[index]),
        )
        tracers[index] = tracer
        trace = tracer.trace(
            request.particle_request(index),
            charge_start_C=float(start_charge[index]),
        )
        _commit_batch_trace(batch, index, trace)
    return batch


def resolve_coupled_charge_valid_mask_prefix(
    tracer: CoupledChargeParticleTracer,
    request: SegmentMotionRequest,
    *,
    charge_start_C: float,
    require_clean_prefix: bool = False,
) -> ValidMaskPrefixResolution:
    """Find the longest safe dyadic prefix using the coupled trace factory."""

    start_position = np.asarray(request.position_m, dtype=np.float64).copy()
    start_velocity = np.asarray(request.velocity_mps, dtype=np.float64).copy()
    duration = max(float(request.duration_s), 0.0)
    retries = int(max(0, request.adaptive_substep_max_splits))
    for split in range(1, retries + 1):
        prefix_duration = duration * (0.5**split)
        trace = tracer.trace(
            request.prefix(prefix_duration),
            charge_start_C=float(charge_start_C),
        )
        status = int(trace.aggregate_support_status)
        status_ok = (
            status == int(VALID_MASK_STATUS_CLEAN)
            if bool(require_clean_prefix)
            else not bool(valid_mask_status_requires_stop(status))
        )
        if bool(trace.local_error_resolved) and status_ok:
            return ValidMaskPrefixResolution(
                position=trace.endpoint_position_m,
                velocity=trace.endpoint_velocity_mps,
                accepted_dt=float(prefix_duration),
                retry_count=int(split),
                found_valid_prefix=True,
                charge_C=float(trace.endpoint_charge_C),
            )
    return ValidMaskPrefixResolution(
        position=start_position,
        velocity=start_velocity,
        accepted_dt=0.0,
        retry_count=int(retries),
        found_valid_prefix=False,
        charge_C=float(charge_start_C),
    )


__all__ = (
    "CoupledChargeMotionBatch",
    "CoupledChargeMotionTrace",
    "CoupledChargeParticleTracer",
    "resolve_coupled_charge_valid_mask_prefix",
    "trace_coupled_charge_batch",
    "trace_coupled_charge_motion",
)
