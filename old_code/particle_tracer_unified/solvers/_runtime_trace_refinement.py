"""Adaptive trace refinement for one free-flight solver step."""

from __future__ import annotations

import numpy as np

from particle_tracer_unified.core.boundary_service import sample_geometry_sdf
from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryQuery

from ._coupled_charge_motion import (
    CoupledChargeMotionBatch,
    CoupledChargeMotionTrace,
)
from .segment_motion import (
    SegmentMotionBatchTrace,
    SegmentMotionTrace,
)
from .segment_trace import (
    TraceRefinementDecision,
    TraceRefinementPolicy,
    assess_trace_geometry,
    geometry_probe_points,
    minimum_geometry_clearance,
    segment_length_required_substeps,
)


def _geometry_refinement_required(
    runtime,
    start: np.ndarray,
    stage_points: np.ndarray,
    *,
    on_boundary_tol_m: float,
) -> bool:
    assessment = assess_trace_geometry(
        start,
        stage_points,
        on_boundary_tolerance_m=float(on_boundary_tol_m),
    )
    if not assessment.needs_clearance:
        return assessment.requires_refinement(float("nan"))
    points = geometry_probe_points(start, stage_points)
    signed_distance = np.asarray(
        [sample_geometry_sdf(runtime, point) for point in points],
        dtype=np.float64,
    )
    # The clearance criterion asks how much room a curved path has before its
    # chord could hide a wall crossing.  The segment start is not such a place:
    # it is the committed state, its position is exact, and nothing can hide
    # there.  A particle released on its boundary, restarted after a
    # reflection, or held in contact starts with zero clearance by
    # construction, which would otherwise demand refinement no curvature can
    # ever satisfy.  Every other probe point still counts.
    if signed_distance.size and abs(float(signed_distance[0])) <= float(
        on_boundary_tol_m
    ):
        signed_distance = signed_distance[1:]
    if signed_distance.size == 0:
        return False
    return assessment.requires_refinement(minimum_geometry_clearance(signed_distance))


def _trace_refinement_decision(
    runtime,
    start: np.ndarray,
    trace: np.ndarray,
    *,
    current_substeps: int,
    policy: TraceRefinementPolicy,
    check_geometry: bool,
) -> TraceRefinementDecision:
    geometry_risk = bool(
        check_geometry
        and _geometry_refinement_required(
            runtime,
            start,
            trace,
            on_boundary_tol_m=policy.on_boundary_tolerance_m,
        )
    )
    return TraceRefinementDecision(
        geometry_risk=geometry_risk,
        support_substeps=segment_length_required_substeps(
            start,
            trace,
            current_substeps=current_substeps,
            target_length_m=policy.support_spacing_m,
            max_substeps=policy.max_substeps,
        ),
        max_substeps=policy.max_substeps,
        resolution_substeps=segment_length_required_substeps(
            start,
            trace,
            current_substeps=current_substeps,
            target_length_m=policy.interpolation_resolution_m,
            max_substeps=policy.max_substeps,
        ),
    )


def _replayed_trace_decision(
    runtime,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    start: np.ndarray,
    trace: np.ndarray,
    *,
    current_substeps: int,
    policy: TraceRefinementPolicy,
) -> TraceRefinementDecision:
    hit = boundary_service.polyline_hit(start, trace)
    if hit is not None:
        return TraceRefinementDecision(
            geometry_risk=False,
            support_substeps=current_substeps,
            max_substeps=policy.max_substeps,
        )
    return _trace_refinement_decision(
        runtime,
        start,
        trace,
        current_substeps=current_substeps,
        policy=policy,
        check_geometry=bool(np.all(boundary_service.contains(trace))),
    )


def _initial_refinement_trace(
    motion_batch: SegmentMotionBatchTrace | CoupledChargeMotionBatch,
    stage_traces: dict[int, np.ndarray],
    index: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    request = motion_batch.request
    dimension = int(request.spatial_dim)
    current_substeps = int(max(1, motion_batch.substep_count[index]))
    saved = stage_traces.get(index)
    complete = bool(
        saved is not None
        and np.asarray(saved).ndim == 2
        and int(np.asarray(saved).shape[0]) == 2 * current_substeps
    )
    if complete:
        trace = np.asarray(saved, dtype=np.float64)
    else:
        trace = np.stack(
            (
                motion_batch.midpoint_position_m[index, :dimension],
                motion_batch.endpoint_position_m[index, :dimension],
            )
        ).astype(np.float64, copy=False)
    start = np.asarray(request.position_m[index, :dimension], dtype=np.float64)
    return start, trace, current_substeps, complete


def _commit_refined_trace(
    motion_batch: SegmentMotionBatchTrace | CoupledChargeMotionBatch,
    index: int,
    motion_trace: SegmentMotionTrace | CoupledChargeMotionTrace,
) -> tuple[np.ndarray, int]:
    dimension = int(motion_batch.request.spatial_dim)
    trace = np.asarray(motion_trace.positions_m, dtype=np.float64)
    substeps = int(motion_trace.substep_count)
    motion_batch.endpoint_position_m[index, :dimension] = (
        motion_trace.endpoint_position_m[:dimension]
    )
    motion_batch.endpoint_velocity_mps[index, :dimension] = (
        motion_trace.endpoint_velocity_mps[:dimension]
    )
    motion_batch.midpoint_position_m[index, :dimension] = trace[
        substeps - 1, :dimension
    ]
    motion_batch.substep_count[index] = substeps
    motion_batch.aggregate_support_status[index] = np.uint8(
        motion_trace.aggregate_support_status
    )
    motion_batch.local_error_resolved[index] = bool(motion_trace.local_error_resolved)
    if isinstance(motion_batch, CoupledChargeMotionBatch):
        if not isinstance(motion_trace, CoupledChargeMotionTrace):
            raise RuntimeError("coupled motion batch returned an uncoupled trace")
        motion_batch.endpoint_charge_C[index] = float(motion_trace.endpoint_charge_C)
        motion_batch.traces[index] = motion_trace
    return trace, substeps


def _resolved_motion_indices(
    motion_batch: SegmentMotionBatchTrace | CoupledChargeMotionBatch,
    stage_traces: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, str]]:
    active = np.flatnonzero(np.asarray(motion_batch.request.active, dtype=bool))
    resolved = np.asarray(motion_batch.local_error_resolved, dtype=bool)
    unresolved_indices = active[~resolved[active]]
    for raw_index in unresolved_indices:
        stage_traces.pop(int(raw_index), None)
    return (
        active[resolved[active]],
        {int(index): "local_error" for index in unresolved_indices},
    )


def refine_deterministic_stage_traces(
    *,
    runtime,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    motion_batch: SegmentMotionBatchTrace | CoupledChargeMotionBatch,
    stage_traces: dict[int, np.ndarray],
    refinement_policy: TraceRefinementPolicy,
) -> dict[int, str]:
    """Refine traces and return particles whose safety proof hit its limit."""

    request = motion_batch.request
    if float(request.duration_s) <= 0.0:
        return {}
    resolved_indices, unresolved = _resolved_motion_indices(
        motion_batch,
        stage_traces,
    )
    for raw_index in resolved_indices:
        index = int(raw_index)
        start, trace, current_substeps, complete = _initial_refinement_trace(
            motion_batch,
            stage_traces,
            index,
        )
        decision = _trace_refinement_decision(
            runtime,
            start,
            trace,
            current_substeps=current_substeps,
            policy=refinement_policy,
            check_geometry=True,
        )
        if not decision.needs_replay(
            current_substeps=current_substeps,
            complete_trace=complete,
        ):
            if current_substeps == 1:
                stage_traces.pop(index, None)
            else:
                stage_traces[index] = trace.copy()
            continue

        minimum_substeps = decision.minimum_substeps(
            current_substeps=current_substeps,
        )
        while True:
            motion_trace = motion_batch.particle_trace(
                index,
                minimum_substeps=minimum_substeps,
            )
            trace, current_substeps = _commit_refined_trace(
                motion_batch,
                index,
                motion_trace,
            )
            if not bool(motion_trace.local_error_resolved):
                unresolved[index] = "local_error"
                break
            decision = _replayed_trace_decision(
                runtime,
                boundary_service,
                start,
                trace,
                current_substeps=current_substeps,
                policy=refinement_policy,
            )
            if decision.resolved(current_substeps=current_substeps):
                break
            if decision.limit_reached(current_substeps=current_substeps):
                unresolved[index] = (
                    "geometry" if decision.geometry_risk else "field_support"
                )
                break
            minimum_substeps = decision.minimum_substeps(
                current_substeps=current_substeps,
            )
        stage_traces[index] = trace.copy()
    return unresolved
