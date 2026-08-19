"""Promote deterministic stage traces into trial collisions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryHit, BoundaryQuery

from ._collision_detection_types import TrialCollisionBatch


def _active_trace_index(raw_index: int, active: np.ndarray) -> int | None:
    index = int(raw_index)
    if index < 0 or index >= active.size or not bool(active[index]):
        return None
    return index


def _classify_trace(
    start: np.ndarray,
    raw_trace: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
) -> tuple[bool, BoundaryHit | None]:
    trace = np.asarray(raw_trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[0] == 0 or not np.all(np.isfinite(trace)):
        return True, None
    hit = boundary_service.polyline_hit(start, trace)
    inside = boundary_service.contains(trace)
    return hit is not None or not bool(np.all(inside)), hit


def promote_stage_trace_collisions(
    trial: TrialCollisionBatch,
    *,
    active: np.ndarray,
    x_start: np.ndarray,
    stage_traces: Mapping[int, np.ndarray],
    boundary_service: BoundaryQuery[TriangleSurface3D],
) -> TrialCollisionBatch:
    """Reclassify trials with every accepted deterministic stage point."""

    if not stage_traces:
        return trial
    colliders = {int(index) for index in np.asarray(trial.colliders, dtype=np.int64)}
    safe = {int(index) for index in np.asarray(trial.safe, dtype=np.int64)}
    prefetched = dict(trial.prefetched_hits)
    active_array = np.asarray(active, dtype=bool)
    starts = np.asarray(x_start, dtype=np.float64)
    for raw_index, raw_trace in stage_traces.items():
        index = _active_trace_index(raw_index, active_array)
        if index is None:
            continue
        collides, hit = _classify_trace(starts[index], raw_trace, boundary_service)
        if not collides:
            continue
        colliders.add(index)
        safe.discard(index)
        if hit is not None:
            prefetched[index] = hit
    return TrialCollisionBatch(
        colliders=np.asarray(sorted(colliders), dtype=np.int64),
        safe=np.asarray(sorted(safe), dtype=np.int64),
        prefetched_hits=prefetched,
    )
