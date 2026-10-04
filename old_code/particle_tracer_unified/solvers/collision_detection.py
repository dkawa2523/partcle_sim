"""Classify trial motion as safe or colliding and dispatch by dimension."""

from __future__ import annotations

import numpy as np

from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryQuery

from ._collision_detection_2d import (
    classify_trial_containment_2d,
    prefetch_safe_trial_hits_2d,
)
from ._collision_detection_3d import classify_trial_collisions_3d
from ._collision_detection_candidates import edge_aabb_candidate_mask_2d
from ._collision_detection_diagnostics import (
    record_boundary_broad_phase_diagnostics,
)
from ._collision_detection_trace import promote_stage_trace_collisions
from ._collision_detection_types import TrialCollisionBatch
from ._runtime_timing import (
    add_detailed_timing_since as _add_detailed_timing_since,
)
from ._runtime_timing import detailed_timer_start as _detailed_timer_start


def classify_trial_collisions_2d(
    runtime,
    *,
    n_particles: int,
    active: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    on_boundary_tol_m: float,
    collision_diagnostics: dict[str, object],
    timing_accumulator: dict[str, float] | None = None,
    valid_mask_status_flags: np.ndarray | None = None,
    boundary_broad_phase_enabled: bool = False,
    boundary_broad_phase_debug_check: bool = False,
) -> TrialCollisionBatch:
    active_indices = np.flatnonzero(active)
    endpoint_inside, midpoint_inside, far_from_wall = classify_trial_containment_2d(
        runtime,
        n_particles=int(n_particles),
        active_indices=active_indices,
        x=x,
        x_trial=x_trial,
        x_mid_trial=x_mid_trial,
        boundary_service=boundary_service,
        on_boundary_tol_m=float(on_boundary_tol_m),
        collision_diagnostics=collision_diagnostics,
        timing_accumulator=timing_accumulator,
        valid_mask_status_flags=valid_mask_status_flags,
        timer_start=_detailed_timer_start,
        timer_since=_add_detailed_timing_since,
    )
    collider_mask = active & ((~endpoint_inside) | (~midpoint_inside))
    safe_mask = active & endpoint_inside & midpoint_inside
    safe_indices = np.flatnonzero(safe_mask & (~far_from_wall))
    prefetched_hits = (
        prefetch_safe_trial_hits_2d(
            runtime,
            safe_indices=safe_indices,
            x=x,
            x_trial=x_trial,
            x_mid_trial=x_mid_trial,
            boundary_service=boundary_service,
            on_boundary_tol_m=float(on_boundary_tol_m),
            collision_diagnostics=collision_diagnostics,
            timing_accumulator=timing_accumulator,
            boundary_broad_phase_enabled=bool(boundary_broad_phase_enabled),
            boundary_broad_phase_debug_check=bool(boundary_broad_phase_debug_check),
            collider_mask=collider_mask,
            safe_mask=safe_mask,
            timer_start=_detailed_timer_start,
            timer_since=_add_detailed_timing_since,
        )
        if safe_indices.size
        else {}
    )
    colliders = np.flatnonzero(collider_mask)
    safe = np.flatnonzero(safe_mask)
    _, broad_unknown = edge_aabb_candidate_mask_2d(
        runtime,
        boundary_service,
        colliders,
        x,
        x_trial,
        x_mid_trial,
        on_boundary_tol_m=float(on_boundary_tol_m),
    )
    record_boundary_broad_phase_diagnostics(
        collision_diagnostics,
        checked_count=int(colliders.size),
        candidate_count=int(colliders.size),
        unknown_count=int(broad_unknown),
        exact_solve_count=int(colliders.size),
    )
    return TrialCollisionBatch(
        colliders=np.asarray(colliders, dtype=np.int64),
        safe=np.asarray(safe, dtype=np.int64),
        prefetched_hits=prefetched_hits,
    )


def classify_trial_collisions(
    runtime,
    *,
    spatial_dim: int,
    n_particles: int,
    active: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    on_boundary_tol_m: float,
    collision_diagnostics: dict[str, object],
    timing_accumulator: dict[str, float] | None = None,
    valid_mask_status_flags: np.ndarray | None = None,
    boundary_broad_phase_enabled: bool = False,
    boundary_broad_phase_debug_check: bool = False,
) -> TrialCollisionBatch:
    if int(spatial_dim) == 2:
        return classify_trial_collisions_2d(
            runtime,
            n_particles=int(n_particles),
            active=active,
            x=x,
            x_trial=x_trial,
            x_mid_trial=x_mid_trial,
            boundary_service=boundary_service,
            on_boundary_tol_m=float(on_boundary_tol_m),
            collision_diagnostics=collision_diagnostics,
            timing_accumulator=timing_accumulator,
            valid_mask_status_flags=valid_mask_status_flags,
            boundary_broad_phase_enabled=bool(boundary_broad_phase_enabled),
            boundary_broad_phase_debug_check=bool(boundary_broad_phase_debug_check),
        )
    return classify_trial_collisions_3d(
        runtime,
        active=active,
        x=x,
        x_trial=x_trial,
        x_mid_trial=x_mid_trial,
        boundary_service=boundary_service,
        on_boundary_tol_m=float(on_boundary_tol_m),
        collision_diagnostics=collision_diagnostics,
        boundary_broad_phase_enabled=bool(boundary_broad_phase_enabled),
        boundary_broad_phase_debug_check=bool(boundary_broad_phase_debug_check),
    )


__all__ = (
    "TrialCollisionBatch",
    "classify_trial_collisions",
    "promote_stage_trace_collisions",
)
