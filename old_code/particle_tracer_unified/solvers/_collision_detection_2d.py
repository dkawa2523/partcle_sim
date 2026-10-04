"""2D containment checks and exact edge-hit prefetching."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from particle_tracer_unified.core.boundary_service import (
    points_inside_geometry_2d,
    polyline_hits_from_boundary_edges_batch,
)
from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN
from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryHit, BoundaryQuery

from ._collision_detection_candidates import (
    edge_aabb_candidate_mask_2d,
    far_from_wall_mask_2d,
    polyline_stage_points,
    sdf_strict_inside_mask_2d,
)
from ._collision_detection_diagnostics import record_boundary_broad_phase_diagnostics
from .diagnostics import increment_count

TimerStart = Callable[[dict[str, float] | None], float]
TimerSince = Callable[[dict[str, float] | None, str, float], None]


def classify_trial_containment_2d(
    runtime,
    *,
    n_particles: int,
    active_indices: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    on_boundary_tol_m: float,
    collision_diagnostics: dict[str, object],
    timing_accumulator: dict[str, float] | None,
    valid_mask_status_flags: np.ndarray | None,
    timer_start: TimerStart,
    timer_since: TimerSince,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoint_inside = np.zeros(n_particles, dtype=bool)
    midpoint_inside = np.ones(n_particles, dtype=bool)
    far_from_wall = np.zeros(n_particles, dtype=bool)
    if not active_indices.size:
        return endpoint_inside, midpoint_inside, far_from_wall
    started_at = timer_start(timing_accumulator)
    far_active = far_from_wall_mask_2d(
        runtime,
        boundary_service,
        active_indices,
        x,
        x_mid_trial,
        x_trial,
        on_boundary_tol_m=float(on_boundary_tol_m),
    )
    if valid_mask_status_flags is not None:
        clean_status = np.asarray(
            valid_mask_status_flags[active_indices], dtype=np.uint8
        ) == np.uint8(VALID_MASK_STATUS_CLEAN)
        far_active &= clean_status
    timer_since(timing_accumulator, "boundary_sdf_prefilter_s", started_at)
    far_from_wall[active_indices] = far_active
    far_indices = active_indices[far_active]
    endpoint_inside[far_indices] = True
    midpoint_inside[far_indices] = True
    far_count = int(np.count_nonzero(far_active))
    increment_count(collision_diagnostics, "boundary_far_skip_count", far_count)
    increment_count(
        collision_diagnostics,
        "boundary_near_check_count",
        int(active_indices.size) - far_count,
    )
    near_indices = active_indices[~far_active]
    if not near_indices.size:
        return endpoint_inside, midpoint_inside, far_from_wall
    endpoint_inside[near_indices] = classify_positions_inside_2d(
        runtime,
        boundary_service=boundary_service,
        positions=x_trial[near_indices],
        on_boundary_tol_m=float(on_boundary_tol_m),
        collision_diagnostics=collision_diagnostics,
        timing_accumulator=timing_accumulator,
        record_midpoint_outside=False,
        timer_start=timer_start,
        timer_since=timer_since,
    )
    midpoint_inside[near_indices] = classify_positions_inside_2d(
        runtime,
        boundary_service=boundary_service,
        positions=x_mid_trial[near_indices],
        on_boundary_tol_m=float(on_boundary_tol_m),
        collision_diagnostics=collision_diagnostics,
        timing_accumulator=timing_accumulator,
        record_midpoint_outside=True,
        timer_start=timer_start,
        timer_since=timer_since,
    )
    return endpoint_inside, midpoint_inside, far_from_wall


def classify_positions_inside_2d(
    runtime,
    *,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    positions: np.ndarray,
    on_boundary_tol_m: float,
    collision_diagnostics: dict[str, object],
    timing_accumulator: dict[str, float] | None,
    record_midpoint_outside: bool,
    timer_start: TimerStart,
    timer_since: TimerSince,
) -> np.ndarray:
    started_at = timer_start(timing_accumulator)
    sdf_inside = sdf_strict_inside_mask_2d(
        runtime,
        boundary_service,
        positions,
        on_boundary_tol_m=float(on_boundary_tol_m),
    )
    timer_since(timing_accumulator, "inside_sdf_prefilter_s", started_at)
    inside = np.asarray(sdf_inside, dtype=bool).copy()
    exact_rows = np.flatnonzero(~sdf_inside)
    if not exact_rows.size:
        return inside
    started_at = timer_start(timing_accumulator)
    exact_inside, on_boundary = points_inside_geometry_2d(
        runtime,
        positions[exact_rows],
        on_boundary_tol_m=float(on_boundary_tol_m),
        return_on_boundary=True,
    )
    inside[exact_rows] = exact_inside
    increment_count(
        collision_diagnostics,
        "on_boundary_promoted_inside_count",
        int(np.count_nonzero(on_boundary)),
    )
    if record_midpoint_outside:
        increment_count(
            collision_diagnostics,
            "etd2_midpoint_outside_count",
            int(np.count_nonzero(~exact_inside)),
        )
    timer_since(timing_accumulator, "inside_check_s", started_at)
    return inside


def prefetch_safe_trial_hits_2d(
    runtime,
    *,
    safe_indices: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    on_boundary_tol_m: float,
    collision_diagnostics: dict[str, object],
    timing_accumulator: dict[str, float] | None,
    boundary_broad_phase_enabled: bool,
    boundary_broad_phase_debug_check: bool,
    collider_mask: np.ndarray,
    safe_mask: np.ndarray,
    timer_start: TimerStart,
    timer_since: TimerSince,
) -> dict[int, BoundaryHit]:
    prefetched_hits: dict[int, BoundaryHit] = {}
    broad_candidates, broad_unknown = edge_aabb_candidate_mask_2d(
        runtime,
        boundary_service,
        safe_indices,
        x,
        x_trial,
        x_mid_trial,
        on_boundary_tol_m=float(on_boundary_tol_m),
    )
    prefetch_indices = (
        safe_indices[broad_candidates] if boundary_broad_phase_enabled else safe_indices
    )
    missed_hit_count = 0
    if boundary_broad_phase_enabled and boundary_broad_phase_debug_check:
        prefetch_indices, missed_hit_count = _promote_pruned_safe_hits(
            pruned_indices=safe_indices[~broad_candidates],
            prefetch_indices=prefetch_indices,
            x=x,
            x_trial=x_trial,
            x_mid_trial=x_mid_trial,
            boundary_service=boundary_service,
            collider_mask=collider_mask,
            safe_mask=safe_mask,
            prefetched_hits=prefetched_hits,
        )
    prefetch_indices = np.unique(np.asarray(prefetch_indices, dtype=np.int64))
    started_at = timer_start(timing_accumulator)
    stage_points = np.stack(
        (x_mid_trial[prefetch_indices], x_trial[prefetch_indices]), axis=1
    )
    batch_hits = (
        polyline_hits_from_boundary_edges_batch(
            runtime,
            x[prefetch_indices],
            stage_points,
            particle_indices=prefetch_indices,
            coordinate_tolerance_m=float(on_boundary_tol_m),
            departure_tolerance_m=float(on_boundary_tol_m),
        )
        if prefetch_indices.size
        else {}
    )
    timer_since(timing_accumulator, "edge_prefetch_s", started_at)
    increment_count(
        collision_diagnostics,
        "edge_prefetch_batch_candidate_count",
        int(prefetch_indices.size),
    )
    for particle_index, hit in batch_hits.items():
        if float(hit.alpha_hint) <= 1.0e-12:
            continue
        index = int(particle_index)
        prefetched_hits[index] = hit
        collider_mask[index] = True
        safe_mask[index] = False
    increment_count(
        collision_diagnostics,
        "edge_prefetch_batch_hit_count",
        len(prefetched_hits),
    )
    record_boundary_broad_phase_diagnostics(
        collision_diagnostics,
        checked_count=int(safe_indices.size),
        candidate_count=int(
            prefetch_indices.size if boundary_broad_phase_enabled else safe_indices.size
        ),
        unknown_count=int(broad_unknown),
        exact_solve_count=int(prefetch_indices.size),
        missed_hit_count=int(missed_hit_count),
    )
    return prefetched_hits


def _promote_pruned_safe_hits(
    *,
    pruned_indices: np.ndarray,
    prefetch_indices: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    collider_mask: np.ndarray,
    safe_mask: np.ndarray,
    prefetched_hits: dict[int, BoundaryHit],
) -> tuple[np.ndarray, int]:
    missed_hit_count = 0
    for raw_index in np.asarray(pruned_indices, dtype=np.int64):
        index = int(raw_index)
        debug_hit = boundary_service.polyline_hit(
            x[index], polyline_stage_points(index, x_trial, x_mid_trial)
        )
        if debug_hit is None:
            continue
        missed_hit_count += 1
        prefetched_hits[index] = debug_hit
        collider_mask[index] = True
        safe_mask[index] = False
        prefetch_indices = np.append(prefetch_indices, index)
    return prefetch_indices, int(missed_hit_count)
