"""Exact 3D trial classification and broad-phase debug promotion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryHit, BoundaryQuery

from ._collision_detection_candidates import (
    polyline_stage_points,
    triangle_aabb_candidate_mask_3d,
)
from ._collision_detection_diagnostics import record_boundary_broad_phase_diagnostics
from ._collision_detection_types import TrialCollisionBatch
from .diagnostics import increment_count


@dataclass(frozen=True, slots=True)
class _ExactCollision:
    collides: bool
    hit: BoundaryHit | None
    midpoint_inside: bool = True
    on_boundary_count: int = 0


def classify_trial_collisions_3d(
    runtime,
    *,
    active: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    on_boundary_tol_m: float,
    collision_diagnostics: dict[str, object],
    boundary_broad_phase_enabled: bool = False,
    boundary_broad_phase_debug_check: bool = False,
) -> TrialCollisionBatch:
    colliders: list[int] = []
    safe: list[int] = []
    prefetched_hits: dict[int, BoundaryHit] = {}
    active_indices = np.flatnonzero(active)
    surface = boundary_service.triangle_surface_3d
    broad_candidates, broad_unknown = triangle_aabb_candidate_mask_3d(
        surface,
        active_indices,
        x,
        x_trial,
        x_mid_trial,
        on_boundary_tol_m=float(on_boundary_tol_m),
    )
    exact_indices, missed_hit_count = _select_exact_trials(
        active_indices=active_indices,
        broad_candidates=broad_candidates,
        x=x,
        x_trial=x_trial,
        x_mid_trial=x_mid_trial,
        boundary_service=boundary_service,
        broad_phase_enabled=bool(boundary_broad_phase_enabled),
        debug_check=bool(boundary_broad_phase_debug_check),
    )
    if boundary_broad_phase_enabled:
        exact_set = {int(value) for value in np.asarray(exact_indices, dtype=np.int64)}
        safe.extend(
            int(value) for value in active_indices if int(value) not in exact_set
        )
    record_boundary_broad_phase_diagnostics(
        collision_diagnostics,
        checked_count=int(active_indices.size),
        candidate_count=int(
            exact_indices.size if boundary_broad_phase_enabled else active_indices.size
        ),
        unknown_count=int(broad_unknown),
        exact_solve_count=int(exact_indices.size),
        missed_hit_count=int(missed_hit_count),
    )
    for raw_index in np.asarray(exact_indices, dtype=np.int64):
        index = int(raw_index)
        _record_exact_trial(
            index=index,
            classification=_classify_exact_trial(
                index=index,
                x=x,
                x_trial=x_trial,
                x_mid_trial=x_mid_trial,
                boundary_service=boundary_service,
            ),
            colliders=colliders,
            safe=safe,
            prefetched_hits=prefetched_hits,
            collision_diagnostics=collision_diagnostics,
        )
    return TrialCollisionBatch(
        colliders=np.asarray(colliders, dtype=np.int64),
        safe=np.asarray(safe, dtype=np.int64),
        prefetched_hits=prefetched_hits,
    )


def _classify_exact_trial(
    *,
    index: int,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
) -> _ExactCollision:
    hit = boundary_service.polyline_hit(
        x[index], polyline_stage_points(index, x_trial, x_mid_trial)
    )
    if hit is not None:
        return _ExactCollision(collides=True, hit=hit)
    inside_midpoint = boundary_service.inside(x_mid_trial[index])
    inside_endpoint = boundary_service.inside(x_trial[index])
    return _ExactCollision(
        collides=not bool(inside_midpoint) or not bool(inside_endpoint),
        hit=None,
        midpoint_inside=bool(inside_midpoint),
    )


def _select_exact_trials(
    *,
    active_indices: np.ndarray,
    broad_candidates: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
    broad_phase_enabled: bool,
    debug_check: bool,
) -> tuple[np.ndarray, int]:
    exact_indices = (
        active_indices[broad_candidates] if broad_phase_enabled else active_indices
    )
    if not broad_phase_enabled or not debug_check:
        return exact_indices, 0
    promoted = _promote_pruned_trials(
        pruned_indices=active_indices[~broad_candidates],
        x=x,
        x_trial=x_trial,
        x_mid_trial=x_mid_trial,
        boundary_service=boundary_service,
    )
    if not promoted.size:
        return exact_indices, 0
    return (
        np.unique(
            np.concatenate((np.asarray(exact_indices, dtype=np.int64), promoted))
        ),
        int(promoted.size),
    )


def _promote_pruned_trials(
    *,
    pruned_indices: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    boundary_service: BoundaryQuery[TriangleSurface3D],
) -> np.ndarray:
    promoted: list[int] = []
    for raw_index in np.asarray(pruned_indices, dtype=np.int64):
        index = int(raw_index)
        classification = _classify_exact_trial(
            index=index,
            x=x,
            x_trial=x_trial,
            x_mid_trial=x_mid_trial,
            boundary_service=boundary_service,
        )
        if classification.collides:
            promoted.append(index)
    return np.asarray(promoted, dtype=np.int64)


def _record_exact_trial(
    *,
    index: int,
    classification: _ExactCollision,
    colliders: list[int],
    safe: list[int],
    prefetched_hits: dict[int, BoundaryHit],
    collision_diagnostics: dict[str, object],
) -> None:
    if classification.hit is not None:
        prefetched_hits[index] = classification.hit
        colliders.append(index)
        return
    increment_count(
        collision_diagnostics,
        "on_boundary_promoted_inside_count",
        int(classification.on_boundary_count),
    )
    if not classification.midpoint_inside:
        increment_count(
            collision_diagnostics,
            "etd2_midpoint_outside_count",
            1,
        )
    if classification.collides:
        colliders.append(index)
    else:
        safe.append(index)
