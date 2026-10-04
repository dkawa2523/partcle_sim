"""Collision broad-phase diagnostic updates."""

from __future__ import annotations

from typing import Any, cast

from .diagnostics import increment_count


def _count(diagnostics: dict[str, object], key: str) -> int:
    return int(cast(Any, diagnostics.get(key, 0)))


def record_boundary_broad_phase_diagnostics(
    collision_diagnostics: dict[str, object],
    *,
    checked_count: int,
    candidate_count: int,
    unknown_count: int,
    exact_solve_count: int,
    missed_hit_count: int = 0,
) -> None:
    if not bool(getattr(collision_diagnostics, "debug", True)):
        return
    checked = int(max(0, checked_count))
    candidates = int(max(0, candidate_count))
    unknown = int(max(0, unknown_count))
    pruned = int(max(0, checked - candidates))
    exact_count = int(max(0, exact_solve_count))
    updates = (
        ("boundary_exact_solve_count", exact_count),
        ("boundary_broad_phase_checked_count", checked),
        ("boundary_broad_phase_candidate_count", candidates),
        ("boundary_broad_phase_pruned_count", pruned),
        ("boundary_broad_phase_missed_hit_count", int(max(0, missed_hit_count))),
        ("boundary_broad_phase_unknown_count", unknown),
    )
    for key, amount in updates:
        increment_count(collision_diagnostics, key, amount)
    total_checked = _count(collision_diagnostics, "boundary_broad_phase_checked_count")
    total_candidates = _count(
        collision_diagnostics, "boundary_broad_phase_candidate_count"
    )
    collision_diagnostics["boundary_broad_phase_candidate_ratio"] = (
        float(total_candidates) / float(total_checked) if total_checked > 0 else 0.0
    )
