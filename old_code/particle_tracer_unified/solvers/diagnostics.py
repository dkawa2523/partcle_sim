from __future__ import annotations

from collections.abc import Mapping

import numpy as np

INVALID_STOP_REASON_NAMES = {
    0: "",
    1: "freeflight_valid_mask_hard_invalid_prefix_clipped",
    2: "freeflight_valid_mask_hard_invalid_retry_exhausted",
    3: "collision_valid_mask_hard_invalid_prefix_clipped",
    4: "collision_valid_mask_hard_invalid_retry_exhausted",
    5: "freeflight_field_support_refinement_exhausted",
    255: "unknown",
}

INVALID_STOP_REASON_CODES = {
    name: code for code, name in INVALID_STOP_REASON_NAMES.items() if name
}


def invalid_stop_reason_code(reason: str) -> int:
    return int(INVALID_STOP_REASON_CODES.get(str(reason).strip(), 255))


def invalid_stop_reason_name(code: int) -> str:
    return INVALID_STOP_REASON_NAMES.get(int(code), "unknown") or "unknown"


def invalid_stop_reason_names(reason_codes: np.ndarray) -> list[str]:
    codes = np.asarray(reason_codes, dtype=np.uint8)
    return [invalid_stop_reason_name(int(code)) for code in codes]


def increment_count(diagnostics: dict[str, object], key: str, value: int = 1) -> None:
    if isinstance(diagnostics, RuntimeDiagnostics) and not diagnostics.collects(key):
        return
    diagnostics[str(key)] = int(diagnostics.get(str(key), 0)) + int(value)


def increment_named_count(diagnostics: dict[str, object], key: str, name: str) -> None:
    if isinstance(diagnostics, RuntimeDiagnostics) and not diagnostics.collects(key):
        return
    reason_name = str(name).strip() or "unknown"
    counts = diagnostics.setdefault(str(key), {})
    if not isinstance(counts, dict):
        counts = {}
        diagnostics[str(key)] = counts
    counts[reason_name] = int(counts.get(reason_name, 0)) + 1


_STANDARD_DIAGNOSTIC_KEYS = frozenset(
    {
        "unresolved_crossing_count",
        "max_hits_reached_count",
        "valid_mask_violation_count",
        "valid_mask_hard_invalid_count",
        "invalid_mask_retry_exhausted_count",
        "invalid_mask_stopped_count",
        "numerical_boundary_stop_count",
        "invalid_mask_stop_reason_counts",
        "numerical_boundary_stop_reason_counts",
        "output_mode",
        "output_debug_enabled",
        "solver_step_count",
        "released_count_final",
    }
)


class RuntimeDiagnostics(dict[str, object]):
    """Dictionary-compatible diagnostics sink with a cheap standard mode.

    Existing numerical primitives can keep incrementing counters without
    branching.  In standard mode nonessential writes are discarded, so the
    payload remains small and nested diagnostic maps are never retained.
    """

    def __init__(self, initial: Mapping[str, object], *, debug: bool) -> None:
        super().__init__(initial)
        self.debug = bool(debug)

    def __missing__(self, key: str) -> object:
        return 0

    def collects(self, key: str) -> bool:
        return bool(self.debug or str(key) in _STANDARD_DIAGNOSTIC_KEYS)

    def __setitem__(self, key: str, value: object) -> None:
        if self.debug or str(key) in _STANDARD_DIAGNOSTIC_KEYS:
            super().__setitem__(str(key), value)

    def setdefault(self, key: str, default: object = None) -> object:
        name = str(key)
        if self.debug or name in _STANDARD_DIAGNOSTIC_KEYS:
            return super().setdefault(name, default)
        # Return an ephemeral value for callers that immediately mutate a
        # nested mapping.  It is intentionally not attached to this sink.
        return default


def initial_collision_diagnostics(*, debug: bool) -> dict[str, object]:
    if not bool(debug):
        return RuntimeDiagnostics(
            {
                "unresolved_crossing_count": 0,
                "max_hits_reached_count": 0,
                "valid_mask_violation_count": 0,
                "valid_mask_hard_invalid_count": 0,
                "invalid_mask_retry_exhausted_count": 0,
                "invalid_mask_stopped_count": 0,
                "numerical_boundary_stop_count": 0,
                "invalid_mask_stop_reason_counts": {},
                "numerical_boundary_stop_reason_counts": {},
            },
            debug=False,
        )
    counters: Mapping[str, object] = {
        "primary_hit_count": 0,
        "edge_hit_count": 0,
        "triangle_hit_count": 0,
        "bisection_fallback_count": 0,
        "nearest_projection_fallback_count": 0,
        "on_boundary_promoted_inside_count": 0,
        "unresolved_crossing_count": 0,
        "multi_hit_events_count": 0,
        "max_hits_reached_count": 0,
        "collision_reintegrated_segments_count": 0,
        "adaptive_substep_segments_count": 0,
        "adaptive_substep_trigger_count": 0,
        "adaptive_substep_limit_reached_count": 0,
        "max_hit_same_wall_count": 0,
        "max_hit_multi_wall_count": 0,
        "max_hit_remaining_dt_total_s": 0.0,
        "max_hit_remaining_dt_max_s": 0.0,
        "contact_sliding_count": 0,
        "contact_sliding_same_wall_count": 0,
        "contact_sliding_time_total_s": 0.0,
        "contact_sliding_remaining_dt_max_s": 0.0,
        "contact_tangent_step_count": 0,
        "contact_tangent_time_total_s": 0.0,
        "contact_release_count": 0,
        "contact_release_probe_reject_count": 0,
        "contact_endpoint_stop_count": 0,
        "contact_endpoint_hold_count": 0,
        "contact_frame_fail_count": 0,
        "contact_valid_mask_reject_count": 0,
        "etd2_polyline_checks_count": 0,
        "etd2_midpoint_outside_count": 0,
        "etd2_polyline_hit_count": 0,
        "etd2_polyline_fallback_count": 0,
        "edge_prefetch_batch_candidate_count": 0,
        "edge_prefetch_batch_hit_count": 0,
        "boundary_far_skip_count": 0,
        "boundary_near_check_count": 0,
        "boundary_exact_solve_count": 0,
        "boundary_broad_phase_checked_count": 0,
        "boundary_broad_phase_candidate_count": 0,
        "boundary_broad_phase_pruned_count": 0,
        "boundary_broad_phase_missed_hit_count": 0,
        "boundary_broad_phase_unknown_count": 0,
        "boundary_broad_phase_candidate_ratio": 0.0,
        "boundary_ambiguous_hit_count": 0,
        "valid_mask_violation_count": 0,
        "valid_mask_violation_particle_count": 0,
        "valid_mask_mixed_stencil_count": 0,
        "valid_mask_mixed_stencil_particle_count": 0,
        "valid_mask_hard_invalid_count": 0,
        "valid_mask_hard_invalid_particle_count": 0,
        "invalid_mask_retry_count": 0,
        "invalid_mask_retry_exhausted_count": 0,
        "invalid_mask_stopped_count": 0,
        "numerical_boundary_stop_count": 0,
    }
    nested = {
        "max_hit_last_part_counts": {},
        "max_hit_last_outcome_counts": {},
        "contact_sliding_part_counts": {},
        "contact_sliding_outcome_counts": {},
        "boundary_ambiguous_part_counts": {},
        "boundary_ambiguous_wall_law_counts": {},
        "boundary_ambiguous_primitive_kind_counts": {},
        "invalid_mask_stop_reason_counts": {},
        "numerical_boundary_stop_reason_counts": {},
    }
    return RuntimeDiagnostics({**counters, **nested}, debug=True)


__all__ = [
    "INVALID_STOP_REASON_NAMES",
    "RuntimeDiagnostics",
    "increment_count",
    "increment_named_count",
    "initial_collision_diagnostics",
    "invalid_stop_reason_code",
    "invalid_stop_reason_name",
    "invalid_stop_reason_names",
]
