from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


INVALID_STOP_REASON_NAMES = {
    0: '',
    1: 'freeflight_valid_mask_hard_invalid_prefix_clipped',
    2: 'freeflight_valid_mask_hard_invalid_retry_exhausted',
    3: 'collision_valid_mask_hard_invalid_prefix_clipped',
    4: 'collision_valid_mask_hard_invalid_retry_exhausted',
    255: 'unknown',
}

INVALID_STOP_REASON_CODES = {name: code for code, name in INVALID_STOP_REASON_NAMES.items() if name}


def invalid_stop_reason_code(reason: str) -> int:
    return int(INVALID_STOP_REASON_CODES.get(str(reason).strip(), 255))


def invalid_stop_reason_name(code: int) -> str:
    return INVALID_STOP_REASON_NAMES.get(int(code), 'unknown') or 'unknown'


def invalid_stop_reason_names(reason_codes: np.ndarray) -> list[str]:
    codes = np.asarray(reason_codes, dtype=np.uint8)
    return [invalid_stop_reason_name(int(code)) for code in codes]


def increment_count(diagnostics: Dict[str, object], key: str, value: int = 1) -> None:
    diagnostics[str(key)] = int(diagnostics.get(str(key), 0)) + int(value)


def increment_named_count(diagnostics: Dict[str, object], key: str, name: str) -> None:
    reason_name = str(name).strip() or 'unknown'
    counts = diagnostics.setdefault(str(key), {})
    if not isinstance(counts, dict):
        counts = {}
        diagnostics[str(key)] = counts
    counts[reason_name] = int(counts.get(reason_name, 0)) + 1


def initial_collision_diagnostics() -> Dict[str, object]:
    counters: Mapping[str, object] = {
        'primary_hit_count': 0,
        'edge_hit_count': 0,
        'triangle_hit_count': 0,
        'bisection_fallback_count': 0,
        'nearest_projection_fallback_count': 0,
        'on_boundary_promoted_inside_count': 0,
        'unresolved_crossing_count': 0,
        'multi_hit_events_count': 0,
        'max_hits_reached_count': 0,
        'collision_reintegrated_segments_count': 0,
        'adaptive_substep_segments_count': 0,
        'adaptive_substep_trigger_count': 0,
        'max_hit_same_wall_count': 0,
        'max_hit_multi_wall_count': 0,
        'max_hit_remaining_dt_total_s': 0.0,
        'max_hit_remaining_dt_max_s': 0.0,
        'contact_sliding_count': 0,
        'contact_sliding_same_wall_count': 0,
        'contact_sliding_time_total_s': 0.0,
        'contact_sliding_remaining_dt_max_s': 0.0,
        'contact_tangent_step_count': 0,
        'contact_tangent_time_total_s': 0.0,
        'contact_release_count': 0,
        'contact_release_probe_reject_count': 0,
        'contact_endpoint_stop_count': 0,
        'contact_endpoint_hold_count': 0,
        'contact_frame_fail_count': 0,
        'contact_valid_mask_reject_count': 0,
        'etd2_polyline_checks_count': 0,
        'etd2_midpoint_outside_count': 0,
        'etd2_polyline_hit_count': 0,
        'etd2_polyline_fallback_count': 0,
        'edge_prefetch_batch_candidate_count': 0,
        'edge_prefetch_batch_hit_count': 0,
        'boundary_far_skip_count': 0,
        'boundary_near_check_count': 0,
        'boundary_ambiguous_hit_count': 0,
        'valid_mask_violation_count': 0,
        'valid_mask_violation_particle_count': 0,
        'valid_mask_mixed_stencil_count': 0,
        'valid_mask_mixed_stencil_particle_count': 0,
        'valid_mask_hard_invalid_count': 0,
        'valid_mask_hard_invalid_particle_count': 0,
        'invalid_mask_retry_count': 0,
        'invalid_mask_retry_exhausted_count': 0,
        'invalid_mask_stopped_count': 0,
        'numerical_boundary_stop_count': 0,
    }
    diagnostics = dict(counters)
    diagnostics.update(
        {
            'max_hit_last_part_counts': {},
            'max_hit_last_outcome_counts': {},
            'contact_sliding_part_counts': {},
            'contact_sliding_outcome_counts': {},
            'boundary_ambiguous_part_counts': {},
            'boundary_ambiguous_wall_law_counts': {},
            'boundary_ambiguous_primitive_kind_counts': {},
            'invalid_mask_stop_reason_counts': {},
            'numerical_boundary_stop_reason_counts': {},
        }
    )
    return diagnostics


__all__ = [
    'INVALID_STOP_REASON_NAMES',
    'increment_count',
    'increment_named_count',
    'initial_collision_diagnostics',
    'invalid_stop_reason_code',
    'invalid_stop_reason_name',
    'invalid_stop_reason_names',
]
