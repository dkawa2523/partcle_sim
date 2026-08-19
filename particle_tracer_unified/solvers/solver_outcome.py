"""Internal numerical outcome passed to the application result assembler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolverDebugOutcome:
    trajectory_positions: np.ndarray
    save_frames: list[dict[str, object]]
    wall_events: list[dict[str, object]]
    max_hit_events: list[dict[str, object]]
    step_summary: list[dict[str, object]]


@dataclass(frozen=True)
class SolverOutcome:
    final_position: np.ndarray
    final_velocity: np.ndarray
    final_charge: np.ndarray
    released: np.ndarray
    active: np.ndarray
    stuck: np.ndarray
    frozen: np.ndarray
    absorbed: np.ndarray
    contact_sliding: np.ndarray
    contact_endpoint_stopped: np.ndarray
    contact_part_id: np.ndarray
    contact_normal: np.ndarray
    escaped: np.ndarray
    invalid_mask_stopped: np.ndarray
    numerical_boundary_stopped: np.ndarray
    invalid_stop_reason_code: np.ndarray
    final_step_name: str
    final_segment_name: str
    wall_summary_counts: dict[tuple[int, str, str], int]
    collision_diagnostics: dict[str, object]
    timing_s: dict[str, float]
    memory_estimate_bytes: dict[str, int]
    debug: SolverDebugOutcome | None = None


__all__ = ("SolverDebugOutcome", "SolverOutcome")
