from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TerminalSegmentOutcome:
    """Accepted part of one solver segment before a particle became terminal."""

    accepted_elapsed_s: float
    position: np.ndarray
    reason: str


def terminal_segment_outcome(
    *,
    accepted_elapsed_s: float,
    segment_duration_s: float,
    position: np.ndarray,
    reason: str,
) -> TerminalSegmentOutcome:
    duration = float(segment_duration_s)
    elapsed = float(accepted_elapsed_s)
    point = np.asarray(position, dtype=np.float64)
    reason_text = str(reason).strip()
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("terminal segment duration must be finite and non-negative")
    if not np.isfinite(elapsed) or elapsed < 0.0 or elapsed > duration:
        raise ValueError(
            "terminal accepted elapsed time must be finite and within the segment"
        )
    if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("terminal position must be a finite coordinate vector")
    if not reason_text:
        raise ValueError("terminal reason must be non-empty")
    accepted_position = point.copy()
    accepted_position.setflags(write=False)
    return TerminalSegmentOutcome(
        accepted_elapsed_s=elapsed,
        position=accepted_position,
        reason=reason_text,
    )


__all__ = ["TerminalSegmentOutcome", "terminal_segment_outcome"]
