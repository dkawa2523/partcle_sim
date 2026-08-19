"""Build independent particle cohorts inside one nominal solver step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .particle_state import activate_release_cursor_until
from .runtime_plan import ReleaseCursor


@dataclass(frozen=True)
class ParticleStepInterval:
    """Particles sharing one physical integration interval in a macro step."""

    particle_indices: np.ndarray
    start_s: float
    end_s: float


def particle_intervals_for_macro_step(
    *,
    cursor: ReleaseCursor,
    released: np.ndarray,
    active: np.ndarray,
    start_s: float,
    end_s: float,
) -> tuple[ParticleStepInterval, ...]:
    """Group active particles by their integration start time.

    The solver is one-way coupled, so a release event must not shorten the
    nominal step of particles that were already active.  Particles released
    inside the step form a new cohort and integrate only from their release
    time to the same nominal step boundary.  A release exactly at ``end_s``
    belongs to the next step and therefore receives zero integration age here.
    """

    step_start = float(start_s)
    step_end = float(end_s)
    activate_release_cursor_until(cursor, released, active, step_start)

    intervals: list[ParticleStepInterval] = []
    active_at_start = np.flatnonzero(active)
    if active_at_start.size:
        intervals.append(ParticleStepInterval(active_at_start, step_start, step_end))

    while True:
        release_time = float(cursor.next_time())
        if not np.isfinite(release_time) or release_time >= step_end:
            break
        activated = activate_release_cursor_until(
            cursor,
            released,
            active,
            release_time,
        )
        intervals.append(ParticleStepInterval(activated, release_time, step_end))

    return tuple(intervals)


__all__ = ("ParticleStepInterval", "particle_intervals_for_macro_step")
