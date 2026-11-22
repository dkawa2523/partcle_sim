"""Exact particle/source selections shared by result consumers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chamber_particles.output import FinalParticles, ResultContext


def select_particles(
    final: FinalParticles,
    context: ResultContext,
    *,
    source_ids: tuple[int, ...] | None,
    particle_ids: tuple[int, ...] | None,
) -> NDArray[np.bool_]:
    """Intersect explicit ID sets, rejecting unknown or repeated requested IDs."""

    selected = np.ones(final.particle_id.size, dtype=np.bool_)
    for requested, available, label in (
        (source_ids, np.arange(len(context.source_names)), "source_ids"),
        (particle_ids, final.particle_id, "particle_ids"),
    ):
        if requested is None:
            continue
        if any(isinstance(value, bool) or not isinstance(value, int) for value in requested):
            raise ValueError(f"{label} must contain integer IDs")
        if len(set(requested)) != len(requested):
            raise ValueError(f"{label} must not repeat an ID")
        if not all(value in available for value in requested):
            raise ValueError(f"{label} contains an unknown ID")
        values = final.source_id if label == "source_ids" else final.particle_id
        selected &= np.isin(values, requested)
    return selected
