"""Result types shared by collision detection stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.domain import BoundaryHit


@dataclass(frozen=True)
class TrialCollisionBatch:
    colliders: np.ndarray
    safe: np.ndarray
    prefetched_hits: dict[int, BoundaryHit]
