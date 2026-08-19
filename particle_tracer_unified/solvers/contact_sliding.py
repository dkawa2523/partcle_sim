"""Dispatch wall-contact sliding to its dimension-specific owner."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from particle_tracer_unified.domain import StageFields

from . import _contact_sliding_2d, _contact_sliding_3d
from ._contact_dynamics import (
    ContactDynamicsBatch,
    advance_contact_relaxation,
    displaced_fluid_factors,
)
from .runtime_execution import RunExecutionContext

StageSampler = Callable[..., StageFields]


def advance_contact_sliding_particles(
    execution: RunExecutionContext,
    *,
    body_acceleration: np.ndarray,
    duration_s: float,
    time_s: float,
    electric_q_over_m_particle: np.ndarray | None,
    sample_stage: StageSampler,
) -> None:
    """Advance all particles in the wall-contact state for one segment."""

    if int(execution.spatial_dim) == 2:
        _contact_sliding_2d.advance_contact_sliding_2d(
            execution,
            body_acceleration=body_acceleration,
            duration_s=float(duration_s),
            time_s=float(time_s),
            electric_q_over_m_particle=electric_q_over_m_particle,
            sample_stage=sample_stage,
        )
        return
    if int(execution.spatial_dim) == 3:
        _contact_sliding_3d.advance_contact_sliding_3d(
            execution,
            body_acceleration=body_acceleration,
            duration_s=float(duration_s),
            time_s=float(time_s),
            electric_q_over_m_particle=electric_q_over_m_particle,
            sample_stage=sample_stage,
        )
        return
    raise ValueError(
        f"contact sliding requires spatial_dim 2 or 3, got {execution.spatial_dim}"
    )


__all__ = (
    "ContactDynamicsBatch",
    "advance_contact_relaxation",
    "advance_contact_sliding_particles",
    "displaced_fluid_factors",
)
