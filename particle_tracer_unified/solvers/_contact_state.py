"""Validate contact support and commit contact-state row transitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from particle_tracer_unified.core.field_sampling import (
    valid_mask_status_requires_stop,
)
from particle_tracer_unified.domain import StageFields

from . import _contact_dynamics
from .base_field_sampling import sample_compiled_valid_mask_statuses
from .runtime_execution import RunExecutionContext
from .runtime_state import SolverState
from .sampling_backend import VALID_MASK_STATUS

StageSampler = Callable[..., StageFields]


def _active_contact_indices(execution: RunExecutionContext) -> np.ndarray:
    state = execution.state
    contact_mask = state.active & state.contact_sliding
    if not _contact_dynamics._compiled_has_transient_time(execution.compiled):
        contact_mask &= ~state.contact_endpoint_stopped
    return np.flatnonzero(contact_mask)


def _clean_support(
    execution: RunExecutionContext,
    *,
    points: np.ndarray,
    time_s: float,
    inside: np.ndarray,
    sample_stage: StageSampler,
) -> np.ndarray:
    sampled = sample_stage(
        execution.state.collision_diagnostics,
        execution.compiled,
        execution.plan.stage_fields,
        points,
        float(time_s),
        spatial_dim=int(execution.spatial_dim),
        need_flow=False,
        need_gas_properties=False,
        need_valid_mask=True,
    )
    status = sampled.values.get(VALID_MASK_STATUS)
    if status is not None:
        status = np.asarray(status, dtype=np.uint8)
    if status is None:
        status = sample_compiled_valid_mask_statuses(execution.compiled, points)
    requires_stop = np.fromiter(
        (valid_mask_status_requires_stop(int(value)) for value in status),
        dtype=bool,
        count=int(status.size),
    )
    return np.asarray(inside, dtype=bool) & ~requires_stop


def _record_release_probe_rejections(
    diagnostics: dict[str, object], clean: np.ndarray
) -> None:
    rejected = int(np.count_nonzero(~np.asarray(clean, dtype=bool)))
    if rejected:
        diagnostics["contact_release_probe_reject_count"] = (
            int(
                cast(
                    int,
                    diagnostics.get("contact_release_probe_reject_count", 0),
                )
            )
            + rejected
        )


def _release_contact_rows(
    state: SolverState,
    indices: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> None:
    state.contact_sliding[indices] = False
    state.contact_endpoint_stopped[indices] = False
    state.contact_edge_index[indices] = -1
    state.contact_part_id[indices] = 0
    state.contact_normal[indices] = 0.0
    state.x[indices] = positions
    state.v[indices] = velocities
    diagnostics = state.collision_diagnostics
    diagnostics["contact_release_count"] = int(
        cast(int, diagnostics.get("contact_release_count", 0))
    ) + int(indices.size)


def _hold_contact_rows(
    state: SolverState,
    indices: np.ndarray,
    positions: np.ndarray,
    primitive_ids: np.ndarray,
    part_ids: np.ndarray,
    normals: np.ndarray,
) -> None:
    state.x[indices] = positions
    state.v[indices] = 0.0
    state.x_trial[indices] = state.x[indices]
    state.v_trial[indices] = state.v[indices]
    state.x_mid_trial[indices] = state.x[indices]
    state.contact_edge_index[indices] = primitive_ids
    state.contact_part_id[indices] = part_ids
    state.contact_normal[indices] = normals
    diagnostics = state.collision_diagnostics
    diagnostics["contact_endpoint_hold_count"] = int(
        cast(int, diagnostics.get("contact_endpoint_hold_count", 0))
    ) + int(indices.size)


def _reject_contact_rows(
    state: SolverState,
    indices: np.ndarray,
    contact_positions: np.ndarray,
) -> None:
    state.x[indices] = contact_positions
    state.v[indices] = 0.0
    state.x_trial[indices] = state.x[indices]
    state.v_trial[indices] = state.v[indices]
    state.x_mid_trial[indices] = state.x[indices]
    diagnostics = state.collision_diagnostics
    diagnostics["contact_valid_mask_reject_count"] = int(
        cast(int, diagnostics.get("contact_valid_mask_reject_count", 0))
    ) + int(indices.size)
