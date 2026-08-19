"""Valid-mask retry and terminal fallback for a runtime step."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from particle_tracer_unified.core.coordinate_systems import (
    canonicalize_axisymmetric_rz_state,
)
from particle_tracer_unified.core.field_sampling import (
    valid_mask_status_requires_stop,
)

from ._coupled_charge_motion import (
    CoupledChargeMotionBatch,
    resolve_coupled_charge_valid_mask_prefix,
)
from ._runtime_terminal_state import mark_invalid_mask_stopped
from .diagnostics import increment_count
from .runtime_execution import RunExecutionContext
from .segment_motion import SegmentMotionRequest, ValidMaskPrefixResolution
from .stochastic_motion import PiecewiseLangevinPath
from .terminal_outcome import TerminalSegmentOutcome, terminal_segment_outcome
from .valid_mask_retry import resolve_valid_mask_retry_then_stop


def _violating_active_indices(
    execution: RunExecutionContext,
    particle_indices: np.ndarray | None,
) -> np.ndarray:
    state = execution.state
    if particle_indices is None:
        candidates = np.flatnonzero(state.active)
    else:
        candidates = np.asarray(particle_indices, dtype=np.int64)
        candidates = candidates[
            (candidates >= 0)
            & (candidates < int(state.active.size))
            & state.active[candidates]
        ]
    requires_stop = np.fromiter(
        (
            valid_mask_status_requires_stop(int(status))
            for status in state.valid_mask_status_flags[candidates]
        ),
        dtype=bool,
        count=int(candidates.size),
    )
    return candidates[requires_stop]


def _motion_request(
    execution: RunExecutionContext,
    *,
    index: int,
    dt_step: float,
    t_end_step: float,
    adaptive_substep_enabled: int,
    electric_q_over_m_particle: np.ndarray | None,
) -> SegmentMotionRequest:
    plan = execution.plan
    state = execution.state
    electric_q_over_m = (
        None
        if electric_q_over_m_particle is None
        else float(electric_q_over_m_particle[index])
    )
    return SegmentMotionRequest(
        position_m=np.asarray(state.x[index], dtype=np.float64).copy(),
        velocity_mps=np.asarray(state.v[index], dtype=np.float64).copy(),
        duration_s=float(dt_step),
        end_time_s=float(t_end_step),
        spatial_dim=int(execution.spatial_dim),
        backend=execution.compiled,
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_max_splits=int(plan.adaptive_substep_max_splits),
        tau_stokes_s=float(execution.tau_p[index]),
        particle_diameter_m=float(execution.particle_diameter[index]),
        particle_density_kgm3=float(execution.particle_density[index]),
        particle_mass_kg=float(execution.particle_mass[index]),
        dep_particle_rel_permittivity=float(
            execution.dep_particle_rel_permittivity[index]
        ),
        thermophoretic_coefficient=float(execution.thermophoretic_coeff[index]),
        body_acceleration_mps2=np.asarray(
            execution.body_acceleration_mps2, dtype=np.float64
        ),
        gas_density_kgm3=float(execution.physics["gas_density_kgm3"]),
        gas_dynamic_viscosity_Pas=float(execution.physics["gas_mu_pas"]),
        gas_temperature_K=float(execution.physics.get("gas_temperature_K", np.nan)),
        gas_molecular_mass_kg=float(
            execution.physics.get("gas_molecular_mass_kg", np.nan)
        ),
        drag_model_mode=int(plan.drag_model_mode),
        electric_q_over_m_Ckg=electric_q_over_m,
        force_runtime=execution.options.force_runtime,
    )


def _resolve_particle_prefix(
    execution: RunExecutionContext,
    *,
    index: int,
    request: SegmentMotionRequest,
    stochastic_paths: Mapping[int, PiecewiseLangevinPath] | None,
    coupled_motion_batch: CoupledChargeMotionBatch | None,
) -> ValidMaskPrefixResolution:
    state = execution.state
    if coupled_motion_batch is None:
        return resolve_valid_mask_retry_then_stop(
            request,
            collision_diagnostics=state.collision_diagnostics,
            require_clean_prefix=False,
            stochastic_path=(
                None if stochastic_paths is None else stochastic_paths.get(index)
            ),
            stochastic_offset_s=0.0,
        )

    resolution = resolve_coupled_charge_valid_mask_prefix(
        coupled_motion_batch.tracers[index],
        request,
        charge_start_C=float(coupled_motion_batch.start_charge_C[index]),
    )
    increment_count(
        state.collision_diagnostics,
        "invalid_mask_retry_count",
        int(resolution.retry_count),
    )
    if not bool(resolution.found_valid_prefix):
        increment_count(
            state.collision_diagnostics,
            "invalid_mask_retry_exhausted_count",
        )
    if resolution.charge_C is None:
        raise RuntimeError("coupled valid-mask prefix omitted charge state")
    state.charge[index] = float(resolution.charge_C)
    return resolution


def _commit_invalid_mask_stop(
    execution: RunExecutionContext,
    *,
    index: int,
    dt_step: float,
    resolution: ValidMaskPrefixResolution,
    terminal_outcomes: dict[int, TerminalSegmentOutcome],
) -> None:
    reason = (
        "freeflight_valid_mask_hard_invalid_prefix_clipped"
        if bool(resolution.found_valid_prefix)
        else "freeflight_valid_mask_hard_invalid_retry_exhausted"
    )
    position = np.asarray(resolution.position, dtype=np.float64)
    velocity = np.asarray(resolution.velocity, dtype=np.float64)
    if str(execution.context.coordinate_system) == "axisymmetric_rz":
        position, velocity = canonicalize_axisymmetric_rz_state(position, velocity)
    mark_invalid_mask_stopped(
        state=execution.state,
        particle_index=index,
        position=position,
        velocity=velocity,
        update_trial_buffers=True,
        reason=reason,
    )
    terminal_outcomes[index] = terminal_segment_outcome(
        accepted_elapsed_s=float(resolution.accepted_dt),
        segment_duration_s=float(dt_step),
        position=position,
        reason=reason,
    )


def apply_valid_mask_retry_then_stop(
    execution: RunExecutionContext,
    *,
    dt_step: float,
    t_end_step: float,
    adaptive_substep_enabled: int,
    terminal_outcomes: dict[int, TerminalSegmentOutcome],
    electric_q_over_m_particle: np.ndarray | None = None,
    particle_indices: np.ndarray | None = None,
    stochastic_paths: Mapping[int, PiecewiseLangevinPath] | None = None,
    coupled_motion_batch: CoupledChargeMotionBatch | None = None,
) -> int:
    violating = _violating_active_indices(execution, particle_indices)
    if violating.size == 0:
        return 0

    for raw_index in violating:
        index = int(raw_index)
        request = _motion_request(
            execution,
            index=index,
            dt_step=dt_step,
            t_end_step=t_end_step,
            adaptive_substep_enabled=adaptive_substep_enabled,
            electric_q_over_m_particle=electric_q_over_m_particle,
        )
        resolution = _resolve_particle_prefix(
            execution,
            index=index,
            request=request,
            stochastic_paths=stochastic_paths,
            coupled_motion_batch=coupled_motion_batch,
        )
        _commit_invalid_mask_stop(
            execution,
            index=index,
            dt_step=dt_step,
            resolution=resolution,
            terminal_outcomes=terminal_outcomes,
        )
    return int(violating.size)
