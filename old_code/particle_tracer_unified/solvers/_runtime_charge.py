"""Charge lifecycle for one high-fidelity runtime segment."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .charge_model import (
    advance_charge_strang_segment,
    apply_charge_model_update,
    merge_charge_model_diagnostics,
    record_terminal_charge_replay,
)
from .kernel_shared_numba import time_roundoff_tolerance
from .runtime_execution import RunExecutionContext
from .terminal_outcome import TerminalSegmentOutcome


def electric_q_over_m_particle(
    *,
    electric_force_enabled: bool,
    charge: np.ndarray,
    particle_mass: np.ndarray,
) -> np.ndarray | None:
    if not bool(electric_force_enabled):
        return None
    charge_arr = np.asarray(charge, dtype=np.float64)
    mass_arr = np.asarray(particle_mass, dtype=np.float64)
    valid_charge = np.isfinite(charge_arr) & (np.abs(charge_arr) > 0.0)
    if not np.any(valid_charge):
        return None
    qom = np.zeros_like(charge_arr, dtype=np.float64)
    np.divide(
        charge_arr,
        mass_arr,
        out=qom,
        where=np.isfinite(mass_arr) & (mass_arr > 0.0),
    )
    return qom


def _update_charge_half_step(
    execution: RunExecutionContext,
    *,
    t_eval: float,
    dt_step: float,
    timer_start,
    record_timing,
) -> None:
    state = execution.state
    options = execution.options
    plan = execution.plan
    detailed_timing = state.timing_accumulator if bool(plan.output.is_debug) else None
    started_at = timer_start(detailed_timing)
    result = apply_charge_model_update(
        config=options.charge_model,
        runtime=execution.context,
        spatial_dim=int(execution.spatial_dim),
        t_eval=float(t_eval),
        delta_t_s=0.5 * float(dt_step),
        active_mask=state.active,
        x=state.x,
        charge=state.charge,
        particle_diameter=execution.particle_physical_diameter,
        plasma_background=options.plasma_background,
        collect_diagnostics=bool(plan.output.is_debug),
    )
    if bool(plan.output.is_debug):
        merge_charge_model_diagnostics(
            state.collision_diagnostics,
            options.charge_model,
            result,
        )
    record_timing(detailed_timing, "charge_model_s", started_at)


def begin_charge_segment(
    execution: RunExecutionContext,
    *,
    t_start: float,
    dt_step: float,
    timer_start,
    record_timing,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not bool(execution.options.charge_model.enabled):
        return None
    state = execution.state
    snapshot = (
        np.asarray(state.charge, dtype=np.float64).copy(),
        np.asarray(state.x, dtype=np.float64).copy(),
    )
    _update_charge_half_step(
        execution,
        t_eval=t_start,
        dt_step=dt_step,
        timer_start=timer_start,
        record_timing=record_timing,
    )
    return snapshot


def _replay_terminal_charge(
    execution: RunExecutionContext,
    *,
    t_start_s: float,
    segment_duration_s: float,
    electric_force_enabled: bool,
    charge_segment_start: np.ndarray,
    charge_position_start: np.ndarray,
    terminal_outcomes: Mapping[int, TerminalSegmentOutcome],
) -> None:
    state = execution.state
    options = execution.options
    time_tolerance_s = float(
        time_roundoff_tolerance(
            float(t_start_s) + float(segment_duration_s),
            float(segment_duration_s),
        )
    )
    for particle_index, outcome in terminal_outcomes.items():
        index = int(particle_index)
        accepted_elapsed_s = float(outcome.accepted_elapsed_s)
        if accepted_elapsed_s == 0.0:
            state.charge[index] = float(charge_segment_start[index])
        elif (
            bool(electric_force_enabled)
            and accepted_elapsed_s < float(segment_duration_s) - time_tolerance_s
        ):
            raise ValueError(
                "dynamic charge with electric force is not supported for a "
                "terminal event "
                f"before the segment endpoint ({outcome.reason}); "
                "event-local charge-motion iteration is required"
            )
        else:
            state.charge[index] = advance_charge_strang_segment(
                config=options.charge_model,
                runtime=execution.context,
                spatial_dim=int(execution.spatial_dim),
                t_start_s=float(t_start_s),
                duration_s=accepted_elapsed_s,
                x_start=np.asarray(
                    charge_position_start[index : index + 1], dtype=np.float64
                ),
                x_end=np.asarray(outcome.position, dtype=np.float64).reshape(
                    1, int(execution.spatial_dim)
                ),
                charge_start=np.asarray(
                    charge_segment_start[index : index + 1], dtype=np.float64
                ),
                particle_diameter=np.asarray(
                    execution.particle_physical_diameter[index : index + 1],
                    dtype=np.float64,
                ),
                plasma_background=options.plasma_background,
            )[0]
        if bool(state.solver_plan.output.is_debug):
            record_terminal_charge_replay(
                state.collision_diagnostics,
                options.charge_model,
                age_s=accepted_elapsed_s,
            )


def finish_charge_segment(
    execution: RunExecutionContext,
    *,
    snapshot: tuple[np.ndarray, np.ndarray] | None,
    t_start: float,
    t_next: float,
    dt_step: float,
    electric_force_enabled: bool,
    terminal_outcomes: Mapping[int, TerminalSegmentOutcome],
    timer_start,
    record_timing,
) -> None:
    if snapshot is None:
        return
    charge_start, position_start = snapshot
    _replay_terminal_charge(
        execution,
        t_start_s=float(t_start),
        segment_duration_s=float(dt_step),
        electric_force_enabled=bool(electric_force_enabled),
        charge_segment_start=charge_start,
        charge_position_start=position_start,
        terminal_outcomes=terminal_outcomes,
    )
    _update_charge_half_step(
        execution,
        t_eval=t_next,
        dt_step=dt_step,
        timer_start=timer_start,
        record_timing=record_timing,
    )
