"""Debug snapshots, runtime accounting, and final solver outcome assembly."""

from __future__ import annotations

import time

import numpy as np

from ._runtime_execution_context import RunExecutionContext, StepLoopResult
from .charge_model import finalize_charge_model_diagnostics
from .compiled_backend_types import CompiledRuntimeBackend
from .kernel_shared_numba import time_roundoff_tolerance
from .output_buffers import DebugBuffers
from .runtime_plan import SolverPlan
from .solver_outcome import SolverDebugOutcome, SolverOutcome

_RUN_NAME = "run"

_COMPILED_MEMORY_ATTRS = (
    "axes",
    "times",
    "ux",
    "uy",
    "uz",
    "electric_x",
    "electric_y",
    "electric_z",
    "gas_density",
    "gas_mu",
    "gas_temperature",
    "valid_mask",
    "core_valid_mask",
    "mesh_vertices",
    "mesh_triangles",
    "accel_origin",
    "accel_cell_size",
    "accel_cell_offsets",
    "accel_triangle_indices",
)


def initialize_debug_buffers(
    plan: SolverPlan, *, capture_outputs: bool
) -> DebugBuffers | None:
    if not bool(plan.output.is_debug) or not bool(capture_outputs):
        return None
    return DebugBuffers()


def append_snapshot(
    save_positions: list[np.ndarray],
    save_meta: list[dict[str, object]],
    *,
    save_index: int,
    t: float,
    position: np.ndarray,
) -> None:
    save_positions.append(np.asarray(position, dtype=np.float64).copy())
    save_meta.append(
        {
            "save_index": int(save_index),
            "time_s": float(t),
            "step_name": _RUN_NAME,
            "segment_name": _RUN_NAME,
        }
    )


def _array_nbytes_once(value: object, seen: set[int]) -> int:
    if value is None:
        return 0
    if isinstance(value, tuple):
        return int(sum(_array_nbytes_once(item, seen) for item in value))
    array = np.asarray(value)
    ident = id(array)
    if ident in seen:
        return 0
    seen.add(ident)
    return int(array.nbytes)


def _compiled_backend_array_bytes(compiled: CompiledRuntimeBackend) -> int:
    seen: set[int] = set()
    return int(
        sum(
            _array_nbytes_once(getattr(compiled, name, None), seen)
            for name in _COMPILED_MEMORY_ATTRS
        )
    )


def _assemble_saved_positions(
    save_positions: list[np.ndarray],
    *,
    n_particles: int,
    spatial_dim: int,
) -> tuple[np.ndarray, float]:
    assembly_started_s = time.perf_counter()
    positions = (
        np.stack(save_positions, axis=0)
        if save_positions
        else np.zeros((0, int(n_particles), int(spatial_dim)), dtype=np.float64)
    )
    return positions, float(time.perf_counter() - assembly_started_s)


def _finite_q_over_m_summary(
    charge: np.ndarray, particle_mass: np.ndarray
) -> dict[str, int | float]:
    charge_arr = np.asarray(charge, dtype=np.float64)
    mass_arr = np.asarray(particle_mass, dtype=np.float64)
    qom = np.full_like(charge_arr, np.nan, dtype=np.float64)
    valid = np.isfinite(charge_arr) & np.isfinite(mass_arr) & (mass_arr > 0.0)
    np.divide(charge_arr, mass_arr, out=qom, where=valid)
    finite = qom[np.isfinite(qom)]
    charged = finite[np.abs(finite) > 0.0]
    if finite.size == 0:
        return {"count": 0, "charged_count": 0}
    quantiles = np.quantile(finite, [0.0, 0.5, 0.9, 1.0])
    return {
        "count": int(finite.size),
        "charged_count": int(charged.size),
        "min": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "max": float(quantiles[3]),
    }


def _record_final_state_diagnostics(
    prepared: RunExecutionContext,
    loop_result: StepLoopResult,
) -> None:
    state = prepared.state
    state.collision_diagnostics["solver_step_count"] = int(loop_result.step_count)
    state.collision_diagnostics["released_count_final"] = int(
        np.count_nonzero(state.released)
    )
    if not bool(prepared.plan.output.is_debug):
        return
    state.collision_diagnostics["release_cursor_position_final"] = int(
        state.release_cursor.position
    )
    state.collision_diagnostics["release_cursor_done"] = int(
        bool(state.release_cursor.done)
    )
    state.collision_diagnostics.setdefault("active_count_samples", 0)
    state.collision_diagnostics.setdefault("active_count_mean", 0.0)
    state.collision_diagnostics.setdefault(
        "active_count_max",
        int(np.count_nonzero(state.active)),
    )


def _append_final_snapshot(
    prepared: RunExecutionContext,
    t: float,
) -> None:
    buffers = prepared.state.debug_buffers
    if buffers is None:
        return
    frames = buffers.save_frames
    if frames and (
        abs(float(np.asarray(frames[-1]["time_s"]).item()) - float(t))
        <= time_roundoff_tolerance(float(t), float(prepared.plan.dt))
    ):
        return
    append_snapshot(
        buffers.trajectory_positions,
        frames,
        save_index=int(prepared.state.save_index),
        t=float(t),
        position=prepared.state.x,
    )


def _debug_outcome(
    prepared: RunExecutionContext,
) -> tuple[SolverDebugOutcome | None, float, int]:
    buffers = prepared.state.debug_buffers
    if buffers is None:
        return None, 0.0, 0
    positions, assembly_s = _assemble_saved_positions(
        buffers.trajectory_positions,
        n_particles=prepared.n_particles,
        spatial_dim=prepared.spatial_dim,
    )
    return (
        SolverDebugOutcome(
            trajectory_positions=positions,
            save_frames=buffers.save_frames,
            wall_events=buffers.wall_events,
            max_hit_events=buffers.max_hit_events,
            step_summary=buffers.step_summary.as_runtime_step_rows(),
        ),
        assembly_s,
        int(positions.nbytes),
    )


def _core_array_bytes(prepared: RunExecutionContext) -> int:
    state = prepared.state
    return int(
        sum(
            int(np.asarray(array).nbytes)
            for array in (
                state.x,
                state.v,
                state.released,
                state.active,
                state.stuck,
                state.frozen,
                state.absorbed,
                state.escaped,
                state.invalid_mask_stopped,
                state.numerical_boundary_stopped,
                state.invalid_stop_reason_code,
                state.x_trial,
                state.v_trial,
                state.x_mid_trial,
                state.substep_counts,
                state.valid_mask_status_flags,
                state.valid_mask_mixed_seen,
                state.valid_mask_hard_seen,
                state.charge,
                prepared.tau_p,
                prepared.particle_diameter,
                state.static.release_time_s,
            )
        )
    )


def _finish_debug_diagnostics(prepared: RunExecutionContext) -> None:
    state = prepared.state
    finalize_charge_model_diagnostics(
        state.collision_diagnostics,
        prepared.options.charge_model,
        state.charge,
    )
    state.collision_diagnostics["electric_q_over_m_particle_stats"] = (
        _finite_q_over_m_summary(state.charge, prepared.particle_mass)
    )
    if state.debug_buffers is not None:
        state.collision_diagnostics["output_buffers"] = dict(
            state.debug_buffers.summary()
        )


def _timing_summary(
    prepared: RunExecutionContext,
    loop_result: StepLoopResult,
    assembly_s: float,
) -> dict[str, float]:
    state = prepared.state
    timing_s = {
        "setup_s": float(prepared.loop_setup_done_s - prepared.setup_started_s),
        "step_loop_s": float(loop_result.elapsed_s),
        "solver_core_s": float(time.perf_counter() - prepared.setup_started_s),
    }
    if not bool(prepared.plan.output.is_debug):
        return timing_s
    timing_s["positions_assembly_s"] = float(assembly_s)
    timing_s["field_sampling_s"] = float(
        np.asarray(state.collision_diagnostics.get("field_sampling_s", 0.0)).item()
    )
    for key, value in sorted(state.timing_accumulator.items()):
        timing_s[str(key)] = float(value)
    return timing_s


def finalize_runtime_execution(
    prepared: RunExecutionContext,
    loop_result: StepLoopResult,
) -> SolverOutcome:
    """Assemble the one public solver result after numerical execution."""

    state = prepared.state
    _record_final_state_diagnostics(prepared, loop_result)
    _append_final_snapshot(prepared, float(loop_result.t))
    debug_outcome, assembly_s, positions_array_bytes = _debug_outcome(prepared)
    core_array_bytes = _core_array_bytes(prepared)
    compiled_field_array_bytes = _compiled_backend_array_bytes(prepared.compiled)
    if bool(prepared.plan.output.is_debug):
        _finish_debug_diagnostics(prepared)
    timing_s = _timing_summary(prepared, loop_result, assembly_s)
    memory_estimate_bytes = {
        "core_array_bytes": int(core_array_bytes),
        "compiled_field_array_bytes": int(compiled_field_array_bytes),
        "positions_array_bytes": int(positions_array_bytes),
        "estimated_numpy_bytes": int(
            core_array_bytes + compiled_field_array_bytes + positions_array_bytes
        ),
    }
    return SolverOutcome(
        final_position=state.x,
        final_velocity=state.v,
        final_charge=state.charge,
        released=state.released,
        active=state.active,
        stuck=state.stuck,
        frozen=state.frozen,
        absorbed=state.absorbed,
        contact_sliding=state.contact_sliding,
        contact_endpoint_stopped=state.contact_endpoint_stopped,
        contact_part_id=state.contact_part_id,
        contact_normal=state.contact_normal,
        escaped=state.escaped,
        invalid_mask_stopped=state.invalid_mask_stopped,
        numerical_boundary_stopped=state.numerical_boundary_stopped,
        invalid_stop_reason_code=state.invalid_stop_reason_code,
        final_step_name=_RUN_NAME,
        final_segment_name=_RUN_NAME,
        wall_summary_counts=state.wall_summary_counts,
        collision_diagnostics=state.collision_diagnostics,
        timing_s=timing_s,
        memory_estimate_bytes=memory_estimate_bytes,
        debug=debug_outcome,
    )


__all__ = (
    "append_snapshot",
    "finalize_runtime_execution",
    "initialize_debug_buffers",
)
