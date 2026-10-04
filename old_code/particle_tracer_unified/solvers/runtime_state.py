from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnostics import initial_collision_diagnostics
from .output_buffers import DebugBuffers
from .particle_state import (
    ParticleStaticArrays,
    initial_position_velocity,
    static_arrays_from_particles,
)
from .runtime_plan import ReleaseCursor, SolverPlan, build_release_schedule


@dataclass
class SolverState:
    """Sole owner of mutable particle state and step-loop scratch arrays.

    Runtime preparation owns immutable inputs and compiled providers; this type
    owns every array that may change while the solver advances.  Keeping that
    boundary explicit prevents setup/output code from growing secondary state
    representations.
    """

    static: ParticleStaticArrays
    x: np.ndarray
    v: np.ndarray
    released: np.ndarray
    active: np.ndarray
    stuck: np.ndarray
    frozen: np.ndarray
    absorbed: np.ndarray
    contact_sliding: np.ndarray
    contact_endpoint_stopped: np.ndarray
    contact_edge_index: np.ndarray
    contact_part_id: np.ndarray
    contact_normal: np.ndarray
    escaped: np.ndarray
    invalid_mask_stopped: np.ndarray
    numerical_boundary_stopped: np.ndarray
    invalid_stop_reason_code: np.ndarray
    wall_summary_counts: dict[tuple[int, str, str], int]
    collision_diagnostics: dict[str, object]
    step_index: int
    save_index: int
    x_trial: np.ndarray
    v_trial: np.ndarray
    x_mid_trial: np.ndarray
    substep_counts: np.ndarray
    valid_mask_status_flags: np.ndarray
    local_error_resolved: np.ndarray
    valid_mask_mixed_seen: np.ndarray
    valid_mask_hard_seen: np.ndarray
    charge: np.ndarray
    release_cursor: ReleaseCursor
    solver_plan: SolverPlan
    debug_buffers: DebugBuffers | None
    stochastic_cohort_index: np.ndarray
    wall_cohort_index: np.ndarray
    timing_accumulator: dict[str, float]


def initialize_solver_state(
    *,
    particles,
    plan: SolverPlan,
    debug_buffers: DebugBuffers | None,
    spatial_dim: int,
) -> SolverState:
    """Create the one mutable state object used for the whole simulation."""

    static = static_arrays_from_particles(particles)
    n_particles = int(static.count)
    x, v = initial_position_velocity(particles, plan)
    return SolverState(
        static=static,
        x=x,
        v=v,
        released=np.zeros(n_particles, dtype=bool),
        active=np.zeros(n_particles, dtype=bool),
        stuck=np.zeros(n_particles, dtype=bool),
        frozen=np.zeros(n_particles, dtype=bool),
        absorbed=np.zeros(n_particles, dtype=bool),
        contact_sliding=np.zeros(n_particles, dtype=bool),
        contact_endpoint_stopped=np.zeros(n_particles, dtype=bool),
        contact_edge_index=np.full(n_particles, -1, dtype=np.int32),
        contact_part_id=np.zeros(n_particles, dtype=np.int32),
        contact_normal=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        escaped=np.zeros(n_particles, dtype=bool),
        invalid_mask_stopped=np.zeros(n_particles, dtype=bool),
        numerical_boundary_stopped=np.zeros(n_particles, dtype=bool),
        invalid_stop_reason_code=np.zeros(n_particles, dtype=np.uint8),
        wall_summary_counts={},
        collision_diagnostics=initial_collision_diagnostics(
            debug=bool(plan.output.is_debug)
        ),
        timing_accumulator={},
        step_index=0,
        save_index=1,
        x_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        v_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        x_mid_trial=np.zeros((n_particles, int(spatial_dim)), dtype=np.float64),
        substep_counts=np.ones(n_particles, dtype=np.int32),
        valid_mask_status_flags=np.zeros(n_particles, dtype=np.uint8),
        local_error_resolved=np.ones(n_particles, dtype=bool),
        valid_mask_mixed_seen=np.zeros(n_particles, dtype=bool),
        valid_mask_hard_seen=np.zeros(n_particles, dtype=bool),
        charge=np.asarray(static.charge_initial_C, dtype=np.float64).copy(),
        release_cursor=ReleaseCursor(build_release_schedule(static.release_time_s)),
        solver_plan=plan,
        debug_buffers=debug_buffers,
        stochastic_cohort_index=np.zeros(n_particles, dtype=np.int64),
        wall_cohort_index=np.zeros(n_particles, dtype=np.int64),
    )


__all__ = (
    "SolverState",
    "initialize_solver_state",
)
