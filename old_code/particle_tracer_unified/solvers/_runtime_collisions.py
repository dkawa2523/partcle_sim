"""Collision classification and result commits for a runtime step."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from particle_tracer_unified.core.boundary_service import (
    contact_frame_on_boundary_edge_2d,
)
from particle_tracer_unified.core.coordinate_systems import (
    canonicalize_axisymmetric_rz_positions,
    canonicalize_axisymmetric_rz_state,
)
from particle_tracer_unified.domain import BoundaryHit

from ._coupled_charge_motion import CoupledChargeMotionBatch
from ._runtime_terminal_state import (
    commit_particle_state,
    mark_invalid_mask_stopped,
    mark_numerical_boundary_stopped,
)
from ._stochastic_randomness import WallRandomContext
from .collision_detection import TrialCollisionBatch, promote_stage_trace_collisions
from .high_fidelity_collision import CollidingParticleAdvanceResult
from .runtime_execution import RunExecutionContext
from .runtime_state import SolverState
from .stochastic_motion import PiecewiseLangevinPath
from .terminal_outcome import TerminalSegmentOutcome, terminal_segment_outcome


def _apply_stochastic_wall_search(
    trial: TrialCollisionBatch,
    *,
    active: np.ndarray,
    stochastic_particle_indices: tuple[int, ...],
    stochastic_prefetched_hits: Mapping[int, BoundaryHit],
) -> TrialCollisionBatch:
    if not stochastic_particle_indices:
        return trial
    colliders = {int(index) for index in np.asarray(trial.colliders, dtype=np.int64)}
    safe = {int(index) for index in np.asarray(trial.safe, dtype=np.int64)}
    prefetched_hits = dict(trial.prefetched_hits)
    for index in (int(value) for value in stochastic_particle_indices):
        colliders.discard(index)
        safe.discard(index)
        prefetched_hits.pop(index, None)
        if not bool(active[index]):
            continue
        hit = stochastic_prefetched_hits.get(index)
        if hit is None:
            safe.add(index)
            continue
        colliders.add(index)
        prefetched_hits[index] = hit
    return TrialCollisionBatch(
        colliders=np.asarray(sorted(colliders), dtype=np.int64),
        safe=np.asarray(sorted(safe), dtype=np.int64),
        prefetched_hits=prefetched_hits,
    )


def classify_step_collisions(
    execution: RunExecutionContext,
    *,
    mobile_active: np.ndarray,
    deterministic_stage_points: Mapping[int, np.ndarray],
    stochastic_paths: Mapping[int, PiecewiseLangevinPath],
    stochastic_prefetched_hits: Mapping[int, BoundaryHit],
    classify_trial,
    timer_start,
    record_timing,
) -> TrialCollisionBatch:
    runtime = execution.context
    state = execution.state
    plan = execution.plan
    boundary_service = execution.boundary_service
    detailed_timing = state.timing_accumulator if bool(plan.output.is_debug) else None
    started_at = timer_start(detailed_timing)
    trial = classify_trial(
        runtime,
        spatial_dim=int(execution.spatial_dim),
        n_particles=int(execution.n_particles),
        active=mobile_active,
        x=state.x,
        x_trial=state.x_trial,
        x_mid_trial=state.x_mid_trial,
        valid_mask_status_flags=state.valid_mask_status_flags,
        boundary_service=boundary_service,
        on_boundary_tol_m=float(plan.boundary.classification_tolerance_m),
        collision_diagnostics=state.collision_diagnostics,
        timing_accumulator=detailed_timing,
        boundary_broad_phase_enabled=bool(plan.boundary_broad_phase_enabled),
        boundary_broad_phase_debug_check=bool(plan.output.is_debug),
    )
    deterministic_traces = {
        index: trace
        for index, trace in deterministic_stage_points.items()
        if int(index) not in stochastic_paths
    }
    trial = promote_stage_trace_collisions(
        trial,
        active=mobile_active,
        x_start=state.x,
        stage_traces=deterministic_traces,
        boundary_service=boundary_service,
    )
    record_timing(detailed_timing, "collision_classify_s", started_at)
    return _apply_stochastic_wall_search(
        trial,
        active=mobile_active,
        stochastic_particle_indices=tuple(int(index) for index in stochastic_paths),
        stochastic_prefetched_hits=stochastic_prefetched_hits,
    )


def _collider_stage_points(
    state: SolverState,
    particle_index: int,
    deterministic_stage_points: Mapping[int, np.ndarray],
    stochastic_stage_points: Mapping[int, np.ndarray],
) -> np.ndarray:
    points = np.stack(
        (
            state.x_mid_trial[particle_index],
            state.x_trial[particle_index],
        ),
        axis=0,
    )
    if particle_index in deterministic_stage_points:
        points = np.asarray(
            deterministic_stage_points[particle_index],
            dtype=np.float64,
        )
    if particle_index in stochastic_stage_points:
        points = np.asarray(
            stochastic_stage_points[particle_index],
            dtype=np.float64,
        )
    return points


def _record_contact_state(
    execution: RunExecutionContext,
    particle_index: int,
    result: CollidingParticleAdvanceResult,
) -> None:
    if not bool(result.contact_sliding):
        return
    state = execution.state
    state.contact_sliding[particle_index] = True
    state.contact_endpoint_stopped[particle_index] = False
    if int(execution.spatial_dim) == 2:
        frame = contact_frame_on_boundary_edge_2d(
            execution.context,
            result.position,
            part_id_hint=int(result.contact_part_id),
            normal_hint=result.contact_normal,
        )
        state.contact_edge_index[particle_index] = (
            -1 if frame is None else int(frame.edge_index)
        )
    else:
        state.contact_edge_index[particle_index] = int(result.contact_primitive_id)
    state.contact_part_id[particle_index] = int(result.contact_part_id)
    if result.contact_normal is not None:
        state.contact_normal[particle_index] = np.asarray(
            result.contact_normal,
            dtype=np.float64,
        )


def _commit_collision_result(
    execution: RunExecutionContext,
    *,
    particle_index: int,
    result: CollidingParticleAdvanceResult,
    dt_step: float,
    terminal_outcomes: dict[int, TerminalSegmentOutcome],
) -> int:
    state = execution.state
    if result.charge_C is not None:
        state.charge[particle_index] = float(result.charge_C)
    position = np.asarray(result.position, dtype=np.float64)
    velocity = np.asarray(result.velocity, dtype=np.float64)
    if str(execution.context.coordinate_system) == "axisymmetric_rz":
        position, velocity = canonicalize_axisymmetric_rz_state(position, velocity)
    state.valid_mask_status_flags[particle_index] = np.uint8(result.valid_mask_status)
    if result.terminal_outcome is not None:
        outcome = result.terminal_outcome
        outcome_position = np.asarray(outcome.position, dtype=np.float64)
        if str(execution.context.coordinate_system) == "axisymmetric_rz":
            outcome_position = canonicalize_axisymmetric_rz_positions(outcome_position)
        terminal_outcomes[particle_index] = terminal_segment_outcome(
            accepted_elapsed_s=float(outcome.accepted_elapsed_s),
            segment_duration_s=float(dt_step),
            position=outcome_position,
            reason=str(outcome.reason),
        )
    if bool(result.invalid_mask_stopped):
        mark_invalid_mask_stopped(
            state=state,
            particle_index=particle_index,
            position=position,
            velocity=velocity,
            update_trial_buffers=False,
            reason=str(result.invalid_stop_reason),
        )
        return 1
    if bool(result.numerical_boundary_stopped):
        mark_numerical_boundary_stopped(
            state=state,
            particle_index=particle_index,
            position=position,
            velocity=velocity,
            update_trial_buffers=False,
            reason=str(result.numerical_boundary_stop_reason),
        )
        return 0
    _record_contact_state(execution, particle_index, result)
    was_active = bool(state.active[particle_index])
    commit_particle_state(
        state.x,
        state.v,
        state.active,
        state.escaped,
        particle_index=particle_index,
        position=position,
        velocity=velocity,
        mins=execution.mins,
        maxs=execution.maxs,
        boundary_tolerance_m=float(execution.plan.boundary.classification_tolerance_m),
    )
    if (
        was_active
        and not bool(state.active[particle_index])
        and bool(state.escaped[particle_index])
    ):
        terminal_outcomes[particle_index] = terminal_segment_outcome(
            accepted_elapsed_s=float(dt_step),
            segment_duration_s=float(dt_step),
            position=state.x[particle_index],
            reason="bounding_box_escape",
        )
    return 0


def resolve_step_colliders(
    execution: RunExecutionContext,
    *,
    trial: TrialCollisionBatch,
    deterministic_stage_points: Mapping[int, np.ndarray],
    stochastic_stage_points: Mapping[int, np.ndarray],
    stochastic_paths: Mapping[int, PiecewiseLangevinPath],
    electric_q_over_m_particle: np.ndarray | None,
    terminal_outcomes: dict[int, TerminalSegmentOutcome],
    t_next: float,
    dt_step: float,
    adaptive_substep_enabled: int,
    advance_particle,
    timer_start,
    record_timing,
    coupled_motion_batch: CoupledChargeMotionBatch | None = None,
) -> int:
    state = execution.state
    plan = execution.plan
    boundary_service = execution.boundary_service
    detailed_timing = state.timing_accumulator if bool(plan.output.is_debug) else None
    started_at = timer_start(detailed_timing)
    phys = execution.physics
    invalid_mask_stopped = 0
    for particle_index_raw in trial.colliders:
        particle_index = int(particle_index_raw)
        coupled_trace = (
            None
            if coupled_motion_batch is None
            else coupled_motion_batch.traces.get(particle_index)
        )
        result = advance_particle(
            runtime=execution.context,
            particles=None,
            particle_index=particle_index,
            rng=None,
            wall_random_context=WallRandomContext(
                seed=int(plan.rng_seed),
                particle_id=int(execution.particle_id[particle_index]),
                macro_step_index=int(state.step_index),
                cohort_index=int(state.wall_cohort_index[particle_index]),
                wall_event_ordinal=0,
            ),
            t=float(t_next),
            x_start=state.x[particle_index],
            v_start=state.v[particle_index],
            dt_step=float(dt_step),
            spatial_dim=int(execution.spatial_dim),
            compiled=execution.compiled,
            base_adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_max_splits=int(plan.adaptive_substep_max_splits),
            tau_p_i=float(execution.tau_p[particle_index]),
            particle_diameter_i=float(execution.particle_diameter[particle_index]),
            particle_density_i=float(execution.particle_density[particle_index]),
            particle_mass_i=float(execution.particle_mass[particle_index]),
            particle_id_i=int(execution.particle_id[particle_index]),
            dep_particle_rel_permittivity_i=float(
                execution.dep_particle_rel_permittivity[particle_index]
            ),
            thermophoretic_coeff_i=float(
                execution.thermophoretic_coeff[particle_index]
            ),
            body_accel=execution.body_acceleration_mps2,
            gas_density_kgm3=float(phys["gas_density_kgm3"]),
            gas_mu_pas=float(phys["gas_mu_pas"]),
            gas_temperature_K=float(phys["gas_temperature_K"]),
            gas_molecular_mass_kg=float(phys["gas_molecular_mass_kg"]),
            drag_model_mode=int(plan.drag_model_mode),
            electric_q_over_m_i=(
                None
                if electric_q_over_m_particle is None
                else float(electric_q_over_m_particle[particle_index])
            ),
            coupled_charge_tracer=(
                None
                if coupled_motion_batch is None
                else coupled_motion_batch.tracers[particle_index]
            ),
            charge_start_C=(
                None
                if coupled_motion_batch is None
                else float(coupled_motion_batch.start_charge_C[particle_index])
            ),
            initial_accepted_trace=coupled_trace,
            force_runtime=execution.options.force_runtime,
            stochastic_path=stochastic_paths.get(particle_index),
            initial_x_next=state.x_trial[particle_index],
            initial_v_next=state.v_trial[particle_index],
            initial_stage_points=_collider_stage_points(
                state,
                particle_index,
                deterministic_stage_points,
                stochastic_stage_points,
            ),
            initial_valid_mask_status=int(
                state.valid_mask_status_flags[particle_index]
            ),
            initial_primary_hit=trial.prefetched_hits.get(particle_index),
            initial_primary_hit_counted=False,
            initial_substep_count=int(state.substep_counts[particle_index]),
            inside_fn=boundary_service.inside,
            strict_inside_fn=boundary_service.inside_strict,
            primary_hit_fn=boundary_service.polyline_hit,
            nearest_projection_fn=boundary_service.nearest_projection,
            primary_hit_counter_key=boundary_service.primary_hit_counter_key,
            collision_diagnostics=state.collision_diagnostics,
            max_hit_rows=(
                None
                if state.debug_buffers is None
                else state.debug_buffers.max_hit_events
            ),
            wall_rows=(
                None if state.debug_buffers is None else state.debug_buffers.wall_events
            ),
            wall_summary_counts=state.wall_summary_counts,
            stuck=state.stuck,
            frozen=state.frozen,
            absorbed=state.absorbed,
            escaped=state.escaped,
            active=state.active,
            max_wall_hits_per_step=int(plan.max_wall_hits_per_step),
            contact_sliding_enabled=bool(plan.contact_sliding_enabled),
            epsilon_offset_m=float(plan.boundary.contact_offset_m),
            on_boundary_tol_m=float(plan.boundary.classification_tolerance_m),
            triangle_surface_3d=boundary_service.triangle_surface_3d,
        )
        state.wall_cohort_index[particle_index] += 1
        invalid_mask_stopped += _commit_collision_result(
            execution,
            particle_index=particle_index,
            result=result,
            dt_step=dt_step,
            terminal_outcomes=terminal_outcomes,
        )
    record_timing(detailed_timing, "collider_resolution_s", started_at)
    return invalid_mask_stopped
