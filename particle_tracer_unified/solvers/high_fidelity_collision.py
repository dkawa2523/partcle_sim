from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from ..core.boundary_service import (
    BoundaryHit,
    BoundaryService,
    inside_geometry as _boundary_inside_geometry,
    inside_geometry_with_boundary as _boundary_inside_geometry_with_boundary,
    points_inside_geometry_2d as _boundary_points_inside_geometry_2d,
)
from ..core.catalogs import resolve_step_wall_model
from ..core.datamodel import ProcessStepRow, WallPartModel
from ..core.field_sampling import valid_mask_status_requires_stop
from ..core.geometry3d import TriangleSurface3D
from .compiled_field_backend import CompiledRuntimeBackendLike
from .high_fidelity_freeflight import (
    _stage_points_from_trial,
    _stage_sample_times,
    advance_freeflight_partial,
    advance_freeflight_segment,
    resolve_valid_mask_prefix,
)
from .integrator_common import INTEGRATOR_ETD2
from .valid_mask_retry import resolve_valid_mask_retry_then_stop


def _step_segment_name(step: ProcessStepRow) -> str:
    return step.output_segment_name.strip() or step.step_name


def _orthonormal_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    dim = n.size
    if dim == 2:
        t = np.array([-n[1], n[0]], dtype=np.float64)
        mag = np.linalg.norm(t)
        return t / max(mag, 1e-30), np.zeros_like(t)
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = a - np.dot(a, n) * n
    t1 /= max(np.linalg.norm(t1), 1e-30)
    t2 = np.cross(n, t1)
    t2 /= max(np.linalg.norm(t2), 1e-30)
    return t1, t2


def _sample_diffuse_reflection(rng: np.random.Generator, normal: np.ndarray, speed: float) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    dim = n.size
    if dim == 2:
        t = np.array([-n[1], n[0]], dtype=np.float64)
        t /= max(np.linalg.norm(t), 1e-30)
        theta = rng.uniform(-0.5 * math.pi, 0.5 * math.pi)
        d = -math.cos(theta) * n + math.sin(theta) * t
        return speed * d / max(np.linalg.norm(d), 1e-30)
    t1, t2 = _orthonormal_tangent_basis(n)
    u = rng.uniform(0.0, 1.0)
    v = rng.uniform(0.0, 1.0)
    cos_theta = math.sqrt(1.0 - u)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * math.pi * v
    d = -cos_theta * n + sin_theta * (math.cos(phi) * t1 + math.sin(phi) * t2)
    return speed * d / max(np.linalg.norm(d), 1e-30)


def _effective_wall_stick_probability(particle_p_stick: float, wall_model: WallPartModel) -> float:
    p_particle = float(np.clip(particle_p_stick, 0.0, 1.0))
    if p_particle > 0.0:
        return p_particle
    return float(np.clip(wall_model.stick_probability, 0.0, 1.0))


def _wall_interaction(
    rng: np.random.Generator,
    v: np.ndarray,
    normal: np.ndarray,
    particle_stick_probability: float,
    wall_model: WallPartModel,
) -> Tuple[str, np.ndarray]:
    mode = str(wall_model.law_name).strip().lower()
    restitution = max(0.0, float(wall_model.restitution))
    diffuse_fraction = float(np.clip(wall_model.diffuse_fraction * (1.0 - wall_model.reflectivity), 0.0, 1.0))
    p_stick = _effective_wall_stick_probability(particle_stick_probability, wall_model)
    n = np.asarray(normal, dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-30)
    speed = float(np.linalg.norm(v))
    vn_signed = float(np.dot(v, n))
    vn_mag = abs(vn_signed)
    if mode in {'stick', 'sticking'}:
        return 'stuck', np.zeros_like(v)
    if mode in {'absorb', 'disappear'}:
        return 'absorbed', np.zeros_like(v)
    if mode in {'critical_sticking_velocity'} and vn_mag <= max(0.0, float(wall_model.critical_sticking_velocity_mps)):
        return 'stuck', np.zeros_like(v)
    if rng.random() < p_stick:
        return 'stuck', np.zeros_like(v)
    if mode in {'diffuse'}:
        return 'reflected_diffuse', _sample_diffuse_reflection(rng, n, restitution * speed)
    if mode in {'mixed_specular_diffuse'} and rng.random() < diffuse_fraction:
        return 'reflected_diffuse', _sample_diffuse_reflection(rng, n, restitution * speed)
    vt = v - vn_signed * n
    v_ref = vt - restitution * vn_signed * n
    return 'reflected_specular', v_ref


def _append_max_hit_event(
    *,
    max_hit_rows: List[Dict[str, object]],
    t: float,
    particle_id: int,
    step: ProcessStepRow,
    hit_count: int,
    remaining_dt: float,
    hit_part_ids: List[int],
    hit_outcomes: List[str],
) -> None:
    max_hit_rows.append(
        {
            'time_s': float(t),
            'particle_id': int(particle_id),
            'step_name': step.step_name,
            'segment_name': _step_segment_name(step),
            'hits_in_step': int(hit_count),
            'remaining_dt_s': float(remaining_dt),
            'last_part_id': int(hit_part_ids[-1]) if hit_part_ids else 0,
            'part_id_sequence': '|'.join(str(int(pid)) for pid in hit_part_ids),
            'outcome_sequence': '|'.join(hit_outcomes),
        }
    )


def _apply_wall_hit_step(
    *,
    runtime,
    step: ProcessStepRow,
    particles,
    particle_index: int,
    rng: np.random.Generator,
    hit: np.ndarray,
    n_out: np.ndarray,
    hit_dt: float,
    part_id: int,
    v_hit: np.ndarray,
    remaining_dt: float,
    segment_dt: float,
    hit_count: int,
    total_hit_count: int,
    hit_part_ids: List[int],
    hit_outcomes: List[str],
    retry_splits_used: int,
    collision_diagnostics: Dict[str, object],
    max_hit_rows: List[Dict[str, object]],
    wall_rows: List[Dict[str, object]],
    wall_law_counts: Dict[str, int],
    wall_summary_counts: Dict[Tuple[int, str, str], int],
    stuck: np.ndarray,
    absorbed: np.ndarray,
    active: np.ndarray,
    max_wall_hits_per_step: int,
    max_hits_retry_splits: int,
    min_remaining_dt: float,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
    t: float,
    triangle_surface_3d: Optional[TriangleSurface3D],
) -> Tuple[np.ndarray, np.ndarray, float, int, int, int, bool]:
    hit_arr = np.asarray(hit, dtype=np.float64)
    n_wall = np.asarray(n_out, dtype=np.float64)
    n_wall_mag = float(np.linalg.norm(n_wall))
    if n_wall_mag > 1e-30:
        n_wall = n_wall / n_wall_mag
    push = max(float(epsilon_offset_m), 1.0e-12)

    def _candidate_inside(candidate: np.ndarray, tol: float) -> bool:
        return bool(
            _boundary_inside_geometry(
                runtime,
                np.asarray(candidate, dtype=np.float64),
                on_boundary_tol_m=float(tol),
                triangle_surface_3d=triangle_surface_3d,
            )
        )

    x_wall = hit_arr.copy()
    if n_wall_mag > 1e-30:
        x_minus = hit_arr - push * n_wall
        x_plus = hit_arr + push * n_wall
        if _candidate_inside(x_minus, 0.0) or _candidate_inside(x_minus, float(on_boundary_tol_m)):
            x_wall = x_minus
        elif _candidate_inside(x_plus, 0.0) or _candidate_inside(x_plus, float(on_boundary_tol_m)):
            n_wall = -n_wall
            x_wall = x_plus
        elif _candidate_inside(hit_arr, float(on_boundary_tol_m)):
            x_wall = hit_arr.copy()
        else:
            x_wall = x_minus

    wall_model = resolve_step_wall_model(runtime.wall_catalog, part_id, step)
    wall_law_counts[wall_model.law_name] = wall_law_counts.get(wall_model.law_name, 0) + 1
    outcome, v_ref = _wall_interaction(rng, v_hit, n_wall, float(particles.stick_probability[particle_index]), wall_model)
    summary_key = (int(part_id), str(outcome), str(wall_model.law_name))
    wall_summary_counts[summary_key] = wall_summary_counts.get(summary_key, 0) + 1

    segment_dt_pos = max(0.0, float(segment_dt))
    hit_dt_clamped = float(np.clip(hit_dt, 0.0, segment_dt_pos))
    min_progress_dt = 0.0
    if segment_dt_pos > 0.0:
        min_progress_dt = min(segment_dt_pos, max(1.0e-12, 1.0e-8 * segment_dt_pos))
    consumed_dt = hit_dt_clamped if hit_dt_clamped > min_progress_dt else min_progress_dt
    consumed_dt = min(consumed_dt, segment_dt_pos)
    alpha_eff = 0.0 if segment_dt_pos <= 1.0e-30 else float(np.clip(consumed_dt / segment_dt_pos, 0.0, 1.0))

    if int(step.output_write_wall_events) != 0:
        wall_rows.append(
            {
                'time_s': float(t),
                'particle_id': int(particles.particle_id[particle_index]),
                'part_id': int(part_id),
                'step_name': step.step_name,
                'segment_name': _step_segment_name(step),
                'outcome': outcome,
                'wall_mode': wall_model.law_name,
                'alpha_hit': float(alpha_eff),
                'material_id': int(wall_model.material_id),
                'material_name': wall_model.material_name,
            }
        )

    hit_count += 1
    total_hit_count += 1
    hit_part_ids.append(int(part_id))
    hit_outcomes.append(str(outcome))

    remaining_dt = max(0.0, float(remaining_dt) - consumed_dt)

    if outcome == 'stuck':
        stuck[particle_index] = True
        active[particle_index] = False
        v_zero = np.zeros_like(v_hit)
        return x_wall, v_zero, remaining_dt, hit_count, total_hit_count, retry_splits_used, True
    if outcome == 'absorbed':
        absorbed[particle_index] = True
        active[particle_index] = False
        v_zero = np.zeros_like(v_hit)
        return x_wall, v_zero, remaining_dt, hit_count, total_hit_count, retry_splits_used, True

    x_curr_next = x_wall
    v_curr_next = np.asarray(v_ref, dtype=np.float64)

    if hit_count >= int(max_wall_hits_per_step):
        if remaining_dt > float(min_remaining_dt):
            if int(retry_splits_used) < int(max_hits_retry_splits):
                remaining_dt *= 0.5
                retry_splits_used += 1
                collision_diagnostics['max_hits_retry_count'] += 1
                hit_count = 0
                hit_part_ids.clear()
                hit_outcomes.clear()
                return x_curr_next, v_curr_next, remaining_dt, hit_count, total_hit_count, retry_splits_used, False
            collision_diagnostics['max_hits_reached_count'] += 1
            if int(max_hits_retry_splits) > 0:
                collision_diagnostics['max_hits_retry_exhausted_count'] += 1
                collision_diagnostics['dropped_remaining_dt_total_s'] += float(remaining_dt)
            _append_max_hit_event(
                max_hit_rows=max_hit_rows,
                t=float(t),
                particle_id=int(particles.particle_id[particle_index]),
                step=step,
                hit_count=int(hit_count),
                remaining_dt=float(remaining_dt),
                hit_part_ids=hit_part_ids,
                hit_outcomes=hit_outcomes,
            )
        return x_curr_next, v_curr_next, remaining_dt, hit_count, total_hit_count, retry_splits_used, True

    return x_curr_next, v_curr_next, remaining_dt, hit_count, total_hit_count, retry_splits_used, False


def _physical_hit_search_times(
    segment_dt: float,
    stage_times: np.ndarray,
    primary_hit_time: Optional[float],
) -> np.ndarray:
    dt_seg = max(float(segment_dt), 0.0)
    candidates: List[float] = []
    for value in np.asarray(stage_times, dtype=np.float64):
        candidates.append(float(value))
    if primary_hit_time is not None and np.isfinite(primary_hit_time):
        t_hit = float(np.clip(primary_hit_time, 0.0, dt_seg))
        if t_hit > 0.0:
            candidates.extend(
                [
                    t_hit,
                    0.5 * t_hit,
                    0.5 * (t_hit + dt_seg),
                    max(0.0, t_hit - 0.25 * dt_seg),
                    min(dt_seg, t_hit + 0.25 * dt_seg),
                ]
            )
    for frac in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        candidates.append(float(frac * dt_seg))
    unique = sorted({float(np.clip(v, 0.0, dt_seg)) for v in candidates if v > 0.0})
    return np.asarray(unique, dtype=np.float64)


def locate_physical_hit_state(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    segment_dt: float,
    t_end_segment: float,
    stage_points: np.ndarray,
    primary_hit: Optional[BoundaryHit],
    strict_inside_fn: Callable[[np.ndarray], bool],
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]],
    inputs: CollisionIntegratorInputs,
    adaptive_substep_enabled: int,
    on_boundary_tol_m: float,
    max_iters: int = 32,
) -> Optional[Tuple[BoundaryHit, np.ndarray, float]]:
    dt_seg = max(float(segment_dt), 0.0)
    if dt_seg <= 0.0:
        return None

    stage_times = _stage_sample_times(dt_seg, stage_points)
    primary_hit_time = None
    if primary_hit is not None:
        primary_hit_time = float(np.clip(float(primary_hit.alpha_hint) * dt_seg, 0.0, dt_seg))
    search_times = _physical_hit_search_times(dt_seg, stage_times, primary_hit_time)

    x_lo = np.asarray(x0, dtype=np.float64).copy()
    v_lo = np.asarray(v0, dtype=np.float64).copy()
    t_lo = 0.0
    x_hi = None
    v_hi = None
    t_hi = None

    for t_candidate in search_times:
        if t_candidate <= t_lo + 1.0e-15:
            continue
        x_candidate, v_candidate = _advance_partial_with_inputs(
            inputs=inputs,
            x0=x0,
            v0=v0,
            dt_partial=float(t_candidate),
            segment_dt=float(dt_seg),
            t_end_segment=float(t_end_segment),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
        )
        if strict_inside_fn(x_candidate):
            x_lo = x_candidate
            v_lo = v_candidate
            t_lo = float(t_candidate)
        else:
            x_hi = x_candidate
            v_hi = v_candidate
            t_hi = float(t_candidate)
            break

    if x_hi is None or v_hi is None or t_hi is None:
        return None

    stop_time_tol = max(1.0e-12, 1.0e-6 * dt_seg)
    stop_pos_tol = max(float(on_boundary_tol_m), 0.0)
    for _ in range(int(max(1, max_iters))):
        if float(t_hi - t_lo) <= stop_time_tol:
            break
        if stop_pos_tol > 0.0 and float(np.linalg.norm(np.asarray(x_hi, dtype=np.float64) - np.asarray(x_lo, dtype=np.float64))) <= stop_pos_tol:
            break
        t_mid = 0.5 * (float(t_lo) + float(t_hi))
        x_mid, v_mid = _advance_partial_with_inputs(
            inputs=inputs,
            x0=x0,
            v0=v0,
            dt_partial=float(t_mid),
            segment_dt=float(dt_seg),
            t_end_segment=float(t_end_segment),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
        )
        if strict_inside_fn(x_mid):
            x_lo = x_mid
            v_lo = v_mid
            t_lo = float(t_mid)
        else:
            x_hi = x_mid
            v_hi = v_mid
            t_hi = float(t_mid)

    t_hit = 0.5 * (float(t_lo) + float(t_hi))
    x_hit_state, v_hit = _advance_partial_with_inputs(
        inputs=inputs,
        x0=x0,
        v0=v0,
        dt_partial=float(t_hit),
        segment_dt=float(dt_seg),
        t_end_segment=float(t_end_segment),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
    )
    nearest = nearest_projection_fn(np.asarray(x_hit_state, dtype=np.float64), np.asarray(x_lo, dtype=np.float64))
    if nearest is None:
        nearest = nearest_projection_fn(np.asarray(x_hi, dtype=np.float64), np.asarray(x_lo, dtype=np.float64))
    if nearest is None:
        return None
    hit_alpha = 0.0 if dt_seg <= 1.0e-30 else float(np.clip(t_hit / dt_seg, 0.0, 1.0))
    return (
        BoundaryHit(
            position=np.asarray(nearest.position, dtype=np.float64),
            normal=np.asarray(nearest.normal, dtype=np.float64),
            part_id=int(nearest.part_id),
            alpha_hint=float(hit_alpha),
        ),
        np.asarray(v_hit, dtype=np.float64),
        float(t_hit),
    )


class CollidingParticleAdvanceResult(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    total_hits: int
    valid_mask_status: int
    extension_band_sampled: bool
    invalid_mask_stopped: bool


@dataclass(frozen=True)
class CollisionSegmentTrial:
    segment_adaptive_enabled: int
    x_next: np.ndarray
    v_next: np.ndarray
    stage_points: np.ndarray
    primary_hit: Optional[BoundaryHit]
    primary_hit_counted: bool
    particle_valid_mask_status: int
    particle_extension_band_sampled: bool
    invalid_stop_result: Optional[CollidingParticleAdvanceResult] = None


@dataclass(frozen=True)
class CollisionSegmentResolution:
    advance_without_hit: bool
    should_break: bool
    x_next: np.ndarray
    v_next: np.ndarray
    hit_event: Optional[BoundaryHit] = None
    v_hit: Optional[np.ndarray] = None
    hit_dt: float = 0.0


@dataclass(frozen=True)
class CollisionIntegratorInputs:
    spatial_dim: int
    compiled: CompiledRuntimeBackendLike
    integrator_mode: int
    adaptive_substep_tau_ratio: float
    adaptive_substep_max_splits: int
    tau_p_i: float
    flow_scale_particle_i: float
    drag_scale_particle_i: float
    body_scale_particle_i: float
    global_flow_scale: float
    global_drag_tau_scale: float
    global_body_accel_scale: float
    body_accel: np.ndarray
    min_tau_p_s: float


def _advance_segment_with_inputs(
    *,
    inputs: CollisionIntegratorInputs,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_segment: float,
    t_end_segment: float,
    adaptive_substep_enabled: int,
):
    return advance_freeflight_segment(
        x0=x0,
        v0=v0,
        dt_segment=float(dt_segment),
        t_end_segment=float(t_end_segment),
        spatial_dim=int(inputs.spatial_dim),
        compiled=inputs.compiled,
        integrator_mode=int(inputs.integrator_mode),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(inputs.adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(inputs.adaptive_substep_max_splits),
        tau_p_i=float(inputs.tau_p_i),
        flow_scale_particle_i=float(inputs.flow_scale_particle_i),
        drag_scale_particle_i=float(inputs.drag_scale_particle_i),
        body_scale_particle_i=float(inputs.body_scale_particle_i),
        global_flow_scale=float(inputs.global_flow_scale),
        global_drag_tau_scale=float(inputs.global_drag_tau_scale),
        global_body_accel_scale=float(inputs.global_body_accel_scale),
        body_accel=np.asarray(inputs.body_accel, dtype=np.float64),
        min_tau_p_s=float(inputs.min_tau_p_s),
    )


def _advance_partial_with_inputs(
    *,
    inputs: CollisionIntegratorInputs,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_partial: float,
    segment_dt: float,
    t_end_segment: float,
    adaptive_substep_enabled: int,
):
    return advance_freeflight_partial(
        x0=x0,
        v0=v0,
        dt_partial=float(dt_partial),
        segment_dt=float(segment_dt),
        t_end_segment=float(t_end_segment),
        spatial_dim=int(inputs.spatial_dim),
        compiled=inputs.compiled,
        integrator_mode=int(inputs.integrator_mode),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(inputs.adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(inputs.adaptive_substep_max_splits),
        tau_p_i=float(inputs.tau_p_i),
        flow_scale_particle_i=float(inputs.flow_scale_particle_i),
        drag_scale_particle_i=float(inputs.drag_scale_particle_i),
        body_scale_particle_i=float(inputs.body_scale_particle_i),
        global_flow_scale=float(inputs.global_flow_scale),
        global_drag_tau_scale=float(inputs.global_drag_tau_scale),
        global_body_accel_scale=float(inputs.global_body_accel_scale),
        body_accel=np.asarray(inputs.body_accel, dtype=np.float64),
        min_tau_p_s=float(inputs.min_tau_p_s),
    )


def _resolve_valid_mask_retry_with_inputs(
    *,
    inputs: CollisionIntegratorInputs,
    collision_diagnostics: Dict[str, object],
    x0: np.ndarray,
    v0: np.ndarray,
    dt_segment: float,
    t_end_segment: float,
    adaptive_substep_enabled: int,
):
    return resolve_valid_mask_retry_then_stop(
        resolve_prefix=resolve_valid_mask_prefix,
        collision_diagnostics=collision_diagnostics,
        x0=x0,
        v0=v0,
        dt_segment=float(dt_segment),
        t_end_segment=float(t_end_segment),
        spatial_dim=int(inputs.spatial_dim),
        compiled=inputs.compiled,
        integrator_mode=int(inputs.integrator_mode),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(inputs.adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(inputs.adaptive_substep_max_splits),
        tau_p_i=float(inputs.tau_p_i),
        flow_scale_particle_i=float(inputs.flow_scale_particle_i),
        drag_scale_particle_i=float(inputs.drag_scale_particle_i),
        body_scale_particle_i=float(inputs.body_scale_particle_i),
        global_flow_scale=float(inputs.global_flow_scale),
        global_drag_tau_scale=float(inputs.global_drag_tau_scale),
        global_body_accel_scale=float(inputs.global_body_accel_scale),
        body_accel=np.asarray(inputs.body_accel, dtype=np.float64),
        min_tau_p_s=float(inputs.min_tau_p_s),
    )


def _prepare_collision_segment_trial(
    *,
    use_precomputed_trial: bool,
    x_curr: np.ndarray,
    v_curr: np.ndarray,
    t: float,
    segment_dt: float,
    inputs: CollisionIntegratorInputs,
    base_adaptive_substep_enabled: int,
    segment_adaptive_enabled_for_retry: Callable[[int], int],
    retry_splits_used: int,
    valid_mask_retry_then_stop_enabled: bool,
    initial_x_next: np.ndarray,
    initial_v_next: np.ndarray,
    initial_stage_points: np.ndarray,
    initial_valid_mask_status: int,
    initial_extension_band_sampled: bool,
    initial_primary_hit: Optional[BoundaryHit],
    initial_primary_hit_counted: bool,
    collision_diagnostics: Dict[str, object],
) -> CollisionSegmentTrial:
    particle_valid_mask_status = int(initial_valid_mask_status)
    particle_extension_band_sampled = bool(initial_extension_band_sampled)
    if bool(use_precomputed_trial):
        return CollisionSegmentTrial(
            segment_adaptive_enabled=int(base_adaptive_substep_enabled),
            x_next=np.asarray(initial_x_next, dtype=np.float64),
            v_next=np.asarray(initial_v_next, dtype=np.float64),
            stage_points=np.asarray(initial_stage_points, dtype=np.float64),
            primary_hit=initial_primary_hit,
            primary_hit_counted=bool(initial_primary_hit_counted),
            particle_valid_mask_status=int(particle_valid_mask_status),
            particle_extension_band_sampled=bool(particle_extension_band_sampled),
        )

    segment_start_x = np.asarray(x_curr, dtype=np.float64).copy()
    segment_start_v = np.asarray(v_curr, dtype=np.float64).copy()
    segment_adaptive_enabled = int(segment_adaptive_enabled_for_retry(int(retry_splits_used)))
    x_next, v_next, n_substeps, stage_points, segment_mask_status, segment_extension_band_sampled = _advance_segment_with_inputs(
        inputs=inputs,
        x0=x_curr,
        v0=v_curr,
        dt_segment=float(segment_dt),
        t_end_segment=float(t),
        adaptive_substep_enabled=int(segment_adaptive_enabled),
    )
    if int(segment_mask_status) > int(particle_valid_mask_status):
        particle_valid_mask_status = int(segment_mask_status)
    if bool(segment_extension_band_sampled):
        particle_extension_band_sampled = True
    collision_diagnostics['collision_reintegrated_segments_count'] += 1
    if int(segment_adaptive_enabled) != 0:
        collision_diagnostics['adaptive_substep_segments_count'] += int(n_substeps)
        if int(n_substeps) > 1:
            collision_diagnostics['adaptive_substep_trigger_count'] += 1
    if bool(valid_mask_retry_then_stop_enabled) and bool(valid_mask_status_requires_stop(int(segment_mask_status))):
        resolution = _resolve_valid_mask_retry_with_inputs(
            inputs=inputs,
            collision_diagnostics=collision_diagnostics,
            x0=segment_start_x,
            v0=segment_start_v,
            dt_segment=float(segment_dt),
            t_end_segment=float(t),
            adaptive_substep_enabled=int(segment_adaptive_enabled),
        )
        return CollisionSegmentTrial(
            segment_adaptive_enabled=int(segment_adaptive_enabled),
            x_next=np.asarray(x_next, dtype=np.float64),
            v_next=np.asarray(v_next, dtype=np.float64),
            stage_points=np.asarray(stage_points, dtype=np.float64),
            primary_hit=None,
            primary_hit_counted=False,
            particle_valid_mask_status=int(particle_valid_mask_status),
            particle_extension_band_sampled=bool(particle_extension_band_sampled),
            invalid_stop_result=CollidingParticleAdvanceResult(
                position=resolution.position,
                velocity=resolution.velocity,
                total_hits=0,
                valid_mask_status=int(particle_valid_mask_status),
                extension_band_sampled=bool(particle_extension_band_sampled),
                invalid_mask_stopped=True,
            ),
        )
    return CollisionSegmentTrial(
        segment_adaptive_enabled=int(segment_adaptive_enabled),
        x_next=np.asarray(x_next, dtype=np.float64),
        v_next=np.asarray(v_next, dtype=np.float64),
        stage_points=np.asarray(stage_points, dtype=np.float64),
        primary_hit=None,
        primary_hit_counted=False,
        particle_valid_mask_status=int(particle_valid_mask_status),
        particle_extension_band_sampled=bool(particle_extension_band_sampled),
    )


def _resolve_collision_segment(
    *,
    x_curr: np.ndarray,
    v_curr: np.ndarray,
    x_next: np.ndarray,
    v_next: np.ndarray,
    stage_points: np.ndarray,
    primary_hit: Optional[BoundaryHit],
    primary_hit_counted: bool,
    inside_fn: Callable[[np.ndarray], bool],
    strict_inside_fn: Callable[[np.ndarray], bool],
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]],
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]],
    primary_hit_counter_key: str,
    collision_diagnostics: Dict[str, object],
    t: float,
    segment_dt: float,
    inputs: CollisionIntegratorInputs,
    adaptive_substep_enabled: int,
    on_boundary_tol_m: float,
) -> CollisionSegmentResolution:
    stage_points_arr = np.asarray(stage_points, dtype=np.float64)
    use_polyline = int(stage_points_arr.shape[0]) >= 2
    stage_inside = np.asarray([bool(inside_fn(pt)) for pt in stage_points_arr], dtype=bool)

    resolved_primary_hit = primary_hit
    resolved_primary_hit_counted = bool(primary_hit_counted)
    if resolved_primary_hit is None:
        if use_polyline:
            collision_diagnostics['etd2_polyline_checks_count'] += 1
        resolved_primary_hit = primary_hit_fn(x_curr, stage_points_arr)
        if resolved_primary_hit is not None:
            if use_polyline:
                collision_diagnostics['etd2_polyline_hit_count'] += 1
            resolved_primary_hit_counted = False
        elif use_polyline:
            collision_diagnostics['etd2_polyline_fallback_count'] += 1
    elif use_polyline and not resolved_primary_hit_counted:
        collision_diagnostics['etd2_polyline_checks_count'] += 1
        collision_diagnostics['etd2_polyline_hit_count'] += 1

    if resolved_primary_hit is not None and not resolved_primary_hit_counted:
        collision_diagnostics[primary_hit_counter_key] += 1
        collision_diagnostics['primary_hit_count'] += 1
        resolved_primary_hit_counted = True

    if resolved_primary_hit is None and bool(np.all(stage_inside)):
        return CollisionSegmentResolution(
            advance_without_hit=True,
            should_break=False,
            x_next=np.asarray(x_next, dtype=np.float64),
            v_next=np.asarray(v_next, dtype=np.float64),
        )

    hit_state = locate_physical_hit_state(
        x0=x_curr,
        v0=v_curr,
        segment_dt=float(segment_dt),
        t_end_segment=float(t),
        stage_points=stage_points_arr,
        primary_hit=resolved_primary_hit,
        strict_inside_fn=strict_inside_fn,
        nearest_projection_fn=nearest_projection_fn,
        inputs=inputs,
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        on_boundary_tol_m=float(on_boundary_tol_m),
    )
    if hit_state is not None:
        hit_event, v_hit, hit_dt = hit_state
        if resolved_primary_hit is None:
            collision_diagnostics['bisection_fallback_count'] += 1
        return CollisionSegmentResolution(
            advance_without_hit=False,
            should_break=False,
            x_next=np.asarray(x_next, dtype=np.float64),
            v_next=np.asarray(v_next, dtype=np.float64),
            hit_event=hit_event,
            v_hit=np.asarray(v_hit, dtype=np.float64),
            hit_dt=float(hit_dt),
        )

    collision_diagnostics['unresolved_crossing_count'] += 1
    fallback_dt = float(segment_dt)
    if resolved_primary_hit is not None:
        fallback_dt = float(np.clip(float(resolved_primary_hit.alpha_hint), 0.0, 1.0) * float(segment_dt))
    x_fallback, v_fallback = _advance_partial_with_inputs(
        inputs=inputs,
        x0=x_curr,
        v0=v_curr,
        dt_partial=float(fallback_dt),
        segment_dt=float(segment_dt),
        t_end_segment=float(t),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
    )
    nearest_hit = nearest_projection_fn(np.asarray(x_fallback, dtype=np.float64), x_curr)
    if nearest_hit is None:
        nearest_hit = nearest_projection_fn(np.asarray(x_next, dtype=np.float64), x_curr)
    if nearest_hit is None:
        return CollisionSegmentResolution(
            advance_without_hit=False,
            should_break=True,
            x_next=np.asarray(x_next, dtype=np.float64),
            v_next=np.asarray(v_next, dtype=np.float64),
        )
    collision_diagnostics['nearest_projection_fallback_count'] += 1
    return CollisionSegmentResolution(
        advance_without_hit=False,
        should_break=False,
        x_next=np.asarray(x_next, dtype=np.float64),
        v_next=np.asarray(v_next, dtype=np.float64),
        hit_event=nearest_hit,
        v_hit=np.asarray(v_fallback, dtype=np.float64),
        hit_dt=float(fallback_dt),
    )


def _advance_colliding_particle(
    *,
    runtime,
    step: ProcessStepRow,
    particles,
    particle_index: int,
    rng: np.random.Generator,
    t: float,
    x_start: np.ndarray,
    v_start: np.ndarray,
    dt_step: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    integrator_mode: int,
    base_adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    min_remaining_dt_ratio: float,
    segment_adaptive_enabled_for_retry: Callable[[int], int],
    tau_p_i: float,
    flow_scale_particle_i: float,
    drag_scale_particle_i: float,
    body_scale_particle_i: float,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    global_body_accel_scale: float,
    body_accel: np.ndarray,
    min_tau_p_s: float,
    valid_mask_retry_then_stop_enabled: bool,
    initial_x_next: np.ndarray,
    initial_v_next: np.ndarray,
    initial_stage_points: np.ndarray,
    initial_valid_mask_status: int,
    initial_extension_band_sampled: bool,
    initial_primary_hit: Optional[BoundaryHit],
    initial_primary_hit_counted: bool,
    inside_fn: Callable[[np.ndarray], bool],
    strict_inside_fn: Callable[[np.ndarray], bool],
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]],
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]],
    primary_hit_counter_key: str,
    collision_diagnostics: Dict[str, object],
    max_hit_rows: List[Dict[str, object]],
    wall_rows: List[Dict[str, object]],
    wall_law_counts: Dict[str, int],
    wall_summary_counts: Dict[Tuple[int, str, str], int],
    stuck: np.ndarray,
    absorbed: np.ndarray,
    active: np.ndarray,
    max_wall_hits_per_step: int,
    max_hits_retry_splits: int,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
    triangle_surface_3d: Optional[TriangleSurface3D],
) -> CollidingParticleAdvanceResult:
    remaining_dt = float(dt_step)
    min_remaining_dt = float(dt_step * min_remaining_dt_ratio)
    x_curr = np.asarray(x_start, dtype=np.float64).copy()
    v_curr = np.asarray(v_start, dtype=np.float64).copy()
    hit_count = 0
    total_hit_count = 0
    retry_splits_used = 0
    hit_part_ids: List[int] = []
    hit_outcomes: List[str] = []
    use_precomputed_trial = True
    particle_valid_mask_status = int(initial_valid_mask_status)
    particle_extension_band_sampled = bool(initial_extension_band_sampled)
    integrator_inputs = CollisionIntegratorInputs(
        spatial_dim=int(spatial_dim),
        compiled=compiled,
        integrator_mode=int(integrator_mode),
        adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        tau_p_i=float(tau_p_i),
        flow_scale_particle_i=float(flow_scale_particle_i),
        drag_scale_particle_i=float(drag_scale_particle_i),
        body_scale_particle_i=float(body_scale_particle_i),
        global_flow_scale=float(global_flow_scale),
        global_drag_tau_scale=float(global_drag_tau_scale),
        global_body_accel_scale=float(global_body_accel_scale),
        body_accel=np.asarray(body_accel, dtype=np.float64),
        min_tau_p_s=float(min_tau_p_s),
    )

    while active[particle_index] and remaining_dt > min_remaining_dt:
        segment_dt = float(remaining_dt)
        segment_trial = _prepare_collision_segment_trial(
            use_precomputed_trial=bool(use_precomputed_trial),
            x_curr=x_curr,
            v_curr=v_curr,
            t=float(t),
            segment_dt=float(segment_dt),
            inputs=integrator_inputs,
            base_adaptive_substep_enabled=int(base_adaptive_substep_enabled),
            segment_adaptive_enabled_for_retry=segment_adaptive_enabled_for_retry,
            retry_splits_used=int(retry_splits_used),
            valid_mask_retry_then_stop_enabled=bool(valid_mask_retry_then_stop_enabled),
            initial_x_next=initial_x_next,
            initial_v_next=initial_v_next,
            initial_stage_points=initial_stage_points,
            initial_valid_mask_status=int(particle_valid_mask_status),
            initial_extension_band_sampled=bool(particle_extension_band_sampled),
            initial_primary_hit=initial_primary_hit,
            initial_primary_hit_counted=bool(initial_primary_hit_counted),
            collision_diagnostics=collision_diagnostics,
        )
        particle_valid_mask_status = int(segment_trial.particle_valid_mask_status)
        particle_extension_band_sampled = bool(segment_trial.particle_extension_band_sampled)
        if segment_trial.invalid_stop_result is not None:
            return CollidingParticleAdvanceResult(
                position=np.asarray(segment_trial.invalid_stop_result.position, dtype=np.float64),
                velocity=np.asarray(segment_trial.invalid_stop_result.velocity, dtype=np.float64),
                total_hits=int(total_hit_count),
                valid_mask_status=int(segment_trial.invalid_stop_result.valid_mask_status),
                extension_band_sampled=bool(segment_trial.invalid_stop_result.extension_band_sampled),
                invalid_mask_stopped=True,
            )

        segment_resolution = _resolve_collision_segment(
            x_curr=x_curr,
            v_curr=v_curr,
            x_next=segment_trial.x_next,
            v_next=segment_trial.v_next,
            stage_points=segment_trial.stage_points,
            primary_hit=segment_trial.primary_hit,
            primary_hit_counted=bool(segment_trial.primary_hit_counted),
            inside_fn=inside_fn,
            strict_inside_fn=strict_inside_fn,
            primary_hit_fn=primary_hit_fn,
            nearest_projection_fn=nearest_projection_fn,
            primary_hit_counter_key=primary_hit_counter_key,
            collision_diagnostics=collision_diagnostics,
            t=float(t),
            segment_dt=float(segment_dt),
            inputs=integrator_inputs,
            adaptive_substep_enabled=int(segment_trial.segment_adaptive_enabled),
            on_boundary_tol_m=float(on_boundary_tol_m),
        )
        if bool(segment_resolution.advance_without_hit):
            x_curr = np.asarray(segment_resolution.x_next, dtype=np.float64)
            v_curr = np.asarray(segment_resolution.v_next, dtype=np.float64)
            break
        if bool(segment_resolution.should_break) or segment_resolution.hit_event is None or segment_resolution.v_hit is None:
            break

        hit = np.asarray(segment_resolution.hit_event.position, dtype=np.float64)
        n_out = np.asarray(segment_resolution.hit_event.normal, dtype=np.float64)
        part_id = int(segment_resolution.hit_event.part_id)

        x_curr, v_curr, remaining_dt, hit_count, total_hit_count, retry_splits_used, should_break = _apply_wall_hit_step(
            runtime=runtime,
            step=step,
            particles=particles,
            particle_index=int(particle_index),
            rng=rng,
            hit=hit,
            n_out=n_out,
            hit_dt=float(segment_resolution.hit_dt),
            part_id=int(part_id),
            v_hit=np.asarray(segment_resolution.v_hit, dtype=np.float64),
            remaining_dt=float(remaining_dt),
            segment_dt=float(segment_dt),
            hit_count=int(hit_count),
            total_hit_count=int(total_hit_count),
            hit_part_ids=hit_part_ids,
            hit_outcomes=hit_outcomes,
            retry_splits_used=int(retry_splits_used),
            collision_diagnostics=collision_diagnostics,
            max_hit_rows=max_hit_rows,
            wall_rows=wall_rows,
            wall_law_counts=wall_law_counts,
            wall_summary_counts=wall_summary_counts,
            stuck=stuck,
            absorbed=absorbed,
            active=active,
            max_wall_hits_per_step=int(max_wall_hits_per_step),
            max_hits_retry_splits=int(max_hits_retry_splits),
            min_remaining_dt=float(min_remaining_dt),
            epsilon_offset_m=float(epsilon_offset_m),
            on_boundary_tol_m=float(on_boundary_tol_m),
            t=float(t),
            triangle_surface_3d=triangle_surface_3d,
        )
        if should_break:
            break
        use_precomputed_trial = False

    if total_hit_count > 1:
        collision_diagnostics['multi_hit_events_count'] += 1
    return CollidingParticleAdvanceResult(
        position=x_curr,
        velocity=v_curr,
        total_hits=int(total_hit_count),
        valid_mask_status=int(particle_valid_mask_status),
        extension_band_sampled=bool(particle_extension_band_sampled),
        invalid_mask_stopped=False,
    )


@dataclass(frozen=True)
class TrialCollisionBatch:
    colliders: np.ndarray
    safe: np.ndarray
    prefetched_hits: Dict[int, BoundaryHit]


def _classify_trial_collisions_2d(
    runtime,
    *,
    n_particles: int,
    active: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    integrator_mode: int,
    on_boundary_tol_m: float,
    collision_diagnostics: Dict[str, object],
) -> TrialCollisionBatch:
    active_idx = np.flatnonzero(active)
    loop_inside = np.zeros(n_particles, dtype=bool)
    loop_mid_inside = np.ones(n_particles, dtype=bool)
    is_etd2 = int(integrator_mode) == INTEGRATOR_ETD2
    if active_idx.size:
        inside_active, on_boundary_active = _boundary_points_inside_geometry_2d(
            runtime,
            x_trial[active_idx],
            on_boundary_tol_m=on_boundary_tol_m,
            return_on_boundary=True,
        )
        loop_inside[active_idx] = inside_active
        collision_diagnostics['on_boundary_promoted_inside_count'] += int(np.count_nonzero(on_boundary_active))
        if is_etd2:
            inside_mid_active, on_boundary_mid_active = _boundary_points_inside_geometry_2d(
                runtime,
                x_mid_trial[active_idx],
                on_boundary_tol_m=on_boundary_tol_m,
                return_on_boundary=True,
            )
            loop_mid_inside[active_idx] = inside_mid_active
            collision_diagnostics['on_boundary_promoted_inside_count'] += int(np.count_nonzero(on_boundary_mid_active))
            collision_diagnostics['etd2_midpoint_outside_count'] += int(np.count_nonzero(~inside_mid_active))
    if is_etd2:
        colliders = np.flatnonzero(active & ((~loop_inside) | (~loop_mid_inside)))
        safe = np.flatnonzero(active & loop_inside & loop_mid_inside)
    else:
        colliders = np.flatnonzero(active & (~loop_inside))
        safe = np.flatnonzero(active & loop_inside)
    return TrialCollisionBatch(
        colliders=np.asarray(colliders, dtype=np.int64),
        safe=np.asarray(safe, dtype=np.int64),
        prefetched_hits={},
    )


def _classify_trial_collisions_3d(
    runtime,
    *,
    active: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    integrator_mode: int,
    boundary_service: BoundaryService,
    on_boundary_tol_m: float,
    collision_diagnostics: Dict[str, object],
) -> TrialCollisionBatch:
    colliders_list: List[int] = []
    safe_list: List[int] = []
    prefetched_hits: Dict[int, BoundaryHit] = {}
    active_idx = np.flatnonzero(active)
    is_etd2 = int(integrator_mode) == INTEGRATOR_ETD2
    triangle_surface_3d = boundary_service.triangle_surface_3d
    for idx in active_idx:
        i = int(idx)
        if is_etd2:
            stage_points = _stage_points_from_trial(
                x_trial[i],
                integrator_mode=int(integrator_mode),
                x_mid=x_mid_trial[i],
            )
            hit3 = boundary_service.polyline_hit(x[i], stage_points)
            if hit3 is not None:
                prefetched_hits[i] = hit3
                colliders_list.append(i)
                continue
            inside_mid, on_boundary_mid = _boundary_inside_geometry_with_boundary(
                runtime,
                x_mid_trial[i],
                on_boundary_tol_m=on_boundary_tol_m,
                triangle_surface_3d=triangle_surface_3d,
            )
            inside_end, on_boundary_end = _boundary_inside_geometry_with_boundary(
                runtime,
                x_trial[i],
                on_boundary_tol_m=on_boundary_tol_m,
                triangle_surface_3d=triangle_surface_3d,
            )
            collision_diagnostics['on_boundary_promoted_inside_count'] += int(on_boundary_mid) + int(on_boundary_end)
            if not bool(inside_mid):
                collision_diagnostics['etd2_midpoint_outside_count'] += 1
            if not bool(inside_mid) or not bool(inside_end):
                colliders_list.append(i)
            else:
                safe_list.append(i)
            continue
        hit3 = boundary_service.segment_hit(x[i], x_trial[i])
        if hit3 is not None:
            prefetched_hits[i] = hit3
            colliders_list.append(i)
            continue
        if triangle_surface_3d is not None and (
            np.any(x_trial[i] < triangle_surface_3d.bbox_min - 1e-12)
            or np.any(x_trial[i] > triangle_surface_3d.bbox_max + 1e-12)
        ):
            colliders_list.append(i)
        else:
            safe_list.append(i)
    return TrialCollisionBatch(
        colliders=np.asarray(colliders_list, dtype=np.int64),
        safe=np.asarray(safe_list, dtype=np.int64),
        prefetched_hits=prefetched_hits,
    )


def _classify_trial_collisions(
    runtime,
    *,
    spatial_dim: int,
    n_particles: int,
    active: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    integrator_mode: int,
    boundary_service: BoundaryService,
    on_boundary_tol_m: float,
    collision_diagnostics: Dict[str, object],
) -> TrialCollisionBatch:
    if int(spatial_dim) == 2:
        return _classify_trial_collisions_2d(
            runtime,
            n_particles=int(n_particles),
            active=active,
            x_trial=x_trial,
            x_mid_trial=x_mid_trial,
            integrator_mode=int(integrator_mode),
            on_boundary_tol_m=float(on_boundary_tol_m),
            collision_diagnostics=collision_diagnostics,
        )
    return _classify_trial_collisions_3d(
        runtime,
        active=active,
        x=x,
        x_trial=x_trial,
        x_mid_trial=x_mid_trial,
        integrator_mode=int(integrator_mode),
        boundary_service=boundary_service,
        on_boundary_tol_m=float(on_boundary_tol_m),
        collision_diagnostics=collision_diagnostics,
    )


__all__ = (
    'TrialCollisionBatch',
    '_apply_wall_hit_step',
    '_classify_trial_collisions',
    '_step_segment_name',
    'locate_physical_hit_state',
)
