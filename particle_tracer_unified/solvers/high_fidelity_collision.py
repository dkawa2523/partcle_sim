from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from ..core.boundary_service import (
    BoundaryHit,
    BoundaryService,
    inside_geometry as _boundary_inside_geometry,
    inside_geometry_with_boundary as _boundary_inside_geometry_with_boundary,
    points_inside_geometry_2d as _boundary_points_inside_geometry_2d,
    polyline_hits_from_boundary_edges_batch,
)
from ..core.boundary_core import sample_geometry_sdf_points_2d
from ..core.catalogs import resolve_step_wall_model
from ..core.datamodel import ProcessStepRow, WallPartModel
from ..core.field_sampling import VALID_MASK_STATUS_CLEAN, valid_mask_status_requires_stop
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

_CONTACT_REFLECTED_OUTCOMES = frozenset({'reflected_specular', 'reflected_diffuse'})


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
    if mode in {'open', 'outflow', 'exhaust', 'escape', 'field_support_exit'}:
        return 'escaped', np.zeros_like(v)
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


def _wall_event_row(
    *,
    t_step_end: float,
    segment_dt: float,
    hit_dt: float,
    particle_id: int,
    particle_mass_kg: float,
    particle_diameter_m: float,
    hit: np.ndarray,
    normal: np.ndarray,
    v_hit: np.ndarray,
    part_id: int,
    step: ProcessStepRow,
    outcome: str,
    wall_model: WallPartModel,
    alpha_hit: float,
    primitive_id: int = -1,
    primitive_kind: str = 'unknown',
    is_ambiguous: bool = False,
) -> Dict[str, object]:
    hit_arr = np.asarray(hit, dtype=np.float64)
    normal_arr = np.asarray(normal, dtype=np.float64)
    velocity_arr = np.asarray(v_hit, dtype=np.float64)
    speed = float(np.linalg.norm(velocity_arr))
    normal_speed = 0.0
    tangential_speed = 0.0
    incidence_angle_deg = 0.0
    if normal_arr.size == velocity_arr.size and float(np.linalg.norm(normal_arr)) > 1.0e-30:
        n_unit = normal_arr / max(float(np.linalg.norm(normal_arr)), 1.0e-30)
        vn_signed = float(np.dot(velocity_arr, n_unit))
        normal_speed = abs(vn_signed)
        tangential = velocity_arr - vn_signed * n_unit
        tangential_speed = float(np.linalg.norm(tangential))
        incidence_angle_deg = math.degrees(math.atan2(tangential_speed, max(normal_speed, 1.0e-30)))
    hit_time_s = float(t_step_end) - max(0.0, float(segment_dt)) + float(np.clip(hit_dt, 0.0, max(0.0, float(segment_dt))))
    row: Dict[str, object] = {
        'time_s': float(t_step_end),
        'hit_time_s': float(hit_time_s),
        'particle_id': int(particle_id),
        'part_id': int(part_id),
        'boundary_primitive_id': int(primitive_id),
        'boundary_primitive_kind': str(primitive_kind),
        'boundary_hit_ambiguous': int(bool(is_ambiguous)),
        'step_name': step.step_name,
        'segment_name': _step_segment_name(step),
        'outcome': outcome,
        'wall_mode': wall_model.law_name,
        'alpha_hit': float(alpha_hit),
        'material_id': int(wall_model.material_id),
        'material_name': wall_model.material_name,
        'particle_mass_kg': float(particle_mass_kg),
        'particle_diameter_m': float(particle_diameter_m),
        'impact_speed_mps': float(speed),
        'impact_normal_speed_mps': float(normal_speed),
        'impact_tangential_speed_mps': float(tangential_speed),
        'impact_angle_deg_from_normal': float(incidence_angle_deg),
    }
    for axis_idx, axis_name in enumerate(('x', 'y', 'z')):
        row[f'hit_{axis_name}_m'] = float(hit_arr[axis_idx]) if axis_idx < hit_arr.size else float('nan')
        row[f'normal_{axis_name}'] = float(normal_arr[axis_idx]) if axis_idx < normal_arr.size else float('nan')
        row[f'v_hit_{axis_name}_mps'] = float(velocity_arr[axis_idx]) if axis_idx < velocity_arr.size else float('nan')
    return row


def _particle_scalar_or_nan(particles, name: str, particle_index: int) -> float:
    values = getattr(particles, name, None)
    if values is None:
        return float('nan')
    arr = np.asarray(values, dtype=np.float64)
    if int(particle_index) >= arr.size:
        return float('nan')
    return float(arr[int(particle_index)])


def _increment_collision_diagnostic(collision_diagnostics: Dict[str, object], key: str, value: int = 1) -> None:
    collision_diagnostics[key] = int(collision_diagnostics.get(key, 0)) + int(value)


class WallHitStepResult(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    remaining_dt: float
    hit_count: int
    total_hit_count: int
    should_break: bool
    entered_contact: bool = False
    contact_part_id: int = 0
    contact_normal: Optional[np.ndarray] = None
    contact_primitive_id: int = -1


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
    primitive_id: int = -1,
    primitive_kind: str = 'unknown',
    is_ambiguous: bool = False,
    v_hit: np.ndarray,
    remaining_dt: float,
    segment_dt: float,
    hit_count: int,
    total_hit_count: int,
    hit_part_ids: List[int],
    hit_outcomes: List[str],
    collision_diagnostics: Dict[str, object],
    max_hit_rows: List[Dict[str, object]],
    wall_rows: List[Dict[str, object]],
    coating_summary_rows: object,
    wall_law_counts: Dict[str, int],
    wall_summary_counts: Dict[Tuple[int, str, str], int],
    stuck: np.ndarray,
    absorbed: np.ndarray,
    escaped: Optional[np.ndarray] = None,
    active: np.ndarray,
    max_wall_hits_per_step: int,
    min_remaining_dt: float,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
    t: float,
    triangle_surface_3d: Optional[TriangleSurface3D],
) -> WallHitStepResult:
    if escaped is None:
        escaped = np.zeros_like(active, dtype=bool)
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
            x_wall = x_plus
            n_wall = -n_wall
        elif _candidate_inside(hit_arr, float(on_boundary_tol_m)):
            x_wall = hit_arr.copy()
        else:
            x_wall = hit_arr - push * n_wall

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

    capture_wall_row = not bool(getattr(wall_rows, 'discarding', False))
    capture_coating_row = not bool(getattr(coating_summary_rows, 'discarding', False))
    event_row: Optional[Dict[str, object]] = None
    if bool(capture_wall_row) or bool(capture_coating_row):
        event_row = _wall_event_row(
            t_step_end=float(t),
            segment_dt=float(segment_dt),
            hit_dt=float(hit_dt_clamped),
            particle_id=int(particles.particle_id[particle_index]),
            particle_mass_kg=_particle_scalar_or_nan(particles, 'mass', particle_index),
            particle_diameter_m=_particle_scalar_or_nan(particles, 'diameter', particle_index),
            hit=hit_arr,
            normal=n_wall,
            v_hit=np.asarray(v_hit, dtype=np.float64),
            part_id=int(part_id),
            step=step,
            outcome=outcome,
            wall_model=wall_model,
            alpha_hit=float(alpha_eff),
            primitive_id=int(primitive_id),
            primitive_kind=str(primitive_kind),
            is_ambiguous=bool(is_ambiguous),
        )
        if bool(capture_wall_row):
            wall_rows.append(event_row)
        if bool(capture_coating_row):
            coating_summary_rows.append(event_row)

    hit_count += 1
    total_hit_count += 1
    hit_part_ids.append(int(part_id))
    hit_outcomes.append(str(outcome))

    remaining_dt = max(0.0, float(remaining_dt) - consumed_dt)

    if outcome == 'stuck':
        stuck[particle_index] = True
        active[particle_index] = False
        v_zero = np.zeros_like(v_hit)
        return WallHitStepResult(x_wall, v_zero, remaining_dt, hit_count, total_hit_count, True)
    if outcome == 'absorbed':
        absorbed[particle_index] = True
        active[particle_index] = False
        v_zero = np.zeros_like(v_hit)
        return WallHitStepResult(x_wall, v_zero, remaining_dt, hit_count, total_hit_count, True)
    if outcome == 'escaped':
        escaped[particle_index] = True
        active[particle_index] = False
        v_zero = np.zeros_like(v_hit)
        return WallHitStepResult(x_wall, v_zero, remaining_dt, hit_count, total_hit_count, True)

    x_curr_next = x_wall
    v_curr_next = np.asarray(v_ref, dtype=np.float64)

    if hit_count >= int(max_wall_hits_per_step):
        if remaining_dt > float(min_remaining_dt):
            contact_state = _same_wall_contact_sliding_state(
                x_wall=x_curr_next,
                v_ref=v_curr_next,
                n_wall=n_wall,
                remaining_dt=float(remaining_dt),
                hit_part_ids=hit_part_ids,
                hit_outcomes=hit_outcomes,
                collision_diagnostics=collision_diagnostics,
            )
            if contact_state is not None:
                x_contact, v_contact, n_contact = contact_state
                return WallHitStepResult(
                    x_contact,
                    v_contact,
                    0.0,
                    hit_count,
                    total_hit_count,
                    True,
                    True,
                    int(part_id),
                    np.asarray(n_contact, dtype=np.float64),
                    int(primitive_id),
                )
            collision_diagnostics['max_hits_reached_count'] += 1
            _record_max_hit_diagnostics(
                collision_diagnostics=collision_diagnostics,
                hit_part_ids=hit_part_ids,
                hit_outcomes=hit_outcomes,
                remaining_dt=float(remaining_dt),
            )
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
        return WallHitStepResult(x_curr_next, v_curr_next, remaining_dt, hit_count, total_hit_count, True)

    return WallHitStepResult(x_curr_next, v_curr_next, remaining_dt, hit_count, total_hit_count, False)


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
            candidates.append(0.5 * t_hit)
            candidates.append(t_hit)
            remaining = max(0.0, dt_seg - t_hit)
            if remaining > 1.0e-15:
                candidates.append(min(dt_seg, t_hit + max(1.0e-9 * dt_seg, 0.05 * remaining)))
                candidates.append(0.5 * (t_hit + dt_seg))
            candidates.append(dt_seg)
            unique = sorted({float(np.clip(v, 0.0, dt_seg)) for v in candidates if v > 0.0})
            return np.asarray(unique, dtype=np.float64)
    for frac in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        candidates.append(float(frac * dt_seg))
    unique = sorted({float(np.clip(v, 0.0, dt_seg)) for v in candidates if v > 0.0})
    return np.asarray(unique, dtype=np.float64)


def _locate_primary_hit_by_local_plane(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    segment_dt: float,
    t_end_segment: float,
    primary_hit: BoundaryHit,
    primary_hit_time: float,
    inputs: CollisionIntegratorInputs,
    adaptive_substep_enabled: int,
    on_boundary_tol_m: float,
    max_iters: int,
) -> Optional[Tuple[BoundaryHit, np.ndarray, float]]:
    dt_seg = max(float(segment_dt), 0.0)
    if dt_seg <= 0.0:
        return None
    hit_position = np.asarray(primary_hit.position, dtype=np.float64)
    normal = np.asarray(primary_hit.normal, dtype=np.float64)
    normal_mag = float(np.linalg.norm(normal))
    if hit_position.ndim != 1 or normal.ndim != 1 or hit_position.size != normal.size or normal_mag <= 1.0e-30:
        return None
    n_unit = normal / normal_mag
    x_start = np.asarray(x0, dtype=np.float64)
    v_start = np.asarray(v0, dtype=np.float64)
    s_start = float(np.dot(x_start - hit_position, n_unit))
    if not np.isfinite(s_start) or abs(s_start) <= max(float(on_boundary_tol_m), 1.0e-14):
        return None

    def _state_at(t_partial: float) -> Tuple[float, np.ndarray, np.ndarray]:
        x_t, v_t = _advance_partial_with_inputs(
            inputs=inputs,
            x0=x_start,
            v0=v_start,
            dt_partial=float(t_partial),
            segment_dt=float(dt_seg),
            t_end_segment=float(t_end_segment),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
        )
        signed = float(np.dot(np.asarray(x_t, dtype=np.float64) - hit_position, n_unit))
        return signed, np.asarray(x_t, dtype=np.float64), np.asarray(v_t, dtype=np.float64)

    t_guess = float(np.clip(primary_hit_time, 0.0, dt_seg))
    candidates = [t_guess, dt_seg]
    if t_guess > 0.0:
        candidates.extend((0.5 * t_guess, min(dt_seg, t_guess + 0.1 * max(0.0, dt_seg - t_guess))))
    candidates = sorted({float(np.clip(v, 0.0, dt_seg)) for v in candidates if v > 1.0e-15})

    t_lo = 0.0
    s_lo = s_start
    x_lo = x_start
    v_lo = v_start
    t_hi: Optional[float] = None
    s_hi: Optional[float] = None
    x_hi: Optional[np.ndarray] = None
    v_hi: Optional[np.ndarray] = None
    for t_candidate in candidates:
        s_candidate, x_candidate, v_candidate = _state_at(float(t_candidate))
        if not np.isfinite(s_candidate):
            continue
        if s_lo == 0.0 or s_lo * s_candidate <= 0.0:
            t_hi = float(t_candidate)
            s_hi = float(s_candidate)
            x_hi = x_candidate
            v_hi = v_candidate
            break
        t_lo = float(t_candidate)
        s_lo = float(s_candidate)
        x_lo = x_candidate
        v_lo = v_candidate
    if t_hi is None or s_hi is None or x_hi is None or v_hi is None:
        return None

    stop_time_tol = max(1.0e-12, 1.0e-7 * dt_seg)
    stop_signed_tol = max(float(on_boundary_tol_m), 1.0e-12)
    for _ in range(int(max(1, max_iters))):
        if float(t_hi - t_lo) <= stop_time_tol or min(abs(float(s_lo)), abs(float(s_hi))) <= stop_signed_tol:
            break
        t_mid = 0.5 * (float(t_lo) + float(t_hi))
        s_mid, x_mid, v_mid = _state_at(float(t_mid))
        if not np.isfinite(s_mid):
            break
        if s_lo == 0.0 or s_lo * s_mid <= 0.0:
            t_hi = float(t_mid)
            s_hi = float(s_mid)
            x_hi = x_mid
            v_hi = v_mid
        else:
            t_lo = float(t_mid)
            s_lo = float(s_mid)
            x_lo = x_mid
            v_lo = v_mid

    if abs(float(s_lo)) <= abs(float(s_hi)):
        t_hit = float(t_lo)
        x_hit_state = np.asarray(x_lo, dtype=np.float64)
        v_hit = np.asarray(v_lo, dtype=np.float64)
        signed_hit = float(s_lo)
    else:
        t_hit = float(t_hi)
        x_hit_state = np.asarray(x_hi, dtype=np.float64)
        v_hit = np.asarray(v_hi, dtype=np.float64)
        signed_hit = float(s_hi)
    # Keep the reported hit point on the finite primitive returned by the
    # primary intersection. The plane refinement improves hit time/velocity,
    # but its normal projection lies on an infinite tangent plane and can drift
    # a few microns beyond short edge endpoints.
    x_projected = np.asarray(primary_hit.position, dtype=np.float64)
    hit_alpha = 0.0 if dt_seg <= 1.0e-30 else float(np.clip(t_hit / dt_seg, 0.0, 1.0))
    return (
        BoundaryHit(
            position=np.asarray(x_projected, dtype=np.float64),
            normal=np.asarray(primary_hit.normal, dtype=np.float64),
            part_id=int(primary_hit.part_id),
            alpha_hint=float(hit_alpha),
            primitive_id=int(primary_hit.primitive_id),
            primitive_kind=str(primary_hit.primitive_kind),
            is_ambiguous=bool(primary_hit.is_ambiguous),
        ),
        np.asarray(v_hit, dtype=np.float64),
        float(t_hit),
    )


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
        if primary_hit_time > 1.0e-15:
            primary_event = _locate_primary_hit_by_local_plane(
                x0=x0,
                v0=v0,
                segment_dt=dt_seg,
                t_end_segment=float(t_end_segment),
                primary_hit=primary_hit,
                primary_hit_time=float(primary_hit_time),
                inputs=inputs,
                adaptive_substep_enabled=int(adaptive_substep_enabled),
                on_boundary_tol_m=float(on_boundary_tol_m),
                max_iters=min(int(max_iters), 18),
            )
            if primary_event is not None:
                return primary_event
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
            primitive_id=int(nearest.primitive_id),
            primitive_kind=str(nearest.primitive_kind),
            is_ambiguous=bool(nearest.is_ambiguous),
        ),
        np.asarray(v_hit, dtype=np.float64),
        float(t_hit),
    )


class CollidingParticleAdvanceResult(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    total_hits: int
    valid_mask_status: int
    invalid_mask_stopped: bool
    invalid_stop_reason: str = ''
    numerical_boundary_stopped: bool = False
    numerical_boundary_stop_reason: str = ''
    contact_sliding: bool = False
    contact_part_id: int = 0
    contact_normal: Optional[np.ndarray] = None
    contact_primitive_id: int = -1


@dataclass(frozen=True)
class CollisionSegmentTrial:
    segment_adaptive_enabled: int
    x_next: np.ndarray
    v_next: np.ndarray
    stage_points: np.ndarray
    primary_hit: Optional[BoundaryHit]
    primary_hit_counted: bool
    particle_valid_mask_status: int
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
    particle_diameter_i: float
    particle_density_i: float
    flow_scale_particle_i: float
    drag_scale_particle_i: float
    body_scale_particle_i: float
    global_flow_scale: float
    global_drag_tau_scale: float
    global_body_accel_scale: float
    body_accel: np.ndarray
    min_tau_p_s: float
    gas_density_kgm3: float
    gas_mu_pas: float
    gas_temperature_K: float
    gas_molecular_mass_kg: float
    drag_model_mode: int
    electric_q_over_m_i: Optional[float] = None


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
        particle_diameter_i=float(inputs.particle_diameter_i),
        particle_density_i=float(inputs.particle_density_i),
        flow_scale_particle_i=float(inputs.flow_scale_particle_i),
        drag_scale_particle_i=float(inputs.drag_scale_particle_i),
        body_scale_particle_i=float(inputs.body_scale_particle_i),
        global_flow_scale=float(inputs.global_flow_scale),
        global_drag_tau_scale=float(inputs.global_drag_tau_scale),
        global_body_accel_scale=float(inputs.global_body_accel_scale),
        body_accel=np.asarray(inputs.body_accel, dtype=np.float64),
        min_tau_p_s=float(inputs.min_tau_p_s),
        gas_density_kgm3=float(inputs.gas_density_kgm3),
        gas_mu_pas=float(inputs.gas_mu_pas),
        gas_temperature_K=float(inputs.gas_temperature_K),
        gas_molecular_mass_kg=float(inputs.gas_molecular_mass_kg),
        drag_model_mode=int(inputs.drag_model_mode),
        electric_q_over_m_i=inputs.electric_q_over_m_i,
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
        particle_diameter_i=float(inputs.particle_diameter_i),
        particle_density_i=float(inputs.particle_density_i),
        flow_scale_particle_i=float(inputs.flow_scale_particle_i),
        drag_scale_particle_i=float(inputs.drag_scale_particle_i),
        body_scale_particle_i=float(inputs.body_scale_particle_i),
        global_flow_scale=float(inputs.global_flow_scale),
        global_drag_tau_scale=float(inputs.global_drag_tau_scale),
        global_body_accel_scale=float(inputs.global_body_accel_scale),
        body_accel=np.asarray(inputs.body_accel, dtype=np.float64),
        min_tau_p_s=float(inputs.min_tau_p_s),
        gas_density_kgm3=float(inputs.gas_density_kgm3),
        gas_mu_pas=float(inputs.gas_mu_pas),
        gas_temperature_K=float(inputs.gas_temperature_K),
        gas_molecular_mass_kg=float(inputs.gas_molecular_mass_kg),
        drag_model_mode=int(inputs.drag_model_mode),
        electric_q_over_m_i=inputs.electric_q_over_m_i,
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
        particle_diameter_i=float(inputs.particle_diameter_i),
        particle_density_i=float(inputs.particle_density_i),
        flow_scale_particle_i=float(inputs.flow_scale_particle_i),
        drag_scale_particle_i=float(inputs.drag_scale_particle_i),
        body_scale_particle_i=float(inputs.body_scale_particle_i),
        global_flow_scale=float(inputs.global_flow_scale),
        global_drag_tau_scale=float(inputs.global_drag_tau_scale),
        global_body_accel_scale=float(inputs.global_body_accel_scale),
        body_accel=np.asarray(inputs.body_accel, dtype=np.float64),
        min_tau_p_s=float(inputs.min_tau_p_s),
        gas_density_kgm3=float(inputs.gas_density_kgm3),
        gas_mu_pas=float(inputs.gas_mu_pas),
        gas_temperature_K=float(inputs.gas_temperature_K),
        gas_molecular_mass_kg=float(inputs.gas_molecular_mass_kg),
        drag_model_mode=int(inputs.drag_model_mode),
        electric_q_over_m_i=inputs.electric_q_over_m_i,
    )


def _increment_named_count(collision_diagnostics: Dict[str, object], key: str, name: str) -> None:
    label = str(name).strip() or 'unknown'
    counts = collision_diagnostics.setdefault(key, {})
    if not isinstance(counts, dict):
        counts = {}
        collision_diagnostics[key] = counts
    counts[label] = int(counts.get(label, 0)) + 1


def _same_wall_contact_sliding_state(
    *,
    x_wall: np.ndarray,
    v_ref: np.ndarray,
    n_wall: np.ndarray,
    remaining_dt: float,
    hit_part_ids: List[int],
    hit_outcomes: List[str],
    collision_diagnostics: Dict[str, object],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(hit_part_ids) < 2:
        return None
    if len({int(pid) for pid in hit_part_ids}) != 1:
        return None
    if any(str(outcome) not in _CONTACT_REFLECTED_OUTCOMES for outcome in hit_outcomes):
        return None
    n = np.asarray(n_wall, dtype=np.float64)
    n_mag = float(np.linalg.norm(n))
    if n_mag <= 1.0e-30:
        return None
    n = n / n_mag
    v = np.asarray(v_ref, dtype=np.float64)
    if v.size != n.size:
        return None
    v_tangent = v - float(np.dot(v, n)) * n
    if float(np.linalg.norm(v_tangent)) <= 1.0e-14:
        v_tangent = np.zeros_like(v)
    _increment_collision_diagnostic(collision_diagnostics, 'contact_sliding_count')
    _increment_collision_diagnostic(collision_diagnostics, 'contact_sliding_same_wall_count')
    collision_diagnostics['contact_sliding_time_total_s'] = float(
        collision_diagnostics.get('contact_sliding_time_total_s', 0.0)
    ) + float(max(0.0, remaining_dt))
    collision_diagnostics['contact_sliding_remaining_dt_max_s'] = max(
        float(collision_diagnostics.get('contact_sliding_remaining_dt_max_s', 0.0)),
        float(max(0.0, remaining_dt)),
    )
    _increment_named_count(collision_diagnostics, 'contact_sliding_part_counts', f'part={int(hit_part_ids[-1])}')
    if hit_outcomes:
        _increment_named_count(collision_diagnostics, 'contact_sliding_outcome_counts', str(hit_outcomes[-1]))
    return np.asarray(x_wall, dtype=np.float64), v_tangent, n


def _record_max_hit_diagnostics(
    *,
    collision_diagnostics: Dict[str, object],
    hit_part_ids: List[int],
    hit_outcomes: List[str],
    remaining_dt: float,
) -> None:
    if not hit_part_ids:
        return
    unique_parts = {int(pid) for pid in hit_part_ids}
    if len(unique_parts) <= 1:
        _increment_collision_diagnostic(collision_diagnostics, 'max_hit_same_wall_count')
    else:
        _increment_collision_diagnostic(collision_diagnostics, 'max_hit_multi_wall_count')
    _increment_named_count(collision_diagnostics, 'max_hit_last_part_counts', f'part={int(hit_part_ids[-1])}')
    if hit_outcomes:
        _increment_named_count(collision_diagnostics, 'max_hit_last_outcome_counts', str(hit_outcomes[-1]))
    collision_diagnostics['max_hit_remaining_dt_total_s'] = float(
        collision_diagnostics.get('max_hit_remaining_dt_total_s', 0.0)
    ) + float(max(0.0, remaining_dt))
    collision_diagnostics['max_hit_remaining_dt_max_s'] = max(
        float(collision_diagnostics.get('max_hit_remaining_dt_max_s', 0.0)),
        float(max(0.0, remaining_dt)),
    )


def _post_wall_acceptance_reason(
    *,
    runtime,
    position: np.ndarray,
    velocity: np.ndarray,
    inside_fn: Callable[[np.ndarray], bool],
) -> str:
    pos = np.asarray(position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
        return 'post_wall_nonfinite_state'
    try:
        if not bool(inside_fn(pos)):
            return 'post_wall_outside_geometry'
    except Exception:
        return 'post_wall_geometry_check_failed'
    return ''


def _prepare_collision_segment_trial(
    *,
    use_precomputed_trial: bool,
    x_curr: np.ndarray,
    v_curr: np.ndarray,
    t: float,
    segment_dt: float,
    inputs: CollisionIntegratorInputs,
    base_adaptive_substep_enabled: int,
    valid_mask_retry_then_stop_enabled: bool,
    initial_x_next: np.ndarray,
    initial_v_next: np.ndarray,
    initial_stage_points: np.ndarray,
    initial_valid_mask_status: int,
    initial_primary_hit: Optional[BoundaryHit],
    initial_primary_hit_counted: bool,
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], Optional[BoundaryHit]],
    collision_diagnostics: Dict[str, object],
) -> CollisionSegmentTrial:
    particle_valid_mask_status = int(initial_valid_mask_status)
    segment_start_x = np.asarray(x_curr, dtype=np.float64).copy()
    segment_start_v = np.asarray(v_curr, dtype=np.float64).copy()
    if bool(use_precomputed_trial):
        segment_adaptive_enabled = int(base_adaptive_substep_enabled)
        x_next = np.asarray(initial_x_next, dtype=np.float64)
        v_next = np.asarray(initial_v_next, dtype=np.float64)
        stage_points = np.asarray(initial_stage_points, dtype=np.float64)
        segment_mask_status = int(initial_valid_mask_status)
        ordering_primary_hit = initial_primary_hit
        ordering_primary_hit_counted = bool(initial_primary_hit_counted)
    else:
        segment_adaptive_enabled = int(base_adaptive_substep_enabled)
        x_next, v_next, n_substeps, stage_points, segment_mask_status = _advance_segment_with_inputs(
            inputs=inputs,
            x0=x_curr,
            v0=v_curr,
            dt_segment=float(segment_dt),
            t_end_segment=float(t),
            adaptive_substep_enabled=int(segment_adaptive_enabled),
        )
        if int(segment_mask_status) > int(particle_valid_mask_status):
            particle_valid_mask_status = int(segment_mask_status)
        collision_diagnostics['collision_reintegrated_segments_count'] += 1
        if int(segment_adaptive_enabled) != 0:
            collision_diagnostics['adaptive_substep_segments_count'] += int(n_substeps)
            if int(n_substeps) > 1:
                collision_diagnostics['adaptive_substep_trigger_count'] += 1
        ordering_primary_hit = None
        ordering_primary_hit_counted = False
    if bool(valid_mask_retry_then_stop_enabled) and bool(valid_mask_status_requires_stop(int(segment_mask_status))):
        # For colliding particles, a hard-invalid trial endpoint can simply be
        # the state beyond a physical wall. Resolve that wall hit first; only
        # stop on valid_mask when no boundary crossing is found for the segment.
        valid_mask_primary_hit = ordering_primary_hit
        if valid_mask_primary_hit is None:
            try:
                valid_mask_primary_hit = primary_hit_fn(segment_start_x, np.asarray(stage_points, dtype=np.float64))
            except Exception:
                valid_mask_primary_hit = None
        if valid_mask_primary_hit is not None:
            ordering_primary_hit = valid_mask_primary_hit
            ordering_primary_hit_counted = False
        else:
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
                invalid_stop_result=CollidingParticleAdvanceResult(
                    position=resolution.position,
                    velocity=resolution.velocity,
                    total_hits=0,
                    valid_mask_status=int(particle_valid_mask_status),
                    invalid_mask_stopped=True,
                    invalid_stop_reason=(
                        'collision_valid_mask_hard_invalid_prefix_clipped'
                        if bool(resolution.found_valid_prefix)
                        else 'collision_valid_mask_hard_invalid_retry_exhausted'
                    ),
                ),
            )
    return CollisionSegmentTrial(
        segment_adaptive_enabled=int(segment_adaptive_enabled),
        x_next=np.asarray(x_next, dtype=np.float64),
        v_next=np.asarray(v_next, dtype=np.float64),
        stage_points=np.asarray(stage_points, dtype=np.float64),
        primary_hit=ordering_primary_hit,
        primary_hit_counted=bool(ordering_primary_hit_counted),
        particle_valid_mask_status=int(particle_valid_mask_status),
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

    if resolved_primary_hit is None:
        stage_inside = np.asarray([bool(inside_fn(pt)) for pt in stage_points_arr], dtype=bool)
        if bool(np.all(stage_inside)):
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

    if resolved_primary_hit is not None:
        fallback_dt = float(np.clip(float(resolved_primary_hit.alpha_hint), 0.0, 1.0) * float(segment_dt))
        if fallback_dt > 1.0e-15:
            _x_at_hit, v_at_hit = _advance_partial_with_inputs(
                inputs=inputs,
                x0=x_curr,
                v0=v_curr,
                dt_partial=float(fallback_dt),
                segment_dt=float(segment_dt),
                t_end_segment=float(t),
                adaptive_substep_enabled=int(adaptive_substep_enabled),
            )
            if np.all(np.isfinite(v_at_hit)):
                collision_diagnostics['primary_hit_direct_resolution_count'] = int(
                    collision_diagnostics.get('primary_hit_direct_resolution_count', 0)
                ) + 1
                return CollisionSegmentResolution(
                    advance_without_hit=False,
                    should_break=False,
                    x_next=np.asarray(x_next, dtype=np.float64),
                    v_next=np.asarray(v_next, dtype=np.float64),
                    hit_event=BoundaryHit(
                        position=np.asarray(resolved_primary_hit.position, dtype=np.float64),
                        normal=np.asarray(resolved_primary_hit.normal, dtype=np.float64),
                        part_id=int(resolved_primary_hit.part_id),
                        alpha_hint=float(np.clip(fallback_dt / max(float(segment_dt), 1.0e-30), 0.0, 1.0)),
                        primitive_id=int(resolved_primary_hit.primitive_id),
                        primitive_kind=str(resolved_primary_hit.primitive_kind),
                        is_ambiguous=bool(resolved_primary_hit.is_ambiguous),
                    ),
                    v_hit=np.asarray(v_at_hit, dtype=np.float64),
                    hit_dt=float(fallback_dt),
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
    tau_p_i: float,
    particle_diameter_i: float,
    flow_scale_particle_i: float,
    drag_scale_particle_i: float,
    body_scale_particle_i: float,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    global_body_accel_scale: float,
    body_accel: np.ndarray,
    min_tau_p_s: float,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    drag_model_mode: int,
    valid_mask_retry_then_stop_enabled: bool,
    initial_x_next: np.ndarray,
    initial_v_next: np.ndarray,
    initial_stage_points: np.ndarray,
    initial_valid_mask_status: int,
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
    coating_summary_rows: object,
    wall_law_counts: Dict[str, int],
    wall_summary_counts: Dict[Tuple[int, str, str], int],
    stuck: np.ndarray,
    absorbed: np.ndarray,
    escaped: Optional[np.ndarray] = None,
    active: np.ndarray,
    max_wall_hits_per_step: int,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
    triangle_surface_3d: Optional[TriangleSurface3D],
    electric_q_over_m_i: Optional[float] = None,
    particle_density_i: float = 1000.0,
    gas_temperature_K: float = 300.0,
    gas_molecular_mass_kg: float = 60.0 * 1.66053906660e-27,
) -> CollidingParticleAdvanceResult:
    if escaped is None:
        escaped = np.zeros_like(active, dtype=bool)
    remaining_dt = float(dt_step)
    min_remaining_dt = float(dt_step * min_remaining_dt_ratio)
    x_curr = np.asarray(x_start, dtype=np.float64).copy()
    v_curr = np.asarray(v_start, dtype=np.float64).copy()
    hit_count = 0
    total_hit_count = 0
    hit_part_ids: List[int] = []
    hit_outcomes: List[str] = []
    use_precomputed_trial = True
    numerical_boundary_stopped = False
    numerical_boundary_stop_reason = ''
    contact_sliding = False
    contact_part_id = 0
    contact_normal: Optional[np.ndarray] = None
    contact_primitive_id = -1
    particle_valid_mask_status = int(initial_valid_mask_status)
    integrator_inputs = CollisionIntegratorInputs(
        spatial_dim=int(spatial_dim),
        compiled=compiled,
        integrator_mode=int(integrator_mode),
        adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        tau_p_i=float(tau_p_i),
        particle_diameter_i=float(particle_diameter_i),
        particle_density_i=float(particle_density_i),
        flow_scale_particle_i=float(flow_scale_particle_i),
        drag_scale_particle_i=float(drag_scale_particle_i),
        body_scale_particle_i=float(body_scale_particle_i),
        global_flow_scale=float(global_flow_scale),
        global_drag_tau_scale=float(global_drag_tau_scale),
        global_body_accel_scale=float(global_body_accel_scale),
        body_accel=np.asarray(body_accel, dtype=np.float64),
        min_tau_p_s=float(min_tau_p_s),
        gas_density_kgm3=float(gas_density_kgm3),
        gas_mu_pas=float(gas_mu_pas),
        gas_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        drag_model_mode=int(drag_model_mode),
        electric_q_over_m_i=electric_q_over_m_i,
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
            valid_mask_retry_then_stop_enabled=bool(valid_mask_retry_then_stop_enabled),
            initial_x_next=initial_x_next,
            initial_v_next=initial_v_next,
            initial_stage_points=initial_stage_points,
            initial_valid_mask_status=int(particle_valid_mask_status),
            initial_primary_hit=initial_primary_hit,
            initial_primary_hit_counted=bool(initial_primary_hit_counted),
            primary_hit_fn=primary_hit_fn,
            collision_diagnostics=collision_diagnostics,
        )
        particle_valid_mask_status = int(segment_trial.particle_valid_mask_status)
        if segment_trial.invalid_stop_result is not None:
            return CollidingParticleAdvanceResult(
                position=np.asarray(segment_trial.invalid_stop_result.position, dtype=np.float64),
                velocity=np.asarray(segment_trial.invalid_stop_result.velocity, dtype=np.float64),
                total_hits=int(total_hit_count),
                valid_mask_status=int(segment_trial.invalid_stop_result.valid_mask_status),
                invalid_mask_stopped=True,
                invalid_stop_reason=str(segment_trial.invalid_stop_result.invalid_stop_reason),
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
        primitive_id = int(segment_resolution.hit_event.primitive_id)
        primitive_kind = str(segment_resolution.hit_event.primitive_kind)
        is_ambiguous = bool(segment_resolution.hit_event.is_ambiguous)

        wall_result = _apply_wall_hit_step(
            runtime=runtime,
            step=step,
            particles=particles,
            particle_index=int(particle_index),
            rng=rng,
            hit=hit,
            n_out=n_out,
            hit_dt=float(segment_resolution.hit_dt),
            part_id=int(part_id),
            primitive_id=int(primitive_id),
            primitive_kind=str(primitive_kind),
            is_ambiguous=bool(is_ambiguous),
            v_hit=np.asarray(segment_resolution.v_hit, dtype=np.float64),
            remaining_dt=float(remaining_dt),
            segment_dt=float(segment_dt),
            hit_count=int(hit_count),
            total_hit_count=int(total_hit_count),
            hit_part_ids=hit_part_ids,
            hit_outcomes=hit_outcomes,
            collision_diagnostics=collision_diagnostics,
            max_hit_rows=max_hit_rows,
            wall_rows=wall_rows,
            coating_summary_rows=coating_summary_rows,
            wall_law_counts=wall_law_counts,
            wall_summary_counts=wall_summary_counts,
            stuck=stuck,
            absorbed=absorbed,
            escaped=escaped,
            active=active,
            max_wall_hits_per_step=int(max_wall_hits_per_step),
            min_remaining_dt=float(min_remaining_dt),
            epsilon_offset_m=float(epsilon_offset_m),
            on_boundary_tol_m=float(on_boundary_tol_m),
            t=float(t),
            triangle_surface_3d=triangle_surface_3d,
        )
        x_curr = np.asarray(wall_result.position, dtype=np.float64)
        v_curr = np.asarray(wall_result.velocity, dtype=np.float64)
        remaining_dt = float(wall_result.remaining_dt)
        hit_count = int(wall_result.hit_count)
        total_hit_count = int(wall_result.total_hit_count)
        should_break = bool(wall_result.should_break)
        if bool(wall_result.entered_contact):
            contact_sliding = True
            contact_part_id = int(wall_result.contact_part_id)
            contact_normal = (
                None
                if wall_result.contact_normal is None
                else np.asarray(wall_result.contact_normal, dtype=np.float64)
            )
            contact_primitive_id = int(wall_result.contact_primitive_id)
        if (
            bool(active[particle_index])
            and not bool(stuck[particle_index])
            and not bool(absorbed[particle_index])
            and not bool(escaped[particle_index])
        ):
            acceptance_reason = _post_wall_acceptance_reason(
                runtime=runtime,
                position=x_curr,
                velocity=v_curr,
                inside_fn=inside_fn,
            )
            if acceptance_reason:
                numerical_boundary_stopped = True
                numerical_boundary_stop_reason = str(acceptance_reason)
                break
        if (
            bool(should_break)
            and bool(active[particle_index])
            and not bool(stuck[particle_index])
            and not bool(absorbed[particle_index])
            and not bool(escaped[particle_index])
            and int(hit_count) >= int(max_wall_hits_per_step)
            and float(remaining_dt) > float(min_remaining_dt)
        ):
            numerical_boundary_stopped = True
            numerical_boundary_stop_reason = 'max_hits_reached'
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
        invalid_mask_stopped=False,
        invalid_stop_reason='',
        numerical_boundary_stopped=bool(numerical_boundary_stopped),
        numerical_boundary_stop_reason=str(numerical_boundary_stop_reason),
        contact_sliding=bool(contact_sliding),
        contact_part_id=int(contact_part_id),
        contact_normal=contact_normal,
        contact_primitive_id=int(contact_primitive_id),
    )


@dataclass(frozen=True)
class TrialCollisionBatch:
    colliders: np.ndarray
    safe: np.ndarray
    prefetched_hits: Dict[int, BoundaryHit]


def _add_timing(timing_accumulator: Optional[Dict[str, float]], key: str, elapsed_s: float) -> None:
    if timing_accumulator is None:
        return
    timing_accumulator[key] = float(timing_accumulator.get(key, 0.0)) + float(max(0.0, elapsed_s))


def _min_geometry_grid_spacing_2d(runtime) -> float:
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None:
        return 0.0
    geom = geometry_provider.geometry
    if int(getattr(geom, 'spatial_dim', 0)) != 2 or len(getattr(geom, 'axes', ())) != 2:
        return 0.0
    spacings = []
    for axis in geom.axes:
        arr = np.asarray(axis, dtype=np.float64)
        diffs = np.diff(arr)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            spacings.append(float(np.min(diffs)))
    return float(min(spacings)) if spacings else 0.0


def _far_from_wall_mask_2d(
    runtime,
    indices: np.ndarray,
    x_start: np.ndarray,
    x_mid: np.ndarray,
    x_end: np.ndarray,
    *,
    on_boundary_tol_m: float,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.zeros(0, dtype=bool)
    geometry_provider = getattr(runtime, 'geometry_provider', None)
    if geometry_provider is None or int(geometry_provider.geometry.spatial_dim) != 2:
        return np.zeros(idx.size, dtype=bool)
    grid_spacing = _min_geometry_grid_spacing_2d(runtime)
    if not np.isfinite(grid_spacing) or grid_spacing <= 0.0:
        return np.zeros(idx.size, dtype=bool)
    start = np.asarray(x_start[idx], dtype=np.float64)
    mid = np.asarray(x_mid[idx], dtype=np.float64)
    end = np.asarray(x_end[idx], dtype=np.float64)
    sdf_start = sample_geometry_sdf_points_2d(runtime, start)
    sdf_mid = sample_geometry_sdf_points_2d(runtime, mid)
    sdf_end = sample_geometry_sdf_points_2d(runtime, end)
    sweep_radius = np.maximum(
        np.linalg.norm(mid - start, axis=1),
        np.linalg.norm(end - start, axis=1),
    )
    margin = float(max(float(on_boundary_tol_m), 2.0 * grid_spacing)) + 0.25 * sweep_radius
    finite = np.isfinite(sdf_start) & np.isfinite(sdf_mid) & np.isfinite(sdf_end) & np.isfinite(sweep_radius)
    return finite & (sdf_start < -(sweep_radius + margin)) & (sdf_mid < -margin) & (sdf_end < -margin)


def _sdf_strict_inside_mask_2d(
    runtime,
    positions: np.ndarray,
    *,
    on_boundary_tol_m: float,
) -> np.ndarray:
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return np.zeros(pts.shape[0], dtype=bool)
    grid_spacing = _min_geometry_grid_spacing_2d(runtime)
    if not np.isfinite(grid_spacing) or grid_spacing <= 0.0:
        return np.zeros(pts.shape[0], dtype=bool)
    sdf = sample_geometry_sdf_points_2d(runtime, pts)
    margin = float(max(float(on_boundary_tol_m), 0.5 * grid_spacing))
    return np.isfinite(sdf) & (sdf < -margin)


def _classify_trial_collisions_2d(
    runtime,
    *,
    n_particles: int,
    active: np.ndarray,
    x: np.ndarray,
    x_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    integrator_mode: int,
    boundary_service: BoundaryService,
    on_boundary_tol_m: float,
    collision_diagnostics: Dict[str, object],
    timing_accumulator: Optional[Dict[str, float]] = None,
    valid_mask_status_flags: Optional[np.ndarray] = None,
) -> TrialCollisionBatch:
    active_idx = np.flatnonzero(active)
    loop_inside = np.zeros(n_particles, dtype=bool)
    loop_mid_inside = np.ones(n_particles, dtype=bool)
    is_etd2 = int(integrator_mode) == INTEGRATOR_ETD2
    far_from_wall = np.zeros(n_particles, dtype=bool)
    if active_idx.size:
        t_prefilter = time.perf_counter()
        far_active = _far_from_wall_mask_2d(
            runtime,
            active_idx,
            x,
            x_mid_trial if is_etd2 else x_trial,
            x_trial,
            on_boundary_tol_m=float(on_boundary_tol_m),
        )
        if valid_mask_status_flags is not None:
            far_active &= (
                np.asarray(valid_mask_status_flags[active_idx], dtype=np.uint8)
                == np.uint8(VALID_MASK_STATUS_CLEAN)
            )
        _add_timing(timing_accumulator, 'boundary_sdf_prefilter_s', time.perf_counter() - t_prefilter)
        if far_active.size:
            far_from_wall[active_idx] = far_active
            loop_inside[active_idx[far_active]] = True
            loop_mid_inside[active_idx[far_active]] = True
            collision_diagnostics['boundary_far_skip_count'] = int(
                collision_diagnostics.get('boundary_far_skip_count', 0)
            ) + int(np.count_nonzero(far_active))
            collision_diagnostics['boundary_near_check_count'] = int(
                collision_diagnostics.get('boundary_near_check_count', 0)
            ) + int(active_idx.size - int(np.count_nonzero(far_active)))
        near_idx = active_idx[~far_active]
        if near_idx.size:
            t_sdf_inside = time.perf_counter()
            end_sdf_inside = _sdf_strict_inside_mask_2d(
                runtime,
                x_trial[near_idx],
                on_boundary_tol_m=float(on_boundary_tol_m),
            )
            loop_inside[near_idx[end_sdf_inside]] = True
            _add_timing(timing_accumulator, 'inside_sdf_prefilter_s', time.perf_counter() - t_sdf_inside)
            end_exact_idx = near_idx[~end_sdf_inside]
            if end_exact_idx.size:
                t_inside = time.perf_counter()
                inside_active, on_boundary_active = _boundary_points_inside_geometry_2d(
                    runtime,
                    x_trial[end_exact_idx],
                    on_boundary_tol_m=on_boundary_tol_m,
                    return_on_boundary=True,
                )
                loop_inside[end_exact_idx] = inside_active
                collision_diagnostics['on_boundary_promoted_inside_count'] += int(np.count_nonzero(on_boundary_active))
                _add_timing(timing_accumulator, 'inside_check_s', time.perf_counter() - t_inside)
            if is_etd2:
                t_sdf_inside = time.perf_counter()
                mid_sdf_inside = _sdf_strict_inside_mask_2d(
                    runtime,
                    x_mid_trial[near_idx],
                    on_boundary_tol_m=float(on_boundary_tol_m),
                )
                loop_mid_inside[near_idx[mid_sdf_inside]] = True
                _add_timing(timing_accumulator, 'inside_sdf_prefilter_s', time.perf_counter() - t_sdf_inside)
                mid_exact_idx = near_idx[~mid_sdf_inside]
                if mid_exact_idx.size:
                    t_inside = time.perf_counter()
                    inside_mid_active, on_boundary_mid_active = _boundary_points_inside_geometry_2d(
                        runtime,
                        x_mid_trial[mid_exact_idx],
                        on_boundary_tol_m=on_boundary_tol_m,
                        return_on_boundary=True,
                    )
                    loop_mid_inside[mid_exact_idx] = inside_mid_active
                    collision_diagnostics['on_boundary_promoted_inside_count'] += int(np.count_nonzero(on_boundary_mid_active))
                    collision_diagnostics['etd2_midpoint_outside_count'] += int(np.count_nonzero(~inside_mid_active))
                    _add_timing(timing_accumulator, 'inside_check_s', time.perf_counter() - t_inside)
    collider_mask = active & ((~loop_inside) | (~loop_mid_inside)) if is_etd2 else active & (~loop_inside)
    safe_mask = active & loop_inside & loop_mid_inside if is_etd2 else active & loop_inside
    prefetched_hits: Dict[int, BoundaryHit] = {}
    safe_idx = np.flatnonzero(safe_mask & (~far_from_wall))
    if safe_idx.size:
        if is_etd2:
            stage_points_batch = np.stack((x_mid_trial[safe_idx], x_trial[safe_idx]), axis=1)
        else:
            stage_points_batch = x_trial[safe_idx][:, None, :]
        t_prefetch = time.perf_counter()
        batch_hits = polyline_hits_from_boundary_edges_batch(
            runtime,
            x[safe_idx],
            stage_points_batch,
            particle_indices=safe_idx,
        )
        _add_timing(timing_accumulator, 'edge_prefetch_s', time.perf_counter() - t_prefetch)
        collision_diagnostics['edge_prefetch_batch_candidate_count'] = int(
            collision_diagnostics.get('edge_prefetch_batch_candidate_count', 0)
        ) + int(safe_idx.size)
        for particle_index, hit in batch_hits.items():
            if float(hit.alpha_hint) <= 1.0e-12:
                continue
            i = int(particle_index)
            prefetched_hits[i] = hit
            collider_mask[i] = True
            safe_mask[i] = False
        collision_diagnostics['edge_prefetch_batch_hit_count'] = int(
            collision_diagnostics.get('edge_prefetch_batch_hit_count', 0)
        ) + int(len(prefetched_hits))
    colliders = np.flatnonzero(collider_mask)
    safe = np.flatnonzero(safe_mask)
    return TrialCollisionBatch(
        colliders=np.asarray(colliders, dtype=np.int64),
        safe=np.asarray(safe, dtype=np.int64),
        prefetched_hits=prefetched_hits,
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
    timing_accumulator: Optional[Dict[str, float]] = None,
    valid_mask_status_flags: Optional[np.ndarray] = None,
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
    timing_accumulator: Optional[Dict[str, float]] = None,
    valid_mask_status_flags: Optional[np.ndarray] = None,
) -> TrialCollisionBatch:
    if int(spatial_dim) == 2:
        return _classify_trial_collisions_2d(
            runtime,
            n_particles=int(n_particles),
            active=active,
            x=x,
            x_trial=x_trial,
            x_mid_trial=x_mid_trial,
            integrator_mode=int(integrator_mode),
            boundary_service=boundary_service,
            on_boundary_tol_m=float(on_boundary_tol_m),
            collision_diagnostics=collision_diagnostics,
            timing_accumulator=timing_accumulator,
            valid_mask_status_flags=valid_mask_status_flags,
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
        timing_accumulator=timing_accumulator,
        valid_mask_status_flags=valid_mask_status_flags,
    )


__all__ = (
    'TrialCollisionBatch',
    '_apply_wall_hit_step',
    '_classify_trial_collisions',
    '_step_segment_name',
    'locate_physical_hit_state',
)
