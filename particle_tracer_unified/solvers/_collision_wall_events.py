"""Wall event records, terminal results, and contact diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np

from particle_tracer_unified.core.datamodel import WallPartModel

from ._collision_types import (
    WallEventParticleMetadata,
    WallHitStepResult,
    WallHitTiming,
)
from .diagnostics import increment_count, increment_named_count

_CONTACT_REFLECTED_OUTCOMES = frozenset({"reflected_specular", "reflected_diffuse"})
_RUN_NAME = "run"


class _ParticleIdSource(Protocol):
    particle_id: np.ndarray


def time_tolerance(
    reference_time_s: float, interval_s: float, relative_fraction: float
) -> float:
    """Resolve a time tolerance without an absolute seconds floor."""

    interval = max(float(interval_s), 0.0)
    if interval <= 0.0:
        return 0.0
    magnitude = max(abs(float(reference_time_s)), interval)
    roundoff = 64.0 * abs(float(np.spacing(np.float64(magnitude))))
    return float(max(float(relative_fraction) * interval, roundoff))


def append_max_hit_event(
    *,
    max_hit_rows: list[dict[str, object]],
    t: float,
    particle_id: int,
    hit_count: int,
    remaining_dt: float,
    hit_part_ids: list[int],
    hit_outcomes: list[str],
) -> None:
    max_hit_rows.append(
        {
            "time_s": float(t),
            "particle_id": int(particle_id),
            "step_name": _RUN_NAME,
            "segment_name": _RUN_NAME,
            "hits_in_step": int(hit_count),
            "remaining_dt_s": float(remaining_dt),
            "last_part_id": int(hit_part_ids[-1]) if hit_part_ids else 0,
            "part_id_sequence": "|".join(str(int(pid)) for pid in hit_part_ids),
            "outcome_sequence": "|".join(hit_outcomes),
        }
    )


def wall_event_row(
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
    outcome: str,
    wall_model: WallPartModel,
    alpha_hit: float,
    primitive_id: int = -1,
    primitive_kind: str = "unknown",
    is_ambiguous: bool = False,
) -> dict[str, object]:
    hit_arr = np.asarray(hit, dtype=np.float64)
    normal_arr = np.asarray(normal, dtype=np.float64)
    velocity_arr = np.asarray(v_hit, dtype=np.float64)
    speed = float(np.linalg.norm(velocity_arr))
    normal_speed = 0.0
    tangential_speed = 0.0
    incidence_angle_deg = 0.0
    if (
        normal_arr.size == velocity_arr.size
        and float(np.linalg.norm(normal_arr)) > 1.0e-30
    ):
        n_unit = normal_arr / max(float(np.linalg.norm(normal_arr)), 1.0e-30)
        vn_signed = float(np.dot(velocity_arr, n_unit))
        normal_speed = abs(vn_signed)
        tangential = velocity_arr - vn_signed * n_unit
        tangential_speed = float(np.linalg.norm(tangential))
        incidence_angle_deg = math.degrees(
            math.atan2(tangential_speed, max(normal_speed, 1.0e-30))
        )
    hit_time_s = (
        float(t_step_end)
        - max(0.0, float(segment_dt))
        + float(np.clip(hit_dt, 0.0, max(0.0, float(segment_dt))))
    )
    row: dict[str, object] = {
        "time_s": float(t_step_end),
        "hit_time_s": float(hit_time_s),
        "particle_id": int(particle_id),
        "part_id": int(part_id),
        "boundary_primitive_id": int(primitive_id),
        "boundary_primitive_kind": str(primitive_kind),
        "boundary_hit_ambiguous": int(bool(is_ambiguous)),
        "step_name": _RUN_NAME,
        "segment_name": _RUN_NAME,
        "outcome": outcome,
        "wall_mode": wall_model.law_name,
        "alpha_hit": float(alpha_hit),
        "material_id": int(wall_model.material_id),
        "material_name": wall_model.material_name,
        "particle_mass_kg": float(particle_mass_kg),
        "particle_diameter_m": float(particle_diameter_m),
        "impact_speed_mps": float(speed),
        "impact_normal_speed_mps": float(normal_speed),
        "impact_tangential_speed_mps": float(tangential_speed),
        "impact_angle_deg_from_normal": float(incidence_angle_deg),
    }
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        row[f"hit_{axis_name}_m"] = (
            float(hit_arr[axis_idx]) if axis_idx < hit_arr.size else float("nan")
        )
        row[f"normal_{axis_name}"] = (
            float(normal_arr[axis_idx]) if axis_idx < normal_arr.size else float("nan")
        )
        row[f"v_hit_{axis_name}_mps"] = (
            float(velocity_arr[axis_idx])
            if axis_idx < velocity_arr.size
            else float("nan")
        )
    return row


def particle_scalar_or_nan(particles, name: str, particle_index: int) -> float:
    values = getattr(particles, name, None)
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if int(particle_index) >= arr.size:
        return float("nan")
    return float(arr[int(particle_index)])


def wall_event_particle_metadata(
    *,
    particles: object,
    particle_index: int,
    particle_id: int | None,
    particle_mass_kg: float | None,
    particle_diameter_m: float | None,
) -> WallEventParticleMetadata:
    return WallEventParticleMetadata(
        particle_id=int(
            cast(_ParticleIdSource, particles).particle_id[particle_index]
            if particle_id is None
            else int(particle_id)
        ),
        mass_kg=(
            particle_scalar_or_nan(particles, "mass", particle_index)
            if particle_mass_kg is None
            else float(particle_mass_kg)
        ),
        diameter_m=(
            particle_scalar_or_nan(particles, "diameter", particle_index)
            if particle_diameter_m is None
            else float(particle_diameter_m)
        ),
    )


def record_ambiguous_wall_hit(
    *,
    is_ambiguous: bool,
    part_id: int,
    primitive_kind: str,
    wall_model: WallPartModel,
    collision_diagnostics: dict[str, object],
) -> None:
    if not bool(is_ambiguous) or not bool(
        getattr(collision_diagnostics, "debug", True)
    ):
        return
    increment_count(collision_diagnostics, "boundary_ambiguous_hit_count")
    increment_named_count(
        collision_diagnostics,
        "boundary_ambiguous_part_counts",
        f"part={int(part_id)}",
    )
    increment_named_count(
        collision_diagnostics,
        "boundary_ambiguous_wall_law_counts",
        str(wall_model.law_name),
    )
    increment_named_count(
        collision_diagnostics,
        "boundary_ambiguous_primitive_kind_counts",
        str(primitive_kind),
    )


def wall_hit_timing(*, t: float, segment_dt: float, hit_dt: float) -> WallHitTiming:
    positive_segment_dt = max(0.0, float(segment_dt))
    clamped_hit_dt = float(np.clip(hit_dt, 0.0, positive_segment_dt))
    minimum_progress_dt = 0.0
    if positive_segment_dt > 0.0:
        minimum_progress_dt = min(
            positive_segment_dt,
            time_tolerance(float(t), positive_segment_dt, 1.0e-8),
        )
    consumed_dt = (
        clamped_hit_dt if clamped_hit_dt > minimum_progress_dt else minimum_progress_dt
    )
    consumed_dt = min(consumed_dt, positive_segment_dt)
    alpha = (
        0.0
        if positive_segment_dt <= 0.0
        else float(np.clip(consumed_dt / positive_segment_dt, 0.0, 1.0))
    )
    return WallHitTiming(clamped_hit_dt, consumed_dt, alpha)


def terminal_wall_hit_result(
    *,
    outcome: str,
    position: np.ndarray,
    hit_velocity: np.ndarray,
    remaining_dt: float,
    hit_count: int,
    total_hit_count: int,
    particle_index: int,
    stuck: np.ndarray,
    frozen: np.ndarray,
    absorbed: np.ndarray,
    escaped: np.ndarray,
    active: np.ndarray,
) -> WallHitStepResult | None:
    terminal_mask = {
        "stuck": stuck,
        "frozen": frozen,
        "absorbed": absorbed,
        "escaped": escaped,
    }.get(str(outcome))
    if terminal_mask is None:
        return None
    terminal_mask[particle_index] = True
    active[particle_index] = False
    return WallHitStepResult(
        np.asarray(position, dtype=np.float64),
        np.asarray(hit_velocity, dtype=np.float64).copy(),
        float(remaining_dt),
        int(hit_count),
        int(total_hit_count),
        True,
    )


def passed_through_wall_hit_result(
    *,
    outcome: str,
    hit: np.ndarray,
    response_velocity: np.ndarray,
    remaining_dt: float,
    hit_count: int,
    total_hit_count: int,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
) -> WallHitStepResult | None:
    if str(outcome) != "passed_through":
        return None
    velocity = np.asarray(response_velocity, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    position = np.asarray(hit, dtype=np.float64).copy()
    if speed > 1.0e-30:
        clearance = max(float(epsilon_offset_m), float(on_boundary_tol_m))
        position = position + clearance * velocity / speed
    return WallHitStepResult(
        position,
        velocity,
        float(remaining_dt),
        int(hit_count),
        int(total_hit_count),
        False,
    )


def _has_same_wall_reflection_history(
    *,
    hit_part_ids: list[int],
    hit_outcomes: list[str],
) -> bool:
    if len(hit_part_ids) < 2:
        return False
    if len({int(part_id) for part_id in hit_part_ids}) != 1:
        return False
    return not any(
        str(outcome) not in _CONTACT_REFLECTED_OUTCOMES for outcome in hit_outcomes
    )


def _contact_frame(
    *,
    normal: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    normalized = np.asarray(normal, dtype=np.float64)
    magnitude = float(np.linalg.norm(normalized))
    if magnitude <= 1.0e-30:
        return None
    normalized = normalized / magnitude
    response_velocity = np.asarray(velocity, dtype=np.float64)
    if response_velocity.size != normalized.size:
        return None
    return normalized, response_velocity


def _tangent_contact_velocity(
    *,
    velocity: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    tangent = velocity - float(np.dot(velocity, normal)) * normal
    if float(np.linalg.norm(tangent)) <= 1.0e-14:
        return np.zeros_like(velocity)
    return tangent


def _record_same_wall_contact_diagnostics(
    *,
    collision_diagnostics: dict[str, object],
    remaining_dt: float,
    part_id: int,
    outcome: str | None,
) -> None:
    if not bool(getattr(collision_diagnostics, "debug", True)):
        return
    increment_count(collision_diagnostics, "contact_sliding_count")
    increment_count(collision_diagnostics, "contact_sliding_same_wall_count")
    accepted_remaining_dt = float(max(0.0, remaining_dt))
    collision_diagnostics["contact_sliding_time_total_s"] = (
        float(
            cast(
                float | int,
                collision_diagnostics.get("contact_sliding_time_total_s", 0.0),
            )
        )
        + accepted_remaining_dt
    )
    collision_diagnostics["contact_sliding_remaining_dt_max_s"] = max(
        float(
            cast(
                float | int,
                collision_diagnostics.get("contact_sliding_remaining_dt_max_s", 0.0),
            )
        ),
        accepted_remaining_dt,
    )
    increment_named_count(
        collision_diagnostics,
        "contact_sliding_part_counts",
        f"part={int(part_id)}",
    )
    if outcome is not None:
        increment_named_count(
            collision_diagnostics,
            "contact_sliding_outcome_counts",
            str(outcome),
        )


def same_wall_contact_sliding_state(
    *,
    x_wall: np.ndarray,
    v_ref: np.ndarray,
    n_wall: np.ndarray,
    remaining_dt: float,
    hit_part_ids: list[int],
    hit_outcomes: list[str],
    collision_diagnostics: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not _has_same_wall_reflection_history(
        hit_part_ids=hit_part_ids,
        hit_outcomes=hit_outcomes,
    ):
        return None
    frame = _contact_frame(normal=n_wall, velocity=v_ref)
    if frame is None:
        return None
    normal, response_velocity = frame
    tangent_velocity = _tangent_contact_velocity(
        velocity=response_velocity,
        normal=normal,
    )
    _record_same_wall_contact_diagnostics(
        collision_diagnostics=collision_diagnostics,
        remaining_dt=float(remaining_dt),
        part_id=int(hit_part_ids[-1]),
        outcome=str(hit_outcomes[-1]) if hit_outcomes else None,
    )
    return np.asarray(x_wall, dtype=np.float64), tangent_velocity, normal


def record_max_hit_diagnostics(
    *,
    collision_diagnostics: dict[str, object],
    hit_part_ids: list[int],
    hit_outcomes: list[str],
    remaining_dt: float,
) -> None:
    if not hit_part_ids:
        return
    unique_parts = {int(pid) for pid in hit_part_ids}
    if len(unique_parts) <= 1:
        increment_count(collision_diagnostics, "max_hit_same_wall_count")
    else:
        increment_count(collision_diagnostics, "max_hit_multi_wall_count")
    increment_named_count(
        collision_diagnostics,
        "max_hit_last_part_counts",
        f"part={int(hit_part_ids[-1])}",
    )
    if hit_outcomes:
        increment_named_count(
            collision_diagnostics, "max_hit_last_outcome_counts", str(hit_outcomes[-1])
        )
    collision_diagnostics["max_hit_remaining_dt_total_s"] = float(
        cast(
            float | int,
            collision_diagnostics.get("max_hit_remaining_dt_total_s", 0.0),
        )
    ) + float(max(0.0, remaining_dt))
    collision_diagnostics["max_hit_remaining_dt_max_s"] = max(
        float(
            cast(
                float | int,
                collision_diagnostics.get("max_hit_remaining_dt_max_s", 0.0),
            )
        ),
        float(max(0.0, remaining_dt)),
    )


def post_wall_acceptance_reason(
    *,
    runtime,
    position: np.ndarray,
    velocity: np.ndarray,
    inside_fn: Callable[[np.ndarray], bool],
) -> str:
    pos = np.asarray(position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
        return "post_wall_nonfinite_state"
    try:
        if not bool(inside_fn(pos)):
            return "post_wall_outside_geometry"
    except Exception:
        return "post_wall_geometry_check_failed"
    return ""
