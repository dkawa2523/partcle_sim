"""Apply wall responses and record collision events for one resolved hit."""

from __future__ import annotations

from typing import cast

import numpy as np

from particle_tracer_unified.core.boundary_service import (
    inside_geometry as _boundary_inside_geometry,
)
from particle_tracer_unified.core.catalogs import resolve_step_wall_model
from particle_tracer_unified.core.geometry3d import TriangleSurface3D

from . import _collision_wall_events
from ._collision_types import (
    CollidingParticleAdvanceResult as CollidingParticleAdvanceResult,
)
from ._collision_types import (
    CollisionSegmentInputs as CollisionSegmentInputs,
)
from ._collision_types import (
    OrientedWallHitState,
    WallHitStepResult,
)
from ._collision_wall_events import (
    append_max_hit_event as _append_max_hit_event,
)
from ._collision_wall_events import (
    passed_through_wall_hit_result,
    record_ambiguous_wall_hit,
    same_wall_contact_sliding_state,
)
from ._collision_wall_events import (
    terminal_wall_hit_result as _terminal_wall_hit_result,
)
from ._collision_wall_events import (
    wall_event_particle_metadata as _wall_event_particle_metadata,
)
from ._collision_wall_events import wall_event_row as _wall_event_row
from ._collision_wall_events import wall_hit_timing as _wall_hit_timing
from ._stochastic_randomness import WallRandomContext
from .diagnostics import increment_count
from .wall_response import _apply_keyed_wall_response, _unit_vector
from .wall_response import apply_wall_response as _apply_wall_response


def _oriented_wall_hit_state(
    *,
    runtime,
    hit: np.ndarray,
    n_out: np.ndarray,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
    triangle_surface_3d: TriangleSurface3D | None,
) -> OrientedWallHitState:
    hit_arr = np.asarray(hit, dtype=np.float64)
    n_wall = _unit_vector(n_out, context="wall-hit normal")
    push = float(epsilon_offset_m)

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
    x_minus = hit_arr - push * n_wall
    x_plus = hit_arr + push * n_wall
    if _candidate_inside(x_minus, 0.0) or _candidate_inside(
        x_minus, float(on_boundary_tol_m)
    ):
        x_wall = x_minus
    elif _candidate_inside(x_plus, 0.0) or _candidate_inside(
        x_plus, float(on_boundary_tol_m)
    ):
        x_wall = x_plus
        n_wall = -n_wall
    elif _candidate_inside(hit_arr, float(on_boundary_tol_m)):
        x_wall = hit_arr.copy()
    else:
        x_wall = hit_arr - push * n_wall
    return OrientedWallHitState(hit=hit_arr, normal=n_wall, wall_position=x_wall)


def _max_hit_wall_result(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    normal: np.ndarray,
    remaining_dt: float,
    hit_count: int,
    total_hit_count: int,
    part_id: int,
    primitive_id: int,
    particle_id: int,
    max_wall_hits_per_step: int,
    hit_part_ids: list[int],
    hit_outcomes: list[str],
    collision_diagnostics: dict[str, object],
    max_hit_rows: list[dict[str, object]] | None,
    t: float,
    allow_contact_sliding: bool = True,
) -> WallHitStepResult | None:
    if int(hit_count) < int(max_wall_hits_per_step):
        return None
    if float(remaining_dt) <= 0.0:
        return WallHitStepResult(
            position, velocity, remaining_dt, hit_count, total_hit_count, True
        )
    contact_state = None
    if bool(allow_contact_sliding):
        contact_state = same_wall_contact_sliding_state(
            x_wall=position,
            v_ref=velocity,
            n_wall=normal,
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
    increment_count(collision_diagnostics, "max_hits_reached_count")
    if bool(getattr(collision_diagnostics, "debug", True)):
        _collision_wall_events.record_max_hit_diagnostics(
            collision_diagnostics=collision_diagnostics,
            hit_part_ids=hit_part_ids,
            hit_outcomes=hit_outcomes,
            remaining_dt=float(remaining_dt),
        )
    if max_hit_rows is not None:
        _append_max_hit_event(
            max_hit_rows=max_hit_rows,
            t=float(t),
            particle_id=int(particle_id),
            hit_count=int(hit_count),
            remaining_dt=float(remaining_dt),
            hit_part_ids=hit_part_ids,
            hit_outcomes=hit_outcomes,
        )
    return WallHitStepResult(
        position, velocity, remaining_dt, hit_count, total_hit_count, True
    )


def apply_wall_hit_step(
    *,
    runtime,
    particles,
    particle_index: int,
    particle_id: int | None = None,
    particle_mass_kg: float | None = None,
    particle_diameter_m: float | None = None,
    rng: np.random.Generator | None,
    wall_random_context: WallRandomContext | None = None,
    hit: np.ndarray,
    n_out: np.ndarray,
    hit_dt: float,
    part_id: int,
    primitive_id: int = -1,
    primitive_kind: str = "unknown",
    is_ambiguous: bool = False,
    v_hit: np.ndarray,
    remaining_dt: float,
    segment_dt: float,
    hit_count: int,
    total_hit_count: int,
    hit_part_ids: list[int],
    hit_outcomes: list[str],
    collision_diagnostics: dict[str, object],
    max_hit_rows: list[dict[str, object]] | None,
    wall_rows: list[dict[str, object]] | None,
    wall_summary_counts: dict[tuple[int, str, str], int],
    stuck: np.ndarray,
    frozen: np.ndarray | None = None,
    absorbed: np.ndarray,
    escaped: np.ndarray | None = None,
    active: np.ndarray,
    max_wall_hits_per_step: int,
    epsilon_offset_m: float,
    on_boundary_tol_m: float,
    t: float,
    triangle_surface_3d: TriangleSurface3D | None,
    allow_contact_sliding: bool = True,
) -> WallHitStepResult:
    if escaped is None:
        escaped = np.zeros_like(active, dtype=bool)
    if frozen is None:
        frozen = np.zeros_like(active, dtype=bool)
    oriented = _oriented_wall_hit_state(
        runtime=runtime,
        hit=hit,
        n_out=n_out,
        epsilon_offset_m=float(epsilon_offset_m),
        on_boundary_tol_m=float(on_boundary_tol_m),
        triangle_surface_3d=triangle_surface_3d,
    )
    hit_arr = np.asarray(oriented.hit, dtype=np.float64)
    n_wall = np.asarray(oriented.normal, dtype=np.float64)
    x_wall = np.asarray(oriented.wall_position, dtype=np.float64)

    wall_model = resolve_step_wall_model(runtime.wall_catalog, part_id)
    record_ambiguous_wall_hit(
        is_ambiguous=bool(is_ambiguous),
        part_id=int(part_id),
        primitive_kind=str(primitive_kind),
        wall_model=wall_model,
        collision_diagnostics=collision_diagnostics,
    )
    metadata = _wall_event_particle_metadata(
        particles=particles,
        particle_index=int(particle_index),
        particle_id=particle_id,
        particle_mass_kg=particle_mass_kg,
        particle_diameter_m=particle_diameter_m,
    )
    response = (
        _apply_wall_response(
            cast(np.random.Generator, rng),
            v_hit,
            n_wall,
            wall_model,
        )
        if wall_random_context is None
        else _apply_keyed_wall_response(
            wall_random_context,
            v_hit,
            n_wall,
            wall_model,
        )
    )
    outcome, v_ref = response
    summary_key = (int(part_id), str(outcome), str(wall_model.law_name))
    wall_summary_counts[summary_key] = wall_summary_counts.get(summary_key, 0) + 1

    timing = _wall_hit_timing(t=float(t), segment_dt=float(segment_dt), hit_dt=hit_dt)
    if wall_rows is not None:
        wall_rows.append(
            _wall_event_row(
                t_step_end=float(t),
                segment_dt=float(segment_dt),
                hit_dt=float(timing.clamped_hit_dt),
                particle_id=int(metadata.particle_id),
                particle_mass_kg=float(metadata.mass_kg),
                particle_diameter_m=float(metadata.diameter_m),
                hit=hit_arr,
                normal=n_wall,
                v_hit=np.asarray(v_hit, dtype=np.float64),
                part_id=int(part_id),
                outcome=outcome,
                wall_model=wall_model,
                alpha_hit=float(timing.alpha),
                primitive_id=int(primitive_id),
                primitive_kind=str(primitive_kind),
                is_ambiguous=bool(is_ambiguous),
            )
        )

    hit_count += 1
    total_hit_count += 1
    hit_part_ids.append(int(part_id))
    hit_outcomes.append(str(outcome))
    remaining_dt = max(0.0, float(remaining_dt) - float(timing.consumed_dt))

    terminal_result = _terminal_wall_hit_result(
        outcome=str(outcome),
        position=(
            hit_arr
            if outcome == "escaped" and wall_model.law_name == "pass_through"
            else x_wall
        ),
        hit_velocity=v_ref,
        remaining_dt=float(remaining_dt),
        hit_count=int(hit_count),
        total_hit_count=int(total_hit_count),
        particle_index=int(particle_index),
        stuck=stuck,
        frozen=frozen,
        absorbed=absorbed,
        escaped=escaped,
        active=active,
    )
    if terminal_result is not None:
        return terminal_result

    pass_through_result = passed_through_wall_hit_result(
        outcome=str(outcome),
        hit=hit_arr,
        response_velocity=v_ref,
        remaining_dt=float(remaining_dt),
        hit_count=int(hit_count),
        total_hit_count=int(total_hit_count),
        epsilon_offset_m=float(epsilon_offset_m),
        on_boundary_tol_m=float(on_boundary_tol_m),
    )
    if pass_through_result is not None:
        return pass_through_result

    x_curr_next = x_wall
    v_curr_next = np.asarray(v_ref, dtype=np.float64)
    max_hit_result = _max_hit_wall_result(
        position=x_curr_next,
        velocity=v_curr_next,
        normal=n_wall,
        remaining_dt=float(remaining_dt),
        hit_count=int(hit_count),
        total_hit_count=int(total_hit_count),
        part_id=int(part_id),
        primitive_id=int(primitive_id),
        particle_id=int(metadata.particle_id),
        max_wall_hits_per_step=int(max_wall_hits_per_step),
        hit_part_ids=hit_part_ids,
        hit_outcomes=hit_outcomes,
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=max_hit_rows,
        t=float(t),
        allow_contact_sliding=bool(allow_contact_sliding),
    )
    if max_hit_result is not None:
        return max_hit_result
    return WallHitStepResult(
        x_curr_next, v_curr_next, remaining_dt, hit_count, total_hit_count, False
    )
