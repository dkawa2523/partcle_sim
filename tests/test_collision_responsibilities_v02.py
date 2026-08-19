from __future__ import annotations

import numpy as np
import pytest

from particle_tracer_unified.domain import BoundaryHit
from particle_tracer_unified.solvers.collision_hit_localization import (
    locate_physical_hit_state,
)


def _time_tolerance(
    reference_time_s: float, interval_s: float, fraction: float
) -> float:
    magnitude = max(abs(float(reference_time_s)), abs(float(interval_s)))
    roundoff = 64.0 * abs(float(np.spacing(np.float64(magnitude))))
    return max(float(fraction) * float(interval_s), roundoff)


def _ballistic_state(partial_time: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([2.0 * partial_time, 0.0], dtype=np.float64),
        np.asarray([2.0, 0.0], dtype=np.float64),
    )


def test_primary_hit_localization_uses_the_segment_state_evaluator() -> None:
    primary = BoundaryHit(
        position=np.asarray([1.0, 0.0], dtype=np.float64),
        normal=np.asarray([1.0, 0.0], dtype=np.float64),
        part_id=7,
        alpha_hint=0.5,
        primitive_id=3,
        primitive_kind="edge",
    )

    event = locate_physical_hit_state(
        x0=np.asarray([0.0, 0.0], dtype=np.float64),
        v0=np.asarray([2.0, 0.0], dtype=np.float64),
        segment_dt=1.0,
        t_end_segment=1.0,
        stage_times=np.asarray([0.5, 1.0], dtype=np.float64),
        primary_hit=primary,
        strict_inside_fn=lambda point: bool(point[0] < 1.0),
        nearest_projection_fn=lambda _point, _inside: primary,
        state_at=_ballistic_state,
        time_tolerance=_time_tolerance,
        on_boundary_tol_m=1.0e-12,
    )

    assert event is not None
    hit, velocity, hit_time = event
    assert hit_time == pytest.approx(0.5)
    assert hit.position.tolist() == pytest.approx([1.0, 0.0])
    assert hit.part_id == 7
    assert velocity.tolist() == pytest.approx([2.0, 0.0])


def test_primary_plane_crossing_outside_the_finite_primitive_is_rejected() -> None:
    primary = BoundaryHit(
        position=np.asarray([1.0, 0.0], dtype=np.float64),
        normal=np.asarray([1.0, 0.0], dtype=np.float64),
        part_id=7,
        alpha_hint=0.5,
        primitive_id=3,
        primitive_kind="edge",
    )

    event = locate_physical_hit_state(
        x0=np.asarray([0.0, 5.0], dtype=np.float64),
        v0=np.asarray([2.0, 0.0], dtype=np.float64),
        segment_dt=1.0,
        t_end_segment=1.0,
        stage_times=np.asarray([0.5, 1.0], dtype=np.float64),
        primary_hit=primary,
        strict_inside_fn=lambda _point: True,
        nearest_projection_fn=lambda _point, _inside: primary,
        state_at=lambda partial_time: (
            np.asarray([2.0 * partial_time, 5.0], dtype=np.float64),
            np.asarray([2.0, 0.0], dtype=np.float64),
        ),
        time_tolerance=_time_tolerance,
        on_boundary_tol_m=1.0e-12,
    )

    assert event is None


def test_geometric_bisection_localization_does_not_own_solver_state() -> None:
    projected = BoundaryHit(
        position=np.asarray([0.6, 0.0], dtype=np.float64),
        normal=np.asarray([1.0, 0.0], dtype=np.float64),
        part_id=9,
        alpha_hint=0.0,
        primitive_id=5,
        primitive_kind="edge",
    )

    event = locate_physical_hit_state(
        x0=np.asarray([0.0, 0.0], dtype=np.float64),
        v0=np.asarray([1.0, 0.0], dtype=np.float64),
        segment_dt=1.0,
        t_end_segment=1.0,
        stage_times=np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        primary_hit=None,
        strict_inside_fn=lambda point: bool(point[0] < 0.6),
        nearest_projection_fn=lambda _point, _inside: projected,
        state_at=lambda partial_time: (
            np.asarray([partial_time, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0], dtype=np.float64),
        ),
        time_tolerance=_time_tolerance,
        on_boundary_tol_m=1.0e-12,
    )

    assert event is not None
    hit, velocity, hit_time = event
    assert hit_time == pytest.approx(0.6, abs=1.0e-6)
    assert hit.position.tolist() == pytest.approx([0.6, 0.0])
    assert hit.part_id == 9
    assert velocity.tolist() == pytest.approx([1.0, 0.0])
