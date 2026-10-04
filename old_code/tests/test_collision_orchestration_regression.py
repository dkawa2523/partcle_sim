from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import WallPartModel
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from particle_tracer_unified.domain import BoundaryHit
from particle_tracer_unified.solvers import _collision_particle as particle_collision
from particle_tracer_unified.solvers import _collision_trial as collision_trial
from particle_tracer_unified.solvers import _collision_types as collision_types
from particle_tracer_unified.solvers import _collision_wall_events as wall_events
from particle_tracer_unified.solvers import high_fidelity_collision as collision
from particle_tracer_unified.solvers import high_fidelity_runtime as runtime
from particle_tracer_unified.solvers.terminal_outcome import terminal_segment_outcome


def _state() -> collision_types._CollisionAdvanceState:
    return particle_collision.initial_collision_advance_state(
        x_start=np.asarray([0.25, 0.5]),
        v_start=np.asarray([1.0, 0.0]),
        dt_step=1.0,
        valid_mask_status=0,
    )


def _search_context() -> collision_types._CollisionSearchContext:
    def no_hit(_start: np.ndarray, _points: np.ndarray):
        return None

    def inside(_point: np.ndarray) -> bool:
        return True

    return collision_types._CollisionSearchContext(
        t=1.0,
        dt_step=1.0,
        base_adaptive_substep_enabled=0,
        initial_x_next=np.asarray([1.25, 0.5]),
        initial_v_next=np.asarray([1.0, 0.0]),
        initial_stage_points=np.asarray([[1.25, 0.5]]),
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=inside,
        strict_inside_fn=inside,
        primary_hit_fn=no_hit,
        nearest_projection_fn=no_hit,
        primary_hit_counter_key="edge_hit_count",
        collision_diagnostics={},
        on_boundary_tol_m=1.0e-10,
    )


def _wall_context() -> collision_types._WallInteractionContext:
    return collision_types._WallInteractionContext(
        runtime=object(),
        particles=object(),
        particle_index=0,
        particle_id=1,
        particle_mass_kg=1.0e-15,
        particle_diameter_m=1.0e-6,
        rng=np.random.default_rng(7),
        collision_diagnostics={},
        max_hit_rows=None,
        wall_rows=None,
        wall_summary_counts={},
        stuck=np.asarray([False]),
        frozen=np.asarray([False]),
        absorbed=np.asarray([False]),
        escaped=np.asarray([False]),
        active=np.asarray([True]),
        max_wall_hits_per_step=2,
        epsilon_offset_m=1.0e-8,
        on_boundary_tol_m=1.0e-10,
        t=1.0,
        triangle_surface_3d=None,
    )


def test_adaptive_substep_diagnostics_accumulate_active_particles_only() -> None:
    diagnostics: dict[str, object] = {
        "adaptive_substep_segments_count": 5,
        "adaptive_substep_trigger_count": 2,
        "adaptive_substep_limit_reached_count": 0,
    }

    runtime._update_adaptive_substep_diagnostics(
        diagnostics,
        adaptive_substep_enabled=1,
        adaptive_substep_max_splits=4,
        active=np.asarray([True, False, True]),
        substep_counts=np.asarray([2, 8, 1]),
    )

    assert diagnostics == {
        "adaptive_substep_segments_count": 8,
        "adaptive_substep_trigger_count": 3,
        "adaptive_substep_limit_reached_count": 0,
    }


@pytest.mark.parametrize(
    ("enabled", "active", "substep_counts", "expected"),
    [
        (0, [True], [8], 0),
        (1, [False], [8], 0),
        (1, [True], [4], 0),
        (1, [True, True, False], [8, 8, 8], 2),
    ],
)
def test_adaptive_substep_limit_diagnostic_counts_exact_active_limit(
    enabled: int,
    active: list[bool],
    substep_counts: list[int],
    expected: int,
) -> None:
    diagnostics: dict[str, object] = {
        "adaptive_substep_segments_count": 0,
        "adaptive_substep_trigger_count": 0,
        "adaptive_substep_limit_reached_count": 0,
    }

    runtime._update_adaptive_substep_diagnostics(
        diagnostics,
        adaptive_substep_enabled=enabled,
        adaptive_substep_max_splits=3,
        active=np.asarray(active),
        substep_counts=np.asarray(substep_counts),
    )

    assert diagnostics["adaptive_substep_limit_reached_count"] == expected


def _segment_inputs(**values: object) -> collision.CollisionSegmentInputs:
    return cast(
        collision.CollisionSegmentInputs,
        SimpleNamespace(adaptive_substep_max_splits=4, **values),
    )


def test_terminal_trial_result_preserves_prior_hits_and_absolute_elapsed_time() -> None:
    state = _state()
    state.total_hit_count = 3
    relative_terminal = terminal_segment_outcome(
        accepted_elapsed_s=0.2,
        segment_duration_s=0.5,
        position=np.asarray([0.45, 0.5]),
        reason="collision_valid_mask_hard_invalid_prefix_clipped",
    )
    invalid_stop = collision.CollidingParticleAdvanceResult(
        position=np.asarray([0.45, 0.5]),
        velocity=np.asarray([1.0, 0.0]),
        total_hits=0,
        valid_mask_status=2,
        invalid_mask_stopped=True,
        invalid_stop_reason="collision_valid_mask_hard_invalid_prefix_clipped",
        terminal_outcome=relative_terminal,
    )

    result = particle_collision._terminal_trial_advance_result(
        state=state,
        stop=invalid_stop,
        segment_dt=0.5,
        dt_step=1.0,
    )

    assert result.total_hits == 3
    assert result.valid_mask_status == 2
    assert result.invalid_mask_stopped is True
    assert result.terminal_outcome is not None
    assert result.terminal_outcome.accepted_elapsed_s == pytest.approx(0.7)
    np.testing.assert_array_equal(result.position, [0.45, 0.5])


def test_terminal_trial_result_requires_a_terminal_outcome() -> None:
    invalid_stop = collision.CollidingParticleAdvanceResult(
        position=np.asarray([0.25, 0.5]),
        velocity=np.asarray([1.0, 0.0]),
        total_hits=0,
        valid_mask_status=2,
        invalid_mask_stopped=True,
        invalid_stop_reason="collision_valid_mask_hard_invalid_retry_exhausted",
    )

    with pytest.raises(RuntimeError, match="requires a terminal outcome"):
        particle_collision._terminal_trial_advance_result(
            state=_state(),
            stop=invalid_stop,
            segment_dt=1.0,
            dt_step=1.0,
        )


@pytest.mark.parametrize(
    ("position", "velocity", "inside_fn", "expected"),
    [
        (
            np.asarray([np.nan, 0.0]),
            np.asarray([0.0, 0.0]),
            lambda _point: True,
            "post_wall_nonfinite_state",
        ),
        (
            np.asarray([2.0, 0.0]),
            np.asarray([0.0, 0.0]),
            lambda _point: False,
            "post_wall_outside_geometry",
        ),
        (
            np.asarray([0.0, 0.0]),
            np.asarray([0.0, 0.0]),
            lambda _point: (_ for _ in ()).throw(ValueError("geometry unavailable")),
            "post_wall_geometry_check_failed",
        ),
    ],
)
def test_post_wall_acceptance_reports_the_exact_failure_reason(
    position: np.ndarray,
    velocity: np.ndarray,
    inside_fn,
    expected: str,
) -> None:
    assert (
        wall_events.post_wall_acceptance_reason(
            runtime=object(),
            position=position,
            velocity=velocity,
            inside_fn=inside_fn,
        )
        == expected
    )


def test_invalid_post_wall_state_takes_precedence_over_max_hits() -> None:
    state = _state()
    state.position = np.asarray([np.nan, 0.5])
    state.hit_count = 2
    wall_result = collision.WallHitStepResult(
        position=state.position,
        velocity=state.velocity,
        remaining_dt=state.remaining_dt,
        hit_count=state.hit_count,
        total_hit_count=state.total_hit_count,
        should_break=True,
    )

    reason = particle_collision._post_wall_stop_reason(
        state=state,
        wall_result=wall_result,
        context=_wall_context(),
        inside_fn=lambda _point: True,
    )

    assert reason == "post_wall_nonfinite_state"


def test_unresolved_segment_stops_without_committing_trial_endpoint() -> None:
    state = _state()
    initial_position = state.position.copy()
    resolution = collision_types.CollisionSegmentResolution(
        advance_without_hit=False,
        should_break=True,
        x_next=np.asarray([1.25, 0.5]),
        v_next=np.asarray([1.0, 0.0]),
    )

    should_continue = particle_collision._advance_resolved_segment(
        state=state,
        resolution=resolution,
        segment_dt=1.0,
        search=_search_context(),
        wall=_wall_context(),
    )

    assert should_continue is False
    np.testing.assert_array_equal(state.position, initial_position)


def test_particle_segment_loop_resolves_before_finishing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    trial = collision_types.CollisionSegmentTrial(
        x_next=np.asarray([0.75, 0.5]),
        v_next=np.asarray([1.0, 0.0]),
        stage_points=np.asarray([[0.75, 0.5]]),
        primary_hit=None,
        primary_hit_counted=False,
        particle_valid_mask_status=0,
    )
    resolution = collision_types.CollisionSegmentResolution(
        advance_without_hit=True,
        should_break=False,
        x_next=trial.x_next,
        v_next=trial.v_next,
    )
    expected = collision.CollidingParticleAdvanceResult(
        position=trial.x_next,
        velocity=trial.v_next,
        total_hits=0,
        valid_mask_status=0,
        invalid_mask_stopped=False,
    )

    def prepare(**_kwargs):
        calls.append("prepare")
        return 1.0, _segment_inputs(), trial

    def resolve(**_kwargs):
        calls.append("resolve")
        return resolution

    def advance(**_kwargs):
        calls.append("advance")
        return False

    def finish(**_kwargs):
        calls.append("finish")
        return expected

    monkeypatch.setattr(particle_collision, "_prepare_state_segment_trial", prepare)
    monkeypatch.setattr(particle_collision, "_resolve_state_segment", resolve)
    monkeypatch.setattr(particle_collision, "_advance_resolved_segment", advance)
    monkeypatch.setattr(particle_collision, "_finish_collision_advance", finish)

    result = particle_collision._advance_collision_segments(
        state=_state(),
        base_inputs=_segment_inputs(),
        search=_search_context(),
        wall=_wall_context(),
    )

    assert result is expected
    assert calls == ["prepare", "resolve", "advance", "finish"]


def test_particle_segment_loop_returns_a_terminal_trial_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "trace_refinement_unresolved"
    stop = collision.CollidingParticleAdvanceResult(
        position=np.asarray([0.25, 0.5]),
        velocity=np.asarray([1.0, 0.0]),
        total_hits=0,
        valid_mask_status=0,
        invalid_mask_stopped=False,
        numerical_boundary_stopped=True,
        numerical_boundary_stop_reason=reason,
        terminal_outcome=terminal_segment_outcome(
            accepted_elapsed_s=0.0,
            segment_duration_s=0.5,
            position=np.asarray([0.25, 0.5]),
            reason=reason,
        ),
    )
    trial = collision_types.CollisionSegmentTrial(
        x_next=np.asarray([0.75, 0.5]),
        v_next=np.asarray([2.0, 0.0]),
        stage_points=np.asarray([[0.75, 0.5]]),
        primary_hit=None,
        primary_hit_counted=False,
        particle_valid_mask_status=0,
        terminal_stop_result=stop,
    )
    monkeypatch.setattr(
        particle_collision,
        "_prepare_state_segment_trial",
        lambda **_kwargs: (0.5, _segment_inputs(), trial),
    )
    monkeypatch.setattr(
        particle_collision,
        "_resolve_state_segment",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a terminal trial was sent to collision resolution")
        ),
    )

    result = particle_collision._advance_collision_segments(
        state=_state(),
        base_inputs=_segment_inputs(),
        search=_search_context(),
        wall=_wall_context(),
    )

    assert result.numerical_boundary_stopped is True
    assert result.numerical_boundary_stop_reason == reason
    assert result.invalid_mask_stopped is False
    assert result.terminal_outcome is not None
    assert result.terminal_outcome.accepted_elapsed_s == pytest.approx(0.5)


def test_particle_metadata_fallback_returns_nan_for_missing_values() -> None:
    assert np.isnan(wall_events.particle_scalar_or_nan(SimpleNamespace(), "mass", 0))
    assert np.isnan(
        wall_events.particle_scalar_or_nan(
            SimpleNamespace(mass=np.asarray([1.0])), "mass", 1
        )
    )


def test_wall_normal_flips_when_only_the_plus_offset_is_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collision,
        "_boundary_inside_geometry",
        lambda _runtime, point, **_kwargs: bool(np.asarray(point)[0] > 0.0),
    )

    oriented = collision._oriented_wall_hit_state(
        runtime=object(),
        hit=np.asarray([0.0, 0.0]),
        n_out=np.asarray([1.0, 0.0]),
        epsilon_offset_m=0.1,
        on_boundary_tol_m=0.01,
        triangle_surface_3d=None,
    )

    np.testing.assert_array_equal(oriented.wall_position, [0.1, 0.0])
    np.testing.assert_array_equal(oriented.normal, [-1.0, 0.0])


def test_ambiguous_wall_hit_records_part_law_and_primitive_metadata() -> None:
    diagnostics: dict[str, object] = {}
    wall_model = WallPartModel(
        part_id=17,
        part_name="wall",
        material_id=1,
        material_name="material",
        law_name="specular",
        stick_probability=0.0,
        restitution=1.0,
        diffuse_fraction=0.0,
        critical_sticking_velocity_mps=0.0,
    )

    wall_events.record_ambiguous_wall_hit(
        is_ambiguous=True,
        part_id=17,
        primitive_kind="edge",
        wall_model=wall_model,
        collision_diagnostics=diagnostics,
    )

    expected = {
        "boundary_ambiguous_hit_count": 1,
        "boundary_ambiguous_part_counts": {"part=17": 1},
        "boundary_ambiguous_wall_law_counts": {"specular": 1},
        "boundary_ambiguous_primitive_kind_counts": {"edge": 1},
    }
    assert all(diagnostics[key] == value for key, value in expected.items())


def test_pass_through_result_moves_outward_by_the_configured_clearance() -> None:
    result = wall_events.passed_through_wall_hit_result(
        outcome="passed_through",
        hit=np.asarray([1.0, 2.0]),
        response_velocity=np.asarray([3.0, 4.0]),
        remaining_dt=0.25,
        hit_count=1,
        total_hit_count=2,
        epsilon_offset_m=0.1,
        on_boundary_tol_m=0.01,
    )

    assert result is not None
    np.testing.assert_allclose(result.position, [1.06, 2.08], rtol=0.0, atol=1.0e-15)
    np.testing.assert_array_equal(result.velocity, [3.0, 4.0])
    assert result.remaining_dt == pytest.approx(0.25)
    assert result.should_break is False


def test_max_hit_at_segment_end_adds_no_unresolved_hit_diagnostic() -> None:
    diagnostics: dict[str, object] = {"max_hits_reached_count": 0}

    result = collision._max_hit_wall_result(
        position=np.asarray([1.0, 0.0]),
        velocity=np.asarray([-1.0, 0.0]),
        normal=np.asarray([1.0, 0.0]),
        remaining_dt=0.0,
        hit_count=2,
        total_hit_count=2,
        part_id=3,
        primitive_id=4,
        particle_id=5,
        max_wall_hits_per_step=2,
        hit_part_ids=[3, 3],
        hit_outcomes=["reflected_specular", "reflected_specular"],
        collision_diagnostics=diagnostics,
        max_hit_rows=[],
        t=1.0,
    )

    assert result is not None
    assert result.should_break is True
    assert diagnostics["max_hits_reached_count"] == 0


def test_coupled_max_hit_does_not_consume_charge_age_as_contact() -> None:
    diagnostics: dict[str, object] = {"max_hits_reached_count": 0}
    remaining_dt = 0.25

    result = collision._max_hit_wall_result(
        position=np.asarray([1.0, 0.5]),
        velocity=np.asarray([0.0, 4.0]),
        normal=np.asarray([1.0, 0.0]),
        remaining_dt=remaining_dt,
        hit_count=2,
        total_hit_count=2,
        part_id=7,
        primitive_id=4,
        particle_id=5,
        max_wall_hits_per_step=2,
        hit_part_ids=[7, 7],
        hit_outcomes=["reflected_specular", "reflected_specular"],
        collision_diagnostics=diagnostics,
        max_hit_rows=[],
        t=1.0,
        allow_contact_sliding=False,
    )

    assert result is not None
    assert result.should_break is True
    assert result.entered_contact is False
    assert result.remaining_dt == pytest.approx(remaining_dt)
    assert diagnostics["max_hits_reached_count"] == 1


def test_precomputed_segment_preserves_hit_and_skips_reintegration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hit = BoundaryHit(
        position=np.asarray([1.0, 0.5]),
        normal=np.asarray([1.0, 0.0]),
        part_id=8,
        alpha_hint=0.75,
    )
    diagnostics: dict[str, object] = {"collision_reintegrated_segments_count": 0}

    def fail_reintegration(**_kwargs):
        raise AssertionError("precomputed segment was reintegrated")

    monkeypatch.setattr(
        collision_trial, "_advance_segment_with_inputs", fail_reintegration
    )
    trial = collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=True,
        x_curr=np.asarray([0.25, 0.5]),
        v_curr=np.asarray([1.0, 0.0]),
        t=1.0,
        segment_dt=1.0,
        inputs=_segment_inputs(),
        base_adaptive_substep_enabled=0,
        initial_x_next=np.asarray([1.25, 0.5]),
        initial_v_next=np.asarray([1.0, 0.0]),
        initial_stage_points=np.asarray([[0.75, 0.5], [1.25, 0.5]]),
        initial_valid_mask_status=0,
        initial_primary_hit=hit,
        initial_primary_hit_counted=True,
        inside_fn=lambda _point: True,
        primary_hit_fn=lambda _start, _points: None,
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics=diagnostics,
    )

    assert trial.primary_hit is hit
    assert trial.primary_hit_counted is True
    assert trial.terminal_stop_result is None
    assert diagnostics["collision_reintegrated_segments_count"] == 0
    np.testing.assert_array_equal(trial.x_next, [1.25, 0.5])


def test_reintegrated_segment_records_adaptive_substeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    diagnostics: dict[str, object] = {
        "collision_reintegrated_segments_count": 0,
        "adaptive_substep_segments_count": 0,
        "adaptive_substep_trigger_count": 0,
        "adaptive_substep_limit_reached_count": 0,
    }

    def advance(**kwargs):
        calls.append(int(kwargs["minimum_substeps"]))
        return (
            np.asarray([0.75, 0.5]),
            np.asarray([1.0, 0.0]),
            2,
            np.asarray([[0.5, 0.5], [0.75, 0.5]]),
            0,
            True,
        )

    monkeypatch.setattr(collision_trial, "_advance_segment_with_inputs", advance)
    monkeypatch.setattr(
        collision_trial,
        "assess_trace_geometry",
        lambda *_args, **_kwargs: SimpleNamespace(
            requires_refinement=lambda _clearance: False
        ),
    )
    trial = collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=False,
        x_curr=np.asarray([0.25, 0.5]),
        v_curr=np.asarray([1.0, 0.0]),
        t=0.5,
        segment_dt=0.5,
        inputs=_segment_inputs(),
        base_adaptive_substep_enabled=1,
        initial_x_next=np.zeros(2),
        initial_v_next=np.zeros(2),
        initial_stage_points=np.zeros((1, 2)),
        initial_valid_mask_status=0,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=lambda _point: True,
        primary_hit_fn=lambda _start, _points: None,
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics=diagnostics,
    )

    assert calls == [2]
    assert trial.primary_hit is None
    assert trial.primary_hit_counted is False
    expected = {
        "collision_reintegrated_segments_count": 1,
        "adaptive_substep_segments_count": 2,
        "adaptive_substep_trigger_count": 1,
        "adaptive_substep_limit_reached_count": 0,
    }
    assert all(diagnostics[key] == value for key, value in expected.items())


def test_reintegrated_segment_doubles_substeps_until_curve_is_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimum_substeps: list[int] = []
    hit_queries: list[np.ndarray] = []
    diagnostics: dict[str, object] = {
        "collision_reintegrated_segments_count": 0,
        "adaptive_substep_segments_count": 0,
        "adaptive_substep_trigger_count": 0,
        "adaptive_substep_limit_reached_count": 0,
    }

    def advance(**kwargs):
        requested = int(kwargs["minimum_substeps"])
        minimum_substeps.append(requested)
        points = np.column_stack(
            (
                np.linspace(0.25, 0.75, requested),
                np.full(requested, 0.5),
            )
        )
        return points[-1], np.asarray([1.0, 0.0]), requested, points, 0, True

    def primary_hit(_start: np.ndarray, points: np.ndarray):
        hit_queries.append(np.asarray(points).copy())
        return None

    geometry_risk = iter((True, True, False))
    monkeypatch.setattr(collision_trial, "_advance_segment_with_inputs", advance)
    monkeypatch.setattr(
        collision_trial,
        "assess_trace_geometry",
        lambda *_args, **_kwargs: SimpleNamespace(
            requires_refinement=lambda _clearance: next(geometry_risk)
        ),
    )

    trial = collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=False,
        x_curr=np.asarray([0.25, 0.5]),
        v_curr=np.asarray([1.0, 0.0]),
        t=0.5,
        segment_dt=0.5,
        inputs=_segment_inputs(),
        base_adaptive_substep_enabled=1,
        initial_x_next=np.zeros(2),
        initial_v_next=np.zeros(2),
        initial_stage_points=np.zeros((1, 2)),
        initial_valid_mask_status=0,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=lambda _point: True,
        primary_hit_fn=primary_hit,
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics=diagnostics,
    )

    assert minimum_substeps == [2, 4, 8]
    assert [points.shape for points in hit_queries] == [(2, 2), (4, 2), (8, 2)]
    assert trial.stage_points.shape == (8, 2)
    expected = {
        "collision_reintegrated_segments_count": 1,
        "adaptive_substep_segments_count": 8,
        "adaptive_substep_trigger_count": 1,
        "adaptive_substep_limit_reached_count": 0,
    }
    assert all(diagnostics[key] == value for key, value in expected.items())


def test_reintegrated_segment_stops_before_events_when_local_error_is_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = np.asarray([0.25, 0.5])
    velocity = np.asarray([1.0, 0.0])
    diagnostics: dict[str, object] = {
        "collision_reintegrated_segments_count": 0,
        "adaptive_substep_segments_count": 0,
        "adaptive_substep_trigger_count": 0,
        "adaptive_substep_limit_reached_count": 0,
        "unresolved_crossing_count": 0,
    }

    monkeypatch.setattr(
        collision_trial,
        "_advance_segment_with_inputs",
        lambda **_kwargs: (
            np.asarray([0.75, 0.5]),
            np.asarray([2.0, 0.0]),
            16,
            np.asarray([[0.5, 0.5], [0.75, 0.5]]),
            0,
            False,
        ),
    )

    trial = collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=False,
        x_curr=start,
        v_curr=velocity,
        t=0.5,
        segment_dt=0.5,
        inputs=_segment_inputs(),
        base_adaptive_substep_enabled=1,
        initial_x_next=np.zeros(2),
        initial_v_next=np.zeros(2),
        initial_stage_points=np.zeros((1, 2)),
        initial_valid_mask_status=0,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=lambda _point: pytest.fail("unverified motion reached geometry"),
        primary_hit_fn=lambda _start, _points: pytest.fail(
            "unverified motion reached event classification"
        ),
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics=diagnostics,
    )

    stopped = trial.terminal_stop_result
    assert stopped is not None
    assert stopped.numerical_boundary_stop_reason == "trace_refinement_unresolved"
    np.testing.assert_array_equal(stopped.position, start)
    np.testing.assert_array_equal(stopped.velocity, velocity)
    assert stopped.terminal_outcome is not None
    assert stopped.terminal_outcome.accepted_elapsed_s == 0.0


def test_primary_polyline_hit_does_not_skip_curvature_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimum_substeps: list[int] = []
    diagnostics: dict[str, object] = {
        "collision_reintegrated_segments_count": 0,
        "adaptive_substep_segments_count": 0,
        "adaptive_substep_trigger_count": 0,
        "adaptive_substep_limit_reached_count": 0,
    }
    hit = BoundaryHit(
        position=np.asarray([0.5, 0.0]),
        normal=np.asarray([1.0, 0.0]),
        part_id=3,
        alpha_hint=0.5,
    )

    def advance(**kwargs):
        requested = int(kwargs["minimum_substeps"])
        minimum_substeps.append(requested)
        points = np.column_stack(
            (
                np.linspace(0.25, 0.75, requested),
                np.full(requested, 0.5),
            )
        )
        return points[-1], np.asarray([1.0, 0.0]), requested, points, 0, True

    geometry_risk = iter((True, False))
    monkeypatch.setattr(collision_trial, "_advance_segment_with_inputs", advance)
    monkeypatch.setattr(
        collision_trial,
        "assess_trace_geometry",
        lambda *_args, **_kwargs: SimpleNamespace(
            requires_refinement=lambda _clearance: next(geometry_risk)
        ),
    )

    trial = collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=False,
        x_curr=np.asarray([0.25, 0.5]),
        v_curr=np.asarray([1.0, 0.0]),
        t=0.5,
        segment_dt=0.5,
        inputs=_segment_inputs(),
        base_adaptive_substep_enabled=1,
        initial_x_next=np.zeros(2),
        initial_v_next=np.zeros(2),
        initial_stage_points=np.zeros((1, 2)),
        initial_valid_mask_status=0,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=lambda _point: True,
        primary_hit_fn=lambda _start, _points: hit,
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics=diagnostics,
    )

    assert minimum_substeps == [2, 4]
    assert trial.primary_hit is hit
    assert trial.stage_points.shape == (4, 2)


def test_reintegrated_segment_records_only_the_final_limit_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimum_substeps: list[int] = []
    diagnostics: dict[str, object] = {
        "collision_reintegrated_segments_count": 0,
        "adaptive_substep_segments_count": 0,
        "adaptive_substep_trigger_count": 0,
        "adaptive_substep_limit_reached_count": 0,
    }

    def advance(**kwargs):
        requested = int(kwargs["minimum_substeps"])
        minimum_substeps.append(requested)
        points = np.column_stack(
            (
                np.linspace(0.25, 0.75, requested),
                np.full(requested, 0.5),
            )
        )
        return points[-1], np.asarray([1.0, 0.0]), requested, points, 0, True

    monkeypatch.setattr(collision_trial, "_advance_segment_with_inputs", advance)
    monkeypatch.setattr(
        collision_trial,
        "assess_trace_geometry",
        lambda *_args, **_kwargs: SimpleNamespace(
            requires_refinement=lambda _clearance: True
        ),
    )

    collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=False,
        x_curr=np.asarray([0.25, 0.5]),
        v_curr=np.asarray([1.0, 0.0]),
        t=0.5,
        segment_dt=0.5,
        inputs=cast(
            collision.CollisionSegmentInputs,
            SimpleNamespace(adaptive_substep_max_splits=3),
        ),
        base_adaptive_substep_enabled=1,
        initial_x_next=np.zeros(2),
        initial_v_next=np.zeros(2),
        initial_stage_points=np.zeros((1, 2)),
        initial_valid_mask_status=0,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=lambda _point: True,
        primary_hit_fn=lambda _start, _points: None,
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics=diagnostics,
    )

    assert minimum_substeps == [2, 4, 8]
    assert diagnostics["adaptive_substep_limit_reached_count"] == 1


def test_reintegrated_segment_limit_stops_an_unresolved_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimum_substeps: list[int] = []
    start = np.asarray([0.25, 0.5])
    velocity = np.asarray([1.0, 0.0])
    diagnostics: dict[str, object] = {
        "collision_reintegrated_segments_count": 0,
        "adaptive_substep_segments_count": 0,
        "adaptive_substep_trigger_count": 0,
        "adaptive_substep_limit_reached_count": 0,
        "unresolved_crossing_count": 0,
    }

    def advance(**kwargs):
        requested = int(kwargs["minimum_substeps"])
        minimum_substeps.append(requested)
        points = np.column_stack(
            (
                np.linspace(0.25, 0.75, requested),
                np.full(requested, 0.5),
            )
        )
        return points[-1], np.asarray([2.0, 0.0]), requested, points, 0, True

    monkeypatch.setattr(collision_trial, "_advance_segment_with_inputs", advance)
    monkeypatch.setattr(
        collision_trial,
        "assess_trace_geometry",
        lambda *_args, **_kwargs: SimpleNamespace(
            requires_refinement=lambda _clearance: True
        ),
    )

    trial = collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=False,
        x_curr=start,
        v_curr=velocity,
        t=0.5,
        segment_dt=0.5,
        inputs=cast(
            collision.CollisionSegmentInputs,
            SimpleNamespace(adaptive_substep_max_splits=3),
        ),
        base_adaptive_substep_enabled=1,
        initial_x_next=np.zeros(2),
        initial_v_next=np.zeros(2),
        initial_stage_points=np.zeros((1, 2)),
        initial_valid_mask_status=0,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=lambda _point: True,
        primary_hit_fn=lambda _start, _points: None,
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics=diagnostics,
    )

    stopped = trial.terminal_stop_result
    assert stopped is not None
    assert stopped.numerical_boundary_stopped is True
    assert stopped.numerical_boundary_stop_reason == ("trace_refinement_unresolved")
    np.testing.assert_array_equal(stopped.position, start)
    np.testing.assert_array_equal(stopped.velocity, velocity)
    assert stopped.terminal_outcome is not None
    assert stopped.terminal_outcome.accepted_elapsed_s == 0.0
    assert minimum_substeps == [2, 4, 8]
    assert diagnostics["unresolved_crossing_count"] == 1


def _assert_invalid_mask_trial_resolves_wall_before_mask_retry(
    monkeypatch: pytest.MonkeyPatch,
    mask_status: int,
) -> None:
    hit = BoundaryHit(
        position=np.asarray([1.0, 0.5]),
        normal=np.asarray([1.0, 0.0]),
        part_id=9,
        alpha_hint=0.75,
    )

    def fail_retry(**_kwargs):
        raise AssertionError("wall crossing was replaced by a mask retry")

    monkeypatch.setattr(
        collision_trial, "_resolve_valid_mask_retry_with_inputs", fail_retry
    )
    trial = collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=True,
        x_curr=np.asarray([0.25, 0.5]),
        v_curr=np.asarray([1.0, 0.0]),
        t=1.0,
        segment_dt=1.0,
        inputs=_segment_inputs(),
        base_adaptive_substep_enabled=0,
        initial_x_next=np.asarray([1.25, 0.5]),
        initial_v_next=np.asarray([1.0, 0.0]),
        initial_stage_points=np.asarray([[0.75, 0.5], [1.25, 0.5]]),
        initial_valid_mask_status=mask_status,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=lambda _point: True,
        primary_hit_fn=lambda _start, _points: hit,
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics={},
    )

    assert trial.primary_hit is hit
    assert trial.primary_hit_counted is False
    assert trial.terminal_stop_result is None


def test_hard_invalid_trial_resolves_wall_before_mask_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_invalid_mask_trial_resolves_wall_before_mask_retry(
        monkeypatch,
        VALID_MASK_STATUS_HARD_INVALID,
    )


def test_mixed_mask_trial_resolves_wall_before_mask_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_invalid_mask_trial_resolves_wall_before_mask_retry(
        monkeypatch,
        VALID_MASK_STATUS_MIXED_STENCIL,
    )


@pytest.mark.parametrize(
    ("mask_status", "found_valid_prefix", "accepted_dt", "expected_reason"),
    [
        (
            mask_status,
            found_valid_prefix,
            accepted_dt,
            expected_reason,
        )
        for mask_status in (
            VALID_MASK_STATUS_MIXED_STENCIL,
            VALID_MASK_STATUS_HARD_INVALID,
        )
        for found_valid_prefix, accepted_dt, expected_reason in (
            (True, 0.25, "collision_valid_mask_hard_invalid_prefix_clipped"),
            (False, 0.0, "collision_valid_mask_hard_invalid_retry_exhausted"),
        )
    ],
    ids=(
        "mixed-True-0.25-collision_valid_mask_hard_invalid_prefix_clipped",
        "mixed-False-0.0-collision_valid_mask_hard_invalid_retry_exhausted",
        "True-0.25-collision_valid_mask_hard_invalid_prefix_clipped",
        "False-0.0-collision_valid_mask_hard_invalid_retry_exhausted",
    ),
)
def test_hard_invalid_trial_preserves_retry_elapsed_and_reason(
    monkeypatch: pytest.MonkeyPatch,
    mask_status: int,
    found_valid_prefix: bool,
    accepted_dt: float,
    expected_reason: str,
) -> None:
    monkeypatch.setattr(
        collision_trial,
        "_resolve_valid_mask_retry_with_inputs",
        lambda **_kwargs: SimpleNamespace(
            position=np.asarray([0.5, 0.5]),
            velocity=np.asarray([1.0, 0.0]),
            found_valid_prefix=found_valid_prefix,
            accepted_dt=accepted_dt,
        ),
    )
    trial = collision_trial.prepare_collision_segment_trial(
        use_precomputed_trial=True,
        x_curr=np.asarray([0.25, 0.5]),
        v_curr=np.asarray([1.0, 0.0]),
        t=1.0,
        segment_dt=1.0,
        inputs=_segment_inputs(),
        base_adaptive_substep_enabled=0,
        initial_x_next=np.asarray([1.25, 0.5]),
        initial_v_next=np.asarray([1.0, 0.0]),
        initial_stage_points=np.asarray([[0.75, 0.5], [1.25, 0.5]]),
        initial_valid_mask_status=mask_status,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=lambda _point: True,
        primary_hit_fn=lambda _start, _points: None,
        on_boundary_tol_m=1.0e-10,
        collision_diagnostics={},
    )

    stopped = trial.terminal_stop_result
    assert stopped is not None
    assert stopped.invalid_stop_reason == expected_reason
    assert stopped.terminal_outcome is not None
    assert stopped.terminal_outcome.reason == expected_reason
    assert stopped.terminal_outcome.accepted_elapsed_s == pytest.approx(accepted_dt)


def test_same_wall_contact_preserves_position_and_removes_normal_velocity() -> None:
    diagnostics: dict[str, object] = {
        "contact_sliding_count": 0,
        "contact_sliding_same_wall_count": 0,
        "contact_sliding_time_total_s": 0.1,
        "contact_sliding_remaining_dt_max_s": 0.2,
        "contact_sliding_part_counts": {},
        "contact_sliding_outcome_counts": {},
    }
    position = np.asarray([1.0, 0.5])

    state = wall_events.same_wall_contact_sliding_state(
        x_wall=position,
        v_ref=np.asarray([3.0, 4.0]),
        n_wall=np.asarray([2.0, 0.0]),
        remaining_dt=0.25,
        hit_part_ids=[7, 7],
        hit_outcomes=["reflected_specular", "reflected_diffuse"],
        collision_diagnostics=diagnostics,
    )

    assert state is not None
    wall_position, tangent_velocity, normal = state
    np.testing.assert_array_equal(wall_position, position)
    np.testing.assert_array_equal(tangent_velocity, [0.0, 4.0])
    np.testing.assert_array_equal(normal, [1.0, 0.0])
    expected = {
        "contact_sliding_count": 1,
        "contact_sliding_same_wall_count": 1,
        "contact_sliding_time_total_s": pytest.approx(0.35),
        "contact_sliding_remaining_dt_max_s": pytest.approx(0.25),
        "contact_sliding_part_counts": {"part=7": 1},
        "contact_sliding_outcome_counts": {"reflected_diffuse": 1},
    }
    assert all(diagnostics[key] == value for key, value in expected.items())
