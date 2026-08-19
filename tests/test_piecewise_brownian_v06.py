from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from field_backend_helpers import (
    geometry_provider,
    regular_axes,
    regular_field_provider,
    regular_valid_mask,
)

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
)
from particle_tracer_unified.solvers import valid_mask_retry as retry_module
from particle_tracer_unified.solvers._collision_trial import (
    _advance_segment_with_inputs,
    prepare_collision_segment_trial,
)
from particle_tracer_unified.solvers._runtime_collisions import (
    _apply_stochastic_wall_search,
)
from particle_tracer_unified.solvers._stochastic_composition import (
    _compose_piecewise_langevin_paths,
    search_piecewise_langevin_wall_crossing,
)
from particle_tracer_unified.solvers._stochastic_path import (
    _integrated_ou_covariances,
)
from particle_tracer_unified.solvers.collision_detection import TrialCollisionBatch
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.high_fidelity_collision import (
    CollisionSegmentInputs,
)
from particle_tracer_unified.solvers.integrator_common import DRAG_MODEL_STOKES
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    SegmentMotionRequest,
    ValidMaskPrefixResolution,
    trace_motion_batch,
    trace_motion_segment,
)
from particle_tracer_unified.solvers.stochastic_motion import PiecewiseLangevinPath
from particle_tracer_unified.solvers.valid_mask_retry import (
    resolve_valid_mask_retry_then_stop,
)


def _path(
    *,
    ends: tuple[float, ...],
    tau: tuple[float, ...],
    thermal: tuple[float, ...],
    sample_count: int,
    seed: int,
) -> PiecewiseLangevinPath:
    rng = np.random.default_rng(seed)
    leaf_count = len(ends)
    return PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray(ends, dtype=np.float64),
        tau_eff_s=np.asarray(tau, dtype=np.float64),
        thermal_velocity_variance_m2s2=np.asarray(thermal, dtype=np.float64),
        z_velocity=rng.normal(size=(leaf_count, sample_count)),
        z_position=rng.normal(size=(leaf_count, sample_count)),
        bridge_seeds=np.arange(leaf_count, dtype=np.int64) + 1701,
    )


def _observed_covariance(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    return np.cov(np.stack((position, velocity)), bias=True)


def test_one_leaf_piecewise_path_retains_exact_integrated_ou_covariance() -> None:
    sample_count = 180_000
    duration = 0.021
    tau = 0.037
    thermal = 2.4
    path = _path(
        ends=(duration,),
        tau=(tau,),
        thermal=(thermal,),
        sample_count=sample_count,
        seed=612,
    )

    position, velocity = path.state_at(duration)
    var_x, var_v, cov_xv = _integrated_ou_covariances(duration, tau, thermal)
    expected = np.asarray(((var_x, cov_xv), (cov_xv, var_v)))

    np.testing.assert_allclose(
        _observed_covariance(position, velocity), expected, rtol=0.018
    )


def test_two_leaf_covariance_is_exact_transition_composition() -> None:
    sample_count = 220_000
    durations = (0.017, 0.043)
    tau = (0.011, 0.089)
    thermal = (1.7, 4.2)
    path = _path(
        ends=(durations[0], sum(durations)),
        tau=tau,
        thermal=thermal,
        sample_count=sample_count,
        seed=991,
    )

    expected = np.zeros((2, 2), dtype=np.float64)
    for duration, leaf_tau, leaf_thermal in zip(durations, tau, thermal, strict=True):
        decay = np.exp(-duration / leaf_tau)
        carry = leaf_tau * (1.0 - decay)
        transition = np.asarray(((1.0, carry), (0.0, decay)))
        var_x, var_v, cov_xv = _integrated_ou_covariances(
            duration,
            leaf_tau,
            leaf_thermal,
        )
        noise = np.asarray(((var_x, cov_xv), (cov_xv, var_v)))
        expected = transition @ expected @ transition.T + noise

    position, velocity = path.state_at(path.duration_s)
    np.testing.assert_allclose(
        path.endpoint_covariance(), expected, rtol=2.0e-15, atol=2.0e-15
    )
    np.testing.assert_allclose(
        _observed_covariance(position, velocity), expected, rtol=0.018
    )


def _split_constant_coefficients_preserves_endpoint_covariance() -> None:
    duration = 0.73
    tau = 0.19
    thermal = 3.4
    path = _path(
        ends=(0.21, duration),
        tau=(tau, tau),
        thermal=(thermal, thermal),
        sample_count=2,
        seed=181,
    )
    var_x, var_v, cov_xv = _integrated_ou_covariances(duration, tau, thermal)

    np.testing.assert_allclose(
        path.endpoint_covariance(),
        np.asarray(((var_x, cov_xv), (cov_xv, var_v))),
        rtol=3.0e-15,
        atol=3.0e-15,
    )


test_splitting_constant_coefficients_into_two_leaves_preserves_endpoint_covariance = (
    _split_constant_coefficients_preserves_endpoint_covariance
)


def test_interval_replay_is_transition_innovation_and_composes() -> None:
    path = _path(
        ends=(0.13, 0.41, 0.9),
        tau=(0.07, 0.22, 0.04),
        thermal=(1.8, 0.6, 2.9),
        sample_count=3,
        seed=443,
    )
    start, split, end = 0.083, 0.37, 0.76
    eta_start_split = path.replay(start, split)
    eta_split_end = path.replay(split, end)
    eta_start_end = path.replay(start, end)
    decay_split_end, carry_split_end = path.transition(split, end)

    np.testing.assert_allclose(
        eta_start_end[0],
        eta_start_split[0] + carry_split_end * eta_start_split[1] + eta_split_end[0],
        rtol=0.0,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        eta_start_end[1],
        decay_split_end * eta_start_split[1] + eta_split_end[1],
        rtol=0.0,
        atol=3.0e-15,
    )

    wall_position = np.asarray([0.2, -0.7, 1.1])
    wall_velocity = np.asarray([1.3, 0.4, -0.8])
    decay, carry = path.transition(start, end)
    replayed_position = wall_position + carry * wall_velocity + eta_start_end[0]
    replayed_velocity = decay * wall_velocity + eta_start_end[1]
    np.testing.assert_allclose(
        replayed_position,
        wall_position + carry * wall_velocity + path.replay(start, end)[0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        replayed_velocity,
        decay * wall_velocity + path.replay(start, end)[1],
        rtol=0.0,
        atol=0.0,
    )


def test_piecewise_bridge_preserves_leaf_boundary_and_cached_queries() -> None:
    path = _path(
        ends=(0.2, 0.55),
        tau=(0.08, 0.17),
        thermal=(1.9, 2.3),
        sample_count=2,
        seed=72,
    )
    boundary = path.state_at(0.2)
    queried = path.state_at(0.31)
    path.state_at(0.44)

    repeated_boundary = path.state_at(0.2)
    repeated_query = path.state_at(0.31)
    np.testing.assert_array_equal(repeated_boundary[0], boundary[0])
    np.testing.assert_array_equal(repeated_boundary[1], boundary[1])
    np.testing.assert_array_equal(repeated_query[0], queried[0])
    np.testing.assert_array_equal(repeated_query[1], queried[1])


def test_dyadic_bridge_nodes_are_independent_of_query_order() -> None:
    first = _path(
        ends=(1.0,),
        tau=(0.19,),
        thermal=(2.7,),
        sample_count=3,
        seed=73,
    )
    second = _path(
        ends=(1.0,),
        tau=(0.19,),
        thermal=(2.7,),
        sample_count=3,
        seed=73,
    )
    times = (0.25, 0.75, 0.5, 0.125, 0.375, 0.625, 0.875)
    forward = {time_s: first.state_at(time_s) for time_s in times}
    reverse = {time_s: second.state_at(time_s) for time_s in reversed(times)}

    for time_s in times:
        np.testing.assert_array_equal(forward[time_s][0], reverse[time_s][0])
        np.testing.assert_array_equal(forward[time_s][1], reverse[time_s][1])


class _VerticalWall:
    def __init__(self, x_coordinate: float) -> None:
        self.x_coordinate = float(x_coordinate)

    def first_hit(self, start_m: np.ndarray, end_m: np.ndarray):
        from particle_tracer_unified.domain import BoundaryHit

        left = float(start_m[0]) - self.x_coordinate
        right = float(end_m[0]) - self.x_coordinate
        if left * right > 0.0 or left == right:
            return None
        alpha = float(np.clip(-left / (right - left), 0.0, 1.0))
        position = np.asarray(start_m, dtype=np.float64) + alpha * (
            np.asarray(end_m, dtype=np.float64) - np.asarray(start_m, dtype=np.float64)
        )
        return BoundaryHit(
            position=position,
            normal=np.asarray([-1.0, 0.0]),
            part_id=7,
            alpha_hint=alpha,
        )

    def inside(self, point_m: np.ndarray) -> bool:
        return bool(float(point_m[0]) <= self.x_coordinate)

    def nearest_projection(self, point_m: np.ndarray, _inside_reference_m: np.ndarray):
        from particle_tracer_unified.domain import BoundaryHit

        position = np.asarray(point_m, dtype=np.float64).copy()
        position[0] = self.x_coordinate
        return BoundaryHit(
            position=position,
            normal=np.asarray([-1.0, 0.0]),
            part_id=7,
        )

    def polyline_hit(self, start_m: np.ndarray, stage_points_m: np.ndarray):
        points = np.asarray(stage_points_m, dtype=np.float64)
        left = np.asarray(start_m, dtype=np.float64)
        for segment_index, right in enumerate(points):
            hit = self.first_hit(left, right)
            if hit is not None:
                return replace(
                    hit,
                    alpha_hint=(segment_index + float(hit.alpha_hint)) / len(points),
                )
            left = right
        return None


class _LeftWall(_VerticalWall):
    def first_hit(self, start_m: np.ndarray, end_m: np.ndarray):
        hit = super().first_hit(start_m, end_m)
        if hit is None:
            return None
        return replace(hit, normal=np.asarray([1.0, 0.0]))

    def inside(self, point_m: np.ndarray) -> bool:
        return bool(float(point_m[0]) >= self.x_coordinate)


def _stationary_trace_at(
    position_m: np.ndarray,
    *,
    duration_s: float = 1.0,
    end_time_s: float = 1.0,
):
    return trace_motion_segment(
        SegmentMotionRequest(
            position_m=np.asarray(position_m, dtype=np.float64),
            velocity_mps=np.zeros(2),
            duration_s=float(duration_s),
            end_time_s=float(end_time_s),
            spatial_dim=2,
            backend=_constant_backend(),
            adaptive_substep_enabled=0,
            adaptive_substep_max_splits=4,
            tau_stokes_s=1.0,
            particle_diameter_m=1.0e-6,
            particle_density_kgm3=1000.0,
            particle_mass_kg=1.0e-15,
            dep_particle_rel_permittivity=np.nan,
            thermophoretic_coefficient=np.nan,
            body_acceleration_mps2=np.zeros(2),
            gas_density_kgm3=1.0,
            gas_dynamic_viscosity_Pas=1.8e-5,
            gas_temperature_K=300.0,
            gas_molecular_mass_kg=6.63e-26,
            drag_model_mode=DRAG_MODEL_STOKES,
        )
    )


def test_dyadic_first_passage_detects_a_wall_excursion_and_returns_real_time() -> None:
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([1.0]),
        thermal_velocity_variance_m2s2=np.asarray([1.0]),
        z_velocity=np.zeros((1, 2)),
        z_position=np.zeros((1, 2)),
        bridge_seeds=np.asarray([10], dtype=np.int64),
    )
    trace = _stationary_trace_at(np.asarray([0.5, 0.0]))
    geometry_tolerance_m = 1.0e-8

    result = search_piecewise_langevin_wall_crossing(
        path=path,
        deterministic_trace=trace,
        boundary_service=cast(Any, _VerticalWall(0.7)),
        geometry_tolerance_m=geometry_tolerance_m,
    )

    assert result.unresolved is False
    assert result.prefetched_hit is not None
    assert result.prefetched_hit.part_id == 7
    assert 0.0 < result.prefetched_hit.alpha_hint < 0.5
    hit_position = trace.state_at(result.prefetched_hit.alpha_hint)[0]
    hit_position += path.state_at(result.prefetched_hit.alpha_hint)[0]
    assert hit_position[0] == pytest.approx(0.7, abs=geometry_tolerance_m)


def test_dyadic_first_passage_prunes_an_interval_with_resolved_clearance() -> None:
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([1.0]),
        thermal_velocity_variance_m2s2=np.asarray([1.0]),
        z_velocity=np.zeros((1, 2)),
        z_position=np.zeros((1, 2)),
        bridge_seeds=np.asarray([10], dtype=np.int64),
    )

    result = search_piecewise_langevin_wall_crossing(
        path=path,
        deterministic_trace=_stationary_trace_at(np.asarray([0.5, 0.0])),
        boundary_service=cast(Any, _VerticalWall(10.0)),
        geometry_tolerance_m=1.0e-12,
    )

    assert result.prefetched_hit is None
    assert result.unresolved is False
    assert result.stage_points.shape == (2, 2)


def test_wall_hit_time_anchors_an_order_independent_remainder_subtree() -> None:
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([1.0]),
        thermal_velocity_variance_m2s2=np.asarray([1.0]),
        z_velocity=np.zeros((1, 2)),
        z_position=np.zeros((1, 2)),
        bridge_seeds=np.asarray([10], dtype=np.int64),
    )
    first = search_piecewise_langevin_wall_crossing(
        path=path,
        deterministic_trace=_stationary_trace_at(np.asarray([0.5, 0.0])),
        boundary_service=cast(Any, _VerticalWall(0.7)),
        geometry_tolerance_m=1.0e-12,
    )
    assert first.prefetched_hit is not None
    offset = float(first.prefetched_hit.alpha_hint)

    remainder = search_piecewise_langevin_wall_crossing(
        path=path,
        deterministic_trace=_stationary_trace_at(
            np.asarray([0.5, 0.0]),
            duration_s=1.0 - offset,
            end_time_s=1.0,
        ),
        boundary_service=cast(Any, _VerticalWall(10.0)),
        geometry_tolerance_m=1.0e-12,
        stochastic_offset_s=offset,
    )

    assert remainder.prefetched_hit is None
    assert remainder.unresolved is False


def test_wall_hit_alpha_is_a_fraction_of_a_nonunit_offset_segment() -> None:
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([1.0]),
        thermal_velocity_variance_m2s2=np.asarray([1.0]),
        z_velocity=np.zeros((1, 2)),
        z_position=np.zeros((1, 2)),
        bridge_seeds=np.asarray([10], dtype=np.int64),
    )
    offset = 0.25
    duration = 0.5
    path.state_at(offset)
    trace = _stationary_trace_at(
        np.asarray([0.5, 0.0]),
        duration_s=duration,
        end_time_s=offset + duration,
    )
    geometry_tolerance_m = 1.0e-8

    result = search_piecewise_langevin_wall_crossing(
        path=path,
        deterministic_trace=trace,
        boundary_service=cast(Any, _VerticalWall(0.53)),
        geometry_tolerance_m=geometry_tolerance_m,
        stochastic_offset_s=offset,
    )

    assert result.unresolved is False
    assert result.prefetched_hit is not None
    alpha = float(result.prefetched_hit.alpha_hint)
    assert 0.0 < alpha < 1.0
    elapsed = alpha * duration
    position = trace.state_at(elapsed)[0]
    position += path.replay(offset, offset + elapsed)[0]
    assert position[0] == pytest.approx(0.53, abs=geometry_tolerance_m)


def test_stochastic_prefetch_is_authoritative_over_the_coarse_trial() -> None:
    hit = _VerticalWall(0.7).first_hit(
        np.asarray([0.5, 0.0]),
        np.asarray([0.8, 0.0]),
    )
    assert hit is not None
    coarse = TrialCollisionBatch(
        colliders=np.asarray([0], dtype=np.int64),
        safe=np.asarray([1, 2], dtype=np.int64),
        prefetched_hits={0: replace(hit, alpha_hint=0.9)},
    )

    resolved = _apply_stochastic_wall_search(
        coarse,
        active=np.asarray([True, True, False]),
        stochastic_particle_indices=(0, 1, 2),
        stochastic_prefetched_hits={1: hit, 2: hit},
    )

    np.testing.assert_array_equal(resolved.colliders, np.asarray([1]))
    np.testing.assert_array_equal(resolved.safe, np.asarray([0]))
    assert 0 not in resolved.prefetched_hits
    assert 2 not in resolved.prefetched_hits
    assert resolved.prefetched_hits[1] is hit


def test_dyadic_wall_nodes_contribute_to_valid_mask_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _constant_backend()
    mass = 3.0 * np.pi * 1.8e-5 * 1.0e-6
    motion_batch = trace_motion_batch(
        SegmentMotionBatchRequest(
            position_m=np.asarray([[0.5, 0.0]]),
            velocity_mps=np.zeros((1, 2)),
            active=np.asarray([True]),
            tau_stokes_s=np.asarray([1.0]),
            particle_diameter_m=np.asarray([1.0e-6]),
            particle_density_kgm3=np.asarray([1000.0]),
            particle_mass_kg=np.asarray([mass]),
            dep_particle_rel_permittivity=np.asarray([np.nan]),
            thermophoretic_coefficient=np.asarray([np.nan]),
            end_time_s=1.0,
            duration_s=1.0,
            spatial_dim=2,
            backend=backend,
            body_acceleration_mps2=np.zeros(2),
            gas_density_kgm3=1.0,
            gas_dynamic_viscosity_Pas=1.8e-5,
            gas_temperature_K=300.0,
            gas_molecular_mass_kg=6.63e-26,
            drag_model_mode=DRAG_MODEL_STOKES,
            adaptive_substep_enabled=0,
            adaptive_substep_max_splits=4,
        )
    )
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([1.0]),
        thermal_velocity_variance_m2s2=np.asarray([1.0]),
        z_velocity=np.zeros((1, 2)),
        z_position=np.zeros((1, 2)),
        bridge_seeds=np.asarray([10], dtype=np.int64),
    )
    sampled_points: list[np.ndarray] = []

    def fake_statuses(_backend, points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        sampled_points.append(points.copy())
        status = (
            VALID_MASK_STATUS_CLEAN
            if len(sampled_points) == 1
            else VALID_MASK_STATUS_HARD_INVALID
        )
        return np.full(points.shape[0], status, dtype=np.uint8)

    monkeypatch.setattr(
        "particle_tracer_unified.solvers._stochastic_composition."
        "sample_compiled_valid_mask_statuses",
        fake_statuses,
    )
    endpoint_position = motion_batch.endpoint_position_m.copy()
    endpoint_velocity = motion_batch.endpoint_velocity_mps.copy()
    midpoint_position = motion_batch.midpoint_position_m.copy()
    support = motion_batch.aggregate_support_status.copy()

    result = _compose_piecewise_langevin_paths(
        paths={0: path},
        motion_batch=motion_batch,
        minimum_substeps=motion_batch.substep_count.copy(),
        endpoint_position_m=endpoint_position,
        endpoint_velocity_mps=endpoint_velocity,
        midpoint_position_m=midpoint_position,
        aggregate_support_status=support,
        boundary_service=cast(Any, _VerticalWall(0.7)),
        geometry_tolerance_m=1.0e-8,
    )

    assert result.prefetched_hits[0].part_id == 7
    assert len(sampled_points) == 2
    assert sampled_points[1].shape[0] > 2
    assert support[0] == VALID_MASK_STATUS_HARD_INVALID


def test_collision_remainder_reuses_non_dyadic_wall_hit_anchor() -> None:
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([1.0]),
        thermal_velocity_variance_m2s2=np.asarray([1.0]),
        z_velocity=np.zeros((1, 2)),
        z_position=np.zeros((1, 2)),
        bridge_seeds=np.asarray([10], dtype=np.int64),
    )
    initial = search_piecewise_langevin_wall_crossing(
        path=path,
        deterministic_trace=_stationary_trace_at(np.asarray([0.5, 0.0])),
        boundary_service=cast(Any, _VerticalWall(0.7)),
        geometry_tolerance_m=1.0e-8,
    )
    assert initial.prefetched_hit is not None
    offset = float(initial.prefetched_hit.alpha_hint)
    assert offset != pytest.approx(round(offset * 16.0) / 16.0)

    duration = 1.0 - offset
    wall = _LeftWall(0.3)
    start = np.asarray([0.69, 0.0])
    particle_mass = 3.0 * np.pi * 1.8e-5 * 1.0e-6
    inputs = CollisionSegmentInputs(
        spatial_dim=2,
        compiled=_constant_backend(),
        adaptive_substep_max_splits=4,
        tau_p_i=1.0,
        particle_diameter_i=1.0e-6,
        particle_density_i=1000.0,
        particle_mass_i=particle_mass,
        dep_particle_rel_permittivity_i=np.nan,
        thermophoretic_coeff_i=np.nan,
        body_accel=np.zeros(2),
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=6.63e-26,
        drag_model_mode=DRAG_MODEL_STOKES,
        stochastic_path=path,
        stochastic_offset_s=offset,
    )

    trial = prepare_collision_segment_trial(
        use_precomputed_trial=False,
        x_curr=start,
        v_curr=np.zeros(2),
        t=1.0,
        segment_dt=duration,
        inputs=inputs,
        base_adaptive_substep_enabled=0,
        initial_x_next=start,
        initial_v_next=np.zeros(2),
        initial_stage_points=np.empty((0, 2)),
        initial_valid_mask_status=0,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=wall.inside,
        primary_hit_fn=wall.polyline_hit,
        nearest_projection_fn=wall.nearest_projection,
        on_boundary_tol_m=1.0e-8,
        collision_diagnostics={},
    )

    assert trial.terminal_stop_result is None
    assert trial.primary_hit is not None
    elapsed = float(trial.primary_hit.alpha_hint) * duration
    assert trial.accepted_trace is not None
    position = trial.accepted_trace.state_at(elapsed)[0]
    position += path.replay(offset, offset + elapsed)[0]
    assert position[0] == pytest.approx(0.3, abs=1.0e-8)


def test_collision_remainder_rejects_a_dyadic_valid_mask_excursion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-wall OU excursion cannot hide between uniform replay nodes."""

    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([1.0]),
        thermal_velocity_variance_m2s2=np.asarray([1.0]),
        z_velocity=np.zeros((1, 2)),
        z_position=np.zeros((1, 2)),
        bridge_seeds=np.asarray([10], dtype=np.int64),
    )
    particle_mass = 3.0 * np.pi * 1.8e-5 * 1.0e-6
    start = np.asarray([0.5, 0.0])
    inputs = CollisionSegmentInputs(
        spatial_dim=2,
        compiled=_constant_backend(),
        adaptive_substep_max_splits=4,
        tau_p_i=1.0,
        particle_diameter_i=1.0e-6,
        particle_density_i=1000.0,
        particle_mass_i=particle_mass,
        dep_particle_rel_permittivity_i=np.nan,
        thermophoretic_coeff_i=np.nan,
        body_accel=np.zeros(2),
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=6.63e-26,
        drag_model_mode=DRAG_MODEL_STOKES,
        stochastic_path=path,
    )
    sampled_points: list[np.ndarray] = []

    def excursion_statuses(_backend, points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        sampled_points.append(points.copy())
        return np.where(
            points[:, 0] > 0.75,
            VALID_MASK_STATUS_HARD_INVALID,
            VALID_MASK_STATUS_CLEAN,
        ).astype(np.uint8)

    monkeypatch.setattr(
        "particle_tracer_unified.solvers._collision_trial."
        "sample_compiled_valid_mask_statuses",
        excursion_statuses,
        raising=False,
    )
    wall = _VerticalWall(0.8)

    trial = prepare_collision_segment_trial(
        use_precomputed_trial=False,
        x_curr=start,
        v_curr=np.zeros(2),
        t=1.0,
        segment_dt=1.0,
        inputs=inputs,
        base_adaptive_substep_enabled=0,
        initial_x_next=start,
        initial_v_next=np.zeros(2),
        initial_stage_points=np.empty((0, 2)),
        initial_valid_mask_status=VALID_MASK_STATUS_CLEAN,
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=wall.inside,
        primary_hit_fn=wall.polyline_hit,
        nearest_projection_fn=wall.nearest_projection,
        on_boundary_tol_m=1.0e-8,
        collision_diagnostics={},
    )

    assert sampled_points
    assert np.max(sampled_points[0][:, 0]) > 0.75
    assert trial.terminal_stop_result is not None
    assert trial.terminal_stop_result.invalid_mask_stopped is True


@pytest.mark.parametrize("ratio", [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3])
def test_integrated_ou_small_leaf_covariance_matches_high_precision_formula(
    ratio: float,
) -> None:
    tau = 0.37
    thermal = 2.1
    var_x, var_v, cov_xv = _integrated_ou_covariances(ratio * tau, tau, thermal)

    with localcontext() as context:
        context.prec = 80
        a = Decimal(str(ratio))
        tau_decimal = Decimal(str(tau))
        thermal_decimal = Decimal(str(thermal))
        decay = (-a).exp()
        decay_two = (-Decimal(2) * a).exp()
        expected_var_x = float(
            thermal_decimal
            * tau_decimal
            * tau_decimal
            * (Decimal(2) * a - Decimal(3) + Decimal(4) * decay - decay_two)
        )
        expected_var_v = float(thermal_decimal * (Decimal(1) - decay_two))
        expected_cov_xv = float(
            thermal_decimal * tau_decimal * (Decimal(1) - decay) * (Decimal(1) - decay)
        )

    assert np.isfinite(var_x)
    assert var_x >= 0.0
    assert np.isfinite(var_v)
    assert var_v >= 0.0
    assert np.isfinite(cov_xv)
    assert cov_xv * cov_xv <= var_x * var_v * (1.0 + 1.0e-9)
    assert var_x == pytest.approx(expected_var_x, rel=1.0e-12, abs=0.0)
    assert var_v == pytest.approx(expected_var_v, rel=3.0e-16, abs=0.0)
    assert cov_xv == pytest.approx(expected_cov_xv, rel=6.0e-16, abs=0.0)


def _constant_backend():
    axes = regular_axes(2)
    valid = regular_valid_mask(2)
    shape = valid.shape
    field = regular_field_provider(
        axes,
        valid,
        {
            "ux": np.zeros(shape, dtype=np.float64),
            "uy": np.zeros(shape, dtype=np.float64),
            "rho_g": np.ones(shape, dtype=np.float64),
            "mu": np.full(shape, 1.8e-5, dtype=np.float64),
            "T": np.full(shape, 300.0, dtype=np.float64),
        },
    )
    geometry = geometry_provider(
        axes,
        valid,
        sdf=-np.ones(shape, dtype=np.float64),
        normal_components=(np.zeros(shape), np.ones(shape)),
    )
    return compile_runtime_backend(
        SimpleNamespace(
            geometry_provider=geometry,
            field_provider=field,
            gas=SimpleNamespace(
                density_kgm3=1.0,
                dynamic_viscosity_Pas=1.8e-5,
                temperature=300.0,
            ),
        ),
        spatial_dim=2,
    )


def test_collision_segment_composes_wall_post_state_with_saved_leaf_innovation() -> (
    None
):
    """The collision ABI must apply the saved forcing to a new wall state."""

    tau = 0.2
    first_duration = 0.17
    remaining_duration = 0.31
    thermal = 1.6
    z_velocity = np.asarray([[0.2, -0.3], [0.7, -1.1]], dtype=np.float64)
    z_position = np.asarray([[-0.1, 0.4], [1.2, 0.5]], dtype=np.float64)
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray(
            [first_duration, first_duration + remaining_duration],
            dtype=np.float64,
        ),
        tau_eff_s=np.asarray([tau, tau]),
        thermal_velocity_variance_m2s2=np.asarray([thermal, thermal]),
        z_velocity=z_velocity,
        z_position=z_position,
        bridge_seeds=np.asarray([9001, 9002], dtype=np.int64),
    )
    wall_position = np.asarray([0.43, 0.61], dtype=np.float64)
    wall_velocity = np.asarray([-0.8, 0.35], dtype=np.float64)
    particle_mass = tau * 3.0 * np.pi * 1.8e-5 * 1.0e-6
    inputs = CollisionSegmentInputs(
        spatial_dim=2,
        compiled=_constant_backend(),
        adaptive_substep_max_splits=4,
        tau_p_i=tau,
        particle_diameter_i=1.0e-6,
        particle_density_i=1000.0,
        particle_mass_i=particle_mass,
        dep_particle_rel_permittivity_i=np.nan,
        thermophoretic_coeff_i=np.nan,
        body_accel=np.zeros(2),
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=6.63e-26,
        drag_model_mode=DRAG_MODEL_STOKES,
        stochastic_path=path,
        stochastic_offset_s=first_duration,
    )

    position, velocity, _substeps, stages, _support, _resolved, _trace = (
        _advance_segment_with_inputs(
            inputs=inputs,
            x0=wall_position,
            v0=wall_velocity,
            dt_segment=remaining_duration,
            t_end_segment=first_duration + remaining_duration,
            adaptive_substep_enabled=0,
        )
    )

    var_x, var_v, cov_xv = _integrated_ou_covariances(
        remaining_duration,
        tau,
        thermal,
    )
    sigma_v = np.sqrt(var_v)
    x_from_velocity = cov_xv / sigma_v
    noise_velocity = sigma_v * z_velocity[1]
    noise_position = (
        x_from_velocity * z_velocity[1]
        + np.sqrt(var_x - x_from_velocity * x_from_velocity) * z_position[1]
    )
    decay = np.exp(-remaining_duration / tau)
    carry = tau * (1.0 - decay)
    expected_position = wall_position + carry * wall_velocity + noise_position
    expected_velocity = decay * wall_velocity + noise_velocity

    np.testing.assert_allclose(position, expected_position, rtol=2.0e-14, atol=2.0e-15)
    np.testing.assert_allclose(velocity, expected_velocity, rtol=2.0e-14, atol=2.0e-15)
    assert stages.shape == (8, 2)
    np.testing.assert_allclose(
        stages[-1], expected_position, rtol=2.0e-14, atol=2.0e-15
    )


def test_valid_mask_retry_clips_the_same_saved_brownian_path() -> None:
    tau = 0.23
    particle_mass = tau * 3.0 * np.pi * 1.8e-5 * 1.0e-6
    path = PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray([1.0]),
        tau_eff_s=np.asarray([tau]),
        thermal_velocity_variance_m2s2=np.asarray([2.0]),
        z_velocity=np.asarray([[0.8, -0.4]]),
        z_position=np.asarray([[0.3, 1.1]]),
        bridge_seeds=np.asarray([9217], dtype=np.int64),
    )
    position0 = np.asarray([0.31, 0.42])
    velocity0 = np.asarray([0.17, -0.09])
    request = SegmentMotionRequest(
        position_m=position0,
        velocity_mps=velocity0,
        duration_s=1.0,
        end_time_s=1.0,
        spatial_dim=2,
        backend=_constant_backend(),
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=4,
        tau_stokes_s=tau,
        particle_diameter_m=1.0e-6,
        particle_density_kgm3=1000.0,
        particle_mass_kg=particle_mass,
        dep_particle_rel_permittivity=np.nan,
        thermophoretic_coefficient=np.nan,
        body_acceleration_mps2=np.zeros(2),
        gas_density_kgm3=1.0,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=6.63e-26,
        drag_model_mode=DRAG_MODEL_STOKES,
    )
    # The deterministic half-segment is valid, while this saved Brownian path
    # leaves support there and first admits the next dyadic prefix.
    path.state_at(0.5)
    diagnostics = {
        "invalid_mask_retry_count": 0,
        "invalid_mask_retry_exhausted_count": 0,
    }

    resolution = resolve_valid_mask_retry_then_stop(
        request,
        collision_diagnostics=diagnostics,
        stochastic_path=path,
    )

    assert resolution.found_valid_prefix is True
    assert resolution.accepted_dt == pytest.approx(0.25)
    noise_position, noise_velocity = path.state_at(resolution.accepted_dt)
    decay = np.exp(-resolution.accepted_dt / tau)
    carry = tau * (1.0 - decay)
    np.testing.assert_allclose(
        resolution.position,
        position0 + carry * velocity0 + noise_position,
        rtol=2.0e-14,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        resolution.velocity,
        decay * velocity0 + noise_velocity,
        rtol=2.0e-14,
        atol=2.0e-15,
    )
    assert diagnostics["invalid_mask_retry_count"] == 2


def test_valid_mask_retry_records_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    resolution = ValidMaskPrefixResolution(
        position=np.zeros(2),
        velocity=np.zeros(2),
        accepted_dt=0.0,
        retry_count=4,
        found_valid_prefix=False,
    )
    monkeypatch.setattr(
        retry_module, "resolve_valid_mask_prefix", lambda *_args, **_kwargs: resolution
    )
    diagnostics = {
        "invalid_mask_retry_count": 1,
        "invalid_mask_retry_exhausted_count": 2,
    }

    actual = retry_module.resolve_valid_mask_retry_then_stop(
        cast(SegmentMotionRequest, SimpleNamespace(adaptive_substep_max_splits=4)),
        collision_diagnostics=diagnostics,
    )

    assert actual is resolution
    assert diagnostics == {
        "invalid_mask_retry_count": 5,
        "invalid_mask_retry_exhausted_count": 3,
    }
