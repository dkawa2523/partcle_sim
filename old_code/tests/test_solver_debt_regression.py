from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from field_backend_helpers import geometry_provider, regular_field_provider

from particle_tracer_unified.core.boundary_service import (
    _build_broad_phase_2d,
    build_boundary_service,
)
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from particle_tracer_unified.core.geometry2d import (
    _next_unused_edge_2d,
    _normalized_loop_coordinates_2d,
    _trace_boundary_loop_vertices_2d,
    build_boundary_loops_2d,
)
from particle_tracer_unified.core.geometry3d import (
    TriangleSurface3D,
    build_triangle_surface,
)
from particle_tracer_unified.domain import BoundaryHit, BoundaryQuery
from particle_tracer_unified.solvers import _segment_motion_scalar as scalar_motion
from particle_tracer_unified.solvers._collision_detection_candidates import (
    boundary_edge_aabb_arrays_2d,
    edge_aabb_candidate_mask_2d,
    geometry_grid_spacing_2d,
    triangle_aabb_candidate_mask_3d,
)
from particle_tracer_unified.solvers._segment_stage_dynamics import (
    _advance_etd2_substep,
)
from particle_tracer_unified.solvers.collision_detection import (
    TrialCollisionBatch,
    classify_trial_collisions_2d,
    classify_trial_collisions_3d,
    promote_stage_trace_collisions,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_NONE,
    DRAG_MODEL_STOKES,
    stokes_relaxation_time,
)
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchDestination,
    SegmentMotionBatchRequest,
    SegmentMotionRequest,
    trace_motion_batch,
    trace_motion_segment,
)


def _constant_regular_backend(spatial_dim: int):
    axes = tuple(
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64) for _ in range(spatial_dim)
    )
    shape = tuple(3 for _ in range(spatial_dim))
    valid = np.ones(shape, dtype=bool)
    field = regular_field_provider(
        axes,
        valid,
        {
            "ux": np.full(shape, 0.02, dtype=np.float64),
            "uy": np.full(shape, -0.01, dtype=np.float64),
            "rho_g": np.full(shape, 1.2, dtype=np.float64),
            "mu": np.full(shape, 1.8e-5, dtype=np.float64),
            "T": np.full(shape, 300.0, dtype=np.float64),
        },
    )
    normal_components = [np.zeros(shape) for _ in range(spatial_dim)]
    normal_components[-1].fill(1.0)
    geometry = geometry_provider(
        axes,
        valid,
        sdf=-np.ones(shape, dtype=np.float64),
        normal_components=tuple(normal_components),
    )
    return compile_runtime_backend(
        SimpleNamespace(
            geometry_provider=geometry,
            field_provider=field,
            gas=SimpleNamespace(
                density_kgm3=1.2,
                dynamic_viscosity_Pas=1.8e-5,
                temperature=300.0,
            ),
        ),
        spatial_dim=spatial_dim,
    )


def _constant_regular_backend_3d():
    return _constant_regular_backend(3)


def _regular_batch_request_2d() -> SegmentMotionBatchRequest:
    mass = 1.0e-15
    diameter = 1.0e-6
    tau = stokes_relaxation_time(mass, 1.8e-5, diameter)
    return SegmentMotionBatchRequest(
        position_m=np.full((2, 2), 0.5, dtype=np.float64),
        velocity_mps=np.asarray([[0.01, 0.0], [0.01, 0.0]], dtype=np.float64),
        active=np.asarray([True, False]),
        tau_stokes_s=np.full(2, tau, dtype=np.float64),
        particle_diameter_m=np.full(2, diameter, dtype=np.float64),
        particle_density_kgm3=np.full(2, 1200.0, dtype=np.float64),
        particle_mass_kg=np.full(2, mass, dtype=np.float64),
        dep_particle_rel_permittivity=np.full(2, np.nan, dtype=np.float64),
        thermophoretic_coefficient=np.full(2, np.nan, dtype=np.float64),
        end_time_s=2.0e-4,
        duration_s=2.0e-4,
        spatial_dim=2,
        backend=_constant_regular_backend(2),
        body_acceleration_mps2=np.zeros(2, dtype=np.float64),
        gas_density_kgm3=1.2,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=6.633521465546082e-26,
        drag_model_mode=int(DRAG_MODEL_STOKES),
        adaptive_substep_enabled=1,
        adaptive_substep_max_splits=4,
    )


def _nested_square_runtime():
    outer = np.asarray(
        [
            [[0.0, 0.0], [4.0, 0.0]],
            [[4.0, 0.0], [4.0, 4.0]],
            [[4.0, 4.0], [0.0, 4.0]],
            [[0.0, 4.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    inner = np.asarray(
        [
            [[1.5, 1.5], [2.5, 1.5]],
            [[2.5, 1.5], [2.5, 2.5]],
            [[2.5, 2.5], [1.5, 2.5]],
            [[1.5, 2.5], [1.5, 1.5]],
        ],
        dtype=np.float64,
    )
    edges = np.concatenate((outer, inner), axis=0)
    shape = (9, 9)
    geometry = SimpleNamespace(
        spatial_dim=2,
        axes=(np.linspace(0.0, 4.0, 9), np.linspace(0.0, 4.0, 9)),
        boundary_loops_2d=build_boundary_loops_2d(edges),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([10] * 4 + [20] * 4, dtype=np.int32),
        sdf=np.zeros(shape, dtype=np.float64),
        nearest_boundary_part_id_map=np.zeros(shape, dtype=np.int32),
        normal_components=(np.zeros(shape), np.ones(shape)),
    )
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=geometry),
        field_provider=None,
    )


def _cube_triangles() -> np.ndarray:
    corners = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    vertex_ids = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (3, 6, 2),
        (3, 7, 6),
        (0, 7, 3),
        (0, 4, 7),
    )
    return np.asarray(
        [[corners[a], corners[b], corners[c]] for a, b, c in vertex_ids],
        dtype=np.float64,
    )


def _cube_runtime(triangles: np.ndarray):
    shape = (9, 9, 9)
    geometry = SimpleNamespace(
        spatial_dim=3,
        axes=tuple(np.linspace(-1.0, 1.0, 9) for _ in range(3)),
        boundary_loops_2d=(),
        boundary_triangles=triangles,
        boundary_triangle_part_ids=np.ones(triangles.shape[0], dtype=np.int32),
        sdf=np.zeros(shape, dtype=np.float64),
        nearest_boundary_part_id_map=np.ones(shape, dtype=np.int32),
        normal_components=(np.zeros(shape), np.zeros(shape), np.ones(shape)),
    )
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=geometry),
        field_provider=None,
    )


def test_2d_candidate_metadata_falls_back_to_geometry_without_reordering() -> None:
    runtime = _nested_square_runtime()
    service = cast(
        BoundaryQuery[TriangleSurface3D], SimpleNamespace(broad_phase_2d=None)
    )

    assert geometry_grid_spacing_2d(runtime, service) == pytest.approx(0.5)
    edge_min, edge_max = boundary_edge_aabb_arrays_2d(
        runtime, service, on_boundary_tol_m=0.1
    )

    assert edge_min is not None
    assert edge_max is not None
    np.testing.assert_allclose(edge_min[0], [-0.1, -0.1])
    np.testing.assert_allclose(edge_max[0], [4.1, 0.1])


def test_boundary_service_broad_phase_owns_spacing_and_padded_edge_bounds() -> None:
    edges = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [2.0, 1.0]],
        ],
        dtype=np.float64,
    )
    geometry = SimpleNamespace(
        spatial_dim=2,
        axes=(
            np.asarray([0.0, 0.5, 2.0]),
            np.asarray([0.0, 0.25, 1.0]),
        ),
        boundary_edges=edges,
    )
    runtime = SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geometry))

    broad_phase = _build_broad_phase_2d(runtime, on_boundary_tol_m=0.1)

    assert broad_phase.grid_spacing_m == pytest.approx(0.25)
    assert broad_phase.edge_aabb_padding_m == pytest.approx(0.1)
    assert broad_phase.edge_aabb_min_padded is not None
    assert broad_phase.edge_aabb_max_padded is not None
    np.testing.assert_allclose(
        broad_phase.edge_aabb_min_padded,
        [[-0.1, -0.1], [1.9, -0.1]],
    )
    np.testing.assert_allclose(
        broad_phase.edge_aabb_max_padded,
        [[2.1, 0.1], [2.1, 1.1]],
    )
    assert not broad_phase.edge_aabb_min_padded.flags.writeable
    assert not broad_phase.edge_aabb_max_padded.flags.writeable


def test_boundary_service_broad_phase_handles_missing_or_malformed_geometry() -> None:
    missing = _build_broad_phase_2d(
        SimpleNamespace(),
        on_boundary_tol_m=1.0e-9,
    )
    malformed = _build_broad_phase_2d(
        SimpleNamespace(
            geometry_provider=SimpleNamespace(
                geometry=SimpleNamespace(
                    spatial_dim=2,
                    axes=(np.asarray([0.0, np.nan]),),
                    boundary_edges=np.zeros((2, 2), dtype=np.float64),
                )
            )
        ),
        on_boundary_tol_m=2.0e-9,
    )

    assert missing.grid_spacing_m == 0.0
    assert missing.edge_aabb_min_padded is None
    assert missing.edge_aabb_max_padded is None
    assert malformed.grid_spacing_m == 0.0
    assert malformed.edge_aabb_min_padded is None
    assert malformed.edge_aabb_max_padded is None


def test_unavailable_candidate_metadata_never_prunes_a_trial() -> None:
    starts_2d = np.asarray([[0.0, 0.0]], dtype=np.float64)
    midpoints_2d = np.asarray([[0.1, 0.0]], dtype=np.float64)
    endpoints_2d = np.asarray([[0.2, 0.0]], dtype=np.float64)
    service = cast(BoundaryQuery[TriangleSurface3D], SimpleNamespace())

    candidate_2d, unknown_2d = edge_aabb_candidate_mask_2d(
        SimpleNamespace(),
        service,
        np.asarray([0]),
        starts_2d,
        endpoints_2d,
        midpoints_2d,
        on_boundary_tol_m=1.0e-9,
    )
    candidate_3d, unknown_3d = triangle_aabb_candidate_mask_3d(
        None,
        np.asarray([0]),
        np.zeros((1, 3), dtype=np.float64),
        np.ones((1, 3), dtype=np.float64),
        np.full((1, 3), 0.5, dtype=np.float64),
        on_boundary_tol_m=1.0e-9,
    )

    assert candidate_2d.tolist() == [True]
    assert unknown_2d == 1
    assert candidate_3d.tolist() == [True]
    assert unknown_3d == 1


def test_nonfinite_trial_remains_a_conservative_candidate() -> None:
    runtime = _nested_square_runtime()
    service = cast(
        BoundaryQuery[TriangleSurface3D], SimpleNamespace(broad_phase_2d=None)
    )
    candidate, unknown = edge_aabb_candidate_mask_2d(
        runtime,
        service,
        np.asarray([0]),
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        np.asarray([[np.nan, 0.0]], dtype=np.float64),
        np.asarray([[0.1, 0.0]], dtype=np.float64),
        on_boundary_tol_m=1.0e-9,
    )

    assert candidate.tolist() == [True]
    assert unknown == 1


def test_etd2_substep_preserves_stage_order_and_float64_results() -> None:
    x0 = np.asarray([0.5, 0.6, 0.7], dtype=np.float64)
    v0 = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
    body = np.asarray([0.5, -0.25, 0.125], dtype=np.float64)
    dt_sub = 0.2
    target_tau_s = 0.4
    particle_mass_kg = 1.0e-15
    particle_diameter_m = 1.0e-6
    local_mu_pas = particle_mass_kg / (3.0 * np.pi * particle_diameter_m * target_tau_s)
    compiled = _constant_regular_backend_3d()
    compiled = replace(
        compiled,
        gas_mu=np.full_like(compiled.gas_mu, local_mu_pas),
    )
    result = _advance_etd2_substep(
        x0=x0,
        v0=v0,
        dt_sub=dt_sub,
        t_sub_start=0.7,
        spatial_dim=3,
        compiled=compiled,
        body=body,
        tau_stokes=target_tau_s,
        particle_diameter_m=particle_diameter_m,
        particle_density_kgm3=1200.0,
        particle_mass_kg=particle_mass_kg,
        dep_particle_rel_permittivity=np.nan,
        thermophoretic_coeff=np.nan,
        gas_density_kgm3=1.2,
        gas_mu_pas=local_mu_pas,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=6.633521465546082e-26,
        drag_model_mode=int(DRAG_MODEL_STOKES),
    )

    local_tau_s = stokes_relaxation_time(
        particle_mass_kg,
        local_mu_pas,
        particle_diameter_m,
    )

    def exact_state(elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        equilibrium_velocity = local_tau_s * body
        decay = np.exp(-elapsed_s / local_tau_s)
        velocity = equilibrium_velocity + (v0 - equilibrium_velocity) * decay
        position = (
            x0
            + equilibrium_velocity * elapsed_s
            + local_tau_s * (v0 - equilibrium_velocity) * (1.0 - decay)
        )
        return position, velocity

    expected_arrays = (*exact_state(dt_sub), *exact_state(0.5 * dt_sub))
    for actual, expected in zip(result[:4], expected_arrays, strict=True):
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)
    assert result[4:6] == (local_tau_s, local_tau_s)
    np.testing.assert_array_equal(result[6], result[2])


def test_motion_rejects_invalid_drag_tau_before_field_sampling() -> None:
    request = SegmentMotionRequest(
        position_m=np.asarray([0.5, 0.6, 0.7], dtype=np.float64),
        velocity_mps=np.asarray([0.1, -0.2, 0.3], dtype=np.float64),
        duration_s=0.2,
        end_time_s=0.9,
        spatial_dim=3,
        backend=_constant_regular_backend_3d(),
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=0,
        tau_stokes_s=np.nan,
        particle_diameter_m=1.0e-6,
        particle_density_kgm3=1200.0,
        particle_mass_kg=1.0e-15,
        dep_particle_rel_permittivity=np.nan,
        thermophoretic_coefficient=np.nan,
        body_acceleration_mps2=np.zeros(3, dtype=np.float64),
        gas_density_kgm3=1.2,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=6.633521465546082e-26,
        drag_model_mode=int(DRAG_MODEL_STOKES),
    )

    with pytest.raises(ValueError, match="relaxation time must be finite and > 0"):
        trace_motion_segment(request)


def test_scalar_motion_trace_preserves_zero_and_prefix_state_contracts() -> None:
    request = _regular_batch_request_2d().particle_request(0)
    zero_trace = trace_motion_segment(replace(request, duration_s=0.0))

    assert zero_trace.times_s.tolist() == [request.end_time_s]
    assert zero_trace.positions_m.shape == (1, 2)
    assert zero_trace.velocities_mps.shape == (1, 2)
    assert zero_trace.support_status.dtype == np.uint8
    assert zero_trace.tau_start_s.dtype == np.float64
    assert zero_trace.tau_mid_s.dtype == np.float64
    assert zero_trace.substep_count == 1

    trace = trace_motion_segment(request.with_minimum_substeps(2))
    start_position, start_velocity = trace.state_at(-1.0)
    end_position, end_velocity = trace.state_at(request.duration_s)
    midpoint_position, midpoint_velocity = trace.state_at(0.5 * request.duration_s)
    midpoint = trace.prefix(0.5 * request.duration_s)

    np.testing.assert_array_equal(start_position, request.position_m)
    np.testing.assert_array_equal(start_velocity, request.velocity_mps)
    np.testing.assert_array_equal(end_position, trace.endpoint_position_m)
    np.testing.assert_array_equal(end_velocity, trace.endpoint_velocity_mps)
    np.testing.assert_array_equal(midpoint_position, midpoint.endpoint_position_m)
    np.testing.assert_array_equal(midpoint_velocity, midpoint.endpoint_velocity_mps)


@pytest.mark.parametrize(
    ("statuses", "require_clean", "expected_retry", "found"),
    [
        ([VALID_MASK_STATUS_CLEAN], False, 1, True),
        (
            [VALID_MASK_STATUS_MIXED_STENCIL, VALID_MASK_STATUS_CLEAN],
            True,
            2,
            True,
        ),
        (
            [VALID_MASK_STATUS_HARD_INVALID, VALID_MASK_STATUS_HARD_INVALID],
            False,
            2,
            False,
        ),
    ],
)
def test_valid_mask_prefix_keeps_halving_and_acceptance_order(
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[int],
    require_clean: bool,
    expected_retry: int,
    found: bool,
) -> None:
    request = _regular_batch_request_2d().particle_request(0)
    durations: list[float] = []
    status_iterator = iter(statuses)

    def trace(prefix_request: SegmentMotionRequest) -> SimpleNamespace:
        durations.append(prefix_request.duration_s)
        duration = float(prefix_request.duration_s)
        return SimpleNamespace(
            aggregate_support_status=next(status_iterator),
            endpoint_position_m=np.asarray([duration, -duration], dtype=np.float64),
            endpoint_velocity_mps=np.asarray([1.0, -1.0], dtype=np.float64),
            local_error_resolved=True,
        )

    monkeypatch.setattr(scalar_motion, "trace_motion_segment", trace)
    result = scalar_motion.resolve_valid_mask_prefix(
        request,
        max_halving_count=len(statuses),
        require_clean_prefix=require_clean,
    )

    expected_durations = [
        request.duration_s * 0.5**split for split in range(1, expected_retry + 1)
    ]
    assert durations == expected_durations
    assert result.retry_count == expected_retry
    assert result.found_valid_prefix is found
    if found:
        assert result.accepted_dt == expected_durations[-1]
        np.testing.assert_array_equal(
            result.position,
            [expected_durations[-1], -expected_durations[-1]],
        )
    else:
        assert result.accepted_dt == 0.0
        np.testing.assert_array_equal(result.position, request.position_m)


@pytest.mark.parametrize(
    ("duration_s", "max_halving_count"),
    [(0.0, 2), (1.0, 0)],
)
def test_valid_mask_prefix_short_circuits_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    duration_s: float,
    max_halving_count: int,
) -> None:
    request = replace(
        _regular_batch_request_2d().particle_request(0),
        duration_s=duration_s,
    )

    def unexpected_trace(_request: SegmentMotionRequest) -> None:
        raise AssertionError("short-circuited prefix must not be replayed")

    monkeypatch.setattr(scalar_motion, "trace_motion_segment", unexpected_trace)
    result = scalar_motion.resolve_valid_mask_prefix(
        request,
        max_halving_count=max_halving_count,
    )

    assert result.accepted_dt == 0.0
    assert result.retry_count == 0
    assert result.found_valid_prefix is False


def test_regular_motion_batch_preserves_inactive_state_and_output_schema() -> None:
    result = trace_motion_batch(_regular_batch_request_2d())

    np.testing.assert_allclose(
        result.endpoint_position_m,
        [[0.5000039410537249, 0.499998058946275], [0.5, 0.5]],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.endpoint_velocity_mps,
        [[0.019999999999999983, -0.009999999999999981], [0.01, 0.0]],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.midpoint_position_m,
        [[0.5000019410537273, 0.4999990589462726], [0.5, 0.5]],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert result.endpoint_position_m.dtype == np.float64
    assert result.endpoint_velocity_mps.dtype == np.float64
    assert result.midpoint_position_m.dtype == np.float64
    assert result.substep_count.dtype == np.int32
    assert result.aggregate_support_status.dtype == np.uint8
    # Constant ETD coefficients are integrated exactly without tau-based splits.
    assert result.substep_count.tolist() == [1, 1]
    assert result.aggregate_support_status.tolist() == [0, 0]


def test_precise_batch_fallback_preserves_inactive_state_and_scalar_rule() -> None:
    request = replace(
        _regular_batch_request_2d(),
        drag_model_mode=int(DRAG_MODEL_NONE),
        tau_stokes_s=np.full(2, np.inf, dtype=np.float64),
    )

    result = trace_motion_batch(request)
    replay = trace_motion_segment(request.particle_request(0))

    np.testing.assert_array_equal(
        result.endpoint_position_m[0], replay.endpoint_position_m
    )
    np.testing.assert_array_equal(
        result.endpoint_velocity_mps[0], replay.endpoint_velocity_mps
    )
    np.testing.assert_array_equal(result.endpoint_position_m[1], request.position_m[1])
    np.testing.assert_array_equal(
        result.endpoint_velocity_mps[1], request.velocity_mps[1]
    )
    np.testing.assert_array_equal(result.midpoint_position_m[1], request.position_m[1])
    assert result.substep_count[1] == 1
    assert result.aggregate_support_status[1] == VALID_MASK_STATUS_CLEAN

    charged_request = replace(
        request,
        electric_q_over_m_Ckg=np.asarray([2.5, -3.5], dtype=np.float64),
    )
    assert charged_request.particle_request(1).electric_q_over_m_Ckg == -3.5


def test_motion_batch_writes_the_supplied_destination_buffers() -> None:
    request = _regular_batch_request_2d()
    destination = SegmentMotionBatchDestination(
        endpoint_position_m=np.full_like(request.position_m, -1.0),
        endpoint_velocity_mps=np.full_like(request.velocity_mps, -2.0),
        midpoint_position_m=np.full_like(request.position_m, -3.0),
        substep_count=np.full(request.position_m.shape[0], -4, dtype=np.int32),
        aggregate_support_status=np.full(
            request.position_m.shape[0],
            255,
            dtype=np.uint8,
        ),
        local_error_resolved=np.zeros(request.position_m.shape[0], dtype=bool),
    )

    result = trace_motion_batch(request, destination)

    assert result.request is request
    assert result.endpoint_position_m is destination.endpoint_position_m
    assert result.endpoint_velocity_mps is destination.endpoint_velocity_mps
    assert result.midpoint_position_m is destination.midpoint_position_m
    assert result.substep_count is destination.substep_count
    assert result.aggregate_support_status is destination.aggregate_support_status
    assert result.local_error_resolved is destination.local_error_resolved
    np.testing.assert_allclose(
        result.endpoint_position_m,
        [[0.5000039410537249, 0.499998058946275], [0.5, 0.5]],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.endpoint_velocity_mps,
        [[0.019999999999999983, -0.009999999999999981], [0.01, 0.0]],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert result.substep_count.tolist() == [1, 1]
    assert result.aggregate_support_status.tolist() == [0, 0]
    assert result.local_error_resolved.tolist() == [True, True]


@pytest.mark.parametrize(
    "invalid_state",
    [
        "position",
        "dimension",
        "active",
    ],
)
def test_motion_batch_preserves_state_validation_order(
    invalid_state: str,
) -> None:
    request = _regular_batch_request_2d()
    if invalid_state == "position":
        request = replace(
            request,
            position_m=np.asarray([0.5, 0.5], dtype=np.float64),
        )
        message = "same 2D shape"
    elif invalid_state == "dimension":
        request = replace(request, spatial_dim=3)
        message = "dimension must match spatial_dim"
    else:
        request = replace(request, active=np.asarray([True]))
        message = "active mask must have shape"

    with pytest.raises(ValueError, match=message):
        trace_motion_batch(request)


def test_motion_batch_rejects_a_mismatched_destination_before_advancing() -> None:
    request = _regular_batch_request_2d()
    destination = SegmentMotionBatchDestination(
        endpoint_position_m=np.empty((1, 2), dtype=np.float64),
        endpoint_velocity_mps=np.empty_like(request.velocity_mps),
        midpoint_position_m=np.empty_like(request.position_m),
        substep_count=np.empty(2, dtype=np.int32),
        aggregate_support_status=np.empty(2, dtype=np.uint8),
        local_error_resolved=np.empty(2, dtype=bool),
    )

    with pytest.raises(ValueError, match="destination buffers do not match request"):
        trace_motion_batch(request, destination)


@pytest.mark.parametrize(
    ("particle_mass", "message"),
    [
        (np.asarray([1.0e-15]), "same shape as tau_p"),
        (np.asarray([1.0e-15, 0.0]), "must be finite and > 0"),
    ],
)
def test_motion_batch_rejects_invalid_particle_mass(
    particle_mass: np.ndarray,
    message: str,
) -> None:
    request = replace(
        _regular_batch_request_2d(),
        particle_mass_kg=particle_mass,
    )

    with pytest.raises(ValueError, match=message):
        trace_motion_batch(request)


def test_stage_trace_promotion_preserves_order_and_collision_evidence() -> None:
    existing_hit = BoundaryHit(
        position=np.asarray([0.0, 0.0]),
        normal=np.asarray([1.0, 0.0]),
        part_id=10,
    )
    promoted_hit = BoundaryHit(
        position=np.asarray([4.5, 0.0]),
        normal=np.asarray([-1.0, 0.0]),
        part_id=40,
        alpha_hint=0.5,
    )
    events: list[tuple[str, int, str]] = []

    def polyline_hit(start: np.ndarray, trace: np.ndarray) -> BoundaryHit | None:
        index = int(start[0])
        events.append(("hit", index, trace.dtype.str))
        return promoted_hit if index == 4 else None

    def contains(trace: np.ndarray) -> np.ndarray:
        index = int(trace[0, 0])
        events.append(("contains", index, trace.dtype.str))
        if index == 1:
            return np.asarray([True, False])
        return np.ones(trace.shape[0], dtype=bool)

    service = cast(
        BoundaryQuery[TriangleSurface3D],
        SimpleNamespace(polyline_hit=polyline_hit, contains=contains),
    )
    trial = TrialCollisionBatch(
        colliders=np.asarray([0], dtype=np.int32),
        safe=np.asarray([5, 4, 3, 2, 1], dtype=np.int32),
        prefetched_hits={0: existing_hit},
    )
    starts = np.asarray([[i, -i] for i in range(6)], dtype=np.float32)
    stage_traces = {
        -1: np.asarray([[9, 0]], dtype=np.int16),
        5: np.asarray([[5, 0]], dtype=np.int16),
        2: np.asarray([[np.nan, 0.0]], dtype=np.float64),
        3: np.asarray([[3, 0], [3, 1]], dtype=np.int16),
        4: np.asarray([[4, 0], [4, 1]], dtype=np.int16),
        1: np.asarray([[1, 0], [1, 1]], dtype=np.int16),
    }

    result = promote_stage_trace_collisions(
        trial,
        active=np.asarray([True, True, True, True, True, False]),
        x_start=starts,
        stage_traces=stage_traces,
        boundary_service=service,
    )

    assert result.colliders.dtype == np.int64
    assert result.safe.dtype == np.int64
    assert result.colliders.tolist() == [0, 1, 2, 4]
    assert result.safe.tolist() == [3, 5]
    assert list(result.prefetched_hits) == [0, 4]
    assert result.prefetched_hits[0] is existing_hit
    assert result.prefetched_hits[4] is promoted_hit
    assert events == [
        ("hit", 3, "<f8"),
        ("contains", 3, "<f8"),
        ("hit", 4, "<f8"),
        ("contains", 4, "<f8"),
        ("hit", 1, "<f8"),
        ("contains", 1, "<f8"),
    ]


def test_stage_trace_promotion_returns_the_original_empty_batch() -> None:
    trial = TrialCollisionBatch(
        colliders=np.empty(0, dtype=np.int64),
        safe=np.asarray([0], dtype=np.int64),
        prefetched_hits={},
    )

    result = promote_stage_trace_collisions(
        trial,
        active=np.asarray([True]),
        x_start=np.asarray([[0.0, 0.0]]),
        stage_traces={},
        boundary_service=cast(BoundaryQuery[TriangleSurface3D], SimpleNamespace()),
    )

    assert result is trial


def test_3d_collision_classification_preserves_broad_phase_diagnostics() -> None:
    triangles = _cube_triangles()
    runtime = _cube_runtime(triangles)
    surface = build_triangle_surface(
        triangles,
        np.arange(1, 13, dtype=np.int32),
        validate_closed=True,
    )
    service = cast(
        BoundaryQuery[TriangleSurface3D],
        build_boundary_service(
            runtime,
            spatial_dim=3,
            on_boundary_tol_m=1.0e-7,
            triangle_surface_3d=surface,
        ),
    )
    starts = np.zeros((4, 3), dtype=np.float64)
    midpoints = np.asarray(
        [[0.5, 0.0, 0.0], [0.1, 0.1, 0.1], [0.0, 0.2, 0.2], [0.2, 0.2, 0.2]],
        dtype=np.float64,
    )
    endpoints = np.asarray(
        [[1.5, 0.0, 0.0], [0.2, 0.2, 0.2], [0.0, 0.3, 0.3], [1.2, 0.2, 0.2]],
        dtype=np.float64,
    )
    diagnostics: dict[str, object] = {
        "on_boundary_promoted_inside_count": 0,
        "etd2_midpoint_outside_count": 0,
    }

    result = classify_trial_collisions_3d(
        runtime,
        active=np.asarray([True, True, False, True]),
        x=starts,
        x_trial=endpoints,
        x_mid_trial=midpoints,
        boundary_service=service,
        on_boundary_tol_m=1.0e-7,
        collision_diagnostics=diagnostics,
        boundary_broad_phase_enabled=True,
        boundary_broad_phase_debug_check=True,
    )

    assert result.colliders.tolist() == [0, 3]
    assert result.safe.tolist() == [1]
    assert sorted(result.prefetched_hits) == [0, 3]
    assert result.prefetched_hits[0].part_id == 7
    assert result.prefetched_hits[0].alpha_hint == pytest.approx(0.75)
    assert result.prefetched_hits[3].alpha_hint == pytest.approx(0.9)
    expected = {
        "on_boundary_promoted_inside_count": 0,
        "etd2_midpoint_outside_count": 0,
        "boundary_exact_solve_count": 2,
        "boundary_broad_phase_checked_count": 3,
        "boundary_broad_phase_candidate_count": 2,
        "boundary_broad_phase_pruned_count": 1,
        "boundary_broad_phase_missed_hit_count": 0,
        "boundary_broad_phase_unknown_count": 0,
        "boundary_broad_phase_candidate_ratio": pytest.approx(2.0 / 3.0),
    }
    assert all(diagnostics[key] == value for key, value in expected.items())


def test_3d_collision_without_hit_uses_midpoint_and_endpoint_containment() -> None:
    triangles = _cube_triangles()
    runtime = _cube_runtime(triangles)
    surface = build_triangle_surface(
        triangles,
        np.ones(12, dtype=np.int32),
        validate_closed=True,
    )
    service = cast(
        BoundaryQuery[TriangleSurface3D],
        build_boundary_service(
            runtime,
            spatial_dim=3,
            on_boundary_tol_m=1.0e-7,
            triangle_surface_3d=surface,
        ),
    )
    diagnostics: dict[str, object] = {
        "on_boundary_promoted_inside_count": 0,
        "etd2_midpoint_outside_count": 0,
    }

    result = classify_trial_collisions_3d(
        runtime,
        active=np.asarray([True, True]),
        x=np.asarray([[0.0, 0.0, 0.0], [1.2, 1.2, 1.2]], dtype=np.float64),
        x_trial=np.asarray([[0.2, 0.2, 0.2], [1.4, 1.4, 1.4]], dtype=np.float64),
        x_mid_trial=np.asarray(
            [[0.1, 0.1, 0.1], [1.3, 1.3, 1.3]],
            dtype=np.float64,
        ),
        boundary_service=service,
        on_boundary_tol_m=1.0e-7,
        collision_diagnostics=diagnostics,
    )

    assert result.colliders.tolist() == [1]
    assert result.safe.tolist() == [0]
    assert result.prefetched_hits == {}
    assert diagnostics["etd2_midpoint_outside_count"] == 1


def test_3d_broad_phase_debug_promotes_a_pruned_service_hit() -> None:
    triangles = _cube_triangles()
    runtime = _cube_runtime(triangles)
    surface = build_triangle_surface(
        triangles,
        np.ones(12, dtype=np.int32),
        validate_closed=True,
    )
    service = build_boundary_service(
        runtime,
        spatial_dim=3,
        on_boundary_tol_m=1.0e-7,
        triangle_surface_3d=surface,
    )
    forced_hit = BoundaryHit(
        position=np.asarray([0.2, 0.0, 0.0], dtype=np.float64),
        normal=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        part_id=99,
        alpha_hint=0.5,
        primitive_id=3,
        primitive_kind="test",
    )
    debug_service = cast(
        BoundaryQuery[TriangleSurface3D],
        replace(service, polyline_hit=lambda _start, _stages: forced_hit),
    )
    diagnostics: dict[str, object] = {
        "on_boundary_promoted_inside_count": 0,
        "etd2_midpoint_outside_count": 0,
    }

    result = classify_trial_collisions_3d(
        runtime,
        active=np.asarray([True]),
        x=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        x_trial=np.asarray([[0.2, 0.0, 0.0]], dtype=np.float64),
        x_mid_trial=np.asarray([[0.1, 0.0, 0.0]], dtype=np.float64),
        boundary_service=debug_service,
        on_boundary_tol_m=1.0e-7,
        collision_diagnostics=diagnostics,
        boundary_broad_phase_enabled=True,
        boundary_broad_phase_debug_check=True,
    )

    assert result.colliders.tolist() == [0]
    assert result.safe.size == 0
    assert result.prefetched_hits[0] is forced_hit
    assert diagnostics["boundary_broad_phase_missed_hit_count"] == 1


def test_2d_collision_prefetch_preserves_internal_wall_and_diagnostics() -> None:
    runtime = _nested_square_runtime()
    service = cast(
        BoundaryQuery[TriangleSurface3D],
        build_boundary_service(
            runtime,
            spatial_dim=2,
            on_boundary_tol_m=1.0e-9,
            triangle_surface_3d=None,
        ),
    )
    diagnostics: dict[str, object] = {
        "on_boundary_promoted_inside_count": 0,
        "etd2_midpoint_outside_count": 0,
    }

    result = classify_trial_collisions_2d(
        runtime,
        n_particles=3,
        active=np.asarray([True, True, False]),
        x=np.asarray([[0.5, 2.0], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64),
        x_trial=np.asarray(
            [[3.5, 2.0], [0.6, 0.5], [1.2, 0.5]],
            dtype=np.float64,
        ),
        x_mid_trial=np.asarray(
            [[1.0, 2.0], [0.55, 0.5], [0.6, 0.5]],
            dtype=np.float64,
        ),
        boundary_service=service,
        on_boundary_tol_m=1.0e-9,
        collision_diagnostics=diagnostics,
        boundary_broad_phase_enabled=True,
        boundary_broad_phase_debug_check=True,
    )

    assert result.colliders.tolist() == [0]
    assert result.safe.tolist() == [1]
    assert sorted(result.prefetched_hits) == [0]
    assert result.prefetched_hits[0].part_id == 20
    assert result.prefetched_hits[0].alpha_hint == pytest.approx(0.6)
    np.testing.assert_array_equal(result.prefetched_hits[0].position, [1.5, 2.0])
    expected = {
        "on_boundary_promoted_inside_count": 0,
        "etd2_midpoint_outside_count": 0,
        "boundary_far_skip_count": 0,
        "boundary_near_check_count": 2,
        "edge_prefetch_batch_candidate_count": 1,
        "edge_prefetch_batch_hit_count": 1,
        "boundary_exact_solve_count": 2,
        "boundary_broad_phase_checked_count": 3,
        "boundary_broad_phase_candidate_count": 2,
        "boundary_broad_phase_pruned_count": 1,
        "boundary_broad_phase_missed_hit_count": 0,
        "boundary_broad_phase_unknown_count": 0,
        "boundary_broad_phase_candidate_ratio": pytest.approx(2.0 / 3.0),
    }
    assert all(diagnostics[key] == value for key, value in expected.items())


def test_boundary_loop_builder_preserves_component_order_and_ccw_winding() -> None:
    edges = np.asarray(
        [
            [[3.0, 1.0], [2.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[3.0, 0.0], [3.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[2.0, 1.0], [2.0, 0.0]],
            [[1.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )

    loops = build_boundary_loops_2d(edges)

    assert len(loops) == 2
    np.testing.assert_array_equal(
        loops[0],
        [[3.0, 1.0], [2.0, 1.0], [2.0, 0.0], [3.0, 0.0]],
    )
    np.testing.assert_array_equal(
        loops[1],
        [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    )


def test_boundary_loop_builder_handles_empty_and_clockwise_input() -> None:
    assert build_boundary_loops_2d(np.empty((0, 2, 2), dtype=np.float64)) == ()
    clockwise_edges = np.asarray(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )

    (loop,) = build_boundary_loops_2d(clockwise_edges)

    np.testing.assert_array_equal(
        loop,
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
    )


def test_boundary_loop_builder_rejects_zero_length_edges() -> None:
    zero_length_edges = np.asarray(
        [[[1.0, 2.0], [1.0, 2.0]]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="at least one positive-length edge"):
        build_boundary_loops_2d(zero_length_edges)


def test_boundary_loop_builder_reports_dangling_topology() -> None:
    open_chain = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match=r"branch/dangling vertices \(v0:degree=1, v2:degree=1\)",
    ):
        build_boundary_loops_2d(open_chain)


def test_incomplete_boundary_walk_terminates_without_a_loop() -> None:
    edge_vertices = np.asarray([[0, 1]], dtype=np.int64)
    adjacency = {0: [0], 1: [0]}
    unused = np.asarray([True])

    loop_vertices = _trace_boundary_loop_vertices_2d(
        0,
        edge_vertices,
        adjacency,
        unused,
    )

    assert loop_vertices == [0, 1]
    assert _next_unused_edge_2d(adjacency[1], unused) is None
    assert (
        _normalized_loop_coordinates_2d(
            loop_vertices,
            [np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0])],
        )
        is None
    )
