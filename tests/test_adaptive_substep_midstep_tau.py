from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from field_backend_helpers import (
    geometry_provider,
    regular_field_provider,
)

from particle_tracer_unified import load_case, simulate
from particle_tracer_unified.domain import BoundaryHit
from particle_tracer_unified.solvers import high_fidelity_runtime, integrator_common
from particle_tracer_unified.solvers._collision_trial import (
    _advance_segment_with_inputs,
)
from particle_tracer_unified.solvers._collision_types import CollisionSegmentInputs
from particle_tracer_unified.solvers.collision_hit_localization import (
    locate_physical_hit_state,
)
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.high_fidelity_runtime import (
    _update_adaptive_substep_diagnostics,
)
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_STOKES,
    stokes_relaxation_time,
)
from particle_tracer_unified.solvers.motion_kernel_numba import (
    advance_etd2_batch_inplace,
)
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    trace_motion_batch,
    trace_motion_segment,
)


def _python(function: Any) -> Any:
    return function.py_func


def _epstein_request(
    density_along_x: np.ndarray,
    *,
    flow_along_x: np.ndarray | None = None,
) -> SegmentMotionBatchRequest:
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    density = np.repeat(
        np.asarray(density_along_x, dtype=np.float64)[:, None],
        3,
        axis=1,
    )
    flow = np.ones(3, dtype=np.float64)
    if flow_along_x is not None:
        flow = np.asarray(flow_along_x, dtype=np.float64)
    field = regular_field_provider(
        axes,
        valid_mask,
        quantities={
            "ux": np.repeat(flow[:, None], 3, axis=1),
            "uy": np.zeros((3, 3), dtype=np.float64),
            "rho_g": density,
            "mu": np.full((3, 3), 1.8e-5, dtype=np.float64),
            "T": np.full((3, 3), 300.0, dtype=np.float64),
        },
    )
    geometry = geometry_provider(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(
            np.zeros((3, 3), dtype=np.float64),
            np.ones((3, 3), dtype=np.float64),
        ),
    )
    backend = compile_runtime_backend(
        SimpleNamespace(
            geometry_provider=geometry,
            field_provider=field,
            gas=SimpleNamespace(
                density_kgm3=1.0e-6,
                dynamic_viscosity_Pas=1.8e-5,
                temperature=300.0,
            ),
        ),
        spatial_dim=2,
    )
    diameter = 1.0e-6
    mass = 1.0e-15
    particle_density = 6.0 * mass / (np.pi * diameter**3)
    return SegmentMotionBatchRequest(
        position_m=np.asarray([[0.0, 0.5]], dtype=np.float64),
        velocity_mps=np.asarray([[1.0, 0.0]], dtype=np.float64),
        active=np.asarray([True]),
        tau_stokes_s=np.asarray([10.0], dtype=np.float64),
        particle_diameter_m=np.asarray([diameter], dtype=np.float64),
        particle_density_kgm3=np.asarray([particle_density], dtype=np.float64),
        particle_mass_kg=np.asarray([mass], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coefficient=np.asarray([np.nan], dtype=np.float64),
        end_time_s=1.0,
        duration_s=1.0,
        spatial_dim=2,
        backend=backend,
        body_acceleration_mps2=np.zeros(2, dtype=np.float64),
        gas_density_kgm3=1.0e-6,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=4.65e-26,
        drag_model_mode=int(DRAG_MODEL_EPSTEIN),
        adaptive_substep_enabled=1,
        adaptive_substep_max_splits=4,
    )


def _transient_stokes_request() -> SegmentMotionBatchRequest:
    request = _epstein_request(np.full(3, 1.0e-6, dtype=np.float64))
    backend = request.backend
    assert isinstance(backend, RegularRectilinearCompiledBackend)
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    x_grid = np.broadcast_to(backend.axes[0][:, None], backend.valid_mask.shape)
    flow = np.stack(
        [0.15 * (1.0 + time_s) * (1.0 + x_grid) for time_s in times],
        axis=0,
    )

    def repeat_time(values: np.ndarray) -> np.ndarray:
        return np.repeat(values, times.size, axis=0)

    return replace(
        request,
        position_m=np.asarray([[0.1, 0.5]], dtype=np.float64),
        velocity_mps=np.asarray([[0.05, 0.0]], dtype=np.float64),
        tau_stokes_s=np.asarray([0.2], dtype=np.float64),
        drag_model_mode=int(DRAG_MODEL_STOKES),
        backend=replace(
            backend,
            times=times,
            ux=flow,
            uy=np.zeros_like(flow),
            gas_density=repeat_time(backend.gas_density),
            gas_mu=repeat_time(backend.gas_mu),
            gas_temperature=repeat_time(backend.gas_temperature),
        ),
    )


def _stiff_time_linear_stokes_request() -> SegmentMotionBatchRequest:
    request = _transient_stokes_request()
    backend = request.backend
    assert isinstance(backend, RegularRectilinearCompiledBackend)
    assert backend.times is not None
    target_tau_s = 0.005
    local_mu_pas = float(request.particle_mass_kg[0]) / (
        3.0 * np.pi * float(request.particle_diameter_m[0]) * target_tau_s
    )
    flow = np.broadcast_to(
        0.3 * backend.times[:, None, None],
        (backend.times.size, *backend.valid_mask.shape),
    ).copy()
    return replace(
        request,
        velocity_mps=np.asarray([[0.2, 0.0]], dtype=np.float64),
        tau_stokes_s=np.asarray([target_tau_s], dtype=np.float64),
        gas_dynamic_viscosity_Pas=local_mu_pas,
        backend=replace(
            backend,
            ux=flow,
            gas_mu=np.full_like(backend.gas_mu, local_mu_pas),
        ),
    )


def test_local_error_uses_state_scale_and_ulp_roundoff_allowance() -> None:
    position_error = _python(integrator_common.etd2_position_error_exceeds_tolerance)
    velocity_error = _python(integrator_common.etd2_velocity_error_exceeds_tolerance)

    assert not position_error(1.0, 1.1, 1.1, 0.1, 0.1, 0.1, 1.0)
    assert not velocity_error(1.0, 1.0, np.nextafter(1.0, np.inf))
    # ETD2 has global order p=2, so a full-vs-two-half difference estimates
    # only (1 - 2**-p) = 3/4 of the accepted coarse step's local error.
    assert velocity_error(1.0, 1.0, 1.0 + 8.0e-6)
    assert position_error(1.0e6, 1.0e6 + 0.1, 1.0e6 + 0.1001, 0.1, 0.1, 0.1, 1.0)
    assert velocity_error(1.0, 1.1, 1.1001)
    assert position_error(0.0, np.nan, 0.0, 0.0, 0.0, 0.0, 1.0)

    advance_affine = _python(integrator_common.advance_affine_stage_component)
    displacement, velocity = advance_affine(
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        2.0,
        1.0,
        0.5,
        0.25,
    )
    tau_quarter = 1.0 / (0.5 / 2.0 + 0.5 / 1.0)
    decay = np.exp(-0.25 / tau_quarter)
    np.testing.assert_allclose(velocity, 1.0 - decay, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(
        displacement,
        0.25 - tau_quarter * (1.0 - decay),
        rtol=2.0e-15,
        atol=0.0,
    )


def test_constant_tau_keeps_the_original_single_substep_bit_exact() -> None:
    request = _epstein_request(np.full(3, 1.0e-6, dtype=np.float64))
    adaptive = trace_motion_batch(request)
    disabled = trace_motion_batch(replace(request, adaptive_substep_enabled=0))
    scalar = trace_motion_segment(request.particle_request(0))

    assert adaptive.substep_count.tolist() == [1]
    assert disabled.substep_count.tolist() == [1]
    assert scalar.substep_count == 1
    assert adaptive.local_error_resolved.tolist() == [True]
    assert disabled.local_error_resolved.tolist() == [True]
    assert scalar.local_error_resolved is True
    np.testing.assert_array_equal(
        adaptive.endpoint_position_m,
        disabled.endpoint_position_m,
    )
    np.testing.assert_array_equal(
        adaptive.endpoint_velocity_mps,
        disabled.endpoint_velocity_mps,
    )
    np.testing.assert_array_equal(
        adaptive.endpoint_position_m[0],
        scalar.endpoint_position_m,
    )
    np.testing.assert_array_equal(
        adaptive.endpoint_velocity_mps[0],
        scalar.endpoint_velocity_mps,
    )


def test_stiff_linear_transient_flow_is_integrated_without_order_reduction() -> None:
    request = _stiff_time_linear_stokes_request()
    scalar = trace_motion_segment(request.particle_request(0))
    batch = trace_motion_batch(request)
    disabled_request = replace(request, adaptive_substep_enabled=0)
    disabled_scalar = trace_motion_segment(disabled_request.particle_request(0))
    disabled_batch = trace_motion_batch(disabled_request)

    duration = request.duration_s
    assert isinstance(request.backend, RegularRectilinearCompiledBackend)
    local_mu_pas = float(request.backend.gas_mu.flat[0])
    tau = stokes_relaxation_time(
        float(request.particle_mass_kg[0]),
        local_mu_pas,
        float(request.particle_diameter_m[0]),
    )
    np.testing.assert_allclose(request.tau_stokes_s[0], tau, rtol=2.0e-16)
    initial_velocity = float(request.velocity_mps[0, 0])
    flow_slope = 0.3

    def exact_state(elapsed: float) -> tuple[float, float]:
        decay = np.exp(-elapsed / tau)
        velocity = (
            flow_slope * (elapsed - tau) + (initial_velocity + flow_slope * tau) * decay
        )
        displacement = (
            0.5 * flow_slope * elapsed**2
            - flow_slope * tau * elapsed
            + (initial_velocity + flow_slope * tau) * tau * (1.0 - decay)
        )
        return request.position_m[0, 0] + displacement, velocity

    exact_position, exact_velocity = exact_state(duration)
    exact_mid_position, exact_mid_velocity = exact_state(0.5 * duration)

    assert scalar.substep_count == 1
    assert scalar.local_error_resolved is True
    np.testing.assert_allclose(
        scalar.endpoint_position_m[0],
        exact_position,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        scalar.endpoint_velocity_mps[0],
        exact_velocity,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_array_equal(
        batch.endpoint_position_m[0], scalar.endpoint_position_m
    )
    np.testing.assert_array_equal(
        batch.endpoint_velocity_mps[0], scalar.endpoint_velocity_mps
    )
    np.testing.assert_allclose(
        disabled_scalar.positions_m[0, 0],
        exact_mid_position,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        disabled_scalar.velocities_mps[0, 0],
        exact_mid_velocity,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_array_equal(
        disabled_batch.midpoint_position_m[0], disabled_scalar.positions_m[0]
    )
    assert not np.array_equal(
        disabled_scalar.coefficient_midpoint_positions_m[0],
        disabled_scalar.positions_m[0],
    )
    collision_inputs = CollisionSegmentInputs(
        spatial_dim=request.spatial_dim,
        compiled=request.backend,
        adaptive_substep_max_splits=request.adaptive_substep_max_splits,
        tau_p_i=tau,
        particle_diameter_i=float(request.particle_diameter_m[0]),
        particle_density_i=float(request.particle_density_kgm3[0]),
        particle_mass_i=float(request.particle_mass_kg[0]),
        dep_particle_rel_permittivity_i=float(request.dep_particle_rel_permittivity[0]),
        thermophoretic_coeff_i=float(request.thermophoretic_coefficient[0]),
        body_accel=request.body_acceleration_mps2,
        gas_density_kgm3=request.gas_density_kgm3,
        gas_mu_pas=request.gas_dynamic_viscosity_Pas,
        gas_temperature_K=request.gas_temperature_K,
        gas_molecular_mass_kg=request.gas_molecular_mass_kg,
        drag_model_mode=request.drag_model_mode,
    )
    collision_motion = _advance_segment_with_inputs(
        inputs=collision_inputs,
        x0=request.position_m[0],
        v0=request.velocity_mps[0],
        dt_segment=duration,
        t_end_segment=request.end_time_s,
        adaptive_substep_enabled=0,
    )
    np.testing.assert_array_equal(
        collision_motion[3][0], disabled_scalar.positions_m[0]
    )


def test_local_error_refines_scalar_and_compiled_batch_to_the_cap() -> None:
    request = replace(
        _epstein_request(np.asarray([1.0e-6, 1.0, 1.0], dtype=np.float64)),
        velocity_mps=np.zeros((1, 2), dtype=np.float64),
    )

    batch = trace_motion_batch(request)
    scalar = trace_motion_segment(request.particle_request(0))

    assert batch.substep_count.tolist() == [16]
    assert scalar.substep_count == 16
    assert batch.local_error_resolved.tolist() == [False]
    assert scalar.local_error_resolved is False
    np.testing.assert_allclose(
        batch.endpoint_position_m[0],
        scalar.endpoint_position_m,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        batch.endpoint_velocity_mps[0],
        scalar.endpoint_velocity_mps,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    assert getattr(advance_etd2_batch_inplace, "nopython_signatures", ())

    diagnostics: dict[str, object] = {}
    _update_adaptive_substep_diagnostics(
        diagnostics,
        adaptive_substep_enabled=1,
        adaptive_substep_max_splits=4,
        active=request.active,
        substep_counts=batch.substep_count,
    )
    assert diagnostics == {
        "adaptive_substep_segments_count": 16,
        "adaptive_substep_trigger_count": 1,
        "adaptive_substep_limit_reached_count": 1,
    }


def test_unresolved_local_error_stops_before_state_commit(
    monkeypatch,
) -> None:
    original_trace_motion_batch = high_fidelity_runtime.trace_motion_batch

    def unresolved_trace(request, destination=None):
        trace = original_trace_motion_batch(request, destination)
        trace.local_error_resolved[np.asarray(request.active, dtype=bool)] = False
        return trace

    monkeypatch.setattr(
        high_fidelity_runtime,
        "trace_motion_batch",
        unresolved_trace,
    )
    result = simulate(load_case(Path("examples/v02_minimal/run_config.yaml")))

    assert result.state.terminal_state.tolist() == [
        "numerical_boundary_stopped",
        "numerical_boundary_stopped",
    ]
    np.testing.assert_array_equal(
        result.state.position_m,
        [[-0.5, -0.2], [-0.5, 0.2]],
    )
    np.testing.assert_array_equal(result.state.velocity_mps, [[0.1, 0.0], [0.1, 0.0]])
    assert result.stats.safety_counters["unresolved_crossing_count"] == 2


def test_smooth_local_error_accepts_before_the_split_cap_and_improves_error() -> None:
    request = replace(
        _epstein_request(np.asarray([1.0e-6, 1.0e-5, 1.0e-4], dtype=np.float64)),
        velocity_mps=np.zeros((1, 2), dtype=np.float64),
        duration_s=0.01,
        end_time_s=0.01,
    )
    adaptive = trace_motion_batch(request)
    scalar = trace_motion_segment(request.particle_request(0))
    coarse = trace_motion_segment(
        replace(request.particle_request(0), adaptive_substep_enabled=0)
    )
    reference = trace_motion_segment(
        replace(
            request.particle_request(0), adaptive_substep_enabled=0
        ).with_minimum_substeps(16)
    )

    assert 1 < adaptive.substep_count[0] < 16
    assert scalar.substep_count == adaptive.substep_count[0]
    assert adaptive.local_error_resolved.tolist() == [True]
    assert scalar.local_error_resolved is True
    np.testing.assert_array_equal(
        adaptive.endpoint_position_m[0], scalar.endpoint_position_m
    )
    np.testing.assert_array_equal(
        adaptive.endpoint_velocity_mps[0], scalar.endpoint_velocity_mps
    )
    assert np.linalg.norm(
        scalar.endpoint_position_m - reference.endpoint_position_m
    ) < np.linalg.norm(coarse.endpoint_position_m - reference.endpoint_position_m)
    assert np.linalg.norm(
        scalar.endpoint_velocity_mps - reference.endpoint_velocity_mps
    ) < np.linalg.norm(coarse.endpoint_velocity_mps - reference.endpoint_velocity_mps)


def test_stokes_drag_refines_for_a_spatially_varying_flow_field() -> None:
    request = replace(
        _epstein_request(
            np.full(3, 1.0e-6, dtype=np.float64),
            flow_along_x=np.asarray([0.0, 5.0, 10.0], dtype=np.float64),
        ),
        tau_stokes_s=np.asarray([0.2], dtype=np.float64),
        drag_model_mode=int(DRAG_MODEL_STOKES),
    )
    adaptive = trace_motion_batch(request)
    scalar = trace_motion_segment(request.particle_request(0))
    coarse = trace_motion_batch(replace(request, adaptive_substep_enabled=0))

    assert 1 < adaptive.substep_count[0] <= 16
    assert adaptive.substep_count[0] == scalar.substep_count
    np.testing.assert_array_equal(
        adaptive.endpoint_position_m[0], scalar.endpoint_position_m
    )
    np.testing.assert_array_equal(
        adaptive.endpoint_velocity_mps[0], scalar.endpoint_velocity_mps
    )
    assert not np.array_equal(adaptive.endpoint_position_m, coarse.endpoint_position_m)


def test_transient_dense_state_is_continuous_and_localizes_the_same_hit() -> None:
    request = _transient_stokes_request().particle_request(0)
    trace = trace_motion_segment(request)
    reference = trace_motion_segment(
        replace(
            request,
            adaptive_substep_enabled=0,
            adaptive_substep_max_splits=10,
        ).with_minimum_substeps(256)
    )

    midpoint_time = request.duration_s / (2.0 * trace.substep_count)
    accepted_midpoint = trace.positions_m[0]
    coarse_schedule = trace_motion_segment(
        replace(request, adaptive_substep_enabled=0).with_minimum_substeps(
            trace.substep_count
        )
    )
    position_at_midpoint, _ = trace.state_at(midpoint_time)
    left_position, _ = trace.state_at(np.nextafter(midpoint_time, 0.0))
    right_position, _ = trace.state_at(np.nextafter(midpoint_time, request.duration_s))
    assert not np.array_equal(accepted_midpoint, coarse_schedule.positions_m[0])
    np.testing.assert_array_equal(position_at_midpoint, accepted_midpoint)
    np.testing.assert_allclose(left_position, position_at_midpoint, atol=2.0e-15)
    np.testing.assert_allclose(right_position, position_at_midpoint, atol=2.0e-15)

    target_time = 0.43 * request.duration_s
    boundary_position, _ = reference.state_at(target_time)
    primary = BoundaryHit(
        position=boundary_position,
        normal=np.asarray([1.0, 0.0], dtype=np.float64),
        part_id=7,
        alpha_hint=target_time / request.duration_s,
        primitive_id=1,
        primitive_kind="edge",
    )

    def time_tolerance(
        reference_time: float, interval: float, fraction: float
    ) -> float:
        magnitude = max(abs(reference_time), abs(interval))
        return max(fraction * interval, 64.0 * abs(float(np.spacing(magnitude))))

    def locate(state_at, stage_times):
        return locate_physical_hit_state(
            x0=request.position_m,
            v0=request.velocity_mps,
            segment_dt=request.duration_s,
            t_end_segment=request.end_time_s,
            stage_times=stage_times,
            primary_hit=primary,
            strict_inside_fn=lambda point: bool(point[0] < boundary_position[0]),
            nearest_projection_fn=lambda _point, _inside: primary,
            state_at=state_at,
            time_tolerance=time_tolerance,
            on_boundary_tol_m=1.0e-12,
        )

    event = locate(trace.state_at, trace.times_s - request.start_time_s)
    reference_event = locate(
        reference.state_at,
        reference.times_s - request.start_time_s,
    )
    assert event is not None
    assert reference_event is not None
    assert abs(event[2] - reference_event[2]) <= 3.0e-4 * request.duration_s
