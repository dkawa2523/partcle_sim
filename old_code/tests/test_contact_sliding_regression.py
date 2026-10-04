from __future__ import annotations

from dataclasses import replace
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from particle_tracer_unified import load_case
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.solvers import (
    _contact_dynamics,
    _contact_geometry,
    _contact_sliding_2d,
    _contact_sliding_3d,
    _contact_state,
    _runtime_valid_mask,
    contact_sliding,
)
from particle_tracer_unified.solvers.contact_sliding import (
    ContactDynamicsBatch,
    advance_contact_sliding_particles,
)
from particle_tracer_unified.solvers.forces.runtime import ForceRuntimeParameters
from particle_tracer_unified.solvers.high_fidelity_runtime import _sample_runtime_stage
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_NONE,
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
    drag_model_name_from_mode,
)
from particle_tracer_unified.solvers.runtime_execution import prepare_runtime_execution

ROOT = Path(__file__).resolve().parents[1]


def test_contact_sliding_public_api_is_directly_owned() -> None:
    assert (
        contact_sliding.ContactDynamicsBatch is _contact_dynamics.ContactDynamicsBatch
    )
    assert (
        contact_sliding.advance_contact_relaxation
        is _contact_dynamics.advance_contact_relaxation
    )
    assert (
        contact_sliding.displaced_fluid_factors
        is _contact_dynamics.displaced_fluid_factors
    )
    assert tuple(
        signature(contact_sliding.advance_contact_sliding_particles).parameters
    ) == (
        "execution",
        "body_acceleration",
        "duration_s",
        "time_s",
        "electric_q_over_m_particle",
        "sample_stage",
    )


def _execution_3d():
    case = load_case(ROOT / "examples/v02_minimal_3d/run_config.yaml")
    debug_plan = replace(
        case._context.plan,
        output=replace(case._context.plan.output, mode="debug"),
    )
    execution = prepare_runtime_execution(
        replace(case._context, plan=debug_plan),
        spatial_dim=3,
        plan=debug_plan,
        debug_buffers=None,
    )
    surface = execution.boundary_service.triangle_surface_3d
    assert surface is not None
    return execution, surface


def _execution_2d():
    case = load_case(ROOT / "examples/v02_minimal/run_config.yaml")
    debug_plan = replace(
        case._context.plan,
        output=replace(case._context.plan.output, mode="debug"),
    )
    return prepare_runtime_execution(
        replace(case._context, plan=debug_plan),
        spatial_dim=2,
        plan=debug_plan,
        debug_buffers=None,
    )


def _set_2d_contacts(execution) -> None:
    state = execution.state
    epsilon = float(execution.plan.boundary.contact_offset_m)
    state.active[:] = True
    state.released[:] = True
    state.contact_sliding[:] = True
    state.contact_endpoint_stopped[:] = False
    state.contact_edge_index[:] = 0
    state.contact_part_id[:] = 10
    state.contact_normal[:] = np.asarray([0.0, -1.0])
    state.x[:] = np.asarray([[-0.5, -1.0 + epsilon], [0.5, -1.0 + epsilon]])
    state.v[:] = np.asarray([[0.1, 0.0], [0.2, 0.0]])


def _contact_dynamics_execution(drag_model_mode: int):
    execution, _ = _execution_3d()
    plan = replace(
        execution.plan,
        drag_model_mode=drag_model_mode,
        drag_model_name=drag_model_name_from_mode(drag_model_mode),
    )
    return replace(execution, plan=plan)


def test_contact_sliding_dispatch_rejects_unknown_dimension_before_sampling() -> None:
    execution = replace(_execution_2d(), spatial_dim=1)

    with pytest.raises(
        ValueError,
        match="contact sliding requires spatial_dim 2 or 3, got 1",
    ):
        contact_sliding.advance_contact_sliding_particles(
            execution,
            body_acceleration=np.zeros(1),
            duration_s=0.1,
            time_s=0.1,
            electric_q_over_m_particle=None,
            sample_stage=cast(Any, None),
        )


def test_contact_dynamics_defaults_and_small_ratio_remain_finite() -> None:
    density = np.asarray([10.0, 20.0], dtype=np.float64)
    for force_runtime in (None, ForceRuntimeParameters()):
        gravity, inertia = _contact_dynamics.displaced_fluid_factors(
            force_runtime,
            np.asarray([1.0, 2.0], dtype=np.float64),
            density,
        )
        np.testing.assert_array_equal(gravity, np.ones(2))
        np.testing.assert_array_equal(inertia, np.ones(2))

    displacement, velocity = _contact_dynamics.advance_contact_relaxation(
        np.asarray([1.0]),
        np.asarray([2.0]),
        np.asarray([3.0]),
        np.asarray([1.0e6]),
        1.0e-6,
    )
    assert displacement.dtype == np.dtype(np.float64)
    assert velocity.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(displacement))
    assert np.all(np.isfinite(velocity))


def test_contact_dynamics_stokes_preserves_force_and_inertia_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _contact_dynamics_execution(DRAG_MODEL_STOKES)
    indices = np.asarray([0], dtype=np.int64)
    sampled_target = np.asarray([[0.5, -0.25, 0.75]], dtype=np.float64)
    sampled_acceleration = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    sampled_viscosity = np.asarray(
        [3.0 * float(execution.physics["gas_mu_pas"])],
        dtype=np.float64,
    )
    sample_options: dict[str, object] = {}

    def sample_stage(*_args: object, **kwargs: object) -> Any:
        sample_options.update(kwargs)
        return SimpleNamespace(
            values={
                _contact_dynamics.FLOW_VELOCITY: sampled_target,
                _contact_dynamics.DYNAMIC_VISCOSITY: sampled_viscosity,
            }
        )

    monkeypatch.setattr(
        _contact_dynamics,
        "sample_compiled_acceleration_vectors",
        lambda *_args, **_kwargs: sampled_acceleration,
    )
    monkeypatch.setattr(
        _contact_dynamics,
        "displaced_fluid_factors",
        lambda *_args, **_kwargs: (
            np.asarray([0.5], dtype=np.float64),
            np.asarray([2.0], dtype=np.float64),
        ),
    )
    body = np.asarray([2.0, 4.0, 6.0], dtype=np.float64)

    result = _contact_dynamics._evaluate_contact_dynamics(
        execution,
        indices=indices,
        contact_position=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        velocity=np.asarray([[0.4, 0.5, 0.6]], dtype=np.float64),
        body_acceleration=body,
        time_s=0.25,
        electric_q_over_m_particle=None,
        sample_stage=sample_stage,
    )

    np.testing.assert_array_equal(result.target_velocity, sampled_target)
    np.testing.assert_array_equal(
        result.body_acceleration,
        (0.5 * body[None, :] + sampled_acceleration) / 2.0,
    )
    np.testing.assert_allclose(
        result.relaxation_time_s,
        np.asarray(execution.tau_p[indices], dtype=np.float64) * (2.0 / 3.0),
        rtol=3.0e-16,
        atol=0.0,
    )
    assert sample_options["need_flow"] is True
    assert sample_options["need_gas_mu"] is True
    assert sample_options["need_valid_mask"] is False


def test_contact_dynamics_nonlinear_drag_uses_local_gas_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _contact_dynamics_execution(DRAG_MODEL_SCHILLER_NAUMANN)
    indices = np.asarray([0], dtype=np.int64)
    target = np.asarray([[0.5, 0.25, -0.5]], dtype=np.float64)
    velocity = np.asarray([[1.5, 0.25, -0.5]], dtype=np.float64)
    effective_arguments: list[tuple[float, ...]] = []

    def sample_stage(*_args: object, **_kwargs: object) -> Any:
        return SimpleNamespace(
            values={
                _contact_dynamics.FLOW_VELOCITY: target,
                _contact_dynamics.GAS_DENSITY: np.asarray([2.0]),
                _contact_dynamics.DYNAMIC_VISCOSITY: np.asarray([3.0]),
                _contact_dynamics.TEMPERATURE: np.asarray([400.0]),
            }
        )

    def effective_tau(*arguments: float) -> float:
        effective_arguments.append(arguments)
        return 7.0

    monkeypatch.setattr(
        _contact_dynamics,
        "sample_compiled_acceleration_vectors",
        lambda *_args, **_kwargs: np.zeros((1, 3), dtype=np.float64),
    )
    monkeypatch.setattr(
        _contact_dynamics,
        "displaced_fluid_factors",
        lambda *_args, **_kwargs: (np.ones(1), np.ones(1)),
    )
    monkeypatch.setattr(
        _contact_dynamics, "effective_tau_from_drag_model", effective_tau
    )

    result = _contact_dynamics._evaluate_contact_dynamics(
        execution,
        indices=indices,
        contact_position=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        velocity=velocity,
        body_acceleration=np.zeros(3, dtype=np.float64),
        time_s=0.25,
        electric_q_over_m_particle=None,
        sample_stage=sample_stage,
    )

    assert len(effective_arguments) == 1
    arguments = effective_arguments[0]
    assert arguments[1] == pytest.approx(1.0)
    assert arguments[3] == pytest.approx(2.0)
    assert arguments[4] == pytest.approx(3.0)
    assert arguments[5] == DRAG_MODEL_SCHILLER_NAUMANN
    assert arguments[7] == pytest.approx(400.0)
    np.testing.assert_array_equal(result.relaxation_time_s, np.asarray([7.0]))


def test_contact_dynamics_without_drag_is_ballistic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _contact_dynamics_execution(DRAG_MODEL_NONE)
    sample_options: dict[str, object] = {}

    def sample_stage(*_args: object, **kwargs: object) -> Any:
        sample_options.update(kwargs)
        return SimpleNamespace(values={})

    monkeypatch.setattr(
        _contact_dynamics,
        "sample_compiled_acceleration_vectors",
        lambda *_args, **_kwargs: np.zeros((1, 3), dtype=np.float64),
    )
    monkeypatch.setattr(
        _contact_dynamics,
        "displaced_fluid_factors",
        lambda *_args, **_kwargs: (np.ones(1), np.ones(1)),
    )

    result = _contact_dynamics._evaluate_contact_dynamics(
        execution,
        indices=np.asarray([0], dtype=np.int64),
        contact_position=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        velocity=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        body_acceleration=np.zeros(3, dtype=np.float64),
        time_s=0.25,
        electric_q_over_m_particle=None,
        sample_stage=sample_stage,
    )

    np.testing.assert_array_equal(result.target_velocity, np.zeros((1, 3)))
    assert np.isposinf(result.relaxation_time_s[0])
    assert sample_options["need_flow"] is False


def test_contact_support_uses_sampled_status_then_compiled_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution_2d()
    points = np.zeros((2, 2), dtype=np.float64)
    inside = np.ones(2, dtype=bool)
    fallback_calls: list[np.ndarray] = []

    def fallback(_compiled: object, values: np.ndarray) -> np.ndarray:
        fallback_calls.append(values)
        return np.asarray([1, 2], dtype=np.uint8)

    monkeypatch.setattr(
        _contact_state,
        "sample_compiled_valid_mask_statuses",
        fallback,
    )
    sampled_status: np.ndarray | None = np.asarray(
        [0, VALID_MASK_STATUS_MIXED_STENCIL], dtype=np.uint8
    )

    def sample_stage(*_args: object, **_kwargs: object) -> StageFields:
        values: dict[str, np.ndarray] = {}
        if sampled_status is not None:
            values[_contact_state.VALID_MASK_STATUS] = sampled_status
        return StageFields(
            points_m=points,
            time_s=0.0,
            values=values,
            supported=np.ones(2, dtype=bool),
        )

    sampled = _contact_state._clean_support(
        execution,
        points=points,
        time_s=0.0,
        inside=inside,
        sample_stage=sample_stage,
    )
    np.testing.assert_array_equal(sampled, np.asarray([True, False]))
    assert not fallback_calls

    sampled_status = None
    fallback_result = _contact_state._clean_support(
        execution,
        points=points,
        time_s=0.0,
        inside=inside,
        sample_stage=sample_stage,
    )
    np.testing.assert_array_equal(fallback_result, np.asarray([False, False]))
    assert len(fallback_calls) == 1
    assert fallback_calls[0] is points


def test_freeflight_retries_mixed_and_hard_valid_mask_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution_2d()
    state = execution.state
    state.active[:] = True
    state.valid_mask_status_flags[:] = np.asarray(
        [VALID_MASK_STATUS_MIXED_STENCIL, VALID_MASK_STATUS_HARD_INVALID],
        dtype=np.uint8,
    )
    retried_positions: list[np.ndarray] = []

    def retry(request, **_kwargs):
        retried_positions.append(np.asarray(request.position_m).copy())
        return SimpleNamespace(
            position=np.asarray(request.position_m).copy(),
            velocity=np.asarray(request.velocity_mps).copy(),
            accepted_dt=0.0,
            found_valid_prefix=False,
        )

    monkeypatch.setattr(
        _runtime_valid_mask,
        "resolve_valid_mask_retry_then_stop",
        retry,
    )
    terminal_outcomes = {}

    stopped = _runtime_valid_mask.apply_valid_mask_retry_then_stop(
        execution,
        dt_step=0.1,
        t_end_step=0.1,
        adaptive_substep_enabled=0,
        terminal_outcomes=terminal_outcomes,
    )

    assert stopped == 2
    assert len(retried_positions) == 2
    assert not np.any(state.active)
    assert set(terminal_outcomes) == {0, 1}


def test_2d_contact_without_boundary_preserves_state_and_records_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution_2d()
    _set_2d_contacts(execution)
    state = execution.state
    positions_before = state.x.copy()
    velocities_before = state.v.copy()
    monkeypatch.setattr(
        _contact_geometry,
        "_boundary_edge_arrays_2d",
        lambda _runtime: (None, np.zeros(0, dtype=np.int32)),
    )

    _contact_sliding_2d.advance_contact_sliding_2d(
        execution,
        body_acceleration=np.zeros(2),
        duration_s=0.1,
        time_s=0.1,
        electric_q_over_m_particle=None,
        sample_stage=cast(Any, None),
    )

    np.testing.assert_array_equal(state.x, positions_before)
    np.testing.assert_array_equal(state.v, velocities_before)
    assert np.all(state.contact_sliding)
    assert state.collision_diagnostics["contact_frame_fail_count"] == 2


def test_2d_contact_geometry_repairs_and_filters_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution_2d()
    _set_2d_contacts(execution)
    state = execution.state
    segments, part_ids = _contact_geometry._boundary_edge_arrays_2d(execution.context)
    assert segments is not None
    assert part_ids.shape == (segments.shape[0],)

    no_edges = SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(boundary_edges=None),
        ),
    )
    missing, missing_part_ids = _contact_geometry._boundary_edge_arrays_2d(no_edges)
    assert missing is None
    assert missing_part_ids.shape == (0,)

    state.contact_edge_index[:] = -1
    repairs = iter((SimpleNamespace(edge_index=0), None))
    monkeypatch.setattr(
        _contact_geometry,
        "contact_frame_on_boundary_edge_2d",
        lambda *_args, **_kwargs: next(repairs),
    )
    diagnostics: dict[str, object] = {}
    repaired = _contact_geometry._repair_contact_edges_2d(
        execution,
        np.asarray([0, 1], dtype=np.int64),
        segments,
        diagnostics,
    )
    np.testing.assert_array_equal(repaired, np.asarray([0], dtype=np.int64))
    assert diagnostics["contact_frame_fail_count"] == 1

    test_segments = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    state.contact_edge_index[:] = np.asarray([0, 1], dtype=np.int32)
    state.contact_normal[:] = 0.0
    state.x[:] = np.asarray([[0.25, 0.5], [0.0, 0.0]], dtype=np.float64)
    frame = _contact_geometry._build_contact_frame_2d(
        execution,
        np.asarray([0, 1], dtype=np.int64),
        test_segments,
        diagnostics,
    )

    assert frame is not None
    np.testing.assert_array_equal(frame.indices, np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(frame.tangent, np.asarray([[1.0, 0.0]]))
    np.testing.assert_array_equal(frame.normal, np.asarray([[-0.0, 1.0]]))
    assert diagnostics["contact_frame_fail_count"] == 2


def test_2d_contact_release_and_endpoint_hold_preserve_state_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution_2d()
    _set_2d_contacts(execution)
    state = execution.state
    state.contact_endpoint_stopped[1] = True
    calls = 0

    monkeypatch.setattr(
        _contact_dynamics, "_compiled_has_transient_time", lambda _: True
    )
    monkeypatch.setattr(
        _contact_dynamics,
        "_evaluate_contact_dynamics",
        lambda *_args, **_kwargs: ContactDynamicsBatch(
            target_velocity=np.zeros((2, 2)),
            body_acceleration=np.zeros((2, 2)),
            relaxation_time_s=np.ones(2),
        ),
    )

    def relaxation(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return np.asarray([[0.0, 0.1], [0.0, 0.0]]), np.zeros((2, 2))

    monkeypatch.setattr(_contact_dynamics, "advance_contact_relaxation", relaxation)
    monkeypatch.setattr(
        _contact_state,
        "_clean_support",
        lambda *_args, **_kwargs: np.asarray([True]),
    )

    _contact_sliding_2d.advance_contact_sliding_2d(
        execution,
        body_acceleration=np.zeros(2),
        duration_s=0.1,
        time_s=0.1,
        electric_q_over_m_particle=None,
        sample_stage=cast(Any, None),
    )

    assert calls == 1
    assert not state.contact_sliding[0]
    assert state.contact_edge_index[0] == -1
    np.testing.assert_array_equal(state.v[0], np.asarray([0.1, 0.0]))
    assert state.contact_sliding[1]
    assert state.contact_endpoint_stopped[1]
    np.testing.assert_array_equal(state.v[1], np.zeros(2))
    assert state.collision_diagnostics["contact_release_count"] == 1
    assert state.collision_diagnostics["contact_endpoint_hold_count"] == 1


def test_2d_contact_tangent_reject_and_endpoint_stop_preserve_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _execution_2d()
    _set_2d_contacts(execution)
    state = execution.state
    calls = 0

    monkeypatch.setattr(
        _contact_dynamics,
        "_evaluate_contact_dynamics",
        lambda *_args, **_kwargs: ContactDynamicsBatch(
            target_velocity=np.zeros((2, 2)),
            body_acceleration=np.zeros((2, 2)),
            relaxation_time_s=np.ones(2),
        ),
    )

    def relaxation(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.zeros((2, 2)), np.zeros((2, 2))
        return np.asarray([-0.2, 1.0]), np.asarray([-0.3, 0.4])

    monkeypatch.setattr(_contact_dynamics, "advance_contact_relaxation", relaxation)
    monkeypatch.setattr(
        _contact_state,
        "_clean_support",
        lambda *_args, **_kwargs: np.asarray([False, True]),
    )

    _contact_sliding_2d.advance_contact_sliding_2d(
        execution,
        body_acceleration=np.zeros(2),
        duration_s=0.1,
        time_s=0.1,
        electric_q_over_m_particle=None,
        sample_stage=cast(Any, None),
    )

    epsilon = float(execution.plan.boundary.contact_offset_m)
    assert calls == 2
    np.testing.assert_allclose(state.x[0], [-0.5, -1.0 + epsilon], rtol=0, atol=0)
    np.testing.assert_array_equal(state.v[0], np.zeros(2))
    np.testing.assert_allclose(state.x[1], [1.0, -1.0 + epsilon], rtol=0, atol=0)
    np.testing.assert_array_equal(state.v[1], np.zeros(2))
    assert not state.contact_endpoint_stopped[0]
    assert state.contact_endpoint_stopped[1]
    assert state.collision_diagnostics["contact_valid_mask_reject_count"] == 1
    assert state.collision_diagnostics["contact_endpoint_stop_count"] == 1
    assert state.collision_diagnostics["contact_tangent_step_count"] == 1


def test_3d_contact_sliding_preserves_state_when_surface_is_unavailable() -> None:
    execution, surface = _execution_3d()
    triangle_index = 0
    triangle = np.asarray(surface.triangles[triangle_index], dtype=np.float64)
    normal = np.asarray(surface.normals[triangle_index], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    centroid = np.mean(triangle, axis=0)
    epsilon = float(execution.plan.boundary.contact_offset_m)
    state = execution.state
    state.active[0] = True
    state.released[0] = True
    state.contact_sliding[0] = True
    state.contact_edge_index[0] = triangle_index
    state.contact_part_id[0] = int(surface.part_ids[triangle_index])
    state.contact_normal[0] = normal
    state.x[0] = centroid - epsilon * normal
    state.v[0] = 0.0
    position_before = state.x.copy()
    velocity_before = state.v.copy()
    missing_surface_execution = replace(
        execution,
        boundary_service=SimpleNamespace(triangle_surface_3d=None),
    )

    advance_contact_sliding_particles(
        missing_surface_execution,
        body_acceleration=np.zeros(3, dtype=np.float64),
        duration_s=1.0e-5,
        time_s=1.0e-5,
        electric_q_over_m_particle=None,
        sample_stage=_sample_runtime_stage,
    )

    assert state.active[0]
    assert state.contact_sliding[0]
    assert not state.contact_endpoint_stopped[0]
    assert state.contact_edge_index[0] == triangle_index
    assert state.contact_part_id[0] == int(surface.part_ids[triangle_index])
    assert state.x[0].shape == (3,)
    assert state.x[0].dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(state.x[0]))
    assert np.all(np.isfinite(state.v[0]))
    np.testing.assert_array_equal(state.x, position_before)
    np.testing.assert_array_equal(state.v, velocity_before)
    assert int(state.collision_diagnostics["contact_frame_fail_count"]) == 1
    assert int(state.collision_diagnostics["contact_valid_mask_reject_count"]) == 0
    assert int(state.collision_diagnostics["contact_tangent_step_count"]) == 0


def test_3d_contact_frame_repairs_projection_and_normal() -> None:
    execution, surface = _execution_3d()
    state = execution.state
    indices = np.asarray([0, 1], dtype=np.int64)
    triangle = np.asarray(surface.triangles[0], dtype=np.float64)
    state.x[indices] = np.mean(triangle, axis=0)
    state.contact_edge_index[indices] = -1
    calls = iter(
        [
            SimpleNamespace(
                primitive_id=0,
                part_id=int(surface.part_ids[0]),
                normal=np.asarray(surface.normals[0], dtype=np.float64),
            ),
            None,
        ]
    )
    boundary = SimpleNamespace(
        nearest_projection=lambda _start, _end: next(calls),
        triangle_surface_3d=surface,
    )
    repaired_execution = replace(execution, boundary_service=cast(Any, boundary))

    repaired = _contact_geometry._repair_contact_triangles_3d(
        repaired_execution,
        indices,
        int(surface.triangles.shape[0]),
        state.collision_diagnostics,
    )

    assert repaired.tolist() == [0]
    assert state.contact_part_id[0] == int(surface.part_ids[0])
    assert state.collision_diagnostics["contact_frame_fail_count"] == 1
    state.contact_normal[0] = 0.0
    frame = _contact_geometry._build_contact_frame_3d(
        execution,
        repaired,
        surface,
        state.collision_diagnostics,
    )
    assert frame is not None
    assert frame.indices.tolist() == [0]
    assert np.linalg.norm(frame.normal[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("stop_after", "expected_calls"),
    [
        ("active", []),
        ("repair", ["repair"]),
        ("frame", ["repair", "frame"]),
        ("release", ["repair", "frame", "dynamics", "release"]),
        ("hold", ["repair", "frame", "dynamics", "release", "hold"]),
        (
            None,
            ["repair", "frame", "dynamics", "release", "hold", "tangent"],
        ),
    ],
)
def test_3d_contact_owner_keeps_repair_release_hold_tangent_order(
    monkeypatch: pytest.MonkeyPatch,
    stop_after: str | None,
    expected_calls: list[str],
) -> None:
    execution, _surface = _execution_3d()
    indices = np.asarray([0], dtype=np.int64)
    frame = SimpleNamespace(
        indices=indices,
        x_contact=np.zeros((1, 3), dtype=np.float64),
        velocity_old=np.zeros((1, 3), dtype=np.float64),
    )
    dynamics = ContactDynamicsBatch(
        target_velocity=np.zeros((1, 3), dtype=np.float64),
        body_acceleration=np.zeros((1, 3), dtype=np.float64),
        relaxation_time_s=np.ones(1, dtype=np.float64),
    )
    calls: list[str] = []

    monkeypatch.setattr(
        _contact_state,
        "_active_contact_indices",
        lambda _execution: (
            np.zeros(0, dtype=np.int64) if stop_after == "active" else indices
        ),
    )

    def repair(*_args: object, **_kwargs: object) -> np.ndarray:
        calls.append("repair")
        return np.zeros(0, dtype=np.int64) if stop_after == "repair" else indices

    def build(*_args: object, **_kwargs: object) -> object:
        calls.append("frame")
        return None if stop_after == "frame" else frame

    def evaluate(*_args: object, **_kwargs: object) -> ContactDynamicsBatch:
        calls.append("dynamics")
        return dynamics

    def release(*_args: object, **_kwargs: object) -> np.ndarray:
        calls.append("release")
        return np.asarray([stop_after == "release"])

    def hold(*_args: object, **_kwargs: object) -> np.ndarray:
        calls.append("hold")
        return np.asarray([stop_after != "hold"])

    def tangent(*_args: object, **_kwargs: object) -> None:
        calls.append("tangent")

    monkeypatch.setattr(_contact_geometry, "_repair_contact_triangles_3d", repair)
    monkeypatch.setattr(_contact_geometry, "_build_contact_frame_3d", build)
    monkeypatch.setattr(_contact_dynamics, "_evaluate_contact_dynamics", evaluate)
    monkeypatch.setattr(_contact_sliding_3d, "_release_contacts_3d", release)
    monkeypatch.setattr(_contact_sliding_3d, "_hold_contact_endpoints_3d", hold)
    monkeypatch.setattr(_contact_sliding_3d, "_advance_contact_tangent_3d", tangent)

    _contact_sliding_3d.advance_contact_sliding_3d(
        execution,
        body_acceleration=np.zeros(3),
        duration_s=0.1,
        time_s=0.1,
        electric_q_over_m_particle=None,
        sample_stage=cast(Any, None),
    )

    assert calls == expected_calls


def test_3d_triangle_membership_preserves_boundary_tolerance() -> None:
    triangles = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    points = np.asarray([[0.25, 0.25, 0.0], [2.0, 2.0, 0.0]], dtype=np.float64)

    inside = _contact_sliding_3d._inside_contact_triangles(points, triangles)

    np.testing.assert_array_equal(inside, np.asarray([True, False]))


def test_3d_contact_release_hold_and_tangent_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution, surface = _execution_3d()
    state = execution.state
    indices = np.asarray([0, 1], dtype=np.int64)
    triangles = np.asarray(surface.triangles[[0, 0]], dtype=np.float64)
    normals = np.asarray(surface.normals[[0, 0]], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    wall = np.mean(triangles, axis=1)
    epsilon = float(execution.plan.boundary.contact_offset_m)
    frame = _contact_geometry._ContactFrame3D(
        indices=indices,
        triangle_index=np.asarray([0, 0], dtype=np.int64),
        triangles=triangles,
        q0=triangles[:, 0],
        normal=normals,
        x_wall=wall,
        x_contact=wall - epsilon * normals,
        velocity_old=np.zeros((2, 3), dtype=np.float64),
        tangent_velocity_old=np.zeros((2, 3), dtype=np.float64),
    )
    dynamics = ContactDynamicsBatch(
        target_velocity=np.zeros((2, 3), dtype=np.float64),
        body_acceleration=np.zeros((2, 3), dtype=np.float64),
        relaxation_time_s=np.ones(2, dtype=np.float64),
    )
    state.contact_sliding[indices] = True
    state.contact_endpoint_stopped[1] = True
    state.contact_edge_index[indices] = 0
    state.contact_part_id[indices] = int(surface.part_ids[0])
    state.contact_normal[indices] = normals
    boundary = SimpleNamespace(
        inside=lambda _point: True,
        triangle_surface_3d=surface,
    )
    helper_execution = replace(execution, boundary_service=cast(Any, boundary))

    monkeypatch.setattr(
        _contact_dynamics,
        "advance_contact_relaxation",
        lambda *_args, **_kwargs: (-normals.copy(), np.ones((2, 3))),
    )
    monkeypatch.setattr(
        _contact_state,
        "_clean_support",
        lambda *_args, **_kwargs: np.asarray([True, False]),
    )
    release = _contact_sliding_3d._release_contacts_3d(
        helper_execution,
        frame,
        dynamics,
        duration_s=0.1,
        time_s=0.1,
        sample_stage=_sample_runtime_stage,
    )
    assert release.tolist() == [True, False]
    assert not state.contact_sliding[0]
    mobile = _contact_sliding_3d._hold_contact_endpoints_3d(
        helper_execution,
        frame,
        ~release,
        surface,
    )
    assert mobile.tolist() == [False, False]
    assert state.collision_diagnostics["contact_endpoint_hold_count"] == 1

    state.contact_endpoint_stopped[indices] = False
    monkeypatch.setattr(
        _contact_dynamics,
        "advance_contact_relaxation",
        lambda *_args, **_kwargs: (np.zeros((2, 3)), np.ones((2, 3))),
    )
    monkeypatch.setattr(
        _contact_sliding_3d,
        "point_triangle_barycentric",
        lambda point, _triangle: (
            None if np.allclose(point, wall[1]) else np.asarray([1 / 3, 1 / 3, 1 / 3])
        ),
    )
    monkeypatch.setattr(
        _contact_state,
        "_clean_support",
        lambda *_args, **_kwargs: np.asarray([False, True]),
    )
    _contact_sliding_3d._advance_contact_tangent_3d(
        helper_execution,
        frame,
        dynamics,
        np.asarray([True, True]),
        surface,
        duration_s=0.1,
        time_s=0.1,
        sample_stage=_sample_runtime_stage,
    )
    assert state.collision_diagnostics["contact_valid_mask_reject_count"] == 1
    assert state.collision_diagnostics["contact_endpoint_stop_count"] == 1
    assert state.collision_diagnostics["contact_tangent_step_count"] == 1
