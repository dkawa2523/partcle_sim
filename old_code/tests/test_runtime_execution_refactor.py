from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from particle_tracer_unified import load_case
from particle_tracer_unified.solvers import (
    _runtime_execution_context,
    _runtime_outcome,
    _runtime_preparation,
    runtime_execution,
)
from particle_tracer_unified.solvers.compiled_backend_types import (
    CompiledRuntimeBackend,
)
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_NONE,
    DRAG_MODEL_STOKES,
    epstein_relaxation_time,
    stokes_relaxation_time,
)
from particle_tracer_unified.solvers.output_buffers import DebugBuffers
from particle_tracer_unified.solvers.runtime_execution import (
    RunExecutionContext,
    StepLoopResult,
    finalize_runtime_execution,
    initialize_debug_buffers,
    prepare_runtime_execution,
)

ROOT = Path(__file__).resolve().parents[1]
MINIMAL_CASE = ROOT / "examples" / "v02_minimal" / "run_config.yaml"


def _debug_execution(
    *, capture_outputs: bool = True
) -> tuple[RunExecutionContext, DebugBuffers | None]:
    context = load_case(MINIMAL_CASE)._context
    plan = replace(
        context.plan,
        output=replace(context.plan.output, mode="debug"),
    )
    context = replace(context, plan=plan)
    buffers = initialize_debug_buffers(plan, capture_outputs=capture_outputs)
    return (
        prepare_runtime_execution(
            context,
            spatial_dim=2,
            plan=plan,
            debug_buffers=buffers,
        ),
        buffers,
    )


def test_runtime_execution_public_signatures_are_stable() -> None:
    expected = {
        "append_snapshot": (
            "(save_positions: 'list[np.ndarray]', "
            "save_meta: 'list[dict[str, object]]', *, save_index: 'int', "
            "t: 'float', position: 'np.ndarray') -> 'None'"
        ),
        "finalize_runtime_execution": (
            "(prepared: 'RunExecutionContext', loop_result: 'StepLoopResult') "
            "-> 'SolverOutcome'"
        ),
        "initialize_debug_buffers": (
            "(plan: 'SolverPlan', *, capture_outputs: 'bool') -> 'DebugBuffers | None'"
        ),
        "prepare_runtime_execution": (
            "(context: 'SolverContext', *, spatial_dim: 'int', "
            "plan: 'SolverPlan', debug_buffers: 'DebugBuffers | None') "
            "-> 'RunExecutionContext'"
        ),
    }

    assert tuple(runtime_execution.__all__) == (
        "RunExecutionContext",
        "StepLoopResult",
        "append_snapshot",
        "finalize_runtime_execution",
        "initialize_debug_buffers",
        "prepare_runtime_execution",
    )
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(runtime_execution, name))) == signature
    assert runtime_execution.RunExecutionContext is (
        _runtime_execution_context.RunExecutionContext
    )
    assert runtime_execution.StepLoopResult is _runtime_execution_context.StepLoopResult
    assert runtime_execution.append_snapshot is _runtime_outcome.append_snapshot
    assert runtime_execution.finalize_runtime_execution is (
        _runtime_outcome.finalize_runtime_execution
    )
    assert runtime_execution.initialize_debug_buffers is (
        _runtime_outcome.initialize_debug_buffers
    )
    assert runtime_execution.prepare_runtime_execution is (
        _runtime_preparation.prepare_runtime_execution
    )


def test_debug_preparation_diagnostics_have_stable_order() -> None:
    execution, buffers = _debug_execution()
    diagnostics = execution.state.collision_diagnostics
    first = list(diagnostics).index("boundary_broad_phase_enabled")

    assert buffers is not None
    assert list(diagnostics)[first:] == [
        "boundary_broad_phase_enabled",
        "output_mode",
        "output_debug_enabled",
        "field_sampling_s",
        "field_sample_point_count",
        "field_sample_call_count",
        "acceleration_source",
        "acceleration_quantity_names",
        "electric_field_names",
        "drag_gas_properties",
        "field_backend_diagnostics",
        "collision_boundary_geometry",
        "contact_tangent_model",
        "force_catalog",
        "force_runtime",
        "stochastic_motion",
        "plasma_background",
        "charge_model",
    ]


def test_finalize_preserves_snapshot_schedule_shape_dtype_and_memory_counts() -> None:
    execution, buffers = _debug_execution()
    assert buffers is not None
    initial_position = execution.state.x.copy()
    execution.state.x[:] += 0.125

    outcome = finalize_runtime_execution(
        execution,
        StepLoopResult(t=execution.plan.dt, step_count=1, elapsed_s=0.25),
    )

    assert outcome.debug is not None
    assert outcome.final_position is execution.state.x
    assert outcome.debug.trajectory_positions.shape == (2, 2, 2)
    assert outcome.debug.trajectory_positions.dtype == np.float64
    np.testing.assert_array_equal(
        outcome.debug.trajectory_positions[0], initial_position
    )
    np.testing.assert_array_equal(
        outcome.debug.trajectory_positions[1], execution.state.x
    )
    assert outcome.debug.save_frames == [
        {
            "save_index": 0,
            "time_s": 0.0,
            "step_name": "run",
            "segment_name": "run",
        },
        {
            "save_index": 1,
            "time_s": execution.plan.dt,
            "step_name": "run",
            "segment_name": "run",
        },
    ]
    assert outcome.memory_estimate_bytes == {
        "core_array_bytes": 256,
        "compiled_field_array_bytes": 71266,
        "positions_array_bytes": 64,
        "estimated_numpy_bytes": 71586,
    }
    final_keys = list(outcome.collision_diagnostics)
    assert final_keys[final_keys.index("solver_step_count") :] == [
        "solver_step_count",
        "released_count_final",
        "release_cursor_position_final",
        "release_cursor_done",
        "active_count_samples",
        "active_count_mean",
        "active_count_max",
        "electric_q_over_m_particle_stats",
        "output_buffers",
    ]


def test_finalize_does_not_duplicate_snapshot_at_the_same_time() -> None:
    execution, buffers = _debug_execution()
    assert buffers is not None

    outcome = finalize_runtime_execution(
        execution,
        StepLoopResult(t=0.0, step_count=0, elapsed_s=0.0),
    )

    assert outcome.debug is not None
    assert outcome.debug.trajectory_positions.shape == (1, 2, 2)
    assert len(outcome.debug.save_frames) == 1


def test_debug_mode_without_capture_has_no_debug_payload() -> None:
    execution, buffers = _debug_execution(capture_outputs=False)

    outcome = finalize_runtime_execution(
        execution,
        StepLoopResult(t=0.0, step_count=0, elapsed_s=0.0),
    )

    assert buffers is None
    assert outcome.debug is None
    assert outcome.memory_estimate_bytes["positions_array_bytes"] == 0
    assert "output_buffers" not in outcome.collision_diagnostics


def test_memory_counting_deduplicates_shared_backend_arrays() -> None:
    shared = np.arange(6, dtype=np.float64)
    flags = np.ones(4, dtype=np.uint8)
    backend = cast(
        CompiledRuntimeBackend,
        SimpleNamespace(
            axes=(shared, flags),
            times=shared,
            ux=flags,
            uy=None,
        ),
    )

    assert _runtime_outcome._compiled_backend_array_bytes(backend) == (
        shared.nbytes + flags.nbytes
    )
    empty, _ = _runtime_outcome._assemble_saved_positions(
        [], n_particles=3, spatial_dim=2
    )
    assert empty.shape == (0, 3, 2)
    assert empty.dtype == np.float64


def test_q_over_m_summary_ignores_invalid_values_and_keeps_quantiles() -> None:
    summary = _runtime_outcome._finite_q_over_m_summary(
        np.asarray([2.0, -3.0, np.nan, 4.0]),
        np.asarray([1.0, 2.0, 1.0, np.inf]),
    )

    assert summary == pytest.approx(
        {
            "count": 2,
            "charged_count": 2,
            "min": -1.5,
            "median": 0.25,
            "p90": 1.65,
            "max": 2.0,
        },
        rel=1.0e-15,
        abs=0.0,
    )
    assert _runtime_outcome._finite_q_over_m_summary(
        np.asarray([np.nan]), np.asarray([1.0])
    ) == {"count": 0, "charged_count": 0}


@pytest.mark.parametrize(
    ("forces", "stochastic", "expected"),
    [
        (ForceRuntimeParameters(), False, set()),
        (ForceRuntimeParameters(), True, {"temperature_K"}),
        (
            ForceRuntimeParameters(thermophoresis_enabled=True),
            False,
            {"temperature_K", "dynamic_viscosity_Pas", "density_kgm3"},
        ),
        (
            ForceRuntimeParameters(lift_enabled=True),
            False,
            {"dynamic_viscosity_Pas", "density_kgm3"},
        ),
        (
            ForceRuntimeParameters(virtual_mass_enabled=True),
            False,
            {"density_kgm3"},
        ),
    ],
)
def test_required_gas_properties_follow_enabled_physics(
    forces: ForceRuntimeParameters,
    stochastic: bool,
    expected: set[str],
) -> None:
    context = load_case(MINIMAL_CASE)._context
    no_drag = replace(
        context.plan,
        drag_model_mode=DRAG_MODEL_NONE,
        drag_model_name="none",
    )

    assert (
        _runtime_preparation._required_runtime_gas_properties(
            no_drag, forces, stochastic_enabled=stochastic
        )
        == expected
    )


def test_relaxation_times_preserve_mode_formula_and_float64_dtype() -> None:
    context = load_case(MINIMAL_CASE)._context
    mass = np.asarray([2.0e-15, 3.0e-15])
    diameter = np.asarray([1.0e-6, 2.0e-6])
    physics = {
        "gas_density_kgm3": 1.2,
        "gas_mu_pas": 1.8e-5,
        "gas_temperature_K": 300.0,
        "gas_molecular_mass_kg": 4.8e-26,
    }
    no_drag = replace(context.plan, drag_model_mode=DRAG_MODEL_NONE)
    epstein = replace(context.plan, drag_model_mode=DRAG_MODEL_EPSTEIN)
    stokes = replace(context.plan, drag_model_mode=DRAG_MODEL_STOKES)

    none_values = _runtime_preparation._base_relaxation_times(
        particle_mass=mass,
        particle_diameter=diameter,
        plan=no_drag,
        physics=physics,
    )
    epstein_values = _runtime_preparation._base_relaxation_times(
        particle_mass=mass,
        particle_diameter=diameter,
        plan=epstein,
        physics=physics,
    )
    stokes_values = _runtime_preparation._base_relaxation_times(
        particle_mass=mass,
        particle_diameter=diameter,
        plan=stokes,
        physics=physics,
    )

    assert none_values.dtype == np.float64
    assert np.all(np.isinf(none_values))
    np.testing.assert_array_equal(
        epstein_values,
        [
            epstein_relaxation_time(
                mass[i],
                physics["gas_density_kgm3"],
                physics["gas_temperature_K"],
                diameter[i],
                physics["gas_molecular_mass_kg"],
            )
            for i in range(2)
        ],
    )
    np.testing.assert_array_equal(
        stokes_values,
        [
            stokes_relaxation_time(mass[i], physics["gas_mu_pas"], diameter[i])
            for i in range(2)
        ],
    )


def test_coordinate_validation_reports_errors_in_stable_priority() -> None:
    context = load_case(MINIMAL_CASE)._context
    particles = replace(
        context.particles,
        position=np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float64),
    )
    stochastic = replace(context.options.stochastic_motion, enabled=True)
    forces = replace(context.options.force_runtime, lift_enabled=True)
    options = replace(
        context.options,
        stochastic_motion=stochastic,
        force_runtime=forces,
    )
    axisymmetric = replace(
        context,
        coordinate_system="axisymmetric_rz",
        particles=particles,
        options=options,
    )

    with pytest.raises(ValueError, match=r"restricted.*2D dynamics"):
        _runtime_preparation._validate_coordinate_system(
            axisymmetric,
            plan=context.plan,
            options=options,
            spatial_dim=3,
        )
    with pytest.raises(ValueError, match="does not support Brownian motion"):
        _runtime_preparation._validate_coordinate_system(
            axisymmetric,
            plan=context.plan,
            options=options,
            spatial_dim=2,
        )

    no_stochastic = replace(
        options,
        stochastic_motion=replace(stochastic, enabled=False),
    )
    with pytest.raises(ValueError, match="does not support the Cartesian lift"):
        _runtime_preparation._validate_coordinate_system(
            axisymmetric,
            plan=context.plan,
            options=no_stochastic,
            spatial_dim=2,
        )

    no_lift = replace(no_stochastic, force_runtime=ForceRuntimeParameters())
    _runtime_preparation._validate_coordinate_system(
        axisymmetric,
        plan=context.plan,
        options=no_lift,
        spatial_dim=2,
    )

    valid_particles = replace(
        particles,
        position=np.asarray([[0.25, 0.0], [0.5, 0.0]], dtype=np.float64),
    )
    _runtime_preparation._validate_coordinate_system(
        replace(axisymmetric, particles=valid_particles),
        plan=context.plan,
        options=no_lift,
        spatial_dim=2,
    )


def test_preparation_rejects_missing_3d_geometry_in_stable_order() -> None:
    with pytest.raises(ValueError, match="requires geometry_provider"):
        _runtime_preparation._prepare_triangle_surface(
            SimpleNamespace(geometry_provider=None),
            3,
        )

    provider = SimpleNamespace(geometry=SimpleNamespace(boundary_triangles=None))
    with pytest.raises(ValueError, match=r"requires geometry\.boundary_triangles"):
        _runtime_preparation._prepare_triangle_surface(
            SimpleNamespace(geometry_provider=provider),
            3,
        )


def test_all_displaced_fluid_forces_accept_positive_particle_density() -> None:
    _runtime_preparation._require_particle_density_for_displaced_fluid_forces(
        np.asarray([1200.0]),
        np.asarray([7]),
        ForceRuntimeParameters(
            pressure_gradient_enabled=True,
            virtual_mass_enabled=True,
            gravity_buoyancy_enabled=True,
        ),
    )


def test_prepare_pads_short_body_acceleration_without_changing_dtype() -> None:
    context = load_case(MINIMAL_CASE)._context
    plan = replace(context.plan, body_acceleration_mps2=(1.25,))
    prepared = prepare_runtime_execution(
        replace(context, plan=plan),
        spatial_dim=2,
        plan=plan,
        debug_buffers=None,
    )

    np.testing.assert_array_equal(
        prepared.body_acceleration_mps2,
        np.asarray([1.25, 0.0], dtype=np.float64),
    )
    assert prepared.body_acceleration_mps2.dtype == np.float64
