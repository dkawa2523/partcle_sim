from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import yaml

from particle_tracer_unified import load_case, simulate
from particle_tracer_unified.solvers import (
    _runtime_preparation,
    collision_detection,
    field_runtime,
    high_fidelity_runtime,
    stochastic_motion,
)
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
)
from particle_tracer_unified.solvers.diagnostics import initial_collision_diagnostics
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
    DRAG_MODEL_STOKES_CUNNINGHAM,
)
from particle_tracer_unified.solvers.runtime_plan import SolverPlan
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    trace_motion_batch,
)
from particle_tracer_unified.solvers.stochastic_motion import StochasticMotionConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDARD_EXAMPLE = REPO_ROOT / "examples" / "v02_minimal" / "run_config.yaml"


def _compiled_backend() -> RegularRectilinearCompiledBackend:
    axes = (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    shape = (1, 2, 2)
    valid = np.ones((2, 2), dtype=bool)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=np.asarray([0.0]),
        ux=np.zeros(shape),
        uy=np.zeros(shape),
        gas_density=np.full(shape, 1.2),
        gas_mu=np.full(shape, 1.8e-5),
        gas_temperature=np.full(shape, 300.0),
        valid_mask=valid,
        core_valid_mask=valid,
    )


def test_standard_solver_outcome_does_not_own_debug_payload_lists() -> None:
    case = load_case(STANDARD_EXAMPLE)

    outcome = high_fidelity_runtime.simulate_context(case.solver_context)

    assert outcome.debug is None
    for legacy_name in (
        "positions",
        "save_meta",
        "wall_rows",
        "max_hit_rows",
        "step_rows",
    ):
        assert not hasattr(outcome, legacy_name)


def test_immutable_runtime_physics_is_resolved_once_per_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = _runtime_preparation._runtime_physics

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_runtime_preparation, "_runtime_physics", counted)

    simulate(load_case(STANDARD_EXAMPLE))

    assert calls == 1


@pytest.mark.parametrize(
    ("mode", "name", "invalid_attribute"),
    [
        (DRAG_MODEL_STOKES, "stokes", "dynamic_viscosity_Pas"),
        (DRAG_MODEL_SCHILLER_NAUMANN, "schiller_naumann", "density_kgm3"),
        (DRAG_MODEL_STOKES_CUNNINGHAM, "stokes_cunningham", "molecular_mass_amu"),
        (DRAG_MODEL_EPSTEIN, "epstein", "temperature_K"),
    ],
)
def test_runtime_physics_rejects_each_drag_models_required_gas_property(
    mode: int,
    name: str,
    invalid_attribute: str,
) -> None:
    gas = SimpleNamespace(
        density_kgm3=1.2,
        dynamic_viscosity_Pas=1.8e-5,
        temperature=300.0,
        molecular_mass_amu=28.97,
    )
    runtime_name = (
        "temperature" if invalid_attribute == "temperature_K" else invalid_attribute
    )
    setattr(gas, runtime_name, 0.0)
    plan = cast(
        SolverPlan,
        SimpleNamespace(drag_model_mode=mode, drag_model_name=name),
    )

    with pytest.raises(ValueError, match=invalid_attribute):
        _runtime_preparation._runtime_physics(SimpleNamespace(gas=gas), plan)


def test_debug_final_snapshot_uses_time_scaled_roundoff_tolerance(
    tmp_path: Path,
) -> None:
    value = yaml.safe_load(STANDARD_EXAMPLE.read_text(encoding="utf-8"))
    value["inputs"]["particles"] = str(
        (STANDARD_EXAMPLE.parent / "particles.csv").resolve()
    )
    value["inputs"]["boundaries"] = str(
        (STANDARD_EXAMPLE.parent / "boundaries.csv").resolve()
    )
    value["time"] = {"dt": 1.0e-18, "t_end": 5.0e-18}
    value["output"] = {"mode": "debug", "trajectory_interval_steps": 10}
    path = tmp_path / "tiny-time-debug.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    result = simulate(load_case(path))

    assert [frame["time_s"] for frame in result.debug["save_frames"]] == pytest.approx(
        [0.0, 5.0e-18],
        abs=1.0e-32,
    )
    assert result.debug["trajectory_m"].shape[0] == 2


def test_debug_capture_does_not_change_the_numerical_result(tmp_path: Path) -> None:
    standard = simulate(load_case(STANDARD_EXAMPLE))
    value = yaml.safe_load(STANDARD_EXAMPLE.read_text(encoding="utf-8"))
    value["inputs"]["particles"] = str(
        (STANDARD_EXAMPLE.parent / "particles.csv").resolve()
    )
    value["inputs"]["boundaries"] = str(
        (STANDARD_EXAMPLE.parent / "boundaries.csv").resolve()
    )
    value["output"] = {"mode": "debug", "trajectory_interval_steps": 1}
    path = tmp_path / "debug.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    debug = simulate(load_case(path))

    np.testing.assert_array_equal(debug.state.position_m, standard.state.position_m)
    np.testing.assert_array_equal(debug.state.velocity_mps, standard.state.velocity_mps)
    np.testing.assert_array_equal(debug.state.charge_C, standard.state.charge_C)
    np.testing.assert_array_equal(
        debug.state.terminal_state, standard.state.terminal_state
    )
    assert debug.wall_summary == standard.wall_summary
    assert debug.stats.safety_counters == standard.stats.safety_counters


def test_runtime_field_sampling_measures_only_debug_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = 0
    original = high_fidelity_runtime.measure_sample_fields_for_stage

    def counted_measure(*args, **kwargs):
        nonlocal measured
        measured += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        high_fidelity_runtime,
        "measure_sample_fields_for_stage",
        counted_measure,
    )
    common = (
        _compiled_backend(),
        None,
        np.asarray([[0.5, 0.5]]),
        0.0,
    )
    options = {
        "spatial_dim": 2,
        "need_flow": True,
        "fallback_density_kgm3": 1.2,
        "fallback_mu_pas": 1.8e-5,
        "fallback_temperature_K": 300.0,
    }

    standard = initial_collision_diagnostics(debug=False)
    high_fidelity_runtime._sample_runtime_stage(standard, *common, **options)
    assert measured == 0
    assert "field_sampling_s" not in standard

    debug = initial_collision_diagnostics(debug=True)
    high_fidelity_runtime._sample_runtime_stage(debug, *common, **options)
    assert measured == 1
    assert debug["field_sample_call_count"] > 0
    assert debug["field_sample_point_count"] > 0


def test_adaptive_substep_limit_counter_is_debug_only() -> None:
    standard = initial_collision_diagnostics(debug=False)
    debug = initial_collision_diagnostics(debug=True)

    assert "adaptive_substep_limit_reached_count" not in standard
    assert debug["adaptive_substep_limit_reached_count"] == 0


def test_standard_solver_skips_detailed_section_timers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime_timer_calls = 0
    collision_timer_calls = 0
    runtime_start = high_fidelity_runtime._detailed_timer_start
    collision_start = collision_detection._detailed_timer_start

    def checked_runtime_start(accumulator):
        nonlocal runtime_timer_calls
        runtime_timer_calls += 1
        assert accumulator is None
        return runtime_start(accumulator)

    def checked_collision_start(accumulator):
        nonlocal collision_timer_calls
        collision_timer_calls += 1
        assert accumulator is None
        return collision_start(accumulator)

    monkeypatch.setattr(
        high_fidelity_runtime,
        "_detailed_timer_start",
        checked_runtime_start,
    )
    monkeypatch.setattr(
        collision_detection,
        "_detailed_timer_start",
        checked_collision_start,
    )
    standard = simulate(load_case(STANDARD_EXAMPLE))

    assert runtime_timer_calls > 0
    assert collision_timer_calls > 0
    assert set(standard.stats.timing_s) == {
        "setup_s",
        "step_loop_s",
        "solver_core_s",
    }

    value = yaml.safe_load(STANDARD_EXAMPLE.read_text(encoding="utf-8"))
    value["inputs"]["particles"] = str(
        (STANDARD_EXAMPLE.parent / "particles.csv").resolve()
    )
    value["inputs"]["boundaries"] = str(
        (STANDARD_EXAMPLE.parent / "boundaries.csv").resolve()
    )
    value["output"] = {"mode": "debug", "trajectory_interval_steps": 1}
    debug_path = tmp_path / "debug-timing.yaml"
    debug_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    # Debug passes a real accumulator, so restore the standard-only spies.
    monkeypatch.setattr(high_fidelity_runtime, "_detailed_timer_start", runtime_start)
    monkeypatch.setattr(collision_detection, "_detailed_timer_start", collision_start)
    debug = simulate(load_case(debug_path))
    assert {
        "freeflight_s",
        "collision_classify_s",
        "positions_assembly_s",
        "field_sampling_s",
    }.issubset(debug.stats.timing_s)
    assert "force_eval_s" not in debug.stats.timing_s
    assert "force_eval_s" not in debug.debug["collision_diagnostics"]


def test_brownian_standard_path_skips_timing_and_diagnostic_re_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perf_calls = 0
    path_evaluations = 0
    original_perf = field_runtime.perf_counter
    original_evaluate = stochastic_motion.PiecewiseLangevinPath.state_at

    def counted_perf() -> float:
        nonlocal perf_calls
        perf_calls += 1
        return original_perf()

    def counted_evaluate(*args, **kwargs):
        nonlocal path_evaluations
        path_evaluations += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(field_runtime, "perf_counter", counted_perf)
    monkeypatch.setattr(
        stochastic_motion.PiecewiseLangevinPath, "state_at", counted_evaluate
    )
    compiled = replace(_compiled_backend(), gas_temperature_source="field:T")
    mass = np.asarray([1.0e-15])
    motion_batch = trace_motion_batch(
        SegmentMotionBatchRequest(
            position_m=np.asarray([[0.5, 0.5]]),
            velocity_mps=np.asarray([[0.0, 0.0]]),
            active=np.asarray([True]),
            tau_stokes_s=np.asarray([0.1]),
            particle_diameter_m=np.asarray([1.0e-6]),
            particle_density_kgm3=np.asarray([1000.0]),
            particle_mass_kg=mass,
            dep_particle_rel_permittivity=np.asarray([np.nan]),
            thermophoretic_coefficient=np.asarray([np.nan]),
            end_time_s=0.01,
            duration_s=0.01,
            spatial_dim=2,
            backend=compiled,
            body_acceleration_mps2=np.zeros(2),
            gas_density_kgm3=1.2,
            gas_dynamic_viscosity_Pas=1.8e-5,
            gas_temperature_K=300.0,
            gas_molecular_mass_kg=4.65e-26,
            drag_model_mode=DRAG_MODEL_STOKES,
            adaptive_substep_enabled=0,
            adaptive_substep_max_splits=4,
        )
    )
    common = {
        "config": StochasticMotionConfig(
            enabled=True, temperature_source="field_T_then_gas"
        ),
        "motion_batch": motion_batch,
        "particle_indices": np.asarray([0], dtype=np.int64),
        "minimum_substeps": motion_batch.substep_count,
        "particle_mass": mass,
        "gas_temperature_K": 300.0,
    }

    paths, standard = stochastic_motion.sample_piecewise_langevin_paths(
        rng=np.random.default_rng(7),
        **common,
    )
    assert paths
    assert standard == {"applied": True}
    assert perf_calls == 0
    assert path_evaluations == 0

    _paths, debug = stochastic_motion.sample_piecewise_langevin_paths(
        rng=np.random.default_rng(7),
        collect_diagnostics=True,
        **common,
    )
    assert perf_calls > 0
    assert path_evaluations == 1
    assert debug["field_sample_call_count"] > 0
    assert debug["rms_velocity_kick_mps"] >= 0.0
