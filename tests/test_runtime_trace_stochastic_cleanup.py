from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from particle_tracer_unified.solvers import (
    _runtime_trace_refinement,
    _stochastic_temperature,
)
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    SegmentMotionBatchTrace,
    SegmentMotionRequest,
)
from particle_tracer_unified.solvers.segment_trace import TraceRefinementPolicy
from particle_tracer_unified.solvers.stochastic_motion import (
    PiecewiseLangevinPath,
    StochasticMotionConfig,
)


class _InsideBoundary:
    def polyline_hit(self, _start: np.ndarray, _trace: np.ndarray) -> None:
        return None

    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.ones(len(points), dtype=bool)


def _motion_batch(
    *, duration_s: float = 0.5, substeps: int = 1
) -> SegmentMotionBatchTrace:
    position = np.asarray([[0.0, 0.0]], dtype=np.float64)
    velocity = np.asarray([[1.0, 2.0]], dtype=np.float64)
    return SegmentMotionBatchTrace(
        request=SegmentMotionBatchRequest(
            position_m=position,
            velocity_mps=velocity,
            active=np.asarray([True]),
            tau_stokes_s=np.asarray([0.2]),
            particle_diameter_m=np.asarray([1.0e-6]),
            particle_density_kgm3=np.asarray([900.0]),
            particle_mass_kg=np.asarray([2.0e-15]),
            dep_particle_rel_permittivity=np.asarray([2.5]),
            thermophoretic_coefficient=np.asarray([0.7]),
            end_time_s=1.25,
            duration_s=duration_s,
            spatial_dim=2,
            backend=cast(Any, object()),
            body_acceleration_mps2=np.asarray([0.0, -9.81]),
            gas_density_kgm3=1.2,
            gas_dynamic_viscosity_Pas=1.8e-5,
            gas_temperature_K=310.0,
            gas_molecular_mass_kg=4.65e-26,
            drag_model_mode=1,
            adaptive_substep_enabled=0,
            adaptive_substep_max_splits=2,
            electric_q_over_m_Ckg=np.asarray([6.0]),
        ),
        endpoint_position_m=np.asarray([[1.0, 0.0]], dtype=np.float64),
        endpoint_velocity_mps=velocity.copy(),
        midpoint_position_m=np.asarray([[0.5, 0.0]], dtype=np.float64),
        substep_count=np.asarray([substeps], dtype=np.int32),
        aggregate_support_status=np.asarray([0], dtype=np.uint8),
        local_error_resolved=np.asarray([True]),
    )


def test_unresolved_local_error_skips_geometry_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motion_batch = _motion_batch()
    motion_batch.local_error_resolved[0] = False
    monkeypatch.setattr(
        _runtime_trace_refinement,
        "_trace_refinement_decision",
        lambda *_args, **_kwargs: pytest.fail("unverified motion reached geometry"),
    )
    stage_traces = {0: np.asarray([[0.5, 0.0], [1.0, 0.0]])}

    unresolved = _runtime_trace_refinement.refine_deterministic_stage_traces(
        runtime=object(),
        boundary_service=cast(Any, _InsideBoundary()),
        motion_batch=motion_batch,
        stage_traces=stage_traces,
        refinement_policy=TraceRefinementPolicy(0.0, np.nan, 1, 2),
    )

    assert unresolved == {0: "local_error"}
    assert stage_traces == {}


def test_trace_refinement_replays_the_original_batch_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry_risk = iter((True, False))
    monkeypatch.setattr(
        _runtime_trace_refinement,
        "_geometry_refinement_required",
        lambda *_args, **_kwargs: next(geometry_risk),
    )
    monkeypatch.setattr(
        _runtime_trace_refinement,
        "segment_length_required_substeps",
        lambda _start, _trace, *, current_substeps, **_kwargs: current_substeps,
    )
    replayed = np.asarray(
        [[0.25, 0.1], [0.5, 0.2], [0.75, 0.3], [1.0, 0.4]],
        dtype=np.float64,
    )
    captured: list[SegmentMotionRequest] = []

    motion_batch = _motion_batch()

    def trace(
        batch,
        index: int,
        *,
        minimum_substeps: int = 1,
    ) -> SimpleNamespace:
        request = batch.request.particle_request(index).with_minimum_substeps(
            minimum_substeps
        )
        captured.append(request)
        return SimpleNamespace(
            positions_m=replayed,
            substep_count=2,
            endpoint_position_m=replayed[-1],
            endpoint_velocity_mps=np.asarray([3.0, 4.0]),
            aggregate_support_status=2,
            local_error_resolved=True,
        )

    monkeypatch.setattr(type(motion_batch), "particle_trace", trace)
    stage_traces: dict[int, np.ndarray] = {}

    unresolved = _runtime_trace_refinement.refine_deterministic_stage_traces(
        runtime=object(),
        boundary_service=cast(Any, _InsideBoundary()),
        motion_batch=motion_batch,
        stage_traces=stage_traces,
        refinement_policy=TraceRefinementPolicy(
            on_boundary_tolerance_m=1.0e-9,
            support_spacing_m=np.nan,
            adaptive_substep_enabled=0,
            adaptive_substep_max_splits=2,
        ),
    )

    assert unresolved == {}
    assert len(captured) == 1
    request = captured[0]
    assert request.minimum_substeps == 2
    assert request.duration_s == 0.5
    assert request.end_time_s == 1.25
    assert request.electric_q_over_m_Ckg == 6.0
    np.testing.assert_array_equal(request.position_m, [0.0, 0.0])
    np.testing.assert_array_equal(request.velocity_mps, [1.0, 2.0])
    np.testing.assert_array_equal(motion_batch.endpoint_position_m[0], replayed[-1])
    np.testing.assert_array_equal(motion_batch.endpoint_velocity_mps[0], [3.0, 4.0])
    np.testing.assert_array_equal(motion_batch.midpoint_position_m[0], replayed[1])
    np.testing.assert_array_equal(stage_traces[0], replayed)
    assert motion_batch.substep_count.tolist() == [2]
    assert motion_batch.aggregate_support_status.tolist() == [2]


def test_trace_refinement_returns_early_for_zero_duration() -> None:
    assert (
        _runtime_trace_refinement.refine_deterministic_stage_traces(
            runtime=object(),
            boundary_service=cast(Any, _InsideBoundary()),
            motion_batch=_motion_batch(duration_s=0.0),
            stage_traces={},
            refinement_policy=TraceRefinementPolicy(0.0, np.nan, 0, 2),
        )
        == {}
    )


def test_trace_refinement_keeps_an_existing_complete_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _runtime_trace_refinement,
        "_geometry_refinement_required",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        _runtime_trace_refinement,
        "segment_length_required_substeps",
        lambda _start, _trace, *, current_substeps, **_kwargs: current_substeps,
    )
    motion_batch = _motion_batch(substeps=2)
    saved = np.arange(8, dtype=np.float64).reshape(4, 2)
    stage_traces = {0: saved}

    unresolved = _runtime_trace_refinement.refine_deterministic_stage_traces(
        runtime=object(),
        boundary_service=cast(Any, _InsideBoundary()),
        motion_batch=motion_batch,
        stage_traces=stage_traces,
        refinement_policy=TraceRefinementPolicy(0.0, np.nan, 0, 2),
    )

    assert unresolved == {}
    np.testing.assert_array_equal(stage_traces[0], saved)
    assert stage_traces[0] is not saved


def _leaf_plan(
    particle_index: int,
    midpoint_times_s: tuple[float, ...],
    x_coordinates: tuple[float, ...],
) -> _stochastic_temperature.ParticleLeafPlan:
    count = len(midpoint_times_s)
    return _stochastic_temperature.ParticleLeafPlan(
        particle_index=particle_index,
        leaf_end_times_s=np.arange(1, count + 1, dtype=np.float64),
        midpoint_times_s=np.asarray(midpoint_times_s, dtype=np.float64),
        midpoint_positions_m=np.column_stack(
            (np.asarray(x_coordinates, dtype=np.float64), np.zeros(count))
        ),
        tau_mid_s=np.ones(count, dtype=np.float64),
        particle_mass_kg=1.0,
    )


def test_temperature_field_sampling_preserves_time_and_particle_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, np.ndarray]] = []

    def measure(
        _compiled: object,
        _plan: object,
        points: np.ndarray,
        time_s: float,
        **_kwargs: object,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        calls.append((time_s, points.copy()))
        values = 100.0 * time_s + points[:, 0]
        return (
            SimpleNamespace(values={_stochastic_temperature.TEMPERATURE: values}),
            SimpleNamespace(elapsed_s=0.25, point_count=len(points), call_count=1),
        )

    monkeypatch.setattr(
        _stochastic_temperature, "measure_sample_fields_for_stage", measure
    )
    plans = [
        _leaf_plan(7, (1.0, 3.0), (10.0, 30.0)),
        _leaf_plan(2, (1.0, 2.0), (11.0, 20.0)),
    ]

    values, elapsed, point_count, call_count = (
        _stochastic_temperature.sample_plan_temperatures(
            config=StochasticMotionConfig(temperature_source="field_T_then_gas"),
            compiled=cast(
                Any, SimpleNamespace(gas_temperature_source="field:temperature")
            ),
            plans=plans,
            spatial_dim=2,
            gas_temperature_K=300.0,
            collect_diagnostics=True,
        )
    )

    assert [time_s for time_s, _points in calls] == [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(calls[0][1][:, 0], [10.0, 11.0])
    np.testing.assert_array_equal(values[0], [110.0, 330.0])
    np.testing.assert_array_equal(values[1], [111.0, 220.0])
    assert all(value.dtype == np.float64 for value in values)
    assert (elapsed, point_count, call_count) == (0.75, 4, 3)


@pytest.mark.parametrize(
    ("source", "temperature", "message"),
    [
        ("unknown", 300.0, "temperature_source"),
        ("gas", np.nan, "finite positive configured gas temperature"),
    ],
)
def test_temperature_configuration_errors_are_reported_before_sampling(
    source: str,
    temperature: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _stochastic_temperature.sample_plan_temperatures(
            config=StochasticMotionConfig(temperature_source=source),
            compiled=cast(Any, SimpleNamespace(gas_temperature_source="gas")),
            plans=[_leaf_plan(3, (1.0,), (0.5,))],
            spatial_dim=2,
            gas_temperature_K=temperature,
            collect_diagnostics=False,
        )


@pytest.mark.parametrize(
    ("sampled_values", "message"),
    [
        (None, "declared but was not returned"),
        (np.asarray([280.0, 290.0]), "must have shape"),
    ],
)
def test_temperature_field_rejects_missing_or_misshaped_samples(
    monkeypatch: pytest.MonkeyPatch,
    sampled_values: np.ndarray | None,
    message: str,
) -> None:
    values = (
        {}
        if sampled_values is None
        else {_stochastic_temperature.TEMPERATURE: sampled_values}
    )
    monkeypatch.setattr(
        _stochastic_temperature,
        "sample_fields_for_stage",
        lambda *_args, **_kwargs: SimpleNamespace(values=values),
    )

    with pytest.raises(ValueError, match=message):
        _stochastic_temperature.sample_plan_temperatures(
            config=StochasticMotionConfig(temperature_source="field_T_then_gas"),
            compiled=cast(Any, SimpleNamespace(gas_temperature_source="field:T")),
            plans=[_leaf_plan(3, (1.0,), (0.5,))],
            spatial_dim=2,
            gas_temperature_K=300.0,
            collect_diagnostics=False,
        )


def _piecewise_inputs() -> dict[str, np.ndarray]:
    return {
        "leaf_end_times_s": np.asarray([0.2, 0.7]),
        "tau_eff_s": np.asarray([0.1, 0.3]),
        "thermal_velocity_variance_m2s2": np.asarray([1.5, 2.5]),
        "z_velocity": np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        "z_position": np.asarray([[0.5, 0.6], [0.7, 0.8]]),
        "bridge_seeds": np.asarray([11, 12]),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("leaf_end_times_s", np.asarray([]), "finite non-empty vector"),
        ("leaf_end_times_s", np.asarray([0.2, 0.1]), "strictly increasing"),
        ("tau_eff_s", np.asarray([0.1]), "coefficient arrays"),
        ("tau_eff_s", np.asarray([0.1, np.nan]), "tau_eff_s"),
        (
            "thermal_velocity_variance_m2s2",
            np.asarray([1.5, -1.0]),
            "thermal variance",
        ),
        ("z_velocity", np.asarray([0.1, 0.2]), "normal arrays must have shape"),
        (
            "z_position",
            np.asarray([[0.5, np.nan], [0.7, 0.8]]),
            "normal arrays must be finite",
        ),
        ("bridge_seeds", np.asarray([11]), "bridge seeds"),
    ],
)
def test_piecewise_path_validation_order(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    inputs = _piecewise_inputs()
    inputs[field] = value

    with pytest.raises(ValueError, match=message):
        PiecewiseLangevinPath(**inputs)


def test_piecewise_path_normalizes_storage_dtypes_without_changing_values() -> None:
    inputs = _piecewise_inputs()
    path = PiecewiseLangevinPath(**inputs)

    assert path.leaf_end_times_s.dtype == np.float64
    assert path.tau_eff_s.dtype == np.float64
    assert path.thermal_velocity_variance_m2s2.dtype == np.float64
    assert path.z_velocity.dtype == np.float64
    assert path.z_position.dtype == np.float64
    assert path.bridge_seeds.dtype == np.int64
    assert [leaf.duration_s for leaf in path._leaves] == pytest.approx([0.2, 0.5])
