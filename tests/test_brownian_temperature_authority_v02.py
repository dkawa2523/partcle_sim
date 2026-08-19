from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.solvers import _stochastic_temperature
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
)
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_STOKES,
)
from particle_tracer_unified.solvers.sampling_backend import (
    FLOW_VELOCITY,
    TEMPERATURE,
)
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    SegmentMotionBatchTrace,
    trace_motion_batch,
    trace_motion_segment,
)
from particle_tracer_unified.solvers.stochastic_motion import (
    K_BOLTZMANN,
    StochasticMotionConfig,
    sample_piecewise_langevin_paths,
)


def _compiled_backend(*, temperature_source: str) -> RegularRectilinearCompiledBackend:
    axes = (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    shape = (1, 2, 2)
    valid = np.ones((2, 2), dtype=bool)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=np.asarray([0.0]),
        ux=np.zeros(shape),
        uy=np.zeros(shape),
        gas_density=np.full(shape, np.nan),
        gas_mu=np.full(shape, np.nan),
        gas_temperature=np.full(shape, np.nan),
        valid_mask=valid,
        core_valid_mask=valid,
        gas_temperature_source=temperature_source,
    )


def _sample_args(
    compiled: RegularRectilinearCompiledBackend,
    rng: np.random.Generator,
) -> dict[str, object]:
    mass = np.asarray([2.0e-15])
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
            gas_temperature_K=350.0,
            gas_molecular_mass_kg=4.65e-26,
            drag_model_mode=DRAG_MODEL_STOKES,
            adaptive_substep_enabled=0,
            adaptive_substep_max_splits=4,
        )
    )
    return {
        "config": StochasticMotionConfig(
            enabled=True, temperature_source="field_T_then_gas"
        ),
        "rng": rng,
        "motion_batch": motion_batch,
        "particle_indices": np.asarray([0], dtype=np.int64),
        "minimum_substeps": motion_batch.substep_count,
        "particle_mass": mass,
        "gas_temperature_K": 350.0,
    }


@pytest.mark.parametrize("invalid_temperature", [np.nan, np.inf, 0.0, -1.0])
def test_declared_brownian_temperature_field_is_never_repaired_by_gas_fallback(
    monkeypatch: pytest.MonkeyPatch,
    invalid_temperature: float,
) -> None:
    compiled = _compiled_backend(temperature_source="field:T")

    def invalid_field_sample(*args, **kwargs) -> StageFields:
        del args, kwargs
        return StageFields(
            points_m=np.asarray([[0.5, 0.5]]),
            time_s=0.0,
            values={
                FLOW_VELOCITY: np.zeros((1, 2)),
                TEMPERATURE: np.asarray([invalid_temperature]),
            },
            supported=np.asarray([True]),
        )

    monkeypatch.setattr(
        _stochastic_temperature, "sample_fields_for_stage", invalid_field_sample
    )
    rng = np.random.default_rng(712)
    state_before = deepcopy(rng.bit_generator.state)

    with pytest.raises(
        ValueError, match=r"declared temperature field.*invalid particle indices: \[0\]"
    ):
        sample_piecewise_langevin_paths(**_sample_args(compiled, rng))

    assert rng.bit_generator.state == state_before


def _field_then_gas_uses_scalar_without_absent_field_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled_backend(temperature_source="context:gas")
    requested_temperature: list[bool] = []
    original_sample = _stochastic_temperature.sample_fields_for_stage

    def recorded_sample(*args, **kwargs) -> StageFields:
        requested_temperature.append(bool(kwargs["need_gas_temperature"]))
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(
        _stochastic_temperature, "sample_fields_for_stage", recorded_sample
    )
    mass = 2.0e-15
    paths, result = sample_piecewise_langevin_paths(
        **_sample_args(compiled, np.random.default_rng(713))
    )

    assert requested_temperature == []
    assert result == {"applied": True}
    assert paths[0].thermal_velocity_variance_m2s2[0] == pytest.approx(
        K_BOLTZMANN * 350.0 / mass,
        rel=1.0e-15,
    )


test_field_then_gas_uses_scalar_without_sampling_when_temperature_quantity_is_absent = (
    _field_then_gas_uses_scalar_without_absent_field_sampling
)


def test_declared_valid_temperature_field_remains_authoritative() -> None:
    compiled = replace(
        _compiled_backend(temperature_source="field:T"),
        gas_temperature=np.full((1, 2, 2), 525.0),
    )
    mass = 2.0e-15

    paths, _result = sample_piecewise_langevin_paths(
        **_sample_args(compiled, np.random.default_rng(714))
    )

    assert paths[0].thermal_velocity_variance_m2s2[0] == pytest.approx(
        K_BOLTZMANN * 525.0 / mass,
        rel=1.0e-15,
    )


def test_compiled_declared_zero_temperature_field_fails_end_to_end() -> None:
    compiled = replace(
        _compiled_backend(temperature_source="field:T"),
        gas_temperature=np.zeros((1, 2, 2)),
    )
    rng = np.random.default_rng(716)
    state_before = deepcopy(rng.bit_generator.state)

    with pytest.raises(ValueError, match="temperature"):
        sample_piecewise_langevin_paths(**_sample_args(compiled, rng))

    assert rng.bit_generator.state == state_before


def test_explicit_gas_temperature_mode_does_not_sample_declared_field() -> None:
    compiled = _compiled_backend(temperature_source="field:T")
    args = _sample_args(compiled, np.random.default_rng(715))
    args["config"] = StochasticMotionConfig(enabled=True, temperature_source="gas")

    paths, _result = sample_piecewise_langevin_paths(**args)

    assert paths[0].thermal_velocity_variance_m2s2[0] == pytest.approx(
        K_BOLTZMANN * 350.0 / 2.0e-15,
        rel=1.0e-15,
    )


def test_path_sampling_preserves_velocity_position_seed_draw_order() -> None:
    seed = 718
    rng = np.random.default_rng(seed)

    paths, _result = sample_piecewise_langevin_paths(
        **_sample_args(_compiled_backend(temperature_source="context:gas"), rng)
    )

    path = paths[0]
    expected_rng = np.random.default_rng(seed)
    expected_velocity = expected_rng.normal(size=path.z_velocity.shape)
    expected_position = expected_rng.normal(size=path.z_position.shape)
    expected_seeds = expected_rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=path.bridge_seeds.size,
        dtype=np.int64,
    )
    np.testing.assert_array_equal(path.z_velocity, expected_velocity)
    np.testing.assert_array_equal(path.z_position, expected_position)
    np.testing.assert_array_equal(path.bridge_seeds, expected_seeds)
    assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_unresolved_brownian_coefficients_stop_before_random_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled_backend(temperature_source="field:T")
    rng = np.random.default_rng(720)
    args = _sample_args(compiled, rng)
    base_motion = args["motion_batch"]
    assert isinstance(base_motion, SegmentMotionBatchTrace)
    motion_batch = trace_motion_batch(
        replace(base_motion.request, adaptive_substep_max_splits=0)
    )
    args["motion_batch"] = motion_batch
    args["minimum_substeps"] = motion_batch.substep_count

    def transient_temperature(*sample_args, **kwargs) -> StageFields:
        points = np.asarray(
            sample_args[2] if len(sample_args) > 2 else kwargs["points_m"],
            dtype=np.float64,
        )
        time_s = float(sample_args[3] if len(sample_args) > 3 else kwargs["time_s"])
        return StageFields(
            points_m=points,
            time_s=time_s,
            values={
                FLOW_VELOCITY: np.zeros_like(points),
                TEMPERATURE: np.full(points.shape[0], 300.0 + 3.0e4 * time_s),
            },
            supported=np.ones(points.shape[0], dtype=bool),
        )

    monkeypatch.setattr(
        _stochastic_temperature, "sample_fields_for_stage", transient_temperature
    )
    state_before = deepcopy(rng.bit_generator.state)

    paths, _result = sample_piecewise_langevin_paths(**args)

    assert paths == {}
    assert motion_batch.local_error_resolved.tolist() == [False]
    assert rng.bit_generator.state == state_before


def test_epstein_brownian_uses_same_midpoint_temperature_and_tau_as_motion_trace() -> (
    None
):
    axes = (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    times = np.asarray([0.0, 1.0])
    x_grid = np.broadcast_to(axes[0][:, None], (2, 2))
    temperature = np.stack((250.0 + 50.0 * x_grid, 450.0 + 100.0 * x_grid))
    shape = temperature.shape
    gas_density = np.full(shape, 2.0e-5, dtype=np.float64)
    compiled = RegularRectilinearCompiledBackend(
        axes=axes,
        times=times,
        ux=np.zeros(shape),
        uy=np.zeros(shape),
        gas_density=gas_density,
        gas_mu=np.full(shape, 1.8e-5),
        gas_temperature=temperature,
        valid_mask=np.ones((2, 2), dtype=bool),
        core_valid_mask=np.ones((2, 2), dtype=bool),
        gas_density_source="field:rho_g",
        gas_mu_source="field:mu",
        gas_temperature_source="field:T",
    )
    mass = 1.0e-15
    diameter = 1.0e-6
    molecular_mass = 6.63e-26
    request = SegmentMotionBatchRequest(
        position_m=np.asarray([[0.2, 0.4]]),
        velocity_mps=np.asarray([[0.1, 0.0]]),
        active=np.asarray([True]),
        tau_stokes_s=np.asarray([1.0]),
        particle_diameter_m=np.asarray([diameter]),
        particle_density_kgm3=np.asarray([1000.0]),
        particle_mass_kg=np.asarray([mass]),
        dep_particle_rel_permittivity=np.asarray([np.nan]),
        thermophoretic_coefficient=np.asarray([np.nan]),
        end_time_s=0.8,
        duration_s=0.6,
        spatial_dim=2,
        backend=compiled,
        body_acceleration_mps2=np.zeros(2),
        gas_density_kgm3=2.0e-5,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=molecular_mass,
        drag_model_mode=DRAG_MODEL_EPSTEIN,
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=8,
    )
    motion_batch = trace_motion_batch(request)
    minimum_substeps = np.asarray([4], dtype=np.int32)

    paths, _result = sample_piecewise_langevin_paths(
        config=StochasticMotionConfig(
            enabled=True, temperature_source="field_T_then_gas"
        ),
        rng=np.random.default_rng(717),
        motion_batch=motion_batch,
        particle_indices=np.asarray([0], dtype=np.int64),
        minimum_substeps=minimum_substeps,
        particle_mass=np.asarray([mass]),
        gas_temperature_K=300.0,
    )
    path = paths[0]
    deterministic_trace = trace_motion_segment(
        request.particle_request(0).with_minimum_substeps(int(minimum_substeps[0]))
    )
    midpoint_times = deterministic_trace.times_s[0::2]
    midpoint_x = deterministic_trace.coefficient_midpoint_positions_m[:, 0]
    expected_temperature = (
        250.0 + 200.0 * midpoint_times + (50.0 + 50.0 * midpoint_times) * midpoint_x
    )
    thermal_speed = np.sqrt(
        8.0 * K_BOLTZMANN * expected_temperature / (np.pi * molecular_mass)
    )
    expected_tau = (
        3.0
        * mass
        / ((1.0 + np.pi / 8.0) * np.pi * diameter * diameter * 2.0e-5 * thermal_speed)
    )

    assert int(minimum_substeps[0]) > 4
    assert motion_batch.local_error_resolved.tolist() == [True]
    assert np.ptp(expected_temperature) > 0.0
    np.testing.assert_allclose(
        path.thermal_velocity_variance_m2s2,
        K_BOLTZMANN * expected_temperature / mass,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        path.tau_eff_s, deterministic_trace.tau_mid_s, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        deterministic_trace.tau_mid_s, expected_tau, rtol=2.0e-14
    )


@pytest.mark.parametrize("varying_coefficient", ["temperature", "viscosity"])
def test_brownian_coefficients_refine_a_deterministically_stationary_schedule(
    varying_coefficient: str,
) -> None:
    axes = (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    times = np.asarray([0.0, 1.0])
    spatial_shape = (2, 2)
    temperature_delta = 3.0e-2 if varying_coefficient == "temperature" else 0.0
    viscosity_delta = 1.8e-8 if varying_coefficient == "viscosity" else 0.0
    temperature = np.stack(
        (
            np.full(spatial_shape, 300.0),
            np.full(spatial_shape, 300.0 + temperature_delta),
        )
    )
    viscosity = np.stack(
        (
            np.full(spatial_shape, 1.8e-5),
            np.full(spatial_shape, 1.8e-5 + viscosity_delta),
        )
    )
    shape = temperature.shape
    compiled = RegularRectilinearCompiledBackend(
        axes=axes,
        times=times,
        ux=np.zeros(shape),
        uy=np.zeros(shape),
        gas_density=np.ones(shape),
        gas_mu=viscosity,
        gas_temperature=temperature,
        valid_mask=np.ones(spatial_shape, dtype=bool),
        core_valid_mask=np.ones(spatial_shape, dtype=bool),
        gas_mu_source=(
            "field:mu" if varying_coefficient == "viscosity" else "context:gas"
        ),
        gas_temperature_source=(
            "field:T" if varying_coefficient == "temperature" else "context:gas"
        ),
    )
    mass = 1.0e-15
    request = SegmentMotionBatchRequest(
        position_m=np.asarray([[0.5, 0.5]]),
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
        backend=compiled,
        body_acceleration_mps2=np.zeros(2),
        gas_density_kgm3=1.0,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=6.63e-26,
        drag_model_mode=DRAG_MODEL_STOKES,
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=8,
    )
    motion_batch = trace_motion_batch(request)
    minimum_substeps = motion_batch.substep_count.copy()

    paths, _result = sample_piecewise_langevin_paths(
        config=StochasticMotionConfig(
            enabled=True, temperature_source="field_T_then_gas"
        ),
        rng=np.random.default_rng(719),
        motion_batch=motion_batch,
        particle_indices=np.asarray([0], dtype=np.int64),
        minimum_substeps=minimum_substeps,
        particle_mass=np.asarray([mass]),
        gas_temperature_K=300.0,
    )

    assert motion_batch.substep_count.tolist() == [1]
    leaf_count = int(minimum_substeps[0])
    assert leaf_count > 1
    assert motion_batch.local_error_resolved.tolist() == [True]
    np.testing.assert_allclose(
        paths[0].leaf_end_times_s,
        np.arange(1, leaf_count + 1, dtype=np.float64) / float(leaf_count),
        rtol=0.0,
        atol=0.0,
    )
    expected_temperature = 300.0 + temperature_delta * (
        np.arange(leaf_count, dtype=np.float64) + 0.5
    ) / float(leaf_count)
    np.testing.assert_allclose(
        paths[0].thermal_velocity_variance_m2s2,
        K_BOLTZMANN * expected_temperature / mass,
        rtol=2.0e-14,
    )
    if varying_coefficient == "viscosity":
        expected_viscosity = 1.8e-5 + viscosity_delta * (
            np.arange(leaf_count, dtype=np.float64) + 0.5
        ) / float(leaf_count)
        np.testing.assert_allclose(
            paths[0].tau_eff_s,
            mass / (3.0 * np.pi * expected_viscosity * 1.0e-6),
            rtol=2.0e-14,
        )
