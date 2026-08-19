from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from particle_tracer_unified.solvers._stochastic_path import (
    _integrated_ou_covariances,
)
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
)
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
)
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    trace_motion_batch,
)
from particle_tracer_unified.solvers.stochastic_motion import (
    K_BOLTZMANN,
    StochasticMotionConfig,
    sample_piecewise_langevin_paths,
)


def _constant_backend() -> RegularRectilinearCompiledBackend:
    axes = (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))
    shape = (1, 2, 2)
    zeros = np.zeros(shape, dtype=np.float64)
    valid = np.ones((2, 2), dtype=bool)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=np.asarray([0.0]),
        ux=zeros,
        uy=zeros.copy(),
        gas_density=np.full(shape, 200.0),
        gas_mu=np.full(shape, 1.8e-5),
        gas_temperature=np.full(shape, 350.0),
        valid_mask=valid,
        core_valid_mask=valid,
        du_dt_x=zeros.copy(),
        du_dt_y=zeros.copy(),
        grad_ux_x=zeros.copy(),
        grad_ux_y=zeros.copy(),
        grad_uy_x=zeros.copy(),
        grad_uy_y=zeros.copy(),
        gas_temperature_source="context:gas",
    )


def _motion_batch(
    *,
    drag_model_mode: int,
    force_runtime: ForceRuntimeParameters | None = None,
):
    mass = 2.0e-15
    request = SegmentMotionBatchRequest(
        position_m=np.asarray([[0.5, 0.5]]),
        velocity_mps=np.asarray([[0.0, 0.0]]),
        active=np.asarray([True]),
        tau_stokes_s=np.asarray([0.1]),
        particle_diameter_m=np.asarray([1.0e-6]),
        particle_density_kgm3=np.asarray([1000.0]),
        particle_mass_kg=np.asarray([mass]),
        dep_particle_rel_permittivity=np.asarray([np.nan]),
        thermophoretic_coefficient=np.asarray([np.nan]),
        end_time_s=0.02,
        duration_s=0.02,
        spatial_dim=2,
        backend=_constant_backend(),
        body_acceleration_mps2=np.zeros(2),
        gas_density_kgm3=200.0,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=350.0,
        gas_molecular_mass_kg=4.65e-26,
        drag_model_mode=drag_model_mode,
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=4,
        force_runtime=force_runtime,
    )
    return trace_motion_batch(request), mass


def _sample(
    motion_batch,
    mass: float,
    rng: np.random.Generator,
    *,
    collect_diagnostics: bool = False,
):
    return sample_piecewise_langevin_paths(
        config=StochasticMotionConfig(enabled=True, temperature_source="gas"),
        rng=rng,
        motion_batch=motion_batch,
        particle_indices=np.asarray([0], dtype=np.int64),
        minimum_substeps=motion_batch.substep_count,
        particle_mass=np.asarray([mass]),
        gas_temperature_K=350.0,
        collect_diagnostics=collect_diagnostics,
    )


def test_virtual_mass_ou_satisfies_equipartition_and_long_time_diffusion() -> None:
    coefficient = 0.5
    motion_batch, mass = _motion_batch(
        drag_model_mode=DRAG_MODEL_STOKES,
        force_runtime=ForceRuntimeParameters(
            virtual_mass_enabled=True,
            virtual_mass_coefficient=coefficient,
        ),
    )

    paths, result = _sample(
        motion_batch,
        mass,
        np.random.default_rng(831),
        collect_diagnostics=True,
    )
    path = paths[0]
    mass_factor = 1.0 + coefficient * 200.0 / 1000.0
    effective_mass = mass * mass_factor
    theta = K_BOLTZMANN * 350.0 / effective_mass
    tau_stokes = mass / (3.0 * np.pi * 1.8e-5 * 1.0e-6)
    tau = tau_stokes * mass_factor

    assert path.tau_eff_s[0] == pytest.approx(tau, rel=1.0e-15)
    assert path.thermal_velocity_variance_m2s2[0] == pytest.approx(theta, rel=1.0e-15)
    assert effective_mass * path.thermal_velocity_variance_m2s2[0] == pytest.approx(
        K_BOLTZMANN * 350.0, rel=1.0e-15
    )

    _, variance_velocity, _ = _integrated_ou_covariances(20.0 * tau, tau, theta)
    assert variance_velocity == pytest.approx(theta, rel=1.0e-15)

    variance_x_50, _, _ = _integrated_ou_covariances(50.0 * tau, tau, theta)
    variance_x_60, _, _ = _integrated_ou_covariances(60.0 * tau, tau, theta)
    diffusion = (variance_x_60 - variance_x_50) / (20.0 * tau)
    expected_diffusion = K_BOLTZMANN * 350.0 * tau_stokes / mass
    assert diffusion == pytest.approx(expected_diffusion, rel=1.0e-15)
    assert result["applied"] is True

    expected_rng = np.random.default_rng(831)
    np.testing.assert_array_equal(
        path.z_velocity, expected_rng.normal(size=path.z_velocity.shape)
    )
    np.testing.assert_array_equal(
        path.z_position, expected_rng.normal(size=path.z_position.shape)
    )
    np.testing.assert_array_equal(
        path.bridge_seeds,
        expected_rng.integers(
            0,
            np.iinfo(np.int64).max,
            size=path.bridge_seeds.size,
            dtype=np.int64,
        ),
    )


def test_velocity_dependent_drag_is_rejected_before_brownian_random_draws() -> None:
    motion_batch, mass = _motion_batch(drag_model_mode=DRAG_MODEL_SCHILLER_NAUMANN)
    rng = np.random.default_rng(832)
    state_before = deepcopy(rng.bit_generator.state)

    with pytest.raises(ValueError, match="slip-independent linear drag"):
        _sample(motion_batch, mass, rng)

    assert rng.bit_generator.state == state_before
