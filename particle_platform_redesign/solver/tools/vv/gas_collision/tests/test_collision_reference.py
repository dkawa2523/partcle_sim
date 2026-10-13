"""Analytic limits check the external oracle; they do not qualify a new solver model."""

import math

import numpy as np
import pytest
from tools.vv.gas_collision.collision_reference import (
    BOLTZMANN_J_K,
    SphereBath,
    collision_moments,
    outgoing_flux_moments,
    sample_collision_jumps,
    sample_frozen_collision_increments,
    surface_moment_coefficients,
)


def test_outgoing_flux_moments_match_rayleigh_and_gaussian_limits() -> None:
    first, normal_second, tangent_second = outgoing_flux_moments()
    assert first == pytest.approx(math.sqrt(math.pi / 2.0), rel=2.0e-13)
    assert normal_second == pytest.approx(2.0, rel=2.0e-13)
    assert tangent_second == pytest.approx(1.0, rel=2.0e-13)


def test_specular_hemisphere_impulse_moments_have_independent_analytic_coefficients() -> None:
    bath = SphereBath(1.0, 0.02, 1.0, 300.0, 1.0, 0.0, 300.0)
    first, parallel, transverse = surface_moment_coefficients(bath, specular_recoil=True)
    recoil = bath.mass_ratio / (1.0 + bath.mass_ratio)
    assert first[1] == pytest.approx(math.pi * recoil, rel=2.0e-13)
    assert parallel[2] == pytest.approx(4.0 * math.pi / 3.0 * recoil**2, rel=2.0e-13)
    assert transverse[2] == pytest.approx(math.pi / 3.0 * recoil**2, rel=2.0e-13)


@pytest.mark.parametrize("sigma", [0.0, 0.9, 1.0])
def test_heavy_equal_temperature_zero_slip_reproduces_linear_fdt(sigma: float) -> None:
    bath = SphereBath(1.2e-21, 1.3e-25, 6.0e-5, 310.0, 1.0e-8, sigma, 310.0)
    moment = collision_moments(bath, 0.0, gas_order=128)
    rate = 4.0 * math.pi / 3.0 * (bath.diameter_m / 2.0) ** 2 * bath.gas_density_kg_m3
    rate *= math.sqrt(8.0 / math.pi) * bath.thermal_speed_m_s / bath.particle_mass_kg
    rate *= 1.0 + sigma * math.pi / 8.0
    fdt = 2.0 * rate * BOLTZMANN_J_K * bath.gas_temperature_K / bath.particle_mass_kg
    np.testing.assert_allclose(np.diag(moment.covariance_rate_m2_s3), fdt, rtol=3.0e-6)


def test_flux_weighted_collision_sampling_recovers_quadrature_moments() -> None:
    bath = SphereBath(1.2e-21, 1.3e-25, 6.0e-5, 310.0, 1.0e-8, 0.9, 310.0)
    reference = collision_moments(bath, 4.0, gas_order=96)
    jumps = sample_collision_jumps(bath, 4.0, 200000, 80123)
    mean = np.mean(jumps, axis=0)
    second = np.mean(jumps**2, axis=0)
    mean_se = np.std(jumps, axis=0, ddof=1) / math.sqrt(jumps.shape[0])
    second_se = np.std(jumps**2, axis=0, ddof=1) / math.sqrt(jumps.shape[0])
    expected_mean = reference.acceleration_m_s2 / reference.collision_rate_s_inv
    expected_second = np.diag(reference.covariance_rate_m2_s3) / reference.collision_rate_s_inv
    assert np.max(np.abs((mean - expected_mean) / mean_se)) < 5.0
    assert np.max(np.abs((second - expected_second) / second_se)) < 5.0


def test_frozen_poisson_increments_reproduce_independent_mean_and_covariance_rate() -> None:
    bath = SphereBath(1.2e-21, 1.3e-25, 6.0e-5, 310.0, 1.0e-8, 0.9, 310.0)
    reference = collision_moments(bath, 4.0, gas_order=96)
    duration = 0.25e-6
    increments = sample_frozen_collision_increments(
        bath, 4.0, reference.collision_rate_s_inv, duration, 100000, 90314
    )
    expected_mean = reference.acceleration_m_s2 * duration
    expected_variance = np.diag(reference.covariance_rate_m2_s3) * duration
    mean_se = np.std(increments, axis=0, ddof=1) / math.sqrt(increments.shape[0])
    squared = (increments - expected_mean) ** 2
    variance_se = np.std(squared, axis=0, ddof=1) / math.sqrt(increments.shape[0])
    assert np.max(np.abs((np.mean(increments, axis=0) - expected_mean) / mean_se)) < 5.0
    assert np.max(np.abs((np.mean(squared, axis=0) - expected_variance) / variance_se)) < 5.0
