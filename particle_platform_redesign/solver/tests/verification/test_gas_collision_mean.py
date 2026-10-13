"""The supported mean drag must match independent molecular/surface quadrature."""

import math

import pytest
from tools.vv.gas_collision.collision_reference import SphereBath, collision_moments
from tools.vv.gas_collision.qualify import Regime, production_mean


@pytest.mark.parametrize("speed_ratio", [0.0, 0.35, 1.0, 4.0, 9.4])
@pytest.mark.parametrize("diffuse_fraction", [0.0, 0.9, 1.0])
def test_finite_speed_sphere_mean_matches_independent_surface_and_gas_integrals(
    speed_ratio: float, diffuse_fraction: float
) -> None:
    bath = SphereBath(1.2e-21, 1.3e-25, 6.0e-5, 310.0, 1.0e-8, diffuse_fraction, 310.0)
    reference = collision_moments(bath, speed_ratio, gas_order=128)
    mean, rate = production_mean(Regime("independent", bath, speed_ratio, 0.002))
    scale = max(abs(mean), rate * bath.thermal_speed_m_s)
    assert abs(mean - reference.acceleration_m_s2[0]) / scale < 3.0e-6
    assert math.isfinite(reference.collision_rate_s_inv)
