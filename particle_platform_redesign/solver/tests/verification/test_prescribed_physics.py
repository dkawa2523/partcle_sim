from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

from chamber_particles.physics.catalog import resolve_physics_plan
from chamber_particles.physics.charge import (
    prescribed_exponential_linear_collection,
    prescribed_exponential_linear_global_bounds,
)
from chamber_particles.physics.forces import ELEMENTARY_CHARGE_C, PhysicsEvaluationError
from chamber_particles.physics.runtime import (
    LocalPrimitiveRange,
    PrimitiveRange,
    prepare_physics_runtime,
)


def _models() -> dict[str, dict[str, object]]:
    properties = {
        "mass_kg": 2.0,
        "drag_diameter_m": 1.0,
        "electrostatic_radius_m": 0.5,
        "displaced_volume_m3": 0.0,
    }
    return {
        "charge": {
            "model": "prescribed_exponential_linear",
            "revision": "prescribed_exponential_linear_collection_v1",
            "electron_rate_reference_number_s": 2.0,
            "electron_exponent_offset_field": "c",
            "electron_charge_slope_field": "alpha",
            "ion_rate_field": "Ai",
            "ion_charge_slope_field": "betaZ",
            "particle_properties": properties,
        },
        "drag": {
            "model": "prescribed_linear_friction",
            "revision": "prescribed_linear_friction_v1",
            "gas_velocity_field": "u",
            "friction_coefficient_field": "friction",
            "particle_properties": properties,
        },
        "noise": {
            "model": "prescribed_force_diffusion",
            "revision": "prescribed_force_diffusion_midpoint_2d_v1",
            "force_spectral_density_field": "D",
            "interval_tree_depth": 1,
            "particle_properties": properties,
        },
        "ion_drag": {
            "model": "prescribed_charge_polynomial",
            "revision": "prescribed_charge_polynomial_v1",
            "constant_force_field": "F0",
            "linear_charge_force_field": "F1",
            "quadratic_charge_force_field": "F2",
            "particle_properties": properties,
        },
        "lift": {
            "model": "prescribed_rotational_lift",
            "revision": "prescribed_rotational_lift_v1",
            "gas_velocity_field": "u",
            "rotation_coefficient_field": "rotation",
            "particle_properties": properties,
        },
    }


def _sampled() -> dict[str, np.ndarray]:
    values = {
        "u": [1.0, -2.0],
        "friction": [6.0],
        "D": [12.0],
        "c": [-1.0],
        "alpha": [0.5],
        "Ai": [1.0],
        "betaZ": [2.0],
        "F0": [1.0, 2.0],
        "F1": [-3.0, 4.0],
        "F2": [5.0, -6.0],
        "rotation": [6.0],
    }
    return {name: np.tile(value, (4, 1)) for name, value in values.items()}


def _runtime(
    models: dict[str, dict[str, object]], sampled: dict[str, np.ndarray], **changes: np.ndarray
):
    properties = {
        "mass_kg": np.full(4, 2.0),
        "drag_diameter_m": np.ones(4),
        "electrostatic_radius_m": np.full(4, 0.5),
        "displaced_volume_m3": np.zeros(4),
        "charge_number": np.asarray([-3.0, -1.0, 0.0, 3.0]),
    }
    properties.update(changes)
    ranges = {
        name: PrimitiveRange(value.min(axis=0), value.max(axis=0), value[0])
        for name, value in sampled.items()
    }
    return prepare_physics_runtime(
        plan=resolve_physics_plan(models, "cartesian_xy"),
        coordinate_system="cartesian_xy",
        primitive_ranges=ranges,
        **properties,
    )


def test_prescribed_models_couple_same_stage_charge_velocity_and_full_bounds() -> None:
    sampled = _sampled()
    runtime = _runtime(_models(), sampled)
    indices = np.arange(4, dtype="<i8")
    z = np.asarray([-3.0, -1.0, 0.0, 3.0])
    velocity = np.tile([4.0, -4.0], (4, 1))
    actual = runtime.evaluate(indices, velocity, z, sampled)
    expected_additive = np.column_stack((6.5 - 1.5 * z + 2.5 * z**2, 10.0 + 2.0 * z - 3.0 * z**2))
    np.testing.assert_allclose(actual.additive_acceleration_m_s2, expected_additive, rtol=2.0e-15)
    np.testing.assert_allclose(
        actual.acceleration_m_s2, expected_additive + np.asarray([-9.0, 6.0])
    )
    np.testing.assert_array_equal(actual.linear_drag_rate_s_inv, np.full(4, 3.0))
    np.testing.assert_array_equal(
        runtime.brownian_thermal_velocity_variance(indices, sampled), np.full(4, 0.5)
    )
    np.testing.assert_allclose(
        actual.charge_rate_number_s, 1.0 - 2.0 * z - 2.0 * np.exp(-1.0 + 0.5 * z)
    )
    np.testing.assert_allclose(actual.charge_rate_derivative_s_inv, -2.0 - np.exp(-1.0 + 0.5 * z))
    bound = runtime.acceleration_abs_upper(indices, np.abs(velocity))
    assert np.all(bound >= np.abs(actual.acceleration_m_s2))
    assert runtime.continuous_applicability(indices, np.abs(velocity)).all()
    local = {
        name: LocalPrimitiveRange(value.copy(), value.copy()) for name, value in sampled.items()
    }
    verdict, status = runtime.local_continuous_applicability_batch(
        indices, velocity, velocity, z, z, local
    )
    assert verdict.all()
    assert not status.any()


@pytest.mark.parametrize(
    "property_name", ["mass_kg", "drag_diameter_m", "electrostatic_radius_m", "displaced_volume_m3"]
)
def test_prescribed_coefficients_reject_stale_particle_property_binding(property_name: str) -> None:
    value = {
        "mass_kg": 2.0,
        "drag_diameter_m": 1.0,
        "electrostatic_radius_m": 0.5,
        "displaced_volume_m3": 0.0,
    }[property_name]
    changed = np.full(4, value)
    changed[2] = np.nextafter(value, np.inf)
    with pytest.raises(PhysicsEvaluationError, match="particle_properties"):
        _runtime(_models(), _sampled(), **{property_name: changed})


@pytest.mark.parametrize(
    "name,value", [("friction", 0.0), ("D", 0.0), ("Ai", -1.0), ("alpha", -1.0), ("betaZ", -1.0)]
)
def test_prescribed_invalid_coefficient_ranges_fail_prepare(name: str, value: float) -> None:
    sampled = _sampled()
    sampled[name][1, 0] = value
    with pytest.raises(PhysicsEvaluationError):
        _runtime(_models(), sampled)


def test_prescribed_charge_wide_exponent_and_invariant_match_decimal_oracle() -> None:
    z = np.asarray([-3.0, -1.0, 0.0, 3.0])
    c = np.asarray([-1200.0, -900.0, -600.0, -470.0])
    rate, derivative = prescribed_exponential_linear_collection(
        z, 1.0e200, c, np.full(4, 0.5), np.zeros(4), np.full(4, 2.0)
    )
    with localcontext() as context:
        context.prec = 60
        expected = np.asarray(
            [
                float(
                    -(
                        Decimal("1e200")
                        * (Decimal(str(offset)) + Decimal("0.5") * Decimal(str(charge))).exp()
                    )
                )
                for offset, charge in zip(c, z, strict=True)
            ]
        )
    np.testing.assert_allclose(rate, expected, rtol=7.0e-14, atol=0.0)
    np.testing.assert_allclose(derivative, expected * 0.5, rtol=7.0e-14, atol=0.0)
    bound = prescribed_exponential_linear_global_bounds(
        z, 2.0, (-3.0, -1.0), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0)
    )
    rng = np.random.default_rng(2)
    for charge in (bound.charge_number_lower, bound.charge_number_upper):
        offsets = rng.uniform(-3.0, -1.0, 2000)
        slopes = rng.uniform(0.2, 0.5, 2000)
        ion = rng.uniform(0.5, 1.0, 2000)
        beta = rng.uniform(1.0, 2.0, 2000)
        g, j = prescribed_exponential_linear_collection(
            np.full(2000, charge), 2.0, offsets, slopes, ion, beta
        )
        assert np.all(g > 0.0) if charge < 0.0 else np.all(g < 0.0)
        assert np.max(np.abs(g)) <= bound.charge_rate_abs_upper_number_s
        assert np.max(np.abs(j)) <= bound.charge_rate_derivative_abs_upper_s_inv


def test_electric_charge_cap_does_not_clip_charge_or_ion_state() -> None:
    sampled = _sampled()
    sampled["E"] = np.tile([1.0e20, -2.0e20], (4, 1))
    models = _models()
    models["electric"] = {
        "model": "coulomb",
        "revision": "electric_coulomb_charge_cap_v1",
        "electric_field": "E",
        "maximum_abs_charge_number": 1.0,
    }
    capped = _runtime(models, sampled)
    z = np.asarray([-3.0, -1.0, 0.0, 3.0])
    velocity = np.tile([4.0, -4.0], (4, 1))
    actual = capped.evaluate(np.arange(4, dtype="<i8"), velocity, z, sampled)
    reference = _runtime(_models(), sampled).evaluate(
        np.arange(4, dtype="<i8"), velocity, z, sampled
    )
    expected = ELEMENTARY_CHARGE_C * np.clip(z, -1.0, 1.0)[:, None] * sampled["E"] / 2.0
    np.testing.assert_allclose(actual.acceleration_m_s2 - reference.acceleration_m_s2, expected)
    np.testing.assert_array_equal(actual.charge_rate_number_s, reference.charge_rate_number_s)
    np.testing.assert_array_equal(
        actual.charge_rate_derivative_s_inv, reference.charge_rate_derivative_s_inv
    )
    assert np.all(
        capped.acceleration_abs_upper(np.arange(4, dtype="<i8"), np.abs(velocity))
        >= np.abs(actual.acceleration_m_s2)
    )


def test_electric_cap_tightens_global_local_and_constant_certificates() -> None:
    sampled = {"E": np.tile([1.0e20, -2.0e20], (4, 1))}
    models: dict[str, dict[str, object]] = {
        "charge": {"model": "fixed"},
        "electric": {
            "model": "coulomb",
            "revision": "electric_coulomb_charge_cap_v1",
            "electric_field": "E",
            "maximum_abs_charge_number": 1.0,
        },
    }
    runtime = _runtime(models, sampled)
    indices = np.arange(4, dtype="<i8")
    z = np.asarray([-3.0, -1.0, 0.0, 3.0])
    velocity = np.zeros((4, 2))
    expected = ELEMENTARY_CHARGE_C * np.clip(z, -1.0, 1.0)[:, None] * sampled["E"] / 2.0
    np.testing.assert_array_equal(runtime.constant_acceleration_m_s2, expected)
    global_bound = runtime.acceleration_abs_upper(indices, velocity)
    np.testing.assert_allclose(global_bound, np.abs(expected), atol=1.0e-300, rtol=3.0e-14)
    local = {"E": LocalPrimitiveRange(sampled["E"], sampled["E"])}
    bound, applicable, status = runtime.local_additive_acceleration_abs_upper_batch(
        indices, velocity, velocity, z, z, local
    )
    assert applicable.all() and not status.any()
    np.testing.assert_allclose(bound, np.abs(expected), atol=1.0e-300, rtol=2.0e-13)


def test_prescribed_stage_failures_remain_particle_local() -> None:
    sampled = _sampled()
    runtime = _runtime(_models(), sampled)
    sampled["friction"][1, 0] = 0.0
    indices = np.arange(4, dtype="<i8")
    _, status = runtime.evaluate_batch(indices, np.zeros((4, 2)), np.zeros(4), sampled)
    assert status[1] != 0
    np.testing.assert_array_equal(status[[0, 2, 3]], np.zeros(3))
    _, noise_status = runtime.brownian_thermal_velocity_variance_batch(indices, sampled)
    assert noise_status[1] != 0
    np.testing.assert_array_equal(noise_status[[0, 2, 3]], np.zeros(3))
