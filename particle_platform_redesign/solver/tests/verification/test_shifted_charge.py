from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from chamber_particles.numerical_status import (
    INTEGRATOR_ACCURACY_FAILURE,
    NUMERICAL_STATUS_OK,
)
from chamber_particles.physics.catalog import resolve_physics_plan
from chamber_particles.physics.charge import (
    oml_shifted_maxwellian_global_bounds,
    oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1,
    oml_stationary_maxwellian_debye_huckel_v1,
    shifted_maxwellian_ion_factors,
)
from chamber_particles.physics.forces import BOLTZMANN_J_K, PhysicsEvaluationError
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime

_ION_MASS_KG = 6.6335209e-26
_REVISION = "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1"


def _charge_model(maximum_drift_ratio: float = 2.0) -> dict[str, object]:
    return {
        "model": "plasma_continuous",
        "revision": _REVISION,
        "electron_number_density_field": "electron_density",
        "positive_ion_number_density_field": "ion_density",
        "electron_temperature_field": "electron_temperature",
        "positive_ion_temperature_field": "ion_temperature",
        "positive_ion_velocity_field": "ion_velocity",
        "positive_ion_mass_kg": _ION_MASS_KG,
        "maximum_ion_drift_ratio": maximum_drift_ratio,
        "applicability": "error",
    }


def _range(lower: tuple[float, ...], upper: tuple[float, ...]) -> PrimitiveRange:
    return PrimitiveRange(np.asarray(lower), np.asarray(upper), None)


def _runtime(
    maximum_drift_ratio: float = 2.0,
    *,
    ion_velocity_lower_m_s: tuple[float, float] = (-500.0, -100.0),
    ion_velocity_upper_m_s: tuple[float, float] = (1_000.0, 300.0),
):
    plan = resolve_physics_plan({"charge": _charge_model(maximum_drift_ratio)}, "cartesian_xy")
    return prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.full(4, 2.0e-15),
        drag_diameter_m=np.full(4, 2.0e-6),
        electrostatic_radius_m=np.asarray([1.0e-8, 1.5e-8, 2.0e-8, 1.2e-8]),
        displaced_volume_m3=np.zeros(4),
        charge_number=np.asarray([-50.0, -20.0, -5.0, 0.0]),
        primitive_ranges={
            "electron_density": _range((1.0e14,), (1.0e14,)),
            "ion_density": _range((1.0e14,), (1.0e14,)),
            "electron_temperature": _range((2.0e4,), (2.0e4,)),
            "ion_temperature": _range((500.0,), (500.0,)),
            "ion_velocity": _range(ion_velocity_lower_m_s, ion_velocity_upper_m_s),
        },
    )


def _sampled(ion_velocity_m_s: np.ndarray) -> dict[str, np.ndarray]:
    count = int(ion_velocity_m_s.shape[0])
    return {
        "electron_density": np.full((count, 1), 1.0e14),
        "ion_density": np.full((count, 1), 1.0e14),
        "electron_temperature": np.full((count, 1), 2.0e4),
        "ion_temperature": np.full((count, 1), 500.0),
        "ion_velocity": ion_velocity_m_s,
    }


def test_shifted_maxwellian_moments_match_zero_shift_and_velocity_quadrature() -> None:
    ratio = np.asarray([0.0, 1.0e-12, 0.5, 2.0, 5.0])
    neutral, attraction = shifted_maxwellian_ion_factors(ratio)

    assert neutral[0] == 1.0
    assert attraction[0] == 1.0
    assert neutral[1] == pytest.approx(1.0, rel=0.0, abs=1.0e-15)
    assert attraction[1] == pytest.approx(1.0, rel=0.0, abs=1.0e-15)
    assert bool((np.diff(neutral) >= 0.0).all())
    assert bool((np.diff(attraction) <= 0.0).all())

    nodes, weights = np.polynomial.hermite.hermgauss(64)
    normal_nodes = math.sqrt(2.0) * nodes
    normal_weights = weights / math.sqrt(math.pi)
    x, y, z = np.meshgrid(normal_nodes, normal_nodes, normal_nodes, indexing="ij")
    weight = (
        normal_weights[:, None, None]
        * normal_weights[None, :, None]
        * normal_weights[None, None, :]
    )
    for shift in (0.5, 2.0):
        speed = np.sqrt((x + shift) ** 2 + y**2 + z**2)
        neutral_quadrature = float(np.sum(weight * speed)) / math.sqrt(8.0 / math.pi)
        attraction_quadrature = math.sqrt(math.pi / 2.0) * float(np.sum(weight / speed))
        expected_neutral, expected_attraction = shifted_maxwellian_ion_factors(np.asarray([shift]))
        assert neutral_quadrature == pytest.approx(expected_neutral[0], rel=1.0e-4)
        assert attraction_quadrature == pytest.approx(expected_attraction[0], rel=1.0e-2)


def test_shifted_oml_zero_drift_reduces_to_stationary_negative_branch() -> None:
    charge = np.asarray([-100.0, -20.0, 0.0])
    radius = np.asarray([1.0e-8, 1.5e-8, 2.0e-8])
    density = np.asarray([1.0e14, 2.0e14, 5.0e13])
    electron_temperature = np.asarray([2.0e4, 3.0e4, 4.0e4])
    ion_temperature = np.asarray([300.0, 500.0, 700.0])
    velocity = np.zeros((3, 2))
    arguments = {
        "charge_number": charge,
        "electrostatic_radius_m": radius,
        "electron_number_density_m3": density,
        "positive_ion_number_density_m3": density,
        "electron_temperature_K": electron_temperature,
        "positive_ion_temperature_K": ion_temperature,
        "particle_velocity_m_s": velocity,
        "positive_ion_velocity_m_s": velocity,
        "positive_ion_mass_kg": _ION_MASS_KG,
    }

    shifted = oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1(
        **arguments,
        maximum_ion_drift_ratio=2.0,
    )
    stationary = oml_stationary_maxwellian_debye_huckel_v1(**arguments)

    np.testing.assert_allclose(
        shifted.charge_rate_number_s,
        stationary.charge_rate_number_s,
        rtol=4.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        shifted.charge_rate_derivative_s_inv,
        stationary.charge_rate_derivative_s_inv,
        rtol=4.0e-15,
        atol=0.0,
    )


def test_shifted_oml_derivative_is_negative_and_matches_left_finite_difference() -> None:
    charges = (-80.0, -15.0, 0.0)
    for mean_speed_ratio in (0.0, 0.75, 2.0):
        ion_mean_speed = math.sqrt(8.0 * BOLTZMANN_J_K * 500.0 / (math.pi * _ION_MASS_KG))
        ion_velocity = np.asarray([[mean_speed_ratio * ion_mean_speed, 0.0]])
        for charge in charges:
            step = 1.0e-6 * max(abs(charge), 1.0)
            center = np.asarray([charge])
            left = center - step
            evaluation = oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1(
                charge_number=center,
                electrostatic_radius_m=np.asarray([1.0e-8]),
                electron_number_density_m3=np.asarray([1.0e14]),
                positive_ion_number_density_m3=np.asarray([1.0e14]),
                electron_temperature_K=np.asarray([2.0e4]),
                positive_ion_temperature_K=np.asarray([500.0]),
                particle_velocity_m_s=np.zeros((1, 2)),
                positive_ion_velocity_m_s=ion_velocity,
                positive_ion_mass_kg=_ION_MASS_KG,
                maximum_ion_drift_ratio=2.0,
            )
            left_evaluation = oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1(
                charge_number=left,
                electrostatic_radius_m=np.asarray([1.0e-8]),
                electron_number_density_m3=np.asarray([1.0e14]),
                positive_ion_number_density_m3=np.asarray([1.0e14]),
                electron_temperature_K=np.asarray([2.0e4]),
                positive_ion_temperature_K=np.asarray([500.0]),
                particle_velocity_m_s=np.zeros((1, 2)),
                positive_ion_velocity_m_s=ion_velocity,
                positive_ion_mass_kg=_ION_MASS_KG,
                maximum_ion_drift_ratio=2.0,
            )
            finite_difference = (
                evaluation.charge_rate_number_s[0] - left_evaluation.charge_rate_number_s[0]
            ) / step
            assert evaluation.charge_rate_derivative_s_inv[0] < 0.0
            assert finite_difference == pytest.approx(
                evaluation.charge_rate_derivative_s_inv[0],
                rel=3.0e-5,
            )


def test_shifted_oml_global_bounds_enclose_primitive_corners_and_reject_bad_domain() -> None:
    bounds = oml_shifted_maxwellian_global_bounds(
        initial_charge_number=np.asarray([-20.0, 0.0]),
        electrostatic_radius_m=np.asarray([1.0e-8, 3.0e-8]),
        electron_number_density_lower_m3=8.0e13,
        electron_number_density_upper_m3=1.2e14,
        positive_ion_number_density_lower_m3=8.0e13,
        positive_ion_number_density_upper_m3=1.2e14,
        electron_temperature_lower_K=2.0e4,
        electron_temperature_upper_K=4.0e4,
        positive_ion_temperature_lower_K=300.0,
        positive_ion_temperature_upper_K=600.0,
        positive_ion_mass_kg=_ION_MASS_KG,
        maximum_ion_drift_ratio=2.0,
    )
    assert bounds.charge_number_lower < -20.0
    assert bounds.charge_number_upper == 0.0

    corners = itertools.product(
        (1.0e-8, 3.0e-8),
        (8.0e13, 1.2e14),
        (8.0e13, 1.2e14),
        (2.0e4, 4.0e4),
        (300.0, 600.0),
        (0.0, 2.0),
        (bounds.charge_number_lower, 0.0),
    )
    for radius, ne, ni, te, ti, drift_ratio, charge in corners:
        ion_mean_speed = math.sqrt(8.0 * BOLTZMANN_J_K * ti / (math.pi * _ION_MASS_KG))
        result = oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1(
            charge_number=np.asarray([charge]),
            electrostatic_radius_m=np.asarray([radius]),
            electron_number_density_m3=np.asarray([ne]),
            positive_ion_number_density_m3=np.asarray([ni]),
            electron_temperature_K=np.asarray([te]),
            positive_ion_temperature_K=np.asarray([ti]),
            particle_velocity_m_s=np.zeros((1, 2)),
            positive_ion_velocity_m_s=np.asarray([[drift_ratio * ion_mean_speed, 0.0]]),
            positive_ion_mass_kg=_ION_MASS_KG,
            maximum_ion_drift_ratio=2.0,
        )
        if charge == bounds.charge_number_lower:
            assert result.charge_rate_number_s[0] >= 0.0
        else:
            assert result.charge_rate_number_s[0] <= 0.0
        assert abs(result.charge_rate_number_s[0]) <= bounds.charge_rate_abs_upper_number_s
        assert (
            abs(result.charge_rate_derivative_s_inv[0])
            <= bounds.charge_rate_derivative_abs_upper_s_inv
        )

    common = {
        "electrostatic_radius_m": np.asarray([1.0e-8]),
        "electron_number_density_lower_m3": 1.0e14,
        "electron_number_density_upper_m3": 1.0e14,
        "positive_ion_number_density_lower_m3": 1.0e14,
        "positive_ion_number_density_upper_m3": 1.0e14,
        "electron_temperature_lower_K": 2.0e4,
        "electron_temperature_upper_K": 2.0e4,
        "positive_ion_temperature_lower_K": 500.0,
        "positive_ion_temperature_upper_K": 500.0,
        "positive_ion_mass_kg": _ION_MASS_KG,
        "maximum_ion_drift_ratio": 2.0,
    }
    with pytest.raises(PhysicsEvaluationError, match="nonpositive initial"):
        oml_shifted_maxwellian_global_bounds(
            initial_charge_number=np.asarray([1.0]),
            **common,
        )
    with pytest.raises(PhysicsEvaluationError, match="nonpositive equilibrium"):
        oml_shifted_maxwellian_global_bounds(
            initial_charge_number=np.asarray([0.0]),
            **(common | {"electron_number_density_lower_m3": 1.0e8}),
        )


def test_shifted_oml_compiled_runtime_matches_reference_and_path_gate() -> None:
    runtime = _runtime()
    ion_mean_speed = math.sqrt(8.0 * BOLTZMANN_J_K * 500.0 / (math.pi * _ION_MASS_KG))
    particle_velocity = np.asarray([[0.0, 0.0], [100.0, -40.0], [-50.0, 30.0], [0.0, 0.0]])
    ion_velocity = np.asarray(
        [
            [0.0, 0.0],
            [0.8 * ion_mean_speed, 0.0],
            [0.0, -1.5 * ion_mean_speed],
            [2.1 * ion_mean_speed, 0.0],
        ]
    )
    charge = np.asarray([-50.0, -20.0, -5.0, 0.0])
    sampled = _sampled(ion_velocity)
    actual, status = runtime.evaluate_batch(
        np.arange(4, dtype=np.int64),
        particle_velocity,
        charge,
        sampled,
    )
    expected = oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1(
        charge_number=charge,
        electrostatic_radius_m=runtime.electrostatic_radius_m,
        electron_number_density_m3=sampled["electron_density"][:, 0],
        positive_ion_number_density_m3=sampled["ion_density"][:, 0],
        electron_temperature_K=sampled["electron_temperature"][:, 0],
        positive_ion_temperature_K=sampled["ion_temperature"][:, 0],
        particle_velocity_m_s=particle_velocity,
        positive_ion_velocity_m_s=ion_velocity,
        positive_ion_mass_kg=_ION_MASS_KG,
        maximum_ion_drift_ratio=2.0,
    )

    np.testing.assert_array_equal(status, NUMERICAL_STATUS_OK)
    np.testing.assert_allclose(
        actual.charge_rate_number_s,
        expected.charge_rate_number_s,
        rtol=5.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual.charge_rate_derivative_s_inv,
        expected.charge_rate_derivative_s_inv,
        rtol=5.0e-15,
        atol=0.0,
    )
    assert bool((actual.charge_rate_derivative_s_inv < 0.0).all())
    np.testing.assert_array_equal(actual.applicable, expected.applicable)
    np.testing.assert_array_equal(actual.applicable, [True, True, True, False])

    assert runtime.charge_bounds is not None
    outside = np.asarray([np.nextafter(runtime.charge_bounds.model.charge_number_upper, np.inf)])
    _, outside_status = runtime.evaluate_batch(
        np.asarray([0]),
        np.zeros((1, 2)),
        outside,
        _sampled(np.zeros((1, 2))),
    )
    np.testing.assert_array_equal(outside_status, INTEGRATOR_ACCURACY_FAILURE)

    path_runtime = _runtime(
        ion_velocity_lower_m_s=(0.0, 0.0),
        ion_velocity_upper_m_s=(0.0, 0.0),
    )
    path_applicable, path_status = path_runtime.continuous_applicability_batch(
        np.asarray([0, 1]),
        np.asarray([[0.5 * ion_mean_speed, 0.0], [2.1 * ion_mean_speed, 0.0]]),
    )
    np.testing.assert_array_equal(path_status, NUMERICAL_STATUS_OK)
    np.testing.assert_array_equal(path_applicable, [True, False])


def test_shifted_charge_catalog_requires_explicit_finite_drift_envelope() -> None:
    plan = resolve_physics_plan({"charge": _charge_model(3.0)}, "axisymmetric_rz")
    assert plan.charge is not None
    assert plan.charge.revision == _REVISION
    assert plan.charge.maximum_ion_drift_ratio == 3.0
    assert plan.resolved_models()["charge"] == {
        "model": "plasma_continuous",
        "revision": _REVISION,
    }

    invalid = _charge_model()
    invalid.pop("maximum_ion_drift_ratio")
    with pytest.raises(ValueError, match="keys do not match"):
        resolve_physics_plan({"charge": invalid}, "cartesian_xy")
    with pytest.raises(ValueError, match="positive finite"):
        resolve_physics_plan(
            {"charge": _charge_model(math.inf)},
            "cartesian_xy",
        )
