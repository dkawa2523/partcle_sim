from __future__ import annotations

import numpy as np

from chamber_particles.numerical_status import (
    INTEGRATOR_ACCURACY_FAILURE,
    NUMERICAL_STATUS_OK,
)
from chamber_particles.physics.catalog import resolve_physics_plan
from chamber_particles.physics.charge import (
    oml_stationary_maxwellian_debye_huckel_v1,
)
from chamber_particles.physics.forces import electric_acceleration_abs_upper
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime

_ION_MASS_KG = 6.6335209e-26


def _charge_model() -> dict[str, object]:
    return {
        "model": "plasma_continuous",
        "revision": "oml_stationary_maxwellian_debye_huckel_v1",
        "electron_number_density_field": "electron_density",
        "positive_ion_number_density_field": "ion_density",
        "electron_temperature_field": "electron_temperature",
        "positive_ion_temperature_field": "ion_temperature",
        "positive_ion_velocity_field": "ion_velocity",
        "positive_ion_mass_kg": _ION_MASS_KG,
        "applicability": "error",
    }


def _scalar_range(value: float) -> PrimitiveRange:
    array = np.asarray([value], dtype="<f8")
    return PrimitiveRange(array, array.copy(), array.copy())


def _vector_range(value: tuple[float, float]) -> PrimitiveRange:
    array = np.asarray(value, dtype="<f8")
    return PrimitiveRange(array, array.copy(), array.copy())


def _primitive_ranges(*, electric: bool = False) -> dict[str, PrimitiveRange]:
    result = {
        "electron_density": _scalar_range(1.0e14),
        "ion_density": _scalar_range(1.0e14),
        "electron_temperature": _scalar_range(10_000.0),
        "ion_temperature": _scalar_range(300.0),
        "ion_velocity": _vector_range((0.0, 0.0)),
    }
    if electric:
        result["electric_field"] = _vector_range((120.0, -80.0))
    return result


def _sampled(
    count: int, *, velocity_x_m_s: float = 0.0, electric: bool = False
) -> dict[str, np.ndarray]:
    result = {
        "electron_density": np.full((count, 1), 1.0e14, dtype="<f8"),
        "ion_density": np.full((count, 1), 1.0e14, dtype="<f8"),
        "electron_temperature": np.full((count, 1), 10_000.0, dtype="<f8"),
        "ion_temperature": np.full((count, 1), 300.0, dtype="<f8"),
        "ion_velocity": np.column_stack(
            (
                np.full(count, velocity_x_m_s, dtype="<f8"),
                np.zeros(count, dtype="<f8"),
            )
        ),
    }
    if electric:
        result["electric_field"] = np.broadcast_to(
            np.asarray([120.0, -80.0], dtype="<f8"),
            (count, 2),
        ).copy()
    return result


def _runtime(
    initial_charge: np.ndarray,
    *,
    electric: bool = False,
):
    models: dict[str, dict[str, object]] = {"charge": _charge_model()}
    if electric:
        models["electric"] = {
            "model": "coulomb",
            "revision": "electric_coulomb_v1",
            "electric_field": "electric_field",
        }
    plan = resolve_physics_plan(models, "cartesian_xy")
    count = int(initial_charge.size)
    return prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.full(count, 2.0e-15, dtype="<f8"),
        drag_diameter_m=np.full(count, 2.0e-6, dtype="<f8"),
        electrostatic_radius_m=np.full(count, 1.0e-8, dtype="<f8"),
        displaced_volume_m3=np.zeros(count, dtype="<f8"),
        charge_number=initial_charge,
        primitive_ranges=_primitive_ranges(electric=electric),
    )


def test_compiled_oml_stage_matches_scalar_reference_across_potential_branches() -> None:
    charge = np.asarray([-50.0, 0.0, 50.0, 0.0], dtype="<f8")
    runtime = _runtime(charge)
    particle_index = np.arange(charge.size, dtype="<i8")
    velocity = np.asarray(
        [[2.0, -1.0], [0.0, 0.0], [-3.0, 1.5], [1.0e308, 1.0e308]],
        dtype="<f8",
    )
    sampled = _sampled(charge.size)

    evaluation, status = runtime.evaluate_batch(
        particle_index,
        velocity,
        charge,
        sampled,
    )
    reference = oml_stationary_maxwellian_debye_huckel_v1(
        charge_number=charge,
        electrostatic_radius_m=np.full(charge.size, 1.0e-8, dtype="<f8"),
        electron_number_density_m3=sampled["electron_density"][:, 0],
        positive_ion_number_density_m3=sampled["ion_density"][:, 0],
        electron_temperature_K=sampled["electron_temperature"][:, 0],
        positive_ion_temperature_K=sampled["ion_temperature"][:, 0],
        particle_velocity_m_s=velocity,
        positive_ion_velocity_m_s=sampled["ion_velocity"],
        positive_ion_mass_kg=_ION_MASS_KG,
    )

    np.testing.assert_array_equal(status, NUMERICAL_STATUS_OK)
    np.testing.assert_allclose(
        evaluation.charge_rate_number_s,
        reference.charge_rate_number_s,
        rtol=4.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(evaluation.applicable, reference.applicable)


def test_charge_invariant_escape_is_an_integrator_accuracy_failure() -> None:
    initial_charge = np.asarray([0.0], dtype="<f8")
    runtime = _runtime(initial_charge)
    assert runtime.charge_bounds is not None
    outside = np.asarray(
        [np.nextafter(runtime.charge_bounds.model.charge_number_upper, np.inf)],
        dtype="<f8",
    )

    evaluation, status = runtime.evaluate_batch(
        np.asarray([0], dtype="<i8"),
        np.zeros((1, 2), dtype="<f8"),
        outside,
        _sampled(1),
    )

    np.testing.assert_array_equal(status, INTEGRATOR_ACCURACY_FAILURE)
    np.testing.assert_array_equal(evaluation.charge_rate_number_s, [0.0])


def test_dynamic_electric_enclosure_uses_invariant_charge_and_disables_exact_path() -> None:
    initial_charge = np.asarray([0.0], dtype="<f8")
    runtime = _runtime(initial_charge, electric=True)
    assert runtime.charge_bounds is not None
    maximum_abs_charge = max(
        abs(runtime.charge_bounds.model.charge_number_lower),
        abs(runtime.charge_bounds.model.charge_number_upper),
    )
    expected = electric_acceleration_abs_upper(
        charge_number=np.asarray([maximum_abs_charge], dtype="<f8"),
        mass_kg=np.asarray([2.0e-15], dtype="<f8"),
        electric_field_abs_upper_V_m=np.nextafter(
            np.asarray([120.0, 80.0], dtype="<f8"),
            np.inf,
        ),
    )
    expected = np.nextafter(expected, np.inf)

    np.testing.assert_array_equal(runtime.external_acceleration_abs_upper_m_s2, expected)
    assert bool((runtime.external_acceleration_abs_upper_m_s2 > 0.0).all())
    assert runtime.constant_acceleration_m_s2 is None
    assert runtime.maximum_dt_charge_lipschitz(1.0) == (
        runtime.charge_bounds.model.charge_rate_derivative_abs_upper_s_inv
    )
    assert runtime.maximum_charge_rate_abs_number_s == (
        runtime.charge_bounds.model.charge_rate_abs_upper_number_s
    )
    assert runtime.localizable_external_base_abs_upper_m_s2 is not None
    expected_bound_bytes = (
        expected.nbytes
        + runtime.localizable_external_base_abs_upper_m_s2.nbytes
        + runtime.charge_bounds.positive_ion_velocity_abs_upper_m_s.nbytes
        + 9 * np.dtype("<f8").itemsize
    )
    assert runtime.bound_array_nbytes == expected_bound_bytes


def test_oml_local_and_continuous_drift_applicability_are_consistent() -> None:
    runtime = _runtime(np.asarray([0.0], dtype="<f8"))
    particle_index = np.asarray([0], dtype="<i8")
    velocity = np.asarray([[50.0, 0.0]], dtype="<f8")

    evaluation, status = runtime.evaluate_batch(
        particle_index,
        velocity,
        np.asarray([0.0], dtype="<f8"),
        _sampled(1),
    )
    continuous, continuous_status = runtime.continuous_applicability_batch(
        particle_index,
        np.abs(velocity),
    )

    np.testing.assert_array_equal(status, NUMERICAL_STATUS_OK)
    np.testing.assert_array_equal(continuous_status, NUMERICAL_STATUS_OK)
    np.testing.assert_array_equal(evaluation.applicable, [False])
    np.testing.assert_array_equal(continuous, [False])
