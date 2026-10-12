from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numba import njit

from chamber_particles.engine import (
    _FAILURE_MODEL_APPLICABILITY,
    _curved_row_failure_codes,
    _NumericSupportContext,
    _StageDynamics,
)
from chamber_particles.fields import PreparedFieldSet
from chamber_particles.integrators import cubic_hermite_step
from chamber_particles.physics.catalog import resolve_physics_plan
from chamber_particles.physics.forces import (
    CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE,
    CONTINUOUS_APPLICABILITY_OK,
    EPSTEIN_MIN_LAMBDA_OVER_RADIUS,
    PhysicsEvaluationError,
)
from chamber_particles.physics.runtime import (
    LocalPrimitiveRange,
    PhysicsRuntimeWorkspace,
    PrimitiveRange,
    evaluate_local_continuous_applicability_into,
    local_additive_acceleration_abs_upper_into,
    prepare_physics_runtime,
)

_GAS_MOLECULAR_MASS_KG = 1.2753471408396638e-25
_ION_MASS_KG = 6.6335209e-26
_RELATIVE_ION_SPEED_LIMIT_M_S = 25_000.0


def test_signed_local_ranges_certify_coflow_rejected_by_global_abs_bounds() -> None:
    runtime = _coflow_runtime()
    particle = np.asarray([0], dtype=np.int64)
    velocity = np.asarray([[100_000.0, 0.0]])

    global_certified, global_status = runtime.continuous_applicability_batch(
        particle,
        np.abs(velocity),
    )
    local_certified, local_status = runtime.local_continuous_applicability_batch(
        particle,
        velocity - np.asarray([[1.0, 0.5]]),
        velocity + np.asarray([[1.0, 0.5]]),
        np.asarray([-10.0]),
        np.asarray([-10.0]),
        _local_ranges(ion_velocity_x_m_s=90_001.0),
    )

    np.testing.assert_array_equal(global_certified, [False])
    np.testing.assert_array_equal(global_status, [CONTINUOUS_APPLICABILITY_OK])
    np.testing.assert_array_equal(local_certified, [True])
    np.testing.assert_array_equal(local_status, [CONTINUOUS_APPLICABILITY_OK])


def test_local_threshold_miss_is_inconclusive_not_a_numerical_failure() -> None:
    runtime = _coflow_runtime()
    particle = np.asarray([0, 0], dtype=np.int64)
    velocity_lower = np.asarray([[100_000.0, 0.0], [100_000.0, 0.0]])
    velocity_upper = velocity_lower.copy()
    ranges = _local_ranges(
        ion_velocity_x_m_s=np.asarray([75_001.0, 74_999.0]),
    )

    certified, status = runtime.local_continuous_applicability_batch(
        particle,
        velocity_lower,
        velocity_upper,
        np.full(2, -10.0),
        np.full(2, -10.0),
        ranges,
    )

    np.testing.assert_array_equal(certified, [True, False])
    np.testing.assert_array_equal(
        status,
        [CONTINUOUS_APPLICABILITY_OK, CONTINUOUS_APPLICABILITY_OK],
    )


def test_local_epstein_threshold_is_rounded_toward_inconclusive() -> None:
    runtime = _coflow_runtime()
    particle = np.asarray([0], dtype=np.int64)
    velocity = np.asarray([[100_000.0, 0.0]])
    threshold_m = EPSTEIN_MIN_LAMBDA_OVER_RADIUS * 5.0e-8

    at_threshold, threshold_status = runtime.local_continuous_applicability_batch(
        particle,
        velocity,
        velocity,
        np.asarray([-10.0]),
        np.asarray([-10.0]),
        _local_ranges(
            ion_velocity_x_m_s=90_001.0,
            gas_mean_free_path_m=threshold_m,
        ),
    )
    above_threshold, above_status = runtime.local_continuous_applicability_batch(
        particle,
        velocity,
        velocity,
        np.asarray([-10.0]),
        np.asarray([-10.0]),
        _local_ranges(
            ion_velocity_x_m_s=90_001.0,
            gas_mean_free_path_m=threshold_m * (1.0 + 1.0e-10),
        ),
    )

    np.testing.assert_array_equal(at_threshold, [False])
    np.testing.assert_array_equal(threshold_status, [CONTINUOUS_APPLICABILITY_OK])
    np.testing.assert_array_equal(above_threshold, [True])
    np.testing.assert_array_equal(above_status, [CONTINUOUS_APPLICABILITY_OK])


def test_continuous_charge_dense_interval_must_stay_in_prepared_invariant() -> None:
    runtime = _coflow_runtime()
    particle = np.asarray([0], dtype=np.int64)
    velocity = np.asarray([[100_000.0, 0.0]])

    certified, status = runtime.local_continuous_applicability_batch(
        particle,
        velocity,
        velocity,
        np.asarray([-10.0]),
        np.asarray([1.0]),
        _local_ranges(ion_velocity_x_m_s=90_001.0),
    )

    np.testing.assert_array_equal(certified, [False])
    np.testing.assert_array_equal(status, [CONTINUOUS_APPLICABILITY_OK])


def test_charge_interval_prepared_invariant_is_a_standalone_global_fast_gate() -> None:
    runtime = _coflow_runtime()

    certified, status = runtime.charge_interval_inside_prepared_invariant(
        np.asarray([-10.0, -10.0, np.nan]),
        np.asarray([-4.0, 1.0, -4.0]),
    )

    np.testing.assert_array_equal(certified, [True, False, False])
    np.testing.assert_array_equal(
        status,
        [
            CONTINUOUS_APPLICABILITY_OK,
            CONTINUOUS_APPLICABILITY_OK,
            CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE,
        ],
    )


def test_cubic_hermite_dense_charge_crossing_prepared_invariant_fails_closed() -> None:
    runtime = _charge_only_runtime()
    assert runtime.charge_bounds is not None
    charge_upper = runtime.charge_bounds.model.charge_number_upper
    proposal = cubic_hermite_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.asarray([[0.0, 0.0]]),
        np.asarray([[0.0, 0.0]]),
        np.asarray([charge_upper - 1.0]),
        np.asarray([[0.0, 0.0]]),
        np.asarray([[0.0, 0.0]]),
        end_charge_number=np.asarray([np.nextafter(charge_upper, np.inf)]),
        support_inside=np.asarray([True]),
        applicability_inside=np.asarray([True]),
        numerical_status=np.zeros(1, dtype=np.uint8),
    )
    prepared = cast(
        Any,
        SimpleNamespace(
            support_context=_NumericSupportContext(True, (-1.0, -1.0), (1.0, 1.0), True, False),
            fields=SimpleNamespace(
                regular_support_box=lambda: (
                    np.asarray([-1.0, -1.0]),
                    np.asarray([1.0, 1.0]),
                )
            ),
            geometry=SimpleNamespace(
                facet_count=0,
                nodes_m=np.asarray([[-1.0, -1.0], [1.0, 1.0]]),
            ),
            case=SimpleNamespace(
                data=SimpleNamespace(coordinate_system="cartesian_xy"),
            ),
            dynamics=_StageDynamics(
                runtime=runtime,
                fields=PreparedFieldSet(None, {}),
                coordinate_system="cartesian_xy",
                last_field_cell=None,
                field_workspace=PreparedFieldSet(None, {}).allocate_workspace(1),
                physics_workspace=PhysicsRuntimeWorkspace.allocate(1),
            ),
        ),
    )

    codes = _curved_row_failure_codes(prepared, proposal, np.asarray([0]))

    np.testing.assert_array_equal(codes, [_FAILURE_MODEL_APPLICABILITY])


def test_local_applicability_rejects_malformed_ranges_without_defaulting() -> None:
    runtime = _coflow_runtime()
    particle = np.asarray([0], dtype=np.int64)
    velocity = np.asarray([[100_000.0, 0.0]])
    reversed_ranges = _local_ranges(ion_velocity_x_m_s=90_001.0)
    reversed_ranges["ug"] = LocalPrimitiveRange(
        np.asarray([[100_001.0, 0.0]]),
        np.asarray([[99_999.0, 0.0]]),
    )

    with pytest.raises(PhysicsEvaluationError, match="reversed"):
        runtime.local_continuous_applicability_batch(
            particle,
            velocity,
            velocity,
            np.asarray([-10.0]),
            np.asarray([-10.0]),
            reversed_ranges,
        )

    certified, status = runtime.local_continuous_applicability_batch(
        particle,
        velocity + 1.0,
        velocity,
        np.asarray([-10.0]),
        np.asarray([-10.0]),
        _local_ranges(ion_velocity_x_m_s=90_001.0),
    )
    np.testing.assert_array_equal(certified, [False])
    np.testing.assert_array_equal(
        status,
        [CONTINUOUS_APPLICABILITY_NUMERICAL_FAILURE],
    )


def test_barnes_static_gate_uses_local_fields_and_dense_charge_interval() -> None:
    runtime = _barnes_runtime_with_remote_invalid_mean_free_path()
    particle = np.asarray([0], dtype=np.int64)
    velocity = np.asarray([[0.0, 0.0]])
    local_ranges = {
        "ne": _local_scalar_range(1.0e14, 1),
        "ni": _local_scalar_range(1.0e14, 1),
        "te": _local_scalar_range(3.0e4, 1),
        "ti": _local_scalar_range(300.0, 1),
        "ui": LocalPrimitiveRange(
            np.asarray([[20.0, 0.0]]),
            np.asarray([[20.0, 0.0]]),
        ),
        "ion_mfp": _local_scalar_range(1.0e-2, 1),
    }

    global_certified, _ = runtime.continuous_applicability_batch(
        particle,
        np.asarray([[20.0, 0.0]]),
    )
    local_certified, local_status = runtime.local_continuous_applicability_batch(
        particle,
        velocity,
        velocity,
        np.asarray([-10.0]),
        np.asarray([-4.0]),
        local_ranges,
    )
    positive_charge_certified, positive_charge_status = (
        runtime.local_continuous_applicability_batch(
            particle,
            velocity,
            velocity,
            np.asarray([-10.0]),
            np.asarray([1.0]),
            local_ranges,
        )
    )

    np.testing.assert_array_equal(global_certified, [False])
    np.testing.assert_array_equal(local_certified, [True])
    np.testing.assert_array_equal(local_status, [CONTINUOUS_APPLICABILITY_OK])
    np.testing.assert_array_equal(positive_charge_certified, [False])
    np.testing.assert_array_equal(positive_charge_status, [CONTINUOUS_APPLICABILITY_OK])


def test_native_local_certificate_skips_unavailable_ranges_and_reuses_prefix() -> None:
    runtime = _coflow_runtime()
    ranges = _local_ranges(ion_velocity_x_m_s=np.full(3, 90_001.0))
    lower = np.column_stack([ranges[item.name].lower for item in runtime.plan.required_fields])
    upper = np.column_stack([ranges[item.name].upper for item in runtime.plan.required_fields])
    lower[1], upper[1] = np.nan, 0.0
    velocity = np.tile([100_000.0, 0.0], (3, 1))
    velocity[2, 0] = np.nan
    certified = np.ones(4, dtype=np.bool_)
    status = np.full(4, 77, dtype=np.uint8)

    # Inputs remain read-only, while output views must be native-call arguments.
    # Passing these columns explicitly avoids closing over mutable Numba globals.
    fill = njit(_fill_local_certificate)
    arguments = (
        runtime.native_context,
        np.zeros(3, dtype=np.int64),
        velocity,
        velocity,
        np.full(3, -10.0),
        np.full(3, -10.0),
        lower,
        upper,
    )
    fill(*arguments, np.asarray([True, False, True]), certified[:3], status[:3])
    np.testing.assert_array_equal(certified, [True, False, False, True])
    np.testing.assert_array_equal(status, [0, 0, 1, 77])
    fill(*arguments, np.zeros(3, dtype=np.bool_), certified[:3], status[:3])
    np.testing.assert_array_equal(certified, [False, False, False, True])
    np.testing.assert_array_equal(status, [0, 0, 0, 77])


@pytest.mark.parametrize("endpoint", [0.0, -1.0, np.nan])
def test_local_certificate_rejects_invalid_primitive_lower_bound(endpoint: float) -> None:
    runtime = _coflow_runtime()
    ranges = _local_ranges(ion_velocity_x_m_s=90_001.0)
    ranges["mfp"] = LocalPrimitiveRange(np.asarray([[endpoint]]), np.asarray([[1.0e-3]]))
    velocity = np.asarray([[100_000.0, 0.0]])
    with pytest.raises(PhysicsEvaluationError):
        runtime.local_continuous_applicability_batch(
            np.asarray([0], dtype=np.int64),
            velocity,
            velocity,
            np.asarray([-10.0]),
            np.asarray([-10.0]),
            ranges,
        )


def _fill_local_certificate(
    context, indices, vlo, vhi, qlo, qhi, lower, upper, available, certified, status
):
    evaluate_local_continuous_applicability_into(
        context, indices, vlo, vhi, qlo, qhi, lower, upper, available, certified, status
    )


def test_native_local_additive_bound_preserves_unavailable_and_invalid_state_meanings() -> None:
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "electric": {
                "model": "coulomb",
                "revision": "electric_coulomb_v1",
                "electric_field": "electric_field",
            },
        },
        "cartesian_xy",
    )
    field = np.asarray([120.0, -80.0])
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([2.0e-18]),
        drag_diameter_m=np.asarray([1.0e-7]),
        electrostatic_radius_m=np.asarray([5.0e-8]),
        displaced_volume_m3=np.zeros(1),
        charge_number=np.asarray([-10.0]),
        primitive_ranges={"electric_field": PrimitiveRange(field, field, field)},
    )
    indices = np.zeros(3, dtype=np.int64)
    velocity = np.zeros((3, 2))
    charge = np.asarray([-10.0, np.nan, -10.0])
    lower = np.tile(field, (3, 1))
    lower[2] = np.nan
    result = np.full((4, 2), -1.0)
    applicable = np.ones(4, dtype=np.bool_)
    status = np.full(4, 77, dtype=np.uint8)

    # An unavailable row has invalid primitive placeholders; another available
    # row has invalid dynamic charge. Only the latter is a numerical failure.
    local_additive_acceleration_abs_upper_into(
        runtime.native_context,
        runtime.native_acceleration_bound_context,
        indices,
        velocity,
        velocity,
        charge,
        charge,
        lower,
        lower,
        np.asarray([True, True, False]),
        result[:3],
        applicable[:3],
        status[:3],
    )
    analytical = 10.0 * 1.602176634e-19 * np.abs(field) / 2.0e-18
    assert np.all(result[0] >= analytical)
    np.testing.assert_allclose(result[0], analytical, rtol=1.0e-12, atol=0.0)
    np.testing.assert_array_equal(result[1:3], np.zeros((2, 2)))
    np.testing.assert_array_equal(result[3], [-1.0, -1.0])
    np.testing.assert_array_equal(applicable, [True, False, False, True])
    np.testing.assert_array_equal(status, [0, 1, 0, 77])


def _coflow_runtime():
    plan = resolve_physics_plan(
        {
            "charge": {
                "model": "plasma_continuous",
                "revision": "aggregate_relative_drift_regularized_two_current_v1",
                "electron_number_density_field": "ne",
                "positive_ion_number_density_field": "ni",
                "electron_thermal_voltage_field": "te_v",
                "positive_ion_thermal_voltage_field": "ti_v",
                "positive_ion_velocity_field": "ui",
                "effective_positive_ion_mass_field": "mi",
                "screening_length_field": "screening",
                "maximum_relative_ion_speed_m_s": _RELATIVE_ION_SPEED_LIMIT_M_S,
                "applicability": "error",
            },
            "drag": {
                "model": "epstein_linear",
                "revision": "epstein_linear_effective_gas_sensitivity_v1",
                "gas_velocity_field": "ug",
                "gas_density_field": "rho",
                "gas_temperature_field": "tg",
                "gas_mean_free_path_field": "mfp",
                "gas_molecular_mass_kg": _GAS_MOLECULAR_MASS_KG,
                "delta": 1.0,
                "maximum_speed_ratio": 0.1,
                "applicability": "error",
            },
        },
        "cartesian_xy",
    )
    global_velocity_lower = np.asarray([-100_000.0, -100_000.0])
    global_velocity_upper = -global_velocity_lower
    ranges = {
        "ug": PrimitiveRange(global_velocity_lower, global_velocity_upper, None),
        "rho": _global_scalar_range(1.0e-4),
        "tg": _global_scalar_range(300.0),
        "mfp": _global_scalar_range(1.0e-3),
        "ne": _global_scalar_range(1.0e14),
        "ni": _global_scalar_range(1.0e14),
        "te_v": _global_scalar_range(3.0),
        "ti_v": _global_scalar_range(0.03),
        "ui": PrimitiveRange(global_velocity_lower, global_velocity_upper, None),
        "mi": _global_scalar_range(_ION_MASS_KG),
        "screening": _global_scalar_range(1.0e-3),
    }
    return prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([2.0e-18]),
        drag_diameter_m=np.asarray([1.0e-7]),
        electrostatic_radius_m=np.asarray([5.0e-8]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([-10.0]),
        primitive_ranges=ranges,
    )


def _charge_only_runtime():
    plan = resolve_physics_plan(
        {
            "charge": {
                "model": "plasma_continuous",
                "revision": "aggregate_relative_drift_regularized_two_current_v1",
                "electron_number_density_field": "ne",
                "positive_ion_number_density_field": "ni",
                "electron_thermal_voltage_field": "te_v",
                "positive_ion_thermal_voltage_field": "ti_v",
                "positive_ion_velocity_field": "ui",
                "effective_positive_ion_mass_field": "mi",
                "screening_length_field": "screening",
                "maximum_relative_ion_speed_m_s": _RELATIVE_ION_SPEED_LIMIT_M_S,
                "applicability": "error",
            },
        },
        "cartesian_xy",
    )
    return prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([2.0e-18]),
        drag_diameter_m=np.asarray([1.0e-7]),
        electrostatic_radius_m=np.asarray([5.0e-8]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([-10.0]),
        primitive_ranges={
            "ne": _global_scalar_range(1.0e14),
            "ni": _global_scalar_range(1.0e14),
            "te_v": _global_scalar_range(3.0),
            "ti_v": _global_scalar_range(0.03),
            "ui": PrimitiveRange(np.zeros(2), np.zeros(2), np.zeros(2)),
            "mi": _global_scalar_range(_ION_MASS_KG),
            "screening": _global_scalar_range(1.0e-3),
        },
    )


def _barnes_runtime_with_remote_invalid_mean_free_path():
    plan = resolve_physics_plan(
        {
            "charge": {"model": "fixed"},
            "ion_drag": {
                "model": "barnes_collisionless",
                "revision": (
                    "barnes_collisionless_effective_speed_single_positive_ion_"
                    "negative_debye_huckel_v1"
                ),
                "electron_number_density_field": "ne",
                "positive_ion_number_density_field": "ni",
                "electron_temperature_field": "te",
                "positive_ion_temperature_field": "ti",
                "positive_ion_velocity_field": "ui",
                "ion_neutral_mean_free_path_field": "ion_mfp",
                "positive_ion_mass_kg": _ION_MASS_KG,
                "maximum_ion_drift_ratio": 1.5,
                "applicability": "error",
            },
        },
        "cartesian_xy",
    )
    return prepare_physics_runtime(
        plan=plan,
        coordinate_system="cartesian_xy",
        mass_kg=np.asarray([1.0e-15]),
        drag_diameter_m=np.asarray([2.0e-7]),
        electrostatic_radius_m=np.asarray([1.0e-7]),
        displaced_volume_m3=np.asarray([0.0]),
        charge_number=np.asarray([-10.0]),
        primitive_ranges={
            "ne": PrimitiveRange(np.asarray([1.0e14]), np.asarray([1.0e14]), None),
            "ni": PrimitiveRange(np.asarray([1.0e14]), np.asarray([1.0e14]), None),
            "te": PrimitiveRange(np.asarray([3.0e4]), np.asarray([3.0e4]), None),
            "ti": PrimitiveRange(np.asarray([300.0]), np.asarray([300.0]), None),
            "ui": PrimitiveRange(
                np.asarray([0.0, 0.0]),
                np.asarray([20.0, 0.0]),
                None,
            ),
            "ion_mfp": PrimitiveRange(
                np.asarray([1.0e-12]),
                np.asarray([1.0e-2]),
                None,
            ),
        },
    )


def _local_ranges(
    *,
    ion_velocity_x_m_s: float | np.ndarray,
    gas_mean_free_path_m: float = 1.0e-3,
) -> dict[str, LocalPrimitiveRange]:
    ion_velocity_x = np.atleast_1d(np.asarray(ion_velocity_x_m_s, dtype=np.float64))
    count = int(ion_velocity_x.size)
    gas_velocity = np.column_stack(
        (np.full(count, 100_000.0), np.zeros(count)),
    )
    ion_velocity = np.column_stack((ion_velocity_x, np.zeros(count)))
    return {
        "ug": LocalPrimitiveRange(gas_velocity - 1.0, gas_velocity + 1.0),
        "rho": _local_scalar_range(1.0e-4, count),
        "tg": _local_scalar_range(300.0, count),
        "mfp": _local_scalar_range(gas_mean_free_path_m, count),
        "ne": _local_scalar_range(1.0e14, count),
        "ni": _local_scalar_range(1.0e14, count),
        "te_v": _local_scalar_range(3.0, count),
        "ti_v": _local_scalar_range(0.03, count),
        "ui": LocalPrimitiveRange(ion_velocity, ion_velocity),
        "mi": _local_scalar_range(_ION_MASS_KG, count),
        "screening": _local_scalar_range(1.0e-3, count),
    }


def _global_scalar_range(value: float) -> PrimitiveRange:
    data = np.asarray([value])
    return PrimitiveRange(data, data, data)


def _local_scalar_range(value: float, count: int) -> LocalPrimitiveRange:
    data = np.full((count, 1), value)
    return LocalPrimitiveRange(data, data)
