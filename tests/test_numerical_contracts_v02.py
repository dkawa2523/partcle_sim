from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from particle_tracer_unified.core.numerical_contracts import (
    FLOAT64_CONTRACT_ULPS,
    float_arrays_equal_ulps,
    float_values_equal_ulps,
)


def test_float_contract_accepts_only_roundoff_scale_differences() -> None:
    value = np.float64(1.0e-13)
    one_ulp_above = np.nextafter(value, np.float64(np.inf))

    assert float_values_equal_ulps(value, one_ulp_above)
    assert not float_values_equal_ulps(value, value + np.float64(1.0e-14))


def test_float_array_contract_rejects_fixed_absolute_tolerance_aliasing() -> None:
    geometry_axis = np.asarray([0.0, 1.0e-13, 2.0e-13], dtype=np.float64)
    roundoff_equivalent = geometry_axis.copy()
    roundoff_equivalent[1] = np.nextafter(roundoff_equivalent[1], np.float64(np.inf))
    physically_different = geometry_axis.copy()
    physically_different[1] += 1.0e-14

    assert float_arrays_equal_ulps(geometry_axis, roundoff_equivalent)
    assert not float_arrays_equal_ulps(geometry_axis, physically_different)


def test_float_contract_rejects_nonfinite_values_and_invalid_ulp_count() -> None:
    assert FLOAT64_CONTRACT_ULPS == 64.0
    assert not float_values_equal_ulps(np.nan, 0.0)
    assert not float_values_equal_ulps(np.inf, np.inf)
    assert not float_values_equal_ulps(0.0, 0.0, ulps=-1.0)
    assert not float_values_equal_ulps(np.finfo(np.float64).max, 0.0)
    assert not float_arrays_equal_ulps(
        np.asarray([0.0, np.nan]), np.asarray([0.0, np.nan])
    )


@given(
    first=st.floats(width=64, allow_nan=False, allow_infinity=False),
    second=st.floats(width=64, allow_nan=False, allow_infinity=False),
)
def test_float_value_contract_is_symmetric_for_finite_float64_values(
    first: float,
    second: float,
) -> None:
    forward = float_values_equal_ulps(first, second)
    reverse = float_values_equal_ulps(second, first)

    assert isinstance(forward, bool)
    assert forward == reverse


@given(exponent=st.integers(min_value=-100, max_value=100))
def test_float_value_contract_honors_the_exact_ulp_boundary(exponent: int) -> None:
    base = np.ldexp(np.float64(1.0), exponent)
    at_boundary = base
    outside_boundary = base
    for _ in range(int(FLOAT64_CONTRACT_ULPS)):
        at_boundary = np.nextafter(at_boundary, np.float64(np.inf))
        outside_boundary = np.nextafter(outside_boundary, np.float64(np.inf))
    outside_boundary = np.nextafter(outside_boundary, np.float64(np.inf))

    assert float_values_equal_ulps(base, at_boundary)
    assert not float_values_equal_ulps(base, outside_boundary)


@given(
    values=st.lists(
        st.floats(width=64, allow_nan=False, allow_infinity=False), max_size=24
    )
)
def test_float_array_contract_preserves_shape_and_finite_value_semantics(
    values: list[float],
) -> None:
    array = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    equivalent = array.copy()

    assert array.dtype == np.dtype(np.float64)
    assert float_arrays_equal_ulps(array, equivalent)
    assert not float_arrays_equal_ulps(array, equivalent.reshape(-1))


@given(nonfinite=st.sampled_from((np.nan, np.inf, -np.inf)))
def test_float_contract_rejects_each_nonfinite_class(nonfinite: float) -> None:
    assert not float_values_equal_ulps(nonfinite, nonfinite)
    assert not float_arrays_equal_ulps(
        np.asarray([0.0, nonfinite], dtype=np.float64),
        np.asarray([0.0, nonfinite], dtype=np.float64),
    )
