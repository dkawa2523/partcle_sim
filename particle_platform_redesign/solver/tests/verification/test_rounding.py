from __future__ import annotations

import numpy as np
from numba import njit

from chamber_particles.cpu import next_float64


@njit(cache=False, fastmath=False, parallel=False, error_model="numpy")
def _compiled_neighbors(values: np.ndarray, toward_positive: bool) -> tuple[np.ndarray, np.ndarray]:
    actual = np.empty(values.size, dtype=np.float64)
    reference = np.empty(values.size, dtype=np.float64)
    target = np.inf if toward_positive else -np.inf
    for row in range(values.size):
        actual[row] = next_float64(values[row], toward_positive)
        reference[row] = np.nextafter(values[row], target)
    return actual, reference


def test_binary64_neighbors_match_numpy_across_binades_and_random_bits() -> None:
    generator = np.random.default_rng(61012)
    random_bits = generator.integers(
        0, np.iinfo(np.uint64).max, size=200_000, dtype=np.uint64, endpoint=True
    )
    boundaries = np.arange(1, 2047, dtype=np.uint64) << np.uint64(52)
    positive = np.concatenate((boundaries - np.uint64(1), boundaries, boundaries + np.uint64(1)))
    bits = np.concatenate((random_bits, positive, positive | np.uint64(0x8000000000000000)))
    values = bits.view(np.float64)

    for toward_positive in (False, True):
        actual, compiled_reference = _compiled_neighbors(values, toward_positive)
        with np.errstate(all="ignore"):
            reference = np.nextafter(values, np.inf if toward_positive else -np.inf)
        np.testing.assert_array_equal(actual.view(np.uint64), compiled_reference.view(np.uint64))
        np.testing.assert_array_equal(actual.view(np.uint64), reference.view(np.uint64))


def test_binary64_neighbors_preserve_signed_zero_overflow_and_nan_payload_semantics() -> None:
    # Zero, the smallest subnormal, the largest finite value, and infinity,
    # with both signs.  NaN payload behavior belongs to the platform oracle.
    finite_and_infinite = np.asarray(
        [
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000001,
            0x8000000000000001,
            0x7FEFFFFFFFFFFFFF,
            0xFFEFFFFFFFFFFFFF,
            0x7FF0000000000000,
            0xFFF0000000000000,
        ],
        dtype=np.uint64,
    )
    expected_positive = np.asarray(
        [
            0x0000000000000001,
            0x0000000000000001,
            0x0000000000000002,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFEFFFFFFFFFFFFE,
            0x7FF0000000000000,
            0xFFEFFFFFFFFFFFFF,
        ],
        dtype=np.uint64,
    )
    expected_negative = np.asarray(
        [
            0x8000000000000001,
            0x8000000000000001,
            0x0000000000000000,
            0x8000000000000002,
            0x7FEFFFFFFFFFFFFE,
            0xFFF0000000000000,
            0x7FEFFFFFFFFFFFFF,
            0xFFF0000000000000,
        ],
        dtype=np.uint64,
    )
    positive_nan = np.asarray(
        [0x7FF0000000000001, 0x7FF8000000000001, 0x7FFFFFFFFFFFFFFF], dtype=np.uint64
    )
    bits = np.concatenate(
        (finite_and_infinite, positive_nan, positive_nan | np.uint64(0x8000000000000000))
    )
    values = bits.view(np.float64)

    for toward_positive in (False, True):
        actual, compiled_reference = _compiled_neighbors(values, toward_positive)
        with np.errstate(all="ignore"):
            reference = np.nextafter(values, np.inf if toward_positive else -np.inf)
        expected = expected_positive if toward_positive else expected_negative
        np.testing.assert_array_equal(actual[: finite_and_infinite.size].view(np.uint64), expected)
        np.testing.assert_array_equal(actual.view(np.uint64), compiled_reference.view(np.uint64))
        np.testing.assert_array_equal(actual.view(np.uint64), reference.view(np.uint64))
