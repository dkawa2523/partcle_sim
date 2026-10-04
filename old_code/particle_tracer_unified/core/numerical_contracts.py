"""Small float64 comparison rules shared by input and adapter contracts."""

from __future__ import annotations

import numpy as np

FLOAT64_CONTRACT_ULPS = 64.0


def float_values_equal_ulps(
    first: float,
    second: float,
    *,
    ulps: float = FLOAT64_CONTRACT_ULPS,
) -> bool:
    left = float(first)
    right = float(second)
    count = float(ulps)
    if not np.isfinite(left) or not np.isfinite(right) or count < 0.0:
        return False
    magnitude = np.float64(max(abs(left), abs(right)))
    with np.errstate(over="ignore", invalid="ignore"):
        tolerance = count * abs(float(np.spacing(magnitude)))
    if not np.isfinite(tolerance):
        return left == right
    return bool(abs(left - right) <= tolerance)


def float_arrays_equal_ulps(
    first: np.ndarray,
    second: np.ndarray,
    *,
    ulps: float = FLOAT64_CONTRACT_ULPS,
) -> bool:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.shape != right.shape
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
    ):
        return False
    return all(
        float_values_equal_ulps(a, b, ulps=ulps)
        for a, b in zip(left.reshape(-1), right.reshape(-1), strict=True)
    )


__all__ = (
    "FLOAT64_CONTRACT_ULPS",
    "float_arrays_equal_ulps",
    "float_values_equal_ulps",
)
