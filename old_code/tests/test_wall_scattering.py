from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from particle_tracer_unified.solvers.wall_response import sample_diffuse_reflection


class _FixedRng:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(low + (high - low) * self.value)


def _fixed_rng(value: float) -> np.random.Generator:
    return cast(np.random.Generator, _FixedRng(value))


def test_diffuse_reflection_2d_uses_cosine_law_inverse_cdf() -> None:
    normal = np.asarray([1.0, 0.0], dtype=np.float64)

    assert sample_diffuse_reflection(
        _fixed_rng(0.5), normal, 2.0
    ).tolist() == pytest.approx([-2.0, 0.0])
    assert sample_diffuse_reflection(
        _fixed_rng(0.0), normal, 2.0
    ).tolist() == pytest.approx([0.0, -2.0])
    assert sample_diffuse_reflection(
        _fixed_rng(1.0), normal, 2.0
    ).tolist() == pytest.approx([0.0, 2.0])
