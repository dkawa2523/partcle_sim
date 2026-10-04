from __future__ import annotations

import numpy as np
import pytest

from particle_tracer_unified.solvers.stochastic_motion import PiecewiseLangevinPath

NIGHTLY_SEEDS = (0, 1, 2, 7, 42, 2**32 - 1)


def _saved_path(seed: int) -> PiecewiseLangevinPath:
    rng = np.random.default_rng(seed)
    leaf_count = 3
    sample_count = 5
    return PiecewiseLangevinPath(
        leaf_end_times_s=np.asarray((0.01, 0.03, 0.08), dtype=np.float64),
        tau_eff_s=np.asarray((0.02, 0.04, 0.06), dtype=np.float64),
        thermal_velocity_variance_m2s2=np.asarray((1.0, 1.5, 2.0), dtype=np.float64),
        z_velocity=rng.normal(size=(leaf_count, sample_count)).astype(np.float64),
        z_position=rng.normal(size=(leaf_count, sample_count)).astype(np.float64),
        bridge_seeds=rng.integers(
            0, np.iinfo(np.int64).max, size=leaf_count, dtype=np.int64
        ),
    )


@pytest.mark.parametrize("seed", NIGHTLY_SEEDS)
def test_saved_brownian_path_is_finite_float64_and_seed_reproducible(seed: int) -> None:
    first = _saved_path(seed)
    second = _saved_path(seed)

    np.testing.assert_array_equal(first.z_velocity, second.z_velocity)
    np.testing.assert_array_equal(first.z_position, second.z_position)
    np.testing.assert_array_equal(first.bridge_seeds, second.bridge_seeds)
    for time_s in (0.0, 0.01, 0.017, 0.03, 0.08):
        position, velocity = first.state_at(time_s)
        replay_position, replay_velocity = first.state_at(time_s)
        assert position.shape == (5,)
        assert velocity.shape == (5,)
        assert position.dtype == np.dtype(np.float64)
        assert velocity.dtype == np.dtype(np.float64)
        assert np.all(np.isfinite(position))
        assert np.all(np.isfinite(velocity))
        np.testing.assert_array_equal(position, replay_position)
        np.testing.assert_array_equal(velocity, replay_velocity)
