from __future__ import annotations

import numpy as np
import pytest

from particle_tracer_unified.solvers._stochastic_randomness import (
    draw_particle_path_randomness,
)


def _draw(
    particle_id: int,
    *,
    dimension: int = 2,
    leaf_count: int = 4,
    macro_step_index: int = 3,
    cohort_index: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return draw_particle_path_randomness(
        seed=712,
        particle_id=particle_id,
        macro_step_index=macro_step_index,
        cohort_index=cohort_index,
        leaf_count=leaf_count,
        dimension=dimension,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_keyed_randomness_is_reproducible_float64_with_int64_bridge_seeds(
    dimension: int,
) -> None:
    first = _draw(9001, dimension=dimension)
    second = _draw(9001, dimension=dimension)

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert first[0].shape == (4, dimension)
    assert first[1].shape == (4, dimension)
    assert first[2].shape == (4,)
    assert first[0].dtype == np.float64
    assert first[1].dtype == np.float64
    assert first[2].dtype == np.int64


def test_particle_draws_ignore_other_particle_count_and_iteration_order() -> None:
    expected = _draw(42)

    forward = {particle_id: _draw(particle_id) for particle_id in (42, 81)}
    expanded_reverse = {
        particle_id: _draw(particle_id) for particle_id in (103, 81, 42)
    }

    for actual in (forward[42], expanded_reverse[42]):
        for left, right in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(left, right)


def test_leaf_extension_and_spatial_dimension_keep_existing_draw_identity() -> None:
    baseline = _draw(42, dimension=2, leaf_count=4)
    extended = _draw(42, dimension=3, leaf_count=7)

    np.testing.assert_array_equal(extended[0][:4, :2], baseline[0])
    np.testing.assert_array_equal(extended[1][:4, :2], baseline[1])
    np.testing.assert_array_equal(extended[2][:4], baseline[2])


@pytest.mark.parametrize(
    "overrides",
    [
        {"particle_id": 43},
        {"macro_step_index": 4},
        {"cohort_index": 2},
    ],
)
def test_particle_macro_and_release_cohort_select_distinct_streams(
    overrides: dict[str, int],
) -> None:
    baseline = _draw(42)
    arguments = {"particle_id": 42, **overrides}
    changed = _draw(**arguments)

    assert not np.array_equal(baseline[0], changed[0])
    assert not np.array_equal(baseline[1], changed[1])
    assert not np.array_equal(baseline[2], changed[2])


def test_public_seed_selects_a_distinct_stream() -> None:
    baseline = draw_particle_path_randomness(
        seed=712,
        particle_id=42,
        macro_step_index=3,
        cohort_index=1,
        leaf_count=4,
        dimension=2,
    )
    changed = draw_particle_path_randomness(
        seed=713,
        particle_id=42,
        macro_step_index=3,
        cohort_index=1,
        leaf_count=4,
        dimension=2,
    )

    for left, right in zip(baseline, changed, strict=True):
        assert not np.array_equal(left, right)


@pytest.mark.parametrize(
    ("leaf_count", "dimension"),
    [(-1, 2), (1, 0)],
)
def test_invalid_path_shape_is_rejected(leaf_count: int, dimension: int) -> None:
    with pytest.raises(ValueError, match="Brownian path shape"):
        _draw(42, leaf_count=leaf_count, dimension=dimension)


def test_keyed_velocity_and_position_draws_have_normal_statistics() -> None:
    velocity, position, _bridge_seeds = _draw(901, leaf_count=4096)
    samples = np.concatenate((velocity.ravel(), position.ravel()))

    assert float(np.mean(samples)) == pytest.approx(0.0, abs=0.04)
    assert float(np.std(samples)) == pytest.approx(1.0, abs=0.04)
