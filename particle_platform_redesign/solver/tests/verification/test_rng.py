from __future__ import annotations

import numpy as np
import pytest

from chamber_particles.rng import (
    BROWNIAN_RNG_REVISION,
    BROWNIAN_ROOT_NORMAL_STREAM,
    BROWNIAN_SPLIT_NORMAL_STREAM,
    SOURCE_FACET_DRAW,
    SOURCE_POSITION_DRAW,
    brownian_normal_pair_batch,
    philox4x32_10,
    source_uniform_open,
    source_uniform_open_batch,
    wall_uniform,
    wall_uniform_batch,
)


def test_philox4x32_10_matches_random123_zero_known_answer() -> None:
    assert philox4x32_10((0, 0, 0, 0), (0, 0)) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )


def test_wall_uniform_has_published_word_order_and_counter_sensitivity() -> None:
    zero_block = philox4x32_10((0, 0, 0, 0), (0, 0))
    zero_integer = ((zero_block[0] << 32) | zero_block[1]) >> 11
    assert float(zero_integer) * 2.0**-53 == pytest.approx(0.3990464708489645, rel=0.0, abs=0.0)
    assert wall_uniform(0, 0, 0) == pytest.approx(0.4173433195660301, rel=0.0, abs=0.0)
    reference = wall_uniform(123, 456, 7)
    assert reference != wall_uniform(124, 456, 7)
    assert reference != wall_uniform(123, 457, 7)
    assert reference != wall_uniform(123, 456, 8)
    assert 0.0 <= reference < 1.0


def test_source_draws_are_open_and_domain_separated() -> None:
    facet = source_uniform_open(9, 3, 11, SOURCE_FACET_DRAW)
    position = source_uniform_open(9, 3, 11, SOURCE_POSITION_DRAW)
    assert 0.0 < facet < 1.0
    assert 0.0 < position < 1.0
    assert facet != position
    assert facet != source_uniform_open(9, 4, 11, SOURCE_FACET_DRAW)
    assert facet != source_uniform_open(9, 3, 12, SOURCE_FACET_DRAW)


def test_batched_source_draws_match_the_scalar_counter_convention() -> None:
    ordinals = np.asarray([0, 1, 2, 11, 2**32 + 3], dtype=np.uint64)

    actual = source_uniform_open_batch(9, 3, ordinals, SOURCE_POSITION_DRAW)
    expected = np.asarray(
        [source_uniform_open(9, 3, int(ordinal), SOURCE_POSITION_DRAW) for ordinal in ordinals],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(actual, expected)


def test_batched_wall_draws_match_scalar_for_full_counter_width() -> None:
    particle_id = np.asarray([0, 1, 2**32 + 7, 2**64 - 1], dtype=np.uint64)
    ordinal = np.asarray([0, 3, 2**16, 2**32 - 1], dtype=np.uint64)

    actual = wall_uniform_batch(2**64 - 1, particle_id, ordinal)
    expected = np.asarray(
        [
            wall_uniform(2**64 - 1, int(particle), int(event))
            for particle, event in zip(particle_id, ordinal, strict=True)
        ]
    )

    np.testing.assert_array_equal(actual, expected)


def test_brownian_normals_have_stable_interval_tree_identity() -> None:
    particles = np.asarray([0, 1, 2**32 + 7], dtype=np.uint64)
    root = brownian_normal_pair_batch(
        0,
        particles,
        0,
        0,
        tree_level=0,
        tree_index=0,
        component=0,
        draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
    )

    assert BROWNIAN_RNG_REVISION == "philox4x32_10_brownian_interval_tree_v1"
    np.testing.assert_allclose(
        root[0],
        np.asarray([1.4413314144879381, -0.6577651968007232]),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_array_equal(
        root,
        brownian_normal_pair_batch(
            0,
            particles[::-1],
            0,
            0,
            tree_level=0,
            tree_index=0,
            component=0,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        )[::-1],
    )
    variants = (
        brownian_normal_pair_batch(
            1,
            particles,
            0,
            0,
            tree_level=0,
            tree_index=0,
            component=0,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        ),
        brownian_normal_pair_batch(
            0,
            particles,
            1,
            0,
            tree_level=0,
            tree_index=0,
            component=0,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        ),
        brownian_normal_pair_batch(
            0,
            particles,
            0,
            0,
            tree_level=0,
            tree_index=0,
            component=1,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        ),
        brownian_normal_pair_batch(
            0,
            particles,
            0,
            0,
            tree_level=3,
            tree_index=5,
            component=0,
            draw_kind=BROWNIAN_SPLIT_NORMAL_STREAM,
        ),
        brownian_normal_pair_batch(
            0,
            particles,
            0,
            1,
            tree_level=0,
            tree_index=0,
            component=0,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        ),
        brownian_normal_pair_batch(
            0,
            particles,
            2**32,
            0,
            tree_level=0,
            tree_index=0,
            component=0,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        ),
        brownian_normal_pair_batch(
            0,
            particles,
            0,
            2**32,
            tree_level=0,
            tree_index=0,
            component=0,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        ),
        brownian_normal_pair_batch(
            0,
            particles,
            0,
            0,
            tree_level=33,
            tree_index=2**32 + 5,
            component=0,
            draw_kind=BROWNIAN_SPLIT_NORMAL_STREAM,
        ),
    )
    assert all(not np.array_equal(root, variant) for variant in variants)


@pytest.mark.parametrize(
    ("counter", "key"),
    [
        ((-1, 0, 0, 0), (0, 0)),
        ((0, 0, 0, 1 << 32), (0, 0)),
        ((0, 0, 0, 0), (-1, 0)),
    ],
)
def test_philox_rejects_words_outside_uint32(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> None:
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        philox4x32_10(counter, key)
