from __future__ import annotations

import numpy as np
import pytest

from chamber_particles.coordinates import (
    canonicalize_rz_enclosure,
    fold_rz_position_vector,
    rz_canonical_vector_to_signed,
    rz_signed_stage_to_canonical,
)


def test_signed_rz_stage_mapping_changes_only_the_radial_basis() -> None:
    position = np.asarray([[0.4, 1.0], [-0.25, 2.0], [0.0, 3.0], [-0.0, 4.0]], dtype="<f8")
    vector = np.asarray([[-0.3, 0.5], [-0.6, 0.7], [-0.8, 0.9], [-1.0, 1.1]], dtype="<f8")
    original_position = position.copy()
    original_vector = vector.copy()

    canonical_position, canonical_vector, radial_sign = rz_signed_stage_to_canonical(
        position, vector
    )
    signed_again = rz_canonical_vector_to_signed(canonical_vector, radial_sign)

    np.testing.assert_array_equal(
        canonical_position,
        [[0.4, 1.0], [0.25, 2.0], [0.0, 3.0], [0.0, 4.0]],
    )
    np.testing.assert_array_equal(
        canonical_vector,
        [[-0.3, 0.5], [0.6, 0.7], [-0.8, 0.9], [-1.0, 1.1]],
    )
    np.testing.assert_array_equal(radial_sign, [1.0, -1.0, 1.0, 1.0])
    np.testing.assert_array_equal(signed_again, original_vector)
    np.testing.assert_array_equal(position, original_position)
    np.testing.assert_array_equal(vector, original_vector)
    assert not bool(np.signbit(canonical_position[:, 0]).any())


def test_signed_rz_enclosure_maps_intervals_through_absolute_radius() -> None:
    lower = np.asarray([[1.0, -2.0], [-3.0, -1.0], [-4.0, 2.0]], dtype="<f8")
    upper = np.asarray([[2.0, 3.0], [-1.0, 4.0], [2.0, 5.0]], dtype="<f8")

    canonical_lower, canonical_upper = canonicalize_rz_enclosure(lower, upper)

    np.testing.assert_array_equal(canonical_lower, [[1.0, -2.0], [1.0, -1.0], [0.0, 2.0]])
    np.testing.assert_array_equal(canonical_upper, [[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    np.testing.assert_array_equal(lower, [[1.0, -2.0], [-3.0, -1.0], [-4.0, 2.0]])
    np.testing.assert_array_equal(upper, [[2.0, 3.0], [-1.0, 4.0], [2.0, 5.0]])


def test_batched_rz_transforms_reject_invalid_shapes_signs_and_bounds() -> None:
    with pytest.raises(ValueError, match="shape"):
        rz_signed_stage_to_canonical(np.zeros(2), np.zeros((1, 2)))
    with pytest.raises(ValueError, match=r"-1 or \+1"):
        rz_canonical_vector_to_signed(np.zeros((1, 2)), np.asarray([0.0]))
    with pytest.raises(ValueError, match="must not exceed"):
        canonicalize_rz_enclosure(
            np.asarray([[1.0, 0.0]]),
            np.asarray([[0.0, 1.0]]),
        )


@pytest.mark.parametrize(
    ("position", "vector", "expected_position", "expected_vector"),
    [
        ([0.4, 1.2], [-0.3, 0.5], [0.4, 1.2], [-0.3, 0.5]),
        ([-0.4, 1.2], [-0.3, 0.5], [0.4, 1.2], [0.3, 0.5]),
        ([0.0, 1.2], [-0.3, 0.5], [0.0, 1.2], [0.3, 0.5]),
    ],
)
def test_rz_axis_fold_is_a_basis_change_not_a_wall_event(
    position: list[float],
    vector: list[float],
    expected_position: list[float],
    expected_vector: list[float],
) -> None:
    original_position = np.asarray(position, dtype=np.float64)
    original_vector = np.asarray(vector, dtype=np.float64)

    folded_position, folded_vector = fold_rz_position_vector(original_position, original_vector)

    np.testing.assert_array_equal(folded_position, expected_position)
    np.testing.assert_array_equal(folded_vector, expected_vector)
    np.testing.assert_array_equal(original_position, position)
    np.testing.assert_array_equal(original_vector, vector)


def test_rz_axis_fold_rejects_nonfinite_or_wrong_dimension() -> None:
    with pytest.raises(ValueError):
        fold_rz_position_vector(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError):
        fold_rz_position_vector(np.asarray([0.0, 1.0]), np.asarray([np.nan, 1.0]))


@pytest.mark.parametrize("radius", [-0.4, -0.0])
def test_rz_axis_fold_normalizes_signed_zero_and_is_idempotent(radius: float) -> None:
    position, vector = fold_rz_position_vector(np.asarray([radius, 1.2]), np.asarray([-0.3, 0.5]))
    folded_again = fold_rz_position_vector(position, vector)

    assert not np.signbit(position[0])
    np.testing.assert_array_equal(folded_again[0], position)
    np.testing.assert_array_equal(folded_again[1], vector)
