from __future__ import annotations

import numpy as np
import pytest

from chamber_particles.boundaries import (
    BOUNDARY_LAW_ESCAPE,
    BOUNDARY_LAW_HOLD,
    BOUNDARY_LAW_PROBABILISTIC_STICK,
    BOUNDARY_LAW_RESTITUTION,
    BOUNDARY_LAW_STICK,
    BOUNDARY_OUTCOME_ESCAPED,
    BOUNDARY_OUTCOME_HELD,
    BOUNDARY_OUTCOME_REFLECTED,
    BOUNDARY_OUTCOME_STUCK,
    BOUNDARY_STATUS_AMBIGUOUS_LAW,
    BOUNDARY_STATUS_INDETERMINATE_POLICY,
    BOUNDARY_STATUS_OK,
    BoundaryLawError,
    prepare_boundary_rule,
    prepare_boundary_rules,
    resolve_boundary_responses_batch,
)


def test_corner_policy_rejects_incompatible_laws_at_the_same_priority() -> None:
    rules = (
        prepare_boundary_rule(0, 10, "specular", {}),
        prepare_boundary_rule(1, 10, "stick", {}),
    )

    response = resolve_boundary_responses_batch(
        prepare_boundary_rules(rules),
        np.asarray([0, 2], dtype="<i8"),
        np.asarray([0, 1], dtype="<i8"),
        np.asarray([20, 21], dtype="<i4"),
        np.asarray([0, 1], dtype="<i4"),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f8"),
        np.asarray([[1.0, 1.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        roundoff_ulps=64,
    )

    assert response.status[0] == BOUNDARY_STATUS_AMBIGUOUS_LAW


def test_priority_response_cannot_exit_through_an_ignored_candidate() -> None:
    rules = (
        prepare_boundary_rule(0, 5, "specular", {}),
        prepare_boundary_rule(1, 20, "specular", {}),
    )

    response = resolve_boundary_responses_batch(
        prepare_boundary_rules(rules),
        np.asarray([0, 2], dtype="<i8"),
        np.asarray([0, 1], dtype="<i8"),
        np.asarray([50, 60], dtype="<i4"),
        np.asarray([0, 1], dtype="<i4"),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f8"),
        np.asarray([[1.0, 1.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        roundoff_ulps=64,
    )

    assert response.status[0] == BOUNDARY_STATUS_INDETERMINATE_POLICY


def test_specular_rejects_coefficients_owned_by_restitution() -> None:
    coefficients = {"normal_restitution": 0.75, "tangential_restitution": 0.25}

    with pytest.raises(BoundaryLawError, match="specular boundary law does not accept parameters"):
        prepare_boundary_rule(0, 10, "specular", coefficients)
    with pytest.raises(BoundaryLawError, match="requires exactly"):
        prepare_boundary_rule(0, 10, "restitution", {"normal_restitution": 0.75})

    rule = prepare_boundary_rule(0, 10, "restitution", coefficients)
    response = resolve_boundary_responses_batch(
        prepare_boundary_rules((rule,)),
        np.asarray([0, 1], dtype="<i8"),
        np.asarray([0], dtype="<i8"),
        np.asarray([10], dtype="<i4"),
        np.asarray([0], dtype="<i4"),
        np.asarray([[1.0, 0.0]], dtype="<f8"),
        np.asarray([[2.0, 3.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        roundoff_ulps=64,
    )

    assert response.status[0] == BOUNDARY_STATUS_OK
    assert response.law[0] == BOUNDARY_LAW_RESTITUTION
    assert response.outcome[0] == BOUNDARY_OUTCOME_REFLECTED
    np.testing.assert_array_equal(response.velocity_post_m_s[0], [-1.5, 0.75])

    with pytest.raises(BoundaryLawError, match="hold boundary law does not accept parameters"):
        prepare_boundary_rule(0, 10, "hold", {"unused": 1.0})


def test_probabilistic_stick_can_fall_back_to_restitution() -> None:
    parameters = {
        "probability": 0.4,
        "otherwise": {
            "law": "restitution",
            "normal_restitution": 0.5,
            "tangential_restitution": 0.25,
        },
    }
    rule = prepare_boundary_rule(0, 10, "probabilistic_stick", parameters)
    response = resolve_boundary_responses_batch(
        prepare_boundary_rules((rule,)),
        np.asarray([0, 1], dtype="<i8"),
        np.asarray([0], dtype="<i8"),
        np.asarray([10], dtype="<i4"),
        np.asarray([0], dtype="<i4"),
        np.asarray([[1.0, 0.0]], dtype="<f8"),
        np.asarray([[2.0, 3.0]], dtype="<f8"),
        np.asarray([0.9], dtype="<f8"),
        roundoff_ulps=64,
    )

    assert response.status[0] == BOUNDARY_STATUS_OK
    assert response.law[0] == BOUNDARY_LAW_PROBABILISTIC_STICK
    assert response.outcome[0] == BOUNDARY_OUTCOME_REFLECTED
    np.testing.assert_array_equal(response.velocity_post_m_s[0], [-1.0, 0.75])


def test_compiled_boundary_rows_apply_laws_and_corner_order() -> None:
    restitution = {"normal_restitution": 0.75, "tangential_restitution": 0.25}
    probabilistic = {
        "probability": 0.4,
        "otherwise": {"law": "restitution", **restitution},
    }
    rules = (
        prepare_boundary_rule(0, 10, "stick", {}),
        prepare_boundary_rule(1, 10, "escape", {}),
        prepare_boundary_rule(2, 5, "restitution", restitution),
        prepare_boundary_rule(3, 5, "restitution", restitution),
        prepare_boundary_rule(4, 2, "probabilistic_stick", probabilistic),
        prepare_boundary_rule(5, 1, "hold", {}),
    )
    boundary_id = np.asarray([20, 10, 31, 30, 40, 50], dtype="<i4")
    group_id = np.arange(6, dtype="<i4")
    facet_normal = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype="<f8",
    )
    rows = ((0,), (1,), (2, 3), (4,), (4,), (5,), (0, 1))
    offsets = np.asarray([0, 1, 2, 4, 5, 6, 7, 9], dtype="<i8")
    candidates = np.asarray([facet for row in rows for facet in row], dtype="<i8")
    velocity = np.asarray(
        [
            [2.0, 3.0],
            [2.0, 3.0],
            [2.0, 3.0],
            [-2.0, 3.0],
            [-2.0, 3.0],
            [2.0, 3.0],
            [1.0, 1.0],
        ],
        dtype="<f8",
    )
    draw = np.asarray([0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0], dtype="<f8")

    actual = resolve_boundary_responses_batch(
        prepare_boundary_rules(rules),
        offsets,
        candidates,
        boundary_id,
        group_id,
        facet_normal,
        velocity,
        draw,
        roundoff_ulps=64,
    )

    np.testing.assert_array_equal(actual.status[:6], np.full(6, BOUNDARY_STATUS_OK))
    np.testing.assert_array_equal(
        actual.law[:6],
        [
            BOUNDARY_LAW_STICK,
            BOUNDARY_LAW_ESCAPE,
            BOUNDARY_LAW_RESTITUTION,
            BOUNDARY_LAW_PROBABILISTIC_STICK,
            BOUNDARY_LAW_PROBABILISTIC_STICK,
            BOUNDARY_LAW_HOLD,
        ],
    )
    np.testing.assert_array_equal(
        actual.outcome[:6],
        [
            BOUNDARY_OUTCOME_STUCK,
            BOUNDARY_OUTCOME_ESCAPED,
            BOUNDARY_OUTCOME_REFLECTED,
            BOUNDARY_OUTCOME_STUCK,
            BOUNDARY_OUTCOME_REFLECTED,
            BOUNDARY_OUTCOME_HELD,
        ],
    )
    np.testing.assert_array_equal(actual.primary_facet_id[:6], [0, 1, 3, 4, 4, 5])
    np.testing.assert_array_equal(
        actual.remains_active[:6],
        [False, False, True, False, True, False],
    )
    np.testing.assert_allclose(
        actual.velocity_post_m_s[:6],
        [[0.0, 0.0], [2.0, 3.0], [-0.6, -2.55], [0.0, 0.0], [1.5, 0.75], [2.0, 3.0]],
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        actual.effective_normal[:6],
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / np.sqrt(10.0), 3.0 / np.sqrt(10.0)],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )
    assert actual.status[6] == BOUNDARY_STATUS_AMBIGUOUS_LAW
