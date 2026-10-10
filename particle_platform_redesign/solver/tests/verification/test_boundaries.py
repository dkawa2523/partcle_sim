from __future__ import annotations

import numpy as np
import pytest

from chamber_particles.boundaries import (
    BOUNDARY_LAW_ESCAPE,
    BOUNDARY_LAW_HOLD,
    BOUNDARY_LAW_MAXWELL_THERMAL,
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
    half_range_maxwell_flux_velocity,
    half_range_maxwell_flux_velocity_batch,
    prepare_boundary_rule,
    prepare_boundary_rules,
    resolve_boundary_responses_batch,
    validate_boundary_rule_frames,
)
from chamber_particles.physics.forces import BOLTZMANN_J_K
from chamber_particles.rng import (
    WALL_MAXWELL_NORMAL_STREAM,
    WALL_MAXWELL_TANGENTIAL_STREAM,
    wall_standard_normal_batch,
    wall_uniform_open_batch,
)


def _resolve_boundary_responses(
    rules: object,
    candidate_offsets: np.ndarray,
    candidate_facet_ids: np.ndarray,
    boundary_id: np.ndarray,
    group_id: np.ndarray,
    facet_normal: np.ndarray,
    velocity_pre_m_s: np.ndarray,
    law_uniform_draw: np.ndarray,
    *,
    roundoff_ulps: int,
) -> object:
    row_count = velocity_pre_m_s.shape[0]
    return resolve_boundary_responses_batch(
        rules,
        candidate_offsets,
        candidate_facet_ids,
        boundary_id,
        group_id,
        facet_normal[candidate_facet_ids],
        velocity_pre_m_s,
        np.ones(row_count, dtype="<f8"),
        law_uniform_draw,
        np.zeros(row_count, dtype="<f8"),
        np.full(row_count, 0.5, dtype="<f8"),
        np.zeros(row_count, dtype="<f8"),
        roundoff_ulps=roundoff_ulps,
    )


def test_corner_policy_rejects_incompatible_laws_at_the_same_priority() -> None:
    rules = (
        prepare_boundary_rule(0, 10, "specular", {}),
        prepare_boundary_rule(1, 10, "stick", {}),
    )

    response = _resolve_boundary_responses(
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

    response = _resolve_boundary_responses(
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


def test_specular_response_uses_candidate_contact_normal() -> None:
    rule = prepare_boundary_rule(0, 10, "specular", {})
    diagonal = 1.0 / np.sqrt(2.0)
    response = resolve_boundary_responses_batch(
        prepare_boundary_rules((rule,)),
        np.asarray([0, 1], dtype="<i8"),
        np.asarray([0], dtype="<i8"),
        np.asarray([10], dtype="<i4"),
        np.asarray([0], dtype="<i4"),
        np.asarray([[diagonal, diagonal]], dtype="<f8"),
        np.asarray([[1.0, 1.0]], dtype="<f8"),
        np.asarray([1.0], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([0.5], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        roundoff_ulps=64,
    )

    assert response.status[0] == BOUNDARY_STATUS_OK
    np.testing.assert_allclose(response.effective_normal, [[diagonal, diagonal]], atol=2.0e-16)
    np.testing.assert_allclose(response.velocity_post_m_s, [[-1.0, -1.0]], atol=5.0e-16)


def test_specular_rejects_coefficients_owned_by_restitution() -> None:
    coefficients = {"normal_restitution": 0.75, "tangential_restitution": 0.25}

    with pytest.raises(BoundaryLawError, match="specular boundary law does not accept parameters"):
        prepare_boundary_rule(0, 10, "specular", coefficients)
    with pytest.raises(BoundaryLawError, match="requires exactly"):
        prepare_boundary_rule(0, 10, "restitution", {"normal_restitution": 0.75})

    rule = prepare_boundary_rule(0, 10, "restitution", coefficients)
    response = _resolve_boundary_responses(
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


def test_half_range_maxwell_flux_uses_fixed_tangent_and_wall_frame() -> None:
    temperature_K = 300.0
    mass_kg = BOLTZMANN_J_K * temperature_K

    velocity = half_range_maxwell_flux_velocity(
        temperature_K,
        mass_kg,
        0.0,
        2.0,
        1.0,
        0.0,
        np.exp(-0.5),
        3.0,
    )

    np.testing.assert_allclose(velocity, [-1.0, 5.0], rtol=0.0, atol=3.0e-16)
    batch = half_range_maxwell_flux_velocity_batch(
        temperature_K,
        np.asarray([mass_kg, mass_kg], dtype="<f8"),
        np.asarray([0.0, 2.0], dtype="<f8"),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="<f8"),
        np.asarray([np.exp(-0.5), np.exp(-2.0)], dtype="<f8"),
        np.asarray([3.0, -1.0], dtype="<f8"),
    )
    np.testing.assert_allclose(batch, [[-1.0, 5.0], [1.0, 0.0]], rtol=0.0, atol=5.0e-16)

    with pytest.raises(ValueError, match="outside their finite domains"):
        half_range_maxwell_flux_velocity(
            temperature_K,
            mass_kg,
            0.0,
            0.0,
            2.0,
            0.0,
            0.5,
            0.0,
        )


def test_maxwell_thermal_switches_between_diffuse_and_wall_frame_specular() -> None:
    temperature_K = 300.0
    mass_kg = BOLTZMANN_J_K * temperature_K
    diffuse = prepare_boundary_rule(
        0,
        10,
        "maxwell_thermal",
        {
            "wall_temperature_K": temperature_K,
            "diffuse_reflection_fraction": 1.0,
            "wall_velocity_m_s": [0.0, 2.0],
        },
    )
    specular = prepare_boundary_rule(
        0,
        10,
        "maxwell_thermal",
        {
            "wall_temperature_K": temperature_K,
            "diffuse_reflection_fraction": 0.0,
            "wall_velocity_m_s": [0.0, 2.0],
        },
    )

    def resolve(rule: object) -> object:
        return resolve_boundary_responses_batch(
            prepare_boundary_rules((rule,)),
            np.asarray([0, 1], dtype="<i8"),
            np.asarray([0], dtype="<i8"),
            np.asarray([10], dtype="<i4"),
            np.asarray([0], dtype="<i4"),
            np.asarray([[1.0, 0.0]], dtype="<f8"),
            np.asarray([[2.0, 3.0]], dtype="<f8"),
            np.asarray([mass_kg], dtype="<f8"),
            np.asarray([0.0], dtype="<f8"),
            np.asarray([0.5], dtype="<f8"),
            np.asarray([np.exp(-2.0)], dtype="<f8"),
            np.asarray([-1.0], dtype="<f8"),
            roundoff_ulps=64,
        )

    diffuse_response = resolve(diffuse)
    specular_response = resolve(specular)
    assert diffuse_response.law[0] == BOUNDARY_LAW_MAXWELL_THERMAL
    np.testing.assert_allclose(
        diffuse_response.velocity_post_m_s[0],
        [-2.0, 1.0],
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_array_equal(specular_response.velocity_post_m_s[0], [-2.0, 3.0])


@pytest.mark.parametrize(
    "parameters, message",
    [
        (
            {
                "diffuse_reflection_fraction": 1.0,
                "wall_velocity_m_s": [0.0, 0.0],
            },
            "requires exactly",
        ),
        (
            {
                "wall_temperature_K": 0.0,
                "diffuse_reflection_fraction": 1.0,
                "wall_velocity_m_s": [0.0, 0.0],
            },
            "wall_temperature_K must be positive",
        ),
        (
            {
                "wall_temperature_K": 300.0,
                "diffuse_reflection_fraction": 1.1,
                "wall_velocity_m_s": [0.0, 0.0],
            },
            "must be in \\[0, 1\\]",
        ),
        (
            {
                "wall_temperature_K": 300.0,
                "diffuse_reflection_fraction": 1.0,
                "wall_velocity_m_s": [0.0],
            },
            "two finite numbers",
        ),
    ],
)
def test_maxwell_thermal_rejects_incomplete_or_out_of_domain_parameters(
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(BoundaryLawError, match=message):
        prepare_boundary_rule(0, 10, "maxwell_thermal", parameters)


def test_probabilistic_stick_can_fall_back_to_maxwell_thermal() -> None:
    temperature_K = 300.0
    mass_kg = BOLTZMANN_J_K * temperature_K
    rule = prepare_boundary_rule(
        0,
        10,
        "probabilistic_stick",
        {
            "probability": 0.25,
            "otherwise": {
                "law": "maxwell_thermal",
                "wall_temperature_K": temperature_K,
                "diffuse_reflection_fraction": 1.0,
                "wall_velocity_m_s": [0.0, 0.0],
            },
        },
    )

    response = resolve_boundary_responses_batch(
        prepare_boundary_rules((rule,)),
        np.asarray([0, 1], dtype="<i8"),
        np.asarray([0], dtype="<i8"),
        np.asarray([10], dtype="<i4"),
        np.asarray([0], dtype="<i4"),
        np.asarray([[1.0, 0.0]], dtype="<f8"),
        np.asarray([[2.0, 3.0]], dtype="<f8"),
        np.asarray([mass_kg], dtype="<f8"),
        np.asarray([0.9], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([np.exp(-0.5)], dtype="<f8"),
        np.asarray([2.0], dtype="<f8"),
        roundoff_ulps=64,
    )

    assert response.status[0] == BOUNDARY_STATUS_OK
    assert response.law[0] == BOUNDARY_LAW_PROBABILISTIC_STICK
    assert response.outcome[0] == BOUNDARY_OUTCOME_REFLECTED
    np.testing.assert_allclose(response.velocity_post_m_s[0], [-1.0, 2.0], atol=3.0e-16)


def test_maxwell_wall_velocity_must_be_tangent_to_static_group() -> None:
    rule = prepare_boundary_rule(
        0,
        10,
        "maxwell_thermal",
        {
            "wall_temperature_K": 300.0,
            "diffuse_reflection_fraction": 1.0,
            "wall_velocity_m_s": [1.0, 0.0],
        },
    )

    with pytest.raises(BoundaryLawError, match="must be tangent"):
        validate_boundary_rule_frames(
            prepare_boundary_rules((rule,)),
            np.asarray([0], dtype="<i4"),
            np.asarray([[1.0, 0.0]], dtype="<f8"),
            roundoff_ulps=64,
        )


def test_half_range_maxwell_flux_moments_match_two_dof_flux_distribution() -> None:
    count = 50_000
    temperature_K = 300.0
    mass_kg = BOLTZMANN_J_K * temperature_K
    particle_id = np.arange(count, dtype="<u8")
    ordinal = np.zeros(count, dtype="<u8")
    normal_draw = wall_uniform_open_batch(
        7,
        particle_id,
        ordinal,
        WALL_MAXWELL_NORMAL_STREAM,
    )
    tangent_draw = wall_standard_normal_batch(
        7,
        particle_id,
        ordinal,
        WALL_MAXWELL_TANGENTIAL_STREAM,
    )
    velocity = half_range_maxwell_flux_velocity_batch(
        temperature_K,
        np.full(count, mass_kg, dtype="<f8"),
        np.zeros(2, dtype="<f8"),
        np.tile(np.asarray([[1.0, 0.0]], dtype="<f8"), (count, 1)),
        normal_draw,
        tangent_draw,
    )

    inward = -velocity[:, 0]
    tangent = velocity[:, 1]
    assert np.mean(inward) == pytest.approx(np.sqrt(np.pi / 2.0), abs=0.015)
    assert np.mean(inward**2) == pytest.approx(2.0, abs=0.03)
    assert np.mean(tangent) == pytest.approx(0.0, abs=0.015)
    assert np.mean(tangent**2) == pytest.approx(1.0, abs=0.025)


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
    response = _resolve_boundary_responses(
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

    actual = _resolve_boundary_responses(
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
