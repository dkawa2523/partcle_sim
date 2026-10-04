from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from chamber_particles.integrators import (
    RelaxationEvaluation,
    exponential_frozen_start_predictor,
)
from chamber_particles.rng import (
    BROWNIAN_ROOT_NORMAL_STREAM,
    BROWNIAN_SPLIT_NORMAL_STREAM,
    brownian_normal_pair_batch,
)
from chamber_particles.stochastic import (
    JOINT_OU_REVISION,
    JOINT_OU_SPLIT_REVISION,
    MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT,
    JointOuIncrement,
    advance_joint_ou,
    advance_joint_ou_with_increment,
    compose_joint_ou_increments,
    joint_ou_covariance,
    joint_ou_increment,
    split_joint_ou_increment_half,
)


def test_stochastic_exponential_midpoint_has_second_order_noise_free_limit() -> None:
    rate = 2.0
    velocity_gradient = -0.4
    additive = 0.3
    initial = np.asarray([1.1, -0.2])
    end_s = 1.0

    matrix = np.asarray([[0.0, 1.0], [rate * velocity_gradient, -rate]])
    forcing = np.asarray([0.0, additive])
    steady = -np.linalg.solve(matrix, forcing)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    exponential = eigenvectors @ np.diag(np.exp(eigenvalues * end_s)) @ np.linalg.inv(eigenvectors)
    exact = steady + np.real_if_close(exponential @ (initial - steady)).astype(np.float64)

    def relaxation(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, velocity_m_s, charge_number
        count = particle_index.size
        target = np.zeros((count, 2), dtype="<f8")
        target[:, 0] = velocity_gradient * position_m[:, 0]
        acceleration = np.zeros((count, 2), dtype="<f8")
        acceleration[:, 0] = additive
        return RelaxationEvaluation(
            linear_drag_rate_s_inv=np.full(count, rate),
            target_velocity_m_s=target,
            additive_acceleration_m_s2=acceleration,
            charge_rate_number_s=np.zeros(count),
            charge_rate_derivative_s_inv=np.zeros(count),
            support_inside=np.ones(count, dtype=np.bool_),
            applicability_inside=np.ones(count, dtype=np.bool_),
            numerical_status=np.zeros(count, dtype=np.uint8),
        )

    errors = []
    for step_count in (8, 16, 32, 64):
        step_s = end_s / step_count
        position = np.asarray([[initial[0], 0.0]])
        velocity = np.asarray([[initial[1], 0.0]])
        charge = np.zeros(1)
        for step in range(step_count):
            start_s = np.asarray([step * step_s])
            midpoint_s = start_s + 0.5 * step_s
            predictor = exponential_frozen_start_predictor(
                np.asarray([1], dtype="<i8"),
                start_s,
                midpoint_s,
                position,
                velocity,
                charge,
                evaluator=relaxation,
            )
            midpoint = relaxation(
                np.asarray([1], dtype="<i8"),
                midpoint_s,
                predictor.position_m,
                predictor.velocity_m_s,
                predictor.charge_number,
            )
            equilibrium = (
                midpoint.target_velocity_m_s
                + midpoint.additive_acceleration_m_s2 / midpoint.linear_drag_rate_s_inv[:, None]
            )
            position, velocity = advance_joint_ou_with_increment(
                position,
                velocity,
                equilibrium,
                midpoint.linear_drag_rate_s_inv,
                np.asarray([step_s]),
                JointOuIncrement(np.zeros((1, 2)), np.zeros((1, 2))),
            )
        difference = [position[0, 0] - exact[0], velocity[0, 0] - exact[1]]
        errors.append(float(np.linalg.norm(difference)))

    observed_orders = [math.log2(coarse / fine) for coarse, fine in pairwise(errors)]
    assert min(observed_orders) > 1.8


def test_charge_electric_stochastic_midpoint_weak_mean_is_at_least_first_order() -> None:
    rate = 1.7
    electric_coupling = 0.8
    charge_rate = -0.35
    initial_position = 0.2
    initial_velocity = -0.1
    initial_charge = 1.4
    end_s = 0.9
    decay = math.exp(-rate * end_s)
    exact_charge = initial_charge + charge_rate * end_s
    exact_velocity = initial_velocity * decay + electric_coupling * (
        initial_charge * (1.0 - decay) / rate
        + charge_rate * (end_s / rate - (1.0 - decay) / rate**2)
    )
    exact_position = (
        initial_position
        + initial_velocity * (1.0 - decay) / rate
        + electric_coupling
        * (
            initial_charge * (end_s / rate - (1.0 - decay) / rate**2)
            + charge_rate * (end_s**2 / (2.0 * rate) - end_s / rate**2 + (1.0 - decay) / rate**3)
        )
    )

    def relaxation(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, position_m, velocity_m_s
        count = particle_index.size
        acceleration = np.zeros((count, 2), dtype="<f8")
        acceleration[:, 0] = electric_coupling * charge_number
        return RelaxationEvaluation(
            linear_drag_rate_s_inv=np.full(count, rate),
            target_velocity_m_s=np.zeros((count, 2)),
            additive_acceleration_m_s2=acceleration,
            charge_rate_number_s=np.full(count, charge_rate),
            charge_rate_derivative_s_inv=np.zeros(count),
            support_inside=np.ones(count, dtype=np.bool_),
            applicability_inside=np.ones(count, dtype=np.bool_),
            numerical_status=np.zeros(count, dtype=np.uint8),
        )

    errors = []
    for step_count in (4, 8, 16, 32):
        step_s = end_s / step_count
        position = np.asarray([[initial_position, 0.0]])
        velocity = np.asarray([[initial_velocity, 0.0]])
        charge = np.asarray([initial_charge])
        for step in range(step_count):
            start_s = np.asarray([step * step_s])
            midpoint_s = start_s + 0.5 * step_s
            predictor = exponential_frozen_start_predictor(
                np.asarray([1], dtype="<i8"),
                start_s,
                midpoint_s,
                position,
                velocity,
                charge,
                evaluator=relaxation,
            )
            midpoint = relaxation(
                np.asarray([1], dtype="<i8"),
                midpoint_s,
                predictor.position_m,
                predictor.velocity_m_s,
                predictor.charge_number,
            )
            equilibrium = (
                midpoint.additive_acceleration_m_s2 / midpoint.linear_drag_rate_s_inv[:, None]
            )
            position, velocity = advance_joint_ou_with_increment(
                position,
                velocity,
                equilibrium,
                midpoint.linear_drag_rate_s_inv,
                np.asarray([step_s]),
                JointOuIncrement(np.zeros((1, 2)), np.zeros((1, 2))),
            )
            charge = charge + step_s * midpoint.charge_rate_number_s
        difference = [
            position[0, 0] - exact_position,
            velocity[0, 0] - exact_velocity,
            charge[0] - exact_charge,
        ]
        errors.append(float(np.linalg.norm(difference)))

    observed_orders = [math.log2(coarse / fine) for coarse, fine in pairwise(errors)]
    assert min(observed_orders) >= 0.9


def test_joint_ou_closed_form_and_short_long_time_limits() -> None:
    rate = np.asarray([2.3, 2.3, 2.3, 2.3])
    thermal = np.asarray([0.7, 0.7, 0.7, 0.7])
    argument = np.asarray([1.0e-8, 0.4, 10.0, 1.0e6])
    duration = argument / rate
    variance_x, covariance_xv, variance_v = joint_ou_covariance(
        rate,
        thermal,
        duration,
    )

    q = -math.expm1(-argument[1])
    np.testing.assert_allclose(
        [variance_x[1], covariance_xv[1], variance_v[1]],
        [
            thermal[1] / rate[1] ** 2 * (2.0 * argument[1] - 2.0 * q - q * q),
            thermal[1] / rate[1] * q * q,
            thermal[1] * -math.expm1(-2.0 * argument[1]),
        ],
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        [
            variance_x[0] / ((2.0 / 3.0) * thermal[0] * rate[0] * duration[0] ** 3),
            covariance_xv[0] / (thermal[0] * rate[0] * duration[0] ** 2),
            variance_v[0] / (2.0 * thermal[0] * rate[0] * duration[0]),
        ],
        np.ones(3),
        rtol=2.0e-8,
        atol=0.0,
    )
    np.testing.assert_allclose(variance_v[-1], thermal[-1], rtol=2.0e-6)
    np.testing.assert_allclose(covariance_xv[-1], thermal[-1] / rate[-1], rtol=2.0e-6)
    np.testing.assert_allclose(
        variance_x[-1],
        2.0 * thermal[-1] * duration[-1] / rate[-1],
        rtol=2.0e-6,
    )

    zeros = np.zeros((1, 2, 2))
    position, velocity = advance_joint_ou(
        np.asarray([[0.2, -0.4]]),
        np.asarray([[0.8, 0.1]]),
        np.asarray([[0.3, -0.2]]),
        np.asarray([2.3]),
        np.asarray([0.0]),
        np.asarray([0.4 / 2.3]),
        zeros,
    )
    decay = math.exp(-0.4)
    displacement = -math.expm1(-0.4) / 2.3
    np.testing.assert_allclose(
        velocity[0],
        np.asarray([0.3, -0.2]) + decay * np.asarray([0.5, 0.3]),
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        position[0],
        np.asarray([0.2, -0.4])
        + np.asarray([0.3, -0.2]) * (0.4 / 2.3)
        + np.asarray([0.5, 0.3]) * displacement,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert JOINT_OU_REVISION == "inertial_joint_ou_v1"
    assert MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT == 1.0e6


def test_joint_ou_supplied_increment_matches_the_direct_update() -> None:
    position = np.asarray([[0.2, -0.4], [1.0, 0.5]])
    velocity = np.asarray([[0.8, 0.1], [-0.2, 0.7]])
    equilibrium = np.asarray([[0.3, -0.2], [0.1, 0.4]])
    rate = np.asarray([2.3, 0.7])
    thermal = np.asarray([0.6, 1.4])
    duration = np.asarray([0.4 / 2.3, 0.8])
    normals = np.asarray(
        [
            [[0.2, -0.3], [1.1, 0.4]],
            [[-0.7, 0.9], [0.5, -1.2]],
        ]
    )
    increment = joint_ou_increment(rate, thermal, duration, normals)

    expected = advance_joint_ou(
        position,
        velocity,
        equilibrium,
        rate,
        thermal,
        duration,
        normals,
    )
    actual = advance_joint_ou_with_increment(
        position,
        velocity,
        equilibrium,
        rate,
        duration,
        increment,
    )

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_joint_ou_declared_relaxation_limit_remains_representable() -> None:
    rate = np.asarray([1.0])
    thermal = np.asarray([0.7])
    duration = np.asarray([MAXIMUM_JOINT_OU_RELAXATION_ARGUMENT])
    normals = np.asarray([[[0.2, -0.3], [1.1, 0.4]]])
    parent = joint_ou_increment(rate, thermal, duration, normals)
    left, right = split_joint_ou_increment_half(
        parent,
        rate,
        thermal,
        duration,
        -normals,
    )
    recomposed = compose_joint_ou_increments(left, right, rate, 0.5 * duration)

    np.testing.assert_allclose(recomposed.position_m, parent.position_m, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(recomposed.velocity_m_s, parent.velocity_m_s, rtol=0.0, atol=1.0e-15)


def test_joint_ou_covariance_and_half_conditioning_match_independent_oracles() -> None:
    for argument in (
        1.0e-12,
        0.999e-3,
        1.0e-3,
        1.001e-3,
        0.0499,
        0.05,
        0.0501,
        0.2,
        10.0,
        1.0e3,
    ):
        rate = np.asarray([1.0])
        thermal = np.asarray([0.7])
        duration = np.asarray([argument])
        actual = joint_ou_covariance(rate, thermal, duration)
        expected = _quadrature_covariance(argument, thermal[0], rate[0])
        np.testing.assert_allclose(
            np.asarray([item[0] for item in actual]),
            expected,
            rtol=3.0e-12,
            atol=np.finfo(np.float64).tiny,
        )

        half_covariance = _covariance_matrix(
            _quadrature_covariance(0.5 * argument, thermal[0], rate[0])
        )
        parent_covariance = _covariance_matrix(expected)
        half_decay = math.exp(-0.5 * argument)
        half_displacement = -math.expm1(-0.5 * argument)
        transition = np.asarray([[1.0, half_displacement], [0.0, half_decay]])
        cross = half_covariance @ transition.T
        inverse_scale = np.diag(np.asarray([1.0 / argument, 1.0]))
        parent_scaled = inverse_scale @ parent_covariance @ inverse_scale
        half_scaled = inverse_scale @ half_covariance @ inverse_scale
        cross_scaled = inverse_scale @ cross @ inverse_scale
        gain_scaled = np.linalg.solve(parent_scaled, cross_scaled.T).T
        residual_scaled = half_scaled - gain_scaled @ cross_scaled.T
        residual_scaled = 0.5 * (residual_scaled + residual_scaled.T)

        zero_normals = np.zeros((1, 2, 2))
        for column, parent_vector in enumerate(
            (np.asarray([argument, 0.0]), np.asarray([0.0, 1.0]))
        ):
            left, _ = split_joint_ou_increment_half(
                _one_component_increment(parent_vector),
                rate,
                thermal,
                duration,
                zero_normals,
            )
            production_vector = inverse_scale @ np.asarray(
                [left.position_m[0, 0], left.velocity_m_s[0, 0]]
            )
            oracle_vector = gain_scaled[:, column]
            np.testing.assert_allclose(
                production_vector,
                oracle_vector,
                rtol=2.0e-10,
                atol=2.0e-13,
            )
        production_cholesky = np.empty((2, 2), dtype=np.float64)
        for column in range(2):
            normals = np.zeros((1, 2, 2))
            normals[0, 0, column] = 1.0
            left, _ = split_joint_ou_increment_half(
                _one_component_increment(np.zeros(2)),
                rate,
                thermal,
                duration,
                normals,
            )
            production_cholesky[:, column] = (
                left.position_m[0, 0],
                left.velocity_m_s[0, 0],
            )
        production_cholesky_scaled = inverse_scale @ production_cholesky
        production_residual_scaled = production_cholesky_scaled @ production_cholesky_scaled.T
        residual_scale = max(
            float(np.max(np.abs(residual_scaled))),
            np.finfo(np.float64).tiny,
        )
        np.testing.assert_allclose(
            production_residual_scaled,
            residual_scaled,
            rtol=3.0e-9,
            atol=5.0e-12 * residual_scale,
        )
        assert abs(float(production_residual_scaled[0, 1])) <= 1.0e-15 * residual_scale


def test_joint_ou_ensemble_matches_mean_and_full_covariance() -> None:
    count = 200_000
    particle_id = np.arange(count, dtype=np.uint64)
    normals = _normal_tensor(
        particle_id,
        seed=741,
        macro_interval=9,
        draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
    )
    rate = np.full(count, 2.3)
    thermal = np.full(count, 0.7)
    duration = np.full(count, 0.4)
    position0 = np.broadcast_to(np.asarray([0.2, -0.4]), (count, 2))
    velocity0 = np.broadcast_to(np.asarray([0.8, 0.1]), (count, 2))
    equilibrium = np.broadcast_to(np.asarray([0.3, -0.2]), (count, 2))

    position, velocity = advance_joint_ou(
        position0,
        velocity0,
        equilibrium,
        rate,
        thermal,
        duration,
        normals,
    )
    argument = rate[0] * duration[0]
    decay = math.exp(-argument)
    displacement = -math.expm1(-argument) / rate[0]
    expected_position = (
        position0[0] + equilibrium[0] * duration[0] + (velocity0[0] - equilibrium[0]) * displacement
    )
    expected_velocity = equilibrium[0] + decay * (velocity0[0] - equilibrium[0])
    variance_x, covariance_xv, variance_v = joint_ou_covariance(
        rate[:1],
        thermal[:1],
        duration[:1],
    )
    expected_covariance = np.asarray(
        [[variance_x[0], covariance_xv[0]], [covariance_xv[0], variance_v[0]]]
    )

    for component in range(2):
        sample = np.column_stack((position[:, component], velocity[:, component]))
        expected_mean = np.asarray([expected_position[component], expected_velocity[component]])
        standard_error = np.sqrt(np.diag(expected_covariance) / count)
        assert np.all(np.abs(sample.mean(axis=0) - expected_mean) <= 6.0 * standard_error)
        np.testing.assert_allclose(
            np.cov(sample, rowvar=False, bias=True),
            expected_covariance,
            rtol=1.5e-2,
            atol=2.0e-4,
        )
    centered_x = position[:, 0] - expected_position[0]
    centered_y = position[:, 1] - expected_position[1]
    assert abs(float(np.corrcoef(centered_x, centered_y)[0, 1])) < 1.5e-2


def test_conditional_half_split_preserves_parent_and_recovers_independent_children() -> None:
    count = 160_000
    particle_id = np.arange(count, dtype=np.uint64)
    root_normals = _normal_tensor(
        particle_id,
        seed=91,
        macro_interval=12,
        draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
    )
    split_normals = _normal_tensor(
        particle_id,
        seed=91,
        macro_interval=12,
        draw_kind=BROWNIAN_SPLIT_NORMAL_STREAM,
        tree_level=0,
        tree_index=0,
    )
    rate = np.full(count, 2.3)
    thermal = np.full(count, 0.7)
    duration = np.full(count, 0.4)
    parent = joint_ou_increment(rate, thermal, duration, root_normals)

    left, right = split_joint_ou_increment_half(
        parent,
        rate,
        thermal,
        duration,
        split_normals,
    )
    recomposed = compose_joint_ou_increments(left, right, rate, 0.5 * duration)
    np.testing.assert_allclose(recomposed.position_m, parent.position_m, rtol=0.0, atol=3.0e-16)
    np.testing.assert_allclose(recomposed.velocity_m_s, parent.velocity_m_s, rtol=0.0, atol=5.0e-16)

    half_x, half_xv, half_v = joint_ou_covariance(rate[:1], thermal[:1], duration[:1] / 2.0)
    expected = np.asarray([[half_x[0], half_xv[0]], [half_xv[0], half_v[0]]])
    left_sample = np.column_stack((left.position_m[:, 0], left.velocity_m_s[:, 0]))
    right_sample = np.column_stack((right.position_m[:, 0], right.velocity_m_s[:, 0]))
    np.testing.assert_allclose(
        np.cov(left_sample, rowvar=False, bias=True),
        expected,
        rtol=1.8e-2,
        atol=2.0e-4,
    )
    np.testing.assert_allclose(
        np.cov(right_sample, rowvar=False, bias=True),
        expected,
        rtol=1.8e-2,
        atol=2.0e-4,
    )
    cross = np.cov(
        np.column_stack((left_sample, right_sample)),
        rowvar=False,
        bias=True,
    )[:2, 2:]
    assert float(np.max(np.abs(cross))) < 2.5e-3
    assert JOINT_OU_SPLIT_REVISION == "conditional_gaussian_half_split_v1"


def test_two_level_interval_tree_reconstructs_the_same_root_increment() -> None:
    count = 4096
    particle_id = np.arange(count, dtype=np.uint64)
    rate = np.full(count, 1.7)
    thermal = np.full(count, 0.4)
    duration = np.full(count, 0.3)
    root = joint_ou_increment(
        rate,
        thermal,
        duration,
        _normal_tensor(
            particle_id,
            seed=19,
            macro_interval=4,
            root_interval=2,
            draw_kind=BROWNIAN_ROOT_NORMAL_STREAM,
        ),
    )
    first_left, first_right = split_joint_ou_increment_half(
        root,
        rate,
        thermal,
        duration,
        _normal_tensor(
            particle_id,
            seed=19,
            macro_interval=4,
            root_interval=2,
            draw_kind=BROWNIAN_SPLIT_NORMAL_STREAM,
        ),
    )
    left_left, left_right = split_joint_ou_increment_half(
        first_left,
        rate,
        thermal,
        duration / 2.0,
        _normal_tensor(
            particle_id,
            seed=19,
            macro_interval=4,
            root_interval=2,
            tree_level=1,
            tree_index=0,
            draw_kind=BROWNIAN_SPLIT_NORMAL_STREAM,
        ),
    )
    right_left, right_right = split_joint_ou_increment_half(
        first_right,
        rate,
        thermal,
        duration / 2.0,
        _normal_tensor(
            particle_id,
            seed=19,
            macro_interval=4,
            root_interval=2,
            tree_level=1,
            tree_index=1,
            draw_kind=BROWNIAN_SPLIT_NORMAL_STREAM,
        ),
    )
    reconstructed_left = compose_joint_ou_increments(
        left_left,
        left_right,
        rate,
        duration / 4.0,
    )
    reconstructed_right = compose_joint_ou_increments(
        right_left,
        right_right,
        rate,
        duration / 4.0,
    )
    reconstructed_root = compose_joint_ou_increments(
        reconstructed_left,
        reconstructed_right,
        rate,
        duration / 2.0,
    )
    np.testing.assert_allclose(
        reconstructed_root.position_m,
        root.position_m,
        rtol=0.0,
        atol=4.0e-16,
    )
    np.testing.assert_allclose(
        reconstructed_root.velocity_m_s,
        root.velocity_m_s,
        rtol=0.0,
        atol=6.0e-16,
    )


def _normal_tensor(
    particle_id: np.ndarray,
    *,
    seed: int,
    macro_interval: int,
    root_interval: int = 0,
    draw_kind: int,
    tree_level: int = 0,
    tree_index: int = 0,
) -> np.ndarray:
    result = np.empty((particle_id.size, 2, 2), dtype=np.float64)
    for component in range(2):
        result[:, component] = brownian_normal_pair_batch(
            seed,
            particle_id,
            macro_interval,
            root_interval,
            tree_level=tree_level,
            tree_index=tree_index,
            component=component,
            draw_kind=draw_kind,
        )
    return result


def _quadrature_covariance(
    argument: float,
    thermal_velocity_variance: float,
    rate: float,
) -> np.ndarray:
    """Integrate the OU Green-function kernel without production formulas."""

    nodes, weights = np.polynomial.legendre.leggauss(16)
    segment_count = max(1, math.ceil(argument / 0.25))
    edges = np.linspace(0.0, argument, segment_count + 1)
    integrals = np.zeros(3, dtype=np.float64)
    for lower, upper in pairwise(edges):
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        sample = midpoint + half_width * nodes
        decay = np.exp(-sample)
        displacement = -np.expm1(-sample)
        integrals += half_width * np.asarray(
            [
                np.dot(weights, 2.0 * displacement * displacement),
                np.dot(weights, 2.0 * displacement * decay),
                np.dot(weights, 2.0 * decay * decay),
            ]
        )
    return np.asarray(
        [
            thermal_velocity_variance * integrals[0] / rate**2,
            thermal_velocity_variance * integrals[1] / rate,
            thermal_velocity_variance * integrals[2],
        ]
    )


def _covariance_matrix(components: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[components[0], components[1]], [components[1], components[2]]],
        dtype=np.float64,
    )


def _one_component_increment(value: np.ndarray) -> JointOuIncrement:
    position = np.zeros((1, 2), dtype=np.float64)
    velocity = np.zeros((1, 2), dtype=np.float64)
    position[0, 0] = value[0]
    velocity[0, 0] = value[1]
    return JointOuIncrement(position, velocity)
