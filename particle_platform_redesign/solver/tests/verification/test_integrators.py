from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal, localcontext
from itertools import pairwise

import numpy as np
import pytest

from chamber_particles.integrators import (
    RK4_DENSE_PATH_REVISION,
    STEP_PROPOSAL_REVISION,
    CurvedPathEnclosure,
    DynamicsEvaluation,
    RelaxationEvaluation,
    cubic_hermite_step,
    curved_chord_deviation_bound,
    curved_chord_deviation_bounds,
    enclose_exponential_midpoint_path,
    enclose_rk4_path,
    exponential_frozen_start_predictor,
    exponential_midpoint_step,
    restrict_cubic_hermite_proposal,
    rk4_step,
)
from chamber_particles.numerical_status import (
    INTEGRATOR_NUMERICAL_FAILURE,
    NUMERICAL_STATUS_OK,
    PHYSICS_NUMERICAL_FAILURE,
)


def _valid_bound(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return value, np.zeros(value.shape[0], dtype=np.uint8)


def _table_acceleration_bounder(
    particle_index: np.ndarray,
    acceleration_abs_upper_m_s2: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def bounder(
        selected_particle: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del velocity_abs_upper_m_s
        row = np.searchsorted(particle_index, selected_particle)
        return _valid_bound(acceleration_abs_upper_m_s2[row])

    return bounder


def test_frozen_start_predictor_does_not_sample_its_predicted_endpoint() -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def relaxation(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del velocity_m_s, charge_number
        calls.append((time_s.copy(), position_m.copy()))
        count = particle_index.size
        return RelaxationEvaluation(
            linear_drag_rate_s_inv=np.full(count, 2.0),
            target_velocity_m_s=np.zeros((count, 2)),
            additive_acceleration_m_s2=np.broadcast_to([0.0, -0.3], (count, 2)).copy(),
            charge_rate_number_s=np.full(count, -4.0),
            charge_rate_derivative_s_inv=np.zeros(count),
            support_inside=np.ones(count, dtype=np.bool_),
            applicability_inside=np.ones(count, dtype=np.bool_),
            numerical_status=np.zeros(count, dtype=np.uint8),
        )

    predictor = exponential_frozen_start_predictor(
        np.asarray([7], dtype="<i8"),
        np.asarray([1.0]),
        np.asarray([1.2]),
        np.asarray([[0.05, 0.1]]),
        np.asarray([[-1.0, 0.2]]),
        np.asarray([3.0]),
        evaluator=relaxation,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [1.0])
    np.testing.assert_array_equal(calls[0][1], [[0.05, 0.1]])
    np.testing.assert_allclose(predictor.charge_number, [2.2], rtol=0.0, atol=1.0e-15)
    assert predictor.position_m[0, 0] < 0.0


def test_cubic_hermite_preserves_endpoint_state_and_derivative() -> None:
    start_position = np.asarray([[0.2, -0.4], [1.1, 0.3]], dtype="<f8")
    start_velocity = np.asarray([[0.7, -0.2], [-0.5, 0.9]], dtype="<f8")
    end_position = np.asarray([[1.4, 0.8], [-0.2, 1.7]], dtype="<f8")
    end_velocity = np.asarray([[-0.3, 0.6], [0.4, -0.7]], dtype="<f8")
    proposal = cubic_hermite_step(
        np.asarray([11, 29], dtype="<i8"),
        np.asarray([1.0, 1.0], dtype="<f8"),
        np.asarray([3.0, 3.0], dtype="<f8"),
        start_position,
        start_velocity,
        np.asarray([-2.0, 4.0], dtype="<f8"),
        end_position,
        end_velocity,
        end_charge_number=np.asarray([6.0, -8.0], dtype="<f8"),
        support_inside=np.asarray([True, False], dtype=np.bool_),
        applicability_inside=np.asarray([True, True], dtype=np.bool_),
        numerical_status=np.zeros(2, dtype=np.uint8),
    )

    rows = np.asarray([0, 1], dtype="<i8")
    start = proposal.state_at_rows(1.0, rows)
    end = proposal.state_at_rows(3.0, rows)

    midpoint = proposal.state_at_rows(2.0, rows)

    assert STEP_PROPOSAL_REVISION == "coupled_fixed_step_proposal_v10"
    assert proposal.path_kind == "cubic_hermite"
    np.testing.assert_array_equal(start.position_m, start_position)
    np.testing.assert_array_equal(start.velocity_m_s, start_velocity)
    np.testing.assert_array_equal(end.position_m, end_position)
    np.testing.assert_array_equal(end.velocity_m_s, end_velocity)
    np.testing.assert_array_equal(start.charge_number, np.asarray([-2.0, 4.0]))
    np.testing.assert_array_equal(midpoint.charge_number, np.asarray([2.0, -2.0]))
    np.testing.assert_array_equal(end.charge_number, np.asarray([6.0, -8.0]))
    np.testing.assert_array_equal(end.support_inside, np.asarray([True, False]))


def test_cubic_hermite_bezier_enclosure_contains_position_and_velocity() -> None:
    proposal = cubic_hermite_step(
        np.asarray([7], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([2.0], dtype="<f8"),
        np.asarray([[0.0, 1.0]], dtype="<f8"),
        np.asarray([[8.0, -6.0]], dtype="<f8"),
        np.asarray([3.0], dtype="<f8"),
        np.asarray([[1.0, -0.5]], dtype="<f8"),
        np.asarray([[-7.0, 5.0]], dtype="<f8"),
        support_inside=np.ones(1, dtype=np.bool_),
        applicability_inside=np.ones(1, dtype=np.bool_),
        numerical_status=np.zeros(1, dtype=np.uint8),
    )
    enclosure = proposal.path_enclosure
    assert enclosure is not None

    row = np.asarray([0], dtype="<i8")
    for time_s in np.linspace(0.0, 2.0, num=257):
        sample = proposal.state_at_rows(float(time_s), row)
        assert bool((sample.position_m >= enclosure.position_lower_m).all())
        assert bool((sample.position_m <= enclosure.position_upper_m).all())
        assert bool((sample.velocity_m_s >= enclosure.velocity_lower_m_s).all())
        assert bool((sample.velocity_m_s <= enclosure.velocity_upper_m_s).all())


def test_cubic_hermite_restriction_keeps_original_polynomial_identity() -> None:
    root = cubic_hermite_step(
        np.asarray([10, 20, 30], dtype="<i8"),
        np.zeros(3, dtype="<f8"),
        np.ones(3, dtype="<f8"),
        np.asarray([[0.0, 0.2], [1.0, -0.3], [-0.5, 0.7]], dtype="<f8"),
        np.asarray([[0.4, -0.1], [0.8, 0.5], [-0.6, 0.9]], dtype="<f8"),
        np.asarray([1.0, 2.0, 3.0], dtype="<f8"),
        np.asarray([[0.7, -0.2], [1.5, 0.4], [0.2, -0.6]], dtype="<f8"),
        np.asarray([[-0.2, 0.6], [0.3, -0.4], [0.5, 0.1]], dtype="<f8"),
        end_charge_number=np.asarray([5.0, 8.0, 11.0], dtype="<f8"),
        support_inside=np.asarray([True, False, True], dtype=np.bool_),
        applicability_inside=np.asarray([True, True, False], dtype=np.bool_),
        numerical_status=np.asarray(
            [NUMERICAL_STATUS_OK, PHYSICS_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK],
            dtype=np.uint8,
        ),
    )
    first = restrict_cubic_hermite_proposal(
        root,
        np.asarray([2, 0], dtype="<i8"),
        np.asarray([0.2, 0.1], dtype="<f8"),
        np.asarray([0.9, 0.8], dtype="<f8"),
    )
    repeated = restrict_cubic_hermite_proposal(
        first,
        np.asarray([1, 0], dtype="<i8"),
        np.asarray([0.3, 0.4], dtype="<f8"),
        np.asarray([0.7, 0.6], dtype="<f8"),
    )
    direct = restrict_cubic_hermite_proposal(
        root,
        np.asarray([0, 2], dtype="<i8"),
        np.asarray([0.3, 0.4], dtype="<f8"),
        np.asarray([0.7, 0.6], dtype="<f8"),
    )

    for name in (
        "particle_index",
        "start_position_m",
        "start_velocity_m_s",
        "start_charge_number",
        "end_position_m",
        "end_velocity_m_s",
        "end_charge_number",
        "support_inside",
        "applicability_inside",
        "numerical_status",
    ):
        np.testing.assert_array_equal(getattr(repeated, name), getattr(direct, name))
    assert repeated.path_enclosure is not None
    assert direct.path_enclosure is not None
    for name in (
        "position_lower_m",
        "position_upper_m",
        "velocity_lower_m_s",
        "velocity_upper_m_s",
        "numerical_status",
    ):
        np.testing.assert_array_equal(
            getattr(repeated.path_enclosure, name),
            getattr(direct.path_enclosure, name),
        )
    rows = np.asarray([0, 1], dtype="<i8")
    repeated_midpoint = repeated.state_at_rows(0.5, rows)
    direct_midpoint = direct.state_at_rows(0.5, rows)
    np.testing.assert_array_equal(repeated_midpoint.position_m, direct_midpoint.position_m)
    np.testing.assert_array_equal(repeated_midpoint.velocity_m_s, direct_midpoint.velocity_m_s)
    np.testing.assert_array_equal(repeated_midpoint.charge_number, direct_midpoint.charge_number)
    np.testing.assert_allclose(repeated.start_charge_number, [2.2, 6.2], rtol=0.0, atol=5.0e-16)
    np.testing.assert_allclose(repeated.end_charge_number, [3.8, 7.8], rtol=0.0, atol=5.0e-16)


def test_cubic_hermite_dense_path_is_stable_under_large_translation() -> None:
    shift = float(2**20)
    translation = np.asarray([[0.0, 0.0], [shift, -shift]], dtype="<f8")
    start = np.asarray([0.125, -0.375], dtype="<f8") + translation
    end = np.asarray([0.875, 0.25], dtype="<f8") + translation
    proposal = cubic_hermite_step(
        np.asarray([3, 4], dtype="<i8"),
        np.zeros(2, dtype="<f8"),
        np.ones(2, dtype="<f8"),
        start,
        np.broadcast_to([0.6, -0.2], (2, 2)).copy(),
        np.zeros(2, dtype="<f8"),
        end,
        np.broadcast_to([-0.3, 0.5], (2, 2)).copy(),
        support_inside=np.ones(2, dtype=np.bool_),
        applicability_inside=np.ones(2, dtype=np.bool_),
        numerical_status=np.zeros(2, dtype=np.uint8),
    )
    rows = np.asarray([0, 1], dtype="<i8")
    coordinate_ulp = np.spacing(shift)

    for time_s in (0.125, 0.5, 0.875):
        sample = proposal.state_at_rows(time_s, rows)
        np.testing.assert_allclose(
            sample.position_m[1] - translation[1],
            sample.position_m[0],
            rtol=0.0,
            atol=2.0 * coordinate_ulp,
        )
        np.testing.assert_allclose(
            sample.velocity_m_s[1],
            sample.velocity_m_s[0],
            rtol=0.0,
            atol=4.0 * np.finfo(np.float64).eps,
        )

    enclosure = proposal.path_enclosure
    assert enclosure is not None
    np.testing.assert_allclose(
        enclosure.position_lower_m[1] - translation[1],
        enclosure.position_lower_m[0],
        rtol=0.0,
        atol=2.0 * coordinate_ulp,
    )
    np.testing.assert_allclose(
        enclosure.position_upper_m[1] - translation[1],
        enclosure.position_upper_m[0],
        rtol=0.0,
        atol=2.0 * coordinate_ulp,
    )


def test_cubic_hermite_adjacent_time_enclosure_has_no_world_scale_floor() -> None:
    origin = float(2**20)
    coordinate_ulp = np.spacing(origin)
    start = np.asarray([[origin, -origin]], dtype="<f8")
    displacement = np.asarray([[32.0, -16.0]], dtype="<f8") * coordinate_ulp
    root = cubic_hermite_step(
        np.asarray([9], dtype="<i8"),
        np.asarray([0.0]),
        np.asarray([1.0]),
        start,
        displacement,
        np.asarray([0.0]),
        start + displacement,
        displacement,
        support_inside=np.ones(1, dtype=np.bool_),
        applicability_inside=np.ones(1, dtype=np.bool_),
        numerical_status=np.zeros(1, dtype=np.uint8),
    )
    interval_start = 0.5
    interval_target = np.nextafter(interval_start, 1.0)
    restricted = restrict_cubic_hermite_proposal(
        root,
        np.asarray([0], dtype="<i8"),
        np.asarray([interval_start]),
        np.asarray([interval_target]),
    )
    enclosure = restricted.path_enclosure
    assert enclosure is not None

    assert bool((restricted.start_position_m >= enclosure.position_lower_m).all())
    assert bool((restricted.start_position_m <= enclosure.position_upper_m).all())
    assert bool((restricted.end_position_m >= enclosure.position_lower_m).all())
    assert bool((restricted.end_position_m <= enclosure.position_upper_m).all())
    width = enclosure.position_upper_m - enclosure.position_lower_m
    old_world_scale_width = 2.0 * 64.0 * np.finfo(np.float64).eps * abs(origin)
    assert float(np.max(width)) < old_world_scale_width / 8.0
    assert float(np.max(width)) <= 8.0 * coordinate_ulp


class _FailingDynamicsStage:
    def __init__(self, failure_call: int, failed_particle: int = 22) -> None:
        self.failure_call = failure_call
        self.failed_particle = failed_particle
        self.calls: dict[int, int] = {}

    def __call__(
        self,
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        status = np.full(particle_index.size, NUMERICAL_STATUS_OK, dtype=np.uint8)
        for row, particle in enumerate(particle_index):
            key = int(particle)
            call = self.calls.get(key, 0)
            self.calls[key] = call + 1
            if key == self.failed_particle and call == self.failure_call:
                status[row] = PHYSICS_NUMERICAL_FAILURE
        acceleration = np.column_stack(
            (
                0.3 * position_m[:, 0] - 0.1 * velocity_m_s[:, 0] + 0.02 * time_s,
                -0.2 * position_m[:, 1] + 0.15 * velocity_m_s[:, 1] - 0.01 * time_s,
            )
        )
        return DynamicsEvaluation(
            acceleration,
            0.01 * charge_number,
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            status,
            particle_index.copy(),
        )


class _FailingRelaxationStage:
    def __init__(self, failure_call: int, failed_particle: int = 22) -> None:
        self.failure_call = failure_call
        self.failed_particle = failed_particle
        self.calls: dict[int, int] = {}

    def __call__(
        self,
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del position_m, velocity_m_s, charge_number
        status = np.full(particle_index.size, NUMERICAL_STATUS_OK, dtype=np.uint8)
        for row, particle in enumerate(particle_index):
            key = int(particle)
            call = self.calls.get(key, 0)
            self.calls[key] = call + 1
            if key == self.failed_particle and call == self.failure_call:
                status[row] = PHYSICS_NUMERICAL_FAILURE
        rate = 0.4 + 0.01 * particle_index
        target = np.column_stack((0.02 * particle_index, -0.01 * particle_index))
        additive = np.column_stack((0.03 + 0.01 * time_s, -0.04 + 0.02 * time_s))
        return RelaxationEvaluation(
            rate,
            target,
            additive,
            np.zeros(particle_index.size, dtype="<f8"),
            np.zeros(particle_index.size, dtype="<f8"),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            status,
            particle_index.copy(),
        )


class _FailingAccelerationBounder:
    def __init__(self, failure_call: int, failed_particle: int = 22) -> None:
        self.failure_call = failure_call
        self.failed_particle = failed_particle
        self.calls: dict[int, int] = {}

    def __call__(
        self,
        particle_index: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        status = np.full(particle_index.size, NUMERICAL_STATUS_OK, dtype=np.uint8)
        for row, particle in enumerate(particle_index):
            key = int(particle)
            call = self.calls.get(key, 0)
            self.calls[key] = call + 1
            if key == self.failed_particle and call == self.failure_call:
                status[row] = PHYSICS_NUMERICAL_FAILURE
        bound = 0.25 + 0.1 * velocity_abs_upper_m_s
        bound[status != NUMERICAL_STATUS_OK] = np.finfo(np.float64).max
        return bound, status


def test_linear_proposal_preserves_release_state_and_exact_endpoint() -> None:
    particle_index = np.asarray([2, 5], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.25], dtype="<f8")
    position_m = np.asarray([[1.0, -2.0], [-0.0, 3.0]], dtype="<f8")
    velocity_m_s = np.asarray([[0.5, 0.25], [-2.0, 1.0]], dtype="<f8")
    charge_number = np.asarray([-1.0, 3.0], dtype="<f8")

    proposal = rk4_step(
        particle_index,
        start_time_s,
        np.full(particle_index.size, 1.0, dtype="<f8"),
        position_m,
        velocity_m_s,
        charge_number,
        requires_stage_evaluation=False,
        evaluator=None,
    )
    at_release = proposal.state_at(0.25)

    assert proposal.end_field_cell_id is None
    np.testing.assert_array_equal(at_release.particle_index, [2, 5])
    np.testing.assert_array_equal(at_release.position_m[1], position_m[1])
    np.testing.assert_array_equal(at_release.charge_number, charge_number)
    assert np.signbit(at_release.position_m[1, 0])
    np.testing.assert_allclose(at_release.position_m[0], [1.125, -1.9375], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        proposal.end_position(),
        [[1.5, -1.75], [-1.5, 3.75]],
        rtol=0.0,
        atol=0.0,
    )


def test_path_enclosure_rejects_status_that_would_wrap_to_ok() -> None:
    position_m = np.asarray([[0.0, 0.0]], dtype="<f8")
    velocity_m_s = np.asarray([[1.0, 0.0]], dtype="<f8")
    enclosure = CurvedPathEnclosure(
        position_m.copy(),
        position_m.copy(),
        velocity_m_s.copy(),
        velocity_m_s.copy(),
        np.asarray([256], dtype="<i2"),
    )

    with pytest.raises(ValueError, match="uint8"):
        rk4_step(
            np.asarray([0], dtype="<i8"),
            np.asarray([0.0], dtype="<f8"),
            np.asarray([0.5], dtype="<f8"),
            position_m,
            velocity_m_s,
            np.asarray([0.0], dtype="<f8"),
            requires_stage_evaluation=False,
            evaluator=None,
            path_enclosure=enclosure,
        )


def test_rk4_proposal_uses_one_row_local_target_batch() -> None:
    proposal = rk4_step(
        np.asarray([2, 5], dtype="<i8"),
        np.asarray([0.0, 0.25], dtype="<f8"),
        np.asarray([1.0, 0.75], dtype="<f8"),
        np.asarray([[1.0, -2.0], [0.0, 3.0]], dtype="<f8"),
        np.asarray([[0.5, 0.25], [-2.0, 1.0]], dtype="<f8"),
        np.asarray([-1.0, 3.0], dtype="<f8"),
        requires_stage_evaluation=False,
        evaluator=None,
    )

    np.testing.assert_array_equal(proposal.target_time_s, [1.0, 0.75])
    np.testing.assert_allclose(
        proposal.end_position_m,
        [[1.5, -1.75], [-1.0, 3.5]],
        rtol=0.0,
        atol=0.0,
    )
    sample = proposal.state_at_rows(0.5, np.asarray([0, 1], dtype="<i8"))
    np.testing.assert_allclose(sample.position_m, [[1.25, -1.875], [-0.5, 3.25]])


def test_mixed_target_rk4_batch_is_bitwise_equal_to_single_rows() -> None:
    particle_index = np.asarray([3, 7], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.2], dtype="<f8")
    target_time_s = np.asarray([0.75, 0.45], dtype="<f8")
    position_m = np.asarray([[-0.4, 0.25], [0.6, -0.1]], dtype="<f8")
    velocity_m_s = np.asarray([[0.7, -0.5], [-0.2, 0.9]], dtype="<f8")
    charge_number = np.asarray([1.0, -2.0], dtype="<f8")

    def evaluate(
        selected_particle: np.ndarray,
        time_s: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        charge: np.ndarray,
    ) -> DynamicsEvaluation:
        del position
        rate = 0.5 + 0.05 * selected_particle
        target = np.column_stack((0.02 * selected_particle, -0.01 * selected_particle))
        additive = np.column_stack((0.1 + 0.0 * rate, -0.2 + 0.0 * rate))
        acceleration = rate[:, None] * (target - velocity) + additive
        return DynamicsEvaluation(
            acceleration,
            0.03 * charge + 0.01 * time_s,
            np.ones(selected_particle.size, dtype=np.bool_),
            np.ones(selected_particle.size, dtype=np.bool_),
            np.zeros(selected_particle.size, dtype=np.uint8),
            np.floor(100.0 * velocity[:, 0]).astype("<i8"),
        )

    def bound_acceleration(
        selected_particle: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rate = 0.5 + 0.05 * selected_particle
        target_abs = np.column_stack((0.02 * selected_particle, 0.01 * selected_particle))
        return _valid_bound(rate[:, None] * (target_abs + velocity_abs_upper_m_s) + [0.1, 0.2])

    batch_enclosure = enclose_rk4_path(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        acceleration_abs_bounder=bound_acceleration,
    )
    batch = rk4_step(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        charge_number,
        requires_stage_evaluation=True,
        evaluator=evaluate,
        path_enclosure=batch_enclosure,
    )

    for row in range(particle_index.size):
        selected = np.asarray([row], dtype="<i8")
        enclosure = enclose_rk4_path(
            particle_index[selected],
            start_time_s[selected],
            target_time_s[selected],
            position_m[selected],
            velocity_m_s[selected],
            acceleration_abs_bounder=bound_acceleration,
        )
        proposal = rk4_step(
            particle_index[selected],
            start_time_s[selected],
            target_time_s[selected],
            position_m[selected],
            velocity_m_s[selected],
            charge_number[selected],
            requires_stage_evaluation=True,
            evaluator=evaluate,
            path_enclosure=enclosure,
        )
        np.testing.assert_array_equal(batch.end_position_m[row], proposal.end_position_m[0])
        np.testing.assert_array_equal(batch.end_velocity_m_s[row], proposal.end_velocity_m_s[0])
        np.testing.assert_array_equal(batch.end_charge_number[row], proposal.end_charge_number[0])
        assert batch.end_field_cell_id is not None
        assert proposal.end_field_cell_id is not None
        assert batch.end_field_cell_id[row] == proposal.end_field_cell_id[0]
        assert batch.path_enclosure is not None
        assert proposal.path_enclosure is not None
        for name in (
            "position_lower_m",
            "position_upper_m",
            "velocity_lower_m_s",
            "velocity_upper_m_s",
        ):
            np.testing.assert_array_equal(
                getattr(batch.path_enclosure, name)[row],
                getattr(proposal.path_enclosure, name)[0],
            )


@pytest.mark.parametrize("failure_call", range(5))
def test_rk4_localizes_each_stage_failure_and_preserves_neighbor(failure_call: int) -> None:
    particle_index = np.asarray([11, 22], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.1], dtype="<f8")
    target_time_s = np.asarray([0.4, 0.5], dtype="<f8")
    position_m = np.asarray([[0.2, -0.3], [-0.4, 0.5]], dtype="<f8")
    velocity_m_s = np.asarray([[0.6, -0.2], [0.1, 0.7]], dtype="<f8")
    charge_number = np.asarray([1.5, -0.5], dtype="<f8")
    evaluator = _FailingDynamicsStage(failure_call)

    mixed = rk4_step(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        charge_number,
        requires_stage_evaluation=True,
        evaluator=evaluator,
    )
    oracle = rk4_step(
        particle_index[[0]],
        start_time_s[[0]],
        target_time_s[[0]],
        position_m[[0]],
        velocity_m_s[[0]],
        charge_number[[0]],
        requires_stage_evaluation=True,
        evaluator=_FailingDynamicsStage(failure_call),
    )

    np.testing.assert_array_equal(
        mixed.numerical_status,
        [NUMERICAL_STATUS_OK, PHYSICS_NUMERICAL_FAILURE],
    )
    np.testing.assert_array_equal(mixed.end_position_m[[0]], oracle.end_position_m)
    np.testing.assert_array_equal(mixed.end_velocity_m_s[[0]], oracle.end_velocity_m_s)
    np.testing.assert_array_equal(mixed.end_charge_number[[0]], oracle.end_charge_number)
    assert np.isfinite(mixed.end_position_m).all()
    assert np.isfinite(mixed.end_velocity_m_s).all()
    assert np.isfinite(mixed.end_charge_number).all()
    assert evaluator.calls[11] == 5
    assert evaluator.calls[22] == failure_call + 1


@pytest.mark.parametrize("failure_call", range(3))
def test_exponential_localizes_each_stage_failure_and_preserves_neighbor(
    failure_call: int,
) -> None:
    particle_index = np.asarray([11, 22], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.1], dtype="<f8")
    target_time_s = np.asarray([0.4, 0.5], dtype="<f8")
    position_m = np.asarray([[0.2, -0.3], [-0.4, 0.5]], dtype="<f8")
    velocity_m_s = np.asarray([[0.6, -0.2], [0.1, 0.7]], dtype="<f8")
    charge_number = np.asarray([1.5, -0.5], dtype="<f8")
    evaluator = _FailingRelaxationStage(failure_call)

    mixed = exponential_midpoint_step(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        charge_number,
        evaluator=evaluator,
    )
    oracle = exponential_midpoint_step(
        particle_index[[0]],
        start_time_s[[0]],
        target_time_s[[0]],
        position_m[[0]],
        velocity_m_s[[0]],
        charge_number[[0]],
        evaluator=_FailingRelaxationStage(failure_call),
    )

    np.testing.assert_array_equal(
        mixed.numerical_status,
        [NUMERICAL_STATUS_OK, PHYSICS_NUMERICAL_FAILURE],
    )
    np.testing.assert_array_equal(mixed.end_position_m[[0]], oracle.end_position_m)
    np.testing.assert_array_equal(mixed.end_velocity_m_s[[0]], oracle.end_velocity_m_s)
    np.testing.assert_array_equal(mixed.end_charge_number[[0]], oracle.end_charge_number)
    assert np.isfinite(mixed.end_position_m).all()
    assert np.isfinite(mixed.end_velocity_m_s).all()
    assert np.isfinite(mixed.end_charge_number).all()
    assert evaluator.calls[11] == 3
    assert evaluator.calls[22] == failure_call + 1


def test_stage_evaluators_reject_status_that_would_wrap_to_ok() -> None:
    particle_index = np.asarray([0], dtype="<i8")
    start_time_s = np.asarray([0.0], dtype="<f8")
    target_time_s = np.asarray([0.25], dtype="<f8")
    position_m = np.asarray([[0.0, 0.0]], dtype="<f8")
    velocity_m_s = np.asarray([[0.1, -0.2]], dtype="<f8")
    charge_number = np.asarray([0.0], dtype="<f8")

    def dynamics(
        selected_particle: np.ndarray,
        time_s: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        charge: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position, velocity
        count = selected_particle.size
        return DynamicsEvaluation(
            np.zeros((count, 2), dtype="<f8"),
            np.zeros_like(charge),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.full(count, 256, dtype="<i2"),
        )

    def relaxation(
        selected_particle: np.ndarray,
        time_s: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        charge: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, position, velocity
        count = selected_particle.size
        return RelaxationEvaluation(
            np.zeros(count, dtype="<f8"),
            np.zeros((count, 2), dtype="<f8"),
            np.zeros((count, 2), dtype="<f8"),
            np.zeros_like(charge),
            np.zeros_like(charge),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.full(count, 256, dtype="<i2"),
        )

    with pytest.raises(ValueError, match="stage numerical status must have dtype uint8"):
        rk4_step(
            particle_index,
            start_time_s,
            target_time_s,
            position_m,
            velocity_m_s,
            charge_number,
            requires_stage_evaluation=True,
            evaluator=dynamics,
        )
    with pytest.raises(ValueError, match="relaxation numerical status must have dtype uint8"):
        exponential_midpoint_step(
            particle_index,
            start_time_s,
            target_time_s,
            position_m,
            velocity_m_s,
            charge_number,
            evaluator=relaxation,
        )


def test_coupled_rk4_keeps_only_endpoint_field_cell_id_across_replay() -> None:
    evaluated_cell_ids: list[np.ndarray] = []

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s
        count = particle_index.size
        acceleration = np.zeros((count, 2), dtype="<f8")
        acceleration[:, 0] = 3.0 * position_m[:, 0] - 0.25 * velocity_m_s[:, 0]
        cell_id = np.floor(100.0 * position_m[:, 0]).astype("<i8")
        evaluated_cell_ids.append(cell_id.copy())
        return DynamicsEvaluation(
            acceleration,
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
            cell_id,
        )

    proposal = rk4_step(
        np.asarray([7, 11], dtype="<i8"),
        np.asarray([0.0, 0.0], dtype="<f8"),
        np.full(2, 0.4, dtype="<f8"),
        np.asarray([[0.25, 0.0], [-0.4, 0.0]], dtype="<f8"),
        np.asarray([[0.6, 0.0], [0.2, 0.0]], dtype="<f8"),
        np.asarray([1.0, -1.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )

    assert len(evaluated_cell_ids) == 5
    assert proposal.end_field_cell_id is not None
    np.testing.assert_array_equal(proposal.end_field_cell_id, evaluated_cell_ids[-1])
    assert not np.array_equal(evaluated_cell_ids[-1], evaluated_cell_ids[-2])
    accepted_endpoint_id = proposal.end_field_cell_id.copy()

    proposal.state_at_rows(0.2, np.asarray([0], dtype="<i8"))

    np.testing.assert_array_equal(proposal.end_field_cell_id, accepted_endpoint_id)


def test_linear_proposal_statuses_nonfinite_derived_position() -> None:
    start_position = np.asarray([[1.0e308, 0.0]], dtype="<f8")
    proposal = rk4_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([2.0], dtype="<f8"),
        start_position,
        np.asarray([[1.0e308, 0.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        requires_stage_evaluation=False,
        evaluator=None,
    )

    np.testing.assert_array_equal(proposal.numerical_status, [3])
    np.testing.assert_array_equal(proposal.end_position_m, start_position)


def test_constant_acceleration_proposal_has_exact_quadratic_dense_state() -> None:
    acceleration_m_s2 = np.asarray([[-2.0, 0.5]], dtype="<f8")
    evaluated_cell_ids: list[np.ndarray] = []

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, velocity_m_s
        count = particle_index.size
        cell_id = np.floor(100.0 * position_m[:, 1]).astype("<i8")
        evaluated_cell_ids.append(cell_id.copy())
        return DynamicsEvaluation(
            np.repeat(acceleration_m_s2, count, axis=0),
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
            cell_id,
        )

    proposal = rk4_step(
        np.asarray([17], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([2.0], dtype="<f8"),
        np.asarray([[0.5, -0.25]], dtype="<f8"),
        np.asarray([[2.0, 1.0]], dtype="<f8"),
        np.asarray([3.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
        constant_acceleration_m_s2=acceleration_m_s2,
    )
    assert len(evaluated_cell_ids) == 3
    assert proposal.end_field_cell_id is not None
    np.testing.assert_array_equal(proposal.end_field_cell_id, evaluated_cell_ids[-1])
    accepted_endpoint_id = proposal.end_field_cell_id.copy()
    sample = proposal.state_at_rows(0.25, np.asarray([0], dtype="<i8"))

    assert proposal.path_kind == "quadratic_exact"
    np.testing.assert_allclose(sample.position_m, [[0.9375, 0.015625]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sample.velocity_m_s, [[1.5, 1.125]], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(sample.charge_number, [3.0])
    np.testing.assert_allclose(proposal.end_position(), [[0.5, 2.75]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(proposal.end_velocity(), [[-2.0, 2.0]], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(proposal.support_inside, [True])
    np.testing.assert_array_equal(proposal.applicability_inside, [True])
    assert len(evaluated_cell_ids) == 6
    np.testing.assert_array_equal(proposal.end_field_cell_id, accepted_endpoint_id)
    assert not np.array_equal(evaluated_cell_ids[-1], accepted_endpoint_id)


def test_quadratic_position_interval_contains_rounded_turning_state() -> None:
    """The support interval must contain float neighbours of the analytic turn."""

    start_x_m = -0.33783093034230727
    velocity_x_m_s = 69.0264926243172
    acceleration_x_m_s2 = -6186.018510728531
    duration_s = 0.013130149978405609

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s
        count = particle_index.size
        return DynamicsEvaluation(
            np.repeat([[acceleration_x_m_s2, 0.0]], count, axis=0),
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([duration_s], dtype="<f8"),
        np.asarray([[start_x_m, 0.0]], dtype="<f8"),
        np.asarray([[velocity_x_m_s, 0.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
        constant_acceleration_m_s2=np.asarray([[acceleration_x_m_s2, 0.0]], dtype="<f8"),
    )
    turn_time_s = -velocity_x_m_s / acceleration_x_m_s2
    neighbour_times_s = (
        np.nextafter(turn_time_s, -np.inf),
        turn_time_s,
        np.nextafter(turn_time_s, np.inf),
    )
    turn_positions_m = np.asarray(
        [
            proposal.state_at_rows(time_s, np.asarray([0], dtype="<i8")).position_m[0, 0]
            for time_s in neighbour_times_s
        ]
    )
    interval = proposal.quadratic_position_interval()

    assert turn_positions_m[0] > turn_positions_m[1]
    assert interval.lower_m[0, 0] <= np.min(turn_positions_m)
    assert interval.upper_m[0, 0] >= np.max(turn_positions_m)


def test_quadratic_position_interval_contains_amplified_square_underflow() -> None:
    """Underflow in t*t can be amplified by a large finite acceleration."""

    acceleration_x_m_s2 = -1.960451740514388e94

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s
        count = particle_index.size
        return DynamicsEvaluation(
            np.repeat([[acceleration_x_m_s2, 0.0]], count, axis=0),
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([1.2987053736711462e-161], dtype="<f8"),
        np.asarray([[2.83736528175861e-269, 0.0]], dtype="<f8"),
        np.asarray([[9.002034348945205e-81, 0.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
        constant_acceleration_m_s2=np.asarray([[acceleration_x_m_s2, 0.0]], dtype="<f8"),
    )
    sample = proposal.state_at_rows(
        1.9415898571471703e-162,
        np.asarray([0], dtype="<i8"),
    )
    interval = proposal.quadratic_position_interval()

    assert interval.lower_m[0, 0] <= sample.position_m[0, 0]
    assert interval.upper_m[0, 0] >= sample.position_m[0, 0]


@pytest.mark.parametrize("failure_call", range(4))
def test_rk4_enclosure_localizes_bound_failure_and_preserves_neighbor(
    failure_call: int,
) -> None:
    particle_index = np.asarray([11, 22], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.1], dtype="<f8")
    target_time_s = np.asarray([0.4, 0.5], dtype="<f8")
    position_m = np.asarray([[0.2, -0.3], [-0.4, 0.5]], dtype="<f8")
    velocity_m_s = np.asarray([[0.6, -0.2], [0.1, 0.7]], dtype="<f8")
    bounder = _FailingAccelerationBounder(failure_call)

    mixed = enclose_rk4_path(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        acceleration_abs_bounder=bounder,
    )
    oracle = enclose_rk4_path(
        particle_index[[0]],
        start_time_s[[0]],
        target_time_s[[0]],
        position_m[[0]],
        velocity_m_s[[0]],
        acceleration_abs_bounder=_FailingAccelerationBounder(failure_call),
    )

    np.testing.assert_array_equal(
        mixed.numerical_status,
        [NUMERICAL_STATUS_OK, PHYSICS_NUMERICAL_FAILURE],
    )
    for name in (
        "position_lower_m",
        "position_upper_m",
        "velocity_lower_m_s",
        "velocity_upper_m_s",
    ):
        values = getattr(mixed, name)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values[[0]], getattr(oracle, name))
    np.testing.assert_array_equal(mixed.position_lower_m[1], position_m[1])
    np.testing.assert_array_equal(mixed.position_upper_m[1], position_m[1])
    np.testing.assert_array_equal(mixed.velocity_lower_m_s[1], velocity_m_s[1])
    np.testing.assert_array_equal(mixed.velocity_upper_m_s[1], velocity_m_s[1])
    assert bounder.calls[11] == 4
    assert bounder.calls[22] == failure_call + 1


def test_exponential_enclosure_localizes_overflow_and_preserves_neighbor() -> None:
    maximum = np.finfo(np.float64).max
    particle_index = np.asarray([11, 22], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.0], dtype="<f8")
    target_time_s = np.asarray([maximum, 0.25], dtype="<f8")
    position_m = np.asarray([[0.2, -0.3], [-0.4, 0.5]], dtype="<f8")
    velocity_m_s = np.asarray([[0.6, -0.2], [0.1, 0.7]], dtype="<f8")
    rate = np.asarray([maximum, 0.5], dtype="<f8")
    target = np.asarray([[1.0, 1.0], [0.2, 0.3]], dtype="<f8")
    additive = np.asarray([[1.0, 1.0], [0.05, 0.04]], dtype="<f8")
    bounder = _table_acceleration_bounder(particle_index, additive)

    mixed = enclose_exponential_midpoint_path(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        linear_drag_rate_upper_s_inv=rate,
        target_velocity_abs_upper_m_s=target,
        additive_acceleration_abs_bounder=bounder,
    )
    oracle = enclose_exponential_midpoint_path(
        particle_index[[1]],
        start_time_s[[1]],
        target_time_s[[1]],
        position_m[[1]],
        velocity_m_s[[1]],
        linear_drag_rate_upper_s_inv=rate[[1]],
        target_velocity_abs_upper_m_s=target[[1]],
        additive_acceleration_abs_bounder=bounder,
    )

    np.testing.assert_array_equal(
        mixed.numerical_status,
        [INTEGRATOR_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK],
    )
    for name in (
        "position_lower_m",
        "position_upper_m",
        "velocity_lower_m_s",
        "velocity_upper_m_s",
    ):
        values = getattr(mixed, name)
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values[[1]], getattr(oracle, name))
    np.testing.assert_array_equal(mixed.position_lower_m[0], position_m[0])
    np.testing.assert_array_equal(mixed.position_upper_m[0], position_m[0])
    np.testing.assert_array_equal(mixed.velocity_lower_m_s[0], velocity_m_s[0])
    np.testing.assert_array_equal(mixed.velocity_upper_m_s[0], velocity_m_s[0])


def test_rk4_path_enclosure_contains_every_shortened_stage_and_dense_state() -> None:
    particle_index = np.asarray([3, 7], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.2], dtype="<f8")
    end_time_s = 0.8
    start_position_m = np.asarray([[-0.4, 0.25], [0.6, -0.1]], dtype="<f8")
    start_velocity_m_s = np.asarray([[0.7, -0.5], [-0.2, 0.9]], dtype="<f8")
    charge_number = np.asarray([1.0, -2.0], dtype="<f8")
    rate_s_inv = np.asarray([1.25, 2.0], dtype="<f8")
    target_velocity_m_s = np.asarray([[0.3, -0.2], [-0.1, 0.4]], dtype="<f8")
    constant_acceleration_m_s2 = np.asarray([[0.5, -0.25], [-0.3, 0.1]], dtype="<f8")

    def acceleration(local_row: int, velocity_m_s: np.ndarray) -> np.ndarray:
        return (
            rate_s_inv[local_row] * (target_velocity_m_s[local_row] - velocity_m_s)
            + constant_acceleration_m_s2[local_row]
        )

    def evaluate(
        selected_particle: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        selected_charge: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m
        local_row = np.searchsorted(particle_index, selected_particle)
        result = (
            rate_s_inv[local_row, None] * (target_velocity_m_s[local_row] - velocity_m_s)
            + constant_acceleration_m_s2[local_row]
        )
        count = selected_particle.size
        return DynamicsEvaluation(
            result,
            np.zeros_like(selected_charge),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    def bound_acceleration(
        selected_particle: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_row = np.searchsorted(particle_index, selected_particle)
        return _valid_bound(
            rate_s_inv[local_row, None]
            * (np.abs(target_velocity_m_s[local_row]) + velocity_abs_upper_m_s)
            + np.abs(constant_acceleration_m_s2[local_row])
        )

    enclosure = enclose_rk4_path(
        particle_index,
        start_time_s,
        np.full(particle_index.size, end_time_s, dtype="<f8"),
        start_position_m,
        start_velocity_m_s,
        acceleration_abs_bounder=bound_acceleration,
    )
    proposal = rk4_step(
        particle_index,
        start_time_s,
        np.full(particle_index.size, end_time_s, dtype="<f8"),
        start_position_m,
        start_velocity_m_s,
        charge_number,
        requires_stage_evaluation=True,
        evaluator=evaluate,
        path_enclosure=enclosure,
    )

    assert proposal.path_enclosure is not enclosure
    for row in range(2):
        duration_s = end_time_s - start_time_s[row]
        chord_deviation_m = curved_chord_deviation_bound(
            start_position_m[row],
            proposal.end_position_m[row],
            enclosure.velocity_lower_m_s[row],
            enclosure.velocity_upper_m_s[row],
            duration_s,
        )
        for elapsed_s in np.linspace(0.0, duration_s, 41):
            position_1 = start_position_m[row]
            velocity_1 = start_velocity_m_s[row]
            acceleration_1 = acceleration(row, velocity_1)
            position_2 = position_1 + 0.5 * elapsed_s * velocity_1
            velocity_2 = velocity_1 + 0.5 * elapsed_s * acceleration_1
            acceleration_2 = acceleration(row, velocity_2)
            position_3 = position_1 + 0.5 * elapsed_s * velocity_2
            velocity_3 = velocity_1 + 0.5 * elapsed_s * acceleration_2
            acceleration_3 = acceleration(row, velocity_3)
            position_4 = position_1 + elapsed_s * velocity_3
            velocity_4 = velocity_1 + elapsed_s * acceleration_3
            acceleration_4 = acceleration(row, velocity_4)
            end_position = position_1 + (elapsed_s / 6.0) * (
                velocity_1 + 2.0 * velocity_2 + 2.0 * velocity_3 + velocity_4
            )
            end_velocity = velocity_1 + (elapsed_s / 6.0) * (
                acceleration_1 + 2.0 * acceleration_2 + 2.0 * acceleration_3 + acceleration_4
            )
            for position in (position_1, position_2, position_3, position_4, end_position):
                assert np.all(position >= enclosure.position_lower_m[row])
                assert np.all(position <= enclosure.position_upper_m[row])
            for velocity in (velocity_1, velocity_2, velocity_3, velocity_4, end_velocity):
                assert np.all(velocity >= enclosure.velocity_lower_m_s[row])
                assert np.all(velocity <= enclosure.velocity_upper_m_s[row])
            dense_position_m = proposal.state_at_rows(
                start_time_s[row] + elapsed_s,
                np.asarray([row], dtype="<i8"),
            ).position_m[0]
            chord_position_m = start_position_m[row] + (elapsed_s / duration_s) * (
                proposal.end_position_m[row] - start_position_m[row]
            )
            assert np.all(np.abs(dense_position_m - chord_position_m) <= chord_deviation_m)


def test_rk4_chord_roundoff_bound_survives_large_coordinate_translation() -> None:
    bound_m = curved_chord_deviation_bound(
        np.asarray([9.0e307, 0.0]),
        np.asarray([9.0e307, 0.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 0.0]),
        1.0,
    )

    assert np.all(np.isfinite(bound_m))
    assert np.all(bound_m > 0.0)


def test_curved_chord_batch_preserves_row_bounds() -> None:
    start = np.asarray([[0.2, -0.4], [9.0e307, 0.0], [0.0, 0.0]], dtype="<f8")
    end = np.asarray([[0.7, -0.1], [9.0e307, 0.0], [0.0, 0.0]], dtype="<f8")
    velocity_lower = np.asarray(
        [[-2.0, -0.5], [0.0, 0.0], [0.0, 0.0]],
        dtype="<f8",
    )
    velocity_upper = np.asarray(
        [[3.0, 1.5], [0.0, 0.0], [0.0, 0.0]],
        dtype="<f8",
    )
    interval = np.asarray(
        [0.25, 1.0, np.finfo(np.float64).smallest_subnormal],
        dtype="<f8",
    )

    batch = curved_chord_deviation_bounds(
        start,
        end,
        velocity_lower,
        velocity_upper,
        interval,
    )
    rows = np.asarray(
        [
            curved_chord_deviation_bound(
                start[row],
                end[row],
                velocity_lower[row],
                velocity_upper[row],
                float(interval[row]),
            )
            for row in range(interval.size)
        ]
    )

    assert np.array_equal(batch, rows)
    assert np.all(np.isfinite(batch))
    assert np.all(batch > 0.0)


def test_rk4_chord_bound_covers_division_underflow_amplified_by_velocity() -> None:
    smallest = np.finfo(np.float64).smallest_subnormal
    interval_s = 4.0 * smallest
    velocity_m_s = np.asarray([[1.0e307, 0.0]], dtype="<f8")

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        stage_velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, stage_velocity_m_s
        return DynamicsEvaluation(
            np.zeros((particle_index.size, 2), dtype="<f8"),
            np.zeros_like(charge_number),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([interval_s], dtype="<f8"),
        np.asarray([[0.0, 0.0]], dtype="<f8"),
        velocity_m_s,
        np.asarray([0.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    halfway = proposal.state_at_rows(0.5 * interval_s, np.asarray([0], dtype="<i8"))
    chord_position_m = 0.5 * proposal.end_position_m[0]
    bound_m = curved_chord_deviation_bound(
        proposal.start_position_m[0],
        proposal.end_position_m[0],
        velocity_m_s[0],
        velocity_m_s[0],
        interval_s,
    )

    assert np.all(np.abs(halfway.position_m[0] - chord_position_m) <= bound_m)


def test_rk4_global_enclosure_remains_conservative_around_the_hermite_path() -> None:
    slope_s2_inverse = -42.7669775174
    offset_m_s2 = 0.2780151248
    step_s = 0.2911182108
    position_m = np.asarray([[0.7178049036, 0.0]], dtype="<f8")
    velocity_m_s = np.asarray([[-7.0408375495, 0.0]], dtype="<f8")

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        stage_position_m: np.ndarray,
        stage_velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, stage_velocity_m_s
        acceleration = np.zeros((particle_index.size, 2), dtype="<f8")
        provisional_x = np.clip(stage_position_m[:, 0], -1.0, 1.0)
        acceleration[:, 0] = slope_s2_inverse * provisional_x + offset_m_s2
        return DynamicsEvaluation(
            acceleration,
            np.zeros_like(charge_number),
            np.abs(stage_position_m[:, 0]) <= 1.0,
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    def bound_acceleration(
        particle_index: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del velocity_abs_upper_m_s
        result = np.zeros((particle_index.size, 2), dtype="<f8")
        result[:, 0] = 44.0
        return _valid_bound(result)

    enclosure = enclose_rk4_path(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([step_s], dtype="<f8"),
        position_m,
        velocity_m_s,
        acceleration_abs_bounder=bound_acceleration,
    )
    proposal = rk4_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([step_s], dtype="<f8"),
        position_m,
        velocity_m_s,
        np.asarray([1.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
        path_enclosure=enclosure,
    )

    np.testing.assert_array_equal(proposal.support_inside, [True])
    assert proposal.end_position_m[0, 0] > -1.0
    assert enclosure.position_lower_m[0, 0] < -1.0
    assert proposal.path_enclosure is not None
    np.testing.assert_array_equal(
        proposal.path_enclosure.position_lower_m,
        enclosure.position_lower_m,
    )
    np.testing.assert_array_equal(
        proposal.path_enclosure.position_upper_m,
        enclosure.position_upper_m,
    )
    theta = 0.87
    provisional = proposal.state_at_rows(theta * step_s, np.asarray([0], dtype="<i8"))
    expected_position = (
        (2.0 * theta**3 - 3.0 * theta**2 + 1.0) * position_m
        + (theta**3 - 2.0 * theta**2 + theta) * step_s * velocity_m_s
        + (-2.0 * theta**3 + 3.0 * theta**2) * proposal.end_position_m
        + (theta**3 - theta**2) * step_s * proposal.end_velocity_m_s
    )
    np.testing.assert_allclose(provisional.position_m, expected_position, rtol=0.0, atol=5.0e-16)
    assert provisional.position_m[0, 0] > -1.0
    # The immutable polynomial does not call the RHS while sampling.  The
    # certificate owner evaluates support/applicability at requested samples.
    np.testing.assert_array_equal(provisional.support_inside, [True])
    np.testing.assert_array_equal(provisional.applicability_inside, [True])


def test_zero_duration_rk4_enclosure_is_the_exact_start_state() -> None:
    position_m = np.asarray([[1.0, -2.0], [0.25, 0.5]], dtype="<f8")
    velocity_m_s = np.asarray(
        [[-np.finfo(np.float64).max, 4.0], [1.0, -0.5]],
        dtype="<f8",
    )

    def zero_acceleration(
        particle_index: np.ndarray,
        velocity_abs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        np.testing.assert_array_equal(particle_index, [17])
        return _valid_bound(np.zeros((particle_index.size, velocity_abs.shape[1]), dtype="<f8"))

    enclosure = enclose_rk4_path(
        np.asarray([9, 17], dtype="<i8"),
        np.asarray([0.5, 0.25], dtype="<f8"),
        np.asarray([0.5, 0.5], dtype="<f8"),
        position_m,
        velocity_m_s,
        acceleration_abs_bounder=zero_acceleration,
    )

    np.testing.assert_array_equal(enclosure.position_lower_m[0], position_m[0])
    np.testing.assert_array_equal(enclosure.position_upper_m[0], position_m[0])
    np.testing.assert_array_equal(enclosure.velocity_lower_m_s[0], velocity_m_s[0])
    np.testing.assert_array_equal(enclosure.velocity_upper_m_s[0], velocity_m_s[0])
    moving_endpoint = position_m[1] + 0.25 * velocity_m_s[1]
    assert bool((moving_endpoint >= enclosure.position_lower_m[1]).all())
    assert bool((moving_endpoint <= enclosure.position_upper_m[1]).all())


def test_coupled_rk4_has_fourth_order_convergence() -> None:
    rate_s_inv = 4.0
    target_velocity_m_s = np.asarray([-0.2, 0.3], dtype="<f8")
    constant_acceleration_m_s2 = np.asarray([0.8, -0.4], dtype="<f8")
    initial_position_m = np.asarray([-0.25, -0.1], dtype="<f8")
    initial_velocity_m_s = np.asarray([0.6, -0.5], dtype="<f8")
    end_time_s = 0.75

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m
        count = particle_index.size
        acceleration = (
            rate_s_inv * (target_velocity_m_s[None, :] - velocity_m_s)
            + constant_acceleration_m_s2[None, :]
        )
        return DynamicsEvaluation(
            acceleration,
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    decay = math.exp(-rate_s_inv * end_time_s)
    relaxation = 1.0 - decay
    tau_s = 1.0 / rate_s_inv
    expected_velocity = (
        target_velocity_m_s
        + decay * (initial_velocity_m_s - target_velocity_m_s)
        + tau_s * relaxation * constant_acceleration_m_s2
    )
    expected_position = (
        initial_position_m
        + target_velocity_m_s * end_time_s
        + tau_s * relaxation * (initial_velocity_m_s - target_velocity_m_s)
        + tau_s * (end_time_s - tau_s * relaxation) * constant_acceleration_m_s2
    )

    position_errors: list[float] = []
    velocity_errors: list[float] = []
    step_sizes = [0.125, 0.0625, 0.03125, 0.015625]
    for step_s in step_sizes:
        position = initial_position_m[None, :].copy()
        velocity = initial_velocity_m_s[None, :].copy()
        charge = np.asarray([2.5], dtype="<f8")
        step_count = round(end_time_s / step_s)
        for step_index in range(step_count):
            start_s = step_index * step_s
            proposal = rk4_step(
                np.asarray([17], dtype="<i8"),
                np.asarray([start_s], dtype="<f8"),
                np.asarray([(step_index + 1) * step_s], dtype="<f8"),
                position,
                velocity,
                charge,
                requires_stage_evaluation=True,
                evaluator=evaluate,
            )
            position = proposal.end_position()
            velocity = proposal.end_velocity()
            charge = proposal.end_charge()
        position_errors.append(float(np.max(np.abs(position[0] - expected_position))))
        velocity_errors.append(float(np.max(np.abs(velocity[0] - expected_velocity))))
        np.testing.assert_array_equal(charge, [2.5])

    assert all(left > right for left, right in pairwise(position_errors))
    assert all(left > right for left, right in pairwise(velocity_errors))
    for errors in (position_errors, velocity_errors):
        observed_orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in (1, 2)]
        assert min(observed_orders) > 3.5
    assert position_errors[-1] < 5.0e-9
    assert velocity_errors[-1] < 2.0e-8


def test_rk4_couples_charge_and_motion_at_fourth_order() -> None:
    decay_rate_s_inv = 1.5
    end_time_s = 1.0

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s
        count = particle_index.size
        acceleration = np.zeros((count, 2), dtype="<f8")
        acceleration[:, 0] = charge_number
        return DynamicsEvaluation(
            acceleration,
            -decay_rate_s_inv * charge_number,
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    decay = math.exp(-decay_rate_s_inv * end_time_s)
    expected_charge = decay
    expected_velocity = (1.0 - decay) / decay_rate_s_inv
    expected_position = end_time_s / decay_rate_s_inv - (1.0 - decay) / decay_rate_s_inv**2
    errors: list[float] = []
    for step_count in (8, 16, 32, 64):
        step_s = end_time_s / step_count
        position = np.zeros((1, 2), dtype="<f8")
        velocity = np.zeros((1, 2), dtype="<f8")
        charge = np.ones(1, dtype="<f8")
        for step_index in range(step_count):
            proposal = rk4_step(
                np.asarray([0], dtype="<i8"),
                np.asarray([step_index * step_s], dtype="<f8"),
                np.asarray([(step_index + 1) * step_s], dtype="<f8"),
                position,
                velocity,
                charge,
                requires_stage_evaluation=True,
                evaluator=evaluate,
            )
            position = proposal.end_position()
            velocity = proposal.end_velocity()
            charge = proposal.end_charge()
        errors.append(
            max(
                abs(float(position[0, 0]) - expected_position),
                abs(float(velocity[0, 0]) - expected_velocity),
                abs(float(charge[0]) - expected_charge),
            )
        )

    assert all(left > right for left, right in pairwise(errors))
    observed_orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in (1, 2)]
    assert min(observed_orders) > 3.5


def test_coupled_rk4_includes_the_accepted_endpoint_in_support_verdict() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, velocity_m_s
        count = particle_index.size
        acceleration = np.zeros((count, 2), dtype="<f8")
        acceleration[:, 0] = 100.0 * position_m[:, 0]
        return DynamicsEvaluation(
            acceleration,
            np.zeros_like(charge_number),
            position_m[:, 0] <= 100.0,
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([1.0], dtype="<f8"),
        np.asarray([[1.0, 0.0]], dtype="<f8"),
        np.asarray([[0.0, 0.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )

    # Every derivative stage is inside (the last one is at x=51), while the
    # weighted RK4 endpoint is x=1403/3 and must independently be rejected.
    np.testing.assert_allclose(proposal.end_position(), [[1403.0 / 3.0, 0.0]])
    np.testing.assert_array_equal(proposal.support_inside, [False])


def test_zero_duration_coupled_proposal_still_checks_support() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, velocity_m_s
        count = particle_index.size
        return DynamicsEvaluation(
            np.zeros((count, 2), dtype="<f8"),
            np.zeros_like(charge_number),
            position_m[:, 0] < 0.0,
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([4], dtype="<i8"),
        np.asarray([2.0], dtype="<f8"),
        np.asarray([2.0], dtype="<f8"),
        np.asarray([[1.0, 0.0]], dtype="<f8"),
        np.asarray([[0.0, 0.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )

    np.testing.assert_array_equal(proposal.support_inside, [False])


def test_exponential_midpoint_is_exact_for_c03_and_replay_keeps_endpoint_hint() -> None:
    rate_s_inv = 4.0
    target_velocity_m_s = np.asarray([-0.2, 0.3], dtype="<f8")
    acceleration_m_s2 = np.asarray([0.8, -0.4], dtype="<f8")
    initial_position_m = np.asarray([-0.25, -0.1], dtype="<f8")
    initial_velocity_m_s = np.asarray([0.6, -0.5], dtype="<f8")
    end_time_s = 0.75
    evaluated_cell_ids: list[np.ndarray] = []

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, velocity_m_s
        count = particle_index.size
        cell_id = np.floor(100.0 * position_m[:, 0]).astype("<i8")
        evaluated_cell_ids.append(cell_id.copy())
        return RelaxationEvaluation(
            np.full(count, rate_s_inv, dtype="<f8"),
            np.repeat(target_velocity_m_s[None, :], count, axis=0),
            np.repeat(acceleration_m_s2[None, :], count, axis=0),
            np.zeros_like(charge_number),
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
            cell_id,
        )

    decay = math.exp(-rate_s_inv * end_time_s)
    relaxation = 1.0 - decay
    tau_s = 1.0 / rate_s_inv
    expected_velocity = (
        target_velocity_m_s
        + decay * (initial_velocity_m_s - target_velocity_m_s)
        + tau_s * relaxation * acceleration_m_s2
    )
    expected_position = (
        initial_position_m
        + target_velocity_m_s * end_time_s
        + tau_s * relaxation * (initial_velocity_m_s - target_velocity_m_s)
        + tau_s * (end_time_s - tau_s * relaxation) * acceleration_m_s2
    )

    retained_proposal = None
    for step_s in (0.75, 0.375, 0.125):
        position = initial_position_m[None, :].copy()
        velocity = initial_velocity_m_s[None, :].copy()
        charge = np.asarray([2.5], dtype="<f8")
        for step_index in range(round(end_time_s / step_s)):
            proposal = exponential_midpoint_step(
                np.asarray([17], dtype="<i8"),
                np.asarray([step_index * step_s], dtype="<f8"),
                np.asarray([(step_index + 1) * step_s], dtype="<f8"),
                position,
                velocity,
                charge,
                evaluator=evaluate,
            )
            position = proposal.end_position()
            velocity = proposal.end_velocity()
            charge = proposal.end_charge()
        np.testing.assert_allclose(position[0], expected_position, rtol=0.0, atol=1.0e-13)
        np.testing.assert_allclose(velocity[0], expected_velocity, rtol=0.0, atol=1.0e-13)
        np.testing.assert_array_equal(charge, [2.5])
        retained_proposal = proposal

    assert retained_proposal is not None
    assert retained_proposal.path_kind == "exponential_midpoint_reintegrated"
    assert retained_proposal.end_field_cell_id is not None
    accepted_endpoint_id = retained_proposal.end_field_cell_id.copy()
    retained_proposal.state_at_rows(
        float(retained_proposal.target_time_s[0]) - 0.5 * 0.125,
        np.asarray([0], dtype="<i8"),
    )
    np.testing.assert_array_equal(retained_proposal.end_field_cell_id, accepted_endpoint_id)


def test_mixed_target_exponential_batch_is_bitwise_equal_to_single_rows() -> None:
    particle_index = np.asarray([3, 7], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.2], dtype="<f8")
    target_time_s = np.asarray([0.75, 0.45], dtype="<f8")
    position_m = np.asarray([[-0.4, 0.25], [0.6, -0.1]], dtype="<f8")
    velocity_m_s = np.asarray([[0.7, -0.5], [-0.2, 0.9]], dtype="<f8")
    charge_number = np.asarray([1.0, -2.0], dtype="<f8")
    rate_s_inv = 0.4 + 0.03 * particle_index
    target_velocity_m_s = np.column_stack((0.02 * particle_index, -0.01 * particle_index))
    acceleration_m_s2 = np.column_stack((0.1 + 0.0 * particle_index, -0.2 + 0.0 * particle_index))

    def evaluate(
        selected_particle: np.ndarray,
        time_s: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        charge: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, velocity
        local_row = np.searchsorted(particle_index, selected_particle)
        return RelaxationEvaluation(
            rate_s_inv[local_row],
            target_velocity_m_s[local_row],
            acceleration_m_s2[local_row],
            np.zeros_like(charge),
            np.zeros_like(charge),
            np.ones(selected_particle.size, dtype=np.bool_),
            np.ones(selected_particle.size, dtype=np.bool_),
            np.zeros(selected_particle.size, dtype=np.uint8),
            np.floor(100.0 * position[:, 0]).astype("<i8"),
        )

    batch_enclosure = enclose_exponential_midpoint_path(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        linear_drag_rate_upper_s_inv=rate_s_inv,
        target_velocity_abs_upper_m_s=np.abs(target_velocity_m_s),
        additive_acceleration_abs_bounder=_table_acceleration_bounder(
            particle_index,
            np.abs(acceleration_m_s2),
        ),
    )
    batch = exponential_midpoint_step(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        charge_number,
        evaluator=evaluate,
        path_enclosure=batch_enclosure,
    )

    for row in range(particle_index.size):
        selected = np.asarray([row], dtype="<i8")
        enclosure = enclose_exponential_midpoint_path(
            particle_index[selected],
            start_time_s[selected],
            target_time_s[selected],
            position_m[selected],
            velocity_m_s[selected],
            linear_drag_rate_upper_s_inv=rate_s_inv[selected],
            target_velocity_abs_upper_m_s=np.abs(target_velocity_m_s[selected]),
            additive_acceleration_abs_bounder=_table_acceleration_bounder(
                particle_index,
                np.abs(acceleration_m_s2),
            ),
        )
        proposal = exponential_midpoint_step(
            particle_index[selected],
            start_time_s[selected],
            target_time_s[selected],
            position_m[selected],
            velocity_m_s[selected],
            charge_number[selected],
            evaluator=evaluate,
            path_enclosure=enclosure,
        )
        np.testing.assert_array_equal(batch.end_position_m[row], proposal.end_position_m[0])
        np.testing.assert_array_equal(batch.end_velocity_m_s[row], proposal.end_velocity_m_s[0])
        np.testing.assert_array_equal(batch.end_charge_number[row], proposal.end_charge_number[0])
        assert batch.end_field_cell_id is not None
        assert proposal.end_field_cell_id is not None
        assert batch.end_field_cell_id[row] == proposal.end_field_cell_id[0]
        assert batch.path_enclosure is not None
        assert proposal.path_enclosure is not None
        for name in (
            "position_lower_m",
            "position_upper_m",
            "velocity_lower_m_s",
            "velocity_upper_m_s",
        ):
            np.testing.assert_array_equal(
                getattr(batch.path_enclosure, name)[row],
                getattr(proposal.path_enclosure, name)[0],
            )


@pytest.mark.parametrize("rate_s_inv", [0.0, 1.0e-16, 1.0e-10, 1.0e-4, 1.0, 1.0e3])
def test_exponential_midpoint_coefficients_cover_wide_relaxation_range(
    rate_s_inv: float,
) -> None:
    position_m = 0.25
    velocity_m_s = 0.75
    target_m_s = -0.2
    acceleration_m_s2 = 0.4
    step_s = 1.0

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, position, velocity
        count = particle_index.size
        return RelaxationEvaluation(
            np.full(count, rate_s_inv, dtype="<f8"),
            np.repeat([[target_m_s, 0.0]], count, axis=0),
            np.repeat([[acceleration_m_s2, 0.0]], count, axis=0),
            np.zeros_like(charge_number),
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    proposal = exponential_midpoint_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([step_s], dtype="<f8"),
        np.asarray([[position_m, 0.0]], dtype="<f8"),
        np.asarray([[velocity_m_s, 0.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        evaluator=evaluate,
    )
    with localcontext() as context:
        context.prec = 80
        h = Decimal.from_float(step_s)
        x0 = Decimal.from_float(position_m)
        v0 = Decimal.from_float(velocity_m_s)
        target = Decimal.from_float(target_m_s)
        acceleration = Decimal.from_float(acceleration_m_s2)
        rate = Decimal.from_float(rate_s_inv)
        if rate.is_zero():
            expected_velocity = v0 + h * acceleration
            expected_position = x0 + h * v0 + h * h * acceleration / 2
        else:
            decay = (-h * rate).exp()
            relaxation = 1 - decay
            tau = 1 / rate
            expected_velocity = target + decay * (v0 - target) + tau * relaxation * acceleration
            expected_position = (
                x0
                + target * h
                + tau * relaxation * (v0 - target)
                + tau * (h - tau * relaxation) * acceleration
            )
    assert math.isfinite(float(proposal.end_position_m[0, 0]))
    assert math.isfinite(float(proposal.end_velocity_m_s[0, 0]))
    np.testing.assert_allclose(
        proposal.end_position_m[0, 0],
        float(expected_position),
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        proposal.end_velocity_m_s[0, 0],
        float(expected_velocity),
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_exponential_midpoint_has_second_order_for_variable_coefficients() -> None:
    def exact(time_s: float) -> tuple[float, float, float]:
        position = 1.0 + 0.2 * time_s + 0.3 * time_s**2 + 0.1 * time_s**3
        velocity = 0.2 + 0.6 * time_s + 0.3 * time_s**2
        acceleration = 0.6 + 0.6 * time_s
        return position, velocity, acceleration

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del velocity_m_s
        count = particle_index.size
        rate = 2.0 + 0.1 * position_m[:, 0]
        target = 0.1 - 0.05 * position_m[:, 0]
        exact_velocity = 0.2 + 0.6 * time_s + 0.3 * time_s**2
        exact_acceleration = 0.6 + 0.6 * time_s
        additive = exact_acceleration + rate * (exact_velocity - target)
        target_vector = np.zeros((count, 2), dtype="<f8")
        additive_vector = np.zeros((count, 2), dtype="<f8")
        target_vector[:, 0] = target
        additive_vector[:, 0] = additive
        return RelaxationEvaluation(
            rate,
            target_vector,
            additive_vector,
            np.zeros_like(charge_number),
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    exact_position, exact_velocity, _ = exact(1.0)
    errors: list[float] = []
    for step_count in (8, 16, 32, 64):
        step_s = 1.0 / step_count
        position = np.asarray([[1.0, 0.0]], dtype="<f8")
        velocity = np.asarray([[0.2, 0.0]], dtype="<f8")
        for step_index in range(step_count):
            proposal = exponential_midpoint_step(
                np.asarray([0], dtype="<i8"),
                np.asarray([step_index * step_s], dtype="<f8"),
                np.asarray([(step_index + 1) * step_s], dtype="<f8"),
                position,
                velocity,
                np.asarray([0.0], dtype="<f8"),
                evaluator=evaluate,
            )
            position = proposal.end_position()
            velocity = proposal.end_velocity()
        errors.append(
            max(
                abs(float(position[0, 0]) - exact_position),
                abs(float(velocity[0, 0]) - exact_velocity),
            )
        )

    assert all(left > right for left, right in pairwise(errors))
    observed_orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in (1, 2)]
    assert min(observed_orders) > 1.8


def test_exponential_midpoint_uses_finite_stiff_limit_when_h_rate_overflows() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, position_m, velocity_m_s
        count = particle_index.size
        return RelaxationEvaluation(
            np.full(count, 1.0e308, dtype="<f8"),
            np.ones((count, 2), dtype="<f8"),
            np.zeros((count, 2), dtype="<f8"),
            np.zeros_like(charge_number),
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    proposal = exponential_midpoint_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([2.0], dtype="<f8"),
        np.zeros((1, 2), dtype="<f8"),
        np.zeros((1, 2), dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        evaluator=evaluate,
    )

    np.testing.assert_array_equal(proposal.end_position_m, [[2.0, 2.0]])
    np.testing.assert_array_equal(proposal.end_velocity_m_s, [[1.0, 1.0]])


def test_exponential_enclosure_contains_shortened_states() -> None:
    particle_index = np.asarray([3, 7], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.2], dtype="<f8")
    end_time_s = 0.8
    position_m = np.asarray([[-0.4, 0.25], [0.6, -0.1]], dtype="<f8")
    velocity_m_s = np.asarray([[0.7, -0.5], [-0.2, 0.9]], dtype="<f8")
    rate_s_inv = np.asarray([1.25, 2.0], dtype="<f8")
    target_velocity_m_s = np.asarray([[0.3, -0.2], [-0.1, 0.4]], dtype="<f8")
    acceleration_m_s2 = np.asarray([[0.5, -0.25], [-0.3, 0.1]], dtype="<f8")

    def evaluate(
        selected_particle: np.ndarray,
        time_s: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, position, velocity
        local_row = np.searchsorted(particle_index, selected_particle)
        count = selected_particle.size
        return RelaxationEvaluation(
            rate_s_inv[local_row],
            target_velocity_m_s[local_row],
            acceleration_m_s2[local_row],
            np.zeros_like(charge_number),
            np.zeros_like(charge_number),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    enclosure = enclose_exponential_midpoint_path(
        particle_index,
        start_time_s,
        np.full(particle_index.size, end_time_s, dtype="<f8"),
        position_m,
        velocity_m_s,
        linear_drag_rate_upper_s_inv=rate_s_inv,
        target_velocity_abs_upper_m_s=np.abs(target_velocity_m_s),
        additive_acceleration_abs_bounder=_table_acceleration_bounder(
            particle_index,
            np.abs(acceleration_m_s2),
        ),
    )
    proposal = exponential_midpoint_step(
        particle_index,
        start_time_s,
        np.full(particle_index.size, end_time_s, dtype="<f8"),
        position_m,
        velocity_m_s,
        np.asarray([0.0, 1.0], dtype="<f8"),
        evaluator=evaluate,
        path_enclosure=enclosure,
    )

    for row in range(2):
        duration_s = end_time_s - start_time_s[row]
        chord_bound = curved_chord_deviation_bound(
            proposal.start_position_m[row],
            proposal.end_position_m[row],
            enclosure.velocity_lower_m_s[row],
            enclosure.velocity_upper_m_s[row],
            duration_s,
        )
        for elapsed_s in np.linspace(0.0, duration_s, 41):
            sample = proposal.state_at_rows(
                start_time_s[row] + elapsed_s,
                np.asarray([row], dtype="<i8"),
            )
            assert bool((sample.position_m[0] >= enclosure.position_lower_m[row]).all())
            assert bool((sample.position_m[0] <= enclosure.position_upper_m[row]).all())
            assert bool((sample.velocity_m_s[0] >= enclosure.velocity_lower_m_s[row]).all())
            assert bool((sample.velocity_m_s[0] <= enclosure.velocity_upper_m_s[row]).all())
            chord_fraction = elapsed_s / duration_s
            chord_position = proposal.start_position_m[row] + chord_fraction * (
                proposal.end_position_m[row] - proposal.start_position_m[row]
            )
            assert bool((np.abs(sample.position_m[0] - chord_position) <= chord_bound).all())


def test_exponential_enclosure_rebounds_velocity_dependent_additive_term() -> None:
    particle_index = np.asarray([3, 7], dtype="<i8")
    start_time_s = np.asarray([0.0, 0.1], dtype="<f8")
    target_time_s = np.asarray([0.8, 0.65], dtype="<f8")
    position_m = np.asarray([[-0.2, 0.1], [0.3, -0.4]], dtype="<f8")
    velocity_m_s = np.asarray([[0.4, -0.7], [-0.6, 0.2]], dtype="<f8")
    rate_s_inv = np.asarray([0.5, 0.9], dtype="<f8")
    target_velocity_m_s = np.asarray([[0.2, -0.1], [-0.3, 0.4]], dtype="<f8")
    base_acceleration_m_s2 = np.asarray([[0.15, 0.08], [0.06, 0.12]], dtype="<f8")
    coupling_s_inv = np.asarray([1.1, 0.7], dtype="<f8")
    bound_inputs: list[np.ndarray] = []

    def additive_bounder(
        selected_particle: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        bound_inputs.append(velocity_abs_upper_m_s.copy())
        row = np.searchsorted(particle_index, selected_particle)
        bound = (
            base_acceleration_m_s2[row]
            + coupling_s_inv[row, None] * (velocity_abs_upper_m_s[:, ::-1])
        )
        return _valid_bound(bound)

    def evaluate(
        selected_particle: np.ndarray,
        time_s: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        charge: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, position
        row = np.searchsorted(particle_index, selected_particle)
        additive = np.empty_like(velocity)
        additive[:, 0] = base_acceleration_m_s2[row, 0] + coupling_s_inv[row] * velocity[:, 1]
        additive[:, 1] = -base_acceleration_m_s2[row, 1] - coupling_s_inv[row] * velocity[:, 0]
        count = selected_particle.size
        return RelaxationEvaluation(
            rate_s_inv[row],
            target_velocity_m_s[row],
            additive,
            np.zeros_like(charge),
            np.zeros_like(charge),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    enclosure = enclose_exponential_midpoint_path(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        linear_drag_rate_upper_s_inv=rate_s_inv,
        target_velocity_abs_upper_m_s=np.abs(target_velocity_m_s),
        additive_acceleration_abs_bounder=additive_bounder,
    )
    assert len(bound_inputs) == 2
    np.testing.assert_array_equal(bound_inputs[0], np.abs(velocity_m_s))
    assert bool((bound_inputs[1] >= bound_inputs[0]).all())
    assert bool((bound_inputs[1] > bound_inputs[0]).any())

    proposal = exponential_midpoint_step(
        particle_index,
        start_time_s,
        target_time_s,
        position_m,
        velocity_m_s,
        np.zeros(2, dtype="<f8"),
        evaluator=evaluate,
        path_enclosure=enclosure,
    )
    for row in range(2):
        duration_s = target_time_s[row] - start_time_s[row]
        for elapsed_s in np.linspace(0.0, duration_s, 41):
            sample = proposal.state_at_rows(
                start_time_s[row] + elapsed_s,
                np.asarray([row], dtype="<i8"),
            )
            assert bool((sample.position_m[0] >= enclosure.position_lower_m[row]).all())
            assert bool((sample.position_m[0] <= enclosure.position_upper_m[row]).all())
            assert bool((sample.velocity_m_s[0] >= enclosure.velocity_lower_m_s[row]).all())
            assert bool((sample.velocity_m_s[0] <= enclosure.velocity_upper_m_s[row]).all())


def test_c03_exponential_enclosure_stays_inside_its_regular_support_box() -> None:
    enclosure = enclose_exponential_midpoint_path(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([0.75], dtype="<f8"),
        np.asarray([[-0.25, -0.1]], dtype="<f8"),
        np.asarray([[0.6, -0.5]], dtype="<f8"),
        linear_drag_rate_upper_s_inv=np.asarray([4.0], dtype="<f8"),
        target_velocity_abs_upper_m_s=np.asarray([[0.2, 0.3]], dtype="<f8"),
        additive_acceleration_abs_bounder=_table_acceleration_bounder(
            np.asarray([0], dtype="<i8"),
            np.asarray([[0.8, 0.4]], dtype="<f8"),
        ),
    )

    assert bool((enclosure.position_lower_m >= -1.0).all())
    assert bool((enclosure.position_upper_m <= 1.0).all())
    np.testing.assert_allclose(enclosure.position_lower_m, [[-0.925, -0.5875]], atol=2.0e-15)
    np.testing.assert_allclose(enclosure.position_upper_m, [[0.425, 0.3875]], atol=2.0e-15)


def test_exponential_enclosure_preserves_strict_departure_velocity_under_drag() -> None:
    enclosure = enclose_exponential_midpoint_path(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([0.5], dtype="<f8"),
        np.asarray([[0.0, 0.5]], dtype="<f8"),
        np.asarray([[1.0, 0.0]], dtype="<f8"),
        linear_drag_rate_upper_s_inv=np.asarray([4.0], dtype="<f8"),
        target_velocity_abs_upper_m_s=np.zeros((1, 2), dtype="<f8"),
        additive_acceleration_abs_bounder=_table_acceleration_bounder(
            np.asarray([0], dtype="<i8"),
            np.zeros((1, 2), dtype="<f8"),
        ),
    )

    assert enclosure.velocity_lower_m_s[0, 0] > 0.0
    assert enclosure.velocity_upper_m_s[0, 0] >= 1.0


def test_exponential_midpoint_charge_is_affine_exact_stiff_and_monotone() -> None:
    root_charge = 3.0
    cases = (
        (0.0, 0.5),
        (-float(np.nextafter(0.0, np.inf)), 0.5),
        (-1.0e-12, 1.0),
        (-1.0, 1.0),
        (-20.0, 1.0),
        (-1.0e3, 1.0),
    )

    def multiplier(derivative_s_inv: float, elapsed_s: float) -> float:
        argument = derivative_s_inv * elapsed_s
        if derivative_s_inv == 0.0 or argument == 0.0:
            return elapsed_s
        return math.expm1(argument) / derivative_s_inv

    for derivative_s_inv, duration_s in cases:
        rate_scale = 1.0 if abs(derivative_s_inv) < 1.0 else abs(derivative_s_inv)
        for direction in (-1.0, 1.0):
            affine_rate_number_s = direction * 2.0 * rate_scale

            def evaluate(
                particle_index: np.ndarray,
                time_s: np.ndarray,
                position_m: np.ndarray,
                velocity_m_s: np.ndarray,
                charge_number: np.ndarray,
                *,
                local_affine_rate_number_s: float = affine_rate_number_s,
                local_derivative_s_inv: float = derivative_s_inv,
            ) -> RelaxationEvaluation:
                del time_s, position_m, velocity_m_s
                count = particle_index.size
                charge_rate = local_affine_rate_number_s + local_derivative_s_inv * (
                    charge_number - root_charge
                )
                return RelaxationEvaluation(
                    np.zeros(count, dtype="<f8"),
                    np.zeros((count, 2), dtype="<f8"),
                    np.zeros((count, 2), dtype="<f8"),
                    charge_rate,
                    np.full(count, local_derivative_s_inv, dtype="<f8"),
                    np.ones(count, dtype=np.bool_),
                    np.ones(count, dtype=np.bool_),
                    np.zeros(count, dtype=np.uint8),
                )

            proposal = exponential_midpoint_step(
                np.asarray([0], dtype="<i8"),
                np.asarray([0.0], dtype="<f8"),
                np.asarray([duration_s], dtype="<f8"),
                np.zeros((1, 2), dtype="<f8"),
                np.zeros((1, 2), dtype="<f8"),
                np.asarray([root_charge], dtype="<f8"),
                evaluator=evaluate,
            )
            sample_times = duration_s * np.asarray([0.0, 0.1, 0.5, 0.9, 1.0])
            actual = np.asarray(
                [
                    proposal.state_at_rows(
                        float(time_s), np.asarray([0], dtype="<i8")
                    ).charge_number[0]
                    for time_s in sample_times
                ]
            )
            expected = np.asarray(
                [
                    root_charge + multiplier(derivative_s_inv, float(time_s)) * affine_rate_number_s
                    for time_s in sample_times
                ]
            )
            absolute_tolerance = (
                32.0
                * np.finfo(np.float64).eps
                * max(
                    1.0,
                    float(np.max(np.abs(expected))),
                )
            )

            np.testing.assert_allclose(
                actual,
                expected,
                rtol=4.0e-15,
                atol=absolute_tolerance,
            )
            np.testing.assert_array_equal(actual[-1:], proposal.end_charge())
            assert direction * (actual[-1] - actual[0]) > 0.0
            assert bool((direction * np.diff(actual) >= -absolute_tolerance).all())
            assert float(np.min(actual)) >= min(actual[0], actual[-1]) - absolute_tolerance
            assert float(np.max(actual)) <= max(actual[0], actual[-1]) + absolute_tolerance


def test_exponential_midpoint_couples_nonlinear_charge_and_motion_at_second_order() -> None:
    relaxation_rate_s_inv = 4.0
    nonlinear_rate_s_inv = 0.75

    def exact_charge(time_s: float) -> float:
        return 1.0 + 0.2 * time_s + 0.1 * time_s**2

    def exact_state(time_s: float) -> np.ndarray:
        return np.asarray(
            [
                0.5 * time_s**2 + time_s**3 / 30.0 + time_s**4 / 120.0,
                time_s + 0.1 * time_s**2 + time_s**3 / 30.0,
                exact_charge(time_s),
            ]
        )

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del position_m, velocity_m_s
        count = particle_index.size
        manufactured_charge = 1.0 + 0.2 * time_s + 0.1 * time_s**2
        charge_error = charge_number - manufactured_charge
        charge_rate = (
            0.2
            + 0.2 * time_s
            - relaxation_rate_s_inv * charge_error
            - nonlinear_rate_s_inv * charge_error**3
        )
        charge_derivative = -relaxation_rate_s_inv - 3.0 * nonlinear_rate_s_inv * charge_error**2
        additive = np.zeros((count, 2), dtype="<f8")
        additive[:, 0] = charge_number
        return RelaxationEvaluation(
            np.zeros(count, dtype="<f8"),
            np.zeros((count, 2), dtype="<f8"),
            additive,
            charge_rate,
            charge_derivative,
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    expected = exact_state(1.0)
    errors: list[np.ndarray] = []
    for step_count in (8, 16, 32):
        step_s = 1.0 / step_count
        position = np.zeros((1, 2), dtype="<f8")
        velocity = np.zeros((1, 2), dtype="<f8")
        charge = np.ones(1, dtype="<f8")
        for step_index in range(step_count):
            proposal = exponential_midpoint_step(
                np.asarray([0], dtype="<i8"),
                np.asarray([step_index * step_s], dtype="<f8"),
                np.asarray([(step_index + 1) * step_s], dtype="<f8"),
                position,
                velocity,
                charge,
                evaluator=evaluate,
            )
            position = proposal.end_position()
            velocity = proposal.end_velocity()
            charge = proposal.end_charge()
        actual = np.asarray([position[0, 0], velocity[0, 0], charge[0]])
        errors.append(np.abs(actual - expected))

    error_table = np.stack(errors)
    assert bool((error_table[:-1] > error_table[1:]).all())
    observed_orders = np.log2(error_table[:-1] / error_table[1:])
    assert float(np.min(observed_orders)) > 1.8


def test_exponential_midpoint_includes_endpoint_in_support_verdict() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> RelaxationEvaluation:
        del time_s, velocity_m_s
        count = particle_index.size
        additive = np.zeros((count, 2), dtype="<f8")
        additive[:, 0] = 2.0
        return RelaxationEvaluation(
            np.zeros(count, dtype="<f8"),
            np.zeros((count, 2), dtype="<f8"),
            additive,
            np.zeros_like(charge_number),
            np.zeros_like(charge_number),
            position_m[:, 0] < 0.75,
            np.ones(count, dtype=np.bool_),
            np.zeros(count, dtype=np.uint8),
        )

    proposal = exponential_midpoint_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([1.0], dtype="<f8"),
        np.asarray([[0.0, 0.0]], dtype="<f8"),
        np.asarray([[0.0, 0.0]], dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        evaluator=evaluate,
    )

    np.testing.assert_allclose(proposal.end_position_m, [[1.0, 0.0]], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(proposal.support_inside, [False])


def test_rk4_dense_path_preserves_endpoint_and_vector_sampling_does_not_reintegrate() -> None:
    calls = 0

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        nonlocal calls
        calls += 1
        acceleration = np.column_stack(
            (
                0.3 * position_m[:, 0] - 0.2 * velocity_m_s[:, 0] + 0.1 * time_s,
                -0.4 * position_m[:, 1] + 0.15 * velocity_m_s[:, 1],
            )
        )
        return DynamicsEvaluation(
            acceleration,
            0.25 * charge_number - 0.1,
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([4, 9], dtype="<i8"),
        np.asarray([0.1, 0.2], dtype="<f8"),
        np.asarray([0.7, 0.9], dtype="<f8"),
        np.asarray([[0.2, -0.3], [0.8, 0.4]], dtype="<f8"),
        np.asarray([[0.7, 0.1], [-0.2, 0.6]], dtype="<f8"),
        np.asarray([1.5, -0.5], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    endpoint_position = proposal.end_position_m.copy()
    endpoint_velocity = proposal.end_velocity_m_s.copy()
    endpoint_charge = proposal.end_charge_number.copy()
    assert calls == 5

    sample = proposal.rk4_dense_state_at_rows(
        np.asarray([1, 0], dtype="<i8"),
        np.asarray([0.9, 0.7], dtype="<f8"),
    )

    assert RK4_DENSE_PATH_REVISION == "rk4_position_hermite_state_extension_v3"
    assert proposal.path_kind == "rk4_dense"
    assert calls == 5
    np.testing.assert_array_equal(sample.position_m, endpoint_position[[1, 0]])
    np.testing.assert_array_equal(sample.velocity_m_s, endpoint_velocity[[1, 0]])
    np.testing.assert_array_equal(sample.charge_number, endpoint_charge[[1, 0]])


def test_rk4_dense_position_is_the_endpoint_hermite_event_path() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, velocity_m_s, charge_number
        acceleration = np.column_stack((-position_m[:, 0], -2.0 * position_m[:, 1]))
        return DynamicsEvaluation(
            acceleration,
            np.zeros(particle_index.size, dtype="<f8"),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    start_position = np.asarray([[1.0, -0.4]], dtype="<f8")
    start_velocity = np.asarray([[0.0, 0.7]], dtype="<f8")
    proposal = rk4_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([1.0], dtype="<f8"),
        start_position,
        start_velocity,
        np.asarray([0.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    theta = np.linspace(0.0, 1.0, 17)
    sample = proposal.rk4_dense_state_at_rows(
        np.zeros(theta.size, dtype="<i8"),
        theta,
    )
    theta_column = theta[:, None]
    expected_position = (
        (2.0 * theta_column**3 - 3.0 * theta_column**2 + 1.0) * start_position
        + (theta_column**3 - 2.0 * theta_column**2 + theta_column) * start_velocity
        + (-2.0 * theta_column**3 + 3.0 * theta_column**2) * proposal.end_position_m
        + (theta_column**3 - theta_column**2) * proposal.end_velocity_m_s
    )
    np.testing.assert_allclose(sample.position_m, expected_position, rtol=0.0, atol=4.0e-16)

    control_theta = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    control_sample = proposal.rk4_dense_state_at_rows(
        np.zeros(4, dtype="<i8"),
        control_theta,
    ).position_m
    bernstein = np.column_stack(
        (
            (1.0 - control_theta) ** 3,
            3.0 * (1.0 - control_theta) ** 2 * control_theta,
            3.0 * (1.0 - control_theta) * control_theta**2,
            control_theta**3,
        )
    )
    reconstructed_controls = np.linalg.solve(bernstein, control_sample)
    expected_controls = np.stack(
        (
            start_position[0],
            start_position[0] + start_velocity[0] / 3.0,
            proposal.end_position_m[0] - proposal.end_velocity_m_s[0] / 3.0,
            proposal.end_position_m[0],
        )
    )
    np.testing.assert_allclose(
        reconstructed_controls,
        expected_controls,
        rtol=0.0,
        atol=8.0e-16,
    )


def test_rk4_dense_subinterval_contains_state_path_derivative_and_shrinks() -> None:
    acceleration = np.asarray([[1.25, -0.5]], dtype="<f8")
    charge_rate = np.asarray([0.75], dtype="<f8")

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s, charge_number
        return DynamicsEvaluation(
            np.repeat(acceleration, particle_index.size, axis=0),
            np.repeat(charge_rate, particle_index.size),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([3], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([1.0], dtype="<f8"),
        np.asarray([[0.2, -0.4]], dtype="<f8"),
        np.asarray([[1.1, 0.3]], dtype="<f8"),
        np.asarray([-0.5], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    rows = np.asarray([0], dtype="<i8")
    root = proposal.rk4_dense_subinterval(rows, np.asarray([0.0]), np.asarray([1.0]))
    late = proposal.rk4_dense_subinterval(rows, np.asarray([0.85]), np.asarray([1.0]))
    assert late.relative_position_control_lower_m.shape == (1, 4, 2)
    assert late.relative_position_control_upper_m.shape == (1, 4, 2)
    assert bool(
        (late.relative_position_control_lower_m <= late.relative_position_control_upper_m).all()
    )
    relative_lower = np.min(late.relative_position_control_lower_m, axis=1)
    relative_upper = np.max(late.relative_position_control_upper_m, axis=1)

    for time_s in np.linspace(0.85, 1.0, 101):
        sample = proposal.rk4_dense_state_at_rows(rows, np.asarray([time_s]))
        assert bool((sample.position_m >= late.position_lower_m).all())
        assert bool((sample.position_m <= late.position_upper_m).all())
        assert bool((sample.velocity_m_s >= late.velocity_lower_m_s).all())
        assert bool((sample.velocity_m_s <= late.velocity_upper_m_s).all())
        assert bool((sample.charge_number >= late.charge_lower_number).all())
        assert bool((sample.charge_number <= late.charge_upper_number).all())
        relative_position = sample.position_m - late.position_control_origin_m
        assert bool((relative_position >= relative_lower).all())
        assert bool((relative_position <= relative_upper).all())
        path_derivative = np.asarray([[1.1, 0.3]]) + time_s * acceleration
        assert bool((path_derivative >= late.velocity_lower_m_s).all())
        assert bool((path_derivative <= late.velocity_upper_m_s).all())

    root_width = root.position_upper_m - root.position_lower_m
    late_width = late.position_upper_m - late.position_lower_m
    assert bool((late_width < root_width).all())


def test_rk4_dense_velocity_enclosure_is_invariant_to_exact_origin_shift() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s, charge_number
        return DynamicsEvaluation(
            np.zeros((particle_index.size, 2), dtype="<f8"),
            np.zeros(particle_index.size, dtype="<f8"),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    origin_shift = float(2**40)
    proposal = rk4_step(
        np.asarray([0, 1], dtype="<i8"),
        np.zeros(2, dtype="<f8"),
        np.full(2, 0.75, dtype="<f8"),
        np.asarray([[0.0, 0.0], [origin_shift, -origin_shift]], dtype="<f8"),
        np.asarray([[4.0, -8.0], [4.0, -8.0]], dtype="<f8"),
        np.zeros(2, dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    enclosure = proposal.rk4_dense_subinterval(
        np.asarray([0, 1], dtype="<i8"),
        np.zeros(2, dtype="<f8"),
        np.full(2, 0.75, dtype="<f8"),
    )

    np.testing.assert_array_equal(enclosure.numerical_status, [NUMERICAL_STATUS_OK] * 2)
    np.testing.assert_array_equal(enclosure.velocity_lower_m_s[0], enclosure.velocity_lower_m_s[1])
    np.testing.assert_array_equal(enclosure.velocity_upper_m_s[0], enclosure.velocity_upper_m_s[1])
    chord_deviation = proposal.rk4_dense_chord_deviation(
        np.asarray([0, 1], dtype="<i8"),
        np.zeros(2, dtype="<f8"),
        np.full(2, 0.75, dtype="<f8"),
    )
    # The relative physical-curvature calculation is translation invariant,
    # while the public bound must also cover evaluation of the returned world
    # coordinates and their chord.  That unavoidable term remains eight ULPs
    # here rather than scaling by a broad absolute-coordinate padding.
    coordinate_roundoff = chord_deviation[1] - chord_deviation[0]
    assert bool((coordinate_roundoff >= 0.0).all())
    assert bool((coordinate_roundoff <= 9.0 * np.spacing(origin_shift)).all())


def test_rk4_dense_position_enclosure_does_not_scale_with_coordinate_origin() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s, charge_number
        return DynamicsEvaluation(
            np.zeros((particle_index.size, 2), dtype="<f8"),
            np.zeros(particle_index.size, dtype="<f8"),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    origin_shift = float(2**20)
    proposal = rk4_step(
        np.asarray([0, 1], dtype="<i8"),
        np.zeros(2, dtype="<f8"),
        np.full(2, 0.75, dtype="<f8"),
        np.asarray([[0.0, 0.0], [origin_shift, -origin_shift]], dtype="<f8"),
        np.asarray([[4.0, -8.0], [4.0, -8.0]], dtype="<f8"),
        np.zeros(2, dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    enclosure = proposal.rk4_dense_subinterval(
        np.asarray([0, 1], dtype="<i8"),
        np.zeros(2, dtype="<f8"),
        np.full(2, 0.75, dtype="<f8"),
    )

    shifted_lower = enclosure.position_lower_m[1] - np.asarray([origin_shift, -origin_shift])
    shifted_upper = enclosure.position_upper_m[1] - np.asarray([origin_shift, -origin_shift])
    origin_ulp = np.spacing(origin_shift)
    np.testing.assert_allclose(
        shifted_lower,
        enclosure.position_lower_m[0],
        rtol=0.0,
        atol=2.0 * origin_ulp,
    )
    np.testing.assert_allclose(
        shifted_upper,
        enclosure.position_upper_m[0],
        rtol=0.0,
        atol=2.0 * origin_ulp,
    )
    for time_s in np.linspace(0.0, 0.75, 9):
        sample = proposal.rk4_dense_state_at_rows(
            np.asarray([0, 1], dtype="<i8"),
            np.full(2, time_s, dtype="<f8"),
        )
        assert bool((sample.position_m >= enclosure.position_lower_m).all())
        assert bool((sample.position_m <= enclosure.position_upper_m).all())


def test_rk4_dense_ulp_scale_position_matches_enclosure_and_chord_contracts() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s, charge_number
        return DynamicsEvaluation(
            np.zeros((particle_index.size, 2), dtype="<f8"),
            np.zeros(particle_index.size, dtype="<f8"),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    origins = np.asarray([1.0, float(2**20)], dtype="<f8")
    velocities = np.asarray(
        [
            -4.0 * np.finfo(np.float64).eps,
            -2.0 * np.spacing(origins[1]),
        ],
        dtype="<f8",
    )
    start_position = np.column_stack((origins, np.zeros(2, dtype="<f8")))
    proposal = rk4_step(
        np.asarray([0, 1], dtype="<i8"),
        np.zeros(2, dtype="<f8"),
        np.ones(2, dtype="<f8"),
        start_position,
        np.column_stack((velocities, np.zeros(2, dtype="<f8"))),
        np.zeros(2, dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    rows = np.asarray([0, 1], dtype="<i8")

    np.testing.assert_array_equal(
        proposal.rk4_dense_state_at_rows(rows, np.zeros(2, dtype="<f8")).position_m,
        start_position,
    )
    np.testing.assert_array_equal(
        proposal.rk4_dense_state_at_rows(rows, np.ones(2, dtype="<f8")).position_m,
        proposal.end_position_m,
    )

    interval_start = np.full(2, 51.0 / 256.0, dtype="<f8")
    interval_target = np.full(2, 52.0 / 256.0, dtype="<f8")
    sample_time = np.full(2, 51.5 / 256.0, dtype="<f8")
    enclosure = proposal.rk4_dense_subinterval(rows, interval_start, interval_target)
    narrow_sample = proposal.rk4_dense_state_at_rows(rows, sample_time).position_m
    assert bool((narrow_sample >= enclosure.position_lower_m).all())
    assert bool((narrow_sample <= enclosure.position_upper_m).all())

    chord_start = np.zeros(2, dtype="<f8")
    chord_target = np.full(2, 0.5, dtype="<f8")
    chord_bound = proposal.rk4_dense_chord_deviation(rows, chord_start, chord_target)
    endpoints = proposal.rk4_dense_state_at_rows(
        np.asarray([0, 1, 0, 1], dtype="<i8"),
        np.asarray([0.0, 0.0, 0.5, 0.5], dtype="<f8"),
    ).position_m
    chord_start_position = endpoints[:2]
    chord_target_position = endpoints[2:]
    for fraction in np.linspace(0.0, 1.0, 129):
        position = proposal.rk4_dense_state_at_rows(
            rows,
            np.full(2, 0.5 * fraction, dtype="<f8"),
        ).position_m
        chord = (1.0 - fraction) * chord_start_position + fraction * chord_target_position
        assert bool((np.abs(position - chord) <= chord_bound).all())


def test_rk4_dense_stationary_tiny_step_has_finite_zero_scale_derivative() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s, charge_number
        return DynamicsEvaluation(
            np.zeros((particle_index.size, 2), dtype="<f8"),
            np.zeros(particle_index.size, dtype="<f8"),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    target = np.asarray([1.0e-300], dtype="<f8")
    proposal = rk4_step(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        target,
        np.asarray([[1.0e200, -1.0e200]], dtype="<f8"),
        np.zeros((1, 2), dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    enclosure = proposal.rk4_dense_subinterval(
        np.asarray([0], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        target,
    )

    np.testing.assert_array_equal(enclosure.numerical_status, [NUMERICAL_STATUS_OK])
    assert bool(np.isfinite(enclosure.velocity_lower_m_s).all())
    assert bool(np.isfinite(enclosure.velocity_upper_m_s).all())
    assert float(np.max(np.abs(enclosure.velocity_lower_m_s))) < 1.0e-300
    assert float(np.max(np.abs(enclosure.velocity_upper_m_s))) < 1.0e-300


def test_rk4_dense_charge_enclosure_preserves_root_control_sign() -> None:
    rates = np.asarray([-3.0, 3.0, 2.0], dtype="<f8")

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s, charge_number
        return DynamicsEvaluation(
            np.zeros((particle_index.size, 2), dtype="<f8"),
            rates[particle_index],
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([0, 1, 2], dtype="<i8"),
        np.zeros(3, dtype="<f8"),
        np.ones(3, dtype="<f8"),
        np.zeros((3, 2), dtype="<f8"),
        np.zeros((3, 2), dtype="<f8"),
        np.asarray([0.0, 0.0, -1.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    enclosure = proposal.rk4_dense_subinterval(
        np.arange(3, dtype="<i8"),
        np.zeros(3, dtype="<f8"),
        np.ones(3, dtype="<f8"),
    )

    assert enclosure.charge_upper_number[0] == 0.0
    assert enclosure.charge_lower_number[1] == 0.0
    assert enclosure.charge_lower_number[2] < 0.0 < enclosure.charge_upper_number[2]
    for time_s in np.linspace(0.0, 1.0, 17):
        sample = proposal.rk4_dense_state_at_rows(
            np.arange(3, dtype="<i8"),
            np.full(3, time_s, dtype="<f8"),
        )
        assert bool((sample.charge_number >= enclosure.charge_lower_number).all())
        assert bool((sample.charge_number <= enclosure.charge_upper_number).all())


def test_rk4_dense_random_subintervals_enclose_nonkinematic_path_derivative() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        acceleration = np.column_stack(
            (
                -3.0 * position_m[:, 0] + 0.4 * velocity_m_s[:, 1] + 0.2 * time_s,
                1.7 * position_m[:, 0] - 0.3 * velocity_m_s[:, 1],
            )
        )
        return DynamicsEvaluation(
            acceleration,
            -0.6 * charge_number + 0.2,
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([8], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([0.8], dtype="<f8"),
        np.asarray([[0.3, -0.2]], dtype="<f8"),
        np.asarray([[1.1, 0.7]], dtype="<f8"),
        np.asarray([2.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    control_parameter = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    control_sample = proposal.rk4_dense_state_at_rows(
        np.zeros(4, dtype="<i8"),
        0.8 * control_parameter,
    )
    vandermonde = np.column_stack(
        (
            np.ones(4),
            control_parameter,
            control_parameter**2,
            control_parameter**3,
        )
    )
    position_coefficients = np.linalg.solve(vandermonde, control_sample.position_m)
    generator = np.random.default_rng(1701)
    derivative_velocity_gap = 0.0
    for _ in range(30):
        parameters = np.sort(generator.uniform(0.0, 1.0, size=2))
        enclosure = proposal.rk4_dense_subinterval(
            np.asarray([0], dtype="<i8"),
            np.asarray([0.8 * parameters[0]]),
            np.asarray([0.8 * parameters[1]]),
        )
        for parameter in np.linspace(parameters[0], parameters[1], 9):
            sample = proposal.rk4_dense_state_at_rows(
                np.asarray([0], dtype="<i8"),
                np.asarray([0.8 * parameter]),
            )
            path_derivative = (
                position_coefficients[1]
                + 2.0 * parameter * position_coefficients[2]
                + 3.0 * parameter * parameter * position_coefficients[3]
            ) / 0.8
            derivative_velocity_gap = max(
                derivative_velocity_gap,
                float(np.max(np.abs(path_derivative - sample.velocity_m_s[0]))),
            )
            assert bool((sample.position_m >= enclosure.position_lower_m).all())
            assert bool((sample.position_m <= enclosure.position_upper_m).all())
            assert bool((sample.velocity_m_s >= enclosure.velocity_lower_m_s).all())
            assert bool((sample.velocity_m_s <= enclosure.velocity_upper_m_s).all())
            assert bool((sample.charge_number >= enclosure.charge_lower_number).all())
            assert bool((sample.charge_number <= enclosure.charge_upper_number).all())
            assert bool((path_derivative >= enclosure.velocity_lower_m_s[0]).all())
            assert bool((path_derivative <= enclosure.velocity_upper_m_s[0]).all())
    assert derivative_velocity_gap > 1.0e-3


def test_rk4_dense_chord_deviation_contains_random_subinterval_paths() -> None:
    acceleration = np.asarray([[1.7, -0.9]], dtype="<f8")

    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s, charge_number
        return DynamicsEvaluation(
            np.repeat(acceleration, particle_index.size, axis=0),
            np.zeros(particle_index.size, dtype="<f8"),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    proposal = rk4_step(
        np.asarray([5], dtype="<i8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([1.0], dtype="<f8"),
        np.asarray([[0.2, -0.1]], dtype="<f8"),
        np.asarray([[0.8, 0.4]], dtype="<f8"),
        np.asarray([-3.0], dtype="<f8"),
        requires_stage_evaluation=True,
        evaluator=evaluate,
    )
    rows = np.asarray([0], dtype="<i8")
    generator = np.random.default_rng(5019)
    for _ in range(40):
        start, target = np.sort(generator.uniform(0.0, 1.0, size=2))
        bound = proposal.rk4_dense_chord_deviation(
            rows,
            np.asarray([start]),
            np.asarray([target]),
        )[0]
        endpoint = proposal.rk4_dense_state_at_rows(
            np.asarray([0, 0], dtype="<i8"),
            np.asarray([start, target]),
        ).position_m
        for fraction in np.linspace(0.0, 1.0, 33):
            time_s = start + fraction * (target - start)
            position = proposal.rk4_dense_state_at_rows(
                rows,
                np.asarray([time_s]),
            ).position_m[0]
            chord = (1.0 - fraction) * endpoint[0] + fraction * endpoint[1]
            assert bool((np.abs(position - chord) <= bound).all())

    wide = proposal.rk4_dense_chord_deviation(
        rows,
        np.asarray([0.2]),
        np.asarray([0.6]),
    )[0]
    narrow = proposal.rk4_dense_chord_deviation(
        rows,
        np.asarray([0.2]),
        np.asarray([0.4]),
    )[0]
    assert bool((narrow < 0.3 * wide).all())


def test_rk4_dense_charge_has_cubic_local_order() -> None:
    def evaluate(
        particle_index: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        charge_number: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s
        return DynamicsEvaluation(
            np.zeros((particle_index.size, 2), dtype="<f8"),
            charge_number.copy(),
            np.ones(particle_index.size, dtype=np.bool_),
            np.ones(particle_index.size, dtype=np.bool_),
            np.zeros(particle_index.size, dtype=np.uint8),
        )

    errors = []
    theta = 0.37
    for step_s in (0.8, 0.4, 0.2, 0.1, 0.05):
        proposal = rk4_step(
            np.asarray([0], dtype="<i8"),
            np.asarray([0.0], dtype="<f8"),
            np.asarray([step_s], dtype="<f8"),
            np.zeros((1, 2), dtype="<f8"),
            np.zeros((1, 2), dtype="<f8"),
            np.asarray([1.0], dtype="<f8"),
            requires_stage_evaluation=True,
            evaluator=evaluate,
        )
        sample = proposal.rk4_dense_state_at_rows(
            np.asarray([0], dtype="<i8"),
            np.asarray([theta * step_s]),
        )
        errors.append(abs(float(sample.charge_number[0]) - math.exp(theta * step_s)))

    observed = [math.log(errors[index] / errors[index + 1], 2.0) for index in (2, 3)]
    assert min(observed) > 3.8
