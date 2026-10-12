from __future__ import annotations

import numpy as np

from chamber_particles.integrators import DynamicsEvaluation, StepProposal, rk4_step
from chamber_particles.numerical_status import (
    NUMERICAL_STATUS_OK,
    PHYSICS_NUMERICAL_FAILURE,
)


def test_rk4_dense_repeated_selection_preserves_root_and_changed_status() -> None:
    """Independent enclosure results must preserve the original physical path."""

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

    origins = np.asarray([1.0, float(2**20), 0.2], dtype="<f8")
    count = origins.size
    velocities = np.asarray(
        [-4.0 * np.finfo(np.float64).eps, -2.0 * np.spacing(origins[1]), 1.1], dtype="<f8"
    )
    position = np.column_stack((origins, np.zeros(count, dtype="<f8")))
    velocity = np.column_stack((velocities, np.zeros(count, dtype="<f8")))
    charge = np.asarray([-0.5, 1.5, 0.25], dtype="<f8")

    def build_proposal() -> StepProposal:
        return rk4_step(
            np.arange(count, dtype="<i8"),
            np.zeros(count, dtype="<f8"),
            np.ones(count, dtype="<f8"),
            position,
            velocity,
            charge,
            requires_stage_evaluation=True,
            evaluator=evaluate,
        )

    proposal = build_proposal()
    reference = build_proposal()
    rows = np.asarray([1, 0, 2, 1], dtype="<i8")
    starts = np.zeros(rows.size, dtype="<f8")
    targets = np.ones(rows.size, dtype="<f8")
    expected = reference.rk4_dense_subinterval(rows, starts, targets)
    first = proposal.rk4_dense_subinterval(rows, starts, targets)
    second = proposal.rk4_dense_subinterval(rows, starts, targets)

    # The repeated row must have independent writable output storage, and a
    # caller's changes to one result must not affect another or a later result.
    first.position_lower_m[0] = -123.0
    np.testing.assert_array_equal(first.position_lower_m[3], expected.position_lower_m[3])
    first.position_upper_m[:] = 123.0
    first.velocity_lower_m_s[:] = -123.0
    first.velocity_upper_m_s[:] = 123.0
    first.charge_lower_number[:] = -123.0
    first.charge_upper_number[:] = 123.0
    first.position_control_origin_m[:] = -123.0
    first.relative_position_control_lower_m[:] = -123.0
    first.relative_position_control_upper_m[:] = 123.0
    first.numerical_status[:] = PHYSICS_NUMERICAL_FAILURE
    later = proposal.rk4_dense_subinterval(rows, starts, targets)
    for result in (second, later):
        for actual, wanted in (
            (result.position_lower_m, expected.position_lower_m),
            (result.position_upper_m, expected.position_upper_m),
            (result.velocity_lower_m_s, expected.velocity_lower_m_s),
            (result.velocity_upper_m_s, expected.velocity_upper_m_s),
            (result.charge_lower_number, expected.charge_lower_number),
            (result.charge_upper_number, expected.charge_upper_number),
            (result.position_control_origin_m, expected.position_control_origin_m),
            (
                result.relative_position_control_lower_m,
                expected.relative_position_control_lower_m,
            ),
            (
                result.relative_position_control_upper_m,
                expected.relative_position_control_upper_m,
            ),
            (result.numerical_status, expected.numerical_status),
        ):
            np.testing.assert_array_equal(actual.view(np.uint8), wanted.view(np.uint8))

    # Reuse the existing ULP-scale dyadic enclosure verification recipe.
    interval_start = np.full(rows.size, 51.0 / 256.0, dtype="<f8")
    interval_target = np.full(rows.size, 52.0 / 256.0, dtype="<f8")
    interval = proposal.rk4_dense_subinterval(rows, interval_start, interval_target)
    assert interval.position_lower_m[2, 0] > position[2, 0]
    assert interval.position_upper_m[2, 0] < proposal.end_position_m[2, 0]
    for fraction in (0.0, 0.5, 1.0):
        times = interval_start + fraction * (interval_target - interval_start)
        sample = proposal.rk4_dense_state_at_rows(rows, times)
        assert bool((sample.position_m >= interval.position_lower_m).all())
        assert bool((sample.position_m <= interval.position_upper_m).all())
        assert bool((sample.velocity_m_s >= interval.velocity_lower_m_s).all())
        assert bool((sample.velocity_m_s <= interval.velocity_upper_m_s).all())
        assert bool((sample.charge_number >= interval.charge_lower_number).all())
        assert bool((sample.charge_number <= interval.charge_upper_number).all())

    # A changed inherited failure preserves the existing failed-root response.
    proposal.numerical_status[1] = PHYSICS_NUMERICAL_FAILURE
    failed = proposal.rk4_dense_subinterval(rows, starts, targets)
    np.testing.assert_array_equal(
        failed.numerical_status,
        [
            PHYSICS_NUMERICAL_FAILURE,
            NUMERICAL_STATUS_OK,
            NUMERICAL_STATUS_OK,
            PHYSICS_NUMERICAL_FAILURE,
        ],
    )
    failed_rows = np.asarray([0, 3], dtype="<i8")
    for bounds, root in (
        (failed.position_lower_m, position),
        (failed.position_upper_m, position),
        (failed.velocity_lower_m_s, velocity),
        (failed.velocity_upper_m_s, velocity),
        (failed.charge_lower_number, charge),
        (failed.charge_upper_number, charge),
    ):
        np.testing.assert_array_equal(bounds[failed_rows], root[rows[failed_rows]])
    np.testing.assert_array_equal(failed.position_lower_m[1], expected.position_lower_m[1])
    np.testing.assert_array_equal(failed.position_control_origin_m[failed_rows], 0.0)
    np.testing.assert_array_equal(failed.relative_position_control_lower_m[failed_rows], 0.0)
    np.testing.assert_array_equal(failed.relative_position_control_upper_m[failed_rows], 0.0)
