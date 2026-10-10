from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chamber_particles import load_case
from chamber_particles.case_format import BoundaryData, GeometryData
from chamber_particles.events import (
    CURVED_STATUS_AXIS,
    CURVED_STATUS_CLEAR,
    CURVED_STATUS_SPLIT,
    CURVED_STATUS_WALL,
    EXACT_DEPARTURE_FINITE_CONTACT_SET,
    EXACT_FAILURE_INDETERMINATE_EVENT,
    EXACT_PATH_LINEAR,
    EXACT_PATH_QUADRATIC,
    EXACT_STATUS_AXIS,
    EXACT_STATUS_CLEAR,
    EXACT_STATUS_FAILURE,
    EXACT_STATUS_WALL,
    SURFACE_ACTION_CURVED_DEPARTURE,
    SURFACE_ACTION_EXACT_DEPARTURE,
    SURFACE_ACTION_RESOLVED,
    SURFACE_ACTION_RESPONSE_ACCELERATION,
    SURFACE_ACTION_RESPONSE_VELOCITY,
    SURFACE_STATE_DEPARTURE,
    SURFACE_STATE_PENDING,
    SURFACE_STATE_RESOLVED,
    SURFACE_STATUS_INDETERMINATE_DIRECTION,
    SURFACE_STATUS_OK,
    CurvedCandidateCapacityError,
    EventLocationError,
    ExactCandidateCapacityError,
    classify_event_point,
    classify_surface_release_batch,
    count_curved_event_candidates,
    count_exact_event_candidates,
    inspect_rk4_axis_piece,
    inspect_rk4_piece,
    locate_ballistic_first_hit,
    locate_constant_acceleration_first_hit,
    locate_curved_first_event_batch,
    locate_exact_first_event_batch,
    preclassify_rk4_piece_batch,
    resolve_event_budget,
)
from chamber_particles.geometry import prepare_geometry
from chamber_particles.integrators import DynamicsEvaluation, enclose_rk4_path, rk4_step
from tests.verification.microcases import materialize_microcase


def test_compiled_surface_release_classifies_rows_and_preserves_event_budget(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-surface-release")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    source_facet = 3
    position = np.repeat([[0.0, 0.5]], 7, axis=0)
    velocity = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype="<f8",
    )
    acceleration = np.zeros((7, 2), dtype="<f8")
    acceleration[4] = [1.0, 0.0]
    acceleration[5] = [-1.0, 0.0]
    state = np.asarray(
        [
            SURFACE_STATE_RESOLVED,
            SURFACE_STATE_DEPARTURE,
            SURFACE_STATE_PENDING,
            SURFACE_STATE_PENDING,
            SURFACE_STATE_PENDING,
            SURFACE_STATE_PENDING,
            SURFACE_STATE_PENDING,
        ],
        dtype="<u1",
    )
    start_time = np.arange(7, dtype="<f8") * 0.125

    actual = classify_surface_release_batch(
        geometry,
        state,
        np.full(7, source_facet, dtype="<i8"),
        position,
        velocity,
        np.zeros(7, dtype="<f8"),
        acceleration,
        start_time,
        interval_s=0.25,
        curved_event_path=False,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    np.testing.assert_array_equal(
        actual.action,
        [
            SURFACE_ACTION_RESOLVED,
            SURFACE_ACTION_EXACT_DEPARTURE,
            SURFACE_ACTION_RESOLVED,
            SURFACE_ACTION_RESPONSE_VELOCITY,
            SURFACE_ACTION_EXACT_DEPARTURE,
            SURFACE_ACTION_RESPONSE_ACCELERATION,
            SURFACE_ACTION_RESOLVED,
        ],
    )
    np.testing.assert_array_equal(
        actual.status,
        [
            SURFACE_STATUS_OK,
            SURFACE_STATUS_OK,
            SURFACE_STATUS_OK,
            SURFACE_STATUS_OK,
            SURFACE_STATUS_OK,
            SURFACE_STATUS_OK,
            SURFACE_STATUS_INDETERMINATE_DIRECTION,
        ],
    )
    np.testing.assert_array_equal(actual.departure_facet_id, [-1, 3, -1, -1, 3, -1, -1])
    for row in (3, 5):
        expected = resolve_event_budget(
            facet_length_m=float(geometry.facet_length_m[source_facet]),
            geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
            position_m=position[row],
            speed_m_s=math.hypot(float(velocity[row, 0]), float(velocity[row, 1])),
            interval_s=0.25,
            time_s=float(start_time[row]),
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
        )
        assert actual.position_budget_m[row] == expected.position_m
        assert actual.time_budget_s[row] == expected.time_s

    curved = classify_surface_release_batch(
        geometry,
        state[:4],
        np.full(4, source_facet, dtype="<i8"),
        position[:4],
        velocity[:4],
        np.zeros(4, dtype="<f8"),
        None,
        start_time[:4],
        interval_s=0.25,
        curved_event_path=True,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    np.testing.assert_array_equal(
        curved.action,
        [
            SURFACE_ACTION_RESOLVED,
            SURFACE_ACTION_CURVED_DEPARTURE,
            SURFACE_ACTION_CURVED_DEPARTURE,
            SURFACE_ACTION_RESPONSE_VELOCITY,
        ],
    )
    np.testing.assert_array_equal(curved.status, np.zeros(4, dtype="<u1"))


def test_c07_exact_ballistic_first_hit_uses_scale_aware_budgets(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    hit = locate_ballistic_first_hit(
        geometry,
        np.asarray([0.25, 0.25]),
        np.asarray([0.5, 0.25]),
        start_time_s=0.0,
        end_time_s=2.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert hit is not None
    assert hit.time_s == 1.5
    np.testing.assert_array_equal(hit.position_m, [1.0, 0.625])
    assert hit.facet_id == 1
    assert hit.candidate_facet_ids == (1,)
    np.testing.assert_array_equal(hit.normal, [1.0, 0.0])
    assert hit.localization_residual_m == 0.0
    speed = math.hypot(0.5, 0.25)
    roundoff_factor = 64.0 * np.finfo(np.float64).eps
    position_norm = math.hypot(1.0, 0.625)
    roundoff_position = roundoff_factor * max(
        geometry.bbox_diagonal_m,
        position_norm,
        1.0,
    )
    position_ulp_floor = 64.0 * max(
        math.ulp(1.0),
        math.ulp(0.625),
        math.ulp(geometry.bbox_diagonal_m),
        math.ulp(1.0),
    )
    expected_position_budget = math.nextafter(
        1.0e-12 + max(roundoff_position, position_ulp_floor),
        math.inf,
    )
    expected_time_budget = max(
        expected_position_budget / speed,
        roundoff_factor * 2.0,
    )
    assert hit.position_budget_m == expected_position_budget
    assert hit.time_budget_s == expected_time_budget


def test_compiled_exact_batch_matches_scalar_line_and_quadratic_rows(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-exact-batch")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    path_kind = np.asarray(
        [
            EXACT_PATH_LINEAR,
            EXACT_PATH_LINEAR,
            EXACT_PATH_QUADRATIC,
            EXACT_PATH_QUADRATIC,
            EXACT_PATH_QUADRATIC,
            EXACT_PATH_QUADRATIC,
        ],
        dtype=np.uint8,
    )
    position = np.asarray(
        [[0.25, 0.25], [0.25, 0.25], [0.5, 0.5], [0.5, 0.5], [0.0, 0.5], [0.75, 0.5]],
        dtype="<f8",
    )
    velocity = np.asarray(
        [[0.5, 0.25], [0.1, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        dtype="<f8",
    )
    acceleration = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [-2.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]],
        dtype="<f8",
    )
    target = np.asarray([2.0, 0.25, 2.0, 1.0, 1.1, 1.0], dtype="<f8")
    departing = np.asarray([-1, -1, -1, -1, 3, -1], dtype="<i8")

    actual = locate_exact_first_event_batch(
        geometry,
        path_kind,
        position,
        velocity,
        acceleration,
        start_time_s=np.zeros(path_kind.size, dtype="<f8"),
        target_time_s=target,
        contact_radius_m=np.zeros(path_kind.size, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count * path_kind.size,
        certified_departing_facet_id=departing,
    )

    assert actual.status.tolist() == [
        EXACT_STATUS_WALL,
        EXACT_STATUS_CLEAR,
        EXACT_STATUS_WALL,
        EXACT_STATUS_CLEAR,
        EXACT_STATUS_WALL,
        EXACT_STATUS_FAILURE,
    ]
    for row in (0, 2, 4):
        if path_kind[row] == EXACT_PATH_LINEAR:
            expected = locate_ballistic_first_hit(
                geometry,
                position[row],
                velocity[row],
                start_time_s=0.0,
                end_time_s=float(target[row]),
                geometry_rtol=1.0e-12,
                roundoff_ulps=64,
            )
        else:
            expected = locate_constant_acceleration_first_hit(
                geometry,
                position[row],
                velocity[row],
                acceleration[row],
                start_time_s=0.0,
                end_time_s=float(target[row]),
                geometry_rtol=1.0e-12,
                roundoff_ulps=64,
                certified_departing_facet_id=(None if departing[row] < 0 else int(departing[row])),
            )
        assert expected is not None
        assert actual.time_s[row] == expected.time_s
        np.testing.assert_array_equal(actual.position_m[row], expected.position_m)
        assert actual.primary_facet_id[row] == expected.facet_id
        np.testing.assert_array_equal(actual.normal[row], expected.normal)
        assert actual.position_budget_m[row] == expected.position_budget_m
        assert actual.time_budget_s[row] == expected.time_budget_s
        assert actual.localization_residual_m[row] == expected.localization_residual_m
        begin, end = actual.candidate_offsets[row : row + 2]
        assert tuple(actual.candidate_facet_ids[begin:end]) == expected.candidate_facet_ids
    assert actual.failure_reason[5] == EXACT_FAILURE_INDETERMINATE_EVENT


def test_exact_candidate_count_fails_closed_before_dense_batch_exceeds_capacity(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-exact-capacity")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    row_count = 3
    path_kind = np.full(row_count, EXACT_PATH_LINEAR, dtype=np.uint8)
    position = np.zeros((row_count, 2), dtype="<f8")
    velocity = np.ones((row_count, 2), dtype="<f8")
    acceleration = np.zeros((row_count, 2), dtype="<f8")
    start_time = np.zeros(row_count, dtype="<f8")
    target_time = np.ones(row_count, dtype="<f8")
    counts = count_exact_event_candidates(
        geometry,
        path_kind,
        position,
        velocity,
        acceleration,
        start_time_s=start_time,
        target_time_s=target_time,
        contact_radius_m=np.zeros(row_count, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    np.testing.assert_array_equal(counts, [4, 4, 4])
    with pytest.raises(ExactCandidateCapacityError) as batch_error:
        locate_exact_first_event_batch(
            geometry,
            path_kind,
            position,
            velocity,
            acceleration,
            start_time_s=start_time,
            target_time_s=target_time,
            contact_radius_m=np.zeros(row_count, dtype="<f8"),
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
            candidate_capacity=11,
        )
    assert batch_error.value.required_count == 12
    assert batch_error.value.capacity == 11
    assert batch_error.value.oversized_row is None

    with pytest.raises(ExactCandidateCapacityError) as row_error:
        locate_exact_first_event_batch(
            geometry,
            path_kind[:1],
            position[:1],
            velocity[:1],
            acceleration[:1],
            start_time_s=start_time[:1],
            target_time_s=target_time[:1],
            contact_radius_m=np.zeros(1, dtype="<f8"),
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
            candidate_capacity=3,
        )
    assert row_error.value.required_count == 4
    assert row_error.value.oversized_row == 0

    fitted = locate_exact_first_event_batch(
        geometry,
        path_kind,
        position,
        velocity,
        acceleration,
        start_time_s=start_time,
        target_time_s=target_time,
        contact_radius_m=np.zeros(row_count, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=12,
    )
    np.testing.assert_array_equal(fitted.status, np.full(row_count, EXACT_STATUS_WALL))
    np.testing.assert_array_equal(fitted.candidate_offsets, [0, 2, 4, 6])
    np.testing.assert_array_equal(fitted.candidate_facet_ids, [1, 2, 1, 2, 1, 2])


def test_compiled_exact_batch_preserves_corner_candidate_identity(tmp_path: Path) -> None:
    paths = materialize_microcase("C10", tmp_path / "C10-exact-batch")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    actual = locate_exact_first_event_batch(
        geometry,
        np.asarray([EXACT_PATH_LINEAR], dtype=np.uint8),
        np.asarray([[1.0, 0.25]], dtype="<f8"),
        np.asarray([[0.0, 1.0]], dtype="<f8"),
        np.zeros((1, 2), dtype="<f8"),
        start_time_s=np.asarray([0.0]),
        target_time_s=np.asarray([1.0]),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
    )

    assert actual.status[0] == EXACT_STATUS_WALL
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 2])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1, 2])


def test_finite_radius_exact_paths_hit_the_wall_at_particle_center_clearance(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-exact")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    path_kind = np.asarray([EXACT_PATH_LINEAR, EXACT_PATH_QUADRATIC], dtype=np.uint8)

    actual = locate_exact_first_event_batch(
        geometry,
        path_kind,
        np.asarray([[0.25, 0.5], [0.25, 0.5]], dtype="<f8"),
        np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype="<f8"),
        np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype="<f8"),
        start_time_s=np.zeros(2, dtype="<f8"),
        target_time_s=np.ones(2, dtype="<f8"),
        contact_radius_m=np.full(2, 0.2, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=2 * geometry.facet_count,
    )

    np.testing.assert_array_equal(actual.status, [EXACT_STATUS_WALL, EXACT_STATUS_WALL])
    np.testing.assert_array_equal(actual.primary_facet_id, [1, 1])
    np.testing.assert_allclose(
        actual.time_s,
        [0.55, math.sqrt(0.55)],
        rtol=0.0,
        atol=5.0e-11,
    )
    np.testing.assert_allclose(actual.position_m, [[0.8, 0.5], [0.8, 0.5]], atol=6.0e-11)
    assert bool((actual.localization_residual_m <= actual.position_budget_m).all())


def test_finite_radius_exact_corner_keeps_both_simultaneous_facets(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-corner")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    actual = locate_exact_first_event_batch(
        geometry,
        np.asarray([EXACT_PATH_LINEAR], dtype=np.uint8),
        np.asarray([[0.5, 0.5]], dtype="<f8"),
        np.asarray([[1.0, 1.0]], dtype="<f8"),
        np.zeros((1, 2), dtype="<f8"),
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.ones(1, dtype="<f8"),
        contact_radius_m=np.asarray([0.2], dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
    )

    np.testing.assert_array_equal(actual.status, [EXACT_STATUS_WALL])
    np.testing.assert_allclose(actual.time_s, [0.3], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(actual.position_m, [[0.8, 0.8]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 2])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1, 2])


def test_finite_radius_corner_departure_omits_all_certified_start_contacts(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-corner-departure")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.asarray([[0.8, 0.8]], dtype="<f8")
    end = np.asarray([[0.7, 0.7]], dtype="<f8")
    velocity = np.asarray([[-1.0, -1.0]], dtype="<f8")
    radius = np.asarray([0.2], dtype="<f8")

    exact = locate_exact_first_event_batch(
        geometry,
        np.asarray([EXACT_PATH_LINEAR, EXACT_PATH_QUADRATIC], dtype=np.uint8),
        np.repeat(start, 2, axis=0),
        np.repeat(velocity, 2, axis=0),
        np.zeros((2, 2), dtype="<f8"),
        start_time_s=np.zeros(2, dtype="<f8"),
        target_time_s=np.full(2, 0.1, dtype="<f8"),
        contact_radius_m=np.full(2, radius[0], dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=2 * geometry.facet_count,
        certified_departing_facet_id=np.asarray(
            [EXACT_DEPARTURE_FINITE_CONTACT_SET, EXACT_DEPARTURE_FINITE_CONTACT_SET],
            dtype="<i8",
        ),
    )
    curved = locate_curved_first_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        np.minimum(start, end),
        np.maximum(start, end),
        velocity,
        velocity,
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.asarray([0.1], dtype="<f8"),
        root_interval_s=np.asarray([0.1], dtype="<f8"),
        contact_radius_m=radius,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        certify_start_contact_departure=np.asarray([True]),
        chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
    )

    np.testing.assert_array_equal(exact.status, [EXACT_STATUS_CLEAR, EXACT_STATUS_CLEAR])
    np.testing.assert_array_equal(curved.status, [CURVED_STATUS_CLEAR])
    np.testing.assert_array_equal(curved.start_contact_departure_certified, [True])


def test_finite_radius_contact_set_departure_preserves_a_later_adjacent_hit(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-adjacent-hit")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    actual = locate_exact_first_event_batch(
        geometry,
        np.asarray([EXACT_PATH_LINEAR], dtype=np.uint8),
        np.asarray([[0.1, 0.5]], dtype="<f8"),
        np.asarray([[1.0, 1.0]], dtype="<f8"),
        np.zeros((1, 2), dtype="<f8"),
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.asarray([0.45], dtype="<f8"),
        contact_radius_m=np.asarray([0.1], dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        certified_departing_facet_id=np.asarray(
            [EXACT_DEPARTURE_FINITE_CONTACT_SET],
            dtype="<i8",
        ),
    )

    np.testing.assert_array_equal(actual.status, [EXACT_STATUS_WALL])
    np.testing.assert_allclose(actual.time_s, [0.4], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(actual.position_m, [[0.5, 0.9]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_array_equal(actual.candidate_facet_ids, [2])


def test_compiled_exact_batch_rejects_simultaneous_disconnected_facets_per_row() -> None:
    gap_m = 5.0e-7
    nodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0 + gap_m, 0.0],
            [2.0 + gap_m, 0.0],
            [2.0 + gap_m, 1.0],
            [1.0 + gap_m, 1.0],
        ],
        dtype="<f8",
    )
    facets = np.asarray(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]],
        dtype="<i8",
    )
    geometry = prepare_geometry(
        GeometryData(
            nodes_m=nodes,
            boundary=BoundaryData(
                line2=facets,
                boundary_id=np.arange(facets.shape[0], dtype="<i4"),
                group_id=np.zeros(facets.shape[0], dtype="<i4"),
                material_id=np.zeros(facets.shape[0], dtype="<i4"),
                owner_cell_type=np.full(facets.shape[0], 2, dtype="<u1"),
                owner_cell_local_index=np.repeat(np.arange(2, dtype="<i8"), 4),
                orientation=np.ones(facets.shape[0], dtype="<i1"),
            ),
            group_names=("wall",),
            quad4=np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="<i8"),
            quad4_domain_id=np.zeros(2, dtype="<i4"),
        ),
        "cartesian_xy",
    )
    path_kind = np.full(2, EXACT_PATH_LINEAR, dtype=np.uint8)
    position = np.asarray([[0.5, 0.5], [1.5 + gap_m, 0.5]], dtype="<f8")
    velocity = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype="<f8")
    acceleration = np.zeros((2, 2), dtype="<f8")
    start_time = np.zeros(2, dtype="<f8")
    target_time = np.ones(2, dtype="<f8")

    actual = locate_exact_first_event_batch(
        geometry,
        path_kind,
        position,
        velocity,
        acceleration,
        start_time_s=start_time,
        target_time_s=target_time,
        contact_radius_m=np.zeros(2, dtype="<f8"),
        geometry_rtol=1.0e-6,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count * 2,
    )
    expected = locate_exact_first_event_batch(
        geometry,
        path_kind[1:],
        position[1:],
        velocity[1:],
        acceleration[1:],
        start_time_s=start_time[1:],
        target_time_s=target_time[1:],
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-6,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
    )

    np.testing.assert_array_equal(actual.status, [EXACT_STATUS_FAILURE, EXACT_STATUS_WALL])
    np.testing.assert_array_equal(
        actual.failure_reason,
        [EXACT_FAILURE_INDETERMINATE_EVENT, 0],
    )
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 0, 2])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [5, 6])
    np.testing.assert_array_equal(actual.time_s[[1]], expected.time_s)
    np.testing.assert_array_equal(actual.position_m[[1]], expected.position_m)
    np.testing.assert_array_equal(actual.primary_facet_id[[1]], expected.primary_facet_id)
    np.testing.assert_array_equal(actual.normal[[1]], expected.normal)
    np.testing.assert_array_equal(actual.position_budget_m[[1]], expected.position_budget_m)
    np.testing.assert_array_equal(actual.time_budget_s[[1]], expected.time_budget_s)
    np.testing.assert_array_equal(
        actual.localization_residual_m[[1]],
        expected.localization_residual_m,
    )
    np.testing.assert_array_equal(expected.candidate_offsets, [0, 2])
    np.testing.assert_array_equal(expected.candidate_facet_ids, [5, 6])


@pytest.mark.parametrize(
    ("start_x_m", "expected_status"),
    [
        (0.75 + 2.0**-20, EXACT_STATUS_WALL),
        (0.75 - 2.0**-20, EXACT_STATUS_CLEAR),
    ],
)
def test_compiled_exact_batch_preserves_near_grazing_quadratic_verdict(
    tmp_path: Path,
    start_x_m: float,
    expected_status: np.uint8,
) -> None:
    paths = materialize_microcase("C07", tmp_path / f"C07-batch-grazing-{start_x_m}")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    actual = locate_exact_first_event_batch(
        geometry,
        np.asarray([EXACT_PATH_QUADRATIC], dtype=np.uint8),
        np.asarray([[start_x_m, 0.5]], dtype="<f8"),
        np.asarray([[1.0, 0.0]], dtype="<f8"),
        np.asarray([[-2.0, 0.0]], dtype="<f8"),
        start_time_s=np.asarray([0.0]),
        target_time_s=np.asarray([1.0]),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
    )

    assert actual.status[0] == expected_status


def test_compiled_exact_batch_reports_rz_axis_as_a_chart_event() -> None:
    nodes = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype="<f8")
    raw_geometry = GeometryData(
        nodes_m=nodes,
        boundary=BoundaryData(
            line2=np.asarray([[0, 1], [1, 2], [2, 3]], dtype="<i8"),
            boundary_id=np.asarray([10, 11, 12], dtype="<i4"),
            group_id=np.zeros(3, dtype="<i4"),
            material_id=np.zeros(3, dtype="<i4"),
            owner_cell_type=np.full(3, 2, dtype="<u1"),
            owner_cell_local_index=np.zeros(3, dtype="<i8"),
            orientation=np.ones(3, dtype="<i1"),
        ),
        group_names=("wall",),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    geometry = prepare_geometry(raw_geometry, "axisymmetric_rz")
    actual = locate_exact_first_event_batch(
        geometry,
        np.full(4, EXACT_PATH_LINEAR, dtype=np.uint8),
        np.asarray([[0.5, 0.5], [0.0, 0.25], [0.5, 0.5], [0.5, 0.5]], dtype="<f8"),
        np.asarray([[-1.0, 0.2], [-1.0, 0.0], [0.1, 0.0], [-1.0, -2.0]], dtype="<f8"),
        np.zeros((4, 2), dtype="<f8"),
        start_time_s=np.zeros(4, dtype="<f8"),
        target_time_s=np.ones(4, dtype="<f8"),
        contact_radius_m=np.zeros(4, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count * 4,
    )

    np.testing.assert_array_equal(
        actual.status,
        [EXACT_STATUS_AXIS, EXACT_STATUS_AXIS, EXACT_STATUS_CLEAR, EXACT_STATUS_WALL],
    )
    np.testing.assert_array_equal(actual.time_s[:2], [0.5, 0.0])
    np.testing.assert_array_equal(actual.position_m[:2], [[0.0, 0.6], [0.0, 0.25]])
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [0])


def test_finite_radius_rz_sphere_hits_material_before_its_center(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-rz")
    case = load_case(paths.case_path)
    raw = case.data.geometry
    geometry = prepare_geometry(
        replace(
            raw,
            boundary=replace(
                raw.boundary,
                line2=raw.boundary.line2[:3],
                boundary_id=raw.boundary.boundary_id[:3],
                group_id=raw.boundary.group_id[:3],
                material_id=raw.boundary.material_id[:3],
                owner_cell_type=raw.boundary.owner_cell_type[:3],
                owner_cell_local_index=raw.boundary.owner_cell_local_index[:3],
                orientation=raw.boundary.orientation[:3],
            ),
        ),
        "axisymmetric_rz",
    )

    actual = locate_exact_first_event_batch(
        geometry,
        np.asarray([EXACT_PATH_LINEAR], dtype=np.uint8),
        np.asarray([[0.5, 0.5]], dtype="<f8"),
        np.asarray([[1.0, 0.0]], dtype="<f8"),
        np.zeros((1, 2), dtype="<f8"),
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.asarray([2.0], dtype="<f8"),
        contact_radius_m=np.asarray([0.2], dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
    )

    np.testing.assert_array_equal(actual.status, [EXACT_STATUS_WALL])
    np.testing.assert_allclose(actual.time_s, [0.3], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(actual.position_m, [[0.8, 0.5]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_array_equal(actual.primary_facet_id, [1])


def test_event_budget_remains_positive_for_smallest_subnormal_interval() -> None:
    interval_s = float(np.nextafter(0.0, np.inf))
    budget = resolve_event_budget(
        facet_length_m=1.0,
        geometry_bbox_diagonal_m=1.0,
        position_m=np.asarray([0.0, 0.0], dtype="<f8"),
        speed_m_s=0.0,
        interval_s=interval_s,
        time_s=0.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert math.isfinite(budget.position_m) and budget.position_m > 0.0
    assert math.isfinite(budget.time_s) and budget.time_s > 0.0


def test_ballistic_path_without_a_boundary_crossing_has_no_event(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-no-hit")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    hit = locate_ballistic_first_hit(
        geometry,
        np.asarray([0.25, 0.25]),
        np.asarray([0.1, 0.0]),
        start_time_s=0.0,
        end_time_s=0.25,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert hit is None


def test_compiled_preclassifier_falls_back_for_nonfinite_speed_bound(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-preclassifier-overflow")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    lower = np.asarray([[0.2, 0.2], [0.8, 0.4]], dtype="<f8")
    upper = np.asarray([[0.3, 0.3], [1.2, 0.6]], dtype="<f8")
    velocity_lower = np.full((2, 2), -1.7e308, dtype="<f8")
    velocity_upper = np.full((2, 2), 1.7e308, dtype="<f8")

    verdict = preclassify_rk4_piece_batch(
        lower,
        upper,
        velocity_lower,
        velocity_upper,
        np.zeros(2, dtype="<f8"),
        1.0,
        np.ones(2, dtype="<f8"),
        np.zeros(2, dtype=np.bool_),
        geometry.bbox_diagonal_m,
        geometry.facet_start_m,
        geometry.facet_end_m,
        geometry.facet_normal,
        geometry.facet_length_m,
        geometry.bvh_facet_id,
        geometry.bvh_lower_m,
        geometry.bvh_upper_m,
        geometry.bvh_left,
        geometry.bvh_right,
        geometry.bvh_begin,
        geometry.bvh_end,
        1.0e-12,
        64,
    )

    np.testing.assert_array_equal(verdict, [0, 0])


def test_rk4_tube_uses_supporting_line_to_reject_a_slanted_aabb_false_positive() -> None:
    nodes = np.asarray([[0.1, 0.3], [0.0, 1.1], [0.0, 0.2]], dtype="<f8")
    raw_geometry = GeometryData(
        nodes_m=nodes,
        boundary=BoundaryData(
            line2=np.asarray([[0, 1], [2, 0]], dtype="<i8"),
            boundary_id=np.asarray([10, 10], dtype="<i4"),
            group_id=np.asarray([0, 0], dtype="<i4"),
            material_id=np.asarray([0, 0], dtype="<i4"),
            owner_cell_type=np.asarray([1, 1], dtype="<u1"),
            owner_cell_local_index=np.asarray([0, 0], dtype="<i8"),
            orientation=np.asarray([1, 1], dtype="<i1"),
        ),
        group_names=("wall",),
        tri3=np.asarray([[0, 1, 2]], dtype="<i8"),
        tri3_domain_id=np.asarray([0], dtype="<i4"),
    )
    geometry = prepare_geometry(raw_geometry, "axisymmetric_rz")

    decision = inspect_rk4_piece(
        geometry,
        np.asarray([0.0, 0.317]),
        np.asarray([0.0, 0.317]),
        np.asarray([-1.0e-6, 0.316]),
        np.asarray([1.0e-6, 0.318]),
        np.asarray([-1.0e-4, -1.0e-4]),
        np.asarray([1.0e-4, 1.0e-4]),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=0.1,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert decision.kind == "clear"
    assert decision.hit is None
    assert decision.candidate_facet_count == 0


def test_rk4_turning_tube_is_not_cleared_by_endpoint_chord(tmp_path: Path) -> None:
    """x=.9+.44t-.44t^2 returns to .9 but crosses the right wall at its turn."""

    paths = materialize_microcase("C07", tmp_path / "C07-rk4-turning-tube")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    particle_index = np.asarray([0], dtype="<i8")
    start_time_s = np.asarray([0.0], dtype="<f8")
    start_position_m = np.asarray([[0.9, 0.5]], dtype="<f8")
    start_velocity_m_s = np.asarray([[0.44, 0.0]], dtype="<f8")
    charge_number = np.asarray([3.0], dtype="<f8")

    def evaluate(
        selected_particle: np.ndarray,
        time_s: np.ndarray,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        selected_charge: np.ndarray,
    ) -> DynamicsEvaluation:
        del time_s, position_m, velocity_m_s
        acceleration = np.zeros((selected_particle.size, 2), dtype="<f8")
        acceleration[:, 0] = -0.88
        return DynamicsEvaluation(
            acceleration,
            np.zeros_like(selected_charge),
            np.ones(selected_particle.size, dtype=np.bool_),
            np.ones(selected_particle.size, dtype=np.bool_),
            np.zeros(selected_particle.size, dtype=np.uint8),
        )

    def bound_acceleration(
        selected_particle: np.ndarray,
        velocity_abs_upper_m_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del velocity_abs_upper_m_s
        bound = np.zeros((selected_particle.size, 2), dtype="<f8")
        bound[:, 0] = 0.88
        return bound, np.zeros(selected_particle.size, dtype=np.uint8)

    enclosure = enclose_rk4_path(
        particle_index,
        start_time_s,
        np.full(particle_index.size, 1.0, dtype="<f8"),
        start_position_m,
        start_velocity_m_s,
        acceleration_abs_bounder=bound_acceleration,
    )
    proposal = rk4_step(
        particle_index,
        start_time_s,
        np.full(particle_index.size, 1.0, dtype="<f8"),
        start_position_m,
        start_velocity_m_s,
        charge_number,
        requires_stage_evaluation=True,
        evaluator=evaluate,
        path_enclosure=enclosure,
    )

    assert proposal.path_kind == "rk4_dense"
    decision = inspect_rk4_piece(
        geometry,
        proposal.start_position_m[0],
        proposal.end_position_m[0],
        enclosure.position_lower_m[0],
        enclosure.position_upper_m[0],
        enclosure.velocity_lower_m_s[0],
        enclosure.velocity_upper_m_s[0],
        start_time_s=0.0,
        end_time_s=1.0,
        root_interval_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert decision.kind == "split"
    assert decision.hit is None
    assert decision.candidate_facet_count == 1


def test_rk4_transverse_chord_deviation_certifies_hit(tmp_path: Path) -> None:
    """A long chord may be accurate even though its swept AABB is not small."""

    paths = materialize_microcase("C07", tmp_path / "C07-rk4-transverse")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start_time_s = 0.25
    interval_s = 2.0e-6
    decision = inspect_rk4_piece(
        geometry,
        np.asarray([1.0 - 1.0e-6, 0.5]),
        np.asarray([1.0 + 1.0e-6, 0.5]),
        np.asarray([1.0 - 1.0e-6, 0.5 - 1.0e-13]),
        np.asarray([1.0 + 1.0e-6, 0.5 + 1.0e-13]),
        np.asarray([1.0 - 5.0e-8, -5.0e-8]),
        np.asarray([1.0 + 5.0e-8, 5.0e-8]),
        start_time_s=start_time_s,
        end_time_s=start_time_s + interval_s,
        root_interval_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert decision.kind == "hit"
    assert decision.hit is not None
    assert decision.candidate_facet_count == 1
    assert decision.hit.facet_id == 1
    assert decision.hit.candidate_facet_ids == (1,)
    assert (
        abs(decision.hit.time_s - (start_time_s + 0.5 * interval_s)) <= decision.hit.time_budget_s
    )
    np.testing.assert_array_equal(decision.hit.position_m, [1.0, 0.5])
    assert decision.hit.localization_residual_m <= decision.hit.position_budget_m


def test_rk4_transverse_shortcut_rejects_large_tangential_deviation(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-rk4-transverse-tangent")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start_time_s = 0.25
    interval_s = 2.0e-6
    decision = inspect_rk4_piece(
        geometry,
        np.asarray([1.0 - 1.0e-6, 0.5]),
        np.asarray([1.0 + 1.0e-6, 0.5]),
        np.asarray([1.0 - 1.0e-6, 0.5 - 1.0e-13]),
        np.asarray([1.0 + 1.0e-6, 0.5 + 1.0e-13]),
        np.asarray([1.0 - 5.0e-8, -2.0e-6]),
        np.asarray([1.0 + 5.0e-8, 2.0e-6]),
        start_time_s=start_time_s,
        end_time_s=start_time_s + interval_s,
        root_interval_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert decision.kind == "split"
    assert decision.hit is None
    assert decision.candidate_facet_count == 1


def test_curved_piece_uses_supplied_componentwise_chord_bound(tmp_path: Path) -> None:
    """A method-owned bound may certify a hit but never hide its uncertainty."""

    paths = materialize_microcase("C07", tmp_path / "C07-method-chord-bound")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    arguments = (
        geometry,
        np.asarray([1.0 - 1.0e-6, 0.5]),
        np.asarray([1.0 + 1.0e-6, 0.5]),
        np.asarray([1.0 - 1.0e-6, 0.5 - 1.0e-13]),
        np.asarray([1.0 + 1.0e-6, 0.5 + 1.0e-13]),
        np.asarray([1.0 - 5.0e-8, -5.0e-8]),
        np.asarray([1.0 + 5.0e-8, 5.0e-8]),
    )
    keywords = {
        "start_time_s": 0.25,
        "end_time_s": 0.25 + 2.0e-6,
        "root_interval_s": 1.0,
        "geometry_rtol": 1.0e-12,
        "roundoff_ulps": 64,
    }

    certified = inspect_rk4_piece(
        *arguments,
        **keywords,
        chord_deviation_bound_m=np.zeros(2, dtype="<f8"),
    )
    conservative = inspect_rk4_piece(
        *arguments,
        **keywords,
        chord_deviation_bound_m=np.full(2, 1.0e-4, dtype="<f8"),
    )

    assert certified.kind == "hit"
    assert certified.hit is not None
    assert conservative.kind == "split"
    assert conservative.hit is None
    assert conservative.candidate_facet_count == 1


def test_small_full_tube_localizes_after_a_conservative_method_bound(tmp_path: Path) -> None:
    """A large chord bound falls back to the established full-tube certificate."""

    paths = materialize_microcase("C07", tmp_path / "C07-method-full-tube")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    interval_s = 2.0e-13
    decision = inspect_rk4_piece(
        geometry,
        np.asarray([1.0 - 2.0e-13, 0.5]),
        np.asarray([1.0 + 2.0e-13, 0.5]),
        np.asarray([1.0 - 2.0e-13, 0.5]),
        np.asarray([1.0 + 2.0e-13, 0.5]),
        np.asarray([2.0, 0.0]),
        np.asarray([2.0, 0.0]),
        start_time_s=0.25,
        end_time_s=0.25 + interval_s,
        root_interval_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        chord_deviation_bound_m=np.full(2, 1.0e-4, dtype="<f8"),
    )

    assert decision.kind == "hit"
    assert decision.hit is not None
    assert decision.hit.localization_residual_m <= decision.hit.position_budget_m


def test_rk4_near_parallel_crossing_is_split_instead_of_shortcut(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-rk4-near-parallel")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    decision = inspect_rk4_piece(
        geometry,
        np.asarray([1.0 - 2.0e-12, 0.25]),
        np.asarray([1.0 + 2.0e-12, 0.35]),
        np.asarray([1.0 - 2.0e-12, 0.25]),
        np.asarray([1.0 + 2.0e-12, 0.35]),
        np.asarray([3.9e-11, 1.0]),
        np.asarray([4.1e-11, 1.0]),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert decision.kind == "split"
    assert decision.hit is None
    assert decision.candidate_facet_count == 1


def test_rk4_start_contact_departure_omits_only_proven_facets(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-rk4-start-contact")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    decision = inspect_rk4_piece(
        geometry,
        np.asarray([0.0, 0.5]),
        np.asarray([1.1, 0.5]),
        np.asarray([0.0, 0.5]),
        np.asarray([1.1, 0.5]),
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
        start_time_s=0.0,
        end_time_s=1.1,
        root_interval_s=1.1,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        certify_start_contact_departure=True,
    )

    assert decision.kind == "hit"
    assert decision.hit is not None
    assert decision.candidate_facet_count == 1
    assert decision.start_contact_departure_certified
    assert decision.hit.facet_id == 1
    assert abs(decision.hit.time_s - 1.0) <= decision.hit.time_budget_s


def test_compiled_curved_batch_matches_scalar_wall_decisions(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-curved-batch")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.asarray(
        [[1.0 - 1.0e-6, 0.5], [1.0 - 1.0e-6, 0.5], [0.25, 0.25], [0.0, 0.5]],
        dtype="<f8",
    )
    end = np.asarray(
        [[1.0 + 1.0e-6, 0.5], [1.0 + 1.0e-6, 0.5], [0.35, 0.25], [1.1, 0.5]],
        dtype="<f8",
    )
    lower = np.asarray(
        [
            [1.0 - 1.0e-6, 0.5 - 1.0e-13],
            [1.0 - 1.0e-6, 0.5 - 1.0e-13],
            [0.25, 0.25],
            [0.0, 0.5],
        ],
        dtype="<f8",
    )
    upper = np.asarray(
        [
            [1.0 + 1.0e-6, 0.5 + 1.0e-13],
            [1.0 + 1.0e-6, 0.5 + 1.0e-13],
            [0.35, 0.25],
            [1.1, 0.5],
        ],
        dtype="<f8",
    )
    velocity_lower = np.asarray(
        [
            [1.0 - 5.0e-8, -5.0e-8],
            [1.0 - 5.0e-8, -2.0e-6],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype="<f8",
    )
    velocity_upper = np.asarray(
        [
            [1.0 + 5.0e-8, 5.0e-8],
            [1.0 + 5.0e-8, 2.0e-6],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype="<f8",
    )
    start_time = np.asarray([0.25, 0.25, 0.0, 0.0], dtype="<f8")
    target_time = np.asarray([0.250002, 0.250002, 0.1, 1.1], dtype="<f8")
    root_interval = np.asarray([1.0, 1.0, 0.1, 1.1], dtype="<f8")
    chord_bound = np.asarray([[0.0, 0.0], [1.0e-4, 1.0e-4], [0.0, 0.0], [0.0, 0.0]])
    departure = np.asarray([False, False, False, True])

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity_lower,
        end,
        velocity_upper,
        lower,
        upper,
        velocity_lower,
        velocity_upper,
        start_time_s=start_time,
        target_time_s=target_time,
        root_interval_s=root_interval,
        contact_radius_m=np.zeros(start.shape[0], dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count * start.shape[0],
        certify_start_contact_departure=departure,
        chord_deviation_bound_m=chord_bound,
    )

    np.testing.assert_array_equal(
        actual.status,
        [CURVED_STATUS_WALL, CURVED_STATUS_SPLIT, CURVED_STATUS_CLEAR, CURVED_STATUS_WALL],
    )
    for row in range(start.shape[0]):
        expected = inspect_rk4_piece(
            geometry,
            start[row],
            end[row],
            lower[row],
            upper[row],
            velocity_lower[row],
            velocity_upper[row],
            start_time_s=float(start_time[row]),
            end_time_s=float(target_time[row]),
            root_interval_s=float(root_interval[row]),
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
            certify_start_contact_departure=bool(departure[row]),
            chord_deviation_bound_m=chord_bound[row],
        )
        expected_status = {
            "clear": CURVED_STATUS_CLEAR,
            "split": CURVED_STATUS_SPLIT,
            "hit": CURVED_STATUS_WALL,
        }[expected.kind]
        assert actual.status[row] == expected_status
        assert actual.start_contact_departure_certified[row] == (
            expected.start_contact_departure_certified
        )
        if expected.hit is not None:
            assert actual.time_s[row] == expected.hit.time_s
            np.testing.assert_array_equal(actual.position_m[row], expected.hit.position_m)
            assert actual.primary_facet_id[row] == expected.hit.facet_id
            assert actual.position_budget_m[row] == expected.hit.position_budget_m
            assert actual.time_budget_s[row] == expected.hit.time_budget_s
            assert actual.localization_residual_m[row] == expected.hit.localization_residual_m
            begin, finish = actual.candidate_offsets[row : row + 2]
            assert tuple(actual.candidate_facet_ids[begin:finish]) == (
                expected.hit.candidate_facet_ids
            )


def test_curved_batch_default_chord_bound_matches_scalar(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-curved-default-chord")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.asarray([[1.0 - 1.0e-6, 0.5]], dtype="<f8")
    end = np.asarray([[1.0 + 1.0e-6, 0.5]], dtype="<f8")
    lower = np.asarray([[1.0 - 1.0e-6, 0.5 - 1.0e-13]], dtype="<f8")
    upper = np.asarray([[1.0 + 1.0e-6, 0.5 + 1.0e-13]], dtype="<f8")
    velocity_lower = np.asarray([[1.0 - 5.0e-8, -5.0e-8]], dtype="<f8")
    velocity_upper = np.asarray([[1.0 + 5.0e-8, 5.0e-8]], dtype="<f8")

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity_lower,
        end,
        velocity_upper,
        lower,
        upper,
        velocity_lower,
        velocity_upper,
        start_time_s=np.asarray([0.25]),
        target_time_s=np.asarray([0.250002]),
        root_interval_s=np.asarray([1.0]),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
    )
    expected = inspect_rk4_piece(
        geometry,
        start[0],
        end[0],
        lower[0],
        upper[0],
        velocity_lower[0],
        velocity_upper[0],
        start_time_s=0.25,
        end_time_s=0.250002,
        root_interval_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert expected.hit is not None
    assert actual.status[0] == CURVED_STATUS_WALL
    assert actual.time_s[0] == expected.hit.time_s
    assert actual.localization_residual_m[0] == expected.hit.localization_residual_m


@pytest.mark.parametrize("certify_monotone", [False, True])
def test_monotone_approach_is_clear_inside_the_localization_budget(
    tmp_path: Path, certify_monotone: bool
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-monotone-budget")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    budget = resolve_event_budget(
        facet_length_m=float(geometry.facet_length_m[1]),
        geometry_bbox_diagonal_m=geometry.bbox_diagonal_m,
        position_m=np.asarray([1.0, 0.5]),
        speed_m_s=0.1,
        interval_s=1.0,
        time_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    clearance = 0.25 * budget.position_m
    start = np.asarray([[0.9, 0.5]], dtype="<f8")
    end = np.asarray([[1.0 - clearance, 0.5]], dtype="<f8")
    velocity = end - start  # x(t)=x0+v*t, 0<=t<=1, strictly inside the right wall.
    assert 0.0 < 1.0 - end[0, 0] < budget.position_m

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        start,
        np.asarray([[1.01, 0.5]], dtype="<f8"),
        np.asarray([[0.09, 0.0]], dtype="<f8"),
        np.asarray([[0.11, 0.0]], dtype="<f8"),
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.ones(1, dtype="<f8"),
        root_interval_s=np.ones(1, dtype="<f8"),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        certify_monotone_approach=certify_monotone,
    )

    # The tolerance is a localization budget, not a wall thickness. The broad
    # position box alone cannot exclude a crossing; positive normal velocity
    # makes the strictly inward endpoint the maximum signed distance.
    np.testing.assert_array_equal(
        actual.status, [CURVED_STATUS_CLEAR if certify_monotone else CURVED_STATUS_SPLIT]
    )


def test_monotone_certificate_does_not_clear_an_exit_and_return_path(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-monotone-turn")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    clearance = 2.5e-13
    start = np.asarray([[0.9, 0.5]], dtype="<f8")
    end = np.asarray([[1.0 - clearance, 0.5]], dtype="<f8")
    # x(t)=.9+(.3-clearance)t-.2t² crosses x=1 and returns inside at t=1.
    slope = 0.3 - clearance
    peak = 0.9 + slope**2 / 0.8
    assert peak > 1.0 and end[0, 0] < 1.0
    actual = locate_curved_first_event_batch(
        geometry,
        start,
        np.asarray([[slope, 0.0]], dtype="<f8"),
        end,
        np.asarray([[slope - 0.4, 0.0]], dtype="<f8"),
        start,
        np.asarray([[peak, 0.5]], dtype="<f8"),
        np.asarray([[slope - 0.4, 0.0]], dtype="<f8"),
        np.asarray([[slope, 0.0]], dtype="<f8"),
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.ones(1, dtype="<f8"),
        root_interval_s=np.ones(1, dtype="<f8"),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        certify_monotone_approach=True,
    )
    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_SPLIT])


@pytest.mark.parametrize("coordinate_shift", [0.0, float(2**20)])
def test_rk4_bernstein_controls_clear_correlated_inside_path_per_row(
    tmp_path: Path,
    coordinate_shift: float,
) -> None:
    paths = materialize_microcase("C07", tmp_path / f"C07-control-clear-{coordinate_shift}")
    case = load_case(paths.case_path)
    shift = np.asarray([coordinate_shift, -coordinate_shift], dtype="<f8")
    raw_geometry = replace(case.data.geometry, nodes_m=case.data.geometry.nodes_m + shift)
    geometry = prepare_geometry(raw_geometry, case.data.coordinate_system)
    start = np.repeat((np.asarray([[0.9, 0.5]]) + shift), 2, axis=0)
    position_lower = np.repeat((np.asarray([[0.9, 0.49]]) + shift), 2, axis=0)
    position_upper = np.repeat((np.asarray([[1.1, 0.51]]) + shift), 2, axis=0)
    velocity_lower = np.full((2, 2), -1.0, dtype="<f8")
    velocity_upper = np.full((2, 2), 1.0, dtype="<f8")
    control_origin = start.copy()
    relative_controls = np.zeros((2, 4, 2), dtype="<f8")

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity_lower,
        start,
        velocity_upper,
        position_lower,
        position_upper,
        velocity_lower,
        velocity_upper,
        start_time_s=np.zeros(2, dtype="<f8"),
        target_time_s=np.full(2, 0.1, dtype="<f8"),
        root_interval_s=np.full(2, 0.1, dtype="<f8"),
        contact_radius_m=np.zeros(2, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=2 * geometry.facet_count,
        chord_deviation_bound_m=np.zeros((2, 2), dtype="<f8"),
        use_position_controls=np.asarray([True, False]),
        position_control_origin_m=control_origin,
        relative_position_control_lower_m=relative_controls,
        relative_position_control_upper_m=relative_controls,
    )

    # The component box overlaps the right wall for both rows.  Only the row
    # carrying a valid correlated Bernstein enclosure may be certified clear.
    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_CLEAR, CURVED_STATUS_SPLIT])


def test_rk4_bernstein_controls_do_not_clear_touching_or_outside_path(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-control-touch")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.repeat(np.asarray([[0.9, 0.5]], dtype="<f8"), 2, axis=0)
    relative_controls = np.zeros((2, 4, 2), dtype="<f8")
    relative_controls[0, 1, 0] = 0.1
    relative_controls[1, 1, 0] = 0.2
    velocity_lower = np.full((2, 2), -1.0, dtype="<f8")
    velocity_upper = np.full((2, 2), 1.0, dtype="<f8")

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity_lower,
        start,
        velocity_upper,
        np.repeat(np.asarray([[0.9, 0.49]], dtype="<f8"), 2, axis=0),
        np.asarray([[1.1, 0.51], [1.2, 0.51]], dtype="<f8"),
        velocity_lower,
        velocity_upper,
        start_time_s=np.zeros(2, dtype="<f8"),
        target_time_s=np.full(2, 0.1, dtype="<f8"),
        root_interval_s=np.full(2, 0.1, dtype="<f8"),
        contact_radius_m=np.zeros(2, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=2 * geometry.facet_count,
        chord_deviation_bound_m=np.zeros((2, 2), dtype="<f8"),
        position_control_origin_m=start,
        relative_position_control_lower_m=relative_controls,
        relative_position_control_upper_m=relative_controls,
    )

    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_SPLIT, CURVED_STATUS_SPLIT])


def test_rk4_bernstein_controls_preserve_true_wall_hit(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-control-hit")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.asarray([[0.9, 0.5]], dtype="<f8")
    end = np.asarray([[1.2, 0.5]], dtype="<f8")
    velocity = np.asarray([[0.3, 0.0]], dtype="<f8")
    relative_controls = np.zeros((1, 4, 2), dtype="<f8")
    relative_controls[0, :, 0] = [0.0, 0.1, 0.2, 0.3]

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        start,
        end,
        velocity,
        velocity,
        start_time_s=np.asarray([0.0]),
        target_time_s=np.asarray([1.0]),
        root_interval_s=np.asarray([1.0]),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
        position_control_origin_m=start,
        relative_position_control_lower_m=relative_controls,
        relative_position_control_upper_m=relative_controls,
    )

    assert actual.status[0] == CURVED_STATUS_WALL
    assert actual.primary_facet_id[0] == 1
    assert actual.time_s[0] == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(actual.position_m[0], [1.0, 0.5], rtol=0.0, atol=1.0e-15)


def test_finite_radius_curved_path_preserves_surface_clearance(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-curved")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.asarray([[0.25, 0.5]], dtype="<f8")
    end = np.asarray([[1.25, 0.5]], dtype="<f8")
    velocity = np.asarray([[1.0, 0.0]], dtype="<f8")

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        np.minimum(start, end),
        np.maximum(start, end),
        velocity,
        velocity,
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.ones(1, dtype="<f8"),
        root_interval_s=np.ones(1, dtype="<f8"),
        contact_radius_m=np.asarray([0.2], dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
    )

    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_WALL])
    np.testing.assert_allclose(actual.time_s, [0.55], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(actual.position_m, [[0.8, 0.5]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1])


def test_curved_candidate_capacity_fails_closed_before_fill(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-curved-capacity")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    row_count = 2
    start = np.zeros((row_count, 2), dtype="<f8")
    end = np.ones((row_count, 2), dtype="<f8")
    lower = start.copy()
    upper = end.copy()
    velocity_lower = np.zeros((row_count, 2), dtype="<f8")
    velocity_upper = np.ones((row_count, 2), dtype="<f8")
    times = np.zeros(row_count, dtype="<f8")
    targets = np.ones(row_count, dtype="<f8")

    counts = count_curved_event_candidates(
        geometry,
        start,
        velocity_lower,
        end,
        velocity_upper,
        lower,
        upper,
        velocity_lower,
        velocity_upper,
        start_time_s=times,
        target_time_s=targets,
        root_interval_s=targets,
        contact_radius_m=np.zeros(row_count, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    np.testing.assert_array_equal(counts, [4, 4])
    with pytest.raises(CurvedCandidateCapacityError) as batch_error:
        locate_curved_first_event_batch(
            geometry,
            start,
            velocity_lower,
            end,
            velocity_upper,
            lower,
            upper,
            velocity_lower,
            velocity_upper,
            start_time_s=times,
            target_time_s=targets,
            root_interval_s=targets,
            contact_radius_m=np.zeros(row_count, dtype="<f8"),
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
            candidate_capacity=7,
        )
    assert batch_error.value.required_count == 8
    assert batch_error.value.oversized_row is None
    with pytest.raises(CurvedCandidateCapacityError) as row_error:
        locate_curved_first_event_batch(
            geometry,
            start[:1],
            velocity_lower[:1],
            end[:1],
            velocity_upper[:1],
            lower[:1],
            upper[:1],
            velocity_lower[:1],
            velocity_upper[:1],
            start_time_s=times[:1],
            target_time_s=targets[:1],
            root_interval_s=targets[:1],
            contact_radius_m=np.zeros(1, dtype="<f8"),
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
            candidate_capacity=3,
        )
    assert row_error.value.required_count == 4
    assert row_error.value.oversized_row == 0


def test_rk4_axis_piece_localizes_certified_inward_crossing() -> None:
    decision = inspect_rk4_axis_piece(
        np.asarray([0.1, 0.5]),
        np.asarray([-2.0, 0.0]),
        np.asarray([-0.1, 0.5]),
        np.asarray([-2.0, 0.0]),
        np.asarray([-0.1, 0.5]),
        np.asarray([0.1, 0.5]),
        np.asarray([-2.0, 0.0]),
        np.asarray([-2.0, 0.0]),
        start_time_s=0.25,
        end_time_s=0.35,
        root_interval_s=0.1,
        geometry_bbox_diagonal_m=2.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert decision.kind == "hit"
    assert decision.hit is not None
    assert abs(decision.hit.time_s - 0.3) <= decision.hit.time_budget_s
    assert decision.hit.localization_residual_m <= decision.hit.position_budget_m


def test_compiled_curved_batch_preserves_rz_axis_and_wall_ordering() -> None:
    nodes = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype="<f8")
    raw_geometry = GeometryData(
        nodes_m=nodes,
        boundary=BoundaryData(
            line2=np.asarray([[0, 1], [1, 2], [2, 3]], dtype="<i8"),
            boundary_id=np.asarray([10, 11, 12], dtype="<i4"),
            group_id=np.zeros(3, dtype="<i4"),
            material_id=np.zeros(3, dtype="<i4"),
            owner_cell_type=np.full(3, 2, dtype="<u1"),
            owner_cell_local_index=np.zeros(3, dtype="<i8"),
            orientation=np.ones(3, dtype="<i1"),
        ),
        group_names=("wall",),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    geometry = prepare_geometry(raw_geometry, "axisymmetric_rz")
    start = np.asarray([[0.1, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype="<f8")
    velocity = np.asarray([[-2.0, 0.0], [-1.0, -2.0], [-1.0, -1.0]], dtype="<f8")
    end = np.asarray([[-0.1, 0.5], [-0.5, -1.5], [-0.5, -0.5]], dtype="<f8")
    lower = np.minimum(start, end)
    upper = np.maximum(start, end)
    target = np.asarray([0.1, 1.0, 1.0], dtype="<f8")

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        lower,
        upper,
        velocity,
        velocity,
        start_time_s=np.zeros(3, dtype="<f8"),
        target_time_s=target,
        root_interval_s=target,
        contact_radius_m=np.zeros(3, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count * start.shape[0],
        chord_deviation_bound_m=np.zeros((3, 2), dtype="<f8"),
    )

    np.testing.assert_array_equal(
        actual.status,
        [CURVED_STATUS_AXIS, CURVED_STATUS_WALL, CURVED_STATUS_SPLIT],
    )
    np.testing.assert_array_equal(actual.time_s[:2], [0.05, 0.25])
    np.testing.assert_array_equal(actual.primary_facet_id, [-1, 0, -1])
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 0, 1, 1])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [0])


def test_compiled_curved_batch_preserves_simultaneous_corner_facets(tmp_path: Path) -> None:
    paths = materialize_microcase("C10", tmp_path / "C10-curved-corner")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    interval_s = 4.0e-13
    start = np.asarray([[1.0, 1.0 - 0.5 * interval_s]], dtype="<f8")
    end = np.asarray([[1.0, 1.0 + 0.5 * interval_s]], dtype="<f8")
    velocity = np.asarray([[0.0, 1.0]], dtype="<f8")

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        np.minimum(start, end),
        np.maximum(start, end),
        velocity,
        velocity,
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.asarray([interval_s], dtype="<f8"),
        root_interval_s=np.asarray([1.0], dtype="<f8"),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
    )

    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_WALL])
    assert int(actual.primary_facet_id[0]) in {1, 2}
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 2])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1, 2])
    np.testing.assert_allclose(actual.position_m, [[1.0, 1.0]], rtol=0.0, atol=2.0e-15)


def test_compiled_curved_batch_discards_broad_but_certified_clear_facet(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C10", tmp_path / "C10-curved-clear-neighbor")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.asarray([[2.0, 0.4]], dtype="<f8")
    end = np.asarray([[2.0, 0.6]], dtype="<f8")
    velocity = np.asarray([[0.0, 1.0]], dtype="<f8")
    relative_controls = np.asarray(
        [[[0.0, 0.0], [0.0, 1.0 / 15.0], [0.0, 2.0 / 15.0], [0.0, 0.2]]],
        dtype="<f8",
    )

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        np.asarray([[0.5, 0.4]], dtype="<f8"),
        np.asarray([[2.5, 0.6]], dtype="<f8"),
        velocity,
        velocity,
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.asarray([0.2], dtype="<f8"),
        root_interval_s=np.asarray([0.2], dtype="<f8"),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
        use_position_controls=np.asarray([True]),
        position_control_origin_m=start,
        relative_position_control_lower_m=relative_controls,
        relative_position_control_upper_m=relative_controls,
    )

    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_WALL])
    np.testing.assert_array_equal(actual.primary_facet_id, [1])
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 1])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1])


def test_compiled_curved_corner_budget_covers_accepted_shared_node(tmp_path: Path) -> None:
    paths = materialize_microcase("C10", tmp_path / "C10-curved-near-corner")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.asarray([[1.0, 1.0 - 3.0e-12]], dtype="<f8")
    end = np.asarray([[1.0, 1.0 - 2.0e-12]], dtype="<f8")
    velocity = np.asarray([[0.0, 1.0]], dtype="<f8")

    actual = locate_curved_first_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        np.minimum(start, end),
        np.maximum(start, end),
        velocity,
        velocity,
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.asarray([1.0e-12], dtype="<f8"),
        root_interval_s=np.asarray([1.0], dtype="<f8"),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        candidate_capacity=geometry.facet_count,
        chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
    )

    node_distance_m = float(np.linalg.norm(actual.position_m[0] - np.asarray([1.0, 1.0])))
    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_WALL])
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 2])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1, 2])
    assert node_distance_m > 0.0
    assert node_distance_m <= actual.position_budget_m[0]


def test_axis_piece_uses_supplied_componentwise_chord_bound() -> None:
    arguments = (
        np.asarray([0.1, 0.5]),
        np.asarray([-2.0, 0.0]),
        np.asarray([-0.1, 0.5]),
        np.asarray([-2.0, 0.0]),
        np.asarray([-0.1, 0.5]),
        np.asarray([0.1, 0.5]),
        np.asarray([-2.0, 0.0]),
        np.asarray([-2.0, 0.0]),
    )
    keywords = {
        "start_time_s": 0.25,
        "end_time_s": 0.35,
        "root_interval_s": 0.1,
        "geometry_bbox_diagonal_m": 2.0,
        "geometry_rtol": 1.0e-12,
        "roundoff_ulps": 64,
    }

    certified = inspect_rk4_axis_piece(
        *arguments,
        **keywords,
        chord_deviation_bound_m=np.zeros(2, dtype="<f8"),
    )
    conservative = inspect_rk4_axis_piece(
        *arguments,
        **keywords,
        chord_deviation_bound_m=np.asarray([0.2, 0.0]),
    )

    assert certified.kind == "hit"
    assert certified.hit is not None
    assert conservative.kind == "split"
    assert conservative.hit is None


@pytest.mark.parametrize(
    "invalid_bound",
    [
        pytest.param(np.asarray([0.0]), id="shape"),
        pytest.param(np.asarray([-1.0, 0.0]), id="negative"),
        pytest.param(np.asarray([np.nan, 0.0]), id="nonfinite"),
    ],
)
def test_method_chord_bound_rejects_invalid_shared_invariants(
    tmp_path: Path,
    invalid_bound: np.ndarray,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-invalid-method-bound")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    with pytest.raises(ValueError, match="chord_deviation_bound_m"):
        inspect_rk4_piece(
            geometry,
            np.asarray([0.25, 0.5]),
            np.asarray([0.35, 0.5]),
            np.asarray([0.25, 0.5]),
            np.asarray([0.35, 0.5]),
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            start_time_s=0.0,
            end_time_s=0.1,
            root_interval_s=0.1,
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=invalid_bound,
        )
    with pytest.raises(ValueError, match="chord_deviation_bound_m"):
        inspect_rk4_axis_piece(
            np.asarray([0.1, 0.5]),
            np.asarray([-2.0, 0.0]),
            np.asarray([-0.1, 0.5]),
            np.asarray([-2.0, 0.0]),
            np.asarray([-0.1, 0.5]),
            np.asarray([0.1, 0.5]),
            np.asarray([-2.0, 0.0]),
            np.asarray([-2.0, 0.0]),
            start_time_s=0.25,
            end_time_s=0.35,
            root_interval_s=0.1,
            geometry_bbox_diagonal_m=2.0,
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=invalid_bound,
        )


def test_rk4_axis_start_is_folded_or_proved_departing() -> None:
    inward = inspect_rk4_axis_piece(
        np.asarray([0.0, 0.5]),
        np.asarray([-1.0, 0.0]),
        np.asarray([-0.1, 0.5]),
        np.asarray([-1.0, 0.0]),
        np.asarray([-0.1, 0.5]),
        np.asarray([0.0, 0.5]),
        np.asarray([-1.0, 0.0]),
        np.asarray([-1.0, 0.0]),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=0.1,
        geometry_bbox_diagonal_m=2.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    outward = inspect_rk4_axis_piece(
        np.asarray([0.0, 0.5]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.1, 0.5]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 0.5]),
        np.asarray([0.1, 0.5]),
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 0.0]),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=0.1,
        geometry_bbox_diagonal_m=2.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert inward.kind == "hit"
    assert inward.hit is not None
    assert inward.hit.time_s == 0.0
    assert outward.kind == "clear"
    assert outward.hit is None


def test_rk4_axis_invariant_and_positive_paths_are_clear() -> None:
    invariant = inspect_rk4_axis_piece(
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.1]),
        np.asarray([0.0, 1.0]),
        np.asarray([-0.1, 0.0]),
        np.asarray([0.1, 0.1]),
        np.asarray([-1.0, 1.0]),
        np.asarray([1.0, 1.0]),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=0.1,
        geometry_bbox_diagonal_m=2.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    positive_tube = inspect_rk4_axis_piece(
        np.asarray([0.2, 0.0]),
        np.asarray([-1.0, 0.0]),
        np.asarray([0.15, 0.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.1, 0.0]),
        np.asarray([0.3, 0.0]),
        np.asarray([-1.0, 0.0]),
        np.asarray([1.0, 0.0]),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=0.1,
        geometry_bbox_diagonal_m=2.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert invariant.kind == "clear"
    assert positive_tube.kind == "clear"


def test_rk4_axis_ambiguous_or_underresolved_crossing_splits() -> None:
    ambiguous = inspect_rk4_axis_piece(
        np.asarray([0.1, 0.0]),
        np.asarray([-1.0, 0.0]),
        np.asarray([-0.1, 0.0]),
        np.asarray([-1.0, 0.0]),
        np.asarray([-0.1, 0.0]),
        np.asarray([0.1, 0.0]),
        np.asarray([-2.0, 0.0]),
        np.asarray([1.0, 0.0]),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=0.1,
        geometry_bbox_diagonal_m=2.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    underresolved = inspect_rk4_axis_piece(
        np.asarray([0.1, 0.0]),
        np.asarray([-2.0, 0.0]),
        np.asarray([-0.1, 0.0]),
        np.asarray([-2.0, 0.0]),
        np.asarray([-0.1, 0.0]),
        np.asarray([0.1, 0.0]),
        np.asarray([-3.0, 0.0]),
        np.asarray([-1.0, 0.0]),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=0.1,
        geometry_bbox_diagonal_m=2.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert ambiguous.kind == "split"
    assert underresolved.kind == "split"


@pytest.mark.parametrize(
    ("start_position_m", "position_lower_m", "velocity_lower_m_s", "velocity_upper_m_s"),
    [
        pytest.param([0.0, 0.5], [0.0, 0.5], [-0.1, 0.0], [1.0, 0.0], id="not-inward"),
        pytest.param([0.1, 0.5], [0.0, 0.5], [1.0, 0.0], [1.0, 0.0], id="deep-interior"),
        pytest.param([0.0, 0.5], [0.0, 0.5], [1.0e-30, 0.0], [1.0e-30, 0.0], id="roundoff-band"),
    ],
)
def test_rk4_start_contact_departure_retains_unproven_candidates(
    tmp_path: Path,
    start_position_m: list[float],
    position_lower_m: list[float],
    velocity_lower_m_s: list[float],
    velocity_upper_m_s: list[float],
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-rk4-unproven-departure")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    decision = inspect_rk4_piece(
        geometry,
        np.asarray(start_position_m),
        np.asarray([0.1, 0.5]),
        np.asarray(position_lower_m),
        np.asarray([0.1, 0.5]),
        np.asarray(velocity_lower_m_s),
        np.asarray(velocity_upper_m_s),
        start_time_s=0.0,
        end_time_s=0.1,
        root_interval_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        certify_start_contact_departure=True,
    )

    assert decision.kind == "split"
    assert decision.hit is None
    assert decision.candidate_facet_count == 1
    assert not decision.start_contact_departure_certified


def test_intersection_beyond_interval_must_satisfy_time_budget(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-after-interval")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    hit = locate_ballistic_first_hit(
        geometry,
        np.asarray([1.0 - 2.0e-12, 0.5]),
        np.asarray([1.5e-12, 0.0]),
        start_time_s=0.0,
        end_time_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert hit is None


def test_remote_large_facets_do_not_change_local_point_or_parallel_path() -> None:
    geometry = prepare_geometry(_multiscale_geometry(), "cartesian_xy")

    classification = classify_event_point(
        geometry,
        np.asarray([0.5, 0.5]),
        speed_m_s=0.1,
        interval_s=1.0,
        time_s=0.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    hit = locate_ballistic_first_hit(
        geometry,
        np.asarray([0.5, 0.5]),
        np.asarray([0.1, 0.0]),
        start_time_s=0.0,
        end_time_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert classification == "inside"
    assert hit is None


def test_c10_corner_localization_reports_all_simultaneous_facets(tmp_path: Path) -> None:
    paths = materialize_microcase("C10", tmp_path / "C10")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    hit = locate_ballistic_first_hit(
        geometry,
        np.asarray([1.0, 0.25]),
        np.asarray([0.0, 1.0]),
        start_time_s=0.0,
        end_time_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert hit is not None
    assert hit.time_s == 0.75
    np.testing.assert_array_equal(hit.position_m, [1.0, 1.0])
    assert hit.facet_id == 1
    assert hit.candidate_facet_ids == (1, 2)


@pytest.mark.parametrize(
    ("start_x_m", "velocity_x_m_s", "end_time_s", "expected_time_s"),
    [
        pytest.param(0.5, 2.0, 2.0, 1.0 - 1.0 / math.sqrt(2.0), id="turning-chord-miss"),
        pytest.param(
            0.75 + 2.0**-20,
            1.0,
            1.0,
            0.5 - 2.0**-10,
            id="near-grazing-hit",
        ),
    ],
)
def test_constant_acceleration_finds_first_parabolic_hit(
    tmp_path: Path,
    start_x_m: float,
    velocity_x_m_s: float,
    end_time_s: float,
    expected_time_s: float,
) -> None:
    paths = materialize_microcase("C07", tmp_path / f"C07-{start_x_m}-{end_time_s}")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    hit = locate_constant_acceleration_first_hit(
        geometry,
        np.asarray([start_x_m, 0.5]),
        np.asarray([velocity_x_m_s, 0.0]),
        np.asarray([-2.0, 0.0]),
        start_time_s=0.0,
        end_time_s=end_time_s,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert hit is not None
    assert abs(hit.time_s - expected_time_s) <= hit.time_budget_s
    np.testing.assert_allclose(
        hit.position_m,
        [1.0, 0.5],
        rtol=0.0,
        atol=hit.position_budget_m,
    )
    assert hit.facet_id == 1
    assert hit.candidate_facet_ids == (1,)


@pytest.mark.parametrize(
    "start_x_m",
    [
        pytest.param(0.5, id="broad-phase-padding-is-not-a-hit"),
        pytest.param(0.75 - 2.0**-20, id="near-grazing-miss"),
    ],
)
def test_constant_acceleration_path_without_crossing_has_no_event(
    tmp_path: Path,
    start_x_m: float,
) -> None:
    paths = materialize_microcase("C07", tmp_path / f"C07-no-parabolic-hit-{start_x_m}")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    hit = locate_constant_acceleration_first_hit(
        geometry,
        np.asarray([start_x_m, 0.5]),
        np.asarray([1.0, 0.0]),
        np.asarray([-2.0, 0.0]),
        start_time_s=0.0,
        end_time_s=1.0,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )

    assert hit is None


def test_certified_quadratic_departure_omits_only_the_source_contact(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-accelerated-departure")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    hit = locate_constant_acceleration_first_hit(
        geometry,
        np.asarray([0.0, 0.5]),
        np.asarray([0.0, 0.0]),
        np.asarray([2.0, 0.0]),
        start_time_s=0.0,
        end_time_s=1.1,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        certified_departing_facet_id=3,
    )

    assert hit is not None
    assert hit.facet_id == 1
    assert abs(hit.time_s - 1.0) <= hit.time_budget_s
    np.testing.assert_allclose(
        hit.position_m,
        [1.0, 0.5],
        rtol=0.0,
        atol=hit.position_budget_m,
    )


@pytest.mark.parametrize(
    ("start_position_m", "velocity_m_s", "acceleration_m_s2"),
    [
        pytest.param([0.75, 0.5], [1.0, 0.0], [-2.0, 0.0], id="tangent"),
        pytest.param([0.25, 0.0], [0.5, 0.0], [1.0, 0.0], id="collinear"),
    ],
)
def test_indeterminate_parabolic_contact_is_not_silently_missed(
    tmp_path: Path,
    start_position_m: list[float],
    velocity_m_s: list[float],
    acceleration_m_s2: list[float],
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-indeterminate")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    with pytest.raises(EventLocationError):
        locate_constant_acceleration_first_hit(
            geometry,
            np.asarray(start_position_m),
            np.asarray(velocity_m_s),
            np.asarray(acceleration_m_s2),
            start_time_s=0.0,
            end_time_s=1.0,
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
        )


def _multiscale_geometry() -> GeometryData:
    nodes = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [100.0, 100.0],
            [1.0e13 + 100.0, 100.0],
            [1.0e13 + 100.0, 101.0],
            [100.0, 101.0],
        ],
        dtype="<f8",
    )
    edges = np.asarray(
        [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]],
        dtype="<i8",
    )
    count = edges.shape[0]
    boundary = BoundaryData(
        line2=edges,
        boundary_id=np.arange(count, dtype="<i4"),
        group_id=np.zeros(count, dtype="<i4"),
        material_id=np.zeros(count, dtype="<i4"),
        owner_cell_type=np.full(count, 2, dtype="<u1"),
        owner_cell_local_index=np.repeat(np.arange(2, dtype="<i8"), 4),
        orientation=np.ones(count, dtype="<i1"),
    )
    return GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("wall",),
        quad4=np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="<i8"),
        quad4_domain_id=np.zeros(2, dtype="<i4"),
    )
