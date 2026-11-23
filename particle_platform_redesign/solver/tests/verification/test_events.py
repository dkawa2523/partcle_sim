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
    EVENT_QUERY_CANDIDATE_CAPACITY,
    EVENT_QUERY_INVALID_RANGE,
    EVENT_QUERY_OK,
    EVENT_QUERY_ROW_CAPACITY,
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
    allocate_event_arena,
    certify_event_point_interiors,
    classify_event_point,
    classify_surface_release_batch,
    curved_event_result_view,
    exact_event_result_view,
    locate_curved_first_event_batch,
    locate_curved_first_event_into,
    locate_exact_first_event_batch,
    locate_exact_first_event_into,
    prepare_curved_event_batch,
    prepare_event_geometry,
    prepare_exact_event_batch,
    resolve_event_budget,
)
from chamber_particles.geometry import points_inside_volume, prepare_geometry
from tests.verification.microcases import materialize_microcase


@pytest.mark.parametrize(
    ("scale_m", "shift_m"),
    ((1.0e-9, 0.0), (1.0, 0.0), (1.0e9, 0.0), (1.0, 1.0e9)),
)
def test_event_interior_certificates_respect_boundary_budgets_across_scales(
    tmp_path: Path,
    scale_m: float,
    shift_m: float,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "point-classification")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(
        replace(case.data.geometry, nodes_m=case.data.geometry.nodes_m * scale_m + shift_m),
        "cartesian_xy",
    )
    right = shift_m + scale_m
    position = np.asarray(
        [
            [shift_m + 0.5 * scale_m, shift_m + 0.5 * scale_m],
            [right, shift_m + 0.5 * scale_m],
            [np.nextafter(right, -np.inf), shift_m + 0.5 * scale_m],
            [shift_m + 1.1 * scale_m, shift_m + 0.5 * scale_m],
            [shift_m + 0.25 * scale_m, shift_m + 0.5 * scale_m],
        ],
        dtype="<f8",
    )
    velocity = np.zeros_like(position)
    time = np.arange(position.shape[0], dtype="<f8")
    eligible = np.asarray([True, True, True, True, False])
    contained = points_inside_volume(geometry, position)
    certified = certify_event_point_interiors(
        geometry,
        position,
        velocity,
        time,
        eligible,
        contained,
        interval_s=0.25,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    np.testing.assert_array_equal(certified, [True, False, False, False, False])
    classifications = [
        classify_event_point(
            geometry,
            position[row],
            speed_m_s=0.0,
            interval_s=0.25,
            time_s=float(time[row]),
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
            volume_containment=bool(contained[row]),
        )
        for row in range(position.shape[0])
    ]
    assert classifications == ["inside", "boundary", "boundary", "outside", "inside"]


@pytest.mark.parametrize("overflow", ("speed", "time"))
def test_event_interior_certificate_does_not_accept_invalid_scalar_budgets(
    tmp_path: Path,
    overflow: str,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "budget-overflow")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, "cartesian_xy")
    position = np.asarray([[0.5, 0.5]], dtype="<f8")
    velocity = np.asarray([[1.7e308, 1.7e308] if overflow == "speed" else [0.0, 0.0]])
    time = np.asarray([0.0 if overflow == "speed" else 1.0e308])
    ulps = 64 if overflow == "speed" else 2**53
    certified = certify_event_point_interiors(
        geometry,
        position,
        velocity,
        time,
        np.ones(1, dtype=np.bool_),
        np.ones(1, dtype=np.bool_),
        interval_s=0.25,
        geometry_rtol=1.0e-12,
        roundoff_ulps=ulps,
    )
    assert not certified[0]
    with pytest.raises(ValueError if overflow == "speed" else EventLocationError):
        classify_event_point(
            geometry,
            position[0],
            speed_m_s=math.hypot(*velocity[0]),
            interval_s=0.25,
            time_s=float(time[0]),
            geometry_rtol=1.0e-12,
            roundoff_ulps=ulps,
            volume_containment=True,
        )


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


def test_exact_line_and_quadratic_hits_match_analytic_times_and_bounded_partitions(
    tmp_path: Path,
) -> None:
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

    prepared = prepare_exact_event_batch(
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
        certified_departing_facet_id=departing,
    )
    preparation_failure = prepared.path.failure_reason.copy()
    actual = locate_exact_first_event_batch(
        geometry,
        prepared,
        candidate_capacity=geometry.facet_count * path_kind.size,
    )

    assert actual.status.tolist() == [
        EXACT_STATUS_WALL,
        EXACT_STATUS_CLEAR,
        EXACT_STATUS_WALL,
        EXACT_STATUS_CLEAR,
        EXACT_STATUS_WALL,
        EXACT_STATUS_FAILURE,
    ]
    # Unit-square intersections derived from x=x0+v*t+a*t^2/2.
    for row, hit_time, hit_z in ((0, 1.5, 0.625), (2, 1.0 - math.sqrt(0.5), 0.5), (4, 1.0, 0.5)):
        assert abs(actual.time_s[row] - hit_time) <= actual.time_budget_s[row]
        np.testing.assert_allclose(
            actual.position_m[row], [1.0, hit_z], rtol=0, atol=actual.position_budget_m[row]
        )
        assert actual.primary_facet_id[row] == 1
        np.testing.assert_array_equal(actual.normal[row], [1.0, 0.0])
        assert actual.localization_residual_m[row] <= actual.position_budget_m[row]
        begin, end = actual.candidate_offsets[row : row + 2]
        assert actual.candidate_facet_ids[begin:end].tolist() == [1]
    assert actual.failure_reason[5] == EXACT_FAILURE_INDETERMINATE_EVENT
    arena = allocate_event_arena(
        path_kind.size, geometry.facet_count * path_kind.size, exact_enabled=True
    )
    context = prepare_event_geometry(geometry)
    for begin, end in ((0, 2), (2, 5), (5, 6), (6, 6)):
        status, rows, candidates, _ = locate_exact_first_event_into(
            context, prepared, arena, begin, end
        )
        assert status == EVENT_QUERY_OK
        assert rows == end - begin
        prefix = exact_event_result_view(arena, rows, candidates)
        for name in (
            "status",
            "failure_reason",
            "time_s",
            "position_m",
            "primary_facet_id",
            "normal",
            "position_budget_m",
            "time_budget_s",
            "localization_residual_m",
        ):
            np.testing.assert_array_equal(
                getattr(prefix, name).view(np.uint8),
                getattr(actual, name)[begin:end].view(np.uint8),
            )
        first, last = actual.candidate_offsets[[begin, end]]
        np.testing.assert_array_equal(
            prefix.candidate_offsets, actual.candidate_offsets[begin : end + 1] - first
        )
        np.testing.assert_array_equal(
            prefix.candidate_facet_ids, actual.candidate_facet_ids[first:last]
        )
    np.testing.assert_array_equal(prepared.path.failure_reason, preparation_failure)


def test_prepared_exact_candidates_fail_closed_before_dense_batch_exceeds_capacity(
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
    prepared = prepare_exact_event_batch(
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

    np.testing.assert_array_equal(prepared.broad_counts, [4, 4, 4])
    with pytest.raises(ExactCandidateCapacityError) as batch_error:
        locate_exact_first_event_batch(
            geometry,
            prepared,
            candidate_capacity=11,
        )
    assert batch_error.value.required_count == 12
    assert batch_error.value.capacity == 11
    assert batch_error.value.oversized_row is None

    with pytest.raises(ExactCandidateCapacityError) as row_error:
        locate_exact_first_event_batch(
            geometry,
            prepared,
            candidate_capacity=3,
            row_start=0,
            row_stop=1,
        )
    assert row_error.value.required_count == 4
    assert row_error.value.oversized_row == 0

    fitted = locate_exact_first_event_batch(
        geometry,
        prepared,
        candidate_capacity=12,
    )
    np.testing.assert_array_equal(fitted.status, np.full(row_count, EXACT_STATUS_WALL))
    np.testing.assert_array_equal(fitted.candidate_offsets, [0, 2, 4, 6])
    np.testing.assert_array_equal(fitted.candidate_facet_ids, [1, 2, 1, 2, 1, 2])
    context = prepare_event_geometry(geometry)
    for rows, capacity, begin, end, query_status in (
        (2, 12, 0, 3, EVENT_QUERY_ROW_CAPACITY),
        (3, 8, 0, 3, EVENT_QUERY_CANDIDATE_CAPACITY),
        (3, 12, -1, 1, EVENT_QUERY_INVALID_RANGE),
        (3, 12, 2, 1, EVENT_QUERY_INVALID_RANGE),
        (3, 12, 0, 4, EVENT_QUERY_INVALID_RANGE),
    ):
        arena = allocate_event_arena(rows, capacity, exact_enabled=True)
        status, count, candidates, _ = locate_exact_first_event_into(context, prepared, arena, 0, 1)
        assert status == EVENT_QUERY_OK
        result = exact_event_result_view(arena, count, candidates)
        columns = (
            result.status,
            result.failure_reason,
            result.time_s,
            result.position_m,
            result.primary_facet_id,
            result.normal,
            result.position_budget_m,
            result.time_budget_s,
            result.localization_residual_m,
            result.candidate_offsets,
            result.candidate_facet_ids,
        )
        before = tuple(column.tobytes() for column in columns)
        status, count, candidates, _ = locate_exact_first_event_into(
            context, prepared, arena, begin, end
        )
        assert status == query_status
        assert count == candidates == 0
        assert tuple(column.tobytes() for column in columns) == before


def test_prepared_exact_prefixes_preserve_center_and_surface_contact(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-exact-contact-prefix")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    finite_mask = geometry.facet_contact_enabled.copy()
    finite_mask[1] = False
    kinds = np.asarray(
        [EXACT_PATH_LINEAR, EXACT_PATH_LINEAR, EXACT_PATH_LINEAR, EXACT_PATH_QUADRATIC],
        dtype=np.uint8,
    )
    position = np.repeat([[0.5, 0.5]], 4, axis=0)
    velocity = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    acceleration = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    prepared = prepare_exact_event_batch(
        geometry,
        kinds,
        position,
        velocity,
        acceleration,
        start_time_s=np.zeros(4),
        target_time_s=np.ones(4),
        contact_radius_m=np.asarray([0.0, 0.1, 0.1, 0.1]),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        finite_contact_enabled=finite_mask,
    )
    full = locate_exact_first_event_batch(
        geometry, prepared, candidate_capacity=int(prepared.broad_counts.sum())
    )
    np.testing.assert_array_equal(full.status, np.full(4, EXACT_STATUS_WALL))
    np.testing.assert_allclose(full.time_s, [0.5, 0.5, 0.4, math.sqrt(0.4)], rtol=0.0, atol=3.0e-12)
    np.testing.assert_array_equal(full.primary_facet_id, [1, 1, 2, 2])
    arena = allocate_event_arena(
        4, int(prepared.broad_counts.sum()), exact_enabled=True, separate_finite_mask=True
    )
    context = prepare_event_geometry(geometry)
    for begin, end in ((0, 4), (0, 2), (2, 4), (3, 4), (1, 2), (0, 1)):
        status, rows, candidates, _ = locate_exact_first_event_into(
            context, prepared, arena, begin, end
        )
        assert status == EVENT_QUERY_OK
        assert rows == end - begin
        prefix = exact_event_result_view(arena, rows, candidates)
        for name in (
            "status",
            "failure_reason",
            "time_s",
            "position_m",
            "primary_facet_id",
            "normal",
            "position_budget_m",
            "time_budget_s",
            "localization_residual_m",
        ):
            np.testing.assert_array_equal(
                getattr(prefix, name).view(np.uint8),
                getattr(full, name)[begin:end].view(np.uint8),
            )
        first, last = full.candidate_offsets[[begin, end]]
        np.testing.assert_array_equal(
            prefix.candidate_offsets, full.candidate_offsets[begin : end + 1] - first
        )
        np.testing.assert_array_equal(
            prefix.candidate_facet_ids, full.candidate_facet_ids[first:last]
        )


def test_compiled_exact_batch_preserves_corner_candidate_identity(tmp_path: Path) -> None:
    paths = materialize_microcase("C10", tmp_path / "C10-exact-batch")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    actual = locate_exact_first_event_batch(
        geometry,
        prepare_exact_event_batch(
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
        ),
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
        prepare_exact_event_batch(
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
        ),
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


def test_finite_radius_exact_preserves_zero_path_clear_failure_and_axis_decisions(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-zero-path")
    raw = load_case(paths.case_path).data.geometry
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
    prepared = prepare_exact_event_batch(
        geometry,
        np.full(4, EXACT_PATH_LINEAR, dtype=np.uint8),
        np.asarray([[0.5, 0.5]] * 3 + [[0.0, 0.5]], dtype="<f8"),
        np.asarray([[0.0, 0.0], [1.0, 0.0], [np.finfo(np.float64).max, 0.0], [-1.0, 0.0]]),
        np.zeros((4, 2), dtype="<f8"),
        start_time_s=np.zeros(4, dtype="<f8"),
        target_time_s=np.asarray([1.0, 1.0, 2.0, 1.0]),
        contact_radius_m=np.full(4, 0.1, dtype="<f8"),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    actual = locate_exact_first_event_batch(
        geometry, prepared, candidate_capacity=4 * geometry.facet_count
    )

    np.testing.assert_array_equal(
        actual.status,
        [
            EXACT_STATUS_CLEAR,
            EXACT_STATUS_WALL,
            EXACT_STATUS_FAILURE,
            EXACT_STATUS_AXIS,
        ],
    )
    np.testing.assert_array_equal(
        actual.failure_reason, [0, 0, EXACT_FAILURE_INDETERMINATE_EVENT, 0]
    )
    np.testing.assert_allclose(actual.time_s[[1, 3]], [0.4, 0.0], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(actual.position_m[[1, 3]], [[0.9, 0.5], [0.0, 0.5]], atol=2.0e-12)
    np.testing.assert_array_equal(actual.primary_facet_id[[1, 3]], [1, -1])
    np.testing.assert_array_equal(np.diff(actual.candidate_offsets), [0, 1, 0, 0])


def test_finite_radius_exact_corner_keeps_both_simultaneous_facets(tmp_path: Path) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-corner")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    actual = locate_exact_first_event_batch(
        geometry,
        prepare_exact_event_batch(
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
        ),
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
        prepare_exact_event_batch(
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
            certified_departing_facet_id=np.asarray(
                [EXACT_DEPARTURE_FINITE_CONTACT_SET, EXACT_DEPARTURE_FINITE_CONTACT_SET],
                dtype="<i8",
            ),
        ),
        candidate_capacity=2 * geometry.facet_count,
    )
    curved = locate_curved_first_event_batch(
        geometry,
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            certify_start_contact_departure=np.asarray([True]),
            chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
        ),
        candidate_capacity=geometry.facet_count,
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
        prepare_exact_event_batch(
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
            certified_departing_facet_id=np.asarray(
                [EXACT_DEPARTURE_FINITE_CONTACT_SET],
                dtype="<i8",
            ),
        ),
        candidate_capacity=geometry.facet_count,
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
        prepare_exact_event_batch(
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
        ),
        candidate_capacity=geometry.facet_count * 2,
    )
    expected = locate_exact_first_event_batch(
        geometry,
        prepare_exact_event_batch(
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
        ),
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
        prepare_exact_event_batch(
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
        ),
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
        prepare_exact_event_batch(
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
        ),
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
        prepare_exact_event_batch(
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
        ),
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


def test_curved_wall_hits_match_analytic_times_and_bounded_partitions(tmp_path: Path) -> None:
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

    prepared = prepare_curved_event_batch(
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
        geometry_rtol=1e-12,
        roundoff_ulps=64,
        certify_start_contact_departure=departure,
        chord_deviation_bound_m=chord_bound,
    )
    actual = locate_curved_first_event_batch(
        geometry, prepared, candidate_capacity=geometry.facet_count * start.shape[0]
    )
    # Capacity partitioning must retain each row verdict and the complete
    # incident facet set; a second projection from state is not required.
    for row in range(start.shape[0]):
        prefix = locate_curved_first_event_batch(
            geometry,
            prepared,
            candidate_capacity=geometry.facet_count,
            row_start=row,
            row_stop=row + 1,
        )
        for column in (
            "status",
            "failure_reason",
            "start_contact_departure_certified",
            "time_s",
            "position_m",
            "primary_facet_id",
            "normal",
            "position_budget_m",
            "time_budget_s",
            "localization_residual_m",
        ):
            np.testing.assert_array_equal(
                getattr(prefix, column).view(np.uint8),
                getattr(actual, column)[row : row + 1].view(np.uint8),
            )
        begin, stop = actual.candidate_offsets[row : row + 2]
        np.testing.assert_array_equal(
            prefix.candidate_facet_ids, actual.candidate_facet_ids[begin:stop]
        )

    np.testing.assert_array_equal(
        actual.status,
        [CURVED_STATUS_WALL, CURVED_STATUS_SPLIT, CURVED_STATUS_CLEAR, CURVED_STATUS_WALL],
    )
    np.testing.assert_array_equal(
        actual.start_contact_departure_certified, [False, False, False, True]
    )
    for row, hit_time in ((0, 0.250001), (3, 1.0)):
        assert abs(actual.time_s[row] - hit_time) <= actual.time_budget_s[row]
        np.testing.assert_array_equal(actual.position_m[row], [1.0, 0.5])
        assert actual.primary_facet_id[row] == 1
        assert actual.localization_residual_m[row] <= actual.position_budget_m[row]


def test_curved_default_chord_bound_certifies_the_analytic_crossing(tmp_path: Path) -> None:
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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
        ),
        candidate_capacity=geometry.facet_count,
    )
    assert actual.status[0] == CURVED_STATUS_WALL
    assert abs(actual.time_s[0] - 0.250001) <= actual.time_budget_s[0]
    np.testing.assert_array_equal(actual.position_m[0], [1.0, 0.5])
    assert actual.localization_residual_m[0] <= actual.position_budget_m[0]


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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            certify_monotone_approach=certify_monotone,
        ),
        candidate_capacity=geometry.facet_count,
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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            certify_monotone_approach=True,
        ),
        candidate_capacity=geometry.facet_count,
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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=np.zeros((2, 2), dtype="<f8"),
            use_position_controls=np.asarray([True, False]),
            position_control_origin_m=control_origin,
            relative_position_control_lower_m=relative_controls,
            relative_position_control_upper_m=relative_controls,
        ),
        candidate_capacity=2 * geometry.facet_count,
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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=np.zeros((2, 2), dtype="<f8"),
            position_control_origin_m=start,
            relative_position_control_lower_m=relative_controls,
            relative_position_control_upper_m=relative_controls,
        ),
        candidate_capacity=2 * geometry.facet_count,
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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
            position_control_origin_m=start,
            relative_position_control_lower_m=relative_controls,
            relative_position_control_upper_m=relative_controls,
        ),
        candidate_capacity=geometry.facet_count,
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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
        ),
        candidate_capacity=geometry.facet_count,
    )

    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_WALL])
    np.testing.assert_allclose(actual.time_s, [0.55], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(actual.position_m, [[0.8, 0.5]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1])


@pytest.mark.parametrize("center_opening", [False, True])
def test_prepared_curved_prefixes_preserve_contact_modes_and_incident_sets(
    tmp_path: Path, center_opening: bool
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-prepared-prefix")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start = np.asarray([[0.25, 0.5], [0.25, 0.5], [0.5, 0.25], [0.25, 0.5]])
    end = np.asarray([[1.25, 0.5], [1.25, 0.5], [0.5, 1.25], [0.35, 0.5]])
    velocity = end - start
    finite_mask = geometry.facet_contact_enabled.copy()
    if center_opening:
        finite_mask[1] = False
    controls = np.asarray([fraction * velocity for fraction in (0.0, 1 / 3, 2 / 3, 1.0)])
    controls = controls.transpose(1, 0, 2).copy()
    prepared = prepare_curved_event_batch(
        geometry,
        start,
        velocity,
        end,
        velocity,
        np.minimum(start, end),
        np.maximum(start, end),
        velocity,
        velocity,
        start_time_s=np.zeros(4),
        target_time_s=np.ones(4),
        root_interval_s=np.ones(4),
        contact_radius_m=np.asarray([0.0, 0.2, 0.2, 0.2]),
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
        chord_deviation_bound_m=np.zeros((4, 2)),
        certify_monotone_approach=True,
        use_position_controls=np.asarray([True, False, True, False]),
        position_control_origin_m=start,
        relative_position_control_lower_m=controls,
        relative_position_control_upper_m=controls,
        finite_contact_enabled=finite_mask,
    )
    actual = locate_curved_first_event_batch(
        geometry, prepared, candidate_capacity=4 * geometry.facet_count
    )
    np.testing.assert_array_equal(
        actual.status,
        [CURVED_STATUS_WALL, CURVED_STATUS_WALL, CURVED_STATUS_WALL, CURVED_STATUS_CLEAR],
    )
    np.testing.assert_allclose(
        actual.time_s[:3],
        [0.75, 0.75 if center_opening else 0.55, 0.55],
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 1, 2, 3, 3])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1, 1, 2])
    arena = allocate_event_arena(4, 4 * geometry.facet_count, separate_finite_mask=center_opening)
    context = prepare_event_geometry(geometry)
    for first, last in ((0, 4), (0, 2), (2, 4), (3, 4), (1, 2), (0, 1)):
        status, rows, candidates, _ = locate_curved_first_event_into(
            context, prepared, arena, first, last
        )
        assert status == EVENT_QUERY_OK
        assert rows == last - first
        prefix = curved_event_result_view(arena, rows, candidates)
        for column in (
            "status",
            "failure_reason",
            "start_contact_departure_certified",
            "time_s",
            "position_m",
            "primary_facet_id",
            "normal",
            "position_budget_m",
            "time_budget_s",
            "localization_residual_m",
        ):
            np.testing.assert_array_equal(
                getattr(prefix, column).view(np.uint8),
                getattr(actual, column)[first:last].view(np.uint8),
            )
        begin, stop = actual.candidate_offsets[[first, last]]
        np.testing.assert_array_equal(
            prefix.candidate_offsets, actual.candidate_offsets[first : last + 1] - begin
        )
        np.testing.assert_array_equal(
            prefix.candidate_facet_ids, actual.candidate_facet_ids[begin:stop]
        )


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

    prepared = prepare_curved_event_batch(
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
        geometry_rtol=1e-12,
        roundoff_ulps=64,
    )
    np.testing.assert_array_equal(prepared.broad_counts, [4, 4])
    with pytest.raises(CurvedCandidateCapacityError) as batch_error:
        locate_curved_first_event_batch(geometry, prepared, candidate_capacity=7)
    assert batch_error.value.required_count == 8
    assert batch_error.value.oversized_row is None
    with pytest.raises(CurvedCandidateCapacityError) as row_error:
        locate_curved_first_event_batch(geometry, prepared, candidate_capacity=3, row_stop=1)
    assert row_error.value.required_count == 4
    assert row_error.value.oversized_row == 0
    context = prepare_event_geometry(geometry)
    for row_capacity, query_status in (
        (1, EVENT_QUERY_ROW_CAPACITY),
        (2, EVENT_QUERY_CANDIDATE_CAPACITY),
    ):
        arena = allocate_event_arena(row_capacity, 4)
        status, rows, candidates, _ = locate_curved_first_event_into(context, prepared, arena, 0, 1)
        assert status == EVENT_QUERY_OK
        result = curved_event_result_view(arena, rows, candidates)
        columns = (
            result.status,
            result.failure_reason,
            result.start_contact_departure_certified,
            result.time_s,
            result.position_m,
            result.primary_facet_id,
            result.normal,
            result.position_budget_m,
            result.time_budget_s,
            result.localization_residual_m,
            result.candidate_offsets,
            result.candidate_facet_ids,
        )
        before = tuple(column.tobytes() for column in columns)
        status, rows, candidates, _ = locate_curved_first_event_into(context, prepared, arena, 0, 2)
        assert status == query_status
        assert rows == candidates == 0
        assert tuple(column.tobytes() for column in columns) == before


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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=np.zeros((3, 2), dtype="<f8"),
        ),
        candidate_capacity=geometry.facet_count * start.shape[0],
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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
        ),
        candidate_capacity=geometry.facet_count,
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
        prepare_curved_event_batch(
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
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
            use_position_controls=np.asarray([True]),
            position_control_origin_m=start,
            relative_position_control_lower_m=relative_controls,
            relative_position_control_upper_m=relative_controls,
        ),
        candidate_capacity=geometry.facet_count,
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
        prepare_curved_event_batch(
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
            target_time_s=np.asarray([1e-12], dtype="<f8"),
            root_interval_s=np.asarray([1.0], dtype="<f8"),
            contact_radius_m=np.zeros(1, dtype="<f8"),
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=np.zeros((1, 2), dtype="<f8"),
        ),
        candidate_capacity=geometry.facet_count,
    )

    node_distance_m = float(np.linalg.norm(actual.position_m[0] - np.asarray([1.0, 1.0])))
    np.testing.assert_array_equal(actual.status, [CURVED_STATUS_WALL])
    np.testing.assert_array_equal(actual.candidate_offsets, [0, 2])
    np.testing.assert_array_equal(actual.candidate_facet_ids, [1, 2])
    assert node_distance_m > 0.0
    assert node_distance_m <= actual.position_budget_m[0]


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


@pytest.mark.parametrize(
    (
        "start",
        "end",
        "lower",
        "upper",
        "v_lower",
        "v_upper",
        "duration",
        "chord",
        "departure",
        "expected",
        "hit_time",
    ),
    [
        pytest.param(
            [0.9, 0.5],
            [0.9, 0.5],
            [0.85, 0.5],
            [1.02, 0.5],
            [-0.44, 0],
            [0.44, 0],
            1.0,
            None,
            False,
            CURVED_STATUS_SPLIT,
            None,
            id="exit-and-return",
        ),
        pytest.param(
            [1 - 1e-6, 0.5],
            [1 + 1e-6, 0.5],
            [1 - 1e-6, 0.5 - 1e-13],
            [1 + 1e-6, 0.5 + 1e-13],
            [1 - 5e-8, -5e-8],
            [1 + 5e-8, 5e-8],
            2e-6,
            [0, 0],
            False,
            CURVED_STATUS_WALL,
            1e-6,
            id="method-chord-hit",
        ),
        pytest.param(
            [1 - 1e-6, 0.5],
            [1 + 1e-6, 0.5],
            [1 - 1e-6, 0.5 - 1e-13],
            [1 + 1e-6, 0.5 + 1e-13],
            [1 - 5e-8, -5e-8],
            [1 + 5e-8, 5e-8],
            2e-6,
            [1e-4, 1e-4],
            False,
            CURVED_STATUS_SPLIT,
            None,
            id="method-chord-uncertain",
        ),
        pytest.param(
            [1 - 1e-6, 0.5],
            [1 + 1e-6, 0.5],
            [1 - 1e-6, 0.5 - 1e-13],
            [1 + 1e-6, 0.5 + 1e-13],
            [1 - 5e-8, -2e-6],
            [1 + 5e-8, 2e-6],
            2e-6,
            None,
            False,
            CURVED_STATUS_SPLIT,
            None,
            id="tangential-uncertainty",
        ),
        pytest.param(
            [1 - 2e-13, 0.5],
            [1 + 2e-13, 0.5],
            [1 - 2e-13, 0.5],
            [1 + 2e-13, 0.5],
            [2, 0],
            [2, 0],
            2e-13,
            [1e-4, 1e-4],
            False,
            CURVED_STATUS_WALL,
            None,
            id="small-full-tube",
        ),
        pytest.param(
            [1 - 2e-12, 0.25],
            [1 + 2e-12, 0.35],
            [1 - 2e-12, 0.25],
            [1 + 2e-12, 0.35],
            [3.9e-11, 1],
            [4.1e-11, 1],
            0.1,
            None,
            False,
            CURVED_STATUS_SPLIT,
            None,
            id="near-parallel",
        ),
        pytest.param(
            [0, 0.5],
            [1.1, 0.5],
            [0, 0.5],
            [1.1, 0.5],
            [1, 0],
            [1, 0],
            1.1,
            [0, 0],
            True,
            CURVED_STATUS_WALL,
            1.0,
            id="departure-then-opposite-hit",
        ),
        pytest.param(
            [0, 0.5],
            [0.1, 0.5],
            [0, 0.5],
            [0.1, 0.5],
            [-0.1, 0],
            [1, 0],
            0.1,
            None,
            True,
            CURVED_STATUS_SPLIT,
            None,
            id="unproven-inward-departure",
        ),
        pytest.param(
            [0.1, 0.5],
            [0.1, 0.5],
            [0, 0.5],
            [0.1, 0.5],
            [1, 0],
            [1, 0],
            0.1,
            None,
            True,
            CURVED_STATUS_SPLIT,
            None,
            id="not-on-source-facet",
        ),
        pytest.param(
            [0, 0.5],
            [0.1, 0.5],
            [0, 0.5],
            [0.1, 0.5],
            [1e-30, 0],
            [1e-30, 0],
            0.1,
            None,
            True,
            CURVED_STATUS_SPLIT,
            None,
            id="roundoff-band-departure",
        ),
    ],
)
def test_curved_certificates_keep_turning_grazing_uncertainty_and_departure(
    tmp_path: Path,
    start,
    end,
    lower,
    upper,
    v_lower,
    v_upper,
    duration,
    chord,
    departure,
    expected,
    hit_time,
) -> None:
    case = load_case(materialize_microcase("C07", tmp_path / "curved-certificates").case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    velocity = (np.asarray([v_lower], dtype="<f8") + np.asarray([v_upper], dtype="<f8")) / 2
    prepared = prepare_curved_event_batch(
        geometry,
        np.asarray([start], dtype="<f8"),
        velocity,
        np.asarray([end], dtype="<f8"),
        velocity,
        np.asarray([lower], dtype="<f8"),
        np.asarray([upper], dtype="<f8"),
        np.asarray([v_lower], dtype="<f8"),
        np.asarray([v_upper], dtype="<f8"),
        start_time_s=np.zeros(1),
        target_time_s=np.asarray([duration]),
        root_interval_s=np.ones(1),
        contact_radius_m=np.zeros(1),
        geometry_rtol=1e-12,
        roundoff_ulps=64,
        certify_start_contact_departure=np.asarray([departure]),
        chord_deviation_bound_m=None if chord is None else np.asarray([chord], dtype="<f8"),
    )
    arena = allocate_event_arena(1, geometry.facet_count)
    status, rows, candidates, _ = locate_curved_first_event_into(
        prepare_event_geometry(geometry), prepared, arena, 0, 1
    )
    assert status == EVENT_QUERY_OK
    result = curved_event_result_view(arena, rows, candidates)
    assert result.status[0] == expected
    if expected == CURVED_STATUS_WALL:
        assert result.primary_facet_id[0] == 1
        assert result.candidate_facet_ids.tolist() == [1]
        assert result.localization_residual_m[0] <= result.position_budget_m[0]
        if hit_time is not None:
            assert abs(result.time_s[0] - hit_time) <= result.time_budget_s[0]
    assert bool(result.start_contact_departure_certified[0]) == (
        departure and expected == CURVED_STATUS_WALL
    )


@pytest.mark.parametrize(
    (
        "start_r",
        "end_r",
        "start_vr",
        "end_vr",
        "lower_r",
        "upper_r",
        "v_lower_r",
        "v_upper_r",
        "chord_r",
        "expected",
        "hit_time",
    ),
    [
        pytest.param(
            0.1,
            -0.1,
            -2,
            -2,
            -0.1,
            0.1,
            -2,
            -2,
            0,
            CURVED_STATUS_AXIS,
            0.05,
            id="certified-crossing",
        ),
        pytest.param(
            0.1,
            -0.1,
            -2,
            -2,
            -0.1,
            0.1,
            -2,
            -2,
            0.2,
            CURVED_STATUS_SPLIT,
            None,
            id="uncertain-chord",
        ),
        pytest.param(
            0, -0.1, -1, -1, -0.1, 0, -1, -1, 0, CURVED_STATUS_AXIS, 0, id="inward-axis-resident"
        ),
        pytest.param(
            0, 0.1, 1, 1, 0, 0.1, 1, 1, 0, CURVED_STATUS_CLEAR, None, id="outward-axis-resident"
        ),
        pytest.param(
            0, 0, 0, 0, -0.1, 0.1, -1, 1, 0, CURVED_STATUS_CLEAR, None, id="invariant-axis-path"
        ),
        pytest.param(
            0.2, 0.15, -1, 1, 0.1, 0.3, -1, 1, 0, CURVED_STATUS_CLEAR, None, id="positive-tube"
        ),
        pytest.param(
            0.1,
            -0.1,
            -1,
            -1,
            -0.1,
            0.1,
            -2,
            1,
            None,
            CURVED_STATUS_SPLIT,
            None,
            id="ambiguous-direction",
        ),
        pytest.param(
            0.1,
            -0.1,
            -2,
            -2,
            -0.1,
            0.1,
            -3,
            -1,
            None,
            CURVED_STATUS_SPLIT,
            None,
            id="underresolved-crossing",
        ),
    ],
)
def test_curved_axis_certificates_use_the_current_signed_chart(
    tmp_path: Path,
    start_r,
    end_r,
    start_vr,
    end_vr,
    lower_r,
    upper_r,
    v_lower_r,
    v_upper_r,
    chord_r,
    expected,
    hit_time,
) -> None:
    case = load_case(materialize_microcase("C07", tmp_path / "axis-certificates").case_path)
    boundary = case.data.geometry.boundary
    # The left rectangle edge is the coordinate axis, not a material wall.
    raw_boundary = replace(
        boundary,
        **{
            name: getattr(boundary, name)[:3].copy()
            for name in (
                "line2",
                "boundary_id",
                "group_id",
                "material_id",
                "owner_cell_type",
                "owner_cell_local_index",
                "orientation",
            )
        },
    )
    geometry = prepare_geometry(
        replace(case.data.geometry, boundary=raw_boundary), "axisymmetric_rz"
    )
    prepared = prepare_curved_event_batch(
        geometry,
        np.asarray([[start_r, 0.5]]),
        np.asarray([[start_vr, 0.0]]),
        np.asarray([[end_r, 0.5]]),
        np.asarray([[end_vr, 0.0]]),
        np.asarray([[lower_r, 0.5]]),
        np.asarray([[upper_r, 0.5]]),
        np.asarray([[v_lower_r, 0.0]]),
        np.asarray([[v_upper_r, 0.0]]),
        start_time_s=np.zeros(1),
        target_time_s=np.full(1, 0.1),
        root_interval_s=np.full(1, 0.1),
        contact_radius_m=np.zeros(1),
        geometry_rtol=1e-12,
        roundoff_ulps=64,
        chord_deviation_bound_m=None if chord_r is None else np.asarray([[chord_r, 0.0]]),
    )
    arena = allocate_event_arena(1, geometry.facet_count)
    status, rows, candidates, _ = locate_curved_first_event_into(
        prepare_event_geometry(geometry), prepared, arena, 0, 1
    )
    assert status == EVENT_QUERY_OK
    result = curved_event_result_view(arena, rows, candidates)
    assert result.status[0] == expected
    assert result.candidate_facet_ids.size == 0
    if hit_time is not None:
        assert abs(result.time_s[0] - hit_time) <= result.time_budget_s[0]
        assert result.localization_residual_m[0] <= result.position_budget_m[0]


@pytest.mark.parametrize(
    "invalid_bound", [np.asarray([0.0]), np.asarray([[-1.0, 0.0]]), np.asarray([[np.nan, 0.0]])]
)
def test_curved_preparation_rejects_invalid_method_chord_bounds(
    tmp_path: Path, invalid_bound
) -> None:
    case = load_case(materialize_microcase("C07", tmp_path / "invalid-bound").case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    start, end, velocity = (
        np.asarray([[0.25, 0.5]]),
        np.asarray([[0.35, 0.5]]),
        np.asarray([[1.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="chord_deviation_bound_m"):
        prepare_curved_event_batch(
            geometry,
            start,
            velocity,
            end,
            velocity,
            start,
            end,
            velocity,
            velocity,
            start_time_s=np.zeros(1),
            target_time_s=np.full(1, 0.1),
            root_interval_s=np.full(1, 0.1),
            contact_radius_m=np.zeros(1),
            geometry_rtol=1e-12,
            roundoff_ulps=64,
            chord_deviation_bound_m=invalid_bound,
        )


@pytest.mark.parametrize("scenario", ("after_interval", "distant_large_facet"))
def test_exact_current_path_keeps_locality_and_requested_time_interval(
    tmp_path: Path, scenario: str
) -> None:
    if scenario == "distant_large_facet":
        geometry = prepare_geometry(_multiscale_geometry(), "cartesian_xy")
        start = np.asarray([[0.5, 0.5]], dtype="<f8")
        velocity = np.asarray([[0.1, 0.0]], dtype="<f8")
        assert (
            classify_event_point(
                geometry,
                start[0],
                speed_m_s=0.1,
                interval_s=1.0,
                time_s=0.0,
                geometry_rtol=1e-12,
                roundoff_ulps=64,
            )
            == "inside"
        )
    else:
        case = load_case(materialize_microcase("C07", tmp_path / "time-interval").case_path)
        geometry = prepare_geometry(case.data.geometry, "cartesian_xy")
        # The wall crossing is at about 4/3 s, beyond this 1 s interval.
        # Being within the position budget does not authorize a late hit.
        start = np.asarray([[1.0 - 2e-12, 0.5]], dtype="<f8")
        velocity = np.asarray([[1.5e-12, 0.0]], dtype="<f8")
    prepared = prepare_exact_event_batch(
        geometry,
        np.asarray([EXACT_PATH_LINEAR], dtype="<u1"),
        start,
        velocity,
        np.zeros((1, 2), dtype="<f8"),
        start_time_s=np.zeros(1, dtype="<f8"),
        target_time_s=np.ones(1, dtype="<f8"),
        contact_radius_m=np.zeros(1, dtype="<f8"),
        geometry_rtol=1e-12,
        roundoff_ulps=64,
    )
    arena = allocate_event_arena(1, geometry.facet_count, exact_enabled=True)
    status, rows, candidates, _ = locate_exact_first_event_into(
        prepare_event_geometry(geometry), prepared, arena, 0, 1
    )
    assert status == EVENT_QUERY_OK
    result = exact_event_result_view(arena, rows, candidates)
    assert result.status.tolist() == [EXACT_STATUS_CLEAR]
