from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from particle_tracer_unified.core._triangle_topology import _coordinate_topology_ids
from particle_tracer_unified.core.geometry3d import (
    build_triangle_surface,
    build_triangle_uniform_grid,
    point_inside_surface,
    segment_hit_from_surface,
    unresolved_triangle_indices,
    validate_closed_surface_triangles,
)
from particle_tracer_unified.domain import BoundaryHit
from particle_tracer_unified.solvers.collision_hit_localization import (
    _physical_hit_search_times,
    locate_physical_hit_state,
)


def _tetrahedron_triangles() -> np.ndarray:
    a = np.asarray([0.0, 0.0, 0.0])
    b = np.asarray([1.0, 0.0, 0.0])
    c = np.asarray([0.0, 1.0, 0.0])
    d = np.asarray([0.0, 0.0, 1.0])
    return np.asarray(
        [
            [a, c, b],
            [a, b, d],
            [a, d, c],
            [b, c, d],
        ],
        dtype=np.float64,
    )


def _cube_triangles() -> np.ndarray:
    corners = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    vertex_ids = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (3, 6, 2),
        (3, 7, 6),
        (0, 7, 3),
        (0, 4, 7),
    )
    return np.asarray(
        [[corners[a], corners[b], corners[c]] for a, b, c in vertex_ids],
        dtype=np.float64,
    )


def _time_tolerance(_reference: float, interval: float, fraction: float) -> float:
    return float(interval) * float(fraction)


def test_physical_hit_search_times_preserve_sorted_unique_probe_order() -> None:
    times = _physical_hit_search_times(
        1.0,
        np.asarray([0.75, -0.2, 1.2, 0.25, 0.25], dtype=np.float32),
        0.4,
    )

    assert times.dtype == np.float64
    np.testing.assert_array_equal(
        times,
        [0.2, 0.25, 0.4, 0.43000000000000005, 0.7, 0.75, 1.0],
    )


def test_hit_localization_preserves_search_refinement_and_projection_order() -> None:
    state_times: list[float] = []
    strict_positions: list[float] = []
    projection_positions: list[tuple[float, float]] = []
    tolerance_fractions: list[float] = []

    def state_at(partial_time: float) -> tuple[np.ndarray, np.ndarray]:
        state_times.append(float(partial_time))
        return (
            np.asarray([partial_time, 0.0], dtype=np.float32),
            np.asarray([1.0 + partial_time, -2.0], dtype=np.float32),
        )

    def strict_inside(point: np.ndarray) -> bool:
        strict_positions.append(float(point[0]))
        return bool(point[0] < 0.6)

    projected = BoundaryHit(
        position=np.asarray([0.6, 0.0], dtype=np.float32),
        normal=np.asarray([1.0, 0.0], dtype=np.float32),
        part_id=17,
        alpha_hint=0.0,
        primitive_id=8,
        primitive_kind="edge",
        is_ambiguous=True,
    )

    def nearest(point: np.ndarray, inside: np.ndarray) -> BoundaryHit | None:
        projection_positions.append((float(point[0]), float(inside[0])))
        return projected if len(projection_positions) == 2 else None

    def tolerance(reference: float, interval: float, fraction: float) -> float:
        del reference
        tolerance_fractions.append(float(fraction))
        return _time_tolerance(0.0, interval, fraction)

    event = locate_physical_hit_state(
        x0=np.asarray([0.0, 0.0], dtype=np.float32),
        v0=np.asarray([1.0, 0.0], dtype=np.float32),
        segment_dt=1.0,
        t_end_segment=2.0,
        stage_times=np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32),
        primary_hit=None,
        strict_inside_fn=strict_inside,
        nearest_projection_fn=nearest,
        state_at=state_at,
        time_tolerance=tolerance,
        on_boundary_tol_m=0.0,
        max_iters=2,
    )

    assert event is not None
    hit, velocity, hit_time = event
    assert tolerance_fractions == [0.0, 1.0e-6]
    assert state_times == pytest.approx(
        [0.125, 0.25, 0.375, 0.5, 0.625, 0.5625, 0.59375, 0.609375]
    )
    assert strict_positions == pytest.approx(state_times[:-1])
    assert projection_positions == pytest.approx(
        [(0.609375, 0.59375), (0.625, 0.59375)]
    )
    assert hit_time == pytest.approx(0.609375)
    assert hit.alpha_hint == pytest.approx(0.609375)
    assert hit.part_id == 17
    assert hit.primitive_id == 8
    assert hit.primitive_kind == "edge"
    assert hit.is_ambiguous
    assert hit.position.dtype == np.float64
    assert hit.normal.dtype == np.float64
    assert velocity.dtype == np.float64
    np.testing.assert_allclose(velocity, [1.609375, -2.0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("triangles", "message"),
    [
        (np.zeros((2, 3, 2), dtype=np.float64), r"shaped as \(n, 3, 3\)"),
        (np.empty((0, 3, 3), dtype=np.float64), "must be non-empty"),
        (
            np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            r"float64-unresolved triangle rows \[0\]",
        ),
    ],
)
def test_closed_surface_validation_preserves_input_error_order(
    triangles: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_closed_surface_triangles(triangles)


def test_unresolved_triangle_indices_preserves_2d_shape_and_dtype_contract() -> None:
    triangles = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ],
        dtype=np.float32,
    )

    unresolved = unresolved_triangle_indices(triangles)
    empty = unresolved_triangle_indices(np.empty((0, 3, 2), dtype=np.float32))

    assert unresolved.dtype == np.int64
    np.testing.assert_array_equal(unresolved, [1])
    assert empty.dtype == np.int64
    assert empty.shape == (0,)
    with pytest.raises(ValueError, match=r"shape \(n, 3, 2\|3\)"):
        unresolved_triangle_indices(np.zeros((1, 2, 3), dtype=np.float64))


def test_coordinate_topology_preserves_first_representative_and_ulp_merge() -> None:
    triangles = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [
                [np.nextafter(0.0, np.inf), 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ],
        dtype=np.float64,
    )

    ids, resolution, coordinate_roundoff, tolerance = _coordinate_topology_ids(
        triangles
    )

    assert ids.dtype == np.int64
    np.testing.assert_array_equal(ids, [[0, 1, 2], [0, 2, 3]])
    assert resolution == pytest.approx(1.0)
    assert 0.0 < coordinate_roundoff <= tolerance


def test_closed_surface_report_and_diagnostic_counts_are_stable() -> None:
    triangles = _tetrahedron_triangles()

    report = validate_closed_surface_triangles(triangles)

    assert list(report) == [
        "triangle_count",
        "unique_vertex_count",
        "edge_count",
        "identity_policy",
        "identity_resolution_m",
        "identity_coordinate_roundoff_m",
        "identity_tolerance_m",
    ]
    assert report["triangle_count"] == 4
    assert report["unique_vertex_count"] == 4
    assert report["edge_count"] == 6
    assert report["identity_resolution_m"] == pytest.approx(1.0)

    with pytest.raises(
        ValueError,
        match=r"3 edge\(s\) do not have exactly two adjacent triangles",
    ):
        validate_closed_surface_triangles(triangles[:-1])

    flipped = triangles.copy()
    flipped[0] = flipped[0, [0, 2, 1]]
    with pytest.raises(
        ValueError,
        match=r"3 edge\(s\) are not oppositely oriented",
    ):
        validate_closed_surface_triangles(flipped)


def test_triangle_uniform_grid_preserves_cell_and_candidate_order() -> None:
    grid = build_triangle_uniform_grid(
        _tetrahedron_triangles(),
        target_triangles_per_cell=1,
        min_cells_per_axis=2,
        max_cells_per_axis=4,
    )

    assert grid.dims == (2, 2, 2)
    np.testing.assert_array_equal(grid.origin, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(grid.cell_size, [0.5, 0.5, 0.5])
    assert [
        (cell, candidates.tolist())
        for cell, candidates in grid.cell_to_triangles.items()
    ] == [
        ((0, 0, 0), [0, 1, 2, 3]),
        ((0, 1, 0), [0, 2, 3]),
        ((1, 0, 0), [0, 1, 3]),
        ((1, 1, 0), [0, 3]),
        ((0, 0, 1), [1, 2, 3]),
        ((1, 0, 1), [1, 3]),
        ((0, 1, 1), [2, 3]),
        ((1, 1, 1), [3]),
    ]


def test_segment_hit_preserves_triangle_tie_and_empty_grid_fallback() -> None:
    triangles = _cube_triangles()
    surface = build_triangle_surface(
        triangles,
        np.arange(101, 113, dtype=np.int32),
    )
    fallback_surface = replace(
        surface,
        grid=replace(surface.grid, cell_to_triangles={}),
    )

    for candidate_surface in (surface, fallback_surface):
        hit = segment_hit_from_surface(
            candidate_surface,
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
            alpha_min=0.5,
            coordinate_tolerance_m=-1.0,
        )

        assert hit is not None
        position, normal, alpha, part_id, triangle_index = hit
        assert position.dtype == np.float64
        assert normal.dtype == np.float64
        np.testing.assert_array_equal(position, [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(normal, [1.0, 0.0, 0.0])
        assert alpha == pytest.approx(0.5)
        assert part_id == 107
        assert triangle_index == 6

        assert (
            segment_hit_from_surface(
                candidate_surface,
                np.asarray([0.0, 0.0, 0.0]),
                np.asarray([2.0, 0.0, 0.0]),
                alpha_min=np.nextafter(0.5, np.inf),
            )
            is None
        )


def test_point_inside_preserves_tolerance_and_empty_grid_fallback() -> None:
    surface = build_triangle_surface(
        _cube_triangles(),
        np.ones(12, dtype=np.int32),
    )
    fallback_surface = replace(
        surface,
        grid=replace(surface.grid, cell_to_triangles={}),
    )

    assert point_inside_surface(
        fallback_surface,
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        on_boundary_tol=0.0,
    ) == (True, False)
    assert point_inside_surface(
        fallback_surface,
        np.asarray([1.0 + 5.0e-8, 0.0, 0.0]),
        on_boundary_tol=1.0e-7,
    ) == (True, True)
    assert point_inside_surface(
        fallback_surface,
        np.asarray([1.0 + 5.0e-8, 0.0, 0.0]),
        on_boundary_tol=0.0,
    ) == (False, False)
    assert point_inside_surface(
        fallback_surface,
        np.asarray([1.0, 0.0, 0.0]),
        on_boundary_tol=-1.0,
    ) == (True, True)
