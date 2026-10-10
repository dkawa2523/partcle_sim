from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chamber_particles import load_case
from chamber_particles.case_format import BoundaryData, GeometryData
from chamber_particles.geometry import (
    GeometryPreparationError,
    centers_respect_contact_radius,
    classify_point,
    contact_normals_for_candidates,
    count_aabb_candidates,
    fill_aabb_candidates_csr,
    points_inside_volume,
    prepare_geometry,
    query_aabb_candidates,
    query_segment_candidates,
)
from tests.verification.microcases import materialize_microcase


def test_c07_square_has_outward_normals_and_queryable_physical_boundary(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07")
    case = load_case(paths.case_path)

    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    np.testing.assert_array_equal(
        geometry.facet_normal,
        [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    )
    np.testing.assert_array_equal(
        query_segment_candidates(
            geometry,
            np.asarray([0.25, 0.25]),
            np.asarray([1.25, 0.75]),
            padding_m=0.0,
        ),
        [1],
    )
    np.testing.assert_array_equal(
        query_segment_candidates(
            geometry,
            np.asarray([1.0, 1.0]),
            np.asarray([1.0, 1.0]),
            padding_m=0.0,
        ),
        [1, 2],
    )

    def budget(_facet_id: int) -> float:
        return 1.0e-12

    assert (
        classify_point(
            geometry,
            np.asarray([0.25, 0.25]),
            candidate_padding_m=1.0e-12,
            facet_position_budget_m=budget,
        )
        == "inside"
    )
    assert (
        classify_point(
            geometry,
            np.asarray([1.0, 0.5]),
            candidate_padding_m=1.0e-12,
            facet_position_budget_m=budget,
        )
        == "boundary"
    )
    assert (
        classify_point(
            geometry,
            np.asarray([1.1, 0.5]),
            candidate_padding_m=1.0e-12,
            facet_position_budget_m=budget,
        )
        == "outside"
    )


def test_finite_radius_clearance_and_candidate_normals_use_segment_distance(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C07", tmp_path / "C07-finite-contact")
    case = load_case(paths.case_path)
    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)

    positions = np.asarray([[0.5, 0.5], [0.8, 0.5], [0.9, 0.5]], dtype="<f8")
    radii = np.full(3, 0.2, dtype="<f8")
    np.testing.assert_array_equal(
        centers_respect_contact_radius(
            geometry,
            positions,
            radii,
            tolerance_m=1.0e-12,
            allow_contact=True,
        ),
        [True, True, False],
    )
    np.testing.assert_array_equal(
        centers_respect_contact_radius(
            geometry,
            positions,
            radii,
            tolerance_m=1.0e-12,
            allow_contact=False,
        ),
        [True, False, False],
    )

    endpoint_position = np.asarray([[0.8, 1.1]], dtype="<f8")
    normal = contact_normals_for_candidates(
        geometry,
        endpoint_position,
        np.asarray([math.hypot(0.2, 0.1)], dtype="<f8"),
        np.asarray([0, 1], dtype="<i8"),
        np.asarray([1], dtype="<i8"),
    )
    np.testing.assert_allclose(
        normal,
        [[2.0 / math.sqrt(5.0), -1.0 / math.sqrt(5.0)]],
        rtol=2.0e-15,
        atol=2.0e-15,
    )


def test_boundary_bvh_stackless_batch_matches_canonical_scalar_queries() -> None:
    facet_count = 16
    angle = 2.0 * math.pi * np.arange(facet_count, dtype=np.float64) / facet_count
    nodes = np.vstack((np.zeros((1, 2)), np.column_stack((np.cos(angle), np.sin(angle)))))
    triangles = np.asarray(
        [[0, index + 1, (index + 1) % facet_count + 1] for index in range(facet_count)],
        dtype="<i8",
    )
    edges = np.asarray(
        [[index + 1, (index + 1) % facet_count + 1] for index in range(facet_count)],
        dtype="<i8",
    )
    raw = GeometryData(
        nodes_m=np.asarray(nodes, dtype="<f8"),
        boundary=BoundaryData(
            line2=edges,
            boundary_id=np.arange(facet_count, dtype="<i4"),
            group_id=np.zeros(facet_count, dtype="<i4"),
            material_id=np.zeros(facet_count, dtype="<i4"),
            owner_cell_type=np.ones(facet_count, dtype="<u1"),
            owner_cell_local_index=np.arange(facet_count, dtype="<i8"),
            orientation=np.ones(facet_count, dtype="<i1"),
        ),
        group_names=("wall",),
        tri3=triangles,
        tri3_domain_id=np.zeros(facet_count, dtype="<i4"),
    )
    geometry = prepare_geometry(raw, "cartesian_xy")
    lower = np.asarray([[-2.0, -2.0], [0.8, -0.2], [-0.2, 0.8], [2.0, 2.0]])
    upper = np.asarray([[2.0, 2.0], [1.1, 0.2], [0.2, 1.1], [3.0, 3.0]])

    counts = count_aabb_candidates(geometry, lower, upper)
    offsets = np.empty(lower.shape[0] + 1, dtype="<i8")
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    candidates = np.empty(int(offsets[-1]), dtype="<i8")
    fill_aabb_candidates_csr(geometry, lower, upper, offsets, candidates)

    assert geometry.bvh_skip[0] == geometry.bvh_skip.size
    assert bool(np.all(geometry.bvh_skip > np.arange(geometry.bvh_skip.size)))
    for row in range(lower.shape[0]):
        expected = query_aabb_candidates(geometry, lower[row], upper[row])
        np.testing.assert_array_equal(candidates[offsets[row] : offsets[row + 1]], expected)
    with pytest.raises(ValueError, match="buffer capacity"):
        fill_aabb_candidates_csr(geometry, lower, upper, offsets, candidates[:-1])
    dense_count = int(counts[0])
    undercounted_offsets = np.asarray([0, dense_count - 1], dtype="<i8")
    with pytest.raises(GeometryPreparationError, match="counts changed"):
        fill_aabb_candidates_csr(
            geometry,
            lower[:1],
            upper[:1],
            undercounted_offsets,
            np.empty(dense_count - 1, dtype="<i8"),
        )


def test_volume_index_matches_convex_cell_union_for_mixed_cells_and_extreme_scales() -> None:
    mixed = GeometryData(
        nodes_m=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
            dtype="<f8",
        ),
        boundary=_boundary([]),
        group_names=(),
        tri3=np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<i8"),
        tri3_domain_id=np.zeros(2, dtype="<i4"),
        quad4=np.asarray([[1, 4, 5, 2]], dtype="<i8"),
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    points = np.asarray(
        [
            [0.25, 0.25],
            [1.5, 0.5],
            [1.0, 0.5],
            [1.0, 0.0],
            [np.nextafter(0.0, -np.inf), 0.5],
            [np.nextafter(2.0, np.inf), 0.5],
            [-1.0, 0.5],
            [3.0, 0.5],
        ],
        dtype="<f8",
    )
    expected = _reference_volume_contains(mixed, points)
    np.testing.assert_array_equal(expected, [True, True, True, True, False, False, False, False])
    np.testing.assert_array_equal(
        points_inside_volume(prepare_geometry(mixed, "cartesian_xy"), points), expected
    )

    origin = 1.0e9
    high_aspect = GeometryData(
        nodes_m=np.asarray(
            [
                [origin, origin],
                [origin + 1.0e6, origin],
                [origin + 1.0e6, origin + 1.0e-3],
                [origin, origin + 1.0e-3],
            ],
            dtype="<f8",
        ),
        boundary=_boundary([]),
        group_names=(),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    scaled_points = np.asarray(
        [
            [origin + 5.0e5, origin + 5.0e-4],
            [origin, origin],
            [np.nextafter(origin, -np.inf), origin + 5.0e-4],
            [origin + 5.0e5, np.nextafter(origin + 1.0e-3, np.inf)],
        ],
        dtype="<f8",
    )
    np.testing.assert_array_equal(
        points_inside_volume(prepare_geometry(high_aspect, "cartesian_xy"), scaled_points),
        _reference_volume_contains(high_aspect, scaled_points),
    )


def test_volume_union_preserves_hole_disconnected_domain_and_omitted_rz_axis_seam() -> None:
    nodes = np.asarray(
        [[float(radius), float(axial)] for axial in range(4) for radius in range(4)]
        + [[5.0, 1.0], [6.0, 1.0], [6.0, 2.0], [5.0, 2.0]],
        dtype="<f8",
    )

    def node_id(radius: int, axial: int) -> int:
        return 4 * axial + radius

    cells = [
        [
            node_id(radius, axial),
            node_id(radius + 1, axial),
            node_id(radius + 1, axial + 1),
            node_id(radius, axial + 1),
        ]
        for axial in range(3)
        for radius in range(3)
        if (radius, axial) != (1, 1)
    ]
    cells.append([16, 17, 18, 19])
    quad4 = np.asarray(cells, dtype="<i8")
    edge_owners: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for cell_id, cell in enumerate(quad4):
        for edge_index, start_value in enumerate(cell):
            start = int(start_value)
            end = int(cell[(edge_index + 1) % 4])
            key = (min(start, end), max(start, end))
            edge_owners.setdefault(key, []).append((cell_id, start, end))
    physical_facets = [
        owner
        for owners in edge_owners.values()
        if len(owners) == 1
        for owner in owners
        if not (nodes[owner[1], 0] == 0.0 and nodes[owner[2], 0] == 0.0)
    ]
    facet_count = len(physical_facets)
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=BoundaryData(
            line2=np.asarray([[start, end] for _, start, end in physical_facets], dtype="<i8"),
            boundary_id=np.full(facet_count, 10, dtype="<i4"),
            group_id=np.zeros(facet_count, dtype="<i4"),
            material_id=np.zeros(facet_count, dtype="<i4"),
            owner_cell_type=np.full(facet_count, 2, dtype="<u1"),
            owner_cell_local_index=np.asarray(
                [cell_id for cell_id, _, _ in physical_facets], dtype="<i8"
            ),
            orientation=np.ones(facet_count, dtype="<i1"),
        ),
        group_names=("wall",),
        quad4=quad4,
        quad4_domain_id=np.zeros(quad4.shape[0], dtype="<i4"),
    )
    prepared = prepare_geometry(geometry, "axisymmetric_rz")
    points = np.asarray(
        [[0.0, 1.5], [1.5, 1.5], [2.5, 1.5], [5.5, 1.5], [4.0, 1.5]],
        dtype="<f8",
    )
    np.testing.assert_array_equal(
        points_inside_volume(prepared, points),
        [True, False, True, True, False],
    )


def test_volume_preparation_rejects_unresolved_shape_but_not_large_coordinate_offset() -> None:
    minimum_subnormal = np.nextafter(0.0, np.inf)
    unresolved = GeometryData(
        nodes_m=np.asarray(
            [[0.0, -minimum_subnormal], [1.0, 0.0], [0.0, minimum_subnormal]],
            dtype="<f8",
        ),
        boundary=_boundary([]),
        group_names=(),
        tri3=np.asarray([[0, 1, 2]], dtype="<i8"),
        tri3_domain_id=np.zeros(1, dtype="<i4"),
    )
    unresolved_quad = GeometryData(
        nodes_m=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, minimum_subnormal], [0.0, 1.0]],
            dtype="<f8",
        ),
        boundary=_boundary([]),
        group_names=(),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    for geometry in (unresolved, unresolved_quad):
        with pytest.raises(
            GeometryPreparationError,
            match="volume cell is unresolved at float64 precision",
        ):
            prepare_geometry(geometry, "cartesian_xy")

    origin = float(2**52)
    translated = GeometryData(
        nodes_m=np.asarray(
            [
                [origin, origin],
                [origin + 1.0, origin],
                [origin + 1.0, origin + 1.0],
                [origin, origin + 1.0],
            ],
            dtype="<f8",
        ),
        boundary=_boundary([]),
        group_names=(),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    prepared = prepare_geometry(translated, "cartesian_xy")
    np.testing.assert_array_equal(
        points_inside_volume(
            prepared,
            np.asarray([[origin, origin], [origin - 1.0, origin]], dtype="<f8"),
        ),
        [True, False],
    )


@pytest.mark.parametrize("element", ("tri3", "quad4"))
def test_compiled_containment_preserves_scalar_hypot_at_stored_vertices(element: str) -> None:
    origin = np.asarray([2.0**40, 2.0**40], dtype="<f8")
    theta = 0.37
    long_edge = np.asarray([math.cos(theta), math.sin(theta)], dtype="<f8") * (2.0**20)
    short_edge = np.asarray([-math.sin(theta), math.cos(theta)], dtype="<f8") * 0.25
    if element == "tri3":
        nodes = np.asarray(
            [origin, origin + long_edge, origin + 0.37 * long_edge + short_edge],
            dtype="<f8",
        )
        geometry = GeometryData(
            nodes_m=nodes,
            boundary=_boundary([]),
            group_names=(),
            tri3=np.asarray([[0, 1, 2]], dtype="<i8"),
            tri3_domain_id=np.zeros(1, dtype="<i4"),
        )
    else:
        nodes = np.asarray(
            [origin, origin + long_edge, origin + long_edge + short_edge, origin + short_edge],
            dtype="<f8",
        )
        geometry = GeometryData(
            nodes_m=nodes,
            boundary=_boundary([]),
            group_names=(),
            quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
            quad4_domain_id=np.zeros(1, dtype="<i4"),
        )

    expected = _reference_volume_contains(geometry, nodes)
    assert bool(expected[1])
    np.testing.assert_array_equal(
        points_inside_volume(prepare_geometry(geometry, "cartesian_xy"), nodes),
        expected,
    )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("duplicate_boundary", "duplicate boundary edge"),
        ("missing_exterior", "missing a boundary"),
        ("internal_boundary", "internal volume edge"),
        ("non_manifold", "non-manifold volume edge"),
        ("crossing_boundary", "boundary self-intersection"),
        ("pinched_boundary", "non-manifold boundary vertex"),
    ],
)
def test_global_topology_errors_are_rejected(kind: str, message: str) -> None:
    with pytest.raises(GeometryPreparationError, match=message):
        prepare_geometry(_invalid_geometry(kind), "cartesian_xy")


def _invalid_geometry(kind: str) -> GeometryData:
    if kind == "non_manifold":
        nodes = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, -1.0], [0.5, 2.0]],
            dtype="<f8",
        )
        return GeometryData(
            nodes_m=nodes,
            boundary=_boundary([]),
            group_names=(),
            tri3=np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype="<i8"),
            tri3_domain_id=np.zeros(3, dtype="<i4"),
        )

    if kind == "crossing_boundary":
        nodes = np.asarray(
            [
                [-2.0, -1.0],
                [2.0, -1.0],
                [0.0, 2.0],
                [-2.0, 1.0],
                [0.0, -2.0],
                [2.0, 1.0],
            ],
            dtype="<f8",
        )
        cells = np.asarray([[0, 1, 2], [3, 4, 5]], dtype="<i8")
        edges = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]
        return GeometryData(
            nodes_m=nodes,
            boundary=_boundary(edges),
            group_names=("wall",),
            tri3=cells,
            tri3_domain_id=np.zeros(2, dtype="<i4"),
        )

    if kind == "pinched_boundary":
        nodes = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype="<f8",
        )
        cells = np.asarray([[0, 1, 2], [0, 3, 4]], dtype="<i8")
        edges = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]]
        return GeometryData(
            nodes_m=nodes,
            boundary=_boundary(edges),
            group_names=("wall",),
            tri3=cells,
            tri3_domain_id=np.zeros(2, dtype="<i4"),
        )

    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype="<f8")
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<i8")
    exterior = [[0, 1], [1, 2], [2, 3], [3, 0]]
    if kind == "duplicate_boundary":
        edges = [*exterior, [0, 1]]
    elif kind == "missing_exterior":
        edges = exterior[:-1]
    elif kind == "internal_boundary":
        edges = [*exterior, [0, 2]]
    else:
        raise AssertionError(f"unknown invalid-geometry kind: {kind}")
    return GeometryData(
        nodes_m=nodes,
        boundary=_boundary(edges),
        group_names=("wall",),
        tri3=cells,
        tri3_domain_id=np.zeros(2, dtype="<i4"),
    )


def _boundary(edges: list[list[int]]) -> BoundaryData:
    count = len(edges)
    return BoundaryData(
        line2=np.asarray(edges, dtype="<i8").reshape(count, 2),
        boundary_id=np.full(count, 10, dtype="<i4"),
        group_id=np.zeros(count, dtype="<i4"),
        material_id=np.zeros(count, dtype="<i4"),
        owner_cell_type=np.ones(count, dtype="<u1"),
        owner_cell_local_index=np.zeros(count, dtype="<i8"),
        orientation=np.ones(count, dtype="<i1"),
    )


def _reference_volume_contains(geometry: GeometryData, points_m: np.ndarray) -> np.ndarray:
    """Small scalar oracle for the pre-index containment predicate."""

    result = np.zeros(points_m.shape[0], dtype=np.bool_)
    for point_index, point in enumerate(points_m):
        for connectivity in (geometry.tri3, geometry.quad4):
            if connectivity is None:
                continue
            for cell in connectivity:
                cell_nodes = geometry.nodes_m[cell]
                contained = True
                for node_index, start in enumerate(cell_nodes):
                    end = cell_nodes[(node_index + 1) % cell_nodes.shape[0]]
                    edge = end - start
                    offset = point - start
                    length = math.hypot(float(edge[0]), float(edge[1]))
                    signed = math.fsum(
                        (
                            -float(edge[1]) / length * float(offset[0]),
                            float(edge[0]) / length * float(offset[1]),
                        )
                    )
                    if signed < 0.0:
                        contained = False
                        break
                if contained:
                    result[point_index] = True
                    break
            if result[point_index]:
                break
    return result
