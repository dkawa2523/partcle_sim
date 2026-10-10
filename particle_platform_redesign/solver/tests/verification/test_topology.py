from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from chamber_particles.case_format import (
    BoundaryData,
    FieldData,
    GeometryData,
    Q1QuadLayout,
    RegularLayout,
)
from chamber_particles.fields import (
    FieldLocationError,
    PreparedFieldSet,
    validate_periodic_field_seams,
)
from chamber_particles.geometry import PreparedGeometry, prepare_geometry
from chamber_particles.topology import (
    PreparedPeriodicTopology,
    TopologyPreparationError,
    TranslationPairRequest,
    prepare_periodic_topology,
)


def _periodic_square_geometry() -> GeometryData:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype="<f8")
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype="<i8"),
        boundary_id=np.asarray([10, 20, 30, 40], dtype="<i4"),
        group_id=np.asarray([0, 1, 2, 3], dtype="<i4"),
        material_id=np.zeros(4, dtype="<i4"),
        owner_cell_type=np.full(4, 2, dtype="<u1"),
        owner_cell_local_index=np.zeros(4, dtype="<i8"),
        orientation=np.ones(4, dtype="<i1"),
    )
    return GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("bottom", "right", "top", "left"),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.asarray([0], dtype="<i4"),
    )


def _split_periodic_square_geometry() -> GeometryData:
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5], [1.0, 1.0], [0.0, 1.0]],
        dtype="<f8",
    )
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 4], [4, 5], [5, 3], [3, 0]], dtype="<i8"),
        boundary_id=np.asarray([10, 20, 21, 30, 40, 41], dtype="<i4"),
        group_id=np.asarray([0, 1, 1, 2, 3, 3], dtype="<i4"),
        material_id=np.zeros(6, dtype="<i4"),
        owner_cell_type=np.full(6, 2, dtype="<u1"),
        owner_cell_local_index=np.asarray([0, 0, 1, 1, 1, 0], dtype="<i8"),
        orientation=np.ones(6, dtype="<i1"),
    )
    return GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("bottom", "right", "top", "left"),
        quad4=np.asarray([[0, 1, 2, 3], [3, 2, 4, 5]], dtype="<i8"),
        quad4_domain_id=np.asarray([0, 0], dtype="<i4"),
    )


def _prepared_topology() -> tuple[PreparedGeometry, PreparedPeriodicTopology]:
    geometry = prepare_geometry(_periodic_square_geometry(), "cartesian_xy")
    topology = prepare_periodic_topology(
        geometry,
        (TranslationPairRequest(3, 1, (1.0, 0.0)),),
        field_match_rtol=1.0e-12,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    assert topology is not None
    return geometry, topology


def test_translated_facets_form_one_reciprocal_read_only_map() -> None:
    geometry, topology = _prepared_topology()

    np.testing.assert_array_equal(topology.periodic_facet_id, [1, 3])
    assert topology.peer_facet_id[3] == 1
    assert topology.peer_facet_id[1] == 3
    np.testing.assert_array_equal(topology.peer_node_ids[3], [2, 1])
    np.testing.assert_array_equal(topology.peer_node_ids[1], [0, 3])
    np.testing.assert_array_equal(topology.translation_m[3], [1.0, 0.0])
    np.testing.assert_array_equal(topology.translation_m[1], [-1.0, 0.0])
    np.testing.assert_array_equal(topology.facet_is_periodic, [False, True, False, True])
    assert topology.pair_count == 1
    assert geometry.facet_contact_enabled.all()
    assert not topology.peer_facet_id.flags.writeable


def test_split_seams_pair_each_conforming_facet_deterministically() -> None:
    geometry = prepare_geometry(_split_periodic_square_geometry(), "cartesian_xy")
    topology = prepare_periodic_topology(
        geometry,
        (TranslationPairRequest(3, 1, (1.0, 0.0)),),
        field_match_rtol=1.0e-12,
        geometry_rtol=1.0e-12,
        roundoff_ulps=64,
    )
    assert topology is not None

    np.testing.assert_array_equal(topology.periodic_facet_id, [1, 2, 4, 5])
    assert topology.peer_facet_id[5] == 1
    assert topology.peer_facet_id[4] == 2
    assert topology.peer_facet_id[1] == 5
    assert topology.peer_facet_id[2] == 4


def test_nonconforming_or_nonopposite_facets_are_rejected() -> None:
    geometry = prepare_geometry(_periodic_square_geometry(), "cartesian_xy")
    with pytest.raises(TopologyPreparationError, match="do not match"):
        prepare_periodic_topology(
            geometry,
            (TranslationPairRequest(3, 1, (0.9, 0.0)),),
            field_match_rtol=1.0e-12,
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
        )

    normals = geometry.facet_normal.copy()
    normals[1] = normals[3]
    normals.setflags(write=False)
    inconsistent_normals = replace(geometry, facet_normal=normals)
    with pytest.raises(TopologyPreparationError, match="opposite outward normals"):
        prepare_periodic_topology(
            inconsistent_normals,
            (TranslationPairRequest(3, 1, (1.0, 0.0)),),
            field_match_rtol=1.0e-12,
            geometry_rtol=1.0e-12,
            roundoff_ulps=64,
        )


def test_q1_field_requires_matching_static_values_at_paired_nodes() -> None:
    geometry, topology = _prepared_topology()
    layout = Q1QuadLayout(
        "mesh",
        geometry.nodes_m,
        np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        np.ones(1, dtype="<u1"),
    )
    values = np.asarray([[2.0], [2.0], [3.0], [3.0]], dtype="<f8")
    field = FieldData("temperature", "mesh", "node", ("value",), "scalar", values, "K")
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    validate_periodic_field_seams(prepared, topology, geometry)

    mismatched = FieldData(
        "temperature",
        "mesh",
        "node",
        ("value",),
        "scalar",
        values + np.asarray([[0.0], [0.0], [0.0], [0.1]]),
        "K",
    )
    with pytest.raises(FieldLocationError, match="discontinuous"):
        validate_periodic_field_seams(
            PreparedFieldSet(layout, MappingProxyType({mismatched.name: mismatched})),
            topology,
            geometry,
        )


def test_regular_field_accepts_only_matching_static_opposite_support_faces() -> None:
    geometry, topology = _prepared_topology()
    axis = np.asarray([0.0, 0.5, 1.0], dtype="<f8")
    layout = RegularLayout("regular", axis, axis, np.ones((2, 2), dtype="<u1"))
    values = np.asarray(
        [[10.0 + y] for _x in axis for y in axis],
        dtype="<f8",
    )
    field = FieldData("temperature", "regular", "node", ("value",), "scalar", values, "K")
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    validate_periodic_field_seams(prepared, topology, geometry)

    mismatched_values = values.copy()
    mismatched_values[-1, 0] += 1.0
    mismatched = FieldData(
        "temperature",
        "regular",
        "node",
        ("value",),
        "scalar",
        mismatched_values,
        "K",
    )
    with pytest.raises(FieldLocationError, match="discontinuous"):
        validate_periodic_field_seams(
            PreparedFieldSet(layout, MappingProxyType({mismatched.name: mismatched})),
            topology,
            geometry,
        )


def test_time_dependent_required_field_is_explicitly_rejected() -> None:
    geometry, topology = _prepared_topology()
    layout = Q1QuadLayout(
        "mesh",
        geometry.nodes_m,
        np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        np.ones(1, dtype="<u1"),
    )
    snapshot = np.asarray([[2.0], [2.0], [3.0], [3.0]], dtype="<f8")
    field = FieldData(
        "temperature",
        "mesh",
        "node",
        ("value",),
        "scalar",
        np.stack((snapshot, snapshot)),
        "K",
        time_s=np.asarray([0.0, 1.0], dtype="<f8"),
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    with pytest.raises(FieldLocationError, match="does not support time-dependent"):
        validate_periodic_field_seams(prepared, topology, geometry)


def test_cell_associated_required_field_is_explicitly_rejected() -> None:
    geometry, topology = _prepared_topology()
    layout = Q1QuadLayout(
        "mesh",
        geometry.nodes_m,
        np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        np.ones(1, dtype="<u1"),
    )
    field = FieldData(
        "temperature",
        "mesh",
        "cell",
        ("value",),
        "scalar",
        np.asarray([[2.0]], dtype="<f8"),
        "K",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    with pytest.raises(FieldLocationError, match="requires nodal"):
        validate_periodic_field_seams(prepared, topology, geometry)
