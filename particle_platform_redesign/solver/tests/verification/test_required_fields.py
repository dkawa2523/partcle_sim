from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    Q1QuadLayout,
    RegularLayout,
)
from chamber_particles.fields import (
    FIELD_LOCATION_ULPS,
    FieldLocationError,
    PreparedFieldSet,
    RequiredFieldMetadata,
    locate_field_cell,
    prepare_required_fields,
    sample_field,
)


def _boundary() -> BoundaryData:
    return BoundaryData(
        line2=np.empty((0, 2), dtype="<i8"),
        boundary_id=np.empty(0, dtype="<i4"),
        group_id=np.empty(0, dtype="<i4"),
        material_id=np.empty(0, dtype="<i4"),
        owner_cell_type=np.empty(0, dtype="<u1"),
        owner_cell_local_index=np.empty(0, dtype="<i8"),
        orientation=np.empty(0, dtype="<i1"),
    )


def _closed_quad_boundary() -> BoundaryData:
    return BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype="<i8"),
        boundary_id=np.full(4, 10, dtype="<i4"),
        group_id=np.zeros(4, dtype="<i4"),
        material_id=np.zeros(4, dtype="<i4"),
        owner_cell_type=np.full(4, 2, dtype="<u1"),
        owner_cell_local_index=np.zeros(4, dtype="<i8"),
        orientation=np.ones(4, dtype="<i1"),
    )


def _quad_geometry() -> GeometryData:
    return GeometryData(
        nodes_m=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype="<f8"),
        boundary=_boundary(),
        group_names=(),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.asarray([0], dtype="<i4"),
    )


def _tri_geometry() -> GeometryData:
    return GeometryData(
        nodes_m=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="<f8"),
        boundary=_boundary(),
        group_names=(),
        tri3=np.asarray([[0, 1, 2]], dtype="<i8"),
        tri3_domain_id=np.asarray([0], dtype="<i4"),
    )


def _bundle(
    geometry: GeometryData,
    layouts: tuple[RegularLayout | P1TriLayout | Q1QuadLayout, ...],
    fields: tuple[FieldData, ...],
    *,
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"] = "cartesian_xy",
) -> DataBundle:
    provenance = json.dumps(
        {
            "producer": "required-field-verification",
            "producer_version": "1",
            "source_sha256": f"sha256:{'0' * 64}",
            "field_semantics_revision": "required-field-verification-v1",
            "producer_metadata": {},
        }
    )
    return DataBundle(coordinate_system, provenance, geometry, layouts, fields)


def _regular_layout(name: str = "regular") -> RegularLayout:
    return RegularLayout(
        name,
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([[1]], dtype="<u1"),
    )


def _scalar_field(
    *,
    name: str = "gas_density",
    layout: str = "regular",
    association: str = "node",
    components: tuple[str, ...] = ("value",),
    stored_basis: str = "scalar",
    unit: str = "kg/m^3",
    node_count: int = 4,
) -> FieldData:
    row_count = node_count if association == "node" else 1
    return FieldData(
        name,
        layout,
        association,  # type: ignore[arg-type]
        components,
        stored_basis,
        np.ones((row_count, len(components)), dtype="<f8"),
        unit,
    )


def _unstructured_strip(
    kind: Literal["p1", "q1"], cell_columns: int = 12
) -> tuple[DataBundle, P1TriLayout | Q1QuadLayout, FieldData]:
    x = np.arange(cell_columns + 1, dtype=np.float64)
    nodes = np.empty((2 * (cell_columns + 1), 2), dtype="<f8")
    nodes[: cell_columns + 1] = np.column_stack((x, np.zeros_like(x)))
    nodes[cell_columns + 1 :] = np.column_stack((x, np.ones_like(x)))
    column = np.arange(cell_columns, dtype=np.int64)
    if kind == "p1":
        connectivity = np.empty((2 * cell_columns, 3), dtype="<i8")
        connectivity[0::2] = np.column_stack((column, column + 1, cell_columns + 2 + column))
        connectivity[1::2] = np.column_stack(
            (column, cell_columns + 2 + column, cell_columns + 1 + column)
        )
        geometry = GeometryData(
            nodes_m=nodes,
            boundary=_boundary(),
            group_names=(),
            tri3=connectivity,
            tri3_domain_id=np.zeros(connectivity.shape[0], dtype="<i4"),
        )
        layout: P1TriLayout | Q1QuadLayout = P1TriLayout(
            "unstructured",
            nodes.copy(),
            connectivity.copy(),
            np.ones(connectivity.shape[0], dtype="<u1"),
        )
    else:
        connectivity = np.column_stack(
            (
                column,
                column + 1,
                cell_columns + 2 + column,
                cell_columns + 1 + column,
            )
        ).astype("<i8")
        geometry = GeometryData(
            nodes_m=nodes,
            boundary=_boundary(),
            group_names=(),
            quad4=connectivity,
            quad4_domain_id=np.zeros(connectivity.shape[0], dtype="<i4"),
        )
        layout = Q1QuadLayout(
            "unstructured",
            nodes.copy(),
            connectivity.copy(),
            np.ones(connectivity.shape[0], dtype="<u1"),
        )
    field = FieldData(
        "gas_density",
        layout.name,
        "node",
        ("value",),
        "scalar",
        (2.0 + nodes[:, 0] + 0.25 * nodes[:, 1])[:, None],
        "kg/m^3",
    )
    return _bundle(geometry, (layout,), (field,)), layout, field


_DENSITY_REQUIREMENT = RequiredFieldMetadata("kg/m^3", ("value",), "scalar", True)
_RZ_VECTOR_REQUIREMENT = RequiredFieldMetadata("m/s", ("r", "z"), "axisymmetric_rz", False)


@pytest.mark.parametrize("kind", ["regular", "p1", "q1"])
@pytest.mark.parametrize("invalid_axis_value", [False, True])
def test_required_rz_vector_field_is_regular_on_the_accessible_axis(
    kind: str, invalid_axis_value: bool
) -> None:
    geometry = _tri_geometry() if kind == "p1" else _quad_geometry()
    layout: RegularLayout | P1TriLayout | Q1QuadLayout
    if kind == "regular":
        layout = _regular_layout()
        field_nodes = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype="<f8")
    elif kind == "p1":
        assert geometry.tri3 is not None
        layout = P1TriLayout(
            "unstructured",
            geometry.nodes_m.copy(),
            geometry.tri3.copy(),
            np.asarray([1], dtype="<u1"),
        )
        field_nodes = layout.nodes_m
    else:
        assert geometry.quad4 is not None
        layout = Q1QuadLayout(
            "unstructured",
            geometry.nodes_m.copy(),
            geometry.quad4.copy(),
            np.asarray([1], dtype="<u1"),
        )
        field_nodes = layout.nodes_m
    values = np.column_stack((field_nodes[:, 0], np.ones(field_nodes.shape[0])))
    if invalid_axis_value:
        values[np.flatnonzero(field_nodes[:, 0] == 0.0)[0], 0] = np.nextafter(0.0, 1.0)
    field = FieldData(
        "gas_velocity",
        layout.name,
        "node",
        ("r", "z"),
        "axisymmetric_rz",
        values,
        "m/s",
    )
    data = _bundle(
        geometry,
        (layout,),
        (field,),
        coordinate_system="axisymmetric_rz",
    )

    if invalid_axis_value:
        with pytest.raises(FieldLocationError, match="zero radial component on the axis"):
            prepare_required_fields(data, {"gas_velocity": _RZ_VECTOR_REQUIREMENT})
    else:
        prepared = prepare_required_fields(data, {"gas_velocity": _RZ_VECTOR_REQUIREMENT})
        assert prepared.layout is layout
        sampled = prepared.sample(np.asarray([[0.0, 0.25]], dtype="<f8"))
        assert sampled.values["gas_velocity"][0, 0] == 0.0


def test_required_rz_vector_preserves_exact_axis_regularity_on_skewed_p1() -> None:
    nodes = np.asarray([[0.1, 0.3], [0.0, 1.1], [0.0, 0.2]], dtype="<f8")
    connectivity = np.asarray([[0, 1, 2]], dtype="<i8")
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=_boundary(),
        group_names=(),
        tri3=connectivity,
        tri3_domain_id=np.asarray([0], dtype="<i4"),
    )
    layout = P1TriLayout(
        "skewed-p1",
        nodes.copy(),
        connectivity.copy(),
        np.asarray([1], dtype="<u1"),
    )
    field = FieldData(
        "gas_velocity",
        layout.name,
        "node",
        ("r", "z"),
        "axisymmetric_rz",
        np.column_stack((nodes[:, 0], np.ones(3, dtype="<f8"))),
        "m/s",
    )
    prepared = prepare_required_fields(
        _bundle(
            geometry,
            (layout,),
            (field,),
            coordinate_system="axisymmetric_rz",
        ),
        {"gas_velocity": _RZ_VECTOR_REQUIREMENT},
    )

    sampled = prepared.sample(np.asarray([[0.0, 0.317], [0.05, 0.317]], dtype="<f8"))

    assert sampled.values["gas_velocity"][0, 0] == 0.0
    assert sampled.values["gas_velocity"][1, 0] > 0.0


def test_required_rz_vector_does_not_constrain_a_material_blocked_layout_axis() -> None:
    base = _quad_geometry()
    geometry = GeometryData(
        nodes_m=base.nodes_m + np.asarray([1.0, 0.0]),
        boundary=_closed_quad_boundary(),
        group_names=("wall",),
        quad4=base.quad4,
        quad4_domain_id=base.quad4_domain_id,
    )
    layout = RegularLayout(
        "regular",
        np.asarray([0.0, 1.0, 2.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((2, 1), dtype="<u1"),
    )
    field = FieldData(
        "gas_velocity",
        layout.name,
        "node",
        ("r", "z"),
        "axisymmetric_rz",
        np.ones((6, 2), dtype="<f8"),
        "m/s",
    )
    data = _bundle(
        geometry,
        (layout,),
        (field,),
        coordinate_system="axisymmetric_rz",
    )

    prepared = prepare_required_fields(data, {"gas_velocity": _RZ_VECTOR_REQUIREMENT})

    assert prepared.layout is layout
    assert not prepared.axis_accessible


def test_required_rz_vector_checks_a_boundaryless_regular_support_axis() -> None:
    base = _quad_geometry()
    geometry = GeometryData(
        nodes_m=base.nodes_m + np.asarray([1.0, 0.0]),
        boundary=base.boundary,
        group_names=base.group_names,
        quad4=base.quad4,
        quad4_domain_id=base.quad4_domain_id,
    )
    layout = RegularLayout(
        "regular",
        np.asarray([0.0, 1.0, 2.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((2, 1), dtype="<u1"),
    )
    field = FieldData(
        "gas_velocity",
        layout.name,
        "node",
        ("r", "z"),
        "axisymmetric_rz",
        np.ones((6, 2), dtype="<f8"),
        "m/s",
    )
    data = _bundle(
        geometry,
        (layout,),
        (field,),
        coordinate_system="axisymmetric_rz",
    )

    with pytest.raises(FieldLocationError, match="zero radial component on the axis"):
        prepare_required_fields(data, {"gas_velocity": _RZ_VECTOR_REQUIREMENT})


@pytest.mark.parametrize(
    "make_field, error",
    [
        (lambda: _scalar_field(association="cell"), "node-associated"),
        (lambda: _scalar_field(unit="g/cm^3"), "metadata"),
        (lambda: _scalar_field(components=("density",)), "metadata"),
        (lambda: _scalar_field(stored_basis="cartesian_xy"), "metadata"),
    ],
)
def test_required_field_binding_requires_exact_canonical_metadata(
    make_field: Callable[[], FieldData], error: str
) -> None:
    data = _bundle(_quad_geometry(), (_regular_layout(),), (make_field(),))

    with pytest.raises(FieldLocationError, match=error):
        prepare_required_fields(data, {"gas_density": _DENSITY_REQUIREMENT})


def test_required_fields_must_share_one_layout() -> None:
    layouts = (_regular_layout("gas"), _regular_layout("electric"))
    fields = (
        _scalar_field(layout="gas"),
        _scalar_field(name="potential", layout="electric", unit="V"),
    )
    requirements = {
        "gas_density": _DENSITY_REQUIREMENT,
        "potential": RequiredFieldMetadata("V", ("value",), "scalar", False),
    }

    with pytest.raises(FieldLocationError, match="one common layout"):
        prepare_required_fields(_bundle(_quad_geometry(), layouts, fields), requirements)


def test_required_regular_layout_must_cover_every_geometry_node() -> None:
    layout = RegularLayout(
        "regular",
        np.asarray([0.0, 0.75], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([[1]], dtype="<u1"),
    )
    data = _bundle(_quad_geometry(), (layout,), (_scalar_field(),))

    with pytest.raises(FieldLocationError, match="does not cover every particle-domain node"):
        prepare_required_fields(data, {"gas_density": _DENSITY_REQUIREMENT})


def test_required_regular_layout_rejects_shared_unresolved_axis_during_prepare() -> None:
    lower = 1.0e200
    upper = float(np.nextafter(lower, np.inf))
    nodes = np.asarray(
        [[lower, 0.0], [upper, 0.0], [upper, 1.0], [lower, 1.0]],
        dtype="<f8",
    )
    geometry = replace(_quad_geometry(), nodes_m=nodes)
    layout = RegularLayout(
        "regular",
        np.asarray([lower, upper], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([[1]], dtype="<u1"),
    )
    data = _bundle(geometry, (layout,), (_scalar_field(),))

    with pytest.raises(FieldLocationError, match=r"regular axis0.*unresolved"):
        prepare_required_fields(data, {"gas_density": _DENSITY_REQUIREMENT})


@pytest.mark.parametrize("kind", ["p1", "q1"])
def test_required_unstructured_layout_accepts_exact_full_support_mesh(kind: str) -> None:
    geometry = _tri_geometry() if kind == "p1" else _quad_geometry()
    connectivity = geometry.tri3 if kind == "p1" else geometry.quad4
    assert connectivity is not None
    layout: P1TriLayout | Q1QuadLayout
    if kind == "p1":
        layout = P1TriLayout(
            "unstructured",
            geometry.nodes_m.copy(),
            connectivity.copy(),
            np.asarray([1], dtype="<u1"),
        )
    else:
        layout = Q1QuadLayout(
            "unstructured",
            geometry.nodes_m.copy(),
            connectivity.copy(),
            np.asarray([1], dtype="<u1"),
        )
    field = _scalar_field(layout="unstructured", node_count=int(geometry.nodes_m.shape[0]))

    prepared = prepare_required_fields(
        _bundle(geometry, (layout,), (field,)), {"gas_density": _DENSITY_REQUIREMENT}
    )

    assert prepared.layout is layout


@pytest.mark.parametrize("kind", ["p1", "q1"])
def test_prepared_unstructured_index_matches_full_search_for_initial_crossing_and_outside(
    kind: Literal["p1", "q1"],
) -> None:
    data, layout, field = _unstructured_strip(kind)
    prepared = prepare_required_fields(data, {"gas_density": _DENSITY_REQUIREMENT})
    rng = np.random.default_rng(1400)
    random_points = np.column_stack((rng.uniform(0.05, 11.95, 48), rng.uniform(0.05, 0.95, 48)))
    boundary_x = np.nextafter(np.float64(6.0), np.inf)
    positions = np.vstack(
        (
            random_points,
            np.asarray(
                [
                    [11.75, 0.25],
                    [boundary_x, 0.25],
                    [6.0, 0.0],
                    [12.25, 0.5],
                ],
                dtype="<f8",
            ),
        )
    )
    hints = np.zeros(positions.shape[0], dtype="<i8")
    hints[:24] = -1

    sampled = prepared.sample(positions, hints)
    expected_support = np.empty(positions.shape[0], dtype=np.bool_)
    expected_cell = np.empty(positions.shape[0], dtype="<i8")
    expected_value = np.empty((positions.shape[0], 1), dtype="<f8")
    for row, position in enumerate(positions):
        location = locate_field_cell(layout, position, cell_hint=int(hints[row]))
        expected_support[row] = location.support_inside
        expected_cell[row] = location.cell_id
        expected_value[row] = sample_field(field, location).value

    assert prepared.prepared_nbytes > 0
    assert prepared.preparation_transient_nbytes > prepared.prepared_nbytes
    np.testing.assert_array_equal(sampled.support_inside, expected_support)
    np.testing.assert_array_equal(sampled.cell_id, expected_cell)
    np.testing.assert_allclose(
        sampled.values["gas_density"], expected_value, rtol=2.0e-14, atol=2.0e-14
    )
    assert expected_support[-3]
    assert expected_support[-2]
    assert expected_cell[-2] == (10 if kind == "p1" else 5)
    assert not expected_support[-1]


@pytest.mark.parametrize("kind", ["p1", "q1"])
def test_indexed_masked_candidate_keeps_exact_supported_projection_fallback(
    kind: Literal["p1", "q1"],
) -> None:
    data, layout, field = _unstructured_strip(kind)
    prepared = prepare_required_fields(data, {"gas_density": _DENSITY_REQUIREMENT})
    support = layout.cell_support.copy()
    support[0] = 0
    if kind == "p1":
        masked_layout: P1TriLayout | Q1QuadLayout = P1TriLayout(
            layout.name, layout.nodes_m, layout.connectivity, support
        )
        position = np.asarray([0.75, 0.25], dtype="<f8")
    else:
        masked_layout = Q1QuadLayout(layout.name, layout.nodes_m, layout.connectivity, support)
        position = np.asarray([0.25, 0.5], dtype="<f8")
    indexed = PreparedFieldSet(
        masked_layout,
        prepared.fields,
        prepared.axis_accessible,
        prepared.cell_index,
    )
    expected = locate_field_cell(masked_layout, position, cell_hint=0)

    sampled = indexed.sample(position[None, :], np.asarray([0], dtype="<i8"))

    assert not expected.support_inside
    assert expected.outside_reason == "masked_cell"
    np.testing.assert_array_equal(sampled.support_inside, [False])
    np.testing.assert_array_equal(sampled.cell_id, [expected.cell_id])
    np.testing.assert_allclose(
        sampled.values["gas_density"][0],
        sample_field(field, expected).value,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_field_index_broad_phase_includes_query_dependent_spacing_tolerance() -> None:
    lower_ulp = np.spacing(np.nextafter(np.float64(2.0), 0.0))
    upper = np.float64(2.0) - 100.0 * lower_ulp
    lower = upper - 1.0
    nodes = np.asarray([[lower, 0.0], [upper, 0.0], [upper, 1.0], [lower, 1.0]], dtype="<f8")
    connectivity = np.asarray([[0, 1, 2, 3]], dtype="<i8")
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=_boundary(),
        group_names=(),
        quad4=connectivity,
        quad4_domain_id=np.zeros(1, dtype="<i4"),
    )
    layout = Q1QuadLayout(
        "unstructured", nodes.copy(), connectivity.copy(), np.ones(1, dtype="<u1")
    )
    field = FieldData(
        "gas_density",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.ones((4, 1), dtype="<f8"),
        "kg/m^3",
    )
    prepared = prepare_required_fields(
        _bundle(geometry, (layout,), (field,)), {"gas_density": _DENSITY_REQUIREMENT}
    )
    positions = np.asarray([[2.0, 0.5], [upper + 200.0 * lower_ulp, 0.5]], dtype="<f8")

    sampled = prepared.sample(positions)
    oracle = [locate_field_cell(layout, point) for point in positions]

    assert 2.0 - upper > FIELD_LOCATION_ULPS * np.spacing(upper)
    assert oracle[0].support_inside
    assert not oracle[1].support_inside
    np.testing.assert_array_equal(
        sampled.support_inside, [location.support_inside for location in oracle]
    )
    np.testing.assert_array_equal(sampled.cell_id, [location.cell_id for location in oracle])


@pytest.mark.parametrize("kind", ["p1", "q1"])
def test_prepared_index_preserves_translated_high_aspect_location(
    kind: Literal["p1", "q1"],
) -> None:
    offset = 10.0
    if kind == "p1":
        nodes = np.asarray([[0.0, 0.0], [1.0e-3, 0.0], [1.0e-3, 1.0e-6]], dtype="<f8")
        connectivity = np.asarray([[0, 1, 2]], dtype="<i8")
        geometry = GeometryData(
            nodes_m=nodes + offset,
            boundary=_boundary(),
            group_names=(),
            tri3=connectivity,
            tri3_domain_id=np.zeros(1, dtype="<i4"),
        )
        layout: P1TriLayout | Q1QuadLayout = P1TriLayout(
            "unstructured",
            geometry.nodes_m.copy(),
            connectivity.copy(),
            np.ones(1, dtype="<u1"),
        )
    else:
        nodes = np.asarray(
            [[0.0, 0.0], [1.0e-3, 0.0], [1.1e-3, 1.0e-4], [0.0, 1.0e-4]],
            dtype="<f8",
        )
        connectivity = np.asarray([[0, 1, 2, 3]], dtype="<i8")
        geometry = GeometryData(
            nodes_m=nodes + offset,
            boundary=_boundary(),
            group_names=(),
            quad4=connectivity,
            quad4_domain_id=np.zeros(1, dtype="<i4"),
        )
        layout = Q1QuadLayout(
            "unstructured",
            geometry.nodes_m.copy(),
            connectivity.copy(),
            np.ones(1, dtype="<u1"),
        )
    field = FieldData(
        "gas_density",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.arange(1, layout.nodes_m.shape[0] + 1, dtype="<f8")[:, None],
        "kg/m^3",
    )
    prepared = prepare_required_fields(
        _bundle(geometry, (layout,), (field,)), {"gas_density": _DENSITY_REQUIREMENT}
    )
    point = np.mean(layout.nodes_m, axis=0)
    location = locate_field_cell(layout, point)

    sampled = prepared.sample(point[None, :])

    assert location.support_inside
    np.testing.assert_array_equal(sampled.support_inside, [True])
    np.testing.assert_array_equal(sampled.cell_id, [location.cell_id])
    np.testing.assert_allclose(
        sampled.values["gas_density"][0],
        sample_field(field, location).value,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_required_p1_layout_rejects_shared_ill_conditioned_cell_during_prepare() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0e-9]], dtype="<f8")
    connectivity = np.asarray([[0, 1, 2]], dtype="<i8")
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=_boundary(),
        group_names=(),
        tri3=connectivity,
        tri3_domain_id=np.asarray([0], dtype="<i4"),
    )
    layout = P1TriLayout(
        "unstructured",
        nodes.copy(),
        connectivity.copy(),
        np.asarray([1], dtype="<u1"),
    )
    field = _scalar_field(layout="unstructured", node_count=3)

    with pytest.raises(FieldLocationError, match="conditioning limit"):
        prepare_required_fields(
            _bundle(geometry, (layout,), (field,)),
            {"gas_density": _DENSITY_REQUIREMENT},
        )


@pytest.mark.parametrize("kind", ["p1", "q1"])
@pytest.mark.parametrize("mismatch", ["nodes", "connectivity", "support"])
def test_required_unstructured_layout_requires_exact_mesh_and_full_support(
    kind: str, mismatch: str
) -> None:
    geometry = _tri_geometry() if kind == "p1" else _quad_geometry()
    geometry_connectivity = geometry.tri3 if kind == "p1" else geometry.quad4
    assert geometry_connectivity is not None
    nodes = geometry.nodes_m.copy()
    connectivity = geometry_connectivity.copy()
    support = np.asarray([1], dtype="<u1")
    if mismatch == "nodes":
        nodes[0, 0] = 0.125
    elif mismatch == "connectivity":
        connectivity[0, :2] = connectivity[0, 1::-1]
    else:
        support[0] = 0
    layout: P1TriLayout | Q1QuadLayout
    if kind == "p1":
        layout = P1TriLayout("unstructured", nodes, connectivity, support)
    else:
        layout = Q1QuadLayout("unstructured", nodes, connectivity, support)
    data = _bundle(
        geometry,
        (layout,),
        (_scalar_field(layout="unstructured", node_count=int(geometry.nodes_m.shape[0])),),
    )
    error = "support every layout cell" if mismatch == "support" else "exactly match"

    with pytest.raises(FieldLocationError, match=error):
        prepare_required_fields(data, {"gas_density": _DENSITY_REQUIREMENT})


def test_prepared_field_sample_keeps_provisional_values_separate_from_support() -> None:
    layout = _regular_layout()
    affine = FieldData(
        "affine",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.asarray([[0.0], [2.0], [1.0], [3.0]], dtype="<f8"),
        "1",
    )
    prepared = prepare_required_fields(
        _bundle(_quad_geometry(), (layout,), (affine,)),
        {"affine": RequiredFieldMetadata("1", ("value",), "scalar", False)},
    )

    sampled = prepared.sample(np.asarray([[0.25, 0.5], [1.25, 0.5]], dtype="<f8"))

    np.testing.assert_allclose(sampled.values["affine"][:, 0], [1.25, 2.0])
    np.testing.assert_array_equal(sampled.support_inside, [True, False])
    assert np.isfinite(sampled.values["affine"]).all()
