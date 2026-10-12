from __future__ import annotations

import json
import math
from types import MappingProxyType
from typing import Literal

import numpy as np
import pytest
from numba import njit

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    Layout,
    P1TriLayout,
    Q1QuadLayout,
    RegularLayout,
)
from chamber_particles.fields import (
    FieldLocationError,
    NumericFieldContext,
    NumericFieldRangeWorkspace,
    NumericFieldWorkspace,
    PreparedFieldSet,
    RequiredFieldMetadata,
    local_prepared_field_ranges_into,
    locate_field_cell,
    prepare_required_fields,
    sample_field,
    sample_prepared_field_gradient_into,
    sample_prepared_fields_into,
)
from chamber_particles.numerical_status import (
    FIELD_NUMERICAL_FAILURE,
    NUMERICAL_STATUS_OK,
)

type LayoutKind = Literal["regular", "p1", "q1"]


def _layout(kind: LayoutKind, support: tuple[int, int] = (1, 1)) -> Layout:
    if kind == "regular":
        return RegularLayout(
            "layout",
            np.asarray([0.0, 1.0, 2.0], dtype="<f8"),
            np.asarray([0.0, 1.0], dtype="<f8"),
            np.asarray(support, dtype="<u1")[:, None],
        )
    if kind == "p1":
        nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="<f8")
        return P1TriLayout(
            "layout",
            nodes,
            np.asarray([[0, 1, 2], [1, 3, 2]], dtype="<i8"),
            np.asarray(support, dtype="<u1"),
        )
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.1, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype="<f8",
    )
    return Q1QuadLayout(
        "layout",
        nodes,
        np.asarray([[0, 1, 2, 3], [1, 4, 5, 2]], dtype="<i8"),
        np.asarray(support, dtype="<u1"),
    )


def _node_coordinates(layout: Layout) -> np.ndarray:
    if isinstance(layout, RegularLayout):
        return np.asarray(
            [[first, second] for first in layout.axis0_m for second in layout.axis1_m],
            dtype="<f8",
        )
    return layout.nodes_m


def _fields(layout: Layout) -> tuple[FieldData, FieldData]:
    nodes = _node_coordinates(layout)
    scalar = FieldData(
        "scalar",
        layout.name,
        "node",
        ("value",),
        "scalar",
        (1.0 + 2.0 * nodes[:, 0] - 3.0 * nodes[:, 1] + 0.5 * nodes[:, 0] * nodes[:, 1])[:, None],
        "1",
    )
    vector = FieldData(
        "vector",
        layout.name,
        "node",
        ("x", "y"),
        "cartesian_xy",
        np.column_stack((nodes[:, 0] + 2.0 * nodes[:, 1], 3.0 * nodes[:, 0] - nodes[:, 1])),
        "1",
    )
    return scalar, vector


def _prepared(layout: Layout, *, axis_accessible: bool = False) -> PreparedFieldSet:
    fields = _fields(layout)
    return PreparedFieldSet(
        layout,
        MappingProxyType({field.name: field for field in fields}),
        axis_accessible,
    )


def _points_and_hints(kind: LayoutKind) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if kind == "regular":
        points = [[0.25, 0.3], [1.0, 0.5], [2.4, 0.5]]
        owners = [0, 0, 1]
    elif kind == "p1":
        points = [[0.8, 0.8], [0.5, 0.5], [1.4, 0.5]]
        owners = [1, 0, 1]
    else:
        points = [[1.525, 0.5], [1.05, 0.5], [2.4, 0.5]]
        owners = [1, 0, 1]
    return (
        np.asarray(points, dtype="<f8"),
        np.asarray([0, 1, 0], dtype="<i8"),
        np.asarray(owners, dtype="<i8"),
    )


def _oracle(
    layout: Layout,
    fields: tuple[FieldData, ...],
    positions: np.ndarray,
    hints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    support = np.empty(positions.shape[0], dtype=np.bool_)
    cell_id = np.empty(positions.shape[0], dtype="<i8")
    values = {
        field.name: np.empty((positions.shape[0], len(field.components)), dtype="<f8")
        for field in fields
    }
    for row, position in enumerate(positions):
        location = locate_field_cell(layout, position, cell_hint=int(hints[row]))
        support[row] = location.support_inside
        cell_id[row] = location.cell_id
        for field in fields:
            values[field.name][row] = sample_field(field, location).value
    return support, cell_id, values


@pytest.mark.parametrize("kind", ["regular", "p1", "q1"])
def test_compiled_stage_matches_scalar_oracle_for_hints_shared_faces_and_outside(
    kind: LayoutKind,
) -> None:
    layout = _layout(kind)
    prepared = _prepared(layout)
    fields = tuple(prepared.fields.values())
    positions, hints, expected_owner = _points_and_hints(kind)

    expected_support, expected_cell, expected_values = _oracle(layout, fields, positions, hints)
    sampled = prepared.sample(positions, hints)
    hintless = prepared.sample(positions)
    owner_hints = prepared.sample(positions, expected_owner)

    np.testing.assert_array_equal(expected_support, [True, True, False])
    np.testing.assert_array_equal(expected_cell, expected_owner)
    np.testing.assert_array_equal(sampled.support_inside, expected_support)
    np.testing.assert_array_equal(sampled.cell_id, expected_cell)
    np.testing.assert_array_equal(hintless.support_inside, expected_support)
    np.testing.assert_array_equal(hintless.cell_id, expected_cell)
    np.testing.assert_array_equal(owner_hints.support_inside, expected_support)
    np.testing.assert_array_equal(owner_hints.cell_id, expected_cell)
    for name, expected in expected_values.items():
        np.testing.assert_allclose(sampled.values[name], expected, rtol=2.0e-12, atol=2.0e-12)
        np.testing.assert_allclose(hintless.values[name], expected, rtol=2.0e-12, atol=2.0e-12)
        np.testing.assert_allclose(owner_hints.values[name], expected, rtol=2.0e-12, atol=2.0e-12)

        split = np.concatenate(
            [
                prepared.sample(positions[:1], hints[:1]).values[name],
                prepared.sample(positions[1:], hints[1:]).values[name],
            ]
        )
        np.testing.assert_array_equal(split, sampled.values[name])

    assert prepared.uses_cell_hint is (kind != "regular")


@pytest.mark.parametrize("kind", ["regular", "p1", "q1"])
def test_compiled_masked_provisional_uses_nearest_supported_closure(kind: LayoutKind) -> None:
    layout = _layout(kind, support=(0, 1))
    prepared = _prepared(layout)
    if kind == "p1":
        position = np.asarray([[0.2, 0.2]], dtype="<f8")
    else:
        position = np.asarray([[0.25, 0.5]], dtype="<f8")
    hints = np.asarray([0], dtype="<i8")
    location = locate_field_cell(layout, position[0], cell_hint=0)

    sampled = prepared.sample(position, hints)

    assert not location.support_inside
    assert location.outside_reason == "masked_cell"
    assert location.cell_id == 1
    np.testing.assert_array_equal(sampled.support_inside, [False])
    np.testing.assert_array_equal(sampled.cell_id, [1])
    for field in prepared.fields.values():
        expected = sample_field(field, location).value
        np.testing.assert_allclose(sampled.values[field.name][0], expected, rtol=0.0, atol=2.0e-12)
        assert np.isfinite(sampled.values[field.name]).all()


@pytest.mark.parametrize("kind", ["regular", "p1", "q1"])
def test_compiled_rz_axis_radial_value_is_exact_positive_zero(kind: LayoutKind) -> None:
    layout = _layout(kind)
    nodes = _node_coordinates(layout)
    field = FieldData(
        "velocity",
        layout.name,
        "node",
        ("r", "z"),
        "axisymmetric_rz",
        np.column_stack((nodes[:, 0], 1.0 + nodes[:, 1])),
        "m/s",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}), axis_accessible=True)
    positions = np.asarray([[0.0, 0.25], [-0.0, 0.75], [0.1, 0.25]], dtype="<f8")
    hints = np.asarray([1, 0, 1], dtype="<i8")

    sampled = prepared.sample(positions, hints).values[field.name]

    np.testing.assert_array_equal(sampled[:2, 0], [0.0, 0.0])
    assert not bool(np.signbit(sampled[:2, 0]).any())
    assert sampled[2, 0] > 0.0


def test_compiled_sample_validates_tile_hints_and_wraps_kernel_errors() -> None:
    layout = _layout("regular", support=(0, 0))
    prepared = _prepared(layout)
    positions = np.asarray([[0.25, 0.5]], dtype="<f8")

    with pytest.raises(ValueError, match="shape"):
        prepared.sample(positions, np.asarray([-1, -1], dtype="<i8"))
    with pytest.raises(ValueError, match="integer array"):
        prepared.sample(positions, np.asarray([-1.0], dtype="<f8"))
    with pytest.raises(ValueError, match="cell range"):
        prepared.sample(positions, np.asarray([2], dtype="<i8"))
    with pytest.raises(FieldLocationError, match="no supported field cells"):
        prepared.sample(positions)


def test_compiled_sample_batch_localizes_bad_row_and_preserves_neighbors() -> None:
    prepared = _prepared(_layout("regular"))
    positions = np.asarray(
        [[0.25, 0.3], [np.nan, 0.5], [1.2, 0.4]],
        dtype="<f8",
    )
    workspace = prepared.allocate_workspace(positions.shape[0])

    sampled, status = prepared.sample_batch(positions, workspace=workspace)

    np.testing.assert_array_equal(
        status,
        [NUMERICAL_STATUS_OK, FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK],
    )
    assert workspace.row_status[1] != 0
    assert sampled.cell_id[1] == -1
    assert not sampled.support_inside[1]
    for name, values in sampled.values.items():
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values[1], np.zeros(values.shape[1]))
        np.testing.assert_array_equal(
            values[[0]],
            prepared.sample(positions[[0]]).values[name],
        )
        np.testing.assert_array_equal(
            values[[2]],
            prepared.sample(positions[[2]]).values[name],
        )

    with pytest.raises(FieldLocationError):
        prepared.sample(positions)


def test_compiled_sample_batch_zeroes_nonfinite_interpolation_payload() -> None:
    layout = _layout("regular")
    field = FieldData(
        "cell_scalar",
        layout.name,
        "cell",
        ("value",),
        "scalar",
        np.asarray([[np.inf], [7.0]], dtype="<f8"),
        "1",
    )
    prepared = PreparedFieldSet(
        layout,
        MappingProxyType({field.name: field}),
    )
    positions = np.asarray([[0.25, 0.5], [1.25, 0.5]], dtype="<f8")

    sampled, status = prepared.sample_batch(positions)

    np.testing.assert_array_equal(
        status,
        [FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK],
    )
    np.testing.assert_array_equal(sampled.values[field.name], [[0.0], [7.0]])
    assert np.isfinite(sampled.values[field.name]).all()


@njit(cache=False)
def _native_field_stage(
    context: NumericFieldContext,
    workspace: NumericFieldWorkspace,
    positions: np.ndarray,
    times: np.ndarray,
) -> None:
    sample_prepared_fields_into(context, positions, times, None, workspace)


@pytest.mark.parametrize("kind", ["regular", "p1", "q1"])
def test_native_stage_samples_mixed_primitives_in_required_order_and_row_times(
    kind: LayoutKind,
) -> None:
    layout = _layout(kind)
    nodes = _node_coordinates(layout)
    scalar = FieldData(
        "z_scalar",
        layout.name,
        "node",
        ("value",),
        "scalar",
        (nodes[:, 0] + 2.0 * nodes[:, 1])[:, None],
        "1",
    )
    base = np.column_stack((nodes[:, 0] - nodes[:, 1], 3.0 * nodes[:, 0] + nodes[:, 1]))
    vector = FieldData(
        "a_vector",
        layout.name,
        "node",
        ("x", "y"),
        "cartesian_xy",
        np.stack((base, base + np.asarray([6.0, -4.0]), base + np.asarray([9.0, -6.0]))),
        "1",
        time_s=np.asarray([0.0, 2.0, 3.0]),
    )
    prepared = PreparedFieldSet(
        layout, MappingProxyType({scalar.name: scalar, vector.name: vector})
    )
    positions = np.asarray([[0.2, 0.3], [0.4, 0.2]])
    times = np.asarray([0.5, 2.5])
    workspace = prepared.allocate_workspace(5)

    _native_field_stage(prepared.numeric_context, workspace.numeric, positions, times)

    x, y = positions.T
    expected = np.column_stack((x + 2.0 * y, x - y + 3.0 * times, 3.0 * x + y - 2.0 * times))
    np.testing.assert_allclose(
        workspace.numeric.packed_values[:2], expected, rtol=0.0, atol=3.0e-15
    )
    np.testing.assert_array_equal(workspace.numeric.numerical_status[:2], [NUMERICAL_STATUS_OK] * 2)
    # Opt-in derivatives consume that stage's basis, including a warped Q1.
    gradient = np.empty((2, 1, 2))
    sample_prepared_field_gradient_into(prepared.numeric_context, 0, workspace.numeric, gradient)
    np.testing.assert_allclose(gradient[:, 0], [[1.0, 2.0]] * 2, rtol=0.0, atol=3.0e-15)


def test_native_stage_retains_first_failure_and_preserves_previous_primitive() -> None:
    layout = _layout("regular")
    nodes = _node_coordinates(layout)
    scalar = FieldData("scalar", layout.name, "node", ("value",), "scalar", nodes[:, :1], "1")
    timed = FieldData(
        "timed",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.stack((np.ones((6, 1)), np.full((6, 1), 2.0))),
        "1",
        time_s=np.asarray([0.0, 1.0]),
    )
    last = FieldData("last", layout.name, "node", ("value",), "scalar", np.ones((6, 1)), "1")
    prepared = PreparedFieldSet(
        layout, MappingProxyType({f.name: f for f in (scalar, timed, last)})
    )
    positions = np.asarray([[0.25, 0.5], [np.nan, 0.5], [1.25, 0.5]])
    times = np.asarray([-1.0, -1.0, 0.5])
    workspace = prepared.allocate_workspace(3)

    _native_field_stage(prepared.numeric_context, workspace.numeric, positions, times)

    np.testing.assert_array_equal(
        workspace.numeric.packed_values, [[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [1.25, 1.5, 1.0]]
    )
    np.testing.assert_array_equal(
        workspace.numerical_status,
        [FIELD_NUMERICAL_FAILURE, FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK],
    )
    assert workspace.row_status[0] != workspace.row_status[1]


def test_native_stage_with_no_required_fields_localizes_nonfinite_positions() -> None:
    prepared = PreparedFieldSet(None, MappingProxyType({}))
    workspace = prepared.allocate_workspace(2)

    _native_field_stage(
        prepared.numeric_context,
        workspace.numeric,
        np.asarray([[0.0, 0.0], [np.inf, 0.0]]),
        np.empty(0),
    )

    np.testing.assert_array_equal(
        workspace.numerical_status, [NUMERICAL_STATUS_OK, FIELD_NUMERICAL_FAILURE]
    )
    np.testing.assert_array_equal(workspace.cell_id, [-1, -1])


@njit(cache=False)
def _native_local_ranges(
    context: NumericFieldContext,
    workspace: NumericFieldRangeWorkspace,
    lower: np.ndarray,
    upper: np.ndarray,
    lower_time: np.ndarray | None,
    upper_time: np.ndarray | None,
) -> None:
    local_prepared_field_ranges_into(context, lower, upper, lower_time, upper_time, workspace)


@pytest.mark.parametrize("kind", ["regular", "p1", "q1"])
def test_native_local_ranges_enclose_signed_primitives_and_bracketing_snapshots(
    kind: LayoutKind,
) -> None:
    layout = _layout(kind)
    nodes = _node_coordinates(layout)
    scalar, vector = _fields(layout)
    snapshot = FieldData(
        vector.name,
        layout.name,
        "node",
        vector.components,
        vector.stored_basis,
        np.stack((vector.values, vector.values + 2.0, vector.values + 1000.0)),
        "1",
        time_s=np.asarray([0.0, 1.0, 2.0]),
    )
    prepared = PreparedFieldSet(
        layout, MappingProxyType({scalar.name: scalar, snapshot.name: snapshot})
    )
    workspace = prepared.allocate_range_workspace(5)
    lower = np.asarray([[0.1, 0.1], [100.0, 100.0], [0.1, 0.1], [np.nan, 0.1]])
    upper = np.asarray([[0.2, 0.2], [101.0, 101.0], [0.2, 0.2], [0.2, 0.2]])

    _native_local_ranges(
        prepared.numeric_context,
        workspace,
        lower,
        upper,
        np.asarray([0.1, 0.1, -0.1, -0.1]),
        np.asarray([0.8, 0.8, 0.8, 0.8]),
    )

    np.testing.assert_array_equal(workspace.range_available[:4], [True, False, False, False])
    assert workspace.row_status[0] == workspace.row_status[1] == 0
    assert workspace.row_status[2] != 0
    assert workspace.row_status[3] != workspace.row_status[2]
    expected_nodes = np.arange(nodes.shape[0]) if kind == "p1" else np.asarray([0, 1, 2, 3])
    extrema = np.column_stack((scalar.values[expected_nodes], vector.values[expected_nodes]))
    expected_lower = np.min(extrema, axis=0)
    expected_upper = np.max(extrema, axis=0) + np.asarray([0.0, 2.0, 2.0])
    assert bool((workspace.packed_lower[0] <= expected_lower).all())
    assert bool((workspace.packed_upper[0] >= expected_upper).all())
    np.testing.assert_allclose(workspace.packed_lower[0], expected_lower, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(workspace.packed_upper[0], expected_upper, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(workspace.packed_lower[1], [0.0] * 3)
    # A later call reuses the same storage and cannot leak the failed row's range.
    reused = prepared.local_component_bounds(lower[:1], upper[:1], workspace=workspace)
    assert reused.range_available[0]
    assert reused.upper[vector.name][0, 0] > 1000.0


def test_native_local_range_availability_does_not_certify_box_support() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    layout = P1TriLayout(
        "triangle", nodes, np.asarray([[0, 1, 2]], dtype="<i8"), np.ones(1, dtype=np.uint8)
    )
    field = FieldData("scalar", layout.name, "node", ("value",), "scalar", nodes[:, :1], "1")
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))
    point = np.asarray([[0.8, 0.8]])
    workspace = prepared.allocate_range_workspace(1)

    _native_local_ranges(prepared.numeric_context, workspace, point, point, None, None)
    sampled, status = prepared.sample_batch(point)

    assert workspace.range_available[0]
    assert workspace.candidate_count[0] == 1
    assert workspace.row_status[0] == status[0] == 0
    assert not sampled.support_inside[0]


def test_native_local_ranges_fail_closed_when_expansion_overflows() -> None:
    layout = RegularLayout(
        "regular",
        np.arange(4, dtype="<f8"),
        np.asarray([0.0, 1.0]),
        np.ones((3, 1), dtype=np.uint8),
    )
    values = np.full((8, 1), 7.0)
    values[:4] = np.finfo(np.float64).max
    field = FieldData("scalar", layout.name, "node", ("value",), "scalar", values, "1")
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))
    points = np.asarray([[0.2, 0.2], [2.2, 0.2]])
    workspace = prepared.allocate_range_workspace(2)

    _native_local_ranges(prepared.numeric_context, workspace, points, points, None, None)

    np.testing.assert_array_equal(workspace.range_available, [False, True])
    assert workspace.row_status[0] != 0
    assert workspace.row_status[1] == 0
    assert np.isfinite(workspace.packed_lower).all()
    assert np.isfinite(workspace.packed_upper).all()
    with pytest.raises(FieldLocationError):
        prepared.local_component_bounds(points, points, workspace=workspace)


def _required_regular_sampling(layout: RegularLayout) -> tuple[PreparedFieldSet, FieldData]:
    """Build a required catalog through the production preparation boundary."""

    boundary = BoundaryData(
        line2=np.empty((0, 2), dtype="<i8"),
        boundary_id=np.empty(0, dtype="<i4"),
        group_id=np.empty(0, dtype="<i4"),
        material_id=np.empty(0, dtype="<i4"),
        owner_cell_type=np.empty(0, dtype="<u1"),
        owner_cell_local_index=np.empty(0, dtype="<i8"),
        orientation=np.empty(0, dtype="<i1"),
    )
    geometry = GeometryData(
        nodes_m=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype="<f8"),
        boundary=boundary,
        group_names=(),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.asarray([0], dtype="<i4"),
    )
    field = FieldData(
        "locator_marker",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.arange(layout.axis0_m.size * layout.axis1_m.size, dtype="<f8")[:, None],
        "1",
    )
    provenance = json.dumps(
        {
            "producer": "compiled-field-verification",
            "producer_version": "1",
            "source_sha256": f"sha256:{'0' * 64}",
            "field_semantics_revision": "compiled-field-verification-v1",
            "producer_metadata": {},
        }
    )
    data = DataBundle("cartesian_xy", provenance, geometry, (layout,), (field,))
    prepared = prepare_required_fields(
        data, {field.name: RequiredFieldMetadata("1", ("value",), "scalar", False)}
    )
    return prepared, field


def test_required_nonuniform_regular_sampling_preserves_faces_and_outside_projection() -> None:
    axis0 = np.concatenate(([0.0], np.cumsum(np.tile([1.0, 2.0], 64))))
    layout = RegularLayout(
        "nonuniform",
        axis0,
        np.asarray([0.0, 0.5, 2.0], dtype="<f8"),
        np.ones((128, 2), dtype="<u1"),
    )
    prepared, field = _required_regular_sampling(layout)
    positions = np.asarray(
        [
            [0.25, 0.125],
            [axis0[64], 0.25],
            [axis0[64], 0.5],
            [-1.0, 0.25],
            [axis0[-1] + 1.0, 1.25],
            [axis0[64] - 0.5, -1.0],
            [axis0[-1] + 1.0, 0.5],
        ],
        dtype="<f8",
    )
    hints = np.asarray([255, 0, 1, 255, 0, 255, 0], dtype="<i8")
    support, cell_id, values = _oracle(layout, (field,), positions, hints)

    np.testing.assert_array_equal(support, [True, True, True, False, False, False, False])
    np.testing.assert_array_equal(cell_id, [0, 126, 126, 0, 255, 126, 254])
    for sampled in (prepared.sample(positions, hints), prepared.sample(positions)):
        np.testing.assert_array_equal(sampled.support_inside, support)
        np.testing.assert_array_equal(sampled.cell_id, cell_id)
        np.testing.assert_array_equal(sampled.values[field.name], values[field.name])


def test_required_regular_nearest_hypot_plateau_keeps_lowest_canonical_cell() -> None:
    axis0 = np.concatenate(([0.0], np.cumsum(np.tile([1.0, 2.0], 64))))
    layout = RegularLayout(
        "rounded_distance",
        axis0,
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((128, 1), dtype="<u1"),
    )
    prepared, field = _required_regular_sampling(layout)
    near_last_cell = axis0[-1] - 0.5
    far_y = float(2**52)
    positions = np.asarray(
        [
            [np.nextafter(near_last_cell, -np.inf), np.nextafter(far_y, -np.inf)],
            [near_last_cell, far_y],
            [np.nextafter(near_last_cell, np.inf), np.nextafter(far_y, np.inf)],
        ],
        dtype="<f8",
    )
    # The vertical distance dominates: even cell 0's horizontal displacement
    # disappears in rounded hypot. Cell 127 is geometrically closer, but all
    # rounded scores tie, so the canonical minimum is cell 0. The far y values
    # remain outside even after the existing 64-ULP support tolerance.
    for x, y in positions:
        vertical_distance = float(y - layout.axis1_m[-1])
        assert math.hypot(float(x - axis0[1]), vertical_distance) == vertical_distance
        assert locate_field_cell(layout, np.asarray([x, y])).outside_reason == "outside_layout"
    hints = np.full(positions.shape[0], 127, dtype="<i8")
    support, cell_id, values = _oracle(layout, (field,), positions, hints)

    np.testing.assert_array_equal(support, [False, False, False])
    np.testing.assert_array_equal(cell_id, [0, 0, 0])
    sampled = prepared.sample(positions, hints)
    np.testing.assert_array_equal(sampled.support_inside, support)
    np.testing.assert_array_equal(sampled.cell_id, cell_id)
    np.testing.assert_array_equal(sampled.values[field.name], values[field.name])
    reverse = prepared.sample(positions[::-1].copy())
    np.testing.assert_array_equal(reverse.cell_id[::-1], cell_id)
    np.testing.assert_array_equal(reverse.values[field.name][::-1], values[field.name])


def test_required_regular_nearest_retains_nonfinite_distant_projection_failure() -> None:
    limit = np.finfo(np.float64).max
    axis0 = np.linspace(-0.75 * limit, 0.125 * limit, 129, dtype="<f8")
    layout = RegularLayout(
        "finite_axis_large_difference",
        axis0,
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((128, 1), dtype="<u1"),
    )
    prepared, field = _required_regular_sampling(layout)
    positions = np.asarray([[axis0[64], 0.5], [0.875 * limit, 0.5], [axis0[96], 0.5]], dtype="<f8")
    # All axis widths and the near endpoint difference are finite. The old
    # canonical search still encounters an overflowing difference to a distant
    # supported cell; pruning must not turn that error into a finite sample.
    assert np.isfinite(positions[1, 0] - axis0[-1])
    with np.errstate(over="ignore", invalid="ignore"):
        assert not np.isfinite(positions[1, 0] - axis0[1])
    with pytest.raises(FieldLocationError, match="projection distance"):
        locate_field_cell(layout, positions[1])
    with pytest.raises(FieldLocationError):
        prepared.sample(positions)

    sampled, status = prepared.sample_batch(positions)

    np.testing.assert_array_equal(
        status, [NUMERICAL_STATUS_OK, FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK]
    )
    assert sampled.cell_id[1] == -1
    assert not sampled.support_inside[1]
    np.testing.assert_array_equal(sampled.values[field.name][1], [0.0])
    assert np.isfinite(sampled.values[field.name]).all()
    good = positions[[0, 2]]
    hints = np.full(good.shape[0], -1, dtype="<i8")
    support, cell_id, values = _oracle(layout, (field,), good, hints)
    np.testing.assert_array_equal(sampled.support_inside[[0, 2]], support)
    np.testing.assert_array_equal(sampled.cell_id[[0, 2]], cell_id)
    np.testing.assert_array_equal(sampled.values[field.name][[0, 2]], values[field.name])


def test_uncertified_regular_nearest_keeps_distant_candidate_failure_before_winner() -> None:
    layout = RegularLayout(
        "unresolved_distant_cell",
        np.asarray([0.0, 1.0e-10, 1.0, 2.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((3, 1), dtype="<u1"),
    )
    prepared = _prepared(layout)
    positions = np.asarray([[3.0, 0.5], [1.5, 0.25]], dtype="<f8")
    with pytest.raises(FieldLocationError):
        locate_field_cell(layout, positions[0])
    with pytest.raises(FieldLocationError):
        prepared.sample(positions)
    sampled, status = prepared.sample_batch(positions)
    np.testing.assert_array_equal(status, [FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK])
    assert sampled.cell_id[0] == -1
    assert not sampled.support_inside[0]
    fields = tuple(prepared.fields.values())
    support, cell_id, values = _oracle(layout, fields, positions[1:], np.asarray([-1], dtype="<i8"))
    np.testing.assert_array_equal(sampled.support_inside[1:], support)
    np.testing.assert_array_equal(sampled.cell_id[1:], cell_id)
    for name, expected in values.items():
        np.testing.assert_array_equal(sampled.values[name][1:], expected)


def test_regular_tolerance_extension_preserves_uncertainty_across_binade() -> None:
    upper = np.nextafter(2.0, -np.inf)
    layout = RegularLayout(
        "binade_extension",
        np.asarray([upper - 1.5e-7, upper], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((1, 1), dtype="<u1"),
    )
    prepared = _prepared(layout)
    positions = np.asarray([[2.0, 0.5], [upper, 0.5]], dtype="<f8")
    with pytest.raises(FieldLocationError):
        locate_field_cell(layout, positions[0])
    with pytest.raises(FieldLocationError):
        prepared.sample(positions)
    sampled, status = prepared.sample_batch(positions)
    np.testing.assert_array_equal(status, [FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK])
    assert sampled.cell_id[0] == -1
    assert not sampled.support_inside[0]
    fields = tuple(prepared.fields.values())
    support, cell_id, values = _oracle(layout, fields, positions[1:], np.asarray([-1], dtype="<i8"))
    np.testing.assert_array_equal(sampled.support_inside[1:], support)
    np.testing.assert_array_equal(sampled.cell_id[1:], cell_id)
    for name, expected in values.items():
        np.testing.assert_array_equal(sampled.values[name][1:], expected)


def test_regular_masked_nearest_tie_preserves_canonical_winner_with_strided_axis() -> None:
    layout = RegularLayout(
        "masked_equal_distance",
        np.asarray([0.0, 99.0, 1.0, 99.0, 2.0, 99.0, 3.0], dtype="<f8")[::2],
        np.asarray([0.0, 1.0, 2.0], dtype="<f8"),
        np.asarray([[0, 1], [1, 0], [0, 0]], dtype="<u1"),
    )
    prepared = _prepared(layout)
    positions = np.asarray([[1.5, 1.5], [4.0, 3.0], [1.0, 0.5]], dtype="<f8")
    hints = np.full(positions.shape[0], -1, dtype="<i8")
    fields = tuple(prepared.fields.values())
    support, cell_id, values = _oracle(layout, fields, positions, hints)
    assert cell_id[0] == 1
    assert values["vector"][0, 0] == 4.0
    sampled, status = prepared.sample_batch(positions, hints)
    np.testing.assert_array_equal(status, np.full(3, NUMERICAL_STATUS_OK))
    np.testing.assert_array_equal(sampled.support_inside, support)
    np.testing.assert_array_equal(sampled.cell_id, cell_id)
    for name, expected in values.items():
        np.testing.assert_array_equal(sampled.values[name], expected)
    reversed_sample, reversed_status = prepared.sample_batch(positions[::-1])
    np.testing.assert_array_equal(reversed_status, status[::-1])
    np.testing.assert_array_equal(reversed_sample.cell_id, cell_id[::-1])
    for name, expected in values.items():
        np.testing.assert_array_equal(reversed_sample.values[name], expected[::-1])
