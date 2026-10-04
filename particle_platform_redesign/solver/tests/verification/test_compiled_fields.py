from __future__ import annotations

from types import MappingProxyType
from typing import Literal

import numpy as np
import pytest

from chamber_particles.case_format import (
    FieldData,
    Layout,
    P1TriLayout,
    Q1QuadLayout,
    RegularLayout,
)
from chamber_particles.fields import (
    FieldLocationError,
    PreparedFieldSet,
    locate_field_cell,
    sample_field,
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
    assert prepared.prepared_nbytes == 0


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
