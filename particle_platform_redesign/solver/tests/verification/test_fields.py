from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from chamber_particles import load_case
from chamber_particles.case_format import FieldData, P1TriLayout, Q1QuadLayout, RegularLayout
from chamber_particles.fields import (
    LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW,
    FieldLocationError,
    PreparedFieldSet,
    locate_field_cell,
    sample_field,
)
from chamber_particles.numerical_status import FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK
from tests.verification.microcases import materialize_microcase


def test_regular_bilinear_sampling_keeps_value_and_support_separate() -> None:
    axis0 = np.asarray([0.0, 1.0, 2.0], dtype="<f8")
    axis1 = np.asarray([-1.0, 1.0], dtype="<f8")
    layout = RegularLayout(
        "regular",
        axis0,
        axis1,
        np.asarray([[1], [0]], dtype="<u1"),
    )
    values = np.asarray(
        [[1.0 + 2.0 * x - 3.0 * y + 4.0 * x * y] for x in axis0 for y in axis1],
        dtype="<f8",
    )
    nodal = FieldData("bilinear", "regular", "node", ("value",), "scalar", values, "1")
    cell = FieldData(
        "cell_value",
        "regular",
        "cell",
        ("value",),
        "scalar",
        np.asarray([[10.0], [20.0]], dtype="<f8"),
        "1",
    )

    inside_location = locate_field_cell(layout, np.asarray([0.25, 0.5]))
    inside = sample_field(nodal, inside_location)
    np.testing.assert_allclose(inside.value, [0.5], rtol=0.0, atol=2.0e-15)
    assert inside.support_inside
    assert inside.cell_id == 0
    assert inside.outside_reason is None
    np.testing.assert_array_equal(sample_field(cell, inside_location).value, [10.0])

    masked_location = locate_field_cell(layout, np.asarray([1.5, 0.0]))
    masked = sample_field(nodal, masked_location)
    assert not masked.support_inside
    assert masked.cell_id == 0
    assert masked.outside_reason == "masked_cell"
    np.testing.assert_array_equal(masked.value, [3.0])
    np.testing.assert_array_equal(sample_field(cell, masked_location).value, [10.0])

    outside = sample_field(nodal, locate_field_cell(layout, np.asarray([-0.25, 0.0])))
    assert not outside.support_inside
    assert outside.outside_reason == "outside_layout"
    assert np.isfinite(outside.value).all()


@pytest.mark.parametrize("layout_kind", ["regular", "p1", "q1"])
def test_snapshot_fields_reuse_spatial_interpolation_at_each_stage_time(
    layout_kind: str,
) -> None:
    if layout_kind == "regular":
        nodes = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        layout: RegularLayout | P1TriLayout | Q1QuadLayout = RegularLayout(
            "layout",
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.ones((1, 1), dtype="<u1"),
        )
        point = np.asarray([0.25, 0.5])
    elif layout_kind == "p1":
        nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        layout = P1TriLayout(
            "layout",
            nodes,
            np.asarray([[0, 1, 2]], dtype="<i8"),
            np.ones(1, dtype="<u1"),
        )
        point = np.asarray([0.25, 0.5])
    else:
        nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        layout = Q1QuadLayout(
            "layout",
            nodes,
            np.asarray([[0, 1, 2, 3]], dtype="<i8"),
            np.ones(1, dtype="<u1"),
        )
        point = np.asarray([0.25, 0.5])
    base = (nodes[:, 0] + 2.0 * nodes[:, 1])[:, None]
    field = FieldData(
        "snapshot",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.stack((base, base + 4.0)),
        "1",
        time_s=np.asarray([0.0, 2.0]),
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))
    positions = np.repeat(point[None, :], 2, axis=0)

    sampled, status = prepared.sample_batch(
        positions,
        time_s=np.asarray([0.5, 1.5]),
    )

    np.testing.assert_array_equal(status, [NUMERICAL_STATUS_OK, NUMERICAL_STATUS_OK])
    np.testing.assert_allclose(sampled.values[field.name][:, 0], [2.25, 4.25], atol=2.0e-15)
    direct = sample_field(field, locate_field_cell(layout, point), time_s=0.5)
    np.testing.assert_allclose(direct.value, [2.25], atol=2.0e-15)
    lower, upper = prepared.component_bounds(field.name)
    assert lower[0] < float(np.min(field.values))
    assert upper[0] > float(np.max(field.values))
    local = prepared.local_component_bounds(
        (point - 0.01)[None, :],
        (point + 0.01)[None, :],
    )
    assert local.range_available[0]
    assert local.lower[field.name][0, 0] < float(np.min(field.values))
    assert local.upper[field.name][0, 0] > float(np.max(field.values))

    failed, failed_status = prepared.sample_batch(
        positions,
        time_s=np.asarray([-np.finfo(np.float64).eps, 2.0]),
    )
    np.testing.assert_array_equal(
        failed_status,
        [FIELD_NUMERICAL_FAILURE, NUMERICAL_STATUS_OK],
    )
    np.testing.assert_array_equal(failed.values[field.name][0], [0.0])
    with pytest.raises(FieldLocationError, match="snapshot range"):
        sample_field(field, locate_field_cell(layout, point), time_s=2.1)


@pytest.mark.parametrize("layout_kind", ["regular", "p1", "q1"])
def test_local_snapshot_bounds_only_visit_the_bracketing_time_interval(
    layout_kind: str,
) -> None:
    if layout_kind == "regular":
        nodes = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        layout: RegularLayout | P1TriLayout | Q1QuadLayout = RegularLayout(
            "layout",
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.ones((1, 1), dtype="<u1"),
        )
    elif layout_kind == "p1":
        nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        layout = P1TriLayout(
            "layout",
            nodes,
            np.asarray([[0, 1, 2]], dtype="<i8"),
            np.ones(1, dtype="<u1"),
        )
    else:
        nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        layout = Q1QuadLayout(
            "layout",
            nodes,
            np.asarray([[0, 1, 2, 3]], dtype="<i8"),
            np.ones(1, dtype="<u1"),
        )
    snapshot_values = np.asarray([0.0, 1.0, 100.0, -100.0, 0.0], dtype="<f8")
    values = np.broadcast_to(snapshot_values[:, None, None], (5, nodes.shape[0], 1)).copy()
    field = FieldData(
        "snapshot",
        layout.name,
        "node",
        ("value",),
        "scalar",
        values,
        "1",
        time_s=np.arange(5, dtype="<f8"),
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))
    point = np.asarray([[0.2, 0.2]], dtype="<f8")

    global_local = prepared.local_component_bounds(point, point)
    interval_local = prepared.local_component_bounds(
        point,
        point,
        time_lower_s=np.asarray([0.2]),
        time_upper_s=np.asarray([0.8]),
    )

    assert global_local.lower[field.name][0, 0] < -100.0
    assert global_local.upper[field.name][0, 0] > 100.0
    assert -1.0e-12 < interval_local.lower[field.name][0, 0] <= 0.0
    assert 1.0 <= interval_local.upper[field.name][0, 0] < 1.0 + 1.0e-12
    with pytest.raises(FieldLocationError, match="outside its snapshot range"):
        prepared.local_component_bounds(
            point,
            point,
            time_lower_s=np.asarray([-0.1]),
            time_upper_s=np.asarray([0.8]),
        )


def test_component_bounds_conservatively_contain_supported_interpolation() -> None:
    axis0 = np.asarray([-1.0, 0.25, 2.0], dtype="<f8")
    axis1 = np.asarray([-0.5, 1.5], dtype="<f8")
    layout = RegularLayout(
        "regular",
        axis0,
        axis1,
        np.ones((2, 1), dtype="<u1"),
    )
    values = np.asarray(
        [[x - 3.0 * y, 2.0 * x + y] for x in axis0 for y in axis1],
        dtype="<f8",
    )
    field = FieldData(
        "vector",
        layout.name,
        "node",
        ("x", "y"),
        "cartesian_xy",
        values,
        "1",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    lower, upper = prepared.component_bounds(field.name)

    assert bool((lower < np.min(values, axis=0)).all())
    assert bool((upper > np.max(values, axis=0)).all())
    assert not lower.flags.writeable
    assert not upper.flags.writeable
    for x in np.linspace(axis0[0], axis0[-1], 13):
        for y in np.linspace(axis1[0], axis1[-1], 11):
            sample = sample_field(field, locate_field_cell(layout, np.asarray([x, y])))
            assert sample.support_inside
            assert bool((sample.value >= lower).all())
            assert bool((sample.value <= upper).all())


def test_component_bounds_fail_when_outward_expansion_is_not_finite() -> None:
    layout = RegularLayout(
        "regular",
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((1, 1), dtype="<u1"),
    )
    field = FieldData(
        "extreme",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.full((4, 1), np.finfo(np.float64).max, dtype="<f8"),
        "1",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    with pytest.raises(FieldLocationError, match="component bounds are not finite"):
        prepared.component_bounds(field.name)


def test_local_regular_bounds_exclude_remote_cells_and_keep_signed_extrema() -> None:
    layout = RegularLayout(
        "regular",
        np.asarray([0.0, 1.0, 2.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((2, 1), dtype="<u1"),
    )
    values = np.asarray(
        [[-2.0, 3.0], [-1.0, 4.0], [10.0, 20.0], [11.0, 21.0], [30.0, 40.0], [31.0, 41.0]],
        dtype="<f8",
    )
    field = FieldData(
        "vector",
        layout.name,
        "node",
        ("x", "y"),
        "cartesian_xy",
        values,
        "1",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    bounds = prepared.local_component_bounds(
        np.asarray([[0.2, 0.2], [0.9, 0.2], [3.0, 0.2]], dtype=np.float64),
        np.asarray([[0.3, 0.8], [1.1, 0.8], [3.1, 0.8]], dtype=np.float64),
    )

    np.testing.assert_array_equal(bounds.candidate_count, [1, 2, 0])
    np.testing.assert_array_equal(bounds.range_available, [True, True, False])
    assert bounds.lower["vector"][0, 0] < -2.0
    assert bounds.upper["vector"][0, 1] > 21.0
    assert bounds.lower["vector"][1, 0] < -2.0
    assert bounds.upper["vector"][1, 1] > 41.0
    np.testing.assert_array_equal(bounds.lower["vector"][2], [0.0, 0.0])
    np.testing.assert_array_equal(bounds.upper["vector"][2], [0.0, 0.0])
    assert not bounds.candidate_count.flags.writeable
    assert not bounds.lower["vector"].flags.writeable


def test_local_bounds_leave_over_capacity_query_unavailable_without_large_arena() -> None:
    cells_per_axis = 9
    assert cells_per_axis * cells_per_axis > LOCAL_FIELD_RANGE_MAX_CELLS_PER_ROW
    axis = np.arange(cells_per_axis + 1, dtype=np.float64)
    layout = RegularLayout(
        "regular",
        axis,
        axis,
        np.ones((cells_per_axis, cells_per_axis), dtype="<u1"),
    )
    values = np.asarray(
        [[x - y] for x in axis for y in axis],
        dtype="<f8",
    )
    field = FieldData(
        "scalar",
        layout.name,
        "node",
        ("value",),
        "scalar",
        values,
        "1",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    bounds = prepared.local_component_bounds(
        np.asarray([[0.0, 0.0], [0.1, 0.1]], dtype=np.float64),
        np.asarray(
            [[float(cells_per_axis), float(cells_per_axis)], [0.2, 0.2]],
            dtype=np.float64,
        ),
    )

    np.testing.assert_array_equal(
        bounds.candidate_count,
        [cells_per_axis * cells_per_axis, 1],
    )
    np.testing.assert_array_equal(bounds.range_available, [False, True])
    np.testing.assert_array_equal(bounds.lower["scalar"][0], [0.0])
    np.testing.assert_array_equal(bounds.upper["scalar"][0], [0.0])
    assert bounds.lower["scalar"][1, 0] < -1.0
    assert bounds.upper["scalar"][1, 0] > 1.0


@pytest.mark.parametrize("layout_kind", ["p1", "q1"])
def test_local_unstructured_bounds_use_only_overlapping_cell_candidates(
    layout_kind: str,
) -> None:
    if layout_kind == "p1":
        nodes = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 0.0], [11.0, 0.0], [10.0, 1.0]],
            dtype="<f8",
        )
        layout: P1TriLayout | Q1QuadLayout = P1TriLayout(
            "unstructured",
            nodes,
            np.asarray([[0, 1, 2], [3, 4, 5]], dtype="<i8"),
            np.ones(2, dtype="<u1"),
        )
    else:
        nodes = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [10.0, 0.0],
                [11.0, 0.0],
                [11.0, 1.0],
                [10.0, 1.0],
            ],
            dtype="<f8",
        )
        layout = Q1QuadLayout(
            "unstructured",
            nodes,
            np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype="<i8"),
            np.ones(2, dtype="<u1"),
        )
    values = np.column_stack((nodes[:, 0] - nodes[:, 1], -nodes[:, 0] - 2.0 * nodes[:, 1]))
    field = FieldData(
        "signed_vector",
        layout.name,
        "node",
        ("x", "y"),
        "cartesian_xy",
        values,
        "1",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))

    bounds = prepared.local_component_bounds(
        np.asarray([[0.1, 0.1], [10.1, 0.1]], dtype=np.float64),
        np.asarray([[0.2, 0.2], [10.2, 0.2]], dtype=np.float64),
    )

    np.testing.assert_array_equal(bounds.candidate_count, [1, 1])
    np.testing.assert_array_equal(bounds.range_available, [True, True])
    first_nodes = layout.connectivity[0]
    second_nodes = layout.connectivity[1]
    assert bool((bounds.lower["signed_vector"][0] < np.min(values[first_nodes], axis=0)).all())
    assert bool((bounds.upper["signed_vector"][0] > np.max(values[first_nodes], axis=0)).all())
    assert bool((bounds.lower["signed_vector"][1] < np.min(values[second_nodes], axis=0)).all())
    assert bool((bounds.upper["signed_vector"][1] > np.max(values[second_nodes], axis=0)).all())


def test_regular_shared_edge_prefers_supported_cell_independent_of_hint() -> None:
    layout = RegularLayout(
        "regular",
        np.asarray([0.0, 1.0, 2.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([[0], [1]], dtype="<u1"),
    )
    field = FieldData(
        "affine",
        "regular",
        "node",
        ("value",),
        "scalar",
        np.asarray([[0.0], [1.0], [1.0], [2.0], [2.0], [3.0]], dtype="<f8"),
        "1",
    )

    for hint in (-1, 0, 1):
        location = locate_field_cell(layout, np.asarray([1.0, 0.5]), cell_hint=hint)
        sample = sample_field(field, location)
        assert sample.support_inside
        assert sample.cell_id == 1
        assert sample.outside_reason is None
        np.testing.assert_array_equal(sample.value, [1.5])


def test_c06_p1_and_mapped_q1_match_the_independent_oracle(tmp_path: Path) -> None:
    paths = materialize_microcase("C06", tmp_path / "C06")
    case = load_case(paths.case_path)
    expected = json.loads(paths.expected_path.read_text(encoding="utf-8"))
    layouts = {layout.name: layout for layout in case.data.layouts}
    fields = {field.name: field for field in case.data.fields}
    probes = expected["field_probes"]

    p1_location = locate_field_cell(layouts["p1"], np.asarray(probes[0]["position_m"]))
    p1_sample = sample_field(fields["p1_affine"], p1_location)
    assert p1_sample.support_inside
    assert p1_sample.cell_id == probes[0]["cell_id"]
    np.testing.assert_allclose(
        p1_sample.value,
        probes[0]["value"],
        rtol=expected["acceptance"]["field_rtol"],
        atol=expected["acceptance"]["field_atol"],
    )

    shared_point = np.asarray(probes[1]["position_m"])
    for cell_hint in probes[1]["candidate_cell_id"]:
        location = locate_field_cell(layouts["p1"], shared_point, cell_hint=cell_hint)
        sample = sample_field(fields["p1_affine"], location)
        assert sample.support_inside
        assert sample.cell_id == 0
        np.testing.assert_allclose(
            sample.value,
            probes[1]["value_from_each_cell"][0],
            rtol=expected["acceptance"]["field_rtol"],
            atol=expected["acceptance"]["field_atol"],
        )

    q1_location = locate_field_cell(layouts["q1"], np.asarray(probes[2]["position_m"]))
    q1_sample = sample_field(fields["q1_bilinear"], q1_location)
    assert q1_sample.support_inside
    assert q1_sample.cell_id == probes[2]["cell_id"]
    np.testing.assert_allclose(
        q1_sample.value,
        probes[2]["value"],
        rtol=expected["acceptance"]["field_rtol"],
        atol=expected["acceptance"]["field_atol"],
    )

    outside_location = locate_field_cell(layouts["q1"], np.asarray(probes[3]["position_m"]))
    outside_sample = sample_field(fields["q1_bilinear"], outside_location)
    assert not outside_sample.support_inside
    assert outside_sample.outside_reason == "outside_layout"
    assert np.isfinite(outside_sample.value).all()


def test_field_location_hint_does_not_change_an_interior_result(tmp_path: Path) -> None:
    paths = materialize_microcase("C06", tmp_path / "C06")
    case = load_case(paths.case_path)
    p1_layout = next(layout for layout in case.data.layouts if layout.name == "p1")
    p1_field = next(field for field in case.data.fields if field.name == "p1_affine")
    point = np.asarray([0.015, 0.0075])

    without_hint = sample_field(p1_field, locate_field_cell(p1_layout, point))
    stale_hint = sample_field(p1_field, locate_field_cell(p1_layout, point, cell_hint=0))

    assert without_hint.cell_id == 1
    assert stale_hint.cell_id == 1
    np.testing.assert_allclose(without_hint.value, [1.75, 1.75], rtol=0.0, atol=2.0e-15)
    np.testing.assert_array_equal(without_hint.value, stale_hint.value)


def test_shared_edge_prefers_supported_p1_cell_independent_of_hint() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="<f8")
    layout = P1TriLayout(
        "p1",
        nodes,
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype="<i8"),
        np.asarray([0, 1], dtype="<u1"),
    )
    field = FieldData(
        "affine",
        "p1",
        "node",
        ("value",),
        "scalar",
        np.asarray([[0.0], [1.0], [1.0], [2.0]], dtype="<f8"),
        "1",
    )

    for hint in (-1, 0, 1):
        sample = sample_field(
            field, locate_field_cell(layout, np.asarray([0.5, 0.5]), cell_hint=hint)
        )
        assert sample.support_inside
        assert sample.cell_id == 1
        assert sample.outside_reason is None
        np.testing.assert_array_equal(sample.value, [1.0])


def test_shared_edge_prefers_supported_q1_cell_independent_of_hint() -> None:
    nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype="<f8",
    )
    layout = Q1QuadLayout(
        "q1",
        nodes,
        np.asarray([[0, 1, 2, 3], [1, 4, 5, 2]], dtype="<i8"),
        np.asarray([0, 1], dtype="<u1"),
    )
    field = FieldData(
        "affine",
        "q1",
        "node",
        ("value",),
        "scalar",
        np.asarray([[0.0], [1.0], [2.0], [1.0], [2.0], [3.0]], dtype="<f8"),
        "1",
    )

    for hint in (-1, 0, 1):
        sample = sample_field(
            field, locate_field_cell(layout, np.asarray([1.0, 0.5]), cell_hint=hint)
        )
        assert sample.support_inside
        assert sample.cell_id == 1
        assert sample.outside_reason is None
        np.testing.assert_array_equal(sample.value, [1.5])


def test_p1_and_nonaffine_q1_are_continuous_across_shared_edges(tmp_path: Path) -> None:
    paths = materialize_microcase("C06", tmp_path / "C06")
    case = load_case(paths.case_path)
    p1_layout = next(layout for layout in case.data.layouts if layout.name == "p1")
    p1_values = next(field.values for field in case.data.fields if field.name == "p1_affine")
    p1_point = np.asarray([0.01, 0.005])
    for cell_id in (0, 1):
        layout_name = f"p1_cell_{cell_id}"
        single_cell = P1TriLayout(
            layout_name,
            p1_layout.nodes_m,
            p1_layout.connectivity[[cell_id]],
            np.asarray([1], dtype="<u1"),
        )
        field = FieldData(
            "p1_affine",
            layout_name,
            "node",
            ("first", "second"),
            "cartesian_xy",
            p1_values,
            "1",
        )
        sample = sample_field(field, locate_field_cell(single_cell, p1_point))
        assert sample.support_inside
        np.testing.assert_allclose(sample.value, [1.5, 0.5], rtol=0.0, atol=2.0e-15)

    q1_nodes = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.1, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]],
        dtype="<f8",
    )
    q1_cells = np.asarray([[0, 1, 2, 3], [1, 4, 5, 2]], dtype="<i8")
    q1_values = (1.0 + 2.0 * q1_nodes[:, 0] - 3.0 * q1_nodes[:, 1])[:, None]
    q1_point = np.asarray([1.05, 0.5])
    for cell_id in (0, 1):
        layout_name = f"q1_cell_{cell_id}"
        single_cell = Q1QuadLayout(
            layout_name,
            q1_nodes,
            q1_cells[[cell_id]],
            np.asarray([1], dtype="<u1"),
        )
        field = FieldData(
            "q1_affine",
            layout_name,
            "node",
            ("value",),
            "scalar",
            q1_values,
            "1",
        )
        sample = sample_field(field, locate_field_cell(single_cell, q1_point))
        assert sample.support_inside
        np.testing.assert_allclose(sample.value, [1.6], rtol=0.0, atol=2.0e-15)

    joined_layout = Q1QuadLayout("q1_joined", q1_nodes, q1_cells, np.asarray([1, 1], dtype="<u1"))
    joined_field = FieldData(
        "q1_affine",
        "q1_joined",
        "node",
        ("value",),
        "scalar",
        q1_values,
        "1",
    )
    joined = sample_field(joined_field, locate_field_cell(joined_layout, q1_point))
    assert joined.support_inside
    assert joined.cell_id == 0
    np.testing.assert_allclose(joined.value, [1.6], rtol=0.0, atol=2.0e-15)


def test_high_aspect_p1_and_q1_edges_are_stable_under_translation() -> None:
    samples: list[np.ndarray] = []
    for offset in (0.0, 1.0):
        p1_nodes = np.asarray([[0.0, 0.0], [1.0e-3, 0.0], [1.0e-3, 1.0e-6]], dtype="<f8")
        p1_nodes += offset
        p1_layout = P1TriLayout(
            "p1_high_aspect",
            p1_nodes,
            np.asarray([[0, 1, 2]], dtype="<i8"),
            np.asarray([1], dtype="<u1"),
        )
        p1_field = FieldData(
            "p1_value",
            p1_layout.name,
            "node",
            ("value",),
            "scalar",
            np.asarray([[0.0], [1.0], [2.0]], dtype="<f8"),
            "1",
        )
        p1_location = locate_field_cell(p1_layout, (p1_nodes[0] + p1_nodes[2]) / 2.0)
        assert p1_location.support_inside
        assert min(p1_location.weights) >= 0.0
        assert sum(p1_location.weights) == pytest.approx(1.0)
        samples.append(sample_field(p1_field, p1_location).value)

        q1_nodes = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.1, 1.0e-6], [0.0, 1.0e-6]],
            dtype="<f8",
        )
        q1_nodes += offset
        q1_layout = Q1QuadLayout(
            "q1_high_aspect",
            q1_nodes,
            np.asarray([[0, 1, 2, 3]], dtype="<i8"),
            np.asarray([1], dtype="<u1"),
        )
        q1_field = FieldData(
            "q1_value",
            q1_layout.name,
            "node",
            ("value",),
            "scalar",
            np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype="<f8"),
            "1",
        )
        q1_location = locate_field_cell(q1_layout, (q1_nodes[2] + q1_nodes[3]) / 2.0)
        assert q1_location.support_inside
        assert min(q1_location.weights) >= 0.0
        assert sum(q1_location.weights) == pytest.approx(1.0)
        samples.append(sample_field(q1_field, q1_location).value)

        p1_outside = locate_field_cell(
            p1_layout, (p1_nodes[0] + p1_nodes[1]) / 2.0 + np.asarray([0.0, -1.0e-9])
        )
        q1_outside = locate_field_cell(
            q1_layout, (q1_nodes[2] + q1_nodes[3]) / 2.0 + np.asarray([0.0, 1.0e-9])
        )
        assert not p1_outside.support_inside
        assert not q1_outside.support_inside

    np.testing.assert_allclose(samples[0], samples[2], rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(samples[1], samples[3], rtol=0.0, atol=2.0e-10)


def test_outside_p1_provisional_uses_physical_nearest_supported_cell() -> None:
    nodes = np.asarray(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [1.0, 0.1],
            [100.0, 0.0],
            [1100.0, 0.0],
            [100.0, 1000.0],
        ],
        dtype="<f8",
    )
    layout = P1TriLayout(
        "physical_nearest",
        nodes,
        np.asarray([[0, 1, 2], [3, 4, 5]], dtype="<i8"),
        np.asarray([1, 1], dtype="<u1"),
    )

    location = locate_field_cell(layout, np.asarray([0.0, 0.0]))

    assert not location.support_inside
    assert location.outside_reason == "outside_layout"
    assert location.cell_id == 0
    assert min(location.weights) >= 0.0
    assert sum(location.weights) == pytest.approx(1.0)


def test_far_outside_provisional_and_sampling_remain_finite() -> None:
    layout = P1TriLayout(
        "finite_provisional",
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="<f8"),
        np.asarray([[0, 1, 2]], dtype="<i8"),
        np.asarray([1], dtype="<u1"),
    )
    field = FieldData(
        "extreme",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.asarray([[1.0e308], [-1.0e308], [0.0]], dtype="<f8"),
        "1",
    )

    location = locate_field_cell(layout, np.asarray([1.0e308, 0.0]))
    sample = sample_field(field, location)

    assert not sample.support_inside
    assert np.isfinite(sample.value).all()
    np.testing.assert_array_equal(sample.value, [-1.0e308])


def test_unresolvable_field_cells_and_missing_support_fail_explicitly() -> None:
    ill_conditioned = P1TriLayout(
        "ill_conditioned",
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0e-16]], dtype="<f8"),
        np.asarray([[0, 1, 2]], dtype="<i8"),
        np.asarray([1], dtype="<u1"),
    )
    with pytest.raises(FieldLocationError, match="conditioning limit"):
        locate_field_cell(ill_conditioned, np.asarray([0.5, 0.0]))

    no_support = RegularLayout(
        "no_support",
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([[0]], dtype="<u1"),
    )
    with pytest.raises(FieldLocationError, match="no supported field cells"):
        locate_field_cell(no_support, np.asarray([0.5, 0.5]))


def test_invalid_cell_hint_is_rejected() -> None:
    layout = RegularLayout(
        "regular",
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.asarray([[1]], dtype="<u1"),
    )

    for invalid_hint in (-2, 1, True):
        with pytest.raises(ValueError, match="cell_hint"):
            locate_field_cell(layout, np.asarray([0.5, 0.5]), cell_hint=invalid_hint)


def test_q1_inverse_mapping_failure_is_not_hidden_by_a_fallback() -> None:
    layout = Q1QuadLayout(
        "singular",
        np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype="<f8"),
        np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        np.asarray([1], dtype="<u1"),
    )

    with pytest.raises(FieldLocationError, match="inverse mapping failed"):
        locate_field_cell(layout, np.asarray([0.5, 0.25]))


@pytest.mark.parametrize("kind", ["p1", "regular", "q1"])
def test_opt_in_spatial_gradient_uses_sampled_basis_and_preserves_values(kind: str) -> None:
    nodes = np.asarray([[0.0, 0.0], [0.0, 3.0], [2.0, 0.0], [2.0, 3.0]], dtype="<f8")
    points = np.asarray([[0.2, 0.3], [1.7, 2.4]], dtype="<f8")
    if kind == "p1":
        layout: P1TriLayout | RegularLayout | Q1QuadLayout = P1TriLayout(
            "gradient",
            nodes,
            np.asarray([[0, 2, 3], [0, 3, 1]], dtype="<i8"),
            np.ones(2, dtype="<u1"),
        )
        value = 2.0 + 3.0 * nodes[:, 0] - 4.0 * nodes[:, 1]
        expected = np.broadcast_to(np.asarray([3.0, -4.0]), (points.shape[0], 2))
    else:
        layout = RegularLayout(
            "gradient", np.asarray([0.0, 2.0]), np.asarray([0.0, 3.0]), np.ones((1, 1), dtype="<u1")
        )
        value = 1.0 + 2.0 * nodes[:, 0] + 3.0 * nodes[:, 1] + 4.0 * nodes[:, 0] * nodes[:, 1]
        expected = np.column_stack((2.0 + 4.0 * points[:, 1], 3.0 + 4.0 * points[:, 0]))
        if kind == "q1":
            layout = Q1QuadLayout(
                "gradient", nodes, np.asarray([[0, 2, 3, 1]], dtype="<i8"), np.ones(1, dtype="<u1")
            )
    field = FieldData(
        "velocity",
        layout.name,
        "node",
        ("x", "y"),
        "cartesian_xy",
        np.column_stack((value, -2.0 * value)),
        "m/s",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))
    workspace = prepared.allocate_workspace(4)
    batch = prepared.sample(points, workspace=workspace)
    sampled = batch.values[field.name].copy()
    gradient = np.empty((points.shape[0], 2, 2), dtype="<f8")
    prepared.spatial_gradient(field.name, workspace, gradient)
    np.testing.assert_allclose(gradient[:, 0], expected, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(gradient[:, 1], -2.0 * expected, rtol=0.0, atol=4e-14)
    np.testing.assert_array_equal(batch.values[field.name], sampled)


def test_q1_spatial_gradient_matches_warped_mapping_chain_rule() -> None:
    # x=(2+v/2)u, y=3v and field=u*v; independent physical chain rule.
    nodes = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.5, 3.0], [0.0, 3.0]])
    layout = Q1QuadLayout(
        "warped_gradient", nodes, np.asarray([[0, 1, 2, 3]], dtype="<i8"), np.ones(1, dtype="<u1")
    )
    reference = np.asarray([[0.2, 0.3], [0.7, 0.8]])
    u, v = reference[:, 0], reference[:, 1]
    points = np.column_stack(((2 + v / 2) * u, 3 * v))
    field = FieldData(
        "value",
        layout.name,
        "node",
        ("value",),
        "scalar",
        np.asarray([[0.0], [0.0], [1.0], [0.0]]),
        "1",
    )
    prepared = PreparedFieldSet(layout, MappingProxyType({field.name: field}))
    workspace = prepared.allocate_workspace(2)
    prepared.sample(points, workspace=workspace)
    gradient = np.empty((2, 1, 2))
    prepared.spatial_gradient(field.name, workspace, gradient)
    np.testing.assert_allclose(
        gradient[:, 0],
        np.column_stack((v / (2 + v / 2), 2 * u / (3 * (2 + v / 2)))),
        rtol=0,
        atol=3e-14,
    )
