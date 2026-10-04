from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    Q1QuadLayout,
    RealizedTableSource,
    RegularLayout,
    content_hash,
    read,
    read_with_info,
    write,
)


def _f8(values: object) -> np.ndarray:
    return np.asarray(values, dtype="<f8")


def _i8(values: object) -> np.ndarray:
    return np.asarray(values, dtype="<i8")


def _i4(values: object) -> np.ndarray:
    return np.asarray(values, dtype="<i4")


def _u1(values: object) -> np.ndarray:
    return np.asarray(values, dtype="<u1")


def _i1(values: object) -> np.ndarray:
    return np.asarray(values, dtype="<i1")


def _bundle() -> DataBundle:
    nodes = _f8([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    triangles = _i8([[0, 1, 2], [0, 2, 3]])
    boundary = BoundaryData(
        line2=_i8([[0, 1], [2, 1], [2, 3], [3, 0]]),
        boundary_id=_i4([10, 11, 12, 13]),
        group_id=_i4([0, 1, 1, 0]),
        material_id=_i4([4, 4, 5, 5]),
        owner_cell_type=_u1([1, 1, 1, 1]),
        owner_cell_local_index=_i8([0, 0, 1, 1]),
        orientation=_i1([1, -1, 1, 1]),
    )
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("wall", "outlet"),
        tri3=triangles,
        tri3_domain_id=_i4([7, 7]),
    )
    layouts = (
        RegularLayout("regular", _f8([0.0, 1.0]), _f8([0.0, 1.0]), _u1([[1]])),
        P1TriLayout("tri", nodes, triangles, _u1([1, 1])),
        Q1QuadLayout("quad", nodes, _i8([[0, 1, 2, 3]]), _u1([1])),
    )
    field = FieldData(
        name="gas_velocity",
        layout="regular",
        association="node",
        components=("x", "y"),
        stored_basis="cartesian_xy",
        values=_f8([[0.0, 0.0], [0.1, 0.0], [0.1, 0.2], [0.0, 0.2]]),
        unit="m/s",
    )
    source = RealizedTableSource(
        name="particles",
        particle_id=_i8([7, 9]),
        release_time_s=_f8([0.0, 0.1]),
        position_m=_f8([[0.25, 0.25], [0.75, 0.75]]),
        velocity_m_s=_f8([[1.0, 0.0], [0.0, -1.0]]),
        charge_number=_f8([-1.0, 0.0]),
        mass_kg=_f8([1.0e-18, 2.0e-18]),
        drag_diameter_m=_f8([1.0e-7, 2.0e-7]),
        electrostatic_radius_m=_f8([5.0e-8, 1.0e-7]),
        displaced_volume_m3=_f8([5.0e-22, 4.0e-21]),
        model_weight=_f8([1.0, 2.0]),
        material_id=_i4([2, 3]),
    )
    provenance = json.dumps(
        {
            "source_sha256": f"sha256:{'0' * 64}",
            "producer_metadata": {"case": "unit-square"},
            "producer_version": "1.0",
            "producer": "verification",
            "field_semantics_revision": "primitive-v1",
        }
    )
    return DataBundle("cartesian_xy", provenance, geometry, layouts, (field,), (source,))


def test_round_trip_uses_logical_hash_and_no_clobber(tmp_path: Path) -> None:
    bundle = _bundle()
    target = tmp_path / "case.h5"

    info = write(target, bundle)
    loaded, read_info = read_with_info(target)
    reordered = replace(bundle, layouts=tuple(reversed(bundle.layouts)))

    assert info.schema_version == 1
    assert read_info == info
    assert info.content_hash == content_hash(bundle) == content_hash(loaded)
    assert content_hash(reordered) == info.content_hash
    assert (
        info.content_hash
        == "sha256:a56cc07421eb27564daae10f5dde95302e8e7af92ecba53e46af8a0347034962"
    )
    assert loaded.provenance_json == json.dumps(
        json.loads(bundle.provenance_json),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert np.array_equal(loaded.geometry.tri3, bundle.geometry.tri3)
    assert not loaded.geometry.nodes_m.flags.writeable
    assert not loaded.fields[0].values.flags.writeable
    assert not loaded.sources[0].mass_kg.flags.writeable

    with pytest.raises(FileExistsError):
        write(target, bundle)
    assert content_hash(read(target)) == info.content_hash


@pytest.mark.parametrize("damage", ["attribute", "unknown_object", "soft_link", "wrong_dtype"])
def test_reader_rejects_noncanonical_hdf5_objects(tmp_path: Path, damage: str) -> None:
    target = tmp_path / f"{damage}.h5"
    write(target, _bundle())
    with h5py.File(target, "r+") as handle:
        if damage == "attribute":
            handle.attrs["unexpected"] = "value"
        elif damage == "unknown_object":
            handle.create_group("unexpected")
        elif damage == "soft_link":
            handle["geometry/node_external_id"] = h5py.SoftLink("/geometry/nodes_m")
        else:
            del handle["geometry/boundary/orientation"]
            handle["geometry/boundary/orientation"] = np.asarray([1, -1, 1, 1], dtype="<i4")

    with pytest.raises(ValueError):
        read(target)


def test_writer_rejects_wrong_dtype_and_wrong_boundary_orientation(tmp_path: Path) -> None:
    bundle = _bundle()
    wrong_dtype = replace(bundle.geometry, nodes_m=np.asarray(bundle.geometry.nodes_m, dtype="<f4"))
    with pytest.raises(ValueError):
        write(tmp_path / "wrong-dtype.h5", replace(bundle, geometry=wrong_dtype))

    without_cells = replace(bundle.geometry, tri3=None, tri3_domain_id=None)
    with pytest.raises(ValueError):
        content_hash(replace(bundle, geometry=without_cells))

    wrong_orientation = replace(bundle.geometry.boundary, orientation=_i1([-1, -1, 1, 1]))
    with pytest.raises(ValueError):
        content_hash(replace(bundle, geometry=replace(bundle.geometry, boundary=wrong_orientation)))

    colliding_source = replace(bundle.sources[0], name="second_table", particle_id=_i8([9, 10]))
    with pytest.raises(ValueError):
        content_hash(replace(bundle, sources=(bundle.sources[0], colliding_source)))


def test_layouts_and_fields_may_both_be_empty(tmp_path: Path) -> None:
    bundle = replace(_bundle(), layouts=(), fields=())
    target = tmp_path / "empty-fields.h5"

    info = write(target, bundle)
    loaded = read(target)

    assert loaded.layouts == ()
    assert loaded.fields == ()
    assert content_hash(loaded) == info.content_hash


@pytest.mark.parametrize(
    "layout",
    [
        P1TriLayout(
            "overflow",
            _f8([[1.0e308, 1.0e308], [-1.0e308, 1.0e308], [1.0e308, -1.0e308]]),
            _i8([[0, 1, 2]]),
            _u1([1]),
        ),
        Q1QuadLayout(
            "overflow",
            _f8(
                [
                    [1.0e308, 1.0e308],
                    [-1.0e308, 1.0e308],
                    [-1.0e308, -1.0e308],
                    [1.0e308, -1.0e308],
                ]
            ),
            _i8([[0, 1, 2, 3]]),
            _u1([1]),
        ),
    ],
)
def test_layout_orientation_overflow_is_rejected(layout: P1TriLayout | Q1QuadLayout) -> None:
    bundle = replace(_bundle(), layouts=(layout,), fields=())

    with pytest.raises(ValueError, match=r"nondegenerate|non-inverted"):
        content_hash(bundle)


def test_empty_boundary_arrays_have_a_stable_hash(tmp_path: Path) -> None:
    bundle = _bundle()
    empty_boundary = BoundaryData(
        line2=_i8([]).reshape(0, 2),
        boundary_id=_i4([]),
        group_id=_i4([]),
        material_id=_i4([]),
        owner_cell_type=_u1([]),
        owner_cell_local_index=_i8([]),
        orientation=_i1([]),
    )
    geometry = replace(bundle.geometry, boundary=empty_boundary, group_names=())
    bundle = replace(bundle, geometry=geometry)
    target = tmp_path / "empty-boundary.h5"

    info = write(target, bundle)

    assert content_hash(read(target)) == info.content_hash
