from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import (
    RegularFieldND,
    TriangleMeshField2D,
)
from particle_tracer_unified.providers import precomputed
from particle_tracer_unified.providers._precomputed_geometry import (
    build_precomputed_geometry as geometry_builder,
)
from particle_tracer_unified.providers._precomputed_regular import (
    build_precomputed_field as regular_field_builder,
)
from particle_tracer_unified.providers._precomputed_triangle import (
    build_precomputed_triangle_mesh_field as triangle_field_builder,
)
from particle_tracer_unified.providers.precomputed import (
    build_precomputed_field,
    build_precomputed_geometry,
    build_precomputed_triangle_mesh_field,
)
from particle_tracer_unified.providers.synthetic import (
    build_synthetic_field,
    build_synthetic_geometry,
)


def test_precomputed_facade_reexports_provider_builders() -> None:
    assert precomputed.build_precomputed_geometry is geometry_builder
    assert precomputed.build_precomputed_field is regular_field_builder
    assert precomputed.build_precomputed_triangle_mesh_field is triangle_field_builder


def test_precomputed_geometry_preserves_scaled_arrays_and_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "geometry.npz"
    raw_axis = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    raw_sdf = np.arange(9, dtype=np.float32).reshape(3, 3) - 4.0
    valid_mask = np.asarray(
        [[True, True, False], [True, True, False], [True, True, False]],
        dtype=bool,
    )
    edges = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [2.0, 2.0]],
            [[2.0, 2.0], [0.0, 2.0]],
            [[0.0, 2.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        axis_0=raw_axis,
        axis_1=raw_axis,
        sdf=raw_sdf,
        valid_mask=valid_mask,
        normal_0=np.ones((3, 3), dtype=np.float32),
        normal_1=np.zeros((3, 3), dtype=np.float32),
        nearest_boundary_part_id_map=np.full((3, 3), 7, dtype=np.int64),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4], dtype=np.int64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "provider_kind": "characterized_geometry",
                    "source_kind": "characterized_source",
                    "owner": "regression",
                }
            )
        ),
    )

    provider = build_precomputed_geometry(
        {"npz_path": str(path), "coordinate_scale_to_si": 0.5},
        spatial_dim=2,
        coordinate_system="cartesian_xy",
    )
    geometry = provider.geometry

    assert provider.kind == "characterized_geometry"
    assert geometry.source_kind == "characterized_source"
    assert geometry.spatial_dim == 2
    assert geometry.coordinate_system == "cartesian_xy"
    assert geometry.valid_mask.dtype == np.bool_
    assert geometry.nearest_boundary_part_id_map.dtype == np.int32
    assert all(axis.dtype == np.float64 for axis in geometry.axes)
    assert geometry.sdf.dtype == np.float64
    assert all(
        component.dtype == np.float64 for component in geometry.normal_components
    )
    np.testing.assert_array_equal(geometry.axes[0], raw_axis.astype(np.float64) * 0.5)
    np.testing.assert_array_equal(geometry.sdf, raw_sdf.astype(np.float64) * 0.5)
    np.testing.assert_array_equal(
        geometry.boundary_edges, edges.astype(np.float64) * 0.5
    )
    assert len(geometry.boundary_loops_2d) == 1
    assert geometry.metadata["coordinate_scale_to_si"] == 0.5
    assert geometry.metadata["owner"] == "regression"
    assert geometry.metadata["boundary_loop_count_2d"] == 1


def test_precomputed_regular_field_preserves_mapping_support_and_time_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.npz"
    raw_axis = np.asarray([0.0, 1.0], dtype=np.float32)
    geometry_axes = tuple(raw_axis.astype(np.float64) * 0.25 for _ in range(2))
    valid_mask = np.asarray([[True, True], [True, False]], dtype=bool)
    source = np.asarray(
        [
            [[1.0, 2.0], [3.0, np.nan]],
            [[2.0, 3.0], [4.0, np.nan]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        axis_0=raw_axis,
        axis_1=raw_axis,
        times=np.asarray([0.0, 2.0], dtype=np.float32),
        valid_mask=valid_mask,
        support_phi=np.ones((2, 2), dtype=np.float32),
        velocity_source=source,
        metadata_json=np.asarray(json.dumps({"owner": "regression"})),
    )

    provider = build_precomputed_field(
        {
            "npz_path": str(path),
            "coordinate_scale_to_si": 0.25,
            "quantity_mapping": {
                "ux": {
                    "source": "velocity_source",
                    "unit": "m/s",
                    "scale_to_si": 2.0,
                    "semantic_quantity": "velocity",
                    "component": "x",
                }
            },
        },
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axes=geometry_axes,
    )
    field = provider.field

    assert isinstance(field, RegularFieldND)
    assert provider.kind == "precomputed_npz"
    assert field.time_mode == "transient"
    assert field.axis_names == ("x", "y")
    assert field.valid_mask.dtype == np.bool_
    assert field.support_phi is not None
    assert field.support_phi.dtype == np.float64
    np.testing.assert_array_equal(field.support_phi, np.full((2, 2), 0.25))
    assert set(field.quantities) == {"ux"}
    quantity = field.quantities["ux"]
    assert quantity.data.dtype == np.float64
    assert quantity.data.shape == (2, 2, 2)
    np.testing.assert_array_equal(quantity.times, np.asarray([0.0, 2.0]))
    np.testing.assert_array_equal(
        quantity.data[:, valid_mask], 2.0 * source[:, valid_mask]
    )
    assert np.isnan(quantity.data[:, ~valid_mask]).all()
    assert quantity.metadata == {
        "source_array": "velocity_source",
        "semantic_quantity": "velocity",
        "component": "x",
        "scale_to_si": 2.0,
    }
    assert field.metadata["owner"] == "regression"
    assert field.metadata["manifest_quantity_mapping"]["ux"]["source"] == (
        "velocity_source"
    )


def test_precomputed_field_rejects_axis_mismatch_before_payload_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mismatched-field.npz"
    np.savez_compressed(
        path,
        axis_0=np.asarray([0.0, 1.0]),
        axis_1=np.asarray([0.0, 2.0]),
        valid_mask=np.ones((1, 1), dtype=bool),
        ux=np.ones((2, 2), dtype=np.float64),
    )

    with pytest.raises(
        ValueError,
        match=r"Field axis_1 must exactly match geometry axis_1",
    ):
        build_precomputed_field(
            {"npz_path": str(path)},
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            axes=(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
        )


def test_precomputed_triangle_field_preserves_mesh_schema_and_scaling(
    tmp_path: Path,
) -> None:
    path = tmp_path / "triangle-field.npz"
    vertices = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
    raw_velocity = np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    np.savez_compressed(
        path,
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        times=np.asarray([0.0, 1.0], dtype=np.float32),
        velocity_source=raw_velocity,
        metadata_json=np.asarray(json.dumps({"owner": "regression"})),
    )

    provider = build_precomputed_triangle_mesh_field(
        {
            "npz_path": str(path),
            "coordinate_scale_to_si": 0.5,
            "quantity_mapping": {
                "ux": {
                    "source": "velocity_source",
                    "unit": "m/s",
                    "scale_to_si": 3.0,
                }
            },
        },
        spatial_dim=2,
        coordinate_system="cartesian_xy",
    )
    field = provider.field

    assert isinstance(field, TriangleMeshField2D)
    assert provider.kind == "precomputed_triangle_mesh_npz"
    assert field.mesh_vertices.dtype == np.float64
    assert field.mesh_triangles.dtype == np.int32
    np.testing.assert_array_equal(
        field.mesh_vertices, vertices.astype(np.float64) * 0.5
    )
    np.testing.assert_array_equal(field.mesh_triangles, triangles.astype(np.int32))
    assert field.time_mode == "transient"
    assert field.quantities["ux"].data.dtype == np.float64
    np.testing.assert_array_equal(
        field.quantities["ux"].data,
        raw_velocity.astype(np.float64) * 3.0,
    )
    assert field.metadata["owner"] == "regression"
    assert field.metadata["field_backend_kind"] == "triangle_mesh_2d"
    assert 0.0 < float(field.metadata["support_tolerance_m"]) < 1.0e-8


@pytest.mark.parametrize(
    ("spatial_dim", "grid_shape", "expected_boundary_count"),
    [(2, [4, 5], 4), (3, [4, 5, 6], 12)],
)
def test_synthetic_geometry_preserves_box_coordinate_contract(
    spatial_dim: int,
    grid_shape: list[int],
    expected_boundary_count: int,
) -> None:
    bounds = [-2.0, 4.0, -3.0, 5.0]
    if spatial_dim == 3:
        bounds.extend([-7.0, 11.0])
    provider = build_synthetic_geometry(
        {"kind": "box", "bounds": bounds, "grid_shape": grid_shape},
        spatial_dim=spatial_dim,
        coordinate_system="cartesian_xy" if spatial_dim == 2 else "cartesian_xyz",
    )
    geometry = provider.geometry

    assert provider.kind == "synthetic_box"
    assert geometry.source_kind == "synthetic_box"
    assert geometry.valid_mask.shape == tuple(grid_shape)
    assert geometry.valid_mask.dtype == np.bool_
    assert geometry.sdf.dtype == np.float64
    assert geometry.sdf.shape == tuple(grid_shape)
    assert all(axis.dtype == np.float64 for axis in geometry.axes)
    assert all(np.all(np.isfinite(axis)) for axis in geometry.axes)
    assert all(
        component.shape == tuple(grid_shape) for component in geometry.normal_components
    )
    assert all(
        np.all(np.isfinite(component)) for component in geometry.normal_components
    )
    assert geometry.metadata["bounds"] == bounds
    if spatial_dim == 2:
        assert geometry.boundary_edges is not None
        assert geometry.boundary_edges.shape == (expected_boundary_count, 2, 2)
        assert len(geometry.boundary_loops_2d) == 1
    else:
        assert geometry.boundary_triangles is not None
        assert geometry.boundary_triangles.shape == (expected_boundary_count, 3, 3)
        assert geometry.metadata["boundary_surface_validation"]["triangle_count"] == 12


@pytest.mark.parametrize("spatial_dim", [2, 3])
def test_synthetic_transient_field_preserves_shape_dtype_and_formula(
    spatial_dim: int,
) -> None:
    axes = tuple(
        np.asarray([-1.0, 0.0, 1.0], dtype=np.float64) for _ in range(spatial_dim)
    )
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    provider = build_synthetic_field(
        {
            "kind": "linear_shear",
            "time_mode": "transient",
            "times": times,
            "shear_rate": 4.0,
            "dynamic_viscosity_Pas": 2.5e-5,
        },
        spatial_dim=spatial_dim,
        coordinate_system="cartesian_xy" if spatial_dim == 2 else "cartesian_xyz",
        axes=axes,
    )
    field = provider.field

    assert isinstance(field, RegularFieldND)
    assert provider.kind == "synthetic_field"
    assert field.time_mode == "transient"
    assert field.valid_mask.dtype == np.bool_
    assert field.valid_mask.shape == (3,) * spatial_dim
    assert set(field.quantities) == (
        {"ux", "uy", "mu"} if spatial_dim == 2 else {"ux", "uy", "uz", "mu"}
    )
    y_grid = np.meshgrid(*axes, indexing="ij")[1]
    modulation = 1.0 + 0.2 * np.sin(2.0 * np.pi * times)
    expected_ux = modulation.reshape((-1,) + (1,) * spatial_dim) * 4.0 * y_grid
    np.testing.assert_allclose(
        field.quantities["ux"].data, expected_ux, rtol=0.0, atol=1.0e-15
    )
    for quantity in field.quantities.values():
        assert quantity.data.dtype == np.float64
        assert quantity.data.shape == (3,) + (3,) * spatial_dim
        assert np.all(np.isfinite(quantity.data))
    np.testing.assert_array_equal(
        field.quantities["mu"].data,
        np.full((3,) + (3,) * spatial_dim, 2.5e-5),
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, r"providers\.npz_path is required"),
        (
            {"npz_path": "unused.npz", "coordinate_scale_to_si": 0.0},
            "coordinate_scale_to_si must be positive and finite",
        ),
    ],
)
def test_precomputed_geometry_validates_configuration_before_loading(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_precomputed_geometry(config, 2, "cartesian_xy")


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"axis_1": None}, "Missing axis in npz: axis_1"),
        ({"axis_0": np.ones((2, 2))}, "must be 1D with at least 2 entries"),
        ({"axis_0": np.asarray([0.0, np.nan])}, "must contain only finite values"),
        ({"axis_0": np.asarray([0.0, 0.0])}, "must be strictly increasing"),
        ({"sdf": np.ones((1, 1))}, "Geometry sdf shape mismatch"),
        (
            {"sdf": np.asarray([[0.0, 0.0], [0.0, np.inf]])},
            "Geometry sdf must contain only finite values",
        ),
        ({"valid_mask": np.ones((1, 1))}, "Geometry valid_mask shape mismatch"),
    ],
)
def test_precomputed_geometry_rejects_invalid_grid_contracts(
    tmp_path: Path,
    replacement: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "axis_0": np.asarray([0.0, 1.0]),
        "axis_1": np.asarray([0.0, 1.0]),
        "sdf": np.zeros((2, 2), dtype=np.float64),
        "valid_mask": np.ones((2, 2), dtype=bool),
    }
    for key, value in replacement.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    path = tmp_path / "invalid-geometry.npz"
    required_arrays = {
        "axis_0": np.asarray(payload["axis_0"]),
        "sdf": np.asarray(payload["sdf"]),
        "valid_mask": np.asarray(payload["valid_mask"]),
    }
    if "axis_1" in payload:
        np.savez_compressed(
            path,
            axis_1=np.asarray(payload["axis_1"]),
            **required_arrays,
        )
    else:
        np.savez_compressed(path, **required_arrays)

    with pytest.raises(ValueError, match=message):
        build_precomputed_geometry({"npz_path": str(path)}, 2, "cartesian_xy")


def test_precomputed_geometry_derives_normals_and_accepts_legacy_part_map(
    tmp_path: Path,
) -> None:
    path = tmp_path / "minimal-geometry.npz"
    axis = np.asarray([0.0, 1.0, 2.0])
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    np.savez_compressed(
        path,
        axis_0=axis,
        axis_1=axis,
        sdf=x_grid + 2.0 * y_grid,
        part_id_map=np.full((3, 3), 4, dtype=np.int64),
        normal_0=np.full((3, 3), 99.0),
    )

    geometry = build_precomputed_geometry(
        {"npz_path": str(path)}, 2, "cartesian_xy"
    ).geometry

    np.testing.assert_array_equal(geometry.nearest_boundary_part_id_map, 4)
    np.testing.assert_array_equal(geometry.normal_components[0], 1.0)
    np.testing.assert_array_equal(geometry.normal_components[1], 2.0)
    assert geometry.boundary_loops_2d == ()


def test_precomputed_geometry_reports_axisymmetric_and_closed_surface_metadata(
    tmp_path: Path,
) -> None:
    axisymmetric_path = tmp_path / "axisymmetric.npz"
    radial_axis = np.asarray([0.0, 0.5, 1.0])
    axial_axis = np.asarray([-1.0, 0.0, 1.0])
    edges = np.asarray(
        [
            [[0.0, -1.0], [1.0, -1.0]],
            [[1.0, -1.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, -1.0]],
        ]
    )
    np.savez_compressed(
        axisymmetric_path,
        axis_0=radial_axis,
        axis_1=axial_axis,
        sdf=np.zeros((3, 3)),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4]),
    )
    geometry_2d = build_precomputed_geometry(
        {"npz_path": str(axisymmetric_path)}, 2, "axisymmetric_rz"
    ).geometry
    assert geometry_2d.metadata["axisymmetric_rz"]["r0_on_grid"] == 1

    geometry_3d = build_synthetic_geometry(
        {
            "kind": "box",
            "grid_shape": [3, 3, 3],
            "boundary_part_ids": [9],
        },
        3,
        "cartesian_xyz",
    ).geometry
    assert geometry_3d.boundary_triangle_part_ids is not None
    np.testing.assert_array_equal(geometry_3d.boundary_triangle_part_ids, 9)


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ([], "quantity_mapping must be a mapping"),
        ({"ux": 1}, r"quantity_mapping\.ux must be a mapping"),
        ({"ux": {}}, r"quantity_mapping\.ux\.source is required"),
        (
            {"ux": {"source": "raw", "scale_to_si": float("inf")}},
            r"quantity_mapping\.ux\.scale_to_si must be positive and finite",
        ),
    ],
)
def test_precomputed_field_rejects_invalid_quantity_mapping_before_loading(
    mapping: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_precomputed_field(
            {"npz_path": "unused.npz", "quantity_mapping": mapping},
            2,
            "cartesian_xy",
            (np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"valid_mask": np.ones((1, 1))}, "Field valid_mask shape mismatch"),
        ({"support_phi": np.ones((1, 1))}, "Field support_phi shape mismatch"),
        ({"times": np.asarray([])}, "Field times must be a non-empty 1D array"),
        (
            {"times": np.asarray([np.nan])},
            "Field times must contain only finite values",
        ),
        (
            {"times": np.asarray([0.0, 0.0])},
            "Field times must be strictly increasing",
        ),
        (
            {"times": np.asarray([0.0, 1.0]), "ux": np.ones((3, 2, 2))},
            "Quantity ux time axis mismatch",
        ),
        ({"ux": np.ones(2)}, "No field quantities found"),
        (
            {"ux": np.ones((2, 2), dtype=np.complex128) * (1.0 + 2.0j)},
            "Quantity ux must be real-valued",
        ),
    ],
)
def test_precomputed_regular_field_rejects_invalid_payload_contracts(
    tmp_path: Path,
    replacement: dict[str, np.ndarray],
    message: str,
) -> None:
    axis = np.asarray([0.0, 1.0])
    payload = {
        "axis_0": axis,
        "axis_1": axis,
        "times": np.asarray([0.0]),
        "valid_mask": np.ones((2, 2), dtype=bool),
        "support_phi": np.ones((2, 2)),
        "ux": np.ones((2, 2)),
    }
    payload.update(replacement)
    path = tmp_path / "invalid-regular-field.npz"
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match=message):
        build_precomputed_field(
            {"npz_path": str(path)}, 2, "cartesian_xy", (axis, axis)
        )


def test_precomputed_regular_field_rejects_missing_mapped_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-mapped-source.npz"
    axis = np.asarray([0.0, 1.0])
    np.savez_compressed(path, axis_0=axis, axis_1=axis)

    with pytest.raises(ValueError, match="'missing' is missing"):
        build_precomputed_field(
            {
                "npz_path": str(path),
                "quantity_mapping": {"ux": {"source": "missing"}},
            },
            2,
            "cartesian_xy",
            (axis, axis),
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"mesh_triangles": None}, "must include mesh_vertices and mesh_triangles"),
        (
            {"mesh_triangles": np.asarray([[0.0, 1.0, 2.0]])},
            "must use integer vertex indices",
        ),
        ({"mesh_vertices": np.ones((3, 3))}, r"must have shape \(n, 2\)"),
        ({"mesh_triangles": np.ones((1, 2), dtype=int)}, r"shape \(m, 3\)"),
        (
            {
                "mesh_vertices": np.asarray([[0.0, 0.0], [1.0, 0.0]]),
                "mesh_triangles": np.asarray([[0, 1, 1]]),
            },
            "at least three vertices",
        ),
        (
            {"mesh_triangles": np.empty((0, 3), dtype=int)},
            "at least one triangle",
        ),
        (
            {"mesh_vertices": np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, np.nan]])},
            "mesh_vertices must contain only finite values",
        ),
        ({"ux": np.asarray([1.0, 2.0])}, "vertex axis mismatch"),
        (
            {"times": np.asarray([0.0, 1.0]), "ux": np.ones((1, 3))},
            "Mesh quantity ux shape mismatch",
        ),
        ({"ux": np.ones((1, 1, 3))}, "No mesh field quantities found"),
        (
            {"ux": np.asarray([1.0, np.inf, 3.0])},
            "contains non-finite values",
        ),
        (
            {"ux": np.ones(3, dtype=np.complex128) * (1.0 + 2.0j)},
            "Quantity ux must be real-valued",
        ),
    ],
)
def test_precomputed_triangle_field_rejects_invalid_payload_contracts(
    tmp_path: Path,
    replacement: dict[str, np.ndarray | None],
    message: str,
) -> None:
    payload: dict[str, np.ndarray] = {
        "mesh_vertices": np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "mesh_triangles": np.asarray([[0, 1, 2]], dtype=np.int32),
        "times": np.asarray([0.0]),
        "ux": np.asarray([1.0, 2.0, 3.0]),
    }
    for key, value in replacement.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    path = tmp_path / "invalid-triangle-field.npz"
    required_arrays = {
        "mesh_vertices": payload["mesh_vertices"],
        "times": payload["times"],
        "ux": payload["ux"],
    }
    if "mesh_triangles" in payload:
        np.savez_compressed(
            path,
            mesh_triangles=payload["mesh_triangles"],
            **required_arrays,
        )
    else:
        np.savez_compressed(path, **required_arrays)

    with pytest.raises(ValueError, match=message):
        build_precomputed_triangle_mesh_field(
            {"npz_path": str(path)}, 2, "cartesian_xy"
        )
