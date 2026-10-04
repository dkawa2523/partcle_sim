from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from particle_tracer_unified.core.boundary_hits import segment_hit_from_boundary_edges
from particle_tracer_unified.core.boundary_service import build_boundary_service
from particle_tracer_unified.core.catalogs import build_wall_catalog
from particle_tracer_unified.core.datamodel import (
    GeometryND,
    GeometryProviderND,
    PartWallRow,
    PartWallTable,
)
from particle_tracer_unified.core.geometry3d import build_triangle_surface


def _outer_square_with_internal_interface_runtime() -> SimpleNamespace:
    outer = np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[2.0, 0.0], [2.0, 1.0]],
            [[2.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    internal = np.asarray([[[1.0, 0.0], [1.0, 1.0]]], dtype=np.float64)
    axes = (np.linspace(0.0, 2.0, 5), np.linspace(0.0, 1.0, 3))
    shape = tuple(len(axis) for axis in axes)
    geometry = GeometryND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axes=axes,
        valid_mask=np.ones(shape, dtype=bool),
        sdf=-np.ones(shape, dtype=np.float64),
        normal_components=(np.zeros(shape), np.zeros(shape)),
        nearest_boundary_part_id_map=np.full(shape, 2, dtype=np.int32),
        boundary_edges=np.concatenate((outer, internal)),
        boundary_edge_part_ids=np.asarray([2, 2, 2, 2, 1], dtype=np.int32),
        boundary_loops_2d=(
            np.asarray(
                [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
                dtype=np.float64,
            ),
        ),
        metadata={
            "containment_boundary_edge_count": 4,
            "internal_interface_edge_count": 1,
        },
    )
    catalog = build_wall_catalog(
        PartWallTable(
            rows=(
                PartWallRow(
                    part_id=1,
                    part_name="interface",
                    role="internal",
                    material_id=1,
                    material_name="none",
                    wall_law="pass_through",
                    wall_stick_probability=0.0,
                    wall_restitution=1.0,
                    wall_diffuse_fraction=0.0,
                    wall_critical_sticking_velocity_mps=0.0,
                ),
                PartWallRow(
                    part_id=2,
                    part_name="outer",
                    role="wall",
                    material_id=1,
                    material_name="wall",
                    wall_law="specular",
                    wall_stick_probability=0.0,
                    wall_restitution=1.0,
                    wall_diffuse_fraction=0.0,
                    wall_critical_sticking_velocity_mps=0.0,
                ),
            )
        )
    )
    return SimpleNamespace(
        geometry_provider=GeometryProviderND(geometry=geometry),
        wall_catalog=catalog,
    )


def _runtime_with_edges(
    laws: dict[int, str],
    *,
    roles: dict[int, str] | None = None,
) -> SimpleNamespace:
    axes = (np.linspace(-1.0, 2.0, 4), np.linspace(-1.0, 1.0, 3))
    shape = (len(axes[0]), len(axes[1]))
    geom = GeometryND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axes=axes,
        valid_mask=np.ones(shape, dtype=bool),
        sdf=np.zeros(shape, dtype=float),
        normal_components=(np.zeros(shape, dtype=float), np.zeros(shape, dtype=float)),
        nearest_boundary_part_id_map=np.zeros(shape, dtype=np.int32),
        boundary_edges=np.asarray(
            [
                [[0.0, -1.0], [0.0, 1.0]],
                [[1.0, -1.0], [1.0, 1.0]],
            ],
            dtype=float,
        ),
        boundary_edge_part_ids=np.asarray([1, 2], dtype=np.int32),
    )
    catalog = build_wall_catalog(
        PartWallTable(
            rows=tuple(
                PartWallRow(
                    part_id=pid,
                    part_name=f"part_{pid}",
                    role=(
                        roles[pid]
                        if roles is not None and pid in roles
                        else ("internal" if law == "pass_through" else "wall")
                    ),
                    material_id=1,
                    material_name="test_material",
                    wall_law=law,
                    wall_stick_probability=0.0,
                    wall_restitution=1.0,
                    wall_diffuse_fraction=0.0,
                    wall_critical_sticking_velocity_mps=0.0,
                )
                for pid, law in sorted(laws.items())
            )
        )
    )
    return SimpleNamespace(
        geometry_provider=GeometryProviderND(geometry=geom), wall_catalog=catalog
    )


def test_pass_through_wall_law_is_ignored_by_2d_collision_hits() -> None:
    runtime = _runtime_with_edges({1: "pass_through", 2: "specular"})

    hit = segment_hit_from_boundary_edges(
        runtime, np.asarray([-0.5, 0.0]), np.asarray([1.5, 0.0])
    )

    assert hit is not None
    assert hit.part_id == 2
    assert hit.position.tolist() == [1.0, 0.0]


def test_all_pass_through_edges_have_no_2d_collision_hit() -> None:
    runtime = _runtime_with_edges({1: "pass_through", 2: "pass_through"})

    hit = segment_hit_from_boundary_edges(
        runtime, np.asarray([-0.5, 0.0]), np.asarray([1.5, 0.0])
    )

    assert hit is None


def test_exterior_pass_through_edge_remains_a_collision_exit() -> None:
    runtime = _runtime_with_edges(
        {1: "pass_through", 2: "specular"},
        roles={1: "outlet"},
    )

    hit = segment_hit_from_boundary_edges(
        runtime,
        np.asarray([-0.5, 0.0]),
        np.asarray([1.5, 0.0]),
    )

    assert hit is not None
    assert hit.part_id == 1
    assert hit.position.tolist() == [0.0, 0.0]


def test_internal_interface_does_not_split_containment_or_projection() -> None:
    runtime = _outer_square_with_internal_interface_runtime()
    service = build_boundary_service(
        runtime,
        spatial_dim=2,
        on_boundary_tol_m=1.0e-9,
        triangle_surface_3d=None,
    )

    assert service.inside(np.asarray([0.5, 0.5]))
    assert service.inside(np.asarray([1.5, 0.5]))
    assert not service.inside(np.asarray([2.5, 0.5]))

    hit = service.segment_hit(np.asarray([0.5, 0.5]), np.asarray([2.5, 0.5]))
    assert hit is not None
    assert hit.part_id == 2
    assert hit.position.tolist() == [2.0, 0.5]

    projection = service.nearest_projection(
        np.asarray([1.01, 0.5]), np.asarray([0.5, 0.5])
    )
    assert projection is not None
    assert projection.part_id == 2
    assert projection.position.tolist() != [1.0, 0.5]


def test_3d_internal_pass_through_surface_is_transparent() -> None:
    surface = build_triangle_surface(
        np.asarray(
            [
                [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]],
                [[3.0, -1.0, -1.0], [3.0, 1.0, -1.0], [3.0, 0.0, 1.0]],
            ]
        ),
        np.asarray([1, 2], dtype=np.int32),
        validate_closed=False,
    )
    runtime = _runtime_with_edges({1: "pass_through", 2: "specular"})
    service = build_boundary_service(
        runtime,
        spatial_dim=3,
        on_boundary_tol_m=0.0,
        triangle_surface_3d=surface,
    )

    hit = service.segment_hit(np.asarray([0.0, 0.0, 0.0]), np.asarray([4.0, 0.0, 0.0]))

    assert hit is not None
    assert hit.part_id == 2
    assert hit.position.tolist() == [3.0, 0.0, 0.0]

    projection = service.nearest_projection(
        np.asarray([1.01, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0]),
    )
    assert projection is not None
    assert projection.part_id == 2
    assert projection.position.tolist() == [3.0, 0.0, 0.0]
