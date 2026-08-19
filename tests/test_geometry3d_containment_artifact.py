from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from particle_tracer_unified._preflight_initial_state import _geometry_statuses
from particle_tracer_unified.core._triangle_surface import build_geometry_surfaces_3d
from particle_tracer_unified.core.boundary_service import build_boundary_service
from particle_tracer_unified.core.catalogs import build_wall_catalog
from particle_tracer_unified.core.datamodel import PartWallRow, PartWallTable
from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryQuery
from particle_tracer_unified.providers.precomputed import build_precomputed_geometry
from particle_tracer_unified.solvers._collision_detection_3d import (
    classify_trial_collisions_3d,
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


def _write_geometry(
    path,
    *,
    collision_triangles: np.ndarray,
    part_ids: np.ndarray | None,
    containment_triangles: np.ndarray | None,
) -> None:
    axis = np.linspace(-1.0, 1.0, 3)
    arrays = {
        "axis_0": axis,
        "axis_1": axis,
        "axis_2": axis,
        "sdf": -np.ones((3, 3, 3), dtype=np.float64),
        "boundary_triangles": collision_triangles,
    }
    if part_ids is not None:
        arrays["boundary_triangle_part_ids"] = part_ids
    if containment_triangles is not None:
        arrays["containment_boundary_triangles"] = containment_triangles
    np.savez_compressed(path, **arrays)


def _wall_catalog():
    rows = (
        PartWallRow(
            part_id=1,
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
        PartWallRow(
            part_id=2,
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
    )
    return build_wall_catalog(PartWallTable(rows=rows))


def test_split_3d_artifact_uses_outer_shell_only_for_containment(tmp_path) -> None:
    outer = _cube_triangles()
    internal = np.asarray(
        [[[0.0, -0.75, -0.75], [0.0, 0.75, -0.75], [0.0, 0.0, 0.75]]],
        dtype=np.float64,
    )
    collision = np.concatenate((outer, internal), axis=0)
    path = tmp_path / "split-geometry.npz"
    _write_geometry(
        path,
        collision_triangles=collision,
        part_ids=np.asarray([1] * len(outer) + [2], dtype=np.int16),
        containment_triangles=outer.astype(np.float32),
    )

    geometry = build_precomputed_geometry(
        {"npz_path": str(path)}, 3, "cartesian_xyz"
    ).geometry
    surfaces = build_geometry_surfaces_3d(geometry)
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=geometry),
        wall_catalog=_wall_catalog(),
    )
    service = build_boundary_service(
        runtime,
        spatial_dim=3,
        on_boundary_tol_m=1.0e-9,
        triangle_surface_3d=surfaces.collision,
        containment_triangle_surface_3d=surfaces.containment,
    )

    assert geometry.boundary_triangles is not None
    assert geometry.boundary_triangles.dtype == np.float64
    assert geometry.boundary_triangle_part_ids is not None
    assert geometry.boundary_triangle_part_ids.dtype == np.int32
    assert geometry.containment_boundary_triangles is not None
    assert geometry.containment_boundary_triangles.dtype == np.float64
    assert surfaces.collision.triangles.shape[0] == 13
    assert surfaces.containment.triangles.shape[0] == 12
    assert service.inside(np.asarray([-0.5, 0.0, 0.0]))
    assert service.inside(np.asarray([0.5, 0.0, 0.0]))
    assert not service.inside(np.asarray([1.5, 0.0, 0.0]))

    runtime.spatial_dim = 3
    runtime.plan = SimpleNamespace(
        boundary=SimpleNamespace(classification_tolerance_m=1.0e-9)
    )
    statuses, _ = _geometry_statuses(
        runtime,
        np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
    )
    assert statuses.tolist() == ["strict_inside", "strict_inside", "outside"]

    hit = service.segment_hit(np.asarray([-0.5, 0.0, 0.0]), np.asarray([1.5, 0.0, 0.0]))
    assert hit is not None
    assert hit.part_id == 1
    np.testing.assert_allclose(hit.position, [1.0, 0.0, 0.0])

    projection = service.nearest_projection(
        np.asarray([0.01, 0.0, 0.0]), np.asarray([-0.5, 0.0, 0.0])
    )
    assert projection is not None
    assert projection.part_id == 1
    assert not np.isclose(projection.position[0], 0.0)

    trial = classify_trial_collisions_3d(
        runtime,
        active=np.asarray([True]),
        x=np.asarray([[-0.75, 0.0, 0.0]]),
        x_mid_trial=np.asarray([[-0.625, 0.0, 0.0]]),
        x_trial=np.asarray([[-0.5, 0.0, 0.0]]),
        boundary_service=cast(BoundaryQuery[TriangleSurface3D], service),
        on_boundary_tol_m=1.0e-9,
        collision_diagnostics={},
    )
    assert trial.colliders.size == 0
    assert trial.safe.tolist() == [0]


def test_legacy_3d_artifact_reuses_closed_collision_surface(tmp_path) -> None:
    outer = _cube_triangles()
    path = tmp_path / "legacy-geometry.npz"
    _write_geometry(
        path,
        collision_triangles=outer,
        part_ids=None,
        containment_triangles=None,
    )

    geometry = build_precomputed_geometry(
        {"npz_path": str(path)}, 3, "cartesian_xyz"
    ).geometry
    surfaces = build_geometry_surfaces_3d(geometry)

    assert geometry.containment_boundary_triangles is None
    assert surfaces.containment is surfaces.collision
    np.testing.assert_array_equal(surfaces.collision.part_ids, 0)
    assert geometry.metadata["boundary_surface_validation"]["triangle_count"] == 12


@pytest.mark.parametrize(
    ("containment", "message"),
    [
        (np.zeros((1, 3), dtype=np.float64), "containment_boundary_triangles"),
        (_cube_triangles()[:-1], "closed 2-manifold"),
    ],
)
def test_split_3d_artifact_rejects_invalid_containment_before_open_collision(
    tmp_path,
    containment: np.ndarray,
    message: str,
) -> None:
    outer = _cube_triangles()
    collision = np.concatenate(
        (
            outer,
            np.asarray([[[0.0, -0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.0, 0.5]]]),
        ),
        axis=0,
    )
    path = tmp_path / "invalid-containment.npz"
    _write_geometry(
        path,
        collision_triangles=collision,
        part_ids=np.ones(len(collision), dtype=np.int32),
        containment_triangles=containment,
    )

    with pytest.raises(ValueError, match=message):
        build_precomputed_geometry({"npz_path": str(path)}, 3, "cartesian_xyz")


def test_split_3d_artifact_requires_collision_triangles(tmp_path) -> None:
    outer = _cube_triangles()
    path = tmp_path / "containment-only.npz"
    axis = np.linspace(-1.0, 1.0, 3)
    np.savez_compressed(
        path,
        axis_0=axis,
        axis_1=axis,
        axis_2=axis,
        sdf=-np.ones((3, 3, 3)),
        containment_boundary_triangles=outer,
    )

    with pytest.raises(
        ValueError,
        match="containment_boundary_triangles requires boundary_triangles",
    ):
        build_precomputed_geometry({"npz_path": str(path)}, 3, "cartesian_xyz")


def test_geometry_surface_builder_requires_collision_triangles() -> None:
    with pytest.raises(ValueError, match="requires boundary_triangles"):
        build_geometry_surfaces_3d(SimpleNamespace(boundary_triangles=None))
