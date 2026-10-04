from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from particle_tracer_unified.core.boundary_numerics import resolve_boundary_numerics
from particle_tracer_unified.core.boundary_service import (
    BoundaryHit,
    build_boundary_service,
    nearest_boundary_edge_features_2d,
    polyline_hits_from_boundary_edges_batch,
)
from particle_tracer_unified.core.geometry2d import build_boundary_loops_2d
from particle_tracer_unified.core.geometry3d import (
    build_triangle_surface,
    point_inside_surface,
    point_triangle_barycentric,
    validate_closed_surface_triangles,
)
from particle_tracer_unified.solvers.collision_detection import (
    classify_trial_collisions,
)


def _square_edges() -> np.ndarray:
    return np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )


def _square_runtime(
    part_ids: tuple[int, int, int, int] = (1, 2, 3, 4),
) -> SimpleNamespace:
    edges = _square_edges()
    shape = (11, 11)
    geometry = SimpleNamespace(
        spatial_dim=2,
        axes=(np.linspace(0.0, 1.0, shape[0]), np.linspace(0.0, 1.0, shape[1])),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray(part_ids, dtype=np.int32),
        boundary_loops_2d=build_boundary_loops_2d(edges),
        sdf=np.zeros(shape, dtype=np.float64),
        nearest_boundary_part_id_map=np.zeros(shape, dtype=np.int32),
        normal_components=(np.zeros(shape), np.ones(shape)),
    )
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=geometry), field_provider=None
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


def _cube_runtime(triangles: np.ndarray) -> SimpleNamespace:
    shape = (9, 9, 9)
    geometry = SimpleNamespace(
        spatial_dim=3,
        axes=tuple(np.linspace(-1.0, 1.0, 9) for _ in range(3)),
        boundary_loops_2d=(),
        boundary_triangles=triangles,
        boundary_triangle_part_ids=np.ones(triangles.shape[0], dtype=np.int32),
        sdf=np.zeros(shape, dtype=np.float64),
        nearest_boundary_part_id_map=np.ones(shape, dtype=np.int32),
        normal_components=(np.zeros(shape), np.zeros(shape), np.ones(shape)),
    )
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=geometry), field_provider=None
    )


def _scaled_runtime(runtime: SimpleNamespace, scale: float) -> SimpleNamespace:
    geometry = runtime.geometry_provider.geometry
    scaled = SimpleNamespace(**vars(geometry))
    scaled.axes = tuple(
        np.asarray(axis, dtype=np.float64) * float(scale) for axis in geometry.axes
    )
    for name in ("boundary_edges", "boundary_triangles"):
        values = getattr(geometry, name, None)
        if values is not None:
            setattr(scaled, name, np.asarray(values, dtype=np.float64) * float(scale))
    scaled.boundary_loops_2d = tuple(
        np.asarray(loop, dtype=np.float64) * float(scale)
        for loop in getattr(geometry, "boundary_loops_2d", ())
    )
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=scaled),
        field_provider=None,
    )


def _collision_diagnostics() -> dict[str, int]:
    return {
        "on_boundary_promoted_inside_count": 0,
        "etd2_midpoint_outside_count": 0,
    }


def test_batch_boundary_hits_report_the_earliest_crossed_part() -> None:
    runtime = _square_runtime()
    starts = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.2, 0.2]], dtype=np.float64)
    stages = np.asarray(
        [
            [[1.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [1.5, 0.5]],
            [[0.3, 0.2], [0.4, 0.2]],
        ],
        dtype=np.float64,
    )

    hits = polyline_hits_from_boundary_edges_batch(
        runtime,
        starts,
        stages,
        particle_indices=np.asarray([10, 11, 12], dtype=np.int64),
    )

    assert set(hits) == {10, 11}
    assert hits[10].part_id == hits[11].part_id == 2
    assert hits[10].alpha_hint == pytest.approx(0.25)
    assert hits[11].alpha_hint == pytest.approx(0.75)


def test_nearest_boundary_features_report_part_and_distance() -> None:
    part_ids, distances = nearest_boundary_edge_features_2d(
        _square_runtime(),
        np.asarray([[0.95, 0.5], [0.5, 0.1]], dtype=np.float64),
    )
    assert part_ids.tolist() == [2, 1]
    assert distances.tolist() == pytest.approx([0.05, 0.1])


def test_tiny_boundary_edge_remains_projectable_without_length_floor() -> None:
    scale = 1.0e-16
    runtime = _scaled_runtime(_square_runtime(), scale)
    policy = resolve_boundary_numerics(runtime.geometry_provider)
    service = build_boundary_service(
        runtime,
        spatial_dim=2,
        on_boundary_tol_m=policy.classification_tolerance_m,
        triangle_surface_3d=None,
    )

    hit = service.nearest_projection(
        np.asarray([0.5, 0.25], dtype=np.float64) * scale,
        np.asarray([0.5, 0.5], dtype=np.float64) * scale,
    )

    assert hit is not None
    np.testing.assert_allclose(hit.position / scale, [0.5, 0.0], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(hit.normal, [0.0, -1.0], rtol=0.0, atol=1.0e-14)


def test_boundary_service_2d_uses_explicit_edge_geometry() -> None:
    service = build_boundary_service(
        _square_runtime((10, 20, 30, 40)),
        spatial_dim=2,
        on_boundary_tol_m=1.0e-9,
        triangle_surface_3d=None,
    )

    assert service.inside(np.asarray([0.5, 0.5]))
    assert not service.inside(np.asarray([1.2, 0.5]))
    hit = service.polyline_hit(
        np.asarray([0.5, 0.5]),
        np.asarray([[0.8, 0.5], [1.2, 0.5]]),
    )
    assert hit is not None
    assert hit.position == pytest.approx([1.0, 0.5])
    assert hit.normal == pytest.approx([1.0, 0.0])
    assert hit.part_id == 20
    assert hit.alpha_hint == pytest.approx(0.75)


def test_closed_surface_validation_rejects_holes_and_orientation_errors() -> None:
    triangles = _cube_triangles()
    with pytest.raises(ValueError, match="closed 2-manifold"):
        validate_closed_surface_triangles(triangles[:-1])

    flipped = triangles.copy()
    flipped[0] = flipped[0][[0, 2, 1], :]
    with pytest.raises(ValueError, match="orientation mismatch"):
        validate_closed_surface_triangles(flipped)


def test_point_inside_surface_distinguishes_interior_boundary_and_exterior() -> None:
    triangles = _cube_triangles()
    surface = build_triangle_surface(
        triangles,
        np.ones(triangles.shape[0], dtype=np.int32),
        validate_closed=True,
    )

    assert point_inside_surface(
        surface, np.asarray([0.0, 0.0, 0.0]), on_boundary_tol=1.0e-8
    ) == (True, False)
    assert point_inside_surface(
        surface, np.asarray([1.0, 0.3, -0.2]), on_boundary_tol=1.0e-7
    ) == (True, True)
    assert point_inside_surface(
        surface, np.asarray([1.2, 0.0, 0.0]), on_boundary_tol=1.0e-7
    ) == (False, False)


def test_boundary_service_3d_returns_triangle_hit_metadata() -> None:
    triangles = _cube_triangles()
    surface = build_triangle_surface(
        triangles,
        np.ones(triangles.shape[0], dtype=np.int32),
        validate_closed=True,
    )
    service = build_boundary_service(
        _cube_runtime(triangles),
        spatial_dim=3,
        on_boundary_tol_m=1.0e-7,
        triangle_surface_3d=surface,
    )

    hit = service.polyline_hit(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
    )

    assert hit is not None
    assert hit.position == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-8)
    assert hit.normal == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-8)
    assert hit.primitive_kind == "triangle"
    assert hit.primitive_id >= 0


@pytest.mark.parametrize("scale", [1.0e-6, 1.0e3])
def test_boundary_policy_and_2d_hit_are_similarity_scale_invariant(
    scale: float,
) -> None:
    base_runtime = _square_runtime((10, 20, 30, 40))
    scaled_runtime = _scaled_runtime(base_runtime, scale)
    base_policy = resolve_boundary_numerics(base_runtime.geometry_provider)
    scaled_policy = resolve_boundary_numerics(scaled_runtime.geometry_provider)

    for name in (
        "reference_length_m",
        "resolution_length_m",
        "classification_tolerance_m",
        "contact_offset_m",
        "radial_axis_tolerance_m",
    ):
        assert getattr(scaled_policy, name) == pytest.approx(
            scale * getattr(base_policy, name),
            rel=2.0e-12,
            abs=0.0,
        )
    roundoff_ratio = scaled_policy.coordinate_roundoff_m / (
        scale * base_policy.coordinate_roundoff_m
    )
    # ULP size is quantized at powers of two; it remains within one bin under
    # an arbitrary decimal similarity factor.
    assert 0.5 <= roundoff_ratio <= 2.0

    service = build_boundary_service(
        scaled_runtime,
        spatial_dim=2,
        on_boundary_tol_m=scaled_policy.classification_tolerance_m,
        triangle_surface_3d=None,
    )
    hit = service.segment_hit(
        scale * np.asarray([0.5, 0.5]),
        scale * np.asarray([1.5, 0.5]),
    )
    assert hit is not None
    np.testing.assert_allclose(hit.position / scale, [1.0, 0.5], rtol=0.0, atol=2.0e-14)
    assert hit.part_id == 20


@pytest.mark.parametrize("scale", [1.0e-13, 1.0, 1.0e3])
def test_triangle_surface_queries_are_similarity_scale_invariant(scale: float) -> None:
    triangles = _cube_triangles() * float(scale)
    runtime = _scaled_runtime(_cube_runtime(_cube_triangles()), scale)
    policy = resolve_boundary_numerics(runtime.geometry_provider)
    topology = validate_closed_surface_triangles(triangles)
    assert topology["triangle_count"] == 12
    assert topology["unique_vertex_count"] == 8
    assert topology["identity_policy"] == policy.policy_version
    assert topology["identity_resolution_m"] == pytest.approx(2.0 * scale)
    surface = build_triangle_surface(
        triangles,
        np.ones(triangles.shape[0], dtype=np.int32),
        validate_closed=True,
    )
    service = build_boundary_service(
        runtime,
        spatial_dim=3,
        on_boundary_tol_m=policy.classification_tolerance_m,
        triangle_surface_3d=surface,
    )

    assert service.inside(scale * np.asarray([0.0, 0.0, 0.0]))
    assert not service.inside(scale * np.asarray([1.2, 0.0, 0.0]))
    hit = service.segment_hit(
        scale * np.asarray([0.0, 0.0, 0.0]),
        scale * np.asarray([1.5, 0.0, 0.0]),
    )
    assert hit is not None
    np.testing.assert_allclose(
        hit.position / scale, [1.0, 0.0, 0.0], rtol=0.0, atol=2.0e-13
    )


@pytest.mark.parametrize("scale", [1.0e-13, 1.0, 1.0e3])
def test_closed_surface_topology_accepts_serialization_ulp_at_every_scale(
    scale: float,
) -> None:
    triangles = _cube_triangles() * float(scale)
    triangles[0, 0, 0] = np.nextafter(triangles[0, 0, 0], np.inf)

    report = validate_closed_surface_triangles(triangles)

    assert report["triangle_count"] == 12
    assert report["unique_vertex_count"] == 8
    assert report["edge_count"] == 18
    assert (
        0.0
        < float(cast(float, report["identity_coordinate_roundoff_m"]))
        <= float(cast(float, report["identity_tolerance_m"]))
    )


@pytest.mark.parametrize("scale", [1.0e-12, 1.0, 1.0e3])
def test_contact_triangle_barycentric_coordinates_are_scale_invariant(
    scale: float,
) -> None:
    triangle = scale * np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=np.float64,
    )
    point = scale * np.asarray([0.5, 0.75, 0.0], dtype=np.float64)

    barycentric = point_triangle_barycentric(point, triangle)

    assert barycentric is not None
    np.testing.assert_allclose(barycentric, [0.5, 0.25, 0.25], rtol=0.0, atol=2.0e-14)


def test_boundary_policy_rejects_float64_ill_conditioned_coordinates() -> None:
    runtime = _square_runtime()
    geometry = runtime.geometry_provider.geometry
    origin = 1.0e9
    geometry.axes = tuple(
        origin + np.linspace(0.0, 1.0e-4, len(axis)) for axis in geometry.axes
    )
    geometry.boundary_edges = origin + 1.0e-4 * _square_edges()
    geometry.boundary_loops_2d = build_boundary_loops_2d(geometry.boundary_edges)

    with pytest.raises(ValueError, match="poorly conditioned"):
        resolve_boundary_numerics(runtime.geometry_provider)


def test_trial_collision_prefetches_3d_boundary_hit() -> None:
    triangles = _cube_triangles()
    runtime = _cube_runtime(triangles)
    surface = build_triangle_surface(
        triangles,
        np.ones(triangles.shape[0], dtype=np.int32),
        validate_closed=True,
    )
    service = build_boundary_service(
        runtime, spatial_dim=3, on_boundary_tol_m=1.0e-7, triangle_surface_3d=surface
    )

    batch = classify_trial_collisions(
        runtime,
        spatial_dim=3,
        n_particles=1,
        active=np.asarray([True]),
        x=np.asarray([[0.0, 0.0, 0.0]]),
        x_trial=np.asarray([[1.5, 0.0, 0.0]]),
        x_mid_trial=np.asarray([[0.5, 0.0, 0.0]]),
        boundary_service=service,
        on_boundary_tol_m=1.0e-7,
        collision_diagnostics=_collision_diagnostics(),
    )

    assert batch.colliders.tolist() == [0]
    assert batch.safe.size == 0
    assert isinstance(batch.prefetched_hits[0], BoundaryHit)


def test_inside_to_inside_path_still_detects_internal_2d_wall() -> None:
    outer = np.asarray(
        [
            [[0.0, 0.0], [4.0, 0.0]],
            [[4.0, 0.0], [4.0, 4.0]],
            [[4.0, 4.0], [0.0, 4.0]],
            [[0.0, 4.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    inner = np.asarray(
        [
            [[1.5, 1.5], [2.5, 1.5]],
            [[2.5, 1.5], [2.5, 2.5]],
            [[2.5, 2.5], [1.5, 2.5]],
            [[1.5, 2.5], [1.5, 1.5]],
        ],
        dtype=np.float64,
    )
    edges = np.concatenate((outer, inner), axis=0)
    geometry = SimpleNamespace(
        spatial_dim=2,
        axes=(np.asarray([0.0, 4.0]), np.asarray([0.0, 4.0])),
        boundary_loops_2d=build_boundary_loops_2d(edges),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([10] * 4 + [20] * 4, dtype=np.int32),
        sdf=np.zeros((2, 2)),
        nearest_boundary_part_id_map=np.zeros((2, 2), dtype=np.int32),
        normal_components=(np.zeros((2, 2)), np.ones((2, 2))),
    )
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=geometry), field_provider=None
    )
    service = build_boundary_service(
        runtime, spatial_dim=2, on_boundary_tol_m=1.0e-9, triangle_surface_3d=None
    )

    batch = classify_trial_collisions(
        runtime,
        spatial_dim=2,
        n_particles=1,
        active=np.asarray([True]),
        x=np.asarray([[0.5, 2.0]]),
        x_trial=np.asarray([[3.5, 2.0]]),
        x_mid_trial=np.asarray([[2.0, 2.0]]),
        boundary_service=service,
        on_boundary_tol_m=1.0e-9,
        collision_diagnostics=_collision_diagnostics(),
    )

    assert batch.colliders.tolist() == [0]
    assert batch.safe.size == 0
    # A midpoint already outside the valid loop is classified immediately;
    # the segment primitive used by collision replay still identifies the
    # exact internal wall and part.
    hit = service.polyline_hit(
        np.asarray([0.5, 2.0]), np.asarray([[2.0, 2.0], [3.5, 2.0]])
    )
    assert hit is not None
    assert hit.part_id == 20
    assert hit.position == pytest.approx([1.5, 2.0], abs=1.0e-8)
