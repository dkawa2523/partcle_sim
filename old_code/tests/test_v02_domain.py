from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from particle_tracer_unified.core import boundary_core
from particle_tracer_unified.core.boundary_service import (
    BoundaryHit as ServiceBoundaryHit,
)
from particle_tracer_unified.core.boundary_service import (
    build_boundary_service,
)
from particle_tracer_unified.core.geometry2d import build_boundary_loops_2d
from particle_tracer_unified.domain import (
    BoundaryHit,
    BoundaryQuery,
    FieldRequest,
    StageFields,
    sample_one,
)


class _RecordingBackend:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def sample(self, points_m, time_s, request):
        points = np.asarray(points_m, dtype=float)
        self.shapes.append(points.shape)
        return StageFields(
            points_m=points,
            time_s=time_s,
            values={
                name: np.zeros((points.shape[0], 2)) for name in request.quantities
            },
            supported=np.ones(points.shape[0], dtype=bool),
        )


def test_scalar_sampling_is_a_one_point_batch() -> None:
    backend = _RecordingBackend()
    sampled = sample_one(backend, [1.0, 2.0], 0.5, FieldRequest(("flow_velocity",)))

    assert backend.shapes == [(1, 2)]
    assert sampled.count == 1
    assert sampled.require("flow_velocity").shape == (1, 2)


def test_stage_fields_rejects_mismatched_particle_axis() -> None:
    with np.testing.assert_raises_regex(ValueError, "particle dimension"):
        StageFields(
            points_m=np.zeros((2, 2)),
            time_s=0.0,
            values={"temperature": np.zeros(1)},
            supported=np.ones(2, dtype=bool),
        )


def _square_boundary_runtime() -> SimpleNamespace:
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    axes = (np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    shape = (3, 3)
    geometry = SimpleNamespace(
        spatial_dim=2,
        axes=axes,
        boundary_edges=edges,
        boundary_edge_part_ids=np.full(4, 7, dtype=np.int32),
        boundary_loops_2d=build_boundary_loops_2d(edges),
        sdf=np.zeros(shape, dtype=np.float64),
        normal_components=(np.zeros(shape), np.ones(shape)),
        nearest_boundary_part_id_map=np.full(shape, 7, dtype=np.int32),
    )
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=geometry),
        field_provider=None,
        wall_catalog=None,
    )


def test_concrete_boundary_service_implements_the_domain_query_contract() -> None:
    service = build_boundary_service(
        _square_boundary_runtime(),
        spatial_dim=2,
        on_boundary_tol_m=1.0e-9,
        triangle_surface_3d=None,
    )

    assert BoundaryQuery not in type(service).__mro__
    assert isinstance(service, BoundaryQuery)
    np.testing.assert_array_equal(
        service.contains(np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)),
        [True, False],
    )
    hit = service.first_hit(
        np.asarray([-0.5, 0.5], dtype=np.float64),
        np.asarray([0.5, 0.5], dtype=np.float64),
    )
    assert isinstance(hit, BoundaryHit)
    assert hit is not None
    np.testing.assert_allclose(hit.position, [0.0, 0.5])
    assert hit.alpha_hint == 0.5
    assert hit.part_id == 7
    assert hit.primitive_kind == "edge"


def test_boundary_query_is_a_complete_structural_solver_contract() -> None:
    required_members = {
        "contains",
        "first_hit",
        "inside",
        "inside_strict",
        "nearest_projection",
        "polyline_hit",
        "primary_hit_counter_key",
        "triangle_surface_3d",
    }

    assert required_members <= set(vars(BoundaryQuery))


def test_boundary_hit_has_one_domain_definition_and_center_distance() -> None:
    assert ServiceBoundaryHit is BoundaryHit
    assert not hasattr(boundary_core, "BoundaryHit")
    hit = BoundaryHit(
        position=np.asarray([1.0, 2.0]),
        normal=np.asarray([0.0, 2.0]),
        part_id=3,
        alpha_hint=0.25,
    )
    assert hit.local_signed_distance(np.asarray([4.0, 5.0])) == 3.0
