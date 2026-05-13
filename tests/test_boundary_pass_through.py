from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from particle_tracer_unified.core.boundary_hits import segment_hit_from_boundary_edges
from particle_tracer_unified.core.datamodel import GeometryND, GeometryProviderND, WallCatalog, WallPartModel


def _wall_model(part_id: int, law: str) -> WallPartModel:
    return WallPartModel(
        part_id=int(part_id),
        part_name=f"part_{part_id}",
        material_id=0,
        material_name="",
        law_name=str(law),
        stick_probability=0.0,
        restitution=1.0,
        diffuse_fraction=0.0,
        critical_sticking_velocity_mps=0.0,
        reflectivity=0.0,
        roughness_rms=0.0,
    )


def _runtime_with_edges(laws: dict[int, str]) -> SimpleNamespace:
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
    catalog = WallCatalog(
        default_model=_wall_model(0, "specular"),
        part_models=tuple(_wall_model(pid, law) for pid, law in sorted(laws.items())),
    )
    return SimpleNamespace(geometry_provider=GeometryProviderND(geometry=geom), wall_catalog=catalog)


def test_pass_through_wall_law_is_ignored_by_2d_collision_hits() -> None:
    runtime = _runtime_with_edges({1: "pass_through", 2: "specular"})

    hit = segment_hit_from_boundary_edges(runtime, np.asarray([-0.5, 0.0]), np.asarray([1.5, 0.0]))

    assert hit is not None
    assert hit.part_id == 2
    assert hit.position.tolist() == [1.0, 0.0]


def test_all_pass_through_edges_have_no_2d_collision_hit() -> None:
    runtime = _runtime_with_edges({1: "pass_through", 2: "inactive"})

    hit = segment_hit_from_boundary_edges(runtime, np.asarray([-0.5, 0.0]), np.asarray([1.5, 0.0]))

    assert hit is None
