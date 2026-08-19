from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, NoReturn, TypeVar

import numpy as np

from .coordinate_systems import (
    axis_names_for_coordinate_system,
    axisymmetric_rz_report_from_metadata,
)

PlanT = TypeVar("PlanT")
OptionsT = TypeVar("OptionsT")


class _ImmutableList(list[Any]):
    def _reject_mutation(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise TypeError("simulation input is read-only")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation
    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation


def readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _immutable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return readonly_array(value)
    if isinstance(value, Mapping):
        return immutable_mapping(value)
    if isinstance(value, list):
        return _ImmutableList(_immutable_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_immutable_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_value(item) for item in value)
    return value


def immutable_mapping(value: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(
        {key: _immutable_value(item) for key, item in value.items()}
    )


@dataclass(frozen=True)
class QuantitySeriesND:
    name: str
    unit: str
    times: np.ndarray
    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegularFieldND:
    spatial_dim: int
    coordinate_system: str
    axis_names: tuple[str, ...]
    axes: tuple[np.ndarray, ...]
    quantities: dict[str, QuantitySeriesND]
    valid_mask: np.ndarray
    support_phi: np.ndarray | None = None
    core_valid_mask: np.ndarray | None = None
    time_mode: str = "steady"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TriangleMeshField2D:
    spatial_dim: int
    coordinate_system: str
    mesh_vertices: np.ndarray
    mesh_triangles: np.ndarray
    quantities: dict[str, QuantitySeriesND]
    accel_origin: np.ndarray
    accel_cell_size: np.ndarray
    accel_shape: tuple[int, int]
    accel_cell_offsets: np.ndarray
    accel_triangle_indices: np.ndarray
    time_mode: str = "steady"
    metadata: dict[str, Any] = field(default_factory=dict)


FieldDataND = RegularFieldND | TriangleMeshField2D


@dataclass(frozen=True)
class FieldProviderND:
    field: FieldDataND
    manifest_path: Path | None = None
    kind: str = "regular_rectilinear"

    def summary(self) -> dict[str, Any]:
        field_obj = self.field
        if isinstance(field_obj, TriangleMeshField2D):
            axis_names = axis_names_for_coordinate_system(
                field_obj.coordinate_system, field_obj.spatial_dim
            )
            return {
                "kind": self.kind,
                "field_backend_kind": str(
                    field_obj.metadata.get("field_backend_kind", "triangle_mesh_2d")
                ),
                "spatial_dim": int(field_obj.spatial_dim),
                "coordinate_system": field_obj.coordinate_system,
                "axis_names": list(axis_names),
                "mesh_vertex_count": int(field_obj.mesh_vertices.shape[0]),
                "mesh_triangle_count": int(field_obj.mesh_triangles.shape[0]),
                "quantities": sorted(field_obj.quantities.keys()),
                "time_mode": field_obj.time_mode,
                "manifest_path": str(self.manifest_path) if self.manifest_path else "",
            }
        return {
            "kind": self.kind,
            "field_backend_kind": str(
                field_obj.metadata.get("field_backend_kind", "regular_rectilinear")
            ),
            "spatial_dim": int(field_obj.spatial_dim),
            "coordinate_system": field_obj.coordinate_system,
            "axis_names": list(field_obj.axis_names),
            "grid_shape": list(field_obj.valid_mask.shape),
            "has_support_phi": field_obj.support_phi is not None,
            "quantities": sorted(field_obj.quantities.keys()),
            "time_mode": field_obj.time_mode,
            "manifest_path": str(self.manifest_path) if self.manifest_path else "",
        }


@dataclass(frozen=True)
class GeometryND:
    spatial_dim: int
    coordinate_system: str
    axes: tuple[np.ndarray, ...]
    valid_mask: np.ndarray
    sdf: np.ndarray
    normal_components: tuple[np.ndarray, ...]
    nearest_boundary_part_id_map: np.ndarray
    source_kind: str = "synthetic"
    metadata: dict[str, Any] = field(default_factory=dict)
    boundary_edges: np.ndarray | None = None
    boundary_edge_part_ids: np.ndarray | None = None
    boundary_loops_2d: tuple[np.ndarray, ...] = ()
    boundary_triangles: np.ndarray | None = None
    boundary_triangle_part_ids: np.ndarray | None = None
    containment_boundary_triangles: np.ndarray | None = None

    @property
    def part_id_map(self) -> np.ndarray:
        return self.nearest_boundary_part_id_map


@dataclass(frozen=True)
class GeometryProviderND:
    geometry: GeometryND
    mphtxt_path: Path | None = None
    kind: str = "synthetic"

    def summary(self) -> dict[str, Any]:
        g = self.geometry
        summary = {
            "kind": self.kind,
            "spatial_dim": int(g.spatial_dim),
            "coordinate_system": g.coordinate_system,
            "axis_names": list(
                axis_names_for_coordinate_system(g.coordinate_system, g.spatial_dim)
            ),
            "source_kind": g.source_kind,
            "grid_shape": list(g.valid_mask.shape),
            "has_boundary_edges": g.boundary_edges is not None,
            "has_boundary_loops_2d": bool(g.boundary_loops_2d),
            "has_boundary_triangles": g.boundary_triangles is not None,
            "has_domain_region_map": bool(
                g.metadata.get("has_domain_region_map", False)
            ),
            "mphtxt_path": str(self.mphtxt_path) if self.mphtxt_path else "",
        }
        axisymmetric_report = axisymmetric_rz_report_from_metadata(g.metadata)
        if axisymmetric_report:
            summary["axisymmetric_rz"] = axisymmetric_report
        return summary


@dataclass(frozen=True)
class ParticleTable:
    spatial_dim: int
    particle_id: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    release_time: np.ndarray
    mass: np.ndarray
    diameter: np.ndarray
    density: np.ndarray
    charge: np.ndarray
    source_part_id: np.ndarray
    material_id: np.ndarray
    dep_particle_rel_permittivity: np.ndarray
    thermophoretic_coeff: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.particle_id.size)


@dataclass(frozen=True)
class PartWallRow:
    """Complete behavior for one registered boundary part."""

    part_id: int
    part_name: str
    role: str
    material_id: int
    material_name: str
    wall_law: str
    wall_stick_probability: float
    wall_restitution: float
    wall_diffuse_fraction: float
    wall_critical_sticking_velocity_mps: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PartWallTable:
    rows: tuple[PartWallRow, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_lookup(self) -> dict[int, PartWallRow]:
        return {int(r.part_id): r for r in self.rows}


@dataclass(frozen=True)
class GasProperties:
    """Explicit gas properties in SI units.

    Missing data is represented by NaN so a force/drag model can reject the
    absent quantity it actually requires.  There is intentionally no hidden
    "standard gas" assumption in the domain model.
    """

    temperature: float = float("nan")
    dynamic_viscosity_Pas: float = float("nan")
    density_kgm3: float = float("nan")
    molecular_mass_amu: float = float("nan")


@dataclass(frozen=True)
class WallPartModel:
    part_id: int
    part_name: str
    material_id: int
    material_name: str
    law_name: str
    stick_probability: float
    restitution: float
    diffuse_fraction: float
    critical_sticking_velocity_mps: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WallCatalog:
    part_models: tuple[WallPartModel, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    _part_lookup: dict[int, WallPartModel] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_part_lookup", {int(r.part_id): r for r in self.part_models}
        )

    def as_lookup(self) -> dict[int, WallPartModel]:
        return dict(self._part_lookup)

    def model_for_part(self, part_id: int) -> WallPartModel:
        try:
            return self._part_lookup[int(part_id)]
        except KeyError as exc:
            raise ValueError(
                f"No boundary contract is registered for part_id={int(part_id)}"
            ) from exc


@dataclass(frozen=True)
class SolverContext(Generic[PlanT, OptionsT]):
    """Fully resolved, immutable input to the numerical solver.

    Configuration dictionaries deliberately stop at the IO adapter.  A solver
    run receives domain objects plus already-resolved plans/options, so the hot
    path cannot reinterpret user input or depend on YAML/COMSOL details.
    """

    spatial_dim: int
    coordinate_system: str
    particles: ParticleTable
    geometry_provider: GeometryProviderND
    field_provider: FieldProviderND
    gas: GasProperties
    wall_catalog: WallCatalog
    force_catalog: Any
    plan: PlanT
    options: OptionsT
