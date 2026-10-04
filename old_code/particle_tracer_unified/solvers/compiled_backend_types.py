from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from particle_tracer_unified.core.datamodel import TriangleMeshField2D


@dataclass(frozen=True, slots=True)
class RegularRectilinearCompiledBackend:
    """Immutable, solver-ready representation of a regular field grid."""

    axes: tuple[np.ndarray, ...]
    times: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    gas_density: np.ndarray
    gas_mu: np.ndarray
    gas_temperature: np.ndarray
    valid_mask: np.ndarray
    core_valid_mask: np.ndarray
    uz: np.ndarray | None = None
    electric_x: np.ndarray | None = None
    electric_y: np.ndarray | None = None
    electric_z: np.ndarray | None = None
    backend_kind: str = "regular_rectilinear"
    acceleration_source: str = "none"
    acceleration_quantity_names: tuple[str, ...] = ()
    electric_field_names: tuple[str, ...] = ()
    gas_density_source: str = "unavailable"
    gas_mu_source: str = "unavailable"
    gas_temperature_source: str = "unavailable"
    grad_T_x: np.ndarray | None = None
    grad_T_y: np.ndarray | None = None
    grad_T_z: np.ndarray | None = None
    grad_E2_x: np.ndarray | None = None
    grad_E2_y: np.ndarray | None = None
    grad_E2_z: np.ndarray | None = None
    vorticity_x: np.ndarray | None = None
    vorticity_y: np.ndarray | None = None
    vorticity_z: np.ndarray | None = None
    fluid_accel_x: np.ndarray | None = None
    fluid_accel_y: np.ndarray | None = None
    fluid_accel_z: np.ndarray | None = None
    du_dt_x: np.ndarray | None = None
    du_dt_y: np.ndarray | None = None
    du_dt_z: np.ndarray | None = None
    grad_ux_x: np.ndarray | None = None
    grad_ux_y: np.ndarray | None = None
    grad_ux_z: np.ndarray | None = None
    grad_uy_x: np.ndarray | None = None
    grad_uy_y: np.ndarray | None = None
    grad_uy_z: np.ndarray | None = None
    grad_uz_x: np.ndarray | None = None
    grad_uz_y: np.ndarray | None = None
    grad_uz_z: np.ndarray | None = None
    coordinate_system: str = "cartesian_xy"


@dataclass(frozen=True, slots=True)
class TriangleMesh2DCompiledBackend:
    """Immutable, solver-ready representation of a 2-D P1 triangle field."""

    field: TriangleMeshField2D
    velocity_names: tuple[str, ...]
    times: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    gas_density: np.ndarray
    gas_mu: np.ndarray
    gas_temperature: np.ndarray
    mesh_vertices: np.ndarray
    mesh_triangles: np.ndarray
    accel_origin: np.ndarray
    accel_cell_size: np.ndarray
    accel_shape: tuple[int, int]
    accel_cell_offsets: np.ndarray
    accel_triangle_indices: np.ndarray
    support_tolerance_m: float
    backend_kind: str = "triangle_mesh_2d"
    acceleration_source: str = "none"
    acceleration_quantity_names: tuple[str, ...] = ()
    electric_field_names: tuple[str, ...] = ()
    gas_density_source: str = "unavailable"
    gas_mu_source: str = "unavailable"
    gas_temperature_source: str = "unavailable"
    gas_property_names: Mapping[str, str] = dataclass_field(default_factory=dict)
    triangle_gradient_sources: Mapping[str, str] = dataclass_field(default_factory=dict)
    coordinate_system: str = "cartesian_xy"


# The compiled dataclass union is the only runtime representation.  Input
# mappings are consumed by the typed case loader and never cross this boundary.
CompiledRuntimeBackend = (
    RegularRectilinearCompiledBackend | TriangleMesh2DCompiledBackend
)


__all__ = (
    "CompiledRuntimeBackend",
    "RegularRectilinearCompiledBackend",
    "TriangleMesh2DCompiledBackend",
)
