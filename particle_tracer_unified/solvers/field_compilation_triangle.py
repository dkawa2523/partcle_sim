from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from particle_tracer_unified.core.datamodel import TriangleMeshField2D
from particle_tracer_unified.core.field_backend import (
    derived_quantity_names,
    triangle_mesh_gradient_source_report,
)
from particle_tracer_unified.core.field_sampling import (
    choose_electric_field_quantity_names,
    choose_velocity_quantity_names,
)
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    field_triangle_support_tolerance,
)

from .compiled_backend_types import TriangleMesh2DCompiledBackend
from .field_compilation_common import (
    GasDefaults,
    common_quantity_times,
    gas_property_quantity_names,
    merge_optional_quantity_times,
    require_positive_declared_gas_grid,
    vertex_time_grid,
)
from .forces import ForceRuntimeParameters
from .triangle_derived_fields import validate_triangle_gradient_geometry


def _validate_primary_force_inputs(
    force: ForceRuntimeParameters,
    velocity_names: tuple[str, ...],
    electric_names: tuple[str, ...],
    gas_names: Mapping[str, str],
) -> None:
    if force.lift_enabled and not velocity_names:
        raise ValueError("solver.forces.lift requires velocity field quantities")
    if force.thermophoresis_enabled and not gas_names.get("gas_temperature"):
        raise ValueError(
            "solver.forces.thermophoresis requires a temperature field quantity"
        )
    if force.dielectrophoresis_enabled and len(electric_names) < 2:
        raise ValueError(
            "solver.forces.dielectrophoresis requires electric field quantities"
        )


def _validate_flow_force_inputs(
    force: ForceRuntimeParameters,
    velocity_names: tuple[str, ...],
    gradient_sources: Mapping[str, str],
) -> None:
    if (
        force.pressure_gradient_enabled
        and gradient_sources.get("fluid_acceleration") == "unavailable"
    ):
        raise ValueError(
            "solver.forces.pressure_gradient requires velocity field quantities "
            "or exported fluid_accel_x/fluid_accel_y on triangle mesh"
        )
    if force.virtual_mass_enabled and not velocity_names:
        message = (
            "solver.forces.virtual_mass requires velocity field quantities on "
            "triangle mesh"
        )
        raise ValueError(message)


def _required_gradient_semantics(
    force: ForceRuntimeParameters,
    sources: Mapping[str, str],
) -> list[str]:
    required: list[str] = []
    if (
        force.pressure_gradient_enabled
        and sources.get("fluid_acceleration") == "triangle_p1_fallback"
    ):
        required.append("fluid_acceleration")
    if force.virtual_mass_enabled:
        required.append("flow_velocity_gradient")
    if force.thermophoresis_enabled and sources.get("grad_T") == "triangle_p1_fallback":
        required.append("temperature_gradient")
    if (
        force.dielectrophoresis_enabled
        and sources.get("grad_E2") == "triangle_p1_fallback"
    ):
        required.append("electric_magnitude_squared_gradient")
    if force.lift_enabled and sources.get("vorticity_z") == "triangle_p1_fallback":
        required.append("vorticity")
    return required


def _quantity_times(
    field: TriangleMeshField2D,
    velocity_names: tuple[str, ...],
    gas_names: Mapping[str, str],
    derived_names: Mapping[str, str],
) -> np.ndarray:
    times = np.asarray([0.0], dtype=np.float64)
    if velocity_names:
        times = common_quantity_times(field, velocity_names)
    times = merge_optional_quantity_times(field, times, tuple(gas_names.values()))
    return merge_optional_quantity_times(field, times, tuple(derived_names.values()))


def _velocity_grids(
    field: TriangleMeshField2D,
    names: tuple[str, ...],
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (int(max(1, times.size)), int(field.mesh_vertices.shape[0]))
    if not names:
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)
    return (
        vertex_time_grid(field.quantities[names[0]].data, times),
        vertex_time_grid(field.quantities[names[1]].data, times),
    )


def _gas_grids(
    field: TriangleMeshField2D,
    names: Mapping[str, str],
    times: np.ndarray,
    defaults: GasDefaults,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (int(max(1, times.size)), int(field.mesh_vertices.shape[0]))
    grids = {
        "gas_density": np.full(shape, defaults.density_kgm3, dtype=np.float64),
        "gas_mu": np.full(shape, defaults.dynamic_viscosity_Pas, dtype=np.float64),
        "gas_temperature": np.full(shape, defaults.temperature_K, dtype=np.float64),
    }
    vertex_support = np.zeros(field.mesh_vertices.shape[0], dtype=bool)
    vertex_support[np.asarray(field.mesh_triangles, dtype=np.int64).reshape(-1)] = True
    for target, name in names.items():
        values = vertex_time_grid(field.quantities[name].data, times)
        require_positive_declared_gas_grid(
            values,
            vertex_support,
            semantic_name=target,
            quantity_name=name,
        )
        grids[target] = values
    return grids["gas_density"], grids["gas_mu"], grids["gas_temperature"]


def _source(name: str, names: Mapping[str, str], default: str) -> str:
    return f"field:{names[name]}" if name in names else default


def compile_triangle_backend(
    field: TriangleMeshField2D,
    spatial_dim: int,
    enable_electric: bool,
    force: ForceRuntimeParameters,
    defaults: GasDefaults,
) -> TriangleMesh2DCompiledBackend:
    velocity_names = tuple(choose_velocity_quantity_names(field, spatial_dim))
    need_electric = enable_electric or force.dielectrophoresis_enabled
    electric_names = (
        tuple(choose_electric_field_quantity_names(field, spatial_dim))
        if need_electric
        else ()
    )
    gas_names = gas_property_quantity_names(field)
    derived_names = derived_quantity_names(field)
    gradient_sources = triangle_mesh_gradient_source_report(field)
    _validate_primary_force_inputs(force, velocity_names, electric_names, gas_names)
    _validate_flow_force_inputs(
        force,
        velocity_names,
        gradient_sources,
    )
    gradient_semantics = _required_gradient_semantics(force, gradient_sources)
    if gradient_semantics:
        validate_triangle_gradient_geometry(field, gradient_semantics[0])
    times = _quantity_times(field, velocity_names, gas_names, derived_names)
    ux, uy = _velocity_grids(field, velocity_names, times)
    density, viscosity, temperature = _gas_grids(field, gas_names, times, defaults)
    return TriangleMesh2DCompiledBackend(
        field=field,
        velocity_names=velocity_names,
        times=times,
        ux=ux,
        uy=uy,
        gas_density=density,
        gas_mu=viscosity,
        gas_temperature=temperature,
        mesh_vertices=np.asarray(field.mesh_vertices, dtype=np.float64),
        mesh_triangles=np.asarray(field.mesh_triangles, dtype=np.int32),
        accel_origin=np.asarray(field.accel_origin, dtype=np.float64),
        accel_cell_size=np.asarray(field.accel_cell_size, dtype=np.float64),
        accel_shape=tuple(np.asarray(field.accel_shape, dtype=np.int32).tolist()),
        accel_cell_offsets=np.asarray(field.accel_cell_offsets, dtype=np.int32),
        accel_triangle_indices=np.asarray(field.accel_triangle_indices, dtype=np.int32),
        support_tolerance_m=field_triangle_support_tolerance(field),
        electric_field_names=electric_names,
        gas_density_source=_source("gas_density", gas_names, defaults.density_source),
        gas_mu_source=_source("gas_mu", gas_names, defaults.viscosity_source),
        gas_temperature_source=_source(
            "gas_temperature", gas_names, defaults.temperature_source
        ),
        gas_property_names=dict(gas_names),
        triangle_gradient_sources=dict(gradient_sources),
        coordinate_system=str(field.coordinate_system),
    )
