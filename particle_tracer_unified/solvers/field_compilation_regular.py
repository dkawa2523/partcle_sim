from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from particle_tracer_unified.core.datamodel import RegularFieldND
from particle_tracer_unified.core.field_backend_reporting import derived_quantity_names
from particle_tracer_unified.core.field_sampling import (
    choose_electric_field_quantity_names,
    choose_velocity_quantity_names,
)

from .compiled_backend_types import RegularRectilinearCompiledBackend
from .field_compilation_common import (
    GasDefaults,
    backend_time_grid,
    common_quantity_times,
    curl_from_velocity_grids,
    gas_property_quantity_names,
    gradient_time_grid,
    merge_optional_quantity_times,
    require_positive_declared_gas_grid,
    time_derivative_time_grid,
)
from .forces import ForceRuntimeParameters

_DERIVED_GROUPS = (
    ("thermophoresis_enabled", ("grad_T_x", "grad_T_y")),
    ("dielectrophoresis_enabled", ("grad_E2_x", "grad_E2_y")),
    ("pressure_gradient_enabled", ("fluid_accel_x", "fluid_accel_y")),
    ("lift_enabled", ("vorticity_z",)),
)


def _empty_backend(
    runtime: Any,
    spatial_dim: int,
    defaults: GasDefaults,
    *,
    allocate_full_grid: bool = True,
) -> RegularRectilinearCompiledBackend:
    geometry = runtime.geometry_provider.geometry
    axes = tuple(np.asarray(axis, dtype=np.float64) for axis in geometry.axes)
    valid_mask = np.asarray(geometry.valid_mask, dtype=bool)
    spatial_shape = (
        tuple(len(axis) for axis in axes)
        if bool(allocate_full_grid)
        else tuple(1 for _axis in axes)
    )
    shape = (1, *spatial_shape)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=np.asarray([0.0], dtype=np.float64),
        ux=np.zeros(shape, dtype=np.float64),
        uy=np.zeros(shape, dtype=np.float64),
        uz=np.zeros(shape, dtype=np.float64) if spatial_dim == 3 else None,
        gas_density=np.full(shape, defaults.density_kgm3, dtype=np.float64),
        gas_mu=np.full(shape, defaults.dynamic_viscosity_Pas, dtype=np.float64),
        gas_temperature=np.full(shape, defaults.temperature_K, dtype=np.float64),
        valid_mask=valid_mask,
        core_valid_mask=valid_mask,
        gas_density_source=defaults.density_source,
        gas_mu_source=defaults.viscosity_source,
        gas_temperature_source=defaults.temperature_source,
        coordinate_system=str(
            getattr(
                runtime,
                "coordinate_system",
                getattr(geometry, "coordinate_system", "cartesian_xy"),
            )
        ),
    )


def _quantity_times(
    field: RegularFieldND,
    velocity_names: tuple[str, ...],
    electric_names: tuple[str, ...],
    gas_names: Mapping[str, str],
    derived_names: Mapping[str, str],
) -> np.ndarray:
    names = velocity_names + electric_names
    times = (
        common_quantity_times(field, names)
        if names
        else np.asarray([0.0], dtype=np.float64)
    )
    times = merge_optional_quantity_times(field, times, tuple(gas_names.values()))
    return merge_optional_quantity_times(field, times, tuple(derived_names.values()))


def _complete_derived_quantity_names(
    field: RegularFieldND,
    spatial_dim: int,
    force: ForceRuntimeParameters,
) -> dict[str, str]:
    if int(spatial_dim) != 2:
        return {}
    resolved = derived_quantity_names(field)
    return {
        target: resolved[target]
        for flag, group in _DERIVED_GROUPS
        if getattr(force, flag) and all(target in resolved for target in group)
        for target in group
    }


def _declared_or_filled_grid(
    values: np.ndarray,
    *,
    declared: bool,
    shape: tuple[int, ...],
    fallback: float,
) -> np.ndarray:
    if declared:
        return values
    return np.full(shape, fallback, dtype=np.float64)


def _resolved_z_velocity_grid(
    backend: RegularRectilinearCompiledBackend,
    *,
    spatial_dim: int,
    velocity_declared: bool,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if spatial_dim != 3:
        return None
    if velocity_declared:
        return backend.uz
    return np.zeros(shape, dtype=np.float64)


def _resize_for_field(
    backend: RegularRectilinearCompiledBackend,
    field: RegularFieldND,
    spatial_dim: int,
    times: np.ndarray,
    defaults: GasDefaults,
    velocity_names: tuple[str, ...],
    gas_names: Mapping[str, str],
) -> RegularRectilinearCompiledBackend:
    shape = (int(max(1, times.size)), *(len(axis) for axis in backend.axes))
    velocity_declared = bool(velocity_names)
    return replace(
        backend,
        times=times,
        ux=_declared_or_filled_grid(
            backend.ux,
            declared=velocity_declared,
            shape=shape,
            fallback=0.0,
        ),
        uy=_declared_or_filled_grid(
            backend.uy,
            declared=velocity_declared,
            shape=shape,
            fallback=0.0,
        ),
        uz=_resolved_z_velocity_grid(
            backend,
            spatial_dim=spatial_dim,
            velocity_declared=velocity_declared,
            shape=shape,
        ),
        gas_density=_declared_or_filled_grid(
            backend.gas_density,
            declared="gas_density" in gas_names,
            shape=shape,
            fallback=defaults.density_kgm3,
        ),
        gas_mu=_declared_or_filled_grid(
            backend.gas_mu,
            declared="gas_mu" in gas_names,
            shape=shape,
            fallback=defaults.dynamic_viscosity_Pas,
        ),
        gas_temperature=_declared_or_filled_grid(
            backend.gas_temperature,
            declared="gas_temperature" in gas_names,
            shape=shape,
            fallback=defaults.temperature_K,
        ),
        valid_mask=np.asarray(field.valid_mask, dtype=bool),
        core_valid_mask=np.asarray(
            field.core_valid_mask
            if field.core_valid_mask is not None
            else field.valid_mask,
            dtype=bool,
        ),
    )


def _load_primary_fields(
    backend: RegularRectilinearCompiledBackend,
    field: RegularFieldND,
    spatial_dim: int,
    velocity_names: tuple[str, ...],
    electric_names: tuple[str, ...],
) -> RegularRectilinearCompiledBackend:
    ux, uy, uz = backend.ux, backend.uy, backend.uz
    electric_x = electric_y = electric_z = None
    if velocity_names:
        ux = backend_time_grid(
            field.quantities[velocity_names[0]].data, spatial_dim, backend.times
        )
        uy = backend_time_grid(
            field.quantities[velocity_names[1]].data, spatial_dim, backend.times
        )
        if spatial_dim == 3:
            uz = backend_time_grid(
                field.quantities[velocity_names[2]].data, 3, backend.times
            )
    if electric_names:
        electric_x = backend_time_grid(
            field.quantities[electric_names[0]].data, spatial_dim, backend.times
        )
        electric_y = backend_time_grid(
            field.quantities[electric_names[1]].data, spatial_dim, backend.times
        )
        if spatial_dim == 3:
            electric_z = backend_time_grid(
                field.quantities[electric_names[2]].data, 3, backend.times
            )
    return replace(
        backend,
        ux=ux,
        uy=uy,
        uz=uz,
        electric_x=electric_x,
        electric_y=electric_y,
        electric_z=electric_z,
        acceleration_source="particle_charge_electric_field"
        if electric_names
        else "none",
        electric_field_names=electric_names,
    )


def _load_gas_fields(
    backend: RegularRectilinearCompiledBackend,
    field: RegularFieldND,
    spatial_dim: int,
    names: Mapping[str, str],
) -> RegularRectilinearCompiledBackend:
    density, viscosity, temperature = (
        backend.gas_density,
        backend.gas_mu,
        backend.gas_temperature,
    )
    density_source = backend.gas_density_source
    viscosity_source = backend.gas_mu_source
    temperature_source = backend.gas_temperature_source
    for target, name in names.items():
        values = backend_time_grid(
            field.quantities[name].data, spatial_dim, backend.times
        )
        require_positive_declared_gas_grid(
            values,
            backend.valid_mask,
            semantic_name=target,
            quantity_name=name,
        )
        if target == "gas_density":
            density, density_source = values, f"field:{name}"
        elif target == "gas_mu":
            viscosity, viscosity_source = values, f"field:{name}"
        elif target == "gas_temperature":
            temperature, temperature_source = values, f"field:{name}"
    return replace(
        backend,
        gas_density=density,
        gas_mu=viscosity,
        gas_temperature=temperature,
        gas_density_source=density_source,
        gas_mu_source=viscosity_source,
        gas_temperature_source=temperature_source,
    )


def _load_exported_derived_fields(
    backend: RegularRectilinearCompiledBackend,
    field: RegularFieldND,
    spatial_dim: int,
    names: Mapping[str, str],
) -> RegularRectilinearCompiledBackend:
    def grid(target: str) -> np.ndarray:
        return backend_time_grid(
            field.quantities[names[target]].data, spatial_dim, backend.times
        )

    if "grad_T_x" in names:
        backend = replace(backend, grad_T_x=grid("grad_T_x"), grad_T_y=grid("grad_T_y"))
    if "grad_E2_x" in names:
        backend = replace(
            backend, grad_E2_x=grid("grad_E2_x"), grad_E2_y=grid("grad_E2_y")
        )
    if "fluid_accel_x" in names:
        backend = replace(
            backend,
            fluid_accel_x=grid("fluid_accel_x"),
            fluid_accel_y=grid("fluid_accel_y"),
        )
    if "vorticity_z" in names:
        backend = replace(backend, vorticity_z=grid("vorticity_z"))
    return backend


def _with_temperature_gradient(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
) -> RegularRectilinearCompiledBackend:
    if spatial_dim == 2 and backend.grad_T_x is not None:
        return backend
    gradient = gradient_time_grid(backend.gas_temperature, backend.axes)
    return replace(
        backend,
        grad_T_x=gradient[0],
        grad_T_y=gradient[1],
        grad_T_z=gradient[2] if spatial_dim == 3 else None,
    )


def _with_electric_magnitude_gradient(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
) -> RegularRectilinearCompiledBackend:
    if spatial_dim == 2 and backend.grad_E2_x is not None:
        return backend
    if backend.electric_x is None or backend.electric_y is None:
        raise ValueError(
            "solver.forces.dielectrophoresis requires electric field quantities"
        )
    magnitude_squared = backend.electric_x**2 + backend.electric_y**2
    if spatial_dim == 3:
        if backend.electric_z is None:
            raise ValueError(
                "solver.forces.dielectrophoresis requires 3D electric field quantities"
            )
        magnitude_squared = magnitude_squared + backend.electric_z**2
    gradient = gradient_time_grid(magnitude_squared, backend.axes)
    return replace(
        backend,
        grad_E2_x=gradient[0],
        grad_E2_y=gradient[1],
        grad_E2_z=gradient[2] if spatial_dim == 3 else None,
    )


def _with_vorticity(
    backend: RegularRectilinearCompiledBackend,
) -> RegularRectilinearCompiledBackend:
    if len(backend.axes) == 2 and backend.vorticity_z is not None:
        return backend
    x, y, z = curl_from_velocity_grids(backend.ux, backend.uy, backend.uz, backend.axes)
    return replace(backend, vorticity_x=x, vorticity_y=y, vorticity_z=z)


def _with_2d_flow_derivatives(
    backend: RegularRectilinearCompiledBackend,
    du_dt_x: np.ndarray,
    du_dt_y: np.ndarray,
    pressure_gradient_enabled: bool,
) -> RegularRectilinearCompiledBackend:
    grad_ux_x, grad_ux_y = gradient_time_grid(backend.ux, backend.axes)
    grad_uy_x, grad_uy_y = gradient_time_grid(backend.uy, backend.axes)
    fluid_accel_x = backend.fluid_accel_x
    fluid_accel_y = backend.fluid_accel_y
    if pressure_gradient_enabled:
        fluid_accel_x = np.asarray(
            du_dt_x + backend.ux * grad_ux_x + backend.uy * grad_ux_y,
            dtype=np.float64,
        )
        fluid_accel_y = np.asarray(
            du_dt_y + backend.ux * grad_uy_x + backend.uy * grad_uy_y,
            dtype=np.float64,
        )
    return replace(
        backend,
        du_dt_x=du_dt_x,
        du_dt_y=du_dt_y,
        grad_ux_x=grad_ux_x,
        grad_ux_y=grad_ux_y,
        grad_uy_x=grad_uy_x,
        grad_uy_y=grad_uy_y,
        fluid_accel_x=fluid_accel_x,
        fluid_accel_y=fluid_accel_y,
    )


def _with_3d_flow_derivatives(
    backend: RegularRectilinearCompiledBackend,
    du_dt_x: np.ndarray,
    du_dt_y: np.ndarray,
    pressure_gradient_enabled: bool,
) -> RegularRectilinearCompiledBackend:
    if backend.uz is None:
        raise ValueError("3D flow derivatives require a z velocity grid")
    du_dt_z = time_derivative_time_grid(backend.uz, backend.times)
    grad_ux = gradient_time_grid(backend.ux, backend.axes)
    grad_uy = gradient_time_grid(backend.uy, backend.axes)
    grad_uz = gradient_time_grid(backend.uz, backend.axes)
    fluid_accel_x = fluid_accel_y = fluid_accel_z = None
    if pressure_gradient_enabled:
        fluid_accel_x = np.asarray(
            du_dt_x
            + backend.ux * grad_ux[0]
            + backend.uy * grad_ux[1]
            + backend.uz * grad_ux[2],
            dtype=np.float64,
        )
        fluid_accel_y = np.asarray(
            du_dt_y
            + backend.ux * grad_uy[0]
            + backend.uy * grad_uy[1]
            + backend.uz * grad_uy[2],
            dtype=np.float64,
        )
        fluid_accel_z = np.asarray(
            du_dt_z
            + backend.ux * grad_uz[0]
            + backend.uy * grad_uz[1]
            + backend.uz * grad_uz[2],
            dtype=np.float64,
        )
    return replace(
        backend,
        du_dt_x=du_dt_x,
        du_dt_y=du_dt_y,
        du_dt_z=du_dt_z,
        grad_ux_x=grad_ux[0],
        grad_ux_y=grad_ux[1],
        grad_ux_z=grad_ux[2],
        grad_uy_x=grad_uy[0],
        grad_uy_y=grad_uy[1],
        grad_uy_z=grad_uy[2],
        grad_uz_x=grad_uz[0],
        grad_uz_y=grad_uz[1],
        grad_uz_z=grad_uz[2],
        fluid_accel_x=fluid_accel_x,
        fluid_accel_y=fluid_accel_y,
        fluid_accel_z=fluid_accel_z,
    )


def _with_flow_derivatives(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
    pressure_gradient_enabled: bool,
) -> RegularRectilinearCompiledBackend:
    du_dt_x = time_derivative_time_grid(backend.ux, backend.times)
    du_dt_y = time_derivative_time_grid(backend.uy, backend.times)
    if spatial_dim == 2:
        return _with_2d_flow_derivatives(
            backend,
            du_dt_x,
            du_dt_y,
            pressure_gradient_enabled,
        )
    return _with_3d_flow_derivatives(
        backend,
        du_dt_x,
        du_dt_y,
        pressure_gradient_enabled,
    )


def _with_requested_flow_derivatives(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
    force: ForceRuntimeParameters,
    velocity_names: tuple[str, ...],
) -> RegularRectilinearCompiledBackend:
    pressure_fallback = force.pressure_gradient_enabled and not (
        spatial_dim == 2
        and backend.fluid_accel_x is not None
        and backend.fluid_accel_y is not None
    )
    if not force.virtual_mass_enabled and not pressure_fallback:
        return backend
    if not velocity_names:
        message = (
            "solver.forces pressure_gradient/virtual_mass require velocity "
            "field quantities"
        )
        raise ValueError(message)
    return _with_flow_derivatives(
        backend,
        spatial_dim,
        pressure_fallback,
    )


def compile_regular_backend(
    runtime: Any,
    spatial_dim: int,
    enable_electric: bool,
    force: ForceRuntimeParameters,
    defaults: GasDefaults,
) -> RegularRectilinearCompiledBackend:
    backend = _empty_backend(
        runtime,
        spatial_dim,
        defaults,
        allocate_full_grid=runtime.field_provider is None,
    )
    if runtime.field_provider is None:
        return backend
    field = runtime.field_provider.field
    if not isinstance(field, RegularFieldND):
        raise TypeError(f"Unsupported regular field type: {type(field).__name__}")
    velocity_names = tuple(choose_velocity_quantity_names(field, spatial_dim))
    need_electric = enable_electric or force.dielectrophoresis_enabled
    electric_names = (
        tuple(choose_electric_field_quantity_names(field, spatial_dim))
        if need_electric
        else ()
    )
    gas_names = gas_property_quantity_names(field)
    derived_names = _complete_derived_quantity_names(field, spatial_dim, force)
    times = _quantity_times(
        field,
        velocity_names,
        electric_names,
        gas_names,
        derived_names,
    )
    backend = replace(
        backend,
        times=times,
        valid_mask=np.asarray(field.valid_mask, dtype=bool),
        core_valid_mask=np.asarray(
            field.core_valid_mask
            if field.core_valid_mask is not None
            else field.valid_mask,
            dtype=bool,
        ),
    )
    backend = _load_primary_fields(
        backend, field, spatial_dim, velocity_names, electric_names
    )
    backend = _load_gas_fields(backend, field, spatial_dim, gas_names)
    backend = _resize_for_field(
        backend,
        field,
        spatial_dim,
        times,
        defaults,
        velocity_names,
        gas_names,
    )
    backend = _load_exported_derived_fields(
        backend,
        field,
        spatial_dim,
        derived_names,
    )
    if force.thermophoresis_enabled:
        backend = _with_temperature_gradient(backend, spatial_dim)
    if force.dielectrophoresis_enabled:
        backend = _with_electric_magnitude_gradient(backend, spatial_dim)
    if force.lift_enabled:
        backend = _with_vorticity(backend)
    return _with_requested_flow_derivatives(backend, spatial_dim, force, velocity_names)
