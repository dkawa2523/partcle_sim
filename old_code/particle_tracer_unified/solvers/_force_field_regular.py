"""Assemble derived force fields from a regular rectilinear backend."""

from __future__ import annotations

import numpy as np

from particle_tracer_unified.domain import StageFields

from ._force_field_sources import (
    _electric_force_field,
    _force_gas_fields,
    _force_gas_requirements,
)
from .base_field_sampling import (
    sample_compiled_flow_vectors,
    sample_regular_components,
    sample_regular_velocity_gradient,
)
from .compiled_backend_types import RegularRectilinearCompiledBackend
from .forces import ForceRuntimeParameters


def _regular_spatial_components(
    spatial_dim: int,
    component_x: np.ndarray | None,
    component_y: np.ndarray | None,
    component_z: np.ndarray | None,
) -> tuple[np.ndarray | None, ...]:
    if int(spatial_dim) == 2:
        return component_x, component_y
    return component_x, component_y, component_z


def _required_regular_components(
    backend: RegularRectilinearCompiledBackend,
    components: tuple[np.ndarray | None, ...],
    t_eval: float,
    points: np.ndarray,
    *,
    error_message: str,
) -> np.ndarray:
    sampled = sample_regular_components(
        backend,
        components,
        float(t_eval),
        points,
    )
    if sampled is None:
        raise ValueError(error_message)
    return sampled


def _regular_virtual_mass_fields(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time_derivative = sample_regular_components(
        backend,
        _regular_spatial_components(
            spatial_dim,
            backend.du_dt_x,
            backend.du_dt_y,
            backend.du_dt_z,
        ),
        float(t_eval),
        points,
    )
    gradient = sample_regular_velocity_gradient(
        backend,
        int(spatial_dim),
        float(t_eval),
        points,
    )
    if time_derivative is None or gradient is None:
        raise ValueError(
            "virtual_mass requires flow time derivative and velocity gradient"
        )
    return time_derivative, gradient


def _regular_flow_velocity(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
    supplied_velocity: np.ndarray | None,
    base_fields: StageFields | None,
) -> np.ndarray:
    if supplied_velocity is not None:
        return np.asarray(supplied_velocity, dtype=np.float64)
    if base_fields is not None and "flow_velocity" in base_fields.values:
        return np.asarray(base_fields.values["flow_velocity"], dtype=np.float64)
    return sample_compiled_flow_vectors(
        backend,
        int(spatial_dim),
        float(t_eval),
        points,
    )


def _regular_vorticity(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
) -> np.ndarray:
    if int(spatial_dim) == 2:
        omega_z = _required_regular_components(
            backend,
            (backend.vorticity_z,),
            float(t_eval),
            points,
            error_message="2D lift requires scalar vorticity",
        )
        vorticity = np.zeros((points.shape[0], 3), dtype=np.float64)
        vorticity[:, 2] = omega_z[:, 0]
        return vorticity
    return _required_regular_components(
        backend,
        (backend.vorticity_x, backend.vorticity_y, backend.vorticity_z),
        float(t_eval),
        points,
        error_message="3D lift requires vector vorticity",
    )


def _regular_force_fields(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
    *,
    params: ForceRuntimeParameters,
    include_electric: bool,
    flow_velocity: np.ndarray | None,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
    base_fields: StageFields | None,
) -> dict[str, np.ndarray]:
    dim = int(spatial_dim)
    values: dict[str, np.ndarray] = {}
    if include_electric:
        values["electric_field"] = _electric_force_field(
            backend,
            dim,
            float(t_eval),
            points,
            base_fields,
        )

    need_density, need_mu, need_temperature = _force_gas_requirements(params)
    values.update(
        _force_gas_fields(
            backend,
            dim,
            float(t_eval),
            points,
            need_density=need_density,
            need_mu=need_mu,
            need_temperature=need_temperature,
            fallback_density_kgm3=float(fallback_density_kgm3),
            fallback_mu_pas=float(fallback_mu_pas),
            fallback_temperature_K=float(fallback_temperature_K),
            base_fields=base_fields,
        )
    )

    if params.pressure_gradient_enabled:
        values["fluid_acceleration"] = _required_regular_components(
            backend,
            _regular_spatial_components(
                dim,
                backend.fluid_accel_x,
                backend.fluid_accel_y,
                backend.fluid_accel_z,
            ),
            float(t_eval),
            points,
            error_message="pressure_gradient requires fluid material acceleration",
        )

    if params.virtual_mass_enabled:
        time_derivative, gradient = _regular_virtual_mass_fields(
            backend,
            dim,
            float(t_eval),
            points,
        )
        values["flow_time_derivative"] = time_derivative
        values["flow_velocity_gradient"] = gradient

    if params.thermophoresis_enabled:
        values["temperature_gradient"] = _required_regular_components(
            backend,
            _regular_spatial_components(
                dim,
                backend.grad_T_x,
                backend.grad_T_y,
                backend.grad_T_z,
            ),
            float(t_eval),
            points,
            error_message="thermophoresis requires temperature gradient",
        )

    if params.dielectrophoresis_enabled:
        values["electric_magnitude_squared_gradient"] = _required_regular_components(
            backend,
            _regular_spatial_components(
                dim,
                backend.grad_E2_x,
                backend.grad_E2_y,
                backend.grad_E2_z,
            ),
            float(t_eval),
            points,
            error_message=(
                "dielectrophoresis requires electric magnitude-squared gradient"
            ),
        )

    if params.lift_enabled:
        values["flow_velocity"] = _regular_flow_velocity(
            backend,
            dim,
            float(t_eval),
            points,
            flow_velocity,
            base_fields,
        )
        values["vorticity"] = _regular_vorticity(
            backend,
            dim,
            float(t_eval),
            points,
        )
    return values
