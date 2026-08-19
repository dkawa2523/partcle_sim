"""Assemble derived force fields from a two-dimensional triangle mesh."""

from __future__ import annotations

import numpy as np

from particle_tracer_unified.core.field_backend import derived_quantity_names
from particle_tracer_unified.domain import StageFields

from ._force_field_sources import (
    _electric_force_field,
    _force_gas_fields,
    _force_gas_requirements,
    _preferred_flow_velocity,
)
from .base_field_sampling import (
    sample_triangle_scalar_gradient,
    sample_triangle_scalar_value,
    sample_triangle_vector,
    sample_triangle_velocity_terms,
)
from .compiled_backend_types import TriangleMesh2DCompiledBackend
from .forces import ForceRuntimeParameters


def _triangle_needs_velocity_terms(
    params: ForceRuntimeParameters,
    derived: dict[str, str],
) -> bool:
    has_exported_acceleration = {
        "fluid_accel_x",
        "fluid_accel_y",
    } <= set(derived)
    return bool(
        params.virtual_mass_enabled
        or params.lift_enabled
        or (params.pressure_gradient_enabled and not has_exported_acceleration)
    )


def _triangle_velocity_terms(
    backend: TriangleMesh2DCompiledBackend,
    t_eval: float,
    points: np.ndarray,
    *,
    required: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_flow = np.zeros((points.shape[0], 2), dtype=np.float64)
    flow_time_derivative = np.zeros_like(sampled_flow)
    flow_velocity_gradient = np.zeros((points.shape[0], 2, 2), dtype=np.float64)
    if not required:
        return sampled_flow, flow_time_derivative, flow_velocity_gradient
    terms = [
        sample_triangle_velocity_terms(
            backend,
            float(t_eval),
            point,
            row_index=row_index,
        )
        for row_index, point in enumerate(points)
    ]
    return (
        np.asarray([item[0] for item in terms], dtype=np.float64),
        np.asarray([item[1] for item in terms], dtype=np.float64),
        np.asarray([item[2] for item in terms], dtype=np.float64),
    )


def _triangle_fluid_acceleration(
    backend: TriangleMesh2DCompiledBackend,
    t_eval: float,
    points: np.ndarray,
    derived: dict[str, str],
    sampled_flow: np.ndarray,
    flow_time_derivative: np.ndarray,
    flow_velocity_gradient: np.ndarray,
) -> np.ndarray:
    if "fluid_accel_x" in derived and "fluid_accel_y" in derived:
        return sample_triangle_vector(
            backend,
            (derived["fluid_accel_x"], derived["fluid_accel_y"]),
            "fluid_acceleration",
            float(t_eval),
            points,
        )
    return flow_time_derivative + np.einsum(
        "nij,nj->ni",
        flow_velocity_gradient,
        sampled_flow,
    )


def _triangle_temperature_gradient(
    backend: TriangleMesh2DCompiledBackend,
    t_eval: float,
    points: np.ndarray,
    derived: dict[str, str],
) -> np.ndarray:
    if "grad_T_x" in derived and "grad_T_y" in derived:
        return sample_triangle_vector(
            backend,
            (derived["grad_T_x"], derived["grad_T_y"]),
            "temperature_gradient",
            float(t_eval),
            points,
        )
    temperature_name = backend.gas_property_names.get("gas_temperature")
    if not temperature_name:
        raise ValueError("thermophoresis requires a temperature field quantity")
    return np.asarray(
        [
            sample_triangle_scalar_gradient(
                backend,
                temperature_name,
                float(t_eval),
                point,
                semantic_quantity="temperature_gradient",
                row_index=row_index,
            )
            for row_index, point in enumerate(points)
        ],
        dtype=np.float64,
    )


def _triangle_electric_magnitude_squared_gradient(
    backend: TriangleMesh2DCompiledBackend,
    t_eval: float,
    points: np.ndarray,
    derived: dict[str, str],
) -> np.ndarray:
    if "grad_E2_x" in derived and "grad_E2_y" in derived:
        return sample_triangle_vector(
            backend,
            (derived["grad_E2_x"], derived["grad_E2_y"]),
            "electric_magnitude_squared_gradient",
            float(t_eval),
            points,
        )
    names = backend.electric_field_names
    if len(names) != 2:
        raise ValueError("dielectrophoresis requires electric field components")
    gradients = np.empty((points.shape[0], 2), dtype=np.float64)
    for index, point in enumerate(points):
        ex = sample_triangle_scalar_value(
            backend,
            names[0],
            float(t_eval),
            point,
            semantic_quantity="electric_field.x",
            row_index=index,
        )
        ey = sample_triangle_scalar_value(
            backend,
            names[1],
            float(t_eval),
            point,
            semantic_quantity="electric_field.y",
            row_index=index,
        )
        gradients[index] = 2.0 * ex * sample_triangle_scalar_gradient(
            backend,
            names[0],
            float(t_eval),
            point,
            semantic_quantity="electric_magnitude_squared_gradient",
            row_index=index,
        ) + 2.0 * ey * sample_triangle_scalar_gradient(
            backend,
            names[1],
            float(t_eval),
            point,
            semantic_quantity="electric_magnitude_squared_gradient",
            row_index=index,
        )
    return gradients


def _triangle_vorticity(
    backend: TriangleMesh2DCompiledBackend,
    t_eval: float,
    points: np.ndarray,
    derived: dict[str, str],
    flow_velocity_gradient: np.ndarray,
) -> np.ndarray:
    vorticity = np.zeros((points.shape[0], 3), dtype=np.float64)
    if "vorticity_z" in derived:
        vorticity[:, 2] = sample_triangle_vector(
            backend,
            (derived["vorticity_z"],),
            "vorticity",
            float(t_eval),
            points,
        )[:, 0]
    else:
        vorticity[:, 2] = (
            flow_velocity_gradient[:, 1, 0] - flow_velocity_gradient[:, 0, 1]
        )
    return vorticity


def _triangle_force_fields(
    backend: TriangleMesh2DCompiledBackend,
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
    values: dict[str, np.ndarray] = {}
    derived = derived_quantity_names(backend.field)
    sampled_flow, flow_time_derivative, flow_velocity_gradient = (
        _triangle_velocity_terms(
            backend,
            float(t_eval),
            points,
            required=_triangle_needs_velocity_terms(params, derived),
        )
    )

    if include_electric:
        values["electric_field"] = _electric_force_field(
            backend,
            2,
            float(t_eval),
            points,
            base_fields,
        )

    need_density, need_mu, need_temperature = _force_gas_requirements(params)
    values.update(
        _force_gas_fields(
            backend,
            2,
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
        values["fluid_acceleration"] = _triangle_fluid_acceleration(
            backend,
            float(t_eval),
            points,
            derived,
            sampled_flow,
            flow_time_derivative,
            flow_velocity_gradient,
        )

    if params.virtual_mass_enabled:
        values["flow_time_derivative"] = flow_time_derivative
        values["flow_velocity_gradient"] = flow_velocity_gradient

    if params.thermophoresis_enabled:
        values["temperature_gradient"] = _triangle_temperature_gradient(
            backend,
            float(t_eval),
            points,
            derived,
        )

    if params.dielectrophoresis_enabled:
        values["electric_magnitude_squared_gradient"] = (
            _triangle_electric_magnitude_squared_gradient(
                backend,
                float(t_eval),
                points,
                derived,
            )
        )

    if params.lift_enabled:
        values["flow_velocity"] = _preferred_flow_velocity(
            flow_velocity,
            base_fields,
            sampled_flow,
        )
        values["vorticity"] = _triangle_vorticity(
            backend,
            float(t_eval),
            points,
            derived,
            flow_velocity_gradient,
        )
    return values
