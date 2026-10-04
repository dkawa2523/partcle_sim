"""Resolve shared gas, electric, and flow inputs for force-field assembly."""

from __future__ import annotations

import numpy as np

from particle_tracer_unified.domain import StageFields

from .base_field_sampling import (
    sample_compiled_electric_vectors,
    sample_compiled_gas_properties_vectors,
)
from .compiled_backend_types import CompiledRuntimeBackend
from .forces import ForceRuntimeParameters


def _force_gas_fields(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
    *,
    need_density: bool,
    need_mu: bool,
    need_temperature: bool,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
    base_fields: StageFields | None,
) -> dict[str, np.ndarray]:
    requested = {
        "gas_density": bool(need_density),
        "dynamic_viscosity": bool(need_mu),
        "temperature": bool(need_temperature),
    }
    values: dict[str, np.ndarray] = {}
    if base_fields is not None:
        for name, needed in requested.items():
            if needed and name in base_fields.values:
                values[name] = np.asarray(base_fields.values[name], dtype=np.float64)
    missing = tuple(
        name for name, needed in requested.items() if needed and name not in values
    )
    if not missing:
        return values
    density, viscosity, temperature = sample_compiled_gas_properties_vectors(
        compiled,
        int(spatial_dim),
        float(t_eval),
        points,
        fallback_density_kgm3=float(fallback_density_kgm3),
        fallback_mu_pas=float(fallback_mu_pas),
        fallback_temperature_K=float(fallback_temperature_K),
    )
    sampled = {
        "gas_density": density,
        "dynamic_viscosity": viscosity,
        "temperature": temperature,
    }
    for name in missing:
        values[name] = np.asarray(sampled[name], dtype=np.float64)
    return values


def _force_gas_requirements(
    params: ForceRuntimeParameters,
) -> tuple[bool, bool, bool]:
    need_density = bool(
        params.pressure_gradient_enabled
        or params.virtual_mass_enabled
        or params.thermophoresis_enabled
        or params.lift_enabled
    )
    need_viscosity = bool(params.thermophoresis_enabled or params.lift_enabled)
    need_temperature = bool(params.thermophoresis_enabled)
    return need_density, need_viscosity, need_temperature


def _electric_force_field(
    backend: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
    base_fields: StageFields | None,
) -> np.ndarray:
    if base_fields is not None and "electric_field" in base_fields.values:
        electric = np.asarray(
            base_fields.values["electric_field"],
            dtype=np.float64,
        )
    else:
        electric = sample_compiled_electric_vectors(
            backend,
            int(spatial_dim),
            float(t_eval),
            points,
        )
    if electric is None:
        raise ValueError("electric force requires exported electric field components")
    return electric


def _preferred_flow_velocity(
    supplied_velocity: np.ndarray | None,
    base_fields: StageFields | None,
    sampled_velocity: np.ndarray,
) -> np.ndarray:
    if supplied_velocity is not None:
        return np.asarray(supplied_velocity, dtype=np.float64)
    if base_fields is not None and "flow_velocity" in base_fields.values:
        return np.asarray(base_fields.values["flow_velocity"], dtype=np.float64)
    return sampled_velocity
