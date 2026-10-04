"""Sample and assemble deterministic first-step force contributions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
)
from particle_tracer_unified.core.datamodel import ParticleTable, SolverContext
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from particle_tracer_unified.solvers.base_field_sampling import (
    sample_compiled_flow_vectors,
    sample_compiled_gas_properties_vectors,
    sample_compiled_valid_mask_statuses,
)
from particle_tracer_unified.solvers.compiled_backend_types import (
    CompiledRuntimeBackend,
)
from particle_tracer_unified.solvers.drag_models import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_NONE,
    effective_tau_from_drag_model,
    stokes_relaxation_time,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.force_field_assembly import (
    sample_compiled_acceleration_vectors,
)
from particle_tracer_unified.solvers.forces import (
    ForceRuntimeParameters,
    compile_force_runtime_parameters,
)
from particle_tracer_unified.solvers.particle_state import (
    ParticleStaticArrays,
    static_arrays_from_particles,
)

_AMU_KG = 1.66053906660e-27
_STATUS_NAMES = {
    int(VALID_MASK_STATUS_CLEAN): "clean",
    int(VALID_MASK_STATUS_MIXED_STENCIL): "mixed_stencil",
    int(VALID_MASK_STATUS_HARD_INVALID): "hard_invalid",
}
_FORCE_PREFIXES = (
    "drag",
    "electric",
    "thermo",
    "dielectrophoretic",
    "lift",
    "pressure_gradient",
    "virtual_mass",
    "brownian",
    "external",
    "total",
)

_ComponentSampler = Callable[..., np.ndarray]


class _SampledForceState(NamedTuple):
    positions: np.ndarray
    velocities: np.ndarray
    time_s: float
    status_codes: np.ndarray
    flow: np.ndarray
    gas_density: np.ndarray
    gas_viscosity: np.ndarray
    gas_temperature: np.ndarray


def _single_component_runtime(
    base: ForceRuntimeParameters, name: str
) -> ForceRuntimeParameters:
    flags = {
        "thermophoresis_enabled": False,
        "dielectrophoresis_enabled": False,
        "lift_enabled": False,
        "pressure_gradient_enabled": False,
        "virtual_mass_enabled": False,
        "gravity_buoyancy_enabled": False,
    }
    if name in flags:
        flags[name] = True
    aliases = {
        "thermophoresis": "thermophoresis_enabled",
        "dielectrophoresis": "dielectrophoresis_enabled",
        "lift": "lift_enabled",
        "pressure_gradient": "pressure_gradient_enabled",
        "virtual_mass": "virtual_mass_enabled",
    }
    key = aliases.get(name)
    if key is not None:
        flags[key] = True
    return replace(base, **flags)


def _q_over_m(charge: np.ndarray, mass: np.ndarray) -> np.ndarray:
    q = np.asarray(charge, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    return np.where(np.isfinite(m) & (np.abs(m) > 1.0e-300), q / m, np.nan)


def _sample_component(
    *,
    compiled: Any,
    context: SolverContext,
    positions: np.ndarray,
    velocities: np.ndarray,
    time_s: float,
    force_runtime: ForceRuntimeParameters,
    electric_q_over_m: np.ndarray | None,
) -> np.ndarray:
    particles = context.particles
    return sample_compiled_acceleration_vectors(
        compiled,
        int(context.spatial_dim),
        float(time_s),
        positions,
        electric_q_over_m=electric_q_over_m,
        force_runtime=force_runtime,
        particle_diameter=particles.diameter,
        particle_density=particles.density,
        particle_mass=particles.mass,
        dep_particle_rel_permittivity=particles.dep_particle_rel_permittivity,
        thermophoretic_coeff=particles.thermophoretic_coeff,
        velocity=velocities,
        gas_density_kgm3=float(context.gas.density_kgm3),
        gas_mu_pas=float(context.gas.dynamic_viscosity_Pas),
        gas_temperature_K=float(context.gas.temperature),
        gas_molecular_mass_kg=float(context.gas.molecular_mass_amu) * _AMU_KG,
    )


def _sample_force_state(
    context: SolverContext,
    compiled: CompiledRuntimeBackend,
    particles: ParticleTable,
    dim: int,
) -> _SampledForceState:
    positions = np.asarray(particles.position[:, :dim], dtype=np.float64)
    velocities = np.asarray(particles.velocity[:, :dim], dtype=np.float64)
    time_s = (
        float(np.nanmin(np.asarray(particles.release_time, dtype=np.float64)))
        if particles.count
        else 0.0
    )
    status_codes = sample_compiled_valid_mask_statuses(compiled, positions)
    flow = sample_compiled_flow_vectors(compiled, dim, time_s, positions)
    gas_density, gas_viscosity, gas_temperature = (
        sample_compiled_gas_properties_vectors(
            compiled,
            dim,
            time_s,
            positions,
            fallback_density_kgm3=float(context.gas.density_kgm3),
            fallback_mu_pas=float(context.gas.dynamic_viscosity_Pas),
            fallback_temperature_K=float(context.gas.temperature),
        )
    )
    return _SampledForceState(
        positions,
        velocities,
        time_s,
        status_codes,
        flow,
        gas_density,
        gas_viscosity,
        gas_temperature,
    )


def _stokes_tau(
    particles: ParticleTable,
    static: ParticleStaticArrays,
    gas_viscosity: np.ndarray,
    drag_model_mode: int,
) -> np.ndarray:
    if drag_model_mode == int(DRAG_MODEL_NONE):
        return np.full(particles.count, np.inf, dtype=np.float64)
    if drag_model_mode == int(DRAG_MODEL_EPSTEIN):
        return np.full(particles.count, np.nan, dtype=np.float64)
    return np.asarray(
        [
            stokes_relaxation_time(
                float(static.mass_kg[i]),
                float(gas_viscosity[i]),
                float(static.diameter_m[i]),
            )
            for i in range(particles.count)
        ],
        dtype=np.float64,
    )


def _drag_contribution(
    context: SolverContext,
    particles: ParticleTable,
    static: ParticleStaticArrays,
    sampled: _SampledForceState,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    slip = np.linalg.norm(sampled.velocities - sampled.flow[:, :dim], axis=1)
    drag_model_mode = int(context.plan.drag_model_mode)
    tau_stokes = _stokes_tau(
        particles,
        static,
        sampled.gas_viscosity,
        drag_model_mode,
    )
    tau_eff = np.asarray(
        [
            effective_tau_from_drag_model(
                float(tau_stokes[i]),
                float(slip[i]),
                float(static.diameter_m[i]),
                float(sampled.gas_density[i]),
                float(sampled.gas_viscosity[i]),
                drag_model_mode,
                float(static.mass_kg[i]),
                float(sampled.gas_temperature[i]),
                float(context.gas.molecular_mass_amu) * _AMU_KG,
                _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
            )
            for i in range(particles.count)
        ],
        dtype=np.float64,
    )
    drag = (sampled.flow[:, :dim] - sampled.velocities) / np.maximum(
        tau_eff,
        1.0e-300,
    )[:, None]
    return tau_eff, drag


def _named_force_component(
    *,
    enabled: bool,
    name: str,
    compiled: CompiledRuntimeBackend,
    context: SolverContext,
    sampled: _SampledForceState,
    force_runtime: ForceRuntimeParameters,
    zeros: np.ndarray,
    component_sampler: _ComponentSampler = _sample_component,
) -> np.ndarray:
    if not enabled:
        return zeros.copy()
    return component_sampler(
        compiled=compiled,
        context=context,
        positions=sampled.positions,
        velocities=sampled.velocities,
        time_s=sampled.time_s,
        force_runtime=_single_component_runtime(force_runtime, name),
        electric_q_over_m=None,
    )


def _external_acceleration(
    context: SolverContext,
    particles: ParticleTable,
    static: ParticleStaticArrays,
    sampled: _SampledForceState,
    force_runtime: ForceRuntimeParameters,
    dim: int,
) -> np.ndarray:
    body = np.asarray(context.plan.body_acceleration_mps2, dtype=np.float64)
    if body.size < dim:
        body = np.pad(body, (0, dim - body.size), constant_values=0.0)
    external = np.tile(body[:dim], (particles.count, 1))
    if bool(force_runtime.gravity_buoyancy_enabled):
        buoyancy = np.where(
            static.density_kgm3 > 0.0,
            1.0 - sampled.gas_density / np.maximum(static.density_kgm3, 1.0e-300),
            1.0,
        )
        external = external * buoyancy[:, None]
    return external


def _force_components(
    context: SolverContext,
    particles: ParticleTable,
    static: ParticleStaticArrays,
    sampled: _SampledForceState,
    compiled: CompiledRuntimeBackend,
    force_runtime: ForceRuntimeParameters,
    force_by_name: Mapping[str, Any],
    drag: np.ndarray,
    dim: int,
    *,
    component_sampler: _ComponentSampler = _sample_component,
) -> dict[str, np.ndarray]:
    zero_runtime = _single_component_runtime(force_runtime, "")
    qom = _q_over_m(particles.charge, particles.mass)
    zeros = np.zeros((particles.count, dim), dtype=np.float64)
    electric_enabled = bool(getattr(force_by_name.get("electric"), "enabled", False))
    electric = (
        component_sampler(
            compiled=compiled,
            context=context,
            positions=sampled.positions,
            velocities=sampled.velocities,
            time_s=sampled.time_s,
            force_runtime=zero_runtime,
            electric_q_over_m=qom,
        )
        if electric_enabled
        else zeros.copy()
    )
    components = {"drag": drag, "electric": electric}
    named_components = (
        ("thermo", "thermophoresis", force_runtime.thermophoresis_enabled),
        (
            "dielectrophoretic",
            "dielectrophoresis",
            force_runtime.dielectrophoresis_enabled,
        ),
        ("lift", "lift", force_runtime.lift_enabled),
        (
            "pressure_gradient",
            "pressure_gradient",
            force_runtime.pressure_gradient_enabled,
        ),
        ("virtual_mass", "virtual_mass", force_runtime.virtual_mass_enabled),
    )
    for prefix, name, enabled in named_components:
        components[prefix] = _named_force_component(
            enabled=bool(enabled),
            name=name,
            compiled=compiled,
            context=context,
            sampled=sampled,
            force_runtime=force_runtime,
            zeros=zeros,
            component_sampler=component_sampler,
        )
    components["brownian"] = zeros.copy()
    components["external"] = _external_acceleration(
        context,
        particles,
        static,
        sampled,
        force_runtime,
        dim,
    )
    total = np.zeros((particles.count, dim), dtype=np.float64)
    for value in components.values():
        total += np.asarray(value, dtype=np.float64)
    components["total"] = total
    return components


def _force_contribution_rows(
    particles: ParticleTable,
    axes: tuple[str, ...],
    sampled: _SampledForceState,
    tau_eff: np.ndarray,
    components: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(particles.count):
        row: dict[str, Any] = {
            "particle_id": int(particles.particle_id[i]),
            "source_part_id": int(particles.source_part_id[i]),
            "time_s": sampled.time_s,
            "drag_tau_eff_s": float(tau_eff[i]),
            "field_status": _STATUS_NAMES.get(
                int(sampled.status_codes[i]),
                str(int(sampled.status_codes[i])),
            ),
            "notes": (
                "brownian acceleration is stochastic and is reported as zero in "
                "deterministic first-step compare"
            ),
        }
        for axis_index, axis in enumerate(axes):
            row[axis] = float(sampled.positions[i, axis_index])
        for prefix in _FORCE_PREFIXES:
            values = components[prefix]
            for axis_index, axis in enumerate(axes):
                row[f"{prefix}_a{axis}"] = float(values[i, axis_index])
        rows.append(row)
    return rows


def _force_contribution_frame(
    context: SolverContext,
    *,
    axes: tuple[str, ...] | None = None,
    component_sampler: _ComponentSampler = _sample_component,
) -> pd.DataFrame:
    particles = context.particles
    if particles is None:
        raise ValueError("Simulation requires particles")
    dim = int(context.spatial_dim)
    if axes is None:
        axes = axis_names_for_coordinate_system(
            context.coordinate_system, context.spatial_dim
        )
    static = static_arrays_from_particles(particles)
    force_runtime = compile_force_runtime_parameters(context.force_catalog.model)
    force_by_name = (
        context.force_catalog.by_name() if context.force_catalog is not None else {}
    )
    compiled = compile_runtime_backend(context, dim, force_runtime=force_runtime)
    sampled = _sample_force_state(context, compiled, particles, dim)
    tau_eff, drag = _drag_contribution(context, particles, static, sampled, dim)
    components = _force_components(
        context,
        particles,
        static,
        sampled,
        compiled,
        force_runtime,
        force_by_name,
        drag,
        dim,
        component_sampler=component_sampler,
    )
    return pd.DataFrame(
        _force_contribution_rows(particles, axes, sampled, tau_eff, components)
    )
