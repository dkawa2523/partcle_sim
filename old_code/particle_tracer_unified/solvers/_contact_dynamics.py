"""Evaluate the dimension-independent force and drag at wall contacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.domain import StageFields

from .compiled_backend_types import CompiledRuntimeBackend
from .drag_models import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    DRAG_MODEL_NONE,
    effective_tau_from_drag_model,
)
from .force_field_assembly import sample_compiled_acceleration_vectors
from .force_validation import (
    require_batch_quantity,
    require_force_parameter,
    require_positive_density_ratio,
)
from .forces import ForceRuntimeParameters
from .runtime_execution import RunExecutionContext
from .runtime_plan import resolve_stage_field_requirements
from .sampling_backend import (
    DYNAMIC_VISCOSITY,
    FLOW_VELOCITY,
    GAS_DENSITY,
    TEMPERATURE,
)

StageSampler = Callable[..., StageFields]


@dataclass(frozen=True)
class ContactDynamicsBatch:
    """Force and drag evaluated at projected wall-contact points."""

    target_velocity: np.ndarray
    body_acceleration: np.ndarray
    relaxation_time_s: np.ndarray


def displaced_fluid_factors(
    force_runtime: ForceRuntimeParameters | None,
    gas_density: np.ndarray,
    particle_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gravity/buoyancy and virtual-mass factors for wall motion."""

    density = np.asarray(particle_density, dtype=np.float64)
    gravity_factor = np.ones_like(density, dtype=np.float64)
    inertia_factor = np.ones_like(density, dtype=np.float64)
    if force_runtime is None:
        return gravity_factor, inertia_factor
    enabled: set[str] = set()
    if bool(force_runtime.gravity_buoyancy_enabled):
        enabled.add("gravity_buoyancy")
    if bool(force_runtime.virtual_mass_enabled):
        enabled.add("virtual_mass")
    if not enabled:
        return gravity_factor, inertia_factor

    density_ratio = require_positive_density_ratio(
        forces=enabled,
        gas_density=np.asarray(gas_density, dtype=np.float64),
        particle_density=density,
    )
    if bool(force_runtime.gravity_buoyancy_enabled):
        gravity_factor = require_batch_quantity(
            "gravity_buoyancy_factor",
            1.0 - density_ratio,
            (density.size,),
            rule="finite",
            forces={"gravity_buoyancy"},
        )
    if bool(force_runtime.virtual_mass_enabled):
        coefficient = require_force_parameter(
            "virtual_mass",
            "coefficient",
            force_runtime.virtual_mass_coefficient,
            rule="positive",
        )
        inertia_factor = require_batch_quantity(
            "virtual_mass_inertia_factor",
            1.0 + coefficient * density_ratio,
            (density.size,),
            rule="positive",
            forces={"virtual_mass"},
        )
    return gravity_factor, inertia_factor


def advance_contact_relaxation(
    velocity_initial: np.ndarray,
    target_velocity: np.ndarray,
    body_acceleration: np.ndarray,
    relaxation_time_s: np.ndarray,
    duration_s: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance exact constant-coefficient drag and body-force wall motion."""

    velocity0, target, body, tau, duration = np.broadcast_arrays(
        np.asarray(velocity_initial, dtype=np.float64),
        np.asarray(target_velocity, dtype=np.float64),
        np.asarray(body_acceleration, dtype=np.float64),
        np.asarray(relaxation_time_s, dtype=np.float64),
        np.asarray(duration_s, dtype=np.float64),
    )
    if np.any(~np.isfinite(duration) | (duration < 0.0)):
        raise ValueError("relaxation duration must be finite and >= 0")
    invalid_tau = np.isnan(tau) | (tau <= 0.0) | np.isneginf(tau)
    if np.any(invalid_tau):
        raise ValueError(
            "relaxation time must be finite and > 0, or +inf for drag_model=none"
        )

    displacement = np.empty(velocity0.shape, dtype=np.float64)
    velocity = np.empty(velocity0.shape, dtype=np.float64)
    ballistic = np.isposinf(tau)
    if np.any(ballistic):
        dt_ballistic = duration[ballistic]
        displacement[ballistic] = (
            velocity0[ballistic] * dt_ballistic
            + 0.5 * body[ballistic] * dt_ballistic * dt_ballistic
        )
        velocity[ballistic] = velocity0[ballistic] + body[ballistic] * dt_ballistic

    finite = ~ballistic
    if np.any(finite):
        tau_finite = tau[finite]
        dt_finite = duration[finite]
        with np.errstate(over="ignore"):
            ratio = dt_finite / tau_finite
        one_minus_decay = -np.expm1(-ratio)
        decay = 1.0 - one_minus_decay
        small = np.abs(ratio) < 1.0e-4
        response_integral = np.empty(ratio.shape, dtype=np.float64)
        if np.any(small):
            ratio_small = ratio[small]
            ratio2 = ratio_small * ratio_small
            response_integral[small] = tau_finite[small] * (
                0.5 * ratio2
                - ratio2 * ratio_small / 6.0
                + ratio2 * ratio2 / 24.0
                - ratio2 * ratio2 * ratio_small / 120.0
            )
        if np.any(~small):
            response_integral[~small] = (
                dt_finite[~small] - tau_finite[~small] * one_minus_decay[~small]
            )
        velocity[finite] = (
            target[finite]
            + (velocity0[finite] - target[finite]) * decay
            + body[finite] * tau_finite * one_minus_decay
        )
        displacement[finite] = (
            velocity0[finite] * dt_finite
            + (target[finite] - velocity0[finite]) * response_integral
            + body[finite] * tau_finite * response_integral
        )
    return displacement, velocity


def _compiled_has_transient_time(compiled: CompiledRuntimeBackend) -> bool:
    times_raw = (
        compiled.get("times")
        if isinstance(compiled, Mapping)
        else getattr(compiled, "times", None)
    )
    if times_raw is None:
        return False
    times = np.asarray(times_raw, dtype=np.float64)
    return bool(times.size > 1)


def _sample_contact_fields(
    execution: RunExecutionContext,
    *,
    indices: np.ndarray,
    contact_position: np.ndarray,
    time_s: float,
    sample_stage: StageSampler,
) -> tuple[StageFields, np.ndarray, bool]:
    plan = execution.plan
    physics = execution.physics
    requirements = resolve_stage_field_requirements(
        drag_model=plan.drag_model_name,
        force_runtime=execution.options.force_runtime,
    )
    sampled = sample_stage(
        execution.state.collision_diagnostics,
        execution.compiled,
        plan.stage_fields,
        contact_position,
        float(time_s),
        spatial_dim=int(execution.spatial_dim),
        need_flow=requirements.need_flow,
        need_gas_density=requirements.need_gas_density,
        need_gas_mu=requirements.need_gas_mu,
        need_gas_temperature=requirements.need_gas_temperature,
        need_valid_mask=False,
        fallback_density_kgm3=float(physics["gas_density_kgm3"]),
        fallback_mu_pas=float(physics["gas_mu_pas"]),
        fallback_temperature_K=float(physics["gas_temperature_K"]),
    )
    sampled_flow = sampled.values.get(FLOW_VELOCITY)
    target = (
        np.asarray(sampled_flow, dtype=np.float64)
        if sampled_flow is not None
        else np.zeros((indices.size, int(execution.spatial_dim)), dtype=np.float64)
    )
    return sampled, target, requirements.needs_gas_properties


def _contact_gas_properties(
    execution: RunExecutionContext,
    *,
    indices: np.ndarray,
    sampled: StageFields,
    use_local_fields: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physics = execution.physics
    density = np.full(
        indices.size,
        float(physics["gas_density_kgm3"]),
        dtype=np.float64,
    )
    viscosity = np.full(
        indices.size,
        float(physics["gas_mu_pas"]),
        dtype=np.float64,
    )
    temperature = np.full(
        indices.size,
        float(physics["gas_temperature_K"]),
        dtype=np.float64,
    )
    if not use_local_fields:
        return density, viscosity, temperature

    sampled_density = sampled.values.get(GAS_DENSITY)
    sampled_viscosity = sampled.values.get(DYNAMIC_VISCOSITY)
    sampled_temperature = sampled.values.get(TEMPERATURE)
    if sampled_density is not None:
        density = np.asarray(sampled_density, dtype=np.float64)
    if sampled_viscosity is not None:
        viscosity = np.asarray(sampled_viscosity, dtype=np.float64)
    if sampled_temperature is not None:
        temperature = np.asarray(sampled_temperature, dtype=np.float64)
    return density, viscosity, temperature


def _require_valid_contact_tau(tau: np.ndarray, drag_model_mode: int) -> None:
    if int(drag_model_mode) != int(DRAG_MODEL_NONE) and np.any(
        ~np.isfinite(tau) | (tau <= 0.0)
    ):
        raise ValueError(
            "effective contact drag relaxation time must be finite and > 0"
        )


def _sample_contact_acceleration(
    execution: RunExecutionContext,
    *,
    indices: np.ndarray,
    contact_position: np.ndarray,
    velocity: np.ndarray,
    time_s: float,
    electric_q_over_m_particle: np.ndarray | None,
) -> np.ndarray:
    spatial_dim = int(execution.spatial_dim)
    electric_qom = (
        None
        if electric_q_over_m_particle is None
        else np.asarray(electric_q_over_m_particle, dtype=np.float64)[indices]
    )
    return sample_compiled_acceleration_vectors(
        execution.compiled,
        spatial_dim,
        float(time_s),
        contact_position,
        electric_q_over_m=electric_qom,
        force_runtime=execution.options.force_runtime,
        particle_diameter=np.asarray(execution.particle_diameter, dtype=np.float64)[
            indices
        ],
        particle_density=np.asarray(execution.particle_density, dtype=np.float64)[
            indices
        ],
        particle_mass=np.asarray(execution.particle_mass, dtype=np.float64)[indices],
        dep_particle_rel_permittivity=np.asarray(
            execution.dep_particle_rel_permittivity,
            dtype=np.float64,
        )[indices],
        thermophoretic_coeff=np.asarray(
            execution.thermophoretic_coeff,
            dtype=np.float64,
        )[indices],
        velocity=np.asarray(velocity, dtype=np.float64)[:, :spatial_dim],
        gas_density_kgm3=float(execution.physics["gas_density_kgm3"]),
        gas_mu_pas=float(execution.physics["gas_mu_pas"]),
        gas_temperature_K=float(execution.physics["gas_temperature_K"]),
        gas_molecular_mass_kg=float(execution.physics["gas_molecular_mass_kg"]),
    )


def _contact_body_acceleration(
    execution: RunExecutionContext,
    *,
    indices: np.ndarray,
    acceleration: np.ndarray,
    body_acceleration: np.ndarray,
    gas_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_dim = int(execution.spatial_dim)
    particle_density = np.asarray(execution.particle_density, dtype=np.float64)[indices]
    gravity_factor, mass_factor = displaced_fluid_factors(
        execution.options.force_runtime,
        gas_density,
        particle_density,
    )
    body = np.broadcast_to(
        np.asarray(body_acceleration, dtype=np.float64)[:spatial_dim],
        (indices.size, spatial_dim),
    )
    effective = (
        gravity_factor[:, None] * body + acceleration[:, :spatial_dim]
    ) / mass_factor[:, None]
    return effective, mass_factor


def _contact_relaxation_times(
    execution: RunExecutionContext,
    *,
    indices: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    gas_density: np.ndarray,
    gas_viscosity: np.ndarray,
    gas_temperature: np.ndarray,
    mass_factor: np.ndarray,
) -> np.ndarray:
    drag_model_mode = int(execution.plan.drag_model_mode)
    tau_stokes = np.asarray(execution.tau_p[indices], dtype=np.float64)
    _require_valid_contact_tau(tau_stokes, drag_model_mode)
    if drag_model_mode == int(DRAG_MODEL_NONE):
        effective = np.full(indices.size, np.inf, dtype=np.float64)
    else:
        spatial_dim = int(execution.spatial_dim)
        slip = np.linalg.norm(
            velocity[:, :spatial_dim] - target[:, :spatial_dim],
            axis=1,
        )
        physics = execution.physics
        effective = np.asarray(
            [
                effective_tau_from_drag_model(
                    float(tau_i),
                    float(slip_i),
                    float(diameter_i),
                    float(density_i),
                    float(viscosity_i),
                    drag_model_mode,
                    float(mass_i),
                    float(temperature_i),
                    float(physics["gas_molecular_mass_kg"]),
                    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
                )
                for (
                    tau_i,
                    slip_i,
                    diameter_i,
                    mass_i,
                    density_i,
                    viscosity_i,
                    temperature_i,
                ) in zip(
                    tau_stokes,
                    slip,
                    execution.particle_diameter[indices],
                    execution.particle_mass[indices],
                    gas_density,
                    gas_viscosity,
                    gas_temperature,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
    effective = effective * mass_factor
    _require_valid_contact_tau(effective, drag_model_mode)
    return effective


def _evaluate_contact_dynamics(
    execution: RunExecutionContext,
    *,
    indices: np.ndarray,
    contact_position: np.ndarray,
    velocity: np.ndarray,
    body_acceleration: np.ndarray,
    time_s: float,
    electric_q_over_m_particle: np.ndarray | None,
    sample_stage: StageSampler,
) -> ContactDynamicsBatch:
    """Evaluate the common force and drag pipeline at wall-contact points."""

    spatial_dim = int(execution.spatial_dim)
    sampled_contact, target, needs_local_gas = _sample_contact_fields(
        execution,
        indices=indices,
        contact_position=contact_position,
        time_s=time_s,
        sample_stage=sample_stage,
    )
    acceleration = _sample_contact_acceleration(
        execution,
        indices=indices,
        contact_position=contact_position,
        velocity=velocity,
        time_s=time_s,
        electric_q_over_m_particle=electric_q_over_m_particle,
    )
    rho_g, mu_g, temperature_g = _contact_gas_properties(
        execution,
        indices=indices,
        sampled=sampled_contact,
        use_local_fields=needs_local_gas,
    )
    body_effective, mass_factor = _contact_body_acceleration(
        execution,
        indices=indices,
        acceleration=acceleration,
        body_acceleration=body_acceleration,
        gas_density=rho_g,
    )
    tau_effective = _contact_relaxation_times(
        execution,
        indices=indices,
        velocity=velocity,
        target=target,
        gas_density=rho_g,
        gas_viscosity=mu_g,
        gas_temperature=temperature_g,
        mass_factor=mass_factor,
    )
    return ContactDynamicsBatch(
        target_velocity=target[:, :spatial_dim],
        body_acceleration=body_effective,
        relaxation_time_s=tau_effective,
    )
