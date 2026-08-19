"""ETD2 stage sampling and deterministic state advancement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.coordinate_systems import axisymmetric_rz_chart_state

from .base_field_sampling import (
    sample_compiled_flow_vector as _sample_flow_vector_at,
)
from .base_field_sampling import (
    sample_compiled_gas_properties as _sample_gas_properties_at,
)
from .compiled_backend_types import CompiledRuntimeBackend
from .drag_models import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    drag_model_name_from_mode,
    effective_tau_from_drag_model,
)
from .field_runtime import sample_scalar_fields_for_stage
from .force_field_assembly import (
    sample_compiled_acceleration_vector as _sample_acceleration_vector_at,
)
from .force_validation import (
    require_batch_quantity,
    require_force_parameter,
    require_positive_density_ratio,
)
from .forces import ForceRuntimeParameters
from .integrator_common import (
    advance_affine_stage_component,
    advance_state_2d,
    advance_state_3d,
    compose_stage_acceleration_2d,
    compose_stage_acceleration_3d,
    etd2_stage_schedule,
)
from .runtime_plan import resolve_stage_field_requirements
from .sampling_backend import DYNAMIC_VISCOSITY, FLOW_VELOCITY, GAS_DENSITY, TEMPERATURE


def _virtual_mass_factor(
    force_runtime: ForceRuntimeParameters | None,
    rho_g: float,
    rho_p: float,
) -> float:
    if force_runtime is None or not bool(force_runtime.virtual_mass_enabled):
        return 1.0
    density_ratio = require_positive_density_ratio(
        forces={"virtual_mass"},
        gas_density=np.asarray([rho_g], dtype=np.float64),
        particle_density=np.asarray([rho_p], dtype=np.float64),
    )
    coefficient = require_force_parameter(
        "virtual_mass",
        "coefficient",
        force_runtime.virtual_mass_coefficient,
        rule="positive",
    )
    factor = require_batch_quantity(
        "virtual_mass_inertia_factor",
        1.0 + coefficient * density_ratio,
        (1,),
        rule="positive",
        forces={"virtual_mass"},
    )
    return float(factor[0])


def _stage_sampling_needs(
    drag_model_mode: int,
    force_runtime: ForceRuntimeParameters | None,
) -> tuple[bool, bool, bool, bool]:
    requirements = resolve_stage_field_requirements(
        drag_model=drag_model_name_from_mode(int(drag_model_mode)),
        force_runtime=force_runtime,
    )
    return (
        requirements.need_flow,
        requirements.need_gas_density,
        requirements.need_gas_mu,
        requirements.need_gas_temperature,
    )


@dataclass(frozen=True, slots=True)
class _StageDynamicsContext:
    spatial_dim: int
    compiled: CompiledRuntimeBackend
    body_acceleration: np.ndarray
    tau_stokes_s: float
    particle_diameter_m: float
    particle_density_kgm3: float
    particle_mass_kg: float
    dep_particle_rel_permittivity: float
    thermophoretic_coefficient: float
    fallback_density_kgm3: float
    fallback_viscosity_Pas: float
    fallback_temperature_K: float
    gas_molecular_mass_kg: float
    drag_model_mode: int
    electric_q_over_m_Ckg: float | None
    force_runtime: ForceRuntimeParameters | None
    sampling_needs: tuple[bool, bool, bool, bool]


@dataclass(frozen=True, slots=True)
class _StageDynamics:
    target_velocity_mps: np.ndarray
    body_acceleration_mps2: np.ndarray
    relaxation_time_s: float


def _sample_stage_dynamics(
    context: _StageDynamicsContext,
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    time_s: float,
) -> _StageDynamics:
    need_flow, need_density, need_viscosity, need_temperature = context.sampling_needs
    if str(context.compiled.coordinate_system) == "axisymmetric_rz":
        sample_position, sample_velocity, chart_sign = axisymmetric_rz_chart_state(
            position_m, velocity_mps
        )
    else:
        sample_position = np.asarray(position_m, dtype=np.float64)
        sample_velocity = np.asarray(velocity_mps, dtype=np.float64)
        chart_sign = 1.0
    sampled_stage = sample_scalar_fields_for_stage(
        context.compiled,
        None,
        sample_position,
        float(time_s),
        spatial_dim=int(context.spatial_dim),
        need_flow=bool(need_flow),
        need_electric=context.electric_q_over_m_Ckg is not None,
        need_gas_density=bool(need_density),
        need_gas_mu=bool(need_viscosity),
        need_gas_temperature=bool(need_temperature),
        need_valid_mask=False,
        fallback_density_kgm3=float(context.fallback_density_kgm3),
        fallback_mu_pas=float(context.fallback_viscosity_Pas),
        fallback_temperature_K=float(context.fallback_temperature_K),
    )
    sampled_flow = sampled_stage.values.get(FLOW_VELOCITY)
    if sampled_flow is not None:
        flow_velocity = np.asarray(sampled_flow, dtype=np.float64)[0]
    elif need_flow:
        flow_velocity = _sample_flow_vector_at(
            context.compiled,
            int(context.spatial_dim),
            float(time_s),
            sample_position,
        )
    else:
        flow_velocity = np.zeros(int(context.spatial_dim), dtype=np.float64)
    sampled_acceleration = _sample_acceleration_vector_at(
        context.compiled,
        int(context.spatial_dim),
        float(time_s),
        sample_position,
        electric_q_over_m=context.electric_q_over_m_Ckg,
        force_runtime=context.force_runtime,
        particle_diameter=float(context.particle_diameter_m),
        particle_density=float(context.particle_density_kgm3),
        particle_mass=float(context.particle_mass_kg),
        dep_particle_rel_permittivity=float(context.dep_particle_rel_permittivity),
        thermophoretic_coeff=float(context.thermophoretic_coefficient),
        velocity=sample_velocity,
        flow_velocity=flow_velocity,
        gas_density_kgm3=float(context.fallback_density_kgm3),
        gas_mu_pas=float(context.fallback_viscosity_Pas),
        gas_temperature_K=float(context.fallback_temperature_K),
        gas_molecular_mass_kg=float(context.gas_molecular_mass_kg),
        stage_fields=sampled_stage,
    )
    slip_speed = float(
        np.linalg.norm(
            sample_velocity[: context.spatial_dim]
            - flow_velocity[: context.spatial_dim]
        )
    )
    sampled_density = sampled_stage.values.get(GAS_DENSITY)
    sampled_viscosity = sampled_stage.values.get(DYNAMIC_VISCOSITY)
    sampled_temperature = sampled_stage.values.get(TEMPERATURE)
    if sampled_density is not None:
        density = float(np.asarray(sampled_density, dtype=np.float64)[0])
        viscosity = (
            float(np.asarray(sampled_viscosity, dtype=np.float64)[0])
            if sampled_viscosity is not None
            else float(context.fallback_viscosity_Pas)
        )
        temperature = (
            float(np.asarray(sampled_temperature, dtype=np.float64)[0])
            if sampled_temperature is not None
            else float(context.fallback_temperature_K)
        )
    elif any(context.sampling_needs[1:]):
        density, viscosity, temperature = _sample_gas_properties_at(
            context.compiled,
            float(time_s),
            sample_position,
            fallback_density_kgm3=float(context.fallback_density_kgm3),
            fallback_mu_pas=float(context.fallback_viscosity_Pas),
            fallback_temperature_K=float(context.fallback_temperature_K),
        )
    else:
        density, viscosity, temperature = (
            float(context.fallback_density_kgm3),
            float(context.fallback_viscosity_Pas),
            float(context.fallback_temperature_K),
        )
    mass_factor = _virtual_mass_factor(
        context.force_runtime,
        float(density),
        float(context.particle_density_kgm3),
    )
    gravity_buoyancy_enabled = int(
        context.force_runtime is not None
        and bool(context.force_runtime.gravity_buoyancy_enabled)
    )
    body = context.body_acceleration
    if int(context.spatial_dim) == 2:
        acceleration_x, acceleration_y = compose_stage_acceleration_2d(
            float(body[0]),
            float(body[1]),
            float(sampled_acceleration[0]),
            float(sampled_acceleration[1]),
            float(density),
            float(context.particle_density_kgm3),
            int(gravity_buoyancy_enabled),
            float(mass_factor),
        )
        body_acceleration = np.asarray(
            [acceleration_x, acceleration_y],
            dtype=np.float64,
        )
    else:
        acceleration_x, acceleration_y, acceleration_z = compose_stage_acceleration_3d(
            float(body[0]),
            float(body[1]),
            float(body[2]),
            float(sampled_acceleration[0]),
            float(sampled_acceleration[1]),
            float(sampled_acceleration[2]),
            float(density),
            float(context.particle_density_kgm3),
            int(gravity_buoyancy_enabled),
            float(mass_factor),
        )
        body_acceleration = np.asarray(
            [acceleration_x, acceleration_y, acceleration_z],
            dtype=np.float64,
        )
    relaxation_time = (
        float(
            effective_tau_from_drag_model(
                float(context.tau_stokes_s),
                float(slip_speed),
                float(context.particle_diameter_m),
                float(density),
                float(viscosity),
                int(context.drag_model_mode),
                float(context.particle_mass_kg),
                float(temperature),
                float(context.gas_molecular_mass_kg),
                float(_EPSTEIN_DEFAULT_ACCOMMODATION_DELTA),
            )
        )
        * mass_factor
    )
    flow_velocity = np.asarray(flow_velocity, dtype=np.float64).copy()
    flow_velocity[0] *= chart_sign
    body_acceleration[0] *= chart_sign
    return _StageDynamics(
        target_velocity_mps=flow_velocity,
        body_acceleration_mps2=body_acceleration,
        relaxation_time_s=float(relaxation_time),
    )


def _advance_with_stage_dynamics(
    context: _StageDynamicsContext,
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    dynamics: _StageDynamics,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    target = dynamics.target_velocity_mps
    body = dynamics.body_acceleration_mps2
    if int(context.spatial_dim) == 2:
        x, y, velocity_x, velocity_y = advance_state_2d(
            float(position_m[0]),
            float(position_m[1]),
            float(velocity_mps[0]),
            float(velocity_mps[1]),
            float(target[0]),
            float(target[1]),
            float(body[0]),
            float(body[1]),
            float(dynamics.relaxation_time_s),
            float(duration_s),
        )
        return (
            np.asarray([x, y], dtype=np.float64),
            np.asarray([velocity_x, velocity_y], dtype=np.float64),
        )
    x, y, z, velocity_x, velocity_y, velocity_z = advance_state_3d(
        float(position_m[0]),
        float(position_m[1]),
        float(position_m[2]),
        float(velocity_mps[0]),
        float(velocity_mps[1]),
        float(velocity_mps[2]),
        float(target[0]),
        float(target[1]),
        float(target[2]),
        float(body[0]),
        float(body[1]),
        float(body[2]),
        float(dynamics.relaxation_time_s),
        float(duration_s),
    )
    return (
        np.asarray([x, y, z], dtype=np.float64),
        np.asarray([velocity_x, velocity_y, velocity_z], dtype=np.float64),
    )


def _advance_with_affine_stage_dynamics(
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    start: _StageDynamics,
    midpoint: _StageDynamics,
    stage_fraction: float,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.empty_like(position_m)
    velocity = np.empty_like(velocity_mps)
    for axis in range(position_m.size):
        displacement, velocity[axis] = advance_affine_stage_component(
            float(velocity_mps[axis]),
            float(start.target_velocity_mps[axis]),
            float(midpoint.target_velocity_mps[axis]),
            float(start.body_acceleration_mps2[axis]),
            float(midpoint.body_acceleration_mps2[axis]),
            float(start.relaxation_time_s),
            float(midpoint.relaxation_time_s),
            float(stage_fraction),
            float(duration_s),
        )
        position[axis] = float(position_m[axis]) + displacement
    return position, velocity


def _advance_etd2_from_start(
    context: _StageDynamicsContext,
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_sub: float,
    t_sub_start: float,
    start_dynamics: _StageDynamics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    t_mid, predictor_dt, corrector_dt = etd2_stage_schedule(t_sub_start, dt_sub)
    coefficient_midpoint_position, coefficient_midpoint_velocity = (
        _advance_with_stage_dynamics(
            context,
            position_m=x0,
            velocity_mps=v0,
            dynamics=start_dynamics,
            duration_s=float(predictor_dt),
        )
    )
    midpoint_dynamics = _sample_stage_dynamics(
        context,
        position_m=coefficient_midpoint_position,
        velocity_mps=coefficient_midpoint_velocity,
        time_s=float(t_mid),
    )
    half_position, half_velocity = _advance_with_affine_stage_dynamics(
        position_m=x0,
        velocity_mps=v0,
        start=start_dynamics,
        midpoint=midpoint_dynamics,
        stage_fraction=0.5,
        duration_s=float(predictor_dt),
    )
    end_position, end_velocity = _advance_with_affine_stage_dynamics(
        position_m=x0,
        velocity_mps=v0,
        start=start_dynamics,
        midpoint=midpoint_dynamics,
        stage_fraction=1.0,
        duration_s=float(corrector_dt),
    )
    return (
        end_position,
        end_velocity,
        half_position,
        half_velocity,
        float(start_dynamics.relaxation_time_s),
        float(midpoint_dynamics.relaxation_time_s),
        coefficient_midpoint_position,
    )


def _advance_etd2_substep(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_sub: float,
    t_sub_start: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackend,
    body: np.ndarray,
    tau_stokes: float,
    particle_diameter_m: float,
    particle_density_kgm3: float,
    particle_mass_kg: float,
    dep_particle_rel_permittivity: float,
    thermophoretic_coeff: float,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    electric_q_over_m_i: float | None = None,
    force_runtime: ForceRuntimeParameters | None = None,
    estimate_local_error: bool = False,
) -> tuple:
    context = _StageDynamicsContext(
        spatial_dim=int(spatial_dim),
        compiled=compiled,
        body_acceleration=body,
        tau_stokes_s=float(tau_stokes),
        particle_diameter_m=float(particle_diameter_m),
        particle_density_kgm3=float(particle_density_kgm3),
        particle_mass_kg=float(particle_mass_kg),
        dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
        thermophoretic_coefficient=float(thermophoretic_coeff),
        fallback_density_kgm3=float(gas_density_kgm3),
        fallback_viscosity_Pas=float(gas_mu_pas),
        fallback_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        drag_model_mode=int(drag_model_mode),
        electric_q_over_m_Ckg=electric_q_over_m_i,
        force_runtime=force_runtime,
        sampling_needs=_stage_sampling_needs(int(drag_model_mode), force_runtime),
    )
    start_dynamics = _sample_stage_dynamics(
        context,
        position_m=x0,
        velocity_mps=v0,
        time_s=float(t_sub_start),
    )
    full_step = _advance_etd2_from_start(
        context,
        x0=x0,
        v0=v0,
        dt_sub=dt_sub,
        t_sub_start=t_sub_start,
        start_dynamics=start_dynamics,
    )
    if not estimate_local_error:
        return full_step
    half_duration = 0.5 * float(dt_sub)
    refined_mid = _advance_etd2_from_start(
        context,
        x0=x0,
        v0=v0,
        dt_sub=half_duration,
        t_sub_start=t_sub_start,
        start_dynamics=start_dynamics,
    )
    refined_start = _sample_stage_dynamics(
        context,
        position_m=refined_mid[0],
        velocity_mps=refined_mid[1],
        time_s=float(t_sub_start) + half_duration,
    )
    refined_end = _advance_etd2_from_start(
        context,
        x0=refined_mid[0],
        v0=refined_mid[1],
        dt_sub=half_duration,
        t_sub_start=float(t_sub_start) + half_duration,
        start_dynamics=refined_start,
    )
    return (
        *full_step,
        refined_mid[0],
        refined_mid[1],
        refined_end[0],
        refined_end[1],
    )
