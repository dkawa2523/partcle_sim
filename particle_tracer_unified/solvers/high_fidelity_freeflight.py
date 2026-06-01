from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from ..core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    valid_mask_status_requires_stop,
)
from .compiled_field_backend import (
    CompiledRuntimeBackend,
    CompiledRuntimeBackendLike,
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
    coerce_compiled_backend as _coerce_compiled_backend,
    compile_runtime_backend as _compile_runtime_arrays,
    sample_compiled_acceleration_vector as _sample_acceleration_vector_at,
    sample_compiled_flow_vector as _sample_flow_vector_at,
    sample_compiled_gas_properties as _sample_gas_properties_at,
    sample_compiled_valid_mask_status as _sample_valid_mask_status,
)
from .field_runtime import sample_scalar_fields_for_stage
from .forces import ForceRuntimeParameters
from .integrator_common import (
    DRAG_MODEL_STOKES,
    INTEGRATOR_ETD2,
    advance_state_2d,
    advance_state_2d_etd,
    advance_state_3d,
    advance_state_3d_etd,
    compute_substep_count,
    effective_tau_from_slip_speed,
)
from .kernel2d_numba import advance_particles_2d_inplace
from .kernel2d_triangle_mesh_numba import advance_particles_2d_triangle_mesh_inplace
from .kernel3d_numba import advance_particles_3d_inplace

@dataclass(slots=True)
class ValidMaskPrefixResolution:
    position: np.ndarray
    velocity: np.ndarray
    accepted_dt: float
    retry_count: int
    found_valid_prefix: bool


def _stage_points_from_trial(
    x_end: np.ndarray,
    *,
    integrator_mode: int,
    x_mid: Optional[np.ndarray] = None,
) -> np.ndarray:
    x_end_arr = np.asarray(x_end, dtype=np.float64)
    if int(integrator_mode) == INTEGRATOR_ETD2 and x_mid is not None:
        x_mid_arr = np.asarray(x_mid, dtype=np.float64)
        return np.stack((x_mid_arr, x_end_arr), axis=0)
    return x_end_arr.reshape(1, x_end_arr.size).copy()


def _virtual_mass_factor(
    force_runtime: ForceRuntimeParameters | None,
    rho_g: float,
    rho_p: float,
) -> float:
    if force_runtime is None or not bool(force_runtime.virtual_mass_enabled):
        return 1.0
    if float(rho_p) <= 0.0 or float(rho_g) <= 0.0:
        return 1.0
    coeff = max(float(force_runtime.virtual_mass_coefficient), 0.0)
    factor = 1.0 + coeff * float(rho_g) / float(rho_p)
    return float(factor) if np.isfinite(factor) and factor > 0.0 else 1.0


def _initial_substep_tau(
    *,
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_start: float,
    x0: np.ndarray,
    v0: np.ndarray,
    tau_stokes: float,
    particle_diameter_m: float,
    particle_density_kgm3: float,
    flow_scale_particle_i: float,
    global_flow_scale: float,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    min_tau_p_s: float,
) -> float:
    if int(drag_model_mode) == int(DRAG_MODEL_STOKES):
        return float(tau_stokes)
    sampled_stage = sample_scalar_fields_for_stage(
        compiled,
        None,
        np.asarray(x0, dtype=np.float64),
        float(t_start),
        spatial_dim=int(spatial_dim),
        need_flow=True,
        need_gas_properties=True,
        need_valid_mask=False,
        fallback_density_kgm3=float(gas_density_kgm3),
        fallback_mu_pas=float(gas_mu_pas),
        fallback_temperature_K=float(gas_temperature_K),
    )
    flow = (
        sampled_stage.flow[0]
        if sampled_stage.flow is not None
        else _sample_flow_vector_at(compiled, int(spatial_dim), float(t_start), np.asarray(x0, dtype=np.float64))
    )
    target = float(global_flow_scale) * float(flow_scale_particle_i) * flow[: int(spatial_dim)]
    slip = float(np.linalg.norm(np.asarray(v0, dtype=np.float64)[: int(spatial_dim)] - target))
    if sampled_stage.gas_density is not None:
        rho_local = float(sampled_stage.gas_density[0])
        mu_local = float(sampled_stage.gas_mu[0]) if sampled_stage.gas_mu is not None else float(gas_mu_pas)
        temp_local = (
            float(sampled_stage.gas_temperature[0])
            if sampled_stage.gas_temperature is not None
            else float(gas_temperature_K)
        )
    else:
        rho_local, mu_local, temp_local = _sample_gas_properties_at(
            compiled,
            float(t_start),
            np.asarray(x0, dtype=np.float64),
            fallback_density_kgm3=float(gas_density_kgm3),
            fallback_mu_pas=float(gas_mu_pas),
            fallback_temperature_K=float(gas_temperature_K),
        )
    tau_eff = float(
        effective_tau_from_slip_speed(
            float(tau_stokes),
            float(slip),
            float(particle_diameter_m),
            float(rho_local),
            float(mu_local),
            int(drag_model_mode),
            float(min_tau_p_s),
            float(particle_density_kgm3),
            float(temp_local),
            float(gas_molecular_mass_kg),
        )
    )
    if np.isfinite(tau_eff) and tau_eff > 0.0:
        return float(min(float(tau_stokes), tau_eff))
    return float(tau_stokes)


def _advance_etd2_substep(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_sub: float,
    t_sub_start: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    flow_scale_particle_i: float,
    global_flow_scale: float,
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
    min_tau_p_s: float,
    electric_q_over_m_i: Optional[float] = None,
    force_runtime: ForceRuntimeParameters | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_start = sample_scalar_fields_for_stage(
        compiled,
        None,
        x0,
        float(t_sub_start),
        spatial_dim=int(spatial_dim),
        need_flow=True,
        need_gas_properties=True,
        need_valid_mask=False,
        fallback_density_kgm3=float(gas_density_kgm3),
        fallback_mu_pas=float(gas_mu_pas),
        fallback_temperature_K=float(gas_temperature_K),
    )
    flow_start = (
        sampled_start.flow[0]
        if sampled_start.flow is not None
        else _sample_flow_vector_at(compiled, spatial_dim, float(t_sub_start), x0)
    )
    accel_start = _sample_acceleration_vector_at(
        compiled,
        spatial_dim,
        float(t_sub_start),
        x0,
        electric_q_over_m=electric_q_over_m_i,
        force_runtime=force_runtime,
        particle_diameter=float(particle_diameter_m),
        particle_density=float(particle_density_kgm3),
        particle_mass=float(particle_mass_kg),
        dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
        thermophoretic_coeff=float(thermophoretic_coeff),
        velocity=np.asarray(v0, dtype=np.float64),
        flow_velocity=flow_start,
        gas_density_kgm3=float(gas_density_kgm3),
        gas_mu_pas=float(gas_mu_pas),
        gas_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
    )
    target_start = float(global_flow_scale) * float(flow_scale_particle_i) * flow_start
    slip_start = float(np.linalg.norm(np.asarray(v0, dtype=np.float64)[:spatial_dim] - target_start[:spatial_dim]))
    if sampled_start.gas_density is not None:
        rho_start = float(sampled_start.gas_density[0])
        mu_start = float(sampled_start.gas_mu[0]) if sampled_start.gas_mu is not None else float(gas_mu_pas)
        temp_start = (
            float(sampled_start.gas_temperature[0])
            if sampled_start.gas_temperature is not None
            else float(gas_temperature_K)
        )
    else:
        rho_start, mu_start, temp_start = _sample_gas_properties_at(
            compiled,
            float(t_sub_start),
            x0,
            fallback_density_kgm3=float(gas_density_kgm3),
            fallback_mu_pas=float(gas_mu_pas),
            fallback_temperature_K=float(gas_temperature_K),
        )
    gravity_factor_start = 1.0
    if force_runtime is not None and bool(force_runtime.gravity_buoyancy_enabled) and float(particle_density_kgm3) > 0.0:
        gravity_factor_start = 1.0 - float(rho_start) / float(particle_density_kgm3)
    mass_factor_start = _virtual_mass_factor(force_runtime, float(rho_start), float(particle_density_kgm3))
    body_start = (gravity_factor_start * body + accel_start[:spatial_dim]) / mass_factor_start
    tau_start = float(
        effective_tau_from_slip_speed(
            float(tau_stokes),
            float(slip_start),
            float(particle_diameter_m),
            float(rho_start),
            float(mu_start),
            int(drag_model_mode),
            float(min_tau_p_s),
            float(particle_density_kgm3),
            float(temp_start),
            float(gas_molecular_mass_kg),
        )
    ) * mass_factor_start
    if spatial_dim == 2:
        xh, yh, vxh, vyh = advance_state_2d_etd(
            float(x0[0]),
            float(x0[1]),
            float(v0[0]),
            float(v0[1]),
            float(target_start[0]),
            float(target_start[1]),
            float(body_start[0]),
            float(body_start[1]),
            float(tau_start),
            0.5 * float(dt_sub),
        )
        t_mid = float(t_sub_start) + 0.5 * float(dt_sub)
        x_half = np.asarray([xh, yh], dtype=np.float64)
        sampled_mid = sample_scalar_fields_for_stage(
            compiled,
            None,
            x_half,
            float(t_mid),
            spatial_dim=int(spatial_dim),
            need_flow=True,
            need_gas_properties=True,
            need_valid_mask=False,
            fallback_density_kgm3=float(gas_density_kgm3),
            fallback_mu_pas=float(gas_mu_pas),
            fallback_temperature_K=float(gas_temperature_K),
        )
        flow_mid = (
            sampled_mid.flow[0]
            if sampled_mid.flow is not None
            else _sample_flow_vector_at(compiled, spatial_dim, t_mid, x_half)
        )
        accel_mid = _sample_acceleration_vector_at(
            compiled,
            spatial_dim,
            t_mid,
            x_half,
            electric_q_over_m=electric_q_over_m_i,
            force_runtime=force_runtime,
            particle_diameter=float(particle_diameter_m),
            particle_density=float(particle_density_kgm3),
            particle_mass=float(particle_mass_kg),
            dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
            thermophoretic_coeff=float(thermophoretic_coeff),
            velocity=np.asarray([vxh, vyh], dtype=np.float64),
            flow_velocity=flow_mid,
            gas_density_kgm3=float(gas_density_kgm3),
            gas_mu_pas=float(gas_mu_pas),
            gas_temperature_K=float(gas_temperature_K),
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        )
        target_mid = float(global_flow_scale) * float(flow_scale_particle_i) * flow_mid
        slip_mid = float(np.linalg.norm(np.asarray([vxh, vyh], dtype=np.float64) - target_mid[:2]))
        if sampled_mid.gas_density is not None:
            rho_mid = float(sampled_mid.gas_density[0])
            mu_mid = float(sampled_mid.gas_mu[0]) if sampled_mid.gas_mu is not None else float(gas_mu_pas)
            temp_mid = (
                float(sampled_mid.gas_temperature[0])
                if sampled_mid.gas_temperature is not None
                else float(gas_temperature_K)
            )
        else:
            rho_mid, mu_mid, temp_mid = _sample_gas_properties_at(
                compiled,
                float(t_mid),
                x_half,
                fallback_density_kgm3=float(gas_density_kgm3),
                fallback_mu_pas=float(gas_mu_pas),
                fallback_temperature_K=float(gas_temperature_K),
            )
        gravity_factor_mid = 1.0
        if force_runtime is not None and bool(force_runtime.gravity_buoyancy_enabled) and float(particle_density_kgm3) > 0.0:
            gravity_factor_mid = 1.0 - float(rho_mid) / float(particle_density_kgm3)
        mass_factor_mid = _virtual_mass_factor(force_runtime, float(rho_mid), float(particle_density_kgm3))
        body_mid = (gravity_factor_mid * body + accel_mid[:spatial_dim]) / mass_factor_mid
        tau_mid = float(
            effective_tau_from_slip_speed(
                float(tau_stokes),
                float(slip_mid),
                float(particle_diameter_m),
                float(rho_mid),
                float(mu_mid),
                int(drag_model_mode),
                float(min_tau_p_s),
                float(particle_density_kgm3),
                float(temp_mid),
                float(gas_molecular_mass_kg),
            )
        ) * mass_factor_mid
        xn, yn, vxn, vyn = advance_state_2d_etd(
            float(x0[0]),
            float(x0[1]),
            float(v0[0]),
            float(v0[1]),
            float(target_mid[0]),
            float(target_mid[1]),
            float(body_mid[0]),
            float(body_mid[1]),
            float(tau_mid),
            float(dt_sub),
        )
        return (
            np.asarray([xn, yn], dtype=np.float64),
            np.asarray([vxn, vyn], dtype=np.float64),
            np.asarray([xh, yh], dtype=np.float64),
        )
    xh, yh, zh, vxh, vyh, vzh = advance_state_3d_etd(
        float(x0[0]),
        float(x0[1]),
        float(x0[2]),
        float(v0[0]),
        float(v0[1]),
        float(v0[2]),
        float(target_start[0]),
        float(target_start[1]),
        float(target_start[2]),
        float(body_start[0]),
        float(body_start[1]),
        float(body_start[2]),
        float(tau_start),
        0.5 * float(dt_sub),
    )
    t_mid = float(t_sub_start) + 0.5 * float(dt_sub)
    x_half = np.asarray([xh, yh, zh], dtype=np.float64)
    sampled_mid = sample_scalar_fields_for_stage(
        compiled,
        None,
        x_half,
        float(t_mid),
        spatial_dim=int(spatial_dim),
        need_flow=True,
        need_gas_properties=True,
        need_valid_mask=False,
        fallback_density_kgm3=float(gas_density_kgm3),
        fallback_mu_pas=float(gas_mu_pas),
        fallback_temperature_K=float(gas_temperature_K),
    )
    flow_mid = (
        sampled_mid.flow[0]
        if sampled_mid.flow is not None
        else _sample_flow_vector_at(compiled, spatial_dim, t_mid, x_half)
    )
    accel_mid = _sample_acceleration_vector_at(
        compiled,
        spatial_dim,
        t_mid,
        x_half,
        electric_q_over_m=electric_q_over_m_i,
        force_runtime=force_runtime,
        particle_diameter=float(particle_diameter_m),
        particle_density=float(particle_density_kgm3),
        particle_mass=float(particle_mass_kg),
        dep_particle_rel_permittivity=float(dep_particle_rel_permittivity),
        thermophoretic_coeff=float(thermophoretic_coeff),
        velocity=np.asarray([vxh, vyh, vzh], dtype=np.float64),
        flow_velocity=flow_mid,
        gas_density_kgm3=float(gas_density_kgm3),
        gas_mu_pas=float(gas_mu_pas),
        gas_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
    )
    target_mid = float(global_flow_scale) * float(flow_scale_particle_i) * flow_mid
    slip_mid = float(np.linalg.norm(np.asarray([vxh, vyh, vzh], dtype=np.float64) - target_mid[:3]))
    if sampled_mid.gas_density is not None:
        rho_mid = float(sampled_mid.gas_density[0])
        mu_mid = float(sampled_mid.gas_mu[0]) if sampled_mid.gas_mu is not None else float(gas_mu_pas)
        temp_mid = (
            float(sampled_mid.gas_temperature[0])
            if sampled_mid.gas_temperature is not None
            else float(gas_temperature_K)
        )
    else:
        rho_mid, mu_mid, temp_mid = _sample_gas_properties_at(
            compiled,
            float(t_mid),
            x_half,
            fallback_density_kgm3=float(gas_density_kgm3),
            fallback_mu_pas=float(gas_mu_pas),
            fallback_temperature_K=float(gas_temperature_K),
        )
    gravity_factor_mid = 1.0
    if force_runtime is not None and bool(force_runtime.gravity_buoyancy_enabled) and float(particle_density_kgm3) > 0.0:
        gravity_factor_mid = 1.0 - float(rho_mid) / float(particle_density_kgm3)
    mass_factor_mid = _virtual_mass_factor(force_runtime, float(rho_mid), float(particle_density_kgm3))
    body_mid = (gravity_factor_mid * body + accel_mid[:spatial_dim]) / mass_factor_mid
    tau_mid = float(
        effective_tau_from_slip_speed(
            float(tau_stokes),
            float(slip_mid),
            float(particle_diameter_m),
            float(rho_mid),
            float(mu_mid),
            int(drag_model_mode),
            float(min_tau_p_s),
            float(particle_density_kgm3),
            float(temp_mid),
            float(gas_molecular_mass_kg),
        )
    ) * mass_factor_mid
    xn, yn, zn, vxn, vyn, vzn = advance_state_3d_etd(
        float(x0[0]),
        float(x0[1]),
        float(x0[2]),
        float(v0[0]),
        float(v0[1]),
        float(v0[2]),
        float(target_mid[0]),
        float(target_mid[1]),
        float(target_mid[2]),
        float(body_mid[0]),
        float(body_mid[1]),
        float(body_mid[2]),
        float(tau_mid),
        float(dt_sub),
    )
    return (
        np.asarray([xn, yn, zn], dtype=np.float64),
        np.asarray([vxn, vyn, vzn], dtype=np.float64),
        np.asarray([xh, yh, zh], dtype=np.float64),
    )


def _stage_sample_times(segment_dt: float, stage_points: np.ndarray) -> np.ndarray:
    points = np.asarray(stage_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    count = int(points.shape[0])
    dt_seg = max(float(segment_dt), 0.0)
    if count == 1 or dt_seg <= 0.0:
        return np.asarray([dt_seg], dtype=np.float64)
    if count == 2:
        return np.asarray([0.5 * dt_seg, dt_seg], dtype=np.float64)
    return np.linspace(dt_seg / float(count), dt_seg, count, dtype=np.float64)


def advance_freeflight_segment(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_segment: float,
    t_end_segment: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    tau_p_i: float,
    particle_diameter_i: float,
    particle_density_i: float,
    particle_mass_i: float,
    dep_particle_rel_permittivity_i: float,
    thermophoretic_coeff_i: float,
    flow_scale_particle_i: float,
    drag_scale_particle_i: float,
    body_scale_particle_i: float,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    global_body_accel_scale: float,
    body_accel: np.ndarray,
    min_tau_p_s: float,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    electric_q_over_m_i: Optional[float] = None,
    force_runtime: ForceRuntimeParameters | None = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
    backend = _coerce_compiled_backend(compiled)
    dt_seg = float(max(dt_segment, 0.0))
    x_curr = np.asarray(x0, dtype=np.float64).copy()
    v_curr = np.asarray(v0, dtype=np.float64).copy()
    if dt_seg <= 0.0:
        return x_curr, v_curr, 1, x_curr.reshape(1, x_curr.size).copy(), int(VALID_MASK_STATUS_CLEAN)
    tau_stokes = float(tau_p_i) * float(global_drag_tau_scale) * max(float(drag_scale_particle_i), 1e-6)
    tau_stokes = max(float(min_tau_p_s), tau_stokes)
    t_start = float(t_end_segment) - dt_seg
    substep_tau = _initial_substep_tau(
        compiled=backend,
        spatial_dim=int(spatial_dim),
        t_start=float(t_start),
        x0=x_curr,
        v0=v_curr,
        tau_stokes=float(tau_stokes),
        particle_diameter_m=float(particle_diameter_i),
        particle_density_kgm3=float(particle_density_i),
        flow_scale_particle_i=float(flow_scale_particle_i),
        global_flow_scale=float(global_flow_scale),
        gas_density_kgm3=float(gas_density_kgm3),
        gas_mu_pas=float(gas_mu_pas),
        gas_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        drag_model_mode=int(drag_model_mode),
        min_tau_p_s=float(min_tau_p_s),
    )
    n_substeps = int(
        compute_substep_count(
            dt_seg,
            substep_tau,
            int(adaptive_substep_enabled),
            float(adaptive_substep_tau_ratio),
            int(adaptive_substep_max_splits),
        )
    )
    dt_sub = dt_seg / float(max(1, n_substeps))
    body_scale = float(global_body_accel_scale) * float(body_scale_particle_i)
    body = np.asarray(body_accel, dtype=np.float64)[:spatial_dim] * body_scale
    is_etd2 = int(integrator_mode) == INTEGRATOR_ETD2
    stage_mid = x_curr.copy()
    stage_mid_captured = False
    mid_target = 0.5 * dt_seg
    elapsed = 0.0
    valid_mask_status = int(VALID_MASK_STATUS_CLEAN)
    for sub_idx in range(n_substeps):
        t_sub_start = t_start + float(sub_idx) * dt_sub
        x_prev = x_curr.copy()
        v_prev = v_curr.copy()
        if is_etd2:
            sample_status = _sample_valid_mask_status(backend, x_prev)
            if sample_status > valid_mask_status:
                valid_mask_status = int(sample_status)
            x_curr, v_curr, x_half = _advance_etd2_substep(
                x0=x_prev,
                v0=v_prev,
                dt_sub=float(dt_sub),
                t_sub_start=float(t_sub_start),
                spatial_dim=int(spatial_dim),
                compiled=backend,
                flow_scale_particle_i=float(flow_scale_particle_i),
                global_flow_scale=float(global_flow_scale),
                body=body,
                tau_stokes=float(tau_stokes),
                particle_diameter_m=float(particle_diameter_i),
                particle_density_kgm3=float(particle_density_i),
                particle_mass_kg=float(particle_mass_i),
                dep_particle_rel_permittivity=float(dep_particle_rel_permittivity_i),
                thermophoretic_coeff=float(thermophoretic_coeff_i),
                gas_density_kgm3=float(gas_density_kgm3),
                gas_mu_pas=float(gas_mu_pas),
                gas_temperature_K=float(gas_temperature_K),
                gas_molecular_mass_kg=float(gas_molecular_mass_kg),
                drag_model_mode=int(drag_model_mode),
                min_tau_p_s=float(min_tau_p_s),
                electric_q_over_m_i=electric_q_over_m_i,
                force_runtime=force_runtime,
            )
            sample_status = _sample_valid_mask_status(backend, x_half)
            if sample_status > valid_mask_status:
                valid_mask_status = int(sample_status)
        else:
            t_eval = t_start + (float(sub_idx) + 1.0) * dt_sub
            sampled_stage = sample_scalar_fields_for_stage(
                backend,
                None,
                x_curr,
                float(t_eval),
                spatial_dim=int(spatial_dim),
                need_flow=True,
                need_gas_properties=True,
                need_valid_mask=True,
                fallback_density_kgm3=float(gas_density_kgm3),
                fallback_mu_pas=float(gas_mu_pas),
                fallback_temperature_K=float(gas_temperature_K),
            )
            sample_status = (
                int(sampled_stage.valid_mask_status[0])
                if sampled_stage.valid_mask_status is not None
                else int(_sample_valid_mask_status(backend, x_curr))
            )
            if sample_status > valid_mask_status:
                valid_mask_status = int(sample_status)
            flow = (
                sampled_stage.flow[0]
                if sampled_stage.flow is not None
                else _sample_flow_vector_at(backend, spatial_dim, t_eval, x_curr)
            )
            accel = _sample_acceleration_vector_at(
                backend,
                spatial_dim,
                t_eval,
                x_curr,
                electric_q_over_m=electric_q_over_m_i,
                force_runtime=force_runtime,
                particle_diameter=float(particle_diameter_i),
                particle_density=float(particle_density_i),
                particle_mass=float(particle_mass_i),
                dep_particle_rel_permittivity=float(dep_particle_rel_permittivity_i),
                thermophoretic_coeff=float(thermophoretic_coeff_i),
                velocity=v_curr[:spatial_dim],
                flow_velocity=flow,
                gas_density_kgm3=float(gas_density_kgm3),
                gas_mu_pas=float(gas_mu_pas),
                gas_temperature_K=float(gas_temperature_K),
                gas_molecular_mass_kg=float(gas_molecular_mass_kg),
            )
            target = float(global_flow_scale) * float(flow_scale_particle_i) * flow
            slip = float(np.linalg.norm(v_curr[:spatial_dim] - target[:spatial_dim]))
            if sampled_stage.gas_density is not None:
                rho_local = float(sampled_stage.gas_density[0])
                mu_local = float(sampled_stage.gas_mu[0]) if sampled_stage.gas_mu is not None else float(gas_mu_pas)
                temp_local = (
                    float(sampled_stage.gas_temperature[0])
                    if sampled_stage.gas_temperature is not None
                    else float(gas_temperature_K)
                )
            else:
                rho_local, mu_local, temp_local = _sample_gas_properties_at(
                    backend,
                    float(t_eval),
                    x_curr,
                    fallback_density_kgm3=float(gas_density_kgm3),
                    fallback_mu_pas=float(gas_mu_pas),
                    fallback_temperature_K=float(gas_temperature_K),
                )
            gravity_factor = 1.0
            if force_runtime is not None and bool(force_runtime.gravity_buoyancy_enabled) and float(particle_density_i) > 0.0:
                gravity_factor = 1.0 - float(rho_local) / float(particle_density_i)
            mass_factor = _virtual_mass_factor(force_runtime, float(rho_local), float(particle_density_i))
            body_eff = (gravity_factor * body + accel[:spatial_dim]) / mass_factor
            tau_eff = float(
                effective_tau_from_slip_speed(
                    float(tau_stokes),
                    float(slip),
                    float(particle_diameter_i),
                    float(rho_local),
                    float(mu_local),
                    int(drag_model_mode),
                    float(min_tau_p_s),
                    float(particle_density_i),
                    float(temp_local),
                    float(gas_molecular_mass_kg),
                )
            ) * mass_factor
            if spatial_dim == 2:
                x0n, y0n, vxn, vyn = advance_state_2d(
                    float(x_curr[0]),
                    float(x_curr[1]),
                    float(v_curr[0]),
                    float(v_curr[1]),
                    float(target[0]),
                    float(target[1]),
                    float(body_eff[0]),
                    float(body_eff[1]),
                    tau_eff,
                    dt_sub,
                    int(integrator_mode),
                )
                x_curr[0], x_curr[1] = x0n, y0n
                v_curr[0], v_curr[1] = vxn, vyn
            else:
                x0n, y0n, z0n, vxn, vyn, vzn = advance_state_3d(
                    float(x_curr[0]),
                    float(x_curr[1]),
                    float(x_curr[2]),
                    float(v_curr[0]),
                    float(v_curr[1]),
                    float(v_curr[2]),
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    float(body_eff[0]),
                    float(body_eff[1]),
                    float(body_eff[2]),
                    tau_eff,
                    dt_sub,
                    int(integrator_mode),
                )
                x_curr[0], x_curr[1], x_curr[2] = x0n, y0n, z0n
                v_curr[0], v_curr[1], v_curr[2] = vxn, vyn, vzn
        if is_etd2 and not stage_mid_captured:
            elapsed_next = elapsed + dt_sub
            if mid_target <= elapsed_next + 1e-15:
                dt_local = float(np.clip(mid_target - elapsed, 0.0, dt_sub))
                if dt_local <= 1e-15:
                    stage_mid = x_prev.copy()
                elif dt_local >= dt_sub - 1e-15:
                    stage_mid = x_curr.copy()
                else:
                    x_mid, _v_mid, _x_half_mid = _advance_etd2_substep(
                        x0=x_prev,
                        v0=v_prev,
                        dt_sub=float(dt_local),
                        t_sub_start=float(t_sub_start),
                        spatial_dim=int(spatial_dim),
                        compiled=backend,
                        flow_scale_particle_i=float(flow_scale_particle_i),
                        global_flow_scale=float(global_flow_scale),
                        body=body,
                        tau_stokes=float(tau_stokes),
                        particle_diameter_m=float(particle_diameter_i),
                        particle_density_kgm3=float(particle_density_i),
                        particle_mass_kg=float(particle_mass_i),
                        dep_particle_rel_permittivity=float(dep_particle_rel_permittivity_i),
                        thermophoretic_coeff=float(thermophoretic_coeff_i),
                        gas_density_kgm3=float(gas_density_kgm3),
                        gas_mu_pas=float(gas_mu_pas),
                        gas_temperature_K=float(gas_temperature_K),
                        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
                        drag_model_mode=int(drag_model_mode),
                        min_tau_p_s=float(min_tau_p_s),
                        electric_q_over_m_i=electric_q_over_m_i,
                        force_runtime=force_runtime,
                    )
                    sample_status = _sample_valid_mask_status(backend, x_mid)
                    if sample_status > valid_mask_status:
                        valid_mask_status = int(sample_status)
                    stage_mid = x_mid
                stage_mid_captured = True
            elapsed = elapsed_next
    if is_etd2:
        if not stage_mid_captured:
            stage_mid = x_curr.copy()
        stage_points = np.stack((stage_mid, x_curr.copy()), axis=0)
    else:
        stage_points = x_curr.reshape(1, x_curr.size).copy()
    return x_curr, v_curr, int(max(1, n_substeps)), stage_points, int(valid_mask_status)


def advance_freeflight_partial(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_partial: float,
    segment_dt: float,
    t_end_segment: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    tau_p_i: float,
    particle_diameter_i: float,
    particle_density_i: float,
    particle_mass_i: float,
    dep_particle_rel_permittivity_i: float,
    thermophoretic_coeff_i: float,
    flow_scale_particle_i: float,
    drag_scale_particle_i: float,
    body_scale_particle_i: float,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    global_body_accel_scale: float,
    body_accel: np.ndarray,
    min_tau_p_s: float,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    electric_q_over_m_i: Optional[float] = None,
    force_runtime: ForceRuntimeParameters | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    dt_seg = max(float(segment_dt), 0.0)
    dt_eval = float(np.clip(dt_partial, 0.0, dt_seg))
    if dt_eval <= 0.0:
        return np.asarray(x0, dtype=np.float64).copy(), np.asarray(v0, dtype=np.float64).copy()
    partial_t_end = float(t_end_segment) - dt_seg + dt_eval
    x_out, v_out, _n_substeps, _stage_points, _valid_mask_status = advance_freeflight_segment(
        x0=x0,
        v0=v0,
        dt_segment=float(dt_eval),
        t_end_segment=float(partial_t_end),
        spatial_dim=int(spatial_dim),
        compiled=compiled,
        integrator_mode=int(integrator_mode),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        tau_p_i=float(tau_p_i),
        particle_diameter_i=float(particle_diameter_i),
        particle_density_i=float(particle_density_i),
        particle_mass_i=float(particle_mass_i),
        dep_particle_rel_permittivity_i=float(dep_particle_rel_permittivity_i),
        thermophoretic_coeff_i=float(thermophoretic_coeff_i),
        flow_scale_particle_i=float(flow_scale_particle_i),
        drag_scale_particle_i=float(drag_scale_particle_i),
        body_scale_particle_i=float(body_scale_particle_i),
        global_flow_scale=float(global_flow_scale),
        global_drag_tau_scale=float(global_drag_tau_scale),
        global_body_accel_scale=float(global_body_accel_scale),
        body_accel=body_accel,
        min_tau_p_s=float(min_tau_p_s),
        gas_density_kgm3=float(gas_density_kgm3),
        gas_mu_pas=float(gas_mu_pas),
        gas_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        drag_model_mode=int(drag_model_mode),
        electric_q_over_m_i=electric_q_over_m_i,
        force_runtime=force_runtime,
    )
    return x_out, v_out


def resolve_valid_mask_prefix(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_segment: float,
    t_end_segment: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    tau_p_i: float,
    particle_diameter_i: float,
    particle_density_i: float,
    particle_mass_i: float,
    dep_particle_rel_permittivity_i: float,
    thermophoretic_coeff_i: float,
    flow_scale_particle_i: float,
    drag_scale_particle_i: float,
    body_scale_particle_i: float,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    global_body_accel_scale: float,
    body_accel: np.ndarray,
    min_tau_p_s: float,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    max_halving_count: int,
    electric_q_over_m_i: Optional[float] = None,
    force_runtime: ForceRuntimeParameters | None = None,
    require_clean_prefix: bool = False,
) -> ValidMaskPrefixResolution:
    x_start = np.asarray(x0, dtype=np.float64).copy()
    v_start = np.asarray(v0, dtype=np.float64).copy()
    dt_seg = max(float(dt_segment), 0.0)
    halving_limit = int(max(0, max_halving_count))
    if dt_seg <= 0.0 or halving_limit <= 0:
        return ValidMaskPrefixResolution(
            position=x_start,
            velocity=v_start,
            accepted_dt=0.0,
            retry_count=0,
            found_valid_prefix=False,
        )

    retry_count = 0
    for split_idx in range(1, halving_limit + 1):
        retry_count += 1
        prefix_dt = float(dt_seg) * (0.5 ** int(split_idx))
        prefix_t_end = float(t_end_segment) - float(dt_seg) + float(prefix_dt)
        x_retry, v_retry, _substeps, _stage_points, retry_status = advance_freeflight_segment(
            x0=x_start,
            v0=v_start,
            dt_segment=float(prefix_dt),
            t_end_segment=float(prefix_t_end),
            spatial_dim=int(spatial_dim),
            compiled=compiled,
            integrator_mode=int(integrator_mode),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            tau_p_i=float(tau_p_i),
            particle_diameter_i=float(particle_diameter_i),
            particle_density_i=float(particle_density_i),
            particle_mass_i=float(particle_mass_i),
            dep_particle_rel_permittivity_i=float(dep_particle_rel_permittivity_i),
            thermophoretic_coeff_i=float(thermophoretic_coeff_i),
            flow_scale_particle_i=float(flow_scale_particle_i),
            drag_scale_particle_i=float(drag_scale_particle_i),
            body_scale_particle_i=float(body_scale_particle_i),
            global_flow_scale=float(global_flow_scale),
            global_drag_tau_scale=float(global_drag_tau_scale),
            global_body_accel_scale=float(global_body_accel_scale),
            body_accel=body_accel,
            min_tau_p_s=float(min_tau_p_s),
            gas_density_kgm3=float(gas_density_kgm3),
            gas_mu_pas=float(gas_mu_pas),
            gas_temperature_K=float(gas_temperature_K),
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
            drag_model_mode=int(drag_model_mode),
            electric_q_over_m_i=electric_q_over_m_i,
            force_runtime=force_runtime,
        )
        status_ok = (
            int(retry_status) == int(VALID_MASK_STATUS_CLEAN)
            if bool(require_clean_prefix)
            else not bool(valid_mask_status_requires_stop(int(retry_status)))
        )
        if bool(status_ok):
            return ValidMaskPrefixResolution(
                position=np.asarray(x_retry, dtype=np.float64),
                velocity=np.asarray(v_retry, dtype=np.float64),
                accepted_dt=float(prefix_dt),
                retry_count=int(retry_count),
                found_valid_prefix=True,
            )

    return ValidMaskPrefixResolution(
        position=x_start,
        velocity=v_start,
        accepted_dt=0.0,
        retry_count=int(retry_count),
        found_valid_prefix=False,
    )


def _force_runtime_needs_precise_substep(force_runtime: ForceRuntimeParameters | None) -> bool:
    if force_runtime is None:
        return False
    return (
        bool(force_runtime.thermophoresis_enabled)
        or bool(force_runtime.dielectrophoresis_enabled)
        or bool(force_runtime.lift_enabled)
        or bool(force_runtime.pressure_gradient_enabled)
        or bool(force_runtime.virtual_mass_enabled)
    )


def _advance_particles_with_precise_substeps(
    *,
    spatial_dim: int,
    backend: CompiledRuntimeBackend,
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    tau_p: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density_arr: np.ndarray,
    particle_mass_arr: np.ndarray,
    dep_eps_arr: np.ndarray,
    thermo_coeff_arr: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
    t: float,
    dt_step: float,
    phys: Mapping[str, object],
    body_accel: np.ndarray,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    drag_model_mode: int,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    x_trial: np.ndarray,
    v_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    substep_counts: np.ndarray,
    valid_mask_status_flags: np.ndarray,
    electric_q_over_m_particle: Optional[np.ndarray],
    force_runtime: ForceRuntimeParameters | None,
) -> None:
    dim = int(spatial_dim)
    qom_arr = None if electric_q_over_m_particle is None else np.asarray(electric_q_over_m_particle, dtype=np.float64)
    for i in range(x.shape[0]):
        if not bool(active[i]):
            x_trial[i, :dim] = x[i, :dim]
            v_trial[i, :dim] = v[i, :dim]
            x_mid_trial[i, :dim] = x[i, :dim]
            substep_counts[i] = 1
            valid_mask_status_flags[i] = int(VALID_MASK_STATUS_CLEAN)
            continue
        qom_i = None if qom_arr is None else float(qom_arr[i])
        x_out, v_out, n_substeps, stage_points, mask_status = advance_freeflight_segment(
            x0=np.asarray(x[i, :dim], dtype=np.float64),
            v0=np.asarray(v[i, :dim], dtype=np.float64),
            dt_segment=float(dt_step),
            t_end_segment=float(t),
            spatial_dim=dim,
            compiled=backend,
            integrator_mode=int(integrator_mode),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            tau_p_i=float(tau_p[i]),
            particle_diameter_i=float(particle_diameter[i]),
            particle_density_i=float(particle_density_arr[i]),
            particle_mass_i=float(particle_mass_arr[i]),
            dep_particle_rel_permittivity_i=float(dep_eps_arr[i]),
            thermophoretic_coeff_i=float(thermo_coeff_arr[i]),
            flow_scale_particle_i=float(flow_scale_particle[i]),
            drag_scale_particle_i=float(drag_scale_particle[i]),
            body_scale_particle_i=float(body_scale_particle[i]),
            global_flow_scale=float(phys['flow_scale']),
            global_drag_tau_scale=float(phys['drag_tau_scale']),
            global_body_accel_scale=float(phys['body_accel_scale']),
            body_accel=np.asarray(body_accel, dtype=np.float64)[:dim],
            min_tau_p_s=float(phys['min_tau_p_s']),
            gas_density_kgm3=float(gas_density_kgm3),
            gas_mu_pas=float(gas_mu_pas),
            gas_temperature_K=float(phys.get('gas_temperature_K', 300.0)),
            gas_molecular_mass_kg=float(phys.get('gas_molecular_mass_kg', 60.0 * 1.66053906660e-27)),
            drag_model_mode=int(drag_model_mode),
            electric_q_over_m_i=qom_i,
            force_runtime=force_runtime,
        )
        x_trial[i, :dim] = np.asarray(x_out, dtype=np.float64)[:dim]
        v_trial[i, :dim] = np.asarray(v_out, dtype=np.float64)[:dim]
        stage_arr = np.asarray(stage_points, dtype=np.float64)
        x_mid_trial[i, :dim] = stage_arr[0, :dim] if stage_arr.ndim == 2 and stage_arr.shape[0] > 0 else x_trial[i, :dim]
        substep_counts[i] = int(max(1, n_substeps))
        valid_mask_status_flags[i] = int(mask_status)


def _advance_trial_particles(
    *,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    tau_p: np.ndarray,
    particle_diameter: np.ndarray,
    particle_mass: np.ndarray | None = None,
    particle_density: np.ndarray | None = None,
    dep_particle_rel_permittivity: np.ndarray | None = None,
    thermophoretic_coeff: np.ndarray | None = None,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
    t: float,
    dt_step: float,
    phys: Mapping[str, object],
    body_accel: np.ndarray,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    drag_model_mode: int,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    x_trial: np.ndarray,
    v_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    substep_counts: np.ndarray,
    valid_mask_status_flags: np.ndarray,
    electric_q_over_m_particle: Optional[np.ndarray] = None,
    force_runtime: ForceRuntimeParameters | None = None,
) -> None:
    backend = _coerce_compiled_backend(compiled)
    if particle_density is None:
        particle_density_arr = np.ones_like(tau_p, dtype=np.float64) * 1000.0
    else:
        particle_density_arr = np.asarray(particle_density, dtype=np.float64)
    if particle_mass is None:
        particle_mass_arr = particle_density_arr * np.pi * np.asarray(particle_diameter, dtype=np.float64) ** 3 / 6.0
    else:
        particle_mass_arr = np.asarray(particle_mass, dtype=np.float64)
    dep_eps_arr = (
        np.ones_like(tau_p, dtype=np.float64) * np.nan
        if dep_particle_rel_permittivity is None
        else np.asarray(dep_particle_rel_permittivity, dtype=np.float64)
    )
    thermo_coeff_arr = (
        np.ones_like(tau_p, dtype=np.float64) * np.nan
        if thermophoretic_coeff is None
        else np.asarray(thermophoretic_coeff, dtype=np.float64)
    )
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        if int(spatial_dim) != 2:
            raise ValueError('triangle_mesh_2d backend currently supports only spatial_dim=2')
        uses_triangle_precise_forces = (
            electric_q_over_m_particle is not None and len(getattr(backend, 'electric_field_names', ())) >= 2
        ) or _force_runtime_needs_precise_substep(force_runtime) or (
            force_runtime is not None and bool(force_runtime.gravity_buoyancy_enabled)
        )
        if bool(uses_triangle_precise_forces):
            _advance_particles_with_precise_substeps(
                spatial_dim=2,
                backend=backend,
                x=x,
                v=v,
                active=active,
                tau_p=tau_p,
                particle_diameter=particle_diameter,
                particle_density_arr=particle_density_arr,
                particle_mass_arr=particle_mass_arr,
                dep_eps_arr=dep_eps_arr,
                thermo_coeff_arr=thermo_coeff_arr,
                flow_scale_particle=flow_scale_particle,
                drag_scale_particle=drag_scale_particle,
                body_scale_particle=body_scale_particle,
                t=float(t),
                dt_step=float(dt_step),
                phys=phys,
                body_accel=body_accel,
                gas_density_kgm3=float(gas_density_kgm3),
                gas_mu_pas=float(gas_mu_pas),
                drag_model_mode=int(drag_model_mode),
                integrator_mode=int(integrator_mode),
                adaptive_substep_enabled=int(adaptive_substep_enabled),
                adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
                adaptive_substep_max_splits=int(adaptive_substep_max_splits),
                x_trial=x_trial,
                v_trial=v_trial,
                x_mid_trial=x_mid_trial,
                substep_counts=substep_counts,
                valid_mask_status_flags=valid_mask_status_flags,
                electric_q_over_m_particle=electric_q_over_m_particle,
                force_runtime=force_runtime,
            )
            return
        extra_accel = np.zeros((x.shape[0], 2), dtype=np.float64)
        kernel_drag_scale_particle = np.asarray(drag_scale_particle, dtype=np.float64)
        kernel_body_scale_particle = np.asarray(body_scale_particle, dtype=np.float64)
        accel_shape = np.asarray(backend.accel_shape, dtype=np.int32)
        advance_particles_2d_triangle_mesh_inplace(
            x, v, active, tau_p, particle_diameter, particle_density_arr, flow_scale_particle, kernel_drag_scale_particle, kernel_body_scale_particle,
            float(t), float(dt_step), float(phys['flow_scale']), float(phys['drag_tau_scale']), float(phys['body_accel_scale']),
            float(body_accel[0]), float(body_accel[1]), float(phys['min_tau_p_s']),
            float(phys.get('gas_molecular_mass_kg', 60.0 * 1.66053906660e-27)), int(drag_model_mode),
            int(integrator_mode),
            int(adaptive_substep_enabled),
            float(adaptive_substep_tau_ratio),
            int(adaptive_substep_max_splits),
            np.asarray(backend.mesh_vertices, dtype=np.float64),
            np.asarray(backend.mesh_triangles, dtype=np.int32),
            np.asarray(backend.accel_origin, dtype=np.float64),
            np.asarray(backend.accel_cell_size, dtype=np.float64),
            int(accel_shape[0]),
            int(accel_shape[1]),
            np.asarray(backend.accel_cell_offsets, dtype=np.int32),
            np.asarray(backend.accel_triangle_indices, dtype=np.int32),
            float(backend.support_tolerance_m),
            np.asarray(backend.times, dtype=np.float64),
            np.asarray(backend.ux, dtype=np.float64),
            np.asarray(backend.uy, dtype=np.float64),
            np.asarray(backend.gas_density, dtype=np.float64),
            np.asarray(backend.gas_mu, dtype=np.float64),
            np.asarray(backend.gas_temperature, dtype=np.float64),
            np.asarray(extra_accel[:, 0], dtype=np.float64),
            np.asarray(extra_accel[:, 1], dtype=np.float64),
            x_trial, v_trial, x_mid_trial, substep_counts, valid_mask_status_flags,
        )
        return
    if bool(_force_runtime_needs_precise_substep(force_runtime)):
        _advance_particles_with_precise_substeps(
            spatial_dim=int(spatial_dim),
            backend=backend,
            x=x,
            v=v,
            active=active,
            tau_p=tau_p,
            particle_diameter=particle_diameter,
            particle_density_arr=particle_density_arr,
            particle_mass_arr=particle_mass_arr,
            dep_eps_arr=dep_eps_arr,
            thermo_coeff_arr=thermo_coeff_arr,
            flow_scale_particle=flow_scale_particle,
            drag_scale_particle=drag_scale_particle,
            body_scale_particle=body_scale_particle,
            t=float(t),
            dt_step=float(dt_step),
            phys=phys,
            body_accel=body_accel,
            gas_density_kgm3=float(gas_density_kgm3),
            gas_mu_pas=float(gas_mu_pas),
            drag_model_mode=int(drag_model_mode),
            integrator_mode=int(integrator_mode),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            x_trial=x_trial,
            v_trial=v_trial,
            x_mid_trial=x_mid_trial,
            substep_counts=substep_counts,
            valid_mask_status_flags=valid_mask_status_flags,
            electric_q_over_m_particle=electric_q_over_m_particle,
            force_runtime=force_runtime,
        )
        return
    extra_accel = np.zeros((x.shape[0], int(spatial_dim)), dtype=np.float64)
    kernel_drag_scale_particle = np.asarray(drag_scale_particle, dtype=np.float64)
    kernel_body_scale_particle = np.asarray(body_scale_particle, dtype=np.float64)
    kernel_electric_q_over_m_particle = (
        None if electric_q_over_m_particle is None else np.asarray(electric_q_over_m_particle, dtype=np.float64)
    )
    gravity_buoyancy_enabled = int(force_runtime is not None and bool(force_runtime.gravity_buoyancy_enabled))
    valid_mask = np.asarray(backend.valid_mask, dtype=bool)
    core_valid_mask = np.asarray(backend.core_valid_mask, dtype=bool)
    if int(spatial_dim) == 2:
        xs, ys = backend.axes
        qom_particle = (
            np.zeros(x.shape[0], dtype=np.float64)
            if kernel_electric_q_over_m_particle is None
            else np.asarray(kernel_electric_q_over_m_particle, dtype=np.float64)
        )
        electric_x = (
            np.zeros_like(backend.ux, dtype=np.float64)
            if backend.electric_x is None
            else np.asarray(backend.electric_x, dtype=np.float64)
        )
        electric_y = (
            np.zeros_like(backend.uy, dtype=np.float64)
            if backend.electric_y is None
            else np.asarray(backend.electric_y, dtype=np.float64)
        )
        dynamic_electric_enabled = int(electric_q_over_m_particle is not None and backend.electric_x is not None and backend.electric_y is not None)
        advance_particles_2d_inplace(
            x, v, active, tau_p, particle_diameter, particle_density_arr, flow_scale_particle, kernel_drag_scale_particle, kernel_body_scale_particle,
            float(t), float(dt_step), float(phys['flow_scale']), float(phys['drag_tau_scale']), float(phys['body_accel_scale']),
            float(body_accel[0]), float(body_accel[1]), float(phys['min_tau_p_s']),
            float(gas_density_kgm3), float(gas_mu_pas), float(phys.get('gas_temperature_K', 300.0)), float(phys.get('gas_molecular_mass_kg', 60.0 * 1.66053906660e-27)), int(drag_model_mode),
            int(integrator_mode),
            int(adaptive_substep_enabled),
            float(adaptive_substep_tau_ratio),
            int(adaptive_substep_max_splits),
            xs, ys, backend.times, backend.ux, backend.uy,
            qom_particle, electric_x, electric_y, int(dynamic_electric_enabled),
            np.asarray(extra_accel[:, 0], dtype=np.float64),
            np.asarray(extra_accel[:, 1], dtype=np.float64),
            int(gravity_buoyancy_enabled),
            backend.gas_density, backend.gas_mu, backend.gas_temperature,
            valid_mask,
            core_valid_mask,
            x_trial, v_trial, x_mid_trial, substep_counts, valid_mask_status_flags,
        )
        return
    xs, ys, zs = backend.axes
    uz = backend.uz if backend.uz is not None else np.zeros((1,) + valid_mask.shape, dtype=np.float64)
    qom_particle = (
        np.zeros(x.shape[0], dtype=np.float64)
        if kernel_electric_q_over_m_particle is None
        else np.asarray(kernel_electric_q_over_m_particle, dtype=np.float64)
    )
    electric_x = (
        np.zeros_like(backend.ux, dtype=np.float64)
        if backend.electric_x is None
        else np.asarray(backend.electric_x, dtype=np.float64)
    )
    electric_y = (
        np.zeros_like(backend.uy, dtype=np.float64)
        if backend.electric_y is None
        else np.asarray(backend.electric_y, dtype=np.float64)
    )
    electric_z = (
        np.zeros_like(uz, dtype=np.float64)
        if backend.electric_z is None
        else np.asarray(backend.electric_z, dtype=np.float64)
    )
    dynamic_electric_enabled = int(
        electric_q_over_m_particle is not None
        and backend.electric_x is not None
        and backend.electric_y is not None
        and backend.electric_z is not None
    )
    advance_particles_3d_inplace(
        x, v, active, tau_p, particle_diameter, particle_density_arr, flow_scale_particle, kernel_drag_scale_particle, kernel_body_scale_particle,
        float(t), float(dt_step), float(phys['flow_scale']), float(phys['drag_tau_scale']), float(phys['body_accel_scale']),
        float(body_accel[0]), float(body_accel[1]), float(body_accel[2]), float(phys['min_tau_p_s']),
        float(gas_density_kgm3), float(gas_mu_pas), float(phys.get('gas_temperature_K', 300.0)), float(phys.get('gas_molecular_mass_kg', 60.0 * 1.66053906660e-27)), int(drag_model_mode),
        int(integrator_mode),
        int(adaptive_substep_enabled),
        float(adaptive_substep_tau_ratio),
        int(adaptive_substep_max_splits),
        xs, ys, zs, backend.times, backend.ux, backend.uy, uz,
        qom_particle, electric_x, electric_y, electric_z, int(dynamic_electric_enabled),
        np.asarray(extra_accel[:, 0], dtype=np.float64),
        np.asarray(extra_accel[:, 1], dtype=np.float64),
        np.asarray(extra_accel[:, 2], dtype=np.float64),
        int(gravity_buoyancy_enabled),
        backend.gas_density, backend.gas_mu, backend.gas_temperature,
        valid_mask,
        core_valid_mask,
        x_trial, v_trial, x_mid_trial, substep_counts, valid_mask_status_flags,
    )


__all__ = (
    'CompiledRuntimeBackend',
    'CompiledRuntimeBackendLike',
    'RegularRectilinearCompiledBackend',
    'TriangleMesh2DCompiledBackend',
    'ValidMaskPrefixResolution',
    '_advance_trial_particles',
    '_compile_runtime_arrays',
    '_sample_flow_vector_at',
    '_stage_points_from_trial',
    '_stage_sample_times',
    'advance_freeflight_partial',
    'advance_freeflight_segment',
    'resolve_valid_mask_prefix',
)
