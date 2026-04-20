from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .compiled_field_backend import CompiledRuntimeBackendLike
from .high_fidelity_freeflight import ValidMaskPrefixResolution


def resolve_valid_mask_retry_then_stop(
    *,
    resolve_prefix: Callable[..., ValidMaskPrefixResolution],
    collision_diagnostics: Dict[str, object],
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
    force_runtime: object | None = None,
) -> ValidMaskPrefixResolution:
    resolution = resolve_prefix(
        x0=x0,
        v0=v0,
        dt_segment=float(dt_segment),
        t_end_segment=float(t_end_segment),
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
        max_halving_count=int(adaptive_substep_max_splits),
    )
    collision_diagnostics['invalid_mask_retry_count'] += int(resolution.retry_count)
    if not bool(resolution.found_valid_prefix):
        collision_diagnostics['invalid_mask_retry_exhausted_count'] += 1
    return resolution
