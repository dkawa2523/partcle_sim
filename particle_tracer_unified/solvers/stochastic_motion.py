from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

from .compiled_field_backend import (
    CompiledRuntimeBackendLike,
    sample_compiled_flow_vectors,
    sample_compiled_gas_properties_vectors,
)
from .integrator_common import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
)

K_BOLTZMANN = 1.380649e-23


@dataclass(frozen=True, slots=True)
class StochasticMotionConfig:
    enabled: bool = False
    model: str = 'underdamped_langevin'
    stride: int = 10
    seed: int = 12345
    temperature_source: str = 'field_T_then_gas'


def _bool_from_config(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'1', 'true', 'yes', 'on'}:
            return True
        if text in {'0', 'false', 'no', 'off'}:
            return False
    return bool(value)


def parse_stochastic_motion_config(
    solver_cfg: Mapping[str, object],
    *,
    default_seed: int,
) -> StochasticMotionConfig:
    raw = solver_cfg.get('stochastic_motion', {}) if isinstance(solver_cfg, Mapping) else {}
    if raw is None:
        raw = {}
    if isinstance(raw, (bool, int, np.integer, str)):
        cfg: Mapping[str, object] = {'enabled': raw}
    elif isinstance(raw, Mapping):
        cfg = raw
    else:
        raise ValueError('solver.stochastic_motion must be a mapping or boolean')

    enabled = _bool_from_config(cfg.get('enabled', False), default=False)
    model = str(cfg.get('model', 'underdamped_langevin')).strip().lower()
    if model not in {'underdamped_langevin'}:
        raise ValueError("solver.stochastic_motion.model must be 'underdamped_langevin'")
    stride = int(cfg.get('stride', 10))
    if stride < 1:
        raise ValueError('solver.stochastic_motion.stride must be >= 1')
    seed = int(cfg.get('seed', default_seed))
    temperature_source = str(cfg.get('temperature_source', 'field_T_then_gas')).strip().lower()
    if temperature_source not in {'field_t_then_gas', 'gas'}:
        raise ValueError("solver.stochastic_motion.temperature_source must be 'field_T_then_gas' or 'gas'")
    if temperature_source == 'field_t_then_gas':
        temperature_source = 'field_T_then_gas'
    return StochasticMotionConfig(
        enabled=bool(enabled),
        model=str(model),
        stride=int(stride),
        seed=int(seed),
        temperature_source=str(temperature_source),
    )


def stochastic_motion_report(config: StochasticMotionConfig) -> Dict[str, object]:
    return {
        'enabled': int(bool(config.enabled)),
        'model': str(config.model),
        'stride': int(config.stride),
        'seed': int(config.seed),
        'temperature_source': str(config.temperature_source),
        'kick_event_count': 0,
        'kicked_particle_count': 0,
        'component_count': 0,
        'velocity_kick_rms_mps': 0.0,
        'last_velocity_kick_rms_mps': 0.0,
        'last_mean_sigma_v_mps': 0.0,
        'last_max_sigma_v_mps': 0.0,
        'last_mean_temperature_K': 0.0,
        'last_mean_tau_eff_s': 0.0,
    }


def _schiller_naumann_correction(reynolds: np.ndarray) -> np.ndarray:
    re = np.maximum(np.asarray(reynolds, dtype=np.float64), 0.0)
    correction = np.ones_like(re, dtype=np.float64)
    mid = (re > 1.0e-12) & (re < 1000.0)
    high = re >= 1000.0
    correction[mid] = 1.0 + 0.15 * np.power(re[mid], 0.687)
    correction[high] = 0.01875 * re[high]
    return np.maximum(correction, 1.0e-30)


def _effective_tau_vector(
    *,
    tau_stokes: np.ndarray,
    slip_speed: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density: np.ndarray,
    gas_density: np.ndarray,
    gas_mu: np.ndarray,
    gas_temperature: np.ndarray,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    min_tau_p_s: float,
) -> np.ndarray:
    tau = np.maximum(np.asarray(tau_stokes, dtype=np.float64), float(min_tau_p_s))
    if int(drag_model_mode) == int(DRAG_MODEL_STOKES):
        return tau
    if int(drag_model_mode) == int(DRAG_MODEL_EPSTEIN):
        rho_p = np.maximum(np.asarray(particle_density, dtype=np.float64), 0.0)
        diameter = np.maximum(np.asarray(particle_diameter, dtype=np.float64), 0.0)
        rho_g = np.maximum(np.asarray(gas_density, dtype=np.float64), 0.0)
        temp = np.maximum(np.asarray(gas_temperature, dtype=np.float64), 1.0)
        mol_mass = max(float(gas_molecular_mass_kg), 1.0e-30)
        thermal_speed = np.sqrt(8.0 * K_BOLTZMANN * temp / (np.pi * mol_mass))
        denom = 2.0 * rho_g * thermal_speed
        tau_epstein = np.divide(
            rho_p * diameter,
            denom,
            out=np.asarray(tau, dtype=np.float64).copy(),
            where=(rho_p > 0.0) & (diameter > 0.0) & (denom > 0.0) & np.isfinite(denom),
        )
        tau_epstein = np.where(np.isfinite(tau_epstein) & (tau_epstein > 0.0), tau_epstein, tau)
        return np.maximum(tau_epstein, float(min_tau_p_s))
    diameter = np.maximum(np.asarray(particle_diameter, dtype=np.float64), 0.0)
    rho_g = np.maximum(np.asarray(gas_density, dtype=np.float64), 0.0)
    mu = np.maximum(np.asarray(gas_mu, dtype=np.float64), 1.0e-30)
    re = rho_g * diameter * np.maximum(np.asarray(slip_speed, dtype=np.float64), 0.0) / mu
    return np.maximum(float(min_tau_p_s), tau / _schiller_naumann_correction(re))


def apply_langevin_velocity_kick(
    *,
    config: StochasticMotionConfig,
    rng: np.random.Generator,
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    t_eval: float,
    delta_t_s: float,
    active_mask: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    tau_p: np.ndarray,
    particle_mass: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    min_tau_p_s: float,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
) -> Dict[str, object]:
    if not bool(config.enabled) or float(delta_t_s) <= 0.0:
        return {'applied': False}
    active = np.asarray(active_mask, dtype=bool)
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return {'applied': False, 'particle_count': 0}

    positions = np.asarray(x[indices, : int(spatial_dim)], dtype=np.float64)
    velocities = np.asarray(v[indices, : int(spatial_dim)], dtype=np.float64)
    flow = sample_compiled_flow_vectors(compiled, int(spatial_dim), float(t_eval), positions)
    flow_scale = float(global_flow_scale) * np.asarray(flow_scale_particle[indices], dtype=np.float64)
    target_flow = flow * flow_scale[:, None]
    slip_speed = np.linalg.norm(velocities - target_flow, axis=1)

    rho_g, mu_g, temp_g = sample_compiled_gas_properties_vectors(
        compiled,
        int(spatial_dim),
        float(t_eval),
        positions,
        fallback_density_kgm3=float(gas_density_kgm3),
        fallback_mu_pas=float(gas_mu_pas),
        fallback_temperature_K=float(gas_temperature_K),
    )
    if str(config.temperature_source) == 'gas':
        temp_g = np.full(indices.size, float(gas_temperature_K), dtype=np.float64)
    temp_g = np.where(np.isfinite(temp_g) & (temp_g > 0.0), temp_g, float(gas_temperature_K))

    tau_stokes = (
        np.asarray(tau_p[indices], dtype=np.float64)
        * float(global_drag_tau_scale)
        * np.maximum(np.asarray(drag_scale_particle[indices], dtype=np.float64), 1.0e-6)
    )
    tau_eff = _effective_tau_vector(
        tau_stokes=tau_stokes,
        slip_speed=slip_speed,
        particle_diameter=np.asarray(particle_diameter[indices], dtype=np.float64),
        particle_density=np.asarray(particle_density[indices], dtype=np.float64),
        gas_density=np.asarray(rho_g, dtype=np.float64),
        gas_mu=np.asarray(mu_g, dtype=np.float64),
        gas_temperature=np.asarray(temp_g, dtype=np.float64),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        drag_model_mode=int(drag_model_mode),
        min_tau_p_s=float(min_tau_p_s),
    )
    mass = np.maximum(np.asarray(particle_mass[indices], dtype=np.float64), 1.0e-30)
    exponent = np.clip(-2.0 * float(delta_t_s) / np.maximum(tau_eff, 1.0e-30), -745.0, 0.0)
    variance = (K_BOLTZMANN * temp_g / mass) * (1.0 - np.exp(exponent))
    variance = np.where(np.isfinite(variance) & (variance > 0.0), variance, 0.0)
    sigma_v = np.sqrt(variance)
    kicks = rng.normal(loc=0.0, scale=1.0, size=(indices.size, int(spatial_dim))) * sigma_v[:, None]
    v[indices, : int(spatial_dim)] += kicks

    component_count = int(kicks.size)
    sum_sq = float(np.sum(kicks * kicks))
    rms = float(np.sqrt(sum_sq / component_count)) if component_count else 0.0
    return {
        'applied': True,
        'particle_count': int(indices.size),
        'component_count': int(component_count),
        'sum_sq': float(sum_sq),
        'rms_velocity_kick_mps': float(rms),
        'mean_sigma_v_mps': float(np.mean(sigma_v)) if sigma_v.size else 0.0,
        'max_sigma_v_mps': float(np.max(sigma_v)) if sigma_v.size else 0.0,
        'mean_temperature_K': float(np.mean(temp_g)) if temp_g.size else 0.0,
        'mean_tau_eff_s': float(np.mean(tau_eff)) if tau_eff.size else 0.0,
    }


def merge_stochastic_motion_diagnostics(
    diagnostics: Dict[str, object],
    config: StochasticMotionConfig,
    result: Mapping[str, object],
) -> None:
    summary = diagnostics.setdefault('stochastic_motion', stochastic_motion_report(config))
    if not isinstance(summary, dict):
        summary = stochastic_motion_report(config)
        diagnostics['stochastic_motion'] = summary
    summary['enabled'] = int(bool(config.enabled))
    summary['model'] = str(config.model)
    summary['stride'] = int(config.stride)
    summary['seed'] = int(config.seed)
    summary['temperature_source'] = str(config.temperature_source)
    if not bool(result.get('applied', False)):
        return
    particle_count = int(result.get('particle_count', 0))
    component_count = int(result.get('component_count', 0))
    sum_sq = float(result.get('sum_sq', 0.0))
    previous_components = int(summary.get('component_count', 0))
    previous_sum_sq = float(summary.get('velocity_kick_sum_sq', 0.0))
    total_components = int(previous_components + component_count)
    total_sum_sq = float(previous_sum_sq + sum_sq)
    summary['kick_event_count'] = int(summary.get('kick_event_count', 0)) + 1
    summary['kicked_particle_count'] = int(summary.get('kicked_particle_count', 0)) + int(particle_count)
    summary['component_count'] = int(total_components)
    summary['velocity_kick_sum_sq'] = float(total_sum_sq)
    summary['velocity_kick_rms_mps'] = (
        float(np.sqrt(total_sum_sq / total_components)) if total_components else 0.0
    )
    summary['last_velocity_kick_rms_mps'] = float(result.get('rms_velocity_kick_mps', 0.0))
    summary['last_mean_sigma_v_mps'] = float(result.get('mean_sigma_v_mps', 0.0))
    summary['last_max_sigma_v_mps'] = float(result.get('max_sigma_v_mps', 0.0))
    summary['last_mean_temperature_K'] = float(result.get('mean_temperature_K', 0.0))
    summary['last_mean_tau_eff_s'] = float(result.get('mean_tau_eff_s', 0.0))
