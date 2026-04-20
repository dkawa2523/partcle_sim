from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from ..core.datamodel import RegularFieldND
from .compiled_field_backend import (
    CompiledRuntimeBackendLike,
    RegularRectilinearCompiledBackend,
    coerce_compiled_backend,
    _sample_regular_time_grid_points_2d,
)
from .plasma_background import PreparedPlasmaBackground, debye_length_m

E_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
EPS0_F_M = 8.8541878128e-12
AMU_KG = 1.66053906660e-27


@dataclass(frozen=True, slots=True)
class ChargeModelConfig:
    enabled: bool = False
    mode: str = 'te_relaxation'
    background_source: str = 'field'
    update_stride: int = 1
    electron_temperature_quantity: str = ''
    electron_temperature_unit: str = 'eV'
    electron_density_quantity: str = ''
    ion_density_quantity: str = ''
    ion_temperature_quantity: str = ''
    ion_temperature_eV: float = 0.03
    ion_mass_amu: float = 69.0
    ion_charge_number: float = 1.0
    electron_sticking: float = 1.0
    ion_sticking: float = 1.0
    ion_velocity_model: str = 'max_thermal_bohm'
    bohm_velocity_factor: float = 1.0
    te_relaxation_alpha: float = 2.5
    relaxation_time_s: float = 1.0e-6
    max_abs_potential_V: float = 100.0
    newton_iterations: int = 6


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


def parse_charge_model_config(solver_cfg: Mapping[str, object]) -> ChargeModelConfig:
    raw = solver_cfg.get('charge_model', {}) if isinstance(solver_cfg, Mapping) else {}
    if raw is None:
        raw = {}
    if isinstance(raw, (bool, int, np.integer, str)):
        cfg: Mapping[str, object] = {'enabled': raw}
    elif isinstance(raw, Mapping):
        cfg = raw
    else:
        raise ValueError('solver.charge_model must be a mapping or boolean')

    enabled = _bool_from_config(cfg.get('enabled', False), default=False)
    mode = str(cfg.get('mode', 'te_relaxation')).strip().lower()
    aliases = {
        'v1': 'te_relaxation',
        'floating_te': 'te_relaxation',
        'te': 'te_relaxation',
        'v2': 'density_temperature_flux_relaxation',
        'flux_relaxation': 'density_temperature_flux_relaxation',
        'density_temperature_flux': 'density_temperature_flux_relaxation',
        'flux_balance': 'finite_rate_flux_balance',
        'saas_constant_flux_balance': 'finite_rate_flux_balance',
        'saas_flux_balance': 'finite_rate_flux_balance',
    }
    mode = aliases.get(mode, mode)
    if mode not in {'te_relaxation', 'density_temperature_flux_relaxation', 'finite_rate_flux_balance'}:
        raise ValueError(
            "solver.charge_model.mode must be 'te_relaxation', "
            "'density_temperature_flux_relaxation', or 'finite_rate_flux_balance'"
        )
    default_background_source = 'plasma_background' if mode == 'finite_rate_flux_balance' else 'field'
    background_source = str(cfg.get('background_source', default_background_source)).strip().lower()
    background_aliases = {
        'plasma': 'plasma_background',
        'saas': 'plasma_background',
        'saas_constant': 'plasma_background',
        'scalar': 'plasma_background',
        'regular_field': 'field',
    }
    background_source = background_aliases.get(background_source, background_source)
    if background_source not in {'field', 'plasma_background'}:
        raise ValueError("solver.charge_model.background_source must be 'field' or 'plasma_background'")
    stride = int(cfg.get('update_stride', cfg.get('stride', 1)))
    if stride < 1:
        raise ValueError('solver.charge_model.update_stride must be >= 1')
    electron_temperature_unit = str(cfg.get('electron_temperature_unit', 'eV')).strip().lower()
    if electron_temperature_unit not in {'ev', 'k'}:
        raise ValueError("solver.charge_model.electron_temperature_unit must be 'eV' or 'K'")
    ion_velocity_model = str(cfg.get('ion_velocity_model', 'max_thermal_bohm')).strip().lower()
    if ion_velocity_model not in {'bohm', 'thermal', 'max_thermal_bohm'}:
        raise ValueError("solver.charge_model.ion_velocity_model must be 'bohm', 'thermal', or 'max_thermal_bohm'")
    return ChargeModelConfig(
        enabled=bool(enabled),
        mode=str(mode),
        background_source=str(background_source),
        update_stride=int(stride),
        electron_temperature_quantity=str(cfg.get('electron_temperature_quantity', '')).strip(),
        electron_temperature_unit='eV' if electron_temperature_unit == 'ev' else 'K',
        electron_density_quantity=str(cfg.get('electron_density_quantity', '')).strip(),
        ion_density_quantity=str(cfg.get('ion_density_quantity', '')).strip(),
        ion_temperature_quantity=str(cfg.get('ion_temperature_quantity', '')).strip(),
        ion_temperature_eV=float(cfg.get('ion_temperature_eV', 0.03)),
        ion_mass_amu=float(cfg.get('ion_mass_amu', 69.0)),
        ion_charge_number=float(cfg.get('ion_charge_number', 1.0)),
        electron_sticking=float(cfg.get('electron_sticking', 1.0)),
        ion_sticking=float(cfg.get('ion_sticking', 1.0)),
        ion_velocity_model=str(ion_velocity_model),
        bohm_velocity_factor=float(cfg.get('bohm_velocity_factor', 1.0)),
        te_relaxation_alpha=float(cfg.get('te_relaxation_alpha', cfg.get('alpha', 2.5))),
        relaxation_time_s=float(cfg.get('relaxation_time_s', cfg.get('tau_s', 1.0e-6))),
        max_abs_potential_V=float(cfg.get('max_abs_potential_V', 100.0)),
        newton_iterations=int(cfg.get('newton_iterations', 6)),
    )


def charge_model_report(
    config: ChargeModelConfig,
    plasma_background: PreparedPlasmaBackground | None = None,
) -> Dict[str, object]:
    return {
        'enabled': int(bool(config.enabled)),
        'mode': str(config.mode),
        'background_source': str(config.background_source),
        'plasma_background_source': 'none' if plasma_background is None else str(plasma_background.source),
        'update_stride': int(config.update_stride),
        'ion_velocity_model': str(config.ion_velocity_model),
        'update_event_count': 0,
        'updated_particle_count': 0,
        'last_updated_particle_count': 0,
        'last_mean_Te_eV': 0.0,
        'last_mean_ne_m3': 0.0,
        'last_mean_ni_m3': 0.0,
        'last_mean_Ti_eV': 0.0,
        'last_mean_debye_length_m': 0.0,
        'last_mean_particle_radius_over_debye': 0.0,
        'last_mean_floating_potential_V': 0.0,
        'last_mean_equilibrium_charge_e': 0.0,
        'last_mean_tau_q_s': 0.0,
        'last_charge_response_regime': 'disabled',
        'last_mean_charge_C': 0.0,
        'last_mean_charge_e': 0.0,
        'final_min_charge_C': 0.0,
        'final_mean_charge_C': 0.0,
        'final_max_charge_C': 0.0,
        'final_min_charge_e': 0.0,
        'final_mean_charge_e': 0.0,
        'final_max_charge_e': 0.0,
    }


def validate_charge_model_support(
    config: ChargeModelConfig,
    runtime,
    compiled: CompiledRuntimeBackendLike,
    spatial_dim: int,
    plasma_background: PreparedPlasmaBackground | None = None,
) -> None:
    if not bool(config.enabled):
        return
    if int(spatial_dim) != 2:
        raise ValueError('solver.charge_model currently supports 2D regular rectilinear fields; 3D is planned separately')
    backend = coerce_compiled_backend(compiled)
    if not isinstance(backend, RegularRectilinearCompiledBackend):
        raise ValueError('solver.charge_model requires a regular rectilinear field backend')
    if backend.electric_x is None or backend.electric_y is None:
        raise ValueError('solver.charge_model requires electric field quantities E_x/E_y or E_r/E_z')
    if str(config.background_source) == 'plasma_background':
        if plasma_background is None:
            raise ValueError('solver.charge_model.background_source=plasma_background requires solver.plasma_background.source=saas_constant')
        return
    field_provider = getattr(runtime, 'field_provider', None)
    field = getattr(field_provider, 'field', None)
    if not isinstance(field, RegularFieldND):
        raise ValueError('solver.charge_model requires a regular field provider')
    _select_quantity(field, _temperature_names(config))
    if _is_flux_balance_mode(config):
        _select_quantity(field, _density_names(config))


def _is_flux_balance_mode(config: ChargeModelConfig) -> bool:
    return str(config.mode) in {'density_temperature_flux_relaxation', 'finite_rate_flux_balance'}


def _temperature_names(config: ChargeModelConfig) -> Sequence[str]:
    if config.electron_temperature_quantity:
        return (config.electron_temperature_quantity,)
    return ('Te', 'T_e', 'electron_temperature_eV', 'electron_temperature', 'Te_eV')


def _density_names(config: ChargeModelConfig) -> Sequence[str]:
    if config.electron_density_quantity:
        return (config.electron_density_quantity,)
    return ('ne', 'n_e', 'electron_density', 'electron_number_density')


def _ion_density_names(config: ChargeModelConfig) -> Sequence[str]:
    if config.ion_density_quantity:
        return (config.ion_density_quantity,)
    return ('ni', 'n_i', 'ion_density', 'ion_number_density')


def _ion_temperature_names(config: ChargeModelConfig) -> Sequence[str]:
    if config.ion_temperature_quantity:
        return (config.ion_temperature_quantity,)
    return ('Ti', 'T_i', 'ion_temperature_eV', 'ion_temperature', 'Ti_eV')


def _select_quantity(field: RegularFieldND, names: Sequence[str]):
    for name in names:
        if str(name) in field.quantities:
            return field.quantities[str(name)]
    raise ValueError(f"Missing required charge-model field quantity; tried {list(names)}")


def _optional_quantity(field: RegularFieldND, names: Sequence[str]):
    for name in names:
        if str(name) in field.quantities:
            return field.quantities[str(name)]
    return None


def _sample_series_2d(field: RegularFieldND, series, t_eval: float, positions: np.ndarray) -> np.ndarray:
    return _sample_regular_time_grid_points_2d(
        np.asarray(series.data, dtype=np.float64),
        tuple(np.asarray(ax, dtype=np.float64) for ax in field.axes),
        np.asarray(series.times, dtype=np.float64),
        float(t_eval),
        np.asarray(positions, dtype=np.float64),
    )


def _sample_temperature_eV(config: ChargeModelConfig, field: RegularFieldND, t_eval: float, positions: np.ndarray) -> np.ndarray:
    series = _select_quantity(field, _temperature_names(config))
    temp = _sample_series_2d(field, series, float(t_eval), positions)
    temp = np.asarray(temp, dtype=np.float64)
    if str(config.electron_temperature_unit).lower() == 'k':
        temp = temp / 11604.51812155008
    return np.where(np.isfinite(temp) & (temp > 0.0), temp, np.nan)


def _sample_optional_positive(field: RegularFieldND, series, t_eval: float, positions: np.ndarray, fallback: np.ndarray | float) -> np.ndarray:
    values = _sample_series_2d(field, series, float(t_eval), positions) if series is not None else fallback
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(positions.shape[0], float(arr), dtype=np.float64)
    fallback_arr = np.asarray(fallback, dtype=np.float64)
    if fallback_arr.ndim == 0:
        fallback_arr = np.full(positions.shape[0], float(fallback_arr), dtype=np.float64)
    return np.where(np.isfinite(arr) & (arr > 0.0), arr, fallback_arr)


def _charge_response_regime(delta_t_s: float, tau_q: np.ndarray) -> str:
    finite = np.asarray(tau_q, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return 'unknown'
    ratio = float(delta_t_s) / float(np.median(finite))
    if ratio < 0.2:
        return 'explicit_transient'
    if ratio < 2.0:
        return 'partially_relaxed'
    return 'quasi_equilibrium'


def _v1_equilibrium_charge(config: ChargeModelConfig, radius_m: np.ndarray, te_eV: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    capacitance = 4.0 * np.pi * EPS0_F_M * radius_m
    phi = -float(config.te_relaxation_alpha) * te_eV
    q_eq = capacitance * phi
    tau_q = np.full(q_eq.shape, max(float(config.relaxation_time_s), 1.0e-30), dtype=np.float64)
    return q_eq, tau_q, phi


def _v2_equilibrium_charge(
    config: ChargeModelConfig,
    radius_m: np.ndarray,
    old_charge: np.ndarray,
    te_eV: np.ndarray,
    ne_m3: np.ndarray,
    ni_m3: np.ndarray,
    ti_eV: np.ndarray,
    *,
    ion_mass_amu: float | None = None,
    ion_charge_number: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.maximum(radius_m, 1.0e-12)
    capacitance = 4.0 * np.pi * EPS0_F_M * radius
    area = 4.0 * np.pi * radius * radius
    te = np.maximum(te_eV, 1.0e-4)
    ti = np.maximum(ti_eV, 1.0e-4)
    ne = np.maximum(ne_m3, 0.0)
    ni = np.maximum(ni_m3, 0.0)
    ion_mass_value = float(config.ion_mass_amu if ion_mass_amu is None else ion_mass_amu)
    ion_charge_value = float(config.ion_charge_number if ion_charge_number is None else ion_charge_number)
    ion_mass = max(ion_mass_value, 1.0e-12) * AMU_KG
    zi = max(ion_charge_value, 1.0e-12)
    se = max(float(config.electron_sticking), 0.0)
    si = max(float(config.ion_sticking), 0.0)
    ve = np.sqrt(E_CHARGE_C * te / (2.0 * np.pi * ELECTRON_MASS_KG))
    vi_thermal = np.sqrt(E_CHARGE_C * ti / (2.0 * np.pi * ion_mass))
    vi_bohm = max(float(config.bohm_velocity_factor), 0.0) * np.sqrt(E_CHARGE_C * te / ion_mass)
    if str(config.ion_velocity_model) == 'thermal':
        vi = vi_thermal
    elif str(config.ion_velocity_model) == 'bohm':
        vi = vi_bohm
    else:
        vi = np.maximum(vi_thermal, vi_bohm)
    max_abs = max(float(config.max_abs_potential_V), 1.0e-6)
    phi = np.divide(old_charge, capacitance, out=-float(config.te_relaxation_alpha) * te, where=capacitance > 0.0)
    phi = np.where(np.isfinite(phi), phi, -float(config.te_relaxation_alpha) * te)
    phi = np.clip(phi, -max_abs, 0.0)
    iterations = max(1, int(config.newton_iterations))
    for _ in range(iterations):
        exp_arg = np.clip(phi / te, -80.0, 40.0)
        electron_flux = se * ne * ve * np.exp(exp_arg)
        ion_factor = np.maximum(1.0 - zi * phi / ti, 0.0)
        ion_flux = si * zi * ni * vi * ion_factor
        residual = ion_flux - electron_flux
        derivative = si * zi * ni * vi * (-zi / ti) - electron_flux / te
        step = np.divide(residual, derivative, out=np.zeros_like(phi), where=np.abs(derivative) > 1.0e-300)
        phi = np.clip(phi - step, -max_abs, 0.0)
    exp_arg = np.clip(phi / te, -80.0, 40.0)
    electron_flux = se * ne * ve * np.exp(exp_arg)
    derivative = si * zi * ni * vi * (-zi / ti) - electron_flux / te
    tau_q = np.divide(
        -capacitance,
        E_CHARGE_C * area * derivative,
        out=np.full_like(phi, max(float(config.relaxation_time_s), 1.0e-30)),
        where=(area > 0.0) & (derivative < -1.0e-300),
    )
    tau_q = np.clip(tau_q, 1.0e-12, max(float(config.relaxation_time_s), 1.0e-12))
    return capacitance * phi, tau_q, phi


def apply_charge_model_update(
    *,
    config: ChargeModelConfig,
    runtime,
    spatial_dim: int,
    t_eval: float,
    delta_t_s: float,
    active_mask: np.ndarray,
    x: np.ndarray,
    charge: np.ndarray,
    particle_diameter: np.ndarray,
    plasma_background: PreparedPlasmaBackground | None = None,
) -> Dict[str, object]:
    if not bool(config.enabled) or float(delta_t_s) <= 0.0:
        return {'applied': False}
    if int(spatial_dim) != 2:
        raise ValueError('solver.charge_model currently supports only 2D; 3D support is planned separately')
    active = np.asarray(active_mask, dtype=bool)
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return {'applied': False, 'particle_count': 0}
    positions = np.asarray(x[indices, :2], dtype=np.float64)
    radius = 0.5 * np.maximum(np.asarray(particle_diameter[indices], dtype=np.float64), 1.0e-12)
    old_charge = np.asarray(charge[indices], dtype=np.float64)
    ne = None
    ni = None
    ti = None
    debye = None
    ion_mass_amu = None
    ion_charge_number = None
    background_source = str(config.background_source)
    if background_source == 'plasma_background':
        if plasma_background is None:
            raise ValueError('solver.charge_model.background_source=plasma_background requires solver.plasma_background.source=saas_constant')
        te_eV = np.full(indices.size, float(plasma_background.electron_temperature_eV), dtype=np.float64)
        ne = np.full(indices.size, float(plasma_background.electron_density_m3), dtype=np.float64)
        ni = np.full(indices.size, float(plasma_background.ion_density_m3), dtype=np.float64)
        ti = np.full(indices.size, float(plasma_background.ion_temperature_eV), dtype=np.float64)
        debye = np.full(indices.size, float(plasma_background.debye_length_m), dtype=np.float64)
        ion_mass_amu = float(plasma_background.ion_mass_amu)
        ion_charge_number = float(plasma_background.ion_charge_number)
    else:
        field_provider = getattr(runtime, 'field_provider', None)
        field = getattr(field_provider, 'field', None)
        if not isinstance(field, RegularFieldND):
            raise ValueError('solver.charge_model requires a regular field provider')
        te_eV = _sample_temperature_eV(config, field, float(t_eval), positions)
    if not np.all(np.isfinite(te_eV)):
        raise ValueError('solver.charge_model sampled non-finite electron temperature inside active particle support')
    if str(config.mode) == 'te_relaxation':
        q_eq, tau_q, phi = _v1_equilibrium_charge(config, radius, te_eV)
    else:
        if ne is None or ni is None or ti is None:
            field_provider = getattr(runtime, 'field_provider', None)
            field = getattr(field_provider, 'field', None)
            if not isinstance(field, RegularFieldND):
                raise ValueError('solver.charge_model requires a regular field provider')
            ne_series = _select_quantity(field, _density_names(config))
            ne = _sample_optional_positive(field, ne_series, float(t_eval), positions, 0.0)
            ni_series = _optional_quantity(field, _ion_density_names(config))
            ni = _sample_optional_positive(field, ni_series, float(t_eval), positions, ne)
            ti_series = _optional_quantity(field, _ion_temperature_names(config))
            ti = _sample_optional_positive(field, ti_series, float(t_eval), positions, float(config.ion_temperature_eV))
            debye = debye_length_m(te_eV, ne, ti, ni, float(config.ion_charge_number))
        q_eq, tau_q, phi = _v2_equilibrium_charge(
            config,
            radius,
            old_charge,
            te_eV,
            np.asarray(ne, dtype=np.float64),
            np.asarray(ni, dtype=np.float64),
            np.asarray(ti, dtype=np.float64),
            ion_mass_amu=ion_mass_amu,
            ion_charge_number=ion_charge_number,
        )
    decay = np.exp(-float(delta_t_s) / np.maximum(tau_q, 1.0e-30))
    new_charge = q_eq + (old_charge - q_eq) * decay
    charge[indices] = np.where(np.isfinite(new_charge), new_charge, old_charge)
    updated = np.asarray(charge[indices], dtype=np.float64)
    debye_arr = np.asarray(debye, dtype=np.float64) if debye is not None else np.asarray([], dtype=np.float64)
    finite_debye = debye_arr[np.isfinite(debye_arr) & (debye_arr > 0.0)]
    r_over_debye = np.divide(
        radius,
        debye_arr,
        out=np.full_like(radius, np.nan, dtype=np.float64),
        where=np.isfinite(debye_arr) & (debye_arr > 0.0),
    ) if debye_arr.size else np.asarray([], dtype=np.float64)
    return {
        'applied': True,
        'particle_count': int(indices.size),
        'background_source': str(background_source),
        'mean_Te_eV': float(np.mean(te_eV)) if te_eV.size else 0.0,
        'mean_ne_m3': float(np.mean(ne)) if ne is not None and np.asarray(ne).size else 0.0,
        'mean_ni_m3': float(np.mean(ni)) if ni is not None and np.asarray(ni).size else 0.0,
        'mean_Ti_eV': float(np.mean(ti)) if ti is not None and np.asarray(ti).size else 0.0,
        'mean_debye_length_m': float(np.mean(finite_debye)) if finite_debye.size else 0.0,
        'mean_particle_radius_over_debye': float(np.nanmean(r_over_debye)) if r_over_debye.size and np.any(np.isfinite(r_over_debye)) else 0.0,
        'mean_floating_potential_V': float(np.mean(phi)) if np.asarray(phi).size else 0.0,
        'mean_equilibrium_charge_e': float(np.mean(q_eq / E_CHARGE_C)) if np.asarray(q_eq).size else 0.0,
        'mean_tau_q_s': float(np.mean(tau_q)) if tau_q.size else 0.0,
        'charge_response_regime': _charge_response_regime(float(delta_t_s), tau_q),
        'mean_charge_C': float(np.mean(updated)) if updated.size else 0.0,
        'mean_charge_e': float(np.mean(updated / E_CHARGE_C)) if updated.size else 0.0,
    }


def merge_charge_model_diagnostics(
    diagnostics: Dict[str, object],
    config: ChargeModelConfig,
    result: Mapping[str, object],
) -> None:
    summary = diagnostics.setdefault('charge_model', charge_model_report(config))
    if not isinstance(summary, dict):
        summary = charge_model_report(config)
        diagnostics['charge_model'] = summary
    summary['enabled'] = int(bool(config.enabled))
    summary['mode'] = str(config.mode)
    summary['background_source'] = str(config.background_source)
    summary['update_stride'] = int(config.update_stride)
    summary['ion_velocity_model'] = str(config.ion_velocity_model)
    if not bool(result.get('applied', False)):
        return
    particle_count = int(result.get('particle_count', 0))
    summary['update_event_count'] = int(summary.get('update_event_count', 0)) + 1
    summary['updated_particle_count'] = int(summary.get('updated_particle_count', 0)) + int(particle_count)
    summary['last_updated_particle_count'] = int(particle_count)
    summary['last_mean_Te_eV'] = float(result.get('mean_Te_eV', 0.0))
    summary['last_mean_ne_m3'] = float(result.get('mean_ne_m3', 0.0))
    summary['last_mean_ni_m3'] = float(result.get('mean_ni_m3', 0.0))
    summary['last_mean_Ti_eV'] = float(result.get('mean_Ti_eV', 0.0))
    summary['last_mean_debye_length_m'] = float(result.get('mean_debye_length_m', 0.0))
    summary['last_mean_particle_radius_over_debye'] = float(result.get('mean_particle_radius_over_debye', 0.0))
    summary['last_mean_floating_potential_V'] = float(result.get('mean_floating_potential_V', 0.0))
    summary['last_mean_equilibrium_charge_e'] = float(result.get('mean_equilibrium_charge_e', 0.0))
    summary['last_mean_tau_q_s'] = float(result.get('mean_tau_q_s', 0.0))
    summary['last_charge_response_regime'] = str(result.get('charge_response_regime', 'unknown'))
    summary['last_mean_charge_C'] = float(result.get('mean_charge_C', 0.0))
    summary['last_mean_charge_e'] = float(result.get('mean_charge_e', 0.0))


def finalize_charge_model_diagnostics(
    diagnostics: Dict[str, object],
    config: ChargeModelConfig,
    charge: np.ndarray,
) -> None:
    summary = diagnostics.setdefault('charge_model', charge_model_report(config))
    if not isinstance(summary, dict):
        summary = charge_model_report(config)
        diagnostics['charge_model'] = summary
    arr = np.asarray(charge, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    summary['final_min_charge_C'] = float(np.min(finite))
    summary['final_mean_charge_C'] = float(np.mean(finite))
    summary['final_max_charge_C'] = float(np.max(finite))
    q_e = finite / E_CHARGE_C
    summary['final_min_charge_e'] = float(np.min(q_e))
    summary['final_mean_charge_e'] = float(np.mean(q_e))
    summary['final_max_charge_e'] = float(np.max(q_e))
