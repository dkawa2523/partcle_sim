from __future__ import annotations

import numpy as np
from numba import njit

from ..core.integrator_registry import get_integrator_spec

INTEGRATOR_DRAG_RELAXATION = int(get_integrator_spec('drag_relaxation').mode)
INTEGRATOR_ETD = int(get_integrator_spec('etd').mode)
INTEGRATOR_ETD2 = int(get_integrator_spec('etd2').mode)

DRAG_MODEL_STOKES = 0
DRAG_MODEL_SCHILLER_NAUMANN = 1
DRAG_MODEL_EPSTEIN = 2

_K_BOLTZMANN = 1.380649e-23
_AMU_KG = 1.66053906660e-27


def drag_model_mode_from_name(name: object) -> int:
    value = str(name if name is not None else 'stokes').strip().lower()
    if value in {'', 'stokes', 'linear_stokes'}:
        return int(DRAG_MODEL_STOKES)
    if value in {'schiller_naumann', 'schiller-naumann', 'finite_re', 're_dependent'}:
        return int(DRAG_MODEL_SCHILLER_NAUMANN)
    if value in {'epstein', 'epstein_low_pressure', 'low_pressure_epstein', 'free_molecular'}:
        return int(DRAG_MODEL_EPSTEIN)
    raise ValueError("solver.drag_model must be 'stokes', 'schiller_naumann', or 'epstein'")


def drag_model_name_from_mode(mode: int) -> str:
    if int(mode) == int(DRAG_MODEL_SCHILLER_NAUMANN):
        return 'schiller_naumann'
    if int(mode) == int(DRAG_MODEL_EPSTEIN):
        return 'epstein'
    return 'stokes'


@njit(cache=True)
def schiller_naumann_drag_correction(reynolds):
    re = max(0.0, reynolds)
    if re <= 1.0e-12:
        return 1.0
    if re < 1000.0:
        return 1.0 + 0.15 * re ** 0.687
    return 0.01875 * re


@njit(cache=True)
def effective_tau_from_slip_speed(
    tau_stokes,
    slip_speed,
    particle_diameter_m,
    gas_density_kgm3,
    gas_mu_pas,
    drag_model_mode,
    min_tau_p_s,
    particle_density_kgm3=0.0,
    gas_temperature_K=300.0,
    gas_molecular_mass_kg=60.0 * _AMU_KG,
):
    tau = max(float(min_tau_p_s), float(tau_stokes))
    if int(drag_model_mode) == DRAG_MODEL_STOKES:
        return tau
    if int(drag_model_mode) == DRAG_MODEL_EPSTEIN:
        rho_p = max(float(particle_density_kgm3), 0.0)
        diameter = max(float(particle_diameter_m), 0.0)
        rho_g = max(float(gas_density_kgm3), 0.0)
        temp = max(float(gas_temperature_K), 1.0)
        mol_mass = max(float(gas_molecular_mass_kg), 1.0e-30)
        if rho_p <= 0.0 or diameter <= 0.0 or rho_g <= 0.0:
            return tau
        thermal_speed = np.sqrt(8.0 * _K_BOLTZMANN * temp / (np.pi * mol_mass))
        if not np.isfinite(thermal_speed) or thermal_speed <= 0.0:
            return tau
        tau_epstein = rho_p * diameter / (2.0 * rho_g * thermal_speed)
        if not np.isfinite(tau_epstein) or tau_epstein <= 0.0:
            return tau
        return max(float(min_tau_p_s), float(tau_epstein))
    diameter = max(float(particle_diameter_m), 0.0)
    rho_g = max(float(gas_density_kgm3), 0.0)
    mu = max(float(gas_mu_pas), 1.0e-30)
    re = rho_g * diameter * max(float(slip_speed), 0.0) / mu
    correction = schiller_naumann_drag_correction(re)
    return max(float(min_tau_p_s), tau / max(correction, 1.0e-30))


@njit(cache=True)
def compute_substep_count(dt, tau_eff, adaptive_enabled, adaptive_tau_ratio, adaptive_max_splits):
    if adaptive_enabled == 0:
        return 1
    if not np.isfinite(adaptive_tau_ratio) or adaptive_tau_ratio <= 0.0:
        return 1
    if not np.isfinite(tau_eff) or tau_eff <= 0.0:
        return 1
    if dt <= 0.0:
        return 1
    target_dt = tau_eff * adaptive_tau_ratio
    if target_dt <= 0.0:
        return 1
    n_substeps = int(np.ceil(dt / target_dt))
    if n_substeps < 1:
        n_substeps = 1
    max_substeps = 1
    splits = int(max(0, adaptive_max_splits))
    for _ in range(splits):
        max_substeps *= 2
    if n_substeps > max_substeps:
        n_substeps = max_substeps
    return n_substeps


@njit(cache=True)
def advance_component(v0, target, body_accel, tau_eff, dt, integrator_mode):
    decay = np.exp(-dt / tau_eff)
    if integrator_mode == INTEGRATOR_ETD or integrator_mode == INTEGRATOR_ETD2:
        c = target + body_accel * tau_eff
        v1 = c + (v0 - c) * decay
        x_delta = c * dt + (v0 - c) * tau_eff * (1.0 - decay)
        return x_delta, v1
    # Backward-compatible drag_relaxation update.
    v1 = target + (v0 - target) * decay + body_accel * dt
    x_delta = v1 * dt
    return x_delta, v1


@njit(cache=True)
def advance_state_2d_drag_relaxation(x0, y0, vx0, vy0, targetx, targety, bax, bay, tau_eff, dt):
    dx, vx1 = advance_component(vx0, targetx, bax, tau_eff, dt, INTEGRATOR_DRAG_RELAXATION)
    dy, vy1 = advance_component(vy0, targety, bay, tau_eff, dt, INTEGRATOR_DRAG_RELAXATION)
    return x0 + dx, y0 + dy, vx1, vy1


@njit(cache=True)
def advance_state_2d_etd(x0, y0, vx0, vy0, targetx, targety, bax, bay, tau_eff, dt):
    dx, vx1 = advance_component(vx0, targetx, bax, tau_eff, dt, INTEGRATOR_ETD)
    dy, vy1 = advance_component(vy0, targety, bay, tau_eff, dt, INTEGRATOR_ETD)
    return x0 + dx, y0 + dy, vx1, vy1


@njit(cache=True)
def advance_state_2d(x0, y0, vx0, vy0, targetx, targety, bax, bay, tau_eff, dt, integrator_mode):
    if integrator_mode == INTEGRATOR_DRAG_RELAXATION:
        return advance_state_2d_drag_relaxation(x0, y0, vx0, vy0, targetx, targety, bax, bay, tau_eff, dt)
    if integrator_mode == INTEGRATOR_ETD or integrator_mode == INTEGRATOR_ETD2:
        return advance_state_2d_etd(x0, y0, vx0, vy0, targetx, targety, bax, bay, tau_eff, dt)
    dx, vx1 = advance_component(vx0, targetx, bax, tau_eff, dt, integrator_mode)
    dy, vy1 = advance_component(vy0, targety, bay, tau_eff, dt, integrator_mode)
    return x0 + dx, y0 + dy, vx1, vy1


@njit(cache=True)
def advance_state_3d_drag_relaxation(x0, y0, z0, vx0, vy0, vz0, targetx, targety, targetz, bax, bay, baz, tau_eff, dt):
    dx, vx1 = advance_component(vx0, targetx, bax, tau_eff, dt, INTEGRATOR_DRAG_RELAXATION)
    dy, vy1 = advance_component(vy0, targety, bay, tau_eff, dt, INTEGRATOR_DRAG_RELAXATION)
    dz, vz1 = advance_component(vz0, targetz, baz, tau_eff, dt, INTEGRATOR_DRAG_RELAXATION)
    return x0 + dx, y0 + dy, z0 + dz, vx1, vy1, vz1


@njit(cache=True)
def advance_state_3d_etd(x0, y0, z0, vx0, vy0, vz0, targetx, targety, targetz, bax, bay, baz, tau_eff, dt):
    dx, vx1 = advance_component(vx0, targetx, bax, tau_eff, dt, INTEGRATOR_ETD)
    dy, vy1 = advance_component(vy0, targety, bay, tau_eff, dt, INTEGRATOR_ETD)
    dz, vz1 = advance_component(vz0, targetz, baz, tau_eff, dt, INTEGRATOR_ETD)
    return x0 + dx, y0 + dy, z0 + dz, vx1, vy1, vz1


@njit(cache=True)
def advance_state_3d(x0, y0, z0, vx0, vy0, vz0, targetx, targety, targetz, bax, bay, baz, tau_eff, dt, integrator_mode):
    if integrator_mode == INTEGRATOR_DRAG_RELAXATION:
        return advance_state_3d_drag_relaxation(x0, y0, z0, vx0, vy0, vz0, targetx, targety, targetz, bax, bay, baz, tau_eff, dt)
    if integrator_mode == INTEGRATOR_ETD or integrator_mode == INTEGRATOR_ETD2:
        return advance_state_3d_etd(x0, y0, z0, vx0, vy0, vz0, targetx, targety, targetz, bax, bay, baz, tau_eff, dt)
    dx, vx1 = advance_component(vx0, targetx, bax, tau_eff, dt, integrator_mode)
    dy, vy1 = advance_component(vy0, targety, bay, tau_eff, dt, integrator_mode)
    dz, vz1 = advance_component(vz0, targetz, baz, tau_eff, dt, integrator_mode)
    return x0 + dx, y0 + dy, z0 + dz, vx1, vy1, vz1
