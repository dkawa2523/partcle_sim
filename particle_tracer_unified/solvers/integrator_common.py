from __future__ import annotations

import numpy as np
from numba import njit

from ..core.integrator_registry import get_integrator_spec

INTEGRATOR_DRAG_RELAXATION = int(get_integrator_spec('drag_relaxation').mode)
INTEGRATOR_ETD = int(get_integrator_spec('etd').mode)
INTEGRATOR_ETD2 = int(get_integrator_spec('etd2').mode)


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
