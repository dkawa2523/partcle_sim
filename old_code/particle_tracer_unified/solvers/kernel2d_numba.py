"""Regular-grid 2D sampling leaves for the canonical motion kernel."""

from __future__ import annotations

import numpy as np
from numba import njit

from .drag_models import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    effective_tau_from_drag_model,
)
from .integrator_common import (
    compose_stage_acceleration_2d,
)
from .kernel_shared_numba import locate_axis, mask_bilinear_status
from .motion_kernel_numba import (
    _axisymmetric_rz_chart_radius,
    advance_etd2_batch_inplace,
)


@njit(cache=True)
def _sample_bilinear(arr2d, xs, ys, x, y):
    ix0, ix1, ax = locate_axis(xs, x)
    iy0, iy1, ay = locate_axis(ys, y)
    c00 = arr2d[ix0, iy0]
    c10 = arr2d[ix1, iy0]
    c01 = arr2d[ix0, iy1]
    c11 = arr2d[ix1, iy1]
    c0 = c00 * (1.0 - ax) + c10 * ax
    c1 = c01 * (1.0 - ax) + c11 * ax
    return c0 * (1.0 - ay) + c1 * ay


@njit(cache=True)
def _sample_time_bilinear(arr, times, xs, ys, t, x, y):
    nt = times.size
    if nt <= 1 or t <= times[0]:
        return _sample_bilinear(arr[0], xs, ys, x, y)
    if t >= times[nt - 1]:
        return _sample_bilinear(arr[nt - 1], xs, ys, x, y)
    hi = np.searchsorted(times, t)
    lo = hi - 1
    denominator = times[hi] - times[lo]
    alpha = 0.0 if denominator == 0.0 else (t - times[lo]) / denominator
    value_lo = _sample_bilinear(arr[lo], xs, ys, x, y)
    value_hi = _sample_bilinear(arr[hi], xs, ys, x, y)
    return value_lo * (1.0 - alpha) + value_hi * alpha


@njit(cache=True)
def _regular_2d_stage(
    particle_index,
    time_s,
    x,
    y,
    _z,
    vx,
    vy,
    _vz,
    tau_stokes,
    particle_diameter,
    particle_density,
    particle_mass,
    gas_molecular_mass_kg,
    drag_model_mode,
    epstein_accommodation_delta,
    body_ax,
    body_ay,
    _fallback_density,
    _fallback_mu,
    _fallback_temperature,
    xs,
    ys,
    times,
    ux,
    uy,
    extra_accel_x,
    extra_accel_y,
    gravity_buoyancy_enabled,
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    valid_mask,
    axisymmetric_rz=0,
):
    sample_x, chart_sign = _axisymmetric_rz_chart_radius(x, axisymmetric_rz)
    flow_x = _sample_time_bilinear(ux, times, xs, ys, time_s, sample_x, y)
    flow_y = _sample_time_bilinear(uy, times, xs, ys, time_s, sample_x, y)
    rho_g = _sample_time_bilinear(gas_density_grid, times, xs, ys, time_s, sample_x, y)
    mu_g = _sample_time_bilinear(gas_mu_grid, times, xs, ys, time_s, sample_x, y)
    temperature = _sample_time_bilinear(
        gas_temperature_grid, times, xs, ys, time_s, sample_x, y
    )
    accel_x, accel_y = compose_stage_acceleration_2d(
        body_ax,
        body_ay,
        extra_accel_x[particle_index],
        extra_accel_y[particle_index],
        rho_g,
        particle_density,
        gravity_buoyancy_enabled,
        1.0,
    )
    flow_x *= chart_sign
    accel_x *= chart_sign
    slip = np.sqrt((vx - flow_x) ** 2 + (vy - flow_y) ** 2)
    tau = effective_tau_from_drag_model(
        tau_stokes,
        slip,
        particle_diameter,
        rho_g,
        mu_g,
        drag_model_mode,
        particle_mass,
        temperature,
        gas_molecular_mass_kg,
        epstein_accommodation_delta,
    )
    status = mask_bilinear_status(valid_mask, xs, ys, sample_x, y)
    return flow_x, flow_y, 0.0, accel_x, accel_y, 0.0, tau, status


@njit(cache=True)
def _regular_2d_support(
    x,
    y,
    _z,
    _epstein_accommodation_delta,
    body_ax,
    body_ay,
    _fallback_density,
    _fallback_mu,
    _fallback_temperature,
    xs,
    ys,
    times,
    ux,
    uy,
    extra_accel_x,
    extra_accel_y,
    gravity_buoyancy_enabled,
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    valid_mask,
    axisymmetric_rz=0,
):
    sample_x, _chart_sign = _axisymmetric_rz_chart_radius(x, axisymmetric_rz)
    return mask_bilinear_status(valid_mask, xs, ys, sample_x, y)


def trace_regular_2d_batch_inplace(
    x,
    v,
    active,
    tau_p,
    particle_diameter,
    particle_density,
    particle_mass,
    t_end,
    duration,
    body_ax,
    body_ay,
    gas_density_kgm3,
    gas_mu_pas,
    gas_temperature_K,
    gas_molecular_mass_kg,
    drag_model_mode,
    adaptive_substep_enabled,
    adaptive_substep_max_splits,
    xs,
    ys,
    times,
    ux,
    uy,
    extra_accel_x_particle,
    extra_accel_y_particle,
    gravity_buoyancy_enabled,
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    valid_mask,
    x_end,
    v_end,
    x_mid,
    substep_counts,
    mask_status_flags,
    local_error_resolved,
    axisymmetric_rz=0,
):
    advance_etd2_batch_inplace(
        _regular_2d_stage,
        _regular_2d_support,
        2,
        x,
        v,
        active,
        tau_p,
        particle_diameter,
        particle_density,
        particle_mass,
        t_end,
        duration,
        gas_molecular_mass_kg,
        drag_model_mode,
        adaptive_substep_enabled,
        adaptive_substep_max_splits,
        x_end,
        v_end,
        x_mid,
        substep_counts,
        mask_status_flags,
        local_error_resolved,
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
        body_ax,
        body_ay,
        gas_density_kgm3,
        gas_mu_pas,
        gas_temperature_K,
        xs,
        ys,
        times,
        ux,
        uy,
        extra_accel_x_particle,
        extra_accel_y_particle,
        gravity_buoyancy_enabled,
        gas_density_grid,
        gas_mu_grid,
        gas_temperature_grid,
        valid_mask,
        int(axisymmetric_rz),
    )


__all__ = ("trace_regular_2d_batch_inplace",)
