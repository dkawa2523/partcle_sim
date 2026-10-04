"""Regular-grid 3D sampling leaves for the canonical motion kernel."""

from __future__ import annotations

import numpy as np
from numba import njit

from .drag_models import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    effective_tau_from_drag_model,
)
from .integrator_common import (
    compose_stage_acceleration_3d,
)
from .kernel_shared_numba import locate_axis, mask_trilinear_status
from .motion_kernel_numba import advance_etd2_batch_inplace


@njit(cache=True)
def _sample_trilinear(arr3d, xs, ys, zs, x, y, z):
    ix0, ix1, ax = locate_axis(xs, x)
    iy0, iy1, ay = locate_axis(ys, y)
    iz0, iz1, az = locate_axis(zs, z)
    c000 = arr3d[ix0, iy0, iz0]
    c100 = arr3d[ix1, iy0, iz0]
    c010 = arr3d[ix0, iy1, iz0]
    c110 = arr3d[ix1, iy1, iz0]
    c001 = arr3d[ix0, iy0, iz1]
    c101 = arr3d[ix1, iy0, iz1]
    c011 = arr3d[ix0, iy1, iz1]
    c111 = arr3d[ix1, iy1, iz1]
    c00 = c000 * (1.0 - ax) + c100 * ax
    c10 = c010 * (1.0 - ax) + c110 * ax
    c01 = c001 * (1.0 - ax) + c101 * ax
    c11 = c011 * (1.0 - ax) + c111 * ax
    c0 = c00 * (1.0 - ay) + c10 * ay
    c1 = c01 * (1.0 - ay) + c11 * ay
    return c0 * (1.0 - az) + c1 * az


@njit(cache=True)
def _sample_time_trilinear(arr, times, xs, ys, zs, t, x, y, z):
    nt = times.size
    if nt <= 1 or t <= times[0]:
        return _sample_trilinear(arr[0], xs, ys, zs, x, y, z)
    if t >= times[nt - 1]:
        return _sample_trilinear(arr[nt - 1], xs, ys, zs, x, y, z)
    hi = np.searchsorted(times, t)
    lo = hi - 1
    denominator = times[hi] - times[lo]
    alpha = 0.0 if denominator == 0.0 else (t - times[lo]) / denominator
    value_lo = _sample_trilinear(arr[lo], xs, ys, zs, x, y, z)
    value_hi = _sample_trilinear(arr[hi], xs, ys, zs, x, y, z)
    return value_lo * (1.0 - alpha) + value_hi * alpha


@njit(cache=True)
def _regular_3d_stage(
    particle_index,
    time_s,
    x,
    y,
    z,
    vx,
    vy,
    vz,
    tau_stokes,
    particle_diameter,
    particle_density,
    particle_mass,
    gas_molecular_mass_kg,
    drag_model_mode,
    epstein_accommodation_delta,
    body_ax,
    body_ay,
    body_az,
    _fallback_density,
    _fallback_mu,
    _fallback_temperature,
    xs,
    ys,
    zs,
    times,
    ux,
    uy,
    uz,
    extra_accel_x,
    extra_accel_y,
    extra_accel_z,
    gravity_buoyancy_enabled,
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    valid_mask,
):
    flow_x = _sample_time_trilinear(ux, times, xs, ys, zs, time_s, x, y, z)
    flow_y = _sample_time_trilinear(uy, times, xs, ys, zs, time_s, x, y, z)
    flow_z = _sample_time_trilinear(uz, times, xs, ys, zs, time_s, x, y, z)
    rho_g = _sample_time_trilinear(gas_density_grid, times, xs, ys, zs, time_s, x, y, z)
    mu_g = _sample_time_trilinear(gas_mu_grid, times, xs, ys, zs, time_s, x, y, z)
    temperature = _sample_time_trilinear(
        gas_temperature_grid, times, xs, ys, zs, time_s, x, y, z
    )
    accel_x, accel_y, accel_z = compose_stage_acceleration_3d(
        body_ax,
        body_ay,
        body_az,
        extra_accel_x[particle_index],
        extra_accel_y[particle_index],
        extra_accel_z[particle_index],
        rho_g,
        particle_density,
        gravity_buoyancy_enabled,
        1.0,
    )
    slip = np.sqrt((vx - flow_x) ** 2 + (vy - flow_y) ** 2 + (vz - flow_z) ** 2)
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
    status = mask_trilinear_status(valid_mask, xs, ys, zs, x, y, z)
    return flow_x, flow_y, flow_z, accel_x, accel_y, accel_z, tau, status


@njit(cache=True)
def _regular_3d_support(
    x,
    y,
    z,
    _epstein_accommodation_delta,
    body_ax,
    body_ay,
    body_az,
    _fallback_density,
    _fallback_mu,
    _fallback_temperature,
    xs,
    ys,
    zs,
    times,
    ux,
    uy,
    uz,
    extra_accel_x,
    extra_accel_y,
    extra_accel_z,
    gravity_buoyancy_enabled,
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    valid_mask,
):
    return mask_trilinear_status(valid_mask, xs, ys, zs, x, y, z)


def trace_regular_3d_batch_inplace(
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
    body_az,
    gas_density_kgm3,
    gas_mu_pas,
    gas_temperature_K,
    gas_molecular_mass_kg,
    drag_model_mode,
    adaptive_substep_enabled,
    adaptive_substep_max_splits,
    xs,
    ys,
    zs,
    times,
    ux,
    uy,
    uz,
    extra_accel_x_particle,
    extra_accel_y_particle,
    extra_accel_z_particle,
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
):
    advance_etd2_batch_inplace(
        _regular_3d_stage,
        _regular_3d_support,
        3,
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
        body_az,
        gas_density_kgm3,
        gas_mu_pas,
        gas_temperature_K,
        xs,
        ys,
        zs,
        times,
        ux,
        uy,
        uz,
        extra_accel_x_particle,
        extra_accel_y_particle,
        extra_accel_z_particle,
        gravity_buoyancy_enabled,
        gas_density_grid,
        gas_mu_grid,
        gas_temperature_grid,
        valid_mask,
    )


__all__ = ("trace_regular_3d_batch_inplace",)
