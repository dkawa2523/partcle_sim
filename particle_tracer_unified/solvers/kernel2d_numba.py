from __future__ import annotations

import numpy as np
from numba import njit

from ..core.field_sampling import VALID_MASK_STATUS_CLEAN
from .integrator_common import (
    INTEGRATOR_ETD2,
    advance_state_2d,
    advance_state_2d_etd,
    compute_substep_count,
    effective_tau_from_slip_speed,
)
from .kernel_shared_numba import (
    locate_axis,
    mask_bilinear_point_valid,
    mask_bilinear_status,
    midpoint_local_dt,
    should_capture_midpoint,
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
    if nt <= 1:
        return _sample_bilinear(arr[0], xs, ys, x, y)
    if t <= times[0]:
        return _sample_bilinear(arr[0], xs, ys, x, y)
    if t >= times[nt - 1]:
        return _sample_bilinear(arr[nt - 1], xs, ys, x, y)
    j = np.searchsorted(times, t)
    lo = j - 1
    hi = j
    denom = times[hi] - times[lo]
    a = 0.0 if abs(denom) <= 1e-30 else (t - times[lo]) / denom
    v0 = _sample_bilinear(arr[lo], xs, ys, x, y)
    v1 = _sample_bilinear(arr[hi], xs, ys, x, y)
    return v0 * (1.0 - a) + v1 * a


@njit(cache=True)
def _sample_electric_accel_2d(electric_x, electric_y, times, xs, ys, t, x, y, q_over_m, electric_enabled):
    ax = 0.0
    ay = 0.0
    if electric_enabled != 0:
        ex = _sample_time_bilinear(electric_x, times, xs, ys, t, x, y)
        ey = _sample_time_bilinear(electric_y, times, xs, ys, t, x, y)
        ax = ax + q_over_m * ex
        ay = ay + q_over_m * ey
    return ax, ay


@njit(cache=True)
def advance_particles_2d_inplace(
    x,
    v,
    active,
    tau_p,
    particle_diameter,
    particle_density,
    flow_scale_particle,
    drag_tau_scale_particle,
    body_accel_scale_particle,
    t,
    dt,
    global_flow_scale,
    global_drag_tau_scale,
    global_body_accel_scale,
    body_ax,
    body_ay,
    min_tau_p_s,
    gas_density_kgm3,
    gas_mu_pas,
    gas_temperature_K,
    gas_molecular_mass_kg,
    drag_model_mode,
    integrator_mode,
    adaptive_substep_enabled,
    adaptive_substep_tau_ratio,
    adaptive_substep_max_splits,
    xs,
    ys,
    times,
    ux,
    uy,
    electric_q_over_m_particle,
    electric_x,
    electric_y,
    dynamic_electric_enabled,
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    valid_mask,
    core_valid_mask,
    x_trial,
    v_trial,
    x_mid_trial,
    substep_counts,
    mask_status_flags,
):
    for i in range(x.shape[0]):
        if not active[i]:
            x_trial[i, 0] = x[i, 0]
            x_trial[i, 1] = x[i, 1]
            v_trial[i, 0] = v[i, 0]
            v_trial[i, 1] = v[i, 1]
            x_mid_trial[i, 0] = x[i, 0]
            x_mid_trial[i, 1] = x[i, 1]
            substep_counts[i] = 1
            mask_status_flags[i] = VALID_MASK_STATUS_CLEAN
            continue
        tau_stokes = tau_p[i] * global_drag_tau_scale * max(drag_tau_scale_particle[i], 1e-6)
        if tau_stokes < min_tau_p_s:
            tau_stokes = min_tau_p_s
        body_scale = global_body_accel_scale * body_accel_scale_particle[i]
        q_over_m_i = 0.0
        if dynamic_electric_enabled != 0:
            q_over_m_i = electric_q_over_m_particle[i]
        bax_base = body_ax * body_scale
        bay_base = body_ay * body_scale
        n_substeps = compute_substep_count(
            dt,
            tau_stokes,
            adaptive_substep_enabled,
            adaptive_substep_tau_ratio,
            adaptive_substep_max_splits,
        )
        substep_counts[i] = n_substeps
        dt_sub = dt / float(n_substeps)
        t_start = t - dt
        xn = x[i, 0]
        yn = x[i, 1]
        vxn = v[i, 0]
        vyn = v[i, 1]
        mask_status = VALID_MASK_STATUS_CLEAN
        if integrator_mode == INTEGRATOR_ETD2:
            half_dt = 0.5 * dt
            elapsed = 0.0
            has_mid = False
            xmid = xn
            ymid = yn
            for sub_idx in range(n_substeps):
                t_sub_start = t_start + float(sub_idx) * dt_sub
                x0 = xn
                y0 = yn
                vx0 = vxn
                vy0 = vyn
                status = mask_bilinear_status(valid_mask, xs, ys, xn, yn)
                if status > mask_status:
                    mask_status = status
                flowx0 = _sample_time_bilinear(ux, times, xs, ys, t_sub_start, xn, yn)
                flowy0 = _sample_time_bilinear(uy, times, xs, ys, t_sub_start, xn, yn)
                accx0, accy0 = _sample_electric_accel_2d(
                    electric_x, electric_y, times, xs, ys,
                    t_sub_start, xn, yn, q_over_m_i, dynamic_electric_enabled,
                )
                rho_g0 = _sample_time_bilinear(gas_density_grid, times, xs, ys, t_sub_start, xn, yn)
                mu_g0 = _sample_time_bilinear(gas_mu_grid, times, xs, ys, t_sub_start, xn, yn)
                temp_g0 = _sample_time_bilinear(gas_temperature_grid, times, xs, ys, t_sub_start, xn, yn)
                if not np.isfinite(rho_g0) or rho_g0 <= 0.0:
                    rho_g0 = gas_density_kgm3
                if not np.isfinite(mu_g0) or mu_g0 <= 0.0:
                    mu_g0 = gas_mu_pas
                if not np.isfinite(temp_g0) or temp_g0 <= 0.0:
                    temp_g0 = gas_temperature_K
                bax0 = bax_base + body_scale * accx0
                bay0 = bay_base + body_scale * accy0
                targetx0 = global_flow_scale * flow_scale_particle[i] * flowx0
                targety0 = global_flow_scale * flow_scale_particle[i] * flowy0
                slip0 = np.sqrt((vxn - targetx0) * (vxn - targetx0) + (vyn - targety0) * (vyn - targety0))
                tau_eff0 = effective_tau_from_slip_speed(
                    tau_stokes,
                    slip0,
                    particle_diameter[i],
                    rho_g0,
                    mu_g0,
                    drag_model_mode,
                    min_tau_p_s,
                    particle_density[i],
                    temp_g0,
                    gas_molecular_mass_kg,
                )
                xh, yh, _vxh, _vyh = advance_state_2d_etd(
                    xn,
                    yn,
                    vxn,
                    vyn,
                    targetx0,
                    targety0,
                    bax0,
                    bay0,
                    tau_eff0,
                    0.5 * dt_sub,
                )
                t_mid = t_sub_start + 0.5 * dt_sub
                status = mask_bilinear_status(valid_mask, xs, ys, xh, yh)
                if status > mask_status:
                    mask_status = status
                flowx_mid = _sample_time_bilinear(ux, times, xs, ys, t_mid, xh, yh)
                flowy_mid = _sample_time_bilinear(uy, times, xs, ys, t_mid, xh, yh)
                accx_mid, accy_mid = _sample_electric_accel_2d(
                    electric_x, electric_y, times, xs, ys,
                    t_mid, xh, yh, q_over_m_i, dynamic_electric_enabled,
                )
                rho_g_mid = _sample_time_bilinear(gas_density_grid, times, xs, ys, t_mid, xh, yh)
                mu_g_mid = _sample_time_bilinear(gas_mu_grid, times, xs, ys, t_mid, xh, yh)
                temp_g_mid = _sample_time_bilinear(gas_temperature_grid, times, xs, ys, t_mid, xh, yh)
                if not np.isfinite(rho_g_mid) or rho_g_mid <= 0.0:
                    rho_g_mid = gas_density_kgm3
                if not np.isfinite(mu_g_mid) or mu_g_mid <= 0.0:
                    mu_g_mid = gas_mu_pas
                if not np.isfinite(temp_g_mid) or temp_g_mid <= 0.0:
                    temp_g_mid = gas_temperature_K
                bax_mid = bax_base + body_scale * accx_mid
                bay_mid = bay_base + body_scale * accy_mid
                targetx_mid = global_flow_scale * flow_scale_particle[i] * flowx_mid
                targety_mid = global_flow_scale * flow_scale_particle[i] * flowy_mid
                slip_mid = np.sqrt((_vxh - targetx_mid) * (_vxh - targetx_mid) + (_vyh - targety_mid) * (_vyh - targety_mid))
                tau_eff_mid = effective_tau_from_slip_speed(
                    tau_stokes,
                    slip_mid,
                    particle_diameter[i],
                    rho_g_mid,
                    mu_g_mid,
                    drag_model_mode,
                    min_tau_p_s,
                    particle_density[i],
                    temp_g_mid,
                    gas_molecular_mass_kg,
                )
                xn, yn, vxn, vyn = advance_state_2d_etd(
                    xn,
                    yn,
                    vxn,
                    vyn,
                    targetx_mid,
                    targety_mid,
                    bax_mid,
                    bay_mid,
                    tau_eff_mid,
                    dt_sub,
                )
                if not has_mid:
                    elapsed_next = elapsed + dt_sub
                    if should_capture_midpoint(has_mid, elapsed, dt_sub, half_dt):
                        dt_mid = midpoint_local_dt(elapsed, dt_sub, half_dt)
                        if dt_mid <= 1.0e-15:
                            xmid = x0
                            ymid = y0
                        elif dt_mid >= dt_sub - 1.0e-15:
                            xmid = xn
                            ymid = yn
                        else:
                            flowx0_mid = _sample_time_bilinear(ux, times, xs, ys, t_sub_start, x0, y0)
                            flowy0_mid = _sample_time_bilinear(uy, times, xs, ys, t_sub_start, x0, y0)
                            accx0_mid, accy0_mid = _sample_electric_accel_2d(
                                electric_x, electric_y, times, xs, ys,
                                t_sub_start, x0, y0, q_over_m_i, dynamic_electric_enabled,
                            )
                            rho_g0_mid = _sample_time_bilinear(gas_density_grid, times, xs, ys, t_sub_start, x0, y0)
                            mu_g0_mid = _sample_time_bilinear(gas_mu_grid, times, xs, ys, t_sub_start, x0, y0)
                            temp_g0_mid = _sample_time_bilinear(gas_temperature_grid, times, xs, ys, t_sub_start, x0, y0)
                            if not np.isfinite(rho_g0_mid) or rho_g0_mid <= 0.0:
                                rho_g0_mid = gas_density_kgm3
                            if not np.isfinite(mu_g0_mid) or mu_g0_mid <= 0.0:
                                mu_g0_mid = gas_mu_pas
                            if not np.isfinite(temp_g0_mid) or temp_g0_mid <= 0.0:
                                temp_g0_mid = gas_temperature_K
                            bax0_mid = bax_base + body_scale * accx0_mid
                            bay0_mid = bay_base + body_scale * accy0_mid
                            targetx0_mid = global_flow_scale * flow_scale_particle[i] * flowx0_mid
                            targety0_mid = global_flow_scale * flow_scale_particle[i] * flowy0_mid
                            slip0_mid = np.sqrt((vx0 - targetx0_mid) * (vx0 - targetx0_mid) + (vy0 - targety0_mid) * (vy0 - targety0_mid))
                            tau_eff0_mid = effective_tau_from_slip_speed(
                                tau_stokes,
                                slip0_mid,
                                particle_diameter[i],
                                rho_g0_mid,
                                mu_g0_mid,
                                drag_model_mode,
                                min_tau_p_s,
                                particle_density[i],
                                temp_g0_mid,
                                gas_molecular_mass_kg,
                            )
                            xh_mid, yh_mid, _vxh_mid, _vyh_mid = advance_state_2d_etd(
                                x0,
                                y0,
                                vx0,
                                vy0,
                                targetx0_mid,
                                targety0_mid,
                                bax0_mid,
                                bay0_mid,
                                tau_eff0_mid,
                                0.5 * dt_mid,
                            )
                            t_mid_eval = t_sub_start + 0.5 * dt_mid
                            status = mask_bilinear_status(valid_mask, xs, ys, xh_mid, yh_mid)
                            if status > mask_status:
                                mask_status = status
                            flowx_mid2 = _sample_time_bilinear(ux, times, xs, ys, t_mid_eval, xh_mid, yh_mid)
                            flowy_mid2 = _sample_time_bilinear(uy, times, xs, ys, t_mid_eval, xh_mid, yh_mid)
                            accx_mid2, accy_mid2 = _sample_electric_accel_2d(
                                electric_x, electric_y, times, xs, ys,
                                t_mid_eval, xh_mid, yh_mid, q_over_m_i, dynamic_electric_enabled,
                            )
                            rho_g_mid2 = _sample_time_bilinear(gas_density_grid, times, xs, ys, t_mid_eval, xh_mid, yh_mid)
                            mu_g_mid2 = _sample_time_bilinear(gas_mu_grid, times, xs, ys, t_mid_eval, xh_mid, yh_mid)
                            temp_g_mid2 = _sample_time_bilinear(gas_temperature_grid, times, xs, ys, t_mid_eval, xh_mid, yh_mid)
                            if not np.isfinite(rho_g_mid2) or rho_g_mid2 <= 0.0:
                                rho_g_mid2 = gas_density_kgm3
                            if not np.isfinite(mu_g_mid2) or mu_g_mid2 <= 0.0:
                                mu_g_mid2 = gas_mu_pas
                            if not np.isfinite(temp_g_mid2) or temp_g_mid2 <= 0.0:
                                temp_g_mid2 = gas_temperature_K
                            bax_mid2 = bax_base + body_scale * accx_mid2
                            bay_mid2 = bay_base + body_scale * accy_mid2
                            targetx_mid2 = global_flow_scale * flow_scale_particle[i] * flowx_mid2
                            targety_mid2 = global_flow_scale * flow_scale_particle[i] * flowy_mid2
                            slip_mid2 = np.sqrt((_vxh_mid - targetx_mid2) * (_vxh_mid - targetx_mid2) + (_vyh_mid - targety_mid2) * (_vyh_mid - targety_mid2))
                            tau_eff_mid2 = effective_tau_from_slip_speed(
                                tau_stokes,
                                slip_mid2,
                                particle_diameter[i],
                                rho_g_mid2,
                                mu_g_mid2,
                                drag_model_mode,
                                min_tau_p_s,
                                particle_density[i],
                                temp_g_mid2,
                                gas_molecular_mass_kg,
                            )
                            xmid, ymid, _vxmid, _vymid = advance_state_2d_etd(
                                x0,
                                y0,
                                vx0,
                                vy0,
                                targetx_mid2,
                                targety_mid2,
                                bax_mid2,
                                bay_mid2,
                                tau_eff_mid2,
                                dt_mid,
                            )
                        has_mid = True
                    elapsed = elapsed_next
            if not has_mid:
                xmid = xn
                ymid = yn
            x_mid_trial[i, 0] = xmid
            x_mid_trial[i, 1] = ymid
        else:
            for sub_idx in range(n_substeps):
                t_eval = t_start + (float(sub_idx) + 1.0) * dt_sub
                status = mask_bilinear_status(valid_mask, xs, ys, xn, yn)
                if status > mask_status:
                    mask_status = status
                flowx = _sample_time_bilinear(ux, times, xs, ys, t_eval, xn, yn)
                flowy = _sample_time_bilinear(uy, times, xs, ys, t_eval, xn, yn)
                accx, accy = _sample_electric_accel_2d(
                    electric_x, electric_y, times, xs, ys,
                    t_eval, xn, yn, q_over_m_i, dynamic_electric_enabled,
                )
                rho_g = _sample_time_bilinear(gas_density_grid, times, xs, ys, t_eval, xn, yn)
                mu_g = _sample_time_bilinear(gas_mu_grid, times, xs, ys, t_eval, xn, yn)
                temp_g = _sample_time_bilinear(gas_temperature_grid, times, xs, ys, t_eval, xn, yn)
                if not np.isfinite(rho_g) or rho_g <= 0.0:
                    rho_g = gas_density_kgm3
                if not np.isfinite(mu_g) or mu_g <= 0.0:
                    mu_g = gas_mu_pas
                if not np.isfinite(temp_g) or temp_g <= 0.0:
                    temp_g = gas_temperature_K
                bax = bax_base + body_scale * accx
                bay = bay_base + body_scale * accy
                targetx = global_flow_scale * flow_scale_particle[i] * flowx
                targety = global_flow_scale * flow_scale_particle[i] * flowy
                slip = np.sqrt((vxn - targetx) * (vxn - targetx) + (vyn - targety) * (vyn - targety))
                tau_eff = effective_tau_from_slip_speed(
                    tau_stokes,
                    slip,
                    particle_diameter[i],
                    rho_g,
                    mu_g,
                    drag_model_mode,
                    min_tau_p_s,
                    particle_density[i],
                    temp_g,
                    gas_molecular_mass_kg,
                )
                xn, yn, vxn, vyn = advance_state_2d(
                    xn,
                    yn,
                    vxn,
                    vyn,
                    targetx,
                    targety,
                    bax,
                    bay,
                    tau_eff,
                    dt_sub,
                    integrator_mode,
                )
            x_mid_trial[i, 0] = xn
            x_mid_trial[i, 1] = yn
        x_trial[i, 0] = xn
        x_trial[i, 1] = yn
        v_trial[i, 0] = vxn
        v_trial[i, 1] = vyn
        mask_status_flags[i] = mask_status
