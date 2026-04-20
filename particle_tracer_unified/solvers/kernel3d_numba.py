from __future__ import annotations

import numpy as np
from numba import njit

from ..core.field_sampling import VALID_MASK_STATUS_CLEAN
from .integrator_common import (
    INTEGRATOR_ETD2,
    advance_state_3d,
    advance_state_3d_etd,
    compute_substep_count,
    effective_tau_from_slip_speed,
)
from .kernel_shared_numba import (
    locate_axis,
    mask_trilinear_point_valid,
    mask_trilinear_status,
    midpoint_local_dt,
    should_capture_midpoint,
)


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
    if nt <= 1:
        return _sample_trilinear(arr[0], xs, ys, zs, x, y, z)
    if t <= times[0]:
        return _sample_trilinear(arr[0], xs, ys, zs, x, y, z)
    if t >= times[nt - 1]:
        return _sample_trilinear(arr[nt - 1], xs, ys, zs, x, y, z)
    j = np.searchsorted(times, t)
    lo = j - 1
    hi = j
    denom = times[hi] - times[lo]
    a = 0.0 if abs(denom) <= 1e-30 else (t - times[lo]) / denom
    v0 = _sample_trilinear(arr[lo], xs, ys, zs, x, y, z)
    v1 = _sample_trilinear(arr[hi], xs, ys, zs, x, y, z)
    return v0 * (1.0 - a) + v1 * a


@njit(cache=True)
def _sample_electric_accel_3d(electric_x, electric_y, electric_z, times, xs, ys, zs, t, x, y, z, q_over_m, electric_enabled):
    ax = 0.0
    ay = 0.0
    az = 0.0
    if electric_enabled != 0:
        ex = _sample_time_trilinear(electric_x, times, xs, ys, zs, t, x, y, z)
        ey = _sample_time_trilinear(electric_y, times, xs, ys, zs, t, x, y, z)
        ez = _sample_time_trilinear(electric_z, times, xs, ys, zs, t, x, y, z)
        ax = q_over_m * ex
        ay = q_over_m * ey
        az = q_over_m * ez
    return ax, ay, az


@njit(cache=True)
def advance_particles_3d_inplace(
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
    body_az,
    min_tau_p_s,
    gas_density_kgm3,
    gas_mu_pas,
    drag_model_mode,
    integrator_mode,
    adaptive_substep_enabled,
    adaptive_substep_tau_ratio,
    adaptive_substep_max_splits,
    xs,
    ys,
    zs,
    times,
    ux,
    uy,
    uz,
    electric_q_over_m_particle,
    electric_x,
    electric_y,
    electric_z,
    dynamic_electric_enabled,
    extra_accel_x_particle,
    extra_accel_y_particle,
    extra_accel_z_particle,
    gravity_buoyancy_enabled,
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
            for j in range(3):
                x_trial[i, j] = x[i, j]
                v_trial[i, j] = v[i, j]
                x_mid_trial[i, j] = x[i, j]
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
        body_x_scaled = body_ax * body_scale
        body_y_scaled = body_ay * body_scale
        body_z_scaled = body_az * body_scale
        gravity_factor = 1.0
        if gravity_buoyancy_enabled != 0 and particle_density[i] > 0.0:
            gravity_factor = 1.0 - gas_density_kgm3 / particle_density[i]
        bax_base = body_x_scaled * gravity_factor + extra_accel_x_particle[i]
        bay_base = body_y_scaled * gravity_factor + extra_accel_y_particle[i]
        baz_base = body_z_scaled * gravity_factor + extra_accel_z_particle[i]
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
        zn = x[i, 2]
        vxn = v[i, 0]
        vyn = v[i, 1]
        vzn = v[i, 2]
        mask_status = VALID_MASK_STATUS_CLEAN
        if integrator_mode == INTEGRATOR_ETD2:
            half_dt = 0.5 * dt
            elapsed = 0.0
            has_mid = False
            xmid = xn
            ymid = yn
            zmid = zn
            for sub_idx in range(n_substeps):
                t_sub_start = t_start + float(sub_idx) * dt_sub
                x0 = xn
                y0 = yn
                z0 = zn
                vx0 = vxn
                vy0 = vyn
                vz0 = vzn
                status = mask_trilinear_status(valid_mask, xs, ys, zs, xn, yn, zn)
                if status > mask_status:
                    mask_status = status
                flowx0 = _sample_time_trilinear(ux, times, xs, ys, zs, t_sub_start, xn, yn, zn)
                flowy0 = _sample_time_trilinear(uy, times, xs, ys, zs, t_sub_start, xn, yn, zn)
                flowz0 = _sample_time_trilinear(uz, times, xs, ys, zs, t_sub_start, xn, yn, zn)
                accx0, accy0, accz0 = _sample_electric_accel_3d(
                    electric_x, electric_y, electric_z, times, xs, ys, zs,
                    t_sub_start, xn, yn, zn, q_over_m_i, dynamic_electric_enabled,
                )
                bax0 = bax_base + body_scale * accx0
                bay0 = bay_base + body_scale * accy0
                baz0 = baz_base + body_scale * accz0
                targetx0 = global_flow_scale * flow_scale_particle[i] * flowx0
                targety0 = global_flow_scale * flow_scale_particle[i] * flowy0
                targetz0 = global_flow_scale * flow_scale_particle[i] * flowz0
                slip0 = np.sqrt(
                    (vxn - targetx0) * (vxn - targetx0)
                    + (vyn - targety0) * (vyn - targety0)
                    + (vzn - targetz0) * (vzn - targetz0)
                )
                tau_eff0 = effective_tau_from_slip_speed(
                    tau_stokes,
                    slip0,
                    particle_diameter[i],
                    gas_density_kgm3,
                    gas_mu_pas,
                    drag_model_mode,
                    min_tau_p_s,
                )
                xh, yh, zh, _vxh, _vyh, _vzh = advance_state_3d_etd(
                    xn,
                    yn,
                    zn,
                    vxn,
                    vyn,
                    vzn,
                    targetx0,
                    targety0,
                    targetz0,
                    bax0,
                    bay0,
                    baz0,
                    tau_eff0,
                    0.5 * dt_sub,
                )
                t_mid = t_sub_start + 0.5 * dt_sub
                status = mask_trilinear_status(valid_mask, xs, ys, zs, xh, yh, zh)
                if status > mask_status:
                    mask_status = status
                flowx_mid = _sample_time_trilinear(ux, times, xs, ys, zs, t_mid, xh, yh, zh)
                flowy_mid = _sample_time_trilinear(uy, times, xs, ys, zs, t_mid, xh, yh, zh)
                flowz_mid = _sample_time_trilinear(uz, times, xs, ys, zs, t_mid, xh, yh, zh)
                accx_mid, accy_mid, accz_mid = _sample_electric_accel_3d(
                    electric_x, electric_y, electric_z, times, xs, ys, zs,
                    t_mid, xh, yh, zh, q_over_m_i, dynamic_electric_enabled,
                )
                bax_mid = bax_base + body_scale * accx_mid
                bay_mid = bay_base + body_scale * accy_mid
                baz_mid = baz_base + body_scale * accz_mid
                targetx_mid = global_flow_scale * flow_scale_particle[i] * flowx_mid
                targety_mid = global_flow_scale * flow_scale_particle[i] * flowy_mid
                targetz_mid = global_flow_scale * flow_scale_particle[i] * flowz_mid
                slip_mid = np.sqrt(
                    (_vxh - targetx_mid) * (_vxh - targetx_mid)
                    + (_vyh - targety_mid) * (_vyh - targety_mid)
                    + (_vzh - targetz_mid) * (_vzh - targetz_mid)
                )
                tau_eff_mid = effective_tau_from_slip_speed(
                    tau_stokes,
                    slip_mid,
                    particle_diameter[i],
                    gas_density_kgm3,
                    gas_mu_pas,
                    drag_model_mode,
                    min_tau_p_s,
                )
                xn, yn, zn, vxn, vyn, vzn = advance_state_3d_etd(
                    xn,
                    yn,
                    zn,
                    vxn,
                    vyn,
                    vzn,
                    targetx_mid,
                    targety_mid,
                    targetz_mid,
                    bax_mid,
                    bay_mid,
                    baz_mid,
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
                            zmid = z0
                        elif dt_mid >= dt_sub - 1.0e-15:
                            xmid = xn
                            ymid = yn
                            zmid = zn
                        else:
                            flowx0_mid = _sample_time_trilinear(ux, times, xs, ys, zs, t_sub_start, x0, y0, z0)
                            flowy0_mid = _sample_time_trilinear(uy, times, xs, ys, zs, t_sub_start, x0, y0, z0)
                            flowz0_mid = _sample_time_trilinear(uz, times, xs, ys, zs, t_sub_start, x0, y0, z0)
                            accx0_mid, accy0_mid, accz0_mid = _sample_electric_accel_3d(
                                electric_x, electric_y, electric_z, times, xs, ys, zs,
                                t_sub_start, x0, y0, z0, q_over_m_i, dynamic_electric_enabled,
                            )
                            bax0_mid = bax_base + body_scale * accx0_mid
                            bay0_mid = bay_base + body_scale * accy0_mid
                            baz0_mid = baz_base + body_scale * accz0_mid
                            targetx0_mid = global_flow_scale * flow_scale_particle[i] * flowx0_mid
                            targety0_mid = global_flow_scale * flow_scale_particle[i] * flowy0_mid
                            targetz0_mid = global_flow_scale * flow_scale_particle[i] * flowz0_mid
                            slip0_mid = np.sqrt(
                                (vx0 - targetx0_mid) * (vx0 - targetx0_mid)
                                + (vy0 - targety0_mid) * (vy0 - targety0_mid)
                                + (vz0 - targetz0_mid) * (vz0 - targetz0_mid)
                            )
                            tau_eff0_mid = effective_tau_from_slip_speed(
                                tau_stokes,
                                slip0_mid,
                                particle_diameter[i],
                                gas_density_kgm3,
                                gas_mu_pas,
                                drag_model_mode,
                                min_tau_p_s,
                            )
                            xh_mid, yh_mid, zh_mid, _vxh_mid, _vyh_mid, _vzh_mid = advance_state_3d_etd(
                                x0,
                                y0,
                                z0,
                                vx0,
                                vy0,
                                vz0,
                                targetx0_mid,
                                targety0_mid,
                                targetz0_mid,
                                bax0_mid,
                                bay0_mid,
                                baz0_mid,
                                tau_eff0_mid,
                                0.5 * dt_mid,
                            )
                            t_mid_eval = t_sub_start + 0.5 * dt_mid
                            status = mask_trilinear_status(valid_mask, xs, ys, zs, xh_mid, yh_mid, zh_mid)
                            if status > mask_status:
                                mask_status = status
                            flowx_mid2 = _sample_time_trilinear(ux, times, xs, ys, zs, t_mid_eval, xh_mid, yh_mid, zh_mid)
                            flowy_mid2 = _sample_time_trilinear(uy, times, xs, ys, zs, t_mid_eval, xh_mid, yh_mid, zh_mid)
                            flowz_mid2 = _sample_time_trilinear(uz, times, xs, ys, zs, t_mid_eval, xh_mid, yh_mid, zh_mid)
                            accx_mid2, accy_mid2, accz_mid2 = _sample_electric_accel_3d(
                                electric_x, electric_y, electric_z, times, xs, ys, zs,
                                t_mid_eval, xh_mid, yh_mid, zh_mid, q_over_m_i, dynamic_electric_enabled,
                            )
                            bax_mid2 = bax_base + body_scale * accx_mid2
                            bay_mid2 = bay_base + body_scale * accy_mid2
                            baz_mid2 = baz_base + body_scale * accz_mid2
                            targetx_mid2 = global_flow_scale * flow_scale_particle[i] * flowx_mid2
                            targety_mid2 = global_flow_scale * flow_scale_particle[i] * flowy_mid2
                            targetz_mid2 = global_flow_scale * flow_scale_particle[i] * flowz_mid2
                            slip_mid2 = np.sqrt(
                                (_vxh_mid - targetx_mid2) * (_vxh_mid - targetx_mid2)
                                + (_vyh_mid - targety_mid2) * (_vyh_mid - targety_mid2)
                                + (_vzh_mid - targetz_mid2) * (_vzh_mid - targetz_mid2)
                            )
                            tau_eff_mid2 = effective_tau_from_slip_speed(
                                tau_stokes,
                                slip_mid2,
                                particle_diameter[i],
                                gas_density_kgm3,
                                gas_mu_pas,
                                drag_model_mode,
                                min_tau_p_s,
                            )
                            xmid, ymid, zmid, _vxmid, _vymid, _vzmid = advance_state_3d_etd(
                                x0,
                                y0,
                                z0,
                                vx0,
                                vy0,
                                vz0,
                                targetx_mid2,
                                targety_mid2,
                                targetz_mid2,
                                bax_mid2,
                                bay_mid2,
                                baz_mid2,
                                tau_eff_mid2,
                                dt_mid,
                            )
                        has_mid = True
                    elapsed = elapsed_next
            if not has_mid:
                xmid = xn
                ymid = yn
                zmid = zn
            x_mid_trial[i, 0] = xmid
            x_mid_trial[i, 1] = ymid
            x_mid_trial[i, 2] = zmid
        else:
            for sub_idx in range(n_substeps):
                t_eval = t_start + (float(sub_idx) + 1.0) * dt_sub
                status = mask_trilinear_status(valid_mask, xs, ys, zs, xn, yn, zn)
                if status > mask_status:
                    mask_status = status
                flowx = _sample_time_trilinear(ux, times, xs, ys, zs, t_eval, xn, yn, zn)
                flowy = _sample_time_trilinear(uy, times, xs, ys, zs, t_eval, xn, yn, zn)
                flowz = _sample_time_trilinear(uz, times, xs, ys, zs, t_eval, xn, yn, zn)
                accx, accy, accz = _sample_electric_accel_3d(
                    electric_x, electric_y, electric_z, times, xs, ys, zs,
                    t_eval, xn, yn, zn, q_over_m_i, dynamic_electric_enabled,
                )
                bax = bax_base + body_scale * accx
                bay = bay_base + body_scale * accy
                baz = baz_base + body_scale * accz
                targetx = global_flow_scale * flow_scale_particle[i] * flowx
                targety = global_flow_scale * flow_scale_particle[i] * flowy
                targetz = global_flow_scale * flow_scale_particle[i] * flowz
                slip = np.sqrt(
                    (vxn - targetx) * (vxn - targetx)
                    + (vyn - targety) * (vyn - targety)
                    + (vzn - targetz) * (vzn - targetz)
                )
                tau_eff = effective_tau_from_slip_speed(
                    tau_stokes,
                    slip,
                    particle_diameter[i],
                    gas_density_kgm3,
                    gas_mu_pas,
                    drag_model_mode,
                    min_tau_p_s,
                )
                xn, yn, zn, vxn, vyn, vzn = advance_state_3d(
                    xn,
                    yn,
                    zn,
                    vxn,
                    vyn,
                    vzn,
                    targetx,
                    targety,
                    targetz,
                    bax,
                    bay,
                    baz,
                    tau_eff,
                    dt_sub,
                    integrator_mode,
                )
            x_mid_trial[i, 0] = xn
            x_mid_trial[i, 1] = yn
            x_mid_trial[i, 2] = zn
        x_trial[i, 0] = xn
        x_trial[i, 1] = yn
        x_trial[i, 2] = zn
        v_trial[i, 0] = vxn
        v_trial[i, 1] = vyn
        v_trial[i, 2] = vzn
        mask_status_flags[i] = mask_status
