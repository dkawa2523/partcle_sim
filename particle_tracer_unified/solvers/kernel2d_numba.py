from __future__ import annotations

import numpy as np
from numba import njit

from ..core.field_sampling import VALID_MASK_STATUS_CLEAN
from .integrator_common import INTEGRATOR_ETD2, advance_state_2d, advance_state_2d_etd, compute_substep_count
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
def advance_particles_2d_inplace(
    x,
    v,
    active,
    tau_p,
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
    integrator_mode,
    adaptive_substep_enabled,
    adaptive_substep_tau_ratio,
    adaptive_substep_max_splits,
    xs,
    ys,
    times,
    ux,
    uy,
    valid_mask,
    core_valid_mask,
    x_trial,
    v_trial,
    x_mid_trial,
    substep_counts,
    mask_status_flags,
    extension_band_sample_flags,
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
            extension_band_sample_flags[i] = False
            continue
        tau_eff = tau_p[i] * global_drag_tau_scale * max(drag_tau_scale_particle[i], 1e-6)
        if tau_eff < min_tau_p_s:
            tau_eff = min_tau_p_s
        bax = body_ax * global_body_accel_scale * body_accel_scale_particle[i]
        bay = body_ay * global_body_accel_scale * body_accel_scale_particle[i]
        n_substeps = compute_substep_count(
            dt,
            tau_eff,
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
        extension_band_sampled = False
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
                if mask_bilinear_point_valid(valid_mask, xs, ys, xn, yn) and (not mask_bilinear_point_valid(core_valid_mask, xs, ys, xn, yn)):
                    extension_band_sampled = True
                flowx0 = _sample_time_bilinear(ux, times, xs, ys, t_sub_start, xn, yn)
                flowy0 = _sample_time_bilinear(uy, times, xs, ys, t_sub_start, xn, yn)
                targetx0 = global_flow_scale * flow_scale_particle[i] * flowx0
                targety0 = global_flow_scale * flow_scale_particle[i] * flowy0
                xh, yh, _vxh, _vyh = advance_state_2d_etd(
                    xn,
                    yn,
                    vxn,
                    vyn,
                    targetx0,
                    targety0,
                    bax,
                    bay,
                    tau_eff,
                    0.5 * dt_sub,
                )
                t_mid = t_sub_start + 0.5 * dt_sub
                status = mask_bilinear_status(valid_mask, xs, ys, xh, yh)
                if status > mask_status:
                    mask_status = status
                if mask_bilinear_point_valid(valid_mask, xs, ys, xh, yh) and (not mask_bilinear_point_valid(core_valid_mask, xs, ys, xh, yh)):
                    extension_band_sampled = True
                flowx_mid = _sample_time_bilinear(ux, times, xs, ys, t_mid, xh, yh)
                flowy_mid = _sample_time_bilinear(uy, times, xs, ys, t_mid, xh, yh)
                targetx_mid = global_flow_scale * flow_scale_particle[i] * flowx_mid
                targety_mid = global_flow_scale * flow_scale_particle[i] * flowy_mid
                xn, yn, vxn, vyn = advance_state_2d_etd(
                    xn,
                    yn,
                    vxn,
                    vyn,
                    targetx_mid,
                    targety_mid,
                    bax,
                    bay,
                    tau_eff,
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
                            targetx0_mid = global_flow_scale * flow_scale_particle[i] * flowx0_mid
                            targety0_mid = global_flow_scale * flow_scale_particle[i] * flowy0_mid
                            xh_mid, yh_mid, _vxh_mid, _vyh_mid = advance_state_2d_etd(
                                x0,
                                y0,
                                vx0,
                                vy0,
                                targetx0_mid,
                                targety0_mid,
                                bax,
                                bay,
                                tau_eff,
                                0.5 * dt_mid,
                            )
                            t_mid_eval = t_sub_start + 0.5 * dt_mid
                            status = mask_bilinear_status(valid_mask, xs, ys, xh_mid, yh_mid)
                            if status > mask_status:
                                mask_status = status
                            if mask_bilinear_point_valid(valid_mask, xs, ys, xh_mid, yh_mid) and (not mask_bilinear_point_valid(core_valid_mask, xs, ys, xh_mid, yh_mid)):
                                extension_band_sampled = True
                            flowx_mid2 = _sample_time_bilinear(ux, times, xs, ys, t_mid_eval, xh_mid, yh_mid)
                            flowy_mid2 = _sample_time_bilinear(uy, times, xs, ys, t_mid_eval, xh_mid, yh_mid)
                            targetx_mid2 = global_flow_scale * flow_scale_particle[i] * flowx_mid2
                            targety_mid2 = global_flow_scale * flow_scale_particle[i] * flowy_mid2
                            xmid, ymid, _vxmid, _vymid = advance_state_2d_etd(
                                x0,
                                y0,
                                vx0,
                                vy0,
                                targetx_mid2,
                                targety_mid2,
                                bax,
                                bay,
                                tau_eff,
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
                if mask_bilinear_point_valid(valid_mask, xs, ys, xn, yn) and (not mask_bilinear_point_valid(core_valid_mask, xs, ys, xn, yn)):
                    extension_band_sampled = True
                flowx = _sample_time_bilinear(ux, times, xs, ys, t_eval, xn, yn)
                flowy = _sample_time_bilinear(uy, times, xs, ys, t_eval, xn, yn)
                targetx = global_flow_scale * flow_scale_particle[i] * flowx
                targety = global_flow_scale * flow_scale_particle[i] * flowy
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
        extension_band_sample_flags[i] = extension_band_sampled
