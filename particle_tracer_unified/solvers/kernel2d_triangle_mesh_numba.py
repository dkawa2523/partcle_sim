from __future__ import annotations

import numpy as np
from numba import njit

from ..core.field_sampling import VALID_MASK_STATUS_CLEAN, VALID_MASK_STATUS_HARD_INVALID
from .integrator_common import INTEGRATOR_ETD2, advance_state_2d, advance_state_2d_etd, compute_substep_count
from .kernel_shared_numba import midpoint_local_dt, should_capture_midpoint


@njit(cache=True)
def _find_triangle_and_barycentric(
    vertices,
    triangles,
    accel_origin,
    accel_cell_size,
    accel_nx,
    accel_ny,
    accel_cell_offsets,
    accel_triangle_indices,
    support_tolerance,
    x,
    y,
):
    eps = support_tolerance
    xmin = accel_origin[0]
    ymin = accel_origin[1]
    xmax = xmin + accel_cell_size[0] * accel_nx
    ymax = ymin + accel_cell_size[1] * accel_ny
    if x < xmin - eps or x > xmax + eps or y < ymin - eps or y > ymax + eps:
        return -1, 0.0, 0.0, 0.0
    ix = int(np.floor((x - xmin) / accel_cell_size[0]))
    iy = int(np.floor((y - ymin) / accel_cell_size[1]))
    if ix < 0:
        ix = 0
    if iy < 0:
        iy = 0
    if ix >= accel_nx:
        ix = accel_nx - 1
    if iy >= accel_ny:
        iy = accel_ny - 1
    cell_id = ix * accel_ny + iy
    start = accel_cell_offsets[cell_id]
    stop = accel_cell_offsets[cell_id + 1]
    best_idx = -1
    best_alpha = 0.0
    best_beta = 0.0
    best_gamma = 0.0
    best_margin = -1.0e300
    for flat_idx in range(start, stop):
        tri_idx = accel_triangle_indices[flat_idx]
        i0 = triangles[tri_idx, 0]
        i1 = triangles[tri_idx, 1]
        i2 = triangles[tri_idx, 2]
        ax = vertices[i0, 0]
        ay = vertices[i0, 1]
        bx = vertices[i1, 0]
        by = vertices[i1, 1]
        cx = vertices[i2, 0]
        cy = vertices[i2, 1]
        v0x = bx - ax
        v0y = by - ay
        v1x = cx - ax
        v1y = cy - ay
        v2x = x - ax
        v2y = y - ay
        den = v0x * v1y - v0y * v1x
        if abs(den) <= 1.0e-30:
            continue
        beta = (v2x * v1y - v2y * v1x) / den
        gamma = (v0x * v2y - v0y * v2x) / den
        alpha = 1.0 - beta - gamma
        margin = min(alpha, beta, gamma)
        if margin < -eps:
            continue
        if margin > best_margin:
            best_margin = margin
            best_idx = tri_idx
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
    return best_idx, best_alpha, best_beta, best_gamma


@njit(cache=True)
def _sample_triangle_vertex_series(arr, times, triangles, tri_idx, alpha, beta, gamma, t):
    i0 = triangles[tri_idx, 0]
    i1 = triangles[tri_idx, 1]
    i2 = triangles[tri_idx, 2]
    if arr.ndim == 1:
        return alpha * arr[i0] + beta * arr[i1] + gamma * arr[i2]
    nt = times.size
    if nt <= 1 or arr.shape[0] <= 1:
        return alpha * arr[0, i0] + beta * arr[0, i1] + gamma * arr[0, i2]
    if t <= times[0]:
        return alpha * arr[0, i0] + beta * arr[0, i1] + gamma * arr[0, i2]
    if t >= times[nt - 1]:
        return alpha * arr[nt - 1, i0] + beta * arr[nt - 1, i1] + gamma * arr[nt - 1, i2]
    hi = np.searchsorted(times, t)
    lo = hi - 1
    denom = times[hi] - times[lo]
    a = 0.0 if abs(denom) <= 1.0e-30 else (t - times[lo]) / denom
    v0 = alpha * arr[lo, i0] + beta * arr[lo, i1] + gamma * arr[lo, i2]
    v1 = alpha * arr[hi, i0] + beta * arr[hi, i1] + gamma * arr[hi, i2]
    return v0 * (1.0 - a) + v1 * a


@njit(cache=True)
def _sample_triangle_mesh_flow(
    vertices,
    triangles,
    accel_origin,
    accel_cell_size,
    accel_nx,
    accel_ny,
    accel_cell_offsets,
    accel_triangle_indices,
    support_tolerance,
    times,
    ux,
    uy,
    t,
    x,
    y,
):
    tri_idx, alpha, beta, gamma = _find_triangle_and_barycentric(
        vertices,
        triangles,
        accel_origin,
        accel_cell_size,
        accel_nx,
        accel_ny,
        accel_cell_offsets,
        accel_triangle_indices,
        support_tolerance,
        x,
        y,
    )
    if tri_idx < 0:
        return 0.0, 0.0, VALID_MASK_STATUS_HARD_INVALID
    flowx = _sample_triangle_vertex_series(ux, times, triangles, tri_idx, alpha, beta, gamma, t)
    flowy = _sample_triangle_vertex_series(uy, times, triangles, tri_idx, alpha, beta, gamma, t)
    return flowx, flowy, VALID_MASK_STATUS_CLEAN


@njit(cache=True)
def advance_particles_2d_triangle_mesh_inplace(
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
    vertices,
    triangles,
    accel_origin,
    accel_cell_size,
    accel_nx,
    accel_ny,
    accel_cell_offsets,
    accel_triangle_indices,
    support_tolerance,
    times,
    ux,
    uy,
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
        tau_eff = tau_p[i] * global_drag_tau_scale * max(drag_tau_scale_particle[i], 1.0e-6)
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
                flowx0, flowy0, status = _sample_triangle_mesh_flow(
                    vertices, triangles, accel_origin, accel_cell_size, accel_nx, accel_ny,
                    accel_cell_offsets, accel_triangle_indices, support_tolerance, times, ux, uy, t_sub_start, xn, yn
                )
                if status > mask_status:
                    mask_status = status
                targetx0 = global_flow_scale * flow_scale_particle[i] * flowx0
                targety0 = global_flow_scale * flow_scale_particle[i] * flowy0
                xh, yh, _vxh, _vyh = advance_state_2d_etd(
                    xn, yn, vxn, vyn, targetx0, targety0, bax, bay, tau_eff, 0.5 * dt_sub
                )
                t_mid = t_sub_start + 0.5 * dt_sub
                flowx_mid, flowy_mid, status = _sample_triangle_mesh_flow(
                    vertices, triangles, accel_origin, accel_cell_size, accel_nx, accel_ny,
                    accel_cell_offsets, accel_triangle_indices, support_tolerance, times, ux, uy, t_mid, xh, yh
                )
                if status > mask_status:
                    mask_status = status
                targetx_mid = global_flow_scale * flow_scale_particle[i] * flowx_mid
                targety_mid = global_flow_scale * flow_scale_particle[i] * flowy_mid
                xn, yn, vxn, vyn = advance_state_2d_etd(
                    xn, yn, vxn, vyn, targetx_mid, targety_mid, bax, bay, tau_eff, dt_sub
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
                            flowx0_mid, flowy0_mid, status = _sample_triangle_mesh_flow(
                                vertices, triangles, accel_origin, accel_cell_size, accel_nx, accel_ny,
                                accel_cell_offsets, accel_triangle_indices, support_tolerance, times, ux, uy, t_sub_start, x0, y0
                            )
                            if status > mask_status:
                                mask_status = status
                            targetx0_mid = global_flow_scale * flow_scale_particle[i] * flowx0_mid
                            targety0_mid = global_flow_scale * flow_scale_particle[i] * flowy0_mid
                            xh_mid, yh_mid, _vxh_mid, _vyh_mid = advance_state_2d_etd(
                                x0, y0, vx0, vy0, targetx0_mid, targety0_mid, bax, bay, tau_eff, 0.5 * dt_mid
                            )
                            t_mid_eval = t_sub_start + 0.5 * dt_mid
                            flowx_mid2, flowy_mid2, status = _sample_triangle_mesh_flow(
                                vertices, triangles, accel_origin, accel_cell_size, accel_nx, accel_ny,
                                accel_cell_offsets, accel_triangle_indices, support_tolerance, times, ux, uy, t_mid_eval, xh_mid, yh_mid
                            )
                            if status > mask_status:
                                mask_status = status
                            targetx_mid2 = global_flow_scale * flow_scale_particle[i] * flowx_mid2
                            targety_mid2 = global_flow_scale * flow_scale_particle[i] * flowy_mid2
                            xmid, ymid, _vxmid, _vymid = advance_state_2d_etd(
                                x0, y0, vx0, vy0, targetx_mid2, targety_mid2, bax, bay, tau_eff, dt_mid
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
                flowx, flowy, status = _sample_triangle_mesh_flow(
                    vertices, triangles, accel_origin, accel_cell_size, accel_nx, accel_ny,
                    accel_cell_offsets, accel_triangle_indices, support_tolerance, times, ux, uy, t_eval, xn, yn
                )
                if status > mask_status:
                    mask_status = status
                targetx = global_flow_scale * flow_scale_particle[i] * flowx
                targety = global_flow_scale * flow_scale_particle[i] * flowy
                xn, yn, vxn, vyn = advance_state_2d(
                    xn, yn, vxn, vyn, targetx, targety, bax, bay, tau_eff, dt_sub, integrator_mode
                )
            x_mid_trial[i, 0] = xn
            x_mid_trial[i, 1] = yn
        x_trial[i, 0] = xn
        x_trial[i, 1] = yn
        v_trial[i, 0] = vxn
        v_trial[i, 1] = vyn
        mask_status_flags[i] = mask_status
        extension_band_sample_flags[i] = False
