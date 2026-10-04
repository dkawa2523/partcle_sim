"""Triangle-mesh 2D sampling leaves for the canonical motion kernel."""

from __future__ import annotations

import numpy as np
from numba import njit

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
)

from .drag_models import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    effective_tau_from_drag_model,
)
from .motion_kernel_numba import (
    _axisymmetric_rz_chart_radius,
    advance_etd2_batch_inplace,
)

_OUTSIDE_MESH_RING_LIMIT = 2


@njit(cache=True, inline="always")
def _outside_acceleration_grid_2d(x, y, xmin, xmax, ymin, ymax, tolerance):
    return (
        x < xmin - tolerance
        or x > xmax + tolerance
        or y < ymin - tolerance
        or y > ymax + tolerance
    )


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
    if _outside_acceleration_grid_2d(x, y, xmin, xmax, ymin, ymax, eps):
        return -1, 0.0, 0.0, 0.0
    ix = min(accel_nx - 1, max(0, int(np.floor((x - xmin) / accel_cell_size[0]))))
    iy = min(accel_ny - 1, max(0, int(np.floor((y - ymin) / accel_cell_size[1]))))
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
        edge_ab = np.sqrt(v0x * v0x + v0y * v0y)
        edge_ca = np.sqrt(v1x * v1x + v1y * v1y)
        determinant_scale = edge_ab * edge_ca
        if determinant_scale <= 0.0:
            continue
        denominator = v0x * v1y - v0y * v1x
        if abs(denominator) <= 64.0 * np.finfo(np.float64).eps * determinant_scale:
            continue
        beta = (v2x * v1y - v2y * v1x) / denominator
        gamma = (v0x * v2y - v0y * v2x) / denominator
        alpha = 1.0 - beta - gamma
        edge_bcx = cx - bx
        edge_bcy = cy - by
        edge_bc = np.sqrt(edge_bcx * edge_bcx + edge_bcy * edge_bcy)
        area2 = abs(denominator)
        h_alpha = area2 / edge_bc
        h_beta = area2 / edge_ca
        h_gamma = area2 / edge_ab
        if alpha < -eps / h_alpha or beta < -eps / h_beta or gamma < -eps / h_gamma:
            continue
        distance_margin = min(alpha * h_alpha, beta * h_beta, gamma * h_gamma)
        if distance_margin > best_margin:
            best_margin = distance_margin
            best_idx = tri_idx
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
    return best_idx, best_alpha, best_beta, best_gamma


@njit(cache=True)
def _triangle_barycentric_margin(vertices, triangles, tri_idx, x, y):
    """Return barycentric weights and the signed inside distance of a point."""

    i0 = triangles[tri_idx, 0]
    i1 = triangles[tri_idx, 1]
    i2 = triangles[tri_idx, 2]
    ax = vertices[i0, 0]
    ay = vertices[i0, 1]
    v0x = vertices[i1, 0] - ax
    v0y = vertices[i1, 1] - ay
    v1x = vertices[i2, 0] - ax
    v1y = vertices[i2, 1] - ay
    edge_ab = np.sqrt(v0x * v0x + v0y * v0y)
    edge_ca = np.sqrt(v1x * v1x + v1y * v1y)
    determinant_scale = edge_ab * edge_ca
    if determinant_scale <= 0.0:
        return False, 0.0, 0.0, 0.0, -1.0e300
    denominator = v0x * v1y - v0y * v1x
    if abs(denominator) <= 64.0 * np.finfo(np.float64).eps * determinant_scale:
        return False, 0.0, 0.0, 0.0, -1.0e300
    v2x = x - ax
    v2y = y - ay
    beta = (v2x * v1y - v2y * v1x) / denominator
    gamma = (v0x * v2y - v0y * v2x) / denominator
    alpha = 1.0 - beta - gamma
    edge_bcx = vertices[i2, 0] - vertices[i1, 0]
    edge_bcy = vertices[i2, 1] - vertices[i1, 1]
    edge_bc = np.sqrt(edge_bcx * edge_bcx + edge_bcy * edge_bcy)
    area2 = abs(denominator)
    margin = min(
        alpha * (area2 / edge_bc),
        beta * (area2 / edge_ca),
        gamma * (area2 / edge_ab),
    )
    return True, alpha, beta, gamma, margin


@njit(cache=True)
def _best_margin_in_cell(
    vertices, triangles, accel_cell_offsets, accel_triangle_indices, cell_id, x, y
):
    best_idx = -1
    best_alpha = 0.0
    best_beta = 0.0
    best_gamma = 0.0
    best_margin = -1.0e300
    for flat_idx in range(accel_cell_offsets[cell_id], accel_cell_offsets[cell_id + 1]):
        tri_idx = accel_triangle_indices[flat_idx]
        ok, alpha, beta, gamma, margin = _triangle_barycentric_margin(
            vertices, triangles, tri_idx, x, y
        )
        if ok and margin > best_margin:
            best_margin = margin
            best_idx = tri_idx
            best_alpha = alpha
            best_beta = beta
            best_gamma = gamma
    return best_idx, best_alpha, best_beta, best_gamma, best_margin


@njit(cache=True)
def _nearest_triangle_and_clamped_barycentric(
    vertices,
    triangles,
    accel_origin,
    accel_cell_size,
    accel_nx,
    accel_ny,
    accel_cell_offsets,
    accel_triangle_indices,
    x,
    y,
    ring_limit,
):
    """Compiled form of ``_nearest_triangle_candidate``.

    See :mod:`particle_tracer_unified.core.triangle_mesh_sampling_2d` for why
    an outside point clamps to the nearest element instead of returning NaN.
    """

    ix = min(
        accel_nx - 1,
        max(0, int(np.floor((x - accel_origin[0]) / accel_cell_size[0]))),
    )
    iy = min(
        accel_ny - 1,
        max(0, int(np.floor((y - accel_origin[1]) / accel_cell_size[1]))),
    )
    best_idx = -1
    best_alpha = 0.0
    best_beta = 0.0
    best_gamma = 0.0
    best_margin = -1.0e300
    for ring in range(int(ring_limit) + 1):
        for cell_x in range(max(0, ix - ring), min(accel_nx, ix + ring + 1)):
            for cell_y in range(max(0, iy - ring), min(accel_ny, iy + ring + 1)):
                if ring > 0 and abs(cell_x - ix) != ring and abs(cell_y - iy) != ring:
                    continue
                idx, alpha, beta, gamma, margin = _best_margin_in_cell(
                    vertices,
                    triangles,
                    accel_cell_offsets,
                    accel_triangle_indices,
                    cell_x * accel_ny + cell_y,
                    x,
                    y,
                )
                if idx >= 0 and margin > best_margin:
                    best_margin = margin
                    best_idx = idx
                    best_alpha = alpha
                    best_beta = beta
                    best_gamma = gamma
        if best_idx >= 0:
            break
    if best_idx < 0:
        return -1, 0.0, 0.0, 0.0
    clamped_alpha = max(0.0, best_alpha)
    clamped_beta = max(0.0, best_beta)
    clamped_gamma = max(0.0, best_gamma)
    total = clamped_alpha + clamped_beta + clamped_gamma
    if not (total > 0.0):
        return -1, 0.0, 0.0, 0.0
    return (
        best_idx,
        clamped_alpha / total,
        clamped_beta / total,
        clamped_gamma / total,
    )


@njit(cache=True)
def _sample_triangle_vertex_series(
    arr, times, triangles, tri_idx, alpha, beta, gamma, t
):
    i0 = triangles[tri_idx, 0]
    i1 = triangles[tri_idx, 1]
    i2 = triangles[tri_idx, 2]
    if arr.ndim == 1:
        return alpha * arr[i0] + beta * arr[i1] + gamma * arr[i2]
    nt = times.size
    if nt <= 1 or arr.shape[0] <= 1 or t <= times[0]:
        return alpha * arr[0, i0] + beta * arr[0, i1] + gamma * arr[0, i2]
    if t >= times[nt - 1]:
        return (
            alpha * arr[nt - 1, i0] + beta * arr[nt - 1, i1] + gamma * arr[nt - 1, i2]
        )
    hi = np.searchsorted(times, t)
    lo = hi - 1
    denominator = times[hi] - times[lo]
    time_alpha = 0.0 if denominator == 0.0 else (t - times[lo]) / denominator
    value_lo = alpha * arr[lo, i0] + beta * arr[lo, i1] + gamma * arr[lo, i2]
    value_hi = alpha * arr[hi, i0] + beta * arr[hi, i1] + gamma * arr[hi, i2]
    return value_lo * (1.0 - time_alpha) + value_hi * time_alpha


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
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    t,
    x,
    y,
):
    triangle_index, alpha, beta, gamma = _find_triangle_and_barycentric(
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
    status = VALID_MASK_STATUS_CLEAN
    if triangle_index < 0:
        status = VALID_MASK_STATUS_HARD_INVALID
        (
            triangle_index,
            alpha,
            beta,
            gamma,
        ) = _nearest_triangle_and_clamped_barycentric(
            vertices,
            triangles,
            accel_origin,
            accel_cell_size,
            accel_nx,
            accel_ny,
            accel_cell_offsets,
            accel_triangle_indices,
            x,
            y,
            _OUTSIDE_MESH_RING_LIMIT,
        )
        if triangle_index < 0:
            return 0.0, 0.0, np.nan, np.nan, np.nan, VALID_MASK_STATUS_HARD_INVALID
    flow_x = _sample_triangle_vertex_series(
        ux, times, triangles, triangle_index, alpha, beta, gamma, t
    )
    flow_y = _sample_triangle_vertex_series(
        uy, times, triangles, triangle_index, alpha, beta, gamma, t
    )
    density = _sample_triangle_vertex_series(
        gas_density_grid, times, triangles, triangle_index, alpha, beta, gamma, t
    )
    viscosity = _sample_triangle_vertex_series(
        gas_mu_grid, times, triangles, triangle_index, alpha, beta, gamma, t
    )
    temperature = _sample_triangle_vertex_series(
        gas_temperature_grid, times, triangles, triangle_index, alpha, beta, gamma, t
    )
    return flow_x, flow_y, density, viscosity, temperature, status


@njit(cache=True)
def _triangle_2d_stage(
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
    _particle_density,
    particle_mass,
    gas_molecular_mass_kg,
    drag_model_mode,
    epstein_accommodation_delta,
    body_ax,
    body_ay,
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
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    extra_accel_x,
    extra_accel_y,
    axisymmetric_rz=0,
):
    sample_x, chart_sign = _axisymmetric_rz_chart_radius(x, axisymmetric_rz)
    flow_x, flow_y, rho_g, mu_g, temperature, status = _sample_triangle_mesh_flow(
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
        gas_density_grid,
        gas_mu_grid,
        gas_temperature_grid,
        time_s,
        sample_x,
        y,
    )
    flow_x *= chart_sign
    tau = effective_tau_from_drag_model(
        tau_stokes,
        np.sqrt((vx - flow_x) ** 2 + (vy - flow_y) ** 2),
        particle_diameter,
        rho_g,
        mu_g,
        drag_model_mode,
        particle_mass,
        temperature,
        gas_molecular_mass_kg,
        epstein_accommodation_delta,
    )
    return (
        flow_x,
        flow_y,
        0.0,
        chart_sign * (body_ax + extra_accel_x[particle_index]),
        body_ay + extra_accel_y[particle_index],
        0.0,
        tau,
        status,
    )


@njit(cache=True)
def _triangle_2d_support(
    x,
    y,
    _z,
    _epstein_accommodation_delta,
    body_ax,
    body_ay,
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
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    extra_accel_x,
    extra_accel_y,
    axisymmetric_rz=0,
):
    sample_x, _chart_sign = _axisymmetric_rz_chart_radius(x, axisymmetric_rz)
    triangle_index, _alpha, _beta, _gamma = _find_triangle_and_barycentric(
        vertices,
        triangles,
        accel_origin,
        accel_cell_size,
        accel_nx,
        accel_ny,
        accel_cell_offsets,
        accel_triangle_indices,
        support_tolerance,
        sample_x,
        y,
    )
    return (
        VALID_MASK_STATUS_CLEAN
        if triangle_index >= 0
        else VALID_MASK_STATUS_HARD_INVALID
    )


def trace_triangle_2d_batch_inplace(
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
    gas_molecular_mass_kg,
    drag_model_mode,
    adaptive_substep_enabled,
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
    gas_density_grid,
    gas_mu_grid,
    gas_temperature_grid,
    extra_accel_x_particle,
    extra_accel_y_particle,
    x_end,
    v_end,
    x_mid,
    substep_counts,
    mask_status_flags,
    local_error_resolved,
    axisymmetric_rz=0,
):
    advance_etd2_batch_inplace(
        _triangle_2d_stage,
        _triangle_2d_support,
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
        gas_density_grid,
        gas_mu_grid,
        gas_temperature_grid,
        extra_accel_x_particle,
        extra_accel_y_particle,
        int(axisymmetric_rz),
    )


__all__ = (
    "_find_triangle_and_barycentric",
    "_sample_triangle_mesh_flow",
    "_sample_triangle_vertex_series",
    "trace_triangle_2d_batch_inplace",
)
