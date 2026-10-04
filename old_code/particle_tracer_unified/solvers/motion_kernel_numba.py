"""Backend-agnostic ETD2 segment progression for compiled batch paths.

Field backends provide two leaf callbacks: one evaluates the local ETD
coefficients and one evaluates field support.  This module alone owns stage
timing, adaptive substeps, midpoint capture, and local-error resolution.
"""

from __future__ import annotations

from numba import njit

from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN

from .integrator_common import (
    advance_affine_stage_component,
    advance_state_2d,
    advance_state_3d,
    doubled_substep_count,
    etd2_position_error_exceeds_tolerance,
    etd2_stage_schedule,
    etd2_velocity_error_exceeds_tolerance,
    uniform_substep_schedule,
)


@njit(cache=True, inline="always")
def _axisymmetric_rz_chart_radius(radius, enabled):
    if enabled and radius < 0.0:
        return -radius, -1.0
    return radius, 1.0


@njit(cache=True, inline="always")
def _commit_inactive_particle(
    particle_index,
    dim,
    x,
    v,
    x_end,
    v_end,
    x_mid,
    substep_counts,
    mask_status_flags,
    local_error_resolved,
):
    for axis in range(dim):
        x_end[particle_index, axis] = x[particle_index, axis]
        v_end[particle_index, axis] = v[particle_index, axis]
        x_mid[particle_index, axis] = x[particle_index, axis]
    substep_counts[particle_index] = 1
    mask_status_flags[particle_index] = VALID_MASK_STATUS_CLEAN
    local_error_resolved[particle_index] = True


@njit(cache=True, inline="always")
def _larger_status(current_status, sampled_status):
    return max(current_status, sampled_status)


@njit(cache=True, inline="always")
def _local_error_exceeds_tolerance(
    dim,
    px,
    py,
    pz,
    vx,
    vy,
    vz,
    full_px,
    full_py,
    full_pz,
    full_vx,
    full_vy,
    full_vz,
    refined_px,
    refined_py,
    refined_pz,
    refined_vx,
    refined_vy,
    refined_vz,
    duration_s,
):
    if etd2_position_error_exceeds_tolerance(
        px, full_px, refined_px, vx, full_vx, refined_vx, duration_s
    ) or etd2_velocity_error_exceeds_tolerance(vx, full_vx, refined_vx):
        return True
    if etd2_position_error_exceeds_tolerance(
        py, full_py, refined_py, vy, full_vy, refined_vy, duration_s
    ) or etd2_velocity_error_exceeds_tolerance(vy, full_vy, refined_vy):
        return True
    if dim != 3:
        return False
    return etd2_position_error_exceeds_tolerance(
        pz, full_pz, refined_pz, vz, full_vz, refined_vz, duration_s
    ) or etd2_velocity_error_exceeds_tolerance(vz, full_vz, refined_vz)


@njit(cache=True, inline="always")
def _schedule_is_final(refinement_required, current_count, refined_count):
    return not refinement_required or refined_count == current_count


@njit(cache=True, inline="always")
def _should_estimate_local_error(adaptive_enabled):
    return int(adaptive_enabled) != 0


@njit(cache=True, inline="always")
def _advance_spatial_state(
    dim,
    px,
    py,
    pz,
    vx,
    vy,
    vz,
    target_x,
    target_y,
    target_z,
    accel_x,
    accel_y,
    accel_z,
    tau,
    dt,
):
    if dim == 2:
        next_x, next_y, next_vx, next_vy = advance_state_2d(
            px,
            py,
            vx,
            vy,
            target_x,
            target_y,
            accel_x,
            accel_y,
            tau,
            dt,
        )
        return next_x, next_y, 0.0, next_vx, next_vy, 0.0
    return advance_state_3d(
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        target_x,
        target_y,
        target_z,
        accel_x,
        accel_y,
        accel_z,
        tau,
        dt,
    )


@njit(cache=True, inline="always")
def _advance_affine_spatial_state(
    dim,
    px,
    py,
    pz,
    vx,
    vy,
    vz,
    target_x,
    target_y,
    target_z,
    target_mid_x,
    target_mid_y,
    target_mid_z,
    accel_x,
    accel_y,
    accel_z,
    accel_mid_x,
    accel_mid_y,
    accel_mid_z,
    tau_start,
    tau_mid,
    stage_fraction,
    dt,
):
    dx, next_vx = advance_affine_stage_component(
        vx,
        target_x,
        target_mid_x,
        accel_x,
        accel_mid_x,
        tau_start,
        tau_mid,
        stage_fraction,
        dt,
    )
    dy, next_vy = advance_affine_stage_component(
        vy,
        target_y,
        target_mid_y,
        accel_y,
        accel_mid_y,
        tau_start,
        tau_mid,
        stage_fraction,
        dt,
    )
    if dim != 3:
        return px + dx, py + dy, 0.0, next_vx, next_vy, 0.0
    dz, next_vz = advance_affine_stage_component(
        vz,
        target_z,
        target_mid_z,
        accel_z,
        accel_mid_z,
        tau_start,
        tau_mid,
        stage_fraction,
        dt,
    )
    return px + dx, py + dy, pz + dz, next_vx, next_vy, next_vz


@njit(inline="always")
def _advance_etd2_leaf(
    stage_evaluator,
    particle_index,
    dim,
    start_time_s,
    duration_s,
    px,
    py,
    pz,
    vx,
    vy,
    vz,
    tau_stokes,
    particle_diameter,
    particle_density,
    particle_mass,
    gas_molecular_mass_kg,
    drag_model_mode,
    backend_payload,
):
    """Sample and advance one ETD2 leaf without allocating stage vectors."""

    (
        target_x,
        target_y,
        target_z,
        accel_x,
        accel_y,
        accel_z,
        tau_start,
        start_status,
    ) = stage_evaluator(
        particle_index,
        start_time_s,
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        tau_stokes,
        particle_diameter,
        particle_density,
        particle_mass,
        gas_molecular_mass_kg,
        drag_model_mode,
        *backend_payload,
    )
    t_mid, predictor_dt, corrector_dt = etd2_stage_schedule(start_time_s, duration_s)
    (
        px_predictor,
        py_predictor,
        pz_predictor,
        vx_predictor,
        vy_predictor,
        vz_predictor,
    ) = _advance_spatial_state(
        dim,
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        target_x,
        target_y,
        target_z,
        accel_x,
        accel_y,
        accel_z,
        tau_start,
        predictor_dt,
    )
    (
        target_mid_x,
        target_mid_y,
        target_mid_z,
        accel_mid_x,
        accel_mid_y,
        accel_mid_z,
        tau_mid,
        midpoint_status,
    ) = stage_evaluator(
        particle_index,
        t_mid,
        px_predictor,
        py_predictor,
        pz_predictor,
        vx_predictor,
        vy_predictor,
        vz_predictor,
        tau_stokes,
        particle_diameter,
        particle_density,
        particle_mass,
        gas_molecular_mass_kg,
        drag_model_mode,
        *backend_payload,
    )
    px_half, py_half, pz_half, _vx_half, _vy_half, _vz_half = (
        _advance_affine_spatial_state(
            dim,
            px,
            py,
            pz,
            vx,
            vy,
            vz,
            target_x,
            target_y,
            target_z,
            target_mid_x,
            target_mid_y,
            target_mid_z,
            accel_x,
            accel_y,
            accel_z,
            accel_mid_x,
            accel_mid_y,
            accel_mid_z,
            tau_start,
            tau_mid,
            0.5,
            predictor_dt,
        )
    )
    px_next, py_next, pz_next, vx_next, vy_next, vz_next = (
        _advance_affine_spatial_state(
            dim,
            px,
            py,
            pz,
            vx,
            vy,
            vz,
            target_x,
            target_y,
            target_z,
            target_mid_x,
            target_mid_y,
            target_mid_z,
            accel_x,
            accel_y,
            accel_z,
            accel_mid_x,
            accel_mid_y,
            accel_mid_z,
            tau_start,
            tau_mid,
            1.0,
            corrector_dt,
        )
    )
    return (
        px_next,
        py_next,
        pz_next,
        vx_next,
        vy_next,
        vz_next,
        px_half,
        py_half,
        pz_half,
        start_status,
        midpoint_status,
    )


@njit(inline="always")
def _embedded_leaf_requires_refinement(
    stage_evaluator,
    particle_index,
    dim,
    start_time_s,
    duration_s,
    px,
    py,
    pz,
    vx,
    vy,
    vz,
    full_px,
    full_py,
    full_pz,
    full_vx,
    full_vy,
    full_vz,
    tau_stokes,
    particle_diameter,
    particle_density,
    particle_mass,
    gas_molecular_mass_kg,
    drag_model_mode,
    backend_payload,
):
    half_duration = 0.5 * duration_s
    (
        refined_mid_px,
        refined_mid_py,
        refined_mid_pz,
        refined_mid_vx,
        refined_mid_vy,
        refined_mid_vz,
        _quarter_px,
        _quarter_py,
        _quarter_pz,
        _fine_start_status,
        _quarter_status,
    ) = _advance_etd2_leaf(
        stage_evaluator,
        particle_index,
        dim,
        start_time_s,
        half_duration,
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        tau_stokes,
        particle_diameter,
        particle_density,
        particle_mass,
        gas_molecular_mass_kg,
        drag_model_mode,
        backend_payload,
    )
    (
        refined_px,
        refined_py,
        refined_pz,
        refined_vx,
        refined_vy,
        refined_vz,
        _three_quarter_px,
        _three_quarter_py,
        _three_quarter_pz,
        _fine_mid_status,
        _three_quarter_status,
    ) = _advance_etd2_leaf(
        stage_evaluator,
        particle_index,
        dim,
        start_time_s + half_duration,
        half_duration,
        refined_mid_px,
        refined_mid_py,
        refined_mid_pz,
        refined_mid_vx,
        refined_mid_vy,
        refined_mid_vz,
        tau_stokes,
        particle_diameter,
        particle_density,
        particle_mass,
        gas_molecular_mass_kg,
        drag_model_mode,
        backend_payload,
    )
    return (
        _local_error_exceeds_tolerance(
            dim,
            px,
            py,
            pz,
            vx,
            vy,
            vz,
            full_px,
            full_py,
            full_pz,
            full_vx,
            full_vy,
            full_vz,
            refined_px,
            refined_py,
            refined_pz,
            refined_vx,
            refined_vy,
            refined_vz,
            duration_s,
        ),
        refined_mid_px,
        refined_mid_py,
        refined_mid_pz,
        refined_mid_vx,
        refined_mid_vy,
        refined_mid_vz,
    )


@njit(cache=True, inline="always")
def _write_position(destination, particle_index, dim, px, py, pz):
    destination[particle_index, 0] = px
    destination[particle_index, 1] = py
    if dim == 3:
        destination[particle_index, 2] = pz


@njit(cache=True, inline="always")
def _capture_midpoint(
    destination,
    particle_index,
    dim,
    substep_index,
    substep_count,
    px_half,
    py_half,
    pz_half,
    px_next,
    py_next,
    pz_next,
):
    half_row = 2 * substep_index
    end_row = half_row + 1
    # With half/end nodes, global duration/2 is always exactly row
    # n_substeps-1, for both odd and even substep counts.
    if half_row == substep_count - 1:
        _write_position(
            destination,
            particle_index,
            dim,
            px_half,
            py_half,
            pz_half,
        )
    if end_row == substep_count - 1:
        _write_position(
            destination,
            particle_index,
            dim,
            px_next,
            py_next,
            pz_next,
        )


@njit(cache=True, inline="always")
def _commit_endpoint(
    x_end,
    v_end,
    particle_index,
    dim,
    px,
    py,
    pz,
    vx,
    vy,
    vz,
):
    _write_position(x_end, particle_index, dim, px, py, pz)
    v_end[particle_index, 0] = vx
    v_end[particle_index, 1] = vy
    if dim == 3:
        v_end[particle_index, 2] = vz


# Dispatcher callbacks are runtime arguments. Numba cannot safely serialize those
# references in its disk cache, and a fresh CLI process can otherwise fail while
# writing the overload with ``ReferenceError: underlying object has vanished``.
@njit(cache=False)
def advance_etd2_batch_inplace(
    stage_evaluator,
    support_evaluator,
    spatial_dim,
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
    *backend_payload,
):
    """Advance a particle batch with one canonical ETD2 progression rule.

    ``stage_evaluator`` returns ``target_xyz, acceleration_xyz, tau, status``
    as a flat eight-value tuple.  A fixed xyz tuple keeps the hot loop common
    to 2D and 3D without allocating stage vectors.  Full traces are replayed
    lazily by the scalar form only for particles that require them.
    """

    dim = int(spatial_dim)
    for i in range(x.shape[0]):
        if not active[i]:
            _commit_inactive_particle(
                i,
                dim,
                x,
                v,
                x_end,
                v_end,
                x_mid,
                substep_counts,
                mask_status_flags,
                local_error_resolved,
            )
            continue

        initial_px = x[i, 0]
        initial_py = x[i, 1]
        initial_pz = x[i, 2] if dim == 3 else 0.0
        initial_vx = v[i, 0]
        initial_vy = v[i, 1]
        initial_vz = v[i, 2] if dim == 3 else 0.0
        tau_stokes = tau_p[i]

        n_substeps, dt_sub, t_start = uniform_substep_schedule(
            duration,
            t_end,
            adaptive_substep_max_splits,
            1,
        )
        while True:
            px = initial_px
            py = initial_py
            pz = initial_pz
            vx = initial_vx
            vy = initial_vy
            vz = initial_vz
            mask_status = VALID_MASK_STATUS_CLEAN
            refinement_required = False
            refined_count = doubled_substep_count(
                n_substeps,
                adaptive_substep_max_splits,
            )
            estimate_local_error = _should_estimate_local_error(
                adaptive_substep_enabled
            )

            for sub_idx in range(n_substeps):
                t_sub_start = t_start + float(sub_idx) * dt_sub
                (
                    px_next,
                    py_next,
                    pz_next,
                    vx_next,
                    vy_next,
                    vz_next,
                    px_half,
                    py_half,
                    pz_half,
                    start_status,
                    midpoint_status,
                ) = _advance_etd2_leaf(
                    stage_evaluator,
                    i,
                    dim,
                    t_sub_start,
                    dt_sub,
                    px,
                    py,
                    pz,
                    vx,
                    vy,
                    vz,
                    tau_stokes,
                    particle_diameter[i],
                    particle_density[i],
                    particle_mass[i],
                    gas_molecular_mass_kg,
                    drag_model_mode,
                    backend_payload,
                )
                mask_status = _larger_status(mask_status, start_status)
                mask_status = _larger_status(mask_status, midpoint_status)
                if not refinement_required and estimate_local_error:
                    (
                        refinement_required,
                        px_half,
                        py_half,
                        pz_half,
                        _vx_half,
                        _vy_half,
                        _vz_half,
                    ) = _embedded_leaf_requires_refinement(
                        stage_evaluator,
                        i,
                        dim,
                        t_sub_start,
                        dt_sub,
                        px,
                        py,
                        pz,
                        vx,
                        vy,
                        vz,
                        px_next,
                        py_next,
                        pz_next,
                        vx_next,
                        vy_next,
                        vz_next,
                        tau_stokes,
                        particle_diameter[i],
                        particle_density[i],
                        particle_mass[i],
                        gas_molecular_mass_kg,
                        drag_model_mode,
                        backend_payload,
                    )

                corrected_midpoint_status = support_evaluator(
                    px_half,
                    py_half,
                    pz_half,
                    *backend_payload,
                )
                mask_status = _larger_status(
                    mask_status,
                    corrected_midpoint_status,
                )

                endpoint_status = support_evaluator(
                    px_next,
                    py_next,
                    pz_next,
                    *backend_payload,
                )
                mask_status = _larger_status(mask_status, endpoint_status)

                _capture_midpoint(
                    x_mid,
                    i,
                    dim,
                    sub_idx,
                    n_substeps,
                    px_half,
                    py_half,
                    pz_half,
                    px_next,
                    py_next,
                    pz_next,
                )

                px = px_next
                py = py_next
                pz = pz_next
                vx = vx_next
                vy = vy_next
                vz = vz_next

            if _schedule_is_final(
                refinement_required,
                n_substeps,
                refined_count,
            ):
                break
            n_substeps = refined_count
            dt_sub = max(duration, 0.0) / float(n_substeps)

        _commit_endpoint(x_end, v_end, i, dim, px, py, pz, vx, vy, vz)
        substep_counts[i] = n_substeps
        mask_status_flags[i] = mask_status
        local_error_resolved[i] = not refinement_required


__all__ = ("advance_etd2_batch_inplace",)
