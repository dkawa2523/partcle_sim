from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from ..core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    valid_mask_status_requires_stop,
)
from .compiled_field_backend import (
    CompiledRuntimeBackend,
    CompiledRuntimeBackendLike,
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
    coerce_compiled_backend as _coerce_compiled_backend,
    compile_runtime_backend as _compile_runtime_arrays,
    sample_compiled_extension_band_active as _sample_extension_band_active,
    sample_compiled_flow_vector as _sample_flow_vector_at,
    sample_compiled_valid_mask_status as _sample_valid_mask_status,
)
from .integrator_common import (
    INTEGRATOR_ETD2,
    advance_state_2d,
    advance_state_2d_etd,
    advance_state_3d,
    advance_state_3d_etd,
    compute_substep_count,
)
from .kernel2d_numba import advance_particles_2d_inplace
from .kernel2d_triangle_mesh_numba import advance_particles_2d_triangle_mesh_inplace
from .kernel3d_numba import advance_particles_3d_inplace

@dataclass(slots=True)
class ValidMaskPrefixResolution:
    position: np.ndarray
    velocity: np.ndarray
    accepted_dt: float
    retry_count: int
    found_valid_prefix: bool


def _stage_points_from_trial(
    x_end: np.ndarray,
    *,
    integrator_mode: int,
    x_mid: Optional[np.ndarray] = None,
) -> np.ndarray:
    x_end_arr = np.asarray(x_end, dtype=np.float64)
    if int(integrator_mode) == INTEGRATOR_ETD2 and x_mid is not None:
        x_mid_arr = np.asarray(x_mid, dtype=np.float64)
        return np.stack((x_mid_arr, x_end_arr), axis=0)
    return x_end_arr.reshape(1, x_end_arr.size).copy()


def _advance_etd2_substep(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_sub: float,
    t_sub_start: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    flow_scale_particle_i: float,
    global_flow_scale: float,
    body: np.ndarray,
    tau_eff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow_start = _sample_flow_vector_at(compiled, spatial_dim, float(t_sub_start), x0)
    target_start = float(global_flow_scale) * float(flow_scale_particle_i) * flow_start
    if spatial_dim == 2:
        xh, yh, _vxh, _vyh = advance_state_2d_etd(
            float(x0[0]),
            float(x0[1]),
            float(v0[0]),
            float(v0[1]),
            float(target_start[0]),
            float(target_start[1]),
            float(body[0]),
            float(body[1]),
            float(tau_eff),
            0.5 * float(dt_sub),
        )
        t_mid = float(t_sub_start) + 0.5 * float(dt_sub)
        flow_mid = _sample_flow_vector_at(compiled, spatial_dim, t_mid, np.asarray([xh, yh], dtype=np.float64))
        target_mid = float(global_flow_scale) * float(flow_scale_particle_i) * flow_mid
        xn, yn, vxn, vyn = advance_state_2d_etd(
            float(x0[0]),
            float(x0[1]),
            float(v0[0]),
            float(v0[1]),
            float(target_mid[0]),
            float(target_mid[1]),
            float(body[0]),
            float(body[1]),
            float(tau_eff),
            float(dt_sub),
        )
        return (
            np.asarray([xn, yn], dtype=np.float64),
            np.asarray([vxn, vyn], dtype=np.float64),
            np.asarray([xh, yh], dtype=np.float64),
        )
    xh, yh, zh, _vxh, _vyh, _vzh = advance_state_3d_etd(
        float(x0[0]),
        float(x0[1]),
        float(x0[2]),
        float(v0[0]),
        float(v0[1]),
        float(v0[2]),
        float(target_start[0]),
        float(target_start[1]),
        float(target_start[2]),
        float(body[0]),
        float(body[1]),
        float(body[2]),
        float(tau_eff),
        0.5 * float(dt_sub),
    )
    t_mid = float(t_sub_start) + 0.5 * float(dt_sub)
    flow_mid = _sample_flow_vector_at(compiled, spatial_dim, t_mid, np.asarray([xh, yh, zh], dtype=np.float64))
    target_mid = float(global_flow_scale) * float(flow_scale_particle_i) * flow_mid
    xn, yn, zn, vxn, vyn, vzn = advance_state_3d_etd(
        float(x0[0]),
        float(x0[1]),
        float(x0[2]),
        float(v0[0]),
        float(v0[1]),
        float(v0[2]),
        float(target_mid[0]),
        float(target_mid[1]),
        float(target_mid[2]),
        float(body[0]),
        float(body[1]),
        float(body[2]),
        float(tau_eff),
        float(dt_sub),
    )
    return (
        np.asarray([xn, yn, zn], dtype=np.float64),
        np.asarray([vxn, vyn, vzn], dtype=np.float64),
        np.asarray([xh, yh, zh], dtype=np.float64),
    )


def _stage_sample_times(segment_dt: float, stage_points: np.ndarray) -> np.ndarray:
    points = np.asarray(stage_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    count = int(points.shape[0])
    dt_seg = max(float(segment_dt), 0.0)
    if count == 1 or dt_seg <= 0.0:
        return np.asarray([dt_seg], dtype=np.float64)
    if count == 2:
        return np.asarray([0.5 * dt_seg, dt_seg], dtype=np.float64)
    return np.linspace(dt_seg / float(count), dt_seg, count, dtype=np.float64)


def advance_freeflight_segment(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_segment: float,
    t_end_segment: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    tau_p_i: float,
    flow_scale_particle_i: float,
    drag_scale_particle_i: float,
    body_scale_particle_i: float,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    global_body_accel_scale: float,
    body_accel: np.ndarray,
    min_tau_p_s: float,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, int, bool]:
    backend = _coerce_compiled_backend(compiled)
    dt_seg = float(max(dt_segment, 0.0))
    x_curr = np.asarray(x0, dtype=np.float64).copy()
    v_curr = np.asarray(v0, dtype=np.float64).copy()
    if dt_seg <= 0.0:
        return x_curr, v_curr, 1, x_curr.reshape(1, x_curr.size).copy(), int(VALID_MASK_STATUS_CLEAN), False
    tau_eff = float(tau_p_i) * float(global_drag_tau_scale) * max(float(drag_scale_particle_i), 1e-6)
    tau_eff = max(float(min_tau_p_s), tau_eff)
    n_substeps = int(
        compute_substep_count(
            dt_seg,
            tau_eff,
            int(adaptive_substep_enabled),
            float(adaptive_substep_tau_ratio),
            int(adaptive_substep_max_splits),
        )
    )
    dt_sub = dt_seg / float(max(1, n_substeps))
    t_start = float(t_end_segment) - dt_seg
    body = np.asarray(body_accel, dtype=np.float64)[:spatial_dim] * float(global_body_accel_scale) * float(body_scale_particle_i)
    is_etd2 = int(integrator_mode) == INTEGRATOR_ETD2
    stage_mid = x_curr.copy()
    stage_mid_captured = False
    mid_target = 0.5 * dt_seg
    elapsed = 0.0
    valid_mask_status = int(VALID_MASK_STATUS_CLEAN)
    extension_band_sampled = False
    for sub_idx in range(n_substeps):
        t_sub_start = t_start + float(sub_idx) * dt_sub
        x_prev = x_curr.copy()
        v_prev = v_curr.copy()
        if is_etd2:
            sample_status = _sample_valid_mask_status(backend, x_prev)
            if sample_status > valid_mask_status:
                valid_mask_status = int(sample_status)
            if _sample_extension_band_active(backend, x_prev):
                extension_band_sampled = True
            x_curr, v_curr, x_half = _advance_etd2_substep(
                x0=x_prev,
                v0=v_prev,
                dt_sub=float(dt_sub),
                t_sub_start=float(t_sub_start),
                spatial_dim=int(spatial_dim),
                compiled=backend,
                flow_scale_particle_i=float(flow_scale_particle_i),
                global_flow_scale=float(global_flow_scale),
                body=body,
                tau_eff=float(tau_eff),
            )
            sample_status = _sample_valid_mask_status(backend, x_half)
            if sample_status > valid_mask_status:
                valid_mask_status = int(sample_status)
            if _sample_extension_band_active(backend, x_half):
                extension_band_sampled = True
        else:
            t_eval = t_start + (float(sub_idx) + 1.0) * dt_sub
            sample_status = _sample_valid_mask_status(backend, x_curr)
            if sample_status > valid_mask_status:
                valid_mask_status = int(sample_status)
            if _sample_extension_band_active(backend, x_curr):
                extension_band_sampled = True
            flow = _sample_flow_vector_at(backend, spatial_dim, t_eval, x_curr)
            target = float(global_flow_scale) * float(flow_scale_particle_i) * flow
            if spatial_dim == 2:
                x0n, y0n, vxn, vyn = advance_state_2d(
                    float(x_curr[0]),
                    float(x_curr[1]),
                    float(v_curr[0]),
                    float(v_curr[1]),
                    float(target[0]),
                    float(target[1]),
                    float(body[0]),
                    float(body[1]),
                    tau_eff,
                    dt_sub,
                    int(integrator_mode),
                )
                x_curr[0], x_curr[1] = x0n, y0n
                v_curr[0], v_curr[1] = vxn, vyn
            else:
                x0n, y0n, z0n, vxn, vyn, vzn = advance_state_3d(
                    float(x_curr[0]),
                    float(x_curr[1]),
                    float(x_curr[2]),
                    float(v_curr[0]),
                    float(v_curr[1]),
                    float(v_curr[2]),
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                    float(body[0]),
                    float(body[1]),
                    float(body[2]),
                    tau_eff,
                    dt_sub,
                    int(integrator_mode),
                )
                x_curr[0], x_curr[1], x_curr[2] = x0n, y0n, z0n
                v_curr[0], v_curr[1], v_curr[2] = vxn, vyn, vzn
        if is_etd2 and not stage_mid_captured:
            elapsed_next = elapsed + dt_sub
            if mid_target <= elapsed_next + 1e-15:
                dt_local = float(np.clip(mid_target - elapsed, 0.0, dt_sub))
                if dt_local <= 1e-15:
                    stage_mid = x_prev.copy()
                elif dt_local >= dt_sub - 1e-15:
                    stage_mid = x_curr.copy()
                else:
                    x_mid, _v_mid, _x_half_mid = _advance_etd2_substep(
                        x0=x_prev,
                        v0=v_prev,
                        dt_sub=float(dt_local),
                        t_sub_start=float(t_sub_start),
                        spatial_dim=int(spatial_dim),
                        compiled=backend,
                        flow_scale_particle_i=float(flow_scale_particle_i),
                        global_flow_scale=float(global_flow_scale),
                        body=body,
                        tau_eff=float(tau_eff),
                    )
                    sample_status = _sample_valid_mask_status(backend, x_mid)
                    if sample_status > valid_mask_status:
                        valid_mask_status = int(sample_status)
                    if _sample_extension_band_active(backend, x_mid):
                        extension_band_sampled = True
                    stage_mid = x_mid
                stage_mid_captured = True
            elapsed = elapsed_next
    if is_etd2:
        if not stage_mid_captured:
            stage_mid = x_curr.copy()
        stage_points = np.stack((stage_mid, x_curr.copy()), axis=0)
    else:
        stage_points = x_curr.reshape(1, x_curr.size).copy()
    return x_curr, v_curr, int(max(1, n_substeps)), stage_points, int(valid_mask_status), bool(extension_band_sampled)


def advance_freeflight_partial(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_partial: float,
    segment_dt: float,
    t_end_segment: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    tau_p_i: float,
    flow_scale_particle_i: float,
    drag_scale_particle_i: float,
    body_scale_particle_i: float,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    global_body_accel_scale: float,
    body_accel: np.ndarray,
    min_tau_p_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dt_seg = max(float(segment_dt), 0.0)
    dt_eval = float(np.clip(dt_partial, 0.0, dt_seg))
    if dt_eval <= 0.0:
        return np.asarray(x0, dtype=np.float64).copy(), np.asarray(v0, dtype=np.float64).copy()
    partial_t_end = float(t_end_segment) - dt_seg + dt_eval
    x_out, v_out, _n_substeps, _stage_points, _valid_mask_status, _extension_band_sampled = advance_freeflight_segment(
        x0=x0,
        v0=v0,
        dt_segment=float(dt_eval),
        t_end_segment=float(partial_t_end),
        spatial_dim=int(spatial_dim),
        compiled=compiled,
        integrator_mode=int(integrator_mode),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        tau_p_i=float(tau_p_i),
        flow_scale_particle_i=float(flow_scale_particle_i),
        drag_scale_particle_i=float(drag_scale_particle_i),
        body_scale_particle_i=float(body_scale_particle_i),
        global_flow_scale=float(global_flow_scale),
        global_drag_tau_scale=float(global_drag_tau_scale),
        global_body_accel_scale=float(global_body_accel_scale),
        body_accel=body_accel,
        min_tau_p_s=float(min_tau_p_s),
    )
    return x_out, v_out


def resolve_valid_mask_prefix(
    *,
    x0: np.ndarray,
    v0: np.ndarray,
    dt_segment: float,
    t_end_segment: float,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    tau_p_i: float,
    flow_scale_particle_i: float,
    drag_scale_particle_i: float,
    body_scale_particle_i: float,
    global_flow_scale: float,
    global_drag_tau_scale: float,
    global_body_accel_scale: float,
    body_accel: np.ndarray,
    min_tau_p_s: float,
    max_halving_count: int,
) -> ValidMaskPrefixResolution:
    x_start = np.asarray(x0, dtype=np.float64).copy()
    v_start = np.asarray(v0, dtype=np.float64).copy()
    dt_seg = max(float(dt_segment), 0.0)
    halving_limit = int(max(0, max_halving_count))
    if dt_seg <= 0.0 or halving_limit <= 0:
        return ValidMaskPrefixResolution(
            position=x_start,
            velocity=v_start,
            accepted_dt=0.0,
            retry_count=0,
            found_valid_prefix=False,
        )

    retry_count = 0
    for split_idx in range(1, halving_limit + 1):
        retry_count += 1
        prefix_dt = float(dt_seg) * (0.5 ** int(split_idx))
        prefix_t_end = float(t_end_segment) - float(dt_seg) + float(prefix_dt)
        x_retry, v_retry, _substeps, _stage_points, retry_status, _retry_extension_band_sampled = advance_freeflight_segment(
            x0=x_start,
            v0=v_start,
            dt_segment=float(prefix_dt),
            t_end_segment=float(prefix_t_end),
            spatial_dim=int(spatial_dim),
            compiled=compiled,
            integrator_mode=int(integrator_mode),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_tau_ratio=float(adaptive_substep_tau_ratio),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            tau_p_i=float(tau_p_i),
            flow_scale_particle_i=float(flow_scale_particle_i),
            drag_scale_particle_i=float(drag_scale_particle_i),
            body_scale_particle_i=float(body_scale_particle_i),
            global_flow_scale=float(global_flow_scale),
            global_drag_tau_scale=float(global_drag_tau_scale),
            global_body_accel_scale=float(global_body_accel_scale),
            body_accel=body_accel,
            min_tau_p_s=float(min_tau_p_s),
        )
        if not bool(valid_mask_status_requires_stop(int(retry_status))):
            return ValidMaskPrefixResolution(
                position=np.asarray(x_retry, dtype=np.float64),
                velocity=np.asarray(v_retry, dtype=np.float64),
                accepted_dt=float(prefix_dt),
                retry_count=int(retry_count),
                found_valid_prefix=True,
            )

    return ValidMaskPrefixResolution(
        position=x_start,
        velocity=v_start,
        accepted_dt=0.0,
        retry_count=int(retry_count),
        found_valid_prefix=False,
    )


def _advance_trial_particles(
    *,
    spatial_dim: int,
    compiled: CompiledRuntimeBackendLike,
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    tau_p: np.ndarray,
    flow_scale_particle: np.ndarray,
    drag_scale_particle: np.ndarray,
    body_scale_particle: np.ndarray,
    t: float,
    dt_step: float,
    phys: Mapping[str, object],
    body_accel: np.ndarray,
    integrator_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_tau_ratio: float,
    adaptive_substep_max_splits: int,
    x_trial: np.ndarray,
    v_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    substep_counts: np.ndarray,
    valid_mask_status_flags: np.ndarray,
    extension_band_sample_flags: np.ndarray,
) -> None:
    backend = _coerce_compiled_backend(compiled)
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        if int(spatial_dim) != 2:
            raise ValueError('triangle_mesh_2d backend currently supports only spatial_dim=2')
        accel_shape = np.asarray(backend.accel_shape, dtype=np.int32)
        advance_particles_2d_triangle_mesh_inplace(
            x, v, active, tau_p, flow_scale_particle, drag_scale_particle, body_scale_particle,
            float(t), float(dt_step), float(phys['flow_scale']), float(phys['drag_tau_scale']), float(phys['body_accel_scale']),
            float(body_accel[0]), float(body_accel[1]), float(phys['min_tau_p_s']),
            int(integrator_mode),
            int(adaptive_substep_enabled),
            float(adaptive_substep_tau_ratio),
            int(adaptive_substep_max_splits),
            np.asarray(backend.mesh_vertices, dtype=np.float64),
            np.asarray(backend.mesh_triangles, dtype=np.int32),
            np.asarray(backend.accel_origin, dtype=np.float64),
            np.asarray(backend.accel_cell_size, dtype=np.float64),
            int(accel_shape[0]),
            int(accel_shape[1]),
            np.asarray(backend.accel_cell_offsets, dtype=np.int32),
            np.asarray(backend.accel_triangle_indices, dtype=np.int32),
            float(backend.support_tolerance_m),
            np.asarray(backend.times, dtype=np.float64),
            np.asarray(backend.ux, dtype=np.float64),
            np.asarray(backend.uy, dtype=np.float64),
            x_trial, v_trial, x_mid_trial, substep_counts, valid_mask_status_flags, extension_band_sample_flags,
        )
        return
    valid_mask = np.asarray(backend.valid_mask, dtype=bool)
    core_valid_mask = np.asarray(backend.core_valid_mask, dtype=bool)
    if int(spatial_dim) == 2:
        xs, ys = backend.axes
        advance_particles_2d_inplace(
            x, v, active, tau_p, flow_scale_particle, drag_scale_particle, body_scale_particle,
            float(t), float(dt_step), float(phys['flow_scale']), float(phys['drag_tau_scale']), float(phys['body_accel_scale']),
            float(body_accel[0]), float(body_accel[1]), float(phys['min_tau_p_s']),
            int(integrator_mode),
            int(adaptive_substep_enabled),
            float(adaptive_substep_tau_ratio),
            int(adaptive_substep_max_splits),
            xs, ys, backend.times, backend.ux, backend.uy,
            valid_mask,
            core_valid_mask,
            x_trial, v_trial, x_mid_trial, substep_counts, valid_mask_status_flags, extension_band_sample_flags,
        )
        return
    xs, ys, zs = backend.axes
    uz = backend.uz if backend.uz is not None else np.zeros((1,) + valid_mask.shape, dtype=np.float64)
    advance_particles_3d_inplace(
        x, v, active, tau_p, flow_scale_particle, drag_scale_particle, body_scale_particle,
        float(t), float(dt_step), float(phys['flow_scale']), float(phys['drag_tau_scale']), float(phys['body_accel_scale']),
        float(body_accel[0]), float(body_accel[1]), float(body_accel[2]), float(phys['min_tau_p_s']),
        int(integrator_mode),
        int(adaptive_substep_enabled),
        float(adaptive_substep_tau_ratio),
        int(adaptive_substep_max_splits),
        xs, ys, zs, backend.times, backend.ux, backend.uy, uz,
        valid_mask,
        core_valid_mask,
        x_trial, v_trial, x_mid_trial, substep_counts, valid_mask_status_flags, extension_band_sample_flags,
    )


__all__ = (
    'CompiledRuntimeBackend',
    'CompiledRuntimeBackendLike',
    'RegularRectilinearCompiledBackend',
    'TriangleMesh2DCompiledBackend',
    'ValidMaskPrefixResolution',
    '_advance_trial_particles',
    '_compile_runtime_arrays',
    '_sample_flow_vector_at',
    '_stage_points_from_trial',
    '_stage_sample_times',
    'advance_freeflight_partial',
    'advance_freeflight_segment',
    'resolve_valid_mask_prefix',
)
