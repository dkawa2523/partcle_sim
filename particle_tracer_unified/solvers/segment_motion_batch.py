"""Deterministic batch orchestration and scalar precise fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN

from ._segment_motion_batch_backend import (
    _normalize_batch_particle_arrays,
    _trace_regular_motion_batch,
    _trace_triangle_motion_batch,
)
from ._segment_motion_batch_state import (
    _motion_batch_buffers,
    _normalize_motion_batch_state,
)
from ._segment_motion_contracts import (
    SegmentMotionBatchDestination,
    SegmentMotionBatchRequest,
    SegmentMotionRequest,
)
from ._segment_motion_scalar import SegmentMotionTrace, trace_motion_segment
from .compiled_backend_types import (
    CompiledRuntimeBackend,
    TriangleMesh2DCompiledBackend,
)
from .forces import ForceRuntimeParameters
from .integrator_common import DRAG_MODEL_NONE


@dataclass(frozen=True, slots=True)
class SegmentMotionBatchTrace:
    """O(N) batch endpoints with candidate-only full-trace replay."""

    request: SegmentMotionBatchRequest
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    midpoint_position_m: np.ndarray
    substep_count: np.ndarray
    aggregate_support_status: np.ndarray
    local_error_resolved: np.ndarray

    def particle_trace(
        self, index: int, *, minimum_substeps: int = 1
    ) -> SegmentMotionTrace:
        i = int(index)
        return trace_motion_segment(
            self.request.particle_request(i).with_minimum_substeps(minimum_substeps)
        )


def _force_runtime_needs_precise_substep(
    force_runtime: ForceRuntimeParameters | None,
) -> bool:
    if force_runtime is None:
        return False
    return (
        bool(force_runtime.thermophoresis_enabled)
        or bool(force_runtime.dielectrophoresis_enabled)
        or bool(force_runtime.lift_enabled)
        or bool(force_runtime.pressure_gradient_enabled)
        or bool(force_runtime.virtual_mass_enabled)
    )


def _advance_particles_with_precise_substeps(
    *,
    spatial_dim: int,
    backend: CompiledRuntimeBackend,
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    tau_p: np.ndarray,
    particle_diameter: np.ndarray,
    particle_density_arr: np.ndarray,
    particle_mass_arr: np.ndarray,
    dep_eps_arr: np.ndarray,
    thermo_coeff_arr: np.ndarray,
    t: float,
    dt_step: float,
    body_accel: np.ndarray,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_max_splits: int,
    x_trial: np.ndarray,
    v_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    substep_counts: np.ndarray,
    valid_mask_status_flags: np.ndarray,
    local_error_resolved: np.ndarray,
    electric_q_over_m_particle: np.ndarray | None,
    force_runtime: ForceRuntimeParameters | None,
) -> None:
    dim = int(spatial_dim)
    qom_arr = (
        None
        if electric_q_over_m_particle is None
        else np.asarray(electric_q_over_m_particle, dtype=np.float64)
    )
    for i in range(x.shape[0]):
        if not bool(active[i]):
            x_trial[i, :dim] = x[i, :dim]
            v_trial[i, :dim] = v[i, :dim]
            x_mid_trial[i, :dim] = x[i, :dim]
            substep_counts[i] = 1
            valid_mask_status_flags[i] = int(VALID_MASK_STATUS_CLEAN)
            local_error_resolved[i] = True
            continue
        qom_i = None if qom_arr is None else float(qom_arr[i])
        trace = trace_motion_segment(
            SegmentMotionRequest(
                position_m=np.asarray(x[i, :dim], dtype=np.float64),
                velocity_mps=np.asarray(v[i, :dim], dtype=np.float64),
                duration_s=float(dt_step),
                end_time_s=float(t),
                spatial_dim=dim,
                backend=backend,
                adaptive_substep_enabled=int(adaptive_substep_enabled),
                adaptive_substep_max_splits=int(adaptive_substep_max_splits),
                tau_stokes_s=float(tau_p[i]),
                particle_diameter_m=float(particle_diameter[i]),
                particle_density_kgm3=float(particle_density_arr[i]),
                particle_mass_kg=float(particle_mass_arr[i]),
                dep_particle_rel_permittivity=float(dep_eps_arr[i]),
                thermophoretic_coefficient=float(thermo_coeff_arr[i]),
                body_acceleration_mps2=np.asarray(body_accel, dtype=np.float64)[:dim],
                gas_density_kgm3=float(gas_density_kgm3),
                gas_dynamic_viscosity_Pas=float(gas_mu_pas),
                gas_temperature_K=float(gas_temperature_K),
                gas_molecular_mass_kg=float(gas_molecular_mass_kg),
                drag_model_mode=int(drag_model_mode),
                electric_q_over_m_Ckg=qom_i,
                force_runtime=force_runtime,
            )
        )
        x_trial[i, :dim] = trace.endpoint_position_m[:dim]
        v_trial[i, :dim] = trace.endpoint_velocity_mps[:dim]
        stage_arr = trace.positions_m
        midpoint_row = max(0, int(trace.substep_count) - 1)
        x_mid_trial[i, :dim] = (
            stage_arr[midpoint_row, :dim]
            if stage_arr.ndim == 2 and stage_arr.shape[0] > midpoint_row
            else x_trial[i, :dim]
        )
        substep_counts[i] = int(max(1, trace.substep_count))
        valid_mask_status_flags[i] = int(trace.aggregate_support_status)
        local_error_resolved[i] = bool(trace.local_error_resolved)


def _requires_precise_motion_batch(
    backend: CompiledRuntimeBackend,
    *,
    drag_model_mode: int,
    electric_q_over_m_particle: np.ndarray | None,
    force_runtime: ForceRuntimeParameters | None,
) -> bool:
    if (
        int(drag_model_mode) == int(DRAG_MODEL_NONE)
        or electric_q_over_m_particle is not None
        or _force_runtime_needs_precise_substep(force_runtime)
    ):
        return True
    return (
        isinstance(backend, TriangleMesh2DCompiledBackend)
        and force_runtime is not None
        and bool(force_runtime.gravity_buoyancy_enabled)
    )


def _fill_motion_batch(
    *,
    spatial_dim: int,
    compiled: CompiledRuntimeBackend,
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    tau_p: np.ndarray,
    particle_diameter: np.ndarray,
    particle_mass: np.ndarray,
    particle_density: np.ndarray | None = None,
    dep_particle_rel_permittivity: np.ndarray | None = None,
    thermophoretic_coeff: np.ndarray | None = None,
    t: float,
    dt_step: float,
    body_accel: np.ndarray,
    gas_density_kgm3: float,
    gas_mu_pas: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_max_splits: int,
    x_trial: np.ndarray,
    v_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    substep_counts: np.ndarray,
    valid_mask_status_flags: np.ndarray,
    local_error_resolved: np.ndarray,
    electric_q_over_m_particle: np.ndarray | None = None,
    force_runtime: ForceRuntimeParameters | None = None,
) -> None:
    backend = compiled
    particles = _normalize_batch_particle_arrays(
        tau_stokes_s=tau_p,
        particle_density_kgm3=particle_density,
        particle_mass_kg=particle_mass,
        dep_relative_permittivity=dep_particle_rel_permittivity,
        thermophoretic_coefficient=thermophoretic_coeff,
    )
    is_triangle_backend = isinstance(backend, TriangleMesh2DCompiledBackend)
    if is_triangle_backend and int(spatial_dim) != 2:
        raise ValueError(
            "triangle_mesh_2d backend currently supports only spatial_dim=2"
        )
    if _requires_precise_motion_batch(
        backend,
        drag_model_mode=int(drag_model_mode),
        electric_q_over_m_particle=electric_q_over_m_particle,
        force_runtime=force_runtime,
    ):
        _advance_particles_with_precise_substeps(
            spatial_dim=int(spatial_dim),
            backend=backend,
            x=x,
            v=v,
            active=active,
            tau_p=tau_p,
            particle_diameter=particle_diameter,
            particle_density_arr=particles.density_kgm3,
            particle_mass_arr=particles.mass_kg,
            dep_eps_arr=particles.dep_relative_permittivity,
            thermo_coeff_arr=particles.thermophoretic_coefficient,
            t=float(t),
            dt_step=float(dt_step),
            body_accel=body_accel,
            gas_density_kgm3=float(gas_density_kgm3),
            gas_mu_pas=float(gas_mu_pas),
            gas_temperature_K=float(gas_temperature_K),
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
            drag_model_mode=int(drag_model_mode),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            x_trial=x_trial,
            v_trial=v_trial,
            x_mid_trial=x_mid_trial,
            substep_counts=substep_counts,
            valid_mask_status_flags=valid_mask_status_flags,
            local_error_resolved=local_error_resolved,
            electric_q_over_m_particle=electric_q_over_m_particle,
            force_runtime=force_runtime,
        )
        return
    if isinstance(backend, TriangleMesh2DCompiledBackend):
        _trace_triangle_motion_batch(
            backend=backend,
            x=x,
            v=v,
            active=active,
            tau_stokes_s=tau_p,
            particle_diameter_m=particle_diameter,
            particles=particles,
            time_s=float(t),
            duration_s=float(dt_step),
            body_acceleration_mps2=body_accel,
            gas_molecular_mass_kg=float(gas_molecular_mass_kg),
            drag_model_mode=int(drag_model_mode),
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_max_splits=int(adaptive_substep_max_splits),
            x_trial=x_trial,
            v_trial=v_trial,
            x_mid_trial=x_mid_trial,
            substep_counts=substep_counts,
            valid_mask_status_flags=valid_mask_status_flags,
            local_error_resolved=local_error_resolved,
        )
        return
    _trace_regular_motion_batch(
        spatial_dim=int(spatial_dim),
        backend=backend,
        x=x,
        v=v,
        active=active,
        tau_stokes_s=tau_p,
        particle_diameter_m=particle_diameter,
        particles=particles,
        time_s=float(t),
        duration_s=float(dt_step),
        body_acceleration_mps2=body_accel,
        fallback_density_kgm3=float(gas_density_kgm3),
        fallback_viscosity_Pas=float(gas_mu_pas),
        fallback_temperature_K=float(gas_temperature_K),
        gas_molecular_mass_kg=float(gas_molecular_mass_kg),
        drag_model_mode=int(drag_model_mode),
        adaptive_substep_enabled=int(adaptive_substep_enabled),
        adaptive_substep_max_splits=int(adaptive_substep_max_splits),
        gravity_buoyancy_enabled=int(
            force_runtime is not None and bool(force_runtime.gravity_buoyancy_enabled)
        ),
        x_trial=x_trial,
        v_trial=v_trial,
        x_mid_trial=x_mid_trial,
        substep_counts=substep_counts,
        valid_mask_status_flags=valid_mask_status_flags,
        local_error_resolved=local_error_resolved,
    )


def trace_motion_batch(
    request: SegmentMotionBatchRequest,
    destination: SegmentMotionBatchDestination | None = None,
) -> SegmentMotionBatchTrace:
    """Advance all requested particles through the canonical segment engine."""

    state = _normalize_motion_batch_state(request)
    buffers = _motion_batch_buffers(state, destination)
    _fill_motion_batch(
        spatial_dim=state.spatial_dim,
        compiled=request.backend,
        x=state.position_m,
        v=state.velocity_mps,
        active=state.active,
        tau_p=np.asarray(request.tau_stokes_s, dtype=np.float64),
        particle_diameter=np.asarray(request.particle_diameter_m, dtype=np.float64),
        particle_mass=np.asarray(request.particle_mass_kg, dtype=np.float64),
        particle_density=np.asarray(request.particle_density_kgm3, dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray(
            request.dep_particle_rel_permittivity,
            dtype=np.float64,
        ),
        thermophoretic_coeff=np.asarray(
            request.thermophoretic_coefficient, dtype=np.float64
        ),
        t=float(request.end_time_s),
        dt_step=float(request.duration_s),
        body_accel=np.asarray(request.body_acceleration_mps2, dtype=np.float64),
        gas_density_kgm3=float(request.gas_density_kgm3),
        gas_mu_pas=float(request.gas_dynamic_viscosity_Pas),
        gas_temperature_K=float(request.gas_temperature_K),
        gas_molecular_mass_kg=float(request.gas_molecular_mass_kg),
        drag_model_mode=int(request.drag_model_mode),
        adaptive_substep_enabled=int(request.adaptive_substep_enabled),
        adaptive_substep_max_splits=int(request.adaptive_substep_max_splits),
        x_trial=buffers.endpoint_position_m,
        v_trial=buffers.endpoint_velocity_mps,
        x_mid_trial=buffers.midpoint_position_m,
        substep_counts=buffers.substep_count,
        valid_mask_status_flags=buffers.aggregate_support_status,
        local_error_resolved=buffers.local_error_resolved,
        electric_q_over_m_particle=request.electric_q_over_m_Ckg,
        force_runtime=request.force_runtime,
    )

    return SegmentMotionBatchTrace(
        request=request,
        endpoint_position_m=buffers.endpoint_position_m,
        endpoint_velocity_mps=buffers.endpoint_velocity_mps,
        midpoint_position_m=buffers.midpoint_position_m,
        substep_count=buffers.substep_count,
        aggregate_support_status=buffers.aggregate_support_status,
        local_error_resolved=buffers.local_error_resolved,
    )


__all__ = ("SegmentMotionBatchTrace", "trace_motion_batch")
