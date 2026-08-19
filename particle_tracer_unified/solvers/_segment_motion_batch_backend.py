"""Compiled regular-grid and triangle-mesh motion batch adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compiled_backend_types import (
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
)
from .kernel2d_numba import trace_regular_2d_batch_inplace
from .kernel2d_triangle_mesh_numba import trace_triangle_2d_batch_inplace
from .kernel3d_numba import trace_regular_3d_batch_inplace


@dataclass(frozen=True, slots=True)
class _BatchParticleArrays:
    density_kgm3: np.ndarray
    mass_kg: np.ndarray
    dep_relative_permittivity: np.ndarray
    thermophoretic_coefficient: np.ndarray


def _normalize_batch_particle_arrays(
    *,
    tau_stokes_s: np.ndarray,
    particle_density_kgm3: np.ndarray | None,
    particle_mass_kg: np.ndarray,
    dep_relative_permittivity: np.ndarray | None,
    thermophoretic_coefficient: np.ndarray | None,
) -> _BatchParticleArrays:
    if particle_density_kgm3 is None:
        density = np.full_like(tau_stokes_s, np.nan, dtype=np.float64)
    else:
        density = np.asarray(particle_density_kgm3, dtype=np.float64)
    mass = np.asarray(particle_mass_kg, dtype=np.float64)
    if mass.shape != np.asarray(tau_stokes_s).shape:
        raise ValueError("particle_mass must have the same shape as tau_p")
    if np.any(~np.isfinite(mass) | (mass <= 0.0)):
        raise ValueError("particle mass_kg must be finite and > 0")
    dep_permittivity = (
        np.ones_like(tau_stokes_s, dtype=np.float64) * np.nan
        if dep_relative_permittivity is None
        else np.asarray(dep_relative_permittivity, dtype=np.float64)
    )
    thermophoretic = (
        np.ones_like(tau_stokes_s, dtype=np.float64) * np.nan
        if thermophoretic_coefficient is None
        else np.asarray(thermophoretic_coefficient, dtype=np.float64)
    )
    return _BatchParticleArrays(
        density_kgm3=density,
        mass_kg=mass,
        dep_relative_permittivity=dep_permittivity,
        thermophoretic_coefficient=thermophoretic,
    )


def _trace_triangle_motion_batch(
    *,
    backend: TriangleMesh2DCompiledBackend,
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    tau_stokes_s: np.ndarray,
    particle_diameter_m: np.ndarray,
    particles: _BatchParticleArrays,
    time_s: float,
    duration_s: float,
    body_acceleration_mps2: np.ndarray,
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
) -> None:
    extra_acceleration = np.zeros((x.shape[0], 2), dtype=np.float64)
    acceleration_shape = np.asarray(backend.accel_shape, dtype=np.int32)
    trace_triangle_2d_batch_inplace(
        x,
        v,
        active,
        tau_stokes_s,
        particle_diameter_m,
        particles.density_kgm3,
        particles.mass_kg,
        float(time_s),
        float(duration_s),
        float(body_acceleration_mps2[0]),
        float(body_acceleration_mps2[1]),
        float(gas_molecular_mass_kg),
        int(drag_model_mode),
        int(adaptive_substep_enabled),
        int(adaptive_substep_max_splits),
        np.asarray(backend.mesh_vertices, dtype=np.float64),
        np.asarray(backend.mesh_triangles, dtype=np.int32),
        np.asarray(backend.accel_origin, dtype=np.float64),
        np.asarray(backend.accel_cell_size, dtype=np.float64),
        int(acceleration_shape[0]),
        int(acceleration_shape[1]),
        np.asarray(backend.accel_cell_offsets, dtype=np.int32),
        np.asarray(backend.accel_triangle_indices, dtype=np.int32),
        float(backend.support_tolerance_m),
        np.asarray(backend.times, dtype=np.float64),
        np.asarray(backend.ux, dtype=np.float64),
        np.asarray(backend.uy, dtype=np.float64),
        np.asarray(backend.gas_density, dtype=np.float64),
        np.asarray(backend.gas_mu, dtype=np.float64),
        np.asarray(backend.gas_temperature, dtype=np.float64),
        np.asarray(extra_acceleration[:, 0], dtype=np.float64),
        np.asarray(extra_acceleration[:, 1], dtype=np.float64),
        x_trial,
        v_trial,
        x_mid_trial,
        substep_counts,
        valid_mask_status_flags,
        local_error_resolved,
        axisymmetric_rz=int(str(backend.coordinate_system) == "axisymmetric_rz"),
    )


def _trace_regular_motion_batch(
    *,
    spatial_dim: int,
    backend: RegularRectilinearCompiledBackend,
    x: np.ndarray,
    v: np.ndarray,
    active: np.ndarray,
    tau_stokes_s: np.ndarray,
    particle_diameter_m: np.ndarray,
    particles: _BatchParticleArrays,
    time_s: float,
    duration_s: float,
    body_acceleration_mps2: np.ndarray,
    fallback_density_kgm3: float,
    fallback_viscosity_Pas: float,
    fallback_temperature_K: float,
    gas_molecular_mass_kg: float,
    drag_model_mode: int,
    adaptive_substep_enabled: int,
    adaptive_substep_max_splits: int,
    gravity_buoyancy_enabled: int,
    x_trial: np.ndarray,
    v_trial: np.ndarray,
    x_mid_trial: np.ndarray,
    substep_counts: np.ndarray,
    valid_mask_status_flags: np.ndarray,
    local_error_resolved: np.ndarray,
) -> None:
    extra_acceleration = np.zeros(
        (x.shape[0], int(spatial_dim)),
        dtype=np.float64,
    )
    valid_mask = np.asarray(backend.valid_mask, dtype=bool)
    if int(spatial_dim) == 2:
        axis_x, axis_y = backend.axes
        trace_regular_2d_batch_inplace(
            x,
            v,
            active,
            tau_stokes_s,
            particle_diameter_m,
            particles.density_kgm3,
            particles.mass_kg,
            float(time_s),
            float(duration_s),
            float(body_acceleration_mps2[0]),
            float(body_acceleration_mps2[1]),
            float(fallback_density_kgm3),
            float(fallback_viscosity_Pas),
            float(fallback_temperature_K),
            float(gas_molecular_mass_kg),
            int(drag_model_mode),
            int(adaptive_substep_enabled),
            int(adaptive_substep_max_splits),
            axis_x,
            axis_y,
            backend.times,
            backend.ux,
            backend.uy,
            np.asarray(extra_acceleration[:, 0], dtype=np.float64),
            np.asarray(extra_acceleration[:, 1], dtype=np.float64),
            int(gravity_buoyancy_enabled),
            backend.gas_density,
            backend.gas_mu,
            backend.gas_temperature,
            valid_mask,
            x_trial,
            v_trial,
            x_mid_trial,
            substep_counts,
            valid_mask_status_flags,
            local_error_resolved,
            axisymmetric_rz=int(str(backend.coordinate_system) == "axisymmetric_rz"),
        )
        return
    axis_x, axis_y, axis_z = backend.axes
    velocity_z = (
        backend.uz
        if backend.uz is not None
        else np.zeros((1, *valid_mask.shape), dtype=np.float64)
    )
    trace_regular_3d_batch_inplace(
        x,
        v,
        active,
        tau_stokes_s,
        particle_diameter_m,
        particles.density_kgm3,
        particles.mass_kg,
        float(time_s),
        float(duration_s),
        float(body_acceleration_mps2[0]),
        float(body_acceleration_mps2[1]),
        float(body_acceleration_mps2[2]),
        float(fallback_density_kgm3),
        float(fallback_viscosity_Pas),
        float(fallback_temperature_K),
        float(gas_molecular_mass_kg),
        int(drag_model_mode),
        int(adaptive_substep_enabled),
        int(adaptive_substep_max_splits),
        axis_x,
        axis_y,
        axis_z,
        backend.times,
        backend.ux,
        backend.uy,
        velocity_z,
        np.asarray(extra_acceleration[:, 0], dtype=np.float64),
        np.asarray(extra_acceleration[:, 1], dtype=np.float64),
        np.asarray(extra_acceleration[:, 2], dtype=np.float64),
        int(gravity_buoyancy_enabled),
        backend.gas_density,
        backend.gas_mu,
        backend.gas_temperature,
        valid_mask,
        x_trial,
        v_trial,
        x_mid_trial,
        substep_counts,
        valid_mask_status_flags,
        local_error_resolved,
    )
