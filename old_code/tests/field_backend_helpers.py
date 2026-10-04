from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GeometryND,
    GeometryProviderND,
    QuantitySeriesND,
    RegularFieldND,
)
from particle_tracer_unified.solvers.field_compilation import (
    compile_runtime_backend as _compile_runtime_arrays,
)
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    trace_motion_batch,
)


def advance_motion_batch_into(**kwargs):
    """Exercise the public batch contract while retaining concise fixtures."""

    position = np.asarray(kwargs["x"], dtype=np.float64)
    particle_count = int(position.shape[0])
    particle_density = kwargs.get("particle_density")
    if particle_density is None:
        particle_density = np.full(particle_count, np.nan, dtype=np.float64)
    dep_permittivity = kwargs.get("dep_particle_rel_permittivity")
    if dep_permittivity is None:
        dep_permittivity = np.full(particle_count, np.nan, dtype=np.float64)
    thermophoretic = kwargs.get("thermophoretic_coeff")
    if thermophoretic is None:
        thermophoretic = np.full(particle_count, np.nan, dtype=np.float64)
    physics = kwargs.get("phys", {})
    trace = trace_motion_batch(
        SegmentMotionBatchRequest(
            position_m=position,
            velocity_mps=np.asarray(kwargs["v"], dtype=np.float64),
            active=np.asarray(kwargs["active"], dtype=bool),
            tau_stokes_s=np.asarray(kwargs["tau_p"], dtype=np.float64),
            particle_diameter_m=np.asarray(
                kwargs["particle_diameter"], dtype=np.float64
            ),
            particle_density_kgm3=np.asarray(particle_density, dtype=np.float64),
            particle_mass_kg=np.asarray(kwargs["particle_mass"], dtype=np.float64),
            dep_particle_rel_permittivity=np.asarray(
                dep_permittivity, dtype=np.float64
            ),
            thermophoretic_coefficient=np.asarray(thermophoretic, dtype=np.float64),
            end_time_s=float(kwargs["t"]),
            duration_s=float(kwargs["dt_step"]),
            spatial_dim=int(kwargs["spatial_dim"]),
            backend=kwargs["compiled"],
            body_acceleration_mps2=np.asarray(kwargs["body_accel"], dtype=np.float64),
            gas_density_kgm3=float(kwargs["gas_density_kgm3"]),
            gas_dynamic_viscosity_Pas=float(kwargs["gas_mu_pas"]),
            gas_temperature_K=float(physics.get("gas_temperature_K", np.nan)),
            gas_molecular_mass_kg=float(physics.get("gas_molecular_mass_kg", np.nan)),
            drag_model_mode=int(kwargs["drag_model_mode"]),
            adaptive_substep_enabled=int(kwargs["adaptive_substep_enabled"]),
            adaptive_substep_max_splits=int(kwargs["adaptive_substep_max_splits"]),
            electric_q_over_m_Ckg=kwargs.get("electric_q_over_m_particle"),
            force_runtime=kwargs.get("force_runtime"),
        )
    )
    kwargs["x_trial"][:] = trace.endpoint_position_m
    kwargs["v_trial"][:] = trace.endpoint_velocity_mps
    kwargs["x_mid_trial"][:] = trace.midpoint_position_m
    kwargs["substep_counts"][:] = trace.substep_count
    kwargs["valid_mask_status_flags"][:] = trace.aggregate_support_status
    return trace


def regular_axes(spatial_dim: int = 2) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(int(spatial_dim))
    )


def regular_valid_mask(spatial_dim: int = 2, *, fill: bool = True) -> np.ndarray:
    return np.full(tuple(3 for _ in range(int(spatial_dim))), bool(fill), dtype=bool)


def regular_field_provider(
    axes: tuple[np.ndarray, ...],
    valid_mask: np.ndarray,
    quantities: Mapping[str, np.ndarray],
) -> FieldProviderND:
    series = {
        name: QuantitySeriesND(
            name=name,
            unit="",
            times=np.asarray([0.0], dtype=np.float64),
            data=np.asarray(values, dtype=np.float64),
        )
        for name, values in quantities.items()
    }
    field = RegularFieldND(
        spatial_dim=len(axes),
        coordinate_system="cartesian_xy" if len(axes) == 2 else "cartesian_xyz",
        axis_names=tuple("xyz"[: len(axes)]),
        axes=tuple(np.asarray(axis, dtype=np.float64) for axis in axes),
        quantities=series,
        valid_mask=np.asarray(valid_mask, dtype=bool),
        time_mode="steady",
        metadata={"provider_kind": "precomputed_npz"},
    )
    return FieldProviderND(field=field, kind="precomputed_npz")


def geometry_provider(
    axes: tuple[np.ndarray, ...],
    valid_mask: np.ndarray,
    sdf: np.ndarray,
    normal_components: tuple[np.ndarray, ...],
) -> GeometryProviderND:
    geometry = GeometryND(
        spatial_dim=len(axes),
        coordinate_system="cartesian_xy" if len(axes) == 2 else "cartesian_xyz",
        axes=tuple(np.asarray(axis, dtype=np.float64) for axis in axes),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        sdf=np.asarray(sdf, dtype=np.float64),
        normal_components=tuple(
            np.asarray(value, dtype=np.float64) for value in normal_components
        ),
        nearest_boundary_part_id_map=np.ones_like(valid_mask, dtype=np.int32),
    )
    return GeometryProviderND(geometry=geometry, kind="synthetic")


def mismatched_velocity_time_axes_provider() -> FieldProviderND:
    axes = regular_axes(2)
    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        quantities={
            "ux": QuantitySeriesND(
                name="ux",
                unit="m/s",
                times=np.asarray([0.0, 1.0]),
                data=np.zeros((2, 3, 3)),
            ),
            "uy": QuantitySeriesND(
                name="uy",
                unit="m/s",
                times=np.asarray([0.0, 0.5, 1.0]),
                data=np.zeros((3, 3, 3)),
            ),
        },
        valid_mask=regular_valid_mask(2),
        time_mode="transient",
        metadata={"provider_kind": "precomputed_npz"},
    )
    return FieldProviderND(field=field, kind="precomputed_npz")


def write_triangle_mesh_field(path: Path) -> Path:
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    ux_t0 = vertices[:, 0] + 2.0 * vertices[:, 1]
    uy_t0 = 3.0 * vertices[:, 0] - vertices[:, 1]
    np.savez_compressed(
        path,
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        times=times,
        ux=np.stack([ux_t0, ux_t0 + 1.0]),
        uy=np.stack([uy_t0, uy_t0 - 0.5]),
        mu=np.stack([np.full(vertices.shape[0], 1.8e-5) for _ in times]),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "provider_kind": "precomputed_triangle_mesh_npz",
                    "field_backend_kind": "triangle_mesh_2d",
                }
            )
        ),
    )
    return path


def adaptive_substep_count(drag_model_mode: int) -> int:
    axes = (
        np.asarray([0.0, 500.0, 1000.0]),
        np.asarray([0.0, 500.0, 1000.0]),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    field = regular_field_provider(
        axes,
        valid_mask,
        quantities={
            "ux": np.zeros((3, 3)),
            "uy": np.zeros((3, 3)),
            "rho_g": np.full((3, 3), 10.0),
            "mu": np.full((3, 3), 1.0e-3),
            "T": np.full((3, 3), 300.0),
        },
    )
    geometry = geometry_provider(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3)),
        normal_components=(np.zeros((3, 3)), np.ones((3, 3))),
    )
    compiled = _compile_runtime_arrays(
        SimpleNamespace(
            geometry_provider=geometry,
            field_provider=field,
            gas=SimpleNamespace(
                density_kgm3=10.0, dynamic_viscosity_Pas=1.0e-3, temperature=300.0
            ),
        ),
        spatial_dim=2,
    )
    density = np.asarray([1000.0])
    diameter = np.asarray([1.0])
    mass = 3.0 * np.pi * 1.0e-3 * diameter
    substeps = np.zeros(1, dtype=np.int32)
    x = np.asarray([[500.0, 500.0]])
    # Re_p=100 keeps the Schiller--Naumann fixture inside its published
    # Re_p<800 range while still reaching the configured 16-substep cap.
    v = np.asarray([[0.01, 0.0]])

    advance_motion_batch_into(
        spatial_dim=2,
        compiled=compiled,
        x=x,
        v=v,
        active=np.asarray([True]),
        tau_p=np.asarray([1.0]),
        particle_diameter=diameter,
        particle_mass=mass,
        particle_density=density,
        dep_particle_rel_permittivity=np.asarray([np.nan]),
        thermophoretic_coeff=np.asarray([np.nan]),
        t=1.0,
        dt_step=1.0,
        phys={},
        body_accel=np.zeros(2),
        gas_density_kgm3=10.0,
        gas_mu_pas=1.0e-3,
        drag_model_mode=int(drag_model_mode),
        adaptive_substep_enabled=1,
        adaptive_substep_max_splits=4,
        x_trial=np.zeros_like(x),
        v_trial=np.zeros_like(v),
        x_mid_trial=np.zeros_like(x),
        substep_counts=substeps,
        valid_mask_status_flags=np.zeros(1, dtype=np.uint8),
    )
    return int(substeps[0])


__all__ = (
    "adaptive_substep_count",
    "advance_motion_batch_into",
    "geometry_provider",
    "mismatched_velocity_time_axes_provider",
    "regular_axes",
    "regular_field_provider",
    "regular_valid_mask",
    "write_triangle_mesh_field",
)
