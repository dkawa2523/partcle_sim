"""Validate physics and build immutable resources before the solver loop."""

from __future__ import annotations

import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import numpy as np

from particle_tracer_unified.core._triangle_surface import (
    GeometrySurfaces3D,
    build_geometry_surfaces_3d,
)
from particle_tracer_unified.core.boundary_service import (
    build_boundary_service,
    runtime_bounds,
)
from particle_tracer_unified.core.coordinate_systems import (
    canonicalize_axisymmetric_rz_state,
)
from particle_tracer_unified.core.datamodel import SolverContext
from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryQuery

from ._particle_geometry import physical_sphere_diameter_m
from ._runtime_execution_context import RunExecutionContext
from ._runtime_outcome import append_snapshot
from .base_field_sampling import compiled_gas_property_report
from .charge_model import (
    charge_model_report,
    validate_charge_model_support,
)
from .drag_models import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_NONE,
    drag_model_structure_from_mode,
    epstein_relaxation_time,
    stokes_relaxation_time,
)
from .field_compilation import compile_runtime_backend
from .forces import (
    ForceRuntimeParameters,
    force_catalog_summary,
    force_runtime_parameters_summary,
)
from .output_buffers import DebugBuffers
from .plasma_background import plasma_background_report
from .runtime_plan import SolverPlan
from .runtime_setup import RuntimeOptions
from .runtime_state import initialize_solver_state
from .stochastic_motion import stochastic_motion_report


def _required_runtime_gas_properties(
    plan: SolverPlan,
    forces: ForceRuntimeParameters,
    *,
    stochastic_enabled: bool,
) -> set[str]:
    required = set(
        drag_model_structure_from_mode(int(plan.drag_model_mode)).gas_requirements
    )
    if stochastic_enabled:
        required.add("temperature_K")
    if forces.thermophoresis_enabled:
        required.update({"temperature_K", "dynamic_viscosity_Pas", "density_kgm3"})
    if forces.lift_enabled:
        required.update({"dynamic_viscosity_Pas", "density_kgm3"})
    if (
        forces.pressure_gradient_enabled
        or forces.virtual_mass_enabled
        or forces.gravity_buoyancy_enabled
    ):
        required.add("density_kgm3")
    return required


def _runtime_physics(
    runtime,
    plan: SolverPlan,
    *,
    force_runtime: ForceRuntimeParameters | None = None,
    stochastic_enabled: bool = False,
) -> Mapping[str, float]:
    """Resolve and validate immutable gas properties once per run."""

    density = float(runtime.gas.density_kgm3)
    viscosity = float(runtime.gas.dynamic_viscosity_Pas)
    temperature = float(runtime.gas.temperature)
    molecular_mass_amu = float(runtime.gas.molecular_mass_amu)
    required_names = _required_runtime_gas_properties(
        plan,
        force_runtime or ForceRuntimeParameters(),
        stochastic_enabled=stochastic_enabled,
    )
    values = {
        "dynamic_viscosity_Pas": viscosity,
        "density_kgm3": density,
        "temperature_K": temperature,
        "molecular_mass_amu": molecular_mass_amu,
    }
    for name in sorted(required_names):
        value = float(values[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"active physics with drag_model={plan.drag_model_name} requires "
                f"finite gas.{name} > 0"
            )
    return MappingProxyType(
        {
            "gas_density_kgm3": density,
            "gas_mu_pas": viscosity,
            "gas_temperature_K": temperature,
            "gas_molecular_mass_kg": molecular_mass_amu * 1.66053906660e-27,
        }
    )


def _require_particle_density_for_displaced_fluid_forces(
    particle_density: np.ndarray,
    particle_id: np.ndarray,
    force_runtime: ForceRuntimeParameters,
) -> None:
    features = []
    if bool(force_runtime.pressure_gradient_enabled):
        features.append("pressure_gradient")
    if bool(force_runtime.virtual_mass_enabled):
        features.append("virtual_mass")
    if bool(force_runtime.gravity_buoyancy_enabled):
        features.append("gravity_buoyancy")
    if not features:
        return
    density = np.asarray(particle_density, dtype=np.float64)
    invalid = np.flatnonzero(~np.isfinite(density) | (density <= 0.0))
    if invalid.size:
        ids = np.asarray(particle_id, dtype=np.int64)[invalid[:12]].tolist()
        raise ValueError(
            "explicit positive particle density_kgm3 is required for "
            + ", ".join(features)
            + f"; invalid particle IDs: {ids}"
        )


def _base_relaxation_times(
    *,
    particle_mass: np.ndarray,
    particle_diameter: np.ndarray,
    plan: SolverPlan,
    physics: Mapping[str, float],
) -> np.ndarray:
    mode = int(plan.drag_model_mode)
    count = int(np.asarray(particle_mass).size)
    if mode == int(DRAG_MODEL_NONE):
        return np.full(count, np.inf, dtype=np.float64)
    if mode == int(DRAG_MODEL_EPSTEIN):
        return np.asarray(
            [
                epstein_relaxation_time(
                    mass_kg=float(particle_mass[index]),
                    gas_density_kgm3=float(physics["gas_density_kgm3"]),
                    gas_temperature_K=float(physics["gas_temperature_K"]),
                    particle_diameter_m=float(particle_diameter[index]),
                    gas_molecular_mass_kg=float(physics["gas_molecular_mass_kg"]),
                )
                for index in range(count)
            ],
            dtype=np.float64,
        )
    return np.asarray(
        [
            stokes_relaxation_time(
                mass_kg=float(particle_mass[index]),
                gas_mu_pas=float(physics["gas_mu_pas"]),
                particle_diameter_m=float(particle_diameter[index]),
            )
            for index in range(count)
        ],
        dtype=np.float64,
    )


def _prepare_triangle_surfaces(runtime, spatial_dim: int) -> GeometrySurfaces3D | None:
    if int(spatial_dim) != 3:
        return None
    if runtime.geometry_provider is None:
        raise ValueError("3D solver requires geometry_provider")
    geometry = runtime.geometry_provider.geometry
    if geometry.boundary_triangles is None:
        raise ValueError(
            "3D solver requires geometry.boundary_triangles as geometry truth source"
        )
    return build_geometry_surfaces_3d(geometry)


def _prepare_triangle_surface(runtime, spatial_dim: int) -> TriangleSurface3D | None:
    """Compatibility accessor for the solver's collision surface."""

    surfaces = _prepare_triangle_surfaces(runtime, spatial_dim)
    return None if surfaces is None else surfaces.collision


def _validate_coordinate_system(
    context: SolverContext,
    *,
    plan: SolverPlan,
    options: RuntimeOptions,
    spatial_dim: int,
) -> None:
    if (
        str(getattr(context, "coordinate_system", "")).strip().lower()
        != "axisymmetric_rz"
    ):
        return
    if int(spatial_dim) != 2:
        raise ValueError(
            "axisymmetric_rz is restricted to no-swirl (r,z,v_r,v_z) 2D dynamics"
        )
    if bool(options.stochastic_motion.enabled):
        raise ValueError("axisymmetric_rz does not support Brownian motion")
    if bool(options.force_runtime.lift_enabled):
        raise ValueError("axisymmetric_rz does not support the Cartesian lift model")


def _initialize_execution_diagnostics(
    context: SolverContext,
    plan: SolverPlan,
    compiled,
    physics: Mapping[str, float],
    state,
) -> None:
    diagnostics = state.collision_diagnostics
    diagnostics["boundary_broad_phase_enabled"] = int(
        bool(plan.boundary_broad_phase_enabled)
    )
    diagnostics["output_mode"] = str(plan.output.mode)
    diagnostics["output_debug_enabled"] = int(bool(plan.output.is_debug))
    if not bool(plan.output.is_debug):
        return
    options = context.options
    diagnostics["field_sampling_s"] = 0.0
    diagnostics["field_sample_point_count"] = 0
    diagnostics["field_sample_call_count"] = 0
    diagnostics["acceleration_source"] = str(
        getattr(compiled, "acceleration_source", "none")
    )
    diagnostics["acceleration_quantity_names"] = list(
        getattr(compiled, "acceleration_quantity_names", ())
    )
    diagnostics["electric_field_names"] = list(
        getattr(compiled, "electric_field_names", ())
    )
    diagnostics["drag_gas_properties"] = dict(
        compiled_gas_property_report(
            compiled,
            fallback_density_kgm3=float(physics["gas_density_kgm3"]),
            fallback_mu_pas=float(physics["gas_mu_pas"]),
            fallback_temperature_K=float(physics["gas_temperature_K"]),
            drag_model_name=str(plan.drag_model_name),
        )
    )
    diagnostics["field_backend_diagnostics"] = {
        "backend_kind": str(getattr(compiled, "backend_kind", "")),
        "gas_density_source": str(
            getattr(compiled, "gas_density_source", "unavailable")
        ),
        "gas_mu_source": str(getattr(compiled, "gas_mu_source", "unavailable")),
        "gas_temperature_source": str(
            getattr(compiled, "gas_temperature_source", "unavailable")
        ),
        "triangle_gradient_sources": dict(
            getattr(compiled, "triangle_gradient_sources", {})
        ),
    }
    diagnostics["collision_boundary_geometry"] = "linear_segment_or_triangle_boundary"
    diagnostics["contact_tangent_model"] = "custom_relaxation_contact_sliding"
    diagnostics["force_catalog"] = force_catalog_summary(context.force_catalog)
    diagnostics["force_runtime"] = force_runtime_parameters_summary(
        options.force_runtime
    )
    diagnostics["stochastic_motion"] = stochastic_motion_report(
        options.stochastic_motion
    )
    diagnostics["plasma_background"] = plasma_background_report(
        options.plasma_background
    )
    diagnostics["charge_model"] = charge_model_report(
        options.charge_model,
        options.plasma_background,
    )


def prepare_runtime_execution(
    context: SolverContext,
    *,
    spatial_dim: int,
    plan: SolverPlan,
    debug_buffers: DebugBuffers | None,
) -> RunExecutionContext:
    """Resolve all immutable run resources before entering the step loop."""

    setup_started_s = time.perf_counter()
    dim = int(spatial_dim)
    options = context.options
    _validate_coordinate_system(
        context,
        plan=plan,
        options=options,
        spatial_dim=dim,
    )
    state = initialize_solver_state(
        particles=context.particles,
        plan=plan,
        debug_buffers=debug_buffers,
        spatial_dim=dim,
    )
    if str(context.coordinate_system) == "axisymmetric_rz":
        state.x[:], state.v[:] = canonicalize_axisymmetric_rz_state(state.x, state.v)
    static = state.static
    mins, maxs = runtime_bounds(context)
    compiled = compile_runtime_backend(
        context,
        dim,
        enable_electric=(
            bool(context.force_catalog.enabled("electric"))
            if context.force_catalog is not None
            else True
        ),
        force_runtime=options.force_runtime,
    )
    validate_charge_model_support(
        options.charge_model,
        context,
        compiled,
        dim,
        options.plasma_background,
    )
    triangle_surfaces_3d = _prepare_triangle_surfaces(context, dim)
    triangle_surface_3d = (
        None if triangle_surfaces_3d is None else triangle_surfaces_3d.collision
    )
    physics = _runtime_physics(
        context,
        plan,
        force_runtime=options.force_runtime,
        stochastic_enabled=bool(options.stochastic_motion.enabled),
    )
    body_acceleration = np.asarray(plan.body_acceleration_mps2, dtype=np.float64)
    if body_acceleration.size < dim:
        body_acceleration = np.pad(
            body_acceleration,
            (0, dim - body_acceleration.size),
            constant_values=0.0,
        )
    particle_density = static.density_kgm3
    particle_physical_diameter = physical_sphere_diameter_m(
        mass_kg=static.mass_kg,
        density_kgm3=particle_density,
        drag_diameter_m=static.diameter_m,
    )
    _require_particle_density_for_displaced_fluid_forces(
        particle_density,
        static.particle_id,
        options.force_runtime,
    )
    tau_p = _base_relaxation_times(
        particle_mass=static.mass_kg,
        particle_diameter=static.diameter_m,
        plan=plan,
        physics=physics,
    )
    _initialize_execution_diagnostics(context, plan, compiled, physics, state)
    if debug_buffers is not None:
        append_snapshot(
            debug_buffers.trajectory_positions,
            debug_buffers.save_frames,
            save_index=0,
            t=0.0,
            position=state.x,
        )
    boundary_service = build_boundary_service(
        context,
        spatial_dim=dim,
        on_boundary_tol_m=float(plan.boundary.classification_tolerance_m),
        triangle_surface_3d=triangle_surface_3d,
        containment_triangle_surface_3d=(
            None if triangle_surfaces_3d is None else triangle_surfaces_3d.containment
        ),
    )
    return RunExecutionContext(
        context=context,
        plan=plan,
        options=options,
        state=state,
        compiled=compiled,
        boundary_service=cast(BoundaryQuery[TriangleSurface3D], boundary_service),
        spatial_dim=dim,
        mins=np.asarray(mins, dtype=np.float64),
        maxs=np.asarray(maxs, dtype=np.float64),
        physics=physics,
        body_acceleration_mps2=body_acceleration,
        tau_p=np.asarray(tau_p, dtype=np.float64),
        particle_mass=static.mass_kg,
        particle_diameter=static.diameter_m,
        particle_physical_diameter=particle_physical_diameter,
        particle_density=particle_density,
        particle_id=static.particle_id,
        dep_particle_rel_permittivity=static.dep_particle_rel_permittivity,
        thermophoretic_coeff=static.thermophoretic_coeff,
        setup_started_s=float(setup_started_s),
        loop_setup_done_s=float(time.perf_counter()),
    )


__all__ = ("prepare_runtime_execution",)
