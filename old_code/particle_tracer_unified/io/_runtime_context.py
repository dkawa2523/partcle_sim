"""Compile resolved adapter inputs into the immutable solver context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import numpy as np

from particle_tracer_unified.configuration import (
    ChargeConfig,
    RunConfig,
    StochasticConfig,
)
from particle_tracer_unified.core.boundary_numerics import resolve_boundary_numerics
from particle_tracer_unified.core.catalogs import build_wall_catalog
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GasProperties,
    GeometryProviderND,
    ParticleTable,
    QuantitySeriesND,
    RegularFieldND,
    SolverContext,
    WallCatalog,
    WallPartModel,
    immutable_mapping,
    readonly_array,
)
from particle_tracer_unified.solvers.charge_model import ChargeModelConfig
from particle_tracer_unified.solvers.forces import (
    compile_force_runtime_parameters,
    resolve_force_catalog,
)
from particle_tracer_unified.solvers.plasma_background import (
    PlasmaBackgroundConfig,
    PreparedPlasmaBackground,
    prepare_plasma_background,
)
from particle_tracer_unified.solvers.runtime_plan import build_solver_plan
from particle_tracer_unified.solvers.runtime_setup import RuntimeOptions
from particle_tracer_unified.solvers.stochastic_motion import StochasticMotionConfig

from ._runtime_adapter import ResolvedAdapterInputs
from .physics_compatibility import validate_coordinate_force_compatibility


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], immutable_mapping(value))


def _optional_array(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else readonly_array(value)


def _readonly_particles(particles: ParticleTable) -> ParticleTable:
    return replace(
        particles,
        particle_id=readonly_array(particles.particle_id),
        position=readonly_array(particles.position),
        velocity=readonly_array(particles.velocity),
        release_time=readonly_array(particles.release_time),
        mass=readonly_array(particles.mass),
        diameter=readonly_array(particles.diameter),
        density=readonly_array(particles.density),
        charge=readonly_array(particles.charge),
        source_part_id=readonly_array(particles.source_part_id),
        material_id=readonly_array(particles.material_id),
        dep_particle_rel_permittivity=readonly_array(
            particles.dep_particle_rel_permittivity
        ),
        thermophoretic_coeff=readonly_array(particles.thermophoretic_coeff),
        metadata=_metadata(particles.metadata),
    )


def _readonly_quantity(series: QuantitySeriesND) -> QuantitySeriesND:
    return replace(
        series,
        times=readonly_array(series.times),
        data=readonly_array(series.data),
        metadata=_metadata(series.metadata),
    )


def _readonly_quantities(
    values: Mapping[str, QuantitySeriesND],
) -> dict[str, QuantitySeriesND]:
    return cast(
        dict[str, QuantitySeriesND],
        immutable_mapping(
            {name: _readonly_quantity(series) for name, series in values.items()}
        ),
    )


def _readonly_field_provider(provider: FieldProviderND) -> FieldProviderND:
    field = provider.field
    quantities = _readonly_quantities(field.quantities)
    if isinstance(field, RegularFieldND):
        frozen_field = replace(
            field,
            axes=tuple(readonly_array(axis) for axis in field.axes),
            quantities=quantities,
            valid_mask=readonly_array(field.valid_mask),
            support_phi=_optional_array(field.support_phi),
            core_valid_mask=_optional_array(field.core_valid_mask),
            metadata=_metadata(field.metadata),
        )
    else:
        frozen_field = replace(
            field,
            mesh_vertices=readonly_array(field.mesh_vertices),
            mesh_triangles=readonly_array(field.mesh_triangles),
            quantities=quantities,
            accel_origin=readonly_array(field.accel_origin),
            accel_cell_size=readonly_array(field.accel_cell_size),
            accel_cell_offsets=readonly_array(field.accel_cell_offsets),
            accel_triangle_indices=readonly_array(field.accel_triangle_indices),
            metadata=_metadata(field.metadata),
        )
    return replace(provider, field=frozen_field)


def _readonly_geometry_provider(provider: GeometryProviderND) -> GeometryProviderND:
    geometry = provider.geometry
    frozen_geometry = replace(
        geometry,
        axes=tuple(readonly_array(axis) for axis in geometry.axes),
        valid_mask=readonly_array(geometry.valid_mask),
        sdf=readonly_array(geometry.sdf),
        normal_components=tuple(
            readonly_array(component) for component in geometry.normal_components
        ),
        nearest_boundary_part_id_map=readonly_array(
            geometry.nearest_boundary_part_id_map
        ),
        metadata=_metadata(geometry.metadata),
        boundary_edges=_optional_array(geometry.boundary_edges),
        boundary_edge_part_ids=_optional_array(geometry.boundary_edge_part_ids),
        boundary_loops_2d=tuple(
            readonly_array(loop) for loop in geometry.boundary_loops_2d
        ),
        boundary_triangles=_optional_array(geometry.boundary_triangles),
        boundary_triangle_part_ids=_optional_array(geometry.boundary_triangle_part_ids),
        containment_boundary_triangles=_optional_array(
            geometry.containment_boundary_triangles
        ),
    )
    return replace(provider, geometry=frozen_geometry)


def _readonly_wall_catalog(catalog: WallCatalog) -> WallCatalog:
    part_models = tuple(
        replace(model, metadata=_metadata(model.metadata))
        for model in catalog.part_models
    )
    frozen = replace(
        catalog,
        part_models=part_models,
        metadata=_metadata(catalog.metadata),
    )
    lookup = cast(
        dict[int, WallPartModel],
        immutable_mapping({model.part_id: model for model in part_models}),
    )
    object.__setattr__(frozen, "_part_lookup", lookup)
    return frozen


def _resolved_charge(config: ChargeConfig | None) -> ChargeModelConfig:
    if config is None:
        return ChargeModelConfig()
    return ChargeModelConfig(
        enabled=config.enabled,
        mode=config.mode,
        **dict(config.parameters),
    )


def _resolved_plasma_background(
    config: ChargeConfig | None,
) -> PreparedPlasmaBackground | None:
    if config is None or not config.background:
        return None
    return prepare_plasma_background(PlasmaBackgroundConfig(**dict(config.background)))


def _resolved_stochastic(
    config: StochasticConfig | None,
    *,
    default_seed: int,
) -> StochasticMotionConfig:
    if config is None:
        return StochasticMotionConfig(seed=default_seed)
    return StochasticMotionConfig(
        enabled=config.enabled,
        model=config.model,
        seed=default_seed if config.seed is None else config.seed,
        temperature_source=config.temperature_source,
    )


def _optional_float(value: float | None) -> float:
    return float("nan") if value is None else value


def _gas_properties(config: RunConfig) -> GasProperties:
    gas = config.physics.gas
    return GasProperties(
        temperature=_optional_float(gas.temperature_K),
        dynamic_viscosity_Pas=_optional_float(gas.dynamic_viscosity_Pas),
        density_kgm3=_optional_float(gas.density_kgm3),
        molecular_mass_amu=_optional_float(gas.molecular_mass_amu),
    )


def assemble_solver_context(
    config: RunConfig,
    adapter: ResolvedAdapterInputs,
) -> SolverContext:
    source_geometry = adapter.providers.geometry_provider
    source_field = adapter.providers.field_provider
    if source_geometry is None or source_field is None:
        raise ValueError("solver context requires geometry and field providers")
    particles = _readonly_particles(adapter.runtime_inputs.particles)
    geometry_provider = _readonly_geometry_provider(source_geometry)
    field_provider = _readonly_field_provider(source_field)

    spatial_dim = config.case.spatial_dim
    coordinate_system = config.case.coordinate_system
    force_catalog = resolve_force_catalog(
        adapter.force_model,
        field_provider=field_provider,
        spatial_dim=spatial_dim,
    )
    validate_coordinate_force_compatibility(
        coordinate_system,
        adapter.force_model,
    )
    charge_model = _resolved_charge(config.physics.charge)
    stochastic_motion = _resolved_stochastic(
        config.physics.stochastic,
        default_seed=config.physics.seed,
    )
    plasma_background = _resolved_plasma_background(config.physics.charge)
    force_runtime = compile_force_runtime_parameters(adapter.force_model)
    plan = build_solver_plan(
        spatial_dim=spatial_dim,
        dt=config.time.dt,
        t_end=config.time.t_end,
        rng_seed=config.physics.seed,
        drag_model=adapter.drag_model,
        output_mode=config.output.mode,
        save_every=config.output.trajectory_interval_steps or 1,
        force_catalog=force_catalog,
        charge_model=charge_model,
        stochastic_motion=stochastic_motion,
        force_runtime=force_runtime,
        boundary=resolve_boundary_numerics(geometry_provider),
        adaptive_substep_max_splits=config.time.max_substep_splits,
        max_wall_hits_per_step=(config.physics.wall_interaction.max_hits_per_step),
        contact_sliding_enabled=(config.physics.wall_interaction.contact_sliding),
    )
    options = RuntimeOptions(
        stochastic_motion=stochastic_motion,
        charge_model=charge_model,
        plasma_background=plasma_background,
        force_runtime=force_runtime,
    )
    wall_catalog = _readonly_wall_catalog(
        build_wall_catalog(adapter.runtime_inputs.walls)
    )
    return SolverContext(
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
        particles=particles,
        geometry_provider=geometry_provider,
        field_provider=field_provider,
        gas=_gas_properties(config),
        wall_catalog=wall_catalog,
        force_catalog=force_catalog,
        plan=plan,
        options=options,
    )


__all__ = ("assemble_solver_context",)
