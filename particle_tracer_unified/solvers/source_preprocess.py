from __future__ import annotations

from typing import Any, Mapping, Optional

from ..core.datamodel import (
    MaterialTable,
    PartWallTable,
    ParticleTable,
    ProcessStepTable,
    SourceEventTable,
    SourcePreprocessResult,
)
from ..core.source_materials import apply_source_models, resolve_source_parameters
from ..providers.source_adapters import ConstantScalarSampler, SourceFlowSampler, SourceNormalSampler, SourceScalarSampler, ZeroFlowSampler


def preprocess_particles_for_solver(
    particles: ParticleTable,
    walls: Optional[PartWallTable],
    materials: Optional[MaterialTable],
    source_events: Optional[SourceEventTable],
    source_cfg: Mapping[str, Any],
    gas_temperature: float,
    gas_viscosity: float,
    gas_density_kgm3: float,
    normal_sampler: SourceNormalSampler,
    flow_sampler: Optional[SourceFlowSampler] = None,
    wall_shear_sampler: Optional[SourceScalarSampler] = None,
    friction_velocity_sampler: Optional[SourceScalarSampler] = None,
    viscosity_sampler: Optional[SourceScalarSampler] = None,
    process_steps: Optional[ProcessStepTable] = None,
    seed: int = 12345,
) -> SourcePreprocessResult:
    resolved = resolve_source_parameters(
        particles=particles,
        walls=walls,
        materials=materials,
        source_cfg=source_cfg,
        gas_temperature=gas_temperature,
        gas_viscosity=gas_viscosity,
    )
    return apply_source_models(
        particles=particles,
        resolved=resolved,
        normal_sampler=normal_sampler,
        flow_sampler=flow_sampler or ZeroFlowSampler(particles.spatial_dim),
        wall_shear_sampler=wall_shear_sampler or ConstantScalarSampler(float('nan')),
        friction_velocity_sampler=friction_velocity_sampler or ConstantScalarSampler(float('nan')),
        viscosity_sampler=viscosity_sampler or ConstantScalarSampler(float('nan')),
        events=source_events,
        process_steps=process_steps,
        gas_density_kgm3=float(gas_density_kgm3),
        seed=seed,
    )
