"""Immutable request and destination contracts for deterministic segment motion."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .compiled_backend_types import CompiledRuntimeBackend
from .forces import ForceRuntimeParameters


@dataclass(slots=True)
class ValidMaskPrefixResolution:
    position: np.ndarray
    velocity: np.ndarray
    accepted_dt: float
    retry_count: int
    found_valid_prefix: bool
    charge_C: float | None = None


@dataclass(frozen=True, slots=True)
class SegmentMotionRequest:
    """Complete immutable input for one deterministic particle segment."""

    position_m: np.ndarray
    velocity_mps: np.ndarray
    duration_s: float
    end_time_s: float
    spatial_dim: int
    backend: CompiledRuntimeBackend
    adaptive_substep_enabled: int
    adaptive_substep_max_splits: int
    tau_stokes_s: float
    particle_diameter_m: float
    particle_density_kgm3: float
    particle_mass_kg: float
    dep_particle_rel_permittivity: float
    thermophoretic_coefficient: float
    body_acceleration_mps2: np.ndarray
    gas_density_kgm3: float
    gas_dynamic_viscosity_Pas: float
    gas_temperature_K: float
    gas_molecular_mass_kg: float
    drag_model_mode: int
    electric_q_over_m_Ckg: float | None = None
    force_runtime: ForceRuntimeParameters | None = None
    minimum_substeps: int = 1

    @property
    def start_time_s(self) -> float:
        return float(self.end_time_s) - max(float(self.duration_s), 0.0)

    def prefix(
        self, elapsed_s: float, *, minimum_substeps: int = 1
    ) -> SegmentMotionRequest:
        duration = float(
            np.clip(float(elapsed_s), 0.0, max(float(self.duration_s), 0.0))
        )
        return replace(
            self,
            duration_s=duration,
            end_time_s=self.start_time_s + duration,
            minimum_substeps=int(max(1, minimum_substeps)),
        )

    def with_minimum_substeps(self, count: int) -> SegmentMotionRequest:
        return replace(self, minimum_substeps=int(max(1, count)))


@dataclass(frozen=True, slots=True)
class SegmentMotionBatchRequest:
    """Immutable normalized request for one same-duration particle batch."""

    position_m: np.ndarray
    velocity_mps: np.ndarray
    active: np.ndarray
    tau_stokes_s: np.ndarray
    particle_diameter_m: np.ndarray
    particle_density_kgm3: np.ndarray
    particle_mass_kg: np.ndarray
    dep_particle_rel_permittivity: np.ndarray
    thermophoretic_coefficient: np.ndarray
    end_time_s: float
    duration_s: float
    spatial_dim: int
    backend: CompiledRuntimeBackend
    body_acceleration_mps2: np.ndarray
    gas_density_kgm3: float
    gas_dynamic_viscosity_Pas: float
    gas_temperature_K: float
    gas_molecular_mass_kg: float
    drag_model_mode: int
    adaptive_substep_enabled: int
    adaptive_substep_max_splits: int
    electric_q_over_m_Ckg: np.ndarray | None = None
    force_runtime: ForceRuntimeParameters | None = None

    @property
    def start_time_s(self) -> float:
        return float(self.end_time_s) - max(float(self.duration_s), 0.0)

    def particle_request(self, index: int) -> SegmentMotionRequest:
        i = int(index)
        q_over_m = None
        if self.electric_q_over_m_Ckg is not None:
            q_over_m = float(
                np.asarray(self.electric_q_over_m_Ckg, dtype=np.float64)[i]
            )
        return SegmentMotionRequest(
            position_m=np.asarray(
                self.position_m[i, : self.spatial_dim], dtype=np.float64
            ),
            velocity_mps=np.asarray(
                self.velocity_mps[i, : self.spatial_dim], dtype=np.float64
            ),
            duration_s=float(self.duration_s),
            end_time_s=float(self.end_time_s),
            spatial_dim=int(self.spatial_dim),
            backend=self.backend,
            adaptive_substep_enabled=int(self.adaptive_substep_enabled),
            adaptive_substep_max_splits=int(self.adaptive_substep_max_splits),
            tau_stokes_s=float(self.tau_stokes_s[i]),
            particle_diameter_m=float(self.particle_diameter_m[i]),
            particle_density_kgm3=float(self.particle_density_kgm3[i]),
            particle_mass_kg=float(self.particle_mass_kg[i]),
            dep_particle_rel_permittivity=float(self.dep_particle_rel_permittivity[i]),
            thermophoretic_coefficient=float(self.thermophoretic_coefficient[i]),
            body_acceleration_mps2=np.asarray(
                self.body_acceleration_mps2, dtype=np.float64
            ),
            gas_density_kgm3=float(self.gas_density_kgm3),
            gas_dynamic_viscosity_Pas=float(self.gas_dynamic_viscosity_Pas),
            gas_temperature_K=float(self.gas_temperature_K),
            gas_molecular_mass_kg=float(self.gas_molecular_mass_kg),
            drag_model_mode=int(self.drag_model_mode),
            electric_q_over_m_Ckg=q_over_m,
            force_runtime=self.force_runtime,
        )


@dataclass(frozen=True, slots=True)
class SegmentMotionBatchDestination:
    """Solver-state-owned destination buffers for a batch trace."""

    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    midpoint_position_m: np.ndarray
    substep_count: np.ndarray
    aggregate_support_status: np.ndarray
    local_error_resolved: np.ndarray
