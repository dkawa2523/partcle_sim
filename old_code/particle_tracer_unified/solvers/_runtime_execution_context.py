"""Shared state passed from run preparation through execution and finalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.datamodel import SolverContext
from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryQuery

from .compiled_backend_types import CompiledRuntimeBackend
from .runtime_plan import SolverPlan
from .runtime_setup import RuntimeOptions
from .runtime_state import SolverState


@dataclass(frozen=True)
class StepLoopResult:
    t: float
    step_count: int
    elapsed_s: float


@dataclass(frozen=True)
class RunExecutionContext:
    """Immutable run resources paired with the mutable solver state."""

    context: SolverContext
    plan: SolverPlan
    options: RuntimeOptions
    state: SolverState
    compiled: CompiledRuntimeBackend
    boundary_service: BoundaryQuery[TriangleSurface3D]
    spatial_dim: int
    mins: np.ndarray
    maxs: np.ndarray
    physics: Mapping[str, float]
    body_acceleration_mps2: np.ndarray
    tau_p: np.ndarray
    particle_mass: np.ndarray
    particle_diameter: np.ndarray  # Drag-law diameter.
    particle_physical_diameter: np.ndarray
    particle_density: np.ndarray
    particle_id: np.ndarray
    dep_particle_rel_permittivity: np.ndarray
    thermophoretic_coeff: np.ndarray
    setup_started_s: float
    loop_setup_done_s: float

    @property
    def n_particles(self) -> int:
        return int(self.state.static.count)


__all__ = ("RunExecutionContext", "StepLoopResult")
