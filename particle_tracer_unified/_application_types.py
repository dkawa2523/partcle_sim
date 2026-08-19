"""Immutable public case, result, and artifact value types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import SCHEMA_VERSION, RunConfig
from .core.datamodel import SolverContext, immutable_mapping


@dataclass(frozen=True)
class SimulationPlan:
    spatial_dim: int
    coordinate_system: str
    dt_s: float
    t_end_s: float
    drag_model: str
    output_mode: str

    @property
    def integrator(self) -> str:
        return "etd2"

    @classmethod
    def from_resolved(
        cls,
        config: RunConfig,
        context: SolverContext,
    ) -> SimulationPlan:
        plan = context.plan
        return cls(
            spatial_dim=int(context.spatial_dim),
            coordinate_system=config.case.coordinate_system,
            dt_s=float(plan.dt),
            t_end_s=float(plan.t_end),
            drag_model=str(plan.drag_model_name),
            output_mode=str(plan.output.mode),
        )


@dataclass(frozen=True)
class SimulationCase:
    config: RunConfig
    config_path: Path
    _context: SolverContext = field(repr=False, compare=False)
    _provenance: Mapping[str, Any] = field(repr=False, compare=False)
    _execution: Mapping[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_provenance", immutable_mapping(self._provenance))
        object.__setattr__(self, "_execution", immutable_mapping(self._execution))

    @property
    def plan(self) -> SimulationPlan:
        return SimulationPlan.from_resolved(self.config, self._context)

    @property
    def solver_context(self) -> SolverContext:
        """Resolved context for read-only validation and comparison tooling."""

        return self._context


@dataclass(frozen=True)
class SimulationState:
    particle_id: np.ndarray
    position_m: np.ndarray
    velocity_mps: np.ndarray
    charge_C: np.ndarray
    release_time_s: np.ndarray
    source_part_id: np.ndarray
    material_id: np.ndarray
    mass_kg: np.ndarray
    drag_diameter_m: np.ndarray
    released: np.ndarray
    terminal_state: np.ndarray
    invalid_stop_reason_code: np.ndarray
    invalid_stop_reason: np.ndarray
    contact_part_id: np.ndarray
    contact_normal: np.ndarray


@dataclass(frozen=True)
class RunStats:
    timing_s: Mapping[str, float]
    memory_estimate_bytes: Mapping[str, int]
    terminal_counts: Mapping[str, int]
    wall_outcome_counts: Mapping[str, int]
    particle_count: int
    released_count: int
    safety_counters: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in (
            "timing_s",
            "memory_estimate_bytes",
            "terminal_counts",
            "wall_outcome_counts",
            "safety_counters",
        ):
            object.__setattr__(self, name, immutable_mapping(getattr(self, name)))


@dataclass(frozen=True)
class SimulationResult:
    plan: SimulationPlan
    state: SimulationState
    stats: RunStats
    wall_summary: Mapping[tuple[int, str, str], int]
    axis_names: tuple[str, ...]
    drag_model: str
    experimental_features: tuple[str, ...]
    final_step_name: str
    final_segment_name: str
    execution_metadata: Mapping[str, Any] = field(repr=False, compare=False)
    debug: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "wall_summary", immutable_mapping(self.wall_summary))
        object.__setattr__(self, "debug", immutable_mapping(self.debug))


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_type: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    output_dir: Path
    records: tuple[ArtifactRecord, ...]
    schema_version: int = SCHEMA_VERSION

    @property
    def files(self) -> Mapping[str, Path]:
        return {record.artifact_type: record.path for record in self.records}


__all__ = (
    "ArtifactManifest",
    "ArtifactRecord",
    "RunStats",
    "SimulationCase",
    "SimulationPlan",
    "SimulationResult",
    "SimulationState",
)
