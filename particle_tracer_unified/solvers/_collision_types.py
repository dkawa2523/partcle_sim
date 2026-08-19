"""Typed state and value objects shared by collision solver responsibilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from particle_tracer_unified.core.geometry3d import TriangleSurface3D
from particle_tracer_unified.domain import BoundaryHit

from ._coupled_charge_motion import (
    CoupledChargeMotionTrace,
    CoupledChargeParticleTracer,
)
from ._stochastic_randomness import WallRandomContext
from .compiled_backend_types import CompiledRuntimeBackend
from .forces import ForceRuntimeParameters
from .segment_motion import SegmentMotionRequest, SegmentMotionTrace
from .stochastic_motion import PiecewiseLangevinPath
from .terminal_outcome import TerminalSegmentOutcome

AcceptedMotionTrace = SegmentMotionTrace | CoupledChargeMotionTrace


class CollisionPartialMotion(Protocol):
    def __call__(
        self,
        *,
        inputs: CollisionSegmentInputs,
        x0: np.ndarray,
        v0: np.ndarray,
        dt_partial: float,
        segment_dt: float,
        t_end_segment: float,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class WallHitStepResult(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    remaining_dt: float
    hit_count: int
    total_hit_count: int
    should_break: bool
    entered_contact: bool = False
    contact_part_id: int = 0
    contact_normal: np.ndarray | None = None
    contact_primitive_id: int = -1


class OrientedWallHitState(NamedTuple):
    hit: np.ndarray
    normal: np.ndarray
    wall_position: np.ndarray


class WallEventParticleMetadata(NamedTuple):
    particle_id: int
    mass_kg: float
    diameter_m: float


class WallHitTiming(NamedTuple):
    clamped_hit_dt: float
    consumed_dt: float
    alpha: float


class CollidingParticleAdvanceResult(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    total_hits: int
    valid_mask_status: int
    invalid_mask_stopped: bool
    invalid_stop_reason: str = ""
    numerical_boundary_stopped: bool = False
    numerical_boundary_stop_reason: str = ""
    contact_sliding: bool = False
    contact_part_id: int = 0
    contact_normal: np.ndarray | None = None
    contact_primitive_id: int = -1
    terminal_outcome: TerminalSegmentOutcome | None = None
    charge_C: float | None = None


@dataclass(frozen=True)
class CollisionSegmentTrial:
    x_next: np.ndarray
    v_next: np.ndarray
    stage_points: np.ndarray
    primary_hit: BoundaryHit | None
    primary_hit_counted: bool
    particle_valid_mask_status: int
    terminal_stop_result: CollidingParticleAdvanceResult | None = None
    accepted_substep_count: int = 1
    accepted_trace: AcceptedMotionTrace | None = None


@dataclass(frozen=True)
class CollisionSegmentResolution:
    advance_without_hit: bool
    should_break: bool
    x_next: np.ndarray
    v_next: np.ndarray
    hit_event: BoundaryHit | None = None
    v_hit: np.ndarray | None = None
    hit_dt: float = 0.0


@dataclass(frozen=True)
class CollisionSegmentInputs:
    spatial_dim: int
    compiled: CompiledRuntimeBackend
    adaptive_substep_max_splits: int
    tau_p_i: float
    particle_diameter_i: float
    particle_density_i: float
    particle_mass_i: float
    dep_particle_rel_permittivity_i: float
    thermophoretic_coeff_i: float
    body_accel: np.ndarray
    gas_density_kgm3: float
    gas_mu_pas: float
    gas_temperature_K: float
    gas_molecular_mass_kg: float
    drag_model_mode: int
    electric_q_over_m_i: float | None = None
    force_runtime: ForceRuntimeParameters | None = None
    stochastic_path: PiecewiseLangevinPath | None = None
    stochastic_offset_s: float = 0.0
    coupled_charge_tracer: CoupledChargeParticleTracer | None = None
    charge_start_C: float | None = None

    def request(
        self,
        *,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        duration_s: float,
        end_time_s: float,
        adaptive_substep_enabled: int,
        minimum_substeps: int = 1,
    ) -> SegmentMotionRequest:
        return SegmentMotionRequest(
            position_m=np.asarray(position_m, dtype=np.float64),
            velocity_mps=np.asarray(velocity_mps, dtype=np.float64),
            duration_s=float(duration_s),
            end_time_s=float(end_time_s),
            spatial_dim=int(self.spatial_dim),
            backend=self.compiled,
            adaptive_substep_enabled=int(adaptive_substep_enabled),
            adaptive_substep_max_splits=int(self.adaptive_substep_max_splits),
            tau_stokes_s=float(self.tau_p_i),
            particle_diameter_m=float(self.particle_diameter_i),
            particle_density_kgm3=float(self.particle_density_i),
            particle_mass_kg=float(self.particle_mass_i),
            dep_particle_rel_permittivity=float(self.dep_particle_rel_permittivity_i),
            thermophoretic_coefficient=float(self.thermophoretic_coeff_i),
            body_acceleration_mps2=np.asarray(self.body_accel, dtype=np.float64),
            gas_density_kgm3=float(self.gas_density_kgm3),
            gas_dynamic_viscosity_Pas=float(self.gas_mu_pas),
            gas_temperature_K=float(self.gas_temperature_K),
            gas_molecular_mass_kg=float(self.gas_molecular_mass_kg),
            drag_model_mode=int(self.drag_model_mode),
            electric_q_over_m_Ckg=self.electric_q_over_m_i,
            force_runtime=self.force_runtime,
            minimum_substeps=int(minimum_substeps),
        )

    def coupled_charge_start(self) -> float:
        if self.charge_start_C is None:
            raise RuntimeError("coupled charge motion requires a segment-start charge")
        return float(self.charge_start_C)


@dataclass(frozen=True)
class _CollisionResolutionContext:
    x_curr: np.ndarray
    v_curr: np.ndarray
    x_next: np.ndarray
    v_next: np.ndarray
    stage_points: np.ndarray
    inside_fn: Callable[[np.ndarray], bool]
    strict_inside_fn: Callable[[np.ndarray], bool]
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None]
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None]
    primary_hit_counter_key: str
    collision_diagnostics: dict[str, object]
    t: float
    segment_dt: float
    inputs: CollisionSegmentInputs
    on_boundary_tol_m: float


@dataclass(frozen=True)
class _CollisionSearchContext:
    t: float
    dt_step: float
    base_adaptive_substep_enabled: int
    initial_x_next: np.ndarray
    initial_v_next: np.ndarray
    initial_stage_points: np.ndarray
    initial_primary_hit: BoundaryHit | None
    initial_primary_hit_counted: bool
    inside_fn: Callable[[np.ndarray], bool]
    strict_inside_fn: Callable[[np.ndarray], bool]
    primary_hit_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None]
    nearest_projection_fn: Callable[[np.ndarray, np.ndarray], BoundaryHit | None]
    primary_hit_counter_key: str
    collision_diagnostics: dict[str, object]
    on_boundary_tol_m: float
    initial_substep_count: int = 1
    initial_accepted_trace: AcceptedMotionTrace | None = None


@dataclass(frozen=True)
class _WallInteractionContext:
    runtime: object
    particles: object
    particle_index: int
    particle_id: int | None
    particle_mass_kg: float
    particle_diameter_m: float
    rng: np.random.Generator | None
    collision_diagnostics: dict[str, object]
    max_hit_rows: list[dict[str, object]] | None
    wall_rows: list[dict[str, object]] | None
    wall_summary_counts: dict[tuple[int, str, str], int]
    stuck: np.ndarray
    frozen: np.ndarray
    absorbed: np.ndarray
    escaped: np.ndarray
    active: np.ndarray
    max_wall_hits_per_step: int
    epsilon_offset_m: float
    on_boundary_tol_m: float
    t: float
    triangle_surface_3d: TriangleSurface3D | None
    contact_sliding_enabled: bool = True
    random_context: WallRandomContext | None = None


@dataclass
class _CollisionAdvanceState:
    position: np.ndarray
    velocity: np.ndarray
    remaining_dt: float
    valid_mask_status: int
    hit_count: int
    total_hit_count: int
    hit_part_ids: list[int]
    hit_outcomes: list[str]
    use_precomputed_trial: bool
    numerical_boundary_stopped: bool
    numerical_boundary_stop_reason: str
    contact_sliding: bool
    contact_part_id: int
    contact_normal: np.ndarray | None
    contact_primitive_id: int
    terminal_outcome: TerminalSegmentOutcome | None
    charge_C: float | None
