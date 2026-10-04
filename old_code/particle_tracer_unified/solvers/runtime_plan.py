from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from particle_tracer_unified.core.boundary_numerics import BoundaryNumerics

from .drag_models import (
    drag_model_mode_from_name,
    drag_model_name_from_mode,
    drag_model_stage_gas_requirements,
)
from .forces import ForceCatalog, ForceRuntimeParameters

if TYPE_CHECKING:
    from ._charge_model_types import ChargeModelConfig
    from ._stochastic_config import StochasticMotionConfig


OUTPUT_MODE_STANDARD = "standard"
OUTPUT_MODE_DEBUG = "debug"
SUPPORTED_OUTPUT_MODES = (OUTPUT_MODE_STANDARD, OUTPUT_MODE_DEBUG)


@dataclass(frozen=True)
class StageFieldPlan:
    """Field quantities sampled together for a solver stage."""

    need_flow: bool = True
    need_electric: bool = False
    need_gas_density: bool = False
    need_gas_mu: bool = False
    need_gas_temperature: bool = False
    need_valid_mask: bool = True

    @property
    def needs_gas_properties(self) -> bool:
        return bool(
            self.need_gas_density or self.need_gas_mu or self.need_gas_temperature
        )


@dataclass(frozen=True)
class OutputPlan:
    """Lightweight output policy for large particle runs."""

    mode: str = OUTPUT_MODE_STANDARD
    save_every: int = 10

    @property
    def is_debug(self) -> bool:
        return self.mode == OUTPUT_MODE_DEBUG


@dataclass(frozen=True)
class SolverPlan:
    """Immutable numerical decisions consumed by the solver loop."""

    spatial_dim: int
    dt: float
    t_end: float
    base_save_every: int
    plot_limit: int
    rng_seed: int
    max_wall_hits_per_step: int
    adaptive_substep_enabled: int
    adaptive_substep_max_splits: int
    boundary: BoundaryNumerics
    boundary_broad_phase_enabled: bool
    drag_model_mode: int
    drag_model_name: str
    body_acceleration_mps2: tuple[float, ...]
    stage_fields: StageFieldPlan
    output: OutputPlan
    contact_sliding_enabled: bool = True


@dataclass(frozen=True)
class ReleaseSchedule:
    """Sorted release order used to avoid scanning all particles every step."""

    order: np.ndarray
    release_time_s: np.ndarray

    @property
    def count(self) -> int:
        return int(self.order.size)


@dataclass
class ReleaseCursor:
    schedule: ReleaseSchedule
    position: int = 0

    @property
    def done(self) -> bool:
        return int(self.position) >= int(self.schedule.count)

    def next_time(self) -> float:
        if self.done:
            return float("inf")
        index = int(self.schedule.order[int(self.position)])
        return float(self.schedule.release_time_s[index])


def build_output_plan(*, mode: str, save_every: int) -> OutputPlan:
    """Resolve already-typed output values without normalization or floors."""

    if mode not in SUPPORTED_OUTPUT_MODES:
        raise ValueError("output.mode must be 'standard' or 'debug'")
    if type(save_every) is not int or save_every <= 0:
        raise ValueError("output trajectory interval must be a positive integer")
    return OutputPlan(mode=mode, save_every=save_every)


def _body_acceleration_from_force_catalog(
    force_catalog: ForceCatalog | None,
    spatial_dim: int,
) -> tuple[float, ...]:
    zeros = tuple(0.0 for _ in range(int(spatial_dim)))
    catalog = force_catalog
    if catalog is None:
        return zeros
    gravity = catalog.model.gravity
    if not gravity.enabled:
        return zeros
    values = np.asarray(gravity.acceleration_mps2, dtype=np.float64).reshape(-1)
    if values.size != int(spatial_dim) or np.any(~np.isfinite(values)):
        raise ValueError(
            "physics.forces.gravity.parameters.acceleration_mps2 must contain "
            f"{int(spatial_dim)} finite values"
        )
    return tuple(float(value) for value in values)


def _active_stage_force_names(
    *,
    drag_model: str,
    force_catalog: ForceCatalog | None,
    force_runtime: ForceRuntimeParameters,
) -> frozenset[str]:
    catalog_names = () if force_catalog is None else force_catalog.enabled_names()
    if not catalog_names and drag_model != "none":
        catalog_names = ("drag",)
    return frozenset((*catalog_names, *force_runtime.enabled_evaluator_names()))


def _stage_needs_flow(
    drag_model: str,
    enabled: frozenset[str],
    stochastic_enabled: bool,
) -> bool:
    return bool(
        (drag_model != "none" and "drag" in enabled)
        or "lift" in enabled
        or "pressure_gradient" in enabled
        or "virtual_mass" in enabled
        or stochastic_enabled
    )


def _stage_needs_electric(
    enabled: frozenset[str],
    charge_enabled: bool,
) -> bool:
    return bool(
        "electric" in enabled or "dielectrophoresis" in enabled or charge_enabled
    )


def _drag_uses_gas_quantity(drag_model: str, quantity: str) -> bool:
    return quantity in drag_model_stage_gas_requirements(drag_model)


def _stage_needs_gas_density(
    drag_model: str,
    enabled: frozenset[str],
    gravity_buoyancy_enabled: bool,
) -> bool:
    density_forces = {
        "pressure_gradient",
        "virtual_mass",
        "thermophoresis",
        "lift",
    }
    return bool(
        _drag_uses_gas_quantity(drag_model, "density_kgm3")
        or enabled.intersection(density_forces)
        or gravity_buoyancy_enabled
    )


def _stage_needs_gas_viscosity(
    drag_model: str,
    enabled: frozenset[str],
) -> bool:
    return bool(
        _drag_uses_gas_quantity(drag_model, "dynamic_viscosity_Pas")
        or "thermophoresis" in enabled
        or "lift" in enabled
    )


def _stage_needs_gas_temperature(
    drag_model: str,
    enabled: frozenset[str],
    *,
    stochastic_field_temperature: bool,
    charge_uses_field_background: bool,
) -> bool:
    return bool(
        _drag_uses_gas_quantity(drag_model, "temperature_K")
        or stochastic_field_temperature
        or "thermophoresis" in enabled
        or charge_uses_field_background
    )


def resolve_stage_field_requirements(
    *,
    drag_model: str,
    force_runtime: ForceRuntimeParameters | None = None,
    force_catalog: ForceCatalog | None = None,
    charge_model: ChargeModelConfig | None = None,
    stochastic_motion: StochasticMotionConfig | None = None,
) -> StageFieldPlan:
    """Resolve the canonical fields required by drag and enabled force laws."""

    drag_name = str(drag_model).strip().lower()
    runtime = force_runtime or ForceRuntimeParameters()
    enabled = _active_stage_force_names(
        drag_model=drag_name,
        force_catalog=force_catalog,
        force_runtime=runtime,
    )
    charge_enabled = bool(charge_model is not None and charge_model.enabled)
    stochastic_enabled = bool(
        stochastic_motion is not None and stochastic_motion.enabled
    )
    return StageFieldPlan(
        need_flow=_stage_needs_flow(drag_name, enabled, stochastic_enabled),
        need_electric=_stage_needs_electric(enabled, charge_enabled),
        need_gas_density=_stage_needs_gas_density(
            drag_name,
            enabled,
            bool(runtime.gravity_buoyancy_enabled),
        ),
        need_gas_mu=_stage_needs_gas_viscosity(drag_name, enabled),
        need_gas_temperature=_stage_needs_gas_temperature(
            drag_name,
            enabled,
            stochastic_field_temperature=bool(
                stochastic_enabled
                and stochastic_motion is not None
                and stochastic_motion.temperature_source == "field_T_then_gas"
            ),
            charge_uses_field_background=bool(
                charge_enabled
                and charge_model is not None
                and charge_model.background_source == "field"
            ),
        ),
        need_valid_mask=True,
    )


def build_stage_field_plan(
    *,
    drag_model: str,
    force_catalog: ForceCatalog | None,
    charge_model: ChargeModelConfig | None,
    stochastic_motion: StochasticMotionConfig | None,
    force_runtime: ForceRuntimeParameters | None,
) -> StageFieldPlan:
    return resolve_stage_field_requirements(
        drag_model=drag_model,
        force_runtime=force_runtime,
        force_catalog=force_catalog,
        charge_model=charge_model,
        stochastic_motion=stochastic_motion,
    )


def build_release_schedule(release_time_s: np.ndarray) -> ReleaseSchedule:
    release_time = np.asarray(release_time_s, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(release_time))
    if finite.size == 0:
        return ReleaseSchedule(
            order=np.zeros(0, dtype=np.int64), release_time_s=release_time
        )
    order = finite[np.argsort(release_time[finite], kind="mergesort")].astype(
        np.int64, copy=False
    )
    return ReleaseSchedule(order=order, release_time_s=release_time)


def build_solver_plan(
    *,
    spatial_dim: int,
    dt: float,
    t_end: float,
    rng_seed: int,
    drag_model: str,
    output_mode: str,
    save_every: int,
    force_catalog: ForceCatalog | None,
    charge_model: ChargeModelConfig | None,
    stochastic_motion: StochasticMotionConfig | None,
    force_runtime: ForceRuntimeParameters | None,
    boundary: BoundaryNumerics,
    adaptive_substep_max_splits: int = 4,
    max_wall_hits_per_step: int = 5,
    contact_sliding_enabled: bool = True,
) -> SolverPlan:
    """Resolve the immutable numerical plan from typed inputs."""

    dim = int(spatial_dim)
    max_splits = int(adaptive_substep_max_splits)
    if max_splits < 0:
        raise ValueError("solver adaptive_substep_max_splits must be >= 0")
    wall_hit_budget = int(max_wall_hits_per_step)
    if wall_hit_budget < 1:
        raise ValueError("solver max_wall_hits_per_step must be >= 1")
    dt = float(dt)
    t_end = float(t_end)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("solver.dt must be finite and > 0")
    if not np.isfinite(t_end) or t_end < 0.0:
        raise ValueError("solver.t_end must be finite and >= 0")

    if type(save_every) is not int or save_every <= 0:
        raise ValueError("solver save interval must be a positive integer")
    drag_model_mode = drag_model_mode_from_name(drag_model)
    drag_model_name = drag_model_name_from_mode(int(drag_model_mode))
    output = build_output_plan(mode=output_mode, save_every=save_every)
    stage_fields = build_stage_field_plan(
        drag_model=drag_model_name,
        force_catalog=force_catalog,
        charge_model=charge_model,
        stochastic_motion=stochastic_motion,
        force_runtime=force_runtime,
    )

    return SolverPlan(
        spatial_dim=dim,
        dt=dt,
        t_end=t_end,
        base_save_every=save_every,
        plot_limit=32,
        rng_seed=int(rng_seed),
        max_wall_hits_per_step=wall_hit_budget,
        contact_sliding_enabled=bool(contact_sliding_enabled),
        adaptive_substep_enabled=1,
        adaptive_substep_max_splits=max_splits,
        boundary=boundary,
        boundary_broad_phase_enabled=False,
        drag_model_mode=int(drag_model_mode),
        drag_model_name=str(drag_model_name),
        body_acceleration_mps2=_body_acceleration_from_force_catalog(
            force_catalog, dim
        ),
        stage_fields=stage_fields,
        output=output,
    )
