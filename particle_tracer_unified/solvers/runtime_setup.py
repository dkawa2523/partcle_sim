from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .charge_model import ChargeModelConfig, parse_charge_model_config
from .forces import (
    ForceCatalog,
    ForceRuntimeParameters,
    force_runtime_parameters_from_catalog,
    solver_cfg_with_force_overrides,
)
from .plasma_background import PreparedPlasmaBackground, parse_plasma_background_config, prepare_plasma_background
from .runtime_outputs import RuntimeOutputOptions, config_bool_flag
from .runtime_plan import SolverPlan
from .stochastic_motion import StochasticMotionConfig, parse_stochastic_motion_config


@dataclass(frozen=True)
class RuntimeOptions:
    """Model/output options that are not fixed scalar execution plan fields."""

    write_collision_diagnostics: int
    output_options: RuntimeOutputOptions
    stochastic_motion: StochasticMotionConfig = field(default_factory=StochasticMotionConfig)
    charge_model: ChargeModelConfig = field(default_factory=ChargeModelConfig)
    plasma_background: PreparedPlasmaBackground | None = None
    force_catalog: ForceCatalog | None = None
    force_runtime: ForceRuntimeParameters = field(default_factory=ForceRuntimeParameters)


def as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def solver_config_with_catalog(config_payload: Mapping[str, object], force_catalog: ForceCatalog | None) -> dict[str, object]:
    config = as_mapping(config_payload)
    raw_solver_cfg = as_mapping(config.get('solver', {}))
    return solver_cfg_with_force_overrides(raw_solver_cfg, force_catalog)


def runtime_options_from_plan(
    *,
    plan: SolverPlan,
    config_payload: Mapping[str, object],
    force_catalog: ForceCatalog | None,
) -> RuntimeOptions:
    config = as_mapping(config_payload)
    solver_cfg = solver_config_with_catalog(config, force_catalog)
    output_cfg = as_mapping(config.get('output', {}))

    if 'contact_tangent_motion' in solver_cfg:
        raise ValueError(
            'solver.contact_tangent_motion is obsolete; implement contact behavior through '
            'the BoundaryEvent/ContactState solver contract'
        )

    output_options = RuntimeOutputOptions.from_output_plan(plan.output, output_cfg)
    write_collision_diagnostics = config_bool_flag(
        output_cfg,
        'write_collision_diagnostics',
        int(bool(plan.output.write_collision_diagnostics)),
    )
    return RuntimeOptions(
        write_collision_diagnostics=int(write_collision_diagnostics),
        output_options=output_options,
        stochastic_motion=parse_stochastic_motion_config(solver_cfg, default_seed=int(plan.rng_seed)),
        charge_model=parse_charge_model_config(solver_cfg),
        plasma_background=prepare_plasma_background(parse_plasma_background_config(solver_cfg)),
        force_catalog=force_catalog,
        force_runtime=force_runtime_parameters_from_catalog(force_catalog),
    )
