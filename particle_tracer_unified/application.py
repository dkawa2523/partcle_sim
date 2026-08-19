"""Application use cases for loading, validating, and simulating cases."""

from __future__ import annotations

import platform
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._application_runtime import (
    build_simulation_result as _build_simulation_result,
)
from ._application_runtime import (
    require_transient_field_time_support as _require_transient_field_time_support,
)
from ._application_types import (
    ArtifactManifest,
    ArtifactRecord,
    RunStats,
    SimulationCase,
    SimulationPlan,
    SimulationResult,
    SimulationState,
)
from ._version import (
    PACKAGE_NAME,
    PACKAGE_VERSION,
    distribution_version,
    installed_package_version,
)
from .configuration import load_run_config
from .force_models import force_parameter_mapping
from .preflight_types import ValidationReport


def load_case(config_path: str | Path) -> SimulationCase:
    """Load and resolve one canonical simulation case."""

    source = Path(config_path).resolve()
    config = load_run_config(source)
    from .io.provenance import collect_case_provenance
    from .io.runtime_builder import build_solver_context

    context = build_solver_context(config, source.parent)
    provenance = collect_case_provenance(config, source)
    unresolved_case = SimulationCase(
        config=config,
        config_path=source,
        _context=context,
        _provenance=provenance,
        _execution={},
    )
    case = replace(
        unresolved_case,
        _execution=_build_execution_metadata(unresolved_case),
    )
    _require_transient_field_time_support(case)
    return case


def validate_case(
    case: SimulationCase,
    *,
    detail: str = "summary",
) -> ValidationReport:
    """Run the unified, side-effect-free preflight."""

    from .preflight import validate_case_preflight

    return validate_case_preflight(case, detail=detail)


def _enabled_force_metadata(case: SimulationCase) -> tuple[Mapping[str, Any], ...]:
    """Return the resolved force selection without evaluating any fields."""

    rows: list[Mapping[str, Any]] = []
    for force in case._context.force_catalog.model.definitions():
        if not force.enabled:
            continue
        rows.append(
            {
                "name": force.name,
                "model": force.model,
                "parameters": force_parameter_mapping(force),
            }
        )
    return tuple(rows)


def _build_execution_metadata(case: SimulationCase) -> Mapping[str, Any]:
    """Snapshot the already-resolved execution contract for reproducibility.

    This reads only small setup objects.  It does not sample fields, query
    geometry, or derive any post-run diagnostics.
    """

    context = case._context
    plan = context.plan
    options = context.options
    stochastic = options.stochastic_motion
    charge = options.charge_model
    plasma_background = options.plasma_background
    return {
        "adapter": str(case.config.case.adapter),
        "dt_s": float(plan.dt),
        "t_end_s": float(plan.t_end),
        "rng_seed": int(plan.rng_seed),
        "stochastic_seed": int(stochastic.seed),
        "forces": _enabled_force_metadata(case),
        "gas": {
            "temperature_K": float(context.gas.temperature),
            "dynamic_viscosity_Pas": float(context.gas.dynamic_viscosity_Pas),
            "density_kgm3": float(context.gas.density_kgm3),
            "molecular_mass_amu": float(context.gas.molecular_mass_amu),
        },
        "charge": asdict(charge) if charge.enabled else {"enabled": False},
        "stochastic": (
            asdict(stochastic) if stochastic.enabled else {"enabled": False}
        ),
        "plasma_background": (
            None if plasma_background is None else asdict(plasma_background)
        ),
        "numerics": {
            "policy_version": "etd2-affine-lte-v3",
            "integrator": "etd2",
            "boundary": dict(plan.boundary.summary()),
            "max_wall_hits_per_step": int(plan.max_wall_hits_per_step),
            "adaptive_substep_enabled": bool(plan.adaptive_substep_enabled),
            "adaptive_substep_max_splits": int(plan.adaptive_substep_max_splits),
            "boundary_broad_phase_enabled": bool(plan.boundary_broad_phase_enabled),
        },
        "provenance": dict(case._provenance),
        "software": {
            "package": PACKAGE_NAME,
            "package_version": PACKAGE_VERSION,
            "installed_distribution_version": installed_package_version(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "numba_version": distribution_version("numba"),
        },
    }


def simulate(case: SimulationCase) -> SimulationResult:
    """Run a case without creating files or consulting output paths."""

    _require_transient_field_time_support(case)
    from .solvers.high_fidelity_runtime import simulate_context

    return _build_simulation_result(case, simulate_context(case._context))


__all__ = [
    "ArtifactManifest",
    "ArtifactRecord",
    "RunStats",
    "SimulationCase",
    "SimulationPlan",
    "SimulationResult",
    "SimulationState",
    "load_case",
    "simulate",
    "validate_case",
]
