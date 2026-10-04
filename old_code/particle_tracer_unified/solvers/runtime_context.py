"""Concrete solver binding for the core-owned runtime context container."""

from typing import TypeAlias

from particle_tracer_unified.core.datamodel import SolverContext

from .runtime_plan import SolverPlan
from .runtime_setup import RuntimeOptions

RuntimeSolverContext: TypeAlias = SolverContext[SolverPlan, RuntimeOptions]
