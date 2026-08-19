"""Resolve a canonical ``RunConfig`` into an immutable solver context."""

from __future__ import annotations

from pathlib import Path

from particle_tracer_unified.configuration import RunConfig
from particle_tracer_unified.core.datamodel import SolverContext

from ._runtime_adapter import resolve_adapter_inputs
from ._runtime_context import assemble_solver_context


def build_solver_context(config: RunConfig, config_dir: Path) -> SolverContext:
    """Build the solver boundary object directly from canonical typed values."""

    if not isinstance(config, RunConfig):
        raise TypeError("build_solver_context requires a typed RunConfig")
    adapter = resolve_adapter_inputs(config, Path(config_dir))
    return assemble_solver_context(config, adapter)


__all__ = ("build_solver_context",)
