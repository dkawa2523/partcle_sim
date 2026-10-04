"""Stable public entry points and public exception translation."""

from os import PathLike

from .case import SimulationCase
from .case import load_case as _load_case
from .engine import EngineError, run_simulation
from .output import (
    IncompleteResult,
    ResultOpenError,
    ResultView,
    ResultWriteError,
    RunSummary,
    open_result_store,
)

type _PathInput = str | PathLike[str]


class CaseError(ValueError):
    """The canonical case is invalid or cannot be read."""


class SimulationError(RuntimeError):
    """The requested simulation cannot be prepared or completed."""


class IncompleteResultError(SimulationError):
    """A result is incomplete and was opened without recovery mode."""


def load_case(path: _PathInput) -> SimulationCase:
    """Load and statically validate a canonical YAML/HDF5 case."""
    try:
        return _load_case(path)
    except (OSError, ValueError) as exc:
        raise CaseError(f"cannot load case {path!s}: {exc}") from exc


def simulate(case: SimulationCase, output: _PathInput) -> RunSummary:
    """Run the supported production profile and atomically publish its result."""
    try:
        return run_simulation(case, output)
    except (EngineError, ResultWriteError, OSError, ValueError) as exc:
        raise SimulationError(f"simulation failed: {exc}") from exc


def open_result(path: _PathInput, *, recovery: bool = False) -> ResultView:
    """Open a completed result through its lazy read-only view."""
    try:
        return open_result_store(path, recovery=recovery)
    except IncompleteResult as exc:
        raise IncompleteResultError(str(exc)) from exc
    except (ResultOpenError, OSError, ValueError) as exc:
        raise SimulationError(f"cannot open result {path!s}: {exc}") from exc
