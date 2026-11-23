"""Stable public entry points and public exception translation."""

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from .case import SimulationCase
from .case import load_case as _load_case
from .engine import _prepare, run_simulation
from .errors import CaseError, IncompleteResultError, SimulationError
from .output import (
    IncompleteResult,
    ResultOpenError,
    ResultView,
    ResultWriteError,
    RunSummary,
    open_result_store,
)
from .transport import EngineError

type _PathInput = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class CheckedModel:
    """One resolved scientific model, without mutable execution bindings."""

    category: str
    model: str
    revision: str
    parameters: Mapping[str, str | float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class CheckedField:
    """Canonical metadata of an actually bound required field."""

    name: str
    unit: str
    components: tuple[str, ...]
    stored_basis: str
    layout: str


@dataclass(frozen=True, slots=True)
class CaseCheckReport:
    """Immutable execution-preparation summary; no particles have advanced."""

    case_name: str
    case_path: Path
    data_path: Path
    case_file_hash: str
    data_content_hash: str
    coordinate_system: str
    motion_mode: str
    particle_count: int
    integrator: str
    backend: str
    workers: int
    start_time_s: float
    end_time_s: float
    dt_s: float
    required_fields: tuple[str, ...]
    models: tuple[CheckedModel, ...]
    field_time_knots_s: tuple[float, ...]
    boundary_groups: tuple[str, ...]
    memory_limit_bytes: int
    planned_memory_bytes: int
    slab_particles: int
    load_peak_bytes: int
    prepare_peak_bytes: int
    run_peak_bytes: int
    required_field_metadata: tuple[CheckedField, ...]
    status: Literal["valid"] = "valid"


def load_case(path: _PathInput) -> SimulationCase:
    """Load and statically validate a canonical YAML/HDF5 case."""
    try:
        return _load_case(path)
    except (OSError, ValueError) as exc:
        raise CaseError(f"cannot load case {path!s}: {exc}") from exc


def check_case(case: SimulationCase, *, workers: int = 1) -> CaseCheckReport:
    """Use simulation's complete prepare without starting workers or a writer.

    This resolves numerical support and solver-owned memory for this machine.
    It does not qualify physical accuracy or reserve operating-system resources.
    """
    try:
        run = _prepare(case, workers=workers)
    except (EngineError, OSError, ValueError) as exc:
        raise CaseError(f"cannot prepare case {case.spec.name}: {exc}") from exc
    spec = case.spec
    prepared = run.transport
    memory = run.memory_plan
    models = tuple(
        CheckedModel(
            category,
            str(model["model"]),
            str(model["revision"]),
            {key: value for key, value in model.items() if key not in ("model", "revision")},
        )
        for category, model in run.facts.model_document().items()
    )
    return CaseCheckReport(
        spec.name,
        case.case_path,
        case.data_path,
        case.case_file_hash,
        case.content_hash,
        case.data.coordinate_system,
        spec.motion.mode,
        prepared.schedule.particle_count,
        spec.solver.integrator,
        spec.solver.backend,
        memory.worker_count,
        spec.time.start_s,
        spec.time.end_s,
        spec.time.dt_s,
        tuple(name for name, _, _ in prepared.dynamics.runtime.primitive_packing),
        models,
        run.field_time_split_s,
        case.data.geometry.group_names,
        memory.limit_bytes,
        memory.planned_bytes,
        memory.slab_particles,
        memory.load_peak_bytes,
        memory.prepare_peak_bytes,
        memory.run_peak_bytes,
        tuple(
            CheckedField(name, item.unit, item.components, item.stored_basis, item.layout)
            for name, item in prepared.fields.fields.items()
        ),
    )


def simulate(case: SimulationCase, output: _PathInput, *, workers: int = 1) -> RunSummary:
    """Run the supported production profile and atomically publish its result."""
    try:
        return run_simulation(case, output, workers=workers)
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
