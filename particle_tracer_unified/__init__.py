"""Public v0.2 particle trajectory API."""

from ._version import PACKAGE_VERSION
from .application import (
    ArtifactManifest,
    RunStats,
    SimulationCase,
    SimulationPlan,
    SimulationResult,
    SimulationState,
    load_case,
    simulate,
    validate_case,
)
from .configuration import RunConfig
from .domain import (
    BoundaryHit,
    BoundaryQuery,
    FieldRequest,
    SamplingBackend,
    StageFields,
)
from .preflight_types import ValidationReport
from .writer import write_result

__version__ = PACKAGE_VERSION

__all__ = [
    "ArtifactManifest",
    "BoundaryHit",
    "BoundaryQuery",
    "FieldRequest",
    "RunConfig",
    "RunStats",
    "SamplingBackend",
    "SimulationCase",
    "SimulationPlan",
    "SimulationResult",
    "SimulationState",
    "StageFields",
    "ValidationReport",
    "__version__",
    "load_case",
    "simulate",
    "validate_case",
    "write_result",
]
