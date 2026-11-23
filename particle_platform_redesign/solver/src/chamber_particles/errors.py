"""Public error categories shared by execution and lazy result readers."""


class CaseError(ValueError):
    """The canonical case is invalid or cannot be read."""


class SimulationError(RuntimeError):
    """Execution or a stored result cannot be completed or consumed safely."""


class IncompleteResultError(SimulationError):
    """A result has no authoritative completed payload for the operation."""
