from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_GEOMETRY_CLEARANCE_FRACTION = 0.25
_SUPPORT_SPACING_ROUNDOFF = 1.0e-12


@dataclass(frozen=True)
class TraceRefinementPolicy:
    """Resolved internal limits for one deterministic segment trace."""

    on_boundary_tolerance_m: float
    support_spacing_m: float
    adaptive_substep_enabled: int
    adaptive_substep_max_splits: int
    interpolation_resolution_m: float = float("nan")
    """Length of one interpolation cell; a longer leaf crosses a cell face.

    The interpolant is only ``C0`` there, so the step-doubling estimate loses
    the smoothness its ``O(h^3)`` model assumes.  This is an accuracy
    requirement, not a safety one: exhausting the budget never terminates a
    particle.
    """

    @property
    def max_substeps(self) -> int:
        return 1 << max(0, int(self.adaptive_substep_max_splits))


@dataclass(frozen=True)
class TraceGeometryAssessment:
    """Geometry-only measurements used to decide whether a trace is resolved."""

    sagitta_m: float
    numerical_tolerance_m: float

    @property
    def needs_clearance(self) -> bool:
        return bool(
            np.isfinite(self.sagitta_m) and self.sagitta_m > self.numerical_tolerance_m
        )

    def requires_refinement(self, clearance_m: float) -> bool:
        if not np.isfinite(self.sagitta_m):
            return True
        if not self.needs_clearance:
            return False
        if not np.isfinite(clearance_m):
            return True
        threshold = max(
            self.numerical_tolerance_m,
            _GEOMETRY_CLEARANCE_FRACTION * float(clearance_m),
        )
        return bool(self.sagitta_m > threshold)


@dataclass(frozen=True)
class TraceRefinementDecision:
    """Immutable refinement decision for a trace at its current resolution.

    ``support_substeps`` and ``geometry_risk`` are safety requirements: an
    unmet one must stop the particle rather than accept an unproven segment.
    ``resolution_substeps`` is an accuracy requirement driven by the field's
    interpolation cell size; it raises the substep count while budget remains
    but never makes a trace unresolved.
    """

    geometry_risk: bool
    support_substeps: int
    max_substeps: int
    resolution_substeps: int = 0

    def needs_replay(self, *, current_substeps: int, complete_trace: bool) -> bool:
        current = max(1, int(current_substeps))
        return bool(
            (current > 1 and not complete_trace)
            or self.geometry_risk
            or int(self.support_substeps) > current
            or (
                int(self.resolution_substeps) > current
                and current < max(1, int(self.max_substeps))
            )
        )

    def minimum_substeps(self, *, current_substeps: int) -> int:
        current = max(1, int(current_substeps))
        maximum = max(1, int(self.max_substeps))
        requested = max(
            current,
            int(self.support_substeps),
            int(self.resolution_substeps),
        )
        if self.geometry_risk:
            requested = max(requested, 2, 2 * current)
        return int(min(maximum, requested))

    def resolved(self, *, current_substeps: int) -> bool:
        current = max(1, int(current_substeps))
        return bool(not self.geometry_risk and int(self.support_substeps) <= current)

    def limit_reached(self, *, current_substeps: int) -> bool:
        """Return whether refinement has exhausted its configured resolution.

        Reaching the resource limit is deliberately independent from resolving
        the trace.  Callers must fail closed when ``limit_reached`` is true but
        ``resolved`` is false; otherwise a narrow wall/support excursion can be
        silently accepted merely because the replay budget was exhausted.
        """

        current = max(1, int(current_substeps))
        return bool(current >= max(1, int(self.max_substeps)))


def trace_max_sagitta(start: np.ndarray, stage_points: np.ndarray) -> float:
    """Return the largest half-stage deviation from its substep chord."""

    trace = np.asarray(stage_points, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[0] < 2 or trace.shape[0] % 2 != 0:
        return float("inf")
    segment_start = np.asarray(start, dtype=np.float64)
    maximum = 0.0
    for row in range(0, int(trace.shape[0]), 2):
        midpoint = trace[row]
        endpoint = trace[row + 1]
        deviation = float(np.linalg.norm(midpoint - 0.5 * (segment_start + endpoint)))
        if not np.isfinite(deviation):
            return float("inf")
        maximum = max(maximum, deviation)
        segment_start = endpoint
    return float(maximum)


def assess_trace_geometry(
    start: np.ndarray,
    stage_points: np.ndarray,
    *,
    on_boundary_tolerance_m: float,
) -> TraceGeometryAssessment:
    sagitta = trace_max_sagitta(start, stage_points)
    # Coordinate roundoff is already part of the geometry-resolved boundary
    # policy.  Re-deriving an absolute metre floor here would make refinement
    # decisions change under a similarity scaling of the same problem.
    numerical_tolerance = float(on_boundary_tolerance_m)
    return TraceGeometryAssessment(
        sagitta_m=float(sagitta),
        numerical_tolerance_m=float(numerical_tolerance),
    )


def geometry_probe_points(start: np.ndarray, stage_points: np.ndarray) -> np.ndarray:
    """Return trace nodes and chord midpoints used for clearance sampling."""

    trace_nodes = np.vstack(
        (
            np.asarray(start, dtype=np.float64).reshape(1, -1),
            np.asarray(stage_points, dtype=np.float64),
        )
    )
    chord_midpoints = 0.5 * (trace_nodes[:-1] + trace_nodes[1:])
    return np.vstack((trace_nodes, chord_midpoints))


def minimum_geometry_clearance(signed_distance_m: np.ndarray) -> float:
    """Convert an inside-negative SDF sample into minimum inside clearance."""

    signed_distance = np.asarray(signed_distance_m, dtype=np.float64)
    if np.any(~np.isfinite(signed_distance)):
        return float("nan")
    if np.any(signed_distance >= 0.0):
        return 0.0
    return float(np.min(-signed_distance))


def support_spacing_from_backend(compiled: object) -> float:
    """Return the smallest resolved spacing where an invalid island may exist."""

    valid_mask = getattr(compiled, "valid_mask", None)
    if valid_mask is not None and bool(np.all(np.asarray(valid_mask, dtype=bool))):
        return float("nan")
    spacings: list[float] = []
    for axis in getattr(compiled, "axes", ()):
        values = np.asarray(axis, dtype=np.float64)
        differences = np.diff(values)
        finite = differences[np.isfinite(differences) & (differences > 0.0)]
        if finite.size:
            spacings.append(float(np.min(finite)))
    if spacings:
        return float(min(spacings))
    cell_size = np.asarray(
        getattr(compiled, "accel_cell_size", ()),
        dtype=np.float64,
    ).reshape(-1)
    finite_cell_size = cell_size[np.isfinite(cell_size) & (cell_size > 0.0)]
    return float(np.min(finite_cell_size)) if finite_cell_size.size else float("nan")


def segment_length_required_substeps(
    start: np.ndarray,
    stage_points: np.ndarray,
    *,
    current_substeps: int,
    target_length_m: float,
    max_substeps: int,
) -> int:
    """Return the uncapped substep count that keeps every leaf under a length.

    ``max_substeps`` remains an input so malformed traces can request one step
    beyond the configured budget.  A valid trace may likewise return a value
    greater than that budget.  Preserving that distinction lets the caller
    decide whether an unmet requirement is a safety failure or only a loss of
    accuracy.
    """

    current = max(1, int(current_substeps))
    maximum = max(1, int(max_substeps))
    if not np.isfinite(target_length_m) or target_length_m <= 0.0:
        return current
    trace = np.asarray(stage_points, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[0] == 0:
        return maximum + 1
    points = np.vstack((np.asarray(start, dtype=np.float64).reshape(1, -1), trace))
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if segment_lengths.size == 0 or np.any(~np.isfinite(segment_lengths)):
        return maximum + 1
    refinement = int(
        max(
            1,
            np.ceil(
                float(np.max(segment_lengths))
                / (float(target_length_m) * (1.0 + _SUPPORT_SPACING_ROUNDOFF))
            ),
        )
    )
    return int(max(1, current * refinement))


def interpolation_resolution_from_backend(compiled: object) -> float:
    """Return the length of one interpolation cell of a compiled field.

    For a regular grid this is the smallest axis spacing.  For a triangle mesh
    the candidate-grid cell is used: it is built from the mesh bounding box and
    triangle count, so it tracks element size without an extra pass over the
    mesh on every step.  ``NaN`` means the backend exposes no resolution and
    the accuracy requirement is simply not applied.
    """

    spacings: list[float] = []
    for axis in getattr(compiled, "axes", ()) or ():
        differences = np.diff(np.asarray(axis, dtype=np.float64))
        finite = differences[np.isfinite(differences) & (differences > 0.0)]
        if finite.size:
            spacings.append(float(np.min(finite)))
    if spacings:
        return float(min(spacings))
    cell_size = np.asarray(
        getattr(compiled, "accel_cell_size", ()),
        dtype=np.float64,
    ).reshape(-1)
    finite_cell_size = cell_size[np.isfinite(cell_size) & (cell_size > 0.0)]
    return float(np.min(finite_cell_size)) if finite_cell_size.size else float("nan")


__all__ = (
    "TraceGeometryAssessment",
    "TraceRefinementDecision",
    "TraceRefinementPolicy",
    "assess_trace_geometry",
    "geometry_probe_points",
    "interpolation_resolution_from_backend",
    "minimum_geometry_clearance",
    "segment_length_required_substeps",
    "support_spacing_from_backend",
    "trace_max_sagitta",
)
