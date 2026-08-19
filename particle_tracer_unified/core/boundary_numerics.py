"""Geometry-scaled tolerances shared by boundary operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np

BOUNDARY_NUMERICS_POLICY_VERSION = "geometry-scaled-float64-v1"

_CLASSIFICATION_RESOLUTION_FRACTION = 1.0e-10
_CONTACT_OFFSET_RESOLUTION_FRACTION = 1.0e-8
_CONTACT_OFFSET_TOLERANCE_MULTIPLIER = 8.0
_COORDINATE_ROUNDOFF_ULPS = 64.0
_MAX_OFFSET_RESOLUTION_FRACTION = 1.0e-2


@dataclass(frozen=True)
class BoundaryNumerics:
    """Immutable, SI-valued boundary policy resolved from one geometry."""

    policy_version: str
    reference_length_m: float
    resolution_length_m: float
    coordinate_roundoff_m: float
    classification_tolerance_m: float
    contact_offset_m: float
    radial_axis_tolerance_m: float

    def summary(self) -> Mapping[str, object]:
        return asdict(self)


def _positive_min(values: Iterable[np.ndarray | float]) -> float:
    minimum = np.inf
    for values_raw in values:
        array = np.asarray(values_raw, dtype=np.float64).reshape(-1)
        valid = array[np.isfinite(array) & (array > 0.0)]
        if valid.size:
            minimum = min(minimum, float(np.min(valid)))
    return float(minimum)


def _edge_lengths(edges_raw: object) -> list[np.ndarray]:
    if edges_raw is None:
        return []
    edges = np.asarray(edges_raw, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1] != 2:
        return []
    return [np.linalg.norm(edges[:, 1, :] - edges[:, 0, :], axis=1)]


def _triangle_lengths(triangles_raw: object) -> list[np.ndarray]:
    if triangles_raw is None:
        return []
    triangles = np.asarray(triangles_raw, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1] != 3:
        return []

    edge_lengths = tuple(
        np.linalg.norm(triangles[:, j, :] - triangles[:, i, :], axis=1)
        for i, j in ((0, 1), (1, 2), (2, 0))
    )
    doubled_area = np.linalg.norm(
        np.cross(
            triangles[:, 1, :] - triangles[:, 0, :],
            triangles[:, 2, :] - triangles[:, 0, :],
        ),
        axis=1,
    )
    lengths = list(edge_lengths)
    for edge_length in edge_lengths:
        positive = edge_length > 0.0
        lengths.append(doubled_area[positive] / edge_length[positive])
    return lengths


def _loop_lengths(loops_raw: Iterable[object]) -> list[np.ndarray]:
    lengths: list[np.ndarray] = []
    for loop_raw in loops_raw:
        loop = np.asarray(loop_raw, dtype=np.float64)
        if loop.ndim != 2 or loop.shape[0] < 2:
            continue
        closed = np.vstack((loop, loop[0]))
        lengths.append(np.linalg.norm(np.diff(closed, axis=0), axis=1))
    return lengths


def _primitive_lengths(geometry: object) -> list[np.ndarray]:
    lengths = _edge_lengths(getattr(geometry, "boundary_edges", None))
    lengths.extend(_triangle_lengths(getattr(geometry, "boundary_triangles", None)))
    if lengths:
        return lengths
    return _loop_lengths(getattr(geometry, "boundary_loops_2d", ()))


def _coordinate_magnitude(geometry: object, reference_length_m: float) -> float:
    magnitude = float(reference_length_m)
    for axis_raw in getattr(geometry, "axes", ()):
        axis = np.asarray(axis_raw, dtype=np.float64)
        if axis.size:
            magnitude = max(magnitude, float(np.max(np.abs(axis))))
    for name in ("boundary_edges", "boundary_triangles"):
        values_raw = getattr(geometry, name, None)
        if values_raw is not None:
            values = np.asarray(values_raw, dtype=np.float64)
            if values.size:
                magnitude = max(magnitude, float(np.max(np.abs(values))))
    return float(magnitude)


def scaled_classification_tolerance(
    coordinates_m: np.ndarray,
    resolution_length_m: float,
) -> tuple[float, float]:
    """Return ``(64 ULP roundoff, classification tolerance)`` in metres."""

    coordinates = np.asarray(coordinates_m, dtype=np.float64)
    resolution = float(resolution_length_m)
    if coordinates.size == 0 or np.any(~np.isfinite(coordinates)):
        raise ValueError("scaled classification tolerance requires finite coordinates")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("scaled classification tolerance requires positive resolution")
    coordinate_magnitude = max(resolution, float(np.max(np.abs(coordinates))))
    coordinate_roundoff = _COORDINATE_ROUNDOFF_ULPS * float(
        np.spacing(np.float64(coordinate_magnitude))
    )
    tolerance = max(
        _CLASSIFICATION_RESOLUTION_FRACTION * resolution,
        coordinate_roundoff,
    )
    return float(coordinate_roundoff), float(tolerance)


def _geometry_scale(geometry: object) -> tuple[tuple[np.ndarray, ...], float]:
    axes = tuple(
        np.asarray(axis, dtype=np.float64) for axis in getattr(geometry, "axes", ())
    )
    if not axes:
        raise ValueError("boundary numerics require geometry axes")
    spans = np.asarray(
        [float(axis[-1] - axis[0]) for axis in axes],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(spans)) or np.any(spans <= 0.0):
        raise ValueError(
            "boundary numerics require finite geometry axes with positive span"
        )
    return axes, float(np.max(spans))


def _resolution_length(geometry: object, axes: tuple[np.ndarray, ...]) -> float:
    sources: list[np.ndarray | float] = [np.diff(axis) for axis in axes]
    sources.extend(_primitive_lengths(geometry))
    resolution = _positive_min(sources)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(
            "boundary numerics could not resolve a positive geometry length"
        )
    return float(resolution)


def _contact_offset(
    resolution_length_m: float,
    classification_tolerance_m: float,
) -> float:
    offset = max(
        _CONTACT_OFFSET_RESOLUTION_FRACTION * resolution_length_m,
        _CONTACT_OFFSET_TOLERANCE_MULTIPLIER * classification_tolerance_m,
    )
    if offset >= _MAX_OFFSET_RESOLUTION_FRACTION * resolution_length_m:
        raise ValueError(
            "geometry coordinates are too poorly conditioned for reliable "
            "boundary contact: the float64 contact offset is at least 1 percent "
            "of the smallest resolved length; "
            "translate coordinates closer to the origin or use a larger geometry scale"
        )
    return float(offset)


def resolve_boundary_numerics(geometry_provider: object) -> BoundaryNumerics:
    """Derive a scale-covariant float64 boundary policy from geometry data.

    ``resolution_length_m`` is the smallest positive grid spacing or explicit
    boundary primitive length.  The classification tolerance is the larger of
    a small fraction of that resolved length and 64 coordinate ULPs.  The
    contact offset is deliberately larger than classification tolerance, while
    remaining negligible relative to the resolved geometry.
    """

    geometry = getattr(geometry_provider, "geometry", None)
    if geometry is None:
        raise ValueError("boundary numerics require a geometry provider")

    axes, reference_length = _geometry_scale(geometry)
    resolution_length = _resolution_length(geometry, axes)
    coordinate_roundoff, classification_tolerance = scaled_classification_tolerance(
        np.asarray(
            [_coordinate_magnitude(geometry, reference_length)],
            dtype=np.float64,
        ),
        resolution_length,
    )
    contact_offset = _contact_offset(
        resolution_length,
        classification_tolerance,
    )

    return BoundaryNumerics(
        policy_version=BOUNDARY_NUMERICS_POLICY_VERSION,
        reference_length_m=float(reference_length),
        resolution_length_m=float(resolution_length),
        coordinate_roundoff_m=float(coordinate_roundoff),
        classification_tolerance_m=float(classification_tolerance),
        contact_offset_m=float(contact_offset),
        radial_axis_tolerance_m=float(classification_tolerance),
    )


__all__ = (
    "BOUNDARY_NUMERICS_POLICY_VERSION",
    "BoundaryNumerics",
    "resolve_boundary_numerics",
    "scaled_classification_tolerance",
)
