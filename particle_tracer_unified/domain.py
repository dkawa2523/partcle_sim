from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FieldRequest:
    """Semantic quantities required at one integration stage."""

    quantities: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = tuple(
            dict.fromkeys(
                str(name).strip() for name in self.quantities if str(name).strip()
            )
        )
        if not normalized:
            raise ValueError(
                "a field request must contain at least one semantic quantity"
            )
        object.__setattr__(self, "quantities", normalized)


@dataclass(frozen=True)
class StageFields:
    """One representation for all regular/mesh and scalar/batch samples."""

    points_m: np.ndarray
    time_s: float
    values: Mapping[str, np.ndarray]
    supported: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("StageFields.points_m must have shape (particle, 2|3)")
        if not np.isfinite(float(self.time_s)):
            raise ValueError("StageFields.time_s must be finite")
        supported = np.asarray(self.supported, dtype=bool)
        if supported.shape != (points.shape[0],):
            raise ValueError("StageFields.supported must have shape (particle,)")
        normalized: dict[str, np.ndarray] = {}
        for name, raw in self.values.items():
            key = str(name).strip()
            value = np.asarray(raw, dtype=np.float64)
            if not key:
                raise ValueError("StageFields quantity names must not be empty")
            if value.ndim == 0 or value.shape[0] != points.shape[0]:
                raise ValueError(
                    f"StageFields.values[{key!r}] must start with the "
                    "particle dimension"
                )
            normalized[key] = value
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "supported", supported)
        object.__setattr__(self, "values", normalized)

    @property
    def count(self) -> int:
        return int(self.points_m.shape[0])

    def require(self, name: str) -> np.ndarray:
        try:
            return self.values[str(name)]
        except KeyError:
            available = ", ".join(sorted(self.values)) or "none"
            raise KeyError(
                f"field quantity {name!r} was not sampled; available: {available}"
            ) from None


@runtime_checkable
class SamplingBackend(Protocol):
    """IO-free semantic field sampling contract used by the solver."""

    def sample(
        self,
        points_m: np.ndarray,
        time_s: float,
        request: FieldRequest,
    ) -> StageFields: ...


def sample_one(
    backend: SamplingBackend,
    point_m: Sequence[float],
    time_s: float,
    request: FieldRequest,
) -> StageFields:
    """Sample one point through the batch implementation."""

    point = np.asarray(point_m, dtype=np.float64)
    if point.ndim != 1:
        raise ValueError("point_m must be a one-dimensional coordinate")
    return backend.sample(point.reshape(1, -1), float(time_s), request)


@dataclass(frozen=True)
class BoundaryHit:
    """First point-particle center hit for one segment.

    ``alpha_hint`` is the segment fraction in ``[0, 1]``. Primitive metadata
    identifies the geometry element that supplied the canonical part ID; it
    does not carry or infer a wall law.
    """

    position: np.ndarray
    normal: np.ndarray
    part_id: int
    alpha_hint: float = 0.0
    primitive_id: int = -1
    primitive_kind: str = "unknown"
    is_ambiguous: bool = False

    def local_signed_distance(self, position: np.ndarray) -> float:
        normal = np.asarray(self.normal, dtype=np.float64)
        normal_magnitude = float(np.linalg.norm(normal))
        if normal_magnitude <= 1.0e-30:
            return float("nan")
        offset = np.asarray(position, dtype=np.float64) - np.asarray(
            self.position, dtype=np.float64
        )
        return float(np.dot(offset, normal / normal_magnitude))


BoundarySurfaceT_co = TypeVar("BoundarySurfaceT_co", covariant=True)


@runtime_checkable
class BoundaryQuery(Protocol[BoundarySurfaceT_co]):
    """Solver-facing point-particle center boundary contract.

    The surface type is supplied by the geometry adapter.  Keeping it generic
    lets the solver use an accelerated 3D surface without making the domain
    contract depend on the concrete geometry implementation.
    """

    @property
    def primary_hit_counter_key(self) -> str: ...

    @property
    def triangle_surface_3d(self) -> BoundarySurfaceT_co | None: ...

    def inside(self, point_m: np.ndarray) -> bool: ...

    def inside_strict(self, point_m: np.ndarray) -> bool: ...

    def contains(self, points_m: np.ndarray) -> np.ndarray: ...

    def first_hit(
        self, start_m: np.ndarray, end_m: np.ndarray
    ) -> BoundaryHit | None: ...

    def polyline_hit(
        self, start_m: np.ndarray, stage_points_m: np.ndarray
    ) -> BoundaryHit | None: ...

    def nearest_projection(
        self, point_m: np.ndarray, inside_reference_m: np.ndarray
    ) -> BoundaryHit | None: ...


__all__ = (
    "BoundaryHit",
    "BoundaryQuery",
    "FieldRequest",
    "SamplingBackend",
    "StageFields",
    "sample_one",
)
