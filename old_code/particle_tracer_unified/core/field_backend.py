"""Adapt configured field providers to the shared stage-sampling contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.domain import FieldRequest, StageFields

from .datamodel import FieldProviderND, TriangleMeshField2D
from .field_backend_reporting import (
    FIELD_BACKEND_RECTILINEAR,
    FIELD_BACKEND_TRIANGLE_MESH_2D,
    derived_quantity_names,
    field_backend_kind,
    field_backend_report,
    triangle_mesh_gradient_source_report,
)
from .field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
    point_within_axes,
    sample_quantity_series,
    sample_valid_mask,
    sample_valid_mask_status,
)
from .grid_sampling import locate_axis_interval
from .triangle_mesh_sampling_2d import (
    field_triangle_support_tolerance,
    locate_triangle_containing_point,
    sample_triangle_mesh_series,
    sample_triangle_mesh_status,
)

VALID_MASK_QUANTITY = "valid_mask_status"

_STATUS_REASON = {
    int(VALID_MASK_STATUS_CLEAN): "clean",
    int(VALID_MASK_STATUS_MIXED_STENCIL): "mixed_stencil",
    int(VALID_MASK_STATUS_HARD_INVALID): "hard_invalid",
}


def _status_reason(status: int) -> str:
    return str(_STATUS_REASON.get(int(status), "unknown"))


def _regular_cell_id(field, position: np.ndarray) -> int:
    axes = tuple(getattr(field, "axes", ()))
    if not point_within_axes(axes, np.asarray(position, dtype=np.float64)):
        return -1
    lows = []
    shape = []
    for axis_index, axis in enumerate(axes):
        values = np.asarray(axis, dtype=np.float64)
        lo, _hi, _alpha = locate_axis_interval(values, float(position[axis_index]))
        lows.append(int(lo))
        shape.append(max(int(values.size) - 1, 1))
    cell_id = 0
    stride = 1
    for lo, count in zip(reversed(lows), reversed(shape), strict=True):
        cell_id += int(lo) * stride
        stride *= int(count)
    return int(cell_id)


def _triangle_cell_id(field: TriangleMeshField2D, position: np.ndarray) -> int:
    tri_idx, _bary = locate_triangle_containing_point(
        vertices=field.mesh_vertices,
        triangles=field.mesh_triangles,
        accel_origin=field.accel_origin,
        accel_cell_size=field.accel_cell_size,
        accel_shape=field.accel_shape,
        accel_cell_offsets=field.accel_cell_offsets,
        accel_triangle_indices=field.accel_triangle_indices,
        position=np.asarray(position, dtype=np.float64),
        eps=field_triangle_support_tolerance(field),
    )
    return int(tri_idx)


def _cell_id_at(field_provider: FieldProviderND, position: np.ndarray) -> int:
    field = field_provider.field
    pos = np.asarray(position, dtype=np.float64)
    if isinstance(field, TriangleMeshField2D):
        return _triangle_cell_id(field, pos)
    return _regular_cell_id(field, pos)


def _valid_status_at(field_provider: FieldProviderND, position: np.ndarray) -> int:
    field = field_provider.field
    pos = np.asarray(position, dtype=np.float64)
    if isinstance(field, TriangleMeshField2D):
        return int(sample_triangle_mesh_status(field, pos))
    return int(
        sample_valid_mask_status(
            np.asarray(field.valid_mask, dtype=bool), field.axes, pos
        )
    )


def _quantity_at(
    field_provider: FieldProviderND,
    quantity_name: str,
    position: np.ndarray,
    t_eval: float,
    *,
    mode: str = "linear",
    default: float = np.nan,
) -> float:
    field = field_provider.field
    series = field.quantities.get(str(quantity_name))
    if series is None:
        return float(default)
    pos = np.asarray(position, dtype=np.float64)
    if isinstance(field, TriangleMeshField2D):
        value = sample_triangle_mesh_series(
            series, field, pos, float(t_eval), mode=mode
        )
        return float(default) if not np.isfinite(value) else float(value)
    if not bool(
        sample_valid_mask(np.asarray(field.valid_mask, dtype=bool), field.axes, pos)
    ):
        return float(default)
    return float(
        sample_quantity_series(series, field.axes, pos, float(t_eval), mode=mode)
    )


def _validated_sample_input(
    points_m: np.ndarray,
    time_s: float,
    spatial_dim: int,
) -> tuple[np.ndarray, float]:
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != spatial_dim:
        raise ValueError(f"points_m must have shape (particle, {spatial_dim})")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_m must contain only finite coordinates")
    time_value = float(time_s)
    if not np.isfinite(time_value):
        raise ValueError("time_s must be finite")
    return points, time_value


def _sample_provider_quantity(
    field_provider: FieldProviderND,
    quantity_name: str,
    points: np.ndarray,
    time_s: float,
    statuses: np.ndarray,
    *,
    interpolation: str,
) -> tuple[np.ndarray, bool]:
    if quantity_name == VALID_MASK_QUANTITY:
        return statuses, False
    if quantity_name not in field_provider.field.quantities:
        return np.full(points.shape[0], np.nan, dtype=np.float64), True

    sampled = np.full(points.shape[0], np.nan, dtype=np.float64)
    for index, point in enumerate(points):
        if statuses[index] != int(VALID_MASK_STATUS_HARD_INVALID):
            sampled[index] = _quantity_at(
                field_provider,
                quantity_name,
                point,
                time_s,
                mode=interpolation,
            )
    return sampled, False


@dataclass(frozen=True, slots=True)
class ProviderSamplingBackend:
    """Batch adapter for validation and comparison of provider quantities."""

    field_provider: FieldProviderND
    interpolation: str = "linear"

    def __post_init__(self) -> None:
        if self.interpolation != "linear":
            raise ValueError("ProviderSamplingBackend interpolation must be 'linear'")

    def sample(
        self,
        points_m: np.ndarray,
        time_s: float,
        request: FieldRequest,
    ) -> StageFields:
        field = self.field_provider.field
        raw_points = np.asarray(points_m, dtype=np.float64)
        spatial_dim = int(
            getattr(
                field,
                "spatial_dim",
                raw_points.shape[1] if raw_points.ndim == 2 else 0,
            )
        )
        points, time_value = _validated_sample_input(raw_points, time_s, spatial_dim)
        statuses = np.asarray(
            [_valid_status_at(self.field_provider, point) for point in points],
            dtype=np.uint8,
        )
        cell_ids = np.asarray(
            [_cell_id_at(self.field_provider, point) for point in points],
            dtype=np.int64,
        )
        values: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for name in request.quantities:
            values[name], is_missing = _sample_provider_quantity(
                self.field_provider,
                name,
                points,
                time_value,
                statuses,
                interpolation=self.interpolation,
            )
            if is_missing:
                missing.append(name)

        return StageFields(
            points_m=points,
            time_s=time_value,
            values=values,
            supported=statuses == int(VALID_MASK_STATUS_CLEAN),
            metadata={
                "backend_kind": str(field_backend_kind(self.field_provider)),
                "interpolation": self.interpolation,
                "valid_mask_status": statuses,
                "valid_mask_reason": tuple(
                    _status_reason(int(status)) for status in statuses
                ),
                "cell_id": cell_ids,
                "missing_quantities": tuple(missing),
            },
        )


__all__ = (
    "FIELD_BACKEND_RECTILINEAR",
    "FIELD_BACKEND_TRIANGLE_MESH_2D",
    "VALID_MASK_QUANTITY",
    "ProviderSamplingBackend",
    "derived_quantity_names",
    "field_backend_kind",
    "field_backend_report",
    "triangle_mesh_gradient_source_report",
)
