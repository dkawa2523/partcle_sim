"""Coordinate-system normalization and axisymmetric RZ metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .boundary_numerics import scaled_classification_tolerance

SUPPORTED_COORDINATE_SYSTEMS = ("cartesian_xy", "axisymmetric_rz", "cartesian_xyz")


def default_coordinate_system(spatial_dim: int) -> str:
    dim = int(spatial_dim)
    if dim == 2:
        return "cartesian_xy"
    if dim == 3:
        return "cartesian_xyz"
    raise ValueError("spatial_dim must be 2 or 3")


def normalize_coordinate_system(value: Any, spatial_dim: int) -> str:
    dim = int(spatial_dim)
    raw = default_coordinate_system(dim) if value is None else str(value).strip()
    if not raw:
        raw = default_coordinate_system(dim)
    token = raw.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cartesian": default_coordinate_system(dim),
        "xy": "cartesian_xy",
        "cartesian_2d": "cartesian_xy",
        "cartesian_xy": "cartesian_xy",
        "axisymmetric": "axisymmetric_rz",
        "axisymmetric_2d": "axisymmetric_rz",
        "axisymmetric_rz": "axisymmetric_rz",
        "cylindrical_rz": "axisymmetric_rz",
        "r_z": "axisymmetric_rz",
        "rz": "axisymmetric_rz",
        "rz_axisymmetric": "axisymmetric_rz",
        "xyz": "cartesian_xyz",
        "cartesian_3d": "cartesian_xyz",
        "cartesian_xyz": "cartesian_xyz",
    }
    normalized = aliases.get(token)
    if normalized is None:
        supported = ", ".join(SUPPORTED_COORDINATE_SYSTEMS)
        raise ValueError(
            f"Unsupported coordinate_system={raw!r}; supported values are: {supported}"
        )
    if dim == 2 and normalized == "cartesian_xyz":
        raise ValueError("coordinate_system=cartesian_xyz requires spatial_dim=3")
    if dim == 3 and normalized != "cartesian_xyz":
        raise ValueError(
            "spatial_dim=3 currently supports coordinate_system=cartesian_xyz"
        )
    return normalized


def is_axisymmetric_rz(coordinate_system: Any, spatial_dim: int) -> bool:
    return (
        normalize_coordinate_system(coordinate_system, spatial_dim) == "axisymmetric_rz"
    )


def axisymmetric_rz_chart_state(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map a signed radial chart state to physical ``(r, z, v_r, v_z)``."""

    position = np.asarray(position_m, dtype=np.float64).copy()
    velocity = np.asarray(velocity_mps, dtype=np.float64).copy()
    sign = -1.0 if float(position[0]) < 0.0 else 1.0
    position[0] = abs(float(position[0]))
    velocity[0] *= sign
    return position, velocity, sign


def canonicalize_axisymmetric_rz_state(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unique externally visible RZ state with ``r >= 0``."""

    position = np.asarray(position_m, dtype=np.float64).copy()
    velocity = np.asarray(velocity_mps, dtype=np.float64).copy()
    flip = (position[..., 0] < 0.0) | (
        (position[..., 0] == 0.0) & (velocity[..., 0] < 0.0)
    )
    position[..., 0] = np.abs(position[..., 0])
    velocity[..., 0] = np.where(flip, -velocity[..., 0], velocity[..., 0])
    return position, velocity


def canonicalize_axisymmetric_rz_positions(points_m: np.ndarray) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64).copy()
    points[..., 0] = np.abs(points[..., 0])
    return points


def axis_names_for_coordinate_system(
    coordinate_system: Any,
    spatial_dim: int,
) -> tuple[str, ...]:
    normalized = normalize_coordinate_system(coordinate_system, spatial_dim)
    return {
        "axisymmetric_rz": ("r", "z"),
        "cartesian_xy": ("x", "y"),
        "cartesian_xyz": ("x", "y", "z"),
    }[normalized]


def ring_area_weight(radius_m: Any) -> Any:
    radius = np.asarray(radius_m, dtype=np.float64)
    if not np.all(np.isfinite(radius)):
        raise ValueError(
            "axisymmetric_rz ring_area_weight radius values must be finite"
        )
    if np.any(radius < 0.0):
        raise ValueError(
            "axisymmetric_rz ring_area_weight radius values must be non-negative"
        )
    weights = 2.0 * math.pi * radius
    if radius.ndim == 0:
        return float(weights)
    return weights


def validate_axisymmetric_rz_radial_axis(
    coordinate_system: Any,
    spatial_dim: int,
    axis_0: Sequence[float],
    *,
    context: str = "axis_0",
) -> None:
    if not is_axisymmetric_rz(coordinate_system, spatial_dim):
        return
    axis = np.asarray(axis_0, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(
            f"axisymmetric_rz radial {context} must be a non-empty 1D axis"
        )
    if not np.all(np.isfinite(axis)):
        raise ValueError(
            f"axisymmetric_rz radial {context} must contain only finite values"
        )
    if np.min(axis) < 0.0:
        raise ValueError(f"axisymmetric_rz radial {context} must be non-negative")


def _axis_tolerance(axis: np.ndarray) -> float:
    axis_values = np.asarray(axis, dtype=np.float64)
    diffs = np.diff(axis_values)
    positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if positive.size == 0:
        raise ValueError(
            "axisymmetric_rz reporting requires a positive radial grid spacing"
        )
    _roundoff, tolerance = scaled_classification_tolerance(
        axis_values,
        float(np.min(positive)),
    )
    return float(tolerance)


def _ring_weight_summary(axis_0: np.ndarray) -> dict[str, Any]:
    weights = np.asarray(ring_area_weight(axis_0), dtype=np.float64)
    return {
        "count": int(weights.size),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "sum": float(np.sum(weights)),
    }


def _axis_boundary_metadata(
    boundary_edges: Any,
    boundary_edge_part_ids: Any,
    tolerance_m: float,
) -> tuple[list[int], list[int]]:
    if boundary_edges is None:
        return [], []
    edges = np.asarray(boundary_edges, dtype=np.float64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2):
        return [], []

    on_axis = np.all(np.abs(edges[:, :, 0]) <= tolerance_m, axis=1)
    edge_indices = [int(value) for value in np.flatnonzero(on_axis).tolist()]
    if boundary_edge_part_ids is None or not edge_indices:
        return edge_indices, []
    raw_part_ids = np.asarray(boundary_edge_part_ids, dtype=np.int32)
    if raw_part_ids.size < edges.shape[0]:
        return edge_indices, []
    part_ids = sorted({int(raw_part_ids[index]) for index in edge_indices})
    return edge_indices, part_ids


def _rz_report(
    axis_0: np.ndarray,
    tolerance_m: float,
    edge_indices: list[int],
    part_ids: list[int],
) -> dict[str, Any]:
    return {
        "coordinate_system": "axisymmetric_rz",
        "axis_names": ["r", "z"],
        "semantics": "2d_meridional_rz",
        "radial_axis_name": "r",
        "axial_axis_name": "z",
        "radial_axis_min_m": float(np.min(axis_0)),
        "radial_axis_max_m": float(np.max(axis_0)),
        "radial_axis_nonnegative": 1,
        "r0_on_grid": int(bool(np.any(np.abs(axis_0) <= tolerance_m))),
        "r0_detection_tolerance_m": float(tolerance_m),
        "r0_axis_boundary_edge_count": len(edge_indices),
        "r0_axis_boundary_edge_indices": edge_indices,
        "r0_axis_boundary_part_ids": part_ids,
        "velocity_components": ["v_r", "v_z"],
        "v_theta_dynamics": "out_of_scope",
        "source_ring_weighting_policy": "not_applied_implicitly",
        "ring_area_weight_formula": "2*pi*r",
        "radial_ring_area_weight": _ring_weight_summary(axis_0),
    }


def axisymmetric_rz_geometry_report(
    *,
    coordinate_system: Any,
    spatial_dim: int,
    axes: Sequence[Sequence[float]],
    boundary_edges: Any = None,
    boundary_edge_part_ids: Any = None,
) -> dict[str, Any]:
    if not is_axisymmetric_rz(coordinate_system, spatial_dim):
        return {}
    axes_tuple = tuple(axes)
    if len(axes_tuple) < 2:
        raise ValueError("axisymmetric_rz geometry reporting requires r and z axes")
    axis_0 = np.asarray(axes_tuple[0], dtype=np.float64)
    validate_axisymmetric_rz_radial_axis(
        coordinate_system,
        spatial_dim,
        axis_0,
        context="axis_0",
    )
    tol = _axis_tolerance(axis_0)
    edge_indices, part_ids = _axis_boundary_metadata(
        boundary_edges,
        boundary_edge_part_ids,
        tol,
    )
    return _rz_report(axis_0, tol, edge_indices, part_ids)


def axisymmetric_rz_report_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    report = metadata.get("axisymmetric_rz")
    return dict(report) if isinstance(report, Mapping) else {}
