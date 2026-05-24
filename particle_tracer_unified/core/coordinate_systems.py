from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


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
        raise ValueError(f"Unsupported coordinate_system={raw!r}; supported values are: {supported}")
    if dim == 2 and normalized == "cartesian_xyz":
        raise ValueError("coordinate_system=cartesian_xyz requires spatial_dim=3")
    if dim == 3 and normalized != "cartesian_xyz":
        raise ValueError("spatial_dim=3 currently supports coordinate_system=cartesian_xyz")
    return normalized


def is_axisymmetric_rz(coordinate_system: Any, spatial_dim: int) -> bool:
    return normalize_coordinate_system(coordinate_system, spatial_dim) == "axisymmetric_rz"


def axis_names_for_coordinate_system(coordinate_system: Any, spatial_dim: int) -> tuple[str, ...]:
    normalized = normalize_coordinate_system(coordinate_system, spatial_dim)
    if normalized == "axisymmetric_rz":
        return ("r", "z")
    if normalized == "cartesian_xy":
        return ("x", "y")
    if normalized == "cartesian_xyz":
        return ("x", "y", "z")
    raise ValueError(f"Unsupported coordinate_system={normalized!r}")


def ring_area_weight(radius_m: Any) -> Any:
    radius = np.asarray(radius_m, dtype=np.float64)
    if not np.all(np.isfinite(radius)):
        raise ValueError("axisymmetric_rz ring_area_weight radius values must be finite")
    if np.any(radius < 0.0):
        raise ValueError("axisymmetric_rz ring_area_weight radius values must be non-negative")
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
        raise ValueError(f"axisymmetric_rz radial {context} must be a non-empty 1D axis")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"axisymmetric_rz radial {context} must contain only finite values")
    if np.min(axis) < 0.0:
        raise ValueError(f"axisymmetric_rz radial {context} must be non-negative")


def _axis_tolerance(axis: np.ndarray) -> float:
    diffs = np.diff(np.asarray(axis, dtype=np.float64))
    positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    step_tol = float(np.min(positive)) * 1.0e-9 if positive.size else 0.0
    return max(1.0e-12, step_tol)


def _ring_weight_summary(axis_0: np.ndarray) -> dict[str, Any]:
    weights = np.asarray(ring_area_weight(axis_0), dtype=np.float64)
    finite = weights[np.isfinite(weights)]
    if finite.size == 0:
        return {"count": 0, "min": None, "max": None, "sum": 0.0}
    return {
        "count": int(weights.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "sum": float(np.sum(finite)),
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
    if int(spatial_dim) != 2:
        raise ValueError("axisymmetric_rz geometry reporting requires spatial_dim=2")
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
    finite = axis_0[np.isfinite(axis_0)]
    edge_indices: list[int] = []
    part_ids: list[int] = []
    if boundary_edges is not None:
        edges = np.asarray(boundary_edges, dtype=np.float64)
        if edges.ndim == 3 and edges.shape[1:] == (2, 2):
            mask = np.all(np.abs(edges[:, :, 0]) <= tol, axis=1)
            edge_indices = [int(v) for v in np.flatnonzero(mask).tolist()]
            if boundary_edge_part_ids is not None and edge_indices:
                raw_part_ids = np.asarray(boundary_edge_part_ids, dtype=np.int32)
                if raw_part_ids.size >= edges.shape[0]:
                    part_ids = sorted({int(raw_part_ids[idx]) for idx in edge_indices})
    return {
        "coordinate_system": "axisymmetric_rz",
        "axis_names": ["r", "z"],
        "radial_axis_name": "r",
        "axial_axis_name": "z",
        "radial_axis_min_m": float(np.min(finite)) if finite.size else None,
        "radial_axis_max_m": float(np.max(finite)) if finite.size else None,
        "radial_axis_nonnegative": 1,
        "r0_on_grid": int(bool(np.any(np.abs(axis_0) <= tol))),
        "r0_detection_tolerance_m": float(tol),
        "r0_axis_boundary_edge_count": int(len(edge_indices)),
        "r0_axis_boundary_edge_indices": edge_indices,
        "r0_axis_boundary_part_ids": part_ids,
        "axis_boundary_policy": "report_only_collision_unchanged",
        "collision_behavior": "unchanged",
        "ring_area_weight_formula": "2*pi*r",
        "radial_ring_area_weight": _ring_weight_summary(axis_0),
    }


def axisymmetric_rz_report_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    report = metadata.get("axisymmetric_rz")
    return dict(report) if isinstance(report, Mapping) else {}
