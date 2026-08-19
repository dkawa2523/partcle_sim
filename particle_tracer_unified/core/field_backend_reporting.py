"""Field support and provenance reports, separate from runtime sampling."""

from __future__ import annotations

from typing import Any

import numpy as np

from .coordinate_systems import axis_names_for_coordinate_system
from .datamodel import FieldDataND, FieldProviderND, TriangleMeshField2D
from .field_sampling import (
    choose_electric_field_quantity_names,
    choose_velocity_quantity_names,
)

FIELD_BACKEND_RECTILINEAR = "regular_rectilinear"
FIELD_BACKEND_TRIANGLE_MESH_2D = "triangle_mesh_2d"

_DERIVED_QUANTITY_ALIASES = {
    "grad_T_x": ("grad_T_x", "dT_dx", "temperature_gradient_x", "grad_temperature_x"),
    "grad_T_y": ("grad_T_y", "dT_dy", "temperature_gradient_y", "grad_temperature_y"),
    "grad_E2_x": ("grad_E2_x", "dE2_dx", "grad_E_squared_x", "grad_esq_x"),
    "grad_E2_y": ("grad_E2_y", "dE2_dy", "grad_E_squared_y", "grad_esq_y"),
    "fluid_accel_x": (
        "fluid_accel_x",
        "fluid_acceleration_x",
        "material_accel_x",
        "a_fluid_x",
    ),
    "fluid_accel_y": (
        "fluid_accel_y",
        "fluid_acceleration_y",
        "material_accel_y",
        "a_fluid_y",
    ),
    "vorticity_z": ("vorticity_z", "omega_z", "curl_u_z"),
}


def field_backend_kind(field_provider: FieldProviderND | None) -> str:
    if field_provider is None:
        return ""
    field = field_provider.field
    if isinstance(field, TriangleMeshField2D):
        return FIELD_BACKEND_TRIANGLE_MESH_2D
    return str(
        getattr(field, "metadata", {}).get(
            "field_backend_kind", FIELD_BACKEND_RECTILINEAR
        )
    )


def _regular_field_support_report(field) -> dict[str, Any]:
    valid_mask = np.asarray(field.valid_mask, dtype=bool)
    node_count = int(valid_mask.size)
    valid_count = int(np.count_nonzero(valid_mask))
    axes = []
    for axis in field.axes:
        arr = np.asarray(axis, dtype=np.float64)
        axes.append(
            {
                "count": int(arr.size),
                "min": float(np.nanmin(arr)) if arr.size else float("nan"),
                "max": float(np.nanmax(arr)) if arr.size else float("nan"),
            }
        )
    support_phi = getattr(field, "support_phi", None)
    support_phi_summary: dict[str, Any] = {"available": False}
    if support_phi is not None:
        phi = np.asarray(support_phi, dtype=np.float64)
        finite = phi[np.isfinite(phi)]
        support_phi_summary = {
            "available": True,
            "finite_count": int(finite.size),
            "min": float(np.min(finite)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }
    return {
        "grid_shape": [int(v) for v in valid_mask.shape],
        "grid_node_count": node_count,
        "valid_node_count": valid_count,
        "invalid_node_count": int(node_count - valid_count),
        "valid_fraction": float(valid_count / node_count) if node_count else 0.0,
        "axes": axes,
        "support_phi": support_phi_summary,
    }


def derived_quantity_names(field: FieldDataND) -> dict[str, str]:
    quantities = getattr(field, "quantities", {})
    selected: dict[str, str] = {}
    for target, aliases in _DERIVED_QUANTITY_ALIASES.items():
        for name in aliases:
            if name in quantities:
                selected[str(target)] = str(name)
                break
    return selected


def _derived_quantity_source(*, exported: bool, derivable: bool) -> str:
    if exported:
        return "exported_quantity"
    if derivable:
        return "triangle_p1_fallback"
    return "unavailable"


def triangle_mesh_gradient_source_report(field: TriangleMeshField2D) -> dict[str, str]:
    quantities = getattr(field, "quantities", {})
    quantity_names = set(quantities)
    names = derived_quantity_names(field)
    gas_names = {"T", "temperature", "temperature_K", "gas_temperature"}
    electric_names = choose_electric_field_quantity_names(field, 2)
    has_velocity = len(choose_velocity_quantity_names(field, 2)) >= 2
    return {
        "grad_T": _derived_quantity_source(
            exported={"grad_T_x", "grad_T_y"} <= set(names),
            derivable=bool(quantity_names & gas_names),
        ),
        "grad_E2": _derived_quantity_source(
            exported={"grad_E2_x", "grad_E2_y"} <= set(names),
            derivable=len(electric_names) >= 2,
        ),
        "fluid_acceleration": _derived_quantity_source(
            exported={"fluid_accel_x", "fluid_accel_y"} <= set(names),
            derivable=has_velocity,
        ),
        "vorticity_z": _derived_quantity_source(
            exported="vorticity_z" in names,
            derivable=has_velocity,
        ),
    }


def _triangle_mesh_field_support_report(field: TriangleMeshField2D) -> dict[str, Any]:
    return {
        "mesh_vertex_count": int(field.mesh_vertices.shape[0]),
        "mesh_triangle_count": int(field.mesh_triangles.shape[0]),
        "accel_grid_shape": [int(v) for v in field.accel_shape],
        "triangle_gradient_sources": triangle_mesh_gradient_source_report(field),
    }


def _field_time_axis_report(field) -> dict[str, Any]:
    quantities = getattr(field, "quantities", {})
    reference_name = ""
    reference_times: np.ndarray | None = None
    mismatches = []
    for name in sorted(quantities.keys()):
        series = quantities[name]
        times = np.asarray(
            getattr(series, "times", np.asarray([0.0], dtype=np.float64)),
            dtype=np.float64,
        )
        if reference_times is None:
            reference_name = str(name)
            reference_times = times
            continue
        if times.shape != reference_times.shape or not np.allclose(
            times, reference_times, rtol=0.0, atol=0.0
        ):
            mismatches.append(str(name))

    times = (
        reference_times
        if reference_times is not None
        else np.asarray([0.0], dtype=np.float64)
    )
    finite = times[np.isfinite(times)]
    return {
        "time_mode": str(getattr(field, "time_mode", "steady")),
        "time_count": int(times.size),
        "time_min_s": float(np.min(finite)) if finite.size else float("nan"),
        "time_max_s": float(np.max(finite)) if finite.size else float("nan"),
        "quantity_time_axis_reference": reference_name,
        "quantity_time_axis_mismatch_count": len(mismatches),
        "quantity_time_axis_mismatches": mismatches[:20],
        "quantity_time_axis_mismatches_truncated": bool(len(mismatches) > 20),
    }


def field_backend_report(field_provider: FieldProviderND | None) -> dict[str, Any]:
    if field_provider is None:
        return {
            "field_backend_kind": "",
            "field_has_support_phi": 0,
            "field_support_phi_kind": "",
        }
    field = field_provider.field
    metadata = getattr(field, "metadata", {})
    spatial_dim = int(getattr(field, "spatial_dim", 0))
    coordinate_system = str(getattr(field, "coordinate_system", ""))
    raw_axis_names = getattr(field, "axis_names", None)
    axis_names = (
        tuple(str(v) for v in raw_axis_names)
        if raw_axis_names is not None
        else axis_names_for_coordinate_system(coordinate_system, spatial_dim)
    )
    report: dict[str, Any] = {
        "field_backend_kind": str(field_backend_kind(field_provider)),
        "spatial_dim": spatial_dim,
        "coordinate_system": coordinate_system,
        "axis_names": list(axis_names),
        "field_has_support_phi": int(getattr(field, "support_phi", None) is not None),
        "field_support_phi_kind": str(metadata.get("field_support_phi_kind", "")),
        "quantity_count": len(getattr(field, "quantities", {})),
        "time_axis": _field_time_axis_report(field),
    }
    if isinstance(field, TriangleMeshField2D):
        report.update(_triangle_mesh_field_support_report(field))
    else:
        report.update(_regular_field_support_report(field))
    return report


__all__ = (
    "FIELD_BACKEND_RECTILINEAR",
    "FIELD_BACKEND_TRIANGLE_MESH_2D",
    "derived_quantity_names",
    "field_backend_kind",
    "field_backend_report",
    "triangle_mesh_gradient_source_report",
)
