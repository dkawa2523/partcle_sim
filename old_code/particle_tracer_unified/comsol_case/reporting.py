"""Human-readable build summary assembly."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
)

from .mesh import ParsedMesh


def build_summary(
    *,
    mphtxt_path: Path,
    out_dir: Path,
    mesh: ParsedMesh,
    arrays: Mapping[str, Any],
    geometry_npz: Path,
    entity_map_files: Mapping[str, str],
    geometry_metadata: Mapping[str, Any],
    field_summary: Mapping[str, Any],
    diagnostic_spacing: float,
    coordinate_scale: float,
    vacuum_domain_ids: tuple[int, ...],
    coordinate_system: str,
    profile_name: str,
    axisymmetric_report: Mapping[str, Any],
) -> dict[str, Any]:
    boundary_parts = np.unique(arrays["boundary_part_ids"]).astype(int).tolist()
    generated = out_dir / "generated"
    summary: dict[str, Any] = {
        "source_mphtxt": str(mphtxt_path.resolve()),
        "sdim": int(mesh.sdim),
        "vertex_count": int(mesh.vertices.shape[0]),
        "edge_count": int(mesh.type_blocks["edg"].elements.shape[0])
        if "edg" in mesh.type_blocks
        else 0,
        "tri_count": int(mesh.type_blocks["tri"].elements.shape[0])
        if "tri" in mesh.type_blocks
        else 0,
        "quad_count": int(mesh.type_blocks["quad"].elements.shape[0])
        if "quad" in mesh.type_blocks
        else 0,
        "surface_triangle_count": int(arrays["triangles"].shape[0]),
        "surface_triangle_part_ids": sorted(
            np.unique(np.asarray(arrays["triangle_part_ids"], dtype=np.int32))
            .astype(int)
            .tolist()
        ),
        "quad_part_ids": sorted(
            np.unique(np.asarray(arrays["quad_part_ids"], dtype=np.int32))
            .astype(int)
            .tolist()
        ),
        "derived_boundary_edge_count": int(arrays["boundary_edge_count"]),
        "boundary_part_ids": boundary_parts,
        "bounds": {
            "xmin": float(arrays["axes_x"][0]),
            "xmax": float(arrays["axes_x"][-1]),
            "ymin": float(arrays["axes_y"][0]),
            "ymax": float(arrays["axes_y"][-1]),
        },
        "generated_files": {
            "geometry_npz": str(geometry_npz.relative_to(out_dir)),
            **{
                key: str((generated / filename).relative_to(out_dir))
                for key, filename in entity_map_files.items()
            },
        },
        "grid_axes": {
            "x_count": len(arrays["axes_x"]),
            "y_count": len(arrays["axes_y"]),
        },
        "geometry_mode": str(geometry_metadata["source_kind"]),
        "field_mode": field_summary["mode"],
        "field_summary": dict(field_summary),
        "field_axis_alignment": dict(field_summary.get("axis_alignment", {})),
        "diagnostic_grid_spacing_m": diagnostic_spacing,
        "field_ghost_cells": 0,
        "coordinate_unit": "m",
        "coordinate_scale_m_per_model_unit": coordinate_scale,
        "vacuum_domain_ids": list(vacuum_domain_ids),
        "coordinate_system": coordinate_system,
        "axis_names": list(axis_names_for_coordinate_system(coordinate_system, 2)),
        "profile": profile_name,
        "note": (
            "Physical geometry is the boundary of explicit COMSOL vacuum domains; "
            "field valid_mask remains field support only."
        ),
    }
    if axisymmetric_report:
        summary["axisymmetric_rz"] = dict(axisymmetric_report)
    return summary
