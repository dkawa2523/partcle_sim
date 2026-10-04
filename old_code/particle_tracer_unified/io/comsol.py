"""Load runtime inputs declared by a validated COMSOL manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ._comsol_provider_validation import validate_comsol_runtime_provider
from ._comsol_release_projection import apply_release_projection
from .comsol_boundary_reader import (
    read_comsol_boundaries,
    wall_laws_to_boundaries,
)
from .comsol_manifest import ComsolCaseManifest
from .runtime_builder_support import LoadedRuntimeInputs
from .tables import load_particles_csv


def _required_input_paths(manifest: ComsolCaseManifest) -> tuple[Path, Path, Path]:
    release_path = manifest.release_path()
    boundaries_path = manifest.boundaries_path()
    geometry_path = manifest.geometry_path()
    if release_path is None or boundaries_path is None or geometry_path is None:
        raise ValueError("COMSOL manifest release and boundary paths must be present")
    return release_path, boundaries_path, geometry_path


def _projection_config(manifest: ComsolCaseManifest) -> dict[str, Any] | None:
    value = manifest.metadata.get("release_boundary_projection")
    return dict(value) if isinstance(value, Mapping) else None


def load_comsol_runtime_inputs(
    *,
    manifest: ComsolCaseManifest,
    spatial_dim: int,
) -> LoadedRuntimeInputs:
    release_path, boundaries_path, geometry_path = _required_input_paths(manifest)
    coordinate_system = manifest.coordinate_system
    if coordinate_system is None:
        raise ValueError("COMSOL manifest coordinate_system is required")

    particles = load_particles_csv(
        release_path,
        int(spatial_dim),
        coordinate_system,
    )
    particles, projection_report = apply_release_projection(
        particles,
        spatial_dim=spatial_dim,
        geometry_path=geometry_path,
        projection_config=_projection_config(manifest),
    )
    particles = replace(
        particles,
        metadata={
            **particles.metadata,
            "source": "comsol_release_table",
            "path": str(release_path),
            "coordinate_scale_m_per_model_unit": float(
                manifest.coordinate_scale_m_per_model_unit
            ),
            "release_boundary_projection": projection_report,
        },
    )
    boundary_rows, wall_rows = read_comsol_boundaries(boundaries_path)
    return LoadedRuntimeInputs(
        particles=particles,
        walls=wall_laws_to_boundaries(wall_rows, boundary_rows),
    )


__all__ = [
    "load_comsol_runtime_inputs",
    "validate_comsol_runtime_provider",
]
