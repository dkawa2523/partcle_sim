"""Validate files and tables referenced by a COMSOL manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from particle_tracer_unified.core.numerical_contracts import float_values_equal_ulps
from particle_tracer_unified.integrity import sha256_file

from ._comsol_manifest_types import ComsolArtifact, expected_axes
from .canonical_tables import validate_particles_csv
from .comsol_boundary_reader import (
    ComsolBoundaryMapRow,
    ComsolWallLawRow,
    read_comsol_boundaries,
    validate_geometry_boundary_coverage,
)

REQUIRED_ARTIFACTS = {"release", "geometry", "field", "boundaries"}
ARTIFACT_FORMATS = {
    "release": {"canonical_particles_csv"},
    "geometry": {"precomputed_npz"},
    "field": {"precomputed_npz", "precomputed_triangle_mesh_npz"},
    "boundaries": {"canonical_boundaries_csv"},
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def time_support_matches(
    declared: tuple[float, float],
    actual: tuple[float, float],
) -> bool:
    return all(
        float_values_equal_ulps(expected, observed)
        for expected, observed in zip(declared, actual, strict=True)
    )


def artifact_contract_errors(name: str, artifact: ComsolArtifact) -> list[str]:
    errors: list[str] = []
    if not artifact.path:
        errors.append(f"artifacts.{name}.path is required")
    allowed_formats = ARTIFACT_FORMATS.get(name, {artifact.format})
    if artifact.format not in allowed_formats:
        errors.append(
            f"artifacts.{name}.format must be one of {sorted(allowed_formats)}, "
            f"got {artifact.format!r}"
        )
    if not SHA256_RE.fullmatch(artifact.sha256):
        errors.append(
            f"artifacts.{name}.sha256 must be a 64-character lowercase SHA-256 digest"
        )
    return errors


def artifact_file_errors(
    name: str,
    artifact: ComsolArtifact,
    path: Path,
    *,
    verify_hashes: bool,
) -> list[str]:
    if not path.is_file():
        return [f"artifacts.{name}.path does not exist: {path}"]
    errors: list[str] = []
    actual_size = path.stat().st_size
    if artifact.size_bytes is not None and artifact.size_bytes != actual_size:
        errors.append(
            f"artifacts.{name}.size_bytes mismatch: "
            f"expected {artifact.size_bytes}, got {actual_size}"
        )
    if verify_hashes and SHA256_RE.fullmatch(artifact.sha256):
        actual_hash = sha256_file(path)
        if actual_hash != artifact.sha256:
            errors.append(
                f"artifacts.{name}.sha256 mismatch: "
                f"expected {artifact.sha256}, got {actual_hash}"
            )
    return errors


@dataclass(frozen=True)
class BoundaryTables:
    boundary_rows: list[ComsolBoundaryMapRow]
    wall_rows: list[ComsolWallLawRow]


@dataclass(frozen=True)
class GeometryBoundaryParts:
    boundary_edge_part_ids: np.ndarray | None
    boundary_triangle_part_ids: np.ndarray | None

    @property
    def are_missing(self) -> bool:
        return (
            self.boundary_edge_part_ids is None
            and self.boundary_triangle_part_ids is None
        )


class ComsolManifestFileValidation:
    def _validate_artifacts(
        self: Any,
        *,
        verify_hashes: bool,
    ) -> list[str]:
        errors: list[str] = []
        missing = sorted(REQUIRED_ARTIFACTS - set(self.artifacts))
        if missing:
            errors.append(f"artifacts is missing required entries: {missing}")
        for name, artifact in sorted(self.artifacts.items()):
            path = artifact.resolve(self.root_dir)
            errors.extend(artifact_contract_errors(name, artifact))
            errors.extend(
                artifact_file_errors(
                    name,
                    artifact,
                    path,
                    verify_hashes=verify_hashes,
                )
            )
        return errors

    def _validate_release_file(self: Any, errors: list[str]) -> None:
        release_path = self.release_path()
        if release_path is None or not release_path.is_file():
            return
        try:
            coordinate_system = self.coordinate_system
            spatial_dim = len(expected_axes(coordinate_system))
            if spatial_dim:
                validate_particles_csv(
                    release_path,
                    spatial_dim=spatial_dim,
                    coordinate_system=str(coordinate_system),
                )
        except ValueError as exc:
            errors.append(str(exc))

    def _read_boundary_tables(self: Any, errors: list[str]) -> BoundaryTables | None:
        boundaries_path = self.boundaries_path()
        if boundaries_path is None or not boundaries_path.is_file():
            return None
        try:
            boundary_rows, wall_rows = read_comsol_boundaries(boundaries_path)
            return BoundaryTables(boundary_rows, wall_rows)
        except ValueError as exc:
            errors.append(str(exc))
            return None

    @staticmethod
    def _load_geometry_boundary_parts(path: Path) -> GeometryBoundaryParts:
        with np.load(path, allow_pickle=False) as payload:
            edge_parts = (
                np.asarray(payload["boundary_edge_part_ids"], dtype=np.int64)
                if "boundary_edge_part_ids" in payload
                else None
            )
            triangle_parts = (
                np.asarray(payload["boundary_triangle_part_ids"], dtype=np.int64)
                if "boundary_triangle_part_ids" in payload
                else None
            )
        return GeometryBoundaryParts(edge_parts, triangle_parts)

    def _validate_geometry_file(
        self: Any,
        boundaries: BoundaryTables | None,
        errors: list[str],
    ) -> None:
        geometry_path = self.geometry_path()
        if boundaries is None or geometry_path is None or not geometry_path.is_file():
            return
        try:
            geometry = self._load_geometry_boundary_parts(geometry_path)
            if geometry.are_missing:
                errors.append(
                    "geometry artifact must include explicit boundary part IDs"
                )
                return
            validate_geometry_boundary_coverage(
                geometry,
                boundaries.boundary_rows,
                boundaries.wall_rows,
                strict=True,
            )
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    @staticmethod
    def _load_field_inventory(path: Path) -> tuple[set[str], np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            names = set(payload.files)
            times = (
                np.asarray(payload["times"], dtype=np.float64)
                if "times" in payload
                else np.asarray([0.0])
            )
        return names, times

    def _missing_field_sources(self: Any, names: set[str]) -> list[str]:
        return sorted(
            {
                str(item["source"])
                for item in self.field_quantity_mapping().values()
                if str(item["source"]) not in names
            }
        )

    def _validate_field_time_support(
        self: Any,
        times: np.ndarray,
        errors: list[str],
    ) -> None:
        support = self.time_support_s
        if support is None or not times.size:
            return
        actual = (float(times[0]), float(times[-1]))
        if not time_support_matches(support, actual):
            errors.append(
                "time.support_s does not match field artifact: "
                f"declared={support}, actual={actual}"
            )

    def _validate_field_file(self: Any, errors: list[str]) -> None:
        field_path = self.field_path()
        if field_path is None or not field_path.is_file():
            return
        try:
            names, times = self._load_field_inventory(field_path)
            missing_sources = self._missing_field_sources(names)
            if missing_sources:
                errors.append(
                    "field artifact is missing manifest component arrays: "
                    f"{missing_sources}"
                )
            self._validate_field_time_support(times, errors)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    def _validate_v2_files(self: Any) -> list[str]:
        errors: list[str] = []
        self._validate_release_file(errors)
        boundaries = self._read_boundary_tables(errors)
        self._validate_geometry_file(boundaries, errors)
        self._validate_field_file(errors)
        return errors


__all__ = (
    "ARTIFACT_FORMATS",
    "REQUIRED_ARTIFACTS",
    "SHA256_RE",
    "ComsolManifestFileValidation",
    "artifact_contract_errors",
    "artifact_file_errors",
    "time_support_matches",
)
