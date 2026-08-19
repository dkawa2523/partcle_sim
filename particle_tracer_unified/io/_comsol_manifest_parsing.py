"""Parse COMSOL manifest YAML into typed values without file validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from particle_tracer_unified.force_models import ForceModel, parse_manifest_force_model

from ._comsol_manifest_types import ComsolArtifact, ComsolFieldSpec


def mapping(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return dict(value)


def exact_text(value: Any, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    if value != value.strip():
        raise ValueError(f"{context} must not contain leading or trailing whitespace")
    if not allow_empty and value == "":
        raise ValueError(f"{context} must not be empty")
    return value


def read_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"COMSOL manifest root must be a mapping: {manifest_path}")
    return manifest_path, dict(raw)


def manifest_schema_version(raw: Mapping[str, Any], manifest_path: Path) -> int:
    schema_version = int(raw.get("schema_version", 0))
    if schema_version != 2:
        raise ValueError(
            "COMSOL manifest schema_version must be 2; migrate legacy manifest "
            f"{manifest_path}"
        )
    return schema_version


def _field_items(raw_fields: Any) -> list[tuple[Any, Mapping[str, Any]]]:
    if not isinstance(raw_fields, Mapping):
        raise ValueError("COMSOL manifest fields must be a semantic-quantity mapping")
    items: list[tuple[Any, Mapping[str, Any]]] = []
    for semantic, payload in raw_fields.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"COMSOL manifest fields.{semantic} must be a mapping")
        items.append((semantic, payload))
    return items


def _field_components(
    semantic_name: str,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    components_raw = payload.get("components", {}) or {}
    if not isinstance(components_raw, Mapping):
        raise ValueError(
            f"COMSOL manifest fields.{semantic_name}.components must be a mapping"
        )
    return {
        exact_text(component, context=f"fields.{semantic_name}.components key"): (
            exact_text(
                source,
                context=f"fields.{semantic_name}.components.{component}",
            )
        )
        for component, source in components_raw.items()
    }


def _parse_field(semantic: Any, payload: Mapping[str, Any]) -> ComsolFieldSpec:
    semantic_name = exact_text(semantic, context="fields semantic quantity")
    unit_raw = payload.get("unit")
    return ComsolFieldSpec(
        semantic_quantity=semantic_name,
        components=_field_components(semantic_name, payload),
        unit=(
            None
            if unit_raw is None
            else exact_text(unit_raw, context=f"fields.{semantic_name}.unit")
        ),
        scale_to_si=float(payload.get("scale_to_si", 1.0)),
        artifact=exact_text(
            payload.get("artifact", "field"),
            context=f"fields.{semantic_name}.artifact",
        ),
    )


def parse_manifest_fields(raw: Mapping[str, Any]) -> tuple[ComsolFieldSpec, ...]:
    return tuple(
        _parse_field(semantic, payload)
        for semantic, payload in _field_items(raw.get("fields", {}))
    )


def _parse_artifact(name: Any, payload: Any) -> ComsolArtifact:
    item = mapping(payload, context=f"artifacts.{name}")
    artifact_name = exact_text(name, context="artifacts entry name")
    size_raw = item.get("size_bytes")
    return ComsolArtifact(
        name=artifact_name,
        path=exact_text(
            item.get("path", ""),
            context=f"artifacts.{artifact_name}.path",
            allow_empty=True,
        ),
        sha256=exact_text(
            item.get("sha256", ""),
            context=f"artifacts.{artifact_name}.sha256",
            allow_empty=True,
        ),
        format=exact_text(
            item.get("format", ""),
            context=f"artifacts.{artifact_name}.format",
            allow_empty=True,
        ),
        size_bytes=None if size_raw is None else int(size_raw),
    )


def parse_manifest_artifacts(raw: Mapping[str, Any]) -> dict[str, ComsolArtifact]:
    raw_artifacts = mapping(raw.get("artifacts", {}), context="artifacts")
    artifacts = (
        _parse_artifact(name, payload) for name, payload in raw_artifacts.items()
    )
    return {artifact.name: artifact for artifact in artifacts}


def manifest_spatial_dim(coordinates: Mapping[str, Any]) -> int:
    coordinate_system = coordinates.get("coordinate_system")
    if not isinstance(coordinate_system, str):
        return 2
    return {
        "cartesian_xy": 2,
        "axisymmetric_rz": 2,
        "cartesian_xyz": 3,
    }.get(coordinate_system, 2)


def parse_manifest_forces(
    raw: Mapping[str, Any],
    *,
    spatial_dim: int,
) -> ForceModel:
    raw_forces = raw.get("forces", []) or []
    if not isinstance(raw_forces, (list, tuple)):
        raise ValueError("COMSOL manifest forces must be a list")
    return parse_manifest_force_model(raw_forces, spatial_dim=spatial_dim)


__all__ = (
    "exact_text",
    "manifest_schema_version",
    "manifest_spatial_dim",
    "mapping",
    "parse_manifest_artifacts",
    "parse_manifest_fields",
    "parse_manifest_forces",
    "read_manifest",
)
