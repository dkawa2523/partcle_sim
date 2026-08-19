"""Public COMSOL manifest model and runtime-provider facade."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from particle_tracer_unified.force_models import ForceModel

from . import _comsol_manifest_files as _files
from . import _comsol_manifest_parsing as _parsing
from . import _comsol_manifest_types as _types
from . import _comsol_manifest_validation as _validation

ALLOWED_COORDINATE_SYSTEMS = _types.ALLOWED_COORDINATE_SYSTEMS
BUILTIN_FIELD_SEMANTICS = _types.BUILTIN_FIELD_SEMANTICS
ComsolArtifact = _types.ComsolArtifact
ComsolFieldSpec = _types.ComsolFieldSpec
classify_field_semantic = _types.classify_field_semantic
field_target = _types.field_target
ComsolManifestValidation = _validation.ComsolManifestValidation


@dataclass(frozen=True)
class ComsolCaseManifest(ComsolManifestValidation):
    schema_version: int
    manifest_path: Path
    root_dir: Path
    model: Mapping[str, Any]
    coordinates: Mapping[str, Any]
    time: Mapping[str, Any]
    artifacts: Mapping[str, ComsolArtifact]
    fields: tuple[ComsolFieldSpec, ...]
    force_model: ForceModel
    metadata: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> ComsolCaseManifest:
        manifest_path, raw = _parsing.read_manifest(path)
        schema_version = _parsing.manifest_schema_version(raw, manifest_path)
        fields = _parsing.parse_manifest_fields(raw)
        artifacts = _parsing.parse_manifest_artifacts(raw)
        coordinates = _parsing.mapping(
            raw.get("coordinates", {}), context="coordinates"
        )
        force_model = _parsing.parse_manifest_forces(
            raw,
            spatial_dim=_parsing.manifest_spatial_dim(coordinates),
        )
        return cls(
            schema_version=schema_version,
            manifest_path=manifest_path,
            root_dir=manifest_path.parent,
            model=_parsing.mapping(raw.get("model", {}), context="model"),
            coordinates=coordinates,
            time=_parsing.mapping(raw.get("time", {}), context="time"),
            artifacts=artifacts,
            fields=fields,
            force_model=force_model,
            metadata=_parsing.mapping(raw.get("metadata", {}), context="metadata"),
            raw=dict(raw),
        )

    def resolve(self, rel_path: str | Path | None) -> Path | None:
        if rel_path is None:
            return None
        text = str(rel_path)
        if text == "":
            return None
        if text != text.strip():
            raise ValueError(
                "COMSOL manifest path must not contain leading or trailing whitespace"
            )
        path = Path(text)
        return path if path.is_absolute() else (self.root_dir / path).resolve()

    def artifact_path(self, name: str) -> Path | None:
        artifact = self.artifacts.get(str(name))
        return None if artifact is None else artifact.resolve(self.root_dir)

    @property
    def coordinate_system(self) -> str | None:
        value = self.coordinates.get("coordinate_system")
        return value if isinstance(value, str) else None

    @property
    def axis_order(self) -> tuple[str, ...]:
        raw = self.coordinates.get("axis_order")
        return tuple(raw) if isinstance(raw, (list, tuple)) else ()

    @property
    def coordinate_scale_m_per_model_unit(self) -> float:
        return float(self.coordinates["coordinate_scale_m_per_model_unit"])

    @property
    def time_support_s(self) -> tuple[float, float] | None:
        raw = self.time.get("support_s")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        return float(raw[0]), float(raw[1])

    @property
    def time_interpolation(self) -> str:
        value = self.time.get("interpolation")
        return value if isinstance(value, str) else ""

    def matches_time_support(self, actual: tuple[float, float]) -> bool:
        declared = self.time_support_s
        return declared is not None and _files.time_support_matches(declared, actual)

    @property
    def source_solution_number(self) -> int | None:
        value = self.metadata.get("source_solution_number")
        return int(value) if type(value) is int and int(value) > 0 else None

    @property
    def vacuum_domain_ids(self) -> tuple[int, ...]:
        raw = self.metadata.get("vacuum_domain_ids")
        if not isinstance(raw, (list, tuple)):
            return ()
        if any(type(value) is not int or int(value) <= 0 for value in raw):
            return ()
        return tuple(int(value) for value in raw)

    @property
    def geometry_source(self) -> str | None:
        value = self.metadata.get("geometry_source")
        return value if isinstance(value, str) and value else None

    def release_path(self) -> Path | None:
        return self.artifact_path("release")

    def boundaries_path(self) -> Path | None:
        return self.artifact_path("boundaries")

    def geometry_path(self) -> Path | None:
        return self.artifact_path("geometry")

    def field_path(self) -> Path | None:
        return self.artifact_path("field")

    def field_quantity_mapping(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for spec in self.fields:
            for component, source in spec.components.items():
                result[_types.field_target(spec.semantic_quantity, str(component))] = {
                    "source": str(source),
                    "unit": str(
                        _types.EXPECTED_SI_UNITS.get(
                            spec.semantic_quantity, spec.unit or ""
                        )
                    ),
                    "scale_to_si": float(spec.scale_to_si),
                    "semantic_quantity": spec.semantic_quantity,
                    "component": str(component),
                }
        return result

    def provider_config(self) -> dict[str, dict[str, Any]]:
        geometry = self.artifacts.get("geometry")
        field = self.artifacts.get("field")
        if geometry is None or field is None:
            return {}
        common = {
            "coordinate_scale_to_si": float(self.coordinate_scale_m_per_model_unit)
        }
        return {
            "geometry": {
                "kind": geometry.format,
                "npz_path": str(geometry.resolve(self.root_dir)),
                **common,
            },
            "field": {
                "kind": field.format,
                "npz_path": str(field.resolve(self.root_dir)),
                "quantity_mapping": self.field_quantity_mapping(),
                "strict_quantity_mapping": True,
                **common,
            },
        }


__all__ = (
    "ALLOWED_COORDINATE_SYSTEMS",
    "BUILTIN_FIELD_SEMANTICS",
    "ComsolArtifact",
    "ComsolCaseManifest",
    "ComsolFieldSpec",
    "classify_field_semantic",
)
