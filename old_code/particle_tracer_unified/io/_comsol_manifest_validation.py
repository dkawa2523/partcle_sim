"""Validate COMSOL manifest model, coordinate, time, and force semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ._comsol_manifest_files import (
    REQUIRED_ARTIFACTS,
    ComsolManifestFileValidation,
)
from ._comsol_manifest_parsing import mapping
from ._comsol_manifest_types import (
    ALLOWED_COORDINATE_SYSTEMS,
    EXPECTED_SI_UNITS,
    ComsolFieldSpec,
    classify_field_semantic,
    expected_axes,
    field_target,
)

V2_TOP_LEVEL_KEYS = {
    "schema_version",
    "model",
    "coordinates",
    "time",
    "artifacts",
    "fields",
    "forces",
    "metadata",
}
MODEL_KEYS = {"name", "study", "dataset", "solution"}
COORDINATE_KEYS = {
    "coordinate_system",
    "axis_order",
    "coordinate_scale_m_per_model_unit",
}
TIME_KEYS = {"interpolation", "support_s"}
ARTIFACT_KEYS = {"path", "format", "sha256", "size_bytes"}
FIELD_KEYS = {"artifact", "components", "unit", "scale_to_si"}


class ComsolManifestValidation(ComsolManifestFileValidation):
    def _validate_nested_keys(self: Any) -> list[str]:
        errors: list[str] = []

        def reject_unknown(
            context: str,
            payload: Mapping[str, Any],
            allowed: set[str],
        ) -> None:
            unknown = sorted(set(payload) - allowed)
            if unknown:
                errors.append(f"{context} has unknown keys: {unknown}")

        reject_unknown("model", self.model, MODEL_KEYS)
        reject_unknown("coordinates", self.coordinates, COORDINATE_KEYS)
        reject_unknown("time", self.time, TIME_KEYS)

        raw_artifacts = mapping(self.raw.get("artifacts", {}), context="artifacts")
        unknown_artifacts = sorted(set(raw_artifacts) - REQUIRED_ARTIFACTS)
        if unknown_artifacts:
            errors.append(f"artifacts has unknown entries: {unknown_artifacts}")
        for name, payload in raw_artifacts.items():
            if isinstance(payload, Mapping):
                reject_unknown(f"artifacts.{name}", payload, ARTIFACT_KEYS)

        raw_fields = self.raw.get("fields", {})
        if isinstance(raw_fields, Mapping):
            for name, payload in raw_fields.items():
                if isinstance(payload, Mapping):
                    reject_unknown(f"fields.{name}", payload, FIELD_KEYS)
        return errors

    def _validate_coordinate_system(self: Any, errors: list[str]) -> None:
        coord_system = self.coordinate_system
        raw_coord_system = self.coordinates.get("coordinate_system")
        if not isinstance(raw_coord_system, str):
            errors.append("coordinates.coordinate_system must be a string")
        elif raw_coord_system != raw_coord_system.strip():
            errors.append(
                "coordinates.coordinate_system must not contain leading or trailing "
                "whitespace"
            )
        if coord_system not in ALLOWED_COORDINATE_SYSTEMS:
            errors.append(
                "coordinates.coordinate_system must be one of "
                f"{sorted(ALLOWED_COORDINATE_SYSTEMS)}, got {coord_system!r}"
            )

    def _validate_coordinate_scale(self: Any, errors: list[str]) -> None:
        if "coordinate_scale_m_per_model_unit" not in self.coordinates:
            errors.append(
                "coordinates.coordinate_scale_m_per_model_unit is required for COMSOL "
                "faithful mode"
            )
            return
        try:
            scale = float(self.coordinates["coordinate_scale_m_per_model_unit"])
            if not math.isfinite(scale) or scale <= 0.0:
                errors.append(
                    "coordinates.coordinate_scale_m_per_model_unit must be positive "
                    "and finite"
                )
        except (TypeError, ValueError):
            errors.append(
                "coordinates.coordinate_scale_m_per_model_unit must be numeric"
            )

    def _validate_axis_order(self: Any, errors: list[str]) -> tuple[str, ...]:
        raw_axes = self.coordinates.get("axis_order")
        if isinstance(raw_axes, (list, tuple)):
            for index, axis in enumerate(raw_axes):
                if not isinstance(axis, str):
                    errors.append(f"coordinates.axis_order[{index}] must be a string")
                elif axis != axis.strip():
                    errors.append(
                        f"coordinates.axis_order[{index}] must not contain leading or "
                        "trailing whitespace"
                    )
        axes = expected_axes(self.coordinate_system)
        if self.axis_order != axes:
            errors.append(
                f"coordinates.axis_order must be {list(axes)!r}, "
                f"got {list(self.axis_order)!r}"
            )
        return axes

    def _validate_coordinates(self: Any, errors: list[str]) -> tuple[str, ...]:
        self._validate_coordinate_system(errors)
        self._validate_coordinate_scale(errors)
        return self._validate_axis_order(errors)

    def _validate_field_semantics(
        self: Any,
        field: ComsolFieldSpec,
        expected_axes: tuple[str, ...],
        errors: list[str],
    ) -> None:
        semantic = field.semantic_quantity
        semantic_kind = classify_field_semantic(semantic)
        if semantic_kind is None:
            errors.append(
                f"fields semantic key {semantic!r} must be an exact built-in name or "
                "an identifier-shaped generic scalar name"
            )
        if not field.components:
            errors.append(f"fields.{semantic}.components must not be empty")
        expected_components = (
            set(expected_axes) if semantic_kind == "vector" else {"value"}
        )
        if set(field.components) != expected_components:
            errors.append(
                f"fields.{semantic}.components must map {sorted(expected_components)}, "
                f"got {sorted(field.components)}"
            )

    def _validate_field_storage(
        self: Any,
        field: ComsolFieldSpec,
        errors: list[str],
    ) -> None:
        semantic = field.semantic_quantity
        if field.artifact != "field":
            errors.append(
                f"fields.{semantic}.artifact must be 'field'; multi-artifact field "
                "loading is not supported"
            )
        elif field.artifact not in self.artifacts:
            errors.append(
                f"fields.{semantic}.artifact must reference a declared artifact, "
                f"got {field.artifact!r}"
            )
        if not field.unit:
            errors.append(f"fields.{semantic}.unit is required")
        expected_unit = EXPECTED_SI_UNITS.get(semantic)
        if expected_unit and field.unit != expected_unit:
            errors.append(
                f"fields.{semantic}.unit must be {expected_unit!r}, got {field.unit!r}"
            )
        if not math.isfinite(field.scale_to_si) or field.scale_to_si <= 0.0:
            errors.append(f"fields.{semantic}.scale_to_si must be positive and finite")

    def _record_field_targets(
        self: Any,
        field: ComsolFieldSpec,
        seen_targets: set[str],
        errors: list[str],
    ) -> None:
        for component in field.components:
            target = field_target(field.semantic_quantity, component)
            if target in seen_targets:
                errors.append(
                    "fields map more than one source array to target quantity "
                    f"{target!r}"
                )
            seen_targets.add(target)

    def _validate_field_spec(
        self: Any,
        field: ComsolFieldSpec,
        expected_axes: tuple[str, ...],
        seen_targets: set[str],
        errors: list[str],
    ) -> None:
        self._validate_field_semantics(field, expected_axes, errors)
        self._validate_field_storage(field, errors)
        self._record_field_targets(field, seen_targets, errors)

    def _validate_fields(
        self: Any,
        expected_axes: tuple[str, ...],
        errors: list[str],
    ) -> None:
        if not self.fields:
            errors.append("fields must contain at least one COMSOL field mapping")
        seen_targets: set[str] = set()
        for field in self.fields:
            self._validate_field_spec(field, expected_axes, seen_targets, errors)

    @staticmethod
    def _required_force_semantic(force: Any) -> str | None:
        if force.name == "drag" and force.model != "none":
            return "velocity"
        if force.name in {"electric", "dielectrophoresis"}:
            return "electric_field"
        if force.name == "thermophoresis":
            return "temperature"
        return None

    def _validate_force_requirements(self: Any, errors: list[str]) -> None:
        enabled_forces = tuple(
            force
            for force in self.force_model.definitions()
            if force.name in self.force_model.declared and force.enabled
        )
        if not enabled_forces:
            errors.append("forces must list at least one enabled force")
        if not any(force.name == "drag" for force in enabled_forces):
            errors.append("forces must include one enabled drag inventory entry")
        self._validate_force_fields(enabled_forces, errors)

    def _validate_force_fields(
        self: Any,
        enabled_forces: tuple[Any, ...],
        errors: list[str],
    ) -> None:
        field_semantics = {field.semantic_quantity for field in self.fields}
        for force in enabled_forces:
            required_semantic = self._required_force_semantic(force)
            if (
                required_semantic is not None
                and required_semantic not in field_semantics
            ):
                errors.append(
                    f"enabled force {force.name!r} requires manifest semantic field "
                    f"{required_semantic!r}"
                )

    def _validate_manifest_keys_and_model(self: Any, errors: list[str]) -> None:
        unknown = sorted(set(self.raw) - V2_TOP_LEVEL_KEYS)
        if unknown:
            errors.append(f"unknown COMSOL manifest keys: {unknown}")
        errors.extend(self._validate_nested_keys())
        for key in ("name", "study", "dataset", "solution"):
            value = self.model.get(key)
            if not isinstance(value, str) or value == "":
                errors.append(f"model.{key} is required")
            elif value != value.strip():
                errors.append(
                    f"model.{key} must not contain leading or trailing whitespace"
                )

    def _validate_time_contract(self: Any, errors: list[str]) -> None:
        if "interpolation" not in self.time:
            errors.append("time.interpolation is required")
        if self.time_interpolation != "linear":
            errors.append("time.interpolation must be linear")
        support = self.time_support_s
        if support is None:
            errors.append("time.support_s must contain [start_s, end_s]")
        elif (
            not all(math.isfinite(value) for value in support)
            or support[0] < 0.0
            or support[1] < support[0]
        ):
            errors.append("time.support_s must be finite, non-negative, and ordered")

    def _validate_model_and_time(self: Any, errors: list[str]) -> None:
        self._validate_manifest_keys_and_model(errors)
        self._validate_time_contract(errors)

    def _validate_release_projection(self: Any, errors: list[str]) -> None:
        projection = self.metadata.get("release_boundary_projection")
        if projection is None:
            return
        if not isinstance(projection, Mapping):
            errors.append("metadata.release_boundary_projection must be a mapping")
            return
        if "inward_offset_m" in projection:
            errors.append(
                "metadata.release_boundary_projection.inward_offset_m is obsolete; "
                "boundary releases stay on their declared entity and the solver "
                "does not treat a segment departing from it as a hit"
            )
        unknown_projection = sorted(
            set(projection) - {"inward_offset_m", "tolerance_m"}
        )
        if unknown_projection:
            errors.append(
                "metadata.release_boundary_projection has unknown keys: "
                f"{unknown_projection}"
            )
        try:
            tolerance = float(projection["tolerance_m"])
        except (KeyError, TypeError, ValueError):
            errors.append(
                "metadata.release_boundary_projection.tolerance_m is required and "
                "must be numeric"
            )
            return
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            errors.append(
                "metadata.release_boundary_projection.tolerance_m must be positive "
                "and finite"
            )

    def _validate_metadata(self: Any, errors: list[str]) -> None:
        if self.source_solution_number is None:
            errors.append("metadata.source_solution_number must be a positive integer")
        raw_vacuum_domains = self.metadata.get("vacuum_domain_ids")
        if not isinstance(raw_vacuum_domains, (list, tuple)) or not raw_vacuum_domains:
            errors.append("metadata.vacuum_domain_ids must be a non-empty integer list")
        elif any(
            type(value) is not int or int(value) <= 0 for value in raw_vacuum_domains
        ) or len(set(raw_vacuum_domains)) != len(raw_vacuum_domains):
            errors.append(
                "metadata.vacuum_domain_ids must contain unique positive integers"
            )
        raw_geometry_source = self.metadata.get("geometry_source")
        if raw_geometry_source != "explicit_comsol_vacuum_domain_selection":
            errors.append(
                "metadata.geometry_source must be "
                "'explicit_comsol_vacuum_domain_selection'"
            )
        self._validate_release_projection(errors)

    def validate(
        self: Any,
        *,
        strict: bool = True,
        verify_hashes: bool = True,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != 2:
            errors.append(f"schema_version must be 2, got {self.schema_version}")
        expected_axes = self._validate_coordinates(errors)
        self._validate_fields(expected_axes, errors)
        self._validate_force_requirements(errors)
        self._validate_model_and_time(errors)
        self._validate_metadata(errors)
        errors.extend(self._validate_artifacts(verify_hashes=verify_hashes))
        errors.extend(self._validate_v2_files())
        if strict and errors:
            raise ValueError(
                "Invalid COMSOL manifest:\n" + "\n".join(f"- {item}" for item in errors)
            )
        return errors


__all__ = ("ComsolManifestValidation",)
