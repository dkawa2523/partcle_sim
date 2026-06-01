from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .comsol_release_reader import validate_comsol_release_table_header


ALLOWED_COORDINATE_SYSTEMS = {"cartesian_xy", "axisymmetric_rz", "cartesian_xyz"}
ALLOWED_FIELD_PHYSICAL_QUANTITIES = {
    "velocity",
    "electric_field",
    "force",
    "acceleration",
    "density",
    "dynamic_viscosity",
    "temperature",
    "pressure",
    "scalar",
}
MODE_COMSOL_FAITHFUL = "comsol_faithful"
MODE_SURFACE_RELEASE_PRODUCTION = "surface_release_production"
SUPPORTED_RUN_MODES = {MODE_COMSOL_FAITHFUL, MODE_SURFACE_RELEASE_PRODUCTION}


@dataclass(frozen=True)
class ComsolFieldSpec:
    name: str
    components: Mapping[str, str]
    unit: str | None = None
    mesh: str | None = None
    interpolation: str | None = None
    physical_quantity: str | None = None


@dataclass(frozen=True)
class ComsolCaseManifest:
    schema_version: int
    root_dir: Path
    model: Mapping[str, Any]
    coordinates: Mapping[str, Any]
    fields: tuple[ComsolFieldSpec, ...]
    particles: Mapping[str, Any]
    forces: tuple[Mapping[str, Any], ...]
    boundaries: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ComsolCaseManifest":
        manifest_path = Path(path).resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"COMSOL manifest root must be a mapping: {manifest_path}")
        field_specs = []
        for item in raw.get("fields", []) or []:
            if not isinstance(item, Mapping):
                raise ValueError("COMSOL manifest fields entries must be mappings")
            field_specs.append(
                ComsolFieldSpec(
                    name=str(item.get("name", "")).strip(),
                    components=dict(item.get("components", {}) or {}),
                    unit=item.get("unit"),
                    mesh=item.get("mesh"),
                    interpolation=item.get("interpolation"),
                    physical_quantity=(
                        None
                        if item.get("physical_quantity") is None
                        else str(item.get("physical_quantity")).strip().lower()
                    ),
                )
            )
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            root_dir=manifest_path.parent,
            model=dict(raw.get("model", {}) or {}),
            coordinates=dict(raw.get("coordinates", {}) or {}),
            fields=tuple(field_specs),
            particles=dict(raw.get("particles", {}) or {}),
            forces=tuple(dict(item) for item in (raw.get("forces", []) or [])),
            boundaries=dict(raw.get("boundaries", {}) or {}),
            raw=dict(raw),
        )

    def resolve(self, rel_path: str | Path | None) -> Path | None:
        if rel_path is None or str(rel_path).strip() == "":
            return None
        path = Path(str(rel_path))
        return path if path.is_absolute() else (self.root_dir / path).resolve()

    @property
    def coordinate_system(self) -> str | None:
        value = self.coordinates.get("coordinate_system", self.coordinates.get("frame"))
        return None if value is None else str(value).strip()

    @property
    def coordinate_scale_m_per_model_unit(self) -> float:
        return float(self.coordinates["coordinate_scale_m_per_model_unit"])

    @property
    def release_velocity_scale_mps_per_input_unit(self) -> float:
        return float(self.particles.get("release_velocity_scale_mps_per_input_unit", 1.0))

    def validate(self, *, strict: bool = True) -> list[str]:
        errors: list[str] = []

        coord_system = self.coordinate_system
        if coord_system not in ALLOWED_COORDINATE_SYSTEMS:
            errors.append(
                "coordinates.coordinate_system must be one of "
                f"{sorted(ALLOWED_COORDINATE_SYSTEMS)}, got {coord_system!r}"
            )
        if "coordinate_scale_m_per_model_unit" not in self.coordinates:
            errors.append("coordinates.coordinate_scale_m_per_model_unit is required for COMSOL faithful mode")
        else:
            try:
                scale = float(self.coordinates["coordinate_scale_m_per_model_unit"])
                if scale <= 0.0:
                    errors.append("coordinates.coordinate_scale_m_per_model_unit must be positive")
            except (TypeError, ValueError):
                errors.append("coordinates.coordinate_scale_m_per_model_unit must be numeric")

        if not self.fields:
            errors.append("fields must contain at least one COMSOL field mapping")
        field_names = {field.name for field in self.fields if field.name}
        if not ({"u", "E"} & field_names):
            errors.append("fields must include at least velocity 'u' or electric field 'E'")
        for field in self.fields:
            if not field.name:
                errors.append("fields entries must include name")
            if strict and not field.physical_quantity:
                errors.append(f"fields[{field.name}].physical_quantity is required for COMSOL faithful mode")
            if field.physical_quantity and field.physical_quantity not in ALLOWED_FIELD_PHYSICAL_QUANTITIES:
                errors.append(
                    f"fields[{field.name}].physical_quantity must be one of "
                    f"{sorted(ALLOWED_FIELD_PHYSICAL_QUANTITIES)}, got {field.physical_quantity!r}"
                )

        release_table = self.resolve(self.particles.get("release_table"))
        if release_table is None or not release_table.exists():
            errors.append(f"particles.release_table does not exist: {release_table}")
        else:
            try:
                release_spatial_dim = 3 if coord_system == "cartesian_xyz" else 2
                validate_comsol_release_table_header(
                    release_table,
                    spatial_dim=release_spatial_dim,
                    strict=strict,
                )
            except ValueError as exc:
                errors.append(str(exc))
        if "release_velocity_scale_mps_per_input_unit" in self.particles:
            try:
                velocity_scale = float(self.particles["release_velocity_scale_mps_per_input_unit"])
                if not math.isfinite(velocity_scale) or velocity_scale <= 0.0:
                    errors.append("particles.release_velocity_scale_mps_per_input_unit must be positive")
            except (TypeError, ValueError):
                errors.append("particles.release_velocity_scale_mps_per_input_unit must be numeric")

        boundary_map = self.resolve(self.boundaries.get("map_file"))
        if boundary_map is None or not boundary_map.exists():
            errors.append(f"boundaries.map_file does not exist: {boundary_map}")

        wall_law = self.resolve(self.boundaries.get("wall_law_file"))
        if wall_law is None or not wall_law.exists():
            errors.append(f"boundaries.wall_law_file does not exist: {wall_law}")

        enabled_forces = [force for force in self.forces if bool(force.get("enabled", True))]
        if strict and not enabled_forces:
            errors.append("forces must list at least one enabled force or explicitly disable strict force inventory")
        for force in enabled_forces:
            if not str(force.get("solver_force", "")).strip():
                errors.append("enabled force entries must include solver_force")
            if str(force.get("solver_force", "")).strip().lower() == "drag" and not str(force.get("law", "")).strip():
                errors.append("drag force entries must include law")

        if strict and errors:
            raise ValueError("Invalid COMSOL manifest:\n" + "\n".join(f"- {item}" for item in errors))
        return errors


def configured_run_mode(config: Mapping[str, Any]) -> str:
    raw = config.get("mode", "")
    if raw is None:
        return ""
    mode = str(raw).strip().lower()
    if not mode:
        return ""
    if mode not in SUPPORTED_RUN_MODES:
        expected = ", ".join(sorted(SUPPORTED_RUN_MODES))
        raise ValueError(f"Unsupported run mode {mode!r}; expected {expected}, or omit mode for the default mode")
    return mode


def is_comsol_faithful_config(config: Mapping[str, Any]) -> bool:
    mode = configured_run_mode(config)
    if mode == MODE_COMSOL_FAITHFUL:
        return True
    if mode == MODE_SURFACE_RELEASE_PRODUCTION:
        return False
    comsol = config.get("comsol", {})
    return isinstance(comsol, Mapping) and bool(str(comsol.get("manifest", "")).strip())


__all__ = (
    "ALLOWED_COORDINATE_SYSTEMS",
    "ALLOWED_FIELD_PHYSICAL_QUANTITIES",
    "ComsolCaseManifest",
    "ComsolFieldSpec",
    "MODE_COMSOL_FAITHFUL",
    "MODE_SURFACE_RELEASE_PRODUCTION",
    "SUPPORTED_RUN_MODES",
    "configured_run_mode",
    "is_comsol_faithful_config",
)
