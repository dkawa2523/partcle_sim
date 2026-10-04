"""Case metadata and runtime input-provider configuration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from ._configuration_core import (
    enum,
    error,
    finite_number,
    integer,
    mapping,
    optional_string,
    parameters,
    reject_unknown,
    required,
)

COORDINATE_SYSTEMS = {
    2: frozenset({"cartesian_xy", "axisymmetric_rz"}),
    3: frozenset({"cartesian_xyz"}),
}
PROVIDER_KINDS = {
    "geometry": frozenset({"box", "precomputed_npz"}),
    "field": frozenset(
        {"linear_shear", "precomputed_npz", "precomputed_triangle_mesh_npz"}
    ),
}
PROVIDER_PARAMETER_KEYS = {
    "box": frozenset({"bounds", "grid_shape", "boundary_part_ids"}),
    "linear_shear": frozenset(
        {"shear_rate", "dynamic_viscosity_Pas", "time_mode", "times"}
    ),
    "precomputed_npz": frozenset(),
    "precomputed_triangle_mesh_npz": frozenset(),
}
PRECOMPUTED_PROVIDER_KINDS = frozenset(
    {"precomputed_npz", "precomputed_triangle_mesh_npz"}
)


def _require_parameter_keys(
    parameters: Mapping[str, Any],
    names: tuple[str, ...],
    path: str,
) -> None:
    for name in names:
        if name not in parameters:
            raise error(path, f"missing required key {name!r}")


def _validate_linear_shear_parameters(
    parameters: Mapping[str, Any],
    path: str,
) -> None:
    _require_parameter_keys(
        parameters,
        ("shear_rate", "dynamic_viscosity_Pas"),
        path,
    )
    finite_number(parameters["shear_rate"], f"{path}.shear_rate")
    finite_number(
        parameters["dynamic_viscosity_Pas"],
        f"{path}.dynamic_viscosity_Pas",
        minimum=0.0,
        exclusive_minimum=True,
    )
    if "time_mode" in parameters:
        enum(parameters["time_mode"], {"steady", "transient"}, f"{path}.time_mode")
    if "times" not in parameters:
        return
    times = parameters["times"]
    if not isinstance(times, (list, tuple)) or not times:
        raise error(f"{path}.times", "must be a non-empty list")
    parsed_times = [
        finite_number(item, f"{path}.times[{index}]")
        for index, item in enumerate(times)
    ]
    if any(right <= left for left, right in pairwise(parsed_times)):
        raise error(f"{path}.times", "must be strictly increasing")


def _validate_provider_sequence(
    parameters: Mapping[str, Any],
    name: str,
    path: str,
) -> None:
    values = parameters[name]
    if not isinstance(values, (list, tuple)) or not values:
        raise error(f"{path}.{name}", "must be a non-empty list")
    for index, item in enumerate(values):
        item_path = f"{path}.{name}[{index}]"
        if name == "bounds":
            finite_number(item, item_path)
        else:
            integer(item, item_path, minimum=2 if name == "grid_shape" else 1)


def _validate_box_parameters(parameters: Mapping[str, Any], path: str) -> None:
    names = ("bounds", "grid_shape", "boundary_part_ids")
    _require_parameter_keys(parameters, names, path)
    for name in names:
        _validate_provider_sequence(parameters, name, path)
    bounds = parameters["bounds"]
    if len(bounds) % 2:
        raise error(f"{path}.bounds", "must contain minimum/maximum pairs")
    if any(
        float(bounds[index + 1]) <= float(bounds[index])
        for index in range(0, len(bounds), 2)
    ):
        raise error(f"{path}.bounds", "each maximum must be greater than its minimum")


def _validate_provider_parameters(
    kind: str,
    parameters: Mapping[str, Any],
    source_path: str | None,
    path: str,
) -> None:
    if kind in PRECOMPUTED_PROVIDER_KINDS and source_path is None:
        raise error(
            f"{path.rsplit('.', 1)[0]}.path", f"is required for provider kind {kind!r}"
        )
    if kind == "linear_shear":
        _validate_linear_shear_parameters(parameters, path)
    elif kind == "box":
        _validate_box_parameters(parameters, path)


@dataclass(frozen=True)
class CaseConfig:
    spatial_dim: int
    coordinate_system: str
    adapter: str

    @classmethod
    def from_mapping(cls, value: Any, path: str = "case") -> CaseConfig:
        data = mapping(value, path)
        reject_unknown(data, {"spatial_dim", "coordinate_system", "adapter"}, path)
        dim = integer(required(data, "spatial_dim", path), f"{path}.spatial_dim")
        if dim not in COORDINATE_SYSTEMS:
            raise error(f"{path}.spatial_dim", "must be 2 or 3")
        coordinate_system = enum(
            required(data, "coordinate_system", path),
            COORDINATE_SYSTEMS[dim],
            f"{path}.coordinate_system",
        )
        adapter = enum(
            required(data, "adapter", path), {"native", "comsol"}, f"{path}.adapter"
        )
        return cls(
            spatial_dim=dim, coordinate_system=coordinate_system, adapter=adapter
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "spatial_dim": int(self.spatial_dim),
            "coordinate_system": self.coordinate_system,
            "adapter": self.adapter,
        }


@dataclass(frozen=True)
class ProviderConfig:
    kind: str
    path: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any, path: str, *, role: str) -> ProviderConfig:
        data = mapping(value, path)
        reject_unknown(data, {"kind", "path", "parameters"}, path)
        kind = enum(
            required(data, "kind", path),
            PROVIDER_KINDS[role],
            f"{path}.kind",
        )
        source_path = optional_string(data.get("path"), f"{path}.path")
        parameter_path = f"{path}.parameters"
        provider_parameters = parameters(data.get("parameters", {}), parameter_path)
        reject_unknown(
            provider_parameters,
            PROVIDER_PARAMETER_KEYS[kind],
            parameter_path,
        )
        _validate_provider_parameters(
            kind, provider_parameters, source_path, parameter_path
        )
        return cls(kind=kind, path=source_path, parameters=provider_parameters)

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.path is not None:
            result["path"] = self.path
        if self.parameters:
            result["parameters"] = deepcopy(dict(self.parameters))
        return result


def _optional_provider(
    value: Any,
    path: str,
    *,
    role: str,
) -> ProviderConfig | None:
    if value is None:
        return None
    return ProviderConfig.from_mapping(value, path, role=role)


def _validate_box_provider_dimensions(
    geometry: ProviderConfig | None,
    spatial_dim: int,
    path: str,
) -> None:
    if geometry is None or geometry.kind != "box":
        return
    expected_lengths = {
        "bounds": 2 * spatial_dim,
        "grid_shape": spatial_dim,
        "boundary_part_ids": 4 if spatial_dim == 2 else 12,
    }
    for name, expected in expected_lengths.items():
        values = geometry.parameters.get(name)
        if values is not None and len(values) != expected:
            raise error(
                f"{path}.geometry.parameters.{name}",
                f"must contain exactly {expected} values",
            )


def _input_entries(
    particles: str | None,
    boundaries: str | None,
    geometry: ProviderConfig | None,
    field_config: ProviderConfig | None,
) -> tuple[tuple[str, object | None], ...]:
    return (
        ("particles", particles),
        ("boundaries", boundaries),
        ("geometry", geometry),
        ("field", field_config),
    )


def _validate_native_inputs(
    entries: tuple[tuple[str, object | None], ...],
    manifest: str | None,
    path: str,
) -> None:
    missing = [name for name, item in entries if item is None]
    if missing:
        raise error(path, f"native adapter requires {', '.join(missing)}")
    if manifest is not None:
        raise error(
            f"{path}.comsol_manifest",
            "is only valid for the comsol adapter",
        )


def _validate_comsol_inputs(
    entries: tuple[tuple[str, object | None], ...],
    manifest: str | None,
    path: str,
) -> None:
    if manifest is None:
        raise error(path, "comsol adapter requires comsol_manifest")
    duplicated = [name for name, item in entries if item is not None]
    if duplicated:
        raise error(
            path,
            "COMSOL artifacts must be declared only by the manifest; remove "
            + ", ".join(duplicated),
        )


@dataclass(frozen=True)
class InputsConfig:
    particles: str | None
    boundaries: str | None
    geometry: ProviderConfig | None
    field: ProviderConfig | None
    comsol_manifest: str | None

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        adapter: str,
        spatial_dim: int,
        path: str = "inputs",
    ) -> InputsConfig:
        data = mapping(value, path)
        reject_unknown(
            data,
            {"particles", "boundaries", "geometry", "field", "comsol_manifest"},
            path,
        )
        particles = optional_string(data.get("particles"), f"{path}.particles")
        boundaries = optional_string(data.get("boundaries"), f"{path}.boundaries")
        geometry = _optional_provider(
            data.get("geometry"),
            f"{path}.geometry",
            role="geometry",
        )
        field_config = _optional_provider(
            data.get("field"),
            f"{path}.field",
            role="field",
        )
        _validate_box_provider_dimensions(geometry, int(spatial_dim), path)
        manifest = optional_string(
            data.get("comsol_manifest"),
            f"{path}.comsol_manifest",
        )
        entries = _input_entries(particles, boundaries, geometry, field_config)
        if adapter == "native":
            _validate_native_inputs(entries, manifest, path)
        else:
            _validate_comsol_inputs(entries, manifest, path)
        return cls(
            particles=particles,
            boundaries=boundaries,
            geometry=geometry,
            field=field_config,
            comsol_manifest=manifest,
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.particles is not None:
            result["particles"] = self.particles
        if self.boundaries is not None:
            result["boundaries"] = self.boundaries
        if self.geometry is not None:
            result["geometry"] = self.geometry.to_mapping()
        if self.field is not None:
            result["field"] = self.field.to_mapping()
        if self.comsol_manifest is not None:
            result["comsol_manifest"] = self.comsol_manifest
        return result
