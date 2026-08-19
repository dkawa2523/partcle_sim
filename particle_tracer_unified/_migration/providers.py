"""Legacy geometry and field provider configuration conversion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .legacy import (
    _canonical_choice,
    _canonical_keys,
    _merge_without_conflicts,
    _relocated_reference,
)

_PROVIDER_KIND_ALIASES = {
    "geometry": {
        "synthetic": "box",
        "synthetic_box": "box",
        "box_geometry": "box",
        "npz": "precomputed_npz",
        "precomputed": "precomputed_npz",
        "regular_grid_npz": "precomputed_npz",
    },
    "field": {
        "synthetic": "linear_shear",
        "shear": "linear_shear",
        "linear_shear_field": "linear_shear",
        "npz": "precomputed_npz",
        "precomputed": "precomputed_npz",
        "regular_grid": "precomputed_npz",
        "regular_grid_npz": "precomputed_npz",
        "triangle_mesh": "precomputed_triangle_mesh_npz",
        "triangle_mesh_npz": "precomputed_triangle_mesh_npz",
        "precomputed_triangle_mesh": "precomputed_triangle_mesh_npz",
    },
}


def _provider(
    value: Any,
    *,
    source_base: Path,
    destination_base: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"legacy providers.{label} must be a mapping")
    raw = dict(value)
    if not raw:
        raise ValueError(f"legacy config is missing providers.{label}")
    parameter_keys = {
        "geometry": ("bounds", "grid_shape", "boundary_part_ids"),
        "field": ("shear_rate", "dynamic_viscosity_Pas", "time_mode", "times"),
    }[label]
    aliases = {
        "provider_kind": "kind",
        "type": "kind",
        "npz_path": "path",
        "file_path": "path",
        "mu_pas": "dynamic_viscosity_Pas",
        "viscosity_pas": "dynamic_viscosity_Pas",
        "dynamic_viscosity": "dynamic_viscosity_Pas",
    }
    normalized = _canonical_keys(
        raw,
        canonical=("kind", "path", "parameters", *parameter_keys),
        aliases=aliases,
        label=f"providers.{label}",
    )
    if "kind" not in normalized or not str(normalized["kind"]).strip():
        raise ValueError(f"legacy providers.{label}.kind is required")
    kinds = {
        "geometry": ("box", "precomputed_npz"),
        "field": ("linear_shear", "precomputed_npz", "precomputed_triangle_mesh_npz"),
    }[label]
    kind = _canonical_choice(
        normalized.pop("kind"),
        canonical=kinds,
        aliases=_PROVIDER_KIND_ALIASES[label],
        label=f"providers.{label}.kind",
    )
    path_value = normalized.pop("path", None)
    nested = normalized.pop("parameters", {})
    if not isinstance(nested, Mapping):
        raise ValueError(f"legacy providers.{label}.parameters must be a mapping")
    parameters = _canonical_keys(
        dict(nested),
        canonical=parameter_keys,
        aliases=aliases,
        label=f"providers.{label}.parameters",
    )
    _merge_without_conflicts(
        parameters,
        normalized,
        label=f"providers.{label} parameters",
    )
    valid_for_kind = {
        "box": ("bounds", "grid_shape", "boundary_part_ids"),
        "linear_shear": ("shear_rate", "dynamic_viscosity_Pas", "time_mode", "times"),
        "precomputed_npz": (),
        "precomputed_triangle_mesh_npz": (),
    }[kind]
    parameters = _canonical_keys(
        parameters,
        canonical=valid_for_kind,
        aliases={},
        label=f"providers.{label} parameters for {kind}",
    )
    if "time_mode" in parameters:
        parameters["time_mode"] = _canonical_choice(
            parameters["time_mode"],
            canonical=("steady", "transient"),
            aliases={"time_independent": "steady", "time_dependent": "transient"},
            label=f"providers.{label}.parameters.time_mode",
        )
    result: dict[str, Any] = {"kind": kind}
    if path_value is not None:
        if not str(path_value).strip():
            raise ValueError(f"legacy providers.{label}.path must not be blank")
        result["path"] = _relocated_reference(source_base, destination_base, path_value)
    if parameters:
        result["parameters"] = parameters
    return result
