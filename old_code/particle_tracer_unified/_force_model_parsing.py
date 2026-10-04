"""Decode native YAML and COMSOL manifest force declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import _force_model_values as _values
from ._force_model_types import (
    DRAG_MODELS,
    FORCE_NAMES,
    OPTIONAL_FORCE_NAMES,
    DragForce,
    ElectricForce,
    ForceModel,
    SemanticForce,
)


def parse_native_force_model(
    drag: Any,
    forces: Any,
    *,
    spatial_dim: int,
    path: str = "physics",
) -> ForceModel:
    raw_forces = _values._mapping(forces, f"{path}.forces")
    unknown = sorted(name for name in raw_forces if name not in OPTIONAL_FORCE_NAMES)
    if unknown:
        raise _values._error(
            f"{path}.forces", f"unknown force(s): {', '.join(unknown)}"
        )
    values: dict[str, Any] = {
        name: _values._parse_optional_force(
            name,
            value,
            path=f"{path}.forces.{name}",
            spatial_dim=spatial_dim,
        )
        for name, value in raw_forces.items()
    }
    return ForceModel(
        drag=_values.parse_drag_force(drag, path=f"{path}.drag"),
        **values,
        declared=frozenset({"drag", *raw_forces}),
    )


def _manifest_entry_header(
    raw: Any,
    context: str,
) -> tuple[dict[str, Any], str, bool]:
    data = _values._mapping(raw, context)
    name_raw = _values._required(data, "solver_force", context)
    if not isinstance(name_raw, str) or name_raw != name_raw.strip():
        raise _values._error(f"{context}.solver_force", "must be an exact string")
    name = name_raw
    if name not in FORCE_NAMES:
        raise _values._error(
            f"{context}.solver_force",
            f"must be one of {list(FORCE_NAMES)}, got {name!r}",
        )
    if "enabled" not in data:
        raise _values._error(f"{context}.enabled", "is required")
    enabled = _values._boolean(data["enabled"], f"{context}.enabled")
    return data, name, enabled


def _parse_manifest_drag(
    data: Mapping[str, Any],
    enabled: bool,
    context: str,
) -> DragForce:
    _values._reject_unknown(data, {"solver_force", "enabled", "law"}, context)
    law_raw = data.get("law")
    if enabled and law_raw is None:
        raise _values._error(f"{context}.law", "is required when drag is enabled")
    law = (
        "none"
        if law_raw is None
        else _values._model(law_raw, DRAG_MODELS, f"{context}.law")
    )
    return DragForce(enabled=enabled, model=law)


def _parse_manifest_optional_force(
    name: str,
    data: Mapping[str, Any],
    enabled: bool,
    *,
    context: str,
    spatial_dim: int,
) -> SemanticForce:
    if "law" in data:
        raise _values._error(f"{context}.law", "is valid only for drag")
    if name == "electric":
        _values._reject_unknown(data, {"solver_force", "enabled"}, context)
        return ElectricForce(enabled=enabled)

    _values._reject_unknown(
        data,
        {"solver_force", "enabled", "model", "parameters"},
        context,
    )
    model, parameters = _values._optional_model_and_parameters(name, data, context)
    return _values._build_optional_force(
        name,
        enabled,
        model,
        parameters,
        path=context,
        spatial_dim=spatial_dim,
    )


def _parse_manifest_entry(
    raw: Any,
    *,
    context: str,
    spatial_dim: int,
) -> tuple[str, SemanticForce]:
    data, name, enabled = _manifest_entry_header(raw, context)
    if name == "drag":
        return name, _parse_manifest_drag(data, enabled, context)
    return name, _parse_manifest_optional_force(
        name,
        data,
        enabled,
        context=context,
        spatial_dim=spatial_dim,
    )


def parse_manifest_force_model(
    entries: Any,
    *,
    spatial_dim: int,
    path: str = "forces",
) -> ForceModel:
    if not isinstance(entries, (list, tuple)):
        raise _values._error(path, "must be a list")
    parsed: dict[str, Any] = {}
    for index, raw in enumerate(entries):
        name, force = _parse_manifest_entry(
            raw,
            context=f"{path}[{index}]",
            spatial_dim=spatial_dim,
        )
        if name in parsed:
            raise _values._error(path, f"contains duplicate solver_force {name!r}")
        parsed[name] = force

    declared = frozenset(parsed)
    if "drag" not in parsed:
        # Keep a complete model so manifest validation can report missing drag
        # together with independent contract errors.
        parsed["drag"] = DragForce()
    return ForceModel(**parsed, declared=declared)
