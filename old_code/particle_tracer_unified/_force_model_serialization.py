"""Serialize semantic force values at native and manifest boundaries."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

from ._force_model_types import (
    FORCE_NAMES,
    OPTIONAL_FORCE_NAMES,
    ForceModel,
    SemanticForce,
)


def force_parameter_mapping(force: SemanticForce) -> dict[str, Any]:
    """Serialize typed coefficients without making the mapping authoritative."""

    result: dict[str, Any] = {}
    for item in fields(force):
        if item.name in {"enabled", "model"}:
            continue
        value = getattr(force, item.name)
        if isinstance(value, tuple):
            if value:
                result[item.name] = list(value)
        elif value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        else:
            result[item.name] = value
    return result


def force_model_to_native_mapping(
    model: ForceModel,
) -> tuple[dict[str, str], dict[str, Any]]:
    forces: dict[str, Any] = {}
    for name in OPTIONAL_FORCE_NAMES:
        if name not in model.declared:
            continue
        force = model.definition(name)
        payload: dict[str, Any] = {
            "enabled": bool(force.enabled),
            "model": force.model,
        }
        parameters = force_parameter_mapping(force)
        if parameters:
            payload["parameters"] = parameters
        forces[name] = payload
    return {"model": model.drag.model}, forces


def force_model_to_manifest_inventory(model: ForceModel) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    for name in FORCE_NAMES:
        if name not in model.declared:
            continue
        force = model.definition(name)
        if name == "drag":
            entry: dict[str, Any] = {
                "solver_force": "drag",
                "enabled": bool(force.enabled),
            }
            if force.enabled:
                entry["law"] = force.model
            entries.append(entry)
            continue
        if name == "electric":
            entries.append({"solver_force": name, "enabled": bool(force.enabled)})
            continue
        entry = {
            "solver_force": name,
            "enabled": bool(force.enabled),
            "model": force.model,
        }
        parameters = force_parameter_mapping(force)
        if parameters:
            entry["parameters"] = parameters
        entries.append(entry)
    return tuple(entries)
