"""Legacy drag and optional-force configuration conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from particle_tracer_unified.force_models import (
    DRAG_MODELS,
    FORCE_MODELS,
    FORCE_PARAMETER_KEYS,
    OPTIONAL_FORCE_NAMES,
)

from .legacy import (
    _canonical_choice,
    _canonical_keys,
    _legacy_bool,
    _merge_without_conflicts,
)

_DRAG_MODEL_ALIASES = {
    "cunningham": "stokes_cunningham",
    "cunningham_stokes": "stokes_cunningham",
    "slip_corrected_stokes": "stokes_cunningham",
    "schiller": "schiller_naumann",
    "epstein_drag": "epstein",
    "free_molecular": "epstein",
    "ballistic": "none",
    "disabled": "none",
}

_FORCE_NAME_ALIASES = {
    "electrostatic": "electric",
    "electric_force": "electric",
    "gravity_buoyancy": "gravity",
    "body_force": "gravity",
    "thermophoretic": "thermophoresis",
    "dep": "dielectrophoresis",
    "saffman_lift": "lift",
    "fluid_pressure_gradient": "pressure_gradient",
    "added_mass": "virtual_mass",
    "brownian_motion": "brownian",
}

_FORCE_MODEL_ALIASES = {
    "electric": {
        "qe": "particle_charge",
        "q_e": "particle_charge",
        "electrostatic": "particle_charge",
    },
    "gravity": {
        "constant": "constant_acceleration",
        "body_acceleration": "constant_acceleration",
    },
    "thermophoresis": {
        "talbot_model": "talbot",
        "continuum_limit": "continuum",
    },
    "dielectrophoresis": {
        "ac": "ac_clausius_mossotti",
        "ac_cm": "ac_clausius_mossotti",
        "clausius_mossotti": "ac_clausius_mossotti",
    },
    "lift": {"saffman_lift": "saffman"},
    "pressure_gradient": {
        "fluid_acceleration": "fluid_material_acceleration",
        "material_acceleration": "fluid_material_acceleration",
    },
    "virtual_mass": {
        "added_mass": "particle_material_acceleration",
        "virtual_mass": "particle_material_acceleration",
    },
}

_FORCE_PARAMETER_ALIASES = {
    "gravity": {
        "acceleration": "acceleration_mps2",
        "body_acceleration": "acceleration_mps2",
        "body_acceleration_mps2": "acceleration_mps2",
        "gravity_mps2": "acceleration_mps2",
        "buoyancy_enabled": "buoyancy",
    },
}


def _normalized_force_entries(solver: Mapping[str, Any]) -> dict[str, Any]:
    raw = solver.get("forces", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("legacy solver.forces must be a mapping")
    canonical_names = (*sorted(OPTIONAL_FORCE_NAMES), "drag", "brownian")
    result: dict[str, Any] = {}
    for raw_name, value in raw.items():
        name = _canonical_choice(
            raw_name,
            canonical=canonical_names,
            aliases=_FORCE_NAME_ALIASES,
            label="force name",
        )
        if name in result:
            raise ValueError(
                f"legacy solver.forces supplies {name!r} more than once through aliases"
            )
        result[name] = value
    return result


def _legacy_force_drag(value: Any) -> tuple[bool, str | None]:
    enabled = True
    model: str | None = None
    if isinstance(value, Mapping):
        drag_cfg = _canonical_keys(
            dict(value),
            canonical=("enabled", "model"),
            aliases={"active": "enabled", "drag_law": "model"},
            label="solver.forces.drag",
        )
        enabled = _legacy_bool(drag_cfg.get("enabled", True), default=True)
        if drag_cfg.get("model") is not None:
            model = _canonical_choice(
                drag_cfg["model"],
                canonical=tuple(DRAG_MODELS),
                aliases=_DRAG_MODEL_ALIASES,
                label="drag model",
            )
    elif value is not None:
        enabled = _legacy_bool(value, default=True)
    return enabled, model


def _legacy_drag_model(solver: Mapping[str, Any], warnings: list[str]) -> str:
    forces = _normalized_force_entries(solver)
    enabled, force_model = _legacy_force_drag(forces.get("drag"))

    solver_model: str | None = None
    if solver.get("drag_model") is not None:
        solver_model = _canonical_choice(
            solver["drag_model"],
            canonical=tuple(DRAG_MODELS),
            aliases=_DRAG_MODEL_ALIASES,
            label="solver.drag_model",
        )
    if (
        force_model is not None
        and solver_model is not None
        and force_model != solver_model
    ):
        raise ValueError(
            "legacy solver.drag_model and solver.forces.drag.model disagree"
        )
    model = force_model or solver_model
    if model is None:
        model = "stokes"
        warnings.append(
            "solver.drag_model was absent; materialized the legacy stokes default"
        )
    return model if enabled else "none"


def _migrate_forces(solver: Mapping[str, Any]) -> dict[str, Any]:
    raw_forces = _normalized_force_entries(solver)
    migrated: dict[str, Any] = {}
    for name, raw in raw_forces.items():
        if name in {"drag", "brownian"}:
            continue
        parameter_aliases = _FORCE_PARAMETER_ALIASES.get(name, {})
        cfg = (
            _canonical_keys(
                dict(raw),
                canonical=(
                    "enabled",
                    "model",
                    "parameters",
                    *FORCE_PARAMETER_KEYS[name],
                ),
                aliases={"active": "enabled", **parameter_aliases},
                label=f"solver.forces.{name}",
            )
            if isinstance(raw, Mapping)
            else {"enabled": raw}
        )
        nested = cfg.pop("parameters", {})
        if not isinstance(nested, Mapping):
            raise ValueError(
                f"legacy solver.forces.{name}.parameters must be a mapping"
            )
        parameters = _canonical_keys(
            dict(nested),
            canonical=tuple(FORCE_PARAMETER_KEYS[name]),
            aliases=parameter_aliases,
            label=f"solver.forces.{name}.parameters",
        )
        direct_parameters = {
            key: cfg.pop(key) for key in tuple(cfg) if key in FORCE_PARAMETER_KEYS[name]
        }
        _merge_without_conflicts(
            parameters,
            direct_parameters,
            label=f"{name} force parameters",
        )
        if "buoyancy" in parameters:
            parameters["buoyancy"] = _legacy_bool(parameters["buoyancy"])
        item: dict[str, Any] = {
            "enabled": _legacy_bool(cfg.get("enabled", True), default=True)
        }
        if cfg.get("model") is not None:
            item["model"] = _canonical_choice(
                cfg["model"],
                canonical=tuple(FORCE_MODELS[name]),
                aliases=_FORCE_MODEL_ALIASES.get(name, {}),
                label=f"{name} force model",
            )
        if parameters:
            item["parameters"] = parameters
        migrated[name] = item
    return migrated
