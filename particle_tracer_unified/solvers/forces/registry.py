from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...core.field_sampling import (
    choose_electric_field_quantity_names,
    choose_velocity_quantity_names,
)


IMPLEMENTED_FORCES = {
    "drag",
    "electric",
    "gravity",
    "brownian",
    "thermophoresis",
    "dielectrophoresis",
    "lift",
}
SUPPORTED_FORCE_NAMES = (
    "drag",
    "electric",
    "gravity",
    "brownian",
    "thermophoresis",
    "dielectrophoresis",
    "lift",
)


@dataclass(frozen=True)
class ForceSpec:
    name: str
    enabled: bool
    model: str
    status: str
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    field_sources: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForceCatalog:
    specs: tuple[ForceSpec, ...]

    def by_name(self) -> dict[str, ForceSpec]:
        return {spec.name: spec for spec in self.specs}

    def enabled(self, name: str) -> bool:
        spec = self.by_name().get(str(name))
        return bool(spec.enabled) if spec is not None else False

    def model(self, name: str, default: str = "") -> str:
        spec = self.by_name().get(str(name))
        return str(spec.model) if spec is not None and spec.model else str(default)


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _force_cfg(solver_cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    forces = _as_mapping(solver_cfg.get("forces", {}))
    cfg = forces.get(name, {})
    if cfg is None:
        return {}
    if isinstance(cfg, (bool, int, str)):
        return {"enabled": cfg}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"solver.forces.{name} must be a mapping or boolean")
    return cfg


def _validate_known_force_names(solver_cfg: Mapping[str, Any]) -> None:
    forces = _as_mapping(solver_cfg.get("forces", {}))
    unknown = sorted(str(name) for name in forces.keys() if str(name) not in SUPPORTED_FORCE_NAMES)
    if unknown:
        raise ValueError(f"unknown solver.forces entries: {', '.join(unknown)}")


def _field_quantities(field_provider: object) -> set[str]:
    if field_provider is None:
        return set()
    field = getattr(field_provider, "field", None)
    quantities = getattr(field, "quantities", {})
    return {str(name) for name in quantities.keys()} if isinstance(quantities, Mapping) else set()


def _field_source_map(names: tuple[str, ...]) -> dict[str, str]:
    return {name: f"field:{name}" for name in names}


def _electric_names(field_provider: object, spatial_dim: int) -> tuple[str, ...]:
    if field_provider is None:
        return ()
    field = getattr(field_provider, "field", None)
    if field is None:
        return ()
    return tuple(choose_electric_field_quantity_names(field, int(spatial_dim)))


def _velocity_names(field_provider: object, spatial_dim: int) -> tuple[str, ...]:
    if field_provider is None:
        return ()
    field = getattr(field_provider, "field", None)
    if field is None:
        return ()
    return tuple(choose_velocity_quantity_names(field, int(spatial_dim)))


def _enabled_from_cfg(cfg: Mapping[str, Any], *, default: bool) -> bool:
    return _bool(cfg.get("enabled", default), default=default)


def _optional_spec(
    name: str,
    cfg: Mapping[str, Any],
    *,
    model_default: str,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...] = (),
) -> ForceSpec:
    enabled = _enabled_from_cfg(cfg, default=False)
    return ForceSpec(
        name=name,
        enabled=enabled,
        model=str(cfg.get("model", model_default)).strip() or str(model_default),
        status="implemented",
        required_fields=required_fields if enabled else (),
        optional_fields=optional_fields,
        field_sources=_field_source_map(required_fields if enabled else ()),
        config=dict(cfg),
    )


def build_force_catalog(
    config_payload: Mapping[str, Any],
    *,
    field_provider: object = None,
    spatial_dim: int = 2,
) -> ForceCatalog:
    config = config_payload if isinstance(config_payload, Mapping) else {}
    solver_cfg = _as_mapping(config.get("solver", {}))
    _validate_known_force_names(solver_cfg)
    available_quantities = _field_quantities(field_provider)

    drag_cfg = _force_cfg(solver_cfg, "drag")
    drag_enabled = _enabled_from_cfg(drag_cfg, default=True)
    if not drag_enabled:
        raise ValueError("solver.forces.drag.enabled=false is not supported by the current drag-relaxation solver")
    drag_model = str(drag_cfg.get("model", solver_cfg.get("drag_model", "stokes"))).strip().lower()
    velocity_names = _velocity_names(field_provider, spatial_dim)

    electric_cfg = _force_cfg(solver_cfg, "electric")
    electric_names = _electric_names(field_provider, spatial_dim)
    electric_default = bool(electric_names)
    electric_enabled = _enabled_from_cfg(electric_cfg, default=electric_default)
    if electric_enabled and not electric_names:
        raise ValueError("solver.forces.electric is enabled but no electric field quantity was found")

    gravity_cfg = _force_cfg(solver_cfg, "gravity")
    has_legacy_gravity = "gravity_mps2" in solver_cfg or "body_acceleration" in solver_cfg
    gravity_enabled = _enabled_from_cfg(gravity_cfg, default=bool(has_legacy_gravity))

    brownian_cfg = _force_cfg(solver_cfg, "brownian")
    stochastic_cfg = solver_cfg.get("stochastic_motion", {})
    brownian_default = _bool(_as_mapping(stochastic_cfg).get("enabled", False), default=False) if isinstance(stochastic_cfg, Mapping) else _bool(stochastic_cfg, default=False)
    brownian_enabled = _enabled_from_cfg(brownian_cfg, default=brownian_default)

    specs = [
        ForceSpec(
            name="drag",
            enabled=True,
            model=drag_model,
            status="implemented",
            required_fields=velocity_names,
            optional_fields=("rho_g", "mu", "T"),
            field_sources=_field_source_map(velocity_names),
            config=dict(drag_cfg),
        ),
        ForceSpec(
            name="electric",
            enabled=bool(electric_enabled),
            model=str(electric_cfg.get("model", "particle_charge")).strip() or "particle_charge",
            status="implemented",
            required_fields=electric_names if electric_enabled else (),
            field_sources=_field_source_map(electric_names if electric_enabled else ()),
            config=dict(electric_cfg),
        ),
        ForceSpec(
            name="gravity",
            enabled=bool(gravity_enabled),
            model="constant_acceleration",
            status="implemented",
            field_sources={"gravity": "constant_config"} if gravity_enabled else {},
            config=dict(gravity_cfg),
        ),
        ForceSpec(
            name="brownian",
            enabled=bool(brownian_enabled),
            model=str(brownian_cfg.get("model", "underdamped_langevin")).strip() or "underdamped_langevin",
            status="implemented",
            optional_fields=("T",),
            field_sources={"T": "field:T_then_gas"} if brownian_enabled and "T" in _field_quantities(field_provider) else {},
            config=dict(brownian_cfg),
        ),
        _optional_spec(
            "thermophoresis",
            _force_cfg(solver_cfg, "thermophoresis"),
            model_default="talbot",
            required_fields=("T",) if "T" in available_quantities else (),
            optional_fields=("rho_g", "mu"),
        ),
        _optional_spec(
            "dielectrophoresis",
            _force_cfg(solver_cfg, "dielectrophoresis"),
            model_default="dc",
            required_fields=electric_names,
        ),
        _optional_spec(
            "lift",
            _force_cfg(solver_cfg, "lift"),
            model_default="saffman",
            required_fields=velocity_names,
            optional_fields=("rho_g", "mu"),
        ),
    ]
    for spec in specs:
        if bool(spec.enabled) and spec.name in {"thermophoresis", "dielectrophoresis", "lift"} and not spec.required_fields:
            raise ValueError(f"solver.forces.{spec.name} is enabled but required field quantities were not found")
    return ForceCatalog(specs=tuple(specs))


def solver_cfg_with_force_overrides(solver_cfg: Mapping[str, Any], catalog: ForceCatalog | None) -> dict[str, Any]:
    out = dict(solver_cfg)
    if catalog is None:
        return out
    by_name = catalog.by_name()
    drag = by_name.get("drag")
    if drag is not None and drag.model:
        out["drag_model"] = drag.model
    gravity = by_name.get("gravity")
    if gravity is not None and not gravity.enabled:
        out.pop("gravity_mps2", None)
        out["body_acceleration"] = []
    brownian = by_name.get("brownian")
    if brownian is not None and brownian.config:
        cfg = dict(brownian.config)
        cfg["enabled"] = bool(brownian.enabled)
        out["stochastic_motion"] = cfg
    return out


def force_catalog_summary(catalog: ForceCatalog | None) -> dict[str, Any]:
    if catalog is None:
        return {"has_force_catalog": False}
    specs = catalog.specs
    return {
        "has_force_catalog": True,
        "enabled_forces": [spec.name for spec in specs if spec.enabled],
        "disabled_forces": [spec.name for spec in specs if not spec.enabled],
        "force_status": {spec.name: spec.status for spec in specs},
        "force_models": {spec.name: spec.model for spec in specs if spec.model},
        "force_required_fields": {spec.name: list(spec.required_fields) for spec in specs if spec.required_fields},
        "force_optional_fields": {spec.name: list(spec.optional_fields) for spec in specs if spec.optional_fields},
        "force_field_sources": {spec.name: dict(spec.field_sources) for spec in specs if spec.field_sources},
    }
