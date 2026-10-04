"""Resolve field bindings for an already-validated semantic force model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from particle_tracer_unified.core.datamodel import TriangleMeshField2D
from particle_tracer_unified.core.field_sampling import (
    choose_electric_field_quantity_names,
    choose_velocity_quantity_names,
)
from particle_tracer_unified.force_models import ForceModel, SemanticForce

_OPTIONAL_FIELDS = {
    "drag": ("rho_g", "mu", "T"),
    "thermophoresis": ("rho_g", "mu"),
    "lift": ("rho_g", "mu"),
    "pressure_gradient": ("rho_g",),
    "virtual_mass": ("rho_g",),
}


@dataclass(frozen=True, slots=True)
class ForceBinding:
    """Provider-specific fields bound to one semantic force definition."""

    force: SemanticForce
    enabled_reason: str
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    field_sources: tuple[tuple[str, str], ...] = ()

    @property
    def name(self) -> str:
        return self.force.name

    @property
    def enabled(self) -> bool:
        return bool(
            self.force.enabled
            and not (self.force.name == "drag" and self.force.model == "none")
        )

    @property
    def model(self) -> str:
        return self.force.model

    @property
    def status(self) -> str:
        return self.force.status


@dataclass(frozen=True, slots=True)
class ForceCatalog:
    model: ForceModel
    bindings: tuple[ForceBinding, ...]

    def by_name(self) -> dict[str, ForceBinding]:
        return {binding.name: binding for binding in self.bindings}

    def enabled(self, name: str) -> bool:
        binding = self.by_name().get(str(name))
        return bool(binding.enabled) if binding is not None else False

    def enabled_names(self) -> tuple[str, ...]:
        """Return semantic forces enabled by the validated model."""

        return tuple(binding.name for binding in self.bindings if binding.enabled)

    def force_model_name(self, name: str, default: str = "") -> str:
        binding = self.by_name().get(str(name))
        return binding.model if binding is not None else str(default)


def _field_quantities(field_provider: object) -> set[str]:
    if field_provider is None:
        return set()
    field = getattr(field_provider, "field", None)
    quantities = getattr(field, "quantities", {})
    return (
        {str(name) for name in quantities} if isinstance(quantities, Mapping) else set()
    )


def _field_quantity_pair(
    field_provider: object,
    x_candidates: tuple[str, ...],
    y_candidates: tuple[str, ...],
) -> tuple[str, ...]:
    quantities = _field_quantities(field_provider)
    x_name = next((name for name in x_candidates if name in quantities), "")
    y_name = next((name for name in y_candidates if name in quantities), "")
    return (x_name, y_name) if x_name and y_name else ()


def _electric_names(field_provider: object, spatial_dim: int) -> tuple[str, ...]:
    field = getattr(field_provider, "field", None)
    return (
        ()
        if field is None
        else tuple(choose_electric_field_quantity_names(field, spatial_dim))
    )


def _velocity_names(field_provider: object, spatial_dim: int) -> tuple[str, ...]:
    field = getattr(field_provider, "field", None)
    return (
        ()
        if field is None
        else tuple(choose_velocity_quantity_names(field, spatial_dim))
    )


def _required_fields(
    field_provider: object, spatial_dim: int
) -> dict[str, tuple[str, ...]]:
    available = _field_quantities(field_provider)
    velocity = _velocity_names(field_provider, spatial_dim)
    electric = _electric_names(field_provider, spatial_dim)
    fluid_acceleration = _field_quantity_pair(
        field_provider,
        ("fluid_accel_x", "fluid_acceleration_x", "material_accel_x", "a_fluid_x"),
        ("fluid_accel_y", "fluid_acceleration_y", "material_accel_y", "a_fluid_y"),
    )
    field = getattr(field_provider, "field", None)
    pressure_fields = (
        fluid_acceleration
        if not velocity and isinstance(field, TriangleMeshField2D)
        else velocity
    )
    return {
        "drag": velocity,
        "electric": electric,
        "gravity": (),
        "thermophoresis": ("T",) if "T" in available else (),
        "dielectrophoresis": electric,
        "lift": velocity,
        "pressure_gradient": pressure_fields,
        "virtual_mass": velocity,
    }


def _bind_force(
    force: SemanticForce,
    required_fields: tuple[str, ...],
    declared: frozenset[str],
) -> ForceBinding:
    active = force.enabled and not (force.name == "drag" and force.model == "none")
    required = required_fields if active else ()
    reason = "explicit_config" if force.name in declared else "default_false"
    if force.name == "drag" and not force.enabled:
        reason = "explicit_none"
    return ForceBinding(
        force=force,
        enabled_reason=reason,
        required_fields=required,
        optional_fields=_OPTIONAL_FIELDS.get(force.name, ()) if active else (),
        field_sources=tuple((name, f"field:{name}") for name in required),
    )


def resolve_force_catalog(
    model: ForceModel,
    *,
    field_provider: object = None,
    spatial_dim: int = 2,
) -> ForceCatalog:
    """Bind provider quantities without changing force laws or coefficients."""

    if not isinstance(model, ForceModel):
        raise TypeError("resolve_force_catalog requires a typed ForceModel")
    required = _required_fields(field_provider, int(spatial_dim))
    bindings = tuple(
        _bind_force(force, required[force.name], model.declared)
        for force in model.definitions()
    )
    return ForceCatalog(model=model, bindings=bindings)


def _field_summary(
    bindings: tuple[ForceBinding, ...], attribute: str, *, sources: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in bindings:
        values = getattr(item, attribute)
        if values:
            result[item.name] = dict(values) if sources else list(values)
    return result


def _binding_summary(bindings: tuple[ForceBinding, ...]) -> dict[str, Any]:
    return {
        "enabled_forces": [item.name for item in bindings if item.enabled],
        "disabled_forces": [item.name for item in bindings if not item.enabled],
        "force_status": {item.name: item.status for item in bindings},
        "force_models": {item.name: item.model for item in bindings},
        "force_enabled_reason": {item.name: item.enabled_reason for item in bindings},
        "force_required_fields": _field_summary(bindings, "required_fields"),
        "force_optional_fields": _field_summary(bindings, "optional_fields"),
        "force_field_sources": _field_summary(bindings, "field_sources", sources=True),
    }


def force_catalog_summary(catalog: ForceCatalog | None) -> dict[str, Any]:
    if catalog is None:
        return {"has_force_catalog": False}
    return {"has_force_catalog": True, **_binding_summary(catalog.bindings)}


__all__ = (
    "ForceBinding",
    "ForceCatalog",
    "force_catalog_summary",
    "resolve_force_catalog",
)
