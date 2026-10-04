from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .core.field_sampling import (
    choose_electric_field_quantity_names,
    choose_velocity_quantity_names,
)
from .preflight_types import ValidationIssue
from .solvers.drag_models import drag_model_gas_requirements

_GAS_FIELD_ALIASES = {
    "temperature_K": frozenset(("T", "temperature", "gas_temperature")),
    "dynamic_viscosity_Pas": frozenset(("mu", "dynamic_viscosity", "gas_mu")),
    "density_kgm3": frozenset(("rho_g", "rho", "gas_density")),
    "molecular_mass_amu": frozenset(),
}

_COMSOL_SPHERE_RELATIVE_TOLERANCE = 1.0e-3


def _force_enabled(by_name: Mapping[str, Any], name: str) -> bool:
    spec = by_name.get(name)
    return spec is not None and bool(getattr(spec, "enabled", False))


def _positive_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0.0)


def _gas_quantity_available(
    configured_gas: Any,
    runtime_gas: Any,
    field_quantities: set[str],
    name: str,
) -> bool:
    runtime_name = "temperature" if name == "temperature_K" else name
    values = (
        getattr(configured_gas, name, None),
        getattr(runtime_gas, runtime_name, None),
    )
    return any(_positive_number(value) for value in values) or bool(
        _GAS_FIELD_ALIASES[name].intersection(field_quantities)
    )


def _fluid_acceleration_fields(
    quantities: set[str],
    spatial_dim: int,
) -> tuple[str, ...]:
    aliases_by_axis = (
        ("fluid_accel_x", "fluid_acceleration_x", "material_accel_x", "a_fluid_x"),
        ("fluid_accel_y", "fluid_acceleration_y", "material_accel_y", "a_fluid_y"),
        ("fluid_accel_z", "fluid_acceleration_z", "material_accel_z", "a_fluid_z"),
    )
    return tuple(
        next((name for name in aliases if name in quantities), "")
        for aliases in aliases_by_axis[:spatial_dim]
    )


def _force_field_issues(
    by_name: Mapping[str, Any],
    velocity_fields: tuple[str, ...],
    electric_fields: tuple[str, ...],
    temperature_field: str,
    fluid_acceleration_fields: tuple[str, ...],
) -> list[ValidationIssue]:
    sources = {
        "drag": (velocity_fields, "flow_velocity"),
        "electric": (electric_fields, "electric_field"),
        "thermophoresis": (
            (temperature_field,) if temperature_field else (),
            "temperature_field",
        ),
        "dielectrophoresis": (electric_fields, "electric_field"),
        "lift": (velocity_fields, "flow_velocity"),
        "virtual_mass": (velocity_fields, "flow_velocity"),
    }
    issues = [
        ValidationIssue(
            "physics.force.field.missing",
            f"{feature} requires a complete {semantic} quantity in the field provider",
            context={"feature": feature, "missing": [semantic]},
        )
        for feature, (resolved, semantic) in sources.items()
        if _force_enabled(by_name, feature) and not resolved
    ]
    if (
        _force_enabled(by_name, "pressure_gradient")
        and not velocity_fields
        and not all(fluid_acceleration_fields)
    ):
        issues.append(
            ValidationIssue(
                "physics.force.field.missing",
                "pressure_gradient requires flow velocity or fluid material "
                "acceleration quantities",
                context={
                    "feature": "pressure_gradient",
                    "missing": ["flow_velocity|fluid_material_acceleration"],
                },
            )
        )
    return issues


def _enabled_gas_requirements(
    by_name: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    requirements: dict[str, tuple[str, ...]] = {}
    drag = by_name.get("drag")
    if _force_enabled(by_name, "drag"):
        drag_model = str(getattr(drag, "model", ""))
        requirements[f"drag:{drag_model}"] = drag_model_gas_requirements(drag_model)
    thermophoresis = by_name.get("thermophoresis")
    thermophoresis_model = str(getattr(thermophoresis, "model", ""))
    force_requirements = {
        "thermophoresis": (
            "temperature_K",
            "dynamic_viscosity_Pas",
            "density_kgm3",
            *(() if thermophoresis_model == "continuum" else ("molecular_mass_amu",)),
        ),
        "lift": ("dynamic_viscosity_Pas", "density_kgm3"),
        "pressure_gradient": ("density_kgm3",),
        "virtual_mass": ("density_kgm3",),
    }
    requirements.update(
        (name, required)
        for name, required in force_requirements.items()
        if _force_enabled(by_name, name)
    )
    return requirements


def _gas_requirements(
    physics: Any,
    by_name: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], set[str], ValidationIssue | None]:
    requirements = _enabled_gas_requirements(by_name)
    density_features = {
        name
        for name in ("pressure_gradient", "virtual_mass")
        if _force_enabled(by_name, name)
    }
    gravity = by_name.get("gravity")
    gravity_force = getattr(gravity, "force", None)
    if _force_enabled(by_name, "gravity") and bool(
        getattr(gravity_force, "buoyancy", False)
    ):
        requirements["gravity_buoyancy"] = ("density_kgm3",)
        density_features.add("gravity_buoyancy")
    stochastic = getattr(physics, "stochastic", None)
    if stochastic is not None and bool(getattr(stochastic, "enabled", False)):
        if not _force_enabled(by_name, "drag"):
            issue = ValidationIssue(
                "physics.stochastic.drag",
                "Brownian motion requires an enabled dissipative drag model",
            )
            return requirements, density_features, issue
        requirements["brownian_motion"] = ("temperature_K",)
    return requirements, density_features, None


def _particle_density_issues(runtime: Any, features: set[str]) -> list[ValidationIssue]:
    if not features:
        return []
    density = np.asarray(runtime.particles.density, dtype=np.float64)
    invalid_rows = np.flatnonzero(~np.isfinite(density) | (density <= 0.0))
    if not invalid_rows.size:
        return []
    return [
        ValidationIssue(
            "physics.particle_density.missing",
            "enabled displaced-fluid forces require explicit positive particle "
            "density_kgm3",
            context={
                "features": sorted(features),
                "particle_ids": [
                    int(runtime.particles.particle_id[index])
                    for index in invalid_rows[:12]
                ],
                "invalid_count": int(invalid_rows.size),
            },
        )
    ]


def _comsol_sphere_consistency_issues(
    case: Any,
    runtime: Any,
) -> list[ValidationIssue]:
    """Check the three exported properties that define a COMSOL sphere.

    Native cases may intentionally use an aerodynamic drag diameter, so this
    relation is not a global particle-table contract. COMSOL adapter rows that
    explicitly provide material density must describe the same sphere as their
    inertial mass and diameter.
    """

    case_config = getattr(getattr(case, "config", None), "case", None)
    if str(getattr(case_config, "adapter", "")).strip().lower() != "comsol":
        return []
    particles = runtime.particles
    density = np.asarray(particles.density, dtype=np.float64)
    declared = np.isfinite(density) & (density > 0.0)
    if not np.any(declared):
        return []
    mass = np.asarray(particles.mass, dtype=np.float64)
    diameter = np.asarray(particles.diameter, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        sphere_mass = density * np.pi * diameter**3 / 6.0
    consistent = np.isclose(
        mass,
        sphere_mass,
        rtol=_COMSOL_SPHERE_RELATIVE_TOLERANCE,
        atol=0.0,
    )
    inconsistent = np.flatnonzero(declared & ~consistent)
    if not inconsistent.size:
        return []
    sample = inconsistent[:12]
    return [
        ValidationIssue(
            "physics.particle.sphere_consistency",
            "COMSOL particle mass_kg, density_kgm3, and drag_diameter_m "
            "must describe the same sphere",
            context={
                "relation": "mass_kg = density_kgm3 * pi * drag_diameter_m^3 / 6",
                "relative_tolerance": _COMSOL_SPHERE_RELATIVE_TOLERANCE,
                "particle_ids": [int(particles.particle_id[index]) for index in sample],
                "inconsistent_count": int(inconsistent.size),
                "actual_mass_kg": [float(mass[index]) for index in sample],
                "sphere_mass_kg": [float(sphere_mass[index]) for index in sample],
            },
        )
    ]


def _dielectrophoresis_particle_issues(
    runtime: Any,
    by_name: Mapping[str, Any],
) -> list[ValidationIssue]:
    if not _force_enabled(by_name, "dielectrophoresis"):
        return []
    config = getattr(by_name["dielectrophoresis"], "config", {})
    configured = (
        config.get("particle_rel_permittivity") if isinstance(config, Mapping) else None
    )
    if _positive_number(configured):
        return []
    permittivity = np.asarray(
        runtime.particles.dep_particle_rel_permittivity,
        dtype=np.float64,
    )
    invalid_rows = np.flatnonzero(~np.isfinite(permittivity) | (permittivity <= 0.0))
    if not invalid_rows.size:
        return []
    return [
        ValidationIssue(
            "physics.dielectrophoresis.particle_permittivity.missing",
            "dielectrophoresis requires particle_rel_permittivity in the force "
            "config or every particle row",
            context={
                "particle_ids": [
                    int(runtime.particles.particle_id[index])
                    for index in invalid_rows[:12]
                ],
                "invalid_count": int(invalid_rows.size),
            },
        )
    ]


def _missing_gas_issues(
    requirements: Mapping[str, tuple[str, ...]],
    configured_gas: Any,
    runtime_gas: Any,
    field_quantities: set[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for feature, required in sorted(requirements.items()):
        missing = [
            name
            for name in required
            if not _gas_quantity_available(
                configured_gas,
                runtime_gas,
                field_quantities,
                name,
            )
        ]
        if missing:
            issues.append(
                ValidationIssue(
                    "physics.gas.missing",
                    f"{feature} requires explicit gas quantities: {', '.join(missing)}",
                    context={"feature": feature, "missing": missing},
                )
            )
    return issues


def physics_requirement_issues(case: Any, runtime: Any) -> list[ValidationIssue]:
    """Reject enabled models whose required quantities have no explicit source."""

    physics = getattr(case.config, "physics", None)
    configured_gas = getattr(physics, "gas", None)
    runtime_gas = getattr(runtime, "gas", None)
    field = getattr(getattr(runtime, "field_provider", None), "field", None)
    quantities = {str(name) for name in getattr(field, "quantities", {})}
    catalog = getattr(runtime, "force_catalog", None)
    by_name = (
        catalog.by_name() if catalog is not None and hasattr(catalog, "by_name") else {}
    )
    spatial_dim = int(getattr(runtime, "spatial_dim", 2))
    velocity_fields = tuple(choose_velocity_quantity_names(field, spatial_dim))
    electric_fields = tuple(choose_electric_field_quantity_names(field, spatial_dim))
    temperature_field = next(
        (
            name
            for name in ("T", "temperature", "gas_temperature")
            if name in quantities
        ),
        "",
    )
    issues = _force_field_issues(
        by_name,
        velocity_fields,
        electric_fields,
        temperature_field,
        _fluid_acceleration_fields(quantities, spatial_dim),
    )
    requirements, density_features, stochastic_issue = _gas_requirements(
        physics,
        by_name,
    )
    if stochastic_issue is not None:
        return [stochastic_issue]
    issues.extend(_particle_density_issues(runtime, density_features))
    issues.extend(_comsol_sphere_consistency_issues(case, runtime))
    issues.extend(_dielectrophoresis_particle_issues(runtime, by_name))
    issues.extend(
        _missing_gas_issues(
            requirements,
            configured_gas,
            runtime_gas,
            quantities,
        )
    )
    return issues


__all__ = ("physics_requirement_issues",)
