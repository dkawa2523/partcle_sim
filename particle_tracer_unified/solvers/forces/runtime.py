from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .registry import ForceCatalog

_ALLOWED_THERMOPHORESIS_MODELS = {"talbot", "continuum"}
_ALLOWED_DIELECTROPHORESIS_MODELS = {"dc", "ac_clausius_mossotti"}
_ALLOWED_LIFT_MODELS = {"saffman"}
_ALLOWED_PRESSURE_GRADIENT_MODELS = {"fluid_material_acceleration"}
_ALLOWED_VIRTUAL_MASS_MODELS = {"particle_material_acceleration"}


@dataclass(frozen=True)
class ForceRuntimeParameters:
    thermophoresis_enabled: bool = False
    thermophoresis_model: str = "talbot"
    gas_thermal_conductivity_W_mK: float = 0.026
    particle_thermal_conductivity_W_mK: float = 1.4
    thermophoresis_Cs: float = 1.17
    thermophoresis_Cm: float = 1.14
    thermophoresis_Ct: float = 2.18

    dielectrophoresis_enabled: bool = False
    dielectrophoresis_model: str = "dc"
    dep_medium_rel_permittivity: float = 1.0006
    dep_particle_rel_permittivity: float = float("nan")
    dep_medium_conductivity_Sm: float = 0.0
    dep_particle_conductivity_Sm: float = 0.0
    dep_frequency_Hz: float = 0.0

    lift_enabled: bool = False
    lift_model: str = "saffman"
    lift_coefficient: float = 6.46

    pressure_gradient_enabled: bool = False
    pressure_gradient_model: str = "fluid_material_acceleration"

    virtual_mass_enabled: bool = False
    virtual_mass_model: str = "particle_material_acceleration"
    virtual_mass_coefficient: float = 0.5

    gravity_buoyancy_enabled: bool = False


def _float_cfg(cfg: Mapping[str, Any], *names: str, default: float) -> float:
    for name in names:
        if name in cfg and cfg[name] is not None:
            return float(cfg[name])
    return float(default)


def _str_cfg(cfg: Mapping[str, Any], *names: str, default: str) -> str:
    for name in names:
        value = str(cfg.get(name, "")).strip()
        if value:
            return value
    return str(default)


def _bool_cfg(cfg: Mapping[str, Any], *names: str, default: bool) -> bool:
    for name in names:
        if name not in cfg:
            continue
        value = cfg.get(name)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    return bool(default)


def _require_model(name: str, model: str, allowed: set[str]) -> str:
    value = str(model).strip().lower()
    if value not in allowed:
        choices = "', '".join(sorted(allowed))
        raise ValueError(f"solver.forces.{name}.model must be one of '{choices}'")
    return value


def force_runtime_parameters_from_catalog(catalog: ForceCatalog | None) -> ForceRuntimeParameters:
    if catalog is None:
        return ForceRuntimeParameters()
    by_name = catalog.by_name()
    thermo = by_name.get("thermophoresis")
    dep = by_name.get("dielectrophoresis")
    lift = by_name.get("lift")
    pressure_gradient = by_name.get("pressure_gradient")
    virtual_mass = by_name.get("virtual_mass")
    gravity = by_name.get("gravity")
    thermo_cfg = thermo.config if thermo is not None else {}
    dep_cfg = dep.config if dep is not None else {}
    lift_cfg = lift.config if lift is not None else {}
    pressure_gradient_cfg = pressure_gradient.config if pressure_gradient is not None else {}
    virtual_mass_cfg = virtual_mass.config if virtual_mass is not None else {}
    gravity_cfg = gravity.config if gravity is not None else {}
    thermophoresis_model = _require_model(
        "thermophoresis",
        _str_cfg(thermo_cfg, "model", default="talbot"),
        _ALLOWED_THERMOPHORESIS_MODELS,
    )
    dielectrophoresis_model = _require_model(
        "dielectrophoresis",
        _str_cfg(dep_cfg, "model", default="dc"),
        _ALLOWED_DIELECTROPHORESIS_MODELS,
    )
    lift_model = _require_model(
        "lift",
        _str_cfg(lift_cfg, "model", default="saffman"),
        _ALLOWED_LIFT_MODELS,
    )
    pressure_gradient_model = _require_model(
        "pressure_gradient",
        _str_cfg(pressure_gradient_cfg, "model", default="fluid_material_acceleration"),
        _ALLOWED_PRESSURE_GRADIENT_MODELS,
    )
    virtual_mass_model = _require_model(
        "virtual_mass",
        _str_cfg(virtual_mass_cfg, "model", default="particle_material_acceleration"),
        _ALLOWED_VIRTUAL_MASS_MODELS,
    )
    return ForceRuntimeParameters(
        thermophoresis_enabled=bool(thermo.enabled) if thermo is not None else False,
        thermophoresis_model=thermophoresis_model,
        gas_thermal_conductivity_W_mK=_float_cfg(
            thermo_cfg,
            "gas_thermal_conductivity_W_mK",
            "fluid_thermal_conductivity_W_mK",
            default=0.026,
        ),
        particle_thermal_conductivity_W_mK=_float_cfg(
            thermo_cfg,
            "particle_thermal_conductivity_W_mK",
            default=1.4,
        ),
        thermophoresis_Cs=_float_cfg(thermo_cfg, "Cs", "C_s", default=1.17),
        thermophoresis_Cm=_float_cfg(thermo_cfg, "Cm", "C_m", default=1.14),
        thermophoresis_Ct=_float_cfg(thermo_cfg, "Ct", "C_t", default=2.18),
        dielectrophoresis_enabled=bool(dep.enabled) if dep is not None else False,
        dielectrophoresis_model=dielectrophoresis_model,
        dep_medium_rel_permittivity=_float_cfg(
            dep_cfg,
            "medium_rel_permittivity",
            "fluid_rel_permittivity",
            "epsilon_r_medium",
            default=1.0006,
        ),
        dep_particle_rel_permittivity=_float_cfg(
            dep_cfg,
            "particle_rel_permittivity",
            "epsilon_r_particle",
            default=float("nan"),
        ),
        dep_medium_conductivity_Sm=_float_cfg(
            dep_cfg,
            "medium_conductivity_Sm",
            "fluid_conductivity_Sm",
            default=0.0,
        ),
        dep_particle_conductivity_Sm=_float_cfg(dep_cfg, "particle_conductivity_Sm", default=0.0),
        dep_frequency_Hz=_float_cfg(dep_cfg, "frequency_Hz", default=0.0),
        lift_enabled=bool(lift.enabled) if lift is not None else False,
        lift_model=lift_model,
        lift_coefficient=_float_cfg(lift_cfg, "coefficient", "saffman_coefficient", default=6.46),
        pressure_gradient_enabled=bool(pressure_gradient.enabled) if pressure_gradient is not None else False,
        pressure_gradient_model=pressure_gradient_model,
        virtual_mass_enabled=bool(virtual_mass.enabled) if virtual_mass is not None else False,
        virtual_mass_model=virtual_mass_model,
        virtual_mass_coefficient=_float_cfg(
            virtual_mass_cfg,
            "coefficient",
            "added_mass_coefficient",
            "Cvm",
            "C_vm",
            default=0.5,
        ),
        gravity_buoyancy_enabled=(
            bool(gravity.enabled) and _bool_cfg(gravity_cfg, "buoyancy", "include_buoyancy", default=False)
            if gravity is not None
            else False
        ),
    )


def force_runtime_parameters_summary(params: ForceRuntimeParameters | None) -> dict[str, object]:
    p = params or ForceRuntimeParameters()
    return {
        "thermophoresis_enabled": int(bool(p.thermophoresis_enabled)),
        "thermophoresis_model": str(p.thermophoresis_model),
        "dielectrophoresis_enabled": int(bool(p.dielectrophoresis_enabled)),
        "dielectrophoresis_model": str(p.dielectrophoresis_model),
        "lift_enabled": int(bool(p.lift_enabled)),
        "lift_model": str(p.lift_model),
        "pressure_gradient_enabled": int(bool(p.pressure_gradient_enabled)),
        "pressure_gradient_model": str(p.pressure_gradient_model),
        "virtual_mass_enabled": int(bool(p.virtual_mass_enabled)),
        "virtual_mass_model": str(p.virtual_mass_model),
        "virtual_mass_coefficient": float(p.virtual_mass_coefficient),
        "gravity_buoyancy_enabled": int(bool(p.gravity_buoyancy_enabled)),
        "rejected_model_policy": "fail_fast",
        "implemented_equations": {
            "thermophoresis": "talbot_like_with_continuum_kn_zero" if bool(p.thermophoresis_enabled) else "",
            "dielectrophoresis": "clausius_mossotti_gradient_e_squared" if bool(p.dielectrophoresis_enabled) else "",
            "lift": "saffman_vorticity_form" if bool(p.lift_enabled) else "",
            "pressure_gradient": "rho_g_over_rho_p_fluid_material_acceleration" if bool(p.pressure_gradient_enabled) else "",
            "virtual_mass": "coefficient_rho_g_over_rho_p_particle_path_fluid_acceleration" if bool(p.virtual_mass_enabled) else "",
            "gravity_buoyancy": "body_acceleration_scaled_by_one_minus_rho_g_over_rho_p" if bool(p.gravity_buoyancy_enabled) else "",
        },
    }
