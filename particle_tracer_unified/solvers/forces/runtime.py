from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .registry import ForceCatalog


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


def force_runtime_parameters_from_catalog(catalog: ForceCatalog | None) -> ForceRuntimeParameters:
    if catalog is None:
        return ForceRuntimeParameters()
    by_name = catalog.by_name()
    thermo = by_name.get("thermophoresis")
    dep = by_name.get("dielectrophoresis")
    lift = by_name.get("lift")
    gravity = by_name.get("gravity")
    thermo_cfg = thermo.config if thermo is not None else {}
    dep_cfg = dep.config if dep is not None else {}
    lift_cfg = lift.config if lift is not None else {}
    gravity_cfg = gravity.config if gravity is not None else {}
    return ForceRuntimeParameters(
        thermophoresis_enabled=bool(thermo.enabled) if thermo is not None else False,
        thermophoresis_model=_str_cfg(thermo_cfg, "model", default="talbot").lower(),
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
        dielectrophoresis_model=_str_cfg(dep_cfg, "model", default="dc").lower(),
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
        lift_model=_str_cfg(lift_cfg, "model", default="saffman").lower(),
        lift_coefficient=_float_cfg(lift_cfg, "coefficient", "saffman_coefficient", default=6.46),
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
        "gravity_buoyancy_enabled": int(bool(p.gravity_buoyancy_enabled)),
    }

