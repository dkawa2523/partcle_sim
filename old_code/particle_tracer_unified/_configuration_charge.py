"""Charging and plasma-background configuration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from ._configuration_core import (
    enum,
    error,
    finite_number,
    integer,
    mapping,
    parameters,
    reject_unknown,
    required,
    strict_bool,
    string,
)

CHARGE_PARAMETER_KEYS = frozenset(
    {
        "background_source",
        "electron_temperature_quantity",
        "electron_temperature_unit",
        "electron_density_quantity",
        "ion_density_quantity",
        "ion_temperature_quantity",
        "ion_temperature_eV",
        "ion_mass_amu",
        "ion_charge_number",
        "electron_sticking",
        "ion_sticking",
        "ion_velocity_model",
        "bohm_velocity_factor",
        "te_relaxation_alpha",
        "relaxation_time_s",
        "max_abs_potential_V",
        "root_iterations",
    }
)
PLASMA_BACKGROUND_KEYS = frozenset(
    {
        "source",
        "electron_density_m3",
        "ion_density_m3",
        "electron_temperature_eV",
        "ion_temperature_eV",
        "ion_mass_amu",
        "ion_charge_number",
        "pressure_Pa",
        "gas_temperature_K",
        "neutral_molecular_mass_amu",
        "electron_neutral_cross_section_m2",
        "ion_neutral_cross_section_m2",
        "electron_collision_frequency_s",
        "ion_collision_frequency_s",
        "electron_ion_collision_frequency_s",
        "conductivity_Sm",
    }
)
CHARGE_QUANTITY_PARAMETERS = (
    "electron_temperature_quantity",
    "electron_density_quantity",
    "ion_density_quantity",
    "ion_temperature_quantity",
)
POSITIVE_CHARGE_PARAMETERS = (
    "ion_temperature_eV",
    "ion_mass_amu",
    "ion_charge_number",
    "bohm_velocity_factor",
    "te_relaxation_alpha",
    "relaxation_time_s",
    "max_abs_potential_V",
)
NONNEGATIVE_CHARGE_PARAMETERS = ("electron_sticking", "ion_sticking")
REQUIRED_PLASMA_BACKGROUND = frozenset(
    {
        "electron_density_m3",
        "ion_density_m3",
        "electron_temperature_eV",
        "ion_temperature_eV",
        "ion_mass_amu",
        "ion_charge_number",
    }
)
RELAXATION_PARAMETERS = frozenset({"te_relaxation_alpha", "relaxation_time_s"})


def _validate_charge_mode_ownership(
    mode: str,
    charge_parameters: Mapping[str, Any],
    path: str,
) -> None:
    if mode != "oml_linearized_relaxation":
        return
    forbidden = sorted(RELAXATION_PARAMETERS.intersection(charge_parameters))
    if forbidden:
        raise error(path, "OML mode does not accept " + ", ".join(forbidden))


def _validate_charge_parameter_text(
    charge_parameters: Mapping[str, Any],
    path: str,
) -> None:
    for name in CHARGE_QUANTITY_PARAMETERS:
        if name in charge_parameters:
            string(charge_parameters[name], f"{path}.{name}", allow_empty=True)
    if "background_source" in charge_parameters:
        enum(
            charge_parameters["background_source"],
            {"field", "plasma_background"},
            f"{path}.background_source",
        )
    if "electron_temperature_unit" in charge_parameters:
        unit_path = f"{path}.electron_temperature_unit"
        unit = string(charge_parameters["electron_temperature_unit"], unit_path)
        if unit not in {"eV", "K"}:
            raise error(unit_path, "must be eV or K")
    if "ion_velocity_model" in charge_parameters:
        enum(
            charge_parameters["ion_velocity_model"],
            {"bohm", "thermal", "max_thermal_bohm"},
            f"{path}.ion_velocity_model",
        )
    if "root_iterations" in charge_parameters:
        integer(
            charge_parameters["root_iterations"],
            f"{path}.root_iterations",
            minimum=1,
        )


def _validate_charge_parameter_numbers(
    charge_parameters: Mapping[str, Any],
    path: str,
) -> None:
    for name in POSITIVE_CHARGE_PARAMETERS:
        if name in charge_parameters:
            finite_number(
                charge_parameters[name],
                f"{path}.{name}",
                minimum=0.0,
                exclusive_minimum=True,
            )
    for name in NONNEGATIVE_CHARGE_PARAMETERS:
        if name in charge_parameters:
            finite_number(charge_parameters[name], f"{path}.{name}", minimum=0.0)


def _validate_plasma_background_values(
    background: Mapping[str, Any],
    path: str,
) -> None:
    if "source" in background:
        enum(background["source"], {"saas_constant"}, f"{path}.source")
    nonnegative = PLASMA_BACKGROUND_KEYS.difference({"source"}).difference(
        REQUIRED_PLASMA_BACKGROUND
    )
    for name in sorted(REQUIRED_PLASMA_BACKGROUND.intersection(background)):
        finite_number(
            background[name],
            f"{path}.{name}",
            minimum=0.0,
            exclusive_minimum=True,
        )
    for name in sorted(nonnegative.intersection(background)):
        finite_number(background[name], f"{path}.{name}", minimum=0.0)


def _validate_plasma_background_selection(
    background: Mapping[str, Any],
    selected_background: str,
    path: str,
) -> None:
    if not background:
        if selected_background == "plasma_background":
            raise error(
                path,
                "is required when parameters.background_source is plasma_background",
            )
        return
    if selected_background != "plasma_background":
        raise error(
            path,
            "is only valid when parameters.background_source is plasma_background",
        )
    if background.get("source") != "saas_constant":
        raise error(f"{path}.source", "must be saas_constant")
    missing = sorted(REQUIRED_PLASMA_BACKGROUND.difference(background))
    if missing:
        raise error(path, "saas_constant requires " + ", ".join(missing))


def _validate_enabled_charge_requirements(
    *,
    enabled: bool,
    mode: str,
    charge_parameters: Mapping[str, Any],
    selected_background: str,
    path: str,
) -> None:
    if not enabled:
        return
    if mode == "te_relaxation":
        missing = sorted(RELAXATION_PARAMETERS.difference(charge_parameters))
        if missing:
            raise error(
                path,
                "enabled te_relaxation requires explicit " + ", ".join(missing),
            )
        return
    if selected_background == "plasma_background":
        return
    missing_ion = sorted(
        {"ion_mass_amu", "ion_charge_number"}.difference(charge_parameters)
    )
    if (
        "ion_temperature_quantity" not in charge_parameters
        and "ion_temperature_eV" not in charge_parameters
    ):
        missing_ion.append("ion_temperature_quantity|ion_temperature_eV")
    if missing_ion:
        raise error(
            path,
            "OML field background requires explicit " + ", ".join(missing_ion),
        )


@dataclass(frozen=True)
class ChargeConfig:
    enabled: bool
    mode: str = "te_relaxation"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    background: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any, path: str = "physics.charge") -> ChargeConfig:
        data = mapping(value, path)
        reject_unknown(data, {"enabled", "mode", "parameters", "background"}, path)
        enabled = strict_bool(required(data, "enabled", path), f"{path}.enabled")
        mode = enum(
            required(data, "mode", path),
            {"te_relaxation", "oml_linearized_relaxation"},
            f"{path}.mode",
        )
        charge_parameters = parameters(data.get("parameters", {}), f"{path}.parameters")
        background = parameters(data.get("background", {}), f"{path}.background")
        reject_unknown(charge_parameters, CHARGE_PARAMETER_KEYS, f"{path}.parameters")
        reject_unknown(background, PLASMA_BACKGROUND_KEYS, f"{path}.background")
        parameter_path = f"{path}.parameters"
        background_path = f"{path}.background"
        _validate_charge_mode_ownership(mode, charge_parameters, parameter_path)
        _validate_charge_parameter_text(charge_parameters, parameter_path)
        _validate_charge_parameter_numbers(charge_parameters, parameter_path)
        _validate_plasma_background_values(background, background_path)
        selected_background = str(charge_parameters.get("background_source", "field"))
        _validate_plasma_background_selection(
            background,
            selected_background,
            background_path,
        )
        _validate_enabled_charge_requirements(
            enabled=enabled,
            mode=mode,
            charge_parameters=charge_parameters,
            selected_background=selected_background,
            path=parameter_path,
        )
        return cls(
            enabled=enabled,
            mode=mode,
            parameters=charge_parameters,
            background=background,
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"enabled": bool(self.enabled), "mode": self.mode}
        if self.parameters:
            result["parameters"] = deepcopy(dict(self.parameters))
        if self.background:
            result["background"] = deepcopy(dict(self.background))
        return result
