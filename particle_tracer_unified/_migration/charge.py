"""Legacy charge-model and plasma-background configuration conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from particle_tracer_unified.configuration import (
    CHARGE_PARAMETER_KEYS,
    PLASMA_BACKGROUND_KEYS,
)

from .legacy import (
    _canonical_choice,
    _canonical_keys,
    _legacy_bool,
    _merge_without_conflicts,
    _token,
)

_CHARGE_PARAMETER_ALIASES = {
    "alpha": "te_relaxation_alpha",
    "te_alpha": "te_relaxation_alpha",
    "tau_s": "relaxation_time_s",
    "tau_q_s": "relaxation_time_s",
    "charge_relaxation_time_s": "relaxation_time_s",
    "background_type": "background_source",
    "te_quantity": "electron_temperature_quantity",
    "te_unit": "electron_temperature_unit",
    "ne_quantity": "electron_density_quantity",
    "ni_quantity": "ion_density_quantity",
    "ti_quantity": "ion_temperature_quantity",
    "ti_ev": "ion_temperature_eV",
    "mi_amu": "ion_mass_amu",
    "zi": "ion_charge_number",
}

_PLASMA_BACKGROUND_ALIASES = {
    "ne": "electron_density_m3",
    "ne_m3": "electron_density_m3",
    "electron_density": "electron_density_m3",
    "ni": "ion_density_m3",
    "ni_m3": "ion_density_m3",
    "ion_density": "ion_density_m3",
    "te": "electron_temperature_eV",
    "te_ev": "electron_temperature_eV",
    "electron_temperature": "electron_temperature_eV",
    "ti": "ion_temperature_eV",
    "ti_ev": "ion_temperature_eV",
    "ion_temperature": "ion_temperature_eV",
    "mi_amu": "ion_mass_amu",
    "ion_mass": "ion_mass_amu",
    "zi": "ion_charge_number",
    "charge_number": "ion_charge_number",
    "pressure": "pressure_Pa",
    "gas_pressure_pa": "pressure_Pa",
    "tg_k": "gas_temperature_K",
    "gas_temperature": "gas_temperature_K",
    "neutral_mass_amu": "neutral_molecular_mass_amu",
    "sigma_en_m2": "electron_neutral_cross_section_m2",
    "sigma_in_m2": "ion_neutral_cross_section_m2",
    "nu_en_s": "electron_collision_frequency_s",
    "nu_in_s": "ion_collision_frequency_s",
    "nu_ei_s": "electron_ion_collision_frequency_s",
    "conductivity": "conductivity_Sm",
    "conductivity_s_m": "conductivity_Sm",
}


def _migrate_plasma_background(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("legacy solver.plasma_background must be a mapping")
    if not raw:
        return None
    background = _canonical_keys(
        dict(raw),
        canonical=tuple(PLASMA_BACKGROUND_KEYS),
        aliases=_PLASMA_BACKGROUND_ALIASES,
        label="solver.plasma_background",
    )
    if "source" not in background:
        raise ValueError("legacy solver.plasma_background.source is required")
    source_token = _token(background["source"])
    if source_token in {"none", "off", "disabled"}:
        if len(background) != 1:
            raise ValueError(
                "disabled legacy plasma background cannot contain physical parameters"
            )
        return None
    background["source"] = _canonical_choice(
        background["source"],
        canonical=("saas_constant",),
        aliases={
            "saas": "saas_constant",
            "constant": "saas_constant",
            "scalar": "saas_constant",
            "scalar_constant": "saas_constant",
        },
        label="plasma background source",
    )
    return background


def _canonical_charge_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"enabled": raw}
    return _canonical_keys(
        dict(raw),
        canonical=(
            "enabled",
            "mode",
            "stride",
            "parameters",
            *CHARGE_PARAMETER_KEYS,
        ),
        aliases={
            "active": "enabled",
            "update_stride": "stride",
            **_CHARGE_PARAMETER_ALIASES,
        },
        label="solver.charge_model",
    )


def _extract_charge_parameters(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.pop("parameters", {})
    if not isinstance(nested, Mapping):
        raise ValueError("legacy solver.charge_model.parameters must be a mapping")
    parameters = _canonical_keys(
        dict(nested),
        canonical=tuple(CHARGE_PARAMETER_KEYS),
        aliases=_CHARGE_PARAMETER_ALIASES,
        label="solver.charge_model.parameters",
    )
    direct_parameters = {
        key: config.pop(key) for key in tuple(config) if key in CHARGE_PARAMETER_KEYS
    }
    _merge_without_conflicts(
        parameters,
        direct_parameters,
        label="charge parameters",
    )
    return parameters


def _validate_charge_stride(config: Mapping[str, Any]) -> None:
    if int(config.get("stride", 1)) != 1:
        raise ValueError(
            "legacy charge update stride cannot be migrated; "
            "v0.2 updates every accepted substep"
        )


def _canonical_charge_mode(config: Mapping[str, Any]) -> str:
    return _canonical_choice(
        config.get("mode", "te_relaxation"),
        canonical=("te_relaxation", "oml_linearized_relaxation"),
        aliases={
            "electron_temperature_relaxation": "te_relaxation",
            "te_temperature_relaxation": "te_relaxation",
            "oml": "oml_linearized_relaxation",
            "oml_linearized": "oml_linearized_relaxation",
            "density_temperature_flux_relaxation": "oml_linearized_relaxation",
            "finite_rate_flux_balance": "oml_linearized_relaxation",
        },
        label="charge model",
    )


def _canonicalize_charge_parameter_choices(parameters: dict[str, Any]) -> None:
    if "background_source" in parameters:
        parameters["background_source"] = _canonical_choice(
            parameters["background_source"],
            canonical=("field", "plasma_background"),
            aliases={
                "fields": "field",
                "provider_field": "field",
                "plasma": "plasma_background",
                "saas": "plasma_background",
                "constant": "plasma_background",
            },
            label="charge background source",
        )
    if "electron_temperature_unit" in parameters:
        parameters["electron_temperature_unit"] = _canonical_choice(
            parameters["electron_temperature_unit"],
            canonical=("eV", "K"),
            aliases={
                "electron_volt": "eV",
                "electron_volts": "eV",
                "kelvin": "K",
            },
            label="electron temperature unit",
        )
    if "ion_velocity_model" in parameters:
        parameters["ion_velocity_model"] = _canonical_choice(
            parameters["ion_velocity_model"],
            canonical=("bohm", "thermal", "max_thermal_bohm"),
            aliases={
                "max_bohm_thermal": "max_thermal_bohm",
                "thermal_or_bohm": "max_thermal_bohm",
            },
            label="ion velocity model",
        )


def _validate_charge_mode_parameters(
    mode: str,
    parameters: Mapping[str, Any],
) -> None:
    if mode != "oml_linearized_relaxation":
        return
    relaxation_parameters = {"te_relaxation_alpha", "relaxation_time_s"}
    forbidden = sorted(relaxation_parameters.intersection(parameters))
    if forbidden:
        raise ValueError(
            "legacy OML charge model cannot migrate Te-relaxation parameters: "
            + ", ".join(forbidden)
        )


_TE_RELAXATION_DEFAULTS = (
    (
        "te_relaxation_alpha",
        2.5,
        "solver.charge_model.te_relaxation_alpha was absent; "
        "materialized the legacy default 2.5",
    ),
    (
        "relaxation_time_s",
        1.0e-6,
        "solver.charge_model.relaxation_time_s was absent; "
        "materialized the legacy default 1e-6 s",
    ),
)


def _materialize_charge_defaults(
    *,
    enabled: bool,
    mode: str,
    parameters: dict[str, Any],
    warnings: list[str],
) -> None:
    if not enabled or mode != "te_relaxation":
        return
    for name, value, warning in _TE_RELAXATION_DEFAULTS:
        if name not in parameters:
            parameters[name] = value
            warnings.append(warning)


def _apply_charge_background(
    parameters: dict[str, Any],
    background: dict[str, Any] | None,
    warnings: list[str],
) -> None:
    if background is None:
        return
    if "background_source" not in parameters:
        parameters["background_source"] = "plasma_background"
        warnings.append(
            "solver.charge_model.background_source was absent; selected "
            "plasma_background "
            "because solver.plasma_background was explicit"
        )
    elif parameters["background_source"] != "plasma_background":
        raise ValueError(
            "legacy charge background_source conflicts with solver.plasma_background"
        )


def _charge_result(
    *,
    enabled: bool,
    mode: str,
    parameters: dict[str, Any],
    background: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"enabled": enabled, "mode": mode}
    if parameters:
        result["parameters"] = parameters
    if background is not None:
        result["background"] = background
    return result


def _migrate_charge(
    solver: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    raw = solver.get("charge_model")
    background = _migrate_plasma_background(solver.get("plasma_background"))
    if raw is None:
        if background is not None:
            raise ValueError(
                "legacy solver.plasma_background requires an explicit "
                "solver.charge_model"
            )
        return None
    config = _canonical_charge_config(raw)
    parameters = _extract_charge_parameters(config)
    _validate_charge_stride(config)
    mode = _canonical_charge_mode(config)
    _canonicalize_charge_parameter_choices(parameters)
    enabled = _legacy_bool(config.get("enabled", False), default=False)
    _validate_charge_mode_parameters(mode, parameters)
    _materialize_charge_defaults(
        enabled=enabled,
        mode=mode,
        parameters=parameters,
        warnings=warnings,
    )
    _apply_charge_background(parameters, background, warnings)
    return _charge_result(
        enabled=enabled,
        mode=mode,
        parameters=parameters,
        background=background,
    )
