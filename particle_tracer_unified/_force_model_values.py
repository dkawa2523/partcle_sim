"""Validate scalar values and build one canonical semantic force."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ._force_model_types import (
    _DEFAULT_MODELS,
    DEP_ELECTRIC_FIELD_AMPLITUDES,
    DEP_MEDIUM_REL_PERMITTIVITY_DEFAULT,
    DRAG_MODELS,
    FORCE_MODELS,
    FORCE_PARAMETER_KEYS,
    LIFT_COEFFICIENT_DEFAULT,
    THERMOPHORESIS_CM_DEFAULT,
    THERMOPHORESIS_CS_DEFAULT,
    THERMOPHORESIS_CT_DEFAULT,
    THERMOPHORESIS_GAS_CONDUCTIVITY_DEFAULT,
    THERMOPHORESIS_PARTICLE_CONDUCTIVITY_DEFAULT,
    VIRTUAL_MASS_COEFFICIENT_DEFAULT,
    DielectrophoresisForce,
    DragForce,
    ElectricForce,
    ForceModelError,
    GravityForce,
    LiftForce,
    PressureGradientForce,
    SemanticForce,
    ThermophoresisForce,
    VirtualMassForce,
)


def _error(path: str, message: str) -> ForceModelError:
    return ForceModelError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "must be a mapping")
    return {str(key): item for key, item in value.items()}


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(str(key) for key in data if str(key) not in allowed)
    if unknown:
        raise _error(path, f"unknown key(s): {', '.join(unknown)}")


def _required(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise _error(path, f"missing required key {key!r}")
    return data[key]


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise _error(path, "must be a YAML boolean (true or false)")
    return bool(value)


def _model(value: Any, allowed: frozenset[str], path: str) -> str:
    if not isinstance(value, str):
        raise _error(path, "must be a string")
    if value != value.strip():
        raise _error(path, "must not contain leading or trailing whitespace")
    if value not in allowed:
        raise _error(path, f"must be one of {sorted(allowed)}, got {value!r}")
    return value


def _finite(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, "must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if exclusive_minimum else result < minimum
        if invalid:
            operator = ">" if exclusive_minimum else ">="
            raise _error(path, f"must be {operator} {minimum:g}")
    return result


def _parameters(
    data: Mapping[str, Any], path: str, allowed: set[str]
) -> dict[str, Any]:
    parameters = _mapping(data.get("parameters", {}), path)
    _reject_unknown(parameters, allowed, path)
    return parameters


def parse_drag_force(value: Any, *, path: str = "physics.drag") -> DragForce:
    data = _mapping(value, path)
    _reject_unknown(data, {"model"}, path)
    model = _model(_required(data, "model", path), DRAG_MODELS, f"{path}.model")
    return DragForce(enabled=True, model=model)


def _optional_model_and_parameters(
    name: str,
    data: Mapping[str, Any],
    path: str,
) -> tuple[str, dict[str, Any]]:
    model = _model(
        data.get("model", _DEFAULT_MODELS[name]),
        FORCE_MODELS[name],
        f"{path}.model",
    )
    parameters = _parameters(
        data,
        f"{path}.parameters",
        set(FORCE_PARAMETER_KEYS[name]),
    )
    return model, parameters


def _parse_gravity_force(
    enabled: bool,
    model: str,
    parameters: Mapping[str, Any],
    *,
    path: str,
    spatial_dim: int,
) -> GravityForce:
    acceleration_path = f"{path}.parameters.acceleration_mps2"
    acceleration_raw = parameters.get("acceleration_mps2")
    if enabled and acceleration_raw is None:
        raise _error(acceleration_path, "is required when gravity is enabled")

    acceleration: tuple[float, ...] = ()
    if acceleration_raw is not None:
        if not isinstance(acceleration_raw, (list, tuple)):
            raise _error(
                acceleration_path,
                f"must contain exactly {spatial_dim} components",
            )
        if len(acceleration_raw) != spatial_dim:
            raise _error(
                acceleration_path,
                f"must contain exactly {spatial_dim} components",
            )
        acceleration = tuple(
            _finite(component, f"{acceleration_path}[{index}]")
            for index, component in enumerate(acceleration_raw)
        )

    buoyancy = (
        _boolean(parameters["buoyancy"], f"{path}.parameters.buoyancy")
        if "buoyancy" in parameters
        else False
    )
    return GravityForce(
        enabled=enabled,
        model=model,
        acceleration_mps2=acceleration,
        buoyancy=buoyancy,
    )


def _parse_thermophoresis_force(
    enabled: bool,
    model: str,
    parameters: Mapping[str, Any],
    *,
    path: str,
) -> ThermophoresisForce:
    required = {
        "gas_thermal_conductivity_W_mK",
        "particle_thermal_conductivity_W_mK",
    }
    missing = sorted(required.difference(parameters)) if enabled else []
    if missing:
        raise _error(
            f"{path}.parameters",
            "enabled thermophoresis requires explicit " + ", ".join(missing),
        )
    return ThermophoresisForce(
        enabled=enabled,
        model=model,
        gas_thermal_conductivity_W_mK=_finite(
            parameters.get(
                "gas_thermal_conductivity_W_mK",
                THERMOPHORESIS_GAS_CONDUCTIVITY_DEFAULT,
            ),
            f"{path}.parameters.gas_thermal_conductivity_W_mK",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        particle_thermal_conductivity_W_mK=_finite(
            parameters.get(
                "particle_thermal_conductivity_W_mK",
                THERMOPHORESIS_PARTICLE_CONDUCTIVITY_DEFAULT,
            ),
            f"{path}.parameters.particle_thermal_conductivity_W_mK",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        Cs=_finite(
            parameters.get("Cs", THERMOPHORESIS_CS_DEFAULT),
            f"{path}.parameters.Cs",
            minimum=0.0,
        ),
        Cm=_finite(
            parameters.get("Cm", THERMOPHORESIS_CM_DEFAULT),
            f"{path}.parameters.Cm",
            minimum=0.0,
        ),
        Ct=_finite(
            parameters.get("Ct", THERMOPHORESIS_CT_DEFAULT),
            f"{path}.parameters.Ct",
            minimum=0.0,
        ),
    )


def _require_dielectrophoresis_parameters(
    enabled: bool,
    model: str,
    parameters: Mapping[str, Any],
    path: str,
) -> None:
    if enabled and "medium_rel_permittivity" not in parameters:
        raise _error(
            f"{path}.parameters",
            "enabled dielectrophoresis requires explicit medium_rel_permittivity",
        )
    if not enabled or model != "ac_clausius_mossotti":
        return
    required = {
        "medium_conductivity_Sm",
        "particle_conductivity_Sm",
        "frequency_Hz",
    }
    missing = sorted(required.difference(parameters))
    if missing:
        raise _error(
            f"{path}.parameters",
            "AC dielectrophoresis requires explicit " + ", ".join(missing),
        )


def _parse_dielectrophoresis_force(
    enabled: bool,
    model: str,
    parameters: Mapping[str, Any],
    *,
    path: str,
) -> DielectrophoresisForce:
    _require_dielectrophoresis_parameters(enabled, model, parameters, path)
    frequency = _finite(
        parameters.get("frequency_Hz", 0.0),
        f"{path}.parameters.frequency_Hz",
        minimum=0.0,
    )
    if enabled and model == "ac_clausius_mossotti" and frequency <= 0.0:
        raise _error(f"{path}.parameters.frequency_Hz", "must be > 0 for AC")
    electric_field_amplitude = _model(
        parameters.get("electric_field_amplitude", "rms"),
        DEP_ELECTRIC_FIELD_AMPLITUDES,
        f"{path}.parameters.electric_field_amplitude",
    )
    return DielectrophoresisForce(
        enabled=enabled,
        model=model,
        medium_rel_permittivity=_finite(
            parameters.get(
                "medium_rel_permittivity",
                DEP_MEDIUM_REL_PERMITTIVITY_DEFAULT,
            ),
            f"{path}.parameters.medium_rel_permittivity",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        particle_rel_permittivity=(
            None
            if "particle_rel_permittivity" not in parameters
            else _finite(
                parameters["particle_rel_permittivity"],
                f"{path}.parameters.particle_rel_permittivity",
                minimum=0.0,
                exclusive_minimum=True,
            )
        ),
        medium_conductivity_Sm=_finite(
            parameters.get("medium_conductivity_Sm", 0.0),
            f"{path}.parameters.medium_conductivity_Sm",
            minimum=0.0,
        ),
        particle_conductivity_Sm=_finite(
            parameters.get("particle_conductivity_Sm", 0.0),
            f"{path}.parameters.particle_conductivity_Sm",
            minimum=0.0,
        ),
        frequency_Hz=frequency,
        electric_field_amplitude=electric_field_amplitude,
    )


def _parse_coefficient(
    parameters: Mapping[str, Any],
    path: str,
    default: float,
) -> float:
    return _finite(
        parameters.get("coefficient", default),
        f"{path}.parameters.coefficient",
        minimum=0.0,
    )


def _build_optional_force(
    name: str,
    enabled: bool,
    model: str,
    parameters: Mapping[str, Any],
    *,
    path: str,
    spatial_dim: int,
) -> SemanticForce:
    if name == "electric":
        return ElectricForce(enabled=enabled, model=model)
    if name == "gravity":
        return _parse_gravity_force(
            enabled,
            model,
            parameters,
            path=path,
            spatial_dim=spatial_dim,
        )
    if name == "thermophoresis":
        return _parse_thermophoresis_force(
            enabled,
            model,
            parameters,
            path=path,
        )
    if name == "dielectrophoresis":
        return _parse_dielectrophoresis_force(
            enabled,
            model,
            parameters,
            path=path,
        )
    if name == "lift":
        return LiftForce(
            enabled=enabled,
            model=model,
            coefficient=_parse_coefficient(
                parameters,
                path,
                LIFT_COEFFICIENT_DEFAULT,
            ),
        )
    if name == "pressure_gradient":
        return PressureGradientForce(enabled=enabled, model=model)
    if name == "virtual_mass":
        return VirtualMassForce(
            enabled=enabled,
            model=model,
            coefficient=_parse_coefficient(
                parameters,
                path,
                VIRTUAL_MASS_COEFFICIENT_DEFAULT,
            ),
        )
    raise AssertionError(f"unhandled force name {name!r}")


def _parse_optional_force(
    name: str,
    value: Any,
    *,
    path: str,
    spatial_dim: int,
) -> SemanticForce:
    data = _mapping(value, path)
    _reject_unknown(data, {"enabled", "model", "parameters"}, path)
    enabled = _boolean(_required(data, "enabled", path), f"{path}.enabled")
    model, parameters = _optional_model_and_parameters(name, data, path)
    return _build_optional_force(
        name,
        enabled,
        model,
        parameters,
        path=path,
        spatial_dim=spatial_dim,
    )
