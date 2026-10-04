"""Numerical OML and electron-temperature charging equilibria."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._charge_model_types import (
    AMU_KG,
    E_CHARGE_C,
    ELECTRON_MASS_KG,
    EPS0_F_M,
    ChargeModelConfig,
)


def te_relaxation_equilibrium(
    config: ChargeModelConfig,
    radius_m: np.ndarray,
    te_eV: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius, te = np.broadcast_arrays(
        np.asarray(radius_m, dtype=np.float64),
        np.asarray(te_eV, dtype=np.float64),
    )
    for name, values in (("particle radius", radius), ("electron temperature", te)):
        invalid = ~np.isfinite(values) | (values <= 0.0)
        if np.any(invalid):
            rows = np.flatnonzero(invalid)[:12].tolist()
            raise ValueError(
                f"te_relaxation {name} must be finite and positive; invalid rows {rows}"
            )
    alpha = float(config.te_relaxation_alpha)
    relaxation_time = float(config.relaxation_time_s)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("te_relaxation alpha must be finite and positive")
    if not np.isfinite(relaxation_time) or relaxation_time <= 0.0:
        raise ValueError("te_relaxation relaxation time must be finite and positive")
    capacitance = 4.0 * np.pi * EPS0_F_M * radius
    phi = -alpha * te
    q_eq = capacitance * phi
    tau_q = np.full(q_eq.shape, relaxation_time, dtype=np.float64)
    return q_eq, tau_q, phi


@dataclass(frozen=True, slots=True)
class OmlPlasmaBatch:
    radius_m: np.ndarray
    electron_temperature_eV: np.ndarray
    electron_density_m3: np.ndarray
    ion_density_m3: np.ndarray
    ion_temperature_eV: np.ndarray
    capacitance_F: np.ndarray
    collection_area_m2: np.ndarray


@dataclass(frozen=True, slots=True)
class OmlCollectionModel:
    ion_charge_number: float
    electron_sticking: float
    ion_sticking: float
    electron_speed_mps: np.ndarray
    ion_speed_mps: np.ndarray
    max_abs_potential_V: float


def invalid_rows(mask: np.ndarray) -> list[int]:
    return np.flatnonzero(mask)[:12].tolist()


def prepare_oml_plasma_batch(
    radius_m: np.ndarray,
    te_eV: np.ndarray,
    ne_m3: np.ndarray,
    ni_m3: np.ndarray,
    ti_eV: np.ndarray,
) -> OmlPlasmaBatch:
    try:
        radius, te, ne, ni, ti = np.broadcast_arrays(
            np.asarray(radius_m, dtype=np.float64),
            np.asarray(te_eV, dtype=np.float64),
            np.asarray(ne_m3, dtype=np.float64),
            np.asarray(ni_m3, dtype=np.float64),
            np.asarray(ti_eV, dtype=np.float64),
        )
    except ValueError as exc:
        raise ValueError(
            "OML particle and plasma arrays must be broadcast-compatible"
        ) from exc
    for name, values in (
        ("particle radius", radius),
        ("electron temperature", te),
        ("ion temperature", ti),
        ("electron density", ne),
        ("ion density", ni),
    ):
        invalid = ~np.isfinite(values) | (values <= 0.0)
        if np.any(invalid):
            raise ValueError(
                f"OML {name} must be finite and positive; "
                f"invalid rows {invalid_rows(invalid)}"
            )
    return OmlPlasmaBatch(
        radius_m=radius,
        electron_temperature_eV=te,
        electron_density_m3=ne,
        ion_density_m3=ni,
        ion_temperature_eV=ti,
        capacitance_F=4.0 * np.pi * EPS0_F_M * radius,
        collection_area_m2=4.0 * np.pi * radius * radius,
    )


def positive_oml_scalar(value: float, error_message: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(error_message)
    return resolved


def oml_sticking_coefficients(config: ChargeModelConfig) -> tuple[float, float]:
    electron = float(config.electron_sticking)
    ion = float(config.ion_sticking)
    if not np.isfinite(electron) or electron < 0.0 or not np.isfinite(ion) or ion < 0.0:
        raise ValueError("OML sticking coefficients must be finite and non-negative")
    return electron, ion


def select_oml_ion_speed(
    velocity_model: str,
    thermal_speed_mps: np.ndarray,
    bohm_speed_mps: np.ndarray,
) -> np.ndarray:
    if velocity_model == "thermal":
        return thermal_speed_mps
    if velocity_model == "bohm":
        return bohm_speed_mps
    if velocity_model == "max_thermal_bohm":
        return np.maximum(thermal_speed_mps, bohm_speed_mps)
    raise ValueError(
        "OML ion velocity model must be 'thermal', 'bohm', or 'max_thermal_bohm'"
    )


def prepare_oml_collection_model(
    config: ChargeModelConfig,
    plasma: OmlPlasmaBatch,
    *,
    ion_mass_amu: float | None,
    ion_charge_number: float | None,
) -> OmlCollectionModel:
    mass_amu = positive_oml_scalar(
        config.ion_mass_amu if ion_mass_amu is None else ion_mass_amu,
        "OML ion mass must be finite and positive",
    )
    charge_number = positive_oml_scalar(
        config.ion_charge_number if ion_charge_number is None else ion_charge_number,
        "OML ion charge number must be finite and positive",
    )
    electron_sticking, ion_sticking = oml_sticking_coefficients(config)
    ion_mass_kg = mass_amu * AMU_KG
    electron_speed = np.sqrt(
        E_CHARGE_C * plasma.electron_temperature_eV / (2.0 * np.pi * ELECTRON_MASS_KG)
    )
    thermal_speed = np.sqrt(
        E_CHARGE_C * plasma.ion_temperature_eV / (2.0 * np.pi * ion_mass_kg)
    )
    bohm_factor = positive_oml_scalar(
        config.bohm_velocity_factor,
        "OML Bohm velocity factor must be finite and positive",
    )
    bohm_speed = bohm_factor * np.sqrt(
        charge_number * E_CHARGE_C * plasma.electron_temperature_eV / ion_mass_kg
    )
    ion_speed = select_oml_ion_speed(
        str(config.ion_velocity_model), thermal_speed, bohm_speed
    )
    max_abs_potential = positive_oml_scalar(
        config.max_abs_potential_V,
        "OML maximum potential magnitude must be finite and positive",
    )
    return OmlCollectionModel(
        ion_charge_number=charge_number,
        electron_sticking=electron_sticking,
        ion_sticking=ion_sticking,
        electron_speed_mps=electron_speed,
        ion_speed_mps=ion_speed,
        max_abs_potential_V=max_abs_potential,
    )


def oml_fluxes(
    plasma: OmlPlasmaBatch,
    model: OmlCollectionModel,
    potential_V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    exp_arg = np.clip(
        potential_V / plasma.electron_temperature_eV,
        -80.0,
        40.0,
    )
    electron_flux = (
        model.electron_sticking
        * plasma.electron_density_m3
        * model.electron_speed_mps
        * np.exp(exp_arg)
    )
    ion_factor = 1.0 - model.ion_charge_number * potential_V / plasma.ion_temperature_eV
    ion_flux = (
        model.ion_sticking
        * model.ion_charge_number
        * plasma.ion_density_m3
        * model.ion_speed_mps
        * ion_factor
    )
    return electron_flux, ion_flux


def oml_residual(
    plasma: OmlPlasmaBatch,
    model: OmlCollectionModel,
    potential_V: np.ndarray,
) -> np.ndarray:
    electron_flux, ion_flux = oml_fluxes(plasma, model, potential_V)
    return ion_flux - electron_flux


def bracket_oml_root(
    plasma: OmlPlasmaBatch,
    model: OmlCollectionModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.full_like(
        plasma.electron_temperature_eV,
        -model.max_abs_potential_V,
        dtype=np.float64,
    )
    upper = np.zeros_like(plasma.electron_temperature_eV, dtype=np.float64)
    lower_residual = oml_residual(plasma, model, lower)
    upper_residual = oml_residual(plasma, model, upper)
    bracketed = (
        np.isfinite(lower_residual)
        & np.isfinite(upper_residual)
        & (
            (lower_residual == 0.0)
            | (upper_residual == 0.0)
            | (np.signbit(lower_residual) != np.signbit(upper_residual))
        )
    )
    if not np.all(bracketed):
        raise ValueError(
            "OML current balance root is not bracketed; "
            f"invalid rows {invalid_rows(~bracketed)}"
        )
    return lower, upper, lower_residual


def validated_oml_root_iterations(config: ChargeModelConfig) -> int:
    if type(config.root_iterations) is not int or config.root_iterations < 1:
        raise ValueError("OML root_iterations must be an integer >= 1")
    return config.root_iterations


def bisect_oml_root(
    plasma: OmlPlasmaBatch,
    model: OmlCollectionModel,
    lower: np.ndarray,
    upper: np.ndarray,
    lower_residual: np.ndarray,
    iterations: int,
) -> np.ndarray:
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_residual = oml_residual(plasma, model, midpoint)
        root_in_lower_half = (
            (lower_residual == 0.0)
            | (midpoint_residual == 0.0)
            | (np.signbit(lower_residual) != np.signbit(midpoint_residual))
        )
        upper = np.where(root_in_lower_half, midpoint, upper)
        lower = np.where(~root_in_lower_half, midpoint, lower)
        lower_residual = np.where(
            ~root_in_lower_half,
            midpoint_residual,
            lower_residual,
        )
    return 0.5 * (lower + upper)


def solve_oml_potential(
    config: ChargeModelConfig,
    plasma: OmlPlasmaBatch,
    model: OmlCollectionModel,
) -> np.ndarray:
    lower, upper, lower_residual = bracket_oml_root(plasma, model)
    return bisect_oml_root(
        plasma,
        model,
        lower,
        upper,
        lower_residual,
        validated_oml_root_iterations(config),
    )


def validate_oml_residual(
    electron_flux: np.ndarray,
    ion_flux: np.ndarray,
) -> None:
    residual = ion_flux - electron_flux
    ion_magnitude = np.abs(ion_flux)
    electron_magnitude = np.abs(electron_flux)
    flux_scale = np.maximum(ion_magnitude, electron_magnitude)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scaled_sum = ion_magnitude / flux_scale + electron_magnitude / flux_scale
        normalized_residual = (np.abs(residual) / flux_scale) / scaled_sum
    normalized_residual = np.where(flux_scale > 0.0, normalized_residual, np.inf)
    residual_tolerance = 1.0e-10
    residual_ok = np.isfinite(normalized_residual) & (
        normalized_residual <= residual_tolerance
    )
    if not np.all(residual_ok):
        worst = float(np.max(normalized_residual[~residual_ok]))
        raise ValueError(
            "OML current balance residual did not converge "
            f"(tolerance={residual_tolerance:g}, worst={worst:g}, "
            f"invalid rows {invalid_rows(~residual_ok)})"
        )


def oml_linearized_relaxation_time(
    plasma: OmlPlasmaBatch,
    model: OmlCollectionModel,
    electron_flux: np.ndarray,
) -> np.ndarray:
    derivative = (
        model.ion_sticking
        * model.ion_charge_number
        * plasma.ion_density_m3
        * model.ion_speed_mps
        * (-model.ion_charge_number / plasma.ion_temperature_eV)
        - electron_flux / plasma.electron_temperature_eV
    )
    tau_q = -plasma.capacitance_F / (
        E_CHARGE_C * plasma.collection_area_m2 * derivative
    )
    invalid = ~np.isfinite(tau_q) | (tau_q <= 0.0)
    if np.any(invalid):
        raise ValueError(
            f"OML linearized relaxation time is invalid at rows {invalid_rows(invalid)}"
        )
    return tau_q


def oml_linearized_equilibrium(
    config: ChargeModelConfig,
    radius_m: np.ndarray,
    te_eV: np.ndarray,
    ne_m3: np.ndarray,
    ni_m3: np.ndarray,
    ti_eV: np.ndarray,
    *,
    ion_mass_amu: float | None = None,
    ion_charge_number: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plasma = prepare_oml_plasma_batch(radius_m, te_eV, ne_m3, ni_m3, ti_eV)
    model = prepare_oml_collection_model(
        config,
        plasma,
        ion_mass_amu=ion_mass_amu,
        ion_charge_number=ion_charge_number,
    )
    potential = solve_oml_potential(config, plasma, model)
    electron_flux, ion_flux = oml_fluxes(plasma, model, potential)
    validate_oml_residual(electron_flux, ion_flux)
    tau_q = oml_linearized_relaxation_time(plasma, model, electron_flux)
    return plasma.capacitance_F * potential, tau_q, potential


__all__ = ("oml_linearized_equilibrium", "te_relaxation_equilibrium")
