"""Dimensionless applicability checks for the supported gas-drag laws.

The functions in this module do not select or modify a drag law.  They only
evaluate the law named by the case, so a validation check can never change the
physical model as a side effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

BOLTZMANN_J_K = 1.380649e-23

# These are validation boundaries, not user-tunable numerical parameters.
# The hard Schiller--Naumann upper bound is the published correlation range.
# The remaining boundaries distinguish asymptotic, transition, and clearly
# incompatible regimes without pretending that an exact universal crossover
# exists.
STOKES_RE_TRANSITION = 0.1
STOKES_RE_OUTSIDE = 1.0
SCHILLER_NAUMANN_RE_OUTSIDE = 800.0
CONTINUUM_KN_TRANSITION = 0.01
CONTINUUM_KN_OUTSIDE = 0.1
FREE_MOLECULAR_KN_TRANSITION = 10.0
FREE_MOLECULAR_KN_OUTSIDE = 1.0
COMPRESSIBILITY_MACH_TRANSITION = 0.3
COMPRESSIBILITY_MACH_OUTSIDE = 1.0


@dataclass(frozen=True)
class DragRegimeDecision:
    """One particle's model-range classification at one sampled state."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def particle_reynolds_number(
    slip_speed_mps: np.ndarray,
    particle_diameter_m: np.ndarray,
    gas_density_kgm3: np.ndarray,
    gas_dynamic_viscosity_Pas: np.ndarray,
) -> np.ndarray:
    speed, diameter, density, viscosity = np.broadcast_arrays(
        np.asarray(slip_speed_mps, dtype=np.float64),
        np.asarray(particle_diameter_m, dtype=np.float64),
        np.asarray(gas_density_kgm3, dtype=np.float64),
        np.asarray(gas_dynamic_viscosity_Pas, dtype=np.float64),
    )
    valid = (
        np.isfinite(speed)
        & (speed >= 0.0)
        & np.isfinite(diameter)
        & (diameter > 0.0)
        & np.isfinite(density)
        & (density > 0.0)
        & np.isfinite(viscosity)
        & (viscosity > 0.0)
    )
    result = np.full(speed.shape, np.nan, dtype=np.float64)
    result[valid] = density[valid] * diameter[valid] * speed[valid] / viscosity[valid]
    return result


def gas_mean_free_path_m(
    gas_dynamic_viscosity_Pas: np.ndarray,
    gas_density_kgm3: np.ndarray,
    gas_temperature_K: np.ndarray,
    gas_molecular_mass_kg: np.ndarray,
) -> np.ndarray:
    """Return the same viscosity-derived mean free path used by Cunningham drag."""

    viscosity, density, temperature, molecular_mass = np.broadcast_arrays(
        np.asarray(gas_dynamic_viscosity_Pas, dtype=np.float64),
        np.asarray(gas_density_kgm3, dtype=np.float64),
        np.asarray(gas_temperature_K, dtype=np.float64),
        np.asarray(gas_molecular_mass_kg, dtype=np.float64),
    )
    valid = (
        np.isfinite(viscosity)
        & (viscosity > 0.0)
        & np.isfinite(density)
        & (density > 0.0)
        & np.isfinite(temperature)
        & (temperature > 0.0)
        & np.isfinite(molecular_mass)
        & (molecular_mass > 0.0)
    )
    result = np.full(viscosity.shape, np.nan, dtype=np.float64)
    result[valid] = (viscosity[valid] / density[valid]) * np.sqrt(
        np.pi * molecular_mass[valid] / (2.0 * BOLTZMANN_J_K * temperature[valid])
    )
    return result


@njit(cache=True)
def gas_mean_free_path_scalar_m(
    gas_dynamic_viscosity_Pas: float,
    gas_density_kgm3: float,
    gas_temperature_K: float,
    gas_molecular_mass_kg: float,
) -> float:
    """Numba-compatible scalar form used by the hot drag path."""

    viscosity = float(gas_dynamic_viscosity_Pas)
    density = float(gas_density_kgm3)
    temperature = float(gas_temperature_K)
    molecular_mass = float(gas_molecular_mass_kg)
    if (
        not np.isfinite(viscosity)
        or viscosity <= 0.0
        or not np.isfinite(density)
        or density <= 0.0
        or not np.isfinite(temperature)
        or temperature <= 0.0
        or not np.isfinite(molecular_mass)
        or molecular_mass <= 0.0
    ):
        return np.nan
    return (viscosity / density) * np.sqrt(
        np.pi * molecular_mass / (2.0 * BOLTZMANN_J_K * temperature)
    )


def relative_knudsen_number(
    mean_free_path_m: np.ndarray,
    particle_diameter_m: np.ndarray,
) -> np.ndarray:
    mean_free_path, diameter = np.broadcast_arrays(
        np.asarray(mean_free_path_m, dtype=np.float64),
        np.asarray(particle_diameter_m, dtype=np.float64),
    )
    valid = (
        np.isfinite(mean_free_path)
        & (mean_free_path >= 0.0)
        & np.isfinite(diameter)
        & (diameter > 0.0)
    )
    result = np.full(mean_free_path.shape, np.nan, dtype=np.float64)
    result[valid] = mean_free_path[valid] / diameter[valid]
    return result


def relative_mach_number(
    slip_speed_mps: np.ndarray,
    sound_speed_mps: np.ndarray,
) -> np.ndarray:
    speed, sound_speed = np.broadcast_arrays(
        np.asarray(slip_speed_mps, dtype=np.float64),
        np.asarray(sound_speed_mps, dtype=np.float64),
    )
    valid = (
        np.isfinite(speed)
        & (speed >= 0.0)
        & np.isfinite(sound_speed)
        & (sound_speed > 0.0)
    )
    result = np.full(speed.shape, np.nan, dtype=np.float64)
    result[valid] = speed[valid] / sound_speed[valid]
    return result


def _upper_limit_decision(
    value: float,
    *,
    error_at: float,
    error_code: str,
    warning_at: float | None = None,
    warning_code: str | None = None,
) -> DragRegimeDecision:
    if value >= error_at:
        return DragRegimeDecision(errors=(error_code,))
    if warning_at is not None and warning_code is not None and value >= warning_at:
        return DragRegimeDecision(warnings=(warning_code,))
    return DragRegimeDecision()


def _reynolds_decision(name: str, reynolds: float) -> DragRegimeDecision:
    if not np.isfinite(reynolds):
        return DragRegimeDecision()
    if name in {"stokes", "stokes_cunningham"}:
        return _upper_limit_decision(
            reynolds,
            error_at=STOKES_RE_OUTSIDE,
            error_code="particle_reynolds_outside_creeping_flow",
            warning_at=STOKES_RE_TRANSITION,
            warning_code="particle_reynolds_near_creeping_flow_limit",
        )
    if name == "schiller_naumann":
        return _upper_limit_decision(
            reynolds,
            error_at=SCHILLER_NAUMANN_RE_OUTSIDE,
            error_code="particle_reynolds_outside_schiller_naumann",
        )
    return DragRegimeDecision()


def _knudsen_decision(name: str, knudsen: float) -> DragRegimeDecision:
    if not np.isfinite(knudsen):
        return DragRegimeDecision()
    if name in {"stokes", "schiller_naumann"}:
        return _upper_limit_decision(
            knudsen,
            error_at=CONTINUUM_KN_OUTSIDE,
            error_code="knudsen_outside_unrarefied_continuum",
            warning_at=CONTINUUM_KN_TRANSITION,
            warning_code="knudsen_requires_rarefaction_review",
        )
    if name == "stokes_cunningham" and knudsen >= FREE_MOLECULAR_KN_TRANSITION:
        return DragRegimeDecision(warnings=("knudsen_free_molecular_epstein_review",))
    if name == "epstein":
        if knudsen <= FREE_MOLECULAR_KN_OUTSIDE:
            return DragRegimeDecision(errors=("knudsen_outside_free_molecular_flow",))
        if knudsen < FREE_MOLECULAR_KN_TRANSITION:
            return DragRegimeDecision(
                warnings=("knudsen_transitional_not_asymptotic_free_molecular",)
            )
    return DragRegimeDecision()


def _mach_decision(name: str, relative_mach: float) -> DragRegimeDecision:
    if name == "none" or not np.isfinite(relative_mach):
        return DragRegimeDecision()
    return _upper_limit_decision(
        relative_mach,
        error_at=COMPRESSIBILITY_MACH_OUTSIDE,
        error_code="relative_mach_supersonic_drag_not_supported",
        warning_at=COMPRESSIBILITY_MACH_TRANSITION,
        warning_code="relative_mach_requires_compressibility_review",
    )


def classify_drag_regime(
    model: str,
    *,
    reynolds: float,
    knudsen: float,
    relative_mach: float = float("nan"),
) -> DragRegimeDecision:
    """Classify a declared law without silently switching to another law."""

    name = str(model).strip().lower()
    re = float(reynolds)
    kn = float(knudsen)
    mach = float(relative_mach)
    reynolds_decision = _reynolds_decision(name, re)
    knudsen_decision = _knudsen_decision(name, kn)
    mach_decision = _mach_decision(name, mach)
    return DragRegimeDecision(
        errors=(
            reynolds_decision.errors + knudsen_decision.errors + mach_decision.errors
        ),
        warnings=(
            reynolds_decision.warnings
            + knudsen_decision.warnings
            + mach_decision.warnings
        ),
    )


__all__ = (
    "BOLTZMANN_J_K",
    "COMPRESSIBILITY_MACH_OUTSIDE",
    "COMPRESSIBILITY_MACH_TRANSITION",
    "CONTINUUM_KN_OUTSIDE",
    "CONTINUUM_KN_TRANSITION",
    "FREE_MOLECULAR_KN_OUTSIDE",
    "FREE_MOLECULAR_KN_TRANSITION",
    "SCHILLER_NAUMANN_RE_OUTSIDE",
    "STOKES_RE_OUTSIDE",
    "STOKES_RE_TRANSITION",
    "DragRegimeDecision",
    "classify_drag_regime",
    "gas_mean_free_path_m",
    "gas_mean_free_path_scalar_m",
    "particle_reynolds_number",
    "relative_knudsen_number",
    "relative_mach_number",
)
