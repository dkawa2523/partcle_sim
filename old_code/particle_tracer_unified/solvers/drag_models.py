"""Gas-drag laws, rarefaction corrections, and legacy model mapping.

COMSOL treats the continuum drag law and the rarefaction correction as
independent choices.  The public case format predates that separation, so this
module maps each legacy model name to one explicit pair without changing the
input schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .drag_regime import (
    BOLTZMANN_J_K,
    SCHILLER_NAUMANN_RE_OUTSIDE,
    gas_mean_free_path_scalar_m,
)

DRAG_MODEL_STOKES = 0
DRAG_MODEL_SCHILLER_NAUMANN = 1
DRAG_MODEL_EPSTEIN = 2
DRAG_MODEL_STOKES_CUNNINGHAM = 3
DRAG_MODEL_NONE = 4

CONTINUUM_DRAG_NONE = -1
CONTINUUM_DRAG_STOKES = 0
CONTINUUM_DRAG_SCHILLER_NAUMANN = 1

RAREFACTION_NONE = 0
RAREFACTION_CUNNINGHAM = 1
RAREFACTION_EPSTEIN = 2

_EPSTEIN_DEFAULT_ACCOMMODATION_DELTA = 1.0 + np.pi / 8.0


@dataclass(frozen=True, slots=True)
class DragModelStructure:
    """Resolved internal factors for one legacy drag model name."""

    name: str
    mode: int
    continuum_law: str
    rarefaction_correction: str
    continuum_mode: int
    rarefaction_mode: int
    gas_requirements: tuple[str, ...]
    stage_gas_requirements: tuple[str, ...]

    @property
    def mode_pair(self) -> tuple[int, int]:
        return int(self.continuum_mode), int(self.rarefaction_mode)


_STRUCTURES = (
    DragModelStructure(
        "stokes",
        DRAG_MODEL_STOKES,
        "stokes",
        "none",
        CONTINUUM_DRAG_STOKES,
        RAREFACTION_NONE,
        ("dynamic_viscosity_Pas",),
        ("dynamic_viscosity_Pas",),
    ),
    DragModelStructure(
        "schiller_naumann",
        DRAG_MODEL_SCHILLER_NAUMANN,
        "schiller_naumann",
        "none",
        CONTINUUM_DRAG_SCHILLER_NAUMANN,
        RAREFACTION_NONE,
        ("dynamic_viscosity_Pas", "density_kgm3"),
        ("dynamic_viscosity_Pas", "density_kgm3"),
    ),
    DragModelStructure(
        "epstein",
        DRAG_MODEL_EPSTEIN,
        "stokes",
        "epstein",
        CONTINUUM_DRAG_STOKES,
        RAREFACTION_EPSTEIN,
        ("temperature_K", "density_kgm3", "molecular_mass_amu"),
        ("temperature_K", "density_kgm3"),
    ),
    DragModelStructure(
        "stokes_cunningham",
        DRAG_MODEL_STOKES_CUNNINGHAM,
        "stokes",
        "cunningham",
        CONTINUUM_DRAG_STOKES,
        RAREFACTION_CUNNINGHAM,
        (
            "temperature_K",
            "dynamic_viscosity_Pas",
            "density_kgm3",
            "molecular_mass_amu",
        ),
        ("temperature_K", "dynamic_viscosity_Pas", "density_kgm3"),
    ),
    DragModelStructure(
        "none",
        DRAG_MODEL_NONE,
        "none",
        "none",
        CONTINUUM_DRAG_NONE,
        RAREFACTION_NONE,
        (),
        (),
    ),
)
_STRUCTURE_BY_NAME = {item.name: item for item in _STRUCTURES}
_STRUCTURE_BY_MODE = {item.mode: item for item in _STRUCTURES}


def drag_model_structure_from_name(name: object) -> DragModelStructure:
    value = str(name).strip().lower()
    try:
        return _STRUCTURE_BY_NAME[value]
    except KeyError as exc:
        raise ValueError(
            "solver.drag_model must be 'none', 'stokes', 'stokes_cunningham', "
            "'schiller_naumann', or 'epstein'"
        ) from exc


def drag_model_structure_from_mode(mode: int) -> DragModelStructure:
    try:
        return _STRUCTURE_BY_MODE[int(mode)]
    except KeyError as exc:
        raise ValueError(f"unknown drag model mode {mode!r}") from exc


def drag_model_mode_from_name(name: object) -> int:
    return int(drag_model_structure_from_name(name).mode)


def drag_model_name_from_mode(mode: int) -> str:
    return str(drag_model_structure_from_mode(mode).name)


def drag_model_gas_requirements(name: object) -> tuple[str, ...]:
    return drag_model_structure_from_name(name).gas_requirements


def drag_model_stage_gas_requirements(name: object) -> tuple[str, ...]:
    """Return gas fields sampled while integrating the resolved legacy model."""

    return drag_model_structure_from_name(name).stage_gas_requirements


def _required_positive_finite(raw_value: float, error_message: str) -> float:
    value = float(raw_value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(error_message)
    return value


def stokes_relaxation_time(
    mass_kg: float, gas_mu_pas: float, particle_diameter_m: float
) -> float:
    """Return the Stokes relaxation time from authoritative particle mass."""

    mass = _required_positive_finite(
        mass_kg,
        "particle mass_kg must be finite and > 0",
    )
    viscosity = _required_positive_finite(
        gas_mu_pas,
        "gas dynamic viscosity must be finite and > 0",
    )
    diameter = _required_positive_finite(
        particle_diameter_m,
        "particle drag_diameter_m must be finite and > 0",
    )
    return mass / (3.0 * np.pi * viscosity * diameter)


def epstein_relaxation_time(
    mass_kg: float,
    gas_density_kgm3: float,
    gas_temperature_K: float,
    particle_diameter_m: float,
    gas_molecular_mass_kg: float,
) -> float:
    """Return Stokes drag with COMSOL's high-Kn Epstein slip correction.

    The algebraically reduced expression is independent of viscosity.  It is
    retained directly so the legacy ``epstein`` input keeps the same required
    gas properties and numerical result.
    """

    mass = _required_positive_finite(
        mass_kg,
        "particle mass_kg must be finite and > 0",
    )
    gas_density = _required_positive_finite(
        gas_density_kgm3,
        "Epstein drag requires gas density_kgm3 > 0",
    )
    temperature = _required_positive_finite(
        gas_temperature_K,
        "Epstein drag requires gas temperature_K > 0",
    )
    diameter = _required_positive_finite(
        particle_diameter_m,
        "particle drag_diameter_m must be finite and > 0",
    )
    molecular_mass = _required_positive_finite(
        gas_molecular_mass_kg,
        "Epstein drag requires gas molecular_mass_kg > 0",
    )
    thermal_speed = np.sqrt(
        8.0 * BOLTZMANN_J_K * temperature / (np.pi * molecular_mass)
    )
    return (
        3.0
        * mass
        / (
            _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA
            * np.pi
            * diameter
            * diameter
            * gas_density
            * thermal_speed
        )
    )


@njit(cache=True)
def schiller_naumann_drag_correction(reynolds):
    re = max(0.0, reynolds)
    if re <= 1.0e-12:
        return 1.0
    if re >= SCHILLER_NAUMANN_RE_OUTSIDE:
        raise ValueError(
            "schiller_naumann drag requires particle Reynolds number < 800"
        )
    return 1.0 + 0.15 * re**0.687


@njit(cache=True)
def cunningham_slip_correction(knudsen_number):
    kn = max(float(knudsen_number), 0.0)
    if kn <= 0.0:
        return 1.0
    return 1.0 + kn * (2.514 + 0.8 * np.exp(-0.55 / max(kn, 1.0e-300)))


@njit(cache=True, inline="always")
def _is_positive_finite(value):
    scalar = float(value)
    return np.isfinite(scalar) and scalar > 0.0


@njit(cache=True, inline="always")
def _epstein_effective_tau(
    particle_diameter_m,
    gas_density_kgm3,
    particle_mass_kg,
    gas_temperature_K,
    gas_molecular_mass_kg,
    accommodation_delta,
):
    diameter = float(particle_diameter_m)
    rho_g = float(gas_density_kgm3)
    temp = float(gas_temperature_K)
    mol_mass = float(gas_molecular_mass_kg)
    particle_mass = float(particle_mass_kg)
    if (
        not _is_positive_finite(particle_mass)
        or not _is_positive_finite(diameter)
        or not _is_positive_finite(rho_g)
        or not _is_positive_finite(temp)
        or not _is_positive_finite(mol_mass)
    ):
        return np.nan
    thermal_speed = np.sqrt(8.0 * BOLTZMANN_J_K * temp / (np.pi * mol_mass))
    return (
        3.0
        * particle_mass
        / (accommodation_delta * np.pi * diameter * diameter * rho_g * thermal_speed)
    )


@njit(cache=True, inline="always")
def _cunningham_effective_tau(
    tau_stokes,
    particle_diameter_m,
    gas_density_kgm3,
    gas_mu_pas,
    gas_temperature_K,
    gas_molecular_mass_kg,
):
    diameter = float(particle_diameter_m)
    rho_g = float(gas_density_kgm3)
    mu = float(gas_mu_pas)
    temp = float(gas_temperature_K)
    mol_mass = float(gas_molecular_mass_kg)
    if (
        not _is_positive_finite(tau_stokes)
        or not _is_positive_finite(diameter)
        or not _is_positive_finite(rho_g)
        or not _is_positive_finite(mu)
        or not _is_positive_finite(temp)
        or not _is_positive_finite(mol_mass)
    ):
        return np.nan
    mean_free_path = gas_mean_free_path_scalar_m(mu, rho_g, temp, mol_mass)
    return tau_stokes * cunningham_slip_correction(mean_free_path / diameter)


@njit(cache=True, inline="always")
def _stage_stokes_relaxation_time(
    tau_stokes_reference,
    particle_diameter_m,
    gas_mu_pas,
    particle_mass_kg,
):
    reference = float(tau_stokes_reference)
    diameter = float(particle_diameter_m)
    viscosity = float(gas_mu_pas)
    mass = float(particle_mass_kg)
    if (
        not _is_positive_finite(reference)
        or not _is_positive_finite(diameter)
        or not _is_positive_finite(viscosity)
        or not _is_positive_finite(mass)
    ):
        return np.nan
    return mass / (3.0 * np.pi * viscosity * diameter)


@njit(cache=True, inline="always")
def _continuum_drag_force_multiplier(
    continuum_drag_mode,
    slip_speed,
    particle_diameter_m,
    gas_density_kgm3,
    gas_mu_pas,
):
    mode = int(continuum_drag_mode)
    if mode == CONTINUUM_DRAG_STOKES:
        return 1.0
    if mode != CONTINUUM_DRAG_SCHILLER_NAUMANN:
        return np.nan
    diameter = float(particle_diameter_m)
    rho_g = float(gas_density_kgm3)
    mu = float(gas_mu_pas)
    if (
        not _is_positive_finite(diameter)
        or not _is_positive_finite(rho_g)
        or not _is_positive_finite(mu)
    ):
        return np.nan
    reynolds = rho_g * diameter * max(float(slip_speed), 0.0) / mu
    return schiller_naumann_drag_correction(reynolds)


@njit(cache=True, inline="always")
def _rarefied_stokes_relaxation_time(
    tau_stokes,
    rarefaction_mode,
    particle_diameter_m,
    gas_density_kgm3,
    gas_mu_pas,
    particle_mass_kg,
    gas_temperature_K,
    gas_molecular_mass_kg,
    epstein_accommodation_delta,
):
    mode = int(rarefaction_mode)
    if mode == RAREFACTION_EPSTEIN:
        return _epstein_effective_tau(
            particle_diameter_m,
            gas_density_kgm3,
            particle_mass_kg,
            gas_temperature_K,
            gas_molecular_mass_kg,
            epstein_accommodation_delta,
        )
    tau_stage = _stage_stokes_relaxation_time(
        tau_stokes,
        particle_diameter_m,
        gas_mu_pas,
        particle_mass_kg,
    )
    if mode == RAREFACTION_CUNNINGHAM:
        return _cunningham_effective_tau(
            tau_stage,
            particle_diameter_m,
            gas_density_kgm3,
            gas_mu_pas,
            gas_temperature_K,
            gas_molecular_mass_kg,
        )
    if mode == RAREFACTION_NONE:
        return tau_stage
    return np.nan


@njit(cache=True)
def effective_tau_from_drag_components(
    tau_stokes,
    slip_speed,
    particle_diameter_m,
    gas_density_kgm3,
    gas_mu_pas,
    continuum_drag_mode,
    rarefaction_mode,
    particle_mass_kg,
    gas_temperature_K,
    gas_molecular_mass_kg,
    epstein_accommodation_delta,
):
    """Compose an independent continuum law and rarefaction correction."""

    rarefied_stokes_tau = _rarefied_stokes_relaxation_time(
        tau_stokes,
        rarefaction_mode,
        particle_diameter_m,
        gas_density_kgm3,
        gas_mu_pas,
        particle_mass_kg,
        gas_temperature_K,
        gas_molecular_mass_kg,
        epstein_accommodation_delta,
    )
    continuum_multiplier = _continuum_drag_force_multiplier(
        continuum_drag_mode,
        slip_speed,
        particle_diameter_m,
        gas_density_kgm3,
        gas_mu_pas,
    )
    if not _is_positive_finite(rarefied_stokes_tau) or not _is_positive_finite(
        continuum_multiplier
    ):
        return np.nan
    return rarefied_stokes_tau / continuum_multiplier


@njit(cache=True, inline="always")
def _drag_model_component_modes(drag_model_mode):
    mode = int(drag_model_mode)
    if mode == DRAG_MODEL_STOKES:
        return CONTINUUM_DRAG_STOKES, RAREFACTION_NONE
    if mode == DRAG_MODEL_STOKES_CUNNINGHAM:
        return CONTINUUM_DRAG_STOKES, RAREFACTION_CUNNINGHAM
    if mode == DRAG_MODEL_SCHILLER_NAUMANN:
        return CONTINUUM_DRAG_SCHILLER_NAUMANN, RAREFACTION_NONE
    if mode == DRAG_MODEL_EPSTEIN:
        return CONTINUUM_DRAG_STOKES, RAREFACTION_EPSTEIN
    return CONTINUUM_DRAG_NONE, RAREFACTION_NONE


@njit(cache=True)
def effective_tau_from_drag_model(
    tau_stokes,
    slip_speed,
    particle_diameter_m,
    gas_density_kgm3,
    gas_mu_pas,
    drag_model_mode,
    particle_mass_kg,
    gas_temperature_K,
    gas_molecular_mass_kg,
    epstein_accommodation_delta,
):
    mode = int(drag_model_mode)
    if mode == DRAG_MODEL_NONE:
        return np.inf
    continuum_mode, rarefaction_mode = _drag_model_component_modes(mode)
    if continuum_mode == CONTINUUM_DRAG_NONE:
        return np.nan
    return effective_tau_from_drag_components(
        tau_stokes,
        slip_speed,
        particle_diameter_m,
        gas_density_kgm3,
        gas_mu_pas,
        continuum_mode,
        rarefaction_mode,
        particle_mass_kg,
        gas_temperature_K,
        gas_molecular_mass_kg,
        epstein_accommodation_delta,
    )


# Compatibility name for existing internal imports.  Runtime callers use the
# canonical name above so a physics-semantic change also invalidates their
# compiled call sites.
effective_tau_from_slip_speed = effective_tau_from_drag_model


__all__ = (
    "CONTINUUM_DRAG_NONE",
    "CONTINUUM_DRAG_SCHILLER_NAUMANN",
    "CONTINUUM_DRAG_STOKES",
    "DRAG_MODEL_EPSTEIN",
    "DRAG_MODEL_NONE",
    "DRAG_MODEL_SCHILLER_NAUMANN",
    "DRAG_MODEL_STOKES",
    "DRAG_MODEL_STOKES_CUNNINGHAM",
    "RAREFACTION_CUNNINGHAM",
    "RAREFACTION_EPSTEIN",
    "RAREFACTION_NONE",
    "DragModelStructure",
    "cunningham_slip_correction",
    "drag_model_gas_requirements",
    "drag_model_mode_from_name",
    "drag_model_name_from_mode",
    "drag_model_stage_gas_requirements",
    "drag_model_structure_from_mode",
    "drag_model_structure_from_name",
    "effective_tau_from_drag_components",
    "effective_tau_from_drag_model",
    "effective_tau_from_slip_speed",
    "epstein_relaxation_time",
    "schiller_naumann_drag_correction",
    "stokes_relaxation_time",
)
