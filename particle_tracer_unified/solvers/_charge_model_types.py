"""Particle-charging constants, configuration, and diagnostic value types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plasma_background import PreparedPlasmaBackground

E_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
EPS0_F_M = 8.8541878128e-12
AMU_KG = 1.66053906660e-27


@dataclass(frozen=True, slots=True)
class ChargeModelConfig:
    enabled: bool = False
    mode: str = "te_relaxation"
    background_source: str = "field"
    electron_temperature_quantity: str = ""
    electron_temperature_unit: str = "eV"
    electron_density_quantity: str = ""
    ion_density_quantity: str = ""
    ion_temperature_quantity: str = ""
    ion_temperature_eV: float = 0.03
    ion_mass_amu: float = 69.0
    ion_charge_number: float = 1.0
    electron_sticking: float = 1.0
    ion_sticking: float = 1.0
    ion_velocity_model: str = "max_thermal_bohm"
    bohm_velocity_factor: float = 1.0
    te_relaxation_alpha: float = 2.5
    relaxation_time_s: float = 1.0e-6
    max_abs_potential_V: float = 100.0
    root_iterations: int = 64


def charge_model_report(
    config: ChargeModelConfig,
    plasma_background: PreparedPlasmaBackground | None = None,
) -> dict[str, object]:
    return {
        "enabled": int(bool(config.enabled)),
        "mode": str(config.mode),
        "background_source": str(config.background_source),
        "plasma_background_source": (
            "none" if plasma_background is None else str(plasma_background.source)
        ),
        "ion_velocity_model": str(config.ion_velocity_model),
        "operator_statistics_scope": "latest_global_half_step_evaluation",
        "update_event_count": 0,
        "updated_particle_count": 0,
        "last_updated_particle_count": 0,
        "last_mean_Te_eV": 0.0,
        "last_mean_ne_m3": 0.0,
        "last_mean_ni_m3": 0.0,
        "last_mean_Ti_eV": 0.0,
        "last_mean_debye_length_m": 0.0,
        "last_mean_particle_radius_over_debye": 0.0,
        "last_mean_floating_potential_V": 0.0,
        "last_mean_equilibrium_charge_e": 0.0,
        "last_mean_tau_q_s": 0.0,
        "last_charge_response_regime": "disabled",
        "last_mean_charge_C": 0.0,
        "last_mean_charge_e": 0.0,
        "terminal_hit_replay_count": 0,
        "terminal_hit_replay_age_total_s": 0.0,
        "terminal_hit_replay_age_max_s": 0.0,
        "final_min_charge_C": 0.0,
        "final_mean_charge_C": 0.0,
        "final_max_charge_C": 0.0,
        "final_min_charge_e": 0.0,
        "final_mean_charge_e": 0.0,
        "final_max_charge_e": 0.0,
    }


def charge_response_regime(delta_t_s: float, tau_q: np.ndarray) -> str:
    finite = np.asarray(tau_q, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return "unknown"
    ratio = float(delta_t_s) / float(np.median(finite))
    if ratio < 0.2:
        return "explicit_transient"
    if ratio < 2.0:
        return "partially_relaxed"
    return "quasi_equilibrium"


__all__ = (
    "AMU_KG",
    "ELECTRON_MASS_KG",
    "EPS0_F_M",
    "E_CHARGE_C",
    "ChargeModelConfig",
    "charge_model_report",
    "charge_response_regime",
)
