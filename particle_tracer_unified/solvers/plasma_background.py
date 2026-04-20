from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np

E_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
EPS0_F_M = 8.8541878128e-12
AMU_KG = 1.66053906660e-27
KB_J_K = 1.380649e-23


@dataclass(frozen=True, slots=True)
class PlasmaBackgroundConfig:
    source: str = 'none'
    electron_density_m3: float = 0.0
    ion_density_m3: float = 0.0
    electron_temperature_eV: float = 0.0
    ion_temperature_eV: float = 0.0
    ion_mass_amu: float = 69.0
    ion_charge_number: float = 1.0
    pressure_Pa: float = 0.0
    gas_temperature_K: float = 0.0
    neutral_molecular_mass_amu: float = 0.0
    electron_neutral_cross_section_m2: float = 0.0
    ion_neutral_cross_section_m2: float = 0.0
    electron_collision_frequency_s: float = 0.0
    ion_collision_frequency_s: float = 0.0
    electron_ion_collision_frequency_s: float = 0.0
    conductivity_Sm: float = 0.0


@dataclass(frozen=True, slots=True)
class PreparedPlasmaBackground:
    source: str
    electron_density_m3: float
    ion_density_m3: float
    electron_temperature_eV: float
    ion_temperature_eV: float
    ion_mass_amu: float
    ion_mass_kg: float
    ion_charge_number: float
    pressure_Pa: float
    gas_temperature_K: float
    neutral_molecular_mass_amu: float
    neutral_molecular_mass_kg: float
    neutral_density_m3: float
    electron_neutral_cross_section_m2: float
    ion_neutral_cross_section_m2: float
    electron_debye_length_m: float
    ion_debye_length_m: float
    debye_length_m: float
    electron_collision_frequency_s: float
    ion_collision_frequency_s: float
    electron_ion_collision_frequency_s: float
    effective_electron_collision_frequency_s: float
    conductivity_Sm: float
    electron_mobility_m2Vs: float
    ion_mobility_m2Vs: float
    electron_thermal_speed_mps: float
    ion_thermal_speed_mps: float
    ion_bohm_speed_mps: float
    electron_plasma_frequency_rad_s: float
    ion_plasma_frequency_rad_s: float
    collision_frequency_source: str
    conductivity_source: str


def _bool_from_config(value: object, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'1', 'true', 'yes', 'on'}:
            return True
        if text in {'0', 'false', 'no', 'off'}:
            return False
    return bool(value)


def _positive_float(raw: Mapping[str, object], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in raw:
            return float(raw.get(name, default))
    return float(default)


def parse_plasma_background_config(solver_cfg: Mapping[str, object]) -> PlasmaBackgroundConfig:
    raw = solver_cfg.get('plasma_background', {}) if isinstance(solver_cfg, Mapping) else {}
    if raw is None:
        raw = {}
    if isinstance(raw, (bool, int, np.integer, str)):
        if not _bool_from_config(raw, default=False):
            return PlasmaBackgroundConfig()
        raise ValueError('solver.plasma_background must be a mapping when enabled')
    if not isinstance(raw, Mapping):
        raise ValueError('solver.plasma_background must be a mapping')

    source = str(raw.get('source', 'none')).strip().lower()
    source_aliases = {
        '': 'none',
        '0': 'none',
        'false': 'none',
        'off': 'none',
        'disabled': 'none',
        'constant': 'saas_constant',
        'scalar': 'saas_constant',
        'saas': 'saas_constant',
    }
    source = source_aliases.get(source, source)
    if source not in {'none', 'saas_constant'}:
        raise ValueError("solver.plasma_background.source must be 'none' or 'saas_constant'")
    if source == 'none':
        return PlasmaBackgroundConfig()

    cfg = PlasmaBackgroundConfig(
        source=source,
        electron_density_m3=_positive_float(raw, 'electron_density_m3', 'ne_m3'),
        ion_density_m3=_positive_float(raw, 'ion_density_m3', 'ni_m3'),
        electron_temperature_eV=_positive_float(raw, 'electron_temperature_eV', 'Te_eV'),
        ion_temperature_eV=_positive_float(raw, 'ion_temperature_eV', 'Ti_eV'),
        ion_mass_amu=float(raw.get('ion_mass_amu', 69.0)),
        ion_charge_number=float(raw.get('ion_charge_number', 1.0)),
        pressure_Pa=_positive_float(raw, 'pressure_Pa', 'gas_pressure_Pa'),
        gas_temperature_K=_positive_float(raw, 'gas_temperature_K', 'neutral_temperature_K'),
        neutral_molecular_mass_amu=_positive_float(
            raw,
            'neutral_molecular_mass_amu',
            'gas_molecular_mass_amu',
        ),
        electron_neutral_cross_section_m2=_positive_float(raw, 'electron_neutral_cross_section_m2'),
        ion_neutral_cross_section_m2=_positive_float(raw, 'ion_neutral_cross_section_m2'),
        electron_collision_frequency_s=float(raw.get('electron_collision_frequency_s', 0.0)),
        ion_collision_frequency_s=float(raw.get('ion_collision_frequency_s', 0.0)),
        electron_ion_collision_frequency_s=float(raw.get('electron_ion_collision_frequency_s', 0.0)),
        conductivity_Sm=float(raw.get('conductivity_Sm', raw.get('conductivity_S_m', 0.0))),
    )
    for name in (
        'electron_density_m3',
        'ion_density_m3',
        'electron_temperature_eV',
        'ion_temperature_eV',
        'ion_mass_amu',
        'ion_charge_number',
    ):
        value = float(getattr(cfg, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f'solver.plasma_background.{name} must be positive for saas_constant')
    for name in (
        'pressure_Pa',
        'gas_temperature_K',
        'neutral_molecular_mass_amu',
        'electron_neutral_cross_section_m2',
        'ion_neutral_cross_section_m2',
        'electron_collision_frequency_s',
        'ion_collision_frequency_s',
        'electron_ion_collision_frequency_s',
        'conductivity_Sm',
    ):
        value = float(getattr(cfg, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f'solver.plasma_background.{name} must be non-negative')
    return cfg


def debye_length_m(
    te_eV: np.ndarray | float,
    ne_m3: np.ndarray | float,
    ti_eV: np.ndarray | float,
    ni_m3: np.ndarray | float,
    ion_charge_number: float,
) -> np.ndarray:
    te = np.asarray(te_eV, dtype=np.float64)
    ne = np.asarray(ne_m3, dtype=np.float64)
    ti = np.asarray(ti_eV, dtype=np.float64)
    ni = np.asarray(ni_m3, dtype=np.float64)
    zi = max(float(ion_charge_number), 1.0e-12)
    electron_term = np.divide(
        ne,
        te,
        out=np.zeros(np.broadcast_shapes(ne.shape, te.shape), dtype=np.float64),
        where=(ne > 0.0) & (te > 0.0),
    )
    ion_term = np.divide(
        zi * zi * ni,
        ti,
        out=np.zeros(np.broadcast_shapes(ni.shape, ti.shape), dtype=np.float64),
        where=(ni > 0.0) & (ti > 0.0),
    )
    inv_lambda2 = (E_CHARGE_C / EPS0_F_M) * (electron_term + ion_term)
    return np.sqrt(
        np.divide(
            1.0,
            inv_lambda2,
            out=np.full_like(inv_lambda2, np.nan, dtype=np.float64),
            where=inv_lambda2 > 0.0,
        )
    )


def _single_species_debye_length_m(temp_eV: float, density_m3: float, charge_number: float = 1.0) -> float:
    density = max(float(density_m3), 0.0)
    temp = max(float(temp_eV), 0.0)
    z = max(float(charge_number), 1.0e-12)
    inv_lambda2 = (E_CHARGE_C / EPS0_F_M) * z * z * density / temp if density > 0.0 and temp > 0.0 else 0.0
    return float(np.sqrt(1.0 / inv_lambda2)) if inv_lambda2 > 0.0 else 0.0


def prepare_plasma_background(config: PlasmaBackgroundConfig) -> PreparedPlasmaBackground | None:
    if str(config.source) == 'none':
        return None
    ion_mass_kg = max(float(config.ion_mass_amu), 1.0e-12) * AMU_KG
    neutral_mass_kg = max(float(config.neutral_molecular_mass_amu), 0.0) * AMU_KG
    te = max(float(config.electron_temperature_eV), 1.0e-12)
    ti = max(float(config.ion_temperature_eV), 1.0e-12)
    ne = max(float(config.electron_density_m3), 0.0)
    ni = max(float(config.ion_density_m3), 0.0)
    zi = max(float(config.ion_charge_number), 1.0e-12)
    neutral_density = 0.0
    if float(config.pressure_Pa) > 0.0 and float(config.gas_temperature_K) > 0.0:
        neutral_density = float(config.pressure_Pa) / (KB_J_K * float(config.gas_temperature_K))
    electron_thermal_speed = float(np.sqrt(E_CHARGE_C * te / (2.0 * np.pi * ELECTRON_MASS_KG)))
    ion_thermal_speed = float(np.sqrt(E_CHARGE_C * ti / (2.0 * np.pi * ion_mass_kg)))
    electron_neutral_nu = 0.0
    ion_neutral_nu = 0.0
    collision_source = 'configured'
    if float(config.electron_collision_frequency_s) > 0.0 or float(config.ion_collision_frequency_s) > 0.0:
        electron_neutral_nu = float(config.electron_collision_frequency_s)
        ion_neutral_nu = float(config.ion_collision_frequency_s)
    elif neutral_density > 0.0:
        collision_source = 'derived_from_pressure_cross_section'
        electron_neutral_nu = neutral_density * max(float(config.electron_neutral_cross_section_m2), 0.0) * electron_thermal_speed
        ion_neutral_nu = neutral_density * max(float(config.ion_neutral_cross_section_m2), 0.0) * ion_thermal_speed
    else:
        collision_source = 'not_available'
    effective_electron_nu = float(electron_neutral_nu + float(config.electron_ion_collision_frequency_s))
    conductivity = float(config.conductivity_Sm)
    conductivity_source = 'configured'
    if conductivity <= 0.0 and effective_electron_nu > 0.0:
        conductivity = ne * E_CHARGE_C * E_CHARGE_C / (ELECTRON_MASS_KG * effective_electron_nu)
        conductivity_source = 'derived_from_effective_electron_collision_frequency'
    elif conductivity <= 0.0:
        conductivity_source = 'not_available'
    electron_mobility = E_CHARGE_C / (ELECTRON_MASS_KG * effective_electron_nu) if effective_electron_nu > 0.0 else 0.0
    ion_mobility = zi * E_CHARGE_C / (ion_mass_kg * ion_neutral_nu) if ion_neutral_nu > 0.0 else 0.0
    combined_debye = float(np.asarray(debye_length_m(te, ne, ti, ni, zi)))
    return PreparedPlasmaBackground(
        source=str(config.source),
        electron_density_m3=float(config.electron_density_m3),
        ion_density_m3=float(config.ion_density_m3),
        electron_temperature_eV=float(config.electron_temperature_eV),
        ion_temperature_eV=float(config.ion_temperature_eV),
        ion_mass_amu=float(config.ion_mass_amu),
        ion_mass_kg=float(ion_mass_kg),
        ion_charge_number=float(config.ion_charge_number),
        pressure_Pa=float(config.pressure_Pa),
        gas_temperature_K=float(config.gas_temperature_K),
        neutral_molecular_mass_amu=float(config.neutral_molecular_mass_amu),
        neutral_molecular_mass_kg=float(neutral_mass_kg),
        neutral_density_m3=float(neutral_density),
        electron_neutral_cross_section_m2=float(config.electron_neutral_cross_section_m2),
        ion_neutral_cross_section_m2=float(config.ion_neutral_cross_section_m2),
        electron_debye_length_m=_single_species_debye_length_m(te, ne, 1.0),
        ion_debye_length_m=_single_species_debye_length_m(ti, ni, zi),
        debye_length_m=float(combined_debye),
        electron_collision_frequency_s=float(electron_neutral_nu),
        ion_collision_frequency_s=float(ion_neutral_nu),
        electron_ion_collision_frequency_s=float(config.electron_ion_collision_frequency_s),
        effective_electron_collision_frequency_s=float(effective_electron_nu),
        conductivity_Sm=float(conductivity),
        electron_mobility_m2Vs=float(electron_mobility),
        ion_mobility_m2Vs=float(ion_mobility),
        electron_thermal_speed_mps=float(electron_thermal_speed),
        ion_thermal_speed_mps=float(ion_thermal_speed),
        ion_bohm_speed_mps=float(np.sqrt(E_CHARGE_C * te / ion_mass_kg)),
        electron_plasma_frequency_rad_s=float(np.sqrt(ne * E_CHARGE_C * E_CHARGE_C / (EPS0_F_M * ELECTRON_MASS_KG))),
        ion_plasma_frequency_rad_s=float(np.sqrt(ni * zi * zi * E_CHARGE_C * E_CHARGE_C / (EPS0_F_M * ion_mass_kg))),
        collision_frequency_source=str(collision_source),
        conductivity_source=str(conductivity_source),
    )


def plasma_background_report(background: PreparedPlasmaBackground | None) -> Dict[str, object]:
    if background is None:
        return {'source': 'none', 'enabled': 0}
    return {
        'source': str(background.source),
        'enabled': 1,
        'electron_density_m3': float(background.electron_density_m3),
        'ion_density_m3': float(background.ion_density_m3),
        'electron_temperature_eV': float(background.electron_temperature_eV),
        'ion_temperature_eV': float(background.ion_temperature_eV),
        'ion_mass_amu': float(background.ion_mass_amu),
        'ion_mass_kg': float(background.ion_mass_kg),
        'ion_charge_number': float(background.ion_charge_number),
        'pressure_Pa': float(background.pressure_Pa),
        'gas_temperature_K': float(background.gas_temperature_K),
        'neutral_molecular_mass_amu': float(background.neutral_molecular_mass_amu),
        'neutral_density_m3': float(background.neutral_density_m3),
        'electron_neutral_cross_section_m2': float(background.electron_neutral_cross_section_m2),
        'ion_neutral_cross_section_m2': float(background.ion_neutral_cross_section_m2),
        'electron_debye_length_m': float(background.electron_debye_length_m),
        'ion_debye_length_m': float(background.ion_debye_length_m),
        'debye_length_m': float(background.debye_length_m),
        'electron_collision_frequency_s': float(background.electron_collision_frequency_s),
        'ion_collision_frequency_s': float(background.ion_collision_frequency_s),
        'electron_ion_collision_frequency_s': float(background.electron_ion_collision_frequency_s),
        'effective_electron_collision_frequency_s': float(background.effective_electron_collision_frequency_s),
        'conductivity_Sm': float(background.conductivity_Sm),
        'electron_mobility_m2Vs': float(background.electron_mobility_m2Vs),
        'ion_mobility_m2Vs': float(background.ion_mobility_m2Vs),
        'electron_thermal_speed_mps': float(background.electron_thermal_speed_mps),
        'ion_thermal_speed_mps': float(background.ion_thermal_speed_mps),
        'ion_bohm_speed_mps': float(background.ion_bohm_speed_mps),
        'electron_plasma_frequency_rad_s': float(background.electron_plasma_frequency_rad_s),
        'ion_plasma_frequency_rad_s': float(background.ion_plasma_frequency_rad_s),
        'collision_frequency_source': str(background.collision_frequency_source),
        'conductivity_source': str(background.conductivity_source),
    }


__all__ = (
    'PlasmaBackgroundConfig',
    'PreparedPlasmaBackground',
    'debye_length_m',
    'parse_plasma_background_config',
    'plasma_background_report',
    'prepare_plasma_background',
)
