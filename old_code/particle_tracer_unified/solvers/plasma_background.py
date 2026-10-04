from __future__ import annotations

from dataclasses import dataclass

import numpy as np

E_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
EPS0_F_M = 8.8541878128e-12
AMU_KG = 1.66053906660e-27
KB_J_K = 1.380649e-23


@dataclass(frozen=True, slots=True)
class PlasmaBackgroundConfig:
    source: str = "none"
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


_SAAS_POSITIVE_FIELDS = (
    "electron_density_m3",
    "ion_density_m3",
    "electron_temperature_eV",
    "ion_temperature_eV",
    "ion_mass_amu",
    "ion_charge_number",
)
_SAAS_NONNEGATIVE_FIELDS = (
    "pressure_Pa",
    "gas_temperature_K",
    "neutral_molecular_mass_amu",
    "electron_neutral_cross_section_m2",
    "ion_neutral_cross_section_m2",
    "electron_collision_frequency_s",
    "ion_collision_frequency_s",
    "electron_ion_collision_frequency_s",
    "conductivity_Sm",
)


def _validate_config_fields(
    config: PlasmaBackgroundConfig,
    names: tuple[str, ...],
    *,
    allow_zero: bool,
) -> None:
    requirement = "non-negative" if allow_zero else "positive for saas_constant"
    for name in names:
        value = float(getattr(config, name))
        outside_range = value < 0.0 if allow_zero else value <= 0.0
        if not np.isfinite(value) or outside_range:
            raise ValueError(
                f"solver.plasma_background.{name} must be finite and {requirement}"
            )


def _validate_plasma_background_config(config: PlasmaBackgroundConfig) -> None:
    if str(config.source) != "saas_constant":
        raise ValueError("solver.plasma_background.source must be 'saas_constant'")
    _validate_config_fields(config, _SAAS_POSITIVE_FIELDS, allow_zero=False)
    _validate_config_fields(config, _SAAS_NONNEGATIVE_FIELDS, allow_zero=True)
    if float(config.pressure_Pa) > 0.0 and float(config.gas_temperature_K) <= 0.0:
        raise ValueError(
            "solver.plasma_background.gas_temperature_K must be positive when "
            "pressure_Pa is positive"
        )


def debye_length_m(
    te_eV: np.ndarray | float,
    ne_m3: np.ndarray | float,
    ti_eV: np.ndarray | float,
    ni_m3: np.ndarray | float,
    ion_charge_number: float,
) -> np.ndarray:
    try:
        te, ne, ti, ni = np.broadcast_arrays(
            np.asarray(te_eV, dtype=np.float64),
            np.asarray(ne_m3, dtype=np.float64),
            np.asarray(ti_eV, dtype=np.float64),
            np.asarray(ni_m3, dtype=np.float64),
        )
    except ValueError as exc:
        raise ValueError(
            "Debye-length plasma arrays must be broadcast-compatible"
        ) from exc
    for name, values in (
        ("electron temperature", te),
        ("electron density", ne),
        ("ion temperature", ti),
        ("ion density", ni),
    ):
        invalid = ~np.isfinite(values) | (values <= 0.0)
        if np.any(invalid):
            rows = np.flatnonzero(invalid)[:12].tolist()
            raise ValueError(
                f"Debye-length {name} must be finite and positive; invalid rows {rows}"
            )
    zi = float(ion_charge_number)
    if not np.isfinite(zi) or zi <= 0.0:
        raise ValueError("Debye-length ion charge number must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inv_lambda2 = (E_CHARGE_C / EPS0_F_M) * (ne / te + zi * zi * ni / ti)
        result = np.sqrt(1.0 / inv_lambda2)
    invalid_result = ~np.isfinite(result) | (result <= 0.0)
    if np.any(invalid_result):
        rows = np.flatnonzero(invalid_result)[:12].tolist()
        raise ValueError(
            f"Debye length is non-finite or non-positive; invalid rows {rows}"
        )
    return np.asarray(result, dtype=np.float64)


def _single_species_debye_length_m(
    temp_eV: float, density_m3: float, charge_number: float = 1.0
) -> float:
    density = float(density_m3)
    temp = float(temp_eV)
    z = float(charge_number)
    for name, value in (
        ("temperature", temp),
        ("density", density),
        ("charge number", z),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Single-species Debye {name} must be finite and positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inv_lambda2 = (E_CHARGE_C / EPS0_F_M) * z * z * density / temp
        result = float(np.sqrt(1.0 / inv_lambda2))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("Single-species Debye length must be finite and positive")
    return result


def _thermal_state(
    config: PlasmaBackgroundConfig,
) -> tuple[float, float, float, float, float]:
    ion_mass_kg = float(config.ion_mass_amu) * AMU_KG
    neutral_mass_kg = float(config.neutral_molecular_mass_amu) * AMU_KG
    te = float(config.electron_temperature_eV)
    ti = float(config.ion_temperature_eV)
    neutral_density = 0.0
    if float(config.pressure_Pa) > 0.0:
        neutral_density = float(config.pressure_Pa) / (
            KB_J_K * float(config.gas_temperature_K)
        )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        electron_thermal_speed = float(
            np.sqrt(E_CHARGE_C * te / (2.0 * np.pi * ELECTRON_MASS_KG))
        )
        ion_thermal_speed = float(
            np.sqrt(E_CHARGE_C * ti / (2.0 * np.pi * ion_mass_kg))
        )
    for name, value in (
        ("ion mass", ion_mass_kg),
        ("neutral mass", neutral_mass_kg),
        ("neutral density", neutral_density),
        ("electron thermal speed", electron_thermal_speed),
        ("ion thermal speed", ion_thermal_speed),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Prepared plasma {name} must be finite and non-negative")
    if ion_mass_kg == 0.0 or electron_thermal_speed == 0.0 or ion_thermal_speed == 0.0:
        raise ValueError("Prepared plasma masses and thermal speeds must be positive")
    return (
        ion_mass_kg,
        neutral_mass_kg,
        neutral_density,
        electron_thermal_speed,
        ion_thermal_speed,
    )


def _collision_frequencies(
    config: PlasmaBackgroundConfig,
    neutral_density: float,
    electron_thermal_speed: float,
    ion_thermal_speed: float,
) -> tuple[float, float, str]:
    electron_neutral_nu = 0.0
    ion_neutral_nu = 0.0
    collision_source = "configured"
    if (
        float(config.electron_collision_frequency_s) > 0.0
        or float(config.ion_collision_frequency_s) > 0.0
    ):
        electron_neutral_nu = float(config.electron_collision_frequency_s)
        ion_neutral_nu = float(config.ion_collision_frequency_s)
    elif neutral_density > 0.0:
        collision_source = "derived_from_pressure_cross_section"
        electron_neutral_nu = (
            neutral_density
            * float(config.electron_neutral_cross_section_m2)
            * electron_thermal_speed
        )
        ion_neutral_nu = (
            neutral_density
            * float(config.ion_neutral_cross_section_m2)
            * ion_thermal_speed
        )
    else:
        collision_source = "not_available"
    return electron_neutral_nu, ion_neutral_nu, collision_source


def _transport_properties(
    config: PlasmaBackgroundConfig,
    *,
    electron_density_m3: float,
    ion_mass_kg: float,
    ion_charge_number: float,
    electron_neutral_nu: float,
    ion_neutral_nu: float,
) -> tuple[float, float, str, float, float]:
    effective_electron_nu = float(
        electron_neutral_nu + float(config.electron_ion_collision_frequency_s)
    )
    conductivity = float(config.conductivity_Sm)
    conductivity_source = "configured"
    if conductivity <= 0.0 and effective_electron_nu > 0.0:
        conductivity = (
            electron_density_m3
            * E_CHARGE_C
            * E_CHARGE_C
            / (ELECTRON_MASS_KG * effective_electron_nu)
        )
        conductivity_source = "derived_from_effective_electron_collision_frequency"
    elif conductivity <= 0.0:
        conductivity_source = "not_available"
    electron_mobility = (
        E_CHARGE_C / (ELECTRON_MASS_KG * effective_electron_nu)
        if effective_electron_nu > 0.0
        else 0.0
    )
    ion_mobility = (
        ion_charge_number * E_CHARGE_C / (ion_mass_kg * ion_neutral_nu)
        if ion_neutral_nu > 0.0
        else 0.0
    )
    return (
        effective_electron_nu,
        conductivity,
        conductivity_source,
        electron_mobility,
        ion_mobility,
    )


def _validate_derived_values(
    values: tuple[tuple[str, float], ...],
    *,
    positive: bool,
) -> None:
    invalid = [
        name
        for name, value in values
        if not np.isfinite(value) or value < 0.0 or (positive and value == 0.0)
    ]
    if invalid:
        raise ValueError("Prepared plasma produced invalid " + ", ".join(invalid))


def _plasma_frequency_state(
    *,
    electron_density_m3: float,
    ion_density_m3: float,
    electron_temperature_eV: float,
    ion_mass_kg: float,
    ion_charge_number: float,
) -> tuple[float, float, float]:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ion_bohm_speed = float(
            np.sqrt(
                ion_charge_number * E_CHARGE_C * electron_temperature_eV / ion_mass_kg
            )
        )
        electron_plasma_frequency = float(
            np.sqrt(
                electron_density_m3
                * E_CHARGE_C
                * E_CHARGE_C
                / (EPS0_F_M * ELECTRON_MASS_KG)
            )
        )
        ion_plasma_frequency = float(
            np.sqrt(
                ion_density_m3
                * ion_charge_number
                * ion_charge_number
                * E_CHARGE_C
                * E_CHARGE_C
                / (EPS0_F_M * ion_mass_kg)
            )
        )
    _validate_derived_values(
        (
            ("ion Bohm speed", ion_bohm_speed),
            ("electron plasma frequency", electron_plasma_frequency),
            ("ion plasma frequency", ion_plasma_frequency),
        ),
        positive=True,
    )
    return ion_bohm_speed, electron_plasma_frequency, ion_plasma_frequency


def prepare_plasma_background(
    config: PlasmaBackgroundConfig,
) -> PreparedPlasmaBackground | None:
    if str(config.source) == "none":
        return None
    _validate_plasma_background_config(config)
    te = float(config.electron_temperature_eV)
    ti = float(config.ion_temperature_eV)
    ne = float(config.electron_density_m3)
    ni = float(config.ion_density_m3)
    zi = float(config.ion_charge_number)
    (
        ion_mass_kg,
        neutral_mass_kg,
        neutral_density,
        electron_thermal_speed,
        ion_thermal_speed,
    ) = _thermal_state(config)
    electron_neutral_nu, ion_neutral_nu, collision_source = _collision_frequencies(
        config,
        neutral_density,
        electron_thermal_speed,
        ion_thermal_speed,
    )
    (
        effective_electron_nu,
        conductivity,
        conductivity_source,
        electron_mobility,
        ion_mobility,
    ) = _transport_properties(
        config,
        electron_density_m3=ne,
        ion_mass_kg=ion_mass_kg,
        ion_charge_number=zi,
        electron_neutral_nu=electron_neutral_nu,
        ion_neutral_nu=ion_neutral_nu,
    )
    combined_debye = float(np.asarray(debye_length_m(te, ne, ti, ni, zi)))
    _validate_derived_values(
        (
            ("electron collision frequency", electron_neutral_nu),
            ("ion collision frequency", ion_neutral_nu),
            ("effective electron collision frequency", effective_electron_nu),
            ("conductivity", conductivity),
            ("electron mobility", electron_mobility),
            ("ion mobility", ion_mobility),
        ),
        positive=False,
    )
    (
        ion_bohm_speed,
        electron_plasma_frequency,
        ion_plasma_frequency,
    ) = _plasma_frequency_state(
        electron_density_m3=ne,
        ion_density_m3=ni,
        electron_temperature_eV=te,
        ion_mass_kg=ion_mass_kg,
        ion_charge_number=zi,
    )
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
        electron_neutral_cross_section_m2=float(
            config.electron_neutral_cross_section_m2
        ),
        ion_neutral_cross_section_m2=float(config.ion_neutral_cross_section_m2),
        electron_debye_length_m=_single_species_debye_length_m(te, ne, 1.0),
        ion_debye_length_m=_single_species_debye_length_m(ti, ni, zi),
        debye_length_m=float(combined_debye),
        electron_collision_frequency_s=float(electron_neutral_nu),
        ion_collision_frequency_s=float(ion_neutral_nu),
        electron_ion_collision_frequency_s=float(
            config.electron_ion_collision_frequency_s
        ),
        effective_electron_collision_frequency_s=float(effective_electron_nu),
        conductivity_Sm=float(conductivity),
        electron_mobility_m2Vs=float(electron_mobility),
        ion_mobility_m2Vs=float(ion_mobility),
        electron_thermal_speed_mps=float(electron_thermal_speed),
        ion_thermal_speed_mps=float(ion_thermal_speed),
        ion_bohm_speed_mps=float(ion_bohm_speed),
        electron_plasma_frequency_rad_s=float(electron_plasma_frequency),
        ion_plasma_frequency_rad_s=float(ion_plasma_frequency),
        collision_frequency_source=str(collision_source),
        conductivity_source=str(conductivity_source),
    )


def plasma_background_report(
    background: PreparedPlasmaBackground | None,
) -> dict[str, object]:
    if background is None:
        return {"source": "none", "enabled": 0}
    return {
        "source": str(background.source),
        "enabled": 1,
        "electron_density_m3": float(background.electron_density_m3),
        "ion_density_m3": float(background.ion_density_m3),
        "electron_temperature_eV": float(background.electron_temperature_eV),
        "ion_temperature_eV": float(background.ion_temperature_eV),
        "ion_mass_amu": float(background.ion_mass_amu),
        "ion_mass_kg": float(background.ion_mass_kg),
        "ion_charge_number": float(background.ion_charge_number),
        "pressure_Pa": float(background.pressure_Pa),
        "gas_temperature_K": float(background.gas_temperature_K),
        "neutral_molecular_mass_amu": float(background.neutral_molecular_mass_amu),
        "neutral_density_m3": float(background.neutral_density_m3),
        "electron_neutral_cross_section_m2": float(
            background.electron_neutral_cross_section_m2
        ),
        "ion_neutral_cross_section_m2": float(background.ion_neutral_cross_section_m2),
        "electron_debye_length_m": float(background.electron_debye_length_m),
        "ion_debye_length_m": float(background.ion_debye_length_m),
        "debye_length_m": float(background.debye_length_m),
        "electron_collision_frequency_s": float(
            background.electron_collision_frequency_s
        ),
        "ion_collision_frequency_s": float(background.ion_collision_frequency_s),
        "electron_ion_collision_frequency_s": float(
            background.electron_ion_collision_frequency_s
        ),
        "effective_electron_collision_frequency_s": float(
            background.effective_electron_collision_frequency_s
        ),
        "conductivity_Sm": float(background.conductivity_Sm),
        "electron_mobility_m2Vs": float(background.electron_mobility_m2Vs),
        "ion_mobility_m2Vs": float(background.ion_mobility_m2Vs),
        "electron_thermal_speed_mps": float(background.electron_thermal_speed_mps),
        "ion_thermal_speed_mps": float(background.ion_thermal_speed_mps),
        "ion_bohm_speed_mps": float(background.ion_bohm_speed_mps),
        "electron_plasma_frequency_rad_s": float(
            background.electron_plasma_frequency_rad_s
        ),
        "ion_plasma_frequency_rad_s": float(background.ion_plasma_frequency_rad_s),
        "collision_frequency_source": str(background.collision_frequency_source),
        "conductivity_source": str(background.conductivity_source),
    }


__all__ = (
    "PlasmaBackgroundConfig",
    "PreparedPlasmaBackground",
    "debye_length_m",
    "plasma_background_report",
    "prepare_plasma_background",
)
