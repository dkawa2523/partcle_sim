from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from particle_tracer_unified.solvers.plasma_background import (
    AMU_KG,
    PlasmaBackgroundConfig,
    debye_length_m,
    prepare_plasma_background,
)


def _background() -> PlasmaBackgroundConfig:
    return PlasmaBackgroundConfig(
        source="saas_constant",
        electron_density_m3=2.0e15,
        ion_density_m3=2.0e15,
        electron_temperature_eV=3.0,
        ion_temperature_eV=0.03,
        ion_mass_amu=39.948,
        ion_charge_number=1.0,
        pressure_Pa=2.0,
        gas_temperature_K=300.0,
        neutral_molecular_mass_amu=39.948,
        electron_neutral_cross_section_m2=1.0e-19,
        ion_neutral_cross_section_m2=2.0e-19,
        electron_ion_collision_frequency_s=1.0e5,
    )


def test_plasma_background_is_prepared_from_resolved_typed_values() -> None:
    config = _background()
    prepared = prepare_plasma_background(config)

    assert prepared is not None
    assert prepared.source == "saas_constant"
    assert prepared.electron_density_m3 == config.electron_density_m3
    assert np.isfinite(prepared.debye_length_m)
    assert prepared.debye_length_m > 0.0


def test_debye_length_matches_two_species_formula_and_rejects_invalid_inputs() -> None:
    te = np.asarray([3.0, 4.0])
    ne = np.asarray([2.0e15, 3.0e15])
    ti = np.asarray([0.03, 0.05])
    ni = np.asarray([2.0e15, 3.0e15])
    zi = 2.0
    actual = debye_length_m(te, ne, ti, ni, zi)
    expected = np.sqrt(
        1.0 / ((1.602176634e-19 / 8.8541878128e-12) * (ne / te + zi * zi * ni / ti))
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-15, atol=0.0)
    with pytest.raises(ValueError, match="electron temperature"):
        debye_length_m(0.0, 2.0e15, 0.03, 2.0e15, 1.0)
    with pytest.raises(ValueError, match="electron density"):
        debye_length_m(3.0, float("nan"), 0.03, 2.0e15, 1.0)
    with pytest.raises(ValueError, match="ion charge number"):
        debye_length_m(3.0, 2.0e15, 0.03, 2.0e15, 0.0)


def test_plasma_preparation_uses_exact_values_without_numeric_floors() -> None:
    config = replace(_background(), ion_mass_amu=1.0e-20)
    prepared = prepare_plasma_background(config)

    assert prepared is not None
    assert prepared.ion_mass_kg == pytest.approx(1.0e-20 * AMU_KG, rel=0.0, abs=0.0)

    with pytest.raises(ValueError, match="ion_mass_amu"):
        prepare_plasma_background(replace(_background(), ion_mass_amu=0.0))


def test_plasma_preparation_uses_ion_charge_and_rejects_incomplete_gas_state() -> None:
    config = replace(_background(), ion_charge_number=3.0)
    prepared = prepare_plasma_background(config)

    assert prepared is not None
    expected_bohm_speed = np.sqrt(
        config.ion_charge_number
        * 1.602176634e-19
        * config.electron_temperature_eV
        / (config.ion_mass_amu * AMU_KG)
    )
    assert prepared.ion_bohm_speed_mps == pytest.approx(
        expected_bohm_speed, rel=1.0e-15
    )

    with pytest.raises(ValueError, match="gas_temperature_K must be positive"):
        prepare_plasma_background(replace(_background(), gas_temperature_K=0.0))
