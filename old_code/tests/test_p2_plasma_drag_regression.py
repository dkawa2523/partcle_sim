from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from particle_tracer_unified.solvers.drag_regime import (
    DragRegimeDecision,
    classify_drag_regime,
    gas_mean_free_path_scalar_m,
)
from particle_tracer_unified.solvers.plasma_background import (
    AMU_KG,
    E_CHARGE_C,
    ELECTRON_MASS_KG,
    KB_J_K,
    PlasmaBackgroundConfig,
    debye_length_m,
    plasma_background_report,
    prepare_plasma_background,
)


def _plasma_config() -> PlasmaBackgroundConfig:
    return PlasmaBackgroundConfig(
        source="saas_constant",
        electron_density_m3=2.0e15,
        ion_density_m3=2.0e15,
        electron_temperature_eV=3.0,
        ion_temperature_eV=0.03,
        ion_mass_amu=39.948,
        ion_charge_number=2.0,
        pressure_Pa=2.0,
        gas_temperature_K=300.0,
        neutral_molecular_mass_amu=39.948,
        electron_neutral_cross_section_m2=1.0e-19,
        ion_neutral_cross_section_m2=2.0e-19,
        electron_ion_collision_frequency_s=1.0e5,
    )


def test_plasma_preparation_preserves_configured_collision_values() -> None:
    config = replace(
        _plasma_config(),
        electron_collision_frequency_s=2.0e6,
        ion_collision_frequency_s=3.0e6,
        electron_ion_collision_frequency_s=4.0e6,
        conductivity_Sm=5.0,
    )

    prepared = prepare_plasma_background(config)

    assert prepared is not None
    assert prepared.collision_frequency_source == "configured"
    assert prepared.conductivity_source == "configured"
    assert prepared.electron_collision_frequency_s == 2.0e6
    assert prepared.ion_collision_frequency_s == 3.0e6
    assert prepared.effective_electron_collision_frequency_s == 6.0e6
    assert prepared.conductivity_Sm == 5.0
    assert prepared.neutral_density_m3 == pytest.approx(
        config.pressure_Pa / (KB_J_K * config.gas_temperature_K)
    )
    report = plasma_background_report(prepared)
    assert report["source"] == "saas_constant"
    assert report["electron_collision_frequency_s"] == 2.0e6


def test_plasma_preparation_preserves_derived_transport_calculation() -> None:
    config = _plasma_config()
    neutral_density = config.pressure_Pa / (KB_J_K * config.gas_temperature_K)
    electron_speed = np.sqrt(
        E_CHARGE_C * config.electron_temperature_eV / (2.0 * np.pi * ELECTRON_MASS_KG)
    )
    ion_mass = config.ion_mass_amu * AMU_KG
    ion_speed = np.sqrt(
        E_CHARGE_C * config.ion_temperature_eV / (2.0 * np.pi * ion_mass)
    )
    electron_collision = (
        neutral_density * config.electron_neutral_cross_section_m2 * electron_speed
    )
    ion_collision = neutral_density * config.ion_neutral_cross_section_m2 * ion_speed
    effective_collision = electron_collision + config.electron_ion_collision_frequency_s

    prepared = prepare_plasma_background(config)

    assert prepared is not None
    assert prepared.collision_frequency_source == "derived_from_pressure_cross_section"
    assert prepared.conductivity_source == (
        "derived_from_effective_electron_collision_frequency"
    )
    assert prepared.electron_thermal_speed_mps == electron_speed
    assert prepared.ion_thermal_speed_mps == ion_speed
    assert prepared.electron_collision_frequency_s == electron_collision
    assert prepared.ion_collision_frequency_s == ion_collision
    assert prepared.effective_electron_collision_frequency_s == effective_collision
    assert prepared.conductivity_Sm == (
        config.electron_density_m3
        * E_CHARGE_C
        * E_CHARGE_C
        / (ELECTRON_MASS_KG * effective_collision)
    )
    assert prepared.electron_mobility_m2Vs == (
        E_CHARGE_C / (ELECTRON_MASS_KG * effective_collision)
    )
    assert prepared.ion_mobility_m2Vs == (
        config.ion_charge_number * E_CHARGE_C / (ion_mass * ion_collision)
    )


def test_plasma_preparation_preserves_unavailable_and_disabled_states() -> None:
    unavailable = replace(
        _plasma_config(),
        pressure_Pa=0.0,
        gas_temperature_K=0.0,
        neutral_molecular_mass_amu=0.0,
        electron_neutral_cross_section_m2=0.0,
        ion_neutral_cross_section_m2=0.0,
        electron_ion_collision_frequency_s=0.0,
    )

    prepared = prepare_plasma_background(unavailable)

    assert prepared is not None
    assert prepared.collision_frequency_source == "not_available"
    assert prepared.conductivity_source == "not_available"
    assert prepared.neutral_density_m3 == 0.0
    assert prepared.electron_collision_frequency_s == 0.0
    assert prepared.ion_collision_frequency_s == 0.0
    assert prepared.effective_electron_collision_frequency_s == 0.0
    assert prepared.conductivity_Sm == 0.0
    assert prepared.electron_mobility_m2Vs == 0.0
    assert prepared.ion_mobility_m2Vs == 0.0
    assert (
        prepare_plasma_background(
            replace(unavailable, source="none", electron_density_m3=float("nan"))
        )
        is None
    )
    assert plasma_background_report(None) == {"source": "none", "enabled": 0}


def test_plasma_preparation_preserves_validation_order_and_messages() -> None:
    with pytest.raises(ValueError, match="source must be 'saas_constant'"):
        prepare_plasma_background(replace(_plasma_config(), source="legacy"))
    with pytest.raises(ValueError, match="pressure_Pa must be finite and non-negative"):
        prepare_plasma_background(replace(_plasma_config(), pressure_Pa=-1.0))
    with pytest.raises(ValueError, match="electron thermal speed"):
        prepare_plasma_background(
            replace(_plasma_config(), electron_temperature_eV=1.0e308)
        )
    with pytest.raises(ValueError, match="masses and thermal speeds must be positive"):
        prepare_plasma_background(
            replace(_plasma_config(), ion_temperature_eV=np.nextafter(0.0, 1.0))
        )
    with (
        np.errstate(over="ignore"),
        pytest.raises(
            ValueError,
            match=(
                "invalid electron collision frequency, "
                "effective electron collision frequency"
            ),
        ),
    ):
        prepare_plasma_background(
            replace(
                _plasma_config(),
                electron_neutral_cross_section_m2=1.0e308,
            )
        )


def test_debye_length_preserves_broadcast_and_derived_value_errors() -> None:
    with pytest.raises(ValueError, match="broadcast-compatible"):
        debye_length_m(
            np.ones(2),
            np.ones(3),
            np.ones(2),
            np.ones(2),
            1.0,
        )
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        debye_length_m(
            1.0e308,
            np.nextafter(0.0, 1.0),
            1.0e308,
            np.nextafter(0.0, 1.0),
            1.0,
        )


def test_scalar_mean_free_path_preserves_valid_and_invalid_contracts() -> None:
    expected = (2.0e-5 / 0.4) * np.sqrt(
        np.pi * (28.0 * AMU_KG) / (2.0 * KB_J_K * 400.0)
    )

    assert gas_mean_free_path_scalar_m(
        2.0e-5,
        0.4,
        400.0,
        28.0 * AMU_KG,
    ) == pytest.approx(expected)
    assert np.isnan(gas_mean_free_path_scalar_m(0.0, 0.4, 400.0, 28.0 * AMU_KG))


@pytest.mark.parametrize(
    ("model", "reynolds", "knudsen", "mach", "expected"),
    [
        (
            " STOKES ",
            1.0,
            0.1,
            1.0,
            DragRegimeDecision(
                errors=(
                    "particle_reynolds_outside_creeping_flow",
                    "knudsen_outside_unrarefied_continuum",
                    "relative_mach_supersonic_drag_not_supported",
                )
            ),
        ),
        (
            "stokes_cunningham",
            0.1,
            10.0,
            0.3,
            DragRegimeDecision(
                warnings=(
                    "particle_reynolds_near_creeping_flow_limit",
                    "knudsen_free_molecular_epstein_review",
                    "relative_mach_requires_compressibility_review",
                )
            ),
        ),
        (
            "schiller_naumann",
            800.0,
            0.01,
            float("nan"),
            DragRegimeDecision(
                errors=("particle_reynolds_outside_schiller_naumann",),
                warnings=("knudsen_requires_rarefaction_review",),
            ),
        ),
        (
            "epstein",
            0.0,
            1.0,
            1.0,
            DragRegimeDecision(
                errors=(
                    "knudsen_outside_free_molecular_flow",
                    "relative_mach_supersonic_drag_not_supported",
                )
            ),
        ),
        (
            "epstein",
            float("nan"),
            5.0,
            0.3,
            DragRegimeDecision(
                warnings=(
                    "knudsen_transitional_not_asymptotic_free_molecular",
                    "relative_mach_requires_compressibility_review",
                )
            ),
        ),
        (
            "epstein",
            0.0,
            10.0,
            float("nan"),
            DragRegimeDecision(),
        ),
        (
            "none",
            1.0e9,
            1.0e9,
            1.0e9,
            DragRegimeDecision(),
        ),
        (
            "unknown",
            float("nan"),
            float("nan"),
            float("nan"),
            DragRegimeDecision(),
        ),
    ],
)
def test_drag_regime_preserves_policy_thresholds_and_message_order(
    model: str,
    reynolds: float,
    knudsen: float,
    mach: float,
    expected: DragRegimeDecision,
) -> None:
    assert (
        classify_drag_regime(
            model,
            reynolds=reynolds,
            knudsen=knudsen,
            relative_mach=mach,
        )
        == expected
    )
