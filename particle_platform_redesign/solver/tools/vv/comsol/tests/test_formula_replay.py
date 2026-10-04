import math
from pathlib import Path

import numpy as np
from tools.vv.comsol.evaluate_dataset import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    charge_formula_metrics,
    epstein_diffuse_factor,
    load_evaluation_config,
    replay_relative_drift_charge,
)


def _reference_charge_row(
    *,
    radius: float,
    charge: float,
    phi1: float,
    ion_density: float,
    electron_density: float,
    ion_mass: float,
    ion_energy: float,
    electron_energy: float,
    relative_velocity: tuple[float, float],
) -> tuple[float, float]:
    speed_squared = (
        relative_velocity[0] ** 2
        + relative_velocity[1] ** 2
        + 8.0 * ELEMENTARY_CHARGE_C * ion_energy / (math.pi * ion_mass)
        + 1.0
    )
    speed = math.sqrt(speed_squared)
    effective_ion_energy = max(ion_mass * speed_squared / (2.0 * ELEMENTARY_CHARGE_C), 0.01)
    ion_base = math.pi * radius**2 * ion_density * speed
    electron_base = (
        math.pi
        * radius**2
        * electron_density
        * math.sqrt(8.0 * ELEMENTARY_CHARGE_C * electron_energy / (math.pi * ELECTRON_MASS_KG))
    )
    potential = charge * phi1
    if potential <= 0.0:
        ion_factor = 1.0 - potential / effective_ion_energy
        electron_argument = potential / electron_energy
        electron_factor = math.exp(max(-50.0, min(50.0, electron_argument)))
        ion_derivative = -phi1 / effective_ion_energy
        electron_derivative = (
            electron_factor * phi1 / electron_energy if -50.0 < electron_argument < 50.0 else 0.0
        )
    else:
        ion_argument = -potential / effective_ion_energy
        ion_factor = math.exp(max(-50.0, min(50.0, ion_argument)))
        electron_factor = 1.0 + potential / electron_energy
        ion_derivative = (
            -ion_factor * phi1 / effective_ion_energy if -50.0 < ion_argument < 50.0 else 0.0
        )
        electron_derivative = phi1 / electron_energy
    return (
        ion_base * ion_factor - electron_base * electron_factor,
        ion_base * ion_derivative - electron_base * electron_derivative,
    )


def test_reference_charge_replay_covers_sign_clamp_and_floor_branches() -> None:
    config = load_evaluation_config(Path(__file__).parents[1] / "cases" / "m3v.yaml")
    count = 4
    radius = np.full(count, 5.0e-8)
    charge = np.asarray([-1.0, 1.0, -1000.0, 1000.0])
    phi1 = np.asarray([0.2, 0.2, 1.0, 1.0])
    ion_density = np.full(count, 2.0e14)
    electron_density = np.full(count, 1.0e14)
    ion_mass = np.full(count, 6.63e-26)
    ion_energy = np.asarray([0.03, 0.03, 0.03, 1.0e-8])
    electron_energy = np.full(count, 2.0)
    relative_r = np.asarray([12.0, 12.0, 12.0, 0.0])
    relative_z = np.asarray([5.0, 5.0, 5.0, 0.0])
    values = {
        "particle_radius_m": radius,
        "charge_number_e": charge,
        "local_single_charge_surface_potential_increment_V": phi1,
        "effective_positive_ion_mass_kg": ion_mass,
        "local_ion_thermal_energy_eV_as_V": ion_energy,
        "electron_temperature_eV_as_V": electron_energy,
        "local_ion_velocity_r_m_per_s": relative_r,
        "local_ion_velocity_z_m_per_s": relative_z,
        "velocity_r_m_per_s": np.zeros(count),
        "velocity_z_m_per_s": np.zeros(count),
        "local_total_positive_ion_density_per_m3": ion_density,
        "local_electron_density_per_m3": electron_density,
    }
    expected = np.asarray(
        [
            _reference_charge_row(
                radius=radius[index],
                charge=charge[index],
                phi1=phi1[index],
                ion_density=ion_density[index],
                electron_density=electron_density[index],
                ion_mass=ion_mass[index],
                ion_energy=ion_energy[index],
                electron_energy=electron_energy[index],
                relative_velocity=(relative_r[index], relative_z[index]),
            )
            for index in range(count)
        ]
    )

    rate, _, derivative = replay_relative_drift_charge(values, config)

    np.testing.assert_allclose(rate, expected[:, 0], rtol=2.0e-15)
    np.testing.assert_allclose(derivative, expected[:, 1], rtol=2.0e-15)
    values["dynamic_charge_rate_dZdt_per_s"] = expected[:, 0]
    metrics = charge_formula_metrics(values, config)
    assert metrics["dataset_drift_aware_charge_rate_formula_parity"] == "PASS"
    assert metrics["dataset_charge_positive_potential_branch_fraction"] == 0.5
    assert metrics["dataset_charge_ion_energy_floor_branch_fraction"] == 0.25
    assert metrics["dataset_charge_electron_exponent_clamp_branch_fraction"] == 0.25
    assert metrics["dataset_charge_ion_exponent_clamp_branch_fraction"] == 0.25


def test_epstein_diffuse_fraction_interpolates_documented_endpoints() -> None:
    assert epstein_diffuse_factor(0.0) == 1.0
    assert epstein_diffuse_factor(1.0) == 1.0 + math.pi / 8.0
    assert epstein_diffuse_factor(0.9) == 1.0 + 0.9 * math.pi / 8.0
