import csv
import math
from pathlib import Path

from tools.vv.comsol.evaluate_aggregate_charge_parity import (
    REQUIRED_COLUMNS,
    evaluate_history,
)

from chamber_particles.physics.charge import ELECTRON_MASS_KG
from chamber_particles.physics.forces import (
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
)


def _reference_rate(row: dict[str, float]) -> float:
    radius = row["particle_radius_m"]
    potential = row["charge_number_e"] * row["local_single_charge_surface_potential_increment_V"]
    relative_speed_squared = (
        (row["local_ion_velocity_r_m_per_s"] - row["velocity_r_m_per_s"]) ** 2
        + (row["local_ion_velocity_z_m_per_s"] - row["velocity_z_m_per_s"]) ** 2
        + 8.0
        * ELEMENTARY_CHARGE_C
        * row["local_ion_thermal_energy_eV_as_V"]
        / (math.pi * row["effective_positive_ion_mass_kg"])
        + 1.0
    )
    ion_speed = math.sqrt(relative_speed_squared)
    ion_energy = max(
        row["effective_positive_ion_mass_kg"]
        * relative_speed_squared
        / (2.0 * ELEMENTARY_CHARGE_C),
        0.01,
    )
    ion_amplitude = math.pi * radius**2 * row["local_total_positive_ion_density_per_m3"] * ion_speed
    electron_amplitude = (
        math.pi
        * radius**2
        * row["local_electron_density_per_m3"]
        * math.sqrt(
            8.0
            * ELEMENTARY_CHARGE_C
            * row["electron_temperature_eV_as_V"]
            / (math.pi * ELECTRON_MASS_KG)
        )
    )
    if potential <= 0.0:
        ion_factor = 1.0 - potential / ion_energy
        electron_factor = math.exp(
            max(-50.0, min(50.0, potential / row["electron_temperature_eV_as_V"]))
        )
    else:
        ion_factor = math.exp(max(-50.0, min(50.0, -potential / ion_energy)))
        electron_factor = 1.0 + potential / row["electron_temperature_eV_as_V"]
    return ion_amplitude * ion_factor - electron_amplitude * electron_factor


def _row(*, active: float, particle_id: float, perturbation: float = 0.0) -> dict[str, float]:
    row = {
        "particle_id": particle_id,
        "time_s": 2.0e-5 * particle_id,
        "active_state_flag": active,
        "charge_number_e": -2.5,
        "particle_radius_m": 5.0e-8,
        "local_electron_density_per_m3": 1.8e14,
        "local_total_positive_ion_density_per_m3": 3.8e14,
        "electron_temperature_eV_as_V": 4.0,
        "effective_positive_ion_mass_kg": 8.3e-26,
        "local_ion_velocity_r_m_per_s": -10.0,
        "local_ion_velocity_z_m_per_s": -1.5e4,
        "velocity_r_m_per_s": 0.2,
        "velocity_z_m_per_s": 0.5,
        "local_ion_thermal_energy_eV_as_V": 0.026,
        "local_bounded_screening_length_m": 4.7e-4,
        "local_single_charge_surface_potential_increment_V": 0.0,
        "dynamic_charge_rate_dZdt_per_s": 0.0,
    }
    radius = row["particle_radius_m"]
    screening = max(radius, row["local_bounded_screening_length_m"])
    row["local_single_charge_surface_potential_increment_V"] = ELEMENTARY_CHARGE_C / (
        4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * radius * (1.0 + radius / screening)
    )
    row["dynamic_charge_rate_dZdt_per_s"] = _reference_rate(row) + perturbation
    return row


def _write(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_saved_primitive_evaluator_filters_inactive_rows_and_passes_formula(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    _write(
        history,
        [_row(active=1.0, particle_id=1.0), _row(active=0.0, particle_id=2.0, perturbation=1.0e20)],
    )

    metrics = evaluate_history(history).metrics

    assert metrics["active_rows"] == 1
    assert metrics["production_core_screening_gate"] == "PASS"
    assert metrics["exported_phi1_formula_gate"] == "PASS"
    normalized = metrics["production_core_current_scale_normalized_residual_max"]
    assert isinstance(normalized, float)
    assert normalized < 2.0e-15


def test_saved_primitive_evaluator_fails_a_material_rate_change(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    _write(history, [_row(active=1.0, particle_id=1.0, perturbation=1.0e6)])

    metrics = evaluate_history(history).metrics

    assert metrics["production_core_screening_gate"] == "FAIL"
    assert metrics["exported_phi1_formula_gate"] == "FAIL"
