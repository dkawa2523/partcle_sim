import csv
import math
from pathlib import Path

import pytest
from tools.vv.comsol.evaluate_aggregate_ion_drag_parity import evaluate_history

from chamber_particles.physics.forces import (
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
)


def _base_row() -> dict[str, float]:
    radius = 5.0e-8
    screening = 2.0e-5
    phi_one = ELEMENTARY_CHARGE_C / (
        4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * radius * (1.0 + radius / screening)
    )
    return {
        "particle_id": 1.0,
        "time_s": 2.0e-5,
        "active_state_flag": 1.0,
        "charge_number_e": -40.0,
        "particle_radius_m": radius,
        "particle_mass_kg": 2.0e-18,
        "velocity_r_m_per_s": 0.4,
        "velocity_z_m_per_s": -0.2,
        "ion_drag_force_r_N": 0.0,
        "ion_drag_force_z_N": 0.0,
        "local_total_positive_ion_density_per_m3": 8.0e14,
        "electron_temperature_eV_as_V": 3.0,
        "effective_positive_ion_mass_kg": 7.0e-26,
        "local_ion_velocity_r_m_per_s": 0.1,
        "local_ion_velocity_z_m_per_s": 0.2,
        "local_bounded_screening_length_m": screening,
        "local_ion_neutral_mean_free_path_m": 8.0e-6,
        "local_ion_thermal_energy_eV_as_V": 0.04,
        "local_single_charge_surface_potential_increment_V": phi_one,
        "local_electric_field_r_V_per_m": 80.0,
        "local_electric_field_z_V_per_m": -30.0,
        "local_sheath_potential_relative_to_bulk_V": -1.0,
    }


def _theory_force(row: dict[str, float]) -> tuple[float, float]:
    relative = (
        row["local_ion_velocity_r_m_per_s"] - row["velocity_r_m_per_s"],
        row["local_ion_velocity_z_m_per_s"] - row["velocity_z_m_per_s"],
    )
    ion_mass = row["effective_positive_ion_mass_kg"]
    speed_square = (
        relative[0] ** 2
        + relative[1] ** 2
        + 8.0 * ELEMENTARY_CHARGE_C * row["local_ion_thermal_energy_eV_as_V"] / (math.pi * ion_mass)
        + 1.0
    )
    radius = row["particle_radius_m"]
    screening = max(
        radius,
        min(
            row["local_bounded_screening_length_m"],
            row["local_ion_neutral_mean_free_path_m"],
        ),
    )
    charge = row["charge_number_e"]
    impact = (
        math.sqrt(charge**2 + 1.0e-20)
        * ELEMENTARY_CHARGE_C**2
        / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * ion_mass * speed_square)
    )
    potential = charge * row["local_single_charge_surface_potential_increment_V"]
    collection_square = min(
        screening**2,
        radius**2
        * max(0.0, 1.0 - 2.0 * ELEMENTARY_CHARGE_C * potential / (ion_mass * speed_square)),
    )
    logarithm = max(
        0.0,
        0.5 * math.log((screening**2 + impact**2) / (collection_square + impact**2)),
    )
    cross_section = math.pi * collection_square + 4.0 * math.pi * impact**2 * logarithm
    factor = (
        row["local_total_positive_ion_density_per_m3"]
        * ion_mass
        * math.sqrt(speed_square)
        * cross_section
    )
    return factor * relative[0], factor * relative[1]


def _image_force(row: dict[str, float], case_kind: str) -> tuple[float, float]:
    ion_mass = row["effective_positive_ion_mass_kg"]
    vector_square = (
        row["local_ion_velocity_r_m_per_s"] ** 2 + row["local_ion_velocity_z_m_per_s"] ** 2
    )
    if case_kind == "P":
        ion_speed = math.sqrt(vector_square + 1.0)
        speed_square = ion_speed**2 + 8.0 * ELEMENTARY_CHARGE_C * row[
            "local_ion_thermal_energy_eV_as_V"
        ] / (math.pi * ion_mass)
    else:
        ion_speed = math.sqrt(
            max(
                1.0e-6,
                ELEMENTARY_CHARGE_C
                * (
                    row["electron_temperature_eV_as_V"]
                    - 2.0 * row["local_sheath_potential_relative_to_bulk_V"]
                )
                / ion_mass,
            )
        )
        speed_square = (
            ion_speed**2
            + 8.0
            * ELEMENTARY_CHARGE_C
            * row["local_ion_thermal_energy_eV_as_V"]
            / (math.pi * ion_mass)
            + 1.0
        )
    radius = row["particle_radius_m"]
    charge = row["charge_number_e"]
    potential = charge * row["local_single_charge_surface_potential_increment_V"]
    collection = (
        math.pi * radius**2 * max(0.0, 1.0 - potential / row["local_ion_thermal_energy_eV_as_V"])
    )
    impact = (
        ELEMENTARY_CHARGE_C**2
        * charge
        / (2.0 * math.pi * VACUUM_PERMITTIVITY_F_M * ion_mass * speed_square)
    )
    image_screening = math.sqrt(
        VACUUM_PERMITTIVITY_F_M
        * row["electron_temperature_eV_as_V"]
        / (ELEMENTARY_CHARGE_C * row["local_total_positive_ion_density_per_m3"])
    )
    orbital = math.pi * impact**2 * math.log(max(1.0 + 1.0e-12, image_screening / radius))
    magnitude = (
        row["local_total_positive_ion_density_per_m3"]
        * ion_mass
        * math.sqrt(speed_square)
        * ion_speed
        * (collection + orbital)
    )
    electric_norm = math.sqrt(
        row["local_electric_field_r_V_per_m"] ** 2
        + row["local_electric_field_z_V_per_m"] ** 2
        + 1.0
    )
    return (
        magnitude * row["local_electric_field_r_V_per_m"] / electric_norm,
        magnitude * row["local_electric_field_z_V_per_m"] / electric_norm,
    )


def _write_history(root: Path, revision: str, case_name: str, row: dict[str, float]) -> Path:
    path = (
        root
        / "cases"
        / revision
        / case_name
        / "external_reproduction"
        / "results"
        / "particle_history_full_tidy.csv"
    )
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def test_theory_native_and_canonical_formula_replay_pass(tmp_path: Path) -> None:
    row = _base_row()
    force = _theory_force(row)
    row["ion_drag_force_r_N"], row["ion_drag_force_z_N"] = force
    history = _write_history(tmp_path, "formal_iondrag_theory_consistent", "caseP_30nm", row)

    metrics = evaluate_history(history).metrics

    assert metrics["native_saved_formula_replay_status"] == "PASS"
    assert metrics["production_canonical_comparison_status"] == "PASS"
    native_residual = metrics["native_saved_formula_force_scale_normalized_residual_max"]
    assert isinstance(native_residual, float)
    assert native_residual < 1.0e-13


@pytest.mark.parametrize("case_kind", ["P", "A"])
def test_image_saved_speed_authority_is_separate_from_canonical_comparison(
    tmp_path: Path, case_kind: str
) -> None:
    row = _base_row()
    force = _image_force(row, case_kind)
    row["ion_drag_force_r_N"], row["ion_drag_force_z_N"] = force
    history = _write_history(
        tmp_path,
        "formal_iondrag_image_minimal_corrected",
        f"case{case_kind}_30nm",
        row,
    )

    metrics = evaluate_history(history).metrics

    assert metrics["native_saved_formula_replay_status"] == "PASS"
    assert (
        metrics["production_canonical_comparison_status"]
        == "DOCUMENTED_MODEL_DEFINITION_DIFFERENCE"
    )
    canonical_residual = metrics["production_canonical_force_scale_normalized_residual_max"]
    assert isinstance(canonical_residual, float)
    assert canonical_residual > 1.0e-3
    assert "differs" in str(metrics["model_definition_difference"])
