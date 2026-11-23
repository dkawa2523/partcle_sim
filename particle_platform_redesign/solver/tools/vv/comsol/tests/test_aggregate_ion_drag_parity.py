import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from tools.vv.comsol.evaluate_aggregate_ion_drag_parity import (
    Correspondence,
    FormulaRevision,
    evaluate_history,
)

from chamber_particles.physics.forces import (
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
)


def _base_row() -> dict[str, float]:
    radius = 5.0e-8
    screening = 2.0e-5
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
        "local_electric_field_r_V_per_m": 80.0,
        "local_electric_field_z_V_per_m": -30.0,
    }


def _surface_potential(row: dict[str, float]) -> float:
    radius = row["particle_radius_m"]
    screening = max(radius, row["local_bounded_screening_length_m"])
    return (
        row["charge_number_e"]
        * ELEMENTARY_CHARGE_C
        / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * radius * (1.0 + radius / screening))
    )


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
    potential = _surface_potential(row)
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


def _image_force(row: dict[str, float]) -> tuple[float, float]:
    ion_mass = row["effective_positive_ion_mass_kg"]
    vector_square = (
        row["local_ion_velocity_r_m_per_s"] ** 2 + row["local_ion_velocity_z_m_per_s"] ** 2
    )
    ion_speed = math.sqrt(vector_square)
    speed_square = (
        vector_square
        + 8.0 * ELEMENTARY_CHARGE_C * row["local_ion_thermal_energy_eV_as_V"] / (math.pi * ion_mass)
        + 1.0
    )
    radius = row["particle_radius_m"]
    charge = row["charge_number_e"]
    potential = _surface_potential(row)
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


def _write_history(path: Path, row: dict[str, float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    return path


@pytest.mark.parametrize(
    "model",
    [
        "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        "electric_field_directed_image_orbital_sensitivity_v1",
    ],
)
def test_current_formula_matches_independent_force_at_an_arbitrary_path(
    tmp_path: Path, model: FormulaRevision
) -> None:
    row = _base_row()
    force = _theory_force(row) if model.startswith("relative_flow") else _image_force(row)
    row["ion_drag_force_r_N"], row["ion_drag_force_z_N"] = force
    history = _write_history(tmp_path / "measurements" / "unrelated_name.csv", row)
    metrics = evaluate_history(
        history, model_revision=model, correspondence="matched", force_scale_residual_limit=1.0e-12
    )
    assert metrics["frozen_force_comparison_status"] == "PASS"
    residual = metrics["force_scale_normalized_residual_max"]
    assert isinstance(residual, float) and residual < 1.0e-13
    assert metrics["trajectory_accuracy"] == "NOT_TESTED"


@pytest.mark.parametrize(
    "correspondence,expected", [("matched", "FAIL"), ("analogous", "NOT_TESTED")]
)
def test_mismatched_force_preserves_raw_residual_with_declared_meaning(
    tmp_path: Path, correspondence: Correspondence, expected: str
) -> None:
    row = _base_row()
    row["ion_drag_force_r_N"], row["ion_drag_force_z_N"] = _image_force(row)
    row["ion_drag_force_r_N"] *= 2.0
    history = _write_history(tmp_path / "native.csv", row)
    report = evaluate_history(
        history,
        model_revision="electric_field_directed_image_orbital_sensitivity_v1",
        correspondence=correspondence,
        force_scale_residual_limit=1.0e-12,
    )
    assert report["frozen_force_comparison_status"] == expected
    assert report["residual_within_allowance"] is False
    residual = report["force_scale_normalized_residual_max"]
    assert isinstance(residual, float) and residual > 1.0e-3


@pytest.mark.parametrize("particle_id", [2**53 + 1, -(2**63), 2**63 - 1])
def test_ion_diagnostic_preserves_exact_int64_particle_identity(
    tmp_path: Path, particle_id: int
) -> None:
    row = _base_row()
    row["particle_id"] = particle_id
    row["ion_drag_force_r_N"], row["ion_drag_force_z_N"] = _image_force(row)
    history = _write_history(tmp_path / "exact_id.csv", row)
    report = evaluate_history(
        history,
        model_revision="electric_field_directed_image_orbital_sensitivity_v1",
        correspondence="matched",
        force_scale_residual_limit=1.0e-12,
    )
    assert report["worst_row"] == {"particle_id": particle_id, "time_s": row["time_s"]}


@pytest.mark.parametrize("particle_id", [1.5, 2**63, -(2**63) - 1, math.inf])
def test_ion_rejects_inexact_or_out_of_range_particle_identity(
    tmp_path: Path, particle_id: float
) -> None:
    row = _base_row()
    row["particle_id"] = particle_id
    history = _write_history(tmp_path / "invalid_id.csv", row)
    with pytest.raises(ValueError, match="particle_id"):
        evaluate_history(
            history,
            model_revision="electric_field_directed_image_orbital_sensitivity_v1",
            correspondence="matched",
            force_scale_residual_limit=1.0e-12,
        )


def test_explicit_ion_csv_cli_selects_current_model_and_writes_report(tmp_path: Path) -> None:
    row = _base_row()
    row["particle_id"] = 2**53 + 1
    row["ion_drag_force_r_N"], row["ion_drag_force_z_N"] = _image_force(row)
    history = _write_history(tmp_path / "input.csv", row)
    output = tmp_path / "force.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.vv.comsol.evaluate_aggregate_ion_drag_parity",
            str(history),
            str(output),
            "--model-revision",
            "electric_field_directed_image_orbital_sensitivity_v1",
            "--correspondence",
            "matched",
            "--force-scale-residual-limit",
            "1e-12",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["frozen_force_comparison_status"] == "PASS"
    assert report["model_correspondence"] == "matched"
    assert report["worst_row"]["particle_id"] == 2**53 + 1
