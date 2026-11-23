"""Compare supplied active-state charge rates with the current aggregate model.

This external V&V tool uses primitive rows for a frozen-state formula comparison,
not solver input and not a golden definition
of particle charging or trajectory accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from chamber_particles.physics.charge import (
    AGGREGATE_EXPONENT_MAX,
    AGGREGATE_EXPONENT_MIN,
    ELECTRON_MASS_KG,
    aggregate_relative_drift_regularized_two_current_v1,
)
from chamber_particles.physics.forces import ELEMENTARY_CHARGE_C, VACUUM_PERMITTIVITY_F_M
from tools.vv.comsol._frozen_state_csv import read_active_states

type FloatArray = NDArray[np.float64]
type Record = dict[str, object]

TOOL_REVISION: Final = "aggregate_charge_frozen_state_v3"
MODEL_REVISION: Final = "aggregate_relative_drift_regularized_two_current_v1"
REQUIRED_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "active_state_flag",
    "charge_number_e",
    "particle_radius_m",
    "local_electron_density_per_m3",
    "local_total_positive_ion_density_per_m3",
    "electron_temperature_eV_as_V",
    "effective_positive_ion_mass_kg",
    "local_ion_velocity_r_m_per_s",
    "local_ion_velocity_z_m_per_s",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "local_ion_thermal_energy_eV_as_V",
    "local_bounded_screening_length_m",
    "local_single_charge_surface_potential_increment_V",
    "dynamic_charge_rate_dZdt_per_s",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_rates(
    values: dict[str, FloatArray],
    surface_potential_V: FloatArray,
    effective_ion_speed_m_s: FloatArray,
    effective_ion_energy_V: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    radius = values["particle_radius_m"]
    ion_amplitude = (
        math.pi
        * radius**2
        * values["local_total_positive_ion_density_per_m3"]
        * effective_ion_speed_m_s
    )
    electron_amplitude = (
        math.pi
        * radius**2
        * values["local_electron_density_per_m3"]
        * np.sqrt(
            8.0
            * ELEMENTARY_CHARGE_C
            * values["electron_temperature_eV_as_V"]
            / (math.pi * ELECTRON_MASS_KG)
        )
    )
    nonpositive = surface_potential_V <= 0.0
    electron_argument = surface_potential_V / values["electron_temperature_eV_as_V"]
    ion_argument = -surface_potential_V / effective_ion_energy_V
    electron_factor = np.where(
        nonpositive,
        np.exp(np.clip(electron_argument, AGGREGATE_EXPONENT_MIN, AGGREGATE_EXPONENT_MAX)),
        1.0 + electron_argument,
    )
    ion_factor = np.where(
        nonpositive,
        1.0 + ion_argument,
        np.exp(np.clip(ion_argument, AGGREGATE_EXPONENT_MIN, AGGREGATE_EXPONENT_MAX)),
    )
    return ion_amplitude * ion_factor, electron_amplitude * electron_factor


def evaluate_history(
    history_path: Path,
    *,
    current_scale_residual_limit: float,
) -> Record:
    """Compare one explicit CSV against the current frozen-state charge equation."""

    if not math.isfinite(current_scale_residual_limit) or current_scale_residual_limit <= 0.0:
        raise ValueError("current_scale_residual_limit must be finite and positive")
    path = history_path.expanduser().resolve()
    particle_ids, values = read_active_states(path, REQUIRED_COLUMNS)
    potential_increment = "local_single_charge_surface_potential_increment_V"
    if np.any(values[potential_increment] <= 0.0):
        raise ValueError(f"{path}: {potential_increment} must be positive")
    particle_velocity = np.column_stack(
        (values["velocity_r_m_per_s"], values["velocity_z_m_per_s"])
    )
    ion_velocity = np.column_stack(
        (
            values["local_ion_velocity_r_m_per_s"],
            values["local_ion_velocity_z_m_per_s"],
        )
    )
    relative_speed = np.hypot(
        ion_velocity[:, 0] - particle_velocity[:, 0],
        ion_velocity[:, 1] - particle_velocity[:, 1],
    )
    sampled_speed_limit = float(np.nextafter(np.max(relative_speed), math.inf))
    if sampled_speed_limit == 0.0:
        sampled_speed_limit = float(np.nextafter(0.0, math.inf))
    evaluation = aggregate_relative_drift_regularized_two_current_v1(
        charge_number=values["charge_number_e"],
        electrostatic_radius_m=values["particle_radius_m"],
        electron_number_density_m3=values["local_electron_density_per_m3"],
        positive_ion_number_density_m3=values["local_total_positive_ion_density_per_m3"],
        electron_thermal_voltage_V=values["electron_temperature_eV_as_V"],
        positive_ion_thermal_voltage_V=values["local_ion_thermal_energy_eV_as_V"],
        particle_velocity_m_s=particle_velocity,
        positive_ion_velocity_m_s=ion_velocity,
        effective_positive_ion_mass_kg=values["effective_positive_ion_mass_kg"],
        screening_length_m=values["local_bounded_screening_length_m"],
        maximum_relative_ion_speed_m_s=sampled_speed_limit,
    )
    exported = values["dynamic_charge_rate_dZdt_per_s"]
    core_residual = evaluation.charge_rate_number_s - exported
    core_ion_rate, core_electron_rate = _collection_rates(
        values,
        evaluation.surface_potential_V,
        evaluation.effective_ion_speed_m_s,
        evaluation.effective_ion_energy_V,
    )
    core_scale = np.abs(core_ion_rate) + np.abs(core_electron_rate)
    core_normalized = np.abs(core_residual) / np.maximum(core_scale, 1.0)

    exported_phi1 = values["local_single_charge_surface_potential_increment_V"]
    formula_surface_potential = values["charge_number_e"] * exported_phi1
    formula_ion_rate, formula_electron_rate = _collection_rates(
        values,
        formula_surface_potential,
        evaluation.effective_ion_speed_m_s,
        evaluation.effective_ion_energy_V,
    )
    formula_rate = formula_ion_rate - formula_electron_rate
    formula_residual = formula_rate - exported
    formula_scale = np.abs(formula_ion_rate) + np.abs(formula_electron_rate)
    formula_normalized = np.abs(formula_residual) / np.maximum(formula_scale, 1.0)

    radius = values["particle_radius_m"]
    effective_screening = np.maximum(radius, values["local_bounded_screening_length_m"])
    core_phi1 = ELEMENTARY_CHARGE_C / evaluation.capacitance_F
    inferred_epsilon = ELEMENTARY_CHARGE_C / (
        4.0 * math.pi * radius * (1.0 + radius / effective_screening) * exported_phi1
    )
    epsilon_relative_difference = (
        inferred_epsilon - VACUUM_PERMITTIVITY_F_M
    ) / VACUUM_PERMITTIVITY_F_M
    potential_increment_relative_difference = (core_phi1 - exported_phi1) / exported_phi1
    finite = bool(
        np.isfinite(core_residual).all()
        and np.isfinite(formula_residual).all()
        and np.isfinite(core_normalized).all()
        and np.isfinite(formula_normalized).all()
        and np.isfinite(epsilon_relative_difference).all()
    )
    core_worst = int(np.argmax(core_normalized))
    formula_worst = int(np.argmax(formula_normalized))
    core_relative_l2 = float(
        np.linalg.norm(core_residual)
        / max(float(np.linalg.norm(exported)), float(np.finfo(np.float64).tiny))
    )
    formula_relative_l2 = float(
        np.linalg.norm(formula_residual)
        / max(float(np.linalg.norm(exported)), float(np.finfo(np.float64).tiny))
    )
    core_pass = bool(
        finite
        and evaluation.applicable.all()
        and core_normalized[core_worst] <= current_scale_residual_limit
    )
    formula_pass = bool(
        finite and formula_normalized[formula_worst] <= current_scale_residual_limit
    )
    metrics: Record = {
        "path": str(path),
        "sha256": _sha256(path),
        "active_rows": int(exported.size),
        "finite": finite,
        "all_rows_within_sampled_speed_envelope": bool(evaluation.applicable.all()),
        "sampled_maximum_relative_ion_speed_m_s": float(np.max(relative_speed)),
        "production_core_screening_gate": "PASS" if core_pass else "FAIL",
        "production_core_charge_rate_global_relative_l2_residual": core_relative_l2,
        "production_core_current_scale_normalized_residual_max": float(core_normalized[core_worst]),
        "exported_phi1_formula_gate": "PASS" if formula_pass else "FAIL",
        "exported_phi1_charge_rate_global_relative_l2_residual": formula_relative_l2,
        "exported_phi1_current_scale_normalized_residual_max": float(
            formula_normalized[formula_worst]
        ),
        "inferred_epsilon0_relative_difference_median": float(
            np.median(epsilon_relative_difference)
        ),
        "core_vs_exported_phi1_relative_difference_median": float(
            np.median(potential_increment_relative_difference)
        ),
        "production_core_worst_row": {
            "particle_id": int(particle_ids[core_worst]),
            "time_s": float(values["time_s"][core_worst]),
        },
    }
    return {
        "tool_revision": TOOL_REVISION,
        "model_revision": MODEL_REVISION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evidence_role": "external_frozen_state_charge_comparison",
        "trajectory_accuracy": "NOT_TESTED",
        "physical_validity": "NOT_TESTED",
        "current_scale_residual_limit": current_scale_residual_limit,
        "metrics": metrics,
        "interpretation": (
            "The production comparison derives capacitance from the supplied screening "
            "primitive and current constants. The separate exported-potential-increment "
            "comparison diagnoses constant-convention differences at these same frozen "
            "states. Neither establishes integrated charge or trajectory accuracy. The "
            "sampled speed envelope is not a global applicability certificate."
            " The normalized current residual is abs(predicted-exported) / "
            "max(abs(ion_rate)+abs(electron_rate), 1 s^-1), using charge-number rates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Required CSV columns: "
            + ", ".join(REQUIRED_COLUMNS)
            + ". Units follow the column suffixes; charge is in elementary-charge numbers. "
            "The single-charge surface-potential increment must be positive."
        ),
    )
    parser.add_argument("history_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--current-scale-residual-limit",
        type=float,
        required=True,
    )
    args = parser.parse_args()
    report = evaluate_history(
        args.history_path, current_scale_residual_limit=args.current_scale_residual_limit
    )
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    metrics = report["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("charge metrics must be a mapping")
    return 0 if metrics["production_core_screening_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
