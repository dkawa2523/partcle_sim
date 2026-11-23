"""Compare one explicit native force CSV with a current aggregate ion-drag model.

The caller declares equation correspondence. Raw residuals remain available for
an analogous equation; those residuals do not receive a parity verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from chamber_particles.physics.forces import (
    electric_field_directed_image_orbital_ion_drag,
    relative_flow_screened_collection_orbital_ion_drag,
)
from tools.vv.comsol._frozen_state_csv import read_active_states

type FloatArray = NDArray[np.float64]
type Record = dict[str, object]
type FormulaRevision = Literal[
    "relative_flow_screened_collection_orbital_aggregate_ion_v1",
    "electric_field_directed_image_orbital_sensitivity_v1",
]
type Correspondence = Literal["matched", "analogous"]

TOOL_REVISION: Final = "aggregate_ion_drag_frozen_state_v2"
MODEL_REVISIONS: Final = (
    "relative_flow_screened_collection_orbital_aggregate_ion_v1",
    "electric_field_directed_image_orbital_sensitivity_v1",
)
COMMON_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "active_state_flag",
    "charge_number_e",
    "particle_radius_m",
    "particle_mass_kg",
    "ion_drag_force_r_N",
    "ion_drag_force_z_N",
    "local_total_positive_ion_density_per_m3",
    "effective_positive_ion_mass_kg",
    "local_ion_velocity_r_m_per_s",
    "local_ion_velocity_z_m_per_s",
    "local_bounded_screening_length_m",
    "local_ion_thermal_energy_eV_as_V",
)
RELATIVE_COLUMNS: Final = (
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "local_ion_neutral_mean_free_path_m",
)
IMAGE_COLUMNS: Final = (
    "electron_temperature_eV_as_V",
    "local_electric_field_r_V_per_m",
    "local_electric_field_z_V_per_m",
)


def _canonical_force(
    values: dict[str, FloatArray],
    revision: FormulaRevision,
) -> FloatArray:
    mass = values["particle_mass_kg"]
    radius = values["particle_radius_m"]
    charge = values["charge_number_e"]
    ion_velocity = np.column_stack(
        (values["local_ion_velocity_r_m_per_s"], values["local_ion_velocity_z_m_per_s"])
    )
    common = {
        "mass_kg": mass,
        "electrostatic_radius_m": radius,
        "charge_number": charge,
        "positive_ion_number_density_m3": values["local_total_positive_ion_density_per_m3"],
        "positive_ion_thermal_voltage_V": values["local_ion_thermal_energy_eV_as_V"],
        "positive_ion_velocity_m_s": ion_velocity,
        "effective_positive_ion_mass_kg": values["effective_positive_ion_mass_kg"],
        "screening_length_m": values["local_bounded_screening_length_m"],
    }
    if revision == "relative_flow_screened_collection_orbital_aggregate_ion_v1":
        particle_velocity = np.column_stack(
            (values["velocity_r_m_per_s"], values["velocity_z_m_per_s"])
        )
        sampled_maximum = float(np.max(np.linalg.norm(ion_velocity - particle_velocity, axis=1)))
        speed_limit = float(np.nextafter(sampled_maximum, math.inf))
        if speed_limit == 0.0:
            speed_limit = float(np.nextafter(0.0, math.inf))
        evaluation = relative_flow_screened_collection_orbital_ion_drag(
            **common,
            velocity_m_s=particle_velocity,
            ion_neutral_mean_free_path_m=values["local_ion_neutral_mean_free_path_m"],
            maximum_relative_ion_speed_m_s=speed_limit,
        )
    else:
        electric = np.column_stack(
            (
                values["local_electric_field_r_V_per_m"],
                values["local_electric_field_z_V_per_m"],
            )
        )
        evaluation = electric_field_directed_image_orbital_ion_drag(
            **common,
            electron_thermal_voltage_V=values["electron_temperature_eV_as_V"],
            electric_field_V_m=electric,
        )
    return evaluation.acceleration_m_s2 * mass[:, None]


def evaluate_history(
    history_path: Path,
    *,
    model_revision: FormulaRevision,
    correspondence: Correspondence,
    force_scale_residual_limit: float,
) -> Record:
    """Evaluate the declared current formula at the supplied finite active states."""

    if model_revision not in MODEL_REVISIONS:
        raise ValueError("model_revision must select a current aggregate ion-drag model")
    if correspondence not in ("matched", "analogous"):
        raise ValueError("correspondence must be matched or analogous")
    if not math.isfinite(force_scale_residual_limit) or force_scale_residual_limit <= 0.0:
        raise ValueError("force_scale_residual_limit must be finite and positive")
    path = history_path.expanduser().resolve()
    required = COMMON_COLUMNS + (
        RELATIVE_COLUMNS
        if model_revision == "relative_flow_screened_collection_orbital_aggregate_ion_v1"
        else IMAGE_COLUMNS
    )
    particle_ids, values = read_active_states(path, required)
    exported = np.column_stack((values["ion_drag_force_r_N"], values["ion_drag_force_z_N"]))
    canonical = _canonical_force(values, model_revision)
    residual = canonical - exported
    scale = np.maximum(
        np.maximum(np.linalg.norm(canonical, axis=1), np.linalg.norm(exported, axis=1)),
        np.finfo(np.float64).tiny,
    )
    normalized = np.linalg.norm(residual, axis=1) / scale
    if not bool(np.isfinite(canonical).all() and np.isfinite(normalized).all()):
        raise ValueError("force comparison produced nonfinite values")
    worst = int(np.argmax(normalized))
    within_allowance = bool(normalized[worst] <= force_scale_residual_limit)
    status = (
        ("PASS" if within_allowance else "FAIL") if correspondence == "matched" else "NOT_TESTED"
    )
    return {
        "tool_revision": TOOL_REVISION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_revision": model_revision,
        "model_correspondence": correspondence,
        "active_rows": int(exported.shape[0]),
        "force_scale_residual_limit": force_scale_residual_limit,
        "frozen_force_comparison_status": status,
        "residual_within_allowance": within_allowance,
        "force_global_relative_l2_residual": float(
            np.linalg.norm(residual)
            / max(float(np.linalg.norm(exported)), float(np.finfo(np.float64).tiny))
        ),
        "force_scale_normalized_residual_max": float(normalized[worst]),
        "worst_row": {
            "particle_id": int(particle_ids[worst]),
            "time_s": float(values["time_s"][worst]),
        },
        "trajectory_accuracy": "NOT_TESTED",
        "physical_validity": "NOT_TESTED",
        "interpretation": (
            "The explicitly selected current model is evaluated at the supplied frozen "
            "states; source filenames and directory names select no equations. The "
            "matched/analogous declaration is caller evidence, not native authenticity "
            "proof. Analogous comparisons retain raw residuals without a parity verdict. "
            "A sampled speed envelope is not a global applicability certificate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Required CSV columns: "
            + ", ".join(COMMON_COLUMNS)
            + ". Relative-flow model additionally requires: "
            + ", ".join(RELATIVE_COLUMNS)
            + ". Image model additionally requires: "
            + ", ".join(IMAGE_COLUMNS)
            + ". Units follow the column suffixes; charge is in elementary-charge numbers."
        ),
    )
    parser.add_argument("history_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model-revision", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--correspondence", choices=("matched", "analogous"), required=True)
    parser.add_argument("--force-scale-residual-limit", type=float, required=True)
    args = parser.parse_args()
    report = evaluate_history(
        args.history_path,
        model_revision=args.model_revision,
        correspondence=args.correspondence,
        force_scale_residual_limit=args.force_scale_residual_limit,
    )
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return 1 if report["frozen_force_comparison_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
