"""Replay saved active-row charge rates with the production aggregate model.

This is an external V&V tool.  The saved provider histories are evidence for a
frozen-state formula comparison, not solver input and not a golden definition
of particle charging or trajectory accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
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

type FloatArray = NDArray[np.float64]
type Record = dict[str, object]

TOOL_REVISION: Final = "p18c_saved_primitive_charge_parity_v2"
MODEL_REVISION: Final = "aggregate_relative_drift_regularized_two_current_v1"
DEFAULT_CURRENT_SCALE_RESIDUAL_LIMIT: Final = 1.0e-10
DEFAULT_EXPECTED_PACKAGES: Final = 12
HISTORY_RELATIVE_PATTERN: Final = (
    "cases/*/*/external_reproduction/results/particle_history_full_tidy.csv"
)
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
COLUMN_ALIASES: Final = {
    "electron_temperature_eV_as_V": ("local_electron_temperature_eV_as_V",),
    "effective_positive_ion_mass_kg": ("local_effective_positive_ion_mass_kg",),
    "local_bounded_screening_length_m": ("local_screening_length_m",),
}


@dataclass(frozen=True, slots=True)
class HistoryEvaluation:
    """One package's public metrics and arrays needed for dataset aggregation."""

    metrics: Record
    core_residual: FloatArray
    formula_residual: FloatArray
    core_normalized_residual: FloatArray
    formula_normalized_residual: FloatArray
    exported_rate: FloatArray
    epsilon_relative_difference: FloatArray
    potential_increment_relative_difference: FloatArray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_columns(fieldnames: Sequence[str] | None, path: Path) -> dict[str, str]:
    if fieldnames is None:
        raise ValueError(f"{path}: missing CSV header")
    available = set(fieldnames)
    sources = {
        name: next(
            (
                candidate
                for candidate in (name, *COLUMN_ALIASES.get(name, ()))
                if candidate in available
            ),
            "",
        )
        for name in REQUIRED_COLUMNS
    }
    missing = sorted(name for name, source in sources.items() if not source)
    if missing:
        raise ValueError(f"{path}: required columns are missing: {missing}")
    return sources


def _active_numeric_row(
    row: dict[str, str], sources: dict[str, str], path: Path, row_number: int
) -> dict[str, float] | None:
    active = float(row[sources["active_state_flag"]])
    if active not in (0.0, 1.0):
        raise ValueError(f"{path}:{row_number}: active_state_flag is not 0 or 1")
    if active == 0.0:
        return None
    result: dict[str, float] = {}
    for name in REQUIRED_COLUMNS:
        value = float(row[sources[name]])
        if not math.isfinite(value):
            raise ValueError(f"{path}:{row_number}: {name} is not finite")
        result[name] = value
    return result


def _read_active(path: Path) -> dict[str, FloatArray]:
    values: dict[str, list[float]] = {name: [] for name in REQUIRED_COLUMNS}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        sources = _resolve_columns(reader.fieldnames, path)
        for row_number, row in enumerate(reader, start=2):
            numeric = _active_numeric_row(row, sources, path, row_number)
            if numeric is not None:
                for name, value in numeric.items():
                    values[name].append(value)
    arrays = {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}
    if arrays["particle_id"].size == 0:
        raise ValueError(f"{path}: no active rows")
    return arrays


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
    current_scale_residual_limit: float = DEFAULT_CURRENT_SCALE_RESIDUAL_LIMIT,
) -> HistoryEvaluation:
    """Evaluate one saved history and return metrics plus aggregate arrays."""

    if not math.isfinite(current_scale_residual_limit) or current_scale_residual_limit <= 0.0:
        raise ValueError("current_scale_residual_limit must be finite and positive")
    path = history_path.expanduser().resolve()
    values = _read_active(path)
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
            "particle_id": int(values["particle_id"][core_worst]),
            "time_s": float(values["time_s"][core_worst]),
        },
        "exact_provider_parity": "PASS" if core_pass else "FAIL",
        "model_formula_mismatch": "NOT_INDICATED" if formula_pass else "INDICATED",
    }
    return HistoryEvaluation(
        metrics,
        core_residual,
        formula_residual,
        core_normalized,
        formula_normalized,
        exported,
        epsilon_relative_difference,
        potential_increment_relative_difference,
    )


def evaluate_dataset(
    dataset_root: Path,
    *,
    expected_packages: int = DEFAULT_EXPECTED_PACKAGES,
    current_scale_residual_limit: float = DEFAULT_CURRENT_SCALE_RESIDUAL_LIMIT,
) -> Record:
    """Evaluate every discovered history without modifying or rerunning it."""

    root = dataset_root.expanduser().resolve()
    histories = tuple(sorted(root.glob(HISTORY_RELATIVE_PATTERN)))
    if len(histories) != expected_packages:
        raise ValueError(f"expected {expected_packages} history packages, found {len(histories)}")
    package_metrics: list[Record] = []
    core_residual_parts: list[FloatArray] = []
    formula_residual_parts: list[FloatArray] = []
    core_normalized_parts: list[FloatArray] = []
    formula_normalized_parts: list[FloatArray] = []
    exported_parts: list[FloatArray] = []
    epsilon_difference_parts: list[FloatArray] = []
    potential_increment_difference_parts: list[FloatArray] = []
    for history in histories:
        result = evaluate_history(
            history,
            current_scale_residual_limit=current_scale_residual_limit,
        )
        metrics = result.metrics
        metrics["package"] = history.relative_to(root).as_posix()
        metrics.pop("path")
        package_metrics.append(metrics)
        core_residual_parts.append(result.core_residual)
        formula_residual_parts.append(result.formula_residual)
        core_normalized_parts.append(result.core_normalized_residual)
        formula_normalized_parts.append(result.formula_normalized_residual)
        exported_parts.append(result.exported_rate)
        epsilon_difference_parts.append(result.epsilon_relative_difference)
        potential_increment_difference_parts.append(result.potential_increment_relative_difference)
    core_residual = np.concatenate(core_residual_parts)
    formula_residual = np.concatenate(formula_residual_parts)
    core_normalized = np.concatenate(core_normalized_parts)
    formula_normalized = np.concatenate(formula_normalized_parts)
    exported = np.concatenate(exported_parts)
    epsilon_difference = np.concatenate(epsilon_difference_parts)
    potential_increment_difference = np.concatenate(potential_increment_difference_parts)
    core_pass = all(
        metrics["production_core_screening_gate"] == "PASS" for metrics in package_metrics
    )
    formula_pass = all(
        metrics["exported_phi1_formula_gate"] == "PASS" for metrics in package_metrics
    )
    convention_diagnosis = (
        "EPSILON0_CONSTANT_CONVENTION_DIFFERENCE"
        if formula_pass and not core_pass
        else "NO_SEPARATE_CONSTANT_CONVENTION_DIAGNOSIS"
    )
    return {
        "tool_revision": TOOL_REVISION,
        "model_revision": MODEL_REVISION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evidence_role": "external_saved_primitive_frozen_rate_parity",
        "golden_truth": "NOT_CLAIMED",
        "trajectory_accuracy": "NOT_TESTED",
        "comsol_rerun": False,
        "dataset_root": str(root),
        "current_scale_residual_limit": current_scale_residual_limit,
        "summary": {
            "packages": len(package_metrics),
            "active_rows": int(exported.size),
            "production_core_screening_gate": "PASS" if core_pass else "FAIL",
            "production_core_charge_rate_absolute_residual_max_number_s": float(
                np.max(np.abs(core_residual))
            ),
            "production_core_charge_rate_global_relative_l2_residual": float(
                np.linalg.norm(core_residual)
                / max(float(np.linalg.norm(exported)), float(np.finfo(np.float64).tiny))
            ),
            "production_core_current_scale_normalized_residual_p90": float(
                np.percentile(core_normalized, 90.0)
            ),
            "production_core_current_scale_normalized_residual_p99": float(
                np.percentile(core_normalized, 99.0)
            ),
            "production_core_current_scale_normalized_residual_max": float(np.max(core_normalized)),
            "exported_phi1_formula_gate": "PASS" if formula_pass else "FAIL",
            "exported_phi1_charge_rate_global_relative_l2_residual": float(
                np.linalg.norm(formula_residual)
                / max(float(np.linalg.norm(exported)), float(np.finfo(np.float64).tiny))
            ),
            "exported_phi1_current_scale_normalized_residual_p90": float(
                np.percentile(formula_normalized, 90.0)
            ),
            "exported_phi1_current_scale_normalized_residual_p99": float(
                np.percentile(formula_normalized, 99.0)
            ),
            "exported_phi1_current_scale_normalized_residual_max": float(
                np.max(formula_normalized)
            ),
            "inferred_epsilon0_relative_difference_median": float(np.median(epsilon_difference)),
            "inferred_epsilon0_relative_difference_min": float(np.min(epsilon_difference)),
            "inferred_epsilon0_relative_difference_max": float(np.max(epsilon_difference)),
            "core_vs_exported_phi1_relative_difference_median": float(
                np.median(potential_increment_difference)
            ),
            "constant_convention_diagnosis": convention_diagnosis,
            "model_formula_mismatch": "NOT_INDICATED" if formula_pass else "INDICATED",
            "exact_provider_parity": "PASS" if core_pass else "FAIL",
        },
        "packages": package_metrics,
        "interpretation": (
            "The production-core gate derives phi1 from screening with the core epsilon0 and "
            "therefore tests exact provider parity. The separate exported-phi1 gate tests the "
            "saved two-current formula without changing core constants. A PASS of only the "
            "second gate diagnoses a constant-convention difference, not a model-form match "
            "claim beyond these frozen states. The sampled speed envelope is not a production "
            "applicability certificate, and no integrated charge or trajectory claim is made."
        ),
    }


def write_report(report: Record, output_directory: Path) -> None:
    """Write a no-clobber JSON report and compact package table."""

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8", errors="strict") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    package_rows = report["packages"]
    if not isinstance(package_rows, list):
        raise TypeError("report packages must be a list")
    columns = (
        "package",
        "sha256",
        "active_rows",
        "production_core_screening_gate",
        "production_core_charge_rate_global_relative_l2_residual",
        "production_core_current_scale_normalized_residual_max",
        "exported_phi1_formula_gate",
        "exported_phi1_charge_rate_global_relative_l2_residual",
        "exported_phi1_current_scale_normalized_residual_max",
        "inferred_epsilon0_relative_difference_median",
        "sampled_maximum_relative_ion_speed_m_s",
    )
    with (output / "package_metrics.csv").open(
        "x", encoding="utf-8", errors="strict", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(package_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-packages", type=int, default=DEFAULT_EXPECTED_PACKAGES)
    parser.add_argument(
        "--current-scale-residual-limit",
        type=float,
        default=DEFAULT_CURRENT_SCALE_RESIDUAL_LIMIT,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = evaluate_dataset(
        args.dataset_root,
        expected_packages=args.expected_packages,
        current_scale_residual_limit=args.current_scale_residual_limit,
    )
    write_report(report, args.output_directory)
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise TypeError("report summary must be a mapping")
    return 0 if summary.get("production_core_screening_gate") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
