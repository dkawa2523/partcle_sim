"""Replay saved aggregate ion-drag forces and compare the production revisions.

This is an external, frozen-state V&V tool.  It neither reruns COMSOL nor makes
the saved packages a solver dependency or a golden definition of ion drag.
The native saved-formula replay and the production-canonical comparison are
reported separately because the image revision intentionally has a different
ion-speed authority in the saved Case P/A models.
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
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from chamber_particles.physics.forces import (
    AGGREGATE_ION_DRAG_CHARGE_SQUARE_REGULARIZATION,
    AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S,
    ELEMENTARY_CHARGE_C,
    IMAGE_ION_DRAG_ELECTRIC_FIELD_SQUARE_FLOOR_V2_M2,
    IMAGE_ION_DRAG_LOG_ARGUMENT_OFFSET,
    VACUUM_PERMITTIVITY_F_M,
    electric_field_directed_image_orbital_ion_drag,
    relative_flow_screened_collection_orbital_ion_drag,
)

type FloatArray = NDArray[np.float64]
type Record = dict[str, object]
type FormulaRevision = Literal["relative_flow_screened", "electric_field_directed_image"]
type CaseKind = Literal["P", "A"]

TOOL_REVISION: Final = "p18i_saved_primitive_ion_drag_parity_v1"
DEFAULT_FORCE_SCALE_RESIDUAL_LIMIT: Final = 1.0e-10
DEFAULT_EXPECTED_PACKAGES: Final = 12
HISTORY_RELATIVE_PATTERN: Final = (
    "cases/formal_iondrag_*/*/external_reproduction/results/particle_history_full_tidy.csv"
)
SAVED_CASE_A_ION_SPEED_FLOOR_M_S: Final = 1.0e-3
SAVED_DENSITY_FLOOR_M3: Final = 1.0e6

COMMON_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "active_state_flag",
    "charge_number_e",
    "particle_radius_m",
    "particle_mass_kg",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "ion_drag_force_r_N",
    "ion_drag_force_z_N",
    "local_total_positive_ion_density_per_m3",
    "electron_temperature_eV_as_V",
    "effective_positive_ion_mass_kg",
    "local_ion_velocity_r_m_per_s",
    "local_ion_velocity_z_m_per_s",
    "local_bounded_screening_length_m",
    "local_ion_thermal_energy_eV_as_V",
    "local_single_charge_surface_potential_increment_V",
)
THEORY_COLUMNS: Final = ("local_ion_neutral_mean_free_path_m",)
IMAGE_COLUMNS: Final = (
    "local_electric_field_r_V_per_m",
    "local_electric_field_z_V_per_m",
)
CASE_A_IMAGE_COLUMNS: Final = ("local_sheath_potential_relative_to_bulk_V",)
COLUMN_ALIASES: Final = {
    "electron_temperature_eV_as_V": ("local_electron_temperature_eV_as_V",),
    "effective_positive_ion_mass_kg": ("local_effective_positive_ion_mass_kg",),
    "local_bounded_screening_length_m": ("local_screening_length_m",),
}


@dataclass(frozen=True, slots=True)
class HistoryEvaluation:
    """One package's metrics and force arrays used by dataset aggregation."""

    metrics: Record
    native_residual: FloatArray
    native_normalized_residual: FloatArray
    canonical_residual: FloatArray
    canonical_normalized_residual: FloatArray
    exported_force: FloatArray


@dataclass(frozen=True, slots=True)
class ForceAggregate:
    """Concatenated force evidence for one formula/comparison group."""

    residual: FloatArray
    normalized_residual: FloatArray
    exported_force: FloatArray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _classify_history(path: Path) -> tuple[FormulaRevision, CaseKind]:
    lowered = tuple(part.lower() for part in path.parts)
    if "formal_iondrag_theory_consistent" in lowered:
        revision: FormulaRevision = "relative_flow_screened"
    elif "formal_iondrag_image_minimal_corrected" in lowered:
        revision = "electric_field_directed_image"
    else:
        raise ValueError(f"{path}: cannot identify the saved ion-drag formula revision")
    case_parts = tuple(part for part in path.parts if part.lower().startswith(("casep_", "casea_")))
    if len(case_parts) != 1:
        raise ValueError(f"{path}: cannot identify exactly one Case P/A package")
    case_kind: CaseKind = "P" if case_parts[0].lower().startswith("casep_") else "A"
    return revision, case_kind


def _required_columns(revision: FormulaRevision, case_kind: CaseKind) -> tuple[str, ...]:
    columns = COMMON_COLUMNS
    if revision == "relative_flow_screened":
        return (*columns, *THEORY_COLUMNS)
    if case_kind == "A":
        return (*columns, *IMAGE_COLUMNS, *CASE_A_IMAGE_COLUMNS)
    return (*columns, *IMAGE_COLUMNS)


def _resolve_columns(
    fieldnames: Sequence[str] | None,
    required: Sequence[str],
    path: Path,
) -> dict[str, str]:
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
        for name in required
    }
    missing = sorted(name for name, source in sources.items() if not source)
    if missing:
        raise ValueError(f"{path}: required columns are missing: {missing}")
    return sources


def _read_active(
    path: Path,
    revision: FormulaRevision,
    case_kind: CaseKind,
) -> dict[str, FloatArray]:
    required = _required_columns(revision, case_kind)
    values: dict[str, list[float]] = {name: [] for name in required}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        sources = _resolve_columns(reader.fieldnames, required, path)
        for row_number, row in enumerate(reader, start=2):
            active = float(row[sources["active_state_flag"]])
            if active not in (0.0, 1.0):
                raise ValueError(f"{path}:{row_number}: active_state_flag is not 0 or 1")
            if active == 0.0:
                continue
            for name in required:
                value = float(row[sources[name]])
                if not math.isfinite(value):
                    raise ValueError(f"{path}:{row_number}: {name} is not finite")
                values[name].append(value)
    arrays = {name: np.asarray(column, dtype=np.float64) for name, column in values.items()}
    if arrays["particle_id"].size == 0:
        raise ValueError(f"{path}: no active rows")
    return arrays


def _saved_epsilon0(values: dict[str, FloatArray]) -> FloatArray:
    """Infer the producer constant from saved phi1 and its saved screening primitive."""

    radius = values["particle_radius_m"]
    screening = values["local_bounded_screening_length_m"]
    phi_one = values["local_single_charge_surface_potential_increment_V"]
    if bool((radius <= 0.0).any() or (screening <= 0.0).any() or (phi_one <= 0.0).any()):
        raise ValueError("saved radius, screening length, and phi1 must be positive")
    return ELEMENTARY_CHARGE_C / (4.0 * math.pi * radius * (1.0 + radius / screening) * phi_one)


def _native_relative_flow_force(
    values: dict[str, FloatArray],
    epsilon0_F_m: FloatArray,
) -> FloatArray:
    radius = values["particle_radius_m"]
    charge = values["charge_number_e"]
    ion_mass = values["effective_positive_ion_mass_kg"]
    ion_velocity = np.column_stack(
        (values["local_ion_velocity_r_m_per_s"], values["local_ion_velocity_z_m_per_s"])
    )
    particle_velocity = np.column_stack(
        (values["velocity_r_m_per_s"], values["velocity_z_m_per_s"])
    )
    relative_velocity = ion_velocity - particle_velocity
    speed_square = (
        np.sum(relative_velocity * relative_velocity, axis=1)
        + 8.0
        * ELEMENTARY_CHARGE_C
        * values["local_ion_thermal_energy_eV_as_V"]
        / (math.pi * ion_mass)
        + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
    )
    screening_radius = np.maximum(
        radius,
        np.minimum(
            values["local_bounded_screening_length_m"],
            values["local_ion_neutral_mean_free_path_m"],
        ),
    )
    orbital_impact = (
        np.sqrt(charge * charge + AGGREGATE_ION_DRAG_CHARGE_SQUARE_REGULARIZATION)
        * ELEMENTARY_CHARGE_C**2
        / (4.0 * math.pi * epsilon0_F_m * ion_mass * speed_square)
    )
    surface_potential = charge * values["local_single_charge_surface_potential_increment_V"]
    collection_square = np.minimum(
        screening_radius**2,
        radius**2
        * np.maximum(
            0.0,
            1.0 - 2.0 * ELEMENTARY_CHARGE_C * surface_potential / (ion_mass * speed_square),
        ),
    )
    logarithm = np.maximum(
        0.0,
        0.5
        * np.log(
            (screening_radius**2 + orbital_impact**2) / (collection_square + orbital_impact**2)
        ),
    )
    cross_section = math.pi * collection_square + 4.0 * math.pi * orbital_impact**2 * logarithm
    factor = (
        values["local_total_positive_ion_density_per_m3"]
        * ion_mass
        * np.sqrt(speed_square)
        * cross_section
    )
    return factor[:, None] * relative_velocity


def _saved_image_speed(
    values: dict[str, FloatArray],
    case_kind: CaseKind,
) -> tuple[FloatArray, str]:
    vector_norm_square = (
        values["local_ion_velocity_r_m_per_s"] ** 2 + values["local_ion_velocity_z_m_per_s"] ** 2
    )
    if case_kind == "P":
        return (
            np.sqrt(vector_norm_square + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2),
            "saved_caseP_sqrt_norm_ui_squared_plus_u_eps_squared",
        )
    ion_mass = values["effective_positive_ion_mass_kg"]
    squared = (
        ELEMENTARY_CHARGE_C * values["electron_temperature_eV_as_V"] / ion_mass
        - 2.0 * ELEMENTARY_CHARGE_C * values["local_sheath_potential_relative_to_bulk_V"] / ion_mass
    )
    return (
        np.sqrt(np.maximum(SAVED_CASE_A_ION_SPEED_FLOOR_M_S**2, squared)),
        "saved_caseA_AS_ui_mag_reconstructed_from_psi_Te_mi_and_u_floor",
    )


def _native_image_force(
    values: dict[str, FloatArray],
    epsilon0_F_m: FloatArray,
    case_kind: CaseKind,
) -> tuple[FloatArray, FloatArray, str]:
    radius = values["particle_radius_m"]
    charge = values["charge_number_e"]
    density = values["local_total_positive_ion_density_per_m3"]
    ion_mass = values["effective_positive_ion_mass_kg"]
    ion_speed, authority = _saved_image_speed(values, case_kind)
    thermal = (
        8.0
        * ELEMENTARY_CHARGE_C
        * values["local_ion_thermal_energy_eV_as_V"]
        / (math.pi * ion_mass)
    )
    if case_kind == "P":
        speed_square = ion_speed**2 + thermal
    else:
        speed_square = (
            ion_speed**2 + thermal + AGGREGATE_ION_DRAG_RELATIVE_SPEED_REGULARIZATION_M_S**2
        )
    surface_potential = charge * values["local_single_charge_surface_potential_increment_V"]
    collection = (
        math.pi
        * radius**2
        * np.maximum(
            0.0,
            1.0 - surface_potential / values["local_ion_thermal_energy_eV_as_V"],
        )
    )
    image_impact = (
        ELEMENTARY_CHARGE_C**2 * charge / (2.0 * math.pi * epsilon0_F_m * ion_mass * speed_square)
    )
    image_screening = np.sqrt(
        epsilon0_F_m
        * values["electron_temperature_eV_as_V"]
        / (ELEMENTARY_CHARGE_C * np.maximum(density, SAVED_DENSITY_FLOOR_M3))
    )
    orbital = (
        math.pi
        * image_impact**2
        * np.log(np.maximum(1.0 + IMAGE_ION_DRAG_LOG_ARGUMENT_OFFSET, image_screening / radius))
    )
    magnitude = density * ion_mass * np.sqrt(speed_square) * ion_speed * (collection + orbital)
    electric = np.column_stack(
        (values["local_electric_field_r_V_per_m"], values["local_electric_field_z_V_per_m"])
    )
    electric_norm = np.sqrt(
        np.sum(electric * electric, axis=1) + IMAGE_ION_DRAG_ELECTRIC_FIELD_SQUARE_FLOOR_V2_M2
    )
    return magnitude[:, None] * electric / electric_norm[:, None], ion_speed, authority


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
    if revision == "relative_flow_screened":
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


def _native_force(
    values: dict[str, FloatArray],
    revision: FormulaRevision,
    case_kind: CaseKind,
    saved_epsilon: FloatArray,
) -> tuple[FloatArray, FloatArray | None, str]:
    if revision == "relative_flow_screened":
        return (
            _native_relative_flow_force(values, saved_epsilon),
            None,
            "saved_relative_ion_particle_velocity_vector",
        )
    force, speed, authority = _native_image_force(values, saved_epsilon, case_kind)
    return force, speed, authority


def _residual_metrics(
    candidate: FloatArray,
    exported: FloatArray,
) -> tuple[FloatArray, FloatArray, float, int]:
    residual = candidate - exported
    candidate_norm = np.linalg.norm(candidate, axis=1)
    exported_norm = np.linalg.norm(exported, axis=1)
    scale = np.maximum(np.maximum(candidate_norm, exported_norm), np.finfo(np.float64).tiny)
    normalized = np.linalg.norm(residual, axis=1) / scale
    relative_l2 = float(
        np.linalg.norm(residual)
        / max(float(np.linalg.norm(exported)), float(np.finfo(np.float64).tiny))
    )
    return residual, normalized, relative_l2, int(np.argmax(normalized))


def _comparison_labels(
    revision: FormulaRevision,
    case_kind: CaseKind,
    native_speed_authority: str,
    native_pass: bool,
    canonical_pass: bool,
) -> tuple[str, str, str]:
    if revision == "relative_flow_screened":
        status = "PASS" if canonical_pass else "FAIL"
        diagnosis = (
            "NOT_INDICATED"
            if canonical_pass
            else (
                "EPSILON0_CONSTANT_CONVENTION_DIFFERENCE_INDICATED"
                if native_pass
                else "UNRESOLVED_NATIVE_REPLAY_FAILURE"
            )
        )
        return status, diagnosis, "NOT_APPLICABLE"
    difference = (
        f"saved {case_kind} speed authority '{native_speed_authority}' differs from "
        "production canonical U=norm(saved_ui_vector); residual is informative, not a gate"
    )
    return (
        "DOCUMENTED_MODEL_DEFINITION_DIFFERENCE",
        "ION_SPEED_AUTHORITY_MODEL_DEFINITION_DIFFERENCE",
        difference,
    )


def evaluate_history(
    history_path: Path,
    *,
    force_scale_residual_limit: float = DEFAULT_FORCE_SCALE_RESIDUAL_LIMIT,
) -> HistoryEvaluation:
    """Evaluate one saved history without changing or rerunning its producer model."""

    if not math.isfinite(force_scale_residual_limit) or force_scale_residual_limit <= 0.0:
        raise ValueError("force_scale_residual_limit must be finite and positive")
    path = history_path.expanduser().resolve()
    revision, case_kind = _classify_history(path)
    values = _read_active(path, revision, case_kind)
    exported = np.column_stack((values["ion_drag_force_r_N"], values["ion_drag_force_z_N"]))
    saved_epsilon = _saved_epsilon0(values)
    native, native_speed, native_speed_authority = _native_force(
        values, revision, case_kind, saved_epsilon
    )
    canonical = _canonical_force(values, revision)
    native_residual, native_normalized, native_l2, native_worst = _residual_metrics(
        native, exported
    )
    canonical_residual, canonical_normalized, canonical_l2, canonical_worst = _residual_metrics(
        canonical, exported
    )
    finite = bool(
        np.isfinite(native).all()
        and np.isfinite(canonical).all()
        and np.isfinite(native_normalized).all()
        and np.isfinite(canonical_normalized).all()
        and np.isfinite(saved_epsilon).all()
    )
    native_pass = bool(
        finite and float(native_normalized[native_worst]) <= force_scale_residual_limit
    )
    canonical_pass = bool(
        finite and float(canonical_normalized[canonical_worst]) <= force_scale_residual_limit
    )
    canonical_status, canonical_diagnosis, difference = _comparison_labels(
        revision,
        case_kind,
        native_speed_authority,
        native_pass,
        canonical_pass,
    )
    metrics: Record = {
        "path": str(path),
        "sha256": _sha256(path),
        "formula_revision": revision,
        "case_kind": case_kind,
        "active_rows": int(exported.shape[0]),
        "finite": finite,
        "native_saved_formula_replay_status": "PASS" if native_pass else "FAIL",
        "native_saved_formula_force_global_relative_l2_residual": native_l2,
        "native_saved_formula_force_scale_normalized_residual_max": float(
            native_normalized[native_worst]
        ),
        "native_speed_authority": native_speed_authority,
        "production_canonical_comparison_status": canonical_status,
        "production_canonical_force_global_relative_l2_residual": canonical_l2,
        "production_canonical_force_scale_normalized_residual_max": float(
            canonical_normalized[canonical_worst]
        ),
        "production_canonical_speed_authority": (
            "relative_ion_particle_velocity_vector"
            if revision == "relative_flow_screened"
            else "norm(saved_ui_vector)"
        ),
        "production_canonical_difference_diagnosis": canonical_diagnosis,
        "model_definition_difference": difference,
        "saved_epsilon0_F_m_median": float(np.median(saved_epsilon)),
        "saved_vs_production_epsilon0_relative_difference_median": float(
            np.median((saved_epsilon - VACUUM_PERMITTIVITY_F_M) / VACUUM_PERMITTIVITY_F_M)
        ),
        "native_worst_row": {
            "particle_id": int(values["particle_id"][native_worst]),
            "time_s": float(values["time_s"][native_worst]),
        },
        "production_canonical_worst_row": {
            "particle_id": int(values["particle_id"][canonical_worst]),
            "time_s": float(values["time_s"][canonical_worst]),
        },
    }
    if native_speed is not None:
        vector_speed = np.hypot(
            values["local_ion_velocity_r_m_per_s"],
            values["local_ion_velocity_z_m_per_s"],
        )
        metrics["native_vs_vector_ion_speed_relative_difference_max"] = float(
            np.max(
                np.abs(native_speed - vector_speed)
                / np.maximum(native_speed, np.finfo(np.float64).tiny)
            )
        )
    return HistoryEvaluation(
        metrics,
        native_residual,
        native_normalized,
        canonical_residual,
        canonical_normalized,
        exported,
    )


def _global_relative_l2(residual: FloatArray, exported: FloatArray) -> float:
    return float(
        np.linalg.norm(residual)
        / max(float(np.linalg.norm(exported)), float(np.finfo(np.float64).tiny))
    )


def _aggregate_forces(
    results: Sequence[HistoryEvaluation],
    *,
    canonical: bool,
) -> ForceAggregate:
    if not results:
        raise ValueError("cannot aggregate an empty ion-drag result group")
    if canonical:
        residual = np.concatenate([result.canonical_residual for result in results])
        normalized = np.concatenate([result.canonical_normalized_residual for result in results])
    else:
        residual = np.concatenate([result.native_residual for result in results])
        normalized = np.concatenate([result.native_normalized_residual for result in results])
    exported = np.concatenate([result.exported_force for result in results])
    return ForceAggregate(residual, normalized, exported)


def _dataset_summary(
    package_metrics: Sequence[Record],
    all_native: ForceAggregate,
    theory_results: Sequence[HistoryEvaluation],
    theory_canonical: ForceAggregate,
    image_results: Sequence[HistoryEvaluation],
    image_canonical: ForceAggregate,
) -> Record:
    native_pass = all(
        metrics["native_saved_formula_replay_status"] == "PASS" for metrics in package_metrics
    )
    theory_native_pass = all(
        result.metrics["native_saved_formula_replay_status"] == "PASS" for result in theory_results
    )
    theory_canonical_pass = all(
        result.metrics["production_canonical_comparison_status"] == "PASS"
        for result in theory_results
    )
    theory_diagnosis = (
        "NOT_INDICATED"
        if theory_canonical_pass
        else (
            "EPSILON0_CONSTANT_CONVENTION_DIFFERENCE_INDICATED"
            if theory_native_pass
            else "UNRESOLVED_NATIVE_REPLAY_FAILURE"
        )
    )
    return {
        "packages": len(package_metrics),
        "active_rows": int(all_native.exported_force.shape[0]),
        "native_saved_formula_replay_gate": "PASS" if native_pass else "FAIL",
        "native_saved_formula_force_global_relative_l2_residual": _global_relative_l2(
            all_native.residual, all_native.exported_force
        ),
        "native_saved_formula_force_scale_normalized_residual_p99": float(
            np.percentile(all_native.normalized_residual, 99.0)
        ),
        "native_saved_formula_force_scale_normalized_residual_max": float(
            np.max(all_native.normalized_residual)
        ),
        "theory_production_canonical_gate": ("PASS" if theory_canonical_pass else "FAIL"),
        "theory_production_canonical_force_global_relative_l2_residual": (
            _global_relative_l2(theory_canonical.residual, theory_canonical.exported_force)
        ),
        "theory_production_canonical_force_scale_normalized_residual_max": float(
            np.max(theory_canonical.normalized_residual)
        ),
        "theory_production_canonical_difference_diagnosis": theory_diagnosis,
        "image_production_canonical_comparison_status": ("DOCUMENTED_MODEL_DEFINITION_DIFFERENCE"),
        "image_production_canonical_force_global_relative_l2_residual": (
            _global_relative_l2(image_canonical.residual, image_canonical.exported_force)
        ),
        "image_production_canonical_force_scale_normalized_residual_max": float(
            np.max(image_canonical.normalized_residual)
        ),
        "image_model_definition_difference_packages": len(image_results),
    }


def evaluate_dataset(
    dataset_root: Path,
    *,
    expected_packages: int = DEFAULT_EXPECTED_PACKAGES,
    force_scale_residual_limit: float = DEFAULT_FORCE_SCALE_RESIDUAL_LIMIT,
) -> Record:
    """Evaluate every saved theory/image package discovered below ``dataset_root``."""

    root = dataset_root.expanduser().resolve()
    histories = tuple(sorted(root.glob(HISTORY_RELATIVE_PATTERN)))
    if len(histories) != expected_packages:
        raise ValueError(f"expected {expected_packages} history packages, found {len(histories)}")
    package_metrics: list[Record] = []
    results: list[HistoryEvaluation] = []
    for history in histories:
        result = evaluate_history(
            history,
            force_scale_residual_limit=force_scale_residual_limit,
        )
        metrics = result.metrics
        metrics["package"] = history.relative_to(root).as_posix()
        metrics.pop("path")
        package_metrics.append(metrics)
        results.append(result)
    theory_results = [
        result
        for result in results
        if result.metrics["formula_revision"] == "relative_flow_screened"
    ]
    image_results = [
        result
        for result in results
        if result.metrics["formula_revision"] == "electric_field_directed_image"
    ]
    all_native = _aggregate_forces(results, canonical=False)
    theory_canonical = _aggregate_forces(theory_results, canonical=True)
    image_canonical = _aggregate_forces(image_results, canonical=True)
    return {
        "tool_revision": TOOL_REVISION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evidence_role": "external_saved_primitive_frozen_force_parity",
        "golden_truth": "NOT_CLAIMED",
        "trajectory_accuracy": "NOT_TESTED",
        "comsol_rerun": False,
        "dataset_root": str(root),
        "force_scale_residual_limit": force_scale_residual_limit,
        "summary": _dataset_summary(
            package_metrics,
            all_native,
            theory_results,
            theory_canonical,
            image_results,
            image_canonical,
        ),
        "packages": package_metrics,
        "interpretation": (
            "Native saved-formula replay is the only all-package parity gate. The theory "
            "production comparison is also a gate because it represents the same formula "
            "revision with production constants. Image Case P saved U=sqrt(norm(ui)^2+u_eps^2); "
            "Image Case A saved U=AS_ui_mag reconstructed from saved psi, Te, and mi. The "
            "production image revision deliberately uses U=norm(ui). Its residual is therefore "
            "a documented model-definition comparison and cannot produce FAIL. These frozen "
            "force checks make no integrated trajectory or physical-validity claim."
        ),
    }


def write_report(report: Record, output_directory: Path) -> None:
    """Write a no-clobber JSON report and compact package CSV."""

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
        "formula_revision",
        "case_kind",
        "active_rows",
        "native_saved_formula_replay_status",
        "native_saved_formula_force_global_relative_l2_residual",
        "native_saved_formula_force_scale_normalized_residual_max",
        "native_speed_authority",
        "production_canonical_comparison_status",
        "production_canonical_force_global_relative_l2_residual",
        "production_canonical_force_scale_normalized_residual_max",
        "production_canonical_speed_authority",
        "production_canonical_difference_diagnosis",
        "model_definition_difference",
        "native_vs_vector_ion_speed_relative_difference_max",
        "saved_epsilon0_F_m_median",
        "saved_vs_production_epsilon0_relative_difference_median",
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
        "--force-scale-residual-limit",
        type=float,
        default=DEFAULT_FORCE_SCALE_RESIDUAL_LIMIT,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = evaluate_dataset(
        args.dataset_root,
        expected_packages=args.expected_packages,
        force_scale_residual_limit=args.force_scale_residual_limit,
    )
    write_report(report, args.output_directory)
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise TypeError("report summary must be a mapping")
    return 0 if summary.get("native_saved_formula_replay_gate") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
