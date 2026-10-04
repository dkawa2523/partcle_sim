"""Diagnose field and force layers of the deterministic matched case.

The tool evaluates the unmodified production field and physics primitives at
the COMSOL reference states.  It is external evidence: no comparison-specific
mode is added to the solver, and the report does not alter acceptance limits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Final

import numpy as np

from chamber_particles import load_case
from chamber_particles.fields import RequiredFieldMetadata, prepare_required_fields
from chamber_particles.geometry import prepare_geometry
from chamber_particles.physics.catalog import (
    ElectricPlan,
    EpsteinDragPlan,
    GravityBuoyancyPlan,
    PhysicsPlan,
    resolve_physics_plan,
)
from chamber_particles.physics.forces import BOLTZMANN_J_K
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime
from chamber_particles.sources import realize_sources

TOOL_REVISION: Final = "m3v_matched_layer_diagnostic_v2"
ELEMENTARY_CHARGE_C: Final = 1.602176634e-19
EXPECTED_ROWS: Final = 287 * 41
ROUNDOFF_RELATIVE_L2_LIMIT: Final = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in columns):
            raise ValueError(f"{resolved}: required columns are missing")
        rows = list(reader)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"{resolved}: expected {EXPECTED_ROWS} rows, found {len(rows)}")
    return rows


def _numeric(rows: list[dict[str, str]], columns: tuple[str, ...]) -> np.ndarray:
    result = np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite values in columns {columns}")
    return result


def _keys(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    particle = _numeric(rows, ("particle_id",))[:, 0]
    time = _numeric(rows, ("time_s",))[:, 0]
    particle_id = np.rint(particle).astype(np.int64)
    if not np.array_equal(particle, particle_id.astype(np.float64)):
        raise ValueError("particle IDs are not exact integers")
    keys = list(zip(time, particle_id, strict=True))
    if len(set(keys)) != EXPECTED_ROWS:
        raise ValueError("reference trajectory keys are not unique")
    return particle_id, time


def _align_to_keys(
    rows: list[dict[str, str]],
    particle_id: np.ndarray,
    time: np.ndarray,
) -> list[dict[str, str]]:
    indexed: dict[tuple[int, float], dict[str, str]] = {}
    for row in rows:
        key = (int(row["particle_id"]), float(row["time_s"]))
        if key in indexed:
            raise ValueError(f"duplicate reference key: {key}")
        indexed[key] = row
    ordered: list[dict[str, str]] = []
    for particle, sample_time in zip(particle_id, time, strict=True):
        key = (int(particle), float(sample_time))
        if key not in indexed:
            raise ValueError(f"missing reference key: {key}")
        ordered.append(indexed[key])
    if len(indexed) != len(ordered):
        raise ValueError("reference artifact has unmatched keys")
    return ordered


def _metric(
    candidate: np.ndarray,
    reference: np.ndarray,
    particle_id: np.ndarray,
    time: np.ndarray,
) -> dict[str, object]:
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("diagnostic arrays have incompatible shapes")
    difference = candidate - reference
    magnitude = (
        np.abs(difference[:, 0]) if difference.shape[1] == 1 else np.linalg.norm(difference, axis=1)
    )
    worst = int(np.argmax(magnitude))
    difference_square = float(np.sum(difference * difference, dtype=np.float64))
    reference_square = float(np.sum(reference * reference, dtype=np.float64))
    return {
        "components": int(candidate.shape[1]),
        "rms_vector_or_scalar": float(np.sqrt(np.mean(magnitude * magnitude))),
        "maximum_vector_or_scalar": float(magnitude[worst]),
        "relative_l2": math.sqrt(difference_square / reference_square)
        if reference_square > 0.0
        else None,
        "p50": float(np.quantile(magnitude, 0.50)),
        "p90": float(np.quantile(magnitude, 0.90)),
        "p99": float(np.quantile(magnitude, 0.99)),
        "worst_key": {
            "particle_id": int(particle_id[worst]),
            "time_s": float(time[worst]),
        },
        "component_rms": [
            float(np.sqrt(np.mean(difference[:, index] ** 2)))
            for index in range(difference.shape[1])
        ],
        "component_maximum_absolute": [
            float(np.max(np.abs(difference[:, index]))) for index in range(difference.shape[1])
        ],
    }


def _above_roundoff(metrics: dict[str, dict[str, object]]) -> bool:
    for metric in metrics.values():
        relative_l2 = metric["relative_l2"]
        if relative_l2 is None:
            maximum = metric["maximum_vector_or_scalar"]
            if not isinstance(maximum, (int, float)):
                raise ValueError("diagnostic maximum is not numeric")
            if maximum > 0.0:
                return True
        else:
            if not isinstance(relative_l2, (int, float)):
                raise ValueError("diagnostic relative L2 is not numeric")
            if relative_l2 > ROUNDOFF_RELATIVE_L2_LIMIT:
                return True
    return False


def _first_difference_owner(
    field_differences: dict[str, dict[str, object]],
    reference_formula_parity: dict[str, dict[str, object]],
    candidate_formula_vs_reference: dict[str, dict[str, object]],
    runtime_formula_parity: dict[str, object],
) -> str:
    if _above_roundoff(field_differences):
        return "field_representation_or_sampling"
    if _above_roundoff(reference_formula_parity):
        return "reference_export_formula_or_settings"
    if _above_roundoff(candidate_formula_vs_reference):
        return "cross_solver_force_or_field_coupling"
    if _above_roundoff({"acceleration": runtime_formula_parity}):
        return "candidate_runtime_or_formula"
    return "NONE_WITHIN_ROUNDOFF"


def _formula_forces(
    plan: PhysicsPlan,
    mass: np.ndarray,
    diameter: np.ndarray,
    displaced_volume: np.ndarray,
    velocity: np.ndarray,
    charge: np.ndarray,
    gas_velocity: np.ndarray,
    gas_density: np.ndarray,
    gas_temperature: np.ndarray,
    electric_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drag = plan.drag
    electric = plan.electric
    gravity = plan.gravity_buoyancy
    if not isinstance(drag, EpsteinDragPlan):
        raise ValueError("matched case requires linear Epstein drag")
    if not isinstance(electric, ElectricPlan):
        raise ValueError("matched case requires Coulomb electric force")
    if not isinstance(gravity, GravityBuoyancyPlan):
        raise ValueError("matched case requires gravity/buoyancy")
    radius = 0.5 * diameter
    mean_speed = np.sqrt(
        8.0 * BOLTZMANN_J_K * gas_temperature / (math.pi * drag.gas_molecular_mass_kg)
    )
    beta = (4.0 * math.pi / 3.0) * radius**2 * gas_density * mean_speed * drag.delta
    drag_force = beta[:, None] * (gas_velocity - velocity)
    electric_force = charge[:, None] * ELEMENTARY_CHARGE_C * electric_field
    gravity_vector = np.asarray(gravity.gravity_m_s2, dtype=np.float64)
    gravity_force = (mass - gas_density * displaced_volume)[:, None] * gravity_vector
    acceleration = (drag_force + electric_force + gravity_force) / mass[:, None]
    return electric_force, drag_force, gravity_force, acceleration


def diagnose(
    case_path: Path,
    trajectory_path: Path,
    force_path: Path,
    field_path: Path,
) -> dict[str, object]:
    case = load_case(case_path)
    trajectory_rows = _read(trajectory_path, ("particle_id", "time_s", "r_m", "z_m"))
    particle_id, time = _keys(trajectory_rows)
    force_rows = _align_to_keys(
        _read(force_path, ("particle_id", "time_s", "acceleration_r_m_per_s2")),
        particle_id,
        time,
    )
    field_rows = _read(field_path, ("probe_id", "r_m", "z_m"))
    if len(field_rows) != len(trajectory_rows):
        raise ValueError("field reference row count differs from trajectory")

    position = _numeric(trajectory_rows, ("r_m", "z_m"))
    velocity = _numeric(trajectory_rows, ("velocity_r_m_per_s", "velocity_z_m_per_s"))
    charge = _numeric(trajectory_rows, ("charge_number_e",))[:, 0]
    reference_field_position = _numeric(field_rows, ("r_m", "z_m"))
    if not np.array_equal(position, reference_field_position):
        raise ValueError("field probes do not use the reference trajectory positions")

    plan = resolve_physics_plan(case.spec.physics.models, case.data.coordinate_system)
    requirements = {
        item.name: RequiredFieldMetadata(
            item.unit,
            item.components,
            item.stored_basis,
            item.positive,
        )
        for item in plan.required_fields
    }
    fields = prepare_required_fields(case.data, requirements)
    sampled = fields.sample(position)
    if not bool(sampled.support_inside.all()):
        raise ValueError("candidate fields do not cover every reference state")

    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    schedule = realize_sources(case, geometry)
    resident = np.searchsorted(schedule.particle_id, particle_id)
    if not np.array_equal(schedule.particle_id[resident], particle_id):
        raise ValueError("reference particle IDs do not match the candidate source")
    mass = schedule.mass_kg[resident]
    diameter = schedule.drag_diameter_m[resident]
    displaced_volume = schedule.displaced_volume_m3[resident]
    if not np.allclose(diameter, 1.0e-7, rtol=1.0e-15, atol=0.0):
        raise ValueError("matched case does not contain only 100 nm particles")

    primitive_ranges = {
        item.name: PrimitiveRange(
            *fields.component_bounds(item.name),
            fields.constant_value(item.name),
        )
        for item in plan.required_fields
    }
    runtime = prepare_physics_runtime(
        plan=plan,
        coordinate_system=case.data.coordinate_system,
        mass_kg=schedule.mass_kg,
        drag_diameter_m=schedule.drag_diameter_m,
        electrostatic_radius_m=schedule.electrostatic_radius_m,
        displaced_volume_m3=schedule.displaced_volume_m3,
        charge_number=schedule.charge_number,
        primitive_ranges=primitive_ranges,
    )
    evaluation = runtime.evaluate(resident, velocity, charge, sampled.values)

    candidate_gas_velocity = sampled.values["gas_velocity"]
    candidate_gas_density = sampled.values["gas_density"][:, 0]
    candidate_gas_temperature = sampled.values["gas_temperature"][:, 0]
    candidate_electric_field = sampled.values["electric_field"]
    reference_gas_velocity = _numeric(
        field_rows, ("gas_velocity_r_m_per_s", "gas_velocity_z_m_per_s")
    )
    reference_gas_density = _numeric(field_rows, ("gas_density_kg_per_m3",))[:, 0]
    reference_gas_temperature = _numeric(field_rows, ("gas_temperature_K",))[:, 0]
    reference_electric_field = _numeric(
        field_rows, ("electric_field_r_V_per_m", "electric_field_z_V_per_m")
    )

    reference_electric_force = _numeric(force_rows, ("electric_force_r_N", "electric_force_z_N"))
    reference_drag_force = _numeric(force_rows, ("epstein_force_r_N", "epstein_force_z_N"))
    reference_gravity_force = _numeric(
        force_rows, ("gravity_buoyancy_force_r_N", "gravity_buoyancy_force_z_N")
    )
    reference_acceleration = _numeric(
        force_rows, ("acceleration_r_m_per_s2", "acceleration_z_m_per_s2")
    )

    expected_reference = _formula_forces(
        plan,
        mass,
        diameter,
        displaced_volume,
        velocity,
        charge,
        reference_gas_velocity,
        reference_gas_density,
        reference_gas_temperature,
        reference_electric_field,
    )
    candidate_formula = _formula_forces(
        plan,
        mass,
        diameter,
        displaced_volume,
        velocity,
        charge,
        candidate_gas_velocity,
        candidate_gas_density,
        candidate_gas_temperature,
        candidate_electric_field,
    )
    labels = ("electric_force", "epstein_force", "gravity_buoyancy_force", "acceleration")
    exported = (
        reference_electric_force,
        reference_drag_force,
        reference_gravity_force,
        reference_acceleration,
    )
    field_differences = {
        "gas_velocity": _metric(candidate_gas_velocity, reference_gas_velocity, particle_id, time),
        "gas_density": _metric(
            candidate_gas_density[:, None],
            reference_gas_density[:, None],
            particle_id,
            time,
        ),
        "gas_temperature": _metric(
            candidate_gas_temperature[:, None],
            reference_gas_temperature[:, None],
            particle_id,
            time,
        ),
        "electric_field": _metric(
            candidate_electric_field, reference_electric_field, particle_id, time
        ),
    }
    reference_formula_parity = {
        label: _metric(expected, actual, particle_id, time)
        for label, expected, actual in zip(labels, expected_reference, exported, strict=True)
    }
    candidate_formula_vs_reference = {
        label: _metric(candidate, actual, particle_id, time)
        for label, candidate, actual in zip(labels, candidate_formula, exported, strict=True)
    }
    runtime_formula_parity = _metric(
        evaluation.acceleration_m_s2,
        candidate_formula[3],
        particle_id,
        time,
    )
    first_difference_owner = _first_difference_owner(
        field_differences,
        reference_formula_parity,
        candidate_formula_vs_reference,
        runtime_formula_parity,
    )
    return {
        "tool_revision": TOOL_REVISION,
        "report_kind": "matched_case_field_force_layer_diagnostic",
        "artifacts": {
            "case": {"path": str(case.case_path), "sha256": case.case_file_hash},
            "trajectory": {
                "path": str(trajectory_path.resolve()),
                "sha256": _sha256(trajectory_path.resolve()),
            },
            "force": {"path": str(force_path.resolve()), "sha256": _sha256(force_path.resolve())},
            "field": {"path": str(field_path.resolve()), "sha256": _sha256(field_path.resolve())},
        },
        "rows": EXPECTED_ROWS,
        "candidate_support_inside_fraction": float(np.mean(sampled.support_inside)),
        "candidate_applicable_fraction": float(np.mean(evaluation.applicable)),
        "field_differences_candidate_vs_reference": field_differences,
        "reference_export_formula_parity": reference_formula_parity,
        "candidate_formula_vs_reference_export": candidate_formula_vs_reference,
        "candidate_runtime_vs_candidate_formula_acceleration": runtime_formula_parity,
        "roundoff_relative_l2_limit": ROUNDOFF_RELATIVE_L2_LIMIT,
        "first_difference_owner": first_difference_owner,
        "interpretation": (
            "The neutral candidate/reference field comparison and independent formula replays "
            "separate field sampling, reference-export settings, and candidate runtime layers. "
            "NONE_WITHIN_ROUNDOFF means every reported relative L2 residual is at or below the "
            "declared roundoff limit; it is not a universal physics-validity claim."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("reference_trajectory", type=Path)
    parser.add_argument("reference_force", type=Path)
    parser.add_argument("reference_field", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = diagnose(
        args.case,
        args.reference_trajectory,
        args.reference_force,
        args.reference_field,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", errors="strict") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
