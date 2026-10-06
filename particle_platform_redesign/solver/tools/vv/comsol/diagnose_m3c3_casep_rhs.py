"""Compare M3-C3 COMSOL and production physics at identical active states.

This external V&V tool reads the diagnostic COMSOL export, samples the normal
production field provider at those saved states, and evaluates the public pure
physics formulas.  It does not integrate a trajectory and does not add a
COMSOL-specific path to the solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from chamber_particles import load_case
from chamber_particles.fields import RequiredFieldMetadata, prepare_required_fields
from chamber_particles.geometry import prepare_geometry
from chamber_particles.physics.catalog import (
    AggregateRelativeDriftChargePlan,
    ElectricPlan,
    EpsteinDragPlan,
    GravityBuoyancyPlan,
    QuasistaticSphericalDielectrophoresisPlan,
    RarefiedVorticityLiftPlan,
    RelativeFlowScreenedIonDragPlan,
    WaldmannGallisThermophoresisPlan,
    resolve_physics_plan,
)
from chamber_particles.physics.charge import (
    AGGREGATE_EXPONENT_MAX,
    AGGREGATE_EXPONENT_MIN,
    ELECTRON_MASS_KG,
    aggregate_relative_drift_regularized_three_current_v1,
)
from chamber_particles.physics.forces import (
    ELEMENTARY_CHARGE_C,
    add_electric_coulomb_acceleration,
    add_gravity_buoyancy_acceleration,
    epstein_linear_relaxation,
    quasistatic_spherical_dep_acceleration,
    rarefied_vorticity_lift,
    relative_flow_screened_collection_orbital_ion_drag,
    waldmann_gallis_thermophoresis,
)
from chamber_particles.physics.runtime import PrimitiveRange, prepare_physics_runtime
from chamber_particles.sources import realize_sources

type FloatArray = NDArray[np.float64]
type Record = dict[str, object]

TOOL_REVISION: Final = "m3c3_casep_frozen_rhs_v2"
EXPECTED_PARTICLES: Final = 287
EXPECTED_FRAMES: Final = 121
FORMULA_PARITY_RELATIVE_L2_LIMIT: Final = 1.0e-8
RUNTIME_DECOMPOSITION_RELATIVE_L2_LIMIT: Final = 1.0e-12

STATE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "current_status_code",
    "final_status_code",
    "stop_or_event_time_s",
)
FORCE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "charge_rate_number_s",
    "positive_collection_rate_number_s",
    "electron_collection_rate_number_s",
    "negative_collection_rate_number_s",
    "electric_force_r_N",
    "electric_force_z_N",
    "ion_drag_force_r_N",
    "ion_drag_force_z_N",
    "epstein_drag_force_r_N",
    "epstein_drag_force_z_N",
    "thermophoretic_force_r_N",
    "thermophoretic_force_z_N",
    "lift_force_r_N",
    "lift_force_z_N",
    "dep_force_r_N",
    "dep_force_z_N",
    "gravity_buoyancy_force_r_N",
    "gravity_buoyancy_force_z_N",
    "total_force_r_N",
    "total_force_z_N",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(cast(Mapping[str, Any], value))


def _read_wide(path: Path, columns: tuple[str, ...]) -> dict[str, FloatArray]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.startswith("%") or not line.strip():
                continue
            try:
                row = [float(value) for value in next(csv.reader([line]))]
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: nonnumeric diagnostic value") from error
            rows.append(row)
    expected_width = len(columns) * EXPECTED_FRAMES
    if len(rows) != EXPECTED_PARTICLES or any(len(row) != expected_width for row in rows):
        raise ValueError(f"{path}: expected {EXPECTED_PARTICLES} rows x {expected_width} columns")
    matrix = np.asarray(rows, dtype=np.float64).reshape(
        EXPECTED_PARTICLES, EXPECTED_FRAMES, len(columns)
    )
    flat = matrix.reshape(EXPECTED_PARTICLES * EXPECTED_FRAMES, len(columns))
    return {name: flat[:, index] for index, name in enumerate(columns)}


def _artifact_by_role(raw: Mapping[str, Any], role: str) -> dict[str, Any]:
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("execution record artifacts must be an array")
    matches = [
        _mapping(item, f"artifact {role}")
        for item in artifacts
        if isinstance(item, Mapping) and item.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"execution record must contain one {role} artifact")
    return matches[0]


def _validate_execution_record(record_path: Path, state_path: Path, force_path: Path) -> Record:
    raw = _mapping(json.loads(record_path.read_text(encoding="utf-8")), "execution record")
    numerical = _mapping(raw.get("numerical_run"), "numerical_run")
    if (
        raw.get("schema_version") != 1
        or raw.get("status") != "LOCKED_FOR_EXECUTION"
        or raw.get("case_id") != "caseP_100nm_three_current"
        or float(numerical.get("fixed_rk4_step_s", math.nan)) != 1.25e-6
        or numerical.get("diagnostic_force_export") is not True
        or numerical.get("comparison_scope") != "active_rows_only"
    ):
        raise ValueError("execution record does not match the frozen M3-C3 diagnostic protocol")
    before = str(raw.get("source_mph_sha256_before", ""))
    after = str(raw.get("source_mph_sha256_after", ""))
    expected = str(raw.get("expected_source_mph_sha256", ""))
    if len(before) != 64 or before != after or before != expected:
        raise ValueError("diagnostic source MPH identity is not stable")
    for role, path in (
        ("diagnostic_state_raw", state_path),
        ("diagnostic_force_raw", force_path),
    ):
        item = _artifact_by_role(raw, role)
        if item.get("path") != path.name or item.get("sha256") != _sha256(path):
            raise ValueError(f"execution record does not bind {path}")
    return {
        "path": str(record_path),
        "sha256": _sha256(record_path),
        "source_mph_sha256": before,
        "source_unchanged": True,
    }


def _vectors(values: Mapping[str, FloatArray], first: str, second: str) -> FloatArray:
    return np.column_stack((values[first], values[second]))


def _active_diagnostic_rows(
    state_all: dict[str, FloatArray],
    force_all: dict[str, FloatArray],
) -> tuple[dict[str, FloatArray], dict[str, FloatArray], NDArray[np.bool_]]:
    if not (
        np.array_equal(state_all["particle_id"], force_all["particle_id"])
        and np.array_equal(state_all["time_s"], force_all["time_s"])
    ):
        raise ValueError("diagnostic state and force keys differ")
    active = state_all["current_status_code"] == 1.0
    if not bool(active.any()):
        raise ValueError("diagnostic export has no active rows")
    state = {name: values[active] for name, values in state_all.items()}
    force = {name: values[active] for name, values in force_all.items()}
    if not all(bool(np.isfinite(values).all()) for values in (*state.values(), *force.values())):
        raise ValueError("active diagnostic rows contain nonfinite values")
    return state, force, active


def _scalar(values: Mapping[str, FloatArray], name: str) -> FloatArray:
    result = values[name]
    if result.ndim != 2 or result.shape[1] != 1:
        raise ValueError(f"sampled scalar {name} has an invalid shape")
    return result[:, 0]


def _metric(candidate: FloatArray, reference: FloatArray, keys: Mapping[str, FloatArray]) -> Record:
    if candidate.shape != reference.shape or candidate.ndim not in (1, 2):
        raise ValueError("diagnostic comparison arrays have incompatible shapes")
    difference = candidate - reference
    residual = np.abs(difference) if candidate.ndim == 1 else np.linalg.norm(difference, axis=1)
    candidate_norm = np.abs(candidate) if candidate.ndim == 1 else np.linalg.norm(candidate, axis=1)
    reference_norm = np.abs(reference) if reference.ndim == 1 else np.linalg.norm(reference, axis=1)
    denominator = max(float(np.linalg.norm(reference)), float(np.finfo(np.float64).tiny))
    relative_l2 = float(np.linalg.norm(difference) / denominator)
    local_scale = np.maximum(np.maximum(candidate_norm, reference_norm), np.finfo(np.float64).tiny)
    normalized = residual / local_scale
    worst = int(np.argmax(residual))
    return {
        "relative_l2": relative_l2,
        "rms": float(np.sqrt(np.mean(residual * residual))),
        "maximum_absolute_or_vector": float(residual[worst]),
        "scale_normalized_p99": float(np.quantile(normalized, 0.99)),
        "worst_key": {
            "particle_id": int(keys["particle_id"][worst]),
            "time_s": float(keys["time_s"][worst]),
        },
        "passed": relative_l2 <= FORMULA_PARITY_RELATIVE_L2_LIMIT,
    }


def _collection_rates(
    charge_number: FloatArray,
    radius_m: FloatArray,
    velocity_m_s: FloatArray,
    values: Mapping[str, FloatArray],
    plan: AggregateRelativeDriftChargePlan,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if not plan.has_negative_ion_current:
        raise ValueError("M3-C3 diagnostic requires the aggregate three-current revision")
    required_names = (
        plan.negative_ion_number_density_field,
        plan.negative_ion_thermal_voltage_field,
        plan.negative_ion_velocity_field,
        plan.effective_negative_ion_mass_field,
    )
    if any(name is None for name in required_names):
        raise ValueError("M3-C3 three-current plan is incomplete")
    negative_density_name = cast(str, plan.negative_ion_number_density_field)
    negative_voltage_name = cast(str, plan.negative_ion_thermal_voltage_field)
    negative_velocity_name = cast(str, plan.negative_ion_velocity_field)
    negative_mass_name = cast(str, plan.effective_negative_ion_mass_field)
    evaluation = aggregate_relative_drift_regularized_three_current_v1(
        charge_number=charge_number,
        electrostatic_radius_m=radius_m,
        electron_number_density_m3=_scalar(values, plan.electron_number_density_field),
        positive_ion_number_density_m3=_scalar(values, plan.positive_ion_number_density_field),
        negative_ion_number_density_m3=_scalar(values, negative_density_name),
        electron_thermal_voltage_V=_scalar(values, plan.electron_thermal_voltage_field),
        positive_ion_thermal_voltage_V=_scalar(values, plan.positive_ion_thermal_voltage_field),
        negative_ion_thermal_voltage_V=_scalar(values, negative_voltage_name),
        particle_velocity_m_s=velocity_m_s,
        positive_ion_velocity_m_s=values[plan.positive_ion_velocity_field],
        negative_ion_velocity_m_s=values[negative_velocity_name],
        effective_positive_ion_mass_kg=_scalar(values, plan.effective_positive_ion_mass_field),
        effective_negative_ion_mass_kg=_scalar(values, negative_mass_name),
        screening_length_m=_scalar(values, plan.screening_length_field),
        maximum_relative_ion_speed_m_s=plan.maximum_relative_ion_speed_m_s,
    )
    potential = evaluation.surface_potential_V
    nonpositive = potential <= 0.0
    positive_energy = evaluation.positive_effective_ion_energy_V
    negative_energy = evaluation.negative_effective_ion_energy_V
    electron_voltage = _scalar(values, plan.electron_thermal_voltage_field)
    positive_factor = np.where(
        nonpositive,
        1.0 - potential / positive_energy,
        np.exp(
            np.clip(-potential / positive_energy, AGGREGATE_EXPONENT_MIN, AGGREGATE_EXPONENT_MAX)
        ),
    )
    electron_factor = np.where(
        nonpositive,
        np.exp(
            np.clip(potential / electron_voltage, AGGREGATE_EXPONENT_MIN, AGGREGATE_EXPONENT_MAX)
        ),
        1.0 + potential / electron_voltage,
    )
    negative_factor = np.where(
        nonpositive,
        np.exp(
            np.clip(potential / negative_energy, AGGREGATE_EXPONENT_MIN, AGGREGATE_EXPONENT_MAX)
        ),
        1.0 + potential / negative_energy,
    )
    area = math.pi * radius_m**2
    positive = (
        area
        * _scalar(values, plan.positive_ion_number_density_field)
        * evaluation.positive_effective_ion_speed_m_s
        * positive_factor
    )
    electron = (
        area
        * _scalar(values, plan.electron_number_density_field)
        * np.sqrt(8.0 * ELEMENTARY_CHARGE_C * electron_voltage / (math.pi * ELECTRON_MASS_KG))
        * electron_factor
    )
    negative = (
        area
        * _scalar(values, negative_density_name)
        * evaluation.negative_effective_ion_speed_m_s
        * negative_factor
    )
    return evaluation.charge_rate_number_s, positive, electron, negative


def _production_values(
    case_path: Path,
    state: Mapping[str, FloatArray],
) -> tuple[dict[str, FloatArray], FloatArray, FloatArray, Record]:
    case = load_case(case_path)
    plan = resolve_physics_plan(case.spec.physics.models, case.data.coordinate_system)
    expected_types = (
        (plan.charge, AggregateRelativeDriftChargePlan),
        (plan.drag, EpsteinDragPlan),
        (plan.electric, ElectricPlan),
        (plan.ion_drag, RelativeFlowScreenedIonDragPlan),
        (plan.thermophoresis, WaldmannGallisThermophoresisPlan),
        (plan.dielectrophoresis, QuasistaticSphericalDielectrophoresisPlan),
        (plan.lift, RarefiedVorticityLiftPlan),
        (plan.gravity_buoyancy, GravityBuoyancyPlan),
    )
    if any(not isinstance(value, expected) for value, expected in expected_types):
        raise ValueError("case does not select the fixed M3-C3 deterministic physics stack")
    charge_plan = cast(AggregateRelativeDriftChargePlan, plan.charge)
    drag_plan = cast(EpsteinDragPlan, plan.drag)
    electric_plan = cast(ElectricPlan, plan.electric)
    ion_plan = cast(RelativeFlowScreenedIonDragPlan, plan.ion_drag)
    thermo_plan = cast(WaldmannGallisThermophoresisPlan, plan.thermophoresis)
    dep_plan = cast(QuasistaticSphericalDielectrophoresisPlan, plan.dielectrophoresis)
    lift_plan = cast(RarefiedVorticityLiftPlan, plan.lift)
    gravity_plan = cast(GravityBuoyancyPlan, plan.gravity_buoyancy)

    requirements = {
        item.name: RequiredFieldMetadata(
            item.unit, item.components, item.stored_basis, item.positive
        )
        for item in plan.required_fields
    }
    fields = prepare_required_fields(case.data, requirements)
    position = _vectors(state, "r_m", "z_m")
    sampled = fields.sample(position)
    if not bool(sampled.support_inside.all()):
        raise ValueError("production field does not cover every active COMSOL state")

    geometry = prepare_geometry(case.data.geometry, case.data.coordinate_system)
    schedule = realize_sources(case, geometry)
    particle_id = np.rint(state["particle_id"]).astype(np.int64)
    resident = np.searchsorted(schedule.particle_id, particle_id)
    if not np.array_equal(schedule.particle_id[resident], particle_id):
        raise ValueError("COMSOL particle IDs do not match the production source schedule")
    mass = schedule.mass_kg[resident]
    diameter = schedule.drag_diameter_m[resident]
    radius = schedule.electrostatic_radius_m[resident]
    volume = schedule.displaced_volume_m3[resident]
    velocity = _vectors(state, "velocity_r_m_per_s", "velocity_z_m_per_s")
    charge_number = state["charge_number_e"]
    values = sampled.values

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
    runtime_evaluation = runtime.evaluate(resident, velocity, charge_number, values)

    charge_rate, positive_rate, electron_rate, negative_rate = _collection_rates(
        charge_number, radius, velocity, values, charge_plan
    )
    electric_acceleration = np.zeros((mass.size, 2), dtype=np.float64)
    add_electric_coulomb_acceleration(
        electric_acceleration,
        charge_number=charge_number,
        mass_kg=mass,
        electric_field_V_m=values[electric_plan.electric_field],
    )
    ion = relative_flow_screened_collection_orbital_ion_drag(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        charge_number=charge_number,
        velocity_m_s=velocity,
        positive_ion_number_density_m3=_scalar(values, ion_plan.positive_ion_number_density_field),
        positive_ion_thermal_voltage_V=_scalar(values, ion_plan.positive_ion_thermal_voltage_field),
        positive_ion_velocity_m_s=values[ion_plan.positive_ion_velocity_field],
        effective_positive_ion_mass_kg=_scalar(values, ion_plan.effective_positive_ion_mass_field),
        screening_length_m=_scalar(values, ion_plan.screening_length_field),
        ion_neutral_mean_free_path_m=_scalar(values, ion_plan.ion_neutral_mean_free_path_field),
        maximum_relative_ion_speed_m_s=ion_plan.maximum_relative_ion_speed_m_s,
    )
    gas_velocity = values[drag_plan.gas_velocity_field]
    drag = epstein_linear_relaxation(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=velocity,
        gas_velocity_m_s=gas_velocity,
        gas_density_kg_m3=_scalar(values, drag_plan.gas_density_field),
        gas_temperature_K=_scalar(values, drag_plan.gas_temperature_field),
        gas_mean_free_path_m=_scalar(values, drag_plan.gas_mean_free_path_field),
        gas_molecular_mass_kg=drag_plan.gas_molecular_mass_kg,
        delta=drag_plan.delta,
        maximum_speed_ratio=drag_plan.maximum_speed_ratio,
    )
    thermo = waldmann_gallis_thermophoresis(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=velocity,
        gas_velocity_m_s=values[thermo_plan.gas_velocity_field],
        gas_temperature_K=_scalar(values, thermo_plan.gas_temperature_field),
        gas_translational_heat_flux_W_m2=values[thermo_plan.gas_translational_heat_flux_field],
        gas_mean_free_path_m=_scalar(values, thermo_plan.gas_mean_free_path_field),
        gas_molecular_mass_kg=thermo_plan.gas_molecular_mass_kg,
        maximum_speed_ratio=thermo_plan.maximum_speed_ratio,
    )
    lift = rarefied_vorticity_lift(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=velocity,
        gas_velocity_m_s=values[lift_plan.gas_velocity_field],
        gas_density_kg_m3=_scalar(values, lift_plan.gas_density_field),
        gas_mean_free_path_m=_scalar(values, lift_plan.gas_mean_free_path_field),
        azimuthal_gas_vorticity_s_inv=_scalar(values, lift_plan.azimuthal_gas_vorticity_field),
        lift_coefficient=lift_plan.lift_coefficient,
    )
    dep = quasistatic_spherical_dep_acceleration(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        gradient_mean_e_squared_V2_m3=values[dep_plan.gradient_mean_e_squared_field],
        medium_relative_permittivity=dep_plan.medium_relative_permittivity,
        real_clausius_mossotti_factor=dep_plan.real_clausius_mossotti_factor,
    )
    gravity = np.zeros((mass.size, 2), dtype=np.float64)
    add_gravity_buoyancy_acceleration(
        gravity,
        mass_kg=mass,
        displaced_volume_m3=volume,
        gas_density_kg_m3=_scalar(values, gravity_plan.gas_density_field),
        gravity_m_s2=gravity_plan.gravity_m_s2,
    )
    forces = {
        "electric": electric_acceleration * mass[:, None],
        "ion_drag": ion.acceleration_m_s2 * mass[:, None],
        "epstein_drag": drag.rate_s_inv[:, None] * (gas_velocity - velocity) * mass[:, None],
        "thermophoresis": thermo.acceleration_m_s2 * mass[:, None],
        "lift": lift.acceleration_m_s2 * mass[:, None],
        "dep": dep * mass[:, None],
        "gravity_buoyancy": gravity * mass[:, None],
    }
    total = sum(forces.values(), start=np.zeros_like(next(iter(forces.values()))))
    coverage = {
        "field_support_inside_fraction": float(np.mean(sampled.support_inside)),
        "runtime_applicable_fraction": float(np.mean(runtime_evaluation.applicable)),
        "charge_applicable_fraction": float(
            np.mean(
                aggregate_relative_drift_regularized_three_current_v1(
                    charge_number=charge_number,
                    electrostatic_radius_m=radius,
                    electron_number_density_m3=_scalar(
                        values, charge_plan.electron_number_density_field
                    ),
                    positive_ion_number_density_m3=_scalar(
                        values, charge_plan.positive_ion_number_density_field
                    ),
                    negative_ion_number_density_m3=_scalar(
                        values, cast(str, charge_plan.negative_ion_number_density_field)
                    ),
                    electron_thermal_voltage_V=_scalar(
                        values, charge_plan.electron_thermal_voltage_field
                    ),
                    positive_ion_thermal_voltage_V=_scalar(
                        values, charge_plan.positive_ion_thermal_voltage_field
                    ),
                    negative_ion_thermal_voltage_V=_scalar(
                        values, cast(str, charge_plan.negative_ion_thermal_voltage_field)
                    ),
                    particle_velocity_m_s=velocity,
                    positive_ion_velocity_m_s=values[charge_plan.positive_ion_velocity_field],
                    negative_ion_velocity_m_s=values[
                        cast(str, charge_plan.negative_ion_velocity_field)
                    ],
                    effective_positive_ion_mass_kg=_scalar(
                        values, charge_plan.effective_positive_ion_mass_field
                    ),
                    effective_negative_ion_mass_kg=_scalar(
                        values, cast(str, charge_plan.effective_negative_ion_mass_field)
                    ),
                    screening_length_m=_scalar(values, charge_plan.screening_length_field),
                    maximum_relative_ion_speed_m_s=(charge_plan.maximum_relative_ion_speed_m_s),
                ).applicable
            )
        ),
        "ion_drag_applicable_fraction": float(np.mean(ion.applicable)),
        "epstein_applicable_fraction": float(np.mean(drag.applicable)),
        "thermophoresis_applicable_fraction": float(np.mean(thermo.applicable)),
        "lift_applicable_fraction": float(np.mean(lift.applicable)),
    }
    return (
        forces,
        charge_rate,
        np.column_stack((positive_rate, electron_rate, negative_rate)),
        {
            "coverage": coverage,
            "runtime_acceleration_m_s2": runtime_evaluation.acceleration_m_s2,
            "runtime_charge_rate_number_s": runtime_evaluation.charge_rate_number_s,
            "component_total_force_N": total,
            "mass_kg": mass,
        },
    )


def diagnose(
    case_path: Path,
    record_path: Path,
    state_path: Path,
    force_path: Path,
) -> Record:
    """Evaluate one locked fine-step diagnostic export."""

    case_path = case_path.expanduser().resolve()
    record_path = record_path.expanduser().resolve()
    state_path = state_path.expanduser().resolve()
    force_path = force_path.expanduser().resolve()
    provenance = _validate_execution_record(record_path, state_path, force_path)
    state_all = _read_wide(state_path, STATE_COLUMNS)
    force_all = _read_wide(force_path, FORCE_COLUMNS)
    state, force, active = _active_diagnostic_rows(state_all, force_all)

    production_forces, production_rate, production_currents, internal = _production_values(
        case_path, state
    )
    keys = {"particle_id": state["particle_id"], "time_s": state["time_s"]}
    exported_forces = {
        "electric": _vectors(force, "electric_force_r_N", "electric_force_z_N"),
        "ion_drag": _vectors(force, "ion_drag_force_r_N", "ion_drag_force_z_N"),
        "epstein_drag": _vectors(force, "epstein_drag_force_r_N", "epstein_drag_force_z_N"),
        "thermophoresis": _vectors(force, "thermophoretic_force_r_N", "thermophoretic_force_z_N"),
        "lift": _vectors(force, "lift_force_r_N", "lift_force_z_N"),
        "dep": _vectors(force, "dep_force_r_N", "dep_force_z_N"),
        "gravity_buoyancy": _vectors(
            force, "gravity_buoyancy_force_r_N", "gravity_buoyancy_force_z_N"
        ),
    }
    exported_total = _vectors(force, "total_force_r_N", "total_force_z_N")
    exported_currents = np.column_stack(
        (
            force["positive_collection_rate_number_s"],
            force["electron_collection_rate_number_s"],
            force["negative_collection_rate_number_s"],
        )
    )
    comparisons = {
        name: _metric(production_forces[name], exported_forces[name], keys)
        for name in production_forces
    }
    comparisons["total_force"] = _metric(
        cast(FloatArray, internal["component_total_force_N"]), exported_total, keys
    )
    comparisons["charge_rate"] = _metric(production_rate, force["charge_rate_number_s"], keys)
    for index, name in enumerate(("positive_current", "electron_current", "negative_current")):
        comparisons[name] = _metric(
            production_currents[:, index], exported_currents[:, index], keys
        )

    mass = cast(FloatArray, internal["mass_kg"])
    runtime_acceleration = cast(FloatArray, internal["runtime_acceleration_m_s2"])
    runtime_rate = cast(FloatArray, internal["runtime_charge_rate_number_s"])
    component_acceleration = cast(FloatArray, internal["component_total_force_N"]) / mass[:, None]
    runtime_force_metric = _metric(runtime_acceleration, component_acceleration, keys)
    runtime_rate_metric = _metric(runtime_rate, production_rate, keys)
    runtime_internal_pass = (
        cast(float, runtime_force_metric["relative_l2"]) <= RUNTIME_DECOMPOSITION_RELATIVE_L2_LIMIT
        and cast(float, runtime_rate_metric["relative_l2"])
        <= RUNTIME_DECOMPOSITION_RELATIVE_L2_LIMIT
    )
    coverage = cast(dict[str, float], internal["coverage"])
    coverage_pass = all(value == 1.0 for value in coverage.values())
    failed = [name for name, value in comparisons.items() if value["passed"] is not True]
    status = "PASS" if not failed and runtime_internal_pass and coverage_pass else "BLOCKED"
    return {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": status,
        "scope": "same_state_active_rows_formula_and_runtime_parity",
        "artifacts": {
            "case": {"path": str(case_path), "sha256": _sha256(case_path)},
            "state": {"path": str(state_path), "sha256": _sha256(state_path)},
            "force": {"path": str(force_path), "sha256": _sha256(force_path)},
            "execution": provenance,
        },
        "rows": {
            "raw": EXPECTED_PARTICLES * EXPECTED_FRAMES,
            "active": int(np.count_nonzero(active)),
        },
        "formula_parity_relative_l2_limit": FORMULA_PARITY_RELATIVE_L2_LIMIT,
        "runtime_decomposition_relative_l2_limit": (RUNTIME_DECOMPOSITION_RELATIVE_L2_LIMIT),
        "production_vs_comsol": comparisons,
        "production_runtime_internal": {
            "status": "PASS" if runtime_internal_pass else "BLOCKED",
            "acceleration": runtime_force_metric,
            "charge_rate": runtime_rate_metric,
        },
        "applicability": {"status": "PASS" if coverage_pass else "BLOCKED", **coverage},
        "failed_comparisons": failed,
        "interpretation": (
            "PASS identifies same-state RHS parity within the predeclared formula tolerance. "
            "It does not relax the trajectory gate, validate the physical closure, or claim "
            "general COMSOL equivalence."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("execution_record", type=Path)
    parser.add_argument("state_raw_wide", type=Path)
    parser.add_argument("force_raw_wide", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = diagnose(
        args.case,
        args.execution_record,
        args.state_raw_wide,
        args.force_raw_wide,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", errors="strict") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
