"""Compare the finest M3-C0b saved states with producer and production formulas.

This external V&V tool intentionally separates three questions: whether the
saved COMSOL force can be replayed with its documented producer convention,
what the production pure physics model returns at the same frozen state, and
whether that saved state is inside the model's local applicability domain.
It does not integrate a trajectory or certify a continuous accepted path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from chamber_particles.physics.charge import (
    aggregate_relative_drift_regularized_two_current_v1,
)
from chamber_particles.physics.forces import (
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
    add_electric_coulomb_acceleration,
    add_gravity_buoyancy_acceleration,
    epstein_linear_relaxation,
    quasistatic_spherical_dep_acceleration,
    rarefied_vorticity_lift,
    relative_flow_screened_collection_orbital_ion_drag,
    waldmann_gallis_thermophoresis,
)
from tools.vv.comsol.m3c0b_pilot import COLUMNS, RAW_EXPORT_TABLES, iter_combined_records

type FloatArray = NDArray[np.float64]
type Record = dict[str, Any]

TOOL_REVISION: Final = "m3c1_pre_event_frozen_rhs_v1"
MODEL_NAMES: Final = (
    "dynamic_charge",
    "electric",
    "relative_flow_ion_drag",
    "epstein_drag",
    "waldmann_thermophoresis",
    "free_molecular_lift_sensitivity",
    "dielectrophoresis",
    "gravity_buoyancy",
)


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """Narrow protocol for the accepted Case-A 100 nm pre-event slice."""

    source: Path
    step_directory: str
    expected_records: int
    expected_particles: int
    expected_output_times: int
    source_model_sha256: str
    raw_sha256: dict[str, str]
    electron_thermal_voltage_V: float
    neutral_molecular_mass_kg: float
    epstein_delta: float
    lift_coefficient: float
    medium_relative_permittivity: float
    clausius_mossotti_factor: float
    gravity_m_s2: tuple[float, float]
    relative_speed_regularization_m_s: float
    minimum_ion_energy_V: float
    maximum_relative_ion_speed_m_s: float
    maximum_effective_gas_speed_ratio: float
    producer_residual_limit: float
    production_residual_limit: float
    thermophoretic_primitive_authority: str
    known_constant_convention_models: frozenset[str]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _raw_hashes(value: object) -> dict[str, str]:
    raw_hashes = _mapping(value, "input.raw_sha256")
    if set(raw_hashes) != set(RAW_EXPORT_TABLES):
        raise ValueError("input.raw_sha256 must identify the five M3-C0b raw tables")
    hashes = {name: str(item) for name, item in raw_hashes.items()}
    if any(len(item) != 64 for item in hashes.values()):
        raise ValueError("input raw hashes must be SHA-256 strings")
    return hashes


def _gravity(value: object) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("parameters.gravity_m_s2 must be a two-vector")
    gravity = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in gravity):
        raise ValueError("parameters.gravity_m_s2 must be finite")
    return gravity


def _relative_permittivity(parameters: dict[str, Any]) -> float:
    value = _positive(
        parameters.get("particle_relative_permittivity"),
        "particle_relative_permittivity",
    )
    if value <= 1.0:
        raise ValueError("particle_relative_permittivity must exceed one for this case")
    return value


def _known_models(value: object) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("known_constant_convention_models must be a string list")
    models = frozenset(value)
    if not models <= set(MODEL_NAMES):
        raise ValueError("known_constant_convention_models contains an unknown model")
    return models


def _diffuse_fraction(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("epstein_diffuse_reflection_fraction must be in [0, 1]")
    return result


def load_config(path: Path) -> FrozenConfig:
    """Load the fixed pre-event comparison protocol."""

    source = path.expanduser().resolve()
    raw = _mapping(json.loads(source.read_text(encoding="utf-8")), "configuration")
    if raw.get("schema_version") != 1 or raw.get("evaluation_revision") != 1:
        raise ValueError("configuration must select M3-C1 frozen-RHS revision 1")
    input_config = _mapping(raw.get("input"), "input")
    parameters = _mapping(raw.get("parameters"), "parameters")
    comparison = _mapping(raw.get("comparison"), "comparison")
    hashes = _raw_hashes(input_config.get("raw_sha256"))
    gravity = _gravity(parameters.get("gravity_m_s2"))
    relative_permittivity = _relative_permittivity(parameters)
    known = _known_models(comparison.get("known_constant_convention_models"))
    diffuse = _diffuse_fraction(parameters.get("epstein_diffuse_reflection_fraction"))
    return FrozenConfig(
        source=source,
        step_directory=str(input_config.get("step_directory")),
        expected_records=int(input_config.get("expected_records", 0)),
        expected_particles=int(input_config.get("expected_particles", 0)),
        expected_output_times=int(input_config.get("expected_output_times", 0)),
        source_model_sha256=str(input_config.get("source_model_sha256")),
        raw_sha256=hashes,
        electron_thermal_voltage_V=_positive(
            parameters.get("electron_thermal_voltage_V"),
            "electron_thermal_voltage_V",
        ),
        neutral_molecular_mass_kg=_positive(
            parameters.get("neutral_molecular_mass_kg"),
            "neutral_molecular_mass_kg",
        ),
        epstein_delta=1.0 + diffuse * math.pi / 8.0,
        lift_coefficient=_positive(parameters.get("lift_coefficient"), "lift_coefficient"),
        medium_relative_permittivity=_positive(
            parameters.get("medium_relative_permittivity"),
            "medium_relative_permittivity",
        ),
        clausius_mossotti_factor=(relative_permittivity - 1.0) / (relative_permittivity + 2.0),
        gravity_m_s2=gravity,
        relative_speed_regularization_m_s=_positive(
            parameters.get("relative_speed_regularization_m_s"),
            "relative_speed_regularization_m_s",
        ),
        minimum_ion_energy_V=_positive(
            parameters.get("minimum_ion_energy_V"),
            "minimum_ion_energy_V",
        ),
        maximum_relative_ion_speed_m_s=_positive(
            parameters.get("maximum_relative_ion_speed_m_s"),
            "maximum_relative_ion_speed_m_s",
        ),
        maximum_effective_gas_speed_ratio=_positive(
            parameters.get("maximum_effective_gas_speed_ratio"),
            "maximum_effective_gas_speed_ratio",
        ),
        producer_residual_limit=_positive(
            comparison.get("producer_scale_normalized_residual_limit"),
            "producer_scale_normalized_residual_limit",
        ),
        production_residual_limit=_positive(
            comparison.get("production_scale_normalized_residual_limit"),
            "production_scale_normalized_residual_limit",
        ),
        thermophoretic_primitive_authority=str(
            comparison.get("thermophoretic_primitive_authority")
        ),
        known_constant_convention_models=known,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(step_directory: Path, config: FrozenConfig) -> dict[str, FloatArray]:
    for name, expected in config.raw_sha256.items():
        path = step_directory / name
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"raw artifact hash mismatch for {path}: {actual}")
    rows = tuple(iter_combined_records(step_directory))
    if len(rows) != config.expected_records:
        raise ValueError(f"expected {config.expected_records} records, found {len(rows)}")
    values = {
        name: np.fromiter((row[name] for row in rows), dtype=np.float64, count=len(rows))
        for name in COLUMNS
    }
    particle_id = values["particle_id"]
    time_s = values["time_s"]
    if (
        np.unique(particle_id).size != config.expected_particles
        or np.unique(time_s).size != config.expected_output_times
    ):
        raise ValueError("particle/time cardinality does not match the comparison protocol")
    if not all(bool(np.isfinite(column).all()) for column in values.values()):
        raise ValueError("pre-event frozen-state input contains a non-finite value")
    if not bool((values["current_status_code"] == 1.0).all()):
        raise ValueError("pre-event frozen-state input contains a non-active row")
    return values


def _verify_provenance(step_directory: Path, config: FrozenConfig) -> dict[str, object]:
    path = step_directory.parent / "provenance.json"
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "provenance")
    before = str(payload.get("source_sha256_before"))
    after = str(payload.get("source_sha256_after"))
    if before != config.source_model_sha256 or after != config.source_model_sha256:
        raise ValueError("source model hash does not match the frozen-RHS protocol")
    if (
        payload.get("source_load_mode") != "ModelUtil.loadCopy"
        or payload.get("model_saved") is not False
    ):
        raise ValueError("M3-C0b source isolation provenance is not valid")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "source_model_sha256": before,
        "source_unchanged": before == after,
        "source_load_mode": payload.get("source_load_mode"),
        "model_saved": payload.get("model_saved"),
    }


def _vectors(values: dict[str, FloatArray], first: str, second: str) -> FloatArray:
    return np.column_stack((values[first], values[second]))


def _saved_epsilon0(values: dict[str, FloatArray]) -> FloatArray:
    radius = 0.5 * values["particle_diameter_m"]
    screening = values["screening_length_m"]
    phi1 = values["particle_surface_potential_per_charge_V"]
    return ELEMENTARY_CHARGE_C / (4.0 * math.pi * radius * (1.0 + radius / screening) * phi1)


def _producer_charge(values: dict[str, FloatArray], config: FrozenConfig) -> FloatArray:
    radius = 0.5 * values["particle_diameter_m"]
    particle_velocity = _vectors(values, "velocity_r_m_per_s", "velocity_z_m_per_s")
    ion_velocity = _vectors(values, "ion_velocity_r_m_per_s", "ion_velocity_z_m_per_s")
    relative_speed_squared = np.sum((ion_velocity - particle_velocity) ** 2, axis=1)
    effective_speed_squared = (
        relative_speed_squared
        + 8.0
        * ELEMENTARY_CHARGE_C
        * values["ion_thermal_energy_eV_as_V"]
        / (math.pi * values["positive_ion_mass_kg"])
        + config.relative_speed_regularization_m_s**2
    )
    effective_speed = np.sqrt(effective_speed_squared)
    ion_energy = np.maximum(
        values["positive_ion_mass_kg"] * effective_speed_squared / (2.0 * ELEMENTARY_CHARGE_C),
        config.minimum_ion_energy_V,
    )
    surface_potential = (
        values["charge_number_e"] * values["particle_surface_potential_per_charge_V"]
    )
    nonpositive = surface_potential <= 0.0
    electron_argument = surface_potential / config.electron_thermal_voltage_V
    ion_argument = -surface_potential / ion_energy
    electron_factor = np.where(
        nonpositive,
        np.exp(np.clip(electron_argument, -50.0, 50.0)),
        1.0 + electron_argument,
    )
    ion_factor = np.where(
        nonpositive,
        1.0 + ion_argument,
        np.exp(np.clip(ion_argument, -50.0, 50.0)),
    )
    ion_rate = (
        math.pi * radius**2 * values["positive_ion_density_per_m3"] * effective_speed * ion_factor
    )
    electron_rate = values["charging_current_scale_per_s"] * electron_factor
    return ion_rate - electron_rate


def _producer_ion_drag(
    values: dict[str, FloatArray], config: FrozenConfig, epsilon0: FloatArray
) -> FloatArray:
    radius = 0.5 * values["particle_diameter_m"]
    charge = values["charge_number_e"]
    ion_mass = values["positive_ion_mass_kg"]
    particle_velocity = _vectors(values, "velocity_r_m_per_s", "velocity_z_m_per_s")
    ion_velocity = _vectors(values, "ion_velocity_r_m_per_s", "ion_velocity_z_m_per_s")
    relative_velocity = ion_velocity - particle_velocity
    speed_squared = (
        np.sum(relative_velocity**2, axis=1)
        + 8.0 * ELEMENTARY_CHARGE_C * values["ion_thermal_energy_eV_as_V"] / (math.pi * ion_mass)
        + config.relative_speed_regularization_m_s**2
    )
    effective_speed = np.sqrt(speed_squared)
    surface_potential = charge * values["particle_surface_potential_per_charge_V"]
    screening_radius = np.maximum(
        radius,
        np.minimum(values["screening_length_m"], values["ion_neutral_mean_free_path_m"]),
    )
    orbital_impact = (
        np.sqrt(charge**2 + 1.0e-20)
        * ELEMENTARY_CHARGE_C**2
        / (4.0 * math.pi * epsilon0 * ion_mass * speed_squared)
    )
    collection_square = np.minimum(
        screening_radius**2,
        radius**2
        * np.maximum(
            0.0,
            1.0 - 2.0 * ELEMENTARY_CHARGE_C * surface_potential / (ion_mass * speed_squared),
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
    factor = values["positive_ion_density_per_m3"] * ion_mass * effective_speed * cross_section
    return factor[:, None] * relative_velocity


def _producer_forces(
    values: dict[str, FloatArray], config: FrozenConfig, epsilon0: FloatArray
) -> dict[str, FloatArray]:
    mass = values["particle_mass_kg"]
    diameter = values["particle_diameter_m"]
    radius = 0.5 * diameter
    particle_velocity = _vectors(values, "velocity_r_m_per_s", "velocity_z_m_per_s")
    gas_velocity = _vectors(values, "gas_velocity_r_m_per_s", "gas_velocity_z_m_per_s")
    relative_velocity = gas_velocity - particle_velocity
    thermal_speed = np.sqrt(
        8.0
        * 1.380649e-23
        * values["gas_temperature_K"]
        / (math.pi * config.neutral_molecular_mass_kg)
    )
    drag_factor = (
        4.0
        * math.pi
        / 3.0
        * radius**2
        * values["gas_density_kg_per_m3"]
        * thermal_speed
        * config.epstein_delta
    )
    heat_flux = _vectors(
        values,
        "effective_heat_flux_r_W_per_m2",
        "effective_heat_flux_z_W_per_m2",
    )
    vorticity = values["azimuthal_vorticity_per_s"]
    lift_factor = (
        config.lift_coefficient
        * math.pi
        * values["gas_density_kg_per_m3"]
        * values["gas_mean_free_path_m"]
        * radius**2
        * vorticity
    )
    lift = np.empty_like(relative_velocity)
    lift[:, 0] = lift_factor * relative_velocity[:, 1]
    lift[:, 1] = -lift_factor * relative_velocity[:, 0]
    gradient = _vectors(values, "gradient_E2_r_V2_per_m3", "gradient_E2_z_V2_per_m3")
    volume = math.pi * diameter**3 / 6.0
    buoyancy_factor = 1.0 - values["gas_density_kg_per_m3"] * volume / mass
    return {
        "electric": values["charge_number_e"][:, None]
        * ELEMENTARY_CHARGE_C
        * _vectors(values, "electric_field_r_V_per_m", "electric_field_z_V_per_m"),
        "relative_flow_ion_drag": _producer_ion_drag(values, config, epsilon0),
        "epstein_drag": drag_factor[:, None] * relative_velocity,
        "waldmann_thermophoresis": ((32.0 / 15.0) * radius**2 / thermal_speed)[:, None] * heat_flux,
        "free_molecular_lift_sensitivity": lift,
        "dielectrophoresis": (
            2.0
            * math.pi
            * epsilon0
            * radius**3
            * config.medium_relative_permittivity
            * config.clausius_mossotti_factor
        )[:, None]
        * gradient,
        "gravity_buoyancy": mass[:, None]
        * buoyancy_factor[:, None]
        * np.asarray(config.gravity_m_s2, dtype=np.float64)[None, :],
    }


def _production_evaluations(
    values: dict[str, FloatArray], config: FrozenConfig
) -> tuple[FloatArray, dict[str, FloatArray], dict[str, Record]]:
    mass = values["particle_mass_kg"]
    diameter = values["particle_diameter_m"]
    radius = 0.5 * diameter
    particle_velocity = _vectors(values, "velocity_r_m_per_s", "velocity_z_m_per_s")
    gas_velocity = _vectors(values, "gas_velocity_r_m_per_s", "gas_velocity_z_m_per_s")
    ion_velocity = _vectors(values, "ion_velocity_r_m_per_s", "ion_velocity_z_m_per_s")
    charge = aggregate_relative_drift_regularized_two_current_v1(
        charge_number=values["charge_number_e"],
        electrostatic_radius_m=radius,
        electron_number_density_m3=values["electron_density_per_m3"],
        positive_ion_number_density_m3=values["positive_ion_density_per_m3"],
        electron_thermal_voltage_V=np.full_like(radius, config.electron_thermal_voltage_V),
        positive_ion_thermal_voltage_V=values["ion_thermal_energy_eV_as_V"],
        particle_velocity_m_s=particle_velocity,
        positive_ion_velocity_m_s=ion_velocity,
        effective_positive_ion_mass_kg=values["positive_ion_mass_kg"],
        screening_length_m=values["screening_length_m"],
        maximum_relative_ion_speed_m_s=config.maximum_relative_ion_speed_m_s,
    )
    electric_acceleration = np.zeros((mass.size, 2), dtype=np.float64)
    add_electric_coulomb_acceleration(
        electric_acceleration,
        charge_number=values["charge_number_e"],
        mass_kg=mass,
        electric_field_V_m=_vectors(values, "electric_field_r_V_per_m", "electric_field_z_V_per_m"),
    )
    ion_drag = relative_flow_screened_collection_orbital_ion_drag(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        charge_number=values["charge_number_e"],
        velocity_m_s=particle_velocity,
        positive_ion_number_density_m3=values["positive_ion_density_per_m3"],
        positive_ion_thermal_voltage_V=values["ion_thermal_energy_eV_as_V"],
        positive_ion_velocity_m_s=ion_velocity,
        effective_positive_ion_mass_kg=values["positive_ion_mass_kg"],
        screening_length_m=values["screening_length_m"],
        ion_neutral_mean_free_path_m=values["ion_neutral_mean_free_path_m"],
        maximum_relative_ion_speed_m_s=config.maximum_relative_ion_speed_m_s,
    )
    drag = epstein_linear_relaxation(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=particle_velocity,
        gas_velocity_m_s=gas_velocity,
        gas_density_kg_m3=values["gas_density_kg_per_m3"],
        gas_temperature_K=values["gas_temperature_K"],
        gas_mean_free_path_m=values["gas_mean_free_path_m"],
        gas_molecular_mass_kg=config.neutral_molecular_mass_kg,
        delta=config.epstein_delta,
        maximum_speed_ratio=config.maximum_effective_gas_speed_ratio,
    )
    thermophoresis = waldmann_gallis_thermophoresis(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=particle_velocity,
        gas_velocity_m_s=gas_velocity,
        gas_temperature_K=values["gas_temperature_K"],
        gas_translational_heat_flux_W_m2=_vectors(
            values,
            "effective_heat_flux_r_W_per_m2",
            "effective_heat_flux_z_W_per_m2",
        ),
        gas_mean_free_path_m=values["gas_mean_free_path_m"],
        gas_molecular_mass_kg=config.neutral_molecular_mass_kg,
        maximum_speed_ratio=config.maximum_effective_gas_speed_ratio,
    )
    lift = rarefied_vorticity_lift(
        mass_kg=mass,
        drag_diameter_m=diameter,
        velocity_m_s=particle_velocity,
        gas_velocity_m_s=gas_velocity,
        gas_density_kg_m3=values["gas_density_kg_per_m3"],
        gas_mean_free_path_m=values["gas_mean_free_path_m"],
        azimuthal_gas_vorticity_s_inv=values["azimuthal_vorticity_per_s"],
        lift_coefficient=config.lift_coefficient,
    )
    dep = quasistatic_spherical_dep_acceleration(
        mass_kg=mass,
        electrostatic_radius_m=radius,
        gradient_mean_e_squared_V2_m3=_vectors(
            values, "gradient_E2_r_V2_per_m3", "gradient_E2_z_V2_per_m3"
        ),
        medium_relative_permittivity=config.medium_relative_permittivity,
        real_clausius_mossotti_factor=config.clausius_mossotti_factor,
    )
    gravity_acceleration = np.zeros((mass.size, 2), dtype=np.float64)
    add_gravity_buoyancy_acceleration(
        gravity_acceleration,
        mass_kg=mass,
        displaced_volume_m3=math.pi * diameter**3 / 6.0,
        gas_density_kg_m3=values["gas_density_kg_per_m3"],
        gravity_m_s2=config.gravity_m_s2,
    )
    forces = {
        "electric": electric_acceleration * mass[:, None],
        "relative_flow_ion_drag": ion_drag.acceleration_m_s2 * mass[:, None],
        "epstein_drag": drag.rate_s_inv[:, None]
        * (gas_velocity - particle_velocity)
        * mass[:, None],
        "waldmann_thermophoresis": thermophoresis.acceleration_m_s2 * mass[:, None],
        "free_molecular_lift_sensitivity": lift.acceleration_m_s2 * mass[:, None],
        "dielectrophoresis": dep * mass[:, None],
        "gravity_buoyancy": gravity_acceleration * mass[:, None],
    }
    relative_gas_speed = np.linalg.norm(gas_velocity - particle_velocity, axis=1)
    gas_speed_ratio = relative_gas_speed / thermophoresis.mean_thermal_speed_m_s
    gas_mean_free_path_over_radius = values["gas_mean_free_path_m"] / radius
    charge_coverage = _coverage(charge.applicable, "declared relative-ion-speed envelope")
    charge_coverage["sampled_maximum_relative_ion_speed_m_s"] = float(
        np.max(charge.relative_ion_speed_m_s)
    )
    ion_drag_coverage = _coverage(ion_drag.applicable, "declared relative-ion-speed envelope")
    ion_drag_coverage["sampled_maximum_effective_ion_speed_m_s"] = float(
        np.max(ion_drag.effective_speed_m_s)
    )
    drag_coverage = _coverage(
        drag.applicable,
        "saved-row high-Kn and effective-gas speed-ratio gates",
    )
    drag_coverage["sampled_minimum_mean_free_path_over_radius"] = float(
        np.min(gas_mean_free_path_over_radius)
    )
    drag_coverage["sampled_maximum_relative_speed_ratio"] = float(np.max(gas_speed_ratio))
    thermophoretic_coverage = _coverage(
        thermophoresis.applicable,
        "saved-row high-Kn and effective-gas speed-ratio gates",
    )
    thermophoretic_coverage["sampled_minimum_mean_free_path_over_radius"] = float(
        np.min(thermophoresis.mean_free_path_over_radius)
    )
    thermophoretic_coverage["sampled_maximum_relative_speed_ratio"] = float(
        np.max(thermophoresis.relative_speed_ratio)
    )
    lift_coverage = _coverage(lift.applicable, "saved-row high-Kn gate")
    lift_coverage["sampled_minimum_mean_free_path_over_radius"] = float(
        np.min(lift.mean_free_path_over_radius)
    )
    applicability: dict[str, Record] = {
        "dynamic_charge": charge_coverage,
        "electric": _not_gated(),
        "relative_flow_ion_drag": ion_drag_coverage,
        "epstein_drag": drag_coverage,
        "waldmann_thermophoresis": thermophoretic_coverage,
        "free_molecular_lift_sensitivity": lift_coverage,
        "dielectrophoresis": {
            "saved_row_status": "NOT_TESTED_POINT_DIPOLE_CERTIFICATION_MISSING",
            "saved_row_coverage": None,
            "continuous_path_status": "NOT_TESTED",
        },
        "gravity_buoyancy": _not_gated(),
    }
    return charge.charge_rate_number_s, forces, applicability


def _coverage(applicable: NDArray[np.bool_], basis: str) -> Record:
    coverage = float(np.mean(applicable))
    return {
        "saved_row_status": "PASS" if bool(applicable.all()) else "NOT_APPLICABLE",
        "saved_row_coverage": coverage,
        "basis": basis,
        "continuous_path_status": "NOT_TESTED",
    }


def _not_gated() -> Record:
    return {
        "saved_row_status": "APPLICABLE_NO_ADDITIONAL_LOCAL_GATE",
        "saved_row_coverage": 1.0,
        "continuous_path_status": "NOT_TESTED",
    }


def _metrics(candidate: FloatArray, reference: FloatArray, values: dict[str, FloatArray]) -> Record:
    if candidate.shape != reference.shape:
        raise ValueError("comparison arrays have different shapes")
    if candidate.ndim == 1:
        residual_norm = np.abs(candidate - reference)
        candidate_norm = np.abs(candidate)
        reference_norm = np.abs(reference)
    elif candidate.ndim == 2 and candidate.shape[1] == 2:
        residual_norm = np.linalg.norm(candidate - reference, axis=1)
        candidate_norm = np.linalg.norm(candidate, axis=1)
        reference_norm = np.linalg.norm(reference, axis=1)
    else:
        raise ValueError("comparison arrays must be scalar or two-vector rows")
    scale = np.maximum(np.maximum(candidate_norm, reference_norm), np.finfo(np.float64).tiny)
    normalized = residual_norm / scale
    worst = int(np.argmax(normalized))
    return {
        "global_relative_l2_residual": float(
            np.linalg.norm(candidate - reference)
            / max(float(np.linalg.norm(reference)), float(np.finfo(np.float64).tiny))
        ),
        "scale_normalized_residual_p99": float(np.percentile(normalized, 99.0)),
        "scale_normalized_residual_max": float(normalized[worst]),
        "absolute_residual_max": float(np.max(residual_norm)),
        "worst_row": {
            "particle_id": int(values["particle_id"][worst]),
            "time_s": float(values["time_s"][worst]),
        },
    }


def _exported_forces(values: dict[str, FloatArray]) -> dict[str, FloatArray]:
    columns = {
        "electric": ("electric_force_r_N", "electric_force_z_N"),
        "relative_flow_ion_drag": ("ion_drag_force_r_N", "ion_drag_force_z_N"),
        "epstein_drag": ("epstein_force_r_N", "epstein_force_z_N"),
        "waldmann_thermophoresis": (
            "thermophoretic_force_r_N",
            "thermophoretic_force_z_N",
        ),
        "free_molecular_lift_sensitivity": ("lift_force_r_N", "lift_force_z_N"),
        "dielectrophoresis": ("dep_force_r_N", "dep_force_z_N"),
        "gravity_buoyancy": (
            "gravity_buoyancy_force_r_N",
            "gravity_buoyancy_force_z_N",
        ),
    }
    return {name: _vectors(values, *names) for name, names in columns.items()}


def _comparison_status(
    model: str,
    metrics: Record,
    limit: float,
    *,
    thermophoretic_authority: str,
    known_constant_convention_models: frozenset[str],
    producer_pass: bool,
) -> str:
    if model == "waldmann_thermophoresis" and thermophoretic_authority != "feature_internal_ppr":
        return "NOT_TESTED_PRIMITIVE_AUTHORITY"
    residual = float(metrics["scale_normalized_residual_max"])
    if residual <= limit:
        return "PASS"
    if producer_pass and model in known_constant_convention_models:
        return "DOCUMENTED_CONSTANT_CONVENTION_DIFFERENCE"
    return "FAIL"


def evaluate(step_directory: Path, config_path: Path) -> Record:
    """Evaluate the frozen pre-event slice without rerunning either solver."""

    config = load_config(config_path)
    step = step_directory.expanduser().resolve()
    if step.name != config.step_directory:
        raise ValueError(f"expected step directory {config.step_directory!r}, found {step.name!r}")
    provenance = _verify_provenance(step, config)
    values = _load_records(step, config)
    epsilon0 = _saved_epsilon0(values)
    producer_charge = _producer_charge(values, config)
    producer_forces = _producer_forces(values, config, epsilon0)
    production_charge, production_forces, applicability = _production_evaluations(values, config)
    exported_charge = values["charge_rate_e_per_s"]
    exported_forces = _exported_forces(values)

    models: dict[str, Record] = {}
    producer_metrics = _metrics(producer_charge, exported_charge, values)
    producer_status = _comparison_status(
        "dynamic_charge",
        producer_metrics,
        config.producer_residual_limit,
        thermophoretic_authority="feature_internal_ppr",
        known_constant_convention_models=frozenset(),
        producer_pass=False,
    )
    production_metrics = _metrics(production_charge, exported_charge, values)
    models["dynamic_charge"] = {
        "producer_formula_replay": {"status": producer_status, **producer_metrics},
        "production_model_comparison": {
            "status": _comparison_status(
                "dynamic_charge",
                production_metrics,
                config.production_residual_limit,
                thermophoretic_authority=config.thermophoretic_primitive_authority,
                known_constant_convention_models=config.known_constant_convention_models,
                producer_pass=producer_status == "PASS",
            ),
            **production_metrics,
        },
        "applicability": applicability["dynamic_charge"],
    }
    for model in MODEL_NAMES[1:]:
        producer_metrics = _metrics(producer_forces[model], exported_forces[model], values)
        producer_status = _comparison_status(
            model,
            producer_metrics,
            config.producer_residual_limit,
            thermophoretic_authority=config.thermophoretic_primitive_authority,
            known_constant_convention_models=frozenset(),
            producer_pass=False,
        )
        production_metrics = _metrics(production_forces[model], exported_forces[model], values)
        models[model] = {
            "producer_formula_replay": {"status": producer_status, **producer_metrics},
            "production_model_comparison": {
                "status": _comparison_status(
                    model,
                    production_metrics,
                    config.production_residual_limit,
                    thermophoretic_authority=config.thermophoretic_primitive_authority,
                    known_constant_convention_models=config.known_constant_convention_models,
                    producer_pass=producer_status == "PASS",
                ),
                **production_metrics,
            },
            "applicability": applicability[model],
        }

    producer_total = sum(
        producer_forces.values(), start=np.zeros_like(next(iter(producer_forces.values())))
    )
    production_total = sum(
        production_forces.values(), start=np.zeros_like(next(iter(production_forces.values())))
    )
    exported_total = sum(
        exported_forces.values(), start=np.zeros_like(next(iter(exported_forces.values())))
    )
    producer_statuses = [str(item["producer_formula_replay"]["status"]) for item in models.values()]
    production_statuses = [
        str(item["production_model_comparison"]["status"]) for item in models.values()
    ]
    return {
        "tool_revision": TOOL_REVISION,
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": "EXTERNAL_VV_FROZEN_STATE_COMPARISON",
        "golden_truth": "NOT_CLAIMED",
        "integrated_trajectory_accuracy": "NOT_TESTED",
        "continuous_path_applicability": "NOT_TESTED",
        "field_interpolation_agreement": "NOT_TESTED",
        "comsol_rerun_performed": False,
        "configuration": {
            "file_name": config.source.name,
            "sha256": _sha256(config.source),
        },
        "resolved_parameters": {
            "electron_thermal_voltage_V": config.electron_thermal_voltage_V,
            "neutral_molecular_mass_kg": config.neutral_molecular_mass_kg,
            "epstein_delta": config.epstein_delta,
            "lift_coefficient": config.lift_coefficient,
            "medium_relative_permittivity": config.medium_relative_permittivity,
            "clausius_mossotti_factor": config.clausius_mossotti_factor,
            "gravity_m_s2": list(config.gravity_m_s2),
            "relative_speed_regularization_m_s": config.relative_speed_regularization_m_s,
            "minimum_ion_energy_V": config.minimum_ion_energy_V,
            "maximum_relative_ion_speed_m_s": config.maximum_relative_ion_speed_m_s,
            "maximum_effective_gas_speed_ratio": config.maximum_effective_gas_speed_ratio,
        },
        "input": {
            "step_directory_name": step.name,
            "records": int(values["particle_id"].size),
            "particles": int(np.unique(values["particle_id"]).size),
            "output_times": int(np.unique(values["time_s"]).size),
            "raw_sha256": config.raw_sha256,
            "provenance": provenance,
        },
        "producer_constant_diagnosis": {
            "inferred_epsilon0_F_m_median": float(np.median(epsilon0)),
            "inferred_epsilon0_relative_spread": float(
                (np.max(epsilon0) - np.min(epsilon0)) / np.median(epsilon0)
            ),
            "inferred_vs_production_epsilon0_relative_difference_median": float(
                np.median((epsilon0 - VACUUM_PERMITTIVITY_F_M) / VACUUM_PERMITTIVITY_F_M)
            ),
        },
        "models": models,
        "combined_force_diagnostic": {
            "producer_formula_candidate": _metrics(producer_total, exported_total, values),
            "production_models": _metrics(production_total, exported_total, values),
            "status": "CHARACTERIZED_NOT_A_GATE_WHILE_THERMOPHORESIS_PRIMITIVE_IS_UNRESOLVED",
        },
        "known_gaps": [
            "COMSOL UsePPR=1 but the saved trajectory exports only unrecovered temperature gradients",
            "DEP point-dipole applicability has no saved-run certificate",
            "saved output frames do not expose intermediate integrator stages or continuous paths",
        ],
        "summary": {
            "producer_formula_replay_pass": producer_statuses.count("PASS"),
            "producer_formula_replay_not_tested": sum(
                status.startswith("NOT_TESTED") for status in producer_statuses
            ),
            "producer_formula_replay_fail": producer_statuses.count("FAIL"),
            "production_model_pass": production_statuses.count("PASS"),
            "production_model_documented_difference": production_statuses.count(
                "DOCUMENTED_CONSTANT_CONVENTION_DIFFERENCE"
            ),
            "production_model_not_tested": sum(
                status.startswith("NOT_TESTED") for status in production_statuses
            ),
            "production_model_fail": production_statuses.count("FAIL"),
            "saved_row_applicability_not_applicable_models": [
                name
                for name, item in models.items()
                if item["applicability"]["saved_row_status"] == "NOT_APPLICABLE"
            ],
        },
        "interpretation": (
            "Producer replay, production-model output, and local saved-row applicability are "
            "separate results. The thermophoretic diagnostic uses an unrecovered Fourier "
            "gradient while the COMSOL feature requested PPR, so it cannot be promoted to "
            "formula parity. Saved output frames do not certify field interpolation, RK stages, "
            "continuous-path applicability, events, or integrated trajectory agreement."
        ),
    }


def write_report(report: Record, output_directory: Path) -> None:
    """Write one no-clobber JSON artifact."""

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "report.json"
    path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_directory", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_directory", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = evaluate(args.step_directory, args.config)
    write_report(report, args.output_directory)
    summary = _mapping(report.get("summary"), "report.summary")
    return 0 if summary["producer_formula_replay_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
