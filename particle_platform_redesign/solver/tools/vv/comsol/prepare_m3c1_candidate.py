"""Prepare and run the M3-C1 Case-A canonical P1 solver candidate.

This external V&V utility translates the audited static COMSOL export into the
canonical case format and executes the ordinary public solver API. It does not
add a COMSOL mode, producer branch, or comparison tolerance to the solver core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from chamber_particles import SimulationError, load_case, open_result, simulate
from chamber_particles.case_format import (
    FieldData,
    P1TriLayout,
    RealizedTableSource,
    read,
    write,
)

TOOL_REVISION: Final = "m3c1_solver_canonical_p1_projection_candidate_v3"
_STEP_ROWS: Final = (
    ("dt_0p625us", 6.25e-7),
    ("dt_0p3125us", 3.125e-7),
    ("dt_0p15625us", 1.5625e-7),
)
_EXPECTED_PARTICLES: Final = 287
_EXPECTED_FRAMES: Final = 46
_EXPECTED_END_S: Final = 4.5e-4
_EXPECTED_OUTPUT_INTERVAL_S: Final = 1.0e-5
_DIAGNOSTIC_ION_SPEED_LIMIT_M_S: Final = 30_000.0
_LIFECYCLE: Final = ("pending", "active", "stuck", "escaped", "failed")

_SCALAR_FIELDS: Final = (
    ("gas_density", "gas_density_kg_per_m3", "kg/m^3"),
    ("gas_dynamic_viscosity", "dynamic_viscosity_Pa_s", "Pa*s"),
    ("gas_temperature", "gas_temperature_K", "K"),
    ("gas_mean_free_path", "gas_mean_free_path_m", "m"),
    ("electron_number_density", "electron_density_per_m3", "1/m^3"),
    ("positive_ion_number_density", "total_positive_ion_density_per_m3", "1/m^3"),
    ("electron_thermal_voltage", "electron_temperature_eV_as_V", "V"),
    ("positive_ion_thermal_voltage", "ion_thermal_energy_eV_as_V", "V"),
    ("effective_positive_ion_mass", "effective_positive_ion_mass_kg", "kg"),
    ("screening_length", "bounded_screening_length_m", "m"),
    ("ion_neutral_mean_free_path", "ion_neutral_mean_free_path_m", "m"),
    ("azimuthal_gas_vorticity", "axisymmetric_azimuthal_vorticity_per_s", "1/s"),
)
_VECTOR_FIELDS: Final = (
    (
        "gas_velocity",
        ("gas_velocity_r_m_per_s", "gas_velocity_z_m_per_s"),
        "m/s",
    ),
    (
        "electric_field",
        ("electric_field_r_V_per_m", "electric_field_z_V_per_m"),
        "V/m",
    ),
    (
        "positive_ion_velocity",
        ("ion_velocity_r_m_per_s", "ion_velocity_z_m_per_s"),
        "m/s",
    ),
    (
        "gradient_mean_e_squared",
        ("gradient_E2_r_V2_per_m3", "gradient_E2_z_V2_per_m3"),
        "V^2/m^3",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be a mapping")
    required = {
        "schema_version",
        "evaluation_id",
        "evaluation_revision",
        "classification",
        "case",
        "parameters",
        "physics",
        "scope",
    }
    if set(payload) != required:
        raise ValueError(f"{path}: expected keys {sorted(required)}")
    if payload["schema_version"] != 1 or payload["evaluation_revision"] != 1:
        raise ValueError(f"{path}: unsupported M3-C1 configuration revision")
    _validate_case_selection(path, payload.get("case"))
    _validate_physics_selection(path, payload.get("physics"))
    _validate_parameters(path, payload.get("parameters"))
    _validate_scope(path, payload.get("scope"))
    return payload


def _validate_case_selection(path: Path, case: object) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"{path}: case must be a mapping")
    expected_case = {
        "workflow": "caseA",
        "diameter_m": 1.0e-7,
        "particle_count": _EXPECTED_PARTICLES,
        "output_times": _EXPECTED_FRAMES,
        "time_end_s": _EXPECTED_END_S,
        "output_interval_s": _EXPECTED_OUTPUT_INTERVAL_S,
        "fixed_rk4_steps_s": [step for _, step in _STEP_ROWS],
    }
    if case != expected_case:
        raise ValueError(f"{path}: case must select the audited M3-C1 pilot matrix")


def _validate_physics_selection(path: Path, physics: object) -> None:
    if not isinstance(physics, dict):
        raise ValueError(f"{path}: physics must be a mapping")
    expected_physics = {
        "brownian_active": False,
        "saffman_active": False,
        "charge_revision": "aggregate_relative_drift_regularized_two_current_v1",
        "drag_revision": "epstein_linear_effective_gas_sensitivity_v1",
        "electric_revision": "electric_coulomb_v1",
        "ion_drag_revision": "relative_flow_screened_collection_orbital_aggregate_ion_v1",
        "thermophoresis_revision": (
            "waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1"
        ),
        "dielectrophoresis_revision": "quasistatic_spherical_gradient_e2_v1",
        "lift_revision": "rarefied_vorticity_sensitivity_rz_v1",
        "gravity_buoyancy_revision": "gravity_buoyancy_standard_v1",
    }
    if physics != expected_physics:
        raise ValueError(f"{path}: physics selection does not match M3-C1 revision 1")


def _validate_parameters(path: Path, parameters: object) -> None:
    if not isinstance(parameters, dict):
        raise ValueError(f"{path}: parameters must be a mapping")
    expected_parameter_keys = {
        "gas_molecular_mass_kg",
        "epstein_delta",
        "maximum_neutral_speed_ratio",
        "maximum_relative_ion_speed_m_s",
        "medium_relative_permittivity",
        "real_clausius_mossotti_factor",
        "maximum_point_dipole_radius_m",
        "lift_coefficient",
        "gravity_m_s2",
    }
    if set(parameters) != expected_parameter_keys:
        raise ValueError(f"{path}: parameter keys do not match M3-C1 revision 1")
    finite_scalars = expected_parameter_keys - {"gravity_m_s2"}
    if any(
        not math.isfinite(float(parameters[name])) or float(parameters[name]) <= 0.0
        for name in finite_scalars
    ):
        raise ValueError(f"{path}: M3-C1 scalar parameters must be finite and positive")
    if parameters["gravity_m_s2"] != [0.0, -9.80665]:
        raise ValueError(f"{path}: gravity_m_s2 must preserve the reference convention")


def _validate_scope(path: Path, scope: object) -> None:
    expected = {
        "reference": "M3-C0b v6 Case A 100 nm Brownian-off pre-event export",
        "field_mode": "canonical_exact_connectivity_P1_projection_from_exported_nodal_values",
        "trajectory_window": "0_to_450_us_before_first_observed_material_event",
        "claims_solver_agreement": False,
        "claims_boundary_parity": False,
        "claims_physical_applicability": False,
        "maximum_point_dipole_radius_m_authority": (
            "benchmark_sensitivity_assumption_not_physical_certification"
        ),
        "heat_flux_authority": "derived_export_not_ppr_nojac_primitive_authority",
        "diagnostic_continuous_relative_ion_speed_limit_m_s": (_DIAGNOSTIC_ION_SPEED_LIMIT_M_S),
        "diagnostic_ion_speed_limit_authority": (
            "component_box_execution_sentinel_not_physical_certification"
        ),
    }
    if scope != expected:
        raise ValueError(f"{path}: scope and diagnostic authorities do not match revision 1")


def _dep_point_dipole_receipt(
    benchmark_limit_m: float,
    maximum_radius_m: float,
) -> dict[str, object]:
    execution_guard_upper_m = math.nextafter(benchmark_limit_m, math.inf)
    return {
        "status": "NOT_TESTED_POINT_DIPOLE_CERTIFICATION_MISSING",
        "benchmark_sensitivity_limit_m": benchmark_limit_m,
        "limit_authority": "benchmark_sensitivity_assumption_not_physical_certification",
        "maximum_electrostatic_radius_m": maximum_radius_m,
        "excess_m": max(0.0, maximum_radius_m - benchmark_limit_m),
        "execution_guard_rounding_policy": (
            "accept_immediate_float64_successor_for_independent_serialization"
        ),
        "execution_guard_upper_m": execution_guard_upper_m,
        "execution_guard_excess_m": max(0.0, maximum_radius_m - execution_guard_upper_m),
        "execution_guard_status": (
            "within_limit" if maximum_radius_m <= execution_guard_upper_m else "BLOCKED"
        ),
    }


def _dictionary(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("%"))
        rows = list(reader)
    if not rows or reader.fieldnames != ["column", "COMSOL_expression", "unit"]:
        raise ValueError(f"invalid COMSOL column dictionary: {path}")
    names = [row["column"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate COMSOL columns: {path}")
    return names, {row["column"]: row["unit"] for row in rows}


def _numeric_table(path: Path, column_count: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(line for line in stream if not line.startswith("%"))
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != column_count:
                raise ValueError(f"{path}: row {line_number} has {len(row)} columns")
            rows.append([float(token) for token in row])
    if not rows:
        raise ValueError(f"empty COMSOL numeric table: {path}")
    return np.asarray(rows, dtype=np.float64)


def _coordinate_match(
    canonical: np.ndarray,
    provider: np.ndarray,
    tolerance_m: float,
) -> tuple[np.ndarray, float]:
    if np.unique(provider, axis=0).shape[0] != provider.shape[0]:
        raise ValueError("provider field coordinates are not unique")
    matched = np.empty(canonical.shape[0], dtype=np.int64)
    distance2 = np.empty(canonical.shape[0], dtype=np.float64)
    tolerance2 = tolerance_m * tolerance_m
    for start in range(0, canonical.shape[0], 256):
        stop = min(start + 256, canonical.shape[0])
        delta = canonical[start:stop, None, :] - provider[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        within = np.count_nonzero(squared <= tolerance2, axis=1)
        if bool((within != 1).any()):
            raise ValueError("canonical field nodes do not have a unique provider match")
        local = np.argmin(squared, axis=1)
        matched[start:stop] = local
        distance2[start:stop] = squared[np.arange(stop - start), local]
    if np.unique(matched).size != matched.size:
        raise ValueError("provider-to-canonical field match is not bijective")
    return matched, float(np.sqrt(np.max(distance2, initial=0.0)))


def _exported_p1_fields(
    package: Path,
    nodes_m: np.ndarray,
    layout_name: str,
) -> tuple[tuple[FieldData, ...], dict[str, object]]:
    dictionary_path = package / "config" / "background_field_column_dictionary.csv"
    values_path = package / "input_fields" / "background_fields_mesh_points.csv"
    names, units = _dictionary(dictionary_path)
    index = {name: position for position, name in enumerate(names)}
    expected_units = {
        "r_m": "m",
        "z_m": "m",
        **{column: unit for _, column, unit in _SCALAR_FIELDS},
        **{column: unit for _, columns, unit in _VECTOR_FIELDS for column in columns},
        "thermal_conductivity_W_per_mK": "W/(m*K)",
        "temperature_gradient_r_K_per_m": "K/m",
        "temperature_gradient_z_K_per_m": "K/m",
    }
    if any(units.get(name) != unit for name, unit in expected_units.items()):
        raise ValueError("exported P1 field columns or units do not match the audited mapping")
    values = _numeric_table(values_path, len(names))
    required_columns = list(expected_units)
    finite = np.isfinite(values[:, [index[name] for name in required_columns]]).all(axis=1)
    provider_coordinates = values[finite][:, [index["r_m"], index["z_m"]]]
    provider_values = values[finite]
    matched, maximum_distance = _coordinate_match(nodes_m, provider_coordinates, 1.0e-14)
    ordered = provider_values[matched]

    fields: list[FieldData] = []
    for field_name, column, unit in _SCALAR_FIELDS:
        fields.append(
            FieldData(
                field_name,
                layout_name,
                "node",
                ("value",),
                "scalar",
                np.ascontiguousarray(ordered[:, [index[column]]], dtype="<f8"),
                unit,
            )
        )

    axis = nodes_m[:, 0] == 0.0
    axis_projection: dict[str, object] = {}
    for field_name, columns, unit in _VECTOR_FIELDS:
        vector = np.ascontiguousarray(
            ordered[:, [index[columns[0]], index[columns[1]]]], dtype="<f8"
        )
        original = vector[axis, 0].copy()
        vector[axis, 0] = 0.0
        axis_projection[field_name] = {
            "corrected_node_count": int(np.count_nonzero(original)),
            "maximum_correction": float(np.max(np.abs(original), initial=0.0)),
            "reason": "canonical axisymmetric vector regularity",
        }
        fields.append(
            FieldData(
                field_name,
                layout_name,
                "node",
                ("r", "z"),
                "axisymmetric_rz",
                vector,
                unit,
            )
        )

    conductivity = ordered[:, index["thermal_conductivity_W_per_mK"]]
    gradient = ordered[
        :,
        [index["temperature_gradient_r_K_per_m"], index["temperature_gradient_z_K_per_m"]],
    ]
    heat_flux = np.ascontiguousarray(-conductivity[:, None] * gradient, dtype="<f8")
    original_heat_flux_axis = heat_flux[axis, 0].copy()
    heat_flux[axis, 0] = 0.0
    axis_projection["gas_translational_heat_flux"] = {
        "corrected_node_count": int(np.count_nonzero(original_heat_flux_axis)),
        "maximum_correction": float(np.max(np.abs(original_heat_flux_axis), initial=0.0)),
        "reason": "canonical axisymmetric vector regularity",
    }
    fields.append(
        FieldData(
            "gas_translational_heat_flux",
            layout_name,
            "node",
            ("r", "z"),
            "axisymmetric_rz",
            heat_flux,
            "W/m^2",
        )
    )
    fields.sort(key=lambda field: field.name)
    evidence: dict[str, object] = {
        "maximum_coordinate_match_distance_m": maximum_distance,
        "matched_node_count": int(nodes_m.shape[0]),
        "axis_radial_projection": axis_projection,
        "heat_flux_formation": {
            "formula": "q_effective=-thermal_conductivity*grad(gas_temperature)",
            "conductivity_column": "thermal_conductivity_W_per_mK",
            "gradient_columns": [
                "temperature_gradient_r_K_per_m",
                "temperature_gradient_z_K_per_m",
            ],
            "classification": "external producer transformation",
            "authority_status": "NOT_TESTED_PRIMITIVE_AUTHORITY",
            "limitation": "COMSOL PPR/nojac pointwise primitive was not recovered",
        },
        "azimuthal_vorticity_policy": (
            "preserve producer scalar; no core gradient recovery or axis repair"
        ),
        "column_dictionary_sha256": _sha256(dictionary_path),
        "field_values_sha256": _sha256(values_path),
        "canonical_field_mapping": {
            **{name: column for name, column, _ in _SCALAR_FIELDS},
            **{name: list(columns) for name, columns, _ in _VECTOR_FIELDS},
            "gas_translational_heat_flux": "derived_from_exported_k_and_temperature_gradient",
        },
    }
    return tuple(fields), evidence


def _release_source(package: Path) -> tuple[RealizedTableSource, dict[str, object]]:
    path = package / "results" / "release_state_t0_tidy.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("%")))
    if len(rows) != _EXPECTED_PARTICLES:
        raise ValueError(f"expected {_EXPECTED_PARTICLES} release rows, got {len(rows)}")

    def vector(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype="<f8")

    particle_id = np.asarray([int(row["particle_id"]) for row in rows], dtype="<i8")
    if not np.array_equal(particle_id, np.arange(1, _EXPECTED_PARTICLES + 1)):
        raise ValueError("release particle IDs must be the audited contiguous 1..287 set")
    diameter = vector("particle_diameter_m")
    radius = vector("particle_radius_m")
    mass = vector("particle_mass_kg")
    if not np.allclose(
        diameter,
        np.full(_EXPECTED_PARTICLES, 1.0e-7),
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise ValueError("release table is not the audited Case-A 100 nm source")
    source = RealizedTableSource(
        name="particles",
        particle_id=particle_id,
        release_time_s=np.zeros(len(rows), dtype="<f8"),
        position_m=np.column_stack((vector("r_m"), vector("z_m"))).astype("<f8"),
        velocity_m_s=np.column_stack(
            (vector("velocity_r_m_per_s"), vector("velocity_z_m_per_s"))
        ).astype("<f8"),
        charge_number=np.full(len(rows), -1.0, dtype="<f8"),
        mass_kg=mass,
        drag_diameter_m=diameter,
        electrostatic_radius_m=radius,
        displaced_volume_m3=(math.pi * diameter**3 / 6.0).astype("<f8"),
        model_weight=np.ones(len(rows), dtype="<f8"),
        material_id=np.zeros(len(rows), dtype="<i4"),
    )
    evidence: dict[str, object] = {
        "release_table_sha256": _sha256(path),
        "particle_count": len(rows),
        "particle_id_min": int(particle_id.min()),
        "particle_id_max": int(particle_id.max()),
        "initial_charge_number_e": -1.0,
        "maximum_electrostatic_radius_m": float(np.max(radius)),
    }
    return source, evidence


def _reference_configuration(reference: Path) -> Path:
    configuration_paths = sorted(reference.glob("m3c0b_caseA_100nm_pilot_v6.json"))
    if len(configuration_paths) != 1:
        raise ValueError("reference must contain the staged M3-C0b v6 configuration")
    configuration_path = configuration_paths[0]
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    reference_case = configuration.get("case")
    if not isinstance(reference_case, dict):
        raise ValueError("reference M3-C0b case receipt is missing")
    expected = {
        "particle_count": _EXPECTED_PARTICLES,
        "output_times": _EXPECTED_FRAMES,
        "time_end_s": _EXPECTED_END_S,
        "fixed_rk4_steps_s": [step for _, step in _STEP_ROWS],
    }
    if any(reference_case.get(name) != value for name, value in expected.items()):
        raise ValueError("reference M3-C0b v6 matrix does not match the candidate")
    return configuration_path


def _reference_trajectory_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != _EXPECTED_PARTICLES * _EXPECTED_FRAMES:
        raise ValueError(f"{path}: unexpected trajectory row count")
    particle_ids = {int(row["particle_id"]) for row in rows}
    times = np.unique(np.asarray([float(row["time_s"]) for row in rows]))
    expected_times = np.asarray(
        [index * _EXPECTED_OUTPUT_INTERVAL_S for index in range(_EXPECTED_FRAMES)]
    )
    if particle_ids != set(range(1, _EXPECTED_PARTICLES + 1)):
        raise ValueError(f"{path}: unexpected particle ID set")
    if not np.array_equal(times, expected_times):
        raise ValueError(f"{path}: unexpected output schedule")
    if any(row["lifecycle"] != "active" for row in rows):
        raise ValueError("reference M3-C0b v6 is not an all-active pre-event baseline")
    return rows, _sha256(path)


def _reference_initial_state(
    rows: list[dict[str, str]],
) -> dict[int, tuple[float, float, float, float, float]]:
    return {
        int(row["particle_id"]): (
            float(row["r_m"]),
            float(row["z_m"]),
            float(row["velocity_r_m_per_s"]),
            float(row["velocity_z_m_per_s"]),
            float(row["charge_number_e"]),
        )
        for row in rows
        if float(row["time_s"]) == 0.0
    }


def _reference_evidence(reference: Path, source: RealizedTableSource) -> dict[str, object]:
    configuration_path = _reference_configuration(reference)

    trajectory_hashes: dict[str, str] = {}
    initial_by_id: dict[int, tuple[float, float, float, float, float]] = {}
    for step_index, (label, _) in enumerate(_STEP_ROWS):
        path = reference / label / "trajectory_reference.csv"
        rows, trajectory_hashes[label] = _reference_trajectory_rows(path)
        if step_index == 0:
            initial_by_id = _reference_initial_state(rows)
    for index, particle_id in enumerate(source.particle_id):
        reference_initial = initial_by_id[int(particle_id)]
        candidate_initial = (
            source.position_m[index, 0],
            source.position_m[index, 1],
            source.velocity_m_s[index, 0],
            source.velocity_m_s[index, 1],
            source.charge_number[index],
        )
        if not np.allclose(reference_initial, candidate_initial, rtol=0.0, atol=1.0e-15):
            raise ValueError("release source and M3-C0b v6 initial state do not match")
    return {
        "reference_root": str(reference.resolve()),
        "configuration_sha256": _sha256(configuration_path),
        "trajectory_sha256": trajectory_hashes,
        "particle_count": _EXPECTED_PARTICLES,
        "output_times": _EXPECTED_FRAMES,
        "all_reference_records_active": True,
    }


def _case_document(
    data_name: str,
    content_hash: str,
    dt_s: float,
    config: dict[str, Any],
    *,
    dep_radius_limit_m: float | None = None,
    ion_speed_limit_m_s: float | None = None,
) -> dict[str, object]:
    parameters = config["parameters"]
    physics = config["physics"]
    selected_dep_radius_limit_m = (
        parameters["maximum_point_dipole_radius_m"]
        if dep_radius_limit_m is None
        else dep_radius_limit_m
    )
    selected_ion_speed_limit_m_s = (
        parameters["maximum_relative_ion_speed_m_s"]
        if ion_speed_limit_m_s is None
        else ion_speed_limit_m_s
    )
    output_times = [
        index * _EXPECTED_OUTPUT_INTERVAL_S for index in range(_EXPECTED_FRAMES - 1)
    ] + [_EXPECTED_END_S]
    return {
        "format_version": 2,
        "case": {
            "name": f"m3c1_caseA_100nm_canonical_p1_deterministic_dt_{dt_s:.9g}",
            "data_path": data_name,
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": "axisymmetric_rz_meridional"},
        "time": {"start_s": 0.0, "end_s": _EXPECTED_END_S, "dt_s": dt_s},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 20261001,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 512},
        "physics": {
            "charge": {
                "model": "plasma_continuous",
                "revision": physics["charge_revision"],
                "electron_number_density_field": "electron_number_density",
                "positive_ion_number_density_field": "positive_ion_number_density",
                "electron_thermal_voltage_field": "electron_thermal_voltage",
                "positive_ion_thermal_voltage_field": "positive_ion_thermal_voltage",
                "positive_ion_velocity_field": "positive_ion_velocity",
                "effective_positive_ion_mass_field": "effective_positive_ion_mass",
                "screening_length_field": "screening_length",
                "maximum_relative_ion_speed_m_s": selected_ion_speed_limit_m_s,
                "applicability": "error",
            },
            "drag": {
                "model": "epstein_linear",
                "revision": physics["drag_revision"],
                "gas_velocity_field": "gas_velocity",
                "gas_density_field": "gas_density",
                "gas_temperature_field": "gas_temperature",
                "gas_mean_free_path_field": "gas_mean_free_path",
                "gas_molecular_mass_kg": parameters["gas_molecular_mass_kg"],
                "delta": parameters["epstein_delta"],
                "maximum_speed_ratio": parameters["maximum_neutral_speed_ratio"],
                "applicability": "error",
            },
            "electric": {
                "model": "coulomb",
                "revision": physics["electric_revision"],
                "electric_field": "electric_field",
            },
            "ion_drag": {
                "model": "screened_collection_orbital",
                "revision": physics["ion_drag_revision"],
                "positive_ion_number_density_field": "positive_ion_number_density",
                "positive_ion_thermal_voltage_field": "positive_ion_thermal_voltage",
                "positive_ion_velocity_field": "positive_ion_velocity",
                "effective_positive_ion_mass_field": "effective_positive_ion_mass",
                "screening_length_field": "screening_length",
                "ion_neutral_mean_free_path_field": "ion_neutral_mean_free_path",
                "maximum_relative_ion_speed_m_s": selected_ion_speed_limit_m_s,
                "applicability": "error",
            },
            "thermophoresis": {
                "model": "waldmann_gallis",
                "revision": physics["thermophoresis_revision"],
                "gas_velocity_field": "gas_velocity",
                "gas_temperature_field": "gas_temperature",
                "gas_translational_heat_flux_field": "gas_translational_heat_flux",
                "gas_mean_free_path_field": "gas_mean_free_path",
                "gas_molecular_mass_kg": parameters["gas_molecular_mass_kg"],
                "maximum_speed_ratio": parameters["maximum_neutral_speed_ratio"],
                "applicability": "error",
            },
            "dielectrophoresis": {
                "model": "quasistatic_spherical",
                "revision": physics["dielectrophoresis_revision"],
                "gradient_mean_e_squared_field": "gradient_mean_e_squared",
                "medium_relative_permittivity": parameters["medium_relative_permittivity"],
                "real_clausius_mossotti_factor": parameters["real_clausius_mossotti_factor"],
                "maximum_point_dipole_radius_m": selected_dep_radius_limit_m,
            },
            "lift": {
                "model": "rarefied_vorticity_sensitivity",
                "revision": physics["lift_revision"],
                "gas_velocity_field": "gas_velocity",
                "gas_density_field": "gas_density",
                "gas_mean_free_path_field": "gas_mean_free_path",
                "azimuthal_gas_vorticity_field": "azimuthal_gas_vorticity",
                "lift_coefficient": parameters["lift_coefficient"],
                "applicability": "error",
            },
            "gravity_buoyancy": {
                "model": "standard",
                "revision": physics["gravity_buoyancy_revision"],
                "gas_density_field": "gas_density",
                "gravity_m_s2": parameters["gravity_m_s2"],
            },
        },
        "sources": [{"name": "release", "type": "table", "table": "particles"}],
        "boundaries": [
            {"boundary_group": name, "priority": 10, "law": law}
            for name, law in (
                ("wafer", "stick"),
                ("grounded_wall", "stick"),
                ("focus_transition", "stick"),
                ("outer_dielectric", "stick"),
                ("pump_outlet", "escape"),
                ("gas_inlet", "escape"),
            )
        ],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": output_times},
            },
            "probes": None,
        },
    }


def prepare(
    config_path: Path,
    package: Path,
    base_data: Path,
    reference: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config = _load_config(config_path)
    base = read(base_data)
    if base.coordinate_system != "axisymmetric_rz" or len(base.layouts) != 1:
        raise ValueError("base canonical data must contain one axisymmetric RZ layout")
    layout = base.layouts[0]
    if not isinstance(layout, P1TriLayout):
        raise ValueError("base canonical data must use the exact triangular P1 layout")
    projected_fields, field_evidence = _exported_p1_fields(package, layout.nodes_m, layout.name)
    source, source_evidence = _release_source(package)
    reference_evidence = _reference_evidence(reference, source)
    dep_radius_limit_m = float(config["parameters"]["maximum_point_dipole_radius_m"])
    maximum_radius_m = float(np.max(source.electrostatic_radius_m))
    field_by_name = {field.name: field for field in projected_fields}
    ion_velocity = field_by_name["positive_ion_velocity"].values
    ion_velocity_abs_upper = np.max(np.abs(ion_velocity), axis=0)
    ion_velocity_box_speed_upper = float(np.linalg.norm(ion_velocity_abs_upper))
    ion_velocity_nodal_speed_max = float(np.max(np.linalg.norm(ion_velocity, axis=1)))
    ion_speed_limit_m_s = float(config["parameters"]["maximum_relative_ion_speed_m_s"])
    provenance = json.loads(base.provenance_json)
    provenance["m3c1_solver_canonical_p1_projection_candidate"] = {
        "tool_revision": TOOL_REVISION,
        "scope": "caseA_100nm_exported_nodal_canonical_p1_all_deterministic_pre_event",
        "configuration_sha256": _sha256(config_path),
        "base_data_sha256": _sha256(base_data),
        **field_evidence,
        **source_evidence,
        "reference": reference_evidence,
    }
    output.mkdir(parents=True)
    data_path = output / "candidate_input.h5"
    data = replace(
        base,
        provenance_json=json.dumps(
            provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        fields=projected_fields,
        sources=(source,),
    )
    info = write(data_path, data)
    cases: dict[str, str] = {}
    diagnostic_cases: dict[str, str] = {}
    for label, dt_s in _STEP_ROWS:
        case_path = output / f"candidate_{label}.yaml"
        case_path.write_text(
            yaml.safe_dump(
                _case_document(data_path.name, info.content_hash, dt_s, config),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        load_case(case_path)
        cases[label] = case_path.name
        diagnostic_case_path = output / f"candidate_diagnostic_{label}.yaml"
        diagnostic_case_path.write_text(
            yaml.safe_dump(
                _case_document(
                    data_path.name,
                    info.content_hash,
                    dt_s,
                    config,
                    dep_radius_limit_m=maximum_radius_m,
                    ion_speed_limit_m_s=_DIAGNOSTIC_ION_SPEED_LIMIT_M_S,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        load_case(diagnostic_case_path)
        diagnostic_cases[label] = diagnostic_case_path.name
    report: dict[str, object] = {
        "status": "prepared",
        "tool_revision": TOOL_REVISION,
        "scope": "caseA_100nm_exported_nodal_canonical_p1_all_deterministic_pre_event",
        "configuration": str(config_path.resolve()),
        "configuration_sha256": _sha256(config_path),
        "base_data": str(base_data.resolve()),
        "base_data_sha256": _sha256(base_data),
        "input_data": data_path.name,
        "input_content_hash": info.content_hash,
        "input_file_sha256": _sha256(data_path),
        "cases": cases,
        "diagnostic_cases": diagnostic_cases,
        "field_evidence": field_evidence,
        "source_evidence": source_evidence,
        "reference_evidence": reference_evidence,
        "settings": {
            "coordinate_system": "axisymmetric_rz_no_swirl",
            "integrator": "classical_rk4_fixed",
            "dt_s": [step for _, step in _STEP_ROWS],
            "end_time_s": _EXPECTED_END_S,
            "output_interval_s": _EXPECTED_OUTPUT_INTERVAL_S,
            "charge": config["physics"]["charge_revision"],
            "brownian": "disabled",
            "saffman": "disabled",
            "deterministic_contributions": [
                "electric",
                "relative_flow_ion_drag",
                "epstein_drag",
                "waldmann_thermophoresis",
                "free_molecular_lift_sensitivity",
                "dielectrophoresis",
                "gravity_buoyancy",
            ],
            "parameters": config["parameters"],
            "diagnostic_execution_sentinel": {
                "classification": (
                    "EXECUTION_SENTINEL_FOR_BENCHMARK_SENSITIVITY_NOT_POINT_DIPOLE_CERTIFICATION"
                ),
                "maximum_point_dipole_radius_m": maximum_radius_m,
                "maximum_relative_ion_speed_m_s": _DIAGNOSTIC_ION_SPEED_LIMIT_M_S,
                "authority": "serialized_source_radius_for_diagnostic_execution_only",
                "ion_speed_limit_authority": (
                    "diagnostic_component_box_sentinel_not_physical_certification"
                ),
            },
        },
        "applicability": {
            "policy": "fail_closed_execution_without_physical_certification_claim",
            "overall_physical_applicability": "NOT_CERTIFIED",
            "execution_guard_parameters": {
                "neutral_speed_ratio_limit": config["parameters"]["maximum_neutral_speed_ratio"],
                "relative_ion_speed_limit_m_s": config["parameters"][
                    "maximum_relative_ion_speed_m_s"
                ],
                "free_molecular_knudsen_minimum": 10.0,
            },
            "continuous_relative_ion_speed_box_guard": {
                "positive_ion_velocity_component_abs_upper_m_s": (ion_velocity_abs_upper.tolist()),
                "exported_nodal_vector_speed_max_m_s": ion_velocity_nodal_speed_max,
                "zero_particle_speed_box_upper_m_s": ion_velocity_box_speed_upper,
                "limit_m_s": ion_speed_limit_m_s,
                "status": (
                    "within_limit"
                    if ion_velocity_box_speed_upper <= ion_speed_limit_m_s
                    else "BLOCKED"
                ),
                "interpretation": (
                    "global componentwise continuous-path certificate, not sampled nodal speed"
                ),
            },
            "dep_point_dipole": _dep_point_dipole_receipt(
                dep_radius_limit_m,
                maximum_radius_m,
            ),
            "heat_flux_primitive_authority": {
                "status": "NOT_TESTED_PRIMITIVE_AUTHORITY",
                "reason": "PPR/nojac pointwise heat-flux primitive was not recovered",
            },
            "interpretation": (
                "guard values only; none is evidence of physical model-domain certification"
            ),
        },
        "claim_policy": {
            "candidate_execution_is_not_solver_agreement": True,
            "pre_event_only": True,
            "boundary_parity": "not_tested",
            "candidate_field_interpolation_authority": (
                "canonical_exact_connectivity_P1_from_exported_nodal_values"
            ),
            "comsol_ppr_nojac_pointwise_parity": "NOT_TESTED",
            "same_field_full_physics_agreement": "NOT_TESTED",
            "common_p1_rerun_required_only_if_cross_representation_diagnostics_are_ambiguous": True,
        },
    }
    (output / "prepare_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _write_trajectory(path: Path, result: object) -> int:
    row_count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "time_s",
                "r_m",
                "z_m",
                "velocity_r_m_per_s",
                "velocity_z_m_per_s",
                "charge_number_e",
                "lifecycle",
            )
        )
        for frame in result.iter_frames():  # type: ignore[attr-defined]
            for index, particle_id in enumerate(frame.particle_id):
                code = int(frame.lifecycle[index])
                writer.writerow(
                    (
                        int(particle_id),
                        frame.time_s,
                        frame.position_m[index, 0],
                        frame.position_m[index, 1],
                        frame.velocity_m_s[index, 0],
                        frame.velocity_m_s[index, 1],
                        frame.charge_number[index],
                        _LIFECYCLE[code],
                    )
                )
                row_count += 1
    return row_count


def _write_events(path: Path, result: object) -> int:
    events = result.read_boundary_events()  # type: ignore[attr-defined]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "event_ordinal",
                "event_time_s",
                "hit_r_m",
                "hit_z_m",
                "normal_r",
                "normal_z",
                "pre_velocity_r_m_per_s",
                "pre_velocity_z_m_per_s",
                "post_velocity_r_m_per_s",
                "post_velocity_z_m_per_s",
                "event_type",
                "boundary_semantic",
                "outcome",
            )
        )
        for index, particle_id in enumerate(events.particle_id):
            writer.writerow(
                (
                    int(particle_id),
                    int(events.event_ordinal[index]),
                    events.time_s[index],
                    events.position_m[index, 0],
                    events.position_m[index, 1],
                    events.normal[index, 0],
                    events.normal[index, 1],
                    events.velocity_pre_m_s[index, 0],
                    events.velocity_pre_m_s[index, 1],
                    events.velocity_post_m_s[index, 0],
                    events.velocity_post_m_s[index, 1],
                    "material_boundary",
                    int(events.boundary_id[index]),
                    events.outcome[index],
                )
            )
    return int(events.particle_id.size)


def _physics_models(manifest: Mapping[str, object]) -> object:
    resolved = manifest.get("resolved")
    if not isinstance(resolved, Mapping) or "physics_models" not in resolved:
        raise ValueError("result manifest is missing resolved physics models")
    return resolved["physics_models"]


def _manifest_revisions(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        name: value
        for name, value in manifest.items()
        if name.endswith("_revision") and value is not None
    }


def _execute_case(
    prepared: Path,
    label: str,
    case_name: str,
) -> dict[str, object]:
    case_path = prepared / case_name
    result_path = prepared / f"result_{label}"
    case = load_case(case_path)
    reused_result = result_path.exists()
    if not reused_result:
        try:
            simulate(case, result_path)
        except SimulationError as error:
            return {
                "case": case_path.name,
                "status": "blocked",
                "decision": "BLOCKED",
                "blocker_class": "candidate_execution_applicability_or_setup",
                "message": str(error),
            }
    result = open_result(result_path)
    if result.manifest.get("case_file_hash") != case.case_file_hash:
        raise ValueError("existing result does not belong to the selected case")
    manifest_path = result_path / "run.json"
    trajectory_path = prepared / f"candidate_trajectory_{label}.csv"
    events_path = prepared / f"candidate_events_{label}.csv"
    trajectory_rows = _write_trajectory(trajectory_path, result)
    event_rows = _write_events(events_path, result)
    failures = result.read_failure_events()
    failure_event_count = int(failures.particle_id.size)
    expected_trajectory_rows = _EXPECTED_PARTICLES * _EXPECTED_FRAMES
    manifest_status = result.manifest.get("status")
    if manifest_status != "complete" or failure_event_count:
        status, decision = "blocked", "BLOCKED"
        blocker_class = "continuous_or_stage_model_applicability"
    elif event_rows:
        status, decision = "incomplete", "INCOMPLETE"
        blocker_class = "material_event_inside_pre_event_window"
    elif trajectory_rows != expected_trajectory_rows:
        status, decision = "incomplete", "INCOMPLETE"
        blocker_class = "pre_event_matrix_not_all_active"
    else:
        status, decision = "complete", "COMPLETE"
        blocker_class = None
    return {
        "case": case_path.name,
        "case_sha256": _sha256(case_path),
        "result": result_path.name,
        "reused_existing_result": reused_result,
        "status": status,
        "decision": decision,
        "blocker_class": blocker_class,
        "result_manifest_sha256": _sha256(manifest_path),
        "result_manifest_status": manifest_status,
        "engine_algorithm_revision": result.manifest["engine_algorithm_revision"],
        "revisions": _manifest_revisions(result.manifest),
        "physics_models": _physics_models(result.manifest),
        "trajectory": trajectory_path.name,
        "trajectory_sha256": _sha256(trajectory_path),
        "trajectory_rows": trajectory_rows,
        "events": events_path.name,
        "events_sha256": _sha256(events_path),
        "event_rows": event_rows,
        "failure_event_count": failure_event_count,
        "failure_time_min_s": (None if not failure_event_count else float(np.min(failures.time_s))),
        "failure_time_max_s": (None if not failure_event_count else float(np.max(failures.time_s))),
        "failure_reason_counts": result.manifest.get("failure_reason_counts"),
    }


def _case_matrix(report: dict[str, Any], key: str) -> dict[str, object]:
    matrix = report.get(key)
    expected = {label for label, _ in _STEP_ROWS}
    if not isinstance(matrix, dict) or set(matrix) != expected:
        raise ValueError(f"prepared {key} matrix is incomplete")
    return matrix


def _run_outcome(runs: Mapping[str, Mapping[str, object]]) -> tuple[str, str]:
    if any(run_info.get("status") == "blocked" for run_info in runs.values()):
        return "blocked", "BLOCKED"
    if len(runs) == len(_STEP_ROWS) and all(
        run_info.get("status") == "complete" for run_info in runs.values()
    ):
        return "complete", "COMPLETE"
    return "incomplete", "INCOMPLETE"


def run(prepared: Path) -> dict[str, object]:
    report_path = prepared / "prepare_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("tool_revision") != TOOL_REVISION:
        raise ValueError("prepared candidate revision does not match this runner")
    cases = _case_matrix(report, "cases")
    runs: dict[str, dict[str, object]] = {}
    for label, _dt_s in _STEP_ROWS:
        runs[label] = _execute_case(
            prepared,
            label,
            str(cases[label]),
        )
    status, decision = _run_outcome(runs)
    trajectory_status = (
        "READY_FOR_LOCKED_CROSS_REPRESENTATION_EVALUATION"
        if status == "complete"
        else "BLOCKED_NO_COMPLETE_TRAJECTORY_MATRIX"
    )
    run_report: dict[str, object] = {
        "status": status,
        "decision": decision,
        "tool_revision": TOOL_REVISION,
        "public_api_path": ["load_case", "simulate", "open_result"],
        "prepare_report": report_path.name,
        "prepare_report_sha256": _sha256(report_path),
        "runs": runs,
        "physical_applicability": "NOT_TESTED",
        "cross_representation_trajectory_comparison": trajectory_status,
        "claim_policy": "comparison and acceptance are separate M3-C1 artifacts",
    }
    (prepared / "candidate_run_report.json").write_text(
        json.dumps(run_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("package", type=Path)
    prepare_parser.add_argument("base_data", type=Path)
    prepare_parser.add_argument("reference", type=Path)
    prepare_parser.add_argument("output", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("prepared", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        report = prepare(
            arguments.config.resolve(),
            arguments.package.resolve(),
            arguments.base_data.resolve(),
            arguments.reference.resolve(),
            arguments.output.resolve(),
        )
    else:
        report = run(arguments.prepared.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"prepared", "complete", "characterized"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
