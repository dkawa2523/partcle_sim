"""Prepare and run the deterministic M3-C1 100 nm Case-A/Case-P matrix.

This external V&V tool maps the two audited theory-consistent COMSOL packages
to producer-neutral canonical exact-connectivity P1 inputs.  It then runs only
the ordinary public solver API and exports a compact comparison projection.
It does not add a COMSOL execution mode or comparison logic to the solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import (
    DataBundle,
    FieldData,
    P1TriLayout,
    RealizedTableSource,
    read,
    write,
)

TOOL_REVISION: Final = "m3c1_theory_consistent_100nm_30ms_candidate_v2"
MANIFEST_RECOVERY_REVISION: Final = "m3c1_completed_cell_manifest_recovery_v1"
CONFIG_COPY: Final = "m3c1_theory_100nm_30ms_v2.json"
PREPARE_REPORT: Final = "prepare_report.json"
FINAL_REPORT: Final = "candidate_run_report.json"
WORKFLOWS: Final = ("caseA", "caseP")
RUN_KEYS: Final = ("coarse", "medium", "fine")
WORKFLOW_STEPS: Final = {
    "caseA": (6.25e-7, 3.125e-7, 1.5625e-7),
    "caseP": (4.6875e-8, 2.34375e-8, 1.171875e-8),
}
CHARGE_LIPSCHITZ_S_INV: Final = {
    "caseA": 287013.1640482939,
    "caseP": 10403456.669581516,
}
MAXIMUM_DT_CHARGE_LIPSCHITZ: Final = 0.5
PARTICLE_COUNT: Final = 287
OUTPUT_COUNT: Final = 121
END_TIME_S: Final = 0.03
_GROUP_NAMES: Final = (
    "wafer",
    "grounded_wall",
    "focus_transition",
    "outer_dielectric",
    "pump_outlet",
    "gas_inlet",
)
_LIFECYCLE: Final = {
    0: "pending",
    1: "active",
    2: "stuck",
    3: "escaped",
    4: "failed",
    5: "held",
}
_TRAJECTORY_HEADER: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
_EVENT_HEADER: Final = (
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
    "charge_number_pre_e",
    "charge_number_post_e",
    "boundary_id",
    "material_id",
    "law",
    "outcome",
    "localization_residual_m",
    "position_budget_m",
    "time_budget_s",
)
_FAILURE_HEADER: Final = (
    "particle_id",
    "event_ordinal",
    "event_time_s",
    "reason_code",
)
_SCALARS: Final = (
    ("gas_density", "gas_density_kg_per_m3", "kg/m^3"),
    ("gas_dynamic_viscosity", "dynamic_viscosity_Pa_s", "Pa*s"),
    ("gas_temperature", "gas_temperature_K", "K"),
    ("gas_mean_free_path", "gas_mean_free_path_m", "m"),
    ("electron_number_density", "electron_density_per_m3", "1/m^3"),
    ("positive_ion_number_density", "total_positive_ion_density_per_m3", "1/m^3"),
    ("electron_thermal_voltage", "electron_temperature_eV_as_V", "V"),
    ("positive_ion_thermal_voltage", "ion_thermal_energy_eV_as_V", "V"),
    ("effective_positive_ion_mass", "effective_positive_ion_mass_kg", "kg"),
    ("screening_length", "screening_length_m", "m"),
    ("ion_neutral_mean_free_path", "ion_neutral_mean_free_path_m", "m"),
    ("azimuthal_gas_vorticity", "axisymmetric_azimuthal_vorticity_per_s", "1/s"),
)
_VECTORS: Final = (
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


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(cast(Mapping[str, Any], value))


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    required = {
        "schema_version",
        "evaluation_id",
        "evaluation_revision",
        "classification",
        "theory_variant",
        "matrix",
        "workflows",
        "shared_geometry_sha256",
        "physics",
        "parameters",
        "boundaries",
        "acceptance",
        "claim_policy",
    }
    if set(config) != required:
        raise ValueError(f"configuration keys differ: {sorted(set(config) ^ required)}")
    if config["schema_version"] != 1 or config["evaluation_revision"] != 2:
        raise ValueError("unsupported M3-C1 matrix configuration revision")
    if config["theory_variant"] != "formal_iondrag_theory_consistent":
        raise ValueError("the matrix must select the theory-consistent ion-drag variant")
    _validate_matrix(_mapping(config["matrix"], "matrix"))
    _validate_workflows(_mapping(config["workflows"], "workflows"))
    _validate_physics(_mapping(config["physics"], "physics"))
    _validate_parameters(_mapping(config["parameters"], "parameters"))
    _validate_boundaries(_mapping(config["boundaries"], "boundaries"))
    _validate_acceptance(_mapping(config["acceptance"], "acceptance"))
    return config


def _validate_matrix(matrix: Mapping[str, object]) -> None:
    expected = {
        "particle_diameter_m": 1.0e-7,
        "particle_count": PARTICLE_COUNT,
        "time_end_s": END_TIME_S,
        "run_keys": list(RUN_KEYS),
        "output_schedule_source": "results/particle_history_full_tidy.csv",
        "output_schedule_segments": [
            {"start_s": 0.0, "stop_s": 5.0e-4, "step_s": 1.0e-5},
            {"start_s": 6.0e-4, "stop_s": 5.0e-3, "step_s": 1.0e-4},
            {"start_s": 6.0e-3, "stop_s": 3.0e-2, "step_s": 1.0e-3},
        ],
        "output_count": OUTPUT_COUNT,
    }
    if matrix != expected:
        raise ValueError("matrix does not match the preregistered 100 nm 30 ms schedule")


def _validate_workflows(workflows: Mapping[str, object]) -> None:
    if set(workflows) != set(WORKFLOWS):
        raise ValueError("workflows must contain exactly caseA and caseP")
    expected = {
        "caseA": ("bounded_screening_length_m", 0.1),
        "caseP": ("screening_length_m", 0.25),
    }
    for name, (screening_column, speed_ratio) in expected.items():
        workflow = _mapping(workflows[name], f"workflows.{name}")
        if workflow.get("screening_length_column") != screening_column:
            raise ValueError(f"{name} screening-length mapping differs")
        if workflow.get("maximum_neutral_speed_ratio") != speed_ratio:
            raise ValueError(f"{name} neutral speed-ratio limit differs")
        if workflow.get("fixed_rk4_steps_s") != list(WORKFLOW_STEPS[name]):
            raise ValueError(f"{name} fixed RK4 step sequence differs")
        if workflow.get("charge_lipschitz_s_inv") != CHARGE_LIPSCHITZ_S_INV[name]:
            raise ValueError(f"{name} charge Lipschitz bound differs")
        for label, step_s in _step_rows(name):
            step_count = round(END_TIME_S / step_s)
            represented_end = step_count * step_s
            if abs(represented_end - END_TIME_S) > max(
                math.ulp(represented_end), math.ulp(END_TIME_S)
            ):
                raise ValueError(f"{name} {label} does not divide the 30 ms interval exactly")
            if step_s * CHARGE_LIPSCHITZ_S_INV[name] > MAXIMUM_DT_CHARGE_LIPSCHITZ:
                raise ValueError(f"{name} {label} violates the charge Lipschitz step bound")
        if workflow.get("neutral_transport_classification") != (
            "producer_effective_gas_sensitivity_not_physical_certification"
        ):
            raise ValueError(f"{name} sensitivity classification differs")
        package = str(workflow.get("package_relative_path", ""))
        if f"formal_iondrag_theory_consistent/{name}_100nm/" not in package.replace("\\", "/"):
            raise ValueError(f"{name} package selection differs")
        hashes = _mapping(workflow.get("expected_sha256"), f"workflows.{name}.expected_sha256")
        if len(hashes) != 5 or any(not _is_sha256(value) for value in hashes.values()):
            raise ValueError(f"{name} expected input hashes are incomplete")


def _validate_physics(physics: Mapping[str, object]) -> None:
    expected = {
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
    if physics != expected:
        raise ValueError("physics does not match the preregistered deterministic model set")


def _validate_parameters(parameters: Mapping[str, object]) -> None:
    expected_keys = {
        "gas_molecular_mass_kg",
        "epstein_diffuse_fraction",
        "maximum_relative_ion_speed_m_s",
        "medium_relative_permittivity",
        "real_clausius_mossotti_factor",
        "maximum_point_dipole_radius_m",
        "point_dipole_classification",
        "lift_coefficient",
        "gravity_m_s2",
    }
    if set(parameters) != expected_keys:
        raise ValueError("parameter keys differ from the preregistered matrix")
    positive = expected_keys - {"gravity_m_s2", "point_dipole_classification"}
    if any(
        not math.isfinite(float(cast(Any, parameters[key])))
        or float(cast(Any, parameters[key])) <= 0
        for key in positive
    ):
        raise ValueError("matrix scalar parameters must be finite and positive")
    if parameters["gravity_m_s2"] != [0.0, -9.80665]:
        raise ValueError("gravity convention differs")
    if parameters["point_dipole_classification"] != (
        "benchmark_sensitivity_execution_guard_not_physical_certification"
    ):
        raise ValueError("point-dipole sensitivity classification differs")


def _validate_boundaries(boundaries: Mapping[str, object]) -> None:
    expected = {
        "wafer": "stick",
        "grounded_wall": "stick",
        "focus_transition": "stick",
        "outer_dielectric": "stick",
        "gas_inlet": "hold",
        "pump_outlet": "escape",
    }
    if boundaries != expected:
        raise ValueError("boundary mapping must be material=stick, inlet=hold, pump=escape")


def _validate_acceptance(acceptance: Mapping[str, object]) -> None:
    expected = {
        "initial_state_ulp_multiplier": 4096,
        "minimum_rms_order": 0.75,
        "roundoff_multiplier": 4096,
        "cross_envelope_safety_factor": 2.0,
        "maximum_dt_charge_lipschitz": MAXIMUM_DT_CHARGE_LIPSCHITZ,
    }
    if acceptance != expected:
        raise ValueError("acceptance constants differ from the preregistered matrix")


def _step_rows(workflow_name: str) -> tuple[tuple[str, float], ...]:
    return tuple(zip(RUN_KEYS, WORKFLOW_STEPS[workflow_name], strict=True))


def _charge_step_receipt(workflow_name: str) -> dict[str, object]:
    lipschitz = CHARGE_LIPSCHITZ_S_INV[workflow_name]
    rows = [
        {
            "run_key": label,
            "dt_s": step_s,
            "dt_charge_lipschitz": step_s * lipschitz,
            "step_count_30ms": round(END_TIME_S / step_s),
        }
        for label, step_s in _step_rows(workflow_name)
    ]
    return {
        "charge_lipschitz_s_inv": lipschitz,
        "maximum_dt_charge_lipschitz": MAXIMUM_DT_CHARGE_LIPSCHITZ,
        "all_steps_admissible": all(
            step_s * lipschitz <= MAXIMUM_DT_CHARGE_LIPSCHITZ
            for _, step_s in _step_rows(workflow_name)
        ),
        "runs": rows,
    }


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _dictionary(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(line for line in stream if not line.startswith("%"))
        rows = list(reader)
    if not rows or reader.fieldnames != ["column", "COMSOL_expression", "unit"]:
        raise ValueError(f"invalid field dictionary: {path}")
    names = [row["column"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate field columns: {path}")
    return names, {row["column"]: row["unit"] for row in rows}


def _numeric_table(path: Path, width: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(line for line in stream if not line.startswith("%"))
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != width:
                raise ValueError(f"{path}: row {line_number} has {len(row)} columns")
            rows.append([float(token) for token in row])
    if not rows:
        raise ValueError(f"empty numeric table: {path}")
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
            raise ValueError("canonical nodes do not have one unique provider match")
        local = np.argmin(squared, axis=1)
        matched[start:stop] = local
        distance2[start:stop] = squared[np.arange(stop - start), local]
    if np.unique(matched).size != matched.size:
        raise ValueError("provider-to-canonical field match is not bijective")
    return matched, float(np.sqrt(np.max(distance2, initial=0.0)))


def _field_columns(screening_column: str) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (name, screening_column if name == "screening_length" else column, unit)
        for name, column, unit in _SCALARS
    )


def _export_fields(
    package: Path,
    nodes_m: np.ndarray,
    layout_name: str,
    screening_column: str,
) -> tuple[tuple[FieldData, ...], dict[str, object]]:
    dictionary_path = package / "config/background_field_column_dictionary.csv"
    values_path = package / "input_fields/background_fields_mesh_points.csv"
    names, units = _dictionary(dictionary_path)
    index = {name: position for position, name in enumerate(names)}
    scalars = _field_columns(screening_column)
    expected_units = {
        "r_m": "m",
        "z_m": "m",
        **{column: unit for _, column, unit in scalars},
        **{column: unit for _, columns, unit in _VECTORS for column in columns},
        "thermal_conductivity_W_per_mK": "W/(m*K)",
        "temperature_gradient_r_K_per_m": "K/m",
        "temperature_gradient_z_K_per_m": "K/m",
    }
    wrong = {
        name: (units.get(name), unit)
        for name, unit in expected_units.items()
        if units.get(name) != unit
    }
    if wrong:
        raise ValueError(f"field columns or units differ: {wrong}")
    values = _numeric_table(values_path, len(names))
    selected = [index[name] for name in expected_units]
    finite = np.isfinite(values[:, selected]).all(axis=1)
    provider = values[finite]
    matched, maximum_distance = _coordinate_match(
        nodes_m,
        provider[:, [index["r_m"], index["z_m"]]],
        1.0e-14,
    )
    ordered = provider[matched]
    fields = _scalar_fields(ordered, index, scalars, layout_name)
    vectors, projections = _vector_fields(ordered, index, nodes_m, layout_name)
    fields.extend(vectors)
    heat_flux, heat_flux_projection = _heat_flux_field(ordered, index, nodes_m, layout_name)
    fields.append(heat_flux)
    projections["gas_translational_heat_flux"] = heat_flux_projection
    fields.sort(key=lambda field: field.name)
    return tuple(fields), {
        "field_count": len(fields),
        "component_count": sum(field.values.shape[1] for field in fields),
        "matched_node_count": int(nodes_m.shape[0]),
        "maximum_coordinate_match_distance_m": maximum_distance,
        "screening_length_source_column": screening_column,
        "axis_radial_projection": projections,
        "heat_flux_formation": {
            "formula": "q_effective=-thermal_conductivity*grad(gas_temperature)",
            "classification": "external_producer_transformation",
            "authority_status": "NOT_TESTED_PPR_NOJAC_PRIMITIVE_AUTHORITY",
        },
        "canonical_field_mapping": {
            **{name: column for name, column, _ in scalars},
            **{name: list(columns) for name, columns, _ in _VECTORS},
            "gas_translational_heat_flux": "derived_from_exported_k_and_temperature_gradient",
        },
    }


def _scalar_fields(
    values: np.ndarray,
    index: Mapping[str, int],
    scalars: Sequence[tuple[str, str, str]],
    layout_name: str,
) -> list[FieldData]:
    return [
        FieldData(
            name,
            layout_name,
            "node",
            ("value",),
            "scalar",
            np.ascontiguousarray(values[:, [index[column]]], dtype="<f8"),
            unit,
        )
        for name, column, unit in scalars
    ]


def _vector_fields(
    values: np.ndarray,
    index: Mapping[str, int],
    nodes_m: np.ndarray,
    layout_name: str,
) -> tuple[list[FieldData], dict[str, object]]:
    axis = nodes_m[:, 0] == 0.0
    fields: list[FieldData] = []
    projections: dict[str, object] = {}
    for name, columns, unit in _VECTORS:
        vector = np.ascontiguousarray(
            values[:, [index[columns[0]], index[columns[1]]]], dtype="<f8"
        )
        original = vector[axis, 0].copy()
        vector[axis, 0] = 0.0
        projections[name] = {
            "corrected_node_count": int(np.count_nonzero(original)),
            "maximum_correction": float(np.max(np.abs(original), initial=0.0)),
            "reason": "canonical_axisymmetric_vector_regularity",
        }
        fields.append(
            FieldData(name, layout_name, "node", ("r", "z"), "axisymmetric_rz", vector, unit)
        )
    return fields, projections


def _heat_flux_field(
    values: np.ndarray,
    index: Mapping[str, int],
    nodes_m: np.ndarray,
    layout_name: str,
) -> tuple[FieldData, dict[str, object]]:
    conductivity = values[:, index["thermal_conductivity_W_per_mK"]]
    gradient = values[
        :,
        [index["temperature_gradient_r_K_per_m"], index["temperature_gradient_z_K_per_m"]],
    ]
    heat_flux = np.ascontiguousarray(-conductivity[:, None] * gradient, dtype="<f8")
    axis = nodes_m[:, 0] == 0.0
    original = heat_flux[axis, 0].copy()
    heat_flux[axis, 0] = 0.0
    field = FieldData(
        "gas_translational_heat_flux",
        layout_name,
        "node",
        ("r", "z"),
        "axisymmetric_rz",
        heat_flux,
        "W/m^2",
    )
    projection: dict[str, object] = {
        "corrected_node_count": int(np.count_nonzero(original)),
        "maximum_correction": float(np.max(np.abs(original), initial=0.0)),
        "reason": "canonical_axisymmetric_vector_regularity",
    }
    return field, projection


def _release_source(package: Path) -> tuple[RealizedTableSource, dict[str, object]]:
    path = package / "results/release_state_t0_tidy.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != PARTICLE_COUNT:
        raise ValueError(f"{path}: expected {PARTICLE_COUNT} release rows")

    def vector(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype="<f8")

    particle_id = np.asarray([int(row["particle_id"]) for row in rows], dtype="<i8")
    expected_ids = np.arange(1, PARTICLE_COUNT + 1, dtype="<i8")
    if not np.array_equal(particle_id, expected_ids):
        raise ValueError("release particle IDs must be the ordered set 1..287")
    release_time = vector("time_s")
    diameter = vector("particle_diameter_m")
    radius = vector("particle_radius_m")
    mass = vector("particle_mass_kg")
    phi_velocity = vector("velocity_phi_m_per_s")
    if not bool((release_time == 0.0).all()) or not bool((phi_velocity == 0.0).all()):
        raise ValueError("the RZ meridional source requires t=0 and zero azimuthal velocity")
    if not np.allclose(diameter, 1.0e-7, rtol=2.0e-15, atol=0.0):
        raise ValueError("release table is not the audited 100 nm source")
    source = RealizedTableSource(
        name="particles",
        particle_id=particle_id,
        release_time_s=release_time,
        position_m=np.column_stack((vector("r_m"), vector("z_m"))).astype("<f8"),
        velocity_m_s=np.column_stack(
            (vector("velocity_r_m_per_s"), vector("velocity_z_m_per_s"))
        ).astype("<f8"),
        charge_number=vector("charge_number_e"),
        mass_kg=mass,
        drag_diameter_m=diameter,
        electrostatic_radius_m=radius,
        displaced_volume_m3=(math.pi * diameter**3 / 6.0).astype("<f8"),
        model_weight=np.ones(PARTICLE_COUNT, dtype="<f8"),
        material_id=np.zeros(PARTICLE_COUNT, dtype="<i4"),
    )
    return source, {
        "release_table_sha256": _sha256(path),
        "particle_count": PARTICLE_COUNT,
        "particle_id_min": 1,
        "particle_id_max": PARTICLE_COUNT,
        "initial_charge_min_e": float(np.min(source.charge_number)),
        "initial_charge_max_e": float(np.max(source.charge_number)),
        "maximum_electrostatic_radius_m": float(np.max(radius)),
        "preserved_t0_columns": [
            "particle_id",
            "time_s",
            "r_m",
            "z_m",
            "velocity_r_m_per_s",
            "velocity_z_m_per_s",
            "charge_number_e",
            "particle_diameter_m",
            "particle_radius_m",
            "particle_mass_kg",
        ],
    }


def _reference_schedule(package: Path) -> np.ndarray:
    path = package / "results/particle_history_full_tidy.csv"
    by_particle: dict[int, list[float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            by_particle.setdefault(int(row["particle_id"]), []).append(float(row["time_s"]))
    if set(by_particle) != set(range(1, PARTICLE_COUNT + 1)):
        raise ValueError(f"{path}: reference particle IDs differ")
    first = np.asarray(by_particle[1], dtype=np.float64)
    if first.size != OUTPUT_COUNT or any(
        not np.array_equal(np.asarray(times, dtype=np.float64), first)
        for times in by_particle.values()
    ):
        raise ValueError(f"{path}: reference output schedule is not a complete rectangular matrix")
    _validate_output_segments(first)
    return first


def _validate_output_segments(times: np.ndarray) -> None:
    segments = (
        (times[:51], 0.0, 5.0e-4, 1.0e-5),
        (times[51:96], 6.0e-4, 5.0e-3, 1.0e-4),
        (times[96:], 6.0e-3, 3.0e-2, 1.0e-3),
    )
    for values, start, stop, step in segments:
        roundoff = 16.0 * float(np.spacing(max(abs(start), abs(stop), abs(step))))
        if not math.isclose(float(values[0]), start, rel_tol=0.0, abs_tol=roundoff):
            raise ValueError("reference output segment start differs")
        if not math.isclose(float(values[-1]), stop, rel_tol=0.0, abs_tol=roundoff):
            raise ValueError("reference output segment stop differs")
        if not np.allclose(np.diff(values), step, rtol=0.0, atol=roundoff):
            raise ValueError("reference output segment spacing differs")


def _validate_package_hashes(
    package: Path,
    workflow: Mapping[str, object],
    shared_geometry: Mapping[str, object],
) -> dict[str, str]:
    expected = {
        **{
            str(name): str(value)
            for name, value in _mapping(workflow["expected_sha256"], "hashes").items()
        },
        **{str(name): str(value) for name, value in shared_geometry.items()},
    }
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = package / relative
        if not path.is_file():
            raise FileNotFoundError(f"package artifact is missing: {path}")
        actual = _sha256(path)
        if actual != expected_hash:
            raise ValueError(f"package artifact hash differs: {path}")
        observed[relative] = actual
    return observed


def _load_base(
    base_data: Path, shared_geometry: Mapping[str, object]
) -> tuple[DataBundle, P1TriLayout]:
    data = read(base_data)
    if data.coordinate_system != "axisymmetric_rz" or len(data.layouts) != 1:
        raise ValueError("base data must contain one axisymmetric RZ layout")
    layout = data.layouts[0]
    if not isinstance(layout, P1TriLayout) or layout.connectivity.shape[1] != 3:
        raise ValueError("base data must contain the exact triangular P1 layout")
    if not bool((layout.cell_support == 1).all()):
        raise ValueError("the exact P1 matrix requires every cell to be supported")
    if tuple(data.geometry.group_names) != _GROUP_NAMES:
        raise ValueError("base boundary group order differs from the matrix contract")
    provenance = _mapping(json.loads(data.provenance_json), "base provenance")
    metadata = _mapping(provenance.get("producer_metadata"), "base producer metadata")
    hashes = _mapping(metadata.get("source_file_hashes"), "base source hashes")
    aliases = {
        "geometry/mesh_vertices_si.csv": "vertices",
        "geometry/domain_triangles.csv": "triangles",
        "geometry/domain_quadrilaterals.csv": "quadrilaterals",
        "geometry/boundary_edges.csv": "boundary_edges",
    }
    for relative, alias in aliases.items():
        actual = str(hashes.get(alias, "")).removeprefix("sha256:")
        if actual != str(shared_geometry[relative]):
            raise ValueError(f"base canonical geometry hash differs for {relative}")
    return data, layout


def _case_document(
    workflow_name: str,
    workflow: Mapping[str, object],
    content_hash: str,
    dt_s: float,
    output_times: np.ndarray,
    config: Mapping[str, object],
) -> dict[str, object]:
    physics = _mapping(config["physics"], "physics")
    parameters = _mapping(config["parameters"], "parameters")
    boundaries = _mapping(config["boundaries"], "boundaries")
    delta = 1.0 + float(parameters["epstein_diffuse_fraction"]) * math.pi / 8.0
    maximum_speed_ratio = float(cast(Any, workflow["maximum_neutral_speed_ratio"]))
    ion_speed_limit = float(cast(Any, parameters["maximum_relative_ion_speed_m_s"]))
    return {
        "format_version": 2,
        "case": {
            "name": f"m3c1_theory_100nm_30ms_{workflow_name}_{dt_s:.9g}",
            "data_path": "candidate_input.h5",
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": "axisymmetric_rz_meridional"},
        "time": {"start_s": 0.0, "end_s": END_TIME_S, "dt_s": dt_s},
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
                "maximum_relative_ion_speed_m_s": ion_speed_limit,
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
                "delta": delta,
                "maximum_speed_ratio": maximum_speed_ratio,
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
                "maximum_relative_ion_speed_m_s": ion_speed_limit,
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
                "maximum_speed_ratio": maximum_speed_ratio,
                "applicability": "error",
            },
            "dielectrophoresis": {
                "model": "quasistatic_spherical",
                "revision": physics["dielectrophoresis_revision"],
                "gradient_mean_e_squared_field": "gradient_mean_e_squared",
                "medium_relative_permittivity": parameters["medium_relative_permittivity"],
                "real_clausius_mossotti_factor": parameters["real_clausius_mossotti_factor"],
                "maximum_point_dipole_radius_m": parameters["maximum_point_dipole_radius_m"],
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
            {"boundary_group": name, "priority": 10, "law": law} for name, law in boundaries.items()
        ],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": output_times.tolist()},
            },
            "probes": None,
        },
    }


def _workflow_bundle(
    base: DataBundle,
    layout: P1TriLayout,
    package: Path,
    workflow_name: str,
    workflow: Mapping[str, object],
    config_hash: str,
    base_hash: str,
) -> tuple[DataBundle, dict[str, object]]:
    fields, field_receipt = _export_fields(
        package,
        layout.nodes_m,
        layout.name,
        str(workflow["screening_length_column"]),
    )
    source, source_receipt = _release_source(package)
    provenance = _mapping(json.loads(base.provenance_json), "base provenance")
    provenance["m3c1_theory_100nm_30ms_candidate"] = {
        "tool_revision": TOOL_REVISION,
        "workflow": workflow_name,
        "theory_variant": "formal_iondrag_theory_consistent",
        "configuration_sha256": config_hash,
        "base_data_sha256": base_hash,
        "field_receipt": field_receipt,
        "source_receipt": source_receipt,
        "claim": "external_vv_common_exact_p1_input_not_physical_certification",
    }
    bundle = replace(
        base,
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        fields=fields,
        sources=(source,),
    )
    return bundle, {"field": field_receipt, "source": source_receipt}


def prepare(config_path: Path, base_data: Path, output: Path) -> dict[str, object]:
    """Prepare both package-specific canonical inputs and six case files."""

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config_path = config_path.resolve()
    base_data = base_data.resolve()
    config_hash = _sha256(config_path)
    base_hash = _sha256(base_data)
    config = _load_config(config_path)
    workflows = _mapping(config["workflows"], "workflows")
    shared_geometry = _mapping(config["shared_geometry_sha256"], "shared_geometry_sha256")
    base, layout = _load_base(base_data, shared_geometry)
    prepared: dict[str, tuple[DataBundle, np.ndarray, dict[str, object], dict[str, str]]] = {}
    common_schedule: np.ndarray | None = None
    for name in WORKFLOWS:
        workflow = _mapping(workflows[name], f"workflows.{name}")
        package = (_repository_root() / str(workflow["package_relative_path"])).resolve()
        hashes = _validate_package_hashes(package, workflow, shared_geometry)
        schedule = _reference_schedule(package)
        if common_schedule is not None and not np.array_equal(schedule, common_schedule):
            raise ValueError("Case A and Case P output schedules differ")
        common_schedule = schedule
        bundle, receipt = _workflow_bundle(
            base,
            layout,
            package,
            name,
            workflow,
            config_hash,
            base_hash,
        )
        prepared[name] = bundle, schedule, receipt, hashes

    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / CONFIG_COPY)
    shutil.copyfile(Path(__file__).resolve(), output / Path(__file__).name)
    workflow_reports: dict[str, dict[str, object]] = {}
    for name in WORKFLOWS:
        bundle, schedule, receipt, source_hashes = prepared[name]
        workflow = _mapping(workflows[name], f"workflows.{name}")
        workflow_root = output / name
        workflow_root.mkdir()
        input_path = workflow_root / "candidate_input.h5"
        info = write(input_path, bundle)
        cases: dict[str, dict[str, object]] = {}
        for label, dt_s in _step_rows(name):
            case_path = workflow_root / f"candidate_{label}.yaml"
            case_path.write_text(
                yaml.safe_dump(
                    _case_document(name, workflow, info.content_hash, dt_s, schedule, config),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            load_case(case_path)
            cases[label] = {
                "path": case_path.name,
                "sha256": _sha256(case_path),
                "dt_s": dt_s,
            }
        workflow_reports[name] = {
            "status": "PREPARED",
            "package_relative_path": workflow["package_relative_path"],
            "package_source_hashes": source_hashes,
            "input_path": f"{name}/candidate_input.h5",
            "input_sha256": _sha256(input_path),
            "input_content_hash": info.content_hash,
            "output_times_s": schedule.tolist(),
            "output_count": int(schedule.size),
            "cases": cases,
            "preparation_receipt": receipt,
            "neutral_transport": {
                "maximum_speed_ratio": workflow["maximum_neutral_speed_ratio"],
                "classification": workflow["neutral_transport_classification"],
            },
            "charge_step_receipt": _charge_step_receipt(name),
        }
    report: dict[str, object] = {
        "status": "PREPARED",
        "tool_revision": TOOL_REVISION,
        "configuration": CONFIG_COPY,
        "configuration_sha256": config_hash,
        "producer_source": Path(__file__).name,
        "producer_source_sha256": _sha256(output / Path(__file__).name),
        "base_data_sha256": base_hash,
        "matrix": config["matrix"],
        "physics_receipt": _physics_receipt(config),
        "boundary_mapping": config["boundaries"],
        "acceptance": config["acceptance"],
        "claim_policy": config["claim_policy"],
        "workflows": workflow_reports,
    }
    with (output / PREPARE_REPORT).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def _physics_receipt(config: Mapping[str, object]) -> dict[str, object]:
    physics = _mapping(config["physics"], "physics")
    return {
        "brownian_active": physics["brownian_active"],
        "saffman_active": physics["saffman_active"],
        "charge_revision": physics["charge_revision"],
        "deterministic_revisions": {
            name: physics[f"{name}_revision"]
            for name in (
                "drag",
                "electric",
                "ion_drag",
                "thermophoresis",
                "dielectrophoresis",
                "lift",
                "gravity_buoyancy",
            )
        },
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _write_trajectory(path: Path, result: Any) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_TRAJECTORY_HEADER)
        for frame in result.iter_frames():
            for row, particle_id in enumerate(frame.particle_id):
                writer.writerow(
                    (
                        int(particle_id),
                        _number(frame.time_s),
                        _number(frame.position_m[row, 0]),
                        _number(frame.position_m[row, 1]),
                        _number(frame.velocity_m_s[row, 0]),
                        _number(frame.velocity_m_s[row, 1]),
                        _number(frame.charge_number[row]),
                        _LIFECYCLE[int(frame.lifecycle[row])],
                    )
                )
                count += 1
    return count


def _write_events(path: Path, result: Any) -> int:
    events = result.read_boundary_events()
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_EVENT_HEADER)
        for row, particle_id in enumerate(events.particle_id):
            writer.writerow(
                (
                    int(particle_id),
                    int(events.event_ordinal[row]),
                    _number(events.time_s[row]),
                    _number(events.position_m[row, 0]),
                    _number(events.position_m[row, 1]),
                    _number(events.normal[row, 0]),
                    _number(events.normal[row, 1]),
                    _number(events.velocity_pre_m_s[row, 0]),
                    _number(events.velocity_pre_m_s[row, 1]),
                    _number(events.velocity_post_m_s[row, 0]),
                    _number(events.velocity_post_m_s[row, 1]),
                    _number(events.charge_number_pre[row]),
                    _number(events.charge_number_post[row]),
                    int(events.boundary_id[row]),
                    int(events.material_id[row]),
                    str(events.law_id[row]),
                    str(events.outcome[row]),
                    _number(events.localization_residual_m[row]),
                    _number(events.position_budget_m[row]),
                    _number(events.time_budget_s[row]),
                )
            )
    return int(events.particle_id.size)


def _write_failures(path: Path, result: Any) -> int:
    failures = result.read_failure_events()
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_FAILURE_HEADER)
        for row, particle_id in enumerate(failures.particle_id):
            writer.writerow(
                (
                    int(particle_id),
                    int(failures.event_ordinal[row]),
                    _number(failures.time_s[row]),
                    int(failures.reason_code[row]),
                )
            )
    return int(failures.particle_id.size)


def _number(value: object) -> str:
    return format(float(cast(Any, value)), ".17g")


def _count_projection_rows(path: Path, expected_header: tuple[str, ...]) -> int:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != list(expected_header):
            raise ValueError(f"projection header differs: {path}")
        count = 0
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(expected_header):
                raise ValueError(f"projection row width differs: {path}:{line_number}")
            count += 1
    return count


def _load_prepared(prepared: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = prepared / PREPARE_REPORT
    report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "prepare report")
    if report.get("status") != "PREPARED" or report.get("tool_revision") != TOOL_REVISION:
        raise ValueError("prepared matrix does not belong to this tool revision")
    config_path = prepared / CONFIG_COPY
    config = _load_config(config_path)
    if _sha256(config_path) != report.get("configuration_sha256"):
        raise ValueError("prepared configuration hash differs")
    return report, config


def _cell_paths(root: Path, label: str) -> dict[str, Path]:
    return {
        "result": root / f"result_{label}",
        "partial": root / f"result_{label}.partial",
        "trajectory": root / f"candidate_trajectory_{label}.csv",
        "events": root / f"candidate_events_{label}.csv",
        "failures": root / f"candidate_failures_{label}.csv",
        "manifest": root / f"candidate_run_manifest_{label}.json",
    }


def _cell_context(
    prepared: Path, workflow_name: str, label: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, Path],
]:
    report, config = _load_prepared(prepared)
    if workflow_name not in WORKFLOWS or label not in RUN_KEYS:
        raise ValueError("unknown workflow or step label")
    workflows = _mapping(report["workflows"], "prepared workflows")
    workflow_report = _mapping(workflows[workflow_name], f"workflows.{workflow_name}")
    cases = _mapping(workflow_report["cases"], "prepared cases")
    case_record = _mapping(cases[label], f"cases.{label}")
    root = prepared / workflow_name
    paths = _cell_paths(root, label)
    case_path = root / str(case_record["path"])
    return report, config, workflow_report, case_record, case_path, paths


def _cell_receipt(
    report: Mapping[str, object],
    config: Mapping[str, object],
    workflow_name: str,
    label: str,
    workflow_report: Mapping[str, object],
    case_record: Mapping[str, object],
    case_path: Path,
    paths: Mapping[str, Path],
    manifest: Mapping[str, object],
    row_counts: tuple[int, int, int],
    manifest_recovery: Mapping[str, object] | None = None,
) -> dict[str, object]:
    trajectory_rows, event_rows, failure_rows = row_counts
    result_counts = _result_counts(manifest)
    complete = (
        manifest.get("status") == "complete"
        and result_counts["particles"] == PARTICLE_COUNT
        and result_counts["frames"] == OUTPUT_COUNT
        and result_counts["release_events"] == PARTICLE_COUNT
        and trajectory_rows == result_counts["frame_rows"]
        and event_rows == result_counts["boundary_events"]
        and failure_rows == result_counts["failure_events"]
        and failure_rows == 0
    )
    cell: dict[str, object] = {
        "status": "COMPLETE" if complete else "BLOCKED",
        "tool_revision": TOOL_REVISION,
        "workflow": workflow_name,
        "step_label": label,
        "dt_s": case_record["dt_s"],
        "public_api_path": ["load_case", "simulate", "open_result"],
        "configuration_sha256": report["configuration_sha256"],
        "case": case_path.name,
        "case_sha256": _sha256(case_path),
        "input_content_hash": workflow_report["input_content_hash"],
        "input_sha256": workflow_report["input_sha256"],
        "result": paths["result"].name,
        "result_manifest_sha256": _sha256(paths["result"] / "run.json"),
        "result_manifest_status": manifest.get("status"),
        "result_counts": result_counts,
        "trajectory": paths["trajectory"].name,
        "trajectory_sha256": _sha256(paths["trajectory"]),
        "trajectory_rows": trajectory_rows,
        "events": paths["events"].name,
        "events_sha256": _sha256(paths["events"]),
        "event_rows": event_rows,
        "failures": paths["failures"].name,
        "failures_sha256": _sha256(paths["failures"]),
        "failure_rows": failure_rows,
        "failure_reason_counts": manifest.get("failure_reason_counts"),
        "algorithm_revisions": {
            key: value
            for key, value in manifest.items()
            if key.endswith("_revision") and value is not None
        },
        "resolved_physics_models": _resolved_physics(manifest),
        "physics_receipt": _physics_receipt(config),
    }
    if manifest_recovery is not None:
        cell["manifest_recovery"] = dict(manifest_recovery)
    return cell


def _write_cell_receipt(path: Path, cell: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(cell, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _result_counts(manifest: Mapping[str, object]) -> dict[str, int]:
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("result manifest is missing counts")
    required = (
        "particles",
        "frames",
        "frame_rows",
        "release_events",
        "boundary_events",
        "failure_events",
    )
    result: dict[str, int] = {}
    for name in required:
        value = counts.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"result manifest count is invalid: {name}")
        result[name] = value
    return result


def run_cell(prepared: Path, workflow_name: str, label: str) -> dict[str, object]:
    """Run one no-clobber matrix cell through the three public APIs."""

    report, config, workflow_report, case_record, case_path, paths = _cell_context(
        prepared, workflow_name, label
    )
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"matrix cell output already exists: {existing}")
    case = load_case(case_path)
    simulate(case, paths["result"])
    result = open_result(paths["result"])
    trajectory_rows = _write_trajectory(paths["trajectory"], result)
    event_rows = _write_events(paths["events"], result)
    failure_rows = _write_failures(paths["failures"], result)
    manifest = _mapping(result.manifest, "result manifest")
    cell = _cell_receipt(
        report,
        config,
        workflow_name,
        label,
        workflow_report,
        case_record,
        case_path,
        paths,
        manifest,
        (trajectory_rows, event_rows, failure_rows),
    )
    _write_cell_receipt(paths["manifest"], cell)
    return cell


def recover_cell_manifest(prepared: Path, workflow_name: str, label: str) -> dict[str, object]:
    """Write only the missing receipt for an already completed matrix cell."""

    report, config, workflow_report, case_record, case_path, paths = _cell_context(
        prepared, workflow_name, label
    )
    if paths["manifest"].exists():
        raise FileExistsError(f"matrix cell manifest already exists: {paths['manifest']}")
    required = ("result", "trajectory", "events", "failures")
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"completed matrix cell artifact is missing: {missing}")
    if paths["partial"].exists():
        raise ValueError(f"partial result remains beside completed cell: {paths['partial']}")

    result = open_result(paths["result"])
    manifest = _mapping(result.manifest, "result manifest")
    row_counts = (
        _count_projection_rows(paths["trajectory"], _TRAJECTORY_HEADER),
        _count_projection_rows(paths["events"], _EVENT_HEADER),
        _count_projection_rows(paths["failures"], _FAILURE_HEADER),
    )
    recovery = {
        "revision": MANIFEST_RECOVERY_REVISION,
        "reason": "post_run_result_manifest_read_only_mapping_rejected",
        "solver_reexecuted": False,
        "projections_rewritten": False,
        "reused_artifacts": [paths[name].name for name in required],
        "recovery_source": Path(__file__).name,
        "recovery_source_sha256": _sha256(Path(__file__).resolve()),
    }
    cell = _cell_receipt(
        report,
        config,
        workflow_name,
        label,
        workflow_report,
        case_record,
        case_path,
        paths,
        manifest,
        row_counts,
        recovery,
    )
    if cell["status"] != "COMPLETE":
        raise ValueError("manifest recovery artifacts differ from the completed result counts")
    _write_cell_receipt(paths["manifest"], cell)
    return cell


def _resolved_physics(manifest: Mapping[str, object]) -> object:
    resolved = manifest.get("resolved")
    if not isinstance(resolved, Mapping) or "physics_models" not in resolved:
        raise ValueError("result manifest is missing resolved physics models")
    return resolved["physics_models"]


def finalize(prepared: Path) -> dict[str, object]:
    """Create the immutable root report after all six cells exist."""

    if (prepared / FINAL_REPORT).exists():
        raise FileExistsError(f"final report already exists: {prepared / FINAL_REPORT}")
    prepare_report, config = _load_prepared(prepared)
    prepared_workflows = _mapping(prepare_report["workflows"], "prepared workflows")
    workflows: dict[str, dict[str, object]] = {}
    overall_complete = True
    for workflow_name in WORKFLOWS:
        source = _mapping(prepared_workflows[workflow_name], workflow_name)
        runs: dict[str, dict[str, object]] = {}
        for label, _ in _step_rows(workflow_name):
            manifest_path = prepared / workflow_name / f"candidate_run_manifest_{label}.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"matrix cell manifest is missing: {manifest_path}")
            cell = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "cell manifest")
            if cell.get("workflow") != workflow_name or cell.get("step_label") != label:
                raise ValueError(f"matrix cell identity differs: {manifest_path}")
            if cell.get("configuration_sha256") != prepare_report["configuration_sha256"]:
                raise ValueError(f"matrix cell configuration hash differs: {manifest_path}")
            overall_complete &= cell.get("status") == "COMPLETE"
            runs[label] = {
                **cell,
                "manifest": manifest_path.name,
                "manifest_sha256": _sha256(manifest_path),
            }
        workflows[workflow_name] = {
            "status": "COMPLETE"
            if all(run["status"] == "COMPLETE" for run in runs.values())
            else "BLOCKED",
            "input_content_hash": source["input_content_hash"],
            "input_sha256": source["input_sha256"],
            "physics_receipt": _physics_receipt(config),
            "neutral_transport": source["neutral_transport"],
            "charge_step_receipt": source["charge_step_receipt"],
            "runs": runs,
        }
    final: dict[str, object] = {
        "status": "COMPLETE" if overall_complete else "BLOCKED",
        "tool_revision": TOOL_REVISION,
        "configuration_sha256": prepare_report["configuration_sha256"],
        "prepare_report": PREPARE_REPORT,
        "prepare_report_sha256": _sha256(prepared / PREPARE_REPORT),
        "physics_receipt": _physics_receipt(config),
        "boundary_mapping": config["boundaries"],
        "acceptance": config["acceptance"],
        "claim_policy": config["claim_policy"],
        "workflows": workflows,
    }
    with (prepared / FINAL_REPORT).open("x", encoding="utf-8") as stream:
        json.dump(final, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("base_data", type=Path)
    prepare_parser.add_argument("output", type=Path)
    cell_parser = subparsers.add_parser("run-cell")
    cell_parser.add_argument("prepared", type=Path)
    cell_parser.add_argument("workflow", choices=WORKFLOWS)
    cell_parser.add_argument("step", choices=RUN_KEYS)
    recovery_parser = subparsers.add_parser("recover-cell-manifest")
    recovery_parser.add_argument("prepared", type=Path)
    recovery_parser.add_argument("workflow", choices=WORKFLOWS)
    recovery_parser.add_argument("step", choices=RUN_KEYS)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("prepared", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        report = prepare(arguments.config, arguments.base_data, arguments.output)
    elif arguments.command == "run-cell":
        report = run_cell(arguments.prepared.resolve(), arguments.workflow, arguments.step)
    elif arguments.command == "recover-cell-manifest":
        report = recover_cell_manifest(
            arguments.prepared.resolve(), arguments.workflow, arguments.step
        )
    else:
        report = finalize(arguments.prepared.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"PREPARED", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
