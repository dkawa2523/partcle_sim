"""Localize the M3-C1 native-field versus exact-P1 difference.

This external V&V tool evaluates the locked canonical P1 fields at the saved
COMSOL native-field states.  It changes neither the production solver nor the
COMSOL model.  Its decision is a layer localization, not an accuracy pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from chamber_particles.case_format import P1TriLayout, read_with_info
from chamber_particles.fields import RequiredFieldMetadata, prepare_required_fields
from tools.vv.comsol.evaluate_pre_event_frozen_rhs import (
    _load_records,
    _production_evaluations,
)
from tools.vv.comsol.evaluate_pre_event_frozen_rhs import load_config as load_frozen_config

type FloatArray = NDArray[np.float64]
type Record = dict[str, Any]

TOOL_REVISION: Final = "m3c1_field_representation_localization_v1"
FIELD_METRICS_FILE: Final = "field_metrics.csv"
RHS_METRICS_FILE: Final = "rhs_metrics.csv"
GATES_FILE: Final = "gates.csv"
REPORT_FILE: Final = "localization_report.json"
_EXPECTED_ROUNDOFF_LIMIT: Final = 1.0e-12
_EXPECTED_COORDINATE_TOLERANCE_M: Final = 1.0e-14

_PPR_PARTICLE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "current_status_code",
    "thermophoretic_force_r_N",
    "thermophoretic_force_z_N",
    "gas_temperature_K",
    "gas_thermal_conductivity_W_per_mK",
    "unrecovered_temperature_gradient_r_K_per_m",
    "unrecovered_temperature_gradient_z_K_per_m",
    "ppr_temperature_gradient_r_K_per_m",
    "ppr_temperature_gradient_z_K_per_m",
    "ppr_heat_flux_r_W_per_m2",
    "ppr_heat_flux_z_W_per_m2",
    "particle_diameter_m",
    "background_gas_molar_mass_kg_per_mol",
)
_PPR_DOMAIN_COLUMNS: Final = (
    "r_m",
    "z_m",
    "domain_id",
    "gas_temperature_K",
    "gas_thermal_conductivity_W_per_mK",
    "ppr_temperature_gradient_r_K_per_m",
    "ppr_temperature_gradient_z_K_per_m",
    "ppr_heat_flux_r_W_per_m2",
    "ppr_heat_flux_z_W_per_m2",
)


@dataclass(frozen=True, slots=True)
class ArtifactLock:
    path: Path
    sha256: str | None


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    native_columns: tuple[str, ...] | None
    unit: str


_FIELD_SPECS: Final = (
    FieldSpec("gas_density", ("gas_density_kg_per_m3",), "kg/m^3"),
    FieldSpec("gas_dynamic_viscosity", ("gas_dynamic_viscosity_Pa_s",), "Pa*s"),
    FieldSpec("gas_temperature", ("gas_temperature_K",), "K"),
    FieldSpec("gas_mean_free_path", ("gas_mean_free_path_m",), "m"),
    FieldSpec("electron_number_density", ("electron_density_per_m3",), "1/m^3"),
    FieldSpec("positive_ion_number_density", ("positive_ion_density_per_m3",), "1/m^3"),
    FieldSpec("electron_thermal_voltage", None, "V"),
    FieldSpec("positive_ion_thermal_voltage", ("ion_thermal_energy_eV_as_V",), "V"),
    FieldSpec("effective_positive_ion_mass", ("positive_ion_mass_kg",), "kg"),
    FieldSpec("screening_length", ("screening_length_m",), "m"),
    FieldSpec("ion_neutral_mean_free_path", ("ion_neutral_mean_free_path_m",), "m"),
    FieldSpec("azimuthal_gas_vorticity", ("azimuthal_vorticity_per_s",), "1/s"),
    FieldSpec(
        "gas_velocity",
        ("gas_velocity_r_m_per_s", "gas_velocity_z_m_per_s"),
        "m/s",
    ),
    FieldSpec(
        "electric_field",
        ("electric_field_r_V_per_m", "electric_field_z_V_per_m"),
        "V/m",
    ),
    FieldSpec(
        "positive_ion_velocity",
        ("ion_velocity_r_m_per_s", "ion_velocity_z_m_per_s"),
        "m/s",
    ),
    FieldSpec(
        "gradient_mean_e_squared",
        ("gradient_E2_r_V2_per_m3", "gradient_E2_z_V2_per_m3"),
        "V^2/m^3",
    ),
    FieldSpec(
        "gas_translational_heat_flux",
        ("effective_heat_flux_r_W_per_m2", "effective_heat_flux_z_W_per_m2"),
        "W/m^2",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _locked_artifact(
    root: Path, payload: object, label: str, *, hash_required: bool
) -> ArtifactLock:
    record = _mapping(payload, label)
    expected_keys = {"path", "sha256"} if hash_required else {"path"}
    if set(record) != expected_keys or not isinstance(record.get("path"), str):
        raise ValueError(f"{label} must contain exactly {sorted(expected_keys)}")
    path = (root / str(record["path"])).resolve()
    if not path.exists():
        raise ValueError(f"locked artifact does not exist: {path}")
    expected_hash = str(record["sha256"]) if hash_required else None
    if expected_hash is not None and _sha256(path) != expected_hash:
        raise ValueError(f"locked artifact hash differs: {path}")
    return ArtifactLock(path=path, sha256=expected_hash)


def _load_protocol(root: Path, config_path: Path) -> tuple[Record, dict[str, ArtifactLock]]:
    payload = _mapping(json.loads(config_path.read_text(encoding="utf-8")), "configuration")
    required = {
        "schema_version",
        "evaluation_id",
        "evaluation_revision",
        "classification",
        "scope",
        "artifacts",
        "acceptance",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("unsupported field-localization configuration")
    if payload["evaluation_revision"] != 1:
        raise ValueError("unsupported field-localization revision")
    scope = _mapping(payload["scope"], "scope")
    expected_scope = {
        "workflow": "caseA",
        "diameter_m": 1.0e-7,
        "particles": 287,
        "frames": 46,
        "records": 13202,
        "time_window_s": [0.0, 4.5e-4],
        "output_interval_s": 1.0e-5,
        "fixed_rk4_step_s": 1.5625e-7,
        "brownian_active": False,
        "comparison_scopes": ["identical_t0_state", "saved_native_path_states"],
        "claims_integrated_trajectory_attribution": False,
        "claims_native_fe_reconstruction": False,
    }
    if scope != expected_scope:
        raise ValueError("field-localization scope differs from the locked slice")
    acceptance = _mapping(payload["acceptance"], "acceptance")
    expected_acceptance = {
        "roundoff_relative_l2": _EXPECTED_ROUNDOFF_LIMIT,
        "ppr_node_coordinate_tolerance_m": _EXPECTED_COORDINATE_TOLERANCE_M,
        "ppr_node_heat_flux_relative_l2": _EXPECTED_ROUNDOFF_LIMIT,
        "require_complete_native_cohort": True,
        "require_all_p1_samples_supported": True,
        "require_t0_axis_incident_cell_exclusion": True,
        "result_dependent_tolerance_tuning": "PROHIBITED",
    }
    if acceptance != expected_acceptance:
        raise ValueError("field-localization acceptance policy differs from revision 1")
    artifacts = _mapping(payload["artifacts"], "artifacts")
    expected_artifacts = {
        "candidate_input",
        "candidate_configuration",
        "native_step_directory",
        "frozen_rhs_configuration",
        "ppr_particle_table",
        "ppr_domain3_table",
        "cross_representation_evidence",
        "common_field_evidence",
        "frozen_rhs_evidence",
        "ppr_evidence",
    }
    if set(artifacts) != expected_artifacts:
        raise ValueError("field-localization artifact set differs from revision 1")
    candidate = _mapping(artifacts["candidate_input"], "candidate_input")
    if set(candidate) != {"path", "sha256", "content_hash"}:
        raise ValueError("candidate_input must also lock its canonical content hash")
    locks = {
        name: _locked_artifact(
            root,
            (
                {"path": candidate["path"], "sha256": candidate["sha256"]}
                if name == "candidate_input"
                else artifacts[name]
            ),
            name,
            hash_required=name != "native_step_directory",
        )
        for name in sorted(expected_artifacts)
    }
    return payload, locks


def _json(path: Path) -> Record:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _verify_prior_decisions(locks: dict[str, ArtifactLock]) -> Record:
    cross = _json(locks["cross_representation_evidence"].path)
    common = _json(locks["common_field_evidence"].path)
    frozen = _json(locks["frozen_rhs_evidence"].path)
    ppr = _json(locks["ppr_evidence"].path)
    cross_comparison = _mapping(cross.get("cross_representation_comparison"), "cross result")
    frozen_summary = _mapping(frozen.get("summary"), "frozen RHS summary")
    if (
        cross.get("evaluation_status") != "COMPLETE"
        or cross_comparison.get("status") != "FAIL"
        or cross_comparison.get("failed_gates") != 6
    ):
        raise ValueError("locked cross-representation evidence is not the 6/6 FAIL")
    if (
        common.get("evaluation_status") != "COMPLETE"
        or common.get("overall_decision")
        != "PASS_LOCKED_SAME_FIELD_COMMON_CANONICAL_P1_CASE_AND_WINDOW"
    ):
        raise ValueError("locked common-field evidence is not PASS")
    if (
        frozen_summary.get("producer_formula_replay_fail") != 0
        or frozen_summary.get("producer_formula_replay_pass") != 7
        or frozen_summary.get("producer_formula_replay_not_tested") != 1
    ):
        raise ValueError("locked frozen-RHS evidence does not retain its 7/8 decision")
    if ppr.get("overall_status") != "PASS":
        raise ValueError("locked PPR closure evidence is not PASS")
    return {
        "cross_representation": "FAIL_6_OF_6",
        "same_field": "PASS_9_OF_9",
        "frozen_producer_forms": "PASS_7_OF_8_WITH_ONE_LATER_PPR_CLOSURE",
        "ppr_thermophoresis": "PASS",
    }


def _validate_native_cohort(values: dict[str, FloatArray], scope: Record) -> Record:
    particle = values["particle_id"]
    rounded = np.rint(particle).astype(np.int64)
    if not np.array_equal(particle, rounded.astype(np.float64)):
        raise ValueError("native particle IDs are not exact integers")
    time = values["time_s"]
    keys = np.rec.fromarrays((rounded, time), names=("particle", "time"))
    expected_ids = np.arange(1, int(scope["particles"]) + 1)
    expected_times = np.arange(int(scope["frames"]), dtype=np.float64) * float(
        scope["output_interval_s"]
    )
    if (
        np.unique(keys).size != int(scope["records"])
        or not np.array_equal(np.unique(rounded), expected_ids)
        or not np.array_equal(np.unique(time), expected_times)
    ):
        raise ValueError("native saved states do not form the complete locked cohort")
    return {
        "status": "PASS",
        "records": int(time.size),
        "particles": int(np.unique(rounded).size),
        "frames": int(np.unique(time).size),
        "all_active": bool((values["current_status_code"] == 1.0).all()),
    }


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"{path}: columns differ from the locked contract")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed or empty CSV")
    return rows


def _authoritative_ppr_heat(
    values: dict[str, FloatArray], path: Path, expected_records: int
) -> tuple[dict[str, FloatArray], Record]:
    rows = _read_csv(path, _PPR_PARTICLE_COLUMNS)
    if len(rows) != expected_records:
        raise ValueError("PPR particle table does not contain the locked cohort")
    indexed: dict[tuple[int, float], dict[str, str]] = {}
    for row in rows:
        key = (int(row["particle_id"]), float(row["time_s"]))
        if key in indexed:
            raise ValueError("PPR particle table contains duplicate keys")
        indexed[key] = row
    ordered: list[dict[str, str]] = []
    for particle, time in zip(values["particle_id"], values["time_s"], strict=True):
        key = (int(particle), float(time))
        if key not in indexed:
            raise ValueError("PPR particle table is missing a native saved-state key")
        ordered.append(indexed[key])
    numeric = np.asarray(
        [
            [
                float(row["r_m"]),
                float(row["z_m"]),
                float(row["current_status_code"]),
                float(row["ppr_heat_flux_r_W_per_m2"]),
                float(row["ppr_heat_flux_z_W_per_m2"]),
            ]
            for row in ordered
        ],
        dtype=np.float64,
    )
    if not bool(np.isfinite(numeric).all()) or not bool((numeric[:, 2] == 1.0).all()):
        raise ValueError("PPR particle table contains nonfinite or non-active rows")
    native_position = np.column_stack((values["r_m"], values["z_m"]))
    if not np.array_equal(numeric[:, :2], native_position):
        raise ValueError("PPR particle coordinates differ from the locked native states")
    updated = {name: column.copy() for name, column in values.items()}
    updated["effective_heat_flux_r_W_per_m2"] = numeric[:, 3]
    updated["effective_heat_flux_z_W_per_m2"] = numeric[:, 4]
    return updated, {
        "status": "PASS",
        "records": len(ordered),
        "keys_and_coordinates_identical_to_native": True,
        "all_active_and_finite": True,
    }


def _match_coordinates(
    canonical: FloatArray, reference: FloatArray
) -> tuple[NDArray[np.int64], float]:
    if np.unique(reference, axis=0).shape[0] != reference.shape[0]:
        raise ValueError("PPR domain coordinates are not unique")
    matched = np.empty(canonical.shape[0], dtype=np.int64)
    distance2 = np.empty(canonical.shape[0], dtype=np.float64)
    for start in range(0, canonical.shape[0], 256):
        stop = min(start + 256, canonical.shape[0])
        delta = canonical[start:stop, None, :] - reference[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        nearest = np.argmin(squared, axis=1)
        matched[start:stop] = nearest
        distance2[start:stop] = squared[np.arange(stop - start), nearest]
    maximum = float(np.sqrt(np.max(distance2, initial=0.0)))
    if maximum > _EXPECTED_COORDINATE_TOLERANCE_M or np.unique(matched).size != matched.size:
        raise ValueError("PPR domain points do not bijectively match canonical nodes")
    return matched, maximum


def _metric(
    candidate: FloatArray, reference: FloatArray, particle: FloatArray, time: FloatArray
) -> Record:
    candidate_2d = candidate[:, None] if candidate.ndim == 1 else candidate
    reference_2d = reference[:, None] if reference.ndim == 1 else reference
    if candidate_2d.shape != reference_2d.shape or candidate_2d.shape[0] != particle.size:
        raise ValueError("metric arrays do not share one row cohort")
    difference = candidate_2d - reference_2d
    row_norm = np.linalg.norm(difference, axis=1)
    worst = int(np.argmax(row_norm))
    denominator = max(float(np.linalg.norm(reference_2d)), float(np.finfo(np.float64).tiny))
    relative = float(np.linalg.norm(difference)) / denominator
    return {
        "count": int(candidate_2d.shape[0]),
        "components": int(candidate_2d.shape[1]),
        "relative_l2": relative,
        "rms_vector_or_scalar": float(np.sqrt(np.mean(row_norm * row_norm))),
        "maximum_vector_or_scalar": float(row_norm[worst]),
        "worst_particle_id": int(particle[worst]),
        "worst_time_s": float(time[worst]),
        "classification": (
            "DIFFERENT_ABOVE_ROUNDOFF"
            if relative > _EXPECTED_ROUNDOFF_LIMIT
            else "INDISTINGUISHABLE_AT_ROUNDOFF"
        ),
    }


def _bind_ppr_nodes(data: Any, path: Path) -> Record:
    layout = data.layouts[0]
    if not isinstance(layout, P1TriLayout):
        raise ValueError("localization requires one canonical P1 layout")
    rows = _read_csv(path, _PPR_DOMAIN_COLUMNS)
    reference_coordinates = np.asarray(
        [[float(row["r_m"]), float(row["z_m"])] for row in rows], dtype=np.float64
    )
    reference_heat = np.asarray(
        [
            [float(row["ppr_heat_flux_r_W_per_m2"]), float(row["ppr_heat_flux_z_W_per_m2"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not bool(np.isfinite(reference_coordinates).all()) or not bool(
        np.isfinite(reference_heat).all()
    ):
        raise ValueError("PPR domain table contains nonfinite values")
    match, maximum_distance = _match_coordinates(layout.nodes_m, reference_coordinates)
    matched_heat = reference_heat[match]
    axis = layout.nodes_m[:, 0] == 0.0
    projected = matched_heat.copy()
    maximum_axis_projection = float(np.max(np.abs(projected[axis, 0]), initial=0.0))
    projected[axis, 0] = 0.0
    fields = {field.name: field for field in data.fields}
    candidate_heat = fields["gas_translational_heat_flux"].values
    dummy = np.arange(layout.nodes_m.shape[0], dtype=np.float64)
    metric = _metric(candidate_heat, projected, dummy, np.zeros_like(dummy))
    if float(metric["relative_l2"]) > _EXPECTED_ROUNDOFF_LIMIT:
        raise ValueError("canonical heat-flux nodes do not match the authoritative PPR samples")
    return {
        "status": "PASS",
        "matched_nodes": int(layout.nodes_m.shape[0]),
        "unique_reference_nodes": int(np.unique(match).size),
        "maximum_coordinate_difference_m": maximum_distance,
        "axis_node_count": int(np.count_nonzero(axis)),
        "maximum_registered_axis_radial_projection_W_per_m2": maximum_axis_projection,
        "after_axis_policy_relative_l2": metric["relative_l2"],
        "after_axis_policy_maximum_absolute_W_per_m2": metric["maximum_vector_or_scalar"],
    }


def _axis_exclusion(data: Any, radius: FloatArray, t0: NDArray[np.bool_]) -> Record:
    layout = data.layouts[0]
    if not isinstance(layout, P1TriLayout):
        raise ValueError("axis exclusion requires a P1 layout")
    axis_nodes = np.flatnonzero(layout.nodes_m[:, 0] == 0.0)
    incident = np.isin(layout.connectivity, axis_nodes).any(axis=1)
    maximum_incident_radius = float(np.max(layout.nodes_m[layout.connectivity[incident], 0]))
    minimum_t0_radius = float(np.min(radius[t0]))
    passed = minimum_t0_radius > maximum_incident_radius
    if not passed:
        raise ValueError("t=0 samples can depend on axis-projected P1 nodes")
    return {
        "status": "PASS",
        "axis_node_count": int(axis_nodes.size),
        "axis_incident_cell_count": int(np.count_nonzero(incident)),
        "axis_incident_cell_maximum_r_m": maximum_incident_radius,
        "minimum_t0_sample_r_m": minimum_t0_radius,
        "minimum_all_native_path_sample_r_m": float(np.min(radius)),
        "claim_scope": "t0 localization; saved output rows are also disjoint",
    }


def _reference_fields(
    values: dict[str, FloatArray], electron_voltage: float
) -> dict[str, FloatArray]:
    result: dict[str, FloatArray] = {}
    for spec in _FIELD_SPECS:
        if spec.native_columns is None:
            result[spec.name] = np.full((values["time_s"].size, 1), electron_voltage)
        else:
            result[spec.name] = np.column_stack([values[name] for name in spec.native_columns])
    return result


def _sample_candidate(data: Any, position: FloatArray) -> Any:
    fields = {field.name: field for field in data.fields}
    if set(fields) != {spec.name for spec in _FIELD_SPECS}:
        raise ValueError("candidate field set differs from the localization contract")
    requirements = {
        field.name: RequiredFieldMetadata(
            field.unit,
            field.components,
            field.stored_basis,
            positive=False,
        )
        for field in data.fields
    }
    sampled = prepare_required_fields(data, requirements).sample(position)
    if not bool(sampled.support_inside.all()):
        raise ValueError("canonical P1 fields do not support every native saved state")
    return sampled


def _field_metrics(
    sampled: Mapping[str, FloatArray],
    reference: dict[str, FloatArray],
    particle: FloatArray,
    time: FloatArray,
) -> list[Record]:
    rows: list[Record] = []
    scopes = (
        ("identical_t0_state", time == 0.0),
        ("saved_native_path_states", np.ones(time.size, bool)),
    )
    for scope, mask in scopes:
        for spec in _FIELD_SPECS:
            metric = _metric(
                sampled[spec.name][mask], reference[spec.name][mask], particle[mask], time[mask]
            )
            rows.append({"scope": scope, "field": spec.name, "unit": spec.unit, **metric})
    return rows


def _p1_values(
    values: dict[str, FloatArray], sampled: Mapping[str, FloatArray]
) -> dict[str, FloatArray]:
    result = {name: column.copy() for name, column in values.items()}
    for spec in _FIELD_SPECS:
        if spec.native_columns is None:
            continue
        field = sampled[spec.name]
        for component, column in enumerate(spec.native_columns):
            result[column] = field[:, component]
    return result


def _rhs_metrics(
    native_values: dict[str, FloatArray],
    p1_values: dict[str, FloatArray],
    config: Any,
) -> tuple[list[Record], Record]:
    native_charge, native_forces, native_applicability = _production_evaluations(
        native_values, config
    )
    p1_charge, p1_forces, p1_applicability = _production_evaluations(p1_values, config)
    particle = native_values["particle_id"]
    time = native_values["time_s"]
    rows: list[Record] = []
    scopes = (
        ("identical_t0_state", time == 0.0),
        ("saved_native_path_states", np.ones(time.size, bool)),
    )
    for scope, mask in scopes:
        rows.append(
            {
                "scope": scope,
                "quantity": "charge_rate",
                "unit": "e/s",
                **_metric(p1_charge[mask], native_charge[mask], particle[mask], time[mask]),
            }
        )
        rows.extend(
            {
                "scope": scope,
                "quantity": f"{name}_force",
                "unit": "N",
                **_metric(
                    p1_forces[name][mask],
                    native_forces[name][mask],
                    particle[mask],
                    time[mask],
                ),
            }
            for name in native_forces
        )
        native_total = sum(native_forces.values()) / native_values["particle_mass_kg"][:, None]
        p1_total = sum(p1_forces.values()) / p1_values["particle_mass_kg"][:, None]
        rows.append(
            {
                "scope": scope,
                "quantity": "total_acceleration",
                "unit": "m/s^2",
                **_metric(p1_total[mask], native_total[mask], particle[mask], time[mask]),
            }
        )
    return rows, {
        "native": native_applicability,
        "canonical_p1": p1_applicability,
        "interpretation": "saved-row applicability only; no continuous-path certification",
    }


def _gate(
    name: str, observed: object, operator: str, limit: object, passed: bool, note: str
) -> Record:
    return {
        "gate": name,
        "observed": observed,
        "operator": operator,
        "limit": limit,
        "status": "PASS" if passed else "FAIL",
        "interpretation": note,
    }


def evaluate(solver_root: Path, config_path: Path) -> Record:
    """Evaluate the locked localization slice without writing artifacts."""

    root = solver_root.expanduser().resolve()
    protocol, locks = _load_protocol(root, config_path.expanduser().resolve())
    scope = _mapping(protocol["scope"], "scope")
    prior = _verify_prior_decisions(locks)
    frozen = load_frozen_config(locks["frozen_rhs_configuration"].path)
    values = _load_records(locks["native_step_directory"].path, frozen)
    cohort = _validate_native_cohort(values, scope)
    values, ppr_particle = _authoritative_ppr_heat(
        values, locks["ppr_particle_table"].path, int(scope["records"])
    )
    data, info = read_with_info(locks["candidate_input"].path)
    candidate_lock = _mapping(
        _mapping(protocol["artifacts"], "artifacts")["candidate_input"], "candidate_input"
    )
    if info.content_hash != candidate_lock["content_hash"]:
        raise ValueError("candidate canonical content hash differs from the lock")
    if data.coordinate_system != "axisymmetric_rz" or len(data.layouts) != 1:
        raise ValueError("localization requires one axisymmetric RZ layout")
    ppr_nodes = _bind_ppr_nodes(data, locks["ppr_domain3_table"].path)
    position = np.column_stack((values["r_m"], values["z_m"]))
    sampled = _sample_candidate(data, position)
    t0 = values["time_s"] == 0.0
    axis = _axis_exclusion(data, values["r_m"], t0)
    reference = _reference_fields(values, frozen.electron_thermal_voltage_V)
    field_rows = _field_metrics(sampled.values, reference, values["particle_id"], values["time_s"])
    candidate_config = _json(locks["candidate_configuration"].path)
    parameters = _mapping(candidate_config.get("parameters"), "candidate parameters")
    execution_config = replace(
        frozen,
        maximum_relative_ion_speed_m_s=float(parameters["maximum_relative_ion_speed_m_s"]),
        maximum_effective_gas_speed_ratio=float(parameters["maximum_neutral_speed_ratio"]),
    )
    p1_values = _p1_values(values, sampled.values)
    rhs_rows, applicability = _rhs_metrics(values, p1_values, execution_config)
    t0_fields = [row for row in field_rows if row["scope"] == "identical_t0_state"]
    t0_rhs = {row["quantity"]: row for row in rhs_rows if row["scope"] == "identical_t0_state"}
    field_differences = sum(
        row["classification"] == "DIFFERENT_ABOVE_ROUNDOFF" for row in t0_fields
    )
    charge_differs = t0_rhs["charge_rate"]["classification"] == "DIFFERENT_ABOVE_ROUNDOFF"
    acceleration_differs = (
        t0_rhs["total_acceleration"]["classification"] == "DIFFERENT_ABOVE_ROUNDOFF"
    )
    localized = field_differences > 0 and charge_differs and acceleration_differs
    gates = [
        _gate(
            "locked_prior_decisions",
            "4_BOUND",
            "==",
            "4_BOUND",
            True,
            "separate historical decisions",
        ),
        _gate(
            "native_saved_state_cohort",
            cohort["records"],
            "==",
            scope["records"],
            True,
            "complete active grid",
        ),
        _gate(
            "ppr_particle_binding",
            ppr_particle["records"],
            "==",
            scope["records"],
            True,
            "same keys and coordinates",
        ),
        _gate(
            "ppr_node_heat_flux_relative_l2",
            ppr_nodes["after_axis_policy_relative_l2"],
            "<=",
            _EXPECTED_ROUNDOFF_LIMIT,
            True,
            "coordinate-matched PPR node values after registered axis policy",
        ),
        _gate(
            "canonical_p1_support",
            float(np.mean(sampled.support_inside)),
            "==",
            1.0,
            True,
            "all native saved states",
        ),
        _gate(
            "t0_axis_projection_excluded",
            axis["minimum_t0_sample_r_m"],
            ">",
            axis["axis_incident_cell_maximum_r_m"],
            True,
            "t0 samples cannot depend on projected axis nodes",
        ),
        _gate(
            "t0_field_difference_above_roundoff",
            field_differences,
            ">=",
            1,
            field_differences >= 1,
            "classification threshold is a pre-existing roundoff limit, not accuracy tolerance",
        ),
        _gate(
            "t0_charge_rate_propagation",
            t0_rhs["charge_rate"]["relative_l2"],
            ">",
            _EXPECTED_ROUNDOFF_LIMIT,
            charge_differs,
            "same state and production formula; primitives differ only by representation",
        ),
        _gate(
            "t0_acceleration_propagation",
            t0_rhs["total_acceleration"]["relative_l2"],
            ">",
            _EXPECTED_ROUNDOFF_LIMIT,
            acceleration_differs,
            "same state and production formula; primitives differ only by representation",
        ),
    ]
    return {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "evaluation_id": protocol["evaluation_id"],
        "classification": protocol["classification"],
        "overall_status": "PASS_LOCALIZED" if localized else "INCONCLUSIVE",
        "first_difference_layer": (
            "FIELD_REPRESENTATION_OR_SAMPLING" if localized else "NOT_LOCALIZED"
        ),
        "comsol_rerun_performed": False,
        "source_artifacts": {
            name: {"path": str(lock.path), "sha256": lock.sha256} for name, lock in locks.items()
        },
        "prior_decisions": prior,
        "native_cohort": cohort,
        "ppr_particle_binding": ppr_particle,
        "ppr_node_binding": ppr_nodes,
        "axis_projection_exclusion": axis,
        "field_metrics": field_rows,
        "rhs_metrics": rhs_rows,
        "saved_row_applicability": applicability,
        "gates": gates,
        "decision_basis": {
            "roundoff_relative_l2": _EXPECTED_ROUNDOFF_LIMIT,
            "basis": "pre-existing M3-V layer-diagnostic roundoff classification",
            "result_dependent_tolerance_tuning": "PROHIBITED",
        },
        "claims": {
            "earliest_observed_difference": (
                "localized at identical t=0 state before trajectory divergence"
            ),
            "same_field_solver_agreement": "UNCHANGED_SEPARATE_PASS",
            "cross_representation_trajectory": "UNCHANGED_SEPARATE_FAIL",
            "native_fe_topology_or_interpolant_reconstruction": "NOT_CLAIMED",
            "integrated_per_field_causal_attribution": "NOT_TESTED",
            "physical_model_validity": "NOT_CLAIMED",
            "universal_comsol_equivalent_accuracy": "NOT_CLAIMED",
            "comsol_is_golden_truth": False,
        },
    }


def _write_csv(path: Path, rows: list[Record], fieldnames: tuple[str, ...]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row[name] for name in fieldnames} for row in rows)


def write_outputs(report: Record, output: Path) -> None:
    """Write one no-clobber localization result."""

    resolved = output.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    metric_columns = (
        "scope",
        "field",
        "unit",
        "count",
        "components",
        "relative_l2",
        "rms_vector_or_scalar",
        "maximum_vector_or_scalar",
        "worst_particle_id",
        "worst_time_s",
        "classification",
    )
    rhs_columns = tuple("quantity" if name == "field" else name for name in metric_columns)
    gate_columns = ("gate", "observed", "operator", "limit", "status", "interpretation")
    field_path = resolved / FIELD_METRICS_FILE
    rhs_path = resolved / RHS_METRICS_FILE
    gates_path = resolved / GATES_FILE
    _write_csv(field_path, report["field_metrics"], metric_columns)
    _write_csv(rhs_path, report["rhs_metrics"], rhs_columns)
    _write_csv(gates_path, report["gates"], gate_columns)
    report["output_artifacts"] = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in (field_path, rhs_path, gates_path)
    }
    (resolved / REPORT_FILE).write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver_root", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.solver_root, args.config)
    write_outputs(report, args.output)
    print(json.dumps({"status": report["overall_status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
