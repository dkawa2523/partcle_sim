"""Prepare and run the deterministic M3-C3 Case-P three-current campaign.

The tool is deliberately external to the solver core.  It owns the one-time
three-current equilibrium release charge shared by the solver candidate and
COMSOL, then executes ordinary candidate cases through the three public APIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case import CASE_FORMAT_VERSION
from chamber_particles.case_format import (
    DataBundle,
    RealizedTableSource,
    read_with_info,
    write,
)
from chamber_particles.fields import RequiredFieldMetadata, prepare_required_fields
from chamber_particles.physics.charge import (
    AggregateThreeCurrentChargeEvaluation,
    aggregate_relative_drift_regularized_three_current_v1,
    aggregate_relative_drift_three_current_global_bounds,
)
from chamber_particles.physics.forces import BOLTZMANN_J_K
from chamber_particles.yaml_input import parse_document
from tools.vv.comsol.actual_run_receipt import write_boundary_meaning
from tools.vv.comsol.meaning_preflight import load_inventory

TOOL_REVISION: Final = "m3c3_casep_three_current_candidate_v5"
CASE_ID: Final = "caseP_100nm_three_current"
CHARGE_REVISION: Final = "aggregate_relative_drift_regularized_three_current_v1"
ION_DRAG_REVISION: Final = "relative_flow_screened_collection_orbital_aggregate_ion_v1"
MAXIMUM_RELATIVE_ION_SPEED_M_S: Final = 1_000_000.0
PARTICLE_SPEED_ENVELOPE_M_S: Final = 1_000.0
EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO: Final = 1.0
GAS_MOLECULAR_MASS_KG: Final = 1.2753471408396638e-25
PARTICLE_COUNT: Final = 287
OUTPUT_COUNT: Final = 121
TIME_END_S: Final = 0.03
EVENT_GEOMETRY_RTOL: Final = 1.0e-9
LEVELS: Final = (
    ("dt_2p5us", 2.5e-6),
    ("dt_1p25us", 1.25e-6),
    ("dt_0p625us", 6.25e-7),
)
TRAJECTORY_HEADER: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
)
EVENT_HEADER: Final = (
    "particle_id",
    "event_time_s",
    "event_type",
    "outcome",
    "boundary_semantic",
)
TERMINAL_OUTCOMES: Final = frozenset({"held", "stuck", "escaped"})
EVENT_PROJECTION_REVISION: Final = "canonical_boundary_id_group_v1"
RELEASE_HEADER: Final = (
    "particle_id",
    "release_time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
)
LIFECYCLE: Final = {
    0: "pending",
    1: "active",
    2: "stuck",
    3: "escaped",
    4: "failed",
    5: "held",
}
FIELD_REQUIREMENTS: Final = {
    "electron_number_density": RequiredFieldMetadata("1/m^3", ("value",), "scalar", True),
    "positive_ion_number_density": RequiredFieldMetadata("1/m^3", ("value",), "scalar", True),
    "negative_ion_number_density": RequiredFieldMetadata("1/m^3", ("value",), "scalar", True),
    "electron_thermal_voltage": RequiredFieldMetadata("V", ("value",), "scalar", True),
    "positive_ion_thermal_voltage": RequiredFieldMetadata("V", ("value",), "scalar", True),
    "negative_ion_thermal_voltage": RequiredFieldMetadata("V", ("value",), "scalar", True),
    "positive_ion_velocity": RequiredFieldMetadata("m/s", ("r", "z"), "axisymmetric_rz", False),
    "negative_ion_velocity": RequiredFieldMetadata("m/s", ("r", "z"), "axisymmetric_rz", False),
    "effective_positive_ion_mass": RequiredFieldMetadata("kg", ("value",), "scalar", True),
    "effective_negative_ion_mass": RequiredFieldMetadata("kg", ("value",), "scalar", True),
    "screening_length": RequiredFieldMetadata("m", ("value",), "scalar", True),
    "gas_velocity": RequiredFieldMetadata("m/s", ("r", "z"), "axisymmetric_rz", False),
    "gas_temperature": RequiredFieldMetadata("K", ("value",), "scalar", True),
}


def _solver_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


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


def _locked_path(root: Path, record: object, key: str) -> Path:
    item = _mapping(record, key)
    path_key = "repository_relative_path" if root == _repository_root() else "solver_relative_path"
    allowed = {path_key, "sha256"}
    if key == "primitive_input":
        allowed.add("content_hash")
    if set(item) != allowed:
        raise ValueError(f"{key} has unsupported keys")
    relative = Path(str(item[path_key]))
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError(f"{key} path must remain below its declared root")
    if not path.is_file() or _sha256(path) != str(item["sha256"]):
        raise ValueError(f"{key} file is missing or differs from its SHA-256 lock")
    return path


def _load_configuration(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    required = {
        "schema_version",
        "campaign_id",
        "campaign_revision",
        "classification",
        "primitive_input",
        "primitive_receipt",
        "candidate_template",
        "source_mph",
        "schedule",
        "numerics",
        "physics",
        "acceptance",
        "claim_policy",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("unsupported M3-C3 configuration schema")
    if (
        config["campaign_revision"] != 5
        or config["campaign_id"] != "M3-C3-caseP-100nm-three-current"
    ):
        raise ValueError("unsupported M3-C3 campaign identity")
    paths = {
        "primitive_input": _locked_path(
            _solver_root(), config["primitive_input"], "primitive_input"
        ),
        "primitive_receipt": _locked_path(
            _solver_root(), config["primitive_receipt"], "primitive_receipt"
        ),
        "candidate_template": _locked_path(
            _solver_root(), config["candidate_template"], "candidate_template"
        ),
        "source_mph": _locked_path(_repository_root(), config["source_mph"], "source_mph"),
    }
    _validate_campaign_values(config)
    return config, paths


def _validate_campaign_values(config: Mapping[str, object]) -> None:
    schedule = _mapping(config["schedule"], "schedule")
    expected_schedule = {
        "particle_count": PARTICLE_COUNT,
        "time_end_s": TIME_END_S,
        "output_count": OUTPUT_COUNT,
        "candidate_steps_s": [value for _, value in LEVELS],
        "comsol_reference_steps_s": [5.0e-6, 2.5e-6, 1.25e-6],
    }
    numerics = _mapping(config["numerics"], "numerics")
    expected_numerics = {
        "event_geometry_rtol": EVENT_GEOMETRY_RTOL,
        "event_roundoff_ulps": 64,
        "event_tolerance_policy": (
            "scale_aware_and_materially_below_the_trajectory_acceptance_floor"
        ),
    }
    physics = _mapping(config["physics"], "physics")
    expected_physics = {
        "brownian_active": False,
        "charge_revision": CHARGE_REVISION,
        "ion_drag_revision": ION_DRAG_REVISION,
        "candidate_integrator": "exponential_midpoint",
        "effective_gas_maximum_speed_ratio": EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO,
        "effective_gas_classification": (
            "producer_effective_gas_sensitivity_not_physical_certification"
        ),
        "maximum_relative_ion_speed_m_s": MAXIMUM_RELATIVE_ION_SPEED_M_S,
        "particle_speed_envelope_m_s": PARTICLE_SPEED_ENVELOPE_M_S,
    }
    if (
        schedule != expected_schedule
        or numerics != expected_numerics
        or physics != expected_physics
    ):
        raise ValueError(
            "M3-C3 schedule, numerics, or physics selection differs from the fixed campaign"
        )


def _scalar_range(prepared: Any, name: str) -> tuple[float, float]:
    lower, upper = prepared.component_bounds(name)
    return float(lower[0]), float(upper[0])


def _speed_envelope(prepared: Any, name: str) -> dict[str, object]:
    lower, upper = prepared.component_bounds(name)
    component_abs = np.maximum(np.abs(lower), np.abs(upper))
    ion_box_bound = float(np.hypot(component_abs[0], component_abs[1]))
    relative_bound = math.nextafter(ion_box_bound + PARTICLE_SPEED_ENVELOPE_M_S, math.inf)
    if relative_bound >= MAXIMUM_RELATIVE_ION_SPEED_M_S:
        raise ValueError(f"{name} component-box relative-speed bound reaches 1.0e6 m/s")
    return {
        "component_lower_m_s": lower.tolist(),
        "component_upper_m_s": upper.tolist(),
        "ion_component_box_speed_bound_m_s": ion_box_bound,
        "particle_speed_envelope_m_s": PARTICLE_SPEED_ENVELOPE_M_S,
        "relative_speed_bound_m_s": relative_bound,
        "configured_maximum_relative_speed_m_s": MAXIMUM_RELATIVE_ION_SPEED_M_S,
        "clipping": False,
        "passed": True,
    }


def _charge_arguments(
    charge_number: np.ndarray,
    source: Any,
    values: Mapping[str, np.ndarray],
) -> dict[str, object]:
    return {
        "charge_number": charge_number,
        "electrostatic_radius_m": source.electrostatic_radius_m,
        "electron_number_density_m3": values["electron_number_density"][:, 0],
        "positive_ion_number_density_m3": values["positive_ion_number_density"][:, 0],
        "negative_ion_number_density_m3": values["negative_ion_number_density"][:, 0],
        "electron_thermal_voltage_V": values["electron_thermal_voltage"][:, 0],
        "positive_ion_thermal_voltage_V": values["positive_ion_thermal_voltage"][:, 0],
        "negative_ion_thermal_voltage_V": values["negative_ion_thermal_voltage"][:, 0],
        "particle_velocity_m_s": source.velocity_m_s,
        "positive_ion_velocity_m_s": values["positive_ion_velocity"],
        "negative_ion_velocity_m_s": values["negative_ion_velocity"],
        "effective_positive_ion_mass_kg": values["effective_positive_ion_mass"][:, 0],
        "effective_negative_ion_mass_kg": values["effective_negative_ion_mass"][:, 0],
        "screening_length_m": values["screening_length"][:, 0],
        "maximum_relative_ion_speed_m_s": MAXIMUM_RELATIVE_ION_SPEED_M_S,
    }


def _evaluate_charge(
    charge_number: np.ndarray,
    source: Any,
    values: Mapping[str, np.ndarray],
) -> AggregateThreeCurrentChargeEvaluation:
    return aggregate_relative_drift_regularized_three_current_v1(
        **_charge_arguments(charge_number, source, values)  # type: ignore[arg-type]
    )


def _global_charge_bracket(prepared: Any, source: Any) -> tuple[float, float, dict[str, object]]:
    scalar_names = (
        "electron_number_density",
        "positive_ion_number_density",
        "negative_ion_number_density",
        "electron_thermal_voltage",
        "positive_ion_thermal_voltage",
        "negative_ion_thermal_voltage",
        "effective_positive_ion_mass",
        "effective_negative_ion_mass",
        "screening_length",
    )
    ranges = {name: _scalar_range(prepared, name) for name in scalar_names}
    bounds = aggregate_relative_drift_three_current_global_bounds(
        initial_charge_number=source.charge_number,
        electrostatic_radius_m=source.electrostatic_radius_m,
        electron_number_density_lower_m3=ranges["electron_number_density"][0],
        electron_number_density_upper_m3=ranges["electron_number_density"][1],
        positive_ion_number_density_lower_m3=ranges["positive_ion_number_density"][0],
        positive_ion_number_density_upper_m3=ranges["positive_ion_number_density"][1],
        negative_ion_number_density_lower_m3=ranges["negative_ion_number_density"][0],
        negative_ion_number_density_upper_m3=ranges["negative_ion_number_density"][1],
        electron_thermal_voltage_lower_V=ranges["electron_thermal_voltage"][0],
        electron_thermal_voltage_upper_V=ranges["electron_thermal_voltage"][1],
        positive_ion_thermal_voltage_lower_V=ranges["positive_ion_thermal_voltage"][0],
        positive_ion_thermal_voltage_upper_V=ranges["positive_ion_thermal_voltage"][1],
        negative_ion_thermal_voltage_lower_V=ranges["negative_ion_thermal_voltage"][0],
        negative_ion_thermal_voltage_upper_V=ranges["negative_ion_thermal_voltage"][1],
        effective_positive_ion_mass_lower_kg=ranges["effective_positive_ion_mass"][0],
        effective_positive_ion_mass_upper_kg=ranges["effective_positive_ion_mass"][1],
        effective_negative_ion_mass_lower_kg=ranges["effective_negative_ion_mass"][0],
        effective_negative_ion_mass_upper_kg=ranges["effective_negative_ion_mass"][1],
        screening_length_lower_m=ranges["screening_length"][0],
        screening_length_upper_m=ranges["screening_length"][1],
        maximum_relative_ion_speed_m_s=MAXIMUM_RELATIVE_ION_SPEED_M_S,
    )
    receipt: dict[str, object] = {
        "primitive_component_ranges": {name: list(interval) for name, interval in ranges.items()},
        "global_charge_number_bracket": [bounds.charge_number_lower, bounds.charge_number_upper],
        "global_charge_rate_abs_upper_number_s": bounds.charge_rate_abs_upper_number_s,
        "global_charge_rate_derivative_abs_upper_s_inv": (
            bounds.charge_rate_derivative_abs_upper_s_inv
        ),
    }
    return bounds.charge_number_lower, bounds.charge_number_upper, receipt


def _solve_equilibrium(
    lower_scalar: float,
    upper_scalar: float,
    source: Any,
    values: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    lower = np.full(PARTICLE_COUNT, lower_scalar, dtype=np.float64)
    upper = np.full(PARTICLE_COUNT, upper_scalar, dtype=np.float64)
    lower_eval = _evaluate_charge(lower, source, values)
    upper_eval = _evaluate_charge(upper, source, values)
    if not bool(lower_eval.applicable.all() and upper_eval.applicable.all()):
        raise ValueError("global equilibrium bracket violates the speed envelope")
    if bool((lower_eval.charge_rate_number_s < 0.0).any()) or bool(
        (upper_eval.charge_rate_number_s > 0.0).any()
    ):
        raise ValueError("global charge bracket does not contain every local equilibrium")
    iterations = 128
    for _ in range(iterations):
        middle = lower + 0.5 * (upper - lower)
        evaluation = _evaluate_charge(middle, source, values)
        positive = evaluation.charge_rate_number_s > 0.0
        lower = np.where(positive, middle, lower)
        upper = np.where(positive, upper, middle)
    roots = np.ascontiguousarray(lower + 0.5 * (upper - lower), dtype="<f8")
    final = _evaluate_charge(roots, source, values)
    correction = np.abs(final.charge_rate_number_s / final.charge_rate_derivative_s_inv)
    scale = np.maximum(1.0, np.abs(roots))
    converged = bool((correction <= 128.0 * np.finfo(np.float64).eps * scale).all())
    if not converged or not bool(final.applicable.all()):
        raise ValueError("three-current release equilibrium did not converge in float64")
    return roots, {
        "method": "monotone_bisection_float64",
        "iterations": iterations,
        "converged": converged,
        "maximum_final_bracket_width_charge_number": float(np.max(upper - lower)),
        "maximum_abs_rate_residual_number_s": float(np.max(np.abs(final.charge_rate_number_s))),
        "maximum_abs_newton_correction_charge_number": float(np.max(correction)),
        "charge_number_range": [float(np.min(roots)), float(np.max(roots))],
        "all_model_applicable": bool(final.applicable.all()),
    }


def _equilibrium_release(
    data: DataBundle,
) -> tuple[np.ndarray, dict[str, object]]:
    if len(data.sources) != 1:
        raise ValueError("M3-C3 input must contain exactly one particle source")
    source = data.sources[0]
    if not isinstance(source, RealizedTableSource):
        raise ValueError("M3-C3 release probes must be an internal table")
    if source.particle_id.size != PARTICLE_COUNT or not np.array_equal(
        source.particle_id, np.arange(1, PARTICLE_COUNT + 1)
    ):
        raise ValueError("M3-C3 source must contain particle IDs 1..287")
    if bool((source.release_time_s != 0.0).any()):
        raise ValueError("M3-C3 source release times must all be zero")
    particle_speed = np.linalg.norm(source.velocity_m_s, axis=1)
    if float(np.max(particle_speed)) > PARTICLE_SPEED_ENVELOPE_M_S:
        raise ValueError("release velocity exceeds the declared 1 km/s particle envelope")
    prepared = prepare_required_fields(data, FIELD_REQUIREMENTS)
    batch = prepared.sample(source.position_m)
    if not bool(batch.support_inside.all()):
        raise ValueError("one or more release points lie outside primitive field support")
    speed_receipt = {
        "derivation": "canonical nodal component extrema plus 1 km/s particle-speed envelope",
        "positive_ion": _speed_envelope(prepared, "positive_ion_velocity"),
        "negative_ion": _speed_envelope(prepared, "negative_ion_velocity"),
        "release_particle_speed_max_m_s": float(np.max(particle_speed)),
        "runtime_clipping": False,
    }
    gas_velocity = batch.values["gas_velocity"]
    gas_temperature = batch.values["gas_temperature"][:, 0]
    mean_thermal_speed = np.sqrt(
        8.0 * BOLTZMANN_J_K * gas_temperature / (math.pi * GAS_MOLECULAR_MASS_KG)
    )
    initial_gas_ratio = np.linalg.norm(source.velocity_m_s - gas_velocity, axis=1) / (
        mean_thermal_speed
    )
    gas_receipt = {
        "configured_maximum_speed_ratio": EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO,
        "initial_release_maximum_speed_ratio": float(np.max(initial_gas_ratio)),
        "model_revisions": [
            "epstein_linear_effective_gas_sensitivity_v1",
            "waldmann_gallis_free_molecular_effective_gas_heat_flux_sensitivity_v1",
        ],
        "classification": "producer_effective_gas_sensitivity_not_physical_certification",
        "runtime_owner": "continuous model-applicability gate",
        "clipping": False,
    }
    lower, upper, bracket_receipt = _global_charge_bracket(prepared, source)
    roots, solve_receipt = _solve_equilibrium(lower, upper, source, batch.values)
    return roots, {
        "speed_envelope": speed_receipt,
        "effective_gas_speed_ratio": gas_receipt,
        "bracket": bracket_receipt,
        "root_solution": solve_receipt,
    }


def _derived_bundle(data: DataBundle, roots: np.ndarray, parent_hash: str) -> DataBundle:
    source = replace(data.sources[0], charge_number=roots)
    provenance = _mapping(json.loads(data.provenance_json), "canonical provenance")
    provenance["m3c3_release_charge"] = {
        "owner": "external_vv_preparer",
        "revision": CHARGE_REVISION,
        "parent_content_hash": parent_hash,
        "shared_with_comsol": "three_current_release_state.csv",
    }
    return replace(
        data,
        provenance_json=json.dumps(provenance, allow_nan=False, sort_keys=True),
        sources=(source,),
    )


def _write_release(path: Path, source: Any) -> int:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(RELEASE_HEADER)
        for index, particle_id in enumerate(source.particle_id):
            writer.writerow(
                (
                    int(particle_id),
                    format(float(source.release_time_s[index]), ".17g"),
                    format(float(source.position_m[index, 0]), ".17g"),
                    format(float(source.position_m[index, 1]), ".17g"),
                    format(float(source.velocity_m_s[index, 0]), ".17g"),
                    format(float(source.velocity_m_s[index, 1]), ".17g"),
                    format(float(source.charge_number[index]), ".17g"),
                )
            )
    return int(source.particle_id.size)


def _candidate_document(
    template: Mapping[str, object],
    content_identity: str,
    level: str,
    dt_s: float,
) -> dict[str, object]:
    document = dict(template)
    case = _mapping(document["case"], "template.case")
    case.update(
        {
            "name": f"m3c3_caseP_100nm_three_current_{level}",
            "data_path": "../../candidate_input_three_current_z0.h5",
            "expected_content_hash": content_identity,
        }
    )
    time = _mapping(document["time"], "template.time")
    time["dt_s"] = dt_s
    solver = _mapping(document["solver"], "template.solver")
    solver["integrator"] = "exponential_midpoint"
    event = _mapping(solver["event"], "template.solver.event")
    event["geometry_rtol"] = EVENT_GEOMETRY_RTOL
    solver["event"] = event
    physics = _mapping(document["physics"], "template.physics")
    charge = _mapping(physics["charge"], "template.physics.charge")
    charge.update(
        {
            "revision": CHARGE_REVISION,
            "negative_ion_number_density_field": "negative_ion_number_density",
            "negative_ion_thermal_voltage_field": "negative_ion_thermal_voltage",
            "negative_ion_velocity_field": "negative_ion_velocity",
            "effective_negative_ion_mass_field": "effective_negative_ion_mass",
            "maximum_relative_ion_speed_m_s": MAXIMUM_RELATIVE_ION_SPEED_M_S,
        }
    )
    ion_drag = _mapping(physics["ion_drag"], "template.physics.ion_drag")
    ion_drag["maximum_relative_ion_speed_m_s"] = MAXIMUM_RELATIVE_ION_SPEED_M_S
    drag = _mapping(physics["drag"], "template.physics.drag")
    drag["maximum_speed_ratio"] = EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO
    thermophoresis = _mapping(physics["thermophoresis"], "template.physics.thermophoresis")
    thermophoresis["maximum_speed_ratio"] = EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO
    physics["charge"] = charge
    physics["ion_drag"] = ion_drag
    physics["drag"] = drag
    physics["thermophoresis"] = thermophoresis
    document["case"] = case
    document["time"] = time
    document["solver"] = solver
    document["physics"] = physics
    return document


def _reference_config(
    config: Mapping[str, object],
    derived_sha256: str,
    derived_content_hash: str,
    release_sha256: str,
    release_receipt_sha256: str,
) -> dict[str, object]:
    source_record = _mapping(config["source_mph"], "source_mph")
    primitive_record = _mapping(config["primitive_input"], "primitive_input")
    receipt_record = _mapping(config["primitive_receipt"], "primitive_receipt")
    return {
        "schema_version": 1,
        "case_id": CASE_ID,
        "source_mph": {
            "path": source_record["repository_relative_path"],
            "sha256": source_record["sha256"],
        },
        "canonical_input": {
            "path": "candidate_input_three_current_z0.h5",
            "file_sha256": derived_sha256,
            "content_hash": derived_content_hash,
            "parent_file_sha256": primitive_record["sha256"],
        },
        "primitive_receipt": {
            "path": "primitive_receipt.json",
            "sha256": receipt_record["sha256"],
        },
        "release_state": {
            "path": "three_current_release_state.csv",
            "sha256": release_sha256,
        },
        "release_receipt": {
            "path": "three_current_release_receipt.json",
            "sha256": release_receipt_sha256,
        },
        "charge_revision": CHARGE_REVISION,
        "ion_drag_revision": ION_DRAG_REVISION,
        "maximum_relative_ion_speed_m_s": MAXIMUM_RELATIVE_ION_SPEED_M_S,
        "comsol": {
            "fixed_rk4_step_s": 1.0e-5,
            "time_end_s": TIME_END_S,
            "output_count": OUTPUT_COUNT,
            "particle_count": PARTICLE_COUNT,
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare(config_path: Path, output: Path) -> dict[str, object]:
    """Prepare the shared Z0 authority and three deterministic candidate levels."""

    config_path = config_path.resolve()
    config, paths = _load_configuration(config_path)
    data, input_info = read_with_info(paths["primitive_input"])
    expected_content_hash = _mapping(config["primitive_input"], "primitive_input")["content_hash"]
    if input_info.content_hash != expected_content_hash:
        raise ValueError("primitive input logical content hash differs")
    roots, numerical_receipt = _equilibrium_release(data)
    derived = _derived_bundle(data, roots, input_info.content_hash)
    template = _mapping(
        parse_document(paths["candidate_template"].read_bytes()),
        "candidate template",
    )
    if template.get("format_version") != CASE_FORMAT_VERSION:
        raise ValueError("candidate template must use the current case format")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(paths["primitive_receipt"], output / "primitive_receipt.json")
    input_path = output / "candidate_input_three_current_z0.h5"
    input_written = write(input_path, derived)
    release_path = output / "three_current_release_state.csv"
    release_rows = _write_release(release_path, derived.sources[0])
    release_receipt: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "PASS",
        "case_id": CASE_ID,
        "formula_revision": CHARGE_REVISION,
        "primitive_input": {
            "path": str(paths["primitive_input"]),
            "file_sha256": _sha256(paths["primitive_input"]),
            "content_hash": input_info.content_hash,
        },
        "derived_input": {
            "path": input_path.name,
            "file_sha256": _sha256(input_path),
            "content_hash": input_written.content_hash,
        },
        "release_state": {
            "path": release_path.name,
            "sha256": _sha256(release_path),
            "rows": release_rows,
        },
        "single_charge_authority": True,
        "comsol_recomputes_equilibrium": False,
        **numerical_receipt,
    }
    release_receipt_path = output / "three_current_release_receipt.json"
    _write_json(release_receipt_path, release_receipt)
    candidate_root = output / "candidate"
    cases: dict[str, object] = {}
    for level, dt_s in LEVELS:
        case_root = candidate_root / level
        case_root.mkdir(parents=True)
        case_path = case_root / "case.yaml"
        document = _candidate_document(template, input_written.content_hash, level, dt_s)
        case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        load_case(case_path)
        cases[level] = {
            "path": str(case_path.relative_to(output)),
            "sha256": _sha256(case_path),
            "dt_s": dt_s,
        }
    reference = _reference_config(
        config,
        _sha256(input_path),
        input_written.content_hash,
        _sha256(release_path),
        _sha256(release_receipt_path),
    )
    _write_json(output / "reference_run_config.json", reference)
    shutil.copyfile(config_path, output / "campaign_config.json")
    report: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "PREPARED",
        "case_id": CASE_ID,
        "configuration_sha256": _sha256(config_path),
        "primitive_receipt_sha256": _sha256(paths["primitive_receipt"]),
        "derived_input_sha256": _sha256(input_path),
        "derived_input_content_hash": input_written.content_hash,
        "release_state_sha256": _sha256(release_path),
        "release_receipt_sha256": _sha256(release_receipt_path),
        "candidate_cases": cases,
        "reference_config_sha256": _sha256(output / "reference_run_config.json"),
        "charge_number_content_identity": hashlib.sha256(roots.tobytes()).hexdigest(),
    }
    _write_json(output / "prepare_report.json", report)
    return report


def _write_trajectory(path: Path, result: Any) -> tuple[int, float]:
    rows = 0
    maximum_speed_m_s = 0.0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(TRAJECTORY_HEADER)
        for frame in result.iter_frames():
            for index, particle_id in enumerate(frame.particle_id):
                velocity = frame.velocity_m_s[index]
                if bool(np.isfinite(velocity).all()):
                    maximum_speed_m_s = max(
                        maximum_speed_m_s,
                        float(np.hypot(velocity[0], velocity[1])),
                    )
                writer.writerow(
                    (
                        int(particle_id),
                        format(float(frame.time_s), ".17g"),
                        format(float(frame.position_m[index, 0]), ".17g"),
                        format(float(frame.position_m[index, 1]), ".17g"),
                        format(float(frame.velocity_m_s[index, 0]), ".17g"),
                        format(float(frame.velocity_m_s[index, 1]), ".17g"),
                        format(float(frame.charge_number[index]), ".17g"),
                        LIFECYCLE[int(frame.lifecycle[index])],
                    )
                )
                rows += 1
    return rows, maximum_speed_m_s


def _write_events(path: Path, result: Any, boundary_meaning: Path) -> int:
    groups = _mapping(load_inventory(boundary_meaning).get("boundary_groups"), "boundary groups")
    events = result.read_boundary_events()
    rows: list[tuple[int, float, str, str, str]] = []
    seen: set[int] = set()
    for index, particle_id_value in enumerate(events.particle_id):
        particle_id = int(particle_id_value)
        outcome = str(events.outcome[index])
        if particle_id in seen or outcome not in TERMINAL_OUTCOMES:
            raise ValueError("M3-C3 requires at most one recognized terminal event per particle")
        seen.add(particle_id)
        boundary_id = str(int(events.boundary_id[index]))
        group = groups.get(boundary_id)
        if not isinstance(group, str) or not group:
            raise ValueError("terminal boundary ID has no canonical semantic group")
        rows.append(
            (
                particle_id,
                float(events.time_s[index]),
                "terminal_boundary",
                outcome,
                group,
            )
        )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(EVENT_HEADER)
        writer.writerows(sorted(rows))
    return len(rows)


def run_cell(prepared_root: Path, level: str) -> dict[str, object]:
    """Run one prepared level through load_case, simulate, and open_result."""

    levels = dict(LEVELS)
    if level not in levels:
        raise ValueError(f"unknown M3-C3 candidate level: {level}")
    prepared_root = prepared_root.resolve()
    report = _mapping(
        json.loads((prepared_root / "prepare_report.json").read_text(encoding="utf-8")),
        "prepare report",
    )
    if report.get("status") != "PREPARED" or report.get("tool_revision") != TOOL_REVISION:
        raise ValueError("prepared root does not belong to this tool revision")
    cell = prepared_root / "candidate" / level
    result_path = cell / "result"
    trajectory_path = cell / "trajectory.csv"
    events_path = cell / "events.csv"
    receipt_path = cell / "run_receipt.json"
    if any(path.exists() for path in (result_path, trajectory_path, events_path, receipt_path)):
        raise FileExistsError(f"candidate output already exists for {level}")
    case_path = cell / "case.yaml"
    simulate(load_case(case_path), result_path)
    result = open_result(result_path)
    rows, maximum_speed_m_s = _write_trajectory(trajectory_path, result)
    meaning_path = write_boundary_meaning(
        prepared_root / "candidate_input_three_current_z0.h5", cell
    )
    event_rows = _write_events(events_path, result, meaning_path)
    counts = _mapping(result.manifest.get("counts"), "result counts")
    status = (
        "PASS"
        if (
            rows == PARTICLE_COUNT * OUTPUT_COUNT
            and event_rows == counts.get("boundary_events")
            and counts.get("failure_events") == 0
        )
        else "BLOCKED"
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": status,
        "case_id": CASE_ID,
        "level": level,
        "dt_s": levels[level],
        "public_api_path": ["load_case", "simulate", "open_result"],
        "case_sha256": _sha256(case_path),
        "derived_input_sha256": report["derived_input_sha256"],
        "release_state_sha256": report["release_state_sha256"],
        "trajectory_sha256": _sha256(trajectory_path),
        "trajectory_rows": rows,
        "events_sha256": _sha256(events_path),
        "event_rows": event_rows,
        "event_projection_revision": EVENT_PROJECTION_REVISION,
        "canonical_boundary_meaning_sha256": _sha256(meaning_path),
        "maximum_observed_particle_speed_m_s": maximum_speed_m_s,
        "preparation_particle_speed_envelope_m_s": PARTICLE_SPEED_ENVELOPE_M_S,
        "preparation_envelope_exceeded": maximum_speed_m_s > PARTICLE_SPEED_ENVELOPE_M_S,
        "runtime_applicability_owner": "physics relative-ion-speed gate at 1.0e6 m/s",
        "result_manifest_sha256": _sha256(result_path / "run.json"),
        "result_counts": counts,
    }
    _write_json(receipt_path, receipt)
    if status != "PASS":
        raise RuntimeError(f"M3-C3 candidate {level} did not complete without failures")
    return receipt


def reproject_events(prepared_root: Path, level: str) -> dict[str, object]:
    """Correct event reporting from an immutable completed result without simulating."""
    if level not in dict(LEVELS):
        raise ValueError(f"unknown M3-C3 candidate level: {level}")
    prepared_root = prepared_root.resolve()
    cell = prepared_root / "candidate" / level
    receipt_path = cell / "run_receipt.json"
    receipt = load_inventory(receipt_path)
    result_path = cell / "result"
    old_events = cell / "events.csv"
    if (
        receipt.get("status") != "PASS"
        or receipt.get("result_manifest_sha256") != _sha256(result_path / "run.json")
        or receipt.get("events_sha256") != _sha256(old_events)
        or receipt.get("case_sha256") != _sha256(cell / "case.yaml")
    ):
        raise ValueError("immutable executed result or original projection identity differs")
    candidate_input = prepared_root / "candidate_input_three_current_z0.h5"
    if receipt.get("derived_input_sha256") != _sha256(candidate_input):
        raise ValueError("event projection must use the executed canonical input")
    meaning_path = write_boundary_meaning(candidate_input, cell)
    projected = cell / "events.canonical.csv"
    result = open_result(result_path)
    count = _write_events(projected, result, meaning_path)
    if count != receipt.get("event_rows"):
        raise ValueError("corrected event population differs from the executed result")
    producer_path = cell / "event_projection_producer.py"
    shutil.copyfile(Path(__file__), producer_path)
    repair: dict[str, object] = {
        "schema_version": 1,
        "projection_revision": EVENT_PROJECTION_REVISION,
        "status": "COMPLETE_REPORTING_REPROJECTION_NO_SIMULATION",
        "level": level,
        "original_run_receipt_sha256": _sha256(receipt_path),
        "original_events_sha256": _sha256(old_events),
        "result_manifest_sha256": _sha256(result_path / "run.json"),
        "canonical_input_sha256": _sha256(candidate_input),
        "canonical_boundary_meaning_sha256": _sha256(meaning_path),
        "producer_sha256": _sha256(producer_path),
        "events": {"path": projected.name, "sha256": _sha256(projected), "rows": count},
        "change": "preserve observed boundary ID and canonical group; remove outcome-to-group inference",
        "comsol_executed": False,
        "candidate_simulated": False,
        "original_execution_receipt_modified": False,
    }
    _write_json(cell / "event_projection_receipt.json", repair)
    return repair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("output", type=Path)
    run_parser = commands.add_parser("run-cell")
    run_parser.add_argument("prepared_root", type=Path)
    run_parser.add_argument("level", choices=[name for name, _ in LEVELS])
    projection_parser = commands.add_parser("reproject-events")
    projection_parser.add_argument("prepared_root", type=Path)
    projection_parser.add_argument("level", choices=[name for name, _ in LEVELS])
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.config, arguments.output)
    elif arguments.command == "run-cell":
        run_cell(arguments.prepared_root, arguments.level)
    else:
        reproject_events(arguments.prepared_root, arguments.level)


if __name__ == "__main__":
    main()
