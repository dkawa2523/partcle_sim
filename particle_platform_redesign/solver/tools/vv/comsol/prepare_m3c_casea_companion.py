"""Prepare the three-cell Case-A size/ion-drag companion campaign.

This is an external V&V adapter.  Geometry and primitive P1 fields are copied
from one canonical authority, while each cell receives only its audited t=0
particle release.  Particle-derived columns in the source CSVs are never
promoted to fields or copied into the canonical input.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case import CASE_FORMAT_VERSION
from chamber_particles.case_format import (
    DataBundle,
    RealizedTableSource,
    content_hash,
    read_with_info,
    write,
)
from chamber_particles.yaml_input import parse_document
from tools.vv.comsol.prepare_m3c1_common_p1_tables import COMPONENT_EXPORTS
from tools.vv.comsol.prepare_m3c1_common_p1_tables import prepare as prepare_p1

TOOL_REVISION: Final = "m3c_casea_size_iondrag_companion_v2"
DATA_PRODUCER_REVISION: Final = "m3c_casea_size_iondrag_companion_v1"
EXPECTED_CASE_IDS: Final = (
    "caseA_10nm_relative_flow",
    "caseA_30nm_relative_flow",
    "caseA_100nm_image",
)
STEP_ROWS: Final = (
    ("coarse", "dt_0p625us", 6.25e-7),
    ("medium", "dt_0p3125us", 3.125e-7),
    ("fine", "dt_0p15625us", 1.5625e-7),
)
PARTICLE_COUNT: Final = 287
EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO: Final = 1.0
RELEASE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_phi_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "particle_diameter_m",
    "particle_radius_m",
    "particle_mass_kg",
)
RUN_SPEC_KEYS: Final = (
    "case_id",
    "diameter_nm",
    "ion_drag_revision",
    "deterministic_contribution_name",
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


@dataclass(frozen=True, slots=True)
class CampaignCase:
    case_id: str
    diameter_nm: int
    source_model: str
    release_path: Path
    release_sha256: str
    ion_drag_revision: str
    deterministic_contribution_name: str


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _solver_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _relative_path(root: Path, value: object, name: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be repository-relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{name} does not resolve to a file: {relative}")
    return path


def _locked_file(root: Path, raw: object, name: str) -> tuple[Path, str]:
    record = _mapping(raw, name)
    if set(record) != {"relative_path", "sha256"}:
        raise ValueError(f"{name} must contain relative_path and sha256")
    path = _relative_path(root, record["relative_path"], f"{name}.relative_path")
    expected = str(record["sha256"])
    if _sha256(path) != expected:
        raise ValueError(f"{name} SHA-256 differs")
    return path, expected


def _load_configuration(path: Path) -> tuple[dict[str, Any], list[CampaignCase]]:
    config = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    required = {
        "schema_version",
        "campaign_id",
        "campaign_revision",
        "classification",
        "expected_comsol_version",
        "shared_field_input",
        "candidate_template",
        "source_models",
        "schedule",
        "cases",
        "input_policy",
        "claim_policy",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("unsupported campaign configuration schema")
    if config["campaign_revision"] != 1:
        raise ValueError("unsupported campaign revision")
    policy = _mapping(config["input_policy"], "input_policy")
    if policy != {
        "field_authority": "one_shared_canonical_exact_connectivity_p1_input",
        "release_authority": "one_size_specific_t0_table_per_case",
        "background_field_tables": "not_read",
        "derived_particle_background_columns": "ignored_not_copied",
    }:
        raise ValueError("campaign input authority policy differs")
    schedule = _mapping(config["schedule"], "schedule")
    if schedule != {
        "particle_count": PARTICLE_COUNT,
        "time_start_s": 0.0,
        "time_end_s": 4.5e-4,
        "output_interval_s": 1.0e-5,
        "output_times": 46,
        "fixed_rk4_steps_s": [row[2] for row in STEP_ROWS],
    }:
        raise ValueError("campaign schedule differs from the common-P1 pre-event window")
    rows = config["cases"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("campaign must contain exactly three cells")
    root = _repository_root()
    cases: list[CampaignCase] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"cases[{index}]")
        if set(row) != {
            "case_id",
            "diameter_nm",
            "source_model",
            "release_table",
            "ion_drag_revision",
            "deterministic_contribution_name",
        }:
            raise ValueError(f"cases[{index}] has unsupported keys")
        release_path, release_hash = _locked_file(
            root, row["release_table"], f"cases[{index}].release_table"
        )
        cases.append(
            CampaignCase(
                case_id=str(row["case_id"]),
                diameter_nm=int(row["diameter_nm"]),
                source_model=str(row["source_model"]),
                release_path=release_path,
                release_sha256=release_hash,
                ion_drag_revision=str(row["ion_drag_revision"]),
                deterministic_contribution_name=str(row["deterministic_contribution_name"]),
            )
        )
    _validate_case_matrix(cases)
    return config, cases


def _validate_case_matrix(cases: list[CampaignCase]) -> None:
    expected = (
        (
            "caseA_10nm_relative_flow",
            10,
            "theory_common",
            "relative_flow_screened_collection_orbital_aggregate_ion_v1",
            "relative_flow_screened_collection_orbital_ion_drag",
        ),
        (
            "caseA_30nm_relative_flow",
            30,
            "theory_common",
            "relative_flow_screened_collection_orbital_aggregate_ion_v1",
            "relative_flow_screened_collection_orbital_ion_drag",
        ),
        (
            "caseA_100nm_image",
            100,
            "theory_common",
            "electric_field_directed_image_orbital_sensitivity_v1",
            "electric_field_directed_image_orbital_ion_drag",
        ),
    )
    observed = tuple(
        (
            case.case_id,
            case.diameter_nm,
            case.source_model,
            case.ion_drag_revision,
            case.deterministic_contribution_name,
        )
        for case in cases
    )
    if observed != expected:
        raise ValueError("campaign cells differ from the locked three-cell matrix")


def _read_release(case: CampaignCase) -> tuple[RealizedTableSource, dict[str, object]]:
    with case.release_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError(f"{case.release_path}: duplicate CSV columns")
        missing = sorted(set(RELEASE_COLUMNS) - set(header))
        if missing:
            raise ValueError(f"{case.release_path}: missing release columns: {missing}")
        indices = [header.index(name) for name in RELEASE_COLUMNS]
        selected = [[row[index] for index in indices] for row in reader]
    if len(selected) != PARTICLE_COUNT:
        raise ValueError(f"{case.release_path}: expected {PARTICLE_COUNT} release rows")
    values = np.asarray(selected, dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{case.release_path}: nonfinite release value")
    particle_id = values[:, 0].astype("<i8")
    if not np.array_equal(values[:, 0], particle_id) or not np.array_equal(
        particle_id, np.arange(1, PARTICLE_COUNT + 1, dtype="<i8")
    ):
        raise ValueError(f"{case.release_path}: particle IDs must be ordered 1..287")
    diameter = values[:, 8].astype("<f8")
    radius = values[:, 9].astype("<f8")
    mass = values[:, 10].astype("<f8")
    expected_diameter = case.diameter_nm * 1.0e-9
    if not np.allclose(diameter, expected_diameter, rtol=3.0e-15, atol=0.0):
        raise ValueError(f"{case.release_path}: particle diameter differs from run spec")
    if not np.allclose(radius, 0.5 * diameter, rtol=3.0e-15, atol=0.0):
        raise ValueError(f"{case.release_path}: electrostatic radius differs from d/2")
    density = 6.0 * mass / (math.pi * diameter**3)
    if not np.allclose(density, 2200.0, rtol=3.0e-15, atol=0.0):
        raise ValueError(f"{case.release_path}: particle mass is not the 2200 kg/m^3 sphere mass")
    if not bool((values[:, 1] == 0.0).all()) or not bool((values[:, 5] == 0.0).all()):
        raise ValueError(f"{case.release_path}: RZ releases require t=0 and zero phi velocity")
    source = RealizedTableSource(
        name="particles",
        particle_id=particle_id,
        release_time_s=values[:, 1].astype("<f8"),
        position_m=np.ascontiguousarray(values[:, [2, 3]], dtype="<f8"),
        velocity_m_s=np.ascontiguousarray(values[:, [4, 6]], dtype="<f8"),
        charge_number=values[:, 7].astype("<f8"),
        mass_kg=mass,
        drag_diameter_m=diameter,
        contact_radius_m=np.zeros(PARTICLE_COUNT, dtype="<f8"),
        electrostatic_radius_m=radius,
        displaced_volume_m3=(math.pi * diameter**3 / 6.0).astype("<f8"),
        model_weight=np.ones(PARTICLE_COUNT, dtype="<f8"),
        material_id=np.zeros(PARTICLE_COUNT, dtype="<i4"),
    )
    ignored = sorted(set(header) - set(RELEASE_COLUMNS))
    return source, {
        "path": str(case.release_path),
        "sha256": case.release_sha256,
        "columns_read": list(RELEASE_COLUMNS),
        "ignored_column_count": len(ignored),
        "derived_particle_columns_copied": [],
        "particle_count": PARTICLE_COUNT,
        "diameter_m": expected_diameter,
        "mass_kg": float(mass[0]),
    }


def _candidate_bundle(
    base: DataBundle,
    case: CampaignCase,
    shared_field_path: Path,
    shared_field_content_hash: str,
) -> tuple[DataBundle, dict[str, object]]:
    source, release_receipt = _read_release(case)
    base_provenance = _mapping(json.loads(base.provenance_json), "shared field provenance")
    provenance = {
        "producer": "prepare_m3c_casea_companion",
        "producer_version": DATA_PRODUCER_REVISION,
        "source_sha256": f"sha256:{_sha256(shared_field_path)}",
        "field_semantics_revision": str(base_provenance["field_semantics_revision"]),
        "producer_metadata": {
            "case_id": case.case_id,
            "shared_primitive_field_authority": str(shared_field_path),
            "shared_primitive_field_content_hash": shared_field_content_hash,
            "size_specific_release": release_receipt,
            "background_field_tables": "not_read",
        },
    }
    bundle = replace(
        base,
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        sources=(source,),
    )
    content_hash(bundle)
    return bundle, release_receipt


def _ion_drag_document(case: CampaignCase) -> dict[str, object]:
    common: dict[str, object] = {
        "positive_ion_number_density_field": "positive_ion_number_density",
        "positive_ion_thermal_voltage_field": "positive_ion_thermal_voltage",
        "positive_ion_velocity_field": "positive_ion_velocity",
        "effective_positive_ion_mass_field": "effective_positive_ion_mass",
        "screening_length_field": "screening_length",
        "applicability": "error",
    }
    if case.ion_drag_revision == "relative_flow_screened_collection_orbital_aggregate_ion_v1":
        return {
            "model": "screened_collection_orbital",
            "revision": case.ion_drag_revision,
            **common,
            "ion_neutral_mean_free_path_field": "ion_neutral_mean_free_path",
            "maximum_relative_ion_speed_m_s": 30000.0,
        }
    return {
        "model": "image_orbital_sensitivity",
        "revision": case.ion_drag_revision,
        **common,
        "electron_thermal_voltage_field": "electron_thermal_voltage",
        "electric_field": "electric_field",
    }


def _candidate_document(
    template: Mapping[str, Any],
    case: CampaignCase,
    input_path: Path,
    case_directory: Path,
    input_content_hash: str,
    step_label: str,
    dt_s: float,
) -> dict[str, Any]:
    document = copy.deepcopy(dict(template))
    document["case"] = {
        "name": f"m3c_{case.case_id}_{step_label}",
        "data_path": Path(os.path.relpath(input_path, case_directory)).as_posix(),
        "expected_content_hash": input_content_hash,
    }
    document["time"] = {"start_s": 0.0, "end_s": 4.5e-4, "dt_s": dt_s}
    document["physics"]["ion_drag"] = _ion_drag_document(case)
    document["physics"]["drag"]["maximum_speed_ratio"] = EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO
    document["physics"]["thermophoresis"]["maximum_speed_ratio"] = EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO
    document["physics"]["dielectrophoresis"]["maximum_point_dipole_radius_m"] = (
        0.5 * case.diameter_nm * 1.0e-9
    )
    document["output"]["trajectories"]["schedule"] = {
        "explicit_times_s": [index * 1.0e-5 if index < 45 else 4.5e-4 for index in range(46)]
    }
    return document


def _run_spec(case: CampaignCase) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "diameter_nm": str(case.diameter_nm),
        "ion_drag_revision": case.ion_drag_revision,
        "deterministic_contribution_name": case.deterministic_contribution_name,
    }


def _reference_config(
    campaign: Mapping[str, Any],
    case: CampaignCase,
    source_model: Mapping[str, Any],
    input_path: Path,
    input_sha256: str,
    input_content_hash: str,
) -> dict[str, object]:
    contributions = [
        "electric",
        case.deterministic_contribution_name,
        "epstein_drag",
        "waldmann_heat_flux_thermophoresis",
        "free_molecular_lift_sensitivity",
        "dielectrophoresis",
        "gravity_buoyancy",
    ]
    return {
        "schema_version": 1,
        "evaluation_id": f"M3-C1-{case.case_id}-common-P1-reference-run",
        "evaluation_revision": 1,
        "classification": "external_comsol_full_physics_common_field_diagnostic",
        "expected_comsol_version": campaign["expected_comsol_version"],
        "source_model": {
            **source_model,
            "load_mode": "ModelUtil.loadCopy",
            "saved": False,
        },
        "candidate_input": {
            "relative_path": Path(os.path.relpath(input_path, _solver_root())).as_posix(),
            "sha256": input_sha256,
            "content_hash": input_content_hash,
        },
        "case": {
            "workflow": "caseA",
            "diameter_m": case.diameter_nm * 1.0e-9,
            **campaign["schedule"],
        },
        "numerics": {
            "integrator": "classical_rk4",
            "integrator_order": 4,
            "relative_tolerance": 1.0e-8,
            "wall_accuracy_order": 1,
            "store_particle_status": True,
            "store_extra": False,
        },
        "physics": {
            "brownian_active": False,
            "saffman_active": False,
            "dynamic_charge_active": True,
            "epstein_delta": 1.3534291735288517,
            "neutral_molecular_mass_kg": 1.2753471408396638e-25,
            "lift_coefficient": 1.0,
            "medium_relative_permittivity": 1.0,
            "real_clausius_mossotti_factor": 0.5161290322580645,
            "deterministic_contributions": contributions,
        },
        "field_contract": {
            "representation": "canonical_exact_connectivity_P1_sectionwise",
            "native_caseA_field_usage_in_particle_rhs": False,
            "release_initial_state": "candidate_realized_source_table",
            "primitive_functions": [item.function for item in COMPONENT_EXPORTS],
        },
        "raw_export": {
            "revision": 1,
            "tables": ["state_raw_wide.csv", "force_raw_wide.csv", "primitive_raw_wide.csv"],
            "join_key": ["particle_id", "time_s"],
        },
        "validation": {
            "require_all_records_active": True,
            "initial_state_roundoff_multiplier": 4096.0,
            "initial_primitive_roundoff_multiplier": 4096.0,
            "require_component_force_sum": True,
        },
        "claim_policy": campaign["claim_policy"],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_shared_field_input(
    root: Path, campaign: Mapping[str, object]
) -> tuple[Mapping[str, object], Path, DataBundle, str]:
    """Authenticate the single primitive-field authority before materialization."""

    field_record = _mapping(campaign["shared_field_input"], "shared_field_input")
    if set(field_record) != {"relative_path", "sha256", "content_hash"}:
        raise ValueError("shared_field_input keys differ")
    shared_field_path = _relative_path(root, field_record["relative_path"], "shared field input")
    if _sha256(shared_field_path) != field_record["sha256"]:
        raise ValueError("shared field input SHA-256 differs")
    base, base_info = read_with_info(shared_field_path)
    shared_field_content_hash = str(field_record["content_hash"])
    if base_info.content_hash != shared_field_content_hash:
        raise ValueError("shared field logical content hash differs")
    if len(base.sources) != 1 or len(base.fields) != 17:
        raise ValueError("shared input must contain one source and the 17 primitive fields")
    expected_fields = {item.field for item in COMPONENT_EXPORTS}
    if {field.name for field in base.fields} != expected_fields:
        raise ValueError("shared input contains a derived or missing particle field")
    return field_record, shared_field_path, base, shared_field_content_hash


def prepare(config_path: Path, output: Path) -> dict[str, object]:
    """Materialize three no-clobber candidate/common-P1 case directories."""

    campaign, cases = _load_configuration(config_path.resolve())
    root = _repository_root()
    field_record, shared_field_path, base, shared_field_content_hash = _load_shared_field_input(
        root, campaign
    )
    template_path, _ = _locked_file(root, campaign["candidate_template"], "candidate_template")
    template = _mapping(parse_document(template_path.read_bytes()), "template")
    if template.get("format_version") != CASE_FORMAT_VERSION:
        raise ValueError("candidate template must use the current case format")
    source_models = _mapping(campaign["source_models"], "source_models")
    if set(source_models) != {"theory_common"}:
        raise ValueError("source_models must contain only the theory common-field source")
    for name, record in source_models.items():
        _locked_file(root, record, f"source_models.{name}")
    prepared = [
        (
            case,
            *_candidate_bundle(base, case, shared_field_path, shared_field_content_hash),
        )
        for case in cases
    ]

    output.mkdir(parents=True, exist_ok=False)
    candidate_root = output / "candidate_inputs"
    cases_root = output / "cases"
    candidate_root.mkdir()
    cases_root.mkdir()
    reports: dict[str, object] = {}
    for case, bundle, release_receipt in prepared:
        input_path = candidate_root / f"{case.case_id}.h5"
        info = write(input_path, bundle)
        case_directory = cases_root / case.case_id
        prepare_p1(input_path, case_directory)
        spec = _run_spec(case)
        (case_directory / "run_spec.properties").write_text(
            "".join(f"{key}={spec[key]}\n" for key in RUN_SPEC_KEYS), encoding="ascii"
        )
        case_documents: dict[str, object] = {}
        for label, _step_name, dt_s in STEP_ROWS:
            case_path = case_directory / f"candidate_{label}.yaml"
            document = _candidate_document(
                template,
                case,
                input_path,
                case_directory,
                info.content_hash,
                label,
                dt_s,
            )
            case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            load_case(case_path)
            case_documents[label] = {
                "path": case_path.name,
                "sha256": _sha256(case_path),
                "dt_s": dt_s,
            }
        source_model = _mapping(source_models[case.source_model], case.source_model)
        reference_config = _reference_config(
            campaign,
            case,
            source_model,
            input_path,
            _sha256(input_path),
            info.content_hash,
        )
        _write_json(case_directory / "reference_run_config.json", reference_config)
        reports[case.case_id] = {
            "status": "PREPARED",
            "diameter_nm": case.diameter_nm,
            "ion_drag_revision": case.ion_drag_revision,
            "deterministic_contribution_name": case.deterministic_contribution_name,
            "candidate_input": {
                "path": input_path.relative_to(output).as_posix(),
                "sha256": _sha256(input_path),
                "content_hash": info.content_hash,
            },
            "release": release_receipt,
            "candidate_cases": case_documents,
            "common_p1_table_receipt": (case_directory / "common_p1_table_receipt.json")
            .relative_to(output)
            .as_posix(),
            "reference_config": (case_directory / "reference_run_config.json")
            .relative_to(output)
            .as_posix(),
            "run_spec": spec,
        }
    report: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "status": "PREPARED",
        "configuration": str(config_path.resolve()),
        "configuration_sha256": _sha256(config_path.resolve()),
        "shared_field_authority": {
            "path": str(shared_field_path),
            "sha256": field_record["sha256"],
            "content_hash": shared_field_content_hash,
            "field_count": len(base.fields),
            "component_count": len(COMPONENT_EXPORTS),
        },
        "release_authority": "size_specific_t0_table_per_case",
        "background_field_tables": "NOT_READ",
        "effective_gas_maximum_speed_ratio": EFFECTIVE_GAS_MAXIMUM_SPEED_RATIO,
        "cases": reports,
        "claim_policy": campaign["claim_policy"],
    }
    _write_json(output / "campaign_prepare_report.json", report)
    return report


def _number(value: float) -> str:
    return format(float(value), ".17g")


def _write_trajectory(path: Path, result: Any) -> int:
    rows = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(_TRAJECTORY_HEADER)
        for frame in result.iter_frames():
            for index, particle_id in enumerate(frame.particle_id):
                writer.writerow(
                    (
                        int(particle_id),
                        _number(frame.time_s),
                        _number(frame.position_m[index, 0]),
                        _number(frame.position_m[index, 1]),
                        _number(frame.velocity_m_s[index, 0]),
                        _number(frame.velocity_m_s[index, 1]),
                        _number(frame.charge_number[index]),
                        _LIFECYCLE[int(frame.lifecycle[index])],
                    )
                )
                rows += 1
    return rows


def run_candidate(prepared_root: Path, case_id: str, step: str) -> dict[str, object]:
    """Run one prepared cell through the ordinary public solver API."""

    if case_id not in EXPECTED_CASE_IDS or step not in {row[0] for row in STEP_ROWS}:
        raise ValueError("unknown companion case or step")
    case_root = prepared_root.resolve() / "cases" / case_id
    case_path = case_root / f"candidate_{step}.yaml"
    output = case_root / "candidate_results" / step
    output.parent.mkdir(exist_ok=True)
    simulate(load_case(case_path), output)
    result = open_result(output)
    trajectory = output.parent / f"trajectory_{step}.csv"
    rows = _write_trajectory(trajectory, result)
    resolved = _mapping(result.manifest.get("resolved"), "result.manifest.resolved")
    physics_models = _mapping(resolved.get("physics_models"), "resolved.physics_models")
    report: dict[str, object] = {
        "schema_version": 1,
        "tool_revision": TOOL_REVISION,
        "case_id": case_id,
        "step": step,
        "case_sha256": _sha256(case_path),
        "result": str(output),
        "trajectory": str(trajectory),
        "trajectory_sha256": _sha256(trajectory),
        "rows": rows,
        "physics_models": physics_models,
    }
    _write_json(output.parent / f"candidate_run_{step}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("output", type=Path)
    run_parser = subparsers.add_parser("run-candidate")
    run_parser.add_argument("prepared_root", type=Path)
    run_parser.add_argument("case_id", choices=EXPECTED_CASE_IDS)
    run_parser.add_argument("step", choices=[row[0] for row in STEP_ROWS])
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.config, arguments.output)
    else:
        run_candidate(arguments.prepared_root, arguments.case_id, arguments.step)


if __name__ == "__main__":
    main()
