"""Validate and normalize the M3-C1 COMSOL thermophoresis PPR export.

This external V&V tool validates one locked, fine-step COMSOL export. It does
not run either solver and does not promote trajectory or physical-applicability
claims. The only physics claim it can pass is saved-row producer-form closure:
the exported PPR heat flux must replay the exported Waldmann force.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

TOOL_REVISION: Final = "m3c1_thermophoresis_ppr_evaluator_v1"
PARTICLE_RAW_TABLE: Final = "thermophoresis_ppr_particle_raw_wide.csv"
MESH_RAW_TABLE: Final = "thermophoresis_ppr_native_mesh_nodes.csv"
PARTICLE_TIDY_TABLE: Final = "thermophoresis_ppr_particle_tidy.csv"
DOMAIN3_TABLE: Final = "thermophoresis_ppr_domain3_points.csv"
REPORT_FILE: Final = "thermophoresis_ppr_evaluation.json"
WALDMANN_COEFFICIENT: Final = 32.0 / 15.0
MOLAR_GAS_CONSTANT_J_PER_MOL_K: Final = 8.31446261815324
HEAT_FLUX_IDENTITY_LIMIT: Final = 1e-12
PARTICLE_COLUMNS: Final = (
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
MESH_COLUMNS: Final = (
    "r_m",
    "z_m",
    "domain_id",
    "gas_temperature_K",
    "gas_thermal_conductivity_W_per_mK",
    "unrecovered_temperature_gradient_r_K_per_m",
    "unrecovered_temperature_gradient_z_K_per_m",
    "ppr_temperature_gradient_r_K_per_m",
    "ppr_temperature_gradient_z_K_per_m",
    "ppr_heat_flux_r_W_per_m2",
    "ppr_heat_flux_z_W_per_m2",
)
DOMAIN3_COLUMNS: Final = (
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
V6_STATE_COLUMNS: Final = (
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
    "charge_rate_e_per_s",
    "particle_mass_kg",
)
V6_FORCE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "electric_force_r_N",
    "electric_force_z_N",
    "ion_drag_force_r_N",
    "ion_drag_force_z_N",
    "epstein_force_r_N",
    "epstein_force_z_N",
    "thermophoretic_force_r_N",
    "thermophoretic_force_z_N",
    "lift_force_r_N",
    "lift_force_z_N",
    "dep_force_r_N",
    "dep_force_z_N",
    "gravity_buoyancy_force_r_N",
    "gravity_buoyancy_force_z_N",
)

type Record = dict[str, float]
type Key = tuple[int, float]
type ComponentPairs = list[tuple[float, float]]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _integer(value: float, name: str, source: Path) -> int:
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{source}: {name} must be a finite integer, found {value!r}")
    return int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numeric_rows(path: Path) -> Iterator[list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("%"):
                continue
            try:
                yield [float(value) for value in next(csv.reader([line]))]
            except (ValueError, csv.Error) as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric CSV row") from exc


def _wide_records(path: Path, columns: Sequence[str], frames: int) -> list[Record]:
    expected_width = len(columns) * frames
    records: list[Record] = []
    particle_rows = 0
    for row in _numeric_rows(path):
        particle_rows += 1
        if len(row) != expected_width:
            raise ValueError(
                f"{path}: expected {expected_width} values per particle row, found {len(row)}"
            )
        for frame in range(frames):
            start = frame * len(columns)
            values = row[start : start + len(columns)]
            records.append(dict(zip(columns, values, strict=True)))
    if particle_rows == 0:
        raise ValueError(f"{path}: no numeric particle rows")
    return records


def _mesh_records(path: Path) -> list[Record]:
    records: list[Record] = []
    for row in _numeric_rows(path):
        if len(row) != len(MESH_COLUMNS):
            raise ValueError(
                f"{path}: expected {len(MESH_COLUMNS)} values per mesh row, found {len(row)}"
            )
        records.append(dict(zip(MESH_COLUMNS, row, strict=True)))
    if not records:
        raise ValueError(f"{path}: no numeric mesh-node rows")
    return records


def _key(record: Mapping[str, float], source: Path) -> Key:
    particle_id = _integer(record["particle_id"], "particle_id", source)
    time_s = float(record["time_s"])
    if not math.isfinite(time_s):
        raise ValueError(f"{source}: time_s must be finite")
    return particle_id, time_s


def _index(records: Iterable[Record], source: Path) -> dict[Key, Record]:
    result: dict[Key, Record] = {}
    for record in records:
        key = _key(record, source)
        if key in result:
            raise ValueError(f"{source}: duplicate particle/time key {key!r}")
        result[key] = record
    return result


def _all_finite(records: Iterable[Mapping[str, float]]) -> bool:
    return all(math.isfinite(value) for record in records for value in record.values())


def _component_metrics(
    candidate: Sequence[tuple[float, float]],
    reference: Sequence[tuple[float, float]],
    keys: Sequence[Key],
) -> dict[str, Any]:
    if len(candidate) != len(reference) or len(candidate) != len(keys) or not candidate:
        raise ValueError("component comparison inputs must have equal nonzero lengths")
    tiny = math.ulp(0.0)
    normalized: list[float] = []
    residual_squares = 0.0
    reference_squares = 0.0
    maximum_absolute = -1.0
    worst_normalized = -1.0
    worst_index = 0
    worst_component = "r"
    for row_index, (actual, expected) in enumerate(zip(candidate, reference, strict=True)):
        for component_index, component in enumerate(("r", "z")):
            residual = abs(actual[component_index] - expected[component_index])
            scale = max(abs(actual[component_index]), abs(expected[component_index]), tiny)
            normalized.append(residual / scale)
            residual_squares += residual * residual
            reference_squares += expected[component_index] * expected[component_index]
            if normalized[-1] > worst_normalized:
                worst_index = row_index
                worst_component = component
                worst_normalized = normalized[-1]
            maximum_absolute = max(maximum_absolute, residual)
    ordered = sorted(normalized)
    p99_index = min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)
    denominator = max(math.sqrt(reference_squares), tiny)
    worst_component_index = 0 if worst_component == "r" else 1
    worst_flat_index = 2 * worst_index + worst_component_index
    return {
        "global_relative_l2_residual": math.sqrt(residual_squares) / denominator,
        "component_scale_normalized_residual_p99": ordered[p99_index],
        "component_scale_normalized_residual_max": normalized[worst_flat_index],
        "absolute_component_residual_max": maximum_absolute,
        "worst_component": worst_component,
        "worst_row": {
            "particle_id": keys[worst_index][0],
            "time_s": keys[worst_index][1],
        },
    }


def _validate_config(path: Path) -> dict[str, Any]:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    if payload.get("schema_version") != 1 or payload.get("evaluation_revision") != 1:
        raise ValueError("configuration must select M3-C1 thermophoresis PPR revision 1")
    if payload.get("classification") != "external_comsol_reference_correction":
        raise ValueError("configuration classification is not the PPR reference correction")
    case = _mapping(payload.get("case"), "case")
    feature = _mapping(payload.get("thermophoretic_feature"), "thermophoretic_feature")
    gradient = _mapping(feature.get("recovered_temperature_gradient"), "recovered gradient")
    flux = _mapping(feature.get("recovered_heat_flux"), "recovered heat flux")
    expected_gradient = {
        "r": "ppr(d(root.comp1.AS_Tg,r))",
        "z": "ppr(d(root.comp1.AS_Tg,z))",
    }
    expected_flux = {
        "r": f"-k_mix*{expected_gradient['r']}",
        "z": f"-k_mix*{expected_gradient['z']}",
    }
    if gradient != expected_gradient or flux != expected_flux:
        raise ValueError("configuration does not pin the exact expression-level PPR primitives")
    if (
        feature.get("model") != "Waldmann"
        or feature.get("UsePPR") is not True
        or feature.get("temperature_input") != "root.comp1.AS_Tg"
        or feature.get("thermal_conductivity") != "k_mix"
        or feature.get("background_gas_molar_mass") != "Mmix"
    ):
        raise ValueError("configuration does not pin the source thermophoretic feature")
    if case.get("fixed_rk4_step_s") != 1.5625e-7:
        raise ValueError("configuration must select only the accepted finest fixed step")
    return payload


def _require_complete_run_status(output: Path) -> Path:
    run_status_path = output / "run_status.json"
    if run_status_path.exists():
        run_status = _mapping(json.loads(run_status_path.read_text(encoding="utf-8")), "run_status")
        if run_status.get("status") != "COMPLETE":
            raise ValueError(
                f"raw PPR export is not complete: {run_status.get('failure', 'unknown failure')}"
            )
    return run_status_path


def _read_provenance(output: Path) -> tuple[Path, dict[str, Any]]:
    provenance_path = output / "provenance.json"
    provenance = _mapping(json.loads(provenance_path.read_text(encoding="utf-8")), "provenance")
    return provenance_path, provenance


def _require_source_isolation(provenance: Mapping[str, Any], expected_hash: str) -> None:
    if (
        provenance.get("source_sha256_before") != expected_hash
        or provenance.get("source_sha256_after") != expected_hash
        or provenance.get("source_unchanged") is not True
        or provenance.get("source_load_mode") != "ModelUtil.loadCopy"
        or provenance.get("model_saved") is not False
    ):
        raise ValueError("source-MPH isolation provenance is invalid")


def _require_v6_preservation(provenance: Mapping[str, Any], preserved: Mapping[str, Any]) -> None:
    if (
        provenance.get("preserved_m3c0b_v6_state_sha256_before")
        != preserved.get("state_raw_sha256")
        or provenance.get("preserved_m3c0b_v6_state_sha256_after")
        != preserved.get("state_raw_sha256")
        or provenance.get("preserved_m3c0b_v6_force_sha256_before")
        != preserved.get("force_raw_sha256")
        or provenance.get("preserved_m3c0b_v6_force_sha256_after")
        != preserved.get("force_raw_sha256")
        or provenance.get("preserved_m3c0b_v6_unchanged") is not True
    ):
        raise ValueError("locked M3-C0b v6 preservation provenance is invalid")


def _require_single_step_receipt(
    output: Path,
    config_path: Path,
    provenance: Mapping[str, Any],
) -> None:
    if provenance.get("process_count") != 1 or provenance.get("steps_run") != 1:
        raise ValueError("PPR provenance must record one process and one fixed step")
    staged_config = output / str(provenance.get("staged_config"))
    if _sha256(staged_config) != _sha256(config_path):
        raise ValueError("staged and selected PPR configurations differ")


def _verify_export_provenance(
    output: Path, config_path: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    run_status_path = _require_complete_run_status(output)
    provenance_path, provenance = _read_provenance(output)
    source = _mapping(config.get("source_model"), "source_model")
    expected_source_hash = str(source.get("sha256"))
    preserved = _mapping(config.get("preserved_reference"), "preserved_reference")
    _require_source_isolation(provenance, expected_source_hash)
    _require_v6_preservation(provenance, preserved)
    _require_single_step_receipt(output, config_path, provenance)
    return {
        "provenance_path": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "run_status_path": str(run_status_path) if run_status_path.exists() else None,
        "source_model_sha256": expected_source_hash,
        "source_model_unchanged": True,
        "m3c0b_v6_unchanged": True,
        "load_mode": "ModelUtil.loadCopy",
        "model_saved": False,
        "process_count": 1,
        "steps_run": 1,
    }


def _reference_paths(output: Path, config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    preserved = _mapping(config.get("preserved_reference"), "preserved_reference")
    solver_root = output.parents[1]
    root = solver_root / str(preserved.get("root"))
    step = root / str(preserved.get("step_directory"))
    return step / "state_raw_wide.csv", step / "force_raw_wide.csv", step


def _load_particle_cohort(
    output: Path,
    case: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> tuple[Path, list[Record], dict[Key, Record], set[int], list[float], int]:
    frames = int(case.get("output_times", 0))
    expected_particles = int(case.get("particle_count", 0))
    expected_records = int(acceptance.get("expected_particle_records", 0))
    step = output / str(case.get("step_directory"))
    particle_path = step / PARTICLE_RAW_TABLE
    records = _wide_records(particle_path, PARTICLE_COLUMNS, frames)
    index = _index(records, particle_path)
    if len(records) != expected_records or len(index) != expected_records:
        raise ValueError(
            f"{particle_path}: expected {expected_records} unique rows, found {len(index)}"
        )
    particle_ids = {key[0] for key in index}
    times = sorted({key[1] for key in index})
    if particle_ids != set(range(1, expected_particles + 1)) or len(times) != frames:
        raise ValueError("PPR particle/time cohort is not the exact configured Cartesian grid")
    if any(later <= earlier for earlier, later in itertools.pairwise(times)):
        raise ValueError("PPR output times are not strictly increasing")
    time_end_s = _positive(case.get("time_end_s"), "time_end_s")
    if not math.isclose(times[0], 0.0, rel_tol=0.0, abs_tol=2e-14) or not math.isclose(
        times[-1], time_end_s, rel_tol=0.0, abs_tol=2e-14
    ):
        raise ValueError("PPR particle time range does not match the configuration")
    if not _all_finite(records):
        raise ValueError("PPR particle export contains a non-finite value")
    if any(record["current_status_code"] != 1.0 for record in records):
        raise ValueError("PPR particle export contains a non-active saved row")
    return particle_path, records, index, particle_ids, times, frames


def _load_locked_v6_reference(
    output: Path,
    config: Mapping[str, Any],
    frames: int,
    particle_index: Mapping[Key, Record],
) -> tuple[Path, Path, Path, dict[Key, Record], dict[Key, Record]]:
    preserved = _mapping(config.get("preserved_reference"), "preserved_reference")
    state_path, force_path, reference_step = _reference_paths(output, config)
    if _sha256(state_path) != preserved.get("state_raw_sha256"):
        raise ValueError("locked M3-C0b v6 state hash mismatch")
    if _sha256(force_path) != preserved.get("force_raw_sha256"):
        raise ValueError("locked M3-C0b v6 force hash mismatch")
    state = _index(_wide_records(state_path, V6_STATE_COLUMNS, frames), state_path)
    force = _index(_wide_records(force_path, V6_FORCE_COLUMNS, frames), force_path)
    if set(particle_index) != set(state) or set(particle_index) != set(force):
        raise ValueError("PPR and locked v6 particle/time cohorts differ")
    return state_path, force_path, reference_step, state, force


def _coordinate_identity_summary(
    index: Mapping[Key, Record],
    state: Mapping[Key, Record],
    keys: Sequence[Key],
    acceptance: Mapping[str, Any],
    reference_step: Path,
    state_path: Path,
) -> dict[str, Any]:
    coordinate_errors = [
        max(
            abs(index[key]["r_m"] - state[key]["r_m"]),
            abs(index[key]["z_m"] - state[key]["z_m"]),
        )
        for key in keys
    ]
    coordinate_max = max(coordinate_errors)
    coordinate_limit = _positive(
        acceptance.get("maximum_v6_coordinate_absolute_difference_m"),
        "maximum_v6_coordinate_absolute_difference_m",
    )
    return {
        "status": "PASS" if coordinate_max <= coordinate_limit else "FAIL",
        "reference_step": str(reference_step),
        "reference_state_sha256": _sha256(state_path),
        "maximum_absolute_difference_m": coordinate_max,
        "limit_m": coordinate_limit,
    }


def _waldmann_components(
    index: Mapping[Key, Record],
    force: Mapping[Key, Record],
    keys: Sequence[Key],
    expected_molar_mass: float,
) -> tuple[ComponentPairs, ComponentPairs, ComponentPairs, ComponentPairs, ComponentPairs]:
    predicted: ComponentPairs = []
    direct: ComponentPairs = []
    v6_direct: ComponentPairs = []
    flux_identity: ComponentPairs = []
    flux_exported: ComponentPairs = []
    for key in keys:
        record = index[key]
        molar_mass = record["background_gas_molar_mass_kg_per_mol"]
        if not math.isclose(molar_mass, expected_molar_mass, rel_tol=2e-15, abs_tol=0.0):
            raise ValueError(
                "exported background molar mass differs from the pinned producer value"
            )
        temperature = record["gas_temperature_K"]
        if temperature <= 0.0:
            raise ValueError("PPR particle temperature must be positive")
        thermal_speed = math.sqrt(
            8.0 * MOLAR_GAS_CONSTANT_J_PER_MOL_K * temperature / (math.pi * molar_mass)
        )
        radius = 0.5 * record["particle_diameter_m"]
        factor = WALDMANN_COEFFICIENT * radius * radius / thermal_speed
        predicted.append(
            (
                factor * record["ppr_heat_flux_r_W_per_m2"],
                factor * record["ppr_heat_flux_z_W_per_m2"],
            )
        )
        direct.append((record["thermophoretic_force_r_N"], record["thermophoretic_force_z_N"]))
        v6_direct.append(
            (force[key]["thermophoretic_force_r_N"], force[key]["thermophoretic_force_z_N"])
        )
        conductivity = record["gas_thermal_conductivity_W_per_mK"]
        flux_identity.append(
            (
                -conductivity * record["ppr_temperature_gradient_r_K_per_m"],
                -conductivity * record["ppr_temperature_gradient_z_K_per_m"],
            )
        )
        flux_exported.append(
            (record["ppr_heat_flux_r_W_per_m2"], record["ppr_heat_flux_z_W_per_m2"])
        )
    return predicted, direct, v6_direct, flux_identity, flux_exported


def _waldmann_replay_summary(
    index: Mapping[Key, Record],
    force: Mapping[Key, Record],
    keys: Sequence[Key],
    feature: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    replay = _mapping(feature.get("waldmann_replay"), "waldmann_replay")
    if replay.get("mean_thermal_speed") != "sqrt(8*R_const*root.comp1.AS_Tg/(pi*Mmix))":
        raise ValueError("configuration does not pin the expected Waldmann thermal speed")
    expected_molar_mass = _positive(
        feature.get("background_gas_molar_mass_kg_per_mol"), "background molar mass"
    )
    predicted, direct, v6_direct, flux_identity, flux_exported = _waldmann_components(
        index, force, keys, expected_molar_mass
    )
    force_metrics = _component_metrics(predicted, direct, keys)
    force_limit = _positive(
        acceptance.get("maximum_waldmann_force_component_scale_normalized_residual"),
        "maximum Waldmann residual",
    )
    force_status = (
        "PASS"
        if force_metrics["component_scale_normalized_residual_max"] <= force_limit
        else "FAIL"
    )
    flux_metrics = _component_metrics(flux_identity, flux_exported, keys)
    flux_limit = HEAT_FLUX_IDENTITY_LIMIT
    flux_status = (
        "PASS" if flux_metrics["component_scale_normalized_residual_max"] <= flux_limit else "FAIL"
    )
    v6_force_metrics = _component_metrics(direct, v6_direct, keys)
    return {
        "status": "PASS" if force_status == flux_status == "PASS" else "FAIL",
        "primitive_authority": "feature_internal_ppr",
        "formula": "F=(32/15)*(d0/2)^2*q/sqrt(8*R*T/(pi*Mmix))",
        "coefficient": WALDMANN_COEFFICIENT,
        "molar_gas_constant_J_per_mol_K": MOLAR_GAS_CONSTANT_J_PER_MOL_K,
        "background_gas_molar_mass_kg_per_mol": expected_molar_mass,
        "force_component_limit": force_limit,
        "force_replay": {"status": force_status, **force_metrics},
        "heat_flux_identity": {"status": flux_status, "limit": flux_limit, **flux_metrics},
        "rerun_vs_locked_v6_exported_force": v6_force_metrics,
    }


def _validate_particle_export(
    output: Path, config: Mapping[str, Any]
) -> tuple[list[Record], dict[str, Any], dict[str, Any], dict[str, Any]]:
    case = _mapping(config.get("case"), "case")
    acceptance = _mapping(config.get("acceptance"), "acceptance")
    particle_path, records, index, particle_ids, times, frames = _load_particle_cohort(
        output, case, acceptance
    )
    state_path, _, reference_step, state, force = _load_locked_v6_reference(
        output, config, frames, index
    )
    keys = sorted(index)
    coordinate_summary = _coordinate_identity_summary(
        index, state, keys, acceptance, reference_step, state_path
    )
    feature = _mapping(config.get("thermophoretic_feature"), "thermophoretic_feature")
    replay_summary = _waldmann_replay_summary(index, force, keys, feature, acceptance)
    particle_summary = {
        "status": "PASS",
        "raw_path": str(particle_path),
        "raw_sha256": _sha256(particle_path),
        "records": len(records),
        "unique_keys": len(index),
        "particles": len(particle_ids),
        "output_times": len(times),
        "first_time_s": times[0],
        "last_time_s": times[-1],
        "all_finite": True,
        "all_active": True,
    }
    return records, particle_summary, coordinate_summary, replay_summary


def _select_mesh_domain(
    records: Sequence[Record], path: Path, selected_domain: int
) -> tuple[list[float], list[Record], int]:
    if not all(
        math.isfinite(record[name]) for record in records for name in ("r_m", "z_m", "domain_id")
    ):
        raise ValueError("dataset mesh-point coordinates or domain IDs contain a non-finite value")
    domains = [record["domain_id"] for record in records]
    selected_mask = [
        math.isclose(value, float(selected_domain), rel_tol=0.0, abs_tol=1e-12) for value in domains
    ]
    selected_count = sum(selected_mask)
    if selected_count == 0:
        raise ValueError(
            f"native mesh-node export does not cover selected domain {selected_domain}"
        )
    selected_records = [
        record for record, selected in zip(records, selected_mask, strict=True) if selected
    ]
    if not _all_finite(selected_records):
        raise ValueError("selected-domain dataset mesh-point PPR values contain a non-finite value")
    return domains, selected_records, selected_count


def _mesh_flux_identity(selected_records: Sequence[Record]) -> tuple[str, dict[str, Any]]:
    flux_expected = [
        (
            -record["gas_thermal_conductivity_W_per_mK"]
            * record["ppr_temperature_gradient_r_K_per_m"],
            -record["gas_thermal_conductivity_W_per_mK"]
            * record["ppr_temperature_gradient_z_K_per_m"],
        )
        for record in selected_records
    ]
    flux_exported = [
        (record["ppr_heat_flux_r_W_per_m2"], record["ppr_heat_flux_z_W_per_m2"])
        for record in selected_records
    ]
    pseudo_keys = [(index + 1, 0.0) for index in range(len(selected_records))]
    metrics = _component_metrics(flux_expected, flux_exported, pseudo_keys)
    limit = HEAT_FLUX_IDENTITY_LIMIT
    status = "PASS" if metrics["component_scale_normalized_residual_max"] <= limit else "FAIL"
    return status, {"status": status, "limit": limit, **metrics}


def _mesh_coverage_summary(
    records: Sequence[Record],
    domains: Sequence[float],
    selected_records: Sequence[Record],
    selected_count: int,
) -> dict[str, Any]:
    coordinates = [(record["r_m"], record["z_m"]) for record in selected_records]
    unique_coordinates = len(set(coordinates))
    all_row_finite_count = sum(_all_finite((record,)) for record in records)
    integer_domains = [
        round(domain)
        for domain in domains
        if math.isclose(domain, round(domain), rel_tol=0.0, abs_tol=1e-12)
    ]
    domain_counts = {
        str(domain): integer_domains.count(domain) for domain in sorted(set(integer_domains))
    }
    return {
        "rows": len(records),
        "columns": len(MESH_COLUMNS),
        "coordinate_and_domain_columns_all_finite": True,
        "all_columns_finite_rows": all_row_finite_count,
        "all_columns_nonfinite_rows": len(records) - all_row_finite_count,
        "selected_domain_all_columns_finite": True,
        "selected_domain_unique_coordinate_pairs": unique_coordinates,
        "selected_domain_duplicate_coordinate_rows": selected_count - unique_coordinates,
        "domain_counts": domain_counts,
        "fractional_smoothed_domain_rows": len(domains) - len(integer_domains),
        "domain_value_min": min(domains),
        "domain_value_max": max(domains),
    }


def _validate_mesh_export(
    output: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[Record]]:
    case = _mapping(config.get("case"), "case")
    feature = _mapping(config.get("thermophoretic_feature"), "thermophoretic_feature")
    step = output / str(case.get("step_directory"))
    path = step / MESH_RAW_TABLE
    records = _mesh_records(path)
    selected_domain = int(feature.get("selected_domain", 0))
    domains, selected_records, selected_count = _select_mesh_domain(records, path, selected_domain)
    status, flux_identity = _mesh_flux_identity(selected_records)
    coverage = _mesh_coverage_summary(records, domains, selected_records, selected_count)
    return {
        "status": status,
        "raw_path": str(path),
        "raw_sha256": _sha256(path),
        **coverage,
        "selected_domain": selected_domain,
        "selected_domain_rows": selected_count,
        "scientific_label": "background_dataset_mesh_point_samples",
        "dataset": "dset_AS_field",
        "location": "fromdataset",
        "export_recover": "off",
        "operator_recovery": "ppr",
        "native_fe_node_identity": "NOT_TESTED_NO_NODE_IDS",
        "mesh_topology": "NOT_EXPORTED",
        "interpolation_ownership": "NOT_CERTIFIED",
        "heat_flux_identity": flux_identity,
    }, records


def _write_tidy(path: Path, records: Sequence[Record]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PARTICLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({name: format(record[name], ".17g") for name in PARTICLE_COLUMNS})


def _write_domain3(path: Path, records: Sequence[Record], domain: int) -> int:
    selected = sorted(
        (
            record
            for record in records
            if math.isclose(record["domain_id"], float(domain), rel_tol=0.0, abs_tol=1e-12)
        ),
        key=lambda record: (record["r_m"], record["z_m"]),
    )
    coordinates = [(record["r_m"], record["z_m"]) for record in selected]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("selected-domain dataset point samples contain duplicate coordinates")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DOMAIN3_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in selected:
            writer.writerow({name: format(record[name], ".17g") for name in DOMAIN3_COLUMNS})
    return len(selected)


def evaluate(
    output_directory: Path,
    config_path: Path,
    *,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate one completed raw export and optionally write normalized artifacts."""

    output = output_directory.expanduser().resolve()
    selected_config = config_path.expanduser().resolve()
    config = _validate_config(selected_config)
    provenance = _verify_export_provenance(output, selected_config, config)
    records, particle, coordinates, replay = _validate_particle_export(output, config)
    mesh, mesh_records = _validate_mesh_export(output, config)
    overall = (
        "PASS"
        if particle["status"]
        == coordinates["status"]
        == replay["status"]
        == mesh["status"]
        == "PASS"
        else "FAIL"
    )
    report: dict[str, Any] = {
        "tool_revision": TOOL_REVISION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "classification": "external_comsol_saved_row_primitive_closure",
        "configuration": {
            "path": str(selected_config),
            "sha256": _sha256(selected_config),
            "evaluation_id": config.get("evaluation_id"),
            "evaluation_revision": config.get("evaluation_revision"),
        },
        "provenance": provenance,
        "particle_saved_states": particle,
        "coordinate_identity_to_locked_v6": coordinates,
        "waldmann_ppr_replay": replay,
        "background_dataset_mesh_point_samples": mesh,
        "overall_status": overall,
        "claims": {
            "saved_row_ppr_primitive_closure": overall,
            "integrated_trajectory": "NOT_TESTED",
            "continuous_path_applicability": "NOT_TESTED",
            "physical_applicability": "NOT_TESTED",
            "native_mesh_topology_or_interpolant": "NOT_TESTED",
            "comsol_is_golden_truth": False,
        },
    }
    if write_artifacts:
        tidy_path = output / PARTICLE_TIDY_TABLE
        domain3_path = output / DOMAIN3_TABLE
        report_path = output / REPORT_FILE
        if tidy_path.exists() or domain3_path.exists() or report_path.exists():
            raise FileExistsError("PPR normalized artifact already exists; refusing to clobber it")
        _write_tidy(tidy_path, records)
        domain3_rows = _write_domain3(
            domain3_path,
            mesh_records,
            int(_mapping(config.get("thermophoretic_feature"), "feature")["selected_domain"]),
        )
        report["normalized_particle_table"] = {
            "path": str(tidy_path),
            "sha256": _sha256(tidy_path),
            "rows": len(records),
            "columns": len(PARTICLE_COLUMNS),
        }
        report["normalized_domain3_point_table"] = {
            "path": str(domain3_path),
            "sha256": _sha256(domain3_path),
            "rows": domain3_rows,
            "columns": len(DOMAIN3_COLUMNS),
            "scientific_label": "background_dataset_mesh_point_samples",
            "native_mesh_topology_or_interpolant": "NOT_CERTIFIED",
        }
        _write_json(report_path, report)
    return report


def _incomplete_report(output: Path, failure: str) -> tuple[Path, dict[str, Any]]:
    report = {
        "tool_revision": TOOL_REVISION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "classification": "external_comsol_saved_row_primitive_closure",
        "overall_status": "INCOMPLETE",
        "failure": failure,
        "claims": {
            "saved_row_ppr_primitive_closure": "NOT_TESTED",
            "integrated_trajectory": "NOT_TESTED",
            "physical_applicability": "NOT_TESTED",
        },
    }
    return output / REPORT_FILE, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and print the report without writing normalized artifacts",
    )
    args = parser.parse_args()
    output = args.output_directory.expanduser().resolve()
    config = args.config.expanduser().resolve()
    try:
        report = evaluate(output, config, write_artifacts=not args.check_only)
    except Exception as exc:
        if not args.check_only:
            report_path, incomplete = _incomplete_report(output, str(exc))
            if output.is_dir() and not report_path.exists():
                _write_json(report_path, incomplete)
        raise SystemExit(f"M3-C1 thermophoresis PPR evaluation incomplete: {exc}") from exc
    if report["overall_status"] != "PASS":
        raise SystemExit("M3-C1 thermophoresis PPR saved-row closure failed")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
